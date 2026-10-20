# SPDX-License-Identifier: Apache-2.0
"""F3 ROW-FENCE: the expert bank's rows go OFF and their cells go back WITHOUT a device-wide sync.

THE ORDER (NF rc26 20.10.: a return / lend move held the scheduler thread 0.3-1.2 s; ``_rows_off_live`` called
``torch.cuda.synchronize()`` twice -- device-wide, i.e. also behind every HiCache load / write-back, PLE and L3
stream that never touches an expert row -- plus 4-6 ``.cpu()`` table reads per layer x 48). User order 20.10.:
no synchronous waiting, the decode is never held for nothing.

A rows-OFF move is three steps, ordered by events only:

  E0  ``begin``: an event at the TAIL of every ROW STREAM (each stream that may still launch or run a kernel
      reading or writing a bank row -- see :func:`row_streams`); the current stream waits on each (device-side
      ``wait_event``, the host does not wait). Every kernel any row stream had enqueued before the decision is
      therefore ordered before what follows on the current stream.
  W   the OFF table writes (and the coldest-first byte moves) on the current stream -- device ops, no host read.
  E1  ``close``: one event on the current stream after W; every row stream waits on it (device-side). Every
      kernel enqueued on a row stream AFTER the decision therefore starts behind the OFF tables and never
      routes to an OFF row.

  UNMAP  the cells of the OFF rows are released only once E1 has COMPLETED (``done()`` = ``event.query()``, or
      ``wait()`` = the host waits on E1 alone, never device-wide). E1 completes only after E0 (stream order on
      the current stream), and E0 only after every kernel enqueued before the decision on every row stream:
      **an unmap never runs before the last kernel that could read the row**. Kernels enqueued after the
      decision read the OFF tables (E1 before them) and never touch the released cells.

W-SEAT-WIPE stays guarded where it was (the plans are cut at the lattice and ``refuse_wipes`` runs at the
decision, before anything moves); the deferred unmap applies exactly the plan checked there. The 58-GB-coredump
class (a kernel reading an unmapped row) is excluded by the UNMAP rule above -- as long as no stream outside
:func:`row_streams` touches a bank row. Every stream that does (forward, pool prefetch, the per-layer fetch
stream, H31 deferred rows, resume warm, rearm prefetch, the lend warm stream) registers itself here.

Without CUDA (CPU desk ranks) every event is complete at once: the order degenerates to program order.
Rank-local, no collective; the verdicts that call it are replicated by their callers.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# NF-TOD 20.10. 20:08:05Z (rc28, P PP1 pid 909 SIGSEGV in PyObject_RichCompare under
# ``WeakSet.add`` <- ``register_row_stream`` <- ``DeferredRowsFill.start``): the registry is STRONG and keyed
# by the CUDA stream's identity, never a ``weakref``. torch 2.11's ``torch.cuda.Stream`` inherits its weakref
# slot from ``torch._C.Stream`` (``__weakrefoffset__`` 48 on Stream, _CudaStreamBase AND torch.cuda.Stream),
# so CPython's ``subtype_dealloc`` leaves the clearing to the C base -- and ``THCPStream_dealloc``
# (torch/csrc/cuda/Stream.cpp) only destroys the CUDAStream and calls ``tp_free``, it never calls
# ``PyObject_ClearWeakRefs``. A weakref to a dead ``torch.cuda.Stream`` therefore keeps pointing at freed
# memory: no callback fires, the WeakSet entry stays, and the next stream from the 32-stream pool with the
# same handle (same hash) makes ``set.add`` compare against the freed object. Same holds for
# ``torch.cuda.Event`` (``THCPEvent_dealloc``). Holding the streams strongly is also the fence's own rule:
# a pool stream whose Python handle died can still run a kernel that reads a row, and the pool hands the
# same stream out again -- it stays a row stream. The registry is bounded by the distinct streams of the
# process (pool size x priorities + the few dedicated ones), not by the number of handles created.
_ROW_STREAMS: Dict[Tuple, object] = {}


def _stream_key(stream) -> Tuple:
    """One key per CUDA stream: ``(device_type, device_index, stream_id)`` of a torch stream (every handle
    of one stream -- ``current_stream()`` returns a fresh object per call -- gives the same key); any other
    object (CPU desk fakes) by identity, which the strong reference keeps valid."""
    sid = getattr(stream, "stream_id", None)
    dix = getattr(stream, "device_index", None)
    dty = getattr(stream, "device_type", None)
    if isinstance(sid, int) and isinstance(dix, int) and isinstance(dty, int):
        return ("stream", dty, dix, sid)
    return ("id", id(stream))


def register_row_stream(stream) -> None:
    """A stream that may read or write expert bank rows (idempotent; None ignored). Held strongly -- see the
    NF-TOD 20.10. note above: a weakref to a torch stream dangles once the stream object dies."""
    if stream is None:
        return
    _ROW_STREAMS.setdefault(_stream_key(stream), stream)


def _same(a, b) -> bool:
    """Two handles of one stream (``torch.cuda.current_stream()`` returns a fresh object each call; torch
    streams compare by stream id)."""
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001
        return False


def row_streams(extra: Iterable[object] = ()) -> List[object]:
    """Every registered row stream plus ``extra`` (the caller's forward stream), duplicates removed."""
    out: List[object] = []
    for s in list(_ROW_STREAMS.values()) + [x for x in extra if x is not None]:
        if not any(_same(s, o) for o in out):
            out.append(s)
    return out


def reset_for_tests() -> None:
    _ROW_STREAMS.clear()
    global _OPS, _WARM_STREAM
    _OPS = None
    _WARM_STREAM = None
    _WARM_JOBS.clear()


class CudaOps:
    """The four stream handles of the fence against CUDA."""

    def current_stream(self):
        import torch

        return torch.cuda.current_stream()

    def event(self):
        import torch

        return torch.cuda.Event()

    def record(self, ev, stream) -> None:
        ev.record(stream)

    def wait(self, stream, ev) -> None:
        stream.wait_event(ev)

    def query(self, ev) -> bool:
        return bool(ev.query())

    def synchronize(self, ev) -> None:
        ev.synchronize()

    def new_stream(self):
        import torch

        return torch.cuda.Stream()

    def stream_ctx(self, stream):
        import torch

        return torch.cuda.stream(stream)


class NullOps:
    """No CUDA (CPU desk ranks): device work is synchronous, every event is complete."""

    def current_stream(self):
        return None

    def event(self):
        return object()

    def record(self, ev, stream) -> None:
        pass

    def wait(self, stream, ev) -> None:
        pass

    def query(self, ev) -> bool:
        return True

    def synchronize(self, ev) -> None:
        pass


_OPS = None


def ops():
    """CUDA's handles once a context exists (cached); before that -- or without CUDA -- the null ops, NOT
    cached while CUDA could still come up (a fence taken before the first kernel has nothing to order)."""
    global _OPS
    if _OPS is not None:
        return _OPS
    try:
        import torch

        if torch.cuda.is_available():
            if torch.cuda.is_initialized():
                _OPS = CudaOps()
                return _OPS
            return NullOps()
    except Exception:  # noqa: BLE001
        pass
    _OPS = NullOps()
    return _OPS


def set_ops_for_tests(o) -> None:
    global _OPS
    _OPS = o


class RowFence:
    """One rows-OFF move's E0 / E1 (module docstring). ``begin`` before the OFF writes, ``close`` after them."""

    __slots__ = ("ops", "cur", "streams", "e1", "waited_ms")

    def __init__(self, o, cur, streams):
        self.ops, self.cur, self.streams, self.e1, self.waited_ms = o, cur, streams, None, 0.0

    @classmethod
    def begin(cls, extra_streams: Iterable[object] = ()) -> "RowFence":
        o = ops()
        cur = o.current_stream()
        streams = [s for s in row_streams(extra_streams) if not _same(s, cur)]
        for s in streams:  # E0: behind every kernel a row stream had enqueued
            ev = o.event()
            o.record(ev, s)
            o.wait(cur, ev)
        return cls(o, cur, streams)

    def close(self) -> "RowFence":
        """E1 after the OFF writes; every row stream's later kernels wait for it."""
        self.e1 = self.ops.event()
        self.ops.record(self.e1, self.cur)
        for s in self.streams:
            self.ops.wait(s, self.e1)
        return self

    def done(self) -> bool:
        return self.e1 is not None and bool(self.ops.query(self.e1))

    def wait(self) -> float:
        """The host waits on E1 ALONE (the rows' last readers + the OFF writes) -- never device-wide. Returns ms."""
        import time

        if self.e1 is None:
            raise RuntimeError("F3 ROW-FENCE: wait before close -- the unmap would not be ordered")
        if self.done():
            return 0.0
        t0 = time.perf_counter()
        self.ops.synchronize(self.e1)
        ms = (time.perf_counter() - t0) * 1000.0
        self.waited_ms += ms
        return ms


class PendingRelease:
    """A deferred unmap: the bank plans to apply once ``fence`` is done (FIFO per controller)."""

    __slots__ = ("fence", "plans", "log")

    def __init__(self, fence: RowFence, plans, log: Optional[dict] = None):
        self.fence, self.plans, self.log = fence, plans, dict(log or {})


# ---- F3: device -> host reads without a wait --------------------------------------------------------------

_READ_STREAM = None


def _read_stream():
    global _READ_STREAM
    if _READ_STREAM is None:
        import torch

        _READ_STREAM = torch.cuda.Stream()
    return _READ_STREAM


class DeferredRead:
    """Device tensors copied to pinned host memory on a side stream behind an event of the current stream;
    ``ready()`` is an event query, ``get()`` the host tensors once ready (None before). Never a wait. On a CPU
    desk rank (NullOps / CPU tensors) the copies are immediate."""

    __slots__ = ("host", "ev", "src")

    def __init__(self, tensors):
        o = ops()
        tensors = list(tensors)
        if isinstance(o, NullOps) or not all(getattr(t, "is_cuda", False) for t in tensors):
            self.host, self.ev, self.src = [t.detach().clone() for t in tensors], None, None
            return
        import torch

        s = _read_stream()
        ev0 = torch.cuda.Event()
        ev0.record(torch.cuda.current_stream())
        s.wait_event(ev0)
        self.host = [torch.empty(tuple(t.shape), dtype=t.dtype, pin_memory=True) for t in tensors]
        with torch.cuda.stream(s):
            for h, t in zip(self.host, tensors):
                h.copy_(t, non_blocking=True)
                t.record_stream(s)
            self.ev = torch.cuda.Event()
            self.ev.record(s)
        self.src = tensors

    def ready(self) -> bool:
        return self.ev is None or bool(self.ev.query())

    def get(self):
        if not self.ready():
            return None
        self.src = None
        return self.host


# ---- F3: the warm refill's copies on their own stream, committed behind their event ------------------------

_WARM_STREAM = None
_WARM_JOBS: List[object] = []


def warm_stream():
    """The lend / seat-grow warm copies' stream -- a ROW STREAM (it writes bank rows)."""
    global _WARM_STREAM
    if _WARM_STREAM is None:
        _WARM_STREAM = ops().new_stream()
        register_row_stream(_WARM_STREAM)
    return _WARM_STREAM


class WarmJob:
    __slots__ = ("cache", "experts", "rows", "valid", "ev", "n", "keep")

    def __init__(self, cache, experts, rows, valid, ev, n, keep=()):
        self.cache, self.experts, self.rows, self.valid, self.ev, self.n, self.keep = (
            cache, experts, rows, valid, ev, int(n), keep)


def submit_warm(cache, experts, rows, valid, src_rows) -> Optional[WarmJob]:
    """Reserved ``rows`` (OFF-like: never routed, never a victim) get their experts' bytes from the store
    rows ``src_rows`` on the WARM STREAM, behind the current stream's reservation; the commit
    (:func:`poll_warm`) runs only after the copies' event completed -- the rows are routed only then."""
    import torch

    from flliper.srt.layers.moe.expert_pool_device import commit_warm_rows_masked, copy_rows

    n = int(rows.numel())
    if n <= 0:
        return None
    count = torch.full((1,), n, dtype=torch.int32, device=rows.device)
    src32, dst32 = src_rows.to(torch.int32), rows.to(torch.int32)
    o = ops()
    if isinstance(o, NullOps):
        copy_rows(cache._pool_srcs, cache._pool_dsts, src32, dst32, count)
        commit_warm_rows_masked(cache._pool_tables, experts, rows, valid)
        cache._pdflip_seat_warmed = int(getattr(cache, "_pdflip_seat_warmed", 0)) + n
        return None
    s = warm_stream()
    ev0 = o.event()
    o.record(ev0, o.current_stream())  # behind the reservation (and the ON writes) on the current stream
    o.wait(s, ev0)
    with o.stream_ctx(s):
        copy_rows(cache._pool_srcs, cache._pool_dsts, src32, dst32, count)
    ev = o.event()
    o.record(ev, s)
    if getattr(src32, "is_cuda", False):
        for t in (src32, dst32, count):
            t.record_stream(s)
    job = WarmJob(cache, experts, rows, valid, ev, n, keep=(src32, dst32, count))
    _WARM_JOBS.append(job)
    return job


def poll_warm() -> int:
    """Commit every warm job whose copies completed (event query, never a wait). Returns the jobs committed."""
    from flliper.srt.layers.moe.expert_pool_device import commit_warm_rows_masked

    n = 0
    o = ops()
    for job in list(_WARM_JOBS):
        if not bool(o.query(job.ev)):
            continue
        _WARM_JOBS.remove(job)
        commit_warm_rows_masked(job.cache._pool_tables, job.experts, job.rows, job.valid)
        job.cache._pdflip_seat_warmed = int(getattr(job.cache, "_pdflip_seat_warmed", 0)) + job.n
        n += 1
    return n


def drop_warm(cache) -> int:
    """A rows-OFF move of ``cache`` (inside its ROW-FENCE, i.e. behind E0, which waited on the warm stream: the
    copies landed): its pending warm rows are released free; the OFF writes that follow overwrite the ones
    above the new k. Returns the jobs dropped."""
    from flliper.srt.layers.moe.expert_pool_device import release_warm_rows

    n = 0
    for job in [j for j in _WARM_JOBS if j.cache is cache]:
        _WARM_JOBS.remove(job)
        release_warm_rows(cache._pool_tables, job.rows)
        n += 1
    return n


def quiesce_warm() -> int:
    """The sleep: every warm copy lands (the host waits on the warm events only), then the jobs are dropped --
    the wake's rearm rewrites the tables."""
    n = len(_WARM_JOBS)
    o = ops()
    for job in list(_WARM_JOBS):
        o.synchronize(job.ev)
    _WARM_JOBS.clear()
    return n
