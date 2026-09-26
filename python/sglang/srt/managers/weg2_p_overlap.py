"""Group P host/GPU overlap for chunked prefill, and the instrument that measures it.

WHAT THE METAL SHOWED (boot weg2xsn420, 4 x ~98k prompts, PP3 42/11/11, depth 0)
-------------------------------------------------------------------------------
The pipeline IS full with chunks of ONE request (slots 0/1/2 carry consecutive
chunks of the same rid, PP0 two chunks ahead of PP2).  What is lost is HOST work
between two forwards of the SAME stage: every P rank's scheduler thread blocks
on the forward it launched last and only then plans, receives and launches the
next one, so the card idles for that host work once per chunk (PP2, the binding
stage at 100k: pass 873 ms against 733 ms gpu-ms, idle ~95-140 ms per chunk,
flat over the prefix depth).  Two host syncs make every stage serial:

* the LAST rank at ``--pp-async-batch-depth 0``: ``next_first_rank_mb_id ==
  mb_id``, so the output send fences the schedule stream on the forward launched
  in THIS pass (``wait_event(q_event)``), the output receive is enqueued behind
  that fence, and ``d2h_event.synchronize()`` then holds the host until that
  forward is done -- for every chunk, although a middle prefill chunk samples
  nothing and nobody consumes its "output";
* the chunked-prefill HiCache publish (``_weg2_publish_at_chunk`` inside
  ``cache_unfinished_req``, i.e. inside the NEXT chunk's plan, before its
  launch): its write path reads device indices on the host
  (``_refuse_unaddressable_kv_rows``: ``int(rows.max())``; with
  ``--hicache-io-backend direct`` + ``layer_first`` also ``move_indices``'
  ``device_indices.cpu()``), which waits for the schedule stream -- fenced on
  the forward that just ran.  The plan of chunk n+1 therefore cannot finish
  before chunk n has, on every rank (PP1: schedule_ms ~ gpu-ms).

WHAT THE OVERLAP MODE CHANGES (``SGLANG_WEG2_P_HOST_OVERLAP=1``, group P only,
set by the launcher's ``--p-host-overlap``; unset = today's code path)
-------------------------------------------------------------------------------
1. The launcher also sets upstream's ``SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=1``:
   a middle prefill chunk (one request, not its last chunk, no logprob) exchanges
   no output at all, on both sides of every hop (``_pp_output_exchange_due`` is
   the one predicate both ends ask), so the last rank no longer fences its host on
   its own current forward.  The last chunk of a request still exchanges.
2. The LAST rank then places the device-side fence itself, at the same point the
   other ranks fence before their proxy send: ``current_stream().wait_event(
   launch_event)``.  It is what orders the NEXT plan's schedule-stream work (the
   mamba anchor copy, the HiCache write's start event) behind the forward whose
   KV rows / state they read -- the ordering the removed ``q_event`` fence used
   to give as a side effect.  Device-side only: the host never waits on it.
3. The chunked-prefill HiCache publish is DEFERRED from inside the plan to right
   after the next forward's launch (``UnifiedRadixCache.
   weg2_flush_deferred_chunk_publish``), so its host sync waits for chunk n while
   chunk n+1 is already queued on the card.  Only the publish moves; the anchor
   copy and the insert stay where they are, and the publish still comes after the
   fence of the forward that produced the rows it copies (that fence was placed
   at the end of the previous pass).

4. A LEAD BOUND for the non-last ranks (``bound_proxy_lead``): with the output
   ring skipped nothing else keeps a faster stage from running ahead of its
   downstream (this fork joins its sends a lap late, #1015e), and every chunk
   ahead keeps an 80 MB proxy frame alive. Before each launch the schedule
   stream waits -- device-side, ``Work.wait()`` on the NCCL half of the send --
   for every frame older than the newest ``SGLANG_WEG2_P_OVERLAP_LEAD`` (1).

What this does NOT change: the write path itself (still the write stream, still
the arena), the retain/finish publish, the flush-time sweep, the order in which
chains are published (parents first), and nothing at all when the variable is
unset -- every branch below is behind :func:`p_host_overlap_on`.

THE INSTRUMENT (``SGLANG_WEG2_P_HOSTGAP=1``, launcher ``--p-hostgap``)
----------------------------------------------------------------------
``bubble_ms`` / ``PP-BUBBLE`` measure HOST time between two launch calls and
include the GPU time of the forward the host was waiting for; they are not card
idle.  This instrument measures the card directly: a timing event at the start
and at the end of every forward on the forward stream, and the device-time gap
``prev_end -> this_start`` is the card's idle before this forward.  Beside it,
the host phases the scheduler spent since the previous launch (plan, of which
anchor/publish/evict-drain/rowcheck, proxy receive, output exchange, d2h wait,
result processing, deferred publish).  One ``#PGAP`` line per forward, harvested
without a host sync (``Event.query()``), so the instrument cannot create the
stall it measures.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

P_HOST_OVERLAP_ENV = "SGLANG_WEG2_P_HOST_OVERLAP"
P_HOSTGAP_ENV = "SGLANG_WEG2_P_HOSTGAP"
SKIP_PURE_CHUNK_OUTPUT_ENV = "SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM"
#: P-NOSYNC: the per-chunk cache path never makes the host wait on a stream --
#: see `p_nosync_on` for the three sites and the measurement.
P_NOSYNC_ENV = "SGLANG_WEG2_P_NOSYNC"
#: FLA l2norm with the row count as a RUN-TIME bound (fla/l2norm.py
#: ``L2NORM_RUNTIME_T_ENV``, same name): no new Triton kernel -- compile/load on
#: the scheduler thread inside the launch -- per token count (weg2xsn423 P: 63
#: cold loads, 3.20 s). Its own switch; --p-host-overlap sets it for group P.
L2NORM_RUNTIME_T_ENV = "SGLANG_FLA_L2NORM_RUNTIME_T"


def p_host_overlap_on() -> bool:
    """The overlap mode.  Read per call (an env lookup), so a test can flip it."""
    return os.environ.get(P_HOST_OVERLAP_ENV, "") == "1"


def hostgap_on() -> bool:
    return os.environ.get(P_HOSTGAP_ENV, "") == "1"


def p_nosync_on() -> bool:
    """P-NOSYNC: no host wait on a CUDA stream in the per-chunk cache path.

    MEASURED (weg2xsn422, --p-host-overlap + #PGAP): the host still sat 409-604
    ms per 4096 chunk in the ANCHOR (plan_parts anchor), and py-spy on PP0 put
    1731 of 2357 samples on ONE line: ``MambaSlotAllocator._do_alloc``'s
    ``self.slot_used[select_index] = True``. Assigning a Python scalar into a
    CUDA tensor copies the scalar host->device with ``non_blocking=False``,
    which PyTorch completes with ``cudaStreamSynchronize`` on the CURRENT
    stream -- the schedule stream, which carries the fence of the forward that
    is still running. So the "anchor" waited for the whole forward; the D2D
    anchor copy itself never blocked. Every such host read/write in the chunk
    path is the same trap:

    * ``MambaSlotAllocator``: ``index_fill_`` (device scalar) instead of the
      scalar assignment in alloc/free, and the #924 double-free answer read
      from PINNED host memory once its event is complete;
    * ``HiCacheController._refuse_unaddressable_kv_rows`` (#923): the bounds
      check is computed on the device and read at the next check/ack, not by
      ``int(rows.max())`` on the host;
    * ``HiCacheController.move_indices`` (direct + layer_first): the device
      indices stay on the device (permuted there) instead of ``.cpu()``.

    SECOND SITE (weg2xsn423, anchor 0 with the above): the PLAN still sat
    417-495 ms per 4096 chunk, and py-spy PP0 put 1538 of 2263 samples on
    ``HybridReqToTokenPool.alloc``'s ``mapping[select_index] = t`` -- a Python
    row list as the index, moved to the device BLOCKING by ``index_put_``. So:

    * ``HybridReqToTokenPool.alloc``: the mapping rows go over pinned memory,
      non-blocking (``_nosync_mapping_rows``);
    * ``ScheduleBatch._collect_deferred_mamba_cow_and_clear``: the #924D
      ``first_state`` note, whose ``extra=`` text ``.tolist()``s a CUDA tensor
      before ``note_924d`` can decline, is only built with the trail on.

    Set for group P by the launcher's ``--p-host-overlap``; unset = the stock
    code paths, byte-identical."""
    return os.environ.get(P_NOSYNC_ENV, "") == "1"


def launcher_env_p_host_overlap() -> Dict[str, str]:
    """The group-P environment ``--p-host-overlap`` adds. ONE place, read by the
    launcher and pinned by the tests, so the two halves cannot drift."""
    return {P_HOST_OVERLAP_ENV: "1", SKIP_PURE_CHUNK_OUTPUT_ENV: "1", P_NOSYNC_ENV: "1",
            L2NORM_RUNTIME_T_ENV: "1"}


def launcher_env_p_hostgap() -> Dict[str, str]:
    return {P_HOSTGAP_ENV: "1"}


# ------------------------------------------------- proxy receive off the fence
#: P-RECV-STREAM (launcher ``--p-recv-stream``, group P only; unset = the stock
#: receive, byte-identical).
#:
#: WHAT THE #PGAP LINES SHOW (docker-acceptance 27b, 26.09.: i8B/i8drt/n4B, PP3,
#: overlap on): the card idle of PP1/PP2 between two chunks does NOT follow the
#: host work after the wait -- on i8drt PP2 it stays 6.5/6.8 ms while ``plan``
#: alternates 2/15-21 ms, and over all 2048-chunks corr(gap, plan+process+recv)
#: is 0.1-0.2 -- it follows the FRAME SIZE (PP2 i8B: 512 -> 1.7 ms, 1024 -> 4.2,
#: 2048 -> 11.1 ms) and it is 0.1 ms on PP0, the one stage that receives no frame.
#: The host is already one forward ahead: it launches chunk k while chunk k-1 is
#: still on the card, and ``d2h_wait`` holds it until chunk k-1 (not k) is done.
#:
#: The idle is the proxy RECEIVE of chunk k, serialised behind forward k-1:
#: ``recv_tensor_dict``'s ``irecv`` is issued on the schedule stream, the NCCL
#: stream first waits for everything queued there, and the schedule stream
#: carries the device fence on forward k-1 (``wait_event(launch_event)`` before
#: the proxy send on PP0/PP1, the overlap fence on the last rank). So the frame
#: can only start to move when forward k-1 has ended, and forward k waits for it.
#:
#: The switch issues that receive on a side stream with no fence on it, so the
#: transfer overlaps forward k-1; the schedule stream then waits for the side
#: stream (device-side), so forward k -- which waits for the schedule stream at
#: launch -- still starts only after its frame has landed. Nothing moves on the
#: host, the fence stays where it is for every other schedule-stream op, and the
#: order of the NCCL operations on each pair is the host order, as before.
P_RECV_STREAM_ENV = "SGLANG_WEG2_P_RECV_STREAM"

#: The scheduler streams that may read a received frame. A tensor allocated on
#: the side stream belongs to that stream's pool, so every other stream that
#: reads it must be recorded, or the allocator may hand the block to the next
#: receive while a forward still reads it.
_FRAME_CONSUMER_STREAMS = ("forward_stream", "copy_stream", "spill_stream")


def p_recv_stream_on() -> bool:
    """P-RECV-STREAM: proxy receive on a side stream. Read per call."""
    return os.environ.get(P_RECV_STREAM_ENV, "") == "1"


def launcher_env_p_recv_stream() -> Dict[str, str]:
    return {P_RECV_STREAM_ENV: "1"}


def _cuda_tensors(message):
    import torch

    if not isinstance(message, dict):
        return []
    return [
        v for v in message.values()
        if isinstance(v, torch.Tensor) and getattr(v, "is_cuda", False)
    ]


def recv_off_fence(holder, recv, device_module, cuda_tensors=_cuda_tensors):
    """Run ``recv(on_wire)`` with the rank's proxy-receive stream current.

    ``recv`` is the blocking host call that posts the NCCL receives (and
    allocates their buffers) on the CURRENT stream; ``on_wire`` must be called
    by it with every message that comes off the wire, the stashed ones
    included. Afterwards:

    * the stream that was current before (the schedule stream) waits for the
      side stream -- device-side, the host never waits here;
    * every CUDA tensor of every wire message is recorded on that stream and on
      the scheduler's forward/copy/spill streams, so the side pool cannot reuse
      its block before those reads are done.

    The side stream waits for nothing else, which is the point: the transfer
    of chunk k is not queued behind the fence on forward k-1.
    """
    side = getattr(holder, "_weg2_recv_stream", None)
    if side is None:
        side = holder._weg2_recv_stream = device_module.Stream()
    consumer = device_module.current_stream()
    wire = []
    with device_module.stream(side):
        out = recv(wire.append)
    consumer.wait_stream(side)
    streams = [consumer]
    for name in _FRAME_CONSUMER_STREAMS:
        s = getattr(holder, name, None)
        if s is not None and all(s is not x for x in streams):
            streams.append(s)
    for message in wire:
        for t in cuda_tensors(message):
            for s in streams:
                t.record_stream(s)
    holder._weg2_recv_stream_n = getattr(holder, "_weg2_recv_stream_n", 0) + 1
    return out


# ---------------------------------------------------------------- lead bound
P_OVERLAP_LEAD_ENV = "SGLANG_WEG2_P_OVERLAP_LEAD"


def overlap_lead() -> int:
    """How many proxy frames a rank may have sent and not yet seen RECEIVED
    when it launches its next forward (default 1)."""
    try:
        return max(0, int(os.environ.get(P_OVERLAP_LEAD_ENV, "1")))
    except ValueError:
        return 1


def note_proxy_send(holder, works) -> None:
    """Remember the DEVICE halves of the proxy send just posted (the NCCL works
    of its CUDA tensors; the metadata goes over gloo and is not a frame)."""
    dev = []
    for w in works or ():
        payload = getattr(w, "payload", None)
        if getattr(w, "work", None) is not None and getattr(payload, "is_cuda", False):
            dev.append(w)
    q = getattr(holder, "_weg2_proxy_lead", None)
    if q is None:
        q = holder._weg2_proxy_lead = deque()
    q.append(dev)


def bound_proxy_lead(holder, lead: Optional[int] = None) -> int:
    """Before a launch: the current stream waits (DEVICE-side, ``Work.wait()``
    on an NCCL work never blocks the host) for every frame older than the newest
    ``lead`` ones, so the forward that follows cannot start before the downstream
    has taken that frame. Why it exists: the fork joins its sends a full lap late
    (#1015e, ``_pp_post_send``), so with the output ring skipped nothing else
    bounds how far a faster stage runs ahead -- measured gap PP0 656 vs PP2 733
    gpu-ms at 100k, i.e. ~3 frames x 80 MB (hidden + residual, 4096 x 5120 x 2 B
    each) by the end of one 98k prompt, growing without bound across a burst.
    Returns how many frames it fenced on."""
    q = getattr(holder, "_weg2_proxy_lead", None)
    if not q:
        return 0
    lead = overlap_lead() if lead is None else int(lead)
    n = 0
    while len(q) > lead:
        for w in q.popleft():
            w.work.wait()
        n += 1
    return n


# ---------------------------------------------------------------- host spans
_SPANS: Dict[str, float] = defaultdict(float)


@contextmanager
def span(name: str):
    """Accumulate the wall time of the block under ``name`` (ms) when the
    instrument is on; a bare ``yield`` otherwise."""
    if not hostgap_on():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _SPANS[name] += (time.perf_counter() - t0) * 1000.0


def take_spans() -> Dict[str, float]:
    out = dict(_SPANS)
    _SPANS.clear()
    return out


# ---------------------------------------------------------------- card gaps
class GapMeter:
    """Per-rank card-idle meter around ``_pp_launch_batch``.

    ``begin()`` (on the forward stream, after its waits, before ``run_batch``)
    records the start event; ``end()`` records the end event and queues the
    record.  ``harvest()`` pops every record whose end event has completed --
    ``query()`` only, never ``synchronize()`` -- and returns the log lines.
    """

    def __init__(self, rank, event_factory=None, cap: int = 64):
        self.rank = rank
        self._event_factory = event_factory
        self._prev_end = None
        self._start = None
        self._pending: Deque[Tuple] = deque()
        self._cap = int(cap)
        self.n = 0

    def _event(self):
        if self._event_factory is not None:
            return self._event_factory()
        import torch

        return torch.cuda.Event(enable_timing=True)

    def begin(self) -> None:
        self._start = self._event()
        self._start.record()

    def end(self, fwd_ct: int, tokens: Optional[int], host: Dict[str, float]) -> None:
        if self._start is None:
            return
        e = self._event()
        e.record()
        self._pending.append((self._prev_end, self._start, e, fwd_ct, tokens, host))
        self._prev_end = e
        self._start = None
        while len(self._pending) > self._cap:  # never unbounded, even if a card hangs
            self._pending.popleft()

    def harvest(self):
        lines = []
        while self._pending and self._pending[0][2].query():
            prev_end, start, end, fwd_ct, tokens, host = self._pending.popleft()
            gap = None if prev_end is None else float(prev_end.elapsed_time(start))
            fwd = float(start.elapsed_time(end))
            self.n += 1
            lines.append(format_line(self.rank, fwd_ct, tokens, gap, fwd, host))
        return lines


_PLAN_PARTS = ("anchor", "publish", "rowcheck", "evict_drain")
#: ``fi_plan`` is PART OF ``launch``: the wall time inside flashinfer's prefill
#: ``plan()``, whose ``qo_indptr.to("cpu")`` waits for everything queued on the
#: forward stream. Once the plan no longer waits for the running forward
#: (P-NOSYNC, memory_pool ``_nosync_mapping_rows``), that wait lands HERE, so
#: ``launch - fi_plan`` is the launch's own host work.
_ORDER = ("plan", "proxy_recv", "launch", "fi_plan", "deferred_publish",
          "output_commit", "d2h_wait", "process")


def format_line(rank, fwd_ct, tokens, gap_ms, fwd_ms, host: Dict[str, float]) -> str:
    """ONE line per forward. ``gpu_gap_ms`` is device time (events), every host
    term is wall time on the scheduler thread since the previous launch."""
    parts = " ".join("%s=%.0f" % (k, host.get(k, 0.0)) for k in _ORDER)
    inner = " ".join("%s=%.0f" % (k, host.get(k, 0.0)) for k in _PLAN_PARTS)
    return (
        "#PGAP pp_rank=%s fwd=%s tokens=%s gpu_gap_ms=%s gpu_fwd_ms=%.1f host[%s] "
        "plan_parts[%s] overlap=%d"
        % (
            rank,
            fwd_ct,
            "-" if tokens is None else tokens,
            "-" if gap_ms is None else "%.1f" % gap_ms,
            fwd_ms,
            parts,
            inner,
            1 if p_host_overlap_on() else 0,
        )
    )
