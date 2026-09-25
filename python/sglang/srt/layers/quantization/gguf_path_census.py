"""GGUF dispatch path census -- the G3 decision instrument (27B line, 2026-09-25).

WHAT IT ANSWERS.  ``fused_mul_mat_gguf`` sends every GGUF linear through one of
four kernels, chosen per call from (M, N, ggml type, card): MMVQ (mat-vec, cost
linear in M), MMQ (quantized GEMM, flat up to ~M=16, K-quants and the standard
types only, M <= ``SGLANG_GGUF_MMQ_MAX_TOKENS``), DEQ (dequantize the whole
weight, then one bf16 cuBLAS GEMM -- every i-quant with N > 5120 at M > 8 and
every K-quant at M > 8) and DENSE (an unquantized shard, plain GEMM).  Whether
G3 (an i-quant MMQ) is worth building is the question "how much of a bs2 verify
round (M = 16) goes to DEQ, per type, per card" -- this instrument measures it.

HOW, WITHOUT A SYNC IN THE HOT PATH.  D's decode rounds are CUDA-graph replays,
so nothing on the host sees them per call.  Each measured call is bracketed by
two single-thread device kernels captured with it: a %globaltimer STAMP before
the GEMM and an ACCUMULATE after it that atomically adds (ns, weight bytes, 1)
into a per-slot int64 accumulator -- slot = (ggml type, M, N, K), labelled with
the branch the dispatch reports for it.
They replay with the graph, so every round counts itself on the card.  The host
only copies the cumulative accumulator to pinned memory on a side stream every
N rounds and reads it when that copy's event has completed (``query``, never
``synchronize``): a line is at worst one window late, never a wait.  Cost when
on: two tiny kernels per GGUF call (~2-4 us each on the replay); when off
(default): one ``None`` check per call, nothing captured.

WHERE THE ACCUMULATOR LIVES: :func:`arm` allocates it in the model runner right
before the graphs are captured, outside every memory-saver region -- born in the
cuda_graph or a weights tag it would come back from a weg2 wake on recycled pages.

BOUNDS, stated: the snapshot is not fenced against in-flight rounds (a window may
miss or gain the calls of the 1-2 rounds still on the card -- over 512 rounds that
is < 0.4 %); calls with M > ``M_MAX`` (prefill) are not measured; the device time
is the span stamp -> accumulate, i.e. the GEMM plus, on DEQ, its dequant kernel;
two calls of one slot running concurrently on two streams share one stamp.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV = "SGLANG_GGUF_PATH_CENSUS"
DEFAULT_ROUNDS = 512
#: Calls above this M are prefill-shaped and not measured (their launch cost on
#: the eager path would be the instrument's own perturbation).
M_MAX = 64
M_BUCKETS: Tuple[Tuple[int, int], ...] = ((1, 8), (9, 16), (17, 32), (33, 64))
PATHS = ("mmvq", "mmq", "deq", "dense")
#: Distinct (type, M, N, K) slots the accumulator can hold; a new one past this
#: is folded into the last slot -- the accumulator cannot grow once a graph has
#: captured its address.
MAX_KEYS = 1024
_OTHER = MAX_KEYS - 1


def census_rounds() -> int:
    """Rounds per line; 0 = off. ``1`` = the default window. A malformed value
    raises here and :func:`census` turns it into a loud WARNING with the
    instrument off."""
    raw = os.environ.get(ENV, "").strip()
    if raw in ("", "0"):
        return 0
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(
            f"{ENV}={raw!r}: expected an integer round count " f"(1 = {DEFAULT_ROUNDS})"
        ) from None
    if n < 0:
        raise ValueError(f"{ENV}={raw!r}: must be >= 0")
    return DEFAULT_ROUNDS if n == 1 else n


def m_bucket(m: int) -> Optional[int]:
    for i, (lo, hi) in enumerate(M_BUCKETS):
        if lo <= int(m) <= hi:
            return i
    return None


def bucket_label(i: int) -> str:
    lo, hi = M_BUCKETS[i]
    return f"m{lo}-{hi}"


def _ignore(path: str) -> None:
    """The reporter handed to an unmeasured call."""


class _TritonOps:
    """The two device kernels (single thread each, graph-capturable)."""

    def __init__(self):
        import triton
        import triton.language as tl
        from triton.language.extra import cuda as tlc

        @triton.jit
        def _stamp(ts_ptr, slot):
            tl.store(ts_ptr + slot, tlc.globaltimer())

        @triton.jit
        def _accum(ts_ptr, acc_ptr, slot, nbytes):
            t1 = tlc.globaltimer()
            t0 = tl.load(ts_ptr + slot)
            tl.atomic_add(acc_ptr + slot * 3 + 0, t1 - t0)
            tl.atomic_add(acc_ptr + slot * 3 + 1, nbytes)
            tl.atomic_add(acc_ptr + slot * 3 + 2, 1)

        self._stamp_k = _stamp
        self._accum_k = _accum

    def alloc(self, device):
        """The accumulator, the stamps, two pinned snapshot buffers and a side
        stream -- and both kernels compiled and loaded NOW (a first launch inside
        a stream capture would load a module there)."""
        import torch

        acc = torch.zeros((MAX_KEYS, 3), dtype=torch.int64, device=device)
        ts = torch.zeros((MAX_KEYS,), dtype=torch.int64, device=device)
        self.stamp(ts, _OTHER)
        self.accum(ts, acc, _OTHER, 0)
        acc.zero_()
        self._host = [
            torch.empty((MAX_KEYS, 3), dtype=torch.int64, pin_memory=True)
            for _ in range(2)
        ]
        self._events = [torch.cuda.Event() for _ in range(2)]
        self._turn = 0
        self._side = torch.cuda.Stream(device=acc.device)
        return acc, ts

    def stamp(self, ts, slot: int) -> None:
        self._stamp_k[(1,)](ts, slot)

    def accum(self, ts, acc, slot: int, nbytes: int) -> None:
        self._accum_k[(1,)](ts, acc, slot, nbytes)

    def snapshot(self, acc):
        """Queue a non-blocking copy of the accumulator into the next pinned
        buffer on the side stream; returns (buffer, ready predicate). The caller
        keeps at most one snapshot pending, so two buffers never overlap."""
        import torch

        i = self._turn
        self._turn = 1 - i
        with torch.cuda.stream(self._side):
            self._host[i].copy_(acc, non_blocking=True)
            self._events[i].record(self._side)
        return self._host[i], self._events[i].query

    def capturing(self) -> bool:
        import torch

        return bool(torch.cuda.is_current_stream_capturing())


class PathCensus:
    """Per-process (= per card) census. ``measure`` brackets one GGUF call;
    ``end_round`` closes one decode round on the host.

    A SLOT is one (ggml type, M, N, K): the dispatch is a pure function of those
    for a fixed process configuration, so the slot is known BEFORE the call (the
    stamp needs it) and the branch the dispatch reports for it labels the slot.
    A slot that ever reports a second branch is labelled ``mixed``."""

    def __init__(self, rounds: int, ops=None):
        self.rounds = int(rounds)
        self._ops = ops if ops is not None else _TritonOps()
        self._acc = None
        self._ts = None
        self._slots: Dict[Tuple[int, int, int, int], int] = {}
        self._label: Dict[int, str] = {}
        self._key_of: Dict[int, Tuple[int, int, int, int]] = {}
        self._rounds_by_bs: Dict[int, int] = {}
        self._n = 0
        self._pending = None  # (host copy, ready predicate, rounds_by_bs, n)
        self._prev = None
        self._unarmed = 0
        self.lines: List[str] = []

    # -- the hot path ---------------------------------------------------------
    def _slot(self, qtype: int, m: int, n: int, k: int) -> int:
        key = (int(qtype), int(m), int(n), int(k))
        slot = self._slots.get(key)
        if slot is None:
            slot = len(self._slots) if len(self._slots) < _OTHER else _OTHER
            if slot != _OTHER:
                self._slots[key] = slot
                self._key_of[slot] = key
        return slot

    def measure(self, fn, x, qweight, qweight_type):
        """``fn(x, qweight, qweight_type, on_path)`` bracketed by the stamp and
        the accumulate. ``on_path`` is the dispatch's OWN report of the branch
        it took -- the census never re-derives the choice. An unmeasured call
        gets a no-op reporter (never None: None is the dispatch's cue to ask
        the census, and the census is the caller)."""
        m = int(x.shape[0])
        if m_bucket(m) is None:
            return fn(x, qweight, qweight_type, _ignore)
        if self._acc is None:
            if self._ops.capturing():
                # The accumulator must exist before a graph captures its
                # address; a first call inside a capture stays unmeasured.
                self._unarmed += 1
                return fn(x, qweight, qweight_type, _ignore)
            self._acc, self._ts = self._ops.alloc(x.device)
        slot = self._slot(qweight_type, m, qweight.shape[0], qweight.shape[-1])
        self._ops.stamp(self._ts, slot)
        chosen: List[str] = []
        y = fn(x, qweight, qweight_type, chosen.append)
        path = chosen[-1] if chosen else "none"
        known = self._label.get(slot)
        self._label[slot] = path if known in (None, path) else "mixed"
        self._ops.accum(
            self._ts,
            self._acc,
            slot,
            int(qweight.numel()) * int(qweight.element_size()),
        )
        return y

    # -- the round, on the host --------------------------------------------------
    def end_round(self, bs: int = 0) -> Optional[List[str]]:
        """Close one decode round. Every ``rounds`` rounds a non-blocking copy of
        the accumulator is queued; a queued copy is read only once its event has
        completed -- never waited for."""
        self._rounds_by_bs[int(bs)] = self._rounds_by_bs.get(int(bs), 0) + 1
        self._n += 1
        out = None
        if self._pending is not None:
            host, ready, by_bs, n = self._pending
            if ready():
                out = self._emit(host, by_bs, n)
                self._pending = None
        if self._n >= self.rounds and self._pending is None and self._acc is not None:
            host, ready = self._ops.snapshot(self._acc)
            self._pending = (host, ready, dict(self._rounds_by_bs), self._n)
            self._rounds_by_bs = {}
            self._n = 0
        return out

    def _emit(self, host, by_bs: Dict[int, int], n: int) -> List[str]:
        snap = [list(map(int, row)) for row in host.tolist()]
        prev = self._prev
        self._prev = snap
        rows = []
        for slot in sorted(
            set(self._key_of) | ({_OTHER} if _OTHER in self._label else set())
        ):
            ns, nbytes, calls = (
                snap[slot][i] - (prev[slot][i] if prev is not None else 0)
                for i in range(3)
            )
            if calls <= 0:
                continue
            qtype, m, nn, kk = self._key_of.get(slot, (-1, 0, 0, 0))
            rows.append(
                (
                    (
                        self._label.get(slot, "none"),
                        qtype,
                        m_bucket(m) if m else -1,
                        nn,
                        kk,
                    ),
                    ns,
                    nbytes,
                    calls,
                )
            )
        lines = [_format(rows, by_bs, n, self._unarmed)]
        lines.extend(_format_paths(rows))
        for line in lines:
            logger.info(line)
        self.lines.extend(lines)
        return lines


def _type_name(qtype: int) -> str:
    try:
        import gguf

        return gguf.GGMLQuantizationType(int(qtype)).name
    except Exception:  # noqa: BLE001
        return str(qtype)


def _format(rows, by_bs, n, skipped) -> str:
    total = sum(r[1] for r in rows) or 1
    by_path = {p: 0 for p in PATHS}
    for (path, *_), ns, _b, _c in rows:
        if path in by_path:
            by_path[path] += ns
    top = sorted(rows, key=lambda r: -r[1])[:3]
    tops = ",".join(
        f"{_type_name(k[1])}[{k[3]}x{k[4]}]@{bucket_label(k[2])}/{k[0]}:{ns / 1e6:.2f}ms"
        for k, ns, _b, _c in top
    )
    return (
        f"#GGUFPATH rounds={n} bs={dict(sorted(by_bs.items()))} "
        f"dev_ms={total / 1e6:.3f} share "
        + " ".join(f"{p}={100.0 * by_path[p] / total:.1f}%" for p in PATHS)
        + f" top={tops} (device %globaltimer spans, cumulative snapshot, no sync;"
        f" unmeasured-before-arm={skipped})"
    )


def _format_paths(rows) -> List[str]:
    """One line per (M bucket, path): calls, device ms, weight GiB, per type."""
    groups: Dict[Tuple[int, str], List] = {}
    for key, ns, nbytes, calls in rows:
        groups.setdefault((key[2], key[0]), []).append((key, ns, nbytes, calls))
    out = []
    for (mb, path), items in sorted(groups.items()):
        per_type: Dict[str, List[int]] = {}
        for key, ns, nbytes, calls in items:
            t = per_type.setdefault(_type_name(key[1]), [0, 0, 0])
            t[0] += calls
            t[1] += ns
            t[2] += nbytes
        calls = sum(v[0] for v in per_type.values())
        ns = sum(v[1] for v in per_type.values())
        gib = sum(v[2] for v in per_type.values()) / 2**30
        types = ",".join(
            f"{name}:{v[0]}/{v[1] / 1e6:.2f}ms/{v[2] / 2**30:.2f}GiB"
            for name, v in sorted(per_type.items(), key=lambda kv: -kv[1][1])
        )
        out.append(
            f"#GGUFPATH {bucket_label(mb) if mb >= 0 else 'm?'} path={path} "
            f"calls={calls} dev_ms={ns / 1e6:.3f} gib={gib:.3f} types={types}"
        )
    return out


_CENSUS: Optional[PathCensus] = None
_RESOLVED = False


def census() -> Optional[PathCensus]:
    """The process's census, or None when off. Resolved once."""
    global _CENSUS, _RESOLVED
    if not _RESOLVED:
        _RESOLVED = True
        try:
            rounds = census_rounds()
        except ValueError as exc:
            logger.warning("#GGUFPATH instrument OFF -- %s", exc)
            rounds = 0
        if rounds:
            try:
                _CENSUS = PathCensus(rounds)
                logger.info(
                    "#GGUFPATH instrument on: one window every %d decode "
                    "rounds (%s)",
                    rounds,
                    ENV,
                )
            except Exception as exc:  # noqa: BLE001 -- never take the boot down
                logger.warning("#GGUFPATH instrument OFF -- %r", exc)
                _CENSUS = None
    return _CENSUS


def arm(device=None) -> bool:
    """Allocate the census accumulator now, if the census is on -- called by the
    model runner right before the graphs are captured, outside every
    memory-saver region. Idempotent; True when armed. Never raises."""
    c = census()
    if c is None or c._acc is not None:
        return c is not None
    try:
        import torch

        if device is None or str(device) in ("cuda", "musa"):
            device = torch.device("cuda", torch.cuda.current_device())
        c._acc, c._ts = c._ops.alloc(device)
        logger.info(
            "#GGUFPATH armed on %s: %d slots, outside every memory-saver "
            "region, before graph capture",
            device,
            MAX_KEYS,
        )
        return True
    except Exception as exc:  # noqa: BLE001 -- an instrument never kills a boot
        logger.warning("#GGUFPATH arm failed (%r) -- calls stay unmeasured", exc)
        return False


def reset_for_tests(instance: Optional[PathCensus] = None) -> None:
    global _CENSUS, _RESOLVED
    _CENSUS = instance
    _RESOLVED = instance is not None
