"""H2 (23.09.): the HiCache write path's host cost, measured, and its device-index form.

WHY THIS EXISTS. Under ``--hicache-io-backend direct --hicache-mem-layout
layer_first`` (the Weg-2 group P form, boots fnFL2x80..x87) every write op
normalises its indices through ``CacheController.move_indices``, whose direct
branch is ``device_indices.cpu()``. That D2H runs on the COMPUTE stream before
the op enters the write stream, so the scheduler thread blocks until the work
queued on this rank's card has finished. Measured as ``WEG2 CHUNK-PUBLISH ...
'ms'`` for the second chunk of one 4544-token prompt: PP0 544/618/896/923/943
ms and PP1 355/1878/1956/1536/1658 ms (x83/x82/x81/x80/x87), against 25-63 ms
for the first chunk's publish on PP0/PP2 -- the publish's own work is tens of
ms, the rest is the scheduler thread waiting for the card.

THE DEVICE-INDEX FORM (``SGLANG_OPT_HICACHE_DEVICE_INDEX_WRITE``, default on).
Every pool on that path copes with device indices on the card: the KV and
mamba arena pools select with ``index_select`` on the device and write with the
pointer/stride kernel (xsn345/xsn351), and the QSA sidecar pool has a
``transfer_kv_all_layer_mla`` branch for page rows. When every pool of the op
says so (``backup_accepts_device_indices`` returns "") and neither the draft
tier nor the uneven-DCP owner rule is armed for the op, the controller skips
``move_hybrid_indices`` and hands the device indices through unchanged. Same
bytes, same write stream behind the same start event, same ack; the sweep
before the flip is untouched -- only the host no longer waits for the card.
Anything else takes the old path and is counted by reason.

THE INSTRUMENT (``H2-WRITE-PATH`` per chunk publish): wall and thread-CPU time
of the publish and their difference (``blocked_ms``: the scheduler thread
waiting on the card or the GIL), the write ops' own issue time and the part of
it spent normalising indices (``move_ms``, where the old D2H sits), and how
many ops entered while the compute stream still had queued work
(``busy_at_entry`` -- the case in which a sync costs the rest of a forward).
``SGLANG_DEBUG_HICACHE_SYNC_TRACE=N`` additionally runs the first N chunk
publishes under ``torch.cuda.set_sync_debug_mode("warn")`` and names each
implicit synchronising call site once (``H2-SYNC-SITE``).
"""

from __future__ import annotations

import logging
import time
import warnings
from contextlib import contextmanager
from typing import Iterator

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class WritePathStats(msgspec.Struct):
    """Cumulative counters of the write ops' host side (one per process)."""

    ops: int = 0
    device_ops: int = 0
    fallback_ops: int = 0
    busy_at_entry: int = 0
    issue_wall_ms: float = 0.0
    issue_cpu_ms: float = 0.0
    move_wall_ms: float = 0.0
    fallback_reasons: dict[str, int] = {}

    def copy(self) -> "WritePathStats":
        return msgspec.structs.replace(
            self, fallback_reasons=dict(self.fallback_reasons)
        )

    def since(self, earlier: "WritePathStats") -> "WritePathStats":
        """What accumulated after ``earlier`` was copied."""
        reasons = {}
        for k, v in self.fallback_reasons.items():
            d = v - earlier.fallback_reasons.get(k, 0)
            if d:
                reasons[k] = d
        return WritePathStats(
            ops=self.ops - earlier.ops,
            device_ops=self.device_ops - earlier.device_ops,
            fallback_ops=self.fallback_ops - earlier.fallback_ops,
            busy_at_entry=self.busy_at_entry - earlier.busy_at_entry,
            issue_wall_ms=self.issue_wall_ms - earlier.issue_wall_ms,
            issue_cpu_ms=self.issue_cpu_ms - earlier.issue_cpu_ms,
            move_wall_ms=self.move_wall_ms - earlier.move_wall_ms,
            fallback_reasons=reasons,
        )

    def record(
        self,
        *,
        device: bool,
        reason: str,
        busy: bool,
        wall_ms: float,
        cpu_ms: float,
        move_ms: float,
    ) -> None:
        self.ops += 1
        if device:
            self.device_ops += 1
        else:
            self.fallback_ops += 1
            self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
        self.busy_at_entry += int(busy)
        self.issue_wall_ms += wall_ms
        self.issue_cpu_ms += cpu_ms
        self.move_wall_ms += move_ms


class PublishTally(msgspec.Struct):
    """Cumulative chunk-publish wall/CPU of this process (the line's tail)."""

    n: int = 0
    wall_ms: float = 0.0
    cpu_ms: float = 0.0


class SyncTraceState(msgspec.Struct):
    used: int = 0
    seen: set[str] = set()


#: One of each per process (= per rank): the write ops of every controller the
#: rank runs (PP stack, flip stack) add here; readers take differences.
STATS = WritePathStats()
TALLY = PublishTally()
_SYNC_TRACE = SyncTraceState()


def device_index_write_on() -> bool:
    return envs.SGLANG_OPT_HICACHE_DEVICE_INDEX_WRITE.get()


def device_index_write_gate(*, enabled: bool, io_backend: str, device_on_card: bool) -> str:
    """The op-independent half of the decision: "" = ask the draft/DCP state
    and the pools next; otherwise the reason the old normalisation runs."""
    if not enabled:
        return "off"
    if io_backend != "direct":
        # the kernel backends move HOST indices to the card (non-blocking);
        # they never had the D2H this form removes
        return "io_backend"
    if not device_on_card:
        return "device_indices_on_host"
    return ""


def compute_stream_busy(device_module) -> bool:
    """True while the current (compute) stream still has queued work -- a
    non-blocking query, never a sync."""
    return not device_module.current_stream().query()


class IssueClock:
    """Times one write op's host side (``start_writing``)."""

    __slots__ = ("_w0", "_c0", "_m0", "move_ms", "busy")

    def __init__(self, busy: bool) -> None:
        self._w0 = time.perf_counter()
        self._c0 = time.thread_time()
        self._m0 = 0.0
        self.move_ms = 0.0
        self.busy = busy

    def move_begin(self) -> None:
        self._m0 = time.perf_counter()

    def move_end(self) -> None:
        self.move_ms += (time.perf_counter() - self._m0) * 1000.0

    def finish(self, *, device: bool, reason: str) -> None:
        STATS.record(
            device=device,
            reason=reason,
            busy=self.busy,
            wall_ms=(time.perf_counter() - self._w0) * 1000.0,
            cpu_ms=(time.thread_time() - self._c0) * 1000.0,
            move_ms=self.move_ms,
        )


def format_publish_line(
    *,
    n: int,
    wall_ms: float,
    cpu_ms: float,
    delta: WritePathStats,
    tally: PublishTally,
) -> str:
    """The ``H2-WRITE-PATH`` line of one chunk publish."""
    mode = "device" if device_index_write_on() else "legacy"
    reasons = ",".join(f"{k}:{v}" for k, v in sorted(delta.fallback_reasons.items()))
    return (
        "H2-WRITE-PATH chunk n=%d mode=%s wall_ms=%.1f cpu_ms=%.1f blocked_ms=%.1f "
        "ops=%d device_ops=%d fallback_ops=%d fallback=[%s] busy_at_entry=%d "
        "issue_wall_ms=%.1f issue_cpu_ms=%.1f move_ms=%.1f | cum n=%d wall_ms=%.0f "
        "blocked_ms=%.0f (blocked = wall - thread CPU: the scheduler thread waiting "
        "on the card or the GIL; move = index normalisation, where the old D2H sits)"
        % (
            n,
            mode,
            wall_ms,
            cpu_ms,
            max(0.0, wall_ms - cpu_ms),
            delta.ops,
            delta.device_ops,
            delta.fallback_ops,
            reasons,
            delta.busy_at_entry,
            delta.issue_wall_ms,
            delta.issue_cpu_ms,
            delta.move_wall_ms,
            tally.n,
            tally.wall_ms,
            max(0.0, tally.wall_ms - tally.cpu_ms),
        )
    )


class PublishClock:
    """Wraps one chunk publish: wall, thread CPU, and the write-op deltas."""

    __slots__ = ("_w0", "_c0", "_s0")

    def __init__(self) -> None:
        self._s0 = STATS.copy()
        self._w0 = time.perf_counter()
        self._c0 = time.thread_time()

    def finish(self, n: int) -> str:
        wall_ms = (time.perf_counter() - self._w0) * 1000.0
        cpu_ms = (time.thread_time() - self._c0) * 1000.0
        TALLY.n += 1
        TALLY.wall_ms += wall_ms
        TALLY.cpu_ms += cpu_ms
        return format_publish_line(
            n=n,
            wall_ms=wall_ms,
            cpu_ms=cpu_ms,
            delta=STATS.since(self._s0),
            tally=TALLY,
        )


@contextmanager
def sync_trace() -> Iterator[None]:
    """For the first ``SGLANG_DEBUG_HICACHE_SYNC_TRACE`` calls: run the body
    under torch's sync-debug ``warn`` mode and log each implicit synchronising
    call site once (``H2-SYNC-SITE``). A diagnostic only: ``catch_warnings`` is
    process-global, so another thread's sync warning in that window lands here
    too (still a sync site, not necessarily this body's)."""
    budget = envs.SGLANG_DEBUG_HICACHE_SYNC_TRACE.get()
    if budget <= 0 or _SYNC_TRACE.used >= budget or not torch.cuda.is_available():
        yield
        return
    _SYNC_TRACE.used += 1
    prev = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            yield
        finally:
            torch.cuda.set_sync_debug_mode(prev)
    for w in caught:
        site = "%s:%d" % (w.filename, w.lineno)
        if site in _SYNC_TRACE.seen:
            continue
        _SYNC_TRACE.seen.add(site)
        logger.warning(
            "H2-SYNC-SITE trace=%d site=%s msg=%s",
            _SYNC_TRACE.used,
            site,
            str(w.message).splitlines()[0][:160],
        )
