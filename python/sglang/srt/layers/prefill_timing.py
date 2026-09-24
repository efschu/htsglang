"""Per-stage forward boundaries for the SGLANG_MOE_OFFLOAD_TIMING instruments.

The prefill instruments (MOE-OFFLOAD-TIMING-PREFILL in expert_offload.py,
ATTN-TIMING-PREFILL in hybrid_linear_attn_backend.py) collect CUDA events per
layer and flush one line per forward "when the next forward starts". They
recognised that moment as ``layer_id in (0, None)``. Under pipeline
parallelism only stage 0 ever runs layer 0: stage 1 of the Next-Flash P group
starts at layer 29, stage 2 at layer 40, so on those ranks the instrument
never flushed -- it logged nothing and its event lists grew without bound
(fn7t 19.09.: 100 MOE-OFFLOAD-TIMING-PREFILL lines, every one of them PP0;
the two 3080 stages had no decomposition at all).

:class:`StageHead` names the forward boundary per process instead: the first
layer id an instrument sees is this stage's first layer, and every later call
with that id opens a new forward. On stage 0 that is layer 0 again, so the
single-stage and PP0 lines are unchanged.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Sequence

import msgspec

from sglang.srt.environ import envs
from sglang.srt.layers import host_contention

logger = logging.getLogger(__name__)

# The key of a caller that carries no layer id (desk stubs): its own value,
# so a None-only process still opens a forward on every call, as before.
_NO_LAYER = -1


class StageHead(msgspec.Struct):
    """The layer id that opens a forward on this pipeline stage.

    Learned from the first call, not configured: an instrument inside a layer
    does not know the stage's layer range, but it sees the layers in forward
    order, and the first one it ever sees is the stage's first.
    """

    head: Optional[int] = None

    def opens_forward(self, layer_id: Optional[int]) -> bool:
        key = _NO_LAYER if layer_id is None else int(layer_id)
        if self.head is None:
            self.head = key
        return key == self.head


def timing_on() -> bool:
    return bool(envs.SGLANG_MOE_OFFLOAD_TIMING.get())


def timing_event_flush_on() -> bool:
    return bool(envs.SGLANG_WEG2_ENABLE_TIMING_EVENT_FLUSH.get())


def flush_wait(events: Sequence[Any], *, instrument: str, forward: int) -> float:
    """fnFL2 H67: make the recorded CUDA events of the forward being flushed
    readable (``elapsed_time``). Returns the host wait in ms and logs it as
    ``TIMING-FLUSH-WAIT``.

    Legacy (``SGLANG_WEG2_ENABLE_TIMING_EVENT_FLUSH`` off): ``torch.cuda.
    synchronize()`` -- a wait for EVERY stream of this process, not for these
    events. On a non-last pipeline stage one of those streams carries the
    async proxy send of the previous chunk (``GroupCoordinator.
    send_tensor_dict`` -> NCCL isend, posted and never joined on the pass path,
    #1015e), and that send completes only when the next stage posts its
    receive -- after it has finished its own previous chunk. The flush runs at
    the stage's head layer INSIDE the running forward, so a stage that is
    faster than its successor sits there for the successor's remaining compute
    and the rank's gpu-ms books it as compute. Measured as FWD-TIMING linear
    minus ATTN-TIMING linear (the flush is the only thing between them at the
    head layer): x167 PP0 chunk 4/5 548/639 ms, burst forward 16 1196 ms; x166
    569/663/1361 ms; 0-3 ms on PP1/PP2 and on PP0 chunk 1-3.

    Switch on: wait on the events themselves -- the last one first (the
    instruments record on the compute stream, so it covers the rest), then any
    event that still reports incomplete. No other stream is joined; the events
    and their sums are the same.
    """
    evs = list(events)
    t0 = time.monotonic()
    if timing_event_flush_on():
        mode = "event"
        if evs:
            evs[-1].synchronize()
            for ev in evs[:-1]:
                if not ev.query():
                    ev.synchronize()
    else:
        mode = "device"
        import torch

        torch.cuda.synchronize()
    wait_ms = (time.monotonic() - t0) * 1000.0
    logger.info(
        "TIMING-FLUSH-WAIT instrument=%s forward=%d mode=%s wait_ms=%.1f events=%d "
        "t_unix_ms=%d (host wall of the instrument flush at the stage's head "
        "layer, inside the running forward, and when it returned; mode=device "
        "also joins every other stream of this process, the async PP send to "
        "the next stage included -- then it returns when the next stage has "
        "taken the previous chunk)",
        instrument,
        int(forward),
        mode,
        wait_ms,
        len(evs),
        int(time.time() * 1000.0),
    )
    return wait_ms


def log_ple_gather(rows: int, zero_rows: int, seconds: float, workers: int) -> None:
    """One line per prefill-sized PLE pread gather while the timing switch is on.

    The CPU gather runs INSIDE the forward (the embedding waits on it), so its
    wall time is idle GPU time that the rank's ``gpu-ms`` contains and none of
    the CUDA-event instruments names. Only stage 0 embeds: this is the one
    stage-0-only term, next to the expert stream that every stage pays."""
    if not timing_on():
        return
    logger.info(
        "PLE-GATHER-PREFILL rows=%d zero_rows=%d ms=%.1f workers=%d "
        "(host wall of the pread gather; the forward waits on it)",
        rows,
        zero_rows,
        seconds * 1000.0,
        workers,
    )
    _log_host_period(rows)


class _HostPeriod(msgspec.Struct):
    """The host snapshot taken at the previous PLE gather of this process."""

    last: Optional[host_contention.HostSample] = None


_PERIOD = _HostPeriod()


def format_ple_host_period(rows: int, host: "host_contention.HostDelta") -> str:
    """The PLE-HOST-PERIOD line (fnFL2 H38): the host between the previous
    PLE gather of this process and this one -- on PP0 one chunk period. With
    PLE-PREFETCH (H32) the next chunk's rows are read in exactly this window,
    by worker processes the gather's own line does not see; the serial gather
    (PLE-GATHER-HOST) is split finer. The first period of a request spans the
    idle time since the previous request: read period_ms."""
    return (
        "PLE-HOST-PERIOD rows=%d period_ms=%.1f host_busy_cores=%.2f "
        "self_cores=%.2f foreign_cores=%.2f psi_cpu_ms=%.1f psi_io_ms=%.1f "
        "arc_hits=%d arc_misses=%d (host since the previous PLE gather of this "
        "process; foreign = host CPU minus this process)"
        % (
            rows,
            host.wall_ms,
            host.host_busy_cores,
            host.self_cores,
            host.foreign_cores,
            host.psi_cpu_ms,
            host.psi_io_ms,
            host.arc_hits,
            host.arc_misses,
        )
    )


def _log_host_period(rows: int) -> None:
    now = host_contention.sample()
    last, _PERIOD.last = _PERIOD.last, now
    if last is not None:
        logger.info("%s", format_ple_host_period(rows, host_contention.delta(last, now)))


def format_ple_gather_host(
    rows: int,
    wall_ms: float,
    prep_ms: float,
    read_ms: float,
    copy_ms: float,
    split,
    host,
) -> str:
    """The PLE-GATHER-HOST line (fnFL2 H38): the host split of one
    PLE-GATHER-PREFILL wall. ``split`` is a host_contention.ThreadSplit over
    the worker tasks, ``host`` a host_contention.HostDelta over the gather.

    prep_ms  = ids to host (a stream sync) + in-range/sort/tolist
    read_ms  = the main thread's wait for the worker tasks
    copy_ms  = staging -> device enqueue (+ the reshape copy)
    thr_*    = summed over the worker tasks: on a core / runnable without a
               core (CPU contention) / neither (IO or the GIL)
    foreign_cores = host CPU busy minus this process, averaged over the gather
    arc_*    = ZFS demand-data hits/misses on the whole host meanwhile
    """
    blocked = split.blocked_ns
    return (
        "PLE-GATHER-HOST rows=%d wall_ms=%.1f prep_ms=%.1f read_ms=%.1f "
        "copy_ms=%.1f tasks=%d thr_wall_ms=%.1f thr_cpu_ms=%.1f "
        "thr_runq_ms=%.1f thr_blocked_ms=%.1f lead=%s host_busy_cores=%.2f "
        "self_cores=%.2f foreign_cores=%.2f psi_cpu_ms=%.1f psi_io_ms=%.1f "
        "arc_hits=%d arc_misses=%d (host split of PLE-GATHER-PREFILL: runq = "
        "runnable without a core, blocked = IO or GIL, foreign = host CPU "
        "minus this process)"
        % (
            rows,
            wall_ms,
            prep_ms,
            read_ms,
            copy_ms,
            split.tasks,
            split.wall_ns / 1e6,
            split.cpu_ns / 1e6,
            split.runq_ns / 1e6,
            blocked / 1e6 if blocked >= 0 else -1.0,
            split.lead(),
            host.host_busy_cores,
            host.self_cores,
            host.foreign_cores,
            host.psi_cpu_ms,
            host.psi_io_ms,
            host.arc_hits,
            host.arc_misses,
        )
    )


def log_ple_gather_host(*args) -> None:
    """Emit PLE-GATHER-HOST while the timing switch is on (see the formatter)."""
    if not timing_on():
        return
    logger.info("%s", format_ple_gather_host(*args))
