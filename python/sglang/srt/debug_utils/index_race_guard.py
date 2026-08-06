"""#616 index-race guard: sync-free, non-fatal instrumentation for the index
tensors of the overlap / speculative-decode path.

WHY THIS EXISTS
---------------
The residual #616 crash is a CUDA device-side assert from ATen's advanced
indexing kernel (``IndexKernel.cu:111``, "index out of bounds") whose output is
``bs * (speculative_num_steps + 1)`` wide, with only a SUBSET of the lanes out
of range. The assert kills the CUDA context, so it surfaces at whatever sync
point happens to run next -- observed at two completely unrelated sites -- and
the surface site is aftermath, never the origin. ``CUDA_LAUNCH_BLOCKING=1``
SUPPRESSES the bug entirely (15 min clean vs 3.5 min to crash), which rules out
a plain logic error and points at a cross-stream race.

Therefore an instrument for this bug must satisfy three constraints that rule
out every ordinary debugging move:

1. It must not synchronize. Any ``.item()`` / ``.cpu()`` / ``synchronize()`` on
   the hot path serializes the producer against the consumer and hides the very
   race we are hunting -- the same way CUDA_LAUNCH_BLOCKING does.
2. It must not abort. A device-side assert destroys the context, so the FIRST
   bad lane would be the last thing we ever learn. We want to keep running and
   collect the full picture across many rounds.
3. It must name the PRODUCING site, not the surface site.

MECHANISM
---------
Each guarded site gets a slot in one small device-resident stats tensor. A
guard call enqueues a handful of tiny elementwise kernels on the CURRENT stream
(so it inherits exactly the ordering of the code it instruments) which
accumulate, per slot: bad-lane count, min/max of the observed values, and a
mutation count. Nothing is ever read back on the hot path.

The host reads those stats with the #517 staged-read pattern: a non-blocking
D2H into a pinned mirror plus an event, and the value is consumed on a LATER
poll once ``event.query()`` reports ready. A poll therefore costs no stream
synchronization; it only trades reporting LATENCY (a few scheduler iterations),
never detection, because the counters are monotonic.

THE STABILITY CHECK IS THE DISCRIMINATOR
----------------------------------------
``snapshot()`` copies a tensor into a private device buffer at its PRODUCTION
point; ``check_stable()`` compares the live tensor against that snapshot at its
CONSUMPTION point and counts differing lanes. Both run on the same stream as
the code around them, so within one stream the comparison is trivially ordered
and MUST report zero. A non-zero mutation count is therefore positive proof
that some OTHER stream wrote the tensor in between -- and it names both
endpoints of the racing pair, which is exactly what the bug report is missing.

A bounds violation reported at the production site means the value was already
wrong when it was computed (a logic / stale-input fault); a clean production
site plus a dirty consumption site means the value was corrupted after the
fact (a write-after-read or allocator-reuse race). The two outcomes are
mutually exclusive and both are actionable.

RELATION TO ``utils/async_probe.maybe_detect_oob``
--------------------------------------------------
That probe already range-checks KV-cache locations, but it fires
``torch._assert_async``: it ABORTS, and a device assert is precisely what we
cannot afford here (constraint 2), because the context dies before the second
data point exists. It also has no notion of a production/consumption pair, so
it cannot distinguish a bad value from a corrupted one. The two mechanisms are
complementary: ``maybe_detect_oob`` is a production invariant, this is a
race-hunting instrument.

KNOBS (all default OFF -- production takes a single module-level bool test)
--------------------------------------------------------------------------
``SGLANG_INDEX_RACE_GUARD=1``       arm the guard
``SGLANG_INDEX_RACE_GUARD_CLAMP=1`` additionally clamp offending values back
                                    into range so the run SURVIVES the first
                                    bad batch and keeps reporting. Diagnostic
                                    only: it trades output correctness for
                                    evidence, and says so in the log.

                                    CLAMP IS NOT A MITIGATION AND MUST NEVER BE
                                    SHIPPED AS ONE. The clamp decision is
                                    rank-LOCAL: under the race only some ranks
                                    observe the bad value, so clamping makes
                                    those ranks take a different accept path
                                    from the others and DESYNCHRONISES the
                                    group. That is acceptable for a diagnostic
                                    run, whose output is already untrustworthy
                                    by the time the clamp fires, and
                                    unacceptable for anything else.
``SGLANG_INDEX_RACE_GUARD_POLL=N``  poll every N scheduler iterations
                                    (default 1).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Module-level latch: production reads one bool and returns. No import-time
# device work, no allocation, no env lookup per call.
_ENABLED: bool = envs.SGLANG_INDEX_RACE_GUARD.get()
_CLAMP: bool = envs.SGLANG_INDEX_RACE_GUARD_CLAMP.get()

# stats layout, per slot
_F_BAD = 0  # cumulative count of out-of-range lanes
_F_MIN = 1  # min value ever observed at this site
_F_MAX = 2  # max value ever observed at this site
_F_MUT = 3  # cumulative count of lanes that changed since snapshot()
_N_FIELDS = 4

_MAX_SLOTS = 64
_INT64_HI = 2**62
_INT64_LO = -(2**62)


class _GuardState:
    """All device-resident guard state for one process."""

    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        # [slot, field] accumulators. int64 so a garbage index cannot overflow
        # the min/max fields whatever it holds.
        self.stats = torch.zeros(
            (_MAX_SLOTS, _N_FIELDS), dtype=torch.int64, device=device
        )
        self.stats[:, _F_MIN].fill_(_INT64_HI)
        self.stats[:, _F_MAX].fill_(_INT64_LO)
        # CPU runs the same bookkeeping without pinning or events -- that is
        # what makes the instrument itself testable hermetically.
        self.mirror = torch.zeros(
            (_MAX_SLOTS, _N_FIELDS), dtype=torch.int64, pin_memory=self.is_cuda
        )
        self.staged_event: Optional[torch.cuda.Event] = None
        self.staged_iteration: int = -1
        # slot bookkeeping (host side only)
        self.slots: Dict[str, int] = {}
        self.slot_names: List[str] = []
        self.bounds: Dict[str, Tuple[int, int]] = {}
        self.streams: Dict[str, int] = {}
        self.reported: Dict[int, Tuple[int, int]] = {}
        self.snapshots: Dict[str, torch.Tensor] = {}
        self.iteration: int = 0
        self.overflow_warned: bool = False

    def slot(self, name: str) -> int:
        idx = self.slots.get(name)
        if idx is None:
            if len(self.slot_names) >= _MAX_SLOTS:
                if not self.overflow_warned:
                    self.overflow_warned = True
                    logger.error(
                        "[INDEX-RACE] more than %d guarded sites; site %r is not "
                        "instrumented. Raise _MAX_SLOTS.",
                        _MAX_SLOTS,
                        name,
                    )
                return -1
            idx = len(self.slot_names)
            self.slots[name] = idx
            self.slot_names.append(name)
        return idx


_state: Optional[_GuardState] = None


def is_enabled() -> bool:
    return _ENABLED


def _capturing() -> bool:
    """Whether the current stream is mid CUDA-graph capture."""
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _get_state(device: torch.device) -> Optional[_GuardState]:
    global _state
    if _state is None:
        # NEVER build the persistent state inside a capture: allocations made
        # during capture come from the graph's PRIVATE pool and are recycled
        # when that pool is reset, so the counters would silently alias
        # whatever the graph reuses next -- an instrument that corrupts itself.
        # Skipping here only defers arming to the first eager round.
        if device.type == "cuda" and _capturing():
            return None
        _state = _GuardState(device)
        logger.warning(
            "[INDEX-RACE] guard ARMED (clamp=%s). This is a diagnostic build: "
            "bounds violations are counted and reported instead of asserting, "
            "and with clamp=1 offending values are forced back into range, so "
            "generated output is NOT trustworthy on a round that reports a hit.",
            _CLAMP,
        )
    return _state


def guard(
    name: str,
    values: torch.Tensor,
    lo: int,
    hi: int,
) -> torch.Tensor:
    """Range-check ``values`` against ``[lo, hi)`` on the current stream.

    Returns ``values`` unchanged, or -- under ``SGLANG_INDEX_RACE_GUARD_CLAMP``
    -- a clamped copy so the caller's kernel cannot assert and the run survives
    to report more. Never synchronizes, never raises.
    """
    if not _ENABLED:
        return values
    if values is None or values.numel() == 0:
        return values
    st = _get_state(values.device)
    if st is None:
        return values
    slot = st.slot(name)
    if slot < 0:
        return values
    st.bounds[name] = (lo, hi)
    # Host-side attribute read, no device work: records which stream this site
    # was issued on, so the report names the stream as well as the site.
    if st.is_cuda:
        try:
            st.streams[name] = torch.cuda.current_stream(values.device).cuda_stream
        except Exception:
            pass

    row = st.stats[slot]
    flat = values.reshape(-1)
    if flat.dtype != torch.int64:
        flat = flat.to(torch.int64)
    bad = (flat < lo) | (flat >= hi)
    row[_F_BAD] += bad.sum()
    row[_F_MIN] = torch.minimum(row[_F_MIN], flat.min())
    row[_F_MAX] = torch.maximum(row[_F_MAX], flat.max())

    if _CLAMP:
        return values.clamp(lo, hi - 1)
    return values


def snapshot(name: str, values: torch.Tensor) -> None:
    """Record ``values`` at its production point for a later stability check."""
    if not _ENABLED or values is None or values.numel() == 0:
        return
    st = _get_state(values.device)
    if st is None:
        return
    prev = st.snapshots.get(name)
    if prev is None or prev.shape != values.shape or prev.dtype != values.dtype:
        # Private buffer: never aliased to the guarded tensor, so a later write
        # to the original cannot silently update the reference copy too.
        prev = torch.empty_like(values)
        st.snapshots[name] = prev
    prev.copy_(values)


def check_stable(name: str, values: torch.Tensor) -> None:
    """Count lanes of ``values`` that changed since ``snapshot(name, ...)``.

    Both calls run on the current stream, so a same-stream comparison is
    trivially ordered and must report zero. Any non-zero count is proof that a
    different stream wrote the tensor in between.
    """
    if not _ENABLED or values is None or values.numel() == 0:
        return
    st = _get_state(values.device)
    if st is None:
        return
    prev = st.snapshots.get(name)
    if prev is None or prev.shape != values.shape:
        return
    slot = st.slot(name)
    if slot < 0:
        return
    st.stats[slot][_F_MUT] += (prev != values).sum()


def poll(iteration: Optional[int] = None) -> None:
    """Stage / consume the counters without synchronizing (the #517 pattern)."""
    if not _ENABLED or _state is None:
        return
    st = _state
    if iteration is not None:
        st.iteration = iteration
    else:
        st.iteration += 1

    every = max(1, envs.SGLANG_INDEX_RACE_GUARD_POLL.get())
    if st.iteration % every:
        return

    if not st.is_cuda:
        # No streams to race: read straight through.
        st.mirror.copy_(st.stats)
        st.staged_iteration = st.iteration
        _report(st)
        return

    # Consume a previously staged copy if the device has caught up with it.
    if st.staged_event is not None and st.staged_event.query():
        _report(st)
        st.staged_event = None

    if st.staged_event is None:
        st.mirror.copy_(st.stats, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        st.staged_event = ev
        st.staged_iteration = st.iteration


def _report(st: _GuardState) -> None:
    mirror = st.mirror
    for slot, name in enumerate(st.slot_names):
        bad = int(mirror[slot][_F_BAD])
        mut = int(mirror[slot][_F_MUT])
        prev_bad, prev_mut = st.reported.get(slot, (0, 0))
        if bad == prev_bad and mut == prev_mut:
            continue
        st.reported[slot] = (bad, mut)
        lo, hi = st.bounds.get(name, (0, 0))
        vmin = int(mirror[slot][_F_MIN])
        vmax = int(mirror[slot][_F_MAX])
        logger.error(
            "[INDEX-RACE] site=%s bad_lanes=%d (+%d) mutated_lanes=%d (+%d) "
            "bounds=[%d,%d) observed_min=%s observed_max=%s stream=%s "
            "staged_at_iter=%d",
            name,
            bad,
            bad - prev_bad,
            mut,
            mut - prev_mut,
            lo,
            hi,
            "n/a" if vmin == _INT64_HI else vmin,
            "n/a" if vmax == _INT64_LO else vmax,
            hex(st.streams.get(name, 0)),
            st.staged_iteration,
        )


def _reset_for_test(enabled: bool = True, clamp: bool = False) -> None:
    """Test hook: re-latch the env gates and drop all accumulated state.

    The gates are module-level latches by design (that is what makes the
    disabled path a single bool test), so a test that wants a different setting
    has to say so explicitly rather than mutate the environment behind them.
    """
    global _ENABLED, _CLAMP, _state
    _ENABLED = enabled
    _CLAMP = clamp
    _state = None


def summary() -> str:
    """Human-readable end-of-run summary (safe to call after a crash)."""
    if _state is None:
        return "[INDEX-RACE] guard was never armed"
    st = _state
    lines = ["[INDEX-RACE] summary (last staged snapshot, may lag the crash):"]
    mirror = st.mirror
    for slot, name in enumerate(st.slot_names):
        lo, hi = st.bounds.get(name, (0, 0))
        lines.append(
            f"  {name}: bad={int(mirror[slot][_F_BAD])} "
            f"mutated={int(mirror[slot][_F_MUT])} bounds=[{lo},{hi}) "
            f"min={int(mirror[slot][_F_MIN])} max={int(mirror[slot][_F_MAX])}"
        )
    return "\n".join(lines)
