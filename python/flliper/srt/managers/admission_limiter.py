# SPDX-License-Identifier: Apache-2.0
"""Floating admission limit under a fixed session ceiling (#287).

The number of CONCURRENT sessions becomes a runtime control variable: the
state pools are dimensioned once, at boot, for the CEILING
(``--max-running-requests-ceiling``), and the effective admission limit
floats below it. Throttling admission is a counter update; rebuilding a KV
pool, a mamba slot table or a captured graph set at runtime is not -- so the
expensive dimension is fixed and the cheap one moves.

WHERE THE CEILING LANDS. ``ServerArgs`` rewrites ``max_running_requests`` to
the ceiling and remembers the user's value as the START of the float
(``max_running_requests_start``). Everything that DIMENSIONS state off
``max_running_requests`` -- ``_resolve_max_num_reqs`` and through it
``req_to_token_pool``, the mamba/GDN slot table, the hybrid KV token caps,
and the decode capture set (``get_batch_sizes_to_capture`` clamps to
``req_to_token_pool.size``) -- therefore sizes to the ceiling with no further
change. Everything that ADMITS reads this limiter instead.

DIRECTION ASYMMETRY. Lowering is immediate: KV pressure is the expensive
direction, and the whole point is to stop the inflow before the retraction
fallback starts discarding sessions that already hold state. Raising needs
proof -- ``release_hysteresis`` consecutive samples of pool occupancy at or
below ``release_low`` -- because raising on a momentary dip is how a
throttle turns into a thrash loop. This is the same asymmetry the KV
pressure ladder's sensor encodes, seen from the actuator side.

PER LANE, NOT GLOBAL (#274). There is no module-level limiter. Each owner
(the scheduler of a group, a dual-group lane) constructs its own and
publishes it into a ``contextvars.ContextVar``, following the Slice-C1
overlay idiom of ``runtime_context``: a read resolves to the limiter of the
calling thread's lane and falls back to ``None``, never to a shared
singleton. Distant readers (the #236 spill budget, the future #242 latency
classes) go through ``current_admission_limiter()`` and therefore always see
their own lane's value.

Nothing here touches torch or CUDA.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Dict, Optional

#: Relief-feature name under which the ladder (#286 Erg. 9) references this
#: actuator. Kept here so the ladder table and the limiter cannot drift.
ADMISSION_RELIEF_FEATURE = "admission_cap"

#: Defaults of the auto controller. The throttle mark sits high: the limiter
#: is the LAST cheap thing before retraction, not a general-purpose governor.
DEFAULT_THROTTLE_HIGH = 0.90
DEFAULT_RELEASE_LOW = 0.70
DEFAULT_RELEASE_HYSTERESIS = 8
DEFAULT_FLOOR = 1

#: A release raises the limit by ``current // RELEASE_STEP_DIVISOR`` (at
#: least 1), so recovering from a deep throttle is geometric rather than
#: linear -- a +1 step would take thousands of decode rounds to walk a
#: 256-wide ceiling back up, which is indistinguishable from never.
RELEASE_STEP_DIVISOR = 8

#: Reasons recorded on a limit change; part of the snapshot contract.
REASON_INIT = "init"
REASON_API = "api"
REASON_KV_PRESSURE = "kv_pressure"
REASON_PRE_RETRACT = "pre_retract"
REASON_RELEASE = "release"


class AdmissionLimitError(ValueError):
    """Invalid admission-limit configuration or runtime request."""


class AdmissionLimiter:
    """The floating concurrent-session limit of ONE group/lane.

    ``ceiling`` is what the pools were built for and can never be exceeded;
    ``current`` is what admission actually honours right now.
    """

    def __init__(
        self,
        ceiling: int,
        start: Optional[int] = None,
        *,
        floor: int = DEFAULT_FLOOR,
        throttle_high: float = DEFAULT_THROTTLE_HIGH,
        release_low: float = DEFAULT_RELEASE_LOW,
        release_hysteresis: int = DEFAULT_RELEASE_HYSTERESIS,
        auto: bool = False,
        lane_id: Optional[int] = None,
    ):
        ceiling = int(ceiling)
        if ceiling < 1:
            raise AdmissionLimitError(f"ceiling must be >= 1, got {ceiling}")
        floor = int(floor)
        if floor < 1:
            raise AdmissionLimitError(f"floor must be >= 1, got {floor}")
        if floor > ceiling:
            raise AdmissionLimitError(
                f"floor {floor} exceeds ceiling {ceiling}: the admission limit "
                f"would have no range to float in."
            )
        throttle_high = float(throttle_high)
        release_low = float(release_low)
        if not 0.0 < throttle_high <= 1.0:
            raise AdmissionLimitError(
                f"throttle_high must be in (0, 1], got {throttle_high}"
            )
        if not 0.0 < release_low < throttle_high:
            raise AdmissionLimitError(
                f"release_low must be in (0, throttle_high={throttle_high}), got "
                f"{release_low}. Equal marks make the controller flap on every "
                f"sample; the gap between them IS the hysteresis band."
            )
        release_hysteresis = int(release_hysteresis)
        if release_hysteresis < 1:
            raise AdmissionLimitError(
                f"release_hysteresis must be >= 1, got {release_hysteresis}"
            )

        start = ceiling if start is None else int(start)
        if not floor <= start <= ceiling:
            raise AdmissionLimitError(
                f"start {start} is outside [floor={floor}, ceiling={ceiling}]"
            )

        self.ceiling = ceiling
        self.floor = floor
        self.start = start
        self.throttle_high = throttle_high
        self.release_low = release_low
        self.release_hysteresis = release_hysteresis
        #: Auto controller on/off. Off = a passive holder: ``current`` stays
        #: at ``start`` unless an operator moves it through the API, so the
        #: admission arithmetic is a no-op and the default path is unchanged.
        self.auto = bool(auto)
        self.lane_id = lane_id

        self._current = start
        self._low_streak = 0
        self._last_reason = REASON_INIT
        self.throttle_count = 0
        self.release_count = 0

    # -- state ------------------------------------------------------------

    @property
    def current(self) -> int:
        """The admission limit in force right now."""
        return self._current

    @property
    def last_reason(self) -> str:
        return self._last_reason

    def _move_to(self, value: int, reason: str) -> bool:
        value = max(self.floor, min(self.ceiling, int(value)))
        if value == self._current:
            return False
        self._current = value
        self._last_reason = reason
        return True

    # -- operator control -------------------------------------------------

    def set_limit(self, value: int) -> None:
        """API setter. Rejects anything outside [floor, ceiling] -- there is
        no silent clamping, an out-of-range request is an operator error."""
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise AdmissionLimitError(
                f"admission limit must be an integer, got {value!r}"
            ) from None
        if not self.floor <= value <= self.ceiling:
            raise AdmissionLimitError(
                f"admission limit {value} is out of range "
                f"[{self.floor}, {self.ceiling}]. The ceiling is what the state "
                f"pools were dimensioned for at boot and cannot be raised at "
                f"runtime; restart with a larger "
                f"--max-running-requests-ceiling."
            )
        self._move_to(value, REASON_API)
        self._low_streak = 0

    # -- automatic control ------------------------------------------------

    def throttle(self, running_bs: int, reason: str = REASON_KV_PRESSURE) -> bool:
        """Lower the limit so no further request is admitted until the
        running batch has drained by at least one.

        The primitive itself does not consult ``auto`` -- that gate belongs to
        the callers, so a test (or a future manual mode) can move the limit
        without arming the controller.
        """
        running_bs = max(0, int(running_bs))
        target = min(self._current, running_bs) - 1
        changed = self._move_to(target, reason)
        if changed:
            self.throttle_count += 1
        self._low_streak = 0
        return changed

    def observe(self, usage: float, running_bs: int) -> bool:
        """One pressure sample (pool occupancy fraction 0..1). Returns True
        when the limit moved. No-op unless the auto controller is armed."""
        if not self.auto:
            return False
        usage = float(usage)
        if usage >= self.throttle_high:
            return self.throttle(running_bs, REASON_KV_PRESSURE)
        if usage <= self.release_low:
            self._low_streak += 1
            if self._low_streak < self.release_hysteresis:
                return False
            self._low_streak = 0
            step = max(1, self._current // RELEASE_STEP_DIVISOR)
            changed = self._move_to(self._current + step, REASON_RELEASE)
            if changed:
                self.release_count += 1
            return changed
        # Inside the hysteresis band: hold, and forget the partial streak so
        # a release always rests on CONSECUTIVE evidence of free headroom.
        self._low_streak = 0
        return False

    # -- reporting --------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "current": self._current,
            "ceiling": self.ceiling,
            "floor": self.floor,
            "start": self.start,
            "auto": self.auto,
            "lane_id": self.lane_id,
            "last_reason": self._last_reason,
            "throttle_count": self.throttle_count,
            "release_count": self.release_count,
        }


# ---------------------------------------------------------------------------
# Per-lane resolution (#274 Slice C1 overlay idiom -- no module singleton).
# ---------------------------------------------------------------------------

_ADMISSION_LIMITER: contextvars.ContextVar[Optional[AdmissionLimiter]] = (
    contextvars.ContextVar("runtime.admission_limiter", default=None)
)


def set_admission_limiter(limiter: Optional[AdmissionLimiter]) -> Any:
    """Publish ``limiter`` for the calling context. Returns the token so a
    caller that owns the whole context (the scheduler's own thread) can reset
    it; lane code should prefer ``admission_limiter_scope``."""
    return _ADMISSION_LIMITER.set(limiter)


def current_admission_limiter() -> Optional[AdmissionLimiter]:
    """The limiter of the calling lane, or None when none is published."""
    return _ADMISSION_LIMITER.get()


@contextlib.contextmanager
def admission_limiter_scope(limiter: Optional[AdmissionLimiter]):
    """Install ``limiter`` for the duration of the block. A lane thread wraps
    its tick in this, so a read inside the lane can never resolve to the
    serving group's limiter (and vice versa)."""
    token = _ADMISSION_LIMITER.set(limiter)
    try:
        yield limiter
    finally:
        _ADMISSION_LIMITER.reset(token)


# ---------------------------------------------------------------------------
# Pure helpers shared with the call sites.
# ---------------------------------------------------------------------------


def resolve_admission_start(
    ceiling_per_worker: int,
    start_raw: Optional[int],
    dp_size: int = 1,
    floor: int = DEFAULT_FLOOR,
) -> int:
    """The float's start value in the units the scheduler admits in.

    ``start_raw`` is the user's ``--max-running-requests``, i.e. a
    SERVER-WIDE figure, and is divided by ``dp_size`` exactly the way
    ``_resolve_max_num_reqs`` divides the ceiling. The result is then clamped
    to ``ceiling_per_worker`` -- the RESOLVED ceiling, which memory pressure
    (KV capacity, the post-capture re-cap) may have pushed below the
    requested one. A start above that is not reachable, so it is not a start.
    """
    ceiling_per_worker = int(ceiling_per_worker)
    if ceiling_per_worker < 1:
        raise AdmissionLimitError(
            f"ceiling_per_worker must be >= 1, got {ceiling_per_worker}"
        )
    if start_raw is None:
        return ceiling_per_worker
    dp_size = max(1, int(dp_size))
    start = max(1, int(start_raw) // dp_size)
    return max(min(floor, ceiling_per_worker), min(start, ceiling_per_worker))


def replicated_pool_usage(held_tokens: int, capacity_tokens: int) -> float:
    """Pool occupancy fraction from REPLICATED inputs only.

    ``held_tokens`` is the token count of the live requests (every rank knows
    every request's length) and ``capacity_tokens`` is the min-reduced
    ``max_total_num_tokens`` the whole group agrees on -- so every rank
    computes the same number without a collective. A rank-local occupancy
    (an allocator's ``available_size``) would diverge under uneven DCP and
    turn a throttle verdict into a source of collective desync.

    Evictable radix-cache tokens are deliberately not counted: they are
    reclaimable on demand and therefore not pressure.
    """
    capacity_tokens = int(capacity_tokens)
    if capacity_tokens <= 0:
        return 0.0
    return max(0.0, min(1.0, int(held_tokens) / capacity_tokens))


def spill_session_cap(
    configured: int, limiter: Optional[AdmissionLimiter] = None
) -> int:
    """Effective concurrent-spilled-session cap for the #236 spill budget.

    The budget's own ``--kv-session-offload-budget-max-sessions`` regler and
    the floating admission limit are the same quantity seen from two sides,
    so the spill budget reads the limiter rather than keeping a second
    number: a throttled server must not keep more sessions parked on the host
    than it is willing to run. 0/absent on either side means "not a limit
    from that side"; the result is the tighter of whichever sides speak.

    Only an ARMED limiter speaks: without --max-running-requests-ceiling the
    limit cannot move, so consulting it could only ever restate a bound the
    spill path already satisfies (a spilled session is a running session).
    """
    configured = max(0, int(configured))
    if limiter is None or not limiter.auto:
        return configured
    live = limiter.current
    if configured <= 0:
        return live
    return min(configured, live)


def throttle_before_retract(
    limiter: Optional[AdmissionLimiter], running_bs: int
) -> bool:
    """Lower admission BEFORE the retraction fallback discards sessions.

    Retraction still runs -- it is what frees the tokens the current decode
    step needs, and nothing else can do that in time. What this prevents is
    the loop AFTER it: without a throttle the freed slots are handed straight
    back to the waiting queue on the next prefill pass, and the same pressure
    retracts the next victim. Throttling the inflow first turns a repeated
    discard into a single one.

    Gated on the auto controller: with no ceiling configured the limit must
    not move on its own, or the default path would silently start throttling.
    """
    if limiter is None or not limiter.auto:
        return False
    return limiter.throttle(running_bs, REASON_PRE_RETRACT)
