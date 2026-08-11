# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#656 spec item 16, THE ACTUATOR: the rebalance tier lends continuously.

WHAT WAS MISSING, IN ONE MEASUREMENT
------------------------------------
Successor 35 priced item 16 over successor 34's own green acceptance window
(28881 samples at 100 ms) and again over an independent window; the two agree
within 6 MiB. At the green run's BINDING INSTANT the free column was

    [1043, 3280, 1541] MiB     (corridor law: 1024 MiB per card)

Card 0 sat 19 MiB above the law while card 1 held 3280 MiB free, 912 MiB of
which the water-fill would have moved onto it. The median levelling
opportunity across the whole window was ~1004 MiB. **The tightest moment of a
green acceptance was a placement problem, not a capacity problem**, and the
corridor's second half -- "fill the cards as fully as possible" -- was being
paid for by one card carrying the whole transient.

WHY THE GATE ALONE CANNOT FIX THAT
----------------------------------
:class:`~sglang.srt.managers.corridor_guard.CorridorGuard` is reactive by
construction: it arms when an allocation would breach the floor. By then the
trough has happened. It is the right shape for a LAW -- never breach -- and
the wrong shape for an OBJECTIVE -- stay level -- because an objective has no
allocation to hang off. So the tier existed, sorted correctly, and had one
provider that was only ever spent at the last possible instant.

WHAT THIS DOES
--------------
The lender is the same ladder on a different schedule. Every time the hot
path passes a pressure point it asks: is this card the one the water-fill
says must SHED, and is it under pressure right now? If both, it spends the
REBALANCE tier and everything cheaper -- allocator cache, then the
non-layer-bound payloads -- up to the water-fill's own bound, and stops.

    trigger   the SAME pressure signal as the gate (free below the upper
              watermark ``floor + delta``), evaluated continuously rather
              than only when an allocation arrives
    direction the water-fill: only the card that must shed lends, never the
              one that should absorb
    bound     ``min(water-fill shed, distance to the upper watermark)`` --
              the objective bounds the spend, so a lend cannot overshoot
              into being the unevenness with the sign flipped
    ceiling   tier REBALANCE. Park and host RAM are unreachable from here by
              construction, not by convention (see ``lend_to_level``).

WHAT IT IS NOT
--------------
* **Not a daemon.** No thread, no timer, no background NVML poll. It runs on
  the caller's stack at a point the caller already reached, so it cannot race
  the flip seam or wake a rank that is inside a collective.
* **Not a KV shrink.** The KV rung is collective and moves ``available_size``;
  a rank-local lender that touched it would be "a smaller pool as the fix".
  The pool is not the funder here and the lender never asks it.
* **Not a cross-card KV move.** That one remains blocked and it is worth
  naming precisely, because it is the half of item 16 this actuator does NOT
  deliver: under the DCP owner rule (``layers/dcp/owner.py:348``) a token's
  physical row is a pure function of the token vector, so changing the vector
  while rows are live re-owns existing KV -- which is why the vector is
  installed once at boot and merely RE-installed at every cutover
  (``phase_flip_runtime.py:866``, "the vector is boot-constant"), and why
  moving KV between cards needs ``kv_reshard``, whose stage-A guards require
  a fully idle instance (``kv_reshard.py:781``). In the PP phase KV is
  layer-bound in any case. So the levelling this actuator performs is the
  one item 16 prescribes for that phase: **everything not layer-bound leaves
  the card that binds, first, and before anything reaches host RAM.**

THRASH IS PRICED, NOT ASSUMED AWAY
----------------------------------
The cheapest provider (torch's allocator cache) is not free -- it
synchronises -- and the drafter's restore costs real milliseconds. So the
lender rate-limits, and backs OFF exponentially when a lend yields little,
recovering immediately when one yields well. A lender that fires every step
and frees 8 MiB is a throughput regression wearing a levelling costume.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

from sglang.srt.managers import corridor_guard as cg

logger = logging.getLogger(__name__)

LOG_PREFIX = "CORRIDOR-REBALANCE"

_MIB = 1024 * 1024

#: Off switch. The lender is ON under ``--enable-phase-flip`` because item 16
#: asks for it; ``=0`` restores the s34 shipped behaviour exactly (the gate
#: alone), which is what an A/B against the green window needs.
ENABLE_ENV = "SGLANG_CORRIDOR_REBALANCE"

#: Below this the fleet is level ENOUGH: the guard's own delta band is 256 MiB
#: and a levelling move smaller than the band it frees to cannot be told from
#: sampling noise on a live rig.
DEFAULT_MIN_SHED_MIB = 256

#: A lend that yields less than this counts as a miss and starts the back-off.
DEFAULT_MIN_YIELD_MIB = 64

DEFAULT_INTERVAL_S = 2.0
DEFAULT_MAX_INTERVAL_S = 30.0

#: How often the LOCAL pressure check may take a ``cudaMemGetInfo``. Much
#: shorter than the lend interval on purpose: the dips this lender exists for
#: last about a second, so a check on the lend clock would sample past them,
#: while an unlimited check runs at whatever rate the scheduler round does
#: (~8 kHz on an armed rank). 100 ms is the sampler's own resolution, which
#: keeps the instrument and the actuator looking at the same timescale.
DEFAULT_PROBE_INTERVAL_S = 0.1


class RebalanceLender:
    """Item 16's first relief stage, driven by the water-fill objective.

    One per rank, beside that rank's :class:`CorridorGuard`. ``nvml_index`` is
    this rank's column in the NVML free vector -- NOT its torch device index.
    Under ``--rank-gpu-id`` each worker sees exactly one card, so the two
    orders diverge routinely and a lender reading the wrong column would
    evacuate a card because a DIFFERENT card was tight. The mapping is
    resolved once, by UUID, at construction (see :func:`build_rebalance_lender`).
    """

    def __init__(
        self,
        guard: cg.CorridorGuard,
        *,
        nvml_index: int,
        min_shed_mib: int = DEFAULT_MIN_SHED_MIB,
        min_yield_mib: int = DEFAULT_MIN_YIELD_MIB,
        interval_s: float = DEFAULT_INTERVAL_S,
        max_interval_s: float = DEFAULT_MAX_INTERVAL_S,
        probe_interval_s: float = DEFAULT_PROBE_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.guard = guard
        self.nvml_index = int(nvml_index)
        self.min_shed_bytes = int(min_shed_mib) * _MIB
        self.min_yield_bytes = int(min_yield_mib) * _MIB
        self.base_interval_s = float(interval_s)
        self.max_interval_s = float(max_interval_s)
        self._clock = clock
        self.probe_interval_s = float(probe_interval_s)
        self._interval_s = float(interval_s)
        self._last_at = float("-inf")
        self._last_probe_at = float("-inf")
        # Counters, all reported in the periodic summary so a window can be
        # judged on them without parsing every line.
        self.consulted = 0
        self.lends = 0
        self.lent_bytes = 0
        self.skips = {
            "rate-limited": 0,
            "probe-limited": 0,
            "no-pressure": 0,
            "fleet-unknown": 0,
            "level-enough": 0,
            "absorber": 0,
        }
        self._misses = 0
        self._summary_at = float("-inf")

    # -- policy ------------------------------------------------------------

    @property
    def watermark_bytes(self) -> int:
        """The upper watermark the gate frees TO, and the lender's trigger.

        ONE number in charge of both, deliberately: a lender with its own
        threshold would be a second policy that has to be kept in agreement
        with the gate's, and this module's history is full of exactly that
        failure. Above this line the card is not under pressure and no amount
        of unevenness makes it the lender's business.
        """
        return self.guard.floor_bytes + self.guard.delta_bytes

    def _skip(self, why: str):
        self.skips[why] = self.skips.get(why, 0) + 1
        return None

    def maybe_lend(self, reason: str = "") -> Optional[cg.GuardResult]:
        """Lend if the water-fill says this card should, else return None.

        WHAT THE COMMON PATH ACTUALLY COSTS, corrected after review: a
        monotonic clock read AND, at most every ``probe_interval_s``, one
        ``cudaMemGetInfo``. The earlier claim that a healthy card costs only
        the clock read was wrong -- the pressure check deliberately does NOT
        arm the lend rate limiter, because a card that is fine must be
        re-examined promptly (the dips this lender exists for last about a
        second, so a 2 s pressure check would sample straight past them). The
        probe therefore has its own, much shorter limiter: responsive enough
        for a 1 s dip, bounded well below the ~8 kHz an armed rank can spin
        this hook at. Measured on this rig: 13081 consultations per rank over
        a 46-minute window.

        The NVML FLEET probe is still only taken once both the lend limiter
        and the local pressure check have passed, which on a healthy rig is
        never.
        """
        self.consulted += 1
        now = self._clock()
        self._maybe_report_inert(now)
        if now - self._last_at < self._interval_s:
            return self._skip("rate-limited")
        if now - self._last_probe_at < self.probe_interval_s:
            return self._skip("probe-limited")
        self._last_probe_at = now

        free_local = self.guard.free_bytes()
        headroom = self.watermark_bytes - free_local
        if headroom <= 0:
            # Not under pressure. Note this is checked BEFORE the fleet probe
            # on purpose: a level-but-idle rig must not pay an NVML round trip
            # per scheduler round to be told it has nothing to do.
            return self._skip("no-pressure")

        column = self.guard.fleet_free()
        if not column or self.nvml_index >= len(column):
            # An unreadable fleet is an UNPROVEN fleet. The guard degrades the
            # same way for the host gate: no evidence closes the door.
            self._last_at = now
            return self._skip("fleet-unknown")

        transfers = cg.water_fill_transfers(column)
        shed = int(transfers[self.nvml_index])
        if shed <= 0:
            # This card is the ABSORBER -- it has more free than the mean.
            # Evacuating it would make the column less level, not more.
            self._last_at = now
            return self._skip("absorber")
        if shed < self.min_shed_bytes:
            self._last_at = now
            return self._skip("level-enough")

        # THE BOUND IS THE OBJECTIVE, CLIPPED BY THE PRESSURE. The water-fill
        # says how much this card should shed to sit level; the watermark says
        # how much it needs right now. Freeing the smaller of the two is the
        # only choice that neither overshoots the objective nor pays a restore
        # for headroom nothing asked for.
        bound = min(shed, headroom)
        self._last_at = now
        result = self.guard.lend_to_level(
            bound,
            column=column,
            max_tier=cg.RELIEF_REBALANCE,
            reason=reason or "item 16 continuous lend",
        )
        self.lends += 1
        self.lent_bytes += int(result.reclaimed)

        if result.reclaimed < self.min_yield_bytes:
            # BACK OFF. A card whose non-layer-bound payload is already gone
            # will keep being the water-fill's shedder -- the unevenness is
            # then structural and no amount of asking changes it. Doubling the
            # interval turns a hot loop into an occasional re-check.
            self._misses += 1
            self._interval_s = min(self.max_interval_s, self._interval_s * 2)
        else:
            self._misses = 0
            self._interval_s = self.base_interval_s

        self._maybe_summarise(now, result)
        return result

    # -- reporting ---------------------------------------------------------

    def _maybe_report_inert(self, now: float) -> None:
        """Say so, periodically, when the lender has never spent anything.

        WHY THIS IS NOT OPTIONAL NOISE, and it cost this shift a window to
        relearn: every other line the lender can emit is conditional on a LEND.
        A lender wired to the wrong hot path is then byte-identical in the log
        to a lender that was never needed -- which is exactly the confusion
        that has shipped seven mechanisms in this chain. The skip histogram
        names WHICH precondition is refusing, so "wrong hook" and "fleet was
        level all window" can be told apart from the log alone.
        """
        if self.lends or self.consulted < 2:
            return
        if now - self._summary_at < 300.0:
            return
        self._summary_at = now
        logger.warning(
            "%s device %d has NOT lent once in %d consultations. Skips %s. "
            "This is the lender reporting itself INERT: either the fleet never "
            "went unlevel under pressure, or this hot path does not overlap "
            "the pressure window. The histogram says which.",
            LOG_PREFIX,
            self.guard.device_index,
            self.consulted,
            {k: v for k, v in self.skips.items() if v},
        )

    def _maybe_summarise(self, now: float, result: cg.GuardResult) -> None:
        if now - self._summary_at < 60.0:
            return
        self._summary_at = now
        logger.info(
            "%s device %d: %d lends, %.0f MiB lent in total, last one %s. "
            "Skips %s. Interval now %.1fs. This is item 16's first relief "
            "stage spending the REBALANCE tier and everything cheaper; host "
            "RAM is not reachable from here.",
            LOG_PREFIX,
            self.guard.device_index,
            self.lends,
            self.lent_bytes / _MIB,
            result.detail,
            {k: v for k, v in self.skips.items() if v},
            self._interval_s,
        )

    def summary(self) -> str:
        return (
            f"{self.lends} lends, {self.lent_bytes / _MIB:.0f} MiB lent, "
            f"{self.consulted} consultations, skips "
            f"{ {k: v for k, v in self.skips.items() if v} }"
        )


#: Where the lender is cached. ON THE GUARD, not on the scheduler, because the
#: guard is the object both call sites already resolve -- the prefill gate has
#: its own two-step lookup and the scheduler round has another. Keying the
#: cache on the guard is what stops those two paths building two lenders with
#: two rate limiters that each think they are the only one.
LENDER_ATTR = "_item16_rebalance_lender"


def lender_for_guard(guard) -> Optional[RebalanceLender]:
    """This card's lender, built once per guard and cached on it.

    ``False`` is cached for "this rank must not lend" so the build (an NVML
    UUID walk) is not retried on every scheduler round.
    """
    if guard is None:
        return None
    cached = getattr(guard, LENDER_ATTR, None)
    if cached is not None:
        return cached or None
    lender = build_rebalance_lender(guard) or False
    setattr(guard, LENDER_ATTR, lender)
    return lender or None


def lend_for_guard(guard, reason: str = "") -> None:
    """Consult the lender for ``guard``'s card. Never raises.

    A levelling optimisation that can take the scheduler down is a worse
    bargain than an unlevel column, so every failure here is a warning and a
    return.
    """
    try:
        lender = lender_for_guard(guard)
        if lender is not None:
            lender.maybe_lend(reason)
    except Exception as e:  # noqa: BLE001
        logger.warning("%s lend failed: %s", LOG_PREFIX, e)


def lend_on_round(scheduler, reason: str = "scheduler round") -> None:
    """The PER-ROUND consultation, and the reason it exists.

    WIRING THE LENDER ONLY TO PREFILL ADMISSION WAS TOO NARROW, and the first
    live window measured it: 0 lends in the first minutes while the tightest
    card spent 7.8% of its samples below the watermark. The two windows do not
    overlap. The corridor's trough is not made by the allocations the gates
    cover -- s34's gate cleared 232 times and never let free fall below
    ``floor + delta`` at a guarded site, yet the sampler still caught 1043 MiB
    -- so the depth comes from allocations no gate prices (activations, KV
    rows, mamba states, replay workspaces) and from the flip seam, where the
    scheduler is not admitting prefill at all.

    The scheduler round is the one clock that ticks inside both. It is
    rank-local and performs no collective, which is the hard requirement here
    (``phase_flip_runtime.on_round``'s docstring records the wedges that a
    collective on a rank-local cadence caused). The rate limiter makes the
    common path a monotonic clock read even at the ~8 kHz an armed rank spins.
    """
    try:
        from sglang.srt.managers.phase_flip_spill import get_corridor_guard

        guard = get_corridor_guard(scheduler)
    except Exception:  # noqa: BLE001
        return
    if guard is not None:
        lend_for_guard(guard, reason)


def lender_enabled(default: bool = True) -> bool:
    raw = os.environ.get(ENABLE_ENV)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def resolve_nvml_index(device_index: int) -> Optional[int]:
    """This rank's column in the NVML free vector, resolved by UUID.

    NOT the torch device index, and the difference is not academic on this
    rig: ``--rank-gpu-id`` pins each worker to one card via
    ``CUDA_VISIBLE_DEVICES``, so every worker's torch index is 0 while its
    NVML index is whatever card it was given, and torch's enumeration order
    can differ from NVML's regardless (registry note: device-order trap).
    A lender reading the wrong column would shed on the wrong card, which is
    worse than not lending at all -- so an unresolvable mapping returns None
    and the lender is not built.
    """
    try:
        import torch

        from sglang.srt.registry import nvml as registry_nvml

        uuid = torch.cuda.get_device_properties(device_index).uuid
        want = str(uuid).replace("GPU-", "").strip().lower()
        with registry_nvml.nvml_session() as pynvml:
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                got = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(got, bytes):
                    got = got.decode()
                if str(got).replace("GPU-", "").strip().lower() == want:
                    return index
    except Exception as e:
        logger.warning("%s could not resolve the NVML index: %s", LOG_PREFIX, e)
        return None
    logger.warning(
        "%s no NVML card matched torch device %d by UUID; the lender stays "
        "off rather than guess a column",
        LOG_PREFIX,
        device_index,
    )
    return None


def build_rebalance_lender(
    guard: cg.CorridorGuard,
    *,
    nvml_index: Optional[int] = None,
    enabled: Optional[bool] = None,
) -> Optional[RebalanceLender]:
    """Build the lender for ``guard``'s card, or None if it must not run."""
    if enabled is None:
        enabled = lender_enabled()
    if not enabled:
        logger.info(
            "%s disabled by %s; the corridor gate runs alone, which is the "
            "pre-item-16 behaviour",
            LOG_PREFIX,
            ENABLE_ENV,
        )
        return None
    if nvml_index is None:
        nvml_index = resolve_nvml_index(guard.device_index)
    if nvml_index is None:
        return None
    lender = RebalanceLender(guard, nvml_index=int(nvml_index))
    logger.info(
        "%s ARMED on torch device %d (NVML column %d), watermark %.0f MiB, "
        "min shed %.0f MiB, interval %.1fs. Item 16's first relief stage is "
        "now an actuator: the card the water-fill says must shed gives up its "
        "unowned and non-layer-bound bytes CONTINUOUSLY, before any allocation "
        "arms the gate and before anything reaches host RAM.",
        LOG_PREFIX,
        guard.device_index,
        nvml_index,
        lender.watermark_bytes / _MIB,
        lender.min_shed_bytes / _MIB,
        lender.base_interval_s,
    )
    return lender
