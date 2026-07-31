# SPDX-License-Identifier: Apache-2.0
"""KV pressure ladder (DESIGN_201 Nachtrag-13 Ergaenzungen 9 / 9b) -- CPU phase.

The counter-direction to Ergaenzung 8. The system runs in the PERFORMANCE-
optimal distribution (few allreduce nodes, fast card); when the KV cache
threatens to burst mid-decode/mid-prefill it climbs, in N PRESETTABLE STEPS,
towards CAPACITY: cheap KV relief first, geometry inside the nesting family
second, another node last.

WHAT THIS MODULE IS. The table, the sensor, the flip contract and the
handover INTERFACE -- nothing moves. Every real handover strategy is a
``NotImplementedError`` stub whose docstring names the measurement that has
to decide it (Ergaenzung 9 point 3 is explicitly the one open design
decision of the whole addendum). The controller plans; the GPU phase
executes.

THE STEP TABLE (``PressureLadder``). An ordered ladder of ``LadderStep``s:

* ``base``          -- rung 0, the performance-optimal state the server boots
                       in. Exactly one, always index 0.
* ``relief``        -- an EXISTING KV-relief feature, referenced BY NAME
                       (``RELIEF_FEATURES``), never reimplemented here: the
                       floating admission cap (#287), uneven-DCP token ratio,
                       KV spill to host RAM (#134/#236), the weightless KV
                       rank (#115), session offload. No KV layout change,
                       hence handover ``none``.
* ``geometry_flip`` -- a geometry INSIDE the nesting family. Thanks to the
                       down-set property the finest cut already holds every
                       coarser geometry in the SAME bytes, so this is a plan
                       flip at a round boundary (per-step graphs captured in
                       advance, cold step graphs are register class
                       ``graph_rungs`` = RAM-parkable) and NEVER a weight
                       reshard.
* ``external``      -- outside the family (extra node, remote PP stage) =
                       the Nachtrag-14 warm-standby path. Stub, longest
                       hysteresis, last rung -- never the first.

ENFORCED INVARIANTS (in code, not by convention):

1. ORDER: ``base`` < all ``relief`` < all ``geometry_flip`` < all
   ``external``. A table that interleaves them is rejected at construction
   (``PressureLadder``), so "cheap relief before expensive geometry" cannot
   be lost by a caller building the table in the wrong order.
2. Ascent is ONE rung at a time; a caller-forced target that skips a relief
   rung raises ``KvLadderError`` (``force_target``), so the ordering holds
   for the runtime path too, not just for the table.
3. CAPTURE GUARD: flips (and pre-stages) happen only at round boundaries and
   never inside an active capture -- the same contract the HTCCL path
   dispatcher and the K-ladder use. Refusal is a no-op plan with a reason,
   the status-quo direction.
4. A step whose graphs are NOT captured in advance may not be the target of a
   flip OR of a pre-stage: hard error (``KvLadderError``), never a silent
   fallback to a different rung.
5. PRIORITY: protected sessions stay on the fast rung; an ascent lists the
   UNPROTECTED sessions as affected. If every session is protected the
   ascent is blocked with a reason rather than breaking the protection.
6. ``external`` steps require a long hysteresis (consecutive ascend verdicts
   >= the step's ``min_hysteresis_rounds``) before they can be a target.

THE SENSOR (``KvPressureSensor``). Water level AND trend: the decision runs
off the PROJECTED exhaustion (rounds until the pool is full, from a least-
squares slope over the occupancy history), not off the momentary value.
Hysteresis is asymmetric by construction and validated as such: ascent is
aggressive (high mark, SHORT window -- bursting costs more than a slow
step), descent is sluggish (low mark, LONG window). There is no new
measurement path: the interface takes an occupancy time series, so the
existing scheduler / ``token_to_kv_pool`` occupancy accounting is attached
later without touching this module. In the CPU phase tests inject the series.

SPECULATIVE PRE-STAGING (Ergaenzung 9b). A SECOND water mark below the flip
mark: when occupancy trends through the PRE-STAGE mark, the KV is waved to
the card that is probably about to join as a SHADOW COPY -- the old layout
stays the source of truth. If the flip happens, the shadow becomes
authoritative and only the DELTA since staging start still moves; if it does
not, discarding is FREE (no copy-back -- which is why this speculation is
cheap, unlike any migrating one). Falling back is deliberately sluggish
(``abort_stage_window``) so flapping does not stage and discard forever. The
shadow is itself a register item, class ``kv_shadow``: lowest priority on the
target card, drop-on-demand; transport rides the bus arbiter, the target
space comes from the peer budget grant.

INDEPENDENTLY SELECTABLE. The ladder and the pre-staging are two features
with two flags: ``--kv-pressure-ladder`` and ``--kv-pressure-pre-stage``.
Ladder on + staging off is a valid, first-class combination (the ladder then
flips with the step's own handover and never builds a shadow); staging
without a ladder is meaningless and is a hard error at argument time. Both
off = today's behavior, byte-identical -- nothing is constructed, no hook is
attached, no sample is taken.

COMBINABILITY (alles greift in alles). No feature exclusion is asserted here
without a named hard limit: the ladder composes with the offload register
(its cold rung graphs ARE register items, its shadow a register class), with
uneven DCP (the DCP token ratio is the ladder's own first relief rung), with
speculative decode and the K-ladder (both flip at the SAME round boundaries
under the SAME capture guard -- the guard is what makes them composable
rather than exclusive) and with the priority classes (invariant 5). The one
structural limit stated in code is invariant 4: a rung without pre-captured
graphs may be neither a flip nor a pre-stage target -- not a policy choice
but the graph-safety contract, since a captured region cannot replay a
geometry it never recorded.

WIRING POINTS (named here, built in the GPU phase -- no rebuilds):

* ``graph_rungs``: each step's cold, pre-captured graphs are register items
  of the EXISTING class ``graph_rungs``; ``LadderPlan.required_resident_items``
  names the ids a flip needs resident, so the flip's wave-in is an ordinary
  register wave-in, not a second mechanism.
* ``kv_shadow``: the new register class for the 9b shadow copy
  (``LadderPlan.shadow_items`` / ``discard_items``).
* The sensor is the same hook style as the 13e/#279 saturation signal --
  a zero-argument callable attached from outside, absent = no pressure.
* NEIGHBOUR PLANNER: ``OffloadRegister.on_admission_boundary`` (the Erg.-8
  session-set ladder) plans at ADMISSION boundaries, this one at PRESSURE
  boundaries. The two must never contradict each other:
  ``plan_conflicts`` names the register items both touch, and
  ``resolve_plan_priority`` gives the deterministic winner (admission --
  correctness before capacity: an arriving session must never meet a parked
  set).

Everything here is pure Python and CPU-testable; nothing imports torch.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

STEP_BASE = "base"
STEP_RELIEF = "relief"
STEP_GEOMETRY = "geometry_flip"
STEP_EXTERNAL = "external"
STEP_TYPES = (STEP_BASE, STEP_RELIEF, STEP_GEOMETRY, STEP_EXTERNAL)

#: The enforced climb order. ``base`` first, then every cheap relief, then
#: geometry inside the family, then the out-of-family (Nachtrag-14) step.
STEP_TYPE_ORDER: Dict[str, int] = {
    STEP_BASE: 0,
    STEP_RELIEF: 1,
    STEP_GEOMETRY: 2,
    STEP_EXTERNAL: 3,
}

#: Existing KV-relief features a relief step may REFERENCE. The value is the
#: user-facing flag / issue the step switches -- this module never implements
#: any of them, it only orders them into the ladder.
RELIEF_FEATURES: Dict[str, str] = {
    "admission_cap": "--max-running-requests-ceiling (#287, the floating "
    "concurrent-session limit; throttles the inflow, moves no KV)",
    "dcp_ratio": "--rank-kv-ratio (uneven-DCP KV-token ownership vector)",
    "kv_spill": "KV spill to host RAM (#134/#236 spill machinery)",
    "weightless_rank": "--weightless-kv-fastlane (#115, KV capacity without "
    "moving layers)",
    "session_offload": "--enable-kv-session-offload (FCFS session spill)",
}

#: Canonical cheapness order of the relief features (cheapest first). A table
#: generator uses it so the same rig always produces the same ladder.
#:
#: The order is a SERVICE-cost order, not an implementation-effort order
#: (user directive, #287: KV-vector flip < admission lowering < session
#: spill):
#:
#: * ``dcp_ratio`` is first because #320 card-proved that a sum-preserving
#:   KV-vector redistribution is service-NEUTRAL: 69,784 -> 431,457 tokens
#:   (6.18x) with prefill at/under the noise floor and a decode surcharge of
#:   +0.9pp, below the 2.72% floor. Re-aiming where NEW tokens land costs no
#:   served session anything.
#: * ``admission_cap`` is second: a counter update that copies nothing, but
#:   it DOES cost service -- every lowered slot is a session the server now
#:   turns away.
#: * The data movers come last: page spill (#134/#236), the weightless rank
#:   role (#115) and finally whole-session offload, which freezes a live
#:   session's tokens out of VRAM -- the most service-expensive relief and
#:   the last one before a geometry flip.
RELIEF_ORDER: Tuple[str, ...] = (
    "dcp_ratio",
    "admission_cap",
    "kv_spill",
    "weightless_rank",
    "session_offload",
)

HANDOVER_NONE = "none"
HANDOVER_NEW_TOKENS_ONLY = "new_tokens_only"
HANDOVER_BACKGROUND_MIGRATE = "background_migrate"
HANDOVER_SPILL_RELOAD = "spill_reload"
#: Erg. 9b: the ANTICIPATORY variant of ``background_migrate`` -- the shadow
#: copy was pre-staged, so the flip only moves the delta. Named as its own
#: strategy because its cost model (and its failure mode: discard rate under
#: flapping) is a different measurement than plain background migration.
HANDOVER_ANTICIPATORY_SHADOW = "anticipatory_shadow"

HANDOVER_STRATEGIES = (
    HANDOVER_NONE,
    HANDOVER_NEW_TOKENS_ONLY,
    HANDOVER_BACKGROUND_MIGRATE,
    HANDOVER_SPILL_RELOAD,
    HANDOVER_ANTICIPATORY_SHADOW,
)

#: Provenance labels of a step's expected figures -- same honesty discipline
#: as the HTCCL path dispatcher's rate profiles: a number without a source is
#: never silently treated as measured.
PROVENANCE_MEASURED = "measured"
PROVENANCE_SOLVER = "solver"
PROVENANCE_PLACEHOLDER = "placeholder"
PROVENANCES = (PROVENANCE_MEASURED, PROVENANCE_SOLVER, PROVENANCE_PLACEHOLDER)

# Plan phases (Erg. 9 + 9b).
PHASE_NONE = "none"
PHASE_PRE_STAGE = "pre_stage"
PHASE_ABORT_STAGE = "abort_stage"
PHASE_FLIP = "flip"
PHASE_DESCEND = "descend"
PLAN_PHASES = (
    PHASE_NONE,
    PHASE_PRE_STAGE,
    PHASE_ABORT_STAGE,
    PHASE_FLIP,
    PHASE_DESCEND,
)

# Sensor verdicts.
VERDICT_ASCEND = "ascend"
VERDICT_HOLD = "hold"
VERDICT_DESCEND = "descend"
STAGE_START = "pre_stage"
STAGE_HOLD = "hold"
STAGE_ABORT = "abort_stage"

#: The offload-register classes this ladder touches. Named so the
#: conflict check against the admission planner is a data question, not a
#: reading exercise.
LADDER_REGISTER_CLASSES = ("graph_rungs", "kv_shadow")

# Defaults. Ascent aggressive (high mark, short window), descent sluggish
# (low mark, long window), pre-stage below the flip mark with a long abort
# window so flapping does not stage/discard forever.
DEFAULT_ASCEND_THRESHOLD = 0.85
DEFAULT_ASCEND_WINDOW = 4
DEFAULT_DESCEND_THRESHOLD = 0.55
DEFAULT_DESCEND_WINDOW = 64
DEFAULT_PRE_STAGE_THRESHOLD = 0.70
DEFAULT_PRE_STAGE_WINDOW = 3
DEFAULT_ABORT_STAGE_WINDOW = 32
DEFAULT_HORIZON_ROUNDS = 32
#: Out-of-family steps run on a seconds time constant (warm standby +
#: handover); the ladder demands this many CONSECUTIVE ascend verdicts before
#: one may be a target.
DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS = 512


class KvLadderError(RuntimeError):
    """A ladder request the invariants forbid outright.

    Distinct from a BLOCKED plan: a blocked plan is a normal control-flow
    answer ("stay where you are, here is why"), this is a configuration or
    caller bug (flip target without captured graphs, forced target skipping
    a relief rung).
    """


# --------------------------------------------------------------------------
# 1. The handover interface -- the one genuinely open design decision
# --------------------------------------------------------------------------


class KvHandover:
    """How the EXISTING KV gets from the old geometry's layout into the new
    one at a step change.

    Ergaenzung 9 point 3: "the expensive core is not the weight, it is the
    KV". The weights are free (down-set property: the finest cut already
    holds the coarser geometries in the same bytes), but the resident KV
    lies in the OLD token/head split. Three real options are named in the
    design, plus the 9b anticipatory variant; which one wins per step is
    DECIDED BY MEASUREMENT, not by argument. Until those measurements exist,
    every real strategy raises ``NotImplementedError`` and its docstring
    carries the questions its measurement has to answer.

    The interface is deliberately two-phase so a background strategy can
    start work before the boundary it completes at:

    * ``prepare(plan)`` -- everything that may run BEFORE the flip boundary
      (background copy, shadow staging). Must be abortable at zero cost.
    * ``execute(plan)`` -- what happens AT the boundary, inside the capture
      guard.
    * ``abort(plan)``   -- discard whatever ``prepare`` built.
    """

    strategy: str = HANDOVER_NONE
    #: Whether ``prepare`` may run ahead of the boundary (9b pre-staging).
    supports_pre_stage: bool = False

    def prepare(self, plan: LadderPlan) -> None:
        raise NotImplementedError

    def execute(self, plan: LadderPlan) -> None:
        raise NotImplementedError

    def abort(self, plan: LadderPlan) -> None:
        raise NotImplementedError


class NoHandover(KvHandover):
    """``none`` -- the step does not change the KV layout at all.

    The only strategy that is fully implemented in the CPU phase, because
    there is nothing to implement: relief steps switch an existing feature
    (DCP ratio, spill, weightless rank, session offload) and leave every
    resident token exactly where it is. Used by every ``relief`` step, and
    only by them.
    """

    strategy = HANDOVER_NONE
    supports_pre_stage = False

    def prepare(self, plan: LadderPlan) -> None:
        return None

    def execute(self, plan: LadderPlan) -> None:
        return None

    def abort(self, plan: LadderPlan) -> None:
        return None


class NewTokensOnlyHandover(KvHandover):
    """``new_tokens_only`` (design option a) -- from the chunk boundary on,
    only NEW tokens are written in the new layout; the old KV stays where it
    is and the server serves MIXED for the rest of those sessions' lives.

    Measurement questions this stub exists to answer (GPU phase):

    * How expensive is mixed serving really -- an attention step then has to
      read two layouts, so what does the extra metadata/kernel path cost per
      round at the rig's context lengths?
    * How long does the mix persist? Sessions with long remaining generation
      keep the old layout resident on the OLD cards, which is exactly the
      capacity the ascent was trying to win. Measure the fraction of the
      target capacity that is actually gained per round after the flip.
    * Does it interact with the DCP owner rule (a token's home rank) in a
      way that breaks the byte gates? The verify path must stay
      bit-identical for the old tokens.
    """

    strategy = HANDOVER_NEW_TOKENS_ONLY
    supports_pre_stage = False

    def prepare(self, plan: LadderPlan) -> None:
        raise NotImplementedError(
            "new_tokens_only handover is a named design option, not an "
            "implementation: it is decided by measurement (see docstring)."
        )

    def execute(self, plan: LadderPlan) -> None:
        raise NotImplementedError(
            "new_tokens_only handover is a named design option, not an "
            "implementation: it is decided by measurement (see docstring)."
        )

    def abort(self, plan: LadderPlan) -> None:
        raise NotImplementedError(
            "new_tokens_only handover is a named design option, not an "
            "implementation: it is decided by measurement (see docstring)."
        )


class BackgroundMigrateHandover(KvHandover):
    """``background_migrate`` (design option b) -- migrate the resident KV
    page by page in the background, using the spill machinery as transport.

    Measurement questions (GPU phase):

    * Bus budget: pages/s achievable while decode runs, against the round
      length. On this rig the bus lies idle during dense decode but is
      contended during MoE expert streaming -- measure BOTH regimes, the
      answer is probably regime-dependent rather than global.
    * Round-time cost of the migration while it runs: ms/round with and
      without a migration in flight, per rank (the slowest rank sets the
      pace).
    * Convergence: does the migration finish before the pressure that
      triggered it becomes an OOM? Measure the time-to-complete against the
      sensor's projected exhaustion at trigger time -- if it loses that
      race, the option is dead for the burst case and only useful as the
      slow-drift strategy.
    """

    strategy = HANDOVER_BACKGROUND_MIGRATE
    supports_pre_stage = False

    def prepare(self, plan: LadderPlan) -> None:
        raise NotImplementedError(
            "background_migrate handover is a named design option, not an "
            "implementation: it is decided by measurement (see docstring)."
        )

    def execute(self, plan: LadderPlan) -> None:
        raise NotImplementedError(
            "background_migrate handover is a named design option, not an "
            "implementation: it is decided by measurement (see docstring)."
        )

    def abort(self, plan: LadderPlan) -> None:
        raise NotImplementedError(
            "background_migrate handover is a named design option, not an "
            "implementation: it is decided by measurement (see docstring)."
        )


class SpillReloadHandover(KvHandover):
    """``spill_reload`` (design option c) -- spill the KV at one round
    boundary and reload it in the new layout.

    Measurement questions (GPU phase):

    * The stall: resident-KV bytes / (D2H + H2D rate) at the rig's real
      rates, i.e. how many rounds of every session freeze. This is the
      simplest option and the only one with a single number deciding it.
    * Does it beat option (a) in TOTAL served tokens over the pressure
      episode? A short hard stall can beat a long mixed-serving tax; only
      the integral over the episode settles it.
    * Host RAM headroom at the moment of the spill (the whole resident KV
      transits host RAM) -- on a swapless box that is a hard fit question,
      not a performance one.
    """

    strategy = HANDOVER_SPILL_RELOAD
    supports_pre_stage = False

    def prepare(self, plan: LadderPlan) -> None:
        raise NotImplementedError(
            "spill_reload handover is a named design option, not an "
            "implementation: it is decided by measurement (see docstring)."
        )

    def execute(self, plan: LadderPlan) -> None:
        raise NotImplementedError(
            "spill_reload handover is a named design option, not an "
            "implementation: it is decided by measurement (see docstring)."
        )

    def abort(self, plan: LadderPlan) -> None:
        raise NotImplementedError(
            "spill_reload handover is a named design option, not an "
            "implementation: it is decided by measurement (see docstring)."
        )


class AnticipatoryShadowHandover(KvHandover):
    """``anticipatory_shadow`` (Ergaenzung 9b) -- the anticipatory variant of
    ``background_migrate``: a SHADOW COPY is staged on the probable new card
    while the old layout stays authoritative. On the flip the shadow becomes
    authoritative and only the DELTA since staging start moves; if the flip
    never comes, discarding is free (no copy-back).

    This is the only handover with ``supports_pre_stage``.

    Measurement questions (GPU phase):

    * DELTA SIZE AT FLIP vs STAGING LEAD TIME: how many tokens accumulate
      between staging start and flip at the rig's decode/prefill rates? The
      whole gain is "flip moves only the delta", so the delta as a fraction
      of the resident KV IS the payoff, and it is a function of the lead
      time the sensor buys.
    * DISCARD RATE UNDER FLAPPING: how often does an occupancy series near
      the pre-stage mark stage and then abort? Free per event, but not free
      on the BUS -- measure the wasted bytes/s against what expert streaming
      and KV spill want in the same window.
    * Target-card cost: the shadow is drop-on-demand and lowest priority,
      but it occupies VRAM that the target card's own posts would otherwise
      use. Measure whether the shadow's presence itself moves the target
      card's pressure -- a staging that triggers the next step is a loop.
    * Which pre-stage mark maximises "flip found a warm shadow" minus
      "staged for nothing"? That is the flag default, and it cannot be
      argued, only swept.
    """

    strategy = HANDOVER_ANTICIPATORY_SHADOW
    supports_pre_stage = True

    def prepare(self, plan: LadderPlan) -> None:
        raise NotImplementedError(
            "anticipatory_shadow handover is a named design option, not an "
            "implementation: it is decided by measurement (see docstring)."
        )

    def execute(self, plan: LadderPlan) -> None:
        raise NotImplementedError(
            "anticipatory_shadow handover is a named design option, not an "
            "implementation: it is decided by measurement (see docstring)."
        )

    def abort(self, plan: LadderPlan) -> None:
        raise NotImplementedError(
            "anticipatory_shadow handover is a named design option, not an "
            "implementation: it is decided by measurement (see docstring)."
        )


_HANDOVERS: Dict[str, Type[KvHandover]] = {
    HANDOVER_NONE: NoHandover,
    HANDOVER_NEW_TOKENS_ONLY: NewTokensOnlyHandover,
    HANDOVER_BACKGROUND_MIGRATE: BackgroundMigrateHandover,
    HANDOVER_SPILL_RELOAD: SpillReloadHandover,
    HANDOVER_ANTICIPATORY_SHADOW: AnticipatoryShadowHandover,
}


def get_handover(strategy: str) -> KvHandover:
    """Instantiate the handover for ``strategy``. Unknown = hard error."""
    if strategy not in _HANDOVERS:
        raise ValueError(
            f"unknown KV handover strategy {strategy!r}; known: "
            f"{', '.join(HANDOVER_STRATEGIES)}."
        )
    return _HANDOVERS[strategy]()


def handover_supports_pre_stage(strategy: str) -> bool:
    """Whether ``strategy`` can be pre-staged ahead of the boundary (9b)."""
    if strategy not in _HANDOVERS:
        raise ValueError(
            f"unknown KV handover strategy {strategy!r}; known: "
            f"{', '.join(HANDOVER_STRATEGIES)}."
        )
    return bool(_HANDOVERS[strategy].supports_pre_stage)


# --------------------------------------------------------------------------
# 1b. Depth- and format-aware operating points (user directive, #287)
# --------------------------------------------------------------------------

#: Execution phases an operating point is keyed on. Distinct namespace from
#: the PLAN phases above (``PHASE_FLIP`` etc.): these are the model's
#: prefill/decode phases, not ladder transitions.
OP_PHASE_PREFILL = "prefill"
OP_PHASE_DECODE = "decode"
OP_PHASES = (OP_PHASE_PREFILL, OP_PHASE_DECODE)


@dataclass(frozen=True)
class OperatingPoint:
    """One (phase x depth/fill) optimum of a rung.

    The staircase is DEPTH- and FORMAT-aware (user directive): per execution
    phase AND per context depth / fill level each rung carries its own
    optimum for the layer split and the KV token vector -- "also in
    between", not only at the poles. Data basis: the #296 extrema and the
    #320 depth measurement (sum-preserving KV-vector redistribution,
    2,11,10 = 6.18x at the prefill pole); the format factors enter through
    the per-(rank, family) scores of #324, read from the planner profile
    and never from a hardcoded arch table.

    ``depth_fraction`` is the normalized context depth / pool fill in
    [0, 1] this point was solved at. ``layer_split`` is the rung's weight
    geometry (Stage 1 never reshards, so within one rung it is constant and
    recorded for honesty); ``kv_vector`` is the KV token-ownership optimum
    at this operating condition.
    """

    phase: str
    depth_fraction: float
    layer_split: Tuple[int, ...]
    kv_vector: Tuple[int, ...]
    provenance: str = PROVENANCE_PLACEHOLDER
    source: str = ""

    def __post_init__(self):
        if self.phase not in OP_PHASES:
            raise ValueError(
                f"unknown operating phase {self.phase!r}; known: "
                f"{', '.join(OP_PHASES)}."
            )
        if not 0.0 <= self.depth_fraction <= 1.0:
            raise ValueError(
                f"depth_fraction must be within [0, 1], got {self.depth_fraction}"
            )
        if not self.kv_vector or any(int(v) <= 0 for v in self.kv_vector):
            raise ValueError(
                f"kv_vector must be non-empty positive integers, got "
                f"{self.kv_vector}"
            )
        if not self.layer_split or any(int(v) <= 0 for v in self.layer_split):
            raise ValueError(
                f"layer_split must be non-empty positive integers, got "
                f"{self.layer_split}"
            )
        if len(self.layer_split) != len(self.kv_vector):
            raise ValueError(
                f"layer_split has {len(self.layer_split)} ranks but kv_vector "
                f"{len(self.kv_vector)}; one entry per rank."
            )
        if self.provenance not in PROVENANCES:
            raise ValueError(
                f"unknown provenance {self.provenance!r}; known: "
                f"{', '.join(PROVENANCES)}."
            )


class StageOperatingGrid:
    """The (phase x depth) grid of one rung, with a DETERMINISTIC lookup.

    Selection is pure floor-bin selection over the sorted depth bins of a
    phase: the point with the largest ``depth_fraction <= requested`` wins,
    below the lowest bin the lowest bin wins. No interpolation of the
    integer vectors -- an interpolated vector would be a number nobody
    solved for; "in between" is expressed by SOLVING intermediate bins into
    the grid, not by inventing vectors at lookup time.
    """

    def __init__(self, points: Sequence[OperatingPoint]):
        points = tuple(points)
        if not points:
            raise ValueError(
                "an operating grid needs at least one point; 'no grid' is "
                "expressed by operating_grid=None on the step."
            )
        by_phase: Dict[str, List[OperatingPoint]] = {}
        for point in points:
            by_phase.setdefault(point.phase, []).append(point)
        for phase, phase_points in by_phase.items():
            fractions = [p.depth_fraction for p in phase_points]
            if len(set(fractions)) != len(fractions):
                raise ValueError(
                    f"operating grid declares duplicate depth bins for phase "
                    f"{phase!r}: {sorted(fractions)}"
                )
            phase_points.sort(key=lambda p: p.depth_fraction)
        self._by_phase = {k: tuple(v) for k, v in by_phase.items()}
        self._points = points

    @property
    def points(self) -> Tuple[OperatingPoint, ...]:
        return self._points

    @property
    def phases(self) -> Tuple[str, ...]:
        return tuple(sorted(self._by_phase))

    def select(self, phase: str, depth_fraction: float) -> OperatingPoint:
        """The operating point for ``(phase, depth_fraction)``.

        Deterministic by construction: same inputs, same point, on every
        rank -- the selection is part of the rank-uniform contract, so it
        may consume only replicated inputs (the caller's obligation) and
        must itself be a pure function (this method's obligation).
        """
        if phase not in self._by_phase:
            raise KeyError(
                f"operating grid has no phase {phase!r}; known: "
                f"{', '.join(sorted(self._by_phase))}."
            )
        if not 0.0 <= depth_fraction <= 1.0:
            raise ValueError(
                f"depth_fraction must be within [0, 1], got {depth_fraction}"
            )
        chosen = self._by_phase[phase][0]
        for point in self._by_phase[phase]:
            if point.depth_fraction <= depth_fraction:
                chosen = point
            else:
                break
        return chosen

    def describe(self) -> List[Dict[str, object]]:
        return [
            {
                "phase": p.phase,
                "depth_fraction": p.depth_fraction,
                "layer_split": list(p.layer_split),
                "kv_vector": list(p.kv_vector),
                "provenance": p.provenance,
                "source": p.source,
            }
            for p in self._points
        ]


# --------------------------------------------------------------------------
# 2. The step table
# --------------------------------------------------------------------------


def graph_rung_item_id(step_name: str) -> str:
    """Register item id (class ``graph_rungs``) of one step's cold graphs."""
    return f"kv_ladder/graph_rung/{step_name}"


def kv_shadow_item_id(step_name: str) -> str:
    """Register item id (class ``kv_shadow``) of one step's shadow copy."""
    return f"kv_ladder/kv_shadow/{step_name}"


@dataclass(frozen=True)
class LadderStep:
    """One rung of the pressure ladder.

    ``expected_kv_tokens`` / ``expected_cost_factor`` are the planner's
    forecast for this rung; both may be ``None``, and ``provenance`` says
    WHY a figure is what it is (``measured`` / ``solver`` / ``placeholder``)
    with ``source`` naming where it came from. A placeholder is never quietly
    treated as a measurement -- the same discipline as the dispatcher's rate
    profiles.

    ``expected_cost_factor`` is relative to the base rung (1.0 = base speed,
    1.30 = 30 % slower). It is a COST, so it grows as the ladder climbs.

    ``graphs_precaptured`` is the flip precondition: a rung whose graphs were
    not captured in advance may not be the target of a flip or a pre-stage
    (hard error in the plan, never a silent fallback).
    """

    name: str
    step_type: str
    relief_feature: Optional[str] = None
    geometry_key: Optional[str] = None
    expected_kv_tokens: Optional[int] = None
    expected_cost_factor: Optional[float] = None
    graphs_precaptured: bool = True
    handover: str = HANDOVER_NONE
    min_hysteresis_rounds: int = 0
    provenance: str = PROVENANCE_PLACEHOLDER
    source: str = ""
    graph_rung_items: Tuple[str, ...] = ()
    shadow_items: Tuple[str, ...] = ()
    #: Depth/format-aware (phase x depth) optima of this rung (user
    #: directive; solved by the planner from the #324 per-(rank, family)
    #: scores). ``None`` = no grid solved -- the rung still works, it just
    #: cannot name a depth-specific KV vector.
    operating_grid: Optional[StageOperatingGrid] = None

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("a ladder step needs a non-empty name")
        if self.step_type not in STEP_TYPES:
            raise ValueError(
                f"unknown ladder step type {self.step_type!r} on step "
                f"{self.name!r}; known: {', '.join(STEP_TYPES)}."
            )
        if self.handover not in HANDOVER_STRATEGIES:
            raise ValueError(
                f"unknown KV handover strategy {self.handover!r} on step "
                f"{self.name!r}; known: {', '.join(HANDOVER_STRATEGIES)}."
            )
        if self.provenance not in PROVENANCES:
            raise ValueError(
                f"unknown provenance {self.provenance!r} on step "
                f"{self.name!r}; known: {', '.join(PROVENANCES)}."
            )
        if self.min_hysteresis_rounds < 0:
            raise ValueError(
                f"min_hysteresis_rounds must be >= 0 on step {self.name!r}, "
                f"got {self.min_hysteresis_rounds}"
            )
        if self.expected_kv_tokens is not None and self.expected_kv_tokens < 0:
            raise ValueError(
                f"expected_kv_tokens must be >= 0 on step {self.name!r}, "
                f"got {self.expected_kv_tokens}"
            )
        if self.expected_cost_factor is not None and self.expected_cost_factor <= 0:
            raise ValueError(
                f"expected_cost_factor must be > 0 on step {self.name!r}, "
                f"got {self.expected_cost_factor}"
            )
        if self.step_type == STEP_RELIEF:
            if self.relief_feature not in RELIEF_FEATURES:
                raise ValueError(
                    f"relief step {self.name!r} references unknown relief "
                    f"feature {self.relief_feature!r}; the ladder only "
                    f"ORDERS existing features, it does not invent them. "
                    f"Known: {', '.join(sorted(RELIEF_FEATURES))}."
                )
            if self.handover != HANDOVER_NONE:
                raise ValueError(
                    f"relief step {self.name!r} declares handover "
                    f"{self.handover!r}; a relief step switches an existing "
                    f"feature and does NOT change the KV layout, so its only "
                    f"valid handover is 'none'."
                )
        else:
            if self.relief_feature is not None:
                raise ValueError(
                    f"step {self.name!r} of type {self.step_type!r} carries a "
                    f"relief_feature; only 'relief' steps reference one."
                )
        if self.step_type in (STEP_GEOMETRY, STEP_EXTERNAL):
            if self.handover == HANDOVER_NONE:
                raise ValueError(
                    f"step {self.name!r} of type {self.step_type!r} declares "
                    f"handover 'none'; a geometry/external step moves the KV "
                    f"into another layout and needs one of "
                    f"{', '.join(s for s in HANDOVER_STRATEGIES if s != HANDOVER_NONE)}."
                )
        if self.step_type == STEP_BASE:
            if self.handover != HANDOVER_NONE:
                raise ValueError(
                    f"base step {self.name!r} declares handover "
                    f"{self.handover!r}; the base rung is where the server "
                    f"already runs, nothing is handed over."
                )
            if not self.graphs_precaptured:
                raise ValueError(
                    f"base step {self.name!r} declares graphs_precaptured="
                    f"False; the rung the server runs on has its graphs."
                )

    @property
    def graph_items(self) -> Tuple[str, ...]:
        """Register items (class ``graph_rungs``) this rung's graphs live in.
        Defaults to the derived id so a table need not spell it out."""
        return self.graph_rung_items or (graph_rung_item_id(self.name),)

    @property
    def shadow_item_ids(self) -> Tuple[str, ...]:
        """Register items (class ``kv_shadow``) a pre-stage for this rung
        would create on the target card."""
        return self.shadow_items or (kv_shadow_item_id(self.name),)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "type": self.step_type,
            "relief_feature": self.relief_feature,
            "geometry_key": self.geometry_key,
            "expected_kv_tokens": self.expected_kv_tokens,
            "expected_cost_factor": self.expected_cost_factor,
            "graphs_precaptured": self.graphs_precaptured,
            "handover": self.handover,
            "min_hysteresis_rounds": self.min_hysteresis_rounds,
            "provenance": self.provenance,
            "source": self.source,
            "operating_grid": (
                self.operating_grid.describe()
                if self.operating_grid is not None
                else None
            ),
        }


class PressureLadder:
    """The ordered step table, validated at construction.

    Nonsense ladders are hard errors here rather than surprises at the first
    boundary:

    * empty, or rung 0 not the single ``base`` step;
    * type order violated (relief after geometry, geometry after external) --
      this is invariant 1, and it is what makes "cheap relief before
      expensive geometry" structural rather than a convention;
    * duplicate step names;
    * KNOWN expected capacities not non-decreasing (a ladder that climbs
      towards capacity may not lose capacity on the way up; ``None`` entries
      are skipped, not guessed);
    * KNOWN expected costs not non-decreasing (each rung is at least as
      expensive as the one below -- otherwise it would be the base);
    * an ``external`` step with a hysteresis below the configured minimum.
    """

    def __init__(
        self,
        steps: Sequence[LadderStep],
        external_min_hysteresis_rounds: int = DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS,
    ):
        steps = tuple(steps)
        if not steps:
            raise ValueError(
                "a pressure ladder needs at least the base rung; an empty "
                "table is 'ladder off', which is expressed by not "
                "configuring one."
            )
        if external_min_hysteresis_rounds < 1:
            raise ValueError(
                f"external_min_hysteresis_rounds must be >= 1, got "
                f"{external_min_hysteresis_rounds}"
            )
        if steps[0].step_type != STEP_BASE:
            raise ValueError(
                f"rung 0 of a pressure ladder must be the 'base' step (the "
                f"performance-optimal state the server boots in), got "
                f"{steps[0].step_type!r} ({steps[0].name!r})."
            )
        names: Dict[str, int] = {}
        prev_order = STEP_TYPE_ORDER[STEP_BASE]
        prev_tokens: Optional[int] = None
        prev_cost: Optional[float] = None
        for index, step in enumerate(steps):
            if step.name in names:
                raise ValueError(
                    f"duplicate ladder step name {step.name!r} (rungs "
                    f"{names[step.name]} and {index})."
                )
            names[step.name] = index
            if index > 0 and step.step_type == STEP_BASE:
                raise ValueError(
                    f"step {step.name!r} at rung {index} is a second 'base' "
                    f"step; exactly one base rung (index 0) exists."
                )
            order = STEP_TYPE_ORDER[step.step_type]
            if order < prev_order:
                raise ValueError(
                    f"ladder step {step.name!r} (rung {index}, type "
                    f"{step.step_type!r}) comes after a "
                    f"{_type_named(prev_order)!r} step: the climb order is "
                    f"base -> relief -> geometry_flip -> external. Cheap KV "
                    f"relief is ALWAYS exhausted before a geometry flip; "
                    f"this order is enforced, not a convention."
                )
            prev_order = order
            if step.step_type == STEP_EXTERNAL:
                if step.min_hysteresis_rounds < external_min_hysteresis_rounds:
                    raise ValueError(
                        f"external step {step.name!r} declares "
                        f"min_hysteresis_rounds="
                        f"{step.min_hysteresis_rounds}, below the required "
                        f"{external_min_hysteresis_rounds}: an out-of-family "
                        f"step (warm standby + handover, seconds time "
                        f"constant) is the LAST rung and needs a long "
                        f"hysteresis."
                    )
            if step.expected_kv_tokens is not None:
                if prev_tokens is not None and step.expected_kv_tokens < prev_tokens:
                    raise ValueError(
                        f"ladder step {step.name!r} (rung {index}) expects "
                        f"{step.expected_kv_tokens} KV tokens, LESS than the "
                        f"{prev_tokens} of a lower rung: a ladder towards "
                        f"capacity may not lose capacity on the way up."
                    )
                prev_tokens = step.expected_kv_tokens
            if step.expected_cost_factor is not None:
                if prev_cost is not None and step.expected_cost_factor < prev_cost:
                    raise ValueError(
                        f"ladder step {step.name!r} (rung {index}) expects "
                        f"cost factor {step.expected_cost_factor}, CHEAPER "
                        f"than the {prev_cost} of a lower rung: a cheaper "
                        f"rung with more capacity would be the base, not a "
                        f"step up."
                    )
                prev_cost = step.expected_cost_factor
        self._steps = steps
        self._by_name = names
        self.external_min_hysteresis_rounds = int(external_min_hysteresis_rounds)

    # -- access -------------------------------------------------------------
    @property
    def steps(self) -> Tuple[LadderStep, ...]:
        return self._steps

    def __len__(self) -> int:
        return len(self._steps)

    def __getitem__(self, index: int) -> LadderStep:
        return self._steps[index]

    def index_of(self, name: str) -> int:
        if name not in self._by_name:
            raise KeyError(
                f"no ladder step named {name!r}; known: "
                f"{', '.join(s.name for s in self._steps)}."
            )
        return self._by_name[name]

    def first_index_of_type(self, step_type: str) -> Optional[int]:
        for index, step in enumerate(self._steps):
            if step.step_type == step_type:
                return index
        return None

    def describe(self) -> List[Dict[str, object]]:
        out = []
        for index, step in enumerate(self._steps):
            entry = step.as_dict()
            entry["rung"] = index
            out.append(entry)
        return out


def _type_named(order: int) -> str:
    for name, value in STEP_TYPE_ORDER.items():
        if value == order:
            return name
    return "?"


# --------------------------------------------------------------------------
# 3. The sensor: water level + trend, asymmetric hysteresis, two marks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OccupancySample:
    """One KV occupancy observation at a round boundary.

    ``round_index`` is the monotone round counter (the trend's x axis);
    ``used_tokens`` / ``total_tokens`` are the pool's occupancy accounting.
    The sensor takes samples rather than reading a pool, so the existing
    scheduler / ``token_to_kv_pool`` accounting attaches later WITHOUT a new
    measurement path -- and tests inject a series.
    """

    round_index: int
    used_tokens: int
    total_tokens: int

    def __post_init__(self):
        if self.total_tokens <= 0:
            raise ValueError(f"total_tokens must be > 0, got {self.total_tokens}")
        if self.used_tokens < 0:
            raise ValueError(f"used_tokens must be >= 0, got {self.used_tokens}")

    @property
    def occupancy(self) -> float:
        return self.used_tokens / self.total_tokens


@dataclass(frozen=True)
class PressureReading:
    """What the sensor says at one boundary. Purely derived from the series,
    hence deterministic and reproducible in tests."""

    samples: int
    occupancy: Optional[float]
    trend_tokens_per_round: Optional[float]
    rounds_to_exhaustion: Optional[float]
    verdict: str
    stage_verdict: str
    reason: str


def _slope(samples: Sequence[OccupancySample]) -> Optional[float]:
    """Least-squares slope of used_tokens over round_index (tokens/round).

    Deterministic closed form -- no iteration, no randomness, no clock. Two
    identical x values (a caller replaying a round) give a zero variance and
    therefore ``None`` rather than an infinity.
    """
    n = len(samples)
    if n < 2:
        return None
    xs = [float(s.round_index) for s in samples]
    ys = [float(s.used_tokens) for s in samples]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) * (x - mx) for x in xs)
    if den == 0.0:
        return None
    return num / den


def _rounds_to_exhaustion(
    samples: Sequence[OccupancySample], slope: Optional[float]
) -> Optional[float]:
    if not samples or slope is None or slope <= 0.0:
        return None
    last = samples[-1]
    headroom = last.total_tokens - last.used_tokens
    if headroom <= 0:
        return 0.0
    return headroom / slope


class KvPressureSensor:
    """KV water level + TREND with asymmetric hysteresis and TWO marks.

    Marks, from low to high: ``descend_threshold`` < ``pre_stage_threshold``
    < ``ascend_threshold``. The ordering is validated, so a configuration
    that would descend while staging cannot be entered.

    Windows: ascent is aggressive (``ascend_window``, short), descent is
    sluggish (``descend_window``, long) and staging aborts sluggishly
    (``abort_stage_window`` > ``pre_stage_window``) so flapping does not
    stage and discard forever. ``descend_window > ascend_window`` and
    ``abort_stage_window > pre_stage_window`` are ENFORCED, not merely
    recommended: they are what "asymmetric" means here.

    Trend: the projected exhaustion drives the ascent as much as the level
    does -- ``occupancy >= mark for the whole window`` OR ``projected
    exhaustion <= horizon`` triggers. The pre-stage mark uses a LONGER
    horizon (default 2x) because its whole point is to act earlier.
    """

    def __init__(
        self,
        ascend_threshold: float = DEFAULT_ASCEND_THRESHOLD,
        ascend_window: int = DEFAULT_ASCEND_WINDOW,
        descend_threshold: float = DEFAULT_DESCEND_THRESHOLD,
        descend_window: int = DEFAULT_DESCEND_WINDOW,
        pre_stage_threshold: float = DEFAULT_PRE_STAGE_THRESHOLD,
        pre_stage_window: int = DEFAULT_PRE_STAGE_WINDOW,
        abort_stage_window: int = DEFAULT_ABORT_STAGE_WINDOW,
        horizon_rounds: int = DEFAULT_HORIZON_ROUNDS,
        pre_stage_horizon_rounds: Optional[int] = None,
    ):
        for label, value in (
            ("ascend_threshold", ascend_threshold),
            ("descend_threshold", descend_threshold),
            ("pre_stage_threshold", pre_stage_threshold),
        ):
            if not (0.0 < value <= 1.0):
                raise ValueError(f"{label} must be within (0, 1], got {value}")
        if not descend_threshold < pre_stage_threshold < ascend_threshold:
            raise ValueError(
                f"the marks must be ordered descend < pre_stage < ascend, "
                f"got descend={descend_threshold}, "
                f"pre_stage={pre_stage_threshold}, ascend={ascend_threshold}; "
                f"otherwise the ladder would descend while it is staging."
            )
        if ascend_window < 1 or pre_stage_window < 1:
            raise ValueError("ascend_window and pre_stage_window must be >= 1")
        if descend_window <= ascend_window:
            raise ValueError(
                f"descend_window ({descend_window}) must be LONGER than "
                f"ascend_window ({ascend_window}): ascent is aggressive "
                f"(bursting costs more than a slow rung), descent is "
                f"sluggish. The asymmetry is the contract."
            )
        if abort_stage_window <= pre_stage_window:
            raise ValueError(
                f"abort_stage_window ({abort_stage_window}) must be LONGER "
                f"than pre_stage_window ({pre_stage_window}): aborting a "
                f"shadow is sluggish on purpose, so flapping around the "
                f"pre-stage mark does not stage and discard forever."
            )
        if horizon_rounds < 1:
            raise ValueError(f"horizon_rounds must be >= 1, got {horizon_rounds}")
        if pre_stage_horizon_rounds is None:
            pre_stage_horizon_rounds = 2 * horizon_rounds
        if pre_stage_horizon_rounds < horizon_rounds:
            raise ValueError(
                f"pre_stage_horizon_rounds ({pre_stage_horizon_rounds}) must "
                f"be >= horizon_rounds ({horizon_rounds}): pre-staging acts "
                f"EARLIER than the flip, so it looks further ahead."
            )
        self.ascend_threshold = float(ascend_threshold)
        self.ascend_window = int(ascend_window)
        self.descend_threshold = float(descend_threshold)
        self.descend_window = int(descend_window)
        self.pre_stage_threshold = float(pre_stage_threshold)
        self.pre_stage_window = int(pre_stage_window)
        self.abort_stage_window = int(abort_stage_window)
        self.horizon_rounds = int(horizon_rounds)
        self.pre_stage_horizon_rounds = int(pre_stage_horizon_rounds)
        self._history: List[OccupancySample] = []
        self._capacity = max(
            self.descend_window, self.abort_stage_window, self.ascend_window
        )

    # -- ingestion ----------------------------------------------------------
    def observe(self, sample: OccupancySample) -> None:
        self._history.append(sample)
        if len(self._history) > self._capacity:
            del self._history[: len(self._history) - self._capacity]

    def observe_series(self, samples: Sequence[OccupancySample]) -> None:
        for sample in samples:
            self.observe(sample)

    def reset(self) -> None:
        self._history.clear()

    @property
    def history(self) -> Tuple[OccupancySample, ...]:
        return tuple(self._history)

    # -- reading ------------------------------------------------------------
    def reading(self) -> PressureReading:
        """Verdict + stage verdict from the current series. No side effects,
        so a caller may read twice and get the same answer."""
        hist = self._history
        if not hist:
            return PressureReading(
                samples=0,
                occupancy=None,
                trend_tokens_per_round=None,
                rounds_to_exhaustion=None,
                verdict=VERDICT_HOLD,
                stage_verdict=STAGE_HOLD,
                reason="no occupancy samples yet",
            )
        occupancy = hist[-1].occupancy
        ascend_win = hist[-self.ascend_window :]
        slope = _slope(ascend_win)
        r2e = _rounds_to_exhaustion(ascend_win, slope)

        verdict = VERDICT_HOLD
        reason = "occupancy between the marks"
        if len(hist) >= self.ascend_window:
            level_hit = all(s.occupancy >= self.ascend_threshold for s in ascend_win)
            trend_hit = r2e is not None and r2e <= self.horizon_rounds
            if level_hit or trend_hit:
                verdict = VERDICT_ASCEND
                reason = (
                    f"ascend: level>={self.ascend_threshold:.3f} over "
                    f"{self.ascend_window} rounds={level_hit}, projected "
                    f"exhaustion<={self.horizon_rounds} rounds={trend_hit}"
                )
        if verdict == VERDICT_HOLD and len(hist) >= self.descend_window:
            descend_win = hist[-self.descend_window :]
            descend_slope = _slope(descend_win)
            if all(s.occupancy <= self.descend_threshold for s in descend_win) and (
                descend_slope is None or descend_slope <= 0.0
            ):
                verdict = VERDICT_DESCEND
                reason = (
                    f"descend: level<={self.descend_threshold:.3f} over "
                    f"{self.descend_window} rounds and non-rising trend"
                )

        stage_verdict, stage_reason = self._stage_verdict(hist, slope)
        return PressureReading(
            samples=len(hist),
            occupancy=occupancy,
            trend_tokens_per_round=slope,
            rounds_to_exhaustion=r2e,
            verdict=verdict,
            stage_verdict=stage_verdict,
            reason=f"{reason}; {stage_reason}",
        )

    def _stage_verdict(
        self, hist: Sequence[OccupancySample], slope: Optional[float]
    ) -> Tuple[str, str]:
        """Erg. 9b: the second water mark. Start staging when the trend goes
        through the pre-stage mark; abort only after a LONG stretch back
        below it with a non-rising trend."""
        if len(hist) >= self.pre_stage_window:
            win = hist[-self.pre_stage_window :]
            win_slope = _slope(win)
            win_r2e = _rounds_to_exhaustion(win, win_slope)
            level_hit = all(s.occupancy >= self.pre_stage_threshold for s in win)
            trend_hit = win_r2e is not None and win_r2e <= self.pre_stage_horizon_rounds
            if level_hit or trend_hit:
                return (
                    STAGE_START,
                    f"pre-stage: level>={self.pre_stage_threshold:.3f} over "
                    f"{self.pre_stage_window} rounds={level_hit}, projected "
                    f"exhaustion<={self.pre_stage_horizon_rounds} "
                    f"rounds={trend_hit}",
                )
        if len(hist) >= self.abort_stage_window:
            win = hist[-self.abort_stage_window :]
            win_slope = _slope(win)
            if all(s.occupancy < self.pre_stage_threshold for s in win) and (
                win_slope is None or win_slope <= 0.0
            ):
                return (
                    STAGE_ABORT,
                    f"abort staging: level<{self.pre_stage_threshold:.3f} "
                    f"over {self.abort_stage_window} rounds and non-rising "
                    f"trend",
                )
        return (STAGE_HOLD, "staging unchanged")


# --------------------------------------------------------------------------
# 4. The plan + the flip contract
# --------------------------------------------------------------------------


@dataclass
class LadderPlan:
    """What the pressure boundary WOULD do. Nothing moves in the CPU phase --
    the same no-op planner pattern as ``on_phase_boundary`` /
    ``on_admission_boundary``.

    ``phase`` is the plan type (Erg. 9b): ``none`` (status quo),
    ``pre_stage`` (build the shadow copy), ``abort_stage`` (discard it --
    free, the old layout never stopped being authoritative), ``flip``
    (change the rung) or ``descend``.

    ``delta_only`` is True exactly when a flip finds a warm shadow for its
    target: then only the delta since staging start still moves and the
    handover is reported as ``anticipatory_shadow``.
    """

    current_rung: int
    target_rung: int
    phase: str = PHASE_NONE
    handover: str = HANDOVER_NONE
    reason: str = ""
    blocked: Optional[str] = None
    delta_only: bool = False
    affected_sessions: List[str] = field(default_factory=list)
    protected_sessions: List[str] = field(default_factory=list)
    required_resident_items: List[str] = field(default_factory=list)
    shadow_items: List[str] = field(default_factory=list)
    discard_items: List[str] = field(default_factory=list)
    reading: Optional[PressureReading] = None

    @property
    def is_noop(self) -> bool:
        return self.phase == PHASE_NONE

    def touched_items(self) -> Tuple[str, ...]:
        """Every offload-register item id this plan would touch. The input of
        the conflict check against the admission planner."""
        return tuple(
            sorted(
                set(self.required_resident_items)
                | set(self.shadow_items)
                | set(self.discard_items)
            )
        )


class KvPressureLadder:
    """The flip contract: sensor + table -> ``LadderPlan``. Plans only.

    Thread-safe (one lock), because the boundary may be reached from a lane
    thread while an adapter reports occupancy from another.
    """

    def __init__(
        self,
        table: PressureLadder,
        sensor: Optional[KvPressureSensor] = None,
        protected_sessions: Sequence[str] = (),
        pre_stage_enabled: bool = False,
    ):
        self._table = table
        self._sensor = sensor or KvPressureSensor()
        self._lock = threading.Lock()
        self._current = 0
        self._staged_target: Optional[int] = None
        self._ascend_streak = 0
        self._capture_active = False
        self._protected = set(protected_sessions)
        # Erg. 9b, independently selectable: the ladder works without any
        # speculation. With staging off, no shadow is ever planned, no
        # ``kv_shadow`` item is ever named and a flip uses the step's own
        # handover -- the ladder is complete on its own.
        self._pre_stage_enabled = bool(pre_stage_enabled)
        self.plans = 0

    # -- state --------------------------------------------------------------
    @property
    def table(self) -> PressureLadder:
        return self._table

    @property
    def sensor(self) -> KvPressureSensor:
        return self._sensor

    @property
    def current_rung(self) -> int:
        return self._current

    @property
    def current_step(self) -> LadderStep:
        return self._table[self._current]

    @property
    def staged_target(self) -> Optional[int]:
        """The rung a shadow copy is currently staged for (9b), or None."""
        return self._staged_target

    def operating_point(
        self, phase: str, depth_fraction: float
    ) -> Optional[OperatingPoint]:
        """The CURRENT rung's depth/format-aware optimum at ``(phase,
        depth_fraction)``, or ``None`` when the rung carries no grid.

        Deterministic stage-detail selection (user directive): the answer is
        a pure function of (current rung, phase, depth) -- the caller feeds
        replicated inputs, so every rank names the same vector.
        """
        with self._lock:
            grid = self._table[self._current].operating_grid
        if grid is None:
            return None
        return grid.select(phase, depth_fraction)

    @property
    def pre_stage_enabled(self) -> bool:
        """Whether speculative pre-staging (Erg. 9b) is switched on. Off is a
        valid, first-class ladder configuration."""
        return self._pre_stage_enabled

    @property
    def capture_active(self) -> bool:
        return self._capture_active

    def protect(self, session_id: str) -> None:
        with self._lock:
            self._protected.add(session_id)

    def unprotect(self, session_id: str) -> None:
        with self._lock:
            self._protected.discard(session_id)

    @property
    def protected_sessions(self) -> Tuple[str, ...]:
        return tuple(sorted(self._protected))

    # -- capture guard (same contract as the HTCCL path dispatcher) ---------
    def begin_capture(self) -> None:
        with self._lock:
            self._capture_active = True

    def end_capture(self) -> None:
        """Capture end is itself a boundary; the next pressure boundary plans
        outside any captured region."""
        with self._lock:
            self._capture_active = False

    # -- the boundary -------------------------------------------------------
    def on_pressure_boundary(
        self,
        occupancy_series: Optional[Sequence[OccupancySample]] = None,
        *,
        at_round_boundary: bool = True,
        sessions: Sequence[str] = (),
        force_target: Optional[int] = None,
    ) -> LadderPlan:
        """Plan one pressure boundary.

        ``occupancy_series`` is fed to the sensor first when given (the CPU
        phase's injected series; the GPU phase observes per round and calls
        this with ``None``).

        ``force_target`` exists for the runtime paths that want a specific
        rung. It does NOT bypass the invariants: a target that skips a relief
        rung raises ``KvLadderError`` (invariant 2), as does any target whose
        graphs are not pre-captured (invariant 4).
        """
        with self._lock:
            self.plans += 1
            if occupancy_series:
                for sample in occupancy_series:
                    self._sensor.observe(sample)
            plan = LadderPlan(current_rung=self._current, target_rung=self._current)
            # Invariant 3: only at round boundaries, never inside a capture.
            if self._capture_active:
                plan.blocked = (
                    "capture active: a flip inside a captured region would "
                    "let the replay run a different geometry than it "
                    "recorded"
                )
                plan.reason = plan.blocked
                return plan
            if not at_round_boundary:
                plan.blocked = (
                    "not at a round boundary: ladder flips happen only between rounds"
                )
                plan.reason = plan.blocked
                return plan

            reading = self._sensor.reading()
            plan.reading = reading
            if reading.verdict == VERDICT_ASCEND:
                self._ascend_streak += 1
            else:
                self._ascend_streak = 0

            unprotected = [s for s in sessions if s not in self._protected]
            protected = [s for s in sessions if s in self._protected]
            plan.protected_sessions = protected

            if force_target is not None:
                return self._plan_forced(plan, force_target, unprotected)

            if reading.verdict == VERDICT_ASCEND:
                return self._plan_ascend(plan, unprotected)
            if self._pre_stage_enabled:
                if reading.stage_verdict == STAGE_START:
                    return self._plan_pre_stage(plan)
                if reading.stage_verdict == STAGE_ABORT and (
                    self._staged_target is not None
                ):
                    return self._plan_abort_stage(plan)
            if reading.verdict == VERDICT_DESCEND:
                return self._plan_descend(plan)
            plan.reason = reading.reason
            return plan

    # -- plan builders (all under the lock) ---------------------------------
    def _check_target(self, target: int) -> LadderStep:
        """Invariants 2 and 4 -- hard errors, never a silent fallback."""
        if target < 0 or target >= len(self._table):
            raise KvLadderError(
                f"ladder target rung {target} is outside the table "
                f"(0..{len(self._table) - 1})."
            )
        step = self._table[target]
        # Invariant 2: never jump over a relief rung into a geometry flip.
        skipped_relief = [
            s.name
            for s in self._table.steps[self._current + 1 : target]
            if s.step_type == STEP_RELIEF
        ]
        if skipped_relief and step.step_type != STEP_RELIEF:
            raise KvLadderError(
                f"ladder target {step.name!r} (rung {target}, type "
                f"{step.step_type!r}) would skip the cheaper relief rung(s) "
                f"{skipped_relief} above the current rung {self._current}: "
                f"cheap KV relief is exhausted BEFORE a geometry flip. "
                f"Ascend one rung at a time."
            )
        # Invariant 4: no flip (and no pre-stage) onto uncaptured graphs.
        if not step.graphs_precaptured:
            raise KvLadderError(
                f"ladder step {step.name!r} (rung {target}) has no "
                f"pre-captured graphs and therefore cannot be the target of "
                f"a flip or a pre-stage. Capture its graphs at boot or drop "
                f"the rung from the table -- there is no silent fallback to "
                f"another rung."
            )
        return step

    def _plan_forced(
        self, plan: LadderPlan, target: int, unprotected: List[str]
    ) -> LadderPlan:
        step = self._check_target(target)
        if target == self._current:
            plan.reason = f"already on rung {target} ({step.name!r})"
            return plan
        if target < self._current:
            plan.phase = PHASE_DESCEND
            plan.target_rung = target
            plan.handover = self._table[target].handover
            plan.required_resident_items = list(self._table[target].graph_items)
            plan.reason = f"forced descend to rung {target} ({step.name!r})"
            return plan
        return self._finish_ascend(plan, target, step, unprotected, forced=True)

    def _plan_ascend(self, plan: LadderPlan, unprotected: List[str]) -> LadderPlan:
        target = self._current + 1
        if target >= len(self._table):
            plan.blocked = (
                f"already on the top rung {self._current} "
                f"({self.current_step.name!r}): the ladder has no further "
                f"capacity to climb to"
            )
            plan.reason = plan.blocked
            return plan
        step = self._check_target(target)
        return self._finish_ascend(plan, target, step, unprotected, forced=False)

    def _finish_ascend(
        self,
        plan: LadderPlan,
        target: int,
        step: LadderStep,
        unprotected: List[str],
        forced: bool,
    ) -> LadderPlan:
        # Invariant 6: external rungs need the long hysteresis.
        if step.step_type == STEP_EXTERNAL and not forced:
            if self._ascend_streak < step.min_hysteresis_rounds:
                plan.blocked = (
                    f"external step {step.name!r} needs "
                    f"{step.min_hysteresis_rounds} consecutive ascend "
                    f"verdicts (seconds time constant, warm standby + "
                    f"handover); streak is {self._ascend_streak}"
                )
                plan.reason = plan.blocked
                return plan
        # Invariant 5: protected sessions stay on the fast rung.
        if plan.protected_sessions and not unprotected:
            plan.blocked = (
                f"every session is priority-protected "
                f"({plan.protected_sessions}); protected sessions stay on the "
                f"fast rung, so there is no unprotected load to move up"
            )
            plan.reason = plan.blocked
            return plan
        plan.phase = PHASE_FLIP
        plan.target_rung = target
        plan.affected_sessions = list(unprotected)
        plan.required_resident_items = list(step.graph_items)
        if self._pre_stage_enabled and self._staged_target == target:
            # Erg. 9b: the shadow becomes authoritative, only the delta moves.
            plan.handover = HANDOVER_ANTICIPATORY_SHADOW
            plan.delta_only = True
            plan.shadow_items = list(step.shadow_item_ids)
        else:
            plan.handover = step.handover
        plan.reason = (
            f"flip rung {self._current} -> {target} ({step.name!r}, "
            f"{step.step_type}); handover {plan.handover!r}"
            f"{' (warm shadow: delta only)' if plan.delta_only else ''}"
        )
        return plan

    def _plan_pre_stage(self, plan: LadderPlan) -> LadderPlan:
        if not self._pre_stage_enabled:
            plan.reason = "pre-stage: speculative staging is switched off"
            return plan
        target = self._current + 1
        if target >= len(self._table):
            plan.reason = "pre-stage: no rung above the current one"
            return plan
        if self._staged_target is not None:
            plan.reason = (
                f"pre-stage: a shadow for rung {self._staged_target} is already staged"
            )
            return plan
        step = self._check_target(target)
        if step.step_type == STEP_RELIEF:
            # A relief rung switches an existing feature; there is no new
            # card and no layout to pre-copy into. Staging it would be a
            # no-op that costs bus budget.
            plan.reason = (
                f"pre-stage: rung {target} ({step.name!r}) is a relief step "
                f"-- no layout change, nothing to shadow"
            )
            return plan
        if not handover_supports_pre_stage(step.handover) and (
            step.handover != HANDOVER_BACKGROUND_MIGRATE
        ):
            plan.reason = (
                f"pre-stage: handover {step.handover!r} of rung {target} "
                f"({step.name!r}) has no anticipatory variant"
            )
            return plan
        plan.phase = PHASE_PRE_STAGE
        plan.target_rung = target
        plan.handover = HANDOVER_ANTICIPATORY_SHADOW
        plan.shadow_items = list(step.shadow_item_ids)
        plan.required_resident_items = list(step.graph_items)
        plan.reason = (
            f"pre-stage a shadow copy for rung {target} ({step.name!r}): the "
            f"old layout stays authoritative, discarding is free"
        )
        return plan

    def _plan_abort_stage(self, plan: LadderPlan) -> LadderPlan:
        target = self._staged_target
        assert target is not None
        step = self._table[target]
        plan.phase = PHASE_ABORT_STAGE
        plan.target_rung = self._current
        plan.handover = HANDOVER_ANTICIPATORY_SHADOW
        plan.discard_items = list(step.shadow_item_ids)
        plan.reason = (
            f"abort the shadow staged for rung {target} ({step.name!r}): the "
            f"trend fell back below the pre-stage mark. Discarding is FREE "
            f"-- the old layout never stopped being the source of truth, so "
            f"nothing is copied back"
        )
        return plan

    def _plan_descend(self, plan: LadderPlan) -> LadderPlan:
        if self._current == 0:
            plan.reason = "descend: already on the base rung"
            return plan
        target = self._current - 1
        step = self._table[target]
        plan.phase = PHASE_DESCEND
        plan.target_rung = target
        plan.handover = self._table[self._current].handover
        plan.required_resident_items = list(step.graph_items)
        plan.reason = (
            f"descend rung {self._current} -> {target} ({step.name!r}): "
            f"pressure gone for {self._sensor.descend_window} rounds"
        )
        return plan

    # -- applying a plan ----------------------------------------------------
    def apply(self, plan: LadderPlan) -> None:
        """Record the plan's effect on the ladder state.

        Executing the plan (moving graphs, handing over KV) is the GPU phase;
        this only advances the ladder's own bookkeeping so a test -- and
        later the runtime -- can walk a whole pressure episode.
        """
        with self._lock:
            if plan.blocked is not None or plan.phase == PHASE_NONE:
                return
            if plan.phase == PHASE_FLIP:
                self._current = plan.target_rung
                self._staged_target = None
                self._ascend_streak = 0
            elif plan.phase == PHASE_DESCEND:
                self._current = plan.target_rung
                self._staged_target = None
            elif plan.phase == PHASE_PRE_STAGE:
                self._staged_target = plan.target_rung
            elif plan.phase == PHASE_ABORT_STAGE:
                self._staged_target = None


# --------------------------------------------------------------------------
# 5. Conflict freedom with the admission planner (Erg. 8 neighbour)
# --------------------------------------------------------------------------

#: Deterministic priority when the two boundary planners collide. Admission
#: wins: an arriving session must never meet a parked state set (correctness),
#: while a pressure step can wait one boundary (capacity).
PLAN_PRIORITY: Tuple[str, ...] = ("admission", "pressure")


def plan_conflicts(ladder_plan: LadderPlan, admission_plan) -> Tuple[str, ...]:
    """Register item ids BOTH boundary planners would touch.

    By construction this is empty: the pressure ladder owns ``graph_rungs``
    and ``kv_shadow`` items, the admission ladder owns ``gdn_state_sets``
    items, and the id namespaces are disjoint. The function exists so that
    stays a checked property rather than an assumption -- if a future item
    class is shared, the collision shows up here instead of as a lost
    wave-in.
    """
    mine = set(ladder_plan.touched_items())
    theirs = set(getattr(admission_plan, "park_candidates", ()) or ()) | set(
        getattr(admission_plan, "wave_in_candidates", ()) or ()
    )
    return tuple(sorted(mine & theirs))


def resolve_plan_priority(ladder_plan: LadderPlan, admission_plan) -> str:
    """Which plan executes first when both are produced at the same boundary.

    ``"disjoint"`` when they touch nothing in common (the normal case),
    otherwise the deterministic winner from ``PLAN_PRIORITY``. The two
    planners may never contradict each other, and "whoever ran last wins" is
    not an answer.
    """
    if not plan_conflicts(ladder_plan, admission_plan):
        return "disjoint"
    return PLAN_PRIORITY[0]


# --------------------------------------------------------------------------
# 6. Flag plumbing
# --------------------------------------------------------------------------

LADDER_SPEC_AUTO = "auto"
_SPEC_TYPE_ALIASES = {
    "relief": STEP_RELIEF,
    "geometry": STEP_GEOMETRY,
    "geometry_flip": STEP_GEOMETRY,
    "external": STEP_EXTERNAL,
}


def parse_kv_pressure_ladder(
    spec: Optional[str], flag: str = "--kv-pressure-ladder"
) -> Union[None, str, Tuple[Tuple[str, str], ...]]:
    """Parse ``--kv-pressure-ladder``.

    * ``None`` / empty        -> ``None`` = ladder off (today's behavior).
    * ``"auto"``              -> the string ``"auto"`` = build the table from
                                 the #272 planner at boot.
    * otherwise               -> a tuple of ``(step_type, name)`` pairs, in
                                 climb order, e.g.
                                 ``relief:dcp_ratio,relief:kv_spill,geometry:tp2``.

    Hard errors at argument time (a typo fails the boot, not the first
    pressure boundary): unknown type, unknown relief feature, duplicate
    names, empty entries, and any order that puts a relief step after a
    geometry step or a geometry step after an external one.
    """
    if spec is None or not spec.strip():
        return None
    spec = spec.strip()
    if spec == LADDER_SPEC_AUTO:
        return LADDER_SPEC_AUTO
    entries: List[Tuple[str, str]] = []
    seen: Dict[str, int] = {}
    prev_order = STEP_TYPE_ORDER[STEP_BASE]
    for position, part in enumerate(spec.split(",")):
        part = part.strip()
        if not part:
            raise ValueError(
                f"{flag} contains an empty entry in {spec!r}; expected "
                f"comma-separated '<type>:<name>' entries, e.g. "
                f"'relief:dcp_ratio,geometry:tp2'."
            )
        if ":" not in part:
            raise ValueError(
                f"{flag} entry {part!r} has no '<type>:<name>' form "
                f"(spec {spec!r}); types: "
                f"{', '.join(sorted(_SPEC_TYPE_ALIASES))}."
            )
        raw_type, name = part.split(":", 1)
        raw_type = raw_type.strip()
        name = name.strip()
        if raw_type not in _SPEC_TYPE_ALIASES:
            raise ValueError(
                f"{flag} entry {part!r} has unknown step type "
                f"{raw_type!r}; known: "
                f"{', '.join(sorted(_SPEC_TYPE_ALIASES))}."
            )
        if not name:
            raise ValueError(f"{flag} entry {part!r} has an empty step name.")
        step_type = _SPEC_TYPE_ALIASES[raw_type]
        if step_type == STEP_RELIEF and name not in RELIEF_FEATURES:
            raise ValueError(
                f"{flag} relief entry {name!r} is not a known relief "
                f"feature; the ladder only ORDERS existing features. Known: "
                f"{', '.join(sorted(RELIEF_FEATURES))}."
            )
        if name in seen:
            raise ValueError(
                f"{flag} names {name!r} twice (entries {seen[name]} and "
                f"{position}); each rung appears once."
            )
        seen[name] = position
        order = STEP_TYPE_ORDER[step_type]
        if order < prev_order:
            raise ValueError(
                f"{flag} entry {part!r} comes after a "
                f"{_type_named(prev_order)!r} entry; the climb order is "
                f"relief -> geometry -> external (cheap KV relief is always "
                f"exhausted before a geometry flip)."
            )
        prev_order = order
        entries.append((step_type, name))
    return tuple(entries)


def ladder_from_spec(
    entries: Sequence[Tuple[str, str]],
    *,
    base_name: str = "base",
    external_min_hysteresis_rounds: int = DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS,
    default_geometry_handover: str = HANDOVER_BACKGROUND_MIGRATE,
    default_external_handover: str = HANDOVER_SPILL_RELOAD,
) -> PressureLadder:
    """Build a ``PressureLadder`` from a parsed explicit spec.

    Every expected figure is a PLACEHOLDER with its provenance labelled: an
    explicit spec says WHICH rungs exist, not what they are worth. The
    ``auto`` path (``planner.kv_ladder_table.build_ladder_table``) is the one
    that fills capacities in.
    """
    steps: List[LadderStep] = [
        LadderStep(
            name=base_name,
            step_type=STEP_BASE,
            provenance=PROVENANCE_PLACEHOLDER,
            source="--kv-pressure-ladder explicit spec (base rung)",
        )
    ]
    for step_type, name in entries:
        if step_type == STEP_RELIEF:
            steps.append(
                LadderStep(
                    name=name,
                    step_type=STEP_RELIEF,
                    relief_feature=name,
                    handover=HANDOVER_NONE,
                    provenance=PROVENANCE_PLACEHOLDER,
                    source=f"--kv-pressure-ladder explicit spec ({name})",
                )
            )
        elif step_type == STEP_GEOMETRY:
            steps.append(
                LadderStep(
                    name=name,
                    step_type=STEP_GEOMETRY,
                    geometry_key=name,
                    handover=default_geometry_handover,
                    provenance=PROVENANCE_PLACEHOLDER,
                    source=f"--kv-pressure-ladder explicit spec ({name})",
                )
            )
        else:
            steps.append(
                LadderStep(
                    name=name,
                    step_type=STEP_EXTERNAL,
                    handover=default_external_handover,
                    min_hysteresis_rounds=external_min_hysteresis_rounds,
                    provenance=PROVENANCE_PLACEHOLDER,
                    source=f"--kv-pressure-ladder explicit spec ({name})",
                )
            )
    return PressureLadder(
        steps, external_min_hysteresis_rounds=external_min_hysteresis_rounds
    )


def sensor_from_server_args(server_args) -> KvPressureSensor:
    """Build the sensor from the ``--kv-pressure-*`` flags."""
    return KvPressureSensor(
        ascend_threshold=float(
            getattr(
                server_args, "kv_pressure_ascend_threshold", DEFAULT_ASCEND_THRESHOLD
            )
        ),
        ascend_window=int(
            getattr(server_args, "kv_pressure_ascend_window", DEFAULT_ASCEND_WINDOW)
        ),
        descend_threshold=float(
            getattr(
                server_args,
                "kv_pressure_descend_threshold",
                DEFAULT_DESCEND_THRESHOLD,
            )
        ),
        descend_window=int(
            getattr(server_args, "kv_pressure_descend_window", DEFAULT_DESCEND_WINDOW)
        ),
        pre_stage_threshold=float(
            getattr(
                server_args,
                "kv_pressure_pre_stage_threshold",
                DEFAULT_PRE_STAGE_THRESHOLD,
            )
        ),
        pre_stage_window=int(
            getattr(
                server_args, "kv_pressure_pre_stage_window", DEFAULT_PRE_STAGE_WINDOW
            )
        ),
        abort_stage_window=int(
            getattr(
                server_args,
                "kv_pressure_abort_stage_window",
                DEFAULT_ABORT_STAGE_WINDOW,
            )
        ),
        horizon_rounds=int(
            getattr(server_args, "kv_pressure_horizon_rounds", DEFAULT_HORIZON_ROUNDS)
        ),
    )


def build_ladder_from_server_args(
    server_args, *, table_fn=None
) -> Optional[KvPressureLadder]:
    """Build the controller from the server args, or ``None`` when the flag
    is unset (= today's behavior, byte-identical: nothing is constructed, no
    hook is attached, no sample is taken).

    The two features are selected independently: ``--kv-pressure-ladder``
    switches the ladder on, ``--kv-pressure-pre-stage`` switches the Erg.-9b
    speculative shadow staging on. Ladder on + staging off is a valid
    combination; staging without a ladder is refused at argument time
    (``_handle_kv_pressure_ladder``), because there is no rung to stage for.

    ``table_fn`` is the ``auto`` path's table source (the #272 planner's
    ``build_ladder_table``, injected so this module never imports the
    planner). ``auto`` without a table source is a hard error rather than a
    silent fallback to a placeholder ladder.
    """
    spec = parse_kv_pressure_ladder(getattr(server_args, "kv_pressure_ladder", None))
    if spec is None:
        return None
    external_hysteresis = int(
        getattr(
            server_args,
            "kv_pressure_external_hysteresis_rounds",
            DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS,
        )
    )
    if spec == LADDER_SPEC_AUTO:
        if table_fn is None:
            raise ValueError(
                "--kv-pressure-ladder auto needs the planner's step-table "
                "source; none was supplied. The auto table is computed once "
                "from the rig/model profile by "
                "sglang.srt.planner.kv_ladder_table.build_ladder_table."
            )
        table = table_fn()
    else:
        assert isinstance(spec, tuple)
        table = ladder_from_spec(
            spec, external_min_hysteresis_rounds=external_hysteresis
        )
    return KvPressureLadder(
        table,
        sensor_from_server_args(server_args),
        pre_stage_enabled=bool(getattr(server_args, "kv_pressure_pre_stage", False)),
    )
