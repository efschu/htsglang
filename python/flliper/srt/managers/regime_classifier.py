# SPDX-License-Identifier: Apache-2.0
"""Regime classification and stage admission for the #363 controller.

Phase 1 of DESIGN_363_regime_controller.md: the part that can be settled
without a card. Everything here is a pure function or a small state machine
over plain integers -- no torch, no scheduler import, no actuator call, no
flag parsing. Phase 2 wires it into the scheduler loop; this module exists so
that wiring attaches something already falsifiable.

THE ONE STRUCTURAL RULE (design section 3.1). The #287 runtime states the
uniformity contract every in-loop controller in this fork inherits: replicated
inputs only, pure functions, so every rank reaches the same verdict without a
collective. Per-rank milliseconds do NOT satisfy it -- they differ per rank by
construction, which is the whole reason they are a signal. Inputs are
therefore in two tiers:

* TIER R -- replicated. Forward-mode mix, KV occupancy, batch size, queue
  composition, round index. These drive the classification, and every branch
  in this module reads only these.
* TIER L -- rank-local. Per-rank forward device ms. These may only be
  consumed AFTER reduction: quantized, packed into the consensus payload that
  already runs unconditionally at the cadence boundary, and read back as a
  spread (max - min). :class:`RegimeSample` carries the reduced summary, never
  a local reading, and the packing helpers at the bottom of this module are
  the convention that gets it there.

The classification decides WHAT KIND OF WORK is arriving. It deliberately does
not decide which rank is the brake: #264 measured that answer being acted on
directly (prefill +8.2 %, decode -13.7 %, KV capacity -47.9 %, net negative),
because 69-75 % of the window is collective floor no cut reallocation touches.
Rank timing enters as a veto and as validation, never as the primary axis.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_ABORT_PRESTAGE_WINDOW",
    "DEFAULT_ENTER_DECODE",
    "DEFAULT_ENTER_PREFILL",
    "DEFAULT_EXIT_DECODE",
    "DEFAULT_EXIT_PREFILL",
    "DEFAULT_PRESTAGE_WINDOW",
    "DEFAULT_WINDOW_ROUNDS",
    "DWELL_AMORTIZATION",
    "KV_ASCEND_MARK",
    "KV_DESCEND_MARK",
    "KV_PRE_STAGE_MARK",
    "REGIMES",
    "REGIME_DECODE_HEAVY",
    "REGIME_KV_PRESSURE",
    "REGIME_MIXED",
    "REGIME_PREFILL_HEAVY",
    "DwellGate",
    "RegimeError",
    "RegimeSample",
    "RegimeSensor",
    "Stage",
    "StageTable",
    "classify_sample",
    "clears_band",
    "min_dwell_rounds",
    "pack_proposal",
    "signal_band",
    "unpack_reduced",
]


class RegimeError(RuntimeError):
    """A configuration or a stage selection that must not proceed.

    Raised at construction wherever possible, following the #287 discipline:
    an invalid mapping fails at boot, not at the first episode.
    """


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------

#: The rig is ingesting: prefill dominates the window, or the queue holds a
#: burst that will make it dominate.
REGIME_PREFILL_HEAVY = "prefill_heavy"
#: The rig is emitting: decode dominates and nothing is waiting.
REGIME_DECODE_HEAVY = "decode_heavy"
#: The KV pool is the binding constraint. Outranks the other two -- it is the
#: CONSTRAINT regime, not a fourth peer, and under it the stage decision
#: belongs to the pressure ladder's arithmetic (design section 3.2/7.3).
REGIME_KV_PRESSURE = "kv_pressure"
#: None of the above held its window. A real answer, and on general traffic
#: the common one; a controller rarely in MIXED on mixed traffic is
#: misclassifying.
REGIME_MIXED = "mixed"

REGIMES = (
    REGIME_PREFILL_HEAVY,
    REGIME_DECODE_HEAVY,
    REGIME_KV_PRESSURE,
    REGIME_MIXED,
)

#: Integer codes for the consensus payload. A regime crosses the wire as an
#: int, never as a string.
REGIME_CODES: Dict[str, int] = {name: i for i, name in enumerate(REGIMES)}
REGIME_BY_CODE: Dict[int, str] = {i: name for name, i in REGIME_CODES.items()}


# ---------------------------------------------------------------------------
# Constants (design section 3.4 -- every one provisional, each paired with the
# experiment that replaces it; none may ship as a default before F3 runs)
# ---------------------------------------------------------------------------

#: Classification window, in scheduler rounds. Starts at the #287 descend
#: window so the two controllers share one cadence rather than beating against
#: each other. Set by F3: the shortest window whose per-signal A-vs-A band is
#: below the smallest threshold gap.
DEFAULT_WINDOW_ROUNDS = 64

#: Prefill share of the window at which PREFILL_HEAVY is entered / left. Set
#: by F3 on ``prefill_share``; the gap must exceed 2x that signal's band.
DEFAULT_ENTER_PREFILL = 0.35
DEFAULT_EXIT_PREFILL = 0.15

#: Decode share at which DECODE_HEAVY is entered / left. Same F3 treatment.
DEFAULT_ENTER_DECODE = 0.90
DEFAULT_EXIT_DECODE = 0.70

#: KV occupancy marks, INHERITED VERBATIM from the #287 sensor
#: (``model_executor/kv_pressure_ladder.py`` DEFAULT_*_THRESHOLD). Deliberately
#: not re-derived: two independently-chosen thresholds on one physical quantity
#: is how two controllers end up disagreeing about the same pool.
KV_ASCEND_MARK = 0.85
KV_PRE_STAGE_MARK = 0.70
KV_DESCEND_MARK = 0.55

#: Pre-stage / abort windows. The abort window is longer, enforced below, for
#: the reason the #287 sensor enforces it: so flapping around the mark does not
#: stage and discard forever.
DEFAULT_PRESTAGE_WINDOW = 3
DEFAULT_ABORT_PRESTAGE_WINDOW = 32

#: How many times its own cost a flip must be held for. POLICY, not a
#: measurement, and labelled as such: at 20x the flip consumes at most 5 % of
#: the interval it bought. Everything else in the dwell derivation is measured
#: (the flip's instrumented duration, the live mean round time).
DWELL_AMORTIZATION = 20


# ---------------------------------------------------------------------------
# The sample
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RegimeSample:
    """One boundary's state. Every field is REPLICATED across the TP group.

    The rank-local timing tier appears here only as ``rank_ms_spread_pct`` --
    the reduced (max - min) summary that came back from the consensus
    reduction, never a local reading. A caller that fills it from a local
    timer has broken the rule this class exists to hold; the packing helpers
    at the bottom of the module are the supported path.
    """

    #: Replicated round counter. The cadence gate, never a local verdict.
    round_index: int
    #: Rounds in the window that ran a prefill forward.
    prefill_rounds: int
    #: Rounds in the window that ran a decode forward.
    decode_rounds: int
    #: KV tokens held by live requests, and the group-agreed pool capacity --
    #: the admission limiter's own sample, identical on every rank.
    held_tokens: int
    capacity_tokens: int
    #: Requests admitted but not started, and their prompt-token mass.
    queued_reqs: int = 0
    queued_prompt_tokens: int = 0
    max_queued_prompt_tokens: int = 0
    #: Running batch size.
    running_bs: int = 0
    #: REDUCED across ranks: (max - min) / max of per-rank forward device ms,
    #: in percent. ``None`` when the reduction had nothing to report -- a
    #: graph-covered forward reports no split at all rather than a wrong zero,
    #: and that absence is carried, not filled.
    rank_ms_spread_pct: Optional[float] = None

    def __post_init__(self) -> None:
        for name in (
            "round_index",
            "prefill_rounds",
            "decode_rounds",
            "held_tokens",
            "capacity_tokens",
            "queued_reqs",
            "queued_prompt_tokens",
            "max_queued_prompt_tokens",
            "running_bs",
        ):
            if getattr(self, name) < 0:
                raise RegimeError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.rank_ms_spread_pct is not None and self.rank_ms_spread_pct < 0:
            raise RegimeError(
                f"rank_ms_spread_pct must be >= 0 or None, got "
                f"{self.rank_ms_spread_pct}"
            )

    @property
    def forward_rounds(self) -> int:
        return self.prefill_rounds + self.decode_rounds

    @property
    def prefill_share(self) -> Optional[float]:
        """Fraction of forwards that were prefill, or ``None`` on an idle
        window. An idle window is not 0 % prefill; it is no measurement."""
        total = self.forward_rounds
        return None if total == 0 else self.prefill_rounds / total

    @property
    def decode_share(self) -> Optional[float]:
        total = self.forward_rounds
        return None if total == 0 else self.decode_rounds / total

    @property
    def occupancy(self) -> Optional[float]:
        """KV fill, or ``None`` when no capacity is known."""
        if self.capacity_tokens <= 0:
            return None
        return self.held_tokens / self.capacity_tokens


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_sample(
    sample: RegimeSample,
    *,
    enter_prefill: float = DEFAULT_ENTER_PREFILL,
    enter_decode: float = DEFAULT_ENTER_DECODE,
    burst_tokens: int = 0,
    kv_ascend_mark: float = KV_ASCEND_MARK,
) -> str:
    """The instantaneous regime of one sample. Pure, tier-R only.

    Precedence is fixed and is part of the contract: KV_PRESSURE outranks
    everything, because when the pool is the constraint the stage decision is
    the pressure arithmetic's, not the regime controller's. Below it,
    PREFILL_HEAVY outranks DECODE_HEAVY -- an arriving burst is the thing with
    a deadline, and a decode-heavy reading during one is a description of the
    rounds already spent rather than of the work about to arrive.
    """
    occupancy = sample.occupancy
    if occupancy is not None and occupancy >= kv_ascend_mark:
        return REGIME_KV_PRESSURE

    share_p = sample.prefill_share
    if share_p is not None and share_p >= enter_prefill:
        return REGIME_PREFILL_HEAVY
    if burst_tokens > 0 and sample.queued_prompt_tokens >= burst_tokens:
        return REGIME_PREFILL_HEAVY

    share_d = sample.decode_share
    if share_d is not None and share_d >= enter_decode and sample.queued_reqs == 0:
        return REGIME_DECODE_HEAVY

    return REGIME_MIXED


class RegimeSensor:
    """Windowed classification with asymmetric hysteresis.

    Two properties, both enforced at construction rather than recommended,
    following ``KvPressureSensor``:

    1. ENTRY AND EXIT DIFFER. A regime is entered on one threshold and left on
       a looser one, so a signal sitting on a mark cannot enter and leave on
       alternating samples.
    2. EXIT IS SLUGGISH. ``exit_window > enter_window``. Entering a regime is
       cheap (the controller only proposes); leaving one discards a
       classification the actuator may already be acting on.

    Hysteresis is about the SIGNAL. It does not bound how often a flip may
    happen -- a load that genuinely alternates faster than a flip amortizes
    would pass every hysteresis test and still thrash. That is
    :class:`DwellGate`'s job, and the two are deliberately separate.
    """

    def __init__(
        self,
        *,
        window_rounds: int = DEFAULT_WINDOW_ROUNDS,
        enter_prefill: float = DEFAULT_ENTER_PREFILL,
        exit_prefill: float = DEFAULT_EXIT_PREFILL,
        enter_decode: float = DEFAULT_ENTER_DECODE,
        exit_decode: float = DEFAULT_EXIT_DECODE,
        enter_window: int = 2,
        exit_window: int = 4,
        kv_ascend_mark: float = KV_ASCEND_MARK,
        kv_descend_mark: float = KV_DESCEND_MARK,
        burst_tokens: int = 0,
    ):
        if window_rounds < 1:
            raise RegimeError(f"window_rounds must be >= 1, got {window_rounds}")
        for label, value in (
            ("enter_prefill", enter_prefill),
            ("exit_prefill", exit_prefill),
            ("enter_decode", enter_decode),
            ("exit_decode", exit_decode),
            ("kv_ascend_mark", kv_ascend_mark),
            ("kv_descend_mark", kv_descend_mark),
        ):
            if not 0.0 < value <= 1.0:
                raise RegimeError(f"{label} must be within (0, 1], got {value}")
        if exit_prefill >= enter_prefill:
            raise RegimeError(
                f"exit_prefill ({exit_prefill}) must be BELOW enter_prefill "
                f"({enter_prefill}): equal marks are not hysteresis, they are "
                f"a threshold a signal can sit on and cross every sample."
            )
        if exit_decode >= enter_decode:
            raise RegimeError(
                f"exit_decode ({exit_decode}) must be BELOW enter_decode "
                f"({enter_decode}): see exit_prefill."
            )
        if kv_descend_mark >= kv_ascend_mark:
            raise RegimeError(
                f"kv_descend_mark ({kv_descend_mark}) must be BELOW "
                f"kv_ascend_mark ({kv_ascend_mark})."
            )
        if enter_window < 1:
            raise RegimeError(f"enter_window must be >= 1, got {enter_window}")
        if exit_window <= enter_window:
            raise RegimeError(
                f"exit_window ({exit_window}) must be LONGER than enter_window "
                f"({enter_window}): entering a regime only proposes, leaving "
                f"one discards a classification the actuator may already be "
                f"acting on. The asymmetry is the contract."
            )
        self.window_rounds = int(window_rounds)
        self.enter_prefill = float(enter_prefill)
        self.exit_prefill = float(exit_prefill)
        self.enter_decode = float(enter_decode)
        self.exit_decode = float(exit_decode)
        self.enter_window = int(enter_window)
        self.exit_window = int(exit_window)
        self.kv_ascend_mark = float(kv_ascend_mark)
        self.kv_descend_mark = float(kv_descend_mark)
        self.burst_tokens = int(burst_tokens)

        self._regime = REGIME_MIXED
        self._candidate: Optional[str] = None
        self._candidate_streak = 0
        self._leave_streak = 0
        self.transitions = 0

    @property
    def regime(self) -> str:
        return self._regime

    def _still_holds(self, sample: RegimeSample) -> bool:
        """Whether the CURRENT regime still holds under its EXIT threshold."""
        occupancy = sample.occupancy
        if self._regime == REGIME_KV_PRESSURE:
            return occupancy is not None and occupancy > self.kv_descend_mark
        if occupancy is not None and occupancy >= self.kv_ascend_mark:
            # Pressure outranks: the incumbent cannot hold against it.
            return False
        if self._regime == REGIME_PREFILL_HEAVY:
            share = sample.prefill_share
            if share is not None and share > self.exit_prefill:
                return True
            return self.burst_tokens > 0 and (
                sample.queued_prompt_tokens >= self.burst_tokens
            )
        if self._regime == REGIME_DECODE_HEAVY:
            share = sample.decode_share
            return share is not None and share > self.exit_decode
        # MIXED asserts nothing, so it holds nothing: there is no incumbent
        # claim for a challenger to disprove, and requiring an exit window for
        # it would make MIXED absorbing.
        return False

    def observe(self, sample: RegimeSample) -> str:
        """Feed one boundary sample; return the regime in force afterwards.

        A regime changes only when BOTH halves agree: the incumbent has failed
        its exit test for ``exit_window`` consecutive samples, AND a challenger
        has passed its entry test for ``enter_window`` consecutive samples.
        Requiring both is what stops a single noisy sample from moving the
        state in either direction.
        """
        instant = classify_sample(
            sample,
            enter_prefill=self.enter_prefill,
            enter_decode=self.enter_decode,
            burst_tokens=self.burst_tokens,
            kv_ascend_mark=self.kv_ascend_mark,
        )

        if self._still_holds(sample):
            self._leave_streak = 0
        else:
            self._leave_streak += 1

        if instant == self._regime:
            self._candidate = None
            self._candidate_streak = 0
            return self._regime

        if instant == self._candidate:
            self._candidate_streak += 1
        else:
            self._candidate = instant
            self._candidate_streak = 1

        # MIXED is the resting state, and the asymmetry is deliberate:
        #
        # * LEAVING MIXED needs only the challenger's entry window. MIXED makes
        #   no claim, so there is nothing to disprove first -- and an exit
        #   window on it would make MIXED absorbing, which is the one regime
        #   the controller must always be able to leave.
        # * FALLING BACK to MIXED needs only the incumbent to have failed. It
        #   proposes nothing and actuates nothing, so it needs no positive
        #   evidence of its own.
        # * Every other transition needs BOTH: the incumbent gone for its exit
        #   window AND a challenger held for its entry window.
        challenger_ready = self._candidate_streak >= self.enter_window
        incumbent_gone = self._leave_streak >= self.exit_window
        if self._regime == REGIME_MIXED:
            commit = challenger_ready
        else:
            commit = incumbent_gone and (challenger_ready or instant == REGIME_MIXED)
        if commit:
            self._regime = instant
            self.transitions += 1
            self._candidate = None
            self._candidate_streak = 0
            self._leave_streak = 0
        return self._regime


# ---------------------------------------------------------------------------
# Dwell: the actuator-cost mechanism, separate from hysteresis
# ---------------------------------------------------------------------------


def min_dwell_rounds(
    flip_cost_s: float,
    mean_round_s: float,
    *,
    amortization: int = DWELL_AMORTIZATION,
) -> int:
    """How long a stage must be held for its flip to have been worth making.

    ``ceil(amortization * flip_cost_s / mean_round_s)``. Both inputs are
    MEASURED -- the flip's own instrumented duration (#297 reports
    ``total_ms`` per move; #330 reports its commit) and the live window's mean
    round time -- so the only judgement in the number is ``amortization``,
    which is named as policy in the design rather than presented as a finding.
    """
    if flip_cost_s < 0:
        raise RegimeError(f"flip_cost_s must be >= 0, got {flip_cost_s}")
    if mean_round_s <= 0:
        raise RegimeError(
            f"mean_round_s must be > 0, got {mean_round_s}: without a measured "
            f"round time there is nothing to amortize the flip against, and a "
            f"substituted one would set the dwell by the substitution."
        )
    if amortization < 1:
        raise RegimeError(f"amortization must be >= 1, got {amortization}")
    return max(1, math.ceil(amortization * flip_cost_s / mean_round_s))


class DwellGate:
    """Refuses a flip that would break the minimum dwell of the current stage.

    Deliberately not merged into :class:`RegimeSensor`. Hysteresis asks "is the
    signal real?"; dwell asks "is the move affordable?". A load that genuinely
    alternates faster than a flip amortizes gives a clean signal and must still
    be refused, and only a separate mechanism can say so.
    """

    def __init__(self, min_rounds: int):
        if min_rounds < 1:
            raise RegimeError(f"min_rounds must be >= 1, got {min_rounds}")
        self.min_rounds = int(min_rounds)
        self._last_flip_round: Optional[int] = None
        self.refusals = 0

    def rounds_since_flip(self, round_index: int) -> Optional[int]:
        if self._last_flip_round is None:
            return None
        return round_index - self._last_flip_round

    def allows(self, round_index: int) -> Tuple[bool, str]:
        since = self.rounds_since_flip(round_index)
        if since is None:
            return True, "no flip yet; the dwell gate has nothing to hold"
        if since >= self.min_rounds:
            return True, f"{since} rounds since the last flip >= {self.min_rounds}"
        self.refusals += 1
        return False, (
            f"dwell: {since} rounds since the last flip, minimum "
            f"{self.min_rounds}. The move would not have paid for itself "
            f"before the next one; a signal this clean and this fast is a "
            f"workload that alternates, not a regime that changed."
        )

    def record_flip(self, round_index: int) -> None:
        self._last_flip_round = int(round_index)


# ---------------------------------------------------------------------------
# The stage table
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Stage:
    """One planner-solved configuration. A TUPLE, not a knob.

    The #354 table is the reason: the FP8 prefill arm buys +22.6 % prefill and
    pays -79 % of ``max_total_num_tokens`` (453 632 -> 96 256). Moving the
    weight cut alone would collapse the pool under a live working set, so a
    stage binds the weight vector, the KV token vector and the VRAM budgets
    together and the flip moves all of them or none.

    ``measured_gain_pct`` and ``measured_band_pct`` are fields rather than
    review notes so that section 4.3's rule -- a stage whose gain does not
    clear its own A-vs-A band is not a stage -- is a constructor check.
    """

    name: str
    #: The regime this stage was solved for.
    regime: str
    #: MLP/GEMM weight vector, ``None`` for the plain VRAM-auto split.
    weight_vector: Optional[Tuple[int, ...]]
    #: KV token vector (#297 reshard target).
    kv_token_vector: Tuple[int, ...]
    #: Per-rank VRAM budget in MiB (#330 dial target).
    vram_budget_mib: Tuple[int, ...]
    #: The pool this stage funds. The admissibility interlock's denominator.
    max_total_num_tokens: int
    #: Measured gain over the reference stage on this stage's own phase, and
    #: the A-vs-A band that measurement was taken against (#360).
    measured_gain_pct: float
    measured_band_pct: float
    #: Instrumented cost of flipping INTO this stage, seconds.
    flip_cost_s: float
    #: #578: this stage was SOLVED, not MEASURED. The planner predicts a
    #: layout; it cannot predict an A-vs-A gain, the band that gain was taken
    #: against, or an instrumented flip cost -- those are measurements, and
    #: `SolverAnswer` carries no counterpart to any of the three.
    #:
    #: The flag exists so that "no measurement" is a STATE the table can name,
    #: rather than three zeros that read like a measured result of zero. A
    #: stage marked this way is refused by the #360 gate with its own reason;
    #: see `StageTable.__init__`. Supplying the measurements is #584's job.
    unmeasured: bool = False

    def __post_init__(self) -> None:
        if self.unmeasured and (
            self.measured_gain_pct or self.measured_band_pct or self.flip_cost_s
        ):
            raise RegimeError(
                f"stage {self.name!r} is marked unmeasured but carries "
                f"gain={self.measured_gain_pct}, "
                f"band={self.measured_band_pct}, "
                f"flip_cost={self.flip_cost_s}. An unmeasured stage must "
                f"carry placeholder zeros; a non-zero value here is a "
                f"measurement claim, and the whole point of the flag is that "
                f"nothing may claim one without having taken it."
            )
        if self.regime not in REGIMES:
            raise RegimeError(
                f"stage {self.name!r} names regime {self.regime!r}; known: "
                f"{', '.join(REGIMES)}"
            )
        if self.max_total_num_tokens <= 0:
            raise RegimeError(
                f"stage {self.name!r} funds no tokens "
                f"({self.max_total_num_tokens}); a stage with no pool cannot "
                f"be admissible for any working set."
            )
        if self.measured_band_pct < 0 or self.flip_cost_s < 0:
            raise RegimeError(
                f"stage {self.name!r}: measured_band_pct and flip_cost_s must "
                f"be >= 0"
            )

    @property
    def clears_its_band(self) -> bool:
        """#360: a gain inside its own A-vs-A band is not a measurement.

        An UNMEASURED stage never clears: there is no gain to compare and no
        band to compare it against. Returning False here rather than letting
        `abs(0.0) > 0.0` decide it is deliberate -- the two are the same
        boolean today, but they mean different things, and only one of them
        survives someone later giving a stage a real band of 0.0.
        """
        if self.unmeasured:
            return False
        return abs(self.measured_gain_pct) > self.measured_band_pct

    def admits(
        self, held_tokens: int, *, ascend_mark: float = KV_ASCEND_MARK
    ) -> Tuple[bool, str]:
        """Whether this stage's pool can hold ``held_tokens`` (design 4.2).

        ``held_tokens <= ascend_mark * max_total_num_tokens``. The mark is the
        pressure sensor's own, not a second one: admissibility and pressure are
        the same arithmetic on the same replicated sample, which is what stops
        the regime controller from flipping toward a pool the pressure ladder
        is simultaneously relieving.
        """
        ceiling = ascend_mark * self.max_total_num_tokens
        if held_tokens <= ceiling:
            return True, (
                f"{held_tokens} held tokens <= {ceiling:.0f} "
                f"({ascend_mark:.2f} x {self.max_total_num_tokens})"
            )
        return False, (
            f"stage {self.name!r} refused: {held_tokens} held tokens exceed "
            f"{ceiling:.0f} = {ascend_mark:.2f} x its pool of "
            f"{self.max_total_num_tokens}. Flipping would put the live working "
            f"set above this stage's ascend mark the moment it committed."
        )


class StageTable:
    """The planner-solved stages for one checkpoint on one rig.

    Refuses at construction, not at the first flip: a stage whose measured gain
    does not clear its own band is a candidate, not a stage, and admitting it
    would let the controller move a live server on a coin flip. The #354 INT8
    prefill arm is the live example -- +6.1 % from one boot against a 4.2 %
    harness floor, with its decode column explicitly undecided at n=1.
    """

    def __init__(self, stages: Sequence[Stage], *, reference: str):
        if not stages:
            raise RegimeError("a stage table needs at least one stage")
        by_name: Dict[str, Stage] = {}
        for stage in stages:
            if stage.name in by_name:
                raise RegimeError(f"duplicate stage name {stage.name!r}")
            by_name[stage.name] = stage
        if reference not in by_name:
            raise RegimeError(
                f"reference stage {reference!r} is not in the table "
                f"({', '.join(sorted(by_name))}); the reference is what every "
                f"other stage's gain was measured against."
            )
        rejected: List[str] = []
        unmeasured: List[str] = []
        for stage in stages:
            if stage.name == reference:
                continue
            if stage.unmeasured:
                unmeasured.append(stage.name)
            elif not stage.clears_its_band:
                rejected.append(
                    f"{stage.name!r}: gain {stage.measured_gain_pct:+.1f}% does "
                    f"not clear its own A-vs-A band of "
                    f"{stage.measured_band_pct:.1f}%"
                )
        if unmeasured:
            # #578: the planner feed is BOUND and produced these stages; what
            # they lack is measurement, not a solver. Naming that distinctly
            # from the inside-the-band case is the whole point -- the two look
            # identical from `clears_its_band` and have completely different
            # remedies ("measure these" vs "re-measure or drop these").
            raise RegimeError(
                "stage table refused (#578): the planner solved "
                f"{len(unmeasured)} stage(s) -- {', '.join(sorted(unmeasured))}"
                " -- but they carry no measurement. Each needs "
                "measured_gain_pct, measured_band_pct and flip_cost_s, taken "
                "once at stage creation (an A-vs-A run on the stage's own "
                "phase plus an instrumented flip), before #360 will let a "
                "controller select it. The solver cannot predict any of the "
                "three. This is NOT the old 'unfed' state: the feed is bound "
                "and working; what is missing is the measurement pass, which "
                "is #584's slice."
            )
        if rejected:
            raise RegimeError(
                "stage table refused (#360): a gain inside its own measured "
                "band is not evidence of a gain, and a controller that flips "
                "on one is flipping on noise with a number written next to "
                "it. Re-measure or drop: " + "; ".join(rejected)
            )
        self._by_name = by_name
        self._stages = tuple(stages)
        self.reference = reference

    def __len__(self) -> int:
        return len(self._stages)

    @property
    def stages(self) -> Tuple[Stage, ...]:
        return self._stages

    def __getitem__(self, name: str) -> Stage:
        return self._by_name[name]

    def for_regime(self, regime: str) -> Optional[Stage]:
        """The stage solved for ``regime``, or ``None`` when the table has
        none -- which is a normal answer and means "hold what you have"."""
        for stage in self._stages:
            if stage.regime == regime:
                return stage
        return None

    def select(
        self,
        regime: str,
        sample: RegimeSample,
        *,
        current: Optional[str] = None,
        ascend_mark: float = KV_ASCEND_MARK,
    ) -> Tuple[Optional[Stage], str]:
        """The stage to move to, or ``None`` with the reason not to.

        Pure: no actuator is called and no state is changed. The caller
        combines this with :class:`DwellGate` before arming anything.
        """
        if regime == REGIME_KV_PRESSURE:
            return None, (
                "KV_PRESSURE outranks stage selection: while the pool is the "
                "binding constraint the decision belongs to the pressure "
                "arithmetic, and a second controller writing the same axis is "
                "how the two end up undoing each other."
            )
        target = self.for_regime(regime)
        if target is None:
            return None, f"no stage is solved for regime {regime!r}; holding"
        if current is not None and target.name == current:
            return None, f"already on stage {target.name!r}"
        ok, why = target.admits(sample.held_tokens, ascend_mark=ascend_mark)
        if not ok:
            return None, why
        return target, (
            f"regime {regime!r} -> stage {target.name!r} "
            f"(gain {target.measured_gain_pct:+.1f}% over band "
            f"{target.measured_band_pct:.1f}%; {why})"
        )


# ---------------------------------------------------------------------------
# The #360 A-vs-A band
# ---------------------------------------------------------------------------


def signal_band(run_a: Sequence[float], run_a_repeat: Sequence[float]) -> float:
    """The band a signal measures on ITSELF across two identical runs.

    The house standard (#360, and the construction ``scripts/dual_group/r12``
    uses for the lane gate): the band a cross-arm delta has to clear is the
    largest difference the arm shows against its own repeat. Anything inside
    it is not a difference, whatever it is called.

    Both sequences must be the same length -- comparing windows that do not
    line up would report the misalignment as noise.
    """
    if len(run_a) != len(run_a_repeat):
        raise RegimeError(
            f"an A-vs-A band needs aligned windows: got {len(run_a)} and "
            f"{len(run_a_repeat)} samples. Comparing unaligned windows would "
            f"report the misalignment as the noise floor."
        )
    if not run_a:
        raise RegimeError(
            "an A-vs-A band needs at least one paired sample; an empty band "
            "would make every threshold clear itself trivially."
        )
    return max(abs(float(a) - float(b)) for a, b in zip(run_a, run_a_repeat))


def clears_band(delta: float, band: float, *, margin: float = 2.0) -> bool:
    """Whether ``delta`` is outside ``band`` by ``margin`` times the band.

    Design section 3.4 requires 2x for a THRESHOLD, because a threshold sitting
    one band away still spends half its time on the wrong side of itself.
    """
    if band < 0 or margin <= 0:
        raise RegimeError(f"band must be >= 0 and margin > 0, got {band}, {margin}")
    return abs(delta) > margin * band


# ---------------------------------------------------------------------------
# Consensus payload: routing the rank-local tier through the reduction
# ---------------------------------------------------------------------------

#: Field order of the packed proposal. Every field is carried as ``(v, -v)``
#: so ONE element-wise MIN reduction returns min and max per field -- the
#: convention #287 and #297 already use. For the tier-L field the interesting
#: output is ``max - min``: the spread across ranks.
PROPOSAL_FIELDS = ("regime_code", "target_stage_index", "epoch", "rank_ms_q")

#: Quantization of the tier-L timing before it enters the payload: hundredths
#: of a millisecond as an int. A float cannot be MIN-reduced into an equality
#: check without the last bits deciding a desync.
RANK_MS_QUANTUM = 0.01

#: Sentinel for "this rank has no timing this boundary". Negative so that a
#: MIN over the group surfaces it: if any rank is blind, the spread is not
#: computed rather than computed from a partial group.
_MS_ABSENT = -1


def pack_proposal(
    regime: str,
    target_stage_index: int,
    epoch: int,
    rank_forward_ms: Optional[float],
) -> List[int]:
    """This rank's proposal, packed for one MIN-reduction.

    ``rank_forward_ms`` is the ONLY rank-local input, and packing it here is
    the whole point: it crosses into the decision through the reduction that
    already runs unconditionally at the cadence boundary, never through a
    local branch. ``None`` (a graph-covered forward reports no split at all
    rather than a wrong zero) packs as a sentinel so the absence survives the
    reduction instead of reading as a fast rank.
    """
    if regime not in REGIME_CODES:
        raise RegimeError(f"unknown regime {regime!r}; known: {', '.join(REGIMES)}")
    ms_q = (
        _MS_ABSENT
        if rank_forward_ms is None
        else max(0, int(round(rank_forward_ms / RANK_MS_QUANTUM)))
    )
    out: List[int] = []
    for value in (REGIME_CODES[regime], int(target_stage_index), int(epoch), ms_q):
        out.append(int(value))
        out.append(-int(value))
    return out


def unpack_reduced(reduced: Sequence[int]) -> Dict[str, object]:
    """Read back a MIN-reduced payload: min, max and the derived spread.

    Returns ``agreed`` per equality-checked field so the caller can raise the
    same loud error on every rank -- the #287 rule that a disagreement fails
    HERE, before any rank acts on a geometry the others did not choose.
    """
    if len(reduced) != 2 * len(PROPOSAL_FIELDS):
        raise RegimeError(
            f"reduced payload has {len(reduced)} values, expected "
            f"{2 * len(PROPOSAL_FIELDS)}; the channel contract is element-wise "
            f"MIN of the packed proposal."
        )
    lo: Dict[str, int] = {}
    hi: Dict[str, int] = {}
    for i, field in enumerate(PROPOSAL_FIELDS):
        lo[field] = int(reduced[2 * i])
        hi[field] = -int(reduced[2 * i + 1])

    disagreements = [
        f"{f}: min={lo[f]} max={hi[f]}"
        for f in ("regime_code", "target_stage_index", "epoch")
        if lo[f] != hi[f]
    ]

    # The spread is the POINT of the tier-L field, so a disagreement there is
    # data, not an error. It is only computable when every rank reported.
    spread_pct: Optional[float] = None
    if lo["rank_ms_q"] > 0 and hi["rank_ms_q"] > 0:
        spread_pct = 100.0 * (hi["rank_ms_q"] - lo["rank_ms_q"]) / hi["rank_ms_q"]

    return {
        "regime": REGIME_BY_CODE.get(lo["regime_code"]),
        "target_stage_index": lo["target_stage_index"],
        "epoch": lo["epoch"],
        "rank_ms_spread_pct": spread_pct,
        "disagreements": disagreements,
        "agreed": not disagreements,
    }
