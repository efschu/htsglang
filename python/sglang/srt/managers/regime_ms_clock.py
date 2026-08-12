# SPDX-License-Identifier: Apache-2.0
"""The INTRA-PHASE axis of the #363 controller: ms/round drives the stage.

WHY THIS MODULE EXISTS
----------------------
The controller shipped before this module decides a stage from the REGIME
LABEL alone: ``RegimeSensor`` classifies a window as prefill-heavy /
decode-heavy / mixed from round shares and occupancy, and
``StageTable.select`` maps that label to a stage
(``regime_runtime.RegimeObserver.on_round``). The per-rank forward time is
collected in the same loop -- ``rank_forward_ms`` is accumulated at
``regime_runtime.py:340`` and averaged at ``:372`` -- but it never reaches the
decision: it is packed into the consensus payload
(``regime_classifier.pack_proposal``) for one purpose only, deriving
``rank_ms_spread_pct``, the cross-rank SKEW that act-mode interlock 4 vetoes
on. So the controller knows how long a round takes and has never once used
that number to choose a stage.

That is the gap this module closes. Two stages solved for the SAME regime
differ in measured gain; which of them is actually better on this rig, under
this load, right now, is an ms/round question and not a label question. The
label says WHICH FAMILY of stage to want. The clock says WHETHER THE MOVE IS
WORTH MAKING, in the unit the rig is actually judged in.

THE UNIT, AND WHY THE SPLIT IS LOAD-BEARING RATHER THAN DECORATIVE
------------------------------------------------------------------
ms per round, split COMPUTE vs WAIT, per rank -- the house measurement canon.
The split is not carried here for reporting. It is the only reason the
measurement enters the decision at all, and the arithmetic below is what makes
that true.

* COMPUTE ms is work. A stage flip changes the KV token vector (#297) and the
  per-rank VRAM budget (#330). Neither makes a GEMM faster.
* WAIT ms is a rank standing at a barrier because another rank is still
  computing -- i.e. the split is wrong. That is precisely the term the KV
  vector and the budget vector can move.

So the term a stage flip may be credited against is the WAIT term:

    predicted_ms = compute_ms + wait_ms * (1 + g_cur/100) / (1 + g_cand/100)

This is deliberately CONSERVATIVE -- it under-promises, because a stage's
measured gain was taken end to end and some of it may genuinely have come from
elsewhere -- and under-promising is the correct direction for a number that
authorizes spending memory.

IT IS ALSO WHAT KEEPS THE AXIS HONEST. The obvious formulation, scaling the
WHOLE round by the ratio of the two measured gains, has the current round time
cancel out of the improvement algebraically:

    100 * (t - t * cur / cand) / t  ==  100 * (1 - cur / cand)

-- a "ms/round-driven" controller whose verdict does not depend on a single
measured millisecond, and which returns the identical answer on a wait-bound
rig and a compute-bound one. That formulation was written here first and
removed for exactly this reason; the test
``test_improvement_depends_on_the_measured_split`` is what keeps it removed.
With the split, a compute-bound window yields an improvement near zero and
proposes no flip however large the table says the gain is -- the right answer,
because there is nothing on that window for these axes to win.

``Stage.measured_gain_pct`` is a gain over the REFERENCE stage on that stage's
own phase, and ``Stage.measured_band_pct`` is the A-vs-A band that gain was
taken against (``regime_classifier.Stage``). Because both stages in a
comparison are quoted against the SAME reference, their ratio is the
candidate's throughput advantage over the incumbent and no third measurement
is needed. The improvement the decider thresholds is ``100 * (total_ms -
predicted_ms) / total_ms``, i.e. the percentage of the round the flip claims
to give back.

THE BAND IS A FLOOR ON THE THRESHOLD, NOT A DECORATION. Section 4.3's rule --
a gain that does not clear its own A-vs-A band is not a gain -- is already a
constructor check for a single stage (``Stage.clears_its_band``). A DIFFERENCE
of two measured gains carries both bands, so the floor here is their
quadrature sum, which is the honest combination for two independent
measurements and is strictly larger than either. A candidate whose predicted
improvement does not clear that combined band is refused however long it
sustains: sustaining noise produces a long run of noise, not a signal.

REFUSING TO PRICE AN UNMEASURED STAGE
-------------------------------------
A planner-solved stage carries ``unmeasured=True`` and placeholder zeros in
all three measurement fields (#578: the solver predicts a layout, it cannot
predict an A-vs-A gain). Those zeros are NOT a measured gain of zero, and this
module refuses to compute an improvement from them rather than reading them as
"this stage is exactly as good as the current one". This is the same lesson as
the transient census's -- an unpriced term must not read as a free one -- and
it is why ``predicted_round_ms`` raises instead of returning ``current_ms``.

HYSTERESIS HERE, DWELL ELSEWHERE
--------------------------------
This module carries the two watermarks and the two windows. It does NOT carry
a minimum dwell: ``regime_classifier.DwellGate`` already owns that, the
separation is deliberate house doctrine ("hysteresis asks 'is the signal
real?'; dwell asks 'is the move affordable?'"), and duplicating it here would
put two different answers to the same question in two modules.

STATUS: desk code. Every claim above is pinned by the unit tests in
``test/registered/unit/managers/test_regime_ms_clock_363.py``; none of it has
run on metal yet. The measurement ticket that would change that is
``docs/dev/363/TICKET_363_STAGE_CLOCK.md``.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_ENTER_MARGIN_PCT",
    "DEFAULT_ENTER_WINDOW",
    "DEFAULT_EXIT_MARGIN_PCT",
    "DEFAULT_EXIT_WINDOW",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_WINDOW_SAMPLES",
    "MS_ABSENT",
    "MS_QUANTUM",
    "MsClockError",
    "MsDecision",
    "MsRoundSample",
    "MsRoundWindow",
    "MsStageDecider",
    "combined_band_pct",
    "improvement_pct",
    "pack_ms_sample",
    "predicted_round_ms",
    "unpack_ms_sample",
]

#: Integer resolution of the packed ms payload, in ms. The collective channel
#: carries ints; 0.01 ms is far finer than any round this decides about.
MS_QUANTUM = 0.01

#: A rank with no readable split packs this. It must survive the reduction as
#: an absence rather than arriving as a fast zero -- the same sentinel
#: discipline ``regime_classifier.pack_proposal`` already uses.
MS_ABSENT = -1


def pack_ms_sample(compute_ms: Optional[float], wait_ms: Optional[float]) -> List[int]:
    """This rank's split, packed for ONE element-wise MIN reduction.

    WHY THE CLOCK IS FED A REDUCED VALUE AND NOT ITS OWN.
    The observer's contract for ``rank_forward_ms`` is that it is the single
    rank-local input, "accumulated, never branched on, and released only
    through the reduction". A decision made from this rank's own milliseconds
    would break exactly that: on a real rig every rank measures a different
    number, so every rank would propose a different stage, and a controller
    whose ranks disagree either hangs or is permanently vetoed. So the ms axis
    consumes a GROUP statistic, computed by this reduction, and is uniform by
    construction rather than by hope.

    The pair idiom (each value and its negation) lets one MIN yield both the
    group minimum and the group maximum, as used in
    ``regime_classifier.pack_proposal`` and the seam's fit reduction.
    """
    if compute_ms is None or wait_ms is None:
        return [MS_ABSENT, MS_ABSENT, MS_ABSENT, MS_ABSENT]
    if compute_ms < 0.0 or wait_ms < 0.0:
        raise MsClockError(
            f"compute_ms and wait_ms must be >= 0, got {compute_ms} / {wait_ms}"
        )
    total_q = max(0, int(round((compute_ms + wait_ms) / MS_QUANTUM)))
    wait_q = max(0, int(round(wait_ms / MS_QUANTUM)))
    return [total_q, -total_q, wait_q, -wait_q]


def unpack_ms_sample(reduced: Sequence[int]) -> Optional[Tuple[float, float]]:
    """Read back the group's (compute_ms, wait_ms), or ``None`` if any rank
    was blind.

    THE TWO TERMS ARE TAKEN FROM OPPOSITE ENDS, DELIBERATELY:

    * ROUND LENGTH is the MAXIMUM total -- the slowest rank sets the barrier,
      which is the house pacemaker law ("the slowest rank is the clock, per
      barrier"). Taking a mean here would price the round at a length no rank
      actually experienced.
    * WAIT is the MINIMUM wait -- the barrier time EVERY rank was at least
      paying. This is the conservative end: it is the part of the round the
      stage axes can be confidently credited against, and understating it
      understates the predicted gain.

    Conservative in both directions at once (a long round, a small addressable
    term) is the correct bias for a number that goes on to authorize spending
    memory. ``compute`` is then the residual, ``round_max - wait_min``.
    """
    if len(reduced) != 4:
        raise MsClockError(
            f"reduced ms payload has {len(reduced)} values, expected 4; the "
            f"channel contract is element-wise MIN of the packed sample."
        )
    total_lo = int(reduced[0])
    total_hi = -int(reduced[1])
    wait_lo = int(reduced[2])
    if total_lo == MS_ABSENT or wait_lo == MS_ABSENT:
        # Some rank reported nothing. A partial group is not a group.
        return None
    total_ms = total_hi * MS_QUANTUM
    wait_ms = min(wait_lo * MS_QUANTUM, total_ms)
    return (total_ms - wait_ms, wait_ms)


class MsClockError(RuntimeError):
    """A misconfigured clock or an attempt to price an unmeasured stage."""


#: Enter watermark: the predicted improvement a challenger must show before it
#: is even a candidate, in percent of the current round time. POLICY, not a
#: finding -- 5 % is roughly twice the A-vs-A band this rig reports on a
#: quiet card, so a candidate at this mark is asking for a move worth more
#: than the noise it was measured against.
DEFAULT_ENTER_MARGIN_PCT = 5.0

#: Exit watermark: below this, the incumbent stage still counts as holding.
#: Strictly below the enter mark, enforced at construction.
DEFAULT_EXIT_MARGIN_PCT = 2.0

#: Consecutive boundaries a challenger must lead by before it may be adopted.
DEFAULT_ENTER_WINDOW = 2

#: Consecutive boundaries the incumbent must fail to hold before it may be
#: abandoned. Strictly LONGER than the enter window, enforced at construction:
#: adopting a challenger only proposes a move, abandoning the incumbent
#: discards a configuration the actuators may already be moving toward.
DEFAULT_EXIT_WINDOW = 4

#: Sliding window of per-round samples the means are taken over.
DEFAULT_WINDOW_SAMPLES = 64

#: Below this many samples the window answers "not ready" rather than a mean
#: of two rounds. A flip decided on two samples is decided on the two.
DEFAULT_MIN_SAMPLES = 8


@dataclasses.dataclass(frozen=True)
class MsRoundSample:
    """One round's device time for this rank, split compute vs wait."""

    round_index: int
    compute_ms: float
    wait_ms: float

    @property
    def total_ms(self) -> float:
        return self.compute_ms + self.wait_ms

    @property
    def wait_share(self) -> Optional[float]:
        total = self.total_ms
        if total <= 0.0:
            return None
        return self.wait_ms / total


class MsRoundWindow:
    """A sliding window of per-round ms samples with a readiness floor.

    Rank-local by construction. Nothing in this class crosses the TP group;
    the decision built on top of it is made group-uniform by the caller's
    consensus reduction, exactly as the regime verdict already is.
    """

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_WINDOW_SAMPLES,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ):
        if capacity < 1:
            raise MsClockError(f"capacity must be >= 1, got {capacity}")
        if min_samples < 1:
            raise MsClockError(f"min_samples must be >= 1, got {min_samples}")
        if min_samples > capacity:
            raise MsClockError(
                f"min_samples ({min_samples}) exceeds capacity ({capacity}): "
                f"the window could never become ready."
            )
        self.capacity = int(capacity)
        self.min_samples = int(min_samples)
        self._samples: Deque[MsRoundSample] = deque(maxlen=self.capacity)

    def observe(
        self, round_index: int, compute_ms: float, wait_ms: float
    ) -> MsRoundSample:
        """Record one round.

        Negative inputs are refused rather than clamped: the producer
        (``metrics_reporter``) already clamps the compute term against event
        granularity, so a negative arriving here means a different bug and
        swallowing it would hide that bug behind a plausible mean.
        """
        if compute_ms < 0.0:
            raise MsClockError(f"compute_ms must be >= 0, got {compute_ms}")
        if wait_ms < 0.0:
            raise MsClockError(f"wait_ms must be >= 0, got {wait_ms}")
        sample = MsRoundSample(
            round_index=int(round_index),
            compute_ms=float(compute_ms),
            wait_ms=float(wait_ms),
        )
        self._samples.append(sample)
        return sample

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def ready(self) -> bool:
        return len(self._samples) >= self.min_samples

    @property
    def mean_compute_ms(self) -> Optional[float]:
        if not self._samples:
            return None
        return sum(s.compute_ms for s in self._samples) / len(self._samples)

    @property
    def mean_wait_ms(self) -> Optional[float]:
        if not self._samples:
            return None
        return sum(s.wait_ms for s in self._samples) / len(self._samples)

    @property
    def mean_total_ms(self) -> Optional[float]:
        if not self._samples:
            return None
        return sum(s.total_ms for s in self._samples) / len(self._samples)

    @property
    def mean_wait_share(self) -> Optional[float]:
        """Share of the mean round spent waiting at a barrier.

        Reported, never branched on (see the module docstring). It is the term
        that tells a ticket reader whether a win came from a rebalanced split
        or from somewhere the stage axes cannot reach.
        """
        total = self.mean_total_ms
        if total is None or total <= 0.0:
            return None
        return (self.mean_wait_ms or 0.0) / total

    def snapshot(self) -> Dict[str, Optional[float]]:
        return {
            "n": len(self._samples),
            "mean_compute_ms": self.mean_compute_ms,
            "mean_wait_ms": self.mean_wait_ms,
            "mean_total_ms": self.mean_total_ms,
            "mean_wait_share": self.mean_wait_share,
        }


def _require_measured(stage, role: str) -> None:
    if getattr(stage, "unmeasured", False):
        raise MsClockError(
            f"{role} stage {stage.name!r} is UNMEASURED (#578: planner-solved, "
            f"placeholder zeros in measured_gain_pct / measured_band_pct / "
            f"flip_cost_s). Those zeros are not a measured gain of zero, and "
            f"reading them as one would price this flip as free. Measure the "
            f"stage or leave it out of the ms-driven axis."
        )


def predicted_round_ms(
    compute_ms: float, wait_ms: float, current, candidate
) -> float:
    """Predicted round time under ``candidate``, given the measured window.

    Only the WAIT term is rescaled. The stage axes (#297 KV token vector, #330
    per-rank VRAM budget) move how work is divided across ranks; they do not
    make the arithmetic on any one rank faster. Crediting a stage's gain
    against the compute term would promise a win from an axis that cannot
    produce one -- and, as the module docstring shows, would also cancel the
    measured round out of the verdict entirely.

    Both stages quote ``measured_gain_pct`` against the SAME reference stage,
    so their ratio is the candidate's throughput advantage over the incumbent
    and no third measurement is required.
    """
    if compute_ms < 0.0 or wait_ms < 0.0:
        raise MsClockError(
            f"compute_ms and wait_ms must be >= 0, got {compute_ms} / {wait_ms}"
        )
    total_ms = float(compute_ms) + float(wait_ms)
    if total_ms <= 0.0:
        raise MsClockError(
            f"total round time must be > 0, got {total_ms}: without a measured "
            f"round there is nothing to predict against, and a substituted "
            f"one would set the prediction by the substitution."
        )
    _require_measured(current, "current")
    _require_measured(candidate, "candidate")
    cur_scale = 1.0 + float(current.measured_gain_pct) / 100.0
    cand_scale = 1.0 + float(candidate.measured_gain_pct) / 100.0
    if cur_scale <= 0.0 or cand_scale <= 0.0:
        raise MsClockError(
            f"a measured gain at or below -100 % ({current.name!r}: "
            f"{current.measured_gain_pct}, {candidate.name!r}: "
            f"{candidate.measured_gain_pct}) describes a stage that produces "
            f"no throughput; it cannot be converted to a round time."
        )
    return float(compute_ms) + float(wait_ms) * cur_scale / cand_scale


def improvement_pct(compute_ms: float, wait_ms: float, current, candidate) -> float:
    """Percent of the measured round the candidate claims to give back.

    Negative when the candidate is the slower stage, which is the normal
    reading for the reverse direction of a pair. Zero on a window with no
    wait at all, whatever the table's gains say -- see the module docstring.
    """
    total_ms = float(compute_ms) + float(wait_ms)
    predicted = predicted_round_ms(compute_ms, wait_ms, current, candidate)
    return 100.0 * (total_ms - predicted) / total_ms


def combined_band_pct(current, candidate) -> float:
    """The A-vs-A band a DIFFERENCE of two measured gains must clear.

    Quadrature sum of the two stages' own bands: both measurements carry
    noise, the difference carries both, and the quadrature sum is the honest
    combination for two independent measurements. It is strictly larger than
    either band alone, so a candidate cannot clear the joint band by having
    been measured on a quieter card than the incumbent.
    """
    _require_measured(current, "current")
    _require_measured(candidate, "candidate")
    a = float(current.measured_band_pct)
    b = float(candidate.measured_band_pct)
    if a < 0.0 or b < 0.0:
        raise MsClockError(
            f"measured_band_pct must be >= 0, got {a} / {b}: a negative band "
            f"is not a tighter measurement."
        )
    return math.sqrt(a * a + b * b)


@dataclasses.dataclass(frozen=True)
class MsDecision:
    """One boundary's verdict from the ms axis.

    ``target`` is ``None`` for every outcome except an adopted challenger, and
    ``reason`` always says which of the several ways that happened.
    """

    target: Optional[str]
    reason: str
    #: Best candidate's predicted improvement, percent of the current round.
    signal_pct: Optional[float] = None
    #: The band that signal had to clear, percent.
    band_pct: Optional[float] = None
    #: Name of the best candidate considered, adopted or not.
    candidate: Optional[str] = None
    #: Window means at the moment of the verdict.
    mean_total_ms: Optional[float] = None
    mean_wait_share: Optional[float] = None
    #: Streak counters, so a trace can show WHY a clear signal did not move.
    candidate_streak: int = 0
    leave_streak: int = 0

    @property
    def wants_flip(self) -> bool:
        return self.target is not None

    def as_dict(self) -> Dict:
        return {
            "target": self.target,
            "reason": self.reason,
            "signal_pct": self.signal_pct,
            "band_pct": self.band_pct,
            "candidate": self.candidate,
            "mean_total_ms": self.mean_total_ms,
            "mean_wait_share": self.mean_wait_share,
            "candidate_streak": self.candidate_streak,
            "leave_streak": self.leave_streak,
        }


class MsStageDecider:
    """Two-watermark, two-window hysteresis over predicted ms/round.

    A flip requires BOTH halves to agree, mirroring ``RegimeSensor``:

    1. THE CHALLENGER LEADS. The same best candidate clears the enter
       watermark (and its combined band) for ``enter_window`` consecutive
       boundaries.
    2. THE INCUMBENT HAS STOPPED HOLDING. Some candidate has cleared the exit
       watermark for ``exit_window`` consecutive boundaries.

    Because ``exit_window > enter_window``, condition 2 is the binding one and
    a burst of noise cannot move anything: the challenger has to keep winning
    across a window longer than the burst. A candidate that changes identity
    between boundaries resets the first streak, which is what makes an
    oscillating trace produce a bounded number of flips rather than one per
    boundary.

    What this class deliberately does NOT do: enforce a minimum dwell
    (``DwellGate`` owns that), price the move in VRAM (``regime_admission``
    owns that), or touch an actuator (``regime_act`` owns that).
    """

    def __init__(
        self,
        *,
        enter_margin_pct: float = DEFAULT_ENTER_MARGIN_PCT,
        exit_margin_pct: float = DEFAULT_EXIT_MARGIN_PCT,
        enter_window: int = DEFAULT_ENTER_WINDOW,
        exit_window: int = DEFAULT_EXIT_WINDOW,
        window: Optional[MsRoundWindow] = None,
    ):
        if enter_margin_pct <= 0.0:
            raise MsClockError(
                f"enter_margin_pct must be > 0, got {enter_margin_pct}: a "
                f"zero mark adopts any candidate whose prediction rounds up."
            )
        if exit_margin_pct < 0.0:
            raise MsClockError(
                f"exit_margin_pct must be >= 0, got {exit_margin_pct}"
            )
        if exit_margin_pct >= enter_margin_pct:
            raise MsClockError(
                f"exit_margin_pct ({exit_margin_pct}) must be BELOW "
                f"enter_margin_pct ({enter_margin_pct}): equal marks are not "
                f"hysteresis, they are a threshold a signal can sit on and "
                f"cross every boundary."
            )
        if enter_window < 1:
            raise MsClockError(f"enter_window must be >= 1, got {enter_window}")
        if exit_window <= enter_window:
            raise MsClockError(
                f"exit_window ({exit_window}) must be LONGER than "
                f"enter_window ({enter_window}): adopting a challenger only "
                f"proposes a move, abandoning the incumbent discards a "
                f"configuration the actuators may already be moving toward. "
                f"The asymmetry is the contract."
            )
        self.enter_margin_pct = float(enter_margin_pct)
        self.exit_margin_pct = float(exit_margin_pct)
        self.enter_window = int(enter_window)
        self.exit_window = int(exit_window)
        self.window = window if window is not None else MsRoundWindow()

        self._candidate: Optional[str] = None
        self._candidate_streak = 0
        self._leave_streak = 0
        self.proposals = 0
        self.last: Optional[MsDecision] = None

    # -- state ---------------------------------------------------------------
    @property
    def candidate(self) -> Optional[str]:
        return self._candidate

    @property
    def candidate_streak(self) -> int:
        return self._candidate_streak

    @property
    def leave_streak(self) -> int:
        return self._leave_streak

    def observe_round(
        self, round_index: int, compute_ms: float, wait_ms: float
    ) -> None:
        """Feed one round into the sliding window. Cheap; called every round."""
        self.window.observe(round_index, compute_ms, wait_ms)

    def note_flip(self) -> None:
        """Reset the streaks after a flip actually happened.

        The window is cleared too: its samples were taken under the OLD stage
        and averaging them into the new stage's evidence would judge the new
        configuration partly by the one it replaced.
        """
        self._candidate = None
        self._candidate_streak = 0
        self._leave_streak = 0
        self.window.clear()

    # -- the verdict ---------------------------------------------------------
    def decide(self, current, candidates: Sequence) -> MsDecision:
        """One boundary's verdict.

        ``current`` is the ``Stage`` in force; ``candidates`` are the stages
        the table judged selectable this boundary. Unmeasured stages and the
        incumbent itself are skipped rather than refused -- a table legitimately
        mixes measured and planner-solved stages, and the ms axis simply has
        no opinion about the latter.
        """
        means = self.window.snapshot()
        mean_total = means["mean_total_ms"]
        mean_compute = means["mean_compute_ms"]
        mean_wait = means["mean_wait_ms"]
        mean_wait_share = means["mean_wait_share"]

        if not self.window.ready:
            return self._record(
                MsDecision(
                    target=None,
                    reason=(
                        f"ms window not ready: {len(self.window)} of "
                        f"{self.window.min_samples} samples. A flip decided on "
                        f"fewer samples is decided by those samples."
                    ),
                    mean_total_ms=mean_total,
                    mean_wait_share=mean_wait_share,
                    candidate_streak=self._candidate_streak,
                    leave_streak=self._leave_streak,
                )
            )
        if not mean_total or mean_total <= 0.0:
            return self._record(
                MsDecision(
                    target=None,
                    reason=(
                        "mean round time is zero or absent; nothing to compare "
                        "a predicted round against"
                    ),
                    mean_total_ms=mean_total,
                    mean_wait_share=mean_wait_share,
                    candidate_streak=self._candidate_streak,
                    leave_streak=self._leave_streak,
                )
            )
        if getattr(current, "unmeasured", False):
            return self._record(
                MsDecision(
                    target=None,
                    reason=(
                        f"the stage in force ({current.name!r}) is UNMEASURED, "
                        f"so no candidate's gain can be expressed relative to "
                        f"it; the ms axis abstains rather than treating its "
                        f"placeholder zeros as a measurement"
                    ),
                    mean_total_ms=mean_total,
                    mean_wait_share=mean_wait_share,
                    candidate_streak=self._candidate_streak,
                    leave_streak=self._leave_streak,
                )
            )

        scored = self._score(mean_compute or 0.0, mean_wait or 0.0, current, candidates)
        if not scored:
            self._candidate = None
            self._candidate_streak = 0
            self._leave_streak = 0
            return self._record(
                MsDecision(
                    target=None,
                    reason=(
                        "no measured candidate differs from the stage in "
                        "force; the ms axis has nothing to compare"
                    ),
                    mean_total_ms=mean_total,
                    mean_wait_share=mean_wait_share,
                )
            )

        best_name, best_signal, best_band = scored[0]

        # Half 2 first: has the incumbent stopped holding? Any candidate over
        # the EXIT mark counts, not only the leader -- the question is whether
        # the current stage is still clearly the right one, and two rotating
        # near-ties answer that as decisively as one steady challenger.
        if best_signal > self.exit_margin_pct:
            self._leave_streak += 1
        else:
            self._leave_streak = 0

        # Half 1: is the SAME challenger leading, over its own band?
        clears = best_signal > self.enter_margin_pct and best_signal > best_band
        if clears and best_name == self._candidate:
            self._candidate_streak += 1
        elif clears:
            self._candidate = best_name
            self._candidate_streak = 1
        else:
            self._candidate = None
            self._candidate_streak = 0

        common = dict(
            signal_pct=best_signal,
            band_pct=best_band,
            candidate=best_name,
            mean_total_ms=mean_total,
            mean_wait_share=mean_wait_share,
            candidate_streak=self._candidate_streak,
            leave_streak=self._leave_streak,
        )

        if best_signal <= best_band:
            return self._record(
                MsDecision(
                    target=None,
                    reason=(
                        f"best candidate {best_name!r} predicts "
                        f"{best_signal:.2f} % which does not clear its "
                        f"combined A-vs-A band of {best_band:.2f} %; a gain "
                        f"inside the band is not a gain however long it "
                        f"sustains"
                    ),
                    **common,
                )
            )
        if self._candidate_streak < self.enter_window:
            return self._record(
                MsDecision(
                    target=None,
                    reason=(
                        f"challenger {best_name!r} leads by "
                        f"{best_signal:.2f} % for {self._candidate_streak} of "
                        f"{self.enter_window} required boundaries"
                    ),
                    **common,
                )
            )
        if self._leave_streak < self.exit_window:
            return self._record(
                MsDecision(
                    target=None,
                    reason=(
                        f"the stage in force ({current.name!r}) has failed to "
                        f"hold for {self._leave_streak} of "
                        f"{self.exit_window} required boundaries; leaving is "
                        f"deliberately slower than arriving"
                    ),
                    **common,
                )
            )

        self.proposals += 1
        return self._record(
            MsDecision(
                target=best_name,
                reason=(
                    f"ms axis proposes {best_name!r}: predicted "
                    f"{best_signal:.2f} % of a {mean_total:.1f} ms round, "
                    f"clearing a {best_band:.2f} % band, sustained "
                    f"{self._candidate_streak}/{self.enter_window} boundaries "
                    f"while {current.name!r} failed to hold for "
                    f"{self._leave_streak}/{self.exit_window}"
                ),
                **common,
            )
        )

    # -- internals -----------------------------------------------------------
    def _score(
        self,
        mean_compute_ms: float,
        mean_wait_ms: float,
        current,
        candidates: Iterable,
    ) -> List[Tuple[str, float, float]]:
        """(name, improvement_pct, band_pct) for every comparable candidate.

        Sorted best first. Ties break on the name so that two stages with an
        identical predicted improvement produce the SAME leader on every rank
        -- a tie broken by dict order is a rank-local decision wearing a
        deterministic-looking hat.
        """
        scored: List[Tuple[str, float, float]] = []
        for cand in candidates:
            if cand is None or cand.name == current.name:
                continue
            if getattr(cand, "unmeasured", False):
                continue
            scored.append(
                (
                    cand.name,
                    improvement_pct(mean_compute_ms, mean_wait_ms, current, cand),
                    combined_band_pct(current, cand),
                )
            )
        scored.sort(key=lambda row: (-row[1], row[0]))
        return scored

    def _record(self, decision: MsDecision) -> MsDecision:
        self.last = decision
        return decision

    def summary(self) -> Dict:
        return {
            "proposals": self.proposals,
            "enter_margin_pct": self.enter_margin_pct,
            "exit_margin_pct": self.exit_margin_pct,
            "enter_window": self.enter_window,
            "exit_window": self.exit_window,
            "window": self.window.snapshot(),
            "candidate": self._candidate,
            "candidate_streak": self._candidate_streak,
            "leave_streak": self._leave_streak,
            "last": self.last.as_dict() if self.last else None,
        }
