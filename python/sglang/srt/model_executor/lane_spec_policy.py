# Copyright 2023-2026 SGLang Team
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
"""Which chain length K the dual-group lane speculates with, per round.

#274 round 7a. Deliberately a module of its own and deliberately free of any
sglang import: it is arithmetic over numbers the lane hands it, it is the piece
round 7b has to extend (turn routing feeds the same object another input), and
a policy that can be unit-tested without a GPU is a policy whose hysteresis and
break-evens can be argued about without a boot.

THE CRITERION IS MARGINAL, NOT AVERAGE, and that distinction is the whole
design. Comparing each rung's TOTAL predicted ms/token against a break-even
looks equivalent and is not, because accept length grows SUBLINEARLY in K: on
prose the acceptance rate collapses after a position or two, so the average
keeps improving slightly while every added row is already pure cost. The
average hides that; the margin states it:

    row j pays  <=>  P(the first j proposals are all accepted) * t_decode  >  t_row

* ``t_decode`` is what one plain decode step costs -- the K=0 rung, measured in
  THIS boot, not a constant. It is the whole saving an accepted proposal buys.
* ``t_row`` is what one more candidate row costs, taken as the SLOPE of the
  per-rung round times against the row count. Affine is not an assumption about
  the hardware: round 6 measured the captured verify at 16.1 / 21.5 / 27.2 ms
  for 1 / 2 / 4 rows. The slope is re-derived per boot and includes the extra
  head forward, because it is fitted against whole ROUNDS.
* On this vehicle those two give a threshold near 3.7 / 16.1 = 0.23, but the
  number is derived, never written down.

``P(first j accepted)`` comes from PER-POSITION counters rather than from
inverting a mean accept length. That is what makes saturation visible: a chain
whose first proposal is usually right and whose third never is has the same
mean as one that degrades evenly, and only the per-position view can tell them
apart -- which is exactly the case the margin exists to catch. Positions the
current rung never reaches are extrapolated geometrically from the deepest
position that WAS measured, so a short rung can still argue for a longer one.

K is then the depth at which the margin flips, rounded DOWN to an available
rung: with a ladder of {0, 1, 3} and a flip at depth 2, the answer is 1, not 3.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_lane_spec_rungs(spec: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Parse ``--dual-group-lane-spec-rungs`` into a sorted tuple of K values.

    Returns None for an unset value (the single-rung, pre-ladder behaviour).
    Raises ValueError with the offending text, because the alternative -- a
    silently dropped rung -- turns into a measurement that quietly never
    visited the rung the operator asked for.
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        parts: Iterable[Any] = spec
    else:
        text = str(spec).strip()
        if not text:
            return None
        parts = [p for p in text.replace(" ", "").split(",") if p]
    rungs = set()
    for part in parts:
        try:
            k = int(part)
        except (TypeError, ValueError):
            raise ValueError(
                f"--dual-group-lane-spec-rungs: {part!r} is not an integer "
                f"(expected a comma-separated list of chain lengths, "
                f"e.g. '0,1,2,3')."
            )
        if k < 0:
            raise ValueError(
                f"--dual-group-lane-spec-rungs: chain length {k} is negative; "
                f"0 (the lane's plain decode step) is the shortest rung."
            )
        rungs.add(k)
    if not rungs:
        return None
    return tuple(sorted(rungs))


class LaneSpecPolicy:
    """Chain-length policy for one dual-group lane.

    Stateful across rounds and across jobs of one boot: the accept statistics
    and the per-rung costs are properties of the vehicle, not of a request, and
    throwing them away per job would put the probe phase into every job.

    Not thread-safe on purpose. The lane's rounds are serial within the lane
    (one worker thread), which is where every call comes from.
    """

    def __init__(
        self,
        rungs: Sequence[int],
        *,
        adaptive: bool = False,
        hysteresis: int = 4,
        probe_rounds: int = 3,
        cost_ema: float = 0.25,
        accept_ema: float = 0.1,
        margin: float = 0.02,
        default_rung: Optional[int] = None,
    ):
        rungs = tuple(sorted({int(k) for k in rungs}))
        if not rungs:
            raise ValueError("LaneSpecPolicy needs at least one rung.")
        self.rungs: Tuple[int, ...] = rungs
        self.adaptive = bool(adaptive) and len(rungs) > 1
        self.hysteresis = max(1, int(hysteresis))
        self.probe_rounds = max(0, int(probe_rounds))
        self.cost_ema = float(cost_ema)
        self.accept_ema = float(accept_ema)
        # A challenger has to be better by this fraction, not merely different.
        # Without it two rungs whose predicted ms/token differ in the fourth
        # decimal would trade places on noise and the hysteresis counter would
        # only slow that down rather than stop it.
        self.margin = float(margin)

        default = self.rungs[-1] if default_rung is None else int(default_rung)
        if default not in self.rungs:
            default = self.rungs[-1]
        self.current: int = default

        # Per-rung EMA of the whole round (propose + verify), and how often the
        # rung has been observed. The count is what decides probe-vs-adapt, so
        # it is kept separately from the EMA rather than inferred from it.
        self._round_ms: Dict[int, float] = {}
        self._round_n: Dict[int, int] = {k: 0 for k in self.rungs}
        # PER-POSITION acceptance, EMA'd so old content stops voting. Index j
        # (0-based) is the j-th proposal of the chain. ``_pos_reached[j]`` is
        # how often that position was actually evaluated (the chain was long
        # enough AND every earlier proposal was accepted); ``_pos_hits[j]`` how
        # often it was accepted. Per position rather than pooled, because the
        # thing the margin has to see is SATURATION -- a head whose first
        # proposal is usually right and whose third never is has the same mean
        # accept length as one that degrades evenly.
        self._pos_hits: Dict[int, float] = {}
        self._pos_reached: Dict[int, float] = {}
        self._rounds: int = 0
        self._switches: int = 0
        self._pending: Optional[int] = None
        self._pending_n: int = 0
        self._last_reason: str = "init"

    # -- observation ---------------------------------------------------

    def observe(self, rung: int, round_ms: float, emitted: int) -> None:
        """Record one completed round.

        ``emitted`` is the number of tokens the round produced, which for a
        speculative round is ``n_accept + 1`` and for the K=0 rung is 1. The
        acceptance events are derived from it rather than passed in, so a
        caller cannot report an accept length and an acceptance count that
        disagree.
        """
        rung = int(rung)
        self._rounds += 1
        if round_ms is not None and round_ms > 0:
            prev = self._round_ms.get(rung)
            self._round_ms[rung] = (
                float(round_ms)
                if prev is None
                else prev + self.cost_ema * (float(round_ms) - prev)
            )
            self._round_n[rung] = self._round_n.get(rung, 0) + 1
        if rung <= 0:
            # A plain decode step carries no proposal, so it is evidence about
            # cost and about nothing else. Folding it into the acceptance
            # counters as a "failure" would drag p down for a reason that has
            # nothing to do with the head's quality.
            return
        n_accept = max(0, int(emitted) - 1)
        # A greedy chain stops at the first rejection, so exactly the positions
        # 0..n_accept were EVALUATED: the first n_accept were accepted and the
        # one after them (if the chain was long enough to have it) was not.
        for j in range(rung):
            if j > n_accept:
                break
            hit = 1.0 if j < n_accept else 0.0
            self._pos_hits[j] = self._pos_hits.get(j, 0.0) + self.accept_ema * (
                hit - self._pos_hits.get(j, 0.0)
            )
            self._pos_reached[j] = self._pos_reached.get(j, 0.0) + self.accept_ema * (
                1.0 - self._pos_reached.get(j, 0.0)
            )

    # -- prediction ----------------------------------------------------

    def position_accept(self, j: int) -> Optional[float]:
        """P(proposal ``j`` accepted | it was evaluated), or None if unseen."""
        reached = self._pos_reached.get(j)
        if not reached or reached <= 1e-9:
            return None
        p = self._pos_hits.get(j, 0.0) / reached
        # Clamp strictly below 1: p == 1 tells the margin an unbounded chain is
        # free, which is exactly the belief the margin exists to prevent.
        return min(max(p, 0.0), 0.999)

    @property
    def p_accept(self) -> Optional[float]:
        """The DEEPEST measured per-position rate, used for extrapolation.

        The margin needs a rate for positions no rung has reached yet, and the
        honest stand-in is the deepest one that WAS measured -- not the average
        over all positions, which the shallow (high) positions dominate and
        which would therefore argue for chains the head cannot sustain. This is
        the conservative direction on saturating content and the neutral one on
        flat content.
        """
        seen = [j for j, r in self._pos_reached.items() if r > 1e-9]
        if not seen:
            return None
        return self.position_accept(max(seen))

    def reach_probability(self, j: int) -> Optional[float]:
        """P(the first ``j`` proposals are ALL accepted).

        The exact quantity the margin multiplies ``t_decode`` by: a round at
        chain length j-1 extended to j buys one extra emitted token precisely
        when every proposal up to and including j-1 is accepted.
        """
        if j <= 0:
            return 1.0
        prob = 1.0
        for i in range(j):
            p = self.position_accept(i)
            if p is None:
                p = self.p_accept
                if p is None:
                    return None
            prob *= p
        return prob

    def predicted_accept(self, rung: int) -> float:
        """Expected tokens emitted by one round at ``rung``.

        Reported (and used for the tables), not used by the decision: the
        decision is marginal. Built from the same per-position rates, so it
        shows saturation rather than smoothing it away.
        """
        total = 1.0
        for j in range(1, int(rung) + 1):
            r = self.reach_probability(j)
            if r is None:
                break
            total += r
        return total

    def _cost_fit(self, min_rung: int = 0) -> Optional[Tuple[float, float]]:
        """Least-squares (fixed, per-row) fit over the rungs seen so far.

        Rows are ``k + 1`` because a round at chain length k verifies k+1
        candidate rows. Two observed rungs are enough; one is not, and the
        caller falls back to the observed value in that case.

        ``min_rung=1`` EXCLUDES the K=0 point, and the marginal criterion uses
        it that way for a reason the measurements made plain: K=0 is not a
        one-row verify, it is the plain DECODE graph, and it does not sit on
        the same line. Measured here: 16.16 / 24.24 / 27.99 / 33.64 ms at
        K = 0 / 1 / 2 / 3, i.e. the first step costs 8.1 ms and the later ones
        3.7 and 5.7. Fitting through K=0 would spread that step over the whole
        ladder and make every rung look equally priced.
        """
        pts: List[Tuple[float, float]] = [
            (float(k + 1), self._round_ms[k])
            for k in sorted(self._round_ms)
            if self._round_n.get(k, 0) > 0 and k >= min_rung
        ]
        if len(pts) < 2:
            return None
        n = float(len(pts))
        sx = sum(x for x, _ in pts)
        sy = sum(y for _, y in pts)
        sxx = sum(x * x for x, _ in pts)
        sxy = sum(x * y for x, y in pts)
        den = n * sxx - sx * sx
        if abs(den) < 1e-9:
            return None
        slope = (n * sxy - sx * sy) / den
        intercept = (sy - slope * sx) / n
        return intercept, slope

    def predicted_round_ms(self, rung: int) -> Optional[float]:
        """Expected ms for one round at ``rung``: measured, else fitted.

        A verify rung is fitted over the VERIFY rungs only, for the reason in
        ``_cost_fit``: K=0 is a different graph and does not sit on that line.
        """
        if self._round_n.get(rung, 0) > 0:
            return self._round_ms[rung]
        fit = self._cost_fit(min_rung=1 if rung >= 1 else 0)
        if fit is None:
            return None
        intercept, slope = fit
        value = intercept + slope * float(rung + 1)
        return value if value > 0 else None

    def predicted_ms_per_token(self, rung: int) -> Optional[float]:
        """Reported, not decided on. See the module docstring: the AVERAGE is
        what hides saturation, so it belongs in the tables and not in the rule.
        """
        ms = self.predicted_round_ms(rung)
        if ms is None:
            return None
        accept = self.predicted_accept(rung)
        if accept <= 0:
            return None
        return ms / accept

    # -- the marginal criterion ----------------------------------------

    @property
    def t_decode(self) -> Optional[float]:
        """What one plain decode step costs -- the K=0 rung, measured.

        This is the entire saving one accepted proposal buys, so it is the
        multiplier on the gain side of the margin. Measured, never a constant;
        if the ladder has no K=0 rung the affine fit stands in for it and the
        stats say so.
        """
        if self._round_n.get(0, 0) > 0:
            return self._round_ms[0]
        fit = self._cost_fit(min_rung=1)
        if fit is None:
            return None
        intercept, slope = fit
        value = intercept + slope
        return value if value > 0 else None

    def marginal_cost(self, j: int) -> Optional[float]:
        """What extending the chain from ``j-1`` to ``j`` costs, in ms/round.

        The measured difference between the two adjacent rungs where both were
        measured -- that is the honest number, and it is what exposed the K=0
        step as the expensive one. Where the pair is not available, the slope
        of the affine fit over the VERIFY rungs stands in.
        """
        if j <= 0:
            return None
        lo = self._round_ms.get(j - 1) if self._round_n.get(j - 1, 0) > 0 else None
        hi = self._round_ms.get(j) if self._round_n.get(j, 0) > 0 else None
        if lo is not None and hi is not None:
            return hi - lo
        fit = self._cost_fit(min_rung=1)
        if fit is None:
            return None
        return fit[1]

    def marginal_gain(self, j: int) -> Optional[float]:
        """What extending the chain from ``j-1`` to ``j`` buys, in ms/round.

        One extra emitted token -- worth a whole decode step -- exactly when
        every proposal up to and including the new one is accepted.
        """
        t = self.t_decode
        reach = self.reach_probability(j)
        if t is None or reach is None:
            return None
        return reach * t

    def marginal_depth(self) -> Optional[int]:
        """The depth at which the margin flips: the largest K worth running.

        Walks OUTWARD from the shortest chain and stops at the first step that
        does not pay, which is what makes it a marginal rule rather than an
        average one: a later step cannot rescue an earlier one, because the
        later step is only reached if the earlier one was accepted.
        """
        t = self.t_decode
        if t is None:
            return None
        depth = 0
        for j in range(1, max(self.rungs) + 1):
            gain = self.marginal_gain(j)
            cost = self.marginal_cost(j)
            if gain is None or cost is None:
                break
            if gain <= cost * (1.0 + self.margin):
                break
            depth = j
        return depth

    def _rung_at_or_below(self, depth: int, candidates: Sequence[int]) -> int:
        """Round the flip point DOWN to an available rung.

        Down, not nearest: with a ladder of {0, 1, 3} and a flip at depth 2,
        rung 3 would run two chain steps the margin has already rejected, so
        the answer is 1. Only when every rung is above the flip does the
        shortest one win by default.
        """
        below = [k for k in candidates if k <= depth]
        return max(below) if below else min(candidates)

    # -- decision ------------------------------------------------------

    def candidate_rungs(self, ctx: Optional[Dict[str, Any]] = None) -> Tuple[int, ...]:
        """Which rungs this round may choose between.

        The extension point round 7b plugs into: turn routing decides which
        ALGORITHM a request runs, and an algorithm may not offer every rung.
        Today every rung on the ladder is a candidate; a ctx carrying an
        explicit ``rungs`` restriction is honoured so the caller in 7b does not
        have to reach into this object's state.
        """
        if ctx:
            allowed = ctx.get("rungs")
            if allowed:
                keep = tuple(k for k in self.rungs if k in set(allowed))
                if keep:
                    return keep
        return self.rungs

    def choose(self, ctx: Optional[Dict[str, Any]] = None) -> int:
        """The rung for the NEXT round.

        Three regimes, in order, and the order is the point: pinned beats
        probing beats adapting, so a fixed-rung measurement is never disturbed
        by the policy and an adaptive run never decides on numbers it has not
        measured.
        """
        candidates = self.candidate_rungs(ctx)
        if ctx and ctx.get("rung") is not None:
            # A pin is honoured even off-ladder: pinning a rung that was never
            # captured is a legitimate measurement (it isolates the graph from
            # the chain length), it stays correct because the lane falls back
            # to the eager verify, and reporting says which happened. What an
            # off-ladder pin must NOT do is become the policy's resting state,
            # so ``current`` only follows a pin that is on the ladder.
            pinned = int(ctx["rung"])
            if pinned in candidates:
                self.current = pinned
            self._last_reason = "pinned"
            return pinned
        if not self.adaptive:
            if self.current not in candidates:
                self.current = candidates[-1]
            self._last_reason = "static"
            return self.current

        # PROBE: every rung has to have been measured in THIS boot before any
        # of them is compared against another. Round-robin over the rungs that
        # are still short of their probe quota; the widest first, so the run
        # starts on the configured operating point rather than on K=0.
        short = [
            k
            for k in sorted(candidates, reverse=True)
            if self._round_n.get(k, 0) < self.probe_rounds
        ]
        if short:
            self.current = short[0]
            self._last_reason = "probe"
            return self.current

        depth = self.marginal_depth()
        if depth is None:
            self._last_reason = "no_estimate"
            return self.current
        best = self._rung_at_or_below(depth, candidates)
        if best == self.current:
            self._pending = None
            self._pending_n = 0
            self._last_reason = "hold"
            return self.current
        if self._pending == best:
            self._pending_n += 1
        else:
            self._pending = best
            self._pending_n = 1
        if self._pending_n >= self.hysteresis:
            self.current = best
            self._pending = None
            self._pending_n = 0
            self._switches += 1
            self._last_reason = "switch"
        else:
            self._last_reason = "hold_hysteresis"
        return self.current

    # -- reporting -----------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "rungs": list(self.rungs),
            "adaptive": self.adaptive,
            "current": self.current,
            "rounds": self._rounds,
            "switches": self._switches,
            "hysteresis": self.hysteresis,
            "pending": self._pending,
            "pending_rounds": self._pending_n,
            "reason": self._last_reason,
            "p_accept": (None if self.p_accept is None else round(self.p_accept, 4)),
            "round_ms": {k: round(v, 3) for k, v in sorted(self._round_ms.items())},
            "round_n": {k: v for k, v in sorted(self._round_n.items())},
            "predicted_accept": {
                k: round(self.predicted_accept(k), 3) for k in self.rungs
            },
            "predicted_ms_per_token": {
                k: (None if v is None else round(v, 3))
                for k in self.rungs
                if (v := self.predicted_ms_per_token(k)) is not None or True
            },
            # The marginal criterion, itemised -- the numbers the decision was
            # actually made on, so a table row can be argued with instead of
            # taken on trust.
            "t_decode_ms": (None if self.t_decode is None else round(self.t_decode, 3)),
            "marginal_depth": self.marginal_depth(),
            "position_accept": {
                j: round(p, 4)
                for j in sorted(self._pos_reached)
                if (p := self.position_accept(j)) is not None
            },
            "reach_probability": {
                j: (None if r is None else round(r, 4))
                for j in range(1, max(self.rungs) + 1)
                if (r := self.reach_probability(j)) is not None or True
            },
            "marginal_gain_ms": {
                j: (None if g is None else round(g, 3))
                for j in range(1, max(self.rungs) + 1)
                if (g := self.marginal_gain(j)) is not None or True
            },
            "marginal_cost_ms": {
                j: (None if c is None else round(c, 3))
                for j in range(1, max(self.rungs) + 1)
                if (c := self.marginal_cost(j)) is not None or True
            },
        }
