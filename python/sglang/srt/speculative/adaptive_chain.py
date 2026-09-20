"""Per-round adaptive draft chain length for topk=1 speculative decoding.

This complements :mod:`adaptive_spec_params`, which picks a chain length from
the *backward-looking* EMA of observed acceptance lengths and switches only
every ``update_interval`` batches.  The policy here is *forward-looking*: it
reads the draft model's own top-1 confidence and picks, for the next round, the
chain length ``k`` that maximises expected accepted tokens per millisecond.

Model
-----
For a topk=1 chain the draft is a sequence of greedy steps.  Let

    survival[i] = P(draft steps 1..i+1 all get accepted)

i.e. the cumulative product of the per-step top-1 probabilities.  Running a
chain of length ``k`` and verifying ``k + 1`` rows yields

    expected accepted tokens  E(k) = 1 + sum(survival[:k])
    round cost                C(k) = verify_ms(k + 1 rows) + k * draft_ms

(the leading ``1`` is the bonus token, which a verify always emits).  The
throughput-optimal chain length is

    k* = argmax_{k_min <= k <= k_max}  E(k) / C(k)

Both halves are deliberately separated: :func:`choose_chain_length` is a pure
function over a survival vector and a cost callable, and :class:`ChainCostModel`
is the cost callable — seeded from a prior and refined online from observed
round durations.  Neither touches CUDA, so both are fully testable off-GPU; the
caller is responsible for measuring the durations it feeds in (CUDA events, no
sync in the hot path).

What the 27B A/B of 2026-09-20 taught this file
-----------------------------------------------
Arm ab27a ran this policy over ``k in [1..5]`` and chose ``k=5`` in 62 % of 400
rounds; the fixed-``k=3`` arm ab27b was 4 % faster on the same maths prompt
(97.4 vs 93.4 tok/s) at a *lower* acceptance length (3.08 vs 3.31).  The policy
was maximising the right expression and being fed two wrong inputs, both of
which were assumptions standing in for measurements the boot actually had:

* **the survival tail.**  ``k`` is chosen before the draft, so a round that
  runs ``k=2`` only ever measures two survival entries; the rest were filled by
  repeating the last one.  A flat tail says every further step is accepted with
  probability 1.0, which makes the score rise monotonically in ``k`` and pins
  the argmax at ``k_max`` regardless of the curve.  See
  :func:`normalize_survival`.
* **the cost of the lengths not being run.**  The per-``k`` EMA only warms for
  lengths the policy picks, so the others keep the static prior forever — and
  the prior's shape (26.0 + 2.5k) was flatter and higher than the measured one
  (21.1 + 3.19k), pricing the unmeasured ``k=3`` at 33.5 ms against a measured
  bracket of 30.7 ms.  See :meth:`ChainCostModel.fit`.

With both replaced by the boot's own numbers the same state scores ``k=3``
highest, which is the arm that actually won.  The lesson generalises past this
file: a regulator that measures its incumbent and assumes about its
alternatives will keep its incumbent.

Nothing here runs unless ``SGLANG_SPEC_ADAPTIVE_CHAIN`` is set (and it also
requires ``--speculative-adaptive``, which owns the per-chain-length runtime
states this selects between); see ``EagleDraftWorker._init_adaptive_chain_probe``
and ``EAGLEWorkerV2._arm_chain_policy`` for the wiring.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

#: Prior used when ``SGLANG_SPEC_ADAPTIVE_CHAIN_COST_MS`` is unset or unparseable.
#: Measured on Qwen3.8 Next Flash, 32k form, bs=1, CUDA graphs: a NEXTN round
#: with 2 draft steps costs ~28.6 ms GPU, and the verify dominates it — the
#: cost is nearly flat in the number of verified rows, so a long chain is
#: almost free once the verify is paid for.
DEFAULT_DRAFT_MS = 2.5
DEFAULT_VERIFY_MS = 26.0

#: Survival values are rounded to this many digits before the argmax, so that
#: ULP-level differences between TP ranks cannot select different chain lengths.
SURVIVAL_QUANTUM_DIGITS = 4


def _sanitize_survival(value: float) -> float:
    """Map one raw survival entry into ``[0.0, 1.0]``.

    NaN and -inf/+inf are treated as "no evidence" (0.0) rather than
    propagating into the score, where a single NaN would poison every
    comparison and make the argmax order-dependent.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def survival_tail_ratio(curve: Sequence[float]) -> float:
    """Per-step acceptance ratio to continue *curve* with, in ``[0, 1]``.

    A survival curve is a cumulative product, so the natural continuation is
    another factor, not another copy of the last value.  The factor is read off
    the curve's own last step (``curve[-1] / curve[-2]``); with a single entry
    the cumulative product *is* one per-step probability, so that entry is the
    ratio.  An empty or zeroed curve continues with 0.0.
    """
    cleaned = [_sanitize_survival(v) for v in curve]
    if not cleaned:
        return 0.0
    if len(cleaned) == 1:
        return min(1.0, max(0.0, cleaned[0]))
    prev, last = cleaned[-2], cleaned[-1]
    if prev <= 0.0:
        return 0.0
    return min(1.0, max(0.0, last / prev))


def normalize_survival(
    survival: Sequence[float], k_max: int, extend: str = "geometric"
) -> list[float]:
    """Clean a raw survival vector and extend it to ``k_max`` entries.

    Three properties matter and are asserted by the tests:

    * Entries are clamped to ``[0, 1]`` and NaN/inf become 0.0.
    * The vector is made non-increasing.  A survival curve is a cumulative
      product of probabilities, so it can only fall; a rise means the producer
      handed us noise, and taking the running minimum is the cheapest repair
      that keeps the curve interpretable.
    * A vector shorter than ``k_max`` is extended by CONTINUING ITS DECAY
      (``extend="geometric"``, the default): each missing entry is the previous
      one times :func:`survival_tail_ratio`.  The vector is short exactly when
      the previous round ran a short chain, so the tail is the part the policy
      has no measurement for and has to assume something about.

      The assumption is the whole ballgame.  ``extend="flat"`` -- repeating the
      last value, which this function used to do unconditionally -- asserts that
      every further draft step is accepted with probability 1.0.  That is the
      most optimistic statement available, and it is not a mild bias: with a
      flat tail each unmeasured step adds a *constant* ``s`` tokens for a
      marginal cost of one draft forward, so the score ``(1 + sum) / cost``
      rises monotonically past the last measured index and the argmax is
      ALWAYS ``k_max``.  Measured on the 27B A/B of 2026-09-20 (boot ab27a):
      a k=2 round reported ``[0.472, 0.343]``, the pad turned it into
      ``[0.472, 0.343, 0.343, 0.343, 0.343]``, and k=5 won 62 % of 400 rounds
      while the fixed-k=3 arm was 4 % faster on the same prompt.  Continuing
      the measured decay (ratio 0.727) instead turns the same curve into
      ``[0.472, 0.343, 0.249, 0.181, 0.132]`` and the argmax becomes k=3.

      Geometric continuation still lets the policy climb: a genuinely confident
      draft has a ratio near 1.0 and its tail stays high, so this does not
      re-introduce the ``k_min`` lock-in that the flat pad was defending
      against -- it only stops asserting confidence nobody measured.
    """
    if k_max <= 0:
        return []

    cleaned: list[float] = []
    running_min = 1.0
    for raw in list(survival)[:k_max]:
        # Quantise before comparing.  Every TP rank runs this policy on its own
        # copy of the curve and must land on the same k, or the ranks replay
        # different CUDA graphs and the next collective deadlocks.  The curve
        # comes from the same logits the existing code already takes `argmax`
        # over — if those diverged across ranks the draft tokens would already
        # differ today — so this is a second line of defence, not the argument:
        # it absorbs ULP-level noise, it does not make divergence impossible
        # (two ranks straddling a quantisation boundary still split).
        v = round(_sanitize_survival(raw), SURVIVAL_QUANTUM_DIGITS)
        running_min = min(running_min, v)
        cleaned.append(running_min)

    if not cleaned:
        return [0.0] * k_max

    if len(cleaned) < k_max:
        ratio = 1.0 if extend == "flat" else survival_tail_ratio(cleaned)
        while len(cleaned) < k_max:
            cleaned.append(round(cleaned[-1] * ratio, SURVIVAL_QUANTUM_DIGITS))
    return cleaned


def eligible_candidates(
    k_max: int, k_min: int = 1, candidates: Sequence[int] | None = None
) -> list[int]:
    """Chain lengths the policy may return, sorted ascending.

    *candidates* restricts the search to lengths that actually have a runtime
    state built for them (``AdaptiveController`` only captures CUDA graphs for
    the configured ``candidate_steps``, and activating an unbuilt length raises).
    ``None`` means every length in ``[k_min, k_max]`` is available.
    """
    if k_max <= 0:
        return []
    k_min = max(0, min(int(k_min), int(k_max)))
    if candidates is None:
        return list(range(max(1, k_min), int(k_max) + 1))
    return sorted(
        {int(c) for c in candidates if k_min <= int(c) <= int(k_max) and int(c) >= 1}
    )


def choose_chain_length(
    survival: Sequence[float],
    cost_ms: Callable[[int], float],
    k_max: int,
    k_min: int = 1,
    candidates: Sequence[int] | None = None,
) -> int:
    """Pick the chain length maximising expected accepted tokens per ms.

    Args:
        survival: ``survival[i]`` = probability that draft steps ``1..i+1`` are
            all accepted (cumulative product of per-step top-1 confidences).
            May be shorter than *k_max*, empty, or contain NaN — see
            :func:`normalize_survival`.
        cost_ms: ``cost_ms(k)`` = estimated cost of a round with chain length
            *k*, in milliseconds.  Must be positive; a non-positive or
            non-finite value makes that *k* ineligible rather than crashing.
        k_max: largest admissible chain length (``--speculative-num-steps``).
        k_min: smallest admissible chain length.
        candidates: optional whitelist of chain lengths that have a runtime
            state; see :func:`eligible_candidates`.

    Returns:
        The chosen ``k``, always an eligible candidate (or *k_max* when none is).

    Degenerate inputs fall back to the largest eligible candidate, i.e. to
    today's static behaviour: an empty survival vector (no measurement yet) and
    a cost model that rejects every candidate both mean "no information", and
    the safe answer to that is the configured chain length, not a silently
    shortened one.
    """
    if k_max <= 0:
        return 0
    k_max = int(k_max)
    allowed = eligible_candidates(k_max, k_min, candidates)
    if not allowed:
        return k_max

    if not len(survival):
        # No confidence measured yet (first round after a state switch).
        return allowed[-1]

    curve = normalize_survival(survival, k_max)
    allowed_set = set(allowed)

    best_k = None
    best_score = -math.inf
    cumulative = 0.0
    for k in range(1, k_max + 1):
        cumulative += curve[k - 1]
        if k not in allowed_set:
            continue
        try:
            cost = float(cost_ms(k))
        except (TypeError, ValueError):
            continue
        if math.isnan(cost) or math.isinf(cost) or cost <= 0.0:
            continue
        score = (1.0 + cumulative) / cost
        # Strict ">" keeps the *smallest* k on a tie: identical throughput at a
        # shorter chain means lower per-round latency and less wasted draft.
        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        return allowed[-1]
    return best_k


def expected_tokens(survival: Sequence[float], k: int) -> float:
    """Expected accepted tokens for a chain of length *k*: ``1 + sum(survival[:k])``.

    The leading 1 is the bonus token a verify always emits, so the value is
    >= 1.0 even for an empty curve.
    """
    if k <= 0:
        return 1.0
    return 1.0 + float(sum(_sanitize_survival(v) for v in list(survival)[:k]))


def switch_is_profitable(
    e_incumbent: float,
    c_incumbent: float,
    e_candidate: float,
    c_candidate: float,
    swap_ms: float,
    dwell_rounds: int,
    margin: float = 0.0,
) -> bool:
    """Does moving to *candidate* pay for the state swap it costs?

    The adaptive ladder keeps one runtime state mapped at a time, so changing
    the chain length can cost a physical graph-memory swap (unmap the outgoing
    state's pages, map the incoming one).  The per-round argmax in
    :func:`choose_chain_length` is blind to that: it compares steady-state
    throughputs and will happily pay a swap for a gain it holds for one round.

    Measured on the Qwen3.8 Next Flash 32k form (boot fn8s4, 2026-09-20): 1270
    swaps at a mean 33.46 ms each, against a ~38 ms decode round -- the policy
    bought a real acceptance gain (3.39 vs 2.69 tokens) and still lost 10 %
    throughput, because a swap costs most of a round and it swapped on 0.6 of
    them.  That boot is the reason this gate exists.

    The comparison is a throughput one, with the swap charged once and
    amortised over the rounds the new state is actually held:

        incumbent:  e_inc / c_inc                        tokens per ms
        candidate:  dwell * e_new / (dwell * c_new + swap_ms)

    Returns True only when the candidate's amortised rate beats the incumbent's
    by more than *margin* (a fraction, so ``0.05`` demands 5 %).  ``swap_ms ==
    0`` (nothing to pay, e.g. the target state is already resident) reduces it
    to the plain rate comparison, and a non-positive dwell means "never hold
    it", which can never pay.

    The margin is hysteresis, and it is what keeps a swap-free ladder honest.
    With every candidate state resident (``plan_residency``) the swap term is
    0.0 and this gate degenerates into a bare ``>`` between two estimates that
    are an EMA and a one-round-old curve -- so two lengths of near-equal
    throughput trade places on noise, and every trade still costs a graph
    switch and a collective.  Requiring a named improvement means a switch is
    only taken when the estimate says something the noise cannot.

    A margin is not free and must stay well under the effect size: on the 27B
    A/B of 2026-09-20 the entire spread between the best and the worst chain
    length was 4.5 %, so a 5 % margin -- which reads as cautious -- suppresses
    the correction the regulator exists to make.  See
    ``SGLANG_SPEC_ADAPTIVE_CHAIN_SWITCH_MARGIN_PCT``.
    """
    try:
        e_inc = float(e_incumbent)
        c_inc = float(c_incumbent)
        e_new = float(e_candidate)
        c_new = float(c_candidate)
        swap = max(0.0, float(swap_ms))
        dwell = int(dwell_rounds)
        rel = max(0.0, float(margin))
    except (TypeError, ValueError):
        return False
    if dwell <= 0:
        return False
    for v in (e_inc, c_inc, e_new, c_new, swap, rel):
        if not math.isfinite(v):
            return False
    if c_inc <= 0.0 or c_new <= 0.0:
        return False
    denom = dwell * c_new + swap
    if denom <= 0.0:
        return False
    return (dwell * e_new) / denom > (e_inc / c_inc) * (1.0 + rel)


def break_even_rounds(
    e_incumbent: float,
    c_incumbent: float,
    e_candidate: float,
    c_candidate: float,
    swap_ms: float,
) -> float:
    """Smallest dwell (in rounds) for which the switch pays, or ``inf``.

    Solving :func:`switch_is_profitable` for ``dwell``:

        dwell * e_new / (dwell * c_new + swap) > e_inc / c_inc
        dwell * (e_new * c_inc - e_inc * c_new) > e_inc * swap

    so the switch pays from ``e_inc * swap / (e_new * c_inc - e_inc * c_new)``
    rounds on.  A non-positive denominator means the candidate is not faster in
    steady state either: no dwell makes it profitable, hence ``inf``.

    Diagnostic counterpart to the boolean gate -- it is what a log line should
    print when a switch is suppressed, because it names the number the operator
    would have to believe about the workload's stability to want the switch.
    """
    try:
        e_inc = float(e_incumbent)
        c_inc = float(c_incumbent)
        e_new = float(e_candidate)
        c_new = float(c_candidate)
        swap = max(0.0, float(swap_ms))
    except (TypeError, ValueError):
        return math.inf
    for v in (e_inc, c_inc, e_new, c_new, swap):
        if not math.isfinite(v):
            return math.inf
    if c_inc <= 0.0 or c_new <= 0.0:
        return math.inf
    denom = e_new * c_inc - e_inc * c_new
    if denom <= 0.0:
        return math.inf
    if swap <= 0.0:
        return 0.0
    return (e_inc * swap) / denom


def parse_cost_ms(spec: str | None) -> tuple[float, float]:
    """Parse ``SGLANG_SPEC_ADAPTIVE_CHAIN_COST_MS`` (e.g. ``"draft:2.5,verify:26"``).

    Returns ``(draft_ms, verify_ms)``.  Unknown keys, malformed numbers and
    non-positive values are ignored with a warning and leave that half at its
    default — a typo in an env var must not take the server down, and must not
    silently install a zero cost that would make every candidate look free.
    """
    draft_ms = DEFAULT_DRAFT_MS
    verify_ms = DEFAULT_VERIFY_MS
    if not spec:
        return draft_ms, verify_ms

    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, raw = item.partition(":")
        if not sep:
            logger.warning(
                "SGLANG_SPEC_ADAPTIVE_CHAIN_COST_MS: ignoring %r (expected 'key:value')",
                item,
            )
            continue
        key = key.strip().lower()
        try:
            value = float(raw.strip())
        except ValueError:
            logger.warning(
                "SGLANG_SPEC_ADAPTIVE_CHAIN_COST_MS: ignoring %r (not a number)", item
            )
            continue
        if not math.isfinite(value) or value <= 0.0:
            logger.warning(
                "SGLANG_SPEC_ADAPTIVE_CHAIN_COST_MS: ignoring %r (must be > 0)", item
            )
            continue
        if key == "draft":
            draft_ms = value
        elif key == "verify":
            verify_ms = value
        else:
            logger.warning(
                "SGLANG_SPEC_ADAPTIVE_CHAIN_COST_MS: ignoring unknown key %r", key
            )
    return draft_ms, verify_ms


class ChainCostModel:
    """Cost of one decode round as a function of the chain length ``k``.

    Starts from the analytic prior ``verify_ms + k * draft_ms`` and replaces it,
    per ``k``, with an EMA over observed round durations once enough rounds with
    that ``k`` have been seen.  Per-``k`` rather than a refitted global
    ``(draft, verify)`` pair: the two-parameter fit assumes the round cost is
    affine in ``k``, and the measured near-flatness of the verify says it is not
    — an EMA per ``k`` makes no shape assumption at all.

    Candidates the policy rarely picks never accumulate observations, and the
    fallback for those is where the static prior went wrong.  A chain length
    that is never chosen is never measured, so it keeps whatever the prior said
    about it forever, while the incumbent's estimate becomes real -- a bandit
    lock-in in which the prior's *shape*, not the workload, decides which
    lengths stay in the running.  Measured on boot ab27a (2026-09-20,
    ``arm_ab27a.log`` round 400): ``c2=27.4ms/n118 c4=34.1ms/n32 c5=36.9ms/n248``
    were real, while ``c1=28.5ms/n0 c3=33.5ms/n0`` were still the prior
    ``26.0 + 2.5k``.  The prior's slope (2.5 ms/step) is well under the measured
    one (3.19 ms/step) and its intercept is well over (26.0 vs 21.1), so it
    overcharges every short unmeasured chain: it put k=3 at 33.5 ms when the
    measured neighbours bracket it at 30.7 ms.  k=3 -- the length the fixed-k
    arm won the same A/B with -- could not win at that price.

    So the fallback is itself measured wherever it can be: with observations at
    two or more distinct ``k``, an affine least-squares fit through the measured
    points replaces the static prior for the unmeasured ones (see :meth:`fit`).
    The static prior survives only as the cold-start answer, before two points
    exist.  This makes no claim the per-``k`` EMA does not already make -- it
    interpolates between measurements instead of between guesses.

    The caller supplies the durations; this class never reads a clock, so it is
    testable without CUDA.
    """

    def __init__(
        self,
        k_max: int,
        draft_ms: float = DEFAULT_DRAFT_MS,
        verify_ms: float = DEFAULT_VERIFY_MS,
        ema_alpha: float = 0.2,
        min_samples: int = 8,
    ):
        self.k_max = int(k_max)
        self.draft_ms = float(draft_ms)
        self.verify_ms = float(verify_ms)
        self.ema_alpha = float(ema_alpha)
        self.min_samples = int(min_samples)
        self._ema: dict[int, float] = {}
        self._counts: Counter[int] = Counter()

    def prior(self, k: int) -> float:
        """Analytic cold-start cost estimate, always finite and > 0."""
        return self.verify_ms + max(0, int(k)) * self.draft_ms

    def is_measured(self, k: int) -> bool:
        """Has *k* accumulated enough observations to speak for itself?"""
        k = int(k)
        return self._counts[k] >= self.min_samples and k in self._ema

    def measured_points(self) -> list[tuple[int, float]]:
        """``(k, cost)`` for every chain length with a warm EMA, ascending."""
        return sorted(
            (k, v) for k, v in self._ema.items() if self._counts[k] >= self.min_samples
        )

    def fit(self) -> tuple[float, float] | None:
        """Affine least-squares fit ``(intercept, slope)`` over measured points.

        ``None`` when fewer than two distinct ``k`` have been measured -- there
        is then no line to draw and the static prior stays in force.

        The slope is clamped at >= 0.  A round with more draft steps and more
        verify rows cannot be cheaper; a negative fitted slope is noise between
        two nearby points, and letting it through would price the longest chain
        below the shortest and hand the argmax straight to ``k_max`` -- the very
        failure this replaces.  The intercept is clamped so the fitted cost at
        ``k = 1`` stays positive, because a non-positive cost makes a candidate
        ineligible in :func:`choose_chain_length` rather than merely cheap.
        """
        points = self.measured_points()
        if len({k for k, _ in points}) < 2:
            return None
        n = len(points)
        mean_k = sum(k for k, _ in points) / n
        mean_c = sum(c for _, c in points) / n
        denom = sum((k - mean_k) ** 2 for k, _ in points)
        if denom <= 0.0:
            return None
        slope = sum((k - mean_k) * (c - mean_c) for k, c in points) / denom
        if not math.isfinite(slope):
            return None
        slope = max(0.0, slope)
        intercept = mean_c - slope * mean_k
        if not math.isfinite(intercept):
            return None
        # Keep cost(1) strictly positive without moving the (measured) slope.
        intercept = max(intercept, 1e-6 - slope)
        return intercept, slope

    def fallback(self, k: int) -> float:
        """Cost for an unmeasured *k*: the measured fit if one exists, else the
        static prior."""
        fitted = self.fit()
        if fitted is None:
            return self.prior(k)
        intercept, slope = fitted
        value = intercept + max(0, int(k)) * slope
        if not math.isfinite(value) or value <= 0.0:
            return self.prior(k)
        return value

    def observe(self, k: int, duration_ms: float) -> None:
        """Record one measured round duration for chain length *k*.

        Non-finite and non-positive durations are dropped: a CUDA event pair
        that was read before both events completed yields garbage, and one such
        sample must not be able to drag the EMA to a value that makes *k* look
        free forever.
        """
        try:
            value = float(duration_ms)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value) or value <= 0.0:
            return
        k = int(k)
        self._counts[k] += 1
        if k not in self._ema:
            self._ema[k] = value
        else:
            self._ema[k] = (1.0 - self.ema_alpha) * self._ema[
                k
            ] + self.ema_alpha * value

    def cost(self, k: int) -> float:
        """Best available cost estimate for *k*.

        In order: the measured EMA for *k*, else the affine fit through the
        other measured points, else the static cold-start prior.
        """
        k = int(k)
        if self._counts[k] >= self.min_samples and k in self._ema:
            return self._ema[k]
        return self.fallback(k)

    # Convenience so the model can be passed directly as ``cost_ms``.
    __call__ = cost

    def snapshot(self) -> dict[int, tuple[float, int]]:
        """``{k: (cost, n_samples)}`` for logging."""
        return {k: (self.cost(k), self._counts[k]) for k in range(1, self.k_max + 1)}


class AdaptiveChainPolicy:
    """Stateful wrapper: survival in, chain length out, plus a log histogram.

    One instance per worker.  ``record_survival`` is fed from the *previous*
    round — the confidences only exist once the draft has run, so the choice for
    round N is necessarily made from round N-1's curve.  That one-round lag is
    inherent to picking ``k`` before the draft, not an approximation taken for
    convenience: the alternative is a host sync mid-chain, which costs more than
    the adaptation saves.
    """

    def __init__(
        self,
        k_max: int,
        k_min: int = 1,
        cost_model: ChainCostModel | None = None,
        log_every: int = 0,
        candidates: Sequence[int] | None = None,
        min_dwell: int = 0,
        swap_ms: float = 0.0,
        swap_ms_for: Callable[[int], float] | None = None,
        consensus: Callable[[int], int] | None = None,
        switch_margin: float = 0.0,
    ):
        self.k_max = int(k_max)
        self.k_min = max(0, min(int(k_min), self.k_max))
        self.candidates = eligible_candidates(self.k_max, self.k_min, candidates)
        self.cost_model = cost_model or ChainCostModel(k_max=self.k_max)
        self.log_every = int(log_every)
        #: Rounds a chain length must be held before another switch may be
        #: considered at all. Floor under the break-even gate: it bounds the
        #: *switch rate* even while the cost/survival estimates are still cold
        #: (the gate needs a warm E and c to be meaningful, the dwell does not).
        self.min_dwell = max(0, int(min_dwell))
        #: Flat swap-cost prior in ms, used when *swap_ms_for* is absent.
        self.swap_ms = max(0.0, float(swap_ms))
        #: Optional per-target swap cost, ``swap_ms_for(k) -> ms``. The graph
        #: memory manager supplies it so an already-resident target costs 0 and
        #: only a genuine unmap/map is charged.
        self.swap_ms_for = swap_ms_for
        #: ``consensus(local_proposal) -> agreed`` -- makes the chain length
        #: identical across TP ranks. Supplied by the worker as a broadcast
        #: from rank 0 over the TP CPU group; None (the default) leaves the
        #: local proposal in force, which is correct for world_size 1 and for
        #: every off-GPU test.
        self.consensus = consensus
        #: Relative throughput improvement a candidate must show before the
        #: incumbent is given up. Hysteresis on top of the break-even gate,
        #: and the only brake that still bites once every candidate state is
        #: resident and the swap cost is honestly 0.
        self.switch_margin = max(0.0, float(switch_margin))
        self._survival: list[float] = []
        self._histogram: Counter[int] = Counter()
        self._rounds = 0
        self._current: int | None = None
        self._dwell = 0
        self._rounds_since_decision = 0
        self._switches = 0
        self._held_dwell = 0
        self._held_breakeven = 0
        self._consensus_overrides = 0
        self._warmup_switches = 0
        self._last_switch: dict[str, float] | None = None

    def record_survival(self, survival: Sequence[float]) -> None:
        self._survival = normalize_survival(survival, self.k_max)

    def record_duration(self, k: int, duration_ms: float) -> None:
        self.cost_model.observe(k, duration_ms)

    def _swap_cost(self, k: int) -> float:
        if self.swap_ms_for is None:
            return self.swap_ms
        try:
            value = float(self.swap_ms_for(k))
        except (TypeError, ValueError):
            return self.swap_ms
        if not math.isfinite(value) or value < 0.0:
            return self.swap_ms
        return value

    def _warmup_target(self) -> int | None:
        """The next candidate that has never been timed, or None when all are.

        A cost model that only learns about the length it is already running
        cannot be talked out of it.  Measured on boot ab27a and reproduced as a
        closed-loop test: the first choice is ``k_max`` (an empty survival
        curve means "no information", and the safe answer to that is the
        configured length), running ``k_max`` is the only thing that ever gets
        timed, every other length keeps a prior that overcharges it, so the
        argmax stays at ``k_max`` and no second measurement is ever produced.
        The fitted fallback in :meth:`ChainCostModel.fit` cannot break that on
        its own -- a line needs two points, and the loop only ever makes one.

        So the policy buys those points once, explicitly, before it starts
        optimising: each decision round during warm-up runs the
        lowest-numbered candidate that still has no measurement of its own.
        The cost is bounded and paid once per boot -- at most one dwell block
        per candidate, i.e. ~40 rounds (~1.2 s) for five candidates at the
        default dwell of 8 -- and once every candidate is warm this returns
        None forever and the policy is a pure argmax from then on.

        Rank-uniformity is unchanged.  *Which* rounds are decision rounds
        stays a pure function of the round counter, and the proposal made on
        one still passes through :attr:`consensus`.  Whether a given rank's
        CUDA-event drain has landed is rank-local -- it already was, which is
        why the broadcast exists; this adds no new kind of divergence, only
        another rank-local input to the same guarded proposal.
        """
        for k in self.candidates:
            if not self.cost_model.is_measured(k):
                return k
        return None

    def _take_warmup(self, target: int) -> int:
        """Adopt *target* for a dwell block, bypassing the break-even gate.

        The gate compares steady-state throughputs, and during warm-up at
        least one side of that comparison is a guess -- which is the whole
        reason this round is happening.  Refusing exploration on the strength
        of the estimate it exists to replace is the lock-in, restated.
        """
        if target == self._current:
            self._dwell += 1
            return target
        previous = self._current
        self._current = target
        self._dwell = 0
        if previous is not None:
            self._switches += 1
            self._warmup_switches += 1
            logger.info(
                "[spec-adaptive] chain warm-up k=%d->%d: no measured round cost "
                "for k=%d yet (have %s), timing it for %d rounds before the "
                "argmax is trusted | round=%d",
                previous,
                target,
                target,
                sorted(k for k, _ in self.cost_model.measured_points()),
                max(1, self.min_dwell),
                self._rounds,
            )
        return target

    def _gate(self, candidate: int) -> int:
        """Hold the incumbent unless *candidate* earns the swap it costs.

        Three outcomes, in order: same length (nothing to pay), still inside
        the minimum dwell (refuse without asking), or a break-even test against
        the measured swap cost. The incumbent is only replaced by the third.
        """
        current = self._current
        if current is None:
            self._current = candidate
            self._dwell = 0
            return candidate
        if candidate == current:
            self._dwell += 1
            return current
        if self._dwell < self.min_dwell:
            self._dwell += 1
            self._held_dwell += 1
            return current
        swap_ms = self._swap_cost(candidate)
        e_inc = expected_tokens(self._survival, current)
        c_inc = self.cost_model.cost(current)
        e_new = expected_tokens(self._survival, candidate)
        c_new = self.cost_model.cost(candidate)
        # A switch taken now is held for at least the dwell floor; with no
        # floor configured, charge the swap against a single round, which is
        # the pessimistic (and correct) reading of "may flip again next round".
        dwell_rounds = max(1, self.min_dwell)
        profitable = switch_is_profitable(
            e_incumbent=e_inc,
            c_incumbent=c_inc,
            e_candidate=e_new,
            c_candidate=c_new,
            swap_ms=swap_ms,
            dwell_rounds=dwell_rounds,
            margin=self.switch_margin,
        )
        if not profitable:
            self._dwell += 1
            self._held_breakeven += 1
            return current
        self._log_switch(current, candidate, e_inc, c_inc, e_new, c_new, swap_ms)
        self._current = candidate
        self._dwell = 0
        self._switches += 1
        return candidate

    def _log_switch(
        self,
        old: int,
        new: int,
        e_inc: float,
        c_inc: float,
        e_new: float,
        c_new: float,
        swap_ms: float,
    ) -> None:
        """One named line per taken switch: who it replaced, and what for.

        Without this the only evidence a boot leaves behind is the periodic
        histogram, which says how often each length was *held* and nothing at
        all about why the length changed -- so a regulator that thrashes and
        one that adapts read identically in the log.  The expected gain is the
        number the switch was taken on: quote it back and a later boot can be
        checked against it instead of re-derived.
        """
        rate_inc = e_inc / c_inc if c_inc > 0 else float("nan")
        rate_new = e_new / c_new if c_new > 0 else float("nan")
        gain_pct = (
            100.0 * (rate_new / rate_inc - 1.0)
            if rate_inc and math.isfinite(rate_inc) and rate_inc > 0
            else float("nan")
        )
        self._last_switch = {
            "from": int(old),
            "to": int(new),
            "gain_pct": gain_pct,
            "swap_ms": float(swap_ms),
        }
        logger.info(
            "[spec-adaptive] chain switch k=%d->%d: steady-state %.4f->%.4f tok/ms "
            "(%+.1f%%) | E %.3f->%.3f tokens, cost %.1f->%.1f ms | swap=%.1f ms "
            "amortised over dwell=%d (break-even %.1f rounds, margin %.0f%%) | "
            "round=%d switches=%d",
            old,
            new,
            rate_inc,
            rate_new,
            gain_pct,
            e_inc,
            e_new,
            c_inc,
            c_new,
            swap_ms,
            max(1, self.min_dwell),
            break_even_rounds(e_inc, c_inc, e_new, c_new, swap_ms),
            100.0 * self.switch_margin,
            self._rounds,
            self._switches + 1,
        )

    def choose(self) -> int:
        """The chain length for this round.  Call exactly once per round.

        Two things make the answer the same on every TP rank, which it must be
        -- a divergent ``k`` replays different CUDA graphs and deadlocks the
        next collective (boot fn8s4, round 859).

        First, the decision happens only on *decision rounds*, and which rounds
        those are is a pure function of the round counter, which every rank
        advances in lockstep (one call per scheduler batch).  In between, the
        held length is re-affirmed without consulting any estimate at all.

        Second, on a decision round the locally-proposed length is passed
        through :attr:`consensus` before it is adopted.  That is the only
        sound construction available here: the proposal is derived from a
        cost EMA over CUDA-event timings and from whether a non-blocking D2H
        copy has landed, and *both are rank-local by nature*.  Quantising the
        survival curve (as :func:`normalize_survival` does) narrows the window
        but cannot close it -- two ranks straddling a quantisation boundary
        still split.  Agreeing on one rank's answer closes it by construction,
        for the price of one small collective per decision round.
        """
        self._rounds += 1
        if self._current is not None and self._rounds_since_decision < max(
            1, self.min_dwell
        ):
            # Frozen: no estimate is read, so nothing rank-local can leak into
            # the answer, and no collective is posted.
            self._rounds_since_decision += 1
            # The dwell clock runs during the freeze too. It must: _gate reads
            # it to decide whether the held length has been kept long enough,
            # and if only decision rounds advanced it, it could never reach
            # min_dwell and the policy would hold its first choice forever.
            self._dwell += 1
            self._histogram[self._current] += 1
            self._maybe_log()
            return self._current

        warmup = self._warmup_target()
        if warmup is not None:
            proposal = self._take_warmup(warmup)
        else:
            raw = choose_chain_length(
                self._survival,
                self.cost_model.cost,
                k_max=self.k_max,
                k_min=self.k_min,
                candidates=self.candidates,
            )
            proposal = self._gate(raw)
        k = self._agree(proposal)
        if k != self._current:
            # Consensus overrode this rank's gate outcome; adopt it wholesale
            # so the residency bookkeeping and the dwell clock stay truthful.
            self._current = k
            self._dwell = 0
        self._rounds_since_decision = 1
        self._histogram[k] += 1
        self._maybe_log()
        return k

    def _agree(self, proposal: int) -> int:
        """Run *proposal* through the consensus hook, falling back to it."""
        if self.consensus is None:
            return proposal
        try:
            agreed = int(self.consensus(proposal))
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "[spec-adaptive] chain-length consensus failed; keeping the "
                "local proposal. Ranks may now disagree.",
                exc_info=True,
            )
            return proposal
        if agreed not in self.candidates:
            logger.warning(
                "[spec-adaptive] consensus returned k=%s which is not a built "
                "candidate %s; keeping local proposal k=%s.",
                agreed,
                self.candidates,
                proposal,
            )
            return proposal
        if agreed != proposal:
            self._consensus_overrides += 1
        return agreed

    def _maybe_log(self) -> None:
        if self.log_every > 0 and self._rounds % self.log_every == 0:
            self.log_histogram()

    @property
    def current(self) -> int | None:
        """The chain length currently held, or None before the first choose."""
        return self._current

    def log_histogram(self) -> None:
        total = sum(self._histogram.values()) or 1
        hist = " ".join(
            f"k={k}:{self._histogram[k]}({100.0 * self._histogram[k] / total:.0f}%)"
            for k in sorted(self._histogram)
        )
        costs = " ".join(
            f"c{k}={c:.1f}ms/n{n}" for k, (c, n) in self.cost_model.snapshot().items()
        )
        surv = " ".join(f"{s:.3f}" for s in self._survival)
        fitted = self.cost_model.fit()
        fit = (
            f"fit={fitted[0]:.1f}+{fitted[1]:.2f}k"
            if fitted is not None
            else f"fit=none(prior {self.cost_model.verify_ms:.1f}"
            f"+{self.cost_model.draft_ms:.2f}k)"
        )
        logger.info(
            "[spec-adaptive] rounds=%d %s | %s %s | survival=[%s] tail_ratio=%.3f | "
            "switches=%d held(dwell=%d,breakeven=%d) dwell=%d/%d margin=%.0f%% k=%s",
            self._rounds,
            hist,
            costs,
            fit,
            surv,
            survival_tail_ratio(self._survival),
            self._switches,
            self._held_dwell,
            self._held_breakeven,
            self._dwell,
            self.min_dwell,
            100.0 * self.switch_margin,
            self._current,
        )

    @property
    def last_switch(self) -> dict[str, float] | None:
        """The most recent taken switch, as logged, or None."""
        return self._last_switch

    @property
    def switch_stats(self) -> dict[str, int]:
        """Switches taken and suppressed -- the numbers that say whether the
        swap-amortisation gate is doing anything on this workload."""
        return {
            "switches": self._switches,
            "warmup_switches": self._warmup_switches,
            "held_dwell": self._held_dwell,
            "held_breakeven": self._held_breakeven,
            "rounds": self._rounds,
        }

    @property
    def warming_up(self) -> bool:
        """True while some candidate still has no measured round cost."""
        return self._warmup_target() is not None

    @property
    def histogram(self) -> dict[int, int]:
        return dict(self._histogram)


class RoundCostProbe:
    """Measures the wall duration of a decode round, tagged with its ``k``.

    Follows ``sglang.srt.utils.device_timer``: record a CUDA event pair around
    the round, queue it, and read ``elapsed_time`` only after ``query()`` says
    the end event has completed.  Nothing here ever synchronises, so a round's
    cost lands one or more rounds later — which is fine, it feeds an EMA.

    On a non-CUDA device it falls back to ``perf_counter`` so the wiring is
    exercisable off-GPU.
    """

    def __init__(self, device: str = "cpu", max_pending: int = 64):
        self._is_cuda = str(device).startswith("cuda")
        self._pending: list = []
        self._open: tuple | None = None
        self.max_pending = int(max_pending)

    def begin(self, k: int) -> None:
        import time

        import torch

        if self._open is not None:
            # Previous round never closed (an exception unwound past end()).
            # Drop it rather than pairing mismatched events.
            self._open = None
        if self._is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._open = (int(k), start)
        else:
            self._open = (int(k), time.perf_counter())

    def end(self) -> None:
        import time

        import torch

        if self._open is None:
            return
        k, start = self._open
        self._open = None
        if self._is_cuda:
            stop = torch.cuda.Event(enable_timing=True)
            stop.record()
            self._pending.append((k, start, stop))
            # Bound the queue: if events stop completing, drop the oldest
            # rather than growing without limit.
            if len(self._pending) > self.max_pending:
                del self._pending[0]
        else:
            self._pending.append((k, (time.perf_counter() - start) * 1000.0, None))

    def drain(self, sink: Callable[[int, float], None]) -> int:
        """Hand every completed measurement to *sink*; return how many."""
        drained = 0
        while self._pending:
            k, start, stop = self._pending[0]
            if stop is None:
                self._pending.pop(0)
                sink(k, start)
                drained += 1
                continue
            if not stop.query():
                break
            self._pending.pop(0)
            sink(k, start.elapsed_time(stop))
            drained += 1
        return drained


class SurvivalProbe:
    """Accumulates the survival curve on-device and reads it back without sync.

    Layout is ``[max_bs, k_max]`` and is allocated **once**, for the largest
    candidate chain length, and never reallocated.  That is a hard requirement,
    not tidiness: the draft CUDA graph is captured against this tensor's
    address, so a reallocation after a capture would leave the replayed graph
    writing into freed memory.  ``_rebuild_topk1_chain_buffers`` may reallocate
    its own constants because those are only ever *read* as graph outputs; this
    one is *written* inside the graph.

    Column ``j`` holds the cumulative product up to draft step ``j + 1``.
    Column 0 is written outside the graph (by draft-extend, which samples the
    first draft token); columns ``1..k-1`` are written inside the draft graph,
    each reading column ``j - 1``.

    Readout is the ``device_timer`` discipline: copy into pinned host memory
    with ``non_blocking=True``, record an event, and read the host buffer only
    once ``event.query()`` is true.  The value therefore lags one round — which
    is inherent anyway, since ``k`` must be fixed before the draft that would
    produce this round's confidences.
    """

    def __init__(self, max_bs: int, k_max: int, device: str = "cpu"):
        import torch

        self.k_max = int(k_max)
        self.max_bs = int(max_bs)
        self.device = device
        self._is_cuda = str(device).startswith("cuda")
        self._buf = torch.zeros(
            (self.max_bs, self.k_max), dtype=torch.float32, device=device
        )
        host = torch.zeros(self.k_max, dtype=torch.float32, device="cpu")
        self._host = host.pin_memory() if self._is_cuda else host
        self._event = None
        self._pending_k = 0

    @property
    def buffer(self):
        return self._buf

    def write_step(self, column: int, step_p, batch_size: int) -> None:
        """Write the cumulative survival for draft step ``column + 1``.

        *step_p* is this step's top-1 probability, shape ``[bs]`` or ``[bs, 1]``.
        Graph-safe: pure tensor ops into a pre-allocated buffer, no host sync
        and no allocation.
        """
        if column < 0 or column >= self.k_max:
            return
        bs = min(int(batch_size), self.max_bs)
        if bs <= 0:
            return
        p = step_p.reshape(-1)[:bs].float()
        if column == 0:
            self._buf[:bs, 0] = p
        else:
            self._buf[:bs, column] = self._buf[:bs, column - 1] * p

    def start_readout(self, batch_size: int, k: int) -> None:
        """Kick off the non-blocking copy of the batch-mean survival curve.

        The mean over requests is the right reduction: the round's cost is paid
        once for the whole batch, so the quantity to maximise is the *summed*
        expected acceptance, and the argmax of a sum over a shared cost is the
        argmax of the mean.
        """
        import torch

        bs = min(int(batch_size), self.max_bs)
        k = max(0, min(int(k), self.k_max))
        if bs <= 0 or k <= 0:
            self._pending_k = 0
            return
        mean = self._buf[:bs, :k].mean(dim=0)
        self._host[:k].copy_(mean, non_blocking=self._is_cuda)
        if self._is_cuda:
            event = torch.cuda.Event()
            event.record()
            self._event = event
        self._pending_k = k

    def poll(self) -> list[float] | None:
        """Return the survival curve if the copy has landed, else ``None``.

        Never blocks: an unfinished copy yields ``None`` and the caller keeps
        the previous curve.
        """
        if self._pending_k <= 0:
            return None
        if self._is_cuda:
            event = self._event
            if event is None or not event.query():
                return None
        k = self._pending_k
        self._pending_k = 0
        self._event = None
        return self._host[:k].tolist()
