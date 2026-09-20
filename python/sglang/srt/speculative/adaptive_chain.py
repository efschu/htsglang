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


def normalize_survival(survival: Sequence[float], k_max: int) -> list[float]:
    """Clean a raw survival vector and extend it to ``k_max`` entries.

    Two properties matter and are asserted by the tests:

    * Entries are clamped to ``[0, 1]`` and NaN/inf become 0.0.
    * The vector is made non-increasing.  A survival curve is a cumulative
      product of probabilities, so it can only fall; a rise means the producer
      handed us noise, and taking the running minimum is the cheapest repair
      that keeps the curve interpretable.
    * A vector shorter than ``k_max`` is extended by repeating its last value.
      This is deliberately optimistic: the vector is short exactly when the
      previous round ran a *short* chain, so filling with 0.0 would make a long
      chain permanently unattractive and lock the policy into ``k_min``.
      Flat extension keeps the door open for the policy to probe upward.
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

    while len(cleaned) < k_max:
        cleaned.append(cleaned[-1])
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

    Returns True only when the candidate's amortised rate strictly beats the
    incumbent's.  ``swap_ms == 0`` (nothing to pay, e.g. the target state is
    already resident) reduces it to the plain rate comparison, and a
    non-positive dwell means "never hold it", which can never pay.
    """
    try:
        e_inc = float(e_incumbent)
        c_inc = float(c_incumbent)
        e_new = float(e_candidate)
        c_new = float(c_candidate)
        swap = max(0.0, float(swap_ms))
        dwell = int(dwell_rounds)
    except (TypeError, ValueError):
        return False
    if dwell <= 0:
        return False
    for v in (e_inc, c_inc, e_new, c_new, swap):
        if not math.isfinite(v):
            return False
    if c_inc <= 0.0 or c_new <= 0.0:
        return False
    denom = dwell * c_new + swap
    if denom <= 0.0:
        return False
    return (dwell * e_new) / denom > (e_inc / c_inc)


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

    Candidates the policy rarely picks never accumulate observations, which is
    exactly why the prior has to stay available as a fallback instead of being
    overwritten.

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
        """Analytic cost estimate, always finite and > 0."""
        return self.verify_ms + max(0, int(k)) * self.draft_ms

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
        """Best available cost estimate for *k*: EMA when warm, else prior."""
        k = int(k)
        if self._counts[k] >= self.min_samples and k in self._ema:
            return self._ema[k]
        return self.prior(k)

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
        profitable = switch_is_profitable(
            e_incumbent=expected_tokens(self._survival, current),
            c_incumbent=self.cost_model.cost(current),
            e_candidate=expected_tokens(self._survival, candidate),
            c_candidate=self.cost_model.cost(candidate),
            swap_ms=swap_ms,
            # A switch taken now is held for at least the dwell floor; with no
            # floor configured, charge the swap against a single round, which
            # is the pessimistic (and correct) reading of "may flip again next
            # round".
            dwell_rounds=max(1, self.min_dwell),
        )
        if not profitable:
            self._dwell += 1
            self._held_breakeven += 1
            return current
        self._current = candidate
        self._dwell = 0
        self._switches += 1
        return candidate

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
        logger.info(
            "[spec-adaptive] rounds=%d %s | %s | survival=[%s] | "
            "switches=%d held(dwell=%d,breakeven=%d) dwell=%d/%d k=%s",
            self._rounds,
            hist,
            costs,
            surv,
            self._switches,
            self._held_dwell,
            self._held_breakeven,
            self._dwell,
            self.min_dwell,
            self._current,
        )

    @property
    def switch_stats(self) -> dict[str, int]:
        """Switches taken and suppressed -- the numbers that say whether the
        swap-amortisation gate is doing anything on this workload."""
        return {
            "switches": self._switches,
            "held_dwell": self._held_dwell,
            "held_breakeven": self._held_breakeven,
            "rounds": self._rounds,
        }

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
