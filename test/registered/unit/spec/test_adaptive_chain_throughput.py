"""The chain regulator must maximise tokens per wall-millisecond, not acceptance.

Regression suite for the 27B A/B of 2026-09-20 (Task #59).  Arm ab27a ran the
per-round adaptive policy over ``k in [1..5]`` and picked ``k=5`` in 62 % of 400
rounds; arm ab27b, pinned to ``k=3``, was 4 % faster on the same maths prompt
(97.4 vs 93.4 tok/s) while accepting *fewer* tokens per round (3.08 vs 3.31).

The boot state these tests replay is quoted verbatim from ``arm_ab27a.log``,
round 400, all three ranks identical::

    rounds=400 k=2:120(30%) k=4:32(8%) k=5:248(62%)
    | c1=28.5ms/n0 c2=27.4ms/n118 c3=33.5ms/n0 c4=34.1ms/n32 c5=36.9ms/n248
    | survival=[0.472 0.343 0.343 0.343 0.343]
    | switches=12 held(dwell=6,breakeven=3) dwell=23/8 k=2

Two numbers in that line are assumptions, not measurements, and both push the
argmax up:

* ``survival=[0.472 0.343 0.343 0.343 0.343]`` is a two-entry readout (the
  round ran ``k=2``) padded flat.  Entries 3..5 are asserted, never measured.
* ``c1``/``c3`` carry ``n0``: never measured, still on the static prior, while
  their measured neighbours ``c2``/``c4`` bracket them.

Each is tested on its own, then together against the round-400 state, then in
a closed loop where the policy's own choice determines what it gets to measure.
"""

import math
import unittest

from sglang.srt.speculative.adaptive_chain import (
    AdaptiveChainPolicy,
    ChainCostModel,
    choose_chain_length,
    normalize_survival,
    switch_is_profitable,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# ---------------------------------------------------------------------------
# The measured state of boot ab27a, round 400.
# ---------------------------------------------------------------------------

#: Per-k round cost EMAs as logged; ``None`` marks a length that was never run
#: (``n0`` in the log) and therefore had no measurement of its own.
AB27A_COSTS = {1: None, 2: 27.4, 3: None, 4: 34.1, 5: 36.9}

#: The survival readout itself -- two entries, because the round ran k=2.
AB27A_SURVIVAL_RAW = [0.472, 0.343]

#: The cost prior in force during that boot (``SGLANG_SPEC_ADAPTIVE_CHAIN_COST_MS``
#: unset, so the built-in default).
AB27A_PRIOR_VERIFY_MS = 26.0
AB27A_PRIOR_DRAFT_MS = 2.5

#: Least squares through the three measured points (2, 4, 5): 21.09 + 3.193k.
#: Reproduced by ChainCostModel.fit(); asserted below rather than trusted.
AB27A_FIT_INTERCEPT = 21.09
AB27A_FIT_SLOPE = 3.193


def warm_model(costs, k_max=5, min_samples=8, **kwargs):
    """A ChainCostModel whose EMA is warm exactly where *costs* has a number."""
    model = ChainCostModel(
        k_max=k_max,
        draft_ms=kwargs.pop("draft_ms", AB27A_PRIOR_DRAFT_MS),
        verify_ms=kwargs.pop("verify_ms", AB27A_PRIOR_VERIFY_MS),
        ema_alpha=1.0,  # each observation replaces the EMA outright
        min_samples=min_samples,
        **kwargs,
    )
    for k, ms in costs.items():
        if ms is None:
            continue
        for _ in range(min_samples):
            model.observe(k, ms)
    return model


class TestFlatTailForcesKMax(unittest.TestCase):
    """A flat survival tail makes the argmax k_max for *any* curve and cost."""

    #: Cost line measured on ab27a: 21.09 ms + 3.193 ms per draft step.
    STEP_MS = AB27A_FIT_SLOPE

    def scores_under_flat_tail(self, first, tail, k_max=8):
        curve = normalize_survival([first, tail], k_max, extend="flat")
        scores, cum = [], 0.0
        for k in range(1, k_max + 1):
            cum += curve[k - 1]
            scores.append((1.0 + cum) / (AB27A_FIT_INTERCEPT + self.STEP_MS * k))
        return scores

    def test_flat_tail_pins_the_argmax_whenever_the_tail_beats_the_step_cost(self):
        # With the tail flat at s, every unmeasured step adds a CONSTANT s
        # tokens for one more draft forward, so the marginal rate is the
        # constant s / step_ms. Whenever that exceeds the rate the chain is
        # already achieving, the score can only rise -- and the argmax is
        # pinned at k_max regardless of what the measured entries said.
        for first, tail in ((0.9, 0.5), (0.472, 0.343), (0.6, 0.25)):
            with self.subTest(first=first, tail=tail):
                scores = self.scores_under_flat_tail(first, tail)
                marginal = tail / self.STEP_MS
                self.assertGreater(
                    marginal, scores[0], "test case must be in the pinning regime"
                )
                self.assertEqual(
                    scores, sorted(scores), "flat tail should be monotone rising"
                )
                self.assertEqual(max(range(1, 9), key=lambda k: scores[k - 1]), 8)

    def test_ab27a_was_squarely_in_that_regime(self):
        # The measured tail 0.343 over a 3.19 ms step is 0.107 tok/ms against
        # an achieved ~0.066 tok/ms -- not a marginal case, a factor of 1.6.
        marginal = 0.343 / self.STEP_MS
        achieved = self.scores_under_flat_tail(0.472, 0.343)[1]
        self.assertGreater(marginal, 1.5 * achieved)

    def test_geometric_tail_admits_an_interior_argmax(self):
        # Same first two entries, same cost -- continuing the decay instead of
        # asserting it away puts the optimum inside the range.
        k = choose_chain_length(
            AB27A_SURVIVAL_RAW, lambda k: 21.09 + 3.193 * k, k_max=5, k_min=1
        )
        self.assertLess(k, 5)


class TestCostFitReplacesThePrior(unittest.TestCase):
    def test_fit_reproduces_the_measured_line(self):
        model = warm_model(AB27A_COSTS)
        fitted = model.fit()
        self.assertIsNotNone(fitted)
        intercept, slope = fitted
        self.assertAlmostEqual(intercept, AB27A_FIT_INTERCEPT, places=2)
        self.assertAlmostEqual(slope, AB27A_FIT_SLOPE, places=3)

    def test_measured_k_still_reports_its_own_ema(self):
        model = warm_model(AB27A_COSTS)
        for k, ms in AB27A_COSTS.items():
            if ms is not None:
                self.assertAlmostEqual(model.cost(k), ms, places=6)

    def test_unmeasured_k_is_interpolated_not_priced_by_the_prior(self):
        model = warm_model(AB27A_COSTS)
        # k=3 sits between measured c2=27.4 and c4=34.1; the prior said 33.5,
        # which is above BOTH neighbours' midpoint and is why k=3 never won.
        prior_c3 = AB27A_PRIOR_VERIFY_MS + 3 * AB27A_PRIOR_DRAFT_MS
        self.assertAlmostEqual(prior_c3, 33.5, places=6)
        self.assertLess(model.cost(3), prior_c3)
        self.assertGreater(model.cost(3), AB27A_COSTS[2])
        self.assertLess(model.cost(3), AB27A_COSTS[4])

    def test_single_measured_point_keeps_the_prior(self):
        # One point is not a line. The cold-start prior has to stay in force.
        model = warm_model({2: 27.4})
        self.assertIsNone(model.fit())
        self.assertAlmostEqual(model.cost(3), 33.5, places=6)

    def test_negative_fitted_slope_is_clamped_to_zero(self):
        # Noise between two nearby points must never price a long chain below
        # a short one -- that hands the argmax straight back to k_max.
        model = warm_model({2: 40.0, 4: 20.0})
        intercept, slope = model.fit()
        self.assertEqual(slope, 0.0)
        self.assertGreaterEqual(model.cost(5), 0.0)
        self.assertGreaterEqual(model.cost(5), model.cost(1))

    def test_fitted_cost_stays_positive(self):
        # A steep fit could extrapolate below zero at k=1, which would make
        # that candidate *ineligible* rather than merely cheap.
        model = warm_model({4: 10.0, 5: 40.0})
        self.assertGreater(model.cost(1), 0.0)


class TestAb27aRoundStateNoLongerPicksKMax(unittest.TestCase):
    """The headline regression: the logged round-400 state must not say k=5."""

    def test_old_behaviour_is_reproduced_when_both_assumptions_are_restored(self):
        # Guards the diagnosis: with the flat pad AND the static prior, the
        # logged state really does score k=5 highest. If this ever stops
        # holding, the tests below are no longer testing what they claim.
        curve = normalize_survival(AB27A_SURVIVAL_RAW, 5, extend="flat")
        logged = {1: 28.5, 2: 27.4, 3: 33.5, 4: 34.1, 5: 36.9}
        cum, best, best_score = 0.0, None, -math.inf
        for k in range(1, 6):
            cum += curve[k - 1]
            score = (1.0 + cum) / logged[k]
            if score > best_score:
                best_score, best = score, k
        self.assertEqual(best, 5)

    def test_fixed_state_prefers_the_arm_b_length(self):
        model = warm_model(AB27A_COSTS)
        k = choose_chain_length(AB27A_SURVIVAL_RAW, model.cost, k_max=5, k_min=1)
        self.assertNotEqual(k, 5, "k=5 is the regression being fixed")
        self.assertEqual(k, 3, "k=3 is the length arm ab27b actually won with")

    def test_policy_on_the_logged_state_warms_the_untimed_lengths_first(self):
        # k=1 and k=3 carried n0 in the log: never run, never timed. The
        # policy now buys those measurements before optimising, rather than
        # optimising against a prior for them.
        policy = AdaptiveChainPolicy(
            k_max=5,
            k_min=1,
            cost_model=warm_model(AB27A_COSTS),
            candidates=[1, 2, 3, 4, 5],
            min_dwell=0,
        )
        policy.record_survival(AB27A_SURVIVAL_RAW)
        self.assertTrue(policy.warming_up)
        self.assertEqual(policy.choose(), 1, "k=1 is the first untimed candidate")

    def test_policy_with_every_length_timed_picks_the_arm_b_length(self):
        # Same curve, but now every candidate has a measured round cost --
        # k=1 and k=3 interpolated onto the measured line for the test, since
        # a real boot gets them by running them (see the warm-up test above).
        costs = dict(AB27A_COSTS)
        for k in (1, 3):
            costs[k] = AB27A_FIT_INTERCEPT + AB27A_FIT_SLOPE * k
        policy = AdaptiveChainPolicy(
            k_max=5,
            k_min=1,
            cost_model=warm_model(costs),
            candidates=[1, 2, 3, 4, 5],
            min_dwell=0,
        )
        self.assertFalse(policy.warming_up)
        policy.record_survival(AB27A_SURVIVAL_RAW)
        self.assertEqual(policy.choose(), 3)


class TestClosedLoopDistribution(unittest.TestCase):
    """Run the regulator against a ground truth and count what it picks.

    The open-loop tests above fix the survival vector.  Here the vector is a
    *consequence* of the choice: a round that runs ``k`` only ever reports the
    first ``k`` entries of the true curve, which is the feedback path that made
    ab27a oscillate between k=2 and k=5.  The ground truth is the ab27a curve
    continued at its own measured ratio, and the ground-truth round cost is the
    line fitted through the three measured points.
    """

    #: True per-step survival, first two entries as measured on ab27a.
    TRUTH = [0.472, 0.343, 0.2493, 0.1812, 0.1317]

    @classmethod
    def true_cost(cls, k):
        return AB27A_FIT_INTERCEPT + AB27A_FIT_SLOPE * k

    @classmethod
    def true_rate(cls, k):
        return (1.0 + sum(cls.TRUTH[:k])) / cls.true_cost(k)

    def test_ground_truth_optimum_is_k3(self):
        # State the answer the simulation is allowed to find, analytically,
        # before running anything -- otherwise the loop grades its own work.
        rates = {k: self.true_rate(k) for k in range(1, 6)}
        self.assertEqual(max(rates, key=rates.get), 3)
        self.assertGreater(rates[3], rates[5])

    def run_loop(self, rounds=400, min_dwell=8, switch_margin=0.02):
        policy = AdaptiveChainPolicy(
            k_max=5,
            k_min=1,
            cost_model=ChainCostModel(
                k_max=5,
                draft_ms=AB27A_PRIOR_DRAFT_MS,
                verify_ms=AB27A_PRIOR_VERIFY_MS,
                ema_alpha=0.2,
                min_samples=8,
            ),
            candidates=[1, 2, 3, 4, 5],
            min_dwell=min_dwell,
            switch_margin=switch_margin,
        )
        chosen = []
        for _ in range(rounds):
            k = policy.choose()
            chosen.append(k)
            # The round runs: it costs what the truth says, and it reveals
            # exactly the k survival entries a k-step draft produces.
            policy.record_duration(k, self.true_cost(k))
            policy.record_survival(self.TRUTH[:k])
        return policy, chosen

    def test_k5_is_not_the_majority(self):
        _, chosen = self.run_loop()
        share_k5 = chosen.count(5) / len(chosen)
        self.assertLess(
            share_k5,
            0.5,
            f"k=5 took {share_k5:.0%} of rounds (ab27a measured 62%)",
        )

    def test_it_settles_on_the_ground_truth_optimum(self):
        policy, chosen = self.run_loop()
        self.assertEqual(policy.current, 3)
        # The tail of the run is where the estimates are warm.
        tail = chosen[-100:]
        self.assertGreater(tail.count(3) / len(tail), 0.9)

    def test_realised_throughput_beats_the_ab27a_distribution(self):
        # The point of the exercise, stated as tokens per millisecond: the new
        # regulator must be at least as fast as the fixed-k=3 arm, and faster
        # than the k-distribution ab27a actually ran.
        _, chosen = self.run_loop()

        def rate(ks):
            return sum(1.0 + sum(self.TRUTH[:k]) for k in ks) / sum(
                self.true_cost(k) for k in ks
            )

        # The k-distribution ab27a actually ran, scored under the same truth.
        mix_rate = rate([5] * 248 + [2] * 120 + [4] * 32)
        self.assertGreater(rate(chosen), mix_rate, "whole run, warm-up included")
        # Once warm, it should be indistinguishable from running the optimum.
        self.assertGreaterEqual(rate(chosen[-200:]), 0.999 * self.true_rate(3))

    def test_switch_count_stays_low(self):
        policy, _ = self.run_loop()
        self.assertLessEqual(
            policy.switch_stats["switches"],
            12,
            "must not thrash more than ab27a did (12 switches in 400 rounds)",
        )


class TestSwitchHysteresis(unittest.TestCase):
    def test_margin_suppresses_a_marginal_gain(self):
        # 2 % better, 5 % demanded: held. Swap-free (resident states), which is
        # exactly the case the break-even gate alone cannot refuse.
        kwargs = dict(
            e_incumbent=2.0,
            c_incumbent=30.0,
            e_candidate=2.04,
            c_candidate=30.0,
            swap_ms=0.0,
            dwell_rounds=8,
        )
        self.assertTrue(switch_is_profitable(**kwargs))
        self.assertFalse(switch_is_profitable(margin=0.05, **kwargs))

    def test_margin_admits_a_real_gain(self):
        self.assertTrue(
            switch_is_profitable(
                e_incumbent=2.0,
                c_incumbent=30.0,
                e_candidate=2.4,
                c_candidate=30.0,
                swap_ms=0.0,
                dwell_rounds=8,
                margin=0.05,
            )
        )

    def test_zero_margin_is_the_old_behaviour(self):
        self.assertTrue(
            switch_is_profitable(
                e_incumbent=2.0,
                c_incumbent=30.0,
                e_candidate=2.0001,
                c_candidate=30.0,
                swap_ms=0.0,
                dwell_rounds=8,
                margin=0.0,
            )
        )

    def test_negative_margin_is_clamped_not_honoured(self):
        # A negative margin would mean "switch even when it is worse".
        self.assertFalse(
            switch_is_profitable(
                e_incumbent=2.0,
                c_incumbent=30.0,
                e_candidate=1.9,
                c_candidate=30.0,
                swap_ms=0.0,
                dwell_rounds=8,
                margin=-0.5,
            )
        )

    def test_policy_holds_through_noise_with_a_margin(self):
        """Two lengths of near-equal throughput must not trade places."""

        def run(margin):
            policy = AdaptiveChainPolicy(
                k_max=3,
                k_min=1,
                cost_model=warm_model({1: 20.0, 2: 24.0, 3: 28.0}, k_max=3),
                candidates=[1, 2, 3],
                min_dwell=0,
                switch_margin=margin,
            )
            # Two curves whose argmax differs, but only just -- the throughput
            # gap between k=2 and k=3 here is under 3 %, well inside the noise
            # a one-round-old curve and an EMA carry.
            for i in range(40):
                policy.record_survival(
                    [0.90, 0.62, 0.40] if i % 2 else [0.90, 0.62, 0.46]
                )
                policy.choose()
            return policy.switch_stats["switches"]

        noisy, damped = run(0.0), run(0.10)
        self.assertGreater(noisy, damped)
        self.assertLessEqual(damped, 1)

    def test_the_noise_case_really_does_straddle_the_argmax(self):
        # Guards the test above: if both curves had the same argmax it would
        # pass for the wrong reason.
        costs = {1: 20.0, 2: 24.0, 3: 28.0}
        picks = {}
        rates = {}
        for c in ([0.90, 0.62, 0.40], [0.90, 0.62, 0.46]):
            picks[tuple(c)] = choose_chain_length(
                c, lambda k: costs[k], k_max=3, k_min=1
            )
            scored = [(1.0 + sum(c[:k])) / costs[k] for k in (1, 2, 3)]
            rates[tuple(c)] = max(scored[1:]) / min(scored[1:])
        self.assertEqual(len(set(picks.values())), 2, picks)
        # ... and that the gap it straddles is inside the 2 % margin, so the
        # margin is what suppresses it rather than the dwell.
        for curve, ratio in rates.items():
            self.assertLess(ratio, 1.02, f"{curve} gap {ratio:.4f} must be < 2 %")


class TestSwitchLog(unittest.TestCase):
    def test_taken_switch_is_logged_with_old_new_and_gain(self):
        policy = AdaptiveChainPolicy(
            k_max=3,
            k_min=1,
            cost_model=warm_model({1: 20.0, 2: 24.0, 3: 90.0}, k_max=3),
            candidates=[1, 2, 3],
            min_dwell=0,
        )
        policy.record_survival([1.0, 1.0, 1.0])
        first = policy.choose()
        with self.assertLogs(
            "sglang.srt.speculative.adaptive_chain", level="INFO"
        ) as captured:
            # Confidence collapses past step 1: the short chain now wins.
            policy.record_survival([1.0, 0.01, 0.0])
            second = policy.choose()
        self.assertNotEqual(first, second)
        line = "\n".join(captured.output)
        self.assertIn("chain switch", line)
        self.assertIn(f"k={first}->{second}", line)
        self.assertIn("tok/ms", line)
        self.assertIn("break-even", line)

        self.assertIsNotNone(policy.last_switch)
        self.assertEqual(policy.last_switch["from"], first)
        self.assertEqual(policy.last_switch["to"], second)
        self.assertGreater(policy.last_switch["gain_pct"], 0.0)

    def test_held_length_logs_nothing(self):
        policy = AdaptiveChainPolicy(
            k_max=3,
            k_min=1,
            cost_model=warm_model({1: 20.0, 2: 24.0, 3: 28.0}, k_max=3),
            candidates=[1, 2, 3],
            min_dwell=0,
        )
        policy.record_survival([0.9, 0.8, 0.7])
        policy.choose()
        logger_name = "sglang.srt.speculative.adaptive_chain"
        with self.assertNoLogs(logger_name, level="INFO"):
            for _ in range(5):
                policy.choose()
        self.assertIsNone(policy.last_switch)


class TestRankUniformityIsUntouched(unittest.TestCase):
    """The new inputs must not become a second source of rank divergence."""

    def test_same_inputs_give_the_same_choice(self):
        def run():
            policy = AdaptiveChainPolicy(
                k_max=5,
                k_min=1,
                cost_model=warm_model(AB27A_COSTS),
                candidates=[1, 2, 3, 4, 5],
                min_dwell=8,
                switch_margin=0.05,
            )
            out = []
            for i in range(50):
                policy.record_survival(AB27A_SURVIVAL_RAW)
                policy.record_duration(2 + i % 3, 27.0 + i % 5)
                out.append(policy.choose())
            return out

        self.assertEqual(run(), run())

    def test_consensus_still_overrides_the_local_proposal(self):
        policy = AdaptiveChainPolicy(
            k_max=5,
            k_min=1,
            cost_model=warm_model(AB27A_COSTS),
            candidates=[1, 2, 3, 4, 5],
            min_dwell=0,
            switch_margin=0.05,
            consensus=lambda _proposal: 4,
        )
        policy.record_survival(AB27A_SURVIVAL_RAW)
        self.assertEqual(policy.choose(), 4)
        self.assertEqual(policy.current, 4)


if __name__ == "__main__":
    unittest.main()
