"""Unit tests for the per-round adaptive draft chain length policy.

Pure CPU: the policy is a function over a survival vector and a cost callable,
and the cost model never reads a clock, so no CUDA is involved anywhere here.
"""

import math
import unittest

from sglang.srt.speculative.adaptive_chain import (
    DEFAULT_DRAFT_MS,
    DEFAULT_VERIFY_MS,
    AdaptiveChainPolicy,
    ChainCostModel,
    choose_chain_length,
    eligible_candidates,
    normalize_survival,
    parse_cost_ms,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _affine_cost(draft_ms=DEFAULT_DRAFT_MS, verify_ms=DEFAULT_VERIFY_MS):
    return lambda k: verify_ms + k * draft_ms


class TestNormalizeSurvival(unittest.TestCase):
    def test_clamps_into_unit_interval(self):
        self.assertEqual(normalize_survival([1.5, -0.2, 0.4], 3), [1.0, 0.0, 0.0])

    def test_nan_and_inf_become_zero(self):
        out = normalize_survival([0.9, float("nan"), float("inf")], 3)
        self.assertEqual(out, [0.9, 0.0, 0.0])
        self.assertTrue(all(not math.isnan(v) for v in out))

    def test_enforces_non_increasing(self):
        # A survival curve is a cumulative product; a rise is noise.
        self.assertEqual(normalize_survival([0.8, 0.9, 0.5], 3), [0.8, 0.8, 0.5])

    def test_short_vector_is_flat_extended(self):
        # Optimistic on purpose: prevents lock-in at k_min (see module docstring).
        self.assertEqual(normalize_survival([0.7], 3), [0.7, 0.7, 0.7])

    def test_empty_vector_becomes_zeros(self):
        self.assertEqual(normalize_survival([], 3), [0.0, 0.0, 0.0])

    def test_k_max_zero_returns_empty(self):
        self.assertEqual(normalize_survival([0.5], 0), [])

    def test_truncates_to_k_max(self):
        self.assertEqual(normalize_survival([0.9, 0.8, 0.7, 0.6], 2), [0.9, 0.8])


class TestChooseChainLength(unittest.TestCase):
    def test_all_confidence_one_picks_k_max(self):
        k = choose_chain_length([1.0] * 4, _affine_cost(), k_max=4, k_min=1)
        self.assertEqual(k, 4)

    def test_all_confidence_zero_picks_k_min(self):
        k = choose_chain_length([0.0] * 4, _affine_cost(), k_max=4, k_min=1)
        self.assertEqual(k, 1)

    def test_all_confidence_zero_respects_raised_k_min(self):
        k = choose_chain_length([0.0] * 4, _affine_cost(), k_max=4, k_min=3)
        self.assertEqual(k, 3)

    def test_empty_survival_falls_back_to_k_max(self):
        # No measurement yet must reproduce today's static behaviour.
        self.assertEqual(choose_chain_length([], _affine_cost(), k_max=3), 3)

    def test_decaying_confidence_picks_interior_k(self):
        # Steep decay: the 3rd and 4th steps add almost nothing but cost.
        survival = [0.95, 0.9, 0.05, 0.01]
        k = choose_chain_length(
            survival, _affine_cost(draft_ms=8.0, verify_ms=10.0), k_max=4, k_min=1
        )
        self.assertEqual(k, 2)

    def test_flat_verify_cost_favours_long_chains(self):
        # The measured regime: verify dominates and barely grows with rows.
        survival = [0.8, 0.6, 0.45, 0.3]
        k = choose_chain_length(
            survival, _affine_cost(draft_ms=2.5, verify_ms=26.0), k_max=4, k_min=1
        )
        self.assertEqual(k, 4)

    def test_expensive_draft_shortens_the_chain(self):
        survival = [0.8, 0.6, 0.45, 0.3]
        k = choose_chain_length(
            survival, _affine_cost(draft_ms=40.0, verify_ms=5.0), k_max=4, k_min=1
        )
        self.assertEqual(k, 1)

    def test_result_always_within_bounds(self):
        for surv in ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], []):
            k = choose_chain_length(surv, _affine_cost(), k_max=3, k_min=2)
            self.assertGreaterEqual(k, 2)
            self.assertLessEqual(k, 3)

    def test_nan_survival_does_not_poison_argmax(self):
        k = choose_chain_length([float("nan")] * 3, _affine_cost(), k_max=3, k_min=1)
        self.assertEqual(k, 1)

    def test_nan_cost_candidates_are_skipped(self):
        def cost(k):
            return float("nan") if k == 3 else 10.0 + k

        # k=3 would otherwise win with an all-ones survival.
        k = choose_chain_length([1.0, 1.0, 1.0], cost, k_max=3, k_min=1)
        self.assertEqual(k, 2)

    def test_non_positive_cost_candidates_are_skipped(self):
        def cost(k):
            return 0.0 if k == 1 else 10.0 + k

        k = choose_chain_length([0.0, 0.0, 0.0], cost, k_max=3, k_min=1)
        self.assertEqual(k, 2)

    def test_all_costs_invalid_falls_back_to_k_max(self):
        k = choose_chain_length([0.5, 0.5], lambda _k: -1.0, k_max=2, k_min=1)
        self.assertEqual(k, 2)

    def test_raising_cost_callable_is_survived(self):
        def cost(k):
            if k == 2:
                raise ValueError("boom")
            return 10.0 + k

        k = choose_chain_length([1.0, 1.0, 1.0], cost, k_max=3, k_min=1)
        self.assertEqual(k, 3)

    def test_k_max_zero_returns_zero(self):
        self.assertEqual(choose_chain_length([0.5], _affine_cost(), k_max=0), 0)

    def test_k_min_above_k_max_is_clamped(self):
        k = choose_chain_length([0.0, 0.0], _affine_cost(), k_max=2, k_min=9)
        self.assertEqual(k, 2)

    def test_tie_prefers_shorter_chain(self):
        # Construct an exact tie: gain doubles while cost doubles.
        # k=1: (1+1.0)/10 = 0.2 ; k=2: (1+1.0+1.0)/15 = 0.2
        def cost(k):
            return {1: 10.0, 2: 15.0}[k]

        self.assertEqual(choose_chain_length([1.0, 1.0], cost, k_max=2, k_min=1), 1)

    def test_monotone_cost_and_full_confidence_is_monotone_in_k(self):
        # Sanity: with survival==1 everywhere, a longer chain is never worse
        # under an affine cost whose intercept exceeds the slope.
        for k_max in range(1, 8):
            self.assertEqual(
                choose_chain_length([1.0] * k_max, _affine_cost(), k_max=k_max),
                k_max,
            )


class TestEligibleCandidates(unittest.TestCase):
    def test_none_means_full_range(self):
        self.assertEqual(eligible_candidates(4, 1, None), [1, 2, 3, 4])

    def test_k_min_trims_the_bottom(self):
        self.assertEqual(eligible_candidates(4, 3, None), [3, 4])

    def test_whitelist_is_filtered_and_sorted(self):
        self.assertEqual(eligible_candidates(7, 1, [7, 3, 1, 9, 0]), [1, 3, 7])

    def test_zero_candidate_is_dropped(self):
        # steps=0 disables drafting entirely; that is not this policy's call.
        self.assertEqual(eligible_candidates(3, 0, [0, 1, 3]), [1, 3])

    def test_empty_when_k_max_non_positive(self):
        self.assertEqual(eligible_candidates(0, 1, [1, 2]), [])


class TestChooseWithCandidates(unittest.TestCase):
    def test_choice_is_restricted_to_candidates(self):
        # Unrestricted this would pick 2; only 1 and 3 have runtime states.
        survival = [0.95, 0.9, 0.05]
        k = choose_chain_length(
            survival,
            _affine_cost(draft_ms=8.0, verify_ms=10.0),
            k_max=3,
            k_min=1,
            candidates=[1, 3],
        )
        self.assertIn(k, (1, 3))
        self.assertNotEqual(k, 2)

    def test_cumulative_gain_still_counts_skipped_steps(self):
        # k=3 must be scored with survival[0]+survival[1]+survival[2],
        # not only the candidate entries.
        seen = {}

        def cost(k):
            seen[k] = True
            return 10.0

        choose_chain_length([1.0, 1.0, 1.0], cost, k_max=3, k_min=1, candidates=[3])
        self.assertEqual(sorted(seen), [3])

    def test_empty_survival_picks_largest_candidate(self):
        k = choose_chain_length([], _affine_cost(), k_max=7, candidates=[1, 3, 7])
        self.assertEqual(k, 7)

    def test_all_zero_survival_picks_smallest_candidate(self):
        k = choose_chain_length(
            [0.0] * 7, _affine_cost(), k_max=7, candidates=[1, 3, 7]
        )
        self.assertEqual(k, 1)

    def test_empty_candidate_list_falls_back_to_k_max(self):
        k = choose_chain_length([0.0, 0.0], _affine_cost(), k_max=2, candidates=[])
        self.assertEqual(k, 2)

    def test_policy_only_returns_candidates(self):
        p = AdaptiveChainPolicy(k_max=7, candidates=[1, 3, 7])
        self.assertEqual(p.candidates, [1, 3, 7])
        for surv in ([], [1.0] * 7, [0.0] * 7, [0.5, 0.2, 0.1]):
            p.record_survival(surv)
            self.assertIn(p.choose(), [1, 3, 7])


class TestParseCostMs(unittest.TestCase):
    def test_parses_both_keys(self):
        self.assertEqual(parse_cost_ms("draft:2.5,verify:26"), (2.5, 26.0))

    def test_tolerates_whitespace_and_order(self):
        self.assertEqual(parse_cost_ms(" verify : 30 , draft : 3 "), (3.0, 30.0))

    def test_none_and_empty_return_defaults(self):
        self.assertEqual(parse_cost_ms(None), (DEFAULT_DRAFT_MS, DEFAULT_VERIFY_MS))
        self.assertEqual(parse_cost_ms(""), (DEFAULT_DRAFT_MS, DEFAULT_VERIFY_MS))

    def test_malformed_entries_keep_defaults(self):
        self.assertEqual(
            parse_cost_ms("draft:abc,verify:,nonsense,unknown:5"),
            (DEFAULT_DRAFT_MS, DEFAULT_VERIFY_MS),
        )

    def test_non_positive_values_are_rejected(self):
        # A zero cost would make every candidate look infinitely cheap.
        self.assertEqual(
            parse_cost_ms("draft:0,verify:-3"), (DEFAULT_DRAFT_MS, DEFAULT_VERIFY_MS)
        )

    def test_partial_override(self):
        self.assertEqual(parse_cost_ms("verify:12"), (DEFAULT_DRAFT_MS, 12.0))


class TestChainCostModel(unittest.TestCase):
    def test_prior_is_affine_before_warmup(self):
        m = ChainCostModel(k_max=3, draft_ms=2.0, verify_ms=20.0, min_samples=4)
        self.assertEqual(m.cost(1), 22.0)
        self.assertEqual(m.cost(3), 26.0)

    def test_prior_holds_until_min_samples(self):
        m = ChainCostModel(k_max=3, draft_ms=2.0, verify_ms=20.0, min_samples=4)
        for _ in range(3):
            m.observe(2, 100.0)
        self.assertEqual(m.cost(2), 24.0)
        m.observe(2, 100.0)
        self.assertAlmostEqual(m.cost(2), 100.0)

    def test_ema_converges_to_observed(self):
        m = ChainCostModel(
            k_max=2, draft_ms=1.0, verify_ms=1.0, ema_alpha=0.5, min_samples=1
        )
        for _ in range(40):
            m.observe(1, 50.0)
        self.assertAlmostEqual(m.cost(1), 50.0, places=3)

    def test_unobserved_k_keeps_prior(self):
        m = ChainCostModel(k_max=3, draft_ms=2.0, verify_ms=20.0, min_samples=1)
        m.observe(1, 99.0)
        self.assertAlmostEqual(m.cost(1), 99.0)
        self.assertEqual(m.cost(3), 26.0)

    def test_garbage_durations_are_dropped(self):
        m = ChainCostModel(k_max=2, draft_ms=1.0, verify_ms=10.0, min_samples=1)
        for bad in (float("nan"), float("inf"), -5.0, 0.0, None, "x"):
            m.observe(1, bad)
        self.assertEqual(m.cost(1), 11.0)

    def test_callable_interface_matches_cost(self):
        m = ChainCostModel(k_max=2, draft_ms=1.0, verify_ms=10.0)
        self.assertEqual(m(2), m.cost(2))

    def test_snapshot_covers_every_k(self):
        m = ChainCostModel(k_max=3, draft_ms=1.0, verify_ms=10.0, min_samples=1)
        m.observe(2, 40.0)
        snap = m.snapshot()
        self.assertEqual(sorted(snap), [1, 2, 3])
        self.assertEqual(snap[2][1], 1)
        self.assertEqual(snap[1][1], 0)

    def test_cost_model_drives_choose_chain_length(self):
        m = ChainCostModel(
            k_max=3, draft_ms=2.5, verify_ms=26.0, ema_alpha=0.5, min_samples=1
        )
        # Teach it that a 3-chain is catastrophically slow in reality.
        for _ in range(30):
            m.observe(3, 500.0)
        k = choose_chain_length([1.0, 1.0, 1.0], m.cost, k_max=3, k_min=1)
        self.assertEqual(k, 2)


class TestAdaptiveChainPolicy(unittest.TestCase):
    def test_first_choice_without_survival_is_k_max(self):
        p = AdaptiveChainPolicy(k_max=3)
        self.assertEqual(p.choose(), 3)

    def test_survival_drives_the_choice(self):
        p = AdaptiveChainPolicy(
            k_max=4,
            cost_model=ChainCostModel(k_max=4, draft_ms=40.0, verify_ms=5.0),
        )
        p.record_survival([0.2, 0.05, 0.01, 0.0])
        self.assertEqual(p.choose(), 1)

    def test_histogram_counts_choices(self):
        p = AdaptiveChainPolicy(k_max=2)
        p.choose()
        p.record_survival([0.0, 0.0])
        p.choose()
        self.assertEqual(p.histogram, {2: 1, 1: 1})

    def test_record_duration_feeds_cost_model(self):
        p = AdaptiveChainPolicy(
            k_max=2, cost_model=ChainCostModel(k_max=2, min_samples=1, ema_alpha=1.0)
        )
        p.record_duration(1, 123.0)
        self.assertAlmostEqual(p.cost_model.cost(1), 123.0)

    def test_k_min_is_respected(self):
        p = AdaptiveChainPolicy(k_max=4, k_min=2)
        p.record_survival([0.0, 0.0, 0.0, 0.0])
        self.assertEqual(p.choose(), 2)

    def test_log_every_emits_and_does_not_raise(self):
        p = AdaptiveChainPolicy(k_max=2, log_every=2)
        with self.assertLogs("sglang.srt.speculative.adaptive_chain", "INFO") as cm:
            p.choose()
            p.choose()
        self.assertTrue(any("[spec-adaptive]" in line for line in cm.output))

    def test_log_every_zero_stays_silent(self):
        p = AdaptiveChainPolicy(k_max=2, log_every=0)
        for _ in range(5):
            p.choose()
        self.assertEqual(sum(p.histogram.values()), 5)

    def test_recovers_upward_after_a_short_chain(self):
        """Flat extension must let the policy climb back out of k_min."""
        p = AdaptiveChainPolicy(
            k_max=3, cost_model=ChainCostModel(k_max=3, draft_ms=2.5, verify_ms=26.0)
        )
        p.record_survival([0.0])
        self.assertEqual(p.choose(), 1)
        # Next round the draft is confident again; only one step was run, so
        # only one survival entry exists.
        p.record_survival([0.99])
        self.assertEqual(p.choose(), 3)


if __name__ == "__main__":
    unittest.main()
