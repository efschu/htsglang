"""Swap-amortisation for the adaptive chain policy (boot fn8s4, 2026-09-20).

fn8s4 ran the per-round survival policy on Qwen3.8 Next Flash (32k form) and
LOST throughput while winning acceptance: code 55.1 tok/s (accept 2.75) and
maths 58.2 (accept 3.39) against a fixed-k=2 baseline of 64.1/64.6 (accept
2.68/2.69). The cause is in the log's own numbers: 1270 graph-memory swaps at
a mean 33.46 ms each, 42.5 s in total, against a decode round of ~38 ms. The
policy bought a real acceptance gain and then spent more than it on moving
runtime states in and out of VRAM.

Two independent defects, one test file each half:

* The swap was charged even when it bought nothing. 630 of the 1270 swaps had
  ``target=<baseline>`` -- they unmapped a tagged state to make room for the
  UNTAGGED baseline, which occupies no tagged pages at all. See
  :class:`TestResidencyPlanner`.
* Nothing compared the gain of a switch against its cost, so the policy
  flipped k on consecutive rounds (the log shows 18 flips between steps=1 and
  steps=3 inside two seconds). See :class:`TestBreakEvenGate`.
"""

import unittest

from sglang.srt.speculative.adaptive_chain import (
    AdaptiveChainPolicy,
    ChainConsensusError,
    ChainCostModel,
    break_even_rounds,
    expected_tokens,
    switch_is_profitable,
)
from sglang.srt.speculative.adaptive_graph_memory import plan_residency
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu")

# Measured on fn8s4 (see module docstring).
FN8S4_ROUND_MS = 38.0
FN8S4_SWAP_MS = 33.46
FN8S4_STATE_BYTES = {
    "adaptive_state_k0": 468 << 20,
    "adaptive_state_k1": 494 << 20,
    "adaptive_state_k2": 516 << 20,
}
# Device-free with every state paused, on the three ranks of that boot.
FN8S4_FREE_MIB = (3535.5, 4001.5, 5720.7)
FN8S4_MARGIN_MIB = 512  # SGLANG_ADAPTIVE_SERVING_MARGIN_MIB default


class TestResidencyPlanner(unittest.TestCase):
    def test_baseline_target_evicts_nothing(self):
        """The 630-swap defect: unmapping to make room for the untagged
        baseline, which needs no room."""
        self.assertEqual(
            plan_residency(
                target=None,
                resident=["adaptive_state_k1"],
                sizes=FN8S4_STATE_BYTES,
                budget_bytes=0,
            ),
            [],
        )

    def test_resident_target_evicts_nothing(self):
        self.assertEqual(
            plan_residency(
                target="adaptive_state_k1",
                resident=["adaptive_state_k1"],
                sizes=FN8S4_STATE_BYTES,
                budget_bytes=0,
            ),
            [],
        )

    def test_fn8s4_budget_holds_every_state_at_once(self):
        """The whole 42.5 s was spent for nothing: on the TIGHTEST of the
        three ranks the budget covers all three states with 2 GiB to spare."""
        budget = int((FN8S4_FREE_MIB[0] - FN8S4_MARGIN_MIB) * (1 << 20))
        total = sum(FN8S4_STATE_BYTES.values())
        self.assertLess(total, budget)
        # 1545 MiB spare with all three mapped AND the 512 MiB serving
        # margin still fully booked.
        self.assertGreater((budget - total) / (1 << 20), 1500)
        # ...so activating any state evicts none of the others.
        resident = ["adaptive_state_k0", "adaptive_state_k1"]
        self.assertEqual(
            plan_residency(
                target="adaptive_state_k2",
                resident=resident,
                sizes=FN8S4_STATE_BYTES,
                budget_bytes=budget,
            ),
            [],
        )

    def test_evicts_least_recently_used_first_and_only_as_far_as_needed(self):
        sizes = {"a": 100, "b": 100, "c": 100, "t": 100}
        # Budget for two states; three are resident (a is the LRU end).
        self.assertEqual(
            plan_residency(
                target="t", resident=["a", "b", "c"], sizes=sizes, budget_bytes=200
            ),
            ["a", "b"],
        )

    def test_never_evicts_the_target(self):
        sizes = {"a": 100, "t": 100}
        self.assertEqual(
            plan_residency(
                target="t", resident=["t", "a"], sizes=sizes, budget_bytes=0
            ),
            ["a"],
        )

    def test_zero_budget_reproduces_one_state_at_a_time(self):
        sizes = {"a": 100, "t": 100}
        self.assertEqual(
            plan_residency(target="t", resident=["a"], sizes=sizes, budget_bytes=0),
            ["a"],
        )


class TestBreakEvenGate(unittest.TestCase):
    def test_fn8s4_switch_does_not_pay_for_a_single_round(self):
        """The measured case: k=3 really is better per round, and a swap held
        for one round still loses."""
        e_inc, c_inc = 2.69, FN8S4_ROUND_MS  # k=2 baseline
        e_new, c_new = 3.39, FN8S4_ROUND_MS + 2.5  # k=3, one more draft step
        # Better in steady state...
        self.assertGreater(e_new / c_new, e_inc / c_inc)
        # ...and still a loss when the swap is charged against one round.
        self.assertFalse(
            switch_is_profitable(e_inc, c_inc, e_new, c_new, FN8S4_SWAP_MS, 1)
        )

    def test_same_switch_pays_once_it_is_held(self):
        e_inc, c_inc = 2.69, FN8S4_ROUND_MS
        e_new, c_new = 3.39, FN8S4_ROUND_MS + 2.5
        n = break_even_rounds(e_inc, c_inc, e_new, c_new, FN8S4_SWAP_MS)
        # Break-even is a handful of rounds, not hundreds.
        self.assertGreater(n, 1.0)
        self.assertLess(n, 10.0)
        self.assertFalse(
            switch_is_profitable(e_inc, c_inc, e_new, c_new, FN8S4_SWAP_MS, 1)
        )
        self.assertTrue(
            switch_is_profitable(e_inc, c_inc, e_new, c_new, FN8S4_SWAP_MS, 32)
        )

    def test_free_swap_reduces_to_the_plain_rate_comparison(self):
        """With the state already resident the swap costs nothing, so any
        steady-state improvement is worth taking immediately."""
        self.assertTrue(
            switch_is_profitable(2.69, 38.0, 3.39, 40.5, swap_ms=0.0, dwell_rounds=1)
        )

    def test_a_worse_candidate_never_pays_however_long_it_is_held(self):
        self.assertEqual(
            break_even_rounds(3.39, 38.0, 2.69, 40.5, FN8S4_SWAP_MS), float("inf")
        )
        self.assertFalse(
            switch_is_profitable(3.39, 38.0, 2.69, 40.5, FN8S4_SWAP_MS, 10**6)
        )

    def test_degenerate_inputs_refuse_the_switch(self):
        for args in (
            (2.0, 0.0, 3.0, 40.0, 30.0, 8),  # zero incumbent cost
            (2.0, 38.0, 3.0, 0.0, 30.0, 8),  # zero candidate cost
            (2.0, 38.0, float("nan"), 40.0, 30.0, 8),
            (2.0, 38.0, 3.0, 40.0, float("inf"), 8),
            (2.0, 38.0, 3.0, 40.0, 30.0, 0),  # never held
        ):
            self.assertFalse(switch_is_profitable(*args), args)

    def test_expected_tokens_counts_the_bonus_token(self):
        self.assertEqual(expected_tokens([], 3), 1.0)
        self.assertAlmostEqual(expected_tokens([0.9, 0.8, 0.7], 2), 2.7)
        self.assertAlmostEqual(expected_tokens([0.9, 0.8, 0.7], 3), 3.4)


class TestPolicyHoldsItsChoice(unittest.TestCase):
    """The policy-level consequence: fn8s4's flip-flop cannot recur."""

    @staticmethod
    def _flapping_policy(min_dwell, swap_ms):
        # Cost model warm at both candidates so the argmax is estimate-driven.
        cost = ChainCostModel(k_max=3, min_samples=1)
        for _ in range(4):
            cost.observe(1, FN8S4_ROUND_MS)
            cost.observe(3, FN8S4_ROUND_MS + 2.5)
        return AdaptiveChainPolicy(
            k_max=3,
            candidates=[1, 3],
            cost_model=cost,
            min_dwell=min_dwell,
            swap_ms=swap_ms,
        )

    def _run(self, policy, rounds=60):
        """Alternate the survival curve every round -- the adversarial input
        that made fn8s4 flip 18 times in two seconds."""
        picks = []
        for i in range(rounds):
            # Period 3, deliberately coprime with the dwell: a period that
            # divides the dwell would land every decision round on the same
            # phase of the curve and hide the flapping instead of testing it.
            policy.record_survival(
                [0.05, 0.0, 0.0] if i % 3 == 0 else [0.95, 0.95, 0.95]
            )
            picks.append(policy.choose())
        return picks

    def test_without_a_dwell_or_swap_cost_the_policy_flaps(self):
        policy = self._flapping_policy(min_dwell=0, swap_ms=0.0)
        self._run(policy)
        # Baseline behaviour: it really does chase the curve every round.
        self.assertGreater(policy.switch_stats["switches"], 20)

    def test_the_dwell_bounds_the_switch_rate(self):
        policy = self._flapping_policy(min_dwell=8, swap_ms=0.0)
        picks = self._run(policy, rounds=60)
        stats = policy.switch_stats
        self.assertEqual(stats["rounds"], 60)
        # At most one switch per dwell window, versus >20 without it.
        self.assertLessEqual(stats["switches"], 60 // 8 + 1)
        self.assertGreater(stats["held_dwell"] + stats["held_breakeven"], 0)
        # And it never returns a chain length it was not built for.
        self.assertTrue(set(picks) <= set(policy.candidates))

    def test_the_break_even_gate_refuses_an_unprofitable_switch(self):
        """Dwell satisfied, but the swap still does not pay: held anyway."""
        policy = self._flapping_policy(min_dwell=1, swap_ms=FN8S4_SWAP_MS)
        self._run(policy, rounds=40)
        self.assertGreater(policy.switch_stats["held_breakeven"], 0)

    def test_a_resident_target_is_free_so_the_gate_allows_it(self):
        """swap_ms_for reports 0 for an already-mapped state, which is the
        common case once residency is budget-driven -- the gate must then get
        out of the way."""
        policy = self._flapping_policy(min_dwell=1, swap_ms=FN8S4_SWAP_MS)
        policy.swap_ms_for = lambda k: 0.0
        self._run(policy, rounds=40)
        self.assertEqual(policy.switch_stats["held_breakeven"], 0)
        self.assertGreater(policy.switch_stats["switches"], 0)


class TestRankConsensus(unittest.TestCase):
    """A divergent k deadlocks the next collective (fn8s4 round 859), so the
    decision has to be identical on every rank by construction."""

    def test_consensus_overrides_the_local_proposal(self):
        policy = AdaptiveChainPolicy(
            k_max=3, candidates=[1, 3], min_dwell=0, consensus=lambda k: 3
        )
        policy.record_survival([0.0, 0.0, 0.0])  # locally argues for k=1
        self.assertEqual(policy.choose(), 3)
        self.assertEqual(policy.current, 3)

    def test_two_ranks_with_different_estimates_still_agree(self):
        """Rank 1 mirrors rank 0's answer even though its own cost model and
        survival curve differ -- which they always do, being CUDA-event
        timings and a non-blocking D2H readout."""
        rank0 = AdaptiveChainPolicy(k_max=3, candidates=[1, 3], min_dwell=4)
        rank1 = AdaptiveChainPolicy(
            k_max=3,
            candidates=[1, 3],
            min_dwell=4,
            consensus=lambda _k: rank0.current,
        )
        for i in range(40):
            rank0.record_survival([0.9, 0.9, 0.9] if i % 2 else [0.1, 0.0, 0.0])
            # Deliberately skewed local evidence on rank 1.
            rank1.record_survival([0.5, 0.5, 0.5])
            rank1.record_duration(1, 40.0 + i)
            k0 = rank0.choose()
            k1 = rank1.choose()
            self.assertEqual(k0, k1, f"ranks diverged at round {i}")

    def test_consensus_outside_the_candidate_set_stops_the_group(self):
        """fnFL2 H27: this used to keep the local proposal. A length no rank
        built is a detected disagreement -- crash-stop (Nutzer-Gesetz
        2026-08-29), never continue on a rank-local k."""
        policy = AdaptiveChainPolicy(
            k_max=3, candidates=[1, 3], min_dwell=0, consensus=lambda _k: 7
        )
        policy.record_survival([0.0, 0.0, 0.0])
        with self.assertRaises(ChainConsensusError):
            policy.choose()

    def test_a_raising_consensus_hook_stops_the_group(self):
        """fnFL2 H27: a failed broadcast used to be swallowed ("Ranks may now
        disagree"); on Form A that rank would size the solo draft-token
        broadcast from its own k and hang or misread the next collective."""

        def boom(_k):
            raise RuntimeError("broadcast failed")

        policy = AdaptiveChainPolicy(
            k_max=3, candidates=[1, 3], min_dwell=0, consensus=boom
        )
        policy.record_survival([0.9, 0.9, 0.9])
        with self.assertRaises(ChainConsensusError):
            policy.choose()

    def test_frozen_rounds_post_no_consensus_call(self):
        """Between decision rounds nothing rank-local is read and no
        collective is posted -- that is what makes the broadcast affordable."""
        calls = []

        def counting(k):
            calls.append(k)
            return k

        policy = AdaptiveChainPolicy(
            k_max=3, candidates=[1, 3], min_dwell=8, consensus=counting
        )
        for i in range(32):
            policy.record_survival([0.9, 0.9, 0.9] if i % 2 else [0.1, 0.0, 0.0])
            policy.choose()
        self.assertEqual(len(calls), 32 // 8)


if __name__ == "__main__":
    unittest.main()
