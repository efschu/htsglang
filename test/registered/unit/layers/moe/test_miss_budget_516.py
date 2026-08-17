"""#516 — the longer-horizon miss budget: it must bind, and OFF must be inert.

WHAT THIS BUDGET IS, and what it deliberately is not. The PER-WAVE miss cap
already exists and is not this: `expert_offload` scratch slots plus
`plan_token_waves`, whose exhaustion policy is a token-wave split that its own
docstring proves byte-identical. What #516's third half actually lacked is a
budget over a LONGER horizon — across windows, deciding whether a re-rank is
worth paying for at all.

The lossless framing matters. At exhaustion a miss cannot be declined (the
wave's GEMM needs those weights) and substituting a resident expert is a
different computation, i.e. lossy and out of scope for a default. So the budget
does not gate ADMISSION; it gates the PLACEMENT CHANGE. That keeps it lossless
by construction: it only ever decides whether to move weights between tiers,
never which expert computes a token.

The mechanism it exploits is one `HeatMigrationConfig` already names in its own
comment — "small values re-rank on noise and pay H2D for it". A window whose
miss rate is already inside the budget is a window whose top-R movement is
noise, and re-ranking to it chases that noise.

Evidence: `scripts/dev/302a_heat_desk/simulate_miss_budget.py` replays the
recorded #302a series (12 cumulative snapshots per rank, differenced into 11
per-window deltas). At budget 0.04 the trigger beat swap-every-window on ALL
NINE recorded rank/series combinations, worst case +0.0021 hit rate, mean
+0.0052, at 15-54% of the swaps. SIMULATION ONLY — nothing here has run on
metal, which is why the default is 0.0.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.layers.moe.expert_heat_migration import (
    HeatMigrationConfig,
    HeatWindow,
)
from sglang.test.test_utils import CustomTestCase


def _window(budget: float, hits: int, misses: int) -> HeatWindow:
    w = HeatWindow(HeatMigrationConfig(enabled=True, miss_budget=budget))
    w.stats.window_hit_activations = hits
    w.stats.window_miss_activations = misses
    return w


class TestOffIsInert(CustomTestCase):
    """The default must change nothing. A budget that alters the OFF path is a
    behaviour change shipped as a knob."""

    def test_the_default_is_zero(self):
        self.assertEqual(HeatMigrationConfig().miss_budget, 0.0)

    def test_a_zero_budget_never_holds_even_at_a_perfect_hit_rate(self):
        """Zero means 'the budget has nothing to say', NOT 'always skip'. If
        this ever returned True the OFF path would silently stop re-ranking."""
        self.assertFalse(_window(0.0, hits=1000, misses=0).budget_holds())

    def test_a_negative_budget_is_treated_as_off(self):
        self.assertFalse(_window(-1.0, hits=1000, misses=0).budget_holds())


class TestTheBudgetBinds(CustomTestCase):
    """THE CAN-FAIL THAT MATTERS. A budget that can never bind measures the
    baseline twice and would report a clean 'no regression' forever."""

    def test_it_holds_when_the_window_is_inside_the_budget(self):
        # 2% miss against a 4% budget: placement is fine, do not churn.
        self.assertTrue(_window(0.04, hits=98, misses=2).budget_holds())

    def test_it_releases_when_the_window_exceeds_the_budget(self):
        # 10% miss against a 4% budget: placement is not fine, re-rank.
        self.assertFalse(_window(0.04, hits=90, misses=10).budget_holds())

    def test_it_actually_suppresses_the_swap_plan(self):
        """Binding must reach the OUTPUT, not just the predicate. This is the
        difference between a budget and a counter."""
        w = _window(0.04, hits=98, misses=2)
        w.heat = {0: 100.0, 1: 90.0, 2: 1.0}
        self.assertEqual(w.plan(resident_ids=[2]), [])

    def test_the_same_state_without_the_budget_DOES_plan_a_swap(self):
        """The falsifier for the test above: if this planned nothing either,
        the suppression proved nothing."""
        w = _window(0.0, hits=98, misses=2)
        w.heat = {0: 100.0, 1: 90.0, 2: 1.0}
        self.assertNotEqual(w.plan(resident_ids=[2]), [])

    def test_a_held_round_is_counted(self):
        """A suppression nobody can see is indistinguishable from a policy that
        never fired."""
        w = _window(0.04, hits=98, misses=2)
        w.heat = {0: 100.0, 1: 90.0, 2: 1.0}
        w.plan(resident_ids=[2])
        self.assertEqual(w.stats.budget_held_rounds, 1)
        self.assertIn("budget_held_rounds", w.stats.as_dict())


class TestBoundaryAndDegenerateInputs(CustomTestCase):
    def test_exactly_at_the_budget_holds(self):
        """<= not <: a window sitting exactly on the budget is meeting it."""
        self.assertTrue(_window(0.04, hits=96, misses=4).budget_holds())

    def test_an_empty_window_does_not_hold(self):
        """No observations is not evidence that placement is fine; deciding
        from zero data is how a policy freezes a bad set in place."""
        self.assertFalse(_window(0.04, hits=0, misses=0).budget_holds())

    def test_an_all_miss_window_releases(self):
        self.assertFalse(_window(0.04, hits=0, misses=50).budget_holds())


class TestTheDecisionIsPureAndLossless(CustomTestCase):
    def test_budget_holds_does_not_mutate_the_window(self):
        """It is consulted before close_round folds the counters; a side effect
        here would corrupt the very numbers it read."""
        w = _window(0.04, hits=98, misses=2)
        before = (w.stats.window_hit_activations, w.stats.window_miss_activations)
        w.budget_holds()
        after = (w.stats.window_hit_activations, w.stats.window_miss_activations)
        self.assertEqual(before, after)

    def test_the_budget_never_touches_routing_vocabulary(self):
        """LOSSLESSNESS, as a property rather than a claim: this policy may
        only decide whether weights MOVE between tiers. If it ever gained the
        power to change which expert computes a token, the default-off argument
        would no longer be sufficient and quality-last would apply."""
        import inspect

        from sglang.srt.layers.moe.expert_heat_migration import HeatWindow as HW

        src = inspect.getsource(HW.budget_holds)
        for forbidden in ("topk", "router", "logits", "route", "fallback"):
            with self.subTest(token=forbidden):
                self.assertNotIn(forbidden, src)


if __name__ == "__main__":
    unittest.main()
