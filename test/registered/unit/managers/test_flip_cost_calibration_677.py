"""#677: the flip economics priced a 22s flip at 3.2s.

MEASURED ON THE LIVE HARVEST BOOT. Its own #758 refill marks report 22-24 s per
flip leg (NOTE_690_refill_commit_split.md, 5 flips x 3 ranks), while its policy
line reads:

    break-even 3.2s / (1/1681 - 1/7245.5)   ->   N=7004

3.2 s is ``DEFAULT_FLIP_COST_S``, derived from 997/1246/1720 ms in
PROD_BRINGUP_BENCH -- the PINNED-image era. The current arm is file-backed and
costs ~7x that. ``SGLANG_PHASE_POLICY_FLIP_COST_S`` is not set on this boot, so
the stale default is live.

The consequence is arithmetic, not opinion: at the measured cost the break-even
backlog is ~49,250 tokens; the policy uses 7,004. **Flips are priced 7.03x too
cheaply**, which is a live candidate root for the observed churn and compounds
with #759 (whose IDLE-LOCK floor is ``flip_tokens``, so it inherits the same
factor).

THE FIX IS SELF-CALIBRATION, with four properties this file pins:

* a fresh MEASUREMENT always beats the config number -- the config is a seed,
  not a belief;
* the estimate is an EMA, so it tracks a regime change (pinned -> file-backed)
  instead of being pinned to whichever boot happened to write the constant;
* it is BOUNDED, so one pathological outlier cannot drive the break-even so high
  that flipping stops entirely -- a policy that never flips is also broken;
* with no measurement yet it is byte-identical to today.
"""

import unittest

from sglang.srt.managers.phase_policy import (
    DEFAULT_FLIP_COST_S,
    FlipCostEstimator,
    break_even_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

TP_TOK_S = 1681.0
PP_TOK_S = 7245.5
#: The live boot's own numbers.
STALE_COST_S = 3.2
MEASURED_COST_S = 22.5
LOGGED_N = 7004


class TestTheGapIsReal(CustomTestCase):
    def test_the_live_boot_break_even_reproduces(self):
        """Anchor the fixture to the specimen: N=7004 at 3.2s."""
        self.assertEqual(break_even_tokens(STALE_COST_S, TP_TOK_S, PP_TOK_S), LOGGED_N)

    def test_the_measured_cost_gives_a_materially_different_window(self):
        """Fed 3.2 vs fed 22.5 must not be a rounding difference."""
        stale = break_even_tokens(STALE_COST_S, TP_TOK_S, PP_TOK_S)
        real = break_even_tokens(MEASURED_COST_S, TP_TOK_S, PP_TOK_S)
        self.assertGreater(real / stale, 6.5)
        self.assertLess(real / stale, 7.5)


class TestTheEstimator(CustomTestCase):
    def test_with_no_measurement_it_is_the_seed(self):
        """Byte-identical to today until something is measured."""
        e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
        self.assertEqual(e.value(), DEFAULT_FLIP_COST_S)
        self.assertFalse(e.calibrated)

    def test_ONE_measurement_beats_the_config_outright(self):
        """A fresh measurement is never averaged against a belief.

        The config is a seed for the un-measured case, not a prior to be
        blended with evidence -- blending is how a 3.2s constant would have
        survived a 22s reality for several flips.
        """
        e = FlipCostEstimator(seed_s=3.2)
        e.observe(22.5)
        self.assertTrue(e.calibrated)
        self.assertAlmostEqual(e.value(), 22.5, places=6)

    def test_it_tracks_a_regime_change(self):
        e = FlipCostEstimator(seed_s=3.2)
        for _ in range(12):
            e.observe(22.5)
        self.assertGreater(e.value(), 20.0)
        for _ in range(12):
            e.observe(3.1)
        self.assertLess(e.value(), 6.0, "it must come back down too")

    def test_a_single_outlier_cannot_freeze_flipping(self):
        """BOUNDED. One 600s pathological sample must not price flips out.

        A break-even that large would stop flipping altogether, which is not a
        conservative failure -- it is a different one.
        """
        e = FlipCostEstimator(seed_s=3.2)
        for _ in range(10):
            e.observe(22.5)
        before = e.value()
        e.observe(600.0)
        self.assertLess(
            e.value(),
            before * 3.0,
            "one outlier moved the estimate more than the bound allows",
        )

    def test_the_bound_is_symmetric_against_a_zero(self):
        """A bogus 0 s must not price flips as free either."""
        e = FlipCostEstimator(seed_s=3.2)
        for _ in range(10):
            e.observe(22.5)
        e.observe(0.0)
        self.assertGreater(e.value(), 5.0)

    def test_nonsense_samples_are_ignored_not_clamped(self):
        """CAN-FAIL: a negative or NaN is not evidence at all."""
        e = FlipCostEstimator(seed_s=3.2)
        e.observe(22.5)
        v = e.value()
        for bad in (-1.0, float("nan"), float("inf")):
            e.observe(bad)
        self.assertAlmostEqual(e.value(), v, places=6)

    def test_the_estimate_drives_the_window(self):
        """End to end: the number the economics actually consumes moves."""
        e = FlipCostEstimator(seed_s=3.2)
        n_before = break_even_tokens(e.value(), TP_TOK_S, PP_TOK_S)
        for _ in range(8):
            e.observe(22.5)
        n_after = break_even_tokens(e.value(), TP_TOK_S, PP_TOK_S)
        self.assertEqual(n_before, LOGGED_N)
        self.assertGreater(n_after, 40_000)


if __name__ == "__main__":
    unittest.main()


class TestTheHookFeedsThePolicy(CustomTestCase):
    """End to end through the module-level hook the seam will call.

    The #758 refill emitter lives on the harvest tip (da818719fe), not on this
    base, so the one-line call site goes in with that lineage. What is pinned
    here is that the hook EXISTS, is inert before use, and actually moves the
    number the economics consumes -- so wiring it is a one-liner and not a
    redesign.
    """

    def setUp(self):
        from sglang.srt.managers import phase_policy as pp

        self._saved = pp._FLIP_COST_ESTIMATOR
        pp._FLIP_COST_ESTIMATOR = None

    def tearDown(self):
        from sglang.srt.managers import phase_policy as pp

        pp._FLIP_COST_ESTIMATOR = self._saved

    def test_the_hook_is_inert_before_the_policy_is_built(self):
        """It may be called from anywhere, at any time, safely."""
        from sglang.srt.managers.phase_policy import observe_flip_cost

        observe_flip_cost(22.5)  # must not raise

    def test_observations_move_the_break_even(self):
        from sglang.srt.managers import phase_policy as pp

        pp._FLIP_COST_ESTIMATOR = pp.FlipCostEstimator(seed_s=STALE_COST_S)
        n_before = break_even_tokens(
            pp._FLIP_COST_ESTIMATOR.value(), TP_TOK_S, PP_TOK_S
        )
        for _ in range(5):
            pp.observe_flip_cost(MEASURED_COST_S)
        n_after = break_even_tokens(pp._FLIP_COST_ESTIMATOR.value(), TP_TOK_S, PP_TOK_S)
        self.assertEqual(n_before, LOGGED_N)
        self.assertGreater(n_after, 40_000)
        self.assertTrue(pp.flip_cost_estimator().calibrated)
