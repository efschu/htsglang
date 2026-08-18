"""#677 follow-up: does the per-sample clamp track BOTH regime directions?

THE QUESTION ASKED. This rig has two measured flip regimes -- 2.4-3.1 s per
leg pinned/warm, 22.5 s cold file-backed (NOTE_690_refill_commit_split.md) --
a 7.3x to 9.4x step. The concern was that ``FlipCostEstimator``'s clamp, a
multiplicative band of ``OUTLIER_BAND`` around the current estimate, might be
tuned for one direction and leave the estimator STICKY when the regime jumps
upward.

MEASURED, and the concern is NOT borne out. Driving synthetic sequences
through the estimator on the base commit:

    3.1 -> 22.5  (7.3x)   8 samples to within 10%
    2.4 -> 22.5  (9.4x)   8 samples
    22.5 -> 3.1           13 samples

The clamp binds only twice on the way up. The asymmetry is real -- the band is
symmetric multiplicatively but the EMA step is not, ``alpha*(B*x - x)`` going
up against ``alpha*(x/B - x)`` coming down, i.e. x1.6 per sample up against
x0.8 down -- and it favours the SAFE direction: the estimator is quicker to
declare flips expensive than to declare them cheap. Under-pricing is the
defect #677 exists to fix; over-pricing for a few extra samples is not.
Both directions are pinned below so a future edit cannot quietly break them.

WHAT IS ACTUALLY BROKEN, found while pinning the above: THE FIRST SAMPLE HAS
NO PLAUSIBILITY GUARD AT ALL. The clamp begins at sample two, because sample
one takes the documented "a measurement beats the seed OUTRIGHT" path and is
written straight into ``_ema``. So the band is derived from a value nothing
checked, and a bad first reading poisons the band it will be measured against:

  * ``observe(0.0)`` FIRST sets ``_ema = 0.0``. The band is then
    ``[0/B, 0*B] == [0, 0]``, every later sample is clamped to zero, and the
    estimator is LATCHED -- measured: 50 subsequent 22.5 s flips move it not
    at all. ``break_even_tokens`` then raises ``PhasePolicyError: flip cost
    must be positive``, from inside config construction.
  * ``observe(1e-6)`` FIRST -- a no-op or short-circuited refill leg, and the
    call site brackets exactly that (``phase_flip_boot.py``, ``arena_refill``)
    -- yields a break-even of ZERO tokens, so every pending prefill funds a
    flip. That is the #748 churn signature reached from the opposite end, and
    it takes 40 samples to recover: at 22.5 s a leg, a quarter of an hour.

The mid-stream zero IS already guarded, and pinned by
``test_flip_cost_calibration_677.py::test_the_bound_is_symmetric_against_a_zero``.
It is only the first one that is unprotected.

THE FIX, in three parts, each of which keeps every existing #677 pin:

1. a zero-second leg is REJECTED, not clamped, on the same argument the
   module already applies to negatives and NaN: "they are not implausible
   readings, they are not readings". A leg brackets a multi-GiB copy; zero is
   a broken timer, never a fast flip.
2. an absolute floor under the estimate, so the band can never collapse
   toward zero from a tiny-but-positive reading. Set far below the fastest
   leg this rig has ever measured (997 ms, the pinned era) so it cannot
   distort a real one.
3. THE CLAMP BOUNDS AN OUTLIER, NOT A REGIME. Its stated purpose is that "one
   pathological reading" must not price flipping out -- ONE. Rate-limiting a
   SUSTAINED change is the EMA's job, and doing it twice is what costs eight
   samples on a step the seam has already reported twice. So a second
   CONSECUTIVE out-of-band sample on the same side is taken as confirmation
   and accepted outright. A lone 600 s outlier is still clamped, because it
   is lone.
"""

import unittest

from sglang.srt.managers.phase_policy import (
    DEFAULT_FLIP_COST_S,
    FlipCostEstimator,
    break_even_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# The two measured regimes, NOTE_690_refill_commit_split.md.
WARM_S = 3.1
WARM_FAST_S = 2.4
COLD_S = 22.5
# The policy's measured rates, so break-even numbers here mean what the boot
# log means by them.
TP_TOK_S = 1681.0
PP_TOK_S = 7245.5
#: The fastest per-rank leg ever measured on this rig (pinned era, 997 ms).
FASTEST_MEASURED_LEG_S = 0.997


def _converge(first, then, target, tol=0.10, cap=200):
    """Samples needed to get within ``tol`` of ``target`` after a regime step."""
    e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
    e.observe(first)
    n = 0
    while n < cap and abs(e.value() - target) / target > tol:
        e.observe(then)
        n += 1
    return n, e.value()


class TestBothDirectionsAreTracked677(CustomTestCase):
    """The question as asked: neither direction may latch."""

    def test_the_upward_regime_jump_is_tracked(self):
        n, v = _converge(WARM_S, COLD_S, COLD_S)
        self.assertLess(n, 200, "the estimator never reached the cold regime")
        self.assertGreater(v, COLD_S * 0.9)

    def test_the_widest_upward_jump_is_tracked(self):
        """9.4x, the worst step this rig can produce."""
        n, v = _converge(WARM_FAST_S, COLD_S, COLD_S)
        self.assertLess(n, 200)
        self.assertGreater(v, COLD_S * 0.9)

    def test_the_downward_regime_jump_is_tracked(self):
        n, v = _converge(COLD_S, WARM_S, WARM_S)
        self.assertLess(n, 200)
        self.assertLess(v, WARM_S * 1.1)

    def test_neither_direction_is_slower_than_the_seam_can_afford(self):
        """A leg costs 22.5 s cold, so a convergence measured in dozens of
        samples is measured in QUARTER HOURS of mispriced flips. Four samples
        each way is the bound: the seam has reported the new regime twice by
        then, and a third and fourth confirm it."""
        up, _ = _converge(WARM_S, COLD_S, COLD_S)
        down, _ = _converge(COLD_S, WARM_S, WARM_S)
        self.assertLessEqual(up, 4, f"upward took {up} samples")
        self.assertLessEqual(down, 4, f"downward took {down} samples")


class TestTheFirstSampleIsGuarded677(CustomTestCase):
    """RED-FIRST: the clamp starts at sample two, so sample one poisons it."""

    def test_a_zero_first_sample_does_not_latch_the_estimator(self):
        """Measured on the base commit: 50 subsequent 22.5 s flips moved the
        estimate not at all, because the band was [0, 0]."""
        e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
        e.observe(0.0)
        for _ in range(10):
            e.observe(COLD_S)
        self.assertGreater(
            e.value(),
            1.0,
            "a single bogus zero pinned the estimator below every later measurement",
        )

    def test_a_zero_first_sample_never_reaches_the_break_even(self):
        """The consequence: break_even_tokens raises from inside config
        construction, because it refuses a non-positive cost."""
        e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
        e.observe(0.0)
        n = break_even_tokens(e.value(), TP_TOK_S, PP_TOK_S)
        self.assertGreater(n, 0)

    def test_a_zero_leg_is_rejected_not_recorded(self):
        """A leg brackets a multi-GiB copy. Zero is a broken timer, and the
        module's own rule for a non-reading is to ignore it."""
        e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
        e.observe(0.0)
        self.assertFalse(
            e.calibrated, "a zero-second flip leg was accepted as a measurement"
        )
        self.assertEqual(e.value(), DEFAULT_FLIP_COST_S)

    def test_a_near_zero_first_sample_does_not_zero_the_break_even(self):
        """1e-6 s -- a no-op refill leg -- priced the break-even at ZERO
        tokens, so every pending prefill funds a flip. That is #748's churn
        signature reached from the other end."""
        e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
        e.observe(1e-6)
        self.assertGreater(
            break_even_tokens(e.value(), TP_TOK_S, PP_TOK_S),
            0,
            "a near-zero leg made every token worth a flip",
        )

    def test_a_near_zero_first_sample_recovers_promptly(self):
        """Measured on the base commit: 40 samples, a quarter of an hour of
        mispriced flips at 22.5 s a leg."""
        e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
        e.observe(1e-6)
        n = 0
        while n < 200 and e.value() < COLD_S * 0.9:
            e.observe(COLD_S)
            n += 1
        self.assertLessEqual(n, 4, f"recovery took {n} samples")

    def test_the_floor_is_far_below_any_real_measurement(self):
        """CAN-FAIL: a floor that could bind on a genuine reading would be a
        second wrong constant, not a guard."""
        e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
        e.observe(FASTEST_MEASURED_LEG_S)
        self.assertAlmostEqual(e.value(), FASTEST_MEASURED_LEG_S, places=6)


class TestTheOutlierBoundSurvives677(CustomTestCase):
    """CAN-FAIL COUNTERWEIGHTS. Everything the clamp is FOR must still hold."""

    def test_a_lone_outlier_is_still_clamped(self):
        """#677's own pin, restated: one 600 s reading must not price flips
        out. It is clamped because it is LONE, which is exactly the
        distinction the fix draws."""
        e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
        for _ in range(10):
            e.observe(COLD_S)
        before = e.value()
        e.observe(600.0)
        self.assertLess(e.value(), before * 3.0)

    def test_a_lone_outlier_between_normal_samples_is_absorbed(self):
        """The confirmation counter must RESET on an in-band sample, or two
        outliers an hour apart would read as a regime.

        SOME upward drift is expected here and is not the defect: a seam that
        really does alternate 600 s and 22.5 s is bimodal, and an estimate
        between the two modes is the honest answer for it. What must never
        happen is the estimate ADOPTING the outlier -- that is the difference
        between a counter that resets and one that does not.
        """
        e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
        for _ in range(10):
            e.observe(COLD_S)
        for _ in range(4):
            e.observe(600.0)
            e.observe(COLD_S)
        self.assertLess(
            e.value(),
            150.0,
            "alternating outliers were adopted as a sustained 600s regime",
        )

    def test_a_sustained_600s_regime_is_eventually_believed(self):
        """The other side of the same rule, and it must be true: if the seam
        really does take 600 s a leg, refusing to believe it is the #677
        defect with a different number."""
        e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
        for _ in range(10):
            e.observe(COLD_S)
        for _ in range(6):
            e.observe(600.0)
        self.assertGreater(e.value(), 100.0)

    def test_nonsense_is_still_ignored(self):
        e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
        e.observe(COLD_S)
        v = e.value()
        for bad in (-1.0, float("nan"), float("inf"), "x", None):
            e.observe(bad)
        self.assertAlmostEqual(e.value(), v, places=6)

    def test_it_is_still_inert_until_it_measures(self):
        e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
        self.assertFalse(e.calibrated)
        self.assertEqual(e.value(), DEFAULT_FLIP_COST_S)

    def test_one_good_measurement_still_beats_the_seed_outright(self):
        e = FlipCostEstimator(seed_s=DEFAULT_FLIP_COST_S)
        e.observe(COLD_S)
        self.assertAlmostEqual(e.value(), COLD_S, places=6)


if __name__ == "__main__":
    unittest.main()
