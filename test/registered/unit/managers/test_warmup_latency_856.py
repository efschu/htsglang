"""#856: the warm-up cost of the no-carry flip is measured, not assumed.

The flip carries no KV, so the first requests after a cutover re-prefill from
the hierarchical cache. That is the price paid for deleting the KV mover, the
staging reserve and the resident carry, and the user named it as THE
validation metric: not "rows carried", but cutover-blocking time plus honest
warm-up cost as SERVED-REQUEST LATENCY.

Nothing measured it. The nearest prior art,
`regime_classifier.PhaseDwellGate.rounds_since_flip`, is a GATE deciding
whether a flip may happen and carries no latency at all.

THE CAN-FAIL DIRECTIONS ARE THE POINT of this file. An instrument that always
reports something reassuring is worse than none, so: `None` must survive where
nothing was compared, "has not flipped" must not fold into "has flipped and
settled", and the summary must say plainly when it has nothing to say.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.warmup_latency import (
    BANDS,
    NO_FLIP,
    STEADY,
    WarmupLatencyLedger,
    band_for,
)
from sglang.test.test_utils import CustomTestCase


class TestTheBandsPartitionCleanly(CustomTestCase):
    def test_every_round_count_lands_in_exactly_one_band(self):
        seen = set()
        for n in range(0, 200):
            seen.add(band_for(n))
        expected = {f"<={hi}" for hi in BANDS} | {STEADY}
        self.assertEqual(seen, expected)

    def test_the_boundaries_are_inclusive(self):
        self.assertEqual(band_for(1), "<=1")
        self.assertEqual(band_for(2), "<=4")
        self.assertEqual(band_for(64), "<=64")
        self.assertEqual(band_for(65), STEADY)

    def test_never_flipped_is_its_own_answer(self):
        # THE CAN-FAIL DIRECTION: folding this into STEADY would let an
        # instance that never flipped supply the control for a warm-up ratio.
        self.assertEqual(band_for(None), NO_FLIP)
        self.assertNotEqual(band_for(None), STEADY)

    def test_a_negative_round_count_is_not_a_measurement(self):
        self.assertEqual(band_for(-3), NO_FLIP)


class TestTheWarmupCostIsVisible(CustomTestCase):
    def test_a_warm_band_reads_as_a_ratio_against_this_instances_own_steady(self):
        led = WarmupLatencyLedger()
        led.note_cutover()
        for _ in range(4):
            led.note_request(1, 4.0)
        for _ in range(20):
            led.note_request(200, 1.0)
        self.assertAlmostEqual(led.warmup_ratio("<=1"), 4.0)
        self.assertIn("4.00x steady", led.summary())

    def test_no_warmup_cost_reads_as_one(self):
        # The other direction: an instrument that only ever reports a penalty
        # cannot show the design working.
        led = WarmupLatencyLedger()
        led.note_cutover()
        led.note_request(1, 1.0)
        led.note_request(200, 1.0)
        self.assertAlmostEqual(led.warmup_ratio("<=1"), 1.0)


class TestItRefusesToInventAControl(CustomTestCase):
    def test_no_steady_samples_yields_None_not_one(self):
        # A ratio against an absent control reads as "no warm-up cost" while
        # meaning "nothing was compared" -- the #606 defaulted-measurement
        # shape, in the one number this ticket is judged on.
        led = WarmupLatencyLedger()
        led.note_cutover()
        led.note_request(1, 9.0)
        self.assertIsNone(led.warmup_ratio("<=1"))

    def test_no_warm_samples_yields_None(self):
        led = WarmupLatencyLedger()
        led.note_request(200, 1.0)
        self.assertIsNone(led.warmup_ratio("<=1"))

    def test_a_zero_steady_mean_does_not_divide(self):
        led = WarmupLatencyLedger()
        led.note_request(200, 0.0)
        led.note_request(1, 5.0)
        self.assertIsNone(led.warmup_ratio("<=1"))


class TestItNeverBreaksTheServingPath(CustomTestCase):
    def test_an_unreadable_latency_is_dropped_not_recorded(self):
        led = WarmupLatencyLedger()
        for bad in (None, "n/a", object(), float("nan"), -1.0):
            with self.subTest(latency=bad):
                led.note_request(1, bad)
        self.assertEqual(led.count("<=1"), 0)

    def test_a_good_sample_still_lands_after_bad_ones(self):
        # The can-fail partner: a guard that dropped everything would pass the
        # test above and measure nothing for ever.
        led = WarmupLatencyLedger()
        led.note_request(1, "n/a")
        led.note_request(1, 2.5)
        self.assertEqual(led.count("<=1"), 1)
        self.assertAlmostEqual(led.mean("<=1"), 2.5)


class TestSilenceIsImpossible(CustomTestCase):
    """A silent instrument is indistinguishable from an absent one."""

    def test_an_empty_ledger_still_says_so(self):
        line = WarmupLatencyLedger().summary()
        self.assertIn("nothing measured yet", line)
        self.assertIn("0 cutover", line)

    def test_the_summary_names_the_cutover_count(self):
        led = WarmupLatencyLedger()
        led.note_cutover()
        led.note_cutover()
        led.note_request(1, 1.0)
        self.assertIn("after 2 cutover(s)", led.summary())


if __name__ == "__main__":
    unittest.main()
