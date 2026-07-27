"""CPU unit tests for per-window results and the four-valued verdict (#218)."""

import unittest

from sglang.srt.planner.comparison import (
    CHANGED,
    NOT_COMPARABLE,
    UNKNOWN,
    VERDICTS,
    WITHIN_NOISE,
    ArmResult,
    HeadlineRefused,
    NoiseFloor,
    WindowResult,
    compare_arms,
    compare_metric,
    headline,
)
from sglang.srt.planner.scenarios import SCENARIOS
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

#: The floor the noise_floor scenario produces on this rig.
FLOOR = NoiseFloor(
    relative={"ms_per_verify_round": 0.0037, "tok_s": 0.042},
    source="noise_floor scenario, 5 interleaved repeats",
)


def arm(label, resident=120_000, accept=2.0, **windows):
    return ArmResult(
        label=label,
        scenario="power_target_sweep",
        conditions={"accept_length": accept, "resident_tokens": resident,
                    "batch_size": 1, "prompt_set": "fixed-64"},
        windows=[
            WindowResult(
                window=k,
                metrics=v,
                excluded_from_headline=(k == "restore_transient"),
            )
            for k, v in windows.items()
        ],
    )


class TestTheVocabularyIsClosed(CustomTestCase):
    def test_there_are_exactly_four_verdicts(self):
        self.assertEqual(
            set(VERDICTS), {CHANGED, WITHIN_NOISE, NOT_COMPARABLE, UNKNOWN}
        )

    def test_every_comparison_returns_one_of_them(self):
        a = arm("A", steady={"ms_per_verify_round": 2.30})
        b = arm("B", steady={"ms_per_verify_round": 1.73})
        for c in compare_arms(a, b, noise=FLOOR):
            self.assertIn(c.verdict, VERDICTS)


class TestChangedAgainstTheFloor(CustomTestCase):
    def test_a_real_effect_is_changed_with_its_direction(self):
        """#210's measured decode gain: 2.296 -> 1.732 ms."""
        a = arm("capacity", steady={"ms_per_verify_round": 2.296})
        b = arm("speed", steady={"ms_per_verify_round": 1.732})
        c = compare_metric(a, b, "ms_per_verify_round", "steady", FLOOR)
        self.assertEqual(c.verdict, CHANGED)
        self.assertEqual(c.direction, "better")
        self.assertLess(c.relative_delta, -0.2)

    def test_a_lower_is_worse_metric_is_not_mislabelled(self):
        a = arm("A", steady={"tok_s": 100.0})
        b = arm("B", steady={"tok_s": 130.0})
        c = compare_metric(a, b, "tok_s", "steady", FLOOR, lower_is_better=False)
        self.assertEqual(c.verdict, CHANGED)
        self.assertEqual(c.direction, "better")

    def test_the_scenario_supplies_the_direction(self):
        a = arm("A", steady={"tok_s": 100.0})
        b = arm("B", steady={"tok_s": 130.0})
        out = compare_arms(a, b, scenario=SCENARIOS["power_target_sweep"],
                           noise=FLOOR)
        c = [x for x in out if x.metric == "tok_s"][0]
        self.assertEqual(c.direction, "better")


class TestWithinNoise(CustomTestCase):
    def test_a_difference_under_the_floor_claims_nothing(self):
        a = arm("A", steady={"ms_per_verify_round": 2.300})
        b = arm("B", steady={"ms_per_verify_round": 2.304})
        c = compare_metric(a, b, "ms_per_verify_round", "steady", FLOOR)
        self.assertEqual(c.verdict, WITHIN_NOISE)
        self.assertIn("nothing may be claimed", c.reason)
        self.assertEqual(c.direction, "")

    def test_the_same_delta_can_be_within_noise_for_one_metric_and_not_another(self):
        """The floors differ by an order of magnitude; one threshold for both
        would discard real effects on the finer metric."""
        a = arm("A", steady={"ms_per_verify_round": 100.0, "tok_s": 100.0})
        b = arm("B", steady={"ms_per_verify_round": 102.0, "tok_s": 102.0})
        fine = compare_metric(a, b, "ms_per_verify_round", "steady", FLOOR)
        coarse = compare_metric(a, b, "tok_s", "steady", FLOOR)
        self.assertEqual(fine.verdict, CHANGED)
        self.assertEqual(coarse.verdict, WITHIN_NOISE)


class TestNotComparable(CustomTestCase):
    def test_a_different_accept_length_makes_round_times_incomparable(self):
        """tok/s = accept length / round time: at a different accept length the
        round covers a different amount of work."""
        a = arm("A", accept=2.0, steady={"ms_per_verify_round": 2.30})
        b = arm("B", accept=3.4, steady={"ms_per_verify_round": 2.30})
        c = compare_metric(a, b, "ms_per_verify_round", "steady", FLOOR)
        self.assertEqual(c.verdict, NOT_COMPARABLE)
        self.assertIn("accept_length", c.reason)

    def test_a_different_prompt_set_makes_throughput_incomparable(self):
        a = arm("A", steady={"tok_s": 100.0})
        b = arm("B", steady={"tok_s": 130.0})
        b.conditions["prompt_set"] = "other"
        c = compare_metric(a, b, "tok_s", "steady", FLOOR)
        self.assertEqual(c.verdict, NOT_COMPARABLE)

    def test_measurement_drift_in_a_condition_is_tolerated(self):
        """Accept length is measured; it is never bit-identical between runs."""
        a = arm("A", accept=2.00, steady={"ms_per_verify_round": 2.30})
        b = arm("B", accept=2.01, steady={"ms_per_verify_round": 1.73})
        self.assertEqual(
            compare_metric(a, b, "ms_per_verify_round", "steady", FLOOR).verdict,
            CHANGED,
        )

    def test_a_transition_window_never_compares(self):
        a = arm("A", restore_transient={"ms_per_verify_round": 9.0})
        b = arm("B", restore_transient={"ms_per_verify_round": 3.0})
        c = compare_metric(a, b, "ms_per_verify_round", "restore_transient", FLOOR)
        self.assertEqual(c.verdict, NOT_COMPARABLE)
        self.assertIn("transition window", c.reason)

    def test_an_unmeasurable_condition_does_not_block_on_its_own(self):
        a = ArmResult(label="A", conditions={},
                      windows=[WindowResult("steady", {"tok_s": 100.0})])
        b = ArmResult(label="B", conditions={},
                      windows=[WindowResult("steady", {"tok_s": 130.0})])
        self.assertNotEqual(
            compare_metric(a, b, "tok_s", "steady", FLOOR).verdict, NOT_COMPARABLE
        )


class TestUnknownIsNotWithinNoise(CustomTestCase):
    def test_without_a_floor_the_verdict_is_unknown_not_no_effect(self):
        a = arm("A", steady={"ms_per_verify_round": 2.30})
        b = arm("B", steady={"ms_per_verify_round": 2.31})
        c = compare_metric(a, b, "ms_per_verify_round", "steady", noise=None)
        self.assertEqual(c.verdict, UNKNOWN)
        self.assertIn("noise_floor scenario", c.reason)
        self.assertIn("no resolution", c.reason)

    def test_a_floor_for_another_metric_does_not_transfer(self):
        a = arm("A", steady={"j_per_token": 1.0})
        b = arm("B", steady={"j_per_token": 2.0})
        self.assertEqual(
            compare_metric(a, b, "j_per_token", "steady", FLOOR).verdict, UNKNOWN
        )

    def test_a_missing_window_is_unknown(self):
        a = arm("A", steady={"tok_s": 100.0})
        b = arm("B", other={"tok_s": 100.0})
        c = compare_metric(a, b, "tok_s", "steady", FLOOR)
        self.assertEqual(c.verdict, UNKNOWN)
        self.assertIn("no window", c.reason)

    def test_a_metric_measured_in_only_one_arm_is_unknown(self):
        a = arm("A", steady={"tok_s": 100.0, "ms_per_verify_round": 2.0})
        b = arm("B", steady={"tok_s": 100.0})
        c = compare_metric(a, b, "ms_per_verify_round", "steady", FLOOR)
        self.assertEqual(c.verdict, UNKNOWN)


class TestHeadlineRefusesTransientWindows(CustomTestCase):
    def test_a_steady_window_yields_a_headline(self):
        a = arm("A", steady_after={"ms_per_verify_round": 2.30})
        self.assertAlmostEqual(
            headline(a, "steady_after", "ms_per_verify_round"), 2.30
        )

    def test_the_restore_transient_is_refused_structurally(self):
        """It contains the speculative-resume backfill: a catch-up rate, not a
        serving rate. Refused rather than returned with a caveat."""
        a = arm("A", restore_transient={"ms_per_verify_round": 9.0})
        with self.assertRaises(HeadlineRefused) as cm:
            headline(a, "restore_transient", "ms_per_verify_round")
        self.assertIn("transition", str(cm.exception))

    def test_an_absent_window_is_refused_rather_than_defaulted(self):
        a = arm("A", steady={"tok_s": 1.0})
        with self.assertRaises(HeadlineRefused):
            headline(a, "missing", "tok_s")


class TestScenarioWindowsCarryTheRule(CustomTestCase):
    def test_the_spill_scenarios_declare_the_transient_as_excluded(self):
        for key in ("ram_clock_spill", "spill_latency_under_concurrency"):
            windows = {w.key: w for w in SCENARIOS[key].windows}
            self.assertTrue(windows["restore_transient"].exclude_from_headline)
            self.assertFalse(windows["steady_after"].exclude_from_headline)


if __name__ == "__main__":
    unittest.main()
