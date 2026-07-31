"""Per-stage rates in the shared cost library, and the bridge to the planner.

#333 Regime-B groundwork. A whole-chain rate per card can only ever rank
cards; it cannot say that one card is *relatively* better at interpolation
than at super-resolution, which is the only fact that makes splitting a stage
across cards worth doing. That fact needs a (stage, card, resolution) table,
and #348b's rule is that pricing lives in the shared library rather than in a
planner's private structure.

Everything here is CPU. The measurement that fills the table needs cards; the
shape of the table, the absence discipline and the comparative-advantage
arithmetic do not.
"""

import unittest

from sglang.srt.planner.cost_model import (
    AbsentRate,
    Provenance,
    StageRateTable,
    stage_rates_from_reports,
    stage_rates_from_samples,
)
from sglang.srt.video_enhance.frame_math import Resolution
from sglang.srt.video_enhance.shard_plan import MissingRateError, RateTable, StageKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def sample(stage, card, resolution, ms, **extra):
    row = {
        "stage": stage,
        "card": card,
        "resolution": resolution,
        "ms_per_frame": ms,
        "post": "P1",
        "dtype": "fp16",
    }
    row.update(extra)
    return row


# The shape of the real measurement, with the 5090 roughly 2.5x a 3080 on the
# convolutional stages and much closer on the cheap ones -- which is exactly
# the asymmetry a Regime-B assignment would exploit.
SAMPLES = [
    sample("sr", "1", "960x540", 35.58),
    sample("sr", "0", "960x540", 90.9),
    sample("resize", "1", "3840x2160", 6.48),
    sample("resize", "0", "3840x2160", 14.1),
    sample("rife", "1", "1920x1080", 3.06),
    sample("rife", "0", "1920x1080", 8.28),
    sample("encode", "1", "1920x1080", 1.56),
    sample("encode", "0", "1920x1080", 2.5),
]


class StageRateTableTest(CustomTestCase):
    def test_a_measured_cell_carries_its_provenance(self):
        table = stage_rates_from_samples(SAMPLES, source="p1")
        cell = table.rate("sr", "1", "960x540")
        self.assertEqual(cell.provenance, Provenance.MEASURED)
        self.assertAlmostEqual(cell.value, 35.58)
        self.assertEqual(cell.unit, "ms")

    def test_an_unmeasured_cell_is_a_named_absence_not_a_keyerror(self):
        """The planner is allowed to know that it does not know."""
        table = stage_rates_from_samples(SAMPLES)
        cell = table.rate("rife", "2", "1920x1080")
        self.assertTrue(cell.is_absent)
        self.assertIn("rife", cell.source)
        self.assertIn("'2'", cell.source)

    def test_requiring_an_absent_cell_raises_with_the_reason(self):
        table = stage_rates_from_samples(SAMPLES)
        with self.assertRaises(AbsentRate) as ctx:
            table.ms("rife", "2", "1920x1080")
        self.assertIn("no measurement", str(ctx.exception))

    def test_the_absence_lists_what_was_measured_for_that_pair(self):
        """So the reader learns whether the card or the resolution is missing."""
        table = stage_rates_from_samples(SAMPLES)
        cell = table.rate("sr", "1", "1920x1080")
        self.assertIn("960x540", cell.source)

    def test_a_nonpositive_time_is_recorded_as_an_absence(self):
        """A zero cell would hand that card an unbounded share of the work."""
        table = stage_rates_from_samples([sample("rife", "2", "1920x1080", 0.0)])
        self.assertTrue(table.rate("rife", "2", "1920x1080").is_absent)

    def test_a_later_good_measurement_supersedes_an_earlier_failure(self):
        table = stage_rates_from_samples(
            [
                sample("rife", "2", "1920x1080", 0.0),
                sample("rife", "2", "1920x1080", 8.28),
            ]
        )
        self.assertAlmostEqual(table.ms("rife", "2", "1920x1080"), 8.28)

    def test_a_later_failure_does_not_erase_a_good_measurement(self):
        """A transient must not delete a number the rig actually produced."""
        table = stage_rates_from_samples(
            [
                sample("rife", "2", "1920x1080", 8.28),
                sample("rife", "2", "1920x1080", float("nan")),
            ]
        )
        self.assertAlmostEqual(table.ms("rife", "2", "1920x1080"), 8.28)

    def test_a_nan_time_is_recorded_as_an_absence(self):
        """A failed probe point writes NaN; NaN > 0 is False, so it lands here."""
        table = stage_rates_from_samples(
            [sample("rife", "2", "1920x1080", float("nan"))]
        )
        self.assertTrue(table.rate("rife", "2", "1920x1080").is_absent)

    def test_repeated_cells_keep_the_fastest(self):
        """A slow repeat is contention, not capability."""
        table = stage_rates_from_samples(
            [
                sample("sr", "1", "960x540", 35.58),
                sample("sr", "1", "960x540", 51.2),
                sample("sr", "1", "960x540", 34.9),
            ]
        )
        self.assertAlmostEqual(table.ms("sr", "1", "960x540"), 34.9)

    def test_coverage_names_the_pairs_a_plan_would_need_and_nobody_measured(self):
        table = stage_rates_from_samples(SAMPLES)
        gaps = table.coverage(["sr", "rife"], ["0", "1", "2"])
        self.assertIn("sr on 2", gaps)
        self.assertIn("rife on 2", gaps)
        self.assertNotIn("sr on 1", gaps)

    def test_stages_and_cards_are_reported(self):
        table = stage_rates_from_samples(SAMPLES)
        self.assertEqual(table.stages, ("encode", "resize", "rife", "sr"))
        self.assertEqual(table.cards, ("0", "1"))

    def test_a_card_may_be_renamed_to_the_key_the_planner_indexes_by(self):
        """Probes record a device name, which is not unique on this rig."""
        table = stage_rates_from_samples(
            [sample("rife", "NVIDIA GeForce RTX 3080", "1920x1080", 8.28)],
            key_of={"NVIDIA GeForce RTX 3080": "2"},
        )
        self.assertEqual(table.cards, ("2",))


class ComparativeAdvantageTest(CustomTestCase):
    """The arithmetic Regime B exists for.

    "Which card is fastest" is the wrong question -- on this rig it is the
    same card for every stage. The question is where each card is *least*
    disadvantaged, because that is what decides whether specialising a stage
    beats replicating the whole chain.
    """

    def test_advantage_normalises_to_the_fastest_card(self):
        table = stage_rates_from_samples(SAMPLES)
        adv = table.advantage("sr", "960x540")
        self.assertAlmostEqual(adv["1"], 1.0)
        self.assertAlmostEqual(adv["0"], round(90.9 / 35.58, 4))

    def test_the_ratio_differs_by_stage_which_is_the_whole_point(self):
        """If every stage had the same ratio, Regime B could never beat
        Regime A and this table would not be worth measuring."""
        table = stage_rates_from_samples(SAMPLES)
        sr_ratio = table.advantage("sr", "960x540")["0"]
        encode_ratio = table.advantage("encode", "1920x1080")["0"]
        self.assertGreater(sr_ratio, 2.0)
        self.assertLess(encode_ratio, 2.0)
        # The 3080 is relatively much better at encoding than at SR, so an
        # assignment that moves encode onto it costs less than its whole-chain
        # rate would suggest.
        self.assertGreater(sr_ratio, encode_ratio)

    def test_advantage_of_an_unmeasured_stage_is_empty_not_invented(self):
        table = stage_rates_from_samples(SAMPLES)
        self.assertEqual(table.advantage("decode", "960x540"), {})


class ReportMergeTest(CustomTestCase):
    def test_reports_from_several_cards_merge_into_one_table(self):
        table = stage_rates_from_reports(
            [
                {
                    "host": {"card_name": "RTX 5090"},
                    "noise_floor_pct": 2.77,
                    "samples": [s for s in SAMPLES if s["card"] == "1"],
                },
                {
                    "host": {"card_name": "RTX 3080"},
                    "noise_floor_pct": 4.1,
                    "samples": [s for s in SAMPLES if s["card"] == "0"],
                },
            ]
        )
        self.assertEqual(table.cards, ("0", "1"))

    def test_the_merged_noise_floor_is_the_worst_contributing_one(self):
        """A comparison across two sessions is only as good as the worse."""
        table = stage_rates_from_reports(
            [
                {"noise_floor_pct": 2.77, "samples": []},
                {"noise_floor_pct": 4.1, "samples": []},
            ]
        )
        self.assertAlmostEqual(table.noise_floor_pct, 4.1)

    def test_a_report_with_no_floor_does_not_invent_one(self):
        table = stage_rates_from_reports([{"samples": []}])
        self.assertIsNone(table.noise_floor_pct)


class PlannerBridgeTest(CustomTestCase):
    """The path from a measured probe report into the shard planner.

    Before this, ``RateTable`` was only ever built by hand in tests, so every
    plan made on a real rig was made from numbers a human typed.
    """

    def test_a_stage_rate_table_becomes_a_planner_rate_table(self):
        shared = stage_rates_from_samples(SAMPLES)
        rates = RateTable.from_stage_rates(shared)
        self.assertAlmostEqual(rates.ms(StageKind.SR, "1", Resolution(960, 540)), 35.58)
        self.assertAlmostEqual(
            rates.ms(StageKind.RIFE, "0", Resolution(1920, 1080)), 8.28
        )

    def test_absent_cells_are_dropped_and_surface_at_the_point_of_use(self):
        shared = stage_rates_from_samples(
            SAMPLES + [sample("rife", "2", "1920x1080", 0.0)]
        )
        rates = RateTable.from_stage_rates(shared)
        with self.assertRaises(MissingRateError):
            rates.ms(StageKind.RIFE, "2", Resolution(1920, 1080))

    def test_a_stage_the_chain_does_not_have_is_ignored_not_an_error(self):
        """The probe grid is allowed to be wider than any one chain."""
        shared = stage_rates_from_samples(
            SAMPLES + [sample("denoise", "1", "960x540", 4.0)]
        )
        rates = RateTable.from_stage_rates(shared)
        self.assertEqual(sorted(rates.cards), ["0", "1"])

    def test_cards_can_be_restricted_to_the_ones_a_plan_offers(self):
        shared = stage_rates_from_samples(SAMPLES)
        rates = RateTable.from_stage_rates(shared, cards=["1"])
        self.assertEqual(rates.cards, ("1",))

    def test_an_empty_table_is_empty_rather_than_a_failure(self):
        rates = RateTable.from_stage_rates(StageRateTable(cells={}))
        self.assertEqual(len(rates), 0)


if __name__ == "__main__":
    unittest.main()
