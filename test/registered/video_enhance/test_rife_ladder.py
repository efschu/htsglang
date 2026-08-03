"""The RIFE version ladder and its auto-selection policy (#460).

Falsifier-first. Every claim here is pinned by a *pair* of runs that differ in
one thing: a frontier cell present or absent, a budget above or below a
measured cost, a pin set or not. A test that only asserted the version it
expected would pass against a selector that always returned that version, so
each one asserts the flip as well.

The frontier used is the real ticket-V seed
(``/spinning/gpu-battery-results/2026-08-03_ticketV/RESULTS.md`` §3) wherever a
measured number is wanted, and a synthetic table wherever the point is the
*shape* of the answer rather than its value. The seed has exactly two versions
in it -- 4.6 and 4.26 -- which is the whole reason the "never auto-pick an
unmeasured variant" rule has teeth on this rig today: six of the eight rungs
are absent.

Everything is CPU arithmetic. No torch, no device.
"""

import unittest
from pathlib import Path

from sglang.srt.planner.cost_model import Provenance, Rate
from sglang.srt.video_enhance.frame_math import R4K, R1080P, Resolution
from sglang.srt.video_enhance.rife import KNOWN_WEIGHT_SHA256, SUPPORTED_VERSIONS
from sglang.srt.video_enhance.rife_ladder import (
    CARD_3080,
    CARD_5090,
    DEFAULT_QUALITY_RANK,
    LadderError,
    RifeFrontier,
    RifeLadder,
    RifeVariant,
    VramClass,
    WeightState,
    default_ladder,
    seeded_frontier,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

#: A directory that does not exist, so no rung is PRESENT and every rung is
#: PINNED. That is deterministic on any host, which a test that read the real
#: weight cache would not be.
NO_WEIGHTS_ON_DISK = Path("/nonexistent/rife-weights-for-a-hermetic-test")


def ladder(**kwargs) -> RifeLadder:
    kwargs.setdefault("weight_dir", NO_WEIGHTS_ON_DISK)
    return default_ladder(**kwargs)


def synthetic_frontier(
    rows,
    *,
    resolution: Resolution = R4K,
    card: str = CARD_5090,
    scale: float = 1.0,
) -> RifeFrontier:
    """``{version: ms}`` -> a frontier with those cells measured and no others."""
    return RifeFrontier(
        cells={
            (version, card, str(resolution), scale): Rate.measured(
                ms, "synthetic table for a unit test", unit="ms"
            )
            for version, ms in rows.items()
        },
        source="synthetic",
    )


class TheRegistryTest(CustomTestCase):
    """What the ladder is willing to hold at all."""

    def test_every_vendored_version_is_a_rung(self):
        rungs = {v.version for v in ladder().variants}
        self.assertEqual(rungs, set(SUPPORTED_VERSIONS))
        self.assertEqual(len(rungs), 8)

    def test_rungs_are_pinned_when_nothing_is_on_disk(self):
        for variant in ladder().variants:
            self.assertIs(variant.weight_state, WeightState.PINNED)
            self.assertEqual(
                variant.weight_sha256, KNOWN_WEIGHT_SHA256[variant.version]
            )
            self.assertTrue(variant.runnable)

    def test_a_rung_present_on_disk_reports_present(self):
        # The flip of the row above: a directory where the file validates
        # against its pin. Built by hand rather than by touching the real
        # cache, so the test is hermetic, with the pin patched to this file's
        # own digest -- which is also the falsifier for the sidecar check,
        # because the *unpatched* run below must NOT report PRESENT.
        import json
        import tempfile
        from unittest import mock

        from sglang.srt.video_enhance import rife

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payload = b"not a real checkpoint, but a real sha256"
            path = directory / rife.weight_filename("4.6")
            path.write_bytes(payload)
            digest = rife.sha256_file(path)
            (directory / (path.name + ".json")).write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "version": "4.6",
                        "source_url": "test",
                        "sha256": digest,
                        "size_bytes": len(payload),
                    }
                )
            )
            # Real pin, wrong bytes: the sidecar does not validate and the rung
            # falls back to PINNED rather than claiming a file it cannot vouch
            # for.
            unmatched = default_ladder(weight_dir=directory, versions=("4.6",))
            self.assertIs(unmatched.variants[0].weight_state, WeightState.PINNED)

            with mock.patch.dict(
                rife.KNOWN_WEIGHT_SHA256, {"4.6": digest}, clear=False
            ):
                matched = default_ladder(weight_dir=directory, versions=("4.6",))
            self.assertIs(matched.variants[0].weight_state, WeightState.PRESENT)
            self.assertEqual(matched.variants[0].weight_path, path)

    def test_registry_rejects_a_rung_with_no_weights_and_no_pin(self):
        # (d) of the ticket. Hand-construct the bad entry: default_ladder skips
        # it, so the refusal has to live in the registry's own invariant.
        bad = RifeVariant(
            version="4.6",
            quality_rank=0,
            vram_class=VramClass.HEADLESS,
            weight_state=WeightState.UNAVAILABLE,
            weight_path=None,
            weight_sha256=None,
        )
        with self.assertRaises(LadderError) as ctx:
            RifeLadder(variants=(bad,), frontier=seeded_frontier())
        message = str(ctx.exception)
        self.assertIn("neither a checkpoint on disk nor a sha256", message)
        self.assertIn("fetch_rife_weights.py", message)

        # The flip: the same entry with a pin is accepted.
        good = RifeVariant(
            version="4.6",
            quality_rank=0,
            vram_class=VramClass.HEADLESS,
            weight_state=WeightState.PINNED,
            weight_path=None,
            weight_sha256=KNOWN_WEIGHT_SHA256["4.6"],
        )
        RifeLadder(variants=(good,), frontier=seeded_frontier())

    def test_registry_rejects_an_unvendored_rung(self):
        bad = RifeVariant(
            version="4.25.lite",
            quality_rank=0,
            vram_class=VramClass.LITE,
            weight_state=WeightState.PINNED,
            weight_path=None,
            weight_sha256="0" * 64,
        )
        with self.assertRaises(LadderError) as ctx:
            RifeLadder(variants=(bad,), frontier=seeded_frontier())
        self.assertIn("no vendored IFNet", str(ctx.exception))

    def test_a_rung_with_no_quality_rank_is_refused_not_guessed(self):
        # The shipped ranking covers every vendored version, so the refusal is
        # exercised by removing one -- which is exactly what adding a ninth
        # vendored architecture without ranking it would do.
        from unittest import mock

        from sglang.srt.video_enhance import rife_ladder as module

        thinned = {k: v for k, v in DEFAULT_QUALITY_RANK.items() if k != "4.18"}
        with mock.patch.object(module, "DEFAULT_QUALITY_RANK", thinned):
            with self.assertRaises(LadderError) as ctx:
                default_ladder(weight_dir=NO_WEIGHTS_ON_DISK, versions=("4.6", "4.18"))
            self.assertIn("no quality rank for '4.18'", str(ctx.exception))
            # The flip: supply the missing rank and the same call succeeds.
            built = default_ladder(
                weight_dir=NO_WEIGHTS_ON_DISK,
                versions=("4.6", "4.18"),
                quality_ranks={"4.18": 1},
            )
        self.assertEqual({v.version for v in built.variants}, {"4.6", "4.18"})

    def test_vram_classes_follow_the_architecture(self):
        classes = {v.version: v.vram_class for v in ladder().variants}
        self.assertIs(classes["4.6"], VramClass.HEADLESS)
        self.assertIs(classes["4.17.lite"], VramClass.LITE)
        self.assertIs(classes["4.18"], VramClass.STANDARD)
        self.assertIs(classes["4.26"], VramClass.DEEP)

    def test_quality_order_is_labelled_an_assumption_everywhere_it_appears(self):
        report = ladder().report().as_dict()
        self.assertIn("ASSUMPTION", report["quality_basis"])
        for entry in report["variants"]:
            self.assertIn("ASSUMPTION", entry["quality_basis"])
        # And the order itself is the user's: the lite family and 4.1x above
        # 4.6, which is the whole claim the ranking encodes.
        order = [entry["version"] for entry in report["variants"]]
        self.assertEqual(order[-1], "4.6")
        self.assertLess(order.index("4.15.lite"), order.index("4.6"))
        self.assertLess(order.index("4.18"), order.index("4.17"))


class TheSeededFrontierTest(CustomTestCase):
    """What ticket V actually measured, and what it did not."""

    def test_only_46_and_426_are_measured(self):
        frontier = seeded_frontier()
        self.assertEqual(frontier.versions(), ("4.26", "4.6"))
        for version in ("4.15", "4.15.lite", "4.16.lite", "4.17", "4.17.lite", "4.18"):
            cell = frontier.rate(version, CARD_5090, R4K, 1.0)
            self.assertIs(cell.provenance, Provenance.ABSENT)
            self.assertIn("no measured RIFE", cell.source)

    def test_the_seeded_numbers_are_the_ticket_v_numbers(self):
        frontier = seeded_frontier()
        self.assertAlmostEqual(
            frontier.rate("4.6", CARD_5090, R4K, 0.5).value, 11.359, places=3
        )
        self.assertAlmostEqual(
            frontier.rate("4.26", CARD_3080, R4K, 0.5).value, 67.520, places=3
        )
        self.assertAlmostEqual(
            frontier.rate("4.6", CARD_3080, R1080P, 1.0).value, 16.088, places=3
        )
        for cell in (
            frontier.rate("4.6", CARD_5090, R4K, 1.0),
            frontier.rate("4.26", CARD_5090, R4K, 1.0),
        ):
            self.assertIs(cell.provenance, Provenance.MEASURED)
            self.assertIn("ticket V", cell.source)

    def test_vram_peaks_are_measured_only_where_p4_ran(self):
        frontier = seeded_frontier()
        self.assertAlmostEqual(
            frontier.vram("4.26", CARD_5090, R4K, 1.0).value, 7855.0, places=1
        )
        self.assertIs(
            frontier.vram("4.26", CARD_5090, R4K, 0.5).provenance, Provenance.ABSENT
        )


class AutoSelectionTest(CustomTestCase):
    """(a), (b) and (c) of the ticket."""

    def test_never_picks_a_variant_with_an_absent_frontier(self):
        # (a). 4.18 outranks 4.6 and is unmeasured; 4.6 is measured. The
        # selector must take the lower-ranked measured one.
        selection = ladder().select(
            card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=1000.0
        )
        self.assertEqual(selection.version, "4.26")  # measured and rank 0
        self.assertIn("4.18", selection.measure_first)
        self.assertNotIn("4.26", selection.measure_first)

        # The flip: take 4.26's cells away and the same call must fall to 4.6
        # rather than climbing to the unmeasured 4.18 that outranks it.
        thin = RifeFrontier(
            cells={
                k: v
                for k, v in seeded_frontier().cells.items()
                if not k[0].startswith("4.26")
            },
            source="ticket V minus 4.26",
        )
        fallen = ladder(frontier=thin).select(
            card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=1000.0
        )
        self.assertEqual(fallen.version, "4.6")
        self.assertIn("4.18", fallen.measure_first)
        self.assertIn("4.26", fallen.measure_first)

    def test_picks_the_highest_ranked_variant_that_fits(self):
        # (b). Four synthetic rungs, all measured, monotone in cost and rank.
        frontier = synthetic_frontier(
            {"4.26": 40.0, "4.18": 30.0, "4.17": 20.0, "4.6": 10.0}
        )
        built = ladder(frontier=frontier)
        for budget, expected in (
            (45.0, "4.26"),
            (35.0, "4.18"),
            (25.0, "4.17"),
            (15.0, "4.6"),
        ):
            with self.subTest(budget=budget):
                selection = built.select(
                    card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=budget
                )
                self.assertEqual(selection.version, expected)
                self.assertIs(selection.provenance, Provenance.MEASURED)

        # Below the cheapest rung nothing is chosen, and the refusal names the
        # cheapest measured cost so the caller knows how far off it is.
        nothing = built.select(card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=5.0)
        self.assertIsNone(nothing.version)
        self.assertIn("cheapest measured rung is 4.6 at 10.000 ms", nothing.reason)

    def test_exactly_at_the_budget_fits(self):
        frontier = synthetic_frontier({"4.18": 20.0, "4.6": 10.0})
        built = ladder(frontier=frontier)
        self.assertEqual(
            built.select(
                card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=20.0
            ).version,
            "4.18",
        )
        self.assertEqual(
            built.select(
                card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=19.999
            ).version,
            "4.6",
        )

    def test_pin_wins_over_the_budget_and_over_the_ranking(self):
        # (c). 4.6 is measured and cheap; the pin asks for it even when 4.26
        # would have been chosen, and asks for 4.26 even when the budget
        # excludes it.
        built = ladder()
        pinned_down = built.select(
            card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=1000.0, pin="4.6"
        )
        self.assertEqual(pinned_down.version, "4.6")
        self.assertTrue(pinned_down.pinned)
        self.assertIn("was pinned explicitly", pinned_down.reason)

        pinned_over = built.select(
            card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=5.0, pin="4.26"
        )
        self.assertEqual(pinned_over.version, "4.26")
        self.assertIn("does NOT fit", pinned_over.reason)
        self.assertIn("the pin overrides the budget gate", pinned_over.reason)

        # Unpinned at the same budget picks nothing at all -- which is what
        # makes the two runs above a pair rather than a coincidence.
        self.assertIsNone(
            built.select(
                card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=5.0
            ).version
        )

    def test_pinning_an_unmeasured_variant_is_allowed_and_labelled(self):
        selection = ladder().select(
            card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=1000.0, pin="4.17.lite"
        )
        self.assertEqual(selection.version, "4.17.lite")
        self.assertIs(selection.provenance, Provenance.ABSENT)
        self.assertIn("unmeasured", selection.reason)
        self.assertIn("no throughput guarantee", selection.reason)

    def test_no_budget_still_requires_a_measurement(self):
        # "No budget" is not "anything goes": with an empty frontier there is
        # still nothing to choose.
        empty = ladder(frontier=RifeFrontier(source="nobody measured anything"))
        selection = empty.select(
            card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=None
        )
        self.assertIsNone(selection.version)
        self.assertIn("has a measured frontier", selection.reason)
        self.assertIn("Run TICKET_460's frontier sweep", selection.reason)
        self.assertEqual(len(selection.measure_first), 8)

        # The flip: one measured cell and the same call answers.
        one = ladder(frontier=synthetic_frontier({"4.6": 10.0}))
        self.assertEqual(
            one.select(
                card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=None
            ).version,
            "4.6",
        )

    def test_several_cards_are_judged_on_the_slowest(self):
        # Regime A replicates the chain, so a version too slow on the weakest
        # card drags the aggregate. 4.26 fits on the 5090 alone at 4K s0.5
        # (25.373 ms) and does not fit once the 3080 (67.520 ms) is in play.
        # 4K scale 0.5: 4.26 costs 25.373 ms on the 5090 and 67.520 on the
        # 3080; 4.6 costs 11.359 and 31.999. A 35 ms budget admits 4.26 on the
        # 5090 alone and only 4.6 once the 3080 has to run the same chain.
        built = ladder()
        self.assertEqual(
            built.select(
                card=CARD_5090, resolution=R4K, scale=0.5, budget_ms=35.0
            ).version,
            "4.26",
        )
        both = built.select(
            card=(CARD_5090, CARD_3080), resolution=R4K, scale=0.5, budget_ms=35.0
        )
        self.assertEqual(both.version, "4.6")
        self.assertIn("worst of 2 cards", both.rows[0].source)

    def test_a_version_unmeasured_on_one_card_is_unmeasured_for_the_set(self):
        frontier = RifeFrontier(
            cells={
                ("4.6", CARD_5090, str(R4K), 1.0): Rate.measured(10.0, "t", unit="ms"),
                ("4.6", CARD_3080, str(R4K), 1.0): Rate.measured(30.0, "t", unit="ms"),
                ("4.18", CARD_5090, str(R4K), 1.0): Rate.measured(11.0, "t", unit="ms"),
            },
            source="one card short",
        )
        built = ladder(frontier=frontier)
        self.assertEqual(
            built.select(
                card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=100.0
            ).version,
            "4.18",
        )
        across = built.select(
            card=(CARD_5090, CARD_3080), resolution=R4K, scale=1.0, budget_ms=100.0
        )
        self.assertEqual(across.version, "4.6")
        self.assertIn("4.18", across.measure_first)

    def test_reranking_moves_the_choice_and_nothing_else(self):
        frontier = synthetic_frontier({"4.26": 40.0, "4.6": 10.0})
        built = ladder(frontier=frontier)
        self.assertEqual(
            built.select(
                card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=50.0
            ).version,
            "4.26",
        )
        reranked = built.with_quality_ranks({"4.6": -1})
        self.assertEqual(
            reranked.select(
                card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=50.0
            ).version,
            "4.6",
        )

    def test_allowed_filters_the_ladder(self):
        built = ladder()
        selection = built.select(
            card=CARD_5090,
            resolution=R4K,
            scale=1.0,
            budget_ms=1000.0,
            allowed=("4.6", "4.18"),
        )
        self.assertEqual(selection.version, "4.6")
        self.assertNotIn("4.26", [row.variant.version for row in selection.rows])
        with self.assertRaises(LadderError):
            built.select(
                card=CARD_5090,
                resolution=R4K,
                scale=1.0,
                budget_ms=1000.0,
                allowed=("4.25.lite",),
            )

    def test_every_row_carries_a_verdict_and_a_source(self):
        selection = ladder().select(
            card=CARD_5090, resolution=R4K, scale=1.0, budget_ms=25.0
        )
        payload = selection.as_dict()
        self.assertEqual(len(payload["ladder"]), 8)
        for row in payload["ladder"]:
            self.assertTrue(row["verdict"])
            self.assertTrue(row["source"])
            self.assertIn(row["provenance"], {"measured", "estimate", "absent"})
        self.assertIn("ASSUMPTION", payload["quality_basis"])


class TheDefaultRankingTest(CustomTestCase):
    def test_covers_every_vendored_version(self):
        self.assertEqual(set(DEFAULT_QUALITY_RANK), set(SUPPORTED_VERSIONS))
        self.assertEqual(
            len(set(DEFAULT_QUALITY_RANK.values())), len(DEFAULT_QUALITY_RANK)
        )


if __name__ == "__main__":
    unittest.main()
