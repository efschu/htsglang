"""Stage-pipeline pricing for the enhance chain (#457, desk half).

Falsifier-first, and for this module that mostly means *pairs of placements*:
a claim about which stage binds is only worth something if moving that stage
moves the answer, so every binding assertion is made twice with one stage
relocated between the runs.

Two tables are used. The synthetic one is deliberately lopsided -- round
numbers, one obviously expensive stage -- so an arithmetic slip is visible by
inspection. The ticket-V one is the real 2026-08-03 measurement
(``/spinning/gpu-battery-results/2026-08-03_ticketV/RESULTS.md``), and the test
that uses it pins the re-derived 1080p@25 -> 2160p@50 verdict that went into
the #451 design doc, so the document and the code cannot drift apart.

Everything is CPU arithmetic. No torch, no device.
"""

import unittest

from sglang.srt.planner.cost_model import Provenance, Rate
from sglang.srt.video_enhance.frame_math import (
    R4K,
    R8K,
    R1080P,
    PixelFormat,
    Resolution,
)
from sglang.srt.video_enhance.stage_pipeline import (
    EIGHT_K_FP16_MIB,
    UNPRICED_CHAIN_STAGES,
    CardProfile,
    barlink_link,
    best_placement,
    compare_regimes,
    host_bounce_links,
    price_placement,
    replicated_throughput,
    stage_table,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# --------------------------------------------------------------------------
# The real table
# --------------------------------------------------------------------------

TICKET_V = "ticket V 2026-08-03 RESULTS.md"

#: Where each stage's output goes next, and how much of it there is per frame
#: entering the chain. ``rife`` emits two frames per source frame at the
#: multiplier the target scenario asks for, which is why the encode boundary
#: carries twice a 4K frame rather than once.
GEOMETRY = {
    "decode": (R1080P, PixelFormat.NV12, 1.0),
    "sr": (R8K, PixelFormat.RGB_FP16, 1.0),
    "resize": (R4K, PixelFormat.RGB_FP16, 1.0),
    "rife": (R4K, PixelFormat.RGB_FP16, 2.0),
    "encode": (None, PixelFormat.RGB_FP16, 0.0),
}

#: The 8K fp16 intermediate must not cross a card. Declared on both stages so
#: the constraint is symmetric however the enumerator orders them.
CO_RESIDENT = {"sr": ("resize",), "resize": ("sr",)}


def ticket_v_stages(*, rife_5090: float, rife_3080: float):
    """The measured per-source-frame chain. ``resize`` on a 3080 is ABSENT."""
    return stage_table(
        [
            ("decode", {"5090": 4.254, "3080_x8": 7.140, "3080_x4": 7.140}),
            ("sr", {"5090": 25.424, "3080_x8": 90.343, "3080_x4": 90.343}),
            ("resize", {"5090": 24.367, "3080_x8": None, "3080_x4": None}),
            ("rife", {"5090": rife_5090, "3080_x8": rife_3080, "3080_x4": rife_3080}),
            # Encode runs twice per source frame at 2160p; the figures are the
            # ffmpeg host-round-trip fallback, which is the only encode path
            # that works on this rig (PyNvVideoCodec NVENC fails Error 8).
            ("encode", {"5090": 40.780, "3080_x8": 47.080, "3080_x4": 47.080}),
        ],
        source=TICKET_V,
        geometry=GEOMETRY,
        co_residency=CO_RESIDENT,
    )


def rig_cards():
    return host_bounce_links(x8_cards=("5090", "3080_x8"), x4_cards=("3080_x4",))


# --------------------------------------------------------------------------
# The synthetic table
# --------------------------------------------------------------------------


def toy_stages(*, heavy_on: float = 100.0, light: float = 10.0):
    """Three stages, two cards, one obviously expensive middle stage."""
    return stage_table(
        [
            ("first", {"a": light, "b": light}),
            ("middle", {"a": heavy_on, "b": heavy_on}),
            ("last", {"a": light, "b": light}),
        ],
        source="synthetic",
        geometry={
            "first": (Resolution(64, 64), PixelFormat.RGB_FP16, 1.0),
            "middle": (Resolution(64, 64), PixelFormat.RGB_FP16, 1.0),
            "last": (None, PixelFormat.RGB_FP16, 0.0),
        },
    )


def toy_cards(**kwargs):
    return host_bounce_links(x8_cards=("a", "b"), **kwargs)


class BindingStageTest(CustomTestCase):
    """The period is the busiest card, not the serial sum."""

    def test_pipeline_beats_the_serial_sum_when_stages_are_spread(self):
        stages = toy_stages(heavy_on=100.0, light=10.0)
        cards = toy_cards()
        serial = price_placement(
            stages, {"first": "a", "middle": "a", "last": "a"}, cards
        )
        self.assertAlmostEqual(serial.period_ms, 120.0, places=3)
        self.assertEqual(serial.binding_card, "a")
        self.assertEqual(serial.binding_stage, "middle")

        # Move the two cheap stages off and the period drops to the heavy stage
        # alone. The transfers are tiny and fully hidden at depth 1.
        spread = price_placement(
            stages, {"first": "b", "middle": "a", "last": "b"}, cards
        )
        self.assertAlmostEqual(spread.period_ms, 100.0, places=3)
        self.assertEqual(spread.binding_card, "a")
        self.assertGreater(spread.throughput_fps, serial.throughput_fps)

    def test_the_binding_stage_moves_when_the_table_moves(self):
        # A claim about which stage binds is only falsifiable if a different
        # table names a different stage.
        cards = toy_cards()
        placement = {"first": "a", "middle": "a", "last": "b"}
        heavy_middle = price_placement(toy_stages(heavy_on=100.0), placement, cards)
        self.assertEqual(heavy_middle.binding_stage, "middle")
        flipped = stage_table(
            [
                ("first", {"a": 200.0, "b": 200.0}),
                ("middle", {"a": 5.0, "b": 5.0}),
                ("last", {"a": 5.0, "b": 5.0}),
            ],
            source="synthetic",
            geometry={
                "first": (Resolution(64, 64), PixelFormat.RGB_FP16, 1.0),
                "middle": (Resolution(64, 64), PixelFormat.RGB_FP16, 1.0),
                "last": (None, PixelFormat.RGB_FP16, 0.0),
            },
        )
        self.assertEqual(
            price_placement(flipped, placement, cards).binding_stage, "first"
        )

    def test_an_unplaced_or_unknown_stage_is_refused_by_name(self):
        stages = toy_stages()
        cards = toy_cards()
        missing = price_placement(stages, {"first": "a", "middle": "a"}, cards)
        self.assertFalse(missing.feasible)
        self.assertIn("'last' was not placed", missing.reason)
        unknown = price_placement(
            stages, {"first": "a", "middle": "a", "last": "zz"}, cards
        )
        self.assertFalse(unknown.feasible)
        self.assertIn("unknown card 'zz'", unknown.reason)

    def test_an_absent_stage_rate_makes_the_placement_unpriceable(self):
        stages = ticket_v_stages(rife_5090=20.539, rife_3080=63.108)
        cards = rig_cards()
        bad = price_placement(
            stages,
            {
                "decode": "5090",
                "sr": "3080_x8",
                "resize": "3080_x8",
                "rife": "5090",
                "encode": "5090",
            },
            cards,
        )
        self.assertFalse(bad.feasible)
        self.assertIn("never measured", bad.reason)
        self.assertIn("resize", bad.reason)
        # The flip: the same placement with resize on the card that has the
        # measurement prices fine.
        good = price_placement(
            stages,
            {
                "decode": "5090",
                "sr": "5090",
                "resize": "5090",
                "rife": "3080_x8",
                "encode": "3080_x4",
            },
            cards,
        )
        self.assertTrue(good.feasible)


class CoResidencyTest(CustomTestCase):
    def test_splitting_sr_from_resize_is_refused(self):
        stages = ticket_v_stages(rife_5090=20.539, rife_3080=63.108)
        cards = rig_cards()
        split = price_placement(
            stages,
            {
                "decode": "5090",
                "sr": "5090",
                "resize": "3080_x8",
                "rife": "3080_x8",
                "encode": "3080_x4",
            },
            cards,
        )
        self.assertFalse(split.feasible)
        self.assertIn("co-residency violated", split.reason)
        self.assertIn("must not cross a card", split.reason)

    def test_the_enumerator_never_returns_a_split_pair(self):
        stages = ticket_v_stages(rife_5090=11.359, rife_3080=31.999)
        best, _refusals = best_placement(stages, rig_cards())
        self.assertIsNotNone(best)
        self.assertEqual(best.placement["sr"], best.placement["resize"])


class TransportTabooTest(CustomTestCase):
    """The x4 card is disqualified as an 8K-intermediate endpoint."""

    def test_the_taboo_threshold_is_the_8k_fp16_frame(self):
        self.assertAlmostEqual(EIGHT_K_FP16_MIB, 189.84375, places=4)

    def test_an_8k_crossing_onto_the_x4_card_is_refused(self):
        # Co-residency is what normally prevents this, so it is lifted here to
        # exercise the taboo on its own: the two constraints must not be one
        # constraint wearing two names.
        stages = stage_table(
            [
                ("sr", {"5090": 25.424, "3080_x4": 90.343}),
                ("resize", {"5090": 24.367, "3080_x4": 24.367}),
            ],
            source=TICKET_V,
            geometry={
                "sr": (R8K, PixelFormat.RGB_FP16, 1.0),
                "resize": (None, PixelFormat.RGB_FP16, 0.0),
            },
        )
        cards = host_bounce_links(x8_cards=("5090",), x4_cards=("3080_x4",))
        refused = price_placement(stages, {"sr": "5090", "resize": "3080_x4"}, cards)
        self.assertFalse(refused.feasible)
        self.assertIn("transport taboo", refused.reason)
        self.assertIn("189.8 MiB", refused.reason)
        self.assertIn("3080_x4", refused.reason)

        # The flip 1: the same crossing between two x8 cards is allowed.
        two_x8 = host_bounce_links(x8_cards=("5090", "other"))
        allowed = price_placement(
            stage_table(
                [
                    ("sr", {"5090": 25.424, "other": 90.343}),
                    ("resize", {"5090": 24.367, "other": 24.367}),
                ],
                source=TICKET_V,
                geometry={
                    "sr": (R8K, PixelFormat.RGB_FP16, 1.0),
                    "resize": (None, PixelFormat.RGB_FP16, 0.0),
                },
            ),
            {"sr": "5090", "resize": "other"},
            two_x8,
        )
        self.assertTrue(allowed.feasible)

        # The flip 2: a 4K payload onto the same x4 card is under the ceiling
        # and passes, so the refusal is about the payload class and not about
        # the card being blacklisted outright.
        four_k = price_placement(
            stage_table(
                [
                    ("rife", {"5090": 11.359, "3080_x4": 31.999}),
                    ("encode", {"5090": 40.780, "3080_x4": 47.080}),
                ],
                source=TICKET_V,
                geometry={
                    "rife": (R4K, PixelFormat.RGB_FP16, 2.0),
                    "encode": (None, PixelFormat.RGB_FP16, 0.0),
                },
            ),
            {"rife": "5090", "encode": "3080_x4"},
            cards,
        )
        self.assertTrue(four_k.feasible)

    def test_barlink_is_named_absent_rather_than_guessed(self):
        stages = toy_stages()
        cards = (
            CardProfile(key="a", link_gib_s=Rate.measured(13.70, "t", unit="GiB/s")),
            barlink_link("b"),
        )
        refused = price_placement(
            stages, {"first": "a", "middle": "a", "last": "b"}, cards
        )
        self.assertFalse(refused.feasible)
        self.assertIn("barlink BAR1 peer bandwidth", refused.reason)
        self.assertIn("unmeasured", refused.reason)
        # A placement that needs no crossing onto b prices fine, so the
        # absence blocks exactly the thing it describes.
        fine = price_placement(
            stages, {"first": "a", "middle": "a", "last": "a"}, cards
        )
        self.assertTrue(fine.feasible)


class PrefetchHidingTest(CustomTestCase):
    """max(0, transfer - overlap window), per the 2026-08-03 directive."""

    def test_depth_zero_pays_every_byte_and_depth_one_hides_what_fits(self):
        # One crossing, payload chosen so the transfer is LARGER than the
        # window the compute can hide it behind: 40.02 MiB of RGB_FP16 at
        # 13.70 GiB/s is 2.853 ms one way against 1 ms of compute per card.
        payload_res = Resolution(2732, 2560)  # 40.02 MiB at 6 B/px
        stages = stage_table(
            [
                ("first", {"a": 1.0, "b": 1.0}),
                ("last", {"a": 1.0, "b": 1.0}),
            ],
            source="synthetic",
            geometry={
                "first": (payload_res, PixelFormat.RGB_FP16, 1.0),
                "last": (None, PixelFormat.RGB_FP16, 0.0),
            },
        )
        cards = toy_cards()
        placement = {"first": "a", "last": "b"}

        unhidden = price_placement(stages, placement, cards, prefetch_depth=0)
        raw = {(t.card, t.direction): t.raw_ms for t in unhidden.transfers}
        self.assertEqual(len(raw), 2)
        one_way = next(iter(raw.values()))
        self.assertGreater(one_way, 0.0)
        for transfer in unhidden.transfers:
            self.assertAlmostEqual(transfer.hidden_ms, 0.0, places=9)
            self.assertAlmostEqual(transfer.unhidden_ms, transfer.raw_ms, places=9)
        # Each card pays 1 ms of compute plus its own half of the crossing.
        self.assertAlmostEqual(unhidden.card_load_ms["a"], 1.0 + one_way, places=6)

        # Depth 1 gives each card a 1 ms window (its own compute), which covers
        # part of the transfer but not all of it.
        partial = price_placement(stages, placement, cards, prefetch_depth=1)
        for transfer in partial.transfers:
            self.assertAlmostEqual(transfer.hidden_ms, 1.0, places=6)
            self.assertAlmostEqual(
                transfer.unhidden_ms, transfer.raw_ms - 1.0, places=6
            )
        self.assertLess(partial.period_ms, unhidden.period_ms)

    def test_a_transfer_smaller_than_the_window_costs_exactly_nothing(self):
        stages = toy_stages(heavy_on=100.0, light=10.0)
        cards = toy_cards()
        placement = {"first": "a", "middle": "b", "last": "b"}
        hidden = price_placement(stages, placement, cards, prefetch_depth=1)
        for transfer in hidden.transfers:
            self.assertAlmostEqual(transfer.unhidden_ms, 0.0, places=9)
        # ... and the period is then pure compute.
        self.assertAlmostEqual(hidden.card_load_ms["b"], 110.0, places=6)
        self.assertAlmostEqual(hidden.card_load_ms["a"], 10.0, places=6)

    def test_deeper_prefetch_never_costs_more(self):
        payload_res = Resolution(2048, 1024)
        stages = stage_table(
            [("first", {"a": 5.0, "b": 5.0}), ("last", {"a": 5.0, "b": 5.0})],
            source="synthetic",
            geometry={
                "first": (payload_res, PixelFormat.RGB_FP16, 1.0),
                "last": (None, PixelFormat.RGB_FP16, 0.0),
            },
        )
        cards = toy_cards()
        placement = {"first": "a", "last": "b"}
        periods = [
            price_placement(stages, placement, cards, prefetch_depth=depth).period_ms
            for depth in (0, 1, 2, 4)
        ]
        self.assertEqual(periods, sorted(periods, reverse=True))
        self.assertGreater(periods[0], periods[-1])

    def test_a_fully_hidden_estimate_does_not_degrade_the_provenance(self):
        # The x4 link rate is an estimate. If prefetch hides the whole crossing
        # it contributes zero milliseconds, so it cannot move the answer and
        # must not be reported as if it had.
        stages = ticket_v_stages(rife_5090=11.359, rife_3080=31.999)
        cards = rig_cards()
        placement = {
            "decode": "3080_x8",
            "sr": "5090",
            "resize": "5090",
            "rife": "3080_x8",
            "encode": "3080_x4",
        }
        hidden = price_placement(stages, placement, cards, prefetch_depth=1)
        self.assertIs(hidden.provenance, Provenance.MEASURED)
        for transfer in hidden.transfers:
            self.assertAlmostEqual(transfer.unhidden_ms, 0.0, places=9)
        # The flip: with no prefetch the estimated x4 half is actually paid,
        # and the verdict is labelled estimate.
        paid = price_placement(stages, placement, cards, prefetch_depth=0)
        self.assertIs(paid.provenance, Provenance.ESTIMATE)


class LatencyBoundTest(CustomTestCase):
    """Deep buffering: allowed, bounded, and reported."""

    def test_frames_in_flight_buy_latency_and_nothing_else(self):
        stages = toy_stages(heavy_on=100.0, light=10.0)
        cards = toy_cards()
        placement = {"first": "b", "middle": "a", "last": "b"}
        shallow = price_placement(stages, placement, cards, frames_in_flight=0)
        deep = price_placement(stages, placement, cards, frames_in_flight=50)
        self.assertAlmostEqual(shallow.throughput_fps, deep.throughput_fps, places=9)
        self.assertAlmostEqual(shallow.latency_s, 0.0, places=9)
        # 50 frames at 10 fps is 5 s of lead.
        self.assertAlmostEqual(deep.latency_s, 5.0, places=6)

    def test_the_latency_bound_is_enforced_when_given(self):
        stages = toy_stages(heavy_on=100.0, light=10.0)
        cards = toy_cards()
        placement = {"first": "b", "middle": "a", "last": "b"}
        within = price_placement(
            stages, placement, cards, frames_in_flight=50, max_latency_s=6.0
        )
        self.assertTrue(within.feasible)
        beyond = price_placement(
            stages, placement, cards, frames_in_flight=50, max_latency_s=4.0
        )
        self.assertFalse(beyond.feasible)
        self.assertIn("5.00 s of latency, above the 4.00 s bound", beyond.reason)

    def test_no_bound_means_no_refusal(self):
        stages = toy_stages()
        cards = toy_cards()
        loose = price_placement(
            stages,
            {"first": "b", "middle": "a", "last": "b"},
            cards,
            frames_in_flight=10_000,
        )
        self.assertTrue(loose.feasible)
        self.assertGreater(loose.latency_s, 100.0)


class ReplicatedComparisonTest(CustomTestCase):
    def test_absent_stages_give_a_lower_bound_by_default(self):
        stages = ticket_v_stages(rife_5090=20.539, rife_3080=63.108)
        cards = rig_cards()
        fps, provenance, gaps = replicated_throughput(stages, cards)
        # Only the 5090 can be shown to run the whole chain.
        self.assertAlmostEqual(fps, 1000.0 / 115.364, places=3)
        self.assertEqual(set(gaps), {"3080_x8", "3080_x4"})
        self.assertIs(provenance, Provenance.ESTIMATE)

    def test_omitting_the_absent_term_reproduces_the_ticket_v_upper_bound(self):
        stages = ticket_v_stages(rife_5090=20.539, rife_3080=63.108)
        cards = rig_cards()
        fps, _prov, gaps = replicated_throughput(stages, cards, omit_absent_stages=True)
        # RESULTS.md: "3-card aggregate = 8.67 + 2 x 4.82 = 18.30 source-fps".
        self.assertAlmostEqual(fps, 18.299, places=2)
        self.assertEqual(set(gaps), {"3080_x8", "3080_x4"})

    def test_the_two_readings_are_labelled_differently(self):
        stages = ticket_v_stages(rife_5090=20.539, rife_3080=63.108)
        cards = rig_cards()
        _best, lower = compare_regimes(stages, cards)
        _best2, upper = compare_regimes(stages, cards, omit_absent_stages=True)
        self.assertIn("LOWER BOUND", lower.note)
        self.assertIn("UPPER BOUND", upper.note)
        self.assertLess(lower.replicated_fps, upper.replicated_fps)


class TicketVVerdictTest(CustomTestCase):
    """The re-derived 1080p@25 -> 2160p@50 verdict, pinned to the doc.

    These numbers are quoted in ``TASK_333_M2_VIDEO_ENHANCE.md`` §17. If the
    pricer changes, this test fails and the document has to be updated with
    it -- which is the only mechanism that keeps a design doc true.
    """

    def test_scale_one_binds_on_rife_at_4k_on_a_3080(self):
        stages = ticket_v_stages(rife_5090=20.539, rife_3080=63.108)
        best, _refusals = best_placement(
            stages,
            rig_cards(),
            prefetch_depth=1,
            unpriced_stages=UNPRICED_CHAIN_STAGES,
        )
        self.assertIsNotNone(best)
        self.assertAlmostEqual(best.period_ms, 63.108, places=2)
        self.assertAlmostEqual(best.throughput_fps, 15.85, places=2)
        self.assertEqual(best.binding_card, "3080_x8")
        self.assertEqual(best.binding_stage, "rife")

    def test_scale_half_binds_on_the_5090_sr_resize_pair(self):
        stages = ticket_v_stages(rife_5090=11.359, rife_3080=31.999)
        best, _refusals = best_placement(
            stages,
            rig_cards(),
            prefetch_depth=1,
            unpriced_stages=UNPRICED_CHAIN_STAGES,
        )
        self.assertIsNotNone(best)
        # 25.424 (SR) + 24.367 (resize), every transfer hidden.
        self.assertAlmostEqual(best.period_ms, 49.791, places=3)
        self.assertAlmostEqual(best.throughput_fps, 20.08, places=2)
        self.assertEqual(best.binding_card, "5090")
        self.assertEqual(best.binding_stage, "sr")
        self.assertIs(best.provenance, Provenance.MEASURED)
        self.assertEqual(best.unpriced_stages, UNPRICED_CHAIN_STAGES)

    def test_the_verdict_is_not_full_at_25_source_fps(self):
        for rife_5090, rife_3080 in ((20.539, 63.108), (11.359, 31.999)):
            with self.subTest(rife_5090=rife_5090):
                best, _refusals = best_placement(
                    ticket_v_stages(rife_5090=rife_5090, rife_3080=rife_3080),
                    rig_cards(),
                    prefetch_depth=1,
                )
                self.assertLess(best.throughput_fps, 25.0)

    def test_fusing_the_tail_resize_moves_the_bind_to_encode(self):
        # FUSED_TAIL_RESIZE_NOTE in sr.py aims at exactly the resize term. With
        # it at zero the 5090 group halves and the encode stage binds again --
        # which is the follow-on this verdict argues for, stated as arithmetic
        # rather than as an opinion.
        stages = stage_table(
            [
                ("decode", {"5090": 4.254, "3080_x8": 7.140, "3080_x4": 7.140}),
                ("sr", {"5090": 25.424, "3080_x8": 90.343, "3080_x4": 90.343}),
                ("resize", {"5090": 0.0, "3080_x8": None, "3080_x4": None}),
                ("rife", {"5090": 11.359, "3080_x8": 31.999, "3080_x4": 31.999}),
                ("encode", {"5090": 40.780, "3080_x8": 47.080, "3080_x4": 47.080}),
            ],
            source=TICKET_V + " with a hypothetical fused tail resize",
            geometry=GEOMETRY,
            co_residency=CO_RESIDENT,
        )
        best, _refusals = best_placement(stages, rig_cards(), prefetch_depth=1)
        self.assertEqual(best.binding_stage, "encode")
        self.assertAlmostEqual(best.throughput_fps, 21.24, places=2)
        self.assertLess(best.throughput_fps, 25.0)


if __name__ == "__main__":
    unittest.main()
