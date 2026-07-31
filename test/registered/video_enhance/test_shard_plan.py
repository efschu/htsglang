"""Cost model and plan shapes for capacity-weighted chunk sharding.

The whole point of DESIGN #333 §8.2's delta over VSGAN is that the split
ratio is derived rather than hand-picked, so the derivation is what gets
pinned here: arity propagation through RIFE, capacity weighting, the chunk
overlap the pair-reading stage forces, and the makespan ordering between the
proposal and the two baselines under one shared cost model.

    python -m pytest test/registered/video_enhance/test_shard_plan.py -v
"""

import unittest

from sglang.srt.video_enhance.chain import ChainRequest, StageKind, build_chain
from sglang.srt.video_enhance.frame_math import GIB, MIB, R1080P
from sglang.srt.video_enhance.shard_plan import (
    CardAvailability,
    MissingRateError,
    PlanStrategy,
    RateTable,
    ReservationInputs,
    ShardPlanError,
    StageRate,
    capacity_weighted_plan,
    chain_cost,
    check_headroom,
    compare_plans,
    predict_makespan,
    static_single_card_plan,
    vsgan_style_modulo_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci

# Pure arithmetic over a measurement table; no device, no NVML, no network.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FAST = "fast"
SLOW_A = "slow_a"
SLOW_B = "slow_b"

#: A budget that comfortably affords the 1080p chain at the depths used here
#: (the chain needs ~3.2 GiB at depth 1 with RIFE), so headroom never
#: accidentally becomes the thing a weighting test is measuring.
AMPLE_BYTES = 12 * GIB

#: RIFE's per-frame-pair footprint is measurement post P4 and has no
#: predicted value, so every reservation call in this file declares one.
RIFE_PAIR_BYTES = 512 * MIB
RESERVATION = ReservationInputs(rife_measured_bytes_per_pair=RIFE_PAIR_BYTES)


def make_chain(fps_multiplier=1, streams_in_flight=1):
    return build_chain(
        ChainRequest(
            source=R1080P,
            target=R1080P,
            fps_multiplier=fps_multiplier,
            streams_in_flight=streams_in_flight,
        )
    )


def make_rates(card_bases):
    """A flat P1 table: every stage on ``card`` costs ``base`` ms per invocation.

    Flat per card is deliberate. It makes the per-card chain rate an exact
    multiple of the base, so a weighting assertion reads as "3x the card, 3x
    the chunk" instead of being an artefact of a hand-tuned table.
    """
    rows = []
    reference = make_chain(fps_multiplier=2)
    for spec in reference.stages:
        for card, base in card_bases.items():
            rows.append(StageRate(spec.kind, card, spec.in_res, base))
    return RateTable(rows)


def cards(*names, reserved_bytes=AMPLE_BYTES, **kwargs):
    return [
        CardAvailability(card=n, reserved_bytes=reserved_bytes, **kwargs) for n in names
    ]


class TestCostModel(unittest.TestCase):
    def test_rife_arity_propagates_downstream(self):
        """RIFE runs once per source frame; everything after it runs twice.

        A naive per-frame sum over the stage list gets both halves wrong: it
        would either halve RIFE (it reads pairs) or leave the encoder at 1x
        (it does not; RIFE doubled the stream).
        """
        rates = make_rates({FAST: 1.0})
        cost = chain_cost(make_chain(fps_multiplier=2), rates, FAST)
        by_stage = dict(cost.per_stage_ms)

        self.assertEqual(by_stage[StageKind.DECODE], 1.0)
        self.assertEqual(by_stage[StageKind.SR], 1.0)
        self.assertEqual(by_stage[StageKind.RIFE], 1.0)
        self.assertEqual(by_stage[StageKind.COLOR_TO_YUV], 2.0)
        self.assertEqual(by_stage[StageKind.ENCODE], 2.0)
        # decode + colour + SR + resize + RIFE + 2x colour + 2x encode
        self.assertEqual(cost.full_ms, 9.0)
        # An overlap frame only has to reach RIFE's input.
        self.assertEqual(cost.prefix_ms, 4.0)

    def test_no_rife_means_no_prefix_and_no_seam_cost(self):
        rates = make_rates({FAST: 1.0})
        cost = chain_cost(make_chain(fps_multiplier=1), rates, FAST)
        self.assertEqual(cost.full_ms, 6.0)
        self.assertEqual(cost.prefix_ms, 0.0)

    def test_rate_scale_derates_the_whole_chain(self):
        rates = make_rates({FAST: 1.0})
        cost = chain_cost(make_chain(), rates, FAST, rate_scale=1.5)
        self.assertEqual(cost.full_ms, 9.0)

    def test_missing_cell_names_the_stage_card_and_resolution(self):
        rates = make_rates({FAST: 1.0})
        with self.assertRaises(MissingRateError) as ctx:
            chain_cost(make_chain(), rates, "unmeasured")
        message = str(ctx.exception)
        self.assertIn("decode", message)
        self.assertIn("unmeasured", message)
        self.assertIn("1920x1080", message)

    def test_zero_or_negative_rate_is_rejected_at_table_build(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(ShardPlanError) as ctx:
                StageRate(StageKind.SR, FAST, R1080P, bad)
            self.assertIn("must be positive", str(ctx.exception))


class TestCapacityWeighting(unittest.TestCase):
    def test_three_times_the_rate_gets_three_times_the_chunk(self):
        rates = make_rates({FAST: 1.0, SLOW_A: 3.0})
        plan = capacity_weighted_plan(
            chain=make_chain(),
            rates=rates,
            cards=cards(FAST, SLOW_A),
            total_frames=1000,
            reservation=RESERVATION,
        )
        by_card = {a.card: a.owned_frames for a in plan.assignments}
        self.assertEqual(by_card, {FAST: 750, SLOW_A: 250})
        self.assertEqual(by_card[FAST] / by_card[SLOW_A], 3.0)

    def test_chunks_are_contiguous_and_tile_the_timeline(self):
        plan = capacity_weighted_plan(
            chain=make_chain(),
            rates=make_rates({FAST: 1.0, SLOW_A: 2.0, SLOW_B: 5.0}),
            cards=cards(FAST, SLOW_A, SLOW_B),
            total_frames=977,
            reservation=RESERVATION,
        )
        self.assertEqual(plan.owned_frames, 977)
        cursor = 0
        for assignment in plan.assignments:
            self.assertEqual(assignment.stride, 1)
            self.assertEqual(assignment.start, cursor)
            cursor = assignment.stop
        self.assertEqual(cursor, 977)

    def test_single_card_offered_is_a_single_chunk_without_seams(self):
        plan = capacity_weighted_plan(
            chain=make_chain(fps_multiplier=2),
            rates=make_rates({FAST: 1.0}),
            cards=cards(FAST),
            total_frames=500,
            reservation=RESERVATION,
        )
        self.assertEqual(len(plan.assignments), 1)
        self.assertEqual(plan.overlap_frames, 0)
        self.assertEqual(plan.predicted_makespan_ms, 500 * 9.0)

    def test_more_cards_than_frames_drops_the_empty_chunks(self):
        names = ("c0", "c1", "c2", "c3", "c4")
        plan = capacity_weighted_plan(
            chain=make_chain(),
            rates=make_rates(dict.fromkeys(names, 1.0)),
            cards=cards(*names),
            total_frames=2,
            reservation=RESERVATION,
        )
        self.assertEqual(plan.owned_frames, 2)
        self.assertEqual(len(plan.assignments), 2)
        for assignment in plan.assignments:
            self.assertGreater(assignment.owned_frames, 0)

    def test_duplicate_card_is_rejected(self):
        with self.assertRaises(ShardPlanError):
            capacity_weighted_plan(
                chain=make_chain(),
                rates=make_rates({FAST: 1.0}),
                cards=cards(FAST, FAST),
                total_frames=10,
                reservation=RESERVATION,
            )

    def test_zero_frames_is_rejected(self):
        with self.assertRaises(ShardPlanError):
            capacity_weighted_plan(
                chain=make_chain(),
                rates=make_rates({FAST: 1.0}),
                cards=cards(FAST),
                total_frames=0,
                reservation=RESERVATION,
            )


class TestOverlapAccounting(unittest.TestCase):
    def test_interior_seams_carry_one_overlap_frame_on_each_side(self):
        plan = capacity_weighted_plan(
            chain=make_chain(fps_multiplier=2),
            rates=make_rates({FAST: 1.0, SLOW_A: 1.0, SLOW_B: 1.0}),
            cards=cards(FAST, SLOW_A, SLOW_B),
            total_frames=900,
            reservation=RESERVATION,
        )
        lead_tail = [(a.lead_overlap, a.tail_overlap) for a in plan.assignments]
        self.assertEqual(lead_tail, [(0, 1), (1, 1), (1, 0)])
        # Two interior seams, one frame on each side of each.
        self.assertEqual(plan.overlap_frames, 4)

    def test_overlap_is_priced_at_the_prefix_not_the_whole_chain(self):
        chain = make_chain(fps_multiplier=2)
        rates = make_rates({FAST: 1.0, SLOW_A: 1.0})
        plan = capacity_weighted_plan(
            chain=chain,
            rates=rates,
            cards=cards(FAST, SLOW_A),
            total_frames=1000,
            reservation=RESERVATION,
        )
        prediction = predict_makespan(plan, rates)
        # 500 owned frames at 9.0 ms, plus one overlap frame at the 4.0 ms
        # pre-RIFE prefix, on each card.
        for cost in prediction.per_card:
            self.assertEqual(cost.owned_ms, 4500.0)
            self.assertEqual(cost.overlap_ms, 4.0)
        self.assertEqual(prediction.makespan_ms, 4504.0)

    def test_a_chain_without_rife_has_no_seam_cost(self):
        plan = capacity_weighted_plan(
            chain=make_chain(fps_multiplier=1),
            rates=make_rates({FAST: 1.0, SLOW_A: 1.0}),
            cards=cards(FAST, SLOW_A),
            total_frames=1000,
            reservation=RESERVATION,
        )
        self.assertEqual(plan.overlap_frames, 0)


class TestBaselinesAndMakespan(unittest.TestCase):
    """The before/after comparison, all three shapes on one cost model."""

    def test_heterogeneous_ordering(self):
        """capacity-weighted <= modulo <= single-card on this table.

        This is a property of the table, not a theorem. With three cards and
        the slower two at 2x, equal shares still beat one card outright; push
        the skew far enough and the modulo baseline loses to single-card,
        which ``test_modulo_can_lose_to_single_card`` pins separately.
        """
        chain = make_chain(fps_multiplier=1)
        rates = make_rates({FAST: 1.0, SLOW_A: 2.0, SLOW_B: 2.0})
        offered = cards(FAST, SLOW_A, SLOW_B)
        scored = compare_plans(
            chain=chain,
            rates=rates,
            cards=offered,
            total_frames=1000,
            reservation=RESERVATION,
        )

        weighted = scored[PlanStrategy.CAPACITY_WEIGHTED].makespan_ms
        modulo = scored[PlanStrategy.VSGAN_MODULO].makespan_ms
        single = scored[PlanStrategy.STATIC_SINGLE_CARD].makespan_ms

        self.assertEqual(weighted, 3000.0)  # 500/250/250 frames, all finish together
        self.assertEqual(modulo, 3996.0)  # 333 frames on a 12 ms card
        self.assertEqual(single, 6000.0)  # 1000 frames on the 6 ms card
        self.assertLessEqual(weighted, modulo)
        self.assertLessEqual(modulo, single)

    def test_homogeneous_table_shows_no_win(self):
        """Identical cards: capacity weighting must not claim an advantage."""
        chain = make_chain(fps_multiplier=1)
        rates = make_rates({FAST: 1.0, SLOW_A: 1.0, SLOW_B: 1.0})
        scored = compare_plans(
            chain=chain,
            rates=rates,
            cards=cards(FAST, SLOW_A, SLOW_B),
            total_frames=1000,
            reservation=RESERVATION,
        )
        self.assertEqual(
            scored[PlanStrategy.CAPACITY_WEIGHTED].makespan_ms,
            scored[PlanStrategy.VSGAN_MODULO].makespan_ms,
        )

    def test_capacity_weighting_leaves_no_card_idle(self):
        rates = make_rates({FAST: 1.0, SLOW_A: 2.0, SLOW_B: 2.0})
        plan = capacity_weighted_plan(
            chain=make_chain(fps_multiplier=1),
            rates=rates,
            cards=cards(FAST, SLOW_A, SLOW_B),
            total_frames=1000,
            reservation=RESERVATION,
        )
        idle = predict_makespan(plan, rates).idle_ms
        self.assertEqual(set(idle.values()), {0.0})

    def test_modulo_pays_the_pair_tax_once_per_owned_frame(self):
        """With RIFE, an interleave puts every pair across a card boundary.

        Each card must re-run the pre-RIFE prefix on the successor of every
        frame it owns, so the overlap count equals the frame count. This is
        the concrete reason the planner emits chunks rather than copying
        ``SelectEvery(cycle=N)``.
        """
        chain = make_chain(fps_multiplier=2)
        rates = make_rates({FAST: 1.0, SLOW_A: 2.0})
        plan = vsgan_style_modulo_plan(
            chain=chain,
            rates=rates,
            cards=cards(FAST, SLOW_A),
            total_frames=1000,
            reservation=RESERVATION,
        )
        self.assertEqual([a.stride for a in plan.assignments], [2, 2])
        self.assertEqual(plan.owned_frames, 1000)
        self.assertEqual(plan.overlap_frames, 1000)

        prediction = predict_makespan(plan, rates)
        # fast: 500 * 9 + 500 * 4; slow: 500 * 18 + 500 * 8.
        self.assertEqual(prediction.makespan_ms, 13000.0)
        self.assertEqual(prediction.busiest_card, SLOW_A)

    def test_modulo_can_lose_to_single_card(self):
        """An honest baseline has to be allowed to lose.

        Under RIFE at a 2x card-speed skew the interleave's pair tax plus its
        equal shares put it behind simply using the fast card alone. Suppress
        this and the comparison would flatter the proposal by comparing it
        against a straw baseline.
        """
        chain = make_chain(fps_multiplier=2)
        scored = compare_plans(
            chain=chain,
            rates=make_rates({FAST: 1.0, SLOW_A: 2.0}),
            cards=cards(FAST, SLOW_A),
            total_frames=1000,
            reservation=RESERVATION,
        )
        self.assertEqual(scored[PlanStrategy.CAPACITY_WEIGHTED].makespan_ms, 6007.0)
        self.assertEqual(scored[PlanStrategy.STATIC_SINGLE_CARD].makespan_ms, 9000.0)
        self.assertEqual(scored[PlanStrategy.VSGAN_MODULO].makespan_ms, 13000.0)
        self.assertLess(
            scored[PlanStrategy.STATIC_SINGLE_CARD].makespan_ms,
            scored[PlanStrategy.VSGAN_MODULO].makespan_ms,
        )

    def test_single_card_baseline_picks_the_fastest_offered_card(self):
        plan = static_single_card_plan(
            chain=make_chain(),
            rates=make_rates({FAST: 1.0, SLOW_A: 2.0}),
            cards=cards(FAST, SLOW_A),
            total_frames=100,
            reservation=RESERVATION,
        )
        self.assertEqual(plan.cards, (FAST,))
        self.assertEqual(plan.overlap_frames, 0)

    def test_single_card_baseline_honours_an_explicit_card(self):
        plan = static_single_card_plan(
            chain=make_chain(),
            rates=make_rates({FAST: 1.0, SLOW_A: 2.0}),
            cards=cards(FAST, SLOW_A),
            total_frames=100,
            card=SLOW_A,
            reservation=RESERVATION,
        )
        self.assertEqual(plan.cards, (SLOW_A,))
        self.assertEqual(plan.predicted_makespan_ms, 1200.0)

    def test_single_card_baseline_rejects_an_unoffered_card(self):
        with self.assertRaises(ShardPlanError):
            static_single_card_plan(
                chain=make_chain(),
                rates=make_rates({FAST: 1.0}),
                cards=cards(FAST),
                total_frames=100,
                card="somewhere_else",
                reservation=RESERVATION,
            )

    def test_cotenant_derate_shifts_the_split(self):
        rates = make_rates({FAST: 1.0, SLOW_A: 1.0})
        plan = capacity_weighted_plan(
            chain=make_chain(),
            rates=rates,
            cards=[
                CardAvailability(FAST, AMPLE_BYTES),
                CardAvailability(
                    SLOW_A, AMPLE_BYTES, has_llm_cotenant=True, rate_scale=3.0
                ),
            ],
            total_frames=1000,
            reservation=RESERVATION,
        )
        by_card = {a.card: a.owned_frames for a in plan.assignments}
        self.assertEqual(by_card, {FAST: 750, SLOW_A: 250})


class TestHeadroom(unittest.TestCase):
    def test_a_card_that_cannot_hold_the_chain_is_refused_at_plan_time(self):
        with self.assertRaises(ShardPlanError) as ctx:
            capacity_weighted_plan(
                chain=make_chain(fps_multiplier=2),
                rates=make_rates({FAST: 1.0, SLOW_A: 1.0}),
                cards=[
                    CardAvailability(FAST, AMPLE_BYTES),
                    CardAvailability(SLOW_A, 1 * GIB),
                ],
                total_frames=100,
                reservation=RESERVATION,
            )
        message = str(ctx.exception)
        self.assertIn(SLOW_A, message)
        self.assertIn("reservation headroom", message)
        self.assertIn("1024 MiB", message)

    def test_headroom_matches_the_reservation_formula(self):
        chain = make_chain(fps_multiplier=2, streams_in_flight=2)
        # ~5.25 GiB is needed at depth 2 with a 512 MiB RIFE pair footprint.
        with self.assertRaises(ShardPlanError):
            check_headroom(chain, CardAvailability(FAST, 5 * GIB), RESERVATION)
        self.assertGreaterEqual(
            check_headroom(chain, CardAvailability(FAST, 6 * GIB), RESERVATION), 2
        )

    def test_missing_rife_probe_is_an_error_not_a_guess(self):
        with self.assertRaises(Exception) as ctx:
            check_headroom(
                make_chain(fps_multiplier=2),
                CardAvailability(FAST, AMPLE_BYTES),
                ReservationInputs(),
            )
        self.assertIn("P4", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
