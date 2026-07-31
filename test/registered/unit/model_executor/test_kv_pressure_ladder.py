"""#286 Erg. 9 / 9b -- KV pressure ladder, CPU skeleton.

Hermetic unit tests (no GPU, no CUDA, no torch): step-table validation,
deterministic trend projection, asymmetric hysteresis, the two water marks
of the 9b pre-staging, the enforced invariants (relief before geometry,
capture guard, priority protection, no flip/pre-stage onto uncaptured
graphs, long hysteresis for out-of-family rungs), the handover interface
(one implemented, four measured-by-decision stubs), flag parsing, both
features off = zero behavior, ladder on / staging off as its own case, and
conflict freedom with the Erg.-8 admission planner.
"""

import unittest
from typing import List

from sglang.srt.model_executor.kv_pressure_ladder import (
    DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS,
    HANDOVER_ANTICIPATORY_SHADOW,
    HANDOVER_BACKGROUND_MIGRATE,
    HANDOVER_NONE,
    HANDOVER_SPILL_RELOAD,
    HANDOVER_STRATEGIES,
    LADDER_REGISTER_CLASSES,
    PHASE_ABORT_STAGE,
    PHASE_DESCEND,
    PHASE_FLIP,
    PHASE_PRE_STAGE,
    PROVENANCE_PLACEHOLDER,
    RELIEF_FEATURES,
    STAGE_ABORT,
    STAGE_START,
    STEP_BASE,
    STEP_EXTERNAL,
    STEP_GEOMETRY,
    STEP_RELIEF,
    VERDICT_ASCEND,
    VERDICT_DESCEND,
    VERDICT_HOLD,
    AnticipatoryShadowHandover,
    BackgroundMigrateHandover,
    KvLadderError,
    KvPressureLadder,
    KvPressureSensor,
    LadderStep,
    NewTokensOnlyHandover,
    NoHandover,
    OccupancySample,
    PressureLadder,
    SpillReloadHandover,
    build_ladder_from_server_args,
    get_handover,
    graph_rung_item_id,
    handover_supports_pre_stage,
    kv_shadow_item_id,
    ladder_from_spec,
    parse_kv_pressure_ladder,
    plan_conflicts,
    resolve_plan_priority,
)
from sglang.srt.model_executor.offload_register import (
    OFFLOAD_CLASSES,
    CpuFakeMovementBackend,
    OffloadRegister,
    resolve_class_policies,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def base_step(name: str = "tp1", **kwargs) -> LadderStep:
    return LadderStep(name=name, step_type=STEP_BASE, **kwargs)


def relief_step(feature: str = "dcp_ratio", **kwargs) -> LadderStep:
    return LadderStep(
        name=kwargs.pop("name", feature),
        step_type=STEP_RELIEF,
        relief_feature=feature,
        **kwargs,
    )


def geometry_step(name: str = "tp2", **kwargs) -> LadderStep:
    kwargs.setdefault("handover", HANDOVER_BACKGROUND_MIGRATE)
    return LadderStep(name=name, step_type=STEP_GEOMETRY, geometry_key=name, **kwargs)


def external_step(name: str = "node2", **kwargs) -> LadderStep:
    kwargs.setdefault("handover", HANDOVER_SPILL_RELOAD)
    kwargs.setdefault("min_hysteresis_rounds", DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS)
    return LadderStep(name=name, step_type=STEP_EXTERNAL, **kwargs)


def full_table() -> PressureLadder:
    return PressureLadder(
        [
            base_step(),
            relief_step("dcp_ratio"),
            relief_step("kv_spill"),
            geometry_step("tp2"),
            geometry_step("tp3_dcp"),
            external_step("node2"),
        ]
    )


def series(
    values: List[float],
    total: int = 1000,
    start_round: int = 0,
) -> List[OccupancySample]:
    """Occupancy fractions -> injected samples (the CPU-phase fake of the
    scheduler / token_to_kv_pool occupancy accounting)."""
    return [
        OccupancySample(
            round_index=start_round + i,
            used_tokens=int(round(v * total)),
            total_tokens=total,
        )
        for i, v in enumerate(values)
    ]


def fast_sensor(**kwargs) -> KvPressureSensor:
    """Short windows so a test does not have to feed 64 samples."""
    kwargs.setdefault("ascend_window", 2)
    kwargs.setdefault("descend_window", 4)
    kwargs.setdefault("pre_stage_window", 2)
    kwargs.setdefault("abort_stage_window", 3)
    kwargs.setdefault("horizon_rounds", 4)
    return KvPressureSensor(**kwargs)


class _Args:
    """Minimal server-args stand-in for the flag plumbing."""

    def __init__(self, **kwargs):
        self.kv_pressure_ladder = None
        self.kv_pressure_pre_stage = False
        self.kv_pressure_ascend_threshold = 0.85
        self.kv_pressure_ascend_window = 2
        self.kv_pressure_descend_threshold = 0.55
        self.kv_pressure_descend_window = 4
        self.kv_pressure_pre_stage_threshold = 0.70
        self.kv_pressure_pre_stage_window = 2
        self.kv_pressure_abort_stage_window = 3
        self.kv_pressure_horizon_rounds = 4
        self.kv_pressure_external_hysteresis_rounds = DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# 1. Step table
# ---------------------------------------------------------------------------


class TestLadderStep(unittest.TestCase):
    def test_unknown_type_and_handover_rejected(self):
        with self.assertRaises(ValueError):
            LadderStep(name="x", step_type="wishful")
        with self.assertRaises(ValueError):
            LadderStep(name="x", step_type=STEP_BASE, handover="telepathy")

    def test_relief_must_reference_a_known_feature(self):
        for feature in RELIEF_FEATURES:
            self.assertEqual(relief_step(feature).relief_feature, feature)
        with self.assertRaises(ValueError) as ctx:
            LadderStep(name="magic", step_type=STEP_RELIEF, relief_feature="magic")
        self.assertIn("only ORDERS existing features", str(ctx.exception))

    def test_relief_may_not_declare_a_handover(self):
        with self.assertRaises(ValueError) as ctx:
            relief_step("kv_spill", handover=HANDOVER_BACKGROUND_MIGRATE)
        self.assertIn("does NOT change the KV layout", str(ctx.exception))

    def test_geometry_and_external_need_a_real_handover(self):
        with self.assertRaises(ValueError) as ctx:
            LadderStep(name="tp2", step_type=STEP_GEOMETRY, handover=HANDOVER_NONE)
        self.assertIn("needs one of", str(ctx.exception))
        with self.assertRaises(ValueError):
            LadderStep(
                name="node2",
                step_type=STEP_EXTERNAL,
                handover=HANDOVER_NONE,
                min_hysteresis_rounds=DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS,
            )

    def test_non_relief_may_not_carry_a_relief_feature(self):
        with self.assertRaises(ValueError):
            LadderStep(
                name="tp2",
                step_type=STEP_GEOMETRY,
                relief_feature="kv_spill",
                handover=HANDOVER_BACKGROUND_MIGRATE,
            )

    def test_base_step_constraints(self):
        with self.assertRaises(ValueError):
            base_step(graphs_precaptured=False)
        with self.assertRaises(ValueError):
            base_step(handover=HANDOVER_SPILL_RELOAD)

    def test_derived_register_item_ids(self):
        step = geometry_step("tp2")
        self.assertEqual(step.graph_items, (graph_rung_item_id("tp2"),))
        self.assertEqual(step.shadow_item_ids, (kv_shadow_item_id("tp2"),))
        pinned = geometry_step("tp3", graph_rung_items=("lane0/graph_rung/k3",))
        self.assertEqual(pinned.graph_items, ("lane0/graph_rung/k3",))

    def test_negative_figures_rejected(self):
        with self.assertRaises(ValueError):
            base_step(expected_kv_tokens=-1)
        with self.assertRaises(ValueError):
            base_step(expected_cost_factor=0.0)
        with self.assertRaises(ValueError):
            base_step(provenance="vibes")


class TestPressureLadderTable(unittest.TestCase):
    def test_empty_table_rejected(self):
        with self.assertRaises(ValueError):
            PressureLadder([])

    def test_rung_zero_must_be_base(self):
        with self.assertRaises(ValueError) as ctx:
            PressureLadder([relief_step("kv_spill")])
        self.assertIn("must be the 'base' step", str(ctx.exception))

    def test_second_base_rejected(self):
        with self.assertRaises(ValueError):
            PressureLadder([base_step("a"), base_step("b")])

    def test_relief_after_geometry_is_a_hard_error(self):
        with self.assertRaises(ValueError) as ctx:
            PressureLadder([base_step(), geometry_step("tp2"), relief_step("kv_spill")])
        msg = str(ctx.exception)
        self.assertIn("climb order", msg)
        self.assertIn("enforced, not a convention", msg)

    def test_geometry_after_external_is_a_hard_error(self):
        with self.assertRaises(ValueError):
            PressureLadder([base_step(), external_step("node2"), geometry_step("tp2")])

    def test_duplicate_names_rejected(self):
        with self.assertRaises(ValueError):
            PressureLadder([base_step("tp1"), geometry_step("tp1")])

    def test_capacity_must_not_shrink_upwards(self):
        with self.assertRaises(ValueError) as ctx:
            PressureLadder(
                [
                    base_step(expected_kv_tokens=100_000),
                    geometry_step("tp2", expected_kv_tokens=90_000),
                ]
            )
        self.assertIn("may not lose capacity", str(ctx.exception))

    def test_cost_must_not_shrink_upwards(self):
        with self.assertRaises(ValueError) as ctx:
            PressureLadder(
                [
                    base_step(expected_cost_factor=1.0),
                    geometry_step("tp2", expected_cost_factor=0.9),
                ]
            )
        self.assertIn("would be the base", str(ctx.exception))

    def test_unknown_figures_skip_the_monotonicity_check(self):
        table = PressureLadder(
            [
                base_step(expected_kv_tokens=100_000),
                relief_step("kv_spill"),  # placeholder, no figure
                geometry_step("tp2", expected_kv_tokens=140_000),
            ]
        )
        self.assertEqual(len(table), 3)
        self.assertIsNone(table[1].expected_kv_tokens)

    def test_external_needs_the_long_hysteresis(self):
        with self.assertRaises(ValueError) as ctx:
            PressureLadder(
                [base_step(), external_step("node2", min_hysteresis_rounds=3)]
            )
        self.assertIn("LAST rung", str(ctx.exception))

    def test_describe_and_lookup(self):
        table = full_table()
        self.assertEqual(table.index_of("tp2"), 3)
        self.assertEqual(table.first_index_of_type(STEP_RELIEF), 1)
        self.assertEqual(table.first_index_of_type(STEP_EXTERNAL), 5)
        rows = table.describe()
        self.assertEqual([r["rung"] for r in rows], [0, 1, 2, 3, 4, 5])
        self.assertEqual(rows[1]["relief_feature"], "dcp_ratio")
        with self.assertRaises(KeyError):
            table.index_of("nope")


# ---------------------------------------------------------------------------
# 2. Handover interface
# ---------------------------------------------------------------------------


class TestHandoverInterface(unittest.TestCase):
    def test_none_is_the_only_implemented_strategy(self):
        handover = get_handover(HANDOVER_NONE)
        self.assertIsInstance(handover, NoHandover)
        self.assertIsNone(handover.prepare(None))
        self.assertIsNone(handover.execute(None))
        self.assertIsNone(handover.abort(None))

    def test_the_three_design_options_are_measured_by_decision_stubs(self):
        for cls in (
            NewTokensOnlyHandover,
            BackgroundMigrateHandover,
            SpillReloadHandover,
            AnticipatoryShadowHandover,
        ):
            handover = cls()
            for method in ("prepare", "execute", "abort"):
                with self.assertRaises(NotImplementedError):
                    getattr(handover, method)(None)
            # The measurement questions live in the docstring; that is the
            # deliverable of this phase, so assert it is there.
            self.assertTrue(cls.__doc__)
            self.assertIn("easure", cls.__doc__)

    def test_only_the_shadow_variant_supports_pre_staging(self):
        self.assertTrue(handover_supports_pre_stage(HANDOVER_ANTICIPATORY_SHADOW))
        for strategy in HANDOVER_STRATEGIES:
            if strategy != HANDOVER_ANTICIPATORY_SHADOW:
                self.assertFalse(handover_supports_pre_stage(strategy))

    def test_unknown_strategy_is_a_hard_error(self):
        with self.assertRaises(ValueError):
            get_handover("telepathy")
        with self.assertRaises(ValueError):
            handover_supports_pre_stage("telepathy")


# ---------------------------------------------------------------------------
# 3. Sensor
# ---------------------------------------------------------------------------


class TestSensorConfiguration(unittest.TestCase):
    def test_mark_ordering_enforced(self):
        with self.assertRaises(ValueError) as ctx:
            KvPressureSensor(
                ascend_threshold=0.6,
                descend_threshold=0.5,
                pre_stage_threshold=0.7,
            )
        self.assertIn("descend < pre_stage < ascend", str(ctx.exception))

    def test_descend_window_must_be_longer_than_ascend(self):
        with self.assertRaises(ValueError) as ctx:
            KvPressureSensor(ascend_window=8, descend_window=8)
        self.assertIn("asymmetry is the contract", str(ctx.exception))

    def test_abort_window_must_be_longer_than_pre_stage(self):
        with self.assertRaises(ValueError) as ctx:
            KvPressureSensor(pre_stage_window=4, abort_stage_window=4)
        self.assertIn("stage and discard forever", str(ctx.exception))

    def test_out_of_range_thresholds(self):
        with self.assertRaises(ValueError):
            KvPressureSensor(ascend_threshold=1.5)
        with self.assertRaises(ValueError):
            KvPressureSensor(descend_threshold=0.0)
        with self.assertRaises(ValueError):
            KvPressureSensor(horizon_rounds=0)

    def test_pre_stage_horizon_looks_further_than_the_flip_horizon(self):
        sensor = KvPressureSensor(horizon_rounds=10)
        self.assertEqual(sensor.pre_stage_horizon_rounds, 20)
        with self.assertRaises(ValueError):
            KvPressureSensor(horizon_rounds=10, pre_stage_horizon_rounds=5)

    def test_sample_validation(self):
        with self.assertRaises(ValueError):
            OccupancySample(round_index=0, used_tokens=1, total_tokens=0)
        with self.assertRaises(ValueError):
            OccupancySample(round_index=0, used_tokens=-1, total_tokens=10)


class TestSensorTrend(unittest.TestCase):
    def test_empty_sensor_holds(self):
        reading = fast_sensor().reading()
        self.assertEqual(reading.verdict, VERDICT_HOLD)
        self.assertEqual(reading.samples, 0)
        self.assertIsNone(reading.occupancy)

    def test_trend_projection_is_deterministic(self):
        sensor = KvPressureSensor(ascend_window=4, descend_window=8)
        # Exactly +50 tokens/round, 600/1000 used -> 400 headroom -> 8 rounds.
        sensor.observe_series(
            [
                OccupancySample(
                    round_index=i, used_tokens=450 + 50 * i, total_tokens=1000
                )
                for i in range(4)
            ]
        )
        reading = sensor.reading()
        self.assertAlmostEqual(reading.trend_tokens_per_round, 50.0)
        self.assertAlmostEqual(reading.rounds_to_exhaustion, 8.0)
        # Reading twice gives the identical answer (no side effects).
        again = sensor.reading()
        self.assertEqual(reading, again)

    def test_flat_series_has_no_projected_exhaustion(self):
        sensor = fast_sensor()
        sensor.observe_series(series([0.4, 0.4, 0.4]))
        reading = sensor.reading()
        self.assertEqual(reading.trend_tokens_per_round, 0.0)
        self.assertIsNone(reading.rounds_to_exhaustion)

    def test_trend_alone_triggers_the_ascent_below_the_mark(self):
        # Level stays well below 0.85, but the slope exhausts the pool inside
        # the horizon: the ladder climbs on the PROJECTION, not the level.
        sensor = KvPressureSensor(ascend_window=3, descend_window=6, horizon_rounds=4)
        sensor.observe_series(
            [
                OccupancySample(
                    round_index=i, used_tokens=300 + 150 * i, total_tokens=1000
                )
                for i in range(3)
            ]
        )
        reading = sensor.reading()
        self.assertLess(reading.occupancy, 0.85)
        self.assertEqual(reading.verdict, VERDICT_ASCEND)

    def test_history_is_bounded(self):
        sensor = fast_sensor()
        sensor.observe_series(series([0.5] * 50))
        self.assertLessEqual(len(sensor.history), 4)


class TestAsymmetricHysteresis(unittest.TestCase):
    def test_ascent_after_a_short_window_descent_only_after_a_long_one(self):
        sensor = KvPressureSensor(
            ascend_threshold=0.85,
            ascend_window=2,
            descend_threshold=0.55,
            descend_window=6,
            pre_stage_window=2,
            abort_stage_window=3,
            horizon_rounds=1,
        )
        # One high sample is not enough...
        sensor.observe_series(series([0.9]))
        self.assertEqual(sensor.reading().verdict, VERDICT_HOLD)
        # ...two are.
        sensor.observe_series(series([0.9], start_round=1))
        self.assertEqual(sensor.reading().verdict, VERDICT_ASCEND)

        # Now drop low. Five low rounds are still not a descent (window 6).
        sensor.reset()
        sensor.observe_series(series([0.3] * 5))
        self.assertEqual(sensor.reading().verdict, VERDICT_HOLD)
        sensor.observe_series(series([0.3], start_round=5))
        self.assertEqual(sensor.reading().verdict, VERDICT_DESCEND)

    def test_rising_series_below_the_descend_mark_does_not_descend(self):
        sensor = KvPressureSensor(ascend_window=2, descend_window=4, horizon_rounds=1)
        sensor.observe_series(
            [
                OccupancySample(
                    round_index=i, used_tokens=100 + 10 * i, total_tokens=1000
                )
                for i in range(4)
            ]
        )
        self.assertEqual(sensor.reading().verdict, VERDICT_HOLD)


class TestPreStageMarks(unittest.TestCase):
    def test_pre_stage_fires_below_the_flip_mark(self):
        sensor = fast_sensor(horizon_rounds=1)
        sensor.observe_series(series([0.72, 0.73]))
        reading = sensor.reading()
        self.assertEqual(reading.verdict, VERDICT_HOLD)
        self.assertEqual(reading.stage_verdict, STAGE_START)

    def test_abort_is_sluggish(self):
        sensor = fast_sensor(horizon_rounds=1)
        sensor.observe_series(series([0.72, 0.73]))
        self.assertEqual(sensor.reading().stage_verdict, STAGE_START)
        # Two low rounds are not enough (abort window is 3).
        sensor.observe_series(series([0.40, 0.39], start_round=2))
        self.assertNotEqual(sensor.reading().stage_verdict, STAGE_ABORT)
        sensor.observe_series(series([0.38], start_round=4))
        self.assertEqual(sensor.reading().stage_verdict, STAGE_ABORT)


# ---------------------------------------------------------------------------
# 4. Flip contract
# ---------------------------------------------------------------------------


class TestFlipContract(unittest.TestCase):
    def _ladder(self, **kwargs) -> KvPressureLadder:
        return KvPressureLadder(
            kwargs.pop("table", full_table()),
            kwargs.pop("sensor", fast_sensor(horizon_rounds=1)),
            **kwargs,
        )

    def test_quiet_series_plans_nothing(self):
        ladder = self._ladder()
        plan = ladder.on_pressure_boundary(series([0.6, 0.6]))
        self.assertTrue(plan.is_noop)
        self.assertEqual(plan.current_rung, 0)
        self.assertEqual(plan.target_rung, 0)
        self.assertEqual(ladder.current_rung, 0)

    def test_ascent_goes_one_rung_and_hits_relief_first(self):
        ladder = self._ladder()
        plan = ladder.on_pressure_boundary(series([0.95, 0.96]))
        self.assertEqual(plan.phase, PHASE_FLIP)
        self.assertEqual(plan.target_rung, 1)
        self.assertEqual(ladder.table[plan.target_rung].step_type, STEP_RELIEF)
        # A relief rung changes no layout, so its handover is 'none'.
        self.assertEqual(plan.handover, HANDOVER_NONE)
        ladder.apply(plan)
        self.assertEqual(ladder.current_rung, 1)

    def test_full_climb_exhausts_relief_before_geometry(self):
        ladder = self._ladder()
        seen = []
        for round_index in range(0, 12, 2):
            plan = ladder.on_pressure_boundary(
                series([0.95, 0.96], start_round=round_index)
            )
            if plan.phase == PHASE_FLIP:
                seen.append(ladder.table[plan.target_rung].step_type)
                ladder.apply(plan)
        # The first two flips are relief, only then geometry -- and the
        # out-of-family rung is never reached (long hysteresis).
        self.assertEqual(seen[:2], [STEP_RELIEF, STEP_RELIEF])
        self.assertEqual(seen[2], STEP_GEOMETRY)
        self.assertNotIn(STEP_EXTERNAL, seen)

    def test_forced_target_skipping_relief_is_a_hard_error(self):
        ladder = self._ladder()
        with self.assertRaises(KvLadderError) as ctx:
            ladder.on_pressure_boundary(series([0.95, 0.96]), force_target=3)
        msg = str(ctx.exception)
        self.assertIn("skip the cheaper relief rung", msg)
        self.assertIn("dcp_ratio", msg)

    def test_flip_onto_uncaptured_graphs_is_a_hard_error(self):
        table = PressureLadder(
            [base_step(), geometry_step("tp2", graphs_precaptured=False)]
        )
        ladder = self._ladder(table=table)
        with self.assertRaises(KvLadderError) as ctx:
            ladder.on_pressure_boundary(series([0.95, 0.96]))
        msg = str(ctx.exception)
        self.assertIn("no silent fallback", msg)

    def test_pre_stage_onto_uncaptured_graphs_is_the_same_hard_error(self):
        table = PressureLadder(
            [base_step(), geometry_step("tp2", graphs_precaptured=False)]
        )
        ladder = self._ladder(table=table, pre_stage_enabled=True)
        with self.assertRaises(KvLadderError):
            # Pre-stage mark, not flip mark.
            ladder.on_pressure_boundary(series([0.72, 0.73]))

    def test_capture_guard_blocks_the_flip(self):
        ladder = self._ladder()
        ladder.begin_capture()
        plan = ladder.on_pressure_boundary(series([0.95, 0.96]))
        self.assertTrue(plan.is_noop)
        self.assertIn("capture active", plan.blocked)
        self.assertEqual(ladder.current_rung, 0)
        ladder.end_capture()
        plan = ladder.on_pressure_boundary()
        self.assertEqual(plan.phase, PHASE_FLIP)

    def test_flip_only_at_round_boundaries(self):
        ladder = self._ladder()
        plan = ladder.on_pressure_boundary(
            series([0.95, 0.96]), at_round_boundary=False
        )
        self.assertTrue(plan.is_noop)
        self.assertIn("round boundary", plan.blocked)

    def test_protected_sessions_stay_on_the_fast_rung(self):
        ladder = self._ladder(protected_sessions=["vip"])
        plan = ladder.on_pressure_boundary(
            series([0.95, 0.96]), sessions=["vip", "bulk-a", "bulk-b"]
        )
        self.assertEqual(plan.phase, PHASE_FLIP)
        self.assertEqual(plan.affected_sessions, ["bulk-a", "bulk-b"])
        self.assertEqual(plan.protected_sessions, ["vip"])

    def test_all_protected_blocks_the_ascent(self):
        ladder = self._ladder(protected_sessions=["vip", "vip2"])
        plan = ladder.on_pressure_boundary(
            series([0.95, 0.96]), sessions=["vip", "vip2"]
        )
        self.assertTrue(plan.is_noop)
        self.assertIn("priority-protected", plan.blocked)
        self.assertEqual(ladder.current_rung, 0)

    def test_external_rung_needs_the_long_hysteresis(self):
        table = PressureLadder([base_step(), external_step("node2")])
        ladder = self._ladder(table=table)
        for i in range(0, 20, 2):
            plan = ladder.on_pressure_boundary(series([0.95, 0.96], start_round=i))
            self.assertTrue(plan.is_noop)
            self.assertIn("consecutive ascend verdicts", plan.blocked)
        self.assertEqual(ladder.current_rung, 0)

    def test_external_hysteresis_streak_resets_on_a_calm_round(self):
        table = PressureLadder(
            [base_step(), external_step("node2", min_hysteresis_rounds=3)],
            external_min_hysteresis_rounds=3,
        )
        ladder = self._ladder(table=table)
        ladder.on_pressure_boundary(series([0.95, 0.96]))
        ladder.on_pressure_boundary(series([0.95, 0.96], start_round=2))
        ladder.sensor.reset()
        ladder.on_pressure_boundary(series([0.10, 0.10], start_round=4))
        ladder.sensor.reset()
        plan = ladder.on_pressure_boundary(series([0.95, 0.96], start_round=6))
        self.assertTrue(plan.is_noop)

        # Now three in a row -> the flip is admitted.
        ladder.on_pressure_boundary(series([0.95, 0.96], start_round=8))
        plan = ladder.on_pressure_boundary(series([0.95, 0.96], start_round=10))
        self.assertEqual(plan.phase, PHASE_FLIP)
        self.assertEqual(plan.handover, HANDOVER_SPILL_RELOAD)

    def test_top_rung_blocks_further_ascent(self):
        table = PressureLadder([base_step()])
        ladder = self._ladder(table=table)
        plan = ladder.on_pressure_boundary(series([0.95, 0.96]))
        self.assertTrue(plan.is_noop)
        self.assertIn("top rung", plan.blocked)

    def test_descent_goes_back_one_rung(self):
        ladder = self._ladder(sensor=fast_sensor(horizon_rounds=1))
        ladder.apply(ladder.on_pressure_boundary(series([0.95, 0.96])))
        self.assertEqual(ladder.current_rung, 1)
        ladder.sensor.reset()
        plan = ladder.on_pressure_boundary(series([0.2] * 4, start_round=2))
        self.assertEqual(plan.phase, PHASE_DESCEND)
        self.assertEqual(plan.target_rung, 0)
        ladder.apply(plan)
        self.assertEqual(ladder.current_rung, 0)
        # And on the base rung a descent verdict plans nothing.
        ladder.sensor.reset()
        plan = ladder.on_pressure_boundary(series([0.2] * 4, start_round=6))
        self.assertTrue(plan.is_noop)

    def test_required_resident_items_name_the_graph_rungs(self):
        ladder = self._ladder()
        plan = ladder.on_pressure_boundary(series([0.95, 0.96]))
        self.assertEqual(
            plan.required_resident_items, [graph_rung_item_id("dcp_ratio")]
        )

    def test_forced_target_zero_from_a_higher_rung_descends(self):
        ladder = self._ladder()
        ladder.apply(ladder.on_pressure_boundary(series([0.95, 0.96])))
        plan = ladder.on_pressure_boundary(force_target=0)
        self.assertEqual(plan.phase, PHASE_DESCEND)
        self.assertEqual(plan.target_rung, 0)

    def test_forced_target_out_of_range(self):
        ladder = self._ladder()
        with self.assertRaises(KvLadderError):
            ladder.on_pressure_boundary(force_target=99)


# ---------------------------------------------------------------------------
# 5. Pre-staging (Erg. 9b)
# ---------------------------------------------------------------------------


def geometry_only_table() -> PressureLadder:
    """A ladder whose first rung above base is a geometry flip -- pre-staging
    only makes sense where a card actually joins."""
    return PressureLadder([base_step(), geometry_step("tp2"), geometry_step("tp3")])


class TestPreStaging(unittest.TestCase):
    def _ladder(self, **kwargs) -> KvPressureLadder:
        kwargs.setdefault("pre_stage_enabled", True)
        return KvPressureLadder(
            kwargs.pop("table", geometry_only_table()),
            kwargs.pop("sensor", fast_sensor(horizon_rounds=1)),
            **kwargs,
        )

    def test_pre_stage_plan_names_the_shadow_items(self):
        ladder = self._ladder()
        plan = ladder.on_pressure_boundary(series([0.72, 0.73]))
        self.assertEqual(plan.phase, PHASE_PRE_STAGE)
        self.assertEqual(plan.target_rung, 1)
        self.assertEqual(plan.handover, HANDOVER_ANTICIPATORY_SHADOW)
        self.assertEqual(plan.shadow_items, [kv_shadow_item_id("tp2")])
        ladder.apply(plan)
        self.assertEqual(ladder.staged_target, 1)
        # The rung itself has NOT moved: the old layout is authoritative.
        self.assertEqual(ladder.current_rung, 0)

    def test_flip_onto_a_warm_shadow_moves_only_the_delta(self):
        ladder = self._ladder()
        ladder.apply(ladder.on_pressure_boundary(series([0.72, 0.73])))
        ladder.sensor.reset()
        plan = ladder.on_pressure_boundary(series([0.95, 0.96], start_round=2))
        self.assertEqual(plan.phase, PHASE_FLIP)
        self.assertTrue(plan.delta_only)
        self.assertEqual(plan.handover, HANDOVER_ANTICIPATORY_SHADOW)
        ladder.apply(plan)
        self.assertEqual(ladder.current_rung, 1)
        self.assertIsNone(ladder.staged_target)

    def test_flip_without_a_shadow_uses_the_steps_own_handover(self):
        ladder = self._ladder()
        plan = ladder.on_pressure_boundary(series([0.95, 0.96]))
        self.assertEqual(plan.phase, PHASE_FLIP)
        self.assertFalse(plan.delta_only)
        self.assertEqual(plan.handover, HANDOVER_BACKGROUND_MIGRATE)

    def test_abort_discards_for_free(self):
        ladder = self._ladder()
        ladder.apply(ladder.on_pressure_boundary(series([0.72, 0.73])))
        plan = ladder.on_pressure_boundary(series([0.3, 0.3, 0.3], start_round=2))
        self.assertEqual(plan.phase, PHASE_ABORT_STAGE)
        # Free means: only a discard, nothing to copy back, and the rung is
        # exactly where it was.
        self.assertEqual(plan.discard_items, [kv_shadow_item_id("tp2")])
        self.assertEqual(plan.shadow_items, [])
        self.assertEqual(plan.required_resident_items, [])
        self.assertEqual(plan.target_rung, plan.current_rung)
        self.assertIn("Discarding is FREE", plan.reason)
        ladder.apply(plan)
        self.assertIsNone(ladder.staged_target)
        self.assertEqual(ladder.current_rung, 0)

    def test_flapping_does_not_stage_permanently(self):
        """A series oscillating around the pre-stage mark must not leave the
        ladder staged forever, and must not stage on every up-tick."""
        ladder = self._ladder()
        stages = 0
        aborts = 0
        round_index = 0
        for _ in range(6):
            for values in ([0.72, 0.73], [0.3, 0.3, 0.3]):
                plan = ladder.on_pressure_boundary(
                    series(values, start_round=round_index)
                )
                round_index += len(values)
                if plan.phase == PHASE_PRE_STAGE:
                    stages += 1
                elif plan.phase == PHASE_ABORT_STAGE:
                    aborts += 1
                ladder.apply(plan)
        self.assertGreater(stages, 0)
        self.assertEqual(stages, aborts)  # every stage was discarded again
        self.assertIsNone(ladder.staged_target)
        self.assertEqual(ladder.current_rung, 0)

    def test_no_second_shadow_while_one_is_staged(self):
        ladder = self._ladder()
        ladder.apply(ladder.on_pressure_boundary(series([0.72, 0.73])))
        plan = ladder.on_pressure_boundary(series([0.72], start_round=2))
        self.assertTrue(plan.is_noop)
        self.assertIn("already staged", plan.reason)

    def test_relief_rung_is_never_pre_staged(self):
        ladder = self._ladder(table=full_table())
        plan = ladder.on_pressure_boundary(series([0.72, 0.73]))
        self.assertTrue(plan.is_noop)
        self.assertIn("no layout change, nothing to shadow", plan.reason)

    def test_ladder_on_staging_off_never_touches_a_shadow(self):
        """The combination the user calls out explicitly: the ladder works on
        its own, and with staging off no kv_shadow item is ever named."""
        ladder = self._ladder(pre_stage_enabled=False)
        self.assertFalse(ladder.pre_stage_enabled)
        round_index = 0
        phases = []
        for values in ([0.72, 0.73], [0.3, 0.3, 0.3], [0.95, 0.96]):
            plan = ladder.on_pressure_boundary(series(values, start_round=round_index))
            round_index += len(values)
            phases.append(plan.phase)
            self.assertEqual(plan.shadow_items, [])
            self.assertEqual(plan.discard_items, [])
            self.assertFalse(plan.delta_only)
            ladder.apply(plan)
        self.assertNotIn(PHASE_PRE_STAGE, phases)
        self.assertNotIn(PHASE_ABORT_STAGE, phases)
        self.assertIn(PHASE_FLIP, phases)
        self.assertIsNone(ladder.staged_target)
        self.assertEqual(ladder.current_rung, 1)


# ---------------------------------------------------------------------------
# 6. Register wiring + conflict freedom with the admission planner
# ---------------------------------------------------------------------------


class TestRegisterWiring(unittest.TestCase):
    def test_kv_shadow_is_a_register_class(self):
        for klass in LADDER_REGISTER_CLASSES:
            self.assertIn(klass, OFFLOAD_CLASSES)
        policies = resolve_class_policies("latency")
        self.assertIn("kv_shadow", policies)

    def test_plans_of_both_boundary_planners_are_disjoint(self):
        reg = OffloadRegister(
            policies=resolve_class_policies("capacity"),
            backend=CpuFakeMovementBackend(),
            hysteresis_window_s=0.0,
        )
        for slot in range(1, 4):
            reg.register(f"gdn_state_set/{slot:05d}", "gdn_state_sets", 1024, 1.0)
        admission_plan = reg.on_admission_boundary(running_sessions=1)

        ladder = KvPressureLadder(
            geometry_only_table(),
            fast_sensor(horizon_rounds=1),
            pre_stage_enabled=True,
        )
        stage_plan = ladder.on_pressure_boundary(series([0.72, 0.73]))
        ladder.apply(stage_plan)
        flip_plan = ladder.on_pressure_boundary(series([0.95, 0.96], start_round=2))

        for plan in (stage_plan, flip_plan):
            self.assertTrue(plan.touched_items())
            self.assertEqual(plan_conflicts(plan, admission_plan), ())
            self.assertEqual(resolve_plan_priority(plan, admission_plan), "disjoint")

    def test_a_real_collision_resolves_deterministically_to_admission(self):
        class _FakeAdmissionPlan:
            park_candidates = [graph_rung_item_id("tp2")]
            wave_in_candidates: List[str] = []

        ladder = KvPressureLadder(geometry_only_table(), fast_sensor(horizon_rounds=1))
        plan = ladder.on_pressure_boundary(series([0.95, 0.96]))
        collision = _FakeAdmissionPlan()
        self.assertEqual(plan_conflicts(plan, collision), (graph_rung_item_id("tp2"),))
        # Correctness before capacity: an arriving session never waits for a
        # pressure step.
        self.assertEqual(resolve_plan_priority(plan, collision), "admission")

    def test_touched_items_are_sorted_and_deduplicated(self):
        ladder = KvPressureLadder(
            geometry_only_table(),
            fast_sensor(horizon_rounds=1),
            pre_stage_enabled=True,
        )
        plan = ladder.on_pressure_boundary(series([0.72, 0.73]))
        items = plan.touched_items()
        self.assertEqual(list(items), sorted(set(items)))


# ---------------------------------------------------------------------------
# 7. Flags
# ---------------------------------------------------------------------------


class TestFlagParsing(unittest.TestCase):
    def test_unset_is_ladder_off(self):
        self.assertIsNone(parse_kv_pressure_ladder(None))
        self.assertIsNone(parse_kv_pressure_ladder("   "))

    def test_auto(self):
        self.assertEqual(parse_kv_pressure_ladder("auto"), "auto")

    def test_explicit_spec(self):
        entries = parse_kv_pressure_ladder(
            "relief:dcp_ratio, relief:kv_spill, geometry:tp2, external:node2"
        )
        self.assertEqual(
            entries,
            (
                (STEP_RELIEF, "dcp_ratio"),
                (STEP_RELIEF, "kv_spill"),
                (STEP_GEOMETRY, "tp2"),
                (STEP_EXTERNAL, "node2"),
            ),
        )

    def test_bad_specs_are_hard_errors(self):
        for spec in (
            "relief",  # no type separator
            "relief:",  # empty name
            "wishful:tp2",  # unknown type
            "relief:magic",  # unknown relief feature
            "geometry:tp2,relief:kv_spill",  # order violated
            "external:n,geometry:tp2",  # order violated
            "geometry:tp2,geometry:tp2",  # duplicate
            "relief:dcp_ratio,,geometry:tp2",  # empty entry
        ):
            with self.assertRaises(ValueError, msg=spec):
                parse_kv_pressure_ladder(spec)

    def test_ladder_from_spec_builds_a_valid_placeholder_table(self):
        entries = parse_kv_pressure_ladder(
            "relief:dcp_ratio,geometry:tp2,external:node2"
        )
        table = ladder_from_spec(entries)
        self.assertEqual(len(table), 4)
        self.assertEqual(table[0].step_type, STEP_BASE)
        self.assertEqual(table[1].relief_feature, "dcp_ratio")
        self.assertEqual(table[2].handover, HANDOVER_BACKGROUND_MIGRATE)
        self.assertEqual(
            table[3].min_hysteresis_rounds, DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS
        )
        for step in table.steps:
            self.assertEqual(step.provenance, PROVENANCE_PLACEHOLDER)
            self.assertIsNone(step.expected_kv_tokens)

    def test_flag_off_builds_nothing_at_all(self):
        self.assertIsNone(build_ladder_from_server_args(_Args()))

    def test_auto_without_a_table_source_is_a_hard_error(self):
        with self.assertRaises(ValueError) as ctx:
            build_ladder_from_server_args(_Args(kv_pressure_ladder="auto"))
        self.assertIn("planner", str(ctx.exception))

    def test_auto_with_an_injected_table(self):
        ladder = build_ladder_from_server_args(
            _Args(kv_pressure_ladder="auto"), table_fn=full_table
        )
        self.assertEqual(len(ladder.table), 6)
        self.assertFalse(ladder.pre_stage_enabled)

    def test_pre_stage_flag_is_independent(self):
        args = _Args(
            kv_pressure_ladder="relief:dcp_ratio,geometry:tp2",
            kv_pressure_pre_stage=True,
        )
        ladder = build_ladder_from_server_args(args)
        self.assertTrue(ladder.pre_stage_enabled)
        args.kv_pressure_pre_stage = False
        self.assertFalse(build_ladder_from_server_args(args).pre_stage_enabled)

    def test_sensor_takes_the_flag_values(self):
        ladder = build_ladder_from_server_args(
            _Args(
                kv_pressure_ladder="geometry:tp2",
                kv_pressure_ascend_threshold=0.9,
                kv_pressure_ascend_window=3,
                kv_pressure_descend_window=9,
            )
        )
        self.assertAlmostEqual(ladder.sensor.ascend_threshold, 0.9)
        self.assertEqual(ladder.sensor.ascend_window, 3)
        self.assertEqual(ladder.sensor.descend_window, 9)


class TestServerArgsValidation(unittest.TestCase):
    """The flags fail at ARGUMENT time, not at the first pressure boundary.

    ``model_path='dummy'`` short-circuits ``__post_init__`` (no accelerator
    needed), so the handler is driven in isolation with exactly the fields
    under test -- the same pattern as the offload-register flag tests.
    """

    def _args(self, **kwargs):
        from sglang.srt.server_args import ServerArgs

        return ServerArgs(model_path="dummy", **kwargs)

    def test_default_is_off_for_both_features(self):
        args = self._args()
        self.assertIsNone(args.kv_pressure_ladder)
        self.assertFalse(args.kv_pressure_pre_stage)
        args._handle_kv_pressure_ladder()  # must not raise
        self.assertIsNone(build_ladder_from_server_args(args))

    def test_bad_spec_rejected(self):
        args = self._args(kv_pressure_ladder="geometry:tp2,relief:kv_spill")
        with self.assertRaisesRegex(ValueError, "climb order"):
            args._handle_kv_pressure_ladder()

    def test_pre_stage_without_a_ladder_rejected(self):
        args = self._args(kv_pressure_pre_stage=True)
        with self.assertRaisesRegex(ValueError, "needs --kv-pressure-ladder"):
            args._handle_kv_pressure_ladder()

    def test_ladder_without_pre_stage_accepted(self):
        args = self._args(kv_pressure_ladder="relief:kv_spill")
        args._handle_kv_pressure_ladder()  # must not raise
        self.assertFalse(args.kv_pressure_pre_stage)
        ladder = build_ladder_from_server_args(args)
        self.assertFalse(ladder.pre_stage_enabled)
        self.assertEqual(len(ladder.table), 2)

    def test_both_on_accepted(self):
        args = self._args(kv_pressure_ladder="geometry:tp2", kv_pressure_pre_stage=True)
        args._handle_kv_pressure_ladder()  # must not raise

    def test_auto_spec_accepted(self):
        self._args(kv_pressure_ladder="auto")._handle_kv_pressure_ladder()

    def test_symmetric_windows_rejected(self):
        args = self._args(
            kv_pressure_ladder="geometry:tp2",
            kv_pressure_ascend_window=8,
            kv_pressure_descend_window=8,
        )
        with self.assertRaisesRegex(ValueError, "asymmetry is the contract"):
            args._handle_kv_pressure_ladder()

    def test_symmetric_stage_windows_rejected(self):
        args = self._args(
            kv_pressure_ladder="geometry:tp2",
            kv_pressure_pre_stage_window=6,
            kv_pressure_abort_stage_window=6,
        )
        with self.assertRaisesRegex(ValueError, "stage and discard forever"):
            args._handle_kv_pressure_ladder()

    def test_admission_rung_needs_the_ceiling_at_argument_time(self):
        """#287 runtime: a rung whose wired actuator cannot exist in this
        configuration fails at parse time, not at the first episode."""
        args = self._args(kv_pressure_ladder="relief:admission_cap")
        with self.assertRaisesRegex(
            ValueError, "max-running-requests-ceiling"
        ):
            args._handle_kv_pressure_ladder()

    def test_admission_rung_with_ceiling_accepted(self):
        args = self._args(
            kv_pressure_ladder="relief:dcp_ratio,relief:admission_cap",
            max_running_requests_ceiling=64,
        )
        args._handle_kv_pressure_ladder()  # must not raise

    def test_session_offload_rung_needs_the_manager_flag(self):
        args = self._args(kv_pressure_ladder="relief:session_offload")
        with self.assertRaisesRegex(ValueError, "enable-kv-session-offload"):
            args._handle_kv_pressure_ladder()
        ok = self._args(
            kv_pressure_ladder="relief:session_offload",
            enable_kv_session_offload=True,
        )
        ok._handle_kv_pressure_ladder()  # must not raise

    def test_consensus_interval_validated(self):
        args = self._args(
            kv_pressure_ladder="geometry:tp2",
            kv_pressure_consensus_interval=0,
        )
        with self.assertRaisesRegex(ValueError, "consensus-interval"):
            args._handle_kv_pressure_ladder()
        self.assertEqual(self._args().kv_pressure_consensus_interval, 8)

    def test_mark_order_rejected(self):
        args = self._args(
            kv_pressure_ladder="geometry:tp2",
            kv_pressure_pre_stage_threshold=0.95,
        )
        with self.assertRaisesRegex(ValueError, "descend < pre_stage < ascend"):
            args._handle_kv_pressure_ladder()

    def test_external_hysteresis_must_be_positive(self):
        args = self._args(
            kv_pressure_ladder="geometry:tp2",
            kv_pressure_external_hysteresis_rounds=0,
        )
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            args._handle_kv_pressure_ladder()


if __name__ == "__main__":
    unittest.main()
