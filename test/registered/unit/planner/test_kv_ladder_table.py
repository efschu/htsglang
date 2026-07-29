"""#286 Erg. 9 -- the KV pressure ladder's STEP TABLE, computed in advance.

Hermetic tests (no GPU, no probe, no checkpoint) for the planner-side table
generation: rung order, the nesting-family check via the #272 solver's
``nesting_hull``, the solver capacity arithmetic, and the honest placeholder
labelling wherever the profile does not carry an input.
"""

import unittest

from sglang.srt.model_executor.kv_pressure_ladder import (
    HANDOVER_BACKGROUND_MIGRATE,
    HANDOVER_NONE,
    HANDOVER_SPILL_RELOAD,
    PROVENANCE_PLACEHOLDER,
    PROVENANCE_SOLVER,
    STEP_BASE,
    STEP_EXTERNAL,
    STEP_GEOMETRY,
    STEP_RELIEF,
    KvPressureLadder,
    KvPressureSensor,
)
from sglang.srt.planner.kv_ladder_table import (
    MIB,
    CardSpec,
    ExternalRungSpec,
    GeometryRungSpec,
    RigModelProfile,
    build_ladder_table,
    capacity_from_report,
    check_geometry_family,
    solver_capacity_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


# The rig, with placeholder physical indices: one big card and two smaller
# ones. The table generator never assumes which index is which card.
FAST = CardSpec(index=0, name="big", total_mib=32768)
SMALL_A = CardSpec(index=1, name="small-a", total_mib=20480)
SMALL_B = CardSpec(index=2, name="small-b", total_mib=20480)


def nesting_geometries():
    """TP=1 -> TP=2 -> TP=3, cuts that refine each other on a 16-unit grid."""
    return (
        GeometryRungSpec(key="tp1", ratio=(16,), gpus=(0,)),
        GeometryRungSpec(key="tp2", ratio=(8, 8), gpus=(0, 1)),
        GeometryRungSpec(key="tp3", ratio=(4, 4, 8), gpus=(0, 1, 2)),
    )


def profile(**kwargs) -> RigModelProfile:
    base = dict(
        cards=(FAST, SMALL_A, SMALL_B),
        geometries=nesting_geometries(),
        reliefs=("kv_spill", "dcp_ratio"),
        kv_bytes_per_token=100_000,
        weight_bytes_total=16 * (1 << 30),
        overhead_mib_per_rank=1024,
        budget_fraction=0.9,
    )
    base.update(kwargs)
    return RigModelProfile(**base)


class TestProfileValidation(unittest.TestCase):
    def test_needs_cards_and_geometries(self):
        with self.assertRaises(ValueError):
            RigModelProfile(cards=(), geometries=nesting_geometries())
        with self.assertRaises(ValueError):
            RigModelProfile(cards=(FAST,), geometries=())

    def test_geometry_must_use_declared_cards(self):
        with self.assertRaisesRegex(ValueError, "does not declare"):
            RigModelProfile(
                cards=(FAST,),
                geometries=(GeometryRungSpec(key="tp2", ratio=(8, 8), gpus=(0, 7)),),
            )

    def test_unknown_relief_rejected(self):
        with self.assertRaisesRegex(ValueError, "only ORDERS existing"):
            profile(reliefs=("telepathy",))

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            RigModelProfile(
                cards=(FAST,),
                geometries=(
                    GeometryRungSpec(key="tp1", ratio=(16,), gpus=(0,)),
                    GeometryRungSpec(key="tp1", ratio=(8,), gpus=(0,)),
                ),
            )
        with self.assertRaises(ValueError):
            profile(externals=(ExternalRungSpec(key="tp2"),))

    def test_rank_and_card_counts_must_match(self):
        with self.assertRaisesRegex(ValueError, "one card per rank"):
            GeometryRungSpec(key="x", ratio=(8, 8), gpus=(0,))
        with self.assertRaisesRegex(ValueError, "every rank carries weight"):
            GeometryRungSpec(key="x", ratio=(8, 0), gpus=(0, 1))

    def test_card_validation(self):
        with self.assertRaises(ValueError):
            CardSpec(index=0, name="broken", total_mib=0)
        with self.assertRaises(ValueError):
            CardSpec(index=0, name="broken", total_mib=1000, budget_mib=0)


class TestFamilyCheck(unittest.TestCase):
    def test_skipped_without_mlp_units_and_says_so(self):
        ok, notes = check_geometry_family(profile())
        self.assertTrue(ok)
        self.assertIn("SKIPPED", notes[0])
        self.assertIn("mlp_units", notes[0])

    def test_single_geometry_is_trivial(self):
        ok, notes = check_geometry_family(
            profile(
                geometries=(GeometryRungSpec(key="tp1", ratio=(16,), gpus=(0,)),),
                mlp_units=136,
            )
        )
        self.assertTrue(ok)

    def test_nesting_family_verified_by_the_solver(self):
        ok, notes = check_geometry_family(profile(mlp_units=136))
        self.assertTrue(ok, notes)

    def test_non_nesting_geometries_are_rejected_by_the_table(self):
        # Two lanes that share cards but cut in different places: [6,2] and
        # [7,1] over the same pair -- the solver's own counter-example.
        bad = profile(
            geometries=(
                GeometryRungSpec(key="a", ratio=(6, 2), gpus=(0, 1)),
                GeometryRungSpec(key="b", ratio=(7, 1), gpus=(0, 1)),
            ),
            mlp_units=136,
        )
        ok, notes = check_geometry_family(bad)
        self.assertFalse(ok)
        self.assertTrue(notes)
        with self.assertRaisesRegex(ValueError, "RESHARD"):
            build_ladder_table(bad)
        # ...and it can still be inspected on request.
        table = build_ladder_table(bad, require_family=False)
        self.assertEqual([s.name for s in table.steps][-1], "b")


class TestSolverCapacity(unittest.TestCase):
    def test_arithmetic(self):
        prof = profile()
        geom = prof.geometries[1]  # tp2 over cards 0 and 1
        tokens, prov, src = solver_capacity_tokens(prof, geom)
        budget0 = 32768 * MIB * 0.9
        budget1 = 20480 * MIB * 0.9
        weight = 16 * (1 << 30) / 2.0
        overhead = 1024 * MIB
        expected = int(
            ((budget0 - weight - overhead) + (budget1 - weight - overhead)) // 100_000
        )
        self.assertEqual(tokens, expected)
        self.assertEqual(prov, PROVENANCE_SOLVER)
        self.assertIn("solver_capacity_tokens", src)

    def test_explicit_budget_mib_beats_the_fraction(self):
        pinned = CardSpec(index=0, name="big", total_mib=32768, budget_mib=24000)
        prof = profile(
            cards=(pinned, SMALL_A, SMALL_B),
            geometries=(GeometryRungSpec(key="tp1", ratio=(16,), gpus=(0,)),),
        )
        tokens, _, _ = solver_capacity_tokens(prof, prof.geometries[0])
        expected = int((24000 * MIB - 16 * (1 << 30) - 1024 * MIB) // 100_000)
        self.assertEqual(tokens, expected)

    def test_missing_inputs_yield_labelled_placeholders(self):
        prof = profile(kv_bytes_per_token=None)
        tokens, prov, src = solver_capacity_tokens(prof, prof.geometries[0])
        self.assertIsNone(tokens)
        self.assertEqual(prov, PROVENANCE_PLACEHOLDER)
        self.assertIn("kv_bytes_per_token", src)

        prof = profile(weight_bytes_total=None)
        tokens, prov, src = solver_capacity_tokens(prof, prof.geometries[0])
        self.assertIsNone(tokens)
        self.assertIn("weight_bytes_total", src)

    def test_a_rung_that_does_not_fit_reports_zero(self):
        prof = profile(weight_bytes_total=200 * (1 << 30))
        tokens, prov, _ = solver_capacity_tokens(prof, prof.geometries[0])
        self.assertEqual(tokens, 0)
        self.assertEqual(prov, PROVENANCE_SOLVER)

    def test_capacity_from_report_adapter(self):
        class _Report:
            feasible = True
            max_context_tokens = 123456.7

        tokens, prov, src = capacity_from_report(_Report())
        self.assertEqual(tokens, 123456)
        # An estimate is 'solver', never 'measured'.
        self.assertEqual(prov, PROVENANCE_SOLVER)
        self.assertIn("predict_capacity", src)

        class _Infeasible:
            feasible = False
            max_context_tokens = -1.0

        tokens, prov, _ = capacity_from_report(_Infeasible())
        self.assertIsNone(tokens)
        self.assertEqual(prov, PROVENANCE_PLACEHOLDER)
        self.assertIsNone(capacity_from_report(None)[0])


class TestBuildLadderTable(unittest.TestCase):
    def test_rung_order_is_base_relief_geometry_external(self):
        table = build_ladder_table(profile(externals=(ExternalRungSpec(key="node2"),)))
        types = [s.step_type for s in table.steps]
        self.assertEqual(
            types,
            [
                STEP_BASE,
                STEP_RELIEF,
                STEP_RELIEF,
                STEP_GEOMETRY,
                STEP_GEOMETRY,
                STEP_EXTERNAL,
            ],
        )
        # Relief steps come out in the CANONICAL cheapness order, not in the
        # order the profile happened to list them.
        self.assertEqual(
            [s.relief_feature for s in table.steps if s.step_type == STEP_RELIEF],
            ["dcp_ratio", "kv_spill"],
        )
        self.assertEqual([s.name for s in table.steps][3:5], ["tp2", "tp3"])

    def test_handovers_per_step_type(self):
        table = build_ladder_table(profile(externals=(ExternalRungSpec(key="node2"),)))
        self.assertEqual(table[0].handover, HANDOVER_NONE)
        self.assertEqual(table[1].handover, HANDOVER_NONE)
        self.assertEqual(table[3].handover, HANDOVER_BACKGROUND_MIGRATE)
        self.assertEqual(table[5].handover, HANDOVER_SPILL_RELOAD)

    def test_capacity_grows_with_the_geometry_rungs(self):
        table = build_ladder_table(profile())
        geometry_tokens = [
            s.expected_kv_tokens
            for s in table.steps
            if s.step_type in (STEP_BASE, STEP_GEOMETRY)
        ]
        self.assertEqual(len(geometry_tokens), 3)
        self.assertTrue(all(t is not None for t in geometry_tokens))
        self.assertLess(geometry_tokens[0], geometry_tokens[1])
        self.assertLess(geometry_tokens[1], geometry_tokens[2])

    def test_relief_gains_are_placeholders_with_a_named_reason(self):
        table = build_ladder_table(profile())
        for step in table.steps:
            if step.step_type == STEP_RELIEF:
                self.assertIsNone(step.expected_kv_tokens)
                self.assertEqual(step.provenance, PROVENANCE_PLACEHOLDER)
                self.assertIn("Messpflicht", step.source)

    def test_costs_are_placeholders_except_the_base_reference(self):
        table = build_ladder_table(profile())
        self.assertEqual(table[0].expected_cost_factor, 1.0)
        self.assertIn("cost reference", table[0].source)
        for step in table.steps[1:]:
            self.assertIsNone(step.expected_cost_factor)

    def test_injected_measurement_functions_are_used(self):
        def capacity_fn(prof, geom):
            return ({"tp1": 100, "tp2": 200, "tp3": 300}[geom.key], "measured", "fake")

        def relief_gain_fn(feature, running):
            return (10, "measured", "fake relief measurement")

        def cost_fn(name, step_type):
            return (
                {
                    "tp1": 1.0,
                    "dcp_ratio": 1.05,
                    "kv_spill": 1.1,
                    "tp2": 1.3,
                    "tp3": 1.6,
                }[name],
                "measured",
                "fake cost measurement",
            )

        table = build_ladder_table(
            profile(),
            capacity_fn=capacity_fn,
            relief_gain_fn=relief_gain_fn,
            cost_fn=cost_fn,
        )
        self.assertEqual(
            [s.expected_kv_tokens for s in table.steps], [100, 110, 120, 200, 300]
        )
        self.assertEqual(table[1].provenance, "measured")
        self.assertAlmostEqual(table[4].expected_cost_factor, 1.6)

    def test_a_measured_table_that_loses_capacity_is_rejected(self):
        def capacity_fn(prof, geom):
            return ({"tp1": 300, "tp2": 200, "tp3": 100}[geom.key], "measured", "fake")

        with self.assertRaisesRegex(ValueError, "may not lose capacity"):
            build_ladder_table(profile(), capacity_fn=capacity_fn)

    def test_uncaptured_rung_is_carried_into_the_table_not_hidden(self):
        geoms = (
            GeometryRungSpec(key="tp1", ratio=(16,), gpus=(0,)),
            GeometryRungSpec(
                key="tp2", ratio=(8, 8), gpus=(0, 1), graphs_precaptured=False
            ),
        )
        table = build_ladder_table(profile(geometries=geoms, reliefs=()))
        self.assertFalse(table[1].graphs_precaptured)

    def test_external_hysteresis_default_and_override(self):
        table = build_ladder_table(
            profile(
                externals=(
                    ExternalRungSpec(key="node2"),
                    ExternalRungSpec(key="node3", min_hysteresis_rounds=900),
                )
            ),
            external_min_hysteresis_rounds=600,
        )
        self.assertEqual(table[-2].min_hysteresis_rounds, 600)
        self.assertEqual(table[-1].min_hysteresis_rounds, 900)

    def test_the_generated_table_drives_the_runtime_controller(self):
        """End to end for the CPU phase: the planner's table is exactly what
        the flip contract consumes -- no adapter in between."""
        table = build_ladder_table(profile())
        ladder = KvPressureLadder(
            table,
            KvPressureSensor(
                ascend_window=2,
                descend_window=4,
                pre_stage_window=2,
                abort_stage_window=3,
                horizon_rounds=1,
            ),
        )
        from sglang.srt.model_executor.kv_pressure_ladder import (
            PHASE_FLIP,
            OccupancySample,
        )

        samples = [
            OccupancySample(round_index=i, used_tokens=980, total_tokens=1000)
            for i in range(2)
        ]
        plan = ladder.on_pressure_boundary(samples)
        self.assertEqual(plan.phase, PHASE_FLIP)
        # The first rung above base is the cheapest relief, never a geometry.
        self.assertEqual(table[plan.target_rung].step_type, STEP_RELIEF)
        self.assertEqual(table[plan.target_rung].relief_feature, "dcp_ratio")


if __name__ == "__main__":
    unittest.main()
