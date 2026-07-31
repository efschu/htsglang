# SPDX-License-Identifier: Apache-2.0
"""Depth/format-aware operating grid of the KV pressure ladder (#287).

Hermetic. The mock rates mirror the shape of the reference rig's #324
scores (one fast card, two slow ones); every assertion is about STRUCTURE
(trajectory, determinism, provenance honesty), not about reproducing a
measured number."""

import unittest

from sglang.srt.planner.kv_ladder_table import (
    CardSpec,
    GeometryRungSpec,
    RankScoreProfile,
    RigModelProfile,
    build_ladder_table,
    solve_operating_grid,
)

GIB = 1 << 30


def _cards():
    return (
        CardSpec(0, "fast", 32768),
        CardSpec(1, "slow-a", 20480),
        CardSpec(2, "slow-b", 20480),
    )


def _geom(key="tp3-7,3,3", ratio=(7, 3, 3)):
    return GeometryRungSpec(key, ratio, (0, 1, 2))


def _scores(**overrides):
    fields = dict(
        card_prefill_tflops={0: 566.9, 1: 178.8, 2: 178.8},
        card_attn_tflops={0: 566.9, 1: 186.6, 2: 186.6},
        card_membw_gbs={0: 1558.0, 1: 723.0, 2: 723.0},
        source="mock of the #324-derived per-card rates",
    )
    fields.update(overrides)
    return RankScoreProfile(**fields)


def _profile(scores=..., reliefs=("dcp_ratio",), geometries=None, **overrides):
    if scores is ...:
        scores = _scores()
    fields = dict(
        cards=_cards(),
        geometries=geometries or (_geom(),),
        reliefs=reliefs,
        kv_bytes_per_token=120_000,
        weight_bytes_total=28 * GIB,
        overhead_mib_per_rank=2048,
        rank_scores=scores,
        attn_context_flops_per_token_pair=4 * 24 * 32 * 128.0,
    )
    fields.update(overrides)
    return RigModelProfile(**fields)


def _vectors(grid, phase):
    return [(p.depth_fraction, p.kv_vector) for p in grid.points if p.phase == phase]


class TestGridTrajectory(unittest.TestCase):
    def test_grid_is_solved_with_solver_provenance(self):
        grid, _notes = solve_operating_grid(_profile(), _geom())
        self.assertIsNotNone(grid)
        self.assertEqual({p.provenance for p in grid.points}, {"solver"})
        self.assertEqual(sorted(grid.phases), ["decode", "prefill"])

    def test_low_fill_concentrates_high_fill_reaches_the_capacity_pole(self):
        """The trajectory of the user directive: the perf pole (fast-card
        concentration) at low fill, the capacity pole (free-bytes
        proportional) at full fill, with genuine optima in between."""
        grid, notes = solve_operating_grid(_profile(), _geom())
        prefill = dict(_vectors(grid, "prefill"))
        low, high = prefill[0.05], prefill[1.0]
        share = lambda v: v[0] / sum(v)  # noqa: E731
        self.assertGreater(
            share(low),
            share(high),
            "low fill must load the fast card harder than full fill",
        )
        # Full fill cannot be held by any integer candidate exactly (the
        # ideal vector is fractional); the solver says so and names the
        # capacity pole instead of inventing a fit.
        self.assertTrue(any("capacity pole" in n for n in notes))

    def test_interior_bins_have_their_own_optima(self):
        """'Also in between, not only at the poles': at least one interior
        bin differs from BOTH poles."""
        grid, _ = solve_operating_grid(_profile(), _geom())
        prefill = dict(_vectors(grid, "prefill"))
        interior = {prefill[b] for b in (0.25, 0.5, 0.75)}
        self.assertTrue(
            any(v != prefill[0.05] and v != prefill[1.0] for v in interior),
            f"interior optima {interior} collapsed onto the poles "
            f"{prefill[0.05]} / {prefill[1.0]}",
        )

    def test_grid_is_deterministic(self):
        a, _ = solve_operating_grid(_profile(), _geom())
        b, _ = solve_operating_grid(_profile(), _geom())
        self.assertEqual(a.describe(), b.describe())


class TestFormatAwareness(unittest.TestCase):
    def test_format_scores_move_the_optimum(self):
        """The format axis is DATA: changing only the per-card rates (what a
        different checkpoint format would measure on the same silicon, #324)
        changes solved optima. Nothing in the solver knows an arch."""
        base_grid, _ = solve_operating_grid(_profile(), _geom())
        # The same cards under a format whose lane on card 0 is ~3x slower
        # (the Marlin-vs-native band of ANALYSE_321).
        slow0 = _scores(
            card_prefill_tflops={0: 216.0, 1: 178.8, 2: 178.8},
            card_attn_tflops={0: 216.0, 1: 186.6, 2: 186.6},
        )
        slow_grid, _ = solve_operating_grid(_profile(scores=slow0), _geom())
        self.assertNotEqual(
            base_grid.describe(),
            slow_grid.describe(),
            "a 3x format factor on one card must move at least one optimum",
        )

    def test_missing_scores_are_named_not_guessed(self):
        grid, notes = solve_operating_grid(_profile(scores=None), _geom())
        self.assertIsNone(grid)
        self.assertTrue(any("no rank_scores" in n for n in notes))

    def test_missing_card_rate_is_named(self):
        scores = _scores(card_membw_gbs={0: 1558.0, 1: 723.0})  # card 2 absent
        grid, notes = solve_operating_grid(_profile(scores=scores), _geom())
        self.assertIsNone(grid)
        self.assertTrue(any("membw" in n and "card 2" in n for n in notes))

    def test_depth_bins_outside_unit_interval_rejected(self):
        with self.assertRaisesRegex(ValueError, "depth_bins"):
            solve_operating_grid(_profile(), _geom(), depth_bins=(0.5, 1.5))


class TestTableAttachment(unittest.TestCase):
    def test_grids_land_on_base_dcp_ratio_and_geometry_rungs(self):
        profile = _profile(
            reliefs=("dcp_ratio", "admission_cap"),
            geometries=(_geom(), _geom("tp3-5,4,4", (5, 4, 4))),
            mlp_units=None,
        )
        table = build_ladder_table(profile)
        by_name = {s.name: s for s in table.steps}
        self.assertIsNotNone(by_name["tp3-7,3,3"].operating_grid)
        self.assertIsNotNone(by_name["dcp_ratio"].operating_grid)
        self.assertIsNotNone(by_name["tp3-5,4,4"].operating_grid)
        self.assertIsNone(by_name["admission_cap"].operating_grid)
        # The dcp_ratio rung flips vectors at the UNCHANGED base split.
        self.assertEqual(
            by_name["dcp_ratio"].operating_grid.describe(),
            by_name["tp3-7,3,3"].operating_grid.describe(),
        )
        self.assertIn("grid:", by_name["tp3-7,3,3"].source)

    def test_unsolved_grid_is_named_in_the_step_source(self):
        table = build_ladder_table(_profile(scores=None))
        base = table.steps[0]
        self.assertIsNone(base.operating_grid)
        self.assertIn("no rank_scores", base.source)


if __name__ == "__main__":
    unittest.main()
