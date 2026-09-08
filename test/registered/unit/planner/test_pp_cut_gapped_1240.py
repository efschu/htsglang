"""#1240 gapped layer sets in the PP cut solver: crossings, depth, one model.

The contiguous solver (#1236) cannot express the layout the user chose on
2026-09-07 -- 48 GDN layers on the 5090, the 16 interleaved full-attention
layers split across the two 3080s -- because a contiguous stage is an
interval and its attention count is a consequence of its boundaries. This
suite covers the three things that had to be added for a gapped map to be
RANKED against the contiguous cuts rather than merely listed beside them:

  * the gapped enumeration, and that the user's own 0/8/8 and 4/6/6 are in it;
  * the crossing term, priced from the measured per-peer link map, absent from
    no candidate and monotone in the number of crossings;
  * the depth axis -- an attention layer's cost per chunk grows with the
    prefix, a GDN layer's does not -- and that it MOVES the optimum, which is
    the whole reason a design prefix has to be named.

The table is synthetic on purpose (same discipline as
``test_pp_cut_launch_1236``): round numbers, so the expected direction is
arithmetic rather than a second run of the solver.
"""

import unittest

import pytest

try:
    from sglang.srt.planner.pp_cut import (
        LAYER_FAMILY_ATTENTION,
        LAYER_FAMILY_LINEAR,
        FamilyDepthCost,
        PhasePoolModel,
        UnpricedCrossing,
        attention_counts,
        attn_counts_of,
        contiguous_layer_sets,
        counts_of,
        crossing_price,
        enumerate_gapped_splits,
        family_costs_from_measurement,
        gapped_layer_sets,
        gapped_phase_pool,
        is_gapped,
        layer_set_flag,
        pp_phase_pool,
    )
    from sglang.srt.planner.pp_cut_launch import PPCutRefused, solve_launch_cut
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(
        f"#249 default-device collection leak broke the import chain: {_import_err}",
        allow_module_level=True,
    )

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

#: The reference checkpoint's own shape: 64 layers, every 4th full attention,
#: so 48 GDN and 16 attention layers -- the exact counts of the user's map.
FAMILIES = tuple(
    LAYER_FAMILY_ATTENTION if (i % 4) == 3 else LAYER_FAMILY_LINEAR for i in range(64)
)
N_LAYERS = 64
#: One fast card, two slow, exactly as the rig is ordered by ``order_cards``.
MS_PER_LAYER = (8.0, 32.0, 32.0)
INCUMBENT = (32, 18, 14)
CARDS = ("SYNTHETIC FAST", "SYNTHETIC SLOW", "SYNTHETIC SLOW")
#: Round link prices: the x8 edges cheap, the x4 edge dear, so the difference
#: shows up in the arithmetic instead of in the third decimal.
PAIR_MS = {(0, 1): 8.0, (1, 0): 8.0, (1, 2): 8.0, (2, 1): 8.0, (0, 2): 4.0, (2, 0): 4.0}


def family_cost(ref_prefix: float = 4096.0) -> FamilyDepthCost:
    """A hand-written split: a GDN layer costs 10/40/40 ms, attention 1/4/4."""
    return FamilyDepthCost(
        linear_ms_per_layer=(10.0, 40.0, 40.0),
        attn_ms_per_layer_at_ref=(1.0, 4.0, 4.0),
        chunk_tokens=4096,
        ref_prefix_tokens=ref_prefix,
    )


def pool_model(free_mib=(28000.0, 17000.0, 16500.0)) -> PhasePoolModel:
    return PhasePoolModel(
        free_mib=free_mib,
        weight_mib_per_layer=360.0,
        # The real cell: fp8_e4m3 KV, 2 x 4 kv-heads x 256 head_dim = 2048 B.
        kv_mib_per_token_per_attn_layer=2048.0 / (1024.0 * 1024.0),
        arming_floor_mib=(1200.0, 1200.0, 1200.0),
    )


class TheGappedFamilyIsEnumerated(unittest.TestCase):
    """The maps the contiguous enumeration cannot reach, and the user's two."""

    def test_every_split_of_the_attention_layers_is_offered(self):
        splits = enumerate_gapped_splits(16, 3, gdn_stage=0)
        # The GDN owner may take zero attention layers; the other two may not
        # own nothing at all.
        self.assertIn((0, 8, 8), splits)
        self.assertIn((4, 6, 6), splits)
        self.assertTrue(all(sum(s) == 16 for s in splits))
        self.assertTrue(all(s[1] >= 1 and s[2] >= 1 for s in splits))
        self.assertFalse(any(s[1] == 0 or s[2] == 0 for s in splits))

    def test_the_gdn_owner_is_enumerated_and_not_assumed(self):
        # "the 5090" is a fact about this rig, not about the solver.
        for gdn in (0, 1, 2):
            splits = enumerate_gapped_splits(16, 3, gdn_stage=gdn)
            self.assertTrue(any(s[gdn] == 0 for s in splits))

    def test_the_user_map_0_8_8_is_the_2026_08_18_boot_map(self):
        owned = gapped_layer_sets(FAMILIES, 0, (0, 8, 8))
        self.assertEqual(counts_of(owned), (48, 8, 8))
        self.assertEqual(attn_counts_of(FAMILIES, owned), (0, 8, 8))
        self.assertEqual(owned[1], (3, 7, 11, 15, 19, 23, 27, 31))
        self.assertEqual(owned[2], (35, 39, 43, 47, 51, 55, 59, 63))
        self.assertTrue(is_gapped(owned))
        self.assertTrue(layer_set_flag(owned).startswith("0-2,4-6,8-10,"))

    def test_the_user_map_4_6_6(self):
        owned = gapped_layer_sets(FAMILIES, 0, (4, 6, 6))
        self.assertEqual(counts_of(owned), (52, 6, 6))
        self.assertEqual(attn_counts_of(FAMILIES, owned), (4, 6, 6))
        # In-order dealing: the four attention layers stage 0 keeps are the
        # LOWEST four, so its ownership stays one unbroken run 0..18.
        self.assertEqual(owned[0][:20], tuple(range(19)) + (20,))

    def test_attention_layers_are_dealt_in_order_because_that_minimises_crossings(self):
        # The same 4/6/6 SPLIT, dealt two ways. In order, the four attention
        # layers the GDN owner keeps are its LOWEST four, so they sit inside
        # its own run and cost nothing. Given its HIGHEST four instead, the
        # terminal layer moves onto the GDN owner and the free last-layer
        # crossing is spent. Same split, more crossings: the dealing order is
        # the minimiser, not a convention.
        in_order = gapped_layer_sets(FAMILIES, 0, (4, 6, 6))
        attn_ids = [i for i, f in enumerate(FAMILIES) if f == LAYER_FAMILY_ATTENTION]
        lin_ids = [i for i, f in enumerate(FAMILIES) if f != LAYER_FAMILY_ATTENTION]
        reversed_deal = (
            tuple(sorted(lin_ids + attn_ids[-4:])),
            tuple(attn_ids[:6]),
            tuple(attn_ids[6:12]),
        )
        a = crossing_price(in_order, N_LAYERS, PAIR_MS)
        b = crossing_price(reversed_deal, N_LAYERS, PAIR_MS)
        self.assertLess(a.crossings, b.crossings)

    def test_a_contiguous_map_expressed_as_a_set_is_not_gapped(self):
        self.assertFalse(is_gapped(contiguous_layer_sets((32, 18, 14))))
        self.assertEqual(layer_set_flag(contiguous_layer_sets((2, 2))), "0-1;2-3")

    def test_attn_counts_of_agrees_with_attention_counts_on_contiguous_maps(self):
        for counts in ((32, 18, 14), (42, 11, 11), (48, 8, 8)):
            self.assertEqual(
                attn_counts_of(FAMILIES, contiguous_layer_sets(counts)),
                attention_counts(FAMILIES, counts),
            )


class TheCrossingTerm(unittest.TestCase):
    """Priced from the measured map, monotone in crossings, never invented."""

    def test_a_contiguous_cut_crosses_pp_size_minus_one_times(self):
        price = crossing_price(contiguous_layer_sets((32, 18, 14)), N_LAYERS, PAIR_MS)
        self.assertEqual(price.crossings, 2)

    def test_the_user_map_crosses_31_times(self):
        price = crossing_price(
            gapped_layer_sets(FAMILIES, 0, (0, 8, 8)), N_LAYERS, PAIR_MS
        )
        # 16 attention layers, the last of them terminal, so 31 and not 32.
        self.assertEqual(price.crossings, 31)

    def test_the_cost_is_monotone_in_the_number_of_crossings(self):
        prices = []
        for split in ((14, 1, 1), (8, 4, 4), (4, 6, 6), (0, 8, 8)):
            prices.append(
                crossing_price(gapped_layer_sets(FAMILIES, 0, split), N_LAYERS, PAIR_MS)
            )
        for a, b in zip(prices, prices[1:]):
            self.assertLess(a.crossings, b.crossings)
            self.assertLess(a.ms, b.ms)

    def test_an_unmeasured_pair_is_refused_and_never_defaulted(self):
        partial = {(0, 1): 8.0, (1, 0): 8.0}
        with self.assertRaises(UnpricedCrossing) as ctx:
            crossing_price(gapped_layer_sets(FAMILIES, 0, (0, 8, 8)), N_LAYERS, partial)
        self.assertIn("no measured link price", str(ctx.exception))

    def test_the_solver_reports_an_unpriceable_candidate_instead_of_ranking_it(self):
        decision = solve_launch_cut(
            layer_families=FAMILIES,
            incumbent_layers=INCUMBENT,
            measured_ms_per_layer=MS_PER_LAYER,
            measured_provenance="synthetic",
            card_names=CARDS,
            pool_model=pool_model(),
            cap_tokens=100_000,
            family_cost=family_cost(),
            design_prefix_tokens=4096,
            # Covers exactly the two FORWARD pairs a contiguous cut uses, and
            # none of the return pairs a gapped map needs.
            per_pair_crossing_ms={(0, 1): 8.0, (1, 2): 8.0},
        )
        self.assertTrue(decision.unpriced)
        self.assertTrue(all(c.kind == "contiguous" for c in decision.ranked))
        self.assertTrue(
            any("UNPRICED" in line for line in decision.table_lines()),
        )


class TheDepthAxis(unittest.TestCase):
    """Attention grows with the prefix, GDN does not -- and that moves things."""

    def test_the_gdn_cost_does_not_move_with_depth(self):
        cost = family_cost()
        shallow = cost.stage_ms((48, 8, 8), (0, 8, 8), 4096)
        deep = cost.stage_ms((48, 8, 8), (0, 8, 8), 262144)
        # Stage 0 holds 48 GDN layers and no attention: identical at any depth.
        self.assertAlmostEqual(shallow[0], deep[0], places=9)
        self.assertGreater(deep[1], shallow[1] * 10)

    def test_the_factor_is_finite_at_zero_prefix(self):
        # A bare d/d_ref would price the FIRST chunk of every prompt at zero.
        cost = family_cost()
        self.assertGreater(cost.depth_factor(0.0), 0.0)
        self.assertAlmostEqual(cost.depth_factor(4096.0), 1.0, places=9)

    def test_depth_moves_the_chosen_cut_towards_the_fast_card(self):
        kwargs = dict(
            layer_families=FAMILIES,
            incumbent_layers=INCUMBENT,
            measured_ms_per_layer=MS_PER_LAYER,
            measured_provenance="synthetic",
            card_names=CARDS,
            pool_model=pool_model(),
            cap_tokens=100_000,
            family_cost=family_cost(),
            per_pair_crossing_ms=PAIR_MS,
        )
        shallow = solve_launch_cut(design_prefix_tokens=4096, **kwargs).chosen
        deep = solve_launch_cut(design_prefix_tokens=262144, **kwargs).chosen
        self.assertNotEqual((shallow.layers, shallow.attn), (deep.layers, deep.attn))
        # Deep prefixes make an attention layer expensive, so the optimum puts
        # MORE of them on the fast card.
        self.assertGreater(deep.attn[0], shallow.attn[0])
        self.assertEqual(shallow.depth_tokens, 4096)
        self.assertEqual(deep.depth_tokens, 262144)

    def test_every_ranked_row_states_its_design_depth(self):
        decision = solve_launch_cut(
            layer_families=FAMILIES,
            incumbent_layers=INCUMBENT,
            measured_ms_per_layer=MS_PER_LAYER,
            measured_provenance="synthetic",
            card_names=CARDS,
            pool_model=pool_model(),
            cap_tokens=100_000,
            family_cost=family_cost(),
            design_prefix_tokens=9999,
            per_pair_crossing_ms=PAIR_MS,
        )
        self.assertTrue(all(c.depth_tokens == 9999 for c in decision.ranked))
        for line in decision.table_lines():
            if " layers=" in line:
                self.assertIn("depth_tokens=9999", line)


class TheFamilySplitIsSolvedNotFitted(unittest.TestCase):
    """Two depths close what one measurement cannot."""

    def test_both_inputs_are_reproduced_by_construction(self):
        cost, prov = family_costs_from_measurement(
            measured_ms_per_layer=(8.10, 35.16, 33.59),
            measured_counts=(32, 18, 14),
            measured_attn_counts=(8, 4, 4),
            chunk_tokens=4096,
            ref_prefix_tokens=4096.0,
            anchor_stage=1,
            anchor_attn_ms_per_layer=400.0,
            anchor_prefix_tokens=262144.0,
        )
        # The calibration point: every stage's total at the reference depth is
        # its measurement.
        for r, (n, ms) in enumerate(zip((32, 18, 14), (8.10, 35.16, 33.59))):
            self.assertAlmostEqual(
                cost.stage_ms((32, 18, 14), (8, 4, 4), 4096.0)[r], ms * n, places=6
            )
        # The anchor: one attention layer on stage 1 at 262,144 costs 400 ms.
        deep = cost.attn_ms_per_layer_at_ref[1] * cost.depth_factor(262144.0)
        self.assertAlmostEqual(deep, 400.0, places=6)
        self.assertIn("attn/linear ratio", prov)

    def test_an_inconsistent_anchor_is_refused_not_clamped(self):
        with self.assertRaises(ValueError) as ctx:
            family_costs_from_measurement(
                measured_ms_per_layer=(8.10, 35.16, 33.59),
                measured_counts=(32, 18, 14),
                measured_attn_counts=(8, 4, 4),
                chunk_tokens=4096,
                ref_prefix_tokens=4096.0,
                anchor_stage=1,
                # Absurdly large: the four attention layers alone would exceed
                # the whole measured chunk scaled to that depth.
                anchor_attn_ms_per_layer=100_000.0,
                anchor_prefix_tokens=262144.0,
            )
        self.assertIn("cannot both be true", str(ctx.exception))

    def test_an_anchor_stage_without_both_families_is_refused(self):
        with self.assertRaises(ValueError):
            family_costs_from_measurement(
                measured_ms_per_layer=(8.0, 32.0),
                measured_counts=(48, 16),
                measured_attn_counts=(0, 16),
                chunk_tokens=4096,
                ref_prefix_tokens=4096.0,
                anchor_stage=1,
                anchor_attn_ms_per_layer=400.0,
                anchor_prefix_tokens=262144.0,
            )


class TheGappedPool(unittest.TestCase):
    """KV lives where attention runs -- including where it does not."""

    def test_a_stage_with_no_attention_layer_is_skipped_not_refused(self):
        # The contiguous rule refuses it, and stays right to do so.
        with self.assertRaises(ValueError):
            pp_phase_pool((48, 8, 8), (0, 8, 8), pool_model())
        pool = gapped_phase_pool((48, 8, 8), (0, 8, 8), pool_model())
        self.assertGreater(pool, 0.0)

    def test_the_pool_is_the_min_over_the_stages_that_hold_kv(self):
        model = pool_model()
        # Stage 2 has the least free memory, so it binds.
        expected = (16500.0 - 8 * 360.0 - 1200.0) / (8 * 2048.0 / (1024.0 * 1024.0))
        self.assertAlmostEqual(
            gapped_phase_pool((48, 8, 8), (0, 8, 8), model), expected, places=6
        )

    def test_moving_attention_off_the_slow_cards_raises_the_pool(self):
        model = pool_model()
        wide = gapped_phase_pool((48, 8, 8), (0, 8, 8), model)
        narrow = gapped_phase_pool((52, 6, 6), (4, 6, 6), model)
        self.assertGreater(narrow, wide)

    def test_a_map_with_no_attention_anywhere_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            gapped_phase_pool((48, 8, 8), (0, 0, 0), pool_model())
        self.assertIn("no KV pool at all", str(ctx.exception))


class TheOneCostModel(unittest.TestCase):
    """Both families ranked by the same function, on the same objective."""

    def decision(self, cap=100_000, **over):
        kwargs = dict(
            layer_families=FAMILIES,
            incumbent_layers=INCUMBENT,
            measured_ms_per_layer=MS_PER_LAYER,
            measured_provenance="synthetic",
            card_names=CARDS,
            pool_model=pool_model(),
            cap_tokens=cap,
            family_cost=family_cost(),
            design_prefix_tokens=4096,
            per_pair_crossing_ms=PAIR_MS,
        )
        kwargs.update(over)
        return solve_launch_cut(**kwargs)

    def test_both_kinds_are_ranked_in_one_list(self):
        decision = self.decision()
        kinds = {c.kind for c in decision.ranked}
        self.assertEqual(kinds, {"contiguous", "gapped"})

    def test_a_gapped_candidate_is_bounded_by_the_sum_and_a_contiguous_by_the_max(self):
        decision = self.decision()
        cost = family_cost()
        for cand in decision.ranked:
            stage = cost.stage_ms(cand.layers, cand.attn, 4096)
            want = sum(stage) if cand.kind == "gapped" else max(stage)
            self.assertAlmostEqual(cand.makespan_ms, want, places=6)

    def test_the_objective_is_makespan_plus_crossings(self):
        decision = self.decision()
        best = min(
            (c for c in decision.ranked if c.pool_tokens >= 100_000),
            key=lambda c: (c.total_ms, -c.pool_tokens),
        )
        self.assertEqual(decision.chosen.layers, best.layers)
        self.assertAlmostEqual(
            decision.chosen.total_ms,
            decision.chosen.makespan_ms + decision.chosen.crossing_ms,
            places=9,
        )

    def test_every_candidate_carries_a_crossing_price(self):
        decision = self.decision()
        self.assertTrue(all(c.crossings >= 2 for c in decision.ranked))
        self.assertTrue(all(c.crossing_ms > 0.0 for c in decision.ranked))

    def test_the_table_always_shows_a_row_of_every_kind(self):
        decision = self.decision()
        lines = decision.table_lines(top=3)
        self.assertTrue(any(" gapped layers=" in ln for ln in lines))
        self.assertTrue(any(" contiguous layers=" in ln for ln in lines))
        self.assertEqual(sum(1 for ln in lines if "CHOSEN" in ln), 1)
        self.assertTrue(any("further candidates" in ln for ln in lines))
        # And the per-kind row is what carries it, not a coincidence of the
        # kv-floor row: a family swept out of the top-N must still get a row
        # marked with its TRUE rank, or an absent row reads as an absent
        # candidate -- the denominator trap applied to a whole family.
        missing = {c.kind for c in decision.ranked} - {
            c.kind for c in decision.ranked[:3]
        }
        for kind in missing:
            self.assertTrue(
                any(("best-%s#" % kind) in ln for ln in lines),
                "no best-%s row in %s" % (kind, lines),
            )
        self.assertTrue(missing, "the fixture no longer exercises the truncation")

    def test_the_row_format_names_every_field(self):
        line = self.decision().table_lines()[0]
        for field in (
            "PP-CUT solver:",
            "layers=",
            "attn=",
            "makespan_ms=",
            "crossing_ms=",
            "pool_tokens=",
            "depth_tokens=",
        ):
            self.assertIn(field, line)

    def test_a_pinned_layer_set_wins_but_is_priced_on_the_same_axes(self):
        owned = gapped_layer_sets(FAMILIES, 0, (0, 8, 8))
        decision = self.decision(pinned_layer_set=layer_set_flag(owned))
        self.assertTrue(decision.pinned)
        self.assertEqual(decision.chosen.kind, "gapped")
        self.assertEqual(decision.chosen.layers, (48, 8, 8))
        self.assertEqual(decision.chosen.attn, (0, 8, 8))
        self.assertEqual(decision.chosen.crossings, 31)
        self.assertGreater(decision.chosen.pool_tokens, 0.0)
        # A pin can rank far outside the top-N; its row is printed anyway,
        # with its true rank, or the table would say "alt" on every line
        # while the provenance line named something else.
        lines = decision.table_lines()
        chosen_rows = [ln for ln in lines if "CHOSEN" in ln]
        self.assertEqual(len(chosen_rows), 1)
        self.assertIn("gapped layers=48,8,8", chosen_rows[0])

    def test_an_unpriceable_pinned_layer_set_is_refused(self):
        owned = gapped_layer_sets(FAMILIES, 1, (1, 0, 15))
        with self.assertRaises(PPCutRefused):
            self.decision(pinned_layer_set=layer_set_flag(owned))

    def test_without_a_family_cost_nothing_changes(self):
        # The #1236 behaviour, byte for byte: contiguous only, no crossings.
        decision = solve_launch_cut(
            layer_families=FAMILIES,
            incumbent_layers=INCUMBENT,
            measured_ms_per_layer=MS_PER_LAYER,
            measured_provenance="synthetic",
            card_names=CARDS,
            pool_model=pool_model(),
            cap_tokens=100_000,
        )
        self.assertTrue(all(c.kind == "contiguous" for c in decision.ranked))
        self.assertTrue(all(c.crossing_ms == 0.0 for c in decision.ranked))


if __name__ == "__main__":
    unittest.main()
