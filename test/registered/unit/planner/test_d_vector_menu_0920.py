# SPDX-License-Identifier: Apache-2.0
"""The D-VECTOR MENU (user order 2026-09-20): choose the decode layout AT THE
FLIP from the backlog, not from a boot flag.

Hermetic. No GPU, no model tree, no card library on disk: every number below
is either quoted from a boot line / the memory laws, or DERIVED here by the
same runtime function the boot uses (``partition_units`` over the measured
card-rate library). NO HAND VECTORS -- the first two tests prove that the two
menu positions reproduce the documented vectors from the measured rates alone.
"""

import math
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.distributed.utils import (  # noqa: E402
    cp_token_context_budget,
    partition_units,
)
from sglang.srt.planner.d_vector_menu import (  # noqa: E402
    TOKEN_GRAIN,
    DBacklog,
    DVectorMenuError,
    MenuCalibration,
    RankCensus,
    build_menu,
    derive_vector,
    require_rank_uniform,
    second_graph_set_trade,
    select_d_vector,
    switch_price,
)

# --- THE FIXTURE, EVERY NUMBER QUOTED -------------------------------------
#
# Rig (memory RANG-LINK-ZUORDNUNG + the hardware inventory): rank 0 = RTX 5090
# 32607 MiB on x8, rank 1 = RTX 3080 20480 MiB on x4, rank 2 = RTX 3080 20480
# MiB on x8. The x4 rank is the taktgeber of every BAR1 leg.
MIB = 1024 * 1024
TOTAL_MIB = (32607, 20480, 20480)
LINK_GBS = (14.4, 6.5, 13.3)  # memory: Rang0 x8 14,4 / Rang1 x4 6,5 / Rang2 x8 13,3

# Today's flip (briefing): Qwen3.8-27B, 27 GB INT8 gdncov, 64 layers,
# P = PP3 with the layer split 39/13/12.
WEIGHT_BYTES_TOTAL = 27_000_000_000
N_LAYERS = 64
P_LAYERS = (39, 13, 12)

# The measured card-rate library, the same artifact the #1241 operating-point
# solve reads by card NAME (quoted in test_weg2_d_operating_points_1241.py
# from /root/.cache/sglang/card_library.json, source "measured").
LIB_GEMM = {"5090": 203.42, "3080": 50.97}
LIB_MEMBW = {"5090": 1661.6, "3080": 716.2}
CARD_ORDER = ("5090", "3080", "3080")


def _position_vector(rates):
    """A weight vector from measured rates, through the runtime's own rounding.

    This is what makes the fixture hand-number-free: the menu's two positions
    are ``partition_units`` over the measured library, exactly as
    ``d_tp_ratio_decision`` derives them.
    """
    scaled = [max(1, int(round(rates[c] * 10))) for c in CARD_ORDER]
    return tuple(partition_units(N_LAYERS, scaled))


#: bs=1 rounds are BANDWIDTH-bound (the shard is streamed once per token).
DECODE_BS1 = _position_vector(LIB_MEMBW)
#: bs>1 rounds are COMPUTE-bound (the same shard is read once for all tokens).
DECODE_BS6 = _position_vector(LIB_GEMM)

# KV bytes per token across ALL ranks, from the pool's row schema: 64 layers x
# 2 (K,V) x the model's KV heads x head_dim at INT8. An INPUT, and every pool
# figure below is a function of it, so a wrong value shows up as a wrong pool
# rather than as a silently different decision. The SPLIT across ranks is NOT
# an input -- it follows the weight vector, which is the whole point.
KV_ROW_BYTES_TOTAL = 88_064

# memory VRAM-TRANSIENT-POSTEN 2026-09-20: prefill transients are booked as an
# explicit KV posten of 2240 MiB per rank, never as an implicit reserve.
TRANSIENT_MIB = 2240
# memory Reserve-Semantik: 5090 1800 MiB / 3080 1400 MiB of USER free space.
RESERVE_MIB = (1800, 1400, 1400)


def rig_census(**over):
    kw = dict(
        total_bytes=tuple((TOTAL_MIB[r] - RESERVE_MIB[r]) * MIB for r in range(3)),
        weight_bytes_total=WEIGHT_BYTES_TOTAL,
        p_layers=P_LAYERS,
        n_layers=N_LAYERS,
        kv_row_bytes_total=KV_ROW_BYTES_TOTAL,
        link_bytes_per_s=tuple(g * 1e9 for g in LINK_GBS),
        other_resident_bytes=tuple(TRANSIENT_MIB * MIB for _ in range(3)),
        graph_set_bytes=(600 * MIB, 400 * MIB, 400 * MIB),
        graph_capture_s=18.0,
    )
    kw.update(over)
    return RankCensus(**kw)


POSITIONS = {"decode-bs1": DECODE_BS1, "decode-bs6": DECODE_BS6}
PREFERENCE = {"deep": "decode-bs1", "wide": "decode-bs6"}


class TestPositionsAreDerived(unittest.TestCase):
    """The two menu positions come from the measured library, not from here."""

    def test_bs6_reproduces_the_documented_vector(self):
        # weg2/launcher.py's D_TP_OBJECTIVE_DEFAULT comment quotes the bs=6
        # vector as [42,11,11] from dec2c's own A/B.
        self.assertEqual(DECODE_BS6, (42, 11, 11))

    def test_bs1_tilts_toward_the_weak_cards_relative_to_bs6(self):
        # The 5090's measured advantage is 3.99x on GEMM but only 2.32x on
        # bandwidth, so the bandwidth-bound position gives it LESS.
        self.assertLess(DECODE_BS1[0], DECODE_BS6[0])
        self.assertEqual(sum(DECODE_BS1), N_LAYERS)
        self.assertEqual(sum(DECODE_BS6), N_LAYERS)

    def test_the_users_own_arithmetic_is_reproduced(self):
        """User 2026-09-20: '20 GB - 4,5 GB Shard - 5,5 GB P-Reste ~ 10 GB'."""
        v = derive_vector("decode-bs6", DECODE_BS6, rig_census())
        # The 3080's D shard: 11/64 of 27 GB = 4.64 GB -> the user's "4,5 GB".
        self.assertAlmostEqual(v.weight_bytes[1] / 1e9, 4.64, places=1)
        # The 5090's D shard: 42/64 of 27 GB = 17.7 GB -> the user's "~18 GB".
        self.assertAlmostEqual(v.weight_bytes[0] / 1e9, 17.72, places=1)
        # The 3080's P stage: 13/64 of 27 GB = 5.48 GB -> the user's "5,5 GB".
        self.assertAlmostEqual(P_LAYERS[1] / N_LAYERS * 27.0, 5.48, places=1)


class TestDerivation(unittest.TestCase):
    def test_pool_is_the_runtime_formula_on_the_derived_rows(self):
        v = derive_vector("decode-bs6", DECODE_BS6, rig_census())
        self.assertEqual(
            v.pool_tokens,
            cp_token_context_budget(list(v.token_ratio), list(v.kv_cap_rows)),
        )
        self.assertEqual(sum(v.token_ratio), TOKEN_GRAIN)

    def test_token_key_tilts_to_the_3080s_under_the_heavier_5090_vector(self):
        """The order's 'KV eher zu den 3080ern' is a CONSEQUENCE, not a knob.

        A heavier 5090 column share leaves the 5090 less free VRAM; the
        capacity-proportional key then hands the 3080s a larger token share.
        """
        bs1 = derive_vector("decode-bs1", DECODE_BS1, rig_census())
        bs6 = derive_vector("decode-bs6", DECODE_BS6, rig_census())
        share_5090_bs1 = bs1.token_ratio[0] / sum(bs1.token_ratio)
        share_5090_bs6 = bs6.token_ratio[0] / sum(bs6.token_ratio)
        self.assertLess(share_5090_bs6, share_5090_bs1)

    def test_the_compute_vector_funds_the_larger_gesamtpool(self):
        """MEASURED BY THIS FIXTURE, AGAINST THE OBVIOUS GUESS.

        One would expect the bandwidth-shaped vector -- which leaves the 5090
        smaller -- to fund the larger pool. It does not, and the reason is the
        second-order term: KV heads ride the SAME vector as the weights, so a
        heavier 5090 also has more EXPENSIVE rows, while the two 3080s gain
        both room and cheap rows. The 3080 side wins by more than the 5090
        side loses.

        This is why the deep shape must select on the pool it MEASURES rather
        than on the operating point its name suggests.
        """
        bs1 = derive_vector("decode-bs1", DECODE_BS1, rig_census())
        bs6 = derive_vector("decode-bs6", DECODE_BS6, rig_census())
        self.assertEqual(bs1.pool_tokens, 807_808)
        self.assertEqual(bs6.pool_tokens, 1_174_592)
        self.assertGreater(bs6.pool_tokens, bs1.pool_tokens)

    def test_kv_row_bytes_follow_the_weight_vector(self):
        bs1 = derive_vector("decode-bs1", DECODE_BS1, rig_census())
        bs6 = derive_vector("decode-bs6", DECODE_BS6, rig_census())
        self.assertGreater(bs6.kv_row_bytes[0], bs1.kv_row_bytes[0])
        self.assertLess(bs6.kv_row_bytes[1], bs1.kv_row_bytes[1])
        self.assertEqual(sum(bs6.kv_row_bytes), KV_ROW_BYTES_TOTAL)

    def test_replicated_kv_rows_do_not_follow_the_vector(self):
        c = rig_census(kv_replicated=True)
        v = derive_vector("decode-bs6", DECODE_BS6, c)
        self.assertEqual(set(v.kv_row_bytes), {KV_ROW_BYTES_TOTAL})

    def test_resident_is_the_intersection_of_p_stage_and_column_range(self):
        v = derive_vector("decode-bs6", DECODE_BS6, rig_census())
        for r in range(3):
            expect = (
                DECODE_BS6[r] / N_LAYERS * (P_LAYERS[r] / N_LAYERS) * WEIGHT_BYTES_TOTAL
            )
            self.assertAlmostEqual(v.resident_bytes[r] / expect, 1.0, places=6)

    def test_p_stage_resident_costs_vram(self):
        keep = derive_vector("decode-bs6", DECODE_BS6, rig_census())
        drop = derive_vector(
            "decode-bs6", DECODE_BS6, rig_census(p_stage_resident=False)
        )
        self.assertGreater(drop.pool_tokens, keep.pool_tokens)

    def test_impossible_vector_is_refused_with_its_terms(self):
        # The big column share on a 3080: 42/64 of 27 GB = 17.7 GB of D shard
        # plus its retained P stage on a 20 GB card that also owes transients.
        with self.assertRaises(DVectorMenuError) as cm:
            derive_vector("big-share-on-3080", (11, 42, 11), rig_census())
        self.assertIn("physical impossibility", str(cm.exception))

    def test_menu_refuses_two_names_for_one_layout(self):
        with self.assertRaises(DVectorMenuError) as cm:
            build_menu({"a": DECODE_BS6, "b": DECODE_BS6}, rig_census())
        self.assertIn("same weight vector", str(cm.exception))


class TestSwitchPrice(unittest.TestCase):
    def test_holding_the_vector_costs_no_weight_bytes(self):
        m = build_menu(POSITIONS, rig_census())
        bs6 = next(v for v in m if v.name == "decode-bs6")
        p = switch_price(bs6, bs6, rig_census())
        self.assertEqual(sum(p.weight_in_bytes), 0)
        self.assertEqual(p.weight_s, 0.0)
        self.assertEqual(p.graph_capture_s, 0.0)

    def test_the_x4_rank_is_the_taktgeber(self):
        """Rank 1 sits on x4, so it binds even with fewer bytes than rank 0."""
        m = build_menu(POSITIONS, rig_census())
        bs1 = next(v for v in m if v.name == "decode-bs1")
        bs6 = next(v for v in m if v.name == "decode-bs6")
        p = switch_price(bs1, bs6, rig_census())
        self.assertEqual(p.taktgeber_rank, 1)
        # Rank 1 gains FEWER columns than rank 0 but is discounted less by its
        # own small P stage, so the two arrive at nearly equal bytes -- and
        # then the x4 link decides. A model that summed bytes over a mean rate
        # would have named rank 0.
        self.assertAlmostEqual(
            p.weight_in_bytes[1] / p.weight_in_bytes[0], 1.0, places=1
        )
        self.assertGreater(
            p.weight_in_bytes[1] / LINK_GBS[1],
            p.weight_in_bytes[0] / LINK_GBS[0],
        )

    def test_switch_moves_only_the_uncovered_column_delta(self):
        m = build_menu(POSITIONS, rig_census())
        bs1 = next(v for v in m if v.name == "decode-bs1")
        bs6 = next(v for v in m if v.name == "decode-bs6")
        p = switch_price(bs1, bs6, rig_census())
        # Rank 2's column range is the same width under both vectors and sits
        # at the top of the checkpoint, so it gains nothing.
        self.assertEqual(p.weight_in_bytes[2], 0)
        # Rank 0 gains exactly the columns bs6 adds, discounted by the part of
        # them already lying in its own P stage.
        delta = (DECODE_BS6[0] - DECODE_BS1[0]) / N_LAYERS
        expect = delta * (1.0 - P_LAYERS[0] / N_LAYERS) * WEIGHT_BYTES_TOTAL
        self.assertAlmostEqual(p.weight_in_bytes[0] / expect, 1.0, places=5)

    def test_resident_kv_makes_a_token_key_change_expensive(self):
        m = build_menu(POSITIONS, rig_census())
        bs1 = next(v for v in m if v.name == "decode-bs1")
        bs6 = next(v for v in m if v.name == "decode-bs6")
        cold = switch_price(bs1, bs6, rig_census(), resident_kv_tokens=0)
        warm = switch_price(bs1, bs6, rig_census(), resident_kv_tokens=262_144)
        self.assertEqual(sum(cold.kv_in_bytes), 0)
        self.assertGreater(sum(warm.kv_in_bytes), 0)
        self.assertGreater(warm.total_s, cold.total_s)

    def test_recapture_is_owed_unless_the_set_is_resident(self):
        m = build_menu(POSITIONS, rig_census())
        bs1 = next(v for v in m if v.name == "decode-bs1")
        bs6 = next(v for v in m if v.name == "decode-bs6")
        owed = switch_price(bs1, bs6, rig_census())
        free = switch_price(bs1, bs6, rig_census(), graph_set_resident=True)
        self.assertEqual(owed.graph_capture_s, 18.0)
        self.assertEqual(free.graph_capture_s, 0.0)
        # The capture dominates the BAR1 legs, which is the finding that makes
        # a second resident graph set the lever worth having.
        self.assertGreater(owed.graph_capture_s, owed.weight_s + owed.kv_s)

    def test_the_capture_term_dominates_the_bar1_legs(self):
        """THE FINDING THAT DECIDES WHETHER A MENU IS WORTH HAVING."""
        m = build_menu(POSITIONS, rig_census())
        bs1 = next(v for v in m if v.name == "decode-bs1")
        bs6 = next(v for v in m if v.name == "decode-bs6")
        p = switch_price(bs1, bs6, rig_census(), resident_kv_tokens=262_144)
        self.assertLess(p.weight_s + p.kv_s, 0.3)
        self.assertEqual(p.graph_capture_s, 18.0)
        # Two orders of magnitude. The menu's cost is recapture, not bytes.
        self.assertGreater(p.graph_capture_s / (p.weight_s + p.kv_s), 50.0)

    def test_second_graph_set_is_a_pool_price_not_a_fit_check(self):
        c = rig_census()
        v = derive_vector("decode-bs6", DECODE_BS6, c)
        t = second_graph_set_trade(v, c)
        self.assertTrue(t.possible)
        self.assertEqual(t.pool_with_one, v.pool_tokens)
        self.assertLess(t.pool_with_two, t.pool_with_one)
        self.assertGreater(t.pool_cost_tokens, 0)
        self.assertEqual(t.recapture_saved_s, 18.0)

    def test_second_graph_set_needs_censused_sizes(self):
        c = rig_census(graph_set_bytes=())
        v = derive_vector("decode-bs6", DECODE_BS6, c)
        t = second_graph_set_trade(v, c)
        self.assertFalse(t.possible)
        self.assertIn("not a set that fits", t.reason)


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.census = rig_census()
        self.menu = build_menu(POSITIONS, self.census)
        self.bs1 = next(v for v in self.menu if v.name == "decode-bs1")
        self.bs6 = next(v for v in self.menu if v.name == "decode-bs6")
        self.calib = MenuCalibration(
            deep_concurrency_max=1,
            horizon_rounds=4000,
            mean_round_s=0.015,
            band_pct=1.5,
            margin=2.0,
            min_dwell_flips=2,
        )

    def test_one_deep_request_takes_the_widest_gesamtpool(self):
        """'bs1-tief': the order's rule is max GESAMTPOOL, and the menu obeys
        the MEASURED pool even when the preference map names the other entry.
        """
        deep = DBacklog(
            queued_reqs=1,
            queued_prompt_tokens=262_144,
            max_queued_prompt_tokens=262_144,
        )
        v = select_d_vector(
            deep,
            self.menu,
            self.census,
            self.calib,
            current=None,
            preference=PREFERENCE,
        )
        widest = max(self.menu, key=lambda x: x.pool_tokens)
        self.assertEqual(v.chosen.name, widest.name)
        self.assertEqual(v.chosen.pool_tokens, 1_174_592)
        self.assertTrue(v.switched)

    def test_many_shallow_requests_pick_the_wide_position(self):
        wide = DBacklog(
            queued_reqs=8,
            queued_prompt_tokens=8 * 4096,
            max_queued_prompt_tokens=4096,
        )
        v = select_d_vector(
            wide,
            self.menu,
            self.census,
            self.calib,
            current=None,
            preference=PREFERENCE,
        )
        self.assertEqual(v.chosen.name, "decode-bs6")

    def test_demand_is_the_gesamtpool_sum_not_the_deepest(self):
        wide = DBacklog(
            queued_reqs=8,
            queued_prompt_tokens=8 * 4096,
            max_queued_prompt_tokens=4096,
            held_tokens=1000,
        )
        self.assertEqual(wide.demand_tokens, 8 * 4096 + 1000)
        self.assertEqual(wide.deepest, 4096)

    def test_a_pool_that_cannot_hold_the_backlog_forces_a_mandatory_switch(self):
        """Capacity outranks hysteresis AND dwell."""
        deep = DBacklog(
            queued_reqs=1,
            queued_prompt_tokens=self.bs1.pool_tokens + 1,
            max_queued_prompt_tokens=self.bs1.pool_tokens + 1,
        )
        self.assertGreater(self.bs6.pool_tokens, self.bs1.pool_tokens)
        v = select_d_vector(
            deep,
            self.menu,
            self.census,
            self.calib,
            current="decode-bs1",
            preference=PREFERENCE,
            flips_on_current=0,  # dwell would otherwise refuse
        )
        self.assertTrue(v.mandatory)
        self.assertTrue(v.switched)
        self.assertEqual(v.chosen.name, "decode-bs6")
        self.assertIn("MANDATORY", v.reason)

    def test_dwell_refuses_a_move_that_has_not_been_held_long_enough(self):
        wide = DBacklog(
            queued_reqs=8,
            queued_prompt_tokens=8 * 4096,
            max_queued_prompt_tokens=4096,
        )
        v = select_d_vector(
            wide,
            self.menu,
            self.census,
            self.calib,
            current="decode-bs1",
            preference=PREFERENCE,
            gain_pct={"decode-bs6": 10.3},
            flips_on_current=1,
        )
        self.assertFalse(v.switched)
        self.assertEqual(v.chosen.name, "decode-bs1")
        self.assertIn("dwell", v.reason)

    def test_a_gain_inside_its_own_band_is_not_a_gain(self):
        wide = DBacklog(
            queued_reqs=8,
            queued_prompt_tokens=8 * 4096,
            max_queued_prompt_tokens=4096,
        )
        v = select_d_vector(
            wide,
            self.menu,
            self.census,
            self.calib,
            current="decode-bs1",
            preference=PREFERENCE,
            gain_pct={"decode-bs6": 2.9},  # < 2.0 x 1.5 % band
            flips_on_current=10,
        )
        self.assertFalse(v.switched)
        self.assertIn("A-vs-A band", v.reason)

    def test_a_measured_gain_that_pays_for_itself_switches(self):
        """dec2c measured the bs=6 vector at +10.3 %.

        THE HORIZON IS THE BINDING TERM, NOT THE PERCENTAGE. At 18 s of
        recapture, a +10.3 % gain on 15 ms rounds needs ~11 800 decode rounds
        (~3 minutes of continuous decode) before the switch has paid for
        itself. The 4000-round horizon of ``self.calib`` REFUSES this same
        gain, which is the test below.
        """
        wide = DBacklog(
            queued_reqs=8,
            queued_prompt_tokens=8 * 4096,
            max_queued_prompt_tokens=4096,
        )
        long_horizon = MenuCalibration(
            deep_concurrency_max=1,
            horizon_rounds=20_000,
            mean_round_s=0.015,
            band_pct=1.5,
            margin=2.0,
            min_dwell_flips=2,
        )
        self.calib = long_horizon
        v = select_d_vector(
            wide,
            self.menu,
            self.census,
            long_horizon,
            current="decode-bs1",
            preference=PREFERENCE,
            gain_pct={"decode-bs6": 10.3},
            flips_on_current=10,
        )
        self.assertTrue(v.switched)
        self.assertEqual(v.chosen.name, "decode-bs6")
        self.assertIsNotNone(v.price)
        self.assertIsNotNone(v.payback_rounds)
        # Payback = switch seconds / seconds saved per round.
        expect = math.ceil(v.price.total_s / (self.calib.mean_round_s * 10.3 / 100.0))
        self.assertEqual(v.payback_rounds, expect)

    def test_a_short_horizon_refuses_the_same_gain(self):
        """Hysteresis is the horizon, not the percentage."""
        wide = DBacklog(
            queued_reqs=8,
            queued_prompt_tokens=8 * 4096,
            max_queued_prompt_tokens=4096,
        )
        short = MenuCalibration(
            deep_concurrency_max=1,
            horizon_rounds=4_000,
            mean_round_s=0.015,
            band_pct=1.5,
            margin=2.0,
            min_dwell_flips=2,
        )
        v = select_d_vector(
            wide,
            self.menu,
            self.census,
            short,
            current="decode-bs1",
            preference=PREFERENCE,
            gain_pct={"decode-bs6": 10.3},
            flips_on_current=10,
        )
        self.assertFalse(v.switched)
        self.assertIn("does not pay for itself", v.reason)

    def test_resident_graph_set_makes_the_same_switch_affordable(self):
        """The recapture term is what a second resident graph set buys back."""
        wide = DBacklog(
            queued_reqs=8,
            queued_prompt_tokens=8 * 4096,
            max_queued_prompt_tokens=4096,
        )
        # 4000 rounds saves 6.18 s: not enough for an 18.2 s switch, but
        # ample for the 0.21 s of BAR1 that is left once the recapture is
        # gone. The second resident graph set is exactly this difference.
        short = MenuCalibration(
            deep_concurrency_max=1,
            horizon_rounds=4_000,
            mean_round_s=0.015,
            band_pct=1.5,
            margin=2.0,
            min_dwell_flips=2,
        )
        common = dict(
            current="decode-bs1",
            preference=PREFERENCE,
            gain_pct={"decode-bs6": 10.3},
            flips_on_current=10,
        )
        cold = select_d_vector(wide, self.menu, self.census, short, **common)
        warm = select_d_vector(
            wide,
            self.menu,
            self.census,
            short,
            resident_graph_sets=("decode-bs6",),
            **common,
        )
        self.assertFalse(cold.switched)
        self.assertTrue(warm.switched)

    def test_holding_is_free_when_the_shape_already_wants_the_installed_vector(self):
        deep = DBacklog(
            queued_reqs=1,
            queued_prompt_tokens=100_000,
            max_queued_prompt_tokens=100_000,
        )
        v = select_d_vector(
            deep,
            self.menu,
            self.census,
            self.calib,
            current="decode-bs6",
            preference=PREFERENCE,
            flips_on_current=10,
        )
        self.assertFalse(v.switched)
        self.assertIsNone(v.price)

    def test_no_entry_funds_the_backlog_takes_the_widest_pool(self):
        huge = DBacklog(
            queued_reqs=64,
            queued_prompt_tokens=64 * 262_144,
            max_queued_prompt_tokens=262_144,
        )
        v = select_d_vector(
            huge,
            self.menu,
            self.census,
            self.calib,
            current="decode-bs1",
            preference=PREFERENCE,
            flips_on_current=10,
        )
        self.assertEqual(v.chosen.name, "decode-bs6")
        self.assertIn("widest pool", v.reason)

    def test_a_current_vector_outside_the_menu_is_refused(self):
        with self.assertRaises(DVectorMenuError) as cm:
            select_d_vector(
                DBacklog(1, 100, 100),
                self.menu,
                self.census,
                self.calib,
                current="not-declared",
            )
        self.assertIn("declared ceiling set", str(cm.exception))


class TestRankUniformity(unittest.TestCase):
    def test_same_inputs_give_the_same_fingerprint_on_every_rank(self):
        census = rig_census()
        menu = build_menu(POSITIONS, census)
        calib = MenuCalibration(horizon_rounds=4000, mean_round_s=0.015)
        backlog = DBacklog(
            queued_reqs=1,
            queued_prompt_tokens=262_144,
            max_queued_prompt_tokens=262_144,
        )
        fps = [
            select_d_vector(
                backlog, menu, census, calib, preference=PREFERENCE
            ).fingerprint()
            for _ in range(3)
        ]
        self.assertEqual(require_rank_uniform(fps), fps[0])

    def test_divergence_is_a_crash_not_a_vote(self):
        with self.assertRaises(DVectorMenuError) as cm:
            require_rank_uniform(["aaaa", "aaaa", "bbbb"])
        self.assertIn("CRASH/STOP", str(cm.exception))

    def test_an_empty_reduction_is_not_agreement(self):
        with self.assertRaises(DVectorMenuError) as cm:
            require_rank_uniform([])
        self.assertIn("not agreement", str(cm.exception))

    def test_prose_differences_do_not_break_uniformity(self):
        """Two ranks may print a different reason without disagreeing."""
        census = rig_census()
        menu = build_menu(POSITIONS, census)
        calib = MenuCalibration(horizon_rounds=4000, mean_round_s=0.015)
        deep = DBacklog(1, 262_144, 262_144)
        a = select_d_vector(deep, menu, census, calib, preference=PREFERENCE)
        b = select_d_vector(deep, menu, census, calib, preference=PREFERENCE)
        object.__setattr__(b, "reason", "a differently worded reason")
        self.assertEqual(a.fingerprint(), b.fingerprint())


class TestBacklogGuards(unittest.TestCase):
    def test_depth_count_disagreement_is_refused(self):
        with self.assertRaises(DVectorMenuError) as cm:
            DBacklog(2, 100, 50, depths=(50, 50, 50))
        self.assertIn("different round", str(cm.exception))

    def test_negative_term_is_a_sensor_defect(self):
        with self.assertRaises(DVectorMenuError) as cm:
            DBacklog(-1, 0, 0)
        self.assertIn("sensor defect", str(cm.exception))

    def test_empty_queue_still_has_concurrency_one(self):
        self.assertEqual(DBacklog(0, 0, 0).concurrency, 1)

    def test_explicit_depths_outrank_the_aggregates(self):
        b = DBacklog(3, 999, 999, depths=(10, 20, 30))
        self.assertEqual(b.deepest, 30)
        self.assertEqual(b.demand_tokens, 60)


if __name__ == "__main__":
    unittest.main()
