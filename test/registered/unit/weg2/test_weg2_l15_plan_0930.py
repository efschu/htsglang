# SPDX-License-Identifier: Apache-2.0
"""L1.5 planner post (L15-PLAN-0930 2.4, AP L15-01): the PURE decision layer.

The post ``l15`` carves the P layout's awake headroom out of the P budget so
D's sleep residue can hold it there (L1.5 = the brach VRAM of the 3080s).
This is the pure half of L15-01: the master/handover switch parsing, the
``SGLANG_WEG2_L15_MIB`` override grammar, the post derivation from the
``P_AWAKE_PEAK_MIB`` record, the NF expert trade (the NF law says free VRAM
belongs to the experts, so an NF hold is PAID with resident expert rows:
rows = floor(l15 / row_mib)), and the dual-layout refusal -- the dual layout
never sleeps P, so L1.5 there is refused by name (W-L15-DUAL), never silently
off.  The wiring into ``budgets_from_dc`` / ``vram_plan_view`` is the separate
L15-01b; nothing here imports the launcher.

Cases (a)-(f) follow the plan's acceptance list, plus the parse errors and the
boot-line format.  Red on the base: ``l15_plan`` does not exist yet.
"""

import os

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import l15_plan as LP  # noqa: E402

#: the w109290020 P budgets (front.log, verbatim terms minus the printed line)
#: and stand-in awake peaks; the 5090 first because order_cards sorts it there
BUDGETS = [15896, 17384, 17160]
PEAKS = [11666, 15000, 15500]
MASTER = {"SGLANG_WEG2_L15": "1"}


class SwitchesAreOffByDefault(CustomTestCase):
    def test_master_and_handover_default_off(self):
        self.assertFalse(LP.master_on({}))
        self.assertFalse(LP.master_on({"SGLANG_WEG2_L15": "0"}))
        self.assertFalse(LP.handover_on({}))
        self.assertFalse(LP.handover_on({"SGLANG_WEG2_HOT_HANDOVER": ""}))

    def test_the_three_spellings_arm(self):
        for v in ("1", "true", "on"):
            self.assertTrue(LP.master_on({"SGLANG_WEG2_L15": v}))
            self.assertTrue(LP.handover_on({"SGLANG_WEG2_HOT_HANDOVER": v}))

    def test_case_and_blanks_are_tolerated(self):
        self.assertTrue(LP.master_on({"SGLANG_WEG2_L15": " ON "}))
        self.assertTrue(LP.master_on({"SGLANG_WEG2_L15": "True"}))


class MasterOffIsZeroOff(CustomTestCase):
    #: (a) the plan's byte-for-byte case: master off means the planner moves nothing
    def test_every_post_zero_off(self):
        posts = LP.resolve_posts("qwen27b", BUDGETS, PEAKS, {})
        self.assertEqual([p.card for p in posts], [0, 1, 2])
        self.assertEqual([p.mib for p in posts], [0, 0, 0])
        self.assertEqual([p.src for p in posts], ["OFF", "OFF", "OFF"])
        self.assertEqual([p.experts_rows_traded for p in posts], [0, 0, 0])

    def test_master_off_beats_an_override(self):
        env = {"SGLANG_WEG2_L15": "0", "SGLANG_WEG2_L15_MIB": "c0=4096"}
        posts = LP.resolve_posts("qwen27b", BUDGETS, PEAKS, env)
        self.assertEqual([p.mib for p in posts], [0, 0, 0])
        self.assertEqual([p.src for p in posts], ["OFF", "OFF", "OFF"])


class PostFromTheRecord(CustomTestCase):
    #: (b) the post is the record's difference, never a reserve
    def test_budget_minus_peak(self):
        self.assertEqual(LP.l15_post_mib(15896, 11666), (4230, "RECORD(P_AWAKE_PEAK_MIB)"))

    def test_clamped_at_zero_not_negative(self):
        self.assertEqual(LP.l15_post_mib(15896, 16000), (0, "RECORD(P_AWAKE_PEAK_MIB)"))

    def test_missing_record_is_unmeasured_zero(self):
        self.assertEqual(LP.l15_post_mib(15896, None), (0, "UNMEASURED"))
        posts = LP.resolve_posts("qwen27b", [15896, 17384], [None, 15000], MASTER)
        self.assertEqual([(p.mib, p.src) for p in posts],
                         [(0, "UNMEASURED"), (2384, "RECORD(P_AWAKE_PEAK_MIB)")])


class OverrideGrammar(CustomTestCase):
    def test_auto_forms(self):
        for v in (None, "", "auto", " auto "):
            self.assertEqual(LP.parse_l15_mib(v), ("auto", {}))

    def test_per_card_pairs(self):
        self.assertEqual(LP.parse_l15_mib("c0=100,c2=3000"), ("override", {0: 100, 2: 3000}))

    def test_malformed_names_the_value(self):
        for bad in ("4096", "c0=abc", "c1", "cX=10", "c0=1,"):
            with self.assertRaises(ValueError) as ctx:
                LP.parse_l15_mib(bad)
            self.assertIn(bad, str(ctx.exception))


class OverrideIsThePost(CustomTestCase):
    #: (c) an operator override shows as OVERRIDE with the given MiB; a card the
    #: override does not name gets 0, named OVERRIDE-UNNAMED (not a hidden auto)
    def test_named_and_unnamed_cards(self):
        env = dict(MASTER, SGLANG_WEG2_L15_MIB="c0=4096,c2=1024")
        posts = LP.resolve_posts("qwen27b", BUDGETS, PEAKS, env)
        self.assertEqual([(p.mib, p.src) for p in posts],
                         [(4096, "OVERRIDE"), (0, "OVERRIDE-UNNAMED"), (1024, "OVERRIDE")])

    def test_override_beats_the_record_even_on_nf(self):
        env = dict(MASTER, SGLANG_WEG2_L15_MIB="c0=10000")
        posts = LP.resolve_posts("nextflash", [15896], [11666], env)
        self.assertEqual([(p.mib, p.src) for p in posts], [(10000, "OVERRIDE")])


class NfDefaultsToZero(CustomTestCase):
    #: (e) the NF law: free VRAM is the experts', so without an override nothing
    #: is held no matter what the record would say
    def test_nf_default_zero(self):
        posts = LP.resolve_posts("nextflash", BUDGETS, PEAKS, MASTER)
        self.assertEqual([p.mib for p in posts], [0, 0, 0])
        self.assertEqual([p.src for p in posts], ["NF-DEFAULT-0"] * 3)
        self.assertEqual([p.experts_rows_traded for p in posts], [0, 0, 0])

    def test_27b_auto_falls_through_to_the_record(self):
        posts = LP.resolve_posts("qwen27b", BUDGETS, PEAKS, MASTER)
        self.assertEqual([p.src for p in posts], ["RECORD(P_AWAKE_PEAK_MIB)"] * 3)
        self.assertEqual([p.mib for p in posts], [4230, 2384, 1660])


class NfTradePaysWithExperts(CustomTestCase):
    #: (f) the user addendum: an NF hold is shown with the expert rows it costs
    def test_nf_override_trades_rows(self):
        env = dict(MASTER, SGLANG_WEG2_L15_MIB="c0=10000")
        posts = LP.resolve_posts("nextflash", [15896], [11666], env, row_mib=[1200.0])
        self.assertEqual(posts[0].mib, 10000)
        self.assertEqual(posts[0].src, "OVERRIDE")
        self.assertEqual(posts[0].experts_rows_traded, 8)  # floor(10000 / 1200)

    def test_no_row_size_no_rows(self):
        env = dict(MASTER, SGLANG_WEG2_L15_MIB="c0=10000")
        posts = LP.resolve_posts("nextflash", [15896], [11666], env)
        self.assertEqual(posts[0].experts_rows_traded, 0)

    def test_rows_only_on_nf(self):
        env = dict(MASTER, SGLANG_WEG2_L15_MIB="c0=10000")
        posts = LP.resolve_posts("qwen27b", [15896], [11666], env, row_mib=[1200.0])
        self.assertEqual(posts[0].experts_rows_traded, 0)

    def test_trade_floor_and_guard(self):
        self.assertEqual(LP.nf_expert_trade(10000, 1200.0), 8)
        self.assertEqual(LP.nf_expert_trade(0, 1200.0), 0)
        self.assertEqual(LP.nf_expert_trade(1199, 1200.0), 0)
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                LP.nf_expert_trade(10000, bad)


class DualLayoutIsRefused(CustomTestCase):
    #: (d) the plan: on --dual-layout L1.5 and the hot handover are refused in
    #: V1 by name, never silently off
    def test_no_dual_no_refusal(self):
        self.assertIsNone(LP.refuse_dual(["--tag", "x"], {}))
        self.assertIsNone(LP.refuse_dual(["--tag", "x"], MASTER))

    def test_dual_without_l15_passes(self):
        self.assertIsNone(LP.refuse_dual(["--dual-layout"], {}))

    def test_dual_with_master_is_w_l15_dual(self):
        msg = LP.refuse_dual(["--dual-layout"], MASTER)
        self.assertIsNotNone(msg)
        self.assertTrue(msg.startswith("W-L15-DUAL"), msg)
        self.assertIn("--dual-layout", msg)
        self.assertIn("SGLANG_WEG2_L15", msg)

    def test_dual_with_handover_is_refused_too(self):
        msg = LP.refuse_dual(["--dual-layout"], {"SGLANG_WEG2_HOT_HANDOVER": "1"})
        self.assertIsNotNone(msg)
        self.assertTrue(msg.startswith("W-L15-DUAL"), msg)
        self.assertIn("SGLANG_WEG2_HOT_HANDOVER", msg)


class BootLine(CustomTestCase):
    def test_post_line_format(self):
        self.assertEqual(
            LP.post_line(LP.L15Post(card=1, mib=2384, src="RECORD(P_AWAKE_PEAK_MIB)",
                                    experts_rows_traded=0)),
            "L15-POST card=1 mib=2384 src=RECORD(P_AWAKE_PEAK_MIB) experts_rows_traded=0")
        self.assertEqual(
            LP.post_line(LP.L15Post(card=0, mib=10000, src="OVERRIDE", experts_rows_traded=8)),
            "L15-POST card=0 mib=10000 src=OVERRIDE experts_rows_traded=8")


if __name__ == "__main__":
    import unittest

    unittest.main()
