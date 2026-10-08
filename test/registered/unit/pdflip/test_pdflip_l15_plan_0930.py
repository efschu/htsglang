# SPDX-License-Identifier: Apache-2.0
"""L1.5 planner post (L15-PLAN-0930 2.4, AP L15-01): the PURE decision layer.

The post ``l15`` carves the P layout's awake headroom out of the P budget so
D's sleep residue can hold it there (L1.5 = the brach VRAM of the 3080s).
This is the pure half of L15-01: the master/handover switch parsing, the
``FLLIPER_PDFLIP_L15_MIB`` override grammar, the post derivation from the
``P_AWAKE_PEAK_MIB`` record, the 27B-only refusal (user correction
2026-10-02: no L1.5 on NF, not even as an option), and the dual-layout refusal -- the dual layout
never sleeps P, so L1.5 there is refused by name (W-L15-DUAL), never silently
off.  The wiring into ``budgets_from_dc`` / ``vram_plan_view`` is the separate
L15-01b; nothing here imports the launcher.

Cases (a)-(f) follow the plan's acceptance list, plus the parse errors and the
boot-line format.  Red on the base: ``l15_plan`` does not exist yet.
"""

import os

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import l15_plan as LP  # noqa: E402

#: the w109290020 P budgets (front.log, verbatim terms minus the printed line)
#: and stand-in awake peaks; the 5090 first because order_cards sorts it there
BUDGETS = [15896, 17384, 17160]
PEAKS = [11666, 15000, 15500]
MASTER = {"FLLIPER_PDFLIP_L15": "1"}


class SwitchesAreOffByDefault(CustomTestCase):
    def test_master_and_handover_default_off(self):
        self.assertFalse(LP.master_on({}))
        self.assertFalse(LP.master_on({"FLLIPER_PDFLIP_L15": "0"}))
        self.assertFalse(LP.handover_on({}))
        self.assertFalse(LP.handover_on({"FLLIPER_PDFLIP_HOT_HANDOVER": ""}))

    def test_the_three_spellings_arm(self):
        for v in ("1", "true", "on"):
            self.assertTrue(LP.master_on({"FLLIPER_PDFLIP_L15": v}))
            self.assertTrue(LP.handover_on({"FLLIPER_PDFLIP_HOT_HANDOVER": v}))

    def test_case_and_blanks_are_tolerated(self):
        self.assertTrue(LP.master_on({"FLLIPER_PDFLIP_L15": " ON "}))
        self.assertTrue(LP.master_on({"FLLIPER_PDFLIP_L15": "True"}))


class MasterOffIsZeroOff(CustomTestCase):
    #: (a) the plan's byte-for-byte case: master off means the planner moves nothing
    def test_every_post_zero_off(self):
        posts = LP.resolve_posts("qwen27b", BUDGETS, PEAKS, {})
        self.assertEqual([p.card for p in posts], [0, 1, 2])
        self.assertEqual([p.mib for p in posts], [0, 0, 0])
        self.assertEqual([p.src for p in posts], ["OFF", "OFF", "OFF"])

    def test_master_off_beats_an_override(self):
        env = {"FLLIPER_PDFLIP_L15": "0", "FLLIPER_PDFLIP_L15_MIB": "c0=4096"}
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
        env = dict(MASTER, FLLIPER_PDFLIP_L15_MIB="c0=4096,c2=1024")
        posts = LP.resolve_posts("qwen27b", BUDGETS, PEAKS, env)
        self.assertEqual([(p.mib, p.src) for p in posts],
                         [(4096, "OVERRIDE"), (0, "OVERRIDE-UNNAMED"), (1024, "OVERRIDE")])


class L15Is27BOnly(CustomTestCase):
    #: user correction 2026-10-02: L1.5 is a 27B-only feature; NF has no variant
    def test_27b_auto_falls_through_to_the_record(self):
        posts = LP.resolve_posts("qwen27b", BUDGETS, PEAKS, MASTER)
        self.assertEqual([p.src for p in posts], ["RECORD(P_AWAKE_PEAK_MIB)"] * 3)
        self.assertEqual([p.mib for p in posts], [4230, 2384, 1660])

    def test_no_posts_for_another_line(self):
        for line in ("nextflash", "gemma"):
            with self.assertRaises(ValueError):
                LP.resolve_posts(line, BUDGETS, PEAKS, MASTER)
        self.assertFalse(hasattr(LP, "nf_expert_trade"))
        self.assertFalse(hasattr(LP, "LINE_NEXTFLASH"))

    def test_nf_with_master_or_handover_is_refused_by_name(self):
        for env, name in ((MASTER, "FLLIPER_PDFLIP_L15"),
                          ({"FLLIPER_PDFLIP_HOT_HANDOVER": "1"}, "FLLIPER_PDFLIP_HOT_HANDOVER")):
            msg = LP.refuse_not_27b("nextflash", env)
            self.assertIsNotNone(msg)
            self.assertTrue(msg.startswith(LP.NOT27B_REFUSAL_CODE), msg)
            self.assertIn(name, msg)

    def test_27b_or_switches_off_pass(self):
        self.assertIsNone(LP.refuse_not_27b("qwen27b", MASTER))
        self.assertIsNone(LP.refuse_not_27b("nextflash", {}))
        self.assertIsNone(LP.refuse_not_27b("nextflash", {"FLLIPER_PDFLIP_L15": "0"}))

    def test_launcher_refuses_before_any_post(self):
        import inspect

        from flliper.srt.pdflip import launcher

        src = inspect.getsource(launcher)
        i = src.index("l15_plan.refuse_not_27b(ns.profile, os.environ)")
        self.assertLess(i, src.index("l15_posts = l15_plan.resolve_posts("))
        self.assertIn("raise PdFlipLaunchRefused(_not27b)", src[i:i + 200])
        self.assertNotIn("PROFILE_NEXTFLASH):", src[i:i + 600])


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
        self.assertIn("FLLIPER_PDFLIP_L15", msg)

    def test_dual_with_handover_is_refused_too(self):
        msg = LP.refuse_dual(["--dual-layout"], {"FLLIPER_PDFLIP_HOT_HANDOVER": "1"})
        self.assertIsNotNone(msg)
        self.assertTrue(msg.startswith("W-L15-DUAL"), msg)
        self.assertIn("FLLIPER_PDFLIP_HOT_HANDOVER", msg)


class BootLine(CustomTestCase):
    def test_post_line_format(self):
        self.assertEqual(
            LP.post_line(LP.L15Post(card=1, mib=2384, src="RECORD(P_AWAKE_PEAK_MIB)")),
            "L15-POST card=1 mib=2384 src=RECORD(P_AWAKE_PEAK_MIB)")
        self.assertEqual(
            LP.post_line(LP.L15Post(card=0, mib=10000, src="OVERRIDE")),
            "L15-POST card=0 mib=10000 src=OVERRIDE")


if __name__ == "__main__":
    import unittest

    unittest.main()
