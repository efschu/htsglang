# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 (F26): the launcher's --dual-layout / --dual-mps.

DANGER DIRECTIONS guarded here:
* off is byte-identical: no --dual-layout in the front argv, flip-weights and
  early-D untouched, no MPS env;
* on forces --flip-weights resident (a sleeping weight family would free the
  bytes the co-resident plan counts on) and turns the early D start off (D is
  sized from P's AWAKE footprint, measured after P's READY);
* combinations the dual layout cannot serve are refused by name: d-adopt on,
  idle-layout pp, --dual-mps without --dual-layout;
* the P sleep leg is skipped under dual and no dormant image is recorded (an
  awake footprint in the dormant record would poison the next boot's ledger).
"""
from __future__ import annotations

import inspect
import unittest.mock as mock
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as L
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

X = 4096


def _ns(*extra):
    return L.build_parser().parse_args(["--tree", "/x", "--tag", "t", *extra])


def _front_argv(ns):
    return L.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 8, 8, X, X, "D")


class DualLayoutLauncher(CustomTestCase):
    def test_off_is_untouched(self):
        ns = _ns()
        L.resolve_dual_layout(ns)
        self.assertEqual(ns.flip_weights, "family")
        self.assertNotIn("--dual-layout", _front_argv(ns))
        self.assertEqual(L.start_dual_mps(ns, lambda *_: None, dry=True), {})

    def test_on_forces_resident_and_reaches_the_front(self):
        ns = _ns("--dual-layout")
        L.resolve_dual_layout(ns)
        self.assertEqual(ns.flip_weights, "resident")
        argv = _front_argv(ns)
        self.assertIn("--dual-layout", argv)
        self.assertIn("--weights-resident", argv)
        self.assertFalse(L._d_early_start_armed(_ns("--dual-layout", "--weg2-d-early-start", "on")))

    def test_refusals(self):
        for extra in (("--dual-layout", "--weg2-d-adopt", "on"),
                      ("--dual-mps", "on")):
            with self.assertRaises(L.Weg2DualLayoutRefused, msg=str(extra)):
                L.resolve_dual_layout(_ns(*extra))

    def test_idle_layout_pp_is_forced_to_tp(self):
        ns = _ns("--dual-layout", "--idle-layout", "pp")
        L.resolve_dual_layout(ns)
        self.assertEqual(ns.idle_layout, "tp")
        ns = _ns("--idle-layout", "pp")
        L.resolve_dual_layout(ns)
        self.assertEqual(ns.idle_layout, "pp")

    def test_mps_env_is_private_to_the_boot(self):
        ns = _ns("--dual-layout", "--dual-mps", "on")
        with mock.patch.dict(os.environ, {L.DUAL_MPS_OPT_IN_ENV: "1"}):
            L.resolve_dual_layout(ns)
        lines = []
        env = L.start_dual_mps(ns, lines.append, dry=True)
        self.assertEqual(env["CUDA_MPS_PIPE_DIRECTORY"], "/tmp/weg2-dual-mps-t/pipe")
        self.assertTrue(env["CUDA_MPS_LOG_DIRECTORY"].startswith("/tmp/weg2-dual-mps-t/"))
        self.assertTrue(any("WEG2-DUAL MPS" in ln for ln in lines))

    def test_main_skips_the_p_sleep_and_the_dormant_record_under_dual(self):
        src = inspect.getsource(L.main)
        i = src.index('if getattr(ns, "dual_layout", False):\n        # DUAL-TP3PP3: P stays awake.')
        blk = src[i:i + 1600]
        self.assertIn("state.sleep_p_ms = 0.0", blk)
        self.assertIn("state.sleep_p_ms = sleep_group(PORT_P", blk)
        self.assertIn('image_rec = None if getattr(ns, "dual_layout", False)', src)
        self.assertIn("if image_rec is not None:\n        log(host_ledger.format_dormant_image(image_rec))\n"
                      "        host_ledger.append_measured_record(", src)
        # MPS env reaches BOTH groups, before each launch.
        self.assertIn("spec_p.env.update(ns._dual_mps_env)\n    spec_p.env.update(dual_p_sm_env(ns))\n"
                      "    spec_p.env.update(dual_duty_env(ns))\n"
                      "    spec_p.env.update(dual_share_env(ns, \"P\"))\n"
                      "    spec_p.env.update(dual_priority_env(ns, \"P\", log))  # DUAL-SHARE: {} unless a switch "
                      "is on\n    launch_group(spec_p", src)
        self.assertIn('spec_d.env.update(getattr(ns, "_dual_mps_env", None) or {})\n'
                      '        spec_d.env.update(dual_priority_env(ns, "D", log))'
                      '  # DUAL-SHARE: {} unless a switch is on\n        launch_group(spec_d',
                      src)

    def test_dual_share_implies_dual_and_splits_the_union_roles(self):
        ns = _ns("--dual-share", "--tag", "x" * 90)
        L.resolve_dual_layout(ns)
        self.assertTrue(ns.dual_layout)
        self.assertEqual(ns.flip_weights, "resident")
        p, d = L.dual_share_env(ns, "P"), L.dual_share_env(ns, "D")
        self.assertEqual((p["SGLANG_WEG2_UNION_MODE"], d["SGLANG_WEG2_UNION_MODE"]), ("bind", "own"))
        self.assertEqual(p["SGLANG_WEG2_DUAL_SHARE"], "1")
        self.assertNotIn("SGLANG_WEG2_DUAL_SHARE", d)
        self.assertEqual(p["SGLANG_WEG2_UNION_DIR"], d["SGLANG_WEG2_UNION_DIR"])
        # a 90-character tag still leaves room for the card socket (<= 107 bytes)
        self.assertLess(len(p["SGLANG_WEG2_UNION_DIR"]) + len("/draft/u-0123456789ab.sock"), 107)
        self.assertEqual(L.dual_share_env(_ns(), "P"), {})

    def test_planned_dc_takes_the_lowered_p_budget_plus_overhead(self):
        import types

        cards = [types.SimpleNamespace(uuid=u) for u in ("A", "B", "C")]
        dc = L.dual_share_planned_dc(cards, [28000, 17000, 17000],
                                     "--max-running-requests=2 --rank-gpu-memory-mib 10100,12600,12000", 1500)
        self.assertEqual(dc, {"A": 11600, "B": 14100, "C": 13500})
        dc = L.dual_share_planned_dc(cards, [28000, 17000, 17000], "", 0)
        self.assertEqual(dc, {"A": 28000, "B": 17000, "C": 17000})

    def test_main_launches_d_right_after_p_under_dual_share(self):
        src = inspect.getsource(L.main)
        i = src.index('spec_p.env.update(dual_share_env(ns, "P"))')
        blk = src[i:i + 2600]
        self.assertLess(blk.index("launch_group(spec_p"), blk.index("launch_group(_sd"))
        self.assertLess(blk.index("launch_group(_sd"), blk.index("if dry:"))
        self.assertIn('spec_d = getattr(ns, "_dual_spec_d", None)', src)

    def test_p_sm_pct_only_with_mps_and_below_100(self):
        ns = _ns("--dual-layout", "--dual-mps", "on", "--dual-p-sm-pct", "50")
        with mock.patch.dict(os.environ, {L.DUAL_MPS_OPT_IN_ENV: "1"}):
            L.resolve_dual_layout(ns)
        self.assertEqual(L.dual_p_sm_env(ns), {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50"})
        ns = _ns("--dual-layout", "--dual-p-sm-pct", "50")
        L.resolve_dual_layout(ns)
        self.assertEqual(L.dual_p_sm_env(ns), {})
        self.assertEqual(L.dual_p_sm_env(_ns("--dual-layout", "--dual-mps", "on")), {})
        with self.assertRaises(L.Weg2DualLayoutRefused):
            L.dual_p_sm_env(_ns("--dual-layout", "--dual-mps", "on", "--dual-p-sm-pct", "0"))
        src = inspect.getsource(L.main)
        self.assertIn("spec_p.env.update(dual_p_sm_env(ns))", src)

    def test_p_loads_only_after_d_is_ready(self):
        from sglang.srt.model_executor import dual_stage_hull as H

        self.assertEqual(L.DUAL_D_READY_FILE, H.D_READY_FILE)
        src = inspect.getsource(L.main)
        i = src.index("launch_group(_sd, tree, log, dry)")
        blk = src[i:i + 1400]
        self.assertLess(blk.index('wait_ready(PORT_D, _sd.pid'), blk.index("DUAL_D_READY_FILE"))
        self.assertLess(src.index("DUAL_D_READY_FILE), \"w\")"),
                        src.index('state.t_ready["P"] = wait_ready(PORT_P'))

    def test_p_cut_reaches_ds_image_filter(self):
        self.assertEqual(L.dual_p_cut_from_argv(["x", "--pp-stage-ratio", "49,8,7", "y"]), "49,8,7")
        self.assertEqual(L.dual_p_cut_from_argv(["--pp-stage-ratio=13,30,21"]), "13,30,21")
        self.assertEqual(L.dual_p_cut_from_argv(["--pp-size", "3"]), "")
        from sglang.srt.weg2.union_arena_bind import dual_share_include_for

        inc0 = dual_share_include_for(["49", "8", "7"], 0)
        inc2 = dual_share_include_for(["49", "8", "7"], 2)
        self.assertTrue(inc0("model.language_model.layers.48.mlp.down_proj.weight_packed"))
        self.assertFalse(inc0("model.language_model.layers.49.mlp.down_proj.weight_packed"))
        self.assertTrue(inc0("model.language_model.embed_tokens.weight"))
        self.assertFalse(inc0("lm_head.weight"))
        self.assertTrue(inc2("model.language_model.layers.63.self_attn.o_proj.weight"))
        self.assertFalse(inc2("model.language_model.layers.56.self_attn.o_proj.weight"))
        self.assertTrue(inc2("lm_head.weight"))
        self.assertTrue(inc2("model.language_model.norm.weight"))
        self.assertFalse(inc2("model.language_model.embed_tokens.weight"))

    def test_dual_p_bar1_window(self):
        # f9fch3: P's flip-default windows + D's overflow the 3080's BAR1 when P never sleeps.
        on = L.build_parser().parse_args(["--tree", "/x", "--tag", "t", "--dual-layout"])
        L.resolve_dual_layout(on)
        self.assertEqual(on.p_barlink_bar1_window_mib, L.DUAL_P_BARLINK_BAR1_WINDOW_MIB)
        pinned = L.build_parser().parse_args(["--tree", "/x", "--tag", "t", "--dual-layout",
                                              "--p-barlink-bar1-window-mib", "8,PP_0=32"])
        L.resolve_dual_layout(pinned)
        self.assertEqual(pinned.p_barlink_bar1_window_mib, "8,PP_0=32")
        off = L.build_parser().parse_args(["--tree", "/x", "--tag", "t"])
        L.resolve_dual_layout(off)
        self.assertEqual(off.p_barlink_bar1_window_mib, L.P_BARLINK_BAR1_WINDOW_MIB)
        # the arithmetic: RM 19 + P + D (16+32+40) + reserve 32 fits 256
        p = sum(int(x.split("=")[-1]) for x in L.DUAL_P_BARLINK_BAR1_WINDOW_MIB.split(","))
        self.assertLessEqual(19 + p + 88 + 32, 256)

    def test_dual_mps_needs_explicit_opt_in(self):
        # Repro v2 scjhru S1 (30.09.): MPS + extend-sized barlink collectives of both
        # groups wedged (P rc=124, D 31 rounds in 45 s); the same arm without MPS (S3)
        # was clean. Metal: kw6pft / ndktv4 died the same way.
        env = {k: v for k, v in os.environ.items() if k != L.DUAL_MPS_OPT_IN_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(L.Weg2DualLayoutRefused) as cm:
                L.resolve_dual_layout(_ns("--dual-layout", "--dual-mps", "on"))
            self.assertIn("scjhru", str(cm.exception))
            self.assertIn(L.DUAL_MPS_OPT_IN_ENV, str(cm.exception))
            ns = _ns("--dual-layout")  # the default (off) is untouched
            L.resolve_dual_layout(ns)
            self.assertEqual(ns.dual_mps, "off")
        with mock.patch.dict(os.environ, {L.DUAL_MPS_OPT_IN_ENV: "1"}):
            ns = _ns("--dual-layout", "--dual-mps", "on")
            L.resolve_dual_layout(ns)
            self.assertEqual(ns.dual_mps, "on")

