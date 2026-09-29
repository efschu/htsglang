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
        self.assertIn("spec_p.env.update(ns._dual_mps_env)\n    spec_p.env.update(dual_share_env(ns, \"P\"))\n"
                      "    launch_group(spec_p", src)
        self.assertIn('spec_d.env.update(getattr(ns, "_dual_mps_env", None) or {})\n        launch_group(spec_d',
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
        blk = src[i:i + 1400]
        self.assertLess(blk.index("launch_group(spec_p"), blk.index("launch_group(_sd"))
        self.assertLess(blk.index("launch_group(_sd"), blk.index("if dry:"))
        self.assertIn('spec_d = getattr(ns, "_dual_spec_d", None)', src)
