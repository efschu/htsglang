"""--d-reshard wake-seg (27B, SB 26.09.): the SEGMENT objective of the D planner.

rc9meas 26.09. refuted the sum objective (D-SPEED balanced the per-rank compute
SUMS; 'wake' 707,191,190 made INT8 slower, NVFP4 98,19,19 gained to 128k and
tipped at 240k).  The round is sum over layer segments of the slowest rank:
round = max_r m_r + max_r x_r + floor (weg2/d_reshard.py "Segment model").

Pinned without a GPU:
  * THE MODEL REPRODUCES THE MEASUREMENT -- step ms of every arm/depth/bs
    (temperature mean) within a named band: calibrated arms h/r INT8 0.35 ms,
    NVFP4 0.6 ms; the out-of-sample rt arms (bandwidth placement) 0.8 / 0.95 ms;
  * THE SIGN -- INT8 'dec' slower than today at every depth, NVFP4 'dec' faster
    at 10k..128k, in model AND measurement; the old sum objective predicts the
    INT8 sign wrong; the NVFP4 240k-bs1 tip (+0.9 ms measured) is NOT
    reproduced (model ~0) -- pinned as a known miss, not hidden;
  * THE PLANNER -- robust vector within 1 % of every per-point optimum, the
    per-depth choice, bs extrapolation flagged, FP8 labelled borrowed;
  * DEFAULT UNCHANGED -- 'off' and 'wake' specs/digests byte-identical to the
    boots (96b0cd92ae412589 / 5b8ad0c5dc248b2e), no objective env, D-SPEED stays
    the sum table unless SGLANG_WEG2_DRESHARD_OBJECTIVE=segment;
  * GC INSTRUMENT -- off by default; warning callback and gc.freeze() only when
    asked.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import types
import unittest

import torch

from sglang.srt.distributed.utils import partition_units
from sglang.srt.weg2 import d_reshard as D
from sglang.srt.weg2 import gc_instrument as G
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

CTX = {"10k": 10240, "32k": 32768, "128k": 131072, "240k": 245760}
UNITS = {"i8h": (584, 252, 252), "i8r": (707, 191, 190), "i8rt": (707, 191, 190),
         "n4h": (73, 32, 31), "n4r": (98, 19, 19), "n4rt": (98, 19, 19)}
#: installed token vectors ("installed measured KV-token ownership vector"); rt = bandwidth placement
TOKV = {"i8h": (26, 19, 19), "i8r": (21, 22, 21), "n4h": (14, 9, 9), "n4r": (6, 5, 5)}
#: rc9meas 26.09. step ms (max over ranks gpu-ms, mean of kalt/warm; measure27b_eval_v2 points,
#: /spinning/gpu-arb/docker/measure27b_v2_retro_0926/ and /root/.claude/jobs/1ab4cd30/tmp/v2n4/)
MEAS = {
    ("i8h", "10k", 1): 31.48, ("i8h", "32k", 1): 31.98, ("i8h", "128k", 1): 34.89, ("i8h", "240k", 1): 38.50,
    ("i8h", "10k", 2): 36.36, ("i8h", "32k", 2): 37.55, ("i8h", "128k", 2): 43.57, ("i8h", "240k", 2): 49.76,
    ("i8r", "10k", 1): 31.67, ("i8r", "32k", 1): 32.39, ("i8r", "128k", 1): 35.50, ("i8r", "240k", 1): 39.23,
    ("i8r", "10k", 2): 36.56, ("i8r", "32k", 2): 37.79, ("i8r", "128k", 2): 44.10, ("i8r", "240k", 2): 50.95,
    ("i8rt", "10k", 1): 31.33, ("i8rt", "32k", 1): 31.98, ("i8rt", "128k", 1): 34.40, ("i8rt", "240k", 1): 37.69,
    ("i8rt", "10k", 2): 36.24, ("i8rt", "32k", 2): 37.66, ("i8rt", "128k", 2): 42.75, ("i8rt", "240k", 2): 48.23,
    ("n4h", "10k", 1): 28.07, ("n4h", "32k", 1): 29.26, ("n4h", "128k", 1): 31.82, ("n4h", "240k", 1): 34.23,
    ("n4h", "10k", 2): 33.27, ("n4h", "32k", 2): 34.95, ("n4h", "128k", 2): 40.03, ("n4h", "240k", 2): 45.62,
    ("n4r", "10k", 1): 27.64, ("n4r", "32k", 1): 27.89, ("n4r", "128k", 1): 30.80, ("n4r", "240k", 1): 35.12,
    ("n4r", "10k", 2): 31.97, ("n4r", "32k", 2): 33.16, ("n4r", "128k", 2): 39.42, ("n4r", "240k", 2): 45.81,
    ("n4rt", "10k", 1): 27.28, ("n4rt", "32k", 1): 27.66, ("n4rt", "128k", 1): 30.67, ("n4rt", "240k", 1): 34.76,
    ("n4rt", "10k", 2): 31.97, ("n4rt", "32k", 2): 33.23, ("n4rt", "128k", 2): 39.28, ("n4rt", "240k", 2): 45.33,
}
BAND = {"i8": 0.35, "n4": 0.60, "i8rt": 0.80, "n4rt": 0.95}


def _fmt(arm):
    return "int8" if arm.startswith("i8") else "nvfp4"


def _model(arm, depth, bs):
    fmt = _fmt(arm)
    t = D.token_share(TOKV[arm]) if arm in TOKV else D.seg_token_share(fmt, UNITS[arm], "bandwidth")
    return D.seg_round_ms(D.rc9_seg_calib(fmt), D.unit_shares(UNITS[arm]), t, bs, CTX[depth])


class TestSegmentModel(unittest.TestCase):
    def test_reproduces_measurement_within_band(self):
        for (arm, depth, bs), meas in MEAS.items():
            band = BAND.get(arm, BAND[arm[:2]])
            with self.subTest(arm=arm, depth=depth, bs=bs):
                self.assertLessEqual(abs(_model(arm, depth, bs) - meas), band)

    def test_r_against_h_sign(self):
        for bs in (1, 2):
            for depth in CTX:
                with self.subTest(fmt="int8", depth=depth, bs=bs):   # INT8 'dec' is a LOSS
                    self.assertGreater(MEAS[("i8r", depth, bs)], MEAS[("i8h", depth, bs)])
                    self.assertGreater(_model("i8r", depth, bs), _model("i8h", depth, bs))
            for depth in ("10k", "32k", "128k"):
                with self.subTest(fmt="nvfp4", depth=depth, bs=bs):  # NVFP4 'dec' GAINS up to 128k
                    self.assertLess(MEAS[("n4r", depth, bs)], MEAS[("n4h", depth, bs)])
                    self.assertLess(_model("n4r", depth, bs), _model("n4h", depth, bs))
        # the INT8 loss grows with depth (capacity token vector moves KV onto the 3080s)
        deltas = [_model("i8r", d, 1) - _model("i8h", d, 1) for d in CTX]
        self.assertEqual(deltas, sorted(deltas))

    def test_nvfp4_240k_tip_is_a_known_miss(self):
        """Measured n4r - n4h at 240k bs1 = +0.89 ms (the tip); the model only
        predicts the gain vanishing (-0.66 ms at 10k -> ~0 at 240k)."""
        meas = MEAS[("n4r", "240k", 1)] - MEAS[("n4h", "240k", 1)]
        model = _model("n4r", "240k", 1) - _model("n4h", "240k", 1)
        self.assertGreater(meas, 0.5)
        self.assertGreater(model, -0.3)
        self.assertLess(model, meas)   # if this flips, the miss is fixed: update the doc sec. 15
        self.assertLess(_model("n4r", "10k", 1) - _model("n4h", "10k", 1), model)

    def test_sum_objective_gets_the_int8_sign_wrong(self):
        g, cal = D.rc9_geometry("int8"), D.rc9_calib("int8")
        ld = D.LoadClass("decode", 1, 32768)
        base = D.round_ms(g, cal, D.Vector(D.RC9_BASE), ld, D.token_share((26, 19, 19)))
        dec = D.round_ms(g, cal, D.rc9_presets("int8", ("dec",))[0].vector(D.RC9_BASE), ld,
                         D.token_share((21, 22, 21)))
        self.assertLess(dec, base * 0.97)                                # sum model: 'dec' >= 3 % faster
        self.assertGreater(MEAS[("i8r", "32k", 1)], MEAS[("i8h", "32k", 1)])  # metal: slower
        self.assertGreater(_model("i8r", "32k", 1), _model("i8h", "32k", 1))  # segment model: slower

    def test_optimum_sits_between_the_measured_vectors(self):
        for fmt, units, lo, hi in (("int8", 1088, 584, 707), ("nvfp4", 136, 73, 98)):
            for p in D.seg_ladder(10240):
                v, _ms = D.seg_best(fmt, p, units)
                with self.subTest(fmt=fmt, bs=p.bs):
                    self.assertTrue(lo < v[0] < hi, v)   # interpolation, not extrapolation

    def test_bs_extrapolation_is_flagged(self):
        cal = D.rc9_seg_calib("int8")
        self.assertFalse(cal.at(1)[5])
        self.assertFalse(cal.at(2)[5])
        self.assertTrue(cal.at(4)[5])
        lines = D.seg_plan_lines("int8", (628, 230, 230), (D.SegPoint(4, 32768),))
        self.assertIn("EXTRAPOLATED(bs)", lines[0])
        self.assertNotIn("EXTRAPOLATED", D.seg_plan_lines("int8", (628, 230, 230), (D.SegPoint(2, 32768),))[0])

    def test_capacity_token_share_follows_the_vector(self):
        # i8r measured capacities 247074/262401/254977 -> rank-0 share 0.323
        t = D.seg_token_share("int8", (707, 191, 190))
        self.assertAlmostEqual(t[0], 247074 / 764452, delta=0.01)
        self.assertAlmostEqual(sum(t), 1.0, places=9)
        base = D.seg_token_share("int8", tuple(partition_units(1088, [58, 25, 25])))
        self.assertAlmostEqual(base[0], 312994 / 768164, places=6)
        bw = D.seg_token_share("int8", (707, 191, 190), "bandwidth")
        self.assertAlmostEqual(bw[0], 937 / (937 + 604 + 604), places=6)
        with self.assertRaises(D.ReshardError):
            D.seg_token_share("int8", (707, 191, 190), "elsewhere")


class TestSegmentPlanner(unittest.TestCase):
    def test_robust_choice_and_regret(self):
        for fmt, units in (("int8", 1088), ("nvfp4", 136)):
            v, worst, reg = D.seg_choose(fmt, D.seg_ladder(), units)
            self.assertEqual(sum(v), units)
            self.assertLessEqual(worst, 0.01)
            self.assertEqual(set(reg), {p.key() for p in D.seg_ladder()})
            self.assertAlmostEqual(worst, max(reg.values()))
            for p in D.seg_ladder():   # the per-point optimum has zero regret against itself
                vb, tb = D.seg_best(fmt, p, units)
                self.assertAlmostEqual(D.seg_ms_of(fmt, vb, p), tb)
        self.assertEqual(D.seg_choose("int8", D.seg_ladder(), 1088)[0], (628, 230, 230))
        self.assertEqual(D.seg_choose("nvfp4", D.seg_ladder(), 136)[0], (92, 22, 22))

    def test_depth_choice_differs_from_robust(self):
        deep = D.seg_choose("int8", D.seg_ladder(245760), 1088)[0]
        shallow = D.seg_choose("int8", D.seg_ladder(10240), 1088)[0]
        self.assertLess(deep[0], shallow[0])   # deeper -> keep KV on the 5090, less MLP there

    def test_seg_vector_beats_today_and_the_sum_preset(self):
        for fmt, units, seg, dec in (("int8", 1088, (628, 230, 230), (707, 191, 190)),
                                     ("nvfp4", 136, (92, 22, 22), (98, 19, 19))):
            base = tuple(partition_units(units, [58, 25, 25]))
            for p in D.seg_ladder(10240) + D.seg_ladder(131072):
                with self.subTest(fmt=fmt, p=p.key()):
                    self.assertLess(D.seg_ms_of(fmt, seg, p), D.seg_ms_of(fmt, base, p))
                    self.assertLessEqual(D.seg_ms_of(fmt, seg, p), D.seg_ms_of(fmt, dec, p) + 0.2)

    def test_fp8_borrowed(self):
        self.assertEqual(D.rc9_seg_calib("fp8").borrowed, "INT8 constants")
        self.assertIn("BORROWED", D.seg_plan_lines("fp8", (628, 230, 230), (D.SegPoint(1, 10240),))[0])
        with self.assertRaises(D.ReshardError):
            D.rc9_seg_calib("gguf")

    def test_seg_advisory_rows(self):
        lines = D.seg_speed_advisory_lines(D._QWEN38_27B, "compressed-tensors", (58, 25, 25),
                                           partition_units(1088, [58, 25, 25]), 1088, (26, 19, 19))
        self.assertIn("wake-seg takes mlp=628,230,230", lines[0])
        rows = [x for x in lines if "D-SPEED decode" in x]
        self.assertEqual(len(rows), 8)
        for r in rows:
            self.assertIn("[segment]", r)
            self.assertGreaterEqual(float(r.split("gain=")[1].split("%")[0]),
                                    float(r.split("wake-seg mlp")[1].split("%")[0]) - 1e-9)
        self.assertIn("No segment speed table",
                      D.seg_speed_advisory_lines(D._QWEN38_27B, "gguf", (58, 25, 25), (1, 1, 1), 3, None)[0])


class TestDefaultUnchanged(unittest.TestCase):
    def test_wake_spec_digest_is_the_boots(self):
        for fmt, digest in (("int8", "96b0cd92ae412589"), ("nvfp4", "5b8ad0c5dc248b2e")):
            s = D.ReshardSpec("wake", D.RC9_BASE, D.rc9_presets(fmt, ("dec",)), "dec", 0.02, fmt)
            self.assertEqual(s.digest(), digest)   # the i8r / n4r D logs of 26.09.
            self.assertNotIn("objective", s.to_json())
        seg = D.ReshardSpec("wake", D.RC9_BASE, (D.Preset("seg", (628, 230, 230)),), "seg", 0.02, "int8",
                            D.OBJECTIVE_SEGMENT)
        self.assertEqual(D.ReshardSpec.from_json(seg.to_json()), seg)
        self.assertIn("objective=segment", D.armed_line(seg))
        with self.assertRaises(D.ReshardError):
            D.ReshardSpec("wake", D.RC9_BASE, seg.presets, "seg", 0.02, "int8", "fastest").validate(
                D.rc9_geometry("int8"))

    def test_objective_env(self):
        self.assertEqual(D.objective_from_env({}), D.OBJECTIVE_SUM)
        self.assertEqual(D.objective_from_env({D.OBJECTIVE_ENV: "garbage"}), D.OBJECTIVE_SUM)
        self.assertEqual(D.objective_from_env({D.OBJECTIVE_ENV: "Segment"}), D.OBJECTIVE_SEGMENT)

    def test_mixin_table_follows_the_objective(self):
        from sglang.srt.distributed import utils as U
        from sglang.srt.model_executor import model_runner_kv_cache_mixin as M

        class Lin(torch.nn.Module):
            tp_family, tp_units = "mlp", 1088

        sa = types.SimpleNamespace(uneven_dcp=True, quantization=None,
                                   uneven_memory_budgets_active=lambda: True)
        mc = types.SimpleNamespace(hf_text_config=dict(D._QWEN38_27B),
                                   hf_config=types.SimpleNamespace(
                                       quantization_config={"quant_method": "compressed-tensors"}))
        r = types.SimpleNamespace(server_args=sa, dcp_size=3, tp_size=3, tp_rank=0, is_draft_worker=False,
                                  model=torch.nn.Sequential(Lin()), model_config=mc)
        orig_info, old = M.logger.info, os.environ.pop(D.OBJECTIVE_ENV, None)
        try:
            for env, want, unwanted in ((None, "D-SPEED prefill-4k", "[segment]"),
                                        ("segment", "[segment]", "D-SPEED prefill-4k")):
                logged = []
                M.logger.info = lambda msg, *a: logged.append(msg % a if a else msg)
                if env:
                    os.environ[D.OBJECTIVE_ENV] = env
                with U.scoped_tp_partition_ratios([58, 25, 25]):
                    self.assertTrue(M._dcp_speed_advisory(r))
                self.assertTrue(any(want in x for x in logged), env)
                self.assertFalse(any(unwanted in x for x in logged), env)
        finally:
            M.logger.info = orig_info
            os.environ.pop(D.OBJECTIVE_ENV, None)
            if old is not None:
                os.environ[D.OBJECTIVE_ENV] = old


class TestLauncherWakeSeg(unittest.TestCase):
    def setUp(self):
        from sglang.srt.weg2 import launcher as L
        self.L = L
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        self.L.apply_d_reshard(self._ns())   # leave 'off' behind

    def _model(self, qm):
        d = tempfile.mkdtemp(dir=self.tmp)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"text_config": dict(D._QWEN38_27B), "quantization_config": {"quant_method": qm}}, f)
        return d

    def _ns(self, *extra, model=None):
        base = ["--tree", "/t", "--tag", "t"] + (["--model", model] if model else [])
        return self.L.build_parser().parse_args(base + list(extra))

    def test_off_and_wake_unchanged(self):
        ns = self._ns()
        self.assertEqual((ns.d_reshard, ns.d_reshard_depth, ns.d_gc_freeze), ("off", "robust", "off"))
        self.L.apply_d_reshard(ns)
        self.assertEqual(self.L.d_reshard_env(), {})
        self.assertEqual(self.L.d_gc_env(ns), {})
        self.L.apply_d_reshard(self._ns("--d-reshard", "wake", model=self._model("compressed-tensors")))
        env = self.L.d_reshard_env()
        self.assertNotIn(D.OBJECTIVE_ENV, env)
        self.assertEqual(D.ReshardSpec.from_json(env[D.SPEC_ENV]).digest(), "96b0cd92ae412589")
        self.assertFalse(any("seg-plan" in x for x in self.L.d_reshard_lines()))

    def test_wake_seg_auto_takes_the_robust_vector(self):
        for qm, want in (("compressed-tensors", "628,230,230"), ("modelopt", "92,22,22")):
            self.L.apply_d_reshard(self._ns("--d-reshard", "wake-seg", model=self._model(qm)))
            self.assertEqual(self.L.d_reshard_argv(("--rank-tp-ratio", "58,25,25"), []),
                             ["--rank-mlp-ratio", want])
            env = self.L.d_reshard_env()
            self.assertEqual(env[D.POLICY_ENV], "wake")
            self.assertEqual(env[D.OBJECTIVE_ENV], "segment")
            lines = self.L.d_reshard_lines()
            self.assertTrue(any("objective=segment" in x for x in lines))
            self.assertEqual(sum("seg-plan" in x for x in lines), 8)

    def test_wake_seg_depth_placement_and_explicit(self):
        m = self._model("compressed-tensors")
        self.L.apply_d_reshard(self._ns("--d-reshard", "wake-seg", "--d-reshard-depth", "240k", model=m))
        self.assertEqual(self.L.d_reshard_argv(("--rank-tp-ratio", "58,25,25"), [])[1], "602,243,243")
        self.L.apply_d_reshard(self._ns("--d-reshard", "wake-seg", "--d-token-placement", "bandwidth", model=m))
        self.assertEqual(self.L.d_reshard_argv(("--rank-tp-ratio", "58,25,25"), [])[1], "646,221,221")
        self.L.apply_d_reshard(self._ns("--d-reshard", "wake-seg", "--d-reshard-presets", "seg=628:230:230",
                                        model=m))
        self.assertEqual(self.L.d_reshard_argv(("--rank-tp-ratio", "58,25,25"), [])[1], "628,230,230")
        self.assertEqual(self.L.d_reshard_env()[D.OBJECTIVE_ENV], "segment")
        with self.assertRaises(SystemExit):
            self.L.apply_d_reshard(self._ns("--d-reshard", "wake-seg", "--d-reshard-depth", "-3", model=m))
        self.assertEqual(self.L.d_reshard_depth("128k"), 131072)
        self.assertIsNone(self.L.d_reshard_depth("robust"))

    def test_gc_freeze_env(self):
        self.assertEqual(self.L.d_gc_env(self._ns("--d-gc-freeze", "on")), {G.FREEZE_ENV: "1"})


class TestGcInstrument(unittest.TestCase):
    def test_off_by_default(self):
        n = len(gc.callbacks)
        out = G.arm_after_boot(types.SimpleNamespace(gc_warning_threshold_secs=0.0), 0, env={})
        self.assertEqual(out, {"warn": None, "freeze": False})
        self.assertEqual(len(gc.callbacks), n)

    def test_warn_and_freeze_when_asked(self):
        n = len(gc.callbacks)
        try:
            out = G.arm_after_boot(types.SimpleNamespace(gc_warning_threshold_secs=0.1), 2,
                                   env={G.FREEZE_ENV: "1"})
            self.assertEqual(out, {"warn": 0.1, "freeze": True})
            self.assertEqual(len(gc.callbacks), n + 1)
            self.assertGreater(gc.get_freeze_count(), 0)
        finally:
            del gc.callbacks[n:]
            gc.unfreeze()


if __name__ == "__main__":
    unittest.main()
