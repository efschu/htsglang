"""VISION-IPC (Nutzer 02.10. ~11:00Z: "dann muss der visiontower laden rechnen
entladen auch mit in die phasenliste ins dashboard"): the transient tower stage
as rankstats ``vision`` -- the dashboard reads IPC, never the log line
(DASHBOARD-AUS-IPC, test_no_new_log_parsers).

Metal shape (y7h-noH4, P PP0, 02.10.):
  run=1 legs_ms=(build 69, load 488, encode 838, attach 0, teardown 439)  fresh image
  run=2 legs_ms=(build 55, load 478, encode 20, attach 0, teardown 413)   cached image

RED on 23c8fb584e: rank_timing has no note_vision_* / vision_block, rankstats
no ``vision`` key, StageOutcome no legs_wall.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.pdflip import rank_timing as rt  # noqa: E402
from flliper.srt.pdflip import rankstats  # noqa: E402
from flliper.srt.pdflip import vision_rank_runner as vrr  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402
from flliper.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

RUN1 = {"build": 69.0, "load": 488.0, "encode": 838.0, "attach": 0.0, "teardown": 439.0}
RUN2 = {"build": 55.0, "load": 478.0, "encode": 20.0, "attach": 0.0, "teardown": 413.0}


def _sched():
    return types.SimpleNamespace(tree_cache=None, _pdflip_store_short_seen=0)


def _runner_helpers():
    p = pathlib.Path(__file__).with_name("test_pdflip_vision_rank_runner.py")
    spec = importlib.util.spec_from_file_location("_vrr_helpers_1002", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestVisionIpc(CustomTestCase):
    def setUp(self):
        rt.reset()

    def tearDown(self):
        rt.reset()

    def test_no_stage_no_key(self):
        self.assertIsNone(rt.vision_block())
        self.assertNotIn("vision", rankstats.scheduler_counters(_sched()))

    def test_a_run_from_legs_ms_lays_the_legs_back_from_its_end(self):
        # the metal run 1: the line came at 10:54:28.5; legs laid back in order
        t1 = 1790938468.5
        rt.note_vision_run(1, ok=True, code="W102", rids=["pdflip-0-1"], legs_wall={}, legs_ms=RUN1,
                           tower_bytes=int(856.3 * (1 << 20)), place="kvtail", card=1, t=t1)
        b = rankstats.scheduler_counters(_sched())["vision"]
        self.assertEqual(b["runs"], 1)
        self.assertIsNone(b["live"])
        r = b["recent"][-1]
        self.assertEqual((r["run"], r["rids"], r["place"], r["card"], r["tower_mib"]),
                         (1, ["pdflip-0-1"], "kvtail", 1, 856.3))
        legs = r["legs"]
        self.assertAlmostEqual(legs["teardown"][1], t1, places=3)
        self.assertAlmostEqual(legs["teardown"][0], t1 - 0.439, places=3)
        self.assertAlmostEqual(legs["build"][0], t1 - 1.834, places=3)
        self.assertAlmostEqual(r["t0"], t1 - 1.834, places=3)
        self.assertEqual(legs["encode"][2], 838.0)
        json.dumps(b)

    def test_live_leg_while_running_cleared_by_the_run(self):
        rt.note_vision_leg(2, "load", ["pdflip-1-5"], t=100.0)
        b = rt.vision_block()
        self.assertEqual(b["live"], {"run": 2, "leg": "load", "since": 100.0, "rids": ["pdflip-1-5"]})
        self.assertEqual(b["runs"], 0)
        rt.note_vision_run(2, ok=True, code="W102", rids=["pdflip-1-5"],
                           legs_wall={k: (100.0, 100.0 + v / 1e3) for k, v in RUN2.items()},
                           legs_ms=RUN2, t=101.0)
        b = rt.vision_block()
        self.assertIsNone(b["live"])
        self.assertEqual(b["recent"][-1]["legs"]["encode"][2], 20.0)   # cached image: nothing to encode

    def test_the_stage_writes_its_legs_wall_and_the_live_leg(self):
        import tempfile

        h = _runner_helpers()
        with tempfile.TemporaryDirectory() as d:
            h._write_model(pathlib.Path(d))
            s = h._stage_sched()
            s._pdflip_vision_runs = 7
            seen = []
            orig = rt.note_vision_leg

            def spy(run, leg, rids, t=None):
                seen.append((run, leg, list(rids)))
                orig(run, leg, rids, t)

            rt.note_vision_leg = spy
            try:
                out = vrr.run_rank_stage(s, [h._req("r1", [h._Item(4)])], model_dir=d, hf_config=None,
                                         device=torch.device("cpu"), build=h._build())
            finally:
                rt.note_vision_leg = orig
        self.assertTrue(out.ok, out.detail)
        self.assertEqual([x[1] for x in seen], ["build", "reserve", "load", "encode", "attach", "teardown"])
        self.assertTrue(all(x[0] == 7 and x[2] == ["r1"] for x in seen))
        self.assertEqual(set(out.legs_wall), {"build", "reserve", "load", "encode", "attach", "teardown"})
        order = [out.legs_wall[k] for k in ("build", "reserve", "load", "encode", "attach", "teardown")]
        for (a0, a1), (b0, _b1) in zip(order, order[1:]):
            self.assertLessEqual(a0, a1)
            self.assertLessEqual(a1, b0 + 1e-6)
        vrr.log_outcome(out, ["r1"], 7)
        r = rt.vision_block()["recent"][-1]
        self.assertEqual(r["run"], 7)
        self.assertIsNone(rt.vision_block()["live"])
        self.assertEqual(set(r["legs"]), set(out.legs_wall))


if __name__ == "__main__":
    unittest.main()
