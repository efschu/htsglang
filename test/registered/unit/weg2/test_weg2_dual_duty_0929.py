# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 --dual-p-duty: P's duty cycle while D decodes (the latency guard
without MPS).

DANGER DIRECTIONS guarded here:
* off by default: no env -> no throttle object, no front task, no argv;
* a missing or unreadable busy file means NO throttle (P must never stall on a
  signal that is not there);
* the pause is sized so P computes ``duty`` of the wall time while D is busy,
  and is capped (a stuck signal cannot stall P for long);
* the writer rewrites only on a change.
"""
from __future__ import annotations

import inspect
import os
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import dual_duty as DD
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class DualDuty(CustomTestCase):
    def test_off_without_env(self):
        self.assertIsNone(DD.DutyThrottle.from_env({}))
        self.assertIsNone(DD.DutyThrottle.from_env({DD.DUTY_ENV: "0.5"}))
        self.assertIsNone(DD.DutyThrottle.from_env({DD.DUTY_ENV: "1.0", DD.DBUSY_FILE_ENV: "/x"}))

    def test_pause_follows_duty_and_signal(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dbusy")
            w = DD.DBusyWriter(path)
            clk, slept = _Clock(), []
            t = DD.DutyThrottle(0.25, path, clock=clk, sleep=slept.append)
            t.after_forward(0.030)
            self.assertEqual(t.before_forward(), 0.0)  # no file -> no throttle
            self.assertTrue(w.update(True))
            self.assertFalse(w.update(True))           # unchanged -> no rewrite
            clk.t += 1.0
            self.assertAlmostEqual(t.before_forward(), 0.030 * 0.75 / 0.25)
            self.assertTrue(w.update(False))
            clk.t += 1.0
            self.assertEqual(t.before_forward(), 0.0)
            # capped
            w.update(True)
            clk.t += 1.0
            t.after_forward(10.0)
            self.assertEqual(t.before_forward(), DD.MAX_SLEEP_S)
            self.assertEqual(t.throttled, 2)

    def test_bad_duty_refused(self):
        with self.assertRaises(ValueError):
            DD.DutyThrottle(0.0, "/x")

    def test_wiring(self):
        from sglang.srt.managers import scheduler_pp_mixin as M
        from sglang.srt.weg2 import front as F
        from sglang.srt.weg2 import launcher as L

        src = inspect.getsource(M.SchedulerPPMixin._pp_launch_batch)
        self.assertIn("_duty = (_dual_duty_throttle(self)", src)
        self.assertIn("_duty.before_forward()", src)
        self.assertIn("self.pp_group.is_first_rank".replace("self", "sched"), inspect.getsource(M._dual_duty_throttle))
        self.assertIn("if os.environ.get(\"SGLANG_WEG2_DUAL_P_DUTY\") else None", src)
        ns = L.build_parser().parse_args(["--tree", "/x", "--tag", "t", "--dual-layout", "--dual-p-duty", "0.3"])
        L.resolve_dual_layout(ns)
        env = L.dual_duty_env(ns)
        self.assertEqual(env[DD.DUTY_ENV], "0.300")
        argv = L.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 8, 8, 4096, 4096, "D")
        self.assertEqual(argv[argv.index("--dual-dbusy-file") + 1], env[DD.DBUSY_FILE_ENV])
        off = L.build_parser().parse_args(["--tree", "/x", "--tag", "t", "--dual-layout"])
        self.assertEqual(L.dual_duty_env(off), {})
        self.assertNotIn("--dual-dbusy-file", L.front_argv_for("py", "/s", 1, 2, {}, [], off, 0, 0, 8, 8,
                                                               4096, 4096, "D"))
        self.assertIn('app["dual_dbusy"] = asyncio.create_task(self.dual_dbusy_writer())',
                      inspect.getsource(F.Front.startup))
