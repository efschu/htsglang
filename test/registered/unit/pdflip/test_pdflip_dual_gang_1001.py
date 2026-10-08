# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 gang window (GangGate / GangDonePublisher, variant C1).

DANGER DIRECTIONS guarded here:
* off by default: no gang env -> no gate (the plain duty throttle or nothing);
* D idle -> P never waits (no burst accounting stall);
* while D is busy P runs exactly K launches per burst, then waits for the LAST
  stage's completion count (pipeline empty), then holds burst*(1-d)/d, capped;
* a completion count that never arrives cannot stall P beyond the drain cap,
  and the gate re-bases so the next burst does not pay it again;
* the hold ends early when D turns idle;
* the scheduler wires the publisher to the last stage only.
"""
from __future__ import annotations

import inspect
import os
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import dual_duty as DD
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _Rig:
    """Fake clock + busy file + a last stage that completes N forwards after
    `lag` seconds of fake time each (driven from the gate's sleep)."""

    def __init__(self, busy=True, lag=0.3, publish=True):
        self.d = tempfile.mkdtemp()
        self.busy = os.path.join(self.d, "wdb")
        self.set_busy(busy)
        self.clock = _Clock()
        self.lag = lag
        self.publish = publish
        self.pub = DD.GangDonePublisher(DD.gang_done_path(self.busy), start_thread=False)
        self.pending = []  # completion times of launched forwards
        self.gate = None

    def set_busy(self, b):
        with open(self.busy, "w") as f:
            f.write("1" if b else "0")

    def sleep(self, s):
        self.clock.t += s
        self.tick()

    def tick(self):
        while self.publish and self.pending and self.pending[0] <= self.clock.t:
            self.pending.pop(0)
            self.pub.step(lambda: None)

    def make(self, duty=0.5, k=4):
        self.gate = DD.GangGate(duty, self.busy, k, clock=self.clock, sleep=self.sleep)
        self.gate._read_t = -1e9
        return self.gate

    def launch(self):
        self.gate._read_t = -1e9  # re-read the busy file every call in tests
        w = self.gate.before_forward()
        start = max([self.clock.t] + self.pending[-1:])
        self.pending.append(start + self.lag)
        self.clock.t += 0.01  # host launch cost
        self.tick()
        return w


class DualGang(CustomTestCase):
    def test_off_without_env(self):
        self.assertEqual(DD.gang_chunks_from_env({}), 0)
        self.assertEqual(DD.gang_chunks_from_env({DD.GANG_ENV: "x"}), 0)
        self.assertEqual(DD.gang_chunks_from_env({DD.GANG_ENV: "0"}), 0)
        self.assertIsNone(DD.GangGate.from_env({DD.GANG_ENV: "8"}))
        self.assertIsNone(DD.GangGate.from_env({DD.GANG_ENV: "8", DD.DUTY_ENV: "1.0", DD.DBUSY_FILE_ENV: "/x"}))
        g = DD.GangGate.from_env({DD.GANG_ENV: "8", DD.DUTY_ENV: "0.5", DD.DBUSY_FILE_ENV: "/x"})
        self.assertEqual((g.chunks, g.duty, g.done_path), (8, 0.5, "/x.gang"))
        with self.assertRaises(ValueError):
            DD.GangGate(0.5, "/x", 0)

    def test_d_idle_never_waits(self):
        r = _Rig(busy=False)
        r.make(k=2)
        self.assertEqual(sum(r.launch() for _ in range(20)), 0.0)
        self.assertEqual(r.gate.bursts, 0)

    def test_burst_drain_hold(self):
        r = _Rig(busy=True, lag=0.3)
        g = r.make(duty=0.5, k=4)
        waits = [r.launch() for _ in range(9)]
        # launches 1-4 free, 5th waits for the drain + hold, 6-8 free, 9th waits
        self.assertEqual([w > 0 for w in waits], [False] * 4 + [True] + [False] * 3 + [True])
        self.assertEqual(g.bursts, 2)
        self.assertEqual(g.rebased, 0)
        # drain: the 5th launch saw all 4 forwards completed before it went
        self.assertGreaterEqual(r.pub.done, 4)
        # hold ~= burst wall (duty 0.5), capped
        self.assertLessEqual(g.held_s, 2 * DD.GANG_MAX_HOLD_S + 0.05)
        self.assertGreater(g.held_s, 0.5)

    def test_hold_capped(self):
        r = _Rig(busy=True, lag=5.0)
        g = r.make(duty=0.1, k=1)
        r.launch()
        w = r.launch()
        self.assertLessEqual(g.held_s, DD.GANG_MAX_HOLD_S + 0.03)
        self.assertLessEqual(w, DD.GANG_MAX_DRAIN_S + DD.GANG_MAX_HOLD_S + 0.1)

    def test_missing_completions_bounded_and_rebased(self):
        r = _Rig(busy=True, lag=0.1, publish=False)
        g = r.make(duty=0.5, k=2)
        r.launch(); r.launch()
        w = r.launch()
        self.assertEqual(g.rebased, 1)
        self.assertLessEqual(g.drain_s, DD.GANG_MAX_DRAIN_S + 0.01)
        self.assertLessEqual(w, DD.GANG_MAX_DRAIN_S + DD.GANG_MAX_HOLD_S + 0.05)
        self.assertFalse(g.blind)  # one timeout re-bases first
        r.launch()
        r.launch()  # second timeout without any completion -> blind (time-only hold)
        self.assertTrue(g.blind)
        r.launch()
        d0 = g.drain_s
        r.launch()
        self.assertLess(g.drain_s - d0, 0.01)

    def test_drifted_counts_rebase_once(self):
        r = _Rig(busy=True, lag=0.1)
        for _ in range(3):  # the last stage counted 3 forwards PP0 never saw
            r.pub.step(lambda: None)
        g = r.make(duty=0.5, k=2)
        # PP0 sees 3 completions ahead -> drained immediately, never times out
        for _ in range(7):
            r.launch()
        self.assertEqual(g.rebased, 0)
        self.assertFalse(g.blind)
        r2 = _Rig(busy=True, lag=0.1)
        g2 = r2.make(duty=0.5, k=2)
        g2.launched = 5  # PP0 counted 5 launches the last stage never saw
        for _ in range(3):
            r2.launch()
        self.assertEqual(g2.rebased, 1)
        d0 = g2.drain_s
        for _ in range(3):
            r2.launch()
        self.assertEqual(g2.rebased, 1)
        self.assertFalse(g2.blind)
        self.assertLess(g2.drain_s - d0, 1.0)

    def test_hold_ends_when_d_turns_idle(self):
        r = _Rig(busy=True, lag=0.3)
        g = r.make(duty=0.2, k=2)
        r.launch(); r.launch()
        orig = r.sleep

        def sleep_and_idle(s):
            orig(s)
            if g.held_s == 0.0 and r.clock.t > 0.0:
                r.set_busy(False)
                g._read_t = -1e9

        g._sleep = sleep_and_idle
        r.launch()
        self.assertLess(g.held_s, 0.1)

    def test_publisher_counts_even_when_wait_raises(self):
        r = _Rig()
        def boom():
            raise RuntimeError("x")
        with self.assertRaises(RuntimeError):
            r.pub.step(boom)
        self.assertEqual(r.pub.done, 1)
        with open(r.pub.path) as f:
            self.assertEqual(f.read(), "1")

    def test_wiring(self):
        from flliper.srt.managers import scheduler_pp_mixin as M

        launch = inspect.getsource(M.SchedulerPPMixin._pp_launch_batch)
        self.assertIn('_gang_pub = getattr(self, "_dual_gang_pub", None)', launch)
        self.assertIn("_gang_pub.submit(event.synchronize)", launch)
        helper = inspect.getsource(M._dual_duty_throttle)
        self.assertIn("_dd.GangGate.from_env()", helper)
        self.assertIn("sched.pp_group.is_last_rank", helper)
        self.assertIn("_dd.GangDonePublisher(", helper)
