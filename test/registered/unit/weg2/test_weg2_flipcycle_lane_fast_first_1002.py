"""FLIPCYCLE H4 (02.10.): fast-receiver-first on a depositor's one D2H engine.

y6z P->D ep 10: P-PP0 (5090) deposited p0 (into the x4 3080, 5.22 GB) and p1
(into the x8 3080, 1.77 GB) in parallel on one D2H engine; p0 ran at 4.36 GB/s
(lane_ms 1198) instead of its receiver's ~6.6. The lane into the widest receiver
now goes first; the narrow one waits only for that lane's latest issued batch.
"""

import inspect
import os
import tempfile
import unittest

from sglang.srt.environ import envs
from sglang.srt.weg2 import bar1_lanes, lane_priority as lp


class _Ops:
    def __init__(self):
        self.synced = []

    def synchronize(self, stream):
        self.synced.append(stream)


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class TheGate(unittest.TestCase):
    def test_widest_receiver_is_top(self):
        g = lp.EngineGate({"p0": 7.9, "p1": 15.8})
        self.assertEqual(g.top, "p1")

    def test_equal_receivers_do_not_arm(self):
        self.assertFalse(lp.EngineGate({"p2": 15.8, "p3": 15.8}).armed)
        self.assertFalse(lp.EngineGate({"p2": 15.8}).armed)

    def test_slow_lane_waits_for_the_fast_lanes_latest_issue(self):
        clk = _Clock()
        g = lp.EngineGate({"p0": 7.9, "p1": 15.8}, clock=clk)
        ops = _Ops()
        self.assertEqual(g.before_issue("p0", ops), 0.0)   # fast lane idle: nothing
        g.issued("p1", 4711)
        g.before_issue("p0", ops)
        self.assertEqual(ops.synced, [4711])
        g.before_issue("p1", ops)                            # the top lane never waits
        self.assertEqual(ops.synced, [4711])

    def test_a_fast_lane_waiting_on_credit_is_not_active(self):
        clk = _Clock()
        g = lp.EngineGate({"p0": 7.9, "p1": 15.8}, clock=clk)
        ops = _Ops()
        g.issued("p1", 1)
        clk.t += lp.ACTIVE_S * 2
        g.before_issue("p0", ops)
        self.assertEqual(ops.synced, [])

    def test_retired_streams_are_never_synced(self):
        g = lp.EngineGate({"p0": 7.9, "p1": 15.8})
        ops = _Ops()
        g.issued("p1", 9)
        g.retire("p1")
        g.before_issue("p0", ops)
        self.assertEqual(ops.synced, [])


class TheLinkRate(unittest.TestCase):
    def _dev(self, root, bdf, width, speed):
        d = os.path.join(root, bdf)
        os.makedirs(d)
        open(os.path.join(d, "current_link_width"), "w").write(f"{width}\n")
        open(os.path.join(d, "max_link_speed"), "w").write(f"{speed} GT/s PCIe\n")

    def test_x4_gen4_is_half_of_x8_gen4(self):
        with tempfile.TemporaryDirectory() as root:
            self._dev(root, "0000:01:00.0", 4, "16.0")
            self._dev(root, "0000:02:00.0", 8, "16.0")
            a = lp.link_gbytes_per_s("0000:01:00.0", root)
            b = lp.link_gbytes_per_s("0000:02:00.0", root)
            self.assertAlmostEqual(b, 2 * a)
            self.assertAlmostEqual(a, 4 * 16.0 * 128 / 130 / 8)
            self.assertIsNone(lp.link_gbytes_per_s("0000:09:00.0", root))


class TheWiring(unittest.TestCase):
    def test_switch_default_on(self):
        self.assertTrue(envs.SGLANG_WEG2_ENABLE_LANE_FAST_FIRST.get())

    def test_deposit_loop_asks_the_gate_and_retires_the_top_lane(self):
        src = inspect.getsource(bar1_lanes._run_bar1_tag_streamed)
        self.assertIn("_gate.before_issue(lane_key, ops)", src)
        self.assertIn("_gate.issued(lane_key, stream)", src)
        self.assertLess(src.index("_gate.before_issue(lane_key, ops)"),
                        src.index("_gate.issued(lane_key, stream)"))
        self.assertIn("_g.retire(lane_key)", inspect.getsource(bar1_lanes._run_bar1_tag))

    def test_switch_off_builds_no_gate(self):
        with envs.SGLANG_WEG2_ENABLE_LANE_FAST_FIRST.override(False):
            self.assertIsNone(lp.gate_for(object()))


if __name__ == "__main__":
    unittest.main()
