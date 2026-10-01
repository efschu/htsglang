# SPDX-License-Identifier: Apache-2.0
"""scripts/dual_layout/yield_wait_bench.py: the pure parts of the lever-(b) bench.

DANGER DIRECTIONS guarded here:
* the latency summary is order-independent and empty-safe;
* the three wait shapes are exactly spin / wait / host (spin = today's barlink);
* the D role never imports torch (its own driver context, no torch allocator
  or caching context to distort the time slice it measures);
* the stream-wait is a GEQ wait on a monotonically growing flag (an EQ wait
  would miss a value the "peer" already moved past).
"""
from __future__ import annotations

import importlib.util
import inspect
import os

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                  "scripts", "dual_layout", "yield_wait_bench.py")


def _load():
    s = importlib.util.spec_from_file_location("yield_wait_bench", _P)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


class YieldWaitBench(CustomTestCase):
    def test_summary(self):
        m = _load()
        self.assertEqual(m.summarize([]), {"n": 0})
        a = m.summarize([5.0, 1.0, 3.0, 2.0, 4.0])
        b = m.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(a, b)
        self.assertEqual((a["n"], a["p50_us"], a["max_us"]), (5, 3.0, 5.0))

    def test_modes(self):
        self.assertEqual(_load().MODES, ("spin", "wait", "host"))

    def test_d_role_has_no_torch(self):
        src = inspect.getsource(_load().run_d)
        self.assertNotIn("torch", src)

    def test_stream_wait_is_geq_on_monotonic_flag(self):
        src = inspect.getsource(_load().run_d)
        self.assertIn("CU_STREAM_WAIT_VALUE_GEQ", src)
        self.assertIn("want = i + 1", src)
