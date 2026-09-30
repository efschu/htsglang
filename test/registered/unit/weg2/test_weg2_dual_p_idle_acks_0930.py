# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3: an idle P stage still drains its write-through acks.

Metal qu97hh (...09301029), rid weg2-0-5: P's leg 1 returned when PP0 finished
(10:32:49); PP1/PP2 wrote their tail layers at 10:33:03, but PP1's WT-ACK was
only PROCESSED at 10:33:28 (latency_ms=137 -- the copy was long done) because
an idle PP stage runs on_idle, not the batch path that flushes the acks. D's
store read stayed 3902 tokens short for 38 s, gave up (W31/W50) 0.5 s before
the ack, and the front re-routed the 40k prompt through P: 66-71 s instead of
28-31 s per long prompt. In the flip form P sleeps and its seam flushes; in the
dual layout P never sleeps.

DANGER DIRECTIONS guarded here:
* dual layout + group P + hierarchical cache: every idle pass flushes;
* anything else (D, flip boots, no hicache): untouched.
"""
from __future__ import annotations

import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import dual_p_kv_stage as S
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _sched(hicache=True):
    calls = []
    tree = types.SimpleNamespace(flush_write_through_acks=lambda: calls.append(1))
    return types.SimpleNamespace(enable_hierarchical_cache=hicache, tree_cache=tree), calls


class DualPIdleAcks(CustomTestCase):
    def test_idle_p_flushes_its_acks(self):
        s, calls = _sched()
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}):
            self.assertTrue(S.flush_acks_when_idle(s))
        self.assertEqual(calls, [1])

    def test_everything_else_untouched(self):
        for env, hic in (({"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"}, True),
                         ({"SGLANG_WEG2_GROUP": "P"}, True),
                         ({"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}, False)):
            s, calls = _sched(hic)
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertFalse(S.flush_acks_when_idle(s))
            self.assertEqual(calls, [])

    def test_wired_at_the_top_of_on_idle(self):
        import inspect

        from sglang.srt.managers import scheduler as SC

        src = inspect.getsource(SC.Scheduler.on_idle)
        self.assertLess(src.index("flush_acks_when_idle(self)"), src.index("if not self.is_fully_idle():"))
