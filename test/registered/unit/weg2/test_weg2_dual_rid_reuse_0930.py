# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3: a rid that comes back to P (the front's W50-REROUTE after D's
X refusal) must not inherit its first instance's prefetch records.

Metal 30.09. boot tnybbz (...09300632): weg2-0-8 finished on P (read 36863 of
40765 from the store), D refused leg 2 (tail not in the store yet), the front
re-routed it through P under the SAME rid. Instance 2 registered at head 36863
with span 3902, but both PP0 and PP1 read the rid's completion record of
instance 1 (36863) as the span: PP0 'TK ABS-TOLD head=36863 span=36863
told=73726' (anchor-clamped to 40765), PP1 'STORE-TOLD MISMATCH told=40765
own_prefix=73726' -> W17.

DANGER DIRECTIONS guarded here:
* a new request's intake drops the rid's leftover completion / loaded records
  on EVERY rank (PP0 and followers), so both sides of the told compare the
  new instance only;
* a LIVE read of the rid is never touched;
* off dual-share (every flip boot) nothing changes.
"""
from __future__ import annotations

import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_store_told as T
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _tree(live=()):
    return types.SimpleNamespace(
        _prefetch_completed_tokens={"weg2-0-8": 36863, "other": 5},
        prefetch_loaded_tokens_by_reqid={"weg2-0-8": 36863},
        ongoing_prefetch={r: object() for r in live},
    )


class DualRidReuse(CustomTestCase):
    def test_leftovers_dropped_under_dual_share(self):
        t = _tree()
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_DUAL_SHARE": "1"}):
            self.assertTrue(T.forget_rid_leftovers(t, "weg2-0-8"))
        self.assertNotIn("weg2-0-8", t._prefetch_completed_tokens)
        self.assertNotIn("weg2-0-8", t.prefetch_loaded_tokens_by_reqid)
        self.assertEqual(t._prefetch_completed_tokens["other"], 5)

    def test_live_read_untouched(self):
        t = _tree(live=("weg2-0-8",))
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_DUAL_SHARE": "1"}):
            self.assertFalse(T.forget_rid_leftovers(t, "weg2-0-8"))
        self.assertEqual(t._prefetch_completed_tokens["weg2-0-8"], 36863)

    def test_off_without_dual_share(self):
        t = _tree()
        env = {k: v for k, v in os.environ.items() if k != "SGLANG_WEG2_DUAL_SHARE"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(T.forget_rid_leftovers(t, "weg2-0-8"))
        self.assertEqual(t._prefetch_completed_tokens["weg2-0-8"], 36863)

    def test_intake_calls_it_first_on_every_rank(self):
        import inspect

        src = inspect.getsource(T.intake)
        i = src.index("forget_rid_leftovers(scheduler.tree_cache, rid)")
        self.assertLess(i, src.index("if int(scheduler.ps.pp_rank) == 0:"))
