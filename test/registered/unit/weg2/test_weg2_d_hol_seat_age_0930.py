# SPDX-License-Identifier: Apache-2.0
"""D's half of the seat policy (from 27B 2d49cd45bf, metal dual20 ...09301417;
only its backfill and displacement cases -- the dual-layout ones are not NF's).

Head-of-line blocking on D: a 358-token request waited 126 s behind a 112k
head that waited for KV growth (NO_TOKEN -> break). User rule (30.09.): age
has precedence, younger requests backfill, and displacement happens only when
it is ENOUGH for the older one ("nicht pauschal den jüngeren verdrängen").
In NF this is the ONE displacement logic (d_park_runtime.displace_for_age);
the front's AGE PLAN parks nothing (test_weg2_arrival_seat_age_plan_0930).
"""
from __future__ import annotations

import inspect
import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import d_park_runtime as DP  # noqa: E402
from sglang.srt.weg2 import hol_overtake as H  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class DAdmissionAgeBackfillDisplace(CustomTestCase):
    def test_displacement_only_when_it_is_enough(self):
        # the user's three cases (30.09.)
        self.assertEqual(DP.victims_needed(1000, 1500, [800, 900]), 0)       # fits free: nobody leaves
        self.assertEqual(DP.victims_needed(1000, 400, [700, 900]), 1)        # exactly one young seat
        self.assertIsNone(DP.victims_needed(5000, 400, [700, 900]))           # not even all: backfill stays
        self.assertEqual(DP.victims_needed(1500, 400, [700, 900]), 2)

    def test_kv_verdict_reads_the_younger_seats_only(self):
        def req(rid, n):
            return types.SimpleNamespace(rid=rid, origin_input_ids=list(range(n)), output_ids=[],
                                         prefix_indices=[])

        older = req("weg2-0-6", 1000)
        sched = types.SimpleNamespace(
            waiting_queue=[older],
            token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: 400),
            tree_cache=types.SimpleNamespace(evictable_size=lambda: 0))
        running_old_big = [req("weg2-0-5", 5000)]                              # older than it: never a victim
        self.assertFalse(DP.kv_displace_would_fit(sched, "weg2-0-6", running_old_big))
        self.assertTrue(DP.kv_displace_would_fit(sched, "weg2-0-6", running_old_big + [req("weg2-0-10", 700)]))

    def test_backfill_gate_group_d_only(self):
        s = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=1))
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "D"}):
            self.assertTrue(H.enabled(s))
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "P"}):
            self.assertFalse(H.enabled(s))
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "D", H.ENV: "0"}):
            self.assertFalse(H.enabled(s))
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "D"}):
            self.assertFalse(H.enabled(types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=3))))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(H.enabled(s))                                     # classic boot: unchanged

    def test_a_no_token_head_no_longer_ends_the_round(self):
        s = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=1))
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "D"}):
            hp = H.HolPass(s)
            head = types.SimpleNamespace(rid="weg2-0-7")                      # the 112k head waiting for growth
            self.assertTrue(hp.may_overtake(head))                            # dual20: this was `break`
            self.assertEqual(hp.head, "weg2-0-7")
            for i in range(H.MAX_SCAN - 1):
                self.assertTrue(hp.may_overtake(types.SimpleNamespace(rid="x%d" % i)))
            self.assertFalse(hp.may_overtake(types.SimpleNamespace(rid="y")))  # bounded scan
        from sglang.srt.managers import scheduler as SC

        src = inspect.getsource(SC.Scheduler._get_new_batch_prefill_raw)
        i = src.index('_note_skip(f"add_result_{res.name}", req.rid)')
        self.assertIn("if _hol_go_on:\n                    continue\n                break", src[i:i + 200])
        j = src.index("_hol_go_on = _hol.may_overtake(req)")
        self.assertLess(j, src.index("elif self.enable_hierarchical_cache:", j))   # no batch_is_full on overtake


if __name__ == "__main__":
    import unittest

    unittest.main()
