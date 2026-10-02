# SPDX-License-Identifier: Apache-2.0
"""gmps4 (dual1m ...10011614) deaths/loops: #1180 row defer after a waiting-queue abort on a
follower, and the #915 threshold refusing short hand-backs.

DANGER DIRECTIONS guarded here:
* #1180-W: a PP follower that still holds a request in its waiting queue does NOT pop it at
  abort receipt -- PP0 may already have admitted it (16:19:24Z weg2-0-9: PP0 chunked-abort
  with two more chunks, PP1 popped, PP0's slot-0 frame named the rid -> #1180, PP1 dead);
  the follower follows PP0's forwarded schedule: named -> keep (admit), admitted -> chunked
  abort path, `laps` frames naming others -> pop, PP0 idle -> pop, no frame -> keep;
* PP0, non-PP engines, abort_all and the pass-aligned form (no row authority) keep the
  immediate pop; a released rid is popped, never deferred twice;
* dual hand-back reads are below the #915 threshold allowed (min_tokens 1) on group D only.
"""
from __future__ import annotations

import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as S
from sglang.srt.weg2 import dual_anchor_claim as C
from sglang.srt.weg2 import pp_abort as A
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

V = A.follower_waiting_abort_verdict


class _Sched:
    """The scheduler surface the two #1180-W methods touch."""

    def __init__(self, pp_rank=1, pp_size=3, rids=("weg2-0-9",)):
        self.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size)
        self.waiting_queue = [types.SimpleNamespace(rid=r) for r in rids]
        self.chunked_req = None
        self.frame = None
        self.aborted = []
        self._791c_pp0_drained = False

    def _pp_scheduled_extents(self):
        return self.frame

    def _abort_request_now(self, recv_req):
        if self._weg2_defer_waiting_abort(recv_req):
            return
        self.aborted.append(recv_req.rid)
        self.waiting_queue = [r for r in self.waiting_queue if r.rid != recv_req.rid]


for _m in ("_weg2_defer_waiting_abort", "_weg2_process_waiting_aborts"):
    setattr(_Sched, _m, getattr(S.Scheduler, _m))


def _abort(rid):
    return S.AbortReq(rid=rid)


class DualAbortHandback(CustomTestCase):
    def setUp(self):
        self._ra = A.row_authority_of
        A.row_authority_of = lambda sched: True

    def tearDown(self):
        A.row_authority_of = self._ra

    def test_verdict_table(self):
        self.assertEqual(V("r", None, 0, laps=3, pp0_drained=False, admitted=True, in_waiting=False)[0], "chunked")
        self.assertEqual(V("r", None, 0, laps=3, pp0_drained=False, admitted=False, in_waiting=False)[0], "gone")
        self.assertEqual(V("r", {"x": 1}, 0, laps=3, pp0_drained=True, admitted=False, in_waiting=True)[0], "pop")
        self.assertEqual(V("r", {}, 2, laps=3, pp0_drained=False, admitted=False, in_waiting=True), ("keep", 2))
        self.assertEqual(V("r", {"r": 1}, 2, laps=3, pp0_drained=False, admitted=False, in_waiting=True), ("keep", 0))
        self.assertEqual(V("r", {"x": 1}, 1, laps=3, pp0_drained=False, admitted=False, in_waiting=True), ("keep", 2))
        self.assertEqual(V("r", {"x": 1}, 2, laps=3, pp0_drained=False, admitted=False, in_waiting=True), ("pop", 3))

    def test_gmps4_sequence_pp0_had_admitted_it(self):
        s = _Sched()
        s._abort_request_now(_abort("weg2-0-9"))
        self.assertEqual(s.aborted, [])                          # NOT popped at receipt
        self.assertEqual([r.rid for r in s.waiting_queue], ["weg2-0-9"])
        s.frame = {"weg2-0-9": (0, 1024)}                        # PP0's slot-0 frame names it
        s._weg2_process_waiting_aborts()
        self.assertEqual(s.aborted, [])                          # kept: this pass admits it
        s.chunked_req = s.waiting_queue.pop(0)                   # the follower admitted it
        s._weg2_process_waiting_aborts()
        self.assertEqual(s.aborted, ["weg2-0-9"])                # handed to the chunked path
        self.assertFalse(s._weg2_pending_waiting_aborts)

    def test_pp0_had_popped_it(self):
        s = _Sched()
        s._abort_request_now(_abort("weg2-0-9"))
        for k in range(3):
            s.frame = {"other": (0, 1)}
            s._weg2_process_waiting_aborts()
            self.assertEqual(s.aborted, [] if k < 2 else ["weg2-0-9"])
        self.assertEqual(s.waiting_queue, [])

    def test_no_frame_keeps_and_drain_releases(self):
        s = _Sched()
        s._abort_request_now(_abort("weg2-0-9"))
        s.frame = None
        for _ in range(5):
            s._weg2_process_waiting_aborts()
        self.assertEqual(s.aborted, [])
        s._791c_pp0_drained = True
        s._weg2_process_waiting_aborts()
        self.assertEqual(s.aborted, ["weg2-0-9"])

    def test_immediate_where_it_must_be(self):
        for s in (_Sched(pp_rank=0), _Sched(pp_size=1, pp_rank=0)):
            s._abort_request_now(_abort("weg2-0-9"))
            self.assertEqual(s.aborted, ["weg2-0-9"])
        s = _Sched()
        s._abort_request_now(S.AbortReq(rid="", abort_all=True))
        self.assertEqual(s._weg2_pending_waiting_aborts if hasattr(s, "_weg2_pending_waiting_aborts") else {}, {})
        A.row_authority_of = lambda sched: False                 # pass-aligned wire
        s = _Sched()
        s._abort_request_now(_abort("weg2-0-9"))
        self.assertEqual(s.aborted, ["weg2-0-9"])

    def test_handback_min_tokens(self):
        self.assertEqual(C.dual_handback_min_tokens({"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"}), 1)
        self.assertIsNone(C.dual_handback_min_tokens({"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}))
        self.assertIsNone(C.dual_handback_min_tokens({}))

    def test_wiring(self):
        # RELEASE-HEAD 1002: the base calls the hooks through the class
        # (bare-namespace harnesses, #791c) and reads the hand-back minimum
        # through handback_claim (27B fe5c55041b, covers the dual layout)
        now = inspect.getsource(S.Scheduler._abort_request_now)
        self.assertLess(now.index("_weg2_defer_waiting_abort(self, recv_req)"),
                        now.index("for i, req in enumerate(self.waiting_queue):"))
        pend = inspect.getsource(S.Scheduler.process_pending_chunked_abort)
        self.assertLess(pend.index("_weg2_process_waiting_aborts(self)"),
                        pend.index("req = self._pending_chunked_abort_req"))
        self.assertIn("handback_min_tokens(has_handoff=_weg2_hb_handoff)",
                      inspect.getsource(S.Scheduler._prefetch_kvcache))
