# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#791T: the #631 row frame must not overtake its Weg2StoreTold either.

THE SPECIMEN (27B proof boot rc12z20 3a86888ba5, Fix B armed,
/spinning/docker-acceptance/27b/evidence/
boot_weg2_dkr27browauthoritybar1w109281300_3a86888ba5_0928_130013.P.log):

    PP0  #1400 STORE-TOLD INTAKE rid=weg2-0-8 verdict=issued span=62505 matched=0
    PP1  REQ-TRACE r24 TokenizedGenerateReqInput weg2-0-8
    PP2  REQ-TRACE r24 TokenizedGenerateReqInput weg2-0-8
    PP1  REQ-TRACE r25 Weg2StoreTold weg2-0-8 -> ABSORBED told=0   (told BEFORE frame)
    PP1  #631 ROW-PROBE drained a proxy at slot=2 ... ('weg2-0-8', 0, 2048)
    PP2  #631 ROW-PROBE drained a proxy at slot=2 ... ('weg2-0-8', 0, 2048)
    PP2  #791 FORWARDED SCHEDULE UNEXECUTABLE STOP rank=2 slot=0 told=[weg2-0-8]
         reached=[] census=loop_skips(weg2_store_told_pending=1(first=weg2-0-8))
                                                                  (frame BEFORE told)

The request chain carries the request (r24) and, one hop later, PP0's told
(r25); the proxy frame travels on the p2p dict wire -- no cross-wire order.
The 631row14 guard defers a frame whose admitted rid is not LOCATABLE; here
the rid was located (r24 had landed), so the frame went through, and the told
gate skipped the rid. The fix: a queued rid whose told the gate would skip is
the same "hop in flight" -- defer (bounded by RowDeferCap, named stop), never
drop, never plan without the told.

RED ON THE PARENT (cb7f2cdc35): the told-pending rid is locatable, so the
probe answers True and the frame is planned into the skip.
"""

from __future__ import annotations

import types
import unittest

from sglang.srt.managers import scheduler_pp_mixin as ppm
from sglang.srt.managers.pp_row_defer_cap import ROW_DEFER_LAP_CAP, PpRowDeferCapExceeded
from sglang.srt.weg2 import p_intake


def _wire_row(rid: str, *, admitted: bool = True, retracted: bool = False):
    return (rid, 0, 1, admitted, retracted, None, None, False, None, (), None)


def _frame(slot: int, epoch: int, rids, *, pass_ct: int = 179):
    return {
        "__stamp__": (slot, pass_ct, 2048, epoch, pass_ct, ("weg2-0-8", 0, 2048)),
        ppm._ADMISSION_DECISION_PAYLOAD_KEY: (slot, tuple(_wire_row(r) for r in rids)),
    }


class ToldOrderProbeTest(unittest.TestCase):
    SLOT = 2
    EPOCH = 2
    RID = "weg2-0-8"

    def setUp(self):
        self._orig_src = ppm.resolve_src
        self._orig_inbox = ppm.typed_inbox
        self.addCleanup(self._restore)

    def _restore(self):
        ppm.resolve_src = self._orig_src
        ppm.typed_inbox = self._orig_inbox

    def _sched(self, queue, *, pp_rank: int, armed: bool = True):
        ppm.resolve_src = lambda group, x: 0
        ppm.typed_inbox = lambda group: {(0, "proxy"): queue}
        sched = types.SimpleNamespace()
        sched.pp_group = object()
        sched.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=3, tp_size=1)
        self.req = types.SimpleNamespace(rid=self.RID)
        sched.waiting_queue = [self.req]          # r24 landed: the rid IS locatable
        sched.chunked_req = None
        sched.running_batch = None
        sched._pp_row_chain_owed = False
        sched._pp_flip_epoch = lambda: self.EPOCH
        # weg2_store_told.armed() resolved once per scheduler (boot constant)
        sched._weg2_store_told_armed = armed
        sched._weg2_store_told = {}
        sched._weg2_store_held = {self.RID: self.req}
        return sched

    def _probe(self, sched):
        return ppm.SchedulerPPMixin._pp_proxy_frame_pending(sched, self.SLOT)

    @staticmethod
    def _absorb(sched, rid, told):
        """What _follower_absorb_impl stores when r25 lands (the part the gate reads)."""
        sched._weg2_store_told[rid] = told
        sched._weg2_store_held.pop(rid, None)

    # ------------------------------------------------ the 1300 order, both ranks

    def _pp2_frame_before_told(self, told):
        queue = [_frame(self.SLOT, self.EPOCH, [self.RID])]
        sched = self._sched(queue, pp_rank=2)
        self.assertIs(self._probe(sched), False, "PP2: frame before told must defer")
        self.assertEqual(len(queue), 1, "the deferred frame stays in the inbox (no drop)")
        self.assertIs(sched._pp_row_chain_owed, True, "the chain hop (r25) is received next")
        self.assertEqual(sched._pp_row_probe_stats.get("defer_told"), 1)
        self._absorb(sched, self.RID, told)        # r25 lands
        self.assertIs(self._probe(sched), True, "told arrived: the frame is this slot's")
        self.assertEqual(len(queue), 1, "and it is still there for the slot's own recv")

    def test_pp2_frame_before_told_zero_defers_until_the_told_lands(self):
        self._pp2_frame_before_told(0)             # the specimen: matched=0 -> told=0

    def test_pp2_frame_before_told_positive_defers_until_the_told_lands(self):
        self._pp2_frame_before_told(61439)         # a store hit (weg2-0-7's shape)

    def test_pp1_told_before_frame_does_not_defer(self):
        for told in (0, 61439):
            queue = [_frame(self.SLOT, self.EPOCH, [self.RID])]
            sched = self._sched(queue, pp_rank=1)
            self._absorb(sched, self.RID, told)    # r25 absorbed before the probe
            self.assertIs(self._probe(sched), True, told)
            self.assertIs(sched._pp_row_chain_owed, False)
            self.assertNotIn("defer_told", sched._pp_row_probe_stats)

    # ------------------------------------------------------------ not deferred

    def test_a_kept_verdict_of_this_request_does_not_defer(self):
        """H91 STORE-TOLD KEPT: the told was consumed by an earlier visit that did
        not admit; told_admission answers from the kept verdict -- not pending."""
        queue = [_frame(self.SLOT, self.EPOCH, [self.RID])]
        sched = self._sched(queue, pp_rank=2)
        setattr(sched, p_intake.KEPT_ATTR,
                {self.RID: p_intake._Kept(req=self.req, told=0, credit=0)})
        self.assertIs(self._probe(sched), True)

    def test_a_kept_verdict_of_another_request_object_still_defers(self):
        queue = [_frame(self.SLOT, self.EPOCH, [self.RID])]
        sched = self._sched(queue, pp_rank=2)
        other = types.SimpleNamespace(rid=self.RID)
        setattr(sched, p_intake.KEPT_ATTR, {self.RID: p_intake._Kept(req=other, told=0, credit=0)})
        self.assertIs(self._probe(sched), False)

    def test_told_not_armed_is_todays_behaviour(self):
        queue = [_frame(self.SLOT, self.EPOCH, [self.RID])]
        sched = self._sched(queue, pp_rank=2, armed=False)
        self.assertIs(self._probe(sched), True)

    def test_pp0_never_defers_on_its_own_told(self):
        sched = self._sched([], pp_rank=0)
        self.assertIs(p_intake.told_pending(sched, self.req), False)

    def test_a_retracted_row_entry_does_not_defer(self):
        queue = [{
            "__stamp__": (self.SLOT, 179, 2048, self.EPOCH, 179, ("weg2-0-8", 0, 2048)),
            ppm._ADMISSION_DECISION_PAYLOAD_KEY: (
                self.SLOT, (_wire_row(self.RID, admitted=True, retracted=True),)),
        }]
        sched = self._sched(queue, pp_rank=2)
        self.assertIs(self._probe(sched), True)

    # ------------------------------------------------------------------ bounded

    def test_an_overdue_told_hop_stops_by_name_and_keeps_the_frame(self):
        queue = [_frame(self.SLOT, self.EPOCH, [self.RID])]
        sched = self._sched(queue, pp_rank=2)
        with self.assertRaises(PpRowDeferCapExceeded) as cm:
            for _ in range(ROW_DEFER_LAP_CAP + 2):
                self.assertIs(self._probe(sched), False)
        msg = str(cm.exception)
        self.assertIn("#791T STORE-TOLD HOP OVERDUE", msg)
        self.assertIn("weg2-0-8", msg)
        self.assertIn("#1180 PP ROW DEFER PAST ITS LAP CAP", msg)
        self.assertEqual(len(queue), 1, "the frame is never consumed on the stop")


if __name__ == "__main__":
    unittest.main()
