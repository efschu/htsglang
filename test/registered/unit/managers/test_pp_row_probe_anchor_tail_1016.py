# SPDX-License-Identifier: Apache-2.0
"""ROW-PROBE-TAILS (NF h16s 1f4b3e73d4, boot 1016_150003, 15:13:26 PP1, RANK_EXCEPTION #1180).

    PP0  PDFLIP END-ANCHOR SPLIT rid=pdflip-6-94: last chunk ends at 89660 of 89664 (the final 4 own chunk)
    PP0  ANCHOR-TAILS minted=1 minted_rids=pdflip-6-94; #969N ADMIT slot=1 extend=60
    PP0  ANCHOR-TAILS tails_ended=1 ended_rids=pdflip-6-94; #969N ADMIT slot=2 extend=4
    PP1  ANCHOR-TAILS minted=1 open=1 minted_rids=pdflip-6-94; #969N ADMIT slot=1 extend=60
    PP1  #631 ROW-PROBE DEFER slot=2: ... not yet locatable here (first=pdflip-6-94) lap 1, 2, 3
    PP1  PpRowDeferCapExceeded: #1180 ... rids=pdflip-6-94

Second death, boot 1016_151644 15:21:52 (the same client request after the restart: 89664 tokens, 89600
cached, uncached rest 64 < twin threshold 256, followers settle early ``equal``): pdflip-0-20, the same chain --
the uncached rest fits one chunk whole, so the END-ANCHOR split makes its last 4 tokens a TAIL.

The END-ANCHOR tail lives in ``Scheduler.anchor_tails`` (fnFL2 H42) -- not in the waiting queue, not in
``chunked_req``, not in the running batch: the probe's own four-place enumeration called a rid this rank
holds "not locatable". Now the probe reads the shared ``pp_request_locations`` (which names the tails).
RED on fd344ad7a1 / 1f4b3e73d4."""

from __future__ import annotations

import types
import unittest

from flliper.srt.managers import scheduler_pp_mixin as ppm

RID = "pdflip-6-94"


def _wire_row(rid):
    return (rid, 89660, 4, True, False, None, None, False, None, (), None)


def _frame(slot, epoch, rid):
    return {"__stamp__": (slot, 38, 4, epoch, 38, (rid, 89660, 89664)),
            ppm._ADMISSION_DECISION_PAYLOAD_KEY: (slot, (_wire_row(rid),))}


class AnchorTailRowProbeTest(unittest.TestCase):
    SLOT, EPOCH = 2, 4

    def setUp(self):
        o_src, o_inbox = ppm.resolve_src, ppm.typed_inbox
        self.addCleanup(lambda: (setattr(ppm, "resolve_src", o_src), setattr(ppm, "typed_inbox", o_inbox)))

    def _sched(self, queue, *, tails=(), waiting=()):
        ppm.resolve_src = lambda group, x: 0
        ppm.typed_inbox = lambda group: {(0, "proxy"): queue}
        s = types.SimpleNamespace()
        s.pp_group = object()
        s.ps = types.SimpleNamespace(pp_rank=1, pp_size=3, tp_size=1)
        s.waiting_queue = list(waiting)
        s.chunked_req = None
        s.running_batch = None
        s.anchor_tails = list(tails)
        s._pp_row_chain_owed = False
        s._pp_flip_epoch = lambda: self.EPOCH
        s._pdflip_store_told_armed = False
        return s

    def _probe(self, s):
        return ppm.SchedulerPPMixin._pp_proxy_frame_pending(s, self.SLOT)

    def test_pdflip_6_94_the_end_anchor_tail_is_locatable(self):
        tail = types.SimpleNamespace(rid=RID)
        queue = [_frame(self.SLOT, self.EPOCH, RID)]
        s = self._sched(queue, tails=[tail])
        for _ in range(5):                         # past the lap cap (3): never a defer, never #1180
            self.assertIs(self._probe(s), True)
        self.assertNotIn("defer_rid", getattr(s, "_pp_row_probe_stats", {}))
        self.assertEqual(len(queue), 1, "the frame stays for the slot's own recv")

    def test_pdflip_0_20_tail_after_an_equal_early_settle(self):
        """15:21:51 PP1: ANCHOR-TAILS minted_rids=pdflip-0-20 at slot 1, the slot-2 row names it."""
        queue = [_frame(self.SLOT, self.EPOCH, "pdflip-0-20")]
        s = self._sched(queue, tails=[types.SimpleNamespace(rid="pdflip-0-20")])
        for _ in range(4):
            self.assertIs(self._probe(s), True)

    def test_a_rid_in_no_place_still_defers(self):
        queue = [_frame(self.SLOT, self.EPOCH, RID)]
        s = self._sched(queue, tails=[types.SimpleNamespace(rid="pdflip-6-93")])
        self.assertIs(self._probe(s), False)
        self.assertEqual(s._pp_row_probe_stats.get("defer_rid"), 1)


if __name__ == "__main__":
    unittest.main()
