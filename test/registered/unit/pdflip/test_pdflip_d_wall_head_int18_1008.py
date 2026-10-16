# SPDX-License-Identifier: Apache-2.0
"""PW-R: the D head the backup wall makes unservable is answered W50, not held.

NF int18, 08.10., D log boot_weg2_dkrnfint4h6ablxcbar1dauer10081149_e69f28a7f6_
1008_115011.D.log TP0:

* 12:26:50Z ``H105d FORM-A-CUT LOAD-BACK ROOM rid=pdflip-38-135 kv_rows=179136 ...
  available=1408 evicted=0 reported_evictable=430528`` and the same for
  pdflip-38-153 (kv_rows=86144); 12:29:51Z ``EVICT-FRONTIER-CENSUS request=88064
  delivered_before=0 delivered_after_repair=0 reported_evictable=523392
  aux_locked={}``. The last decode round ran 12:27:36 (``#running-req: 1``
  before); from there D made no progress until the process was ended
  (12:29:52) -- nothing it held could leave the device.
* 12:12:54-12:17:30Z the first form: pdflip-28-97 (182428 tokens) waited to the
  flip while each round re-ran the futile peel.

RED on 265e273594 (the free basis alone): no module, no wall exit -- the head
stays queued. GREEN: when every rank finds the head unservable under a measured
wall, it is handed back with the named W50 whose extent the front parses as the
device rows; without a measured wall nothing changes.
"""

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.mem_cache import evict_frontier_census as EF  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    BASE_COMPONENT_TYPE as FULL,
)
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402

HEAD = "pdflip-38-135"
PROMPT = 182886  # 'PDFLIP X-EXACT-PRICE rid=pdflip-38-135 ... tokens=182886' (front)
AVAILABLE = 1408  # 'available=1408' (12:26:50Z)
REPORTED = 430528  # 'reported_evictable=430528'


class _Tree:
    """The count methods are the REAL UnifiedRadixCache ones (read on the class)."""

    evictable_size = UnifiedRadixCache.evictable_size
    deliverable_evictable_size = UnifiedRadixCache.deliverable_evictable_size
    payable_evictable_size = getattr(UnifiedRadixCache, "payable_evictable_size", None)

    def __init__(self, reported=REPORTED):
        self.component_evictable_size_ = {FULL: reported}
        self.component_protected_size_ = {FULL: 0}
        self.ongoing_write_through = {}
        self.cache_controller = types.SimpleNamespace(
            mem_pool_host=types.SimpleNamespace(available_size=lambda: 0))


def _walled():
    tree = _Tree()
    EF.note_peel_short(tree, FULL)  # what FullComponent.drive_eviction records on a short peel
    return tree


def _req(rid=HEAD, n=PROMPT):
    return types.SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(n)),
                                 prefix_indices=torch.empty(0, dtype=torch.int64))


class _MinGroup:
    def __init__(self, peers=(True, True)):
        self.peers, self.calls = peers, 0

    def __call__(self, flags):
        self.calls += 1
        return [int(bool(flags[0]) and all(self.peers))]


class WallHeadTest(unittest.TestCase):
    def test_12_26_50_unservable_head_is_handed_back(self):
        from flliper.srt.pdflip import d_wall_head as W

        head = _req()
        gm = _MinGroup()
        with self.assertLogs(W.logger, level="WARNING") as cm:
            got = W.pick_unservable_head(head_rid=HEAD, waiting=[head, _req("pdflip-38-153", 102400)],
                                         tree=_walled(), available=AVAILABLE, group_min=gm)
        self.assertIs(got, head)
        self.assertEqual(gm.calls, 1)
        self.assertEqual(W.device_need_rows(head), PROMPT)
        self.assertIn("PDFLIP D-WALL-HEAD rid=pdflip-38-135 need_rows=182886 available=1408", cm.output[0])

    def test_no_measured_wall_no_hand_back(self):
        """The same numbers without a short peel: the head waits as before (the
        evictable room is payable), and the group vote is still entered alike."""
        from flliper.srt.pdflip import d_wall_head as W

        gm = _MinGroup()
        self.assertIsNone(W.pick_unservable_head(head_rid=HEAD, waiting=[_req()], tree=_Tree(),
                                                 available=AVAILABLE, group_min=gm))
        self.assertEqual(gm.calls, 1)

    def test_a_head_that_fits_the_payable_room_waits(self):
        from flliper.srt.pdflip import d_wall_head as W

        small = _req(n=1000)
        self.assertIsNone(W.pick_unservable_head(head_rid=HEAD, waiting=[small], tree=_walled(),
                                                 available=AVAILABLE, group_min=_MinGroup()))

    def test_one_rank_with_room_keeps_it_queued_for_the_group(self):
        from flliper.srt.pdflip import d_wall_head as W

        self.assertIsNone(W.pick_unservable_head(head_rid=HEAD, waiting=[_req()], tree=_walled(),
                                                 available=AVAILABLE, group_min=_MinGroup((True, False))))

    def test_no_head_no_vote(self):
        from flliper.srt.pdflip import d_wall_head as W

        gm = _MinGroup()
        self.assertIsNone(W.pick_unservable_head(head_rid=None, waiting=[_req()], tree=_walled(),
                                                 available=AVAILABLE, group_min=gm))
        self.assertEqual(gm.calls, 0)


class WallRefusalMessageTest(unittest.TestCase):
    def test_the_front_parses_the_device_rows_as_the_extent(self):
        """The W50 carries the marker the front re-routes on and the device rows
        as its extent (the front prices the re-route at what D could not hold,
        so FLIP-ECONOMICS does not hold a 182k request for its 358 uncached)."""
        from flliper.srt.managers.scheduler import Scheduler
        from flliper.srt.pdflip import front as F

        sent = []
        stub = types.SimpleNamespace(
            server_args=types.SimpleNamespace(tp_prefill_max_tokens=12288),
            waiting_queue=[],
            tree_cache=None,
            enable_hicache_storage=False,
            enable_hierarchical_cache=False,
            ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
                send_output=lambda out, req: sent.append(out))),
            pdflip_uncached_extent=lambda req, hi: 358,
        )
        req = _req()
        req.stream = False
        req.time_stats = types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **kw: None))
        stub.waiting_queue = [req]
        with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_GROUP": "D"}):
            Scheduler._pdflip_answer_x_refusals(stub, [req], None, wall_rows={id(req): PROMPT})
        self.assertEqual(stub.waiting_queue, [])
        msg = sent[0].finished_reason["message"]
        self.assertTrue(F.x_refusal_marker_in(msg))
        self.assertEqual(int(F._D_EXTENT_RE.search(msg).group(1)), PROMPT)
        self.assertIn("D-WALL-HEAD", msg)


if __name__ == "__main__":
    unittest.main()
