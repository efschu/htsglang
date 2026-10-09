# SPDX-License-Identifier: Apache-2.0
"""PW: ONE free basis for the D admission under the backup wall (NF int18, 08.10.).

D log boot_weg2_dkrnfint4h6ablxcbar1dauer10081149_e69f28a7f6_1008_115011.D.log,
rid pdflip-28-97 (182428 tokens, host-backed prefix 178112, 924 uncached):

* 12:12:55Z TP0 ``H105d FORM-A-CUT LOAD-BACK ROOM ... need=180928
  available=144832 evicted=0 reported_evictable=116544`` -- the cut asks the peel
  every pass; 12:15:00Z ``EVICT-FRONTIER-CENSUS request=31552 delivered_before=0
  delivered_after_repair=0 reported_evictable=182528 ... on_frontier=181696
  aux_locked={}`` with four ``#1427 ARENA-DROP ... freed=0 stages=i:0,ii:0,iii:0``
  beside it: the write_back leaves need a host arena slot for their backup and the
  arena has none. Per decode round host gap 5-9 ms -> 124-129 ms, decode
  100 -> 35 tok/s; at running=0 (12:27:36-12:29:52) the group stood still.
* 12:13:14Z TP0 ``SEAT-AGE KV-DISPLACE-VERDICT older=pdflip-28-97 need=182428
  free=190528 younger_running=1 basis=legacy -> fits free, nobody leaves`` -- the
  same capacity, read with the reported evictable count, said the opposite.

RED on e69f28a7f6: the short peel leaves no trace, the deliverable count stays the
reported one, the cut re-runs the peel each pass and SEAT-AGE prices the
unpayable tokens. GREEN: the short peel's remainder is KNOWN-UNPAYABLE until an
input of the wall moves; the ED count, the cut and SEAT-AGE all read it.
"""

import logging
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.managers import schedule_policy as sp  # noqa: E402
from flliper.srt.mem_cache import evict_frontier_census as EF  # noqa: E402
from flliper.srt.mem_cache.base_prefix_cache import EvictParams  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.full_component import (  # noqa: E402
    FullComponent,
)
from flliper.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    BASE_COMPONENT_TYPE as FULL,
)
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.srt.pdflip import d_park_runtime as DP  # noqa: E402

REPORTED = 182528  # 'reported_evictable=182528' (12:15:00Z TP0)
REQUEST = 31552  # 'request=31552'
KV_ROWS = 178112  # 'kv_rows=178112'
CHUNK_ROWS = 2816  # 'chunk_rows=2816' (TP0)
AVAILABLE_1439 = 150080  # 'available=150080' (12:14:39Z TP0, n=512)
TOKEN_CUT = "flliper.srt.rank_role.form_a_token_cut_active"


class _CD:
    def __init__(self, n):
        self.value = torch.arange(n)
        self.lock_ref = 0


class _Node:
    def __init__(self, n, parent=None):
        self.id = id(self)
        self.parent = parent
        self.evicted = False
        self.children = {}
        self.component_data = {FULL: _CD(n)}


class _WallTree:
    """A tree whose only frontier leaf is an un-backed write_back leaf the full
    host arena refuses: the peel pays 0 of the reported count. The count methods
    are the REAL UnifiedRadixCache ones (looked up on the class, as the readers do)."""

    deliverable_evictable_size = UnifiedRadixCache.deliverable_evictable_size
    payable_evictable_size = getattr(UnifiedRadixCache, "payable_evictable_size", None)
    evictable_size = UnifiedRadixCache.evictable_size

    def __init__(self, reported=REPORTED, arena_free=0):
        self.root_node = _Node(0)
        self.leaf = _Node(reported, self.root_node)
        self.component_evictable_size_ = {FULL: reported}
        self.component_protected_size_ = {FULL: 0}
        self.evictable_device_leaves = {self.leaf}
        self.ongoing_write_through = {}
        self.arena_free = arena_free
        self.cache_controller = types.SimpleNamespace(
            mem_pool_host=types.SimpleNamespace(available_size=lambda: self.arena_free))
        self.eviction_strategy = types.SimpleNamespace(get_priority=lambda n: 0)
        self.peels = 0

    # the peel's leaf step: the backup is refused (#1421 arena_claim), nothing freed
    def _evict_device_leaf(self, node, tracker):
        self.peels += 1

    def _collect_all_nodes(self):
        return [self.root_node, self.leaf]

    def _update_evictable_leaf_sets(self, node):
        return None

    def evict(self, params):
        comp = object.__new__(FullComponent)
        comp.cache = self
        tracker = {FULL: 0}
        comp.drive_eviction(params=params, tracker=tracker)
        return types.SimpleNamespace(num_tokens_evicted=tracker[FULL])


def _short_peel(tree):
    with mock.patch.dict(os.environ, {EF.ENV: "1"}):
        res = tree.evict(EvictParams(num_tokens=REQUEST))
    assert res.num_tokens_evicted == 0
    return tree


class BackupWallTest(unittest.TestCase):
    def test_short_peel_makes_the_reported_rest_unpayable(self):
        """12:15:00Z: delivered 0 of 31552 with 182528 reported -- the deliverable
        count is 0 until an input of the wall moves."""
        tree = _short_peel(_WallTree())
        with mock.patch.dict(os.environ, {EF.ENV_DELIVERABLE: "1"}):
            self.assertEqual(tree.deliverable_evictable_size(), 0)
            self.assertEqual(tree.evictable_size(), REPORTED, "the reported count is untouched")
            # the arena frees room (a host leaf left, a claim was reaped): measure again
            tree.arena_free = 1052 * 64
            self.assertEqual(tree.deliverable_evictable_size(), REPORTED)
            tree.arena_free = 0
            self.assertEqual(tree.deliverable_evictable_size(), 0)
            # the tree moved (a request finished into it): measure again
            tree.component_evictable_size_[FULL] += 512
            self.assertEqual(tree.deliverable_evictable_size(), REPORTED + 512)

    def test_the_wall_is_remeasured_after_its_ttl(self):
        tree = _short_peel(_WallTree())
        t0 = getattr(tree, EF.UNPAYABLE_ATTR)[2]
        self.assertEqual(EF.known_unpayable(tree, FULL, now=t0 + 0.5), REPORTED)
        self.assertEqual(EF.known_unpayable(tree, FULL, now=t0 + EF.UNPAYABLE_TTL_S + 0.1), 0)

    def test_no_short_peel_no_change(self):
        """A tree whose peel never ended short reads exactly as before."""
        tree = _WallTree()
        with mock.patch.dict(os.environ, {EF.ENV_DELIVERABLE: "1"}):
            self.assertEqual(tree.deliverable_evictable_size(), REPORTED)

    def test_cut_12_14_39_refuses_without_the_futile_peel(self):
        """The H105d cut prices what the peel can PAY: after one measured short
        peel it refuses by name and does not ask the peel again every pass."""
        tree = _short_peel(_WallTree())
        peels = tree.peels
        alloc = types.SimpleNamespace(available_size=lambda: AVAILABLE_1439)
        adder = types.SimpleNamespace(tree_cache=tree, token_to_kv_pool_allocator=alloc,
                                      rem_total_tokens=331403.776)
        req = types.SimpleNamespace(rid="pdflip-28-97", best_match_node=None)
        with mock.patch(TOKEN_CUT, return_value=True), \
                mock.patch.object(sp, "_pp_load_back_extent", return_value=KV_ROWS), \
                mock.patch.object(sp, "_h105d_load_back_kv_rows", return_value=KV_ROWS), \
                mock.patch.object(sp, "_h110_chunk_rows", return_value=CHUNK_ROWS), \
                mock.patch.object(sp, "_h110_promised_rows", return_value=0):
            for _ in range(3):
                self.assertFalse(sp._h105d_cut_load_back_room(adder, req))
        self.assertEqual(tree.peels, peels, "no peel re-run against the measured wall")


class SeatAgeSameBasisTest(unittest.TestCase):
    def _sched(self, tree, available):
        older = types.SimpleNamespace(rid="pdflip-28-97", origin_input_ids=list(range(182428)),
                                      output_ids=[], prefix_indices=torch.empty(0, dtype=torch.int64))
        return types.SimpleNamespace(
            waiting_queue=[older], tree_cache=tree,
            token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: available))

    def test_12_13_14_free_is_what_the_load_back_gets(self):
        """'need=182428 free=190528 -> fits free' was 106560 available + 83968
        reported evictable the peel could not pay: the verdict reads 106560."""
        tree = _short_peel(_WallTree(reported=83968))
        sched = self._sched(tree, 106560)
        young = types.SimpleNamespace(rid="pdflip-28-98", origin_input_ids=list(range(57856)), output_ids=[])
        with mock.patch.dict(os.environ, {EF.ENV_DELIVERABLE: "1"}), \
                self.assertLogs(DP.logger, level=logging.INFO) as cm:
            self.assertFalse(DP.kv_displace_would_fit(sched, "pdflip-28-97", [young]))
        line = [m for m in cm.output if "DISPLACE-VERDICT" in m][-1]
        self.assertIn("free=106560", line)
        self.assertIn("not even with all younger seats", line)
        self.assertNotIn("fits free", line)

    def test_no_futile_displacement_under_the_wall(self):
        """Under the wall a displaced seat's retained KV goes back to the tree as
        unpayable as the rest: displacing it frees no row, so nobody is displaced."""
        tree = _short_peel(_WallTree(reported=40000))
        sched = self._sched(tree, 106560)
        young = types.SimpleNamespace(rid="pdflip-28-98", origin_input_ids=list(range(80000)), output_ids=[])
        with mock.patch.dict(os.environ, {EF.ENV_DELIVERABLE: "1"}):
            self.assertFalse(DP.kv_displace_would_fit(sched, "pdflip-28-97", [young]))
            # the same numbers without a measured wall: the youngest is enough
            free_tree = _WallTree(reported=40000)
            self.assertTrue(DP.kv_displace_would_fit(self._sched(free_tree, 106560), "pdflip-28-97", [young]))


if __name__ == "__main__":
    unittest.main()
