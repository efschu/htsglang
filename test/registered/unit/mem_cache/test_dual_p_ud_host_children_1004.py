# SPDX-License-Identifier: Apache-2.0
"""Dual-P UD extension (V1, desk analysis 1290, y9d3 P PP0 pool OOM 07:18:49Z) on the REAL UnifiedRadixCache
(FULL + MAMBA, CPU).

THE DEATH CHAIN (1290 Gl. 3-5): the shared arena is full -> ``write_backup`` of the last device leaf (node 266,
parent 265 also un-backed) is refused (#1421 arena_claim, 14x) -> the leaf stays on the device -> UD
(``pp_slot_fidelity.unbacked_drop_allowed``) does not drop it because it has CHILDREN -> eviction delivers 22
of 1024 -> ``alloc_token_slots`` raises -> rank death.

THE HYPOTHESIS (1290 Gl. 4, no tree dump in the log): the children of 266 are HOST-ONLY nodes. This file
builds that state from the tree's OWN writers and shows (1) it is reachable under ``write_back``
(``_insert_helper_host`` attaches a host-only tail under an un-backed device parent: the #841 gate is armed
only for non-write_back policies), (2) the tree calls such a node a D-leaf (``_is_device_leaf`` asks only for
"no child with a DEVICE value"), (3) UD refuses it and the eviction delivers nothing -- the defect --, and (4)
what the V1 extension does (dual P only): the leaf and its host-only subtree go, the pool is paid, the
books stay sane.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import os
import unittest
from array import array

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.weg2 import pp_slot_fidelity as SF
from sglang.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
PAGE = 1
DUAL_ENV = {
    "SGLANG_WEG2_DUAL_LAYOUT": "1",
    "SGLANG_WEG2_GROUP": "P",
    "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS": "32768",
}
_KEYS = tuple(DUAL_ENV) + (SF.ENV, "SGLANG_WEG2_DUAL_UD_HOST_CHILDREN", "SGLANG_EVICT_FRONTIER_REPAIR")


class _WriteBackController:
    """Only what eviction and the host insert consult: policy write_back, nothing in flight."""

    write_policy = "write_back"

    def append_host_mem_release(self, *args, **kwargs):
        return None


def _insert(cache, alloc, r2t, tokens, rid):
    value = alloc.alloc(len(tokens))
    assert value is not None
    req = Req(rid=rid, origin_input_text="", origin_input_ids=array("q"),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    r2t.alloc([req])
    cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=value,
                              mamba_value=req.mamba_pool_idx.unsqueeze(0)))


def _host_insert(cache, tokens):
    """The fetched / handed-off tail enters the tree host-only (the writer of the metal state)."""
    key = RadixKey(array("q", tokens))
    host = torch.arange(7000, 7000 + len(tokens), dtype=torch.int64)
    return cache._insert_helper_host(cache.root_node, key, host, [f"h{t}" for t in tokens])


def _tree():
    """265 -> 266 on the device (un-backed), host-only children below 266 (the y9d3 shape)."""
    for k in _KEYS:
        os.environ.pop(k, None)                                   # a second _tree() in one test
    cfg = CacheConfig(page_size=PAGE, components=(FULL, MAMBA))
    cache, alloc, r2t = build_fixture(cfg)
    cache.cache_controller = _WriteBackController()
    _insert(cache, alloc, r2t, list(range(1, 5)), "r1")           # node A (265): [1..4]
    _insert(cache, alloc, r2t, list(range(1, 9)), "r2")           # node B (266): [5..8] below A
    res = _host_insert(cache, list(range(1, 13)))                 # host-only C [9..12] below B
    assert res.inserted_host_node is not None
    _host_insert(cache, list(range(1, 17)))                       # host-only D [13..16] below C
    cache.write_backup = lambda node, write_back=False, kv_only_if_mamba_refused=False: 0  # arena claim refused
    setattr(cache, SF.FLOOR_LOCAL_PP_ATTR, True)                  # dual P: tp group of one, pp > 1
    # the dual gate is armed AFTER the fixture: with it set the pool constructor demands a saver allocation
    os.environ.update(DUAL_ENV)
    nodes = {}
    stack = [cache.root_node]
    while stack:
        n = stack.pop()
        nodes[int(n.key.token_ids[0]) if n is not cache.root_node else 0] = n
        stack.extend(n.children.values())
    return cache, alloc, nodes


class DualPUdHostChildren(CustomTestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _KEYS}
        for k in _KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ---- the hypothesis, from the tree's own writers (green on the base: it is the PREMISE) ----

    def test_premise_host_only_children_under_an_unbacked_device_parent_are_reachable(self):
        cache, _, n = _tree()
        a, b, c, d = n[1], n[5], n[9], n[13]
        self.assertFalse(a.backuped)
        self.assertFalse(b.backuped)
        self.assertIsNone(b.component_data[FULL].host_value)
        self.assertTrue(c.backuped and c.evicted)                     # host-only child under un-backed B
        self.assertTrue(d.backuped and d.evicted)
        self.assertIn(c.id, [x.id for x in b.children.values()])
        # the tree names B a D-leaf although it has children (only "no child with a device value")
        self.assertTrue(cache._is_device_leaf(b))
        self.assertFalse(cache._is_device_leaf(a))                    # A has a device child (B)
        # ... and UD (children guard) refuses to drop it
        self.assertFalse(SF.unbacked_drop_allowed(cache, b))

    # ---- the defect: eviction delivers nothing ----

    def test_eviction_with_a_refused_backup_frees_the_leaf_with_host_children(self):
        """RED on the base: 1290 Gl. 3-5 -- delivered 0, node B and its device rows stay, A stays behind it."""
        cache, alloc, n = _tree()
        a, b = n[1], n[5]
        before = alloc.available_size()
        res = cache.evict(EvictParams(num_tokens=8))
        self.assertGreaterEqual(res.num_tokens_evicted, 4)            # B's 4 device rows at least
        self.assertGreaterEqual(alloc.available_size() - before, 4)
        # B left the tree and its host-only subtree went first (never an orphan: no node of the tree
        # is left whose parent chain does not reach the root, and the books are sane)
        live = {x.id for x in cache._collect_all_nodes()}
        self.assertNotIn(b.id, live)
        self.assertNotIn(n[9].id, live)
        self.assertNotIn(n[13].id, live)
        self.assertEqual(len(b.children), 0)
        cache.sanity_check()

    def test_the_whole_chain_is_paid_leaf_then_parent(self):
        cache, alloc, n = _tree()
        before = alloc.available_size()
        res = cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(res.num_tokens_evicted, 8)                   # B, then A (no children left)
        self.assertEqual(alloc.available_size() - before, 8)
        self.assertEqual(len(cache.root_node.children), 0)
        cache.sanity_check()

    # ---- gates: the extension is dual P only, and every other form keeps the base behaviour ----

    def _evicted_with_env(self, env_edit):
        cache, alloc, n = _tree()
        env_edit()
        res = cache.evict(EvictParams(num_tokens=8))
        return cache, n, res

    def _assert_old_behaviour(self, cache, n, res):
        self.assertEqual(res.num_tokens_evicted, 0)                   # delivered nothing, as on the base
        live = {x.id for x in cache._collect_all_nodes()}
        self.assertIn(n[5].id, live)
        self.assertIn(n[9].id, live)                                  # the subtree is untouched
        self.assertIsNone(SF.unbacked_drop_subtree(cache, n[5]))

    def test_switch_zero_is_the_old_behaviour(self):
        def edit():
            os.environ["SGLANG_WEG2_DUAL_UD_HOST_CHILDREN"] = "0"
        self._assert_old_behaviour(*self._evicted_with_env(edit))

    def test_flip_nf_int8_form_is_unchanged(self):
        """No dual layout: the flip form / NF / 27B INT8 -- even with the floor local (NF runs TP=1/PP>1)."""
        def edit():
            for k in DUAL_ENV:
                os.environ.pop(k, None)
        self._assert_old_behaviour(*self._evicted_with_env(edit))

    def test_dual_d_and_uncapped_p_are_unchanged(self):
        def edit_d():
            os.environ["SGLANG_WEG2_GROUP"] = "D"
        self._assert_old_behaviour(*self._evicted_with_env(edit_d))

        def edit_nocap():
            os.environ["SGLANG_WEG2_GROUP"] = "P"
            os.environ.pop("SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS", None)
        self._assert_old_behaviour(*self._evicted_with_env(edit_nocap))

    def test_not_the_local_pp_floor_is_unchanged(self):
        """A TP group's tree (floor not local): a rank-local drop would split the replicas."""
        def edit():
            pass
        cache, alloc, n = _tree()
        setattr(cache, SF.FLOOR_LOCAL_PP_ATTR, False)
        res = cache.evict(EvictParams(num_tokens=8))
        self._assert_old_behaviour(cache, n, res)

    def test_sf_switch_zero_is_the_old_behaviour(self):
        def edit():
            os.environ[SF.ENV] = "0"
        self._assert_old_behaviour(*self._evicted_with_env(edit))

    # ---- guards: only a plain host-only subtree goes ----

    def test_a_host_locked_descendant_keeps_the_whole_subtree(self):
        """A prefetch pin / load-back on a descendant: nothing is released, the leaf stays (old behaviour)."""
        cache, alloc, n = _tree()
        n[13].component_data[FULL].host_lock_ref += 1
        res = cache.evict(EvictParams(num_tokens=8))
        self._assert_old_behaviour(cache, n, res)
        n[13].component_data[FULL].host_lock_ref -= 1

    def test_an_in_flight_write_through_keeps_the_leaf(self):
        cache, alloc, n = _tree()
        self.assertIsNotNone(SF.unbacked_drop_subtree(cache, n[5]))      # verdict without a write in flight
        cache.ongoing_write_through[n[5].id] = object()
        self.assertIsNone(SF.unbacked_drop_subtree(cache, n[5]))         # the leaf's own write: refused
        cache.ongoing_write_through.pop(n[5].id)
        cache.ongoing_write_through[n[13].id] = object()
        self.assertIsNone(SF.unbacked_drop_subtree(cache, n[5]))         # a descendant's write: refused
        cache.ongoing_write_through.pop(n[13].id)

    def test_an_unbacked_descendant_keeps_the_leaf(self):
        """Not a plain host node (no host copy): outside the verdict, never guessed at."""
        cache, alloc, n = _tree()
        n[13].component_data[FULL].host_value = None
        self.assertIsNone(SF.unbacked_drop_subtree(cache, n[5]))

    def test_two_host_branches_go_and_a_sibling_branch_of_the_parent_stays(self):
        cache, alloc, n = _tree()
        _host_insert(cache, [1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24])   # second host child of B
        sibling = _host_insert(cache, [1, 2, 3, 4, 31, 32, 33, 34])      # host-only sibling of B under A
        res = cache.evict(EvictParams(num_tokens=4))
        self.assertGreaterEqual(res.num_tokens_evicted, 4)
        live = {x.id for x in cache._collect_all_nodes()}
        self.assertNotIn(n[5].id, live)
        self.assertNotIn(n[9].id, live)
        self.assertNotIn(n[13].id, live)
        self.assertIn(sibling.inserted_host_node.id, live)               # A's other (host-only) child survives
        cache.sanity_check()

    def test_a_device_child_is_never_touched(self):
        """B has a device child: B is not a D-leaf, the child is -- the child goes by the plain UD path."""
        cache, alloc, r2t = None, None, None
        cfg = CacheConfig(page_size=PAGE, components=(FULL, MAMBA))
        cache, alloc, r2t = build_fixture(cfg)
        cache.cache_controller = _WriteBackController()
        _insert(cache, alloc, r2t, list(range(1, 5)), "r1")
        _insert(cache, alloc, r2t, list(range(1, 9)), "r2")
        cache.write_backup = lambda node, write_back=False, kv_only_if_mamba_refused=False: 0
        setattr(cache, SF.FLOOR_LOCAL_PP_ATTR, True)
        os.environ.update(DUAL_ENV)
        b = next(x for x in cache._collect_all_nodes() if x.key is not None and len(x.key) and int(x.key.token_ids[0]) == 5)
        self.assertEqual(len(b.children), 0)
        res = cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(res.num_tokens_evicted, 8)                      # childless chain: plain UD, unchanged

    # ---- review 08:31Z A1: in-flight / held states of the leaf AND of every descendant keep the subtree ----

    def test_a_split_pending_write_id_on_a_descendant_or_the_leaf_keeps_the_subtree(self):
        """After _replace_pending_write_through_node a split node's pending id is the OLD node id, so
        the ongoing_write_through lookup by node.id misses it: write_through_pending_id is consulted."""
        for key in (13, 5):
            cache, alloc, n = _tree()
            self.assertIsNotNone(SF.unbacked_drop_subtree(cache, n[5]))
            n[key].write_through_pending_id = 99999
            cache.ongoing_write_through[99999] = object()
            self.assertIsNone(SF.unbacked_drop_subtree(cache, n[5]), key)
            cache.ongoing_write_through.pop(99999)

    def test_direct_mamba_rows_in_flight_keep_the_subtree(self):
        """#1427: a direct write's mamba rows in flight (leaf or descendant) are never taken."""
        for key in (13, 5):
            cache, alloc, n = _tree()
            cache._weg2_direct_mamba_rows = {n[key].id: object()}
            self.assertIsNone(SF.unbacked_drop_subtree(cache, n[5]), key)

    def test_a_host_backed_end_anchor_yields_unless_a_told_names_it(self):
        """Q-1500 V3 (1006 PP1 W17, test_dual_p_evict_end_anchor_1006): a host-backed END anchor at or below
        the refused leaf no longer keeps the subtree -- the carrier hold owns only the RESET's rows, D's
        hand-off is kept by the arena's order (handoff_pending #243). The END anchor a standing told names
        (y9d4) still keeps it. Was (V1): kept whenever it carried a mamba host value."""

        class _Told:
            def __init__(self, depth):
                self.depth = depth

            def depths(self, tick=False):
                return {self.depth: ["r"]}

        for key, depth in ((13, 16), (5, 8)):
            cache, alloc, n = _tree()
            n[key]._weg2_end_anchor = True
            n[key].component_data[MAMBA].host_value = torch.tensor([555], dtype=torch.int64)
            self.assertIsNotNone(SF.unbacked_drop_subtree(cache, n[5]), key)
            cache._weg2_told_hold = _Told(depth)                         # a standing told at its END depth
            self.assertIsNone(SF.unbacked_drop_subtree(cache, n[5]), key)
        cache, alloc, n = _tree()
        n[13]._weg2_end_anchor = True                                    # no mamba host value
        self.assertIsNotNone(SF.unbacked_drop_subtree(cache, n[5]))

    # ---- review A2: PP0-only mode ----

    def test_pp0_mode_drops_on_pp0_only(self):
        cache, alloc, n = _tree()
        os.environ["SGLANG_WEG2_DUAL_UD_HOST_CHILDREN"] = "pp0"
        cache.pp_rank = 1
        self.assertIsNone(SF.unbacked_drop_subtree(cache, n[5]))
        self._assert_old_behaviour(cache, n, cache.evict(EvictParams(num_tokens=8)))
        cache, alloc, n = _tree()
        os.environ["SGLANG_WEG2_DUAL_UD_HOST_CHILDREN"] = "pp0"
        cache.pp_rank = 0
        self.assertIsNotNone(SF.unbacked_drop_subtree(cache, n[5]))
        self.assertEqual(cache.evict(EvictParams(num_tokens=8)).num_tokens_evicted, 8)

    # ---- review A3: the log names the rank and the end anchors ----

    def test_the_log_names_rank_and_end_anchors(self):
        cache, alloc, n = _tree()
        n[13]._weg2_end_anchor = True                                    # flagged, nothing to hold
        cache.pp_rank = 2
        with self.assertLogs("sglang.srt.weg2.pp_slot_fidelity", level="WARNING") as cm:
            cache.evict(EvictParams(num_tokens=4))
        line = next(m for m in cm.output if "EVICT-UNBACKED-DROP SUBTREE" in m)
        self.assertIn("end_anchors=1", line)
        self.assertIn("rank=pp2", line)
        self.assertIn("subtree_nodes=2", line)

    # ---- review nice-to-have: mutants M4 and M15 ----

    def test_mamba_host_rows_and_lru_of_the_descendants_are_returned(self):
        """M4: the subtree release must hand the mamba host rows back (pool stub) and take the
        descendants off the mamba host LRU -- through _evict_host_leaf, not just deleting the edge."""
        cache, alloc, n = _tree()
        comp = cache.components[MAMBA]

        class Pool:
            def __init__(self):
                self.freed = []

            def free(self, idx):
                self.freed.append(int(idx.reshape(-1)[0]))
                return len(idx)

        pool = Pool()
        comp._mamba_pool_host = pool
        for k in (9, 13):
            n[k].component_data[MAMBA].host_value = torch.tensor([100 + k], dtype=torch.int64)
            cache.host_lru_lists[MAMBA].insert_mru(n[k])
        res = cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(res.num_tokens_evicted, 8)
        self.assertEqual(sorted(pool.freed), [109, 113])
        for k in (9, 13):
            self.assertFalse(cache.host_lru_lists[MAMBA].in_list(n[k]))
        cache.sanity_check()

    def test_the_leaf_drop_itself_does_not_lean_on_the_frontier_retry(self):
        """M15: with the EF retry (FullComponent.drive_eviction) off, the plain eviction still pays
        the leaf -- the hook is what delivers, not the retry that would rescue a missing one."""
        cache, alloc, n = _tree()
        os.environ["SGLANG_EVICT_FRONTIER_REPAIR"] = "0"
        res = cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(res.num_tokens_evicted, 8)
        self.assertEqual(len(cache.root_node.children), 0)


if __name__ == "__main__":
    unittest.main()
