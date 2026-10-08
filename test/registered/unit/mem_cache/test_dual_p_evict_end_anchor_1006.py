# SPDX-License-Identifier: Apache-2.0
"""Q-1500 V3 (desk dual-evict-oom-1006): a host-backed END anchor below a refused, un-backed device leaf
no longer keeps the dual-P eviction from paying the pool. REAL UnifiedRadixCache (FULL + MAMBA, CPU).

THE DEATH (27B NVFP4 dual P/D stages, boot ...dualstufenbar1fs10060932 @173161c595, P PP1 09:48:38Z):
  PP1 tree: [126976 un-backed device tokens] -> node 221 (2573, un-backed, depth 129549)
            -> node 220 (2 tokens, END anchor of pdflip-0-50 at 129551, KV + mamba on the host:
               '#1469 EVICT node=220 backuped=True host=True parent=221')
  '#1421 BACKUP-REFUSED why=parent_unbacked node=221' (arena fill 0.998), UD refuses 221 (children),
  Q-1500 V1 refuses the subtree for the host-backed END anchor, 'EVICT-FRONTIER-CENSUS request=1024
  delivered_before=0 ... on_frontier=2573 behind_device_child=126976', 'EXTEND-RELIEF evicted=0',
  alloc_token_slots raises 'Out of memory ... Available full tokens: 130048 (499 + 129549)' -> W17.
  PP0 had the same END anchor UN-backed and dropped it by plain UD (EVICT-UNBACKED-DROP node=220/221).

Scaled shape here: A [1..8] (the 126976), L [9..12] (node 221), E [13..14] (node 220, host-only, END
anchor with a mamba host row). The base (173161c595) delivers 0 and alloc_token_slots raises; V3 drops
E then L and pays the pool. Mutant-facing guards: a told-named END anchor, a host lock, a write in
flight keep everything; an allowed backup is a demotion, never a drop; the flip / NF / INT8 form,
dual D and the switch at 0 are unchanged; the V2 trim census keeps the V1 guard.
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import os
import unittest
from array import array

import torch

from flliper.srt.managers.schedule_batch import Req
from flliper.srt.mem_cache import common as C
from flliper.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.sampling.sampling_params import SamplingParams
from flliper.srt.pdflip import dual_arena_spill as DAS
from flliper.srt.pdflip import pp_slot_fidelity as SF
from flliper.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
DUAL_ENV = {
    "FLLIPER_PDFLIP_DUAL_LAYOUT": "1",
    "FLLIPER_PDFLIP_GROUP": "P",
    "FLLIPER_PDFLIP_DUAL_P_KV_MAX_TOKENS": "32768",
}
_KEYS = tuple(DUAL_ENV) + (SF.ENV, "FLLIPER_PDFLIP_DUAL_UD_HOST_CHILDREN", "FLLIPER_EVICT_FRONTIER_REPAIR")
E_MAMBA_ROW = 4242


class _WriteBackController:
    write_policy = "write_back"

    def append_host_mem_release(self, *args, **kwargs):
        return None


class _MambaHostPool:
    """Records every host row handed back (the arena reference release on the metal)."""

    def __init__(self):
        self.freed = []

    def free(self, idx):
        self.freed.extend(int(x) for x in idx.reshape(-1))
        return len(idx)


class _ToldHold:
    """The y9d4 hold's interface as ``UnifiedRadixCache._pdflip_told_held`` reads it."""

    def __init__(self, depths):
        self._d = depths

    def depths(self, tick=False):
        return dict(self._d)


def _insert(cache, alloc, r2t, tokens, rid):
    value = alloc.alloc(len(tokens))
    assert value is not None
    req = Req(rid=rid, origin_input_text="", origin_input_ids=array("q"),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    r2t.alloc([req])
    cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=value,
                              mamba_value=req.mamba_pool_idx.unsqueeze(0)))


def _host_insert(cache, tokens):
    key = RadixKey(array("q", tokens))
    host = torch.arange(7000, 7000 + len(tokens), dtype=torch.int64)
    return cache._insert_helper_host(cache.root_node, key, host, [f"h{t}" for t in tokens])


def _metal_tree(dual=True):
    """A [1..8] -> L [9..12] on the device (un-backed), E [13..14] host-only END anchor below L."""
    for k in _KEYS:
        os.environ.pop(k, None)
    cfg = CacheConfig(page_size=1, components=(FULL, MAMBA))
    cache, alloc, r2t = build_fixture(cfg)
    cache.cache_controller = _WriteBackController()
    _insert(cache, alloc, r2t, list(range(1, 9)), "r-a")
    _insert(cache, alloc, r2t, list(range(1, 13)), "pdflip-0-50")
    res = _host_insert(cache, list(range(1, 15)))
    e = res.inserted_host_node
    assert e is not None
    pool = _MambaHostPool()
    cache.components[MAMBA]._mamba_pool_host = pool
    e._pdflip_end_anchor = True                                         # #1481 mark (END-ANCHOR ok=True)
    e.pdflip_anchor_rid = "pdflip-0-50"
    e.component_data[MAMBA].host_value = torch.tensor([E_MAMBA_ROW], dtype=torch.int64)
    cache.host_lru_lists[MAMBA].insert_mru(e)
    cache.write_backup = lambda node, write_back=False, kv_only_if_mamba_refused=False: 0  # arena full
    setattr(cache, SF.FLOOR_LOCAL_PP_ATTR, True)                      # TP=1 / PP=3: the floor is local
    if dual:
        os.environ.update(DUAL_ENV)                                   # after the fixture (saver alloc)
    by_first = {}
    for n in cache._collect_all_nodes():
        if n is not cache.root_node:
            by_first[int(n.key.token_ids[0])] = n
    return cache, alloc, pool, by_first[1], by_first[9], by_first[13]


class DualPEvictEndAnchor1006(CustomTestCase):
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

    def _live(self, cache):
        return {x.id for x in cache._collect_all_nodes()}

    # ---- the premise: the metal state, built by the tree's own writers ----

    def test_premise_is_the_metal_state(self):
        cache, _, _, a, l, e = _metal_tree()
        self.assertIs(e.parent, l)
        self.assertIs(l.parent, a)
        self.assertFalse(a.backuped or l.backuped)                    # 126976 + 221 un-backed
        self.assertTrue(e.backuped and e.evicted)                     # 220: host-only, backed
        self.assertTrue(cache._is_device_leaf(l))                     # the one frontier leaf (2573)
        self.assertFalse(cache._is_device_leaf(a))                    # behind_device_child
        self.assertTrue(cache._is_host_leaf(e))
        self.assertFalse(SF.unbacked_drop_allowed(cache, l))          # UD: L has a child

    # ---- the defect (RED on 173161c595) ----

    def test_eviction_pays_the_pool_through_a_host_backed_end_anchor(self):
        cache, alloc, pool, a, l, e = _metal_tree()
        before = alloc.available_size()
        ev_before = cache.component_evictable_size_[FULL]
        res = cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(res.num_tokens_evicted, 4)                   # L's 4 device rows; E's host rows not counted
        self.assertEqual(alloc.available_size() - before, 4)
        self.assertEqual(ev_before - cache.component_evictable_size_[FULL], 4)
        live = self._live(cache)
        self.assertNotIn(l.id, live)
        self.assertNotIn(e.id, live)
        self.assertIn(a.id, live)                                     # only what was asked: A is the next leaf
        self.assertTrue(cache._is_device_leaf(a))
        self.assertEqual(pool.freed, [E_MAMBA_ROW])                   # the END anchor's reference went back once
        self.assertFalse(cache.host_lru_lists[MAMBA].in_list(e))
        cache.sanity_check()

    def test_alloc_token_slots_no_longer_raises_on_the_metal_state(self):
        """1006 PP1: 'Try to allocate 1024 tokens ... full_available_size=499 + full_evictable_size_=129549'."""
        cache, alloc, pool, a, l, e = _metal_tree()
        hold = alloc.alloc(alloc.available_size() - 2)                # the pool is full but for 2 rows
        self.assertIsNotNone(hold)
        self.assertEqual(alloc.available_size(), 2)
        out = C.alloc_token_slots(cache, 10)                          # needs L (4) and A (8)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 10)
        self.assertEqual(len(cache.root_node.children), 0)
        cache.sanity_check()

    def test_the_log_names_the_yielded_end_anchor(self):
        cache, *_ = _metal_tree()
        cache.pp_rank = 1
        with self.assertLogs("flliper.srt.pdflip.pp_slot_fidelity", level="WARNING") as cm:
            cache.evict(EvictParams(num_tokens=4))
        line = next(m for m in cm.output if "EVICT-UNBACKED-DROP SUBTREE" in m)
        self.assertIn("end_anchors=1", line)
        self.assertIn("end_anchors_yielded=1", line)
        self.assertIn("rank=pp1", line)

    # ---- danger direction 1: an allowed backup is a demotion, never a drop ----

    def test_an_allowed_backup_demotes_and_keeps_the_end_anchor(self):
        cache, alloc, pool, a, l, e = _metal_tree()

        def backup_ok(node, write_back=False, kv_only_if_mamba_refused=False):
            cd = node.component_data[FULL]
            cd.host_value = torch.arange(9000, 9000 + len(cd.value), dtype=torch.int64)
            return len(cd.value)

        cache.write_backup = backup_ok
        res = cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(res.num_tokens_evicted, 4)
        live = self._live(cache)
        self.assertIn(l.id, live)                                     # L stays, host-backed
        self.assertTrue(l.evicted and l.backuped)
        self.assertIn(e.id, live)                                     # the END anchor is untouched
        self.assertIsNotNone(e.component_data[MAMBA].host_value)
        self.assertEqual(pool.freed, [])

    # ---- danger direction 2: nothing held, locked or named is dropped ----

    def _assert_kept(self, cache, alloc, pool, a, l, e):
        before = alloc.available_size()
        res = cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(res.num_tokens_evicted, 0)                   # the base behaviour
        self.assertEqual(alloc.available_size(), before)
        live = self._live(cache)
        self.assertIn(l.id, live)
        self.assertIn(e.id, live)
        self.assertEqual(pool.freed, [])
        self.assertIsNone(SF.unbacked_drop_subtree(cache, l))

    def test_a_told_named_end_anchor_keeps_the_subtree(self):
        cache, alloc, pool, a, l, e = _metal_tree()
        cache._pdflip_told_hold = _ToldHold({14: ["pdflip-0-50"]})       # END depth of E == a standing told
        self._assert_kept(cache, alloc, pool, a, l, e)

    def test_a_told_at_another_depth_does_not_keep_it(self):
        cache, alloc, pool, a, l, e = _metal_tree()
        cache._pdflip_told_hold = _ToldHold({8: ["pdflip-0-50"]})        # 1006: told=32768, END at 129551
        self.assertIsNotNone(SF.unbacked_drop_subtree(cache, l))
        self.assertEqual(cache.evict(EvictParams(num_tokens=4)).num_tokens_evicted, 4)

    def test_a_host_locked_end_anchor_keeps_the_subtree(self):
        cache, alloc, pool, a, l, e = _metal_tree()
        e.component_data[MAMBA].host_lock_ref += 1                    # a load-back / prefetch pin
        try:
            self._assert_kept(cache, alloc, pool, a, l, e)
        finally:
            e.component_data[MAMBA].host_lock_ref -= 1

    def test_an_end_anchor_write_in_flight_keeps_the_subtree(self):
        cache, alloc, pool, a, l, e = _metal_tree()
        cache.ongoing_write_through[e.id] = object()
        self.assertIsNone(SF.unbacked_drop_subtree(cache, l))
        cache.ongoing_write_through.pop(e.id)
        cache._pdflip_direct_mamba_rows = {e.id: object()}              # #1427 direct mamba rows in flight
        self.assertIsNone(SF.unbacked_drop_subtree(cache, l))

    def test_a_device_locked_leaf_is_never_a_candidate(self):
        cache, alloc, pool, a, l, e = _metal_tree()
        cache.inc_lock_ref(l)                                         # a running request on the leaf
        self.assertFalse(cache._is_device_leaf(l))
        res = cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(res.num_tokens_evicted, 0)
        self.assertIn(e.id, self._live(cache))
        self.assertEqual(pool.freed, [])

    def test_a_host_backed_end_anchor_leaf_itself_yields_unless_told(self):
        """The refused leaf is itself an END anchor with a mamba host row (KV un-backed): same rule."""
        cache, alloc, pool, a, l, e = _metal_tree()
        l._pdflip_end_anchor = True
        l.component_data[MAMBA].host_value = torch.tensor([777], dtype=torch.int64)
        self.assertIsNotNone(SF.unbacked_drop_subtree(cache, l))
        cache._pdflip_told_hold = _ToldHold({12: ["pdflip-0-50"]})       # END depth of L
        self.assertIsNone(SF.unbacked_drop_subtree(cache, l))

    # ---- danger direction 4: every other form is the base behaviour ----

    def test_flip_nf_int8_form_is_unchanged(self):
        cache, alloc, pool, a, l, e = _metal_tree(dual=False)        # floor local (NF runs TP=1/PP>1)
        self._assert_kept(cache, alloc, pool, a, l, e)

    def test_dual_d_is_unchanged(self):
        cache, alloc, pool, a, l, e = _metal_tree()
        os.environ["FLLIPER_PDFLIP_GROUP"] = "D"
        self._assert_kept(cache, alloc, pool, a, l, e)

    def test_switch_zero_is_unchanged(self):
        cache, alloc, pool, a, l, e = _metal_tree()
        os.environ["FLLIPER_PDFLIP_DUAL_UD_HOST_CHILDREN"] = "0"
        self._assert_kept(cache, alloc, pool, a, l, e)

    def test_pp0_mode_keeps_it_on_a_follower(self):
        cache, alloc, pool, a, l, e = _metal_tree()
        os.environ["FLLIPER_PDFLIP_DUAL_UD_HOST_CHILDREN"] = "pp0"
        cache.pp_rank = 1
        self._assert_kept(cache, alloc, pool, a, l, e)

    def test_not_the_local_pp_floor_is_unchanged(self):
        cache, alloc, pool, a, l, e = _metal_tree()
        setattr(cache, SF.FLOOR_LOCAL_PP_ATTR, False)
        self._assert_kept(cache, alloc, pool, a, l, e)

    def test_the_v2_trim_census_keeps_the_v1_guard(self):
        """dual_arena_spill._reason still names the END anchor blocked (the trim never takes it)."""
        cache, alloc, pool, a, l, e = _metal_tree()
        self.assertEqual(DAS._reason(cache, e, object(), set()), "blocked_end_anchor")
        self.assertTrue(SF._subtree_blocked(cache, e, cache.ongoing_write_through))
        self.assertFalse(SF._subtree_blocked(cache, e, cache.ongoing_write_through, end_anchor_yield=True))


if __name__ == "__main__":
    unittest.main()
