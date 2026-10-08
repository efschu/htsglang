# SPDX-License-Identifier: Apache-2.0
"""PUBLISH-SWEEP-BG on the REAL UnifiedRadixCache (FULL + MAMBA, CPU), review 1270 finding 8.

The mock-tree test (pdflip/test_pdflip_publish_sweep_bg_1004.py) pins the selection rules; this one runs the
real tree's `publish_unbacked_sweep(background=True)` walk (real lock_ref, parent/backuped, key length,
host LRU books) and then the demotion and the idle sanity walk that killed D on NF (host-locked mamba node
on the host LRU, 1210b/Y8P). `write_backup` itself needs the full controller/pool stack; it is replaced by
a minimal stand-in that does what the real one commits on the tree (host_value on FULL and MAMBA).
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import unittest
from array import array

import torch

from flliper.srt.managers.schedule_batch import Req
from flliper.srt.mem_cache.base_prefix_cache import InsertParams
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.sampling.sampling_params import SamplingParams
from flliper.test.test_utils import CustomTestCase

from test_prefetch_pin_host_lru_sanity_1417b import PAGE, _Controller
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA


def _fixture():
    cfg = CacheConfig(page_size=PAGE, components=(FULL, MAMBA))
    cache, alloc, r2t = build_fixture(cfg)
    cache.cache_controller = _Controller()
    return cache, alloc, r2t


def _insert(cache, alloc, r2t, tokens, rid):
    value = alloc.alloc(len(tokens))
    assert value is not None
    req = Req(rid=rid, origin_input_text="", origin_input_ids=array("q"),
              sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
    r2t.alloc([req])
    cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=value,
                              mamba_value=req.mamba_pool_idx.unsqueeze(0)))


def _fake_write_backup(cache, issued):
    """What the real write_backup commits on the tree at issue (host_value on FULL, and on MAMBA for a
    node that carries an anchor); the ack's host LRU filing follows at the node's device eviction."""
    def wb(node, write_back=False, kv_only_if_mamba_refused=False):
        if node.parent is not cache.root_node and not node.parent.backuped:
            if wb(node.parent) <= 0:
                return 0
        n = len(node.key)
        node.component_data[FULL].host_value = torch.arange(5000 + 100 * node.id,
                                                            5000 + 100 * node.id + n, dtype=torch.int64)
        mcd = node.component_data[MAMBA]
        if mcd.value is not None and mcd.host_value is None:
            mcd.host_value = torch.tensor([900 + node.id], dtype=torch.int64)
        issued.append(node.id)
        return 1
    cache.write_backup = wb


def _nodes(cache):
    out, stack = [], [cache.root_node]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children.values())
    return [n for n in out if n is not cache.root_node]


class BgSweepRealTree(CustomTestCase):
    def _two_chains(self):
        cache, alloc, r2t = _fixture()
        _insert(cache, alloc, r2t, list(range(1, 9)), "r1")          # chain 1: [1..8]
        _insert(cache, alloc, r2t, list(range(1, 13)), "r2")         # extends chain 1 by [9..12]
        _insert(cache, alloc, r2t, list(range(101, 109)), "r3")      # chain 2
        return cache

    def test_bg_walk_skips_a_locked_node_and_a_node_under_an_unbacked_parent(self):
        cache = self._two_chains()
        issued = []
        _fake_write_backup(cache, issued)
        by_first = {int(n.key.token_ids[0]): n for n in cache.root_node.children.values()}
        locked = by_first[101]
        locked.component_data[FULL].lock_ref += 1           # a running request references chain 2
        st = cache.publish_unbacked_sweep(max_issue=64, background=True, bg_max_tokens=8192)
        self.assertNotIn(locked.id, issued)
        self.assertGreaterEqual(st["skipped_bg"], 1)
        self.assertTrue(by_first[1].backuped)                # the unlocked chain went, parent first
        order = [i for i in issued]
        for n in _nodes(cache):
            if n.parent is not cache.root_node and n.backuped and n.id in order:
                self.assertLess(order.index(n.parent.id), order.index(n.id))
        locked.component_data[FULL].lock_ref -= 1
        cache.sanity_check()

    def test_flush_after_bg_publishes_the_rest_and_demotion_keeps_the_host_lru_sane(self):
        cache = self._two_chains()
        issued = []
        _fake_write_backup(cache, issued)
        by_first = {int(n.key.token_ids[0]): n for n in cache.root_node.children.values()}
        locked = by_first[101]
        locked.component_data[FULL].lock_ref += 1
        cache.publish_unbacked_sweep(max_issue=64, background=True, bg_max_tokens=8192)
        self.assertFalse(locked.backuped)                    # BG left it
        locked.component_data[FULL].lock_ref -= 1            # the flip: nothing runs
        cache.publish_unbacked_sweep(max_issue=256)          # the flush walk (background False)
        self.assertTrue(all(n.backuped for n in _nodes(cache) if n.component_data[FULL].value is not None))
        # an unlocked BG/flush-published leaf demotes cleanly and files onto the host LRU
        leaf = next(n for n in _nodes(cache) if not n.children and n is not locked)
        cache._evict_to_host(leaf, {ct: 0 for ct in cache.tree_components})
        self.assertIsNone(leaf.component_data[FULL].value)
        cache.sanity_check()

    def test_size_threshold_on_the_real_tree(self):
        cache, alloc, r2t = _fixture()
        _insert(cache, alloc, r2t, list(range(1, 9)), "r1")
        issued = []
        _fake_write_backup(cache, issued)
        cache.publish_unbacked_sweep(max_issue=64, background=True, bg_max_tokens=4)
        self.assertEqual(issued, [])                         # 8 tokens > 4: left to the flush
        cache.publish_unbacked_sweep(max_issue=64, background=True, bg_max_tokens=8)
        self.assertEqual(len(issued), 1)


    @unittest.skip("1210b host-lock case: needs d70370eec1 (HOSTLOCK-EVICT-TO-HOST, _evict_to_host guard), "
                   "not on this line yet; its own test lives in that commit. Expected RED without the fix.")
    def test_demotion_of_a_host_locked_bg_published_anchor_stays_off_the_host_lru(self):
        pass


if __name__ == "__main__":
    unittest.main()
