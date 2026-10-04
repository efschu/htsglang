"""1210b HOSTLOCK-EVICT-TO-HOST (NF y9nf4 e23f7dff30 boot 1004_031945, 04:06:42Z): D TP1 and TP2 died at the
idle sanity walk right after the epoch-82 park ("Sanity check FAILED (1 violations across 28 nodes): mamba
host-locked node(s) on the host LRU: {1272}", D.log 263217-263265; TP0, already in its sleep leg, died on the
broken gloo pair, 263845) -> WEG2-FLIP STALL epoch=82.

Y8P-HOSTLOCK-LRU (a3f574c24e) closed two writers that filed a host-locked mamba node into the host LRU (the
mamba component's device eviction and the reclaim disown). The third one stayed: the FULL demotion
``_evict_to_host`` files every aux component that has a host copy with ``_for_each_component_lru(insert_mru,
target=HOST)`` -- without the host lock. A node whose anchor is on BOTH sides and host-locked (#1417 prefetch
pin / anchor lock, the park retracts such requests) demoted by a backed-leaf eviction (WEG2-LOADBACK-EVICT,
D-MEM-SCHED stage down, D-KV-EVICT) lands on the host LRU while locked.

Hermetic: the real UnifiedRadixCache (FULL + MAMBA) of the #1417b / Y8P tests, CPU only.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import unittest

from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.test.test_utils import CustomTestCase

from test_hostlock_node_stays_off_host_lru_y8p import _give_device_value
from test_prefetch_pin_host_lru_sanity_1417b import HEAD, _metal_form, _tree

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA


def _both_sides(cache, node, n_tokens=HEAD):
    """The pinned anchor's node is back on the device (KV and state), its host copies kept: the form a
    load-back or a retained park leaves. Size books follow, so PART 4 of the walk stays honest."""
    full = node.component_data[FULL]
    full.value = cache.token_to_kv_pool_allocator.alloc(n_tokens)
    assert full.value is not None
    cache.component_evictable_size_[FULL] += n_tokens
    _give_device_value(cache, node)
    cache.components[MAMBA]._free_mamba_value = lambda v: None  # the slot return is not under test
    cache._update_evictable_leaf_sets(node)
    cache._update_evictable_leaf_sets(node.parent)


class HostLockSurvivesEvictToHost(CustomTestCase):
    def test_full_demotion_of_a_host_locked_anchor_keeps_it_off_the_host_lru(self):
        cache = _tree()
        n249, _ = _metal_form(cache)
        _both_sides(cache, n249)
        cd = n249.component_data[MAMBA]
        self.assertEqual(cd.host_lock_ref, 1)
        cache._evict_to_host(n249, {ct: 0 for ct in cache.tree_components})
        self.assertIsNone(n249.component_data[FULL].value)
        self.assertIsNone(cd.value)
        self.assertFalse(cache.host_lru_lists[MAMBA].in_list(n249))
        cache.sanity_check()  # RED on e23f7dff30: "mamba host-locked node(s) on the host LRU: {249}"

    def test_the_last_unlock_files_the_demoted_anchor_back(self):
        cache = _tree()
        n249, _ = _metal_form(cache)
        _both_sides(cache, n249)
        cache._evict_to_host(n249, {ct: 0 for ct in cache.tree_components})
        cache.pop_prefetch_loaded_tokens("weg2-16-31")
        cache.pop_prefetch_loaded_tokens("weg2-16-38")
        self.assertEqual(n249.component_data[MAMBA].host_lock_ref, 0)
        self.assertTrue(cache.host_lru_lists[MAMBA].in_list(n249))
        cache.sanity_check()

    def test_an_unlocked_anchor_still_joins_the_host_lru_on_demotion(self):
        cache = _tree()
        n249, _ = _metal_form(cache)
        cache.pop_prefetch_loaded_tokens("weg2-16-31")
        cache.pop_prefetch_loaded_tokens("weg2-16-38")
        cache.host_lru_lists[MAMBA].remove_node(n249)
        _both_sides(cache, n249)
        self.assertEqual(n249.component_data[MAMBA].host_lock_ref, 0)
        cache._evict_to_host(n249, {ct: 0 for ct in cache.tree_components})
        self.assertTrue(cache.host_lru_lists[MAMBA].in_list(n249))
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
