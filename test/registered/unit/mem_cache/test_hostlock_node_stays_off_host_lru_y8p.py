"""Y8P-HOSTLOCK-LRU (NF y8p, 03.10. 08:57:45Z): both D TP1/TP2 ranks died at the idle sanity walk right
after the park of the D>P flip ("Sanity check FAILED ... mamba host-locked node(s) on the host LRU: {266}",
RANK-DEATH lifecycle=dead; the front's quiesce then waited 12.6 s on a dead D -> WEG2-FLIP STALL).

A host-locked node (#1417 prefetch pin) is OFF the host LRU by design (#1417b); the last host unlock files
it. Two writers filed a host-only node into the host LRU without asking for the host lock: the mamba
component's device eviction (`evict_component(DEVICE)`) and the reclaim's `_disown_reclaimed_value`.

Hermetic: the real UnifiedRadixCache (FULL + MAMBA) of the #1417b test, CPU only.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import unittest

import torch

from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    EvictLayer,
    ComponentType,
)
from sglang.test.test_utils import CustomTestCase

from test_prefetch_pin_host_lru_sanity_1417b import _metal_form, _tree

MAMBA = ComponentType.MAMBA


def _give_device_value(cache, node, slot=3):
    """The pinned anchor ALSO holds a device slot (the park retracts a request whose state is on both
    sides); the size books follow so PART 4 of the walk stays honest."""
    cd = node.component_data[MAMBA]
    cd.value = torch.tensor([slot], dtype=torch.int64)
    cache.component_evictable_size_[MAMBA] += 1
    cache.lru_lists[MAMBA].insert_mru(node)
    return cd


class HostLockStaysOffHostLru(CustomTestCase):
    def test_device_eviction_of_a_host_locked_anchor_keeps_it_off_the_host_lru(self):
        cache = _tree()
        n249, _ = _metal_form(cache)
        cd = _give_device_value(cache, n249)
        self.assertEqual(cd.host_lock_ref, 1)
        comp = cache.components[MAMBA]
        comp._free_mamba_value = lambda v: None  # the slot return is not under test
        comp.evict_component(n249, EvictLayer.DEVICE)
        cache.lru_lists[MAMBA].remove_node(n249)  # the eviction driver's half
        self.assertIsNone(cd.value)
        self.assertFalse(cache.host_lru_lists[MAMBA].in_list(n249))
        cache.sanity_check()  # RED before the fix: "mamba host-locked node(s) on the host LRU"

    def test_the_last_unlock_files_it_back(self):
        cache = _tree()
        n249, _ = _metal_form(cache)
        _give_device_value(cache, n249)
        comp = cache.components[MAMBA]
        comp._free_mamba_value = lambda v: None
        comp.evict_component(n249, EvictLayer.DEVICE)
        cache.lru_lists[MAMBA].remove_node(n249)  # the eviction driver's half
        cache.pop_prefetch_loaded_tokens("weg2-16-31")
        cache.pop_prefetch_loaded_tokens("weg2-16-38")
        self.assertEqual(n249.component_data[MAMBA].host_lock_ref, 0)
        self.assertTrue(cache.host_lru_lists[MAMBA].in_list(n249))
        cache.sanity_check()

    def test_reclaim_disown_of_a_host_locked_anchor_keeps_it_off_the_host_lru(self):
        cache = _tree()
        n249, _ = _metal_form(cache)
        cd = _give_device_value(cache, n249)
        cache._disown_reclaimed_value(n249, MAMBA)
        self.assertIsNone(cd.value)
        self.assertFalse(cache.host_lru_lists[MAMBA].in_list(n249))
        cache.sanity_check()

    def test_an_unlocked_host_only_anchor_still_joins_the_host_lru(self):
        cache = _tree()
        n249, _ = _metal_form(cache)
        cache.pop_prefetch_loaded_tokens("weg2-16-31")
        cache.pop_prefetch_loaded_tokens("weg2-16-38")
        cache.host_lru_lists[MAMBA].remove_node(n249)
        cd = _give_device_value(cache, n249)
        cache._disown_reclaimed_value(n249, MAMBA)
        self.assertEqual(cd.host_lock_ref, 0)
        self.assertTrue(cache.host_lru_lists[MAMBA].in_list(n249))
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
