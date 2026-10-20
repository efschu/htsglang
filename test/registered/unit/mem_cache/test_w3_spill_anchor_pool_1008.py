"""W3-ANCHOR-POOL (NF int20, boot 1008_135412 @ 45cdd0a290, P PP0 13:58-14:01).

THE MEASUREMENT. From the first reads after the boot the shared KV arena
(6485 slots) held only pages P's own tree referenced. ``W3-ARENA SPILL`` -- the
release made for exactly this -- printed 0 lines (0 in the 15 NF P logs before
it, too): the hybrid KV host pool is a ``HostPoolGroup`` whose ``__getattr__``
forwards the claim calls only, ``hasattr(pool, "secure_rows_to_l3")`` said
False and the spill returned 0 before its log line. ``PUBLISH-SWEEP
issued=0 refused=5130`` for the whole phase, and every L3->L2 fill of a store
hit ended at the full arena (``#1436 ARENA-GET MISS``, ``#1157 PREFETCH
REAPED req=pdflip-0-11 hit_pages=1318 completed=0``): 705k tokens the store held
were prefilled again (P hit rate 17-22 % from 13:59 to 14:01).

Switched on (``FLLIPER_PDFLIP_ENABLE_W3_SPILL_ANCHOR_POOL``) the spill goes
through the group's anchor (arena) pool -- the one ``alloc_write`` already
forwards to. Off, it stops as before.
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import types
import unittest
from unittest import mock

import torch

from flliper.srt.environ import envs
from flliper.srt.mem_cache.base_prefix_cache import InsertParams
from flliper.srt.mem_cache.memory_pool_host import HostPoolGroup
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from flliper.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

SPAN = 16      # tokens per node (page_size 1)


class _AnchorPool:
    """The arena pool behind the group: every row it is asked about is COMPLETE
    and already on disk (the write-behind secured it), so the spill may let go."""

    staging_rows = 0

    def __init__(self):
        self.arena = object()
        self.secured = []

    def secure_rows_to_l3(self, host_indices):
        n = int(host_indices.numel())
        self.secured.append(n)
        return {"pages": n, "on_disk": n, "written": 0, "lost": 0}


def _group(anchor):
    group = HostPoolGroup.__new__(HostPoolGroup)
    object.__setattr__(group, "anchor_entry", types.SimpleNamespace(host_pool=anchor))
    return group


def _tree_with_host_copies():
    """Two device-resident chains whose Full host value names arena rows: the
    pages a P read loaded (device KV + its L2 copy), the tree's references."""
    cache, allocator, _ = build_fixture(CacheConfig(page_size=1, components=(ComponentType.FULL,)))
    for first in (1, 5000):
        for depth in (1, 2):
            n = depth * SPAN
            cache.insert(InsertParams(
                key=RadixKey(list(range(first, first + n)), None),
                value=allocator.alloc(n).to(dtype=torch.int64),
            ))
    row = 100
    for node in cache._collect_all_nodes():
        if node is cache.root_node:
            continue
        k = len(node.key)
        node.component_data[ComponentType.FULL].host_value = torch.arange(row, row + k, dtype=torch.int64)
        row += k
    return cache


def _spill(cache, pool, *, on):
    released_nodes = []

    def _release(self, node, comp, target=None, tracker=None):
        hv = node.component_data[ComponentType.FULL].host_value
        released_nodes.append(node.id)
        return 0, int(hv.numel())

    with envs.FLLIPER_PDFLIP_ENABLE_W3_SPILL_ANCHOR_POOL.override(on), \
            mock.patch.object(UnifiedRadixCache, "_evict_component_and_detach_lru", _release):
        got = cache._w3_arena_spill(pool, 32)
    return got, released_nodes


class TestW3SpillThroughTheAnchorPool(CustomTestCase):
    def test_1008_135412_hybrid_group_spill_releases_the_tree_pages(self):
        """RED before the fix: the group has no secure_rows_to_l3, the spill
        answers 0 and the full arena stays full (PUBLISH-SWEEP issued=0,
        ARENA-GET MISS on every L3 fill)."""
        cache = _tree_with_host_copies()
        anchor = _AnchorPool()
        group = _group(anchor)
        self.assertFalse(hasattr(group, "secure_rows_to_l3"),
                         "the group itself still lacks the call: the anchor pool is the way")
        got, released = _spill(cache, group, on=True)
        self.assertGreaterEqual(got, 32, "W3_SPILL_MIN_PAGES worth of the tree's pages go back")
        self.assertEqual(sum(anchor.secured), got,
                         "every released page got its L3 copy through the anchor pool first")
        for node in cache._collect_all_nodes():
            if node.id in released:
                self.assertTrue(node.l3_present, "a spilled node stays on the device, store-backed")

    def test_switch_off_is_the_old_silent_stop(self):
        """27B / flip lines without the switch: nothing secured, nothing released."""
        cache = _tree_with_host_copies()
        anchor = _AnchorPool()
        got, released = _spill(cache, _group(anchor), on=False)
        self.assertEqual((got, released, anchor.secured), (0, [], []))


if __name__ == "__main__":
    unittest.main()
