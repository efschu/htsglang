"""W3-HOST-LEAF (NF int22, boot 1008_171755 @ 6b3bd1a6df, P 18:33:37-18:34:20Z).

THE MEASUREMENT. ``W3-ARENA SPILL n=1 need=64 released_pages=0 nodes=0
candidates=0`` on PP0/PP1/PP2 (P.log 216254-216572): the spill releases only
host copies of nodes whose KV is still on the device, and there were none off
the claimer's chain. The KV arena (6485 slots, all complete) freed nothing for
a claim (``#1427 ARENA-DROP ... freed=0``), every backup was refused (``#1421
arena_claim`` / ``parent_unbacked``), and PP1's peel stopped at two un-backed
frontier leaves with host-only children ([620] -> [621] -> [619], [576] ->
[577] ...): ``EVICT-FRONTIER-CENSUS on_frontier=8192 behind_device_child=187776``
-> ``Prefill out of memory``.

Switched on (``FLLIPER_PDFLIP_ENABLE_W3_SPILL_HOST_LEAVES``), on the local-PP
floor, the 27B line's Q-697c host-only spill (ported: pdflip/host_only_spill.py)
runs after the device-resident round: host-only H-LEAVES are secured to L3 for
every page first and then leave the tree -- their arena references go; the
claimer's host-only children included, so the un-backed claimer becomes
childless and droppable (UD). A leaf without a secured copy, with a host lock
or without page hashes stays. Off (default) / not the local-PP floor: the old
spill.
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import unittest

import torch

from flliper.srt.environ import envs
from flliper.srt.mem_cache.base_prefix_cache import InsertParams
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.pdflip import pp_slot_fidelity as SF
from flliper.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture

SPAN = 16      # tokens per node (page_size 1)


class _AnchorPool:
    """The arena pool: secures what it is asked (or loses ``lose`` rows)."""

    staging_rows = 0

    def __init__(self, lose=()):
        self.arena = object()
        self.secured = []
        self.lose = set(lose)

    def secure_rows_to_l3(self, host_indices):
        n = int(host_indices.numel())
        if int(host_indices.min()) in self.lose:
            return {"pages": n, "on_disk": 0, "written": 0, "lost": n}
        self.secured.append(n)
        return {"pages": n, "on_disk": n, "written": 0, "lost": 0}


def _nodes_by_depth(cache):
    out = {}
    for n in cache._collect_all_nodes():
        if n is cache.root_node:
            continue
        d, p = 0, n
        while p is not cache.root_node:
            d += len(p.key)
            p = p.parent
        out[d] = n
    return out


def _death_shape(host_lock=0):
    """[620]-like: an un-backed device node whose two children are host-only
    (backed up, demoted), the deeper one a host leaf; arena rows from 100."""
    cache, allocator, _ = build_fixture(CacheConfig(page_size=1, components=(ComponentType.FULL,)))
    for depth in (1, 2, 3):
        n = depth * SPAN
        cache.insert(InsertParams(key=RadixKey(list(range(1, 1 + n)), None),
                                  value=allocator.alloc(n).to(dtype=torch.int64)))
    nodes = _nodes_by_depth(cache)
    top, mid, leaf = nodes[SPAN], nodes[2 * SPAN], nodes[3 * SPAN]
    for i, node in enumerate((leaf, mid)):  # demote the deepest first
        node.component_data[ComponentType.FULL].host_value = torch.arange(
            100 + i * SPAN, 100 + (i + 1) * SPAN, dtype=torch.int64)
        node.hash_value = ["h%d-%d" % (i, j) for j in range(SPAN)]   # one per page
        cache._evict_to_host(node)
    leaf.component_data[ComponentType.FULL].host_lock_ref = host_lock
    setattr(cache, SF.FLOOR_LOCAL_PP_ATTR, True)
    return cache, top, mid, leaf


def _spill(cache, pool, claimer, *, on, floor=True):
    setattr(cache, SF.FLOOR_LOCAL_PP_ATTR, floor)
    with envs.FLLIPER_PDFLIP_ENABLE_W3_SPILL_HOST_LEAVES.override(on):
        return cache._w3_arena_spill(pool, SPAN, claimer=claimer)


class TestW3SpillHostLeaves(CustomTestCase):
    def test_fixture_is_the_death_shape(self):
        cache, top, mid, leaf = _death_shape()
        self.assertFalse(top.evicted)
        self.assertFalse(top.backuped)
        self.assertTrue(mid.evicted and mid.backuped and leaf.evicted and leaf.backuped)
        self.assertTrue(cache._is_host_leaf(leaf))
        self.assertEqual(list(top.children.values()), [mid],
                         "the un-backed device node still has its host-only child")
        self.assertTrue(cache._is_device_leaf(top), "on the frontier, as [620] was")

    def test_18_34_host_only_leaves_leave_with_their_l3_copy(self):
        """The old spill: candidates=0, released 0 (P.log 216292). On: both
        host-only nodes below the claimer are secured, then leave; the claimer
        is childless (UD may drop it)."""
        cache, top, mid, leaf = _death_shape()
        pool = _AnchorPool()
        got = _spill(cache, pool, top, on=True)
        self.assertEqual(got, 2 * SPAN)
        self.assertEqual(pool.secured, [SPAN, SPAN], "every page got its L3 copy first")
        self.assertEqual(len(top.children), 0)
        self.assertNotIn(mid, cache._collect_all_nodes())
        self.assertNotIn(leaf, cache._collect_all_nodes())
        self.assertFalse(top.evicted, "the claimer itself is not touched")

    def test_switch_off_is_the_old_spill(self):
        cache, top, mid, leaf = _death_shape()
        pool = _AnchorPool()
        self.assertEqual(_spill(cache, pool, top, on=False), 0)
        self.assertEqual(pool.secured, [])
        self.assertIn(leaf, cache._collect_all_nodes())

    def test_not_the_local_pp_floor_edits_nothing(self):
        cache, top, mid, leaf = _death_shape()
        pool = _AnchorPool()
        self.assertEqual(_spill(cache, pool, top, on=True, floor=False), 0)
        self.assertEqual(pool.secured, [])
        self.assertIn(leaf, cache._collect_all_nodes())

    def test_a_page_without_l3_copy_stays(self):
        cache, top, mid, leaf = _death_shape()
        pool = _AnchorPool(lose={100})   # the leaf's rows start at 100
        self.assertEqual(_spill(cache, pool, top, on=True), 0)
        self.assertIn(leaf, cache._collect_all_nodes())
        self.assertIn(mid, cache._collect_all_nodes())

    def test_a_leaf_without_page_hashes_stays(self):
        cache, top, mid, leaf = _death_shape()
        leaf.hash_value = None
        pool = _AnchorPool()
        self.assertEqual(_spill(cache, pool, top, on=True), 0)
        self.assertIn(leaf, cache._collect_all_nodes())

    def test_a_host_locked_leaf_stays(self):
        cache, top, mid, leaf = _death_shape(host_lock=1)
        pool = _AnchorPool()
        self.assertEqual(_spill(cache, pool, top, on=True), 0)
        self.assertIn(leaf, cache._collect_all_nodes())


if __name__ == "__main__":
    unittest.main()
