"""efeu-TP14: CPU unit tests for the HiCache host staging ring
(patch_host_staging_ring.py) on the UnifiedRadixCache tree logic.

The host pool is a capacity-limited fake; the tree, its components, eviction
and the prefetch admission are the real serving-tree code.
Run (no GPU needed):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=<serving tree>/python python -m pytest -q test_host_staging_ring.py
with test_unified_radix_cache_unittest.py importable (fixtures).
"""
import os
import sys
import unittest
from array import array
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import test_unified_radix_cache_unittest as urc
except ImportError:  # local copy name
    import test_urc as urc

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)

FULL = ComponentType.FULL


class FakeHostPool:
    def __init__(self, cap):
        self.cap = cap
        self.free_slots = list(range(cap))

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, n):
        if n > len(self.free_slots):
            return None
        out, self.free_slots = self.free_slots[:n], self.free_slots[n:]
        return torch.tensor(out, dtype=torch.int64)

    def free(self, idx):
        self.free_slots.extend(int(i) for i in idx)

    def clear(self):
        self.free_slots = list(range(self.cap))


class FakeCC:
    write_policy = "write_through"

    def __init__(self, cap):
        self.mem_pool_host = FakeHostPool(cap)
        self.enable_storage = True
        self.prefetch_tokens_occupied = 0
        self.prefetch_capacity_limit = 1 << 30
        self.prefetched = []
        self.prefetch_revoke_queue = Queue()
        self.ack_backup_queue = Queue()
        self.host_mem_release_queue = Queue()
        self.extra_host_mem_release_queues = {}

    def write(self, device_indices, node_id=None, extra_pools=None):
        return self.mem_pool_host.alloc(len(device_indices))

    def prefetch_rate_limited(self):
        return False

    def prefetch(self, request_id, host_indices, new_input_tokens, last_hash=None,
                 prefix_keys=None, extra_pools=None):
        self.prefetched.append((request_id, len(host_indices)))
        return mock.Mock()


class HostStagingRingTest(unittest.TestCase):
    cfg = urc.CacheConfig(page_size=1, kv_size=256, max_context_len=256)

    def setUp(self):
        os.environ["SGLANG_HICACHE_HOST_STAGING_RING"] = "1"
        with mock.patch.object(urc, "get_device", return_value="cpu"):
            self.cache, self.alloc, _ = urc.build_fixture(self.cfg)
        self.cc = FakeCC(cap=40)
        self.cache.cache_controller = self.cc
        self.cache.components[FULL]._full_kv_pool_host = self.cc.mem_pool_host
        self.cache.write_through_threshold = 1 << 30  # backups only when asked
        self.cache.prefetch_threshold = 1

    # -- helpers -----------------------------------------------------------
    def insert(self, tokens):
        v = self.alloc.alloc(len(tokens))
        self.cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=v))
        m = self.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        return m.last_device_node

    def backup(self, node, persisted=True):
        n = self.cache.write_backup(node)
        if n > 0:
            for nid in list(self.cache.ongoing_write_through):
                self.cache._finish_write_through_ack(nid)
            node.l3_persisted = persisted
        return n

    def chain(self, start, n_nodes, length):
        """n_nodes nodes of `length` tokens each, as one growing path."""
        toks, nodes = [], []
        for i in range(n_nodes):
            toks = toks + list(range(start + i * length, start + (i + 1) * length))
            nodes.append(self.insert(toks))
        return toks, nodes

    # -- tests -------------------------------------------------------------
    def test_full_host_of_resident_copies_blocks_backup_without_the_ring(self):
        os.environ["SGLANG_HICACHE_HOST_STAGING_RING"] = "0"
        _, nodes = self.chain(1000, 4, 10)  # 40 tokens = host capacity
        for n in nodes:
            self.assertGreater(self.backup(n), 0)
        self.assertEqual(self.cc.mem_pool_host.available_size(), 0)
        other = self.insert(list(range(5000, 5010)))
        self.assertEqual(self.cache.write_backup(other), 0)  # upstream: L3 starves

    def test_ring_drops_persisted_resident_copies_lru_first(self):
        _, nodes = self.chain(1000, 4, 10)
        for n in nodes:
            self.backup(n)
        other = self.insert(list(range(5000, 5010)))
        self.assertEqual(self.cache.write_backup(other), 10)
        dropped = [n for n in nodes if not n.backuped]
        self.assertEqual(len(dropped), 1)
        self.assertFalse(dropped[0].evicted)  # still on the device

    def test_unpersisted_or_locked_copies_are_never_dropped(self):
        _, nodes = self.chain(1000, 4, 10)
        for n in nodes:
            self.backup(n, persisted=False)
        other = self.insert(list(range(5000, 5010)))
        self.assertEqual(self.cache.write_backup(other), 0)
        for n in nodes:
            n.l3_persisted = True
        lock = self.cache.inc_lock_ref(nodes[-1])  # locks the whole path
        self.assertEqual(self.cache.write_backup(other), 0)
        self.cache.dec_lock_ref(nodes[-1], lock.to_dec_params())
        self.assertEqual(self.cache.write_backup(other), 10)

    def test_long_path_tail_can_be_staged_past_host_capacity(self):
        toks, nodes = self.chain(1000, 4, 10)
        for n in nodes:
            self.backup(n)
        staged = 0
        for i in range(4):  # the conversation grows by 4 x 10 more tokens
            toks = toks + list(range(2000 + i * 10, 2010 + i * 10))
            tail = self.insert(toks)
            staged += self.backup(tail)
        self.assertEqual(staged, 40)  # 80-token path through a 40-token host

    def test_tombstone_keeps_host_only_children_reachable_for_cleanup(self):
        toks, nodes = self.chain(1000, 2, 10)
        parent, child = nodes
        self.backup(parent)
        self.backup(child)
        # drop the parent's host copy (resident + persisted), child keeps its own
        self.cache._drop_resident_host_backups(10, protected={child}, reason="t")
        self.assertFalse(parent.backuped)
        self.assertTrue(child.backuped)
        self.cache.evict(EvictParams(num_tokens=20))
        self.assertTrue(child.evicted and child.backuped)  # demoted to host
        self.assertTrue(parent.evicted and not parent.backuped)  # tombstone
        self.assertIs(child.parent, parent)
        self.assertIn(parent, parent.parent.children.values())
        # host eviction of the child sweeps the tombstone away
        self.cache.evict_host(10)
        self.assertNotIn(parent, self.cache.root_node.children.values())
        self.assertEqual(self.cc.mem_pool_host.available_size(), 40)

    def test_prefetch_drops_off_path_copies_instead_of_host_kv_full(self):
        _, nodes = self.chain(1000, 4, 10)
        for n in nodes:
            self.backup(n)
        self.cache.enable_storage = True
        new = array("q", range(7000, 7030))
        self.cache.prefetch_from_storage("r1", self.cache.root_node, new)
        self.assertEqual(self.cc.prefetched, [("r1", 30)])

    def test_prefetch_never_drops_its_own_path(self):
        toks, nodes = self.chain(1000, 4, 10)
        for n in nodes:
            self.backup(n)
        self.cache.enable_storage = True
        # request continues the cached path from nodes[0]: nodes[1..3] are on it
        self.cache.prefetch_from_storage("r2", nodes[0], array("q", toks[10:] + [9, 9]))
        for n in nodes[1:]:
            self.assertTrue(n.backuped)

    def test_storage_ack_marks_persisted_only_when_complete(self):
        _, nodes = self.chain(1000, 2, 10)
        for n in nodes:
            self.backup(n, persisted=False)
        for i, (n, done) in enumerate(((nodes[0], 10), (nodes[1], 7))):
            self.cache.ongoing_backup[i] = (
                n, self.cache.inc_host_lock_ref(n).to_dec_params())
            self.cc.ack_backup_queue.put(SimpleNamespace(id=i, completed_tokens=done))
        self.cache._drain_storage_control_queues_impl(0, None, 0, None, False)
        self.assertTrue(getattr(nodes[0], "l3_persisted", False))
        self.assertFalse(getattr(nodes[1], "l3_persisted", False))


if __name__ == "__main__":
    unittest.main()
