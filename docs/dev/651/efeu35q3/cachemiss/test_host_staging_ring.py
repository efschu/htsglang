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
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)

FULL = ComponentType.FULL
MAMBA = ComponentType.MAMBA


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
        self.mamba_host = FakeHostPool(8)
        self.enable_storage = True
        self.prefetch_tokens_occupied = 0
        self.prefetch_capacity_limit = 1 << 30
        self.prefetched = []
        self.prefetch_revoke_queue = Queue()
        self.ack_backup_queue = Queue()
        self.host_mem_release_queue = Queue()
        self.extra_host_mem_release_queues = {}

    def write(self, device_indices, node_id=None, extra_pools=None):
        for x in extra_pools or ():
            if x.name == PoolName.MAMBA and x.device_indices is not None:
                x.host_indices = self.mamba_host.alloc(len(x.device_indices))
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
            self.cache, self.alloc, self.r2t = urc.build_fixture(self.cfg)
        self.cc = FakeCC(cap=40)
        self.cache.cache_controller = self.cc
        self.cache.components[FULL]._full_kv_pool_host = self.cc.mem_pool_host
        if MAMBA in self.cache.components:
            self.cache.components[MAMBA]._mamba_pool_host = self.cc.mamba_host
        self.cache.write_through_threshold = 1 << 30  # backups only when asked
        self.cache.prefetch_threshold = 1
        self._rid = 0
        self.dev_total = (
            self.alloc.available_size()
            + self.cache.full_evictable_size()
            + self.cache.full_protected_size()
        )

    def tearDown(self):
        # The serving scheduler runs this on every idle tick (invariant
        # checker); 19:45:27 died here. Every scenario must leave a tree it
        # accepts.
        self.cache.sanity_check()
        # device KV pool balance (the on-idle leak check): every device slot
        # is free, evictable in the tree, or protected by a lock.
        total = self.alloc.size
        if True:
            self.assertEqual(
                self.alloc.available_size()
                + self.cache.full_evictable_size()
                + self.cache.full_protected_size(),
                self.dev_total,
            )
        # host pool balance: free + held by tree nodes == capacity
        held = 0
        stack = [self.cache.root_node]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            hv = n.component_data[FULL].host_value
            if n is not self.cache.root_node and hv is not None:
                held += len(hv)
        for entry in self.cache.ongoing_prefetch.values():  # staged prefetches
            held += len(entry[2])
        self.assertEqual(self.cc.mem_pool_host.available_size() + held, 40)
        mheld = 0
        stack = [self.cache.root_node]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if n is not self.cache.root_node and MAMBA in self.cache.components:
                mv = n.component_data[MAMBA].host_value
                mheld += 0 if mv is None else len(mv)
        for entry in self.cache.ongoing_prefetch.values():  # staged prefetches
            for xfers in entry[5].values():
                for x in xfers:
                    if x.name == PoolName.MAMBA and x.host_indices is not None:
                        mheld += len(x.host_indices)
        self.assertEqual(self.cc.mamba_host.available_size() + mheld, 8)

    # -- helpers -----------------------------------------------------------
    def insert(self, tokens):
        v = self.alloc.alloc(len(tokens))
        params = InsertParams(key=RadixKey(array("q", tokens)), value=v)
        if self.cfg.has_mamba:
            req = urc.UnifiedRadixCacheSuite._make_req(self, self.r2t)
            params.mamba_value = req.mamba_pool_idx.unsqueeze(0)
        self.cache.insert(params)
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
    def assert_parent_first(self):
        """The upstream invariant v1 broke: a backed-up node's parent is
        backed up (root excepted). Checked explicitly, independent of
        sanity_check, after every ring action."""
        stack = [self.cache.root_node]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if n is self.cache.root_node or n.parent is self.cache.root_node:
                continue
            if n.backuped:
                self.assertTrue(
                    n.parent.backuped,
                    f"node {n.id} backed up but parent {n.parent.id} not",
                )

    def test_crash_shape_19_45(self):
        """backend_185833 19:45:27: the ring dropped the host copy of a
        resident parent while its child was not yet backed up; the child's
        backup was then accepted without the parent -> on-idle sanity_check
        raised 'node 22 backed up but parent 21 not backed up'.
        v1: red. v2: the child's backup re-stages the parent first."""
        toks, nodes = self.chain(1000, 2, 10)
        parent, child = nodes
        self.backup(parent)
        self.cache._drop_resident_host_backups(10, protected=set(), reason="t")
        self.assertFalse(parent.backuped)  # it was a leaf among backed-up nodes
        self.assertGreater(self.backup(child), 0)
        self.assert_parent_first()
        self.cache.sanity_check()

    def test_never_drops_a_parent_whose_child_is_backed_up(self):
        toks, nodes = self.chain(1000, 3, 10)
        for n in nodes:
            self.backup(n)
        self.cache._drop_resident_host_backups(10, protected=set(), reason="t")
        self.assertEqual([n.backuped for n in nodes], [True, True, False])
        self.assert_parent_first()
        self.cache._drop_resident_host_backups(20, protected=set(), reason="t")
        self.assertEqual([n.backuped for n in nodes], [False, False, False])
        self.assert_parent_first()

    def test_full_host_of_resident_copies_blocks_backup_without_the_ring(self):
        os.environ["SGLANG_HICACHE_HOST_STAGING_RING"] = "0"
        _, nodes = self.chain(1000, 4, 10)  # 40 tokens = host capacity
        for n in nodes:
            self.assertGreater(self.backup(n), 0)
        self.assertEqual(self.cc.mem_pool_host.available_size(), 0)
        other = self.insert(list(range(5000, 5010)))
        self.assertEqual(self.cache.write_backup(other), 0)  # upstream: L3 starves

    def test_ring_frees_another_branch_leaf_first(self):
        _, a = self.chain(1000, 2, 10)  # old branch, 20 tokens
        _, b = self.chain(3000, 2, 10)  # newer branch, 20 tokens -> host full
        for n in a + b:
            self.backup(n)
        new = self.insert(list(range(5000, 5010)))
        self.assertEqual(self.cache.write_backup(new), 10)
        self.assertEqual([n.backuped for n in a], [True, False])  # LRU leaf
        self.assertTrue(all(n.backuped for n in b))
        self.assert_parent_first()

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
        self.assert_parent_first()

    def test_active_path_is_protected_documented_limit(self):
        """One path longer than the host pool: its ancestors are protected,
        so the tail cannot be staged past the host size (backup returns 0,
        nothing is broken)."""
        toks, nodes = self.chain(1000, 4, 10)
        for n in nodes:
            self.backup(n)
        toks = toks + list(range(2000, 2010))
        tail = self.insert(toks)
        self.assertEqual(self.backup(tail), 0)
        self.assertTrue(all(n.backuped for n in nodes))
        self.assert_parent_first()

    def test_prefetch_drops_off_path_copies_instead_of_host_kv_full(self):
        _, nodes = self.chain(1000, 4, 10)
        for n in nodes:
            self.backup(n)
        self.cache.enable_storage = True
        new = array("q", range(7000, 7030))
        self.cache.prefetch_from_storage("r1", self.cache.root_node, new)
        self.assertEqual(self.cc.prefetched, [("r1", 30)])
        self.assert_parent_first()

    def test_prefetch_never_drops_its_own_path(self):
        toks, nodes = self.chain(1000, 4, 10)
        for n in nodes:
            self.backup(n)
        self.cache.enable_storage = True
        self.cache.prefetch_from_storage("r2", nodes[0], array("q", toks[10:] + [9, 9]))
        for n in nodes:
            self.assertTrue(n.backuped)
        self.assert_parent_first()

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


class HostStagingRingMambaTest(HostStagingRingTest):
    """Same scenarios on the served tree shape (Full KV + GDN/mamba states)."""

    cfg = urc.CacheConfig(
        page_size=1,
        kv_size=256,
        max_context_len=256,
        components=(ComponentType.FULL, ComponentType.MAMBA),
    )


if __name__ == "__main__":
    unittest.main()
