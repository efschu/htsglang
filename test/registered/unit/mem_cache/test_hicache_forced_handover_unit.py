"""Forced host write-through for a session hand-over (#242 spill-budget demotion).

The hierarchical cache writes a node to the host tier only once it has been hit
``write_through_threshold`` times. That hit-rate heuristic is right for ordinary
caching and wrong for a HAND-OVER: when the spill budget demotes a session, the
donating insert is the session's only surviving copy, because the same finish
frees the device slots. The leaves under the threshold -- the newest tokens --
would be dropped silently.

These tests pin both halves: a marked request reaches the host tier in full
under every write policy, and an unmarked request keeps the stock heuristic.
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    requests_forced_host_write_through,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.utils import get_eviction_strategy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeCacheController:
    """Enough of HiCacheController for write_backup: hand out host indices and
    record which node ids were staged."""

    def __init__(self, write_policy: str, host_size: int = 4096):
        self.write_policy = write_policy
        self._free = list(range(host_size))
        self.written_node_ids = []
        self.ack_write_queue = []

    def write(self, device_indices=None, node_id=None, **kwargs):
        n = len(device_indices)
        if len(self._free) < n:
            return None
        out = torch.tensor(self._free[:n], dtype=torch.int64)
        self._free = self._free[n:]
        self.written_node_ids.append(node_id)
        return out

    def evict_host(self, host_value):
        return len(host_value)


def _build_cache(write_policy: str) -> HiRadixCache:
    """A HiRadixCache carrying only the state its insert path touches.

    HiRadixCache.__init__ builds host pools, a controller thread and CUDA
    streams; the write-through decision under test needs none of that, so the
    real methods run against a hand-built instance on CPU.
    """
    cache = HiRadixCache.__new__(HiRadixCache)
    cache.page_size = 1
    cache.disable = False
    cache.is_eagle = False
    cache.enable_storage = False
    cache.enable_kv_cache_events = False
    cache.enable_session_radix_cache = False
    cache.evictable_size_ = 0
    cache.protected_size_ = 0
    cache.evictable_leaves = set()
    cache.evictable_host_leaves = set()
    cache.ongoing_write_through = {}
    cache.ongoing_backup = {}
    cache.kv_cache = None
    cache.eviction_strategy = get_eviction_strategy("lru")
    cache.cache_controller = _FakeCacheController(write_policy)
    # #810: HiRadixCache.__init__ builds this unconditionally, and the whole
    # insert path guards on `is not None`. None is the shipped state under
    # `--hicache-host-role retention` -- the default, and the regime these
    # handover cases are written against -- so it is what a bare instance
    # must carry. UnifiedRadixCache spells the same default out at
    # unified_radix_cache.py:757; HiRadixCache reaches it through __init__,
    # which this helper deliberately skips.
    cache.staging_write_ring = None
    # Same derivation as HiRadixCache.__init__.
    cache.write_through_threshold = 1 if write_policy == "write_through" else 2

    TreeNode.counter = 0
    root = TreeNode()
    root.key = RadixKey(array("q"))
    root.value = None
    root.lock_ref = 1
    cache.root_node = root
    return cache


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(list(indices.tolist()))


def _attach_finish_path(cache: HiRadixCache, n_slots: int = 64) -> None:
    """The little bit of pool plumbing ``cache_finished_req`` reaches for."""
    cache.disable_finished_insert = False
    cache.token_to_kv_pool_allocator = _FakeAllocator()
    cache.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(n_slots, dtype=torch.int64).reshape(1, n_slots)
    )


def _demoted_req(tokens, *, marked: bool):
    """A finishing request as the spill-budget demotion leaves it: capped
    output, and (when ``marked``) flagged for a lossless hand-over."""
    req = SimpleNamespace(
        rid="demoted",
        req_pool_idx=0,
        origin_input_ids=list(tokens[:2]),
        output_ids=list(tokens[2:]),
        extra_key=None,
        cache_protected_len=0,
        last_node=None,
        priority=0,
        pop_committed_kv_cache=lambda: len(tokens),
    )
    if marked:
        req.force_host_write_through = True
    return req


def _insert(cache: HiRadixCache, tokens, *, force: bool = False):
    return cache.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)),
            value=torch.arange(len(tokens), dtype=torch.int64),
            force_host_write_through=force,
        )
    )


def _chain(cache: HiRadixCache):
    """All non-root nodes as (token_ids, backuped), parent-first."""
    out = []

    def walk(node):
        for child in node.children.values():
            out.append((list(child.key.token_ids), child.backuped))
            walk(child)

    walk(cache.root_node)
    return out


class TestForcedHostHandover(CustomTestCase):
    # -- the hand-over, through the real finish path ---------------------

    def test_demoted_session_hands_its_whole_prefix_to_the_host_tier(self):
        """End to end over ``cache_finished_req``: a demoted session's donation
        must leave NOTHING on device only. Red before the fix -- the leaf was
        under the write-through threshold and was dropped silently."""
        cache = _build_cache("write_through_selective")
        _attach_finish_path(cache)
        cache.cache_finished_req(_demoted_req([1, 2, 3, 4, 5, 6], marked=True))

        chain = _chain(cache)
        self.assertEqual(chain, [([1, 2, 3, 4, 5, 6], True)])
        self.assertTrue(
            all(backuped for _, backuped in chain),
            f"demoted session lost leaves below the threshold: {chain}",
        )

    def test_undemoted_session_finish_keeps_the_stock_heuristic(self):
        cache = _build_cache("write_through_selective")
        _attach_finish_path(cache)
        cache.cache_finished_req(_demoted_req([1, 2, 3, 4, 5, 6], marked=False))
        self.assertEqual(_chain(cache), [([1, 2, 3, 4, 5, 6], False)])
        self.assertEqual(cache.cache_controller.written_node_ids, [])

    # -- the loss, per write policy -------------------------------------

    def test_unforced_leaf_below_threshold_never_reaches_the_host_tier(self):
        """The failure this fix removes: a freshly donated leaf is under the
        threshold, so the host tier never sees it."""
        for policy in ("write_through_selective", "write_back"):
            with self.subTest(policy=policy):
                cache = _build_cache(policy)
                _insert(cache, [1, 2, 3, 4])
                self.assertEqual(_chain(cache), [([1, 2, 3, 4], False)])
                self.assertEqual(cache.cache_controller.written_node_ids, [])

    def test_forced_leaf_below_threshold_reaches_the_host_tier(self):
        for policy in ("write_through", "write_through_selective", "write_back"):
            with self.subTest(policy=policy):
                cache = _build_cache(policy)
                _insert(cache, [1, 2, 3, 4], force=True)
                self.assertEqual(_chain(cache), [([1, 2, 3, 4], True)])
                self.assertEqual(len(cache.cache_controller.written_node_ids), 1)

    def test_forced_handover_covers_the_whole_chain_not_just_the_leaf(self):
        """A continuation lands on a shared prefix that is itself unbacked; the
        hand-over must carry the prefix AND the new tail, parent-first, so
        write_backup's contiguity invariant holds."""
        cache = _build_cache("write_through_selective")
        _insert(cache, [1, 2, 3, 4])  # prefix parked under the threshold
        self.assertEqual(_chain(cache), [([1, 2, 3, 4], False)])

        _insert(cache, [1, 2, 3, 4, 5, 6], force=True)
        self.assertEqual(
            _chain(cache),
            [([1, 2, 3, 4], True), ([5, 6], True)],
        )

    def test_forced_handover_carries_a_split_prefix(self):
        """Diverging continuation: the split parent and both fragments below the
        hand-over must be on host."""
        cache = _build_cache("write_through_selective")
        _insert(cache, [1, 2, 3, 4])
        _insert(cache, [1, 2, 7, 8], force=True)
        self.assertEqual(
            sorted(_chain(cache)),
            sorted([([1, 2], True), ([3, 4], False), ([7, 8], True)]),
        )

    def test_forced_handover_does_not_grow_hit_counts_under_write_back(self):
        """write_back never counts hits; forcing a hand-over must not start."""
        cache = _build_cache("write_back")
        _insert(cache, [1, 2, 3, 4], force=True)
        leaf = next(iter(cache.root_node.children.values()))
        self.assertEqual(leaf.hit_count, 0)
        self.assertTrue(leaf.backuped)

    # -- the normal path stays exactly as it was ------------------------

    def test_unforced_write_through_is_unchanged(self):
        cache = _build_cache("write_through")
        _insert(cache, [1, 2, 3, 4])
        self.assertEqual(_chain(cache), [([1, 2, 3, 4], True)])
        self.assertEqual(len(cache.cache_controller.written_node_ids), 1)

    def test_unforced_selective_still_needs_a_second_hit(self):
        cache = _build_cache("write_through_selective")
        _insert(cache, [1, 2, 3, 4])
        self.assertEqual(_chain(cache), [([1, 2, 3, 4], False)])
        _insert(cache, [1, 2, 3, 4])  # second hit crosses the threshold
        self.assertEqual(_chain(cache), [([1, 2, 3, 4], True)])

    def test_unforced_chunked_insert_is_still_skipped(self):
        cache = _build_cache("write_through")
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3, 4])),
                value=torch.arange(4, dtype=torch.int64),
                chunked=True,
            )
        )
        self.assertEqual(_chain(cache), [([1, 2, 3, 4], False)])
        self.assertEqual(cache.cache_controller.written_node_ids, [])

    def test_default_insert_params_do_not_force(self):
        self.assertFalse(InsertParams().force_host_write_through)

    # -- the request marker --------------------------------------------

    def test_request_marker_defaults_to_off(self):
        self.assertFalse(requests_forced_host_write_through(SimpleNamespace()))
        self.assertTrue(
            requests_forced_host_write_through(
                SimpleNamespace(force_host_write_through=True)
            )
        )


if __name__ == "__main__":
    unittest.main()
