"""Hermetic (CPU-only) regression tests for mamba slot starvation.

Production incident: all three schedulers died twice on

    assert slot is not None, "Can not alloc mamba cache"

in `MambaRadixCache._alloc_mamba_slot`. The allocation retry after
`evict(EvictParams(num_tokens=0, mamba_num=1))` cannot succeed when every
cached mamba state belongs to a running request: `evict_mamba` walks the LRU
via `LRUList.get_lru_no_lock`, which skips every node with `mamba_lock_ref > 0`
and therefore returns `None` immediately, evicting nothing.

The fix makes slot exhaustion non-fatal:

* CACHE-INSERT paths (checkpoint donation in `cache_unfinished_req`, the int8
  checkpoint commit in `cache_finished_req`) skip caching this state -- a later
  cache miss, never a crash.
* The REQUIRED path (`_match_post_processor`'s prefix-resume CoW, which would
  hold the request's OWN state) degrades to a full cache miss, so the request
  re-prefills and takes its slot through normal admission.

These tests never touch a GPU: every pool is built on `torch.device("cpu")`.
"""

import unittest
from array import array

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20)

NUM_LAYERS = 8
GLOBAL_INTERVAL = 4
KV_POOL_SIZE = 512
MAX_CONTEXT_LEN = 128


def _build_tree(mamba_size=6, enable_mamba_extra_buffer=False, max_num_reqs=10):
    """MambaRadixCache + pools pinned to CPU (no accelerator required)."""
    server_args = ServerArgs(model_path="dummy", page_size=1)
    # The property would otherwise load the HF config of the dummy model.
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    set_global_server_args_for_scheduler(server_args)

    device = "cpu"
    full_attention_layer_ids = [
        i for i in range(GLOBAL_INTERVAL - 1, NUM_LAYERS, GLOBAL_INTERVAL)
    ]
    mamba_layers = [i for i in range(NUM_LAYERS) if i not in full_attention_layer_ids]
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=512,
            n_groups=4,
            num_heads=8,
            head_dim=64,
            state_size=32,
            conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

    req_to_token_pool = HybridReqToTokenPool(
        size=max_num_reqs,
        mamba_size=mamba_size,
        mamba_spec_state_size=max_num_reqs,
        max_context_len=MAX_CONTEXT_LEN,
        device=device,
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=mamba_layers,
        enable_mamba_extra_buffer=enable_mamba_extra_buffer,
        enable_linear_replayssm=False,
    )
    kv_pool = HybridLinearKVPool(
        size=KV_POOL_SIZE,
        dtype=torch.bfloat16,
        page_size=1,
        head_num=2,
        head_dim=64,
        full_attention_layer_ids=full_attention_layer_ids,
        device=device,
        enable_memory_saver=False,
        mamba_pool=req_to_token_pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=KV_POOL_SIZE,
        dtype=torch.bfloat16,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
    )
    tree = MambaRadixCache(
        params=CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
            enable_kv_cache_events=False,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
        )
    )
    return tree, allocator, req_to_token_pool


def _make_req(req_to_token_pool, allocator, token_ids, rid="r0"):
    """A running request holding its own active mamba slot and KV row."""
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", token_ids),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    req_to_token_pool.alloc([req])
    kv_indices = allocator.alloc(len(token_ids))
    req_to_token_pool.req_to_token[req.req_pool_idx, : len(token_ids)] = kv_indices
    req._refresh_fill_ids()
    req.set_extend_range(0, len(token_ids))
    req.cache_protected_len = 0
    return req


def _fill_pool_with_locked_nodes(tree, allocator, req_to_token_pool, base=1000):
    """Consume every remaining mamba slot with cached nodes that are locked by a
    (simulated) running request, i.e. exactly the production state: pool empty,
    mamba_evictable_size() == 0."""
    locked = []
    while True:
        slot = req_to_token_pool.mamba_allocator.alloc(1)
        if slot is None:
            break
        token_ids = list(range(base + 50 * len(locked), base + 50 * len(locked) + 5))
        kv_indices = allocator.alloc(len(token_ids))
        tree.insert(
            InsertParams(
                key=RadixKey(array("q", token_ids)),
                value=kv_indices,
                mamba_value=slot,
            )
        )
        node = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", token_ids)))
        ).last_device_node
        tree.inc_lock_ref(node)
        locked.append(node)
    return locked


def _match_len(tree, token_ids):
    result = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", token_ids))))
    return len(result.device_indices)


class TestStarvedPoolIsNotFatal(unittest.TestCase):
    """Falsifiers: each of these raised AssertionError before the fix."""

    def test_lru_walk_skips_locked_nodes(self):
        """Root cause pin: with every cached state locked, the LRU walk that
        eviction relies on yields nothing, so eviction can free 0 slots."""
        tree, allocator, pool = _build_tree(mamba_size=4)
        _fill_pool_with_locked_nodes(tree, allocator, pool)

        self.assertEqual(pool.mamba_allocator.available_size(), 0)
        self.assertEqual(tree.mamba_evictable_size(), 0)
        self.assertGreater(tree.mamba_protected_size(), 0)
        self.assertIsNone(tree.mamba_lru_list.get_lru_no_lock())
        self.assertEqual(tree.evict_mamba(1), 0)

    def test_alloc_mamba_slot_returns_none_instead_of_asserting(self):
        """FALSIFIER: the exact production call site."""
        tree, allocator, pool = _build_tree(mamba_size=4)
        _fill_pool_with_locked_nodes(tree, allocator, pool)

        # Before the fix this raised AssertionError("Can not alloc mamba cache")
        # and killed the scheduler process.
        self.assertIsNone(tree._alloc_mamba_slot())

    def test_cache_unfinished_req_skips_donation_when_starved(self):
        """FALSIFIER: chunked-prefill checkpoint donation under a starved pool.

        The request must survive, keep its own active slot, and simply not be
        cached (a later cache miss)."""
        tree, allocator, pool = _build_tree(mamba_size=6)
        token_ids = list(range(500, 508))
        req = _make_req(pool, allocator, token_ids)
        req.last_node = tree.root_node
        own_slot = int(req.mamba_pool_idx.item())
        _fill_pool_with_locked_nodes(tree, allocator, pool)

        tree.cache_unfinished_req(req, chunked=True)

        # No crash; the request keeps computing with its own slot.
        self.assertEqual(int(req.mamba_pool_idx.item()), own_slot)
        # prefix_indices is still handed back for PrefillAdder::add_chunked_req.
        self.assertEqual(len(req.prefix_indices), len(token_ids))
        # Nothing was cached -> the next identical request misses.
        self.assertEqual(_match_len(tree, token_ids), 0)

    def test_starved_donation_leaks_no_slot(self):
        """A skipped donation must return the pool to exactly its prior state."""
        tree, allocator, pool = _build_tree(mamba_size=6)
        token_ids = list(range(500, 508))
        req = _make_req(pool, allocator, token_ids)
        req.last_node = tree.root_node
        _fill_pool_with_locked_nodes(tree, allocator, pool)

        before = (
            pool.mamba_allocator.available_size(),
            tree.mamba_evictable_size(),
            tree.mamba_protected_size(),
        )
        tree.cache_unfinished_req(req, chunked=True)
        after = (
            pool.mamba_allocator.available_size(),
            tree.mamba_evictable_size(),
            tree.mamba_protected_size(),
        )
        self.assertEqual(before, after)

    def test_extra_buffer_strategy_skips_donation_when_starved(self):
        """FALSIFIER: same, for the extra_buffer (ping-pong) strategy."""
        tree, allocator, pool = _build_tree(
            mamba_size=8, enable_mamba_extra_buffer=True
        )
        token_ids = list(range(700, 708))
        req = _make_req(pool, allocator, token_ids)
        req.last_node = tree.root_node
        req.mamba_last_track_seqlen = len(token_ids)
        ping_pong_before = req.mamba_ping_pong_track_buffer.clone()
        _fill_pool_with_locked_nodes(tree, allocator, pool)

        tree.cache_unfinished_req(req, chunked=True)

        # The ping-pong buffer must not be left half-swapped: every allocation
        # that can fail happens before the donate.
        self.assertTrue(torch.equal(req.mamba_ping_pong_track_buffer, ping_pong_before))
        self.assertEqual(len(req.prefix_indices), len(token_ids))
        self.assertEqual(_match_len(tree, token_ids), 0)

    def test_burst_of_chunk_boundaries_never_crashes(self):
        """Burst path: many consecutive chunk-boundary donations while the pool
        is starved. Admission budgets mamba slots once per scheduling round, so
        the per-step donations are not individually gated -- they must degrade,
        not crash, however often they fire."""
        tree, allocator, pool = _build_tree(mamba_size=6)
        req = _make_req(pool, allocator, list(range(300, 316)))
        req.last_node = tree.root_node
        _fill_pool_with_locked_nodes(tree, allocator, pool)

        for step in range(1, 9):
            req.set_extend_range(0, 2 * step)
            req.cache_protected_len = 0
            tree.cache_unfinished_req(req, chunked=True)

        self.assertEqual(pool.mamba_allocator.available_size(), 0)
        self.assertIsNotNone(req.mamba_pool_idx)

    def test_cow_match_degrades_to_cache_miss_when_starved(self):
        """FALSIFIER (REQUIRED path): prefix-resume CoW cannot 'skip caching'
        -- it degrades to a full miss so the request re-prefills and takes its
        slot through normal admission."""
        tree, allocator, pool = _build_tree(mamba_size=5)
        cached_tokens = list(range(900, 908))

        # Cache one state and leave it UNLOCKED, so it is a real resume
        # candidate; then starve the pool with locked nodes.
        slot = pool.mamba_allocator.alloc(1)
        tree.insert(
            InsertParams(
                key=RadixKey(array("q", cached_tokens)),
                value=allocator.alloc(len(cached_tokens)),
                mamba_value=slot,
            )
        )
        _fill_pool_with_locked_nodes(tree, allocator, pool)

        # Sanity: the resume candidate is present and would normally be matched.
        self.assertGreater(tree.mamba_evictable_size(), 0)

        req = Req(
            rid="cow",
            origin_input_text="",
            origin_input_ids=array("q", cached_tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.mamba_pool_idx = None

        # The single evictable state is the resume node itself; it is locked for
        # the duration of the retry, so the retry cannot free anything.
        result = tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", cached_tokens)), req=req, cow_mamba=True
            )
        )

        if req.mamba_pool_idx is None:
            # Degraded: a full miss, no slot handed out, no crash.
            self.assertEqual(len(result.device_indices), 0)
            self.assertIs(result.last_device_node, tree.root_node)
            self.assertIsNone(result.mamba_branching_seqlen)
        else:
            # The retry could evict the resume candidate itself -> normal CoW.
            self.assertIsNotNone(result.last_device_node)


class TestDefaultPathNeutrality(unittest.TestCase):
    """With slots available nothing may change."""

    def test_donation_caches_normally_with_ample_pool(self):
        tree, allocator, pool = _build_tree(mamba_size=16)
        token_ids = list(range(500, 508))
        req = _make_req(pool, allocator, token_ids)
        req.last_node = tree.root_node
        avail_before = pool.mamba_allocator.available_size()

        tree.cache_unfinished_req(req, chunked=True)

        # The donated checkpoint is in the tree and locked as the new last_node.
        self.assertEqual(_match_len(tree, token_ids), len(token_ids))
        self.assertIsNot(req.last_node, tree.root_node)
        self.assertIsNotNone(req.last_node.mamba_value)
        self.assertGreater(req.last_node.mamba_lock_ref, 0)
        # Exactly one slot moved from free into the cache.
        self.assertEqual(pool.mamba_allocator.available_size(), avail_before - 1)
        self.assertEqual(tree.mamba_protected_size(), 1)
        # No degradation happened.
        self.assertEqual(getattr(tree, "_mamba_starvation_count", 0), 0)

    def test_extra_buffer_donation_normal_with_ample_pool(self):
        tree, allocator, pool = _build_tree(
            mamba_size=16, enable_mamba_extra_buffer=True
        )
        token_ids = list(range(700, 708))
        req = _make_req(pool, allocator, token_ids)
        req.last_node = tree.root_node
        req.mamba_last_track_seqlen = len(token_ids)
        donate_idx = pool.get_mamba_ping_pong_keep_idx(req)
        donated = int(req.mamba_ping_pong_track_buffer[donate_idx].item())

        tree.cache_unfinished_req(req, chunked=True)

        self.assertEqual(_match_len(tree, token_ids), len(token_ids))
        # The old ping-pong slot was donated to the tree, a fresh one installed.
        self.assertEqual(int(req.last_node.mamba_value[0].item()), donated)
        self.assertNotEqual(
            int(req.mamba_ping_pong_track_buffer[donate_idx].item()), donated
        )
        self.assertEqual(getattr(tree, "_mamba_starvation_count", 0), 0)

    def test_eviction_still_runs_when_states_are_unlocked(self):
        """The degradation must not short-circuit normal eviction: an unlocked
        cached state is still evicted to serve an allocation."""
        tree, allocator, pool = _build_tree(mamba_size=3)
        for i in range(3):
            token_ids = list(range(100 + 20 * i, 100 + 20 * i + 4))
            tree.insert(
                InsertParams(
                    key=RadixKey(array("q", token_ids)),
                    value=allocator.alloc(len(token_ids)),
                    mamba_value=pool.mamba_allocator.alloc(1),
                )
            )
        self.assertEqual(pool.mamba_allocator.available_size(), 0)
        self.assertEqual(tree.mamba_evictable_size(), 3)

        slot = tree._alloc_mamba_slot()

        self.assertIsNotNone(slot)
        self.assertEqual(getattr(tree, "_mamba_starvation_count", 0), 0)

    def test_cow_match_normal_with_ample_pool(self):
        tree, allocator, pool = _build_tree(mamba_size=16)
        token_ids = list(range(900, 908))
        tree.insert(
            InsertParams(
                key=RadixKey(array("q", token_ids)),
                value=allocator.alloc(len(token_ids)),
                mamba_value=pool.mamba_allocator.alloc(1),
            )
        )
        req = Req(
            rid="cow-ok",
            origin_input_text="",
            origin_input_ids=array("q", token_ids),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        result = tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", token_ids)), req=req, cow_mamba=True
            )
        )
        self.assertEqual(len(result.device_indices), len(token_ids))
        self.assertIsNotNone(req.mamba_pool_idx)
        self.assertIsNotNone(req.mamba_cow_src_index)
        self.assertEqual(getattr(tree, "_mamba_starvation_count", 0), 0)


class TestCallerSurveyCoverage(unittest.TestCase):
    """Source-level pins so a future edit cannot reintroduce a fatal path."""

    MODULES = (
        "sglang.srt.mem_cache.mamba_radix_cache",
        "sglang.srt.mem_cache.unified_cache_components.mamba_component",
    )

    def _source(self, module_name):
        import importlib
        import inspect

        return inspect.getsource(importlib.import_module(module_name))

    def test_no_fatal_assert_on_mamba_slot_allocation(self):
        for module_name in self.MODULES:
            source = self._source(module_name)
            for needle in (
                'assert slot is not None, "Can not alloc mamba cache"',
                'assert dst_index is not None, "Can not alloc mamba cache"',
                'assert slot is not None, "Can not alloc int8 mamba checkpoint slot"',
            ):
                self.assertNotIn(
                    needle,
                    source,
                    f"{module_name} reintroduced a fatal mamba slot assert: {needle}",
                )

    def test_alloc_helpers_are_declared_optional(self):
        import inspect

        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        for cls in (MambaRadixCache, MambaComponent):
            for name in (
                "_alloc_mamba_slot",
                "_alloc_int8_ckpt_slot",
                "_commit_int8_checkpoint",
            ):
                annotation = inspect.signature(getattr(cls, name)).return_annotation
                self.assertIn(
                    "Optional",
                    str(annotation),
                    f"{cls.__name__}.{name} must be able to return None",
                )

    def test_every_alloc_call_site_checks_for_none(self):
        """Every `_alloc_mamba_slot()` result must be None-checked nearby."""
        import re

        for module_name in self.MODULES:
            source = self._source(module_name).splitlines()
            for idx, line in enumerate(source):
                if "self._alloc_mamba_slot()" not in line:
                    continue
                match = re.search(r"(\w+)\s*=\s*.*_alloc_mamba_slot\(\)", line)
                if match is None:
                    continue
                name = match.group(1)
                window = "\n".join(source[idx : idx + 8])
                self.assertIn(
                    f"{name} is None",
                    window,
                    f"{module_name}:{idx + 1} does not handle a None slot",
                )


if __name__ == "__main__":
    unittest.main()
