# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================
"""nf-poolleak-wurzel-1011: the two NF servers killed on 11.10. (boots ...060945 and
...090503, ``ValueError: pool memory leak detected! [full] total=262144,
available=955|1856, evictable=201088|143424``).

ROOT. A P chunked-prefill PARK (``scheduler.process_pending_pdflip_park`` ->
``release_kv_cache(req, tree, is_insert=True)`` -> ``UnifiedRadixCache.cache_finished_req``)
finishes the request at ``kv_committed_len`` == its last chunk end. Under BIGRAM keys
(``FLLIPER_HICACHE_BIGRAM_KEYS=1`` forces them on P) the chunk's insert used the EXACT
key (``bigram_anchor_key`` takes the next prompt token: N units) and moved
``cache_protected_len`` to N; the finish key is built from the committed ids only,
so it has N-1 units and page alignment drops the WHOLE last page. The unaligned-tail
free then returned that page -- owned by the tree node of the previous insert -- to the
free list: one page in the tree AND on the free list. In the idle ledger that reads as
+page_size tokens (``total=262144`` vs ``118784 + 143424 = 262208``).

SECOND DEFECT (amplifier, same specimen). ``_check_full_pool``'s census step took
``read_free_rows().count`` -- the number of free PAGE IDS -- as TOKENS. At page_size 64 a
surplus the census could explain (a page present in both free lists) therefore became a
fake deficit and killed the rank anyway; the log line then printed ``available=1856``
(pages) for 118784 free tokens and "116864 tokens" (= 262144-1856-143424) of leak that
never existed.

Hermetic: real UnifiedRadixCache / PagedTokenToKVPoolAllocator / SchedulerInvariantChecker
over CPU tensors. No CUDA.
"""

import os
import unittest
from array import array

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from flliper.srt.managers.schedule_batch import Req
from flliper.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from flliper.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from flliper.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from flliper.srt.mem_cache.cache_init_params import CacheInitParams
from flliper.srt.mem_cache.common import release_kv_cache
from flliper.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from flliper.srt.sampling.sampling_params import SamplingParams
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PS = 64
POOL = PS * 64  # 64 pages


def _tree():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=PS))
    pool = ReqToTokenPool(size=4, max_context_len=1024, device="cpu", enable_memory_saver=False)
    kv = MHATokenToKVPool(
        size=POOL, page_size=PS, dtype=torch.bfloat16, head_num=2, head_dim=64,
        layer_num=2, device="cpu", enable_memory_saver=False,
    )
    alloc = PagedTokenToKVPoolAllocator(
        size=POOL, page_size=PS, dtype=torch.bfloat16, device="cpu", kvcache=kv, need_sort=False
    )
    params = CacheInitParams(
        req_to_token_pool=pool, token_to_kv_pool_allocator=alloc, page_size=PS, disable=False,
        sliding_window_size=None, tree_components=(ComponentType.FULL,),
        enable_mamba_extra_buffer=False, enable_kv_cache_events=False,
        eviction_policy="lru",
        is_eagle=True,  # P runs FLLIPER_HICACHE_BIGRAM_KEYS=1 -> bigram keys without a drafter
    )
    cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    # NF: FLLIPER_PDFLIP_BIGRAM_ANCHOR_EXACT on AND a MAMBA component; this FULL-only tree
    # pins the same keying (read once per tree, see ``bigram_anchor_exact``).
    cache.__dict__["_pdflip_bigram_anchor_exact"] = True
    return cache, alloc, pool


def _req(pool, cache, prompt):
    req = Req(
        rid="park-1", origin_input_text="", origin_input_ids=array("q", prompt),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", prompt)
    req.swa_uuid_for_lock = None
    req.extra_key = None
    req.prefix_indices = torch.empty(0, dtype=torch.int64)
    req.cache_protected_len = 0
    req.last_node = cache.root_node
    cache.inc_lock_ref(req.last_node)
    return req


def _run_chunks(cache, alloc, pool, req, chunks, chunk_len):
    done = 0
    for _ in range(chunks):
        rows = alloc.alloc(chunk_len)
        pool.write((req.req_pool_idx, slice(done, done + chunk_len)), rows)
        req.set_extend_range(done, done + chunk_len)
        done += chunk_len
        req.kv_committed_len = done
        req.kv_allocated_len = done
        cache.cache_unfinished_req(req, chunked=True)  # stash_chunked_request
    return done


class TestParkDoesNotFreeTheTreesLastPage1011(CustomTestCase):
    def _ledger(self, cache, alloc):
        return alloc.available_size() + cache.full_evictable_size() + cache.full_protected_size() - POOL

    def test_park_after_chunks_keeps_the_ledger_balanced(self):
        """RED BEFORE THE FIX: surplus == page_size and one page id in tree AND free list."""
        cache, alloc, pool = _tree()
        prompt = list(range(1000, 1000 + 6 * PS + 20))
        req = _req(pool, cache, prompt)
        done = _run_chunks(cache, alloc, pool, req, chunks=2, chunk_len=2 * PS)
        self.assertEqual(self._ledger(cache, alloc), 0, "before the park the pool is whole")

        release_kv_cache(req, cache, is_insert=True)  # process_pending_pdflip_park

        free = set(alloc.free_pages.tolist()) | set(alloc.release_pages.tolist())
        mr = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", prompt[: done + 1]), is_bigram=True))
        )
        tree_pages = set((mr.device_indices // PS).tolist())
        self.assertEqual(sorted(free & tree_pages), [], "a page the tree owns is on the free list")
        self.assertEqual(self._ledger(cache, alloc), 0, "the idle ledger reads a surplus")
        self.assertEqual(len(mr.device_indices), done, "the park still hands the whole span to the tree")

    def test_a_normal_finish_is_unchanged(self):
        """The tail page of a finished prompt (N % page != 0) is still freed."""
        cache, alloc, pool = _tree()
        prompt = list(range(1000, 1000 + 3 * PS + 20))
        req = _req(pool, cache, prompt)
        n = len(prompt)
        rows = alloc.alloc(4 * PS)
        pool.write((req.req_pool_idx, slice(0, 4 * PS)), rows)
        req.set_extend_range(0, n)
        req.kv_committed_len = n
        req.kv_allocated_len = n
        free_before = alloc.available_size()
        release_kv_cache(req, cache, is_insert=True)
        # 3 pages kept by the tree (bigram: N-1 units -> 3 pages), the 4th page freed
        self.assertEqual(cache.full_evictable_size(), 3 * PS)
        self.assertEqual(alloc.available_size() - free_before, PS)


    def test_a_retracted_running_request_is_unchanged(self):
        """D shape (park_running / cutover retract, ``retain=True`` -> ``is_insert=True``):
        the request is past its prefill, committed = prompt + outputs > the last chunk
        end, so the finish key reaches beyond ``cache_protected_len`` and the tail free
        is exactly what it was. Control for the guard: no page moves, the ledger holds."""
        cache, alloc, pool = _tree()
        prompt = list(range(1000, 1000 + 6 * PS + 20))
        req = _req(pool, cache, prompt)
        done = _run_chunks(cache, alloc, pool, req, chunks=2, chunk_len=2 * PS)
        n = len(prompt)
        total = n + 3  # prompt + 3 outputs
        rest = alloc.alloc(-(-(total - done) // PS) * PS)  # whole pages for rows [done, total)
        pool.write((req.req_pool_idx, slice(done, total)), rest[: total - done])
        req.output_ids = array("q", [7, 8, 9])
        req.kv_committed_len = total
        req.kv_allocated_len = total
        req.set_extend_range(done, n)
        self.assertGreaterEqual(req.kv_committed_len, req.cache_protected_len + PS)
        release_kv_cache(req, cache, is_insert=True)
        free = set(alloc.free_pages.tolist()) | set(alloc.release_pages.tolist())
        mr = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", prompt + [7, 8, 9]), is_bigram=True))
        )
        self.assertEqual(sorted(free & set((mr.device_indices // PS).tolist())), [])
        self.assertEqual(self._ledger(cache, alloc), 0)

    def test_an_unaligned_protected_len_never_hides_a_page_the_tree_does_not_own(self):
        """FLOOR, not ceil. ``cache_unfinished_req`` asserts ``cpl <= tree-owned + page_size - 1``:
        a NON-aligned protected length (``truncate_prefix_to``, the PP-admission stamps) sits
        inside a page the request still owns. The tail free must hand that page back; rounding
        the protected length UP skipped it (the ledger then reads a deficit of one page)."""
        cache, alloc, pool = _tree()
        prompt = list(range(1000, 1000 + 2 * PS + 42))  # bigram key: 169 units -> 2 pages
        req = _req(pool, cache, prompt)
        done = _run_chunks(cache, alloc, pool, req, chunks=1, chunk_len=2 * PS)
        self.assertEqual(req.cache_protected_len, done)
        rows = alloc.alloc(PS)  # the request's own third page, 42 rows used
        n = len(prompt)
        pool.write((req.req_pool_idx, slice(done, n)), rows[: n - done])
        req.kv_committed_len = n
        req.kv_allocated_len = n
        req.cache_protected_len = done + 10  # stale / unaligned: inside the request's own page
        release_kv_cache(req, cache, is_insert=True)
        self.assertEqual(self._ledger(cache, alloc), 0, "the request's own page was not freed")


class _Tree:
    def __init__(self, evictable):
        self._e = evictable

    def protected_size(self):
        return 0

    def full_protected_size(self):
        return 0

    def supports_mamba(self):
        return False

    def is_tree_cache(self):
        return True

    def evictable_size(self):
        return self._e


class _Observer:
    def session_held_tokens(self):
        return 0

    def session_held_full_tokens(self):
        return 0


class _Args:
    dcp_size = 1


class TestCensusCountsPagesAsTokens1011(CustomTestCase):
    def test_a_page_in_both_free_lists_is_explained_not_fatal_at_page_size_64(self):
        """The #912 surplus (a page id in free_pages AND release_pages) must be closed by
        the census at page_size > 1 as it is at page_size 1. RED BEFORE THE FIX: the census
        compared 24 page ids against a token ledger and raised."""
        _tree()  # global server args (page_size 64)
        pool = ReqToTokenPool(size=2, max_context_len=64, device="cpu", enable_memory_saver=False)
        kv = MHATokenToKVPool(
            size=POOL, page_size=PS, dtype=torch.bfloat16, head_num=2, head_dim=64,
            layer_num=2, device="cpu", enable_memory_saver=False,
        )
        alloc = PagedTokenToKVPoolAllocator(
            size=POOL, page_size=PS, dtype=torch.bfloat16, device="cpu", kvcache=kv, need_sort=False
        )
        held = alloc.alloc(40 * PS)  # "tree": 40 pages
        self.assertEqual(len(held), 40 * PS)
        dup_page = alloc.free_pages[:1].clone()
        alloc.release_pages = dup_page  # same id as free_pages[0]: len-sum counts it twice
        evictable = 40 * PS
        checker = SchedulerInvariantChecker(
            is_hybrid_swa=False, is_hybrid_ssm=False, disaggregation_mode=None, page_size=PS,
            full_tokens_per_layer=None, swa_tokens_per_layer=None, max_total_num_tokens=POOL,
            server_args=_Args(), tree_cache=_Tree(evictable), token_to_kv_pool_allocator=alloc,
            req_to_token_pool=pool, pool_stats_observer=_Observer(),
            get_last_batch=lambda: None, get_running_batch=lambda: None,
        )
        ps = PoolStats(
            full_num_used=0, full_token_usage=0.0,
            full_available_size=alloc.available_size(), full_evictable_size=evictable,
        )
        self.assertEqual(alloc.available_size() + evictable - POOL, PS, "cheap ledger reads +1 page")
        leak, msg = checker._check_full_pool(ps)
        self.assertFalse(leak, msg)


if __name__ == "__main__":
    unittest.main()
