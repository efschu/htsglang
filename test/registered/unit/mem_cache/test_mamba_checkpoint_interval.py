"""CPU tests for --mamba-checkpoint-interval (deterministic GDN checkpoint
boundaries), the flush-time mamba pool reset, and the --disable-radix-cache
joint memory fit.

Covers:
* boundary arithmetic (mamba_ckpt_utils),
* the resume-determinism property on the radix tree: identical insert
  history => identical resume point, also after eviction of off-grid
  checkpoints; strict mode falls back to 0 instead of a shallower survivor,
* the eviction live-window for the deepest on-grid checkpoints,
* HybridReqToTokenPool.clear() returning the mamba pool (states, ReplaySSM
  rings, write cursors) to the freshly-initialized contents,
* the budget-fitted mamba pool size when the radix cache is disabled.
"""

import ast
import inspect
import textwrap
import unittest
from array import array
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.distributed.utils import set_cp_token_ratios
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_ckpt_utils import (
    floor_to_interval,
    is_on_interval,
    mamba_checkpoint_track_target,
)
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    zero_kv_data_buffers,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

GB = 1 << 30


class TestCkptBoundaryArithmetic(unittest.TestCase):
    def test_floor_and_grid(self):
        self.assertEqual(floor_to_interval(5000, 2048), 4096)
        self.assertEqual(floor_to_interval(4096, 2048), 4096)
        self.assertEqual(floor_to_interval(2047, 2048), 0)
        # interval off -> identity / always on grid
        self.assertEqual(floor_to_interval(5000, None), 5000)
        self.assertTrue(is_on_interval(5000, None))
        self.assertTrue(is_on_interval(4096, 2048))
        self.assertFalse(is_on_interval(5000, 2048))
        self.assertTrue(is_on_interval(0, 2048))

    def test_track_target_basic(self):
        # Step [0, 8192) with G=2048: deepest boundary is 8192 itself.
        self.assertEqual(mamba_checkpoint_track_target(0, 8192, 2048, 64), 8192)
        # Step [2048, 3048): no boundary crossed (next is 4096).
        self.assertIsNone(mamba_checkpoint_track_target(2048, 1000, 2048, 64))
        # Step [4096, 23000): boundary 22528 crossed mid-step.
        self.assertEqual(
            mamba_checkpoint_track_target(4096, 23000 - 4096, 2048, 64), 22528
        )

    def test_track_target_needs_chunk_aligned_offset(self):
        # Unaligned prefix (5000): boundary 6144 has offset 1144 % 64 != 0
        # -> not snapshotable, deterministic skip.
        self.assertIsNone(mamba_checkpoint_track_target(5000, 4000, 2048, 64))
        # Aligned prefix (4096): offset of 6144 is 2048 % 64 == 0 -> ok.
        self.assertEqual(mamba_checkpoint_track_target(4096, 4000, 2048, 64), 6144)

    def test_track_target_boundary_at_prefix_not_counted(self):
        # Boundary equal to the resume point carries no NEW state.
        self.assertIsNone(mamba_checkpoint_track_target(2048, 512, 2048, 64))

    def test_track_target_write_pos_style_positions(self):
        # The finished-request donate position is len - write_pos; only exact
        # grid positions are cacheable (the state cannot be rewound).
        for write_pos in range(0, 16):
            pos = 4500 - write_pos
            self.assertEqual(is_on_interval(pos, 256), pos % 256 == 0)

    def test_track_target_interval_variants(self):
        for interval in (256, 512, 2048, 8192):
            for prefix in (0, interval, 3 * interval):
                target = mamba_checkpoint_track_target(
                    prefix, 2 * interval + 17, interval, 64
                )
                self.assertEqual(target, prefix + 2 * interval)
                self.assertTrue(is_on_interval(target, interval))


def _build_tree(
    interval,
    mamba_cache_size=20,
    enable_linear_replayssm=False,
    max_context_len=128,
    size=256,
    enable_mamba_extra_buffer=False,
):
    """MambaRadixCache + pools on CPU, mirroring test_mamba_unittest's setup."""
    server_args = ServerArgs(model_path="dummy", page_size=1)
    # The property would otherwise load the HF config of the dummy model.
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    if interval is not None:
        object.__setattr__(server_args, "mamba_checkpoint_interval", interval)
    set_global_server_args_for_scheduler(server_args)
    dtype = torch.bfloat16
    num_layers = 8
    global_interval = 4
    max_num_reqs = 10
    device = get_device()
    full_attention_layer_ids = [
        i for i in range(global_interval - 1, num_layers, global_interval)
    ]
    mamba_layers = [i for i in range(num_layers) if i not in full_attention_layer_ids]
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
        mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

    req_to_token_pool = HybridReqToTokenPool(
        size=max_num_reqs,
        mamba_size=mamba_cache_size,
        mamba_spec_state_size=max_num_reqs,
        max_context_len=max_context_len,
        device=device,
        enable_memory_saver=False,
        cache_params=mamba2_cache_params,
        mamba_layer_ids=mamba_layers,
        enable_mamba_extra_buffer=enable_mamba_extra_buffer,
        enable_linear_replayssm=enable_linear_replayssm,
    )
    pool = HybridLinearKVPool(
        size=size,
        dtype=dtype,
        page_size=1,
        head_num=2,
        head_dim=64,
        full_attention_layer_ids=full_attention_layer_ids,
        device=device,
        enable_memory_saver=False,
        mamba_pool=req_to_token_pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=size,
        dtype=dtype,
        device=device,
        kvcache=pool,
        need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        disable=False,
        enable_kv_cache_events=False,
    )
    tree = MambaRadixCache(params=params)

    def make_dummy_req():
        req = Req(
            rid=0,
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req_to_token_pool.alloc([req])
        return req

    return tree, allocator, req_to_token_pool, make_dummy_req


def _insert_seq(tree, allocator, req_to_token_pool, token_ids):
    """Insert token_ids with a fresh mamba checkpoint at its full depth."""
    slot = req_to_token_pool.mamba_allocator.alloc(1)
    assert slot is not None
    kv = allocator.alloc(len(token_ids))
    result = tree.insert(
        InsertParams(
            key=RadixKey(array("q", token_ids)),
            value=kv,
            mamba_value=slot,
        )
    )
    if result.mamba_exist:
        req_to_token_pool.mamba_allocator.free(slot)
    return result


def _match_len(tree, token_ids):
    result = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", token_ids))))
    return len(result.device_indices), result


SEQ = list(range(100, 100 + 12))  # 12 tokens: grid-4 boundaries at 4, 8


class TestCheckpointIntervalResume(unittest.TestCase):
    """Interval mode: resume points live only on the absolute grid."""

    INTERVAL = 4

    def test_default_mode_unchanged(self):
        tree, allocator, pool, _ = _build_tree(None)
        _insert_seq(tree, allocator, pool, SEQ[:6])  # off-grid depth 6
        matched, _ = _match_len(tree, SEQ)
        self.assertEqual(matched, 6)  # upstream: deepest checkpoint wins

    def test_resume_only_on_grid(self):
        tree, allocator, pool, _ = _build_tree(self.INTERVAL)
        _insert_seq(tree, allocator, pool, SEQ[:8])  # on-grid depth 8
        _insert_seq(tree, allocator, pool, SEQ[:10])  # off-grid depth 10
        matched, result = _match_len(tree, SEQ)
        self.assertEqual(matched, 8)
        self.assertIsNotNone(result.last_device_node.mamba_value)

    def test_identical_history_identical_resume(self):
        """Same insert history => same resume point, even when one tree
        additionally churned an off-grid checkpoint through insert+evict."""

        def build():
            tree, allocator, pool, _ = _build_tree(self.INTERVAL)
            _insert_seq(tree, allocator, pool, SEQ[:4])
            _insert_seq(tree, allocator, pool, SEQ[:8])
            return tree, allocator, pool

        tree_a, alloc_a, pool_a = build()
        tree_b, alloc_b, pool_b = build()
        # Extra churn on B only: off-grid checkpoint appears and is evicted.
        _insert_seq(tree_b, alloc_b, pool_b, SEQ[:11])
        tree_b.evict(EvictParams(num_tokens=0, mamba_num=1))

        len_a, _ = _match_len(tree_a, SEQ)
        len_b, _ = _match_len(tree_b, SEQ)
        self.assertEqual(len_a, 8)
        self.assertEqual(len_b, 8)

    def test_offgrid_eviction_does_not_move_resume(self):
        tree, allocator, pool, _ = _build_tree(self.INTERVAL)
        _insert_seq(tree, allocator, pool, SEQ[:8])
        _insert_seq(tree, allocator, pool, SEQ[:10])
        before, _ = _match_len(tree, SEQ)
        # Evict one mamba state; the off-grid node (depth 10) is a leaf and
        # LRU-eligible. Whatever the LRU picks, the resume stays on-grid.
        tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        after, _ = _match_len(tree, SEQ)
        self.assertEqual(before, 8)
        self.assertIn(after, (8, 4, 0))
        self.assertTrue(is_on_interval(after, self.INTERVAL))

    def test_branching_seqlen_on_grid(self):
        tree, allocator, pool, _ = _build_tree(self.INTERVAL)
        _insert_seq(tree, allocator, pool, SEQ[:10])  # only an off-grid ckpt
        matched, result = _match_len(tree, SEQ[:10])
        # Full-KV match is 10 deep, but the only checkpoint (10) is off-grid:
        # resume 0, and the branch point re-establishes floor(10/4)*4 = 8.
        self.assertEqual(matched, 0)
        self.assertEqual(result.mamba_branching_seqlen, 8)

    def test_strict_resume_falls_back_to_zero(self):
        with envs.SGLANG_MAMBA_CKPT_STRICT_RESUME.override(True):
            tree, allocator, pool, _ = _build_tree(self.INTERVAL)
            _insert_seq(tree, allocator, pool, SEQ[:4])
            _insert_seq(tree, allocator, pool, SEQ[:8])
            # Both grid checkpoints present: deepest boundary of the match
            # (floor(8/4)*4 = 8) has its checkpoint -> resume 8.
            matched, _ = _match_len(tree, SEQ[:8])
            self.assertEqual(matched, 8)
            # Kill the deepest checkpoint: strict mode must NOT resume at 4.
            node8 = _match_len(tree, SEQ[:8])[1].last_device_node
            self.assertIsNotNone(node8.mamba_value)
            tree._free_mamba_value(node8.mamba_value)
            if len(node8.children) > 0:
                tree.mamba_lru_list.remove_node(node8)
                tree._tombstone_internal_node(node8)
            else:
                # leaf: emulate tombstoning by hand for the test
                tree.mamba_evictable_size_ -= len(node8.mamba_value)
                tree.mamba_lru_list.remove_node(node8)
                node8.mamba_value = None
            matched, result = _match_len(tree, SEQ[:8])
            self.assertEqual(matched, 0)
            self.assertIs(result.last_device_node, tree.root_node)
            # ... and the branching point re-establishes the deep checkpoint.
            self.assertEqual(result.mamba_branching_seqlen, 8)

    def test_non_strict_resumes_next_shallower(self):
        tree, allocator, pool, _ = _build_tree(self.INTERVAL)
        _insert_seq(tree, allocator, pool, SEQ[:4])
        _insert_seq(tree, allocator, pool, SEQ[:8])
        node8 = _match_len(tree, SEQ[:8])[1].last_device_node
        tree._free_mamba_value(node8.mamba_value)
        tree.mamba_evictable_size_ -= len(node8.mamba_value)
        tree.mamba_lru_list.remove_node(node8)
        node8.mamba_value = None
        matched, _ = _match_len(tree, SEQ[:8])
        self.assertEqual(matched, 4)

    def test_prompt_shorter_than_interval_always_resumes_zero(self):
        """Identity-prompt scenario (GPU round 2, hint 1): the prompt is
        shorter than the interval, so NO grid boundary exists inside it.
        Surviving off-grid checkpoints on the shared template path (e.g.
        decode-track positions other traffic would leave behind if any
        insert guard were bypassed) must never be accepted — the resume
        falls back to 0 deterministically, before and after churn."""
        interval = 16  # prompt (12 tokens) < interval
        tree, allocator, pool, _ = _build_tree(interval)
        _insert_seq(tree, allocator, pool, SEQ[:6])  # off-grid survivors
        _insert_seq(tree, allocator, pool, SEQ[:12])
        matched, result = _match_len(tree, SEQ[:12])
        self.assertEqual(matched, 0)
        self.assertIs(result.last_device_node, tree.root_node)
        # No grid boundary inside the match -> no branching target either.
        self.assertIsNone(result.mamba_branching_seqlen)
        # Churn one of the off-grid checkpoints: resume stays 0.
        tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        matched, result = _match_len(tree, SEQ[:12])
        self.assertEqual(matched, 0)
        self.assertIs(result.last_device_node, tree.root_node)


class TestCheckpointEvictionWindow(unittest.TestCase):
    INTERVAL = 4

    def _chain(self, tree, allocator, pool):
        # Grid checkpoints at depths 4, 8, 12 along one path.
        _insert_seq(tree, allocator, pool, SEQ[:4])
        _insert_seq(tree, allocator, pool, SEQ[:8])
        _insert_seq(tree, allocator, pool, SEQ[:12])

    def _depths_with_ckpt(self, tree):
        depths = []

        def dfs(node, depth):
            if node.mamba_value is not None:
                depths.append(depth)
            for child in node.children.values():
                dfs(child, depth + len(child.key))

        dfs(tree.root_node, 0)
        return sorted(depths)

    def test_window_protects_deepest(self):
        with envs.SGLANG_MAMBA_CKPT_WINDOW.override(2):
            tree, allocator, pool, _ = _build_tree(self.INTERVAL)
            self._chain(tree, allocator, pool)
            self.assertEqual(self._depths_with_ckpt(tree), [4, 8, 12])
            evicted = tree.evict_mamba(1)
            self.assertEqual(evicted, 1)
            # The two deepest (8, 12) are inside the live window.
            self.assertEqual(self._depths_with_ckpt(tree), [8, 12])

    def test_window_yields_under_pressure(self):
        with envs.SGLANG_MAMBA_CKPT_WINDOW.override(2):
            tree, allocator, pool, _ = _build_tree(self.INTERVAL)
            self._chain(tree, allocator, pool)
            # Demand all three: the second pass ignores the window.
            evicted = tree.evict_mamba(3)
            self.assertEqual(evicted, 3)
            self.assertEqual(self._depths_with_ckpt(tree), [])

    def test_window_disabled_without_interval(self):
        with envs.SGLANG_MAMBA_CKPT_WINDOW.override(2):
            tree, allocator, pool, _ = _build_tree(None)
            self._chain(tree, allocator, pool)
            # Upstream behavior: plain LRU, the shallowest (least recently
            # bumped) internal node goes first — one pass suffices, no
            # protection is consulted.
            evicted = tree.evict_mamba(3)
            self.assertEqual(evicted, 3)


class TestFlushResetsMambaPool(unittest.TestCase):
    """F: HybridReqToTokenPool.clear() == freshly initialized mamba pool."""

    def test_clear_resets_states_rings_and_cursors(self):
        tree, allocator, pool, make_req = _build_tree(
            None, enable_linear_replayssm=True
        )
        mamba_pool = pool.mamba_pool
        # Dirty everything a running server would touch.
        req = make_req()
        for conv in mamba_pool.mamba_cache.conv:
            conv.fill_(1.0)
        mamba_pool.mamba_cache.temporal.fill_(2.0)
        for name in ("replayssm_d", "replayssm_k", "replayssm_g"):
            getattr(mamba_pool.mamba_cache, name).fill_(3.0)
        mamba_pool.replayssm_write_pos.fill_(7)
        pool.req_index_to_mamba_index_mapping.fill_(5)

        pool.clear()

        for conv in mamba_pool.mamba_cache.conv:
            self.assertTrue(torch.all(conv == 0))
        self.assertTrue(torch.all(mamba_pool.mamba_cache.temporal == 0))
        for name in ("replayssm_d", "replayssm_k", "replayssm_g"):
            self.assertTrue(torch.all(getattr(mamba_pool.mamba_cache, name) == 0))
        self.assertTrue(torch.all(mamba_pool.replayssm_write_pos == 0))
        self.assertTrue(torch.all(pool.req_index_to_mamba_index_mapping == 0))
        # Allocator back to full capacity.
        self.assertEqual(pool.mamba_allocator.available_size(), mamba_pool.size)

    def test_flush_equals_fresh_boot_bitwise(self):
        """Full flush invariance: after the scheduler-side flush sequence
        (req_to_token_pool.clear + kv allocator clear + KV data zeroing),
        every stateful tensor of the pool stack must be bit-identical to a
        freshly built twin, and the allocators must hand out identical
        slot/token orders."""
        dirty = _build_tree(None, enable_linear_replayssm=True)
        fresh = _build_tree(None, enable_linear_replayssm=True)
        d_tree, d_alloc, d_pool, d_make = dirty
        f_tree, f_alloc, f_pool, _ = fresh

        # Simulate traffic on the dirty stack.
        req = d_make()
        _insert_seq(d_tree, d_alloc, d_pool, SEQ[:8])
        d_pool.req_to_token[req.req_pool_idx, :12] = 7
        d_pool.req_to_token[0, :] = 9  # padding row
        for conv in d_pool.mamba_pool.mamba_cache.conv:
            conv.fill_(1.0)
        d_pool.mamba_pool.mamba_cache.temporal.fill_(2.0)
        d_pool.mamba_pool.replayssm_write_pos.fill_(7)
        d_kvcache = d_alloc.get_kvcache()
        for buf in d_kvcache.full_kv_pool.k_buffer:
            buf.fill_(3.0)
        for buf in d_kvcache.full_kv_pool.v_buffer:
            buf.fill_(4.0)

        # Scheduler flush sequence (scheduler.flush_cache order).
        d_tree.reset()
        d_pool.clear()
        d_alloc.clear()
        zeroed = zero_kv_data_buffers(d_kvcache)
        self.assertGreater(zeroed, 0)

        # Bitwise comparison against the fresh twin.
        f_kvcache = f_alloc.get_kvcache()
        pairs = [
            (d_pool.req_to_token, f_pool.req_to_token),
            (d_pool.req_generation, f_pool.req_generation),
            (
                d_pool.req_index_to_mamba_index_mapping,
                f_pool.req_index_to_mamba_index_mapping,
            ),
            (
                d_pool.mamba_pool.replayssm_write_pos,
                f_pool.mamba_pool.replayssm_write_pos,
            ),
            (
                d_pool.mamba_pool.mamba_cache.temporal,
                f_pool.mamba_pool.mamba_cache.temporal,
            ),
        ]
        pairs += list(
            zip(d_pool.mamba_pool.mamba_cache.conv, f_pool.mamba_pool.mamba_cache.conv)
        )
        for name in ("replayssm_d", "replayssm_k", "replayssm_g"):
            pairs.append(
                (
                    getattr(d_pool.mamba_pool.mamba_cache, name),
                    getattr(f_pool.mamba_pool.mamba_cache, name),
                )
            )
        pairs += list(
            zip(d_kvcache.full_kv_pool.k_buffer, f_kvcache.full_kv_pool.k_buffer)
        )
        pairs += list(
            zip(d_kvcache.full_kv_pool.v_buffer, f_kvcache.full_kv_pool.v_buffer)
        )
        for i, (a, b) in enumerate(pairs):
            self.assertTrue(torch.equal(a, b), f"tensor pair {i} differs after flush")

        # Allocation order identical to a fresh boot.
        self.assertEqual(d_pool.free_slots, f_pool.free_slots)
        self.assertEqual(
            d_pool.mamba_allocator.available_size(),
            f_pool.mamba_allocator.available_size(),
        )
        d_first = d_alloc.alloc(4)
        f_first = f_alloc.alloc(4)
        self.assertTrue(torch.equal(d_first, f_first))


class TestPoolClaimPoison(unittest.TestCase):
    """Round 5: read-before-write hygiene at the mamba slot claim.

    Poison the pool, run the CPU-reachable claim paths, and assert that every
    claimed slot is queued for the deferred forward-stream clear and that the
    clear actually removes the poison — i.e. no kernel can ever observe a
    previous occupant's state in a freshly claimed slot. (RED on the pre-fix
    code: ping-pong claims were never queued for clearing.)
    """

    def _poison(self, mamba_pool):
        for conv in mamba_pool.mamba_cache.conv:
            conv.fill_(float("nan"))
        mamba_pool.mamba_cache.temporal.fill_(float("nan"))

    def test_pingpong_claim_is_queued_and_cleared(self):
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        tree, allocator, pool, make_req = _build_tree(
            None, enable_mamba_extra_buffer=True
        )
        self._poison(pool.mamba_pool)

        req = make_req()  # HybridReqToTokenPool.alloc -> _alloc_ping_pong_buffer
        self.assertIsNotNone(req.mamba_ping_pong_track_buffer)
        pp_slots = req.mamba_ping_pong_track_buffer[
            req.mamba_ping_pong_track_buffer >= 0
        ]
        self.assertGreater(pp_slots.numel(), 0)
        # Claim contract: the fresh ping-pong slots are queued for clearing.
        self.assertIsNotNone(req.mamba_pingpong_clear_indices)
        queued = set(req.mamba_pingpong_clear_indices.tolist())
        self.assertEqual(queued, set(pp_slots.tolist()))
        # ... and the active slot uses the existing mamba_needs_clear contract.
        self.assertTrue(req.mamba_needs_clear)

        # Collector picks both up (as the extend prepare would).
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch._collect_deferred_mamba_cow_and_clear([req])
        self.assertIsNone(req.mamba_pingpong_clear_indices)
        clear_indices = batch.mamba_clear_indices
        expected = {int(req.mamba_pool_idx.item())} | queued
        self.assertEqual(set(clear_indices.tolist()), expected)

        # The deferred clear (forward-stream in production) removes the poison.
        pool.mamba_pool.clear_slots(pool.translate_mamba_indices(clear_indices))
        for idx in expected:
            for conv in pool.mamba_pool.mamba_cache.conv:
                self.assertFalse(torch.isnan(conv[:, idx]).any())
            self.assertFalse(
                torch.isnan(pool.mamba_pool.mamba_cache.temporal[:, idx]).any()
            )

    def test_donate_replacement_slot_is_queued(self):
        tree, allocator, pool, make_req = _build_tree(
            None, enable_mamba_extra_buffer=True
        )
        req = make_req()
        req.mamba_pingpong_clear_indices = None  # consumed by a prior forward
        new_slot = pool.mamba_allocator.alloc(1)
        pool.donate_mamba_ping_pong_slot(req, new_slot)
        self.assertIsNotNone(req.mamba_pingpong_clear_indices)
        self.assertEqual(req.mamba_pingpong_clear_indices.tolist(), new_slot.tolist())

    def test_poison_helper_touches_only_float_buffers(self):
        from sglang.srt.mem_cache.memory_pool import maybe_poison_pool_data

        f = torch.zeros(4, dtype=torch.float32)
        i = torch.zeros(4, dtype=torch.int32)
        with envs.SGLANG_POISON_POOL_DATA.override(True):
            maybe_poison_pool_data([f, i, None], "test")
        self.assertTrue(torch.isnan(f).all())
        self.assertTrue(torch.all(i == 0))
        f2 = torch.zeros(4, dtype=torch.float32)
        maybe_poison_pool_data([f2], "test")  # env off -> no-op
        self.assertTrue(torch.all(f2 == 0))

    def test_poison_helper_covers_fp8_as_uint8_storage(self):
        # fp8 KV pools store their data as torch.uint8 (index_put lacks fp8
        # support); the poison must cover them with 0xFF, which is NaN in
        # both float8_e5m2 and float8_e4m3fn.
        from sglang.srt.mem_cache.memory_pool import maybe_poison_pool_data

        u = torch.zeros(8, dtype=torch.uint8)
        i32 = torch.zeros(8, dtype=torch.int32)
        with envs.SGLANG_POISON_POOL_DATA.override(True):
            maybe_poison_pool_data([u, i32], "test")
        self.assertTrue(torch.all(u == 0xFF))
        self.assertTrue(torch.all(i32 == 0))  # non-uint8 ints stay semantic
        for fp8_dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            decoded = u.view(fp8_dtype).float()
            self.assertTrue(
                torch.isnan(decoded).all(),
                f"0xFF must decode to NaN under {fp8_dtype}",
            )


class _FakeSpecAlgo:
    def __init__(self, none):
        self._none = none

    def is_none(self):
        return self._none


class _FakeServerArgs:
    """Just enough surface for handle_max_mamba_cache's disable-radix branch."""

    def __init__(
        self,
        *,
        max_running_requests,
        speculative_num_draft_tokens,
        disable_radix_cache=True,
        dcp_size=1,
    ):
        self.max_mamba_cache_size = None
        self.disable_radix_cache = disable_radix_cache
        self.max_running_requests = max_running_requests
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        # #656: 8b48e32968 (2026-07-22) switched the intermediate-state
        # reservation to the MAX draft width -- the adaptive k-ladder and the
        # cross-algorithm secondary rung can exceed the boot shape's. Equal to
        # the boot width here, which is what the expectations below assume and
        # what a non-adaptive launch has.
        self.max_speculative_num_draft_tokens = speculative_num_draft_tokens
        self.enable_dp_attention = False
        self.dp_size = 1
        # #656: read by _auto_mamba_demand_active, which handle_max_mamba_cache
        # consults BEFORE it can reach the disable-radix branch. Absent, the
        # gate raised on its own first argument.
        self.dcp_size = dcp_size
        self.max_running_requests_user_set = False
        self.max_running_requests_ceiling = None
        self.mamba_full_memory_ratio = 0.9
        self.overrides = []

    def enable_mamba_extra_buffer(self):
        return False

    def uneven_memory_budgets_active(self):
        return False

    def override(self, source, **fields):
        self.overrides.append((source, fields))
        for k, v in fields.items():
            setattr(self, k, v)


#: Every mixin method ``handle_max_mamba_cache`` invokes on ``self``.
#:
#: DECLARED HERE AND CHECKED AGAINST THE CODE (#504-a one-source-of-truth,
#: #656). This stub used to bind a hand-picked two of them, and when
#: a0ed7dc810 added ``elif self._auto_mamba_demand_active():`` the list was not
#: extended -- so the stub raised AttributeError on the new call before it
#: could reach the branch these tests are about. Binding a name at a time as
#: each one blows up just re-arms the same trap for the next branch, so the
#: list is declared and
#: ``TestTheStubTracksTheFunctionItDrives`` fails when it drifts.
_HANDLE_MAX_MAMBA_CACHE_SELF_CALLS = (
    "_auto_mamba_demand_active",
    "_auto_mamba_demand_size",
    "_auto_mamba_target_concurrency",
    "_calculate_mamba_ratio",
    "_fit_mamba_pool_to_budget",
    "_mamba_pool_budget_cost_gb",
    "_stage_local_mamba_cache_per_req",
    "_stage_mamba_layer_counts",
    "_sync_uneven_mamba_cache_size",
)


def _direct_mixin_self_calls(func_name):
    """Mixin methods called as ``self.X(...)`` directly inside ``func_name``.

    A STATIC read of the real source, never a re-implementation of the call
    graph: a test that restated which helpers the function uses would keep
    passing after the function changed, which is the failure this exists to
    catch.
    """
    src = textwrap.dedent(
        inspect.getsource(getattr(ModelRunnerKVCacheMixin, func_name))
    )
    return {
        node.func.attr
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and hasattr(ModelRunnerKVCacheMixin, node.func.attr)
    }


def _mixin_self_calls(func_name):
    """TRANSITIVE closure of the above.

    One level is not enough, and finding that out cost a second round: with
    the direct callees bound, the demand path still died on
    ``_mamba_pool_budget_cost_gb``, which ``handle_max_mamba_cache`` never
    names -- ``_fit_mamba_pool_to_budget`` calls it. A stub that binds only
    what the entry point mentions re-arms the same trap one frame deeper.
    """
    seen, stack = set(), [func_name]
    while stack:
        for callee in _direct_mixin_self_calls(stack.pop()):
            if callee not in seen:
                seen.add(callee)
                stack.append(callee)
    return seen


def _make_mock_runner(per_req_bytes, server_args, has_spec):
    mock = SimpleNamespace(
        mambaish_config=SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(mamba_cache_per_req=per_req_bytes)
        ),
        server_args=server_args,
        spec_algorithm=_FakeSpecAlgo(none=not has_spec),
        dp_size=1,
    )
    # BOUND FROM THE REAL CLASS, not stubbed out. The point of driving the
    # production function is that the branches it takes are the production
    # branches; replacing a gate with a lambda would make the disable-radix
    # tests below pass without ever asking the gate anything.
    for name in _HANDLE_MAX_MAMBA_CACHE_SELF_CALLS:
        setattr(mock, name, MethodType(getattr(ModelRunnerKVCacheMixin, name), mock))
    mock.handle_max_mamba_cache = MethodType(
        ModelRunnerKVCacheMixin.handle_max_mamba_cache, mock
    )
    return mock


class TestCheckpointIntervalValidation(unittest.TestCase):
    """--mamba-checkpoint-interval resolution/validation on ServerArgs."""

    def _args(self, interval, **kw):
        args = ServerArgs(model_path="dummy", page_size=1)
        args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
        object.__setattr__(args, "mamba_checkpoint_interval", interval)
        for k, v in kw.items():
            object.__setattr__(args, k, v)
        return args

    def _view(self, args, track_interval=256, page_size=1):
        return SimpleNamespace(
            page_size=page_size,
            mamba_track_interval=track_interval,
        )

    def test_none_is_noop(self):
        args = self._args(None)
        args._handle_mamba_checkpoint_interval(self._view(args))

    def test_unifies_track_interval(self):
        args = self._args(2048)
        args._handle_mamba_checkpoint_interval(self._view(args))
        self.assertEqual(args.mamba_track_interval, 2048)

    def test_conflicting_track_interval_rejected(self):
        args = self._args(2048)
        with self.assertRaises(ValueError):
            args._handle_mamba_checkpoint_interval(self._view(args, track_interval=512))

    def test_chunk_multiple_required(self):
        args = self._args(FLA_CHUNK_SIZE + 1)
        with self.assertRaises(ValueError):
            args._handle_mamba_checkpoint_interval(self._view(args))

    def test_disable_radix_cache_rejected(self):
        args = self._args(2048, disable_radix_cache=True)
        with self.assertRaises(ValueError):
            args._handle_mamba_checkpoint_interval(self._view(args))

    def test_hierarchical_cache_composes(self):
        """#747: the interval x hierarchical-cache refusal is lifted -- the
        unified mamba component now mirrors the checkpoint grid at every
        decision MambaRadixCache makes (seam map in NOTE_747), so the
        combination is accepted and the tracking grid is still unified."""
        args = self._args(2048, enable_hierarchical_cache=True)
        view = self._view(args)
        args._handle_mamba_checkpoint_interval(view)  # must not raise
        self.assertEqual(args.mamba_track_interval, 2048)

    def test_hierarchical_cache_composition_still_validates_the_grid(self):
        """Lifting the refusal must not have widened the grid rules: an
        off-chunk interval stays rejected WITH hierarchical cache on."""
        args = self._args(FLA_CHUNK_SIZE + 1, enable_hierarchical_cache=True)
        with self.assertRaises(ValueError):
            args._handle_mamba_checkpoint_interval(self._view(args))

    def test_a_divisible_interval_above_the_chunk_budget_is_accepted(self):
        """#750, the user-confirmed lift: snapshot-every-8k only needs
        ``interval % chunked_prefill_size == 0`` -- with 8192 = 16 x 512
        every 16th chunk end lands exactly ON the grid, and the retention
        rule (mamba_radix_cache.py: an off-grid finish is NOT cached at
        all) cleanly drops the fifteen ends between. The old <= gate
        refused exactly this configuration."""
        args = self._args(8192, chunked_prefill_size=512)
        args._handle_mamba_checkpoint_interval(self._view(args))  # must not raise
        args2 = self._args(16384, chunked_prefill_size=8192)
        args2._handle_mamba_checkpoint_interval(self._view(args2))

    def test_a_non_divisible_interval_above_the_chunk_budget_refuses(self):
        """The other direction: 8000 % 512 = 320, so NO chunk end ever
        lands on the grid and no anchor could ever be written -- the flag
        would be silently inert (the #742 class). Refused, naming the law."""
        args = self._args(8000, chunked_prefill_size=512)
        with self.assertRaises(ValueError) as ctx:
            args._handle_mamba_checkpoint_interval(self._view(args))
        self.assertIn("multiple", str(ctx.exception))
        self.assertIn("chunked-prefill-size", str(ctx.exception))

    def test_an_interval_at_or_below_the_chunk_budget_keeps_the_old_arm(self):
        """interval <= chunk folds into the truncation alignment exactly as
        before #750 -- divisibility is not required there because the
        scheduler clips every step END to the interval itself."""
        args = self._args(2048, chunked_prefill_size=8192)
        args._handle_mamba_checkpoint_interval(self._view(args))
        args2 = self._args(2048, chunked_prefill_size=2048)
        args2._handle_mamba_checkpoint_interval(self._view(args2))


class TestSparseGridFalsifikator(unittest.TestCase):
    """#750 FALSIFIKATOR: interval=8192 at chunk=512 writes an anchor at
    EXACTLY the 16th chunk end and NONE between.

    Driven through the real tracking arithmetic
    (``mamba_checkpoint_track_target``) over 32 consecutive full prefill
    steps, plus the retention rule that guards the cache side. Both
    directions: the grid position must be reachable (a target that never
    fires would make the whole lift a silently-inert flag), and no
    off-grid end may produce one.
    """

    INTERVAL = 8192
    PREFILL_CHUNK = 512

    def test_anchor_exactly_at_the_16th_chunk_end_and_none_between(self):
        from sglang.srt.mem_cache.mamba_ckpt_utils import (
            mamba_checkpoint_track_target,
        )

        targets = []
        for step in range(32):
            prefix = step * self.PREFILL_CHUNK
            t = mamba_checkpoint_track_target(
                prefix, self.PREFILL_CHUNK, self.INTERVAL, FLA_CHUNK_SIZE
            )
            if t is not None:
                targets.append((step, t))
        self.assertEqual(
            targets,
            [(15, 8192), (31, 16384)],
            "expected an anchor target at exactly the 16th and 32nd chunk "
            "ends and none between",
        )

    def test_every_target_is_the_step_end_itself(self):
        """With a divisible interval > chunk, a grid boundary inside a step
        can only be the step's END (the only 512-multiple in a 512-token
        window), so the scheduler's last-position routing
        (last_recurrent_state) serves every anchor and the mid-step
        ``+1``/intermediate-h arm is never needed."""
        from sglang.srt.mem_cache.mamba_ckpt_utils import (
            mamba_checkpoint_track_target,
        )

        for step in range(64):
            prefix = step * self.PREFILL_CHUNK
            end = prefix + self.PREFILL_CHUNK
            t = mamba_checkpoint_track_target(
                prefix, self.PREFILL_CHUNK, self.INTERVAL, FLA_CHUNK_SIZE
            )
            if t is not None:
                self.assertEqual(t, end)

    def test_off_grid_ends_are_refused_by_the_retention_rule(self):
        """The cache half of the falsifier: all fifteen intermediate chunk
        ends are off-grid and the retention rule drops them; only 8192
        itself is a legal anchor position."""
        for n in range(1, 16):
            self.assertFalse(is_on_interval(n * self.PREFILL_CHUNK, self.INTERVAL))
        self.assertTrue(is_on_interval(16 * self.PREFILL_CHUNK, self.INTERVAL))

    def test_a_non_divisible_interval_collapses_the_end_anchor_cadence(self):
        """WHY the validation refuses 8000@512. The END-donation path (the
        live ``no_buffer`` strategy caches only at step ends) anchors a
        chunk end only where ``n * 512`` is an 8000-multiple -- every
        lcm(8000, 512) = 64,000 tokens, an 8x collapse below the requested
        cadence. The user asked for anchors every ~8k; 8000@512 silently
        delivers one per 64k while LOOKING configured. (The extra_buffer
        tracking arm could still hit mid-step targets where the offset is
        FLA-aligned, which is exactly why this is refused at validation
        rather than left as a strategy-dependent surprise.)"""
        import math

        lcm = math.lcm(8000, self.PREFILL_CHUNK)
        self.assertEqual(lcm, 64000)
        on_grid_ends = [
            n
            for n in range(1, lcm // self.PREFILL_CHUNK + 1)
            if is_on_interval(n * self.PREFILL_CHUNK, 8000)
        ]
        self.assertEqual(
            on_grid_ends,
            [lcm // self.PREFILL_CHUNK],
            "chunk ends land on an 8000 grid only at the lcm -- one end "
            "anchor per 64k tokens instead of per 8k",
        )


class TestCheckpointTruncationAlign(unittest.TestCase):
    """#750 (2): the interval folds into the prefill truncation alignment
    ONLY while interval <= chunk; a divisible sparse grid leaves the chunk
    budget alone (512 stays 512) and the deterministic-inference alignment
    untouched."""

    def _fold(self, existing, interval, chunk):
        from sglang.srt.mem_cache.mamba_ckpt_utils import (
            checkpoint_truncation_align,
        )

        return checkpoint_truncation_align(existing, interval, chunk)

    def test_small_interval_folds_exactly_as_before(self):
        align, folded = self._fold(None, 2048, 8192)
        self.assertEqual((align, folded), (2048, True))

    def test_sparse_interval_does_not_fold(self):
        align, folded = self._fold(None, 8192, 512)
        self.assertEqual((align, folded), (None, False))

    def test_det_inference_lcm_still_applies_when_folded(self):
        align, folded = self._fold(4096, 2048, 8192)
        self.assertEqual((align, folded), (4096, True))  # lcm(4096, 2048)

    def test_det_inference_align_survives_a_sparse_interval_untouched(self):
        """Both sources present, interval > chunk: the deterministic-
        inference alignment stays exactly what the backend set -- folding
        an 8192 interval in would inflate it 16x and starve the chunk
        budget (the C30 refusal the old coupling forced)."""
        align, folded = self._fold(4096, 8192, 4096)
        self.assertEqual((align, folded), (4096, False))

    def test_no_interval_is_identity(self):
        self.assertEqual(self._fold(4096, None, 512), (4096, False))
        self.assertEqual(self._fold(None, None, 512), (None, False))

    def test_chunked_prefill_off_keeps_the_old_fold(self):
        """Without chunked prefill there is no chunk budget to preserve and
        the pre-#750 behaviour (align to the interval) stands."""
        align, folded = self._fold(None, 8192, None)
        self.assertEqual((align, folded), (8192, True))

    def test_the_scheduler_routes_through_the_helper(self):
        """Source pin, the #747 discipline: the fold decision must live in
        ONE place or the validation's promise and the scheduler's behaviour
        drift apart."""
        import inspect

        from sglang.srt.managers import scheduler as sched_mod

        src = inspect.getsource(sched_mod)
        self.assertIn("checkpoint_truncation_align(", src)


class TestDisableRadixJointFit(unittest.TestCase):
    """C: --disable-radix-cache sizes the mamba pool against the budget."""

    PER_REQ = 64 << 20  # 64 MiB per request state

    def test_large_budget_keeps_requested_size(self):
        args = _FakeServerArgs(max_running_requests=48, speculative_num_draft_tokens=4)
        mock = _make_mock_runner(self.PER_REQ, args, has_spec=True)
        rest = mock.handle_max_mamba_cache(1000.0)  # 1000 GB: no cap
        self.assertEqual(args.max_mamba_cache_size, 48)
        # main (48) + intermediate (48*4) states subtracted
        expected = 1000.0 - (48 * 4 + 48) * self.PER_REQ / GB
        self.assertAlmostEqual(rest, expected, places=3)
        self.assertGreater(rest, 0)

    def test_small_budget_caps_size_and_keeps_rest_positive(self):
        # Old behavior reserved 48*(1+4)*64MiB = 15 GB from a 4 GB budget and
        # drove rest_memory negative -> misleading "no memory for KV" error.
        args = _FakeServerArgs(max_running_requests=48, speculative_num_draft_tokens=4)
        mock = _make_mock_runner(self.PER_REQ, args, has_spec=True)
        total = 4.0  # GB
        rest = mock.handle_max_mamba_cache(total)
        budget_bytes = total * 0.9 / 1.9 * GB
        expected_size = int(budget_bytes // (self.PER_REQ * (1 + 4)))
        self.assertEqual(args.max_mamba_cache_size, expected_size)
        self.assertLess(args.max_mamba_cache_size, 48)
        self.assertGreater(args.max_mamba_cache_size, 0)
        self.assertGreater(rest, 0)

    def test_no_spec_dec_path(self):
        args = _FakeServerArgs(
            max_running_requests=48, speculative_num_draft_tokens=None
        )
        mock = _make_mock_runner(self.PER_REQ, args, has_spec=False)
        rest = mock.handle_max_mamba_cache(4.0)
        budget_bytes = 4.0 * 0.9 / 1.9 * GB
        self.assertEqual(args.max_mamba_cache_size, int(budget_bytes // self.PER_REQ))
        self.assertGreater(rest, 0)

    def test_zero_budget_raises_actionable_error(self):
        args = _FakeServerArgs(max_running_requests=48, speculative_num_draft_tokens=4)
        mock = _make_mock_runner(self.PER_REQ, args, has_spec=True)
        with self.assertRaises(RuntimeError) as ctx:
            mock.handle_max_mamba_cache(0.01)
        self.assertIn("max_mamba_cache_size", str(ctx.exception))


class TestTheStubTracksTheFunctionItDrives(unittest.TestCase):
    """#656: the stub's bound-method list against the real call graph.

    ORIGIN. a0ed7dc810 ("[DCP] Auto-size the mamba state pool by demand under
    uneven DCP", 2026-07-15) added 132 lines to model_runner_kv_cache_mixin.py
    -- including `elif self._auto_mamba_demand_active():` in
    handle_max_mamba_cache -- and touched no test file. This stub bound two of
    the mixin methods that function calls, so from that commit on it raised

        AttributeError: 'types.SimpleNamespace' object has no attribute
        '_auto_mamba_demand_active'

    before reaching the disable-radix branch TestDisableRadixJointFit is about.

    IT WAS INVISIBLE FOR SIX WEEKS FOR A SEPARATE REASON. d8a5a25c36
    (upstream #25173, 2026-06-01) had already made this whole directory
    uncollectable with an unguarded NIXL import, six weeks BEFORE the defect
    landed -- the guard was down when the break arrived. Both halves are
    needed to explain it, and only the second was ever recorded.

    NOT A LIVE SERVING DEFECT. ModelRunner inherits ModelRunnerKVCacheMixin
    (model_runner.py:426), the sole production caller is
    `self.handle_max_mamba_cache(rest_memory)` inside that same mixin
    (model_runner_kv_cache_mixin.py:725), and no production code anywhere in
    model_executor binds these methods with MethodType. Every real `self` has
    every one of them. The defect is confined to this harness.
    """

    def test_the_declared_list_matches_the_functions_real_self_calls(self):
        self.assertEqual(
            set(_HANDLE_MAX_MAMBA_CACHE_SELF_CALLS),
            _mixin_self_calls("handle_max_mamba_cache"),
            "handle_max_mamba_cache calls a different set of mixin methods "
            "than this stub binds. Add the new name to "
            "_HANDLE_MAX_MAMBA_CACHE_SELF_CALLS -- an unbound one raises "
            "AttributeError on whichever branch reaches it first, which is "
            "how a0ed7dc810 broke four tests nobody could see",
        )

    def test_the_gate_is_bound_from_the_real_class(self):
        """Not stubbed. A lambda here would let every test below pass while
        the production gate went unasked."""
        args = _FakeServerArgs(max_running_requests=48, speculative_num_draft_tokens=4)
        mock = _make_mock_runner(1 << 20, args, has_spec=True)
        self.assertIs(
            mock._auto_mamba_demand_active.__func__,
            ModelRunnerKVCacheMixin._auto_mamba_demand_active,
        )


class TestTheDemandGateIsConsultedBothWays(unittest.TestCase):
    """Both directions of `_auto_mamba_demand_active`, through the real call.

    The four TestDisableRadixJointFit cases pin the gate answering FALSE -- if
    binding the method had flipped them to the demand branch, their expected
    values would have moved and the "fix" would have been a behaviour change
    wearing a harness commit. That direction is asserted here explicitly
    rather than left implicit in four unchanged numbers.
    """

    PER_REQ = 64 << 20

    def tearDown(self):
        set_cp_token_ratios(None)

    def test_the_stock_stub_takes_the_fixed_fraction_path(self):
        set_cp_token_ratios(None)
        args = _FakeServerArgs(max_running_requests=48, speculative_num_draft_tokens=4)
        mock = _make_mock_runner(self.PER_REQ, args, has_spec=True)
        self.assertFalse(mock._auto_mamba_demand_active())
        mock.handle_max_mamba_cache(1000.0)
        self.assertNotIn(
            "mamba_pool.demand_driven", [src for src, _f in args.overrides]
        )

    def test_an_uneven_vector_with_radix_on_takes_the_demand_path(self):
        """The TRUE direction. Without it, binding the method would be pinned
        only where it answers False, i.e. the new branch would still be
        untested by this file."""
        set_cp_token_ratios([30, 17, 17])
        args = _FakeServerArgs(
            max_running_requests=48,
            speculative_num_draft_tokens=None,
            disable_radix_cache=False,
            dcp_size=3,
        )
        mock = _make_mock_runner(self.PER_REQ, args, has_spec=False)
        self.assertTrue(mock._auto_mamba_demand_active())
        mock.handle_max_mamba_cache(1000.0)
        self.assertIn("mamba_pool.demand_driven", [src for src, _f in args.overrides])
        self.assertIsNotNone(args.max_mamba_cache_size)
        self.assertGreater(args.max_mamba_cache_size, 0)

    def test_a_uniform_vector_is_not_uneven_and_stays_on_the_fixed_path(self):
        """All-equal vectors use the even modulo fast path, so the gate must
        say False even with a vector installed and radix on."""
        set_cp_token_ratios([1, 1, 1])
        args = _FakeServerArgs(
            max_running_requests=48,
            speculative_num_draft_tokens=None,
            disable_radix_cache=False,
            dcp_size=3,
        )
        mock = _make_mock_runner(self.PER_REQ, args, has_spec=False)
        self.assertFalse(mock._auto_mamba_demand_active())

    def test_an_explicit_cache_size_never_reaches_the_gate(self):
        """max_mamba_cache_size is the branch ABOVE the elif; an explicit pin
        must win even under uneven DCP."""
        set_cp_token_ratios([30, 17, 17])
        args = _FakeServerArgs(
            max_running_requests=48,
            speculative_num_draft_tokens=None,
            disable_radix_cache=False,
            dcp_size=3,
        )
        args.max_mamba_cache_size = 1234
        mock = _make_mock_runner(self.PER_REQ, args, has_spec=False)
        mock.handle_max_mamba_cache(1000.0)
        self.assertIn("mamba_pool.per_dp_shard", [src for src, _f in args.overrides])
        self.assertNotIn(
            "mamba_pool.demand_driven", [src for src, _f in args.overrides]
        )


class TestReplaySsmRadixSeam(unittest.TestCase):
    """#700: the ReplaySSM ring vs the radix/prefix-cache seam.

    The kernel header of ``fla/fused_recurrent_linear_replayssm.py`` states the
    feature is "NOT yet wired into the memory pool / radix cache / scheduler /
    backend dispatch". These tests EXECUTE the pool half of that claim, because
    a doc read alone cannot distinguish a stale comment from an unwired path.

    The seam that matters: ``temporal[slot]`` lags the live state by the slot's
    unflushed ring depth, so any donate/copy of an un-flushed slot would pair a
    stale checkpoint with a longer key. ``MambaPool.copy_from`` documents the
    invariant (source must satisfy ``write_pos == 0``) and names its three
    compliant callers.
    """

    def _pool(self):
        _tree, _alloc, pool, _mk = _build_tree(None, enable_linear_replayssm=True)
        return pool.mamba_pool

    def test_flag_actually_allocates_the_rings_and_cursor(self):
        """Execution proof that the flag engages the POOL wiring."""
        mp = self._pool()
        self.assertIsNotNone(mp.replayssm_write_pos)
        d = mp.mamba_cache.replayssm_d
        k = mp.mamba_cache.replayssm_k
        g = mp.mamba_cache.replayssm_g
        for t in (d, k, g):
            self.assertIsNotNone(t)
        # Documented layout: d [layers, slots, HV, L, V]; k [layers, slots, H, L, K];
        # g [layers, slots, HV, L]. Pin the invariants, not the rig's dims.
        self.assertEqual(d.ndim, 5)
        self.assertEqual(k.ndim, 5)
        self.assertEqual(g.ndim, 4)
        self.assertEqual(d.shape[:2], k.shape[:2])
        self.assertEqual(d.shape[:2], g.shape[:2])
        # The ring depth L is the same axis in all three.
        self.assertEqual(d.shape[3], k.shape[3])
        self.assertEqual(d.shape[3], g.shape[3])
        # The gate ring is fp32 even when the state rings are not.
        self.assertEqual(g.dtype, torch.float32)
        # One cursor per slot, plus the guard row.
        self.assertEqual(mp.replayssm_write_pos.ndim, 1)
        self.assertEqual(mp.replayssm_write_pos.shape[0], d.shape[1])

    def test_copy_from_resets_the_destination_cursor(self):
        """A copied checkpoint has no pending ring entries, so dst must be 0."""
        mp = self._pool()
        src = torch.tensor([1], dtype=torch.int64)
        dst = torch.tensor([2], dtype=torch.int64)
        mp.replayssm_write_pos[dst] = 5
        mp.copy_from(src, dst)
        self.assertEqual(int(mp.replayssm_write_pos[dst].item()), 0)

    def test_unflushed_source_is_caught_when_the_debug_guard_is_on(self):
        """The can-fail proof: the documented invariant really can trip."""
        mp = self._pool()
        src = torch.tensor([1], dtype=torch.int64)
        dst = torch.tensor([2], dtype=torch.int64)
        mp.replayssm_write_pos[src] = 3
        mp.debug_memory_pool = True
        with self.assertRaises(AssertionError):
            mp.copy_from(src, dst)

    def test_unflushed_source_is_NOT_guarded_in_production(self):
        """Pins the gap as a fact, not a wish.

        The ``copy_from`` invariant is enforced only under ``debug_memory_pool``,
        which is off by default. In production the invariant rests entirely on
        caller discipline documented in a docstring, so a fourth caller added
        later would violate it silently. This test exists so that if the guard
        is ever made unconditional, it fails and someone deletes this test on
        purpose rather than the protection regressing unnoticed.
        """
        mp = self._pool()
        src = torch.tensor([1], dtype=torch.int64)
        dst = torch.tensor([2], dtype=torch.int64)
        mp.replayssm_write_pos[src] = 3
        mp.debug_memory_pool = False
        mp.copy_from(src, dst)  # silently proceeds
        self.assertEqual(int(mp.replayssm_write_pos[dst].item()), 0)


if __name__ == "__main__":
    unittest.main()
