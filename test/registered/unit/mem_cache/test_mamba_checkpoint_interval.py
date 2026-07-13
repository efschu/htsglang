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

import unittest
from array import array
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
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
        self.assertEqual(
            pool.mamba_allocator.available_size(), mamba_pool.size
        )

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
            (d_pool.mamba_pool.mamba_cache.temporal, f_pool.mamba_pool.mamba_cache.temporal),
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
        pairs += list(zip(d_kvcache.full_kv_pool.k_buffer, f_kvcache.full_kv_pool.k_buffer))
        pairs += list(zip(d_kvcache.full_kv_pool.v_buffer, f_kvcache.full_kv_pool.v_buffer))
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
        self.assertEqual(
            req.mamba_pingpong_clear_indices.tolist(), new_slot.tolist()
        )

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


class _FakeSpecAlgo:
    def __init__(self, none):
        self._none = none

    def is_none(self):
        return self._none


class _FakeServerArgs:
    """Just enough surface for handle_max_mamba_cache's disable-radix branch."""

    def __init__(self, *, max_running_requests, speculative_num_draft_tokens):
        self.max_mamba_cache_size = None
        self.disable_radix_cache = True
        self.max_running_requests = max_running_requests
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.enable_dp_attention = False
        self.dp_size = 1
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


def _make_mock_runner(per_req_bytes, server_args, has_spec):
    mock = SimpleNamespace(
        mambaish_config=SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(mamba_cache_per_req=per_req_bytes)
        ),
        server_args=server_args,
        spec_algorithm=_FakeSpecAlgo(none=not has_spec),
    )
    mock._calculate_mamba_ratio = MethodType(
        ModelRunnerKVCacheMixin._calculate_mamba_ratio, mock
    )
    mock._sync_uneven_mamba_cache_size = MethodType(
        ModelRunnerKVCacheMixin._sync_uneven_mamba_cache_size, mock
    )
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
            args._handle_mamba_checkpoint_interval(
                self._view(args, track_interval=512)
            )

    def test_chunk_multiple_required(self):
        args = self._args(FLA_CHUNK_SIZE + 1)
        with self.assertRaises(ValueError):
            args._handle_mamba_checkpoint_interval(self._view(args))

    def test_disable_radix_cache_rejected(self):
        args = self._args(2048, disable_radix_cache=True)
        with self.assertRaises(ValueError):
            args._handle_mamba_checkpoint_interval(self._view(args))

    def test_hierarchical_cache_rejected(self):
        args = self._args(2048, enable_hierarchical_cache=True)
        with self.assertRaises(ValueError):
            args._handle_mamba_checkpoint_interval(self._view(args))

    def test_exceeding_chunked_prefill_size_rejected(self):
        args = self._args(16384, chunked_prefill_size=8192)
        with self.assertRaises(ValueError):
            args._handle_mamba_checkpoint_interval(self._view(args))


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
        self.assertEqual(
            args.max_mamba_cache_size, int(budget_bytes // self.PER_REQ)
        )
        self.assertGreater(rest, 0)

    def test_zero_budget_raises_actionable_error(self):
        args = _FakeServerArgs(max_running_requests=48, speculative_num_draft_tokens=4)
        mock = _make_mock_runner(self.PER_REQ, args, has_spec=True)
        with self.assertRaises(RuntimeError) as ctx:
            mock.handle_max_mamba_cache(0.01)
        self.assertIn("max_mamba_cache_size", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
