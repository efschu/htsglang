"""Hermetic (CPU-only) regression tests for the mamba pool demand floor (#581).

Production incident chain, all on Qwen3.6-27B (GDN hybrid), TP=3 uneven DCP:

* crashes 2+3 died in `MambaRadixCache._alloc_mamba_slot` -- fixed earlier by
  making CACHE-INSERT paths degrade (see test_mamba_slot_starvation.py).
* crash 6 died at a SIBLING site that fix did not cover:
  `HybridReqToTokenPool._alloc_ping_pong_buffer`, a bare
  ``assert slots is not None`` with no evict attempt, reached from
  `HybridReqToTokenPool.alloc`. Ping-pong track buffers are a REQUIRED
  allocation, so "skip the cache insert" is not available -- the correct
  degradation is to fail the whole `alloc` cleanly and let admission defer.

The pool was sized 64 by hand after a 25-slot auto value died; both numbers
are far ABOVE the structural floor (20 for this configuration), which is the
point: the pool was not undersized, it was being eaten by hicache
write-through pins that the eviction walk cannot reclaim. Hence three
independent guards, one test class each:

1. the demand FLOOR is computed, not guessed, and refused at parse time;
2. the REQUIRED allocation sites evict and then degrade instead of asserting;
3. the write-through pin set is BOUNDED, so no pool size can be exhausted by
   pins alone.

These tests never touch a GPU: every pool is built on `torch.device("cpu")`.
"""

import types
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
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.mamba_pool_floor import (
    MAMBA_FLOOR_ACTIVE_SLOTS,
    MAMBA_FLOOR_DONATION_SLOTS,
    MAMBA_FLOOR_PINNED_CHECKPOINT_SLOTS,
    mamba_hard_floor,
    mamba_ping_pong_slots,
    mamba_slots_per_running_req,
)
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25)

NUM_LAYERS = 8
GLOBAL_INTERVAL = 4
KV_POOL_SIZE = 512
MAX_CONTEXT_LEN = 128


def _server_args(**kwargs) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy", page_size=1, **kwargs)
    # The property would otherwise load the HF config of the dummy model.
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    return server_args


def _build_tree(mamba_size=6, enable_mamba_extra_buffer=False, max_num_reqs=10):
    """MambaRadixCache + pools pinned to CPU (no accelerator required)."""
    server_args = _server_args()
    if enable_mamba_extra_buffer:
        server_args.mamba_radix_cache_strategy = "extra_buffer"
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


def _make_req(token_ids, rid="r0"):
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", token_ids),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )


def _fill_pool_with_locked_nodes(tree, allocator, req_to_token_pool, base=1000):
    """Consume every remaining mamba slot with cached nodes locked by a
    (simulated) running request: pool empty, `mamba_evictable_size() == 0`."""
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


class TestFloorFormula(unittest.TestCase):
    """The floor is DERIVED from the allocation sites, not a copied literal.

    Every expectation below is expressed in terms of the module constants and
    the ServerArgs shape, so changing an allocation site without updating the
    formula fails here rather than in production.
    """

    def test_ping_pong_slots_follow_overlap_schedule(self):
        """Mirrors HybridReqToTokenPool.mamba_ping_pong_track_buffer_size."""
        sa = _server_args()
        sa.mamba_radix_cache_strategy = "extra_buffer"
        sa.disable_overlap_schedule = False
        self.assertEqual(mamba_ping_pong_slots(sa), 2)
        sa.disable_overlap_schedule = True
        self.assertEqual(mamba_ping_pong_slots(sa), 1)

    def test_no_extra_buffer_has_no_ping_pong_slots(self):
        sa = _server_args()
        sa.mamba_radix_cache_strategy = "no_buffer"
        self.assertEqual(mamba_ping_pong_slots(sa), 0)

    def test_per_req_is_sum_of_the_named_terms(self):
        sa = _server_args()
        sa.mamba_radix_cache_strategy = "extra_buffer"
        sa.disable_overlap_schedule = False
        expected = (
            MAMBA_FLOOR_ACTIVE_SLOTS
            + mamba_ping_pong_slots(sa)
            + MAMBA_FLOOR_DONATION_SLOTS
            + MAMBA_FLOOR_PINNED_CHECKPOINT_SLOTS
        )
        self.assertEqual(mamba_slots_per_running_req(sa), expected)

    def test_per_req_matches_the_sizing_ratio(self):
        """`_calculate_mamba_ratio()` (3 + 2 with extra_buffer + overlap) is the
        same demand model seen from the sizing side; the two must agree or one
        of them is wrong."""
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO,
            MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP,
        )

        sa = _server_args()
        sa.mamba_radix_cache_strategy = "extra_buffer"
        sa.disable_overlap_schedule = False
        self.assertEqual(
            mamba_slots_per_running_req(sa),
            MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO
            + MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP,
        )

    def test_radix_disabled_only_charges_the_active_slot(self):
        sa = _server_args()
        sa.disable_radix_cache = True
        self.assertEqual(mamba_slots_per_running_req(sa), MAMBA_FLOOR_ACTIVE_SLOTS)

    def test_production_config_floor(self):
        """The exact shape of the crashed boot: max_running_requests=4,
        extra_buffer, overlap on."""
        sa = _server_args()
        sa.mamba_radix_cache_strategy = "extra_buffer"
        sa.disable_overlap_schedule = False
        self.assertEqual(mamba_slots_per_running_req(sa), 5)
        self.assertEqual(mamba_hard_floor(sa, 4), 20)


class TestBootFloorValidation(unittest.TestCase):
    """Explicit --max-mamba-cache-size below the floor is refused at parse time."""

    def _args(self, size, running=4):
        sa = _server_args()
        sa.mamba_radix_cache_strategy = "extra_buffer"
        sa.disable_overlap_schedule = False
        sa.max_mamba_cache_size = size
        sa.max_running_requests = running
        return sa

    def test_below_floor_is_refused_with_the_numbers(self):
        """FALSIFIER: this is the value class ops kept guessing at."""
        sa = self._args(19)
        with self.assertRaises(ValueError) as ctx:
            sa._validate_max_mamba_cache_size(sa)
        msg = str(ctx.exception)
        self.assertIn("19", msg)
        self.assertIn("20", msg)  # the computed floor
        self.assertIn("4 x 5", msg)  # the derivation

    def test_can_fail_proof_floor_boundary(self):
        """Exactly at the floor passes, one below raises -- so the assertion
        above is not vacuous."""
        self._args(20)._validate_max_mamba_cache_size(self._args(20))
        with self.assertRaises(ValueError):
            sa = self._args(19)
            sa._validate_max_mamba_cache_size(sa)

    def test_above_floor_is_untouched(self):
        sa = self._args(64)
        sa._validate_max_mamba_cache_size(sa)
        self.assertEqual(sa.max_mamba_cache_size, 64)

    def test_unset_size_is_not_validated(self):
        sa = self._args(None)
        sa._validate_max_mamba_cache_size(sa)

    def test_the_crashed_auto_value_would_now_be_accepted(self):
        """25 (the auto value that died) is above the floor of 20 -- the floor
        check alone would NOT have caught it, which is why the pin bound below
        exists. Pinning this keeps the diagnosis honest."""
        sa = self._args(25)
        sa._validate_max_mamba_cache_size(sa)


class TestRequiredAllocSitesDegrade(unittest.TestCase):
    """FALSIFIERS for the two sibling sites that killed rank 0 in crash 6."""

    def test_alloc_returns_none_instead_of_asserting_on_active_slot(self):
        """FALSIFIER: memory_pool.py HybridReqToTokenPool.alloc active-slot site.

        Before the fix this raised
        ``AssertionError: Not enough space for mamba cache``."""
        tree, allocator, pool = _build_tree(mamba_size=4)
        _fill_pool_with_locked_nodes(tree, allocator, pool)

        self.assertIsNone(pool.alloc([_make_req([1, 2, 3])]))

    def test_alloc_returns_none_instead_of_asserting_on_ping_pong(self):
        """FALSIFIER: the exact crash-6 site, `_alloc_ping_pong_buffer`.

        Reached with a request that ALREADY owns its active state slot (a
        chunked-prefill continuation, `req.mamba_pool_idx is not None`), so
        the active-slot branch is skipped and the ping-pong allocation is the
        first one that can fail -- exactly the production traceback:
        `alloc` -> `_alloc_ping_pong_buffer` -> ``AssertionError: Not enough
        space for mamba ping pong idx``.
        """
        tree, allocator, pool = _build_tree(
            mamba_size=8, enable_mamba_extra_buffer=True
        )
        req = _make_req([1, 2, 3])
        # Give the request its active slot first, then starve the pool.
        own = pool.mamba_allocator.alloc(1)
        self.assertIsNotNone(own)
        req.mamba_pool_idx = own[0]
        req.inflight_middle_chunks = 1
        _fill_pool_with_locked_nodes(tree, allocator, pool)
        self.assertIsNone(req.mamba_ping_pong_track_buffer)

        self.assertIsNone(pool.alloc([req]))
        # The request keeps the active slot it already owned: the rollback
        # must only undo what THIS call allocated.
        self.assertIsNotNone(req.mamba_pool_idx)
        self.assertEqual(int(req.mamba_pool_idx), int(own[0]))

    def test_ping_pong_exhaustion_leaks_nothing(self):
        """A deferred batch must return the pool to exactly its prior state --
        the old assert fired AFTER partially mutating it."""
        tree, allocator, pool = _build_tree(
            mamba_size=8, enable_mamba_extra_buffer=True
        )
        _fill_pool_with_locked_nodes(tree, allocator, pool)

        before = (
            pool.mamba_allocator.available_size(),
            len(pool.free_slots),
            tree.mamba_evictable_size(),
            tree.mamba_protected_size(),
        )
        self.assertIsNone(pool.alloc([_make_req([1, 2, 3]), _make_req([4, 5], "r1")]))
        after = (
            pool.mamba_allocator.available_size(),
            len(pool.free_slots),
            tree.mamba_evictable_size(),
            tree.mamba_protected_size(),
        )
        self.assertEqual(before, after)

    def test_deferred_reqs_keep_no_pool_state(self):
        """Rolled-back requests must look untouched, or the next admission
        attempt double-frees or silently reuses a returned slot."""
        tree, allocator, pool = _build_tree(
            mamba_size=8, enable_mamba_extra_buffer=True
        )
        _fill_pool_with_locked_nodes(tree, allocator, pool)

        reqs = [_make_req([1, 2, 3]), _make_req([4, 5], "r1")]
        self.assertIsNone(pool.alloc(reqs))
        for req in reqs:
            self.assertIsNone(req.req_pool_idx)
            self.assertIsNone(req.mamba_pool_idx)
            self.assertIsNone(req.mamba_ping_pong_track_buffer)

    def test_alloc_evicts_before_giving_up(self):
        """With evictable (unlocked) cached states the pool must reclaim them
        rather than defer -- otherwise the degradation would throttle a server
        that has plenty of reclaimable cache."""
        tree, allocator, pool = _build_tree(
            mamba_size=8, enable_mamba_extra_buffer=True
        )
        # Fill the pool with cached states that are NOT locked.
        while True:
            slot = pool.mamba_allocator.alloc(1)
            if slot is None:
                break
            ids = list(range(2000 + 10 * int(slot[0]), 2000 + 10 * int(slot[0]) + 4))
            tree.insert(
                InsertParams(
                    key=RadixKey(array("q", ids)),
                    value=allocator.alloc(len(ids)),
                    mamba_value=slot,
                )
            )
        self.assertEqual(pool.mamba_allocator.available_size(), 0)
        self.assertGreater(tree.mamba_evictable_size(), 0)

        self.assertIsNotNone(pool.alloc([_make_req([1, 2, 3])]))


class _PinHarness:
    """Binds the REAL HiMambaRadixCache pin-accounting methods onto a plain
    MambaRadixCache instance.

    The methods under test (`_mamba_pins_held`, `_forget_write_through`,
    `_drain_acked_write`, and the `_mamba_pin_budget` property) touch only the
    pin bookkeeping, the request pool and inc/dec_lock_ref -- all of which a
    MambaRadixCache already provides. This exercises the shipped code without
    standing up a hicache controller and its DMA threads.
    """

    def __init__(self, tree, pool, pin_budget=None):
        self.tree = tree
        tree.ongoing_write_through = {}
        tree._write_through_pinned = {}
        tree._write_through_inflight = {}
        tree._mamba_pin_budget_cached = pin_budget
        # `_mamba_pin_budget` is a property on HiMambaRadixCache and cannot be
        # bound onto a foreign instance; mirror the resolved value instead.
        tree._mamba_pin_budget = pin_budget
        tree._mamba_pin_skipped = 0
        tree.enable_storage = False
        tree._record_store_event = lambda *a, **k: None
        for name in (
            "_mamba_pins_held",
            "_forget_write_through",
            "_drain_acked_write",
        ):
            setattr(
                tree, name, types.MethodType(getattr(HiMambaRadixCache, name), tree)
            )

    def pin(self, node):
        """What `write_backup` does for a node it decides to back up."""
        self.tree.ongoing_write_through[node.id] = node
        self.tree._write_through_inflight[node.id] = (
            self.tree._write_through_inflight.get(node.id, 0) + 1
        )
        self.tree.inc_lock_ref(node)
        self.tree._write_through_pinned[node.id] = (
            self.tree._write_through_pinned.get(node.id, 0) + 1
        )


class TestWriteThroughPinAccounting(unittest.TestCase):
    """FALSIFIERS for the retention leak: hicache write-through pins.

    `write_backup` takes an `inc_lock_ref` for the duration of the D->H copy.
    On a node carrying a mamba checkpoint that lock also makes the state slot
    unevictable, and with `--hicache-write-policy write_through` the threshold
    is 1, so EVERY inserted checkpoint is pinned the moment it is created.
    """

    def _tree_with_cached_nodes(self, count):
        tree, allocator, pool = _build_tree(mamba_size=count + 2)
        nodes = []
        for i in range(count):
            slot = pool.mamba_allocator.alloc(1)
            ids = list(range(3000 + 10 * i, 3000 + 10 * i + 4))
            tree.insert(
                InsertParams(
                    key=RadixKey(array("q", ids)),
                    value=allocator.alloc(len(ids)),
                    mamba_value=slot,
                )
            )
            nodes.append(
                tree.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", ids)))
                ).last_device_node
            )
        return tree, allocator, pool, nodes

    def test_pins_make_cached_states_unevictable(self):
        """Root-cause pin: a pinned checkpoint is skipped by the eviction walk
        exactly like one held by a running request -- which is why the crash
        log said 'all cached states are locked by running requests' while only
        two requests were running."""
        tree, _, _, nodes = self._tree_with_cached_nodes(4)
        harness = _PinHarness(tree, None)

        self.assertGreater(tree.mamba_evictable_size(), 0)
        for node in nodes:
            harness.pin(node)
        self.assertEqual(tree.mamba_evictable_size(), 0)
        self.assertEqual(tree.evict_mamba(1), 0)

    def test_acked_write_releases_the_pin(self):
        """The non-blocking drain must return the state to evictable."""
        tree, _, _, nodes = self._tree_with_cached_nodes(3)
        harness = _PinHarness(tree, None)
        for node in nodes:
            harness.pin(node)
        self.assertEqual(tree.mamba_evictable_size(), 0)

        for node in nodes:
            tree._drain_acked_write(node.id)

        self.assertGreater(tree.mamba_evictable_size(), 0)
        self.assertEqual(tree._write_through_pinned, {})
        self.assertEqual(tree.ongoing_write_through, {})

    def test_write_back_drain_also_releases_the_pin(self):
        """FALSIFIER: `writing_check(write_back=True)` used to pop these
        entries WITHOUT a dec_lock_ref, leaking the lock -- and the mamba slot
        -- permanently. Both drains now share `_drain_acked_write`."""
        tree, _, _, nodes = self._tree_with_cached_nodes(3)
        harness = _PinHarness(tree, None)
        for node in nodes:
            harness.pin(node)
        protected_before = tree.mamba_protected_size()
        self.assertGreater(protected_before, 0)

        # Exactly what the blocking branch now does per ack id.
        for node in nodes:
            tree._drain_acked_write(node.id)

        self.assertEqual(tree.mamba_protected_size(), 0)

    def test_node_removal_reconciles_a_pin_whose_ack_never_arrives(self):
        """FALSIFIER: a node dropped from the tree mid-write was never popped
        by the ack drain, so its pin was held forever. `HiRadixCache` has this
        reconciliation; the mamba variant had none."""
        tree, _, _, nodes = self._tree_with_cached_nodes(2)
        harness = _PinHarness(tree, None)
        for node in nodes:
            harness.pin(node)
        self.assertEqual(tree.mamba_protected_size(), 2)

        for node in nodes:
            tree._forget_write_through(node)

        self.assertEqual(tree.mamba_protected_size(), 0)
        self.assertEqual(tree._write_through_pinned, {})
        self.assertEqual(tree.ongoing_write_through, {})

    def test_forget_is_idempotent(self):
        """Reconciliation may race the ack drain; a double release would
        corrupt the lock accounting."""
        tree, _, _, nodes = self._tree_with_cached_nodes(1)
        harness = _PinHarness(tree, None)
        harness.pin(nodes[0])

        tree._forget_write_through(nodes[0])
        tree._forget_write_through(nodes[0])

        self.assertEqual(tree.mamba_protected_size(), 0)

    def test_pin_budget_bounds_the_protected_set(self):
        """The hard guarantee: pins may never exceed the pool above the floor,
        so a REQUIRED allocation always has somewhere to go no matter how far
        behind the ack drain is."""
        tree, _, _, nodes = self._tree_with_cached_nodes(6)
        budget = 2
        harness = _PinHarness(tree, None, pin_budget=budget)

        pinned = 0
        for node in nodes:
            if tree._mamba_pins_held() >= tree._mamba_pin_budget:
                break
            harness.pin(node)
            pinned += 1

        self.assertEqual(pinned, budget)
        self.assertLessEqual(tree._mamba_pins_held(), budget)
        # Everything above the budget stays evictable -> the pool can serve.
        self.assertGreater(tree.mamba_evictable_size(), 0)
        self.assertGreater(tree.evict_mamba(1), 0)

    def test_budget_is_pool_above_the_floor(self):
        """The budget is derived from the same formula as the boot floor."""
        sa = _server_args()
        sa.mamba_radix_cache_strategy = "extra_buffer"
        sa.disable_overlap_schedule = False
        sa.max_running_requests = 4
        set_global_server_args_for_scheduler(sa)

        tree, _, pool = _build_tree(mamba_size=64, enable_mamba_extra_buffer=True)
        set_global_server_args_for_scheduler(sa)
        tree._mamba_pin_budget_cached = None
        tree.req_to_token_pool = pool
        # Call the real property getter against this instance.
        budget = HiMambaRadixCache._mamba_pin_budget.fget(tree)

        self.assertEqual(budget, 64 - mamba_hard_floor(sa, 4))
        self.assertEqual(budget, 44)


if __name__ == "__main__":
    unittest.main()
