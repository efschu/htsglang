"""Hermetic (CPU-only) regression tests for the #581/#583 mamba slot leak.

Production symptom: the mamba slot pool (`--max-mamba-cache-size 96`) filled
monotonically under load until every cached state was locked --

    ... nothing evictable ... mamba_evictable=0 mamba_protected=90

with only TWO running requests -- and the next `alloc_req_slots` died. ~90
tree nodes held `mamba_lock_ref > 0` that no live request referenced.

The pairing invariant that was broken:

    a release may only decrement the refs its paired acquire took, and every
    ref an acquire takes must be representable in the registry that the
    release resolves through.

Two independent violations, both exercised below:

1. `inc_lock_ref` on a MAMBA TOMBSTONE takes no mamba ref, but `load_back`
   -- which runs INSIDE that lock, from `PrefillAdder._lock_node` ->
   `init_load_back` -- revives the checkpoint and takes its OWN pin. The
   release then found a lockable mamba value and consumed the load-back's
   pin. The restored state became evictable while its H2D copy was still in
   flight, was evicted again, and the same node had to be loaded back once
   more. `IncLockRefResult.skip_lock_node_ids` is the existing contract for
   exactly this (see `unified_cache_components/mamba_component.py`); the
   mamba radix caches did not honour it.

2. `ongoing_load_back` / `_write_through_pinned` are keyed by node id and
   held ONE entry per node. The re-entry produced by (1) registered the same
   node twice while the first copy was in flight: two `inc_lock_ref`s, one
   `dec_lock_ref`, and the surplus ref -- with it the mamba slot -- stranded
   on that node forever. The second ack then also hit `KeyError` on the
   `pop`ped id.

NOT the mechanism (checked, and pinned by
`test_split_between_acquire_and_release_keeps_the_pairing`): a radix split
does not strand the ref. `_split_node` leaves `mamba_value` AND
`mamba_lock_ref` on the deeper child -- the same object the acquirer holds --
and gives the new upper half `mamba_value=None, mamba_lock_ref=0`.

Every pool here is built on `torch.device("cpu")`; the hierarchical cache runs
against a fake cache controller, so no GPU and no host pool assembly.
"""

import unittest
from array import array
from typing import List, Optional

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache, HostLRUList
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache, TreeNode
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25)

NUM_LAYERS = 8
GLOBAL_INTERVAL = 4
KV_POOL_SIZE = 4096
MAX_CONTEXT_LEN = 256


# --------------------------------------------------------------------------
# CPU pools
# --------------------------------------------------------------------------


def _build_pools(mamba_size: int, max_num_reqs: int = 8):
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
        enable_mamba_extra_buffer=False,
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
    return req_to_token_pool, allocator


def _cache_init_params(req_to_token_pool, allocator) -> CacheInitParams:
    return CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        disable=False,
        enable_kv_cache_events=False,
        enable_mamba_extra_buffer=False,
    )


# --------------------------------------------------------------------------
# Fake host tier (no GPU, no threads): transfers complete when the test says so
# --------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self, ready: bool = True):
        self.ready = ready

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        pass


class _FakeHostPool:
    def __init__(self, size: int):
        self.size = size
        self.free_list = list(range(size))

    def alloc(self, n: int) -> Optional[torch.Tensor]:
        if len(self.free_list) < n:
            return None
        return torch.tensor([self.free_list.pop() for _ in range(n)], dtype=torch.int64)

    def free(self, indices: torch.Tensor) -> None:
        self.free_list.extend(int(v) for v in indices.tolist())

    def clear(self) -> None:
        self.free_list = list(range(self.size))

    def available_size(self) -> int:
        return len(self.free_list)


class _FakeController:
    """Mirrors the parts of HybridCacheController the cache drives.

    Writes/loads are queued and only acked when the test calls `complete()`,
    so a transfer can be held in flight across tree mutations.
    """

    write_policy = "write_through"

    def __init__(self, kv_host, mamba_host, mamba_allocator, kv_device):
        self.ack_write_queue: List[tuple] = []
        self.ack_load_queue: List[tuple] = []
        self.kv_host = kv_host
        self.mamba_host = mamba_host
        self.mamba_allocator = mamba_allocator
        self.kv_device = kv_device
        self.queued_writes: List[int] = []
        self.queued_loads: List[int] = []

    def write(self, device_indices, node_id, extra_pools=None):
        host_indices = self.kv_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        for transfer in extra_pools or []:
            if transfer.host_indices is None:
                transfer.host_indices = self.mamba_host.alloc(1)
        self.queued_writes.append(node_id)
        return host_indices

    def load(self, host_indices, node_id, extra_pools=None):
        if len(host_indices) > 0:
            device_indices = self.kv_device.alloc(len(host_indices))
            if device_indices is None:
                return None
        else:
            device_indices = torch.empty((0,), dtype=torch.int64)
        for transfer in extra_pools or []:
            if transfer.device_indices is None:
                allocated = self.mamba_allocator.alloc(len(transfer.host_indices))
                if allocated is None:
                    return None
                transfer.device_indices = allocated
        self.queued_loads.append(node_id)
        return device_indices

    def complete(self) -> None:
        """Ack everything queued so far."""
        if self.queued_writes:
            self.ack_write_queue.append((None, _FakeEvent(), self.queued_writes))
            self.queued_writes = []
        if self.queued_loads:
            self.ack_load_queue.append((None, _FakeEvent(), self.queued_loads))
            self.queued_loads = []

    def start_loading(self) -> int:
        return 0

    def evict_device(self, indices) -> None:
        pass

    def evict_host(self, host_value) -> int:
        return len(host_value)

    def reset(self) -> None:
        self.ack_write_queue = []
        self.ack_load_queue = []


class _HiCache(HiMambaRadixCache):
    """HiMambaRadixCache with the host-pool/controller assembly replaced.

    Sets exactly the attributes `HiMambaRadixCache.__init__` would set before
    delegating to `MambaRadixCache.__init__`, so every method under test is
    the production one.
    """

    def __init__(self, params: CacheInitParams):
        self._enable_metrics_flag = False
        self.page_size = params.page_size
        self.hybrid_kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        self.kvcache = self.hybrid_kv_cache.full_kv_pool
        self.tp_group = None
        self.tp_world_size = 1
        self.enable_storage = False
        self.enable_storage_metrics = False
        self.extra_metric_labels = None
        self.full_kv_pool_host = _FakeHostPool(KV_POOL_SIZE)
        self.mamba_pool_host = _FakeHostPool(256)
        self.cache_controller = _FakeController(
            self.full_kv_pool_host,
            self.mamba_pool_host,
            params.req_to_token_pool.mamba_allocator,
            params.token_to_kv_pool_allocator,
        )
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self._load_back_pins = {}
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        self._write_through_pinned = {}
        self._write_through_inflight = {}
        # The pin budget is a separate backpressure mechanism; keep it out of
        # the way so these tests observe the pairing, not the budget.
        self._mamba_pin_budget_cached = 1 << 30
        self._mamba_pin_skipped = 0
        self.prefetch_loaded_tokens_by_reqid = {}
        self.write_through_threshold = 1
        self.load_back_threshold = 10
        self.evictable_full_device_leaves = set()
        self.evictable_full_host_leaves = set()
        self.mamba_host_lru_list = HostLRUList()
        MambaRadixCache.__init__(self, params=params)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _key(token_ids) -> RadixKey:
    return RadixKey(array("q", token_ids))


def _insert(tree, allocator, pool, token_ids) -> TreeNode:
    slot = pool.mamba_allocator.alloc(1)
    assert slot is not None, "test setup: mamba pool exhausted"
    tree.insert(
        InsertParams(
            key=_key(token_ids),
            value=allocator.alloc(len(token_ids)),
            mamba_value=slot,
        )
    )
    return tree.match_prefix(MatchPrefixParams(key=_key(token_ids))).last_device_node


def _locked_nodes(tree) -> List[TreeNode]:
    out, stack = [], [tree.root_node]
    while stack:
        node = stack.pop()
        if node is not tree.root_node and node.mamba_lock_ref > 0:
            out.append(node)
        stack.extend(node.children.values())
    return out


def _build_hi(mamba_size: int = 8):
    pool, allocator = _build_pools(mamba_size)
    tree = _HiCache(_cache_init_params(pool, allocator))
    return tree, allocator, pool


def _stage_mamba_tombstone(tree, allocator, pool, token_ids) -> TreeNode:
    """Cache a checkpoint, back it up to host, then evict the DEVICE state.

    Leaves the node as a mamba tombstone with a host copy -- the exact
    precondition `init_load_back` acts on.
    """
    node = _insert(tree, allocator, pool, token_ids)
    tree.cache_controller.complete()
    tree.writing_check()
    assert node.mamba_backuped, "test setup: no host mamba copy"
    tree.evict(EvictParams(num_tokens=0, mamba_num=1))
    assert node.mamba_evicted, "test setup: device mamba not evicted"
    return node


class TestLockRefPairing(unittest.TestCase):
    """Each of these reproduced a stranded ref (or an assert) before the fix."""

    def test_admission_release_does_not_steal_the_load_back_pin(self):
        """FALSIFIER 1: the acquire skipped the mamba tombstone, so its release
        must leave the load-back's own pin alone."""
        tree, allocator, pool = _build_hi()
        node = _stage_mamba_tombstone(tree, allocator, pool, list(range(2000, 2032)))

        # PrefillAdder._lock_node: acquire, load back inside, release.
        acquired = tree.inc_lock_ref(node)
        tree.init_load_back(InitLoadBackParams(best_match_node=node, host_hit_length=0))
        self.assertEqual(node.mamba_lock_ref, 1, "load-back pin not taken")
        tree.dec_lock_ref(node, acquired.to_dec_params())

        # Before the fix this released the load-back's pin: the state under an
        # in-flight H2D copy became evictable again.
        self.assertEqual(node.mamba_lock_ref, 1)
        self.assertIn(node.id, acquired.skip_lock_node_ids[ComponentType.MAMBA])
        self.assertEqual(tree.mamba_protected_size(), 1)
        self.assertEqual(tree.mamba_evictable_size(), 0)

        # The pin is released by the ack, and only by the ack.
        tree.cache_controller.complete()
        tree.loading_check()
        self.assertEqual(node.mamba_lock_ref, 0)
        self.assertEqual(tree.mamba_protected_size(), 0)

    def test_second_in_flight_load_back_releases_both_pins(self):
        """FALSIFIER 2a: two loads of the same node in flight -> two pins; the
        id-keyed registry must hold both."""
        tree, allocator, pool = _build_hi()
        node = _stage_mamba_tombstone(tree, allocator, pool, list(range(2000, 2032)))

        tree.load_back(node)
        self.assertEqual(node.mamba_lock_ref, 1)
        # Re-enter while the first copy is still in flight.
        tree.load_back(node)
        self.assertEqual(node.mamba_lock_ref, 2, "second pin not taken")

        tree.cache_controller.complete()
        tree.loading_check()

        # Before the fix the registry held one entry for two pins: one ref
        # stranded forever (and the second ack raised KeyError).
        self.assertEqual(node.mamba_lock_ref, 0)
        self.assertEqual(tree.mamba_protected_size(), 0)
        self.assertEqual(tree.ongoing_load_back, {})
        self.assertEqual(tree._load_back_pins, {})

    def test_second_in_flight_backup_releases_both_pins(self):
        """FALSIFIER 2b: same registry defect on the write-through pin set."""
        tree, allocator, pool = _build_hi()
        node = _insert(tree, allocator, pool, list(range(2000, 2032)))
        # The insert already wrote this node through (threshold 1).
        self.assertEqual(node.mamba_lock_ref, 1)
        tree.write_backup(node)
        self.assertEqual(node.mamba_lock_ref, 2, "second pin not taken")

        tree.cache_controller.complete()
        tree.writing_check()

        self.assertEqual(node.mamba_lock_ref, 0)
        self.assertEqual(tree.mamba_protected_size(), 0)
        self.assertEqual(tree.ongoing_write_through, {})
        self.assertEqual(tree._write_through_pinned, {})

    def test_split_between_acquire_and_release_keeps_the_pairing(self):
        """A radix split planted between acquire and release must not move the
        ref away from the handle the release resolves through.

        This is the lead that turned out NOT to be the leak: the deeper child
        keeps both `mamba_value` and `mamba_lock_ref`, and it is the same
        object the acquirer holds.
        """
        tree, allocator, pool = _build_hi()
        prefix = list(range(2000, 2032))
        node = _insert(tree, allocator, pool, prefix)
        tree.cache_controller.complete()
        tree.writing_check()  # drain the insert's write-through pin
        self.assertEqual(node.mamba_lock_ref, 0)

        tree.inc_lock_ref(node)
        self.assertEqual(node.mamba_lock_ref, 1)

        # Plant the split: a sibling key that diverges INSIDE `node`.
        branch = prefix[:16] + list(range(9000, 9016))
        _insert(tree, allocator, pool, branch)
        self.assertIsNot(node.parent, tree.root_node, "test setup: no split happened")

        # The state and its ref stayed on the node that owns them.
        self.assertIsNotNone(node.mamba_value)
        self.assertEqual(node.mamba_lock_ref, 1)
        self.assertIsNone(node.parent.mamba_value)
        self.assertEqual(node.parent.mamba_lock_ref, 0)

        tree.dec_lock_ref(node)
        self.assertEqual(node.mamba_lock_ref, 0)
        self.assertNotIn(node, _locked_nodes(tree))
        # (the branch insert left its own write-through pin behind; the
        # accounting must still match the set of locked nodes exactly)
        self.assertEqual(
            tree.mamba_protected_size(),
            sum(len(n.mamba_value) for n in _locked_nodes(tree)),
        )

    def test_checkpoint_reattached_under_lock_is_not_released_by_that_lock(self):
        """FALSIFIER 3 (non-hierarchical `MambaRadixCache`): an insert can
        re-attach a checkpoint to a tombstone that is already locked. The
        release must not claim that checkpoint's ref -- it asserted instead."""
        pool, allocator = _build_pools(mamba_size=4)
        tree = MambaRadixCache(params=_cache_init_params(pool, allocator))

        prefix = list(range(3000, 3016))
        deeper = prefix + list(range(4000, 4016))
        _insert(tree, allocator, pool, prefix)
        leaf = _insert(tree, allocator, pool, deeper)
        internal = tree.match_prefix(
            MatchPrefixParams(key=_key(prefix))
        ).last_device_node

        # Force the internal node to be the eviction victim -> mamba tombstone.
        tree.inc_lock_ref(leaf)
        tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertTrue(internal.mamba_evicted)

        acquired = tree.inc_lock_ref(internal)
        _insert(tree, allocator, pool, prefix)  # re-attaches a checkpoint
        self.assertIsNotNone(internal.mamba_value)
        self.assertEqual(internal.mamba_lock_ref, 0)

        # Before the fix: AssertionError("dec_lock_ref on node with
        # mamba_lock_ref=0"), i.e. a dead scheduler.
        tree.dec_lock_ref(internal, acquired.to_dec_params())
        self.assertEqual(internal.mamba_lock_ref, 0)
        self.assertIn(internal.id, acquired.skip_lock_node_ids[ComponentType.MAMBA])

    def test_repeated_admissions_return_every_mamba_ref(self):
        """Node-identity invariant over a workload: after every holder has
        released, NO node may still carry a mamba ref.

        The production ramp was ~one stranded ref per admission that had to
        load a checkpoint back; this drives that seam repeatedly.
        """
        tree, allocator, pool = _build_hi(mamba_size=8)
        node = _stage_mamba_tombstone(tree, allocator, pool, list(range(2000, 2032)))

        for _ in range(20):
            acquired = tree.inc_lock_ref(node)
            if node.mamba_evicted and node.mamba_backuped:
                tree.init_load_back(
                    InitLoadBackParams(best_match_node=node, host_hit_length=0)
                )
            tree.dec_lock_ref(node, acquired.to_dec_params())
            tree.cache_controller.complete()
            tree.check_hicache_events()
            # Whatever is cached and unreferenced must stay evictable.
            tree.evict(EvictParams(num_tokens=0, mamba_num=1))

        tree.cache_controller.complete()
        tree.check_hicache_events()
        self.assertEqual(_locked_nodes(tree), [])
        self.assertEqual(tree.mamba_protected_size(), 0)
        self.assertEqual(tree.ongoing_load_back, {})
        self.assertEqual(tree.ongoing_write_through, {})


if __name__ == "__main__":
    unittest.main()
