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

`TestAdmissionLockPrecondition` pins the emergent invariant that keeps the
SIBLING site safe: `PrefillAdder._req_inc_lock_ref` (schedule_policy.py) drops
the acquire's `skip_lock_node_ids`, and the paired release
(`cache_finished_req` / `cache_unfinished_req` -> `dec_lock_ref(req.last_node)`)
passes no params, so that lock would steal a ref by the same mechanism as (1)
IF `req.last_node` could be a mamba tombstone there. It cannot -- but only
because three separate facts line up, so they are asserted mechanically.

Every pool here is built on `torch.device("cpu")`; the hierarchical cache runs
against a fake cache controller, so no GPU and no host pool assembly.
"""

import threading
import unittest
from array import array
from collections import Counter
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


class TestAdmissionLockPrecondition(unittest.TestCase):
    """The REQUEST lock (`_req_inc_lock_ref` -> `dec_lock_ref(req.last_node)`)
    carries no skip set in either direction, so it is only safe while
    `req.last_node` cannot be a mamba tombstone when it is taken.

    `match_prefix` DOES hand out mamba tombstones as `last_device_node`
    (`_match_prefix_helper` selects on `mamba_value is not None OR
    mamba_backuped`), and `init_next_round_input` assigns that node straight to
    `req.last_node`. What saves the request lock is that admission reaches
    `_req_inc_lock_ref` only after `init_load_back` has run, and a tombstone
    always forces `init_load_back` to run:

      mamba_backuped => backuped        a mamba host copy is only ever written
                                        where the KV host copy is written too
                                        (`mamba_backup_commit`,
                                        `_insert_helper_host`)
      => `last_host_node` stops AT the tombstone rather than walking past it
      => `mamba_host_hit_length == 1`
      => `Req.needs_host_load_back()` is True
      => `init_load_back` runs and resolves the tombstone.

    Each link is asserted below. If a future path ever creates a mamba host
    copy without a KV host copy, the second link breaks, `mamba_host_hit_length`
    goes to 0, the tombstone reaches `_req_inc_lock_ref`, and the request lock
    starts stealing refs -- so this test failing means the skip set has to be
    threaded through `Req` (like `swa_uuid_for_lock`) to the release sites.
    """

    def _staged_internal_tombstone(self):
        """A tombstone that KEEPS its device KV -- the shape where the KV side
        gives no load-back signal at all, so `mamba_host_hit_length` is the
        only thing that forces `init_load_back`."""
        tree, allocator, pool = _build_hi(mamba_size=8)
        prefix = list(range(2000, 2032))
        deeper = prefix + list(range(3000, 3032))
        internal = _insert(tree, allocator, pool, prefix)
        leaf = _insert(tree, allocator, pool, deeper)
        tree.cache_controller.complete()
        tree.writing_check()

        # Lock the leaf so the internal node is the mamba eviction victim; an
        # internal node is tombstoned in place and keeps its device KV.
        tree.inc_lock_ref(leaf)
        tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        tree.dec_lock_ref(leaf)
        self.assertTrue(internal.mamba_evicted, "test setup: not a tombstone")
        self.assertFalse(internal.evicted, "test setup: KV left the device")
        return tree, prefix, internal

    def test_mamba_host_copy_implies_kv_host_copy(self):
        """Link 2: the reason `last_host_node` cannot walk past a tombstone."""
        tree, _, internal = self._staged_internal_tombstone()
        self.assertTrue(internal.mamba_backuped)
        self.assertTrue(
            internal.backuped,
            "a mamba host copy without a KV host copy breaks the admission "
            "precondition -- see this class's docstring",
        )

    def test_tombstone_last_device_node_forces_a_load_back(self):
        """Links 1+3: the node admission would lock IS the tombstone, and the
        match still reports a host hit, so `init_load_back` cannot be skipped."""
        tree, prefix, internal = self._staged_internal_tombstone()

        match = tree.match_prefix(MatchPrefixParams(key=_key(prefix)))
        self.assertIs(match.last_device_node, internal)
        self.assertIsNone(match.last_device_node.mamba_value)
        # The KV side is silent here: only the mamba term forces the load back.
        self.assertEqual(match.host_hit_length, 0)
        self.assertEqual(match.mamba_host_hit_length, 1)

    def test_load_back_resolves_the_tombstone_before_the_request_lock(self):
        """Link 4: what `_req_inc_lock_ref` finally locks always has a value."""
        tree, prefix, internal = self._staged_internal_tombstone()
        match = tree.match_prefix(MatchPrefixParams(key=_key(prefix)))

        _, last_node = tree.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
            )
        )
        self.assertIsNotNone(
            last_node.mamba_value,
            "the request lock would be taken on a tombstone and its paramless "
            "release would steal a ref",
        )


class _MinAllReduce:
    """MIN all_reduce over N threads, one per simulated TP rank."""

    def __init__(self, n: int):
        self.n = n
        self.barrier = threading.Barrier(n, timeout=30)
        self.values = [None] * n
        self.local = threading.local()

    def __call__(self, tensor, op=None, group=None):
        self.values[self.local.rank] = int(tensor.item())
        self.barrier.wait()
        reduced = min(v for v in self.values if v is not None)
        self.barrier.wait()
        tensor.fill_(reduced)


class TestAckDrainAcrossRanks(unittest.TestCase):
    """FALSIFIERS for the production-surviving half of #581.

    The transfer queues are RANK-LOCAL. `write_backup` backs a node up on one
    rank and skips it on another -- host pool full, pin budget reached, parent
    not backed up yet -- and under uneven TP/DCP the ranks' host pools differ
    by construction (`scheduler.py`: "RANK-LOCAL: `backuped` means full KV
    present in THIS rank's host pool ... it can be true here and false on a
    peer for the same node"). So the queues differ in LENGTH.

    `writing_check` min-reduced each rank's "how many of MY acks are ready"
    across the group. A rank with an EMPTY queue contributed 0, so no rank
    drained anything, so every write-through pin -- each of which makes a
    mamba checkpoint unevictable -- was held forever. One cached checkpoint
    per finished request, `mamba_evictable` at 0, and only the backing-up rank
    hitting the wall: the signature that survived the first fix.
    """

    RANKS = 2
    TURNS = 12

    def _run_ranks(self, rank0_backs_up=True, rank1_backs_up=False, prime=None):
        allreduce = _MinAllReduce(self.RANKS)
        trees = []
        for rank in range(self.RANKS):
            tree, allocator, pool = _build_hi(mamba_size=32)
            tree.tp_world_size = self.RANKS
            tree.tp_group = None
            backs_up = rank0_backs_up if rank == 0 else rank1_backs_up
            if not backs_up:
                # Rank-local host pressure: write_backup returns 0 here, so
                # this rank's registry and ack queue both stay empty.
                tree.cache_controller.write = lambda *a, **k: None
            trees.append((tree, allocator, pool))

        results, errors = {}, {}

        def run(rank):
            allreduce.local.rank = rank
            tree, allocator, pool = trees[rank]
            try:
                if prime is not None:
                    prime(rank, tree, allocator, pool)
                for turn in range(self.TURNS):
                    _insert(
                        tree,
                        allocator,
                        pool,
                        list(range(1000, 1032))
                        + list(range(2000 + 100 * turn, 2032 + 100 * turn)),
                    )
                    tree.cache_controller.complete()
                    tree.writing_check()
                results[rank] = tree
            except BaseException as exc:  # surface, never hang the barrier
                errors[rank] = exc
                self.addCleanup(lambda: None)
                raise

        threads = [threading.Thread(target=run, args=(r,)) for r in range(self.RANKS)]
        original = torch.distributed.all_reduce
        torch.distributed.all_reduce = allreduce
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)
        finally:
            torch.distributed.all_reduce = original
        self.assertEqual(errors, {}, f"rank thread raised: {errors}")
        self.assertFalse(any(t.is_alive() for t in threads), "rank thread hung")
        return results

    def test_a_rank_with_no_backups_does_not_stall_the_drain(self):
        """FALSIFIER: rank 1 writes nothing; rank 0 must still drain its acks.

        Before the fix rank 0 ended with protected == TURNS and evictable == 0
        -- the production ramp, one pinned checkpoint per finished request.
        """
        results = self._run_ranks()
        backing_up = results[0]

        self.assertEqual(
            backing_up.mamba_protected_size(),
            0,
            "write-through pins were never released",
        )
        self.assertEqual(backing_up.mamba_evictable_size(), self.TURNS)
        self.assertEqual(backing_up.ongoing_write_through, {})
        self.assertEqual(backing_up._write_through_pinned, {})
        self.assertEqual(backing_up.cache_controller.ack_write_queue, [])

    def test_pins_are_bounded_by_in_flight_copies_not_by_request_count(self):
        """The invariant the ramp violated: what is pinned at any moment is
        what is still being copied, never a function of how many requests have
        been served."""
        results = self._run_ranks()
        tree = results[0]
        self.assertEqual(len(tree._write_through_pinned), 0)
        self.assertLessEqual(
            tree.mamba_protected_size(), len(tree.ongoing_write_through)
        )

    def test_a_rank_with_no_loads_does_not_stall_the_load_back_drain(self):
        """FALSIFIER, and the mechanism the production log points at.

        `loading_check` min-reduces the same way, and load-back pins are
        subject to NO budget (unlike write-through pins, which `write_backup`
        caps at pool - hard_floor). In the boot that survived the first fix the
        pin-budget warning never fired even though protected reached 93 of 96,
        i.e. the write-through pins stayed under their cap of 76 -- so the ramp
        was carried by the unbudgeted load-back pins, one per turn, each
        holding a mamba checkpoint unevictable forever.
        """
        allreduce = _MinAllReduce(self.RANKS)
        trees, results, errors = [], {}, {}
        for _ in range(self.RANKS):
            trees.append(_build_hi(mamba_size=16))

        def run(rank):
            allreduce.local.rank = rank
            tree, allocator, pool = trees[rank]
            try:
                if rank == 0:
                    # Only this rank holds a host copy to load back -- exactly
                    # the RANK-LOCAL `backuped` divergence uneven TP produces.
                    # Staged while still single-rank: its internal drain must
                    # not inject a collective the other rank never enters.
                    node = _stage_mamba_tombstone(
                        tree, allocator, pool, list(range(2000, 2032))
                    )
                tree.tp_world_size = self.RANKS
                for _ in range(self.TURNS):
                    if rank == 0 and node.mamba_evicted:
                        tree.load_back(node)
                    tree.cache_controller.complete()
                    tree.loading_check()  # collective: every rank enters
                results[rank] = tree
            except BaseException as exc:
                errors[rank] = exc
                raise

        threads = [threading.Thread(target=run, args=(r,)) for r in range(self.RANKS)]
        original = torch.distributed.all_reduce
        torch.distributed.all_reduce = allreduce
        for tree, _, _ in trees:
            tree.tp_group = None
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)
        finally:
            torch.distributed.all_reduce = original
        self.assertEqual(errors, {}, f"rank thread raised: {errors}")

        loading = results[0]
        self.assertEqual(loading.ongoing_load_back, {}, "load-back pins never drained")
        self.assertEqual(loading._load_back_pins, {})
        self.assertEqual(loading.mamba_protected_size(), 0)

    def test_ranks_that_both_back_up_stay_in_lockstep(self):
        """The fix must not turn the drain into a free-for-all: when every rank
        HAS a queue, the reduction still throttles all of them to the slowest
        live transfer."""
        results = self._run_ranks(rank0_backs_up=True, rank1_backs_up=True)
        for rank, tree in results.items():
            self.assertEqual(tree.mamba_protected_size(), 0, f"rank{rank}")
            self.assertEqual(tree.mamba_evictable_size(), self.TURNS, f"rank{rank}")

    def test_stale_ack_without_registry_entry_does_not_stall_the_drain(self):
        """FALSIFIER: the removed `len(ongoing_write_through) > 0` gate.

        A node that leaves the tree mid-copy is popped from the registry by
        `_forget_write_through`, leaving an ack queued with no registry entry.
        The gate then reported "0 ready" forever and froze the drain on EVERY
        rank through the MIN.
        """

        def prime(rank, tree, allocator, pool):
            if rank != 0:
                return
            node = _insert(tree, allocator, pool, list(range(5000, 5032)))
            tree.cache_controller.complete()
            # The copy is queued and acked, but the node left the tree first.
            tree._forget_write_through(node)
            self.assertEqual(tree.ongoing_write_through, {})
            self.assertNotEqual(tree.cache_controller.ack_write_queue, [])

        results = self._run_ranks(prime=prime)
        tree = results[0]
        self.assertEqual(tree.cache_controller.ack_write_queue, [])
        self.assertEqual(tree.mamba_protected_size(), 0)

    def test_single_rank_drain_needs_no_collective(self):
        """tp_world_size == 1 keeps the default path: drain what is ready, no
        all_reduce at all."""
        tree, allocator, pool = _build_hi(mamba_size=32)
        self.assertEqual(tree.tp_world_size, 1)

        def explode(*a, **k):
            raise AssertionError("single-rank drain must not collective-reduce")

        original = torch.distributed.all_reduce
        torch.distributed.all_reduce = explode
        try:
            for turn in range(6):
                _insert(
                    tree,
                    allocator,
                    pool,
                    list(range(3000 + 100 * turn, 3032 + 100 * turn)),
                )
                tree.cache_controller.complete()
                tree.writing_check()
        finally:
            torch.distributed.all_reduce = original

        self.assertEqual(tree.mamba_protected_size(), 0)
        self.assertEqual(tree.mamba_evictable_size(), 6)


class TestMultiTurnRetireReturnsEveryCheckpoint(unittest.TestCase):
    """The production shape -- one conversation, many short prefills over a
    growing cached prefix -- on a SINGLE rank.

    This is a NEGATIVE result, not a falsifier: it passes with and without the
    drain fix, which is the point. The ramp needs TP > 1 to reproduce, because
    a single rank never min-reduces its ack count against anyone. Anybody
    trying to reproduce #581 on one rank will see nothing; see
    `TestAckDrainAcrossRanks` for the shape that fails.
    """

    def test_protected_does_not_ratchet_with_turn_count(self):
        tree, allocator, pool = _build_hi(mamba_size=64)
        prefix = list(range(1000, 1064))
        protected_seen = []

        for turn in range(40):
            # Each turn caches one more checkpoint on the shared prefix.
            _insert(
                tree,
                allocator,
                pool,
                prefix + list(range(4000 + 50 * turn, 4032 + 50 * turn)),
            )
            tree.cache_controller.complete()
            tree.check_hicache_events()
            protected_seen.append(tree.mamba_protected_size())

        self.assertEqual(
            max(protected_seen),
            0,
            f"protected ratcheted across turns: {protected_seen}",
        )
        self.assertEqual(_locked_nodes(tree), [])
        self.assertGreater(tree.mamba_evictable_size(), 0)


class TestPinTrace(unittest.TestCase):
    """SGLANG_MAMBA_PIN_TRACE: the field diagnostic for #581.

    Production gives `mamba_protected` only in the dying breath. This emits
    the pin ledger every N scheduler ticks so a ramp can be attributed to a
    queue depth and a call site while it is still climbing.
    """

    def test_trace_line_renders_with_the_pin_ledger(self):
        with envs.SGLANG_MAMBA_PIN_TRACE.override(1):
            tree, allocator, pool = _build_hi(mamba_size=16)
            self.assertEqual(tree._pin_trace_every, 1)
            node = _insert(tree, allocator, pool, list(range(2000, 2032)))
            tree.inc_lock_ref(node)
            with self.assertLogs(
                "sglang.srt.mem_cache.hi_mamba_radix_cache", level="INFO"
            ) as captured:
                tree.check_hicache_events()

        line = next(m for m in captured.output if "MAMBA-PIN-TRACE" in m)
        for field in (
            "tick=",
            "ack_write=",
            "ack_load=",
            "wt_pins=",
            "wt_inflight=",
            "lb_pins=",
            "ongoing_wt=",
            "ongoing_lb=",
            "protected=",
            "evictable=",
            "mamba_avail=",
            "ops[",
        ):
            self.assertIn(field, line)
        # `protected` counts SLOTS, not refs: this node carries two refs (the
        # insert's write-through pin and the explicit lock) but one slot.
        self.assertIn("protected=1", line)
        self.assertIn("wt_pins=1", line)
        self.assertEqual(node.mamba_lock_ref, 2)
        # inc/dec traffic is attributed to the calling function.
        self.assertIn("inc_mamba@write_backup=1", line)
        self.assertIn("inc_mamba@test_trace_line_renders_with_the_pin_ledger=1", line)

    def test_counters_reset_between_lines(self):
        with envs.SGLANG_MAMBA_PIN_TRACE.override(1):
            tree, allocator, pool = _build_hi(mamba_size=16)
            _insert(tree, allocator, pool, list(range(2000, 2032)))
            with self.assertLogs(
                "sglang.srt.mem_cache.hi_mamba_radix_cache", level="INFO"
            ) as first:
                tree.check_hicache_events()
            with self.assertLogs(
                "sglang.srt.mem_cache.hi_mamba_radix_cache", level="INFO"
            ) as second:
                tree.check_hicache_events()

        self.assertIn("inc_mamba@write_backup=1", first.output[0])
        line = next(m for m in second.output if "MAMBA-PIN-TRACE" in m)
        self.assertIn("ops[]", line)

    def test_interval_throttles_the_line(self):
        with envs.SGLANG_MAMBA_PIN_TRACE.override(3):
            tree, allocator, pool = _build_hi(mamba_size=16)
            with self.assertLogs(
                "sglang.srt.mem_cache.hi_mamba_radix_cache", level="INFO"
            ) as captured:
                for _ in range(6):
                    tree.check_hicache_events()
        self.assertEqual(sum(1 for m in captured.output if "MAMBA-PIN-TRACE" in m), 2)

    def test_default_is_off(self):
        """Unset env: the traced path is never entered."""
        tree, _, _ = _build_hi(mamba_size=16)
        self.assertEqual(tree._pin_trace_every, 0)
        tree.check_hicache_events()
        self.assertEqual(tree._pin_trace_ops, Counter())


if __name__ == "__main__":
    unittest.main()
