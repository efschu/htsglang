"""#773: the #581 write-through pin bound, on the lineage that actually runs.

THE GAP. `mamba_pool_floor.py` computes a hard floor and calls itself the
"single source of truth"; the thing that makes that floor a GUARANTEE rather
than an arithmetic exercise is the write-through pin budget -- caching may pin
every slot the running set does not structurally require, "and not one more"
(`hi_mamba_radix_cache.py:_mamba_pin_budget`). Without it a `write_through`
policy pins EVERY inserted checkpoint the moment it is created
(`write_through_threshold == 1`), the eviction walk skips pinned nodes, and the
pinned set ratchets until a REQUIRED allocation has nowhere to go: #581.

That budget is enforced in `HiMambaRadixCache.write_backup` only, and
`registry.py` states outright that "HiMambaRadixCache has no construction
site anywhere" -- every hybrid-SSM boot under `--enable-hierarchical-cache`
is routed to `UnifiedRadixCache` instead. So on the lineage that runs, the
floor is charged at boot and the bound that justifies it is absent:
`UnifiedRadixCache.write_backup` takes `inc_lock_ref(node)` with no budget
check at all.

`inc_lock_ref` on a node carrying a mamba checkpoint is what makes a STATE
SLOT unevictable (`mamba_component.acquire_component_lock`), so this is the
#581 mechanism verbatim, on the live class.

These tests never touch a GPU and never stand up a DMA controller: the pin
sequence they drive is the verbatim tail of `UnifiedRadixCache.write_backup`
(`inc_lock_ref(node).to_dec_params()` then `_track_write_through_node`), which
is the same technique `test_mamba_pool_floor.py::_PinHarness` already uses for
the dead lineage. Everything except the D->H copy is shipped code.
"""

import unittest
from array import array

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_pool_floor import mamba_hard_floor
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20)

MAMBA_POOL_SIZE = 8
MAX_RUNNING_REQUESTS = 2
NUM_LAYERS = 8
FULL_ATTENTION_LAYER_IDS = (3, 7)
KV_SIZE = 512
MAX_CONTEXT_LEN = 256
MAX_NUM_REQS = 16


def _build(
    mamba_pool_size: int = MAMBA_POOL_SIZE,
    max_running_requests: int = MAX_RUNNING_REQUESTS,
):
    """Live UnifiedRadixCache with a mamba component, pinned to CPU.

    Mirrors `test_unified_radix_cache_unittest.build_fixture`'s mamba branch,
    but hard-codes `device="cpu"` instead of resolving `get_device()`, so this
    suite stays hermetic exactly like `test_mamba_pool_floor.py`. The class
    under test is the real one -- `registry.py` routes every hybrid-SSM boot
    under hierarchical cache here.
    """
    device = "cpu"
    server_args = ServerArgs(model_path="dummy", page_size=1)
    # The property would otherwise load the HF config of the dummy model.
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    server_args.max_running_requests = max_running_requests
    set_global_server_args_for_scheduler(server_args)

    mamba_layers = [i for i in range(NUM_LAYERS) if i not in FULL_ATTENTION_LAYER_IDS]
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=256,
            n_groups=1,
            num_heads=2,
            head_dim=16,
            state_size=16,
            conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

    req_to_token_pool = HybridReqToTokenPool(
        size=MAX_NUM_REQS,
        mamba_size=mamba_pool_size,
        mamba_spec_state_size=MAX_NUM_REQS,
        max_context_len=MAX_CONTEXT_LEN,
        device=device,
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=mamba_layers,
        enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=3,
    )
    kv_pool = HybridLinearKVPool(
        size=KV_SIZE,
        dtype=torch.bfloat16,
        page_size=1,
        head_num=2,
        head_dim=64,
        full_attention_layer_ids=list(FULL_ATTENTION_LAYER_IDS),
        device=device,
        enable_memory_saver=False,
        mamba_pool=req_to_token_pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=KV_SIZE,
        dtype=torch.bfloat16,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
    )
    cache_init_params = CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        disable=False,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=False,
        enable_kv_cache_events=False,
    )
    cache = UnifiedRadixCache(params=cache_init_params)
    cache.cache_init_params = cache_init_params
    return cache, allocator, req_to_token_pool, server_args


def _cache_one_finished_req(cache, allocator, req_to_token_pool, tokens, rid):
    """Insert one finished request, leaving a node that carries a checkpoint."""
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    if req_to_token_pool.alloc([req]) is None:
        return None
    req.output_ids = array("q")
    req.kv_committed_len = len(tokens)
    req.kv_allocated_len = len(tokens)
    req.cache_protected_len = 0
    req.swa_uuid_for_lock = None
    req.extra_key = None
    req.mamba_last_track_seqlen = len(tokens)
    kv_indices = allocator.alloc(len(tokens))
    if kv_indices is None:
        return None
    req_to_token_pool.write((req.req_pool_idx, slice(0, len(tokens))), kv_indices)
    req.last_node = cache.root_node
    cache.cache_finished_req(req, is_insert=True)
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens)))
    ).last_device_node


def _checkpoint_nodes(cache, allocator, req_to_token_pool, count):
    """Cache `count` distinct prefixes, each leaving a mamba checkpoint."""
    nodes = []
    for i in range(count):
        tokens = list(range(2000 + 100 * i, 2000 + 100 * i + 8))
        node = _cache_one_finished_req(
            cache, allocator, req_to_token_pool, tokens, rid=f"ckpt-{i}"
        )
        if node is None:
            break
        if _mamba_value(node) is not None:
            nodes.append(node)
    return nodes


def _mamba_value(node):
    # `component_data` is a list indexed by the integer ComponentType.
    if len(node.component_data) <= int(ComponentType.MAMBA):
        return None
    return node.component_data[ComponentType.MAMBA].value


def _pin_like_write_backup(cache, node):
    """The verbatim tail of `UnifiedRadixCache.write_backup` for a write-through.

    This is what the shipped code does for every backed-up node once
    `--hicache-write-policy write_through` sets `write_through_threshold` to 1,
    minus the D->H copy itself.
    """
    lock_params = cache.inc_lock_ref(node).to_dec_params()
    cache._track_write_through_node(node, lock_params)


def _mamba_evictable(cache):
    return cache.component_evictable_size_[ComponentType.MAMBA]


def _required_alloc_succeeds(req_to_token_pool):
    """A REQUIRED allocation: one active state slot for one new running request.

    This is the site the #581 crash died in (`HybridReqToTokenPool.alloc`,
    formerly a bare assert). It returns None when the pool cannot serve.
    """
    req = Req(
        rid="newcomer",
        origin_input_text="",
        origin_input_ids=array("q", [7, 7, 7]),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    return req_to_token_pool.alloc([req]) is not None


class TestTheStarvationIsReachable(CustomTestCase):
    """CAN-FAIL PROOF, direction 1: with pins unbounded, the pool starves.

    This test asserts the DEFECT. It passes before and after the fix, because
    it drives the pin sequence directly rather than through the guard -- its
    job is to prove that the thing the budget prevents is actually reachable
    on this lineage, so the budget is not decoration.
    """

    def test_unbounded_write_through_pins_starve_a_required_alloc(self):
        cache, allocator, req_to_token_pool, _ = _build()
        nodes = _checkpoint_nodes(cache, allocator, req_to_token_pool, MAMBA_POOL_SIZE)
        self.assertGreater(len(nodes), 0, "fixture produced no checkpoints")

        for node in nodes:
            _pin_like_write_backup(cache, node)

        self.assertEqual(
            _mamba_evictable(cache),
            0,
            "every checkpoint is pinned, so the eviction walk can free nothing",
        )
        self.assertFalse(
            _required_alloc_succeeds(req_to_token_pool),
            "THE #581 MECHANISM: an unbounded pinned set owns the whole pool "
            "and a required active-slot allocation has nowhere to go",
        )


class TestTheLivePathHasTheBudget(CustomTestCase):
    """The fix: the live lineage answers the same question the dead one does."""

    def test_budget_is_the_pool_above_the_floor(self):
        cache, _, req_to_token_pool, server_args = _build()
        expected = MAMBA_POOL_SIZE - mamba_hard_floor(server_args, MAX_RUNNING_REQUESTS)
        self.assertEqual(cache._mamba_pin_budget, expected)
        self.assertGreater(expected, 0, "fixture must leave room above the floor")

    def test_pins_taken_through_the_guard_never_starve_the_pool(self):
        cache, allocator, req_to_token_pool, _ = _build()
        nodes = _checkpoint_nodes(cache, allocator, req_to_token_pool, MAMBA_POOL_SIZE)
        self.assertGreater(len(nodes), 0)

        pinned = 0
        for node in nodes:
            if not cache._mamba_write_through_pin_admissible(node):
                continue
            _pin_like_write_backup(cache, node)
            pinned += 1

        self.assertLessEqual(pinned, cache._mamba_pin_budget)
        self.assertGreater(
            _mamba_evictable(cache),
            0,
            "slots above the budget must stay evictable so the pool can serve",
        )
        self.assertTrue(
            _required_alloc_succeeds(req_to_token_pool),
            "with the pinned set bounded, a required allocation always has "
            "somewhere to go, no matter how far behind the ack drain is",
        )

    def test_the_guard_counts_only_checkpoint_carrying_pins(self):
        """A pinned node with no mamba value costs no state slot.

        `inc_lock_ref` bumps `mamba_lock_ref` only on a node that carries a
        value, so counting every write-through would over-charge the budget
        and refuse backups that cost nothing.
        """
        cache, allocator, req_to_token_pool, _ = _build()
        nodes = _checkpoint_nodes(cache, allocator, req_to_token_pool, 2)
        self.assertGreater(len(nodes), 0)
        before = cache._mamba_pins_held()

        # The root carries no checkpoint.
        cache._track_write_through_node(cache.root_node, None)
        self.assertEqual(cache._mamba_pins_held(), before)

    def test_write_back_is_not_charged(self):
        """Only write-THROUGH takes a pin, so only it is budgeted.

        `write_backup(write_back=True)` reaches `_track_write_through_node`
        with `lock_params=None` -- it never took a lock, so refusing it would
        cost a demotion for no slot saved.
        """
        cache, allocator, req_to_token_pool, _ = _build()
        nodes = _checkpoint_nodes(cache, allocator, req_to_token_pool, MAMBA_POOL_SIZE)
        self.assertGreater(len(nodes), 0)
        for node in nodes:
            _pin_like_write_backup(cache, node)
        # Budget is exhausted, yet a write_back demotion is still admissible.
        self.assertTrue(
            cache._mamba_write_through_pin_admissible(nodes[0], write_back=True)
        )


class _ReachedTheTransfer(Exception):
    """Raised when `write_backup` gets PAST the pin guard."""


class _ExplodingController:
    """A cache controller that is only ever allowed to be non-None.

    `write_backup` consults the pin budget before it touches the controller
    for anything else, so any access to `mem_pool_host` proves the guard let
    the call through. This is how the guard gets a can-fail proof without
    standing up a real DMA controller: the assertion is on CONTROL FLOW, not
    on a return value the refusal path happens to share with a failed write.
    """

    enable_storage = False
    write_policy = "write_through"
    ack_write_queue: list = []
    ack_load_queue: list = []

    @property
    def mem_pool_host(self):
        raise _ReachedTheTransfer()

    def write(self, *args, **kwargs):
        raise _ReachedTheTransfer()


class TestWriteBackupActuallyConsultsTheBudget(CustomTestCase):
    """The gate, not just the predicate.

    Without this, disabling the guard inside `write_backup` leaves every other
    test in this file green -- they call the predicate directly. That mutant
    was run and survived, which is exactly why this class exists.
    """

    def _saturated(self):
        cache, allocator, req_to_token_pool, _ = _build()
        nodes = _checkpoint_nodes(cache, allocator, req_to_token_pool, MAMBA_POOL_SIZE)
        self.assertGreater(len(nodes), 0)
        # Spend the whole budget on real pins.
        for node in nodes:
            if not cache._mamba_write_through_pin_admissible(node):
                break
            _pin_like_write_backup(cache, node)
        self.assertGreaterEqual(cache._mamba_pins_held(), cache._mamba_pin_budget)
        cache.cache_controller = _ExplodingController()
        return cache, nodes

    def test_a_saturated_budget_refuses_before_touching_the_controller(self):
        cache, nodes = self._saturated()
        victim = next(n for n in nodes if _mamba_value(n) is not None)
        self.assertIs(victim.parent, cache.root_node, "victim must not recurse")
        before = cache._mamba_pin_skipped

        self.assertEqual(
            cache.write_backup(victim),
            0,
            "a backup over budget must be declined, not attempted",
        )
        self.assertEqual(
            cache._mamba_pin_skipped, before + 1, "the refusal must be counted"
        )

    def test_the_same_call_proceeds_when_the_budget_allows(self):
        """The other direction: the guard is not refusing unconditionally."""
        cache, allocator, req_to_token_pool, _ = _build()
        nodes = _checkpoint_nodes(cache, allocator, req_to_token_pool, 1)
        self.assertGreater(len(nodes), 0)
        cache.cache_controller = _ExplodingController()

        self.assertGreater(cache._mamba_pin_budget, 0)
        with self.assertRaises(_ReachedTheTransfer):
            cache.write_backup(nodes[0])


class TestTheLivePoolCanEvictBeforeFailing(CustomTestCase):
    """The second live-lineage gap: `bind_tree_cache` was never called here.

    `HybridReqToTokenPool._alloc_mamba_slots_or_evict` degrades a REQUIRED
    allocation by evicting cached checkpoints and retrying -- but only
    `if self.tree_cache is not None`. `MambaRadixCache.__init__` binds itself
    to the pool; `UnifiedRadixCache.__init__` did not. On the lineage that
    runs, the pool therefore reported exhaustion while cached, EVICTABLE
    checkpoints sat in the tree, and #639b's rank-parity tombstone branch --
    which hangs off the same handle -- never ran either.
    """

    def test_the_pool_is_bound_to_this_tree(self):
        cache, _, req_to_token_pool, _ = _build()
        self.assertIs(
            req_to_token_pool.tree_cache,
            cache,
            "without the back-handle the pool cannot evict before failing",
        )

    def test_a_required_alloc_evicts_an_unpinned_checkpoint(self):
        """Fill the pool with checkpoints that NOTHING pins, then demand a slot.

        Every slot is taken but every slot is evictable, so the correct
        outcome is an eviction and a successful allocation. Unbound, the same
        pool returns None and admission defers a request it could have served.
        """
        cache, allocator, req_to_token_pool, _ = _build()
        nodes = _checkpoint_nodes(cache, allocator, req_to_token_pool, MAMBA_POOL_SIZE)
        self.assertGreater(len(nodes), 0)
        self.assertGreater(
            _mamba_evictable(cache), 0, "fixture must leave evictable state"
        )

        self.assertTrue(
            _required_alloc_succeeds(req_to_token_pool),
            "a pool full of UNPINNED checkpoints must evict and serve",
        )


class TestTheFloorTermIsLoadBearing(CustomTestCase):
    """CAN-FAIL PROOF, direction 2: a floor that is too LOW gives #581 back.

    The briefing's own constraint. If the budget were derived from a floor
    that under-counts what a running request holds, the bound would admit
    exactly as many pins as it takes to starve the pool again -- so the
    guard's safety is a property of the FLOOR, not of the guard.
    """

    def test_a_zero_floor_makes_the_starvation_reachable_again(self):
        cache, allocator, req_to_token_pool, _ = _build()
        nodes = _checkpoint_nodes(cache, allocator, req_to_token_pool, MAMBA_POOL_SIZE)
        self.assertGreater(len(nodes), 0)

        # A floor of 0 -> the budget is the WHOLE pool.
        cache._mamba_pin_budget_cached = MAMBA_POOL_SIZE
        for node in nodes:
            if cache._mamba_write_through_pin_admissible(node):
                _pin_like_write_backup(cache, node)

        self.assertEqual(_mamba_evictable(cache), 0)
        self.assertFalse(
            _required_alloc_succeeds(req_to_token_pool),
            "with the floor removed the guard admits the starving pin set, "
            "which is the proof that the floor is what makes it safe",
        )

    def test_the_real_floor_refuses_where_the_zero_floor_admitted(self):
        cache, allocator, req_to_token_pool, server_args = _build()
        nodes = _checkpoint_nodes(cache, allocator, req_to_token_pool, MAMBA_POOL_SIZE)
        self.assertGreater(len(nodes), 0)

        real_budget = cache._mamba_pin_budget
        self.assertLess(
            real_budget,
            MAMBA_POOL_SIZE,
            "the floor must reserve something, or the guard is a no-op",
        )
        self.assertEqual(
            real_budget,
            MAMBA_POOL_SIZE - mamba_hard_floor(server_args, MAX_RUNNING_REQUESTS),
        )


if __name__ == "__main__":
    unittest.main()
