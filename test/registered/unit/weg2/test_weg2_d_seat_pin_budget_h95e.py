"""H95e: the write-through pin budget follows the D phase's reachable pool.

H95c limits the mamba allocator of a D phase of n < --d-bs seats to slots
1..L(n) (NF form, 38 slots at --d-bs 6: 7/13/19/25/32/38) and admits at most
n running requests. The #581 pin budget -- "floor (running set) + budget =
pool", the construction that makes a REQUIRED allocation independent of the
ack drain -- was computed once from the BOOT pool and the boot's
--max-running-requests (38 - 6 x 5 = 8, boot log ``MAMBA-FLOOR pool=38
floor=30 retention_budget=8``) and cached. In an n=1 phase the budget (8)
exceeds every reachable slot (7): write-through pins may hold all of them.

Two holes, one guarantee:
(a) the budget: per phase it is L(n) - n x 5 = 2/3/4/5/7/8;
(b) ``write_backup`` asked the budget ABOVE its parent recursion, so a forced
    insert over a chain of k unbacked checkpoints (D's flip park, the finish
    of a request resumed from it) took k pins after k checks at one count.

Plus the reading: ``mamba usage`` counted the withheld slots as occupied
(n=1 idle: 31/38 = 0.82).

WHAT MUST HOLD.
(1) n=1, pins taken up to the budget: a running request still gets its
    slots (lock + active + ping-pong pair + donation). Red on 192ae05816.
(2) The budget per phase: 2/3/4/5/7/8, recomputed at every wake.
(3) n=6: unchanged (8), six running requests get their slots.
(4) A pin inside the parent recursion is charged before this node's pin.
(5) mamba usage at n=1 idle is 0, not 0.82; n=6 unchanged.

Real allocator, real ``UnifiedRadixCache``, real pool on CPU; the pin is the
verbatim tail of ``write_backup`` (the #773 test's technique).
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
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_pool_floor import (
    mamba_hard_floor,
    mamba_slots_per_running_req,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20)

MAMBA_SLOTS = 38
CAP = 6
NUM_LAYERS = 24
FULL_LAYER_IDS = (3, 7, 11, 15, 19, 23)
NON_FULL_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in set(FULL_LAYER_IDS)]


class _Fx:
    def __init__(self, cache, allocator, pool, server_args):
        self.cache, self.allocator, self.pool = cache, allocator, pool
        self.server_args = server_args
        self.seq = 0

    @property
    def mamba(self):
        return self.pool.mamba_allocator


def _build() -> _Fx:
    """The NF D floor (extra_buffer + overlap: 5 slots per running request,
    --max-running-requests 6 -> floor 30, budget 8 on 38 slots) over the
    H95d CPU fixture."""
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    server_args.max_running_requests = CAP
    server_args.mamba_radix_cache_strategy = "extra_buffer"
    server_args.disable_overlap_schedule = False
    set_global_server_args_for_scheduler(server_args)
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=256, n_groups=1, num_heads=2,
            head_dim=16, state_size=16, conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=NON_FULL_LAYER_IDS)
    pool = HybridReqToTokenPool(
        size=16, mamba_size=MAMBA_SLOTS, mamba_spec_state_size=16,
        max_context_len=512, device="cpu", enable_memory_saver=False,
        cache_params=cache_params, mamba_layer_ids=NON_FULL_LAYER_IDS,
        enable_mamba_extra_buffer=False, speculative_num_draft_tokens=3,
    )
    kv_pool = HybridLinearKVPool(
        size=1024, dtype=torch.bfloat16, page_size=1, head_num=2, head_dim=64,
        full_attention_layer_ids=list(FULL_LAYER_IDS), device="cpu",
        enable_memory_saver=False, mamba_pool=pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=1024, dtype=torch.bfloat16, device="cpu", kvcache=kv_pool,
        need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=pool, token_to_kv_pool_allocator=allocator,
        page_size=1, disable=False, sliding_window_size=None,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=False, enable_kv_cache_events=False,
        eviction_policy="lru", is_eagle=False,
    )
    cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    return _Fx(cache, allocator, pool, server_args)


def _set_limit(mamba, limit, n) -> bool:
    """``d_seat_vram.on_wake``'s call. The seat count is new in H95e; on the
    base the allocator takes the limit alone, so the base runs its own
    behaviour rather than dying on the signature."""
    try:
        return mamba.set_phase_limit(limit, seats=n)
    except TypeError:
        return mamba.set_phase_limit(limit)


def _wake(fx: _Fx, n: int) -> int:
    """The D wake: slot limit, then the flush (empty tree, fresh pools)."""
    from sglang.srt.weg2.d_seat_vram import phase_slot_limit

    fx.cache.reset()
    want = None if n >= CAP else phase_slot_limit(MAMBA_SLOTS, n, CAP)
    assert _set_limit(fx.mamba, want, n), (n, want)
    fx.pool.clear()
    return MAMBA_SLOTS if want is None else want


def _mamba_value(node):
    if len(node.component_data) <= int(ComponentType.MAMBA):
        return None
    return node.component_data[ComponentType.MAMBA].value


def _serve(fx: _Fx, prefix=None):
    """One finished request; its anchor stays in the tree, evictable.
    Returns the anchor node."""
    fx.seq += 1
    i = fx.seq
    tokens = list(prefix or []) + list(range(1000 * i, 1000 * i + 8))
    req = Req(
        rid=f"h95e-{i}", origin_input_text="",
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    assert fx.pool.alloc([req]) is not None
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", tokens)
    req.set_extend_range(0, len(tokens))
    kv = fx.allocator.alloc(len(tokens))
    assert kv is not None
    fx.pool.write((req.req_pool_idx, slice(0, len(tokens))), kv)
    req.kv_committed_len = len(tokens)
    req.kv_allocated_len = len(tokens)
    req.last_node = fx.cache.root_node
    req.cache_protected_len = 0
    req.swa_uuid_for_lock = None
    req.extra_key = None
    req.mamba_last_track_seqlen = len(tokens)
    fx.cache.cache_finished_req(req, is_insert=True)
    fx.pool.free(req)
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    return fx.cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens)))
    ).last_device_node, tokens


def _anchors(fx: _Fx):
    out = []
    stack = [fx.cache.root_node]
    while stack:
        node = stack.pop()
        stack.extend(node.children.values())
        if node is not fx.cache.root_node and _mamba_value(node) is not None:
            out.append(node)
    return out


def _pin_like_write_backup(cache, node):
    """The verbatim tail of ``UnifiedRadixCache.write_backup`` (write-through
    pin), minus the D->H copy -- the #773 test's harness."""
    lock_params = cache.inc_lock_ref(node).to_dec_params()
    cache._track_write_through_node(node, lock_params)


def _pin_up_to_budget(fx: _Fx) -> int:
    pinned = 0
    for node in _anchors(fx):
        if not fx.cache._mamba_write_through_pin_admissible(node):
            continue
        _pin_like_write_backup(fx.cache, node)
        pinned += 1
    return pinned


def _run_one(fx: _Fx, tag: str) -> bool:
    """One running request's REQUIRED slots, through the shipped sites:
    the checkpoint it resumes from stays locked (``inc_lock_ref``), the
    active slot (``HybridReqToTokenPool.alloc``), the ping-pong pair and
    the donation's replacement (``_alloc_mamba_slots_or_evict``, the
    evict-then-alloc path every one of them takes). 5 slots = the floor's
    per-request term."""
    free_anchor = next(
        (n for n in _anchors(fx)
         if n.id not in fx.cache.ongoing_write_through
         and n.component_data[ComponentType.MAMBA].lock_ref == 0),
        None,
    )
    if free_anchor is not None:
        fx.cache.inc_lock_ref(free_anchor)
    req = Req(
        rid=f"run-{tag}", origin_input_text="",
        origin_input_ids=array("q", [7, 7, 7]),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    if fx.pool.alloc([req]) is None:
        return False
    if fx.pool._alloc_mamba_slots_or_evict(2) is None:
        return False
    if fx.pool._alloc_mamba_slots_or_evict(1) is None:
        return False
    return True


class TheRunningRequestGetsItsSlotsAtN1(CustomTestCase):
    def test_n1_pins_up_to_budget_leave_the_running_request_its_floor(self):
        """(1) n=1: 7 reachable slots, every one a cached anchor. Pins taken
        through the guard up to the budget; then one running request needs
        lock + active + 2 ping-pong + donation. Base: budget 8 pins all 7,
        eviction frees nothing, the active slot has nowhere to go."""
        fx = _build()
        self.assertEqual(mamba_slots_per_running_req(fx.server_args), 5)
        self.assertEqual(mamba_hard_floor(fx.server_args, CAP), 30)
        limit = _wake(fx, 1)
        self.assertEqual(limit, 7)
        for _ in range(9):
            _serve(fx)
        self.assertEqual(len(_anchors(fx)), 7, "every reachable slot holds an anchor")
        pinned = _pin_up_to_budget(fx)
        self.assertTrue(
            _run_one(fx, "n1"),
            "the running request was starved of a slot (%d of %d reachable slots pinned)"
            % (pinned, limit),
        )
        self.assertLessEqual(pinned, limit - 5, "pins must leave one request's floor")

    def test_n1_budget_is_the_reachable_pool_above_one_seat(self):
        fx = _build()
        _wake(fx, 1)
        self.assertEqual(fx.cache._mamba_pin_budget, 7 - 5)


class TheBudgetFollowsEveryWake(CustomTestCase):
    def test_budget_per_phase(self):
        """(2) 7-5, 13-10, 19-15, 25-20, 32-25, 38-30 -- and back, over wakes
        on ONE cache (the cached value must not survive a wake)."""
        fx = _build()
        want = {1: 2, 2: 3, 3: 4, 4: 5, 5: 7, 6: 8}
        for n in (6, 1, 2, 3, 4, 5, 6, 3, 1):
            _wake(fx, n)
            self.assertEqual(fx.cache._mamba_pin_budget, want[n], n)

    def test_a_refused_limit_keeps_the_cap_form_budget(self):
        """A wake whose limit is refused (a slot above it still live) keeps
        the cap form, and with it the boot budget."""
        fx = _build()
        _wake(fx, 6)
        held = [fx.mamba.alloc(1) for _ in range(20)]
        self.assertTrue(all(h is not None for h in held))
        self.assertFalse(_set_limit(fx.mamba, 7, 1))
        self.assertIsNone(fx.mamba.phase_limit)
        self.assertEqual(fx.cache._mamba_pin_budget, 8)


class N6IsUnchanged(CustomTestCase):
    def test_n6_budget_and_six_running_requests(self):
        """(3) The cap form: budget 8 (boot line ``retention_budget=8``),
        pins up to it, six running requests still get 6 x 5 = 30."""
        fx = _build()
        self.assertEqual(_wake(fx, 6), MAMBA_SLOTS)
        self.assertIsNone(fx.mamba.phase_limit)
        self.assertEqual(fx.cache._mamba_pin_budget, 8)
        for _ in range(40):
            _serve(fx)
        self.assertEqual(len(_anchors(fx)), MAMBA_SLOTS)
        self.assertEqual(_pin_up_to_budget(fx), 8)
        for k in range(CAP):
            self.assertTrue(_run_one(fx, f"n6-{k}"), k)


class _ReachedTheTransfer(Exception):
    pass


class _ExplodingController:
    enable_storage = False
    write_policy = "write_back"
    ack_write_queue: list = []
    ack_load_queue: list = []

    @property
    def mem_pool_host(self):
        raise _ReachedTheTransfer()

    def write(self, *args, **kwargs):
        raise _ReachedTheTransfer()


class TheRecursionIsCharged(CustomTestCase):
    def test_a_parent_pin_taken_in_the_recursion_counts_for_this_node(self):
        """(4) Budget - 1 pins held; ``write_backup(child)`` passes its check,
        the parent recursion takes the last pin, and the child must NOT pin
        on top (base: it goes on to the transfer -- one pin over)."""
        fx = _build()
        _wake(fx, 6)  # the cap form: budget 8 on base and fix alike
        parent, ptoks = _serve(fx)
        child, _ = _serve(fx, prefix=ptoks)
        self.assertIs(child.parent, parent)
        self.assertIsNotNone(_mamba_value(parent))
        self.assertIsNotNone(_mamba_value(child))
        budget = fx.cache._mamba_pin_budget
        self.assertEqual(budget, 8)
        for _ in range(budget - 1):
            other, _ = _serve(fx)
            _pin_like_write_backup(fx.cache, other)
        self.assertEqual(fx.cache._mamba_pins_held(), budget - 1)
        self.assertTrue(fx.cache._mamba_write_through_pin_admissible(child))

        orig = UnifiedRadixCache.write_backup

        def patched(self_, node, write_back=False):
            if node is parent:
                _pin_like_write_backup(self_, parent)
                return 1
            return orig(self_, node, write_back)

        fx.cache.write_backup = types.MethodType(patched, fx.cache)
        fx.cache.cache_controller = _ExplodingController()
        try:
            got = fx.cache.write_backup(child)
        except _ReachedTheTransfer:
            self.fail("the child went on to its transfer and pin past the budget")
        self.assertEqual(got, 0)
        self.assertEqual(fx.cache._mamba_pins_held(), budget)


class MambaUsageCountsReachableSlots(CustomTestCase):
    def _observer(self, fx):
        from sglang.srt.managers.scheduler_components.pool_stats_observer import (
            SchedulerPoolStatsObserver,
        )

        return SchedulerPoolStatsObserver(
            tree_cache=fx.cache, token_to_kv_pool_allocator=fx.allocator,
            req_to_token_pool=fx.pool, session_controller=None,
            hisparse_coordinator=None, is_hybrid_swa=False, is_hybrid_ssm=True,
            enable_hisparse=False, full_tokens_per_layer=None,
            swa_tokens_per_layer=None, max_total_num_tokens=1024,
            get_last_batch=lambda: None, get_running_batch=lambda: None,
        )

    def test_n1_idle_reads_zero(self):
        """(5) Base: (38 - 7) / 38 = 0.82 with nothing running."""
        fx = _build()
        _wake(fx, 1)
        ps = self._observer(fx).get_pool_stats()
        self.assertEqual(ps.mamba_available_size, 7)
        self.assertEqual(ps.mamba_num_used, 0)
        self.assertEqual(ps.mamba_usage, 0.0)

    def test_n1_one_running_request_reads_its_share_of_seven(self):
        fx = _build()
        _wake(fx, 1)
        taken = [fx.mamba.alloc(1) for _ in range(4)]
        self.assertTrue(all(t is not None for t in taken))
        ps = self._observer(fx).get_pool_stats()
        self.assertEqual(ps.mamba_num_used, 4)
        self.assertAlmostEqual(ps.mamba_usage, 4 / 7)

    def test_n6_unchanged(self):
        fx = _build()
        _wake(fx, 6)
        taken = [fx.mamba.alloc(1) for _ in range(4)]
        self.assertTrue(all(t is not None for t in taken))
        ps = self._observer(fx).get_pool_stats()
        self.assertEqual(ps.mamba_num_used, 4)
        self.assertAlmostEqual(ps.mamba_usage, 4 / 38)


if __name__ == "__main__":
    unittest.main()
