"""H95d: the slots a D phase of n < cap seats withholds are a NAMED posten of
the idle mamba ledger, not a leak.

MEASURED, boot fnFL2h91bb2 (tree e17bd548b5, the first metal boot with H95c),
D log ``/spinning/evidence-665-f1/boot_weg2_fnFL2h91bb2_e17bd548b5_0926_142251.D.log``
Z. 14022/14057/14100-14163, 14:31:00: the first 97k flip P->D woke D with
``handoff_n=1 -> n=1 of cap 6`` and ``mamba_slots=1..7``; the first idle pass
after the wake raised on ALL THREE ranks::

    ValueError: pool memory leak detected! ...
    [mamba] total=38, available=7, evictable=0, withheld=0, ...
            leaked_mamba_pages={8, ..., 38}   (every one slot_used=False)

ROOT: ``MambaSlotAllocator.set_phase_limit`` (H95c) keeps the slots above
L(n) in ``_phase_withheld`` -- free, unowned, without pages -- and the idle
ledger (``SchedulerInvariantChecker._check_mamba_pool``) passed ``withheld=0``,
so 31 free slots read as lost. Fix: the allocator publishes the count
(``phase_withheld_slots``) and the ids, the ledger carries them as its
``withheld`` term with a name (``withheld_by=d-seat-phase(H95c)``), the #924
occupancy and the double-free census see them as free ids.

WHAT MUST HOLD.
(1) bb2 form (38 slots, n=1, 7 free): no leak verdict, the term is named.
(2) A slot that lost its owner INSIDE the phase's 1..L still is a leak, and
    the diagnosis names exactly that slot (not the withheld ones).
(3) n = 6 (full pool): nothing withheld, the ledger as before H95c.
(4) n = 6 -> n = 1 -> n = 3 over two wakes, with served requests and cached
    anchors: the ledger balances in every phase.
(5) A duplicate inside the withheld set is a double free: fatal and named.
(6) Replicated: three ranks with the same inputs read the same verdict.

The tests drive the real allocator, a real ``UnifiedRadixCache`` and the real
checker on CPU, and assert on the ledger the boot died on.
"""

import unittest
from array import array

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

#: the NF D form of bb2: 38 GDN slots at --d-bs 6 (phase_slot_limit 7/13/19/25/32/38)
MAMBA_SLOTS = 38
CAP = 6
NUM_LAYERS = 24
FULL_LAYER_IDS = (3, 7, 11, 15, 19, 23)
NON_FULL_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in set(FULL_LAYER_IDS)]


class _Fx:
    def __init__(self, cache, allocator, pool):
        self.cache, self.allocator, self.pool = cache, allocator, pool

    @property
    def mamba(self):
        return self.pool.mamba_allocator


def _build() -> _Fx:
    """FULL+MAMBA ``UnifiedRadixCache`` on CPU (the #924 aliasing test's
    fixture, 38 mamba slots)."""
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    set_global_server_args_for_scheduler(server_args)
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=256, n_groups=1, num_heads=2,
            head_dim=16, state_size=16, conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=NON_FULL_LAYER_IDS)
    pool = HybridReqToTokenPool(
        size=10, mamba_size=MAMBA_SLOTS, mamba_spec_state_size=10,
        max_context_len=512, device="cpu", enable_memory_saver=False,
        cache_params=cache_params, mamba_layer_ids=NON_FULL_LAYER_IDS,
        enable_mamba_extra_buffer=False, speculative_num_draft_tokens=3,
    )
    kv_pool = HybridLinearKVPool(
        size=512, dtype=torch.bfloat16, page_size=1, head_num=2, head_dim=64,
        full_attention_layer_ids=list(FULL_LAYER_IDS), device="cpu",
        enable_memory_saver=False, mamba_pool=pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=512, dtype=torch.bfloat16, device="cpu", kvcache=kv_pool,
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
    return _Fx(cache, allocator, pool)


def _serve(fx: _Fx, i: int) -> None:
    """One served request (prompt -> unfinished insert -> output -> finished
    insert -> free): leaves its mamba anchors in the tree, evictable."""
    prompt = list(range(1000 * (i + 1), 1000 * (i + 1) + 8))
    out = list(range(50000 + 100 * i, 50000 + 100 * i + 4))
    req = Req(
        rid=f"h95d-{i}", origin_input_text="",
        origin_input_ids=array("q", prompt),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    fx.pool.alloc([req])
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", prompt)
    req.set_extend_range(0, len(prompt))
    kv = fx.allocator.alloc(len(prompt))
    fx.pool.write((req.req_pool_idx, slice(0, len(prompt))), kv)
    req.kv_committed_len = len(prompt)
    req.last_node = fx.cache.root_node
    req.cache_protected_len = 0
    req.swa_uuid_for_lock = None
    req.extra_key = None
    req.mamba_last_track_seqlen = len(prompt)
    fx.cache.cache_unfinished_req(req)
    total = len(prompt) + len(out)
    req.output_ids = array("q", out)
    req.full_untruncated_fill_ids = array("q", prompt + out)
    kv2 = fx.allocator.alloc(len(out))
    fx.pool.write((req.req_pool_idx, slice(len(prompt), total)), kv2)
    req.kv_committed_len = total
    req.set_extend_range(0, total)
    req.mamba_last_track_seqlen = total
    fx.cache.cache_finished_req(req, is_insert=True)
    fx.pool.free(req)


def _wake(fx: _Fx, n: int, *, reset_tree: bool) -> int:
    """The D wake as the log orders it: ``d_seat_vram.on_wake`` sets the slot
    limit (``WEG2 D-SEAT-VRAM (H95c) ... mamba_slots=1..L``), then the pools
    are flushed (``Reset HybridReqToTokenPool``). ``reset_tree`` = the tree
    the D phase starts with is empty (bb2: ``TREE CENSUS nodes=1``)."""
    from sglang.srt.weg2.d_seat_vram import phase_slot_limit

    if reset_tree:
        fx.cache.reset()
    want = None if n >= CAP else phase_slot_limit(MAMBA_SLOTS, n, CAP)
    assert fx.mamba.set_phase_limit(want), (n, want)
    if reset_tree:
        fx.pool.clear()
    return MAMBA_SLOTS if want is None else want


def _checker(fx: _Fx):
    from sglang.srt.managers.scheduler_components.invariant_checker import (
        SchedulerInvariantChecker,
    )

    class _Observer:
        @staticmethod
        def session_held_mamba_slots():
            return 0

    checker = SchedulerInvariantChecker.__new__(SchedulerInvariantChecker)
    checker.req_to_token_pool = fx.pool
    checker.tree_cache = fx.cache
    checker.pool_stats_observer = _Observer()
    checker.server_args = ServerArgs(model_path="dummy", page_size=1)
    checker.token_to_kv_pool_allocator = fx.allocator
    checker.get_token_to_kv_pool_allocator = None
    return checker


def _ledger(fx: _Fx):
    """``_check_mamba_pool`` over the live stats, the way on_idle reads them
    (``PoolStatsObserver._get_mamba_token_info``)."""
    from sglang.srt.managers.scheduler_components.pool_stats_observer import (
        PoolStats,
    )

    available = fx.mamba.available_size()
    evictable = fx.cache.mamba_evictable_size()
    ps = PoolStats(
        full_num_used=0, full_token_usage=0.0, full_available_size=0,
        full_evictable_size=0, is_hybrid_ssm=True,
        mamba_num_used=MAMBA_SLOTS - (available + evictable), mamba_usage=0.0,
        mamba_available_size=available, mamba_evictable_size=evictable,
    )
    return _checker(fx)._check_mamba_pool(ps)


def _leaked_mamba(msg: str):
    head = msg.split("leaked_mamba_pages=", 1)[1]
    if head.startswith("None"):
        return set()
    return {int(x) for x in head[1: head.index("}")].split(",")}


class TheBb2FormIsNotALeak(CustomTestCase):
    def test_n1_after_the_wake_balances_and_names_the_withheld_slots(self):
        """(1) The bb2 line, rebuilt: 38 slots, n=1, 7 free -- red on
        e17bd548b5 (``withheld=0`` -> leak, leaked 8..38)."""
        fx = _build()
        limit = _wake(fx, 1, reset_tree=True)
        self.assertEqual(limit, 7)
        self.assertEqual(fx.mamba.available_size(), 7)
        leak, msg = _ledger(fx)
        self.assertFalse(leak, msg)
        self.assertIn("total=38, available=7, evictable=0, withheld=31", msg)
        self.assertIn("withheld_by=d-seat-phase(H95c) limit=1..7", msg)
        self.assertNotIn("#924 MAMBA SLOT ALIASING", msg)

    def test_the_allocator_publishes_the_withheld_count_and_ids(self):
        fx = _build()
        _wake(fx, 1, reset_tree=True)
        self.assertEqual(fx.mamba.phase_withheld_slots, 31)
        self.assertEqual(sorted(fx.mamba.phase_withheld_ids()), list(range(8, 39)))
        self.assertTrue(fx.mamba.set_phase_limit(None))
        self.assertEqual(fx.mamba.phase_withheld_slots, 0)
        self.assertEqual(fx.mamba.phase_withheld_ids(), [])


class ARealLeakInsideTheSeatsStaysFatal(CustomTestCase):
    def test_a_slot_whose_owner_is_gone_inside_1_to_L_is_a_leak(self):
        """(2) The withheld term never absorbs a lost slot: nothing above the
        limit is ever handed out, so the lost one is inside 1..7 and missing
        from available + evictable + withheld."""
        fx = _build()
        _wake(fx, 1, reset_tree=True)
        lost = fx.mamba.alloc(1)
        self.assertIsNotNone(lost)
        lost_id = int(lost.item())
        self.assertLessEqual(lost_id, 7)
        del lost  # the owner is gone, the slot never came back
        leak, msg = _ledger(fx)
        self.assertTrue(leak, msg)
        self.assertIn("available=6, evictable=0, withheld=31", msg)
        # the diagnosis names exactly the lost slot, not the withheld 8..38
        self.assertEqual(_leaked_mamba(msg), {lost_id}, msg)

    def test_a_duplicate_in_the_withheld_set_is_a_named_double_free(self):
        """(5) A withheld id listed twice would inflate the term; it is the
        #924 double free and stays fatal with its id."""
        fx = _build()
        _wake(fx, 1, reset_tree=True)
        fx.mamba._phase_withheld = torch.cat(
            (fx.mamba._phase_withheld, torch.tensor([20], dtype=torch.int64))
        )
        leak, msg = _ledger(fx)
        self.assertTrue(leak, msg)
        self.assertIn("duplicate_slot_ids=[20]", msg)

    def test_a_withheld_id_the_tree_also_names_is_aliasing(self):
        """A withheld id is a free id: a tree node still naming it is the
        #924 double count, visible to the occupancy term."""
        fx = _build()
        _wake(fx, 1, reset_tree=True)
        _serve(fx, 0)
        cached = sorted(int(v) for v in fx.cache.all_mamba_values_flatten().tolist())
        self.assertTrue(cached, "the served request left no anchor")
        fx.mamba._phase_withheld = torch.cat(
            (fx.mamba._phase_withheld, torch.tensor(cached[:1], dtype=torch.int64))
        )
        leak, msg = _ledger(fx)
        self.assertTrue(leak, msg)
        self.assertIn("#924 MAMBA SLOT ALIASING", msg)


class EveryPhaseFormBalances(CustomTestCase):
    def test_n6_the_full_pool_withholds_nothing(self):
        """(3) n = cap: the limit is lifted, the ledger is the pre-H95c one."""
        fx = _build()
        limit = _wake(fx, 6, reset_tree=True)
        self.assertEqual(limit, MAMBA_SLOTS)
        self.assertIsNone(fx.mamba.phase_limit)
        leak, msg = _ledger(fx)
        self.assertFalse(leak, msg)
        self.assertIn("total=38, available=38, evictable=0, withheld=0", msg)
        self.assertNotIn("withheld_by", msg)
        for i in range(4):
            _serve(fx, i)
        leak, msg = _ledger(fx)
        self.assertFalse(leak, msg)

    def test_n6_then_n1_then_n3_over_two_wakes(self):
        """(4) Wake 1 n=6, wake 2 n=1 (tree emptied, pools flushed), then the
        phase grows to n=3 while the n=1 phase's anchors stay cached -- the
        ledger balances after every step and names the withheld count."""
        fx = _build()
        _wake(fx, 6, reset_tree=True)
        for i in range(3):
            _serve(fx, i)
        leak, msg = _ledger(fx)
        self.assertFalse(leak, msg)
        self.assertIn("withheld=0", msg)

        limit = _wake(fx, 1, reset_tree=True)
        self.assertEqual(limit, 7)
        leak, msg = _ledger(fx)
        self.assertFalse(leak, msg)
        self.assertIn("withheld=31", msg)
        _serve(fx, 10)
        cached = [int(v) for v in fx.cache.all_mamba_values_flatten().tolist()]
        self.assertTrue(cached and max(cached) <= 7, cached)
        leak, msg = _ledger(fx)
        self.assertFalse(leak, msg)
        self.assertIn("withheld=31", msg)

        limit = _wake(fx, 3, reset_tree=False)
        self.assertEqual(limit, 19)
        self.assertEqual(fx.mamba.phase_withheld_slots, MAMBA_SLOTS - 19)
        leak, msg = _ledger(fx)
        self.assertFalse(leak, msg)
        self.assertIn("withheld=19", msg)
        self.assertIn("limit=1..19", msg)
        for i in range(11, 14):
            _serve(fx, i)
        leak, msg = _ledger(fx)
        self.assertFalse(leak, msg)
        self.assertIn("withheld=19", msg)


class TheTermIsReplicated(CustomTestCase):
    def test_three_ranks_same_inputs_same_verdict(self):
        """(6) The limit is a function of the wake request, the term a
        function of the limit and the rank's own ledger: no collective, the
        same reading on TP0/TP1/TP2."""
        msgs = []
        for _rank in range(3):
            fx = _build()
            _wake(fx, 1, reset_tree=True)
            leak, msg = _ledger(fx)
            self.assertFalse(leak, msg)
            msgs.append(msg)
        self.assertEqual(len(set(msgs)), 1, msgs)


if __name__ == "__main__":
    unittest.main()
