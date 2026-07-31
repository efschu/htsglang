"""Task #346: under the even-modulo DCP owner rule the reported context
ceiling is ``dcp_size`` times smaller than the pool can actually hold.

The two owner rules of the token-sharded lane disagree about what
``max_total_num_tokens`` MEANS, and only one of them was carried through to
the consumers:

  WEIGHTED (``--rank-tp-ratio`` + a token vector)
      ``max_total_num_tokens`` is the GLOBAL context budget C.  The allocator
      index space is C as well, and each rank physically stores its
      ``ratio_r / S`` share (``dcp_compact_pool_rows``).  Every consumer that
      reads ``max_total_num_tokens`` as a global token count is right.

  EVEN-MODULO (``SGLANG_UNEVEN_DCP=1`` with ``SGLANG_UNEVEN_DCP_WEIGHTED=0``)
      ``max_total_num_tokens`` is this rank's PHYSICAL pool P, while the
      allocator hands out ``P * cp_token_split_factor(dcp_size)`` GLOBAL slot
      ids -- global slot L lives on rank ``L % S`` at compact row ``L // S``.
      A consumer reading it as a global token count is off by a factor S.

#345 fixed the first such consumer (the leak check, which killed the boot).
This task is the same defect in the two consumers that decide how much
CONTEXT a request may have:

  * ``ModelRunner.max_token_pool_size`` -> ``max_req_len`` /
    ``max_req_input_len``, i.e. the length at which a request is REFUSED.
  * ``Scheduler.init_req_max_new_tokens`` -> the generation budget, which
    SUBTRACTS the (global) input length from that same per-rank number and
    therefore clamps to 0 for any input above the per-rank pool.

Both had already been fixed for the weightless spill lane alone
(``_wl_spill_global_capacity``); every other even-modulo deployment kept the
small number.

"Capacity, not correctness" is verified here rather than assumed:
:class:`TestReclaimedWindowIsAddressable` walks the whole reclaimed window
and checks that each slot in it maps to a valid owner rank and a compact row
inside the pool AND inside the #352/#355 store bound.  If that walk fails the
ceiling must NOT be raised.

CPU-only integer math plus one tiny tensor for the real ``kv_store_bound``.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.distributed.utils import (
    cp_token_split_factor,
    get_cp_token_ratios,
    get_tp_partition_ratios,
    set_cp_token_ratios,
    set_tp_partition_ratios,
    uneven_dcp_active,
    uneven_dcp_kv_replicated,
)
from sglang.srt.layers.dcp.owner import (
    dcp_accounting_total_slots,
    dcp_global_context_slots,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.memory_pool import graph_safe_store_bound, kv_store_bound
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

#: (dcp_size, --rank-tp-ratio vector). The token axis is UNIFORM in every one
#: of them -- that is what makes the owner rule even-modulo; the head axis is
#: uneven, which is what puts the deployment on the lane at all.
LANE_GEOMETRIES = (
    (2, [3, 1]),
    (3, [4, 2, 1]),
    (4, [5, 3, 2, 1]),
)


class _PoolRowStub(ModelRunnerKVCacheMixin):
    """Just enough runner for ``_dcp_token_sharded_pool_rows`` (see #345)."""

    def __init__(self, *, dcp_size, is_draft_worker=False):
        self.dcp_size = dcp_size
        self.is_draft_worker = is_draft_worker


class _LaneFixture(CustomTestCase):
    """Installs the even-modulo lane process-globally (see the #345 fixture)."""

    def setUp(self):
        self._saved_tp = get_tp_partition_ratios()
        self._saved_cp = get_cp_token_ratios()

    def tearDown(self):
        set_tp_partition_ratios(self._saved_tp)
        set_cp_token_ratios(self._saved_cp)

    def install_even_modulo(self, tp_ratio, *, token_vector=None):
        """The even-modulo owner rule: a head plan, no uneven token vector."""
        set_tp_partition_ratios(list(tp_ratio))
        set_cp_token_ratios(token_vector)
        dcp_size = len(tp_ratio)
        self.assertTrue(uneven_dcp_kv_replicated(dcp_size))
        self.assertFalse(uneven_dcp_active(dcp_size))
        return cp_token_split_factor(dcp_size)

    @staticmethod
    def runner_stub(*, dcp_size, max_total, **kw):
        return SimpleNamespace(
            dcp_size=dcp_size,
            max_total_num_tokens=max_total,
            is_hybrid_swa=False,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
            **kw,
        )

    @staticmethod
    def pool_size_of(stub):
        return ModelRunner.max_token_pool_size.fget(stub)


class TestContextCeiling(_LaneFixture):
    """Defect 1: the ceiling itself."""

    per_rank_pool = 28640  # the pool from the #345 boot log

    def test_even_modulo_ceiling_is_the_global_slot_span(self):
        # FALSIFIER. The allocator index space is P * S and every id in it is
        # addressable (TestReclaimedWindowIsAddressable), so the ceiling that
        # decides max_req_len must be that number, not the per-rank pool.
        for dcp_size, tp_ratio in LANE_GEOMETRIES:
            for token_vector in (None, [1] * dcp_size):
                with self.subTest(dcp_size=dcp_size, token_vector=token_vector):
                    split = self.install_even_modulo(
                        tp_ratio, token_vector=token_vector
                    )
                    self.assertEqual(split, dcp_size)
                    stub = self.runner_stub(
                        dcp_size=dcp_size, max_total=self.per_rank_pool
                    )
                    self.assertEqual(
                        self.pool_size_of(stub),
                        self.per_rank_pool * dcp_size,
                        "the even-modulo ceiling under-reports by exactly "
                        "dcp_size",
                    )

    def test_ceiling_equals_the_leak_checks_allocator_total(self):
        # The #345 leak check already counts the allocator's index space.
        # The ceiling and that total are the same quantity; if they ever
        # disagree, one of the two is reading the wrong space again.
        for dcp_size, tp_ratio in LANE_GEOMETRIES:
            with self.subTest(dcp_size=dcp_size):
                self.install_even_modulo(tp_ratio)
                stub = self.runner_stub(
                    dcp_size=dcp_size, max_total=self.per_rank_pool
                )
                allocator_size = self.per_rank_pool * dcp_size
                self.assertEqual(
                    self.pool_size_of(stub),
                    dcp_accounting_total_slots(
                        self.per_rank_pool, allocator_size, token_sharded_dcp=True
                    ),
                )

    def test_weighted_rule_is_untouched(self):
        # max_total_num_tokens is ALREADY the global context C there.
        set_tp_partition_ratios([3, 1])
        set_cp_token_ratios([17, 15])
        self.assertTrue(uneven_dcp_active(2))
        stub = self.runner_stub(dcp_size=2, max_total=61088)
        self.assertEqual(self.pool_size_of(stub), 61088)

    def test_off_the_lane_is_byte_identical(self):
        # No DCP at all, and stock even DCP (no --rank-tp-ratio head plan):
        # neither is this fork's lane, so neither moves.
        set_tp_partition_ratios(None)
        set_cp_token_ratios(None)
        self.assertEqual(
            self.pool_size_of(self.runner_stub(dcp_size=1, max_total=4096)), 4096
        )
        self.assertFalse(uneven_dcp_kv_replicated(3))
        self.assertEqual(
            self.pool_size_of(self.runner_stub(dcp_size=3, max_total=4096)), 4096
        )

    def test_spill_lanes_precomputed_capacity_still_wins(self):
        # _wl_spill_global_capacity is ALREADY per-rank x dcp_size; applying
        # the factor on top of it would inflate the ceiling by dcp_size^2.
        self.install_even_modulo([3, 1])
        stub = self.runner_stub(
            dcp_size=2, max_total=1000, _wl_spill_global_capacity=2000
        )
        self.assertEqual(self.pool_size_of(stub), 2000)

    def test_hybrid_swa_takes_the_full_subpool_through_the_same_factor(self):
        self.install_even_modulo([4, 2, 1])
        stub = self.runner_stub(dcp_size=3, max_total=0)
        stub.is_hybrid_swa = True
        stub.full_max_total_num_tokens = 900
        stub.swa_max_total_num_tokens = 300
        self.assertEqual(self.pool_size_of(stub), 2700)


class TestGenerationBudget(_LaneFixture):
    """Defect 2: the consumer that re-subtracts.

    ``init_req_max_new_tokens`` subtracts the GLOBAL input length from the
    pool capacity. Fed the per-rank pool it hands out a 0-token generation
    budget for inputs the group can hold comfortably -- the same mistake as
    the ceiling, one subtraction further down.
    """

    class _StubScheduler:
        init_req_max_new_tokens = Scheduler.init_req_max_new_tokens
        _global_kv_capacity_tokens = Scheduler._global_kv_capacity_tokens

    def scheduler_stub(self, *, dcp_size, max_total, max_req_len, wl_global=0):
        sched = self._StubScheduler()
        sched.page_size = 1
        sched.max_total_num_tokens = max_total
        sched.max_req_len = max_req_len
        sched.server_args = SimpleNamespace(dcp_size=dcp_size)
        sched.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(_wl_spill_global_capacity=wl_global)
        )
        return sched

    @staticmethod
    def request(input_len):
        return SimpleNamespace(
            origin_input_ids=[7] * input_len,
            sampling_params=SimpleNamespace(max_new_tokens=None),
        )

    def test_input_that_fits_globally_gets_a_generation_budget(self):
        # FALSIFIER. P = 1000 per rank, S = 3 -> 3000 global slots. A 1500
        # token prompt fits (it costs 500 slots per rank) but the per-rank
        # number makes `_pool_cap - paged_input_len - page_size - 1` negative.
        for dcp_size, tp_ratio in LANE_GEOMETRIES:
            with self.subTest(dcp_size=dcp_size):
                self.install_even_modulo(tp_ratio)
                per_rank = 1000
                global_cap = per_rank * dcp_size
                input_len = per_rank + 500
                sched = self.scheduler_stub(
                    dcp_size=dcp_size,
                    max_total=per_rank,
                    max_req_len=global_cap - 1,
                )
                req = self.request(input_len)
                sched.init_req_max_new_tokens(req)
                self.assertEqual(
                    req.sampling_params.max_new_tokens,
                    global_cap - input_len - 1 - 1,
                )
                self.assertGreater(req.sampling_params.max_new_tokens, 0)

    def test_off_the_lane_the_budget_is_byte_identical(self):
        set_tp_partition_ratios(None)
        set_cp_token_ratios(None)
        sched = self.scheduler_stub(dcp_size=1, max_total=1000, max_req_len=999)
        req = self.request(400)
        sched.init_req_max_new_tokens(req)
        self.assertEqual(req.sampling_params.max_new_tokens, 1000 - 400 - 1 - 1)

    def test_spill_lane_keeps_its_own_global_capacity(self):
        self.install_even_modulo([3, 1])
        sched = self.scheduler_stub(
            dcp_size=2, max_total=1000, max_req_len=9999, wl_global=5000
        )
        req = self.request(400)
        sched.init_req_max_new_tokens(req)
        self.assertEqual(req.sampling_params.max_new_tokens, 5000 - 400 - 1 - 1)

    def test_an_explicit_max_new_tokens_is_still_respected(self):
        self.install_even_modulo([3, 1])
        sched = self.scheduler_stub(dcp_size=2, max_total=1000, max_req_len=1999)
        req = self.request(100)
        req.sampling_params.max_new_tokens = 16
        sched.init_req_max_new_tokens(req)
        self.assertEqual(req.sampling_params.max_new_tokens, 16)


class TestReclaimedWindowIsAddressable(_LaneFixture):
    """The premise check: is the window the fix reclaims actually usable?

    Raising a ceiling is only a capacity fix if every slot it newly admits is
    one the write path can address. Under the even-modulo rule global slot L
    belongs to rank ``L % S`` and compacts to row ``L // S``; the pool holds
    ``size + page_size`` rows and the #352/#355 writers bound every store by
    ``kv_store_bound(size + page_size, ...)``.
    """

    def test_every_slot_in_the_window_maps_into_the_pool_and_the_bound(self):
        page_size = 1
        for dcp_size, tp_ratio in LANE_GEOMETRIES:
            with self.subTest(dcp_size=dcp_size):
                self.install_even_modulo(tp_ratio)
                per_rank = 64  # the pool `size`, i.e. _dcp_token_sharded_pool_rows
                pool_rows = per_rank + page_size
                head_num, head_dim = 2, 4
                row_dim = head_num * head_dim
                k_buf = torch.empty(pool_rows * row_dim, dtype=torch.uint8)
                bound = kv_store_bound(pool_rows, k_buf, row_dim)
                self.assertEqual(bound, graph_safe_store_bound(pool_rows, pool_rows))

                old_ceiling = per_rank
                new_ceiling = per_rank * dcp_size
                reclaimed = range(old_ceiling + 1, new_ceiling + 1)
                self.assertEqual(len(reclaimed), per_rank * (dcp_size - 1))

                seen_owners = set()
                # Allocator slot ids run 1..size inclusive (page 0 is the
                # dummy), so the walk includes the top id itself.
                for loc in reclaimed:
                    owner = loc % dcp_size
                    compact = loc // dcp_size
                    seen_owners.add(owner)
                    self.assertLess(
                        compact,
                        pool_rows,
                        f"slot {loc} compacts to row {compact}, outside a "
                        f"{pool_rows}-row pool",
                    )
                    self.assertLess(compact, bound)
                self.assertEqual(seen_owners, set(range(dcp_size)))

    def test_the_fix_allocates_nothing_new(self):
        # The claim is "capacity report, not allocation". Both numbers the
        # allocation is built from -- the pool row count and, through
        # max_total_num_tokens, the allocator index space -- are untouched by
        # the ceiling, and the ceiling lands exactly on the index space that
        # already exists. Pinned so a later change cannot quietly turn the
        # widened report into a widened allocation.
        per_rank = 28640
        for dcp_size, tp_ratio in LANE_GEOMETRIES:
            with self.subTest(dcp_size=dcp_size):
                self.install_even_modulo(tp_ratio)
                runner = _PoolRowStub(dcp_size=dcp_size)
                self.assertEqual(
                    runner._dcp_token_sharded_pool_rows(per_rank), per_rank
                )
                self.assertEqual(
                    dcp_global_context_slots(per_rank, dcp_size),
                    per_rank * dcp_size,  # == the allocator's dcp_alloc_size
                )

    def test_the_window_is_exactly_the_factor_the_fix_claims(self):
        for dcp_size, tp_ratio in LANE_GEOMETRIES:
            with self.subTest(dcp_size=dcp_size):
                self.install_even_modulo(tp_ratio)
                per_rank = 28640
                self.assertEqual(
                    dcp_global_context_slots(per_rank, dcp_size) - per_rank,
                    per_rank * (dcp_size - 1),
                )


class TestGlobalContextSlotsHelper(_LaneFixture):
    """The one place the lane is resolved, checked on its own."""

    def test_weighted_lane_returns_the_budget_unchanged(self):
        set_tp_partition_ratios([3, 1])
        set_cp_token_ratios([17, 15])
        self.assertEqual(dcp_global_context_slots(61088, 2), 61088)

    def test_even_modulo_lane_multiplies_by_the_split_factor(self):
        self.install_even_modulo([3, 1])
        self.assertEqual(dcp_global_context_slots(28640, 2), 57280)

    def test_no_dcp_and_stock_dcp_are_the_identity(self):
        set_tp_partition_ratios(None)
        set_cp_token_ratios(None)
        self.assertEqual(dcp_global_context_slots(4096, 1), 4096)
        self.assertEqual(dcp_global_context_slots(4096, 4), 4096)

    def test_degenerate_inputs(self):
        self.install_even_modulo([3, 1])
        self.assertEqual(dcp_global_context_slots(0, 2), 0)
        self.assertEqual(dcp_global_context_slots(100, None), 100)


if __name__ == "__main__":
    unittest.main()
