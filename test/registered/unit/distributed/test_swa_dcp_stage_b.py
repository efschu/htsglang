"""#96 SWA-DCP Stage B: the parts that decide correctness and need no GPU.

Stage B shards the ~10 GLOBAL full-attention layers of an SWA-hybrid model
(Gemma-4 class) with the weighted owner rule of #173, and leaves the ~50
sliding-window layers on their unsharded local path. What that costs in code is
a lane predicate, a per-layer dispatch, one sizing expression, and one head-count
base -- all integer/boolean, all pinned here. The Triton kernels are unchanged,
so the device half is a boot recipe (docs_new/swa_dcp_stage_b_triton.md section
8), not a unit test.

The five things checked:

1. ``dcp_compact_pool_rows``: the compact full-pool sizing, including that the
   HIGHEST allocator slot still lands inside the sized pool (the ceil-to-a-whole-
   owner-block rule, which was an out-of-bounds scatter before it existed).
2. ``swa_hybrid_dcp_lane``: the truth table, and in particular that it is OFF
   for a draft worker, without a plan, without cap sizing, and for a model that
   is not actually hybrid.
3. ``dcp_token_sharded_layer``: off-lane every layer is sharded (byte-identical
   #173), on-lane exactly the window layers are not.
4. The DCP group's q-head counts must be an EXHAUSTIVE partition, which
   ``max()`` over a hybrid model's two kv-head bases is not.
5. Routing: an SWA layer's write never carries a DCP owner mask, and the
   per-layer dispatch is spelled with the shared predicate at every call site.
"""

import pathlib
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    set_tp_partition_ratios,
)
from sglang.srt.layers.dcp.owner import (
    dcp_compact_pool_rows,
    dcp_token_sharded_layer,
    dcp_weighted_owner_bounds,
    dcp_weighted_write_slots,
    swa_hybrid_dcp_lane,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


LANE_ON = dict(
    is_hybrid_swa=True,
    uneven_plan=True,
    is_draft_worker=False,
    num_full_layers=10,
    num_swa_layers=50,
    swa_pool_sizing_capped=True,
)


class TestCompactPoolRows(CustomTestCase):
    def test_the_even_and_aligned_case_is_the_obvious_fraction(self):
        # S = 3 ranks x ratio 1: a third of the context each, plus the ceil block
        self.assertEqual(dcp_compact_pool_rows(300, 3, 1), 101)
        # weighted [2,1,1]: rank 0 holds half of a 400-slot context
        self.assertEqual(dcp_compact_pool_rows(400, 4, 2), 202)

    def test_a_single_rank_owning_everything_holds_everything(self):
        """ratio == S is the degenerate one-owner case; it must not be short."""
        self.assertGreaterEqual(dcp_compact_pool_rows(1000, 4, 4), 1000)

    def test_a_context_smaller_than_one_owner_block_still_gets_rows(self):
        """C < S happens at tiny --max-total-tokens; flooring alone would give
        ZERO rows and every write would scatter out of bounds."""
        self.assertEqual(dcp_compact_pool_rows(2, 64, 30), 30)

    def test_the_highest_allocator_slot_lands_inside_the_sized_pool(self):
        """THE property the ``+ 1`` block exists for (found as an async illegal
        memory access with --max-total-tokens 3000 on S=64).

        The allocator hands out slot ids up to C itself. For every rank, the
        compact row of every owned slot in [0, C] must be < the sized rows.
        """
        for C in (2, 3, 63, 64, 65, 3000, 4096, 262160):
            for plan in ([1, 1, 1], [2, 1, 1], [13, 30, 21], [30, 17, 17]):
                S = sum(plan)
                for rank in range(len(plan)):
                    set_tp_partition_ratios(None)
                    lo = sum(plan[:rank])
                    ratio = plan[rank]
                    rows = dcp_compact_pool_rows(C, S, ratio)
                    slots = torch.arange(C + 1, dtype=torch.int64)
                    loc, mask = dcp_weighted_write_slots(
                        slots, S, lo, lo + ratio, ratio
                    )
                    owned = loc[mask]
                    with self.subTest(C=C, plan=plan, rank=rank):
                        if owned.numel():
                            self.assertLess(
                                int(owned.max()),
                                rows,
                                f"slot {int(slots[mask][owned.argmax()])} compacts "
                                f"to row {int(owned.max())} but the pool holds "
                                f"{rows}",
                            )

    def test_the_bounds_helper_and_the_sizing_agree_on_the_same_numbers(self):
        """The rows come from (cp_S, cp_ratio) of dcp_weighted_owner_bounds, so
        a pool sized from one derivation and written through another cannot
        happen. Pinned by deriving both from the same installed vector."""
        from sglang.srt.distributed.utils import set_cp_token_ratios

        saved = get_tp_partition_ratios()
        try:
            set_cp_token_ratios([30, 17, 17])
            for rank in range(3):
                S, lo, hi, ratio = dcp_weighted_owner_bounds(3, rank)
                self.assertEqual(S, 64)
                self.assertEqual(ratio, hi - lo)
                self.assertEqual(
                    dcp_compact_pool_rows(443904, S, ratio),
                    (443904 // 64 + 1) * ratio,
                )
        finally:
            set_cp_token_ratios(None)
            set_tp_partition_ratios(saved)

    def test_a_nonsense_split_is_refused_rather_than_sized(self):
        with self.assertRaises(ValueError):
            dcp_compact_pool_rows(1000, 0, 1)
        with self.assertRaises(ValueError):
            dcp_compact_pool_rows(1000, 3, 0)


class TestLanePredicate(CustomTestCase):
    def test_the_gemma4_class_configuration_is_the_lane(self):
        self.assertTrue(swa_hybrid_dcp_lane(**LANE_ON))

    def test_every_single_condition_is_necessary(self):
        for key, off in (
            ("is_hybrid_swa", False),
            ("uneven_plan", False),
            ("is_draft_worker", True),
            ("num_full_layers", 0),
            ("num_swa_layers", 0),
            ("swa_pool_sizing_capped", False),
        ):
            with self.subTest(off=key):
                self.assertFalse(swa_hybrid_dcp_lane(**{**LANE_ON, key: off}))

    def test_a_pure_swa_model_is_not_the_lane(self):
        """No global layers means nothing that grows with context to shard: DCP
        would buy zero capacity and pay two collectives per layer."""
        self.assertFalse(swa_hybrid_dcp_lane(**{**LANE_ON, "num_full_layers": 0}))

    def test_the_draft_worker_is_excluded_exactly_as_in_173(self):
        """The draft/NEXTN runner keeps a full-context pool with local heads, so
        reading it through the owner rule would compact indices that were never
        compacted."""
        self.assertFalse(swa_hybrid_dcp_lane(**{**LANE_ON, "is_draft_worker": True}))

    def test_ratio_sizing_is_not_the_lane_because_it_cannot_fit(self):
        """Stage B requires Stage A: with a ratio-sized SWA pool the unsharded
        SWA side is scaled by the GLOBAL context budget."""
        self.assertFalse(
            swa_hybrid_dcp_lane(**{**LANE_ON, "swa_pool_sizing_capped": False})
        )


class TestPerLayerDispatch(CustomTestCase):
    def test_off_the_lane_every_layer_is_sharded(self):
        """#173 behaviour, byte-identical: a non-hybrid model's layers all go
        through the owner rule."""
        for is_swa in (False, True):
            self.assertTrue(
                dcp_token_sharded_layer(is_swa, swa_hybrid_lane=False)
            )

    def test_on_the_lane_exactly_the_window_layers_are_not_sharded(self):
        self.assertTrue(dcp_token_sharded_layer(False, swa_hybrid_lane=True))
        self.assertFalse(dcp_token_sharded_layer(True, swa_hybrid_lane=True))

    def test_the_dispatch_is_used_at_all_three_call_sites(self):
        """Source-pinned: the write, the extend and the decode path must ask the
        SAME question. A future edit that re-routes one of them (say, leaves the
        decode branch on a bare ``dcp_size > 1``) would give a layer whose KV was
        written unsharded a sharded read -- silently wrong output, no crash."""
        import sglang.srt.layers.attention.triton_backend as tb

        src = pathlib.Path(tb.__file__).read_text()
        self.assertEqual(
            src.count("self._dcp_layer_token_sharded(layer)"),
            3,
            "expected the per-layer dispatch in _set_kv_buffer, forward_extend "
            "and forward_decode",
        )
        # and the lane flag itself comes from the shared predicate, not a local
        # re-derivation
        self.assertIn("self.swa_hybrid_dcp = swa_hybrid_dcp_lane(", src)


class _Recorder:
    def __init__(self):
        self.calls = []

    def set_kv_buffer(self, *args, **kwargs):
        self.calls.append(kwargs)


class TestSwaWriteNeverCarriesAnOwnerMask(CustomTestCase):
    """The pool half of the dispatch.

    An SWA layer's KV is replicated across the DCP group; masking that write
    would store only this rank's owned share of a window that every rank must
    hold in full -- the model then attends holes, coherently enough to look like
    a quality regression rather than a bug.
    """

    def _pool(self):
        pool = SWAKVPool.__new__(SWAKVPool)
        pool.layers_mapping = {0: (0, False), 1: (0, True)}  # 0 full, 1 swa
        pool.full_kv_pool = _Recorder()
        pool.swa_kv_pool = _Recorder()
        return pool

    def test_the_full_layer_forwards_the_mask(self):
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc

        pool = self._pool()
        mask = torch.tensor([True, False])
        loc = KVWriteLoc(torch.tensor([4, 9]), torch.tensor([1, 2]))
        SWAKVPool.set_kv_buffer(
            pool,
            SimpleNamespace(layer_id=0),
            loc,
            torch.zeros(2),
            torch.zeros(2),
            dcp_kv_mask=mask,
        )
        self.assertEqual(len(pool.full_kv_pool.calls), 1)
        self.assertIs(pool.full_kv_pool.calls[0]["dcp_kv_mask"], mask)
        self.assertEqual(pool.swa_kv_pool.calls, [])

    def test_without_a_mask_the_kwarg_is_not_even_passed(self):
        """Sub-pool classes without a dcp_kv_mask parameter (NPU / compress
        variants) must keep their exact signature off the lane."""
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc

        pool = self._pool()
        SWAKVPool.set_kv_buffer(
            pool,
            SimpleNamespace(layer_id=0),
            KVWriteLoc(torch.tensor([4]), torch.tensor([1])),
            torch.zeros(1),
            torch.zeros(1),
        )
        self.assertNotIn("dcp_kv_mask", pool.full_kv_pool.calls[0])

    def test_an_swa_layer_with_a_mask_is_an_assertion_not_a_write(self):
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc

        pool = self._pool()
        with self.assertRaises(AssertionError) as ctx:
            SWAKVPool.set_kv_buffer(
                pool,
                SimpleNamespace(layer_id=1),
                KVWriteLoc(torch.tensor([4]), torch.tensor([1])),
                torch.zeros(1),
                torch.zeros(1),
                dcp_kv_mask=torch.tensor([True]),
            )
        self.assertIn("#96", str(ctx.exception))
        self.assertEqual(pool.swa_kv_pool.calls, [])


class TestGroupQHeadCountsMustPartition(CustomTestCase):
    """#96's collective-vs-workspace distinction.

    A hybrid model carries TWO kv-head bases and therefore two different q
    partitions, each exhaustive. Taking the per-rank max (right for a buffer
    size) produces a vector that partitions nothing, and the DCP head collectives
    are driven by exactly that vector.
    """

    TOTAL_Q = 32
    PLAN = [5, 3, 2]

    def setUp(self):
        self._saved = get_tp_partition_ratios()
        set_tp_partition_ratios(self.PLAN)

    def tearDown(self):
        set_tp_partition_ratios(self._saved)

    def _counts(self, rank, total_kv, swa_kv=None):
        import sglang.srt.layers.attention.triton_backend as tb

        cfg = SimpleNamespace(
            num_attention_heads=self.TOTAL_Q,
            get_total_num_kv_heads=lambda: total_kv,
            hf_text_config=SimpleNamespace(swa_num_key_value_heads=swa_kv),
        )
        saved = tb.get_parallel
        tb.get_parallel = lambda: SimpleNamespace(attn_tp_size=3, attn_tp_rank=rank)
        try:
            return tb._plan_aware_dcp_group_q_head_counts(cfg, 3, local_heads=0)
        finally:
            tb.get_parallel = saved

    def test_the_two_bases_of_a_hybrid_really_do_disagree(self):
        """The premise, verified against the partition helpers rather than
        assumed: 32 q heads over [5,3,2] split differently for kv=16 and kv=8."""
        from sglang.srt.distributed.utils import (
            attn_q_partition_groups,
            attn_q_partition_units,
            tp_partition_size,
        )

        def vec(kv):
            return [
                tp_partition_size(
                    self.TOTAL_Q,
                    3,
                    r,
                    attn_q_partition_units(self.TOTAL_Q, kv, 3),
                    groups=attn_q_partition_groups(kv, 3),
                )
                for r in range(3)
            ]

        self.assertEqual(vec(16), [16, 10, 6])
        self.assertEqual(vec(8), [16, 8, 8])
        # the OLD expression, transcribed: per-rank max over both bases
        old = [max(a, b) for a, b in zip(vec(16), vec(8))]
        self.assertEqual(old, [16, 10, 8])
        self.assertNotEqual(sum(old), self.TOTAL_Q)  # 34 -- not a partition

    def test_the_counts_come_from_the_full_attention_base_only(self):
        """Full-attention layers are the only ones that enter a DCP collective
        under the lane, so their base is the one the counts must describe."""
        for rank in range(3):
            with self.subTest(rank=rank):
                self.assertEqual(self._counts(rank, 16, swa_kv=8), [16, 10, 6])

    def test_the_counts_are_exhaustive_for_every_hybrid_base_pair(self):
        for total_kv, swa_kv in ((16, 8), (8, 16), (2, 8), (4, 4), (16, None)):
            with self.subTest(total_kv=total_kv, swa_kv=swa_kv):
                counts = self._counts(0, total_kv, swa_kv=swa_kv)
                self.assertEqual(sum(counts), self.TOTAL_Q)

    def test_a_single_base_model_is_unchanged(self):
        """Everything #173 validated has one kv base, where old and new
        expressions are the same list."""
        self.assertEqual(self._counts(1, 8, swa_kv=None), [16, 8, 8])
        self.assertEqual(self._counts(1, 8, swa_kv=8), [16, 8, 8])

    def test_a_non_exhaustive_partition_is_refused_loudly(self):
        """The assertion that would have caught the max() bug at the first
        forward instead of inside a collective."""
        import sglang.srt.layers.attention.triton_backend as tb

        cfg = SimpleNamespace(
            # 33 q heads cannot be split into kv-head units of 16
            num_attention_heads=33,
            get_total_num_kv_heads=lambda: 16,
            hf_text_config=SimpleNamespace(swa_num_key_value_heads=None),
        )
        saved = tb.get_parallel
        tb.get_parallel = lambda: SimpleNamespace(attn_tp_size=3, attn_tp_rank=0)
        try:
            with self.assertRaises(ValueError):
                tb._plan_aware_dcp_group_q_head_counts(cfg, 3, local_heads=0)
        finally:
            tb.get_parallel = saved

    def test_no_plan_and_dcp_off_stay_pure_identities(self):
        import sglang.srt.layers.attention.triton_backend as tb

        cfg = SimpleNamespace(
            num_attention_heads=self.TOTAL_Q,
            get_total_num_kv_heads=lambda: 16,
            hf_text_config=SimpleNamespace(swa_num_key_value_heads=8),
        )
        self.assertEqual(
            tb._plan_aware_dcp_group_q_head_counts(cfg, 1, local_heads=7), [7]
        )
        set_tp_partition_ratios(None)
        saved = tb.get_parallel
        tb.get_parallel = lambda: SimpleNamespace(attn_tp_size=3, attn_tp_rank=0)
        try:
            self.assertEqual(
                tb._plan_aware_dcp_group_q_head_counts(cfg, 3, local_heads=7),
                [7, 7, 7],
            )
        finally:
            tb.get_parallel = saved


class TestPostCaptureCannotResizeTheShardedPool(CustomTestCase):
    def test_post_capture_sizing_requires_dcp_off(self):
        """SWAKVPool.finalize_backing would rewrite the full sub-pool back to the
        GLOBAL context C, undoing the compact sizing. It is unreachable because
        post-capture sizing is only planned at dcp_size == 1 -- pinned here, so a
        future relaxation of that condition fails a test instead of a rank."""
        import sglang.srt.server_args as sa_mod

        src = pathlib.Path(sa_mod.__file__).read_text()
        block = src.split("def post_capture_kv_sizing_planned")[1].split("def ")[0]
        self.assertIn("self.dcp_size == 1", block)


if __name__ == "__main__":
    unittest.main()
