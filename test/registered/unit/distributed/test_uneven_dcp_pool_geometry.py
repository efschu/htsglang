"""Task #345: the full-attention KV pool must be shaped for what the
uneven-DCP attention path actually writes into it.

Two scalars decide that, and until this task only the HYBRID pool families
(mamba/GDN, SWA-hybrid) set both:

  head_num   Under ``uneven_dcp_kv_replicated`` every rank stores the FULL,
             replicated kv-heads -- ``_dcp_write_gather`` gathers this rank's
             uneven projection shard up to ``get_total_num_kv_heads()`` and
             the paged wrappers are planned with that count. The plain-MHA
             pool (the family a DENSE model lands in) read
             ``_pool_kv_head_num`` and got the per-rank SHARD.
  rows       Under the WEIGHTED owner rule ``max_total_num_tokens`` is the
             GLOBAL context budget C, of which this rank stores its
             ``ratio_r / S`` share (``dcp_compact_pool_rows``). The plain-MHA
             pool was sized at C.

Why the head mismatch is CORRUPTION and not merely a shape bug:
``masked_set_kv_buffer_kernel`` stores at ``loc * H * D`` with ``H`` taken
from the CACHE tensor -- the full replicated count -- while the pool's real
row stride is ``H_pool * D``. Every owned slot therefore lands at an address
that drifts from its own row by ``loc * (H_write - H_pool) * D``: zero for
slot 0, growing with the slot id. Which slot ids a request gets is decided by
the allocator's free list, i.e. by how many requests ran before it -- that is
the request-ORDER dependence #343 measured (two identical greedy requests
splitting at token index 5), and it is fully deterministic, which is why the
same run reproduces byte for byte across boots.

Everything below is CPU-only integer math. The tests are written from the
rule as prose so a shared bug cannot hide in both sides.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed.utils import (
    cp_token_split_factor,
    get_cp_token_ratios,
    get_tp_partition_ratios,
    set_cp_token_ratios,
    set_tp_partition_ratios,
    uneven_dcp_kv_replicated,
)
from sglang.srt.layers.dcp.owner import (
    dcp_accounting_total_slots,
    dcp_compact_pool_rows,
)
from sglang.srt.model_executor import model_runner_kv_cache_mixin as mixin_mod
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeModelConfig:
    """Just the two head-count questions the pool sizing asks."""

    def __init__(self, total_kv_heads: int, per_rank_kv_heads):
        self._total = total_kv_heads
        self._per_rank = per_rank_kv_heads

    def get_total_num_kv_heads(self) -> int:
        return self._total

    def get_num_kv_heads(self, tp_size, rank=None) -> int:
        del tp_size
        return self._per_rank if rank is None else self._per_rank


class _StubRunner(ModelRunnerKVCacheMixin):
    """The smallest object the two pool-geometry methods read.

    ``server_args`` is part of that minimum since #108: --draft-kv-layout is
    a third input to the draft pool's geometry, alongside dcp_size and
    is_draft_worker.
    """

    def __init__(
        self,
        *,
        dcp_size,
        model_config,
        is_draft_worker=False,
        draft_kv_layout="replicated",
    ):
        self.dcp_size = dcp_size
        self.model_config = model_config
        self.is_draft_worker = is_draft_worker
        # #797 made the pool-geometry decision sites read a SEPARATE flag:
        #
        #   model_runner.py:512
        #   self.is_draft_pool_worker = (
        #       is_draft_worker and not is_phase_flip_tp_stack)
        #
        # and the comment above it states the rule these two methods now obey
        # -- they "consult THIS flag, never is_draft_worker directly", because
        # the flip's TP stack rides the draft construction gates while its
        # POOLS take the target-model treatment. So this runner, which exists
        # to be the smallest object those methods read, has to carry both.
        #
        # Derived rather than pinned: no case here constructs a phase-flip TP
        # stack, so the second term is False and the pool flag follows the
        # draft flag -- including for the two draft cases below, which are
        # ordinary draft runners and whose geometry expectations are therefore
        # unchanged. A test that did build a flip stack would have to pass the
        # two apart, and writing the rule out is what makes that visible.
        self.is_draft_pool_worker = is_draft_worker
        self.server_args = SimpleNamespace(draft_kv_layout=draft_kv_layout)


class _DcpPlanFixture(CustomTestCase):
    """Installs a --rank-tp-ratio plan plus a token vector, process-global."""

    tp_ratio = [3, 1]
    token_vector = [17, 15]
    tp_size = 2
    dcp_rank = 0
    total_kv_heads = 8
    per_rank_kv_heads = 6  # partition_sizes(8, [3,1], units=8)[0]

    def setUp(self):
        self._saved_tp = get_tp_partition_ratios()
        self._saved_cp = get_cp_token_ratios()
        set_tp_partition_ratios(self.tp_ratio)
        set_cp_token_ratios(self.token_vector)
        self._parallel = SimpleNamespace(
            attn_tp_size=self.tp_size,
            attn_tp_rank=self.dcp_rank,
            attn_dcp_size=self.dcp_size(),
            attn_dcp_rank=self.dcp_rank,
        )
        self._patch = patch.object(
            mixin_mod, "get_parallel", return_value=self._parallel
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        set_tp_partition_ratios(self._saved_tp)
        set_cp_token_ratios(self._saved_cp)

    def dcp_size(self):
        return self.tp_size

    def runner(self, **kw):
        return _StubRunner(
            dcp_size=self.dcp_size(),
            model_config=_FakeModelConfig(self.total_kv_heads, self.per_rank_kv_heads),
            **kw,
        )


class TestUnevenDcpPoolHeadNum(_DcpPlanFixture):
    def test_lane_pool_is_shaped_for_the_replicated_head_count(self):
        # THE BUG, stated as its own falsifier: the pool head count and the
        # count the attention write/read uses are the SAME number, and it is
        # the full replicated one -- not this rank's projection shard.
        self.assertTrue(uneven_dcp_kv_replicated(self.dcp_size()))
        runner = self.runner()
        self.assertEqual(
            runner._pool_kv_head_num(),
            self.total_kv_heads,
            "the uneven-DCP write gathers to the full replicated kv-heads; a "
            "pool shaped for the per-rank shard makes every owned slot land "
            "at the wrong address",
        )
        self.assertNotEqual(self.per_rank_kv_heads, self.total_kv_heads)

    def test_draft_worker_keeps_its_per_rank_shard(self):
        # The NEXTN/EAGLE draft pool is NOT token-sharded by default (the
        # flashinfer backend gates the whole uneven_dcp path on
        # `draft_pool_is_replicated`), so it must keep the local head shard or
        # the two disagree the other way round.
        runner = self.runner(is_draft_worker=True)
        self.assertEqual(runner._pool_kv_head_num(), self.per_rank_kv_heads)

    def test_draft_kv_layout_dcp_takes_the_replicated_head_count(self):
        # #108: opted in, the draft pool is the target pool's twin -- full
        # replicated kv-heads, because the same _dcp_write_gather runs for it.
        # Storing the per-rank shard here would be the #345 corruption again,
        # this time in the draft pool.
        runner = self.runner(is_draft_worker=True, draft_kv_layout="dcp")
        self.assertEqual(runner._pool_kv_head_num(), self.total_kv_heads)

    def test_no_plan_is_untouched(self):
        set_tp_partition_ratios(None)
        set_cp_token_ratios(None)
        runner = _StubRunner(
            dcp_size=1,
            model_config=_FakeModelConfig(self.total_kv_heads, self.per_rank_kv_heads),
        )
        self.assertEqual(runner._pool_kv_head_num(), self.per_rank_kv_heads)


class TestUnevenDcpPoolRows(_DcpPlanFixture):
    global_context = 61088  # C measured on the #343 boot (vector [17,15])

    def test_rows_are_this_ranks_owned_share(self):
        S = cp_token_split_factor(self.dcp_size())
        self.assertEqual(S, sum(self.token_vector))
        for rank, ratio in enumerate(self.token_vector):
            self._parallel.attn_dcp_rank = rank
            rows = self.runner()._dcp_token_sharded_pool_rows(self.global_context)
            self.assertEqual(rows, dcp_compact_pool_rows(self.global_context, S, ratio))
            self.assertLess(rows, self.global_context)

    def test_every_reachable_compact_slot_is_inside_the_pool(self):
        # The allocator's index space under the weighted rule is C itself, so
        # a global slot id can be any value in [0, C). Walk the whole space
        # and check the owner rule never addresses past the pool.
        S = cp_token_split_factor(self.dcp_size())
        prefix = [0]
        for r in self.token_vector:
            prefix.append(prefix[-1] + r)
        for rank, ratio in enumerate(self.token_vector):
            self._parallel.attn_dcp_rank = rank
            rows = self.runner()._dcp_token_sharded_pool_rows(self.global_context)
            lo, hi = prefix[rank], prefix[rank + 1]
            worst = -1
            for loc in range(self.global_context):
                off = loc % S
                if lo <= off < hi:
                    worst = max(worst, (loc // S) * ratio + (off - lo))
            self.assertGreaterEqual(worst, 0)
            self.assertLess(
                worst,
                rows,
                f"rank {rank}: compact slot {worst} outside a {rows}-row pool",
            )

    def test_draft_and_off_lane_keep_the_global_size(self):
        self.assertEqual(
            self.runner(is_draft_worker=True)._dcp_token_sharded_pool_rows(
                self.global_context
            ),
            self.global_context,
        )
        # Even-modulo owner rule (SGLANG_UNEVEN_DCP_WEIGHTED=0): no token
        # vector, so max_total_num_tokens IS the per-rank pool already and
        # the allocator carries the inflated index space instead.
        set_cp_token_ratios(None)
        self.assertEqual(
            self.runner()._dcp_token_sharded_pool_rows(self.global_context),
            self.global_context,
        )

    def test_draft_kv_layout_dcp_shards_the_draft_rows(self):
        """#108's whole point, as a number: the opted-in draft pool holds this
        rank's owned share, not the global context.

        This is the sizing falsifier -- with the flag on and nothing else
        changed, per-rank draft KV rows must drop by the shard factor
        ratio_r / S. If it ever returns global_context again the flag is
        inert and the feature buys nothing.
        """
        S = cp_token_split_factor(self.dcp_size())
        for rank, ratio in enumerate(self.token_vector):
            self._parallel.attn_dcp_rank = rank
            rows = self.runner(
                is_draft_worker=True, draft_kv_layout="dcp"
            )._dcp_token_sharded_pool_rows(self.global_context)
            self.assertEqual(rows, dcp_compact_pool_rows(self.global_context, S, ratio))
            self.assertLess(rows, self.global_context)
            # and it is the SAME row count the target pool gets on this rank
            self.assertEqual(
                rows,
                self.runner()._dcp_token_sharded_pool_rows(self.global_context),
            )


class TestUnevenDcpWriteAddressing(_DcpPlanFixture):
    """The corruption itself, and why it depends on request ORDER.

    A pool row is ``H_pool * D`` elements; the store kernel writes ``H_write *
    D`` elements starting at ``compact * H_write * D``. Model both as flat
    element ranges and ask the only question that matters: does slot ``c``'s
    write stay inside row ``c``?
    """

    head_dim = 128

    def _writes(self, locs, *, h_write, h_pool, rank):
        S = cp_token_split_factor(self.dcp_size())
        prefix = [0]
        for r in self.token_vector:
            prefix.append(prefix[-1] + r)
        lo, hi = prefix[rank], prefix[rank + 1]
        ratio = hi - lo
        out = []
        for loc in locs:
            off = loc % S
            if not (lo <= off < hi):
                continue
            compact = (loc // S) * ratio + (off - lo)
            start = compact * h_write * self.head_dim
            row_start = compact * h_pool * self.head_dim
            out.append((loc, compact, start, row_start))
        return out

    def test_shard_shaped_pool_writes_outside_the_slots_own_row(self):
        # FALSIFIER for the old geometry. Not a boundary case: the drift is
        # linear in the slot id, so it is wrong for every slot but the first.
        bad = self._writes(
            range(0, 400),
            h_write=self.total_kv_heads,
            h_pool=self.per_rank_kv_heads,
            rank=0,
        )
        offenders = [(loc, c) for loc, c, start, row in bad if start != row]
        self.assertTrue(offenders)
        self.assertEqual(min(c for _, c in offenders), 1)

    def test_replicated_pool_writes_land_in_their_own_row(self):
        good = self._writes(
            range(0, 400),
            h_write=self.total_kv_heads,
            h_pool=self.total_kv_heads,
            rank=0,
        )
        self.assertTrue(good)
        for loc, compact, start, row in good:
            self.assertEqual(start, row, f"loc {loc} -> compact {compact}")

    def test_request_order_decides_which_rows_are_clobbered(self):
        # The occupancy pattern behind #343's "two identical greedy requests
        # split at token index 5": request A takes the first slots, request B
        # -- byte-identical prompt, same sampling -- starts where A stopped.
        # Under the shard-shaped pool the SAME logical token of the same
        # prompt is written to a different physical row depending on that
        # offset, so the second request reads back rows the first one's tail
        # overwrote. Under the fixed geometry the map is the identity on rows
        # and the offset is irrelevant.
        prompt_len = 24
        first = range(0, prompt_len)
        second = range(prompt_len, 2 * prompt_len)

        def rows_touched(locs, h_pool):
            span = self.total_kv_heads * self.head_dim
            row = h_pool * self.head_dim
            touched = set()
            for _loc, _c, start, _row in self._writes(
                locs, h_write=self.total_kv_heads, h_pool=h_pool, rank=0
            ):
                touched.update(range(start // row, (start + span - 1) // row + 1))
            return touched

        broken_a = rows_touched(first, self.per_rank_kv_heads)
        broken_b = rows_touched(second, self.per_rank_kv_heads)
        self.assertTrue(
            broken_a & broken_b,
            "the shard-shaped pool must show the two requests overlapping; "
            "that overlap IS the order dependence",
        )

        fixed_a = rows_touched(first, self.total_kv_heads)
        fixed_b = rows_touched(second, self.total_kv_heads)
        self.assertFalse(
            fixed_a & fixed_b,
            "with the replicated head count the owner rule is injective on "
            "rows again, so no request can reach another's rows",
        )


class TestDcpAccountingTotalSlots(unittest.TestCase):
    """The second #345 defect: ``SGLANG_UNEVEN_DCP=1`` with ``_WEIGHTED=0``
    died at the first idle check with ``available`` exactly ``dcp_size`` x
    ``total``, because the two scalars counted different spaces."""

    def test_off_the_lane_nothing_moves(self):
        self.assertEqual(
            dcp_accounting_total_slots(1000, 1000, token_sharded_dcp=False), 1000
        )
        self.assertEqual(
            dcp_accounting_total_slots(1000, 4000, token_sharded_dcp=False), 1000
        )

    def test_even_modulo_lane_reports_the_allocator_space(self):
        # The measured numbers from the failing boot.
        self.assertEqual(
            dcp_accounting_total_slots(28640, 57280, token_sharded_dcp=True), 57280
        )

    def test_weighted_lane_is_already_consistent(self):
        self.assertEqual(
            dcp_accounting_total_slots(61088, 61088, token_sharded_dcp=True), 61088
        )

    def test_missing_allocator_size_falls_back(self):
        self.assertEqual(
            dcp_accounting_total_slots(28640, None, token_sharded_dcp=True), 28640
        )


if __name__ == "__main__":
    unittest.main()
