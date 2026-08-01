"""Dual-group runtime (#121) slice A: nesting algebra and local collectives.

Every test here is a claim about why the shared card holds the full weights
exactly ONCE, or about the collectives that replace NCCL inside a lane whose
two ranks are the same process.
"""

import unittest

import torch

from sglang.srt.distributed.dual_group import (
    DUPLICATED,
    NESTED,
    SHARED,
    NestedGroupPlan,
    NestingProbe,
    check_nesting,
    derive_nested_plan,
    format_vram_posts,
    lane_vram_posts,
    local_column_gather,
    local_row_reduce,
    local_row_split,
    nesting_failures,
    transformer_nesting_probes,
)
from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    partition_units,
    scoped_tp_partition_ratios,
    set_tp_partition_ratios,
    tp_plan_active,
)


class TestPlanShape(unittest.TestCase):
    def test_rig_case(self):
        plan = derive_nested_plan([6, 1, 1])
        self.assertEqual(plan.fast_ratio, (6, 2))
        self.assertEqual(plan.segments, ((0,), (1, 2)))
        self.assertEqual(plan.shared_fast_ranks, (0,))
        self.assertEqual(plan.complement_fast_ranks, (1,))
        self.assertEqual(plan.shared_big_rank(0), 0)
        self.assertIsNone(plan.shared_big_rank(1))

    def test_last_rank_may_be_the_shared_one(self):
        plan = derive_nested_plan([1, 1, 6], shared_big_rank=2)
        self.assertEqual(plan.segments, ((0, 1), (2,)))
        self.assertEqual(plan.fast_ratio, (2, 6))

    def test_middle_rank_is_rejected_with_the_reason(self):
        with self.assertRaises(ValueError) as ctx:
            derive_nested_plan([1, 6, 1], shared_big_rank=1)
        self.assertIn("not contiguous", str(ctx.exception))

    def test_noncontiguous_segment_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            NestedGroupPlan(big_ratio=(6, 1, 1), segments=((0, 2), (1,)))
        self.assertIn("contiguous run", str(ctx.exception))

    def test_segments_must_cover_every_big_rank(self):
        with self.assertRaises(ValueError) as ctx:
            NestedGroupPlan(big_ratio=(6, 1, 1), segments=((0,), (1,)))
        self.assertIn("must partition", str(ctx.exception))

    def test_family_vectors_get_their_own_segment_sums(self):
        plan = NestedGroupPlan(
            big_ratio=(6, 1, 1),
            segments=((0,), (1, 2)),
            family_ratios=(("mlp", (4, 2, 2)),),
        )
        self.assertEqual(plan.fast_ratio_for("mlp"), (4, 4))
        self.assertEqual(plan.fast_ratio_for(None), (6, 2))
        self.assertEqual(plan.fast_ratio_for("moe"), (6, 2))  # falls back
        self.assertEqual(plan.fast_family_ratios, (("mlp", (4, 4)),))

    def test_family_vector_length_is_checked(self):
        with self.assertRaises(ValueError) as ctx:
            NestedGroupPlan(
                big_ratio=(6, 1, 1),
                segments=((0,), (1, 2)),
                family_ratios=(("mlp", (4, 2)),),
            )
        self.assertIn("2 entries", str(ctx.exception))


class TestNesting(unittest.TestCase):
    """The property the whole feature rests on: the shared segment occupies
    the SAME unit range in both groups, so its bytes are the same bytes."""

    def test_rig_case_nests(self):
        plan = derive_nested_plan([6, 1, 1])
        probes = [
            NestingProbe(what="weights", units=8),
            NestingProbe(what="MLP", units=96, family="mlp"),
            NestingProbe(what="kv heads", units=8),
        ]
        self.assertEqual(nesting_failures(plan, probes), [])
        check_nesting(plan, probes)

    def test_shared_prefix_is_the_identical_unit_range(self):
        """Equal sizes on contiguous ordered segments == equal prefix sums."""
        plan = derive_nested_plan([6, 1, 1])
        for units in range(3, 400):
            big = partition_units(units, plan.big_ratio)
            fast = partition_units(units, plan.fast_ratio)
            if nesting_failures(plan, [NestingProbe("u", units)]):
                continue
            # rank 0's range is [0, big[0]) in BIG and [0, fast[0]) in FAST
            self.assertEqual(big[0], fast[0], units)
            self.assertEqual(sum(big[1:]), fast[1], units)

    def test_a_remainder_bump_breaks_nesting_and_is_reported(self):
        """The rounding really can diverge -- this is why it is checked."""
        plan = derive_nested_plan([1, 1, 1, 3])
        broken = [
            u
            for u in range(4, 200)
            if nesting_failures(plan, [NestingProbe("units", u)])
        ]
        self.assertTrue(
            broken,
            "expected at least one unit count where largest-remainder "
            "rounding gives the shared segment a different size in the two "
            "groups; if this ever becomes empty the check is not the reason "
            "the feature is safe",
        )
        msg = nesting_failures(plan, [NestingProbe("units", broken[0])])[0]
        self.assertIn("cannot be shared", msg)
        self.assertIn("FAST rank", msg)

    def test_check_nesting_names_plan_and_fix(self):
        plan = derive_nested_plan([1, 1, 1, 3])
        broken = next(
            u
            for u in range(4, 200)
            if nesting_failures(plan, [NestingProbe("units", u)])
        )
        with self.assertRaises(ValueError) as ctx:
            check_nesting(plan, [NestingProbe("units", broken)])
        text = str(ctx.exception)
        self.assertIn("not nested", text)
        self.assertIn("FAST ratio", text)
        self.assertIn("Fix:", text)

    def test_the_nesting_refusal_is_in_english(self):
        """#381: this message said "the Verband's weight bytes".

        The #295/#358 sweeps translated the barlink and bar1ep strands and did
        not reach the dual-group planner, so one German word survived in a
        message an operator reads when their ratio pair is rejected. It is
        replaced by this file's own English term for the same thing -- the BIG
        group -- so the refusal and the code that raises it use one vocabulary.
        """
        plan = derive_nested_plan([1, 1, 1, 3])
        broken = next(
            u
            for u in range(4, 200)
            if nesting_failures(plan, [NestingProbe("units", u)])
        )
        with self.assertRaises(ValueError) as ctx:
            check_nesting(plan, [NestingProbe("units", broken)])
        text = str(ctx.exception)
        self.assertNotIn("Verband", text)
        self.assertIn("BIG group's weight bytes", text)

    def test_differing_geometry_is_undefined_not_merely_violated(self):
        plan = derive_nested_plan([6, 1, 1])
        probe = NestingProbe(
            what="attention q heads",
            units=4,
            groups=2,
            fast_units=2,
            fast_groups=None,
            fast_groups_set=True,
        )
        msg = nesting_failures(plan, [probe])[0]
        self.assertIn("DIFFERENT geometry", msg)
        self.assertIn("REPLICATED-KV", msg)

    def test_an_exactly_dividing_ratio_always_nests(self):
        plan = derive_nested_plan([6, 1, 1])
        for k in range(1, 40):
            probe = NestingProbe(what="exact", units=8 * k)
            self.assertEqual(nesting_failures(plan, [probe]), [])


class TestModelProbes(unittest.TestCase):
    def test_dense_model_probes_nest_on_the_rig_plan(self):
        plan = derive_nested_plan([6, 1, 1])
        probes = transformer_nesting_probes(
            plan,
            num_attention_heads=32,
            num_kv_heads=8,
            intermediate_size=6144,
            linear_attn_units=16,
        )
        self.assertTrue(any(p.family == "mlp" for p in probes))
        check_nesting(plan, probes)

    def test_replicated_kv_threshold_between_the_groups_is_caught(self):
        """kv=2: replicated-kv at TP=3 (2 < 3), normal at TP=2 (2 == 2)."""
        plan = derive_nested_plan([6, 1, 1])
        probes = transformer_nesting_probes(
            plan, num_attention_heads=16, num_kv_heads=2
        )
        failures = nesting_failures(plan, probes)
        self.assertTrue(failures)
        self.assertIn("DIFFERENT geometry", failures[0])

    def test_probe_builder_leaves_the_process_plan_untouched(self):
        set_tp_partition_ratios([3, 2, 1], {"mlp": [4, 4, 4]})
        try:
            plan = derive_nested_plan([6, 1, 1])
            transformer_nesting_probes(plan, num_attention_heads=32, num_kv_heads=8)
            self.assertEqual(get_tp_partition_ratios(), [3, 2, 1])
            self.assertEqual(get_tp_partition_ratios("mlp"), [4, 4, 4])
        finally:
            set_tp_partition_ratios(None)


class TestScopedRatios(unittest.TestCase):
    """The missing primitive: a second group must install its own vector and
    hand the process back unchanged."""

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_restores_base_and_families(self):
        set_tp_partition_ratios([6, 1, 1], {"mlp": [5, 2, 1]})
        with scoped_tp_partition_ratios([6, 2], {"mlp": [5, 3]}):
            self.assertEqual(get_tp_partition_ratios(), [6, 2])
            self.assertEqual(get_tp_partition_ratios("mlp"), [5, 3])
        self.assertEqual(get_tp_partition_ratios(), [6, 1, 1])
        self.assertEqual(get_tp_partition_ratios("mlp"), [5, 2, 1])

    def test_restores_after_an_exception(self):
        set_tp_partition_ratios([6, 1, 1])
        with self.assertRaises(RuntimeError):
            with scoped_tp_partition_ratios([6, 2]):
                raise RuntimeError("boom")
        self.assertEqual(get_tp_partition_ratios(), [6, 1, 1])

    def test_nests(self):
        set_tp_partition_ratios([6, 1, 1])
        with scoped_tp_partition_ratios([6, 2]):
            with scoped_tp_partition_ratios([4, 4]):
                self.assertEqual(get_tp_partition_ratios(), [4, 4])
            self.assertEqual(get_tp_partition_ratios(), [6, 2])
        self.assertEqual(get_tp_partition_ratios(), [6, 1, 1])

    def test_this_is_why_it_is_needed(self):
        """Without a scope the 3-entry vector silently does NOT apply to a
        2-rank group -- it falls back to the even split, which would load the
        wrong complement units without raising anything."""
        set_tp_partition_ratios([6, 1, 1])
        self.assertTrue(tp_plan_active(3))
        self.assertFalse(tp_plan_active(2))  # silent even-split fallback
        with scoped_tp_partition_ratios([6, 2]):
            self.assertTrue(tp_plan_active(2))

    def test_none_clears_and_restores(self):
        set_tp_partition_ratios([6, 1, 1])
        with scoped_tp_partition_ratios(None):
            self.assertIsNone(get_tp_partition_ratios())
        self.assertEqual(get_tp_partition_ratios(), [6, 1, 1])


class TestLocalCollectives(unittest.TestCase):
    """The FAST group's collectives are tensor ops because both shards are in
    one address space. These tests pin down exactly how exact they are."""

    def test_column_gather_is_bit_identical_to_an_all_gather(self):
        """The gather itself is pure data movement: given the SAME per-rank
        outputs, it produces exactly what a real all-gather would."""
        torch.manual_seed(0)
        a = torch.randn(7, 36, dtype=torch.float32)
        b = torch.randn(7, 12, dtype=torch.float32)
        got = local_column_gather([a, b], [[36], [12]])
        self.assertTrue(torch.equal(got, torch.cat([a, b], dim=-1)))

    def test_column_gather_does_not_reproduce_a_monolithic_gemm(self):
        """Measured, not assumed: even the column path is not bit-identical
        to one wide GEMM. Splitting the output dimension changes the kernel's
        blocking, so the per-element accumulation over k differs. Whatever
        the byte gate compares, it cannot be 'lane == one big matmul'."""
        torch.manual_seed(0)
        x = torch.randn(7, 32, dtype=torch.float32)
        w = torch.randn(48, 32, dtype=torch.float32)
        got = local_column_gather([x @ w[:36].t(), x @ w[36:].t()], [[36], [12]])
        torch.testing.assert_close(got, x @ w.t(), rtol=1e-5, atol=1e-5)

    def test_column_gather_regroups_sub_outputs(self):
        """cat of whole rank slices would give [q0,k0,v0,q1,k1,v1]; the
        canonical layout every consumer expects is [q_all, k_all, v_all]."""
        a = torch.tensor([[1.0, 2.0, 10.0, 100.0]])  # q=2, k=1, v=1
        b = torch.tensor([[3.0, 20.0, 200.0]])  # q=1, k=1, v=1
        got = local_column_gather([a, b], [[2, 1, 1], [1, 1, 1]])
        self.assertTrue(
            torch.equal(got, torch.tensor([[1.0, 2.0, 3.0, 10.0, 20.0, 100.0, 200.0]]))
        )

    def test_column_gather_rejects_mismatched_sub_sizes(self):
        a = torch.zeros(1, 4)
        with self.assertRaises(ValueError) as ctx:
            local_column_gather([a, a], [[2, 2], [3, 2]])
        self.assertIn("sum to", str(ctx.exception))

    def test_column_gather_rejects_differing_sub_output_counts(self):
        a = torch.zeros(1, 4)
        with self.assertRaises(ValueError) as ctx:
            local_column_gather([a, a], [[2, 2], [4]])
        self.assertIn("same number", str(ctx.exception))

    def test_row_split_then_reduce_matches_a_two_rank_all_reduce(self):
        torch.manual_seed(0)
        x = torch.randn(5, 40, dtype=torch.float32)
        w = torch.randn(16, 40, dtype=torch.float32)
        parts = local_row_split(x, [30, 10])
        self.assertEqual([p.shape[-1] for p in parts], [30, 10])
        got = local_row_reduce([parts[0] @ w[:, :30].t(), parts[1] @ w[:, 30:].t()])
        # A real 2-rank all-reduce is exactly this one addition.
        ref = (x[:, :30] @ w[:, :30].t()) + (x[:, 30:] @ w[:, 30:].t())
        self.assertTrue(torch.equal(got, ref))

    def test_row_reduce_against_a_monolith_is_close_not_equal_by_contract(self):
        """Documented tolerance, predicted before the byte gate rather than
        conceded after it: split-and-add is a different accumulation order
        than one GEMM over the full k axis. Sometimes the orders coincide and
        the results ARE equal -- that is luck, not a guarantee, so the
        contract is closeness in both directions and the test asserts only
        what is guaranteed."""
        torch.manual_seed(1)
        x = torch.randn(64, 512, dtype=torch.float32)
        w = torch.randn(64, 512, dtype=torch.float32)
        parts = local_row_split(x, [384, 128])
        got = local_row_reduce([parts[0] @ w[:, :384].t(), parts[1] @ w[:, 384:].t()])
        torch.testing.assert_close(got, x @ w.t(), rtol=1e-4, atol=1e-3)

    def test_row_split_hands_out_contiguous_slices(self):
        """The contract a stride-blind kernel depends on (#274, round 5).

        On a real rank the row-parallel input is that rank's own densely packed
        activation. The shell's slice of a FULL-width tensor is the only place
        that input can arrive strided, and GGUF's mat-VEC kernel -- which
        ``fused_mul_mat_gguf`` selects for <= 8 rows, i.e. exactly a lane verify
        of K+1 candidates -- reads the activation with an assumed contiguous row
        stride. Under a view it lands on row 0 and misses every row after it,
        which is precisely how the lane's rows >= 1 broke while row 0 stayed
        byte-exact. Values must of course be unchanged: this asserts both.
        """
        x = torch.arange(4 * 40, dtype=torch.float32).reshape(4, 40)
        parts = local_row_split(x, [30, 10])
        for i, p in enumerate(parts):
            self.assertTrue(p.is_contiguous(), f"part {i} is not contiguous")
        self.assertTrue(torch.equal(parts[0], x[:, :30]))
        self.assertTrue(torch.equal(parts[1], x[:, 30:]))

    def test_row_split_does_not_copy_when_the_slice_is_already_dense(self):
        """One row, or one part, is contiguous already -- the byte-green decode
        path must stay literally the same tensor, not a copy of it."""
        x = torch.randn(4, 40)
        # One part: the slice is the whole tensor, so no copy is made.
        self.assertEqual(local_row_split(x, [40])[0].data_ptr(), x.data_ptr())
        # One row: every slice is dense already, so none is copied.
        one = torch.randn(1, 40)
        parts = local_row_split(one, [30, 10])
        self.assertEqual(parts[0].data_ptr(), one.data_ptr())
        self.assertEqual(parts[1].data_ptr(), one.data_ptr() + 30 * one.element_size())

    def test_row_split_rejects_a_width_mismatch(self):
        with self.assertRaises(ValueError) as ctx:
            local_row_split(torch.zeros(2, 10), [6, 3])
        self.assertIn("sum to", str(ctx.exception))

    def test_row_reduce_rejects_empty(self):
        with self.assertRaises(ValueError):
            local_row_reduce([])


class TestVramPosts(unittest.TestCase):
    def test_full_weights_exactly_once(self):
        plan = derive_nested_plan([6, 1, 1])
        posts = lane_vram_posts(
            plan,
            shared_fast_rank=0,
            total_weight_mib=13210,
            weight_units=8,
            lane_kv_mib=800,
            lane_state_mib=120,
            lane_scratch_mib=1000,
        )
        by_status = {}
        for p in posts:
            by_status.setdefault(p.status, []).append(p)
        shared = next(p for p in posts if p.status == SHARED and p.mib > 0)
        nested = next(p for p in posts if p.status == NESTED)
        self.assertEqual(shared.mib + nested.mib, 13210)
        self.assertAlmostEqual(shared.mib / 13210, 6 / 8, places=3)
        self.assertTrue(by_status[DUPLICATED])

    def test_shell_tree_and_context_cost_nothing(self):
        plan = derive_nested_plan([6, 1, 1])
        posts = lane_vram_posts(
            plan,
            shared_fast_rank=0,
            total_weight_mib=13210,
            weight_units=8,
            lane_kv_mib=0,
            lane_state_mib=0,
            lane_scratch_mib=0,
        )
        zero_shared = [p for p in posts if p.status == SHARED and p.mib == 0]
        self.assertEqual(len(zero_shared), 2)  # meta shell tree, CUDA context

    def test_complement_rank_is_not_a_shared_rank(self):
        plan = derive_nested_plan([6, 1, 1])
        with self.assertRaises(ValueError) as ctx:
            lane_vram_posts(
                plan,
                shared_fast_rank=1,
                total_weight_mib=100,
                weight_units=8,
                lane_kv_mib=0,
                lane_state_mib=0,
                lane_scratch_mib=0,
            )
        self.assertIn("complement", str(ctx.exception))

    def test_format_lists_every_item_and_the_added_sum(self):
        plan = derive_nested_plan([6, 1, 1])
        posts = lane_vram_posts(
            plan,
            shared_fast_rank=0,
            total_weight_mib=8000,
            weight_units=8,
            lane_kv_mib=500,
            lane_state_mib=100,
            lane_scratch_mib=400,
        )
        text = format_vram_posts(posts, "GPU 0 (RTX 5090)")
        self.assertIn("RTX 5090", text)
        for p in posts:
            self.assertIn(p.name, text)
        # 2000 complement + 500 + 100 + 400
        self.assertIn("3000 MiB", text)


if __name__ == "__main__":
    unittest.main()
