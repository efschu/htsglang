"""Repro + fix proof for the dense-MoE uneven-TP activation alignment (#367).

The #283 boot proof of Qwen3.5-35B-A3B-GPTQ found this: the checkpoint's
`dynamic` map exempts `mtp.*`, so the #318 namespace probe correctly builds
the MTP draft DENSE -- and the dense draft's expert MoE then died in its first
forward with

    RuntimeError: Runtime check failed at elementwise/activation.cuh:168:
    hidden size must be divisible by vector size

`moe_uneven_tp_units` returned an ELEMENT-granular unit count whenever no
quantization constrained the intermediate dimension, so `--rank-tp-ratio 5,4`
cut the A3B's 512-wide expert intermediate into [284, 228]. Neither is a
multiple of the 16-element Blackwell activation vector, and neither is a
multiple of the 8-element Ampere one either -- the split is wrong on every
card in the rig, and it only says so after the weights are loaded and the
graphs are captured.

This is the #82/#289/#300/#316/#353 alignment family arriving on the one lane
that has no `weight_block_size` to anchor on. The quantized branches of
`moe_uneven_tp_units` already return coarser units (group 32, block 128 --
both multiples of 16); the unconstrained branch was the gap.

Fix, in two halves, both pinned here:

1. COARSEN where the planner splits: the unconstrained lane returns
   `intermediate // ACTIVATION_VEC_ELEMS` units, the same 16-element grain
   #82 established for the dense MLP. One grain for every rank, because the
   plan is built before anyone knows which rank lands on which arch -- and a
   multiple of 16 is a multiple of 8, so one plan is valid on a mixed rig.
2. REJECT LOUDLY where a foreign split arrives: `FusedMoE.__init__` calls
   `assert_activation_aligned_shards` on the expert intermediate, so a split
   no unit grain can fix (a hand-set SGLANG_UNEVEN_MOE_VECTOR, an
   intermediate size that is not a multiple of the vector) fails at
   construction naming the constraint instead of in the kernel mid-forward.

Pure functions, no GPU, no server.
"""

import unittest

from sglang.srt.distributed.utils import (
    ACTIVATION_VEC_ELEMS,
    assert_activation_aligned_shards,
    set_tp_partition_ratios,
    tp_partition_sizes,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import moe_uneven_tp_units
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

#: Qwen3.5-35B-A3B moe_intermediate_size, and the #283 vehicle's shard plan.
A3B_MOE_INTERMEDIATE = 512
T283_RATIO = [5, 4]
#: The rig's validated TP=3 auto vector (5090 + 2x3080).
TP3_RATIO = [30, 17, 17]
#: Ampere's narrower vector. A plan aligned to 16 must satisfy this too.
AMPERE_VEC_ELEMS = 8


class _Plan:
    """Install a ratio vector for the duration of a block, then clear it."""

    def __init__(self, ratios):
        self.ratios = ratios

    def __enter__(self):
        set_tp_partition_ratios(self.ratios)
        return self

    def __exit__(self, *exc):
        set_tp_partition_ratios(None)
        return False


class TestRepro(CustomTestCase):
    """The pre-fix geometry, kept falsifiable."""

    def test_element_granular_units_produce_the_kernel_fatal_shards(self):
        with _Plan(T283_RATIO):
            # What moe_uneven_tp_units used to return for an unquantized
            # expert: one unit per element.
            sizes = tp_partition_sizes(
                A3B_MOE_INTERMEDIATE, 2, A3B_MOE_INTERMEDIATE, "moe"
            )
        self.assertEqual(sizes, [284, 228])
        # Fatal on Blackwell (16) AND on Ampere (8) -- this split has no card
        # it is valid on, which is why the crash reproduced on a 3080 rank.
        self.assertTrue(any(s % ACTIVATION_VEC_ELEMS for s in sizes))
        self.assertTrue(any(s % AMPERE_VEC_ELEMS for s in sizes))

    def test_the_guard_would_have_caught_it_at_plan_time(self):
        with _Plan(T283_RATIO):
            with self.assertRaises(ValueError) as cm:
                assert_activation_aligned_shards(
                    A3B_MOE_INTERMEDIATE,
                    2,
                    A3B_MOE_INTERMEDIATE,
                    "moe",
                    what="MoE expert intermediate",
                )
        msg = str(cm.exception)
        self.assertIn("MoE expert intermediate", msg)
        self.assertIn("[284, 228]", msg)
        self.assertIn(str(ACTIVATION_VEC_ELEMS), msg)


class TestFix(CustomTestCase):
    def test_the_unconstrained_lane_coarsens_to_the_activation_vector(self):
        self.assertEqual(
            moe_uneven_tp_units(A3B_MOE_INTERMEDIATE, None),
            A3B_MOE_INTERMEDIATE // ACTIVATION_VEC_ELEMS,
        )

    def test_the_283_vehicle_now_splits_aligned(self):
        units = moe_uneven_tp_units(A3B_MOE_INTERMEDIATE, None)
        with _Plan(T283_RATIO):
            sizes = tp_partition_sizes(A3B_MOE_INTERMEDIATE, 2, units, "moe")
            assert_activation_aligned_shards(
                A3B_MOE_INTERMEDIATE, 2, units, "moe"
            )  # no raise
        self.assertEqual(sizes, [288, 224])
        self.assertEqual(sum(sizes), A3B_MOE_INTERMEDIATE)
        for s in sizes:
            self.assertEqual(s % ACTIVATION_VEC_ELEMS, 0)
            self.assertEqual(s % AMPERE_VEC_ELEMS, 0)

    def test_alignment_holds_across_the_rigs_real_plans(self):
        cases = [
            (A3B_MOE_INTERMEDIATE, TP3_RATIO),
            (A3B_MOE_INTERMEDIATE, [1, 2]),
            (704, T283_RATIO),  # Gemma-4-26B-A4B expert width
            (704, TP3_RATIO),
            (768, TP3_RATIO),
            (1536, [9, 5, 5]),
        ]
        for inter, ratio in cases:
            with self.subTest(intermediate=inter, ratio=ratio):
                units = moe_uneven_tp_units(inter, None)
                with _Plan(ratio):
                    sizes = tp_partition_sizes(inter, len(ratio), units, "moe")
                    assert_activation_aligned_shards(inter, len(ratio), units, "moe")
                self.assertEqual(sum(sizes), inter)
                for s in sizes:
                    self.assertEqual(s % ACTIVATION_VEC_ELEMS, 0)
                    self.assertGreater(s, 0)

    def test_units_are_still_fine_enough_to_track_the_ratio(self):
        """Coarsening must not flatten the plan: a 5:4 request has to stay
        recognisably 5:4, or the whole point of uneven TP is lost."""
        units = moe_uneven_tp_units(A3B_MOE_INTERMEDIATE, None)
        with _Plan(T283_RATIO):
            sizes = tp_partition_sizes(A3B_MOE_INTERMEDIATE, 2, units, "moe")
        got = sizes[0] / sizes[1]
        want = T283_RATIO[0] / T283_RATIO[1]
        self.assertLess(abs(got - want) / want, 0.05)


class TestNoGrainAvailable(CustomTestCase):
    """An intermediate the vector does not divide cannot be coarsened. It must
    stay element-granular AND be rejected -- never silently pass."""

    ODD = 500  # 500 % 16 = 4

    def test_units_stay_element_granular(self):
        self.assertEqual(moe_uneven_tp_units(self.ODD, None), self.ODD)

    def test_and_the_geometry_is_rejected_at_plan_time(self):
        units = moe_uneven_tp_units(self.ODD, None)
        with _Plan(T283_RATIO):
            with self.assertRaises(ValueError) as cm:
                assert_activation_aligned_shards(
                    self.ODD, 2, units, "moe", what="MoE expert intermediate"
                )
        self.assertIn("MoE expert intermediate", str(cm.exception))


class TestQuantizedLanesUnchanged(CustomTestCase):
    """Inertness: the branches that already had a grain keep their answer."""

    def test_block_quantized_experts(self):
        qc = _Cfg(weight_block_size=[128, 128])
        self.assertEqual(moe_uneven_tp_units(768, qc), 6)

    def test_group_quantized_experts(self):
        qc = _Cfg(group_size=32)
        self.assertEqual(moe_uneven_tp_units(A3B_MOE_INTERMEDIATE, qc), 16)

    def test_group_halving_still_applies(self):
        # Gemma-4-26B-A4B: 704 with AWQ group 128 -> halves to 64.
        qc = _Cfg(group_size=128)
        self.assertEqual(moe_uneven_tp_units(704, qc), 11)

    def test_a_group_that_halves_below_32_falls_to_the_vector_grain(self):
        # 272 with group 64: 64 -> 32 -> 16, and the loop stops below the
        # 32-element floor MoeWNA16Method enforces. No usable group grain is
        # left, so the activation vector is what binds -- the #367 lane, not
        # element granularity.
        qc = _Cfg(group_size=64)
        self.assertEqual(moe_uneven_tp_units(272, qc), 272 // ACTIVATION_VEC_ELEMS)

    def test_every_quantized_grain_is_already_vector_aligned(self):
        """The reason only the unconstrained lane needed fixing."""
        for qc, inter in (
            (_Cfg(weight_block_size=[128, 128]), 768),
            (_Cfg(group_size=32), A3B_MOE_INTERMEDIATE),
            (_Cfg(group_size=128), 704),
        ):
            with self.subTest(inter=inter):
                units = moe_uneven_tp_units(inter, qc)
                self.assertEqual((inter // units) % ACTIVATION_VEC_ELEMS, 0)


class TestDefaultPathUnchanged(CustomTestCase):
    def test_no_plan_means_the_units_are_never_consulted(self):
        set_tp_partition_ratios(None)
        try:
            units = moe_uneven_tp_units(A3B_MOE_INTERMEDIATE, None)
            # The even split is what FusedMoE takes without a plan, and it
            # does not go through the unit machinery at all.
            self.assertEqual(A3B_MOE_INTERMEDIATE % 2, 0)
            self.assertEqual(A3B_MOE_INTERMEDIATE // 2, 256)
            # The guard is a no-op without an installed plan.
            assert_activation_aligned_shards(A3B_MOE_INTERMEDIATE, 2, units, "moe")
            assert_activation_aligned_shards(500, 2, 500, "moe")
        finally:
            set_tp_partition_ratios(None)

    def test_a_plan_of_a_different_size_does_not_apply(self):
        with _Plan(TP3_RATIO):  # 3 ranks
            assert_activation_aligned_shards(500, 2, 500, "moe")  # no raise


class _Cfg:
    """Minimal quant-config stand-in: only the attributes the unit function
    reads."""

    def __init__(self, weight_block_size=None, group_size=None):
        self.weight_block_size = weight_block_size
        self.group_size = group_size
        self.target_scheme_map = None


if __name__ == "__main__":
    unittest.main()
