"""INT8 W8A8 x uneven TP: the kernel's alignment, exposed and enforced (#353).

`sgl_kernel.int8_scaled_mm` asserts `K % 16 == 0` on both operands and
`N % 8 == 0` on the output (`sgl-kernel/csrc/gemm/int8_gemm_kernel.cu`), and it
asserts them in CUDA, at the first forward, after the whole checkpoint has been
loaded. Channel-strategy INT8 exposed no `weight_block_size` at all, so
`_quant_block_aligned_units` / `moe_uneven_tp_units` had nothing to coarsen and
an uneven `--rank-tp-ratio` split landed wherever the raw unit family put it.
The #327 boot passed only because its MLP split happened to be 16-aligned.

Sixth member of the alignment family (#37 / #86 / #289 / #300 / #316 / #323),
and it needs BOTH halves of that family's answer, because the two halves cover
different unit families:

* the block (`#289`/`#323` shape) coarsens element-granular families -- MoE
  expert intermediates -- so their shards become kernel-valid;
* the shape guard (`#300`/`#316` shape) names the shards that CANNOT be
  coarsened, because their unit family is COUPLED across layers of different
  widths: Qwen3.5's GDN hands one `gdn_tp_units` to `in_proj_qkvz`
  (1024-element units) and `in_proj_ba` (3-element units), and coarsening the
  latter locally would mis-shard it against the former.

Card-proven 2026-07-31 on an RTX 5090: at N = 42 and N = 30 the guard raises
and `int8_scaled_mm` independently answers `mat_b.size(1) must be multiple of 8
for memory alignment`; at N = 24 and N = 10432 the guard passes and the kernel
runs. The numbers below are that constellation.

Pure functions, no GPU, no server.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import math
import unittest

from sglang.srt.distributed.utils import (
    set_tp_partition_ratios,
    tp_partition_sizes,
)
from sglang.srt.layers.linear import _quant_block_aligned_units
from sglang.srt.layers.moe.fused_moe_triton.layer import moe_uneven_tp_units
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.w8a8_int8 import (
    INT8_SCALED_MM_ALIGN_K,
    INT8_SCALED_MM_ALIGN_N,
    W8A8Int8Config,
    int8_w8a8_uneven_tp_block,
    verify_int8_scaled_mm_supports_shape,
)
from sglang.test.test_utils import CustomTestCase

# Qwen3.6-27B-INT8-W8A8 (Avesed), the #327 A/B vehicle.
HIDDEN = 5120
INTERMEDIATE = 17408
# GDN: 16 k heads, 48 v heads, 128-element heads.
NUM_K_HEADS, NUM_V_HEADS = 16, 48
VALUE_DIM = 6144
QKVZ_PART = 2048  # output_sizes[0] of in_proj_qkvz (key_dim)
BA_PART = NUM_V_HEADS  # output_sizes[0] of in_proj_ba: 48 per part, b and a
#: The vector --rank-tp-ratio auto-performance installed for the #327 boot.
S327_WEIGHTS = [29607, 17780, 17780]
#: A small MoE expert intermediate (Qwen3.5-35B-A3B class).
MOE_INTERMEDIATE = 512


def _weights(num_bits, qtype, strategy="channel", group_size=None):
    class _Args:
        pass

    a = _Args()
    a.num_bits = num_bits
    a.type = qtype
    a.strategy = strategy
    a.group_size = group_size
    a.block_structure = None
    a.symmetric = True
    a.dynamic = False
    return a


def _acts(num_bits, qtype="int", strategy="token"):
    if num_bits is None:
        return None
    a = _weights(num_bits, qtype, strategy)
    a.dynamic = True
    return a


def _ct_config(scheme_map):
    return CompressedTensorsConfig(
        target_scheme_map=scheme_map,
        ignore=[],
        quant_format="int-quantized",
        sparsity_scheme_map={},
        sparsity_ignore_list=[],
    )


def _int8_w8a8_ct():
    """The Avesed checkpoint's config_groups, as CompressedTensorsConfig holds
    them: int8 channel weights, dynamic per-token int8 activations."""
    return _ct_config(
        {"Linear": {"weights": _weights(8, "int"), "input_activations": _acts(8)}}
    )


def _int8_w8a16_ct():
    """pack-quantized weight-only INT8: no input_activations at all. Runs
    through Marlin wNa16, not through int8_scaled_mm."""
    return _ct_config(
        {
            "Linear": {
                "weights": _weights(8, "int", strategy="group", group_size=128),
                "input_activations": None,
            }
        }
    )


def _fp8_block_ct():
    weights = _weights(8, "float", strategy="block")
    weights.block_structure = [128, 128]
    return _ct_config(
        {"Linear": {"weights": weights, "input_activations": _acts(8, "float")}}
    )


def _int4_group_ct():
    return _ct_config(
        {
            "Linear": {
                "weights": _weights(4, "int", strategy="group", group_size=128),
                "input_activations": None,
            }
        }
    )


def _mlp_units(intermediate, quant_config):
    """Mirrors the derivation in sglang.srt.models.qwen2_moe.Qwen2MoeMLP."""
    units = intermediate // math.gcd(intermediate, 16)
    return _quant_block_aligned_units(intermediate, units, quant_config, 1)


class TestTheBlockItself(CustomTestCase):
    def test_it_is_one_shared_value_folding_both_kernel_alignments(self):
        block = int8_w8a8_uneven_tp_block()
        self.assertEqual(block, [16, 16])
        self.assertEqual(
            block[0], math.lcm(INT8_SCALED_MM_ALIGN_N, INT8_SCALED_MM_ALIGN_K)
        )
        # A per-dim [8, 16] -- what ANALYSE_319 sec. 2d sketched -- would
        # coarsen gate_up's OUTPUT and down's INPUT of the SAME intermediate
        # dimension differently. Both dims must carry one value.
        self.assertEqual(block[0], block[1])
        self.assertEqual(block[0] % INT8_SCALED_MM_ALIGN_N, 0)
        self.assertEqual(block[0] % INT8_SCALED_MM_ALIGN_K, 0)

    def test_both_int8_config_classes_expose_it(self):
        self.assertEqual(W8A8Int8Config({}).weight_block_size, [16, 16])
        self.assertEqual(_int8_w8a8_ct().weight_block_size, [16, 16])


class TestBlockDetection(CustomTestCase):
    """Only a genuine W8A8 scheme picks up int8_scaled_mm's alignment."""

    def test_weight_only_int8_is_not_w8a8(self):
        # pack-quantized W8A16 runs through Marlin; its own group block (and
        # the Marlin K tile folded into it) is the right answer, unchanged.
        cfg = _int8_w8a16_ct()
        self.assertIsNone(
            CompressedTensorsConfig._int8_w8a8_block(cfg.target_scheme_map)
        )
        self.assertEqual(cfg.weight_block_size, [128, 128])

    def test_fp8_block_structure_still_wins(self):
        self.assertEqual(_fp8_block_ct().weight_block_size, [128, 128])

    def test_int4_group_scheme_is_untouched(self):
        cfg = _int4_group_ct()
        self.assertIsNone(
            CompressedTensorsConfig._int8_w8a8_block(cfg.target_scheme_map)
        )
        self.assertEqual(cfg.weight_block_size, [128, 128])

    def test_a_mixed_checkpoint_satisfies_both(self):
        # A config that groups some modules at int4 and others at int8 W8A8
        # must land on a block valid for BOTH kernels, not on whichever one
        # the dict happens to yield first.
        cfg = _ct_config(
            {
                "Linear": {
                    "weights": _weights(4, "int", strategy="group", group_size=128),
                    "input_activations": None,
                },
                "re:.*mlp.*": {
                    "weights": _weights(8, "int"),
                    "input_activations": _acts(8),
                },
            }
        )
        self.assertEqual(cfg.weight_block_size, [math.lcm(128, 16)] * 2)


class TestTheSplitThatWouldHaveAborted(CustomTestCase):
    """GDN in_proj_ba under the #327 shard plan: 48 per part over 16 k-head
    units gives ranks 7/5/4 units -> per-part 21/15/12 -> merged N 42/30/24.

    42 % 8 == 2 and 30 % 8 == 6, so the first two ranks abort inside CUTLASS
    mid-forward. This family CANNOT be coarsened: `gdn_tp_units` is one number
    shared by in_proj_qkvz, in_proj_ba, out_proj, conv1d and the state cache,
    and re-deriving it per layer breaks the coupling. So the shard is rejected
    by name instead -- #300's rule, bad shards stay loud.
    """

    def setUp(self):
        set_tp_partition_ratios(S327_WEIGHTS)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_the_geometry_is_the_one_that_aborts(self):
        parts = tp_partition_sizes(BA_PART, 3, units=NUM_K_HEADS)
        self.assertEqual(parts, [21, 15, 12])
        merged = [2 * p for p in parts]
        self.assertEqual(merged, [42, 30, 24])
        self.assertEqual([n % INT8_SCALED_MM_ALIGN_N for n in merged], [2, 6, 0])

    def test_the_guard_names_the_two_bad_shards(self):
        for n, bad in ((42, True), (30, True), (24, False)):
            with self.subTest(N=n):
                if bad:
                    with self.assertRaises(ValueError) as ctx:
                        verify_int8_scaled_mm_supports_shape(
                            n, HIDDEN, "linear_attn.in_proj_ba"
                        )
                    msg = str(ctx.exception)
                    # The exact numbers, so the log alone identifies the shard.
                    self.assertIn("linear_attn.in_proj_ba", msg)
                    self.assertIn(f"output_size_per_partition = {n}", msg)
                    self.assertIn("multiple of 8", msg)
                    self.assertIn("int8_scaled_mm", msg)
                else:
                    verify_int8_scaled_mm_supports_shape(
                        n, HIDDEN, "linear_attn.in_proj_ba"
                    )

    def test_the_guard_catches_the_input_dim_too(self):
        # A row-parallel shard whose K misses 16: the other half of the
        # kernel's rule, and the half a merged-output check would not see.
        with self.assertRaises(ValueError) as ctx:
            verify_int8_scaled_mm_supports_shape(HIDDEN, 4744, "mlp.down_proj")
        msg = str(ctx.exception)
        self.assertIn("input_size_per_partition = 4744", msg)
        self.assertIn("multiple of 16", msg)

    def test_the_327_split_that_passed_by_accident_still_passes(self):
        # The MLP shards of the #327 boot. 16-element activation units already
        # satisfy both halves of the rule, which is exactly why that boot did
        # not hit the gap -- the guard must not turn a working split into an
        # error.
        units = _mlp_units(INTERMEDIATE, _int8_w8a8_ct())
        sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
        self.assertEqual(sum(sizes), INTERMEDIATE)
        for size in sizes:
            with self.subTest(shard=size):
                verify_int8_scaled_mm_supports_shape(
                    2 * size, HIDDEN, "mlp.gate_up_proj"
                )
                verify_int8_scaled_mm_supports_shape(HIDDEN, size, "mlp.down_proj")


class TestElementGranularFamiliesNowCoarsen(CustomTestCase):
    """What the block actually buys: a family with one element per unit."""

    def setUp(self):
        set_tp_partition_ratios(S327_WEIGHTS)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_moe_experts_were_element_granular_and_now_are_not(self):
        self.assertEqual(moe_uneven_tp_units(MOE_INTERMEDIATE, None), MOE_INTERMEDIATE)
        self.assertEqual(moe_uneven_tp_units(MOE_INTERMEDIATE, _int8_w8a8_ct()), 32)
        self.assertEqual(moe_uneven_tp_units(MOE_INTERMEDIATE, W8A8Int8Config({})), 32)

    def test_the_expert_shards_go_from_invalid_to_valid(self):
        before = tp_partition_sizes(
            MOE_INTERMEDIATE, 3, units=moe_uneven_tp_units(MOE_INTERMEDIATE, None)
        )
        self.assertEqual(before, [232, 140, 140])
        # w1/w3 are merged, so N is 2*shard; w2's K is the shard itself. Here
        # every N is even enough but no K is a multiple of 16 -- w2 aborts on
        # all three ranks.
        self.assertEqual([s % INT8_SCALED_MM_ALIGN_K for s in before], [8, 12, 12])
        for shard in before:
            with self.subTest(bad_shard=shard):
                with self.assertRaisesRegex(ValueError, "multiple of 16"):
                    verify_int8_scaled_mm_supports_shape(HIDDEN, shard, "experts.w2")

        cfg = _int8_w8a8_ct()
        after = tp_partition_sizes(
            MOE_INTERMEDIATE, 3, units=moe_uneven_tp_units(MOE_INTERMEDIATE, cfg)
        )
        self.assertEqual(after, [224, 144, 144])
        self.assertEqual(sum(after), MOE_INTERMEDIATE)
        for shard in after:
            with self.subTest(shard=shard):
                verify_int8_scaled_mm_supports_shape(2 * shard, HIDDEN, "experts.w13")
                verify_int8_scaled_mm_supports_shape(HIDDEN, shard, "experts.w2")
        # Still strictly uneven: the strong rank keeps the largest shard.
        self.assertGreater(after[0], after[1])


class TestCoupledAndHeadGranularFamiliesAreUnchanged(CustomTestCase):
    """A 16-wide block is an order of magnitude finer than a real quant block,
    so the families it must NOT disturb are checked explicitly."""

    def test_gdn_and_attention_units_match_the_no_block_answer(self):
        cfg = _int8_w8a8_ct()
        for total, units, idx, label in (
            (VALUE_DIM, NUM_K_HEADS, 1, "gdn_tp_units basis"),
            (QKVZ_PART, NUM_K_HEADS, 0, "in_proj_qkvz"),
            (VALUE_DIM, NUM_K_HEADS, 1, "out_proj"),
        ):
            with self.subTest(layer=label):
                self.assertEqual(
                    _quant_block_aligned_units(total, units, cfg, idx),
                    _quant_block_aligned_units(total, units, None, idx),
                )

    def test_the_dense_mlp_family_is_unchanged(self):
        # 16-element activation units already divide the block, so the dense
        # MLP split is byte-identical with and without it. The #327 boot's
        # "passed by accident" is precisely this.
        self.assertEqual(
            _mlp_units(INTERMEDIATE, _int8_w8a8_ct()),
            _mlp_units(INTERMEDIATE, None),
        )

    def test_even_tp_is_untouched(self):
        # tp_partition_sizes ignores units entirely without an installed ratio
        # plan, so no default-path boot can see this change at all.
        set_tp_partition_ratios(None)
        for units in (NUM_K_HEADS, 1, None):
            with self.subTest(units=units):
                self.assertEqual(tp_partition_sizes(BA_PART, 2, units=units), [24, 24])


if __name__ == "__main__":
    unittest.main()
