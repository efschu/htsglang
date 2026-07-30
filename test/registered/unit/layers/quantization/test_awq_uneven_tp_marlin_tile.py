"""Repro + inertness proof for the AWQ x uneven-TP Marlin tile fix (task #289).

The s03 battery boot of Huihui-Qwen3.6-27B-abliterated-AWQ-MTP (the F16-head
arm) died at the first dense MLP with

    Weight output_size_per_partition = 9504 is not divisible by
    min_thread_n = 64

Root cause: `AWQConfig` / `AWQMarlinConfig` never exposed a
`weight_block_size`, so `_quant_block_aligned_units` had nothing to coarsen
with and Qwen2MoeMLP's 16-element activation units survived into the split.
Over the boot's weight vector that hands ranks 1 and 2 a 4752-wide half of
gate_up (9504 merged) — group- and tile-invalid.

The fix mirrors the two already-solved siblings, `AutoRoundConfig` (#86) and
`CompressedTensorsConfig._group_size_block` (#37): expose
`lcm(group_size, GPTQ_MARLIN_MIN_THREAD_K)` on both dims so the existing
uneven-TP machinery coarsens the split at plan time.

Pure functions, no GPU, no server.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import math
import unittest
from unittest import mock

from sglang.srt.distributed.utils import (
    ACTIVATION_VEC_ELEMS,
    set_tp_partition_ratios,
    tp_partition_sizes,
)
from sglang.srt.layers.linear import _quant_block_aligned_units
from sglang.srt.layers.moe.fused_moe_triton.layer import moe_uneven_tp_units
from sglang.srt.layers.quantization.awq.awq import (
    AWQConfig,
    AWQMarlinConfig,
    awq_uneven_tp_block,
)
from sglang.srt.layers.quantization.marlin_utils import (
    GPTQ_MARLIN_MIN_THREAD_K,
    GPTQ_MARLIN_MIN_THREAD_N,
    verify_marlin_supports_shape,
)
from sglang.test.test_utils import CustomTestCase

# Huihui-Qwen3.6-27B-abliterated-AWQ-MTP, AWQ 4-bit group 128.
INTERMEDIATE = 17408
HIDDEN = 5120
GROUP_SIZE = 128
# The exact vector the s03 boot installed: --rank-tp-ratio auto-performance
# resolved to the plain VRAM-auto split on 5090 + 2x3080 with
# --rank-auto-reserve-mib 3000,2700,2700.
S03_WEIGHTS = [29607, 17780, 17780]
# The same 3+1+1 co-location layout the #82 tests use for TP=5.
TP5_WEIGHTS = [9869, 9869, 9869, 18280, 18280]

# Head-granular families of this checkpoint, which must NOT be coarsened:
# GDN k-head units over the value dim, and the q block of qkv_proj in
# kv-head units.
GDN_VALUE_DIM, GDN_K_HEADS = 6144, 16
Q_DIM, KV_HEADS = 6144, 4
# The vision tower's intermediate size: not a multiple of the block at all.
VISION_INTERMEDIATE = 4304


def _awq_config(group_size: int = GROUP_SIZE) -> AWQConfig:
    return AWQConfig(
        weight_bits=4, group_size=group_size, zero_point=True,
        modules_to_not_convert=None,
    )


def _awq_marlin_config(group_size: int = GROUP_SIZE) -> AWQMarlinConfig:
    # The constructor's kernel-support probe reads the device capability,
    # which no CPU test host can satisfy. The shard plan does not depend on
    # it, so stub it out rather than gate this file behind a GPU.
    with mock.patch(
        "sglang.srt.layers.quantization.awq.awq.verify_marlin_supported"
    ):
        return AWQMarlinConfig(
            weight_bits=4,
            group_size=group_size,
            zero_point=True,
            lm_head_quantized=False,
            modules_to_not_convert=None,
            full_config={},
        )


def _mlp_units(intermediate: int, quant_config) -> int:
    """Mirrors the derivation in sglang.srt.models.qwen2_moe.Qwen2MoeMLP."""
    units = intermediate // math.gcd(intermediate, 16)
    return _quant_block_aligned_units(intermediate, units, quant_config, 1)


class TestAWQUnevenTPBlock(CustomTestCase):
    """The block itself: group- AND tile-aligned, on both dims."""

    def test_block_folds_group_and_marlin_k_tile(self):
        for group_size, expected in ((32, 128), (64, 128), (128, 128), (256, 256)):
            block = awq_uneven_tp_block(group_size)
            self.assertEqual(block, [expected, expected])
            self.assertEqual(block[0] % group_size, 0)
            self.assertEqual(block[0] % GPTQ_MARLIN_MIN_THREAD_K, 0)
            # min_thread_k dominates min_thread_n, so the same value keeps
            # the column-parallel output dim tile-valid.
            self.assertEqual(block[0] % GPTQ_MARLIN_MIN_THREAD_N, 0)

    def test_both_awq_configs_expose_it(self):
        for cfg in (_awq_config(), _awq_marlin_config()):
            self.assertEqual(cfg.weight_block_size, [128, 128])

    def test_degenerate_group_size_falls_back_to_the_tile(self):
        # group_size -1 (per-channel) carries no group constraint; the Marlin
        # tile still does.
        self.assertEqual(
            awq_uneven_tp_block(-1), [GPTQ_MARLIN_MIN_THREAD_K] * 2
        )


class TestS03BootRepro(CustomTestCase):
    """The 9504 constellation, before and after."""

    def setUp(self):
        set_tp_partition_ratios(S03_WEIGHTS)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_pre_change_reproduces_the_boot_crash(self):
        # No block exposed -> 16-element activation units survive.
        units = _mlp_units(INTERMEDIATE, None)
        self.assertEqual(units, INTERMEDIATE // 16)
        sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
        self.assertEqual(sizes, [7904, 4752, 4752])
        # gate_up is merged: the layer verifies the SUM of both halves.
        merged = [2 * s for s in sizes]
        self.assertEqual(merged, [15808, 9504, 9504])
        with self.assertRaisesRegex(ValueError, "9504.*min_thread_n = 64"):
            verify_marlin_supports_shape(
                output_size_per_partition=merged[1],
                input_size_per_partition=HIDDEN,
                input_size=HIDDEN,
                group_size=GROUP_SIZE,
            )
        # down_proj's K is group-invalid on the same ranks.
        self.assertTrue(any(s % GROUP_SIZE for s in sizes))

    def test_post_change_is_group_and_tile_clean(self):
        for cfg in (_awq_config(), _awq_marlin_config()):
            units = _mlp_units(INTERMEDIATE, cfg)
            self.assertEqual(units, INTERMEDIATE // 128)  # 136
            sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
            self.assertEqual(sizes, [7936, 4736, 4736])
            self.assertEqual(sum(sizes), INTERMEDIATE)
            for s in sizes:
                # gate_up (merged column-parallel output).
                verify_marlin_supports_shape(
                    output_size_per_partition=2 * s,
                    input_size_per_partition=HIDDEN,
                    input_size=HIDDEN,
                    group_size=GROUP_SIZE,
                )
                # down_proj (row-parallel input).
                verify_marlin_supports_shape(
                    output_size_per_partition=HIDDEN,
                    input_size_per_partition=s,
                    input_size=INTERMEDIATE,
                    group_size=GROUP_SIZE,
                )
                self.assertEqual(s % ACTIVATION_VEC_ELEMS, 0)

    def test_gate_up_and_down_agree(self):
        # Coarsening is idempotent on both dims, so the coupled pair
        # partitions the shared intermediate dimension identically.
        cfg = _awq_marlin_config()
        units = _mlp_units(INTERMEDIATE, cfg)
        self.assertEqual(_quant_block_aligned_units(INTERMEDIATE, units, cfg, 0), units)
        self.assertEqual(_quant_block_aligned_units(INTERMEDIATE, units, cfg, 1), units)

    def test_uneven_ratio_is_preserved(self):
        units = _mlp_units(INTERMEDIATE, _awq_marlin_config())
        sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
        total_w = sum(S03_WEIGHTS)
        for size, weight in zip(sizes, S03_WEIGHTS):
            share = size / INTERMEDIATE
            target = weight / total_w
            # The block is 128 wide, so the worst-case rounding error is well
            # under a percentage point of the dimension.
            self.assertLess(abs(share - target), 0.01)
        # Still strictly uneven: the strong rank keeps the largest shard.
        self.assertGreater(sizes[0], sizes[1])


class TestHeadGranularFamiliesPassThrough(CustomTestCase):
    """The block must coarsen only families FINER than itself."""

    def setUp(self):
        set_tp_partition_ratios(S03_WEIGHTS)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_gdn_k_head_units_unchanged(self):
        # 6144 / 16 = 384-element units, already a multiple of 128.
        cfg = _awq_marlin_config()
        self.assertEqual(
            _quant_block_aligned_units(GDN_VALUE_DIM, GDN_K_HEADS, cfg, 1),
            GDN_K_HEADS,
        )

    def test_qkv_q_block_units_unchanged(self):
        # q partitions in kv-head units of 1536 elements.
        cfg = _awq_marlin_config()
        self.assertEqual(
            _quant_block_aligned_units(Q_DIM, KV_HEADS, cfg, 0), KV_HEADS
        )

    def test_non_block_multiple_dimension_unchanged(self):
        # The vision intermediate is not block-quantizable at all
        # (4304 % 128 != 0) -> the quant method owns that case, not the
        # unit family.
        cfg = _awq_marlin_config()
        units = VISION_INTERMEDIATE
        self.assertEqual(
            _quant_block_aligned_units(VISION_INTERMEDIATE, units, cfg, 0), units
        )


class TestInertOnTheDefaultPath(CustomTestCase):
    """No installed plan -> the split is the classic even one."""

    def setUp(self):
        set_tp_partition_ratios(None)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_even_split_identical_with_and_without_the_block(self):
        pre = _mlp_units(INTERMEDIATE, None)
        post = _mlp_units(INTERMEDIATE, _awq_marlin_config())
        for tp in (1, 2, 4, 8):
            classic = [INTERMEDIATE // tp] * tp
            self.assertEqual(
                tp_partition_sizes(INTERMEDIATE, tp, units=pre, family="mlp"), classic
            )
            self.assertEqual(
                tp_partition_sizes(INTERMEDIATE, tp, units=post, family="mlp"), classic
            )

    def test_even_tp_shards_stay_marlin_valid(self):
        for tp in (1, 2, 4, 8):
            shard = INTERMEDIATE // tp
            verify_marlin_supports_shape(
                output_size_per_partition=2 * shard,
                input_size_per_partition=HIDDEN,
                input_size=HIDDEN,
                group_size=GROUP_SIZE,
            )


class TestTP5CoLocation(CustomTestCase):
    """Three ranks on one card: 136 units still feed five ranks."""

    def setUp(self):
        set_tp_partition_ratios(TP5_WEIGHTS)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_tp5_shards_are_tile_clean(self):
        units = _mlp_units(INTERMEDIATE, _awq_marlin_config())
        sizes = tp_partition_sizes(INTERMEDIATE, 5, units=units, family="mlp")
        self.assertEqual(sum(sizes), INTERMEDIATE)
        self.assertTrue(all(s >= 128 for s in sizes))
        for s in sizes:
            self.assertEqual(s % 128, 0)
            self.assertEqual((2 * s) % GPTQ_MARLIN_MIN_THREAD_N, 0)


class _FakeFp8BlockConfig:
    weight_block_size = [128, 128]


class _FakeCompressedTensorsWeights:
    def __init__(self, group_size):
        self.group_size = group_size


class _FakeCompressedTensorsConfig:
    def __init__(self, group_size):
        self.target_scheme_map = {
            "Linear": {"weights": _FakeCompressedTensorsWeights(group_size)}
        }
        self.weight_block_size = [math.lcm(group_size, 128)] * 2


class TestMoEGrainUnchanged(CustomTestCase):
    """The dense block must not hijack the expert intermediate grain.

    The fused-MoE tile is 64, and expert ffn sizes are small, so the
    128-wide dense block would starve the unit family. These pin the grain
    each config class produced before the AWQ block existed.
    """

    def test_a3b_awq_keeps_group_grain(self):
        # Qwen3.6-35B-A3B-AWQ-4bit: group 32, moe_intermediate 512.
        for cfg in (_awq_config(32), _awq_marlin_config(32)):
            self.assertEqual(moe_uneven_tp_units(512, cfg), 512 // 32)

    def test_gemma4_a4b_awq_keeps_halved_group_grain(self):
        # 704 is not a multiple of the config group 128 -> halve to 64.
        for cfg in (_awq_config(128), _awq_marlin_config(128)):
            self.assertEqual(moe_uneven_tp_units(704, cfg), 704 // 64)

    def test_block_quantized_experts_still_use_the_block(self):
        self.assertEqual(moe_uneven_tp_units(1024, _FakeFp8BlockConfig()), 8)

    def test_compressed_tensors_grain_unchanged(self):
        # CompressedTensorsConfig exposes no `group_size` attribute, so the
        # guard cannot fire for it and its grain is exactly what it was:
        # the derived dense block where that divides the dimension, and the
        # lcm(group, MoE tile) fallback where it does not (704 % 128 != 0).
        cfg = _FakeCompressedTensorsConfig(32)
        self.assertEqual(moe_uneven_tp_units(512, cfg), 512 // 128)
        self.assertEqual(moe_uneven_tp_units(704, cfg), 704 // 64)

    def test_unquantized_experts_stay_element_granular(self):
        self.assertEqual(moe_uneven_tp_units(512, None), 512)


if __name__ == "__main__":
    unittest.main()
