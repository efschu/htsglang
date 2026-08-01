"""Repro + inertness proof for the GPTQ x uneven-TP unit-count fix (task #300).

The round-7b GPTQ arm of the #274 battery
(Qwen3.6-27B-...-Native-MTP-Preserved-GPTQ-Int4, dense, intermediate 17408,
GPTQ INT4 group 128) died at the first dense MLP with

    Dimension of size 136 is not a multiple of its unit count 1088

That is NOT the AWQ failure of #289 (a Marlin tile assert on 9504) even though
it has the same origin. `GPTQConfig` / `GPTQMarlinConfig` never exposed a
`weight_block_size`, so `_quant_block_aligned_units` had nothing to coarsen
with and Qwen2MoeMLP's 16-element activation units survived into the split:
K = [7904, 4752, 4752], none of them a multiple of the group. The Marlin
scheme then sizes down_proj's `scales` at `K // group_size` rows
(61 / 37 / 37) against a checkpoint dimension of 17408 // 128 = 136, and the
row-parallel loader is asked to partition 136 elements in the WEIGHT's 1088
units. 136 is the coarse (block) count, 1088 the fine (16-element) one --
hence the two numbers in the message.

The fix mirrors the three already-solved siblings, `AWQConfig` (#289),
`AutoRoundConfig` (#86) and `CompressedTensorsConfig._group_size_block` (#37):
expose `lcm(group_size, GPTQ_MARLIN_MIN_THREAD_K)` on both dims so the
existing uneven-TP machinery coarsens the split at plan time.

The validated GPTQ configuration of #77 (Qwen3.5-35B-A3B-GPTQ, TP=3 uneven
2:1:1, per-rank expert groups 2/1/1; Qwen3.5-122B-A10B-GPTQ) must be
unaffected: its shard geometry comes from `moe_uneven_tp_units`, whose
group-quant guard neutralises the new block, and its dense layers are excluded
from quantization by the checkpoint's `dynamic` map. Both are pinned below.

Plan-time arithmetic plus one pass through the real
`GPTQMarlinLinearScheme.create_weights`; no GPU, no server.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import copy
import math
import unittest
from unittest import mock

import torch

from sglang.srt.distributed.utils import (
    ACTIVATION_VEC_ELEMS,
    set_tp_partition_ratios,
    tp_loaded_shard_start,
    tp_partition_sizes,
)
from sglang.srt.layers.linear import _quant_block_aligned_units
from sglang.srt.layers.moe.fused_moe_triton.layer import moe_uneven_tp_units
from sglang.srt.layers.quantization.gptq.gptq import (
    GPTQConfig,
    GPTQMarlinConfig,
    gptq_uneven_tp_block,
)
from sglang.srt.layers.quantization.gptq.schemes.gptq_marlin import (
    GPTQMarlinLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils import (
    GPTQ_MARLIN_MIN_THREAD_K,
    GPTQ_MARLIN_MIN_THREAD_N,
    verify_marlin_supports_shape,
)
from sglang.srt.layers.quantization.utils import get_dynamic_override
from sglang.test.test_utils import CustomTestCase

# Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4:
# dense, GPTQ 4-bit group 128, desc_act False.
INTERMEDIATE = 17408
HIDDEN = 5120
GROUP_SIZE = 128
# The weight vector round 7b installed: --rank-tp-ratio auto-performance on
# 5090 + 2x3080 with --rank-auto-reserve-mib 3000,2700,2700.
R7B_WEIGHTS = [29607, 17780, 17780]
# The same 3+1+1 co-location layout the #82 tests use for TP=5.
TP5_WEIGHTS = [9869, 9869, 9869, 18280, 18280]

# Head-granular families of this checkpoint, which must NOT be coarsened:
# GDN k-head units over the value dim, and the q block of qkv_proj in
# kv-head units (head_dim 256, 24 q heads, 4 kv heads).
GDN_VALUE_DIM, GDN_K_HEADS = 6144, 16
Q_DIM, KV_HEADS = 6144, 4
# A dimension the block does not divide at all.
VISION_INTERMEDIATE = 4304

# #77's validated GPTQ configurations.
A3B_MOE_INTERMEDIATE = 512  # Qwen3.5-35B-A3B-GPTQ-Int4
A3B_WEIGHTS = [2, 1, 1]  # TP=3 uneven auto 2:1:1 on 5090 + 2x3080
MOE_122B_INTERMEDIATE = 1024  # Qwen3.5-122B-A10B-GPTQ-Int4
# The `dynamic` map both MoE checkpoints ship: every dense module is a
# negative match, so no dense layer of theirs is quantized at all.
MOE_DYNAMIC = {
    "lm_head": {},
    "model.language_model.embed_tokens": {},
    "-:.*attn.*": {},
    "-:.*shared_expert.*": {},
    "-:.*mtp.*": {},
    "-:.*visual.*": {},
}


def _gptq_config(group_size: int = GROUP_SIZE, dynamic=None) -> GPTQConfig:
    return GPTQConfig(
        weight_bits=4,
        group_size=group_size,
        desc_act=False,
        lm_head_quantized=False,
        dynamic=dynamic or {},
        checkpoint_format="gptq",
    )


def _gptq_marlin_config(group_size: int = GROUP_SIZE, dynamic=None) -> GPTQMarlinConfig:
    return GPTQMarlinConfig(
        weight_bits=4,
        group_size=group_size,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic=dynamic or {},
        full_config={},
    )


def _without_the_block(config):
    """The same config as it behaved BEFORE this fix: no weight_block_size."""
    clone = copy.deepcopy(config)
    del clone.weight_block_size
    return clone


def _mlp_units(intermediate: int, quant_config) -> int:
    """Mirrors the derivation in sglang.srt.models.qwen2_moe.Qwen2MoeMLP."""
    units = intermediate // math.gcd(intermediate, ACTIVATION_VEC_ELEMS)
    return _quant_block_aligned_units(intermediate, units, quant_config, 1)


def _marlin_scheme(config) -> GPTQMarlinLinearScheme:
    """The real scheme, with only its two GPU touch points stubbed: the
    kernel-support probe reads the device capability and `_init_kernel`
    imports the CUDA backend. Neither takes part in weight creation."""
    with (
        mock.patch(
            "sglang.srt.layers.quantization.gptq.schemes.gptq_marlin.verify_marlin_supported"
        ),
        mock.patch.object(
            GPTQMarlinLinearScheme, "_init_kernel", return_value=mock.MagicMock()
        ),
    ):
        return GPTQMarlinLinearScheme(config)


def _down_proj_params(scheme, k_per_rank: int):
    """The parameters GPTQMarlinLinearScheme allocates for one rank's
    down_proj shard (row-parallel: input dim sharded, output dim whole)."""
    layer = torch.nn.Module()
    scheme.create_weights(
        layer=layer,
        input_size_per_partition=k_per_rank,
        output_partition_sizes=[HIDDEN],
        input_size=INTERMEDIATE,
        output_size=HIDDEN,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **kw: None,
    )
    return layer


class TestGPTQUnevenTPBlock(CustomTestCase):
    """The block itself: group- AND tile-aligned, on both dims."""

    def test_block_folds_group_and_marlin_k_tile(self):
        for group_size, expected in ((32, 128), (64, 128), (128, 128), (256, 256)):
            block = gptq_uneven_tp_block(group_size)
            self.assertEqual(block, [expected, expected])
            self.assertEqual(block[0] % group_size, 0)
            self.assertEqual(block[0] % GPTQ_MARLIN_MIN_THREAD_K, 0)
            # min_thread_k dominates min_thread_n, so the same value keeps
            # the column-parallel output dim tile-valid.
            self.assertEqual(block[0] % GPTQ_MARLIN_MIN_THREAD_N, 0)

    def test_both_gptq_configs_expose_it(self):
        for cfg in (_gptq_config(), _gptq_marlin_config()):
            self.assertEqual(cfg.weight_block_size, [128, 128])

    def test_degenerate_group_size_falls_back_to_the_tile(self):
        # group_size -1 (per-channel) carries no group constraint; the Marlin
        # tile still does.
        self.assertEqual(gptq_uneven_tp_block(-1), [GPTQ_MARLIN_MIN_THREAD_K] * 2)

    def test_positive_dynamic_group_override_is_folded_in(self):
        # GPTQModel's per-module rules may raise a layer's group size above
        # the base one. The partition is planned from the BASE config, so the
        # block has to cover every layer.
        dynamic = {r"+:.*\.mlp\..*": {"group_size": 256}}
        for cfg in (
            _gptq_config(dynamic=dynamic),
            _gptq_marlin_config(dynamic=dynamic),
        ):
            self.assertEqual(cfg.weight_block_size, [256, 256])

    def test_negative_and_non_group_dynamic_rules_change_nothing(self):
        # Skip rules and overrides of other keys leave the block at the base
        # value -- this is the shape both #77 MoE checkpoints ship.
        for dynamic in (MOE_DYNAMIC, {r"+:.*\.1[0-5]\..*": {"bits": 8}}):
            for cfg in (
                _gptq_config(dynamic=dynamic),
                _gptq_marlin_config(dynamic=dynamic),
            ):
                self.assertEqual(cfg.weight_block_size, [128, 128])


class TestRound7bBootRepro(CustomTestCase):
    """The 136-vs-1088 constellation, before and after."""

    def setUp(self):
        set_tp_partition_ratios(R7B_WEIGHTS)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_pre_change_reproduces_the_boot_crash(self):
        # No block exposed -> 16-element activation units survive.
        cfg = _without_the_block(_gptq_marlin_config())
        units = _mlp_units(INTERMEDIATE, cfg)
        self.assertEqual(units, INTERMEDIATE // ACTIVATION_VEC_ELEMS)  # 1088
        sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
        self.assertEqual(sizes, [7904, 4752, 4752])
        # Not one rank's K is group-aligned, so GPTQMarlinLinearScheme sizes
        # down_proj's scales at floor(K / 128) rows against a 136-row
        # checkpoint dimension.
        self.assertTrue(all(s % GROUP_SIZE for s in sizes))
        scale_rows = [s // GROUP_SIZE for s in sizes]
        self.assertEqual(scale_rows, [61, 37, 37])
        self.assertNotEqual(sum(scale_rows), INTERMEDIATE // GROUP_SIZE)
        for rank, rows in enumerate(scale_rows):
            with self.assertRaisesRegex(
                ValueError,
                r"Dimension of size 136 is not a multiple of its unit count 1088",
            ):
                tp_loaded_shard_start(
                    INTERMEDIATE // GROUP_SIZE,
                    None,
                    rank,
                    rows,
                    units,
                    family="mlp",
                )

    def test_post_change_is_group_and_tile_clean(self):
        for cfg in (_gptq_config(), _gptq_marlin_config()):
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

    def test_post_change_scales_partition_tiles_the_checkpoint(self):
        # The actual crash site: the row-parallel loader narrowing the 136-row
        # scales tensor. Every rank's slice must fit head to tail.
        units = _mlp_units(INTERMEDIATE, _gptq_marlin_config())
        sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
        full_rows = INTERMEDIATE // GROUP_SIZE
        offset = 0
        for rank, s in enumerate(sizes):
            rows = s // GROUP_SIZE
            self.assertEqual(rows * GROUP_SIZE, s)  # exact, no floor
            start = tp_loaded_shard_start(
                full_rows, None, rank, rows, units, family="mlp"
            )
            self.assertEqual(start, offset)
            offset += rows
        self.assertEqual(offset, full_rows)

    def test_scheme_and_loader_end_to_end(self):
        """Through the real GPTQMarlinLinearScheme, not just the arithmetic.

        Pre-change every rank raises the round-7b message; post-change the
        three ranks' scale slices tile the checkpoint's 136 rows exactly and
        the packed qweight rows tile its 2176.
        """
        cfg = _gptq_marlin_config()
        full_rows = INTERMEDIATE // GROUP_SIZE

        pre_units = _mlp_units(INTERMEDIATE, _without_the_block(cfg))
        pre_scheme = _marlin_scheme(_without_the_block(cfg))
        for rank, k in enumerate(
            tp_partition_sizes(INTERMEDIATE, 3, units=pre_units, family="mlp")
        ):
            rows = _down_proj_params(pre_scheme, k).scales.shape[0]
            with self.assertRaisesRegex(
                ValueError,
                r"Dimension of size 136 is not a multiple of its unit count 1088",
            ):
                tp_loaded_shard_start(
                    full_rows, None, rank, rows, pre_units, family="mlp"
                )

        post_units = _mlp_units(INTERMEDIATE, cfg)
        post_scheme = _marlin_scheme(cfg)
        rows_seen, packed_seen, offset = 0, 0, 0
        for rank, k in enumerate(
            tp_partition_sizes(INTERMEDIATE, 3, units=post_units, family="mlp")
        ):
            layer = _down_proj_params(post_scheme, k)
            rows = layer.scales.shape[0]
            self.assertEqual(
                tp_loaded_shard_start(
                    full_rows, None, rank, rows, post_units, family="mlp"
                ),
                offset,
            )
            self.assertEqual(layer.g_idx.shape[0], k)
            offset += rows
            rows_seen += rows
            packed_seen += layer.qweight.shape[0]
        self.assertEqual(rows_seen, full_rows)  # 62 + 37 + 37
        self.assertEqual(packed_seen, INTERMEDIATE // cfg.pack_factor)  # 2176

    def test_gate_up_and_down_agree(self):
        # Coarsening is idempotent on both dims, so the coupled pair
        # partitions the shared intermediate dimension identically.
        cfg = _gptq_marlin_config()
        units = _mlp_units(INTERMEDIATE, cfg)
        self.assertEqual(_quant_block_aligned_units(INTERMEDIATE, units, cfg, 0), units)
        self.assertEqual(_quant_block_aligned_units(INTERMEDIATE, units, cfg, 1), units)

    def test_uneven_ratio_is_preserved(self):
        units = _mlp_units(INTERMEDIATE, _gptq_marlin_config())
        sizes = tp_partition_sizes(INTERMEDIATE, 3, units=units, family="mlp")
        total_w = sum(R7B_WEIGHTS)
        for size, weight in zip(sizes, R7B_WEIGHTS):
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
        set_tp_partition_ratios(R7B_WEIGHTS)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_gdn_k_head_units_unchanged(self):
        # 6144 / 16 = 384-element units, already a multiple of 128.
        cfg = _gptq_marlin_config()
        self.assertEqual(
            _quant_block_aligned_units(GDN_VALUE_DIM, GDN_K_HEADS, cfg, 1),
            GDN_K_HEADS,
        )

    def test_qkv_q_block_units_unchanged(self):
        # q partitions in kv-head units of 1536 elements.
        cfg = _gptq_marlin_config()
        self.assertEqual(_quant_block_aligned_units(Q_DIM, KV_HEADS, cfg, 0), KV_HEADS)

    def test_non_block_multiple_dimension_unchanged(self):
        # 4304 % 128 != 0 -> not block-quantizable at all; the quant method
        # owns that case, not the unit family.
        cfg = _gptq_marlin_config()
        self.assertEqual(
            _quant_block_aligned_units(
                VISION_INTERMEDIATE, VISION_INTERMEDIATE, cfg, 0
            ),
            VISION_INTERMEDIATE,
        )


class TestInertOnTheDefaultPath(CustomTestCase):
    """No installed plan -> the split is the classic even one."""

    def setUp(self):
        set_tp_partition_ratios(None)

    def tearDown(self):
        set_tp_partition_ratios(None)

    def test_even_split_identical_with_and_without_the_block(self):
        cfg = _gptq_marlin_config()
        pre = _mlp_units(INTERMEDIATE, _without_the_block(cfg))
        post = _mlp_units(INTERMEDIATE, cfg)
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
        units = _mlp_units(INTERMEDIATE, _gptq_marlin_config())
        sizes = tp_partition_sizes(INTERMEDIATE, 5, units=units, family="mlp")
        self.assertEqual(sum(sizes), INTERMEDIATE)
        for s in sizes:
            self.assertGreaterEqual(s, 128)
            self.assertEqual(s % 128, 0)
            self.assertEqual((2 * s) % GPTQ_MARLIN_MIN_THREAD_N, 0)


class Test77RegressionMoEGrainUnchanged(CustomTestCase):
    """#77's validated GPTQ MoE geometry must compute identically.

    Its shard plan comes from `moe_uneven_tp_units`, not from the dense
    `_quant_block_aligned_units`: the group-quant guard added with #289
    neutralises a config whose weight_block_size IS lcm(group, K tile), which
    is exactly what this fix installs. So the expert grain is byte-identical
    before and after.
    """

    def test_a3b_expert_grain_and_shards_identical(self):
        # Qwen3.5-35B-A3B-GPTQ-Int4, TP=3 uneven auto 2:1:1 -- the boot #77
        # validated (per-rank expert groups 2/1/1, 32/32 token-identical).
        set_tp_partition_ratios(A3B_WEIGHTS)
        try:
            for cfg in (
                _gptq_config(dynamic=MOE_DYNAMIC),
                _gptq_marlin_config(dynamic=MOE_DYNAMIC),
            ):
                pre = moe_uneven_tp_units(A3B_MOE_INTERMEDIATE, _without_the_block(cfg))
                post = moe_uneven_tp_units(A3B_MOE_INTERMEDIATE, cfg)
                self.assertEqual(pre, A3B_MOE_INTERMEDIATE // GROUP_SIZE)  # 4
                self.assertEqual(post, pre)
                shards = tp_partition_sizes(
                    A3B_MOE_INTERMEDIATE, 3, units=post, family="moe"
                )
                self.assertEqual(shards, [256, 128, 128])
                # The per-rank group counts the #77 boot log recorded.
                self.assertEqual([s // GROUP_SIZE for s in shards], [2, 1, 1])
        finally:
            set_tp_partition_ratios(None)

    def test_122b_expert_grain_identical(self):
        for cfg in (
            _gptq_config(dynamic=MOE_DYNAMIC),
            _gptq_marlin_config(dynamic=MOE_DYNAMIC),
        ):
            pre = moe_uneven_tp_units(MOE_122B_INTERMEDIATE, _without_the_block(cfg))
            post = moe_uneven_tp_units(MOE_122B_INTERMEDIATE, cfg)
            self.assertEqual(pre, MOE_122B_INTERMEDIATE // GROUP_SIZE)  # 8
            self.assertEqual(post, pre)

    def test_grain_identical_across_the_group_size_range(self):
        for group in (32, 64, 128):
            for intermediate in (512, 704, 768, 1024, 1536):
                cfg = _gptq_marlin_config(group)
                self.assertEqual(
                    moe_uneven_tp_units(intermediate, cfg),
                    moe_uneven_tp_units(intermediate, _without_the_block(cfg)),
                    f"grain moved for group={group} intermediate={intermediate}",
                )

    def test_dense_layers_of_those_checkpoints_are_not_quantized(self):
        # The other half of "#77 untouched": every dense module of both MoE
        # checkpoints is a negative `dynamic` match, so its linear method is
        # UnquantizedLinearMethod and `linear._quant_block_aligned_units`
        # receives quant_config=None -- the block cannot reach it.
        cfg = _gptq_marlin_config(dynamic=MOE_DYNAMIC)
        for prefix in (
            "model.language_model.layers.0.self_attn.qkv_proj",
            "model.language_model.layers.0.self_attn.o_proj",
            "model.language_model.layers.0.mlp.shared_expert.gate_up_proj",
            "model.language_model.layers.0.mlp.shared_expert.down_proj",
            "model.language_model.mtp.layers.0.mlp.gate_up_proj",
        ):
            self.assertIs(get_dynamic_override(cfg, prefix), False, prefix)
        raw = INTERMEDIATE // ACTIVATION_VEC_ELEMS
        self.assertEqual(_quant_block_aligned_units(INTERMEDIATE, raw, None, 1), raw)


class TestMoEGrainUnchangedForSiblingConfigs(CustomTestCase):
    """The dense block must not hijack the expert grain of other classes."""

    def test_block_quantized_experts_still_use_the_block(self):
        class _FakeFp8BlockConfig:
            weight_block_size = [128, 128]

        self.assertEqual(moe_uneven_tp_units(1024, _FakeFp8BlockConfig()), 8)

    def test_unquantized_experts_use_the_activation_vector_grain(self):
        # #367 replaced the element-granular fall-through with the activation
        # kernel's vector width. The property this class exists for holds
        # either way: the GPTQ dense block does not reach the unquantized
        # lane, and 32 units are still finer than any quant grain.
        from sglang.srt.distributed.utils import ACTIVATION_VEC_ELEMS

        self.assertEqual(moe_uneven_tp_units(512, None), 512 // ACTIVATION_VEC_ELEMS)


if __name__ == "__main__":
    unittest.main()
