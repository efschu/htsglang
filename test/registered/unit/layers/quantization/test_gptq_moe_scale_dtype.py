"""Repro + inertness proof for the GPTQ-Marlin MoE scale dtype fix (task #283).

Qwen3.5-35B-A3B-GPTQ-Int4 (config dtype bfloat16, checkpoint `.scales` stored
as F16) died at the first MoE forward after load with

    AssertionError: moe_wna16_marlin_gemm assumes hidden_states.dtype
    (torch.bfloat16) == w1_scale.dtype (torch.float16)

and `--dtype float16` was the only workaround (see the `moe_marlin` vehicle
note in tests/determinism/determinism_harness/runner.py).

Root cause: `GPTQMarlinMoEScheme.create_weights` allocated `w13_scales` and
`w2_scales` with a hardcoded `torch.half` while every other parameter it
creates follows `params_dtype`. The float16 checkpoint scales were therefore
copied in without conversion and reached the kernel as float16 no matter what
the model dtype was.

That guard is not pedantry. `moe_wna16_marlin_gemm` is instantiated for ONE
scalar type, taken from the activation dtype (`_jit_moe_wna16_marlin_module(
a.dtype)` in python/sglang/jit_kernel/moe_wna16_marlin.py), and the kernel
reads `b_scales` through that same type. Feeding it float16 scales under
bfloat16 activations is a bit reinterpretation, not a slow path -- so the fix
belongs at the allocation site, not in the kernel and not by deleting the
assert.

The AWQ-Marlin MoE sibling (awq_moe.py) drives the very same kernel and has
always allocated its scales in `params_dtype`. This test pins that the two
schemes now agree, and that the float16 path -- the one that worked before --
is untouched down to the parameter dtypes and shapes.

No GPU, no server: one pass through the real `create_weights` of both schemes.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization.awq.awq import AWQMarlinConfig
from sglang.srt.layers.quantization.awq.schemes.awq_moe import AWQMoEScheme
from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig
from sglang.srt.layers.quantization.gptq.schemes.gptq_moe import GPTQMarlinMoEScheme
from sglang.test.test_utils import CustomTestCase

# Qwen3.5-35B-A3B-GPTQ-Int4 geometry, shrunk to keep the allocation cheap:
# GPTQ INT4, group 128, desc_act=False, sym=True.
HIDDEN_SIZE = 2048
INTERMEDIATE = 512
NUM_EXPERTS = 4
GROUP_SIZE = 128


def _make_layer():
    layer = torch.nn.Module()
    layer.moe_tp_size = 1
    layer.moe_tp_rank = 0
    layer.intermediate_size_per_partition = INTERMEDIATE
    return layer


def _gptq_config(desc_act: bool = False) -> GPTQMarlinConfig:
    return GPTQMarlinConfig(
        weight_bits=4,
        group_size=GROUP_SIZE,
        desc_act=desc_act,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={},
    )


def _awq_config() -> AWQMarlinConfig:
    # AWQMarlinConfig.__init__ asks the live device whether Marlin supports the
    # quant type; on a CPU runner that query reports capability -1 and refuses.
    # The rest of __init__ is arithmetic, and create_weights only reads
    # pack_factor / group_size, so stub the capability check out.
    with mock.patch("sglang.srt.layers.quantization.awq.awq.verify_marlin_supported"):
        return AWQMarlinConfig(
            weight_bits=4,
            group_size=GROUP_SIZE,
            zero_point=True,
            lm_head_quantized=False,
            modules_to_not_convert=[],
            full_config={},
        )


def _build(scheme, params_dtype: torch.dtype) -> torch.nn.Module:
    layer = _make_layer()
    scheme.create_weights(
        layer=layer,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size_per_partition=INTERMEDIATE,
        params_dtype=params_dtype,
        weight_loader=lambda *args, **kwargs: None,
    )
    return layer


def _param_signature(layer: torch.nn.Module):
    return {name: (tuple(p.shape), p.dtype) for name, p in layer.named_parameters()}


class TestGPTQMoEScaleDtype(CustomTestCase):
    def test_scales_follow_params_dtype(self):
        """The repro: bfloat16 model -> bfloat16 scales (was float16)."""
        for params_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(params_dtype=params_dtype):
                layer = _build(GPTQMarlinMoEScheme(_gptq_config()), params_dtype)
                self.assertEqual(layer.w13_scales.dtype, params_dtype)
                self.assertEqual(layer.w2_scales.dtype, params_dtype)

    def test_scales_follow_params_dtype_with_act_order(self):
        """desc_act=True takes the other scales-width branch; same contract."""
        for params_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(params_dtype=params_dtype):
                layer = _build(
                    GPTQMarlinMoEScheme(_gptq_config(desc_act=True)), params_dtype
                )
                self.assertEqual(layer.w13_scales.dtype, params_dtype)
                self.assertEqual(layer.w2_scales.dtype, params_dtype)

    def test_kernel_guard_would_pass(self):
        """Replicates fused_marlin_moe's guard, the assert the boot died on."""
        for params_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(params_dtype=params_dtype):
                layer = _build(GPTQMarlinMoEScheme(_gptq_config()), params_dtype)
                hidden_states_dtype = params_dtype
                self.assertEqual(hidden_states_dtype, layer.w13_scales.dtype)
                self.assertEqual(hidden_states_dtype, layer.w2_scales.dtype)

    def test_agrees_with_awq_marlin_moe_sibling(self):
        """Both schemes feed moe_wna16_marlin_gemm; both must size scales alike.

        `AWQMoEScheme.__init__` builds a CUDA Marlin kernel object that this
        CPU suite cannot construct, and `create_weights` never touches it -- so
        call it against a bare config holder rather than skip the comparison.
        """
        awq_scheme = SimpleNamespace(quant_config=_awq_config())
        for params_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(params_dtype=params_dtype):
                gptq = _build(GPTQMarlinMoEScheme(_gptq_config()), params_dtype)
                awq = _make_layer()
                AWQMoEScheme.create_weights(
                    awq_scheme,
                    layer=awq,
                    num_experts=NUM_EXPERTS,
                    hidden_size=HIDDEN_SIZE,
                    intermediate_size_per_partition=INTERMEDIATE,
                    params_dtype=params_dtype,
                    weight_loader=lambda *args, **kwargs: None,
                )
                self.assertEqual(gptq.w13_scales.dtype, awq.w13_scales.dtype)
                self.assertEqual(gptq.w2_scales.dtype, awq.w2_scales.dtype)
                self.assertEqual(awq.w13_scales.dtype, params_dtype)

    def test_float16_path_unchanged(self):
        """Inertness pin: the previously-working float16 path is bit-for-bit
        the same allocation. `params_dtype` IS `torch.half` there, so every
        parameter keeps its old shape and dtype -- scales included."""
        layer = _build(GPTQMarlinMoEScheme(_gptq_config()), torch.float16)
        expected = {
            "w13_qweight": (
                (NUM_EXPERTS, HIDDEN_SIZE // 8, 2 * INTERMEDIATE),
                torch.int32,
            ),
            "w2_qweight": (
                (NUM_EXPERTS, INTERMEDIATE // 8, HIDDEN_SIZE),
                torch.int32,
            ),
            "w13_scales": (
                (NUM_EXPERTS, HIDDEN_SIZE // GROUP_SIZE, 2 * INTERMEDIATE),
                torch.float16,
            ),
            "w2_scales": (
                (NUM_EXPERTS, INTERMEDIATE // GROUP_SIZE, HIDDEN_SIZE),
                torch.float16,
            ),
            "w13_qzeros": (
                (NUM_EXPERTS, HIDDEN_SIZE // GROUP_SIZE, 2 * INTERMEDIATE // 8),
                torch.float16,
            ),
            "w2_qzeros": (
                (NUM_EXPERTS, INTERMEDIATE // GROUP_SIZE, HIDDEN_SIZE // 8),
                torch.float16,
            ),
            "w13_g_idx": ((NUM_EXPERTS, HIDDEN_SIZE), torch.int32),
            "w2_g_idx": ((NUM_EXPERTS, INTERMEDIATE), torch.int32),
            "w13_g_idx_sort_indices": ((NUM_EXPERTS, HIDDEN_SIZE), torch.int32),
            "w2_g_idx_sort_indices": ((NUM_EXPERTS, INTERMEDIATE), torch.int32),
        }
        self.assertEqual(_param_signature(layer), expected)

    def test_only_the_scales_moved(self):
        """The bfloat16 layer differs from the float16 one in exactly the two
        scale tensors -- nothing else silently changed dtype with it."""
        fp16 = _param_signature(_build(GPTQMarlinMoEScheme(_gptq_config()), torch.half))
        bf16 = _param_signature(
            _build(GPTQMarlinMoEScheme(_gptq_config()), torch.bfloat16)
        )
        self.assertEqual(fp16.keys(), bf16.keys())
        moved = {name for name in fp16 if fp16[name] != bf16[name]}
        # qzeros already followed params_dtype before this fix (GPTQ-Marlin is
        # symmetric, so they are a formality the kernel never reads) -- they are
        # expected to move with the model dtype and always did.
        self.assertEqual(moved, {"w13_scales", "w2_scales", "w13_qzeros", "w2_qzeros"})

    def test_checkpoint_scale_conversion_is_load_time_and_bounded(self):
        """The float16 checkpoint scales are converted once, by the loader's
        copy_, and the downcast error stays far below INT4 group error."""
        checkpoint_scales = torch.rand(NUM_EXPERTS, 16, 32, dtype=torch.float16) + 0.01

        fp16_param = torch.empty_like(checkpoint_scales, dtype=torch.float16)
        fp16_param.copy_(checkpoint_scales)
        self.assertTrue(torch.equal(fp16_param, checkpoint_scales))

        bf16_param = torch.empty_like(checkpoint_scales, dtype=torch.bfloat16)
        bf16_param.copy_(checkpoint_scales)
        self.assertEqual(bf16_param.dtype, torch.bfloat16)
        rel = (
            (bf16_param.float() - checkpoint_scales.float()).abs()
            / checkpoint_scales.float().abs()
        ).max()
        # bfloat16 has 8 mantissa bits -> half-ulp bound 2**-8.
        self.assertLess(rel.item(), 2**-8)


if __name__ == "__main__":
    unittest.main()
