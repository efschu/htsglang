"""Marlin MoE runner core with hook support for LoRA injection.

Uses Marlin int4/int8 kernels for the base MoE projections.
LoRA deltas are injected via hooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.kernels.ops.activation import silu_and_mul
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        get_scalar_type,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        moe_align_block_size,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
        moe_sum_reduce_triton,
    )
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace


def _acquire_workspace(device) -> torch.Tensor:
    """The Marlin LoRA workspace, through the lane-keyed buffer accessor.

    This was the one ``resources.buffers`` entry managed past ``get_buffer``
    with a raw name key (DESIGN_121 §13.11 item 3): harmless while nothing
    on a lane path reached it, and the next place the workspace-family
    question would have been asked the moment a LoRA+MoE+Marlin model ran on
    a concurrent lane -- two threads sharing one Marlin lock buffer is the
    same silent-wrong-output shape as the flashinfer workspace collision.
    Routing it through the accessor gives every lane its own tensor and the
    serving group its usual single one, and books it in the offload register
    like the attention workspaces.

    The device belongs in the NAME: the old code re-created the workspace on
    a device change, and a keyed-lazy accessor expresses that as one entry
    per device (the workspace is a few hundred ints -- keeping a stale
    per-device entry is cheaper than an invalidation protocol on the hot
    path).
    """
    from sglang.srt.runtime_context import get_buffer

    return get_buffer(
        f"marlin_lora_workspace:{device}",
        lambda: marlin_make_workspace(device, max_blocks_per_sm=4),
    )


class MarlinLoraRunnerCore:
    """
    MoE runner using Marlin kernels for base projections, with hooks for LoRA.

    Pipeline:
      1. moe_wna16_marlin_gemm (gate_up)
      1.5. hooks.after_gate_up
      2. silu_and_mul
      3. moe_wna16_marlin_gemm (down)
      3.5. hooks.after_down
      4. moe_sum_reduce
    """

    def __init__(self, config: MoeRunnerConfig):
        self.config = config

    def run_from_dispatch(
        self,
        dispatch_output: StandardDispatchOutput,
        quant_info: MarlinMoeQuantInfo,
        runner_config: MoeRunnerConfig,
        hooks=None,
    ) -> StandardCombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        assert hooks is not None, "hooks must be provided for MarlinLoraRunnerCore"

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

        assert runner_config.activation == "silu", "Only SiLU activation is supported."
        # Vendor first: "compute capability >= 9" is an NVIDIA statement, and
        # get_device_capability() answers in the caller's vendor namespace. The
        # two COLLIDE -- gfx900 reports (9, 0) -- so the bare comparison lets an
        # AMD card through an NVIDIA gate and it then dies inside a Marlin
        # kernel that does not exist on ROCm at all. Refusing here is the loud,
        # correct direction; sailing through is the dangerous one.
        assert is_cuda(), (
            "MarlinLoraRunnerCore requires CUDA: Marlin has no ROCm kernel. "
            "(Any capability this build reports is in its own vendor's "
            "namespace and is NOT comparable to the NVIDIA minimum 9.)"
        )
        assert (
            torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
        ), "MarlinLoraRunnerCore requires CUDA compute capability >= 9"
        inplace = runner_config.inplace
        routed_scaling_factor = runner_config.routed_scaling_factor

        M, K = hidden_states.shape
        E = quant_info.w13_qweight.shape[0]
        N = quant_info.w2_qweight.shape[1] * 16
        topk = topk_ids.shape[1]
        num_bits = quant_info.weight_bits

        for block_size_m in [8, 16, 32, 48, 64]:
            if M * topk / E / block_size_m < 0.9:
                break

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size_m, E
        )

        workspace = _acquire_workspace(hidden_states.device)

        scalar_type1 = get_scalar_type(num_bits, quant_info.w13_qzeros is not None)
        scalar_type2 = get_scalar_type(num_bits, quant_info.w2_qzeros is not None)

        # Stage 1: Gate/Up (Marlin)
        intermediate_cache1 = torch.empty(
            (M * topk, 2 * N), device=hidden_states.device, dtype=hidden_states.dtype
        )
        intermediate_cache1 = moe_wna16_marlin_gemm(
            hidden_states,
            intermediate_cache1,
            quant_info.w13_qweight,
            None,
            quant_info.w13_scales,
            None,
            quant_info.w13_qzeros,
            quant_info.w13_g_idx,
            quant_info.w13_g_idx_sort_indices,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=quant_info.expert_map is not None,
            b_q_type=scalar_type1,
            size_m=M,
            size_n=2 * N,
            size_k=K,
            is_k_full=quant_info.is_k_full,
            use_atomic_add=True,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

        # Hook: after gate_up
        if hooks.after_gate_up:
            intermediate_cache1_3d = intermediate_cache1.view(M, topk, 2 * N)
            hooks.after_gate_up(
                hidden_states, intermediate_cache1_3d, topk_weights, topk_ids
            )

        # Stage 2: Activation
        intermediate_cache2 = torch.empty(
            (M * topk, N), device=hidden_states.device, dtype=hidden_states.dtype
        )
        silu_and_mul(intermediate_cache1.view(-1, 2 * N), intermediate_cache2)

        # Stage 3: Down (Marlin)
        intermediate_cache3 = torch.empty(
            (M * topk, K), device=hidden_states.device, dtype=hidden_states.dtype
        )
        if quant_info.expert_map is not None:
            intermediate_cache3.zero_()

        intermediate_cache3 = moe_wna16_marlin_gemm(
            intermediate_cache2,
            intermediate_cache3,
            quant_info.w2_qweight,
            None,
            quant_info.w2_scales,
            None,
            quant_info.w2_qzeros,
            quant_info.w2_g_idx,
            quant_info.w2_g_idx_sort_indices,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=True,
            is_ep=quant_info.expert_map is not None,
            b_q_type=scalar_type2,
            size_m=M * topk,
            size_n=K,
            size_k=N,
            is_k_full=quant_info.is_k_full,
            use_atomic_add=True,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
        intermediate_cache3 = intermediate_cache3.view(M, topk, K)

        # Hook: after down
        if hooks.after_down:
            hooks.after_down(
                intermediate_cache2, intermediate_cache3, topk_weights, topk_ids
            )

        # Stage 4: Reduction
        output = hidden_states if inplace else torch.empty_like(hidden_states)
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0
        # NOTE: fusion opportunity here
        moe_sum_reduce_triton(intermediate_cache3, output, routed_scaling_factor)

        return StandardCombineInput(hidden_states=output)
