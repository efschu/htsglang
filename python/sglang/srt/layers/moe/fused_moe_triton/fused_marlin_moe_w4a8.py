"""Fused Marlin MoE with int8 activations (W4A8), H88-B (2026-10-07).

New module next to ``fused_marlin_moe.py``; the A16 function is not touched.
Differences to ``fused_marlin_moe``:

* the activations of BOTH grouped GEMMs are quantised dynamically per token to
  int8 (``int8_kernel.per_token_quant_int8``): the hidden states before GEMM 1
  and the SwiGLU output before GEMM 2;
* the per-token scales are multiplied by the per-layer-tensor factor
  ``max|group_scale| / 4096`` that belongs to the int16 x4096 group scales
  (``marlin_w4a8_utils.marlin_act_int8_process_scales``); ``None`` for channelwise
  scales, exactly as vLLM does it (oracle/int_wna16.py);
* the kernel is ``moe_wna16_marlin_gemm_w4a8`` (H88-A). It has no ``is_ep``
  switch: ``expert_ids`` must not contain -1, and the block sizes are
  {16, 32, 48, 64} (no 8);
* atomic add is never used (vLLM marlin_moe does the same: use_atomic_add=False;
  the fp32 reduce buffer stays on).

-1 in ``topk_ids`` (padded tokens of a DP-attention batch, tokens routed to a
non-local expert) makes ``moe_align_block_size`` emit blocks with expert id -1.
The A16 kernel skips them; the A8 kernel would read expert -1. Without a host
sync (CUDA-graph safe) this module clamps the block expert ids to >= 0 and sets
the routing weight of every -1 entry to 0, so that the entry's GEMM-2 row is
multiplied by zero. The activation row of such an entry is a real (finite)
token row, so 0 * finite = 0; only a padded token's own, discarded row can carry
garbage.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()

#: block sizes with an int8-activation MoE kernel (marlin_w4a8_utils.W4A8_MOE_BLOCK_SIZES)
W4A8_MOE_BLOCK_SIZES = (16, 32, 48, 64)


def select_block_size_m(num_tokens: int, topk: int, num_experts: int) -> int:
    """The A16 rule (``for bs in [8,16,32,48,64]: if M*topk/E/bs < 0.9: break``)
    restricted to the sizes the int8 kernel has. Pure python, tested on CPU."""
    block_size_m = W4A8_MOE_BLOCK_SIZES[-1]
    for bs in W4A8_MOE_BLOCK_SIZES:
        block_size_m = bs
        if num_tokens * topk / num_experts / bs < 0.9:
            break
    return block_size_m


def sanitize_routing_for_a8(
    expert_ids: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
):
    """No host sync. Returns (expert_ids >= 0, topk_weights with 0 where topk_ids == -1)."""
    expert_ids = expert_ids.clamp(min=0)
    topk_weights = torch.where(
        topk_ids >= 0, topk_weights, torch.zeros_like(topk_weights)
    )
    return expert_ids, topk_weights


def scale_a_for_gemm(a_scales: torch.Tensor, factor: Optional[torch.Tensor]) -> torch.Tensor:
    """Kernel a_scales: float32 [rows]; times the per-tensor factor of the int16
    group scales when there are group scales."""
    a_scales = a_scales.reshape(-1).to(torch.float32)
    if factor is not None:
        a_scales = a_scales * factor.to(a_scales.device, torch.float32)
    return a_scales


def _quant_act(x: torch.Tensor, factor: Optional[torch.Tensor]):
    from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8

    x = x if x.is_contiguous() else x.contiguous()
    x_q, x_s = per_token_quant_int8(x)
    return x_q, scale_a_for_gemm(x_s, factor)


def _scalar_type(has_zp: bool):
    from sgl_kernel.scalar_type import scalar_types

    return scalar_types.uint4 if has_zp else scalar_types.uint4b8


@register_custom_op(out_shape="hidden_states")
def fused_marlin_moe_w4a8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_act_factor: Optional[torch.Tensor] = None,
    w2_act_factor: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
    clamp_limit: Optional[float] = None,
    activation: str = "silu",
    is_gated: bool = True,
) -> torch.Tensor:
    """MoE layer, int4 expert weights x int8 activations.

    w1 int32 [E, K/16, 2N*16/8], w2 int32 [E, N/16, K*16/8] in the Marlin W4A8
    layout; w1_scale/w2_scale = int16-x4096 group scales viewed as the model dtype
    (or plain permuted channel scales); w*_act_factor float32 0-d (or None for
    channel scales)."""
    from sglang.jit_kernel.activation import gelu_and_mul, silu_and_mul
    from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm_w4a8
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        swiglu_limit_func,
    )
    from sgl_kernel import moe_sum_reduce

    assert is_gated, "W4A8 Marlin MoE: only gated experts are wired"
    assert activation in ("silu", "gelu"), f"unsupported activation {activation}"
    assert hidden_states.dtype in (torch.float16, torch.bfloat16)
    assert hidden_states.dtype == w1_scale.dtype == w2_scale.dtype, (
        "the int8 kernel writes in the scale dtype: "
        f"hidden {hidden_states.dtype}, w1_scale {w1_scale.dtype}, w2_scale {w2_scale.dtype}"
    )
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert w1.is_contiguous() and w2.is_contiguous(), "expert weights must be contiguous"
    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16
    assert K == w1.shape[1] * 16, "hidden size mismatch w1"
    assert K == w2.shape[2] // 2, "hidden size mismatch w2 (K*16/8 columns)"
    topk = topk_ids.shape[1]
    gemm1_n = 2 * N

    block_size_m = select_block_size_m(M, topk, E)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, E
    )
    expert_ids, topk_weights_k = sanitize_routing_for_a8(
        expert_ids, topk_weights, topk_ids
    )

    if workspace is None:
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        workspace = torch.zeros(sms * 4, dtype=torch.int, device=device)

    has_zp1 = w1_zeros is not None
    has_zp2 = w2_zeros is not None

    a1_q, a1_scales = _quant_act(hidden_states, w1_act_factor)
    intermediate_cache1 = moe_wna16_marlin_gemm_w4a8(
        a1_q,
        a1_scales,
        None,
        w1,
        None,
        w1_scale,
        w1_zeros,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights_k,
        block_size_m,
        topk,
        False,
        _scalar_type(has_zp1),
        M,
        gemm1_n,
        K,
        use_atomic_add=False,
        use_fp32_reduce=True,
    )

    intermediate_cache2 = torch.empty(
        (M * topk, N), device=hidden_states.device, dtype=hidden_states.dtype
    )
    if activation == "silu" and clamp_limit is not None:
        swiglu_limit_func(
            intermediate_cache2, intermediate_cache1.view(-1, gemm1_n), clamp_limit
        )
    elif activation == "silu":
        silu_and_mul(intermediate_cache1.view(-1, gemm1_n), intermediate_cache2)
    else:
        assert clamp_limit is None, "clamp_limit is not supported for gelu"
        gelu_and_mul(intermediate_cache1.view(-1, gemm1_n), intermediate_cache2)

    a2_q, a2_scales = _quant_act(intermediate_cache2, w2_act_factor)
    intermediate_cache3 = moe_wna16_marlin_gemm_w4a8(
        a2_q,
        a2_scales,
        None,
        w2,
        None,
        w2_scale,
        w2_zeros,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights_k,
        block_size_m,
        1,
        True,
        _scalar_type(has_zp2),
        M * topk,
        K,
        N,
        use_atomic_add=False,
        use_fp32_reduce=True,
    ).view(-1, topk, K)

    output = hidden_states if inplace else torch.empty_like(hidden_states)
    moe_sum_reduce(
        intermediate_cache3,
        output,
        1.0 if routed_scaling_factor is None else routed_scaling_factor,
    )
    return output
