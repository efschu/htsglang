from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )

MARLIN_MOE_WORKSPACE: Optional[torch.Tensor] = None


@dataclass
class MarlinRunnerInput(RunnerInput):
    """Input bundle passed to the Marlin runner core."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinRunnerOutput(RunnerOutput):
    """Output bundle returned from the Marlin runner core."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinMoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by the Marlin backend."""

    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    w13_g_idx_sort_indices: Optional[torch.Tensor]
    w2_g_idx_sort_indices: Optional[torch.Tensor]
    weight_bits: int

    # FP8 (e4m3) weight-only fallback for GPUs without native FP8 compute.
    # Distinguishes 8-bit FP8 weights from 8-bit integer (uint8b128) weights.
    weight_is_fp8: bool = False

    # GPTQ specific (Optional)
    w13_g_idx: Optional[torch.Tensor] = None
    w2_g_idx: Optional[torch.Tensor] = None
    is_k_full: bool = True

    # AWQ specific (Optional)
    w13_qzeros: Optional[torch.Tensor] = None
    w2_qzeros: Optional[torch.Tensor] = None

    # Optional
    expert_map: Optional[torch.Tensor] = None
    global_num_experts: int = -1
    w13_global_scale: Optional[torch.Tensor] = None
    w2_global_scale: Optional[torch.Tensor] = None
    w13_bias: Optional[torch.Tensor] = None
    w2_bias: Optional[torch.Tensor] = None


@register_fused_func("none", "marlin")
def fused_experts_none_to_marlin(
    dispatch_output: StandardDispatchOutput,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    global MARLIN_MOE_WORKSPACE
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    if runner_config.is_gated:
        assert runner_config.activation == "silu", "Only gated SiLU is supported."
    elif runner_config.activation not in {"silu", "relu2"}:
        raise ValueError(
            f"Unsupported Marlin MoE activation: {runner_config.activation}"
        )

    if (
        MARLIN_MOE_WORKSPACE is None
        or MARLIN_MOE_WORKSPACE.device != hidden_states.device
    ):
        MARLIN_MOE_WORKSPACE = marlin_make_workspace(
            hidden_states.device, max_blocks_per_sm=4
        )

    marlin_hidden_states = hidden_states
    # Avoid aliasing the MoE input buffer until Marlin output semantics are
    # fully validated across shared-expert and overlap paths.
    marlin_inplace = False
    if (
        quant_info.weight_bits == 4
        and quant_info.w13_qzeros is None
        and quant_info.w2_qzeros is None
        and quant_info.w13_scales.dtype == torch.float8_e8m0fnu
        and quant_info.w2_scales.dtype == torch.float8_e8m0fnu
        and hidden_states.dtype == torch.float16
    ):
        # MXFP4(E8M0) Marlin kernels are only numerically valid on the bf16
        # activation path. The fp16 + E8M0 path is intentionally not generated
        # in sgl-kernel, so upcast activations here and cast the result back.
        marlin_hidden_states = hidden_states.to(torch.bfloat16)
        marlin_inplace = False

    output = fused_marlin_moe(
        hidden_states=marlin_hidden_states,
        w1=quant_info.w13_qweight,
        w2=quant_info.w2_qweight,
        w1_scale=quant_info.w13_scales,
        w2_scale=quant_info.w2_scales,
        gating_output=topk_output.router_logits,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        global_num_experts=quant_info.global_num_experts,
        expert_map=quant_info.expert_map,
        g_idx1=quant_info.w13_g_idx,
        g_idx2=quant_info.w2_g_idx,
        sort_indices1=quant_info.w13_g_idx_sort_indices,
        sort_indices2=quant_info.w2_g_idx_sort_indices,
        w1_zeros=quant_info.w13_qzeros,
        w2_zeros=quant_info.w2_qzeros,
        w1_global_scale=quant_info.w13_global_scale,
        w2_global_scale=quant_info.w2_global_scale,
        w1_bias=quant_info.w13_bias,
        w2_bias=quant_info.w2_bias,
        workspace=MARLIN_MOE_WORKSPACE,
        num_bits=quant_info.weight_bits,
        is_fp8=quant_info.weight_is_fp8,
        is_k_full=quant_info.is_k_full,
        inplace=marlin_inplace,
        routed_scaling_factor=runner_config.routed_scaling_factor,
        clamp_limit=(
            runner_config.gemm1_clamp_limit
            if runner_config.gemm1_alpha is not None
            else runner_config.swiglu_limit
        ),
        gemm1_alpha=runner_config.gemm1_alpha,
        activation=runner_config.activation,
        is_gated=runner_config.is_gated,
    ).to(hidden_states.dtype)

    return StandardCombineInput(
        hidden_states=output,
    )


@register_fused_func("bar1ep", "marlin")
def fused_experts_bar1ep_to_marlin(
    dispatch_output: "DeepEPNormalDispatchOutput",
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> "DeepEPNormalCombineInput":
    """Marlin MoE over tokens the BAR1 expert-parallel dispatch delivered (#374).

    WHY THIS EXISTS. ``MoeRunner`` builds Marlin with ``runner_core = None``
    ("Marlin only supports fused path"), so a Marlin MoE can serve exactly the
    a2a backends that have a fused func registered. Until now that was ``none``
    alone, and #361 measured the consequence: on sm86 BOTH available MoE
    formats hard-wire the Marlin runner (``quantization/fp8.py`` and
    ``gptq_kernels.py`` construct ``MoeRunner(MoeRunnerBackend.MARLIN, ...)``
    unconditionally -- FP8 lands there too because sm86 has no native FP8), so
    ``--moe-a2a-backend bar1ep`` refused at model load with
    "requires a fused func for a2a backend bar1ep, but none is registered".
    The only runner that consumed the dispatch format, deep_gemm, is disabled
    on every card in that rig (sm86 < 90, and sm120 is explicitly excluded).
    That made BAR1-EP a Hopper/SM100 feature by omission rather than by design.

    THE CONTRACT, and why this is a composition and not a kernel. The
    ``DEEPEP_NORMAL`` output this consumes is per-RECEIVED-token, and the
    runner's job on this path is exactly what ``ep_gather`` does for deep_gemm
    (``moe_runner/deep_gemm.py`` post-permute): apply ``topk_weights`` and
    return ONE ROW PER RECEIVED TOKEN, summed over that token's topk slots
    **on this rank only**. Summing across ranks is the combine's job.
    ``fused_marlin_moe`` already returns ``[M, K]`` weighted and reduced over
    topk, so the shapes line up with no new kernel.

    Three details are load-bearing and none of them is obvious:

    * EXPERT IDS ARE LOCAL. bar1ep hands over ids in
      ``[0, num_local_experts)`` and the weights here hold only the local
      experts, so ``global_num_experts`` must be the LOCAL count.
      ``quant_info.global_num_experts`` is the global one and would be wrong;
      ``-1`` makes ``fused_marlin_moe`` take ``w1.shape[0]``, which IS the
      local count. ``expert_map`` stays None for the same reason.
    * ``-1`` MARKS AN UNUSED SLOT, and it must never reach the kernel.
      ``moe_align_block_size`` indexes by expert id, so a negative id is an
      out-of-bounds read -- the #112 exp_idx bound. The slot is pointed at
      expert 0 and its WEIGHT zeroed, which is numerically exact (0 x finite
      = 0) and reproduces ``ep_gather``'s "weight only where expert_id >= 0".
    * ROUTED SCALING IS NOT APPLIED HERE. The Standard post-permute passes
      ``routed_scaling_factor`` into the reduction; the DEEPEP_NORMAL one
      (``ep_gather``) does not, because on this path the factor is applied
      after the combine. Passing it here would scale every EP token twice,
      silently and by a model-dependent constant.
    """
    global MARLIN_MOE_WORKSPACE
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_ids
    topk_weights = dispatch_output.topk_weights

    if runner_config.is_gated:
        assert runner_config.activation == "silu", "Only gated SiLU is supported."
    elif runner_config.activation not in {"silu", "relu2"}:
        raise ValueError(
            f"Unsupported Marlin MoE activation: {runner_config.activation}"
        )

    # An fp8 dispatch carries its scales beside the payload and the Marlin MoE
    # path has no place to put them. Refused by name rather than silently
    # ignored: a dropped scale is a wrong number, not a slower one. bar1ep only
    # quantizes when DeepGEMM is on, and DeepGEMM is what this path exists to
    # stand in for, so in practice this is unreachable -- which is why it is a
    # named error and not a fallback.
    if dispatch_output.hidden_states_scale is not None:
        raise NotImplementedError(
            "bar1ep dispatched fp8 activations with per-token scales, which the "
            "Marlin MoE path cannot consume. Run the dispatcher in bf16 "
            "(--deepep-dispatcher-output-dtype bf16) or use a runner that takes "
            "the scales, i.e. deep_gemm on sm90+."
        )
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(
            f"bar1ep -> Marlin needs bf16 or fp16 activations, got "
            f"{hidden_states.dtype}."
        )

    num_tokens = hidden_states.shape[0]
    if num_tokens == 0:
        # A rank can legitimately receive nothing. Every rank still reaches the
        # combine, so the empty case returns the right shape rather than
        # skipping ahead of its peers.
        return DeepEPNormalCombineInput(
            hidden_states=torch.zeros_like(hidden_states),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    # The #112 bound: point unused slots at a valid expert and zero their weight.
    valid = topk_ids >= 0
    safe_ids = torch.where(valid, topk_ids, torch.zeros_like(topk_ids))
    safe_weights = torch.where(
        valid, topk_weights, torch.zeros_like(topk_weights)
    )

    if (
        MARLIN_MOE_WORKSPACE is None
        or MARLIN_MOE_WORKSPACE.device != hidden_states.device
    ):
        MARLIN_MOE_WORKSPACE = marlin_make_workspace(
            hidden_states.device, max_blocks_per_sm=4
        )

    marlin_hidden_states = hidden_states
    if (
        quant_info.weight_bits == 4
        and quant_info.w13_qzeros is None
        and quant_info.w2_qzeros is None
        and quant_info.w13_scales.dtype == torch.float8_e8m0fnu
        and quant_info.w2_scales.dtype == torch.float8_e8m0fnu
        and hidden_states.dtype == torch.float16
    ):
        # Same reservation as the `none` path: the MXFP4(E8M0) Marlin kernels
        # are only numerically valid on bf16 activations and the fp16 variant
        # is not generated in sgl-kernel. #283 lives in this corner.
        marlin_hidden_states = hidden_states.to(torch.bfloat16)

    output = fused_marlin_moe(
        hidden_states=marlin_hidden_states,
        w1=quant_info.w13_qweight,
        w2=quant_info.w2_qweight,
        w1_scale=quant_info.w13_scales,
        w2_scale=quant_info.w2_scales,
        # The dispatch format carries no router logits, and the reduction does
        # not need them -- topk_weights drives it. But the callee is not
        # indifferent: fused_marlin_moe reads gating_output.shape[0] for its
        # "Number of tokens mismatch" assert and nothing else, so None would
        # AttributeError on the first real call. topk_weights has exactly the
        # leading dimension that assert is checking. test_bar1ep_marlin_fused
        # pins that this remains the ONLY use, so if the parameter ever starts
        # carrying real weight the substitution fails loudly there instead of
        # producing quiet nonsense here.
        gating_output=safe_weights,
        topk_weights=safe_weights,
        topk_ids=safe_ids,
        # -1: take the expert count from w1, which holds THIS rank's experts.
        global_num_experts=-1,
        expert_map=None,
        g_idx1=quant_info.w13_g_idx,
        g_idx2=quant_info.w2_g_idx,
        sort_indices1=quant_info.w13_g_idx_sort_indices,
        sort_indices2=quant_info.w2_g_idx_sort_indices,
        w1_zeros=quant_info.w13_qzeros,
        w2_zeros=quant_info.w2_qzeros,
        w1_global_scale=quant_info.w13_global_scale,
        w2_global_scale=quant_info.w2_global_scale,
        w1_bias=quant_info.w13_bias,
        w2_bias=quant_info.w2_bias,
        workspace=MARLIN_MOE_WORKSPACE,
        num_bits=quant_info.weight_bits,
        is_fp8=quant_info.weight_is_fp8,
        is_k_full=quant_info.is_k_full,
        inplace=False,
        # Deliberately NOT runner_config.routed_scaling_factor -- see docstring.
        routed_scaling_factor=None,
        clamp_limit=(
            runner_config.gemm1_clamp_limit
            if runner_config.gemm1_alpha is not None
            else runner_config.swiglu_limit
        ),
        gemm1_alpha=runner_config.gemm1_alpha,
        activation=runner_config.activation,
        is_gated=runner_config.is_gated,
    ).to(hidden_states.dtype)

    return DeepEPNormalCombineInput(
        hidden_states=output,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
