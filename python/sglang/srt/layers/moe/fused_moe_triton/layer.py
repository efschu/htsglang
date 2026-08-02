# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/layer.py

import logging
import math
import threading
from enum import Enum
from functools import cached_property
from typing import List, Optional, Tuple

import torch
from torch.nn.parameter import UninitializedParameter

from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.distributed import (
    get_moe_ep_group,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.utils import (
    assert_activation_aligned_shards,
    tp_partition_offset,
    tp_partition_size,
    tp_plan_active,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import (
    MoeRunnerConfig,
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTEPWrapperMethod,
    create_kt_config_from_server_args,
)
from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatcher
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatcher,
)
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
    TopKOutput,
    TopKOutputChecker,
)
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    has_per_rank_fused_shared_slots,
    uses_per_rank_fused_shared_slots,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMxInt4MoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.fp8_utils import quantize_block_fp8_weight_to_mxfp4
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.model_loader.weight_utils import narrow_padded_param_and_loaded_weight
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_hip,
    is_npu,
    print_info_once,
    round_up,
)
from sglang.srt.utils.custom_op import register_custom_op

_is_hip = is_hip()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

#: #391: the GGUF stream-staging latches below (does this layer stage at all,
#: which stagers exist, has the layer boundary been claimed) are read on every
#: arriving expert shard from the weight loaders' thread pool. One process-wide
#: lock covers all of them: every one is a check-then-build that runs at most
#: twice per layer, so the fast path never touches it and there is nothing to
#: gain from a per-layer lock -- while a per-layer lock would itself need to be
#: created by a check-then-build.
_GGUF_STREAM_LATCH_LOCK = threading.Lock()


def _get_deepep_comm_group(a2a_backend):
    group = get_tp_group().device_group

    if a2a_backend.is_mori():
        group = get_tp_group()

    elif a2a_backend.is_bar1ep():
        # Der BAR1-Direktpfad haengt an get_tp_group().barlink_comm -- Peer-
        # Zeiger-Tabelle und Schlitze gibt es nur fuer diese Gruppe. Eine
        # andere Gruppe waere ein Haenger, kein Fehler; deshalb steht dieser
        # Zweig hier und nicht als Sonderfall im Dispatcher.
        group = get_tp_group().device_group

    elif _is_npu:
        group = get_moe_ep_group().device_group

    return group


def create_moe_dispatcher(moe_runner_config: MoeRunnerConfig) -> BaseDispatcher:
    a2a_backend = get_moe_a2a_backend()
    if (
        a2a_backend.is_none()
        or a2a_backend.is_megamoe()
        or a2a_backend.is_ascend_fuseep()
    ):
        # ascend_fuseep bypasses the dispatcher abstraction (see
        # forward_fuseep in hardware_backend/npu/moe/fuseep.py); a
        # StandardDispatcher is created but never invoked.
        return StandardDispatcher(moe_runner_config)
    elif (
        a2a_backend.is_deepep()
        or a2a_backend.is_mooncake()
        or a2a_backend.is_mori()
        or a2a_backend.is_nixl()
    ):
        # bar1ep falls under is_deepep() here: same contract, same arguments,
        # same output format. Which class actually gets built is decided by
        # MaybeTboDeepEPDispatcher -- the single place that constructs one.
        if a2a_backend.is_bar1ep():
            from sglang.srt.layers.moe.token_dispatcher.bar1ep import (
                bar1ep_available,
            )

            ok, reason = bar1ep_available()
            if not ok:
                # No silent fallback: whoever chose bar1ep gets either BAR1
                # or a reason. Falling back to DeepEP or the gloo level
                # would look like BAR1 in the measurement.
                raise NotImplementedError(
                    f"--moe-a2a-backend bar1ep is not available on this "
                    f"machine: {reason}"
                )
        return MaybeTboDeepEPDispatcher(
            group=_get_deepep_comm_group(a2a_backend),
            router_topk=moe_runner_config.top_k,
            permute_fusion=True,
            num_experts=moe_runner_config.num_experts,
            num_local_experts=moe_runner_config.num_local_experts,
            hidden_size=moe_runner_config.hidden_size,
            params_dtype=moe_runner_config.params_dtype,
            deepep_mode=get_deepep_mode(),
            async_finish=True,
            return_recv_hook=True,
        )
    elif a2a_backend.is_flashinfer():
        return FlashinferDispatcher(
            group=get_tp_group().device_group,
            router_topk=moe_runner_config.top_k,
            num_experts=moe_runner_config.num_experts,
            num_local_experts=moe_runner_config.num_local_experts,
            hidden_size=moe_runner_config.hidden_size,
        )
    else:
        raise NotImplementedError(f"Unsupported a2a backend: {a2a_backend}")


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


#: Marlin's fused-MoE thread tile. w13's n (= 2 * I) and w2's k (= I) both
#: have to be multiples of it, which is a weaker requirement than the dense
#: GEMM's min_thread_k = 128.
MOE_MARLIN_TILE = 64


def _activation_vec_units(intermediate_size: int) -> int:
    """Coarsen an otherwise unconstrained intermediate dimension to whole
    activation vectors (#367). Falls back to element granularity when the
    dimension is not a multiple of the vector -- the construction-time guard
    then names the constraint instead of the kernel raising mid-forward."""
    from sglang.srt.distributed.utils import ACTIVATION_VEC_ELEMS

    if intermediate_size % ACTIVATION_VEC_ELEMS == 0:
        return intermediate_size // ACTIVATION_VEC_ELEMS
    return intermediate_size


def moe_uneven_tp_units(intermediate_size: int, quant_config) -> int:
    """Unit count for the expert intermediate dimension under an uneven shard
    plan (`--rank-tp-ratio`).

    The units are the indivisible packets the plan distributes, so they have
    to be as fine as the weight format allows while still landing every cut
    on a boundary the kernels accept.

    When no quantization constrains the dimension the finest LEGAL grain is
    not one element (task #367). The expert intermediate feeds
    ``silu_and_mul`` / ``gelu_and_mul``, whose jit kernel is instantiated on a
    vector width and hard-checks ``hidden_size % kVecSize == 0``
    (elementwise/activation.cuh:168). Element-granular units let the plan cut
    anywhere: the dense MTP draft of Qwen3.5-35B-A3B at ``--rank-tp-ratio
    5,4`` gets 512 -> [284, 228], neither a multiple of the 16-element
    Blackwell vector nor of the 8-element Ampere one, and the boot dies in
    the draft's FIRST MoE forward with "hidden size must be divisible by
    vector size" -- after the weights are loaded and the graphs are captured.

    So the unconstrained lane coarsens to ``ACTIVATION_VEC_ELEMS``, the same
    16-element MLP unit #82 established for the dense side and #353 for the
    INT8 lane. The quantized branches below already return a coarser unit
    (group 32 / block 128 are multiples of 16), so this is the one lane that
    was missing its boundary. Only ONE grain is used for every rank because
    the plan is built before anyone knows which rank lands on which arch, and
    a multiple of 16 is a multiple of 8: one plan, valid on a mixed rig.

    An intermediate size that is not itself a multiple of the vector cannot be
    coarsened at all; the element-granular count is returned and
    ``assert_activation_aligned_shards`` in ``FusedMoE.__init__`` rejects the
    geometry at construction with the constraint named, rather than letting
    the kernel raise at the first forward.

    Units are only consulted under an installed plan, so the even split and
    the no-plan path are unaffected by the coarsening.
    """
    block = getattr(quant_config, "weight_block_size", None) if quant_config else None
    group = getattr(quant_config, "group_size", None) if quant_config else None

    if block and group and int(group) > 0:
        from sglang.srt.layers.quantization.marlin_utils import (
            GPTQ_MARLIN_MIN_THREAD_K,
        )

        if block[0] == math.lcm(int(group), GPTQ_MARLIN_MIN_THREAD_K):
            # A group-quantized config (AWQ / AutoRound) carries no real weight
            # block: its weight_block_size is the group size folded with the
            # DENSE Marlin K tile, and that is coarser than this path needs.
            # Expert ffn sizes are small (A3B 512), so the dense block would
            # leave 4 units for 3+ ranks and distort the ratio. Use the group
            # branch below, which derives the finest grain the experts allow.
            block = None

    if block and intermediate_size % block[0] == 0:
        # Block-quantized experts (FP8 with weight_block_size, GGUF): one unit
        # count divides both the weight grid and the scale grid.
        return intermediate_size // block[0]

    if group and int(group) > 0:
        # Group-quantized MoE (AWQ/GPTQ wna16, e.g. A3B AWQ group=32): shard
        # cuts must land on group boundaries, exactly like the block case —
        # element-granular units would hand ranks non-group-aligned
        # intermediate shards, which MoeWNA16Method.create_weights rejects
        # (group_size >= 32 assert after halving). When the CONFIG group size
        # does not divide the intermediate size (e.g. Gemma-4-26B-A4B: 704
        # with AWQ group 128), mirror MoeWNA16Method.create_weights' halving
        # (128 -> 64 -> 32) so the unit granularity matches the effective
        # runtime group size.
        g = int(group)
        while g >= 32 and intermediate_size % g:
            g //= 2
        if g >= 32:
            return intermediate_size // g
        # No usable group grain left: the dimension is unconstrained by the
        # weight format, so the activation vector is the binding boundary.
        return _activation_vec_units(intermediate_size)

    ct_map = getattr(quant_config, "target_scheme_map", None)
    if ct_map:
        # compressed-tensors group-quantized experts (e.g. Gemma-4-26B-A4B
        # pack-quantized INT4 group 32): shard cuts must land on
        # lcm(group sizes, MOE_MARLIN_TILE) boundaries. The dense-path
        # 128-K-tile coarsening of _group_size_block would be too coarse here
        # (704 % 128 != 0). No group-quantized scheme -> units stay
        # element-granular (fp8/channel schemes have their own paths).
        group_sizes = set()
        for scheme in ct_map.values():
            weights = scheme.get("weights") if isinstance(scheme, dict) else None
            g = getattr(weights, "group_size", None)
            if g and int(g) > 0:
                group_sizes.add(int(g))
        if group_sizes:
            g = math.lcm(*group_sizes, MOE_MARLIN_TILE)
            if intermediate_size % g == 0:
                return intermediate_size // g

    return _activation_vec_units(intermediate_size)


_HOST_SHARD_UNREACHABLE_WARNED = False


def _warn_host_shard_unreachable_once() -> None:
    """Say once why #394 is inert on a disjoint expert shard.

    Silence here would look identical to "no ratio configured", and an operator
    who set the ratio deliberately would be left guessing.
    """
    global _HOST_SHARD_UNREACHABLE_WARNED
    if _HOST_SHARD_UNREACHABLE_WARNED:
        return
    _HOST_SHARD_UNREACHABLE_WARNED = True
    import logging

    logging.getLogger(__name__).info(
        "MoE cold-expert host shard (#394) is INERT on this layer: the #82 "
        "expert-dim shard gives the ranks disjoint expert ranges, so a "
        "delegated cold expert is not relocated to a peer -- it is absent, and "
        "the first token routed to it fails in ExpertResidencyPlanner.resolve "
        "(measured 2026-08-02, V4-Flash TP=3). Delegation needs a shared-memory "
        "host pool or a replicated-expert EP dispatch first. "
        "SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE=1 re-enables it for developing "
        "that mechanism; it is not a performance option."
    )


class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to apply all_reduce on the output of the layer
        quant_config: Quantization configuration.
        inplace: suggestion to compute inplace (modify input activation).
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        top_k: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_presharded_weights: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
        gemm1_alpha: Optional[float] = None,
        gemm1_clamp_limit: Optional[float] = None,
        swiglu_limit: Optional[float] = None,
        use_weight_loader_fused: bool = False,
        with_bias=False,
        routing_method_type: Optional[RoutingMethodType] = None,
        is_gated: bool = True,
        gate_up_interleaved: bool = True,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.layer_id = layer_id
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_fused_shared_experts = num_fused_shared_experts

        self.enable_flashinfer_cutlass_moe = (
            get_moe_runner_backend().is_flashinfer_cutlass()
        )
        self.moe_ep_size = get_parallel().moe_ep_size
        self.moe_ep_rank = get_parallel().moe_ep_rank
        self.moe_tp_size = get_parallel().moe_tp_size
        self.moe_tp_rank = get_parallel().moe_tp_rank

        # For fused shared experts, DeepEP-class and MegaMOE backends use
        # per-rank physical shared slots, while other backends keep fused
        # shared experts as global shared slots. When fusion is disabled,
        # num_fused_shared_experts is 0 and no shared slots are added here.
        if has_per_rank_fused_shared_slots(num_fused_shared_experts):
            num_shared_slots = num_fused_shared_experts * self.moe_ep_size
        else:
            num_shared_slots = num_fused_shared_experts

        assert (num_experts - num_shared_slots) % self.moe_ep_size == 0
        self._num_global_routed = num_experts - num_shared_slots
        self._num_local_routed = self._num_global_routed // self.moe_ep_size
        self.num_local_experts = self._num_local_routed + num_fused_shared_experts
        self._has_fused_shared = num_fused_shared_experts > 0
        self._pending_fp8_shared_weights: dict[tuple[int, str], torch.Tensor] = {}
        self._pending_fp8_shared_scales: dict[tuple[int, str], torch.Tensor] = {}

        # Uneven TP (--rank-tp-ratio): partition the expert intermediate
        # size by the shard plan in whole quant units (see
        # moe_uneven_tp_units), so every rank's share stays aligned for the
        # weight format in use. Without a plan this is the classic even split.
        #
        # The expert weights form the "moe" family
        # (--rank-moe-ratio / SGLANG_UNEVEN_MOE_VECTOR): together with
        # the dense-MLP "mlp" family this is the shiftable weight mass
        # of the KV-pool self-calibration. Without an installed moe
        # vector the family falls back to the base plan (bit-identical
        # to before).
        self.moe_tp_units = moe_uneven_tp_units(intermediate_size, quant_config)
        self.moe_tp_family = "moe"
        # GGUF MoE under an uneven plan: shard along the EXPERT dim instead
        # of the intermediate dim. The intermediate dim of w2 is the ggml
        # QUANTIZED dim, and typical expert ffn sizes (A3B 512, 26B-A4B 704)
        # cannot be cut on ggml block boundaries for 3 ranks (e.g. 512 =
        # two 256er K-blocks). Whole experts have NO quant constraint on
        # dim 0: each rank owns a plan-proportional subset of complete
        # experts (VRAM scales with the plan), foreign topk hits a zero
        # padding expert (see forward_impl remap + materialize), and the
        # existing reduce_results all-reduce combines the disjoint expert
        # contributions. Even TP and no-plan paths are untouched.
        self._gguf_expert_shard = (
            quant_config is not None
            and quant_config.get_name() == "gguf"
            and tp_plan_active(self.moe_tp_size, self.moe_tp_family)
        )
        if self._gguf_expert_shard:
            lo = tp_partition_offset(
                self.num_experts,
                self.moe_tp_size,
                self.moe_tp_rank,
                self.num_experts,
                self.moe_tp_family,
            )
            n_local = tp_partition_size(
                self.num_experts,
                self.moe_tp_size,
                self.moe_tp_rank,
                self.num_experts,
                self.moe_tp_family,
            )
            self._gguf_expert_range = (lo, lo + n_local)
            self.intermediate_size_per_partition = intermediate_size
            # The family's shard unit IS one expert here — expose that to
            # the sizing/calibration machinery (_family_local_stats would
            # otherwise partition the quant-block units (e.g. 512/256 = 2)
            # over the ranks and raise for 2 units < 3 ranks).
            self.moe_tp_units = self.num_experts
            logging.getLogger(__name__).info(
                "GGUF MoE uneven TP: expert-dim sharding active — rank %d "
                "owns experts [%d, %d) of %d (full intermediate %d per "
                "expert).",
                self.moe_tp_rank,
                lo,
                lo + n_local,
                self.num_experts,
                intermediate_size,
            )
        elif tp_plan_active(self.moe_tp_size, self.moe_tp_family):
            # #367: every rank's expert intermediate feeds silu_and_mul /
            # gelu_and_mul, whose kernel hard-checks its vector alignment at
            # the FIRST FORWARD -- after weight load and graph capture. The
            # units above coarsen the splits this planner makes; a split that
            # still lands unaligned (a foreign SGLANG_UNEVEN_MOE_VECTOR, or an
            # intermediate size no unit grain can fix) is rejected HERE, at
            # construction, with the constraint named. The dense MLP side has
            # had this guard since #82; the expert side was the gap this
            # closes.
            assert_activation_aligned_shards(
                intermediate_size,
                self.moe_tp_size,
                self.moe_tp_units,
                self.moe_tp_family,
                what="MoE expert intermediate",
            )
            self.intermediate_size_per_partition = tp_partition_size(
                intermediate_size,
                self.moe_tp_size,
                self.moe_tp_rank,
                self.moe_tp_units,
                self.moe_tp_family,
            )
        else:
            assert intermediate_size % self.moe_tp_size == 0
            self.intermediate_size_per_partition = intermediate_size // self.moe_tp_size
        self.reduce_results = reduce_results
        self.use_presharded_weights = use_presharded_weights

        self.use_triton_kernels = get_moe_runner_backend().is_triton_kernels()

        self.use_flashinfer_trtllm_moe = (
            get_moe_runner_backend().is_flashinfer_trtllm()
            or get_moe_runner_backend().is_flashinfer_trtllm_routed()
        )
        self.use_deep_gemm = get_moe_runner_backend().is_deep_gemm()

        # flashinfer_trtllm kernel requires intermediate_size to be a multiple of 128
        # Pad the intermediate_size_per_partition if necessary
        if (
            self.use_flashinfer_trtllm_moe
            and self.intermediate_size_per_partition % 128 != 0
        ):
            self.intermediate_size_per_partition = round_up(
                self.intermediate_size_per_partition, 128
            )

        self.quant_config = quant_config
        self.use_flashinfer_mxfp4_moe = get_moe_runner_backend().is_flashinfer_mxfp4()
        # TODO maybe we should remove this `if`, since `Mxfp4MoEMethod` does another round-up logic
        if (
            self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.use_flashinfer_mxfp4_moe
        ):
            hidden_size = round_up(hidden_size, 256)
        self.hidden_size = hidden_size

        self.moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            layer_id=layer_id,
            top_k=top_k,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=gemm1_alpha,
            gemm1_clamp_limit=gemm1_clamp_limit,
            swiglu_limit=swiglu_limit,
            is_gated=is_gated,
            routing_method_type=routing_method_type,
            gate_up_interleaved=gate_up_interleaved,
        )

        self.quant_method: Optional[FusedMoEMethodBase] = None
        server_args = get_server_args()
        kt_config = create_kt_config_from_server_args(server_args, layer_id)
        if kt_config is not None:
            if quant_config is not None:
                gpu_method = quant_config.get_quant_method(self, prefix)
            else:
                gpu_method = UnquantizedFusedMoEMethod(self.use_triton_kernels)
            self.quant_method = KTEPWrapperMethod(gpu_method, kt_config)
        else:
            if quant_config is not None:
                self.quant_method = quant_config.get_quant_method(self, prefix)
            if self.quant_method is None:
                self.quant_method = UnquantizedFusedMoEMethod(
                    self.use_triton_kernels,
                    self.use_flashinfer_trtllm_moe,
                    self.use_deep_gemm,
                )
        self.supports_deferred_finalize = (
            envs.SGLANG_ENABLE_MOE_DEFERRED_FINALIZE.get()
            and get_moe_runner_backend().is_flashinfer_trtllm()
            and isinstance(self.quant_method, ModelOptNvFp4FusedMoEMethod)
        )
        print_info_once(
            "FlashInfer TRTLLM MoE deferred finalize is "
            f"{'enabled' if self.supports_deferred_finalize else 'disabled'} "
            f"(moe_runner_backend={server_args.moe_runner_backend}, "
            f"quant_method={type(self.quant_method).__name__})."
        )

        self.quant_method.create_weights(
            layer=self,
            num_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=(
                self.weight_loader
                if not use_weight_loader_fused
                else self.weight_loader_fused
            ),
            with_bias=with_bias,
            moe_intermediate_size=intermediate_size,
        )

        self.quant_method.create_moe_runner(self, self.moe_runner_config)
        self.dispatcher = create_moe_dispatcher(self.moe_runner_config)

        # --- MoE expert-offload (feat/moe-expert-offload, M-B/M-C) -----------
        # Fully gated: unless SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0 (offload)
        # or SGLANG_MOE_OFFLOAD_TRACE is set (routing trace for M-C), the only
        # cost on the default path is a single bool read in run_moe_core -> the
        # MoE math stays byte-identical. The offload cache itself is installed
        # lazily on the first forward (weights must already be loaded and
        # processed by then); see _apply_expert_offload / run_moe_core.
        from sglang.srt.layers.moe.resident_fraction import (
            resident_fraction_for_rank,
        )

        # SIZING: cached here and later handed to plan_load_time_staging.
        self._expert_offload_fraction = resident_fraction_for_rank()
        self._moe_offload_trace_path = envs.SGLANG_MOE_OFFLOAD_TRACE.get()
        self._moe_offload_enabled = self._expert_offload_fraction < 1.0 or bool(
            self._moe_offload_trace_path
        )
        self._expert_offload = None  # MoEExpertOffloadCache, lazily installed
        self._expert_offload_install_failed = False
        self._moe_offload_trace_step = 0
        # CUDA-graph guard -> MODE SWITCH (Stage-3). The eager offload path
        # (run_waves) does a data-dependent device->host sync per forward
        # (topk_ids.tolist()) and is illegal during CUDA-graph capture. With
        # SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1 (opt-in) decode instead takes the
        # capturable path: frozen residency + on-device index math
        # (prepare_capturable) + a captured UVA gather -- no host dependency,
        # so the decode graph is allowed. Without the opt-in, keep the original
        # fail-fast guard requiring --disable-cuda-graph.
        #
        # #452: that opt-in is REFUTED on hardware (B2 content divergence, B4
        # 6.60x decode regression) and refuses by name here -- at the one point
        # where the mode is selected, before any weight is staged. The mechanism
        # stays in-tree behind the refusal; SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE=1
        # re-opens it for a card window. Nothing on the default path reaches the
        # call: it needs fraction < 1.0 AND the opt-in env.
        from sglang.srt.layers.moe.offload_capture_gate import resolve_graph_mode

        self._moe_offload_graph_mode = resolve_graph_mode(
            self._expert_offload_fraction,
            envs.SGLANG_MOE_OFFLOAD_CUDA_GRAPH.get(),
            self.layer_id,
        )
        if self._moe_offload_enabled:
            try:
                _disable_cg = bool(get_server_args().disable_cuda_graph)
            except Exception:
                _disable_cg = True  # server args not wired (unit/test context)
            if not _disable_cg and not self._moe_offload_graph_mode:
                raise RuntimeError(
                    "MoE expert-offload / routing-trace "
                    "(SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0 or "
                    "SGLANG_MOE_OFFLOAD_TRACE) requires --disable-cuda-graph: "
                    "the per-forward expert residency plan is data-dependent and "
                    "cannot be captured into a CUDA graph. Re-launch with "
                    "--disable-cuda-graph, or opt in to the capturable offload "
                    "decode path with SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1 "
                    "(requires offload fraction < 1.0)."
                )
            if self._moe_offload_graph_mode and envs.SGLANG_MOE_HOT_RESIDENCY.get():
                # §5 fail-fast: live calibration cannot be frozen pre-capture.
                raise RuntimeError(
                    "SGLANG_MOE_HOT_RESIDENCY (live hot calibration) cannot be "
                    "combined with SGLANG_MOE_OFFLOAD_CUDA_GRAPH: the residency "
                    "layout must be frozen BEFORE graph capture. Supply "
                    "SGLANG_MOE_HOTSET_FILE or use static residency."
                )

        # #390: the router/residency counters live in the offload path, so
        # asking for them without an offload records nothing. Say so at boot
        # rather than leaving an empty dump to be explained later.
        if self._expert_offload_fraction >= 1.0 and envs.SGLANG_EXPERT_STATS.get():
            from sglang.srt.layers.moe.expert_stats import (
                warn_if_stats_without_offload,
            )

            warn_if_stats_without_offload()

        self._use_ascend_fuseep = get_moe_a2a_backend().is_ascend_fuseep()

        if (
            get_moe_runner_backend().is_flashinfer_trtllm_routed()
            or get_moe_runner_backend().is_flashinfer_trtllm()
        ):
            if self.moe_runner_config.inplace:
                print_info_once(
                    "Setting inplace to False for FlashInfer TRTLLM MoE backend."
                )
            self.moe_runner_config.inplace = False

        self.should_fuse_routed_scaling_factor_in_topk = (
            isinstance(self.quant_method, ModelOptNvFp4FusedMoEMethod)
            or (
                isinstance(self.quant_method, Fp8MoEMethod)
                and (
                    get_moe_runner_backend().is_cutlass()
                    or get_moe_runner_backend().is_flashinfer_trtllm_routed()
                )
            )
            or (
                isinstance(self.quant_method, UnquantizedFusedMoEMethod)
                and get_moe_runner_backend().is_flashinfer_trtllm_routed()
            )
        )

        self.routing_method_type = routing_method_type

        # overlap args
        self.down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

        if self.quant_method is not None and hasattr(self.quant_method, "runner"):
            self.runner = self.quant_method.runner

    @cached_property
    def use_padded_loading(self) -> bool:
        # This handles the case where the loaded weights are smaller than the padded expert_data
        # Use narrow_padded_param_and_loaded_weight for:
        # 1. CPU (always)
        # 2. GPU with flashinfer_trtllm padding (when intermediate_size is padded to 128)
        # 3. GPU with Aiter padding
        aiter_padded = (
            _use_aiter
            and hasattr(self, "w2_weight")
            and getattr(self.w2_weight, "weight_padded", False)
        )

        return _is_cpu or self.use_flashinfer_trtllm_moe or aiter_padded

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            if self.moe_runner_config.is_gated:
                param_data[expert_id][idx] = loaded_weight
            else:
                param_data[expert_id] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):
        # Load grouped weight scales for group quantization
        # or model weights
        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )
        elif shard_id in ("w1", "w3", "w13"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        assert shard_id in {"w1", "w3", "w13"}

        if is_bias:
            # if this weight is a bias, the last dimension must be the sharded dimension
            shard_dim = -1

        if shard_id in {"w1", "w3"} and self.moe_runner_config.is_gated:
            # non-fused version
            shard_size = expert_data.shape[shard_dim] // 2
        elif shard_id in {"w13"} or (
            shard_id in {"w1", "w3"} and not self.moe_runner_config.is_gated
        ):
            # fused version
            shard_size = expert_data.shape[shard_dim]
        else:
            raise NotImplementedError

        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        # w3, up_proj: Load into second logical weight of w13.
        # trtllm cutlass kernel assumes differently
        switch_w13 = getattr(self.quant_method, "load_up_proj_weight_first", False)
        if (
            (switch_w13 and shard_id == "w1") or (not switch_w13 and shard_id == "w3")
        ) and self.moe_runner_config.is_gated:
            start = shard_size
        else:
            start = 0

        if self.use_padded_loading:
            if _is_cpu and is_bias:
                shard_dim = 1
            expert_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                expert_data,
                loaded_weight,
                start,
                self._moe_src_start(
                    loaded_weight.shape[shard_dim], shard_size, tp_rank
                ),
                shard_dim,
                shard_size,
                not self.use_presharded_weights,
            )
        else:
            if not self.use_presharded_weights:
                if not is_bias and self.use_triton_kernels:
                    # do not transpose for bias
                    loaded_weight = loaded_weight.transpose(-2, -1)
                loaded_weight = loaded_weight.narrow(
                    shard_dim,
                    self._moe_src_start(
                        loaded_weight.shape[shard_dim], shard_size, tp_rank
                    ),
                    shard_size,
                )

            expert_data = expert_data.narrow(shard_dim, start, shard_size)
        expert_data.copy_(loaded_weight)

    def _moe_src_start(self, loaded_total: int, shard_size: int, tp_rank: int) -> int:
        """Source-side start of this rank's shard in a full checkpoint tensor.

        INVARIANT: the source-side start belongs to the CHECKPOINT tensor's
        real geometry and must never be derived from the DESTINATION
        ``shard_size``. The two are equal only while nothing pads the
        destination, which is why ``shard_size * tp_rank`` survived so long.

        Upstream sgl-project/sglang#32781 is that invariant broken on the
        padded path: Marlin rounds a TP16 shard of 3072/16 = 192 up to 256, so
        rank 13 asked the 3072-wide checkpoint tensor for a narrow starting at
        256 * 13 = 3328 and got ``IndexError: start out of range (expected to
        be in range of [-3072, 3072], but got 3328)``. Deriving the start from
        ``loaded_total`` instead keeps every rank in range whatever the
        destination is padded to.

        Even TP: the checkpoint tensor's own even split. Identical to the old
        ``shard_size * tp_rank`` whenever the destination is unpadded, because
        then ``loaded_total == shard_size * tp_size``.
        Uneven TP: prefix sum of the plan partition ("moe" family, falling
        back to the base plan); works for weight AND block-scale grids because
        moe_tp_units divides both -- that branch was already source-derived.
        """
        if not tp_plan_active(self.moe_tp_size, self.moe_tp_family):
            tp_size = self.moe_tp_size
            # getattr: the attribute is always set on a real FusedMoE, but this
            # helper is also driven from lightweight stand-ins that carry only
            # the plan fields.
            presharded = getattr(self, "use_presharded_weights", False)
            if not presharded and tp_size > 0 and loaded_total % tp_size == 0:
                return (loaded_total // tp_size) * tp_rank
            # Presharded checkpoints hand each rank its own tensor (there is no
            # full-tensor geometry to read), and a source length the TP size
            # does not divide has no even split to speak of. Both keep the
            # legacy expression rather than inventing an offset here.
            return shard_size * tp_rank
        return tp_partition_offset(
            loaded_total,
            self.moe_tp_size,
            tp_rank,
            self.moe_tp_units,
            self.moe_tp_family,
        )

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):
        """Load w2 weights for down projection.

        Args:
            expert_data: The expert data tensor to load into
            shard_dim: The dimension to shard along
            shard_id: The shard ID (must be "w2")
            loaded_weight: The weight tensor to load from
            tp_rank: The tensor parallel rank
        """
        if not isinstance(expert_data, torch.Tensor) or not isinstance(
            loaded_weight, torch.Tensor
        ):
            raise ValueError("expert_data and loaded_weight must be torch.Tensor")

        if (
            self.quant_config is not None
            and "modelopt" in self.quant_config.get_name()
            and (expert_data.dim() != 2 or loaded_weight.dim() != 2)
        ):
            raise ValueError(
                f"Expected 2D tensors, got expert_data shape {expert_data.shape} and loaded_weight shape {loaded_weight.shape}"
            )

        if shard_id != "w2":
            raise ValueError(f"shard_id must be 'w2', got {shard_id}")

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        if is_bias:
            # this expert_data is a bias, not weight,
            # for w2_weight_bias in TP, it does not need to be sharded
            shard_size = expert_data.shape[-1]
        else:
            # this parameter is a weight matrix
            # for w2 in TP, it shards the input_features, i.e., shard_dim=2
            shard_size = expert_data.shape[shard_dim]

        if self.use_padded_loading:
            if _is_cpu and is_bias:
                shard_dim = 1
            expert_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                expert_data,
                loaded_weight,
                0,  # param_data_start
                self._moe_src_start(
                    loaded_weight.shape[shard_dim], shard_size, tp_rank
                ),
                shard_dim,
                shard_size,
                not self.use_presharded_weights,
            )
        else:
            if not is_bias and not self.use_presharded_weights:
                if self.use_triton_kernels:
                    loaded_weight = loaded_weight.transpose(-2, -1)
                loaded_weight = loaded_weight.narrow(
                    shard_dim,
                    self._moe_src_start(
                        loaded_weight.shape[shard_dim], shard_size, tp_rank
                    ),
                    shard_size,
                )

        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _maybe_load_fp8_shared_expert_as_fp4(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        shard_dim: int,
        tp_rank: int,
    ) -> bool:
        if (
            not self._has_fused_shared
            or expert_id < self._num_local_routed
            or self.quant_config is None
            or not getattr(self.quant_config, "is_fp4_experts", False)
            or shard_id not in ("w1", "w2", "w3")
        ):
            return False

        is_weight = (
            "weight" in weight_name
            and "scale" not in weight_name
            and loaded_weight.dtype == torch.float8_e4m3fn
        )
        is_scale = "weight_scale_inv" in weight_name and loaded_weight.dtype in (
            torch.float8_e8m0fnu,
            torch.float32,
        )
        if not is_weight and not is_scale:
            return False

        weight_param = self.w2_weight if shard_id == "w2" else self.w13_weight
        scale_param = (
            self.w2_weight_scale_inv if shard_id == "w2" else self.w13_weight_scale_inv
        )
        if param is not weight_param and param is not scale_param:
            return False

        key = (expert_id, shard_id)
        if is_weight:
            fp8_weight = loaded_weight
            fp8_scale = self._pending_fp8_shared_scales.pop(key, None)
            if fp8_scale is None:
                self._pending_fp8_shared_weights[key] = loaded_weight
                return True
        else:
            fp8_weight = self._pending_fp8_shared_weights.pop(key, None)
            fp8_scale = loaded_weight
            if fp8_weight is None:
                self._pending_fp8_shared_scales[key] = loaded_weight
                return True

        logging.getLogger(__name__).warning_once(
            "Loading FP8 shared expert weights into FP4 fused MoE weights. "
            "The shared expert is quantized at load time and may differ "
            "slightly from a checkpoint that stores shared experts directly "
            "in FP4."
        )

        weight_block_size = getattr(self.quant_config, "weight_block_size", None)
        if weight_block_size is None:
            raise ValueError(
                "Loading FP8 shared expert weights into FP4 fused MoE weights "
                "requires block-FP8 weight_block_size."
            )
        fp4_weight, fp4_scale = quantize_block_fp8_weight_to_mxfp4(
            fp8_weight, fp8_scale, weight_block_size
        )

        weight_data = weight_param.data[expert_id]
        scale_data = scale_param.data[expert_id]
        self._load_model_weight_or_group_weight_scale(
            shard_dim=shard_dim,
            expert_data=weight_data,
            shard_id=shard_id,
            loaded_weight=fp4_weight,
            tp_rank=tp_rank,
        )
        self._load_model_weight_or_group_weight_scale(
            shard_dim=shard_dim,
            expert_data=scale_data,
            shard_id=shard_id,
            loaded_weight=fp4_scale,
            tp_rank=tp_rank,
        )
        return True

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        start_idx = self.moe_ep_rank * self._num_local_routed
        end_idx = start_idx + self._num_local_routed
        if start_idx <= expert_id < end_idx:
            return expert_id - start_idx
        elif self._has_fused_shared and expert_id >= self._num_global_routed:
            return expert_id - self._num_global_routed + self._num_local_routed
        else:
            return -1

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: Optional[int],
    ) -> None:
        # if expert_id is None, then
        # all the experts are loaded at the same time
        if (
            not expert_id
            and self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.quant_config.is_static_cfg()
        ):
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return

        global_expert_location_metadata = get_global_expert_location_metadata()
        if global_expert_location_metadata is None:
            if not getattr(param, "_sglang_require_global_experts", False):
                expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
                if expert_id == -1:
                    return

            self._weight_loader_impl(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            return

        require_global_experts = getattr(param, "_sglang_require_global_experts", False)
        shared_expert_id = (
            expert_id - global_expert_location_metadata.num_logical_experts
            if self._has_fused_shared and expert_id is not None
            else -1
        )
        if 0 <= shared_expert_id < self.num_fused_shared_experts:
            # Checkpoint shared experts start after logical routed experts, while
            # local fused MoE weights store them after physical routed experts.
            if require_global_experts and uses_per_rank_fused_shared_slots():
                physical_expert_ids = [
                    rank * self.num_local_experts
                    + self._num_local_routed
                    + shared_expert_id
                    for rank in range(self.moe_ep_size)
                ]
            else:
                physical_expert_ids = [self._num_global_routed + shared_expert_id]
        else:
            physical_expert_ids = (
                global_expert_location_metadata.logical_to_all_physical(
                    self.layer_id, expert_id, require_global_experts
                )
            )

        for physical_expert_id in physical_expert_ids:
            self._weight_loader_physical(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=physical_expert_id,
            )

    def _weight_loader_physical(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        # WARN: This makes the `expert_id` mean "local" and "global" in different cases
        if not getattr(param, "_sglang_require_global_experts", False):
            expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
            if expert_id < 0 or expert_id >= self.num_local_experts:
                return

        if isinstance(
            self.quant_method,
            KTEPWrapperMethod,
        ):
            if self.quant_method.num_gpu_experts != -1:
                if expert_id >= self.quant_method.num_gpu_experts:
                    return

        self._weight_loader_impl(
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
        )

    def _load_gguf_weight(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str,
        expert_id: int,
        tp_rank: int,
    ) -> bool:
        """Handle GGUF weight loading.

        Args:
            param: The parameter to load the weight into.
            loaded_weight: The weight tensor to load.
            shard_id: The shard ID (w1, w2, or w3).
            expert_id: The expert ID.
            tp_rank: The tensor parallel rank.

        Returns:
            True if the weight was handled as a GGUF weight, False otherwise.
        """
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)

        if is_gguf_weight_type:
            # Store weight type for this expert
            param.weight_type = loaded_weight.item()
            return True

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            if getattr(self, "_gguf_expert_shard", False):
                # Uneven-TP expert-dim sharding: keep the FULL per-expert
                # tensor, but only for experts this rank owns (see
                # __init__); foreign experts are dropped here and served
                # by their owner rank.
                lo, hi = self._gguf_expert_range
                if not (lo <= expert_id < hi):
                    return True
            elif self.moe_tp_size > 1:
                if shard_id in ["w1", "w3", "w2"] and output_dim == 0:
                    shard_size = loaded_weight.size(0) // self.moe_tp_size
                    start_idx = tp_rank * shard_size
                    loaded_weight = loaded_weight.narrow(
                        0, start_idx, shard_size
                    ).clone()

            # #391c: on an offload-covered layer this expert goes straight into
            # its tier and is never held. Everything below accumulates.
            if self._gguf_stream_stage(param, loaded_weight, shard_id, expert_id):
                return True

            # Store in data_container with expert/shard info
            if not hasattr(param, "expert_data_map"):
                param.expert_data_map = {}

            key = (expert_id, shard_id)
            param.expert_data_map[key] = loaded_weight
            param.data_container.append(loaded_weight)
            return True

        return False

    # -- #391c: streaming staging into the expert-offload tiers -------------
    #
    # The GGUF weight iterator's first pass yields EVERY qweight_type marker
    # before its second pass yields a single payload byte (weight_utils.py,
    # gguf_quant_weights_iterator), and the MXFP4->Q5_0 repack rewrites both the
    # marker and the payload inside that iterator. So by the time the first
    # expert tensor reaches the callback below, ``w13_qweight_type`` and
    # ``w2_qweight_type`` already hold the POST-repack ggml types and
    # ``_gguf_moe_offload_eligible`` can answer for real -- which is what makes
    # the residency decision available at load time rather than at
    # process_weights_after_loading.

    #: Which expert-major parameter a GGUF MoE shard belongs to.
    _GGUF_SHARD_TO_ATTR = {"w1": "w13_qweight", "w3": "w13_qweight", "w2": "w2_qweight"}

    def _gguf_stream_staging_enabled(self) -> bool:
        """Latched: does this layer stage from the stream? Decided once.

        Double-checked: the weight loaders run on a thread pool (#391), so the
        first shards of one layer arrive concurrently and would otherwise run
        this latch several times over -- each run REPLACING
        ``_gguf_stream_stagers`` with a fresh dict and losing whatever the
        earlier thread had already staged into it. The fast path is a plain
        attribute read; only a latch miss takes the lock.
        """
        cached = getattr(self, "_gguf_stream_staging_on", None)
        if cached is not None:
            return cached
        from sglang.srt.environ import envs

        with _GGUF_STREAM_LATCH_LOCK:
            cached = getattr(self, "_gguf_stream_staging_on", None)
            if cached is not None:
                return cached
            on = (
                bool(envs.SGLANG_MOE_GGUF_STREAM_STAGING.get())
                and self._gguf_moe_offload_eligible()
            )
            if on:
                self._gguf_stream_stagers = {}
            # Published last: a fast-path reader that sees True must find the
            # registry already there.
            self._gguf_stream_staging_on = on
            return on

    def _gguf_owned_expert_count(self, param) -> int:
        """Expert count the plan is built over -- the same one the pull path uses.

        Under the #82 expert-dim shard that is the owned range plus the trailing
        all-zero padding expert; otherwise the parameter's declared expert
        dimension, exactly as ``materialize_gguf_weights`` reads it.
        """
        if getattr(self, "_gguf_expert_shard", False):
            lo, hi = self._gguf_expert_range
            return int(hi) - int(lo) + 1
        return int(getattr(param, "tensor_shape")[0])

    def _gguf_local_expert_index(self, expert_id: int) -> int:
        """Stream expert id -> the local index the tiers are addressed by.

        ``_gguf_expert_source`` numbers experts by their position in the
        ascending list of ids this rank actually received; under the expert-dim
        shard that is ``expert_id - lo``, and everywhere else the id itself.
        """
        if getattr(self, "_gguf_expert_shard", False):
            lo, _hi = self._gguf_expert_range
            return int(expert_id) - int(lo)
        return int(expert_id)

    def _gguf_cold_shard_context(self):
        """#394: this rank's share of the cold pool, or ``None``.

        REFUSED BY DEFAULT, and the reason is a measurement rather than
        caution. Booted on the reference rig 2026-08-02 (V4-Flash UD-IQ3_XXS,
        TP=3): all 43 layers staged, the ratio resolved, and every rank died on
        the first forward inside ``ExpertResidencyPlanner.resolve`` --
        "experts [80, 83, 94] were delegated to a peer rank's host tier ... but
        this rank's router asked for them".

        The premise does not hold here. ``partition_cold_experts`` keeps this
        rank's share of ITS OWN cold experts and drops the rest. Under the #82
        GGUF expert-dim shard the ranks hold DISJOINT expert ranges, so no peer
        holds rank 0's expert 80: a delegated expert is not relocated, it is
        absent, and the first token routed to it has nowhere to go. Delegation
        is sound only when a delegated expert stays REACHABLE -- either the
        pinned host pools are shared memory so a rank can DMA out of a peer's
        pool (they are all host DRAM already, so this is a mapping question),
        or the experts are replicated with an EP-style dispatch to the owner.

        So ``_gguf_expert_shard`` is not the eligibility test it was documented
        as: it is TRUE exactly when experts are sharded disjointly, i.e. exactly
        when delegation is UNSOUND. Until a reachability mechanism exists, the
        honest answer is ``None`` -- the pre-#394 plan, field for field. The
        escape hatch below exists so the mechanism can be developed against a
        real boot; it is not a performance option, and it reproduces the crash
        above on this model.

        The per-rank card UUIDs come from the vector the LAUNCHER published
        (:mod:`sglang.srt.registry.rank_cards`, #407 cut 2) -- an environment
        read, not a collective. That distinction is the whole reason the vector
        exists: this method runs inside the weight-load loop, and a group
        collective there is the rank-local-before-group hazard (a rank that
        reaches the load path on a different schedule hangs the group with no
        diagnosis). An absent vector is not an error; the ratio then resolves
        from ``SGLANG_MOE_HOST_SHARD_RATIO`` or not at all, and without either
        this returns ``None`` and the plan is the pre-#394 plan field for field.

        The vector is length-checked against THIS group's world size inside
        ``rank_card_vector``: a MoE-TP subgroup narrower than the serving group
        would otherwise read another rank's card as its own.
        """
        if not getattr(self, "_gguf_expert_shard", False):
            return None
        world = int(getattr(self, "moe_tp_size", 1) or 1)
        if world < 2:
            return None

        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.cold_tier_fetch import cold_tier_enabled

        # #394 slice 2: the shared cold tier IS the reachability mechanism the
        # refusal above names as missing. With it on, a delegated expert lives
        # in a peer's named segment and any rank can DMA its row, so the
        # premise of the refusal no longer holds. With it off the refusal is
        # unchanged, because without a shared tier the expert really is absent.
        if (
            not cold_tier_enabled()
            and not envs.SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE.get()
        ):
            _warn_host_shard_unreachable_once()
            return None

        from sglang.srt.layers.moe.expert_offload import cold_shard_context
        from sglang.srt.registry.rank_cards import rank_card_uuids

        return cold_shard_context(
            int(self.moe_tp_rank), world, card_uuids=rank_card_uuids(world)
        )

    def _gguf_stream_stage(self, param, loaded_weight, shard_id, expert_id) -> bool:
        """Route one arriving expert shard into its tier. True when consumed."""
        if not self._gguf_stream_staging_enabled():
            return False
        attr = self._GGUF_SHARD_TO_ATTR.get(shard_id)
        if attr is None or getattr(self, attr, None) is not param:
            return False
        stagers = self._gguf_stream_stagers
        stager = stagers.get(attr)
        if stager is None:
            # Same double-check as the latch above: gate and up of one layer
            # both route into ``w13_qweight`` and arrive on two loader threads,
            # so an unguarded build lets the second stager overwrite the first
            # and the shards already taken by the first are simply lost (#391).
            with _GGUF_STREAM_LATCH_LOCK:
                stager = stagers.get(attr)
                if stager is None:
                    stager = self._new_gguf_stream_stager(attr, param)
                    if stager is None:
                        return False
                    stagers[attr] = stager
        stager.submit(self._gguf_local_expert_index(expert_id), shard_id, loaded_weight)
        self._maybe_close_gguf_stream_layer()
        return True

    def _maybe_close_gguf_stream_layer(self) -> None:
        """Layer boundary DURING the stream: trim the allocator, emit the trace.

        Both have to happen here rather than at the drain. The trim because
        dropping an expert's last reference is not the same as returning its
        pages -- glibc keeps them in its arena, and over 40+ layers that
        retention is the whole expert set again (#256). The trace because a
        line printed at the drain would print for every layer at once, after
        the load, and could not be lined up against a time-series RAM monitor;
        printed here, the cumulative pinned figure is directly comparable to
        the monitor's anon curve at that instant.

        Publishing the tiers still waits for ``process_weights_after_loading``:
        the stream is only provably over there, and closing early would turn a
        late-arriving expert from a hard error into a silently dropped one.
        """
        if getattr(self, "_gguf_stream_layer_closed", False):
            return
        stagers = getattr(self, "_gguf_stream_stagers", None)
        if not stagers or set(stagers) != {"w13_qweight", "w2_qweight"}:
            return
        if not all(stager.is_complete for stager in stagers.values()):
            return
        from sglang.srt.layers.moe.expert_offload import (
            log_streaming_staging_layer,
            trim_host_allocator,
        )

        # The two stagers can report complete to two loader threads at once
        # (#391); claim the boundary under the lock so the trim and the trace
        # line stay exactly one per layer. The work itself runs outside it.
        with _GGUF_STREAM_LATCH_LOCK:
            if getattr(self, "_gguf_stream_layer_closed", False):
                return
            self._gguf_stream_layer_closed = True
        trim_host_allocator()
        log_streaming_staging_layer(
            f"layer {getattr(self, 'layer_id', '?')}",
            next(iter(stagers.values())).plan,
        )

    def _new_gguf_stream_stager(self, attr: str, param):
        """Build the plan and the stager for one expert-major parameter.

        The plan needs only config-level facts -- expert count, resident
        fraction, the pinned #82 pad expert, the #394 ratio -- so it is
        computable here, BEFORE the first byte, which is the whole difference
        between this and the materialization-time door.
        """
        from sglang.srt.layers.moe.expert_offload import (
            StreamingExpertStager,
            plan_load_time_staging,
        )

        count = self._gguf_owned_expert_count(param)
        expert_shard = getattr(self, "_gguf_expert_shard", False)
        cold_shard = self._gguf_cold_shard_context()
        plan = plan_load_time_staging(
            count,
            # The layer latched its fraction at construction; the cache sizes
            # itself from the SAME value, so the plan must not re-read the env.
            fraction=getattr(self, "_expert_offload_fraction", None),
            pinned_experts=(count - 1,) if expert_shard else (),
            cold_shard=cold_shard,
        )
        if plan is None:
            # Too few experts to split at this fraction: fall back to the
            # accumulate path for the whole layer, not just this tensor.
            self._gguf_stream_staging_on = False
            return None

        def allocate(row_shape, dtype):
            # The loader's flat list is only ever read for its truthiness; not
            # filling it is what lets each expert's bytes go back to the
            # allocator the moment they are copied.
            param.data_container = []
            param.materialize((plan.buffer_slots,) + tuple(row_shape), dtype=dtype)
            return param.data

        return StreamingExpertStager(
            plan,
            ("w1", "w3") if attr.startswith("w13") else ("w2",),
            allocate,
            zero_experts=(count - 1,) if expert_shard else (),
            label=f"layer {getattr(self, 'layer_id', '?')} {attr}",
            cold_tier=self._gguf_cold_tier_owner(plan, cold_shard),
            param_attr=attr,
        )

    def _gguf_cold_tier_owner(self, plan, cold_shard):
        """#394 slice 2: the shared-segment handle for this layer, or ``None``.

        Built once per layer and shared by the two stagers (``w13``/``w2``), so
        one manifest entry describes both tensors and the layer is sealed once.
        The owner map is stashed on the layer here rather than recomputed in
        the cache: it is a pure function of the plan's cold pool and the
        resolved ratio, so computing it twice could only introduce a way for
        the two copies to disagree.
        """
        if plan is None or cold_shard is None or not plan.delegated_ids:
            return None
        from sglang.srt.layers.moe.cold_tier_fetch import (
            assign_cold_experts,
            cold_tier_enabled,
            layer_key_for,
            owner_for_layer,
        )
        from sglang.srt.registry.rank_cards import rank_card_uuids

        if not cold_tier_enabled():
            return None
        existing = getattr(self, "_moe_cold_tier_owner", None)
        if existing is not None:
            return existing
        world = int(self.moe_tp_size)
        # The pool BEFORE the split: the owner map has to cover every cold
        # expert, not just the ones this rank kept.
        cold_pool = tuple(sorted(set(plan.spill_ids) | set(plan.delegated_ids)))
        layer_key = layer_key_for(self)
        self._moe_cold_tier_assignment = assign_cold_experts(
            cold_pool, cold_shard.ratio, int(self.moe_tp_rank), world
        )
        self._moe_cold_tier_layer_key = layer_key
        owner = owner_for_layer(
            layer_key, int(self.moe_tp_rank), world, rank_card_uuids(world)
        )
        self._moe_cold_tier_owner = owner
        return owner

    def _drain_gguf_stream_stagers(self) -> bool:
        """Close the streaming stagers and publish their tiers. True if any.

        Called from ``materialize_gguf_weights``, i.e. at exactly the moment the
        old path would have STARTED building the stack -- by which time this
        path has nothing left to do but hand the finished tiers to the cache.
        Idempotent: the stagers are dropped here, so a second
        ``process_weights_after_loading`` finds nothing to drain and the
        parameters are no longer ``UninitializedParameter``.
        """
        from sglang.srt.layers.moe.expert_offload import (
            log_streaming_staging_layer,
            register_load_time_presplit,
        )

        stagers = getattr(self, "_gguf_stream_stagers", None)
        if not stagers:
            return False
        self._gguf_stream_stagers = {}
        staged_attrs = []
        staged_plan = None
        for attr, stager in stagers.items():
            resident_buf, spill = stager.finalize()
            register_load_time_presplit(self, attr, resident_buf, spill, stager.plan)
            staged_attrs.append(attr)
            staged_plan = stager.plan
            param = getattr(self, attr)
            param.expert_data_map = {}
            param.data_container = []
        # #394 slice 2: every cold row of this layer is now written, so the
        # segment headers can be stamped and the manifest (re)published. Seal
        # LAST, always: before it the magic is zero and a peer that maps the
        # segment early refuses instead of reading a half-filled pool.
        cold_tier_owner = getattr(self, "_moe_cold_tier_owner", None)
        if cold_tier_owner is not None:
            # Dropped before sealing so a second drain (the method is
            # idempotent by contract) cannot re-stamp a live segment.
            self._moe_cold_tier_owner = None
            cold_tier_owner.seal()
        if not getattr(self, "_gguf_stream_layer_closed", False):
            # The layer never reached both-tensors-complete during the stream
            # (a checkpoint with only one expert tensor, say), so its boundary
            # is here. Exactly one trace line per layer either way.
            self._gguf_stream_layer_closed = True
            log_streaming_staging_layer(
                f"layer {getattr(self, 'layer_id', '?')}", staged_plan
            )
        self._finish_gguf_moe_offload_staging(
            staged_attrs, staged_plan, door="load time (streamed)"
        )
        return True

    def _weight_loader_impl(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        tp_rank = self.moe_tp_rank

        # Special case for GGUF weights
        if self._load_gguf_weight(param, loaded_weight, shard_id, expert_id, tp_rank):
            return

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        method = self.quant_method
        if hasattr(self, "scheme"):
            method = self.scheme
        if method.__class__.__name__ == "KTEPWrapperMethod":
            method = method.gpu_method

        # For flashinfer TRT-LLM BF16 path, process_weights_after_loading reshapes
        # expert weights into block layout. During weight update, we must restore
        # canonical load-time shapes before copying checkpoint tensors.
        if isinstance(method, UnquantizedFusedMoEMethod):
            method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
                layer=self,
                param=param,
                weight_name=weight_name,
            )
        elif isinstance(method, Fp8MoEMethod) and (
            get_moe_runner_backend().is_flashinfer_trtllm_routed()
            or get_moe_runner_backend().is_flashinfer_trtllm()
        ):
            # Drop the GPU mxfp8 shuffle-index cache on every reload for mxfp8 trtllm, trtllm_routed
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                clear_mxfp8_shuffle_index_cache,
            )

            clear_mxfp8_shuffle_index_cache()

        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                method.__class__.__name__
                in [
                    "CompressedTensorsWNA16MarlinMoE",
                    "CompressedTensorsWNA16MoE",
                    "CompressedTensorsWNA16TritonMoE",
                ]
            )
            and "zero" not in weight_name
            else loaded_weight
        )

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Flashinfer assumes w31 format for w13_weight. Same for the scales.
        if self.use_flashinfer_trtllm_moe and (
            isinstance(method, ModelOptNvFp4FusedMoEMethod)
            or isinstance(method, Fp8MoEMethod)
            or isinstance(method, UnquantizedFusedMoEMethod)
            or isinstance(method, CompressedTensorsMxInt4MoE)
        ):
            shard_id = {"w1": "w3", "w3": "w1", "w2": "w2"}[shard_id]

        WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        expert_data = param.data[expert_id]

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if self.use_triton_kernels:
            is_transposed = True
        if is_transposed:
            shard_dim = int(not shard_dim)

        if self._maybe_load_fp8_shared_expert_as_fp4(
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
            shard_dim=shard_dim,
            tp_rank=tp_rank,
        ):
            return

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust input_scale for e4m3fnuz (AMD)
            if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                loaded_weight = loaded_weight * 2.0

            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                (
                    "compressed" in method.__class__.__name__.lower()
                    or "w4afp8" in self.quant_config.get_name()
                )
                and (param.data[expert_id] != 1).any()
                and ((param.data[expert_id] - loaded_weight).abs() > 1e-5).any()
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return

        if "ModelOpt" in method.__class__.__name__:
            # Determine per-tensor weight scale patterns based on variant
            is_fp4_variant = isinstance(method, ModelOptNvFp4FusedMoEMethod)

            # FP4 uses "weight_scale_2" for per-tensor, FP8 uses "weight_scale" for per-tensor
            per_tensor_conditions = (
                "weight_scale_2" in weight_name
                if is_fp4_variant
                else "weight_scale" in weight_name
            ) or "input_scale" in weight_name

            if per_tensor_conditions:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            elif "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            return

        # Case weight scales and zero_points
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust INT4 column-wise scaling number to e4m3fnuz (AMD)
                if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                    loaded_weight = loaded_weight * 0.5

                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust FP8 per-tensor scaling number for e4m3fnuz (AMD)
                if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                    loaded_weight = loaded_weight * 2.0

                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return

        if (
            "bias" in weight_name
            and self.quant_config.quant_description["quant_method"] == "modelslim"
        ):
            self._load_per_channel_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def weight_loader_fused(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
    ) -> None:
        tp_rank = self.moe_tp_rank

        if (
            self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.quant_config.is_static_cfg()
        ):
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            elif "scale" in weight_name:
                param.data.copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO: check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        method = self.quant_method
        if hasattr(self, "scheme"):
            method = self.scheme
        if isinstance(method, Fp8MoEMethod) and (
            get_moe_runner_backend().is_flashinfer_trtllm_routed()
            or get_moe_runner_backend().is_flashinfer_trtllm()
        ):
            # Drop the GPU mxfp8 shuffle-index cache on every reload for mxfp8 trtllm, trtllm_routed
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                clear_mxfp8_shuffle_index_cache,
            )

            clear_mxfp8_shuffle_index_cache()
        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                method.__class__.__name__
                in [
                    "CompressedTensorsWNA16MoE",
                    "CompressedTensorsWNA16TritonMoE",
                ]
            )
            and "zero" not in weight_name
            else loaded_weight
        )

        if shard_id not in ("w13", "w2"):
            raise ValueError(f"shard_id must be ['w13','w2'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w13": 1, "w2": 2}
        SHARD_ID_TO_SHARDED_DIM_TRANSPOSE = {"w13": 2, "w2": 1}

        expert_data = param.data
        is_bias = expert_data.dim() == 2

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)

        if self.use_triton_kernels:
            is_transposed = True
        shard_dim = (
            SHARD_ID_TO_SHARDED_DIM[shard_id]
            if not is_transposed
            else SHARD_ID_TO_SHARDED_DIM_TRANSPOSE[shard_id]
        )

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )
            return
        else:
            logging.warning(
                f"Unsupported weight_name {weight_name} for FusedMoE weight_loader_fused. Nothing is loaded."
            )

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        if self._use_ascend_fuseep:
            from sglang.srt.hardware_backend.npu.moe.fuseep import forward_fuseep

            return forward_fuseep(self, hidden_states, topk_output)
        if is_in_tc_piecewise_cuda_graph():
            if TopKOutputChecker.format_is_standard(topk_output):
                return moe_forward_piecewise_cuda_graph_impl(
                    hidden_states,
                    topk_output.topk_weights,
                    topk_output.topk_ids,
                    topk_output.router_logits,
                    self.layer_id,
                )
            elif TopKOutputChecker.format_is_bypassed(topk_output):
                return fused_moe_bypassed_piecewise_cuda_graph_impl(
                    hidden_states,
                    topk_output.router_logits,
                    topk_output.topk_config.top_k,
                    topk_output.topk_config.topk_group,
                    topk_output.topk_config.num_expert_group,
                    topk_output.topk_config.correction_bias,
                    topk_output.topk_config.renormalize,
                    self.layer_id,
                    topk_output.topk_config.allow_routed_experts_capture,
                )
            else:
                # Make sure there is torch lib op registration for the whole moe layer
                return self.forward_impl(hidden_states, topk_output)
        else:
            return self.forward_impl(hidden_states, topk_output)

    def forward_impl(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        final_hidden_states = self.forward_local(hidden_states, topk_output)
        if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_local(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        """This rank's contribution, WITHOUT the group all-reduce.

        Both shard modes of this layer produce a partial sum of the same
        full-width output -- disjoint experts under expert-dim sharding,
        partial per-expert GEMMs under intermediate-dim sharding -- which is
        why one all-reduce combines either of them. Splitting that reduce off
        is what lets the dual-group lane replace it with a LOCAL sum over the
        resident shards (#274 families slice C), the same substitution
        LaneRowParallelShell performs for a row-parallel linear.
        """
        origin_hidden_states_dim = hidden_states.shape[-1]
        assert self.quant_method is not None

        if getattr(self, "_gguf_expert_shard", False):
            # Uneven-TP GGUF MoE (expert-dim sharding): translate the
            # GLOBAL topk expert ids to this rank's LOCAL ids; foreign
            # experts map to the trailing all-zero padding expert, so their
            # local contribution is exactly 0 and the TP all-reduce sums
            # the true per-owner contributions. Routing (replicated router
            # + identical softmax/topk) is byte-identical on every rank.
            assert TopKOutputChecker.format_is_standard(topk_output)
            topk_output = topk_output._replace(
                topk_ids=self._gguf_topk_remap[topk_output.topk_ids.long()]
            )

        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )

        combine_input = self.run_moe_core(
            dispatch_output=dispatch_output,
        )

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            final_hidden_states = self.dispatcher.combine(combine_input=combine_input)

            # TODO: should we add some conditions here?
            final_hidden_states = final_hidden_states[
                ..., :origin_hidden_states_dim
            ].contiguous()

        return final_hidden_states

    def forward_deferred_finalize(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ):
        assert self.quant_method is not None
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            flashinfer_trtllm_deferred_finalize_context,
        )

        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )

        with flashinfer_trtllm_deferred_finalize_context():
            combine_input = self.run_moe_core(dispatch_output=dispatch_output)

        return self.dispatcher.combine(combine_input=combine_input)

    def run_moe_core(self, dispatch_output: DispatchOutput) -> CombineInput:
        # TODO: consider using symmetric memory
        if getattr(self, "_moe_offload_enabled", False):
            return self._run_moe_core_with_offload(dispatch_output)
        return self.quant_method.apply(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def _run_moe_core_with_offload(self, dispatch_output: DispatchOutput):
        """Expert-offload + routing-trace hook (feat/moe-expert-offload).

        Runs only when SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0 and/or
        SGLANG_MOE_OFFLOAD_TRACE is set. It operates on the STANDARD dispatch
        format (the pure-TP path used for the M-B/M-C measurement vehicle); any
        other dispatch format is passed through untouched so unsupported
        backends can never be silently corrupted.

        When active it (a) optionally records the per-token routed expert ids
        for the offline hit-rate simulator, and (b) fetches missing experts into
        the resident GPU slot buffer and runs the grouped-GEMM over the small
        resident buffer, WAVE-SPLITTING the forward when it needs more unique
        experts than there are resident slots (the prefill overflow case). The
        remap and per-token wave assignment are loss-free, so greedy outputs are
        identical across fractions.
        """

        def _apply(disp):
            return self.quant_method.apply(layer=self, dispatch_output=disp)

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            # Only the standard (pure-TP) path is in V1 scope; leave EP / bypass
            # / triton-kernels dispatch outputs exactly as-is.
            return _apply(dispatch_output)

        topk_ids = topk_output.topk_ids

        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        # (a) M-C routing trace: log the *pre-remap* (real) routed expert ids.
        # Guarded off the capture path: .tolist() is a device->host sync and
        # illegal inside graph capture (trace is measurement tooling only).
        if self._moe_offload_trace_path and not get_is_capture_mode():
            from sglang.srt.layers.moe.expert_offload import write_routing_trace

            rank_tag = f"tp{self.moe_tp_rank}ep{self.moe_ep_rank}"
            write_routing_trace(
                self._moe_offload_trace_path,
                rank_tag,
                self.layer_id,
                self._moe_offload_trace_step,
                topk_ids.detach().to("cpu").tolist(),
            )
            self._moe_offload_trace_step += 1

        # (b) Expert-offload remap (skipped entirely when fraction >= 1.0).
        if self._expert_offload_fraction >= 1.0 or self._expert_offload_install_failed:
            return _apply(dispatch_output)

        if self._expert_offload is None:
            self._install_expert_offload()
            if self._expert_offload is None:  # install declined / failed
                return _apply(dispatch_output)

        # Stage-3 captured decode: under graph capture (and the opt-in), take
        # the host-sync-free capturable path -- on-device remap + captured UVA
        # gather + a SINGLE-wave apply. Everything else (prefill, eager decode,
        # buckets beyond the captured sizes) keeps run_waves.
        if self._moe_offload_graph_mode and get_is_capture_mode():
            return self._run_moe_core_offload_capturable(
                dispatch_output, topk_output, topk_ids, _apply
            )

        # run_waves handles both the single-wave decode fast path (one apply over
        # the full batch) and the multi-wave prefill-overflow path (disjoint
        # token subsets, each fully computed once -> byte-identical accumulation).
        return self._expert_offload.run_waves(dispatch_output, _apply)

    def _run_moe_core_offload_capturable(
        self, dispatch_output, topk_output, topk_ids, apply_fn
    ):
        """Captured-decode offload step (design §2): enforce the fixed-shape
        single-wave invariant, remap on device, issue the captured gather, and
        run ONE apply over the full (fixed-bs) batch. No host sync anywhere."""
        from sglang.srt.layers.moe.expert_offload import worst_case_unique_spill

        cache = self._expert_offload
        # §2 invariant: worst-case unique spill must fit the scratch region, or
        # a captured step could silently drop spill experts (wrong output, not
        # epsilon). Fail loudly at capture/warmup.
        #
        # The bound is min(routed slots, cold experts), not routed slots alone:
        # a step cannot route more DISTINCT experts than this rank's cold set
        # holds. Under the #82 GGUF expert-dim shard that is the binding half,
        # because forward_impl has already collapsed every foreign id onto the
        # resident zero-pad expert. Reads shapes, never contents -- capture-safe.
        needed = worst_case_unique_spill(
            topk_ids.numel(), cache.num_local_experts, cache.resident_count
        )
        if needed > cache.scratch:
            raise RuntimeError(
                f"MoE offload CUDA-graph capture: bucket needs up to "
                f"{needed} unique spill experts (min of tokens x top_k = "
                f"{topk_ids.numel()} and the cold set = "
                f"{cache.num_local_experts - cache.resident_count}) but only "
                f"{cache.scratch} scratch slots exist. Raise "
                f"SGLANG_MOE_SCRATCH_SLOTS to >= {needed}, or cap the captured "
                f"decode batch sizes (SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS or "
                f"--cuda-graph-max-bs)."
            )
        remapped = cache.prepare_capturable(topk_ids)
        sub = dispatch_output._replace(
            topk_output=topk_output._replace(topk_ids=remapped)
        )
        return apply_fn(sub)

    def _install_expert_offload(self):
        """Lazily build the per-layer offload cache on first forward, once the
        expert weights are fully loaded/processed. On any error the layer falls
        back to the resident (default) path and never retries.

        #268/#323b: the quant-path check below runs BEFORE the try/except so it
        is a hard boot abort for MoeWNA16 / NVFP4 MoE / Ascend GGUF-MoE (no
        load-time offload half) and for any CUDA GGUF-MoE layer the #123 half
        declined to stage, never the silent per-layer fallback used for genuine
        install failures.
        """
        from sglang.srt.layers.moe.expert_offload import (
            MoEExpertOffloadCache,
            assert_expert_offload_quant_supported,
        )

        assert_expert_offload_quant_supported(
            self.quant_method,
            self.layer_id,
            # compressed-tensors hides the real layout behind a delegating
            # wrapper; the scheme is the class that owns the tensor names.
            scheme=getattr(self, "scheme", None),
            # #123-GGUF: GGUFMoEMethod is admitted only on a layer its
            # materialization-time half actually staged.
            layer=self,
        )

        try:
            cache = MoEExpertOffloadCache(self, self._expert_offload_fraction)
            if cache.planner.fully_resident:
                # ceil(fraction * num_local_experts) == num_local_experts:
                # nothing to offload, keep the default path.
                self._expert_offload_install_failed = True
                return
            cache.install()
            self._expert_offload = cache
            logging.getLogger(__name__).info(
                "MoE expert-offload active on layer %s: %d/%d experts resident "
                "+ %d scratch (buffer=%d, fraction=%.3f)",
                self.layer_id,
                cache.resident_count,
                cache.num_local_experts,
                cache.scratch,
                cache.planner.buffer_size,
                self._expert_offload_fraction,
            )
        except Exception as e:  # pragma: no cover - GPU-window failure path
            self._expert_offload_install_failed = True
            logging.getLogger(__name__).warning(
                "MoE expert-offload install failed on layer %s (%s); falling "
                "back to fully-resident experts.",
                self.layer_id,
                e,
            )
            return

        # Stage-3 residency freeze (file-driven). SGLANG_MOE_HOTSET_FILE freezes
        # the resident set from the offline hot set in BOTH eager and graph
        # mode -- the freeze is independent of graph capture, and running it
        # eagerly too lets an eager server serve the SAME residency as a
        # captured one (residency-matched A/B baselines). No file => static
        # [0,R) (unchanged). Then, only under graph mode, build the capturable
        # buffers so decode can be captured. This lazy install runs on the
        # first (warmup) forward -- eager, before DecodeCudaGraphRunner's stream
        # capture (§5 ordering). Deliberately OUTSIDE the silent-fallback
        # try/except: a mis-configured capturable path must abort the boot.
        if self._expert_offload is not None:
            if envs.SGLANG_MOE_HOTSET_FILE.get():
                self._expert_offload.freeze_from_source()
            if self._moe_offload_graph_mode:
                self._expert_offload.install_capturable_buffers()

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:
        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                ),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    @classmethod
    def make_expert_params_mapping_fused(
        cls,
        ckpt_gate_up_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_gate_up_proj_bias_name: str,
        ckpt_down_proj_bias_name: str,
    ):
        return [
            ("experts.w13_weight", f"experts.{ckpt_gate_up_proj_name}", "w13"),
            (
                "experts.w13_weight_bias",
                f"experts.{ckpt_gate_up_proj_bias_name}",
                "w13",
            ),
            ("experts.w2_weight", f"experts.{ckpt_down_proj_name}", "w2"),
            ("experts.w2_weight_bias", f"experts.{ckpt_down_proj_bias_name}", "w2"),
        ]

    @classmethod
    def make_expert_params_mapping_fused_mxfp4(
        cls,
        ckpt_gate_up_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_gate_up_proj_bias_name: str,
        ckpt_down_proj_bias_name: str,
        ckpt_gate_up_proj_scale_name: str,
        ckpt_down_proj_scale_name: str,
    ):
        return [
            ("experts.w13_weight", f"experts.{ckpt_gate_up_proj_name}", "w13"),
            (
                "experts.w13_weight_bias",
                f"experts.{ckpt_gate_up_proj_bias_name}",
                "w13",
            ),
            ("experts.w2_weight", f"experts.{ckpt_down_proj_name}", "w2"),
            ("experts.w2_weight_bias", f"experts.{ckpt_down_proj_bias_name}", "w2"),
            (
                "experts.w13_weight_scale",
                f"experts.{ckpt_gate_up_proj_scale_name}",
                "w13",
            ),
            ("experts.w2_weight_scale", f"experts.{ckpt_down_proj_scale_name}", "w2"),
        ]

    @classmethod
    def make_expert_input_scale_params_mapping(
        cls,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:
        # (param_name, weight_name, expert_id, shard_id)
        return [
            (
                "experts.w13_" if shard_id in ["w1", "w3"] else "experts.w2_",
                f"experts.{expert_id}.{shard_id}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id in ["w1", "w2", "w3"]
        ]

    def set_overlap_args(
        self, down_gemm_overlap_args: DownGemmOverlapArgs, meta_overlap_args: dict
    ):
        if hasattr(self, "runner"):
            self.runner.set_overlap_args(down_gemm_overlap_args, meta_overlap_args)
        else:
            # TODO: remove this branch after MoE refactor
            self.down_gemm_overlap_args = down_gemm_overlap_args
            self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        if hasattr(self, "runner"):
            self.runner.clear_overlap_args()
        else:
            # TODO: remove this branch after MoE refactor
            self.down_gemm_overlap_args = None
            self.meta_overlap_args = None

    def _gguf_moe_offload_eligible(self) -> bool:
        """Whether the #123 GGUF MoE offload half covers THIS layer.

        Three conditions, all checked before a single byte is staged so an
        uncovered checkpoint falls back to the full stack (and then hits the
        #268 guard at install time with a named reason, rather than silently
        running a half-tiered layer):

        1. a resident fraction < 1.0 is configured at all;
        2. the layer's quant method is the CUDA ``GGUFMoEMethod`` -- the Ascend
           MoE method materializes and pre-dequantizes on its own path;
        3. BOTH expert tensors carry a ggml type with a GGUF MoE kernel --
           tiering bytes no kernel dispatches on would produce a layer nothing
           can execute at expert granularity. Note what this does NOT see any
           more: MXFP4 (type 39, DeepSeek-V4-Flash's routed down projections)
           is rewritten to Q5_0 by the #391 load-time repack inside the weight
           iterator, marker and payload both, so the type read here is already
           a covered one.

        Memoized: #391c asks the same question at the FIRST expert of the
        stream and again at materialization, and both answers must be the one
        answer (the ggml types are fixed by the iterator's first pass, before
        either call). Memoizing also keeps the decline warning to one line per
        layer instead of one per door.
        """
        memo = getattr(self, "_gguf_offload_eligible_memo", None)
        if memo is not None:
            return memo
        eligible = self._gguf_moe_offload_eligible_uncached()
        self._gguf_offload_eligible_memo = eligible
        return eligible

    def _gguf_moe_offload_eligible_uncached(self) -> bool:
        from sglang.srt.layers.moe.resident_fraction import (
            resident_fraction_for_rank,
        )

        fraction = getattr(self, "_expert_offload_fraction", None)
        if fraction is None:
            fraction = resident_fraction_for_rank()
        if fraction >= 1.0:
            return False
        if type(getattr(self, "quant_method", None)).__name__ != "GGUFMoEMethod":
            return False
        from sglang.srt.layers.quantization.gguf import gguf_moe_offload_covered_type

        for attr in ("w13_qweight_type", "w2_qweight_type"):
            holder = getattr(self, attr, None)
            if holder is None or not gguf_moe_offload_covered_type(
                getattr(holder, "weight_type", None)
            ):
                logging.getLogger(__name__).warning(
                    "GGUF MoE expert-offload declined on layer %s: %s is ggml "
                    "type %s, which has no GGUF MoE kernel. The layer keeps "
                    "every expert resident and the #268 guard will refuse the "
                    "offload installer for it.",
                    getattr(self, "layer_id", "?"),
                    attr,
                    getattr(holder, "weight_type", None),
                )
                return False
        return True

    def _finish_gguf_moe_offload_staging(
        self, staged_attrs, plan, door: str = "materialization"
    ) -> None:
        """Mark the layer as covered and release the loader's host copies.

        The marker is what lifts the #268 guard for this layer specifically
        (``_OFFLOAD_CONDITIONAL_QUANT_METHOD_NAMES``); it is set only after
        BOTH expert tensors were staged, because a layer with one tiered and
        one full tensor is precisely the half-staged state the guard exists to
        refuse.
        """
        from sglang.srt.layers.moe.expert_offload import trim_host_allocator

        expected = {"w13_qweight", "w2_qweight"}
        if set(staged_attrs) != expected:
            raise RuntimeError(
                f"GGUF MoE expert-offload staged {sorted(staged_attrs)} on "
                f"layer {getattr(self, 'layer_id', '?')} but needs exactly "
                f"{sorted(expected)}; a partially tiered layer would pair one "
                f"expert's gate/up with another expert's down projection."
            )
        self._moe_offload_gguf_staged = True
        # glibc/torch retain the freed per-expert CPU buffers in their
        # allocator pools; across 40+ MoE layers that is the whole expert set
        # squeezing a swapless host. Same per-layer trim the repack presplit
        # does, for the same reason. (The streaming door has already trimmed at
        # its own layer boundary; one more here is idempotent and cheap.)
        trim_host_allocator()
        logging.getLogger(__name__).info(
            "GGUF MoE expert-offload staged at %s on layer %s: "
            "%d/%d experts resident + %d scratch slots, %d experts in the "
            "pinned host tier; the full expert stack was never allocated.",
            door,
            getattr(self, "layer_id", "?"),
            plan.resident_count,
            plan.num_experts,
            plan.buffer_slots - plan.resident_count,
            len(plan.spill_ids),
        )

    @staticmethod
    def _gguf_expert_source(
        name, expert_weights, num_experts, expert_shard, expert_data_map=None
    ):
        """Per-expert accessor for one GGUF MoE parameter, without stacking.

        Returns ``(count, row_shape, dtype, get, drop)`` or ``None`` when this
        parameter has no expert data. ``get(i)`` yields the i-th expert's
        quantized tensor -- for w13 the gate/up pair concatenated on the row
        axis, which is exactly what the stacked path builds -- and ``drop(i)``
        releases that expert's loaded bytes from BOTH holders (the local
        by-expert view and the parameter's ``expert_data_map``), which is what
        actually returns them to the allocator; dropping one while the other
        still refers to the tensor frees nothing.

        Splitting the accessor out of the stacking is the whole trick of the
        #123 GGUF half: the load-time offload has to place experts in two
        different tiers, so it must be able to take them ONE AT A TIME. The
        default (fully resident) path below still stacks, byte-for-byte as
        before.
        """
        ordered = []
        if "w13" in name:
            for e in range(num_experts):
                shards = expert_weights.get(e)
                if (
                    shards
                    and shards.get("w1") is not None
                    and shards.get("w3") is not None
                ):
                    ordered.append(e)
            keys = ("w1", "w3")
        elif "w2" in name:
            for e in range(num_experts):
                shards = expert_weights.get(e)
                if shards and shards.get("w2") is not None:
                    ordered.append(e)
            keys = ("w2",)
        else:
            return None
        if not ordered:
            return None

        first = expert_weights[ordered[0]]
        rows = sum(int(first[k].shape[0]) for k in keys)
        row_shape = (rows,) + tuple(first[keys[0]].shape[1:])
        dtype = first[keys[0]].dtype
        # The trailing all-zero padding expert of the uneven-TP expert-dim
        # shard (#82): target of every foreign topk id. Zero ggml bytes decode
        # to 0 for every block type (d/min scales are zero), so it contributes
        # exactly nothing to the TP all-reduce.
        count = len(ordered) + (1 if expert_shard else 0)

        def get(i):
            if i >= len(ordered):
                return torch.zeros(row_shape, dtype=dtype)
            shards = expert_weights[ordered[i]]
            if len(keys) == 1:
                return shards[keys[0]]
            return torch.cat([shards[k] for k in keys], dim=0)

        def drop(i):
            if i >= len(ordered):
                return
            expert_weights.pop(ordered[i], None)
            if expert_data_map is not None:
                for k in keys:
                    expert_data_map.pop((ordered[i], k), None)

        return count, row_shape, dtype, get, drop

    def materialize_gguf_weights(self) -> None:
        """Process weights after loading, especially for GGUF quantization.

        This materializes GGUF UninitializedParameters from their data_containers.

        #123-GGUF: this is ALSO the expert-offload interception point for GGUF
        MoE. The other quant paths split an already-materialized ``[E, ...]``
        stack in ``presplit_expert_offload_after_repack``; GGUF has no such
        stack to split -- it is created right here, and creating it is exactly
        the device (and host) peak the offload exists to avoid. So when a
        resident fraction < 1.0 is configured and the checkpoint's ggml type
        has a MoE kernel, the loop below never builds the full stack: it
        materializes the parameter at the ``[R+C]`` buffer size, writes only
        the resident experts into it, and streams every other expert straight
        into the pinned host tier. ``MoEExpertOffloadCache.install()`` then
        picks the two tiers up from ``_moe_offload_presplit`` exactly as it
        does for the fp8 / GPTQ / AWQ presplit.

        #391c: on a covered layer the tiers are already FULL by the time this
        runs -- ``_load_gguf_weight`` staged every expert as it left the weight
        stream, so all that is left here is to close the stagers and publish
        what they built. That matters because the loader runs this hook only
        after the COMPLETE load_weights pass: intercepting here alone means the
        whole owned expert set has already been paid for in host RAM, which is
        the peak boot attempt 5 of #391 was OOM-killed at. The loop below is
        then the fallback door (streaming disabled, or a non-expert GGUF
        parameter) and the default full-stack path, both unchanged.
        """
        from sglang.srt.layers.moe.expert_offload import (
            plan_load_time_staging,
            register_load_time_presplit,
            stage_experts_into_tiers,
        )

        self._drain_gguf_stream_stagers()

        staged_attrs = []
        staged_plan = None
        for name, param in list(self.named_parameters()):
            is_gguf_weight = getattr(param, "is_gguf_weight", False)

            if is_gguf_weight and isinstance(param, UninitializedParameter):
                data_container = getattr(param, "data_container", [])
                expert_data_map = getattr(param, "expert_data_map", {})
                tensor_shape = getattr(param, "tensor_shape", None)

                if data_container and tensor_shape:
                    # Determine the structure from expert_data_map
                    num_experts = tensor_shape[0]

                    # Collect weights by expert
                    expert_weights = {}
                    for (expert_id, shard_id), weight in expert_data_map.items():
                        if expert_id not in expert_weights:
                            expert_weights[expert_id] = {}
                        expert_weights[expert_id][shard_id] = weight

                    # Build the full tensor
                    expert_shard = getattr(self, "_gguf_expert_shard", False)

                    source = self._gguf_expert_source(
                        name,
                        expert_weights,
                        num_experts,
                        expert_shard,
                        expert_data_map=expert_data_map,
                    )
                    if source is None:
                        continue
                    count, row_shape, dtype, get, drop = source

                    plan = (
                        plan_load_time_staging(
                            count,
                            # The layer latched its fraction at construction;
                            # the cache will size itself from the SAME value,
                            # so the plan must not re-read the environment.
                            fraction=getattr(self, "_expert_offload_fraction", None),
                            pinned_experts=(count - 1,) if expert_shard else (),
                            # #394 activation point for the pull door; the
                            # streaming door passes the same context. ``None``
                            # -- i.e. today's plan, field for field -- unless
                            # SGLANG_MOE_HOST_SHARD_RATIO names a ratio.
                            cold_shard=self._gguf_cold_shard_context(),
                        )
                        if self._gguf_moe_offload_eligible()
                        else None
                    )

                    if plan is None:
                        # Default path: one [E, ...] stack, unchanged.
                        stacked = torch.stack([get(i) for i in range(count)], dim=0)
                        param.materialize(stacked.shape, dtype=stacked.dtype)
                        param.data.copy_(stacked)
                        continue

                    # The loader's flat list is only read for the truthiness
                    # test above; dropping it now lets ``drop`` below actually
                    # return each expert's bytes to the allocator as they are
                    # consumed, instead of at the end of the layer. Holding the
                    # loaded copies alongside the two tiers would double the
                    # host footprint of the very set the offload exists to keep
                    # off the card.
                    param.data_container = []
                    param.materialize((plan.buffer_slots,) + row_shape, dtype=dtype)
                    spill = stage_experts_into_tiers(
                        plan, get, param.data, release=drop
                    )
                    register_load_time_presplit(self, name, param.data, spill, plan)
                    staged_attrs.append(name)
                    staged_plan = plan
                    param.expert_data_map = {}
                    expert_weights.clear()

        if staged_attrs:
            self._finish_gguf_moe_offload_staging(staged_attrs, staged_plan)

        if getattr(self, "_gguf_expert_shard", False) and not hasattr(
            self, "_gguf_topk_remap"
        ):
            # Global -> local expert id translation for forward_impl:
            # owned experts map to their local slot (global order), all
            # foreign experts map to the zero padding expert (index
            # n_local). Lives on the weights' device; plain indexing, so
            # CUDA-graph safe.
            lo, hi = self._gguf_expert_range
            n_local = hi - lo
            device = next(
                (
                    p.device
                    for p in self.parameters()
                    if not isinstance(p, UninitializedParameter)
                ),
                torch.device("cuda"),
            )
            remap = torch.full(
                (self.num_experts,), n_local, dtype=torch.int32, device=device
            )
            remap[lo:hi] = torch.arange(n_local, dtype=torch.int32, device=device)
            self._gguf_topk_remap = remap


@register_custom_op(out_shape="hidden_states")
def moe_forward_piecewise_cuda_graph_impl(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    router_logits: torch.Tensor,
    layer_id: int,
) -> torch.Tensor:
    # only standard topk output is supported for piecewise cuda graph
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
    )
    forward_context = get_tc_piecewise_forward_context()
    moe_layer = forward_context.moe_layers[layer_id]
    return moe_layer.forward_impl(hidden_states, topk_output)


@register_custom_op(out_shape="hidden_states")
def fused_moe_bypassed_piecewise_cuda_graph_impl(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    topk_group: Optional[int],
    num_expert_group: Optional[int],
    correction_bias: Optional[torch.Tensor],
    renormalize: bool,
    layer_id: int,
    allow_routed_experts_capture: bool,
) -> torch.Tensor:
    topk_output = BypassedTopKOutput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_config=TopKConfig(
            top_k=top_k,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            correction_bias=correction_bias,
            renormalize=renormalize,
            allow_routed_experts_capture=allow_routed_experts_capture,
        ),
    )
    forward_context = get_tc_piecewise_forward_context()
    moe_layer = forward_context.moe_layers[layer_id]
    return moe_layer.forward_impl(hidden_states, topk_output)
