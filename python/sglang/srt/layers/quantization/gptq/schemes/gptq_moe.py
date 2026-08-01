# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.linear import set_weight_attrs
from sglang.srt.layers.moe import MoeRunnerConfig

from .gptq_scheme import GPTQMoESchemeBase

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig, GPTQMarlinConfig

__all__ = ["GPTQMoEAscendScheme", "GPTQMarlinMoEScheme"]


def _moe_offload_active() -> bool:
    """Is MoE expert offload on anywhere in the group?

    Imported lazily to keep this module's import graph unchanged. Group-wide
    on purpose: this gates where weights are placed, and a rank that answered
    differently from its peers would build a structurally different model.
    """
    from sglang.srt.layers.moe.resident_fraction import offload_active

    return offload_active()


class GPTQMoEAscendScheme(GPTQMoESchemeBase):
    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config
        from sglang.srt.hardware_backend.npu.quantization.gptq_kernels import (
            GPTQMoEAscendKernel,
        )

        self.kernel = GPTQMoEAscendKernel(quant_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        pack_factor = self.quant_config.pack_factor

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = intermediate_size_per_partition // self.quant_config.group_size

        extra_weight_attrs.update(
            {
                "is_transposed": True,
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            }
        )

        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // pack_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // pack_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition // pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.kernel.create_moe_runner(layer, moe_runner_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)
        # Variant-C B2b: right after the marlin repack (repacked tensors on GPU),
        # split into a fixed-resident GPU buffer + pinned-host spill so the full
        # [E] expert stack never gets pinned back to host (load-time RAM cap).
        # No-op unless SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0.
        from sglang.srt.layers.moe.expert_offload import (
            presplit_expert_offload_after_repack,
        )

        presplit_expert_offload_after_repack(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ):
        return self.kernel.apply(layer, dispatch_output)


class GPTQMarlinMoEScheme(GPTQMoESchemeBase):
    def __init__(self, quant_config: GPTQMarlinConfig):
        self.quant_config = quant_config
        from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
            GPTQMarlinMoEKernel,
        )

        self.kernel = GPTQMarlinMoEKernel(quant_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.kernel.is_k_full = (
            not self.quant_config.desc_act
        ) or layer.moe_tp_size == 1

        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            if self.quant_config.desc_act:
                # act-order (desc_act=True): g_idx permutes groups, so w2 needs
                # the FULL scales replicated per rank (full-width + load_full_w2).
                # Untouched by this bugfix.
                w2_scales_size = intermediate_size_per_partition
            else:
                # Bugfix (desc_act=False, TP>1 -- even AND uneven): the stock
                # full-width allocation (intermediate_per_partition * moe_tp_size)
                # with load_full_w2=False (below) made the loader narrow a
                # full-width w2_scales at a per-rank offset -> OOB ("start (K) +
                # length (K) exceeds dimension size (K)"): even TP via
                # _moe_src_start's shard_size*tp_rank, uneven TP via
                # tp_partition_offset over the coarse group grid. w2 is the
                # down-proj (row-parallel, sharded along intermediate); with
                # group quant along intermediate and desc_act=False the groups
                # are contiguous, so each rank needs ONLY its intermediate-shard
                # of the scales. intermediate_size_per_partition IS that shard
                # (even: /tp; uneven: *weight/sum), and the marlin w2 repack
                # derives size_k from w2_scales.shape[1]*group_size, so the
                # sharded width keeps qweight (already sharded) and scales
                # k-consistent. Loading full scales per rank (flipping
                # load_full_w2) would feed the wrong rows -> silent garbage.
                # CORRECTNESS PRECONDITION (uneven TP): each rank's intermediate
                # shard must be group-aligned, else a scale group straddles a
                # rank boundary and the sharded scales are silently wrong. Assert
                # it loudly here rather than let it pass a coherence glance.
                gs = self.quant_config.group_size
                if intermediate_size_per_partition % gs != 0:
                    raise ValueError(
                        "GPTQ MoE w2_scales group-alignment violated on "
                        f"moe_tp_rank={getattr(layer, 'moe_tp_rank', '?')}: "
                        f"intermediate_size_per_partition="
                        f"{intermediate_size_per_partition} is not a multiple of "
                        f"group_size={gs} (moe_tp_size={layer.moe_tp_size}). Under "
                        "uneven TP the per-rank intermediate shard must align to "
                        "group_size, not just the 16-unit; the shard boundaries "
                        "need a group_size-aware coarsening -- this is a bigger "
                        "fix than the scales width. STOP."
                    )
                import logging as _logging

                _logging.getLogger(__name__).info(
                    "GPTQ MoE w2_scales sharded: moe_tp_rank=%s "
                    "intermediate_per_partition=%d group_size=%d groups=%d "
                    "(moe_tp_size=%d) -- group-aligned OK",
                    getattr(layer, "moe_tp_rank", "?"),
                    intermediate_size_per_partition,
                    gs,
                    intermediate_size_per_partition // gs,
                    layer.moe_tp_size,
                )
                w2_scales_size = intermediate_size_per_partition
            scales_size2 = w2_scales_size // self.quant_config.group_size
            strategy = FusedMoeWeightScaleSupported.GROUP.value
        else:
            scales_size13 = 1
            scales_size2 = 1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})

        # Per-expert MoE offload (Variant-C B2b): when the resident-expert
        # fraction < 1.0 the big expert-major tensors (w13/w2 qweight + scales)
        # are built on CPU instead of the ambient cuda device. The standard load
        # path then moves each FusedMoE module to GPU for the marlin repack via
        # device_loading_context and copies the repacked result back to
        # CPU-pinned on exit (loader.py), so the full [E,...] stack never sits on
        # GPU at once (bounds the load-time OOM). The MoEExpertOffloadCache later
        # keeps only n_slots experts resident on GPU and streams cold ones from
        # the pinned host pool. The small/unused qzeros + (desc_act=False) empty
        # g_idx stay on the default device. fraction>=1.0 -> _moe_dev is None
        # (default cuda) -> byte-identical stock path.

        _moe_dev = "cpu" if _moe_offload_active() else None

        # Bring-up footnote, 2026-07-19. This line is why.
        # create_weights() commits the ENTIRE expert set up front: torch.empty over
        # all num_experts here, ~61 GB for the 122B-A10B, on CPU, before a single
        # marlin repack runs. The peak is therefore fraction- AND tensor-parallel-
        # independent -- sharding splits it across processes, not the aggregate. On
        # the 80 GB no-swap box this dropped MemAvailable to ~5 GB mid-load and the
        # boot aborted one allocation short of a freeze.
        #
        #   operator:  it aborted again.
        #   assistant: the 61 GB is committed right here, before any repack, so no
        #              resident fraction can help. The correct fix is a streaming
        #              loader: meta-allocate, then per decoder layer materialize ->
        #              load -> repack -> presplit -> free, so the host never holds
        #              all 61 GB at once. It's a real per-layer load loop. Give me
        #              some time.
        #   operator:  or the box has 30 GB more RAM now.
        #   assistant: you cannot download more RA--
        #   operator:  I downloaded more RAM.
        #   assistant: ...the floor is ~28 GB now. It boots on the existing loader.
        #              Thank you for the RAM.
        #
        # So the streaming loader was never built. If you're reading this because
        # create_weights just OOM'd the host on a model bigger than the 122B, that
        # is your signal to finally build it. Until then: 108 GiB and a low fraction.
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.quant_config.pack_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
                device=_moe_dev,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.quant_config.pack_factor,
                hidden_size,
                dtype=torch.int32,
                device=_moe_dev,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        # Task #283: the group scales must be allocated in params_dtype, not a
        # hardcoded float16. `moe_wna16_marlin_gemm` is templated on a SINGLE
        # scalar type taken from the activation dtype and reads `b_scales`
        # through that same type -- a float16 scale tensor against bfloat16
        # activations is not a slow path, it is a bit reinterpretation. Stock
        # sglang pinned these two allocations to torch.half, so every bfloat16
        # GPTQ MoE checkpoint (Qwen3.5-35B-A3B-GPTQ-Int4 among them) died at the
        # first MoE forward on fused_marlin_moe's
        # "hidden_states.dtype == w1_scale.dtype" guard, with --dtype float16 as
        # the only workaround. The AWQ-Marlin MoE sibling that feeds the exact
        # same kernel (awq_moe.py) has always used params_dtype here; this makes
        # GPTQ agree with it.
        #
        # Numerics: for a float16 model params_dtype IS torch.half, so the
        # allocation, the loader copy and the kernel input are unchanged
        # bit-for-bit. For a bfloat16 model the checkpoint's float16 scales are
        # downcast once at load (inside the weight loader's copy_), never per
        # forward. Measured on INT4 group-128 weights: the downcast perturbs
        # the dequantized weight by 0.167% while the INT4 grid it scales is
        # already 11.15% off the original -- 67x smaller than the error it
        # rides on.
        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
                device=_moe_dev,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size2,
                hidden_size,
                dtype=params_dtype,
                device=_moe_dev,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        set_weight_attrs(w2_scales, {"load_full_w2": self.quant_config.desc_act})

        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size2,
                hidden_size // self.quant_config.pack_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)
        set_weight_attrs(w2_qzeros, {"load_full_w2": self.quant_config.desc_act})

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.kernel.create_moe_runner(layer, moe_runner_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)
        # Variant-C B2b: right after the marlin repack (repacked tensors on GPU),
        # split into a fixed-resident GPU buffer + pinned-host spill so the full
        # [E] expert stack never gets pinned back to host (load-time RAM cap).
        # No-op unless SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0.
        from sglang.srt.layers.moe.expert_offload import (
            presplit_expert_offload_after_repack,
        )

        presplit_expert_offload_after_repack(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ):
        return self.kernel.apply(layer, dispatch_output)
