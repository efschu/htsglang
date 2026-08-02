# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.linear import set_weight_attrs
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_runner_backend,
)

from .awq_scheme import AWQMoESchemeBase

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.quantization.awq.awq import AWQConfig, AWQMarlinConfig

__all__ = ["AWQMoEScheme", "AWQAscendMoEScheme"]


def _moe_offload_active() -> bool:
    """Is MoE expert offload on anywhere in the group?

    Imported lazily to keep this module's import graph unchanged. Group-wide
    on purpose: this gates where weights are placed, and a rank that answered
    differently from its peers would build a structurally different model.
    """
    from sglang.srt.layers.moe.resident_fraction import offload_active

    return offload_active()


class AWQMoEScheme(AWQMoESchemeBase):
    def __init__(self, quant_config: AWQMarlinConfig):
        self.quant_config = quant_config
        if self.quant_config.weight_bits != 4:
            raise ValueError("AWQMoEScheme only supports 4bit now.")
        self.kernel = self._init_kernel(quant_config)

    def _init_kernel(self, quant_config: AWQMarlinConfig):
        from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import (
            AWQMoEKernel,
        )

        return AWQMoEKernel(quant_config)

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

        extra_weight_attrs.update(
            {
                "is_transposed": True,
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            }
        )

        # Per-expert MoE offload (Variant-C B2b), load-time half -- the same
        # construction gptq_moe.create_weights uses. With a resident-expert
        # fraction < 1.0 the expert-major tensors are built on the host instead
        # of the ambient cuda device, so the full [E,...] AWQ stack never sits
        # on GPU at once. The standard load path still runs the awq-marlin
        # repack on GPU: loader.py wraps process_weights_after_loading of each
        # module in device_loading_context, which moves CPU params to the target
        # device and restores them on exit. No kernel change is needed.
        #
        # Unlike GPTQ, the AWQ qzeros are NOT a formality: AWQ is asymmetric, so
        # w13_qzeros/w2_qzeros carry real checkpoint data that
        # moe_awq_to_marlin_zero_points consumes at repack time and the marlin
        # apply reads per expert afterwards (they are staged by the offload cache
        # alongside qweight/scales). They are expert-major and therefore belong
        # on _moe_dev too -- leaving them on the default device would keep an
        # [E]-sized tensor on GPU and defeat part of the load-time cap.
        #
        # fraction >= 1.0 -> _moe_dev is None -> byte-identical stock path.

        _moe_dev = "cpu" if _moe_offload_active() else None

        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
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
                intermediate_size_per_partition,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
                device=_moe_dev,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = intermediate_size_per_partition // self.quant_config.group_size

        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                intermediate_size_per_partition * 2,
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
                num_groups_w2,
                hidden_size,
                dtype=params_dtype,
                device=_moe_dev,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
                device=_moe_dev,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
                device=_moe_dev,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)
        # Variant-C B2b: after the awq-marlin repack, split into a fixed-resident
        # GPU buffer + pinned-host spill so the full [E] expert stack never gets
        # pinned back to host (load-time RAM cap). No-op unless
        # SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0.
        from sglang.srt.layers.moe.expert_offload import (
            presplit_expert_offload_after_repack,
        )

        # #421 F8: no ``cold_shard=`` here, deliberately. The #394
        # link-proportional cold-expert policy is REFUSED at this door --
        # see expert_offload.refuse_cold_shard_at_repack_door for the
        # measured reason. Passing one raises rather than staging a plan
        # whose delegated experts no rank can reach.
        presplit_expert_offload_after_repack(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        assert get_moe_runner_backend().is_auto()
        self.moe_runner_config = moe_runner_config
        self.kernel.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ):
        return self.kernel.apply(layer, dispatch_output)


class AWQAscendMoEScheme(AWQMoEScheme):
    def _init_kernel(self, quant_config: AWQConfig):
        from sglang.srt.hardware_backend.npu.quantization.awq_kernels import (
            AWQAscendMoEKernel,
        )

        return AWQAscendMoEKernel(quant_config)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
