from __future__ import annotations

import logging
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, cast

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_host_cpu_arm64,
    set_weight_attrs,
    use_intel_amx_backend,
)
from sglang.srt.utils.patch_torch import register_fake_if_exists

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput

_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_cpu_arm64 = is_host_cpu_arm64()

_has_sgl_int8_scaled_mm = False
if _is_cuda:
    # Turing (sm75): see compressed_tensors_w8a8_int8.py -- sgl-kernel holds no
    # code for this arch. INT8 scaled matmul is also the operation Turing is
    # arithmetically strongest at, so this is a future optimisation target for
    # the Turing feature, not just a blocker.
    try:
        from sgl_kernel import int8_scaled_mm

        _has_sgl_int8_scaled_mm = True
    except ImportError:
        int8_scaled_mm = None

if _is_cuda and _has_sgl_int8_scaled_mm:

    @register_fake_if_exists("sgl_kernel::int8_scaled_mm")
    def _int8_scaled_mm_abstract(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias=None,
    ):
        M = mat_a.shape[-2]
        N = mat_b.shape[-1]
        return mat_a.new_empty((M, N), dtype=out_dtype)


logger = logging.getLogger(__name__)


#: Runbook section that documents the wheel, its provenance and the pin.
INT8_ARM_RUNBOOK_SECTION = "docs/rig-runbook.md section 2.1 (sgl-kernel INT8 arm)"

#: Quantization names whose serving path needs ``sgl_kernel.int8_scaled_mm``.
INT8_ARM_QUANTIZATIONS = ("w8a8_int8", "compressed-tensors", "compressed_tensors")


def int8_arm_available() -> bool:
    """Is ``sgl_kernel.int8_scaled_mm`` importable in THIS interpreter?

    A function, not the module-level flag, so a test can drive both answers
    without reimporting the package.
    """
    try:
        from sgl_kernel import int8_scaled_mm  # noqa: F401

        return True
    except ImportError:
        return False


def require_int8_arm(
    quantization: Optional[str],
    *,
    is_cuda_group: bool = True,
    available: Optional[bool] = None,
) -> None:
    """Refuse an INT8-W8A8 boot whose wheel has no INT8 arm -- AT BOOT (#384).

    WHY THIS EXISTS WHEN A LOUD ERROR ALREADY DOES.
    ``CompressedTensorsW8A8Int8.__init__`` already raises when the arm is
    missing, but it raises during LAYER CONSTRUCTION -- inside the JIT
    cold-build window -- so the operator sees ``ColdBuildWindowError`` and a
    suggestion to lower ``--mem-fraction-static``, which is unrelated and
    unactionable. The real cause (an sgl-kernel wheel without the arm) is
    decidable from ``server_args.quantization`` alone, before a byte of weight
    is read, and that is where it belongs.

    Its message is also the wrong diagnosis for this case: it blames the sm75
    gencode floor, which is right for a 2080 Ti and wrong for a modern card
    running a pypi wheel that simply does not carry the arm. This one names
    the wheel and the runbook section instead.

    Pure: ``available`` is injectable so the refusal is testable on a host
    where the arm happens to be present.
    """
    if not is_cuda_group:
        return
    if not quantization:
        return
    if str(quantization).lower().replace("-", "_") not in (
        q.replace("-", "_") for q in INT8_ARM_QUANTIZATIONS
    ):
        return
    have = int8_arm_available() if available is None else bool(available)
    if have:
        return
    raise RuntimeError(
        f"--quantization {quantization} needs sgl_kernel.int8_scaled_mm and "
        "this interpreter's sgl-kernel does not provide it. The stock pypi "
        "wheel ships without the INT8 arm; the fork wheel carries it. Install "
        f"the fork wheel and pin it -- see {INT8_ARM_RUNBOOK_SECTION}. "
        "Refused here, at argument resolution, because the same failure "
        "during layer construction surfaces as a ColdBuildWindowError telling "
        "you to lower --mem-fraction-static, which is neither the cause nor a "
        "fix."
    )


# ---------------------------------------------------------------------------
# Uneven-TP alignment for the INT8 W8A8 GEMM (task #353).
#
# ``sgl_kernel.int8_scaled_mm`` (sgl-kernel/csrc/gemm/int8_gemm_kernel.cu) makes
# three unconditional TORCH_CHECKs on the shard it is handed:
#
#     mat_a.size(1) % 16 == 0    # K, the activation / input dim
#     mat_b.size(0) % 16 == 0    # K, the weight's input dim
#     mat_b.size(1) %  8 == 0    # N, output_size_per_partition
#
# They are checked at the first forward, i.e. AFTER weight load and after the
# whole shard plan is already committed, so a split that misses them costs a
# full model load before it fails.
# ---------------------------------------------------------------------------

#: ``int8_scaled_mm``'s N (output) alignment: ``mat_b.size(1) % 8 == 0``.
INT8_SCALED_MM_ALIGN_N = 8
#: ``int8_scaled_mm``'s K (input) alignment: ``mat_a.size(1) % 16 == 0``.
INT8_SCALED_MM_ALIGN_K = 16


def int8_w8a8_uneven_tp_block() -> List[int]:
    """Uneven-TP shard block for a dynamic-activation INT8 W8A8 config.

    Sixth member of the alignment family (#37 / #86 / #289 / #300 / #316 /
    #318 / #323): a quantization config that never exposed
    ``weight_block_size`` left ``_quant_block_aligned_units`` /
    ``tp_partition_size`` / ``moe_uneven_tp_units`` with nothing to coarsen, so
    an uneven ``--rank-tp-ratio`` split produced per-rank shards the kernel
    refuses. Channel-strategy INT8 has neither a ``block_structure`` nor a
    ``group_size``, so it returned ``None`` and no coarsening fired at all.

    Unlike every earlier member, this block carries NO quantization meaning.
    Channel-strategy scales are one scalar per output row and are therefore
    insensitive to where the row dimension is cut; there is no scale grid to
    stay inside. The block exists solely to keep the CUTLASS kernel's own
    alignment check satisfied under an uneven split -- do not go looking for a
    scale-grid reason, there is not one.

    Both dims carry the SAME value, ``lcm(8, 16) = 16``, not the ``[8, 16]``
    that ANALYSE_319 sec. 2d sketched. A single shared block is mandatory for
    the coupled MLP pair: ``gate_up``'s OUTPUT units and ``down``'s INPUT units
    partition the same intermediate dimension, so a per-dim block would
    coarsen the two ends of that dimension differently and the per-rank shards
    would diverge. ``CompressedTensorsConfig._group_size_block`` and
    ``modelopt_fp4_uneven_tp_block`` set both dims for exactly this reason.
    """
    return [math.lcm(INT8_SCALED_MM_ALIGN_N, INT8_SCALED_MM_ALIGN_K)] * 2


def verify_int8_scaled_mm_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    layer_name: str = "",
) -> None:
    """Raise if this rank's shard cannot be fed to ``int8_scaled_mm``.

    The kernel's checks live in CUDA and fire mid-forward. This mirrors them at
    layer-construction time so the failure names the layer, the numbers and the
    cause instead of surfacing as a bare ``TORCH_CHECK`` after a full model
    load.

    The block above cannot cover every family. A unit family that is COUPLED
    across several layers of different widths -- Qwen3.5's GDN, where
    ``gdn_tp_units`` is derived once from ``value_dim`` and handed unchanged to
    ``in_proj_qkvz`` (1024-element units), ``in_proj_ba`` (3-element units) and
    ``out_proj`` -- cannot be coarsened per layer without breaking the coupling
    the model depends on. ``in_proj_ba`` is the concrete case: 48 per part over
    16 k-head units, so an uneven split hands ranks N = 42 / 30 / 24 and the
    first two are not multiples of 8. Coarsening it locally would collapse it
    to a single unit and mis-shard the GDN against ``in_proj_qkvz``, so this is
    #300 territory: the bad shard stays loud, by name.
    """
    where = f" ({layer_name})" if layer_name else ""
    if output_size_per_partition % INT8_SCALED_MM_ALIGN_N:
        raise ValueError(
            f"INT8 W8A8 layer{where}: output_size_per_partition = "
            f"{output_size_per_partition} is not a multiple of "
            f"{INT8_SCALED_MM_ALIGN_N}, which sgl_kernel.int8_scaled_mm "
            "requires of N (mat_b.size(1)). Under an uneven --rank-tp-ratio "
            "this comes from a unit family too fine or too coupled to coarsen "
            "to the kernel's block; pick a shard plan whose per-rank output is "
            f"a multiple of {INT8_SCALED_MM_ALIGN_N}, or exclude this module "
            "from quantization in the checkpoint."
        )
    if input_size_per_partition % INT8_SCALED_MM_ALIGN_K:
        raise ValueError(
            f"INT8 W8A8 layer{where}: input_size_per_partition = "
            f"{input_size_per_partition} is not a multiple of "
            f"{INT8_SCALED_MM_ALIGN_K}, which sgl_kernel.int8_scaled_mm "
            "requires of K (mat_a.size(1) / mat_b.size(0)). Under an uneven "
            "--rank-tp-ratio this comes from a unit family too fine or too "
            "coupled to coarsen to the kernel's block; pick a shard plan whose "
            f"per-rank input is a multiple of {INT8_SCALED_MM_ALIGN_K}, or "
            "exclude this module from quantization in the checkpoint."
        )


class W8A8Int8Config(QuantizationConfig):
    """Config class for W8A8 Quantization.

    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(self, quant_config: Dict[str, Any] = {}):
        super().__init__()
        self.quant_description = quant_config
        self.is_dynamic = quant_config.get("is_dynamic", False)
        ignore = cast(List[str], quant_config.get("ignore", []))
        self.ignore = ignore if ignore is not None else []
        packed_modules_mapping = quant_config.get("packed_modules_mapping", {})
        self.packed_modules_mapping = (
            packed_modules_mapping if packed_modules_mapping is not None else {}
        )

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_name(self) -> str:
        return "w8a8_int8"

    @property
    def weight_block_size(self) -> List[int]:
        """Shard-alignment block for `--rank-tp-ratio` (see
        :func:`int8_w8a8_uneven_tp_block`).

        Not a quantisation-semantic block -- this scheme's scales are
        per-output-channel and carry no grid. The value encodes the GEMM
        alignment ``sgl_kernel.int8_scaled_mm`` requires of an uneven per-rank
        shard, which is what makes the shard valid rather than merely
        loadable."""
        return int8_w8a8_uneven_tp_block()

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        filenames = []
        return filenames

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W8A8Int8Config:
        return cls(config)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if should_ignore_layer(
            prefix, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
        ):
            return UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            return W8A8Int8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return W8A8Int8MoEMethod(self)
        return None

    def is_layer_skipped(
        self, prefix: str, fused_mapping: Mapping[str, List[str]] = MappingProxyType({})
    ):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = (
                    self.quant_description[shard_prefix + ".weight"] == "FLOAT"
                )

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision."
                    )
        else:
            is_skipped = self.quant_description[prefix + ".weight"] == "FLOAT"

        assert is_skipped is not None
        return is_skipped

    def get_scaled_act_names(self) -> List[str]:
        return []


class W8A8Int8LinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: W8A8Int8Config):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu:
            if _is_cpu_amx_available:
                _amx_process_weight_after_loading(layer, ["weight"])
            elif _is_cpu_arm64:
                layer.weight = Parameter(layer.weight.data, requires_grad=False)
            else:
                assert False, "W8A8Int8LinearMethod on CPU only works on AMX or Arm64"
        else:
            layer.weight = Parameter(layer.weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        if not _is_cpu:
            # The CPU path runs int8_scaled_mm_with_quant (AMX/VNNI), which has
            # its own layout rules; only the CUDA CUTLASS kernel carries the
            # 16/8 alignment this checks.
            verify_int8_scaled_mm_supports_shape(
                sum(output_partition_sizes),
                input_size_per_partition,
                getattr(layer, "prefix", "") or type(layer).__name__,
            )

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        if use_intel_amx_backend(layer) or _is_cpu_arm64:
            return torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
                x,
                layer.weight,
                layer.weight_scale,
                bias,
                x.dtype,
                True,  # is_vnni
            )
        x_q, x_scale = per_token_quant_int8(x)

        x_q_2d = x_q.view(-1, x_q.shape[-1])
        x_scale_2d = x_scale.view(-1, x_scale.shape[-1])
        output_shape = [*x_q.shape[:-1], layer.weight.shape[1]]

        output = int8_scaled_mm(
            x_q_2d,
            layer.weight,
            x_scale_2d,
            layer.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )

        return output.view(output_shape)


class W8A8Int8MoEMethod(FusedMoEMethodBase):
    """MoE method for INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: W8A8Int8Config):
        self.quant_config = quant_config

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

        tp_size = get_parallel().tp_size

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])
        else:
            layer.w13_weight = Parameter(layer.w13_weight, requires_grad=False)
            layer.w2_weight = Parameter(layer.w2_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        if use_intel_amx_backend(layer) or _is_cpu_arm64:
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            topk_ids = topk_ids.int()
            x, topk_weights = apply_topk_weights_cpu(
                self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace See [Note] inplace should be False in fused_experts.
                CPUQuantMethod.INT8_W8A8,
                layer.w13_weight_scale,  # w1_scale
                layer.w2_weight_scale,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                None,  # block_size
                None,  # w1 bias
                None,  # w3 bias
                None,  # alpha
                None,  # limit
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)

        quant_info = self.get_triton_quant_info(layer)
        return self.runner.run(dispatch_output, quant_info)
