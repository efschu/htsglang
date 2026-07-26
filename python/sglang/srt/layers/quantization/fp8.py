# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/fp8.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmFp8MoeQuantInfo,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    get_moe_a2a_backend,
    get_moe_padding_size,
    get_moe_runner_backend,
    get_moe_weight_sizes,
)
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
    scaled_fp8_quant,
)
from sglang.srt.layers.quantization.fp8_utils import (
    _use_aiter_bpreshuffle_gfx95,
    apply_fp8_linear,
    can_auto_enable_marlin_fp8,
    cutlass_fp8_supported,
    deepgemm_w8a8_block_fp8_linear_with_fallback,
    cached_dequant,
    dequant_fp8_block_weight,
    dequant_fp8_weight,
    dispatch_w8a8_block_fp8_linear,
    dispatch_w8a8_mxfp8_linear,
    fp8_needs_dequant_fallback,
    get_fp8_gemm_runner_backend,
    input_to_float8,
    mxfp8_group_quantize,
    normalize_e4m3fn_to_e4m3fnuz,
    requant_block_scale_ue8m0_for_deepgemm,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.quantization.utils import (
    all_close_1d,
    convert_to_channelwise,
    is_layer_skipped,
    per_tensor_dequantize,
    requantize_with_max_scale,
)
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_musa,
    is_npu,
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
    log_info_on_rank0,
    mxfp8_block_convert_required,
    print_warning_once,
    set_weight_attrs,
    use_intel_amx_backend,
    use_intel_xpu_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.aiter import AiterMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config
    from sglang.srt.models.utils import WeightsMapper

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_fp8_fnuz = is_fp8_fnuz()
_is_gfx95_supported = is_gfx95_supported()
# gfx942 (MI300) has no MX matmul HW; MXFP8 checkpoints are converted to
# block-fp8 [128,128] at load and run through the native block-fp8 kernels.
_mxfp8_to_block_fp8_required = mxfp8_block_convert_required()
_use_hip_int4 = get_bool_env_var("SGLANG_INT4_WEIGHT") and _is_hip
_use_aiter = envs.SGLANG_USE_AITER.get() and _is_hip
_is_shuffle_moe_mxfp4 = is_gfx95_supported()


def _require_fp4_dtype():
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        raise RuntimeError(
            "DeepSeek-V4 FP4 experts require torch.float4_e2m1fn_x2 support."
        )
    return fp4_dtype


if _use_aiter or _use_hip_int4:
    from aiter.ops.shuffle import (
        shuffle_scale,
        shuffle_weight,
    )

if _use_aiter:
    from sglang.srt.layers.quantization.fp8_utils import (
        aiter_w8a8_block_fp8_linear,
        use_aiter_triton_gemm_w8a8_tuned_gfx950,
    )


ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)

DSV4_DEQUANT_FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def cast_e2m1fn_to_e4m3fn(
    x: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Casts a tensor from e2m1fn to e4m3fn losslessly.
    """
    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    fp8_block_size = 128
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    table = DSV4_DEQUANT_FP4_TABLE.to(x.device)
    x = torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(2)

    # max_fp4 (6.0) * MAX_OFFSET must fit in e4m3fn (max 448)
    # 6.0 * 2^6 = 384 < 448; 6.0 * 2^7 = 768 > 448; so MAX_OFFSET_BITS = 6
    MAX_OFFSET_BITS = 6

    bOut = out_dim // fp8_block_size
    bIn = in_dim // fp8_block_size
    # bOut, bIn, 128, 128
    x = x.view(bOut, fp8_block_size, bIn, fp8_block_size).transpose(1, 2)
    # bOut, bIn, 128*4
    scale = scale.float().view(bOut, fp8_block_size, bIn, -1).transpose(1, 2).flatten(2)
    ## bOut, bIn, 1
    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**MAX_OFFSET_BITS)
    # bOut, bIn, 128*4
    offset = scale / scale_max_offset_bits
    # bOut, bIn, 128, 128
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(
        fp4_block_size, dim=-1
    )
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(
        torch.float8_e8m0fnu
    )


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        use_mxfp8: bool = False,
        is_fp4_experts: bool = False,
    ) -> None:
        super().__init__()
        # DSV4 mxfp4-packed (True) vs converted FP8 (False); injected by
        # model_loader from ModelConfig. Default False off the DSV4 path.
        self.is_fp4_experts = is_fp4_experts
        self.dequant_fp4_to_fp8 = False
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            log_info_on_rank0(logger, "Detected fp8 checkpoint.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if ignored_layers_str := envs.SGLANG_FP8_IGNORED_LAYERS.get():
            self.ignored_layers.extend(
                [
                    layer.strip()
                    for layer in ignored_layers_str.split(",")
                    if layer.strip()
                ]
            )
        self.packed_modules_mapping = packed_modules_mapping or {}
        self.use_mxfp8 = use_mxfp8
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    f"The block-wise quantization only supports fp8-serialized checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    f"The quantization block size of weight must have 2 dimensions, but got {len(weight_block_size)} dimensions."
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    f"The block-wise quantization only supports dynamic activation scheme for now, but got {activation_scheme} activation scheme."
                )
        if self.use_mxfp8:
            if weight_block_size is None:
                weight_block_size = [1, 32]
            elif weight_block_size != [1, 32]:
                raise ValueError("MXFP8 requires weight_block_size=[1, 32].")
        self.weight_block_size = weight_block_size

    def get_name(self) -> str:
        return "mxfp8" if self.use_mxfp8 else "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    def get_min_capability(self) -> int:
        if is_npu():
            return 0  # NPU bypasses CUDA capability checks
        if _is_musa:
            return 31
        if self.use_mxfp8 and _is_hip and _is_gfx95_supported:
            return 95
        if self.use_mxfp8 and _mxfp8_to_block_fp8_required:
            return 94

        return 100 if self.use_mxfp8 else 80

    def needs_device_kernel(self) -> bool:
        """Whether this config depends on a device kernel with a capability floor.

        A plain (non-block, non-mxfp8) fp8 checkpoint can be served by
        dequantising the weights and using F.linear, which needs no fp8 kernel
        and therefore has no floor. Block-scaled and mxfp8 checkpoints keep
        their floor: their scale layouts have no plain-torch route here.

        Instance-level, because the answer depends both on the checkpoint and
        on what the device can actually do.
        """
        # mxfp8 keeps its floor: UE8M0 scales have no plain-torch route here.
        # BLOCK-scaled fp8 does have one (dequant_fp8_block_weight), so it is
        # treated like the per-tensor/per-channel case: no kernel, no floor.
        if self.use_mxfp8:
            return True
        return not fp8_needs_dequant_fallback()

    def supports_current_device(self) -> Optional[bool]:
        """Vendor-aware support answer; see QuantizationConfig.

        Only answers outside the NVIDIA namespace, where the numeric floor is
        meaningless. There the question is asked FUNCTIONALLY -- does a native
        fp8 GEMM exist on this device? -- which keeps MI300-class parts working
        exactly as before while giving gfx900 a clean "no" at startup instead
        of a missing-kernel death later.

        On CUDA this returns None on purpose, so the caller falls through to the
        unchanged numeric comparison and NVIDIA behaviour does not move.
        """
        if torch.version.hip is None:
            return None
        from sglang.srt.layers.quantization.fp8_utils import (
            fp8_native_gemm_available,
        )

        return fp8_native_gemm_available()

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Fp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        use_mxfp8 = "mxfp8" in quant_method
        is_checkpoint_fp8_serialized = ("fp8" in quant_method) or use_mxfp8
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        packed_modules_mapping = (
            cls.get_from_keys_or(config, ["packed_modules_mapping"], {}) or {}
        )
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        if ignored_layers:
            # Keep both "model." and non-"model." variants for robust prefix matching.
            normalized = []
            for layer in ignored_layers:
                base = layer.removeprefix("model.")
                normalized.append(base)
                normalized.append(f"model.{base}")
            ignored_layers = normalized
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        if use_mxfp8:
            # MXFP8 (OCP) spec fixes block size to [1, 32]; ckpt field is metadata only.
            if weight_block_size is not None and weight_block_size != [1, 32]:
                logger.warning(
                    "MXFP8 overriding weight_block_size=%s from config.json -> [1, 32].",
                    weight_block_size,
                )
            weight_block_size = [1, 32]
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            packed_modules_mapping=packed_modules_mapping,
            use_mxfp8=use_mxfp8,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.radix_attention import RadixAttention

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedLinearMethod()
            if is_npu() and self.use_mxfp8:
                from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
                    NPUMXFP8LinearMethod,
                )

                return NPUMXFP8LinearMethod(self)
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedFusedMoEMethod(
                    layer.use_triton_kernels, layer.use_flashinfer_trtllm_moe
                )

            fp8_method = Fp8MoEMethod(self)

            if self.is_fp4_experts and self.dequant_fp4_to_fp8:
                assert (
                    get_moe_runner_backend().is_auto()
                ), f"{get_moe_runner_backend()} is not compatible with SGLANG_DSV4_FP4_DEQUANT=1"
                return fp8_method

            if self.is_fp4_experts and get_moe_runner_backend().is_marlin():
                from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
                    Mxfp4MarlinMoEMethod,
                )

                return Mxfp4MarlinMoEMethod(fp8_method, prefix=prefix)

            if self.is_fp4_experts and get_moe_runner_backend().is_flashinfer_mxfp4():
                # SM100 (Blackwell) -> trtllm-gen path.
                # SM90  (Hopper)    -> cutlass mixed-input path (FlashInfer #3084).
                if is_sm90_supported() and not is_sm100_supported():
                    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
                        Mxfp4FlashinferCutlassMoEMethod,
                    )

                    return Mxfp4FlashinferCutlassMoEMethod(fp8_method, prefix=prefix)

                from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
                    Mxfp4FlashinferTrtllmMoEMethod,
                )

                return Mxfp4FlashinferTrtllmMoEMethod(fp8_method, prefix=prefix)
            return fp8_method
        elif isinstance(layer, RadixAttention):
            return Fp8KVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: WeightsMapper):
        if self.ignored_layers:
            self.ignored_layers = list(
                dict.fromkeys(hf_to_sglang_mapper.apply_list(self.ignored_layers))
            )


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.

    It supports the following quantization schemes:
    - Per-channel weight quantization + per-token activation quantization
    - Per-tensor weight quantization + per-tensor activation quantization
    - Blockwise weight quantization + blockwise activation quantization

    It supports the following checkpoint formats:
    - FP8 checkpoint
    - FP16/BF16 checkpoint. In this case, the weights will be quantized to FP8 during the weight loading.

    Notes:
    - The activation quantization scheme can be static or dynamic. The dynamic activation quantization is more commonly used.
    - On NV platforms, the per-channel weight quantization is used by default, if block quantization is not enabled.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Union[Fp8Config, W4AFp8Config]):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = False
        if _is_cuda:
            force_marlin = get_bool_env_var("SGLANG_FORCE_FP8_MARLIN")
            auto_enable = can_auto_enable_marlin_fp8()
            self.use_marlin = force_marlin or auto_enable

        self.use_mxfp8 = getattr(self.quant_config, "use_mxfp8", False)
        self.block_quant = (
            self.use_mxfp8 or self.quant_config.weight_block_size is not None
        )
        # Last resort: no native fp8 GEMM, no cutlass, no Marlin. Weights stay
        # fp8 in VRAM and are dequantised per forward (W8A16); activations are
        # never quantised. Only for plain per-tensor/per-channel checkpoints --
        # block-scaled and mxfp8 layouts have no plain-torch route here.
        self.use_dequant = (
            not self.block_quant and not self.use_marlin and fp8_needs_dequant_fallback()
        )
        # Block-scaled fp8 with no block-fp8 GEMM anywhere: dequantise the
        # [block_n, block_k] tiles per forward. mxfp8 is excluded -- its UE8M0
        # scales are a different encoding with no plain-torch route here.
        self.use_block_dequant = (
            self.block_quant and not self.use_mxfp8 and fp8_needs_dequant_fallback()
        )
        self.convert_mxfp8_to_block = self.use_mxfp8 and _mxfp8_to_block_fp8_required
        self.weight_block_size = self.quant_config.weight_block_size
        self.w8a8_block_fp8_linear = None
        self.w8a8_mxfp8_linear = None
        if self.use_block_dequant:
            # Do not dispatch a block-fp8 GEMM runner we have already
            # established does not exist on this device; the dispatch itself
            # can import kernels this platform cannot build.
            pass
        elif self.use_mxfp8 and not self.convert_mxfp8_to_block:
            self.w8a8_mxfp8_linear = dispatch_w8a8_mxfp8_linear()
        else:
            self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()
        self.is_checkpoint_fp8_serialized = (
            self.quant_config.is_checkpoint_fp8_serialized
        )
        self.use_aiter_fp8_per_token = envs.SGLANG_USE_AITER_FP8_PER_TOKEN.get()
        self.use_per_token_if_dynamic = False

    def validate_block_quant_shapes(
        self,
        input_size: int,
        input_size_per_partition: int,
        output_size: int,
        output_size_per_partition: int,
        output_partition_sizes: List[int],
        skip_block_quant_check: bool = False,
    ):
        tp_size = get_parallel().tp_size
        block_n, block_k = (
            self.quant_config.weight_block_size[0],
            self.quant_config.weight_block_size[1],
        )

        if skip_block_quant_check:
            print_warning_once(
                "Skipping block quantization checks for weight partition."
            )
        else:
            # Required by row parallel
            if tp_size > 1 and input_size // input_size_per_partition == tp_size:
                if input_size_per_partition % block_k != 0:
                    raise ValueError(
                        f"Weight input_size_per_partition = "
                        f"{input_size_per_partition} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )
            # Required by column parallel or enabling merged weights
            if (
                tp_size > 1 and output_size // output_size_per_partition == tp_size
            ) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}."
                        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        skip_block_quant_check: bool = False,
        **extra_weight_attrs,
    ):
        # Copy the layer attributes
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")

        if self.block_quant:
            block_n, block_k = self.quant_config.weight_block_size
            self.validate_block_quant_shapes(
                input_size,
                input_size_per_partition,
                output_size,
                output_size_per_partition,
                output_partition_sizes,
                skip_block_quant_check,
            )

        # Create the weight
        weight_dtype = (
            torch.float8_e4m3fn if self.is_checkpoint_fp8_serialized else params_dtype
        )
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if self.block_quant:
                if hasattr(self.quant_config, "activation_scheme"):
                    assert self.quant_config.activation_scheme == "dynamic"
                elif hasattr(self.quant_config, "linear_activation_scheme"):
                    assert self.quant_config.linear_activation_scheme == "dynamic"
                if self.use_mxfp8 and not self.is_checkpoint_fp8_serialized:
                    raise ValueError(
                        "MXFP8 requires fp8-serialized checkpoint for linear layers."
                    )
                scale_dtype = torch.uint8 if self.use_mxfp8 else torch.float32
                scale_init = torch.zeros if scale_dtype == torch.uint8 else torch.empty
                scale = BlockQuantScaleParameter(
                    data=scale_init(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=scale_dtype,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale.format_ue8m0 = self.use_mxfp8
                if scale_dtype != torch.uint8:
                    scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale_inv", scale)
            else:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)

            # INPUT ACTIVATION SCALE
            if (
                hasattr(self.quant_config, "activation_scheme")
                and self.quant_config.activation_scheme == "static"
            ) or (
                hasattr(self.quant_config, "linear_activation_scheme")
                and self.quant_config.linear_activation_scheme == "static"
            ):
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )

                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading_block_quant(self, layer: Module) -> None:
        if self.convert_mxfp8_to_block:
            from sglang.srt.layers.quantization.mxfp8_block_convert import (
                convert_mxfp8_weight_to_block_fp8,
            )

            qweight, scale = convert_mxfp8_weight_to_block_fp8(
                layer.weight.data, layer.weight_scale_inv.data, block=128
            )
            layer.weight = Parameter(qweight, requires_grad=False)
            layer.weight_scale_inv = Parameter(scale, requires_grad=False)
            self.use_mxfp8 = False
            self.convert_mxfp8_to_block = False
            self.weight_block_size = [128, 128]
        elif self.use_mxfp8:
            # MXFP8 (e4m3fn + UE8M0) must NOT be fnuz-normalized; check before
            # the fnuz branch since is_fp8_fnuz() is also True on gfx942.
            if not self.is_checkpoint_fp8_serialized:
                self._quantize_mxfp8_weights(layer)
                return
            # MXFP8 scales are stored as UE8M0 uint8; no requantization here.
            # Keep parameter object to preserve weight_loader attrs for hot reload.
            layer.weight_scale_inv.requires_grad_(False)
            layer.weight_scale_inv.format_ue8m0 = True
            self._process_mxfp8_linear_weight_scale(layer)
            return
        # If ROCm, normalize the weights and scales to e4m3fnuz
        if _is_fp8_fnuz:
            # activation_scheme: dynamic
            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.weight,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
            )
            layer.input_scale = None
        elif _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "Fp8LinearMethod on CPU requires that CPU has AMX support"
            _amx_process_weight_after_loading(layer, ["weight"])
            layer.weight_scale_inv = torch.nn.Parameter(
                layer.weight_scale_inv.data, requires_grad=False
            )
            return
        else:
            # Requantize block scales to UE8M0 when DeepGEMM is the active runner.
            use_deepgemm_runner = (
                self.w8a8_block_fp8_linear
                is deepgemm_w8a8_block_fp8_linear_with_fallback
            )
            requant_block_scale_ue8m0_for_deepgemm(
                layer.weight,
                layer.weight_scale_inv,
                getattr(self.quant_config, "weight_block_size", None),
                use_deepgemm_runner=use_deepgemm_runner,
                output_dtype=getattr(layer, "orig_dtype", None),
                weight_shape=layer.weight.shape,
            )
            weight, weight_scale = layer.weight.data, layer.weight_scale_inv.data

        layer.weight.data = weight.data
        layer.weight_scale_inv.data = weight_scale.data

        if (
            _use_aiter_bpreshuffle_gfx95
            and self.w8a8_block_fp8_linear is aiter_w8a8_block_fp8_linear
        ):
            n, k = layer.weight.shape
            if not use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k):
                # TODO(1am9trash), to deal with case that this branch chance
                # drops as use_aiter_triton_gemm_w8a8_tuned_gfx950() expands
                t = shuffle_weight(layer.weight, (16, 16))
                layer.weight.copy_(t)
                del t

    def _process_mxfp8_linear_weight_scale(self, layer: Module) -> None:
        if not self.use_mxfp8:
            return

        backend = get_fp8_gemm_runner_backend()
        if backend.is_flashinfer_trtllm():
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = layer.weight.data
            scale_u8 = layer.weight_scale_inv.data
            n, k = weight.shape
            epilogue_tile_m = 128
            sf_cols = k // 32

            scale_u8 = scale_u8.contiguous().view(torch.uint8).reshape(n, sf_cols)
            padded_n = ((n + epilogue_tile_m - 1) // epilogue_tile_m) * (
                epilogue_tile_m
            )
            pad_rows = padded_n - n

            if pad_rows:
                scale_u8 = F.pad(
                    scale_u8,
                    (0, 0, 0, pad_rows),
                    mode="constant",
                    value=0,
                )

            copy_or_rebind_param(
                layer,
                "weight",
                shuffle_matrix_a(
                    weight.contiguous().view(torch.uint8), epilogue_tile_m
                ).view(torch.float8_e4m3fn),
            )
            copy_or_rebind_param(
                layer,
                "weight_scale_inv_shuffled",
                shuffle_matrix_sf_a(
                    scale_u8,
                    epilogue_tile_m,
                    num_elts_per_sf=32,
                )
                .reshape_as(scale_u8)
                .contiguous(),
            )
        elif backend.is_flashinfer_cutlass():
            from flashinfer import block_scale_interleave

            scale_u8 = layer.weight_scale_inv.data
            # block_scale_interleave may pad and/or reshape scales,
            # so store swizzled scales separately to keep weight update working
            copy_or_rebind_param(
                layer,
                "weight_scale_inv_swizzled",
                block_scale_interleave(scale_u8.contiguous()).contiguous(),
            )
        elif get_fp8_gemm_runner_backend().is_deep_gemm():
            from sglang.srt.layers.deep_gemm_wrapper.configurer import (
                DEEPGEMM_SCALE_UE8M0,
            )

            n, k = layer.weight.shape
            scale_u8 = layer.weight_scale_inv.data
            scale_fp32 = (
                (scale_u8.contiguous().view(-1).to(torch.int32) << 23)
                .view(torch.float32)
                .view(n, k // 32)
            )
            if DEEPGEMM_SCALE_UE8M0:
                # Pre-packed; GEMM must be called with disable_ue8m0_cast=True.
                import deep_gemm.utils.layout

                scale_packed = (
                    deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(
                        scale_fp32
                    )
                )
            else:
                scale_packed = scale_fp32
            copy_or_rebind_param(layer, "weight_scale_inv_deepgemm", scale_packed)
        else:
            # Triton path consumes canonical 2D UE8M0 uint8 scales directly.
            return

    def _quantize_mxfp8_weights(self, layer: Module) -> None:
        weight = layer.weight.data
        qweight, weight_scale = mxfp8_group_quantize(weight)
        # Keep parameter objects to preserve weight_loader attrs for hot reload.
        layer.weight.data = qweight
        layer.weight.requires_grad_(False)
        if hasattr(layer, "weight_scale_inv") and layer.weight_scale_inv is not None:
            layer.weight_scale_inv.data = weight_scale
            layer.weight_scale_inv.requires_grad_(False)
        else:
            # First-time online MXFP8 quantization (no serialized scales).
            layer.register_parameter(
                "weight_scale_inv", Parameter(weight_scale, requires_grad=False)
            )
        layer.weight_scale_inv.format_ue8m0 = True
        self._process_mxfp8_linear_weight_scale(layer)
        layer.input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        if self.block_quant:
            self.process_weights_after_loading_block_quant(layer)
        else:
            layer.weight = Parameter(layer.weight.data, requires_grad=False)

            # If checkpoint not serialized fp8, quantize the weights.
            if not self.is_checkpoint_fp8_serialized:
                if (
                    self.cutlass_fp8_supported
                    or self.use_marlin
                    or (_use_aiter and self.use_aiter_fp8_per_token)
                ):
                    # apply per-channel quantization default as
                    # cutlass sgl-kernel and marlin only support per-channel scale
                    qweight, weight_scale = per_token_group_quant_fp8(
                        layer.weight, layer.weight.shape[-1]
                    )
                    weight_scale = weight_scale.t().contiguous()
                    if _use_aiter and self.use_aiter_fp8_per_token:
                        self.use_per_token_if_dynamic = True
                        qweight = shuffle_weight(qweight.contiguous(), (16, 16))
                else:
                    # per-tensor quantization
                    qweight, weight_scale = input_to_float8(layer.weight)

                # Update the layer with the new values.
                layer.weight = Parameter(qweight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                layer.input_scale = None

            # If checkpoint is fp8, handle that there are N scales for N
            # shards in a fused module
            else:
                layer.weight_scale = Parameter(
                    layer.weight_scale.data, requires_grad=False
                )
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.data, requires_grad=False
                    )

                # cutlass sgl-kernel and marlin only support per-channel scale; aiter supports per-channel scale
                # The dequant fallback joins this branch on purpose: it wants
                # per-channel scales too, and -- more importantly -- this branch
                # is the one that needs NO kernel. The per-tensor branch below
                # requantises via scaled_fp8_quant, which is exactly the kind of
                # kernel the fallback exists because the device does not have.
                # Keeping the loaded scales also avoids a requantisation step,
                # so the fallback is if anything slightly more accurate.
                if (
                    self.cutlass_fp8_supported
                    or self.use_marlin
                    or self.use_dequant
                    or (_use_aiter and self.use_aiter_fp8_per_token)
                ):
                    weight = layer.weight
                    weight_scale = convert_to_channelwise(
                        layer.weight_scale, layer.logical_widths
                    )
                    if _use_aiter and self.use_aiter_fp8_per_token:
                        # Otherwise, by default, aiter only uses per-tensor quantization
                        self.use_per_token_if_dynamic = True
                        if _is_fp8_fnuz:
                            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                weight=weight,
                                weight_scale=weight_scale,
                            )
                        weight = shuffle_weight(weight.contiguous(), (16, 16))
                else:
                    # Dequant -> Quant with max scale so we can run per tensor.
                    weight = layer.weight
                    weight_scale = layer.weight_scale
                    # If ROCm, normalize the weights and scales to e4m3fnuz
                    if _is_fp8_fnuz:
                        weight, weight_scale, input_scale = (
                            normalize_e4m3fn_to_e4m3fnuz(
                                weight=weight,
                                weight_scale=weight_scale,
                                input_scale=layer.input_scale,
                            )
                        )
                        if input_scale is not None:
                            layer.input_scale = Parameter(
                                input_scale, requires_grad=False
                            )

                    weight_scale, weight = requantize_with_max_scale(
                        weight=weight,
                        weight_scale=weight_scale,
                        logical_widths=layer.logical_widths,
                    )

                # Update layer with new values.
                layer.weight = Parameter(weight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.max(), requires_grad=False
                    )

        if self.use_marlin:
            if self.block_quant:
                layer.weight_block_size = self.quant_config.weight_block_size
            prepare_fp8_layer_for_marlin(layer, not self.block_quant)
            # Activations not quantized for marlin.
            del layer.input_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_marlin:
            return torch.ops.sglang.apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

        if self.use_mxfp8:
            backend = get_fp8_gemm_runner_backend()
            if backend.is_flashinfer_cutlass():
                weight_scale = layer.weight_scale_inv_swizzled
            elif backend.is_flashinfer_trtllm():
                weight_scale = layer.weight_scale_inv_shuffled
            elif get_fp8_gemm_runner_backend().is_deep_gemm():
                weight_scale = getattr(
                    layer, "weight_scale_inv_deepgemm", layer.weight_scale_inv
                )
                if isinstance(x, tuple):
                    return self.w8a8_mxfp8_linear(
                        input=x[0],
                        weight=layer.weight,
                        weight_scale=weight_scale,
                        input_scale=x[1],
                        bias=bias,
                        weight_scale_fallback=layer.weight_scale_inv,
                    )
                return self.w8a8_mxfp8_linear(
                    input=x,
                    weight=layer.weight,
                    weight_scale=weight_scale,
                    input_scale=None,
                    bias=bias,
                    weight_scale_fallback=layer.weight_scale_inv,
                )
            else:
                weight_scale = layer.weight_scale_inv
            if isinstance(x, tuple):
                return self.w8a8_mxfp8_linear(
                    input=x[0],
                    weight=layer.weight,
                    weight_scale=weight_scale,
                    input_scale=x[1],
                    bias=bias,
                )
            return self.w8a8_mxfp8_linear(
                input=x,
                weight=layer.weight,
                weight_scale=weight_scale,
                input_scale=None,
                bias=bias,
            )

        if self.use_block_dequant:
            # W8A16, block-scaled: one scale per [block_n, block_k] tile,
            # expanded per forward. Weights stay fp8 resident; activations are
            # never quantised, so layer.input_scale plays no part. Unlike the
            # per-channel case this cannot be folded into a vector multiply --
            # the scale varies along BOTH axes -- so it costs more, which is
            # measured rather than assumed.
            if isinstance(x, tuple):
                x = x[0]  # (input, scale) pairs only arrive on quantised paths
            # Cached when a budget is set: the expansion is a pure function of
            # (weight, scale, block_size, dtype), so it need not be redone every
            # forward. Measured cost of NOT caching on this rig: -69.9% decode.
            return torch.nn.functional.linear(
                x,
                cached_dequant(
                    layer.weight,
                    x.dtype,
                    lambda: dequant_fp8_block_weight(
                        layer.weight,
                        layer.weight_scale_inv,
                        self.weight_block_size,
                        x.dtype,
                    ),
                ),
                bias,
            )

        if self.use_dequant:
            # W8A16: weights are dequantised per forward, activations stay in
            # their compute dtype and are never quantised, so layer.input_scale
            # (if the checkpoint carried one) is deliberately unused.
            # layer.weight was stored transposed as (in, out); .t() is a view
            # back to (out, in), which is what F.linear wants. weight_scale is
            # either per-tensor (scalar) or per-channel (out, 1); both
            # broadcast correctly against (out, in).
            return torch.nn.functional.linear(
                x,
                cached_dequant(
                    layer.weight,
                    x.dtype,
                    lambda: dequant_fp8_weight(
                        layer.weight.t(), layer.weight_scale, x.dtype
                    ),
                ),
                bias,
            )

        if self.block_quant:
            if use_intel_amx_backend(layer):
                return torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
                    x,
                    layer.weight,
                    layer.weight_scale_inv,
                    self.weight_block_size,
                    bias,
                    x.dtype,
                    True,  # is_vnni
                )

            if isinstance(x, tuple):
                return self.w8a8_block_fp8_linear(
                    input=x[0],
                    weight=layer.weight,
                    block_size=self.weight_block_size,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=x[1],
                    bias=bias,
                )

            return self.w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
                bias=bias,
            )

        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
        )


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.use_mxfp8 = getattr(self.quant_config, "use_mxfp8", False)
        self.block_quant = (
            self.use_mxfp8 or self.quant_config.weight_block_size is not None
        )
        self.convert_mxfp8_to_block = self.use_mxfp8 and _mxfp8_to_block_fp8_required
        self.weight_block_size = self.quant_config.weight_block_size
        self.is_fp4_expert = self.quant_config.is_fp4_experts
        self.dequant_fp4_to_fp8 = self.quant_config.dequant_fp4_to_fp8
        self.with_bias = False
        # For GPUs that lack native FP8 compute (sm < 89, e.g. sm86), fall back
        # to the Marlin W8A16 fused-MoE kernel — mirrors the dense
        # Fp8LinearMethod.use_marlin path. NOTE: this decision is RANK-LOCAL
        # (capability of the current device). Under mixed-arch TP (e.g. sm120 +
        # sm86 ranks) some ranks run the native fp8 path while others run
        # Marlin; that is numerically fine — expert shards are disjoint per
        # rank and the TP all-reduce combines partial outputs, exactly like
        # the existing dense hetero path. Shard geometry / weight loading is
        # identical for both paths (Marlin conversion is post-load, in
        # process_weights_after_loading).
        self.use_marlin = False
        if (
            _is_cuda
            and not self.use_mxfp8
            and not self.is_fp4_expert
            and (
                get_moe_runner_backend().is_auto()
                or get_moe_runner_backend().is_triton()
            )
        ):
            force_marlin = get_bool_env_var("SGLANG_FORCE_FP8_MARLIN")
            self.use_marlin = force_marlin or can_auto_enable_marlin_fp8()
        if get_moe_runner_backend().is_cutlass():
            assert (
                cutlass_fp8_supported()
            ), "cutlass_fp8 MoE requires CUDA 12.0+ with SM90 or CUDA 12.4+ with SM89"
            assert self.block_quant, "cutlass_fp8 MoE requires block quantization"
            assert (
                is_sm100_supported() or is_sm90_supported() or is_sm120_supported()
            ), "cutlass_fp8 MoE requires SM90, SM100, or SM120 GPUs"

    @staticmethod
    def is_deepgemm_moe_runner_backend_enabled() -> bool:
        """Check if MoE will actually use DeepGEMM runner for FP8."""
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_deep_gemm():
            return True
        if moe_runner_backend.is_auto():
            return deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and (
                get_moe_a2a_backend().is_deepep()
                or get_moe_a2a_backend().is_mooncake()
                or get_moe_a2a_backend().is_nixl()
            )
        return False

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # Preserve the activation dtype before it is shadowed by the fp8
        # storage dtype below; the Marlin fallback needs it for scale
        # conversion (mirrors Fp8LinearMethod.create_weights).
        layer.orig_dtype = params_dtype

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.uint32 if _use_hip_int4 else torch.float8_e4m3fn
        tp_size = get_parallel().tp_size

        w13_up_dim, w2_up_dim, weight_padded = get_moe_weight_sizes(
            intermediate_size_per_partition,
            is_aiter_moe=_use_aiter,
            is_concat=True,
            is_packed=False,
        )

        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )

            padding_size = get_moe_padding_size(_use_aiter)
            if not (_use_aiter and padding_size == block_n == block_k):
                # NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
                # Required by column parallel or enabling merged weights
                if intermediate_size_per_partition % block_n != 0:
                    raise ValueError(
                        f"The output_size of gate's and up's weight = "
                        f"{intermediate_size_per_partition} is not divisible by "
                        f"weight quantization block_n = {block_n}."
                    )
                if tp_size > 1:
                    # Required by row parallel
                    if intermediate_size_per_partition % block_k != 0:
                        raise ValueError(
                            f"The input_size of down's weight = "
                            f"{intermediate_size_per_partition} is not divisible by "
                            f"weight quantization block_k = {block_k}."
                        )

        # WEIGHTS
        if self.is_fp4_expert:
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // 2,
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // 2,
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
        elif _is_hip and _use_hip_int4:
            # INT4 MoE weight - INT32 packed
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // 8,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // 8,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
        else:
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    w13_up_dim,
                    hidden_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    w2_up_dim,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )

        extra_weight_attrs.update(
            {"weight_padded": weight_padded},
        )

        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # BIAS (optional, e.g. GPT-OSS)
        if self.with_bias:
            w13_up_dim = (
                2 * intermediate_size_per_partition
                if layer.moe_runner_config.is_gated
                else intermediate_size_per_partition
            )
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.is_fp4_expert:
            fp4_block_k = 32
            fp4_scale_dtype = torch.float8_e8m0fnu if _use_aiter else torch.float32
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // fp4_block_k,
                    dtype=fp4_scale_dtype,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // fp4_block_k,
                    dtype=fp4_scale_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        elif self.block_quant:
            scale_dtype = torch.uint8 if self.use_mxfp8 else torch.float32
            scale_init = torch.zeros if scale_dtype == torch.uint8 else torch.ones
            w13_weight_scale = torch.nn.Parameter(
                scale_init(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                scale_init(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
            # w13_weight and w2_weight are always requanted together
            w13_weight_scale.format_ue8m0 = self.use_mxfp8
            w2_weight_scale.format_ue8m0 = self.use_mxfp8
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"
            if get_moe_runner_backend().is_cutlass():
                self._ensure_cutlass_buffers_initialized(layer)

        else:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

            if _is_hip:  # _use_aiter: TODO: add check back after triton kernel
                # ROCm - using column scaling, duplicate scaling numbers in case per tensor scaling
                w13_weight_scale1 = torch.nn.Parameter(
                    torch.ones(
                        num_experts,
                        2 * intermediate_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                w2_weight_scale1 = torch.nn.Parameter(
                    torch.ones(num_experts, hidden_size, dtype=torch.float32),
                    requires_grad=False,
                )
                layer.register_parameter("w13_weight_scale1", w13_weight_scale1)
                layer.register_parameter("w2_weight_scale1", w2_weight_scale1)

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

            if _is_hip and _use_hip_int4:
                extra_weight_attrs.update(
                    {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
                )
                set_weight_attrs(w13_weight_scale1, extra_weight_attrs)
                set_weight_attrs(w2_weight_scale1, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading_block_quant(self, layer: Module) -> None:
        # AMD FP4 experts: use aiter's native MXFP4 MoE path
        if _use_aiter and self.is_fp4_expert:
            gu_intv = envs.SGLANG_USE_AITER_MOE_GU_ITLV.get()
            fp4_weight_dtype = _require_fp4_dtype()

            # CK FP4 MoE kernel requires K_packed divisible by 128
            # (i.e., K_logical divisible by 256).
            # Pad intermediate_size_per_partition if needed.
            fp4_k_align = 256
            E, w13_N, w13_K_packed = layer.w13_weight.shape
            _, w2_N, w2_K_packed = layer.w2_weight.shape
            inter_per_part = w13_N // 2
            padded_inter = (
                (inter_per_part + fp4_k_align - 1) // fp4_k_align * fp4_k_align
            )
            # Record the padding so fused_moe is told the real intermediate size
            # (aiter fused_moe needs intermediate_pad = padded - real; ATOM passes
            # 128, SGLang previously defaulted to 0 -> computed the padded region).
            layer.intermediate_pad = padded_inter - inter_per_part
            layer.hidden_pad = 0
            if padded_inter != inter_per_part:
                pad_amount = padded_inter - inter_per_part
                fp4_block_k = 32

                # Pad w13_weight: (E, 2*inter, K_packed) → (E, 2*padded, K_packed)
                old_w13 = layer.w13_weight.data
                new_w13 = torch.zeros(
                    E,
                    2 * padded_inter,
                    w13_K_packed,
                    dtype=old_w13.dtype,
                    device=old_w13.device,
                )
                new_w13[:, :inter_per_part, :] = old_w13[:, :inter_per_part, :]
                new_w13[:, padded_inter : padded_inter + inter_per_part, :] = old_w13[
                    :, inter_per_part:, :
                ]
                layer.w13_weight = torch.nn.Parameter(new_w13, requires_grad=False)

                # Pad w2_weight: (E, N, inter_packed) → (E, N, padded_packed)
                old_w2 = layer.w2_weight.data
                new_w2 = torch.zeros(
                    E,
                    w2_N,
                    padded_inter // 2,
                    dtype=old_w2.dtype,
                    device=old_w2.device,
                )
                new_w2[:, :, :w2_K_packed] = old_w2
                layer.w2_weight = torch.nn.Parameter(new_w2, requires_grad=False)

                # Pad w13 scale: (E, 2*inter, K/block_k) → (E, 2*padded, K/block_k)
                old_s13 = layer.w13_weight_scale_inv.data
                _, _, s13_K = old_s13.shape
                new_s13 = torch.zeros(
                    E,
                    2 * padded_inter,
                    s13_K,
                    dtype=old_s13.dtype,
                    device=old_s13.device,
                )
                new_s13[:, :inter_per_part, :] = old_s13[:, :inter_per_part, :]
                new_s13[:, padded_inter : padded_inter + inter_per_part, :] = old_s13[
                    :, inter_per_part:, :
                ]
                layer.w13_weight_scale_inv = torch.nn.Parameter(
                    new_s13, requires_grad=False
                )

                # Pad w2 scale: (E, N, inter/block_k) → (E, N, padded/block_k)
                old_s2 = layer.w2_weight_scale_inv.data
                new_s2 = torch.zeros(
                    E,
                    w2_N,
                    padded_inter // fp4_block_k,
                    dtype=old_s2.dtype,
                    device=old_s2.device,
                )
                new_s2[:, :, : old_s2.shape[2]] = old_s2
                layer.w2_weight_scale_inv = torch.nn.Parameter(
                    new_s2, requires_grad=False
                )

            for scale_name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
                scale = getattr(layer, scale_name)
                num_experts, num_rows, _ = scale.shape
                is_w13_scale = scale_name == "w13_weight_scale_inv"
                scale_2d = scale.reshape(-1, scale.shape[-1])
                scale.data = shuffle_scale(scale_2d, num_experts, gu_intv, is_w13_scale)

            layer.w13_weight.data = layer.w13_weight.data.view(fp4_weight_dtype)
            layer.w2_weight.data = layer.w2_weight.data.view(fp4_weight_dtype)

            is_shuffled = _is_shuffle_moe_mxfp4
            if is_shuffled:
                layer.w13_weight.data = shuffle_weight(
                    layer.w13_weight,
                    is_guinterleave=gu_intv,
                    gate_up=True,
                )
                layer.w2_weight.data = shuffle_weight(
                    layer.w2_weight,
                    is_guinterleave=gu_intv,
                    gate_up=False,
                )
            layer.w13_weight.is_shuffled = is_shuffled
            layer.w2_weight.is_shuffled = is_shuffled
            return

        if self.convert_mxfp8_to_block:
            # Only aiter-shuffle when the MoE runner is aiter; the triton runner
            # consumes un-shuffled weights (shuffling the wrong runner corrupts output).
            self._convert_mxfp8_moe_to_block_fp8(layer)
            self.use_mxfp8 = False
            self.convert_mxfp8_to_block = False
            self.weight_block_size = [128, 128]
            if _is_fp8_fnuz:
                w13_weight, w13_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.w13_weight,
                    weight_scale=layer.w13_weight_scale_inv,
                    input_scale=None,
                )
                w2_weight, w2_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.w2_weight,
                    weight_scale=layer.w2_weight_scale_inv,
                    input_scale=None,
                )
                layer.w13_weight = Parameter(w13_weight, requires_grad=False)
                layer.w13_weight_scale_inv = Parameter(
                    w13_weight_scale, requires_grad=False
                )
                layer.w2_weight = Parameter(w2_weight, requires_grad=False)
                layer.w2_weight_scale_inv = Parameter(
                    w2_weight_scale, requires_grad=False
                )
                layer.w13_input_scale = None
                layer.w2_input_scale = None
            runner_is_aiter = (
                getattr(self, "runner", None) is not None
                and self.runner.runner_backend.is_aiter()
            )
            if _use_aiter and runner_is_aiter:
                layer.w13_weight.data = shuffle_weight(
                    layer.w13_weight.contiguous(), (16, 16)
                )
                layer.w2_weight.data = shuffle_weight(
                    layer.w2_weight.contiguous(), (16, 16)
                )
            return
        elif self.use_mxfp8:
            self._process_mxfp8_moe_weights(
                layer, quantize=not self.quant_config.is_checkpoint_fp8_serialized
            )
            return

        # If ROCm, normalize the weights and scales to e4m3fnuz
        if _is_fp8_fnuz:
            # activation_scheme: dynamic
            w13_weight, w13_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.w13_weight,
                weight_scale=layer.w13_weight_scale_inv,
                input_scale=None,
            )
            w2_weight, w2_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.w2_weight,
                weight_scale=layer.w2_weight_scale_inv,
                input_scale=None,
            )
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale_inv = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            layer.w13_input_scale = None
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale_inv = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )
            layer.w2_input_scale = None
            if _use_aiter:
                layer.w13_weight.data = shuffle_weight(
                    layer.w13_weight.contiguous(), (16, 16)
                )
                layer.w2_weight.data = shuffle_weight(
                    layer.w2_weight.contiguous(), (16, 16)
                )
        elif _use_aiter:
            # Pre-shuffle weights
            t = shuffle_weight(layer.w13_weight, (16, 16))
            layer.w13_weight.copy_(t)
            del t
            t = shuffle_weight(layer.w2_weight, (16, 16))
            layer.w2_weight.copy_(t)
            del t
        elif _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "Fp8MoEMethod on CPU requires that CPU has AMX support"
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])
        else:
            # For fp8 moe run with deepgemm, the expert weights and scales need be requantized to ue8m0
            from sglang.srt.layers import deep_gemm_wrapper
            from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE

            # Check if MoE will actually use DeepGEMM runner
            will_use_deepgemm = self.is_deepgemm_moe_runner_backend_enabled()

            if self.is_fp4_expert and self.dequant_fp4_to_fp8:
                for weight_param, scale_param in [
                    (layer.w13_weight, layer.w13_weight_scale_inv),
                    (layer.w2_weight, layer.w2_weight_scale_inv),
                ]:
                    num_experts = weight_param.shape[0]
                    new_weights = []
                    new_scales = []
                    for e in range(num_experts):
                        w, s = cast_e2m1fn_to_e4m3fn(
                            weight_param.data[e], scale_param.data[e]
                        )
                        new_weights.append(w)
                        new_scales.append(s)
                    weight_param.data = torch.stack(new_weights)
                    scale_param.data = torch.stack(new_scales).float()
                    scale_param.format_ue8m0 = False
                self.is_fp4_expert = False
                logger.warning_once("Dequantized FP4 expert weights to FP8.")

            if self.is_fp4_expert:
                if get_moe_runner_backend().is_marlin():
                    layer.w13_weight.data = layer.w13_weight.data.view(torch.int8)
                    layer.w2_weight.data = layer.w2_weight.data.view(torch.int8)
                    return

                fp4_weight_dtype = _require_fp4_dtype() if _use_aiter else torch.int8
                layer.w13_weight.data = layer.w13_weight.data.view(fp4_weight_dtype)
                layer.w2_weight.data = layer.w2_weight.data.view(fp4_weight_dtype)

                if get_moe_a2a_backend().is_megamoe():
                    from sglang.srt.layers.moe.mega_moe import (
                        build_mega_moe_experts_weights,
                    )

                    build_mega_moe_experts_weights(layer)
                    return

                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 and will_use_deepgemm:
                    from deep_gemm import transform_sf_into_required_layout

                    for scale_param, weight_param in [
                        (layer.w13_weight_scale_inv, layer.w13_weight),
                        (layer.w2_weight_scale_inv, layer.w2_weight),
                    ]:
                        num_experts, n, _ = scale_param.data.shape
                        k = weight_param.shape[2] * 2
                        scale_param.data = transform_sf_into_required_layout(
                            scale_param.data,
                            mn=n,
                            k=k,
                            recipe=(1, 32),
                            num_groups=num_experts,
                            disable_ue8m0_cast=False,
                        )
                    layer.w13_weight_scale_inv.format_ue8m0 = True
                    layer.w2_weight_scale_inv.format_ue8m0 = True

            if not self.is_fp4_expert:
                weight_block_size = self.quant_config.weight_block_size
                if requant_block_scale_ue8m0_for_deepgemm(
                    layer.w13_weight,
                    layer.w13_weight_scale_inv,
                    weight_block_size,
                    use_deepgemm_runner=will_use_deepgemm,
                ):
                    assert isinstance(
                        layer, DeepEPMoE
                    ), "DeepGemm MoE is only supported with DeepEPMoE"
                    requant_block_scale_ue8m0_for_deepgemm(
                        layer.w2_weight,
                        layer.w2_weight_scale_inv,
                        weight_block_size,
                        use_deepgemm_runner=True,
                    )

    def _convert_mxfp8_moe_to_block_fp8(self, layer: Module) -> None:
        from sglang.srt.layers.quantization.mxfp8_block_convert import (
            convert_mxfp8_weight_to_block_fp8,
        )

        def convert(w, s):
            E, N, K = w.shape
            qw = torch.empty_like(w)
            sn = (N + 127) // 128
            sk = (K + 127) // 128
            scale = torch.empty((E, sn, sk), dtype=torch.float32, device=w.device)
            for e in range(E):
                qe, se = convert_mxfp8_weight_to_block_fp8(w[e], s[e], block=128)
                qw[e] = qe
                scale[e] = se
            return qw, scale

        w13_q, w13_s = convert(layer.w13_weight.data, layer.w13_weight_scale_inv.data)
        w2_q, w2_s = convert(layer.w2_weight.data, layer.w2_weight_scale_inv.data)
        layer.w13_weight = Parameter(w13_q, requires_grad=False)
        layer.w2_weight = Parameter(w2_q, requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(w13_s, requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(w2_s, requires_grad=False)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def _process_mxfp8_moe_weights(self, layer: Module, quantize: bool = True) -> None:

        if not (
            (_is_cuda and is_sm100_supported()) or (_is_hip and _is_gfx95_supported)
        ):
            raise RuntimeError(
                "MXFP8 MoE quantization requires SM100 or ROCm gfx95 "
                "(gfx942 converts MXFP8 to block-fp8 at load instead)."
            )

        def _quantize_and_swizzle_with_cutlass_es_kernel(weight: torch.Tensor):
            from sgl_kernel import es_sm100_mxfp8_blockscaled_grouped_quant

            weight = weight.contiguous()
            num_experts, m, k = weight.shape
            assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"

            weight_flat = weight.view(-1, k).contiguous()
            problem_sizes = torch.empty(
                (num_experts, 3), dtype=torch.int32, device=weight.device
            )
            problem_sizes[:, 0] = m
            problem_sizes[:, 1] = 0
            problem_sizes[:, 2] = k
            expert_offsets = torch.arange(
                0, num_experts * m, m, dtype=torch.int32, device=weight.device
            )
            aligned_m = ((m + 127) // 128) * 128
            blockscale_offsets = torch.arange(
                0,
                num_experts * aligned_m,
                aligned_m,
                dtype=torch.int32,
                device=weight.device,
            )
            qweight = torch.empty_like(weight_flat, dtype=torch.float8_e4m3fn)
            scale = torch.empty(
                (num_experts * aligned_m, k // 32),
                dtype=torch.uint8,
                device=weight.device,
            )
            es_sm100_mxfp8_blockscaled_grouped_quant(
                weight_flat,
                problem_sizes,
                expert_offsets,
                blockscale_offsets,
                qweight,
                scale,
            )
            qweight = qweight.view_as(weight)
            scale = scale.view(num_experts, aligned_m, k // 32)
            if aligned_m != m:
                scale = scale[:, :m, :]
            return qweight, scale

        def _swizzle_mxfp8_sf(scale, num_warps):
            from triton_kernels.tensor import convert_layout, wrap_torch_tensor
            from triton_kernels.tensor_details import layout

            scale_layout, scale_layout_opts = (
                layout.make_default_matmul_mxfp4_w_scale_layout(
                    mx_axis=1, num_warps=num_warps
                )
            )
            scale = scale.transpose(-2, -1)
            scale = convert_layout(
                wrap_torch_tensor(scale), scale_layout, **scale_layout_opts
            )
            return scale

        def _swizzle_with_triton_kernel(
            weight_shape: tuple[int, int, int], scale: torch.Tensor
        ):
            num_experts, m, k = weight_shape
            aligned_m = ((m + 127) // 128) * 128
            scale = scale.view(num_experts, aligned_m, k // 32)
            num_warps = 8
            scale = _swizzle_mxfp8_sf(scale, num_warps)
            # convert_layout may pad for alignment; we can't view back to the
            # unpadded shape, so return the (possibly padded) swizzled tensor.
            return scale.data

        def _quantize_and_swizzle_with_triton_kernel(weight: torch.Tensor):

            weight = weight.contiguous()
            _, _, k = weight.shape
            assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"

            weight_flat = weight.view(-1, k).contiguous()
            qweight, scale = mxfp8_group_quantize(weight_flat)
            qweight = qweight.view_as(weight)
            scale = _swizzle_with_triton_kernel(weight.shape, scale)
            return qweight, scale

        def _quantize_with_flashinfer_trtllm(weight: torch.Tensor):
            weight = weight.contiguous()
            num_experts, m, k = weight.shape
            assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"
            from flashinfer import mxfp8_quantize

            weight_flat = weight.view(-1, k).contiguous()
            qweight, scale = mxfp8_quantize(weight_flat, False)
            scale_u8 = (
                scale.view(torch.uint8).contiguous().view(num_experts, m, k // 32)
            )
            return qweight.view_as(weight), scale_u8

        from sglang.srt.layers.quantization.mxfp8_block_convert import (
            _ue8m0_to_fp32,
        )

        def _quantize_for_deepgemm(weight: torch.Tensor):
            weight = weight.contiguous()
            num_experts, m, k = weight.shape
            assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"

            weight_flat = weight.view(-1, k).contiguous()
            qweight, scale_u8 = mxfp8_group_quantize(weight_flat)
            qweight = qweight.view_as(weight)
            scale_fp32 = _ue8m0_to_fp32(scale_u8).view(num_experts, m, k // 32)
            scale_packed = _pack_moe_scale_for_deepgemm(scale_fp32)
            return qweight, scale_packed

        def _pack_moe_scale_for_deepgemm(scale_fp32: torch.Tensor) -> torch.Tensor:
            """Blackwell: int32 MN-major TMA-packed. Hopper returns fp32 (FP4 API converts)."""
            from sglang.srt.layers.deep_gemm_wrapper.configurer import (
                DEEPGEMM_SCALE_UE8M0,
            )

            if DEEPGEMM_SCALE_UE8M0:
                import deep_gemm.utils.layout

                return (
                    deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(
                        scale_fp32
                    )
                )
            return scale_fp32

        def _convert_ue8m0_scales_for_deepgemm(
            scale_u8: torch.Tensor, shape: tuple
        ) -> torch.Tensor:
            num_experts, m, k_groups = shape[0], shape[1], scale_u8.shape[-1]
            scale_fp32 = _ue8m0_to_fp32(scale_u8.contiguous().view(-1)).view(
                num_experts, m, k_groups
            )
            return _pack_moe_scale_for_deepgemm(scale_fp32)

        if quantize:
            if _is_hip:
                w13_q, w13_s_u8 = mxfp8_group_quantize(
                    layer.w13_weight.data.contiguous().view(
                        -1, layer.w13_weight.data.shape[-1]
                    )
                )
                w2_q, w2_s_u8 = mxfp8_group_quantize(
                    layer.w2_weight.data.contiguous().view(
                        -1, layer.w2_weight.data.shape[-1]
                    )
                )
                w13_q = w13_q.view_as(layer.w13_weight.data)
                w2_q = w2_q.view_as(layer.w2_weight.data)
                w13_s = w13_s_u8.view(
                    layer.w13_weight.data.shape[0],
                    layer.w13_weight.data.shape[1],
                    layer.w13_weight.data.shape[2] // 32,
                )
                w2_s = w2_s_u8.view(
                    layer.w2_weight.data.shape[0],
                    layer.w2_weight.data.shape[1],
                    layer.w2_weight.data.shape[2] // 32,
                )
            elif get_moe_runner_backend().is_cutlass():
                w13_q, w13_s = _quantize_and_swizzle_with_cutlass_es_kernel(
                    layer.w13_weight.data
                )
                w2_q, w2_s = _quantize_and_swizzle_with_cutlass_es_kernel(
                    layer.w2_weight.data
                )
            elif get_moe_runner_backend().is_deep_gemm():
                w13_q, w13_s = _quantize_for_deepgemm(layer.w13_weight.data)
                w2_q, w2_s = _quantize_for_deepgemm(layer.w2_weight.data)
            elif (
                get_moe_runner_backend().is_flashinfer_trtllm()
                or get_moe_runner_backend().is_flashinfer_trtllm_routed()
            ):
                # Match FlashInfer TRT-LLM MoE test contracts:
                # 1) quantize in canonical (non-swizzled) scale layout, and
                # 2) do row/layout shuffling in align_mxfp8_moe_weights_for_flashinfer_trtllm.
                w13_q, w13_s = _quantize_with_flashinfer_trtllm(layer.w13_weight.data)
                w2_q, w2_s = _quantize_with_flashinfer_trtllm(layer.w2_weight.data)
            else:
                w13_q, w13_s = _quantize_and_swizzle_with_triton_kernel(
                    layer.w13_weight.data
                )
                w2_q, w2_s = _quantize_and_swizzle_with_triton_kernel(
                    layer.w2_weight.data
                )
        else:
            if _is_hip:
                w13_q = layer.w13_weight.data
                w2_q = layer.w2_weight.data
                w13_s = layer.w13_weight_scale_inv.data
                w2_s = layer.w2_weight_scale_inv.data
            elif (
                get_moe_runner_backend().is_flashinfer_trtllm()
                or get_moe_runner_backend().is_flashinfer_trtllm_routed()
            ):
                w13_q = layer.w13_weight.data
                w2_q = layer.w2_weight.data
                w13_s = layer.w13_weight_scale_inv.data
                w2_s = layer.w2_weight_scale_inv.data
            elif get_moe_runner_backend().is_deep_gemm():
                w13_q = layer.w13_weight.data
                w2_q = layer.w2_weight.data
                w13_s = _convert_ue8m0_scales_for_deepgemm(
                    layer.w13_weight_scale_inv.data, layer.w13_weight.data.shape
                )
                w2_s = _convert_ue8m0_scales_for_deepgemm(
                    layer.w2_weight_scale_inv.data, layer.w2_weight.data.shape
                )
            else:
                w13_q = layer.w13_weight.data
                w2_q = layer.w2_weight.data
                w13_s = _swizzle_with_triton_kernel(
                    layer.w13_weight.data.shape, layer.w13_weight_scale_inv.data
                )
                w2_s = _swizzle_with_triton_kernel(
                    layer.w2_weight.data.shape, layer.w2_weight_scale_inv.data
                )

        # Keep parameter objects to preserve weight_loader attrs for hot reload.
        # Prefer in-place copy; rebind only when shape/dtype changes (online quantize).
        def _copy_or_rebind(param: Parameter, new_value: torch.Tensor) -> None:
            if (
                param.data.shape == new_value.shape
                and param.data.dtype == new_value.dtype
            ):
                param.data.copy_(new_value)
            else:
                param.data = new_value

        _copy_or_rebind(layer.w13_weight, w13_q)
        _copy_or_rebind(layer.w2_weight, w2_q)
        _copy_or_rebind(layer.w13_weight_scale_inv, w13_s)
        _copy_or_rebind(layer.w2_weight_scale_inv, w2_s)
        layer.w13_weight.requires_grad_(False)
        layer.w2_weight.requires_grad_(False)
        layer.w13_weight_scale_inv.requires_grad_(False)
        layer.w2_weight_scale_inv.requires_grad_(False)
        layer.w13_weight_scale_inv.format_ue8m0 = True
        layer.w2_weight_scale_inv.format_ue8m0 = True
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        if (
            get_moe_runner_backend().is_flashinfer_trtllm()
            or get_moe_runner_backend().is_flashinfer_trtllm_routed()
        ):
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                align_mxfp8_moe_weights_for_flashinfer_trtllm,
            )

            align_mxfp8_moe_weights_for_flashinfer_trtllm(layer)

    def process_weights_after_loading(self, layer: Module) -> None:
        if _is_hip and _use_hip_int4:
            self.process_weights_hip_int4(layer)

        elif self.block_quant:
            # Block quant doesn't need to process weights after loading
            self.process_weights_after_loading_block_quant(layer)

        # If checkpoint is fp16 or bfloat16, quantize in place.
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            # If ROCm, fp8_dtype will be float8_e4m3fnuz (MI300x HW)
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.num_local_experts,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )
            for expert in range(layer.num_local_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

            if _is_hip:
                self.process_weights_hip_scale_padding(layer)

        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            # Fp8 moe kernels require a single activation scale.
            # We take the max of all the scales in case they differ.
            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                    layer.w2_input_scale
                ):
                    print_warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer. "
                    )
                layer.w13_input_scale = torch.nn.Parameter(
                    layer.w13_input_scale.max(), requires_grad=False
                )
                layer.w2_input_scale = torch.nn.Parameter(
                    layer.w2_input_scale.max(), requires_grad=False
                )

            # If ROCm, normalize the weights and scales to e4m3fnuz
            if _is_fp8_fnuz:
                # Normalize the weights and scales
                w13_weight, w13_weight_scale, w13_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                    )
                )
                w2_weight, w2_weight_scale, w2_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
                    )
                )
                # Reset the parameter
                layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False
                )
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(
                        w13_input_scale, requires_grad=False
                    )
                layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
                layer.w2_weight_scale = torch.nn.Parameter(
                    w2_weight_scale, requires_grad=False
                )
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(
                        w2_input_scale, requires_grad=False
                    )
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.num_local_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    (
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        _,
                    ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

            if _is_hip:
                self.process_weights_hip_scale_padding(layer)

            # Align FP8 weights to FlashInfer per-tensor kernel layout if enabled
            if (
                get_moe_runner_backend().is_flashinfer_trtllm()
                or get_moe_runner_backend().is_flashinfer_trtllm_routed()
            ):
                from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                    align_fp8_moe_weights_for_flashinfer_trtllm,
                )

                align_fp8_moe_weights_for_flashinfer_trtllm(layer)

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"weight_dtype": layer.w13_weight.dtype})

        if self.use_marlin:
            self._prepare_marlin_moe(layer)

    def _prepare_marlin_moe(self, layer: Module) -> None:
        """Convert the fp8 expert weights of this rank to Marlin W8A16 format.

        Rank-local fallback for GPUs without native FP8 compute (sm < 89).
        Runs after the standard fp8 post-load processing, so the weights are
        already requantized to a single per-expert scale (non-block) or carry
        their block scales. Shard geometry is untouched globally; if this
        rank's expert intermediate shard is not Marlin-tile aligned (n % 64,
        possible under uneven TP unit splits), the intermediate dimension is
        zero-padded locally — padded gate/up channels produce silu(0)*0 = 0
        and the padded w2 input columns are zero weights, so the result is
        numerically identical.
        """
        from sglang.srt.layers.quantization.marlin_utils import (
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.marlin_utils_fp8 import (
            prepare_moe_fp8_layer_for_marlin,
        )

        n = layer.intermediate_size_per_partition
        pad_to = 64
        n_pad = (n + pad_to - 1) // pad_to * pad_to
        if n_pad != n:
            if self.block_quant:
                raise ValueError(
                    "FP8 Marlin MoE fallback: block-quantized experts with a "
                    f"non-tile-aligned intermediate shard ({n} % {pad_to} != 0) "
                    "are not supported (padding would break block-scale "
                    "alignment)."
                )
            e = layer.w13_weight.shape[0]
            k = layer.w13_weight.shape[2]
            w13 = layer.w13_weight.data.view(e, 2, n, k)
            w13_padded = torch.zeros(
                (e, 2, n_pad, k), dtype=w13.dtype, device=w13.device
            )
            w13_padded[:, :, :n, :] = w13
            layer.w13_weight = torch.nn.Parameter(
                w13_padded.view(e, 2 * n_pad, k), requires_grad=False
            )
            w2 = layer.w2_weight.data
            w2_padded = torch.zeros(
                (e, w2.shape[1], n_pad), dtype=w2.dtype, device=w2.device
            )
            w2_padded[:, :, :n] = w2
            layer.w2_weight = torch.nn.Parameter(w2_padded, requires_grad=False)
            layer.intermediate_size_per_partition = n_pad
            logger.info(
                "FP8 Marlin MoE fallback: zero-padded expert intermediate "
                "shard %d -> %d for Marlin tile alignment.",
                n,
                n_pad,
            )

        group_size = (
            -1 if self.weight_block_size is None else self.weight_block_size[1]
        )
        if not check_moe_marlin_supports_layer(layer, group_size):
            raise ValueError(
                "FP8 Marlin MoE fallback: layer does not satisfy Marlin "
                f"constraints (hidden_size={layer.hidden_size}, "
                f"intermediate_size_per_partition="
                f"{layer.intermediate_size_per_partition}, "
                f"group_size={group_size})."
            )

        # w13/w2 are stored (num_experts, size_n, size_k) -> size_k_first=False
        prepare_moe_fp8_layer_for_marlin(layer, size_k_first=False)
        torch.cuda.empty_cache()

    def process_weights_hip_int4(self, layer: Module):
        # TODO: _use_aiter: add after triton kernel added
        # INT4-FP8 (INT4 MoE Weight, FP8 Compute)
        # Weight Permutation
        layer.w13_weight = torch.nn.Parameter(
            shuffle_weight(layer.w13_weight.data, (16, 16)),
            requires_grad=False,
        )
        torch.cuda.empty_cache()
        layer.w2_weight = torch.nn.Parameter(
            shuffle_weight(layer.w2_weight.data, (16, 16)),
            requires_grad=False,
        )
        torch.cuda.empty_cache()

        # INT4-FP8 : offset INT4 w13_weight_scale1 to single w13_weight_scale
        # Fp8 moe kernel needs single fp8 w13_weight_scale for w13 per expert.
        # We won't do requant each expert's fp8 weight (not direct available),
        # instead we adjust half of INT4 w13_weight_scale1 numbers
        assert layer.w13_weight_scale is not None
        shard_size = layer.intermediate_size_per_partition
        max_w13_scales = layer.w13_weight_scale.max(dim=1).values
        for expert_id in range(layer.num_local_experts):
            start = 0
            max_w13_scale_fp8 = max_w13_scales[expert_id]
            for shard_id in range(2):
                if layer.w13_weight_scale[expert_id][shard_id] != max_w13_scale_fp8:
                    int4_rescale = (
                        layer.w13_weight_scale[expert_id][shard_id] / max_w13_scale_fp8
                    )
                    layer.w13_weight_scale1[expert_id][
                        start : start + shard_size
                    ] *= int4_rescale
                start += shard_size

        layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales, requires_grad=False)

        # special hack to asm_moe, which takes (weight_scale1 * weight_scale) as post GEMM scaling
        # optimal design - shall apply per-column weight_scale1 before GEMM, and weight_scale post
        for expert_id in range(layer.num_local_experts):
            layer.w13_weight_scale1[expert_id] *= max_w13_scales[expert_id]
            layer.w2_weight_scale1[expert_id] *= layer.w2_weight_scale[expert_id]

    def process_weights_hip_scale_padding(self, layer: Module):
        padding_size = get_moe_padding_size(_use_aiter)
        if _use_aiter:
            layer.w13_weight = torch.nn.Parameter(
                shuffle_weight(layer.w13_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                shuffle_weight(layer.w2_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()

            # ROCm (_use_aiter): using column-wise scaling
            layer.w13_weight_scale1 *= layer.w13_weight_scale.unsqueeze(-1)
            layer.w2_weight_scale1 *= layer.w2_weight_scale.unsqueeze(-1)
        elif get_bool_env_var("SGLANG_MOE_PADDING"):
            # If ROCm, apply weight padding (min. Mem channel contention) only if set
            layer.w13_weight = torch.nn.Parameter(
                F.pad(layer.w13_weight.data, (0, padding_size), "constant", 0),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                F.pad(layer.w2_weight.data, (0, padding_size), "constant", 0),
                requires_grad=False,
            )
            torch.cuda.empty_cache()

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        moe_runner_backend = get_moe_runner_backend()

        if self.use_marlin:
            # Rank-local Marlin W8A16 fallback (no native FP8 compute).
            # Import registers the "none"->"marlin" fused func before the
            # FusedOpPool lookup inside MoeRunner.
            from sglang.srt.layers.moe.moe_runner import marlin  # noqa: F401

            self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)
            return

        if moe_runner_backend.is_auto():
            if self.is_deepgemm_moe_runner_backend_enabled():
                moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
            elif (
                _is_hip
                and (_use_aiter or _use_hip_int4)
                and get_moe_a2a_backend().supports_aiter()
            ):
                moe_runner_backend = MoeRunnerBackend.AITER
            else:
                moe_runner_backend = MoeRunnerBackend.TRITON

        if (
            moe_runner_backend.is_deep_gemm()
            or moe_runner_backend.is_triton()
            or moe_runner_backend.is_aiter()
            or moe_runner_backend.is_flashinfer_trtllm()
            or moe_runner_backend.is_flashinfer_trtllm_routed()
        ):
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
        else:
            # TODO(cwan): refactor other backends
            pass

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        use_rocm_mxfp8 = self.use_mxfp8 and _is_hip and _is_gfx95_supported
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            b13=getattr(layer, "w13_weight_bias", None),
            b2=getattr(layer, "w2_weight_bias", None),
            use_mxfp8=use_rocm_mxfp8,
            use_fp8_w8a8=not use_rocm_mxfp8,
            w13_scale=(
                layer.w13_weight_scale_inv
                if self.block_quant
                else layer.w13_weight_scale
            ),
            w2_scale=(
                layer.w2_weight_scale_inv if self.block_quant else layer.w2_weight_scale
            ),
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=self.weight_block_size,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        moe_runner_config = self.moe_runner_config

        if self.use_marlin:
            # Rank-local Marlin W8A16 fallback for GPUs without native FP8
            # compute (see __init__). Weights were repacked in
            # _prepare_marlin_moe.
            from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo

            quant_info = MarlinMoeQuantInfo(
                w13_qweight=layer.w13_weight,
                w2_qweight=layer.w2_weight,
                w13_scales=layer.w13_weight_scale,
                w2_scales=layer.w2_weight_scale,
                w13_g_idx_sort_indices=None,
                w2_g_idx_sort_indices=None,
                weight_bits=8,
                weight_is_fp8=True,
                w13_bias=getattr(layer, "w13_bias", None),
                w2_bias=getattr(layer, "w2_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = dispatch_output.topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )

            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace See [Note] inplace should be False in fused_experts.
                CPUQuantMethod.FP8_W8A16,
                layer.w13_weight_scale_inv,  # w1_scale
                layer.w2_weight_scale_inv,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                self.quant_config.weight_block_size,  # block_size
                None,  # w1 bias
                None,  # w3 bias
                None,  # alpha
                None,  # limit
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)

        if (
            _is_hip
            and getattr(self, "runner", None) is not None
            and self.runner.runner_backend.is_aiter()
        ):
            quant_info = self.maybe_get_hip_aiter_quant_info(
                layer,
                moe_runner_config.no_combine,
            )
            if quant_info is not None:
                return self.runner.run(dispatch_output, quant_info)

        if use_intel_xpu_backend():
            # sgl-kernel-xpu path
            from sgl_kernel import fused_experts

            topk_weights, topk_ids, _ = dispatch_output.topk_output
            assert layer.w13_weight.dtype == layer.w2_weight.dtype
            use_fp8_w8a8 = layer.w13_weight.dtype == torch.float8_e4m3fn
            use_mxfp4_w4a16 = layer.w13_weight.dtype == torch.int8
            assert self.is_fp4_expert == use_mxfp4_w4a16
            output = fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                b1=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                use_mxfp4_w4a16=use_mxfp4_w4a16,
                use_fp8_w8a8=use_fp8_w8a8,
                w1_scale=(
                    layer.w13_weight_scale_inv
                    if self.block_quant
                    else layer.w13_weight_scale
                ),
                w2_scale=(
                    layer.w2_weight_scale_inv
                    if self.block_quant
                    else layer.w2_weight_scale
                ),
                activation=moe_runner_config.activation,
                routed_scaling_factor=moe_runner_config.routed_scaling_factor,
                gemm1_alpha=moe_runner_config.gemm1_alpha,
                gemm1_limit=moe_runner_config.gemm1_clamp_limit,
                swiglu_limit=moe_runner_config.swiglu_limit,
            )
            return StandardCombineInput(hidden_states=output)

        if get_moe_runner_backend().is_cutlass():
            from sglang.srt.layers.moe.cutlass_moe import cutlass_fused_experts_fp8

            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                symm_output = torch.empty_like(x)

            topk_weights, topk_ids, _ = dispatch_output.topk_output
            use_mxfp8 = getattr(self.quant_config, "use_mxfp8", False)
            output = cutlass_fused_experts_fp8(
                x,
                layer.w13_weight.transpose(1, 2),
                layer.w2_weight.transpose(1, 2),
                layer.w13_weight_scale_inv.transpose(1, 2),
                layer.w2_weight_scale_inv.transpose(1, 2),
                topk_weights,
                topk_ids,
                self.ab_strides1,
                self.c_strides1,
                self.ab_strides2,
                self.c_strides2,
                self.workspace,
                self.a_ptr,
                self.b_ptr,
                self.out_ptr,
                self.a_scales_ptr,
                self.b_scales_ptr,
                self.expert_offsets,
                self.problem_sizes1,
                self.problem_sizes2,
                use_fp8_blockscale=True,
                use_mxfp8=use_mxfp8,
                output=symm_output,
                enable_es=(use_mxfp8, use_mxfp8),
            )
            return StandardCombineInput(hidden_states=output)

        if self.runner.runner_backend.is_deep_gemm():

            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight

            if self.block_quant:
                block_shape = self.quant_config.weight_block_size
                w13_scale = layer.w13_weight_scale_inv
                w2_scale = layer.w2_weight_scale_inv
            else:
                # Convert per-tensor quant to per-block quant by repeating scales for forward_deepgemm
                scale_block_size = 128
                block_shape = [scale_block_size, scale_block_size]
                w13_scale_n = (w13_weight.shape[1] - 1) // scale_block_size + 1
                w13_scale_k = (w13_weight.shape[2] - 1) // scale_block_size + 1
                w13_scale = (
                    layer.w13_weight_scale.unsqueeze(1)
                    .repeat_interleave(w13_scale_n, dim=1)
                    .unsqueeze(2)
                    .repeat_interleave(w13_scale_k, dim=2)
                )
                w2_scale_n = (w2_weight.shape[1] - 1) // scale_block_size + 1
                w2_scale_k = (w2_weight.shape[2] - 1) // scale_block_size + 1
                w2_scale = (
                    layer.w2_weight_scale.unsqueeze(1)
                    .repeat_interleave(w2_scale_n, dim=1)
                    .unsqueeze(2)
                    .repeat_interleave(w2_scale_k, dim=2)
                )
            quant_info = DeepGemmMoeQuantInfo(
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                use_fp8=True,
                w13_scale=w13_scale,
                w2_scale=w2_scale,
                block_shape=block_shape,
                is_fp4_experts=self.is_fp4_expert,
                use_mxfp8=self.use_mxfp8,
            )
        elif (
            self.runner.runner_backend.is_flashinfer_trtllm()
            or self.runner.runner_backend.is_flashinfer_trtllm_routed()
        ):
            # FlashInfer TRT-LLM backend only supports fused execution and consumes
            # router logits directly (no separate apply_with_router_logits needed).
            # FlashInfer TRT-LLM routed backend consumes SGLang-computed
            # top-k ids/weights (packed into int32) instead of router logits.
            global_num_experts = int(getattr(layer, "num_experts"))
            num_local_experts = int(getattr(layer, "num_local_experts"))
            moe_ep_rank = int(getattr(layer, "moe_ep_rank"))

            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                get_activation_type,
            )

            activation_type = get_activation_type(
                self.moe_runner_config.activation,
                is_gated=self.moe_runner_config.is_gated,
            )

            quant_info = FlashInferTrtllmFp8MoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                global_num_experts=global_num_experts,
                local_expert_offset=moe_ep_rank * num_local_experts,
                local_num_experts=num_local_experts,
                intermediate_size=layer.w2_weight.shape[2],
                routing_method_type=int(
                    getattr(layer, "routing_method_type", None)
                    or RoutingMethodType.DeepSeekV3
                ),
                block_quant=self.block_quant,
                use_mxfp8=getattr(self.quant_config, "use_mxfp8", False),
                weight_block_k=(
                    None
                    if self.quant_config.weight_block_size is None
                    else self.quant_config.weight_block_size[1]
                ),
                w13_weight_scale_inv=(
                    layer.w13_weight_scale_inv if self.block_quant else None
                ),
                w2_weight_scale_inv=(
                    layer.w2_weight_scale_inv if self.block_quant else None
                ),
                w13_input_scale=layer.w13_input_scale if not self.block_quant else None,
                output1_scales_scalar=(
                    getattr(layer, "output1_scales_scalar", None)
                    if not self.block_quant
                    else None
                ),
                output1_scales_gate_scalar=(
                    getattr(layer, "output1_scales_gate_scalar", None)
                    if not self.block_quant
                    else None
                ),
                output2_scales_scalar=(
                    getattr(layer, "output2_scales_scalar", None)
                    if not self.block_quant
                    else None
                ),
                activation_type=activation_type,
            )
        elif self.runner.runner_backend.is_triton():
            quant_info = self.get_triton_quant_info(layer)
        else:
            raise NotImplementedError(
                "Unsupported runner backend: %s" % self.runner.runner_backend
            )

        return self.runner.run(dispatch_output, quant_info)

    def _ensure_cutlass_buffers_initialized(self, layer: Module) -> None:
        if getattr(self, "_cutlass_buffers_ready", False):
            return

        device = layer.w13_weight.device
        num_experts = layer.w13_weight.shape[0]
        hidden_size = layer.w2_weight.shape[1]
        intermediate_size_per_partition = layer.intermediate_size_per_partition

        self.ab_strides1 = torch.full(
            (num_experts,), hidden_size, device=device, dtype=torch.int64
        )
        self.c_strides1 = torch.full(
            (num_experts,),
            2 * intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.ab_strides2 = torch.full(
            (num_experts,),
            intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides2 = torch.full(
            (num_experts,), hidden_size, device=device, dtype=torch.int64
        )
        self.workspace = torch.empty(90000, device=device, dtype=torch.uint8)
        self.a_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.b_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.out_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.a_scales_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.b_scales_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.expert_offsets = torch.empty(
            num_experts + 1, device=device, dtype=torch.int32
        )
        self.problem_sizes1 = torch.empty(
            num_experts, 3, device=device, dtype=torch.int32
        )
        self.problem_sizes2 = torch.empty(
            num_experts, 3, device=device, dtype=torch.int32
        )

        self._cutlass_buffers_ready = True

    def maybe_get_hip_aiter_quant_info(
        self,
        layer: torch.nn.Module,
        no_combine: bool = False,
    ) -> Optional[AiterMoeQuantInfo]:
        if not (_use_aiter or _use_hip_int4):
            return None
        assert not no_combine, f"{no_combine=} is not supported."

        from sglang.srt.layers.moe.moe_runner.aiter import (
            AiterMoeQuantInfo,
            AiterQuantType,
        )

        w13_weight = layer.w13_weight
        w2_weight = layer.w2_weight

        if self.block_quant:
            quant_type = (
                AiterQuantType.PER_1X32
                if self.is_fp4_expert
                else AiterQuantType.PER_128X128
            )

            if self.is_fp4_expert:
                fp4_weight_dtype = _require_fp4_dtype()
                w13_weight = w13_weight.view(fp4_weight_dtype)
                w2_weight = w2_weight.view(fp4_weight_dtype)
                if getattr(layer.w13_weight, "is_shuffled", False):
                    w13_weight.is_shuffled = True
                    w2_weight.is_shuffled = True
            w13_scale = layer.w13_weight_scale_inv
            w2_scale = layer.w2_weight_scale_inv
        else:
            quant_type = AiterQuantType.PER_TOKEN
            w13_scale = layer.w13_weight_scale1
            w2_scale = layer.w2_weight_scale1
        return AiterMoeQuantInfo(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            quant_type=quant_type,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            expert_mask=layer.dispatcher.expert_mask_gpu if _use_aiter else None,
            swiglu_limit=self.moe_runner_config.swiglu_limit or 0.0,
            hidden_pad=getattr(layer, "hidden_pad", 0),
            intermediate_pad=getattr(layer, "intermediate_pad", 0),
        )


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
