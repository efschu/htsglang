# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import math
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton.layer import get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    get_hip_version,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_musa,
    is_npu,
    is_nvidia_cublas_version_ge_12_9,
    is_xpu,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_musa = is_musa()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()
_device_sm = get_device_sm()
_is_gfx95_supported = is_gfx95_supported()
_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
_use_aiter_bpreshuffle_gfx95 = _use_aiter_gfx95 and get_hip_version() >= (7, 2, 0)


_is_cublas_ge_129 = is_nvidia_cublas_version_ge_12_9()

logger = logging.getLogger(__name__)

NVFP4_CKPT_FP8_ATTN_QUANT_MODULES = ["q_b_proj"]

FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = [
    "fa3",
    "dsa",
    "nsa",  # Deprecated alias for "dsa"
    "flashinfer",
    "cutlass_mla",
    "trtllm_mla",
    "cutedsl_mla",
    "tokenspeed_mla",
    "ascend",
    "intel_xpu",
]


def dense_weight_dtype(layer: torch.nn.Module) -> Optional[torch.dtype]:
    """dtype of a layer's dense ``weight``, or ``None`` when it has none.

    Packed quantization formats register ``qweight`` and never materialize a
    ``weight`` attribute: GGUF (``GGUFLinearMethod``) and the AWQ/GPTQ family
    all do. Every dtype test in the DeepSeek family selects a fast path that
    only exists for a dense, int8 or fp8 weight tensor, so "there is no dense
    weight" has to read as "not this fast path". Reading the attribute
    directly instead raises ``AttributeError: 'MergedColumnParallelLinear'
    object has no attribute 'weight'`` at construction time, which is what
    kept the whole DeepSeek family from being built under GGUF.

    Behavior for the dense, int8 and fp8 paths is unchanged: those layers do
    carry ``weight``, so the dtype returned is the one the direct attribute
    access returned.
    """
    weight = getattr(layer, "weight", None)
    return getattr(weight, "dtype", None)


def is_packed_layer(layer: torch.nn.Module) -> bool:
    """True when a layer stores packed weights instead of a dense ``weight``.

    Classify by what was actually built, not by the quantization method's
    name. A name list has to enumerate every packed format and silently
    misclassifies the ones it forgot: ``{"awq", "awq_marlin", "moe_wna16"}``
    answers "dense" for ``gptq``, ``auto-round``, ``gguf`` and every packed
    ``compressed-tensors`` scheme, all of which register ``qweight`` (or
    ``weight_packed``) and no ``weight``. The structural test cannot go stale
    when a new packed format is added.

    This is the same predicate ``dense_weight_dtype`` answers with, phrased as
    a boolean for the sites that only need packed-vs-dense and not the dtype.
    """
    return getattr(layer, "weight", None) is None


class UnfusableAProjParameter(ValueError):
    """Raised when ``q_a_proj`` and ``kv_a_proj_with_mqa`` cannot be fused.

    The fused ``fused_qkv_a_proj_with_mqa`` layer holds exactly one copy of
    every per-input-channel parameter, so the two source projections have to
    agree on it. GPTQ with ``desc_act=True`` is the case that can disagree:
    each projection is quantized independently, so each gets its own
    activation-order permutation in ``g_idx``.
    """


def fuse_q_kv_a_proj(
    param: torch.nn.Parameter,
    param_name: str,
    q_a_proj_weight: torch.Tensor,
    kv_a_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """Build one ``fused_qkv_a_proj_with_mqa`` tensor from the two sources.

    The two MLA down-projections read the same hidden state and are fused
    along the OUTPUT axis. Which axis of the checkpoint tensor that is depends
    on the quantization format, and the destination parameter already records
    it: every ``create_weights`` sets ``output_dim`` on the parameters it
    registers. Reading it there is exact for any format, present or future:

    ==========================  ==========  ============================
    format                      output_dim  registered parameters
    ==========================  ==========  ============================
    unquantized / fp8 / int8    0           ``weight``, scales
    gguf                        0           ``qweight``
    compressed-tensors wNa16    0           ``weight_packed``, scales
    awq / awq_marlin            1           ``qweight``, ``qzeros``, scales
    gptq / gptq_marlin          1           ``qweight``, ``qzeros``, scales
    ==========================  ==========  ============================

    The quantization-name list this replaces granted axis 1 to ``awq``,
    ``awq_marlin`` and ``moe_wna16`` only, so the GPTQ family -- whose
    ``qweight`` is ``[in // pack_factor, out]``, the same output-on-axis-1
    layout as AWQ -- was concatenated along the input axis instead.

    Two parameter kinds are not concatenations:

    * A 0-d checkpoint tensor is a marker rather than a shard: the GGUF
      ``qweight_type`` byte and the AutoFP8 per-tensor scale. The fused layer
      takes one value, as it did before this helper existed.
    * A parameter with an ``input_dim`` and no ``output_dim`` holds one entry
      per INPUT channel: ``g_idx`` (GPTQ) and ``weight_g_idx``
      (compressed-tensors wNa16). Both sources describe the same input
      channels, so the fused layer needs one copy of the vector, not a
      doubled one. When the two copies disagree the fusion has no answer at
      all and is refused by name.
    """
    if q_a_proj_weight.shape == torch.Size([]) and kv_a_proj_weight.shape == torch.Size(
        []
    ):
        return q_a_proj_weight

    output_dim = getattr(param, "output_dim", None)
    if output_dim is None and getattr(param, "input_dim", None) is not None:
        if q_a_proj_weight.shape != kv_a_proj_weight.shape or not torch.equal(
            q_a_proj_weight, kv_a_proj_weight
        ):
            raise UnfusableAProjParameter(
                f"Cannot fuse q_a_proj and kv_a_proj_with_mqa into {param_name}: "
                f"the per-input-channel parameter differs between the two "
                f"projections (shapes {tuple(q_a_proj_weight.shape)} and "
                f"{tuple(kv_a_proj_weight.shape)}). The fused layer holds a "
                f"single copy of it, so the two projections must agree. This is "
                f"what a GPTQ checkpoint quantized with desc_act=True produces: "
                f"each projection carries its own activation-order permutation "
                f"in g_idx. Re-quantize with desc_act=False, or serve a format "
                f"without a per-input-channel index."
            )
        return q_a_proj_weight

    return torch.cat(
        [q_a_proj_weight, kv_a_proj_weight], dim=0 if output_dim is None else output_dim
    )


def layer_quant_method_name(layer: torch.nn.Module) -> Optional[str]:
    """Quant method a layer was built with, or None when it is unquantized.

    Read off the layer rather than off the model config, because attention
    submodules can be built with an overridden quant config (see
    ``DeepseekV2AttentionMLA._get_q_b_proj_quant_config`` and the DSV4
    ``wo_a``).
    """
    quant_config = getattr(getattr(layer, "quant_method", None), "quant_config", None)
    return None if quant_config is None else quant_config.get_name()


def is_gguf_quant_config(quant_config: Optional[QuantizationConfig]) -> bool:
    """True for GGUF checkpoints, whose linears are all packed."""
    return quant_config is not None and quant_config.get_name() == "gguf"


def awq_dequantize_func():
    """
    Get the AWQ dequantize function for the current device

    Return:
        - The AWQ dequantize function for the current device.
        - None if the current device is not supported.
    """
    if _is_cuda:
        from sgl_kernel import awq_dequantize

        return awq_dequantize
    elif _is_hip:
        from sglang.kernel_api_logging import debug_kernel_api
        from sglang.srt.layers.quantization.awq.awq_triton import (
            awq_dequantize_triton as awq_dequantize,
        )

        return debug_kernel_api(awq_dequantize, op_name="DeepseekCommon.awq_dequantize")
    elif _is_npu:
        from sglang.kernel_api_logging import debug_kernel_api
        from sglang.srt.layers.quantization.awq.awq_triton import (
            awq_dequantize_decomposition as awq_dequantize,
        )

        return debug_kernel_api(awq_dequantize, op_name="DeepseekCommon.awq_dequantize")
    else:
        return None


def enable_nextn_moe_bf16_cast_to_fp8(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    return (
        envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get()
        and quant_config is not None
        and quant_config.get_name() == "modelopt_fp4"
        and get_moe_runner_backend().is_deep_gemm()
    )


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings)
    )
    return scaling[..., None, None]
