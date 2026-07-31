# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from collections.abc import Callable
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
from sglang.srt.layers.quantization.fp8_utils import is_blackwell_supported
from sglang.srt.layers.quantization.marlin_utils import GPTQ_MARLIN_MIN_THREAD_N
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_nvfp4_layer_for_marlin,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    enable_flashinfer_fp4_gemm,
    fp4_gemm,
    fp4_quantize,
)
from sglang.srt.layers.quantization.utils import swizzle_blockscale

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A4Fp4"]


class CompressedTensorsW4A4Fp4(CompressedTensorsLinearScheme):
    """compressed-tensors ``nvfp4-pack-quantized`` linear scheme.

    Two execution lanes, selected per rank by the FP4 GEMM backend that
    ``initialize_fp4_gemm_config`` resolved for that rank's device:

    * **native FP4 (W4A4)** on Blackwell -- activations are quantised to E2M1
      and the block-scaled CUTLASS / FlashInfer GEMM runs the multiply in FP4.
    * **Marlin (W4A16)** on sm_80-sm_89 -- activations stay bf16/fp16 and the
      E2M1 weight is dequantised inside ``gptq_marlin_gemm``
      (``b_q_type = float4_e2m1f``). This is the same kernel family these
      cards already use for FP8 (``fp8_marlin``), at 0.93-0.96x of their own
      dense bf16 rate, and it reads 4.5 bits per weight instead of 8 -- so it
      is the best quantised lane on that hardware, not a compatibility
      minimum (#291-S3, ``docs/dev/ANALYSE_321_nvfp4_asymmetry.md`` §4.1/§9.1).

    ``ModelOptFp4LinearMethod`` has carried the same pair since upstream
    relanded NVFP4 Marlin as SM80+; the compressed-tensors half was never
    restored, which is what made every all-Linear-NVFP4 checkpoint (the only
    NVFP4 variant that is simultaneously VRAM-, context- and decode-positive)
    unbootable on a rig with any pre-Blackwell rank.

    Ported from vLLM, which already solved exactly this
    (``vllm/model_executor/layers/quantization/compressed_tensors/schemes/
    compressed_tensors_w4a4_nvfp4.py`` + ``vllm/model_executor/kernels/linear/
    nvfp4/marlin.py``): the scheme there carries no Blackwell floor and lets a
    kernel registry pick Marlin when no native FP4 kernel reports support. Two
    deliberate differences, both fork-idiom rather than semantics:

    * Lane selection reuses the fork's existing per-rank
      ``get_fp4_gemm_runner_backend()`` instead of importing vLLM's kernel
      registry. That resolver is already called once per scheduler process, so
      a mixed-architecture rig resolves a different lane per rank for free.
    * ``get_min_capability`` is 80, not vLLM's 75: the fork's own
      ``query_marlin_supported_quant_types`` returns nothing below 80, so 75
      would admit a scheme with no kernel behind it.

    The scale semantics are taken verbatim from vLLM (see
    ``_process_weights_for_marlin``).
    """

    def __init__(self):
        self.group_size = 16

    @classmethod
    def get_min_capability(cls) -> int:
        # Marlin's own floor (marlin_utils.query_marlin_supported_quant_types
        # returns [] below 80), not Blackwell's. Above it the scheme picks the
        # lane; below it there is no NVFP4 kernel of any kind and
        # _get_scheme_from_parts still raises NotImplementedError.
        return 80

    @staticmethod
    def _use_marlin() -> bool:
        return get_fp4_gemm_runner_backend().is_marlin()

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        # prepare_nvfp4_layer_for_marlin / apply_fp4_marlin_linear read the
        # activation dtype off the layer and reject anything but fp16/bf16.
        layer.params_dtype = params_dtype

        if self._use_marlin() and output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N:
            # Two different causes end up here and they need different fixes,
            # so name which one this is. Measured 2026-07-31: the all-Linear
            # NVFP4 checkpoint ocicek/Qwen3.6-27B-NVFP4 quantises the GDN b/a
            # gate, whose FULL width is 96 rows -- it fails this check at TP=1
            # on a single card, where there is no shard plan to blame.
            full_size = kwargs.get("output_size")
            if isinstance(full_size, int) and full_size % GPTQ_MARLIN_MIN_THREAD_N:
                cause = (
                    f"The UNSHARDED width is {full_size}, which is itself not a "
                    f"multiple of {GPTQ_MARLIN_MIN_THREAD_N}, so no shard plan "
                    "can satisfy this check -- the checkpoint quantises a "
                    "projection narrower than the Marlin tile. Serving it on a "
                    "pre-Blackwell rank requires the projection to be excluded "
                    "from quantisation, not a different --rank-tp-ratio."
                )
            else:
                cause = (
                    f"The unsharded width is {full_size}, so the tile is "
                    "reachable: under an uneven --rank-tp-ratio this means the "
                    "shard plan was not coarsened to the Marlin tile."
                )
            raise ValueError(
                "NVFP4 Marlin requires output_size_per_partition to be a "
                f"multiple of {GPTQ_MARLIN_MIN_THREAD_N}, got "
                f"{output_size_per_partition}. {cause}"
            )

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

        input_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer) -> None:
        # Ported from vLLM: a fused linear (qkv_proj, gate_up_proj) holds one
        # global scale per logical shard and both lanes collapse them with
        # max(), which silently costs accuracy when the shards disagree. Say so
        # once instead of collapsing in silence.
        for name in ("weight_global_scale", "input_global_scale"):
            scales = getattr(layer, name, None)
            if scales is not None and torch.unique(scales).numel() != 1:
                logger.warning_once(
                    "NVFP4 linear: %s differs across the fused shards of this "
                    "layer (e.g. q/k/v or gate/up). They are collapsed with "
                    "max(), which reduces accuracy. Prefer a checkpoint with a "
                    "shared global NVFP4 scale for fused layers.",
                    name,
                )

        global_input_scale = layer.input_global_scale.max().to(torch.float32)
        layer.input_global_scale = Parameter(global_input_scale, requires_grad=False)

        layer.weight_global_scale = Parameter(
            layer.weight_global_scale.max().to(torch.float32), requires_grad=False
        )

        if self._use_marlin():
            self._process_weights_for_marlin(layer)
            return

        if not is_blackwell_supported():
            raise ValueError(
                "compressed-tensors NVFP4 native dense GEMM backends require "
                "SM100+. On SM80-SM89 use --fp4-gemm-backend marlin (which is "
                "what --fp4-gemm-backend auto already resolves to there)."
            )

        if get_fp4_gemm_runner_backend().is_flashinfer_trtllm():
            # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
            # FlashInfer provides nvfp4_quantize to quantize + shuffle the
            # layout but we use our own quantization so we have to call
            # shuffles ourselves.
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = layer.weight_packed.data
            weight_scale = layer.weight_scale.data

            epilogue_tile_m = 128
            weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)
            weight_scale = (
                shuffle_matrix_sf_a(weight_scale.view(torch.uint8), epilogue_tile_m)
                .reshape(weight_scale.shape)
                .view(torch.float8_e4m3fn)
            )

            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(weight, requires_grad=False)
        else:
            swizzled_weight_scale = swizzle_blockscale(layer.weight_scale)
            layer.weight_scale = Parameter(swizzled_weight_scale, requires_grad=False)
            layer.weight_packed = Parameter(
                layer.weight_packed.data, requires_grad=False
            )

        layer.alpha = Parameter(
            1 / (layer.input_global_scale * layer.weight_global_scale),
            requires_grad=False,
        )

    def _process_weights_for_marlin(self, layer: torch.nn.Module) -> None:
        """Repack the raw (pre-swizzle) NVFP4 tensors into the Marlin layout.

        Runs INSTEAD of the swizzle above, not after it: Marlin consumes the
        checkpoint's plain ``[N, K/16]`` FP8 block scales and does its own
        permutation, so it must see them before ``swizzle_blockscale`` touches
        them.

        Two naming/convention gaps between the compressed-tensors layout and
        the Marlin helpers, which were written against the ModelOpt layout:

        * ``weight_packed`` here is ``weight`` there. Rebound, not copied.
        * The global weight scale is stored the other way round. Compressed
          tensors stores the MULTIPLY direction (``weight_global_scale ~
          FP8_E4M3_MAX * FP4_E2M1_MAX / amax``; the native path undoes it with
          ``alpha = 1 / (input_global_scale * weight_global_scale)``), while
          ModelOpt stores its reciprocal ``weight_scale_2 ~ amax /
          (FP8_E4M3_MAX * FP4_E2M1_MAX)`` and multiplies with it directly.
          ``prepare_nvfp4_layer_for_marlin`` / ``apply_fp4_marlin_linear``
          follow the ModelOpt direction, so the reciprocal is taken here.
          Getting this wrong does not crash -- it silently scales every output
          by ``global_scale**2``.

        The activation-side scales (``input_global_scale``, ``alpha``) are
        deliberately left untouched and unused: this lane is W4A16, the
        activations stay in ``params_dtype``, and nothing quantises them.

        Both steps follow vLLM's reference implementation of the same lane
        (``CompressedTensorsW4A4Fp4.process_weights_after_loading``: "CT stores
        as divisors, i.e. 1/scale" followed by ``layer.weight =
        layer.weight_packed; del layer.weight_packed``, then
        ``MarlinNvFp4LinearKernel``). vLLM applies the reciprocal on both lanes
        and folds it into ``alpha``; here it stays local to the Marlin branch
        so the native path's arithmetic is byte-for-byte unchanged.
        """
        weight_global_scale = layer.weight_global_scale.data
        if not torch.all(weight_global_scale > 0):
            raise ValueError(
                "NVFP4 Marlin needs a strictly positive weight_global_scale to "
                f"invert, got {weight_global_scale}."
            )

        if self.group_size != 16:
            # The scheme selector only builds this class for group 16, but the
            # Marlin scale permutation hardcodes it, so state it here too
            # rather than relying on prepare_nvfp4_layer_for_marlin's
            # layer.quant_config probe -- LinearBase binds the
            # CompressedTensorsConfig there, and that object has no
            # `group_size` to read.
            raise ValueError(
                f"NVFP4 Marlin requires group_size=16, got {self.group_size}."
            )

        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed
        layer.weight_global_scale = Parameter(
            1.0 / weight_global_scale, requires_grad=False
        )
        prepare_nvfp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._use_marlin():
            return apply_fp4_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                weight_global_scale=layer.weight_global_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

        output_dtype = x.dtype
        w_n, _ = layer.weight_packed.shape
        output_shape = [x.shape[0], w_n]

        # quantize BF16 or FP16 to (FP4 and interleaved block scale)
        x_fp4, x_blockscale = fp4_quantize(x, layer.input_global_scale)

        assert x_fp4.dtype == torch.uint8
        assert layer.weight_packed.dtype == torch.uint8
        assert layer.weight_scale.dtype == torch.float8_e4m3fn
        assert layer.alpha.dtype == torch.float32

        w = layer.weight_packed
        w_blockscale = layer.weight_scale
        if (
            enable_flashinfer_fp4_gemm
            and not get_fp4_gemm_runner_backend().is_cutlass()
        ):
            w = layer.weight_packed.T
            w_blockscale = layer.weight_scale.T

        out = fp4_gemm(
            x_fp4,
            w,
            x_blockscale,
            w_blockscale,
            layer.alpha,
            output_dtype,
            w_n,
        )
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
