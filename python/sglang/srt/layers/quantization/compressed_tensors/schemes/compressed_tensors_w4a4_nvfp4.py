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

__all__ = [
    "CompressedTensorsW4A4Fp4",
    "CompressedTensorsW4A4Fp4Dequant",
    "NVFP4_BLOCK_SIZE",
    "NVFP4_NATIVE_MIN_N",
    "dequantize_nvfp4",
    "nvfp4_marlin_unpackable_reason",
    "nvfp4_native_unpackable_reason",
    "nvfp4_unpackable_reason",
]

#: Elements sharing one FP8-E4M3 block scale in ``nvfp4-pack-quantized``.
NVFP4_BLOCK_SIZE = 16

#: N granularity the NATIVE (CUTLASS / FlashInfer TRTLLM) block-scaled FP4 GEMM
#: requires. ``nvfp4_scaled_mm_kernels.cuh:81`` asserts it outright -- measured
#: 2026-07-31 as "Expected n to be divisible by 32, but got n: 42" on the TP0
#: rank of an uneven TP=3 plan. Unlike the Marlin tile this is a condition on
#: the width the kernel actually receives, i.e. the SHARDED one.
NVFP4_NATIVE_MIN_N = 32

#: The 16 E2M1 code points, indexed by nibble. Same table as
#: ``kvfp4_tensor.E2M1_VALUES``, rebuilt locally because that one is pinned to
#: a device at import time and this helper follows its input's device.
_E2M1_CODE_POINTS = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def dequantize_nvfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Materialise a compressed-tensors NVFP4 weight as a dense tensor.

    ``weight_packed`` is ``[N, K // 2]`` uint8 with two E2M1 nibbles per byte,
    LOW nibble first (element ``2*j`` in the low half of byte ``j``), which is
    the layout ``fp4_quantize`` / ``break_fp4_bytes`` produce and consume.
    ``weight_scale`` is ``[N, K // 16]`` FP8-E4M3, one scale per group of
    ``NVFP4_BLOCK_SIZE`` input channels.

    Scale direction: compressed-tensors stores ``weight_global_scale`` in the
    QUANTISE direction (``FP8_E4M3_MAX * FP4_E2M1_MAX / amax``), so recovering
    the original value DIVIDES by it -- the same convention the Marlin lane
    handles by inverting the scale before handing it to the ModelOpt-shaped
    helpers (see ``_process_weights_for_marlin``). Getting the direction wrong
    does not crash, it scales the whole row block by ``global_scale**2``.

    This reproduces exactly what the producer's quantiser encoded; it adds no
    error of its own. The rounding loss is already baked into the four-bit
    codes on disk.
    """
    if weight_packed.dtype != torch.uint8:
        raise ValueError(
            f"NVFP4 dequantisation expects uint8 packed weights, got "
            f"{weight_packed.dtype}."
        )
    n, k_half = weight_packed.shape
    k = k_half * 2
    if weight_scale.shape != (n, k // NVFP4_BLOCK_SIZE):
        raise ValueError(
            f"NVFP4 dequantisation expects block scales of shape "
            f"{(n, k // NVFP4_BLOCK_SIZE)} for a {(n, k)} weight, got "
            f"{tuple(weight_scale.shape)}."
        )

    codes = torch.empty(n, k, dtype=torch.uint8, device=weight_packed.device)
    codes[:, 0::2] = weight_packed & 0x0F
    codes[:, 1::2] = (weight_packed >> 4) & 0x0F
    lut = torch.tensor(
        _E2M1_CODE_POINTS, dtype=torch.float32, device=weight_packed.device
    )
    values = lut[codes.long()].view(n, k // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    values = values * weight_scale.to(torch.float32).unsqueeze(-1)
    dense = values.view(n, k) / weight_global_scale.to(torch.float32)
    return dense.to(out_dtype)


def nvfp4_marlin_unpackable_reason(layer: torch.nn.Module) -> Optional[str]:
    """Why NVFP4 Marlin can serve no shard of ``layer``, or None if it can.

    Judged on the UNSHARDED geometry (``layer.output_size``), the width a TP=1
    load would see -- available at ``get_quant_method`` time because
    ``ColumnParallelLinear`` computes ``output_size_per_partition`` only after
    ``LinearBase.__init__`` returns. Same verdict, same reasoning and the same
    deliberate split as ``gptq_marlin_unpackable_reason`` (#316): a SHARD that
    misses Marlin's tile is a shard-plan problem and stays a loud error, while
    a module whose FULL output dimension misses the tile can never be served
    by Marlin at any TP -- including TP=1, where there is no plan to blame.

    Task #332, measured 2026-07-31 on ``ocicek/Qwen3.6-27B-NVFP4``: that
    checkpoint's ``config_groups`` targets ``Linear`` wholesale, so the
    gated-delta-net's merged b/a gate is quantised too. Its full width is
    2 x 48 = 96 rows and ``GPTQ_MARLIN_MIN_THREAD_N`` is 64.

    Unlike #316's GPTQ case, the tensor IS packed on disk here, so building the
    layer dense-and-empty would leave it unloaded. The caller therefore routes
    to ``CompressedTensorsW4A4Fp4Dequant`` instead of refusing.
    """
    output_size = _linear_output_size(layer)
    if output_size is None or output_size % GPTQ_MARLIN_MIN_THREAD_N == 0:
        return None
    return (
        f"the unsharded output width is {output_size}, not a multiple of the "
        f"NVFP4 Marlin thread tile {GPTQ_MARLIN_MIN_THREAD_N}"
    )


def nvfp4_native_unpackable_reason(width: int, *, sharded: bool) -> Optional[str]:
    """Why the NATIVE FP4 GEMM cannot serve a ``width``-row output, or None.

    Task #336. The native lane has its own geometry condition, and it is not
    the Marlin one: ``cutlass_scaled_fp4_mm`` asserts ``n % 32 == 0`` on the
    width it is handed, which is the SHARDED width. That is the difference
    that matters, because it inverts #316's rule about who is to blame.

    For Marlin the verdict is taken on the unsharded width on purpose: the
    fork's shard planner coarsens partitions to the quant block, so a shard
    that misses the 64-tile while the module could hit it is a plan bug and
    stays a loud error. The native lane has no such coarsening axis, and on
    the geometry that produced this task no plan exists at all. Measured
    2026-07-31 over three ratios on ``ocicek/Qwen3.6-27B-NVFP4``
    (``auto`` -> 42, ``2,1,1`` -> 48, ``4,1,1`` -> 60): the gated-delta-net
    splits in groups of three value heads (``linear_num_value_heads`` 48
    against ``linear_num_key_heads`` 16), and the merged b/a gate carries two
    rows per value head, so a rank holding ``g`` groups gets ``n = 6 * g``.
    ``6g % 32 == 0`` needs ``3g % 16 == 0``, and 3 is coprime to 16, so it
    needs ``g % 16 == 0`` -- every group on one rank. NO tensor-parallel split
    of that layer satisfies the native lane, the even split included.

    So a sharded miss here is not a plan to fix, it is a lane to leave: the
    caller routes to ``CompressedTensorsW4A4Fp4Dequant``, exactly as the
    Marlin verdict does, and says so per layer in the log.
    """
    if width % NVFP4_NATIVE_MIN_N == 0:
        return None
    which = "sharded" if sharded else "unsharded"
    return (
        f"the {which} output width is {width}, not a multiple of the native "
        f"FP4 GEMM's N granularity {NVFP4_NATIVE_MIN_N}"
    )


def nvfp4_unpackable_reason(
    layer: torch.nn.Module,
    output_size_per_partition: Optional[int] = None,
) -> Optional[str]:
    """Why THIS rank's resolved FP4 lane can serve no form of ``layer``.

    Task #336, generalising #332's Marlin-only verdict. A mixed-architecture
    rig runs both lanes at once -- ``get_fp4_gemm_runner_backend()`` resolves
    per rank against that rank's device -- and the two lanes have DIFFERENT
    geometry conditions on DIFFERENT widths:

    ==========  ======================  =================================
    lane        condition               width it is judged on
    ==========  ======================  =================================
    Marlin      ``% 64`` (thread tile)  unsharded (``layer.output_size``)
    native FP4  ``% 32`` (kernel N)     sharded (this rank's partition)
    ==========  ======================  =================================

    Hence the two moments. ``get_quant_method`` can only see the unsharded
    width -- ``ColumnParallelLinear`` computes ``output_size_per_partition``
    after ``LinearBase.__init__`` returns -- so it is called there without
    ``output_size_per_partition`` and answers for Marlin, plus for the native
    lane in the case no split can rescue either (unsharded width off the 32).
    ``CompressedTensorsLinearMethod.create_weights`` calls it again WITH the
    shard width, which is where the native verdict of #336 actually lands.

    Both answers route to the same place, ``CompressedTensorsW4A4Fp4Dequant``:
    load packed, materialise dense once at load, serve with ``F.linear``. The
    numerics are exact either way (see ``dequantize_nvfp4``).
    """
    if get_fp4_gemm_runner_backend().is_marlin():
        # Marlin keeps #332's unsharded-only verdict; its sharded misses stay
        # a loud error in create_weights, because the plan can fix those.
        return nvfp4_marlin_unpackable_reason(layer)
    if output_size_per_partition is not None:
        return nvfp4_native_unpackable_reason(output_size_per_partition, sharded=True)
    output_size = _linear_output_size(layer)
    if output_size is None:
        return None
    return nvfp4_native_unpackable_reason(output_size, sharded=False)


def _linear_output_size(layer: torch.nn.Module) -> Optional[int]:
    """The UNSHARDED output width of a linear layer, or None if it has none.

    ``LinearBase.__init__`` records ``output_size`` before it asks the quant
    config for a method, so this is readable at scheme-selection time; a
    non-linear module (or a stub without the field) simply has no verdict.
    """
    from sglang.srt.layers.linear import LinearBase

    if not isinstance(layer, LinearBase):
        return None
    output_size = getattr(layer, "output_size", None)
    return output_size if isinstance(output_size, int) else None


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

    #: Overridden by ``CompressedTensorsW4A4Fp4Dequant``: with no kernel in the
    #: picture neither the Marlin tile nor the native N granularity is a
    #: constraint, so ``create_weights`` must not apply either.
    dequantize_on_load = False

    def __init__(self):
        self.group_size = NVFP4_BLOCK_SIZE

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

        if (
            not self.dequantize_on_load
            and self._use_marlin()
            and output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N
        ):
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
                    "projection narrower than the Marlin tile. "
                    "CompressedTensorsConfig.get_linear_scheme routes such a "
                    "layer to CompressedTensorsW4A4Fp4Dequant (#332); reaching "
                    "this raise means the routing was bypassed."
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

        if (
            not self.dequantize_on_load
            and not self._use_marlin()
            and output_size_per_partition % NVFP4_NATIVE_MIN_N
        ):
            # #336. Without this the width reaches cutlass_scaled_fp4_mm and
            # the run dies mid-graph-capture on a kernel assert
            # ("Expected n to be divisible by 32, but got n: 42") with no
            # layer name attached. CompressedTensorsLinearMethod.create_weights
            # routes such a shard to CompressedTensorsW4A4Fp4Dequant before it
            # gets here, so reaching this raise means the routing was bypassed.
            raise ValueError(
                "The native NVFP4 GEMM requires output_size_per_partition to "
                f"be a multiple of {NVFP4_NATIVE_MIN_N}, got "
                f"{output_size_per_partition}. "
                "CompressedTensorsLinearMethod.create_weights routes such a "
                "shard to CompressedTensorsW4A4Fp4Dequant (#336); reaching "
                "this raise means the routing was bypassed."
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


class CompressedTensorsW4A4Fp4Dequant(CompressedTensorsW4A4Fp4):
    """Load NVFP4, serve dense: the fallback for a projection with no tile.

    Third lane of the compressed-tensors NVFP4 scheme, reached only through
    ``nvfp4_unpackable_reason`` (tasks #332 and #336). Both kernel lanes carry
    a width condition and neither can be met by every checkpoint:

    * Marlin needs a multiple of ``GPTQ_MARLIN_MIN_THREAD_N`` (64) UNSHARDED.
      A checkpoint whose ``config_groups`` target ``Linear`` wholesale
      quantises narrow projections too -- on Qwen3.5/3.6 the gated-delta-net's
      merged b/a gate, 96 rows against a 64-wide tile -- and no shard plan and
      no card can widen 96 (#332).
    * The native FP4 GEMM needs a multiple of ``NVFP4_NATIVE_MIN_N`` (32) on
      the SHARDED width, and on that same gate no tensor-parallel split
      delivers one (#336; the arithmetic is in
      ``nvfp4_native_unpackable_reason``).

    Before this lane the outcomes were a load-time abort on every
    pre-Blackwell rank, a mid-graph-capture kernel assert on every Blackwell
    rank at TP > 1, or a checkpoint edit.

    So: keep the packed parameters (the loader must still find somewhere to put
    ``weight_packed`` / ``weight_scale`` -- unlike #316's GPTQ case, the tensor
    IS on disk here), then materialise them once, at load, into a dense
    ``weight`` in ``params_dtype`` and run ``F.linear``.

    **This is not a lossy step.** ``dequantize_nvfp4`` reproduces exactly the
    values the producer's quantiser encoded; the rounding error is already in
    the four-bit codes on disk and dequantisation adds none of its own. What it
    costs is VRAM: bf16 instead of ~4.5 bits per weight, for the layers that
    take this lane. On Qwen3.6-27B those are the 48 GDN b/a gates, together
    ~11.8 M parameters -- 23 MiB dense against 7 MiB packed. It is charged
    against the layers that cannot be served otherwise, never against layers
    a kernel can take.

    Prior art. vLLM solves the same shape two ways, and this is a third:
    ``EmulationNvFp4LinearKernel``
    (``vllm/model_executor/kernels/linear/nvfp4/emulation.py``) is the terminal
    entry of its NVFP4 kernel list and always reports support, but it
    dequantises inside ``apply_weights``, i.e. on EVERY forward;
    ``prepare_fp4_layer_for_marlin`` instead zero-pads N up to the tile
    (``marlin_padded_nk``). Materialising once at load is strictly cheaper than
    the first at the same numerics, and unlike the second it needs no
    agreement between the padded buffer, the per-rank shard widths and the
    column-parallel gather. Padding stays the better answer if these layers
    ever stop being negligible in VRAM.
    """

    dequantize_on_load = True

    def needs_device_kernel(self) -> bool:
        # Dequantisation plus F.linear: no kernel, hence no capability floor.
        return False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Per LOGICAL shard, with that shard's own global scale. The kernel
        # lanes have to collapse the fused shards' scales with max() because
        # one GEMM serves them all (and warn that it costs accuracy); this lane
        # has no such constraint, so a fused qkv/gate_up/b+a layer is
        # reconstructed exactly as the producer quantised it.
        packed = layer.weight_packed
        scales = layer.weight_scale
        global_scales = layer.weight_global_scale.data.flatten()
        widths = list(layer.logical_widths)
        if len(global_scales) not in (1, len(widths)):
            raise ValueError(
                "NVFP4 dequantisation expected one weight_global_scale per "
                f"logical shard ({len(widths)}) or a single shared one, got "
                f"{len(global_scales)}."
            )

        pieces = []
        row = 0
        for index, width in enumerate(widths):
            gs = global_scales[index if len(global_scales) > 1 else 0]
            pieces.append(
                dequantize_nvfp4(
                    packed.data[row : row + width],
                    scales.data[row : row + width],
                    gs,
                    layer.params_dtype,
                )
            )
            row += width
        dense = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]

        for name in (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
            "input_global_scale",
        ):
            if hasattr(layer, name):
                delattr(layer, name)
        layer.register_parameter("weight", Parameter(dense, requires_grad=False))

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, bias)
