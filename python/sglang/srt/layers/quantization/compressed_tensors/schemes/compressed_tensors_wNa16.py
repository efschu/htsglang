# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from typing import Callable, Optional

import math
from fractions import Fraction

import torch
from compressed_tensors.quantization import ActivationOrdering

# yapf conflicts with isort for this block
# yapf: disable
from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
    permute_param_layout_,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils import (
    MarlinLinearLayerConfig,
    apply_gptq_marlin_linear,
    check_marlin_supports_shape,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace,
    marlin_permute_scales,
    marlin_repeat_scales_on_all_ranks,
    marlin_sort_g_idx,
    marlin_zero_points,
)
from sglang.srt.layers.quantization.utils import (
    get_scalar_types,
    replace_parameter,
    unpack_cols,
)
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack


ScalarType, scalar_types = get_scalar_types()

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_TYPES_MAP = {
    4: scalar_types.uint4b8,
    8: scalar_types.uint8b128
}
WNA16_ZP_SUPPORTED_TYPES_MAP = {4: scalar_types.uint4, 8: scalar_types.uint8}
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())


def dequantize_pack_quantized_weight(
    packed: torch.Tensor, scale: torch.Tensor, shape: torch.Size
) -> torch.Tensor:
    """Dense float weight of a compressed-tensors ``pack-quantized`` linear
    (symmetric, group strategy, ``packed_dim=1``): ``packed`` is int32
    ``[out, in/pack_factor]``, ``scale`` is ``[out, in/group]``. Used for the
    handful of layers this line keeps as plain ``nn.Linear`` (the
    hyper-connection mixers, a few MB each) -- their bits are read from the
    checkpoint as they are and widened to the module dtype at load time."""
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

    out_features, in_features = int(shape[0]), int(shape[1])
    if packed.dtype != torch.int32 or packed.dim() != 2 or packed.shape[0] != out_features:
        raise ValueError(
            f"pack-quantized weight: packed {tuple(packed.shape)} {packed.dtype} "
            f"does not fit a [{out_features}, {in_features}] weight"
        )
    if in_features % packed.shape[1] != 0 or 32 % (in_features // packed.shape[1]) != 0:
        raise ValueError(
            f"pack-quantized weight: {in_features} inputs in {packed.shape[1]} "
            "int32 columns is no whole pack factor"
        )
    num_bits = 32 // (in_features // packed.shape[1])
    if scale.dim() != 2 or scale.shape[0] != out_features or in_features % scale.shape[1] != 0:
        raise ValueError(
            f"pack-quantized weight: scale {tuple(scale.shape)} does not fit "
            f"[{out_features}, {in_features}]"
        )
    group = in_features // scale.shape[1]
    q = unpack_from_int32(packed, num_bits, torch.Size((out_features, in_features)))
    dense = q.view(out_features, scale.shape[1], group).to(torch.float32) * scale.to(
        torch.float32
    ).unsqueeze(-1)
    return dense.view(out_features, in_features)


def unpack_dense_subbyte(packed: torch.Tensor, bits: int, n_elems: int) -> torch.Tensor:
    """Unpack a dense little-endian bitstream of ``bits``-wide unsigned values
    (compressed-tensors ``pack-quantized`` for 32 % bits != 0) into int32.

    ``packed`` is [rows, ceil(n_elems*bits/32)] int32; element e occupies bits
    [e*bits, e*bits+bits) of the row's bitstream, so a value may straddle two
    words. Returns [rows, n_elems] int32 in [0, 2**bits).
    """
    rows, words = packed.shape
    w = packed.to(torch.int64) & 0xFFFFFFFF
    e = torch.arange(n_elems, device=packed.device, dtype=torch.int64)
    bit0 = e * bits
    word = bit0 // 32
    shift = bit0 % 32
    mask = (1 << bits) - 1
    lo = (w[:, word] >> shift) & mask
    # bits that spill into the next word (shift + bits > 32)
    spill = shift + bits - 32
    has_spill = spill > 0
    nxt = torch.clamp(word + 1, max=words - 1)
    hi = (w[:, nxt] << (bits - spill.clamp(min=0))) & mask
    hi = torch.where(has_spill.unsqueeze(0), hi, torch.zeros_like(hi))
    return ((lo | hi) & mask).to(torch.int32)


def widen_dense_packed_to_8bit(
    packed: torch.Tensor, src_bits: int, in_features: int
) -> torch.Tensor:
    """Lossless widening of a dense ``src_bits`` packing to the 8-bit
    compressed-tensors packing (4 values per int32, little-endian, unsigned
    with offset 128): q_signed = raw - 2**(src_bits-1); stored8 = q_signed + 128.
    The group scales are unchanged (value = q_signed * scale)."""
    rows = packed.shape[0]
    raw = unpack_dense_subbyte(packed, src_bits, in_features)
    q8 = raw - (1 << (src_bits - 1)) + 128  # in [96, 159] for 6 bit
    assert in_features % 4 == 0
    q8 = q8.view(rows, in_features // 4, 4).to(torch.int64)
    shifts = torch.arange(4, device=packed.device, dtype=torch.int64) * 8
    out = (q8 << shifts).sum(dim=2)
    return out.to(torch.int32)


class CompressedTensorsWNA16(CompressedTensorsLinearScheme):
    _kernel_backends_being_used: set[str] = set()

    def __init__(self,
                 strategy: str,
                 num_bits: int,
                 group_size: Optional[int] = None,
                 symmetric: Optional[bool] = True,
                 actorder: Optional[ActivationOrdering] = None):

        # Line extension (Qwen3.8-Flash-Next, Minachist INT4/INT6 mixed): a
        # symmetric 6-bit group checkpoint is loaded in its dense
        # compressed-tensors packing (32/6 values per int32, little-endian
        # bitstream; measured on the checkpoint: [10240, 480] for 2560
        # inputs) and WIDENED to 8 bit at process_weights_after_loading --
        # lossless (q in [-32, 31] keeps its scale) -- so the existing 8-bit
        # Marlin (uint8b128) path computes it. User order 16.09.2026: no
        # checkpoint rewrite, compute on the int8-capable kernels.
        self.src_num_bits = num_bits
        if num_bits == 6:
            if not symmetric:
                raise ValueError("6-bit compressed-tensors: only symmetric is supported")
            num_bits = 8
        self.pack_factor = (
            Fraction(32, self.src_num_bits)
            if 32 % self.src_num_bits
            else 32 // self.src_num_bits
        )
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError("Marlin kernels require group quantization or "
                             "channelwise quantization, but found no group "
                             "size and strategy is not channelwise.")

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {WNA16_SUPPORTED_TYPES_MAP.keys()}")

        self.quant_type = (WNA16_ZP_SUPPORTED_TYPES_MAP[num_bits]
                           if not self.symmetric else
                           WNA16_SUPPORTED_TYPES_MAP[num_bits])

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def create_weights(self, layer: torch.nn.Module, output_size: int,
                       input_size: int, output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        output_size_per_partition = sum(output_partition_sizes)

        self.kernel_config = MarlinLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=self.group_size,
            zero_points=not self.symmetric,
            has_g_idx=self.has_g_idx
        )

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = (input_size != input_size_per_partition)
        partition_scales = not marlin_repeat_scales_on_all_ranks(
            self.has_g_idx, self.group_size, row_parallel)

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        # dense packing: ceil(in * bits / 32) int32 words per row (== in //
        # pack_factor for 4/8 bit, where 32 % bits == 0)
        packed_input_dim = math.ceil(input_size_per_partition * self.src_num_bits / 32)
        weight = PackedvLLMParameter(input_dim=1,
                                     output_dim=0,
                                     weight_loader=weight_loader,
                                     packed_factor=self.pack_factor,
                                     packed_dim=1,
                                     data=torch.empty(
                                         output_size_per_partition,
                                         packed_input_dim,
                                         dtype=torch.int32,
                                     ))

        weight_scale_args = {
            "weight_loader":
            weight_loader,
            "data":
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            )
        }

        zeros_args = {
            "weight_loader":
            weight_loader,
            "data":
            torch.zeros(
                output_size_per_partition // self.pack_factor,
                scales_and_zp_size,
                dtype=torch.int32,
            )
        }

        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0,
                                                      **weight_scale_args)

            if not self.symmetric:
                qzeros = PackedColumnParameter(output_dim=0,
                                               packed_dim=0,
                                               packed_factor=self.pack_factor,
                                               **zeros_args)
        else:
            weight_scale = GroupQuantScaleParameter(output_dim=0,
                                                    input_dim=1,
                                                    **weight_scale_args)
            if not self.symmetric:
                qzeros = PackedvLLMParameter(input_dim=1,
                                             output_dim=0,
                                             packed_dim=0,
                                             packed_factor=self.pack_factor,
                                             **zeros_args)

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = BasevLLMParameter(data=torch.empty(2,
                                                          dtype=torch.int64),
                                         weight_loader=weight_loader)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        if not self.symmetric:
            layer.register_parameter("weight_zero_point", qzeros)

        # group index (for activation reordering)
        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(data=torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
                                            input_dim=0,
                                            weight_loader=weight_loader)
            layer.register_parameter("weight_g_idx", weight_g_idx)

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Default names since marlin requires empty parameters for these,
        # TODO: remove this requirement from marlin (allow optional tensors)
        self.w_q_name = "weight_packed"
        self.w_s_name = "weight_scale"
        self.w_zp_name = "weight_zero_point"
        self.w_gidx_name = "weight_g_idx"

        device = getattr(layer, self.w_q_name).device
        c = self.kernel_config

        check_marlin_supports_shape(
            c.partition_weight_shape[1],  # out_features
            c.partition_weight_shape[0],  # in_features
            c.full_weight_shape[0],  # in_features
            c.group_size,
        )

        row_parallel = c.partition_weight_shape[0] != c.full_weight_shape[0]
        self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)

        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace(device)

        def _transform_param(
            layer: torch.nn.Module, name: Optional[str], fn: Callable
        ) -> None:
            if name is not None and getattr(layer, name, None) is not None:

                old_param = getattr(layer, name)
                new_param = fn(old_param)
                # replace the parameter with torch.nn.Parameter for TorchDynamo
                # compatibility
                replace_parameter(
                    layer, name, torch.nn.Parameter(new_param.data, requires_grad=False)
                )

        if self.src_num_bits != c.weight_type.size_bits:
            # widen the dense sub-byte packing to the kernel's 8-bit packing
            wp = getattr(layer, self.w_q_name)
            wp.data = widen_dense_packed_to_8bit(
                wp.data,
                src_bits=self.src_num_bits,
                in_features=c.partition_weight_shape[0],
            )
            if hasattr(wp, "_packed_factor"):
                wp._packed_factor = 4

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = gptq_marlin_repack(
                x.data.contiguous(),
                perm=layer.g_idx_sort_indices,
                size_k=c.partition_weight_shape[0],
                size_n=c.partition_weight_shape[1],
                num_bits=c.weight_type.size_bits,
            )
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = marlin_permute_scales(
                x.data.contiguous(),
                size_k=c.partition_weight_shape[0],
                size_n=c.partition_weight_shape[1],
                group_size=c.group_size,
            )
            return x

        if c.has_g_idx:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(
                getattr(layer, self.w_gidx_name)
            )
            _transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if c.zero_points:
            grouped_k = (
                c.partition_weight_shape[0] // c.group_size if c.group_size != -1 else 1
            )
            _transform_param(
                layer,
                self.w_zp_name,
                lambda x: marlin_zero_points(
                    unpack_cols(
                        x.t(),
                        c.weight_type.size_bits,
                        grouped_k,
                        c.partition_weight_shape[1],
                    ),
                    size_k=grouped_k,
                    size_n=c.partition_weight_shape[1],
                    num_bits=c.weight_type.size_bits,
                ),
            )
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))
        _transform_param(layer, self.w_q_name, transform_w_q)
        _transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        c = self.kernel_config

        def _get_weight_params(
            layer: torch.nn.Module,
        ) -> tuple[
            torch.Tensor,  # w_q
            torch.Tensor,  # w_s
            Optional[torch.Tensor],  # w_zp,
            Optional[torch.Tensor],  # w_gidx
        ]:
            return (
                getattr(layer, self.w_q_name),
                getattr(layer, self.w_s_name),
                getattr(layer, self.w_zp_name or "", None),
                getattr(layer, self.w_gidx_name or "", None),
            )

        w_q, w_s, w_zp, w_gidx = _get_weight_params(layer)

        # `process_weights_after_loading` will ensure w_zp and w_gidx are not
        #  None for marlin
        return apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            g_idx=w_gidx,  # type: ignore
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            is_k_full=self.is_k_full,
            bias=bias,
        )
