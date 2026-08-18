# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/linear.py"""

from __future__ import annotations

import itertools
import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter

from sglang.kernel_api_logging import wrap_method_with_debug_kernel_once
from sglang.srt.distributed import (
    divide,
    get_tp_group,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_quant_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.distributed.tp_ar_pipeline import (
    issue_deferred_all_reduce,
    pipelined_row_all_reduce,
    tp_ar_deferred_enabled,
    tp_ar_pipeline_enabled,
)
from sglang.srt.distributed.utils import (
    attn_kv_replicated,
    attn_q_partition_groups,
    attn_q_partition_units,
    tp_loaded_shard_start,
    tp_partition_size,
    tp_partition_sizes,
    tp_plan_active,
)
from sglang.srt.layers.dp_attention import (
    is_allocation_symmetric,
)
from sglang.srt.layers.moe.utils import (
    barlink_mlp_ar_overlap_comm,
    should_skip_mlp_all_reduce,
)
from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    BlockQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
    RowvLLMParameter,
    _ColumnvLLMParameter,
)
from sglang.srt.layers.utils import pad_or_narrow_weight
from sglang.srt.runtime_context import get_forward, get_parallel, get_server_args
from sglang.srt.utils import get_bool_env_var, is_cpu, is_hip, is_npu, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )

_is_hip = is_hip()
_disable_hip_linear_quant = _is_hip and get_bool_env_var(
    "SGLANG_ROCM_DISABLE_LINEARQUANT"
)

logger = logging.getLogger(__name__)


def _tp_ar_pipeline_quant_comms(forward_batch) -> bool:
    """Mirror of the quantized-communications condition used below.

    Hoisted so the pipeline can decline that path: a quantized all-reduce
    changes the payload dtype per collective, which the token-slice cost
    model does not describe.
    """
    if forward_batch is None:
        return False
    return (
        not forward_batch.forward_mode.is_decode_or_idle()
        and get_server_args().enable_quant_communications
    )


def _tp_ar_pipeline_max_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """Element-wise max of a small scalar vector across the TP group.

    The calibration's only job here is to make every rank agree on the same
    numbers before they derive a slice count from them; the max also picks
    the slowest rank per term, which is the rank that paces the group.
    """
    torch.distributed.all_reduce(
        tensor,
        op=torch.distributed.ReduceOp.MAX,
        group=get_tp_group().device_group,
    )
    return tensor


WEIGHT_LOADER_V2_SUPPORTED = [
    "CompressedTensorsLinearMethod",
    "AWQLinearMethod",
    "GPTQMarlinLinearMethod",
    "Fp8LinearMethod",
    "BlockInt8LinearMethod",
    "MarlinLinearMethod",
    "QQQLinearMethod",
    "GPTQMarlin24LinearMethod",
    "TPUInt8LinearMethod",
    "GPTQLinearMethod",
    "FBGEMMFp8LinearMethod",
    "GPTQLinearAscendMethod",
    "GPTQLinearIntelAMXMethod",
    "GPTQMoEAscendMethod",
    "GPTQMoEIntelAMXMethod",
    "ModelOptFp8LinearMethod",
    "ModelOptFp4LinearMethod",
    "IPEXAWQLinearMethod",
    "PetitNvFp4LinearMethod",
    "QuarkInt4Fp8LinearMethod",
]

_is_cpu = is_cpu()
_is_npu = is_npu()


def adjust_marlin_shard(param, shard_size, shard_offset):
    marlin_tile_size = getattr(param, "marlin_tile_size", None)
    if marlin_tile_size is None:
        return shard_size, shard_offset

    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def adjust_bitsandbytes_4bit_shard(
    param: Parameter, shard_offsets: Dict[str, Tuple[int, int]], loaded_shard_id: str
) -> Tuple[int, int]:
    """Adjust the quantization offsets and sizes for BitsAndBytes sharding."""

    total, _ = shard_offsets["total"]
    orig_offset, orig_size = shard_offsets[loaded_shard_id]

    quantized_total = param.data.shape[0]
    quantized_offset = orig_offset * quantized_total // total
    quantized_size = orig_size * quantized_total // total

    return quantized_size, quantized_offset


def adjust_scalar_to_fused_array(param, loaded_weight, shard_id):
    """For fused modules (QKV and MLP) we have an array of length
    N that holds 1 scale for each "logical" matrix. So the param
    is an array of length N. The loaded_weight corresponds to
    one of the shards on disk. Here, we slice the param based on
    the shard_id for loading.
    """
    qkv_idxs = {"q": 0, "k": 1, "v": 2}

    if isinstance(shard_id, str):
        shard_id = qkv_idxs[shard_id]
    elif not isinstance(shard_id, int):
        raise ValueError(f"Unknown Shard Id {shard_id}")

    # AutoFP8 scales do not have a shape
    # compressed-tensors scales do have a shape
    if len(loaded_weight.shape) != 0:
        assert loaded_weight.shape[0] == 1
        loaded_weight = loaded_weight[0]

    return param[shard_id], loaded_weight


def adjust_shard_offsets(shard_offsets, loaded_weight, dim):
    actual_weight_size = loaded_weight.size(dim)
    target_weight_size = shard_offsets[-1][-1] + shard_offsets[-1][-2]
    if actual_weight_size != target_weight_size:
        new_shard_offsets = []
        new_offset = 0
        for shard_id, shard_offset, shard_size in shard_offsets:
            actual_shard_size = actual_weight_size * shard_size // target_weight_size
            new_shard_offsets.append((shard_id, new_offset, actual_shard_size))
            new_offset += actual_shard_size
        return new_shard_offsets
    return shard_offsets


def _marlin_uneven_tp_block() -> int:
    """The ONE marlin block both dims coarsen by: ``lcm(MIN_THREAD_N,
    MIN_THREAD_K)`` = 128.

    SYMMETRIC, and that is the correction #385 made to this rule. The obvious
    reading of marlin's constraints is per-axis -- 64 on the output dim, 128 on
    the input -- and #383 shipped exactly that. It is wrong, for the reason
    every existing sibling states in its own docstring
    (``awq_uneven_tp_block``): "Both dims carry the same value so a
    column-parallel OUTPUT split (gate_up) lands on the same boundary as its
    coupled row-parallel INPUT split (down) -- they partition the same
    intermediate dimension and must coarsen identically."

    gate_up's OUTPUT and down_proj's INPUT are the same intermediate dimension.
    Coarsening them by different blocks splits that dimension two different
    ways: measured at the rig vector, gate_up implies per-rank intermediate
    [14880, 8960, 8928] while down_proj is built for [14848, 8960, 8960]. The
    weight is then repacked for one k and handed an activation of another, and
    the GEMM's own shape check catches it --
    ``gptq_marlin.cuh:836`` verifies ``b_q_weight.size(0) == size_k / 16``
    against the ACTIVATION's k, which is how #377 gap 3 presented
    (``Tensor match failed for Tensor<568, 20480>``). Not an alignment gap: a
    coupled-dimension disagreement that asymmetric alignment created.
    """
    n, k = _marlin_min_thread_pair()
    return math.lcm(n, k)


def _marlin_min_thread_pair() -> tuple:
    """``(GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MIN_THREAD_K)`` = (64, 128).

    IMPORTED, not restated -- these are the constants
    ``marlin_utils.verify_marlin_supports_shape`` checks
    ``output_size_per_partition`` and ``input_size_per_partition`` against, and
    the five existing siblings already coarsen to them. They are STRICTER than
    the repack tiles (``marlin.cuh``: tile_n 64, tile_k 16): k needs 128, so
    aligning to the repack tile alone still fails the GEMM's own check.
    """
    from sglang.srt.layers.quantization.marlin_utils import (
        GPTQ_MARLIN_MIN_THREAD_K,
        GPTQ_MARLIN_MIN_THREAD_N,
    )

    return (GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MIN_THREAD_K)


def _marlin_packable_family(quant_config) -> bool:
    """Could THIS CHECKPOINT be marlin-packed on some rank? Rank-uniform.

    DELIBERATELY DEVICE-FREE, and that is the whole design of this predicate.
    The obvious implementation -- read the resolved scheme's ``use_marlin``,
    which is what ``CompressedTensorsW8A16Fp8`` actually sets -- is WRONG here,
    and wrong in a way that is worse than the bug it fixes.

    ``use_marlin`` is rank-local. ``CompressedTensorsW8A8Fp8.get_min_capability()
    is 89, so on a mixed rig the SAME checkpoint resolves differently per rank:
    measured on this one (#377), TP0 on the 5090 (sm120) took the native FP8
    scheme while TP1/TP2 on the 3080s (sm86) fell back to
    ``CompressedTensorsW8A16Fp8`` + marlin -- and only TP1/TP2 raised. A unit
    count derived from that verdict would differ BETWEEN RANKS, so the ranks
    would disagree about the shard plan: silently mismatched shapes instead of
    a loud repack abort. ``modelopt_fp4_uneven_tp_block`` makes the same
    argument for the same reason -- the block must be a property of the
    checkpoint, not of the local device.

    So the question asked here is not "does this rank use marlin" but "can this
    checkpoint be marlin-packed anywhere in the group".

    ASKED OF THE BACKEND, NOT OF ITS NAME (#500-B18). This used to be a
    class-NAME list held here --
    ``("fp8config", "compressedtensorsconfig", "fbgemmfp8config")`` -- and a
    name list in a module that owns none of the kernels is the §12 family
    #443/#446 fixed twice already. It missed ``MarlinConfig`` itself, which
    exposes no ``weight_block_size`` to coarsen by and repacks through marlin
    by definition, so an element-granular family on a marlin checkpoint landed
    mid-tile: the #377/#383 abort, reached again through a different class.
    The capability is now declared where the kernels are known, on
    ``QuantizationConfig.marlin_packable_linear``.

    Coarsening is only ever safe-but-coarser, so a backend that declares it
    without needing it on some rank pays a slightly coarser split and never
    correctness -- while keeping every rank's plan identical. A backend that
    needs it and does not declare it aborts at weight load. Declare when in
    doubt.
    """
    if quant_config is None:
        return False
    return bool(getattr(quant_config, "marlin_packable_linear", False))


def _quant_block_aligned_units(
    total: int,
    units: Optional[int],
    quant_config,
    block_idx: int,
) -> Optional[int]:
    """Coarsen an element-granular unit family so every rank's shard is a
    multiple of the weight-quant block (block-quantized FP8 etc.).

    Head-granular families (unit element counts that are already
    multiples of the quant block) pass through unchanged; only
    fine-grained families (e.g. units == total, one element per unit)
    are re-expressed in quant-block units. block_idx: 0 = output/block_n,
    1 = input/block_k.
    """
    if units is None or quant_config is None:
        return units
    # GGUF quantizes along the INPUT dim only: every output row is a whole
    # sequence of input-dim ggml blocks, so OUTPUT-dim (block_idx == 0)
    # sharding never splits a quant block and must NOT be coarsened. GGUFConfig
    # reports a nominal weight_block_size of [256, 256], which would otherwise
    # wrongly merge fine-grained GDN head units — e.g. in_proj_qkv's key_dim
    # 2048 has 128-element k-head units (2048/16); coarsening them to the 256
    # block yields 8 units instead of 16, so in_proj partitions heads [8,4,4]
    # while the model's gdn_tp_units (value_dim basis, 256-elem units, no
    # coarsening) and every other GDN tensor use 16 -> [7,5,4]. That mismatch
    # misaligns per-rank q/k/v/z against A_log/conv/scan and collapses the
    # linear-attention output on all but the last rank (uneven-TP GGUF garbage).
    if block_idx == 0 and getattr(quant_config, "get_name", lambda: "")() == "gguf":
        return units
    raw = getattr(quant_config, "weight_block_size", None)
    block = raw[block_idx] if raw else None
    # ASYMMETRIC EXPOSED BLOCK (#444b) -- the eighth sibling, and the first one
    # whose config exposes a block that was never an ALIGNMENT registration.
    # MXFP8 pins ``weight_block_size = [1, 32]``
    # (``quantization/fp8.py:from_config``) because the OCP spec fixes the
    # QUANTIZATION block to one row by 32 columns; that value is a fact about
    # the scale layout, not a decision about how a shard may be cut. Read
    # per-axis it says "output dim: element granularity, input dim: 32", which
    # is precisely the coupled-dimension disagreement ``_marlin_uneven_tp_block``
    # documents for #385: gate_up's OUTPUT and down_proj's INPUT are the same
    # intermediate dimension, so cutting one at 1 and the other at 32 builds the
    # two halves of one MLP for two different intermediates.
    #
    # lcm of the two axes, then the family's usual marlin fold. For MXFP8 that
    # is lcm(1, 32) = 32 and then lcm(32, 128) = 128 -- the same block AWQ
    # already imposes for its group size 32, so it costs no extra partition
    # tax. Symmetric exposures (FP8 block [128,128], GGUF [256,256], and every
    # group-size sibling) are untouched: for them ``raw[0] == raw[1]`` and this
    # branch does not fire.
    asymmetric_block = bool(raw) and len(raw) == 2 and raw[0] != raw[1]
    if asymmetric_block:
        block = math.lcm(int(raw[0]), int(raw[1]))
    # MARLIN (#383). A marlin-packed weight is repacked into 16x64 tiles, so
    # the per-rank shard must be a multiple of the tile on its axis -- 64 on
    # the output dim, 16 on the input dim. That constraint is INDEPENDENT of
    # weight_block_size, and the checkpoints that hit it are exactly the ones
    # without it: FP8-dynamic (no block size), GPTQ, AWQ. The #82 16-element
    # MLP coarsening is finer than 64, so an uneven split lands mid-tile and
    # the repack refuses at weight load -- measured on Mistral-Small-24B FP8
    # at ratio [29607,17780,17780]: gate_up 65536 in 4096 units partitions to
    # [29776, 17888, 17872] and dies with "size_n = 17888 is not divisible by
    # tile_n_size = 64" (#377). Sixth sibling of the alignment family --
    # and the one whose config exposes NO weight_block_size to coarsen by,
    # which is why the other five (group-size based) never reached it.
    #
    # lcm, not max: for the power-of-two blocks that occur (128, 64, 32) they
    # are the same number, and lcm stays correct if a block size ever is not a
    # multiple of the tile.
    # Only when NO sibling has already claimed this config. A config that
    # exposes a block has had its alignment decided by the family member that
    # knows its kernel -- AWQ/GPTQ/AutoRound/NVFP4 fold marlin's 128 in
    # already, and INT8-W8A8 deliberately folds only 16 because its path is
    # NOT marlin. Folding 64 on top of an existing block would silently
    # re-plan those vehicles; measured, it changes the INT8-W8A8 dense MLP
    # family (block 16 -> lcm 64), which is the fork's default serving model.
    # The gap this closes is exactly the configs that expose nothing -- plus,
    # since #444b, the ones whose exposed block is a quantization fact rather
    # than an alignment registration (asymmetric, see above).
    if (block is None or asymmetric_block) and _marlin_packable_family(quant_config):
        block = math.lcm(block or 1, _marlin_uneven_tp_block())
    if not block:
        return units
    from sglang.srt.distributed.utils import block_aligned_units

    return block_aligned_units(total, units, block)


class LinearBase(torch.nn.Module):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        # The state-dict path of this layer, available to every quant method
        # from create_weights onwards. Several subclasses already re-assign it
        # after super().__init__(); doing it here as well makes a shard-shape
        # error name the module instead of its class (#353).
        self.prefix = prefix
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        if quant_config is None:
            from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

        if self.quant_method is not None:
            wrap_method_with_debug_kernel_once(
                self.quant_method,
                "apply",
                op_name=f"sglang.quant_method.{self.quant_method.__class__.__name__}.apply",
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
        )

        # All the linear layer supports quant method.
        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            self.input_size,
            [self.output_size],
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # If the weight on disk does not have a shape, give it one
        # (such scales for AutoFp8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        # Special case for GGUF. Every OTHER parallel linear (Column, Merged,
        # QKV, Row) already carries this branch; ReplicatedLinear did not,
        # which made a GGUF-packed replicated module unloadable: `qweight` is
        # a GGUFUninitializedParameter whose `.size()` is `torch.Size([0])`
        # until `materialize()` runs, so the shape assert below rejected the
        # packed bytes, and `qweight_type` was copied into `.data` while the
        # `.weight_type` ATTRIBUTE the kernels actually read stayed at its
        # 0 (= F32) default. Replicated means no output sharding, so the
        # materialized shape is the loaded shape as-is.
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            param.materialize(tuple(loaded_weight.shape), dtype=loaded_weight.dtype)

        # The per-tensor quant-scale must be 1 dimension
        if _is_npu:
            if param.size() != loaded_weight.size() and param.size(0) == 1:
                if torch.allclose(loaded_weight, loaded_weight[0]):
                    loaded_weight = loaded_weight[:1]
                else:
                    raise ValueError(f"{loaded_weight} are not all equal")

            if param.dtype == torch.int8 or loaded_weight.dtype == torch.int8:
                assert param.dtype == loaded_weight.dtype, (
                    "init para dtype and loaded weight dtype should be the same"
                )

        assert param.size() == loaded_weight.size(), (
            f"{param.shape=} {param.dtype=} {loaded_weight.shape=} {loaded_weight.dtype=}"
        )
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        output = self.quant_method.apply(self, x, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s


class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        output_sizes: list of output sizes packed into one output, like for QKV
                       the list would be size 3.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        output_sizes: Optional[List[int]] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        use_presharded_weights: bool = False,
        skip_block_quant_check: bool = False,
        tp_units: Optional[int] = None,
        tp_family: Optional[str] = None,
        tp_q_groups: Optional[int] = None,
    ):
        super().__init__(
            input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix
        )

        self.gather_output = gather_output
        self.use_presharded_weights = use_presharded_weights

        # Divide the weight matrix along the last dimension.
        if tp_rank is None:
            tp_rank = get_parallel().tp_rank
        if tp_size is None:
            tp_size = get_parallel().tp_size
        self.tp_rank, self.tp_size = tp_rank, tp_size
        # Uneven TP (--rank-tp-ratio): unit count of this layer's output
        # dimension (e.g. kv heads); shard cuts land on unit boundaries.
        # Without an installed ratio plan, tp_partition_size() reproduces
        # divide() exactly, so the default path is unchanged.
        if not hasattr(self, "tp_units"):
            self.tp_units = tp_units
        # tp_family names an optional FAMILY shard plan (e.g. "mlp",
        # --rank-mlp-ratio / SGLANG_UNEVEN_MLP_VECTOR) that overrides the
        # base vector for this layer only. Families without an installed
        # vector fall back to the base plan, so passing a family never
        # changes behavior on its own.
        if not hasattr(self, "tp_family"):
            self.tp_family = tp_family
        # kv-boundary alignment for the Q dimension (task #116): kv_total
        # under REPLICATED-KV, else None. None on every non-q layer and the
        # default path -> the split stays byte-identical. Only QKVParallelLinear
        # sets a non-None value (in __init__, before this super() runs).
        if not hasattr(self, "tp_q_groups"):
            self.tp_q_groups = tp_q_groups
        assert self.quant_method is not None
        # tp_units is defined against the PER-PART output size (merged
        # layers partition each part independently), so quant-block
        # coarsening must use the same basis — the total would halve the
        # effective granularity for gate_up-style two-part layers.
        _units_basis = (
            self.output_sizes[0]
            if hasattr(self, "output_sizes") and self.output_sizes
            else self.output_size
        )
        # Quant-block alignment is a property of THIS layer's weights, not of
        # the model-global quant config: layers the config skips (e.g. a bf16
        # vision tower under `ignore` / block_name_to_quantize resolve to
        # UnquantizedLinearMethod) carry no group/block constraint and must
        # keep their raw units. Coarsening them can be fatal, not just
        # wasteful — e.g. a vision o_proj with 16 heads of 72 elems under a
        # 128-block config coarsens to lcm(72,128)=1152 -> ONE unit, which
        # cannot be split across ranks.
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        _layer_quant_config = (
            None
            if isinstance(self.quant_method, UnquantizedLinearMethod)
            else quant_config
        )
        self.tp_units = _quant_block_aligned_units(
            _units_basis, self.tp_units, _layer_quant_config, 0
        )
        if getattr(self, "kv_replicated_parts", None):
            # REPLICATED-KV QKV mode (TP > num_kv_heads, task #62): the k/v
            # output blocks are FULLY replicated on every rank (entry True)
            # while the q block is partitioned by the shard plan in
            # kv_total-sized q-head units (self.tp_units). The whole-output
            # partition below cannot express this mix, so build the per-part
            # sizes directly. Only QKVParallelLinear sets this attribute; the
            # default path is untouched.
            assert hasattr(self, "output_sizes") and len(
                self.kv_replicated_parts
            ) == len(self.output_sizes)
            self.output_partition_sizes = [
                (
                    output_size
                    if replicated
                    else tp_partition_size(
                        output_size,
                        tp_size,
                        tp_rank,
                        self.tp_units,
                        self.tp_family,
                        self.tp_q_groups,
                    )
                )
                for output_size, replicated in zip(
                    self.output_sizes, self.kv_replicated_parts
                )
            ]
            self.output_size_per_partition = sum(self.output_partition_sizes)
        else:
            self.output_size_per_partition = tp_partition_size(
                self.output_size, tp_size, tp_rank, self.tp_units, self.tp_family
            )
            self.output_partition_sizes = [self.output_size_per_partition]
            # If QKV or MergedColumn, use output size of each partition.
            if hasattr(self, "output_sizes"):
                self.output_partition_sizes = [
                    tp_partition_size(
                        output_size, tp_size, tp_rank, self.tp_units, self.tp_family
                    )
                    for output_size in self.output_sizes
                ]

        if output_sizes is None:
            output_sizes = [output_size]

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            skip_block_quant_check=skip_block_quant_check,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if self.tp_units is not None and tp_plan_active(self.tp_size, self.tp_family):
            # Uneven TP only: expose the unit count (and family, so the
            # loaders resolve the SAME weight vector the shapes were
            # partitioned with) to the v2 weight loaders in
            # layers/parameter.py (AFTER create_weights, which registers
            # the parameters). Not set on the default path.
            for p in self.parameters():
                set_weight_attrs(
                    p,
                    {
                        "tp_units": self.tp_units,
                        "tp_family": self.tp_family,
                        "tp_q_groups": self.tp_q_groups,
                    },
                )
        if bias:
            self.bias = Parameter(
                torch.zeros(self.output_size_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)
        param_data = param.data

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if output_dim is not None:
                # Uneven TP: this rank's row (output) partition. Reproduces the
                # even `// tp_size` split when no ratio plan is installed. GGUF
                # rows are whole, so output sharding never splits a quant block
                # (the narrow below uses the matching prefix-sum start).
                weight_shape[output_dim] = tp_partition_size(
                    weight_shape[output_dim],
                    self.tp_size,
                    self.tp_rank,
                    self.tp_units,
                    self.tp_family,
                )
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)
            param_data = param.data

        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        if output_dim is not None and not use_bitsandbytes_4bit:
            shard_size = param_data.shape[output_dim]
            # Even TP: rank * shard_size. Uneven TP: prefix-sum offset of
            # this rank's partition in the checkpoint dimension.
            start_idx = tp_loaded_shard_start(
                loaded_weight.shape[output_dim],
                self.tp_size,
                self.tp_rank,
                shard_size,
                self.tp_units,
                self.tp_family,
            )

            if _is_cpu:
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    param_data,
                    loaded_weight,
                    0,  # param_data_start
                    start_idx,
                    output_dim,
                    shard_size,
                    not self.use_presharded_weights,
                )
            else:
                if not self.use_presharded_weights:
                    loaded_weight = loaded_weight.narrow(
                        output_dim, start_idx, shard_size
                    )

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape, (
            f"param_data.shape={param_data.shape} != loaded_weight.shape={loaded_weight.shape}"
        )
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: Parameter, loaded_weight: torch.Tensor):
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        if isinstance(param, _ColumnvLLMParameter):
            param.load_column_parallel_weight(
                loaded_weight,
                tp_rank=self.tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )
        else:
            # FIXME: This branch is needed to load deepseek v3 awq.
            # However, we should fix this and avoid the branching here.
            # After QuantizedRL reload, params might still need tp_rank
            try:
                param.load_column_parallel_weight(
                    loaded_weight,
                    tp_rank=self.tp_rank,
                    use_presharded_weights=self.use_presharded_weights,
                )
            except TypeError:
                # Fallback for parameters that don't accept additional args
                param.load_column_parallel_weight(loaded_weight)

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", gather_output={self.gather_output}"
        return s


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        use_presharded_weights: bool = False,
        tp_units: Optional[int] = None,
        tp_family: Optional[str] = None,
    ):
        self.output_sizes = output_sizes
        if tp_rank is None:
            tp_rank = get_parallel().tp_rank
        if tp_size is None:
            tp_size = get_parallel().tp_size
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.tp_units = tp_units
        self.tp_family = tp_family
        for output_size in output_sizes:
            # Validates partitionability of every packed output. Even TP:
            # divisibility by tp_size (assert, as before). Uneven TP:
            # raises a ValueError naming the dimension and the weight
            # vector if the shard plan cannot split it.
            tp_partition_sizes(output_size, tp_size, tp_units, tp_family)
        self.use_presharded_weights = use_presharded_weights
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
            use_presharded_weights=use_presharded_weights,
        )
        self.prefix = prefix

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ):
        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)

        if isinstance(loaded_shard_id, tuple):
            # GDN in_proj_qkvz packs [q, k, v, z]. For GGUF the q,k,v part
            # arrives as ONE already-fused tensor (attn_qkv) tagged with the
            # multi-index shard (0, 1, 2); load it as a single combined shard
            # keyed by the whole tuple (the fused output already has q|k|v in
            # order, so downstream padding/apply treat it as one contiguous
            # shard). bf16/fp8 params instead take the v2 merged-column path.
            if is_gguf_weight_type:
                for idx in loaded_shard_id:
                    param.data[idx].copy_(loaded_weight)
                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
                return
            if is_gguf_weight:
                output_dim = getattr(param, "output_dim", None)
                # The GGUF attn_qkv tensor fuses the sub-components named by the
                # tuple (e.g. q|k|v = output_sizes[0:3]) in one tensor. For TP>1
                # each sub-component must be split PER HEAD independently, then
                # re-fused for this rank — splitting the whole fused block would
                # cut through the q/k/v boundaries. At TP=1 each partition is the
                # whole component, so this reproduces the un-sharded fused tensor.
                parts = []
                offset = 0
                for comp in loaded_shard_id:
                    comp_size = self.output_sizes[comp]
                    comp_w = loaded_weight.narrow(output_dim, offset, comp_size)
                    offset += comp_size
                    my_size = tp_partition_size(
                        comp_size,
                        self.tp_size,
                        self.tp_rank,
                        self.tp_units,
                        self.tp_family,
                    )
                    my_start = tp_loaded_shard_start(
                        comp_size,
                        self.tp_size,
                        self.tp_rank,
                        my_size,
                        self.tp_units,
                        self.tp_family,
                    )
                    parts.append(comp_w.narrow(output_dim, my_start, my_size))
                loaded_weight = (
                    parts[0] if len(parts) == 1 else torch.cat(parts, dim=output_dim)
                )
                param.shard_id.append(loaded_shard_id)
                param.shard_id_map[loaded_shard_id] = len(param.data_container)
                param.data_container.append(loaded_weight)
                return
            if hasattr(param, "load_merged_column_weight"):
                return self.weight_loader_v2(param, loaded_weight, loaded_shard_id)
            raise NotImplementedError(
                "Shard id with multiple indices is not supported in weight_loader, "
                "please use weight_loader_v2 instead."
            )

        if is_gguf_weight_type:
            param.data[loaded_shard_id].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            return

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            total = loaded_weight.size(output_dim)
            # Uneven TP: this rank's row (output) partition; reproduces even
            # //tp_size with no plan. Rows are whole -> no quant-block concern.
            shard_size = tp_partition_size(
                total, self.tp_size, self.tp_rank, self.tp_units, self.tp_family
            )
            start_idx = tp_loaded_shard_start(
                total,
                self.tp_size,
                self.tp_rank,
                shard_size,
                self.tp_units,
                self.tp_family,
            )

            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)
        # Special case for per-tensor scale to load scalar into fused array.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv/mlp).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            current_shard_offset = 0
            shard_offsets: List[Tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                effective_size = (
                    output_size // self.tp_size
                    if self.use_presharded_weights
                    else output_size
                )
                shard_offsets.append((i, current_shard_offset, effective_size))
                current_shard_offset += effective_size
            packed_dim = getattr(param, "packed_dim", None)

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            if _is_cpu:
                shard_offsets = adjust_shard_offsets(
                    shard_offsets, loaded_weight, output_dim
                )

            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantization.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                if use_bitsandbytes_4bit:
                    index = list(itertools.accumulate([0] + self.output_sizes))
                    orig_offsets = {
                        str(i): (index[i], size)
                        for i, size in enumerate(self.output_sizes)
                    }
                    orig_offsets["total"] = (self.output_size, 0)
                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_offsets, str(shard_id)
                    )

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes)
        if output_dim is not None:
            # Per packed output its own partition: offset = sum of the
            # preceding outputs' per-rank sizes, size = this output's
            # per-rank size (uneven TP: from the shard plan; even TP:
            # exactly output_size // tp_size as before).
            shard_offset = sum(
                tp_partition_size(
                    sz, self.tp_size, self.tp_rank, self.tp_units, self.tp_family
                )
                for sz in self.output_sizes[:loaded_shard_id]
            )
            shard_size = tp_partition_size(
                self.output_sizes[loaded_shard_id],
                self.tp_size,
                self.tp_rank,
                self.tp_units,
                self.tp_family,
            )
            # Special case for quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            if use_bitsandbytes_4bit:
                shard_size = loaded_weight.shape[output_dim]
                shard_offset = loaded_weight.shape[output_dim] * loaded_shard_id

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            start_idx = tp_loaded_shard_start(
                loaded_weight.shape[output_dim],
                self.tp_size,
                self.tp_rank,
                shard_size,
                self.tp_units,
                self.tp_family,
            )

            if _is_cpu:
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    param_data,
                    loaded_weight,
                    0,  # param_data_start
                    start_idx,
                    output_dim,
                    shard_size,
                    not use_bitsandbytes_4bit and not self.use_presharded_weights,
                )
            else:
                # bitsandbytes loads the weights of the specific portion
                # no need to narrow here
                if not use_bitsandbytes_4bit and not self.use_presharded_weights:
                    # Padding for special case like qwen2_5_VL's mlp which is not 8-aligned
                    end_idx = start_idx + shard_size
                    if end_idx > loaded_weight.shape[output_dim]:
                        loaded_weight = pad_or_narrow_weight(
                            loaded_weight, output_dim, start_idx, shard_size
                        )
                    else:
                        loaded_weight = loaded_weight.narrow(
                            output_dim, start_idx, shard_size
                        )

        # Special case for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_offset = loaded_shard_id * shard_size
            param_data = param_data.narrow(0, shard_offset, shard_size)

        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def _load_fused_module_from_checkpoint(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        output_sizes: list[int] | None = None,
    ):
        """
        Handle special case for models where MLP layers are already
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """

        current_shard_offset = 0
        shard_offsets: List[Tuple[int, int, int]] = []
        output_sizes = output_sizes or self.output_sizes
        for i, output_size in enumerate(output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size
        if _is_cpu:
            from sglang.srt.model_loader.weight_utils import (
                pad_loaded_weight,
            )

            loaded_weight = pad_loaded_weight(
                loaded_weight, param.output_dim, output_sizes
            )

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if (
                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )
            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def _load_merged_block_scale(
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        """
        Handle block-wise scale loading for MergedColumnParallelLinear.
        Similar to QKVParallelLinear._load_qkv_block_scale, but for merged column layers.
        """
        if tp_plan_active(self.tp_size):
            raise NotImplementedError(
                "Block-quant scale loading (fused on disk) is not supported "
                "with uneven TP (--rank-tp-ratio) yet."
            )
        weight_block_size = self.quant_method.quant_config.weight_block_size
        block_n, _ = weight_block_size[0], weight_block_size[1]
        block_n = 1 if getattr(param, "format_ue8m0", False) else block_n

        # Calculate block sizes for each shard
        shard_block_sizes = []
        shard_block_offsets = []
        current_block_offset = 0
        for output_size in self.output_sizes:
            shard_block_size = (output_size + block_n - 1) // block_n
            shard_block_sizes.append(shard_block_size)
            shard_block_offsets.append(current_block_offset)
            current_block_offset += shard_block_size

        if _is_cpu:
            from sglang.srt.model_loader.weight_utils import (
                pad_loaded_weight,
            )

            loaded_weight = pad_loaded_weight(
                loaded_weight, param.output_dim, shard_block_sizes
            )

        # Load each shard
        for shard_id, (shard_block_offset, shard_block_size) in enumerate(
            zip(shard_block_offsets, shard_block_sizes)
        ):
            # Extract the shard from loaded_weight
            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_block_offset, shard_block_size
            )

            # Calculate per-rank offset and size (considering TP)
            rank_shard_offset = shard_block_offset // self.tp_size
            rank_shard_size = shard_block_size // self.tp_size

            # Load into the parameter
            param.load_merged_column_weight(
                loaded_weight=loaded_weight_shard,
                shard_id=shard_id,
                shard_offset=rank_shard_offset,
                shard_size=rank_shard_size,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                use_presharded_weights=self.use_presharded_weights,
            )

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ):
        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            if isinstance(param, PerTensorScaleParameter):
                param.load_merged_column_weight(
                    loaded_weight=loaded_weight,
                    shard_id=0,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
                return
            elif isinstance(param, BlockQuantScaleParameter):
                self._load_merged_block_scale(param, loaded_weight)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(
                    loaded_weight=loaded_weight,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
                return
            output_sizes = (
                [self.output_sizes[idx] for idx in loaded_shard_id]
                if loaded_shard_id
                else None
            )
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(
                param, loaded_weight, output_sizes=output_sizes
            )
            return

        assert loaded_shard_id < len(self.output_sizes)

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = self.quant_method.quant_config.weight_block_size
            raw_block_n, _ = weight_block_size[0], weight_block_size[1]
            block_n = 1 if getattr(param, "format_ue8m0", False) else raw_block_n
            if (
                tp_plan_active(self.tp_size, self.tp_family)
                and self.tp_units is not None
            ):
                # Uneven TP: the units were coarsened to quant-block
                # multiples in __init__, so every rank's weight shard is
                # block_n-aligned and the scale-grid shard is exactly the
                # weight shard / block_n. The source-side offset follows
                # from the same unit distribution via the param's
                # tp_units attribute in load_merged_column_weight.
                shard_offset = (
                    sum(
                        tp_partition_size(
                            sz,
                            self.tp_size,
                            self.tp_rank,
                            self.tp_units,
                            self.tp_family,
                        )
                        for sz in self.output_sizes[:loaded_shard_id]
                    )
                    // block_n
                )
                shard_size = (
                    tp_partition_size(
                        self.output_sizes[loaded_shard_id],
                        self.tp_size,
                        self.tp_rank,
                        self.tp_units,
                        self.tp_family,
                    )
                    // block_n
                )
            else:
                shard_offset = (
                    (sum(self.output_sizes[:loaded_shard_id]) + block_n - 1) // block_n
                ) // self.tp_size
                shard_size = (
                    (self.output_sizes[loaded_shard_id] + block_n - 1)
                    // block_n
                    // self.tp_size
                )
        else:
            # Per packed output its own partition (see weight_loader).
            shard_offset = sum(
                tp_partition_size(
                    sz, self.tp_size, self.tp_rank, self.tp_units, self.tp_family
                )
                for sz in self.output_sizes[:loaded_shard_id]
            )
            shard_size = tp_partition_size(
                self.output_sizes[loaded_shard_id],
                self.tp_size,
                self.tp_rank,
                self.tp_units,
                self.tp_family,
            )

        param.load_merged_column_weight(
            loaded_weight=loaded_weight,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            use_presharded_weights=self.use_presharded_weights,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )


class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        load_presharded_attn: bool = False,
        v_head_size: Optional[int] = None,
        skip_block_quant_check: bool = False,
        q_shard_unit_count: Optional[int] = None,
        q_shard_groups: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        if tp_rank is None:
            tp_rank = get_parallel().tp_rank
        if tp_size is None:
            tp_size = get_parallel().tp_size
        self.tp_rank, self.tp_size = tp_rank, tp_size
        if tp_plan_active(tp_size):
            if self.total_num_heads % self.total_num_kv_heads != 0:
                raise ValueError(
                    f"--rank-tp-ratio: total_num_heads "
                    f"({self.total_num_heads}) must be a multiple of "
                    f"total_num_kv_heads ({self.total_num_kv_heads}) to "
                    "distribute whole GQA groups per rank."
                )
            if attn_kv_replicated(tp_size, self.total_num_kv_heads):
                # TP > num_kv_heads (task #62): REPLICATED-KV mode. Every
                # rank holds ALL kv heads (k/v projections fully replicated
                # -> identical K/V recomputed per rank, upstream-replication
                # semantics, no broadcast); the q heads split in units of
                # kv_total heads so per-rank num_qo % num_kv == 0 holds for
                # the attention kernels. `q_shard_unit_count` is the unit
                # COUNT of the q block — callers with a fused q(+gate) block
                # must pass the real q-head-based count
                # (q_real // kv_total), since the fused slot count would
                # halve the unit size.
                self.tp_units = (
                    q_shard_unit_count
                    if q_shard_unit_count is not None
                    else attn_q_partition_units(
                        self.total_num_heads, self.total_num_kv_heads, tp_size
                    )
                )
                # kv-boundary alignment (task #116): constrain the q split so
                # no rank straddles a global kv-head group (the #105 ragged
                # kernel cannot represent that). Derived from THE single source
                # attn_q_partition_groups; cross-check the caller-supplied
                # q_shard_groups (the o_proj gets the SAME value) so a future
                # model wiring the two differently fails loudly at construction
                # rather than silently mis-sharding o_proj.
                self.tp_q_groups = attn_q_partition_groups(
                    self.total_num_kv_heads, tp_size
                )
                if q_shard_groups is not None and q_shard_groups != self.tp_q_groups:
                    raise ValueError(
                        f"QKV q_shard_groups {q_shard_groups} disagrees with "
                        f"the kv-derived value {self.tp_q_groups} "
                        f"(kv={self.total_num_kv_heads}, tp={tp_size}); the "
                        "qkv q block and o_proj input must align to the same "
                        "kv-group boundaries."
                    )
                self.num_heads = tp_partition_size(
                    self.total_num_heads,
                    tp_size,
                    tp_rank,
                    self.tp_units,
                    groups=self.tp_q_groups,
                )
                self.num_kv_heads = self.total_num_kv_heads
                # Marks the "every rank loads the full k/v checkpoint
                # width" case for the qkv weight loaders (shard_id
                # tp_rank // replicas == 0, full-width narrow).
                self.num_kv_head_replicas = tp_size
                # Base-class directive: partition q by the plan, keep k/v
                # full on every rank (see ColumnParallelLinear.__init__).
                self.kv_replicated_parts = [False, True, True]
            else:
                # Uneven TP (--rank-tp-ratio): heads are distributed by the
                # shard plan with the KV heads as indivisible units, so
                # every rank owns whole GQA groups (its q heads are an
                # exact multiple of its kv heads).
                self.tp_units = self.total_num_kv_heads
                self.num_heads = tp_partition_size(
                    self.total_num_heads, tp_size, tp_rank, self.tp_units
                )
                self.num_kv_heads = tp_partition_size(
                    self.total_num_kv_heads, tp_size, tp_rank, self.tp_units
                )
                self.num_kv_head_replicas = 1
        else:
            self.tp_units = None
            self.num_heads = divide(self.total_num_heads, tp_size)
            if tp_size >= self.total_num_kv_heads:
                self.num_kv_heads = 1
                self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
            else:
                self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
                self.num_kv_head_replicas = 1
        self.q_proj_shard_size = self.num_heads * self.head_size
        self.kv_proj_shard_size = self.num_kv_heads * self.head_size
        self.v_proj_shard_size = self.num_kv_heads * self.v_head_size
        input_size = self.hidden_size
        if self.tp_units is not None:
            # Uneven TP: per-rank sizes vary, so hand the checkpoint
            # totals to the base class; it partitions them per rank via
            # the shard plan (kv-head units).
            output_size = (
                self.total_num_heads * self.head_size
                + self.total_num_kv_heads * self.head_size
                + self.total_num_kv_heads * self.v_head_size
            )
            self.output_sizes = [
                self.total_num_heads * self.head_size,  # q_proj
                self.total_num_kv_heads * self.head_size,  # k_proj
                self.total_num_kv_heads * self.v_head_size,  # v_proj
            ]
        else:
            output_size = (
                self.num_heads * self.head_size
                + self.num_kv_heads * self.head_size
                + self.num_kv_heads * self.v_head_size
            ) * tp_size
            self.output_sizes = [
                self.num_heads * self.head_size * tp_size,  # q_proj
                self.num_kv_heads * self.head_size * tp_size,  # k_proj
                self.num_kv_heads * self.v_head_size * tp_size,  # v_proj
            ]
        self.use_presharded_weights = load_presharded_attn
        quant_config = None if _disable_hip_linear_quant else quant_config

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
            use_presharded_weights=self.use_presharded_weights,
            skip_block_quant_check=skip_block_quant_check,
        )

    def _get_shard_offset_mapping(self, loaded_shard_id: str):
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + self.num_kv_heads) * self.head_size
            + self.num_kv_heads * self.v_head_size,
        }
        return shard_offset_mapping.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str):
        shard_size_mapping = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.v_head_size,
        }
        return shard_size_mapping.get(loaded_shard_id)

    def _load_fused_module_from_checkpoint(
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        """
        Handle special case for models where QKV layers are already
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, self.total_num_heads * self.head_size),
            (
                "k",
                self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
            (
                "v",
                (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                self.total_num_kv_heads * self.v_head_size,
            ),
        ]

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if (
                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )

            if not self.use_presharded_weights:
                loaded_weight_shard = loaded_weight.narrow(
                    param.output_dim, shard_offset, shard_size
                )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def _load_qkv_block_scale(
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        if tp_plan_active(self.tp_size):
            raise NotImplementedError(
                "Block-quant scale loading (fused on disk) is not supported "
                "with uneven TP (--rank-tp-ratio) yet."
            )
        block_n, _ = self.quant_method.quant_config.weight_block_size
        q_size = self.total_num_heads * self.head_size // block_n
        k_size = self.total_num_kv_heads * self.head_size // block_n
        v_size = self.total_num_kv_heads * self.v_head_size // block_n
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, q_size),
            ("k", q_size, k_size),
            ("v", q_size + k_size, v_size),
        ]
        for shard_id, shard_offset, shard_size in shard_offsets:
            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            rank_shard_offset = self._get_shard_offset_mapping(shard_id) // block_n
            rank_shard_size = self._get_shard_size_mapping(shard_id) // block_n
            param.load_qkv_weight(
                loaded_weight=loaded_weight_shard,
                num_heads=self.num_kv_head_replicas,
                shard_id=shard_id,
                shard_offset=rank_shard_offset,
                shard_size=rank_shard_size,
                tp_rank=self.tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ):
        if loaded_shard_id is None:  # special case for certain models
            if isinstance(param, PerTensorScaleParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=0)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight)
                return
            elif isinstance(param, BlockQuantScaleParameter):
                self._load_qkv_block_scale(param, loaded_weight)
                return
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = self.quant_method.quant_config.weight_block_size
            raw_block_n, _ = weight_block_size[0], weight_block_size[1]
            block_n = 1 if getattr(param, "format_ue8m0", False) else raw_block_n
            shard_offset = (shard_offset + block_n - 1) // block_n
            shard_size = (shard_size + block_n - 1) // block_n

        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=self.num_kv_head_replicas,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
            use_presharded_weights=self.use_presharded_weights,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ):

        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type and loaded_shard_id is not None:
            idx_map = {"q": 0, "k": 1, "v": 2}
            param.data[idx_map[loaded_shard_id]].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            return

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            total = loaded_weight.size(output_dim)
            if self.num_kv_head_replicas == self.tp_size and loaded_shard_id in (
                "k",
                "v",
            ):
                # REPLICATED-KV mode (TP > num_kv_heads, task #62): every
                # rank keeps the FULL k/v projection rows; only the q rows
                # are partitioned (in kv_total-sized head units).
                shard_size, start_idx = total, 0
            else:
                # Uneven TP: this rank's row (output) partition; reproduces
                # even //tp_size with no plan. Rows are whole -> no
                # quant-block concern. groups (task #116): the q rows align to
                # kv-head-group boundaries under REPLICATED-KV, matching the
                # aligned num_heads used everywhere else (None -> unchanged).
                shard_size = tp_partition_size(
                    total,
                    self.tp_size,
                    self.tp_rank,
                    self.tp_units,
                    self.tp_family,
                    self.tp_q_groups,
                )
                start_idx = tp_loaded_shard_start(
                    total,
                    self.tp_size,
                    self.tp_rank,
                    shard_size,
                    self.tp_units,
                    self.tp_family,
                    self.tp_q_groups,
                )

            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)

        # Special case for per-tensor scales in fused case.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv/mlp).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_heads * self.head_size),
                (
                    "k",
                    self.total_num_heads * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
                (
                    "v",
                    (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                    self.total_num_kv_heads * self.v_head_size,
                ),
            ]
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

            packed_dim = getattr(param, "packed_dim", None)
            if _is_cpu:
                shard_offsets = adjust_shard_offsets(
                    shard_offsets, loaded_weight, output_dim
                )

            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantized Weights.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor

                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                if use_bitsandbytes_4bit:
                    orig_qkv_offsets = {
                        "q": (0, self.total_num_heads * self.head_size),
                        "k": (
                            self.total_num_heads * self.head_size,
                            self.total_num_kv_heads * self.head_size,
                        ),
                        "v": (
                            (self.total_num_heads + self.total_num_kv_heads)
                            * self.head_size,
                            self.total_num_kv_heads * self.v_head_size,
                        ),
                        "total": (
                            (self.total_num_heads + self.total_num_kv_heads)
                            * self.head_size
                            + self.total_num_kv_heads * self.v_head_size,
                            0,
                        ),
                    }

                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_qkv_offsets, shard_id
                    )

                if not self.use_presharded_weights:
                    loaded_weight_shard = loaded_weight.narrow(
                        output_dim, shard_offset, shard_size
                    )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        # If output dim is defined, use the default loading process.
        if output_dim is not None:
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.v_head_size
            # Special case for Quantized Weights.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor

                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            if use_bitsandbytes_4bit:
                orig_qkv_offsets = {
                    "q": (0, self.num_heads * self.head_size),
                    "k": (
                        self.num_heads * self.head_size,
                        self.num_kv_heads * self.head_size,
                    ),
                    "v": (
                        (self.num_heads + self.num_kv_heads) * self.head_size,
                        self.num_kv_heads * self.v_head_size,
                    ),
                    "total": (
                        (self.num_heads + self.num_kv_heads) * self.head_size
                        + self.num_kv_heads * self.v_head_size,
                        0,
                    ),
                }
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_qkv_offsets, loaded_shard_id
                )

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            if loaded_shard_id == "q":
                shard_id = self.tp_rank
            else:
                shard_id = self.tp_rank // self.num_kv_head_replicas
            # Even TP: shard_id * shard_size (kv replication folds several
            # ranks onto the same checkpoint shard). Uneven TP: prefix-sum
            # offset by the shard plan (kv-head units; replication is
            # rejected in __init__, so shard_id == tp_rank here).
            # groups (task #116): the q block aligns to kv-head-group
            # boundaries under REPLICATED-KV; k/v are full-width so
            # tp_loaded_shard_start early-returns 0 (groups unused there).
            start_idx = tp_loaded_shard_start(
                loaded_weight.shape[output_dim],
                self.tp_size,
                shard_id,
                shard_size,
                self.tp_units,
                groups=self.tp_q_groups,
            )

            if _is_cpu:
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    param_data,
                    loaded_weight,
                    0,  # param_data_start
                    start_idx,
                    output_dim,
                    shard_size,
                    not use_bitsandbytes_4bit and not self.use_presharded_weights,
                )
            else:
                # bitsandbytes loads the weights of the specific portion
                # no need to narrow here
                if not use_bitsandbytes_4bit and not self.use_presharded_weights:
                    loaded_weight = loaded_weight.narrow(
                        output_dim, start_idx, shard_size
                    )

        # Special case for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_index = ["q", "k", "v"].index(loaded_shard_id)
            param_data = param_data.narrow(0, shard_index * shard_size, shard_size)
        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions."
                )

        assert param_data.shape == loaded_weight.shape, (
            f"{param_data.shape=} {loaded_weight.shape=}"
        )
        param_data.copy_(loaded_weight)


def _dense_deferred_eligible(module, output_parallel) -> bool:
    """#588(b): may THIS dense reduce defer to the comm stream?

    The #597 deferred issue extended to the producer-owned dense site
    (DESIGN_588_collective_floor par.4b). Four gates, every one
    group-uniform (module config, envs, tensor shape shared across TP):

    * ``defer_all_reduce_ok`` -- a PER-INSTANCE opt-in set only where the
      downstream path is TRACED to reach a ``join_deferred`` consumer
      before anything reads the values (qwen3_5: o_proj -> prepare_mlp
      joins at communicator.py:675-shape entry; down_proj ->
      postprocess_layer / next prepare_attn). A generic hook would also
      defer vision towers and draft heads whose consumers never join --
      unreduced activations, silently.
    * the #597 infrastructure flag AND the dense extension flag -- default
      off, byte-identical path otherwise.
    * dim/min-token gates identical to the MoE site.

    The communicator-owned site (communicator.py:1204-shape) is
    DELIBERATELY not in reach: this helper is consulted only where the
    call that would have reduced is the call that defers, which is the
    whole of #597's safety argument. No suppression exists anywhere.
    """
    return (
        getattr(module, "defer_all_reduce_ok", False)
        and tp_ar_deferred_enabled()
        and envs.SGLANG_TP_AR_PIPELINE_DENSE.get()
        and output_parallel.dim() == 2
        and output_parallel.shape[0]
        >= int(envs.SGLANG_TP_AR_PIPELINE_DEFERRED_MIN_TOKENS.get())
    )


class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        use_presharded_weights: bool = False,
        use_dp_attention_reduce: bool = False,
        tp_units: Optional[int] = None,
        tp_family: Optional[str] = None,
        tp_q_groups: Optional[int] = None,
    ):
        quant_config = None if _disable_hip_linear_quant else quant_config
        super().__init__(
            input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        self.use_dp_attention_reduce = use_dp_attention_reduce

        # Divide the weight matrix along the last dimension.
        if tp_rank is None:
            tp_rank = get_parallel().tp_rank
        if tp_size is None:
            tp_size = get_parallel().tp_size
        self.tp_rank, self.tp_size = tp_rank, tp_size
        # Uneven TP: unit count of the INPUT dimension (matches the paired
        # column-parallel layer's units, e.g. kv heads for o_proj) and the
        # optional family plan (must match the paired layer's family).
        # Without an installed ratio plan, tp_partition_size() reproduces
        # divide() exactly, so the default path is unchanged.
        # See ColumnParallelLinear.__init__: only coarsen units when THIS
        # layer's weights are actually quantized (ignored layers resolve to
        # UnquantizedLinearMethod and keep their raw units).
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        _layer_quant_config = (
            None
            if isinstance(self.quant_method, UnquantizedLinearMethod)
            else quant_config
        )
        self.tp_units = _quant_block_aligned_units(
            input_size, tp_units, _layer_quant_config, 1
        )
        self.tp_family = tp_family
        # kv-boundary alignment for the o_proj INPUT dim (task #116): must be
        # the SAME value the paired qkv q block used, so the per-rank q-head
        # counts agree. None on every non-attention row-parallel layer.
        self.tp_q_groups = tp_q_groups
        self.input_size_per_partition = tp_partition_size(
            input_size,
            self.tp_size,
            self.tp_rank,
            self.tp_units,
            self.tp_family,
            self.tp_q_groups,
        )
        assert self.quant_method is not None
        self.use_presharded_weights = use_presharded_weights

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if self.tp_units is not None and tp_plan_active(self.tp_size, self.tp_family):
            # Uneven TP only: expose the unit count (and family) to the
            # v2 weight loaders in layers/parameter.py (AFTER
            # create_weights, which registers the parameters). Not set on
            # the default path.
            for p in self.parameters():
                set_weight_attrs(
                    p,
                    {
                        "tp_units": self.tp_units,
                        "tp_family": self.tp_family,
                        "tp_q_groups": self.tp_q_groups,
                    },
                )

        if bias:
            self.bias = Parameter(torch.zeros(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter (row-parallel: the sharded
        # input dim of the qweight holds PACKED bytes, not elements).
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            if input_dim is not None and tp_plan_active(self.tp_size):
                # Uneven TP: shard the packed input on quant-block boundaries.
                # The byte dim holds `in_elems // ggml_block` blocks of
                # `type_size` bytes; partition in ELEMENT space (units already
                # coarsened to whole blocks by _quant_block_aligned_units via
                # GGUFConfig.weight_block_size) and map to byte offsets. The
                # ggml type is on the sibling qweight_type param, loaded first.
                import gguf as _gguf

                wtype = _gguf.GGMLQuantizationType(self.qweight_type.weight_type)
                block_size, type_size = _gguf.GGML_QUANT_SIZES[wtype]
                total_bytes = loaded_weight.shape[input_dim]
                in_features = total_bytes // type_size * block_size
                my_elems = tp_partition_size(
                    in_features,
                    self.tp_size,
                    self.tp_rank,
                    self.tp_units,
                    self.tp_family,
                    self.tp_q_groups,
                )
                elem_start = tp_loaded_shard_start(
                    in_features,
                    self.tp_size,
                    self.tp_rank,
                    my_elems,
                    self.tp_units,
                    self.tp_family,
                    self.tp_q_groups,
                )
                # my_elems / elem_start are whole-block multiples -> exact bytes
                byte_start = elem_start // block_size * type_size
                byte_size = my_elems // block_size * type_size
                weight_shape = list(loaded_weight.shape)
                weight_shape[input_dim] = byte_size
                param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)
                param.data.copy_(loaded_weight.narrow(input_dim, byte_start, byte_size))
                return
            weight_shape = list(loaded_weight.shape)
            if input_dim:
                weight_shape[input_dim] = weight_shape[input_dim] // self.tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

        param_data = param.data
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        if (
            input_dim is not None
            and not use_bitsandbytes_4bit
            and not self.use_presharded_weights
        ):
            shard_size = param_data.shape[input_dim]
            # Even TP: rank * shard_size. Uneven TP: prefix-sum offset of
            # this rank's partition in the checkpoint dimension.
            start_idx = tp_loaded_shard_start(
                loaded_weight.shape[input_dim],
                self.tp_size,
                self.tp_rank,
                shard_size,
                self.tp_units,
                self.tp_family,
                self.tp_q_groups,
            )

            if _is_cpu:
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    param_data,
                    loaded_weight,
                    0,  # param_data_start
                    start_idx,
                    input_dim,
                    shard_size,
                )
            else:
                # Padding for special case like qwen2_5_VL's mlp which is not 8-aligned
                end_idx = start_idx + shard_size
                if end_idx > loaded_weight.shape[input_dim]:
                    loaded_weight = pad_or_narrow_weight(
                        loaded_weight, input_dim, start_idx, shard_size
                    )
                else:
                    loaded_weight = loaded_weight.narrow(
                        input_dim, start_idx, shard_size
                    )

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape, (
            f"{param_data.shape=} {loaded_weight.shape=}"
        )
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        if isinstance(param, RowvLLMParameter):
            # This `BasevLLMParameter` is defined in sglang/srt/layers/parameter.py,
            # It supports additional parameters like tp_rank and use_presharded_weights.
            param.load_row_parallel_weight(
                loaded_weight,
                tp_rank=self.tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )
        else:
            # `params` is defined in `vllm/model_executor/parameter.py`,
            # It does not support additional parameters.
            # However, after QuantizedRL reload, params might still need tp_rank
            try:
                param.load_row_parallel_weight(
                    loaded_weight,
                    tp_rank=self.tp_rank,
                    use_presharded_weights=self.use_presharded_weights,
                )
            except TypeError:
                # Fallback for parameters that don't accept additional args
                param.load_row_parallel_weight(loaded_weight)

    def forward(self, input_, skip_all_reduce=False, forward_batch=None):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias

        # Task #588: token-slice pipelining of the TP all-reduce. Opt-in; when
        # SGLANG_TP_AR_PIPELINE is unset this costs one cached bool read and
        # the path below is byte-for-byte the one that ran before.
        #
        # Eligibility is deliberately narrow, and every condition is
        # group-uniform (module config, env, forward mode, token count) --
        # the slice count is part of the collective sequence, so a
        # rank-divergent condition here is a hang, not a slow path. The
        # symmetric-memory context of the default path below is skipped
        # because it is required to be DISABLED for the pipeline to engage,
        # and use_symmetric_memory(disabled=True) is a nullcontext.
        if (
            tp_ar_pipeline_enabled()
            and self.reduce_results
            and self.tp_size > 1
            and not skip_all_reduce
            and not self.use_dp_attention_reduce
            and not is_allocation_symmetric()
            and input_parallel.dim() == 2
            and not should_skip_mlp_all_reduce()
            and not _tp_ar_pipeline_quant_comms(forward_batch)
        ):
            output = pipelined_row_all_reduce(
                input_parallel,
                apply_fn=lambda chunk: self.quant_method.apply(self, chunk, bias=bias_),
                all_reduce_fn=tensor_model_parallel_all_reduce,
                max_reduce_fn=_tp_ar_pipeline_max_reduce,
                out_features=self.output_size,
            )
            return output, (self.bias if self.skip_bias_add else None)

        if self.use_dp_attention_reduce:
            symm_ctx = use_symmetric_memory(get_parallel().attn_tp_group)
        else:
            symm_ctx = use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            )
        with symm_ctx:
            output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)

        # skip_all_reduce: explicit call-site override. Also honor
        # ForwardFlags (fuse_mlp_allreduce / mlp_reduce_scatter) published by
        # the decoder — callers should not thread those flags into modules.
        if (
            self.reduce_results
            and self.tp_size > 1
            and not skip_all_reduce
            and not should_skip_mlp_all_reduce()
        ):
            if self.use_dp_attention_reduce:
                output = get_parallel().attn_tp_group.all_reduce(output_parallel)
            else:
                quantize_communications = (
                    (
                        not forward_batch.forward_mode.is_decode_or_idle()
                        and get_server_args().enable_quant_communications
                    )
                    if forward_batch is not None
                    else False
                )
                if quantize_communications:
                    output = tensor_model_parallel_quant_all_reduce(output_parallel)
                elif _dense_deferred_eligible(self, output_parallel):
                    # #588(b): same single reduction, issued on the comm
                    # stream and joined by the first communicator consumer.
                    # Safety per instance: see _dense_deferred_eligible.
                    output = issue_deferred_all_reduce(
                        output_parallel, tensor_model_parallel_all_reduce
                    )
                else:
                    output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            # barlink async overlap (SGLANG_BARLINK_UCX_OVERLAP=1): when this AR
            # was skipped because the NEXT layer's prepare_attn absorbs it
            # (fuse_mlp_allreduce), issue it asynchronously HERE so the wire
            # crossing runs under the intervening host work. The handle
            # rides on the tensor exactly like _sglang_needs_allreduce_fusion
            # does; prepare_attn completes it. All conditions are
            # group-uniform (env flag, transport class, published forward
            # flags, shapes shared across the TP group). A None handle means
            # the async issue is unavailable -- prepare_attn then falls back
            # to its unchanged sync all-reduce.
            if (
                self.reduce_results
                and self.tp_size > 1
                and not skip_all_reduce
                and get_forward().fuse_mlp_allreduce
                and output_parallel.numel() > 0
            ):
                _comm = barlink_mlp_ar_overlap_comm()
                if _comm is not None:
                    _h = _comm.all_reduce_async(output_parallel)
                    if _h is not None:
                        output_parallel._barlink_ar_handle = (_comm, _h)
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias

    def extra_repr(self) -> str:
        s = f"input_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s


class MergedColumnParallelRepeatedLinear(LinearBase):
    """Merged column parallel linear and repeated linear layer.

    TODO: quantization is not supported yet.
    Args:
        input_size: input dimension of the linear layer.
        column_output_sizes: output dimension of the column linear layers.
        repeated_output_sizes: output dimension of the repeated linear layers.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        input_size: int,
        column_output_sizes: List[int],
        repeated_output_sizes: List[int],
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        output_size = sum(column_output_sizes) + sum(repeated_output_sizes)
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.num_column_parallel = len(column_output_sizes)
        self.tp_rank = get_parallel().tp_rank
        self.tp_size = get_parallel().tp_size

        self.output_partition_sizes = [
            divide(x, self.tp_size) for x in column_output_sizes
        ] + repeated_output_sizes
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            skip_block_quant_check=True,
            weight_loader=self.weight_loader,
        )

        self.prefix = prefix

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, input_)

    def weight_loader(
        self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ) -> torch.Tensor:
        output_dim = param.output_dim
        shard_offset = sum(self.output_partition_sizes[:loaded_shard_id])
        shard_size = self.output_partition_sizes[loaded_shard_id]
        param_data = param.data.narrow(output_dim, shard_offset, shard_size)

        if loaded_shard_id < self.num_column_parallel:
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        param_data.copy_(loaded_weight)


class ColumnParallelBatchedLinear(nn.Module):
    """Column parallel batched linear layer.

    TODO: quantization is not supported yet.
    Args:
        batch: batch dimension of the linear layer.
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        dtype: Data type for the parameters.
    """

    def __init__(
        self, batch: int, input_size: int, output_size: int, dtype: torch.dtype
    ):
        super().__init__()
        self.tp_rank = get_parallel().tp_rank
        self.tp_size = get_parallel().tp_size
        self.weight = nn.Parameter(
            torch.empty(batch, output_size // self.tp_size, input_size, dtype=dtype),
            requires_grad=False,
        )
        setattr(self.weight, "weight_loader", self.weight_loader)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.bmm(input, self.weight.transpose(-1, -2))

    def weight_loader(
        self, param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ) -> torch.Tensor:
        shard_size = self.weight.shape[-2]
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param.data[loaded_shard_id].copy_(loaded_weight)
