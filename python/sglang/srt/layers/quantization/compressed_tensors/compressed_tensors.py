# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import logging
import math
from contextlib import suppress
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    cast,
)

import torch
from compressed_tensors.config import (
    CompressionFormat,
    SparsityCompressionConfig,
    SparsityStructure,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from pydantic import BaseModel

from sglang.srt.layers.moe import MoeRunnerConfig, get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    WNA16_SUPPORTED_BITS,
    CompressedTensorsLinearScheme,
    CompressedTensorsMoEScheme,
    CompressedTensorsMxInt4MoE,
    CompressedTensorsW4A4Fp4,
    CompressedTensorsW4A4Fp4Dequant,
    CompressedTensorsW4A4Nvfp4MoE,
    CompressedTensorsW8A8Fp8,
    CompressedTensorsW8A8Fp8MoE,
    CompressedTensorsW8A8Int8,
    CompressedTensorsW8A16Fp8,
    CompressedTensorsWNA16,
    CompressedTensorsWNA16MoE,
    CompressedTensorsWNA16TritonMoE,
    NPUCompressedTensorsW4A8Int8DynamicMoE,
    NPUCompressedTensorsW4A16Int4DynamicMoE,
    NPUCompressedTensorsW8A8Int8,
    NPUCompressedTensorsW8A8Int8DynamicMoE,
    nvfp4_unpackable_reason,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    is_activation_quantization_format,
    should_ignore_layer,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu

_is_cuda = is_cuda()
_is_npu = is_npu()
_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.models.utils import WeightsMapper

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsLinearMethod"]

SPARSITY_CONFIG_NAME: Literal["sparsity_config"] = "sparsity_config"
QUANTIZATION_SCHEME_MAP_TYPE = Dict[str, Optional[Dict[str, QuantizationArgs]]]


class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """
        Express device capability as an integer ``<major><minor>``.

        It is assumed that the minor version is always a single digit.
        """
        assert 0 <= self.minor < 10
        return self.major * 10 + self.minor


class CompressedTensorsConfig(QuantizationConfig):
    # Two of its schemes repack through marlin:
    # ``CompressedTensorsW8A16Fp8`` (``prepare_fp8_layer_for_marlin``) and
    # ``CompressedTensorsWNA16`` (``gptq_marlin_repack``).
    marlin_packable_linear = True

    def __init__(
        self,
        target_scheme_map: Dict[str, Any],
        ignore: List[str],
        quant_format: str,
        sparsity_scheme_map: Dict[str, SparsityCompressionConfig],
        sparsity_ignore_list: List[str],
        kv_cache_scheme: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        linear_fp8_config: Optional[Any] = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.quant_format = quant_format
        # Map from [target -> scheme]
        self.target_scheme_map = target_scheme_map
        self.kv_cache_scheme = kv_cache_scheme
        self.sparsity_scheme_map = sparsity_scheme_map
        self.sparsity_ignore_list = sparsity_ignore_list
        self.config = config
        self.packed_modules_mapping = packed_modules_mapping or {}
        self.linear_fp8_config = linear_fp8_config

    @staticmethod
    def _group_size_block(target_scheme_map: Dict[str, Any]) -> Optional[List[int]]:
        """Uneven-TP group alignment for group-quantized schemes (AWQ/GPTQ INT4,
        group_size G).

        Such a scheme groups `group_size` input channels under one scale, so
        every per-rank shard boundary must be a multiple of G along the grouped
        dim — exactly like block-quantized FP8/GGUF. The uneven-TP machinery
        (`_quant_block_aligned_units` / `tp_partition_size` /
        `fused_moe_triton.layer`) already coarsens a split to
        `quant_config.weight_block_size`; a group-quant config simply has to
        expose G so that coarsening fires. Without it, an uneven
        `--rank-tp-ratio` split produces `input_size_per_partition % group_size
        != 0` and the wNa16 scheme asserts (this is what blocked AWQ under
        uneven TP; even TP happened to divide cleanly).

        Both dims are set to the same value so a column-parallel OUTPUT split
        (gate_up / w1) lands on the same boundary as its coupled row-parallel
        INPUT split (down / w2) — mirroring GGUFConfig.weight_block_size =
        [256, 256]. A single shared block is mandatory for the coupled MLP
        pair: gate_up's output units and down's input units partition the SAME
        intermediate dimension, so they must coarsen identically or the
        per-rank shards diverge.

        The block is NOT simply the group size. These group-quantized schemes
        execute through the Marlin (wNa16) GEMM, which additionally requires
        the partitioned K dim to be a multiple of its thread tile
        (min_thread_k = 128). A group-only multiple (e.g. 32) can still land on
        a K that is group-valid but NOT tile-valid under an uneven
        --rank-tp-ratio split (e.g. down_proj K=4768 = 149*32, 4768 % 128 = 32)
        and then crash inside the kernel with "Invalid thread config" even
        though the group constraint is satisfied. So coarsen to the lcm of the
        group size(s) AND the Marlin K tile — every uneven shard is then
        simultaneously group-aligned and tile-aligned. (min_thread_k=128 also
        dominates min_thread_n=64, so the shared value keeps the column-parallel
        OUTPUT dim tile-valid too.) Dense / ignored layers carry no scheme here
        and are unaffected: _quant_block_aligned_units only re-expresses a unit
        family that is not already block-aligned, so a component whose units
        already divide the block (e.g. GDN k-head units) passes through
        unchanged.

        Returns None when no scheme is group-quantized (channel/tensor/fp8 →
        the existing even-only path is unchanged and untouched).
        """
        group_sizes = set()
        for scheme in (target_scheme_map or {}).values():
            weights = scheme.get("weights") if isinstance(scheme, dict) else None
            g = getattr(weights, "group_size", None)
            if g and int(g) > 0:
                group_sizes.add(int(g))
        if not group_sizes:
            return None
        g = math.lcm(*group_sizes)  # a shard that is a multiple of the lcm is a
        # multiple of every layer's group_size; fold in the Marlin K tile so it
        # is also a valid GEMM thread-tile boundary under uneven TP.
        try:
            from sglang.srt.layers.quantization.marlin_utils import (
                GPTQ_MARLIN_MIN_THREAD_K,
            )

            marlin_k_tile = int(GPTQ_MARLIN_MIN_THREAD_K)
        except Exception:
            marlin_k_tile = 128
        g = math.lcm(g, marlin_k_tile)
        return [g, g]

    @staticmethod
    def _int8_w8a8_block(target_scheme_map: Dict[str, Any]) -> Optional[List[int]]:
        """Uneven-TP alignment block for a dynamic/static-activation INT8 W8A8
        scheme (``CompressedTensorsW8A8Int8`` -> ``sgl_kernel.int8_scaled_mm``).

        These schemes are per-CHANNEL or per-TENSOR: no ``block_structure``, no
        ``group_size``, so both branches above return ``None`` and the uneven-TP
        coarsening never fires -- the #353 gap. What still constrains the split
        is the CUTLASS kernel's own 16 (K) / 8 (N) alignment; see
        :func:`~sglang.srt.layers.quantization.w8a8_int8.int8_w8a8_uneven_tp_block`
        for why that is expressed as one shared block and why it is not a
        quantization block.

        Detection requires an INT weight AND an 8-bit activation block, so that
        weight-only INT8 (``pack-quantized`` W8A16, which runs through Marlin
        wNa16 and is already served by ``_group_size_block``) does not pick up
        the wrong kernel's alignment. Returns ``None`` when no scheme is INT8
        W8A8, leaving every other format's path untouched."""
        try:
            from compressed_tensors.quantization import QuantizationType

            int_type = QuantizationType.INT.value
        except Exception:  # pragma: no cover - library shape guard
            int_type = "int"
        for scheme in (target_scheme_map or {}).values():
            if not isinstance(scheme, dict):
                continue
            weights = scheme.get("weights")
            acts = scheme.get("input_activations")
            if weights is None or acts is None:
                continue
            wtype = getattr(weights, "type", None)
            wtype = getattr(wtype, "value", wtype)
            if str(wtype).lower() != str(int_type).lower():
                continue
            if getattr(weights, "num_bits", None) != 8:
                continue
            if getattr(acts, "num_bits", None) != 8:
                continue
            from sglang.srt.layers.quantization.w8a8_int8 import (
                int8_w8a8_uneven_tp_block,
            )

            return int8_w8a8_uneven_tp_block()
        return None

    def get_linear_method(self) -> CompressedTensorsLinearMethod:
        return CompressedTensorsLinearMethod(self)

    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_name(self) -> str:
        return "compressed_tensors"

    def get_scaled_act_names(self) -> List[str]:
        return []

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: WeightsMapper):
        self.target_scheme_map = hf_to_sglang_mapper.apply_dict(self.target_scheme_map)
        self.ignore = hf_to_sglang_mapper.apply_list(self.ignore)
        self.sparsity_scheme_map = hf_to_sglang_mapper.apply_dict(
            self.sparsity_scheme_map
        )
        self.sparsity_ignore_list = hf_to_sglang_mapper.apply_list(
            self.sparsity_ignore_list
        )
        if self.kv_cache_scheme is not None:
            self.kv_cache_scheme = hf_to_sglang_mapper.apply_dict(self.kv_cache_scheme)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedEmbeddingMethod,
        )

        if isinstance(layer, LinearBase):
            # If linear_fp8_config is set, use FP8 for linear layers
            # This allows mixed quantization: experts with int4, linear layers with fp8
            if self.linear_fp8_config is not None:
                return Fp8LinearMethod(self.linear_fp8_config)
            scheme = self.get_linear_scheme(layer=layer, layer_name=prefix)
            if scheme is None:
                return UnquantizedLinearMethod()
            layer.scheme = scheme
            return CompressedTensorsLinearMethod(self)
        # #727: a vocab matrix the checkpoint actually quantized. Gated on the
        # checkpoint's own ignore list, so every checkpoint that still excludes
        # embed_tokens -- which is all of ours today -- falls through to the
        # dense path exactly as before. Checked BEFORE FusedMoE only because a
        # VocabParallelEmbedding is neither a LinearBase nor a FusedMoE and
        # would otherwise reach the end and get nothing.
        from sglang.srt.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        if isinstance(layer, VocabParallelEmbedding):
            from sglang.srt.layers.quantization.compressed_tensors.ct_embedding import (
                CompressedTensorsEmbeddingMethod,
                vocab_is_quantized,
            )

            if not vocab_is_quantized(getattr(self, "config", None) or {}, prefix):
                return UnquantizedEmbeddingMethod()
            return CompressedTensorsEmbeddingMethod()

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            layer.scheme = self.get_moe_scheme(layer=layer, layer_name=prefix)
            if layer.scheme is None:  # ignored layer
                use_triton_kernels = get_moe_runner_backend().is_triton_kernels()
                use_flashinfer_trtllm_moe = (
                    get_moe_runner_backend().is_flashinfer_trtllm()
                )
                use_deep_gemm = get_moe_runner_backend().is_deep_gemm()
                return UnquantizedFusedMoEMethod(
                    use_triton_kernels, use_flashinfer_trtllm_moe, use_deep_gemm
                )
            return CompressedTensorsFusedMoEMethod(self)
        return None

    def _add_fused_moe_to_target_scheme_map(self):
        """
        Helper function to update target_scheme_map
        since linear layers get fused into FusedMoE
        targeting 'Linear' needs to also match
        FusedMoE modules.
        """
        if (
            "Linear" not in self.target_scheme_map
            or "FusedMoE" in self.target_scheme_map
        ):
            return
        self.target_scheme_map["FusedMoE"] = self.target_scheme_map["Linear"]
        self.target_scheme_map["DeepEPMoE"] = self.target_scheme_map["Linear"]

    @property
    def weight_block_size(self) -> Optional[List[int]]:
        """Get the weight block size from the quantization config.

        Block-structured schemes (FP8) expose an explicit block_structure.
        Group-quantized schemes (AWQ/GPTQ INT4) instead carry a group_size,
        which serves the same uneven-TP alignment role — fall back to it so a
        `--rank-tp-ratio` split coarsens per-rank shards to group-multiples
        (see _group_size_block). INT8 W8A8 (channel/tensor strategy) has
        neither, but its GEMM still constrains the split — see
        _int8_w8a8_block.

        A checkpoint that mixes a group-quantized and an INT8 W8A8 scheme must
        satisfy both, so the two blocks are folded with lcm rather than one
        winning."""
        if "Linear" in self.target_scheme_map:
            weights_config = self.target_scheme_map["Linear"].get("weights")
            block_structure = getattr(weights_config, "block_structure", None)
            if block_structure:
                return block_structure
        group_block = self._group_size_block(self.target_scheme_map)
        int8_block = self._int8_w8a8_block(self.target_scheme_map)
        if group_block and int8_block:
            return [math.lcm(g, i) for g, i in zip(group_block, int8_block)]
        return group_block or int8_block

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> CompressedTensorsConfig:
        ignore: List[str] = cast(List[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))
        target_scheme_map = cls._quantization_scheme_map_from_config(config=config)
        sparsity_scheme_map, sparsity_ignore_list = cls._parse_sparsity_config(
            config=config
        )
        packed_modules_mapping = config.get("packed_modules_mapping", {})

        # Parse linear_fp8_config if present (for mixed quantization scenarios)
        # Format: {"activation_scheme": "dynamic", "fmt": "e4m3",
        #          "quant_method": "fp8", "weight_block_size": [128, 128]}
        linear_fp8_config = None
        if "linear_fp8_config" in config:
            from sglang.srt.layers.quantization.fp8 import Fp8Config

            fp8_cfg = config["linear_fp8_config"]
            # Check if it's fp8 format based on quant_method field
            is_fp8 = fp8_cfg.get("quant_method") == "fp8"
            linear_fp8_config = Fp8Config(
                is_checkpoint_fp8_serialized=is_fp8,
                activation_scheme=fp8_cfg.get("activation_scheme", "dynamic"),
                ignored_layers=fp8_cfg.get("ignored_layers"),
                weight_block_size=fp8_cfg.get("weight_block_size"),
            )

        return cls(
            target_scheme_map=target_scheme_map,
            ignore=ignore,
            quant_format=quant_format,
            sparsity_scheme_map=sparsity_scheme_map,
            sparsity_ignore_list=sparsity_ignore_list,
            config=config,
            packed_modules_mapping=packed_modules_mapping,
            linear_fp8_config=linear_fp8_config,
        )

    @classmethod
    def _parse_sparsity_config(
        cls, config: Dict[str, Any]
    ) -> Tuple[Dict[str, SparsityCompressionConfig], List[str]]:
        """
        :param config: The `quantization_config` dictionary from config.json
        :return: A tuple with two elements
            1. A dictionary mapping target layer names to their corresponding
                sparsity_config
            2. A list of layer names to ignore for sparsity
        """
        if not (sparsity_config := config.get(SPARSITY_CONFIG_NAME)):
            return dict(), []

        sparsity_config = SparsityCompressionConfig.model_validate(sparsity_config)
        sparse_scheme_map: Dict[str, SparsityCompressionConfig] = {
            target: sparsity_config for target in sparsity_config.targets or list()
        }
        sparsity_ignore_list = sparsity_config.ignore or list()
        return sparse_scheme_map, sparsity_ignore_list

    @classmethod
    def _quantization_scheme_map_from_config(
        cls, config: Dict[str, Any]
    ) -> QUANTIZATION_SCHEME_MAP_TYPE:
        """
        :param config: The `quantization_config` dictionary from config.json
        :return: A dictionary mapping target layer names to their corresponding
            quantization_args for weights and input activations
        """
        target_scheme_map: Dict[str, Any] = dict()
        quant_format = cast(str, config.get("format"))

        # The quant_config has multiple config_groups, each containing
        # an input_activations key with details about how the activations are
        # quantized, a weights key indicating how the weights are quantized,
        # and a list of targets under the `targets` key, dictating which
        # layers are impacted by the quantization details. The quantization
        # details follow the structure defined by the QuantizationArgs
        # pydantic model, which is used to verify the structure of the
        # quant_config and also store the details for later use.

        config_groups = config.get("config_groups", dict())
        for _, quant_config in config_groups.items():
            targets = quant_config.get("targets")
            for target in targets:
                target_scheme_map[target] = {}
                target_scheme_map[target]["weights"] = QuantizationArgs.model_validate(
                    quant_config.get("weights")
                )

                target_scheme_map[target]["input_activations"] = None
                if is_activation_quantization_format(quant_format):
                    input_activations = quant_config.get("input_activations")
                    # The only case where we have activation quant supported
                    # but no input_activations provided in the config
                    # should be w8a16fp8 w8a16fp8 can also run for cases where
                    # there is an input_quant but it is ignored
                    if not input_activations:
                        assert (
                            target_scheme_map[target]["weights"].type
                            == QuantizationType.FLOAT
                        )
                    else:
                        target_scheme_map[target]["input_activations"] = (
                            QuantizationArgs.model_validate(  # noqa: E501
                                quant_config.get("input_activations")
                            )
                        )
        return target_scheme_map

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    def _check_scheme_supported(self, min_capability: int, error: bool = True) -> bool:
        """Is this scheme's device kernel available here?

        The ``min_capability`` values in this file are NVIDIA compute
        capabilities, so they may only be compared against a capability read
        in the NVIDIA namespace. ``torch.cuda.get_device_capability()`` answers
        in whichever namespace the build belongs to, and the two collide:
        gfx900 reports ``(9, 0)``, the same integer as Hopper.

        Comparing an AMD arch against a CUDA threshold is therefore not merely
        imprecise, it fails in the DANGEROUS direction: an 8 GB Vega 64 sails
        through an sm89 fp8 gate and dies later on a missing kernel, instead of
        being told no at startup. So: vendor first, capability second, each in
        its own namespace.

        On ROCm there is no meaningful CUDA capability to compare, so the
        question is asked functionally instead -- does a native fp8 GEMM exist
        on this device? That keeps MI300-class parts (which have one) working
        exactly as before, while gfx900 and friends get a clean "no" and route
        to the dequant fallback.
        """
        # Imported lazily: fp8_utils pulls in the kernel modules, and this file
        # is imported while the quantization registry is still being built.
        from sglang.srt.layers.quantization.fp8_utils import (
            fp8_native_gemm_available,
        )

        if torch.version.hip is not None:
            # AMD: the CUDA threshold does not apply in this namespace.
            supported = fp8_native_gemm_available()
            if error and not supported:
                raise RuntimeError(
                    "Quantization scheme is not supported on this AMD GPU: no "
                    "native fp8 GEMM. (Note the capability reported here, "
                    f"{torch.cuda.get_device_capability()}, is an AMD arch and "
                    f"is NOT comparable to the CUDA min capability "
                    f"{min_capability}.)"
                )
            return supported

        capability_tuple = DeviceCapability(*torch.cuda.get_device_capability())

        if capability_tuple is not None:
            capability = capability_tuple.to_int()
            supported = capability >= min_capability
            if error and not supported:
                raise RuntimeError(
                    "Quantization scheme is not supported for ",
                    f"the current GPU. Min capability: {min_capability}. ",
                    f"Current capability: {capability}.",
                )
            return supported
        else:
            return False

    def _is_dynamic_token_w4a8(
        self, weight_quant: BaseModel, input_quant: BaseModel
    ) -> bool:
        is_weight_4_bits = weight_quant.num_bits == 4
        is_activation_8_bits = input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.GROUP.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        )
        is_token = (
            weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value
        )
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        return (
            is_weight_4_bits
            and is_activation_8_bits
            and is_token
            and weight_quant.symmetric
            and is_dynamic
        )

    def _is_static_tensor_w8a8(
        self, weight_quant: BaseModel, input_quant: BaseModel
    ) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        )
        is_tensor = (
            weight_strategy
            and input_quant.strategy == QuantizationStrategy.TENSOR.value
        )
        is_static = not weight_quant.dynamic and not input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_tensor and weight_quant.symmetric and is_static

    def _is_dynamic_token_w8a8(
        self, weight_quant: BaseModel, input_quant: BaseModel
    ) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
        )
        is_token = (
            weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value
        )
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_token and weight_quant.symmetric and is_dynamic

    def _is_fp8_w8a8(
        self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        # Confirm weights and activations quantized.
        if weight_quant is None or input_quant is None:
            return False

        # Confirm weight scheme is supported.
        is_floating_point = (
            weight_quant.type == QuantizationType.FLOAT
            and input_quant.type == QuantizationType.FLOAT
        )
        is_symmetric_weight = weight_quant.symmetric
        is_static_weight = not weight_quant.dynamic
        is_tensor_or_channel_or_block_weight = weight_quant.strategy in [
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
            QuantizationStrategy.BLOCK,
        ]
        if not (
            is_floating_point
            and is_symmetric_weight
            and is_static_weight
            and is_tensor_or_channel_or_block_weight
        ):
            return False

        # Dynamic quantization is always supported if weights supported.
        if input_quant.dynamic:
            return True

        # Confirm activation scheme is supported.
        is_symmetric_activation = input_quant.symmetric
        is_per_tensor_activation = input_quant.strategy == QuantizationStrategy.TENSOR
        return is_symmetric_activation and is_per_tensor_activation

    def _is_fp8_w8a16(self, weight_quant: BaseModel, input_quant: BaseModel) -> bool:
        # Confirm weights quantized.
        if weight_quant is None:
            return False

        # Confirm we have floating points.
        if weight_quant.type != QuantizationType.FLOAT:
            return False

        # Confirm weight scheme is supported.
        is_symmetric_weight = weight_quant.symmetric
        is_static_weight = not weight_quant.dynamic
        is_per_tensor_or_channel_weight = weight_quant.strategy in [
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
        ]
        if not (
            is_symmetric_weight
            and is_static_weight  # noqa: SIM103
            and is_per_tensor_or_channel_weight
        ):
            return False

        # All conditions satisfied.
        return True

    def _is_fp4a4_nvfp4(
        self, weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ):
        if weight_quant is None or input_quant is None:
            return False

        is_tensor_group_quant = (
            weight_quant.strategy == QuantizationStrategy.TENSOR_GROUP.value
            and input_quant.strategy == QuantizationStrategy.TENSOR_GROUP.value
        )
        is_symmetric = weight_quant.symmetric and input_quant.symmetric

        is_group_size_16 = (
            weight_quant.group_size == 16 and input_quant.group_size == 16
        )
        is_float_type = (
            weight_quant.type == QuantizationType.FLOAT
            and input_quant.type == QuantizationType.FLOAT
        )
        is_4_bits = weight_quant.num_bits == 4 and input_quant.num_bits == 4

        return (
            is_tensor_group_quant
            and is_float_type
            and is_4_bits
            and is_group_size_16
            and is_symmetric
        )

    def _is_wNa16_group_channel(
        self, weight_quant: BaseModel, input_quant: BaseModel
    ) -> bool:
        input_quant_none = input_quant is None
        is_channel_group = (
            weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value
        )
        is_static = not weight_quant.dynamic

        # Both symmetric and asymmetric weight quant are handled by
        # CompressedTensorsWNA16 via the Marlin kernel path; asymmetric
        # checkpoints carry a weight zero-point.
        return is_channel_group and input_quant_none and is_static

    def _is_mxint4a16(self, weight_quant: BaseModel, input_quant: BaseModel) -> bool:
        input_quant_none = input_quant is None
        is_symmetric = weight_quant.symmetric
        is_mxint4 = (
            weight_quant.num_bits == 4
            and weight_quant.type == QuantizationType.INT
            and weight_quant.strategy == QuantizationStrategy.GROUP.value
            and weight_quant.group_size == 32
        )
        is_static = not weight_quant.dynamic

        return is_mxint4 and input_quant_none and is_symmetric and is_static

    def _is_dynamic_token_w4(
        self, weight_quant: BaseModel, input_quant: BaseModel
    ) -> bool:
        is_w4 = weight_quant.num_bits == 4
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value
        )
        if input_quant is not None:
            is_token = (
                weight_strategy
                and input_quant.strategy == QuantizationStrategy.TOKEN.value
            )
            is_dynamic = not weight_quant.dynamic and input_quant.dynamic
        else:
            is_token = weight_strategy
            is_dynamic = not weight_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_w4 and weight_quant.symmetric and is_token and is_dynamic

    def _get_scheme_from_parts(
        self, weight_quant: BaseModel, input_quant: BaseModel
    ) -> CompressedTensorsLinearScheme:

        # Detect If Mixed Precision
        if self._is_wNa16_group_channel(weight_quant, input_quant):
            if (
                self.quant_format == CompressionFormat.pack_quantized.value
                and weight_quant.num_bits in WNA16_SUPPORTED_BITS
            ):
                return CompressedTensorsWNA16(
                    num_bits=weight_quant.num_bits,
                    strategy=weight_quant.strategy,
                    group_size=weight_quant.group_size,
                    symmetric=weight_quant.symmetric,
                    actorder=weight_quant.actorder,
                )
            else:
                raise ImportError(
                    "Other method (CompressedTensorsW4A16Sparse24) is not supported now"
                )

        if is_activation_quantization_format(self.quant_format):
            if self._is_fp4a4_nvfp4(weight_quant, input_quant):
                is_fp4a4_nvfp4_supported = self._check_scheme_supported(
                    CompressedTensorsW4A4Fp4.get_min_capability(), error=False
                )
                if is_fp4a4_nvfp4_supported:
                    return CompressedTensorsW4A4Fp4()
                else:
                    # The floor is Marlin's SM80, not Blackwell's SM100
                    # (#291-S3): from SM80 up the scheme has a lane, either
                    # native FP4 or in-kernel E2M1 dequant. Below it there is
                    # no NVFP4 kernel at all.
                    raise NotImplementedError(
                        "Current platform does not support nvfp4 quantization: "
                        "the compressed-tensors NVFP4 scheme needs compute "
                        f"capability >= {CompressedTensorsW4A4Fp4.get_min_capability()} "
                        "(SM80+ runs it through Marlin, SM100+ natively)."
                    )

            if self._is_fp8_w8a8(weight_quant, input_quant):
                is_fp8_w8a8_supported = self._check_scheme_supported(
                    CompressedTensorsW8A8Fp8.get_min_capability(), error=False
                )
                if is_fp8_w8a8_supported:
                    return CompressedTensorsW8A8Fp8(
                        weight_quant=weight_quant,
                        is_static_input_scheme=(
                            input_quant and not input_quant.dynamic
                        ),
                    )
                else:
                    # note: input_quant will be present for converted models;
                    # will be ignored during inference post loading
                    return CompressedTensorsW8A16Fp8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=not input_quant.dynamic,
                    )

            # note: input_quant can be None
            if self._is_fp8_w8a16(weight_quant, input_quant):
                is_static_input_scheme = input_quant and not input_quant.dynamic
                return CompressedTensorsW8A16Fp8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=is_static_input_scheme,
                )

            if self._is_static_tensor_w8a8(weight_quant, input_quant):
                if not _is_npu:
                    return CompressedTensorsW8A8Int8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=True,
                        input_symmetric=input_quant.symmetric,
                    )
                else:
                    return NPUCompressedTensorsW8A8Int8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=True,
                        input_symmetric=input_quant.symmetric,
                    )

            if self._is_dynamic_token_w8a8(weight_quant, input_quant):
                if not _is_npu:
                    return CompressedTensorsW8A8Int8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=False,
                        input_symmetric=input_quant.symmetric,
                    )
                else:
                    return NPUCompressedTensorsW8A8Int8(
                        strategy=weight_quant.strategy,
                        is_static_input_scheme=False,
                        input_symmetric=input_quant.symmetric,
                    )

        raise NotImplementedError("No compressed-tensors compatible scheme was found.")

    def get_moe_scheme(
        self, layer: torch.nn.Module, layer_name: Optional[str] = None
    ) -> Optional[CompressedTensorsMoEScheme]:
        """
        compressed-tensors supports non uniform in the following way:

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsMoEScheme used for infernece.
        """

        # FusedMoE was made by combining multiple Linears so need to
        # make sure quantization config for Linear can target it
        self._add_fused_moe_to_target_scheme_map()
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
        ]
        # TODO: refactor this to use expert_mapping and check all layer numbers
        all_scheme_dicts = [self.get_scheme_dict(layer, name) for name in unfused_names]
        scheme_dict = all_scheme_dicts[0] if all_scheme_dicts else None

        # multiple schemes found
        if not all(d == scheme_dict for d in all_scheme_dicts):
            raise ValueError(
                "All MoE projections need to have same "
                "quantization scheme but found multiple"
            )

        if scheme_dict is None:  # ignored layer
            return None

        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")

        if self._is_wNa16_group_channel(weight_quant, input_quant):
            if not _is_npu:
                if (
                    self._is_mxint4a16(weight_quant, input_quant)
                    and get_moe_runner_backend().is_flashinfer_trtllm()
                ):
                    logger.info_once(
                        "Using CompressedTensorsMxInt4MoE with flashinfer_trtllm backend"
                    )
                    return CompressedTensorsMxInt4MoE(self, weight_quant=weight_quant)
                elif _is_hip:
                    logger.info_once("Using CompressedTensorsWNA16TritonMoE (ROCm)")
                    return CompressedTensorsWNA16TritonMoE(
                        self, weight_quant=weight_quant
                    )
                else:
                    moe_backend = get_moe_runner_backend()
                    if moe_backend.is_triton():
                        logger.info_once(
                            "Using CompressedTensorsWNA16TritonMoE "
                            "(moe_runner_backend=triton)"
                        )
                        return CompressedTensorsWNA16TritonMoE(
                            self, weight_quant=weight_quant
                        )
                    logger.info_once("Using CompressedTensorsWNA16MarlinMoEMethod")
                    return CompressedTensorsWNA16MoE(self, weight_quant=weight_quant)
            else:
                if (
                    self._is_dynamic_token_w4(weight_quant, input_quant)
                    and input_quant is None
                ):
                    logger.info_once("Using NPUCompressedTensorsW4A16Int4DynamicMoE")
                    return NPUCompressedTensorsW4A16Int4DynamicMoE(self)
        elif self._is_fp4a4_nvfp4(weight_quant, input_quant):
            logger.info_once("Using CompressedTensorsW4A4Nvfp4MoE")
            return CompressedTensorsW4A4Nvfp4MoE()
        elif self._is_fp8_w8a8(weight_quant, input_quant):
            logger.info_once("Using CompressedTensorsW8A8Fp8MoE")
            return CompressedTensorsW8A8Fp8MoE(weight_quant, input_quant)
        elif self._is_dynamic_token_w8a8(weight_quant, input_quant):
            if _is_npu:
                logger.info_once("Using NPUCompressedTensorsW8A8Int8DynamicMoE")
                return NPUCompressedTensorsW8A8Int8DynamicMoE(weight_quant, input_quant)
            else:
                raise NotImplementedError(
                    f"The W8A8Int8 Fused MoE scheme is implemented only for NPU for now."
                )
        elif self._is_dynamic_token_w4a8(weight_quant, input_quant):
            if _is_npu:
                logger.info_once("Using NPUCompressedTensorsW4A8Int8DynamicMoE")
                return NPUCompressedTensorsW4A8Int8DynamicMoE(self)
            else:
                raise NotImplementedError(
                    f"The W4A8Int8 Fused MoE scheme is implemented only for NPU for now."
                )
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )

    def get_linear_scheme(
        self, layer: torch.nn.Module, layer_name: Optional[str] = None
    ) -> Optional[CompressedTensorsLinearScheme]:
        """
        compressed-tensors supports non uniform in the following way:

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for infernece.
        """

        # Find the "target" in the compressed-tensors config
        # that our layer conforms to.
        # TODO : add compressed-tensors as dep
        # so we do not have to re-write these functions
        # need to make accelerate optional in ct to do this

        # Use the new get_scheme_dict method to extract QuantizationArgs
        scheme_dict = self.get_scheme_dict(layer, layer_name)
        weight_quant = None
        input_quant = None
        if scheme_dict:
            weight_quant = scheme_dict.get("weights")
            input_quant = scheme_dict.get("input_activations")

        # Find the sparsity scheme of the layer
        # assume that fused layers inerhit first component's sparsity scheme
        sparsity_targets = self.sparsity_scheme_map.keys() - set(
            self.sparsity_ignore_list
        )
        sparsity_scheme: Optional[SparsityCompressionConfig] = None
        with suppress(ValueError):
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=sparsity_targets,
                fused_mapping=self.packed_modules_mapping,
            )
            sparsity_scheme = self.sparsity_scheme_map[matched_target]

        if self.supports_cutlass_24(
            weight_quant=weight_quant,
            input_quant=input_quant,
            sparsity_scheme=sparsity_scheme,
        ):
            raise ImportError("CompressedTensors24 is not supported now")
        elif weight_quant is None:
            logger.warning_once(
                "Acceleration for non-quantized schemes is "
                "not supported by Compressed Tensors. "
                "Falling back to UnquantizedLinearMethod"
            )
            return None

        else:
            # Find the quant_scheme
            scheme = self._get_scheme_from_parts(  # type: ignore
                weight_quant=weight_quant,
                input_quant=input_quant,
            )
            scheme = self._maybe_dequantize_unpackable(scheme, layer, layer_name)

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        # Note: NPU devices do not support min_capability function
        # A scheme that needs no device kernel has no capability floor to
        # check -- see CompressedTensorsLinearScheme.needs_device_kernel.
        if not _is_npu and scheme.needs_device_kernel():
            self._check_scheme_supported(scheme.get_min_capability())
        logger.debug("Using scheme: %s for %s", scheme.__class__.__name__, layer_name)
        return scheme

    @staticmethod
    def _maybe_dequantize_unpackable(
        scheme: CompressedTensorsLinearScheme,
        layer: torch.nn.Module,
        layer_name: Optional[str],
        output_size_per_partition: Optional[int] = None,
    ) -> CompressedTensorsLinearScheme:
        """Route a quantised layer with no kernel-legal form to the dense lane.

        Task #332, extended to the native lane by #336. The order is deliberate
        and mirrors #316: the GEOMETRY verdict comes first
        (``nvfp4_unpackable_reason``), because that is what decides WHICH
        layers can never be served; only then does the consequence change.
        #316 could build the layer dense-and-empty because GPTQ leaves such a
        module dense on disk, while a compressed-tensors NVFP4 checkpoint that
        targets ``Linear`` wholesale has really packed it, so it must be loaded
        packed and materialised.

        Called twice per layer, because the two FP4 lanes are judged on
        different widths (see ``nvfp4_unpackable_reason``): once from
        ``get_quant_method`` without ``output_size_per_partition``, where only
        the unsharded width exists, and once from
        ``CompressedTensorsLinearMethod.create_weights`` with it, which is the
        only place the native lane's ``n % 32`` can be answered. The second
        call is a no-op for a layer the first already re-routed.

        Only the NVFP4 scheme has such a lane; every other scheme is returned
        untouched, so no other checkpoint class changes behaviour.
        """
        if not isinstance(scheme, CompressedTensorsW4A4Fp4):
            return scheme
        if isinstance(scheme, CompressedTensorsW4A4Fp4Dequant):
            return scheme
        reason = nvfp4_unpackable_reason(layer, output_size_per_partition)
        if reason is None:
            return scheme
        # Loud, once per layer, and named: a lane change that silently costs
        # VRAM must be readable off the boot log without a debugger.
        logger.warning(
            "NVFP4: layer '%s' has no form this rank's FP4 lane can serve "
            "(%s), so the checkpoint's packed weight is DEQUANTISED to dense "
            "%s at load (#332/#336). No precision is lost -- the quantisation "
            "error is already in the stored codes -- but this layer costs "
            "bf16 VRAM instead of 4 bits.",
            layer_name,
            reason,
            "bf16/fp16",
        )
        return CompressedTensorsW4A4Fp4Dequant()

    def get_scheme_dict(
        self, layer: torch.nn.Module, layer_name: str | None = None
    ) -> dict[str, QuantizationArgs | str | None] | None:
        """
        Extract the QuantizationArgs for a given layer.

        Returns:
            dict with {
                "weights": QuantizationArgs,
                "input_activations": QuantizationArgs | None,
                "format": str | None
            } | None
        """
        if should_ignore_layer(
            layer_name, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
        ):
            return None

        # Will be empty for models with only sparsity
        if self.target_scheme_map:
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=self.target_scheme_map.keys(),
                fused_mapping=self.packed_modules_mapping,
            )

            return self.target_scheme_map[matched_target]

        return None

    def get_cache_scale(self, name: str) -> Optional[str]:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        # If no matches, return None
        return None

    @staticmethod
    def supports_cutlass_24(
        weight_quant: Optional[QuantizationArgs],
        input_quant: Optional[QuantizationArgs],
        sparsity_scheme: Optional[SparsityCompressionConfig] = None,
    ) -> bool:
        """
        Check if the layer is supported by the Cutlass 2:4 Kernel
        Conditions:
            - Overarching condition: Sparsity Structure is 2:4
            - Unquantized cases are supported
            - Weight only quantization is not-supported
            - Supported weight quantization strategies are TENSOR and CHANNEL
            - Supported input quantization strategies are TENSOR and TOKEN
            - Only 8 bit quantization is supported

        :return: True if the layer is supported by the Cutlass 2:4 Kernel
            False otherwise
        """
        if sparsity_scheme is None:
            return False

        is_valid_sparsity_structure: bool = (
            sparsity_scheme.sparsity_structure == SparsityStructure.TWO_FOUR.value
        )

        valid_compressors = {
            CompressionFormat.dense.value,
            CompressionFormat.sparse_24_bitmask.value,
        }

        is_valid_sparsity = (
            is_valid_sparsity_structure and sparsity_scheme.format in valid_compressors
        )

        if not is_valid_sparsity:
            return False

        # Unquantized cases are supported
        if weight_quant is None and input_quant is None:
            return True

        # Weight only quantization is not-supported
        if weight_quant is not None and input_quant is None:
            return False

        supported_weight_quant_strategies = [
            QuantizationStrategy.TENSOR.value,
            QuantizationStrategy.CHANNEL.value,
        ]

        assert weight_quant is not None
        assert input_quant is not None
        if weight_quant.strategy not in supported_weight_quant_strategies:
            return False

        supported_input_quant_strategies = [
            QuantizationStrategy.TENSOR.value,
            QuantizationStrategy.TOKEN.value,
        ]

        if input_quant.strategy not in supported_input_quant_strategies:
            return False

        return weight_quant.num_bits == input_quant.num_bits == 8


class CompressedTensorsLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: CompressedTensorsConfig):
        self.quantization_config = quantization_config
        self.quant_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

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
        """
        Use the CompressedTensorsScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        # #336: the native FP4 lane's geometry condition is on the SHARDED
        # width (cutlass_scaled_fp4_mm asserts n % 32), and this is the first
        # moment that width exists -- ColumnParallelLinear computes it after
        # LinearBase.__init__ returns, i.e. after get_quant_method already
        # picked the scheme. Re-run the verdict with it and swap the lane
        # before a single parameter is created; layer.scheme is also what
        # process_weights_after_loading and apply read afterwards, so the swap
        # carries through the whole layer lifetime.
        layer.scheme = self.quantization_config._maybe_dequantize_unpackable(
            layer.scheme,
            layer,
            getattr(layer, "prefix", None),
            output_size_per_partition=sum(output_partition_sizes),
        )
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)


class CompressedTensorsFusedMoEMethod(FusedMoEMethodBase):

    def __init__(self, quantization_config: CompressedTensorsConfig):
        self.quantization_config = quantization_config
        self.quant_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the CompressedTensorsScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        layer.scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        return layer.scheme.create_moe_runner(layer, moe_runner_config)

    def get_triton_quant_info(self, layer: torch.nn.Module):
        return layer.scheme.get_triton_quant_info(layer)

    def get_marlin_quant_info(self, layer: torch.nn.Module):
        return layer.scheme.get_marlin_quant_info(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, dispatch_output)

    def apply_weights_with_router_logits(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights_with_router_logits(layer, dispatch_output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        return layer.scheme.apply_without_routing_weights(
            layer,
            hidden_states,
            hidden_states_scale,
            group_list_type,
            group_list,
            output_dtype,
        )
