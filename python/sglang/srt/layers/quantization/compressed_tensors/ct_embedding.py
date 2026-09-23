# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#727: quantized-resident vocab weights for compressed-tensors checkpoints.

WHAT THIS CLOSES. #724 found the dequant-on-gather capability wired but scoped
to GGUF: ``qwen3_5.py`` hands a ``quant_config`` to ``VocabParallelEmbedding``
only when the config's name is ``gguf``, and the tree carried exactly two
embedding methods (``UnquantizedEmbeddingMethod``, ``GGUFEmbeddingMethod``).
Compressed-tensors' ``get_quant_method`` answers for ``LinearBase`` and
``FusedMoE`` and nothing else, so an int8 vocab tensor had no method able to
load it. That is a missing component, not a missing flag, which is why this
module exists rather than a one-line widening of the gate.

WHY THE GATHER MAKES THIS CHEAP. The checkpoint's scheme is symmetric
per-output-channel int8 -- for a vocab matrix that is one scale PER ROW, i.e.
per token id. So a lookup dequantizes exactly the rows it gathered: a handful
per batch, never the 248320-row matrix. Storage falls from BF16 2425.0 MiB to
INT8 1212.5 MiB (plus a 0.5 MiB scale column) and the runtime cost is one
multiply on the gathered rows.

``lm_head`` is deliberately NOT covered by the same reasoning. It is a GEMM
against the whole vocab producing logits directly, so its error lands on logit
DIFFERENCES where softmax and argmax can see it. It is a separate decision with
its own A/B, and this module does not quietly enable it.

DEFAULT UNCHANGED. Every checkpoint we serve today lists ``embed_tokens`` in
``quantization_config.ignore``, and :func:`vocab_is_quantized` reads exactly
that list. On such a checkpoint this method is never selected and the dense
BF16 path runs byte-identically.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.utils import set_weight_attrs


def is_compressed_tensors_config(quant_config: Any) -> bool:
    """#763: does this config belong to the compressed-tensors family?

    The family answers ``get_name()`` with the UNDERSCORE spelling
    (``compressed_tensors``, see ``CompressedTensorsConfig.get_name``), while
    the checkpoint's ``quant_method`` field and most prose use the HYPHEN
    spelling. #727's model-side gate compared against the hyphen, so it never
    matched a real config and the int8 vocab silently kept the dense path --
    loading int8 rows into a BF16 embedding with no scale applied, which is
    what token soup looks like from the outside.

    Matching on the normalized name rather than one literal is what keeps a
    second spelling from re-opening this. The predicate lives here, beside the
    method it selects, so it can be unit-tested against the real config object
    instead of being duplicated as a bare string in a model file.
    """
    if quant_config is None:
        return False
    getter = getattr(quant_config, "get_name", None)
    if not callable(getter):
        return False
    try:
        name = getter()
    except Exception:  # a config that cannot name itself is not a match
        return False
    if not isinstance(name, str):
        return False
    return name.strip().lower().replace("-", "_") == "compressed_tensors"


_SCALE_INDEX_CACHE: Dict[str, Optional[set]] = {}


def _checkpoint_scale_leaves(model_path: Optional[str]) -> Optional[set]:
    """The leaf names (``embed_tokens``, ``lm_head`` ...) for which the
    checkpoint at ``model_path`` carries a ``<name>.weight_scale`` tensor --
    read from ``model.safetensors.index.json`` or a single-file header.
    ``None`` when nothing readable is there (the caller then falls back to
    the ignore-list reading)."""
    if not model_path:
        return None
    if model_path in _SCALE_INDEX_CACHE:
        return _SCALE_INDEX_CACHE[model_path]
    leaves: Optional[set] = None
    try:
        import json
        import os

        keys = None
        idx = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isfile(idx):
            with open(idx) as f:
                keys = list((json.load(f).get("weight_map") or {}).keys())
        else:
            single = os.path.join(model_path, "model.safetensors")
            if os.path.isfile(single):
                from safetensors import safe_open

                with safe_open(single, "pt") as f:
                    keys = list(f.keys())
        if keys is not None:
            leaves = set()
            for k in keys:
                if k.endswith(".weight_scale"):
                    parts = k.split(".")
                    if len(parts) >= 2:
                        leaves.add(parts[-2])
    except Exception:  # noqa: BLE001 -- an unreadable index is not a boot killer
        leaves = None
    _SCALE_INDEX_CACHE[model_path] = leaves
    return leaves


def vocab_is_quantized(
    quant_config: Dict[str, Any], layer_name: str, model_path: Optional[str] = None
) -> bool:
    """Does this checkpoint actually carry ``layer_name`` quantized?

    17.09. (boot df2l, lued/Qwen3.8-27B-INT8-W8A16-MTP): the ignore list
    CANNOT tell.  Both that ORIGINAL checkpoint (BF16 vocab, targets
    ['Linear'], embed_tokens not in ignore) and our #727 requant (int8 vocab
    with weight_scale, same targets, same ignore) answer "quantized" here,
    so the original loaded BF16 rows through the int8 method: the target
    answered '!!!' (token 0) on every prompt and NEXTN's MTP head died in
    gemma_rmsnorm on a Char tensor.  The tensor file is the authority: when
    ``model_path`` is known, the vocab is quantized iff the checkpoint has a
    ``<leaf>.weight_scale`` beside it; the ignore-list reading below stays
    the fallback for a checkpoint whose index cannot be read.

    The authority is ``quantization_config.ignore``: a producer that excluded a
    tensor did not write scales for it, so a method that assumed otherwise
    would look for a ``weight_scale`` that is not in the file. Entries are
    either literal names or ``re:``-prefixed patterns, matching
    compressed-tensors' own convention.

    A malformed pattern is treated as a LITERAL rather than raised. Refusing to
    boot over someone else's bad regex would be the wrong trade, and the
    fallback can only ever make the answer more conservative -- a literal
    almost never matches, so the layer is treated as quantized only when the
    list genuinely does not mention it.
    """
    leaves = _checkpoint_scale_leaves(model_path)
    if leaves is not None:
        leaf = layer_name.split(".")[-1] if layer_name else ""
        return leaf in leaves
    for entry in quant_config.get("ignore", []) or []:
        if not isinstance(entry, str):
            continue
        if entry.startswith("re:"):
            pattern = entry[3:]
            try:
                if re.fullmatch(pattern, layer_name) or re.search(pattern, layer_name):
                    return False
            except re.error:
                if entry[3:] == layer_name:
                    return False
            continue
        if entry == layer_name or layer_name.endswith("." + entry):
            return False
    return True


def vocab_named_in_targets(quant_config: Dict[str, Any], layer_name: str) -> bool:
    """Does a config group NAME this vocab layer -- verbatim or by a ``re:``
    target? A class-name target (``Linear``) never covers an embedding: HF's
    ``embed_tokens`` is an ``nn.Embedding``, so an llm-compressor export with
    ``targets: [Linear]`` (cyankiwi) carries a dense embedding even though
    its ignore list, which only lists Linears, does not mention it. An
    AutoRound export that packs the vocab names it (Minachist:
    ``re:.*embed_tokens``). This is the positive evidence a model class
    needs before it hands the vocab a quant_config at all; the ignore-list
    scan in ``vocab_is_quantized`` stays the per-row int8 rule (#727)."""
    groups = quant_config.get("config_groups") or {}
    for group in groups.values():
        for target in (group or {}).get("targets") or []:
            if not isinstance(target, str):
                continue
            if target.startswith("re:"):
                try:
                    if re.fullmatch(target[3:], layer_name) or re.search(
                        target[3:], layer_name
                    ):
                        return True
                except re.error:
                    if target[3:] == layer_name:
                        return True
            elif target == layer_name:
                return True
    return False


class CompressedTensorsEmbeddingMethod(QuantizeMethodBase):
    """Symmetric per-row int8 vocab, dequantized on gather.

    The parameter layout mirrors the checkpoint exactly: ``weight`` as int8
    ``[rows, dim]`` beside ``weight_scale`` as ``[rows, 1]``. Naming them the
    way the file names them is what lets the stock weight loader place them.
    """

    def __init__(self, params_dtype: Optional[torch.dtype] = None):
        self.params_dtype = params_dtype or torch.get_default_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: Optional[torch.dtype] = None,
        weight_loader=None,
        **extra,
    ) -> None:
        rows = int(sum(output_partition_sizes))
        dim = int(input_size_per_partition)
        if params_dtype is not None:
            self.params_dtype = params_dtype

        weight = torch.nn.Parameter(
            torch.empty(rows, dim, dtype=torch.int8), requires_grad=False
        )
        scale = torch.nn.Parameter(
            torch.empty(rows, 1, dtype=torch.float32), requires_grad=False
        )
        # #763: BOTH parameters must declare that the vocab rows are dim 0.
        # VocabParallelEmbedding.weight_loader reads `output_dim` to decide
        # whether to narrow the checkpoint tensor to this rank's row range;
        # a parameter without it takes the "copy onto all gpus" branch meant
        # for shard-invariant tensors (gptq's g_idx), which for a vocab
        # matrix is the wrong contract entirely. It is silent at tp_size 1 --
        # the whole vocab IS the local shard, so PP=3 serving is correct --
        # and only bites once the vocab is row-sharded across TP ranks.
        # The scale carries one entry PER ROW, so it shards on dim 0 exactly
        # like the rows it belongs to; slicing one without the other would
        # pair each row with a stranger's scale.
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(scale, {"output_dim": 0})
        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", scale)
        if weight_loader is not None:
            for param in (weight, scale):
                setattr(param, "weight_loader", weight_loader)
        if extra:
            for param in (weight, scale):
                set_weight_attrs(param, extra)

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Gather rows, then scale them. In that order, deliberately.

        Dequantizing first would materialize the whole vocab in the activation
        dtype -- 2.4 GiB at BF16 on this checkpoint -- which is precisely the
        cost this path exists to avoid.
        """
        rows = torch.nn.functional.embedding(x, layer.weight)
        scales = torch.nn.functional.embedding(x, layer.weight_scale)
        return rows.to(self.params_dtype) * scales.to(self.params_dtype)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias=None):
        """The ParallelLMHead path: a dense matmul against the dequantized rows.

        Present so the class is usable for a head as well, but note that
        enabling it for ``lm_head`` is a separate decision (see the module
        docstring) -- nothing here selects it.
        """
        weight = layer.weight.to(self.params_dtype) * layer.weight_scale.to(
            self.params_dtype
        )
        out = torch.nn.functional.linear(x, weight, bias)
        return out


class CompressedTensorsPackedEmbeddingMethod(QuantizeMethodBase):
    """``pack-quantized`` group-int vocab (e.g. INT8 g128 as AutoRound writes
    Qwen3.8-Flash-Next's ``embed_tokens``), dequantized on gather.

    Parameter layout mirrors the checkpoint: ``weight_packed`` int32
    ``[rows, dim*bits/32]`` (little-endian, values stored unsigned with offset
    2**(bits-1)), ``weight_scale`` ``[rows, dim/group]``, ``weight_shape``.
    Only whole rows are gathered and unpacked, so the cost stays per token.
    """

    def __init__(
        self, num_bits: int, group_size: int, symmetric: bool = True, params_dtype=None
    ):
        if not symmetric:
            raise NotImplementedError("packed vocab: only symmetric groups are supported")
        if 32 % num_bits:
            raise NotImplementedError(f"packed vocab: {num_bits}-bit dense packing unsupported")
        self.num_bits = int(num_bits)
        self.pack_factor = 32 // self.num_bits
        self.group_size = int(group_size)
        self.params_dtype = params_dtype or torch.get_default_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: Optional[torch.dtype] = None,
        weight_loader=None,
        **extra,
    ) -> None:
        rows = int(sum(output_partition_sizes))
        dim = int(input_size_per_partition)
        if params_dtype is not None:
            self.params_dtype = params_dtype
        if dim % self.pack_factor or dim % self.group_size:
            raise ValueError(
                f"packed vocab: dim {dim} must be a multiple of the pack factor "
                f"{self.pack_factor} and the group size {self.group_size}"
            )
        weight = torch.nn.Parameter(
            torch.empty(rows, dim // self.pack_factor, dtype=torch.int32),
            requires_grad=False,
        )
        scale = torch.nn.Parameter(
            torch.empty(rows, dim // self.group_size, dtype=torch.float16),
            requires_grad=False,
        )
        shape = torch.nn.Parameter(torch.empty(2, dtype=torch.int64), requires_grad=False)
        # rows are dim 0 for the vocab loader (see the per-row method, #763);
        # the packed axis is the hidden dim, never sharded by the vocab loader.
        set_weight_attrs(
            weight,
            {"input_dim": 1, "output_dim": 0, "packed_dim": 1, "packed_factor": self.pack_factor},
        )
        set_weight_attrs(scale, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", scale)
        layer.register_parameter("weight_shape", shape)
        for param in (weight, scale, shape):
            if weight_loader is not None:
                setattr(param, "weight_loader", weight_loader)
            if extra:
                set_weight_attrs(param, extra)

    def _dequant_rows(self, packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        n = packed.shape[0]
        w = packed.to(torch.int64) & 0xFFFFFFFF
        shifts = torch.arange(self.pack_factor, device=packed.device, dtype=torch.int64)
        mask = (1 << self.num_bits) - 1
        raw = (w.unsqueeze(-1) >> (shifts * self.num_bits)) & mask  # [n, words, pf]
        q = raw.reshape(n, -1) - (1 << (self.num_bits - 1))
        q = q.view(n, -1, self.group_size).to(self.params_dtype)
        return (q * scale.to(self.params_dtype).unsqueeze(-1)).reshape(n, -1)

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        packed = layer.weight_packed[x]
        scale = layer.weight_scale[x]
        return self._dequant_rows(packed, scale)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias=None):
        """Dense fallback (the head goes through the Marlin linear scheme
        instead; this dequantizes the whole vocab and is for tests only)."""
        weight = self._dequant_rows(layer.weight_packed, layer.weight_scale)
        return torch.nn.functional.linear(x, weight, bias)
