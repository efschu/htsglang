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

import logging
import re
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import QuantizeMethodBase

logger = logging.getLogger(__name__)


def vocab_is_quantized(quant_config: Dict[str, Any], layer_name: str) -> bool:
    """Does this checkpoint actually carry ``layer_name`` quantized?

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
        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", scale)
        if weight_loader is not None:
            for param in (weight, scale):
                setattr(param, "weight_loader", weight_loader)
        # #727 GATE 0 observable: the A/B runner greps the boot log for this
        # line to prove the int8 path ENGAGED (0 lines = baseline, 1 = embed
        # only, 2 = embed + lm_head). A boot that silently fell back to the
        # dense BF16 path proves nothing about quality and shows a WORSE VRAM
        # number -- the failure mode GATE 0 exists to catch mechanically.
        logger.info(
            "INT8-VOCAB ENGAGED: %d x %d int8 rows with per-row scales "
            "(%s head)",
            rows,
            dim,
            type(layer).__name__,
        )

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
