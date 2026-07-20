# SPDX-License-Identifier: Apache-2.0
"""GGUF loading support for the Gemma-4 family (GGUF arch ``gemma4``).

Second bespoke family entry beside ``gguf_qwen35.py``.  Dense Gemma-4 only —
MoE / MTP / vision GGUF are deferred (S3) and fail fast below.

Structure (#129 S2): the quality/speed-neutral scaffolding (arch resolution,
file-tensor probe, the emit/audit ``build_name_map`` skeleton, the F32-carve-out
``unquantized_module_prefixes`` skeleton) lives in ``GGUFAdapterBase``; this
module supplies the gemma4 emit *tables* + the irreducible per-family hooks
(checkpoint-name rewrite, MoE fail-fast, dropped-tensor audit, module-prefix
spelling) and the verbatim ``transform_stream`` (embed dequant + identity
norms).

Launch recipe (validated: Gemma-4-31B-it GGUF-Q4_K_M, TP=1 on an RTX 5090,
coherent + self-deterministic + ~61 tok/s). Two non-obvious traps, neither of
which is a code issue in this file but both of which block a naive launch:

  * Attention backend: Gemma-4 REJECTS the flashinfer backend (it asserts
    "only supports trtllm_mha, triton, or intel_xpu"). Launch with
    ``--attention-backend triton``.
  * Device identity (torch enumeration != NVML/nvidia-smi order): CUDA's
    default ``FASTEST_FIRST`` puts a fast card (e.g. a 5090 among 3080s) at
    ``cuda:0``, so an NVML index does NOT map to the same ``cuda:N``. To pin
    an NVML-resolved physical GPU, set
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so ``CUDA_VISIBLE_DEVICES=<nvml_idx>``
    selects that exact card (without it, the wrong card is chosen -> OOM).

  Full working invocation (this fork's worktree on PYTHONPATH):

    PYTHONPATH=<worktree>/python \\
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<nvml_idx> \\
    python -m sglang.launch_server \\
      --model-path <gemma4>.gguf --load-format gguf --dtype bfloat16 \\
      --tp-size 1 --attention-backend triton \\
      --mem-fraction-static 0.85 --context-length 4096 --trust-remote-code

Scope: only Q4_K_M GGUF is empirically verified. Other GGUF quant levels
(and non-GGUF quants) use the same code paths and should work in principle
but are not tested.

Why the generic GGUF path cannot load Gemma-4:

1. **Config.** transformers' GGUF metadata reader crashes on the gemma4
   metadata (it derives a per-layer ``num_key_value_heads`` LIST, which the
   strict Gemma4TextConfig rejects), so ``config.py`` reads the sibling
   ``config.json`` instead (same workaround as qwen35; see
   ``_peek_bespoke_gguf_arch`` there).  That yields the multimodal wrapper
   config (``Gemma4ForConditionalGeneration``); the GGUF text-only force-off
   in ``model_config.py`` keeps the vision tower unexercised.

2. **Name map.** The generic ``_get_gguf_weights_map`` instantiates the HF
   model on ``meta`` to enumerate param names; the fork's Gemma-4 wrapper
   expects HF-checkpoint spellings (``model.language_model.*``,
   ``layer_scalar``) that differ from what that path would produce.  We build
   the map from an explicit template against gguf-py's ``gemma4`` tensor map
   (which already knows every dense tensor incl.
   ``layer_scalar -> blk.N.layer_output_scale``) with present-in-file
   filtering, which also auto-handles the ``attention_k_eq_v`` layers: the
   GGUF export omits ``attn_v`` on full-attention layers and the model's
   ``load_weights`` duplicates the k shard into v (``should_dup_k_to_v``).

3. **Embedding.** ``token_embd`` is quantized (e.g. Q4_K) but the fork's
   Gemma-4 builds ``embed_tokens`` as a plain ``nn.Embedding`` subclass
   (``Gemma4TextScaledWordEmbedding``) — no GGUFEmbeddingMethod — so the
   embedding must be dequantized to a dense ``.weight`` on load
   (``transform_stream``).  With ``tie_word_embeddings`` the lm_head shares
   the same dense module, and the GGUF ships no ``output`` tensor.

Norm handling (the design's single highest-risk detail, resolved
empirically): the verified Gemma-4 export (unsloth Gemma-4-31B-it Q4_K_M,
llama.cpp arch ``gemma4``) stores every norm gamma — input/post_attention/
pre_ff/post_ff layernorms, model.norm, attn q/k norms — and ``layer_scalar``
BYTE-IDENTICAL to the bf16 safetensors checkpoint.  Unlike the qwen35 GGUFs,
this exporter does NOT bake ``+1`` into the Gemma RMSNorm gammas.  The
correct load transform for every norm is therefore IDENTITY: the tensors go
through the exact same weight loaders as the safetensors path (RMSNorm's
``(1+w)`` precompute, Gemma4RMSNorm's plain ``w`` with ``scale_shift=0``).
``SGLANG_GGUF_GEMMA4_NORM_UNBAKE=1`` is an escape hatch for hypothetical
exports that DO bake +1 (it subtracts 1.0 from the ``(1+w)``-consumed
RMSNorm gammas only, mirroring qwen35's ``_GEMMA_NORM_*`` treatment; the
q/k norms are consumed as plain ``w`` here and are never touched).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, Optional, Tuple

import torch

from sglang.srt.model_loader.gguf_adapter_base import GGUFAdapterBase

logger = logging.getLogger(__name__)

# HF model_type -> GGUF arch name (values as in gguf.MODEL_ARCH_NAMES).
_MODEL_TYPE_TO_GGUF_ARCH = {
    "gemma4": "gemma4",
    "gemma4_text": "gemma4",
}


def is_gemma4_gguf_arch(model_type: Optional[str]) -> bool:
    return model_type in _MODEL_TYPE_TO_GGUF_ARCH


# Norm gammas the model consumes as ``x * (1 + w)`` (fork RMSNorm class).
# Only these are touched by the SGLANG_GGUF_GEMMA4_NORM_UNBAKE=1 escape
# hatch.  q_norm/k_norm are Gemma4RMSNorm(scale_shift=0) — plain ``w`` —
# and must NEVER get a -1 (the verified export stores them raw anyway).
_GEMMA_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
)
_GEMMA_NORM_NAMES = ("model.language_model.norm.weight",)

# Non-block top-level tensors llama.cpp emits that have no HF param in the
# fork's model and are safe to drop:
#  - rope_freqs: precomputed inverse-frequency scaling for the partial-rotary
#    ("proportional") full-attention rope; the model derives the identical
#    rotation from config.rope_parameters (partial_rotary_factor), so the
#    table is redundant at load time.
_DROPPED_TENSORS = ("rope_freqs.weight",)


def _unbake_norms() -> bool:
    return os.environ.get("SGLANG_GGUF_GEMMA4_NORM_UNBAKE", "0") == "1"


class Gemma4GGUFAdapter(GGUFAdapterBase):
    """Builds the GGUF<->HF name map and load-time transforms for one dense
    Gemma-4 checkpoint.  Tables + hooks over ``GGUFAdapterBase``; the verbatim
    ``transform_stream`` (embed dequant + identity norms) is the irreducible
    per-family bit."""

    FAMILY = "gemma4"
    MODEL_TYPE_TO_ARCH = _MODEL_TYPE_TO_GGUF_ARCH

    # ``lm_head`` (GGUF ``output``) is absent from tied exports; present-in-file
    # filtering keeps the map honest either way.
    GLOBAL_ENTRIES = (
        ("model.embed_tokens", "weight"),
        ("lm_head", "weight"),
        ("model.norm", "weight"),
    )
    LAYER_ENTRIES = (
        ("input_layernorm", "weight"),
        ("post_attention_layernorm", "weight"),
        ("pre_feedforward_layernorm", "weight"),
        ("post_feedforward_layernorm", "weight"),
        # attention (attn_v omitted by the export on k_eq_v layers; the
        # model's should_dup_k_to_v fills the v shard from k)
        ("self_attn.q_proj", "weight"),
        ("self_attn.k_proj", "weight"),
        ("self_attn.v_proj", "weight"),
        ("self_attn.o_proj", "weight"),
        ("self_attn.q_norm", "weight"),
        ("self_attn.k_norm", "weight"),
        # dense MLP (gelu gate-up; gate/up fuse via stacked_params_mapping)
        ("mlp.gate_proj", "weight"),
        ("mlp.up_proj", "weight"),
        ("mlp.down_proj", "weight"),
        # learned per-layer residual scalar: GGUF blk.N.layer_output_scale
        # .weight -> bare HF buffer model.language_model.layers.N.layer_scalar
        ("layer_scalar", "weight", ""),
    )
    # Split projections -> fused module for the F32 carve-out fold.
    FUSE_MAP = {
        "self_attn.q_proj": "self_attn.qkv_proj",
        "self_attn.k_proj": "self_attn.qkv_proj",
        "self_attn.v_proj": "self_attn.qkv_proj",
        "mlp.gate_proj": "mlp.gate_up_proj",
        "mlp.up_proj": "mlp.gate_up_proj",
    }

    # ------------------------------------------------------------------
    # Name-map hooks
    # ------------------------------------------------------------------

    @staticmethod
    def _to_checkpoint_name(hf_base: str) -> str:
        """gguf-py map spelling -> HF-checkpoint spelling.

        gguf-py's gemma4 tensor map is keyed on ``model.layers.*`` /
        ``model.embed_tokens`` / ``model.norm``; the actual checkpoint (and
        the wrapper's ``load_weights``, which strips the leading ``model.``)
        uses ``model.language_model.*``.  ``Gemma4ForCausalLM.load_weights``
        also accepts these (it rewrites ``model.language_model.`` ->
        ``model.``), so the emitted names work for both classes.
        """
        if hf_base.startswith("model."):
            return "model.language_model." + hf_base[len("model.") :]
        return hf_base

    def _map_base_to_hf(self, map_base: str) -> str:
        return self._to_checkpoint_name(map_base)

    def _pre_map_hook(self, file_tensors: set) -> None:
        # Dense-only (S1): a MoE Gemma-4 GGUF needs the stacked-experts
        # split re-prefixed for the wrapper's param tree (S3); fail fast
        # instead of silently skipping expert weights (NaN hazard).
        moe_tensors = [
            n for n in file_tensors if "_exps." in n or ".ffn_gate_inp." in n
        ]
        if moe_tensors:
            raise RuntimeError(
                "Gemma-4 MoE GGUF is not supported yet (dense-only S1); found "
                f"expert tensors, e.g. {sorted(moe_tensors)[:4]}"
            )

    def _is_unmapped_ignorable(self, name: str) -> bool:
        return name.startswith(("v.", "mm.")) or name in _DROPPED_TENSORS

    def _module_prefix_spelling(self, base: str) -> str:
        # module prefixes lack the checkpoint's leading "model."
        if base.startswith("model."):
            return base[len("model.") :]
        return base

    # ------------------------------------------------------------------
    # Weight-value transforms
    # ------------------------------------------------------------------

    _EMBED_PREFIX = "model.language_model.embed_tokens."

    def _target_dtype(self) -> torch.dtype:
        dtype = getattr(self.config, "torch_dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str) and hasattr(torch, dtype):
            return getattr(torch, dtype)
        return torch.get_default_dtype()

    def transform_stream(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """Wrap the raw GGUF weight iterator.

        - ``token_embd``: dequantized to a dense ``.weight`` (the model's
          embed_tokens is a plain nn.Embedding; with tie_word_embeddings the
          lm_head shares it).
        - Norms: IDENTITY by default (the verified export stores raw
          checkpoint values; see module docstring).  Optional -1 un-bake for
          ``(1+w)``-consumed gammas under SGLANG_GGUF_GEMMA4_NORM_UNBAKE=1.
        - Everything else passes through untouched (the quantized
          projections load via GGUFLinearMethod exactly like qwen35's).
        """
        import gguf
        from gguf.quants import dequantize

        unbake = _unbake_norms()
        if unbake:
            logger.warning(
                "SGLANG_GGUF_GEMMA4_NORM_UNBAKE=1: subtracting 1.0 from "
                "(1+w)-consumed Gemma-4 norm gammas. The verified unsloth "
                "Q4_K_M export does NOT need this (it stores raw values)."
            )
        embed_qtype: Optional[int] = None
        for name, weight in weights:
            if name.startswith(self._EMBED_PREFIX):
                if name.endswith(".qweight_type"):
                    embed_qtype = int(weight.item())
                    continue
                if name.endswith(".qweight"):
                    assert embed_qtype is not None, "embed qweight_type not seen"
                    qtype = gguf.GGMLQuantizationType(embed_qtype)
                    if qtype == gguf.GGMLQuantizationType.BF16:
                        # gguf-py exposes BF16 as raw uint8 (last dim doubled)
                        dense = torch.tensor(weight).view(torch.bfloat16)
                    else:
                        dense = torch.from_numpy(
                            dequantize(weight.numpy(), qtype).copy()
                        )
                    yield (
                        self._EMBED_PREFIX + "weight",
                        dense.to(self._target_dtype()),
                    )
                    continue
                # already F32 `.weight`: fall through unchanged
            if unbake and (
                name.endswith(_GEMMA_NORM_SUFFIXES) or name in _GEMMA_NORM_NAMES
            ):
                yield name, weight.float() - 1.0
                continue
            yield name, weight
