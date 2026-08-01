# SPDX-License-Identifier: Apache-2.0
"""Bespoke GGUF adapter for DeepSeek V4 Flash (#391).

Third bespoke family beside qwen35 and gemma4. Two things force the bespoke
path here, both hard:

1. ``general.architecture`` in the file is ``deepseek4``. Neither transformers'
   GGUF config reader (``GGUF_SUPPORTED_ARCHITECTURES``) nor gguf-py's
   ``MODEL_ARCH_NAMES`` knows that architecture, so the generic path dies in
   transformers before any sglang code runs, and the sibling-config route is
   the only way to a config.
2. Because gguf-py does not know the arch, ``GGUFAdapterBase.build_name_map``'s
   engine -- ``gguf.get_tensor_name_map(arch_enum, n_layers)`` -- is not
   available either. This adapter therefore carries an EXPLICIT name table
   instead of the emit tables the other two families drive that engine with,
   and overrides ``build_name_map``. Everything else (file probe, F32 carve-out,
   registry dispatch, sibling-config reconciliation) is reused unchanged.

Name mapping targets the DeepSeek-NATIVE checkpoint spelling
(``layers.N.attn.wq_a.weight``, ``layers.N.ffn.shared_experts.w1.weight``, ...)
rather than sglang's internal parameter names. That is deliberate and is the
short path: ``DeepseekV4ForCausalLM.load_weights`` puts every incoming key
through ``remap_weight_name_to_dpsk_hf_format`` first, which is the same
translation the safetensors checkpoint gets. Emitting native names means the
GGUF path and the safetensors path converge on one translation instead of two.

Verified against unsloth/DeepSeek-V4-Flash-0731-GGUF UD-Q3_K_XL (1328 tensors,
4 shards): the map is a bijection onto the tensor set, and every produced name
occurs in the upstream ``model.safetensors.index.json`` of
deepseek-ai/DeepSeek-V4-Flash-0731.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from sglang.srt.model_loader.gguf_adapter_base import GGUFAdapterBase

logger = logging.getLogger(__name__)


#: GGUF ``general.architecture`` for this family.
DEEPSEEK4_GGUF_ARCH = "deepseek4"

#: Non-layer tensors: ``gguf_name -> native checkpoint name``.
_GLOBAL_MAP: Dict[str, str] = {
    "token_embd.weight": "embed.weight",
    "output.weight": "head.weight",
    "output_norm.weight": "norm.weight",
    # Hyper-connection head parameters are bare tensors, not module weights.
    "output_hc_base.weight": "hc_head_base",
    "output_hc_fn.weight": "hc_head_fn",
    "output_hc_scale.weight": "hc_head_scale",
}

#: Per-block tensors: ``gguf suffix -> native suffix under layers.{i}.``.
#: A native suffix without ``.weight`` is a bare parameter (attention sink,
#: absolute positional embedding of a compressor, hyper-connection triples,
#: the hash-router table).
_LAYER_MAP: Dict[str, str] = {
    # --- norms -----------------------------------------------------------
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    # --- MLA-style attention ---------------------------------------------
    "attn_q_a.weight": "attn.wq_a.weight",
    "attn_q_b.weight": "attn.wq_b.weight",
    "attn_kv.weight": "attn.wkv.weight",
    "attn_output_a.weight": "attn.wo_a.weight",
    "attn_output_b.weight": "attn.wo_b.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_sinks.weight": "attn.attn_sink",
    # --- KV compressor (layers with a non-zero compress_ratio) -----------
    "attn_compressor_ape.weight": "attn.compressor.ape",
    "attn_compressor_gate.weight": "attn.compressor.wgate.weight",
    "attn_compressor_kv.weight": "attn.compressor.wkv.weight",
    "attn_compressor_norm.weight": "attn.compressor.norm.weight",
    # --- DSA-style sparse indexer (every second layer) -------------------
    "indexer.attn_q_b.weight": "attn.indexer.wq_b.weight",
    "indexer.proj.weight": "attn.indexer.weights_proj.weight",
    "indexer_compressor_ape.weight": "attn.indexer.compressor.ape",
    "indexer_compressor_gate.weight": "attn.indexer.compressor.wgate.weight",
    "indexer_compressor_kv.weight": "attn.indexer.compressor.wkv.weight",
    "indexer_compressor_norm.weight": "attn.indexer.compressor.norm.weight",
    # --- MoE router ------------------------------------------------------
    "ffn_gate_inp.weight": "ffn.gate.weight",
    "exp_probs_b.bias": "ffn.gate.bias",
    # The first num_hash_layers blocks route by a token-id -> expert-id table
    # instead of a learned gate, so they carry tid2eid and no bias.
    "ffn_gate_tid2eid.weight": "ffn.gate.tid2eid",
    # --- shared expert (w1 = gate, w2 = down, w3 = up) -------------------
    "ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight",
    "ffn_down_shexp.weight": "ffn.shared_experts.w2.weight",
    "ffn_up_shexp.weight": "ffn.shared_experts.w3.weight",
    # --- routed experts --------------------------------------------------
    # llama.cpp stacks all 256 experts of a projection into ONE 3D tensor.
    # These entries exist only to satisfy the unmapped-tensor audit: the real
    # per-expert split happens earlier, in the generic
    # weight_utils.gguf_quant_weights_iterator, which emits the hardcoded
    # model.layers.{L}.mlp.experts.{e}.{gate,up,down}_proj.qweight names
    # directly (same as for qwen35).
    "ffn_gate_exps.weight": "ffn.experts.w1.weight",
    "ffn_down_exps.weight": "ffn.experts.w2.weight",
    "ffn_up_exps.weight": "ffn.experts.w3.weight",
    # --- hyper-connections (bare parameters) -----------------------------
    "hc_attn_base.weight": "hc_attn_base",
    "hc_attn_fn.weight": "hc_attn_fn",
    "hc_attn_scale.weight": "hc_attn_scale",
    "hc_ffn_base.weight": "hc_ffn_base",
    "hc_ffn_fn.weight": "hc_ffn_fn",
    "hc_ffn_scale.weight": "hc_ffn_scale",
}

#: The stacked routed-expert tensors, whose collective entries above are
#: audit-only (see the comment there).
_COLLECTIVE_EXPERT_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_down_exps.weight",
    "ffn_up_exps.weight",
)

#: Native suffixes whose payload is an integer table, not a quantized weight.
#: The generic iterator has no notion of that (see transform_stream).
_INTEGER_TABLE_SUFFIXES = (".gate.tid2eid",)

#: Native prefix of the token embedding, whose module is dense (see
#: transform_stream). The output projection has no counterpart here on
#: purpose: this family does not tie the two, so ``head`` stays packed.
_EMBED_PREFIX = "embed."


def _supported_ggml_types() -> set:
    """The GGML types this stack can actually execute, as a set of
    ``gguf.GGMLQuantizationType``.

    Single source of truth is the quantization layer's own type sets, so this
    check cannot drift away from what the kernels dispatch on, plus the types
    the load-time repack (gguf_mxfp4_repack) converts into that set before any
    consumer sees them. The repack contributes nothing when it is switched off,
    so the gate then refuses exactly what it refused before it existed.
    """
    import gguf

    from sglang.srt.layers.quantization.gguf import DEQUANT_TYPES
    from sglang.srt.model_loader.gguf_mxfp4_repack import repack_source_types

    unquantized = {
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
        gguf.GGMLQuantizationType.I32,
    }
    return set(DEQUANT_TYPES) | unquantized | repack_source_types()


class Deepseek4GGUFAdapter(GGUFAdapterBase):
    """DeepSeek V4 Flash GGUF adapter (arch ``deepseek4``)."""

    FAMILY = "deepseek4"
    MODEL_TYPE_TO_ARCH = {"deepseek_v4": DEEPSEEK4_GGUF_ARCH}
    # The emit tables stay empty: build_name_map is overridden because gguf-py
    # has no deepseek4 tensor map to drive them with.
    GLOBAL_ENTRIES = ()
    LAYER_ENTRIES = ()
    FUSE_MAP: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Name map
    # ------------------------------------------------------------------

    def _build_name_map_unchecked(self) -> Dict[str, str]:
        """``{gguf_tensor_name: native_checkpoint_name}`` for the tensors that
        are actually in the file, with no executability check.

        Split out from :meth:`build_name_map` so the mapping can be audited on
        a machine that cannot run the quantization kernels the checked variant
        insists on.
        """
        file_tensors = self._file_tensors()
        gguf_to_native: Dict[str, str] = {}

        for gguf_name, native in _GLOBAL_MAP.items():
            if gguf_name in file_tensors:
                gguf_to_native[gguf_name] = native

        for i in range(self.num_layers):
            for suffix, native_suffix in _LAYER_MAP.items():
                gguf_name = f"blk.{i}.{suffix}"
                if gguf_name in file_tensors:
                    gguf_to_native[gguf_name] = f"layers.{i}.{native_suffix}"

        unmapped = [
            name
            for name in file_tensors
            if name not in gguf_to_native and not self._is_unmapped_ignorable(name)
        ]
        if unmapped:
            raise RuntimeError(
                f"{self.FAMILY} GGUF: {len(unmapped)} tensors not mapped: "
                f"{sorted(unmapped)[:12]}"
            )
        return gguf_to_native

    def build_name_map(self) -> Dict[str, str]:
        name_map = self._build_name_map_unchecked()
        self.assert_quant_types_executable()
        logger.info(
            "%s GGUF name map: %d tensors for %d layers (arch %s)",
            self.FAMILY,
            len(name_map),
            self.num_layers,
            self.arch,
        )
        return name_map

    def _is_unmapped_ignorable(self, name: str) -> bool:
        # The published GGUF exports the 43 backbone blocks only -- no MTP /
        # NEXTN block -- so there is nothing to ignore beyond the base class's
        # vision prefixes. A future export that adds blk.>=num_layers would
        # need a draft name map (is_draft), not a silent skip, so do not widen
        # this.
        return super()._is_unmapped_ignorable(name)

    # ------------------------------------------------------------------
    # Executability
    # ------------------------------------------------------------------

    def assert_quant_types_executable(self) -> None:
        """Fail at load time, not inside a CUDA kernel, on a GGML type this
        stack has no path for.

        The published UD-* quantizations store the routed ``down`` projection
        (and, on layer 26, ``gate``/``up`` as well) as MXFP4, the model's native
        expert dtype. MXFP4 is in none of the GGUF type sets and in no
        dequantize/MMQ/MMVQ kernel case, so without a repair the failure
        surfaces as a bare ``NotImplementedError`` from ``fused_mul_mat_gguf``
        after the whole 119 GiB file has been read.

        The repair is a value-exact load-time repack to Q5_0 in the weight
        stream (``gguf_mxfp4_repack``), so MXFP4 is no longer an offender here
        unless ``SGLANG_GGUF_MXFP4_REPACK=0`` switched the repack off. This gate
        keeps its job for every type the repack does not cover.
        """
        from sglang.srt.model_loader.gguf_mxfp4_repack import repack_enabled
        from sglang.srt.model_loader.gguf_shards import iter_gguf_tensors

        supported = _supported_ggml_types()
        offenders: Dict[str, List[str]] = {}
        for tensor in iter_gguf_tensors(self.shard_paths()):
            ttype = tensor.tensor_type
            if ttype not in supported:
                offenders.setdefault(ttype.name, []).append(str(tensor.name))
        if not offenders:
            return
        detail = "; ".join(
            f"{tname}: {len(names)} tensors (e.g. {sorted(names)[0]})"
            for tname, names in sorted(offenders.items())
        )
        hint = (
            " MXFP4 is among them only because SGLANG_GGUF_MXFP4_REPACK=0 "
            "disabled the value-exact load-time repack to Q5_0; unset that "
            "variable to load this checkpoint."
            if "MXFP4" in offenders and not repack_enabled()
            else ""
        )
        raise RuntimeError(
            f"{self.FAMILY} GGUF {self.gguf_file} uses GGML type(s) this build "
            f"cannot execute -- {detail}. sglang's GGUF path dispatches only on "
            "the types in layers/quantization/gguf.py (standard, K-quant, "
            "I-matrix) plus F32/F16/BF16/I32, plus whatever the load-time "
            "repack in model_loader/gguf_mxfp4_repack.py converts into that "
            "set; there is no dequantize or matmul kernel for the others. Use "
            "a quantization whose tensors are all within that set, or add the "
            f"missing type to the GGUF kernels first.{hint}"
        )

    # ------------------------------------------------------------------
    # Weight stream
    # ------------------------------------------------------------------

    def _target_dtype(self) -> torch.dtype:
        """The model's compute dtype, for tensors this adapter dequantizes."""
        dtype = getattr(self.config, "torch_dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str) and hasattr(torch, dtype):
            return getattr(torch, dtype)
        return torch.get_default_dtype()

    def transform_stream(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """Repair the three places where the generic GGUF iterator's
        assumptions do not hold for this family.

        Everything else -- including the already-split per-expert names the
        iterator emits for the stacked ``ffn_*_exps`` tensors -- passes through
        untouched. In particular the OUTPUT projection (``head``) is left
        packed: ``tie_word_embeddings`` is false for this family, so the head
        is its own ``ParallelLMHead`` built with the model's quant_config, and
        a quantized-resident vocab head is this stack's GGUF default.
        """
        embed_qtype: Optional[int] = None
        for name, tensor in weights:
            # (1) Token embedding. `DeepseekV4Model` builds `embed_tokens` as a
            # VocabParallelEmbedding with NO quant_config, i.e. a plain dense
            # `.weight` -- while token_embd is Q8_0 in the published export and
            # so arrives as `embed.qweight` plus a marker. Nothing downstream
            # can put a packed payload into a dense embedding, and before this
            # dequant the two markers were simply reported missing and the
            # embedding stayed at its uninitialized values (#391 wall 10).
            # Same shape of repair as the gemma4 adapter's, for the same
            # reason.
            if name.startswith(_EMBED_PREFIX):
                leaf = name[len(_EMBED_PREFIX) :]
                if leaf == "qweight_type":
                    embed_qtype = int(tensor.item())
                    continue
                if leaf == "qweight":
                    assert (
                        embed_qtype is not None
                    ), "embed qweight arrived before its qweight_type marker"
                    yield (
                        _EMBED_PREFIX + "weight",
                        _dequantize_to(tensor, embed_qtype, self._target_dtype()),
                    )
                    continue
                # An F32 export already ships `embed.weight`: fall through.

            # (2) Integer routing table. The iterator treats every non-F32
            # tensor as quantized and emits a `qweight_type` marker before the
            # payload, deriving both names from the dense one.
            # `...gate.tid2eid` has no `.weight` leaf to rename, so BOTH
            # emissions arrive under the SAME name and the 0-dim type marker
            # would overwrite the table. Drop the marker; the table is a plain
            # int32 tensor.
            if any(name.endswith(suffix) for suffix in _INTEGER_TABLE_SUFFIXES):
                if tensor.dim() == 0:
                    continue
                yield name, tensor
                continue

            # The general form of the collision the branch above handles: a
            # name with no `.weight` leaf gets no rename, so a quantized
            # payload and its 0-dim marker arrive under ONE name and one
            # silently replaces the other. Every such tensor in the published
            # export is F32 (attention sinks, compressor APEs, the
            # hyper-connection triples) and F32 emits no marker at all, so this
            # can only fire on a re-export that quantizes one of them. Refuse
            # by name instead of loading whichever of the two arrived last.
            if tensor.dim() == 0 and not name.endswith(".qweight_type"):
                raise RuntimeError(
                    f"{self.FAMILY} GGUF: {name!r} is a bare parameter (no "
                    "`.weight` leaf to rename) but is stored quantized, so its "
                    "ggml type marker and its payload both arrive under that "
                    "one name and overwrite each other. Export it as F32, or "
                    "give this tensor an explicit case in transform_stream."
                )

            # (3) BF16 router gate. The iterator renames every non-F32 tensor
            # to `.qweight` and hands over the raw little-endian bytes as
            # uint8. The DeepSeek V4 router gate is a plain parameter, not a
            # GGUF-quantized linear, so it must arrive as `.weight` holding
            # real bf16 values.
            if name.endswith(".gate.qweight_type"):
                continue
            if name.endswith(".gate.qweight"):
                yield name[: -len(".qweight")] + ".weight", _bytes_to_bf16(tensor)
                continue

            yield name, tensor

    # ------------------------------------------------------------------
    # F32 carve-out
    # ------------------------------------------------------------------

    def _extend_unquantized_prefixes(
        self, prefixes: set, name_map: Dict[str, str], types: Dict[str, object]
    ) -> None:
        # The base class's generic carve-out keys off "proj" in the module
        # name, which is a Qwen/Gemma spelling. DeepSeek V4 spells its
        # projections wq_a / wq_b / wkv / wo_a / wo_b / wgate / w1 / w2 / w3,
        # so add every F32 linear under those spellings explicitly.
        import gguf

        f32 = gguf.GGMLQuantizationType.F32
        for gguf_name, native in name_map.items():
            if not native.endswith(".weight"):
                continue
            if types.get(gguf_name) != f32:
                continue
            base = native[: -len(".weight")]
            if base.rsplit(".", 1)[-1] in _DEEPSEEK4_LINEAR_LEAVES:
                prefixes.add(base)


#: Leaf module names that are Linear projections in DeepSeek V4's spelling.
#: Used only by the F32 carve-out; norms and bare parameters are excluded on
#: purpose (they are not GGUF-quantized layers).
_DEEPSEEK4_LINEAR_LEAVES = frozenset(
    {
        "wq_a",
        "wq_b",
        "wkv",
        "wo_a",
        "wo_b",
        "wgate",
        "weights_proj",
        "w1",
        "w2",
        "w3",
    }
)


def _dequantize_to(
    packed: torch.Tensor, qtype_value: int, dtype: torch.dtype
) -> torch.Tensor:
    """Unpack one GGUF payload into a dense tensor of *dtype*."""
    import gguf
    from gguf.quants import dequantize

    qtype = gguf.GGMLQuantizationType(qtype_value)
    if qtype == gguf.GGMLQuantizationType.BF16:
        # gguf-py hands BF16 back as raw uint8 with the last dim doubled.
        return _bytes_to_bf16(packed).to(dtype)
    return torch.from_numpy(dequantize(packed.numpy(), qtype).copy()).to(dtype)


def _bytes_to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret the raw uint8 payload gguf-py returns for a BF16 tensor.

    gguf-py hands BF16 back as uint8 with the last dimension doubled; a view
    is enough, no conversion. A tensor that already has a float dtype (a
    future gguf-py that decodes BF16 itself) is passed through.
    """
    if tensor.dtype != torch.uint8:
        return tensor
    return tensor.contiguous().view(torch.bfloat16)


def audit_name_map(gguf_file: str, config) -> Dict[str, str]:
    """Build the name map without the executability gate.

    Exposed for the offline audit (test/registered/unit/model_loader/
    test_gguf_deepseek4_name_map.py): a machine with the file but without the
    kernels can still check that the mapping covers it.
    """
    adapter = Deepseek4GGUFAdapter(config, gguf_file)
    return adapter._build_name_map_unchecked()
