# SPDX-License-Identifier: Apache-2.0
"""GGUF loading support for the Qwen3.5/Qwen3.6 family (GGUF arch ``qwen35`` /
``qwen35moe``).

Ported and adapted from ``efschu/vllm-gguf-plugin`` (branch ``qwen35-support``,
``weights_adapter/qwen35.py``).  These hybrid GDN + full-attention models cannot
be loaded by sglang's generic GGUF path for three reasons:

1. **Name mapping.**  ``model_type`` ``qwen3_5``/``qwen3_5_moe`` map to GGUF arch
   ``qwen35``/``qwen35moe`` (the generic reverse lookup over
   ``gguf.MODEL_ARCH_NAMES`` fails), and gguf-py's tensor map does not know the
   ``linear_attn.dt_bias`` spelling (only the older ``dt_proj``); suffix-less
   params such as ``linear_attn.A_log`` need exact GGUF tensor names.  We build
   the map from a *template* of HF param names (no ``transformers`` model
   instantiation — the reference ``AutoModelForCausalLM.from_config`` path is
   fragile here, it trips over a missing ``layer_types`` attribute).

2. **Weight-value transforms.**  llama.cpp bakes ``+1`` into the Gemma-style
   RMSNorm gammas, stores ``ssm_a = -exp(A_log)``, reshapes conv1d, and — when
   ``num_v_heads != num_k_heads`` — re-tiles every GDN v-head dimension from
   grouped to tiled order.  All of these must be inverted on load.

3. **Layer classification.**  A qwen35 checkpoint interleaves GDN linear-attn
   layers (``ssm_*`` tensors) with periodic full-attention layers
   (``attn_q/k/v`` tensors).  We classify each layer from the tensors actually
   present in the file rather than from config metadata.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# HF model_type -> GGUF arch name (values as in gguf.MODEL_ARCH_NAMES).
_MODEL_TYPE_TO_GGUF_ARCH = {
    "qwen3_5": "qwen35",
    "qwen3_5_text": "qwen35",
    "qwen3_5_moe": "qwen35moe",
    "qwen3_5_moe_text": "qwen35moe",
}


def is_qwen35_gguf_arch(model_type: Optional[str]) -> bool:
    return model_type in _MODEL_TYPE_TO_GGUF_ARCH


# Gemma-style RMSNorm gammas (implemented as ``x * (1 + w)`` with zero-centered
# weights).  llama.cpp bakes the +1 into the GGUF, so it must be subtracted.
# ``linear_attn.norm`` is RMSNormGated and stored raw in both formats.
_GEMMA_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)
_GEMMA_NORM_NAMES = (
    "model.norm.weight",
    # NEXTN/MTP draft norms (all GemmaRMSNorm): shared_head_norm -> mtp.norm,
    # enorm -> pre_fc_norm_embedding, hnorm -> pre_fc_norm_hidden. (The MTP
    # decoder-layer norms already match _GEMMA_NORM_SUFFIXES.)
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)


class Qwen35GGUFAdapter:
    """Builds the GGUF<->HF name map and applies llama.cpp inverse transforms
    for one qwen35/qwen35moe checkpoint."""

    def __init__(self, config, gguf_file: str):
        # ``config`` may be the multimodal wrapper (Qwen3_5Config) or the text
        # config directly; the GDN dims + num_hidden_layers live on the text
        # config.
        text_config = (
            config.get_text_config() if hasattr(config, "get_text_config") else config
        )
        self.config = text_config
        self.gguf_file = gguf_file
        self.arch = _MODEL_TYPE_TO_GGUF_ARCH[text_config.model_type]
        self.num_layers = int(text_config.num_hidden_layers)
        # NEXTN/MTP draft: the model loader rewrites the draft ModelConfig's
        # architecture to "Qwen3_5ForCausalLMMTP" (model_type stays qwen3_5), so
        # the same adapter is reused but must emit the MTP-block name map. The
        # MTP block is stored as blk.<num_hidden_layers>.
        archs = getattr(config, "architectures", None) or []
        self.is_draft = archs[:1] == ["Qwen3_5ForCausalLMMTP"]
        self.num_k = int(getattr(text_config, "linear_num_key_heads", 0) or 0)
        self.num_v = int(getattr(text_config, "linear_num_value_heads", 0) or 0)
        self.head_k_dim = int(getattr(text_config, "linear_key_head_dim", 0) or 0)
        self.head_v_dim = int(getattr(text_config, "linear_value_head_dim", 0) or 0)
        self._file_tensor_names = None  # set lazily

    # ------------------------------------------------------------------
    # Name mapping
    # ------------------------------------------------------------------

    def _file_tensors(self) -> set:
        if self._file_tensor_names is None:
            import gguf

            reader = gguf.GGUFReader(self.gguf_file)
            self._file_tensor_names = {t.name for t in reader.tensors}
        return self._file_tensor_names

    def build_name_map(self) -> Dict[str, str]:
        """Return ``{gguf_tensor_name: hf_param_name}`` for every tensor present
        in the file.  HF names use the split GDN spelling
        (``in_proj_qkv/z/b/a``); the model's ``stacked_params_mapping`` fuses
        them into ``in_proj_qkvz`` / ``in_proj_ba`` at load time."""
        if self.is_draft:
            return self._build_mtp_name_map()
        import gguf

        arch_enum = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == self.arch:
                arch_enum = key
                break
        if arch_enum is None:
            raise RuntimeError(f"gguf-py has no arch {self.arch!r}")
        name_map = gguf.get_tensor_name_map(arch_enum, self.num_layers)
        file_tensors = self._file_tensors()

        # (hf_base, hf_suffix) candidates. hf_suffix "" == bare param (A_log,
        # dt_bias). Both GDN and full-attn per-layer bases are generated; only
        # those whose GGUF tensor actually exists in the file are kept, which
        # auto-classifies each layer.
        gguf_to_hf: Dict[str, str] = {}

        def emit(hf_base: str, suffix: str) -> None:
            gguf_base = name_map.get_name(hf_base)
            if gguf_base is None:
                # gguf-py knows ``dt_proj`` but not the ``dt_bias`` spelling.
                if hf_base.endswith("linear_attn.dt_bias"):
                    layer = hf_base.split(".layers.")[1].split(".")[0]
                    gguf_full = f"blk.{layer}.ssm_dt.bias"
                    if gguf_full in file_tensors:
                        gguf_to_hf[gguf_full] = hf_base  # bare param
                return
            gguf_full = gguf_base + ("." + suffix if suffix else "")
            if gguf_full in file_tensors:
                gguf_to_hf[gguf_full] = (
                    hf_base + ("." + suffix if suffix else "")
                )

        # Global tensors.
        emit("model.embed_tokens", "weight")
        emit("lm_head", "weight")
        emit("model.norm", "weight")

        for i in range(self.num_layers):
            p = f"model.layers.{i}."
            # shared by both layer kinds
            emit(p + "input_layernorm", "weight")
            emit(p + "post_attention_layernorm", "weight")
            # GDN linear-attention layer
            emit(p + "linear_attn.in_proj_qkv", "weight")
            emit(p + "linear_attn.in_proj_z", "weight")
            emit(p + "linear_attn.in_proj_b", "weight")
            emit(p + "linear_attn.in_proj_a", "weight")
            emit(p + "linear_attn.A_log", "")
            emit(p + "linear_attn.dt_bias", "")
            emit(p + "linear_attn.conv1d", "weight")
            emit(p + "linear_attn.out_proj", "weight")
            emit(p + "linear_attn.norm", "weight")
            # full-attention layer
            emit(p + "self_attn.q_proj", "weight")
            emit(p + "self_attn.k_proj", "weight")
            emit(p + "self_attn.v_proj", "weight")
            emit(p + "self_attn.o_proj", "weight")
            emit(p + "self_attn.q_norm", "weight")
            emit(p + "self_attn.k_norm", "weight")
            # MLP (dense)
            emit(p + "mlp.gate_proj", "weight")
            emit(p + "mlp.up_proj", "weight")
            emit(p + "mlp.down_proj", "weight")
            # MoE experts (fused single tensor -> expert 0; loader side-loads)
            emit(p + "mlp.experts.0.gate_proj", "weight")
            emit(p + "mlp.experts.0.up_proj", "weight")
            emit(p + "mlp.experts.0.down_proj", "weight")
            emit(p + "mlp.gate", "weight")

        # Report any file tensor we failed to map. Excluded: vision/mmproj
        # (v./mm.) and the extra MTP/NEXTN block(s) at blk.>=num_hidden_layers
        # (loaded separately by the NEXTN draft; see _build_mtp_name_map).
        def _blk_index(n: str):
            if n.startswith("blk."):
                try:
                    return int(n.split(".")[1])
                except (IndexError, ValueError):
                    return None
            return None

        unmapped = [
            n
            for n in file_tensors
            if n not in gguf_to_hf
            and not n.startswith(("v.", "mm."))
            and not (
                (bi := _blk_index(n)) is not None and bi >= self.num_layers
            )
        ]
        if unmapped:
            raise RuntimeError(
                f"qwen35 GGUF: {len(unmapped)} tensors not mapped: "
                f"{sorted(unmapped)[:12]}"
            )
        logger.info(
            "qwen35 GGUF name map: %d tensors for %d layers (arch %s)",
            len(gguf_to_hf),
            self.num_layers,
            self.arch,
        )
        return gguf_to_hf

    # GGUF MTP-block tensor suffix -> HF (checkpoint-style) name that
    # Qwen3_5ForCausalLMMTP.load_weights expects. token_embd/output are NOT
    # listed: the NEXTN worker injects embed_tokens/lm_head from the target via
    # set_embed_and_head. The MTP decoder layer is full-attention (no GDN).
    _MTP_TENSOR_MAP = {
        "nextn.eh_proj.weight": "mtp.fc.weight",
        "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
        "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
        "nextn.shared_head_norm.weight": "mtp.norm.weight",
        "attn_q.weight": "mtp.layers.0.self_attn.q_proj.weight",
        "attn_k.weight": "mtp.layers.0.self_attn.k_proj.weight",
        "attn_v.weight": "mtp.layers.0.self_attn.v_proj.weight",
        "attn_output.weight": "mtp.layers.0.self_attn.o_proj.weight",
        "attn_q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
        "attn_k_norm.weight": "mtp.layers.0.self_attn.k_norm.weight",
        "attn_norm.weight": "mtp.layers.0.input_layernorm.weight",
        "post_attention_norm.weight": "mtp.layers.0.post_attention_layernorm.weight",
        "ffn_gate.weight": "mtp.layers.0.mlp.gate_proj.weight",
        "ffn_up.weight": "mtp.layers.0.mlp.up_proj.weight",
        "ffn_down.weight": "mtp.layers.0.mlp.down_proj.weight",
    }

    def _build_mtp_name_map(self) -> Dict[str, str]:
        """Name map for the NEXTN/MTP draft: the single extra block at
        ``blk.<num_hidden_layers>`` -> ``mtp.*`` params of Qwen3_5ForCausalLMMTP."""
        n = self.num_layers
        file_tensors = self._file_tensors()
        gguf_to_hf: Dict[str, str] = {}
        for suffix, hf in self._MTP_TENSOR_MAP.items():
            gname = f"blk.{n}.{suffix}"
            if gname in file_tensors:
                gguf_to_hf[gname] = hf
        if not gguf_to_hf:
            raise RuntimeError(
                f"qwen35 MTP draft: no blk.{n}.* tensors found in {self.gguf_file}"
            )
        logger.info(
            "qwen35 GGUF MTP draft name map: %d tensors (blk.%d)",
            len(gguf_to_hf),
            n,
        )
        return gguf_to_hf

    def unquantized_module_prefixes(self) -> List[str]:
        """Linear modules whose GGUF tensors are stored unquantized (F32/F16/
        BF16) and must therefore be built with UnquantizedLinearMethod instead
        of GGUFLinearMethod.

        In qwen35 the GDN ``in_proj_ba`` (b/a projections, GGUF ``ssm_alpha`` /
        ``ssm_beta``) is F32 while every other projection is quantized; feeding
        an F32 ``.weight`` into a GGUF-quantized layer (which expects
        ``qweight``) would fail. Detected from the file so other unquantized
        projections (if any) are also covered. Returned as substrings for
        ``is_layer_skipped_gguf`` (which does a substring match on the prefix).
        """
        import gguf

        reader = gguf.GGUFReader(self.gguf_file)
        types = {t.name: t.tensor_type for t in reader.tensors}
        unq_types = {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
        }
        # Map split GDN spellings back to the fused module built by the model.
        fuse = {
            "linear_attn.in_proj_b": "linear_attn.in_proj_ba",
            "linear_attn.in_proj_a": "linear_attn.in_proj_ba",
            "linear_attn.in_proj_qkv": "linear_attn.in_proj_qkvz",
            "linear_attn.in_proj_z": "linear_attn.in_proj_qkvz",
        }
        prefixes = set()
        name_map = self.build_name_map()
        for gname, hf in name_map.items():
            if types.get(gname) not in unq_types or not hf.endswith(".weight"):
                continue
            base = hf[: -len(".weight")]
            # Only Linear projections matter (norms/conv1d are not GGUF-quantized
            # linears); key off "proj".
            if "proj" not in base:
                continue
            for split, fused in fuse.items():
                if base.endswith(split):
                    base = base[: -len(split)] + fused
                    break
            prefixes.add(base)

        # GDN out_proj whose value head is NOT a multiple of the ggml block
        # (e.g. K-quant block 256 with head_v_dim 128) is dequantized on the fly
        # in transform_stream (the raw-byte column permutation would corrupt it),
        # so the layer must also be built unquantized. Q8_0 out_proj (block 32,
        # head_v_dim 128) stays quantized and is NOT listed here.
        for gname, hf in name_map.items():
            if not hf.endswith("linear_attn.out_proj.weight"):
                continue
            gt = types.get(gname)
            if gt in unq_types or (
                gt is not None and self._out_proj_dequant_needed(int(gt))
            ):
                prefixes.add(hf[: -len(".weight")])

        # lm_head must be a DENSE weight for GGUF: NEXTN spec decode shares the
        # target's lm_head.weight with the draft (get_embed_and_head), and a
        # quantized lm_head (GGUFEmbeddingMethod) has only `qweight`. The main
        # model dequantizes `output`->lm_head.weight (transform_stream); the
        # draft's lm_head is injected from the target.
        prefixes.add("lm_head")
        return sorted(prefixes)

    # ------------------------------------------------------------------
    # Weight-value transforms
    # ------------------------------------------------------------------

    def _v_retiling_active(self) -> bool:
        return self.num_k > 0 and self.num_v > 0 and self.num_k != self.num_v

    def _undo_v_tiling(
        self, weight: torch.Tensor, dim: int, head_units: int
    ) -> torch.Tensor:
        """Invert llama.cpp's GDN v-head reorder (grouped -> tiled)."""
        if not self._v_retiling_active():
            return weight
        num_v_per_k = self.num_v // self.num_k
        shape = list(weight.shape)
        d = dim if dim >= 0 else dim + len(shape)
        new_shape = shape[:d] + [num_v_per_k, self.num_k, head_units] + shape[d + 1 :]
        weight = weight.reshape(*new_shape)
        perm = list(range(len(new_shape)))
        perm[d], perm[d + 1] = perm[d + 1], perm[d]
        return weight.permute(*perm).contiguous().reshape(*shape)

    def _out_proj_dequant_needed(self, qtype: int) -> bool:
        if not self._v_retiling_active():
            return False
        import gguf

        block_size, _ = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(qtype)]
        return self.head_v_dim % block_size != 0

    @staticmethod
    def _is_dense_gguf_type(qtype: int) -> bool:
        """True for GGUF "unquantized" tensor types (F16/BF16/F32), whose data
        is a plain element matrix rather than block-packed quantized bytes.

        This matters for out_proj (see ``_undo_gdn_reorder`` / ``transform_stream``):
        heretic-style GGUFs store GDN out_proj as a quantized type (e.g. Q8_0,
        block 32) whose ``.qweight`` columns are RAW BYTES, so the v-head un-tile
        must operate on byte-granularity (``head_bytes``).  unsloth "UD"
        (Ultra-Dynamic) quants instead KEEP out_proj as dense F16, but sglang's
        GGUF iterator still renames any non-F32 tensor ``.weight`` -> ``.qweight``
        (+ a ``.qweight_type``).  A dense F16 out_proj therefore reaches the
        ``.qweight`` code path even though its columns are plain value-dim
        ELEMENTS, not packed bytes -- it must be un-tiled element-wise with
        ``head_v_dim`` (exactly like the dense ``out_proj.weight`` path), NOT with
        a byte stride.  (F32 tensors are never renamed to ``.qweight`` by the
        iterator, so they only appear here for completeness.)
        """
        import gguf

        return gguf.GGMLQuantizationType(qtype) in (
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
            gguf.GGMLQuantizationType.F32,
        )

    def _undo_gdn_reorder(
        self, hf_name: str, weight: torch.Tensor, qweight_types: Dict[str, int]
    ) -> torch.Tensor:
        import gguf

        num_k = self.num_k
        head_k_dim = self.head_k_dim
        head_v_dim = self.head_v_dim
        if hf_name.endswith(("in_proj_qkv.qweight", "in_proj_qkv.weight")):
            qk_rows = 2 * head_k_dim * num_k
            v_part = self._undo_v_tiling(weight[qk_rows:], 0, head_v_dim)
            return torch.cat([weight[:qk_rows], v_part], dim=0)
        if hf_name.endswith(("in_proj_z.qweight", "in_proj_z.weight")):
            return self._undo_v_tiling(weight, 0, head_v_dim)
        if hf_name.endswith(
            (
                "in_proj_b.qweight",
                "in_proj_b.weight",
                "in_proj_a.qweight",
                "in_proj_a.weight",
            )
        ):
            return self._undo_v_tiling(weight, 0, 1)
        if hf_name.endswith(("linear_attn.A_log", "linear_attn.dt_bias")):
            return self._undo_v_tiling(weight.unsqueeze(-1), 0, 1).squeeze(-1)
        if hf_name.endswith("conv1d.weight"):
            qk_channels = 2 * head_k_dim * num_k
            v_part = self._undo_v_tiling(weight[qk_channels:], 0, head_v_dim)
            return torch.cat([weight[:qk_channels], v_part], dim=0)
        if hf_name.endswith("out_proj.qweight"):
            qtype = qweight_types.get(hf_name)
            assert qtype is not None, "out_proj qweight_type not seen yet"
            block_size, type_size = gguf.GGML_QUANT_SIZES[
                gguf.GGMLQuantizationType(qtype)
            ]
            if head_v_dim % block_size != 0:
                raise ValueError(
                    f"Cannot undo GDN v-head reorder of {hf_name}: head_v_dim="
                    f"{head_v_dim} is not a multiple of the "
                    f"{gguf.GGMLQuantizationType(qtype).name} block size "
                    f"{block_size}. Re-quantize out_proj with a block-aligned "
                    "type (e.g. Q8_0)."
                )
            head_bytes = head_v_dim // block_size * type_size
            return self._undo_v_tiling(weight, 1, head_bytes)
        if hf_name.endswith("out_proj.weight"):
            return self._undo_v_tiling(weight, 1, head_v_dim)
        return weight

    def transform_stream(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """Wrap the raw GGUF weight iterator, applying all inverse transforms.

        ``embed_tokens`` is created without a quant_config (plain ``weight``),
        so a quantized ``token_embd`` must be dequantized on the fly.
        """
        import gguf
        from gguf.quants import dequantize

        qweight_types: Dict[str, int] = {}

        # Modules built WITHOUT a quant_config expect a plain `.weight`, so a
        # quantized GGUF tensor for them must be dequantized on the fly:
        #  - main model: `model.embed_tokens` (lm_head stays quantized via
        #    GGUFEmbeddingMethod, so `output`->lm_head keeps its qweight).
        #  - MTP draft: `mtp.fc` (eh_proj, a plain nn.Linear); the draft's
        #    embed_tokens/lm_head are injected from the target, not loaded here.
        dequant_prefixes = (
            ("mtp.fc.",)
            if self.is_draft
            else ("model.embed_tokens.", "lm_head.")
        )
        deq_qtype: Dict[str, int] = {}
        for name, weight in weights:
            hit = next((p for p in dequant_prefixes if name.startswith(p)), None)
            if hit is not None:
                if name.endswith(".qweight_type"):
                    deq_qtype[hit] = int(weight.item())
                    continue
                if name.endswith(".qweight"):
                    deq = dequantize(
                        weight.numpy(), gguf.GGMLQuantizationType(deq_qtype[hit])
                    )
                    yield hit + "weight", torch.from_numpy(deq).to(
                        self.config.torch_dtype
                    )
                    continue
                # already unquantized (.weight): fall through, passed as-is.

            # --- track quant types so out_proj retiling can see them ---
            if name.endswith(".qweight_type"):
                qtype = int(weight.item())
                qweight_types[name.removesuffix("_type")] = qtype
                if name.endswith("linear_attn.out_proj.qweight_type") and (
                    self._out_proj_dequant_needed(qtype)
                    or self._is_dense_gguf_type(qtype)
                ):
                    # Layer is loaded unquantized (block-misaligned quant that we
                    # dequantize, OR a dense F16/BF16 out_proj as in unsloth "UD"
                    # quants); either way the module has a plain `.weight` param
                    # and no `.qweight_type`, so drop the type param.
                    continue
                yield name, weight
                continue

            # --- GDN out_proj: un-tile the v-head reorder ---
            if name.endswith("linear_attn.out_proj.qweight"):
                qtype = qweight_types.get(name)
                assert qtype is not None, "out_proj qweight_type not seen yet"
                if self._is_dense_gguf_type(qtype):
                    # unsloth "UD": out_proj kept DENSE (F16/BF16), but sglang's
                    # GGUF iterator renamed `.weight`->`.qweight`. The tensor is a
                    # plain element matrix, so un-tile its columns element-wise
                    # with head_v_dim (like the dense out_proj.weight path) and
                    # re-emit as `.weight` for the unquantized module. Do NOT run
                    # the byte-stride path below (it would compute head_units =
                    # head_v_dim * type_size and reshape to 2x the real size).
                    dense = self._undo_v_tiling(weight, 1, self.head_v_dim)
                    yield name.removesuffix(".qweight") + ".weight", dense.to(
                        self.config.torch_dtype
                    )
                    continue
                if self._out_proj_dequant_needed(qtype):
                    # Block-misaligned quantized out_proj (e.g. K-quant block 256
                    # with head_v_dim 128): dequantize, then element-wise un-tile.
                    deq = torch.from_numpy(
                        dequantize(weight.numpy(), gguf.GGMLQuantizationType(qtype))
                    )
                    deq = self._undo_v_tiling(deq, 1, self.head_v_dim)
                    yield name.removesuffix(".qweight") + ".weight", deq.to(
                        self.config.torch_dtype
                    )
                    continue
                # Block-aligned quantized out_proj (heretic's Q8_0, block 32):
                # `.qweight` columns are raw bytes; un-tile at byte granularity.
                yield name, self._undo_gdn_reorder(name, weight, qweight_types)
                continue

            # --- value transforms on plain (unquantized) tensors ---
            if name.endswith("conv1d.weight") and weight.dim() == 2:
                weight = weight.unsqueeze(1)
                weight = self._undo_gdn_reorder(name, weight, qweight_types)
                yield name, weight
                continue
            if name.endswith("linear_attn.A_log"):
                weight = torch.log(torch.neg(weight.float()))
                weight = self._undo_gdn_reorder(name, weight, qweight_types)
                yield name, weight
                continue
            if name.endswith("linear_attn.dt_bias"):
                yield name, self._undo_gdn_reorder(name, weight, qweight_types)
                continue
            if name.endswith(_GEMMA_NORM_SUFFIXES) or name in _GEMMA_NORM_NAMES:
                yield name, weight.float() - 1.0
                continue

            # --- remaining GDN in_proj tensors: retile to grouped v-head order.
            # Suffix-AGNOSTIC (mirrors the vLLM plugin): in_proj_b / in_proj_a
            # (GGUF ssm_beta/ssm_alpha) are F32 ".weight", not ".qweight", but
            # their value heads must be retiled just like every other V-indexed
            # GDN tensor — otherwise beta/alpha stay in llama.cpp tiled order
            # while q/k/v/z/A_log/dt_bias/conv1d/out_proj are grouped, pairing
            # each value head with the wrong gate -> incoherent output.
            # (conv1d/A_log/dt_bias/norms/out_proj already `continue`d above;
            # ssm_norm hits no _undo_gdn_reorder branch and passes through.)
            if ".linear_attn." in name:
                yield name, self._undo_gdn_reorder(name, weight, qweight_types)
                continue

            yield name, weight
