# SPDX-License-Identifier: Apache-2.0
"""Shared scaffolding for the bespoke GGUF family adapters (#129 S2).

The Qwen3.5/3.6 and Gemma-4 GGUF adapters need bespoke name maps + llama.cpp
inverse weight transforms that sglang's generic GGUF path cannot express.  Both
families share a large amount of *quality/speed-neutral* scaffolding — GGUF arch
resolution, the file-tensor probe, the present-in-file emit/audit skeleton of
``build_name_map``, and the F32-carve-out skeleton of
``unquantized_module_prefixes``.  This base class factors out exactly that
neutral machinery and drives it from per-family DATA (the emit *tables*) and a
handful of small *hooks*.

What stays PER-FAMILY and byte-exact (never generalized here):
  * ``transform_stream`` — each family keeps its own verbatim body (the abstract
    method below). The load-time value transforms (qwen35's GDN retiling / norm
    ``-1`` un-bake / out_proj byte-vs-element un-tile; gemma4's identity-norm +
    embed dequant) are the whole point of the bespoke path and must not move.
  * The emit tables (``GLOBAL_ENTRIES`` / ``LAYER_ENTRIES``), the fuse map, and
    every hook override — they encode each family's exact tensor set + spelling.

Adding a family = a new module with an adapter subclass (tables + hooks + its
verbatim ``transform_stream``) plus one line in ``gguf_registry._FAMILIES``.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# A per-layer / global emit entry is ``(map_base, gguf_suffix)`` or
# ``(map_base, gguf_suffix, hf_suffix)``.  ``map_base`` is the gguf-py tensor-map
# spelling (for per-layer entries it is relative to ``model.layers.{i}.``);
# ``gguf_suffix`` is the GGUF tensor suffix ("" == a bare param such as
# ``A_log``); ``hf_suffix`` (default: same as ``gguf_suffix``) is the HF param
# suffix ("" lets e.g. ``layer_scalar`` map a ``.weight`` GGUF tensor onto a bare
# HF buffer).
EmitEntry = Tuple[str, ...]


class GGUFAdapterBase:
    """Common scaffolding for the bespoke GGUF family adapters.

    Subclasses provide:
      * class attrs ``FAMILY`` (log/error label), ``MODEL_TYPE_TO_ARCH``
        (HF model_type -> GGUF arch), ``GLOBAL_ENTRIES`` / ``LAYER_ENTRIES``
        (the emit tables), ``FUSE_MAP`` (split-proj -> fused-module folding for
        the F32 carve-out);
      * a verbatim ``transform_stream``;
      * optional hook overrides for the irreducible per-family bits.
    """

    # ---- required per-family class data (subclasses override) --------------
    FAMILY: str = ""
    MODEL_TYPE_TO_ARCH: Dict[str, str] = {}
    GLOBAL_ENTRIES: Tuple[EmitEntry, ...] = ()
    LAYER_ENTRIES: Tuple[EmitEntry, ...] = ()
    FUSE_MAP: Dict[str, str] = {}

    # ---- neutral defaults (qwen35 overrides via _post_init) ----------------
    #: A multimodal backbone with an mmproj GGUF beside it sets this (qwen35);
    #: gemma4 vision is deferred, so it stays None and the loader stays
    #: text-only.
    mmproj_path: Optional[object] = None
    #: NEXTN/MTP draft flag (qwen35); gemma4 has no draft GGUF path.
    is_draft: bool = False

    def __init__(self, config, gguf_file: str):
        # ``config`` may be the multimodal wrapper or the text config directly;
        # layer count / GDN dims live on the text config.
        text_config = (
            config.get_text_config() if hasattr(config, "get_text_config") else config
        )
        self.config = text_config
        self.gguf_file = gguf_file
        self.arch = self._resolve_arch(config, text_config)
        self.num_layers = int(text_config.num_hidden_layers)
        self._file_tensor_names = None  # set lazily
        self._shard_paths: Optional[List[str]] = None  # set lazily
        self._post_init(config)

    # ------------------------------------------------------------------
    # Hooks (neutral defaults; subclasses override the irreducible bits)
    # ------------------------------------------------------------------

    def _resolve_arch(self, config, text_config) -> str:
        """HF model_type -> GGUF arch string.  Default: prefer the wrapper's
        ``model_type``, fall back to the text config's (handles the multimodal
        wrapper whose top-level type is not in the map)."""
        model_type = getattr(config, "model_type", None)
        if model_type not in self.MODEL_TYPE_TO_ARCH:
            model_type = text_config.model_type
        return self.MODEL_TYPE_TO_ARCH[model_type]

    def _post_init(self, config) -> None:
        """Family-specific attribute setup after the common __init__.  Default
        no-op (gemma4); qwen35 sets its draft/vision/GDN-dim attributes here."""

    def _pre_map_hook(self, file_tensors: set) -> None:
        """Runs before the emit loop (e.g. gemma4's MoE fail-fast).  No-op."""

    def _map_base_to_hf(self, map_base: str) -> str:
        """gguf-py map spelling -> HF param spelling.  Default identity
        (qwen35); gemma4 rewrites ``model.`` -> ``model.language_model.``."""
        return map_base

    def _emit_fallback(
        self, map_base: str, gguf_suffix: str, file_tensors: set, out: Dict[str, str]
    ) -> None:
        """Called when gguf-py's tensor map has no entry for ``map_base`` (e.g.
        qwen35's ``linear_attn.dt_bias`` spelling gap).  Default no-op."""

    def _is_unmapped_ignorable(self, name: str) -> bool:
        """True for a file tensor that is legitimately not in the name map (so
        the unmapped-tensor audit ignores it).  Default: vision/mmproj only."""
        return name.startswith(("v.", "mm."))

    def _module_prefix_spelling(self, base: str) -> str:
        """Fused-module base -> the spelling ``is_layer_skipped_gguf`` matches.
        Default identity (qwen35); gemma4 strips the leading ``model.``."""
        return base

    def _extend_unquantized_prefixes(
        self, prefixes: set, name_map: Dict[str, str], types: Dict[str, object]
    ) -> None:
        """Add any family-specific unquantized modules beyond the generic F32
        carve-out (qwen35's block-misaligned/dense out_proj + dense-vocab
        lm_head).  Default no-op."""

    # ------------------------------------------------------------------
    # transform_stream is irreducibly per-family (verbatim in each subclass)
    # ------------------------------------------------------------------

    def transform_stream(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared machinery
    # ------------------------------------------------------------------

    def shard_paths(self) -> List[str]:
        """Every file of this checkpoint, in shard order.

        A plain GGUF is a one-element list, so every caller below is written
        against the set and the unsplit case falls out of it.
        """
        from sglang.srt.model_loader.gguf_shards import resolve_gguf_shard_paths

        if self._shard_paths is None:
            self._shard_paths = resolve_gguf_shard_paths(self.gguf_file)
        return self._shard_paths

    def _file_tensors(self) -> set:
        """Union of the tensor names over the whole shard set.

        The union, not part 1's tensors: llama.cpp puts the KV block in part 1
        and (for a large export) every tensor in the later parts, so the
        unmapped-tensor audit run against part 1 alone would audit nothing.
        """
        from sglang.srt.model_loader.gguf_shards import iter_gguf_tensors

        if self._file_tensor_names is None:
            self._file_tensor_names = {
                str(t.name) for t in iter_gguf_tensors(self.shard_paths())
            }
        return self._file_tensor_names

    def _resolve_arch_enum(self):
        import gguf

        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == self.arch:
                return key
        raise RuntimeError(f"gguf-py has no arch {self.arch!r}")

    def build_name_map(self) -> Dict[str, str]:
        """Return ``{gguf_tensor_name: hf_param_name}`` for every tensor present
        in the file.  Generic emit/audit skeleton driven by the per-family
        tables + hooks; present-in-file filtering auto-classifies layers."""
        if self.is_draft:
            return self._build_draft_name_map()
        import gguf

        arch_enum = self._resolve_arch_enum()
        name_map = gguf.get_tensor_name_map(arch_enum, self.num_layers)
        file_tensors = self._file_tensors()
        self._pre_map_hook(file_tensors)

        gguf_to_hf: Dict[str, str] = {}

        def emit(map_base: str, gguf_suffix: str, hf_suffix: Optional[str] = None):
            gguf_base = name_map.get_name(map_base)
            if gguf_base is None:
                self._emit_fallback(map_base, gguf_suffix, file_tensors, gguf_to_hf)
                return
            gguf_full = gguf_base + ("." + gguf_suffix if gguf_suffix else "")
            if gguf_full not in file_tensors:
                return
            hs = gguf_suffix if hf_suffix is None else hf_suffix
            hf_base = self._map_base_to_hf(map_base)
            gguf_to_hf[gguf_full] = hf_base + ("." + hs if hs else "")

        for entry in self.GLOBAL_ENTRIES:
            emit(*entry)
        for i in range(self.num_layers):
            p = f"model.layers.{i}."
            for entry in self.LAYER_ENTRIES:
                emit(p + entry[0], *entry[1:])

        unmapped = [
            n
            for n in file_tensors
            if n not in gguf_to_hf and not self._is_unmapped_ignorable(n)
        ]
        if unmapped:
            raise RuntimeError(
                f"{self.FAMILY} GGUF: {len(unmapped)} tensors not mapped: "
                f"{sorted(unmapped)[:12]}"
            )
        logger.info(
            "%s GGUF name map: %d tensors for %d layers (arch %s)",
            self.FAMILY,
            len(gguf_to_hf),
            self.num_layers,
            self.arch,
        )
        return gguf_to_hf

    def _build_draft_name_map(self) -> Dict[str, str]:
        """Only families with a NEXTN/MTP draft GGUF path override this."""
        raise NotImplementedError(f"{self.FAMILY} has no draft name map")

    def unquantized_module_prefixes(self) -> List[str]:
        """Linear modules whose GGUF tensors are stored F32 (the GGUF iterator
        keeps F32 as a plain ``.weight``, which cannot load into a
        GGUF-quantized layer) and must therefore be built unquantized.  Dense
        F16/BF16 shards stay quantized-resident (mixed-dtype, task #64).

        Generic F32 carve-out with the per-family fuse map + spelling hook and
        the "F32 split beside a quantized sibling" fail-fast; families extend it
        via ``_extend_unquantized_prefixes``.
        """
        import gguf

        from sglang.srt.model_loader.gguf_shards import iter_gguf_tensors

        types = {
            str(t.name): t.tensor_type for t in iter_gguf_tensors(self.shard_paths())
        }
        unq_types = {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
        }
        f32_type = gguf.GGMLQuantizationType.F32
        name_map = self.build_name_map()
        prefixes = set()
        module_split_types: Dict[str, set] = {}
        for gname, hf in name_map.items():
            if not hf.endswith(".weight"):
                continue
            base = hf[: -len(".weight")]
            # Only Linear projections matter (norms/conv1d are not GGUF-quantized
            # linears); key off "proj".
            if "proj" not in base:
                continue
            for split, fused in self.FUSE_MAP.items():
                if base.endswith(split):
                    base = base[: -len(split)] + fused
                    break
            base = self._module_prefix_spelling(base)
            module_split_types.setdefault(base, set()).add(types.get(gname))
            if types.get(gname) == f32_type:
                prefixes.add(base)
        for base in sorted(prefixes):
            tset = module_split_types.get(base, set())
            if any(t is not None and t not in unq_types for t in tset):
                raise RuntimeError(
                    f"{self.FAMILY} GGUF: module {base!r} mixes an F32 split "
                    f"with a quantized sibling "
                    f"({sorted(t.name for t in tset if t is not None)}); "
                    "this cannot be loaded either dense or quantized. "
                    "Re-quantize the F32 tensor (e.g. to Q8_0/F16)."
                )
        self._extend_unquantized_prefixes(prefixes, name_map, types)
        return sorted(prefixes)
