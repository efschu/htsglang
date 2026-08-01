# SPDX-License-Identifier: Apache-2.0
"""Registry of the bespoke GGUF family adapters (#129 S2).

sglang's generic GGUF path cannot load a few families (Qwen3.5/3.6 hybrid GDN,
Gemma-4) — they need a bespoke name map + llama.cpp inverse weight transforms.
Each such family is an adapter subclass of ``GGUFAdapterBase`` living in its own
module; this registry is the single dispatch table that maps an HF ``model_type``
to its adapter.

Adding a family = write the adapter module (emit tables + hooks + its verbatim
``transform_stream``) and add ONE line to ``_FAMILIES`` below.  No edits to
``loader.py`` (uses :func:`create_gguf_adapter`) or to the config-peek (uses
:func:`sibling_config_gguf_archs`) are needed.

Not every checkpoint that needs the sibling config.json is a family: one whose
``model_type`` is shared with a whole generic family (the DFLASH drafter is
``qwen3``) must dispatch on its GGUF architecture instead, and contributes its
arch through ``_ARCH_ONLY_SIBLING_CONTRIBUTORS``.

Import discipline: this module stays import-light (importlib only).  The adapter
modules — which pull in torch/gguf — are imported LAZILY inside the functions so
importing the registry (e.g. from the early config-peek path) is cheap and
cannot create an import cycle.
"""

from __future__ import annotations

import importlib
import logging
import re
from typing import List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

# (family_name, module_path, adapter_class_name).  Order matters only for the
# arch-tuple ordering exposed by sibling_config_gguf_archs().
_FAMILIES: Tuple[Tuple[str, str, str], ...] = (
    ("qwen35", "sglang.srt.model_loader.gguf_qwen35", "Qwen35GGUFAdapter"),
    ("gemma4", "sglang.srt.model_loader.gguf_gemma4", "Gemma4GGUFAdapter"),
    ("deepseek4", "sglang.srt.model_loader.gguf_deepseek4", "Deepseek4GGUFAdapter"),
)

# (module_path, attribute) of checkpoints that need the sibling config.json but
# are NOT families: they dispatch on the HF ARCHITECTURE rather than on
# model_type, so they have no MODEL_TYPE_TO_ARCH to harvest.
#
# The DFLASH drafter is the case this exists for. Its config carries
# ``model_type: "qwen3"``, so registering it in _FAMILIES would capture every
# plain Qwen3 GGUF as well (gguf_dflash.py states this); its GGUF header, on
# the other hand, says "dflash-draft" and is unambiguous. The two halves were
# split before: the WEIGHT loader dispatches on the architecture and works, but
# the architecture is only readable once the sibling config.json has been read,
# and nothing routed the config peek there -- so the drafter died in
# transformers' GGUF reader before its own loader ever ran.
_ARCH_ONLY_SIBLING_CONTRIBUTORS: Tuple[Tuple[str, str], ...] = (
    ("sglang.srt.model_loader.gguf_dflash", "DFLASH_GGUF_ARCHS"),
)


def _iter_adapter_classes():
    """Yield ``(family_name, adapter_class)`` for every registered family.

    Lazily imports each adapter module (torch/gguf) on first use.
    """
    for family_name, module_path, class_name in _FAMILIES:
        module = importlib.import_module(module_path)
        yield family_name, getattr(module, class_name)


def get_gguf_adapter_class(model_type: Optional[str]) -> Optional[Type]:
    """Return the bespoke GGUF adapter class for ``model_type``, or None if the
    model_type belongs to no registered bespoke family (generic GGUF path)."""
    if model_type is None:
        return None
    for _family_name, cls in _iter_adapter_classes():
        if model_type in cls.MODEL_TYPE_TO_ARCH:
            return cls
    return None


def create_gguf_adapter(hf_config, gguf_file: str):
    """Instantiate the bespoke GGUF adapter for ``hf_config``'s model_type, or
    return None to leave the generic GGUF path unchanged."""
    model_type = getattr(hf_config, "model_type", None)
    cls = get_gguf_adapter_class(model_type)
    if cls is None:
        return None
    return cls(hf_config, gguf_file)


def sibling_config_gguf_archs() -> Tuple[str, ...]:
    """The set of GGUF ``general.architecture`` strings whose config/tokenizer
    must be read from the sibling files (config.json / tokenizer.json) instead
    of the GGUF metadata — the union of every bespoke family's GGUF archs plus
    the arch-dispatched contributors. Used by the config/tokenizer peek in
    ``utils/hf_transformers``.
    """
    archs: List[str] = []
    for _family_name, cls in _iter_adapter_classes():
        for arch in cls.MODEL_TYPE_TO_ARCH.values():
            if arch not in archs:
                archs.append(arch)
    for module_path, attr in _ARCH_ONLY_SIBLING_CONTRIBUTORS:
        module = importlib.import_module(module_path)
        for arch in getattr(module, attr):
            if arch not in archs:
                archs.append(arch)
    return tuple(archs)


# ---------------------------------------------------------------------------
# Sibling-config reconciliation
# ---------------------------------------------------------------------------
# The bespoke families read their geometry from a SIBLING config.json rather
# than from the GGUF metadata (see sibling_config_gguf_archs).  That config is
# routinely borrowed from a same-family reference repo, because community
# quantizations ship the .gguf alone.  A borrowed config is only correct as far
# as the two checkpoints share a geometry -- and depth-merged fine-tunes (e.g.
# DavidAU's Qwen3.6-40B, a 96-block re-stack of the 64-block Qwen3.6-27B) share
# everything EXCEPT the block count.  Nothing downstream re-reads the file, so
# the mismatch is silent: the loader builds a 64-layer model, maps 64 of the
# file's 96 blocks and drops the rest, and serves fluent nonsense.
#
# The GGUF file is authoritative about its own shape, so reconcile against it:
# override the depth (that is the one difference a depth-merge legitimately
# has) and hard-fail on every other geometry disagreement, which cannot be
# repaired by renumbering and means the sibling config belongs to a different
# model.

#: GGUF KV suffix (under ``<arch>.``) -> config attribute that must MATCH.
#: A disagreement here is not reconcilable: fail fast.
_GEOMETRY_CHECKS: Tuple[Tuple[str, str], ...] = (
    ("embedding_length", "hidden_size"),
    ("feed_forward_length", "intermediate_size"),
    ("attention.head_count", "num_attention_heads"),
    ("attention.head_count_kv", "num_key_value_heads"),
    ("attention.key_length", "head_dim"),
)


def _gguf_layer_index(tensor_name: str) -> Optional[int]:
    m = re.match(r"blk\.(\d+)\.", tensor_name)
    return int(m.group(1)) if m else None


def _rebuild_layer_types(
    old: List[str], n_layers: int, recurrent: Optional[set]
) -> Optional[List[str]]:
    """Extend a sibling config's ``layer_types`` list to ``n_layers`` entries.

    Preferred source is ``recurrent`` -- the set of block indices that carry
    recurrent (SSM/conv) tensors in the actual file, which is what the adapters
    classify layers from anyway.  Falls back to tiling the sibling pattern at
    its own period when the family has no recurrent layers (e.g. gemma4, whose
    layer_types encode a sliding/full attention cycle instead).
    """
    if not old:
        return None
    if recurrent:
        # Two labels: whatever the sibling used at a recurrent position and at
        # a non-recurrent one.  Reading them off the old list keeps the family's
        # own spelling instead of hard-coding "linear_attention".
        rec_label = next((old[i] for i in sorted(recurrent) if i < len(old)), None)
        non_label = next(
            (
                old[i]
                for i in range(len(old))
                if i not in recurrent and old[i] != rec_label
            ),
            None,
        )
        if rec_label is not None and non_label is not None:
            return [rec_label if i in recurrent else non_label for i in range(n_layers)]
    # Periodic tiling: smallest period that reproduces the sibling pattern.
    for period in range(1, len(old) + 1):
        if all(old[i] == old[i % period] for i in range(len(old))):
            return [old[i % period] for i in range(n_layers)]
    return None


def reconcile_sibling_config(config, gguf_file: str, arch: str) -> None:
    """Reconcile a sibling-config geometry against the GGUF file it describes.

    Mutates ``config``'s text config in place (depth only) and raises
    ``ValueError`` on a non-reconcilable disagreement.  Best-effort: if the
    GGUF metadata does not carry a field, that field is simply not checked.
    """
    import gguf  # lazy: keeps the registry import-light

    from sglang.srt.model_loader.gguf_shards import (
        gguf_metadata_path,
        iter_gguf_tensors,
        resolve_gguf_shard_paths,
    )

    # Split export: the KV block lives ONLY in the first part, the tensors live
    # ONLY in the later ones. Read each from where it actually is -- the vocab
    # cross-check below reads token_embd.weight, which for the DeepSeek V4 Flash
    # export is on part 2 while every KV field is on part 1.
    shard_paths = resolve_gguf_shard_paths(gguf_file)
    reader = gguf.GGUFReader(gguf_metadata_path(gguf_file), "r")

    def kv(suffix: str) -> Optional[int]:
        """The ``<arch>.<suffix>`` metadata value as an int, or None if it is
        absent or not a single number.

        Not every family stores these as scalars: gemma4 writes a PER-LAYER
        ``attention.head_count_kv`` list (the same list that makes
        transformers' GGUF reader unusable for gemma4 in the first place, see
        _peek_bespoke_gguf_arch). A uniform list still carries one value and is
        collapsed; a genuinely per-layer one has no scalar counterpart in the
        config, so it is skipped rather than guessed at.
        """
        field = reader.fields.get(f"{arch}.{suffix}")
        if field is None:
            return None
        try:
            value = field.contents()
        except Exception:
            return None
        if isinstance(value, (list, tuple)):
            if not value or len(set(value)) != 1:
                return None
            value = value[0]
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    text_config = (
        config.get_text_config() if hasattr(config, "get_text_config") else config
    )

    mismatches: List[str] = []
    for suffix, attr in _GEOMETRY_CHECKS:
        want = kv(suffix)
        have = getattr(text_config, attr, None)
        if want is None or not isinstance(have, int) or isinstance(have, bool):
            continue
        if have != want:
            mismatches.append(f"{attr}: config.json says {have}, GGUF says {want}")

    # One pass over the tensor directory: the embedding shape (for the vocab
    # check) and the recurrent-block set (for layer_types) both come from it.
    embd = None
    recurrent = set()
    for tensor in iter_gguf_tensors(shard_paths):
        if tensor.name == "token_embd.weight":
            embd = tensor
        elif "ssm" in tensor.name or "conv1d" in tensor.name:
            idx = _gguf_layer_index(tensor.name)
            if idx is not None:
                recurrent.add(idx)

    # Vocabulary is not in the GGUF KV block as a scalar; take it from the
    # embedding tensor, whose row count IS the vocabulary.
    vocab = getattr(text_config, "vocab_size", None)
    if vocab is not None and embd is not None:
        file_vocab = int(max(embd.shape[:2]))
        if int(vocab) != file_vocab:
            mismatches.append(
                f"vocab_size: config.json says {vocab}, GGUF token_embd has {file_vocab} rows"
            )

    # ``block_count`` counts the MTP/NEXTN draft block(s) as blocks: an
    # MTP-carrying Qwen3.6-27B GGUF reports 65 for a 64-layer model, with the
    # draft stored as blk.64.  num_hidden_layers counts only the backbone
    # (the draft lives in mtp_num_hidden_layers), so subtract the draft blocks
    # before comparing -- otherwise every MTP GGUF looks like a depth-merge
    # one layer deep and gets a bogus 65th layer.
    n_blocks = kv("block_count")
    if n_blocks is not None:
        n_blocks -= kv("nextn_predict_layers") or 0
    n_config = getattr(text_config, "num_hidden_layers", None)

    if mismatches:
        raise ValueError(
            f"GGUF/config.json geometry mismatch for {gguf_file}:\n  "
            + "\n  ".join(mismatches)
            + "\nThe sibling config.json describes a different model. Point "
            "--model-path at a GGUF whose config.json matches, or supply the "
            "checkpoint's own config.json next to the .gguf file."
        )

    if n_blocks is None or n_config is None or int(n_config) == n_blocks:
        return

    # Depth-only difference: the file wins.
    old_types = list(getattr(text_config, "layer_types", []) or [])
    new_types = _rebuild_layer_types(old_types, n_blocks, recurrent)
    if old_types and new_types is None:
        raise ValueError(
            f"GGUF {gguf_file} has {n_blocks} blocks but the sibling "
            f"config.json declares {n_config} layers, and its layer_types "
            "pattern could not be extended to the file's depth. Supply the "
            "checkpoint's own config.json next to the .gguf file."
        )

    logger.warning(
        "GGUF depth reconciliation: %s has %d blocks, sibling config.json "
        "declares num_hidden_layers=%d. Using the file's %d (the rest of the "
        "geometry matches, so this is a depth-merge of the config's model). "
        "layer_types rebuilt from the file's %d recurrent block(s).",
        gguf_file,
        n_blocks,
        n_config,
        n_blocks,
        len(recurrent),
    )
    text_config.num_hidden_layers = n_blocks
    if new_types is not None:
        text_config.layer_types = new_types
