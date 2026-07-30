# SPDX-License-Identifier: Apache-2.0
"""GGUF name map for the DFLASH draft checkpoint (#274 round 7c).

A quantised DFLASH drafter is the cheapest way to put a second draft head on a
card that is not the one the serving group's biggest shard lives on: the BF16
checkpoint is 3300 MiB, the Q8_0 GGUF of the very same 58 tensors is 1753.

Why this file exists rather than a generic path or a family adapter:

* The GENERIC path (``GGUFModelLoader._get_gguf_weights_map``) derives its HF
  names from ``AutoModelForCausalLM.from_config(config)``.  A DFLASH draft
  config declares ``architectures: ["DFlashDraftModel"]`` and maps
  ``AutoModel`` -- not ``AutoModelForCausalLM`` -- at its remote code, so that
  instantiation cannot succeed.  The names are fully determined by two config
  fields anyway, so they are generated here instead of discovered.
* A REGISTRY family adapter (``gguf_registry``) dispatches on ``model_type``,
  and a DFLASH draft config carries ``model_type: "qwen3"``.  Registering that
  would capture every plain Qwen3 GGUF as well.  Dispatch here is on the
  ARCHITECTURE, which is what actually distinguishes the drafter.

Measured against ``Qwen3.6-27B-DFlash-Q8_0.gguf``: 56 of the 58 tensors already
match llama.cpp's stock qwen3 naming; the delta is seven names in three classes
(``dflash_fc``, ``dflash_hidden_norm``, and ``blk.N.post_attention_norm`` where
the stock map emits ``ffn_norm``).  Rather than patch a stock map with three
exceptions, the whole map is written out -- it is 14 lines of table, it is
exact, and it can be tested against the file without a GPU.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Architectures whose checkpoint is a DFLASH draft (no embedding, no lm_head:
# the drafter borrows both from its target).  These are HF ``architectures``
# entries, i.e. they are only visible once the config has been READ.
DFLASH_ARCHITECTURES = ("DFlashDraftModel", "DFlashLagunaForCausalLM")

# GGUF ``general.architecture`` strings of the same checkpoints -- the name in
# the FILE HEADER, which is what is available BEFORE any config is read.
#
# Everything above dispatches on the HF architecture, and can only do so
# because the sibling config.json was read in the first place. That read is not
# automatic: transformers' GGUF reader does not know "dflash-draft" and raises
# "GGUF model with architecture dflash-draft is not supported yet" when handed
# the file, exactly as it does for qwen35/gemma4. The config peek
# (utils/hf_transformers/config.py: _peek_bespoke_gguf_arch) routes those
# around it by reading the sibling config.json instead, and it decides from
# this header string. Exported here so gguf_registry can add it to
# sibling_config_gguf_archs() without importing this module eagerly.
DFLASH_GGUF_ARCHS = ("dflash-draft",)

# Per-layer tensors: GGUF local name -> HF local name, both under their own
# layer prefix.  Every entry here was confirmed present in the released Q8_0
# export; nothing is speculative.
_LAYER_ENTRIES = (
    ("attn_norm", "input_layernorm"),
    # llama.cpp's stock qwen3 map calls this ``ffn_norm``; the DFLASH exports
    # write ``post_attention_norm``.  This one line is why the stock map misses
    # five tensors.
    ("post_attention_norm", "post_attention_layernorm"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("attn_q_norm", "self_attn.q_norm"),
    ("attn_k_norm", "self_attn.k_norm"),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("ffn_down", "mlp.down_proj"),
)

# Model-level tensors.  ``dflash_fc`` is the projection from the concatenated
# target-layer features into the draft hidden size -- the largest single tensor
# in the checkpoint and the one that forced ``fc`` off ``nn.Linear``.
_GLOBAL_ENTRIES = (
    ("dflash_fc", "fc"),
    ("dflash_hidden_norm", "hidden_norm"),
    ("output_norm", "norm"),
)


def is_dflash_gguf_config(hf_config) -> bool:
    """Does this config describe a DFLASH draft checkpoint?"""
    archs = getattr(hf_config, "architectures", None) or ()
    return any(a in DFLASH_ARCHITECTURES for a in archs)


def build_dflash_name_map(hf_config) -> Dict[str, str]:
    """GGUF tensor name -> HF parameter name, for a DFLASH draft checkpoint.

    Generated from ``num_hidden_layers`` alone; no model is instantiated and no
    file is opened, so this is usable from a config peek.
    """
    num_layers = int(getattr(hf_config, "num_hidden_layers", 0) or 0)
    if num_layers <= 0:
        raise ValueError(
            "DFLASH GGUF name map needs a positive num_hidden_layers; got "
            f"{num_layers!r}"
        )
    name_map: Dict[str, str] = {}
    for gguf_local, hf_local in _GLOBAL_ENTRIES:
        name_map[f"{gguf_local}.weight"] = f"{hf_local}.weight"
    for i in range(num_layers):
        for gguf_local, hf_local in _LAYER_ENTRIES:
            name_map[f"blk.{i}.{gguf_local}.weight"] = f"layers.{i}.{hf_local}.weight"
    return name_map


def dflash_unquantized_module_prefixes(hf_config) -> List[str]:
    """Modules that must be built DENSE because the export keeps them F32.

    Norms are F32 in every DFLASH export (all 22 of them in the Q8_0 build), so
    they must not get a packed parameter -- the loader would then look for a
    ``qweight`` the file does not contain.  They are RMSNorm modules rather than
    Linear ones and so carry no quant method anyway; the list is returned for
    the loader's ``modules_to_not_convert``, where naming them is free and
    guards against a future export quantising one.
    """
    num_layers = int(getattr(hf_config, "num_hidden_layers", 0) or 0)
    prefixes = ["hidden_norm", "norm"]
    for i in range(num_layers):
        prefixes.append(f"layers.{i}.input_layernorm")
        prefixes.append(f"layers.{i}.post_attention_layernorm")
        prefixes.append(f"layers.{i}.self_attn.q_norm")
        prefixes.append(f"layers.{i}.self_attn.k_norm")
    return prefixes


def audit_dflash_name_map(name_map: Dict[str, str], file_tensor_names) -> Optional[str]:
    """Compare a built map against the tensors a GGUF file actually holds.

    Returns None when they agree exactly, otherwise a human-readable diff.  The
    round-7c lesson: this comparison found the ``ffn_norm`` /
    ``post_attention_norm`` split in one run, where reading the two naming
    conventions side by side had not.
    """
    have = set(file_tensor_names)
    want = set(name_map)
    missing = sorted(want - have)
    extra = sorted(have - want)
    if not missing and not extra:
        return None
    parts = []
    if missing:
        parts.append(f"expected but absent from the file ({len(missing)}): {missing}")
    if extra:
        parts.append(f"in the file but unclaimed ({len(extra)}): {extra}")
    return "; ".join(parts)
