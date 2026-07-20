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

Import discipline: this module stays import-light (importlib only).  The adapter
modules — which pull in torch/gguf — are imported LAZILY inside the functions so
importing the registry (e.g. from the early config-peek path) is cheap and
cannot create an import cycle.
"""

from __future__ import annotations

import importlib
from typing import List, Optional, Tuple, Type

# (family_name, module_path, adapter_class_name).  Order matters only for the
# arch-tuple ordering exposed by sibling_config_gguf_archs().
_FAMILIES: Tuple[Tuple[str, str, str], ...] = (
    ("qwen35", "sglang.srt.model_loader.gguf_qwen35", "Qwen35GGUFAdapter"),
    ("gemma4", "sglang.srt.model_loader.gguf_gemma4", "Gemma4GGUFAdapter"),
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
    of the GGUF metadata — i.e. the union of every bespoke family's GGUF archs.
    Used by the config/tokenizer peek in ``utils/hf_transformers``.
    """
    archs: List[str] = []
    for _family_name, cls in _iter_adapter_classes():
        for arch in cls.MODEL_TYPE_TO_ARCH.values():
            if arch not in archs:
                archs.append(arch)
    return tuple(archs)
