"""VRAM bytes by tensor family, per rank -- one log line, measured not derived.

19.09. (fn7g): the non-expert VRAM per rank (3.9-5.8 GB after the expert
offload) was explained three different ways from load/offload deltas, all
wrong. This sums the CUDA-resident parameters and buffers of the model by
family name so the next balance sheet comes from the tensors themselves.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, Tuple

import torch

logger = logging.getLogger(__name__)

# order matters: first match wins
_FAMILIES = [
    ("mtp", re.compile(r"(^|\.)mtp\.")),
    ("visual", re.compile(r"(^|\.)visual\.")),
    ("ple", re.compile(r"\.ple\.")),
    ("experts", re.compile(r"\.experts(\.|$)")),
    ("shared_expert", re.compile(r"\.shared_expert")),
    ("moe_gate", re.compile(r"\.mlp\.gate(\.|$)")),
    ("linear_attn", re.compile(r"\.linear_attn\.")),
    ("self_attn", re.compile(r"\.self_attn\.")),
    ("hyper_connection", re.compile(r"hyper_connection")),
    ("embed_tokens", re.compile(r"embed_tokens")),
    ("lm_head", re.compile(r"(^|\.)lm_head")),
    ("norm", re.compile(r"norm")),
]


def family_of(name: str) -> str:
    for fam, rx in _FAMILIES:
        if rx.search(name):
            return fam
    return "other"


def census(named: Iterable[Tuple[str, torch.Tensor]], cuda_only: bool = True) -> Dict[str, int]:
    """Bytes per family over ``named`` (name, tensor) pairs; dedups storages
    so a tensor registered twice (tied weights, views) is counted once."""
    seen = set()
    out: Dict[str, int] = {}
    for name, t in named:
        if t is None or (cuda_only and not t.is_cuda):
            continue
        try:
            key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        except Exception:  # noqa: BLE001 -- meta/fake tensors
            key = (id(t), 0)
        if key in seen:
            continue
        seen.add(key)
        out[family_of(name)] = out.get(family_of(name), 0) + t.numel() * t.element_size()
    return out


def log_vram_family_census(model: torch.nn.Module, tag: str, where: str) -> Dict[str, int]:
    named = list(model.named_parameters()) + list(model.named_buffers())
    fam = census(named)
    total = sum(fam.values())
    parts = ", ".join(
        f"{k} {v / 2**30:.2f}" for k, v in sorted(fam.items(), key=lambda kv: -kv[1])
    )
    try:
        alloc = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
    except Exception:  # noqa: BLE001
        alloc = reserved = float("nan")
    logger.info(
        "[vram-census] %s %s: model tensors on device %.2f GiB = {%s}; "
        "torch allocated %.2f GiB, reserved %.2f GiB (the gap to allocated is "
        "non-model: workspaces, pool tables, KV, activations)",
        tag, where, total / 2**30, parts, alloc, reserved,
    )
    return fam
