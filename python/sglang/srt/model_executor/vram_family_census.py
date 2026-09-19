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
    if where == "after pools":
        # from here on the allocator peak is the RUNTIME transient (prefill
        # chunk, graphs, decode), not the load-time peak of the un-offloaded
        # weights -- [vram-peak] below reads it
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            pass
    return fam


# --- [vram-peak]: the transient the planner has to leave room for -----------
PEAK_EXTEND_MIN_TOKENS = 2048
PEAK_DECODE_AT = 64


def _peak_state(runner):
    st = getattr(runner, "_vram_peak_state", None)
    if st is None:
        st = {"extend": False, "decode": 0, "decode_done": False}
        try:
            runner._vram_peak_state = st
        except Exception:  # noqa: BLE001 -- slots classes
            pass
    return st


def maybe_log_vram_peak(runner, forward_batch, cuda=torch.cuda) -> Optional[str]:
    """Once after the first big extend (>= PEAK_EXTEND_MIN_TOKENS rows) and
    once at the PEAK_DECODE_AT-th decode forward: allocator peak since the
    pools, allocated, reserved and the card's real free bytes. That peak
    minus what was allocated before the forward is the transient the sizing
    must subtract instead of the hand-tuned air (19.09.). Returns the kind
    logged ('extend'/'decode') or None."""
    st = _peak_state(runner)
    mode = getattr(forward_batch, "forward_mode", None)
    kind = None
    if mode is not None and mode.is_extend() and not st["extend"]:
        ids = getattr(forward_batch, "input_ids", None)
        n = int(ids.shape[0]) if ids is not None else 0
        if n >= PEAK_EXTEND_MIN_TOKENS:
            st["extend"] = True
            kind = "extend"
    elif mode is not None and mode.is_decode() and not st["decode_done"]:
        st["decode"] += 1
        if st["decode"] >= PEAK_DECODE_AT:
            st["decode_done"] = True
            kind = "decode"
    if kind is None:
        return None
    try:
        peak = cuda.max_memory_allocated() / 2**30
        alloc = cuda.memory_allocated() / 2**30
        reserved = cuda.memory_reserved() / 2**30
        free, total = cuda.mem_get_info()
        n = int(getattr(forward_batch, "input_ids").shape[0]) if getattr(forward_batch, "input_ids", None) is not None else -1
        logger.info(
            "[vram-peak] %s (%s rows): allocator peak since pools %.2f GiB, allocated now %.2f, "
            "reserved %.2f, card free %.2f of %.2f GiB -> transient headroom used = peak - allocated %.2f GiB",
            kind, n, peak, alloc, reserved, free / 2**30, total / 2**30, peak - alloc,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[vram-peak] skipped: %s", exc)
    return kind
