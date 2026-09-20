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


def census(
    named: Iterable[Tuple[str, torch.Tensor]], cuda_only: bool = True
) -> Dict[str, int]:
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
        out[family_of(name)] = (
            out.get(family_of(name), 0) + t.numel() * t.element_size()
        )
    return out


def log_vram_family_census(
    model: torch.nn.Module, tag: str, where: str
) -> Dict[str, int]:
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
        tag,
        where,
        total / 2**30,
        parts,
        alloc,
        reserved,
    )
    if where == "after pools":
        # from here on the allocator peak is the RUNTIME transient (prefill
        # chunk, graphs, decode), not the load-time peak of the un-offloaded
        # weights -- [vram-peak] below reads it
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            pass
        # #58: and THIS is the moment the idle reading is honest -- pools
        # built, nothing forwarding yet. The vision stage is placed against
        # this number, not against [vram-peak]'s.
        log_vram_idle(runner, "after pools")
    return fam


# --- [vram-idle]: the free air with NO forward in flight ---------------------
#
# Task #58. `[vram-peak]` answers "how much did this rank draw at its worst",
# which is the right instrument for sizing a POOL that coexists with the
# prefill. It is the WRONG instrument for the transient vision stage, which
# runs before the P prefill really starts and is gone before it: sizing that
# stage against a number with the prefill transient already subtracted
# understates the card by the whole transient (1.69-2.47 GiB per rank on
# fn8aj) and refuses placements that would have fit.
#
# `weg2/corridor_budget.py:143` already distinguishes `free_idle_mib` from
# `free_load_mib` in its data model -- what was missing was an emitter that
# prints the idle half. Until this line exists in a boot log, the vision
# stage planner is fed fn8aj's quietest `[vram-peak] decode` sample as a
# STAND-IN, and that stand-in is named as one in
# `tests/moe_offload/test_vision_stage_planner_0920.py`.


def log_vram_idle(runner, where: str, cuda=torch.cuda) -> Optional[float]:
    """Print this rank's card free/total with no forward in flight.

    Returns the free GiB, or ``None`` when the reading could not be taken --
    and ``None`` is NOT logged as zero, because a card that could not be read
    is not a full card.

    ``where`` names the moment ("after pools", "idle", "before vision stage"),
    so two lines from one boot are never confused for one instrument sampled
    twice.
    """
    try:
        free, total = cuda.mem_get_info()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[vram-idle] skipped: %s", exc)
        return None
    try:
        alloc = cuda.memory_allocated() / 2**30
        reserved = cuda.memory_reserved() / 2**30
    except Exception:  # noqa: BLE001
        alloc = reserved = float("nan")
    free_gib = free / 2**30
    logger.info(
        "[vram-idle] %s: card free %.3f of %.3f GiB, allocated %.2f, reserved %.2f "
        "-- NO forward in flight; this is free_idle, the input the transient "
        "vision stage is placed against (#58)",
        where,
        free_gib,
        total / 2**30,
        alloc,
        reserved,
    )
    return free_gib


# --- [vram-peak]: the transient the planner has to leave room for -----------
PEAK_EXTEND_MIN_TOKENS = 2048
PEAK_DECODE_AT = 64


#: 20.09. (fn8ak3): a new high-water has to beat the last LOGGED one by this
#: much before it is re-emitted, so a long prefill does not print a line per
#: chunk.  Small enough that the number the planner reads is the real maximum
#: to within one step of this size.
PEAK_HIGHWATER_STEP_GIB = 0.25


def _peak_state(runner):
    st = getattr(runner, "_vram_peak_state", None)
    if st is None:
        st = {"extend": False, "decode": 0, "decode_done": False, "logged": 0.0}
        try:
            runner._vram_peak_state = st
        except Exception:  # noqa: BLE001 -- slots classes
            pass
    return st


def maybe_log_vram_peak(runner, forward_batch, cuda=torch.cuda) -> Optional[str]:
    """Once after the first big extend (>= PEAK_EXTEND_MIN_TOKENS rows), once
    at the PEAK_DECODE_AT-th decode forward, and AGAIN on every new allocator
    high-water: allocator peak since the pools, allocated, reserved and the
    card's real free bytes. That peak minus what was allocated before the
    forward is the transient the sizing must subtract instead of the hand-tuned
    air (19.09.). Returns the kind logged ('extend'/'decode'/'high-water') or
    None.

    THE HIGH-WATER KIND EXISTS BECAUSE THE LATCH LIED (20.09., fn8ak3).
    ``st["extend"]`` latched on the FIRST extend of >= 2048 rows, which under
    chunked prefill is the first chunk of the first request.  The transient a
    deep prefill draws is not constant across chunks: the attention workspace
    and the partials scale with the CACHED context the chunk attends over, so
    the worst chunk of a 259k needle is the last one, minutes after the latch.
    Measured consequence: fn8aj printed ``allocated now 13.29`` for rank 2 and
    a planner reading it as the peak sized 18 extra pool rows onto that card;
    the real prefill peak was ``18.60 GiB allocated by PyTorch`` and the boot
    died with 73.5 MiB free.  A once-per-kind instrument cannot report a
    maximum -- it reports the first sample and calls it one.
    """
    st = _peak_state(runner)
    mode = getattr(forward_batch, "forward_mode", None)
    kind = None
    if mode is not None and mode.is_extend() and not st["extend"]:
        ids = getattr(forward_batch, "input_ids", None)
        n = int(ids.shape[0]) if ids is not None else 0
        if n >= PEAK_EXTEND_MIN_TOKENS:
            st["extend"] = True
            kind = "extend"
    elif (
        mode is not None
        and not st["decode_done"]
        and (
            mode.is_decode()
            # fn7v (19.09.): under spec decoding the target runs verify
            # forwards, not decode ones -- count them as the decode kind
            or bool(getattr(mode, "is_target_verify", lambda: False)())
        )
    ):
        st["decode"] += 1
        if st["decode"] >= PEAK_DECODE_AT:
            st["decode_done"] = True
            kind = "decode"
    try:
        peak = cuda.max_memory_allocated() / 2**30
    except Exception as exc:  # noqa: BLE001
        logger.debug("[vram-peak] skipped: %s", exc)
        return None
    if kind is None and peak >= st["logged"] + PEAK_HIGHWATER_STEP_GIB:
        kind = "high-water"
    if kind is None:
        return None
    st["logged"] = max(st["logged"], peak)
    try:
        alloc = cuda.memory_allocated() / 2**30
        reserved = cuda.memory_reserved() / 2**30
        free, total = cuda.mem_get_info()
        n = (
            int(getattr(forward_batch, "input_ids").shape[0])
            if getattr(forward_batch, "input_ids", None) is not None
            else -1
        )
        logger.info(
            "[vram-peak] %s (%s rows): allocator peak since pools %.2f GiB, allocated now %.2f, "
            "reserved %.2f, card free %.2f of %.2f GiB -> transient headroom used = peak - allocated %.2f GiB",
            kind,
            n,
            peak,
            alloc,
            reserved,
            free / 2**30,
            total / 2**30,
            peak - alloc,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[vram-peak] skipped: %s", exc)
    return kind
