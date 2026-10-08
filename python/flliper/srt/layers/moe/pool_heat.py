"""#276 heat record of the device-planned expert pool (D decode).

``FLLIPER_DEBUG_MOE_HEAT=<dir>`` (default unset = off): every pool layer keeps
ONE device histogram ``[E + 2]`` int64 -- per LOCAL expert id how many routed
(token, k) lanes of a captured decode step chose it, then the lanes routed to
an id this rank does not own (``-1`` / outside ``[0, E)``), then the number of
captured steps. The count is one ``index_add_`` inside the captured step
(``MoEExpertOffloadCache.prepare_pool``), on the device, with no host read; it
is written out only at D's phase boundary -- the sleep, before any tag is
paused (``flush``) -- as one JSON record per rank, and zeroed at the wake
(``reset``) because the tables live under a paused tag.

Off, nothing exists: no tensor is allocated, ``prepare_pool`` adds no op, the
captured graph is the one before #276.

The record is the input of the planner's hot-set stage (#276: the resident
list per phase and rank chosen by measured heat, second stage after the
ownership solve, written into the expert map). It carries the layer's global
id window (``_layer_expert_window``: ``lo`` and ``pad``) so the planner maps
local ids to global ones with the one conversion the house has. What it does
NOT see: eager forwards (extend / prefill, ``run_waves``), the speculative
prefetch, and rows of a captured bucket wider than the batch that the top-k
does not mask (NF captures every seat count 1..--d-bs, so only a partially
filled verify bucket can add such lanes).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MARKER = "MOE-HEAT (#276)"
RECORD_KIND = "moe_heat"
RECORD_VERSION = 1


def heat_dir() -> Optional[str]:
    """The record directory, or ``None`` when the instrument is off."""
    try:
        from flliper.srt.environ import envs

        value = envs.FLLIPER_DEBUG_MOE_HEAT.get()
    except Exception:  # noqa: BLE001 - an instrument never kills a boot
        return None
    value = (value or "").strip()
    return value or None


def allocate(device, num_experts: int, width: int):
    """``(heat, ones)`` for one pool layer, or ``(None, None)`` when off.

    ``heat`` = ``[E + 2]`` int64: lanes per local id, lanes not owned here,
    captured steps. ``ones`` = ``[width]`` int64, the addend of the step's
    ``index_add_`` (allocated once, so the captured step allocates nothing
    for it)."""
    if heat_dir() is None:
        return None, None
    import torch

    E = int(num_experts)
    heat = torch.zeros(E + 2, dtype=torch.int64, device=device)
    ones = torch.ones(max(1, int(width)), dtype=torch.int64, device=device)
    return heat, ones


def count(heat, ones, flat, num_experts: int) -> None:
    """Add one step's routed ids (``flat``, one per (token, k) lane) to
    ``heat``. Device ops only, capture-safe: lanes outside ``[0, E)`` go to
    slot ``E``, slot ``E + 1`` counts the step."""
    import torch

    E = int(num_experts)
    n = int(flat.numel())
    if n > ones.numel():
        # a step wider than the plan width cannot be captured anyway; count
        # what the addend covers rather than allocate inside the step
        flat = flat[: ones.numel()]
        n = int(flat.numel())
    idx = torch.where((flat >= 0) & (flat < E), flat, torch.full_like(flat, E)).long()
    heat.index_add_(0, idx, ones[:n])
    heat[E + 1 :].add_(1)


def _caches(models) -> List[Any]:
    out = []
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            cache = getattr(module, "_expert_offload", None)
            if cache is None or getattr(cache, "_pool_heat", None) is None:
                continue
            out.append(cache)
    return out


def layer_record(cache, counts: List[int]) -> Dict[str, Any]:
    """One layer of the record from its histogram read back as ``counts``."""
    from flliper.srt.layers.moe.expert_offload import _layer_expert_window

    layer = getattr(cache, "layer", None)
    E = int(cache.num_local_experts)
    window = _layer_expert_window(layer) if layer is not None else None
    lo, pad = (window if window is not None else (None, False))
    planner = getattr(cache, "planner", None)
    resident_ids = getattr(planner, "resident_ids", None) if planner is not None else None
    R = int(getattr(cache, "resident_count", 0) or 0)
    resident = sorted(int(e) for e in resident_ids) if resident_ids is not None else list(range(R))
    return {
        "layer_id": getattr(layer, "layer_id", None),
        "num_local_experts": E,
        "global_lo": lo,
        "pad": bool(pad),
        "resident_local": resident,
        "counts": [int(c) for c in counts[:E]],
        "not_local": int(counts[E]),
        "steps": int(counts[E + 1]),
    }


def record(layers: List[Dict[str, Any]], *, rank: int, group: str, reason: str,
           phase_index: Optional[int] = None) -> Dict[str, Any]:
    return {
        "kind": RECORD_KIND,
        "version": RECORD_VERSION,
        "group": group,
        "rank": int(rank),
        "reason": reason,
        "phase_index": phase_index,
        "time_unix": time.time(),
        "scope": "captured decode pool steps (prepare_pool wave 1); eager extend/prefill "
                 "and the speculative prefetch are not counted",
        "layers": layers,
    }


def _write(directory: str, rec: Dict[str, Any]) -> str:
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(rec["time_unix"]))
    name = "moe_heat_%s_tp%d_%s_p%s.json" % (
        rec["group"], rec["rank"], stamp,
        "-" if rec["phase_index"] is None else int(rec["phase_index"]))
    path = os.path.join(directory, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, path)
    return path


def flush(models, *, rank: int, group: str, reason: str,
          phase_index: Optional[int] = None) -> Optional[str]:
    """At D's phase end (before any pause): read every layer's histogram (ONE
    host read of the stacked counters), write the record, zero the counters.
    Returns the record path; ``None`` when off, when no pool layer counted,
    or on any error (logged, never raised)."""
    try:
        directory = heat_dir()
        if directory is None:
            return None
        caches = _caches(models)
        if not caches:
            return None
        import torch

        stacked = torch.cat([c._pool_heat for c in caches]).tolist()
        layers, at = [], 0
        for c in caches:
            n = int(c._pool_heat.numel())
            layers.append(layer_record(c, stacked[at : at + n]))
            at += n
            c._pool_heat.zero_()
        steps = max((l["steps"] for l in layers), default=0)
        if steps <= 0:
            return None
        rec = record(layers, rank=rank, group=group, reason=reason, phase_index=phase_index)
        path = _write(directory, rec)
        lanes = sum(sum(l["counts"]) for l in layers)
        ids = sum(sum(1 for v in l["counts"] if v) for l in layers)
        logger.info(
            "%s wrote %s layers=%d steps=%d lanes=%d not_local=%d ids_hit=%d reason=%s",
            MARKER, path, len(layers), steps, lanes,
            sum(l["not_local"] for l in layers), ids, reason,
        )
        return path
    except Exception as exc:  # noqa: BLE001 - an instrument never kills a sleep
        logger.warning("%s flush failed: %s: %s", MARKER, type(exc).__name__, exc)
        return None


def reset(models) -> int:
    """At the wake, after the rearm: zero every layer's histogram (the tables
    live under a paused tag; what the pages hold after the resume is not a
    count). Returns the layers zeroed; never raises."""
    try:
        caches = _caches(models)
        for c in caches:
            c._pool_heat.zero_()
        return len(caches)
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s reset failed: %s: %s", MARKER, type(exc).__name__, exc)
        return 0
