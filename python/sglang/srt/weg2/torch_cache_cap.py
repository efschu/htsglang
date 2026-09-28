"""VRAM loop P0 (28.09.): bound the torch caching allocator per rank process.

rc12z26 (NF, WEG2-VRAM-PEAK): the near-OOM moments were not torch DEMAND but
the caching allocator's swing -- 0.7-3.5 GiB reserved-but-unused per rank; at
card_free 76 MiB D's own cache held 1.8 GiB. That hurts only allocations
OUTSIDE the torch pool (cuMem/VMM: TMS, KV stages, expert banks; NCCL; the
other process on the card) -- the fn8i / cu_mem_create OOM classes. It is also
why any VRAM handed to the experts would be taken back at the next peak.

The cap is a BOOKED size, not a reserve: the launcher writes, per rank, the
D-RANK VRAM verdict's ``verfuegbar`` (card - foreign - non-torch - reserve)
minus the corridor floor into ``SGLANG_WEG2_TORCH_CACHE_CAP_MIB``. torch then
empties its own cache before it would cross that line (and raises its own OOM
if live tensors truly need more -- the same bytes that today die outside
torch). Off unless ``SGLANG_WEG2_TORCH_CACHE_CAP=1``. The named empty_cache at
a phase boundary is the sleep's (weg2 sleep_staging); nothing here runs in the
decode loop.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_TORCH_CACHE_CAP"
MIB_ENV = "SGLANG_WEG2_TORCH_CACHE_CAP_MIB"
MARKER = "WEG2-TORCH-CACHE-CAP"
_MIB = float(1 << 20)


def enabled() -> bool:
    return str(os.environ.get(ENV, "0")).strip() == "1"


def cap_mib_for(index: int, raw: Optional[str] = None) -> Optional[float]:
    """This rank's booked cap in MiB, or None (unset / unreadable / no entry)."""
    raw = os.environ.get(MIB_ENV, "") if raw is None else raw
    try:
        vals = [float(x) for x in str(raw).split(",") if str(x).strip()]
    except ValueError:
        return None
    if index < 0 or index >= len(vals) or vals[index] <= 0:
        return None
    return vals[index]


def fraction_for(cap_mib: float, total_bytes: int) -> Optional[float]:
    if total_bytes <= 0:
        return None
    return max(0.0, min(1.0, cap_mib * _MIB / float(total_bytes)))


def rank_index(tp_rank: int, pp_rank: int) -> int:
    """P is a pipeline (PP ordinals), D a tensor group (TP ordinals)."""
    group = str(os.environ.get("SGLANG_WEG2_GROUP", "")).strip().upper()
    return int(pp_rank) if group == "P" else int(tp_rank)


def arm(tp_rank: int, pp_rank: int, gpu_id: int, torch_mod=None) -> Optional[float]:
    """Set the per-process fraction once, after the device is selected. Never
    raises; one named line either way when the switch is on."""
    if not enabled():
        return None
    try:
        torch = torch_mod
        if torch is None:
            import torch  # noqa: F811
        idx = rank_index(tp_rank, pp_rank)
        cap = cap_mib_for(idx)
        if cap is None:
            logger.warning("%s rank=%d: %s=1 but no entry in %s=%r -- NOT armed (uncapped)",
                           MARKER, idx, ENV, MIB_ENV, os.environ.get(MIB_ENV, ""))
            return None
        _free, total = torch.cuda.mem_get_info(gpu_id)
        frac = fraction_for(cap, int(total))
        if frac is None:
            return None
        torch.cuda.set_per_process_memory_fraction(frac, gpu_id)
        logger.info("%s rank=%d gpu=%d: cap %.0f MiB of %.0f MiB -> fraction %.4f (booked: verdict "
                    "verfuegbar minus corridor floor; the allocator empties its own cache before "
                    "crossing it, so cuMem/NCCL keep their room)", MARKER, idx, gpu_id, cap,
                    total / _MIB, frac)
        return frac
    except Exception as exc:  # noqa: BLE001 -- a cap never kills a rank
        logger.warning("%s arm skipped (%s: %s)", MARKER, type(exc).__name__, str(exc)[:160])
        return None


def launcher_caps(verdicts: Sequence[object], corridor_floor_mib: float) -> str:
    """The launcher's per-rank cap vector: verdict ``available_mib`` minus the
    corridor floor, integer MiB, comma-joined in rank order."""
    return ",".join(str(int(max(0.0, float(v.available_mib) - float(corridor_floor_mib))))
                    for v in verdicts)
