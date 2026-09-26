"""H95 demand probe of the device-planned expert pool.

``SGLANG_DEBUG_MOE_POOL_DEMAND=N`` (default 0 = off): every N decode graph
replays each rank logs ONE line ``MOE-POOL-DEMAND (H95)`` with, per MoE layer
since the previous line, the MAXIMUM number of distinct non-resident expert
ids one pool step routed (``D_nr``: LRU hits + misses, everything the rows
above the fixed residents had to hold in that step) against ``C`` = LRU +
staging rows, and how many steps exceeded ``C`` (with overflow waves those
are the steps a second wave actually served; without, they are the steps the
Task #40 bound exists to exclude).

``take_report`` in the pool only reports SUMS (misses per forward), which
cannot say whether a bs-n step ever needs more than ``C`` rows -- the whole
question of the pool bound. Reading the counters is one host rendezvous per
report (all layers stacked), taken BEFORE a replay, so the probe never sits
inside a captured forward. Never raises.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MARKER = "MOE-POOL-DEMAND (H95)"
_STATE = {"every": None, "n": 0, "caches": None}


def _every() -> int:
    if _STATE["every"] is None:
        try:
            from sglang.srt.environ import envs

            _STATE["every"] = max(0, int(envs.SGLANG_DEBUG_MOE_POOL_DEMAND.get() or 0))
        except Exception:  # noqa: BLE001 - an instrument never kills a replay
            _STATE["every"] = 0
    return int(_STATE["every"])


def _caches(model) -> List[Tuple[object, object]]:
    if _STATE["caches"] is None:
        found = []
        for module in model.modules():
            cache = getattr(module, "_expert_offload", None)
            if cache is None or getattr(cache, "_pool_tables", None) is None:
                continue
            if getattr(cache._pool_tables, "demand", None) is None:
                continue
            found.append((getattr(getattr(cache, "layer", None), "layer_id", "?"), cache))
        if not found:
            return []  # tables not built yet (first replays): look again later
        _STATE["caches"] = found
    return _STATE["caches"]


def demand_line(
    rows: Sequence[Tuple[object, int, int, int, int, Optional[int]]], *, replays: int
) -> str:
    """The report line. ``rows`` = per layer ``(layer_id, max D_nr, steps over
    C, steps, C, waves or None)``."""
    live = [r for r in rows if r[3] > 0]
    if not live:
        return "%s replays=%d layers=%d steps=0 (no captured pool step ran)" % (
            MARKER, int(replays), len(rows))
    worst = max(live, key=lambda r: (r[1] / max(1, r[4]), r[1]))
    maxes = sorted(r[1] for r in live)
    over = sum(r[2] for r in live)
    steps = sum(r[3] for r in live)
    top = sorted(live, key=lambda r: (r[1] / max(1, r[4]), r[1]), reverse=True)[:6]
    waves = sorted({r[5] for r in live if r[5] is not None})
    return (
        "%s replays=%d layers=%d max_nonres_per_step=%d at layer %s (C=%d, %.2f of C) "
        "median_layer_max=%d over_C_steps=%d of %d layer-steps worst=[%s] waves=%s"
        % (
            MARKER, int(replays), len(live), worst[1], worst[0], worst[4],
            worst[1] / max(1, worst[4]), maxes[len(maxes) // 2], over, steps,
            ", ".join("L%s:%d/%d" % (r[0], r[1], r[4]) for r in top),
            waves if waves else "-",
        )
    )


def maybe_report(model) -> None:
    """Count one replay; every N-th, read and reset every layer's counters and
    log the line. Called from the decode graph runner before a replay."""
    try:
        every = _every()
        if every <= 0:
            return
        _STATE["n"] += 1
        if _STATE["n"] % every:
            return
        caches = _caches(model)
        if not caches:
            return
        import torch

        from sglang.srt.layers.moe.expert_pool_device import pool_row_capacity

        stacked = torch.stack([c._pool_tables.demand for _l, c in caches]).tolist()
        for _l, c in caches:
            c._pool_tables.demand.zero_()
        rows = []
        for (lid, c), (mx, over, steps) in zip(caches, stacked):
            seen = getattr(c, "_pool_waves_seen", None) or {}
            rows.append((lid, int(mx), int(over), int(steps),
                         pool_row_capacity(c._pool_tables),
                         max(seen.values()) if seen else None))
        logger.info("%s", demand_line(rows, replays=every))
    except Exception as exc:  # noqa: BLE001 - an instrument never kills a replay
        logger.warning("%s failed: %s: %s", MARKER, type(exc).__name__, exc)
