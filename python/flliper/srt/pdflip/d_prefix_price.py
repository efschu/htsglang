"""D-PREFIX-PRICE 1024 (#215): the lifetime price credits the device rows the load-back MAPS, not allocates.

SCHEDULER-SOLL v2 R7 (a request under X that D does not take is a defect, never a flip reason) and R11 (one verdict per
group): D's lifetime gate (``PrefillAdder.add_one_req``) must charge what the admission really allocates.

THE GAP (27B int4ar ``_1024_003732``, D TP0/TP1 01:02:45, rid pdflip-18-188, N=100380):
  ``D-ADMIT-REFUSED gate=lifetime price=104477 budget=64355 extend=100380 prefix=0 uncached=3676``
  ``#823 HEAD-VOTE ANCHOR matched=78567 (device 0 + host 78567) anchor_depth=96704``
The price charged ``extend = N - len(prefix_indices) = 100380``. The match had cut the device part to 0: the mamba
validator ends ``prefix_indices`` at the deepest node WITH a recurrent state (``last_device_node``), and the 18137-row
system prompt every agent request shares is a split node without one (``branch_point_never_anchored``). The host walk
counts only ``host_value`` (78567). The prompt's rows were on the device all the time -- locked by the three other
running decodes -- and the load-back MAPS them: ``UnifiedRadixCache.init_load_back._collect_new_prefix_indices`` takes
every node value from ``best_match_node`` up to ``req.last_node``, device-resident ones included, and ``load_back``
allocates only for the evicted (host) part. The request later ran with ``cached_device=18137 cached_l2=78567``. So the
price was 18137 too high: 86340 instead of 104477 -- after the displacement of 18-189 the budget was 100799 and the head
would have been admitted (181 refusals in 18.7 s otherwise).

THE CREDIT (``locked_device_gap``). Of the rows on the path ``best_match_node -> req.last_node`` (exclusive), only the
device-resident rows of nodes ANOTHER holder keeps FULL-locked (``lock_ref >= 1``):
  * a locked node is not evictable, so the lifetime budget (available + deliverable evictable) does not count it, and
    nothing inside ``add_one_req`` can unlock it before the load-back's own ``inc_lock_ref(best_match_node)`` and the
    admission's ``inc_lock_ref(req.last_node)`` take it over -- releases happen in result processing, never between
    the gate and the lock. The credit is secured until admission; it is never an overbooking.
  * an UNLOCKED device node (``lock_ref == 0``) is NOT credited: it sits in the budget's evictable count, and the
    load-back locks it out of there -- charging it is exact (upstream vLLM: ``get_num_blocks_to_allocate`` counts a
    computed block with ``ref_cnt == 0`` as one to allocate, ``single_type_kv_cache_manager.py``).
  * a node not on the device (value None) is the host hit: the load-back allocates it, it stays charged.
Only when the load-back runs (``_pp_load_back_extent`` truthy -- otherwise the gap is never mapped), and only inside
the load-back extent (counted from ``len(prefix_indices)``, the clamp keeps the first ``extent`` rows of the walk).

UNREADABLE (a node without an int FULL lock, a broken parent chain, any exception): 0 -- the old price. Never raises:
the gate runs on every rank and its verdict enters the group agreement (``tp_admission_agree``, R11); trees are
replicated, so every rank reads the same credit, and the gate stays this rank's number like ``prefix_len``."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from flliper.srt.pdflip.d_leaving_rows import _full_cell, _key_rows, _Unreadable

logger = logging.getLogger(__name__)

#: a parent walk longer than this is a broken chain (a 1M-token path in 16-token nodes is 65536 hops)
MAX_HOPS = 1 << 20

#: every fallback to the old price by kind, cumulative per process (KVD review MUSS 3 pattern: no silent fallback)
FALLBACKS: Dict[str, int] = {"unreadable": 0, "broken": 0, "error": 0}
_last_log: Dict[str, float] = {}


def _fallback(kind: str) -> int:
    FALLBACKS[kind] = FALLBACKS.get(kind, 0) + 1
    now = time.monotonic()
    if now - _last_log.get(kind, -1e9) >= 10.0:
        _last_log[kind] = now
        logger.warning("D-PREFIX-PRICE-FALLBACK kind=%s n=%d (no device-gap credit: the price stays the old one)",
                       kind, FALLBACKS[kind])
    return 0


def locked_device_gap(tree: Any, req: Any, extent: Optional[int]) -> int:
    """Device rows of the match path below ``req.last_node`` that another holder keeps locked and the load-back of
    ``extent`` rows maps without allocating (module doc). 0 without a load-back, without a gap, or unreadable."""
    try:
        if not extent or int(extent) <= 0:
            return 0
        extent = int(extent)
        top = getattr(req, "last_node", None)
        node = getattr(req, "best_match_node", None)
        if node is None or top is None or node is top:
            return 0
        root = getattr(tree, "root_node", None) if tree is not None else None
        path: List[Any] = []                   # (rows on the device or None, FULL lock, span) bottom-up
        hops = 0
        while node is not top:
            if node is None or node is root or hops >= MAX_HOPS:
                return _fallback("broken")     # the walk never meets last_node: not the load-back's path
            rows, lock = _full_cell(node)      # read while walking: an unreadable class stops at its first node
            path.append((rows, lock, _key_rows(node) if rows is None else rows))
            node, hops = getattr(node, "parent", None), hops + 1
        credit, cum = 0, 0
        for rows, lock, span in reversed(path):  # top-down: the load-back's row order
            if cum >= extent:
                break
            if rows is not None and lock >= 1:
                credit += max(0, min(cum + span, extent) - cum)
            cum += span
        return credit
    except _Unreadable:
        return _fallback("unreadable")
    except Exception:  # noqa: BLE001 -- a price term never breaks the gate: the old price
        return _fallback("error")
