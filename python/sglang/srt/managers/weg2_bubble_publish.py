"""write_back plus a BUBBLE PUBLISHER (user decision 2026-09-16).

The write policy stays write_back: a node's KV leaves the card only when the
card evicts it or the publish sweep before a flip asks for it. What changed
on boots xsn134-146 is WHEN the sweep pays: everything at the flip, 2-4 s of
drain after three long prefills, while write_through paid it during the
prefill and cost the main thread 2-9 % of its rate (the backup thread as a
GIL competitor, measured xsn145).

This module issues the same publish sweep, bounded and rate-limited, in the
BUBBLES of the PP event loop -- the gaps in which this rank has no batch and
the card waits for the pipeline anyway (``bubble_ms`` 70-280 ms per chunk on
xsn140) -- and hands the backup thread a soft gate so its host->arena copies
prefer those gaps too. Nothing is left for the flip, and the forwards are
not competed with. Armed by SGLANG_WEG2_BUBBLE_PUBLISH=1 (launcher default).
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_BUBBLE_PUBLISH"
MIN_INTERVAL_S = 0.2
MAX_ISSUE = 16


def armed(scheduler) -> bool:
    cached = getattr(scheduler, "_weg2_bubble_armed", None)
    if cached is None:
        cached = (
            os.environ.get(ENV, "0") == "1"
            and bool(getattr(scheduler, "enable_hicache_storage", False))
            and getattr(scheduler, "tree_cache", None) is not None
            and hasattr(scheduler.tree_cache, "publish_unbacked_sweep")
        )
        scheduler._weg2_bubble_armed = cached
    return bool(cached)


def gate(scheduler) -> threading.Event | None:
    """The soft gate the backup thread reads: set = this rank is in a bubble."""
    controller = getattr(getattr(scheduler, "tree_cache", None), "cache_controller", None)
    if controller is None:
        return None
    ev = getattr(controller, "_weg2_bubble_gate", None)
    if ev is None:
        ev = controller._weg2_bubble_gate = threading.Event()
    return ev


def bubble_begin(scheduler) -> None:
    """Called where the PP loop notes 'no batch': open the gate, issue a
    bounded sweep at most every MIN_INTERVAL_S."""
    if not armed(scheduler):
        return
    ev = gate(scheduler)
    if ev is not None:
        ev.set()
    now = time.monotonic()
    last = getattr(scheduler, "_weg2_bubble_last", 0.0)
    if now - last < MIN_INTERVAL_S:
        return
    scheduler._weg2_bubble_last = now
    try:
        stats = scheduler.tree_cache.publish_unbacked_sweep(max_issue=MAX_ISSUE)
    except Exception as e:  # noqa: BLE001 - a publisher never takes the loop down
        logger.warning("WEG2 BUBBLE-PUBLISH raised %s: %s", type(e).__name__, e)
        return
    issued = int((stats or {}).get("issued", 0) or 0)
    n = getattr(scheduler, "_weg2_bubble_sweeps", 0) + 1
    scheduler._weg2_bubble_sweeps = n
    scheduler._weg2_bubble_issued = getattr(scheduler, "_weg2_bubble_issued", 0) + issued
    if issued and (n <= 16 or n % 64 == 0):
        logger.info("WEG2 BUBBLE-PUBLISH sweep=%d %s cumulative_issued=%d",
                    n, stats, scheduler._weg2_bubble_issued)


def bubble_end(scheduler) -> None:
    """Called when a batch is launched: close the gate."""
    if not armed(scheduler):
        return
    ev = gate(scheduler)
    if ev is not None:
        ev.clear()


def backup_wait_for_bubble(controller, timeout_s: float = 0.05) -> bool:
    """Backup thread, before each storage batch: prefer a bubble, never
    block for more than ``timeout_s`` (the D group decodes without bubbles;
    a hard gate would stop its backups). True = copied inside a bubble."""
    ev = getattr(controller, "_weg2_bubble_gate", None)
    if ev is None:
        return False
    return bool(ev.wait(timeout_s))
