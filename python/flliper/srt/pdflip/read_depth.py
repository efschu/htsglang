"""READ-DEPTH 1014 (C5 of done/nf-row-inventory-1014.md): ONE line per short
store read that names WHERE the read's prefix ended and WHY, on every stage.

THE GAP (NF h16h 853534a9ff, boot _211215, P, 30 min): five rids where the
three P stages landed the same prefix at different depths (pdflip-51-579 PP0
300 / PP1 1556 / PP2 694 of 1556 pages; pdflip-60-639 PP0+PP2 304 of 1204;
pdflip-68-683 PP0+PP1 0, PP2 144; pdflip-72-690 PP0 557 of 636; pdflip-78-784
PP0+PP2 761 of 1148). The told is PP0's read; every stage that read less
than the store held cost the group the difference (~177k tokens). The log
carried the landed count (``SA SHORT-READ ANCHOR landed=``) but not the
cause: the ``L3-FILL`` lines that could explain it (``LATE-COMPLETE
REFUSED``, ``JOIN-WAIT still_claimed=``, ``ROOM-WAIT still_full=``) are
rate limited per process (k <= 16 or a power of two), and the causes that
end a walk silently (a page evicted between the find / the fill and the
reference) have no line at all. The Prefill-Cache monitor could not count
causes.

THE LINE (rank-local, an instrument -- it decides nothing)::

    PDFLIP-READ-DEPTH rid=<rid> pp=<rank> landed=<pages> of=<pages> anchor=<pages>
        cut_at=<page> cause=<cause> ms=<read ms> census=<cause:n,...>

once per read that ended short of its own key list (landed < of), on every
stage, never rate limited (the boot had 18 short reads in 2.7 h). The rid
joins the three stages' lines: that is the read cap of every stage per rid.
``census`` is this rank's running count per cause.

Causes (the first page the walk did not take):

* ``late_complete`` -- the fill's own claim was recycled (3) or reaped as
  stale (6) under it (the arena reap race: READ-DEPTH 1014 fix in arena.c);
* ``join_unfinished`` -- another writer's claim (a JOIN) did not complete
  inside the JOIN wait (or its stem vanished);
* ``inflight_unfinished`` -- the same for a page not yet on disk that a live
  writer was publishing (IN-FLIGHT);
* ``arena_full`` -- no slot after the ROOM wait;
* ``read_failed`` -- the disk read or the completion failed;
* ``not_on_disk`` -- the store does not hold the page (the probe's hit was
  above what the store has now);
* ``l2_evicted_before_ref`` -- a page found COMPLETE in L2 was evicted
  between the find and the reference;
* ``fill_evicted_before_ref`` -- a page the fill landed was evicted (or
  re-claimed) before the reference (REF-AT-LAND closes this);
* ``fill_none`` -- the fill returned no slot without a recorded reason;
* ``terminated`` -- the operation was stopped at a batch boundary (revoke,
  stop, draft claim);
* ``arena_chain_recheck`` -- (integration line, 89e7f70d7e ARENA-CHAIN) the
  reference recheck found the stem no longer naming the referenced slot;
* ``unknown`` -- short with none of the above recorded.

The anchor is the hybrid controller's SA anchor (the deepest page <= landed
whose recurrent state was read; 0 = none); ``of`` is the read's own key
list (the probe's hit), not the request length."""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

LATE_COMPLETE = "late_complete"
JOIN_UNFINISHED = "join_unfinished"
INFLIGHT_UNFINISHED = "inflight_unfinished"
ARENA_FULL = "arena_full"
READ_FAILED = "read_failed"
NOT_ON_DISK = "not_on_disk"
L2_EVICTED_BEFORE_REF = "l2_evicted_before_ref"
FILL_EVICTED_BEFORE_REF = "fill_evicted_before_ref"
FILL_NONE = "fill_none"
TERMINATED = "terminated"
ARENA_CHAIN_RECHECK = "arena_chain_recheck"   # the integration line's ARENA-CHAIN reference recheck cut
UNKNOWN = "unknown"

_TL = threading.local()
_CENSUS: Dict[str, int] = {}
_CENSUS_LOCK = threading.Lock()
_N = [0]


# --- the fill side (hicache_storage.HiCacheFile.arena_fill_from_disk) --------

def fill_reasons_reset() -> None:
    """Start of a fill on this thread: no reasons recorded yet."""
    _TL.fill = {}


def fill_reason(index: int, cause: str) -> None:
    """The fill could not give stem ``index`` (position in ITS stem list) a
    slot, for ``cause``. The first reason of an index wins."""
    d = getattr(_TL, "fill", None)
    if d is None:
        d = _TL.fill = {}
    d.setdefault(int(index), str(cause))


def fill_reasons_take() -> Dict[int, str]:
    """The reasons of the last fill on this thread (and forget them)."""
    d = getattr(_TL, "fill", None) or {}
    _TL.fill = {}
    return d


# --- the walk side (cache_controller._arena_page_get / _page_transfer) ------

def note_cut(operation, page: int, cause: str) -> None:
    """The read of ``operation`` ended at absolute page ``page`` for
    ``cause``. The LOWEST cut of an operation wins: the walk's prefix ends
    there (a later check that cuts below an earlier break -- e.g. a reference
    recheck after the walk -- names the page the read really ended at)."""
    try:
        cur = getattr(operation, "_pdflip_read_cut", None)
        if cur is None or int(page) < int(cur[0]):
            operation._pdflip_read_cut = (int(page), str(cause))
    except Exception:  # noqa: BLE001 - an instrument never breaks a read
        pass


def cut_of(operation):
    return getattr(operation, "_pdflip_read_cut", None)


def _census_text() -> str:
    with _CENSUS_LOCK:
        return ",".join(f"{k}:{v}" for k, v in sorted(_CENSUS.items()))


def census() -> Dict[str, int]:
    with _CENSUS_LOCK:
        return dict(_CENSUS)


def line(operation, page_size: int, pp_rank, total_ms: float) -> Optional[str]:
    """Emit the READ-DEPTH line for a short read; returns the cause (None =
    the read was not short, nothing emitted). Never raises."""
    try:
        of = len(getattr(operation, "hash_value", ()) or ())
        ps = max(1, int(page_size))
        landed = int(getattr(operation, "completed_tokens", 0) or 0) // ps
        if of <= 0 or landed >= of:
            return None
        cut = cut_of(operation)
        if cut is None:
            cut_at, cause = landed, UNKNOWN
        else:
            cut_at, cause = int(cut[0]), str(cut[1])
        anchor = getattr(operation, "_pdflip_short_anchor_pages", None)
        anchor = landed if anchor is None else int(anchor)
        with _CENSUS_LOCK:
            _CENSUS[cause] = _CENSUS.get(cause, 0) + 1
        _N[0] += 1
        logger.warning(
            "PDFLIP-READ-DEPTH rid=%s pp=%s landed=%d of=%d anchor=%d cut_at=%d cause=%s ms=%.0f "
            "census=%s (n=%d): this stage's store read ended short of its own key list -- the "
            "rid joins the three stages' lines (the read cap of every stage); PP0's is the told",
            getattr(operation, "request_id", "?"), pp_rank, landed, of, anchor, cut_at, cause,
            float(total_ms), _census_text(), _N[0],
        )
        return cause
    except Exception:  # noqa: BLE001 - an instrument never breaks a read
        return None
