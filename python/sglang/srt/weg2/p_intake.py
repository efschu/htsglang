"""H91a (25.09.2026): #1400's told verdict stands until the request leaves
P's waiting queue. 27B port of the NF fix bf6c97171b (desk/nf-h91a-p-0925),
part 1 only -- the told bookkeeping; the NF commit's second part (the
free + evictable intake verdict in ``_weg2_intake_stall_observe``) is NOT
ported, the 27B intake stall keeps its pre-H91 answer.

THE SPECIMEN (NF). cu130 acceptance dkrnfbar1agent09252021 (1a4e6ff19a), P
log 20:40:49-20:42:10: rid 53572be2 (42,442 tokens) sat in P's queue for
61 s with NOTHING running and ``pool_avail=220480``; the admission-wedge
detector answered it 503 WEG2-INTAKE-STALL (``gate=admission-wedge``) and the
front paid a flip for it.

WHAT BLOCKED. #1400's told verdict (``weg2_store_told.admission``) is taken
ONCE: it pops ``told_map[rid]`` (and the loaded credit) on the first
admission visit. The adder had no room for that request in that pass (it ran
beside the predecessor's chunked continuation), so it stayed queued. Every
later visit found no told (``pp0_publish`` had already dropped the rid from
its held set, so nothing re-publishes) and skipped it as
``weg2_store_told_pending`` for ever. The 27B tree carries the identical
gate and call site (scheduler.py, the #1400 branch of the admission loop).

:func:`told_admission` KEEPS a rank's told verdict until the request leaves
the waiting queue. A later visit of the SAME request object admits on the
kept verdict (same prefix cap, same credit); a fresh told for the rid (a
re-intake) replaces it; :func:`settle_told` drops what left the queue
(admitted or aborted). Every rank keeps its own verdict the same way, so the
ranks stay equal.

Bookkeeping only; nothing here allocates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

#: scheduler attribute holding the kept verdicts ({rid: _Kept})
KEPT_ATTR = "_weg2_told_kept"
_LOG_FIRST = 8
_LOG_EVERY = 256


@dataclass
class _Kept:
    req: Any
    told: int
    credit: int


def _kept(scheduler) -> Dict[str, _Kept]:
    kept = getattr(scheduler, KEPT_ATTR, None)
    if kept is None:
        kept = {}
        setattr(scheduler, KEPT_ATTR, kept)
    return kept


def told_admission(
    scheduler,
    req,
    note_skip: Callable[[str, Any], None],
    admission: Callable[..., Optional[int]],
) -> Optional[int]:
    """The #1400 admission gate with the verdict kept across visits.

    ``admission`` is ``weg2_store_told.admission``. Returns what it returns:
    ``None`` = skip this pass (no told yet), else the loaded credit."""
    kept = _kept(scheduler)
    rid = str(getattr(req, "rid", ""))
    told_map = getattr(scheduler, "_weg2_store_told", None) or {}
    entry = kept.get(rid)
    if entry is not None:
        if entry.req is req and rid not in told_map:
            try:
                req._weg2_prefix_cap = int(entry.told)
            except Exception:  # noqa: BLE001 - a double without the field
                pass
            n = getattr(scheduler, "_h91_told_kept_n", 0) + 1
            scheduler._h91_told_kept_n = n
            if n <= _LOG_FIRST or n % _LOG_EVERY == 0:
                logger.info(
                    "H91 STORE-TOLD KEPT rid=%s told=%d credit=%d (n=%d): the "
                    "verdict of an earlier visit that did not admit stands "
                    "(it used to be consumed there and the rid skipped as "
                    "weg2_store_told_pending for ever: cu130 53572be2)",
                    rid[:8], entry.told, entry.credit, n,
                )
            return entry.credit
        # another request object under this rid, or a fresh told arrived for
        # it (re-intake): the old verdict is not this visit's
        kept.pop(rid, None)
    told = told_map.get(rid)
    credit = admission(scheduler, req, note_skip)
    if credit is not None and told is not None:
        kept[rid] = _Kept(req=req, told=int(told), credit=int(credit))
    return credit


def settle_told(scheduler, waiting_queue: Iterable[Any]) -> int:
    """After a committed pass: drop the kept verdicts of requests that are no
    longer queued (admitted into the batch, or aborted). Returns the count."""
    kept = getattr(scheduler, KEPT_ATTR, None)
    if not kept:
        return 0
    queued = {id(r) for r in waiting_queue}
    gone = [rid for rid, e in kept.items() if id(e.req) not in queued]
    for rid in gone:
        kept.pop(rid, None)
    return len(gone)


__all__ = [
    "KEPT_ATTR",
    "settle_told",
    "told_admission",
]
