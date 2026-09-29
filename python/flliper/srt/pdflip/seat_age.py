"""SA SEAT BY AGE (#244 rebuilt to the user's design, 27.09.2026).

User, verbatim: "der request der zuletzt ankam wird zuletzt bedient, außer er passt zufällig in einen
sitz, weil er wenig kv verbraucht, ein anderer sitz der mehr kv brauchen würde aber nicht rechnen
kann und eigentlich vorher dran wäre, aber wegen des vrams nicht kann würde ja die leistung unnötig
verschwenden. sobald dann der älteste request fertig ist, rückt der zweitälteste nach, falls er
vorher parken musste und würde ggf. jüngere aktuell noch laufende requests verdrängen ganz oder
teilweise"

THE ORDER. One priority over running, parked and waiting requests: the FIRST ARRIVAL at the front.
The front's rid is ``pdflip-<epoch>-<seq>`` with ``seq`` its arrival counter, kept by every re-route
(the rid never changes) -- so the age is read off the rid, identically on the front and on every D
rank, with no new field and no collective (:func:`rid_age`).

  (1) seats in age order, while seats (bs) and D's KV reach; a younger request that fits may run when
      the next older one does not fit the free KV (backfill -- no idle capacity);
  (2) when a seat frees, the next older one moves in, a parked one included;
  (3) an older request waiting while every seat is held by younger ones displaces the YOUNGEST
      running one (whole park: retract with the span retained, a pressure park). Partial parking
      (only part of a request's KV out) does not exist in this tree -- named follow-up work;
  (4) every park victim is the youngest by arrival (never the last admitted).

Switch ``FLLIPER_PDFLIP_SEAT_ROTATE`` (this semantics, default on; ``0`` = parked resume first, the
pre-#244 behaviour).
"""
from __future__ import annotations

import os
import re
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

ENV = "FLLIPER_PDFLIP_SEAT_ROTATE"
_RID_RE = re.compile(r"^pdflip-\d+-(\d+)")
#: a rid that does not carry the front's counter sorts after every front rid
#: (youngest), in its given order.
UNKNOWN_AGE = 1 << 62


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def rid_age(rid) -> int:
    """The front's arrival counter of ``rid`` (smaller = older)."""
    m = _RID_RE.match(str(rid or ""))
    return int(m.group(1)) if m else UNKNOWN_AGE


def by_age(items: Iterable, rid_of: Callable = lambda x: getattr(x, "rid", x)) -> List:
    """``items`` oldest first (stable for equal / unknown ages)."""
    return sorted(items, key=lambda x: rid_age(rid_of(x)))


def plan_seats(candidates: Sequence[Tuple[str, int]], seats: int,
               budget_tokens: Optional[int] = None) -> List[str]:
    """(1): the rids that get a seat, oldest first with backfill.
    ``candidates`` = (rid, KV tokens it needs); ``budget_tokens`` None = seats only."""
    out: List[str] = []
    left = None if budget_tokens is None else int(budget_tokens)
    for rid, need in by_age(candidates, rid_of=lambda c: c[0]):
        if len(out) >= int(seats):
            break
        if left is not None and int(need) > left:
            continue  # backfill: a younger one that fits may take the seat
        out.append(rid)
        if left is not None:
            left -= int(need)
    return out


def displace_victim(waiting_rids: Sequence[str], running_rids: Sequence[str],
                    seats_full: bool) -> Optional[Tuple[str, str]]:
    """(3)/(4): ``(older_waiting, youngest_running)`` when the oldest waiting
    request is older than the youngest running one and no seat is free; None
    otherwise. Never a younger over an older."""
    if not seats_full or not waiting_rids or not running_rids:
        return None
    oldest_wait = min(waiting_rids, key=rid_age)
    youngest_run = max(running_rids, key=rid_age)
    if rid_age(oldest_wait) < rid_age(youngest_run):
        return oldest_wait, youngest_run
    return None
