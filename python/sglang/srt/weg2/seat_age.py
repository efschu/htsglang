"""SA SEAT BY AGE (#244 rebuilt to the user's design, 27.09.2026).

User, verbatim: "der request der zuletzt ankam wird zuletzt bedient, außer er passt zufällig in einen
sitz, weil er wenig kv verbraucht, ein anderer sitz der mehr kv brauchen würde aber nicht rechnen
kann und eigentlich vorher dran wäre, aber wegen des vrams nicht kann würde ja die leistung unnötig
verschwenden. sobald dann der älteste request fertig ist, rückt der zweitälteste nach, falls er
vorher parken musste und würde ggf. jüngere aktuell noch laufende requests verdrängen ganz oder
teilweise"

THE ORDER. One priority over running, parked and waiting requests: the FIRST ARRIVAL at the front.
The front's rid is ``weg2-<epoch>-<seq>`` with ``seq`` its arrival counter, kept by every re-route
(the rid never changes) -- so the age is read off the rid, identically on the front and on every D
rank, with no new field and no collective (:func:`rid_age`).

PRIO-AS-AGE 1014 (user 14.10.2026, weg2/prio_age.py): the request's ``priority`` acts as AGE -- the order
is the pair (priority desc, front arrival asc), :func:`order_key`. Every function below takes an optional
``lane_of``; without it every request is priority 0 and the order is the arrival alone, as before.

  (1) seats in age order, while seats (bs) and D's KV reach; a younger request that fits may run when
      the next older one does not fit the free KV (backfill -- no idle capacity);
  (2) when a seat frees, the next older one moves in, a parked one included;
  (3) an older request waiting while every seat is held by younger ones displaces the YOUNGEST
      running one (whole park: retract with the span retained, a pressure park). Partial parking
      (only part of a request's KV out) does not exist in this tree -- named follow-up work;
  (4) every park victim is the youngest by arrival (never the last admitted).

Switch ``SGLANG_WEG2_SEAT_ROTATE`` (this semantics, default on; ``0`` = parked resume first, the
pre-#244 behaviour).
"""
from __future__ import annotations

import os
import re
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

ENV = "SGLANG_WEG2_SEAT_ROTATE"
_RID_RE = re.compile(r"^weg2-\d+-(\d+)")
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


def order_key(rid, lane: int = 0, on: Optional[bool] = None) -> Tuple[int, int]:
    """PRIO-AS-AGE 1014 (weg2/prio_age.py): THE comparison key of the SA order -- ``(-priority, rid_age)``
    with the mode on (a higher priority is OLDER than every lower one; inside a priority the front arrival
    decides), ``(0, rid_age)`` off.  Smaller = older.  Both halves are replicated (the request's own
    ``priority`` field, the front's rid counter), so every rank keys alike."""
    from sglang.srt.weg2 import prio_age as _pa

    return _pa.key(lane, rid_age(rid), on=on)


def _lane_fn(lane_of: Optional[Callable]) -> Callable:
    return (lambda _r: 0) if lane_of is None else lane_of


def by_age(items: Iterable, rid_of: Callable = lambda x: getattr(x, "rid", x),
           lane_of: Optional[Callable] = None) -> List:
    """``items`` oldest first (stable for equal / unknown ages).  ``lane_of(item)`` (PRIO-AS-AGE 1014): the
    item's priority -- the order is then ``(priority desc, age asc)``; None = every item priority 0 (the old
    order)."""
    from sglang.srt.weg2 import prio_age as _pa

    on = _pa.enabled() if lane_of is not None else False
    lf = _lane_fn(lane_of)
    return sorted(items, key=lambda x: order_key(rid_of(x), lf(x) if on else 0, on=on))


def plan_seats(candidates: Sequence[Tuple[str, int]], seats: int,
               budget_tokens: Optional[int] = None, lane_of: Optional[Callable] = None) -> List[str]:
    """(1): the rids that get a seat, oldest first with backfill.
    ``candidates`` = (rid, KV tokens it needs); ``budget_tokens`` None = seats only.
    ``lane_of(rid)`` (PRIO-AS-AGE 1014): the rid's priority; "oldest" is then the order key."""
    out: List[str] = []
    left = None if budget_tokens is None else int(budget_tokens)
    lf = None if lane_of is None else (lambda c: lane_of(c[0]))
    for rid, need in by_age(candidates, rid_of=lambda c: c[0], lane_of=lf):
        if len(out) >= int(seats):
            break
        if left is not None and int(need) > left:
            continue  # backfill: a younger one that fits may take the seat
        out.append(rid)
        if left is not None:
            left -= int(need)
    return out


def displace_victim(waiting_rids: Sequence[str], running_rids: Sequence[str],
                    seats_full: bool, lane_of: Optional[Callable] = None) -> Optional[Tuple[str, str]]:
    """(3)/(4): ``(older_waiting, youngest_running)`` when the oldest waiting
    request is older than the youngest running one and no seat is free; None
    otherwise. Never a younger over an older.

    PRIO-AS-AGE 1014: ``lane_of(rid)`` gives the priority; "oldest" / "youngest" are then the order key
    (:func:`order_key`) -- the victim is the YOUNGEST of the LOWEST priority, exactly one, and only when
    the waiter's key is older (a priority-0 waiter never displaces a priority-1 runner)."""
    if not seats_full or not waiting_rids or not running_rids:
        return None
    from sglang.srt.weg2 import prio_age as _pa

    on = _pa.enabled() if lane_of is not None else False
    lf = _lane_fn(lane_of)

    def k(r):
        return order_key(r, lf(r) if on else 0, on=on)

    oldest_wait = min(waiting_rids, key=k)
    youngest_run = max(running_rids, key=k)
    if k(oldest_wait) < k(youngest_run):
        return oldest_wait, youngest_run
    return None


def is_older(a_rid, b_rid, lane_of: Optional[Callable] = None) -> bool:
    """``a`` is served before ``b`` (strictly older by the order key)."""
    from sglang.srt.weg2 import prio_age as _pa

    on = _pa.enabled() if lane_of is not None else False
    lf = _lane_fn(lane_of)
    return order_key(a_rid, lf(a_rid) if on else 0, on=on) < order_key(b_rid, lf(b_rid) if on else 0, on=on)
