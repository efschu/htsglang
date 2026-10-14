"""PRIO-AS-AGE 1014 (user decision 14.10.2026 18:27Z): a request's ``priority`` acts as AGE, not as a lane floor.

User, verbatim: "priority 1 immer älter ... in einer priority klasse ist es aber natürlich wieder altersabhängig
wer bedient wird".

THE RULE.  Every place that orders or compares requests by age ("oldest first", "the youngest is the victim",
"an older one keeps its right of way") orders by the PAIR

    (priority descending, age ascending)

-- a request of priority p+1 is older than every request of priority p; inside one priority the real age decides.
Priorities 0..1000 and more stay valid (``lanes.parse_lane``: a whole number >= 0, clamped to int32).  This module
is the ONE spelling of that pair; nothing adds a year to a timestamp -- only the COMPARISON KEY changes, and every
clock (wait bounds, liveness, collect windows, flip urgency's ``oldest_wait_s``, starvation bounds, DP-WAIT,
monitors) keeps reading the real arrival.

WHAT IT SWITCHES OFF.  With the mode on, the lane FLOOR machinery of PLAN-PRIO-LANES-1008 -- LANE-PREEMPT, the floor
(LANE-RAISE / LANE-RESUME), LANE-WAITER-HOLD, the lane hold of P legs, D's lane park (``weg2_lane_held``,
``WEG2-D-PARK park(lane)``) -- is never entered: the floor stays 0, so a lower priority is never parked or held for a
higher one.  The code stays; :func:`floor_mode` is its gate.  The ``priority`` field is still read by the front
(``lanes.parse_lane``), normalised and carried to P and D (leg bodies, ``Req.priority``).

RANKS NEVER DISAGREE.  The key is a function of two replicated quantities: the request's own ``priority`` field (the
same broadcast request on every rank) and its age (the front's rid counter / ``kv_arrival_seq`` -- both replicated).
Where an order already exists in a group, :func:`lane_sorted` applies a STABLE sort by priority only, so the order
inside a priority is exactly the one the group already agreed on.

THE SWITCH.  ``SGLANG_WEG2_PRIO_AS_AGE`` (default ON).  It acts only with ``SGLANG_WEG2_LANES`` on (the front reads
``priority`` only then); off = the lane floor of 1008, byte for byte.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Tuple

#: env switch (declared in ``environ.py``)
ENV = "SGLANG_WEG2_PRIO_AS_AGE"
#: the marker of the D->P flip urgency a priority request causes (DECODE-COLLECT release / FLIP-ECONOMICS)
MARK_URGENT = "WEG2 PRIO-URGENT"

_OFF = ("0", "false", "no", "off")


def enabled(env: Optional[Any] = None) -> bool:
    """The mode is on: ``SGLANG_WEG2_LANES`` on AND ``SGLANG_WEG2_PRIO_AS_AGE`` on (both default on).

    ``env`` (a mapping) is for tests that hand an environment instead of patching ``os.environ``."""
    if env is not None:
        lanes_raw = str(env.get("SGLANG_WEG2_LANES", "1") or "1").strip().lower()
        raw = str(env.get(ENV, "1") or "1").strip().lower()
        return lanes_raw not in _OFF and raw not in _OFF
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_LANES.get()) and bool(envs.SGLANG_WEG2_PRIO_AS_AGE.get())
    except Exception:  # noqa: BLE001 -- an unreadable switch is the default (on), never a crash in an order
        return True


def floor_mode() -> bool:
    """The lane FLOOR controller of 1008 (LANE-PREEMPT, floor RPC, lane hold, D lane park) may act: lanes on and
    PRIO-AS-AGE off.  Every entry into that machinery tests this, not ``lanes.enabled()``."""
    try:
        from sglang.srt.weg2 import lanes

        return bool(lanes.enabled()) and not enabled()
    except Exception:  # noqa: BLE001
        return False


def lane(raw: Any) -> int:
    """The priority of a raw value (``None`` / malformed / negative -> 0), as the front normalises it."""
    from sglang.srt.weg2 import lanes

    return lanes.parse_lane(raw)[0]


def lane_of_req(req: Any) -> int:
    """The priority of a scheduler request (``Req.priority``) or of any object with a ``priority`` / ``lane``
    attribute (a front Pending carries ``lane``)."""
    v = getattr(req, "priority", None)
    if v is None:
        v = getattr(req, "lane", None)
    return lane(v)


def key(prio: Any, age: Any, on: Optional[bool] = None) -> Tuple[int, Any]:
    """THE order key: ``(-priority, age)`` with the mode on, ``(0, age)`` off.  Smaller = OLDER = served first.
    ``on``: the caller's one read of :func:`enabled` for a whole sort (None = read it here)."""
    if on is None:
        on = enabled()
    if not on:
        return (0, age)
    try:
        p = int(prio or 0)
    except (TypeError, ValueError):
        p = 0
    return (-max(0, p), age)


def lane_sorted(items: Iterable[Any], lane_of: Callable[[Any], int] = lane_of_req,
                on: Optional[bool] = None) -> List[Any]:
    """``items`` with the higher priority first and, inside one priority, the order they HAD (stable).  The tool
    for an order a group already agreed on (it keeps that order inside a priority).  Off: ``list(items)``."""
    items = list(items)
    if on is None:
        on = enabled()
    if not on or len(items) < 2:
        return items
    lanes_ = []
    for x in items:
        try:
            lanes_.append(int(lane_of(x) or 0))
        except Exception:  # noqa: BLE001 -- an unreadable priority is 0
            lanes_.append(0)
    if not any(lanes_):
        return items
    order = sorted(range(len(items)), key=lambda i: -lanes_[i])
    return [items[i] for i in order]


def head_index(items: Iterable[Any], lane_of: Callable[[Any], int] = lane_of_req,
               on: Optional[bool] = None) -> int:
    """The index of the request served first in ``items`` (kept in arrival order): the first one of the highest
    priority.  0 with the mode off or an empty sequence."""
    if on is None:
        on = enabled()
    if not on:
        return 0
    best_i, best = 0, None
    for i, x in enumerate(items):
        try:
            p = int(lane_of(x) or 0)
        except Exception:  # noqa: BLE001
            p = 0
        if best is None or p > best:
            best_i, best = i, p
    return best_i
