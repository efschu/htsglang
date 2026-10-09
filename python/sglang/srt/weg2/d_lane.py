"""PRIORITY LANES 1008, part L2 (D side): the floor, the lane of a request and the lane hold.

Plan: deskq/PLAN-PRIO-LANES-1008.md section 2 "D (Decode)".  Names (RPC, markers, switch) are fixed in
``weg2/lanes.py`` (L1); this module holds what only the scheduler half of group D needs:

* the FLOOR the scheduler last took (``weg2_lane_floor`` / ``weg2_lane_epoch`` on the scheduler, set by
  ``POST /weg2/lane_floor`` -- ``d_park_runtime.lane_floor``): D's admission lets in only requests of a lane
  at or above it;
* the LANE of a request: ``req.priority`` (``schedule_batch.Req.priority``, the integer the front forwarded)
  read as ``lanes.parse_lane`` reads the field -- no number, a negative or a malformed value is lane 0;
* the HOLD of a lane park: a request ``park_running(rids=..., hold="lane")`` parked carries
  :data:`LANE_HOLD_ATTR`.  A held request takes part in NEITHER the 30-s awake re-queue (``park_tick``) NOR
  the #248h capacity re-queue NOR the sleep's dormant hold, and ``d_seats.order_waiting`` does not prefer it;
  it leaves the hold in exactly one place -- ``d_park_runtime.lane_floor``, when the floor falls to its lane.

SWITCH OFF (``SGLANG_WEG2_LANES=0``, the default): the floor stays 0, no request ever carries the hold
attribute, so every function here answers "no lane" and every caller in the park path reads exactly what it
read before.  Nothing in this module reads a clock or a rank-local quantity: the floor arrives as one
broadcast control request and the lane of a request is a field of the request, so every rank decides alike.
"""
from __future__ import annotations

from typing import Any

#: scheduler attributes: the floor / epoch the last accepted ``/weg2/lane_floor`` set (start 0 / 0)
FLOOR_ATTR = "weg2_lane_floor"
EPOCH_ATTR = "weg2_lane_epoch"

#: the hold value ``park_running(hold=...)`` takes, and the request attribute that carries it
HOLD_LANE = "lane"
LANE_HOLD_ATTR = "_weg2_lane_hold"

#: log line of an accepted floor (the requeue has its own line, ``lanes.MARK_D_PARK_REQUEUE``)
MARK_D_FLOOR = "WEG2-D-LANE floor"
#: the scheduler's census key of a request the floor keeps out of an admission pass
SKIP_KEY = "weg2_lane_floor"


def lane_of_req(req: Any) -> int:
    """The lane of a scheduler request: ``req.priority`` read as :func:`lanes.parse_lane` reads the field.
    ``None`` (the request carried no priority) is lane 0. Never raises."""
    from sglang.srt.weg2 import lanes

    return lanes.parse_lane(getattr(req, "priority", None))[0]


def floor_of(sched: Any) -> int:
    """The floor this scheduler took (0 = no lane is held back)."""
    try:
        return max(0, int(getattr(sched, FLOOR_ATTR, 0) or 0))
    except (TypeError, ValueError):
        return 0


def epoch_of(sched: Any) -> int:
    try:
        return max(0, int(getattr(sched, EPOCH_ATTR, 0) or 0))
    except (TypeError, ValueError):
        return 0


def lane_held(req: Any) -> bool:
    """True while ``req`` sits in a lane park's hold."""
    return getattr(req, LANE_HOLD_ATTR, None) == HOLD_LANE


def below_floor(sched: Any, req: Any) -> bool:
    """True when the floor keeps ``req`` out of admission (its lane is below the floor)."""
    floor = floor_of(sched)
    return floor > 0 and lane_of_req(req) < floor


def clear_hold(req: Any) -> None:
    setattr(req, LANE_HOLD_ATTR, None)


def mark_hold(req: Any) -> None:
    setattr(req, LANE_HOLD_ATTR, HOLD_LANE)


def admissible_for_displace(sched: Any, waiter: Any) -> bool:
    """A waiting request the floor keeps out cannot be admitted, so it is no reason to displace a running
    one (``d_park_runtime.displace_for_age``)."""
    return not below_floor(sched, waiter)
