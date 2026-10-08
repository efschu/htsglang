"""PRIORITY LANES, part L1 (field + state): the request field, the state record and the names
that parts L2 (D side), L3 (P side) and L4 (front controller) share.

Plan: deskq/PLAN-PRIO-LANES-1008.md sections 0-2.  A request carries an integer ``priority`` (no number =
0; a higher number is a higher lane; the same number is the same lane).  Between lanes the rule is strict
displacement, inside a lane everything stays as it is today.  THIS MODULE HAS NO CONTROLLER LOGIC: it
reads the field, holds ``lane_floor`` / ``lane_epoch`` / the per-lane counters and fixes the spelling of
every marker, env name and RPC name, so L2-L4 are written against names that cannot drift.

THE SWITCH.  ``SGLANG_WEG2_LANES`` (default 0).  Off, nothing in this module is reached from a request
path: the front does not read ``priority``, writes no lane field into state.json / the request book / a
log line, and the Anthropic adapter does not forward the field (it drops it, as it always did).

NAMES FIXED FOR L2-L4 (binding)
  * RPC  ``POST /weg2/lane_floor``  body ``{"floor": int, "epoch": int}``  (:data:`RPC_LANE_FLOOR`)
  * markers: :data:`MARK_PREEMPT`, :data:`MARK_RESUME`, :data:`MARK_DEFER`, :data:`MARK_REPREFILL`,
    :data:`MARK_D_PARK_PARK`, :data:`MARK_D_PARK_REQUEUE`, :data:`MARK_PR_FLOOR`, and the field warning
    :data:`MARK_FIELD`
  * state.json ``front.lane_floor`` / ``front.lane_epoch`` / ``front.lanes`` (only with the switch on)
"""
from __future__ import annotations

import collections
import logging
import struct
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger("weg2.front")

# ---------------------------------------------------------------------------
# names (the contract with L2-L4)
# ---------------------------------------------------------------------------

#: env switch (0/1, default 0), the keepalive period of a held SSE stream in seconds, and the P-side chunk
#: size while a higher lane waits (0 = the normal chunk).  Declared in ``environ.py``.
ENV_LANES = "SGLANG_WEG2_LANES"
ENV_KEEPALIVE_S = "SGLANG_WEG2_LANE_KEEPALIVE_S"
ENV_PREEMPT_CHUNK_TOKENS = "SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS"

#: the request field (the OpenAI protocol already has it, protocol.py ``priority``)
FIELD = "priority"

#: the front -> P/D RPC that carries the floor: ``POST {group_url}/weg2/lane_floor`` with
#: ``{"floor": int, "epoch": int}``.  P: PP0 receives it and stamps the floor into the PP-room vote;
#: D: its admission reads it.  (L2 / L3 implement the endpoints; L4 sends it.)
RPC_LANE_FLOOR = "/weg2/lane_floor"
#: the two keys of the body
RPC_KEY_FLOOR = "floor"
RPC_KEY_EPOCH = "epoch"

#: log markers.  An acceptance marker is a logger emitter (the front's emitter rule); these are the
#: leading literals of the lines L2-L4 write.
MARK_PREEMPT = "WEG2 LANE-PREEMPT"      # floor n->N: epoch++, lower lanes held/parked
MARK_RESUME = "WEG2 LANE-RESUME"        # LANE-EMPTY: floor falls to M, the parked of lane M run again
MARK_DEFER = "WEG2 LANE-DEFER"          # a higher lane arrives during a flip: park after WEG2-FLIP done
MARK_REPREFILL = "WEG2 LANE-REPREFILL"  # resume found no prefix: the request is prefilled again via P (must count 0 on metal)
MARK_D_PARK_PARK = "WEG2-D-PARK park(lane)"        # D side: rids parked for a lane, hold="lane"
MARK_D_PARK_REQUEUE = "WEG2-D-PARK requeue(lane)"  # D side: held parks requeued in arrival order on a floor fall
MARK_PR_FLOOR = "PR LANE-FLOOR"          # P side: the floor stamped into the PP-room vote
#: the warning of a ``priority`` the front had to correct (negative / not a whole number / out of range)
MARK_FIELD = "WEG2 LANE-FIELD"

#: Pending.lane_state / the request book: the states of a request under the lane rule (plan section 1)
LANE_STATES = ("active", "held", "parked", "resuming", "reprefill")

#: per-lane counters in state.json ``front.lanes[<n>]``
LANE_COUNT_KEYS = ("pending", "running_p", "running_d", "parked")

#: an integer ``priority`` above this is clamped (a lane number is a small integer; this keeps the value an
#: int32 on every wire it may still travel, the scheduler's ``Req.priority`` among them)
LANE_MAX = 2**31 - 1


# ---------------------------------------------------------------------------
# env readers
# ---------------------------------------------------------------------------

def enabled() -> bool:
    """``SGLANG_WEG2_LANES`` (default off)."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_WEG2_LANES.get())


def keepalive_s() -> int:
    """``SGLANG_WEG2_LANE_KEEPALIVE_S`` (default 15): seconds between the comment lines of a held SSE stream."""
    from sglang.srt.environ import envs

    return max(0, int(envs.SGLANG_WEG2_LANE_KEEPALIVE_S.get()))


def preempt_chunk_tokens() -> int:
    """``SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS`` (default 0 = the normal chunk): the P chunk size while a higher lane waits."""
    from sglang.srt.environ import envs

    return max(0, int(envs.SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS.get()))


# ---------------------------------------------------------------------------
# the field
# ---------------------------------------------------------------------------

#: why :func:`parse_lane` corrected a value (None = taken as given or absent)
WHY_NEGATIVE = "negative"
WHY_INVALID = "invalid"
WHY_CLAMPED = "clamped"


def parse_lane(raw: Any) -> Tuple[int, Optional[str]]:
    """``(lane, why)`` of a raw ``priority`` value.

    * absent / ``None`` -> ``(0, None)`` (no number = lane 0);
    * a whole number (``int``, an integral ``float`` such as 1.0, or a string of digits as some clients
      send it) -> itself; negative -> ``(0, "negative")``; above :data:`LANE_MAX` -> clamped;
    * anything else (``bool``, a fraction, text, NaN, a list) -> ``(0, "invalid")``.

    Never raises: a malformed field costs the request its priority, never the request."""
    if raw is None:
        return 0, None
    if isinstance(raw, bool):  # True is an int in Python and never a lane
        return 0, WHY_INVALID
    n: Optional[int] = None
    if isinstance(raw, int):
        n = raw
    elif isinstance(raw, float):
        if raw == raw and raw not in (float("inf"), float("-inf")) and raw == int(raw):
            n = int(raw)
    elif isinstance(raw, str):
        s = raw.strip()
        try:
            n = int(s)
        except ValueError:
            n = None
    if n is None:
        return 0, WHY_INVALID
    if n < 0:
        return 0, WHY_NEGATIVE
    if n > LANE_MAX:
        return LANE_MAX, WHY_CLAMPED
    return n, None


def lane_of(payload: Any) -> int:
    """The lane of a request payload (the raw JSON dict the front read): ``payload["priority"]`` as
    :func:`parse_lane` reads it; a payload that is no dict is lane 0."""
    if not isinstance(payload, dict):
        return 0
    return parse_lane(payload.get(FIELD))[0]


# ---------------------------------------------------------------------------
# the state record
# ---------------------------------------------------------------------------

class LaneState:
    """``lane_floor`` (the active lane, start 0), ``lane_epoch`` (counts every floor change), the lane of
    every open rid and the per-lane counters of state.json.  NO controller logic: this class stores and
    counts; L4 decides when the floor moves and calls :meth:`set_floor`."""

    MAX_RIDS = 8192

    def __init__(self) -> None:
        self.lane_floor = 0
        self.lane_epoch = 0
        #: rid -> lane of every rid the front noted and has not seen end (bounded, oldest out)
        self.rid_lane: "collections.OrderedDict[str, int]" = collections.OrderedDict()

    # ---- the lane of a rid ----
    def note(self, rid: Any, lane: int) -> None:
        r = str(rid)
        self.rid_lane.pop(r, None)
        self.rid_lane[r] = int(lane)
        while len(self.rid_lane) > self.MAX_RIDS:
            self.rid_lane.popitem(last=False)

    def lane_for(self, rid: Any) -> int:
        """The noted lane of ``rid``; a rid the front never noted (or whose note aged out) is lane 0."""
        return self.rid_lane.get(str(rid), 0)

    def end(self, rid: Any) -> None:
        self.rid_lane.pop(str(rid), None)

    # ---- the floor ----
    def set_floor(self, floor: int) -> bool:
        """Move the floor; the epoch counts only a real change.  True = it moved."""
        floor = max(0, int(floor))
        if floor == self.lane_floor:
            return False
        self.lane_floor = floor
        self.lane_epoch += 1
        return True

    # ---- the counters ----
    def counts(self, pending: Iterable[Any] = (), running_p: Iterable[Any] = (),
               running_d: Iterable[Any] = (), parked: Iterable[Any] = ()) -> Dict[str, Dict[str, int]]:
        """``{"<lane>": {pending, running_p, running_d, parked}}`` for the given rid sets.  A rid in more than
        one set counts once, in the order parked > running_d > running_p > pending (a parked rid still has
        its D stream open; a rid in D's hand is no longer pending).  Lanes without a request are absent;
        the keys are strings (JSON)."""
        out: Dict[str, Dict[str, int]] = {}
        seen = set()
        for key, rids in (("parked", parked), ("running_d", running_d),
                          ("running_p", running_p), ("pending", pending)):
            for rid in rids:
                r = str(rid)
                if r in seen:
                    continue
                seen.add(r)
                row = out.setdefault(str(self.lane_for(r)), {k: 0 for k in LANE_COUNT_KEYS})
                row[key] += 1
        return dict(sorted(out.items(), key=lambda kv: int(kv[0])))

    def state_block(self, pending: Iterable[Any] = (), running_p: Iterable[Any] = (),
                    running_d: Iterable[Any] = (), parked: Iterable[Any] = ()) -> Dict[str, Any]:
        """The three state.json keys (``front.lane_floor`` / ``front.lane_epoch`` / ``front.lanes``)."""
        return {"lane_floor": self.lane_floor, "lane_epoch": self.lane_epoch,
                "lanes": self.counts(pending, running_p, running_d, parked)}


# ---------------------------------------------------------------------------
# the progress beacon's lane trailer
# ---------------------------------------------------------------------------

#: progress_beacon.py keeps one 32-byte file per scheduler rank (forward_ct, t_start_ns, t_done_ns, pid).
#: With the switch on the file is 16 bytes longer: ``(lane_floor, lane_epoch)`` the rank last SAW (L2 / L3
#: write them with ``progress_beacon.beat_lane``).  The first 32 bytes keep their layout, so every reader of
#: today reads on unchanged.  A watchdog (H86 / W17 / park_stuck) that sees a rank whose floor is above the
#: lane of its waiting requests reads a hold, not a stall.
BEACON_LANE_FMT = "<qq"
BEACON_LANE_SIZE = struct.calcsize(BEACON_LANE_FMT)
