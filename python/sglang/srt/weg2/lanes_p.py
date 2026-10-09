"""PRIORITY LANES, part L3 (P side): the lane floor on the PP ranks of group P, and the rules that read it.

Plan: deskq/PLAN-PRIO-LANES-1008.md section 2 ("P (Prefill, PP=3)"), line L3, risks section 6.  Names (RPC, markers,
env) are fixed in ``weg2/lanes.py`` (L1); this module only USES them.

WHAT P CAN AND CANNOT DO.  There is no commandable in-flight park on P: the one park a running prefill knows is the
pressure park (``schedule_policy.add_chunked_req`` #679 -> ``weg2_pool_parked`` -> ``Scheduler.process_pending_weg2_park``,
``release_kv_cache(is_insert=True)``: the computed span stays in the tree, the resume is an ordinary prefix match).  The lane
park is the SAME mechanism with another trigger: a running prefill whose lane is below the floor is not given its next
chunk (it parks in place at the chunk boundary), and at the head of the next step its rows go back to the tree and it
returns to the head of the waiting queue; while the floor stays above its lane the admission loop skips it; when the floor
falls it is admitted whole (``weg2_parked_span``) with the prefix the tree still holds -- no recomputation of the anchor.

RANK-UNIFORMITY (the heavy part, #1004).  Three stages must act on one floor in the SAME pass, or they admit different
batches (``#1004 SLOT DISAGREEMENT``).  The floor reaches PP0 as ``POST /weg2/lane_floor`` (a one-way control request, only
PP0's scheduler acts on it).  PP0 does NOT apply it when the request arrives: it takes the newest (floor, epoch) at the
top of its next pass (:func:`pp0_pass`), applies it THEN and stamps it onto list m of the request chain
(``pp_room_vote.Weg2PpLaneFloor``, the ``Weg2PpRoomCap`` convention: "PP0 decides pass m on the SAME value its followers
read in their pass m").  Every follower takes the stamp off list m after relaying it (:func:`apply_follower`) and applies it
in pass m.  PP0 keeps stamping the standing value every pass once the first epoch arrived, so a stage never depends on a
single message.  The follower facts echo the applied (floor, epoch) back (``Weg2PpRoomFact.lane_floor/lane_epoch``), so
PP0 can read whether all stages stand on one epoch.  Switch off (``SGLANG_WEG2_LANES=0``) or no RPC ever: no state, no
stamp, byte-identical lists.

WHO DECIDES WHAT.  Membership of a pass is PP0's decision (``#791`` forwarded schedule: followers admit only what the
decision names).  So: the admission loop skips a held request only on the rank that owns its admission truth
(:func:`owns_admission`: PP0 or a non-PP boot); a follower never skips a request PP0 named.  The chunked continuation is the
one request a follower decides about when the schedule does not name it: there ``add_chunked_req`` reads the SAME stamped
floor, so PP0 and a follower both park in place.  Where the schedule DOES name it, the schedule is executed (it wins).

WHY THE VOTE WAY CARRIES AND ``Weg2ParkReq`` STAYS UNWIRED.  ``Weg2ParkReq`` (``weg2/park.py``) would name a request to
tear down on every rank at the head of the same step.  What the lane needs uniform is the DECISION (park / do not run the
next chunk), and that is a function of (floor_m, request lane) -- both uniform per pass through the stamp.  The moment the
rows go back is rank-local (``inflight_middle_chunks`` drains per rank) and that skew is already the shape of the pressure
park; it is safe because a follower that still holds the continuation executes what the schedule names
(``_add_scheduled_req``: a resident continuation does not make a named request refuse, ``note_second_continuation_refused``).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterable, Optional, Tuple

from sglang.srt.weg2 import lanes

logger = logging.getLogger(__name__)

#: scheduler attribute holding the :class:`LaneP` of this rank; absent until an RPC or a stamp arrived
STATE_ATTR = "_weg2_lane_p"
#: scheduler attribute: the stamp PP0 puts on list m this pass (read by ``_pp_forward_and_process_input_requests``)
STAMP_ATTR = "_weg2_lane_stamp"
#: Req attributes
LANE_PARKED_ATTR = "weg2_lane_parked"  # the in-place / rows-back park of this request was triggered by the lane floor
#: the park trigger named in the ``WEG2-PARK (lane)`` marker
TRIGGER = "lane"

MARK_PARK_LANE = "WEG2-PARK (lane)"  # sch:10151 pattern: ``WEG2-PARK n=...`` with the trigger in the marker


# ---------------------------------------------------------------------------
# pure: the lane of a request and the hold rule
# ---------------------------------------------------------------------------

def req_lane(req: Any) -> int:
    """The lane of a scheduler ``Req``: ``req.priority`` (None = 0, negative = 0).  The front already normalised the field
    (L1); a request that reached the scheduler without it is lane 0."""
    p = getattr(req, "priority", None)
    if p is None or isinstance(p, bool):
        return 0
    try:
        n = int(p)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def holds(lane: int, floor: int) -> bool:
    """A request of ``lane`` is held (not run, not given a chunk) while the floor stands above it.  Floor 0 holds nothing."""
    return int(floor) > 0 and int(lane) < int(floor)


def highest_waiting_lane(waiting: Iterable[Any]) -> int:
    """The highest lane among the waiting requests (-1 for none)."""
    best = -1
    for r in waiting or ():
        n = req_lane(r)
        if n > best:
            best = n
    return best


def chunk_cap_tokens(cap: int, page_size: int) -> int:
    """The lane chunk size as a whole number of pages (at least one page); 0 = no cap."""
    cap = int(cap)
    if cap <= 0:
        return 0
    page = max(1, int(page_size or 1))
    return max(page, cap // page * page)


# ---------------------------------------------------------------------------
# the state of one rank
# ---------------------------------------------------------------------------

class LaneP:
    """``floor`` / ``epoch`` this rank APPLIES (what its adder reads), plus on PP0 the newest RPC not yet stamped.

    Epochs only move forward: a (floor, epoch) with an epoch not above the newest seen is stale and dropped, so an RPC
    that overtook an older one on the way cannot move the floor back."""

    def __init__(self, rank: int = 0) -> None:
        self.rank = int(rank)
        self.floor = 0
        self.epoch = 0
        self.rpc: Optional[Tuple[int, int]] = None  # PP0: (floor, epoch) taken from the RPC, not yet applied/stamped
        self.rpc_epoch = 0                          # PP0: newest epoch the RPC carried
        self.stale_n = 0
        self.changes = 0

    # ---- PP0 / single rank: the RPC ----
    def offer(self, floor: int, epoch: int) -> str:
        """The RPC body.  ``'taken'`` | ``'stale'`` (an epoch not above the newest seen)."""
        floor, epoch = max(0, int(floor)), int(epoch)
        if epoch <= max(self.rpc_epoch, self.epoch):
            self.stale_n += 1
            return "stale"
        self.rpc = (floor, epoch)
        self.rpc_epoch = epoch
        return "taken"

    def promote(self) -> bool:
        """Apply the pending RPC value now (PP0's pass top, or the single-rank handler).  True = the floor/epoch moved."""
        if self.rpc is None:
            return False
        floor, epoch = self.rpc
        self.rpc = None
        return self.apply(floor, epoch)

    # ---- every rank: the applied value ----
    def apply(self, floor: int, epoch: int) -> bool:
        floor, epoch = max(0, int(floor)), int(epoch)
        if epoch <= self.epoch:
            if epoch < self.epoch:
                self.stale_n += 1
            return False
        prev = self.floor
        self.floor, self.epoch = floor, epoch
        self.changes += 1
        logger.warning(
            "%s floor=%d epoch=%d rank=%d prev_floor=%d: this stage applies the lane floor in this pass "
            "(PP0 stamped it on list m, every stage reads the same value in the same pass; a lane below the "
            "floor gets no chunk, a running prefill of it parks at the chunk boundary)",
            lanes.MARK_PR_FLOOR, floor, epoch, self.rank, prev,
        )
        try:  # the progress beacon's lane trailer: a watchdog reads a hold, not a stall (L1)
            from sglang.srt.weg2 import progress_beacon

            progress_beacon.beat_lane(floor, epoch)
        except Exception:  # noqa: BLE001 - a beacon never costs a pass
            pass
        return True

    def stamp(self):
        """PP0: the standing value for list m (None until the first epoch)."""
        if self.epoch <= 0:
            return None
        from sglang.srt.weg2.pp_room_vote import Weg2PpLaneFloor

        return Weg2PpLaneFloor(floor=int(self.floor), epoch=int(self.epoch))


# ---------------------------------------------------------------------------
# scheduler glue (every function is a no-op without a state)
# ---------------------------------------------------------------------------

def _pp(sched: Any) -> Tuple[int, int]:
    ps = getattr(sched, "ps", None)
    return int(getattr(ps, "pp_size", 1) or 1), int(getattr(ps, "pp_rank", 0) or 0)


def state_of(sched: Any, create: bool = False) -> Optional[LaneP]:
    st = getattr(sched, STATE_ATTR, None)
    if st is None and create:
        st = LaneP(rank=_pp(sched)[1])
        setattr(sched, STATE_ATTR, st)
    return st


def active() -> bool:
    """The P-side lane rule runs: the switch is on AND this is group P with the park on (the lane park IS the park)."""
    if not lanes.enabled():
        return False
    from sglang.srt.weg2 import park

    return bool(park.park_active())


def on_rpc(sched: Any, recv_req: Any) -> None:
    """``POST /weg2/lane_floor`` reached this scheduler (``Weg2LaneFloorReqInput``): PP0 records it for its next pass; a
    non-PP boot applies it; PP1/PP2 ignore it (they get the value stamped).  Group D is part L2's."""
    if not lanes.enabled():
        return
    from sglang.srt.weg2 import park

    if not park.group_is_p():
        return  # D: part L2 (its admission reads the floor)
    if not park.park_active():
        if not getattr(sched, "_weg2_lane_p_inert_logged", False):
            sched._weg2_lane_p_inert_logged = True
            logger.warning(
                "WEG2 LANE-P INERT: SGLANG_WEG2_LANES=1 but SGLANG_WEG2_PARK is off on group P -- a running prefill of "
                "a lower lane cannot give its rows back, so the P side takes no lane floor"
            )
        return
    pp_size, pp_rank = _pp(sched)
    if pp_size > 1 and pp_rank != 0:
        return
    st = state_of(sched, create=True)
    st.offer(getattr(recv_req, "floor", 0), getattr(recv_req, "epoch", 0))
    if pp_size <= 1:
        st.promote()  # one rank (or one TP group, every rank got the request in the same pass): apply now


def pp0_pass(sched: Any):
    """PP0, top of the pass: take the newest RPC value, apply it, and set the stamp for list m (None = nothing to stamp)."""
    st = getattr(sched, STATE_ATTR, None)
    if st is None:
        return None  # no RPC ever: no stamp, the list is byte-identical
    st.promote()
    stamp = st.stamp()
    setattr(sched, STAMP_ATTR, stamp)
    return stamp


def apply_follower(sched: Any, stamp: Any) -> bool:
    """A follower, after relaying list m: apply the stamp it took off it in THIS pass."""
    if stamp is None:
        return False
    st = state_of(sched, create=True)
    return st.apply(int(stamp.floor), int(stamp.epoch))


def echo(sched: Any) -> Tuple[int, int]:
    """(floor, epoch) this rank applies, for its fact; (-1, -1) = no lane state."""
    st = getattr(sched, STATE_ATTR, None)
    return (-1, -1) if st is None else (int(st.floor), int(st.epoch))


def floor_for_pass(sched: Any) -> int:
    """The floor this rank's adder reads in this pass (0 without a state)."""
    st = getattr(sched, STATE_ATTR, None)
    return 0 if st is None else int(st.floor)


def owns_admission(sched: Any) -> bool:
    """True on the rank whose admission decision is the group's: PP0 or a non-PP boot.  A follower executes the forwarded
    schedule and never skips a request by lane."""
    pp_size, pp_rank = _pp(sched)
    return pp_size <= 1 or pp_rank == 0


def configure_adder(sched: Any, adder: Any) -> None:
    """Hand the pass's floor (and the lane chunk) to the adder.  Nothing is written without a state: the adder keeps its
    class defaults (floor 0, no cap)."""
    st = getattr(sched, STATE_ATTR, None)
    if st is None:
        return
    adder.lane_floor = int(st.floor)
    try:
        cap = chunk_cap_tokens(lanes.preempt_chunk_tokens(), int(getattr(adder, "page_size", 1) or 1))
    except Exception:  # noqa: BLE001 - an unreadable env is no cap
        cap = 0
    if cap > 0 and owns_admission(sched):
        adder.lane_chunk_cap_tokens = cap
        adder.lane_higher_waiting = highest_waiting_lane(getattr(sched, "waiting_queue", None))


# ---------------------------------------------------------------------------
# the park at the head of the step (Scheduler.process_pending_weg2_park)
# ---------------------------------------------------------------------------

def park_is_lane(req: Any) -> bool:
    return getattr(req, LANE_PARKED_ATTR, False) is True


def park_still_due(sched: Any, req: Any) -> bool:
    """A lane-parked chunked request still has to give its rows back: its lane is still below the standing floor."""
    return holds(req_lane(req), floor_for_pass(sched))


def note_parked(sched: Any, req: Any, *, span: int, total: int) -> int:
    """The ``WEG2-PARK (lane) n=...`` marker (``sch:10151`` pattern).  Returns n."""
    n = getattr(sched, "_weg2_park_lane_n", 0) + 1
    sched._weg2_park_lane_n = n
    st = getattr(sched, STATE_ATTR, None)
    logger.info(
        "%s n=%d rid=%s span=%d of %d lane=%d floor=%d epoch=%d: rows given back to the tree (evictable, "
        "is_insert=True), request back at the head of the queue of its lane; resume by prefix match when the floor "
        "falls to its lane, admitted whole",
        MARK_PARK_LANE, n, getattr(req, "rid", "?"), int(span), int(total), req_lane(req),
        0 if st is None else st.floor, 0 if st is None else st.epoch,
    )
    return n


@contextmanager
def chunk_scope(adder: Any, req: Any):
    """While a higher lane waits, a lower lane's chunk is at most ``lane_chunk_cap_tokens``: the adder's chunk budget is
    narrowed for this one request and, afterwards, charged only what the request really took."""
    cap = int(getattr(adder, "lane_chunk_cap_tokens", 0) or 0)
    outer = getattr(adder, "rem_chunk_tokens", None)
    if (cap <= 0 or outer is None or getattr(adder, "scheduled_extents", None)
            or req_lane(req) >= int(getattr(adder, "lane_higher_waiting", -1))):
        yield
        return
    cap = min(cap, int(outer))
    adder.rem_chunk_tokens = cap
    try:
        yield
    finally:
        adder.rem_chunk_tokens = int(outer) - (cap - int(adder.rem_chunk_tokens))
