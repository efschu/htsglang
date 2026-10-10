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
top of its next pass (:func:`pp0_pass`), names the batch from whose plan on the floor stands (FIX 3: ``eff``, below) and stamps
it onto list m of the request chain (``pp_room_vote.Weg2PpLaneFloor``, the ``Weg2PpRoomCap`` convention: "PP0 decides pass m on the
SAME value its followers read in their pass m").  Every follower takes the stamp off list m after relaying it
(:func:`apply_follower`) and applies it in the plan of that same batch.  PP0 keeps stamping the standing value every pass once the first epoch arrived, so a stage never depends on a
single message.  The follower facts echo the applied (floor, epoch) back (``Weg2PpRoomFact.lane_floor/lane_epoch``), so
PP0 can read whether all stages stand on one epoch.  Switch off (``SGLANG_WEG2_LANES=0``) or no RPC ever: no state, no
stamp, byte-identical lists.

WHO DECIDES WHAT.  Membership of a pass is PP0's decision (``#791`` forwarded schedule: followers admit only what the
decision names).  So: the admission loop skips a held request only on the rank that owns its admission truth
(:func:`owns_admission`: PP0 or a non-PP boot); a follower never skips a request PP0 named.  The chunked continuation is the
one request a follower decides about when the schedule does not name it.  Under the default row authority every follower
holds an effective map for every pass (a map or ``{}``), so the ``#992`` gate of the scheduler would refuse the seat of a
continuation the decision does not name and never reach ``add_chunked_req`` -- on PP1/PP2 the lane park would never happen
and their device rows of the parked lane would stay locked while PP0 holds.  :func:`continuation_held` exempts exactly the
continuation the stamped floor holds from that refusal (it takes no seat), so ``add_chunked_req`` reads the SAME stamped
floor on every stage and PP0 and both followers park in place; ``process_pending_weg2_park`` then returns the rows on each.
Where the schedule DOES name the continuation, the schedule is executed (it wins).

FIX 3 (NF metal 211536, #1004 at 21:22:35, done/lanes-metall4-1011): TWO things were wrong, and the stamp's transport was not
one of them (PP1 applied the floor at 21:22:32 in the pass that planned its slot 0, BEFORE its plan -- P log line 22968 vs the
``#969 EXTENT n=21`` of 6-17 at 23005).  (1) On a boot form WITHOUT a row carrier (``pp_row_carrier_present`` False: group P, "#631
ROW AUTHORITY DISABLED: no pp_flip_counters side channel", P log line 3100) the followers do NOT execute PP0's schedule, they plan
for themselves; the lane skip of the admission loop was reserved for PP0 (``owns_admission``), so PP0 held the lane-0 request
weg2-6-17 (floor 1) while PP1 admitted it -> slot 0 on PP1 vs slot 1 on PP0.  The skip is now :func:`skips_by_lane`: every rank that
decides for itself skips, a rank that executes a schedule never does.  (2) The effective pass was implicit (the pass that
happens to absorb list m).  It is now explicit: PP0 stamps ``eff`` = its ``forward_ct`` + :data:`FLOOR_LEAD_BATCHES`, and EVERY
stage -- PP0 included -- applies the floor in the plan of the batch with index ``eff`` (``forward_ct`` is the one batch counter all
stages share: a batch is planned identically on every stage, a stamp is read before the plan of batch ``eff`` or it is LATE and
named, never silently applied one batch off).  Until then the in-flight chunks of the lower lane run on all stages identically;
a higher lane waits at most FLOOR_LEAD_BATCHES batches (one chunk) on each stage.

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

#: FIX 3: how many batches after PP0's own ``forward_ct`` at the stamp the floor becomes effective on EVERY stage (PP0 too) WHILE WORK IS
#: IN FLIGHT.  A follower with an occupied slot or a chunked continuation does not take the request chain in every pass
#: (``_pp_event_loop`` gate: "the ordinary 2 ms of pipeline skew while an occupied slot waits for its upstream's statement"), and it
#: plans batch N as soon as batch N-1 has run -- which needs PP0's frame of N-1, not the list of the stamping pass.  Only batch N+1 is
#: planned after the follower's own batch N, whose frame PP0 sends AFTER the list that carries the stamp: lead 1 is the smallest
#: lead that cannot be overtaken, and it costs a higher lane at most the one chunk planned before the floor stands.  With NOTHING in
#: flight (idle, or everything held: no batch will ever launch to advance ``forward_ct``) the lead is 0 -- otherwise the floor could
#: never fall while every request of the group is held.  Not an env (no catalog surface); named in the report.
FLOOR_LEAD_BATCHES = 1


class LaneP:
    """``floor`` / ``epoch`` this rank APPLIES (what its adder reads), plus on PP0 the newest RPC not yet stamped.

    Epochs only move forward: a (floor, epoch) with an epoch not above the newest seen is stale and dropped, so an RPC
    that overtook an older one on the way cannot move the floor back.

    FIX 3: a floor does not apply when it is taken (PP0) or absorbed (follower) but in the plan of batch ``eff`` -- the
    ``forward_ct`` value PP0 named in the stamp.  ``pending`` holds ``(epoch, floor, eff, stamped_ct)`` in epoch order until
    the stage's own ``forward_ct`` reaches ``eff`` (:meth:`resolve`).  ``fc is None`` (a stand-in without a batch counter, or the
    single-rank boot) means "now": the legacy same-pass behaviour."""

    def __init__(self, rank: int = 0) -> None:
        self.rank = int(rank)
        self.floor = 0
        self.epoch = 0
        self.rpc: Optional[Tuple[int, int]] = None  # PP0: (floor, epoch) taken from the RPC, not yet applied/stamped
        self.rpc_epoch = 0                          # PP0: newest epoch the RPC carried
        self.stale_n = 0
        self.changes = 0
        self.late_n = 0
        self.pending: list = []                     # (epoch, floor, eff, stamped_ct), epoch ascending
        self.standing: Optional[Tuple[int, int, int, int]] = None  # PP0: (floor, epoch, eff, stamped_ct) put on list m

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

    def promote(self, fc: Optional[int] = None, busy: bool = True) -> bool:
        """PP0's pass top / the single-rank handler: turn the pending RPC value into a stamp and queue it for this stage.

        ``fc`` = this stage's ``forward_ct`` (None = apply now).  The floor takes effect at ``fc + FLOOR_LEAD_BATCHES`` when work is
        in flight (``busy``), at ``fc`` itself when nothing is (:meth:`resolve`); True = the applied floor/epoch moved in this call."""
        if self.rpc is None:
            return False
        floor, epoch = self.rpc
        self.rpc = None
        if fc is None:
            self.standing = (floor, epoch, -1, -1)  # no batch counter: the stamp applies in the pass that absorbs it
            return self.apply(floor, epoch)
        eff = int(fc) + (FLOOR_LEAD_BATCHES if busy else 0)
        self.standing = (floor, epoch, eff, int(fc))
        self._enqueue(epoch, floor, eff, int(fc))
        return self.resolve(fc)

    # ---- every rank: the queue and the applied value ----
    def _enqueue(self, epoch: int, floor: int, eff: int, stamped_ct: int) -> bool:
        if epoch <= max(self.epoch, *(e for e, *_ in self.pending), 0):
            return False  # the standing stamp of an epoch already seen, or an older one
        self.pending.append((int(epoch), max(0, int(floor)), int(eff), int(stamped_ct)))
        self.pending.sort(key=lambda t: t[0])
        return True

    def offer_stamp(self, floor: int, epoch: int, eff: int, stamped_ct: int, fc: Optional[int]) -> str:
        """A follower, list m: queue the stamp.  ``'queued'`` | ``'seen'`` (an epoch already known) | ``'late'`` (this stage's
        ``forward_ct`` is already past ``eff``: it may have planned batch ``eff`` without the floor -- applied at once, counted and
        named, the ``#1004`` detector is the net)."""
        for i, (e, f, x, st) in enumerate(self.pending):
            if e == int(epoch):  # an epoch still waiting for its batch: PP0 may only LOWER it (:meth:`relax`)
                if 0 <= eff < x:
                    self.pending[i] = (e, f, int(eff), st)
                    return "relaxed"
                return "seen"
        if eff < 0 or fc is None:  # a stamp without an effective batch (legacy list) applies in this pass
            return "queued" if self._enqueue(epoch, floor, -1, stamped_ct) else "seen"
        if int(epoch) <= max(self.epoch, *(e for e, *_ in self.pending), 0):
            return "seen"
        if int(fc) > int(eff):
            self.late_n += 1
            logger.warning(
                "%s LATE floor=%d epoch=%d rank=%d eff_fwd_ct=%d at_fwd_ct=%d stamped_fwd_ct=%d late_n=%d: the stamp reached this stage "
                "after it launched batch %d; the stages may have planned that batch on different floors (#1004 shape) -- applied "
                "now, the slot check names a split", lanes.MARK_PR_FLOOR, int(floor), int(epoch), self.rank, int(eff), int(fc),
                int(stamped_ct), self.late_n, int(eff),
            )
            self._enqueue(epoch, floor, -1, stamped_ct)
            return "late"
        self._enqueue(epoch, floor, eff, stamped_ct)
        return "queued"

    def relax(self, fc: Optional[int]) -> bool:
        """PP0, NOTHING in flight any more: the stamp of an epoch that is still waiting for its batch names the CURRENT batch from
        now on (the standing stamp is re-sent with the lower ``eff``; followers take it, :meth:`offer_stamp` ``'relaxed'``).  Without
        it a floor stamped while the last in-flight batch was still draining (``eff = fc + lead``) could never stand once every
        request of the group is held: no batch launches, ``forward_ct`` stays.  Every stage has launched what PP0 launched when
        PP0's batches are all back (the last stage's output ring), so the lowered ``eff`` is read by all of them at their next plan."""
        if fc is None or self.standing is None:
            return False
        floor, epoch, eff, stamped = self.standing
        if eff <= int(fc) or not any(e == epoch for e, *_ in self.pending):
            return False
        self.standing = (floor, epoch, int(fc), stamped)
        self.pending = [(e, f, int(fc) if e == epoch else x, st) for e, f, x, st in self.pending]
        logger.info("%s RELAX floor=%d epoch=%d rank=%d eff_fwd_ct %d -> %d: nothing is in flight any more, the floor stands in the next plan",
                    lanes.MARK_PR_FLOOR, floor, epoch, self.rank, eff, int(fc))
        return True

    def resolve(self, fc: Optional[int]) -> bool:
        """Apply every queued floor whose effective batch this stage has reached (``fc >= eff``); True = the applied value moved.
        Called before every read of the floor (the adder, the admission loop, the park), so the floor a stage plans batch
        ``fc`` on is a function of ``fc`` alone."""
        moved = False
        while self.pending:
            epoch, floor, eff, stamped = self.pending[0]
            if eff >= 0 and fc is not None and int(fc) < eff:
                break
            self.pending.pop(0)
            moved = self.apply(floor, epoch, eff=eff, fc=fc, stamped=stamped) or moved
        return moved

    def apply(self, floor: int, epoch: int, *, eff: int = -1, fc: Optional[int] = None, stamped: int = -1) -> bool:
        floor, epoch = max(0, int(floor)), int(epoch)
        if epoch <= self.epoch:
            if epoch < self.epoch:
                self.stale_n += 1
            return False
        prev = self.floor
        self.floor, self.epoch = floor, epoch
        self.changes += 1
        logger.warning(
            "%s floor=%d epoch=%d rank=%d prev_floor=%d eff_fwd_ct=%s at_fwd_ct=%s stamped_fwd_ct=%s lead=%d: this stage applies the "
            "lane floor in the plan of batch eff_fwd_ct (PP0 stamped it on list m with that batch index; every stage, PP0 too, "
            "waits for it, so every stage plans the batch on the same floor; a lane below the floor gets no chunk, a running "
            "prefill of it parks at the chunk boundary)",
            lanes.MARK_PR_FLOOR, floor, epoch, self.rank, prev,
            "now" if eff < 0 else eff, "-" if fc is None else int(fc), "-" if stamped < 0 else stamped, FLOOR_LEAD_BATCHES,
        )
        try:  # the progress beacon's lane trailer: a watchdog reads a hold, not a stall (L1)
            from sglang.srt.weg2 import progress_beacon

            progress_beacon.beat_lane(floor, epoch)
        except Exception:  # noqa: BLE001 - a beacon never costs a pass
            pass
        return True

    def stamp(self):
        """PP0: the standing stamp for list m (None until the first epoch).  It carries the EFFECTIVE batch, not "now"."""
        if self.standing is None:
            return None
        from sglang.srt.weg2.pp_room_vote import Weg2PpLaneFloor

        floor, epoch, eff, stamped = self.standing
        return Weg2PpLaneFloor(floor=int(floor), epoch=int(epoch), eff=int(eff), stamped=int(stamped))


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


def forward_ct_of(sched: Any) -> Optional[int]:
    """The stage's batch counter (``Scheduler.forward_ct``: batches launched, boot-monotone, the same index on every stage for the
    same batch).  None for a stand-in without one -- the lane code then applies a floor in the pass that takes it (legacy)."""
    fc = getattr(sched, "forward_ct", None)
    return fc if isinstance(fc, int) and not isinstance(fc, bool) else None


def work_in_flight(sched: Any) -> bool:
    """PP0 has a chunked continuation or a launched batch in an occupied slot -- the two terms the follower's chain-receive gate
    reads too (``_event_loop_pp_body``: ``chunked_req`` / ``mbs``).  A stand-in without them is idle."""
    if getattr(sched, "chunked_req", None) is not None:
        return True
    return any(b is not None for b in (getattr(sched, "mbs", None) or ()))


def pp0_pass(sched: Any):
    """PP0, top of the pass: take the newest RPC value, name its effective batch, queue it for this stage and set the stamp for
    list m (None = nothing to stamp).  PP0 applies the floor itself only in the plan of that batch (:func:`floor_for_pass`)."""
    st = getattr(sched, STATE_ATTR, None)
    if st is None:
        return None  # no RPC ever: no stamp, the list is byte-identical
    fc = forward_ct_of(sched)
    busy = work_in_flight(sched)
    st.promote(fc, busy=busy)
    if not busy:
        st.relax(fc)
    st.resolve(fc)
    stamp = st.stamp()
    setattr(sched, STAMP_ATTR, stamp)
    return stamp


def apply_follower(sched: Any, stamp: Any) -> bool:
    """A follower, after relaying list m: queue the stamp it took off it.  The floor stands from the plan of the batch the stamp
    names (``eff``); True = the applied value moved in this call (a stamp whose batch this stage already holds)."""
    if stamp is None:
        return False
    st = state_of(sched, create=True)
    fc = forward_ct_of(sched)
    st.offer_stamp(int(stamp.floor), int(stamp.epoch), int(getattr(stamp, "eff", -1)), int(getattr(stamp, "stamped", -1)), fc)
    return st.resolve(fc)


def echo(sched: Any) -> Tuple[int, int]:
    """(floor, epoch) this rank applies, for its fact; (-1, -1) = no lane state."""
    st = getattr(sched, STATE_ATTR, None)
    return (-1, -1) if st is None else (int(st.floor), int(st.epoch))


def floor_for_pass(sched: Any) -> int:
    """The floor this rank's adder reads in THIS plan (0 without a state): the queued floors whose effective batch the stage has
    reached are applied first, so the value is a function of the batch index alone."""
    st = getattr(sched, STATE_ATTR, None)
    if st is None:
        return 0
    st.resolve(forward_ct_of(sched))
    return int(st.floor)


def owns_admission(sched: Any) -> bool:
    """True on the rank whose admission decision is the group's: PP0 or a non-PP boot.  A follower that EXECUTES a forwarded
    schedule never skips a request by lane (see :func:`skips_by_lane` for the rank-local form)."""
    pp_size, pp_rank = _pp(sched)
    return pp_size <= 1 or pp_rank == 0


def executes_schedule(sched: Any) -> bool:
    """A follower whose pass is driven by PP0's forwarded row (``_pp_admission_incoming_effective`` is a map for this pass: the
    row carrier exists and delivered).  False on PP0, a non-PP boot, and on a follower of a form WITHOUT a row carrier ("#631 ROW
    AUTHORITY DISABLED: no pp_flip_counters side channel" -- Weg 2 group P), where it plans for itself."""
    pp_size, pp_rank = _pp(sched)
    if pp_size <= 1 or pp_rank == 0:
        return False
    return getattr(sched, "_pp_admission_incoming_effective", None) is not None


def skips_by_lane(sched: Any) -> bool:
    """FIX 3: this rank's admission loop skips a request the floor holds.  Every rank that DECIDES membership itself does --
    PP0, a non-PP boot, and every follower of a form without a row carrier (their local plan must equal PP0's, or the batches
    split: ``#1004 SLOT DISAGREEMENT``, NF metal 211536: PP0 held weg2-6-17, PP1 admitted it).  A follower that executes the
    forwarded schedule never skips: the schedule names its members."""
    return not executes_schedule(sched)


def cap_applies(sched: Any) -> bool:
    """The lane chunk cap (``SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS``) narrows PP0's chunk and is carried to the followers by
    the forwarded schedule, so it exists only where a schedule exists (a non-PP boot, or PP0 of a form WITH a row carrier).  On a
    form without one a follower would plan the normal chunk against PP0's narrowed one: the cap is inert there (named once)."""
    pp_size, pp_rank = _pp(sched)
    if pp_size <= 1:
        return True
    if pp_rank != 0:
        return False
    try:
        from sglang.srt.managers.pp_admission_congruence import pp_row_carrier_present

        return bool(pp_row_carrier_present(sched))
    except Exception:  # noqa: BLE001 - unreadable: no cap (a split batch is worse than a normal chunk)
        return False


def continuation_held(sched: Any, req: Any) -> bool:
    """The chunked continuation ``req`` is held by the floor THIS rank applies in this pass (the stamped one on a follower).

    The ``#992`` gate in ``Scheduler._get_new_batch_prefill_raw`` skips ``add_chunked_req`` for a continuation the forwarded
    schedule does not name.  A held continuation is exempt from that skip: it takes no seat (no chunk, no budget, not in
    ``can_run_list``), and reaching ``add_chunked_req`` is what parks it in place on every stage, so that
    ``process_pending_weg2_park`` gives the rows of PP1/PP2 back as it does PP0's.  A continuation the schedule DOES name
    never depends on this (``add_chunked_req`` executes the schedule first).  False without lane state."""
    return holds(req_lane(req), floor_for_pass(sched))


def configure_adder(sched: Any, adder: Any) -> None:
    """Hand the pass's floor (and the lane chunk) to the adder.  Nothing is written without a state: the adder keeps its
    class defaults (floor 0, no cap)."""
    st = getattr(sched, STATE_ATTR, None)
    if st is None:
        return
    adder.lane_floor = floor_for_pass(sched)
    try:
        cap = chunk_cap_tokens(lanes.preempt_chunk_tokens(), int(getattr(adder, "page_size", 1) or 1))
    except Exception:  # noqa: BLE001 - an unreadable env is no cap
        cap = 0
    if cap > 0 and owns_admission(sched):
        if cap_applies(sched):
            adder.lane_chunk_cap_tokens = cap
            adder.lane_higher_waiting = highest_waiting_lane(getattr(sched, "waiting_queue", None))
        elif not getattr(sched, "_weg2_lane_cap_inert_logged", False):
            sched._weg2_lane_cap_inert_logged = True
            logger.warning(
                "WEG2 LANE-P CHUNK-CAP INERT: SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS=%d but this P form has no row carrier -- the "
                "followers plan for themselves and would not narrow the chunk PP0 narrowed (#1004); the lane chunk stays the "
                "normal chunk", cap,
            )


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
