"""PRIORITY LANES 1008, part L4 (front controller): the lane state machine of the front.

Plan: deskq/PLAN-PRIO-LANES-1008.md sections 1 and 2 (row L4).  ``weg2/lanes.py`` (L1) fixes the field, the
state record (``lane_floor`` / ``lane_epoch``) and every name; this module is the CONTROLLER that moves the
floor.  It is written against the RPC names and bodies of L1 and works with stubbed groups: L2 (D side) and L3
(P side) implement the endpoints, L5 wires them.

THE RULE (user 08.10.): a higher lane displaces every lower lane at the shortest safe border; inside a lane
everything stays as it is today.  The floor is therefore ONE number derived from what is open:

    floor = the highest lane of any request the front still holds an open handler for (0 when none)

Every arrival and every end reconciles the floor to that number (:func:`reconcile`).  Going UP is the LANE-
PREEMPT, going DOWN the LANE-EMPTY / LANE-RESUME of the plan; both are the same function, so a cascade
2 -> 1 -> 0 is two calls of it and "a request of the lane left" has exactly one definition (its handler ended:
answered, failed, client gone).

WHERE A REQUEST BELOW THE FLOOR WAITS (never anywhere the rest of the controller looks at)
  * not yet routed (it arrived below the floor): at the GATE in ``handle_generate``, before pricing, a seat or
    a queue place -- a SHORT request must not reach a D seat past a running higher lane;
  * waiting in ``Front.queue`` / ``_ready_for_d``: taken OUT of both deques into :attr:`LaneCtl.held` and put back
    in the order of its ORIGINAL arrival.  The flip policy, ARRIVAL-SEAT, X-route / X-SOLO, MIN-DWELL, the drain
    and ``_flip_ledger`` then see only the requests of the floor lane, with no filter at their 97 + 99 sites;
  * running on D: parked by ``POST /weg2/park_running`` for exactly those rids with ``hold="lane"``
    (:func:`phase_policy.lane_park_body`); in the front's book ``_d_parked`` with a stamp that never lapses
    (:data:`phase_policy.LANE_PARK_STAMP`: no 30-s PARK-LAPSED, ``_flip_ledger`` excludes it);
  * in flight on P: the floor RPC (L3 parks it at the chunk border); the front counts such a leg as held (not
    against the pool plan, not for the P->D drain, not as a stall).  When only held legs are left in the drain
    pool they are TAKEN OFF P (:func:`p_take`: ``/abort_request`` by rid, KV of the finished chunks stays in P's
    tree) and kept in ``held`` like any waiting Pending: a leg parked in P's waiting queue would make the P->D
    witness read "rank not idle" (W3) and nothing would run it after the floor fell;
  * waiting for a D seat in ARRIVAL-SEAT (``_arrival_seat_wait``): the waiter takes itself out of
    ``st["waiters"]`` (:func:`waiter_hold`) and re-enters with its ORIGINAL order stamp and the held time as its
    clock offset.

CLOCKS (user decision 1): a held request keeps its ORIGINAL arrival (``Pending.t_arrive``: ordering, "oldest
first") and the time it was held is ``Pending.lane_held_s``; every wait clock (``d_wait_bound_s``, ARRIVAL-SEAT
age, fairness, the idle flip's oldest wait) reads :func:`clock_t` = ``t_arrive + lane_held_s``.  With the switch
off ``lane_held_s`` is always 0: ``clock_t(p) == p.t_arrive``.

Nothing here runs with ``SGLANG_WEG2_LANES`` off: the callers of this module test :func:`enabled` first, and
every Front hook returns before it looks at a lane.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

from sglang.srt.weg2 import lanes as _ln
from sglang.srt.weg2 import phase_policy as _pp

logger = logging.getLogger("weg2.front")

#: the lane loop's longest sleep (a kick ends it early)
TICK_S = 0.1
#: a floor RPC the group did not answer 200 is asked again after this long (the awake group sooner)
FLOOR_RETRY_S = 1.0
#: ... and a group that is asleep is not pestered more often than this
FLOOR_RETRY_ASLEEP_S = 5.0
#: a straggler park (a lower-lane rid on D the first park missed) is retried after this long
PARK_RETRY_S = 1.0
#: the arrival gate wakes this often to look for a client that left
GATE_TICK_S = 0.5
#: the P semaphore of the controller grows by this many slots with the switch on: legs held for a lane keep their
#: slot while a higher lane's leg needs one (the pool, not the semaphore, enforces the real concurrency)
SEM_EXTRA = 64
#: rids remembered for the REPREFILL count / the gate stamps
KEPT = 4096
#: LANES FIX 2: a floor-lane arrival that still has no place after this long no longer holds the P drain / the take
#: (liveness bound only: its own floor RPC times out after 10 s awake, ``arrive`` ends in ``forget`` -- a leaked entry
#: must not keep a P phase open for good)
ARRIVING_MAX_S = 30.0

#: the SSE comment a held stream gets (L4 -> client; an SSE comment line starts with ':' and is ignored by every
#: SSE reader, owui_proxy.py:215 skips every line that is not 'data:')
KEEPALIVE_FMT = ": lane-hold floor=%d\n\n"

#: LANES FIX 1 (metal jjbbrx probe 4): the floor went up (begin of the preempt, BEFORE the awaits of its RPCs) with the
#: time since ``WEG2-FLIP done`` that ended its deferral.  Not a prefix of ``MARK_PREEMPT`` (greps count that one).
MARK_RAISE = "WEG2 LANE-RAISE"
#: a held P leg taken off P before the P->D flip (L4 only, not an L1 name)
MARK_P_TAKE = "WEG2 LANE-P-TAKE"
#: a LEG1-EARLY leg still in flight on P cancelled and aborted by rid when its Pending is held (L4 FR2)
MARK_EARLY_TAKE = "WEG2 LANE-EARLY-TAKE"
#: the re-run of a leg that was taken off P: its cached tokens against what the first run left (L4 FR2)
MARK_P_RESUME = "WEG2 LANE-P-RESUME"
#: a held SSE stream was opened (200 + text/event-stream) so that it can get its keepalive (L4 FR2)
MARK_SSE_OPEN = "WEG2 LANE-SSE-OPEN"
#: LANES FIX 3 (metal 211536): the boot's lifecycle went ``dead`` (a rank died) while streams were held: every held stream gets
#: ONE named error event and is closed at once, instead of an empty 200 until the dead group's HTTP leg finally breaks (76 s there)
MARK_GROUP_DEAD = "WEG2 LANE-GROUP-DEAD"
#: how often the lane loop reads the boot lifecycle while it holds streams
DEATH_CHECK_S = 1.0


def enabled() -> bool:
    return _ln.enabled()


def clock_t(p: Any, default: Optional[float] = None) -> float:
    """The arrival the WAIT CLOCKS count from: the original arrival plus the time the request was held for a
    higher lane (user decision 1: held time counts against no bound).  ``t_arrive`` alone stays the ORDER.
    ``default``: the value of a partial test double without (or with a zero) ``t_arrive`` -- the reader's
    ``getattr(p, "t_arrive", default) or default`` of the call sites that tolerate one."""
    if default is None:
        t = float(p.t_arrive)
    else:
        t = float(getattr(p, "t_arrive", default) or default)
    return t + float(getattr(p, "lane_held_s", 0.0) or 0.0)


class Held:
    """A waiting Pending taken out of ``queue`` / ``_ready_for_d``."""

    __slots__ = ("p", "where", "t_held")

    def __init__(self, p: Any, where: str, t_held: float) -> None:
        self.p = p
        self.where = where
        self.t_held = t_held


class Gate:
    """A request that arrived below the floor and waits before it is routed."""

    __slots__ = ("rid", "lane", "t0", "evt")

    def __init__(self, rid: str, lane: int, t0: float) -> None:
        self.rid = rid
        self.lane = lane
        self.t0 = t0
        self.evt = asyncio.Event()


class LaneCtl:
    """The controller's registers (created on first use, so a front with the switch off never has one)."""

    def __init__(self) -> None:
        #: id(Pending) -> Held, in the order they were taken out
        self.held: "collections.OrderedDict[int, Held]" = collections.OrderedDict()
        #: rid -> the real time D confirmed the lane park (``_d_parked`` carries the never-lapsing stamp)
        self.parked_d: Dict[str, float] = {}
        #: rid -> real time the front BOOKED a lane park for a dormant (asleep) D request that D never confirmed;
        #: confirmed by a park RPC after the P->D wake (:func:`_park_d`), then it moves to ``parked_d``
        self.dormant: Dict[str, float] = {}
        #: rid -> (order stamp, time held): an ARRIVAL-SEAT waiter held below the floor
        self.waiter_held: Dict[str, Tuple[float, float]] = {}
        self.gates: Dict[str, Gate] = {}
        #: rid -> (t_arrive of the first arrival, seconds held at the gate), read once at the Pending's creation
        self.gate_done: "collections.OrderedDict[str, Tuple[float, float]]" = collections.OrderedDict()
        #: rid -> open SSE stream {resp, boundary, t_last}
        self.streams: Dict[str, Dict[str, Any]] = {}
        #: group name -> (floor, epoch) the group acknowledged with 200
        self.acked: Dict[str, Tuple[int, int]] = {}
        self.floor_tried: Dict[str, float] = {}
        self.floor_warned: Dict[str, float] = {}
        self.unsupported: set = set()
        self.park_unsupported = False
        self.park_tried = 0.0
        #: every rid that was ever held / parked for a lane (a RESUME-VIA-P leg of one is a LANE-REPREFILL)
        self.ever_held: "collections.OrderedDict[str, bool]" = collections.OrderedDict()
        self.reprefill: set = set()
        self.defer_logged: Optional[Tuple[int, int]] = None
        #: LANES FIX 1: real time of the first DEFER of the standing deferral / of the last ``WEG2-FLIP done``
        self.defer_t0: Optional[float] = None
        self.flip_done_t: Optional[float] = None
        #: LANES FIX 1 (probe 2b): rid -> real time of its arrival while it has no place yet (between ``arrive`` and
        #: the creation of its Pending, ``stamp``): a request of the floor lane that is on its way to a queue.
        self.arriving: Dict[str, float] = {}
        #: rid -> the stream request's early-open record (L4 FR2): {request, resp, lock, taken, held_t0}
        self.pre: Dict[str, Dict[str, Any]] = {}
        #: the abort tasks of cancelled early legs (a fresh leg 1 of the Pending waits for its own)
        self.early_tasks: set = set()
        #: LANES FIX 3: real time of the last lifecycle read, and whether the held streams were answered with the death already
        self.death_checked_t = 0.0
        self.death_announced = False
        self.lock: Optional[asyncio.Lock] = None
        self.kick_evt: Optional[asyncio.Event] = None

    # ---- helpers ----
    def mark_held(self, rid: str) -> None:
        self.ever_held.pop(rid, None)
        self.ever_held[rid] = True
        while len(self.ever_held) > KEPT:
            self.ever_held.popitem(last=False)

    def held_rids(self) -> List[str]:
        return [h.p.rid for h in self.held.values()]

    def forget(self, rid: str) -> None:
        """The rid's handler ended: it leaves every register."""
        r = str(rid)
        self.parked_d.pop(r, None)
        self.dormant.pop(r, None)
        self.waiter_held.pop(r, None)
        self.streams.pop(r, None)
        self.pre.pop(r, None)
        self.gates.pop(r, None)
        self.gate_done.pop(r, None)
        self.arriving.pop(r, None)
        self.reprefill.discard(r)
        for k in [k for k, h in self.held.items() if h.p.rid == r]:
            self.held.pop(k, None)


def ctl(fr: Any) -> LaneCtl:
    lc = fr.__dict__.get("_lane_ctl_obj")
    if lc is None:
        lc = fr.__dict__["_lane_ctl_obj"] = LaneCtl()
    return lc


def _lock(lc: LaneCtl) -> asyncio.Lock:
    if lc.lock is None:
        lc.lock = asyncio.Lock()
    return lc.lock


def kick(fr: Any) -> None:
    """Wake the lane loop (an arrival, an end, a flip that closed)."""
    lc = fr.__dict__.get("_lane_ctl_obj")
    if lc is not None and lc.kick_evt is not None:
        lc.kick_evt.set()


def flip_done(fr: Any) -> None:
    """LANES FIX 1: ``WEG2-FLIP done`` is an EVENT of the controller: the deferral of a higher lane that arrived
    during the flip (plan decision 2) ends now, not at the next tick.  Records the time (the latency marker reads it)
    and wakes the lane loop.  No state without the lane controller: a front that never saw a lane has no register."""
    if not enabled():
        return
    lc = ctl(fr)
    lc.flip_done_t = time.time()
    kick(fr)


def flip_begin_note(fr: Any, g: Any) -> str:
    """LANES FIX 2: the tail of ``WEG2-FLIP begin`` with the switch on -- how many of ``outstanding`` the front books as
    lane-held (they are NOT in the flip ledger the drain and the W3 witness read: :meth:`Front._flip_ledger`) and how
    many the flip ledger keeps.  Empty with no lane booking, so a boot without lane traffic prints the old line."""
    try:
        booked = fr._lane_parked_set()
        if not booked:
            return ""
        n_held = sum(1 for r in g.outstanding if r in booked)
        return " lane_held=%d ledger=%d" % (n_held, len(fr._flip_ledger(g)))
    except Exception:  # noqa: BLE001 -- an instrument, never the flip
        return ""


def lane_of_pending(p: Any) -> int:
    return int(getattr(p, "lane", 0) or 0)


def p_held(fr: Any, lane: int) -> bool:
    """A leg of this lane is held: its lane is below the floor."""
    ls = fr._lane_state()
    return ls.lane_floor > 0 and int(lane) < ls.lane_floor


def target_floor(fr: Any) -> int:
    """The floor the open requests ask for: the highest lane among them (0 when none)."""
    return max(fr._lane_state().rid_lane.values(), default=0)


# ---------------------------------------------------------------------------
# the transition
# ---------------------------------------------------------------------------

async def reconcile(fr: Any, cause: str = "tick", rid: Optional[str] = None) -> None:
    """Bring ``lane_floor`` to :func:`target_floor`.  Cheap when nothing moves (no lock, no await)."""
    if not enabled():
        return
    ls = fr._lane_state()
    lc = ctl(fr)
    target = target_floor(fr)
    if target == ls.lane_floor:
        lc.defer_logged = lc.defer_t0 = None  # LANES FIX 1: nothing left to defer (its lane ended during the flip)
        sweep(fr)
        return
    async with _lock(lc):
        ls = fr._lane_state()
        target = target_floor(fr)
        floor = ls.lane_floor
        if target == floor:
            lc.defer_logged = lc.defer_t0 = None
            return
        if fr.state != "serving":
            # user decision 2: a flip runs to its end, THEN the lane displaces ("kein Flip-Abbruch").  The arrival
            # goes its normal way (it queues: nothing is admitted during a flip); the first pass after
            # ``WEG2-FLIP done`` finds floor < target and preempts.
            if target > floor:
                now = time.time()
                if lc.defer_logged != (target, getattr(fr, "epoch", 0)):
                    lc.defer_logged = (target, getattr(fr, "epoch", 0))
                    lc.defer_t0 = now
                    fr.counters["lane_defer"] += 1
                    logger.warning(
                        "%s lane=%d floor=%d epoch=%d state=%s rid=%s -- a higher lane arrived while the front is "
                        "not serving (a flip runs to its end, plan decision 2): the floor rises after "
                        "WEG2-FLIP done", _ln.MARK_DEFER, target, floor, ls.lane_epoch, fr.state, rid or "-")
                # LANES FIX 1 (metal jjbbrx probe 4): the flip is not aborted, but a LOWER lane's speculative leg 1
                # (the D->P flip's LEG1-EARLY, posted before the lane arrived and waiting on P for its wake) must
                # not START its first chunk the moment P wakes: a chunk in flight cannot be taken back before its
                # border (probe 4: the 16 000-token leg started 1 ms before the abort, 6 s of P pipeline fill
                # stood in front of the lane-1 leg, TTFT 11.7 s).  It is cancelled and aborted by rid now, the
                # Pending stays in the queue and is held by the first pass after the flip.
                _cancel_early_lower(fr, lc, target, now)
            return
        deferred = lc.defer_logged is not None
        since_done_ms = None
        if deferred and lc.flip_done_t is not None and lc.defer_t0 is not None and lc.flip_done_t >= lc.defer_t0:
            since_done_ms = max(0.0, (time.time() - lc.flip_done_t) * 1000.0)
        lc.defer_logged = None
        lc.defer_t0 = None
        if target > floor:
            await _preempt(fr, lc, ls, floor, target, cause, rid, deferred=deferred, since_done_ms=since_done_ms)
        else:
            await _resume(fr, lc, ls, floor, target, cause)


def _hold_waiting(fr: Any, lc: LaneCtl, floor: int, now: float) -> int:
    """Take every waiting Pending below ``floor`` out of ``queue`` / ``_ready_for_d`` (same deque objects: the P
    drain pool holds a reference to ``queue``)."""
    n = 0
    for dq, where in ((fr.queue, "queue"), (fr._ready_for_d, "ready")):
        take: List[Any] = []
        keep: List[Any] = []
        for p in list(dq):
            fut = getattr(p, "fut", None)
            if lane_of_pending(p) < floor and (fut is None or not fut.done()):
                take.append(p)
            else:
                keep.append(p)
        if not take:
            continue
        dq.clear()
        dq.extend(keep)
        for p in take:
            lc.held[id(p)] = Held(p, where, now)
            p.lane_state = "held"
            lc.mark_held(p.rid)
            n += 1
            if where == "queue":
                _cancel_early(fr, lc, p, now)
        if where == "ready":
            fr._sync_batch_gate()
    if n:
        fr._park_stuck().hold(lc.held_rids())
    return n


def _cancel_early_lower(fr: Any, lc: LaneCtl, target: int, now: float) -> int:
    """LANES FIX 1: during a deferral (a flip runs) cancel the early leg 1 of every waiting Pending BELOW ``target``
    (:func:`_cancel_early`).  Idempotent (a cancelled leg is gone from the Pending).  Returns how many were cancelled."""
    n = 0
    for p in list(fr.queue):
        fut = getattr(p, "fut", None)
        if lane_of_pending(p) < target and (fut is None or not fut.done()):
            if _cancel_early(fr, lc, p, now, when="defer", target=target):
                n += 1
    return n


def _cancel_early(fr: Any, lc: LaneCtl, p: Any, now: float, when: str = "hold", target: int = -1) -> bool:
    """A Pending taken out of ``queue`` may carry a LEG1-EARLY leg still in flight on P (DP-NACHLAUF posts the
    queue head's leg 1 at the D->P flip's begin; with a lane deferred over that flip, the first pass after
    ``WEG2-FLIP done`` holds the Pending under it).  Such a leg is in no drain pool, so :func:`p_take` never sees
    it: left alone it would be parked by L3 in P's waiting queue and the P->D witness would read "rank not idle"
    (W3).  So it is treated like a taken leg: the task is cancelled, the rid is aborted on P by name (the KV of the
    finished chunks stays in P's tree), the Pending keeps its original arrival and runs a fresh leg 1 after the
    resume (its re-run is checked for the prefix, :func:`retake_check`).  A finished early leg (task done) stays on
    the Pending: the drain consumes its verdict at the resume.  Returns True when a running leg was cancelled."""
    early = getattr(p, "_leg1_early", None)
    if early is None or early.done():
        return False
    P = fr.groups["P"]
    t_on = P.outstanding.get(p.rid)
    cancel = getattr(fr, "_leg1_early_cancel", None)
    if cancel is None:
        return False
    on_p = bool(cancel(p, "lane-hold"))
    p.lane_retake = True
    # when=defer: the leg waited on a P that was still asleep / waking while the flip ran -- nothing ran, so nothing can be
    # LOST (``retake_check``'s rule "stood on P and found nothing cached" would count a REPREFILL for a leg that was
    # never admitted: the instrument that must read 0 must not be fed by the fix)
    p.lane_p_ran_s = 0.0 if when == "defer" else (max(0.0, now - t_on) if t_on else 0.0)
    fr.counters["lane_early_taken"] += 1
    logger.warning("%s rid=%s lane=%d floor=%d on_p=%s ran_s=%.1f when=%s target=%d -- the early leg 1 of a held "
                   "request was still in flight on P: cancelled and aborted by rid (a leg parked in P's waiting "
                   "queue is not idle for the P->D witness); it runs a fresh leg 1 after the resume, with the "
                   "finished chunks as prefix (when=defer: cancelled while the flip still runs, before P's wake "
                   "starts its first chunk)",
                   MARK_EARLY_TAKE, p.rid, lane_of_pending(p), fr._lane_state().lane_floor, on_p, p.lane_p_ran_s,
                   when, int(target))
    if on_p:
        try:
            t = asyncio.ensure_future(fr.rpc(P, "/abort_request", {"rid": p.rid}, 30))
        except RuntimeError:  # no running loop (a sync caller): the witness / the next abort decides
            return True
        p._leg1_abort = t
        lc.early_tasks.add(t)

        def _done(task: Any, p: Any = p) -> None:
            lc.early_tasks.discard(task)
            if getattr(p, "_leg1_abort", None) is task:
                p._leg1_abort = None
            if not task.cancelled() and task.exception() is not None:
                fr.counters["lane_early_abort_failed"] += 1

        t.add_done_callback(_done)
    return True


def sweep(fr: Any) -> int:
    """The invariant, kept every tick and at every reconcile: no request below the floor waits in the deques the
    controller reads (a RESUME-VIA-P leg, a W31 requeue, an SK re-add may put one back).  Returns the number moved."""
    if not enabled():
        return 0
    ls = fr._lane_state()
    if ls.lane_floor <= 0:
        return 0
    lc = ctl(fr)
    n = _hold_waiting(fr, lc, ls.lane_floor, time.time())
    if n:
        fr.counters["lane_held"] += n
    return n


async def _preempt(fr: Any, lc: LaneCtl, ls: Any, prev: int, floor: int, cause: str, rid: Optional[str],
                   deferred: bool = False, since_done_ms: Optional[float] = None) -> None:
    t0 = time.time()
    ls.set_floor(floor)
    # LANES FIX 1: the LANE-PREEMPT line below is written AFTER the RPCs (its ms= is the whole preempt, probe 4: 5986 ms
    # of which the front waited for P's reply, P answering only after its in-flight chunk); this line is the moment
    # the floor rose, with the latency from the end of the flip that deferred it.
    logger.warning("%s floor=%d epoch=%d prev_floor=%d cause=%s rid=%s deferred=%d since_flip_done_ms=%s -- the floor "
                   "rose (the RPCs of this preempt follow; LANE-PREEMPT is written when they have answered)",
                   MARK_RAISE, floor, ls.lane_epoch, prev, cause, rid or "-", 1 if deferred else 0,
                   "-" if since_done_ms is None else "%.0f" % since_done_ms)
    held = _hold_waiting(fr, lc, floor, t0)
    dorm0 = fr.counters["lane_parked_d_dormant"]
    P, D = fr.groups["P"], fr.groups["D"]
    parked_p = [r for r in list(P.outstanding) if ls.lane_for(r) < floor]
    res = await asyncio.gather(_park_d(fr, lc, ls, floor), sync_floor(fr, force=True),
                               *list(lc.early_tasks), return_exceptions=True)
    parked_d = res[0] if isinstance(res[0], int) else 0
    for r in res:
        if isinstance(r, BaseException):
            logger.warning("WEG2 LANE-PREEMPT step failed: %r", r)
    fr.counters["lane_preempt"] += 1
    fr.counters["lane_held"] += held
    logger.warning(
        "%s floor=%d epoch=%d parked_d=%d parked_p=%d held=%d dormant=%d prev_floor=%d cause=%s rid=%s ms=%.0f "
        "since_flip_done_ms=%s -- "
        "a higher lane displaces every lower one: D's running requests of lower lanes are parked (hold=lane), "
        "P's legs stop at the chunk border (floor RPC), waiting ones are held out of queue/_ready_for_d (original "
        "arrival kept); dormant = booked for a sleeping D, confirmed by a park RPC after its wake; ms = the whole "
        "preempt (begin: LANE-RAISE), since_flip_done_ms = end of the deferring flip -> begin of the preempt; "
        "parked_p = the P legs below the floor at the raise that were ASKED to stop at their next chunk border, NOT a "
        "confirmation (a leg whose last chunk is already in P's pipeline finishes instead; the park of a leg is the "
        "P-side line 'WEG2-PARK (lane)')",
        _ln.MARK_PREEMPT, floor, ls.lane_epoch, parked_d, len(parked_p), held,
        fr.counters["lane_parked_d_dormant"] - dorm0, prev, cause, rid or "-", (time.time() - t0) * 1000.0,
        "-" if since_done_ms is None else "%.0f" % since_done_ms)
    fr._kick_controller("arrival")
    _changed(fr)


def _changed(fr: Any) -> None:
    try:
        fr._rb_changed()
    except Exception:  # noqa: BLE001 -- an instrument, never the route
        pass


def _release_seats(fr: Any, rids: Iterable[str]) -> int:
    """A lane-parked request gives its FRONT seat back (the ARRIVAL-SEAT youngest-park's convention, front.py
    ``_arrival_seat_park``): otherwise the higher lane waits for a seat the displaced request holds and the lane
    never runs (NF's D is bs1). When D runs the parked request again it runs without a front seat, exactly like
    an ARRIVAL-SEAT victim: the rule counts D's ledger, not the semaphore."""
    want = set(rids)
    n = 0
    for s in list(getattr(fr, "_d_seats_live", ()) or ()):
        if s.rid in want:
            s.release("lane-park")
            n += 1
    return n


async def _park_d(fr: Any, lc: LaneCtl, ls: Any, floor: int, retry: bool = False) -> int:
    """Park D's open requests below ``floor`` for the lane.  Returns how many D confirmed (0 with D asleep:
    the dormant bookings are unconfirmed until the first park RPC after the wake)."""
    if lc.park_unsupported:
        return 0
    D = fr.groups["D"]
    rids = [r for r in list(D.outstanding) if ls.lane_for(r) < floor and r not in lc.parked_d]
    if not rids:
        return 0
    if fr.awake != "D":
        # D sleeps (P phase): what it still holds is parked / held dormant already -- the park RPC has nothing to
        # take off a batch and a sleeping group cannot be asked for it.  The front BOOKS the hold (``dormant``:
        # out of the flip ledger, no 30-s lapse, no seat) but it is UNCONFIRMED: D never acknowledged it, and the
        # wake message cannot carry it (``ResumeMemoryOccupationReqInput`` ignores unknown keys).  After the P->D
        # wake the lane loop finds these rids in D's ledger and not in ``parked_d`` and sends the park RPC by
        # rid (this function, awake branch); only D's answer moves them to ``parked_d``.
        fresh = [r for r in rids if r not in lc.dormant]
        t_park = time.time()
        for r in fresh:
            lc.dormant[r] = t_park
            fr._d_parked[r] = _pp.LANE_PARK_STAMP
            lc.mark_held(r)
            fr._req_book().park(r, t_park, f"lane:{floor}", fr.epoch)
        _release_seats(fr, fresh)
        fr._park_stuck().hold(fresh)  # after the seat release: Seat.release ends the rid's park streak
        fr.counters["lane_parked_d_dormant"] += len(fresh)
        return 0
    body = _pp.lane_park_body(fr.epoch, rids, floor, ls.lane_epoch)
    t_park = time.time()
    try:
        code, text = await fr.rpc(D, _pp.PARK_PATH, body, _pp.PARK_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        code, text = 0, f"{type(e).__name__}: {e}"
    verdict, got, why = _pp.park_verdict(code, text)
    if verdict == _pp.PARK_UNSUPPORTED:
        lc.park_unsupported = True
        fr.counters["lane_park_unsupported"] += 1
        logger.error(
            "WEG2 LANE-PARK-UNSUPPORTED status=%d path=%s -- this D has no park endpoint: the lower lanes on D "
            "run on beside the higher one (nothing can displace them; L2 / the old D is the cause, not the lane)",
            code, _pp.PARK_PATH)
        return 0
    if verdict == _pp.PARK_FAILED:
        fr.counters["lane_park_failed"] += 1
        logger.warning("WEG2 LANE-PARK-FAILED status=%d (%s) floor=%d rids=%s -- retried every %.0f s", code, why,
                       floor, rids[:8], PARK_RETRY_S)
        return 0
    n = 0
    over: List[str] = []
    for r in got:
        if r not in D.outstanding:
            continue
        if ls.lane_for(r) < floor:
            lc.dormant.pop(r, None)  # a dormant booking D confirms now (counted as booked when it was made)
            n += 1
            lc.parked_d[r] = t_park
            fr._d_parked[r] = _pp.LANE_PARK_STAMP
            lc.mark_held(r)
            fr._req_book().park(r, t_park, f"lane:{floor}", fr.epoch)
        else:
            # a D that ignores the ``rids`` filter parked a request of the floor lane too: it is a plain park
            # for the front (it resumes with the next wake), and the floor lane is not displaced on purpose.
            fr._d_parked[r] = t_park
            over.append(r)
    mine = [r for r in got if r in lc.parked_d]
    if mine:
        _release_seats(fr, mine)
        fr._park_stuck().hold(mine)  # after the seat release: Seat.release ends the rid's park streak
    if n:
        fr.counters["lane_parked_d"] += n
    if over:
        fr.counters["lane_park_overreach"] += len(over)
        logger.warning("WEG2 LANE-PARK-OVERREACH floor=%d rids=%s -- D parked requests of the floor lane too (its "
                       "park ignores the rids filter): parked like a flip park", floor, over[:8])
    return n


async def sync_floor(fr: Any, force: bool = False, now: Optional[float] = None) -> None:
    """``POST /weg2/lane_floor`` ``{"floor", "epoch"}`` to P and D until each acknowledges the current pair."""
    ls = fr._lane_state()
    lc = ctl(fr)
    want = (ls.lane_floor, ls.lane_epoch)
    now = time.time() if now is None else now
    todo = []
    for g in fr.groups.values():
        if g.name in lc.unsupported or lc.acked.get(g.name) == want:
            continue
        gap = FLOOR_RETRY_S if g.name == fr.awake else FLOOR_RETRY_ASLEEP_S
        if not force and now - lc.floor_tried.get(g.name, 0.0) < gap:
            continue
        lc.floor_tried[g.name] = now
        todo.append(g)
    if not todo:
        return
    body = {_ln.RPC_KEY_FLOOR: want[0], _ln.RPC_KEY_EPOCH: want[1]}

    async def one(g: Any) -> None:
        try:
            code, text = await fr.rpc(g, _ln.RPC_LANE_FLOOR, body, 10 if g.name == fr.awake else 2)
        except Exception as e:  # noqa: BLE001
            code, text = 0, f"{type(e).__name__}: {e}"
        if code == 200:
            if want == (ls.lane_floor, ls.lane_epoch):
                lc.acked[g.name] = want
            fr.counters["lane_floor_rpc"] += 1
            return
        if code in (404, 501):
            lc.unsupported.add(g.name)
            fr.counters["lane_floor_unsupported"] += 1
            logger.error("WEG2 LANE-FLOOR-UNSUPPORTED group=%s status=%d path=%s -- this group has no lane_floor "
                         "endpoint (L2 / L3 not deployed): the floor stays the front's own on it",
                         g.name, code, _ln.RPC_LANE_FLOOR)
            return
        fr.counters["lane_floor_failed"] += 1
        if now - lc.floor_warned.get(g.name, 0.0) >= 30.0:
            lc.floor_warned[g.name] = now
            logger.warning("WEG2 LANE-FLOOR-FAILED group=%s status=%d floor=%d epoch=%d (%s) -- asked again",
                           g.name, code, want[0], want[1], str(text)[:120])

    await asyncio.gather(*(one(g) for g in todo))


def _restore_held(fr: Any, lc: LaneCtl, floor: int, now: float) -> Tuple[int, int]:
    """Put every held Pending of lane >= ``floor`` back into ``queue`` / ``_ready_for_d`` in the order of its
    ORIGINAL arrival (``t_arrive``); the time it was held becomes its clock offset.  Returns (queue, ready)."""
    back: Dict[str, List[Any]] = {"queue": [], "ready": []}
    for k, h in list(lc.held.items()):
        p = h.p
        fut = getattr(p, "fut", None)
        if fut is not None and fut.done():
            lc.held.pop(k, None)  # it ended while held (client gone, stop): nothing to resume
            continue
        if lane_of_pending(p) >= floor:
            lc.held.pop(k, None)
            p.lane_held_s = float(getattr(p, "lane_held_s", 0.0) or 0.0) + max(0.0, now - h.t_held)
            p.lane_state = "resuming"
            if p.lane_p_taken:
                # a leg taken off P prefills again (the chunks it finished are P's prefix hit); a request that
                # waited in ``_ready_for_d`` with its leg 1 DONE keeps ``leg1_done``
                p.lane_p_taken = False
                p.leg1_done = False
                p.lane_retake = True  # its re-run leg 1 is checked for the prefix (LANE-REPREFILL)
            back[h.where].append(p)
    n_q = n_r = 0
    for dq, where in ((fr.queue, "queue"), (fr._ready_for_d, "ready")):
        if not back[where]:
            continue
        merged = sorted(list(dq) + back[where], key=lambda q: q.t_arrive)
        dq.clear()
        dq.extend(merged)
        if where == "ready":
            n_r = len(back[where])
            fr._sync_batch_gate()
        else:
            n_q = len(back[where])
    ids = [p.rid for p in back["queue"] + back["ready"]]
    if ids:
        fr._park_stuck().release(ids)
    return n_q, n_r


async def _resume(fr: Any, lc: LaneCtl, ls: Any, prev: int, floor: int, cause: str) -> None:
    """LANE-EMPTY: no open request of the old floor lane is left; the floor falls to the highest lane with work."""
    t0 = time.time()
    ls.set_floor(floor)
    # 1. held waiting Pendings of the new floor lane, back in the order of their ORIGINAL arrival
    n_q, n_r = _restore_held(fr, lc, floor, t0)
    # 2. D's lane parks of the new floor lane end (D requeues them in arrival order on the floor RPC below)
    rel = [r for r in list(lc.parked_d) + [r for r in lc.dormant if r not in lc.parked_d]
           if ls.lane_for(r) >= floor]
    for r in rel:
        lc.parked_d.pop(r, None)
        lc.dormant.pop(r, None)
        fr._d_parked.pop(r, None)
    if rel:
        fr._park_stuck().release(rel)
    if rel:
        fr._rb_resume(rel, "lane_resume")
        fr.counters["lane_resumed_d"] += len(rel)
    # 3. the floor to P and D (D resumes the parked, P's legs run on)
    await sync_floor(fr, force=True)
    # 4. new arrivals of the lane last (the gate releases after the parked)
    n_gate = _release_gates(fr, lc, floor)
    # ARRIVAL-SEAT waiters of the lane re-enter ``waiters`` on their next tick (:func:`waiter_hold`), by their
    # ORIGINAL order stamp -- before every newer waiter, like the parked
    n_wait = sum(1 for r in lc.waiter_held if ls.lane_for(r) >= floor)
    resumed = n_q + n_r + len(rel) + n_gate + n_wait
    fr.counters["lane_resume"] += 1
    logger.warning(
        "%s floor=%d->%d epoch=%d resumed=%d held_back=%d parked_d=%d gate=%d waiters=%d cause=%s ms=%.0f -- no "
        "request of lane %d is left: the floor falls to the highest lane with work; its parked requests run "
        "first, with their original arrival, the arrivals held at the gate after them", _ln.MARK_RESUME, prev,
        floor, ls.lane_epoch, resumed, n_q + n_r, len(rel), n_gate, n_wait, cause,
        (time.time() - t0) * 1000.0, prev)
    fr._kick_controller("arrival")
    _changed(fr)


def _release_gates(fr: Any, lc: LaneCtl, floor: int) -> int:
    n = 0
    for rid, g in sorted(lc.gates.items(), key=lambda kv: kv[1].t0):
        if g.lane >= floor and not g.evt.is_set():
            g.evt.set()
            n += 1
    return n


# ---------------------------------------------------------------------------
# the arrival gate (handle_generate)
# ---------------------------------------------------------------------------

async def arrive(fr: Any, request: Any, rid: str, client_gone: Callable[[Any], bool], web: Any) -> Optional[Any]:
    """The request's lane is noted (``Front._lane_note``).  Reconcile the floor (a higher lane preempts at once,
    during a flip it is deferred), then, when the request is BELOW the floor, hold it here.  Returns None to go
    on routing, or a response (the client left while the request was held)."""
    if not enabled():
        return None
    ls = fr._lane_state()
    lc = ctl(fr)
    lane = ls.lane_for(rid)
    floor0 = ls.lane_floor
    lc.arriving[rid] = time.time()  # LANES FIX 1: on its way to a queue until its Pending exists (``stamp``)
    await reconcile(fr, "arrive", rid)
    if lane > 0 or ls.lane_floor > 0:
        logger.info("WEG2 LANE-ARRIVE rid=%s lane=%d floor=%d->%d epoch=%d action=%s", rid, lane, floor0,
                    ls.lane_floor, ls.lane_epoch,
                    "hold" if lane < ls.lane_floor else "preempt" if lane > floor0 else "run")
    if lane >= ls.lane_floor:
        return None
    gate = Gate(rid, lane, time.time())
    lc.gates[rid] = gate
    lc.mark_held(rid)
    fr.counters["lane_gate_held"] += 1
    fr._park_stuck().hold([rid])
    try:
        while lane < ls.lane_floor:
            try:
                await asyncio.wait_for(gate.evt.wait(), GATE_TICK_S)
            except asyncio.TimeoutError:
                pass
            if gate.evt.is_set():
                if lane >= ls.lane_floor:
                    break
                gate.evt.clear()
            if client_gone(request):
                fr.counters["client_gone_lane_hold"] += 1
                logger.warning("WEG2-CLIENT-GONE rid=%s state=lane-hold action=not-routed wait_s=%.1f (H102, "
                               "PRIORITY LANES)", rid, time.time() - gate.t0)
                kick(fr)
                return web.json_response({"error": f"WEG2-CLIENT-GONE rid={rid} state=lane-hold"}, status=499)
    finally:
        lc.gates.pop(rid, None)
        fr._park_stuck().release([rid])
    now = time.time()
    lc.gate_done[rid] = (gate.t0, max(0.0, now - gate.t0))
    while len(lc.gate_done) > KEPT:
        lc.gate_done.popitem(last=False)
    return None


def stamp(fr: Any, p: Any) -> None:
    """A Pending created after the gate keeps the ORIGINAL arrival as its order and carries the hold as its
    clock offset (the arrival the clocks count from is the release)."""
    lc = fr.__dict__.get("_lane_ctl_obj")
    if lc is None:
        return
    lc.arriving.pop(p.rid, None)  # it has a place now: the drain sees it in the queue
    got = lc.gate_done.pop(p.rid, None)
    if got is not None:
        p.t_arrive = got[0]
        p.lane_held_s = float(getattr(p, "lane_held_s", 0.0) or 0.0) + got[1]


# ---------------------------------------------------------------------------
# P side hooks
# ---------------------------------------------------------------------------

def p_ledger(fr: Any, g: Any) -> List[str]:
    """The rids the P->D flip's drain waits for: P's ledger minus the legs held for a higher lane."""
    ls = fr._lane_state()
    if ls.lane_floor <= 0:
        return list(g.outstanding)
    return [r for r in g.outstanding if ls.lane_for(r) >= ls.lane_floor]


def lane_held_leg(fr: Any, p: Any) -> bool:
    """For the P drain pool and the leg-1 stall check: is this leg held for a higher lane?"""
    return enabled() and p_held(fr, lane_of_pending(p))


def take_wait(fr: Any) -> bool:
    """LANES FIX 1 (metal jjbbrx probe 2b): True = the P drain pool must NOT take its held legs off P yet, because a
    request of the floor lane (or above) has arrived and has no place yet -- it is between ``arrive`` (whose reconcile
    raised the floor, and awaits P's reply to it) and the creation of its Pending.  The pool sees only the held legs
    in that window and would take them off P and flip P->D although the floor lane's OWN leg is about to be queued
    for P: probe 2b took the 98 000-token leg, flipped P->D and, 5 s later, D->P again for the 16 000-token lane-1
    leg (TTFT 18.3 s), and the L3 chunk park (``WEG2-PARK (lane)``) could never fire.  With the leg queued the pool
    dispatches it beside the held leg, P applies the floor at its next chunk border and parks the lower lane there
    (the plan path); a lane-1 request that is NOT P-bound leaves P with only held legs and they are taken then.
    A rid that already stands in a group's outstanding set has its place.

    LANES FIX 2 (metal jjbbrx 18:03:35Z probe 2b, second run): the SAME predicate also keeps the P PHASE open
    (``_p_drain_pool``: nothing in flight, queue empty) while such an arrival has no place -- the drain ended 77 ms before
    the lane-1 request's Pending existed and P->D started (a flip pair for a request that needed P).  An arrival older
    than :data:`ARRIVING_MAX_S` is not waited for."""
    lc = fr.__dict__.get("_lane_ctl_obj")
    if not enabled() or lc is None or not lc.arriving:
        return False
    ls = fr._lane_state()
    placed = set(fr.groups["P"].outstanding) | set(fr.groups["D"].outstanding)
    now = time.time()
    for rid, t0 in list(lc.arriving.items()):
        if rid not in placed and ls.lane_for(rid) >= ls.lane_floor and now - t0 <= ARRIVING_MAX_S:
            return True
    return False


async def p_take(fr: Any, items: List[Any], cancel: Callable[[], Any]) -> int:
    """The P drain pool's ONLY legs left are held for a higher lane: take them OFF P before the P->D flip.

    Why: L3 parks such a leg at the chunk border into P's waiting queue, and ``is_fully_idle`` still asks an empty
    waiting queue -- the flip's witness reads "front drained, rank NOT idle" and W3 stops; and after the floor
    fell nothing would run the leg again (the D phase flips back to P only for a queue entry).  So: mark the
    Pendings (``lane_p_taken``: their ``one`` tasks end un-prefilled, no failure answer, no hand-off), cancel the
    tasks, ``/abort_request`` by rid on P (every PP rank drops it; the KV of the chunks already finished stays in
    P's radix tree, the later prefill is a prefix hit -- L3's park contract, proven by L5), and keep the
    Pendings in ``held`` with their original arrival.  Returns how many were taken."""
    lc = ctl(fr)
    ls = fr._lane_state()
    P = fr.groups["P"]
    now = time.time()
    for p in items:
        p.lane_p_taken = True
        t_on = P.outstanding.get(p.rid)
        p.lane_p_ran_s = max(0.0, now - t_on) if t_on else 0.0
    cancel()
    rids = [p.rid for p in items]
    res = await asyncio.gather(*(fr.rpc(P, "/abort_request", {"rid": r}, 30) for r in rids), return_exceptions=True)
    bad = []
    for r, x in zip(rids, res):
        if isinstance(x, BaseException) or (isinstance(x, tuple) and x[0] != 200):
            bad.append(r)
    if bad:
        fr.counters["lane_p_take_abort_failed"] += len(bad)
        logger.warning("WEG2 LANE-P-TAKE abort on P failed for rids=%s -- the leg may stay in P's queue (the "
                       "flip's witness decides)", bad[:8])
    n = 0
    for p in items:
        fut = getattr(p, "fut", None)
        if p.client_gone or (fut is not None and fut.done()):
            continue
        p.lane_state = "held"
        p.leg1_done = False
        lc.held[id(p)] = Held(p, "queue", now)
        lc.mark_held(p.rid)
        n += 1
    if rids:
        fr._park_stuck().hold(rids)
    fr.counters["lane_p_taken"] += n
    logger.warning("%s rid=%s floor=%d epoch=%d n=%d abort_failed=%d -- only legs held for a higher lane were left "
                   "in the P drain: taken off P by rid before the P->D flip (a leg parked in P's waiting queue "
                   "is not idle for the witness), kept with their original arrival for the lane resume",
                   MARK_P_TAKE, ",".join(rids[:8]), ls.lane_floor, ls.lane_epoch, n, len(bad))
    # the floor may have fallen while the aborts were on the wire: nothing else would resume them then
    _restore_held(fr, lc, ls.lane_floor, time.time())
    _changed(fr)
    return n


def retake_check(fr: Any, p: Any, pt: int, ct: int) -> bool:
    """The fresh leg 1 of a Pending that was taken off P (or whose early leg was cancelled) for a lane has answered:
    did the prefix the first run left come back?  ``ct`` = P's ``cached_tokens`` of THIS run, ``pt`` its prompt.

    The reference is what P reports finished at the take: ``p.lane_p_done_tokens``.  P reports no such number today
    (``/abort_request`` answers nothing and the floor RPC is one-way: lanes.py), so it is 0 = unknown, and then the
    only provable loss is the whole of it: a leg that STOOD ON P at the take (``lane_p_ran_s`` > 0) and whose re-run
    finds nothing cached (``ct`` <= 0) is counted, LANE-REPREFILL, plan decision 3.  That rule can count a leg that
    was admitted by nobody before the take (nothing to keep) -- the direction a "must be 0" instrument errs in; each
    re-run is logged with ``ran_s`` and ``cached`` (LANE-P-RESUME) so metal can read a count against L3's
    ``WEG2-PARK (lane) ... span=`` line of the same rid.  With a known reference every shortfall counts.
    Returns True when a prefix loss was counted."""
    p.lane_retake = False
    done = int(getattr(p, "lane_p_done_tokens", 0) or 0)
    ran = float(getattr(p, "lane_p_ran_s", 0.0) or 0.0)
    if done > 0:
        lost = int(ct) < done
    else:
        lost = int(ct) <= 0 and ran > 0.0
    fr.counters["lane_p_retake"] += 1
    logger.info("%s rid=%s cached=%d prompt=%d done_ref=%d ran_s=%.1f lost=%s -- the leg 1 of a request taken off P "
                "for a lane ran again", MARK_P_RESUME, p.rid, int(ct), int(pt), done, ran, lost)
    if lost:
        fr._lane_reprefill(p.rid, "p-retake: cached=%d prompt=%d done_ref=%d ran_s=%.1f" % (int(ct), int(pt), done, ran))
    return lost


def waiter_hold(fr: Any, st: Dict[str, Any], rid: str) -> bool:
    """ARRIVAL-SEAT waiter ``rid`` (its ``_arrival_seat_wait`` loop, once per tick): True = the lane holds it.

    A waiter below the floor takes itself out of ``st["waiters"]`` -- the head choice, the backfill order, the
    wait bound, the displacement victim all read that dict -- and may not be granted a seat.  It re-enters with
    its ORIGINAL order stamp (before every newer waiter), and the time it was held is its clock offset
    (``st["lane_off"]``, read by ``Front._asr_waiter_clocks``): held time counts against no bound.  Returns
    False (and does nothing) with the switch off."""
    if not enabled():
        return False
    ls = fr._lane_state()
    lc = ctl(fr)
    waiters = st["waiters"]
    now = time.time()
    if ls.lane_floor > 0 and ls.lane_for(rid) < ls.lane_floor:
        if rid in waiters:
            lc.waiter_held[rid] = (waiters.pop(rid), now)
            st.setdefault("fits", {}).pop(rid, None)
            st.setdefault("age_plan", {}).pop(rid, None)
            lc.mark_held(rid)
            fr.counters["lane_waiter_held"] += 1
            logger.info("WEG2 LANE-WAITER-HOLD rid=%s lane=%d floor=%d epoch=%d -- an ARRIVAL-SEAT waiter below the "
                        "floor takes no seat, counts for no wait clock and displaces nobody", rid, ls.lane_for(rid),
                        ls.lane_floor, ls.lane_epoch)
        return True
    got = lc.waiter_held.pop(rid, None)
    if got is not None:
        waiters[rid] = got[0]
        off = st.setdefault("lane_off", {})
        off[rid] = float(off.get(rid, 0.0)) + max(0.0, now - got[1])
    return False


# ---------------------------------------------------------------------------
# keepalive
# ---------------------------------------------------------------------------

def stream_open(fr: Any, rid: str, request: Any, resp: Any) -> Optional[Dict[str, Any]]:
    """leg 2 prepared the client's SSE response: register it for the keepalive (None with the switch off)."""
    if not enabled():
        return None
    e = {"resp": resp, "request": request, "boundary": True, "t_last": time.time()}
    ctl(fr).streams[str(rid)] = e
    return e


def new_sse_response() -> Any:
    """The response of an early-opened stream (a function so a test can swap the aiohttp object)."""
    from aiohttp import web

    resp = web.StreamResponse(status=200, headers={"Cache-Control": "no-cache"})
    resp.content_type = "text/event-stream"
    return resp


def pre_register(fr: Any, rid: str, request: Any) -> None:
    """A ``stream=true`` request entered ``handle_generate`` (switch on): note it, so a hold that lasts one keepalive
    period opens its SSE response (:func:`_pre_open`) before any group answered.  Nothing else changes for it."""
    if not enabled():
        return
    lc = ctl(fr)
    lc.pre[str(rid)] = {"request": request, "resp": None, "lock": None, "taken": False, "held_t0": None}
    try:
        request["weg2_lane_rid"] = str(rid)
    except Exception:  # noqa: BLE001 -- a request object without item assignment
        pass


async def pre_take(fr: Any, request: Any, rid: str) -> Optional[Any]:
    """Leg 2 reached its response: the early-opened one when there is one (the loop opens none after this call), else
    None and leg 2 builds its own as always."""
    lc = fr.__dict__.get("_lane_ctl_obj")
    e = lc.pre.get(str(rid)) if lc is not None else None
    if e is None:
        return None
    if e["lock"] is None:
        e["lock"] = asyncio.Lock()
    async with e["lock"]:
        e["taken"] = True
        return e["resp"]


async def _pre_open(fr: Any, lc: LaneCtl, rid: str, e: Dict[str, Any], floor: int, now: float) -> bool:
    """Open the held request's SSE response now: 200 + ``text/event-stream`` + the first keepalive comment.  Under the
    entry's lock against leg 2's :func:`pre_take`.  From here on the request's status is 200 -- an error later is an
    error event (``Front._lane_pre_finish``)."""
    if e["lock"] is None:
        e["lock"] = asyncio.Lock()
    async with e["lock"]:
        if e["taken"] or e["resp"] is not None:
            return False
        req = e["request"]
        resp = new_sse_response()
        try:
            await resp.prepare(req)
            await resp.write((KEEPALIVE_FMT % floor).encode())
        except Exception:  # noqa: BLE001 -- a client that left: its own handler ends the rid
            lc.pre.pop(rid, None)
            return False
        e["resp"] = resp
        try:
            req["weg2_prepared"] = True
            req["weg2_lane_pre"] = resp
        except Exception:  # noqa: BLE001
            pass
    lc.streams[rid] = {"resp": resp, "request": req, "boundary": True, "t_last": now}
    fr.counters["lane_pre_opened"] += 1
    fr.counters["lane_keepalives"] += 1
    logger.warning("%s rid=%s lane=%d floor=%d held_s=%.0f -- a held stream request got its SSE response (200, "
                   "text/event-stream) and a first %r comment before any group answered; an error from now on "
                   "is an error event", MARK_SSE_OPEN, rid, fr._lane_state().lane_for(rid), floor,
                   now - (e["held_t0"] or now), (KEEPALIVE_FMT % floor).strip())
    return True


async def keepalive_tick(fr: Any, now: Optional[float] = None) -> int:
    """One SSE comment line ``: lane-hold floor=N`` into every held open stream that has been quiet for
    ``SGLANG_WEG2_LANE_KEEPALIVE_S`` and stands at an event border.  A held stream request whose response is not
    open yet (waiting at the gate, in ``queue`` / ``_ready_for_d``, for an ARRIVAL-SEAT seat, or already past
    leg 2's ``prepare`` never) is opened after it has been held for one period (:func:`_pre_open`); its first
    comment goes out with it.  Non-stream requests get none (they hold)."""
    period = _ln.keepalive_s()
    lc = fr.__dict__.get("_lane_ctl_obj")
    if period <= 0 or lc is None or not (lc.streams or lc.pre):
        return 0
    ls = fr._lane_state()
    now = time.time() if now is None else now
    n = 0
    if lc.pre:
        for rid, e in list(lc.pre.items()):
            if e["resp"] is not None or e["taken"]:
                continue
            if ls.lane_floor <= 0 or ls.lane_for(rid) >= ls.lane_floor:
                e["held_t0"] = None
                continue
            if e["held_t0"] is None:
                e["held_t0"] = now
            if now - e["held_t0"] >= period and await _pre_open(fr, lc, rid, e, ls.lane_floor, now):
                n += 1
    if ls.lane_floor <= 0:
        return n
    for rid, e in list(lc.streams.items()):
        if ls.lane_for(rid) >= ls.lane_floor or not e["boundary"] or now - e["t_last"] < period:
            continue
        try:
            await e["resp"].write((KEEPALIVE_FMT % ls.lane_floor).encode())
        except Exception:  # noqa: BLE001 -- a client that left: its own handler ends the rid
            lc.streams.pop(rid, None)
            continue
        e["t_last"] = now
        n += 1
        fr.counters["lane_keepalives"] += 1
    return n


# ---------------------------------------------------------------------------
# LANES FIX 3: the death of the boot reaches the held streams
# ---------------------------------------------------------------------------

def boot_death_reason(fr: Any) -> Optional[str]:
    """Why the boot is dead (the lifecycle the dying rank wrote through the one state writer, ``state_file.note_rank_death``),
    or None.  No state directory / no file / an unreadable one: None -- a reading problem is no death."""
    try:
        from sglang.srt.environ import envs
        from sglang.srt.weg2 import state_file

        d = envs.WEG2_STATE_DIR.get() or None
        if not d:
            return None
        st = state_file.read(d)
    except (Exception, SystemExit):  # noqa: BLE001 -- StateFileError is a SystemExit
        return None
    life = st.get("lifecycle") or {}
    if life.get("state") != "dead":
        return None
    cause = st.get("cause") or {}  # the writer puts the cause beside the lifecycle (``state_file.transition``)
    code = cause.get("code") or "dead"
    detail = str(cause.get("detail_full") or cause.get("detail") or "")[-200:]
    grp = cause.get("group")
    return f"group {grp} died ({code}{': ' + detail if detail else ''})" if grp else f"the server died ({code}{': ' + detail if detail else ''})"


async def abort_held_streams(fr: Any, lc: LaneCtl, reason: str, now: float) -> int:
    """Every held SSE stream gets ONE named error event and is closed: the ones already open, and the ones a hold of this
    long would have opened at the next keepalive (opened here first: the status is 200 either way, only the event is new).
    Returns how many.  The handlers still waiting on their legs end later and find the stream closed (``_lane_pre_finish``
    swallows the write to a closed stream)."""
    from sglang.srt.weg2.front import named_error_chunk

    ls = fr._lane_state()
    floor = max(1, int(ls.lane_floor))
    for rid, e in list(lc.pre.items()):
        if e["resp"] is None and not e["taken"]:
            await _pre_open(fr, lc, rid, e, floor, now)
    n = 0
    for rid, e in list(lc.streams.items()):
        lc.streams.pop(rid, None)
        req = e.get("request")
        path = getattr(req, "path", "/v1/chat/completions")
        try:
            if e.get("boundary", True):  # inside an SSE event the event is not completed with a foreign one: only the close
                await e["resp"].write(named_error_chunk(path, f"WEG2 lane hold ended: {reason}"))
            await e["resp"].write_eof()
        except Exception:  # noqa: BLE001 -- a client that left
            continue
        n += 1
    fr.counters["lane_group_dead_streams"] += n
    logger.warning("%s streams=%d lane_floor=%d -- %s: every held SSE stream was answered with one named error event and closed "
                   "(the group answers nobody any more)", MARK_GROUP_DEAD, n, int(ls.lane_floor), reason)
    return n


async def group_death_tick(fr: Any, lc: LaneCtl, now: float) -> int:
    """Once a second while streams are held: read the boot lifecycle, and on ``dead`` answer them (once per death)."""
    if lc.death_announced or not (lc.streams or lc.pre) or now - lc.death_checked_t < DEATH_CHECK_S:
        return 0
    if fr._lane_state().lane_floor <= 0:
        return 0  # nothing is held: the ordinary death paths answer the streams as they always did
    lc.death_checked_t = now
    reason = await asyncio.to_thread(boot_death_reason, fr)
    if reason is None:
        return 0
    lc.death_announced = True
    return await abort_held_streams(fr, lc, reason, now)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

async def step(fr: Any, now: Optional[float] = None) -> None:
    """One pass of the lane loop (also called by the tests)."""
    if not enabled():
        return
    now = time.time() if now is None else now
    await reconcile(fr, "tick")
    ls = fr._lane_state()
    lc = ctl(fr)
    if lc.held:
        _restore_held(fr, lc, ls.lane_floor, now)  # a hold whose lane the floor no longer exceeds goes back
    if fr.state == "serving":
        if ls.lane_floor > 0 or lc.acked.get("P") or lc.acked.get("D"):
            await sync_floor(fr, now=now)
        if ls.lane_floor > 0 and not lc.park_unsupported and now - lc.park_tried >= PARK_RETRY_S:
            D = fr.groups["D"]
            if any(ls.lane_for(r) < ls.lane_floor and r not in lc.parked_d and r not in (fr.__dict__.get("_rvp_inflight") or ())
                   for r in list(D.outstanding)):
                lc.park_tried = now
                await _park_d(fr, lc, ls, ls.lane_floor, retry=True)
    await group_death_tick(fr, lc, now)
    await keepalive_tick(fr, now)


async def loop(fr: Any) -> None:
    lc = ctl(fr)
    lc.kick_evt = asyncio.Event()
    logger.info("WEG2 LANES on: lane loop started (tick %.2f s, keepalive %d s, plan PLAN-PRIO-LANES-1008 L4)",
                TICK_S, _ln.keepalive_s())
    while True:
        try:
            await asyncio.wait_for(lc.kick_evt.wait(), TICK_S)
        except asyncio.TimeoutError:
            pass
        lc.kick_evt.clear()
        try:
            await step(fr)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- the lane loop never takes the front down
            logger.exception("WEG2 lane loop error: %s", e)


# ---------------------------------------------------------------------------
# state.json
# ---------------------------------------------------------------------------

def hold_block(fr: Any) -> Dict[str, Any]:
    """``front.lane_hold`` (only with the switch on, and only once the controller has a register)."""
    lc = fr.__dict__.get("_lane_ctl_obj")
    if lc is None:
        return {}
    ls = fr._lane_state()
    if not (lc.held or lc.parked_d or lc.dormant or lc.waiter_held or lc.gates or lc.reprefill
            or ls.lane_floor > 0):
        return {}
    return {"lane_hold": {
        "held": len(lc.held), "parked_d": len(lc.parked_d), "dormant": len(lc.dormant),
        "waiters": len(lc.waiter_held), "gate": len(lc.gates),
        "reprefill": len(lc.reprefill),
        "sse_open": sum(1 for e in lc.pre.values() if e["resp"] is not None),
        "acked": {k: list(v) for k, v in sorted(lc.acked.items())},
        "unsupported": sorted(lc.unsupported) + (["park"] if lc.park_unsupported else []),
    }}
