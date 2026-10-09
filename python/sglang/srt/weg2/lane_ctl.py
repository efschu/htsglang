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
  * in flight on P: the floor RPC alone (L3 parks it at the chunk border); the front counts such a leg as held
    (not against the pool plan, not for the P->D drain, not as a stall).

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

#: the SSE comment a held stream gets (L4 -> client; an SSE comment line starts with ':' and is ignored by every
#: SSE reader, owui_proxy.py:215 skips every line that is not 'data:')
KEEPALIVE_FMT = ": lane-hold floor=%d\n\n"


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
        self.streams.pop(r, None)
        self.gates.pop(r, None)
        self.gate_done.pop(r, None)
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
        sweep(fr)
        return
    async with _lock(lc):
        ls = fr._lane_state()
        target = target_floor(fr)
        floor = ls.lane_floor
        if target == floor:
            return
        if fr.state != "serving":
            # user decision 2: a flip runs to its end, THEN the lane displaces ("kein Flip-Abbruch").  The arrival
            # goes its normal way (it queues: nothing is admitted during a flip); the first pass after
            # ``WEG2-FLIP done`` finds floor < target and preempts.
            if target > floor and lc.defer_logged != (target, getattr(fr, "epoch", 0)):
                lc.defer_logged = (target, getattr(fr, "epoch", 0))
                fr.counters["lane_defer"] += 1
                logger.warning(
                    "%s lane=%d floor=%d epoch=%d state=%s rid=%s -- a higher lane arrived while the front is "
                    "not serving (a flip runs to its end, plan decision 2): the floor rises after "
                    "WEG2-FLIP done", _ln.MARK_DEFER, target, floor, ls.lane_epoch, fr.state, rid or "-")
            return
        lc.defer_logged = None
        if target > floor:
            await _preempt(fr, lc, ls, floor, target, cause, rid)
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
        if where == "ready":
            fr._sync_batch_gate()
    if n:
        fr._park_stuck().hold(lc.held_rids())
    return n


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


async def _preempt(fr: Any, lc: LaneCtl, ls: Any, prev: int, floor: int, cause: str, rid: Optional[str]) -> None:
    t0 = time.time()
    ls.set_floor(floor)
    held = _hold_waiting(fr, lc, floor, t0)
    P, D = fr.groups["P"], fr.groups["D"]
    parked_p = [r for r in list(P.outstanding) if ls.lane_for(r) < floor]
    res = await asyncio.gather(_park_d(fr, lc, ls, floor), sync_floor(fr, force=True), return_exceptions=True)
    parked_d = res[0] if isinstance(res[0], int) else 0
    for r in res:
        if isinstance(r, BaseException):
            logger.warning("WEG2 LANE-PREEMPT step failed: %r", r)
    fr.counters["lane_preempt"] += 1
    fr.counters["lane_held"] += held
    logger.warning(
        "%s floor=%d epoch=%d parked_d=%d parked_p=%d held=%d prev_floor=%d cause=%s rid=%s ms=%.0f -- a higher "
        "lane displaces every lower one: D's running requests of lower lanes are parked (hold=lane), P's legs "
        "stop at the chunk border (floor RPC), waiting ones are held out of queue/_ready_for_d (original "
        "arrival kept)", _ln.MARK_PREEMPT, floor, ls.lane_epoch, parked_d, len(parked_p), held, prev, cause,
        rid or "-", (time.time() - t0) * 1000.0)
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
    """Park D's open requests below ``floor`` for the lane.  Returns how many D confirmed."""
    if lc.park_unsupported:
        return 0
    D = fr.groups["D"]
    rids = [r for r in list(D.outstanding) if ls.lane_for(r) < floor and r not in lc.parked_d]
    if not rids:
        return 0
    if fr.awake != "D":
        # D sleeps (P phase): what it still holds is parked / held dormant already -- the park RPC has nothing to
        # take off a batch and a sleeping group cannot be asked for it.  The front books the hold; D learns the
        # floor with the wake message and the floor RPC right after the wake (stragglers are re-parked by rid).
        t_park = time.time()
        for r in rids:
            lc.parked_d[r] = t_park
            fr._d_parked[r] = _pp.LANE_PARK_STAMP
            lc.mark_held(r)
            fr._req_book().park(r, t_park, f"lane:{floor}", fr.epoch)
        _release_seats(fr, rids)
        fr._park_stuck().hold(rids)  # after the seat release: Seat.release ends the rid's park streak
        fr.counters["lane_parked_d"] += len(rids)
        fr.counters["lane_parked_d_dormant"] += len(rids)
        return len(rids)
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
            lc.parked_d[r] = t_park
            fr._d_parked[r] = _pp.LANE_PARK_STAMP
            lc.mark_held(r)
            fr._req_book().park(r, t_park, f"lane:{floor}", fr.epoch)
            n += 1
        else:
            # a D that ignores the ``rids`` filter parked a request of the floor lane too: it is a plain park
            # for the front (it resumes with the next wake), and the floor lane is not displaced on purpose.
            fr._d_parked[r] = t_park
            over.append(r)
    if n:
        mine = [r for r in got if r in lc.parked_d]
        _release_seats(fr, mine)
        fr._park_stuck().hold(mine)  # after the seat release: Seat.release ends the rid's park streak
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


async def _resume(fr: Any, lc: LaneCtl, ls: Any, prev: int, floor: int, cause: str) -> None:
    """LANE-EMPTY: no open request of the old floor lane is left; the floor falls to the highest lane with work."""
    t0 = time.time()
    ls.set_floor(floor)
    # 1. held waiting Pendings of the new floor lane, back in the order of their ORIGINAL arrival
    back: Dict[str, List[Any]] = {"queue": [], "ready": []}
    for k, h in list(lc.held.items()):
        p = h.p
        fut = getattr(p, "fut", None)
        if fut is not None and fut.done():
            lc.held.pop(k, None)  # it ended while held (client gone, stop): nothing to resume
            continue
        if lane_of_pending(p) >= floor:
            lc.held.pop(k, None)
            p.lane_held_s = float(getattr(p, "lane_held_s", 0.0) or 0.0) + max(0.0, t0 - h.t_held)
            p.lane_state = "resuming"
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
    # 2. D's lane parks of the new floor lane end (D requeues them in arrival order on the floor RPC below)
    rel = [r for r in list(lc.parked_d) if ls.lane_for(r) >= floor]
    for r in rel:
        lc.parked_d.pop(r, None)
        fr._d_parked.pop(r, None)
    resumed_ids = [p.rid for p in back["queue"] + back["ready"]] + rel
    if resumed_ids:
        fr._park_stuck().release(resumed_ids)
    if rel:
        fr._rb_resume(rel, "lane_resume")
        fr.counters["lane_resumed_d"] += len(rel)
    # 3. the floor to P and D (D resumes the parked, P's legs run on)
    await sync_floor(fr, force=True)
    # 4. new arrivals of the lane last (the gate releases after the parked)
    n_gate = _release_gates(fr, lc, floor)
    resumed = n_q + n_r + len(rel) + n_gate
    fr.counters["lane_resume"] += 1
    logger.warning(
        "%s floor=%d->%d epoch=%d resumed=%d held_back=%d parked_d=%d gate=%d cause=%s ms=%.0f -- no request of "
        "lane %d is left: the floor falls to the highest lane with work; its parked requests run first, with "
        "their original arrival, the arrivals held at the gate after them", _ln.MARK_RESUME, prev, floor,
        ls.lane_epoch, resumed, n_q + n_r, len(rel), n_gate, cause, (time.time() - t0) * 1000.0, prev)
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


async def keepalive_tick(fr: Any, now: Optional[float] = None) -> int:
    """One SSE comment line ``: lane-hold floor=N`` into every held open stream that has been quiet for
    ``SGLANG_WEG2_LANE_KEEPALIVE_S`` and stands at an event border.  Non-stream requests get none (they hold)."""
    period = _ln.keepalive_s()
    lc = fr.__dict__.get("_lane_ctl_obj")
    if period <= 0 or lc is None or not lc.streams:
        return 0
    ls = fr._lane_state()
    if ls.lane_floor <= 0:
        return 0
    now = time.time() if now is None else now
    n = 0
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
    if fr.state == "serving":
        if ls.lane_floor > 0 or lc.acked.get("P") or lc.acked.get("D"):
            await sync_floor(fr, now=now)
        if ls.lane_floor > 0 and not lc.park_unsupported and now - lc.park_tried >= PARK_RETRY_S:
            D = fr.groups["D"]
            if any(ls.lane_for(r) < ls.lane_floor and r not in lc.parked_d and r not in (fr.__dict__.get("_rvp_inflight") or ())
                   for r in list(D.outstanding)):
                lc.park_tried = now
                await _park_d(fr, lc, ls, ls.lane_floor, retry=True)
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
    if not (lc.held or lc.parked_d or lc.gates or lc.reprefill or ls.lane_floor > 0):
        return {}
    return {"lane_hold": {
        "held": len(lc.held), "parked_d": len(lc.parked_d), "gate": len(lc.gates),
        "reprefill": len(lc.reprefill),
        "acked": {k: list(v) for k, v in sorted(lc.acked.items())},
        "unsupported": sorted(lc.unsupported) + (["park"] if lc.park_unsupported else []),
    }}
