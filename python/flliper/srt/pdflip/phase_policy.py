"""H91 part C: the Weg-2 front's PHASE POLICY (user design 25.09.2026).

The pure half of the policy, so every verdict can be read at a desk without a
front, a socket or a loop. :mod:`flliper.srt.pdflip.front` owns the state and the
calls; this module owns the arithmetic and the wire shapes.

THE THREE RULES (user, 25.09., binding):

1. P PHASE -- at most ``--p-phase-max-requests`` (6) requests per P phase.
   How many of them are prefilled OVERLAPPING is planned against P's unified
   pool (``--p-pool-tokens``, 262k on NF) from the waiting requests'
   ``est_prompt``: the next request is dispatched only while the in-flight
   ones plus it fit the pool (one is always admitted, so a request larger than
   the pool is refused by P by name -- PDFLIP P-INTAKE-TOO-LARGE -- instead of
   starving here). Finished prefills leave P's VRAM into the L2/L3 chain
   (part A), so the per-phase count is bounded by the cap, not by the pool.
   The P phase ends when the cap is dispatched or the queue is empty.
2. D PHASE -- D decodes EVERY request this P phase handed over before the
   front flips back; the count rides on the wake message to D
   (``handoff_n``, read by part B).
3. WAIT BOUND -- "wartegrenze 60 s, dann zurueck zu P": a request waiting for
   P for longer than ``--d-wait-bound-s`` (60) DURING the D phase makes the
   front park D's running decodes (``POST /pdflip/park_running``, part B) and
   flip to P. A parked request stays in flight for the front (its client
   stream keeps running, no requeue, no second leg 1) and continues in the
   next D phase before the new ones; D orders them.

WHAT "WAITING DURING THE D PHASE" MEASURES. The wait is taken from the later
of the request's arrival at the front and the start of the current D phase:
``now - max(t_arrive, t_awake)``. For a request that arrives during the D
phase that IS its arrival. For one that already waited through a P phase
(left behind by the 6-cap) the D phase's own start is the origin -- counting
its P-phase wait too would fire the bound in the first seconds of every D
phase after a capped P phase and so contradict rule 2 by construction (a D
phase of 2 s, a park, a flip, again). ``Pending.t_arrive`` is the front's
arrival stamp; it is re-stamped only where D stopped serving the request
(reroute, W31/W50 requeue), which is rule 4: a request D serves itself is not
waiting.
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# H91c3-3: imported with this module (the front imports it before its loop),
# never lazily from `d_phase_seats` -- d_seats is stdlib-only at import time.
from flliper.srt.pdflip import d_seats as _d_seats

#: Rule 1: requests per P phase (user 25.09.: "bis zu 6").
P_PHASE_MAX_REQUESTS_DEFAULT = 6
#: Rule 1: P's unified KV pool in tokens (NF: 262k, memory `262K-PFLICHT`). Since S55 (P-POOL-FROM-P) only the
#: value until P answered: the standard form then plans against P's realised pool (resolve_p_pool_tokens).
P_POOL_TOKENS_DEFAULT = 262144
#: Rule 3: the wait bound in seconds (user verbatim: "wartegrenze 60 s").
D_WAIT_BOUND_S_DEFAULT = 60.0
#: Rule 3: D's park endpoint (part B builds it).
PARK_PATH = "/pdflip/park_running"
#: The park RPC's bound. Part B answers after it took the running requests
#: off the batch and started their forced write-through (GiB for two large
#: contexts) -- the front waits for that answer and flips only then, so the
#: sleep's own 10 s drain bound (W120) is not what pays for it.
PARK_TIMEOUT_S = 60.0
#: Part B: a parked request that no sleep follows is re-queued by D itself
#: after this long. Past it the front counts the rid as RUNNING again, so its
#: drain and W3 ledger cannot disagree with a D that took the request back.
PARK_REQUEUE_S = 30.0
#: A D without the endpoint: the route does not exist (404) or the server
#: says it does not implement it (501). Both mean "old D" -- the front falls
#: back to the pre-H91 behaviour for the rest of the boot. Anything else
#: (409 = the server is not group D, 5xx, a timeout) is a failed park: the
#: fallback holds for this D phase only.
PARK_UNSUPPORTED_STATUSES = (404, 501)
#: Part A (P) no longer refuses an admissible request with 503: it waits in
#: P's queue. A leg 1 that then never starts would hold the P phase forever,
#: so the front bounds it itself: ``--p-leg1-stall-s`` (default below) with
#: neither a leg-1 completion nor a change in P's own progress counters
#: inside the window = stalled -> aborted on P, requeued at the head, the
#: drain ends and the flip to D follows (the PDFLIP-INTAKE-STALL path).
P_LEG1_STALL_S_DEFAULT = 180.0
#: How often an in-flight leg 1 is checked against that bound.
P_LEG1_STALL_CHECK_S = 1.0

PARK_PARKED = "parked"
PARK_UNSUPPORTED = "unsupported"
PARK_FAILED = "failed"


def p_request_cost(est_prompt: int, leg1_prompt_tokens: int = 0,
                   skip_leg1: bool = False) -> int:
    """Tokens a request occupies in P's pool while its leg 1 runs.

    The WHOLE prompt, not the uncached remainder: P materialises the cached
    head from the store into its own pool before it prefills the rest. The
    realised count wins over the arrival estimate once known. A request that
    skips leg 1 (route CARRIER-EXCEEDS) never enters P's pool.
    """
    if skip_leg1:
        return 0
    return max(0, int(leg1_prompt_tokens or 0), int(est_prompt or 0))


def p_continuation_credit(pairs: Iterable[Tuple[int, int]]) -> int:
    """P-POOL-SHARED (rc29 pphase 21.10.): tokens of a dispatch candidate that P's pool already holds for a leg in flight.

    ``pairs`` = ``(common, depth)`` per leg in flight on P: ``common`` = the token prefix the candidate shares with that
    leg (front token ids), ``depth`` = the cached prefix that leg loads into P's device pool (its store depth, the told
    point, anchor-aligned). When ``common >= depth`` the leg's whole loaded prefix IS the candidate's prefix: P's radix
    tree matches it there (the leg's told point carries the anchor), so the candidate occupies only what lies beyond.
    A shorter common prefix credits NOTHING -- where the candidate's match would end needs an anchor below ``common``
    that the front does not know (conservative: never under-count P's pool). The largest credit wins (one leg's
    nodes). Metal rc27 _023225: pdflip-0-21 (prompt 149741, u 2029) continues pdflip-0-17 (store depth 147712, in
    flight) -- counted as 149741 it was held 23.2 s by the overlap plan while P's pool usage read 0.17-0.43.
    """
    best = 0
    for common, depth in pairs:
        c, d = int(common or 0), int(depth or 0)
        if d > 0 and c >= d and d > best:
            best = d
    return best


def p_overlap_admits(inflight_tokens: int, inflight_n: int, next_cost: int,
                     pool_tokens: int) -> bool:
    """May the next request's leg 1 start while ``inflight_n`` run?

    ``pool_tokens <= 0`` = no token plan (count bound only). With nothing in
    flight the answer is always yes: a request the pool cannot hold at all
    is P's to refuse by name, never the front's to starve.
    """
    if pool_tokens <= 0 or inflight_n <= 0:
        return True
    return int(inflight_tokens) + int(next_cost) <= int(pool_tokens)


#: S55 P-POOL-FROM-P: the front's overlap budget is P's REALISED KV pool, read once off P's own
#: ``/get_server_info`` (``max_total_num_tokens``: the scheduler handshake's value, min-reduced across
#: P's PP ranks -- ``KV token sizing ... min-reduced across ranks to N``). One reading, made by the
#: front; the ranks get dispatches as before (SOLL R11). ``FLLIPER_PDFLIP_P_POOL_FROM_P=0`` = the
#: configured value alone, byte-identical to rc14.
P_POOL_SRC_REALISED = "p_realised"
P_POOL_SRC_WRITTEN = "written"
P_POOL_SRC_CLAMPED = "written_clamped"
P_POOL_SRC_CONFIGURED = "configured"
P_POOL_SRC_OFF = "off"


def p_pool_realised(info) -> Optional[int]:
    """P's realised KV pool in tokens from a ``/get_server_info`` body, else None.

    Only the top-level ``max_total_num_tokens`` counts: that is the scheduler's handshake
    value (``Scheduler.get_init_info``), i.e. the pool P actually allocated after the
    cross-rank min-reduce. ``max_total_tokens`` is the ARGV cap (a profile pin) and is never
    read here -- a pin above P's own sizing would plan an overlap P cannot hold. A list body
    (one entry per DP rank) reads its first entry. Anything not a positive int is None."""
    if isinstance(info, list):
        info = info[0] if info else None
    if not isinstance(info, dict):
        return None
    raw = info.get("max_total_num_tokens")
    if isinstance(raw, bool):
        return None
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def resolve_p_pool_tokens(configured: int, written: bool,
                          realised: Optional[int]) -> Tuple[int, str]:
    """The overlap budget the front plans against, and why (S55).

    * ``configured <= 0`` -- the rule is OFF (qwen27b, ``--p-pool-tokens 0``): stays 0.
    * P not read yet -- the configured value (rc14 behaviour).
    * not written (the H91 standard-form default) -- P's realised pool: the overlap goes
      "je nachdem wie viel vram context zur verfuegung steht" (user 25.09., H91 design).
    * written -- the operator's value, but NEVER above P's realised pool: a plan above it
      dispatches a leg 1 whose pages P cannot hold (it waits in P's queue, its store read
      fights the running request for room).
    """
    c = max(0, int(configured or 0))
    if c <= 0:
        return 0, P_POOL_SRC_OFF
    if realised is None or int(realised) <= 0:
        return c, P_POOL_SRC_CONFIGURED
    r = int(realised)
    if not written:
        return r, P_POOL_SRC_REALISED
    if c > r:
        return r, P_POOL_SRC_CLAMPED
    return c, P_POOL_SRC_WRITTEN


def plan_p_overlap(costs: Sequence[int], pool_tokens: int, cap: int) -> int:
    """How many of the waiting requests (head first) P prefills at once.

    The same rule :func:`p_overlap_admits` applies per dispatch, folded over
    the queue head for the drain's plan line. ``cap <= 0`` = no count cap.
    Always >= 1 when anything is waiting.
    """
    n = 0
    used = 0
    for c in costs:
        if cap > 0 and n >= cap:
            break
        if not p_overlap_admits(used, n, c, pool_tokens):
            break
        used += int(c)
        n += 1
    return n


def phase_cap_reached(dispatched: int, cap: int) -> bool:
    """Rule 1: the P phase has dispatched its ``cap`` requests (0 = no cap)."""
    return cap > 0 and dispatched >= cap


def d_phase_wait_s(t_arrive: float, t_phase_start: float, now: float) -> float:
    """Rule 3: how long a request has waited for P DURING this D phase."""
    return max(0.0, float(now) - max(float(t_arrive), float(t_phase_start)))


def oldest_d_phase_wait(arrivals: Iterable[float], t_phase_start: float,
                        now: float) -> Optional[float]:
    """The longest :func:`d_phase_wait_s` over ``arrivals``; None when empty."""
    waits = [d_phase_wait_s(t, t_phase_start, now) for t in arrivals]
    return max(waits) if waits else None


def wait_bound_fired(wait_s: Optional[float], bound_s: float) -> bool:
    """Rule 3: the bound is armed (> 0) and the oldest wait reached it."""
    return bound_s > 0 and wait_s is not None and wait_s >= bound_s


#: PRIORITY LANES 1008 (L4): the park stamp of a request D holds for a LANE (``hold="lane"``): it never lapses.
#: D keeps such a park until the lane floor falls (no 30-s requeue, no #248h capacity requeue), so the front's
#: part-B clock must not count it as running again. ``float("inf")`` makes every ``now - t`` comparison False
#: without a special case at any of the lapse sites.
LANE_PARK_STAMP = float("inf")
#: the body keys of the lane park (``POST /pdflip/park_running``, L2 reads them)
LANE_PARK_HOLD = "lane"
PARK_REASON_LANE = "lane-preempt"


def lane_park_body(epoch: int, rids: Sequence[str], floor: int, lane_epoch: int) -> dict:
    """PRIORITY LANES 1008 (L4 -> L2): ``POST /pdflip/park_running`` for exactly ``rids`` (the lower-lane requests D
    runs), held for the lane: ``{"epoch", "reason", "rids", "hold": "lane", "floor", "lane_epoch"}``. Without
    ``rids`` / ``hold`` the body is :func:`park_body`'s, so an old D that ignores the new keys would park every
    running request -- the front reads the answer's ``parked`` list and parks nothing it was not told about."""
    body = park_body(epoch, 0.0, reason=PARK_REASON_LANE)
    body["rids"] = [str(r) for r in rids]
    body["hold"] = LANE_PARK_HOLD
    body["floor"] = int(floor)
    body["lane_epoch"] = int(lane_epoch)
    return body


def park_answered_at(status: int, text: str) -> Optional[float]:
    """FLIPSTALL-1015: D's flip park answer's ``answered_at`` (wall clock of
    the moment D's park answered and started its re-queue clock); None when
    absent, non-positive or unreadable (old D, failed park)."""
    if status != 200:
        return None
    try:
        js = json.loads(text)
    except Exception:  # noqa: BLE001
        return None
    v = js.get("answered_at") if isinstance(js, dict) else None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
        return None
    return float(v)


def park_lapse_stamp(t_park: float, answered_at: Optional[float], t_recv: float) -> float:
    """FLIPSTALL-1015: where the front's lapse clock (:func:`parked_lapsed`)
    starts for the rids D parked. H91c3-1 stamps BEFORE the RPC so the front
    lapses no later than D re-queues -- right while D's clock started in the
    RPC. D's clock now starts when its park ANSWERS (``answered_at``; a park
    took 31 s on NF h16n 06:56:31-06:57:02Z, and a pre-RPC stamp would lapse
    the front 54 s early: it would count as running what D still holds). The
    answer's moment, clamped into ``[t_park, t_recv]`` (same host clock; the
    clamp only absorbs a skewed stamp), is no later than D's re-queue, which
    fires once every D rank's stamp is ``PARK_REQUEUE_S`` old. Without the
    field: ``t_park``, the H91c3-1 stamp."""
    if answered_at is None:
        return float(t_park)
    return float(min(max(float(answered_at), float(t_park)), float(t_recv)))


def parked_lapsed(t_parked: float, now: float) -> bool:
    """Part B re-queues a parked request no sleep followed after
    :data:`PARK_REQUEUE_S`; from then on it is running on D again. A lane park
    (:data:`LANE_PARK_STAMP`) is held by D until the floor falls and never lapses."""
    if t_parked == LANE_PARK_STAMP:
        return False
    return float(now) - float(t_parked) >= PARK_REQUEUE_S


def leg1_stalled(now: float, t_dispatch: float, t_last_evidence: float,
                 bound_s: float) -> bool:
    """A leg 1 in flight for at least ``bound_s`` while P showed NO work for
    ``bound_s`` either: no leg 1 of this drain completed and P's progress
    counters did not move (``t_last_evidence`` is the later of the two, or
    the dispatch itself). ``bound_s <= 0`` = off."""
    if bound_s <= 0:
        return False
    return (float(now) - float(t_dispatch) >= bound_s
            and float(now) - float(t_last_evidence) >= bound_s)


def park_reason(bound_s: float) -> str:
    """``wait-bound-60s`` for the default; the configured bound otherwise."""
    return f"wait-bound-{float(bound_s):g}s"


#: 27B PARK (user 26.09.): the reason an immediate park names on the wire.
PARK_REASON_IMMEDIATE = "immediate-over-x"


def park_body(epoch: int, bound_s: float, reason: Optional[str] = None) -> dict:
    """The park request: ``{"epoch": <int>, "reason": "wait-bound-60s"}``
    (``reason`` given: that string, e.g. :data:`PARK_REASON_IMMEDIATE`)."""
    return {"epoch": int(epoch), "reason": reason or park_reason(bound_s)}


# ---------------------------------------------------------------------------
# 27B PARK (user decision 26.09. ~19:00Z, memory d2p-sofort-flippen-und-x-
# exakt-0926): "Wartegrenze 0". A request whose PENDING tokens exceed X while D
# decodes parks D's running decodes at once and the front flips to P -- no
# 60 s bound, no waiting for D's decodes to end. Only a request that NEEDS P
# fires it: one D could serve itself once idle (a SHORT behind D's budget, a
# SHORT-only backlog the 27B idle policy (b) drains) never does -- for those no
# flip is due at all.
#
# PK2 (metal dkr27bparkdraftbar1w209270712, 27.09.): a BAND request the RC7-X
# busy/idle split deferred (Pending.x_deferred: X_busy < uncached <= live X,
# routed LONG because D was decoding) DOES fire. While D decodes, the X in force
# is X_busy (the router's own verdict); the idle re-grant only serves it once D
# is idle, i.e. after D's decodes end -- pdflip-28-62 (4172 > X_busy 4096 <= live X
# 7290) waited 45 s for the fairness bound and 65 s more for the drain (flip 67 s).
# ---------------------------------------------------------------------------

def needs_p(est_uncached: int, x_tokens: int, *, skip_leg1: bool = False,
            leg1_done: bool = False, p_only: bool = False, x_requeues: int = 0,
            x_deferred: bool = False, x_routed: int = 0, handoff_lost_reroutes: int = 0,
            reroutes: int = 0) -> bool:
    """Does this queued request need P's prefill (the immediate park's
    trigger)? ``est_uncached > X`` (law 4: D never prefills above X), a
    P-only request (an image under transient vision, long by rule), or one D
    already refused as over X (W31, ``x_requeues``). Never a request whose
    leg 1 is done (it waits for D, not for P) or that skips leg 1 (route
    CARRIER-EXCEEDS: D prefills it once, a flip to P buys nothing).

    #246b (NF y4c 04:39:57, pdflip-22-37): ``x_routed`` > 0 is the X the front
    ROUTED this request on (the X-SOLO band floor X_busy while D is busy); above
    it the request was sent to P and needs P, even when it lies below the live
    X ``x_tokens`` -- otherwise it is neither D's (routed away) nor P's (no
    trigger) and waits until D runs empty."""
    if skip_leg1 or leg1_done:
        return False
    if p_only or int(x_requeues or 0) > 0 or x_deferred:
        return True  # x_deferred: over X_busy, the X in force while D decodes (PK2)
    if int(reroutes or 0) > 0:
        # SOLL R9/N1: a request that MUST run on P by state (leg-2 reroute) is P-only by state even under X: a set
        # member that counts as big, never a starved queue entry
        return True
    if int(handoff_lost_reroutes or 0) > 0 and int(est_uncached) > int(x_tokens):
        # S28 (R7/R1, 18.10. "flip auf P fuer 34 token"): a lost hand-off is P-by-state only while its CURRENT price is
        # over X. The store holds the span (L3-INDEX-PRESENCE / P-ANCHOR-PRESENCE): at u <= X D takes the tail itself
        # (R2) -- "kann D einen Request unter X nicht nehmen, ist das ein Defekt, kein Flipgrund" (R7)
        return True
    if int(x_routed or 0) > 0 and int(est_uncached) > int(x_routed):
        return True  # #246b: routed to P on this X
    return int(est_uncached) > int(x_tokens)


DECIDER_ENV = "FLLIPER_PDFLIP_DECIDER"


def decider_on(env=None) -> bool:
    """EZ-S2a switch ``FLLIPER_PDFLIP_DECIDER`` (default ON, user 14.10.: new features on). Off = the S1 set code (the old
    code stays reachable until S4)."""
    if env is None:
        try:
            from flliper.srt.environ import envs
            return bool(envs.FLLIPER_PDFLIP_DECIDER.get())
        except Exception:  # noqa: BLE001 -- a partial env: ON (new features on)
            return True
    raw = (env.get(DECIDER_ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def needs_p_class(est_uncached: int, x_own: int, *, skip_leg1: bool = False, leg1_done: bool = False,
                  p_only: bool = False, reroutes: int = 0, no_number: bool = False) -> bool:
    """EZ-S2a (plan A5, R4/R7/R9): does this request need P -- read from its CLASS, not from state flags. ``u >= X_own``
    (gross, R4) or ``p_only`` (vision / only-P, R9) or a leg-2 reroute (P by state, the N1 note of :func:`needs_p`). A
    request D refused (``x_requeues``), a carrier, a deferred or a lost hand-off UNDER its X is none of these: D can take
    it, that it did not is a defect (R7: a ``DECISION rule=R7`` line, never a flip reason). Halb is a SET member
    (R3), not P-bound by itself."""
    if skip_leg1 or leg1_done:
        return False
    if p_only or int(reroutes or 0) > 0 or no_number:
        return True      # no_number: D refused it and named no extent -- D's word, not the front's stale estimate, stands
    return int(est_uncached) >= max(1, int(x_own))


def immediate_park_trigger(queue: Iterable, x_tokens: int):
    """The first queued request (head first) that :func:`needs_p`, or None.
    Reads ``Pending``'s own fields by name (getattr: partial test doubles). EZ-S2a: with the decider ON the class
    decides (:func:`needs_p_class`) on the entry's own X (``x_set``, else ``x_tokens`` as the cap)."""
    if decider_on():
        for p in queue:
            if getattr(p, "measure_reoffer", False):
                continue      # S2b: D measures it (MEASURE barrier): its price is no number yet -- never a P trigger
            _xo = int(getattr(p, "x_set", 0) or 0) or int(x_tokens)
            if needs_p_class(int(getattr(p, "est_uncached", 0) or 0), _xo,
                             skip_leg1=bool(getattr(p, "skip_leg1", False)),
                             leg1_done=bool(getattr(p, "leg1_done", False)),
                             p_only=bool(getattr(p, "p_only", False)),
                             reroutes=int(getattr(p, "reroutes", 0) or 0),
                             no_number=bool(int(getattr(p, "x_requeues", 0) or 0) > 0 and getattr(p, "x_ref_nonum", False))):
                return p
        return None
    for p in queue:
        # EZ-S1 review B1: a MEASURE returnee D measured under X (kurz/halb) is not "P-only by the state requeued" -- it is
        # executed by what D measured (a halb member of the set, R3; a kurz one goes back to D, R2); over X it needs P as ever
        _xr = int(getattr(p, "x_requeues", 0) or 0)
        if getattr(p, "measure_class", "") in ("kurz", "halb"):
            _xr = 0
        if needs_p(int(getattr(p, "est_uncached", 0) or 0), x_tokens,
                   skip_leg1=bool(getattr(p, "skip_leg1", False)),
                   leg1_done=bool(getattr(p, "leg1_done", False)),
                   p_only=bool(getattr(p, "p_only", False)),
                   x_requeues=_xr,
                   x_deferred=bool(getattr(p, "x_deferred", False)),
                   x_routed=int(getattr(p, "x_routed", 0) or 0),
                   handoff_lost_reroutes=int(getattr(p, "handoff_lost_reroutes", 0) or 0),
                   reroutes=int(getattr(p, "reroutes", 0) or 0)):
            return p
    return None


def immediate_park_dwell_ok(awake_s: float, min_dwell_ms: float, floor_ms: float) -> bool:
    """The immediate park pre-empts D like the fairness bound does, so it
    keeps the fairness bound's floor (weg2xsn291: never leave a group that
    woke 200 ms ago) AND the derived min-dwell (K7: a D phase lasts at least
    the price of the flip that started it) -- the parked decodes get that
    much decode per D phase, so a stream of long arrivals cannot starve them
    to zero progress. Both are inputs: the front owns the clock and K7."""
    return float(awake_s) * 1000.0 >= max(float(min_dwell_ms), float(floor_ms))


#: PARK-CYCLE DWELL (27B rc12k27 b23, 27.09.): switch, default on; 0 = K7's
#: one-flip dwell for every D phase, byte for byte.
PARK_CYCLE_DWELL_ENV = "FLLIPER_PDFLIP_PARK_CYCLE_DWELL"


def park_cycle_dwell_on(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(PARK_CYCLE_DWELL_ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def park_cycle_dwell_ms(d_to_p_ms: float, p_to_d_ms: float, resumed_phase: bool,
                        on: bool = True) -> float:
    """The dwell an immediate park waits for in a D phase.

    K7 prices a D phase at ONE flip (the D->P flip the park starts). A D phase
    that began by RESUMING parked requests paid the whole park cycle for them
    already -- the D->P flip, P's work, the P->D flip back -- and re-parking
    them after one flip's worth of decode is the measured thrash (b23 10:02:38-
    10:04:24: 6 parks, D awake 3.0/3.2/10.2/2.6/18.7/20.5 s, the same six rids
    parked up to 5 times in 80 s, D decoding 55 % of the wall). Such a phase is
    priced at the cycle's two flips (D->P + P->D, both measured, K7's own
    source), so every park round trip buys at least as much decode as it costs
    in flips; an over-X arrival inside that window rides on the one park that
    follows it (the queue drains on P as a batch). Every other D phase -- a
    fresh wake, a phase that parked nothing -- keeps K7 unchanged, and so does
    the switch off. Pure: the front owns the clock and the flip log."""
    d_to_p = max(0.0, float(d_to_p_ms))
    if not on or not resumed_phase:
        return d_to_p
    return d_to_p + max(0.0, float(p_to_d_ms))


PARK_DECODE_DWELL_ENV = "FLLIPER_PDFLIP_PARK_DECODE_DWELL"


def park_decode_dwell_on(env=None) -> bool:
    """NF rc12z22 (boot ...dauer09281447, 14:52-14:58): 22 flips in 374 s, D
    decoding 1-3 s per 7-16 s D phase -- the resumed requests' store->device
    reload took 4-8 s after every wake, and the dwell (priced in FLIP time
    from the wake) fired the next park just as decode began: 335 completion
    tokens/min for every agent together. Default ON; "0" = the wake clock."""
    e = os.environ if env is None else env
    raw = (e.get(PARK_DECODE_DWELL_ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


PARK_DECODE_DUTY_ENV = "FLLIPER_PDFLIP_PARK_DECODE_DUTY"


def park_decode_duty(env=None) -> float:
    """Target decode share of a park cycle (default 0.5: D decodes at least
    as long as the cycle costs); clamped to [0.1, 0.9]."""
    e = os.environ if env is None else env
    try:
        v = float((e.get(PARK_DECODE_DUTY_ENV, "") or "0.5").strip())
    except ValueError:
        v = 0.5
    return min(0.9, max(0.1, v))


def park_decode_dwell_ok(awake_s: float, decode_s: Optional[float], cycle_ms: float,
                         floor_ms: float, duty: float = 0.5) -> bool:
    """The immediate park waits until D has DECODED at least the cycle it
    costs: ``cycle_ms`` = the D->P + P->D flips (K7, both measured) plus this
    phase's own resume (wake -> first decoded chunk), so every park round trip
    buys at least as much decode as it costs -- decode duty >= 50 %.
    ``decode_s`` is the time since this phase's first decoded chunk (None:
    none seen yet). Without a decoded chunk the phase has not decoded at all:
    it holds for twice the cycle on the wake clock, then fires (a phase that
    streams nothing -- non-stream legs only -- must not hold to the fairness
    bound). Pure: the front owns the clocks."""
    d = min(0.9, max(0.1, float(duty)))
    need = max(float(cycle_ms) * d / (1.0 - d), float(floor_ms)) / 1000.0
    if decode_s is None:
        return float(awake_s) >= 2.0 * need
    return float(decode_s) >= need


# --------------------------------------------------------------------------- SET SCHEDULER (SOLL v2, 17.10.)
#
# SCHEDULER-SOLL.md v2 (R1-R11) as pure functions: no clock, no I/O. The front owns the clocks, the set's
# membership and the flip; this module owns the arithmetic and the verdict names (SOLL section 4).
#
# This REPLACES the Ski-Rental ``park_collect_window`` (opening at S > X, firing on rent >= price, never fired in
# the field) and the DECODE-COLLECT window/epoch rule: opening at halbX, timer ``T_halb`` (a flag), firing at S >= X.

from dataclasses import dataclass  # noqa: E402

CLASS_KURZ = "kurz"
CLASS_HALB = "halb"
CLASS_GROSS = "gross"
CLASS_P_ONLY = "p_only"

V_COLLECT = "COLLECT"
V_FLIP_P = "FLIP_P"
V_D_AFTER_TIMER = "D_AFTER_TIMER"
V_PARK_NO_SEAT = "PARK_NO_SEAT"
V_REROUTE_P = "REROUTE_P"
#: The verdict of a kurz arrival (R2); it never reaches :func:`set_verdict`, the front logs it.
V_D_NOW = "D_NOW"


@dataclass(frozen=True)
class SetMember:
    rid: str
    uncached: int
    t_arrive: float
    p_only: bool = False
    #: H16V F1/A5: the request's OWN uncached tokens at arrival when ``uncached`` was reduced by the prefix an earlier
    #: member of the same session already brings (0 = ``uncached`` is the own figure). ``uncached`` is what P prefills
    #: (it matches the shared prefix once: P-log 913/912 at 15:36); the own figure is what D must be assumed to prefill
    #: (D's in-batch dedupe is not guaranteed: schedule_policy's check only runs for lpm/dfs-weight and a match of
    #: <= 32 tokens), so the DECISION S (against X, the D-prefill max) sums the own figure, and classes follow it too
    #: (R2: classified at arrival). The deduplicated sum is reported as ``S_P`` only.
    own_uncached: int = 0
    #: EZ-S2a (R2a): the member's OWN X (``x_own(depth)``, one function for every site). 0 = the set's class X (section 4 callers
    #: that know no per-member X: byte-identical verdicts). The class of a member follows ITS X, S is checked against the
    #: smallest of them.
    x_own: int = 0


@dataclass(frozen=True)
class SetVerdict:
    verdict: str
    why: str
    left_s: float
    s_tokens: int
    half_x: int


def half_x_of(x_tokens: int, half_frac: float) -> int:
    """halbX = F_halb x X (SOLL section 2), truncated to whole tokens."""
    return int(max(0.0, float(half_frac)) * max(0, int(x_tokens)))


def classify_arrival(uncached: int, x_tokens: int, half_frac: float, p_only: bool) -> str:
    """R2/R3/R4/R9: the class of ONE request at its arrival -- the only place a single request decides anything
    (R1). ``p_only`` (vision / only-P) counts as big (R9) and wins over the length."""
    if p_only:
        return CLASS_P_ONLY
    u = int(uncached)
    if u >= int(x_tokens):
        return CLASS_GROSS
    if u >= half_x_of(x_tokens, half_frac):
        return CLASS_HALB
    return CLASS_KURZ


def set_verdict(members: Sequence[SetMember], *, x_tokens: int, half_frac: float, t_half_s: float,
                now: float, t_open: Optional[float], d_decoding: bool, seats_free: int,
                d_prefill_running_uncached: int = 0, new_since_last: int = 0,
                class_x_tokens: Optional[int] = None, d_granted_unstarted: int = 0) -> SetVerdict:
    """The set's verdict (SOLL R1, R3-R6, R8, R9). ``members`` are the collected halb/big requests (a kurz one never
    belongs to a set, R2). ``t_open`` is the stamp of the FIRST member (None: the oldest member's arrival) -- the
    timer starts there ONCE per set (R3), later members never restart it. ``S`` = sum of the members' uncached
    tokens plus ``d_prefill_running_uncached`` (what D is prefilling right now, R6).

    Order: (1) R6 a D prefill runs and S > X -> REROUTE_P; (2) nothing decodes on D (R8) -> decide now: P if S >= X
    or a big member, else D; (3) D decodes: a big member (R4/R9) waits ``t_half_s`` for company -- a second
    member or the timer's end flips; without a big member S >= X flips at once (R3), the timer's end with S < X
    sends the set to D (R3). A flip verdict is PARK_NO_SEAT when fewer free D seats than members (R5): the set is
    prefilled on P and parked, its decode starts from L2/L3 when a seat frees. ``new_since_last`` (members that
    joined since the caller's last look) is informational: it only enters the ``why`` text. ``class_x_tokens`` (H16V
    F2): the X the members are CLASSIFIED with (their own X at arrival, R2a) when ``x_tokens`` is the larger X of a set
    of k members (P fixed costs of the further members included); default: ``x_tokens``.

    EZ-S2a (plan §2.6 / review S49, tag ``NS:S49``): ``d_granted_unstarted`` = tokens of halb/big D prefills D has been
    GRANTED but has not BEGUN (a kurz grant never reaches this term: R2 "sofort auf D", R6k "sein Rest zählt nicht zu S";
    the front leaves kurz rows out). They are no R6 reason (user 18.10. "Nur echt laufend zaehlt": only
    ``d_prefill_running_uncached`` reaches R6), but a set whose T_halb ran out WHILE D DECODES (R3 timer end) and that
    would go to D while ``S + granted > X`` WAITS (COLLECT, no flip, R7) until D begins or finishes the grant. Only that D
    verdict is held, every flip verdict stands. R8 (nothing decodes on D) is NEVER held: "ohne Decode auf D wird sofort
    entschieden: D bei S < X, P bei S >= X" -- a COLLECT at d_decoding=0 is the rc30 06:36:19Z violation (pdflip-0-4,
    S=3821 + 2226 kurz grants). 0 (default) = nothing held: the verdicts are byte-identical to before S2a (the section-4
    compatibility test pins it)."""
    x = int(x_tokens)
    xc = int(class_x_tokens) if class_x_tokens else x
    hx = half_x_of(xc, half_frac)
    running = max(0, int(d_prefill_running_uncached))
    if not members:
        return SetVerdict(V_COLLECT, "empty", 0.0, running, hx)
    # R1u (user decision 20.10.): S = the tokens to compute NEW, a shared uncached piece once (``uncached`` already is the
    # members' union figure, ``own_uncached`` the figure at arrival: it classes the member, R2, and no longer sums); the
    # running D prefill's rest arrives with its overlap against the members already taken out (``running_overlap``)
    s = sum(max(0, int(m.uncached)) for m in members) + running
    classes = [classify_arrival(m.own_uncached or m.uncached, int(m.x_own) or xc, half_frac, m.p_only) for m in members]
    big = any(c in (CLASS_GROSS, CLASS_P_ONLY) for c in classes)
    t0 = float(t_open) if t_open is not None else min(float(m.t_arrive) for m in members)
    left = max(0.0, float(t_half_s) - (float(now) - t0))
    new = int(new_since_last)
    seats_ok = int(seats_free) >= len(members)

    def _flip(why: str) -> SetVerdict:
        if seats_ok:
            return SetVerdict(V_FLIP_P, why, left, s, hx)
        return SetVerdict(V_PARK_NO_SEAT, "R5 %d seats < %d members; %s" % (max(0, int(seats_free)), len(members), why),
                          left, s, hx)

    if running > 0 and s > x:
        return SetVerdict(V_REROUTE_P, "R6 S=%d > X=%d with a D prefill of %d running (new=%d)" % (s, x, running, new),
                          left, s, hx)
    def _d_verdict(why: str, left_s: float) -> SetVerdict:
        gu = max(0, int(d_granted_unstarted))
        if gu > 0 and s + gu > x:          # S49: a grant D has not begun is no flip reason, but D is not handed more on top
            return SetVerdict(V_COLLECT, "NS:S49 X-SUM S=%d + %d granted on D not started > X=%d: waits for D to begin it "
                              "(R6) or finish it, no flip (R7)" % (s, gu, x), 0.0, s, hx)
        return SetVerdict(V_D_AFTER_TIMER, why, left_s, s, hx)

    if not d_decoding:
        if big or s >= x:
            return _flip("R8 no decode on D: decided now, S=%d >= X=%d or a big member" % (s, x))
        # R8: decided NOW -- never the S49 hold (a grant D has not begun is not in S: R1, R6 user 18.10.; kurz: R2/R6k)
        return SetVerdict(V_D_AFTER_TIMER, "R8 no decode on D: decided now, S=%d < X=%d" % (s, x), 0.0, s, hx)
    if big:
        if len(members) >= 2:
            return _flip("R4 a big member and company (n=%d, S=%d, new=%d)" % (len(members), s, new))
        if left <= 0.0:
            return _flip("R4 T_halb over, big member alone (S=%d)" % s)
        return SetVerdict(V_COLLECT, "R4 big member waits %.1f s for company" % left, left, s, hx)
    if s >= x:
        return _flip("R3 S=%d >= X=%d inside T_halb (new=%d)" % (s, x, new))
    if left <= 0.0:
        return _d_verdict("R3 T_halb over, S=%d < X=%d" % (s, x), 0.0)
    return SetVerdict(V_COLLECT, "R3 S=%d < X=%d, %.1f s left" % (s, x, left), left, s, hx)


@dataclass(frozen=True)
class SetDecision:
    """EZ-S2a ``decide(kind="set")``: the verdict plus what it was decided on (rule, per-member class, who counted in S,
    who was left out as kurz)."""
    verdict: SetVerdict
    rule: str
    classes: Tuple[str, ...]
    in_set: Tuple[str, ...]
    kurz_left_out: Tuple[str, ...]


def decide(kind: str, **kw) -> "SetDecision":
    """ONE decision site of the scheduler (plan B2). S2a: ``kind="set"`` -- :func:`set_verdict` on the members that BELONG to
    the set. R2: a kurz member (own u < halbX at ITS X, not p_only) never counts in S and never makes a flip (it is left
    out and named); R1/A4: S is the sum of the members' OWN u. ``kind`` other than ``"set"`` is S2b."""
    if kind in ("admit", "arrival"):
        return decide_arrival(kind, **kw)
    if kind != "set":
        raise ValueError("decide(kind=%r): kinds are 'set' (S2a), 'admit' and 'arrival' (S2b)" % (kind,))
    members = list(kw.pop("members"))
    half_frac = kw["half_frac"]
    xc = int(kw.get("class_x_tokens") or kw["x_tokens"])
    keep, out, classes = [], [], []
    for m in members:
        c = classify_arrival(m.own_uncached or m.uncached, int(m.x_own) or xc, half_frac, m.p_only)
        if c == CLASS_KURZ:
            out.append(m.rid)
            continue
        keep.append(m)
        classes.append(c)
    sv = set_verdict(keep, **kw)
    return SetDecision(sv, _rule_tag(sv.why, sv.verdict), tuple(classes), tuple(m.rid for m in keep), tuple(out))


# ----------------------------------------------------------------------------------------------------------- S2b: arrival
ACT_D_NOW = "D_NOW"
ACT_RIDE_P = "RIDE_P"
ACT_WAIT_SEAT = "WAIT_SEAT"
ACT_COLLECT = "COLLECT"
ACT_P_CAND = "P_CAND"
ACT_MEASURE = "MEASURE_ON_D"


@dataclass(frozen=True)
class ArrivalDecision:
    """EZ-S2b ``decide(kind="admit"|"arrival")``: what ONE arriving request does, decided once at its arrival."""
    action: str
    rule: str
    cls: str
    why: str
    waits: bool          # True = the request waits for something (a seat, a set verdict); False = it goes now


def decide_arrival(kind: str, *, uncached: Optional[int], x_own: int, half_frac: float, p_only: bool = False,
                   d_open: bool = True, flip_decided: bool = False, p_phase: bool = False,
                   exact: bool = True) -> "ArrivalDecision":
    """ONE decision at the arrival (SOLL R1a, R2, R2a, R4, R5, R7, R9; plan B2/S2b). Pure.

    * no exact number (``uncached`` None or ``exact`` False) and not p_only: ``MEASURE_ON_D`` (A1/R.1) -- never a class
      from an estimate;
    * p_only (R9) or u >= X_own (R4): a P candidate, the set decides the flip (``kind="arrival"``);
    * halb: a set member (R3);
    * kurz (u < halbX, R2): D at once while D's door is open. A flip that is DECIDED (D closing for it, or a P phase) is
      the one thing a kurz request rides (R1a: it is prefilled on P with the batch); a flip that is merely foreseeable
      (a big request waits its T_halb, R4) is no reason -- D takes it (R2). Door closed without a decided flip: it waits
      for its seat (R5), never for P.
    ``kind="admit"`` answers the same for the seat path (the caller knows the door and the flip state at that moment);
    both kinds share the rule table so no site can disagree."""
    if p_only:
        return ArrivalDecision(ACT_P_CAND, "R9", CLASS_P_ONLY, "p_only (vision / only-P) counts as big", False)
    if uncached is None or not exact:
        return ArrivalDecision(ACT_MEASURE, "NS:MEASURE", "unknown", "no exact number: D measures (halfX barrier)", True)
    cls = classify_arrival(int(uncached), int(x_own), half_frac, False)
    if cls == CLASS_GROSS:
        return ArrivalDecision(ACT_P_CAND, "R4", cls, "u=%d >= X=%d: P candidate, waits T_halb for company" % (uncached, x_own),
                               True)
    if cls == CLASS_HALB:
        return ArrivalDecision(ACT_COLLECT, "R3", cls, "halb u=%d: set member" % uncached, True)
    if p_phase or flip_decided:
        return ArrivalDecision(ACT_RIDE_P, "R1a", cls, "kurz u=%d, a flip is decided (%s): rides it, never a trigger (R2)"
                               % (uncached, "P phase" if p_phase else "D closing"), False)
    if d_open:
        return ArrivalDecision(ACT_D_NOW, "R2", cls, "kurz u=%d: D at once, no wait" % uncached, False)
    return ArrivalDecision(ACT_WAIT_SEAT, "R5", cls, "kurz u=%d, D cannot take it now and no flip is decided: waits for its "
                           "seat, never P" % uncached, True)


def _rule_tag(why: str, verdict: str) -> str:
    import re
    if (why or "").startswith("NS:S49"):
        return "NS:S49"
    m = re.search(r"\b(R[1-9]a?|R1[01]|R2a)\b", why or "")
    return m.group(1) if m else ("R2" if verdict == V_D_NOW else "R?")


def shared_prefix_discount(entries: Sequence[Tuple[str, int, int, float, str]],
                           common: Callable[[str, str], int]) -> Dict[str, int]:
    """H16V F1 (SOLL R1: S is the SUM of what P/D must NEW prefill): ``entries`` = ``(rid, uncached, depth, t_arrive,
    session)``; returns ``{rid: uncached'}`` where a member of a session counts only what its EARLIER session members
    do not already bring. Member B's uncached tokens are the positions ``[depth_B, N_B)``; an earlier member A of the
    same session prefills ``[depth_A, N_A)``; the ids both prompts share (``common(A, B)``) cover
    ``[max(depth_A, depth_B), common)`` of B -- computed once, not twice. The earliest member keeps its own figure;
    several earlier members discount by the deepest single overlap, ``max_A (common(A, B) - max(depth_A, depth_B))``:
    the covered ranges of different A's start at different positions (``max(depth_A, depth_B)``), so the max of the
    overlaps is NOT their union; it is at most the union, i.e. the discount never exceeds what the earlier members
    really bring (the error is on the safe side: S too large, never too small). A session unknown here ("") or
    ``common`` failing (-1/0) discounts nothing.

    H16V A5 (review): the figure is what P prefills (P matches the shared prefix once, P-log 913/912). It is NOT
    assumed for D: D's in-batch prefix dedupe (``schedule_policy._compute_prefix_matches``) only runs for the
    cache-aware policies and a cached match <= 32 tokens, so a D set after T_halb is priced at the members' own
    figures (``set_verdict`` sums ``SetMember.own_uncached``); the discount rides the set to P only (``S_P``)."""
    out: Dict[str, int] = {}
    seen: List[Tuple[str, int, str]] = []
    for rid, u, depth, _t, sess in sorted(entries, key=lambda e: e[3]):
        u = max(0, int(u))
        best = 0
        if sess:
            for arid, adepth, asess in seen:
                if asess != sess:
                    continue
                try:
                    c = int(common(arid, rid))
                except Exception:  # noqa: BLE001 -- a missing prompt discounts nothing
                    c = 0
                best = max(best, c - max(int(adepth), int(depth)))
        out[rid] = max(0, u - max(0, best))
        seen.append((rid, int(depth), sess))
    return out


def running_overlap(runs: Sequence[Tuple[str, int, int, str]], members: Sequence[Tuple[str, int, str]],
                    common: Callable[[str, str], int]) -> int:
    """SOLL R1u (user decision 20.10.): how many of the running D prefills' REST tokens a member of the set brings too (counted
    ONCE in S). ``runs`` = ``(rid, depth_A, rest, session)`` of a running prefill A: its uncached positions are
    ``[depth_A, depth_A + rest)`` (depth_A = N_A - rest); ``members`` = ``(rid, depth_B, session)``. A member B of the same
    session shares ids ``[0, common(A, B))`` with A, so the piece ``[max(depth_A, depth_B), min(common, N_A))`` is computed
    for both: the UNION of those pieces over the members is taken out of A's rest (members already discount one another,
    ``shared_prefix_discount``). Exact from ids and depths; ``common`` failing / a session unknown discounts nothing."""
    total = 0
    _run_sess = {(str(r[0]), r[3]) for r in runs}
    earlier: List[Tuple[str, int, int]] = []        # (rid, depth, rest) of the runs counted before (rid order: deterministic)
    for _arid, adepth, rest, asess in sorted(runs, key=lambda r: str(r[0])):
        rest = max(0, int(rest))
        if rest <= 0 or not asess:
            continue
        a_end = int(adepth) + rest
        spans: List[Tuple[int, int]] = []
        for brid, bdepth, bsess in members:
            if bsess != asess:
                continue
            try:
                c = int(common(_arid, brid))
            except Exception:  # noqa: BLE001
                c = 0
            lo, hi = max(int(adepth), int(bdepth)), min(c, a_end)
            if hi > lo:
                spans.append((lo, hi))
        # R1u, run against run: an EARLIER counted run of the same session brings the shared piece already -- its span joins
        # THIS run's spans, so a piece shared by run, run and member is taken out of this rest once, not twice
        for orid, odepth, orest in earlier:
            if (orid, asess) not in _run_sess:
                continue
            try:
                c = int(common(orid, _arid))
            except Exception:  # noqa: BLE001
                c = 0
            lo, hi = max(int(adepth), int(odepth)), min(c, a_end, int(odepth) + int(orest))
            if hi > lo:
                spans.append((lo, hi))
        earlier.append((_arid, int(adepth), rest))
        cur_lo = cur_hi = None
        covered = 0
        for lo, hi in sorted(spans):
            if cur_hi is None or lo > cur_hi:
                if cur_hi is not None:
                    covered += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
            else:
                cur_hi = max(cur_hi, hi)
        if cur_hi is not None:
            covered += cur_hi - cur_lo
        total += min(rest, covered)
    return total


def _nearest_rank(vals: Sequence[float], q: float) -> float:
    v = sorted(float(x) for x in vals)
    return v[min(len(v) - 1, max(0, int(-(-q * len(v) // 1)) - 1))]


#: R10: how far T_halb may stand from half of the measured p90 sum before the instrument says so.
HALF_FLIP_DEV_PCT = 30.0


def half_flip_check(t_half_s: float, dp_s: Sequence[float], pd_s: Sequence[float]) -> Optional[dict]:
    """R10 (SOLL v2): ``T_halb`` against the half of the measured user flip times -- p50/p90 per direction
    (PDFLIP-FLIP-USERZEIT, seconds). Reports only, nothing is derived from it. ``None`` until both directions have
    a sample. ``half_p90_s`` = (p90(D->P) + p90(P->D)) / 2; ``verdict`` is ``abweichung`` when T_halb differs from
    it by more than :data:`HALF_FLIP_DEV_PCT` percent."""
    if not dp_s or not pd_s:
        return None
    p90_dp, p90_pd = _nearest_rank(dp_s, 0.9), _nearest_rank(pd_s, 0.9)
    half = (p90_dp + p90_pd) / 2.0
    dev = abs(float(t_half_s) - half) / half * 100.0 if half > 0 else 0.0
    return {"t_half_s": float(t_half_s), "p50_dp_s": _nearest_rank(dp_s, 0.5), "p50_pd_s": _nearest_rank(pd_s, 0.5),
            "p90_dp_s": p90_dp, "p90_pd_s": p90_pd, "half_p90_s": half, "dev_pct": dev,
            "verdict": "abweichung" if dev > HALF_FLIP_DEV_PCT else "ok"}


#: PARK-COLLECT-WINDOW: the measured-record sidecar entry of one flip round
#: trip. The sidecar's other readers filter on their own fields
#: (``rss_shmem_gib``/``flip_ratchet_gib``, ``kind == "pd_free0"``) and do
#: not see it.
PARK_ROUND_TRIP_KIND = "park_round_trip"
PARK_ROUND_TRIP_GROUP = "PARK_RT"
#: The first round trips of a boot that are appended (the sidecar is
#: append-only across every boot of the rig; the newest one seeds the next).
PARK_ROUND_TRIP_RECORDS = 3


def warm_resume_ms(resume_log: Sequence[float], window: int = 5) -> Optional[float]:
    """The median wake -> first decoded chunk of the last ``window`` WARM D
    phases, or ``None`` before there is one. The boot's first resume is never
    warm (first park, JIT, pinning -- the same exclusion as H34b's first
    flip), and one slow resume must not move the price: the caller charges it
    once per running stream, so a single cold 10 s resume at bs6 would hold D
    for 60 s (27B review 29.09.)."""
    warm = [float(ms) for ms in list(resume_log)[1:]][-max(1, int(window)):]
    return statistics.median(warm) if warm else None


def park_round_trip_s(dp_ms: float, pd_ms: float, resume_ms: Optional[float]) -> Optional[float]:
    """The ski-rental price of one flip round trip in seconds: K7's D->P and
    P->D prices plus D's wake -> first decoded chunk; ``None`` while any of
    the three is unmeasured (K7 prices 0 before its first flip)."""
    if resume_ms is None or float(dp_ms) <= 0.0 or float(pd_ms) <= 0.0:
        return None
    return (float(dp_ms) + float(pd_ms) + float(resume_ms)) / 1000.0


def park_round_trip_record(*, form_key: str, dp_ms: float, pd_ms: float, resume_ms: float,
                           boot_tag: str, commit: Optional[str], at: str) -> dict:
    """The sidecar entry of one measured round trip, keyed by ``form_key``
    (checkpoint x form, see the front's ``_park_form_key``) so a qwen27b
    boot never reads a nextflash round trip and vice versa."""
    return {
        "kind": PARK_ROUND_TRIP_KIND, "group": PARK_ROUND_TRIP_GROUP, "form_key": str(form_key),
        "round_trip_s": park_round_trip_s(dp_ms, pd_ms, resume_ms),
        "dp_ms": float(dp_ms), "pd_ms": float(pd_ms), "resume_ms": float(resume_ms),
        "boot_tag": str(boot_tag), "commit": commit, "at": str(at),
    }


def newest_park_round_trip(samples: Iterable[dict], form_key: str) -> Optional[dict]:
    """The newest round-trip entry of ``form_key`` among ``samples``, or ``None``."""
    rows = [e for e in samples
            if isinstance(e, dict) and e.get("kind") == PARK_ROUND_TRIP_KIND
            and e.get("form_key") == form_key and isinstance(e.get("round_trip_s"), (int, float))
            and float(e["round_trip_s"]) > 0.0]
    return max(rows, key=lambda e: str(e.get("at", "")), default=None)


def read_park_round_trip(path: str, form_key: str) -> Optional[dict]:
    """:func:`newest_park_round_trip` of the sidecar at ``path``; a missing or
    malformed sidecar is an ABSENCE (``None``), never a zero."""
    if not path or not form_key:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    samples = data.get("samples") if isinstance(data, dict) else None
    return newest_park_round_trip(samples if isinstance(samples, list) else [], form_key)


# ---------------------------------------------------------------------------
# X-COST-LINE (29.09., third part of the ski-rental decision): X from D's
# measured cost line, priced by the measured round trip, amortised over the
# requests one P phase carries.
#
# THE MODEL. D prefilling ``n`` uncached tokens behind a cached prefix of
# ``p`` tokens costs ``a + b*n + c*n*p`` ms (one forward's fixed cost -- NF's
# expert stream, H118 -- plus a per-token cost that grows with the prefix the
# attention reads). Routing it through P costs its share of one round trip
# plus ``n / r_P``. Break-even:
#
#     a + (b + c*p) * n = price / k + 1000 * n / r_P
#     X* = (1000 * price / k - a) / (b + c*p - 1000 / r_P)
#
# ``price`` is the ski instrument's warm round trip (D->P + P->D + wake ->
# first decoded chunk), the one price source of the collect window; ``k`` the
# mean number of requests a P phase prefills (FLIP-ECONOMICS "amortising
# once over the backlog", 27B review 29.09.). Measured inputs (D logs
# 09290827 NF / 09290020 27B, TP0 lines with n >= 64): NF a=1900 ms,
# b=1.352, c=0.00088 ms/tok per k prefix; 27B a=227, b=0.611, c=0.00093.
# ---------------------------------------------------------------------------

#: The measured-record sidecar entry of one fitted D cost line (plus the
#: amortisation k and the prefix depth it was solved at).
X_COST_LINE_KIND = "x_cost_line"
X_COST_LINE_GROUP = "X_COST"
#: The first fitted lines of a boot that are appended (the newest seeds the next).
X_COST_LINE_RECORDS = 3


def _det3(m: Sequence[Sequence[float]]) -> float:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


#: A fit needs this many forwards of at least FIT_BIG_TOKENS (27B review of
#: 800bb82ac6): the break-even sits at thousands of tokens, and a window of
#: small extends alone moved P's slope 0.039 <-> 0.247 ms/tok on 27B 09290020,
#: X between 8.6k and 11.6k.
FIT_BIG_TOKENS = 1024


def x_step_limited(x_new: float, x_prev: int, max_step: float) -> Tuple[int, bool]:
    """Hysteresis of the live X (27B review of 800bb82ac6): one re-solve moves
    X by at most ``max_step`` of the previous value, so a noisy fit cannot
    flip the D/P route back and forth. Returns ``(x, limited)``."""
    if max_step <= 0.0 or x_prev <= 0:
        return int(x_new), False
    lo, hi = x_prev * (1.0 - max_step), x_prev * (1.0 + max_step)
    x = min(max(float(x_new), lo), hi)
    return int(x), x != float(x_new)


def fit_cost_line(rows: Iterable[Tuple[int, int, float]], *, min_tokens: int,
                  min_samples: int, min_spread: float = 4.0,
                  min_big: int = 8) -> Tuple[Optional[dict], str]:
    """Least squares of ``ms = a + b*n + c*n*p_k`` over D's prefill forwards
    ``(n, prefix_tokens, ms)`` with ``n >= min_tokens`` (``p_k`` = prefix in
    thousands). Returns ``(line, why)``; ``line`` is ``None`` with a named
    ``why`` when the rows cannot carry a line: too few, no spread in ``n``
    (``n_hi / n_lo`` below ``min_spread`` or fewer than ``min_big`` forwards
    of at least :data:`FIT_BIG_TOKENS` -- the slope would be the small
    extends' noise), or a non-positive slope.

    The depth term falls back to 0 (``why`` says so) when the prefixes do not
    span enough to separate it from ``b`` or it comes out negative -- a
    per-token cost that FALLS with the prefix read is not a measurement of
    attention, it is the fit using ``c`` to absorb noise."""
    pts = [(float(n), float(p) / 1000.0, float(ms)) for n, p, ms in rows
           if int(n) >= int(min_tokens) and float(ms) > 0.0]
    if len(pts) < max(3, int(min_samples)):
        return None, f"samples={len(pts)}<{max(3, int(min_samples))}"
    n_lo = min(x[0] for x in pts)
    n_hi = max(x[0] for x in pts)
    n_big = sum(1 for x in pts if x[0] >= FIT_BIG_TOKENS)
    if n_hi < float(min_spread) * n_lo or n_big < int(min_big):
        return None, f"no-spread(n {int(n_lo)}..{int(n_hi)}, {n_big} >= {FIT_BIG_TOKENS})"
    why = "3p"
    a = b = c = None
    xs = [(1.0, n, n * pk) for n, pk, _ in pts]
    ys = [ms for _, _, ms in pts]
    A = [[sum(x[i] * x[j] for x in xs) for j in range(3)] for i in range(3)]
    B = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(3)]
    d = _det3(A)
    scale = abs(A[0][0] * A[1][1] * A[2][2]) or 1.0
    if abs(d) > 1e-9 * scale:
        sol = []
        for col in range(3):
            m = [row[:] for row in A]
            for i in range(3):
                m[i][col] = B[i]
            sol.append(_det3(m) / d)
        a, b, c = sol
        if c < 0.0:
            a = None
            why = "2p(depth-term<0)"
    else:
        why = "2p(no-depth-spread)"
    if a is None:
        mn = statistics.mean(x[0] for x in pts)
        my = statistics.mean(ys)
        sxx = sum((x[0] - mn) ** 2 for x in pts)
        b = sum((x[0] - mn) * (y - my) for x, y in zip(pts, ys)) / sxx
        a, c = my - b * mn, 0.0
    if not b > 0.0:
        return None, f"slope<=0(b={b:.4f})"
    return {"a_ms": max(0.0, a), "b_ms": b, "c_ms": c, "n": len(pts),
            "n_lo": int(n_lo), "n_hi": int(n_hi),
            "prefix_med": int(statistics.median(x[1] for x in pts) * 1000.0)}, why


def solve_x_cost_line(*, price_s: float, k: float, line: dict, r_p: Optional[float],
                      prefix_tokens: float,
                      line_p: Optional[dict] = None) -> Tuple[Optional[float], str]:
    """``X*`` of the model above, in tokens (the caller clamps it to its
    measured floor and the ceiling), or ``(None, why)`` without a price or an
    r_P. When D's marginal token at this depth is not dearer than P's there
    is no crossing: the round trip never pays back per token, so D is the
    cheaper route for EVERY size while its fixed cost stays under the price
    share (``inf``, the ceiling decides) and P for every size otherwise
    (0, the floor decides). A price share below D's fixed cost gives 0 too.

    ``line_p`` (P's own cost line on its bottleneck stage PP0, same
    instrument) replaces the drain rate ``r_p`` when given: P's side is then
    ``a_P + (b_P + c_P*p) * n`` -- compute-honest, where the drain rate also
    counts P's waits (27B 09290020: drain 1437 tok/s, PP0 9180)."""
    if not price_s > 0.0:
        return None, "no-price"
    pk = float(prefix_tokens) / 1000.0
    per_req_ms = 1000.0 * float(price_s) / max(1.0, float(k))
    if line_p is not None:
        p_fixed = float(line_p["a_ms"])
        p_marg = float(line_p["b_ms"]) + float(line_p["c_ms"]) * pk
    elif r_p is not None and r_p > 0.0:
        p_fixed, p_marg = 0.0, 1000.0 / float(r_p)
    else:
        return None, "no-r_p"
    marg_ms = float(line["b_ms"]) + float(line["c_ms"]) * pk
    denom = marg_ms - p_marg
    num = per_req_ms + p_fixed - float(line["a_ms"])
    if denom <= 0.0:
        side = "d-never-dearer" if num > 0.0 else "p-always-cheaper"
        return (float("inf") if num > 0.0 else 0.0), (
            f"{side}(d={marg_ms:.3f}ms/tok<=p={p_marg:.3f})")
    return max(0.0, num / denom), "ok"


def x_cost_line_record(*, form_key: str, line: dict, k: float, boot_tag: str,
                       commit: Optional[str], at: str, side: str = "D") -> dict:
    """The sidecar entry of one fitted cost line of group ``side`` (D, or P's
    PP0), keyed by ``form_key`` (checkpoint x form) -- a qwen27b boot never
    reads a nextflash line, and a P line never seeds D's."""
    return {
        "kind": X_COST_LINE_KIND, "group": X_COST_LINE_GROUP, "form_key": str(form_key),
        "side": str(side),
        "a_ms": float(line["a_ms"]), "b_ms": float(line["b_ms"]), "c_ms": float(line["c_ms"]),
        "n": int(line["n"]), "n_lo": int(line["n_lo"]), "n_hi": int(line["n_hi"]),
        "prefix_med": int(line.get("prefix_med", 0)), "k": float(k),
        "boot_tag": str(boot_tag), "commit": commit, "at": str(at),
    }


def newest_x_cost_line(samples: Iterable[dict], form_key: str,
                       side: str = "D") -> Optional[dict]:
    """The newest cost-line entry of ``form_key`` and group ``side`` among
    ``samples``, or ``None`` (an entry without ``side`` is D's)."""
    rows = [e for e in samples
            if isinstance(e, dict) and e.get("kind") == X_COST_LINE_KIND
            and e.get("form_key") == form_key and e.get("side", "D") == side
            and isinstance(e.get("b_ms"), (int, float)) and float(e["b_ms"]) > 0.0]
    return max(rows, key=lambda e: str(e.get("at", "")), default=None)


def read_x_cost_line(path: str, form_key: str, side: str = "D") -> Optional[dict]:
    """:func:`newest_x_cost_line` of the sidecar at ``path``; a missing or
    malformed sidecar is an ABSENCE (``None``), never a zero."""
    if not path or not form_key:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    samples = data.get("samples") if isinstance(data, dict) else None
    return newest_x_cost_line(samples if isinstance(samples, list) else [], form_key, side)


def park_verdict(status: int, text: str) -> Tuple[str, List[str], str]:
    """Read D's answer to :data:`PARK_PATH`.

    Returns ``(verdict, rids, why)``: ``parked`` with the rids D parked (an
    empty list is a valid answer: nothing was running), ``unsupported`` for
    an old D (:data:`PARK_UNSUPPORTED_STATUSES`), ``failed`` for anything
    else, including a 200 whose body is not ``{"parked": [rid, ...]}`` -- a
    malformed answer must never be read as "nothing parked", because the
    front would then flip on requests it believes D is no longer running.

    H91c2: the rids include D's ``held`` list -- requests that were only
    QUEUED on D when the park came (an X-route request behind the H95c seat
    cap n, a newcomer behind a pressure-park barrier). Part B keeps them in
    the same park list (``pdflip_d_parked``), holds them over the sleep and
    resumes them after the parked ones, so for the front they are parked
    too. Read as running, they held the D->P drain for its whole window
    (120 s) and W1b then aborted them by rid. A malformed ``held`` is a
    failed park for the same reason as a malformed ``parked``.
    """
    if status in PARK_UNSUPPORTED_STATUSES:
        return PARK_UNSUPPORTED, [], f"http {status}"
    if status != 200:
        return PARK_FAILED, [], f"http {status}: {str(text)[:200]}"
    try:
        js = json.loads(text)
    except Exception as e:  # noqa: BLE001
        return PARK_FAILED, [], f"http 200 with an unreadable body ({type(e).__name__})"
    rids = js.get("parked") if isinstance(js, dict) else None
    if not isinstance(rids, list) or not all(isinstance(r, str) for r in rids):
        return PARK_FAILED, [], f"http 200 without a parked=[rid, ...] list: {str(text)[:200]}"
    held = js.get("held", [])
    if not isinstance(held, list) or not all(isinstance(r, str) for r in held):
        return PARK_FAILED, [], f"http 200 with a malformed held list: {str(text)[:200]}"
    return PARK_PARKED, list(rids) + [r for r in held if r not in rids], ""


def resolve_front_defaults(args, standard_form: bool) -> None:
    """UNIFY (operator 26.09.): the four part-C flags a launcher did not write
    (``None``) take the H91 defaults under the NF standard form
    (FLLIPER_PDFLIP_STANDARD_FORM, profile field ``standard_form``) and 0 -- the
    rule off, the pre-H91 front -- otherwise (qwen27b). A written value wins."""
    for name, default in (("p_phase_max_requests", P_PHASE_MAX_REQUESTS_DEFAULT),
                          ("p_pool_tokens", P_POOL_TOKENS_DEFAULT),
                          ("d_wait_bound_s", D_WAIT_BOUND_S_DEFAULT),
                          ("p_leg1_stall_s", P_LEG1_STALL_S_DEFAULT)):
        if getattr(args, name, None) is None:
            setattr(args, name, default if standard_form else type(default)(0))


def park_late_hold(status: int, text: str) -> bool:
    """H91c3-2: did D's park answer promise to hold every request that reaches
    it after the park (``"late_hold": true``)? Only then are the front's
    hand-offs still in flight to D's scheduler (in ``D.outstanding``, in
    neither of D's lists) parked for the front too. Anything else -- an old D,
    a failed park, a malformed body -- is False: the drain waits for them as
    before."""
    if status != 200:
        return False
    try:
        js = json.loads(text)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(js, dict) and js.get("late_hold") is True


def d_phase_seats(handoff_n: int, parked_n: int, d_bs: int) -> int:
    """H91c3-3: the D phase's seat count n the wake of D fixes (H95) -- the
    very function D applies to the same two integers
    (``d_seats.phase_seats``), with the front's ``--d-bs`` as the cap (the
    launcher checks it against D's --max-running-requests)."""
    # Module-level import (below): the front calls this at the P->D wake, on
    # its event loop, and a first import there is the H78 loop stall.
    return _d_seats.phase_seats(handoff_n, parked_n, cap=d_bs).n
