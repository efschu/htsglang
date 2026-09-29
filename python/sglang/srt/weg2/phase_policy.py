"""H91 part C: the Weg-2 front's PHASE POLICY (user design 25.09.2026).

The pure half of the policy, so every verdict can be read at a desk without a
front, a socket or a loop. :mod:`sglang.srt.weg2.front` owns the state and the
calls; this module owns the arithmetic and the wire shapes.

THE THREE RULES (user, 25.09., binding):

1. P PHASE -- at most ``--p-phase-max-requests`` (6) requests per P phase.
   How many of them are prefilled OVERLAPPING is planned against P's unified
   pool (``--p-pool-tokens``, 262k on NF) from the waiting requests'
   ``est_prompt``: the next request is dispatched only while the in-flight
   ones plus it fit the pool (one is always admitted, so a request larger than
   the pool is refused by P by name -- WEG2 P-INTAKE-TOO-LARGE -- instead of
   starving here). Finished prefills leave P's VRAM into the L2/L3 chain
   (part A), so the per-phase count is bounded by the cap, not by the pool.
   The P phase ends when the cap is dispatched or the queue is empty.
2. D PHASE -- D decodes EVERY request this P phase handed over before the
   front flips back; the count rides on the wake message to D
   (``handoff_n``, read by part B).
3. WAIT BOUND -- "wartegrenze 60 s, dann zurueck zu P": a request waiting for
   P for longer than ``--d-wait-bound-s`` (60) DURING the D phase makes the
   front park D's running decodes (``POST /weg2/park_running``, part B) and
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
from typing import Iterable, List, Optional, Sequence, Tuple

# H91c3-3: imported with this module (the front imports it before its loop),
# never lazily from `d_phase_seats` -- d_seats is stdlib-only at import time.
from sglang.srt.weg2 import d_seats as _d_seats

#: Rule 1: requests per P phase (user 25.09.: "bis zu 6").
P_PHASE_MAX_REQUESTS_DEFAULT = 6
#: Rule 1: P's unified KV pool in tokens (NF: 262k, memory `262K-PFLICHT`).
P_POOL_TOKENS_DEFAULT = 262144
#: Rule 3: the wait bound in seconds (user verbatim: "wartegrenze 60 s").
D_WAIT_BOUND_S_DEFAULT = 60.0
#: Rule 3: D's park endpoint (part B builds it).
PARK_PATH = "/weg2/park_running"
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
#: drain ends and the flip to D follows (the WEG2-INTAKE-STALL path).
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


def parked_lapsed(t_parked: float, now: float) -> bool:
    """Part B re-queues a parked request no sleep followed after
    :data:`PARK_REQUEUE_S`; from then on it is running on D again."""
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
# is idle, i.e. after D's decodes end -- weg2-28-62 (4172 > X_busy 4096 <= live X
# 7290) waited 45 s for the fairness bound and 65 s more for the drain (flip 67 s).
# ---------------------------------------------------------------------------

def needs_p(est_uncached: int, x_tokens: int, *, skip_leg1: bool = False,
            leg1_done: bool = False, p_only: bool = False, x_requeues: int = 0,
            x_deferred: bool = False) -> bool:
    """Does this queued request need P's prefill (the immediate park's
    trigger)? ``est_uncached > X`` (law 4: D never prefills above X), a
    P-only request (an image under transient vision, long by rule), or one D
    already refused as over X (W31, ``x_requeues``). Never a request whose
    leg 1 is done (it waits for D, not for P) or that skips leg 1 (route
    CARRIER-EXCEEDS: D prefills it once, a flip to P buys nothing)."""
    if skip_leg1 or leg1_done:
        return False
    if p_only or int(x_requeues or 0) > 0 or x_deferred:
        return True  # x_deferred: over X_busy, the X in force while D decodes (PK2)
    return int(est_uncached) > int(x_tokens)


def immediate_park_trigger(queue: Iterable, x_tokens: int):
    """The first queued request (head first) that :func:`needs_p`, or None.
    Reads ``Pending``'s own fields by name (getattr: partial test doubles)."""
    for p in queue:
        if needs_p(int(getattr(p, "est_uncached", 0) or 0), x_tokens,
                   skip_leg1=bool(getattr(p, "skip_leg1", False)),
                   leg1_done=bool(getattr(p, "leg1_done", False)),
                   p_only=bool(getattr(p, "p_only", False)),
                   x_requeues=int(getattr(p, "x_requeues", 0) or 0),
                   x_deferred=bool(getattr(p, "x_deferred", False))):
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
PARK_CYCLE_DWELL_ENV = "SGLANG_WEG2_PARK_CYCLE_DWELL"


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


PARK_DECODE_DWELL_ENV = "SGLANG_WEG2_PARK_DECODE_DWELL"


def park_decode_dwell_on(env=None) -> bool:
    """NF rc12z22 (boot ...dauer09281447, 14:52-14:58): 22 flips in 374 s, D
    decoding 1-3 s per 7-16 s D phase -- the resumed requests' store->device
    reload took 4-8 s after every wake, and the dwell (priced in FLIP time
    from the wake) fired the next park just as decode began: 335 completion
    tokens/min for every agent together. Default ON; "0" = the wake clock."""
    e = os.environ if env is None else env
    raw = (e.get(PARK_DECODE_DWELL_ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


PARK_DECODE_DUTY_ENV = "SGLANG_WEG2_PARK_DECODE_DUTY"


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


def park_collect_window(items: Sequence[Tuple[float, int]], now: float, t_awake: float,
                        n_running: int, price_s: float, threshold_tokens: int, *,
                        timer: bool = False, max_requests: int = 0, pool_tokens: int = 0,
                        wait_bound_s: float = 0.0) -> Tuple[bool, str, Optional[float], float]:
    """PARK-COLLECT-WINDOW: may the immediate park stop D's running decodes now?

    NF z30w-park 08:31-08:46: 21 immediate parks in 15 min, all fired by ONE
    over-X arrival (7 of them 4224-4761 uncached against X=4096), 37 parked
    streams, park -> resume 8.4 s median / 17.8 s p90. The user's model: D
    keeps decoding; once the pending P work passes the bound, it collects
    while D decodes on, and the flip comes when collecting has cost as much as
    the flip would -- or earlier when D's decodes end.

    SKI RENTAL (user decision 29.09.), both sides in request-seconds: the
    RENT is what the queued requests have waited since the window opened
    (their sum), the PRICE of the flip is ``price_s`` -- the caller passes one
    measured round trip per running stream, which is what a park stalls. One
    stream and one waiting request = "wait one round trip". ``timer`` (the
    fixed-x override) instead closes the window ``price_s`` after it opened.

    ``items`` are the queued requests that need P as ``(t_arrive, uncached)``.
    The window opens when their uncached sum first passes
    ``threshold_tokens`` (head first by arrival), never before this D phase
    woke. Fires on: nothing running (the park stops nobody), a hard cap --
    ``max_requests`` queued (P's phase cap), ``pool_tokens`` queued (P's
    pool), the oldest waited ``wait_bound_s`` -- or the rent reaching the
    price. Returns ``(fire, why, t_start, left_s)`` (``left_s`` at the current
    queue); pure -- the front owns the clocks."""
    rows = sorted((float(t), int(u)) for t, u in items)
    if not rows:
        return False, "empty", None, 0.0
    total, t_cross = 0, None
    for t, u in rows:
        total += u
        if total > int(threshold_tokens):
            t_cross = t
            break
    if t_cross is None:
        return False, "below-threshold", None, 0.0
    t_start = max(t_cross, float(t_awake))
    if timer:
        left = max(0.0, float(price_s) - (float(now) - t_start))
    else:
        rent = sum(max(0.0, float(now) - max(t, t_start)) for t, _ in rows)
        left = max(0.0, (float(price_s) - rent) / len(rows))
    if int(n_running) <= 0:
        return True, "d-idle", t_start, left
    if int(max_requests) > 0 and len(rows) >= int(max_requests):
        return True, "cap-requests", t_start, left
    if int(pool_tokens) > 0 and sum(u for _, u in rows) >= int(pool_tokens):
        return True, "cap-pool", t_start, left
    if float(wait_bound_s) > 0.0 and float(now) - rows[0][0] >= float(wait_bound_s):
        return True, "wait-bound", t_start, left
    if left <= 0.0:
        return True, "timer" if timer else "rent", t_start, 0.0
    return False, "collect", t_start, left


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
    the same park list (``weg2_d_parked``), holds them over the sleep and
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
    (SGLANG_WEG2_STANDARD_FORM, profile field ``standard_form``) and 0 -- the
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
