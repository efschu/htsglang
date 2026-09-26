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
from typing import Iterable, List, Optional, Sequence, Tuple

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


def park_body(epoch: int, bound_s: float) -> dict:
    """The park request: ``{"epoch": <int>, "reason": "wait-bound-60s"}``."""
    return {"epoch": int(epoch), "reason": park_reason(bound_s)}


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
