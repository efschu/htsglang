"""ARRIVAL-SEAT rule (user 29.09. ~19:40Z): D's decode seats stay filled in
parallel, arrivals do not wait behind a collect window or detour over P.

Wording of the rule: "wenn im aktuellen decode noch platz frei wäre für einen
weiteren parallelen decode mit ebendiesem kv bedarf des ankommenden requests,
dann soll das decode direkt pausiert werden (also halt an der chunk
größe/grenze falls es das bei decode gibt) und prefillt werden (entweder in D
größe oder mit flip in P)".

ONE decision site, in the front, while D is awake:

(a) a seat is free and the arrival's KV need (prompt + decode reserve) fits
    D's free KV: uncached <= X -> D prefills it at its next round boundary
    (the ordinary SHORT hand-off: D's scheduler interleaves the extend between
    decode rounds); uncached > X -> the flip to P is taken NOW (the running
    decodes park at the round boundary), no collect window, no dwell.
(b) no seat free or the KV does not fit: no flip, no P detour. The arrival
    waits for the next free seat, in arrival order (#244/#246).
(c) the H91 wait bound stays, as the #246 displacement: the oldest waiter past
    the bound parks the YOUNGEST running decode at its round boundary; the
    seat it frees is taken by (a).

With the rule on, PARK-COLLECT-WINDOW and PARK-WINDOW-GATE are inert (no case
is left for them: a free seat flips at once, no seat never flips); X-COST-LINE
only supplies X. Off (default until the first series): nothing here is read.

Pure functions only -- the front's wiring is in ``pdflip/front.py``
(``_arrival_seat_*``); every verdict is counted into state.json ``front``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

MARKER = "PDFLIP ARRIVAL-SEAT"

D_PREFILL = "d_prefill"
FLIP_NOW = "flip_now"
WAIT_SEAT = "wait_seat"

#: the park reason D sees for (c)
REASON_YOUNGEST = "arrival-seat-youngest"
#: the park reason D sees for the KV displacement (NF-STAU-KV, #246)
REASON_KV = "arrival-seat-kv"
#: one KV displacement per this many seconds: longer than the front's KV
#: reading cache (1 s), so the next verdict reads D after the park landed
KV_PARK_COOLDOWN_S = 1.5
#: AGE PLAN marker (one line per displacement and per plan verdict change)
AGE_MARKER = "PDFLIP SEAT-AGE-PLAN"

#: counters the front keeps (all published under state.json front.arrival_seat)
COUNTERS = (
    "arrival_seat_d_prefill", "arrival_seat_flip_now", "arrival_seat_wait_seat",
    "arrival_seat_wait_kv", "arrival_seat_youngest_park", "arrival_seat_youngest_park_refused",
    # NF-STAU (29.09.): KV backfill past a head that does not fit, and the
    # two TTFT clocks (IPC, no log parsing): arrival -> verdict and arrival ->
    # first token, count/sum/max ms
    "arrival_seat_backfill",
    "arrival_seat_verdict_n", "arrival_seat_verdict_ms_sum", "arrival_seat_verdict_ms_max",
    "arrival_seat_ttft_n", "arrival_seat_ttft_ms_sum", "arrival_seat_ttft_ms_max",
    # NF-STAU-KV (30.09., Klasse J): the head whose KV does not fit parks the
    # youngest running decode that ARRIVED after it (user rule #246 "Ältester
    # rückt nach und verdrängt Jüngere"), and the KV need's two honest terms
    "arrival_seat_kv_displace", "arrival_seat_kv_displace_refused",
    "arrival_seat_kv_shared_tokens", "arrival_seat_kv_decode_clipped",
    # AGE PLAN (30.09. ~14:45Z): heads handed to D although they do not fit
    # now (D's SEAT-AGE displaces younger ones for them), heads the plan found
    # blocked by OLDER running requests (-> backfill)
    "arrival_seat_age_to_d", "arrival_seat_age_blocked_by_elders",
    # MIN-DWELL (30.09., y5c): flip_now verdicts held because the decodes
    # resumed this D phase had not yet decoded one flip round trip
    "arrival_seat_min_dwell_hold",
    # KV READ BUDGET (01.10., NF D->P flip): KV tests decided on the last
    # reading because D's /server_info did not answer within the budget (D
    # inside a pass) -- the refresh lands for the next tick
    "arrival_seat_kv_stale",
)


def kv_read_budget_s(env=None) -> float:
    """How long a KV test waits for a FRESH ``/server_info`` reading from D
    before it decides on the last one (FLLIPER_PDFLIP_ARRIVAL_KV_READ_BUDGET_S,
    default 0.05 s). D answers only at its scheduler pass boundary -- the same
    boundary the park RPC of a flip_now waits for -- so a blocking read
    serialised two boundaries in front of every D->P flip. Never negative."""
    if env is not None:
        raw = env.get("FLLIPER_PDFLIP_ARRIVAL_KV_READ_BUDGET_S")
        try:
            return max(0.0, float(raw)) if raw not in (None, "") else 0.05
        except (TypeError, ValueError):
            return 0.05
    from flliper.srt.environ import envs

    return max(0.0, float(envs.FLLIPER_PDFLIP_ARRIVAL_KV_READ_BUDGET_S.get()))


def enabled(env=None) -> bool:
    if env is not None:
        raw = str(env.get("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "") or "").strip().lower()
        return raw in ("1", "true", "yes", "on")
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE.get())


def default_reserve() -> int:
    from flliper.srt.environ import envs

    return int(envs.FLLIPER_PDFLIP_ARRIVAL_DECODE_RESERVE_TOKENS.get())


def decode_reserve(max_tokens: Optional[int], default_reserve_tokens: int) -> int:
    """The decode part of an arrival's KV need: its own ``max_tokens`` when the
    client set one, else the configured reserve. Never negative."""
    try:
        mt = int(max_tokens) if max_tokens is not None else 0
    except (TypeError, ValueError):
        mt = 0
    if mt > 0:
        return mt
    return max(0, int(default_reserve_tokens))


def max_tokens_of(payload) -> Optional[int]:
    """The client's decode budget from an OpenAI-shaped body (``max_tokens`` or
    ``max_completion_tokens``), else None."""
    if not isinstance(payload, dict):
        return None
    for k in ("max_tokens", "max_completion_tokens", "max_new_tokens"):
        v = payload.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    sp = payload.get("sampling_params")
    if isinstance(sp, dict) and sp.get("max_new_tokens") is not None:
        try:
            return int(sp["max_new_tokens"])
        except (TypeError, ValueError):
            return None
    return None


def default_decode_clip() -> int:
    """D's own decode reservation cap per request (upstream
    ``FLLIPER_CLIP_MAX_NEW_TOKENS_ESTIMATION``, ``schedule_policy.CLIP_MAX_NEW_TOKENS``)
    -- the fallback when D does not publish ``pdflip_kv.decode_clip``."""
    from flliper.srt.environ import envs

    return int(envs.FLLIPER_CLIP_MAX_NEW_TOKENS_ESTIMATION.get())


def d_decode_clip() -> int:
    """D side (``/server_info`` ``pdflip_kv.decode_clip``): the cap D's own
    admission (``PrefillAdder``) reserves per request for its decode."""
    from flliper.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS

    return int(CLIP_MAX_NEW_TOKENS)


def decode_part(max_tokens: Optional[int], default_reserve_tokens: int, clip: Optional[int]) -> int:
    """NF-STAU-KV (30.09., Klasse J): the decode term of an arrival's KV need,
    INCREMENTAL like D's own admission, never the client's worst case.

    y3u 00:42:24 pdflip-38-56: need=141854 = 77854 prompt + 64000 (Claude Code's
    ``max_tokens``) > free=131820 -> 13.6 s wait; it decoded 47 tokens. D
    itself reserves ``min(max_new_tokens, CLIP_MAX_NEW_TOKENS)`` per request
    (``PrefillAdder``) and grows past it elastically: a decode-pressure
    retraction parks the youngest with its span retained (retract_retain,
    ResumeBook resumes it when it fits). The front's fit test now asks what
    D's admission asks -- the same clip, read from D (``pdflip_kv.decode_clip``)."""
    r = decode_reserve(max_tokens, default_reserve_tokens)
    try:
        c = int(clip) if clip is not None else 0
    except (TypeError, ValueError):
        c = 0
    return min(r, c) if c > 0 else r


def shared_prefix_credit(est_prompt: int, est_uncached: Optional[int], common: Optional[int],
                         prev_running: bool) -> int:
    """NF-STAU-KV: the prefix an arrival shares with a request RUNNING on D
    (the same session's previous turn: front ``SESSION-PREFIX`` common) --
    those pages are locked by the running turn, already counted in D's used
    tokens, and the new turn maps them instead of allocating (radix match).
    Capped by the MEASURED cached-on-D span (``est_prompt - est_uncached``,
    presence: Mamba anchor depth). A finished or parked previous turn gives no
    credit: its pages are evictable, D's free reading already counts them."""
    if not prev_running or common is None or est_uncached is None:
        return 0
    try:
        present = max(0, int(est_prompt or 0) - int(est_uncached))
        return max(0, min(int(common), present))
    except (TypeError, ValueError):
        return 0


def kv_free(kv_reading: Optional[Mapping]) -> Optional[int]:
    """The free tokens :func:`kv_fits` tests against (the larger of D's stage
    reading and its ladder room), or None without a reading."""
    if not kv_reading:
        return None
    try:
        free = int(kv_reading.get("available", 0) or 0) + int(kv_reading.get("evictable", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        return None
    lad = ladder_free(kv_reading)
    return max(free, lad[0]) if lad is not None else free


def kv_displace_victim(head_arrival: Optional[float], running: Iterable[str],
                       arrival_of: Mapping[str, float], tokens_of: Mapping[str, int],
                       deficit: int) -> Optional[str]:
    """NF-STAU-KV (user rule #246, 27.09.: "strikte Ankunftsreihenfolge +
    KV-Backfill, Ältester rückt nach und verdrängt Jüngere"): the head's KV
    does not fit a free seat -> the running decode that ARRIVED last, and
    after the head, parks (SA (3)'s pressure shape, span retained) so the
    head gets its room now instead of after the 60 s bound (c).

    Only a younger one is ever a victim (an older running request has the
    right of way; then the head is blocked by its elders and KV backfill
    applies). Only when the younger ones' KV can cover the deficit at all --
    ``tokens_of`` is the front's LOWER bound (the prompt it priced), so a
    park that cannot make the head fit is never taken (no churn). A rid with
    no arrival stamp counts as old."""
    if head_arrival is None or int(deficit) <= 0:
        return None
    younger = [(float(arrival_of[r]), r) for r in running
               if r in arrival_of and float(arrival_of[r]) > float(head_arrival)]
    if not younger:
        return None
    if sum(max(0, int(tokens_of.get(r, 0) or 0)) for _, r in younger) < int(deficit):
        return None
    return max(younger)[1]


def age_plan_enabled(env=None) -> bool:
    """AGE PLAN switch (default on) -- acts only with the rule itself on."""
    if env is not None:
        raw = str(env.get("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", "") or "").strip().lower()
        return raw not in ("0", "false", "no", "off") and enabled(env)
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN.get()) and enabled()


def displace_plan(head_arrival: Optional[float], seat_is_free: bool, deficit: int,
                  running: Iterable[str], arrival_of: Mapping[str, float],
                  tokens_of: Mapping[str, int]) -> Optional[List[str]]:
    """AGE PLAN (Nutzer 30.09. ~14:40Z: "der ältere ist immer bevorzugt ...
    sobald keine ältesten mehr da sind die den kv für den mittleren
    blockieren, wird der mittlere auf die karten gezogen und verdrängt ggf.
    auch den jüngsten"; ~14:45Z: "natürlich muss ein jüngerer nur verdrängt
    werden, wenn der ältere nicht draufpasst. nicht pauschal").

    ``[]``: the head fits (a seat is free and ``deficit`` <= 0) -- nobody is
    displaced. ``None``: parking every running decode that ARRIVED after the
    head still gives it no seat or not its KV -- older running requests block
    it; nobody is displaced and younger ones that fit are backfilled. Else the
    FEWEST running decodes, youngest arrival first, all younger than the head,
    whose parking gives it a seat (one, when none is free) and covers the KV
    ``deficit``. ``tokens_of`` is the front's lower bound of a decode's KV (the
    prompt it priced). The front only ADMITS by it (a non-empty plan hands the
    head to D); the displacement itself is D's alone
    (``d_park_runtime.displace_for_age`` / ``victims_needed`` on D's real KV).
    A rid without an arrival stamp counts as old."""
    if head_arrival is None:
        return None
    need_seat = 0 if seat_is_free else 1
    deficit = max(0, int(deficit or 0))
    if need_seat == 0 and deficit == 0:
        return []
    younger = sorted(((float(arrival_of[r]), r) for r in running
                      if r in arrival_of and float(arrival_of[r]) > float(head_arrival)),
                     reverse=True)
    out: List[str] = []
    freed = 0
    for _t, r in younger:
        out.append(r)
        freed += max(0, int(tokens_of.get(r, 0) or 0))
        if len(out) >= need_seat and freed >= deficit:
            return out
    return None


def kv_need(est_prompt_tokens: int, reserve_tokens: int) -> int:
    """Prompt + decode reserve: the whole context lives in D's KV while it
    decodes (the cached prefix too -- a store hit is loaded onto the device)."""
    return max(0, int(est_prompt_tokens or 0)) + max(0, int(reserve_tokens or 0))


def ladder_free(kv_reading: Optional[Mapping]) -> Optional[Tuple[int, int, int]]:
    """``(free, ceiling, used)`` against D's KV LADDER, or None when D does not
    publish one. NF-STAU (29.09., y3m 22:03:44 "kv need=130563 > free=54016"
    while D's ladder reached 524288 with ~77k used): the mapped stage is not
    the limit -- D-MEM-SCHED grows it at the admission and the experts go to
    host RAM (the user's law: free VRAM belongs to the experts, KV displaces
    them when needed). The ceiling is the highest stage D may take this phase
    (``d_seat_vram.kv_ladder_reading``: ``ladder_ceiling``), ``used`` the
    global tokens its running requests hold."""
    if not kv_reading:
        return None
    try:
        ceiling = int(kv_reading.get("ladder_ceiling", 0) or 0)
        used = kv_reading.get("ladder_used")
        if ceiling <= 0 or used is None:
            return None
        used = max(0, int(used))
    except (TypeError, ValueError, AttributeError):
        return None
    return max(0, ceiling - used), ceiling, used


def kv_fits(need: int, kv_reading: Optional[Mapping]) -> Tuple[bool, str]:
    """``(fits, why)`` against D's reading ``{"available", "evictable"}`` (TP0's
    ``/server_info`` ``pdflip_kv``) -- and, when D publishes its KV ladder, against
    the ladder's ceiling (:func:`ladder_free`): the larger of the two decides.
    No reading (an old D, a failed read): the seat plan alone decides --
    H95c's phase seats are KV-derived already -- and the verdict says so
    (``kv=unread``)."""
    if not kv_reading:
        return True, "kv=unread"
    try:
        free = int(kv_reading.get("available", 0) or 0) + int(kv_reading.get("evictable", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        return True, "kv=unreadable"
    lad = ladder_free(kv_reading)
    tail = ""
    if lad is not None:
        tail = f" (ladder {lad[1]}-{lad[2]}={lad[0]}, stage free {free})"
        free = max(free, lad[0])
    if int(need) <= free:
        return True, f"kv need={int(need)} free={free}{tail}"
    return False, f"kv need={int(need)} > free={free}{tail}"


def min_dwell_enabled(env=None) -> bool:
    """MIN-DWELL switch (FLLIPER_PDFLIP_ARRIVAL_MIN_DWELL, default OFF since 02.10.:
    an arriving request is prefilled at once). With an explicit ``env`` an unset
    value reads as the default too."""
    if env is not None:
        raw = str(env.get("FLLIPER_PDFLIP_ARRIVAL_MIN_DWELL", "") or "").strip().lower()
        return raw in ("1", "true", "yes", "on")
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_ARRIVAL_MIN_DWELL.get())


def min_dwell_hold(resumed_t: Mapping[str, float], running: Iterable[str], now: float,
                   need_s: Optional[float]) -> Optional[Tuple[str, float]]:
    """MIN-DWELL (NF-Operator 30.09., ski rental): may a flip_now to P come
    now? ``resumed_t``: rid -> the time D resumed it in THIS D phase;
    ``running``: the decodes D runs now; ``need_s``: the measured flip round
    trip (None/<= 0 = unmeasured, never a constant -> no hold). Returns the
    resumed running decode with the SHORTEST dwell and that dwell when it is
    below ``need_s`` (hold), else None (the flip may come)."""
    if need_s is None or float(need_s) <= 0.0:
        return None
    worst = None
    for rid in running:
        t = resumed_t.get(rid)
        if t is None:
            continue
        dwell = max(0.0, float(now) - float(t))
        if worst is None or dwell < worst[1]:
            worst = (rid, dwell)
    if worst is None or worst[1] >= float(need_s):
        return None
    return worst


def backfill_allowed(head_wait_s: Optional[float], bound_s: float) -> bool:
    """#246 (user 17:50Z): strict arrival order PLUS KV backfill -- a later
    arrival that fits may take a free seat the head cannot use (its KV does
    not fit). The head does not starve: every tick asks the head FIRST, and
    once it waited ``bound_s`` backfill stops (the freed room stays for it)
    and the bound's displacement (c) runs instead."""
    return not bound_fired(head_wait_s, bound_s)


def note_ms(counters, key: str, ms: float) -> None:
    """One sample of an IPC clock ``key`` (``verdict`` / ``ttft``): count, sum
    and max in the front's counters (state.json ``front.arrival_seat``)."""
    v = max(0, int(round(float(ms))))
    counters[f"arrival_seat_{key}_n"] += 1
    counters[f"arrival_seat_{key}_ms_sum"] += v
    if v > int(counters.get(f"arrival_seat_{key}_ms_max", 0) or 0):
        counters[f"arrival_seat_{key}_ms_max"] = v


def seat_free(taken: int, seats: int) -> bool:
    """One more parallel decode fits the seat count (H95c phase seats n, else
    ``--d-bs``)."""
    return int(seats) > 0 and int(taken) < int(seats)


def verdict(seat_is_free: bool, fits: bool, uncached: int, x_tokens: int) -> str:
    """(a)/(b): the one decision. X comes from X-COST-LINE / resolve_x -- this
    rule never moves it."""
    if not (seat_is_free and fits):
        return WAIT_SEAT
    return D_PREFILL if int(uncached) <= int(x_tokens) else FLIP_NOW


def oldest_wait_s(t_arrivals: Iterable[float], t_awake: float, now: float) -> Optional[float]:
    """The D-phase wait of the oldest waiter (queue for P plus the arrivals
    waiting for a D seat): ``now - max(arrival, D phase start)``."""
    best = None
    for t in t_arrivals:
        w = float(now) - max(float(t), float(t_awake))
        if best is None or w > best:
            best = w
    return best


def youngest_running(admit_t: Mapping[str, float], running: Iterable[str]) -> Optional[str]:
    """(c): the running decode admitted LAST to a D seat (#244's order). A rid
    with no admission stamp (resumed before this boot's stamps) counts as the
    oldest -- it is never chosen over a stamped one."""
    best, best_t = None, None
    for rid in running:
        t = admit_t.get(rid)
        if t is None:
            continue
        if best_t is None or t > best_t:
            best, best_t = rid, t
    return best


def bound_fired(wait_s: Optional[float], bound_s: float) -> bool:
    return wait_s is not None and float(bound_s) > 0 and float(wait_s) >= float(bound_s)


def state_block(counters: Mapping[str, int], waiters: Dict[str, float], armed: bool) -> dict:
    """state.json ``front.arrival_seat``."""
    out = {"armed": bool(armed), "waiting_for_seat": len(waiters)}
    for k in COUNTERS:
        out[k[len("arrival_seat_"):]] = int(counters.get(k, 0) or 0)
    return out
