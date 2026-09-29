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

Pure functions only -- the front's wiring is in ``weg2/front.py``
(``_arrival_seat_*``); every verdict is counted into state.json ``front``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

MARKER = "WEG2 ARRIVAL-SEAT"

D_PREFILL = "d_prefill"
FLIP_NOW = "flip_now"
WAIT_SEAT = "wait_seat"

#: the park reason D sees for (c)
REASON_YOUNGEST = "arrival-seat-youngest"

#: counters the front keeps (all published under state.json front.arrival_seat)
COUNTERS = (
    "arrival_seat_d_prefill", "arrival_seat_flip_now", "arrival_seat_wait_seat",
    "arrival_seat_wait_kv", "arrival_seat_youngest_park", "arrival_seat_youngest_park_refused",
)


def enabled(env=None) -> bool:
    if env is not None:
        raw = str(env.get("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", "") or "").strip().lower()
        return raw in ("1", "true", "yes", "on")
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE.get())


def default_reserve() -> int:
    from sglang.srt.environ import envs

    return int(envs.SGLANG_WEG2_ARRIVAL_DECODE_RESERVE_TOKENS.get())


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


def kv_need(est_prompt_tokens: int, reserve_tokens: int) -> int:
    """Prompt + decode reserve: the whole context lives in D's KV while it
    decodes (the cached prefix too -- a store hit is loaded onto the device)."""
    return max(0, int(est_prompt_tokens or 0)) + max(0, int(reserve_tokens or 0))


def kv_fits(need: int, kv_reading: Optional[Mapping]) -> Tuple[bool, str]:
    """``(fits, why)`` against D's reading ``{"available", "evictable"}`` (TP0's
    ``/server_info`` ``weg2_kv``). No reading (an old D, a failed read): the
    seat plan alone decides -- H95c's phase seats are KV-derived already --
    and the verdict says so (``kv=unread``)."""
    if not kv_reading:
        return True, "kv=unread"
    try:
        free = int(kv_reading.get("available", 0) or 0) + int(kv_reading.get("evictable", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        return True, "kv=unreadable"
    if int(need) <= free:
        return True, f"kv need={int(need)} free={free}"
    return False, f"kv need={int(need)} > free={free}"


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
