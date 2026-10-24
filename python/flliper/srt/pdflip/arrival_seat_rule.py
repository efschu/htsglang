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

import os

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


def _dual_default_off(env) -> bool:
    """27B port 20.10. (user 20.10.: the dual layout stays as it was): the 27B line ran the rule OFF (code
    default False, no profile set it). The NF default ON reaches the flip form only; under the DUAL layout
    an UNSET switch stays off. The ranks carry FLLIPER_PDFLIP_DUAL_LAYOUT=1 (launcher), the front sets it
    for itself from --dual-layout (front.main), so every process of a dual boot reads the same default."""
    return str(env.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "") or "").strip() == "1"


def enabled(env=None) -> bool:
    src = os.environ if env is None else env
    raw = str(src.get("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_RULE", "") or "").strip().lower()
    if not raw:
        # unset/blank = the code default: ON on the flip form (NF 110), OFF under the 27B dual layout
        return not _dual_default_off(src)
    return raw not in ("0", "false", "no", "off")


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


def backup_wall(kv_reading: Optional[Mapping]) -> bool:
    """S94: D publishes its backup wall (``pdflip_kv.backup_wall``: payable < reported evictable). True only when D
    says so -- an old D without the field reads False (unchanged)."""
    try:
        return bool(kv_reading and kv_reading.get("backup_wall"))
    except AttributeError:
        return False


def payable_room(kv_reading: Optional[Mapping]) -> Optional[int]:
    """S94: under D's backup wall, the rows D can really give a new leg: ``available + payable_evictable`` -- D's own
    test (``d_wall_head.wall_unservable``, ``schedule_policy._h105d_cut_load_back_room``). None without the wall or
    without D's number (then nothing changes)."""
    if not backup_wall(kv_reading):
        return None
    try:
        if kv_reading.get("payable_evictable") is None:
            return None
        return max(0, int(kv_reading.get("available", 0) or 0)) + max(0, int(kv_reading["payable_evictable"]))
    except (TypeError, ValueError, AttributeError):
        return None


def lifetime_room(kv_reading: Optional[Mapping], ladder: Optional[Tuple[int, int, int]] = None) -> Optional[int]:
    """KVOFFSET-1024 (SOLL R5/R7/R11): the largest price D's lifetime gate admits now, from D's own terms
    (``pdflip_kv.lifetime_budget``, ``pdflip/d_lifetime_reading.py``: available + DELIVERABLE evictable - decode
    reserve - dcp deficit, the group floor, - ledger). With the KV ladder D grows its stage at the admission and the
    gate then reads the grown stage minus the SAME reserve and ledger -- so the ladder room carries them too (the
    larger of the two decides, as for the free count). None from an old D without the field: nothing changes."""
    if not kv_reading:
        return None
    try:
        lb = kv_reading.get("lifetime_budget")
        if lb is None:
            return None
        room = int(lb)
        if ladder is not None:
            # review S2: the grown stage carries the same reserve, ledger and uneven-DCP deficit (the deficit pins TP0's
            # number to the binding rank's); the ladder ceiling and its used tokens are group numbers already
            terms = (int(kv_reading.get("decode_reserve", 0) or 0) + int(kv_reading.get("ledger_outstanding", 0) or 0)
                     + int(kv_reading.get("dcp_deficit", 0) or 0))
            room = max(room, int(ladder[0]) - terms - 1)
        return room
    except (TypeError, ValueError, AttributeError):
        return None


def _kv_room(kv_reading: Optional[Mapping]) -> Tuple[Optional[int], str]:
    free, tail, _life = _kv_room_terms(kv_reading)
    return free, tail


def _kv_room_terms(kv_reading: Optional[Mapping]) -> Tuple[Optional[int], str, bool]:
    """``(free, tail)``: the ONE room :func:`kv_free` and :func:`kv_fits` test against -- the larger of D's stage
    reading and its ladder room, under D's backup wall never more than its payable room (S94), and never more than
    the largest price D's lifetime gate admits (KVOFFSET-1024, :func:`lifetime_room`). None without a reading."""
    if not kv_reading:
        return None, "", False
    try:
        free = int(kv_reading.get("available", 0) or 0) + int(kv_reading.get("evictable", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        return None, "", False
    lad = ladder_free(kv_reading)
    tail = ""
    if lad is not None:
        tail = f" (ladder {lad[1]}-{lad[2]}={lad[0]}, stage free {free})"
        free = max(free, lad[0])
    room = payable_room(kv_reading)
    if room is not None and room < free:
        # S94: D's backup wall stands -- the reported evictable rows are not D's to give (rc19 07:15:57 D-WALL-HEAD
        # available=13568 reported_evictable=305024 payable_evictable=0 while the front read free=312036)
        # S94 review A1 (transition until D exports its group MIN): the three numbers are D TP0's; D's admission binds
        # per rank, so this room is an UPPER bound of what the group can begin -- the line says so (room=TP0)
        tail += f" [S94 backup wall: payable room {room} = available {int(kv_reading.get('available', 0) or 0)} + " \
                f"payable {int(kv_reading.get('payable_evictable', 0) or 0)}, room=TP0]"
        free = room
    life = lifetime_room(kv_reading, lad)
    if life is not None and life < free:
        # KVOFFSET-1024: D's gate subtracts its decode reserve (pdflip-25-238: 87 refusals in 23.4 s on offset alone)
        # and counts only the DELIVERABLE evictable (hybrid SSM) -- the front prices against what D admits
        tail += (f" [D lifetime room {life}: deliverable {kv_reading.get('deliverable_evictable', '-')} - reserve "
                 f"{kv_reading.get('decode_reserve', '-')}@{kv_reading.get('decode_reserve_scope', '-')} - ledger "
                 f"{kv_reading.get('ledger_outstanding', '-')}]")
        free = max(0, life)
        return free, tail, True
    return free, tail, False


def kv_free(kv_reading: Optional[Mapping]) -> Optional[int]:
    """The free tokens :func:`kv_fits` tests against (the larger of D's stage
    reading and its ladder room), or None without a reading. S94: under D's backup wall never more than D's
    payable room (:func:`payable_room`). KVOFFSET-1024: never more than D's lifetime room (:func:`lifetime_room`)."""
    return _kv_room(kv_reading)[0]


def price_extra(kv_reading: Optional[Mapping]) -> int:
    """KVOFFSET-1024: what D's gate adds to a new request's ``extend + decode clip`` (one page + the shared-gap Mamba
    cost, ``pdflip_kv.price_extra``); 0 from an old D."""
    try:
        return max(0, int((kv_reading or {}).get("price_extra", 0) or 0))
    except (TypeError, ValueError, AttributeError):
        return 0


def _younger(head_arrival: float, running: Iterable[str], arrival_of: Mapping[str, float],
             lane_of: Optional[Mapping[str, int]], head_lane: int) -> List[Tuple[tuple, str]]:
    """PRIO-AS-AGE 1014 (pdflip/prio_age.py): the running rids YOUNGER than the head by the order key
    ``(-priority, arrival)`` -- a lower priority is younger than every higher one, inside a priority the
    arrival decides -- each with its key.  ``lane_of`` None = every rid priority 0 (the arrival order)."""
    from flliper.srt.pdflip import prio_age as _pa

    on = _pa.enabled() if lane_of is not None else False
    lanes_ = lane_of or {}
    k_head = _pa.key(int(head_lane or 0) if on else 0, float(head_arrival), on=on)
    out = []
    for r in running:
        if r not in arrival_of:
            continue
        k = _pa.key(int(lanes_.get(r, 0) or 0) if on else 0, float(arrival_of[r]), on=on)
        if k > k_head:
            out.append((k, r))
    return out


def kv_displace_victim(head_arrival: Optional[float], running: Iterable[str],
                       arrival_of: Mapping[str, float], tokens_of: Mapping[str, int],
                       deficit: int, lane_of: Optional[Mapping[str, int]] = None,
                       head_lane: int = 0, freed_of=None) -> Optional[str]:
    """NF-STAU-KV (user rule #246, 27.09.: "strikte Ankunftsreihenfolge +
    KV-Backfill, Ältester rückt nach und verdrängt Jüngere"): the head's KV
    does not fit a free seat -> the running decode that ARRIVED last, and
    after the head, parks (SA (3)'s pressure shape, span retained) so the
    head gets its room now instead of after the 60 s bound (c).

    Only a younger one is ever a victim (an older running request has the
    right of way; then the head is blocked by its elders and KV backfill
    applies). Only when the younger ones' KV can cover the deficit at all --
    ``tokens_of`` is the guaranteed release (:func:`victim_release`, KVD-1024), so a
    park that cannot make the head fit is never taken (no churn). A rid with
    no arrival stamp counts as old.  PRIO-AS-AGE 1014: younger/youngest by the order key (``lane_of`` /
    ``head_lane``) -- the victim is the youngest of the lowest priority, never a higher one than the head."""
    if head_arrival is None or int(deficit) <= 0:
        return None
    younger = _younger(head_arrival, running, arrival_of, lane_of, head_lane)
    if not younger:
        return None
    total = sum(max(0, int(tokens_of.get(r, 0) or 0)) for _, r in younger)
    if freed_of is not None:   # KVD review MUSS 2: what all younger ones give back together
        total = max(total, int(freed_of([r for _, r in sorted(younger, reverse=True)])))
    if total < int(deficit):
        return None
    return max(younger)[1]


def victim_release(seat_tokens: Mapping[str, int], basis: Optional[Mapping]) -> Dict[str, int]:
    """KVD-1024 (front KV class D, SOLL R7/R11): what displacing each running decode is GUARANTEED to give back, in
    the unit of the lifetime room the deficit is formed against -- the ONE victim price of the AGE PLAN and of the
    front's own KV park (#246).

    * rows: D's ``leaving_rows_by_rid`` (pdflip/d_leaving_rows.py: the seat minus the path rows other running decodes
      keep locked -- 27B ``_1024_003732`` 01:02:43: 18-189 seat 54133, its displacement freed 36043, the 18137-token
      shared system prompt stayed), never above the front's seat count; a rid D's map does not name: 0 (D does not
      run it, nothing of it is guaranteed). An old D without the map: the seat, as before.
    * under D's backup wall (S94): 0 rows -- the retained span goes back to the tree unpayable.
    * plus its share of D's decode reserve (``decode_reserve_by_rid``, KVOFFSET-1024): D gives it back on leaving.
    """
    b = basis or {}
    rb = b.get("reserve_by_rid") or {}
    lv = b.get("leaving_by_rid")
    wall = bool(b.get("wall"))
    out: Dict[str, int] = {}
    for r, v in seat_tokens.items():
        rows = max(0, int(v or 0))
        if isinstance(lv, Mapping):
            rows = min(rows, max(0, int(lv.get(r, 0) or 0)))
        if wall:
            rows = 0
        out[r] = rows + max(0, int(rb.get(r, 0) or 0))
    return out


def joint_release(victims, basis: Optional[Mapping], seat_tokens: Mapping[str, int],
                  tokens_of: Mapping[str, int]) -> int:
    """KVD review MUSS 2: what displacing ``victims`` TOGETHER is guaranteed to give back -- D's joint release
    (``pdflip_kv.leaving_rows_joint``: cumulative rows of the k youngest in D's displacement order; rows only they hold
    between them, e.g. the system prompt once no older request holds it, come free with the k-th) when the set is
    exactly D's first k, never above their seats, plus their reserve shares; else (another set, an old D, the backup
    wall) the sum of the single prices ``tokens_of`` (:func:`victim_release`). Never below that sum."""
    vs = [str(v) for v in victims]
    indiv = sum(max(0, int(tokens_of.get(v, 0) or 0)) for v in vs)
    b = basis or {}
    joint = b.get("leaving_joint")
    if b.get("wall") or not vs or not isinstance(joint, (list, tuple)) or len(joint) < len(vs):
        return indiv
    try:
        head = [str(e[0]) for e in joint[:len(vs)]]
        rows = int(joint[len(vs) - 1][1])
    except (TypeError, ValueError, IndexError):
        return indiv
    if set(head) != set(vs):
        return indiv
    rows = min(rows, sum(max(0, int(seat_tokens.get(v, 0) or 0)) for v in vs))
    rb = b.get("reserve_by_rid") or {}
    return max(indiv, rows + sum(max(0, int(rb.get(v, 0) or 0)) for v in vs))


def age_plan_enabled(env=None) -> bool:
    """AGE PLAN switch (default on) -- acts only with the rule itself on."""
    if env is not None:
        raw = str(env.get("FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN", "") or "").strip().lower()
        return raw not in ("0", "false", "no", "off") and enabled(env)
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_ENABLE_ARRIVAL_SEAT_AGE_PLAN.get()) and enabled()


def displace_plan(head_arrival: Optional[float], seat_is_free: bool, deficit: int,
                  running: Iterable[str], arrival_of: Mapping[str, float],
                  tokens_of: Mapping[str, int], lane_of: Optional[Mapping[str, int]] = None,
                  head_lane: int = 0, freed_of=None) -> Optional[List[str]]:
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
    ``deficit``. ``tokens_of`` is what displacing a decode is guaranteed to give
    back (:func:`victim_release`, KVD-1024 -- not its seat length). The front only ADMITS by it (a non-empty plan hands the
    head to D); the displacement itself is D's alone
    (``d_park_runtime.displace_for_age`` / ``victims_needed`` on D's real KV).
    A rid without an arrival stamp counts as old.  PRIO-AS-AGE 1014: "arrived after the head" and "youngest
    first" are the order key (``lane_of`` / ``head_lane``). KVD review MUSS 2: ``freed_of(victims)`` -- what the
    set gives back TOGETHER (:func:`joint_release`; default the sum of ``tokens_of``)."""
    if head_arrival is None:
        return None
    need_seat = 0 if seat_is_free else 1
    deficit = max(0, int(deficit or 0))
    if need_seat == 0 and deficit == 0:
        return []
    younger = sorted(_younger(head_arrival, running, arrival_of, lane_of, head_lane), reverse=True)
    out: List[str] = []
    freed = 0
    for _t, r in younger:
        out.append(r)
        freed += max(0, int(tokens_of.get(r, 0) or 0))
        got = max(freed, int(freed_of(list(out)))) if freed_of is not None else freed
        if len(out) >= need_seat and got >= deficit:
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


def kv_fits(need: int, kv_reading: Optional[Mapping], debit: int = 0, decode: int = 0) -> Tuple[bool, str]:
    """``(fits, why)`` against D's reading ``{"available", "evictable"}`` (TP0's
    ``/server_info`` ``pdflip_kv``) -- and, when D publishes its KV ladder, against
    the ladder's ceiling (:func:`ladder_free`): the larger of the two decides.
    No reading (an old D, a failed read): the seat plan alone decides --
    H95c's phase seats are KV-derived already -- and the verdict says so
    (``kv=unread``). KVOFFSET-1024: the room is :func:`kv_free`'s (D's lifetime room included); ``debit`` = the rows
    the front promised to requests D does not run yet in this reading (the burst debit) comes off it."""
    if not kv_reading:
        return True, "kv=unread"
    free, tail, life_binds = _kv_room_terms(kv_reading)
    if free is None:
        return True, "kv=unreadable"
    if int(debit or 0) > 0:
        free = free - int(debit)    # the caller names the debit (front: ``D-line debit ...``)
    if int(need) <= free:
        return True, f"kv need={int(need)} free={free}{tail}"
    try:
        gap = int(kv_reading.get("cap_blind_gap", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        gap = 0
    born = int(need) - max(0, int(decode or 0))
    if life_binds and gap > 0 and born <= free and int(need) <= free + gap:
        # review S1: D's cap waiver (``PrefillAdder._cap_lifetime_waiver``) -- with an engaged residency cap the gate
        # admits up to its budget + the group's cap-blind gap when the IMMEDIATE demand (price - decode) fits
        return True, f"kv need={int(need)} free={free}{tail} [D cap waiver: born {born} <= {free}, gap {gap}]"
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


def youngest_running(admit_t: Mapping[str, float], running: Iterable[str],
                     lane_of: Optional[Mapping[str, int]] = None,
                     than: Optional[tuple] = None) -> Optional[str]:
    """(c): the running decode admitted LAST to a D seat (#244's order). A rid
    with no admission stamp (resumed before this boot's stamps) counts as the
    oldest -- it is never chosen over a stamped one.

    PRIO-AS-AGE 1014: with ``lane_of`` the youngest is by the order key -- the last admitted of the LOWEST
    priority; ``than`` (the oldest waiter's key) excludes every runner not younger than it, so a priority-0
    waiter never parks a priority-1 decode."""
    from flliper.srt.pdflip import prio_age as _pa

    on = _pa.enabled() if lane_of is not None else False
    lanes_ = lane_of or {}
    best, best_k = None, None
    for rid in running:
        t = admit_t.get(rid)
        if t is None:
            continue
        k = _pa.key(int(lanes_.get(rid, 0) or 0) if on else 0, t, on=on)
        if on and than is not None and k[0] < than[0]:
            continue  # a higher priority than the waiter's: older by the key, never its victim
        if best_k is None or k > best_k:
            best, best_k = rid, k
    return best


def bound_fired(wait_s: Optional[float], bound_s: float) -> bool:
    return wait_s is not None and float(bound_s) > 0 and float(wait_s) >= float(bound_s)


def state_block(counters: Mapping[str, int], waiters: Dict[str, float], armed: bool) -> dict:
    """state.json ``front.arrival_seat``."""
    out = {"armed": bool(armed), "waiting_for_seat": len(waiters)}
    for k in COUNTERS:
        out[k[len("arrival_seat_"):]] = int(counters.get(k, 0) or 0)
    return out
