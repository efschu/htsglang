"""DUAL-TP3PP3 D PRIORITY UNDER KV PRESSURE (user decision 01.10., verbatim):
"das darf nie passieren. davor muss P aufhören zu prefillen und seinen kv
freigeben und wenn es dann immer noch nicht reicht, muss P schlafen und seine
gewichte im systemram parken. es geht um unterbrechungsfreies decode. wenn ein
sitz fertig decoded hat wird sein platz ja frei und P kann zurückkehren und
weitermachen".

Metal gmps7 (boot dkr27bnvfp4dual1mbar1fs10011748, D 17:53:15): "KV cache pool is
full. Retract requests. #retracted_reqs: 4" -> four W50 re-routes -> P's grant for
four contexts at once -> PP0 died of cuMemCreate OOM. In the dual layout a
retracted D decode is an interrupted decode; the stages that keep D's pool from
running full act BEFORE it (P stops and frees its KV, then P sleeps), and a
retract that still comes is a NAMED STOP here, never a silent path.

This module holds the D-side tripwire (no retract in the dual layout).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MARK = "W-DUAL-D-RETRACT"


class Weg2DualDRetract(RuntimeError):
    """W-DUAL-D-RETRACT: group D of the dual layout was about to retract a
    running decode. Decode is never interrupted in the dual layout; the pool
    must be kept clear by the P stages (P stops and frees its KV, P sleeps)."""


def d_retract_forbidden(env=None) -> bool:
    """True on a group-D rank of the dual layout (both groups carry
    SGLANG_WEG2_DUAL_LAYOUT=1)."""
    e = os.environ if env is None else env
    return ((e.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() == "1"
            and (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() == "D")


def refuse_d_retract(batch, *, kv_full: bool, reason, env=None) -> None:
    """Raise W-DUAL-D-RETRACT on a dual D rank instead of retracting. Group-
    uniform: the decision to retract is uniform over the D ranks (#583/#603),
    and this predicate reads only the env, the same on every rank."""
    if not d_retract_forbidden(env):
        return
    reqs = list(getattr(batch, "reqs", None) or ())
    rids = [str(getattr(r, "rid", "?"))[:16] for r in reqs]
    raise Weg2DualDRetract(
        "%s: group D of the dual layout was about to retract %s (kv_full=%s, reason=%s) -- decode is "
        "never interrupted here; P must have stopped, freed its KV or slept before D's pool ran full"
        % (MARK, rids, bool(kv_full), reason or "-"))


# -- D HOLD FOR GROW: emergency growth instead of the retract (D side) -----------
#
# gmps12 (boot ...dual1mpsleepbar1fs10020008, D 00:16:45-53): D's stage shrank to
# 69632 for a waiting P prompt, a 35736-token hand-back came in, the decode round
# found the pool full (uniform_min_avail < num_tokens_next) and W-DUAL-D-RETRACT
# stopped the layout -- while the card ledger had 2.4 GB free. The decode is never
# interrupted (user rule); instead D grows on the spot (group-uniform), and when
# the card is really short it HOLDS the batch for this iteration: the short request
# left pressure on P in the ledger (stage 1, then stage 2), the next iteration asks
# again. Only a hold that outlasts HOLD_MAX_S ends in the named stop.

HOLD_MARK = "D-HOLD-FOR-GROW"
HOLD_MAX_ENV = "SGLANG_WEG2_DUAL_D_HOLD_MAX_S"
#: 120 s. What the hold waits for, measured on gmps12: the sleep leg 2.2 s
#: (front 00:11:52.153 -> 00:11:54.368), stage 1 one P chunk boundary, the stages'
#: SLEEP_AFTER_TICKS x the front's 0.2 s tick -- the whole ladder in ~10 s. 120 s is
#: >10x that and stays under the 300 s a client waits for its first byte (H102), so
#: the named stop comes only when the ladder cannot free the pool at all (stage 2
#: refused by the host floor, D at its top, a hung P), not for a slow sleep.
HOLD_MAX_S_DEFAULT = 120.0
#: grow target in the hold path: the deficit at this factor plus two lattice steps.
#: The deficit is read on the tightest rank; the factor covers an uneven owner
#: share (one rank's rows vs the group's tokens) so ONE grow suffices -- the
#: re-read after the grow is the truth either way.
HOLD_DEFICIT_FACTOR = 4
_BATCH_ATTR = "_weg2_hold_for_grow"
_SCHED_ATTR = "_weg2_d_hold"


def hold_max_s(env=None) -> float:
    e = os.environ if env is None else env
    try:
        v = float(e.get(HOLD_MAX_ENV, "") or HOLD_MAX_S_DEFAULT)
    except ValueError:
        return HOLD_MAX_S_DEFAULT
    return v if v > 0 else HOLD_MAX_S_DEFAULT


def _group_avail(actor, gmin) -> int:
    """The pool's free rows, MIN over the D ranks -- the same reading
    ``uniform_min_avail`` reduces (allocator.available_size), taken AFTER the
    eviction ``update_running_batch`` already ran."""
    return int(gmin([int(actor.allocator.available_size())])[0])


def end_hold(sched, why: str, now=time.monotonic) -> None:
    """The hold episode is over: one line with its count and wall time."""
    ep = getattr(sched, _SCHED_ATTR, None)
    if ep is None:
        return
    setattr(sched, _SCHED_ATTR, None)
    logger.warning("%s n=%d ms=%d released=%s -- the decode runs again, nothing retracted", HOLD_MARK,
                   int(ep["n"]), int((float(now()) - ep["t0"]) * 1000), why)


def take_hold_for_grow(batch) -> bool:
    """The caller's half: True when update_running_batch held ``batch`` this
    iteration (the decode does not run; the batch stays the running batch).
    Consumes the mark."""
    if batch is None or not getattr(batch, _BATCH_ATTR, False):
        return False
    setattr(batch, _BATCH_ATTR, False)
    return True


def grow_or_hold(sched, batch, num_tokens_next: int, *, now=time.monotonic, env=None) -> str:
    """Dual D only, called where the decode round found the pool full (the flag
    is group-uniform, #603). Returns

    * ``"go"``: the pool has the rows (after the eviction, or after an emergency
      group grow to the deficit x HOLD_DEFICIT_FACTOR + 2 lattice steps) -- run
      the decode, no retract;
    * ``"hold"``: the card ledger is short -- the request left pressure/demand on
      P (the front's stages 1 and 2), this iteration skips the decode, the batch
      is marked (``take_hold_for_grow``) and stays; ``D-HOLD-FOR-GROW n= ms=``;
    * ``"retract"``: not dual D, no D actor, or the hold outlasted
      ``SGLANG_WEG2_DUAL_D_HOLD_MAX_S`` -- the caller's retract runs and
      ``refuse_d_retract`` stops by name (W-DUAL-D-RETRACT).

    Group-uniform: every branch is taken on a MIN-reduced value (the avail
    re-reads, group_grow's verdict, the expiry vote), so every D rank enters the
    same collectives and returns the same verdict."""
    if not d_retract_forbidden(env):
        return "retract"
    from sglang.srt.weg2 import dual_d_kv_stage as _ddk
    from sglang.srt.weg2 import dual_p_kv_stage as _pk

    runner = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    actor = getattr(runner, _ddk.ACTOR_ATTR, None)
    if actor is None:
        return "retract"
    gmin = getattr(sched, "_weg2_group_min_ints", None) or actor.gmin
    actor.gmin = gmin
    need = int(num_tokens_next)
    avail = _group_avail(actor, gmin)
    if avail >= need:
        end_hold(sched, "evicted", now)
        return "go"
    target = int(actor.mapped_tokens) + _pk.round_up(
        2 * int(actor.step) + HOLD_DEFICIT_FACTOR * (need - avail), int(actor.step))
    if actor.group_grow(target):
        avail = _group_avail(actor, gmin)
        if avail >= need:
            end_hold(sched, "grown", now)
            return "go"
    t = float(now())
    ep = getattr(sched, _SCHED_ATTR, None)
    if ep is None:
        ep = {"t0": t, "n": 0, "next": 0.0, "iv": 0.5}
        setattr(sched, _SCHED_ATTR, ep)
    ep["n"] += 1
    ms = int((t - ep["t0"]) * 1000)
    expired = -int(gmin([-(1 if ms >= hold_max_s(env) * 1000 else 0)])[0]) > 0
    rids = [str(getattr(r, "rid", "?"))[:16] for r in (getattr(batch, "reqs", None) or ())]
    if expired:
        setattr(sched, _SCHED_ATTR, None)
        logger.error("%s n=%d ms=%d EXPIRED after %s=%.0f s (need=%d avail=%d mapped=%d top=%d) -- the "
                     "ladder did not free the pool; the retract's named stop follows", HOLD_MARK, ep["n"], ms,
                     HOLD_MAX_ENV, hold_max_s(env), need, avail, actor.mapped_tokens, actor.top)
        return "retract"
    if t >= ep["next"]:
        ep["iv"] = min(10.0, 2.0 * float(ep["iv"]))
        ep["next"] = t + ep["iv"]
        logger.warning("%s n=%d ms=%d need=%d avail=%d mapped=%d want=%d top=%d reqs=%s -- the card is "
                       "short: the decode waits this iteration (nothing retracted), D's request pressed P "
                       "(stage 1, then stage 2)", HOLD_MARK, ep["n"], ms, need, avail, actor.mapped_tokens,
                       min(int(actor.top), target), actor.top, rids)
    setattr(batch, _BATCH_ATTR, True)
    return "hold"


# -- the P stages (front, dual only) -------------------------------------------

#: one instrument line per stage transition
LINE = "WEG2 DUAL-KV-PRESSURE"
#: pressure ticks with P fully released (stage 1 done) before P sleeps (stage 2)
SLEEP_AFTER_TICKS_ENV = "SGLANG_WEG2_DUAL_P_SLEEP_AFTER_TICKS"
SLEEP_AFTER_TICKS_DEFAULT = 3
#: P sleeps only when the host keeps this much MemAvailable above the weights
#: image (operator order 01.10.: MemAvailable - peak >= 6 GiB, else refused)
HOST_FLOOR_BYTES = 6 << 30
#: the front's capability switch: set by the launcher when P boots with the
#: weights CPU backup (dual profile, P only) -- absent = stage 2 unavailable
P_SLEEP_ENV = "SGLANG_WEG2_DUAL_P_SLEEP"


def p_sleep_capable(env=None) -> bool:
    e = os.environ if env is None else env
    return (e.get(P_SLEEP_ENV, "") or "").strip() == "1"


def host_allows_sleep(mem_available: int, weights_bytes: int, floor: int = HOST_FLOOR_BYTES) -> bool:
    """KEIN-DAUER-HOSTRAM: the weights image lies in host RAM only while P
    sleeps; the sleep is refused when it would leave less than ``floor``."""
    return int(mem_available) - int(weights_bytes) >= int(floor)


# -- Q-660: D's shortage beyond the ledger bytes --------------------------------
#
# Dual y8v (fs10031504, 15:20:53): D stood at "full token usage 0.96" (its 1M-row id
# space, locked) and the shared Mamba arena was full (ARENA-REF-CENSUS complete=112
# of 112) while the card ledgers showed free bytes -- no ledger demand, so no stage
# ever fired. D publishes both readings (``publish_d_signal``, D's tick); the front
# counts either as D pressure next to the ledger's demand.

D_SIGNAL_ID_ENV = "SGLANG_WEG2_DUAL_D_ID_PRESSURE"
#: D's id space this full (rows used and not evictable / all rows) presses P
D_SIGNAL_ID_DEFAULT = 0.90
#: the arena presses P when at most this many slots are left
D_SIGNAL_ARENA_MARGIN = 2
#: a D reading older than this is no reading (D stopped publishing)
D_SIGNAL_MAX_AGE_S = 5.0
#: D publishes at most this often
D_SIGNAL_EVERY_S = 0.5


def d_signal_file(tag: str, root: str = "/dev/shm") -> str:
    import hashlib

    return os.path.join(root, "wkvd-%s.json" % hashlib.sha1(str(tag).encode()).hexdigest()[:10])


def d_id_threshold(env=None) -> float:
    e = os.environ if env is None else env
    try:
        v = float(e.get(D_SIGNAL_ID_ENV, "") or D_SIGNAL_ID_DEFAULT)
    except ValueError:
        return D_SIGNAL_ID_DEFAULT
    return v if 0.0 < v <= 1.0 else D_SIGNAL_ID_DEFAULT


def d_signal_short(sig: Optional[dict], *, now: float, unit: int, id_threshold: float = D_SIGNAL_ID_DEFAULT,
                   arena_margin: int = D_SIGNAL_ARENA_MARGIN) -> Tuple[int, str]:
    """D's published reading -> (pressure bytes, why). ``unit``: the bytes one
    reason counts as (one P grant step, so stage 1 sees real pressure); 0 when
    D is not short or the reading is missing/stale."""
    if not sig:
        return 0, ""
    try:
        if float(now) - float(sig.get("ts", 0.0)) > D_SIGNAL_MAX_AGE_S:
            return 0, ""
        why = []
        frac = float(sig.get("id_frac", 0.0) or 0.0)
        if frac >= float(id_threshold):
            why.append("id_space=%.2f" % frac)
        slots = int(sig.get("arena_slots", 0) or 0)
        complete = int(sig.get("arena_complete", 0) or 0)
        if slots > 0 and complete >= slots - int(arena_margin):
            why.append("arena=%d/%d" % (complete, slots))
    except (TypeError, ValueError):
        return 0, ""
    return (max(1, int(unit)), ",".join(why)) if why else (0, "")


def d_signal_seat_gate(sig_short: int, sig_why: str, *, d_has_seats: bool, armed: bool) -> Tuple[int, str]:
    """#1540 D-SIGNAL-SEATS: D's id-space / arena reading is pressure only while D has work that needs
    the rows. ``d_has_seats``: a live D seat or a leg-1-done request waiting for one (the front's
    ``_d_seats_live`` / ``_ready_for_d``). Disarmed: the reading passes unchanged (the old behaviour)."""
    if armed and not d_has_seats:
        return 0, ""
    return sig_short, sig_why


def publish_d_signal(path: str, *, id_frac: float, arena_complete: int, arena_slots: int, now: float) -> None:
    import json

    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump({"ts": float(now), "id_frac": float(id_frac), "arena_complete": int(arena_complete),
                   "arena_slots": int(arena_slots)}, f)
    os.replace(tmp, path)


def read_d_signal(path: str) -> Optional[dict]:
    import json

    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


#: Q-660: calm ticks (no pressure on P) before an awake loan of stage 1 comes back
RECLAIM_AFTER_TICKS_ENV = "SGLANG_WEG2_DUAL_P_RECLAIM_AFTER_TICKS"
#: 10 x the front's 0.2 s tick = 2 s without D pressure (or sooner: a D seat ended)
RECLAIM_AFTER_TICKS_DEFAULT = 10


class PressureStages:
    """The P side of D priority, REPLICATED nowhere (the front alone decides;
    P's ranks only execute RPCs). Pure: the caller feeds one reading per tick and
    executes the returned action -- ``stop`` (pause leg 1 at its chunk boundary,
    P releases its KV), ``lend`` (Q-660: P, awake and released, lends its freed
    device bytes to the card pool), ``sleep`` (P parks its weights in host RAM),
    ``reclaim`` (the awake loan back), ``resume`` (the paused head may go back),
    ``wake`` (P maps its weights back) -- and prints the returned line.

    Order is the law (user rules 01.10. and 03.10. "wenn auf D kv knapp wird,
    gibt P seinen kv auf"): stage 1 = stop, and -- once P commits 0 bytes on
    every card -- lend at once, awake; stage 2 (sleep) only when the pressure
    still holds ``sleep_after`` ticks after the loan. D never retracts. Return
    only with hysteresis: from the loan after ``reclaim_after`` calm ticks or a
    D seat that ended, with the card's free bytes covering the loan plus one P
    grant step and D's look-ahead -- P prefills again only once the loan is
    back (``p_lent`` 0); from sleep additionally the P weights and a D seat that
    finished since P went to sleep."""

    def __init__(self, sleep_capable: bool, sleep_after: int = SLEEP_AFTER_TICKS_DEFAULT,
                 reclaim_after: int = RECLAIM_AFTER_TICKS_DEFAULT):
        self.sleep_capable = bool(sleep_capable)
        self.sleep_after = max(1, int(sleep_after))
        self.reclaim_after = max(1, int(reclaim_after))
        self.p_state = "serving"
        self._held = 0
        self._calm = 0
        self._seat_mark = 0
        self._lend_seat_mark = 0
        self._said = set()
        self.counts = {"stage1": 0, "lend": 0, "stage2": 0, "reclaim": 0, "resume": 0, "wake": 0,
                       "unavailable": 0, "refused": 0}

    def _line(self, stage: str, d_need: int, freed: int, extra: str = "") -> str:
        return "%s stage=%s d_need=%d freed=%d p_state=%s%s" % (LINE, stage, int(d_need), int(freed),
                                                                  self.p_state, extra)

    @staticmethod
    def _room(card_room, free_min: int, need: int) -> bool:
        if card_room is not None:
            return bool(card_room) and all(int(f) >= int(n) for f, n in card_room)
        return int(free_min) >= int(need)

    def tick(self, *, pressure: int, p_committed: int, free_min: int, p_grant_bytes: int,
             d_air_bytes: int, seats_done: int, weights_bytes: int = 0,
             host_ok: bool = True, p_lent: int = 0,
             card_room: Optional[Sequence[Tuple[int, int]]] = None) -> Tuple[Optional[str], Optional[str]]:
        """``card_room``: per card (free bytes, bytes the return needs there = that
        card's loan + one P grant step + D's look-ahead). The return is judged PER
        CARD -- the loan sits on each card separately (gmps12: 7.96 / 3.22 / 3.51
        GB), and a total against the tightest card's free can never hold on a 3080
        (budget 6.4 GB). Without it, the old total formula. ``p_lent``: the bytes
        P's stages still lend (sleep + awake loan, from the stage files)."""
        pressure, p_committed, free_min = int(pressure), int(p_committed), int(free_min)
        if self.p_state == "serving":
            if pressure > 0:
                self.p_state = "stopped"
                self._held = 0
                self._said.clear()
                self.counts["stage1"] += 1
                return "stop", self._line("1", pressure, p_committed)
            return None, None
        if self.p_state == "stopped":
            if pressure > 0:
                if p_committed > 0:
                    self._held = 0                     # stage 1 not complete yet
                    return None, None
                # stage 1 complete (P commits 0 B on every card): lend at once, awake (Q-660)
                self.p_state = "lent"
                self._held = 0
                self._calm = 0
                self._lend_seat_mark = int(seats_done)
                self.counts["lend"] += 1
                return "lend", self._line("1-lend", pressure, 0)
            if free_min >= int(p_grant_bytes) + int(d_air_bytes):
                self.p_state = "serving"
                self.counts["resume"] += 1
                return "resume", self._line("resume", 0, free_min)
            return None, None
        if self.p_state == "lent":
            if pressure > 0:
                self._calm = 0
                self._held += 1
                if self._held < self.sleep_after:
                    return None, None
                if not self.sleep_capable:
                    if "unavailable" not in self._said:
                        self._said.add("unavailable")
                        self.counts["unavailable"] += 1
                        return None, self._line("2", pressure, 0, " unavailable (weights resident)")
                    return None, None
                if not host_ok:
                    if "refused" not in self._said:
                        self._said.add("refused")
                        self.counts["refused"] += 1
                        return None, self._line("2", pressure, 0, " refused host_ram weights=%d" % int(weights_bytes))
                    return None, None
                self.p_state = "sleeping"
                self._seat_mark = int(seats_done)
                self.counts["stage2"] += 1
                return "sleep", self._line("2", pressure, int(weights_bytes))
            self._held = 0
            self._calm += 1
            calm = self._calm >= self.reclaim_after or int(seats_done) > self._lend_seat_mark
            if calm and self._room(card_room, free_min, int(p_grant_bytes) + int(d_air_bytes)):
                self.p_state = "reclaiming"
                self._calm = 0
                self.counts["reclaim"] += 1
                return "reclaim", self._line("1-reclaim", 0, free_min)
            return None, None
        if self.p_state == "reclaiming":
            if int(p_lent) <= 0:
                if pressure > 0:                       # the loan is back but D presses again
                    self.p_state = "stopped"
                    self._held = 0
                    return None, None
                self.p_state = "serving"
                self.counts["resume"] += 1
                return "resume", self._line("resume", 0, free_min, " from=lend")
            if pressure > 0:                           # the loan stays with D: lent again
                self.p_state = "lent"
                self._calm = 0
                return None, None
            self._calm += 1
            if self._calm >= self.reclaim_after:       # D still held part of the loan: ask again
                self._calm = 0
                return "reclaim", self._line("1-reclaim", 0, free_min, " retry")
            return None, None
        # sleeping
        room_ok = self._room(card_room, free_min, int(weights_bytes) + int(p_grant_bytes) + int(d_air_bytes))
        if pressure <= 0 and int(seats_done) > self._seat_mark and room_ok:
            self.p_state = "serving"
            self.counts["wake"] += 1
            return "wake", self._line("resume", 0, free_min, " from=sleep")
        return None, None
