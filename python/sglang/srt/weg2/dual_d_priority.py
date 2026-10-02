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


class PressureStages:
    """The P side of D priority, REPLICATED nowhere (the front alone decides;
    P's ranks only execute RPCs). Pure: the caller feeds one reading per tick and
    executes the returned action -- ``stop`` (pause leg 1 at its chunk boundary,
    P releases its KV), ``sleep`` (P parks its weights in host RAM), ``resume``
    (the paused head may go back), ``wake`` (P maps its weights back) -- and
    prints the returned line.

    Order is the law: stage 2 only after stage 1 completed (P commits 0 bytes on
    every card) and the pressure held ``sleep_after`` ticks more. Return only
    with hysteresis: no pressure, and the card's free bytes cover one whole P
    grant step plus D's look-ahead; from sleep additionally the P weights and a
    D seat that finished since P went to sleep."""

    def __init__(self, sleep_capable: bool, sleep_after: int = SLEEP_AFTER_TICKS_DEFAULT):
        self.sleep_capable = bool(sleep_capable)
        self.sleep_after = max(1, int(sleep_after))
        self.p_state = "serving"
        self._held = 0
        self._seat_mark = 0
        self._said = set()
        self.counts = {"stage1": 0, "stage2": 0, "resume": 0, "wake": 0, "unavailable": 0, "refused": 0}

    def _line(self, stage: str, d_need: int, freed: int, extra: str = "") -> str:
        return "%s stage=%s d_need=%d freed=%d p_state=%s%s" % (LINE, stage, int(d_need), int(freed),
                                                                  self.p_state, extra)

    def tick(self, *, pressure: int, p_committed: int, free_min: int, p_grant_bytes: int,
             d_air_bytes: int, seats_done: int, weights_bytes: int = 0,
             host_ok: bool = True,
             card_room: Optional[Sequence[Tuple[int, int]]] = None) -> Tuple[Optional[str], Optional[str]]:
        """``card_room``: per card (free bytes, bytes the wake needs there = that
        card's loan + one P grant step + D's look-ahead). The wake from sleep is
        judged PER CARD -- the loan sits on each card separately (gmps12: 7.96 /
        3.22 / 3.51 GB), and a total against the tightest card's free can never
        hold on a 3080 (budget 6.4 GB). Without it, the old total formula."""
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
            if free_min >= int(p_grant_bytes) + int(d_air_bytes):
                self.p_state = "serving"
                self.counts["resume"] += 1
                return "resume", self._line("resume", 0, free_min)
            return None, None
        # sleeping
        if card_room is not None:
            room_ok = bool(card_room) and all(int(f) >= int(n) for f, n in card_room)
        else:
            room_ok = free_min >= int(weights_bytes) + int(p_grant_bytes) + int(d_air_bytes)
        if pressure <= 0 and int(seats_done) > self._seat_mark and room_ok:
            self.p_state = "serving"
            self.counts["wake"] += 1
            return "wake", self._line("resume", 0, free_min, " from=sleep")
        return None, None
