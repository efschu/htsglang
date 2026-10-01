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

import os
from typing import Optional, Tuple

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
             host_ok: bool = True) -> Tuple[Optional[str], Optional[str]]:
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
        if (pressure <= 0 and int(seats_done) > self._seat_mark
                and free_min >= int(weights_bytes) + int(p_grant_bytes) + int(d_air_bytes)):
            self.p_state = "serving"
            self.counts["wake"] += 1
            return "wake", self._line("resume", 0, free_min, " from=sleep")
        return None, None
