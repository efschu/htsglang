"""S7b P-RESTART (24.10., Auftrag #214): in the 27B dual layout a dead P group no longer ends the boot.

WHERE THIS COMES FROM. #210 (desk/27b-d-survives-p-1024 @ 2c854c53c5) made D survive a P death on the GPU side
(a dying MPS client drains or is terminated before it exits). The front still stopped the WHOLE boot on every group
death: ``PDFLIP STOP W17 PdFlipGroupDead`` (health poller), ``W-DUAL-P-WAKE`` / ``W-DUAL-P-SLEEP`` (a stage leg to a
dead P), each followed by the 503 of every request and the SIGTERM of D (10240650: W-DUAL-P-WAKE 46 s after the
P death, D decoding until then; 10210031: 3 min). S7 expects: D serves on, P comes back.

STAGE 1 (this module + four call sites in front.py, dual layout only):
  * ``on_p_dead(front, why)`` -- every site that would have stopped the boot for P calls it first; True = taken
    over, the caller does NOT stop. P is marked absent (state ``dead``), no P leg is dispatched any more (the dual
    pump is not run, the running pass's ``may_dispatch`` turns False), one named line ``S7B P-DEAD``.
  * in-flight P legs (``_dual_inflight``) are marked ``dual_pause`` with one named line each (``S7B P-LEG-ABORTED``);
    they end at P's transport failure (bounded by P's own death: the server's crash-diagnose window and
    ``kill_process_tree``, 38-67 s on the specimens) and the dual pass's OWN pause branch requeues them at the head
    (``_dual_requeue_paused``: never failed to the client, never handed to D half-prefilled). A leg that fails before
    the poller saw the death marks itself through ``leg1_rpc`` when P's session is gone or the poller holds a
    dead/held fact. The 27B drain pass body stays byte-identical (test_27b_port_dual_unchanged_1020 pins it); the new
    leg gate sits in ``Front._dual_dispatch_held``, which that pass reads before every new leg. Dual boots without
    card ledgers keep the old per-request failure (no dispatch gate there, a requeue would loop).
  * ``tick(front, now)`` (the controller, every loop while P is absent): every queued request D may take
    (SK-kept SHORT, or u <= X and not p_only) goes to D's admission line (``S7B TO-D``); D never prefills over X
    (Phasen-Gesetz, user veto 10.09.). A request u > X (or p_only, R9) waits for P -- no flip/P phase without P --
    at most ``FLLIPER_PDFLIP_DUAL_P_RESTART_WAIT_S`` (default 900 s) and is then answered 503 by name
    (``S7B P-WAIT-EXPIRED``); without an armed restart it is answered at once (``S7B P-ABSENT-NO-RESTART``).
  * the front's /health stays 200 while D serves (the P column says ``absent``), ``/pdflip/state`` carries
    ``p_restart``.

STAGE 2 (``restart_driver``): the launcher writes P's exec record (argv, env, cwd, log) at P's spawn
(``write_respawn_spec``; path in ``FLLIPER_PDFLIP_DUAL_P_RESPAWN_SPEC``). The driver
  1. waits for P's old session to end (grace ``FLLIPER_PDFLIP_DUAL_P_RESTART_GRACE_S``, default 90 s -- the server's
     own crash-diagnose window ends in ``kill_process_tree`` on the specimens); past the grace it ends the session
     itself: MPS ``terminate_client`` for every old P pid first (#210: a SIGKILL on a computing MPS client is the
     undefined case), then ``kill_process_tree``;
  1b. CARD ROOM: every P card must have P's PLANNED peak free again (budget + overhead, the number the launcher
     sized D against, written into the record as ``card_need_mib``; NVML allocatable free) before the spawn -- a P that slept had lent its
     weights' bytes to the card pool (#1956 WAKE-SEES-LOAN), D may still hold them; P comes back only once the card
     has them free (bound ``FLLIPER_PDFLIP_DUAL_P_RESTART_CARD_WAIT_S``, default 600 s; no reserve on top);
  2. spawns P again with the SAME argv/env (same cards, same planner cut -- the record is the launcher's exec'd
     argv, nothing re-planned), new session, log appended, ``groups["P"].sid`` = the new pid;
  3. BARRIER (R11, the front decides alone): P counts as back only when its /health answers 200 AND every P rank
     rewrote its dual stage file after the spawn (the ranks' own sign-in; a stale file of the dead P never
     counts). Bound ``FLLIPER_PDFLIP_DUAL_P_RESTART_READY_S`` (default 900 s);
  4. resets the front's P-side state (stage ladder, in-flight map, health streak, progress snapshot) and returns
     P to ``serving``: the dual P boots awake with no KV committed (``dual_p_kv_stage``: contribute(committed=0)),
     which IS the dormant form of this layout; its KV comes per grant from the card ledger exactly as at boot.
  At most ``FLLIPER_PDFLIP_DUAL_P_RESTART_MAX`` (default 2) restarts per boot; a failed attempt is named
  (``S7B P-RESTART FAILED``) and the state becomes ``failed`` (P-bound requests answered, D serves on).

MEMORY RULES at the restart (named, nothing new invented): the card KV ledger reaps a dead pid's slot and its
contribution (card_kv_ledger._reap), so D cannot grow its KV into P's freed bytes while P is away; the new P
contributes again under the same I1 rule (``contribute``: the other group's released contribution is not counted
twice). No reserve on top (KEINE-KORRIDOR-RESERVE); the corridor floor is the ranks' own runtime law. The #213
wake-floor rule (pending, unmapped claims are no budget item) belongs to the flip wake credit, which the dual
layout never takes (every flip is refused under --dual-layout).

Switches (default ON, user order 14.10.): ``FLLIPER_PDFLIP_DUAL_P_SURVIVE`` (stage 1; 0 = the old STOP on every P
death) and ``FLLIPER_PDFLIP_DUAL_P_RESTART`` (stage 2; 0 = P stays absent, P-bound requests are answered at once).
Outside ``--dual-layout`` nothing here runs.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

MARK = "S7B"
ENV = "FLLIPER_PDFLIP_DUAL_P_SURVIVE"
RESTART_ENV = "FLLIPER_PDFLIP_DUAL_P_RESTART"
SPEC_ENV = "FLLIPER_PDFLIP_DUAL_P_RESPAWN_SPEC"
WAIT_ENV = "FLLIPER_PDFLIP_DUAL_P_RESTART_WAIT_S"
GRACE_ENV = "FLLIPER_PDFLIP_DUAL_P_RESTART_GRACE_S"
READY_ENV = "FLLIPER_PDFLIP_DUAL_P_RESTART_READY_S"
MAX_ENV = "FLLIPER_PDFLIP_DUAL_P_RESTART_MAX"
CARD_WAIT_ENV = "FLLIPER_PDFLIP_DUAL_P_RESTART_CARD_WAIT_S"
DEFAULT_CARD_WAIT_S = 600.0
#: review SOLL 3: the per-request wait covers the driver's own bounds (grace 90 + kill 30 + card 600 + ready 900)
DEFAULT_WAIT_S = 1800.0
DEFAULT_GRACE_S = 90.0
DEFAULT_READY_S = 900.0
DEFAULT_MAX = 2
SPEC_SCHEMA = "pdflip.p_respawn/1"
#: review SOLL 7: what the respawn record never holds in clear (env NAMES, case-insensitive; argv flags)
SECRET_NAME_RX = r"TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_KEY|ADMIN_KEY|ACCESS_KEY|PRIVATE_KEY"
SECRET_ARGV_FLAGS = ("--admin-api-key",)   # restored from the front's own admin key
MASK = "S7B-MASKED"
#: review MUSS 4: #1223 DEBUG-HOLD (debug_hold.HOLD_ENV / HOLD_S_ENV / DEFAULT_HOLD_S, and the pre-rename names)
HOLD_ENVS = ("FLLIPER_DEBUG_HOLD", "SGLANG_DEBUG_HOLD")
HOLD_S_ENVS = ("FLLIPER_DEBUG_HOLD_S", "SGLANG_DEBUG_HOLD_S")
DEFAULT_HOLD_S = 1800.0
STATUS_EVERY_S = 30.0

SERVING = "serving"
DEAD = "dead"
RESTARTING = "restarting"
JOINING = "joining"
FAILED = "failed"
ABSENT_STATES = (DEAD, RESTARTING, JOINING, FAILED)


class PdFlipPDead(Exception):
    """A P-bound request answered while P is absent (503, by name). Not a STOP: D serves on."""

    def __init__(self, name: str, detail: str):
        super().__init__(f"{name}: {detail}")
        self.name = name
        self.detail = detail


def _on(env, name: str) -> bool:
    return str(env.get(name, "1")).strip().lower() not in ("0", "false", "off", "no")


def _num(env, name: str, default: float) -> float:
    try:
        v = float(str(env.get(name, "") or default))
    except (TypeError, ValueError):
        return default
    return v if v >= 0 else default


def armed(front, env=None) -> bool:
    """Stage 1 takes over: the dual layout and the switch (default on)."""
    env = os.environ if env is None else env
    return bool(getattr(front, "dual_layout", False)) and _on(env, ENV)


def restart_armed(env=None) -> bool:
    """Stage 2 may respawn P: the switch (default on) and a readable respawn record."""
    env = os.environ if env is None else env
    if not _on(env, RESTART_ENV):
        return False
    return read_spec(str(env.get(SPEC_ENV, "") or "")) is not None


@dataclasses.dataclass
class PRestartState:
    state: str = SERVING
    why: str = ""
    t_dead: float = 0.0
    old_sid: int = 0
    new_pid: int = 0
    t_spawn: float = 0.0
    restarts: int = 0
    deaths: int = 0
    legs_aborted: int = 0
    to_d: int = 0
    answered: int = 0
    last_status_t: float = 0.0
    note: str = ""
    no_restart: bool = False
    arena_dead: Any = None
    arena_back: Any = None
    task: Any = None

    def snapshot(self) -> dict:
        return {"state": self.state, "why": self.why[:300], "t_dead": round(self.t_dead, 3),
                "old_sid": self.old_sid, "new_pid": self.new_pid, "t_spawn": round(self.t_spawn, 3),
                "restarts": self.restarts, "deaths": self.deaths, "legs_aborted": self.legs_aborted,
                "to_d": self.to_d, "answered": self.answered, "note": self.note[:300]}


def state_of(front) -> PRestartState:
    d = front.__dict__
    st = d.get("_s7b_state")
    if st is None:
        st = d["_s7b_state"] = PRestartState()
    return st


def p_absent(front, env=None) -> bool:
    """True while stage 1 holds P as absent (dead, restarting, joining or failed)."""
    if not armed(front, env):
        return False
    st = front.__dict__.get("_s7b_state")
    return st is not None and st.state in ABSENT_STATES


def state_block(front, env=None) -> dict:
    """``/pdflip/state`` addition: {} outside the dual layout (the state stays byte-identical there)."""
    if not getattr(front, "dual_layout", False):
        return {}
    st = front.__dict__.get("_s7b_state")
    snap = st.snapshot() if st is not None else PRestartState().snapshot()
    snap["armed"] = armed(front, env)
    snap["restart_armed"] = restart_armed(env)
    return {"p_restart": snap}


def health_reason(front, env=None) -> Optional[str]:
    """Review MUSS 5: the front's /health is non-200 while P is gone for good (FAILED, or no restart armed) --
    with X = 1 every new prompt is answered 503 then. None while P serves or may still come back."""
    if not p_absent(front, env):
        return None
    st = state_of(front)
    if st.state != FAILED:
        return None
    return "%s %s: %s" % (MARK, "P-ABSENT-NO-RESTART" if st.no_restart else "P-RESTART FAILED", st.note[:200])


def _fail_name(st) -> str:
    return "P-ABSENT-NO-RESTART" if st.no_restart else "P-RESTART-FAILED"


def _x(front) -> int:
    try:
        return int(getattr(front, "tp_prefill_max_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def d_capable(front, p) -> bool:
    """D may take this queued request: an SK-kept SHORT, or u <= X and not p_only (R9). Never over X."""
    if getattr(p, "skip_leg1", False):
        return True
    if getattr(p, "p_only", False) or getattr(p, "leg1_done", False):
        return False
    try:
        u = int(getattr(p, "est_uncached", -1))
    except (TypeError, ValueError):
        return False
    x = _x(front)
    return 0 <= u <= x and x > 0


def on_p_dead(front, why: str, *, now: Optional[float] = None, env=None,
              spawn_driver: Optional[Callable[[Any], Any]] = None) -> bool:
    """A site that would stop the boot for P calls this first. True = taken over (no STOP); False = the caller
    keeps its old path (not the dual layout, switch off, or the front already stopped)."""
    if not armed(front, env):
        return False
    if getattr(front, "state", None) == "STOP":
        return False
    st = state_of(front)
    now = time.time() if now is None else float(now)
    if st.state in ABSENT_STATES:
        if st.state == JOINING and "joining" not in str(why):
            st.note = f"while joining: {why}"[:300]
        return True
    groups = getattr(front, "groups", {}) or {}
    P = groups.get("P")
    st.state, st.why, st.t_dead = DEAD, str(why), now
    st.old_sid = int(getattr(P, "sid", 0) or 0)
    st.deaths += 1
    st.arena_dead = ARENA_CENSUS(front)   # review MUSS 6: the shared arena as the dead P left it
    inflight = dict(getattr(front, "_dual_inflight", None) or {})
    outstanding = len(getattr(P, "outstanding", {}) or {})
    queue = list(getattr(front, "queue", ()) or ())
    live = [p for p in queue if not getattr(getattr(p, "fut", None), "done", lambda: False)()]
    n_d = sum(1 for p in live if d_capable(front, p))
    rearm = restart_armed(env)
    front.counters["s7b_p_dead"] += 1
    logger.error(
        "%s P-DEAD why=%s p_sid=%s inflight_legs=%d p_outstanding=%d queue=%d d_capable=%d p_bound=%d X=%d "
        "restart=%s -- D serves on (no STOP); no P leg is dispatched; in-flight P legs end at P's transport "
        "failure and are requeued through the dual pause path (%s P-LEG-ABORTED); requests u<=X go to D's admission line, u>X / p_only wait "
        "for P (no P phase without P, D never prefills over X)",
        MARK, str(why)[:300], st.old_sid, len(inflight), outstanding, len(live), n_d, len(live) - n_d, _x(front),
        "armed" if rearm else "off (P-bound requests are answered at once)", MARK)
    for p in inflight.values():
        try:
            _mark_leg(front, p, "in flight at the death (%s)" % str(why)[:120])
        except Exception:  # noqa: BLE001 -- a partial Pending: its leg's own failure path still marks it
            pass
    hand_to_d(front)
    if rearm:
        st.state = RESTARTING
        if spawn_driver is not None:
            st.task = spawn_driver(front)
        else:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                st.task = loop.create_task(restart_driver(front, RealIO.from_env(front, env)))
    else:
        # review MUSS 5: P never comes back -- the same end state as a failed restart (front /health non-200)
        st.state, st.no_restart = FAILED, True
        st.note = "no restart armed (%s=0 or no respawn record in %s)" % (RESTART_ENV, SPEC_ENV)
        answer_p_bound(front, "P-ABSENT-NO-RESTART", "P is dead and " + st.note)
    _kick(front)
    return True


def _kick(front) -> None:
    kick = getattr(front, "_kick_controller", None)
    if callable(kick):
        try:
            kick("arrival")
        except Exception:  # noqa: BLE001 -- a kick only ends a wait early
            pass


def hand_to_d(front) -> int:
    """Every queued request D may take goes to D's admission line now (same form as DECODE-COLLECT to-d)."""
    q = getattr(front, "queue", None)
    if not q:
        return 0
    moved, tokens = [], 0
    for p in list(q):
        if getattr(getattr(p, "fut", None), "done", lambda: False)():
            continue
        if not d_capable(front, p):
            continue
        try:
            q.remove(p)
        except ValueError:
            continue
        if getattr(p, "skip_leg1", False):
            p.leg1_done = True        # an SK SHORT: D keeps it, no leg 1 (the drain's own path for it)
        else:
            p.d_direct = True         # leg 2 is the SHORT route's: D prefills u <= X
        tokens += max(0, int(getattr(p, "est_uncached", 0) or 0))
        front._ready_for_d.append(p)
        moved.append(p)
    if moved:
        sync = getattr(front, "_sync_batch_gate", None)
        if callable(sync):
            sync()
        st = state_of(front)
        st.to_d += len(moved)
        front.counters["s7b_to_d"] += len(moved)
        logger.warning("%s TO-D n=%d tokens=%d X=%d rids=%s state=%s -- P is absent: D's admission line takes "
                       "what D may prefill (u <= X)", MARK, len(moved), tokens, _x(front),
                       [getattr(p, "rid", "?") for p in moved][:8], st.state)
    return len(moved)


def answer_p_bound(front, name: str, detail: str, pred=None) -> int:
    """Answer (503, by name) every queued request only P can serve (and ``pred`` accepts). D-capable ones are left
    to ``hand_to_d``."""
    q = getattr(front, "queue", None)
    if not q:
        return 0
    n = 0
    for p in list(q):
        fut = getattr(p, "fut", None)
        if fut is None or fut.done() or d_capable(front, p):
            continue
        if pred is not None and not pred(p):
            continue
        try:
            q.remove(p)
        except ValueError:
            continue
        fut.set_exception(PdFlipPDead("%s %s" % (MARK, name), "rid=%s u=%s X=%d: %s" % (
            getattr(p, "rid", "?"), getattr(p, "est_uncached", "?"), _x(front), detail)))
        n += 1
        logger.error("%s %s rid=%s u=%s X=%d p_only=%s: %s -- answered 503 (D serves on)", MARK, name,
                     getattr(p, "rid", "?"), getattr(p, "est_uncached", "?"), _x(front),
                     int(bool(getattr(p, "p_only", False))), detail)
    if n:
        state_of(front).answered += n
        front.counters["s7b_answered"] += n
    return n


def _mark_leg(front, p, why: str) -> None:
    if getattr(p, "dual_pause", False):
        return
    p.dual_pause = True       # the dual pass's pause branch requeues it at the head (_dual_requeue_paused)
    p.s7b_requeues = int(getattr(p, "s7b_requeues", 0) or 0) + 1
    st = state_of(front)
    st.legs_aborted += 1
    front.counters["s7b_legs_aborted"] += 1
    logger.error("%s P-LEG-ABORTED rid=%s why=%s: P is dead -- the leg ends at P's transport failure and goes back "
                 "to the head of the queue through the dual pause path (u=%s X=%d -> %s); never failed to the "
                 "client, never handed to D half-prefilled", MARK, getattr(p, "rid", "?"), str(why)[:200],
                 getattr(p, "est_uncached", "?"), _x(front),
                 "D's admission line" if d_capable(front, p) else "waits for P")


def mark_failed_leg(front, p, exc: BaseException, *, env=None, p_looks_dead=None) -> bool:
    """A P leg failed: if P is dead (marked, or its session gone / the poller holds a dead or held fact), mark the
    leg for the dual pass's pause requeue. True = marked. Needs the card ledgers (the dispatch gate)."""
    if not armed(front, env) or not getattr(front, "dual_kv_ledgers", None):
        return False
    if getattr(p, "dual_pause", False):
        return True
    st = state_of(front)
    if st.state not in ABSENT_STATES:
        probe = p_looks_dead if p_looks_dead is not None else _p_looks_dead
        try:
            dead = bool(probe(front))
        except Exception:  # noqa: BLE001 -- no verdict = the old path (fail the request)
            dead = False
        if not dead:
            return False
        on_p_dead(front, "leg1 rid=%s failed (%s: %s) and P's group is dead" % (
            getattr(p, "rid", "?"), type(exc).__name__, str(exc)[:200]), env=env)
    _mark_leg(front, p, "%s: %s" % (type(exc).__name__, str(exc)[:160]))
    return True


async def leg1_rpc(front, p, coro, *, env=None):
    """``Front.leg1``'s RPC: await it; a transport failure or a non-200 while P is dead marks the leg (see
    ``mark_failed_leg``) before the caller raises as before."""
    try:
        status, body = await coro
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 -- marked, then re-raised unchanged
        mark_failed_leg(front, p, e, env=env)
        raise
    if status != 200 and getattr(front, "dual_layout", False):
        mark_failed_leg(front, p, RuntimeError("leg1 on P returned %s" % status), env=env)
    return status, body


def _p_looks_dead(front) -> bool:
    """P's session is gone, or the health poller holds a dead/held fact for P."""
    P = (getattr(front, "groups", {}) or {}).get("P")
    if P is None:
        return False
    _reap_respawned(front)   # review SOLL 5: an unreaped respawned leader reads 'alive' in ps
    f = getattr(P, "health_facts", None)
    if f is not None and (getattr(f, "hold", None) is not None or not getattr(f, "alive", True)):
        return True
    from flliper.srt.pdflip import front as _front_mod

    return not _front_mod._sid_alive(int(getattr(P, "sid", 0) or 0))


def _reap_respawned(front) -> Optional[int]:
    proc = front.__dict__.get("_s7b_proc")
    if proc is None:
        return None
    try:
        return proc.poll()
    except Exception:  # noqa: BLE001
        return None


def tick(front, now: Optional[float] = None, env=None) -> None:
    """The controller's step while P is absent: hand to D, bound the P-bound wait, one status line per 30 s."""
    env = os.environ if env is None else env
    now = time.time() if now is None else float(now)
    st = state_of(front)
    hand_to_d(front)
    if st.state == FAILED:
        answer_p_bound(front, _fail_name(st), "P does not come back: %s" % (st.note or st.why)[:200])
    else:
        # review SOLL 3: the wait is per request, from its arrival (or the death, whichever is later)
        wait = _num(env, WAIT_ENV, DEFAULT_WAIT_S)
        answer_p_bound(front, "P-WAIT-EXPIRED", "waited > %s=%.0f s for P (state=%s)" % (WAIT_ENV, wait, st.state),
                       pred=lambda p: now - max(float(getattr(p, "t_arrive", st.t_dead) or st.t_dead), st.t_dead) > wait)
    if now - st.last_status_t >= STATUS_EVERY_S:
        st.last_status_t = now
        q = [p for p in list(getattr(front, "queue", ()) or ()) if not p.fut.done()]
        logger.warning("%s P-ABSENT state=%s for_s=%.0f waiting_for_p=%d to_d=%d legs_aborted=%d answered=%d "
                       "restarts=%d new_pid=%s note=%s", MARK, st.state, now - st.t_dead, len(q), st.to_d,
                       st.legs_aborted, st.answered, st.restarts, st.new_pid or "-", st.note[:160] or "-")


def p_back(front, *, now: Optional[float] = None) -> None:
    """The barrier passed: reset the front's P-side state and serve with P again."""
    st = state_of(front)
    now = time.time() if now is None else float(now)
    P = front.groups["P"]
    P.outstanding.clear()
    P.health_fail_streak = 0
    P.health_facts = None
    for k in ("_dual_stages_obj", "_dual_task"):
        front.__dict__[k] = None                       # a fresh P ladder: p_state serving, nothing lent or asleep
    inflight = front.__dict__.get("_dual_inflight")
    if isinstance(inflight, dict):
        inflight.clear()
    fp_prev = front.__dict__.get("_fp_prev")
    if isinstance(fp_prev, dict):
        fp_prev.pop("P", None)
    front.__dict__["_dual_backoff_until"] = 0.0
    absent_s = now - st.t_dead
    st.state, st.restarts = SERVING, st.restarts + 1
    st.arena_back = ARENA_CENSUS(front)
    front.counters["s7b_p_back"] += 1
    logger.warning("%s P-BACK pid=%s absent_s=%.1f restarts=%d to_d=%d legs_aborted=%d answered=%d arena=%s -- P "
                   "serves again (health 200 + every rank's stage file rewritten after the spawn; R11: the front "
                   "decides, the ranks only signed in; arena = ARENA-REF-CENSUS at P-DEAD -> P-BACK: reader refs "
                   "and open writers of the dead P carry no pid and are not reaped, review MUSS 6)", MARK,
                   st.new_pid, absent_s, st.restarts, st.to_d, st.legs_aborted, st.answered,
                   arena_delta_text(st.arena_dead, st.arena_back))
    _kick(front)


# ---------------------------------------------------------------- review MUSS 6: the shared arena's census
def arena_paths(front, tags=None) -> List[str]:
    import glob

    tags = list(tags or [])
    for t in (str(getattr(front, "tag", "") or ""),):
        if t and t not in tags:
            tags.append(t)
    out: List[str] = []
    for t in tags:
        for p in sorted(glob.glob("/dev/shm/pdflip-arena-%s/*.bin" % t)):
            if p not in out:
                out.append(p)
    return out


def arena_census_of(path: str) -> Optional[Dict[str, int]]:
    """slots / complete / pinned (COMPLETE with a reader ref) / refs / claimed of ONE arena file, read-only through
    the front's lookup view (the same header read as ShmArena.ref_census). None = unreadable."""
    try:
        import ctypes

        import numpy as np

        from flliper.srt.mem_cache.storage.file.hicache_arena import ArenaView

        v = ArenaView(path)
        try:
            out = (ctypes.c_int64 * 6)()
            v._lib.arena_layout(v.slots, v.slot_bytes, out)
            hb, hope = int(out[0]), int(out[3])
            u32 = np.frombuffer(v._mm, dtype=np.uint32, count=v.slots * hb // 4, offset=hope)
            hdr = u32.reshape(v.slots, hb // 4)
            state, ref = hdr[:, 0], hdr[:, 1]
            res = {"slots": int(v.slots), "complete": int((state == 2).sum()),
                   "pinned": int(((state == 2) & (ref > 0)).sum()), "refs": int(ref.sum(dtype=np.int64)),
                   "claimed": int((state == 1).sum())}
            del u32, hdr, state, ref
            return res
        finally:
            v.close()
    except Exception:  # noqa: BLE001 -- an instrument never stops the front
        return None


def arena_census(front) -> Dict[str, Any]:
    tags = []
    spec = read_spec(str(os.environ.get(SPEC_ENV, "") or "")) or {}
    if spec.get("tag"):
        tags.append(str(spec["tag"]))
    return {os.path.basename(p): arena_census_of(p) for p in arena_paths(front, tags)}


#: replaceable in tests (the census reads /dev/shm)
ARENA_CENSUS = arena_census


def arena_delta_text(before, after) -> str:
    if not before and not after:
        return "none"
    parts = []
    for name in sorted(set(before or {}) | set(after or {})):
        a, b = (before or {}).get(name), (after or {}).get(name)
        if not a or not b:
            parts.append("%s:unreadable" % name)
            continue
        parts.append("%s:pinned %d->%d refs %d->%d claimed %d->%d complete %d->%d" % (
            name, a["pinned"], b["pinned"], a["refs"], b["refs"], a["claimed"], b["claimed"],
            a["complete"], b["complete"]))
    return "; ".join(parts)


# ---------------------------------------------------------------- stage 2: the respawn record and the driver
def respawn_path(gpu_arb: str, tag: str) -> str:
    return os.path.join(gpu_arb, "p_respawn_%s.json" % str(tag))


def remove_spec(gpu_arb: str, tag: str) -> str:
    """Review SOLL 7: the record dies with the boot (launcher --teardown)."""
    path = respawn_path(gpu_arb, tag)
    try:
        os.unlink(path)
        return "removed %s" % path
    except FileNotFoundError:
        return "none"
    except OSError as e:
        return "not removed (%s)" % e


def _secret_name(name: str) -> bool:
    import re

    return re.search(SECRET_NAME_RX, str(name), re.IGNORECASE) is not None


def mask_argv(argv: List[str]) -> List[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            out.append(MASK)
            skip = False
            continue
        hit = next((f for f in SECRET_ARGV_FLAGS if a == f or a.startswith(f + "=")), None)
        if hit is None:
            out.append(a)
        elif a == hit:
            out.append(a)
            skip = True
        else:
            out.append(hit + "=" + MASK)
    return out


def mask_env(env: Dict[str, str]) -> Dict[str, str]:
    return {k: (MASK if _secret_name(k) and str(v) != "" else str(v)) for k, v in env.items()}


def unmask(spec: dict, front_env: Dict[str, str], admin_key: str) -> "tuple[List[str], Dict[str, str], List[str]]":
    """The exec record with its secrets put back: env values from the front's own environment (the launcher built
    both from one os.environ), the admin key from the front's. Returns (argv, env, names not restorable)."""
    missing: List[str] = []
    env = {}
    for k, v in (spec.get("env") or {}).items():
        if v == MASK:
            if k in front_env:
                env[k] = front_env[k]
            else:
                missing.append(k)
        else:
            env[k] = v
    argv = []
    for a in spec.get("argv") or []:
        if a == MASK or a.endswith("=" + MASK):
            if not admin_key:
                missing.append("argv:" + a)
                argv.append(a)
                continue
            argv.append(admin_key if a == MASK else a[: -len(MASK)] + admin_key)
        else:
            argv.append(a)
    return argv, env, missing


def rewrite_boot_pids(path: str, group: str, pid: int) -> str:
    """Review SOLL 4: the launcher's boot json (what --teardown reads) names the respawned P."""
    try:
        with open(path) as f:
            st = json.load(f)
        old = (st.get("pids") or {}).get(group)
        st.setdefault("pids", {})[group] = int(pid)
        tmp = "%s.tmp.s7b.%d" % (path, os.getpid())
        with open(tmp, "w") as f:
            json.dump(st, f, indent=1, default=str)
        os.replace(tmp, path)
        return "pids.%s %s -> %d in %s" % (group, old, int(pid), path)
    except (OSError, ValueError) as e:
        return "pids.%s not rewritten (%s: %s)" % (group, type(e).__name__, e)


def hold_boot(spec: dict) -> "tuple[bool, float]":
    """Review MUSS 4: is P's boot a #1223 DEBUG-HOLD boot, and how long does a held rank hold."""
    env = spec.get("env") or {}
    on = any(str(env.get(k, "")) == "1" for k in HOLD_ENVS)
    hold_s = DEFAULT_HOLD_S
    for k in HOLD_S_ENVS:
        try:
            if env.get(k):
                hold_s = float(env[k])
                break
        except (TypeError, ValueError):
            pass
    return on, hold_s


def end_session_pids(sid: int, *, session_pids, terminate, kill_tree, kill_pid, alive, own_sid: int) -> str:
    """Review MUSS 2: end EVERY process of P's old session -- MPS terminate_client first (#210), then the leader's
    tree, then SIGKILL for each pid of the session still alive (a dead leader leaves orphaned schedulers that
    kill_process_tree never sees: it returns at the leader's NoSuchProcess). Never the front's own session."""
    if int(sid) <= 1 or int(sid) == int(own_sid):
        return "refused: sid=%s (own sid %s)" % (sid, own_sid)
    pids = sorted(int(p) for p in session_pids(int(sid)))
    done = terminate(pids) if pids else []
    tree = kill_tree(int(sid))
    left = [p for p in pids if alive(p)]
    for p in left:
        kill_pid(p)
    return "pids=%s mps_terminated=%s kill_process_tree=%s sigkill=%s" % (pids, sorted(done or []), tree, left)


def write_spec(path: str, *, argv: List[str], env: Dict[str, str], cwd: str, log: str, port: int, tag: str,
               pid: int, card_need_mib: Optional[Dict[int, int]] = None) -> str:
    """The launcher's exec record of P (the argv/env it actually exec'd), mode 0600. Secrets are masked (SOLL 7) and
    put back from the front's environment at the spawn. ``card_need_mib``: P's PLANNED peak per NVML card (budget +
    overhead, the number D was sized against -- review MUSS 1); the card gate checks exactly this."""
    rec = {"schema": SPEC_SCHEMA, "group": "P", "argv": mask_argv(list(argv)), "env": mask_env(dict(env)),
           "cwd": str(cwd), "log": str(log), "port": int(port), "tag": str(tag), "pid": int(pid),
           "card_need_mib": {str(int(k)): int(v) for k, v in (card_need_mib or {}).items()},
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return path


def _flag_last(argv: List[str], flag: str) -> Optional[str]:
    val = None
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            val = argv[i + 1]
        elif a.startswith(flag + "="):
            val = a[len(flag) + 1:]
    return val


def spec_card_need(spec: dict) -> Dict[int, int]:
    """P's planned peak per NVML card from the record (review MUSS 1). {} = not recorded."""
    try:
        return {int(k): int(v) for k, v in (spec.get("card_need_mib") or {}).items()}
    except (TypeError, ValueError):
        return {}


def p_ranks(argv: List[str], default: int) -> int:
    """Review SOLL 6: P's rank count from its own argv (--pp-size x --tp-size)."""
    try:
        pp = int(_flag_last(list(argv), "--pp-size") or 0)
        tp = int(_flag_last(list(argv), "--tp-size") or 1)
    except ValueError:
        return default
    return pp * max(1, tp) if pp > 0 else default


def card_room_verdict(need: Dict[int, int], free: Dict[int, int]) -> "tuple[bool, str]":
    """Pure: every card's allocatable free >= P's planned peak there (no reserve on top). No recorded need = no
    spawn (review MUSS 1: the argv budget alone is 2500 MiB/card short of P's plan)."""
    if not need:
        return False, "no planned P peak in the respawn record (card_need_mib) -- no spawn on a guess"
    if not free:
        return False, "NVML unreadable"
    short = ["nvml%d free=%s<need=%d" % (c, free.get(c, "?"), m) for c, m in sorted(need.items())
             if int(free.get(c, -1)) < int(m)]
    if short:
        return False, " ".join(short)
    return True, " ".join("nvml%d free=%d>=need=%d" % (c, free[c], m) for c, m in sorted(need.items()))


def read_spec(path: str) -> Optional[dict]:
    if not path:
        return None
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or rec.get("schema") != SPEC_SCHEMA or not rec.get("argv"):
        return None
    if not isinstance(rec.get("env"), dict) or not isinstance(rec["argv"], list):
        return None
    return rec


def _pid_live(pid: int) -> bool:
    """A process exists and is not a zombie."""
    try:
        with open("/proc/%d/stat" % int(pid)) as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


class RealIO:
    """The driver's effects on the real rig. Tests hand the driver a fake with the same methods."""

    def __init__(self, front, spec: dict, env):
        self.front, self.spec, self.env = front, spec, env

    @classmethod
    def from_env(cls, front, env=None) -> "RealIO":
        env = os.environ if env is None else env
        return cls(front, read_spec(str(env.get(SPEC_ENV, "") or "")) or {}, env)

    def now(self) -> float:
        return time.time()

    async def sleep(self, s: float) -> None:
        await asyncio.sleep(s)

    def session_alive(self, sid: int) -> bool:
        """Any non-zombie process of the session (SOLL 5: the respawned leader is reaped first)."""
        _reap_respawned(self.front)
        if not sid:
            return False
        from flliper.srt.pdflip import front as _front_mod

        return any(_pid_live(p) for p in _front_mod._session_pids(int(sid)))

    async def end_session(self, sid: int) -> str:
        def _run() -> str:
            import signal

            from flliper.srt.pdflip import front as _front_mod
            from flliper.srt.pdflip import mps_exit_quiesce as _s7q

            pipe = str((self.spec.get("env") or {}).get("CUDA_MPS_PIPE_DIRECTORY", "") or "").strip()

            def terminate(pids):
                if not pipe:
                    return []
                spec_env = dict(os.environ)
                spec_env["CUDA_MPS_PIPE_DIRECTORY"] = pipe
                return _s7q.terminate_clients(pids, why="s7b_restart",
                                              ctl=lambda c: _s7q.control(c, env=spec_env, timeout_s=5.0))

            def kill_tree(leader):
                try:
                    from flliper.srt.utils.common import kill_process_tree

                    kill_process_tree(int(leader), include_parent=True, wait_timeout=30)
                    return "ok"
                except Exception as e:  # noqa: BLE001 -- named in the line, the per-pid kill follows
                    return "%s: %s" % (type(e).__name__, str(e)[:120])

            def kill_pid(p):
                try:
                    os.kill(int(p), signal.SIGKILL)
                except OSError:
                    pass

            return end_session_pids(int(sid), session_pids=_front_mod._session_pids, terminate=terminate,
                                    kill_tree=kill_tree, kill_pid=kill_pid, alive=_pid_live, own_sid=os.getsid(0))

        return await asyncio.to_thread(_run)

    def card_room(self) -> "tuple[bool, str]":
        from flliper.srt.pdflip import front as _front_mod

        need = spec_card_need(self.spec)
        free = {int(c.nvml_index): int(c.free_mib) for c in _front_mod._nvml_free()} if need else {}
        return card_room_verdict(need, free)

    def exec_record(self) -> "tuple[List[str], Dict[str, str], List[str]]":
        return unmask(self.spec, dict(os.environ), str(getattr(self.front, "admin_key", "") or ""))

    def spawn(self) -> int:
        spec = self.spec
        argv, env, missing = self.exec_record()
        if any(m.startswith("argv:") for m in missing):
            raise RuntimeError("the masked admin key cannot be put back (the front has none): %s" % missing)
        if missing:
            logger.warning("%s P-RESTART env secrets not in the front's environment, left out: %s", MARK, missing)
        fh = open(spec["log"], "ab")
        fh.write(("=== PDFLIP group P RE-launched by the front (%s) %s ===\n" % (
            MARK, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))).encode())
        fh.flush()
        p = subprocess.Popen(argv, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             cwd=spec.get("cwd") or None, start_new_session=True)
        self.front.__dict__["_s7b_proc"] = p
        try:
            # the arm's group stop (H135b registry) and --teardown's boot json (SOLL 4) name the new P
            from flliper.srt.pdflip import launcher as _launcher

            tag = str(spec.get("tag", "") or "")
            logger.warning("%s %s", MARK, _launcher.register_boot_group(tag, "P", int(p.pid)))
            logger.warning("%s %s", MARK, rewrite_boot_pids(
                _launcher._operator_dir(_launcher.GPU_ARB, "boot_%s.json" % tag), "P", int(p.pid)))
        except Exception as e:  # noqa: BLE001 -- a lost entry costs the teardown its pid, not the restart
            logger.warning("%s P-RESTART registry / boot json not written: %s: %s", MARK, type(e).__name__, e)
        return int(p.pid)

    def spawned_exited(self) -> Optional[int]:
        return _reap_respawned(self.front)

    async def health_ok(self) -> bool:
        P = self.front.groups["P"]
        return bool(await self.front._probe_group_health(P, 8.0))

    def ranks_signed_in(self, t_spawn: float) -> Optional[bool]:
        """Every P rank rewrote its dual stage file after the spawn. None = this boot has no stage files."""
        try:
            from flliper.srt.pdflip import dual_p_kv_stage as _pk
        except Exception:  # noqa: BLE001
            return None
        if not getattr(self.front, "dual_kv_ledgers", None):
            return None
        tags = []
        for t in (str(getattr(self.front, "tag", "") or ""), str(self.spec.get("tag", "") or ""),
                  self.env.get("FLLIPER_PDFLIP_DUAL_KV_TAG", "") or self.env.get("FLLIPER_PDFLIP_TAG", "pdflip")):
            if t and t not in tags:
                tags.append(t)
        for r in range(p_ranks(list(self.spec.get("argv") or []), _pk.LEND_GATE_RANKS)):
            fresh = False
            for t in tags:
                try:
                    if os.path.getmtime(_pk.stage_file(t, r)) > t_spawn:
                        fresh = True
                        break
                except OSError:
                    continue
            if not fresh:
                return False
        return True


async def restart_driver(front, io, env=None) -> str:
    """Stage 2: end the old session, respawn P from the launcher's record, wait for the R11 barrier, P back.
    Returns the final state. Never raises (a failure is the named state ``failed``)."""
    env = os.environ if env is None else env
    st = state_of(front)
    try:
        if not getattr(io, "spec", None):
            return _fail(front, "no respawn record (%s)" % SPEC_ENV)
        if st.restarts >= int(_num(env, MAX_ENV, DEFAULT_MAX)):
            return _fail(front, "restart budget spent (%d of %s=%d)" % (
                st.restarts, MAX_ENV, int(_num(env, MAX_ENV, DEFAULT_MAX))))
        grace = _num(env, GRACE_ENV, DEFAULT_GRACE_S)
        hold, hold_s = hold_boot(io.spec)
        if hold:
            # review MUSS 4 (#1223, user order 06.09.): a held P rank is never killed -- the live attach at the wall is
            # what a hold boot is for. The old session ends by itself when the hold ends; D serves meanwhile.
            grace = grace + hold_s
            logger.warning("%s P-RESTART hold boot (#1223): the old P session %s is not killed; waiting up to %.0f s "
                           "(hold %.0f s + grace) for it to end by itself", MARK, st.old_sid, grace, hold_s)
        t0 = io.now()
        while io.session_alive(st.old_sid) and io.now() - t0 < grace:
            if getattr(front, "state", None) == "STOP":
                return _fail(front, "front stopped while the old P session ended")
            await io.sleep(1.0)
        if io.session_alive(st.old_sid) and hold:
            return _fail(front, "hold boot: old P session %s still alive after %.0f s -- not killed (#1223)" % (
                st.old_sid, grace))
        if io.session_alive(st.old_sid):
            how = await io.end_session(st.old_sid)
            logger.warning("%s P-RESTART old session %s outlived the grace %.0f s: %s", MARK, st.old_sid, grace, how)
            t1 = io.now()
            while io.session_alive(st.old_sid) and io.now() - t1 < 30.0:
                await io.sleep(1.0)
            if io.session_alive(st.old_sid):
                return _fail(front, "old P session %s still alive after terminate + kill of every pid" % st.old_sid)
        cwait = _num(env, CARD_WAIT_ENV, DEFAULT_CARD_WAIT_S)
        t2 = io.now()
        while True:
            room, detail = io.card_room()
            if room:
                break
            if io.now() - t2 >= cwait:
                return _fail(front, "card room for P's boot not free within %s=%.0f s: %s" % (CARD_WAIT_ENV, cwait, detail))
            if getattr(front, "state", None) == "STOP":
                return _fail(front, "front stopped while P waited for card room")
            await io.sleep(5.0)
        st.note = "card room: %s" % detail
        if getattr(front, "state", None) == "STOP":
            return _fail(front, "front stopped while P was away")
        st.t_spawn = io.now()
        st.new_pid = int(io.spawn())
        front.groups["P"].sid = st.new_pid
        st.state = JOINING
        logger.warning("%s P-RESTART spawned pid=%d (session = pid) from the launcher's record (same argv/env: same "
                       "cards, same planner cut) after old session %s ended; %s; waiting for health 200 + every "
                       "rank's stage file (R11 barrier)", MARK, st.new_pid, st.old_sid, st.note)
        ready = _num(env, READY_ENV, DEFAULT_READY_S)
        while io.now() - st.t_spawn < ready:
            rc = io.spawned_exited()
            if rc is not None:
                return await _fail_spawned(front, io, "respawned P exited rc=%s during its boot" % rc)
            if getattr(front, "state", None) == "STOP":
                return await _fail_spawned(front, io, "front stopped while P was joining")
            if await io.health_ok():
                signed = io.ranks_signed_in(st.t_spawn)
                if signed is None or signed:
                    p_back(front)
                    return SERVING
            await io.sleep(2.0)
        return await _fail_spawned(front, io, "P not back within %s=%.0f s (health or a rank's stage file missing)"
                                   % (READY_ENV, ready))
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 -- the driver never takes the front down
        why = "driver raised %s: %s" % (type(e).__name__, str(e)[:200])
        if st.state == JOINING and st.new_pid:
            try:
                return await _fail_spawned(front, io, why)
            except Exception:  # noqa: BLE001
                pass
        return _fail(front, why)


async def _fail_spawned(front, io, why: str) -> str:
    """Review MUSS 3: a failure after the spawn ends the respawned session (the poller skips an absent P and
    --teardown may not know it) -- except in a hold boot, where a held new rank stays for the attach (MUSS 4)."""
    st = state_of(front)
    if hold_boot(io.spec)[0]:
        return _fail(front, "%s -- respawned session %s left standing (hold boot, #1223)" % (why, st.new_pid))
    try:
        how = await io.end_session(st.new_pid)
    except Exception as e:  # noqa: BLE001 -- named in the line
        how = "end_session raised %s: %s" % (type(e).__name__, e)
    return _fail(front, "%s -- respawned session %s ended: %s" % (why, st.new_pid, how))


def _fail(front, why: str) -> str:
    st = state_of(front)
    st.state, st.note = FAILED, str(why)
    front.counters["s7b_restart_failed"] += 1
    logger.error("%s P-RESTART FAILED: %s -- P stays absent, D serves on; P-bound requests are answered by name",
                 MARK, why)
    answer_p_bound(front, _fail_name(st), why[:200])
    return FAILED
