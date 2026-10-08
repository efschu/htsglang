"""Q-697b DUAL GRANT-WAIT: a dual P rank whose only queued work is legs that wait
for their card grant gives its KV context back, the front stops pausing those
legs, and PP0 takes no new grant while D's demand stands -- one change in three
parts, because each part alone is wrong.

METAL (27B NVFP4 dual y8x/y9, desk analysis 1010 section 2.4): pdflip-0-44 waited
181-184 s. 22:04:37 GRANT -> 22:04:39 P-PAUSE (D-GROW for the hand-back of -19)
-> 22:04:42 resent, GRANT, 0.2 s later paused again (GRANT-RETURN why=abort) ->
22:04:43 WAIT 55 s -> 22:05:38 P-PAUSE although -44 held nothing ('PP0 WAIT ...
nothing held' since 22:04:43). The second instances of a rid that this
abort/resend loop makes are the ones Q-697's A-death happened on.

THE CAUSE, at the code (all three facts are in the tree at c2f6e819cb):

  * A leg whose group grant is short is HELD, not refused: ``pdflip_store_told.intake``
    returns ``declined:dual_kv_wait`` and the request STAYS in ``waiting_queue``
    (``_dual_kv_wait``; ``_dual_kv_retry`` retries it at the top of every PP0 pass).
    On the followers the same request sits in their waiting queue too (PP0 puts it
    on the ring, the told never comes while PP0 holds it).
  * P hands its mapping back only at FULL idle: ``Scheduler.on_idle`` returns before
    ``dual_p_kv_stage.on_idle`` while ``waiting_queue`` is not empty
    (``is_fully_idle``). A leg that holds NOTHING therefore keeps every other leg's
    leftover mapping -- and D's pressure on that card -- alive.
  * The front's answer to pressure is ``_dual_pause_inflight``: abort EVERY leg 1 in
    flight on P, grant-less ones included, so that P can reach full idle. The
    aborted leg is requeued at the head and waits for ``_dual_resume_held`` (all
    zeros) -- 55 s and more, then it is sent again, gets a grant, meets the next
    D-GROW and is paused again.

WHY THREE PARTS (the first draft, discarded: 'front stops pausing grant-less legs'
alone, deadlocks: the waiter keeps P from full idle, so P never releases, D waits
for P's pressure answer, nobody moves):

  1. RELEASE (P ranks, ``release_for_grant_waiters``, from ``Scheduler.on_idle``
     when ``is_fully_idle`` is false): when EVERYTHING that makes the rank not
     idle is the waiting queue, and every queued rid is a leg PP0 holds for its
     grant, the rank takes the normal idle path (``dual_p_kv_stage.on_idle``: PP0's
     idle stamp for the followers' held aborts, evict, ``release_all``). PP0 knows
     its waiters (``_dual_kv_wait``); a follower learns them from PP0's marker
     (``publish`` / ``GrantWaitState``, same /dev/shm pattern as RO's reading set).
     A missing or stale marker names no waiter: the rank behaves as before.
  2. FRONT (``Front._dual_pause_inflight``): a leg named in the marker is not
     paused. It holds no KV on P; pausing it frees nothing and only starts the
     abort -> requeue -> RESUME-WAIT -> resend loop. Not while P's stage is
     'sleeping' (stage 2 needs P quiescent); everything else is paused as before.
  3. GRANT HOLD (``pp0_grant``): a waiter does not take bytes while a card shows D's
     unmet demand or pressure on P. Without this the released bytes go to the first
     waiter before D's tick takes them: the waiter is granted, paused, released,
     granted... and D never gets them. The front's own gate for a PAUSED head is
     ``card_kv_ledger.p_resume_ready`` (pressure 0, P committed 0, D demand 0); an
     unpaused waiter had no such gate -- this is it, bounded (never wedges P on a
     stale ledger).

Dual layout + group P only (``dual_decode_join.dual_layout_rank``). Switch
``FLLIPER_PDFLIP_DUAL_GRANT_WAIT`` (default on; 0/false/off/no = the pre-Q-697b
behaviour, byte for byte). The flip / INT8 / NF forms return before reading a
scheduler attribute, a file or a ledger (the 'Flip unveraendert' test).

Stdlib only (the front imports it).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Iterable, Optional, Set

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_DUAL_GRANT_WAIT"
ENV_HOLD_S = "FLLIPER_PDFLIP_DUAL_GRANT_HOLD_S"
HOLD_S_DEFAULT = 30.0
#: a marker older than this names no waiter (PP0 rewrites it at least every HEARTBEAT_S)
FRESH_S = 30.0
HEARTBEAT_S = 1.0
#: at most one marker-write failure line per this many seconds
WARN_EVERY_S = 30.0
FILE_FMT = "pdflip_p_grantwait_%d.json"

RELEASE_MARK = "Q-697b DUAL GRANT-WAIT RELEASE"
SKIP_MARK = "Q-697b DUAL P-PAUSE-SKIP"
HOLD_MARK = "Q-697b DUAL GRANT-HOLD"

#: rate limit of the named lines: the first LOG_FIRST, then every LOG_EVERY-th
LOG_FIRST = 16
LOG_EVERY = 64


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    return str(e.get(ENV, "") or "").strip().lower() not in ("0", "false", "no", "off")


def _group_p(env=None) -> bool:
    e = os.environ if env is None else env
    return str(e.get("FLLIPER_PDFLIP_GROUP", "") or "").strip().upper() == "P"


def dual_p(env=None) -> bool:
    """Dual layout + group P + the switch. Nothing else is read before this is True."""
    from flliper.srt.pdflip.dual_decode_join import dual_layout_rank

    return dual_layout_rank(env) and _group_p(env) and enabled(env)


def state_path(port) -> str:
    from flliper.srt.pdflip import p_read_overlap as _ro

    base = os.environ.get(_ro.ENV_DIR, "") or _ro.DIR_DEFAULT
    return os.path.join(base, FILE_FMT % int(port))


def _scheduler_port(sched) -> Optional[int]:
    sa = getattr(sched, "server_args", None)
    try:
        port = getattr(sa, "port", None)
        return int(port) if port is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# PP0 side: the waiters' marker
# ---------------------------------------------------------------------------

#: module state of the writer: None until this process ever named a waiter
_PUB: dict = {"rids": None, "t": 0.0}


def _reset_for_tests() -> None:
    _PUB.clear()
    _PUB.update(rids=None, t=0.0)
    _HOLD_T0.clear()
    _P_HELD["t"] = None


def publish(sched, rids: Iterable[Any], now: Optional[float] = None, env=None) -> bool:
    """PP0, once per pass (``pdflip_store_told._dual_kv_retry``): write the set of
    rids this rank holds for their card grant -- when it changed, and as a
    heartbeat every HEARTBEAT_S while it is not empty. True = a file was written.
    A process that never named a waiter returns here without touching anything
    (every flip / INT8 / NF form). Never raises into the pass."""
    rids = frozenset(str(r) for r in rids)
    if not rids and _PUB["rids"] is None:
        return False
    try:
        if not dual_p(env):
            return False
        if int(getattr(getattr(sched, "ps", None), "pp_rank", 0) or 0) != 0:
            return False
        port = _scheduler_port(sched)
        if port is None:
            return False
        t = time.time() if now is None else float(now)
        changed = rids != _PUB["rids"]
        if not changed and (not rids or t - float(_PUB["t"]) < HEARTBEAT_S):
            return False
        path = state_path(port)
        tmp = "%s.%d.tmp" % (path, os.getpid())
        with open(tmp, "w") as fh:
            json.dump({"rids": sorted(rids), "pid": os.getpid(), "t": t}, fh)
        os.replace(tmp, path)
        _PUB["rids"], _PUB["t"] = rids, t
        return True
    except Exception as e:  # noqa: BLE001 - an instrument never breaks the pass
        # rate-limited: PP0 calls this every pass, a full /dev/shm must not flood the log
        tm = time.monotonic()
        if tm - float(_PUB.get("warn_t", -1e9)) >= WARN_EVERY_S:
            _PUB["warn_t"] = tm
            logger.warning("%s marker write failed: %r (further failures within %.0fs are not logged)",
                           RELEASE_MARK, e, WARN_EVERY_S)
        return False


class GrantWaitState:
    """Reader of PP0's marker (the front, and the P followers). mtime-cached;
    only a FRESH marker names waiters -- a missing, unreadable, foreign or old
    file names none (the pre-Q-697b behaviour)."""

    def __init__(self, port) -> None:
        self.path = state_path(port)
        self._key = None
        self._t = 0.0
        self._rids: frozenset = frozenset()

    def rids(self, now: Optional[float] = None) -> frozenset:
        try:
            st = os.stat(self.path)
        except OSError:
            self._key, self._t, self._rids = None, 0.0, frozenset()
            return self._rids
        key = (st.st_mtime_ns, st.st_size, st.st_ino)
        if key != self._key:
            try:
                with open(self.path) as fh:
                    js = json.load(fh)
                self._rids = frozenset(str(r) for r in (js.get("rids") or ()))
                self._t = float(js.get("t") or 0.0)
            except Exception:  # noqa: BLE001 - half a file or none: no waiter
                self._rids, self._t = frozenset(), 0.0
            self._key = key
        t = time.time() if now is None else float(now)
        if t - self._t > FRESH_S:
            return frozenset()
        return self._rids


def _follower_state(sched) -> Optional[GrantWaitState]:
    st = getattr(sched, "_dual_gw_state", None)
    if st is None:
        port = _scheduler_port(sched)
        if port is None:
            return None
        st = GrantWaitState(port)
        sched._dual_gw_state = st
    return st


# ---------------------------------------------------------------------------
# part 1: the release
# ---------------------------------------------------------------------------

_N = {"release": 0}


def grant_waiters(sched) -> Set[str]:
    """The rids this rank knows to be held for their card grant: PP0 reads its
    own held map, a follower reads PP0's marker."""
    if int(getattr(getattr(sched, "ps", None), "pp_rank", 0) or 0) == 0:
        held = getattr(sched, "_pdflip_store_held", None) or {}
        return {str(getattr(r, "rid", "")) for r in held.values() if getattr(r, "_dual_kv_wait", False)}
    st = _follower_state(sched)
    return set(st.rids()) if st is not None else set()


def _idle_but_queue(sched) -> bool:
    """``Scheduler.is_fully_idle`` with the waiting queue taken out of it: every
    OTHER term (batches, chunked request, microbatches, HiCache in-flight ops...)
    must hold. The queue is put back whatever happens."""
    saved = sched.waiting_queue
    sched.waiting_queue = []
    try:
        return bool(sched.is_fully_idle())
    finally:
        sched.waiting_queue = saved


def only_grant_waiters_queued(sched) -> bool:
    """True when the waiting queue is not empty, every queued rid is a leg held for
    its grant, and nothing else keeps the rank from being idle."""
    queue = list(getattr(sched, "waiting_queue", None) or ())
    if not queue:
        return False
    waiters = grant_waiters(sched)
    if not waiters:
        return False
    if not {str(getattr(r, "rid", "")) for r in queue} <= waiters:
        return False
    return _idle_but_queue(sched)


def release_for_grant_waiters(sched, env=None) -> int:
    """From ``Scheduler.on_idle`` in the branch where ``is_fully_idle`` said no. On a
    dual P rank whose only queued work is grant-waiting legs: the normal idle path
    (PP0's idle stamp, device tree evicted, mapping released). Returns the bytes
    given back; 0 off the dual P layout (nothing read) or with anything else live."""
    if not dual_p(env):
        return 0
    from flliper.srt.pdflip import dual_p_kv_stage as _dpk

    if _dpk._actor(sched) is None:
        return 0
    if not only_grant_waiters_queued(sched):
        return 0
    actor = _dpk._actor(sched)
    before = int(getattr(actor, "mapped_tokens", 0) or 0)
    # R2: the marker names the waiters BEFORE PP0's idle stamp goes out. A request PP0 took in
    # this very loop iteration is named by _dual_kv_retry only at the top of the NEXT one; a
    # follower reading the stamp in between would not see its rid in the marker. (No-op off PP0.)
    publish(sched, grant_waiters(sched))
    n = int(_dpk.on_idle(sched) or 0)
    if before > 0 and n > 0:
        _N["release"] += 1
        c = _N["release"]
        if c <= LOG_FIRST or c % LOG_EVERY == 0:
            logger.info(
                "%s pp_rank=%s waiters=%s mapped=%d->0 released=%d B n=%d: the only queued work is "
                "legs that wait for their card grant (they hold no KV here) -- the context goes back "
                "to the card pool as at full idle (Q-697b)",
                RELEASE_MARK, getattr(getattr(sched, "ps", None), "pp_rank", "?"),
                sorted(grant_waiters(sched))[:4], before, n, c)
    return n


def conflicting_rids(sched, held_rids: Iterable[Any]) -> Set[str]:
    """Follower guard for ``follower_release_aborted_chunk``: the held-abort rids that are
    ALSO grant-waiting legs of PP0 (the front sent the rid again). Such a hold is not applied
    now -- its verdict reads the rid, not the object, and would take the new instance with it
    (the Q-697 zombie class). Only with a fresh marker; the other rids are not touched."""
    if not dual_p():
        return set()
    st = _follower_state(sched)
    if st is None:
        return set()
    return {str(r) for r in held_rids} & set(st.rids())


def same_rid_waits(sched, held_rids: Iterable[Any]) -> bool:
    return bool(conflicting_rids(sched, held_rids))


# ---------------------------------------------------------------------------
# part 3: the grant hold
# ---------------------------------------------------------------------------

def hold_s(env=None) -> float:
    e = os.environ if env is None else env
    try:
        return max(0.0, float(e.get(ENV_HOLD_S, "") or HOLD_S_DEFAULT))
    except ValueError:
        return HOLD_S_DEFAULT


#: rid -> first time its grant was held for D (the clock belongs to the RID: a resent
#: instance does not start a new 30 s), cleared when no demand stands any more
_HOLD_T0: dict = {}
#: monotonic time P was last seen holding bytes on a card (None = never)
_P_HELD: dict = {"t": None}
#: after P was seen holding bytes, the hold still applies this long (P just released them
#: and D's tick has not claimed them yet); beyond it a stale demand costs a request nothing
GRACE_S = 5.0


def d_demand_stands(ledger_paths: Iterable[str]):
    """``(line, p_committed)``: a card line when D's unmet demand or a pressure on P stands
    on any of the stage ledgers (the signals of ``p_resume_ready``), else None; and the sum
    of P's committed bytes over the cards read."""
    from flliper.srt.pdflip.card_kv_ledger import peek

    line, p_committed = None, 0
    for pth in ledger_paths or ():
        try:
            st = peek(pth)
        except Exception:  # noqa: BLE001 - a ledger read never decides a hold
            continue
        if st is None:
            continue
        p_committed += int(st.committed.get("P", 0) or 0)
        pressure = int(st.pressure.get("P", 0) or 0)
        demand = int(st.demand.get("D", 0) or 0)
        if line is None and (pressure > 0 or demand > 0):
            line = "pressure_on_P=%d d_demand=%d card=%s" % (pressure, demand, os.path.basename(str(pth))[-24:])
    return line, p_committed


def grant_held_by_d(req, stages, now: Optional[float] = None, env=None) -> Optional[str]:
    """PP0, ``pp0_grant`` before the group grant: a reason string when this request's grant
    is held back because a card shows D's demand / pressure AND P holds bytes on a card (or
    held them within GRACE_S: the bytes it just released are D's first), None when the grant
    may be tried. With nothing to release and no recent release a standing demand is D's
    own business (stale or not) and costs P nothing. Bounded per RID: at most ``hold_s``
    (a stale ledger never wedges P). Nothing is read off the dual P layout."""
    if not dual_p(env):
        return None
    why, p_committed = d_demand_stands([s.get("ledger") for s in (stages or ())])
    t = time.monotonic() if now is None else float(now)
    if p_committed > 0:
        _P_HELD["t"] = t
    rid = str(getattr(req, "rid", ""))
    if why is None:
        _HOLD_T0.pop(rid, None)
        return None
    last = _P_HELD["t"]
    if p_committed <= 0 and (last is None or t - float(last) > GRACE_S):
        return None
    t0 = _HOLD_T0.get(rid)
    if t0 is None:
        if len(_HOLD_T0) > 4096:
            _HOLD_T0.clear()
        _HOLD_T0[rid] = t0 = t
    if t - float(t0) >= hold_s(env):
        return None
    return why


def note_hold(rid: str, why: str, req) -> None:
    _N["hold"] = _N.get("hold", 0) + 1
    c = _N["hold"]
    if c <= LOG_FIRST or c % 256 == 0:
        t0 = _HOLD_T0.get(str(rid))
        logger.info("%s rid=%s waited=%.1fs %s n=%d: D's demand stands on a card -- no grant now, the "
                    "bytes P released are D's first (Q-697b; bounded by %s=%.0fs)",
                    HOLD_MARK, rid, 0.0 if t0 is None else time.monotonic() - float(t0), why, c, ENV_HOLD_S,
                    hold_s())


# ---------------------------------------------------------------------------
# part 2: the front
# ---------------------------------------------------------------------------

def front_skip_set(front, now: Optional[float] = None) -> frozenset:
    """Front, ``_dual_pause_inflight``: the in-flight rids that must not be paused.
    Empty off the dual layout (the marker is not even looked at) and while P's
    stage is 'sleeping' (stage 2 needs P quiescent)."""
    if not getattr(front, "dual_layout", False) or not enabled():
        return frozenset()
    stages = getattr(front, "_dual_stages_obj", None)
    if stages is not None and getattr(stages, "p_state", "serving") == "sleeping":
        return frozenset()
    st = getattr(front, "_dual_gw_state", None)
    if st is None:
        try:
            import urllib.parse

            port = urllib.parse.urlparse(front.groups["P"].url).port
            if port is None:
                return frozenset()
            st = front._dual_gw_state = GrantWaitState(port)
        except Exception:  # noqa: BLE001 - no marker reader: pause as before
            return frozenset()
    return st.rids(now)


def note_skip(front, rid: str, pressure: int) -> None:
    seen = getattr(front, "_dual_gw_skipped", None)
    if seen is None:
        seen = front._dual_gw_skipped = set()
    if rid in seen:
        return
    seen.add(rid)
    front.counters["dual_p_pause_skipped_grant_wait"] += 1
    n = int(front.counters["dual_p_pause_skipped_grant_wait"])
    if n <= LOG_FIRST or n % LOG_EVERY == 0:
        logger.warning(
            "%s rid=%s pressure=%d B n=%d: the leg waits for its card grant on P and holds no KV there -- "
            "no abort, no requeue, no resend (P gives its context back at idle-except-waiters, "
            "PP0 grants after D's demand)", SKIP_MARK, rid, int(pressure), n)
    if len(seen) > 4096:
        seen.clear()
