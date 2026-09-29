"""The front's writes into the boot's state directory (IPC §2.2, writer ``front``).

User law 2026-09-28 ~20:45Z: control and IPC never over log lines -- state.json,
events.jsonl, stop_request.json; logs are for humans only. Every write goes
through the ONE writer, ``weg2/state_file.py``; nothing here defines a second
format.

* ``publish_front_stop`` (W98-STOP-FINAL, z30u 07:14:10): a front STOP is a
  named, controlled teardown. It lands as the event ``front_stop`` and -- while
  the boot is live -- as the A5 stop request (``stop_request.json``, origin
  ``front``). The host writer turns that request into ``dead`` with this cause
  and ``cause.rc = 24`` at the boot's end (``state_file finish``); the Phase-2
  healthcheck (``state_file health``) reads it as dead at once. The lifecycle
  itself is NOT written from here: ``stopping``/``dead`` are host states, and
  both make the arms' group watchers stand down (27B ``gwatch`` returns on
  them BEFORE its ``docker stop``), so a front-written lifecycle would turn a
  torn-down boot into one that holds until its window ends.
* ``publish_flip_cushion``: one ``flip_cushion`` event per completed flip --
  the host cushion's course without waiting for a W98.
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import List, Optional, Tuple

from sglang.srt.weg2 import state_file

#: lifecycle states in which a front STOP writes NO stop request: the end is
#: already running (stopping) or done (terminal) -- the first cause wins.
NO_STOP_REQUEST = ("stopping",) + tuple(state_file.TERMINAL)


def stop_cause_code(name: str) -> str:
    """``"W98 Weg2HostRateLatched"`` -> ``"W98_Weg2HostRateLatched"``: W-number
    plus name, never the bare W-number (IPC §2.2 cause.code, 27B requirement 6)."""
    return re.sub(r"\s+", "_", str(name).strip()) or "FRONT_STOP"


def publish_front_stop(d: str, *, name: str, detail: str, state_before: Optional[str],
                       epoch: Optional[int], awake: Optional[str]) -> dict:
    """Event ``front_stop`` always; ``stop_request.json`` while the boot is live
    and no earlier stop request stands (a watcher's or the deadman's cause is
    never overwritten). Returns what was written, for the human line.

    ``StateFileError`` is a ``SystemExit`` (the CLI's exit path) -- it is turned
    into a named result here, so a foreign schema can never end the front."""
    try:
        return _publish_front_stop(d, name=name, detail=detail, state_before=state_before,
                                   epoch=epoch, awake=awake)
    except state_file.StateFileError as e:
        return {"written": False, "stop_request": False, "lifecycle": None,
                "why": f"StateFileError: {e}"}


def _publish_front_stop(d: str, *, name: str, detail: str, state_before: Optional[str],
                        epoch: Optional[int], awake: Optional[str]) -> dict:
    st = state_file.read(d)
    if not st:
        return {"written": False, "stop_request": False, "lifecycle": None,
                "why": f"{d}/state.json missing -- the host writer creates it"}
    code = stop_cause_code(name)
    cur = (st.get("lifecycle") or {}).get("state")
    state_file.add_event(
        d, "front_stop",
        {"name": str(name), "detail_full": str(detail), "front_state_before": state_before,
         "epoch": epoch, "awake": awake, "lifecycle": cur},
        writer="front", code=code)
    path = os.path.join(d, "stop_request.json")
    req = False
    if cur not in NO_STOP_REQUEST and not os.path.exists(path):
        state_file.write_json_atomic(path, {
            "code": code, "origin": "front", "group": None, "rank": None,
            "detail_full": f"{name}: {detail}"})
        req = True
    return {"written": True, "stop_request": req, "lifecycle": cur, "code": code}


def publish_flip_cushion(d: str, rec: dict) -> bool:
    """One ``flip_cushion`` event (writer front). False = no state.json."""
    if not d or not os.path.exists(os.path.join(d, "state.json")):
        return False
    try:
        state_file.add_event(d, "flip_cushion", dict(rec), writer="front")
    except state_file.StateFileError:
        return False
    return True


# ---------------- DASHBOARD-AUS-IPC (a)/(b), 29.09. ----------------
# User order (via 27B): the dashboard is fed from the IPC, not from log lines.
# The front writes the flip as events and its own keys under state.json
# `front` -- through state_file (writer front), never by a direct open, and
# never a lifecycle field (those are the host's and the launcher's).

#: the host mirrors these five keys of `front` from /weg2/state (state_file beat);
#: the front never writes them, so every key has one writer.
HOST_FRONT_KEYS = ("state", "epoch", "awake", "queue", "outstanding")
#: kept out of a flip_done event: the chunk list (a line over state_file.EVENT_MAX
#: would be cut into invalid JSON).
FLIP_DONE_DROP = ("chunks",)


def publish_event(d: str, typ: str, data: dict) -> bool:
    """One event of the front into the boot's events.jsonl (writer front).
    False = no state.json (no state dir, or the host has not created it)."""
    if not d or not os.path.exists(os.path.join(d, "state.json")):
        return False
    try:
        state_file.add_event(d, typ, dict(data), writer="front")
    except state_file.StateFileError:
        return False
    return True


def publish_front_fields(d: str, fields: dict) -> bool:
    """The front's own keys under state.json `front`, dotted, so the host's mirror
    keys (:data:`HOST_FRONT_KEYS`) stay the host's. A host key here is a ValueError."""
    if not d or not os.path.exists(os.path.join(d, "state.json")):
        return False
    bad = [k for k in fields if k in HOST_FRONT_KEYS]
    if bad:
        raise ValueError(f"front_state_ipc: {bad} are the host's mirror keys of `front`")
    try:
        state_file.transition(d, None, fields={f"front.{k}": v for k, v in fields.items()},
                              writer="front")
    except state_file.StateFileError:
        return False
    return True


def flip_done_payload(rec: dict, flip_begin_ts: float) -> dict:
    """The ``flip_done`` event: the front's flip_log record (drain/sleep/wake/flip_ms,
    legs, overlap, critical path) without the chunk list, plus the begin stamp."""
    out = {k: v for k, v in rec.items() if k not in FLIP_DONE_DROP}
    out["chunks_n"] = len(rec.get("chunks") or ())
    out["flip_begin_ts"] = round(float(flip_begin_ts), 3)
    return out


#: A14: the group-health verdicts on whose CHANGE the front looks for rank stops.
RANK_STOP_VERDICTS = ("failing", "held", "dead")


def rank_stops_of(rankstate_dir: str, group: str) -> List[Tuple[str, dict]]:
    """``(rank_key, stop)`` for every stop the group's ranks recorded in their
    rankstats files (weg2/rankstats.py ``stops.last``). A file of another
    schema or a torn one is skipped, never raised."""
    from sglang.srt.weg2 import rankstats

    out: List[Tuple[str, dict]] = []
    for p in sorted(glob.glob(os.path.join(rankstate_dir, f"{group}.*{rankstats.SUFFIX}"))):
        try:
            with open(p) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict) or rec.get("schema") != rankstats.SCHEMA:
            continue
        rk = f"tp{rec.get('tp_rank')}pp{rec.get('pp_rank')}"
        for s in ((rec.get("stops") or {}).get("last") or ()):
            if isinstance(s, dict):
                out.append((rk, dict(s, pid=rec.get("pid"))))
    return out


def publish_rank_stops(d: str, group: str, seen: set) -> int:
    """A14: every stop of ``group``'s ranks not yet published -> one event
    ``rank_stop`` (writer front; envelope group/rank/code, data t/reason/code/
    exc/ticket/text/pid). The rank directory is the launcher's
    ``groups.<G>.rankstate_dir``. ``seen`` belongs to the caller's ONE IPC
    thread. Returns the number published; no state dir = 0."""
    if not d or not os.path.exists(os.path.join(d, "state.json")):
        return 0
    try:
        st = state_file.read(d)
    except state_file.StateFileError:
        return 0
    rs_dir = ((st.get("groups") or {}).get(group) or {}).get("rankstate_dir")
    if not rs_dir:
        return 0
    n = 0
    for rk, s in rank_stops_of(rs_dir, group):
        key = (group, rk, s.get("t"), s.get("code"))
        if key in seen:
            continue
        try:
            state_file.add_event(d, "rank_stop", dict(s, group=group, rank=rk),
                                 writer="front", group=group, rank=rk, code=s.get("code"))
        except state_file.StateFileError:
            return n
        seen.add(key)
        n += 1
    return n


def group_health_verdict(http_ok: bool, alive: bool, streak: int, held: bool) -> str:
    """One word per FH poll (front_health GroupFacts): dead (process gone),
    held (#1223 DEBUG-HOLD), failing (a counted /health failure), busy (a slow
    /health while the group computes, FP beacon: not counted), ok."""
    if not alive:
        return "dead"
    if held:
        return "held"
    if int(streak or 0) > 0:
        return "failing"
    return "ok" if http_ok else "busy"


#: the front's IPC queue bound (27B review of b02cec1ad5): a writer blocked on the
#: disk or on the state lock must never make the front pile up anon RAM.
IPC_QUEUE_MAX = 1024


class BoundedWriter:
    """ONE writer thread over a BOUNDED FIFO. On overflow the OLDEST entry is
    dropped and counted (``dropped``) -- a newer record supersedes it for the
    dashboard, and the front never waits here. A failed write is counted
    (``failed``) and handed to ``on_error``, never raised into the front."""

    def __init__(self, maxlen: int = IPC_QUEUE_MAX, name: str = "weg2-ipc", on_error=None) -> None:
        import collections
        import threading

        self.maxlen = int(maxlen)
        self.dropped = 0
        self.failed = 0
        self._q = collections.deque()
        self._cv = threading.Condition()
        self._on_error = on_error
        self._t = threading.Thread(target=self._run, name=name, daemon=True)
        self._t.start()

    def depth(self) -> int:
        with self._cv:
            return len(self._q)

    def submit(self, fn, *args) -> None:
        with self._cv:
            if len(self._q) >= self.maxlen:
                self._q.popleft()
                self.dropped += 1
            self._q.append((fn, args))
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._q:
                    self._cv.wait()
                fn, args = self._q.popleft()
            try:
                fn(*args)
            except Exception as e:  # noqa: BLE001 -- a lost record never stops the front
                self.failed += 1
                if self._on_error is not None:
                    self._on_error(fn, e)


class FirstWorkClock:
    """Flip time from ONE clock (time.time() of the front's own process): the flip's
    begin stamp and the woken group's first work are both taken here.

    P->D: the first content D streams after the flip = the first decode token (the
    user's flip time, P end -> first decode token; a P->D flip begins at P's end).
    D->P: the first leg 1 the front dispatches to P after the flip. Armed at
    ``WEG2-FLIP done``, fired once; the next flip re-arms it, so a flip whose woken
    group never worked before the next one simply never fires. No model switch:
    the 27B flip-time tile reads the same event."""

    def __init__(self) -> None:
        self._armed: Optional[dict] = None

    def arm(self, epoch: int, sleep: str, wake: str, flip_begin_ts: float) -> None:
        self._armed = {"epoch": int(epoch), "dir": f"{sleep}>{wake}", "wake": wake,
                       "flip_begin_ts": float(flip_begin_ts)}

    def seen(self, group: str, what: str, rid: Optional[str], now: float) -> Optional[dict]:
        a = self._armed
        if a is None or a["wake"] != group:
            return None
        self._armed = None
        return {"epoch": a["epoch"], "dir": a["dir"], "flip_begin_ts": round(a["flip_begin_ts"], 3),
                "first_work_ts": round(float(now), 3),
                "flip_time_ms": round((float(now) - a["flip_begin_ts"]) * 1000.0),
                "what": what, "rid": rid, "clock": "time.time front"}
