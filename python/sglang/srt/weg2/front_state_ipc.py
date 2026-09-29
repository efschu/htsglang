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

import os
import re
from typing import Optional

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
