"""WEG2 MODE-PARITY (h16m integration 1014, review nf-prio-alter N4): front, group P and group D run ONE priority mode.

WHY.  ``SGLANG_WEG2_PRIO_AS_AGE`` and ``SGLANG_WEG2_LANES`` are read in three kinds of process -- the front (its
order keys, the lane controller), every rank of group P (the P batch order, ``lanes_p``) and every rank of group D
(admission order, seat / KV victims, D's lane park).  The launcher hands all of them the same environment, but nothing
CHECKED that: a front in the floor mode against groups in the age mode loops on ``W-LANE refused`` (the groups refuse
the floor RPC), a front in the age mode against groups in the floor mode lets a lower request be parked by a floor
nobody raises.  Neither is a crash; both serve wrongly and quietly.

WHAT.  Each process states its mode once at start (``WEG2 MODE-PARITY role=...``): every scheduler rank in
``init_request_receiver`` (no collective -- a log line and an attribute), the front in its startup.  Every group also
publishes its mode in ``/get_server_info`` -> ``internal_states[0].weg2_mode`` (the endpoint the front already polls).
The front reads both groups once at start (retried until each answered) and compares: equal -> ``WEG2 MODE-PARITY
OK``; different, or a group that answers without the block -> the named front stop ``W-MODE-PARITY`` (the law
raenge-nie-uneins: a detected disagreement stops, it is never served around).

ONE DECIDER.  The front decides (one process); the groups only report.  Ranks inside one group inherit the group's
environment (one server process spawns them); their lines are the metal witness (checklist: every
``WEG2 MODE-PARITY role=group`` line of a boot shows the same pair).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

MARK = "WEG2 MODE-PARITY"
#: the front's named stop
STOP_NAME = "W-MODE-PARITY"
#: key under ``internal_states[0]`` of ``/get_server_info``
INFO_KEY = "weg2_mode"
#: the switches compared (raw effective values) -- the derived ``age``/``floor`` follow from them
SWITCHES = ("SGLANG_WEG2_LANES", "SGLANG_WEG2_PRIO_AS_AGE")
_FIELDS = ("lanes", "prio_as_age")


def local_mode() -> Dict[str, int]:
    """This process's mode: the two switches as read by the code (``envs``), and what they mean
    (``age`` = ``prio_age.enabled()``, ``floor`` = ``prio_age.floor_mode()``)."""
    from sglang.srt.environ import envs
    from sglang.srt.weg2 import prio_age

    return {
        "lanes": int(bool(envs.SGLANG_WEG2_LANES.get())),
        "prio_as_age": int(bool(envs.SGLANG_WEG2_PRIO_AS_AGE.get())),
        "age": int(bool(prio_age.enabled())),
        "floor": int(bool(prio_age.floor_mode())),
    }


def _fmt(mode: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(mode, Mapping):
        return "absent"
    return " ".join(f"{k}={mode.get(k)}" for k in ("lanes", "prio_as_age", "age", "floor"))


def log_rank(sched: Any) -> Dict[str, int]:
    """The rank's start line (every rank of a Weg-2 group). No collective, never raises."""
    try:
        mode = local_mode()
        ps = getattr(sched, "ps", None)
        logger.info("%s role=group group=%s pp_rank=%s tp_rank=%s %s raw=%s", MARK,
                    os.environ.get("SGLANG_WEG2_GROUP", "") or "-",
                    getattr(ps, "pp_rank", getattr(sched, "pp_rank", "?")),
                    getattr(ps, "tp_rank", getattr(sched, "tp_rank", "?")), _fmt(mode),
                    {k: os.environ.get(k) for k in SWITCHES})
        return mode
    except Exception as e:  # noqa: BLE001 - an instrument never stops a rank; the front's check stops
        logger.warning("%s role=group unreadable (%s: %s)", MARK, type(e).__name__, e)
        return {}


def mode_of_info(info: Any) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """``(answered, mode)`` out of a ``/get_server_info`` JSON body. ``answered`` False = no usable body (retry);
    ``mode`` None with ``answered`` True = the group answered WITHOUT the block (other code -- a stop)."""
    if isinstance(info, list) and info:
        info = info[0]
    if not isinstance(info, dict):
        return False, None
    st = info.get("internal_states")
    if isinstance(st, list) and st and isinstance(st[0], dict):
        blk = st[0].get(INFO_KEY)
    elif isinstance(st, dict):
        blk = st.get(INFO_KEY)
    else:
        blk = info.get(INFO_KEY)
    return True, (dict(blk) if isinstance(blk, dict) else None)


def compare(front: Mapping[str, Any], groups: Mapping[str, Optional[Mapping[str, Any]]]) -> List[str]:
    """The disagreements, by name (empty = one mode everywhere). Compared: the two raw switches."""
    out: List[str] = []
    for name in sorted(groups):
        g = groups[name]
        if not isinstance(g, Mapping):
            out.append(f"group {name} answered without {INFO_KEY} (other code than the front)")
            continue
        for k in _FIELDS:
            if int(g.get(k, -1)) != int(front.get(k, -2)):
                out.append(f"{k}: front={front.get(k)} {name}={g.get(k)}")
    return out


def verdict_line(front: Mapping[str, Any], groups: Mapping[str, Optional[Mapping[str, Any]]]) -> Tuple[bool, str]:
    """``(ok, line)`` -- the front's one log line for the start check."""
    diff = compare(front, groups)
    parts = [f"front[{_fmt(front)}]"] + [f"{n}[{_fmt(groups[n])}]" for n in sorted(groups)]
    if not diff:
        return True, f"{MARK} OK " + " ".join(parts)
    return False, (f"{MARK} MISMATCH " + " ".join(parts) + " -- " + "; ".join(diff)
                   + " (front and groups must run one priority mode: SGLANG_WEG2_LANES and SGLANG_WEG2_PRIO_AS_AGE equal "
                     "in the environment of the front and of every group)")
