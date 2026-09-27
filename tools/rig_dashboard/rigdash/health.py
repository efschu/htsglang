"""Is a running model DEAD or HUNG?  One verdict per boot, with its reasons.

Operator order 2026-09-27 (after 27B-b1 sat P-dead from 09:45:58 with D and
the front alive, the container "unhealthy" and requests hanging, unnoticed):
a state on the model card nobody can overlook.  Sources, each named in the
reason line it produces:

* WEG2-HEALTH (front log): ``process_alive=False`` -> the group is dead;
  ``http_ok=False`` with a streak >= 2 while alive -> the group is hung.
* named stops in the P/D logs (parse.STOP_RE: W27, #791b, SPLIT refused,
  ADMISSION SPLIT, a scheduler traceback, CUDA OOM, DEBUG-HOLD) with no
  prefill/decode line of that group after them.
* Docker health: ``(unhealthy)`` in the container status.
* the hang indicator: work is queued, but no prefill/decode line for
  HANG_S seconds.

Pure function over the snapshot dicts, so it is unit-tested without a boot.
"""

from __future__ import annotations

from typing import List, Optional

HEALTH_FRESH_S = 120.0   # a WEG2-HEALTH line older than this no longer describes now
STOP_RECOVERED_S = 30.0  # activity this long after a stop means it ran on
HANG_S = 60.0            # queued work and no prefill/decode line for this long


def _age(now: float, t: Optional[float]) -> Optional[float]:
    return None if t is None else max(0.0, now - t)


def assess(b: dict, now: float, docker_ok: bool = True) -> dict:
    """``{"state": "TOT"|"HAENGT"|"WARNUNG"|None, "reasons": [{level, group, text, t}]}``.

    Only boots that are live (log written within LIVE_S) or whose container
    still runs are judged; a finished boot gets ``state=None`` and, if it
    ended on a named stop, that stop as ``ended_on``.
    """
    reasons: List[dict] = []
    c = b.get("container") or {}
    judged = bool(b.get("live")) or c.get("State") == "running"

    def add(level, group, text, t=None):
        reasons.append({"level": level, "group": group, "text": text, "t": t})

    # 1. WEG2-HEALTH
    for g, h in sorted((b.get("health") or {}).items()):
        if _age(now, h.get("t")) is None or _age(now, h["t"]) > HEALTH_FRESH_S:
            continue
        if not h.get("alive"):
            add("dead", g, "Gruppe %s tot: WEG2-HEALTH process_alive=False%s" % (
                g, " (streak %d)" % h["streak"] if h.get("streak") else ""), h["t"])
        elif not h.get("http_ok") and (h.get("streak") or 0) >= 2:
            add("hang", g, "Gruppe %s antwortet nicht: WEG2-HEALTH http_ok=False, streak %d, Prozess lebt" % (
                g, h["streak"]), h["t"])

    # 2. named stops, newest per group; "why" lines beat the bare traceback line
    by_group = {}
    for s in b.get("stops") or []:
        cur = by_group.get(s["group"])
        if cur is None or s["t"] > cur["t"] or (s["t"] == cur["t"] and cur.get("bare") and not s.get("bare")):
            by_group[s["group"]] = s
    ended_on = None
    for g, s in sorted(by_group.items()):
        act = (b.get("last_activity") or {}).get(g)
        recovered = act is not None and act > s["t"] + STOP_RECOVERED_S
        text = "Gruppe %s gestoppt: %s" % (g, s["text"][:300])
        if ended_on is None or s["t"] > ended_on["t"]:
            ended_on = {"group": g, "text": s["text"][:300], "t": s["t"], "recovered": recovered}
        if recovered:
            add("warn", g, text + " (danach lief die Gruppe weiter)", s["t"])
        else:
            add("dead", g, text, s["t"])

    # 3. Docker health
    status = c.get("Status") or ""
    if "unhealthy" in status:
        add("dead" if any(r["level"] == "dead" for r in reasons) else "hang", None,
            "Docker meldet den Container %s: %s" % (c.get("Names"), status))

    # 4. queued work, no progress
    fr = b.get("front") or {}
    q_front = fr.get("queue")
    outstanding = sum((fr.get("outstanding") or {}).values()) if fr.get("outstanding") else 0
    q_log = (b.get("queue") or {}).get("queue") if b.get("queue") else None
    q_log_age = _age(now, (b.get("queue") or {}).get("t")) if b.get("queue") else None
    queued = None
    src = None
    if q_front is not None:
        queued, src = (q_front or 0) + outstanding, "Front /weg2/state (queue %s + outstanding %s)" % (q_front, outstanding)
    elif q_log is not None and q_log_age is not None and q_log_age < 600:
        queued, src = q_log, "letzte WEG2-ROUTE-Zeile"
    idle = _age(now, b.get("last_activity_any"))
    if judged and queued and idle is not None and idle >= HANG_S:
        add("hang", None, "HÄNGT: %d Anfrage(n) warten (%s), seit %d s keine Prefill-/Decode-Zeile" % (
            queued, src, idle))

    if not judged:
        return {"state": None, "reasons": [], "ended_on": ended_on}
    levels = {r["level"] for r in reasons}
    state = "TOT" if "dead" in levels else "HAENGT" if "hang" in levels else "WARNUNG" if levels else None
    return {"state": state, "reasons": reasons, "ended_on": ended_on}
