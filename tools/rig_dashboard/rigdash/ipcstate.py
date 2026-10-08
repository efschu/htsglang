"""The boots' IPC state (IPC-STATE-PLAN §2.2), read instead of their logs.

User order 2026-09-29 (via 27B): "das dashboard soll auch aus der inter prozess
kommunikation gespeist werden, nicht aus logs".  Per boot, the writers keep

  /spinning/docker-acceptance/<line>/state/<boot_id>/state.json    (weg2.state/1, atomic)
  /spinning/docker-acceptance/<line>/state/<boot_id>/events.jsonl  (weg2.event/1, append-only)
  /spinning/docker-acceptance/<line>/state/<boot_id>/stop_request.json

and ``state.json.tag`` is the launcher tag the boot's logs carry (``WEG2 BOOT
tag=``), which is how a log-discovered boot finds its state.  Read-only: this
module never writes there (the one writer is weg2/state_file.py).

What is read from here and what still comes from a log line is listed in
/spinning/gpu-arb/docs/DASHBOARD-AUS-IPC-INVENTAR-0929.md; a display fed by a
log carries the label "aus Log (Übergang)" in the page.
"""

from __future__ import annotations

import json
import os
import threading
import time

from typing import Dict, List, Optional

from . import names as N

STATE_ROOTS = ("/spinning/docker-acceptance/nf/state", "/spinning/docker-acceptance/27b/state")
SCHEMA = "weg2.state/1"
EVENT_SCHEMA = "weg2.event/1"
TERMINAL = ("refused_preflight", "stopped_clean", "dead")
SHOW_S = 6 * 3600.0           # same horizon as the log-discovered boots (live.SHOW_S)
#: the events the field readers use (ipcfields.py, DASHBOARD-AUS-IPC-INVENTAR rows A12/A14/B1-B7/D4)
FIELD_EVENT_TYPES = ("flip_begin", "flip_done", "flip_first_work", "flip_user_time", "group_health", "rank_stop", "post_wake_pass",
                     "group_ready")
EVENT_TYPES = ("lifecycle", "hold_begin", "hold_end", "deadman_verdict", "front_stop", "flip_cushion",
               "group_ready", "launcher_done") + FIELD_EVENT_TYPES
EVENTS_KEEP = 2000
# flip_first_work (front's own clock, ms): kept apart from `rows` so a long boot's flip
# times are not pushed out by the other event types' window.
FIRST_WORK_KEEP = 5000
# request_done (front, one per finished request): Token x-y (n neu) per rid for the prefill/decode hover
# (Nutzer 02.10. ~12:04Z); ~1,2 kB each
REQ_DONE_KEEP = 2000


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _argv_opt(argv: List[str], name: str) -> Optional[str]:
    for i, a in enumerate(argv or ()):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _int(x) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def topology(groups: dict) -> dict:
    """{G: {tp, pp}} from ``groups.<G>.launch.argv`` -- the launcher's own argv, not a log line."""
    out = {}
    for g, v in (groups or {}).items():
        argv = ((v or {}).get("launch") or {}).get("argv") or []
        if not argv:
            continue
        out[g] = {"tp": _int(_argv_opt(argv, "--tp-size")) or 1, "pp": _int(_argv_opt(argv, "--pp-size")) or 1}
    return out


def served_model(groups: dict) -> Optional[str]:
    for v in (groups or {}).values():
        argv = ((v or {}).get("launch") or {}).get("argv") or []
        name = _argv_opt(argv, "--served-model-name")
        if name:
            return name
        path = _argv_opt(argv, "--model-path")
        if path:
            return os.path.basename(path.rstrip("/"))
    return None


def transport(groups: dict) -> Optional[str]:
    for v in (groups or {}).values():
        env = ((v or {}).get("launch") or {}).get("env") or {}
        t = N.env_get(env, "HTSGLANG_TRANSPORT") if isinstance(env, dict) else None
        if t:
            return str(t)
    return None


class _Events:
    """Tails one events.jsonl by byte offset; keeps the types this page reads."""

    def __init__(self, path: str):
        self.path = path
        self.off = 0
        self.buf = b""
        self.rows: List[dict] = []
        self.first_work: List[dict] = []
        self.user_time: List[dict] = []
        self.req_done: List[dict] = []
        self.counts: Dict[str, int] = {}

    def poll(self):
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.off)
                data = fh.read()
        except OSError:
            return
        if not data:
            return
        self.off += len(data)
        data = self.buf + data
        lines = data.split(b"\n")
        self.buf = lines.pop()          # a torn last line waits for its newline
        for raw in lines:
            try:
                e = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(e, dict) or not N.schema_ok(e.get("schema"), EVENT_SCHEMA):
                continue
            t = e.get("type")
            self.counts[t] = self.counts.get(t, 0) + 1
            if t in EVENT_TYPES:
                self.rows.append(e)
            if t == "flip_first_work" and isinstance(e.get("data"), dict):
                self.first_work.append(e["data"])
            if t == "flip_user_time" and isinstance(e.get("data"), dict):
                self.user_time.append(dict(e["data"], ts=e.get("ts")))
            if t == "request_done" and isinstance(e.get("data"), dict):
                self.req_done.append(e["data"])
        del self.rows[:-EVENTS_KEEP]
        del self.first_work[:-FIRST_WORK_KEEP]
        del self.user_time[:-FIRST_WORK_KEEP]
        del self.req_done[:-REQ_DONE_KEEP]


def boot_view(d: str, st: dict, ev: Optional[_Events], now: float) -> dict:
    """What the page shows of one state dir.  Pure over its inputs (unit-tested)."""
    rows = ev.rows if ev else []
    groups = st.get("groups") or {}
    lc = st.get("lifecycle") or {}
    hb = st.get("heartbeat") or {}
    hb_ts = max([float(v.get("ts") or 0) for v in hb.values() if isinstance(v, dict)] or [0]) or None
    stopping_ts = next((e.get("ts") for e in rows if e.get("type") == "lifecycle"
                        and (e.get("data") or {}).get("state") == "stopping"), None)
    return {
        "src": "state.json",
        "dir": d,
        "boot_id": st.get("boot_id"),
        "kind": st.get("kind"),
        "tag": st.get("tag"),
        "line": st.get("line"),
        "rev": st.get("rev"),
        "profile": st.get("profile"),
        "image": st.get("image"),
        "container": st.get("container"),
        "gpuq_id": st.get("gpuq_id"),
        "lifecycle": lc.get("state"),
        "lifecycle_since": lc.get("since_ts"),
        "terminal": lc.get("state") in TERMINAL,
        "serving_since_ts": st.get("serving_since_ts"),
        "cause": st.get("cause"),
        "front": st.get("front"),
        "heartbeat_age_s": round(now - hb_ts, 1) if hb_ts else None,
        "stop_request": _read_json(os.path.join(d, "stop_request.json")),
        "stopping_ts": stopping_ts,
        "topology": topology(groups),
        "model": served_model(groups),
        "transport": transport(groups),
        "launch": {g: (v or {}).get("launch") for g, v in groups.items() if (v or {}).get("launch")},
        "group_state": {g: (v or {}).get("state") for g, v in groups.items()},
        "forms": {g: (v or {}).get("form") for g, v in groups.items() if (v or {}).get("form")},
        "ipc_events": [e for e in rows if e.get("type") in FIELD_EVENT_TYPES][-600:],
        "events": {"counts": dict(ev.counts) if ev else {},
                   "hold_end": [e for e in rows if e.get("type") == "hold_end"][-1:],
                   "deadman_verdict": [e for e in rows if e.get("type") == "deadman_verdict"][-3:],
                   "front_stop": [e for e in rows if e.get("type") == "front_stop"][-3:]},
        "flip_first_work": list(ev.first_work) if ev else [],
        # D>P Flipzeit nach Nutzerdefinition (Decode-Ende -> P-Prefill-Start), ab Build y4z (53977b2b67)
        "flip_user_time": list(ev.user_time) if ev else [],
        # Token x-y (n neu) per rid (request_done prefill.{P,D}, decode_tokens, first_token_ts/end_ts)
        "request_done": list(ev.req_done) if ev else [],
    }


class IpcStates:
    """All boot state dirs of the last SHOW_S; ``for_tag`` finds a log boot's one."""

    def __init__(self, roots=STATE_ROOTS):
        self.roots = roots
        self.lock = threading.Lock()
        self._mt: Dict[str, float] = {}
        self._st: Dict[str, dict] = {}
        self._ev: Dict[str, _Events] = {}
        self.last_error: Optional[str] = None
        self.last_poll: Optional[float] = None     # history.py waits for the first poll before a backfill

    def poll(self, now: Optional[float] = None):
        now = now or time.time()
        seen = set()
        for root in self.roots:
            try:
                names = os.listdir(root)
            except OSError:
                continue
            for n in names:
                d = os.path.join(root, n)
                if n.startswith("current") or os.path.islink(d) or not os.path.isdir(d):
                    continue
                p = os.path.join(d, "state.json")
                try:
                    mt = os.stat(p).st_mtime
                except OSError:
                    continue
                if now - mt > SHOW_S:
                    continue
                seen.add(d)
                if self._mt.get(d) != mt:
                    st = _read_json(p)
                    if st and N.schema_ok(st.get("schema"), SCHEMA):
                        with self.lock:
                            self._st[d] = st
                            self._mt[d] = mt
                ev = self._ev.get(d)
                if ev is None:
                    ev = self._ev[d] = _Events(os.path.join(d, "events.jsonl"))
                ev.poll()
        with self.lock:
            for d in list(self._st):
                if d not in seen:
                    self._st.pop(d, None)
                    self._mt.pop(d, None)
                    self._ev.pop(d, None)
        self.last_poll = now

    def boots(self, now: Optional[float] = None) -> List[dict]:
        """Every ``kind=boot`` state dir of the last SHOW_S (history.Recorder.ingest_ipc walks these:
        the model series come from the ranks' rankstats, no log is involved)."""
        now = now or time.time()
        with self.lock:
            items = [(d, st) for d, st in self._st.items() if st.get("kind") == "boot"]
            return [boot_view(d, st, self._ev.get(d), now) for d, st in items]

    def for_tag(self, tag: Optional[str], now: Optional[float] = None) -> Optional[dict]:
        """The ``kind=boot`` state of this launcher tag (the newest one when a tag was reused)."""
        if not tag:
            return None
        now = now or time.time()
        with self.lock:
            cand = [(d, st) for d, st in self._st.items() if st.get("tag") == tag and st.get("kind") == "boot"]
            if not cand:
                return None
            d, st = max(cand, key=lambda x: self._mt.get(x[0], 0))
            return boot_view(d, st, self._ev.get(d), now)
