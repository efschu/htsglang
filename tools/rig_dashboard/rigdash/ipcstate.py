"""The boots' IPC state (IPC-STATE-PLAN §2.2), read instead of their logs.

User order 2026-09-29 (via 27B): "das dashboard soll auch aus der inter prozess
kommunikation gespeist werden, nicht aus logs".  Per boot, the writers keep

  /spinning/docker-acceptance/<line>/state/<boot_id>/state.json    (pdflip.state/1, atomic)
  /spinning/docker-acceptance/<line>/state/<boot_id>/events.jsonl  (pdflip.event/1, append-only)
  /spinning/docker-acceptance/<line>/state/<boot_id>/stop_request.json

and ``state.json.tag`` is the launcher tag the boot's logs carry (``PDFLIP BOOT
tag=``), which is how a log-discovered boot finds its state.  Read-only: this
module never writes there (the one writer is pdflip/state_file.py).

What is read from here and what still comes from a log line is listed in
/spinning/gpu-arb/docs/DASHBOARD-AUS-IPC-INVENTAR-0929.md; a display fed by a
log carries the label "aus Log (Übergang)" in the page.
"""

from __future__ import annotations

import glob
import json
import os
import stat
import threading
import time

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from . import names as N

# 21.10.: state-dev = the NF dev container (flliper-nf-dev, Dev-Staende im Fenster) writes its boots there; without it the page says
# "no IPC sample" while the dev container serves (patched into the running release 8d32c66997 at 06:50Z, carried here)
STATE_ROOTS = ("/spinning/docker-acceptance/nf/state", "/spinning/docker-acceptance/nf/state-dev", "/spinning/docker-acceptance/27b/state")
# 23.10.: flliper-27b-nvfp4-duo writes to 27b/state-duo, a root none of the fixed ones names -- /api/live showed "boots: []" while
# it served.  The roots are now taken from the containers themselves (``StateRoots``): every docker container's bind mount on the
# state volume (``CONTAINER_STATE_DEST``, sources.container_state_roots); the fixed roots stay (a removed container's last boot);
# the glob is only the fallback for a process that has no docker reading (no ssh, the docker sample failed, not yet sampled).
CONTAINER_STATE_DEST = "/var/lib/htsglang/stateroot"
STATE_ROOT_GLOB = "/spinning/docker-acceptance/*/state*"
SCHEMA = "pdflip.state/1"
EVENT_SCHEMA = "pdflip.event/1"
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


def _dir_ident(path: str) -> Optional[Tuple[int, int]]:
    try:
        s = os.stat(path)
    except OSError:
        return None
    return (s.st_dev, s.st_ino)


def dedupe_roots(roots: Iterable[str]) -> List[str]:
    """``roots`` in order, each directory once: by normalised path, and by (st_dev, st_ino) for one that exists (a symlinked or
    bind-mounted second name of a root is the same root).  A root that does not exist (yet) is kept: ``poll`` skips it."""
    out, paths, idents = [], set(), set()
    for r in roots:
        if not r:
            continue
        p = os.path.normpath(str(r))
        if p in paths:
            continue
        paths.add(p)
        ident = _dir_ident(p)
        if ident is not None:
            if ident in idents:
                continue
            idents.add(ident)
        out.append(p)
    return out


class StateRoots:
    """The state roots to scan, as a callable (``IpcStates(roots=StateRoots())``).

    Order: the roots the containers have mounted (``containers()`` -> a list of host paths; sources.Sources' docker rows carry
    them as ``state_root``), then every container root seen before in this process (a container that is removed keeps its
    root here until the process restarts), then the fixed ``STATE_ROOTS``.  Only when there is no container reading at all
    (``containers`` unset, or it returns None: no ssh, the docker sample failed before its first value) the glob
    ``STATE_ROOT_GLOB`` is added as the fallback.  Resolved at most every ``refresh_s``; duplicates removed (``dedupe_roots``)."""

    def __init__(self, static: Sequence[str] = STATE_ROOTS, containers: Optional[Callable[[], Optional[List[str]]]] = None,
                 pattern: Optional[str] = STATE_ROOT_GLOB, refresh_s: float = 5.0):
        self.static = tuple(static)
        self.containers = containers
        self.pattern = pattern
        self.refresh_s = refresh_s
        self.seen: List[str] = []            # container roots, first seen first
        self.source = "static"               # "containers" | "glob" | "static": what the last resolution was based on
        self.last_error: Optional[str] = None
        self._t: Optional[float] = None
        self._val: List[str] = []

    def _container_roots(self) -> Optional[List[str]]:
        if self.containers is None:
            return None
        try:
            got = self.containers()
        except Exception as e:  # noqa: BLE001 -- a broken reading falls back to the glob, it never stops the poll
            self.last_error = "%s: %s" % (type(e).__name__, e)
            return None
        self.last_error = None
        if got is None:
            return None
        return [os.path.normpath(str(r)) for r in got if r]

    def resolve(self) -> List[str]:
        cont = self._container_roots()
        extra: List[str] = []
        if cont is not None:
            for r in cont:
                if r not in self.seen:
                    self.seen.append(r)
            self.source = "containers"
        elif self.pattern:
            extra = sorted(p for p in glob.glob(self.pattern) if os.path.isdir(p))
            self.source = "glob"
        else:
            self.source = "static"
        return dedupe_roots(list(cont or ()) + self.seen + list(self.static) + extra)

    def __call__(self, now: Optional[float] = None) -> List[str]:
        now = now or time.time()
        if self._t is None or now - self._t >= self.refresh_s or now < self._t:
            self._val = self.resolve()
            self._t = now
        return list(self._val)


class IpcStates:
    """All boot state dirs of the last SHOW_S; ``for_tag`` finds a log boot's one.

    ``roots``: a sequence of root dirs (tests, fixed setups) or a callable returning them (default: ``StateRoots()``).  One boot
    dir reachable under two roots (23.10.: a boot dir of 27b/state-duo bind-mounted into 27b/state) is read once: dirs are told
    apart by (st_dev, st_ino), and the name it was first read under is kept while it is still there."""

    def __init__(self, roots: Union[Sequence[str], Callable[..., Sequence[str]], None] = None):
        self.roots = StateRoots() if roots is None else roots
        self.roots_used: List[str] = []
        self.lock = threading.Lock()
        self._ident: Dict[Tuple[int, int], str] = {}
        self._mt: Dict[str, float] = {}
        self._st: Dict[str, dict] = {}
        self._ev: Dict[str, _Events] = {}
        self.last_error: Optional[str] = None
        self.last_poll: Optional[float] = None     # history.py waits for the first poll before a backfill

    def attach_containers(self, fn: Callable[[], Optional[List[str]]]) -> None:
        """Feed the default ``StateRoots`` the container roots (``fn`` -> host paths, None = no docker reading)."""
        if isinstance(self.roots, StateRoots):
            self.roots.containers = fn

    def _current_roots(self, now: float) -> List[str]:
        if callable(self.roots):
            return list(self.roots(now))
        return list(self.roots)

    def _boot_dirs(self, now: float) -> List[str]:
        """Every boot dir under the roots, each directory once (see the class doc)."""
        roots = self._current_roots(now)
        self.roots_used = roots
        by_ident: Dict[Tuple[int, int], List[str]] = {}
        order: List[Tuple[int, int]] = []
        for root in roots:
            try:
                names = os.listdir(root)
            except OSError:
                continue
            for n in names:
                d = os.path.join(root, n)
                if n.startswith("current"):
                    continue
                try:
                    ls = os.lstat(d)
                except OSError:
                    continue
                if stat.S_ISLNK(ls.st_mode) or not stat.S_ISDIR(ls.st_mode):
                    continue
                ident = (ls.st_dev, ls.st_ino)
                if ident not in by_ident:
                    by_ident[ident] = []
                    order.append(ident)
                by_ident[ident].append(d)
        out = []
        for ident in order:
            names = by_ident[ident]
            prev = self._ident.get(ident)
            d = prev if prev in names else names[0]
            self._ident[ident] = d
            out.append(d)
        for ident in [i for i in self._ident if i not in by_ident]:
            self._ident.pop(ident, None)
        return out

    def poll(self, now: Optional[float] = None):
        now = now or time.time()
        seen = set()
        for d in self._boot_dirs(now):
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
