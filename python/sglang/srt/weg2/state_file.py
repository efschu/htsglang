#!/usr/bin/env python3
"""Boot-Zustand als Datei statt Log-Zeile: der EINE Schreibcode (IPC §2.2).

Nutzer 28.09. ~20:45Z: „diese kommunikation über logs? … das muss man professionell
ordentlich machen“. Format: /spinning/gpu-arb/docs/IPC-STATE-PLAN-0928.md §2.2
(state.json `weg2.state/1`, events.jsonl `weg2.event/1`, rc-Tabelle).

Dieser Code liegt im Baum (python/sglang/srt/weg2/state_file.py) und als byte-
gleiche Host-Kopie in /spinning/gpu-arb/docker/acc_state.py (27B-Abnahme A1b);
test_weg2_state_file_0928 vergleicht beide. Nur stdlib: der Host ruft ihn mit
nacktem python3, der Launcher importiert ihn.

Schreiber (A1c, Feld-Eigentum): `host` (host_acceptance_v2.sh + Herzschlag-
Schleife), `launcher` (weg2 launcher im Container), `front` (Phase 2). Jeder
setzt nur seine Felder und Zustände; Übergänge laufen nur vorwärts (außer
serving ⇄ flipping), terminal bleibt terminal, `dead` schlägt jeden nicht-
terminalen Zustand. Eine Sperre für alle: flock auf <boot_id>/.lock (A1a;
derselbe Kernel unter dem Bind-Mount).

Schreiben: lesen-ändern-schreiben unter der Sperre, tmp + fsync + rename.
events.jsonl: ein write() je Zeile mit O_APPEND, < 4 KiB (sonst steht
detail_full in cause-<seq>.txt und das Event verweist darauf).

  state_file.py init   --root R --boot-id B --kind boot|d2|dry [--field k=v ...]
  state_file.py set    --dir D [--writer host|launcher|front] [--state S] [--if-state a,b]
                              [--cause-code C --cause-origin O --cause-detail T [--cause-group G]]
                              [--field k=v] [--json k=JSON]
  state_file.py event  --dir D --type T [--json data=JSON]
  state_file.py beat   --dir D [--container NAME] [--front-json JSON]
  state_file.py dead   --dir D --container NAME [--code C]
  state_file.py finish --dir D
  state_file.py rc     --dir D
  state_file.py get    --dir D [--key a.b]
  state_file.py new-boot-id --prefix P --kind K
"""
import argparse
import contextlib
import datetime as _dt
import fcntl
import json
import os
import secrets
import subprocess
import sys
import time

STATE_SCHEMA = "weg2.state/1"
EVENT_SCHEMA = "weg2.event/1"
STATES = ("preflight", "refused_preflight", "launching", "loading", "ready",
          "serving", "flipping", "stopping", "stopped_clean", "dead")
TERMINAL = ("refused_preflight", "stopped_clean", "dead")
#: ein verschwundener Container in diesen Zuständen ist ein TOD (stopping = gewollt)
LIVE = ("launching", "loading", "ready", "serving", "flipping")
#: Rang je nicht-terminalem Zustand: ein Übergang nur auf gleichen oder höheren
#: Rang (serving ⇄ flipping teilen einen). dead schlägt alles Nicht-Terminale,
#: refused_preflight nur aus preflight, stopped_clean nur aus stopping.
ORDER = {"preflight": 0, "launching": 1, "loading": 2, "ready": 3,
         "serving": 4, "flipping": 4, "stopping": 5}
KINDS = ("dry", "d2", "boot")
#: §2.2 origin-Enum; `host` (das Abnahme-Skript selbst) aus der 27B-Abnahme A2
ORIGINS = ("preflight", "launcher", "rank", "front", "deadman", "operator", "container_exit", "oom", "host")
#: stopping-/stopped_clean-Gründe (cause.code, origin operator); stopped_clean trägt cause.rc = 0 (A3)
STOP_REASONS = ("stop_file", "operator", "window_end", "max_deaths", "probes_done", "d2_done")
WRITERS = ("host", "launcher", "front")
#: A1c Feld-Eigentum: oberstes Feld je Schreiber (heartbeat.<name> schreibt jeder für sich)
OWNED_FIELDS = {
    "host": ("tag", "line", "rev", "image", "image_id", "container", "profile", "gpuq_id", "front"),
    "launcher": ("groups", "invariants"),
    "front": ("front",),
}
#: A1c Zustands-Eigentum
OWNED_STATES = {
    "host": ("preflight", "refused_preflight", "launching", "loading", "serving", "flipping",
             "stopping", "stopped_clean", "dead"),
    "launcher": ("loading", "ready", "dead"),
    "front": ("serving", "flipping"),
}
#: A1c: ein `dead` des Launchers trägt nur diese Ursprünge (Host: Container-Exit, OOM, Wächter, host)
LAUNCHER_DEAD_ORIGINS = ("launcher", "rank")
#: Heartbeat-Name je Schreiber (H3: die Namen sind offen, Ränge schreiben "<G>.tp<t>pp<p>")
HEARTBEAT_NAME = {"host": "host_acceptance", "launcher": "launcher", "front": "front"}
#: Verbraucher-Regel §2.2: Herzschlag älter als das = verdächtig
HEARTBEAT_STALE_S = 30.0
#: Aus Abwärtsverträglichkeit: der Host-Schreiber durfte immer genau diese Felder
SETTABLE = OWNED_FIELDS["host"]
EVENT_MAX = 4000

#: rc-Tabelle (IPC-STATE-PLAN §2.2, verbindlich); steht zusätzlich in cause.rc
RC_OK = 0
RC_HOST_ABORT = 1
RC_USAGE = 2
RC_SEED_NEEDED = 10
RC_REFUSED_PREFLIGHT = 20
RC_DEAD_BEFORE_SERVING = 21
RC_LOAD_TIMEOUT = 22
RC_DEAD_AFTER_SERVING = 23
RC_STOPPED_BY_WATCHER = 24
#: rc, die ein TOD sind (nur bei kind=boot zählt ein Tod, H2)
RC_DEATH = (RC_HOST_ABORT, RC_DEAD_BEFORE_SERVING, RC_LOAD_TIMEOUT, RC_DEAD_AFTER_SERVING, RC_STOPPED_BY_WATCHER)


class StateFileError(SystemExit):
    """Ein Aufruf, der das Format oder das Feld-/Zustands-Eigentum verletzt."""


def _now() -> float:
    return round(time.time(), 3)


def _path(d: str) -> str:
    return os.path.join(d, "state.json")


def read(d: str) -> dict:
    try:
        with open(_path(d)) as f:
            st = json.load(f)
    except FileNotFoundError:
        return {}
    if st.get("schema") != STATE_SCHEMA:
        raise StateFileError(f"state_file: {_path(d)} hat schema {st.get('schema')!r}, erwartet {STATE_SCHEMA}")
    return st


def events(d: str) -> list:
    try:
        with open(os.path.join(d, "events.jsonl")) as f:
            return [json.loads(x) for x in f if x.strip()]
    except FileNotFoundError:
        return []


def served(d: str) -> bool:
    return any(e.get("type") == "lifecycle" and (e.get("data") or {}).get("state") == "serving" for e in events(d))


def counts_as_death(st: dict) -> bool:
    """H2: MAX_DEATHS zählt nur `dead` eines kind=boot. Ein dead im d2/dry ist ein
    roter Trockenlauf, refused_preflight ist nie ein Tod."""
    return st.get("kind") == "boot" and (st.get("lifecycle") or {}).get("state") == "dead"


def may_transition(cur: str, new: str) -> bool:
    """Nur vorwärts; terminal bleibt terminal; dead schlägt jeden nicht-terminalen Zustand."""
    if cur in TERMINAL or new == cur:
        return False
    if new == "dead":
        return True
    if new == "refused_preflight":
        return cur == "preflight"
    if new == "stopped_clean":
        return cur == "stopping"
    return ORDER.get(new, -1) >= ORDER.get(cur, 99)


def _atomic_write(path: str, obj: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def write_json_atomic(path: str, obj: dict) -> None:
    """Für Nebendateien des Boot-Verzeichnisses (stop_request.json, A5)."""
    _atomic_write(path, obj)


@contextlib.contextmanager
def _locked(d: str):
    os.makedirs(d, exist_ok=True)
    fd = os.open(os.path.join(d, ".lock"), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _event(d: str, st: dict, typ: str, *, group=None, rank=None, code=None, data=None) -> None:
    ev = {"schema": EVENT_SCHEMA, "seq": st["seq"], "ts": _now(), "boot_id": st["boot_id"], "type": typ,
          "group": group, "rank": rank, "code": code, "data": data or {}}
    line = json.dumps(ev, sort_keys=True)
    if len(line.encode()) > EVENT_MAX:   # §2.2: detail_full > 4 KiB in die Nebendatei
        cause = dict((ev["data"].get("cause") or {}))
        side = f"cause-{st['seq']}.txt"
        with open(os.path.join(d, side), "w") as f:
            f.write(str(cause.get("detail_full") or ""))
        cause["detail_full"] = {"file": side}
        ev["data"] = dict(ev["data"], cause=cause)
        line = json.dumps(ev, sort_keys=True)[:EVENT_MAX]
    fd = os.open(os.path.join(d, "events.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, (line + "\n").encode())
    finally:
        os.close(fd)


def _in_container() -> bool:
    return os.path.exists("/.dockerenv")


def _heartbeat(st: dict, writer: str = "host", container=None) -> None:
    """A4: je Eintrag {pid, ts, seq, container, host_pid}. `pid` im Namensraum des
    Schreibers; ein Host-Leser prüft `host_pid` (kill -0) bzw. `container`
    (docker inspect .State.Running), nie eine Container-PID."""
    hb = st.setdefault("heartbeat", {})
    name = HEARTBEAT_NAME[writer]
    me = hb.get(name) or {"seq": 0}
    if writer == "host":
        pid = int(os.environ.get("ACC_WRITER_PID") or os.getppid())
    else:
        pid = os.getpid()
    inside = _in_container()
    host_pid = int(os.environ["WEG2_HOST_PID"]) if os.environ.get("WEG2_HOST_PID") else (None if inside else pid)
    if container is None:
        container = os.environ.get("WEG2_CONTAINER") or (os.environ.get("HOSTNAME") if inside else None) \
            or me.get("container")
    hb[name] = {"pid": pid, "ts": _now(), "seq": int(me.get("seq", 0)) + 1,
                "container": container, "host_pid": host_pid}


def heartbeat_verdict(entry: dict, now=None, *, container_running=None, pid_alive=None) -> str:
    """Verbraucher-Regel §2.2/A4: `alive` | `unknown` (Hänger) | `dead`. Frisch = alive.
    Veraltet und der Prozess nachweislich weg (Container nicht Running bzw. host_pid
    tot) = dead; veraltet ohne diesen Nachweis = unknown. Nie `alive` aus Stille."""
    now = time.time() if now is None else now
    if entry and now - float(entry.get("ts") or 0) <= HEARTBEAT_STALE_S:
        return "alive"
    gone = container_running is False or pid_alive is False
    return "dead" if gone else "unknown"


def make_cause(code, origin, detail="", *, group=None, rank=None, name=None, exception_type=None,
               log_ref=None, rc=None) -> dict:
    if origin not in ORIGINS:
        raise StateFileError(f"state_file: origin {origin!r} (erlaubt: {', '.join(ORIGINS)})")
    return {"code": code, "name": name, "origin": origin, "group": group, "rank": rank,
            "exception_type": exception_type, "detail_full": detail, "log_ref": log_ref, "rc": rc}


def _check_owner(writer: str, state, cause, fields) -> None:
    if writer not in WRITERS:
        raise StateFileError(f"state_file: Schreiber {writer!r} (erlaubt: {', '.join(WRITERS)})")
    if state is not None and state not in OWNED_STATES[writer]:
        raise StateFileError(f"state_file: Zustand {state!r} gehoert nicht dem Schreiber {writer!r}")
    for k in (fields or {}):
        if k.split(".")[0] not in OWNED_FIELDS[writer]:
            raise StateFileError(f"state_file: Feld {k!r} gehoert nicht zu {STATE_SCHEMA}/{writer} "
                                 f"(erlaubt: {', '.join(OWNED_FIELDS[writer])})")
    if writer == "launcher" and state == "dead" and (cause or {}).get("origin") not in LAUNCHER_DEAD_ORIGINS:
        raise StateFileError(f"state_file: dead des Launchers braucht origin in {LAUNCHER_DEAD_ORIGINS}")


def transition(d: str, state=None, *, if_state=None, cause=None, fields=None, heartbeat_only=False,
               writer: str = "host", container=None) -> dict:
    """Ein Schreibvorgang unter der Sperre. Übergänge nur vorwärts (``may_transition``);
    ``if_state`` begrenzt die Ausgangszustände zusätzlich. Ein Übergang wird als
    ``lifecycle``-Event geschrieben. Ein Übergang, den die Ordnung verbietet (ein
    zweiter Schreiber war schon weiter), ist KEIN Fehler, sondern ein No-op."""
    if state is not None and state not in STATES:
        raise StateFileError(f"state_file: unbekannter Zustand {state!r} (erlaubt: {', '.join(STATES)})")
    _check_owner(writer, state, cause, fields)
    with _locked(d):
        st = read(d)
        if not st:
            raise StateFileError(f"state_file: {_path(d)} fehlt (erst init)")
        cur = st["lifecycle"]["state"]
        change = (state is not None and may_transition(cur, state)
                  and (if_state is None or cur in if_state))
        if not (heartbeat_only or change or fields or (cause is not None and state is None)):
            return st
        st["seq"] = int(st.get("seq", 0)) + 1
        _heartbeat(st, writer, container)
        for k, v in (fields or {}).items():
            cur_obj, parts = st, k.split(".")
            for p in parts[:-1]:
                cur_obj = cur_obj.setdefault(p, {})
            cur_obj[parts[-1]] = v
        if cause is not None and (change or state is None):
            st["cause"] = cause
        if change:
            st["lifecycle"] = {"state": state, "since_ts": _now(), "prev": cur}
            if state == "serving" and not st.get("serving_since_ts"):
                st["serving_since_ts"] = st["lifecycle"]["since_ts"]   # H1
            if state == "stopped_clean":
                c = dict(st.get("cause") or make_cause("operator", "operator"))
                c["rc"] = RC_OK   # A3: die stopping-Ursache bleibt stehen, rc 0
                st["cause"] = c
            if state == "refused_preflight" and cause is not None:
                st.setdefault("preflight", {}).setdefault("checks", []).append(
                    {"name": cause["code"], "ok": False, "value": cause.get("detail_full"), "need": None})
        _atomic_write(_path(d), st)
        if change:
            c = st.get("cause") if state in ("refused_preflight", "dead", "stopping") else None
            _event(d, st, "lifecycle", group=(c or {}).get("group"), code=(c or {}).get("code"),
                   data={"state": state, "prev": cur, **({"cause": c} if c else {})})
        return st


def add_event(d: str, typ: str, data: dict, *, writer: str = "host", group=None, rank=None, code=None) -> None:
    with _locked(d):
        st = read(d)
        st["seq"] = int(st.get("seq", 0)) + 1
        _heartbeat(st, writer)
        _atomic_write(_path(d), st)
        _event(d, st, typ, group=group, rank=rank, code=code, data=data)


def new_boot_id(prefix: str, kind: str) -> str:
    """§2.2/A7: <label>-<kind>-<UTC yyyymmddThhmmssZ>-<4 hex>; <label> = Aufrufer-Label,
    nicht der Container-Tag (der steht in state.tag)."""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{kind}-{ts}-{secrets.token_hex(2)}"


def pointer_name(kind: str) -> str:
    """A6: `current` nur für kind=boot, je andere Laufart ein eigener Zeiger."""
    return "current" if kind == "boot" else f"current-{kind}"


def init(root: str, boot_id: str, kind: str, fields: dict) -> str:
    if kind not in KINDS:
        raise StateFileError(f"state_file: kind {kind!r} (erlaubt: {', '.join(KINDS)})")
    for k in fields:
        if k not in OWNED_FIELDS["host"]:
            raise StateFileError(f"state_file: Feld {k!r} gehoert nicht zu {STATE_SCHEMA}")
    d = os.path.join(root, boot_id)
    with _locked(d):
        if os.path.exists(_path(d)):
            raise StateFileError(f"state_file: {_path(d)} existiert schon (boot_id doppelt)")
        st = {"schema": STATE_SCHEMA, "seq": 1, "boot_id": boot_id, "kind": kind,
              "tag": None, "line": None, "rev": None, "image": None, "image_id": None,
              "container": None, "profile": None, "gpuq_id": None,
              "lifecycle": {"state": "preflight", "since_ts": _now(), "prev": None},
              "serving_since_ts": None,
              "cause": None, "heartbeat": {}, "groups": {},
              "front": {"state": "down", "epoch": 0, "awake": None, "queue": 0, "outstanding": 0},
              "preflight": {"checks": []}, "invariants": {}}
        st.update(fields)
        _heartbeat(st, "host")
        _atomic_write(_path(d), st)
        _event(d, st, "lifecycle", data={"state": "preflight", "prev": None})
    link = os.path.join(root, pointer_name(kind))   # atomar per rename
    tmp = f"{link}.tmp.{os.getpid()}"
    with contextlib.suppress(FileNotFoundError):
        os.unlink(tmp)
    os.symlink(boot_id, tmp)
    os.replace(tmp, link)
    return d


def _docker_state(container: str) -> dict:
    try:
        out = subprocess.run(["docker", "inspect", "-f", "{{json .State}}", container],
                             capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else {}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def _docker_tail(container: str, n: int = 3) -> str:
    try:
        out = subprocess.run(["docker", "logs", "--tail", str(n), container],
                             capture_output=True, text=True, timeout=30)
        return (out.stdout + out.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def stop_request(d: str):
    """A5: stop_request.json des Wächters {code, origin, group, rank, detail_full}. Er hat
    den Container gestoppt; seine Ursache ist die wahre, rc 24."""
    try:
        with open(os.path.join(d, "stop_request.json")) as f:
            r = json.load(f)
    except (FileNotFoundError, ValueError):
        return None
    origin = str(r.get("origin") or "deadman")
    if origin not in ORIGINS:
        origin = "deadman"
    return make_cause(str(r.get("code") or "STOP_REQUEST"), origin, str(r.get("detail_full") or ""),
                      group=r.get("group"), rank=r.get("rank"), rc=RC_STOPPED_BY_WATCHER)


def death_cause(d: str, container: str, code=None) -> dict:
    """Ursache eines verschwundenen Containers: Stop-Anfrage des Wächters vor docker inspect
    (ExitCode, OOMKilled, Error). detail_full nennt die letzten Container-Zeilen, NUR für
    Menschen."""
    req = stop_request(d)
    if req is not None:
        return req
    ds = _docker_state(container)
    origin = "container_exit"
    if code is None:
        if ds.get("OOMKilled"):
            code, origin = "OOM_KILLED", "oom"
        elif ds:
            code = f"CONTAINER_EXIT_{ds.get('ExitCode')}"
        else:
            code = "CONTAINER_GONE"
    parts = []
    if ds:
        parts.append(f"exit={ds.get('ExitCode')} oom={ds.get('OOMKilled')} error={ds.get('Error') or '-'}")
    tail = _docker_tail(container)
    if tail:
        parts.append("letzte Container-Zeilen (nur fuer Menschen, Phase 2): " + tail.replace("\n", " | "))
    return make_cause(code, origin, "; ".join(parts))


def rc_of(st: dict, was_serving: bool) -> int:
    s = (st.get("lifecycle") or {}).get("state")
    cause = st.get("cause") or {}
    if cause.get("rc") is not None and s in TERMINAL:
        return int(cause["rc"])
    if s == "stopped_clean":
        return RC_OK
    if s == "refused_preflight":
        return RC_REFUSED_PREFLIGHT
    if s == "dead":
        if cause.get("origin") == "host":
            return RC_HOST_ABORT
        if cause.get("code") == "LOAD_TIMEOUT":
            return RC_LOAD_TIMEOUT
        return RC_DEAD_AFTER_SERVING if was_serving else RC_DEAD_BEFORE_SERVING
    return RC_HOST_ABORT   # nicht terminal am Ende = Host-Fehler


def _front_of(js: str):
    try:
        f = json.loads(js)
    except (TypeError, ValueError):
        return None
    out = f.get("outstanding")
    return {"state": f.get("state") or "down", "epoch": int(f.get("epoch") or 0), "awake": f.get("awake"),
            "queue": int(f.get("queue") or 0),
            "outstanding": sum(out.values()) if isinstance(out, dict) else int(out or 0)}


def _kv(items):
    out = {}
    for it in items or ():
        k, _, v = it.partition("=")
        out[k] = v
    return out


def _kvjson(items):
    out = {}
    for it in items or ():
        k, _, v = it.partition("=")
        out[k] = json.loads(v)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("--root", required=True); p.add_argument("--boot-id", required=True)
    p.add_argument("--kind", required=True); p.add_argument("--field", action="append")
    p = sub.add_parser("set"); p.add_argument("--dir", required=True); p.add_argument("--state")
    p.add_argument("--writer", default="host")
    p.add_argument("--if-state"); p.add_argument("--cause-code"); p.add_argument("--cause-origin")
    p.add_argument("--cause-detail", default=""); p.add_argument("--cause-group")
    p.add_argument("--field", action="append"); p.add_argument("--json", action="append")
    p = sub.add_parser("event"); p.add_argument("--dir", required=True); p.add_argument("--type", required=True)
    p.add_argument("--json", action="append")
    p = sub.add_parser("beat"); p.add_argument("--dir", required=True); p.add_argument("--container")
    p.add_argument("--front-json", default="")
    p = sub.add_parser("dead"); p.add_argument("--dir", required=True); p.add_argument("--container", required=True)
    p.add_argument("--code")
    p = sub.add_parser("finish"); p.add_argument("--dir", required=True)
    p = sub.add_parser("rc"); p.add_argument("--dir", required=True)
    p = sub.add_parser("get"); p.add_argument("--dir", required=True); p.add_argument("--key")
    p = sub.add_parser("new-boot-id"); p.add_argument("--prefix", required=True); p.add_argument("--kind", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "new-boot-id":
        print(new_boot_id(a.prefix, a.kind))
        return 0
    if a.cmd == "init":
        print(init(a.root, a.boot_id, a.kind, _kv(a.field)))
        return 0
    if a.cmd == "set":
        cause = None
        if a.cause_code:
            cause = make_cause(a.cause_code, a.cause_origin or "host", a.cause_detail, group=a.cause_group)
        transition(a.dir, a.state, if_state=a.if_state.split(",") if a.if_state else None, cause=cause,
                   fields={**_kv(a.field), **_kvjson(a.json)}, writer=a.writer)
        return 0
    if a.cmd == "event":
        add_event(a.dir, a.type, _kvjson(a.json).get("data") or {})
        return 0
    if a.cmd == "beat":
        # Herzschlag des Host-Schreibers + Front-Spiegel; Container weg in einem lebenden
        # Zustand = Tod; serving <-> flipping folgt der Front (sonst nur vorwärts).
        st = read(a.dir)
        cur = (st.get("lifecycle") or {}).get("state")
        if cur in TERMINAL:
            return 3
        if a.container and not _docker_state(a.container).get("Running") and cur in LIVE:
            transition(a.dir, "dead", if_state=list(LIVE), cause=death_cause(a.dir, a.container),
                       container=a.container)
            return 3
        front = _front_of(a.front_json)
        fields = {"front": front} if front else {}
        fs = (front or {}).get("state")
        if fs in ("serving", "flipping") and cur in ("serving", "flipping") and fs != cur:
            transition(a.dir, fs, if_state=["serving", "flipping"], fields=fields, container=a.container)
        else:
            transition(a.dir, None, fields=fields, heartbeat_only=True, container=a.container)
        return 0
    if a.cmd == "dead":
        transition(a.dir, "dead", cause=death_cause(a.dir, a.container, a.code), container=a.container)
        return 0
    if a.cmd == "finish":
        req = stop_request(a.dir)
        if req is not None:
            transition(a.dir, "dead", cause=req)
        else:
            transition(a.dir, "stopped_clean", if_state=["stopping"])
        st = read(a.dir)
        rc = rc_of(st, served(a.dir))
        cause = dict(st.get("cause") or make_cause("operator", "operator"))
        if cause.get("rc") != rc:
            cause["rc"] = rc
            transition(a.dir, None, cause=cause)
        return 0
    if a.cmd == "get":
        v = read(a.dir)
        for part in (a.key or "").split(".") if a.key else ():
            v = v.get(part) if isinstance(v, dict) else None
        print(json.dumps(v) if isinstance(v, (dict, list)) else ("" if v is None else v))
        return 0
    if a.cmd == "rc":
        print(rc_of(read(a.dir), served(a.dir)))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
