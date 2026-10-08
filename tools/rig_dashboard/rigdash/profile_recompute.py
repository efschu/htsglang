"""Profil-Editor S4b (Auftrag 1432): ``POST /api/profil/recompute`` — der Dienst hinter dem Balken.

Der Dashboard-Prozess rechnet nichts selbst (kein flliper-Import, MemoryMax=1G).  ``CouplingsService`` hält EINEN langlebigen Kindprozess
(``kartenplan_build/couplings_worker.py``, JSON-Zeilen über stdin/stdout) mit dem Python der flliper-Umgebung, startet ihn bei der ersten Anfrage
(nie im Hintergrund), startet ihn nach Tod oder Zeitüberschreitung neu und antwortet bei jedem Fehler mit ``{"ok": False, "error": ...}``.

``build_request`` baut aus dem Serverprofil (``flliper.server/1``), dem Hardwareprofil und dem Modellprofil die Anfrage an
``planner/profile_couplings.run``.  Hardware- und Modellprofil holt der Aufrufer (server.py) aus den vorhandenen Diensten
(``hwprofil.get``, ``modellprofil.estimate``); dieses Modul kennt weder HTTP noch Dateien des Hosts."""

from __future__ import annotations

import json
import os
import select
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(os.path.dirname(HERE), "kartenplan_build", "couplings_worker.py")
DEFAULT_PYTHON = os.environ.get("RIGDASH_COUPLINGS_PYTHON") or "/spinning/htsglang-gpu/.venv/bin/python"
#: höchstens so lange wartet eine Anfrage auf den Worker (der Start importiert flliper.srt.planner)
TIMEOUT_S = 60.0
START_TIMEOUT_S = 120.0
#: Operationen, die der Editor verlangen darf
WHAT = ("bars", "phase_bars", "compute", "move", "chunk", "context")
#: Gruppenzeilen des Profils: ``--extra-p/-d`` tragen Flags der Gruppe, ``--env-p/-d`` ihre Umgebung (AP-H2: Balken je Phase)
GROUP_EXTRA = {"--extra-p": "P", "--extra-d": "D"}
GROUP_ENV = {"--env-p": "P", "--env-d": "D"}


class RecomputeError(ValueError):
    """Eine Anfrage, die schon vor dem Worker unbrauchbar ist (400)."""


def args_of(doc: Mapping[str, Any]) -> Dict[str, str]:
    """Flag -> letzter Wert aus ``doc["args"]`` (wie ``profile_json.args_dict``; hier lokal, damit der Dienst ohne den Planer-Baum auskommt)."""
    out: Dict[str, str] = {}
    for e in (doc.get("args") or []):
        if isinstance(e, Mapping) and "flag" in e:
            v = e.get("values") or []
            out[str(e["flag"])] = " ".join(str(x) for x in v) if v else ""
    return out


def _entry_text(e: Mapping[str, Any]) -> str:
    """Der Text eines Argumenteintrags wie er im Profil stand (``flag=value`` bei ``eq``, sonst ``flag value...``)."""
    vals = [str(v) for v in (e.get("values") or [])]
    if e.get("eq") and vals:
        return str(e["flag"]) + "=" + vals[0]
    return " ".join([str(e["flag"])] + vals)


def group_texts(doc: Mapping[str, Any]) -> Dict[str, str]:
    """``--extra-p`` / ``--extra-d`` / ``--env-p`` / ``--env-d`` -> Text.  Zwei Darstellungen kommen vor: der Wert steht in ``values``
    (Profil mit Launcher-Specs) oder -- ohne Specs importiert -- als eigener Eintrag danach (der ganze Text steht dann in ``flag``)."""
    args = [e for e in (doc.get("args") or []) if isinstance(e, Mapping)]
    out: Dict[str, str] = {}
    i = 0
    while i < len(args):
        e = args[i]
        f = e.get("flag")
        if f in GROUP_EXTRA or f in GROUP_ENV:
            text = " ".join(str(v) for v in (e.get("values") or []))
            if not text and i + 1 < len(args) and "flag" in args[i + 1] and (" " in str(args[i + 1]["flag"]) or "=" in str(args[i + 1]["flag"])):
                text = _entry_text(args[i + 1])
                i += 1
            if text:
                out[str(f)] = text
        i += 1
    return out


def scoped_args(text: str) -> Dict[str, str]:
    """Flags eines ``--extra-p/-d``-Textes: Flag -> Wert (mehrere Werte mit Leerzeichen, ``--flag=value`` und ``--flag value`` beide)."""
    try:
        toks = shlex.split(text)
    except ValueError:
        toks = text.split()
    out: Dict[str, str] = {}
    cur: Optional[str] = None
    vals: List[str] = []

    def flush() -> None:
        if cur is not None:
            out[cur] = " ".join(vals)

    for t in toks:
        if t.startswith("--"):
            flush()
            if "=" in t:
                cur, v = t.split("=", 1)
                vals = [v]
            else:
                cur, vals = t, []
        elif cur is not None:
            vals.append(t)
    flush()
    return out


def env_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in str(text).split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def phase_inputs(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """Gruppenzeilen eines Profils fuer ``what=phase_bars``: ``phase_args`` und ``phase_env`` je ``P``/``D`` und die nackten Token."""
    gt = group_texts(doc)
    pa = {ph: scoped_args(gt.get(f, "")) for f, ph in GROUP_EXTRA.items()}
    pe = {ph: env_map(gt.get(f, "")) for f, ph in GROUP_ENV.items()}
    tokens = [str(e["token"]) for e in (doc.get("args") or []) if isinstance(e, Mapping) and "token" in e]
    return {"phase_args": pa, "phase_env": pe, "tokens": tokens}


def draft_path_of(doc: Mapping[str, Any], body: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Das Draft-Verzeichnis des Profils: Koerper, ``--dflash-draft-path``, ``--speculative-draft-model-path`` (Gruppenzeilen), ``PROFILE_DRAFT``."""
    if body and body.get("draft_path"):
        return str(body["draft_path"])
    a = args_of(doc)
    if a.get("--dflash-draft-path"):
        return a["--dflash-draft-path"]
    pi = phase_inputs(doc)
    for ph in ("D", "P"):
        v = pi["phase_args"][ph].get("--speculative-draft-model-path")
        if v:
            return v
    if a.get("--speculative-draft-model-path"):
        return a["--speculative-draft-model-path"]
    for v in (doc.get("vars") or []):
        if isinstance(v, Mapping) and v.get("name") == "PROFILE_DRAFT" and v.get("value"):
            return str(v["value"])
    return None


def build_request(body: Mapping[str, Any], *, hardware: Mapping[str, Any], model: Mapping[str, Any]) -> Dict[str, Any]:
    """Anfrage an ``profile_couplings.run`` aus dem Körper der Route."""
    what = str(body.get("what") or "bars")
    if what not in WHAT:
        raise RecomputeError("what must be one of %s" % ", ".join(WHAT))
    doc = body.get("doc", {})
    if not isinstance(doc, dict):
        raise RecomputeError("doc must be a JSON object")
    settings = body.get("settings", {})
    phases = body.get("phases")
    if not isinstance(settings, dict) or (phases is not None and not isinstance(phases, dict)):
        raise RecomputeError("settings and phases must be JSON objects")
    req: Dict[str, Any] = {"what": what, "hardware": hardware, "model": model, "settings": settings, "server_args": args_of(doc)}
    if phases:
        req["phases"] = phases
    for k in ("src", "dst", "n", "new_chunk_tokens"):
        if k in body:
            req[k] = body[k]
    if what == "phase_bars":
        req.update(phase_inputs(doc))
        if body.get("form"):
            req["form"] = str(body["form"])
    return req


class CouplingsService:
    """Ein langlebiger Worker, eine Anfrage nach der anderen (Sperre)."""

    def __init__(self, tree_python: Optional[str], python: Optional[str] = None, worker: str = WORKER,
                 timeout_s: float = TIMEOUT_S, start_timeout_s: float = START_TIMEOUT_S, prefix: Optional[Sequence[str]] = None):
        self.tree_python = tree_python
        #: Befehlspraefix des Kindprozesses (z. B. ``systemd-run --scope -q -p MemoryMax=4G``: eigener cgroup-Rahmen ausserhalb der Unit);
        #: startet der Kindprozess mit Praefix sofort nicht (kein systemd-run / kein D-Bus), laeuft er einmal ohne ihn (``prefix_fallback``)
        self.prefix: List[str] = list(prefix or [])
        self.prefix_fallback = False
        self.python = python or DEFAULT_PYTHON
        self.worker = worker
        self.timeout_s = timeout_s
        self.start_timeout_s = start_timeout_s
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._buf = b""
        self._seq = 0
        self.starts = 0

    # ------------------------------------------------------------------ Prozess
    def _env(self) -> Dict[str, str]:
        return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": self.tree_python or "", "CUDA_VISIBLE_DEVICES": "",
                "HOME": os.environ.get("KARTENPLAN_RIG_HOME", "/root"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONWARNINGS": "ignore"}

    def _readline(self, timeout: float) -> Optional[str]:
        proc = self._proc
        deadline = time.time() + timeout
        while b"\n" not in self._buf:
            left = deadline - time.time()
            if left <= 0:
                return None
            r, _, _ = select.select([proc.stdout], [], [], min(left, 1.0))
            if r:
                chunk = os.read(proc.stdout.fileno(), 1 << 16)
                if not chunk:
                    return ""             # EOF: Worker tot
                self._buf += chunk
            elif proc.poll() is not None:
                return ""
        line, _, self._buf = self._buf.partition(b"\n")
        return line.decode("utf-8", "replace")

    def _stop(self) -> None:
        proc, self._proc, self._buf = self._proc, None, b""
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 -- schon tot
                pass

    def _ensure(self) -> Optional[str]:
        """Fehlertext oder ``None``, wenn der Worker bereit ist."""
        if self._proc is not None and self._proc.poll() is None:
            return None
        self._stop()
        if not self.tree_python or not os.path.isdir(self.tree_python):
            return "no planner tree with planner/profile_couplings.py (KARTENPLAN_TREE or install_510.sh)"
        if not os.path.isfile(self.python):
            return "Python of the flliper environment is missing: %s (RIGDASH_COUPLINGS_PYTHON)" % self.python

        def spawn(cmd: List[str]) -> Optional[str]:
            try:
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                              env=self._env(), close_fds=True)
            except OSError:
                return ""                 # wie EOF: der Befehl (Praefix) war nicht startbar
            self.starts += 1
            return self._readline(self.start_timeout_s)

        line = spawn((([] if self.prefix_fallback else self.prefix)) + [self.python, self.worker])
        if line == "" and self.prefix and not self.prefix_fallback:
            # der Praefix selbst scheiterte (nicht startbar / EOF vor dem ersten Wort): einmal ohne ihn, mit Vermerk
            self._stop()
            self.prefix_fallback = True
            line = spawn([self.python, self.worker])
        if self._proc is None:
            return "Couplings worker does not start (command cannot be started)"
        if not line:
            self._stop()
            return "Couplings worker does not start (timeout or exited at once)"
        try:
            hello = json.loads(line)
        except ValueError:
            self._stop()
            return "Couplings worker answers without JSON"
        if not hello.get("ok"):
            self._stop()
            return str(hello.get("error") or "Couplings worker not ready")
        return None

    # ------------------------------------------------------------------ Anfrage
    def request(self, req: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            err = self._ensure()
            if err:
                return {"ok": False, "error": err}
            self._seq += 1
            rid = self._seq
            try:
                self._proc.stdin.write((json.dumps(dict(req, id=rid), default=str) + "\n").encode("utf-8"))
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._stop()
                return {"ok": False, "error": "Couplings worker has exited (restarted at the next input)"}
            while True:
                line = self._readline(self.timeout_s)
                if line is None:
                    self._stop()
                    return {"ok": False, "error": "Couplings calculation not finished after %.0f s (worker restarted)" % self.timeout_s}
                if line == "":
                    self._stop()
                    return {"ok": False, "error": "Couplings worker died during the calculation"}
                try:
                    res = json.loads(line)
                except ValueError:
                    continue
                if res.get("id") == rid:
                    res.pop("id", None)
                    return res               # eine verspätete Antwort einer früheren Anfrage wird übersprungen

    def topology(self, n: int) -> Dict[str, Any]:
        """Auftrag 1984 (C): ``topology.plan_topology(n)`` im Kindprozess.  ``{"ok": True, "refused": None | "<Text>"}`` oder ``{"ok": False, "error": ..}``
        (Kind nicht verfügbar oder Importfehler: das ist keine Ablehnung)."""
        res = self.request({"what": "topology", "n": int(n)})
        if res.get("ok"):
            return {"ok": True, "refused": res.get("refused")}
        return {"ok": False, "error": str(res.get("error") or "unknown error")}

    def close(self) -> None:
        with self._lock:
            self._stop()
