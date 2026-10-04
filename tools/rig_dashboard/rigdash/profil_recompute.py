"""Profil-Editor S4b (Auftrag 1432): ``POST /api/profil/recompute`` — der Dienst hinter dem Balken.

Der Dashboard-Prozess rechnet nichts selbst (kein sglang-Import, MemoryMax=1G).  ``CouplingsService`` hält EINEN langlebigen Kindprozess
(``kartenplan_build/couplings_worker.py``, JSON-Zeilen über stdin/stdout) mit dem Python der sglang-Umgebung, startet ihn bei der ersten Anfrage
(nie im Hintergrund), startet ihn nach Tod oder Zeitüberschreitung neu und antwortet bei jedem Fehler mit ``{"ok": False, "error": ...}``.

``build_request`` baut aus dem Serverprofil (``flliper.server/1``), dem Hardwareprofil und dem Modellprofil die Anfrage an
``planner/profile_couplings.run``.  Hardware- und Modellprofil holt der Aufrufer (server.py) aus den vorhandenen Diensten
(``hwprofil.get``, ``modellprofil.estimate``); dieses Modul kennt weder HTTP noch Dateien des Hosts."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Mapping, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(os.path.dirname(HERE), "kartenplan_build", "couplings_worker.py")
DEFAULT_PYTHON = os.environ.get("RIGDASH_COUPLINGS_PYTHON") or "/spinning/htsglang-gpu/.venv/bin/python"
#: höchstens so lange wartet eine Anfrage auf den Worker (der Start importiert sglang.srt.planner)
TIMEOUT_S = 60.0
START_TIMEOUT_S = 120.0
#: Operationen, die der Editor verlangen darf
WHAT = ("bars", "compute", "move", "chunk", "context")


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


def build_request(body: Mapping[str, Any], *, hardware: Mapping[str, Any], model: Mapping[str, Any]) -> Dict[str, Any]:
    """Anfrage an ``profile_couplings.run`` aus dem Körper der Route."""
    what = str(body.get("what") or "bars")
    if what not in WHAT:
        raise RecomputeError("what muss eines von %s sein" % ", ".join(WHAT))
    doc = body.get("doc", {})
    if not isinstance(doc, dict):
        raise RecomputeError("doc muss ein JSON-Objekt sein")
    settings = body.get("settings", {})
    phases = body.get("phases")
    if not isinstance(settings, dict) or (phases is not None and not isinstance(phases, dict)):
        raise RecomputeError("settings und phases müssen JSON-Objekte sein")
    req: Dict[str, Any] = {"what": what, "hardware": hardware, "model": model, "settings": settings, "server_args": args_of(doc)}
    if phases:
        req["phases"] = phases
    for k in ("src", "dst", "n", "new_chunk_tokens"):
        if k in body:
            req[k] = body[k]
    return req


class CouplingsService:
    """Ein langlebiger Worker, eine Anfrage nach der anderen (Sperre)."""

    def __init__(self, tree_python: Optional[str], python: Optional[str] = None, worker: str = WORKER,
                 timeout_s: float = TIMEOUT_S, start_timeout_s: float = START_TIMEOUT_S):
        self.tree_python = tree_python
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
            return "kein Planer-Baum mit planner/profile_couplings.py (KARTENPLAN_TREE bzw. install_510.sh)"
        if not os.path.isfile(self.python):
            return "Python der sglang-Umgebung fehlt: %s (RIGDASH_COUPLINGS_PYTHON)" % self.python
        self._proc = subprocess.Popen([self.python, self.worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      env=self._env(), close_fds=True)
        self.starts += 1
        line = self._readline(self.start_timeout_s)
        if not line:
            self._stop()
            return "Kopplungs-Worker startet nicht (Zeitüberschreitung oder sofort beendet)"
        try:
            hello = json.loads(line)
        except ValueError:
            self._stop()
            return "Kopplungs-Worker meldet sich ohne JSON"
        if not hello.get("ok"):
            self._stop()
            return str(hello.get("error") or "Kopplungs-Worker nicht bereit")
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
                return {"ok": False, "error": "Kopplungs-Worker ist beendet (wird bei der nächsten Eingabe neu gestartet)"}
            while True:
                line = self._readline(self.timeout_s)
                if line is None:
                    self._stop()
                    return {"ok": False, "error": "Kopplungsrechnung nicht fertig nach %.0f s (Worker neu gestartet)" % self.timeout_s}
                if line == "":
                    self._stop()
                    return {"ok": False, "error": "Kopplungs-Worker ist während der Rechnung gestorben"}
                try:
                    res = json.loads(line)
                except ValueError:
                    continue
                if res.get("id") == rid:
                    res.pop("id", None)
                    return res               # eine verspätete Antwort einer früheren Anfrage wird übersprungen

    def close(self) -> None:
        with self._lock:
            self._stop()
