"""Hardwareprofil messen und anzeigen (Auftrag 950, Profil-Editor S2): Routen und JSON, keine Oberfläche.

Zwei Routen im Rig-Dashboard (nur Edition rig, nur im LAN):

  GET  /api/hwprofil          das aktuelle Hardwareprofil ``flliper.hardware/1`` (reine Sicht auf die vorhandenen
                              Messquellen, startet nichts), dazu Fensterstatus, letzte Messung, gpuq-Karten
  POST /api/hwprofil/measure  {"cards": [<NVML-Index>, ...]}  bucht ein gpuq-Fenster und misst darin
  POST /api/hwprofil/cancel   gibt ein noch wartendes Fenster zurück

Die Messung läuft NUR in einem gebuchten gpuq-Fenster (Rig-Regel): die Route bucht selbst über die gpuq-HTTP-API,
Eigentümer ``profil-editor``, nur die zu messenden Karten, 15 min (Auftrag 1006, siehe WINDOW), ohne not_before.

  * Ist das Fenster nicht ``running`` (``pending``), gibt die Route den Fensterstatus zurück und misst NICHT.  Die
    Buchung bleibt in der Warteschlange; ein erneuter Knopfdruck derselben Karten nimmt sie wieder auf (kein
    Hintergrund-Poller, nur die offene Seite fragt; Memory KEINE-CRONS).
  * Läuft das Fenster, prüft die Route zuerst die Karten (gpuq ``used_mib`` stammt aus nvidia-smi: der Dienst sperrt
    nichts) und startet dann den Messlauf: ein Kindprozess ``python -m sglang.srt.rigmon.card_probe --run`` mit genau
    den gewählten Karten (derselbe Weg wie ``ProbeJobStore``: der Dashboard-Prozess bekommt nie einen CUDA-Kontext).
  * Nach der Messung (auch nach Fehler oder Zeitüberschreitung) gibt die Route das Fenster SOFORT zurück.

Das Fenster ist exklusiv (kein ``mib``): eine Ratenmessung neben fremder Last misst die fremde Last mit.  Wer
bewusst einen Anteil buchen will, gibt ``mib`` im Body an.

Dieser Prozess bleibt stdlib-only und ohne CUDA: ``hardware_profile.py`` des Planer-Baums wird per Dateipfad geladen
(wie ``kartenplan_gate``), NICHT per ``import sglang``.  Token der gpuq-Buchung verlassen den Prozess nie (nicht in
der JSON-Antwort); die Buchung steht zusätzlich in ``<state-dir>/hwprofil_window.json``, damit ein Neustart ein
verwaistes Fenster zurückgeben kann.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

OWNER = "profil-editor"
#: Auftrag 1006: der Messlauf umfasst jetzt je Karte alle Rechenformate (int8, NVFP4 W4A8 über JIT, Marlin) UND die
#: BAR1-Strecke je Paar (drei Kindprozesse, Erweiterung, Byte-Beweis, sechs gerichtete Paare).  Warm (JIT-Cache da)
#: etwa 4-5 min, mit kalter JIT-Übersetzung des W4A8-Kernels 8-12 min: 10 min waren dafür zu knapp, 15 min tragen den
#: kalten Lauf (Kindprozess-Deckel CHILD_CAP_S, BAR1-Schritt darin höchstens 300 s).
WINDOW = "15m"
PURPOSE = "Hardwareprofil messen (Profil-Editor S2)"
#: gpuq: ab so viel belegtem Speicher gilt eine Karte als belegt (CLAUDE.md-Regel: ~500 MiB).
BUSY_MIB = 500
#: Obergrenze für den Kindprozess (Fenster 15 min minus Sicherheitsabstand); kürzer, wenn das Fenster früher endet.
CHILD_CAP_S = 840.0
#: Weniger Restzeit als das: nicht mehr starten, Fenster zurückgeben (ein warmer Lauf braucht ~3 min allein für die Karten).
MIN_LEFT_S = 180.0
#: Sicherheitsabstand zwischen Kindprozess-Ende und Fensterende.
END_MARGIN_S = 25.0
#: Ein laufendes Fenster, das so lange ungenutzt blieb, wird beim nächsten Seitenaufruf zurückgegeben.
IDLE_RELEASE_S = 180.0
#: Das Profil ist eine Sicht, die Dateien liest; hier nur gegen Dauerfeuer der offenen Seite.
VIEW_TTL_S = 4.0

#: Release-Ausgabe (Auftrag 1984): das Hardwareprofil wird nur ANGEZEIGT; Messen bucht ein gpuq-Fenster des Rigs und bleibt zu
RELEASE_NO_MEASURE = ("Hardware messen braucht gpuq (den Fensterplan des Rigs: es bucht ein Karten-Fenster und startet einen Messlauf) "
                      "und ist in der Release-Ausgabe gesperrt. Das Hardwareprofil wird hier nur angezeigt.")
RELEASE_NO_GPUQ = "gpuq gibt es in der Release-Ausgabe nicht (Hardware messen ist dort gesperrt)"

MODULE_REL = os.path.join("sglang", "srt", "rigmon", "hardware_profile.py")
#: Planer-Bäume, in denen ``hardware_profile.py`` gesucht wird (KARTENPLAN_TREE wie beim Kartenplaner)
TREE_CANDIDATES = (
    "/opt/rigdash/kartenplan/current/python",
)


class GpuqUnavailable(RuntimeError):
    """Der gpuq-Dienst antwortet nicht: benannt, nie geraten."""


def _find_tree(explicit: Optional[str]) -> Optional[str]:
    """Ein ausdrücklich genannter Baum gilt allein (fehlt die Datei darin: kein Baum, kein stiller Ausweichpfad)."""
    if explicit:
        return explicit if os.path.isfile(os.path.join(explicit, MODULE_REL)) else None
    for t in (os.environ.get("HWPROFIL_TREE"), os.environ.get("KARTENPLAN_TREE"), *TREE_CANDIDATES):
        if t and os.path.isfile(os.path.join(t, MODULE_REL)):
            return t
    return None


def _load_module(tree_python: str):
    path = os.path.join(tree_python, MODULE_REL)
    name = "rd_hardware_profile"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def urllib_http(method: str, url: str, body: Optional[dict] = None, headers: Optional[dict] = None,
                timeout: float = 10.0) -> Tuple[int, Any]:
    """(status, geparstes JSON).  Netzfehler -> GpuqUnavailable; HTTP-Fehler liefern ihren Status und ``detail``."""
    data = json.dumps(body).encode() if body is not None else None
    h = {"content-type": "application/json"} if data is not None else {}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    except (urllib.error.URLError, OSError) as e:
        raise GpuqUnavailable("gpuq nicht erreichbar (%s): %s" % (url, e))
    try:
        return status, json.loads(raw.decode() or "null")
    except ValueError:
        return status, {"detail": raw.decode(errors="replace")[:300]}


def _window_view(b: Optional[dict]) -> Optional[dict]:
    """Was die Seite vom Fenster sieht.  Ohne Token."""
    if not b:
        return None
    return {
        "id": b.get("id"),
        "state": b.get("state"),
        "cards": b.get("cards"),
        "start": (b.get("start") or {}).get("local"),
        "end": (b.get("end") or {}).get("local"),
        "seconds_left": b.get("seconds_left"),
        "note": b.get("note") or "",
        "plan_revision": b.get("plan_revision"),
        "moved": b.get("moved"),
    }


class HwProfil:
    def __init__(self, *, gpuq: str = "http://127.0.0.1:8770", tree: Optional[str] = None, measure_tree: Optional[str] = None,
                 python: Optional[str] = None, prefix: Sequence[str] = (), cache_dir: Optional[str] = None,
                 state_dir: Optional[str] = None, http: Optional[Callable] = None, runner: Optional[Callable] = None,
                 clock: Callable[[], float] = time.time, synchronous: bool = False, edition: str = "rig"):
        self.gpuq = gpuq.rstrip("/")
        self.tree = _find_tree(tree)
        self.measure_tree = measure_tree or self.tree
        self.python = python
        self.prefix = list(prefix)
        self.cache_dir = cache_dir
        self.http = http or urllib_http
        self.runner = runner
        self.clock = clock
        self.synchronous = synchronous
        self.edition = edition
        self._mod = None
        self._lock = threading.Lock()
        self._booking: Optional[dict] = None     # {id, token, cards, booked_at, running_since}
        self._job: dict = {"state": "idle"}
        self._view_cache: Optional[Tuple[float, dict]] = None
        self._state_file = os.path.join(state_dir, "hwprofil_window.json") if state_dir else None
        self._release_orphan()

    # ------------------------------------------------------------------ Modul
    def _module(self):
        if self._mod is None:
            if not self.tree:
                raise RuntimeError(
                    "kein Planer-Baum mit sglang/srt/rigmon/hardware_profile.py gefunden (--hw-tree, HWPROFIL_TREE bzw. "
                    "KARTENPLAN_TREE; stagen: deploy/stage_hwprofil.sh)")
            self._mod = _load_module(self.tree)
        return self._mod

    # ------------------------------------------------------------------ gpuq
    def _gq(self, method: str, path: str, body: Optional[dict] = None, token: Optional[str] = None) -> Tuple[int, Any]:
        headers = {"X-Gpuq-Token": token} if token else None
        return self.http(method, self.gpuq + path, body, headers)

    def gpuq_cards(self) -> dict:
        """Die Karten des Fensterplans (Index = NVML-Index).  Nie ein Fehler nach außen: ``reachable`` sagt es.
        Release-Ausgabe: gpuq wird gar nicht angesprochen."""
        if self.edition == "release":
            return {"reachable": False, "error": RELEASE_NO_GPUQ, "cards": []}
        try:
            st, js = self._gq("GET", "/api/v1/cards")
        except GpuqUnavailable as e:
            return {"reachable": False, "error": str(e), "cards": []}
        if st != 200 or not isinstance(js, list):
            return {"reachable": False, "error": "gpuq /cards: HTTP %s" % st, "cards": []}
        return {"reachable": True, "cards": [
            {k: c.get(k) for k in ("index", "name", "short_name", "total_mib", "used_mib", "free_mib", "busy")} for c in js]}

    def _fetch_booking(self, b: dict) -> Optional[dict]:
        st, js = self._gq("GET", "/api/v1/bookings/%s" % b["id"])
        return js if st == 200 and isinstance(js, dict) else None

    # ------------------------------------------------------------------ Verwaiste Buchung
    def _persist(self):
        if not self._state_file:
            return
        try:
            if self._booking:
                os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
                with open(self._state_file, "w") as fh:
                    json.dump({k: self._booking[k] for k in ("id", "token", "cards")}, fh)
            elif os.path.exists(self._state_file):
                os.unlink(self._state_file)
        except OSError:
            pass

    def _release_orphan(self):
        """Neustart: eine Buchung, die der alte Prozess hielt, wird zurückgegeben (sie würde sonst 10 min blockieren)."""
        if not self._state_file or not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file) as fh:
                old = json.load(fh)
            self._gq("DELETE", "/api/v1/bookings/%s?reason=neustart" % old["id"], token=old.get("token"))
        except (OSError, ValueError, KeyError, GpuqUnavailable):
            pass
        try:
            os.unlink(self._state_file)
        except OSError:
            pass

    def _release(self, reason: str) -> None:
        b = self._booking
        if not b:
            return
        try:
            self._gq("DELETE", "/api/v1/bookings/%s?reason=%s" % (b["id"], reason), token=b.get("token"))
        except GpuqUnavailable:
            pass
        self._booking = None
        self._persist()

    # ------------------------------------------------------------------ Lesen
    def get(self) -> dict:
        now = self.clock()
        try:
            mod = self._module()
        except Exception as e:
            return {"ok": False, "error": str(e), "gpuq": self.gpuq_cards(), "job": dict(self._job)}
        with self._lock:
            hit = self._view_cache
            if hit and now - hit[0] < VIEW_TTL_S and self._job.get("state") != "running":
                prof = hit[1]
            else:
                prof = mod.build(cache_dir=self.cache_dir)
                self._view_cache = (now, prof)
        window = None
        with self._lock:
            b = self._booking
        if b:
            try:
                live = self._fetch_booking(b)
            except GpuqUnavailable:
                live = None
            window = _window_view(live) or {"id": b["id"], "state": "unbekannt", "cards": b["cards"]}
            self._housekeeping(b, live, now)
        return {"ok": True, "profile": prof, "problems": mod.validate(prof), "window": window,
                "job": self._job_view(), "gpuq": self.gpuq_cards(), "owner": OWNER, "window_len": WINDOW}

    def _housekeeping(self, b: dict, live: Optional[dict], now: float) -> None:
        """Ein beendetes oder verfallenes Fenster verlässt den Speicher; ein ungenutzt laufendes geht zurück."""
        state = (live or {}).get("state")
        if self._job.get("state") == "running":
            return
        with self._lock:
            if state in ("over", "released", "cancelled", "unplannable"):
                self._booking = None
                self._persist()
                return
            if state == "running":
                b.setdefault("running_since", now)
                if now - b["running_since"] > IDLE_RELEASE_S:
                    self._release("ungenutzt")
                    self._job = dict(self._job, note="Fenster ungenutzt zurückgegeben (kein Knopfdruck innerhalb %d s)" % IDLE_RELEASE_S)

    def _job_view(self) -> dict:
        j = dict(self._job)
        j.pop("result", None)
        return j

    # ------------------------------------------------------------------ Messen
    def measure(self, req: dict) -> dict:
        """Ergebnis: ``action`` = messung_gestartet | wartet | abgelehnt | laeuft_bereits."""
        if self.edition == "release":
            raise ValueError(RELEASE_NO_MEASURE)
        cards = req.get("cards")
        if (not isinstance(cards, list) or not cards or not all(isinstance(c, int) and not isinstance(c, bool) for c in cards)
                or len(set(cards)) != len(cards)):
            raise ValueError("cards muss eine nichtleere Liste verschiedener NVML-Indizes sein, z. B. [0, 1, 2]")
        mib = req.get("mib")
        if mib is not None and not (isinstance(mib, (int, str)) and not isinstance(mib, bool)):
            raise ValueError("mib: Zahl oder Text wie \"8g\"")
        mod = self._module()
        known = self.gpuq_cards()
        if not known["reachable"]:
            return {"ok": False, "action": "abgelehnt", "error": known["error"]}
        have = {c["index"] for c in known["cards"]}
        bad = [c for c in cards if c not in have]
        if bad:
            raise ValueError("unbekannte Karte(n) %s; gpuq kennt %s" % (bad, sorted(have)))
        with self._lock:
            if self._job.get("state") == "running":
                return {"ok": False, "action": "laeuft_bereits", "job": self._job_view(), "window": None}
            b = self._booking
        # ein altes Fenster für andere Karten geht zurück, bevor ein neues gebucht wird
        if b and sorted(b["cards"]) != sorted(cards):
            with self._lock:
                self._release("andere_karten")
            b = None
        if b is None:
            body = {"owner": OWNER, "purpose": PURPOSE, "cards": sorted(cards), "duration": WINDOW}
            if mib is not None:
                body["mib"] = mib
            st, js = self._gq("POST", "/api/v1/bookings", body)
            if st != 200 or not isinstance(js, dict) or not js.get("id"):
                detail = (js or {}).get("detail") if isinstance(js, dict) else None
                return {"ok": False, "action": "abgelehnt", "error": "gpuq-Buchung abgelehnt (HTTP %s): %s" % (st, detail)}
            b = {"id": js["id"], "token": js.get("token"), "cards": sorted(cards), "booked_at": self.clock()}
            with self._lock:
                self._booking = b
                self._persist()
            live = js
        else:
            live = self._fetch_booking(b)
            if live is None:
                with self._lock:
                    self._booking = None
                    self._persist()
                return {"ok": False, "action": "abgelehnt", "error": "die Buchung %s ist beim Fensterplan nicht mehr bekannt; bitte erneut drücken" % b["id"]}
        state = live.get("state")
        window = _window_view(live)
        if state == "pending":
            return {"ok": True, "action": "wartet", "window": window,
                    "message": "Fenster wartet (%s). Nichts gemessen. Erneut drücken, sobald es läuft." % (live.get("note") or "in der Warteschlange")}
        if state != "running":
            with self._lock:
                self._booking = None
                self._persist()
            return {"ok": False, "action": "abgelehnt", "window": window,
                    "error": "Fenster %s ist %s: %s" % (b["id"], state, live.get("note") or "")}
        left = live.get("seconds_left")
        if left is not None and left < MIN_LEFT_S:
            with self._lock:
                self._release("zu_kurz")
            return {"ok": False, "action": "abgelehnt", "window": window,
                    "error": "Fenster hat nur noch %.0f s; Fenster zurückgegeben, bitte neu buchen" % left}
        # Gegenprobe (nvidia-smi über gpuq): der Dienst sperrt nichts
        used = {c["index"]: c.get("used_mib") for c in self.gpuq_cards()["cards"]}
        busy = {i: used.get(i) for i in cards if (used.get(i) or 0) >= BUSY_MIB}
        if busy:
            with self._lock:
                self._release("karte_belegt")
            return {"ok": False, "action": "abgelehnt", "window": window,
                    "error": "Karte(n) belegt trotz Fenster (used_mib %s): Fenster zurückgegeben, nichts gemessen" % busy}
        timeout = CHILD_CAP_S if left is None else max(30.0, min(CHILD_CAP_S, left - END_MARGIN_S))
        with self._lock:
            b["running_since"] = b.get("running_since") or self.clock()
            self._job = {"state": "running", "cards": sorted(cards), "started": self.clock(), "timeout_s": timeout}
        if self.synchronous:
            self._run(mod, sorted(cards), timeout)
        else:
            threading.Thread(target=self._run, args=(mod, sorted(cards), timeout), daemon=True).start()
        return {"ok": True, "action": "messung_gestartet", "window": window, "job": self._job_view()}

    def _run(self, mod, cards: List[int], timeout: float) -> None:
        job: Dict[str, Any] = {"cards": cards, "started": self._job.get("started")}
        try:
            res = mod.run_measurement(cards, python=self.python, prefix=self.prefix, timeout_s=timeout,
                                      pythonpath=self.measure_tree, runner=self.runner)
            nvml = mod.read_nvml()[0]
            job.update(state="ok" if res.get("ok") else "error", rc=res.get("rc"), seconds=res.get("seconds"),
                       line=mod.duration_line(res, nvml), warnings=res.get("warnings") or [],
                       error=None if res.get("ok") else (res.get("stderr_tail") or "Messlauf ohne Ergebnis"))
        except Exception as e:  # noqa: BLE001 -- jeder Fehlschlag muss das Fenster freigeben und gemeldet werden
            job.update(state="error", error="%s: %s" % (type(e).__name__, e))
        finally:
            with self._lock:
                self._release("fertig")
                self._view_cache = None
                job["finished"] = self.clock()
                self._job = job

    # ------------------------------------------------------------------ Zurückgeben
    def cancel(self) -> dict:
        if self.edition == "release":
            return {"ok": False, "error": RELEASE_NO_MEASURE}
        with self._lock:
            if self._job.get("state") == "running":
                return {"ok": False, "error": "Messung läuft; das Fenster geht nach dem Lauf von selbst zurück"}
            had = self._booking is not None
            self._release("abgebrochen")
        return {"ok": True, "released": had}
