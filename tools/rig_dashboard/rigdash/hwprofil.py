"""Hardwareprofil messen und anzeigen (Auftrag 950, Profil-Editor S2): Routen und JSON, keine Oberfläche.

Zwei Routen im Rig-Dashboard (nur Edition rig, nur im LAN):

  GET  /api/hwprofil          das aktuelle Hardwareprofil ``flliper.hardware/1`` (reine Sicht auf die vorhandenen
                              Messquellen, startet nichts), dazu Fensterstatus, letzte Messung, gpuq-Karten
  POST /api/hwprofil/measure  {"cards": [<NVML-Index>, ...]}  bucht ein gpuq-Fenster und misst darin
  POST /api/hwprofil/cancel   gibt ein noch wartendes Fenster zurück
  POST /api/hwprofil/recapture  "Neu erfassen" (AP-A): liest NVML neu und ersetzt das gespeicherte Profil; kein GPU-Fenster, auch in release
  GET  /api/hwprofil/issue    Issue-Text "Hardwareprofil": ein Markdown-Block zum Kopieren (Secrets und Hostpfade redigiert)

Gespeichert (AP-A, Plan Profil-Planer 06.10.): beim ersten Aufruf auf einer Maschine schreibt ``hardware_profile.capture`` das Profil nach
``FLLIPER_HARDWARE_PROFILE`` (Voreinstellung ``/var/lib/flliper/hardware.json``; Rig und Release gleich).  Die Antwort von GET /api/hwprofil
trägt dazu ``persist`` (Zustand, Alter, Abweichung zu den lebenden Karten).  Ohne Pfad (``persist_path`` leer, Tests) wird nichts geschrieben.

Die Messung läuft NUR in einem gebuchten gpuq-Fenster (Rig-Regel): die Route bucht selbst über die gpuq-HTTP-API,
Eigentümer ``profil-editor``, nur die zu messenden Karten, 10 min, ohne not_before.

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
import inspect
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import kartenplan_catalog as CAT
from . import redact

OWNER = "profil-editor"
WINDOW = "10m"
PURPOSE = "Hardwareprofil messen (Profil-Editor S2)"
#: gpuq: ab so viel belegtem Speicher gilt eine Karte als belegt (CLAUDE.md-Regel: ~500 MiB).
BUSY_MIB = 500
#: Obergrenze für den Kindprozess; kürzer, wenn das Fenster früher endet.
CHILD_CAP_S = 540.0
#: Weniger Restzeit als das: nicht mehr starten, Fenster zurückgeben.
MIN_LEFT_S = 90.0
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

#: Wo das Profil gespeichert wird (Env überstimmt; das Modul im Baum trägt dieselben Namen, hier gelten sie auch ohne Modul)
PERSIST_ENV = "FLLIPER_HARDWARE_PROFILE"
DEFAULT_PERSIST = "/var/lib/flliper/hardware.json"

#: Zustand der Speicherung -> Klartext für die Seite
PERSIST_LABELS = {
    "erst_erfasst": "beim ersten Start gespeichert",
    "neu_erfasst": "neu erfasst und gespeichert",
    "vorhanden": "gespeichert, stimmt mit den Karten überein",
    "abweichend": "gespeichert, WEICHT von den Karten ab (Datei unverändert; Neu erfassen ersetzt sie)",
    "nur_gespeichert": "NVML meldet nichts: es gilt das gespeicherte Profil",
    "keine_karten": "keine Karte gemeldet, nichts gespeichert",
    "nicht_schreibbar": "Speichern fehlgeschlagen",
}

MODULE_REL = os.path.join("sglang", "srt", "rigmon", "hardware_profile.py")
#: Planer-Bäume, in denen ``hardware_profile.py`` gesucht wird (KARTENPLAN_TREE wie beim Kartenplaner)
TREE_CANDIDATES = (
    "/opt/rigdash/kartenplan/current/python",
)


def default_persist_path(env: Optional[dict] = None) -> str:
    """Der Speicherort des Hardwareprofils für den Dienst: ``$FLLIPER_HARDWARE_PROFILE`` oder ``/var/lib/flliper/hardware.json``."""
    return (os.environ if env is None else env).get(PERSIST_ENV) or DEFAULT_PERSIST


def datasheet_provider(mod) -> Callable[[dict], dict]:
    """Datenblatt-Suche für ``hardware_profile.build``: SM-Zahl aus ``weg2/hw_sim.py`` (über das Profilmodul), Nennbandbreite und
    Katalogkarte aus ``kartenplan_catalog``.  Beide gelten als "Datenblatt"; ein Gemessenes steht im Profil davor."""
    sm = getattr(mod, "hw_sim_datasheet", None)

    def provider(row: dict) -> dict:
        out: Dict[str, Any] = {}
        if sm is not None:
            out.update(sm(row) or {})
        out.update(CAT.datasheet_of(row))
        return out

    return provider


# ---------------------------------------------------------------------------------------------------------- Issue-Text
_SRC_SHORT = {"gemessen": "gem.", "NVML": "NVML", "Datenblatt": "Datenbl.", "geschätzt": "gesch."}


def _md(x) -> str:
    """Ein Wert in einer Markdown-Tabellenzelle: kein Zeilenumbruch, kein Trennstrich."""
    return str("" if x is None else x).replace("|", "/").replace("\n", " ").replace("\r", " ").strip()


def _cell(n, digits: int = 1) -> str:
    """Wertknoten ``{v, src, unit}`` -> ``62.7 TFLOPS (gem.)``; ohne Wert ``nicht gemessen``."""
    if not isinstance(n, dict) or n.get("v") is None:
        return "nicht gemessen"
    v = n["v"]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        txt = str(v)
    else:
        txt = "%d" % round(v) if digits == 0 else "%.*f" % (digits, v)
    unit = (" " + n["unit"]) if n.get("unit") else ""
    return "%s%s (%s)" % (txt, unit, _SRC_SHORT.get(n.get("src"), n.get("src") or "?"))


def _utc(ts) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts)) if isinstance(ts, (int, float)) else "unbekannt"


def _tree_rev(tree: Optional[str]) -> Optional[str]:
    """Die Revision eines gestagten Baums (``.../releases/<sha>/python``), sonst ``None`` (nie aus dem Pfad geraten)."""
    tail = (tree or "").split("/releases/", 1)
    seg = tail[1].split("/", 1)[0] if len(tail) == 2 else ""
    return seg if 7 <= len(seg) <= 40 and all(ch in "0123456789abcdef" for ch in seg) else None


def _short_name(name) -> str:
    """Kartenname ohne Herstellerzusatz ("NVIDIA GeForce RTX 5090" -> "RTX 5090")."""
    n = str(name or "")
    for pre in ("NVIDIA GeForce ", "NVIDIA "):
        if n.startswith(pre):
            return n[len(pre):]
    return n


def version_facts(doc: dict, versions: Optional[dict] = None) -> dict:
    """Die Versionsangaben, die ein Issue-Text nennt (Hardwareprofil UND Laufbericht lesen dieselben): Image (``SGLANG_IMAGE_TAG``),
    Baum-Revision (nur aus einem gestagten Baumpfad ``.../releases/<sha>/python`` oder einer übergebenen ``tree_rev``), Treiber, CUDA/torch des
    Messprozesses und Dashboard-Version.  Was nicht belegt ist, ist ``None`` (die Texte schreiben dann "unbelegt"); nie geraten."""
    versions = versions or {}
    return {"image": os.environ.get("SGLANG_IMAGE_TAG") or versions.get("image") or None,
            "tree_rev": _tree_rev(versions.get("tree")) or versions.get("tree_rev") or None,
            "driver": doc.get("driver") or None, "cuda": doc.get("cuda") or None, "torch": doc.get("torch") or None,
            "rigdash": versions.get("rigdash") or None}


def issue_short(doc: dict, *, persist: Optional[dict] = None, level: int = 3) -> str:
    """Hardwareprofil in KURZFORM als Markdown-Block für den Laufbericht: eine Zeile je Karte (Name, cc, SM-Zahl, VRAM, PCIe, Speicherbandbreite
    mit Herkunft).  Dieselben Zellenbausteine wie ``issue_text``; die Langform (Takt, Messraten, Katalogherkunft) steht dort.  Redigiert."""
    cards = doc.get("cards") or []
    h = "#" * level
    cap = (persist or {}).get("captured_at")
    L: List[str] = ["%s Hardwareprofil (Kurzform)" % h, ""]
    pid = str(doc.get("id") or "")
    L.append("Profil `%s`, %s; %d Karte(n). Die Langform (Takt, Messraten, Katalogherkunft) steht im Issue-Text des Abschnitts Hardware."
             % (_md(pid[:19] if pid else "unbelegt"), _utc(cap) + " gespeichert" if cap else "lebende Sicht, nicht gespeichert", len(cards)))
    L.append("")
    L.append("| Ord | NVML | Name | cc | SM-Zahl | VRAM | PCIe max. | Speicherbandbreite |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in cards:
        pc = c.get("pcie") or {}
        gen, wd = (pc.get("max_gen") or {}).get("v"), (pc.get("max_width") or {}).get("v")
        g = c.get("mem_gbs") or {}
        bw = g.get("read") if (g.get("read") or {}).get("v") is not None else (g.get("nominal") if (g.get("nominal") or {}).get("v") is not None else g.get("nameplate"))
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % tuple(_md(x) for x in (
            c.get("ord"), c.get("nvml_index"), _short_name(c.get("name")), ".".join(str(x) for x in (c.get("cc") or [])) or "?",
            _cell(c.get("sm_count"), 0), _cell(c.get("vram_total_mib"), 0),
            ("Gen%s x%s" % (gen, wd)) if gen is not None and wd is not None else "nicht gemessen", _cell(bw, 0))))
    if not cards:
        L.append("| | | keine Karte gemeldet | | | | | |")
    if doc.get("measure_needed"):
        L.append("")
        L.append("Das Profil ist unvollständig: es fehlen Messwerte.")
    return redact.text_for_issue("\n".join(L)) + "\n"


def issue_text(doc: dict, *, persist: Optional[dict] = None, versions: Optional[dict] = None, now: Optional[float] = None) -> str:
    """Der Issue-Text "Hardwareprofil" (GitHub-Markdown): NVML-Identität, Größen, cc, SM, Takt, Messraten (soweit vorhanden),
    Treiber/Image/Baum.  Geheimnisse und Hostpfade sind entfernt (``redact.text_for_issue``).  Ein Wert ohne Messung steht als
    "nicht gemessen", ein Wert ohne bekannte Version als "unbelegt"."""
    versions = versions or {}
    cards = doc.get("cards") or []
    now = time.time() if now is None else now
    vf = version_facts(doc, versions)
    image, rev = vf["image"], vf["tree_rev"]
    L: List[str] = []
    L.append("## Hardwareprofil (`%s`)" % _md(doc.get("schema") or "flliper.hardware/1"))
    L.append("")
    L.append("| Angabe | Wert |")
    L.append("|---|---|")
    pid = str(doc.get("id") or "")
    L.append("| Profil-ID | `%s` |" % _md(pid[:19] if pid else "unbelegt"))
    cap = ((persist or {}).get("captured_at"))
    L.append("| Erfasst | %s |" % (_utc(cap) + " (gespeichert)" if cap else _utc(doc.get("created")) + " (lebende Sicht, nicht gespeichert)"))
    L.append("| Treiber | %s |" % _md(doc.get("driver") or "unbelegt"))
    L.append("| CUDA / torch (Messprozess) | %s / %s |" % (_md(doc.get("cuda") or "unbelegt"), _md(doc.get("torch") or "unbelegt")))
    L.append("| Image | %s |" % _md(image or "unbelegt (SGLANG_IMAGE_TAG nicht gesetzt)"))
    L.append("| Baum | %s |" % _md(rev or "unbelegt"))
    L.append("| Dashboard | %s |" % _md(versions.get("rigdash") or "unbelegt"))
    L.append("| Karten | %d |" % len(cards))
    L.append("")
    L.append("### Karten (NVML-Identität)")
    L.append("")
    L.append("| Ord | NVML | Name | Klasse | cc | SM-Zahl | VRAM | BAR1 | PCIe max. | SM-Takt max. | Speichertakt max. | Busbreite | Leistungsgrenze | PCI-Bus | UUID |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cards:
        pc = c.get("pcie") or {}
        gen, wd = (pc.get("max_gen") or {}).get("v"), (pc.get("max_width") or {}).get("v")
        ck = c.get("clocks") or {}
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % tuple(_md(x) for x in (
            c.get("ord"), c.get("nvml_index"), _short_name(c.get("name")), c.get("class_key"),
            ".".join(str(x) for x in (c.get("cc") or [])) or "?", _cell(c.get("sm_count"), 0), _cell(c.get("vram_total_mib"), 0),
            _cell(c.get("bar1_total_mib"), 0), ("Gen%s x%s" % (gen, wd)) if gen is not None and wd is not None else "nicht gemessen",
            _cell(ck.get("sm_max_mhz"), 0), _cell(ck.get("mem_max_mhz"), 0), _cell(c.get("mem_bus_bits"), 0),
            _cell((c.get("power") or {}).get("limit_w"), 0), c.get("pci_bus_id") or "unbelegt", c.get("uuid") or "unbelegt")))
    L.append("")
    L.append("### Speicher und Rechenleistung")
    L.append("")
    L.append("| Wert | " + " | ".join("Karte %s (NVML %s)" % (_md(c.get("ord")), _md(c.get("nvml_index"))) for c in cards) + " |")
    L.append("|---|" + "---|" * len(cards))

    def row(label, f):
        L.append("| %s | %s |" % (_md(label), " | ".join(_md(f(c)) for c in cards)))

    row("Speicher lesen", lambda c: _cell((c.get("mem_gbs") or {}).get("read"), 0))
    row("Speicher kopieren (D2D)", lambda c: _cell((c.get("mem_gbs") or {}).get("copy"), 0))
    row("Speicher GEMV (Decode)", lambda c: _cell((c.get("mem_gbs") or {}).get("gemv"), 0))
    row("Spitze aus NVML (Busbreite x Takt)", lambda c: _cell((c.get("mem_gbs") or {}).get("nameplate"), 0))
    row("Nennbandbreite (Katalog)", lambda c: _cell((c.get("mem_gbs") or {}).get("nominal"), 0))
    for f in doc.get("formats") or []:
        row(f.get("label") or f.get("key"), lambda c, k=f.get("key"): _cell((c.get("compute") or {}).get(k), 1))
    row("Host -> Karte (H2D)", lambda c: _cell((c.get("h2d") or {}).get("gbs"), 1))
    row("Karte -> Host (D2H)", lambda c: _cell((c.get("d2h") or {}).get("gbs"), 1))
    cat = [c for c in cards if c.get("catalog")]
    if cat:
        L.append("")
        L.append("### Katalogkarte und Herkunft der Datenblattwerte")
        L.append("")
        for c in cards:
            k = c.get("catalog")
            if not k:
                L.append("- Karte %s: kein Katalogeintrag" % _md(c.get("ord")))
                continue
            of = k.get("origin_fields") or {}
            extra = "; Nennbandbreite: %s" % _md(of.get("mem_bw")) if of.get("mem_bw") and of.get("mem_bw") != k.get("origin") else ""
            L.append("- Karte %s: %s, Herkunft: %s%s" % (_md(c.get("ord")), _md(k.get("label")), _md(k.get("origin")), extra))
    meas = [lk for lk in (doc.get("links") or []) if (lk.get("gbs") or {}).get("src") == "gemessen" and lk.get("transport") != "bar1"]
    if meas:
        L.append("")
        L.append("### Karte-zu-Karte (gemessen)")
        L.append("")
        L.append("| Von | Nach | Weg | Bandbreite |")
        L.append("|---|---|---|---|")
        for lk in meas:
            L.append("| %s | %s | %s | %s |" % (_md(lk.get("src")), _md(lk.get("dst")), _md(lk.get("transport_label") or lk.get("transport")), _md(_cell(lk.get("gbs"), 2))))
    L.append("")
    L.append("### Herkunft der Werte")
    L.append("")
    L.append("gem. = Messlauf auf dieser Karte, NVML = vom Treiber gelesen, Datenbl. = Herstellerangabe aus dem Katalog (nicht gemessen), "
             "gesch. = abgeleitet, nicht gemessen = keine Messung vorhanden.")
    if doc.get("measure_needed"):
        L.append("")
        L.append("Das Profil ist unvollständig: es fehlen Messwerte (Hardwareprofil messen).")
    notes = ((doc.get("sources") or {}).get("nvml") or {}).get("issues") or []
    if notes:
        L.append("")
        for n in notes:      # je Hinweis eine Zeile: ein Geheimnis in einem Hinweis nimmt nur diese Zeile mit
            L.append("- NVML-Hinweis: " + _md(n))
    return redact.text_for_issue("\n".join(L)) + "\n"


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
                 clock: Callable[[], float] = time.time, synchronous: bool = False, edition: str = "rig",
                 persist_path: Optional[str] = None, versions: Optional[Callable[[], dict]] = None):
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
        #: Speicherort des Profils; ohne Angabe gilt nur die Env-Variable, sonst wird nichts geschrieben (Tests)
        self.persist_path = persist_path if persist_path is not None else os.environ.get(PERSIST_ENV) or None
        self.versions = versions
        self._mod = None
        self._lock = threading.Lock()
        self._booking: Optional[dict] = None     # {id, token, cards, booked_at, running_since}
        self._job: dict = {"state": "idle"}
        self._view_cache: Optional[Tuple[float, dict, Optional[dict]]] = None
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
                prof, persist = hit[1], hit[2]
            else:
                prof, persist = self._assemble(mod)
                self._view_cache = (now, prof, persist)
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
                "job": self._job_view(), "gpuq": self.gpuq_cards(), "owner": OWNER, "window_len": WINDOW,
                "persist": persist or {"enabled": False}}

    # ------------------------------------------------------------------ Speichern (AP-A)
    def _build(self, mod) -> dict:
        """Das lebende Profil; die Datenblatt-Suche nur an ein Modul, dessen ``build`` sie kennt (ein älterer Baum nicht)."""
        kw: Dict[str, Any] = {"cache_dir": self.cache_dir}
        try:
            if "datasheet" in inspect.signature(mod.build).parameters:
                kw["datasheet"] = datasheet_provider(mod)
        except (TypeError, ValueError):
            pass
        return mod.build(**kw)

    def _can_persist(self, mod) -> bool:
        return bool(self.persist_path) and hasattr(mod, "capture")

    def _persist_view(self, res: dict) -> dict:
        """Was die Seite vom Speicherzustand sieht (kein Pfad: der Ort ist Sache des Betreibers, die Seite zeigt nur den Zustand)."""
        p = res.get("persisted") or {}
        st = res.get("state")
        return {"enabled": True, "state": st, "label": PERSIST_LABELS.get(st, st),
                "captured_at": (p.get("capture") or {}).get("at"), "reason": (p.get("capture") or {}).get("reason"),
                "id": p.get("id"), "drift": res.get("drift"), "error": res.get("error"),
                "from_persisted": st == "nur_gespeichert",
                "file": os.path.basename(self.persist_path or "")}

    def _assemble(self, mod) -> Tuple[dict, Optional[dict]]:
        """Profil + Speicherzustand.  Ohne Pfad oder ohne ``capture`` im Modul: das lebende Profil, nichts geschrieben."""
        live = self._build(mod)
        if not self._can_persist(mod):
            return live, None
        res = mod.capture(self.persist_path, live=live, now=self.clock())
        return res["show"], self._persist_view(res)

    def recapture(self) -> dict:
        """"Neu erfassen": NVML neu lesen und das gespeicherte Profil ersetzen.  Kein GPU-Fenster, in beiden Ausgaben."""
        try:
            mod = self._module()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}
        if not self._can_persist(mod):
            return {"ok": False, "error": "Speichern ist nicht eingerichtet (kein Speicherort: Env %s) oder das Profilmodul des Baums kann es nicht" % PERSIST_ENV}
        with self._lock:
            live = self._build(mod)
            res = mod.capture(self.persist_path, live=live, force=True, now=self.clock())
            self._view_cache = None
        ok = res["state"] == "neu_erfasst"
        return {"ok": ok, "persist": self._persist_view(res), "error": None if ok else (res.get("error") or res["state"])}

    # ------------------------------------------------------------------ Issue-Text
    def issue(self) -> dict:
        """Der Issue-Text "Hardwareprofil" (Markdown, redigiert) aus dem Profil, das ``get`` zeigt."""
        got = self.get()
        if not got.get("ok"):
            return {"ok": False, "error": got.get("error") or "kein Profil"}
        v = dict(self.versions() or {}) if self.versions else {}
        v.setdefault("tree", self.tree)
        return {"ok": True, "format": "markdown", "text": issue_text(got["profile"], persist=got.get("persist"), versions=v, now=self.clock())}

    def issue_parts(self) -> dict:
        """Bausteine für den Laufbericht des Profil-Editors: der Hardwareprofil-Block in Kurzform und die Versionsangaben (``version_facts``)."""
        got = self.get()
        if not got.get("ok"):
            return {"ok": False, "error": got.get("error") or "kein Profil"}
        v = dict(self.versions() or {}) if self.versions else {}
        v.setdefault("tree", os.path.realpath(self.tree) if self.tree else None)     # ``current`` -> ``releases/<sha>``: so der Baum seine Revision nennt
        return {"ok": True, "short": issue_short(got["profile"], persist=got.get("persist")), "versions": version_facts(got["profile"], v),
                "profile_id": got["profile"].get("id")}

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
            if job.get("state") == "ok" and self.persist_path and hasattr(mod, "capture"):
                try:   # eine ausdrückliche Messung ist eine Neuerfassung: das gespeicherte Profil trägt die neuen Raten
                    self.recapture()
                except Exception:  # noqa: BLE001 -- das Speichern darf das Ergebnis der Messung nie verdecken
                    pass

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
