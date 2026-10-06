"""Profil-Editor (Auftrag 930, S1): ein Serverprofil laden, bearbeiten, prüfen, speichern, als .env exportieren.

Das Dashboard ERSTELLT nur ein Profil (Nutzer-Entscheid 03.10. ~20:15Z): vorfüllen, bearbeiten, prüfen, als Datei
im State-Volume speichern.  Es startet nichts, hat keinen Force-Schalter und liefert keinen Startauftrag.  Den Start
macht der Nutzer am Server: ein Profil angeben (``FLLIPER_PROFILE=<name>``) und, wenn der Planer Werte ablehnt,
EIN Schalter ``FLLIPER_FORCE=1`` bzw. ``--force``.  Das Dashboard zeigt nur, welche Ablehnungen der Planer hätte
und welche davon der Force-Schalter übergeht.

Die Rechnung liegt im Planer-Baum, stdlib-rein und per Dateipfad geladen (wie ``kartenplan_gate``):

* ``weg2/profile_json.py``  Profil als JSON (``flliper.server/1``), ``.env`` <-> JSON, Zeilen, Herkunft, Bearbeitung;
* ``weg2/refusals.py``      das Ablehnungsregister (Wert-Ablehnung forcebar / nicht forcebar, je mit Begründung);
* ``weg2/card_identity.py`` und ``topology.py`` für das Trockenlauf-Gate (über ``kartenplan``).

Erklärungen und Abhängigkeiten kommen aus ``profil_data/catalog.json`` (erzeugt von
``python -m sglang.srt.weg2.profile_catalog``): kuratiert, aus argparse/environ geerntet, sonst "unerklärt".
Release-Profile (``.env``) werden mit bash ausgewertet (vertrauenswürdiges Verzeichnis, nur Lesen); Nutzerprofile
sind JSON-Dateien im State-Volume.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional

from . import kartenplan_catalog as CAT
from . import redact
from .kartenplan import MAX_CARDS
from . import kvheads as KVH
from . import kartenplan_transport as TR

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "profil_data")
CATALOG_FILE = os.path.join(DATA_DIR, "catalog.json")
DEFAULT_RELEASE_DIR = "/spinning/gpu-arb/docker/profiles_release"
#: ein gemeinsamer Ort fuer Dashboard und Entrypoint (Koordinator 03.10.): Env FLLIPER_PROFILES_DIR, Standard /var/lib/flliper/profiles
DEFAULT_USER_DIR = os.environ.get("FLLIPER_PROFILES_DIR") or "/var/lib/flliper/profiles"


class _NameCheck:
    """Profile names: ``[a-z0-9][a-z0-9._-]{0,63}`` (no regex: no path separator, no leading dot)."""

    ALPHA = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")
    REST = ALPHA | frozenset("._-")

    @classmethod
    def match(cls, name):
        n = str(name or "")
        return bool(0 < len(n) <= 64 and n[0] in cls.ALPHA and all(c in cls.REST for c in n))


NAME_RE = _NameCheck
#: Planer-Bäume, aus denen profile_json/refusals gelesen werden (``KARTENPLAN_TREE`` überschreibt); zuletzt der Baum dieses Checkouts
TREE_CANDIDATES = (
    "/opt/rigdash/kartenplan/current/python",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "python")),
)
NEEDED = ("profile_json.py", "refusals.py")
MAX_BODY = 1 << 20


class ProfilError(ValueError):
    pass


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # @dataclass löst Typen über sys.modules[__module__] auf
    spec.loader.exec_module(mod)
    return mod


def find_tree(explicit: Optional[str] = None) -> Optional[str]:
    for t in (explicit, os.environ.get("KARTENPLAN_TREE"), *TREE_CANDIDATES):
        if not t:
            continue
        base = os.path.join(t, "sglang", "srt", "weg2")
        if all(os.path.isfile(os.path.join(base, n)) for n in NEEDED):
            return t
    return None


FORCE_ENV = "FLLIPER_FORCE=1"
MAX_CODE_TEXT = 400


def _clip_text(code: str, text) -> str:
    """Der Ablehnungstext für die Force-Liste: ein führendes ``<CODE>:`` entfällt (der Code steht schon davor), ein zu langer Text endet an
    einer Wortgrenze mit ``…`` statt mitten im Wort (nie länger als ``MAX_CODE_TEXT``)."""
    t = str(text or "").strip()
    if t.startswith(code + ":"):
        t = t[len(code) + 1:].strip()
    if len(t) <= MAX_CODE_TEXT:
        return t
    cut = t[:MAX_CODE_TEXT - 1]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > MAX_CODE_TEXT // 2 else cut).rstrip(" ,;:") + "…"


def force_verdict(r: dict):
    """(Text, ``force_state``, ``force_via``) für eine Zeile des Ablehnungsregisters ``r`` (leer = Code unbekannt).

    ``force_state``: ``force`` (der Serverstart übergeht es) | ``blockiert`` (auch mit Force nicht) | ``ungeprueft`` (der Launcher dieser Linie
    prüft es nicht).  ``force_via``: ``launcher`` | ``entrypoint`` | ``None``.  Der Entrypoint (docker/entrypoint.sh) übergeht PROFIL-STATUS, SHM,
    STORE und MEMAVAIL mit FLLIPER_FORCE=1, der Launcher nicht (Auftrag 2002 A); ein älteres Register ohne ``wired_entrypoint`` wird über
    ``enforced_by`` gelesen."""
    if r and not r.get("forcebar"):
        return "nein, nicht forcebar", "blockiert", None
    if r.get("wired"):
        return "ja, Force übergeht es (FORCED-PAST im Boot-Log)", "force", "launcher"
    ep = r.get("wired_entrypoint")
    if ep is None:
        ep = r.get("enforced_by") == "entrypoint"
    if ep:
        return ("ja, im Docker-Start (Entrypoint) forcebar, im reinen Launcher-Aufruf nicht (FORCED-PAST im Boot-Log)", "force", "entrypoint")
    if r.get("enforced_by") == "planner-gate":
        return "der Launcher dieser Linie prüft das noch nicht: beim Start keine Verweigerung", "ungeprueft", None
    return "forcebar, aber weder im Launcher noch im Entrypoint dieser Linie verdrahtet: der Start verweigert weiter", "blockiert", None


def force_hint(dry, register: List[dict], line: str = "") -> dict:
    """Der Force-Teil des Export-Hinweises aus dem LETZTEN Trockenlauf (``dry`` = dessen Antwort oder ``None``).  Nur Text; startet nichts.

    Fälle (``fall``): ``kein_trockenlauf`` | ``keine_ablehnung`` | ``nur_forcebar`` | ``gemischt`` | ``nur_nicht_forcebar`` | ``ungeprueft``.
    Die Zeile ``-e FLLIPER_FORCE=1`` (``show_line``) gibt es genau dann, wenn mindestens eine Ablehnung forcebar ist.  Gerechnet wird NEU aus den
    Codes gegen das Register: was der Browser als ``force_state`` mitschickt, wird nicht geglaubt; nur Code und (gekürzter) Text werden gelesen."""
    reg = {r.get("code"): r for r in register or []}
    rejs = []
    if isinstance(dry, dict) and isinstance(dry.get("rejections"), list):
        for q in dry["rejections"][:64]:
            if isinstance(q, dict) and isinstance(q.get("code"), str) and 0 < len(q["code"]) <= 40:
                rejs.append((q["code"], _clip_text(q["code"], q.get("text"))))
    have_dry = isinstance(dry, dict) and isinstance(dry.get("rejections"), list)
    force, blocked, openl, seen = [], [], [], set()
    for code, text in rejs:
        if code in seen:
            continue
        seen.add(code)
        r = reg.get(code) or {}
        label, state, via = force_verdict(r) if r else ("unbekannter Code: nicht als forcebar behandelt", "blockiert", None)
        row = {"code": code, "text": text or r.get("title") or code, "via": via, "scope": r.get("force_scope") or label}
        {"force": force, "blockiert": blocked, "ungeprueft": openl}[state].append(row)
    if not have_dry:
        fall, text = "kein_trockenlauf", ("Noch kein Trockenlauf für dieses Profil: ob der Serverstart Force braucht, zeigt der Trockenlauf "
                                          "(Karten wählen, prüfen lassen), danach erneut exportieren.")
    elif not rejs:
        fall, text = "keine_ablehnung", "Der Planer lehnt nichts ab; Force wird nicht gebraucht."
    elif force and blocked:
        fall, text = "gemischt", "Force übergeht einen Teil der Ablehnungen, aber nicht alle: so startet der Server nicht."
    elif force:
        fall, text = "nur_forcebar", "Mit FLLIPER_FORCE=1 übergeht der Server diese Ablehnungen und listet jede im Boot-Log als FORCED-PAST."
    elif blocked:
        fall, text = "nur_nicht_forcebar", "Force hilft hier nicht: diese Ablehnungen bleiben auch mit Force bestehen."
    else:
        fall, text = "ungeprueft", "Der Launcher dieser Linie prüft das noch nicht: kein Force nötig."
    out = {"fall": fall, "text": text, "show_line": bool(force), "force_env": FORCE_ENV if force else None,
           "force_codes": force, "blocked_codes": blocked, "open_codes": openl,
           "records_note": ("Ein Force-Boot schreibt keine Records; seine Messwerte zählen nicht als abgenommen." if force else ""),
           "line_note": ""}
    if force and str(line) == "nf":
        out["line_note"] = ("Linie nf: der Launcher übergeht dort nur HW-COUNT, HW-UNCALIBRATED und HOST-MEM; "
                            "die Entrypoint-Codes (PROFIL-STATUS, SHM, STORE, MEMAVAIL) gelten auch dort.")
    return out


def docker_run_example(name: str, force: dict) -> List[str]:
    """Beispiel-``docker run`` als Zeilen (Platzhalter in <>, zum Anpassen); die Force-Zeile nur, wenn ``force["show_line"]``.
    Ein Kommentar steht VOR dem Befehl, nie hinter einem ``\\`` (hinter dem Zeilenumbruch-Backslash bricht Bash die Fortsetzung)."""
    lines = []
    if force.get("show_line"):
        lines.append("# FLLIPER_FORCE=1 nur nötig, weil der Planer ablehnt: %s" % ", ".join(c["code"] for c in force["force_codes"]))
    lines += ["docker run -d --name htsglang-mine \\",
              "  <die Flags aus Abschnitt 3.3 der README: --gpus, --shm-size, --memory, -p, Modell-Mounts> \\",
              "  -v flliper-state:/var/lib/flliper \\",
              "  -e MODE=weg2 -e FLLIPER_PROFILE=%s \\" % name]
    if force.get("show_line"):
        lines.append("  -e %s \\" % FORCE_ENV)
    lines.append("  ghcr.io/efschu/htsglang:<tag> serve")
    return lines


# ---------------------------------------------------------------------------------------------------------- Issue-Text "Laufbericht" (AP-I)
#: Höchstzahl der Zeilen je Tabelle im Laufbericht (ein Issue ist kein Profil-Dump; der Rest steht als "und N weitere")
ISSUE_MAX_ROWS = 120
ISSUE_CELL = 200
#: Überschriften des Laufberichts in fester Reihenfolge (der Test prüft jede)
ISSUE_BLOCKS = ("Hardwareprofil (Kurzform)", "Modellprofil", "Betriebsform", "Vorschlag und Übersteuerungen", "Verdikte und Force",
                "Versionen", "Messergebnis / Boot-Log-Auszug")


def _md(x, limit: int = ISSUE_CELL) -> str:
    """Ein Wert in einer Markdown-Tabellenzelle: kein Zeilenumbruch, kein Trennstrich, auf ``limit`` Zeichen gekürzt."""
    t = str("" if x is None else x).replace("|", "/").replace("\n", " ").replace("\r", " ").strip()
    return t if len(t) <= limit else t[:limit - 1].rstrip() + "…"


def _lv(o):
    """Wertknoten des Modellprofils ``{v, src}`` -> (Wert, Quelle); alles andere -> (None, None)."""
    return (o.get("v"), o.get("src")) if isinstance(o, dict) and "v" in o else (None, None)


def _gib(n) -> str:
    return "%.2f GiB" % (float(n) / (1 << 30))


def issue_betriebsform(names, n_cards) -> dict:
    """Die Betriebsform AUS DEN FLAGS DES PROFILS gelesen (nicht vom Planer gewählt: ``propose()`` kommt später): ``--dual-layout`` /
    ``--dual-share`` = Dual PP/TP, ``--d-only`` = nur TP, ein Karte = Einzelkarte, sonst Flip PP/TP (die Standardform des Launchers).
    ``names`` = die Zeilennamen des Profils, ``n_cards`` = Kartenzahl im Trockenlauf (``None`` = unbekannt)."""
    names = set(names)
    dual = sorted(n for n in names if n in ("--dual-layout", "--dual-share"))
    if dual:
        return {"form": "Dual PP/TP", "why": "Flag %s im Profil: P und D gleichzeitig wach auf denselben Karten" % ", ".join(dual)}
    if "--d-only" in names:
        return {"form": "nur TP", "why": "Flag --d-only im Profil"}
    if n_cards == 1:
        return {"form": "Einzelkarte", "why": "eine Karte im Trockenlauf (der weg2-Launcher braucht mindestens zwei)"}
    return {"form": "Flip PP/TP", "why": "weder --d-only noch --dual-* im Profil, also die Standardform des Launchers"}


def _issue_cards(dry, cards) -> tuple:
    """(Kartenlabels, Quelle): die Karten des letzten Trockenlaufs, sonst die gewählten Karten (noch ohne Trockenlauf), sonst leer."""
    if isinstance(dry, dict) and isinstance(dry.get("cards"), list) and dry["cards"]:
        return [str(c.get("label") or "?") for c in dry["cards"] if isinstance(c, dict)][:16], "Trockenlauf"
    out = []
    for rc in (cards if isinstance(cards, list) else [])[:16]:
        e = CAT.card(rc.get("card")) if isinstance(rc, dict) else None
        out.append(CAT.label(e) if e else "unbekannte Karte")
    return out, "gewählt, noch kein Trockenlauf"


def _issue_model(model, doc_rows) -> List[str]:
    """Block Modellprofil: die Werte des Schätzprofils ``flliper.model/1`` mit ihrer Quelle; ohne Profil nur, was das Serverprofil nennt.
    Der Pfad des Modells steht nie da, nur der Ordnername."""
    L = ["### Modellprofil", ""]
    p = model.get("profile") if isinstance(model, dict) and isinstance(model.get("profile"), dict) and "arch" not in model else model
    if not isinstance(p, dict) or p.get("schema") != "flliper.model/1":
        name = next((os.path.basename(str(r["value"]).rstrip("/")) for r in doc_rows if r["name"] in ("PROFILE_MODEL", "--model") and r["value"]), "")
        L.append("Kein Modellprofil geschätzt (Abschnitt Modelle: Modellprofil schätzen, dann den Laufbericht neu erzeugen)."
                 + (" Das Serverprofil nennt das Modell `%s`." % _md(name) if name else ""))
        return L
    out: List[tuple] = []

    def add(label, o, fmt=None):
        v, src = _lv(o)
        if v is None or v == "" or v == [] or v == {}:
            return
        out.append((label, "%s (%s)" % (fmt(v) if fmt else v, src or "?")))

    a, w, kv, st, ex, dr, cx = (p.get(k) or {} for k in ("arch", "weights", "kv", "state", "experts", "draft", "context"))
    name = os.path.basename(str(p.get("path") or "").rstrip("/"))
    out.append(("Modell", "`%s`" % _md(name) if name else "unbelegt"))
    add("Format", p.get("format"))
    add("Art", a.get("family"), lambda v: "MoE" if v == "moe" else "dicht")
    add("Hybrid (GDN/Mamba)", a.get("hybrid"), lambda v: "ja" if v else "nein")
    add("Layer", a.get("n_layers"))
    add("Layertypen", a.get("layer_counts"), lambda v: ", ".join("%s %s" % (k, v[k]) for k in sorted(v) if v[k]))
    add("Hidden-Größe", a.get("hidden"))
    add("Köpfe Q / KV / Kopfgröße", {"v": "%s / %s / %s" % (_lv(a.get("heads_q"))[0], _lv(a.get("heads_kv"))[0], _lv(a.get("head_dim"))[0]),
                                      "src": _lv(a.get("heads_q"))[1]} if _lv(a.get("heads_q"))[0] is not None else None)
    add("Attention", a.get("attention"))
    add("Gewichte gesamt", w.get("total_bytes"), _gib)
    add("Experten (Anzahl)", ex.get("n"))
    add("Experten je Token (top_k)", ex.get("top_k"))
    add("KV je Token und Attention-Layer", kv.get("cell_bytes_per_attn_layer_token"), lambda v: "%s B" % v)
    add("Mamba/GDN-Zustand je Linear-Layer und Request", st.get("per_linear_layer_per_slot_mib"), lambda v: "%.4g MiB" % v)
    add("MTP-Schichten (Draft im Modell)", dr.get("mtp_layers"))
    if isinstance(dr.get("external"), dict):
        ext = dr["external"]
        add("Externer Draft", ext.get("total_bytes"), lambda v: "%s, %s" % (os.path.basename(str(ext.get("path") or "").rstrip("/")) or "?", _gib(v)))
    add("Kontext (max. Positionen)", cx.get("max_position_embeddings"))
    if p.get("config_sha"):
        out.append(("config-Prüfsumme", "`%s`" % _md(p["config_sha"])))
    L += ["| Angabe | Wert (Quelle) |", "|---|---|"] + ["| %s | %s |" % (_md(k), _md(v)) for k, v in out]
    L += ["", "Quelle: config = steht in der config.json, Index = aus den Tensorköpfen, geschätzt = gerechnet, stat = Dateigröße."]
    return L


def _issue_cell(row: dict, v) -> str:
    """Wert einer Zeile für die Tabelle: Schalter ohne Wert = ``an``; fehlt der Wert = ``–``; ein Geheimnis nach Namen = ``<entfernt>``."""
    if v is None:
        return "–"
    if v == "" and row.get("bare"):
        return "an"
    return _md(redact.value_for_issue(str(row.get("name") or ""), v)) or "(leer)"


def issue_diff_rows(view: dict) -> dict:
    """Die Zeilen des Profils, die vom Profil oder vom Planer-Vorschlag abweichen: ``{"rows": [...], "counts": {...}}``.  Eine Zeile zählt, wenn
    sie gegenüber dem geladenen Profil geändert ist (``changed``), ihre Herkunft ``nutzer`` oder ``planer`` ist oder der Planer-Vorschlag einen
    anderen Wert nennt.  Zusätzliche Felder (``state``, ``verdict``), die der Orakel-Weg (AP-D) einer Zeile mitgibt, bleiben erhalten."""
    rows = view.get("rows") or []
    sel = [r for r in rows if r.get("changed") or r.get("origin") in ("nutzer", "planer")
           or (r.get("planner_value") is not None and r.get("planner_value") != r.get("value"))]
    counts = {"rows": len(rows), "geaendert": sum(1 for r in rows if r.get("changed")), "nutzer": sum(1 for r in rows if r.get("origin") == "nutzer"),
              "mit_vorschlag": sum(1 for r in rows if r.get("planner_value") is not None),
              "weicht_vom_vorschlag_ab": sum(1 for r in rows if r.get("planner_value") is not None and r.get("planner_value") != r.get("value"))}
    return {"rows": sel, "counts": counts}


def _issue_proposal(view: dict) -> List[str]:
    L = ["### Vorschlag und Übersteuerungen", ""]
    d = issue_diff_rows(view)
    c, sel = d["counts"], d["rows"]
    removed, only = view.get("removed") or [], view.get("planner_only") or []
    L.append("Quelle: Herkunft, Profilwert und Planer-Vorschlag je Zeile stehen im Profil (meta.origins, meta.profile_values, meta.planner). "
             "%d Werte, davon %d gegenüber dem geladenen Profil geändert, %d als Nutzer gesetzt, %d mit Planer-Vorschlag, %d weichen vom Vorschlag ab."
             % (c["rows"], c["geaendert"], c["nutzer"], c["mit_vorschlag"], c["weicht_vom_vorschlag_ab"]))
    if not c["mit_vorschlag"] and not only:
        L.append("")
        L.append("Für dieses Profil liegt kein Planer-Vorschlag vor (meta.planner leer); die Spalte Vorschlag bleibt leer.")
    extra = [k for k in ("state", "verdict") if any(k in r for r in sel)]
    head = ["Wert", "Aktuell", "Profil", "Vorschlag (Planer)", "Herkunft"] + [{"state": "Zustand", "verdict": "Verdikt"}[k] for k in extra]
    if sel:
        L += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in sel[:ISSUE_MAX_ROWS]:
            cells = ["`%s`" % _md(r.get("key") or r.get("name")), _issue_cell(r, r.get("value")), _issue_cell(r, r.get("profile_value")),
                     _issue_cell(r, r.get("planner_value")), _md(r.get("origin_label") or r.get("origin"))]
            for k in extra:
                v = r.get(k)
                cells.append(_md((v.get("code") or v.get("text")) if isinstance(v, dict) else v) or "–")
            L.append("| " + " | ".join(cells) + " |")
        if len(sel) > ISSUE_MAX_ROWS:
            L.append("")
            L.append("… und %d weitere abweichende Werte (gekürzt)." % (len(sel) - ISSUE_MAX_ROWS))
    else:
        L += ["", "Keine Abweichung: alle Werte stehen wie im geladenen Profil und, wo es einen gibt, wie im Vorschlag."]
    if removed:
        L += ["", "Gegenüber dem geladenen Profil entfernt: " + ", ".join("`%s`" % _md(x.get("key"), 80) for x in removed[:40]) + ("" if len(removed) <= 40 else " …")]
    if only:
        L += ["", "Vorschlag ohne Zeile im Profil: " + ", ".join("`%s` = %s" % (_md(x.get("key"), 80), _md(redact.value_for_issue(str(x.get("key")), x.get("value")), 80) or "(leer, Schalter an)")
                                                              for x in only[:40]) + ("" if len(only) <= 40 else " …")]
    return L


def _issue_verdicts(dry, reg_rows: List[dict], line: str) -> List[str]:
    """Block Verdikte und Force: aus dem LETZTEN Trockenlauf; Forcebarkeit wird aus dem Register NEU gelesen (dem Browser wird sie nicht geglaubt)."""
    L = ["### Verdikte und Force", ""]
    have = isinstance(dry, dict) and isinstance(dry.get("rejections"), list)
    if not have:
        L.append("Kein Trockenlauf gefahren (Abschnitt Trockenlauf: Karten wählen, prüfen lassen, den Laufbericht neu erzeugen).")
    else:
        L.append("Trockenlauf: %s" % _md(dry.get("verdict") or ("Der Planer lehnt nichts ab." if not dry["rejections"] else ""), 400))
    reg = {r.get("code"): r for r in reg_rows or []}
    if have and dry["rejections"]:
        L += ["", "| Code | Klasse | Force | Text | Folge |", "|---|---|---|---|---|"]
        seen = set()
        for q in dry["rejections"][:64]:
            if not isinstance(q, dict) or not isinstance(q.get("code"), str) or not 0 < len(q["code"]) <= 40 or q["code"] in seen:
                continue
            seen.add(q["code"])
            r = reg.get(q["code"]) or {}
            label, state, _via = force_verdict(r) if r else ("unbekannter Code: nicht als forcebar behandelt", "blockiert", None)
            L.append("| `%s` | %s | %s | %s | %s |" % (_md(q["code"], 40), _md(r.get("klass_label") or r.get("klass") or "?", 60), _md("%s: %s" % (state, label), 160),
                                                       _md(_clip_text(q["code"], q.get("text")), 300), _md(r.get("consequence") or "–", 240)))
    fh = force_hint(dry, reg_rows, line)
    L += ["", "Force: %s" % _md(fh["text"], 400)]
    if fh["show_line"]:
        L.append("")
        L.append("Beim Serverstart `%s` setzen; übergangen werden: %s." % (fh["force_env"], ", ".join("`%s`" % c["code"] for c in fh["force_codes"])))
    if fh["blocked_codes"]:
        L.append("")
        L.append("Auch mit Force bestehen bleiben: %s." % ", ".join("`%s`" % c["code"] for c in fh["blocked_codes"]))
    if fh["records_note"]:
        L.append("")
        L.append(fh["records_note"])
    if have and dry.get("notes"):
        L += [""] + ["- Hinweis: " + _md(n, 300) for n in dry["notes"][:8]]
    return L


class ProfilEditor:
    def __init__(self, *, kartenplaner, release_dir: str = DEFAULT_RELEASE_DIR, user_dir: str = DEFAULT_USER_DIR,
                 tree: Optional[str] = None, catalog_file: str = CATALOG_FILE, topology=None):
        self.kp = kartenplaner
        #: Auftrag 1984 (C): ``topology(n) -> {"ok": True, "refused": None | text} | {"ok": False, "error": ..}``, gerechnet im Kindprozess mit der
        #: sglang-Umgebung (``CouplingsService.topology``); ohne sie rechnet der Trockenlauf wie bisher im Prozess
        self.topology = topology
        self.release_dir = release_dir
        self.user_dir = user_dir
        self.tree = find_tree(tree)
        self.catalog_file = catalog_file
        self._mods = None
        self._cat = None
        self._rel_cache: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ Planer-Module und Katalog
    def mods(self):
        if self._mods is None:
            if not self.tree:
                raise ProfilError("kein Planer-Baum mit weg2/profile_json.py und refusals.py gefunden (KARTENPLAN_TREE bzw. install_510.sh)")
            base = os.path.join(self.tree, "sglang", "srt", "weg2")
            self._mods = (_load("kp_profile_json", os.path.join(base, "profile_json.py")),
                          _load("kp_refusals", os.path.join(base, "refusals.py")))
        return self._mods

    def catalog(self) -> dict:
        if self._cat is None:
            try:
                with open(self.catalog_file, encoding="utf-8") as fh:
                    self._cat = json.load(fh)
            except (OSError, ValueError) as exc:
                raise ProfilError("Katalog %s nicht lesbar: %s (python -m sglang.srt.weg2.profile_catalog -o ...)" % (self.catalog_file, exc))
        return self._cat

    def specs(self) -> dict:
        ent = self.catalog()["entries"]
        return {n: {"bare": bool(e.get("bare")), "nargs": e.get("nargs")} for n, e in ent.items() if e["kind"] == "flag"}

    # ------------------------------------------------------------------ Profile auflisten
    def _release_path(self, name: str) -> str:
        if not NAME_RE.match(name or ""):
            raise ProfilError("ungültiger Profilname %r" % name)
        p = os.path.join(self.release_dir, name + ".env")
        if not os.path.isfile(p):
            raise ProfilError("Release-Profil %r nicht gefunden in %s" % (name, self.release_dir))
        return p

    def list(self) -> dict:
        rel = []
        try:
            names = sorted(n[:-4] for n in os.listdir(self.release_dir) if n.endswith(".env") and NAME_RE.match(n[:-4]))
        except OSError:
            names = []
        for n in names:
            with self._lock:
                hit = self._rel_cache.get(n)
            vars_ = {v["name"]: v.get("value", "") for v in (hit[1].get("vars") or [])} if hit else {}
            rel.append({"name": n, "line": vars_.get("PROFILE_LINE", ""), "status": vars_.get("PROFILE_STATUS", ""),
                        "format": vars_.get("PROFILE_FORMAT", ""), "loaded": bool(hit)})
        user = []
        try:
            for n in sorted(os.listdir(self.user_dir)):
                if n.endswith(".json") and NAME_RE.match(n[:-5]):
                    try:
                        with open(os.path.join(self.user_dir, n), encoding="utf-8") as fh:
                            d = json.load(fh)
                        user.append({"name": n[:-5], "line": d.get("line", ""), "based_on": (d.get("meta") or {}).get("based_on"),
                                     "saved": (d.get("meta") or {}).get("saved")})
                    except (OSError, ValueError):
                        user.append({"name": n[:-5], "line": "", "based_on": None, "saved": None, "broken": True})
        except OSError:
            pass
        cat = self.catalog()
        return {"release": rel, "user": user, "release_dir": self.release_dir, "user_dir": self.user_dir,
                "cards": CAT.catalog_public(), "pcie": {"gens": list(TR.GENS), "lanes": list(TR.LANES)},
                "rig_preset": self.kp.catalog().get("rig_preset"),
                "register": self.register(), "coverage": cat.get("stats"), "tree_rev": cat.get("tree_rev"),
                "planner_tree": self.tree}

    def known_models(self) -> dict:
        """Every model / draft path the release profiles name, with what THIS container can read of it.  A path that is not readable here
        stays in the list, named ("Modell im Container nicht gemountet"): the estimator reads only config.json and shard headers, so an empty mount
        point cannot be estimated, but it is still the profile's model (Koordinator 03.10.: die Zeile nicht einfach weglassen)."""
        seen: Dict[str, dict] = {}
        try:
            names = sorted(n[:-4] for n in os.listdir(self.release_dir) if n.endswith(".env") and NAME_RE.match(n[:-4]))
        except OSError:
            names = []
        for n in names:
            try:
                doc = self._import_release(n)
            except Exception:       # noqa: BLE001 -- one broken profile must not hide the others
                continue
            vars_ = {v["name"]: v for v in doc.get("vars") or []}
            for var, role in (("PROFILE_MODEL", "Modell"), ("PROFILE_DRAFT", "Draft")):
                path = str((vars_.get(var) or {}).get("value") or "")
                if not path:
                    continue
                e = seen.setdefault(path, {"path": path, "name": os.path.basename(path.rstrip("/")) or path, "roles": [], "used_by": []})
                if role not in e["roles"]:
                    e["roles"].append(role)
                e["used_by"].append(n)
        out = []
        for e in sorted(seen.values(), key=lambda x: x["name"]):
            e["state"], e["why"] = self._mount_state(e["path"])
            e["readable"] = e["state"] == "lesbar"
            out.append(e)
        return {"ok": True, "models": out}

    @staticmethod
    def _mount_state(path: str):
        if os.path.isfile(path):
            return "lesbar", "Datei lesbar (%s Bytes)" % os.path.getsize(path)
        if not os.path.isdir(path):
            return "nicht_gemountet", "Modell im Container nicht gemountet (Pfad fehlt)"
        try:
            entries = os.listdir(path)
        except OSError as exc:
            return "nicht_gemountet", "Modell im Container nicht gemountet (Verzeichnis nicht lesbar: %s)" % exc
        if not entries:
            return "nicht_gemountet", "Modell im Container nicht gemountet (Verzeichnis leer: Mountpunkt)"
        return "lesbar", "%d Einträge" % len(entries)

    def register(self) -> List[dict]:
        _pj, ref = self.mods()
        return ref.public_register(self.catalog().get("register_wired") or [])

    # ------------------------------------------------------------------ Laden
    def _import_release(self, name: str) -> dict:
        pj, _ref = self.mods()
        path = self._release_path(name)
        mt = os.path.getmtime(path)
        with self._lock:
            hit = self._rel_cache.get(name)
            if hit and hit[0] == mt:
                return json.loads(json.dumps(hit[1]))
        doc = pj.import_env(path, self.specs())
        pj.freeze_profile_values(doc, self.specs())
        with self._lock:
            self._rel_cache[name] = (mt, doc)
        return json.loads(json.dumps(doc))

    def _planner_values(self, profile_name: str, doc: dict) -> Dict[str, str]:
        """Planer-Werte aus der Aufzeichnung des Referenz-Boots (``kartenplan_data``), nur für Werte, die der Planer rechnet
        (Katalog ``planner_derived``): Schlüssel ``extra:P:--flag`` / ``env:P:NAME`` (Gruppen-argv und -env des Boots)."""
        prof = next((p for p in CAT.PROFILES if p["release_profile"] == profile_name), None)
        if not prof:
            return {}
        rec = self.kp._record(prof["record"])
        if not rec:
            return {}
        ent = self.catalog()["entries"]
        derived = {n for n, e in ent.items() if e.get("planner_derived")}
        out: Dict[str, str] = {}
        launch = rec.get("launch") or {}
        for g in ("P", "D"):
            argv = list((launch.get("argv") or {}).get(g) or [])
            i = 0
            while i < len(argv):
                a = argv[i]
                if a.startswith("--"):
                    if "=" in a:
                        n, v = a.split("=", 1)
                        i += 1
                    elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                        n, v = a, argv[i + 1]
                        i += 2
                    else:
                        n, v = a, ""
                        i += 1
                    if n in derived:
                        out["extra:%s:%s" % (g, n)] = v
                else:
                    i += 1
            for n, v in ((launch.get("env") or {}).get(g) or {}).items():
                if n in derived:
                    out["env:%s:%s" % (g, n)] = str(v)
        return out

    def load(self, kind: str, name: str) -> dict:
        pj, _ref = self.mods()
        comments: dict = {}
        if kind == "release":
            doc = self._import_release(name)
            base_rel = name
            comments = self._profile_comments(self._release_path(name))
            planner = self._planner_values(name, doc)
            doc["meta"]["planner"] = planner
            doc["meta"]["based_on"] = {"kind": "release", "name": name, "sha256": (doc.get("source") or {}).get("sha256")}
        elif kind == "user":
            doc = self.read_user(name)
            based = (doc.get("meta") or {}).get("based_on") or {}
            if based.get("kind") == "release" and NAME_RE.match(str(based.get("name", ""))) and os.path.isfile(
                    os.path.join(self.release_dir, str(based["name"]) + ".env")):
                comments = self._profile_comments(self._release_path(based["name"]))
                doc.setdefault("meta", {})["planner"] = self._planner_values(based["name"], doc)
        else:
            raise ProfilError("kind muss release oder user sein")
        return self.render_view(doc, comments)

    def _profile_comments(self, path: str) -> dict:
        """Kommentare über den Zeilen des Profils (profile_catalog.harvest_profile_comments), aus dem Planer-Baum."""
        cmod = os.path.join(self.tree, "sglang", "srt", "weg2", "profile_catalog.py") if self.tree else ""
        if not cmod or not os.path.isfile(cmod):
            return {}
        mod = getattr(self, "_catmod", None)
        if mod is None:
            mod = self._catmod = _load("kp_profile_catalog", cmod)
        return mod.harvest_profile_comments(path)

    #: Felder des Katalogeintrags, die eine Zeile zusätzlich zeigt: in welchem Code-Baum der Wert steht (``baeume``), wo die Bäume bei
    #: Standardwert oder Beschreibung abweichen (``abweichung``) und woher ein handgeschriebener Satz stammt (``satz_quelle``)
    ORIGIN_FIELDS = ("baeume", "abweichung", "satz_quelle")

    def _add_origin(self, rows: List[dict]) -> None:
        entries = self.catalog()["entries"]
        for r in rows:
            e = entries.get(r["name"])
            if e is None:
                continue
            for k in self.ORIGIN_FIELDS:
                if k in e:
                    r["explain"][k] = e[k]

    def render_view(self, doc: dict, comments: Optional[dict] = None) -> dict:
        pj, _ref = self.mods()
        v = pj.view(doc, self.catalog()["entries"], comments, self.specs())
        model_dir = next((str(x.get("value") or "") for x in doc.get("vars") or [] if x.get("name") == "PROFILE_MODEL"), "")
        v["kvheads"] = KVH.view(v["rows"], model_dir, v["planner_only"])
        v["glossar"] = self.catalog().get("glossar") or {}
        self._add_origin(v["rows"])
        return {"ok": True, "doc": doc, "view": v, "name": doc.get("name"), "line": doc.get("line"),
                "groups": sorted({r["explain"]["group"] for r in v["rows"] if r["explain"]["group"]})}

    def edit(self, doc: dict, edits: list) -> dict:
        pj, _ref = self.mods()
        if not isinstance(doc, dict) or doc.get("schema") != pj.SCHEMA:
            raise ProfilError("doc ist kein %s" % pj.SCHEMA)
        for e in edits or []:
            if not isinstance(e, dict) or "key" not in e:
                raise ProfilError("jede Änderung braucht key")
        new = pj.apply_edits(doc, edits or [], self.specs())
        comments = {}
        based = (new.get("meta") or {}).get("based_on") or {}
        if based.get("kind") == "release" and NAME_RE.match(str(based.get("name", ""))):
            try:
                comments = self._profile_comments(self._release_path(based["name"]))
            except ProfilError:
                comments = {}
        return self.render_view(new, comments)

    # ------------------------------------------------------------------ Nutzerprofile (State-Volume)
    def _user_path(self, name: str) -> str:
        if not NAME_RE.match(name or ""):
            raise ProfilError("Profilname muss [a-z0-9][a-z0-9._-]* sein (höchstens 64 Zeichen), nicht %r" % name)
        return os.path.join(self.user_dir, name + ".json")

    def read_user(self, name: str) -> dict:
        try:
            with open(self._user_path(name), encoding="utf-8") as fh:
                return json.load(fh)
        except OSError:
            raise ProfilError("Nutzerprofil %r nicht gefunden in %s" % (name, self.user_dir))
        except ValueError as exc:
            raise ProfilError("Nutzerprofil %r ist kein gültiges JSON: %s" % (name, exc))

    def save(self, doc: dict, name: str) -> dict:
        pj, _ref = self.mods()
        if not isinstance(doc, dict) or doc.get("schema") != pj.SCHEMA:
            raise ProfilError("doc ist kein %s" % pj.SCHEMA)
        path = self._user_path(name)
        d = json.loads(json.dumps(doc))
        d["name"] = name
        meta = d.setdefault("meta", {"origins": {}, "planner": {}, "notes": []})
        meta["saved"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # PROFILE_NAME ist der Name, unter dem der Entrypoint das Profil führt
        pj._set_in_doc(d, "var:PROFILE_NAME", name, self.specs())
        d["id"] = pj.doc_id(d)
        try:
            os.makedirs(self.user_dir, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".profil.", dir=self.user_dir)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=1, sort_keys=True, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            raise ProfilError("Speichern nach %s scheitert: %s (State-Volume beschreibbar? --profile-dir)" % (self.user_dir, exc))
        return {"ok": True, "name": name, "path": path, "id": d["id"], "saved": meta["saved"]}

    def delete(self, name: str) -> dict:
        path = self._user_path(name)
        try:
            os.unlink(path)
        except FileNotFoundError:
            raise ProfilError("Nutzerprofil %r nicht gefunden" % name)
        except OSError as exc:
            raise ProfilError("Löschen scheitert: %s" % exc)
        return {"ok": True, "name": name}

    # ------------------------------------------------------------------ Export (.env im heutigen Dialekt)
    def export_env(self, doc: dict, dry=None) -> dict:
        pj, _ref = self.mods()
        if not isinstance(doc, dict) or doc.get("schema") != pj.SCHEMA:
            raise ProfilError("doc ist kein %s" % pj.SCHEMA)
        text = pj.render_env(doc)
        problems = self.verify_render(doc, text)
        name = str(doc.get("name") or "profil")
        return {"ok": True, "env": text, "filename": "%s.env" % name, "verified": not problems, "problems": problems,
                "check": "Das exportierte .env wurde mit bash ausgewertet (HTSGLANG_INSTRUMENTS 0 und 1) und mit den Werten des Profils verglichen: "
                         + ("gleich." if not problems else "ABWEICHUNG, nicht verwenden."),
                "use": self.use_hint(name, dry, str(doc.get("line") or ""))}

    def verify_render(self, doc: dict, text: str) -> List[str]:
        pj, _ref = self.mods()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "check.env")
            with open(p, "w", encoding="utf-8", errors="surrogateescape") as fh:
                fh.write(text)
            got = pj.effective(p)
        want = pj.expected_effective(doc)
        return pj.diff_effective(want, got)

    def use_hint(self, name: str, dry=None, line: str = "") -> dict:
        """Wie das Profil beim Serverstart gebraucht wird: Beispiel-docker-run (nur Text, das Dashboard startet nichts).  ``dry`` = Antwort des
        letzten Trockenlaufs; nur wenn er forcebare Ablehnungen ergab, steht die Zeile ``-e FLLIPER_FORCE=1`` mit der Code-Liste darin
        (``force_env`` ist sonst ``None``).  Nicht übergehbare Codes stehen in ``force.blocked_codes`` (die Oberfläche zeigt sie rot)."""
        try:
            reg = self.register()
        except Exception:       # noqa: BLE001 -- ohne Register keine Force-Aussage, aber der Export lebt weiter
            reg = []
        fh = force_hint(dry, reg, line)
        return {"profile_env": "FLLIPER_PROFILE=%s" % name, "force_env": fh["force_env"], "force": fh,
                "docker_run": docker_run_example(name, fh),
                "text": "Am Server: das Profil angeben (FLLIPER_PROFILE=%s, JSON aus dem State-Volume /var/lib/flliper/profiles/%s.json "
                        "oder die exportierte Datei als <profiles>/%s.env). Das Dashboard startet nichts; unten steht ein Beispielaufruf zum Anpassen "
                        "(Image, Mounts und Flags sind Platzhalter)." % (name, name, name)}

    # ------------------------------------------------------------------ Issue-Text "Laufbericht" (AP-I)
    def issue_report(self, doc: dict, dry=None, cards=None, model=None, hardware_md: str = "", versions: Optional[dict] = None,
                     now: Optional[float] = None) -> dict:
        """Der Laufbericht als EIN Markdown-Block für ein GitHub-Issue: Hardwareprofil (Kurzform, ``hardware_md`` kommt aus ``hwprofil.issue_short``),
        Modellprofil, Betriebsform, Vorschlag + Übersteuerungen, Verdikte/Force, Versionen und der Platzhalter für Messergebnis und Boot-Log-Auszug.
        Nur Text; das Dashboard startet nichts.  ``dry`` = Antwort des letzten Trockenlaufs (oder ``None``), ``cards`` = die gewählten Karten
        ``[{card, pcie}]``, ``model`` = ein Modellprofil ``flliper.model/1`` (oder ``None``), ``versions`` = ``hwprofil.version_facts``.
        Geheimnisse (nach Name und nach Wert) und Hostpfade sind entfernt (``redact``); Forcebarkeit wird aus dem Register NEU gelesen."""
        pj, _ref = self.mods()
        if not isinstance(doc, dict) or doc.get("schema") != pj.SCHEMA:
            raise ProfilError("doc ist kein %s" % pj.SCHEMA)
        view = pj.view(doc, self.catalog()["entries"], None, self.specs())
        try:
            reg = self.register()
        except Exception:       # noqa: BLE001 -- ohne Register keine Force-Aussage (alle Codes "unbekannt"), der Bericht lebt weiter
            reg = []
        v = versions or {}
        now = time.time() if now is None else now
        name = _md(doc.get("name") or "profil", 80)
        line = str(doc.get("line") or "")
        labels, src = _issue_cards(dry, cards)
        form = issue_betriebsform([r["name"] for r in view["rows"]], len(labels) or None)
        by_name = {r["name"]: r.get("value") for r in view["rows"] if r["kind"] == "var"}
        meta = doc.get("meta") or {}
        based = meta.get("based_on") or {}
        L: List[str] = ["## Laufbericht (Profil-Editor): `%s`" % name, "",
                        "Erzeugt %s im Profil-Editor des Dashboards; er startet nichts. Alle Werte stammen aus dem Profil, dem Trockenlauf und den "
                        "Profilen von Hardware und Modell; was nicht belegt ist, steht als \"unbelegt\"." % time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now)), ""]
        L += (hardware_md.strip().split("\n") if hardware_md and hardware_md.strip() else
              ["### Hardwareprofil (Kurzform)", "", "Hardwareprofil nicht verfügbar (unbelegt)."])
        L += [""] + _issue_model(model, view["rows"])
        L += ["", "### Betriebsform", "", "| Angabe | Wert |", "|---|---|",
              "| Betriebsform | %s |" % _md(form["form"]),
              "| Abgeleitet aus | %s (aus den Flags des Profils gelesen, keine Wahl des Planers) |" % _md(form["why"]),
              "| Linie | %s |" % _md(line or "unbelegt"),
              "| Profil | `%s`%s |" % (name, (", Basis %s `%s`" % (_md(based.get("kind"), 20), _md(based.get("name"), 60))) if based.get("name") else ""),
              "| Profilstand | %s |" % _md(by_name.get("PROFILE_STATUS") or "unbelegt"),
              "| Karten (%s) | %d: %s |" % (src, len(labels), _md(", ".join(labels) or "keine", 400)),
              "| Kartenzahl laut Profil | %s |" % _md(by_name.get("PROFILE_CARD_COUNT") or "unbelegt"),
              "| Inventar laut Profil | %s |" % _md(by_name.get("PROFILE_INVENTORY") or "unbelegt")]
        L += [""] + _issue_proposal(view)
        L += [""] + _issue_verdicts(dry, reg, line)
        pid = str(doc.get("id") or "")
        sha = str(based.get("sha256") or "")
        sha = sha[len("sha256:"):] if sha.startswith("sha256:") else sha
        L += ["", "### Versionen", "", "| Angabe | Wert |", "|---|---|",
              "| Baum (Revision) | %s |" % _md(v.get("tree_rev") or "unbelegt"),
              "| Image | %s |" % _md(v.get("image") or "unbelegt (SGLANG_IMAGE_TAG nicht gesetzt)"),
              "| Treiber | %s |" % _md(v.get("driver") or "unbelegt"),
              "| CUDA / torch (Messprozess) | %s / %s |" % (_md(v.get("cuda") or "unbelegt"), _md(v.get("torch") or "unbelegt")),
              "| Dashboard | %s |" % _md(v.get("rigdash") or "unbelegt"),
              "| Profil-ID | `%s` |" % _md(pid[:19] if pid else "unbelegt"),
              "| Basisprofil (sha256) | %s |" % (("`%s`" % _md(sha[:16])) if sha else "unbelegt")]
        L += ["", "### Messergebnis / Boot-Log-Auszug", "",
              "<!-- Ergebnis des Starts eintragen: läuft / bricht ab, Messwerte (Durchsatz, Rundenzeit), die ersten Zeilen des Boot-Logs mit den "
              "Ablehnungen (REFUSED) und FORCED-PAST-Zeilen. Keine Schlüssel, keine Pfade des Rechners. -->", "",
              "Ergebnis: _(hier eintragen)_", "", "```text", "(Boot-Log-Auszug hier einfügen)", "```"]
        text = redact.text_for_issue("\n".join(L)) + "\n"
        return {"ok": True, "format": "markdown", "text": text, "blocks": [b for b in ISSUE_BLOCKS if ("### " + b) in text],
                "filename": "laufbericht-%s.md" % (name if NAME_RE.match(name) else "profil")}

    # ------------------------------------------------------------------ Topologie-Urteil (Kindprozess zuerst, Auftrag 1984 C)
    def _topology_verdict(self, n: int, tp, notes: List[str]) -> Optional[str]:
        """Der Text einer Topologie-Ablehnung für ``n`` Karten, oder ``None`` (durchgelassen / nicht prüfbar, dann steht eine Notiz in ``notes``).

        ``topology.plan_topology`` importiert für N != 3 ``weg2/weight_exchange_region`` (import sglang): das geht nur in der sglang-Umgebung, also
        fragt der Editor zuerst den Kopplungs-Worker (Kindprozess).  Ist keiner da oder antwortet er mit einem Fehler, rechnet die Funktion wie
        bisher im Prozess (N=3 braucht sglang nicht); scheitert auch das am Import, ist es KEINE Ablehnung und nie ein HTTP 500 (Browsertest 1979 F2),
        sondern die benannte Notiz "nicht geprüft"."""
        child_err = ""
        if self.topology is not None:
            try:
                res = self.topology(n)
            except Exception as exc:    # noqa: BLE001 -- ein kaputter Worker darf den Trockenlauf nicht beenden
                res = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
            if res.get("ok"):
                return res.get("refused") or None
            child_err = str(res.get("error") or "")
        try:
            tp.plan_topology(n)
        except ImportError as exc:
            notes.append("Topologie für %d Karte(n) nicht geprüft: das Planer-Gate braucht dafür die sglang-Umgebung (%s: %s)%s."
                         % (n, type(exc).__name__, exc,
                            "; Kopplungs-Python/Kindprozess nicht verfügbar: " + child_err if child_err else
                            " (kein Kindprozess mit --couplings-python / RIGDASH_COUPLINGS_PYTHON konfiguriert)"))
            return None
        except tp.TopologyRefused as exc:
            return str(exc)
        return None

    # ------------------------------------------------------------------ Trockenlauf: welche Ablehnungen hätte der Planer
    def dry_run(self, doc: dict, cards_req: list, host_patched: bool = True) -> dict:
        pj, ref = self.mods()
        if not isinstance(doc, dict) or doc.get("schema") != pj.SCHEMA:
            raise ProfilError("doc ist kein %s" % pj.SCHEMA)
        if not cards_req:
            raise ProfilError("mindestens eine Karte wählen")
        if len(cards_req) > MAX_CARDS:
            raise ProfilError("höchstens %d Karten" % MAX_CARDS)
        rows = {r["name"]: r for r in pj.rows(doc, self.specs()) if r["kind"] == "var"}

        def var(n: str) -> str:
            return str((rows.get(n) or {}).get("value", ""))

        cards = []
        for i, rc in enumerate(cards_req):
            e = CAT.card(rc.get("card"))
            if e is None:
                raise ProfilError("Karte %r nicht im Katalog" % rc.get("card"))
            link = TR.per_card_link(e, rc.get("pcie"))
            cards.append({"index": i, "entry": e, "label": CAT.label(e), "link": link})
        found: List[dict] = []
        notes: List[str] = []
        gate_rows = []
        for c in cards:
            e, l = c["entry"], c["link"]
            gate_rows.append({"nvml_index": c["index"], "uuid": "synthetisch-%d" % c["index"], "name": e["nvml_name"],
                              "total_mib": e["usable_mib"], "cc": e["cc"], "bar1_total_mib": l["bar1_mib"],
                              "pcie_max_gen": l["effective"]["gen"], "pcie_max_width": l["effective"]["lanes"]})
        try:
            ci, tp = self.kp._mods()
        except Exception as exc:        # noqa: BLE001
            ci = tp = None
            notes.append("Planer-Gate nicht verfügbar (%s): HW-Ablehnungen können nicht gezeigt werden." % exc)
        if ci is not None:
            from . import kartenplan_gate as GATE

            card_objs = [GATE._row(r) for r in gate_rows]
            bad = []
            for c_ in card_objs:
                try:
                    ci.arch_gate([c_])
                except ci.CardInventoryRefused as exc:
                    bad.append(str(exc))
            if bad:
                found.append({"code": "HW-ARCH", "text": bad[0] + ("  [gleiche Meldung für %d weitere Karte(n)]" % (len(bad) - 1) if len(bad) > 1 else ""),
                              "source": "weg2/card_identity.arch_gate"})
            try:
                want_n = int(var("PROFILE_CARD_COUNT") or 0) or None
            except ValueError:
                want_n = None
            try:
                ordered = ci.order_cards(card_objs, want_n, gate=False)
            except ci.CardInventoryRefused as exc:
                found.append({"code": "HW-COUNT", "text": str(exc), "source": "weg2/card_identity.order_cards (PROFILE_CARD_COUNT=%s)" % want_n})
                ordered = ci.order_cards(card_objs, None, gate=False)
            inv = ci.parse_inventory(var("PROFILE_INVENTORY")) or tuple(ci.REFERENCE_INVENTORY)
            positional = sorted({r["name"] for r in pj.rows(doc, self.specs())
                                 if _is_vector(r["value"], len(inv)) and r["kind"] != "var"})
            msg = ci.uncalibrated_message(ordered, list(inv), positional, "profile %r" % (doc.get("name") or var("PROFILE_NAME")))
            if msg:
                found.append({"code": "HW-UNCALIBRATED", "text": msg, "source": "weg2/card_identity.uncalibrated_message"})
            refused = self._topology_verdict(len(cards), tp, notes)
            if refused:
                # N inside the range that is only not proven ("N cards would be P = ..., proven on metal only for N in [3]") is the value
                # refusal HW-COUNT (the 27B line names it so); N with no topology at all is HW-TOPOLOGY (not forceable)
                code = "HW-COUNT" if (refused.startswith("HW-COUNT") or " would be " in refused) else "HW-TOPOLOGY"
                found.append({"code": code, "text": refused, "source": "weg2/topology.plan_topology"})
        st = var("PROFILE_STATUS") or ("platzhalter" if var("PROFILE_PLACEHOLDER") == "1" else "abgenommen")
        if st != "abgenommen":
            found.append({"code": "PROFIL-STATUS", "text": "Profil %r hat den Stand %s (%s)" % (doc.get("name"), st.upper(), var("PROFILE_OWNER") or "Eigentümer offen"),
                          "source": "docker/entrypoint.sh (PROFILE_STATUS)"})
        transport = TR.choose_transport([c["link"] for c in cards], [c["label"] for c in cards], host_patched=host_patched)
        if transport["transport"] == "nccl":
            notes.append("Transport NCCL statt barlink BAR1: " + " ".join(transport["reasons"]))
        notes.append("Nicht geprüft (das Dashboard sieht den Host nicht): Pfade des Modells, Drafts und Stores, SHM-Größe, freier Host-Speicher, "
                     "Belegung der Karten. Der Server prüft sie beim Start; die Belegungsprüfung hebt Force nie auf.")
        reg = {r["code"]: r for r in self.register()}
        out = []
        for f in found:
            r = reg.get(f["code"]) or {}
            force, state, via = force_verdict(r)
            out.append(dict(f, klass=r.get("klass"), klass_label=r.get("klass_label"), forcebar=bool(r.get("forcebar")),
                            wired=r.get("wired"), wired_at=r.get("wired_at"), force=force, force_state=state, force_via=via,
                            why_class=r.get("why_class"), consequence=r.get("consequence")))
        n_force = sum(1 for o in out if o["force_state"] == "force")
        n_open = sum(1 for o in out if o["force_state"] == "ungeprueft")
        n_block = len(out) - n_force - n_open
        if not out:
            verdict = "Der Planer lehnt dieses Profil auf den gewählten Karten nicht ab."
        else:
            verdict = "Der Planer lehnt %d Punkt(e) ab: Force übergeht %d beim Serverstart" % (len(out), n_force)
            n_ep = sum(1 for o in out if o["force_state"] == "force" and o.get("force_via") == "entrypoint")
            if n_ep:
                verdict += " (davon %d nur im Docker-Start, nicht im reinen Launcher-Aufruf)" % n_ep
            if n_open:
                verdict += ", %d prüft der Launcher noch nicht" % n_open
            if n_block:
                verdict += ", %d bleiben auch mit Force bestehen (nicht forcebar bzw. noch nicht verdrahtet)" % n_block
            verdict += "."
        return {"ok": True, "goes": not out, "verdict": verdict, "rejections": out, "notes": notes,
                "cards": [{"index": c["index"], "label": c["label"], "arch": c["entry"]["arch"]} for c in cards],
                "force_note": "Force gibt es nur am Serverstart (FLLIPER_FORCE=1 / --force), nicht im Dashboard. Er hebt alle Wert-Ablehnungen auf, "
                              "die der Launcher verdrahtet hat, und im Docker-Start zusätzlich die, die der Entrypoint selbst prüft "
                              "(PROFIL-STATUS, SHM, STORE, MEMAVAIL: im Docker-Start (Entrypoint) forcebar, im reinen Launcher-Aufruf nicht), "
                              "listet jede im Boot-Log als FORCED-PAST <CODE> <Grund> und schreibt keine Records. "
                              "Nicht übergangen werden: Belegungsprüfung (fremder Prozess/Fenster auf der Karte), fehlendes oder kaputtes Modell, "
                              "nicht unterstützte Architektur.",
                "reference": {"inventory": list(ci.REFERENCE_INVENTORY) if ci is not None else None}}


def _is_vector(value: str, n: int) -> bool:
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != n or n < 2:
        return False
    try:
        [float(p) for p in parts]
    except ValueError:
        return False
    return True
