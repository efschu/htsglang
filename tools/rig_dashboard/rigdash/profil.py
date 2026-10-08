"""Profil-Editor (Auftrag 930, S1): ein Serverprofil laden, bearbeiten, prüfen, speichern, als .env exportieren.

Das Dashboard ERSTELLT nur ein Profil (Nutzer-Entscheid 03.10. ~20:15Z): vorfüllen, bearbeiten, prüfen, als Datei
im State-Volume speichern.  Es startet nichts, hat keinen Force-Schalter und liefert keinen Startauftrag.  Den Start
macht der Nutzer am Server: ein Profil angeben (``FLLIPER_PROFILE=<name>``) und, wenn der Planer Werte ablehnt,
EIN Schalter ``FLLIPER_FORCE=1`` bzw. ``--force``.  Das Dashboard zeigt nur, welche Ablehnungen der Planer hätte
und welche davon der Force-Schalter übergeht.

Die Rechnung liegt im Planer-Baum, stdlib-rein und per Dateipfad geladen (wie ``kartenplan_gate``):

* ``pdflip/profile_json.py``  Profil als JSON (``flliper.server/1``), ``.env`` <-> JSON, Zeilen, Herkunft, Bearbeitung;
* ``pdflip/refusals.py``      das Ablehnungsregister (Wert-Ablehnung forcebar / nicht forcebar, je mit Begründung);
* ``pdflip/card_identity.py`` und ``topology.py`` für das Trockenlauf-Gate (über ``kartenplan``).

Erklärungen und Abhängigkeiten kommen aus ``profil_data/catalog.json`` (erzeugt von
``python -m flliper.srt.pdflip.profile_catalog``): kuratiert, aus argparse/environ geerntet, sonst "unerklärt".
Release-Profile (``.env``) werden mit bash ausgewertet (vertrauenswürdiges Verzeichnis, nur Lesen); Nutzerprofile
sind JSON-Dateien im State-Volume.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import shlex
import tempfile
import threading
import time
from typing import Dict, List, Optional

from . import hwprofil as HW
from . import kartenplan_catalog as CAT
from . import redact
from .kartenplan import MAX_CARDS
from . import kvheads as KVH
from . import kartenplan_transport as TR
from . import profil_oracle as ORA
from . import profil_planer as PLANER

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
        base = os.path.join(t, "flliper", "srt", "pdflip")
        if all(os.path.isfile(os.path.join(base, n)) for n in NEEDED):
            return t
    return None


#: Module der Dual-Form im Planer-Baum (``propose_dual`` importiert ``dual_layout_plan``, die Dual-ENV-Tabelle liest ``dual_green``): nur die 27B-Linie
#: traegt sie.  Auf der NF-Linie fehlen sie (Nutzerentscheid 07.10. "2 nein": der NF-Editor bietet Dual nicht an; vorher endete der Dual-Vorschlag dort mit
#: ImportError).
DUAL_MODULES = ("dual_layout_plan.py", "dual_green.py")
DUAL_FEHLT = "Dual is not available on this line"


def dual_line_probe(tree: Optional[str]) -> bool:
    """Traegt der Planer-Baum die Dual-Module?  Sonde auf Dateiebene (die Module existieren), nie ein Baum-SHA oder ein Branchname."""
    if not tree:
        return False
    base = os.path.join(tree, "flliper", "srt", "pdflip")
    return all(os.path.isfile(os.path.join(base, n)) for n in DUAL_MODULES)


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
        return "no, not forceable", "blockiert", None
    if r.get("wired"):
        return "yes, force overrides it (FORCED-PAST in the boot log)", "force", "launcher"
    ep = r.get("wired_entrypoint")
    if ep is None:
        ep = r.get("enforced_by") == "entrypoint"
    if ep:
        return ("yes, forceable in the Docker start (entrypoint), not in a plain launcher call (FORCED-PAST in the boot log)", "force", "entrypoint")
    if r.get("enforced_by") == "planner-gate":
        return "the launcher of this line does not check this yet: no refusal at start", "ungeprueft", None
    return "forceable, but wired neither in the launcher nor in the entrypoint of this line: the start still refuses", "blockiert", None


def force_hint(dry, register: List[dict], line: str = "") -> dict:
    """Der Force-Teil des Export-Hinweises aus dem LETZTEN Trockenlauf (``dry`` = dessen Antwort oder ``None``).  Nur Text; startet nichts.

    Fälle (``fall``): ``kein_trockenlauf`` | ``kein_launcher_lauf`` (Vorschlag ohne Launcher-Lauf: Planer-Rechnung oder Orakel-Fehler, kein Urteil über Force) | ``keine_ablehnung`` | ``nur_forcebar`` | ``gemischt`` | ``nur_nicht_forcebar`` | ``ungeprueft``.
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
        label, state, via = force_verdict(r) if r else ("unknown code: not treated as forceable", "blockiert", None)
        row = {"code": code, "text": text or r.get("title") or code, "via": via, "scope": r.get("force_scope") or label}
        {"force": force, "blockiert": blocked, "ungeprueft": openl}[state].append(row)
    if not have_dry:
        fall, text = "kein_trockenlauf", ("No dry run for this profile yet: whether the server start needs force is shown by the dry run (select cards, have it checked); export again afterwards.")
    elif dry.get("kein_lauf"):
        fall, text = "kein_launcher_lauf", str(dry.get("force_satz") or "No launcher run: the proposal says nothing about force; a dry run (Check again) asks the launcher.")[:600]
    elif not rejs:
        fall, text = "keine_ablehnung", "The planner refuses nothing; force is not needed."
    elif force and blocked:
        fall, text = "gemischt", "Force overrides part of the refusals but not all: the server does not start this way."
    elif force:
        fall, text = "nur_forcebar", "With FLLIPER_FORCE=1 the server overrides these refusals and lists each one in the boot log as FORCED-PAST."
    elif blocked:
        fall, text = "nur_nicht_forcebar", "Force does not help here: these refusals remain even with force."
    else:
        fall, text = "ungeprueft", "The launcher of this line does not check this yet: no force needed."
    out = {"fall": fall, "text": text, "show_line": bool(force), "force_env": FORCE_ENV if force else None,
           "force_codes": force, "blocked_codes": blocked, "open_codes": openl,
           "records_note": ("A force boot writes no records; its measured values do not count as accepted." if force else ""),
           "line_note": ""}
    if force and str(line) == "nf":
        out["line_note"] = ("Line nf: the launcher there overrides only HW-COUNT, HW-UNCALIBRATED and HOST-MEM; the entrypoint codes (PROFIL-STATUS, SHM, STORE, MEMAVAIL) apply there too.")
    return out


def docker_run_example(name: str, force: dict) -> List[str]:
    """Beispiel-``docker run`` als Zeilen (Platzhalter in <>, zum Anpassen); die Force-Zeile nur, wenn ``force["show_line"]``.
    Ein Kommentar steht VOR dem Befehl, nie hinter einem ``\\`` (hinter dem Zeilenumbruch-Backslash bricht Bash die Fortsetzung)."""
    lines = []
    if force.get("show_line"):
        lines.append("# FLLIPER_FORCE=1 is needed only because the planner refuses: %s" % ", ".join(c["code"] for c in force["force_codes"]))
    lines += ["docker run -d --name htsglang-mine \\",
              "  <the flags from section 3.3 of the README: --gpus, --shm-size, --memory, -p, model mounts> \\",
              "  -v flliper-state:/var/lib/flliper \\",
              "  -e MODE=pdflip -e FLLIPER_PROFILE=%s \\" % name]
    if force.get("show_line"):
        lines.append("  -e %s \\" % FORCE_ENV)
    lines.append("  ghcr.io/efschu/htsglang:<tag> serve")
    return lines


# ---------------------------------------------------------------------------------------------------------- Issue-Text "Laufbericht" (AP-I)
#: Höchstzahl der Zeilen je Tabelle im Laufbericht (ein Issue ist kein Profil-Dump; der Rest steht als "und N weitere")
ISSUE_MAX_ROWS = 120
ISSUE_CELL = 200
#: Überschriften des Laufberichts in fester Reihenfolge (der Test prüft jede)
ISSUE_BLOCKS = ("Hardware profile (short form)", "Model profile", "Operating mode", "Proposal and overrides", "Verdicts and force",
                "Versions", "Measurement result / boot log excerpt")


def _md(x, limit: int = ISSUE_CELL) -> str:
    """Ein Wert in einer Markdown-Tabellenzelle: kein Zeilenumbruch, kein Trennstrich, auf ``limit`` Zeichen gekürzt."""
    t = str("" if x is None else x).replace("|", "/").replace("\n", " ").replace("\r", " ").strip()
    return t if len(t) <= limit else t[:limit - 1].rstrip() + "…"


#: display words of the source tags of the model profile (the schema value ``geschätzt`` stays the key; the report shows English)
_SRC_DISPLAY = {"geschätzt": "estimated"}
#: display words of the ``force_state`` values in the run report (the API values stay German: ``force`` | ``blockiert`` | ``ungeprueft``)
_STATE_DISPLAY = {"force": "force", "blockiert": "blocked", "ungeprueft": "unchecked"}


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
        return {"form": "Dual PP/TP", "why": "Flag %s in the profile: P and D awake at the same time on the same cards" % ", ".join(dual)}
    if "--d-only" in names:
        return {"form": "TP only", "why": "Flag --d-only in the profile"}
    if n_cards == 1:
        return {"form": "Single card", "why": "one card in the dry run (the pdflip launcher needs at least two)"}
    return {"form": "Flip PP/TP", "why": "neither --d-only nor --dual-* in the profile, so the launcher's standard form"}


def _issue_cards(dry, cards) -> tuple:
    """(Kartenlabels, Quelle): die Karten des letzten Trockenlaufs, sonst die gewählten Karten (noch ohne Trockenlauf), sonst leer."""
    if isinstance(dry, dict) and isinstance(dry.get("cards"), list) and dry["cards"]:
        return [str(c.get("label") or "?") for c in dry["cards"] if isinstance(c, dict)][:16], "Dry run"
    out = []
    for rc in (cards if isinstance(cards, list) else [])[:16]:
        e = CAT.card(rc.get("card")) if isinstance(rc, dict) else None
        out.append(CAT.label(e) if e else "unknown card")
    return out, "selected, no dry run yet"


_AUSGANG_TEXT = {"geht": "passes without force", "geht_mit_force": "passes only with force", "verweigert": "refused, even with force",
                 "absturz": "dry run crashed (no verdict on the values)", "orakel_fehler": "the oracle could not be asked (no verdict)",
                 "passt": "Planner calculation: fits (no launcher run)", "passt_nicht": "Planner calculation: does not fit (no launcher run)",
                 "unbelegt": "Planner calculation: cannot be calculated"}


#: Ausgänge eines Vorschlags, bei denen KEIN Launcher-Lauf stattfand (Planer-Rechnung der Einzelkarte, Orakel nicht erreichbar)
_OHNE_LAUF = ("passt", "passt_nicht", "unbelegt", "orakel_fehler")


def _dry_ohne_lauf(vd: dict, ausgang: str) -> dict:
    """Der Vorschlag ohne Launcher-Lauf in der Form eines Trockenlaufs: ``kein_lauf`` True, KEINE Ablehnungen (``rejections`` leer) und ein eigener Block statt
    ``Trockenlauf:``: ``Planer-Rechnung: <Ausgang>`` mit den Prüfungen der Ebene ``fit`` oder ``Orakel-Fehler: <Text>``.  ``force_satz`` sagt ehrlich, dass daraus
    kein Force-Urteil folgt (nicht ``Der Planer lehnt nichts ab``)."""
    row_list = []
    for x in vd["verdikte"][:128]:
        if not isinstance(x, dict):
            continue
        if ausgang == "orakel_fehler":
            if x.get("ebene") == "orakel" and not row_list:
                row_list.append(str(x.get("text") or x.get("grund") or x.get("code") or "")[:600])
        elif x.get("ebene") == "fit" and isinstance(x.get("code"), str) and x["code"] != "EINZEL-PASSUNG":
            row_list.append("`%s`: %s" % (x["code"][:40], str(x.get("text") or x.get("grund") or "")[:300]))
    if ausgang == "orakel_fehler":
        t = row_list[0] if row_list else "no text"
        return {"rejections": [], "kein_lauf": True, "quelle": "vorschlag", "block_titel": "Oracle error: %s" % t, "block_zeilen": [],
                "verdict": "Oracle error: %s" % t,
                "force_satz": "The oracle could not be asked: there is no verdict, none on force either; Check again asks once more.",
                "notes": ["No launcher run: the proposal is not checked."]}
    titel = "Planner calculation: %s" % _AUSGANG_TEXT.get(ausgang, ausgang or "no outcome").replace("Planner calculation: ", "")
    return {"rejections": [], "kein_lauf": True, "quelle": "vorschlag", "block_titel": titel, "block_zeilen": row_list[:40], "verdict": titel,
            "force_satz": "No launcher run: the planner calculation names no register code that force could override; whether the start brings a refusal is shown only "
                          "by a dry run (Check again).",
            "notes": ["This is a planner calculation, not a verdict of the launcher."]}


def dry_from_vorschlag(vd) -> Optional[dict]:
    """Der Orakel-Lauf des Vorschlags (``verdikt`` der Antwort von ``propose``, ``flliper.verdikt/1``) in der Form eines Trockenlaufs, damit der Laufbericht
    auch ohne ``Neu prüfen`` sagt, was der Launcher zum Vorschlag gesagt hat.  Gelesen werden nur Code, Ebene und Text der Verdikte der Ebene Lauf/Absturz/Orakel;
    die Forcebarkeit rechnet der Bericht aus dem Register neu (dem Browser wird sie nicht geglaubt).  ``None`` = kein gültiges Verdikt."""
    if not isinstance(vd, dict) or vd.get("schema") != "flliper.verdikt/1" or not isinstance(vd.get("verdikte"), list):
        return None
    ausgang = str(vd.get("ausgang") or "")
    orakel = vd.get("orakel") if isinstance(vd.get("orakel"), dict) else {}
    laeufe = orakel.get("laeufe")
    if ausgang in _OHNE_LAUF or laeufe == 0:
        return _dry_ohne_lauf(vd, ausgang)
    rej, seen = [], set()
    for x in vd["verdikte"][:128]:
        if not isinstance(x, dict) or x.get("ebene") not in ("lauf", "absturz", "orakel") or x.get("parent"):
            continue
        code = x.get("code")
        if not isinstance(code, str) or not 0 < len(code) <= 40 or code in seen:
            continue
        seen.add(code)
        q = {"code": code, "text": str(x.get("text") or x.get("grund") or "")[:2000]}
        if x.get("klasse") == "nicht_forcebar":                    # nur zur Anzeige eines Codes ohne Registerzeile (W71-CENSUS, W64-OPPOINT): die Forcebarkeit bleibt "blockiert"
            q["klass_label"] = "not forceable"
        if x.get("konsequenz"):
            q["consequence"] = str(x["konsequenz"])[:400]
        rej.append(q)
    head = "Oracle run of the proposal%s: %s." % (" (launcher dry run, %s run(s))" % laeufe if isinstance(laeufe, int) and laeufe > 0 else "",
                                                 _AUSGANG_TEXT.get(ausgang, ausgang or "no outcome"))
    return {"rejections": rej, "verdict": head, "quelle": "vorschlag",
            "notes": ["The oracle run applies to the proposal as it was; changes made afterwards are not checked in it (Check again asks once more)."]}


def _issue_model(model, doc_rows) -> List[str]:
    """Block Modellprofil: die Werte des Schätzprofils ``flliper.model/1`` mit ihrer Quelle; ohne Profil nur, was das Serverprofil nennt.
    Der Pfad des Modells steht nie da, nur der Ordnername."""
    L = ["### Model profile", ""]
    p = model.get("profile") if isinstance(model, dict) and isinstance(model.get("profile"), dict) and "arch" not in model else model
    if not isinstance(p, dict) or p.get("schema") != "flliper.model/1":
        name = next((os.path.basename(str(r["value"]).rstrip("/")) for r in doc_rows if r["name"] in ("PROFILE_MODEL", "--model") and r["value"]), "")
        L.append("No model profile estimated (Models section: estimate the model profile, then regenerate the run report)."
                 + (" The server profile names the model `%s`." % _md(name) if name else ""))
        return L
    out: List[tuple] = []

    def add(label, o, fmt=None):
        v, src = _lv(o)
        if v is None or v == "" or v == [] or v == {}:
            return
        out.append((label, "%s (%s)" % (fmt(v) if fmt else v, _SRC_DISPLAY.get(src, src) or "?")))

    a, w, kv, st, ex, dr, cx = (p.get(k) or {} for k in ("arch", "weights", "kv", "state", "experts", "draft", "context"))
    name = os.path.basename(str(p.get("path") or "").rstrip("/"))
    out.append(("Model", "`%s`" % _md(name) if name else "unverified"))
    add("Format", p.get("format"))
    add("Type", a.get("family"), lambda v: "MoE" if v == "moe" else "dense")
    add("Hybrid (GDN/Mamba)", a.get("hybrid"), lambda v: "yes" if v else "no")
    add("Layer", a.get("n_layers"))
    add("Layer types", a.get("layer_counts"), lambda v: ", ".join("%s %s" % (k, v[k]) for k in sorted(v) if v[k]))
    add("Hidden size", a.get("hidden"))
    add("Heads Q / KV / head size", {"v": "%s / %s / %s" % (_lv(a.get("heads_q"))[0], _lv(a.get("heads_kv"))[0], _lv(a.get("head_dim"))[0]),
                                      "src": _lv(a.get("heads_q"))[1]} if _lv(a.get("heads_q"))[0] is not None else None)
    add("Attention", a.get("attention"))
    add("Total weights", w.get("total_bytes"), _gib)
    add("Experts (count)", ex.get("n"))
    add("Experts per token (top_k)", ex.get("top_k"))
    add("KV per token and attention layer", kv.get("cell_bytes_per_attn_layer_token"), lambda v: "%s B" % v)
    add("Mamba/GDN state per linear layer and request", st.get("per_linear_layer_per_slot_mib"), lambda v: "%.4g MiB" % v)
    add("MTP layers (draft in the model)", dr.get("mtp_layers"))
    if isinstance(dr.get("external"), dict):
        ext = dr["external"]
        add("External draft", ext.get("total_bytes"), lambda v: "%s, %s" % (os.path.basename(str(ext.get("path") or "").rstrip("/")) or "?", _gib(v)))
    add("Context (max. positions)", cx.get("max_position_embeddings"))
    if p.get("config_sha"):
        out.append(("config checksum", "`%s`" % _md(p["config_sha"])))
    L += ["| Item | Value (source) |", "|---|---|"] + ["| %s | %s |" % (_md(k), _md(v)) for k, v in out]
    L += ["", "Source: config = stated in config.json, index = from the tensor headers, estimated = calculated, stat = file size."]
    return L


def _issue_cell(row: dict, v, known) -> str:
    """Wert einer Zeile für die Tabelle: Schalter ohne Wert = ``on``; fehlt der Wert = ``–``; ein Geheimnis nach Namen = ``<redacted>``; ein Schlüssel,
    den der Katalog nicht kennt (``known`` = die Katalognamen), zeigt seinen Wert nie (``redact.HIDDEN_UNKNOWN``)."""
    if v is None:
        return "–"
    if v == "" and row.get("bare"):
        return "on"
    return _md(redact.value_for_issue(str(row.get("name") or ""), v, known)) or "(empty)"


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


def _issue_proposal(view: dict, known) -> List[str]:
    L = ["### Proposal and overrides", ""]
    d = issue_diff_rows(view)
    c, sel = d["counts"], d["rows"]
    removed, only = view.get("removed") or [], view.get("planner_only") or []
    L.append("Source: origin, profile value and planner proposal per row are in the profile (meta.origins, meta.profile_values, meta.planner). %d values, of which %d changed against the loaded profile, %d set by the user, %d with a planner proposal, %d differ from the proposal."
             % (c["rows"], c["geaendert"], c["nutzer"], c["mit_vorschlag"], c["weicht_vom_vorschlag_ab"]))
    if not c["mit_vorschlag"] and not only:
        L.append("")
        L.append("There is no planner proposal for this profile (meta.planner empty); the Proposal column stays empty.")
    extra = [k for k in ("state", "verdict") if any(k in r for r in sel)]
    head = ["Value", "Current", "Profile", "Proposal (planner)", "Origin"] + [{"state": "State", "verdict": "Verdict"}[k] for k in extra]
    if sel:
        L += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in sel[:ISSUE_MAX_ROWS]:
            cells = ["`%s`" % _md(r.get("key") or r.get("name")), _issue_cell(r, r.get("value"), known), _issue_cell(r, r.get("profile_value"), known),
                     _issue_cell(r, r.get("planner_value"), known), _md(r.get("origin_label") or r.get("origin"))]
            for k in extra:
                v = r.get(k)
                cells.append(_md((v.get("code") or v.get("text")) if isinstance(v, dict) else v) or "–")
            L.append("| " + " | ".join(cells) + " |")
        if len(sel) > ISSUE_MAX_ROWS:
            L.append("")
            L.append("… and %d more deviating values (truncated)." % (len(sel) - ISSUE_MAX_ROWS))
    else:
        L += ["", "No deviation: all values stand as in the loaded profile and, where there is one, as in the proposal."]
    if removed:
        L += ["", "Removed against the loaded profile: " + ", ".join("`%s`" % _md(x.get("key"), 80) for x in removed[:40]) + ("" if len(removed) <= 40 else " …")]
    if only:
        L += ["", "Proposal without a row in the profile: " + ", ".join("`%s` = %s" % (_md(x.get("key"), 80), _md(redact.value_for_issue(str(x.get("key")), x.get("value"), known), 80) or "(empty, switch on)")
                                                              for x in only[:40]) + ("" if len(only) <= 40 else " …")]
    return L


def _issue_verdicts(dry, reg_rows: List[dict], line: str) -> List[str]:
    """Block Verdicts and force: aus dem LETZTEN Trockenlauf; Forcebarkeit wird aus dem Register NEU gelesen (dem Browser wird sie nicht geglaubt)."""
    L = ["### Verdicts and force", ""]
    have = isinstance(dry, dict) and isinstance(dry.get("rejections"), list)
    if not have:
        L.append("No dry run was made (Dry run section: select cards, have it checked, regenerate the run report).")
    elif dry.get("kein_lauf"):
        L.append(_md(dry.get("block_titel") or dry.get("verdict"), 600).rstrip(".") + ".")
        if dry.get("block_zeilen"):
            L += [""] + ["- %s" % _md(z, 400) for z in dry["block_zeilen"][:40]]
    else:
        L.append("Dry run: %s" % _md(dry.get("verdict") or ("The planner refuses nothing." if not dry["rejections"] else ""), 400))
    reg = {r.get("code"): r for r in reg_rows or []}
    if have and dry["rejections"]:
        L += ["", "| Code | Class | Force | Text | Consequence |", "|---|---|---|---|---|"]
        seen = set()
        for q in dry["rejections"][:64]:
            if not isinstance(q, dict) or not isinstance(q.get("code"), str) or not 0 < len(q["code"]) <= 40 or q["code"] in seen:
                continue
            seen.add(q["code"])
            r = reg.get(q["code"]) or {}
            label, state, _via = force_verdict(r) if r else (("not forceable: remains even with force (code of the planner verdict, not in the launcher register)"
                                                              if q.get("klass_label") == "not forceable" else "unknown code: not treated as forceable"), "blockiert", None)
            L.append("| `%s` | %s | %s | %s | %s |" % (_md(q["code"], 40), _md(r.get("klass_label") or r.get("klass") or q.get("klass_label") or "?", 60), _md("%s: %s" % (_STATE_DISPLAY.get(state, state), label), 160),
                                                       _md(_clip_text(q["code"], q.get("text")), 300), _md(r.get("consequence") or q.get("consequence") or "–", 240)))
    fh = force_hint(dry, reg_rows, line)
    L += ["", "Force: %s" % _md(fh["text"], 400)]
    if fh["show_line"]:
        L.append("")
        L.append("Set `%s` at the server start; overridden: %s." % (fh["force_env"], ", ".join("`%s`" % c["code"] for c in fh["force_codes"])))
    if fh["blocked_codes"]:
        L.append("")
        L.append("Remaining even with force: %s." % ", ".join("`%s`" % c["code"] for c in fh["blocked_codes"]))
    if fh["records_note"]:
        L.append("")
        L.append(fh["records_note"])
    if have and dry.get("notes"):
        L += [""] + ["- Note: " + _md(n, 300) for n in dry["notes"][:8]]
    return L


class ProfilEditor:
    def __init__(self, *, kartenplaner, release_dir: str = DEFAULT_RELEASE_DIR, user_dir: str = DEFAULT_USER_DIR,
                 tree: Optional[str] = None, catalog_file: str = CATALOG_FILE, topology=None, oracle=None, hardware=None, check_path=None):
        self.kp = kartenplaner
        #: AP-D: das Orakel (``profil_oracle.OracleService``: Kindprozess mit dem Launcher-Trockenlauf + Cache).  ``None`` = ohne Orakel
        #: rechnet der Trockenlauf mit der Teilliste des Planer-Gates (die einzige Ausnahme: Orakel nicht konfiguriert oder nicht startbar)
        self.oracle = oracle
        #: ``hardware() -> {"ok", "profile": flliper.hardware/1}`` (der Dienst ``hwprofil.get``); gewaehlte Karten, die genau die NVML-Karten
        #: dieses Rigs sind, fragen das Orakel mit ihren echten UUIDs statt mit synthetischen
        self.hardware = hardware
        #: ``check_path(path, what) -> path`` (der Modellwurzel-Wachposten ``modellprofil.check_path``) fuer Modellpfade im Vorschlags-Aufruf
        self.check_path = check_path
        #: Auftrag 1984 (C): ``topology(n) -> {"ok": True, "refused": None | text} | {"ok": False, "error": ..}``, gerechnet im Kindprozess mit der
        #: flliper-Umgebung (``CouplingsService.topology``); ohne sie rechnet der Trockenlauf wie bisher im Prozess
        self.topology = topology
        self.release_dir = release_dir
        self.user_dir = user_dir
        self.tree = find_tree(tree)
        self.catalog_file = catalog_file
        self._mods = None
        self._cat = None
        self._rel_cache: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def dual_available(self) -> bool:
        """Bietet dieser Editor die Betriebsform Dual an?  Nur wenn der Planer-Baum die Dual-Module traegt (``dual_line_probe``)."""
        return dual_line_probe(self.tree)

    def forms(self) -> tuple:
        """Die Formen, die ``propose`` auf diesem Baum bedient: ``FORMS`` ohne ``dual``, wenn der Baum die Dual-Module nicht traegt (NF-Linie)."""
        return tuple(self.FORMS) if self.dual_available() else tuple(f for f in self.FORMS if f != "dual")

    # ------------------------------------------------------------------ Planer-Module und Katalog
    def mods(self):
        if self._mods is None:
            if not self.tree:
                raise ProfilError("no planner tree with pdflip/profile_json.py and refusals.py found (KARTENPLAN_TREE or install_510.sh)")
            base = os.path.join(self.tree, "flliper", "srt", "pdflip")
            self._mods = (_load("kp_profile_json", os.path.join(base, "profile_json.py")),
                          _load("kp_refusals", os.path.join(base, "refusals.py")))
        return self._mods

    def catalog(self) -> dict:
        if self._cat is None:
            try:
                with open(self.catalog_file, encoding="utf-8") as fh:
                    self._cat = json.load(fh)
            except (OSError, ValueError) as exc:
                raise ProfilError("Catalog %s not readable: %s (python -m flliper.srt.pdflip.profile_catalog -o ...)" % (self.catalog_file, exc))
        return self._cat

    def specs(self) -> dict:
        ent = self.catalog()["entries"]
        return {n: {"bare": bool(e.get("bare")), "nargs": e.get("nargs")} for n, e in ent.items() if e["kind"] == "flag"}

    # ------------------------------------------------------------------ Profile auflisten
    def _release_path(self, name: str) -> str:
        if not NAME_RE.match(name or ""):
            raise ProfilError("invalid profile name %r" % name)
        p = os.path.join(self.release_dir, name + ".env")
        if not os.path.isfile(p):
            raise ProfilError("Release profile %r not found in %s" % (name, self.release_dir))
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
                "planner_tree": self.tree,
                # AP-H1: die Daten der einen Seite (Betriebsformen, Abschnitte, Dual-ENV-Tabelle, Reglergrenzen); fehlt der Schlüssel, zeichnet die Seite wie bisher
                "planer": PLANER.ui_info(self.forms(), cat.get("entries"), self.oracle is not None, dual=self.dual_available())}

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
            for var, role in (("PROFILE_MODEL", "Model"), ("PROFILE_DRAFT", "Draft")):
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
            return "lesbar", "File readable (%s bytes)" % os.path.getsize(path)
        if not os.path.isdir(path):
            return "nicht_gemountet", "Model not mounted in the container (path missing)"
        try:
            entries = os.listdir(path)
        except OSError as exc:
            return "nicht_gemountet", "Model not mounted in the container (directory not readable: %s)" % exc
        if not entries:
            return "nicht_gemountet", "Model not mounted in the container (directory empty: mount point)"
        return "lesbar", "%d entries" % len(entries)

    def register(self) -> List[dict]:
        """The refusal register of the planner tree.  ``wired`` (which forcebar codes the launcher consults) is read from the launcher.py
        of THAT tree, not from the shipped catalog: the catalog covers both code lines and carries the wired list of one launcher, which on
        the other line marks codes the launcher does not have (NF line 07.10.: P-CARD/D-BUDGET/WAKE-CREDIT shown wired, HW-BORROWED not).
        Without a readable launcher the catalog's list is the fallback."""
        _pj, ref = self.mods()
        wired = None
        try:
            with open(os.path.join(self.tree, "flliper", "srt", "pdflip", "launcher.py"), encoding="utf-8") as fh:
                wired = ref.wired_codes(fh.read())
        except (OSError, AttributeError):
            wired = None
        return ref.public_register(wired if wired is not None else (self.catalog().get("register_wired") or []))

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
            raise ProfilError("kind must be release or user")
        return self.render_view(doc, comments)

    def _profile_comments(self, path: str) -> dict:
        """Kommentare über den Zeilen des Profils (profile_catalog.harvest_profile_comments), aus dem Planer-Baum."""
        cmod = os.path.join(self.tree, "flliper", "srt", "pdflip", "profile_catalog.py") if self.tree else ""
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
            raise ProfilError("doc is not a %s" % pj.SCHEMA)
        for e in edits or []:
            if not isinstance(e, dict) or "key" not in e:
                raise ProfilError("every change needs key")
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
            raise ProfilError("Profile name must be [a-z0-9][a-z0-9._-]* (at most 64 characters), not %r" % name)
        return os.path.join(self.user_dir, name + ".json")

    def read_user(self, name: str) -> dict:
        try:
            with open(self._user_path(name), encoding="utf-8") as fh:
                return json.load(fh)
        except OSError:
            raise ProfilError("User profile %r not found in %s" % (name, self.user_dir))
        except ValueError as exc:
            raise ProfilError("User profile %r is not valid JSON: %s" % (name, exc))

    def save(self, doc: dict, name: str) -> dict:
        pj, _ref = self.mods()
        if not isinstance(doc, dict) or doc.get("schema") != pj.SCHEMA:
            raise ProfilError("doc is not a %s" % pj.SCHEMA)
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
            raise ProfilError("Saving to %s fails: %s (state volume writable? --profile-dir)" % (self.user_dir, exc))
        return {"ok": True, "name": name, "path": path, "id": d["id"], "saved": meta["saved"]}

    def delete(self, name: str) -> dict:
        path = self._user_path(name)
        try:
            os.unlink(path)
        except FileNotFoundError:
            raise ProfilError("User profile %r not found" % name)
        except OSError as exc:
            raise ProfilError("Deleting fails: %s" % exc)
        return {"ok": True, "name": name}

    # ------------------------------------------------------------------ Export (.env im heutigen Dialekt)
    def export_env(self, doc: dict, dry=None) -> dict:
        pj, _ref = self.mods()
        if not isinstance(doc, dict) or doc.get("schema") != pj.SCHEMA:
            raise ProfilError("doc is not a %s" % pj.SCHEMA)
        text = pj.render_env(doc)
        problems = self.verify_render(doc, text)
        name = str(doc.get("name") or "profil")
        return {"ok": True, "env": text, "filename": "%s.env" % name, "verified": not problems, "problems": problems,
                "check": "The exported .env was evaluated with bash (HTSGLANG_INSTRUMENTS 0 and 1) and compared with the values of the profile: "
                         + ("gleich." if not problems else "DEVIATION, do not use."),
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
                "text": "On the server: name the profile (FLLIPER_PROFILE=%s, JSON from the state volume /var/lib/flliper/profiles/%s.json or the exported file as <profiles>/%s.env). The dashboard starts nothing; an example call to adapt is below (image, mounts and flags are placeholders)." % (name, name, name)}

    # ------------------------------------------------------------------ Issue-Text "Laufbericht" (AP-I)
    def issue_report(self, doc: dict, dry=None, cards=None, model=None, hardware_md: str = "", versions: Optional[dict] = None,
                     now: Optional[float] = None, vorschlag=None) -> dict:
        """Der Laufbericht als EIN Markdown-Block für ein GitHub-Issue: Hardwareprofil (Kurzform, ``hardware_md`` kommt aus ``hwprofil.issue_short``),
        Modellprofil, Betriebsform, Vorschlag + Übersteuerungen, Verdikte/Force, Versionen und der Platzhalter für Messergebnis und Boot-Log-Auszug.
        Nur Text; das Dashboard startet nichts.  ``dry`` = Antwort des letzten Trockenlaufs (oder ``None``), ``cards`` = die gewählten Karten
        ``[{card, pcie}]``, ``model`` = ein Modellprofil ``flliper.model/1`` (oder ``None``), ``versions`` = ``hwprofil.version_facts``.
        Geheimnisse (nach Name und nach Wert) und Hostpfade sind entfernt (``redact``); Forcebarkeit wird aus dem Register NEU gelesen.
        ``vorschlag`` = das ``verdikt`` der letzten Antwort von ``propose``: gibt es keinen Trockenlauf (``Neu prüfen``), nimmt der Block Verdikte/Force den
        Orakel-Lauf des Vorschlags (``dry_from_vorschlag``); ein Trockenlauf ist jünger und geht vor."""
        pj, _ref = self.mods()
        if not isinstance(doc, dict) or doc.get("schema") != pj.SCHEMA:
            raise ProfilError("doc is not a %s" % pj.SCHEMA)
        if not (isinstance(dry, dict) and isinstance(dry.get("rejections"), list)):
            dry = dry_from_vorschlag(vorschlag) or dry
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
        known = frozenset(self.catalog()["entries"])
        by_name = {r["name"]: redact.value_for_issue(r["name"], r.get("value"), known) for r in view["rows"] if r["kind"] == "var"}
        meta = doc.get("meta") or {}
        based = meta.get("based_on") or {}
        L: List[str] = ["## Run report (profile editor): `%s`" % name, "",
                        "Generated %s in the profile editor of the dashboard; it starts nothing. All values come from the profile, the dry run and the hardware and model profiles; what is not verified is shown as \"unverified\"." % time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now)), ""]
        L += (hardware_md.strip().split("\n") if hardware_md and hardware_md.strip() else
              ["### Hardware profile (short form)", "", "Hardware profile not available (unverified)."])
        L += [""] + _issue_model(model, view["rows"])
        L += ["", "### Operating mode", "", "| Item | Value |", "|---|---|",
              "| Operating mode | %s |" % _md(form["form"]),
              "| Derived from | %s (read from the flags of the profile, not a choice of the planner) |" % _md(form["why"]),
              "| Line | %s |" % _md(line or "unverified"),
              "| Profile | `%s`%s |" % (name, (", Basis %s `%s`" % (_md(based.get("kind"), 20), _md(based.get("name"), 60))) if based.get("name") else ""),
              "| Profile status | %s |" % _md(by_name.get("PROFILE_STATUS") or "unverified"),
              "| Cards (%s) | %d: %s |" % (src, len(labels), _md(", ".join(labels) or "none", 400)),
              "| Card count per profile | %s |" % _md(by_name.get("PROFILE_CARD_COUNT") or "unverified"),
              "| Inventory per profile | %s |" % _md(by_name.get("PROFILE_INVENTORY") or "unverified")]
        L += [""] + _issue_proposal(view, known)
        L += [""] + _issue_verdicts(dry, reg, line)
        pid = str(doc.get("id") or "")
        sha = str(based.get("sha256") or "")
        sha = sha[len("sha256:"):] if sha.startswith("sha256:") else sha
        L += ["", "### Versions", "", "| Item | Value |", "|---|---|",
              "| Tree (revision) | %s |" % _md(HW.version_tree_text(v)),
              "| Image | %s |" % _md(HW.version_image_text(v)),
              "| Driver | %s |" % _md(v.get("driver") or "unverified"),
              "| CUDA / torch (measuring process) | %s / %s |" % (_md(v.get("cuda") or "unverified"), _md(v.get("torch") or "unverified")),
              "| Dashboard | %s |" % _md(v.get("rigdash") or "unverified"),
              "| Profile ID | `%s` |" % _md(pid[:19] if pid else "unverified"),
              "| Base profile (sha256) | %s |" % (("`%s`" % _md(sha[:16])) if sha else "unverified")]
        L += ["", "### Measurement result / boot log excerpt", "",
              "<!-- Enter the result of the start: runs / aborts, measured values (throughput, round time), the first lines of the boot log with the refusals (REFUSED) and FORCED-PAST lines. No keys, no paths of the machine. -->", "",
              "Result: _(enter here)_", "", "```text", "(paste the boot log excerpt here)", "```"]
        text = redact.text_for_issue("\n".join(L)) + "\n"
        return {"ok": True, "format": "markdown", "text": text, "blocks": [b for b in ISSUE_BLOCKS if ("### " + b) in text],
                "filename": "laufbericht-%s.md" % (name if NAME_RE.match(name) else "profil")}

    # ------------------------------------------------------------------ Topologie-Urteil (Kindprozess zuerst, Auftrag 1984 C)
    def _topology_verdict(self, n: int, tp, notes: List[str]) -> Optional[str]:
        """Der Text einer Topologie-Ablehnung für ``n`` Karten, oder ``None`` (durchgelassen / nicht prüfbar, dann steht eine Notiz in ``notes``).

        ``topology.plan_topology`` importiert für N != 3 ``pdflip/weight_exchange_region`` (import flliper): das geht nur in der flliper-Umgebung, also
        fragt der Editor zuerst den Kopplungs-Worker (Kindprozess).  Ist keiner da oder antwortet er mit einem Fehler, rechnet die Funktion wie
        bisher im Prozess (N=3 braucht flliper nicht); scheitert auch das am Import, ist es KEINE Ablehnung und nie ein HTTP 500 (Browsertest 1979 F2),
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
            notes.append("Topology for %d card(s) not checked: the planner gate needs the flliper environment for it (%s: %s)%s."
                         % (n, type(exc).__name__, exc,
                            "; couplings Python / child process not available: " + child_err if child_err else
                            " (no child process with --couplings-python / RIGDASH_COUPLINGS_PYTHON configured)"))
            return None
        except tp.TopologyRefused as exc:
            return str(exc)
        return None

    # ------------------------------------------------------------------ Trockenlauf: welche Ablehnungen hätte der Planer
    def _build_cards(self, cards_req: list) -> List[dict]:
        if not cards_req:
            raise ProfilError("select at least one card")
        if len(cards_req) > MAX_CARDS:
            raise ProfilError("at most %d cards" % MAX_CARDS)
        cards = []
        for i, rc in enumerate(cards_req):
            e = CAT.card(rc.get("card"))
            if e is None:
                raise ProfilError("Card %r not in the catalog" % rc.get("card"))
            link = TR.per_card_link(e, rc.get("pcie"))
            cards.append({"index": i, "entry": e, "label": CAT.label(e), "link": link})
        return cards

    @staticmethod
    def _gate_rows(cards: List[dict]) -> List[dict]:
        out = []
        for c in cards:
            e, l = c["entry"], c["link"]
            out.append({"nvml_index": c["index"], "uuid": "synthetisch-%d" % c["index"], "name": e["nvml_name"],
                        "total_mib": e["usable_mib"], "cc": e["cc"], "bar1_total_mib": l["bar1_mib"],
                        "pcie_max_gen": l["effective"]["gen"], "pcie_max_width": l["effective"]["lanes"]})
        return out

    def _gate_found(self, pj, doc: dict, cards: List[dict], var, notes: List[str]) -> List[dict]:
        """Die Teilprüfung des Planer-Gates (Karten, Kartenzahl, Inventar, Topologie) ohne Launcher: NUR der Rückfall, wenn kein Orakel
        konfiguriert oder nicht startbar ist (die Notiz sagt es).  Der Orakel-Weg (``_dry_oracle``) ersetzt diese Liste durch das, was der
        Launcher selbst sagt."""
        found: List[dict] = []
        gate_rows = self._gate_rows(cards)
        try:
            ci, tp = self.kp._mods()
        except Exception as exc:        # noqa: BLE001
            ci = tp = None
            notes.append("Planner gate not available (%s): HW refusals cannot be shown." % exc)
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
                found.append({"code": "HW-ARCH", "text": bad[0] + ("  [same message for %d more card(s)]" % (len(bad) - 1) if len(bad) > 1 else ""),
                              "source": "pdflip/card_identity.arch_gate"})
            try:
                want_n = int(var("PROFILE_CARD_COUNT") or 0) or None
            except ValueError:
                want_n = None
            try:
                ordered = ci.order_cards(card_objs, want_n, gate=False)
            except ci.CardInventoryRefused as exc:
                found.append({"code": "HW-COUNT", "text": str(exc), "source": "pdflip/card_identity.order_cards (PROFILE_CARD_COUNT=%s)" % want_n})
                ordered = ci.order_cards(card_objs, None, gate=False)
            inv = ci.parse_inventory(var("PROFILE_INVENTORY")) or tuple(ci.REFERENCE_INVENTORY)
            positional = sorted({r["name"] for r in pj.rows(doc, self.specs())
                                 if _is_vector(r["value"], len(inv)) and r["kind"] != "var"})
            msg = ci.uncalibrated_message(ordered, list(inv), positional, "profile %r" % (doc.get("name") or var("PROFILE_NAME")))
            if msg:
                found.append({"code": "HW-UNCALIBRATED", "text": msg, "source": "pdflip/card_identity.uncalibrated_message"})
            refused = self._topology_verdict(len(cards), tp, notes)
            if refused:
                # N inside the range that is only not proven ("N cards would be P = ..., proven on metal only for N in [3]") is the value
                # refusal HW-COUNT (the 27B line names it so); N with no topology at all is HW-TOPOLOGY (not forceable)
                code = "HW-COUNT" if (refused.startswith("HW-COUNT") or " would be " in refused) else "HW-TOPOLOGY"
                found.append({"code": code, "text": refused, "source": "pdflip/topology.plan_topology"})
        return found

    def _count_found(self, cards: List[dict], var) -> List[dict]:
        """``PROFILE_CARD_COUNT`` gegen die gewählten Karten (das prüft der Entrypoint, nicht der Launcher): dieselbe Prüfung und derselbe Text
        wie im Rückfall-Gate; der Orakel-Weg ergänzt sie, damit sie nicht verloren geht."""
        try:
            want_n = int(var("PROFILE_CARD_COUNT") or 0) or None
        except ValueError:
            want_n = None
        if want_n is None:
            return []
        try:
            ci, _tp = self.kp._mods()
        except Exception:               # noqa: BLE001 -- ohne Gate keine Zählung; das Orakel urteilt trotzdem
            return []
        from . import kartenplan_gate as GATE

        try:
            ci.order_cards([GATE._row(r) for r in self._gate_rows(cards)], want_n, gate=False)
        except ci.CardInventoryRefused as exc:
            return [{"code": "HW-COUNT", "text": str(exc), "source": "pdflip/card_identity.order_cards (PROFILE_CARD_COUNT=%s)" % want_n}]
        return []

    @staticmethod
    def _hw_cards_match(hw: dict, cards: List[dict]) -> bool:
        """Sind die gewählten Karten genau die NVML-Karten dieses Rigs (gleiche Namen, gleiche Größen, gleiche Zahl)?"""
        try:
            have = sorted((str(c["name"]), int((c["vram_total_mib"] or {}).get("v") if isinstance(c["vram_total_mib"], dict) else c["vram_total_mib"]))
                          for c in hw.get("cards") or [])
            want = sorted((str(c["entry"]["nvml_name"]), int(c["entry"]["usable_mib"])) for c in cards)
        except (KeyError, TypeError, ValueError):
            return False
        return bool(have) and have == want

    def _hardware_profile(self) -> Optional[dict]:
        if self.hardware is None:
            return None
        try:
            hw = self.hardware()
        except Exception:               # noqa: BLE001 -- ohne Hardwareprofil gibt es synthetische Karten
            return None
        prof = hw.get("profile") if isinstance(hw, dict) else None
        return prof if isinstance(prof, dict) and prof.get("cards") else None

    def _inventar_for(self, cards: List[dict], notes: List[str]) -> dict:
        """Das Inventar der Orakel-Frage: die NVML-Karten dieses Rigs (echte UUIDs), wenn die gewählten Karten genau sie sind, sonst
        synthetische Karten aus dem Katalog (Datenblatt, synthetische UUID: der Launcher kennt solche UUIDs nicht, z. B. im Census)."""
        hw = self._hardware_profile()
        if hw is not None and self._hw_cards_match(hw, cards):
            notes.append("The selected cards are the NVML cards of this rig (hardware profile): the dry run runs with their real UUIDs.")
            return {"hardware": hw}
        notes.append("Synthetic cards from the catalog (datasheet, synthetic UUID): what the launcher binds to card UUIDs (census) does not know them.")
        return {"cards": [{"entry": c["entry"], "link": c["link"]} for c in cards]}

    @staticmethod
    def _inventar_key(inv: dict) -> list:
        if inv.get("hardware"):
            return [["nvml", str(c.get("uuid")), str(c.get("name")), c.get("vram_total_mib"), c.get("cc")] for c in inv["hardware"].get("cards") or []]
        return [["katalog", str(c["entry"].get("id")), (c.get("link") or {}).get("effective"), (c.get("link") or {}).get("bar1_mib")]
                for c in inv.get("cards") or []]

    def _dry_oracle(self, doc: dict, cards: List[dict], var, notes: List[str]) -> Optional[dict]:
        """Das Orakel zum Profil auf den gewählten Karten: ``{"found": [...], "verdikt": ..., "res": ...}``; ``None`` (mit Notiz), wenn es
        nicht fragen konnte -- dann gilt die Teilprüfung des Planer-Gates."""
        pj, _ref = self.mods()
        try:
            text = pj.render_env(doc)
        except Exception as exc:        # noqa: BLE001 -- ein Profil, das sich nicht darstellen lässt, geht an den Rückfall
            notes.append("Oracle not asked: the profile cannot be rendered as .env (%s: %s). The partial check of the planner gate applies."
                         % (type(exc).__name__, exc))
            return None
        inv = self._inventar_for(cards, notes)
        parts = {"inventar": self._inventar_key(inv), "env_sha256": ORA.sha256_text(text), "form": None}
        based = (doc.get("meta") or {}).get("based_on") or {}
        res = self.oracle.ask("verdikt", {"basis": {"env_text": text, "source": "%s.env" % (based.get("name") or doc.get("name") or "profil")}, "inventar": inv}, parts)
        if not res.get("ok"):
            notes.append("Oracle (launcher dry run) not available: %s. The partial check of the planner gate applies (cards, card count, topology); the launcher calculation is missing." % (res.get("error") or "unknown error"))
            return None
        v = res.get("verdikt") or {}
        if v.get("ausgang") == "orakel_fehler":
            why = next((x.get("text") for x in v.get("verdikte") or [] if x.get("code") == "ORAKEL-FEHLER"), "")
            notes.append("The oracle could not be asked: %s. The partial check of the planner gate applies." % why)
            return None
        found = self._found_from_verdikt(v)
        for f in self._count_found(cards, var):
            if not any(x["code"] == "HW-COUNT" and x["text"] == f["text"] for x in found):
                found.append(f)
        return {"found": found, "verdikt": v, "res": res}

    @staticmethod
    def _found_from_verdikt(v: dict) -> List[dict]:
        """Ablehnungen der Rückgabeform ``{code, text, source}`` aus den Verdikten der Ebene Lauf (durchgelassen oder beendend); die Blocker
        im Text von HW-COUNT, FIT und Hinweise stehen in ``verdikte`` der Antwort, nicht als eigene Ablehnung."""
        out = []
        for x in v.get("verdikte") or []:
            if x.get("ebene") in ("lauf", "absturz", "orakel") and not x.get("parent"):
                src = "Launcher dry run (oracle, pdflip/propose_verdict)"
                if x.get("launcher_code"):
                    src += ", %s" % x["launcher_code"]
                if x.get("wo"):
                    src += ", %s" % x["wo"]
                out.append({"code": x["code"], "text": x.get("text") or x.get("grund") or x["code"], "source": src, "verdikt": x})
        return out

    @staticmethod
    def _register_row(f: dict, reg: dict) -> dict:
        """Die Registerzeile zu einer Ablehnung; ein Orakel-Code ohne Registerzeile (ORAKEL-ABSTURZ ...) bekommt eine nicht forcebare aus seinem Verdikt."""
        r = reg.get(f["code"])
        if r is None and f.get("verdikt"):
            v = f["verdikt"]
            return {"code": f["code"], "klass": "nicht_forcebar", "klass_label": "not forceable", "forcebar": False, "wired": None,
                    "wired_at": None, "why_class": "%s: %s" % (v.get("titel") or f["code"], v.get("konsequenz") or ""), "consequence": v.get("konsequenz")}
        return r or {}

    def dry_run(self, doc: dict, cards_req: list, host_patched: bool = True) -> dict:
        """Trockenlauf: welche Ablehnungen hätte der Planer für dieses Profil auf diesen Karten.

        AP-D: mit Orakel (``self.oracle``) sagt der LAUNCHER selbst, was er daraus macht (Trockenlauf auf einem NVML-Replay der gewählten Karten,
        erst ohne, dann mit ``--force``: jede Wert-Ablehnung, die Force übergeht, und was auch dann noch beendet; ein Absturz des Launchers ist
        ein Verdikt ``ORAKEL-ABSTURZ``).  Das Rückgabeformat bleibt (``ok, goes, verdict, rejections, notes, cards, force_note, reference``);
        neu sind ``quelle`` (``orakel`` | ``gate``), ``orakel`` (Ausgang, Profil-Hash, Cache) und ``verdikte`` (alle Verdikte, auch Blocker und Hinweise)."""
        pj, ref = self.mods()
        if not isinstance(doc, dict) or doc.get("schema") != pj.SCHEMA:
            raise ProfilError("doc is not a %s" % pj.SCHEMA)
        rows = {r["name"]: r for r in pj.rows(doc, self.specs()) if r["kind"] == "var"}

        def var(n: str) -> str:
            return str((rows.get(n) or {}).get("value", ""))

        cards = self._build_cards(cards_req)
        notes: List[str] = []
        orakel = self._dry_oracle(doc, cards, var, notes) if self.oracle is not None else None
        if orakel is not None:
            found = list(orakel["found"])
            source = "orakel"
        else:
            found = self._gate_found(pj, doc, cards, var, notes)
            source = "gate"
        st = var("PROFILE_STATUS") or ("platzhalter" if var("PROFILE_PLACEHOLDER") == "1" else "abgenommen")
        if st != "abgenommen":
            found.append({"code": "PROFIL-STATUS", "text": "Profile %r has the status %s (%s)" % (doc.get("name"), st.upper(), var("PROFILE_OWNER") or "owner open"),
                          "source": "docker/entrypoint.sh (PROFILE_STATUS)"})
        transport = TR.choose_transport([c["link"] for c in cards], [c["label"] for c in cards], host_patched=host_patched)
        if transport["transport"] == "nccl":
            notes.append("Transport NCCL instead of barlink BAR1: " + " ".join(transport["reasons"]))
        if orakel is not None:
            notes.append("Not checked (the launcher dry run runs on a replica of the cards and a fixed quiet host): model and draft files (header snapshots or sibling checkpoints stand in for empty mount points), real free host memory, SHM size, occupancy of the cards. The server checks them at start; force never lifts the occupancy check.")
            notes += [str(x) for x in (orakel["verdikt"].get("orakel") or {}).get("notizen") or []]
        else:
            notes.append("Not checked (the dashboard cannot see the host): paths of the model, drafts and stores, SHM size, free host memory, occupancy of the cards. The server checks them at start; force never lifts the occupancy check.")
        reg = {r["code"]: r for r in self.register()}
        out = []
        for f in found:
            r = self._register_row(f, reg)
            force, state, via = force_verdict(r)
            row = dict({k: v for k, v in f.items() if k != "verdikt"}, klass=r.get("klass"), klass_label=r.get("klass_label"), forcebar=bool(r.get("forcebar")),
                       wired=r.get("wired"), wired_at=r.get("wired_at"), force=force, force_state=state, force_via=via,
                       why_class=r.get("why_class"), consequence=r.get("consequence"))
            if f.get("verdikt"):
                row["verdikt"] = f["verdikt"]
            out.append(row)
        n_force = sum(1 for o in out if o["force_state"] == "force")
        n_open = sum(1 for o in out if o["force_state"] == "ungeprueft")
        n_block = len(out) - n_force - n_open
        if not out:
            verdict = "The planner does not refuse this profile on the selected cards."
        else:
            verdict = "The planner refuses %d item(s): force overrides %d at the server start" % (len(out), n_force)
            n_ep = sum(1 for o in out if o["force_state"] == "force" and o.get("force_via") == "entrypoint")
            if n_ep:
                verdict += " (of which %d only in the Docker start, not in a plain launcher call)" % n_ep
            if n_open:
                verdict += ", %d the launcher does not check yet" % n_open
            if n_block:
                verdict += ", %d remain even with force (not forceable or not wired yet)" % n_block
            verdict += "."
        res = {"ok": True, "goes": not out, "verdict": verdict, "rejections": out, "notes": notes,
               "cards": [{"index": c["index"], "label": c["label"], "arch": c["entry"]["arch"]} for c in cards],
               "force_note": "Force exists only at the server start (FLLIPER_FORCE=1 / --force), not in the dashboard. It lifts all value refusals that the launcher has wired and, in the Docker start, additionally those the entrypoint checks itself (PROFIL-STATUS, SHM, STORE, MEMAVAIL: forceable in the Docker start (entrypoint), not in a plain launcher call), lists each one in the boot log as FORCED-PAST <CODE> <reason> and writes no records. Not overridden: the occupancy check (foreign process/window on the card), a missing or broken model, an unsupported architecture.",
               "reference": {"inventory": list(ci.REFERENCE_INVENTORY) if (ci := self._ci()) is not None else None},
               "quelle": source}
        if orakel is not None:
            v = orakel["verdikt"]
            res["verdikte"] = v.get("verdikte") or []
            based = (doc.get("meta") or {}).get("based_on") or {}
            res["orakel"] = {"ausgang": v.get("ausgang"), "geht": v.get("geht"), "geht_mit_force": v.get("geht_mit_force"), "profil": v.get("profil"),
                             # Plan 4c: der Hash des Profils, nach dem gefragt wurde: Text (wie dem Orakel gegeben), Datei des Release-Profils (beim Laden)
                             "profil_text_sha256": ORA.sha256_text(pj.render_env(doc)), "profil_datei_sha256": based.get("sha256"), "basis": based or None,
                             "argv_sha256": v.get("argv_sha256"), "dauer_s": (v.get("orakel") or {}).get("dauer_s"),
                             "version": (v.get("orakel") or {}).get("version"), "laeufe": (v.get("orakel") or {}).get("laeufe"),
                             "cached": bool(orakel["res"].get("cached")), "cache_key": orakel["res"].get("cache_key"), "plan": v.get("plan"),
                             "zaehlung": v.get("zaehlung")}
        return res

    def _ci(self):
        try:
            return self.kp._mods()[0]
        except Exception:               # noqa: BLE001 -- ohne Gate keine Referenzliste
            return None

    # ------------------------------------------------------------------ Vorschlag (Stufe A + Orakel + Verdikte, AP-D)
    PROPOSE_SCHEMA = "flliper.propose-d/1"
    #: die Formen des Vorschlags: flip, tp und dual = Planer + Launcher-Trockenlauf (Orakel); single = die Einzelkarte, Planer-Rechnung ohne Launcher
    #: (``pdflip/propose_single``, kein pdflip-Launcher bei N=1: ``topology.py`` MIN_CARDS=2)
    FORMS = ("flip", "tp", "dual", "single")
    #: Namen der Einzelkarte in einer Anfrage (der Plan sagt ``einzel``, die Seite schickt ``single``)
    FORM_ALIAS = {"einzel": "single", "einzelkarte": "single"}
    #: Betriebsform des Vorschlags -> Form des Balkenvertrags ``flliper.balken/1`` (AP-H2: ``what=phase_bars``, ``form``; ``profile_couplings.FORMS``)
    BALKEN_FORM = {"flip": "flip", "tp": "d_only", "dual": "dual", "single": "single"}
    ZIELE_INT = {"seats": (1, 256), "kv_tokens": (1024, 8 << 20)}
    ZIELE_CHOICE = {"kv_dtype": ("auto", "fp8_e4m3"), "p_cut": ("auto", "pin", "seed"), "draft_kv_on_p": ("on", "off")}
    #: Ziele, die nur die Einzelkarte kennt (``propose_single``: Host-RAM-Budget fuer HiCache, gemessenes ``pre_model_load_memory``, gemessene Reserve, Draft-Wahl)
    ZIELE_INT_EINZEL = {"host_ram_mib": (1, 8 << 20), "pre_load_free_mib": (1, 1 << 20), "reserve_mib": (0, 1 << 20)}
    ZIELE_CHOICE_EINZEL = {"draft": ("auto", "on", "off", "nextn", "external")}

    def _ziele(self, z, form: str = "flip") -> dict:
        """Die Ziele des Vorschlags (``propose(ziele)``), geprüft: nur bekannte Schlüssel, Zahlen im Bereich, Auswahl aus der Liste."""
        if z in (None, {}):
            return {}
        if not isinstance(z, dict):
            raise ProfilError("ziele must be a JSON object")
        ints, choices = dict(self.ZIELE_INT), dict(self.ZIELE_CHOICE)
        if form == "single":
            ints.update(self.ZIELE_INT_EINZEL)
            choices.update(self.ZIELE_CHOICE_EINZEL)
        known = set(ints) | set(choices) | {"d_objective", "force_rules"}
        bad = sorted(set(z) - known)
        if bad:
            raise ProfilError("unknown goals: %s (allowed: %s)" % (", ".join(bad), ", ".join(sorted(known))))
        out: dict = {}
        for k, (lo, hi) in ints.items():
            if z.get(k) not in (None, ""):
                try:
                    v = int(z[k])
                except (TypeError, ValueError):
                    raise ProfilError("Goal %s must be an integer" % k)
                if not lo <= v <= hi:
                    raise ProfilError("Goal %s must be between %d and %d" % (k, lo, hi))
                out[k] = v
        for k, opts in choices.items():
            if z.get(k) not in (None, ""):
                if z[k] not in opts:
                    raise ProfilError("Goal %s must be one of %s" % (k, ", ".join(opts)))
                out[k] = z[k]
        if z.get("d_objective") not in (None, ""):
            v = str(z["d_objective"])
            if not (0 < len(v) <= 32 and all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in v)):
                raise ProfilError("Goal d_objective: lowercase letters, digits and - (at most 32 characters)")
            out["d_objective"] = v
        if z.get("force_rules"):
            out["force_rules"] = True
        return out

    @staticmethod
    def doc_key(label: str) -> Optional[str]:
        """Schlüssel der Profilzeile (``profile_json.rows``) zur Bezeichnung eines Werts im Vorschlag (``propose.slot_label``): ``--flag`` ->
        ``flag:--flag``, ``--extra-p --x`` -> ``extra:P:--x``, ``--env-d NAME`` -> ``env:D:NAME``, ``env NAME`` -> ``export:NAME``; ``None`` für
        eine Zeile, die keine Profilzeile ist (der P-Schnitt-Seed ``--pp-stage-ratio (Seed)``)."""
        lab = str(label)
        if lab.endswith(" (Seed)"):
            return None
        for pre, kind in (("--extra-p ", "extra:P:"), ("--extra-d ", "extra:D:"), ("--env-p ", "env:P:"), ("--env-d ", "env:D:")):
            if lab.startswith(pre):
                return kind + lab[len(pre):]
        if lab.startswith("env "):
            return "export:" + lab[4:]
        if lab.startswith("--"):
            return "flag:" + lab
        return None

    @staticmethod
    def _argv_tokens(argv) -> List[str]:
        """Die Tokens des argv, wie der Launcher sie sieht: ein Token mit Leerzeichen (``--extra-p=--flag a b``) wird mit ``shlex`` zerlegt, ``--flag=wert`` an
        der ersten ``=`` getrennt; so steht ein Flag an JEDER Stelle eines ``--extra-*=...``-Tokens als eigenes Token da (nicht nur am Ende)."""
        out: List[str] = []

        def eq(x: str) -> None:                       # ``--flag=wert`` -> ``--flag``, ``wert`` (auch ``--extra-p=--flag=wert``); ein Wert mit Leerzeichen bleibt EIN Token
            if x.startswith("--") and "=" in x:
                k, v = x.split("=", 1)
                out.append(k)
                eq(v)
            else:
                out.append(x)

        def add(t: str) -> None:
            if " " in t.strip():
                try:
                    parts = shlex.split(t)
                except ValueError:
                    parts = t.split()
                for x in parts:
                    eq(x)
            else:
                eq(t)

        for t in argv:
            add(str(t))
        return out

    @staticmethod
    def argv_has(launch, flag: str, value: str) -> bool:
        """Steht ``flag value`` im argv des Vorschlags (``launch = {argv, env}``): als zwei Tokens (``--pp-stage-ratio 31,17,16``) oder als ``--flag=value`` /
        ``--extra-p=... --flag value ...`` an jeder Position des Tokens (``ProfilEditor._argv_tokens``).  Nur der Vergleich der Zeichenketten, keine Rechnung."""
        toks = ProfilEditor._argv_tokens((launch or {}).get("argv") or [])
        return any(t == flag and toks[i + 1] == value for i, t in enumerate(toks[:-1]))

    def _startprofil(self, base: dict, vorschlag: dict, basis_name: str) -> tuple:
        """Das Startprofil ``flliper.server/1``: das Basisprofil mit den Werten, die der Vorschlag ändert (Herkunft ``planer``).  Gibt
        ``(doc, keys, nicht_uebernommen)``: ``keys`` = Label -> Profilschlüssel der geänderten Zeilen."""
        pj, _ref = self.mods()
        edits, keys, skipped = [], {}, []
        planner_only: Dict[str, str] = {}
        for w in vorschlag.get("werte") or []:
            if not w.get("geaendert"):
                continue
            key = self.doc_key(w["key"])
            if key is None:
                if w.get("wert") is not None:          # der P-Schnitt-Seed: kein Profilwert, aber die Rechnung des Planers (planner_only)
                    planner_only["flag:" + str(w["key"]).replace(" (Seed)", "")] = str(w["wert"])
                else:
                    skipped.append(w["key"])
                continue
            keys[w["key"]] = key
            if w.get("wert") is None:
                if w.get("alt") is not None:
                    edits.append({"key": key, "op": "delete"})
            else:
                edits.append({"key": key, "op": "set", "value": str(w["wert"])})
        new = pj.apply_edits(base, edits, self.specs())
        meta = new.setdefault("meta", {"origins": {}, "planner": {}, "notes": []})
        for e in edits:                                  # ein Wert des Planers ist nicht "vom Nutzer": Herkunft planer
            meta["origins"][e["key"]] = pj.ORIGIN_PLANER
            if e["op"] == "set":
                meta.setdefault("planner", {})[e["key"]] = e["value"]
        meta.setdefault("planner", {}).update(planner_only)
        new["name"] = ("%s-vorschlag" % basis_name)[:64]
        return new, keys, skipped

    def _single_base(self, name: str, model_path: str) -> dict:
        """Das leere Profil der Einzelkarte (``flliper.server/1`` ohne pdflip-Zeilen): nur Name, Linie und Modellpfad; die Argumente des normalen Servers setzt
        der Vorschlag hinein.  Kein Release-Profil als Grundlage: dessen Zeilen (``--pp-size``, ``--d-bs`` ...) gehören zum pdflip-Launcher, den die Einzelkarte nicht hat."""
        pj, _ref = self.mods()
        doc = {"schema": pj.SCHEMA, "name": ("%s" % name)[:64], "line": "einzel", "source": {"kind": "planer", "file": "", "sha256": "", "rc": 0},
               "vars": [{"name": "PROFILE_LINE", "value": "einzel"}, {"name": "PROFILE_MODEL", "value": model_path}, {"name": "PROFILE_NAME", "value": name}],
               "exports": [], "args": [], "form": [], "instr": [], "meta": {"origins": {}, "planner": {}, "notes": [], "caller_switches": []}}
        doc["id"] = pj.doc_id(doc)
        return doc

    def propose(self, body: dict) -> dict:
        """``POST /api/profil/propose``: Vorschlag (``pdflip/propose.propose``) + Orakel (Launcher-Trockenlauf) + Verdikte je Wert, als Startprofil
        ``flliper.server/1`` mit Herkunft, Verdikt und Kanten je Wert.

        Körper: ``basis`` {kind: release|user, name}, ``form`` flip|tp|dual|single (``einzel`` ist ein Name für ``single``), ``inventar`` ``"rig"`` (das
        Hardwareprofil dieses Rigs) oder eine Kartenliste ``[{card, pcie}]`` (Katalog, wie der Trockenlauf), ``ziele`` {seats, kv_tokens, kv_dtype, p_cut,
        d_objective, draft_kv_on_p, force_rules; nur single: host_ram_mib, pre_load_free_mib, reserve_mib, draft}, ``model_path`` / ``draft_path`` (sonst die
        des Basisprofils).  Läuft im Kindprozess (``self.oracle``), Cache je (Inventar, Form, Ziele, Profil-Hash, Modell); ein Fehler des Kindprozesses kommt
        als ``ok: false`` mit Grund, nie als Absturz.

        Die Form ``single`` (Einzelkarte, genau EINE Karte; ``karte`` = Ordinal im Hardwareprofil, Standard 0) hat keinen Launcher-Lauf: ihr Verdikt ist
        eine Planer-Rechnung (``ausgang`` passt | passt_nicht | unbelegt, ``art`` Planer-Rechnung) und ihr Startprofil ein neues Profil aus den Argumenten des
        normalen Servers (kein Basisprofil nötig; ``model_path`` oder das ``PROFILE_MODEL`` des Basisprofils)."""
        pj, _ref = self.mods()
        if isinstance(body, dict) and self.FORM_ALIAS.get(str(body.get("form") or ""), str(body.get("form") or "")) == "dual" and not self.dual_available():
            raise ProfilError(DUAL_FEHLT)              # NF-Linie: kein Dual-Vorschlag, kein ImportError im Kindprozess (HTTP 400)
        if self.oracle is None:
            raise ProfilError("The oracle is not configured (child process with the Python of the flliper environment: --couplings-python / RIGDASH_COUPLINGS_PYTHON)")
        if not isinstance(body, dict):
            raise ProfilError("Body must be a JSON object")
        basis = body.get("basis") or {}
        if not isinstance(basis, dict):
            raise ProfilError("basis must be {kind, name}")
        kind, name = str(basis.get("kind") or "release"), str(basis.get("name") or "")
        form = str(body.get("form") or "flip")
        form = self.FORM_ALIAS.get(form, form)
        if form not in self.FORMS:
            raise ProfilError("form must be flip, tp, dual or single (single card)")
        single = form == "single"
        ziele = self._ziele(body.get("ziele"), form)
        notes: List[str] = []
        if single and not name:
            doc, bas, bsha, comments = None, None, "", {}                 # die Einzelkarte braucht kein Profil: Modellpfad + Karte genügen
            name = "einzelkarte"
            kind = "keines"
        elif kind == "release":
            doc = self._import_release(name)
            path = self._release_path(name)
            bas = {"env_path": path}
            bsha = ORA.sha256_file(path) or ""
            comments = self._profile_comments(path)
        elif kind == "user":
            doc = self.read_user(name)
            text = pj.render_env(doc)
            bas = {"env_text": text, "source": name + ".json"}
            bsha = ORA.sha256_text(text)
            comments = {}
        else:
            raise ProfilError("basis.kind must be release or user")
        vars_ = {v["name"]: str(v.get("value", "")) for v in (doc or {}).get("vars") or [] if "name" in v}
        paths = {}
        for fld, var_name in (("model_path", "PROFILE_MODEL"), ("draft_path", "PROFILE_DRAFT")):
            p = body.get(fld)
            if p:
                if self.check_path is None:
                    raise ProfilError("%s: not allowed without a model root check (only the path of the base profile)" % fld)
                try:
                    p = self.check_path(p, fld)
                except ValueError as exc:
                    raise ProfilError(str(exc))
            else:
                # das Draft-Verzeichnis eines pdflip-Profils ist ein pdflip-Draft: die Einzelkarte nimmt nur einen ausdrücklich genannten
                p = "" if (single and fld == "draft_path") else (vars_.get(var_name) or "")
            if p:
                paths[fld] = str(p)
        if single and not paths.get("model_path"):
            raise ProfilError("Single card: give model_path (or choose a base profile with PROFILE_MODEL)")
        inv_req = body.get("inventar", "rig")
        if inv_req == "rig":
            hw = self._hardware_profile()
            if hw is None:
                raise ProfilError("Hardware profile of this rig not available: for a synthetic inventory give the card list [{card, pcie}]")
            inventar = {"hardware": hw}
            notes.append("Inventory: the NVML cards of this rig (hardware profile), real UUIDs.")
            if single:
                try:
                    emap = int(body.get("karte") or 0)
                except (TypeError, ValueError):
                    raise ProfilError("karte must be the ordinal of a card of the hardware profile")
                if not 0 <= emap < len(hw["cards"]):
                    raise ProfilError("karte %d: the hardware profile has %d cards" % (emap, len(hw["cards"])))
                inventar["karte"] = emap
                notes.append("Single card: card %d of the hardware profile (other card: karte=<ordinal>); the remaining cards of the rig are not taken into account." % emap)
        elif isinstance(inv_req, list):
            if single and len(inv_req) != 1:
                raise ProfilError("Single card: select exactly one card (not %d)" % len(inv_req))
            cards = self._build_cards(inv_req)
            inventar = self._inventar_for(cards, notes)
        else:
            raise ProfilError('inventar must be "rig" or a card list [{card, pcie}]')
        parts = {"inventar": self._inventar_key(inventar) + ([["karte", inventar.get("karte")]] if single and "karte" in inventar else []), "form": form, "ziele": ziele, "basis_sha256": bsha, "basis": [kind, name], "paths": paths,
                 "modell": [self._path_stamp(p) for p in paths.values()]}
        req = {"basis": bas, "inventar": inventar, "form": form, "ziele": ziele, "model_path": paths.get("model_path"), "draft_path": paths.get("draft_path")}
        res = self.oracle.ask("propose", req, parts)
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error") or "unknown error", "schema": self.PROPOSE_SCHEMA, "notes": notes + list(res.get("notizen") or [])}
        v = res["vorschlag"]
        if single:
            doc = self._single_base(name, paths["model_path"])
        new, keys, skipped = self._startprofil(doc, v, name)
        problems = self.verify_render(new, pj.render_env(new))
        new["meta"]["vorschlag"] = {"schema": self.PROPOSE_SCHEMA, "form": form, "n": v["n"], "ziele": v["ziele"], "basis": [kind, name], "basis_sha256": bsha,
                                    "ausgang": res["verdikt"].get("ausgang"), "argv_sha256": res["verdikt"].get("argv_sha256")}
        if single:
            new["meta"].setdefault("notes", []).append("Server profile of the single card (planner calculation, no pdflip launcher): the arguments of the normal server (python -m flliper.launch_server) for model %s" % paths["model_path"])
        else:
            new["meta"].setdefault("notes", []).append("Planner server profile from %s (%s) for %d cards, form %s" % (name, bsha[:12], v["n"], form))
        new["id"] = pj.doc_id(new)
        view = self.render_view(new, comments)
        by_key = {r["key"]: r for r in view["view"]["rows"]}
        entries = self.catalog()["entries"]
        werte = []
        je_wert = res.get("je_wert") or {}
        for w in v.get("werte") or []:
            dkey = keys.get(w["key"]) or self.doc_key(w["key"])
            row = by_key.get(dkey) if dkey else None
            ent = entries.get(dkey.split(":", 2)[-1].split("#")[0] if dkey else "") or entries.get(str(w["key"]).split()[-1]) or {}
            kanten = (row["explain"]["depends"] if row else [dict(d) for d in ent.get("depends", [])])
            wv = {"key": dkey, "label": w["key"], "wert": w.get("wert"), "alt": w.get("alt"), "zustand": w.get("zustand"), "herkunft": w.get("herkunft"),
                  "grund": w.get("grund"), "geaendert": bool(w.get("geaendert")), "in_argv": bool(w.get("in_argv")), "eintraege": w.get("eintraege"),
                  "verdikte": je_wert.get(w["key"], []), "kanten": kanten}
            if dkey is None and str(w["key"]).endswith(" (Seed)"):
                # der P-Schnitt-Seed ist keine Profilzeile: ``seed`` + der Wert, den das Profil unter diesem Namen schon hat (None = setzt es nicht), damit die
                # Seite weder "nicht gesetzt" sagt, wo das Profil den Wert setzt, noch einen Wert als Änderung zählt, der nicht im argv steht
                sname = str(w["key"])[: -len(" (Seed)")]
                pw = next((r["value"] for r in pj.rows(doc, self.specs()) if r.get("name") == sname and r.get("value") not in (None, "")), None)    # das BASISprofil, nicht das Startprofil (der Vorschlag kann die Zeile entfernen)
                wv["seed"] = True
                wv["profil_wert"] = None if pw is None else str(pw)
                # ``in_argv`` des Planers heisst "der Planer wendet den Seed an", nicht "der Wert steht im argv": im Dual (Profil-Modus) bleibt der Schnitt des Profils im
                # argv, und der Seed (GEMM-Rate) ist nur die Rechnung dahinter.  Fuer die Seite zaehlt, was der Launcher wirklich bekommt.
                wv["in_argv_planer"] = wv["in_argv"]
                wv["in_argv"] = bool(wv["in_argv"]) and w.get("wert") is not None and self.argv_has(res.get("launch"), sname, str(w["wert"]))
            werte.append(wv)
            if row is not None:
                row["vorschlag"] = {k: wv[k] for k in ("zustand", "herkunft", "grund", "verdikte", "kanten", "geaendert")}
        keep = ("schema", "form", "n", "cards", "inventory", "seeds", "fit", "ziele", "unbelegt", "hinweise", "blocker", "vektorlaengen", "vektoren_ok", "vektoren_falsch", "basis",
                "dual", "einzelkarte")
        vd = res["verdikt"]
        return {"ok": True, "schema": self.PROPOSE_SCHEMA, "form": form, "n": v["n"],
                "basis": {"kind": kind, "name": name, "sha256": bsha, "profil": vd.get("profil")},
                "startprofil": {"schema": pj.SCHEMA, "doc": new, "view": view["view"], "name": new["name"], "line": new.get("line"),
                                "verifiziert": not problems, "probleme": problems},
                "werte": werte, "verdikt": vd, "vorschlag": {k: v[k] for k in keep if k in v},
                "launch": res.get("launch"),
                "nicht_uebernommen": skipped,
                "balken": {"route": "/api/profil/recompute", "what": "phase_bars", "form": self.BALKEN_FORM[form], "doc": new},
                "orakel": {"cached": bool(res.get("cached")), "cache_key": res.get("cache_key"), "dauer_s": (vd.get("orakel") or {}).get("dauer_s"),
                           "version": (vd.get("orakel") or {}).get("version")},
                "notes": notes + [str(x) for x in res.get("notizen") or []]}

    @staticmethod
    def _path_stamp(path: str) -> list:
        """Stand eines Modellverzeichnisses für den Cache-Schlüssel (config.json: Größe, mtime; fehlt sie: nur der Pfad)."""
        try:
            st = os.stat(os.path.join(path, "config.json"))
            return [path, st.st_size, st.st_mtime_ns]
        except OSError:
            return [path, None, None]


def _is_vector(value: str, n: int) -> bool:
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != n or n < 2:
        return False
    try:
        [float(p) for p in parts]
    except ValueError:
        return False
    return True
