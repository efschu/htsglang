"""Keep /spinning/gpu-arb/docs/features.json current -- the one writer next to hand edits.

Every builder / operator who finishes, picks, switches or measures a feature runs
this (user order 29.09.: "das muss immer aktuell gehalten werden"). The file is
validated before it is replaced (tmp + rename under a lock), so the dashboard
never reads half a file. "im Image" and "aktiv" are NOT written here -- rigdash
computes them from git and the boot's state.json.

  # a feature (upsert by id; lists given again replace the old ones)
  python3 /opt/rigdash/current/rigdash/features_update.py set --id H106 --modell NF \\
      --titel "Form-A-Worker folgt dem ADMIT des Hosts" --fertig ja \\
      --zweig desk/nf-h106-0928=49a2a04af3 \\
      --schalter SGLANG_WEG2_H106=env:D:1:an --verantwortlich NF-Implementierer
  # 27B picked it: add its sha as a further branch (keeps the others)
  ... add-zweig --id H106 --zweig desk/27b-unified-0926=0293076975
  # a gain -- modell is mandatory when the feature is for both models
  ... gewinn --id H106 --modell NF --metrik Flipzeit --vorher 3.1 --nachher 2.4 --einheit s \\
      --art gemessen --quelle "fliptimes rc12z30p" --boot nfh91...-f700
  ... begruendung --id H63 --text "aus bis Faltung am Metall belegt (H63d)"
  ... show [--id H106]      ... check
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import time
import urllib.request

if __package__:
    from . import features
else:                                   # run as a file: /opt/rigdash/current/rigdash/features_update.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rigdash import features        # noqa: E402


def _load(path: str) -> dict:
    try:
        with open(path) as fh:
            d = json.load(fh)
    except FileNotFoundError:
        return {"schema": "rigdash.features/1", "features": []}
    if not isinstance(d, dict) or not isinstance(d.get("features"), list):
        raise SystemExit("%s: kein Objekt mit Array 'features'" % path)
    return d


def _save(path: str, d: dict):
    probs = features.validate_doc(d)
    if probs:
        raise SystemExit("nicht geschrieben, Datei waere ungueltig:\n  " + "\n  ".join(probs))
    d["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


def _find(d: dict, fid: str, create: bool = False) -> dict:
    for f in d["features"]:
        if f.get("id") == fid:
            return f
    if not create:
        raise SystemExit("kein Feature mit id %s" % fid)
    f = {"id": fid}
    d["features"].append(f)
    return f


def _zweig(s: str) -> dict:
    b, sep, sha = s.rpartition("=")
    if not sep or not b:
        raise SystemExit("--zweig braucht branch=sha, nicht %r" % s)
    return {"branch": b, "sha": sha}


def _schalter(s: str) -> dict:
    """NAME=art:gruppe:an_wert:default -- e.g. SGLANG_X=env:D:1:aus, --d-foo=flag:D::aus."""
    name, sep, rest = s.partition("=")
    parts = rest.split(":") if sep else []
    if len(parts) != 4:
        raise SystemExit("--schalter braucht NAME=art:gruppe:an_wert:default, nicht %r" % s)
    art, gruppe, an_wert, default = parts
    return {"name": name, "art": art, "gruppe": gruppe, "an_wert": an_wert, "default": default}


def _bool(s: str) -> bool:
    if s.lower() in ("ja", "j", "1", "true", "yes"):
        return True
    if s.lower() in ("nein", "n", "0", "false", "no"):
        return False
    raise SystemExit("ja|nein erwartet, nicht %r" % s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=features.DEFAULT_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set", help="Feature anlegen oder aendern (upsert nach id)")
    s.add_argument("--id", required=True)
    s.add_argument("--modell", choices=features.MODELL_VALUES)
    s.add_argument("--titel")
    s.add_argument("--fertig", type=_bool)
    s.add_argument("--zweig", action="append", type=_zweig, help="branch=sha (wiederholbar; ersetzt die Liste)")
    s.add_argument("--schalter", action="append", type=_schalter,
                   help="NAME=art:gruppe:an_wert:default (wiederholbar; ersetzt die Liste)")
    s.add_argument("--aus-begruendung")
    s.add_argument("--verantwortlich")
    s.add_argument("--produkt", action="append", help="als Baustein an dieses Produkt-Feature hängen (F1..F24, wiederholbar)")
    z = sub.add_parser("add-zweig", help="weiteren Zweig/Pick anhaengen (z. B. 27B-Pick unter neuem sha)")
    z.add_argument("--id", required=True)
    z.add_argument("--zweig", required=True, type=_zweig)
    g = sub.add_parser("gewinn", help="Gewinn anhaengen (gleiche metrik+modell ersetzt den alten)")
    g.add_argument("--id", required=True)
    g.add_argument("--metrik", required=True, help="z. B. Decode tok/s, Flipzeit, KV Token, bs parallel")
    g.add_argument("--vorher")
    g.add_argument("--nachher", required=True)
    g.add_argument("--einheit", default="")
    g.add_argument("--art", required=True, choices=features.GAIN_ART)
    g.add_argument("--modell", choices=("27B", "NF"), help="Pflicht bei modell=beide")
    g.add_argument("--quelle", default="")
    g.add_argument("--boot", default="")
    b = sub.add_parser("begruendung", help="warum es im Image aus ist")
    b.add_argument("--id", required=True)
    b.add_argument("--text", required=True)
    r = sub.add_parser("rm", help="Feature entfernen")
    r.add_argument("--id", required=True)
    sh = sub.add_parser("show")
    sh.add_argument("--id")
    sub.add_parser("check")
    # Produkt-Features (Nutzer 29.09.): Soll/Ist je Modell, Commits nur als Bausteine
    ps = sub.add_parser("produkt-set", help="Produkt-Feature anlegen/aendern (Soll, Bausteine)")
    ps.add_argument("--id", required=True)
    ps.add_argument("--nr", type=int)
    ps.add_argument("--titel")
    ps.add_argument("--soll", help="ein Satz, messbar")
    ps.add_argument("--bausteine", help="Baustein-ids (features[].id), kommagetrennt; ersetzt die Liste")
    pi = sub.add_parser("produkt-ist", help="Ist eines Modells setzen -- nur mit Beleg, sonst 'unbelegt'")
    pi.add_argument("--id", required=True)
    pi.add_argument("--modell", required=True, choices=features.MODELS)
    pi.add_argument("--status", required=True, choices=features.PRODUKT_STATUS)
    pi.add_argument("--wert", default="")
    pi.add_argument("--grund", default="", help="warum aus / offen")
    pi.add_argument("--beleg", default="", help="Boot, Zahl, Datei")
    pi.add_argument("--belegt-am", default=None,
                    help="wann der Beleg entstand (ISO, z. B. 2026-09-29T07:10Z); Default jetzt. Älter als der "
                         "letzte Boot des Modells = 'Ist veraltet, neu messen'")
    kz = sub.add_parser("kreuz", help="Zelle der Kreuztabelle (Feature F2) setzen")
    kz.add_argument("--id", default="F2")
    kz.add_argument("--modell", required=True, choices=features.MODELS)
    kz.add_argument("--a", required=True, choices=[k for k, _ in features.KREUZ_ACHSEN])
    kz.add_argument("--b", required=True, choices=[k for k, _ in features.KREUZ_ACHSEN])
    kz.add_argument("--status", required=True, choices=features.KREUZ_STATUS)
    kz.add_argument("--note", default="")
    fi = sub.add_parser("zeile-ist", help="Ist einer Unterzeile setzen (F12 Format, F23 Prompt-Länge)")
    fi.add_argument("--id", required=True)
    fi.add_argument("--zeile", required=True, help="Name der Unterzeile, z. B. NVFP4")
    fi.add_argument("--modell", required=True, choices=features.MODELS)
    fi.add_argument("--status", choices=features.PRODUKT_STATUS)
    fi.add_argument("--wert", default="")
    fi.add_argument("--beleg", default="")
    fi.add_argument("--belegt-am", default=None, help="ISO; Default jetzt")
    mx = sub.add_parser("matrix", help="Zelle der Decode-Matrix (F24): Wert oder 'ungültig'; keine Zelle = ungemessen")
    mx.add_argument("--id", default="F24")
    mx.add_argument("--modell", required=True, choices=features.MODELS)
    mx.add_argument("--form", required=True)
    mx.add_argument("--bs", required=True, choices=features.MATRIX_BS)
    mx.add_argument("--tiefe", required=True, choices=features.MATRIX_TIEFE)
    mx.add_argument("--text", required=True, choices=features.MATRIX_TEXT)
    mx.add_argument("--wert", default="", help="mit Einheit, z. B. '24,7 ms' oder '131,9 tok/s'")
    mx.add_argument("--ungueltig", action="store_true", help="Zelle ungültig (z. B. EOS unter 500 Tokens)")
    mx.add_argument("--boot", default="")
    mx.add_argument("--beleg", required=True)
    im = sub.add_parser("import-27b", help="27B-Ist, Kreuztabelle, P1/P2 und Marker aus der 27B-Datei übernehmen")
    im.add_argument("--md", default="/spinning/gpu-arb/docs/features_27b_ist_0929.md")
    bo = sub.add_parser("boot-override", help="Lebenszyklus eines Boots korrigiert anzeigen (state.json bleibt)")
    bo.add_argument("--boot", required=True, help="boot_id aus state.json")
    bo.add_argument("--lifecycle", required=True)
    bo.add_argument("--beleg", required=True)
    md = sub.add_parser("md", help="Produkt-Tabelle als Markdown (mit berechnetem im Image/aktiv der Bausteine)")
    md.add_argument("--out", help="Datei, sonst stdout")
    md.add_argument("--live-url", default="http://127.0.0.1:8890/api/live",
                    help="laufendes rigdash für 'Wert im aktuellen Boot' ('' = ohne)")
    a = ap.parse_args(argv)

    if a.cmd == "md":
        view = features.Features(a.file, background=False).view()
        boots, gpus = [], None
        if a.live_url:
            try:        # the running dashboard's log view: current values need the live boots
                with urllib.request.urlopen(a.live_url, timeout=10) as r:
                    lv = json.load(r)
                boots, gpus = lv.get("boots") or [], lv.get("gpus")
            except (OSError, ValueError) as e:
                print("Warnung: %s nicht lesbar (%s) -- nur state.json-Werte" % (a.live_url, e), file=sys.stderr)
        text = markdown(features.attach_current(view, boots, gpus))
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(text)
            print("geschrieben: %s" % a.out)
        else:
            print(text)
        return 0

    if a.cmd in ("show", "check"):
        d = _load(a.file)
        if a.cmd == "check":
            probs = features.validate_doc(d)
            print("\n".join(probs) if probs else "ok: %d Features" % len(d["features"]))
            return 1 if probs else 0
        out = [_find(d, a.id)] if a.id else d["features"]
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0

    lock = open(a.file + ".lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        d = _load(a.file)
        if a.cmd == "set":
            f = _find(d, a.id, create=True)
            for key, val in (("modell", a.modell), ("titel", a.titel), ("fertig", a.fertig),
                             ("zweige", a.zweig), ("schalter", a.schalter),
                             ("aus_begruendung", a.aus_begruendung), ("verantwortlich", a.verantwortlich)):
                if val is not None:
                    f[key] = val
            f.setdefault("zweige", [])
            f.setdefault("schalter", [])
            f.setdefault("gewinn", [])
            f.setdefault("fertig", False)
            for pid in a.produkt or []:
                bs = _find_produkt(d, pid).setdefault("bausteine", [])
                if a.id not in bs:
                    bs.append(a.id)
        elif a.cmd == "add-zweig":
            f = _find(d, a.id)
            zs = [x for x in f.get("zweige") or [] if x.get("sha") != a.zweig["sha"]]
            f["zweige"] = zs + [a.zweig]
        elif a.cmd == "gewinn":
            f = _find(d, a.id)
            new = {k: v for k, v in (("metrik", a.metrik), ("vorher", a.vorher), ("nachher", a.nachher),
                                    ("einheit", a.einheit), ("art", a.art), ("modell", a.modell),
                                    ("quelle", a.quelle), ("boot", a.boot)) if v not in (None, "")}
            keep = [x for x in f.get("gewinn") or []
                    if not (x.get("metrik") == a.metrik and x.get("modell") == a.modell)]
            f["gewinn"] = keep + [new]
        elif a.cmd == "begruendung":
            _find(d, a.id)["aus_begruendung"] = a.text
        elif a.cmd == "rm":
            f = _find(d, a.id)
            d["features"].remove(f)
        elif a.cmd == "produkt-set":
            p = _find_produkt(d, a.id, create=True)
            for key, val in (("nr", a.nr), ("titel", a.titel), ("soll", a.soll)):
                if val is not None:
                    p[key] = val
            if a.bausteine is not None:
                p["bausteine"] = [x.strip() for x in a.bausteine.split(",") if x.strip()]
        elif a.cmd == "produkt-ist":
            p = _find_produkt(d, a.id)
            p.setdefault("ist", {})[a.modell] = {k: v for k, v in (
                ("status", a.status), ("wert", a.wert), ("grund", a.grund), ("beleg", a.beleg),
                ("belegt_am", _belegt_am(a.belegt_am))) if v}
        elif a.cmd == "kreuz":
            p = _find_produkt(d, a.id)
            cells = p.setdefault("kreuztabelle", {}).setdefault("zellen", {}).setdefault(a.modell, {})
            cells[features.kreuz_key(a.a, a.b)] = {k: v for k, v in (("status", a.status), ("note", a.note)) if v}
        elif a.cmd == "zeile-ist":
            p = _find_produkt(d, a.id)
            rows = p.setdefault("untertabelle", {}).setdefault("zeilen", [])
            row = next((r for r in rows if r.get("name") == a.zeile), None)
            if row is None:
                row = {"name": a.zeile, "ist": {}}
                rows.append(row)
            row.setdefault("ist", {})[a.modell] = {k: v for k, v in (
                ("status", a.status), ("wert", a.wert), ("beleg", a.beleg),
                ("belegt_am", _belegt_am(a.belegt_am))) if v}
        elif a.cmd == "matrix":
            p = _find_produkt(d, a.id)
            cells = p.setdefault("matrix", {}).setdefault("zellen", {}).setdefault(a.modell, {})
            cells[features.matrix_key(a.form, a.bs, a.tiefe, a.text)] = {k: v for k, v in (
                ("status", "ungültig" if a.ungueltig else "wert"), ("wert", a.wert), ("boot", a.boot),
                ("beleg", a.beleg)) if v}
        elif a.cmd == "import-27b":
            import_27b(d, a.md)
        elif a.cmd == "boot-override":
            d.setdefault("boot_overrides", {})[a.boot] = {"lifecycle": a.lifecycle, "beleg": a.beleg}
        _save(a.file, d)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    print("geschrieben: %s (%s %s)" % (a.file, a.cmd, getattr(a, "id", None) or getattr(a, "boot", "")))
    return 0


def _belegt_am(s) -> str:
    if s is None:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if features.belegt_ts(s) is None:
        raise SystemExit("--belegt-am braucht ISO (2026-09-29T07:10Z oder 2026-09-29), nicht %r" % s)
    return s


def _find_produkt(d: dict, pid: str, create: bool = False) -> dict:
    for p in d.setdefault("produkt", []):
        if p.get("id") == pid:
            return p
    if not create:
        raise SystemExit("kein Produkt-Feature mit id %s" % pid)
    p = {"id": pid, "ist": {}, "bausteine": []}
    d["produkt"].append(p)
    return p


# ------------------------------------------------------------------ 27B-Import (27B-Sitz, 29.09.)

QUELLE_27B = "27B-Sitz, features_27b_ist_0929.md"


def _md_rows(lines):
    """Rows of the markdown table that starts at lines[0] (header), as lists of stripped cells."""
    out = []
    for ln in lines:
        if not ln.lstrip().startswith("|"):
            break
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        out.append(cells)
    return out


def _tables(text: str) -> list:
    """(heading, rows) for every table; heading = the last '#'/bold line before it."""
    lines, out, head, i = text.splitlines(), [], "", 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#") or (ln.startswith("**") and ln.rstrip().endswith(":**")):
            head = ln.strip("#* :")
        if ln.lstrip().startswith("|"):
            rows = _md_rows(lines[i:])
            out.append((head, rows))
            i += len([x for x in lines[i:] if x.lstrip().startswith("|")][:len(rows) + 1])
            continue
        i += 1
    return out


def _plain(s: str) -> str:
    return re.sub(r"\*\*|`", "", s or "").strip()


def map_status(text: str) -> str:
    """The 27B file's free status text onto the fixed status set; the text itself is kept beside.
    The earliest keyword wins ("INT8 fertig+aktiv; NVFP4 im Image aus" is fertig+aktiv)."""
    t = _plain(text).lower()
    hits = []
    for pat, st in ((r"entfällt", "entfällt"), (r"^offen", "offen"), (r"fertig\+aktiv", "fertig+aktiv"),
                    (r"^phase 1 aktiv", "fertig+aktiv"), (r"im image,? aber aus|im image aus", "im Image aber aus"),
                    (r"\bdesk\b", "Desk")):
        m = re.search(pat, t)
        if m:
            hits.append((m.start(), st))
    return min(hits)[1] if hits else "unbelegt"


def _stand(text: str, path: str) -> str:
    """'Stand 29.09.2026 ~07:10Z' in the file's title, else its mtime -- the rows' 'zuletzt belegt'."""
    m = re.search(r"Stand (\d\d)\.(\d\d)\.(\d{4}) ~?(\d\d):(\d\d)Z", text)
    if m:
        return "%s-%s-%sT%s:%sZ" % (m.group(3), m.group(2), m.group(1), m.group(4), m.group(5))
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(path)))


def import_27b(d: dict, path: str) -> None:
    text = open(path).read()
    stand = _stand(text, path)
    prod = {p["id"]: p for p in d.setdefault("produkt", [])}
    achsen = [k for k, _ in features.KREUZ_ACHSEN]
    for head, rows in _tables(text):
        hdr = [c.lower() for c in rows[0]] if rows else []
        if hdr[:5] == ["f", "feature", "status", "ist-wert", "beleg / grund"]:
            for r in rows[1:]:
                p = prod.get("F%s" % r[0].strip())
                if p is None:
                    continue
                p.setdefault("ist", {})["27B"] = {k: v for k, v in (
                    ("status", map_status(r[2])), ("status_text", _plain(r[2])), ("wert", _plain(r[3])),
                    ("beleg", _plain(r[4])), ("quelle", QUELLE_27B), ("belegt_am", stand)) if v and v != "–"}
        elif hdr[:1] == [""] and hdr[1:] == achsen:
            cells = prod["F2"].setdefault("kreuztabelle", {}).setdefault("zellen", {})
            cells["27B"] = {}
            for r in rows[1:]:
                a = _plain(r[0])
                for b, c in zip(achsen, r[1:]):
                    c = _plain(c)
                    if not c:
                        continue
                    st = next((s for s in features.KREUZ_STATUS if c.startswith(s)), None)
                    note = c[len(st):].strip(" ()") if st else c
                    cells["27B"][features.kreuz_key(a, b)] = {k: v for k, v in (
                        ("status", st or "unbelegt"), ("note", note), ("quelle", QUELLE_27B)) if v}
        elif hdr[:2] == ["format / form", "instrument"]:
            p = prod["F23"]
            zeilen = p.setdefault("untertabelle", {"spalte": "Prompt-Länge / Form", "zeilen": []}).setdefault("zeilen", [])
            zeilen[:] = [z for z in zeilen if not z.get("name", "").startswith("27B ")]
            for r in rows[1:]:
                werte = "; ".join("%s: %s" % (h, _plain(v)) for h, v in zip(rows[0][2:-1], r[2:-1]) if _plain(v))
                zeilen.append({"name": "27B " + _plain(r[0]), "ist": {"27B": {
                    "wert": "%s [Instrument %s]" % (werte, _plain(r[1])), "beleg": _plain(r[-1]), "quelle": QUELLE_27B,
                    "belegt_am": stand}}})
        elif hdr[:3] == ["form", "tiefe", "text"]:
            m = re.search(r"(\w+) D-only \(([^)]*)\)", head)
            fmt, beleg = (m.group(1), m.group(2)) if m else ("", head)
            cells = prod["F24"].setdefault("matrix", {}).setdefault("zellen", {}).setdefault("27B", {})
            for r in rows[1:]:
                form = ("%s %s" % (fmt, _plain(r[0]))).strip()
                tiefe, txt = _plain(r[1]).lower(), _plain(r[2]).lower()
                for bs, c in zip(features.MATRIX_BS, r[3:9]):
                    c = _plain(c)
                    if not c or c == "ungemessen":
                        continue
                    k = features.matrix_key(form, bs, tiefe, txt)
                    cells[k] = {"status": "ungültig", "beleg": beleg, "quelle": QUELLE_27B} if c == "ungültig" else \
                        {"status": "wert", "wert": c + " ms", "beleg": beleg, "quelle": QUELLE_27B}
        elif hdr[:2] == ["f", "marker (quelle des werts)"]:
            fmts = rows[0][2:]
            for r in rows[1:]:
                key = _plain(r[0]).split()[0]
                pid = {"P1": "F23", "P2": "F24"}.get(key, "F" + key)
                if pid in prod:
                    prod[pid].setdefault("marker", {})["27B"] = {
                        "marker": _plain(r[1]), "je_format": {f: _plain(v) for f, v in zip(fmts, r[2:])},
                        "quelle": QUELLE_27B}
    m = re.search(r"Gemessene Runde \(gpu-ms\):([^\n]*)", text)
    if m and "F24" in prod:
        cells = prod["F24"].setdefault("matrix", {}).setdefault("zellen", {}).setdefault("27B", {})
        for bs, ms, n in re.findall(r"bs(\d) ([0-9,]+) \(n=(\d+)\)", m.group(1)):
            cells[features.matrix_key("INT8 uneven DCP D TP3", bs, "gemischt", "gemischt")] = {
                "status": "wert", "wert": "%s ms (n=%s)" % (ms, n), "boot": "z30j",
                "beleg": "z30j bar1 Agentenlast, Randwert Tiefe gemischt", "quelle": QUELLE_27B}


def _cell(x) -> str:
    return str(x or "").replace("|", "/").replace("\n", " ")


def _aktuell(p: dict, m: str) -> str:
    a = (p.get("aktuell") or {}).get(m) or {}
    if a.get("wert"):
        txt = "%s [%s]" % (a["wert"], a.get("instrument", ""))
    elif a.get("leer"):
        txt = a["leer"]
    elif a.get("kein_instrument"):
        txt = "kein Instrument: " + a["kein_instrument"]
    else:
        return "—"
    jf = a.get("je_format") or {}
    run = [f for f, v in jf.items() if v == "dieser Boot"]
    rest = [f for f in jf if f not in run]
    if run:
        txt += " (Format %s%s)" % ("/".join(run), "; %s: kein Boot in diesem Format" % "/".join(rest) if rest else "")
    elif jf:
        txt += " (Format des Boots unbekannt)"
    return txt


def markdown(view: dict) -> str:
    """The product table for humans (FEATURES-SOLL-IST-*.md), from the same view the dashboard shows."""
    boots = {m["model"]: m.get("boot") or {} for m in view.get("models") or []}
    out = ["# Features Soll/Ist je Modell", "",
           "Erzeugt mit `features_update.py md` aus %s am %s UTC. Ist-Werte nur mit Beleg, sonst „unbelegt“; "
           "27B-Werte vom 27B-Sitz (features_27b_ist_0929.md, per `import-27b`). Bausteine = Commits/Fixes, die das "
           "Feature tragen; *im Image* / *aktiv* rechnet rigdash gegen den laufenden oder letzten Boot je Modell. "
           "„Wert im aktuellen Boot“ zieht rigdash aus dem Boot (live.py/state.json), sonst steht der fehlende Marker da."
           % (view.get("path"), time.strftime("%Y-%m-%d %H:%M", time.gmtime())), ""]
    for m in features.MODELS:
        b = boots.get(m) or {}
        out.append("- %s-Boot: %s, REV %s, %s%s" % (m, b.get("rc") or b.get("boot_id") or "—", b.get("rev") or "—",
                                                  b.get("lifecycle") or "—",
                                                  " (Override: %s)" % b["override_beleg"] if b.get("override_beleg") else ""))
    out += ["", "| # | Feature | Soll | Ist 27B | Ist NF | Status 27B / NF | Beleg | Wert im aktuellen Boot 27B / NF | Bausteine |",
            "|---|---|---|---|---|---|---|---|---|"]
    for p in view.get("produkt") or []:
        ist = p["ist"]

        def val(m):
            x = ist[m]
            v = x.get("wert") or ("unbelegt" if x["status"] == "unbelegt" else "—")
            if x.get("veraltet"):
                v += " — **Ist veraltet, neu messen** (belegt %s, älter als der letzte Boot)" % x.get("belegt_am")
            return _cell(v)

        def st(m):
            x = ist[m]
            txt = x.get("status_text") if x.get("status_text") and x["status_text"] != x["status"] else ""
            return _cell(x["status"] + (" (%s)" % (x.get("grund") or txt) if (x.get("grund") or txt) else ""))
        beleg = "; ".join("%s: %s%s" % (m, ist[m]["beleg"], " (%s)" % ist[m]["belegt_am"] if ist[m].get("belegt_am") else "")
                          for m in features.MODELS if ist[m].get("beleg"))
        bs = []
        for r in p["bausteine"]:
            nf = (r["je_modell"] or {}).get("NF") or next(iter((r["je_modell"] or {}).values()), None)
            bs.append("%s%s" % (r["id"], " [%s/%s]" % (nf["im_image"]["state"], nf["aktiv"]) if nf else ""))
        out.append("| %s | %s | %s | %s | %s | 27B: %s / NF: %s | %s | 27B: %s / NF: %s | %s |" % (
            p.get("nr") or "", _cell(p["titel"]), _cell(p["soll"]), val("27B"), val("NF"), st("27B"), st("NF"),
            _cell(beleg) or "—", _cell(_aktuell(p, "27B")), _cell(_aktuell(p, "NF")), _cell(", ".join(bs)) or "—"))
    mk = [(p, m, x) for p in view.get("produkt") or [] for m, x in (p.get("marker") or {}).items()]
    if mk:
        fmts = []
        for _, _, x in mk:
            fmts += [f for f in x.get("je_format") or {} if f not in fmts]
        out += ["", "## Instrument-Marker je Feature und Format (Quelle des Werts im aktuellen Boot)", "",
                "| Feature | Modell | Marker | " + " | ".join(fmts) + " | Quelle |",
                "|---|---|---|" + "---|" * len(fmts) + "---|"]
        for p, m, x in mk:
            jf = x.get("je_format") or {}
            out.append("| %s %s | %s | %s | %s | %s |" % (p["id"], _cell(p["titel"]), m, _cell(x.get("marker")),
                                                         " | ".join(_cell(jf.get(f)) or "—" for f in fmts),
                                                         _cell(x.get("quelle"))))
    for p in view.get("produkt") or []:
        kt = p.get("kreuztabelle")
        if kt:
            achsen = view.get("kreuz_achsen") or []
            for m in features.MODELS:
                cells = (kt.get("zellen") or {}).get(m) or {}
                out += ["", "## %s Kreuztabelle %s" % (p["id"], m), "",
                        "| | " + " | ".join(x["name"] for x in achsen) + " |",
                        "|---|" + "---|" * len(achsen)]
                for i, a in enumerate(achsen):
                    row = []
                    for j, b in enumerate(achsen):
                        if j < i:
                            row.append("")
                            continue
                        c = cells.get(features.kreuz_key(a["key"], b["key"])) or {}
                        row.append(_cell(c.get("status", "unbelegt") + (" — " + c["note"] if c.get("note") else "")))
                    out.append("| %s | %s |" % (a["name"], " | ".join(row)))
        ut = p.get("untertabelle")
        if ut and ut.get("zeilen"):
            out += ["", "## %s %s" % (p["id"], p["titel"]), "",
                    "| %s | Soll | Ist 27B | Ist NF | im aktuellen Boot |" % (ut.get("spalte") or "Zeile"),
                    "|---|---|---|---|---|"]
            for z in ut["zeilen"]:
                def zc(m):
                    x = (z.get("ist") or {}).get(m) or {}
                    if not x:
                        return "—"
                    return _cell("%s%s%s" % (x.get("status", ""), (": " if x.get("status") else "") + x["wert"] if x.get("wert") else "",
                                             (" [" + x["beleg"] + "]") if x.get("beleg") else "")) or "—"
                akt = "; ".join("%s: %s" % (m, v) for m, v in (z.get("aktuell") or {}).items())
                out.append("| %s | %s | %s | %s | %s |" % (_cell(z.get("name")), _cell(z.get("soll")) or "—",
                                                          zc("27B"), zc("NF"), _cell(akt) or "—"))
        mx = p.get("matrix")
        if mx:
            out += matrix_markdown(p, mx)
    if view.get("problems"):
        out += ["", "## Probleme", ""] + ["- " + _cell(x) for x in view["problems"]]
    return "\n".join(out) + "\n"


def matrix_markdown(p: dict, mx: dict) -> list:
    """One table per model and form: rows Tiefe x Text that carry at least one cell, columns bs 1..6.
    A missing cell is written "ungemessen", never interpolated."""
    out = []
    for m in features.MODELS:
        cells = (mx.get("zellen") or {}).get(m) or {}
        forms = []
        for k in cells:
            f = k.split("|")[0]
            if f not in forms:
                forms.append(f)
        out += ["", "## %s Decode-Matrix %s" % (p["id"], m)]
        if not forms:
            out += ["", "alle Zellen ungemessen"]
            continue
        for f in forms:
            rows = []
            for t in features.MATRIX_TIEFE:
                for x in features.MATRIX_TEXT:
                    if any(features.matrix_key(f, bs, t, x) in cells for bs in features.MATRIX_BS):
                        rows.append((t, x))
            out += ["", "**%s** (übrige Tiefe/Text-Zeilen: ungemessen)" % _cell(f), "",
                    "| Tiefe | Text | " + " | ".join("bs%s" % b for b in features.MATRIX_BS) + " | Beleg |",
                    "|---|---|" + "---|" * len(features.MATRIX_BS) + "---|"]
            for t, x in rows:
                vals, belege = [], []
                for bs in features.MATRIX_BS:
                    c = cells.get(features.matrix_key(f, bs, t, x))
                    if not c:
                        vals.append("ungemessen")
                        continue
                    vals.append("ungültig" if c.get("status") == "ungültig" else c.get("wert", "?"))
                    if c.get("beleg") and c["beleg"] not in belege:
                        belege.append(c["beleg"])
                out.append("| %s | %s | %s | %s |" % (t, x, " | ".join(_cell(v) for v in vals), _cell("; ".join(belege))))
    return out


if __name__ == "__main__":
    sys.exit(main())
