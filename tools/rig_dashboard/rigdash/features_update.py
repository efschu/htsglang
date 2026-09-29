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
import sys
import time

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
    probs = features.validate(d["features"])
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
    a = ap.parse_args(argv)

    if a.cmd in ("show", "check"):
        d = _load(a.file)
        if a.cmd == "check":
            probs = features.validate(d["features"])
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
        _save(a.file, d)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    print("geschrieben: %s (%s %s)" % (a.file, a.cmd, a.id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
