"""Auftrag 910 / Punkt 3: Release-Profile mit dem Werkzeug selbst nach fLLiper umsetzen (profconv/).

usage: profconv.py [--src /spinning/gpu-arb/docker/profiles_release] [--dst <kit>/profconv] [--check]

Die Umsetzung ist rename_to_flliper.rewrite_all(text, is_py=False, weg2=True, ident-map) -- dieselbe Funktion, mit der das
Kit den Baum umbenennt (gegengeprueft: die 28.09.-Stande 27b-base.env ergeben byte-gleich die alte profconv/27b-base.env).
Quelle sind die Release-Profile des Rigs (docker/profiles_release/), Ziel profconv/ unter dem Namen, den dry.sh als
<profil>.env nimmt:

  27b-base.env          -> 27b-base.env        (INT8-Basisform, von 27b*.env und dual per `source` genommen)
  27b.env               -> 27b.env             (INT8, Release-Form)
  27b-nvfp4-dual.env    -> dual.env            (Dual NVFP4)  [+ Alias 27b-nvfp4-dual.env]
  27b.env               -> int8.env            (Alias: Linienname des Kits fuer INT8 = 27b)
  nf.env                -> nf.env ; nf-int4.env -> nf-int4.env

Eine vorhandene, abweichende Zieldatei wird vorher als <datei>.bak_910 gesichert (nie ueberschrieben, wenn .bak_910 schon
da ist). --check schreibt nichts und meldet nur, ob profconv/ dem Stand der Quelle entspricht (Exit 1 bei Abweichung).
Die `source "$(dirname ...)/27b-base.env"`-Zeilen bleiben gueltig, weil die Basis im selben Verzeichnis liegt."""
import argparse, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rename_to_flliper as R

KIT = os.path.dirname(os.path.abspath(__file__))
# Quelle -> Ziele
PLAN = [
    ("27b-base.env", ["27b-base.env"]),
    ("27b.env", ["27b.env", "int8.env"]),
    ("27b-nvfp4-dual.env", ["27b-nvfp4-dual.env", "dual.env"]),
    ("nf.env", ["nf.env"]),
    ("nf-int4.env", ["nf-int4.env"]),
]
# F0-A (07.10.): the NF line is driven in its abliterated form; its profile is not a release profile but lives in the rig profile dir
# (--src2). Same conversion function, same target dir.
PLAN2 = [
    ("nf-int4-h6-abl.env", ["nf-int4-h6-abl.env"]),
]


# data files the profiles read next to themselves (symlinks in profiles_release/): copied byte for byte (measured data, no names to convert)
AUX = ["27b-nvfp4.graphcal.json", "27b-nvfp4.pchunk.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/spinning/gpu-arb/docker/profiles_release")
    ap.add_argument("--src2", default="/spinning/gpu-arb/docker/profiles", help="dir of the PLAN2 profiles (non-release rig profiles)")
    ap.add_argument("--dst", default=os.path.join(KIT, "profconv"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    imap = R._load_imap(os.path.join(KIT, "data", "merged_0928.json"))
    bad = 0
    jobs = [(os.path.join(a.src, s), ds) for s, ds in PLAN] + [(os.path.join(a.src2, s), ds) for s, ds in PLAN2]
    for srcpath, dsts in jobs:
        text = open(srcpath).read()
        new = R.rewrite_all(text, False, True, imap)[0]
        for d in dsts:
            p = os.path.join(a.dst, d)
            old = open(p).read() if os.path.exists(p) else None
            if old == new:
                print("same   ", d)
                continue
            if a.check:
                print("DIFFERS", d, "(missing)" if old is None else "")
                bad += 1
                continue
            if old is not None and not os.path.exists(p + ".bak_910"):
                shutil.copy2(p, p + ".bak_910")
            open(p, "w").write(new)
            print("written", d, "(was missing)" if old is None else "(backup .bak_910)")
    for f in AUX:
        data = open(os.path.join(a.src, f), "rb").read()      # follows the symlink
        p = os.path.join(a.dst, f)
        old = open(p, "rb").read() if os.path.exists(p) else None
        if old == data:
            print("same   ", f); continue
        if a.check:
            print("DIFFERS", f); bad += 1; continue
        open(p, "wb").write(data); print("written", f)
    sys.exit(1 if bad else 0)


main()
