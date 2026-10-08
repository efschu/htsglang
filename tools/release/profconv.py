"""Auftrag 910 / Punkt 3: Release-Profile mit dem Werkzeug selbst nach fLLiper umsetzen (profconv/).

usage: profconv.py [--src /spinning/gpu-arb/docker/profiles_release] [--dst <kit>/profconv] [--check]
       profconv.py --tree-out <tree>/docker/flliper [--check]       (F0-G: the release set INTO the tree, with *.env.alt)
       profconv.py --list-live [--src ...] [--src2 ...]            (F0-G: the live files that need converting, one per line)
       profconv.py --convert-live [--src ...] [--src2 ...] [--apply]   (F0-G: convert the live dirs IN PLACE: X.env -> X.env.alt + new X.env)

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
import argparse, os, re, shutil, sys
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
# F0-G: data files of the release dir that are not read by a release profile but travel with the set (the X-curve file the Dashboard/launcher reads)
AUX_TREE = AUX + ["27b.xcurves.json"]
ALT = ".alt"          # F0-G: <profile>.env.alt = the old file, byte for byte, next to the converted <profile>.env
# a pre-rename spelling: SGLANG_* (HTSGLANG_* is the product env and stays: the letter in front excludes it), --weg2-* flags, WEG2-* log
# markers, the sglang.srt package path. Host paths and evidence names (/spinning/gpu-arb/weg2, boot_weg2_*) are R2 and are not matched.
# (written in pieces: this file is itself a text file the rename kit rewrites -- a whole legacy token here would be renamed along with the tree)
OLD_NAME_RE = re.compile(r"(?<![A-Za-z])SG" r"LANG_|--we" r"g2-|(?<![A-Za-z_])WE" r"G2-|sg" r"lang\.srt")


def convert(text, imap):
    """The conversion function of the kit (rename_to_flliper.rewrite_all) -- ONE function for profiles and tree."""
    return R.rewrite_all(text, False, True, imap)[0]


def old_name_lines(text):
    """1-based numbers of the lines of a profile that still carry a pre-rename spelling (see OLD_NAME_RE)."""
    return [i + 1 for i, l in enumerate(text.splitlines()) if OLD_NAME_RE.search(l)]


def live_files(src, src2):
    """(dir, file) for every *.env of the two live dirs (backups/staged variants are not *.env and stay out)."""
    out = []
    for d in (src, src2):
        if os.path.isdir(d):
            out += [(d, f) for f in sorted(os.listdir(d)) if f.endswith(".env") and os.path.isfile(os.path.join(d, f))]
    return out


def tree_out(a, imap):
    """F0-G: the release set into the tree: <out>/profiles_release/*.env (+ *.env.alt) with the data files, <out>/profiles/ the rig profiles
    named in PLAN2 (the NF line's abl form). Returns the number of differences when a.check is set."""
    bad = 0
    jobs = [(os.path.join(a.src, f), os.path.join(a.tree_out, "profiles_release"), f) for _, f in live_files(a.src, "/nonexistent")]
    jobs += [(os.path.join(a.src2, s), os.path.join(a.tree_out, "profiles"), s) for s, _ in PLAN2]
    for srcpath, dd, name in jobs:
        old = open(srcpath, encoding="utf-8").read()
        new = convert(old, imap)
        for fn, content in ((name, new), (name + ALT, old)):
            p = os.path.join(dd, fn)
            cur = open(p, encoding="utf-8").read() if os.path.exists(p) else None
            if cur == content:
                print("same   ", os.path.relpath(p, a.tree_out)); continue
            if a.check:
                print("DIFFERS", os.path.relpath(p, a.tree_out), "(missing)" if cur is None else ""); bad += 1; continue
            os.makedirs(dd, exist_ok=True)
            open(p, "w", encoding="utf-8").write(content)
            print("written", os.path.relpath(p, a.tree_out), "(new)" if cur is None else "(changed)")
    for f in AUX_TREE:
        sp = os.path.join(a.src, f)
        if not os.path.exists(sp):
            sp = os.path.join(a.src2, f)
        data = open(sp, "rb").read()      # follows the symlink
        p = os.path.join(a.tree_out, "profiles_release", f)
        cur = open(p, "rb").read() if os.path.exists(p) else None
        if cur == data:
            print("same   ", os.path.relpath(p, a.tree_out)); continue
        if a.check:
            print("DIFFERS", os.path.relpath(p, a.tree_out)); bad += 1; continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(data); print("written", os.path.relpath(p, a.tree_out))
    return bad


def convert_live(a, imap):
    """F0-G: convert the live dirs IN PLACE. Run by the operator at the switch-over, never by an agent. Per file: X.env -> X.env.alt
    (never overwritten when it exists) and the converted text as X.env. Without --apply only the plan is printed. Idempotent: a file whose
    conversion equals itself is skipped."""
    n_conv = n_same = 0
    for d, f in live_files(a.src, a.src2):
        p = os.path.join(d, f)
        old = open(p, encoding="utf-8").read()
        new = convert(old, imap)
        if new == old:
            n_same += 1; continue
        n_conv += 1
        if not a.apply:
            print("would convert", p, "-> %s%s" % (f, ALT)); continue
        alt = p + ALT
        if not os.path.exists(alt):
            shutil.copy2(p, alt)
        tmp = p + ".tmp_f0g"
        open(tmp, "w", encoding="utf-8").write(new)
        shutil.copymode(p, tmp)
        os.replace(tmp, p)
        print("converted", p)
    print("%s: %d to convert, %d already converted/without old names" % ("applied" if a.apply else "plan (use --apply)", n_conv, n_same))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/spinning/gpu-arb/docker/profiles_release")
    ap.add_argument("--src2", default="/spinning/gpu-arb/docker/profiles", help="dir of the PLAN2 profiles (non-release rig profiles)")
    ap.add_argument("--dst", default=os.path.join(KIT, "profconv"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tree-out", help="F0-G: <tree>/docker/flliper -- write profiles_release/ + profiles/ (converted + *.env.alt) there")
    ap.add_argument("--list-live", action="store_true", help="F0-G: print the live *.env files of --src/--src2")
    ap.add_argument("--convert-live", action="store_true", help="F0-G: convert the live dirs in place (X.env -> X.env.alt); needs --apply to write")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    imap = R._load_imap(os.path.join(KIT, "data", "merged_0928.json"))
    if a.list_live:
        for d, f in live_files(a.src, a.src2):
            print(os.path.join(d, f))
        return 0
    if a.convert_live:
        return convert_live(a, imap)
    if a.tree_out:
        return 1 if tree_out(a, imap) else 0
    bad = 0
    jobs = [(os.path.join(a.src, s), ds) for s, ds in PLAN] + [(os.path.join(a.src2, s), ds) for s, ds in PLAN2]
    for srcpath, dsts in jobs:
        text = open(srcpath).read()
        new = convert(text, imap)
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
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
