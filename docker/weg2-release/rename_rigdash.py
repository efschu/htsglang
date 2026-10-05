#!/usr/bin/env python3
"""rename_rigdash.py -- der Profil-Editor (tools/rig_dashboard) im fLLiper-Namensraum (Auftrag 2012, Baustopper L1).

Das fLLiper-Image traegt einen umbenannten Planer-Baum (python/flliper, srt/pdflip, ENV FLLIPER_PDFLIP_*). Der Editor aus dem
htsglang-Repo kennt nur die alten Namen (sglang/srt/weg2, SGLANG_WEG2_*; Katalog 902 Mal). Dieses Skript wendet DIESELBE
Umbenennungsmaschine an wie das Release-Kit (rename_to_flliper.py apply --weg2 --ident-map, gleiche Tabelle) auf den
Editor-Paketinhalt -- nicht mehr und nicht weniger. Kein zweiter Regelsatz: was im Baum umbenannt wird, wird im Editor
genauso umbenannt, und `verify` (der unabhaengige Pruefer des Kits) beweist es.

    rename_rigdash.py <pkg_dir> [--out DIR] [--kit DIR] [--dry-run]

  <pkg_dir>  Inhalt von tools/rig_dashboard (rigdash/, kartenplan_build/, entrypoint_rigdash.sh; Teilmengen sind erlaubt,
             z. B. das Paket in ctx/tools/rigdash). Ohne --out wird <pkg_dir> an Ort und Stelle umgeschrieben.
  --kit      Verzeichnis des Kits (Standard $RELEASE_KIT_TOOLS oder /spinning/flliper/tools): rename_to_flliper.py und
             release/data/merged_0928.json (die Ident-Tabelle) muessen dort liegen.

Was es tut (und was nicht):
  * Der Paketinhalt wird in ein Wegwerf-Repo unter tools/rig_dashboard/ gestellt (so greift der Bereich INCLUDE tools/** des Kits).
  * Inhaltlich GESPERRT bleiben `rigdash/kartenplan_data/*`: Boot-Aufzeichnungen, deren plan_id ein sha256 ueber den Inhalt ist
    (kartenplan.plan_id_ok); umbenannt faellt jede Aufzeichnung durch (gemessen: test_plan_id_wird_nachgerechnet rot). Sie sind
    Beleg-Daten wie die Evidenz-Dateien des Kits und wandern byte-gleich zurueck.
  * Kollisionen (zwei alte Woerter werden ein neues, `flliper` <- {`flliper`, `sglang`}) sind bei 20 Paketdateien Absicht (der Editor
    schreibt `flliper.server/1` und `sglang-Umgebung` nebeneinander). Freigegeben sind genau die Dateien in
    rename_rigdash_collision_ok.json; eine NEUE Kollisionsdatei bricht ab (Exit 3), bis sie gelesen und eingetragen ist.
  * Die Ident-Tabelle (NAME-Tokens, merged_0928.json) laeuft mit; der zweite Kit-Schritt `ident_fix.py` (Strings, .sh/.md/.json
    nach Vollwort) laeuft NICHT: er schreibt deutsche Prosa um (`blockiert` -> `is_blocked` in Katalogtext) und kennt die
    Dashboard-eigenen Statuswerte nicht. Der Planer-Baum bekommt ihn trotzdem (Kit-Schritt 2); die Kopplung ist im Test
    test_rename_editor_2012 gepinnt (die Woerter der Ident-Fix-Tabelle, die der Editor liest).
  * Der Prosa-Uebersetzungsschritt (Kit-Schritt 3) laeuft nicht; Kommentare und Texte des Editors bleiben deutsch.
  * Danach: Rest-Pruefung (kein SGLANG_WEG2_, kein sglang.srt, kein "sglang","srt" ausserhalb der gesperrten Daten).

Exit: 0 ok, 2 Aufruf, 3 Pruefung gescheitert (nichts geschrieben), 4 Kit fehlt / Ausfuehrung gescheitert.
Keine GPU, kein Docker, keine Netzwerkzugriffe. Das Wegwerf-Repo liegt unter tempfile; das htsglang-Repo wird nicht beruehrt.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KIT = os.environ.get("RELEASE_KIT_TOOLS") or "/spinning/flliper/tools"
COLLISION_OK = os.path.join(HERE, "rename_rigdash_collision_ok.json")
PKG_PREFIX = "tools/rig_dashboard"
#: inhaltlich gesperrt (Pfade relativ zum Paketinhalt, fnmatch)
LOCKED = ("rigdash/kartenplan_data/*",)
SKIP_DIRS = {"__pycache__", ".git"}
#: harte Reste: so etwas darf im umbenannten Paket (ausser den gesperrten Daten) nirgends mehr stehen
RESIDUE = (
    ("SGLANG_WEG2_", re.compile(r"SGLANG_WEG2_")),
    ("sglang.srt", re.compile(r"\bsglang\.srt\b")),
    ("sglang/srt", re.compile(r"\bsglang/srt\b")),
    ('"sglang", "srt"', re.compile(r"""["']sglang["']\s*,\s*["']srt["']""")),
)
TEXT_EXT = {".py", ".sh", ".js", ".html", ".json", ".md", ".txt", ".yml", ".yaml", ".service", ".css", ".env", ""}


def die(code, msg):
    print("rename_rigdash: " + msg, file=sys.stderr)
    sys.exit(code)


def is_locked(rel):
    import fnmatch
    return any(fnmatch.fnmatchcase(rel, p) for p in LOCKED)


def walk(root):
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for f in sorted(files):
            full = os.path.join(d, f)
            yield os.path.relpath(full, root), full


def run(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pkg_dir")
    ap.add_argument("--out")
    ap.add_argument("--kit", default=DEFAULT_KIT)
    ap.add_argument("--dry-run", action="store_true", help="rechnen und pruefen, nichts schreiben")
    a = ap.parse_args(argv)

    pkg = os.path.abspath(a.pkg_dir)
    if not os.path.isdir(pkg):
        die(2, "%s ist kein Verzeichnis" % pkg)
    engine = os.path.join(a.kit, "rename_to_flliper.py")
    imap = os.path.join(a.kit, "release", "data", "merged_0928.json")
    for p in (engine, imap, COLLISION_OK):
        if not os.path.isfile(p):
            die(4, "fehlt: %s (Kit: --kit bzw. RELEASE_KIT_TOOLS)" % p)

    tmp = tempfile.mkdtemp(prefix="rename_rigdash_")
    try:
        repo = os.path.join(tmp, "repo")           # Git-Wurzel des Wegwerf-Repos; Hilfsdateien liegen daneben, nicht darin
        stage = os.path.join(repo, *PKG_PREFIX.split("/"))
        locked = []
        for rel, full in walk(pkg):
            if is_locked(rel):
                locked.append(rel)
                keep = os.path.join(tmp, "locked", rel)
                os.makedirs(os.path.dirname(keep), exist_ok=True)
                shutil.copy2(full, keep)
                continue
            dst = os.path.join(stage, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(full, dst)
        if not os.path.isdir(stage):
            die(3, "%s ist leer" % pkg)

        g = lambda *x: run(["git", "-C", repo, *x])
        for step in (["init", "-q", "."], ["config", "user.name", "rename-rigdash"], ["config", "user.email", "rename-rigdash@localhost"],
                     ["add", "-A"], ["commit", "-q", "-m", "base"]):
            r = g(*step)
            if r.returncode:
                die(4, "git %s: %s" % (step[0], r.stdout.strip()[-300:]))

        ok = {k: v for k, v in json.load(open(COLLISION_OK)).items() if not k.startswith("_")}
        okfile = os.path.join(tmp, "collision_ok.json")
        json.dump({"%s/%s" % (PKG_PREFIX, k): v for k, v in ok.items()}, open(okfile, "w"))
        env = dict(os.environ, COLLISION_OK_FILE=okfile)
        manifest = os.path.join(tmp, "manifest.json")
        r = run([sys.executable, engine, "apply", "--root", repo, "--weg2", "--ident-map", imap, "--manifest", manifest], env=env)
        if r.returncode:
            msg = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "rc %d" % r.returncode
            die(3, "Umbenennung abgelehnt: %s (neue Kollisionsdatei? dann lesen und in rename_rigdash_collision_ok.json eintragen)" % msg)
        # Unabhaengiger Pruefer des Kits (rechnet nicht mit der Engine): byte-gleich bis auf Namens-Tokens
        r = run([sys.executable, engine, "verify", "--repo", repo, "--base", "HEAD", "--root", repo, "--weg2", "--ident-map", imap], env=env)
        last = (r.stdout.strip().splitlines() or ["?"])[-1]
        if r.returncode or "PASS" not in last:
            fails = [l for l in r.stdout.splitlines() if l.startswith("FAIL")][:5]
            die(3, "verify nicht PASS: %s | %s" % (last, " | ".join(fails)))
        # zweiter Durchlauf aendert nichts (Idempotenz, wie im Kit)
        before = g("diff", "--stat").stdout
        r2 = run([sys.executable, engine, "apply", "--root", repo, "--weg2", "--ident-map", imap], env=env)
        if r2.returncode or g("diff", "--stat").stdout != before:
            die(3, "zweiter Durchlauf aendert den Baum (nicht idempotent)")

        # Rest-Pruefung auf dem umbenannten Baum
        new_root = stage
        residue = {}
        for rel, full in walk(new_root):
            if os.path.splitext(rel)[1] not in TEXT_EXT:
                continue
            try:
                txt = open(full, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            for name, rx in RESIDUE:
                n = len(rx.findall(txt))
                if n:
                    residue.setdefault(rel, {})[name] = n
        if residue:
            die(3, "Reste alter Namen im umbenannten Paket: " + json.dumps(residue, sort_keys=True)[:600])

        summary = json.load(open(manifest))
        res = {"files": len(list(walk(new_root))), "locked_kept": sorted(locked), "paths_moved": summary["paths_moved"],
               "replacements": summary["replacements_total"], "tree_digest": summary["tree_digest"][:16],
               "collisions_allowed": len(summary.get("collisions_allowed", {}))}
        if a.dry_run:
            print("rename_rigdash: DRY-RUN ok " + json.dumps(res, sort_keys=True))
            return 0

        out = os.path.abspath(a.out) if a.out else pkg
        if a.out:
            if os.path.exists(out) and os.listdir(out):
                die(3, "--out %s ist nicht leer" % out)
        else:
            for rel, full in list(walk(pkg)):
                os.remove(full)
            for d, dirs, files in os.walk(pkg, topdown=False):
                if d != pkg and not os.listdir(d):
                    os.rmdir(d)
        for rel, full in walk(new_root):
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(full, dst)
        # gesperrte Daten byte-gleich zurueck (an ihren alten Platz: sie liegen im Namen nicht unter weg2)
        for rel in locked:
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(tmp, "locked", rel), dst)
        print("rename_rigdash: ok " + json.dumps(res, sort_keys=True))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
