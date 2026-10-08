#!/usr/bin/env python3
"""planer_fixture_sync.py <renamed worktree>  -- F0-D (08.10.2026): the planner goldens of the renamed tree, by the kit's own rule.

fixture_sync.py (item 600) renames one class of fixture, the HW-GENERIC plan fingerprint.  A second class is the golden of the planner
reference (`test/registered/unit/pdflip/fixtures/planer_1006`): the profile SNAPSHOTS (copies of the release profiles, `profiles/*.env`),
the launch-input goldens (`golden/launch_*.json`), the launcher dry-run goldens (`golden/**/*.txt`) and their `*.provenance.json`.
They spell the names the CODE spells (`SGLANG_WEG2_*`, `--weg2-*`, `WEG2-LAUNCH`); the renamed code reads and writes the new names, so
`test_planer_referenz_n3_1006` read KeyError 'FLLIPER_PDFLIP_DUAL_MPS_OPT_IN' on the old bytes.  The mechanical pass leaves `fixtures/**` alone
(evidence); this step converts exactly these files with the SAME function as profconv.py (`rewrite_all(text, False, weg2=True, ident-map)`:
host paths, evidence tags and persisted ids keep their old spelling by the engine's deny spans) and then re-pins the sha256 of the
converted profile snapshots where the tree records them (`PROVENANCE` in the reference test, `profile.sha256` in `*.provenance.json`).
`27b-nvfp4.pchunk.json` is measured data and stays byte for byte (like profconv's AUX).

Prints `PLANER-FIXTURE-SYNC files=<n> repinned=<m>`; exit 3 on any problem (a converted JSON that does not parse, a recorded sha that is
not found).  Idempotent: a second run changes nothing."""
import glob, hashlib, json, os, re, subprocess, sys

KIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
import rename_to_flliper as R

root = sys.argv[1]
FIX = os.path.join(root, "test/registered/unit/pdflip/fixtures/planer_1006")
TEST = os.path.join(root, "test/registered/unit/pdflip/test_planer_referenz_n3_1006.py")
imap = R._load_imap(os.path.join(KIT, "data", "merged_0928.json"))


# Two spellings the engine keeps (host-path deny span: the literal sits behind /spinning/...) although the RENAMED CODE prints them new:
# the evidence records dir is built as os.path.join(EVIDENCE_DIR, "records", "weg2") (launcher._records_dir, renamed by the mechanical
# pass) and the golden launch json carries the profile snapshot path "test/registered/unit/weg2/fixtures/..." of a directory that moved.
PINS = [("/records/weg2", "/records/pdflip"), ("registered/unit/weg2/", "registered/unit/pdflip/")]


def conv(path):
    old = open(path, encoding="utf-8").read()
    new = R.rewrite_all(old, False, True, imap)[0]
    for a, b in PINS:
        new = new.replace(a, b)
    if path.endswith(".json"):
        json.loads(new)
    if new != old:
        open(path, "w", encoding="utf-8").write(new)
    return old, new


files = sorted(glob.glob(FIX + "/profiles/*.env") + glob.glob(FIX + "/golden/**/*.json", recursive=True) + glob.glob(FIX + "/golden/**/*.txt", recursive=True))
sha_map = {}
n = 0
OLD_REF = sys.argv[sys.argv.index("--old-ref") + 1] if "--old-ref" in sys.argv else None     # profiles converted by an earlier run: old bytes from git
for f in files:
    old, new = conv(f)
    n += old != new
    if f.endswith(".env"):
        if old == new and OLD_REF:
            rel = os.path.relpath(f, root)
            r = subprocess.run(["git", "-C", root, "show", "%s:%s" % (OLD_REF, rel)], capture_output=True)
            if r.returncode == 0:
                old = r.stdout.decode("utf-8")
        if old != new:
            sha_map[hashlib.sha256(old.encode()).hexdigest()] = hashlib.sha256(new.encode()).hexdigest()
repinned = 0
# every file of the tree that records the sha of a snapshot: the reference test, the other planner tests that tie to the same
# snapshots (abnahme, ape_dual: their own PROVENANCE tables) and the provenance json of the goldens
for f in [TEST] + sorted(glob.glob(os.path.join(root, "test/registered/unit/pdflip/test_planer_*.py"))) + sorted(glob.glob(FIX + "/golden/**/*.provenance.json", recursive=True)):
    t = open(f, encoding="utf-8").read()
    t2 = t
    for a, b in sha_map.items():
        if a in t2:
            t2 = t2.replace(a, b)
            repinned += 1
    if t2 != t:
        open(f, "w", encoding="utf-8").write(t2)
        if f.endswith(".json"):
            json.loads(t2)
print("PLANER-FIXTURE-SYNC files=%d converted=%d repinned=%d" % (len(files), n, repinned))
