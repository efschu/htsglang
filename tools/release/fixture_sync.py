#!/usr/bin/env python3
"""fixture_sync.py <renamed worktree>  -- item 600 (03.10.2026).

The mechanical pass leaves `test/**/fixtures/**` alone on purpose (dual-name parsers are tested against unchanged old logs).
One class of fixture is NOT a log but a golden of the code's OWN output: the HW-GENERIC reference-rig plan fingerprint
(`.../fixtures/hw_generic_1002/rig_plan_fingerprint_*.json`, item 260 / 030-hwgen). The planner writes marker text such as
`WEG2-DORMANT-SERVED record` into it; after the rename the planner says `PDFLIP-DORMANT-SERVED`, the golden still says `WEG2-`
and `test_hw_generic_profile_gate_1003` fails (found by the item 600 dry-run on y8t 968d99d312: the only new test failure).

This step renames exactly two spellings in those goldens and nothing else: the marker prefix `WEG2-` -> `PDFLIP-` and the module path
`weg2.` -> `pdflip.` (NF tree: `ImportError: weg2.l15_plan not in this tree`); boot tags such as `weg2ls1b2` stay, they are evidence names.
It checks that the file still parses as JSON and that the change is exactly reversible (nothing else touched).
Prints `FIXTURE-SYNC files=<n> replacements=<m>`; exit 3 on any problem."""
import glob, json, os, re, sys

root = sys.argv[1]
GLOBS = ["test/registered/unit/pdflip/fixtures/hw_generic_1002/rig_plan_fingerprint_*.json"]
files = []
for g in GLOBS:
    files += sorted(glob.glob(os.path.join(root, g)))
n_rep = 0
for f in files:
    old = open(f, encoding="utf-8").read()
    new, k = re.subn(r"WEG2-", "PDFLIP-", old)
    new, k2 = re.subn(r"\bweg2\.", "pdflip.", new)
    k += k2
    json.loads(new)                                   # still JSON
    if re.sub(r"\bpdflip\.", "weg2.", re.sub(r"PDFLIP-", "WEG2-", new)) != old:   # reversible == only those spellings changed
        sys.exit("fixture_sync: not reversible for " + f)
    if k:
        open(f, "w", encoding="utf-8").write(new)
    n_rep += k
print("FIXTURE-SYNC files=%d replacements=%d" % (len(files), n_rep))
