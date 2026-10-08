#!/usr/bin/env python3
"""fixture_sync.py <renamed worktree>  -- item 600 (03.10.2026).

The mechanical pass leaves `test/**/fixtures/**` alone on purpose (dual-name parsers are tested against unchanged old logs).
One class of fixture is NOT a log but a golden of the code's OWN output: the HW-GENERIC reference-rig plan fingerprint
(`.../fixtures/hw_generic_1002/rig_plan_fingerprint_*.json`, item 260 / 030-hwgen). The planner writes marker text such as
`WEG2-DORMANT-SERVED record` into it; after the rename the planner says `PDFLIP-DORMANT-SERVED`, the golden still says `WEG2-`
and `test_hw_generic_profile_gate_1003` fails (found by the item 600 dry-run on y8t 968d99d312: the only new test failure).

This step renames exactly three spellings in those goldens and nothing else: the marker prefix `WEG2-` -> `PDFLIP-` and the module path
`weg2.` / `weg2/` -> `pdflip.` / `pdflip/` (NF tree: `ImportError: weg2.l15_plan not in this tree`); boot tags such as `weg2ls1b2` stay, they are evidence names.
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
    # F0-D (08.10.2026): the module path written with a slash (`(weg2/power_limit.RIG_POWER_LIMIT_W_0924)` in the note of the builtin rig limits)
    new, k3 = re.subn(r"(?<![\w/.-])weg2/", "pdflip/", new)          # not behind a path: /spinning/gpu-arb/weg2/ is a host dir and stays
    k += k3
    json.loads(new)                                   # still JSON
    def _canon(t):      # both spellings of the three renamed forms collapse to one token: nothing else may differ
        return re.sub(r"(?<![\w/.-])(?:weg2|pdflip)/", "@", re.sub(r"\b(?:weg2|pdflip)\.", "@", re.sub(r"(?:WEG2|PDFLIP)-", "#", t)))
    if _canon(new) != _canon(old):   # reversible == only those spellings changed (also on a file an earlier pass already converted)
        sys.exit("fixture_sync: not reversible for " + f)
    if k:
        open(f, "w", encoding="utf-8").write(new)
    n_rep += k
print("FIXTURE-SYNC files=%d replacements=%d" % (len(files), n_rep))
