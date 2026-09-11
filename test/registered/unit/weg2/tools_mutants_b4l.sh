#!/usr/bin/env bash
# #1328 B4l mutant harness -- DANGER DIRECTION FIRST: every mutant makes the
# ratchet MORE permissive (or blinds its scope check), because a ratchet that
# passes when it should fail is the only failure mode that matters here.
# Each mutant must turn the file RED; the baseline must be GREEN before and
# after.  Run from the repo root:  bash test/registered/unit/weg2/tools_mutants_b4l.sh
set -u
BASE=test/registered/unit/weg2/test_weg2_xchg_refusal_reachability_1328.py
WORK=test/registered/unit/weg2/_b4l_mutant_tmp.py
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$(pwd)/python"

run() {  # $1 = label, expectation in $2 (GREEN|RED)
  # ANSI codes sit between "passed" and the comma in pytest's summary, so the
  # verdict is read off the ESCAPE-STRIPPED tail -- a colour code once made the
  # green baseline read as RED here, which is a harness lie in the direction
  # that hides a surviving mutant.
  out=$(timeout 900 python3 -m pytest "$WORK" -q -p no:cacheprovider 2>&1 \
        | sed -r 's/\x1b\[[0-9;]*m//g' | tail -3)
  if echo "$out" | grep -qE "[0-9]+ (failed|error)"; then
    got=RED
  elif echo "$out" | grep -qE "[0-9]+ passed"; then
    got=GREEN
  else
    got=RED
  fi
  printf '%-58s expected=%-5s got=%s\n' "$1" "$2" "$got"
  [ "$got" = "$2" ] || printf '    ^^ MUTANT SURVIVED / BASELINE BROKE\n%s\n' "$out"
}

cp "$BASE" "$WORK"; run "M0 baseline (unmutated)" GREEN

# M1: every function counts as wired -> the debt goes stale, silently
cp "$BASE" "$WORK"
python3 - "$WORK" <<'EOF'
import sys, re
p=sys.argv[1]; s=open(p).read()
s=s.replace('    _rel, fn, _lineno = idx.defs[qual]\n', '    return True  # M1\n', 1)
open(p,'w').write(s)
EOF
run "M1 _referenced_anywhere always True" RED

# M2: everything reachable -> no orphan can exist, no site can be unwired
cp "$BASE" "$WORK"
python3 - "$WORK" <<'EOF'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace('    seeds = [q for rel, fn in entries', '    return set(idx.defs)  # M2\n    seeds = [q for rel, fn in entries', 1)
open(p,'w').write(s)
EOF
run "M2 reachable() returns every def" RED

# M3: the orphan rule never reports (the W84 assertion cannot fail)
cp "$BASE" "$WORK"
python3 - "$WORK" <<'EOF'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace('        if not live_readers:', '        if False:  # M3', 1)
open(p,'w').write(s)
EOF
run "M3 orphan_verdicts never reports" RED

# M4: the raise scan stops seeing the lane's W-codes
cp "$BASE" "$WORK"
python3 - "$WORK" <<'EOF'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace('if name and name.startswith("Weg2"):', 'if name and name.startswith("ZZnope"):  # M4', 1)
open(p,'w').write(s)
EOF
run "M4 raise scan matches nothing" RED

# M5: TEST modules count as production wiring -- the #1001 mention-vs-use trap.
# refuse_if_not_ok IS called by test_weg2_xchg_cover_1273.py, so this mutant
# turns the W84 debt entry "wired" from a test alone.
cp "$BASE" "$WORK"
python3 - "$WORK" <<'EOF'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace('        if rel.startswith("test") or "/test" in rel:\n            continue\n',
            '        pass  # M5: tests count as production\n', 1)
open(p,'w').write(s)
EOF
run "M5 test modules count as production wiring" RED

# M6: the entry set is emptied -> every ratchet goes vacuously green
cp "$BASE" "$WORK"
python3 - "$WORK" <<'EOF'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace('PRODUCTION_ENTRIES = (\n    ("weg2/launcher.py", "cli"),',
            'PRODUCTION_ENTRIES = (  # M6\n    ("weg2/launcher.py", "nosuchentry"),', 1)
open(p,'w').write(s)
EOF
run "M6 entry set emptied (vacuous green)" RED

cp "$BASE" "$WORK"; run "M7 baseline again (unmutated)" GREEN
rm -f "$WORK"
