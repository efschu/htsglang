#!/usr/bin/env bash
# Mutant harness for the #1273 B4c slice (the census producer + the ring
# table's per-rank tag retention).
#
# SAME SNAPSHOT DISCIPLINE AS tools/mutants_s6b.sh AND FOR THE SAME MEASURED
# REASON: a harness that restores with `git checkout --` reverts to the last
# COMMIT and therefore DISCARDS uncommitted work. It ate two slices of seat 4's
# before that was fixed. So the tree is snapshot by copy, every restore comes
# from the snapshot, the snapshot is verified byte-for-byte afterwards, and
# nothing here consults git for content.
#
# WHAT IS DIFFERENT FROM ITS SIBLING: the worktree is a parameter (WT=... in the
# environment) rather than a literal, because that literal is exactly what made
# the sibling unusable from another seat's tree, and the mutant set is this
# slice's. The sibling is left untouched -- the S6-BOUNCE record cites it.
#
# SELF-TEST: `--selfcheck` plants an UNCOMMITTED edit, runs one mutation cycle
# over it, and asserts the planted edit SURVIVED. Run it first; a harness that
# cannot prove it preserves the tree is the instrument-that-cannot-fail shape
# one level up.
#
# EMPTY IS NOT A PASS. Two guards, both learned the hard way: `mutate` REFUSES
# when its anchor is not unique (a stale anchor silently mutates nothing and
# scores GREEN-SURVIVED), and it VERIFIES the file changed after writing.
set -uo pipefail

WT="${WT:-/spinning/wt-desk-b4c}"
SNAP=""

cleanup() { [ -n "$SNAP" ] && rm -rf -- "$SNAP"; }
trap cleanup EXIT

snapshot() {
  SNAP="$(mktemp -d /tmp/mutsnap.XXXXXX)" || exit 2
  local f
  for f in "$@"; do
    mkdir -p "$SNAP/$(dirname "$f")"
    cp -a -- "$WT/$f" "$SNAP/$f" || exit 2
  done
  echo "snapshot: $# file(s) -> $SNAP"
}

restore() {
  local f="$1"
  cp -a -- "$SNAP/$f" "$WT/$f" || { echo "RESTORE FAILED: $f"; exit 3; }
  cmp -s -- "$SNAP/$f" "$WT/$f" || { echo "RESTORE NOT BYTE-EQUAL: $f"; exit 3; }
}

verify_all() {
  local f rc=0
  while IFS= read -r f; do
    cmp -s -- "$SNAP/$f" "$WT/$f" || { echo "TREE DRIFTED: $f"; rc=1; }
  done < <(cd "$SNAP" && find . -type f | sed 's|^\./||')
  [ "$rc" -eq 0 ] && echo "tree verified: every snapshotted file byte-identical"
  return "$rc"
}

mutate() {  # mutate <file> <old> <new>
  python3 - "$WT/$1" "$2" "$3" <<'PY'
import hashlib, sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old, new = sys.argv[2], sys.argv[3]
n = s.count(old)
if n != 1:
    print(f"ANCHOR NOT UNIQUE ({n}) -- mutation skipped, NOT a pass")
    sys.exit(4)
before = hashlib.sha256(s.encode()).hexdigest()
p.write_text(s.replace(old, new))
after = hashlib.sha256(p.read_text().encode()).hexdigest()
if before == after:
    print("FILE UNCHANGED after write -- mutation did NOT apply, NOT a pass")
    sys.exit(4)
PY
}

CE=python/sglang/srt/weg2/xchg_census.py
RT=python/sglang/srt/weg2/ring_table.py
T="test/registered/unit/weg2/test_weg2_xchg_census_1273.py
test/registered/unit/weg2/test_weg2_tag_by_rank_1273.py"

if [ "${1:-}" = "--selfcheck" ]; then
  PLANT="# MUTANT-HARNESS-SELFCHECK-PLANTED-$$"
  cd "$WT" || exit 2
  if ! git diff --quiet -- "$CE"; then
    echo "SELFCHECK SKIPPED: $CE is already dirty; run on a clean file"; exit 2
  fi
  printf '\n%s\n' "$PLANT" >> "$CE"
  snapshot "$CE"
  mutate "$CE" 'def write_census(' 'def write_census_MUTANT('
  restore "$CE"
  if grep -qF "$PLANT" "$CE"; then
    echo "SELFCHECK PASS: the planted uncommitted edit SURVIVED a mutation cycle"
    rc=0
  else
    echo "SELFCHECK FAIL: the harness destroyed uncommitted work"; rc=1
  fi
  python3 - "$WT/$CE" "$PLANT" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
p.write_text(p.read_text().replace("\n" + sys.argv[2] + "\n", ""))
PY
  git diff --quiet -- "$CE" && echo "tree restored to HEAD-clean" || echo "WARNING: $CE still dirty"
  exit "$rc"
fi

cd "$WT" || exit 2
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT/python" OMP_NUM_THREADS=1
run() { timeout 900 python3 -m pytest $T -q --tb=no -p no:cacheprovider \
          --color=no 2>&1 | grep -oE '[0-9]+ (passed|failed|error)' | tr '\n' ' '; }

snapshot "$CE" "$RT"
echo "BASELINE: $(run)"

m() { echo "$1"; if mutate "$2" "$3" "$4"; then run; else echo "(anchor stale -- FIX THE ANCHOR)"; fi; restore "$2"; echo; }

# --- the danger direction: an absence read as a zero -----------------------
m "M1 the completeness identity is dropped (absence becomes a zero unproven)" "$CE" \
  '        complete = [c for c in candidates if c[1] == c[2] and c[2] > 0]' \
  '        complete = list(candidates)'
m "M2 the zero-fill becomes an omission (check_partition then refuses)" "$CE" \
  '        out[uuid] = {t: int(tags.get(t, 0)) for t in want}' \
  '        out[uuid] = {t: int(v) for t, v in tags.items()}'
m "M3 a measured tag outside the family is silently intersected away" "$CE" \
  '    extra = sorted(set(glog.tag_by_rank) - set(want))' \
  '    extra = []'

# --- the danger direction: the dormant term, which solve() DOUBLES ---------
m "M4 the dormant reading takes the last sample, not the peak" "$CE" \
  '                    if value > per.get(uuid, 0):' \
  '                    if True:'
m "M5 the SMALLER of the two groups' residues is charged" "$CE" \
  '        best = max(per_group, key=lambda g: per_group[g])' \
  '        best = min(per_group, key=lambda g: per_group[g])'
m "M6 the spec-1.6 EXPECTATIONS row becomes the constant" "$CE" \
  '        return launcher.DC_MEASURED_D_5090_MIB, (' \
  '        return launcher.DC_EXPECT_5090_MIB, ('
m "M7 an unrecognised board borrows the 3080's constant" "$CE" \
  '    if "3080" in name:' \
  '    if True:'

# --- the danger direction: a census of the wrong boot ----------------------
m "M8 the xchg-shadow exclusion is dropped" "$CE" \
  '    if marked:' \
  '    if False:'
m "M9 the selection oracle's disagreement is ignored" "$CE" \
  '        if chose != stem:' \
  '        if False:'
m "M10 the published wave map is not cross-checked against the rebuild" "$CE" \
  '    if published and published != rebuilt:' \
  '    if False:'

# --- the danger direction: a bound printed as a reading -------------------
m "M11 a weights-only census no longer says LOWER BOUND" "$CE" \
  '        if not glog.covers_all_backed_up_tags:' \
  '        if False:'

# --- the danger direction: the wave arms swapped --------------------------
m "M12 the default arm becomes the one that cannot fit" "$CE" \
  '    wave_map_arm: str = "launcher",' \
  '    wave_map_arm: str = "ranks",'
m "M13 the partition ratchet is dropped" "$CE" \
  '    if sorted(flat) != sorted(dict.fromkeys(str(t) for t in family)):' \
  '    if False:'

# --- the ring table: both directions of the retention ---------------------
m "M14 tag_by_rank is fed from steps again (the base tag disappears)" "$RT" \
  '        for tag, mib in singles:' \
  '        for tag, mib in steps:'
m "M15 tag_totals is fed from singles (the byte-identity breaks)" "$RT" \
  '        for tag, mib in steps:' \
  '        for tag, mib in singles:'

# --- the danger direction: a foreign layer placement read as this form's ----
m "M16 an unequal split is reported as a match" "$CE" \
  '    if src == mine:' \
  '    if True:'
m "M17 an incomparable form is bounded instead of refused" "$CE" \
  '    if len(src) != len(mine) or src_chunk != my_chunk or src_layers != my_layers:' \
  '    if False:'
m "M18 the bound loses its direction" "$CE" \
  "        f\" -> {'OVER' if d > 0 else 'UNDER'}-priced by {abs(d)} layer(s)\"" \
  '        " -> differs"'
m "M19 the split note never reaches the census" "$CE" \
  '        + ("; " + form_note if form_note else "")' \
  "        + \"\""

# --- the danger direction: the CLI needs hardware to refuse a typo ---------
m "M20 the argument check falls behind the NVML read" "$CE" \
  "        raise _refuse(\"pass --form-from or --weight-chunks: the tag family is read, never guessed\")" \
  '        family = []'

echo "BASELINE AFTER: $(run)"
verify_all
