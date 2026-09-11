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

WT="${WT:-/spinning/wt-desk-b4h}"
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

CE=python/sglang/srt/weg2/launcher.py
RT=python/sglang/srt/weg2/launcher.py
T="test/registered/unit/weg2/test_weg2_w19_form_residue_1273.py
test/registered/unit/weg2/test_weg2_xchg_reserve_1273.py
test/registered/unit/weg2/test_weg2_corridor_budget_1257.py"

if [ "${1:-}" = "--selfcheck" ]; then
  PLANT="# MUTANT-HARNESS-SELFCHECK-PLANTED-$$"
  cd "$WT" || exit 2
  if ! git diff --quiet -- "$CE"; then
    echo "SELFCHECK SKIPPED: $CE is already dirty; run on a clean file"; exit 2
  fi
  printf '\n%s\n' "$PLANT" >> "$CE"
  snapshot "$CE"
  mutate "$CE" 'def dc_measured_d_mib(' 'def dc_measured_d_mib_MUTANT('
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

snapshot "$CE"
echo "BASELINE: $(run)"

m() { echo "$1"; if mutate "$2" "$3" "$4"; then run; else echo "(anchor stale -- FIX THE ANCHOR)"; fi; restore "$2"; echo; }

# --- THE danger direction: the xchg constant leaking into the serving path --
m "M1 the xchg triple leaks into the serving path (every serving boot loses 666-856 MiB)" "$CE" \
  '    xchg = str(weight_source) == WEIGHT_SOURCE_EXCHANGE' \
  '    xchg = True'
m "M2 the arm never selects the xchg triple (W19 fires again)" "$CE" \
  '    xchg = str(weight_source) == WEIGHT_SOURCE_EXCHANGE' \
  '    xchg = False'
m "M3 the inject arm is conjoined, so shadow reads the serving residue" "$CE" \
  '    xchg = str(weight_source) == WEIGHT_SOURCE_EXCHANGE' \
  '    xchg = str(weight_source) == WEIGHT_SOURCE_EXCHANGE and weight_exchange.inject_authoritative()'
m "M4 the two boards are swapped" "$CE" \
  '        return DC_MEASURED_D_XCHG_5090_MIB if xchg else DC_MEASURED_D_5090_MIB' \
  '        return DC_MEASURED_D_XCHG_3080_MIB if xchg else DC_MEASURED_D_5090_MIB'
m "M5 an unknown board borrows the 3080 triple" "$CE" \
  '    if "3080" in name:' \
  '    if True:'
m "M6 the measured triple drifts from the boot" "$CE" \
  'DC_MEASURED_D_XCHG_MIB = (2588, 3084, 2588)' \
  'DC_MEASURED_D_XCHG_MIB = (2588, 3080, 2588)'
m "M7 main stops going through the one selector" "$CE" \
  '        c.uuid: dc_measured_d_mib(c, ns.weg2_weight_source) + slack_mib' \
  '        c.uuid: (DC_MEASURED_D_5090_MIB if "5090" in c.name else DC_MEASURED_D_3080_MIB) + slack_mib'
m "M8 the residual is folded into the priced value" "$CE" \
  '        residual[c.uuid] = int(measured) - int(out[c.uuid])' \
  '        residual[c.uuid] = 0'
m "M9 the line stops naming its measured source" "$CE" \
  '            f"measured_mib={measured} source=measured:weg2xsn14 "' \
  '            f"measured_mib={measured} "'

echo "BASELINE AFTER: $(run)"
verify_all
