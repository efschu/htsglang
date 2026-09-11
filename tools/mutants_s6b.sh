#!/usr/bin/env bash
# Mutant harness for the #1273 S6-BOUNCE slices.
#
# WHY THIS FILE IS NOT `git checkout --` ANY MORE, and it is a measured lesson,
# not a preference: the earlier harness restored each mutated file with
# `git checkout -- <file>`, which reverts to the last COMMIT. Run against a
# working tree that had uncommitted work -- which is the normal state while
# building -- it DISCARDED that work. It ate step 5's adaptation once and the
# whole of step 6c's three product files once. A harness whose failure mode is
# "your edits are gone" is worse than no harness.
#
# So: the tree is SNAPSHOT first (a real copy of every file this run will
# touch), every restore comes from that snapshot, and the snapshot is verified
# byte-for-byte after the run. Nothing here consults git for content.
#
# SELF-TEST: `--selfcheck` plants an uncommitted edit, runs one mutation cycle
# over it, and asserts the planted edit SURVIVED. A harness that cannot prove
# it preserves the tree is the instrument-that-cannot-fail shape one level up.
set -uo pipefail

WT=/spinning/wt-desk-s6b
SNAP=""

cleanup() { [ -n "$SNAP" ] && rm -rf -- "$SNAP"; }
trap cleanup EXIT

snapshot() {  # snapshot <file>...
  SNAP="$(mktemp -d /tmp/mutsnap.XXXXXX)" || exit 2
  local f
  for f in "$@"; do
    mkdir -p "$SNAP/$(dirname "$f")"
    cp -a -- "$WT/$f" "$SNAP/$f" || exit 2
  done
  echo "snapshot: $# file(s) -> $SNAP"
}

restore() {  # restore <file>  -- FROM THE SNAPSHOT, never from git
  local f="$1"
  cp -a -- "$SNAP/$f" "$WT/$f" || { echo "RESTORE FAILED: $f"; exit 3; }
  cmp -s -- "$SNAP/$f" "$WT/$f" || { echo "RESTORE NOT BYTE-EQUAL: $f"; exit 3; }
}

verify_all() {  # every snapshotted file must be byte-identical to its snapshot
  local f rc=0
  while IFS= read -r f; do
    cmp -s -- "$SNAP/$f" "$WT/$f" || { echo "TREE DRIFTED: $f"; rc=1; }
  done < <(cd "$SNAP" && find . -type f | sed 's|^\./||')
  [ "$rc" -eq 0 ] && echo "tree verified: every snapshotted file byte-identical"
  return "$rc"
}

mutate() {  # mutate <file> <old> <new>
  python3 - "$WT/$1" "$2" "$3" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old, new = sys.argv[2], sys.argv[3]
n = s.count(old)
if n != 1:
    print(f"ANCHOR NOT UNIQUE ({n}) -- mutation skipped, NOT a pass")
    sys.exit(4)
p.write_text(s.replace(old, new))
PY
}

# ---------------------------------------------------------------- self-test
if [ "${1:-}" = "--selfcheck" ]; then
  F=python/sglang/srt/weg2/weight_exchange_bounce.py
  PLANT="# MUTANT-HARNESS-SELFCHECK-PLANTED-$$"
  cd "$WT" || exit 2
  if ! git diff --quiet -- "$F"; then
    echo "SELFCHECK SKIPPED: $F is already dirty; run on a clean file"; exit 2
  fi
  printf '\n%s\n' "$PLANT" >> "$F"          # an UNCOMMITTED edit, as in real use
  snapshot "$F"
  mutate "$F" 'def widest_run(' 'def widest_run_MUTANT('
  restore "$F"
  if grep -qF "$PLANT" "$F"; then
    echo "SELFCHECK PASS: the planted uncommitted edit SURVIVED a mutation cycle"
    rc=0
  else
    echo "SELFCHECK FAIL: the harness destroyed uncommitted work"
    rc=1
  fi
  # leave the tree as it was found
  python3 - "$WT/$F" "$PLANT" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
s = p.read_text().replace("\n" + sys.argv[2] + "\n", "")
p.write_text(s)
PY
  git diff --quiet -- "$F" && echo "tree restored to HEAD-clean" || echo "WARNING: $F still dirty"
  exit "$rc"
fi

# ---------------------------------------------------------------- the run
cd "$WT" || exit 2
BX=python/sglang/srt/weg2/weight_exchange_bounce.py
WU=python/sglang/srt/managers/scheduler_components/weight_updater.py
WX=python/sglang/srt/weg2/weight_exchange.py
T="test/registered/unit/weg2/test_weg2_xchg_inject_modes_1273.py
test/registered/unit/weg2/test_weg2_xchg_seam_wiring_1273.py
test/registered/unit/weg2/test_weg2_xchg_bounce_execution_smoke_1273.py
test/registered/unit/weg2/test_weg2_xchg_plan_provider_1273.py"

export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT/python" OMP_NUM_THREADS=1
run() { timeout 900 python3 -m pytest $T -q --tb=no -p no:cacheprovider \
          --color=no 2>&1 | grep -oE '[0-9]+ (passed|failed|error)' | tr '\n' ' '; }

snapshot "$BX" "$WU" "$WX"
echo "BASELINE: $(run)"

m() {  # m <label> <file> <old> <new>
  echo "$1"
  if mutate "$2" "$3" "$4"; then run; else echo "(anchor stale -- FIX THE ANCHOR)"; fi
  restore "$2"; echo
}

m "P1 the compare cannot fail" "$BX" '        if staged != live:' '        if False:'
m "P3 NO-COMPARE reads as MATCH" "$BX" '        if self.pieces <= 0:' '        if False:'
m "P4 authoritative falls back to disk" "$WU" \
  '            if wx.exchange_armed() and wx.inject_authoritative():' \
  '            if False:'
m "P5 shadow writes the live weights" "$BX" '                if comparing:' '                if False:'
m "P6 the mode is re-read in the loop" "$BX" \
  '    comparing = mode == wx.INJECT_SHADOW' \
  '    comparing = wx.inject_mode() == wx.INJECT_SHADOW'
m "P7 an unknown mode arms authority" "$WX" \
  '    return value if value in INJECT_CHOICES else INJECT_SHADOW' \
  '    return value if value in INJECT_CHOICES else INJECT_AUTHORITATIVE'
m "P8 the compare slot is borrowed, not priced" "$BX" \
  '    slots = int(depth) + (1 if comparing else 0)' '    slots = int(depth)'

echo "BASELINE AFTER: $(run)"
verify_all
