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

WT="${WT:-/spinning/wt-desk-b4g}"
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

CE=python/sglang/srt/weg2/weight_exchange.py
RT=python/sglang/srt/weg2/launcher.py
T="test/registered/unit/weg2/test_weg2_coverage_verdict_1273.py
test/registered/unit/weg2/test_weg2_xchg_reserve_1273.py
test/registered/unit/weg2/test_weg2_xchg_plan_provider_1273.py"

if [ "${1:-}" = "--selfcheck" ]; then
  PLANT="# MUTANT-HARNESS-SELFCHECK-PLANTED-$$"
  cd "$WT" || exit 2
  if ! git diff --quiet -- "$CE"; then
    echo "SELFCHECK SKIPPED: $CE is already dirty; run on a clean file"; exit 2
  fi
  printf '\n%s\n' "$PLANT" >> "$CE"
  snapshot "$CE"
  mutate "$CE" 'def coverage_leg_decision(' 'def coverage_leg_decision_MUTANT('
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

# --- THE danger direction: the refusal silent again ------------------------
m "M1 W84 is silent again (the verdict computed and dropped)" "$CE" \
  '    vote = boot_vote()
    if vote is None or vote.ok:' \
  '    vote = boot_vote()
    if True:'
m "M2 shadow becomes a STOP (the flip is lost to a report-only finding)" "$CE" \
  '    if inject_authoritative():' \
  '    if True:'
m "M3 authoritative becomes report-only (undefined weights served)" "$CE" \
  '    if inject_authoritative():' \
  '    if False:'
m "M4 exempt starts hiding a real uncovered tensor" "$CE" \
  '        return not (self.uncovered or self.short or self.missing)' \
  '        return not (self.short or self.missing)'
m "M5 a partially-tiled parameter is exempted as zerofill" "$CE" \
  '    return int(planned_bytes) == 0 and int(live_bytes) > 0' \
  '    return int(live_bytes) > 0'
m "M6 a storage-less parameter is exempted too" "$CE" \
  '    return int(planned_bytes) == 0 and int(live_bytes) > 0' \
  '    return int(planned_bytes) == 0'
m "M7 zerofill names are omitted again (absent by design == forgotten)" "$CE" \
  '        add = 0 if getattr(d, "kind", None) == ZEROFILL else int(getattr(d, "nbytes", 0))
        bucket[name] = bucket.get(name, 0) + add' \
  '        if getattr(d, "kind", None) == ZEROFILL:
            continue
        bucket[name] = bucket.get(name, 0) + int(getattr(d, "nbytes", 0))'
m "M8 zerofill starts contributing its bytes" "$CE" \
  '        add = 0 if getattr(d, "kind", None) == ZEROFILL else int(getattr(d, "nbytes", 0))' \
  '        add = int(getattr(d, "nbytes", 0))'
m "M9 the uncovered tensors stop being named" "$CE" \
  '        for ln in rows[tag].uncovered_lines():
            emit(ln)' \
  '        pass'
m "M11 the reserve swallows the residual (serving constants stand in)" "$RT" \
  '            raise xchg_residency.Weg2XchgResidencyUnarmable(' \
  '            out[c.uuid] = DC_MEASURED_D_5090_MIB; raise SystemExit(0) if False else None; raise xchg_residency.Weg2XchgResidencyUnarmable('
m "M12 the named terms collapse to the region alone" "$RT" \
  '    named = int(XCHG_RESIDENT_REGION_MIB) + slots_mib' \
  '    named = int(XCHG_RESIDENT_REGION_MIB)'

echo "BASELINE AFTER: $(run)"
verify_all
