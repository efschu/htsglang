#!/usr/bin/env bash
# #770/#812 mutation check for the per-rank floor clamp. Each mutation is a
# plausible WRONG implementation in the direction of the defect. All must die.
# Restores the file unconditionally on exit.
set -u

ROOT=/spinning/wt-770-solver
MOD=$ROOT/python/sglang/srt/managers/kv_backing_relief.py
PY=/spinning/htsglang-gpu/.venv/bin/python
BAK=$(mktemp)
cp "$MOD" "$BAK"
trap 'cp "$BAK" "$MOD"; rm -f "$BAK"' EXIT

run_suite() {
  cd "$ROOT" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT/python" \
    timeout 200 "$PY" -m pytest test/srt/test_floor_local_cap_812.py -q \
    --no-header --color=no -p no:cacheprovider 2>&1 | tail -1
}

survived=0
mutate() {
  local name="$1" from="$2" to="$3"
  cp "$BAK" "$MOD"
  if ! grep -qF -- "$from" "$MOD"; then
    echo "MUTANT $name: PATTERN NOT FOUND -- harness stale, treat as FAILURE"
    survived=$((survived + 1)); return
  fi
  "$PY" - "$MOD" "$from" "$to" <<'PY'
import sys
p, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
open(p, "w").write(s.replace(a, b, 1))
PY
  local out; out=$(run_suite)
  if echo "$out" | grep -q "failed"; then
    echo "MUTANT $name: KILLED   ($out)"
  else
    echo "MUTANT $name: SURVIVED ($out)   <-- LAW IS UNASSERTED"
    survived=$((survived + 1))
  fi
}

echo "=== baseline (must be green) ==="
cp "$BAK" "$MOD"; run_suite
echo
echo "=== mutants ==="

# MA1 -- THE DANGEROUS ONE: the floor is lowered to the cap. This is the fix I
# first shipped and had to withdraw -- it authorises a cap BELOW the live set,
# which test_residency_cap_flip_levelling_792 exists to forbid.
mutate MA1-floor-lowered-to-cap \
  '        if floor_exceeds_local_cap(floor, cap):' \
  '        if floor_exceeds_local_cap(floor, cap) and setattr(self, "_x", 0) is None and (floor := cap) is not None:'

# MA2 -- the detector reports a FULL pool as under-backed, turning a healthy
# rank into a defect report.
mutate MA2-healthy-misread-as-defect \
  'return int(current_rows) > 0 and int(floor_rows) > int(current_rows)' \
  'return int(current_rows) > 0 and int(floor_rows) >= int(current_rows)'

# MA3 -- the detector goes silent, which is the state before this ticket: an
# under-backed rank freezes the group and nothing says why.
mutate MA3-detector-always-false \
  'return int(current_rows) > 0 and int(floor_rows) > int(current_rows)' \
  'return False'

# MA4 -- the floor stops covering the live set by one row.
mutate MA4-floor-off-by-one \
  'int(max_live) + 1 + self._margin_rows + self._admission_reserve_rows,' \
  'int(max_live) + 0 + self._margin_rows + self._admission_reserve_rows,'

# MA5 IS AN EQUIVALENT MUTANT AND IS DELIBERATELY NOT RUN.
#   Removing the `min(_SHRINK_SCALE, ...)` guard in _floor_ppm changes nothing
#   observable: the early return already handles floor >= current, so the
#   expression is only reached with floor < current, where
#   ceil(floor*1e6/current) <= 999999 < _SHRINK_SCALE. Verified numerically.

echo
if [ "$survived" -eq 0 ]; then echo "ALL MUTANTS KILLED"; else echo "$survived MUTANT(S) SURVIVED"; fi
exit "$survived"
