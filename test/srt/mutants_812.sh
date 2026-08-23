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

# MA1 -- THE ONE THE OPERATOR NAMED: the floor goes back to being ABSOLUTE,
# i.e. never derived against the rank's own cap. PP1's 102.9% returns.
mutate MA1-floor-absolute-again \
  '        if cap > 0 and floor > cap:' \
  '        if False:'

# MA2 -- the clamp overshoots and reports a FULL pool as under-backed, which
# would turn a healthy rank into a defect report.
mutate MA2-healthy-misread-as-defect \
  'return int(current_rows) > 0 and int(floor_rows) > int(current_rows)' \
  'return int(current_rows) > 0 and int(floor_rows) >= int(current_rows)'

# MA3 -- the clamp is applied but rounds UP past the cap, so the floor still
# exceeds the backing by up to one page.
mutate MA3-clamp-rounds-up-past-cap \
  'floor = int(math.floor(cap / page) * page)' \
  'floor = int(math.ceil(cap / page) * page)'

# MA4 -- a broken cap probe silently clamps the floor to zero instead of
# leaving it alone, which would let a rank shrink below its live set.
mutate MA4-broken-probe-clamps-to-zero \
  '            cap = 0
        if cap > 0 and floor > cap:' \
  '            cap = 1
        if cap > 0 and floor > cap:'

# MA5 IS AN EQUIVALENT MUTANT AND IS DELIBERATELY NOT RUN.
#   Removing the `min(_SHRINK_SCALE, ...)` guard in _floor_ppm changes nothing
#   observable: the early return already handles floor >= current, so the
#   expression is only reached with floor < current, where
#   ceil(floor*1e6/current) <= 999999 < _SHRINK_SCALE. Verified numerically
#   across several (floor, current) pairs: max observed 999999.
#   The guard is kept as defence-in-depth against a future edit to the early
#   return, but a harness that lists an unkillable mutant reports a permanent
#   false gap, so it is recorded here instead of run.

echo
if [ "$survived" -eq 0 ]; then echo "ALL MUTANTS KILLED"; else echo "$survived MUTANT(S) SURVIVED"; fi
exit "$survived"
