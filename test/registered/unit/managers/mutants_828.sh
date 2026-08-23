#!/usr/bin/env bash
# #828 mutation check. Each mutation is a plausible WRONG implementation in the
# direction of the defect this ticket exists to close, and every one MUST make
# the suite fail. A survivor means the law is asserted nowhere.
#
# Three files carry the fix, so three files are mutated:
#   memory_pool.py         the dial that latched (backing vs exposed size)
#   funding_authority.py   law 2, delivered-not-promised
#   phase_flip_runtime.py  the wiring that carries the measurement
#
# Hermetic: no CUDA, no NVML. Restores every file unconditionally on exit.
set -u

ROOT=${ROOT:-/spinning/htsglang/.claude/worktrees/agent-a598d5da61d3598a3}
POOL=$ROOT/python/sglang/srt/mem_cache/memory_pool.py
AUTH=$ROOT/python/sglang/srt/managers/funding_authority.py
FLIP=$ROOT/python/sglang/srt/managers/phase_flip_runtime.py
PY=${PY:-/bin/python3}

BAK_DIR=$(mktemp -d)
cp "$POOL" "$BAK_DIR/pool.py"
cp "$AUTH" "$BAK_DIR/auth.py"
cp "$FLIP" "$BAK_DIR/flip.py"
restore() {
  cp "$BAK_DIR/pool.py" "$POOL"
  cp "$BAK_DIR/auth.py" "$AUTH"
  cp "$BAK_DIR/flip.py" "$FLIP"
}
trap 'restore; rm -rf "$BAK_DIR"' EXIT

run_suite() {
  cd "$ROOT" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT/python" \
    timeout 300 "$PY" -m pytest \
      test/registered/unit/managers/test_funder_delivers_828.py -q 2>&1 |
    grep -E "passed|failed|error" | tail -1
}

survived=0
mutate() {
  local name="$1" file="$2" from="$3" to="$4"
  restore
  if ! grep -qF -- "$from" "$file"; then
    echo "MUTANT $name: PATTERN NOT FOUND -- harness is stale, treat as FAILURE"
    survived=$((survived + 1))
    return
  fi
  "$PY" - "$file" "$from" "$to" <<'PY'
import sys
p, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
open(p, "w").write(s.replace(a, b, 1))
PY
  local out
  out=$(run_suite)
  if echo "$out" | grep -qE "failed|error"; then
    echo "MUTANT $name: KILLED   ($out)"
  else
    echo "MUTANT $name: SURVIVED ($out)   <-- LAW IS UNASSERTED"
    survived=$((survived + 1))
  fi
}

echo "=== baseline (must be green) ==="
restore
run_suite

echo
echo "=== mutants ==="

# M1 -- THE DEFECT ITSELF, restored: decommit only when the target is below the
# EXPOSED size. This is the shipped behaviour that made boot_827's 14-granule
# shrink return zero bytes and latched the dial permanently.
mutate M1-decommit-branches-on-exposed-size "$POOL" \
  'released = int(owner.shrink(n)) if n < backed else 0' \
  'released = int(owner.shrink(n)) if n < prev else 0'

# M2 -- the mapping half of the same defect: finalize against the exposed size,
# so a target above the backing but below the size never maps its rows.
mutate M2-finalize-branches-on-exposed-size "$POOL" \
  '        if n > backed:
            owner.finalize(n)' \
  '        if n > prev:
            owner.finalize(n)'

# M3 -- THE DANGER DIRECTION. Zero only on a backing grow, which is what a
# careless re-route of the branch produces: rows re-exposed by a call that
# SHRANK the backing are handed out still carrying another request's KV.
mutate M3-re-exposed-rows-not-zeroed "$POOL" \
  '        if n > prev:
            buffers = list(self.k_buffer) + list(self.v_buffer)' \
  '        if n > backed:
            buffers = list(self.k_buffer) + list(self.v_buffer)'

# M4 -- the early return keys on the exposed size alone, so the specimen row
# (size already equal to the target, backing far above it) returns 0 untouched.
mutate M4-early-return-ignores-the-backing "$POOL" \
  'if n == prev and n == backed:' \
  'if n == prev:'

# M5 -- law 2 disconnected again: the measurement is accepted and discarded.
mutate M5-delivery-not-derated "$AUTH" \
  '        cache_derate_den = cache_promised' \
  '        cache_derate_den = 0'

# M6 -- the clamp that keeps a noisy probe from raising on the refusal path.
mutate M6-delivery-above-promise-unclamped "$AUTH" \
  '        cache_derate_num = min(
            cache_promised, max(0, int(allocator_cache_delivered_bytes))
        )' \
  '        cache_derate_num = max(0, int(allocator_cache_delivered_bytes))'

# M7 -- the wiring is cut: the census builds its authority without the
# measurement, exactly as it did before this ticket.
mutate M7-census-drops-the-measurement "$FLIP" \
  '                allocator_cache_delivered_bytes=delivered,' \
  '                allocator_cache_delivered_bytes=None,'

# M8 -- the ratio is taken against the POST-reclaim cache instead of what the
# draw actually saw, so the derate is computed against a denominator the draw
# never had.
mutate M8-derate-against-the-wrong-denominator "$FLIP" \
  '                cached = int(promised)' \
  '                cached = int(cached)'

echo
if [ "$survived" -eq 0 ]; then
  echo "ALL MUTANTS KILLED"
else
  echo "$survived MUTANT(S) SURVIVED"
fi
exit "$survived"
