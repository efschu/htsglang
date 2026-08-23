#!/usr/bin/env bash
# #829 mutation check. Each mutation is a plausible WRONG implementation in the
# direction of the defect, and every one MUST make the suite fail.
# Hermetic: CPU gloo only, CVD="". Restores the file unconditionally on exit.
set -u
ROOT=${ROOT:-/spinning/htsglang/.claude/worktrees/agent-a598d5da61d3598a3}
MOD=$ROOT/python/sglang/srt/mem_cache/hicache_collective.py
PY=${PY:-/spinning/htsglang-gpu/.venv/bin/python}
BAK=$(mktemp); cp "$MOD" "$BAK"
trap 'cp "$BAK" "$MOD"; rm -f "$BAK"' EXIT

run_suite() {
  cd "$ROOT" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT/python" \
    timeout 300 "$PY" -m pytest \
      test/registered/unit/mem_cache/test_bounded_wait_pair_survives_829.py -q 2>&1 |
    sed 's/\x1b\[[0-9;]*m//g' | grep -E "passed|failed|error" | tail -1
}

survived=0
mutate() {
  local name="$1" from="$2" to="$3"
  cp "$BAK" "$MOD"
  if ! grep -qF -- "$from" "$MOD"; then
    echo "MUTANT $name: PATTERN NOT FOUND -- harness stale, treat as FAILURE"
    survived=$((survived+1)); return
  fi
  "$PY" - "$MOD" "$from" "$to" <<'PY'
import sys
p, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
open(p, "w").write(s.replace(a, b, 1))
PY
  local out; out=$(run_suite)
  if echo "$out" | grep -qE "failed|error"; then
    echo "MUTANT $name: KILLED   ($out)"
  else
    echo "MUTANT $name: SURVIVED ($out)   <-- LAW IS UNASSERTED"
    survived=$((survived+1))
  fi
}

echo "=== baseline (must be green) ==="; cp "$BAK" "$MOD"; run_suite
echo; echo "=== mutants ==="

# M1 -- THE DEFECT ITSELF, restored: hand the deadline back to the Work. This
# is the shipped #630 line, and on expiry it closes the gloo pair.
mutate M1-deadline-handed-to-the-work \
'    parked = ParkedWait(work, label)
    try:
        completed = parked.join(timeout_s)' \
'    import datetime as _dt
    try:
        completed = work.wait(timeout=_dt.timedelta(seconds=timeout_s))'

# M2 -- expiry silently ignored: the terminal contract callers rely on is gone,
# so a caller would reuse a buffer whose receive is still outstanding.
mutate M2-expiry-no-longer-raises \
'    if not completed:' \
'    if False:'

# M3 -- the #734 discriminator disabled: a dead peer is reported as a timeout,
# which is the self-contradicting line ("within 600s (waited 34.3s)") that sent
# an operator looking for a slow rank while the corpse was elsewhere.
mutate M3-dead-peer-reported-as-timeout \
'        if waited < timeout_s * 0.95:' \
'        if False:'

echo
if [ "$survived" -eq 0 ]; then echo "ALL MUTANTS KILLED"; else echo "$survived MUTANT(S) SURVIVED"; fi
exit "$survived"
