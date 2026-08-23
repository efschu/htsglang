#!/usr/bin/env bash
# #829 mutation check. Each mutation is a plausible WRONG implementation in the
# direction of the defect, and every one MUST make the suite fail.
# Hermetic: CPU gloo only, CVD="". Restores the file unconditionally on exit.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Four levels up from test/registered/unit/mem_cache/ is the tree root.
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
MOD=$ROOT/python/sglang/srt/mem_cache/hicache_collective.py
PY=${PY:-/spinning/htsglang-gpu/.venv/bin/python}
BAK=$(mktemp); cp "$MOD" "$BAK"
trap 'cp "$BAK" "$MOD"; rm -f "$BAK"' EXIT

run_suite() {
  cd "$ROOT" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT/python" \
    timeout 300 "$PY" -m pytest \
      test/registered/unit/mem_cache/test_bounded_wait_pair_survives_829.py \
      test/registered/unit/mem_cache/test_collective_discriminator_825.py -q 2>&1 |
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

# M3 -- a dead peer reported as a timeout. #734's law is now carried by CONTROL
# FLOW rather than by a time ratio (#825): everything reaching the except IS a
# transport failure, so the TYPE raised there is the whole discriminator.
# Mutating it reinstates the self-contradicting line ("within 600s (waited
# 34.3s)") that sent an operator hunting a slow rank while the corpse was
# elsewhere. Anchored on the message text, which is unique to this raise.
mutate M3-dead-peer-reported-as-timeout \
'        raise HiCacheCollectiveError(
            f"HiCache control collective ' \
'        raise HiCacheCollectiveTimeoutError(
            f"HiCache control collective '

# M4 -- the #825 guard neutered. The control-flow discriminator is sound only
# while the bound stays under the group own timeout; above it the parked wait
# raises genuine expiries and M3 defect returns by the back door. A guard that
# never refuses is the same as no guard.
mutate M4-bound-guard-never-refuses \
'    if timeout_s < limit:
        return' \
'    if True:
        return'

echo
if [ "$survived" -eq 0 ]; then echo "ALL MUTANTS KILLED"; else echo "$survived MUTANT(S) SURVIVED"; fi
exit "$survived"
