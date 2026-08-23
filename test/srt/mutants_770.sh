#!/usr/bin/env bash
# #770 mutation check: each mutation below is a plausible WRONG implementation
# in the direction of the defect this module exists to catch. Every one of them
# MUST make the suite fail. A mutant that survives means the corresponding law
# is asserted nowhere and the test file is decoration.
#
# Hermetic: no CUDA, no NVML. Restores the file unconditionally on exit.
set -u

ROOT=/spinning/wt-770-solver
MOD=$ROOT/python/sglang/srt/managers/funding_authority.py
BAK=$(mktemp)
cp "$MOD" "$BAK"
trap 'cp "$BAK" "$MOD"; rm -f "$BAK"' EXIT

run_suite() {
  cd "$ROOT" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT/python" \
    timeout 200 /spinning/htsglang-gpu/.venv/bin/python -m pytest test/srt/test_funding_authority_770.py -q 2>&1 | tail -1
}

survived=0
mutate() {
  local name="$1" from="$2" to="$3"
  cp "$BAK" "$MOD"
  if ! grep -qF -- "$from" "$MOD"; then
    echo "MUTANT $name: PATTERN NOT FOUND -- mutation harness is stale, treat as FAILURE"
    survived=$((survived + 1))
    return
  fi
  python3 - "$MOD" "$from" "$to" <<'PY'
import sys
p, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
open(p, "w").write(s.replace(a, b, 1))
PY
  local out
  out=$(run_suite)
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

# M1 -- a peer veto is silently reclassified as scarcity, which re-arms the
# exponential stand-down over slack that is sitting on this very rank.
mutate M1-veto-becomes-scarcity \
  'elif vetoed_bytes > 0:
            cause = CAUSE_PEER_VETO' \
  'elif False:
            cause = CAUSE_PEER_VETO'

# M2 -- granularity rounds DOWN, reproducing the 3437-vs-8192 silent zero.
mutate M2-granule-rounds-down \
  'rounded = ((draw + gran - 1) // gran) * gran' \
  'rounded = (draw // gran) * gran'

# M3 -- promised capacity is trusted at face value (the claimed=0 accounting lie).
mutate M3-no-derating \
  'return int(self.available_bytes) * int(self.derate_num) // int(self.derate_den)' \
  'return int(self.available_bytes)'

# M4 -- the user reserve becomes fundable, violating DESIGN_584 R2 / #582.
mutate M4-user-reserve-fundable \
  'if post.is_user_reserve:
            raise FundingError(' \
  'if False:
            raise FundingError('

# M5 -- the group agrees the SMALLEST floor, which fails to clear the live set
# of the rank that needs the most.
mutate M5-min-instead-of-max-floor \
  'return max(int(f) for f in per_rank_floors)' \
  'return min(int(f) for f in per_rank_floors)'

# M6 -- negative slack leaks out for a rank whose floor exceeds its own cap
# (PP1: 128549 against 124928), which then propagates as a constraint.
mutate M6-negative-slack \
  'return max(0, int(cap) - int(uniform_floor))' \
  'return int(cap) - int(uniform_floor)'

# M7 -- the unsatisfiable-floor boundary goes strict, so an exactly-reachable
# configuration is condemned.
mutate M7-floor-band-off-by-one \
  'if need <= ceiling:' \
  'if need < ceiling:'

# M8 -- zero-draws are dropped from the verdict, recreating corridor_guard's
# `if got <= 0: continue` and with it the "[nothing]" ambiguity.
mutate M8-drop-zero-draws \
  'draws.append(Draw(post.name, post.tier, remaining, drawn, reason))' \
  'draws.append(Draw(post.name, post.tier, remaining, drawn, reason)) if drawn > 0 else None'

# M9 -- the arming-floor solver accepts a reserve one MiB past the headroom,
# so an unreachable watermark is reported as fine.
mutate M9-arming-floor-off-by-one \
  'if requested <= max_reserve:' \
  'if requested <= max_reserve + 1:'

# M10 -- the band CEILING is quietly moved to make the floor fit, violating the
# hard user rule the solver exists to respect.
mutate M10-band-ceiling-moved \
  'headroom = ceiling - floor_base - margin' \
  'headroom = ceiling - floor_base - margin + 512'

# M11 -- a frozen break-even input stops being reported as frozen, which is the
# whole of #819.
mutate M11-frozen-input-hidden \
  'if self.pp_provenance == PROV_FROZEN:' \
  'if False:'

echo
if [ "$survived" -eq 0 ]; then
  echo "ALL MUTANTS KILLED"
else
  echo "$survived MUTANT(S) SURVIVED"
fi
exit "$survived"
