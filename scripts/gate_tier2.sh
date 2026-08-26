#!/usr/bin/env bash
# #860 tier-2 gate runner. ONE definition of the gate, two engines.
#
# WHY THIS EXISTS AS A SCRIPT. The gate command has been retyped by hand in
# every window result since W35, and the flags are not decoration: `-p
# no:randomly` is what makes the failure set comparable between runs at all,
# and `PYTHONPATH` pointing at the WORKTREE's python is the difference between
# gating your fix and gating the tree you happen to be standing in
# (Worktree-PYTHONPATH-Messfalle). A hand-typed gate is a gate that eventually
# measures something else.
#
# USAGE
#   scripts/gate_tier2.sh                  # serial, the canary form
#   scripts/gate_tier2.sh -n 8             # xdist, loadscope, 8 workers
#   GATE_PATH=test/registered/unit/managers scripts/gate_tier2.sh -n 8
#
# CUDA_VISIBLE_DEVICES IS FORCED EMPTY AND NOT OVERRIDABLE FROM THE
# ENVIRONMENT. The gate runs at the desk, often while a GPU window holds the
# cards; a suite that quietly initialises a context takes VRAM from the boot
# that is running. If a test genuinely needs a device it must be excluded by
# name and reported, never accommodated by loosening this line.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${GATE_PY:-/spinning/htsglang-gpu/.venv/bin/python3}"
GATE_PATH="${GATE_PATH:-test/registered/unit/managers}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${GATE_OUT:-/tmp/gate_tier2_${STAMP}.log}"

# --dist loadscope keeps every MODULE whole on one worker. That is not a
# performance choice: this suite's modules carry module-level state (the #249
# `set_default_device` shim, module-scoped fixtures, monkeypatched singletons),
# and splitting a module across workers would change what each test sees. With
# loadscope a worker's view of one module is exactly the serial view of it.
DIST=()
if [[ "${1:-}" == -n* ]]; then
  DIST=("$1" --dist loadscope)
  shift
fi

cd "$ROOT"
echo "# gate_tier2 ${STAMP}"
echo "# tree     $ROOT"
echo "# commit   $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "# engine   ${DIST[*]:-serial}"
echo "# path     $GATE_PATH"
echo "# log      $OUT"

set +e
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT/python" \
  /usr/bin/time -f 'GATE_WALL_SECONDS %e' \
  "$PY" -m pytest "$GATE_PATH" \
    -q -p no:randomly -p no:cacheprovider --durations=10 \
    "${DIST[@]}" "$@" > "$OUT" 2>&1
rc=$?
set -e

# THE FAILURE SET IS THE VERDICT, NOT THE COUNT. Two runs with the same number
# of failures and different names are not the same gate, and a count is what a
# tired reader compares. Printed sorted so a diff between two runs is a diff of
# names.
echo "--- failure set ---"
grep -aoE '^FAILED [^ ]+' "$OUT" | sed 's/^FAILED //' | sort || true
echo "--- totals ---"
grep -aoE '[0-9]+ (failed|passed|skipped|error[s]?)' "$OUT" | tail -6 || true
grep -a 'GATE_WALL_SECONDS' "$OUT" || true
exit $rc
