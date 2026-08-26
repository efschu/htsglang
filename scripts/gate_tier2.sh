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

# --dist loadfile, NEVER loadscope (#868 §0.2, corrected #895).
#
# The intent below was always "keep every MODULE whole on one worker", because
# this suite's modules carry module-level state (the #249 `set_default_device`
# shim, module-scoped fixtures, monkeypatched singletons). `loadscope` does NOT
# do that here: it groups plain test functions by module but test METHODS by
# CLASS, and this suite is unittest/TestCase based throughout, so it splits a
# single FILE across workers. Measured on one module with nothing else in the
# run: test_chunked_commitment_701.py alone is 17 passed serially and under
# `-n 2 --dist loadfile`, and 4 failed under `-n 2 --dist loadscope`.
# `loadfile` is the grouping this comment always meant.
DIST=()
if [[ "${1:-}" == -n* ]]; then
  DIST=("$1" --dist loadfile)
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
#
# Extraction goes through gate_partition_lib (#895), not through a grep. The
# grep that stood here matched `^FAILED` only, which is three known ways of
# printing a smaller failure set than the run produced: an ANSI colour code
# sits in front of `FAILED` and defeats the anchor, a parametrised subtest
# emits `SUBFAILED`, and a collection or fixture error emits `ERROR`. The lib
# also enforces the tally: names extracted must equal names counted, or the
# EXTRACTION is broken and this script says so instead of printing a verdict
# it cannot substantiate.
echo "--- failure set ---"
"$PY" - "$ROOT/scripts" "$OUT" <<'PYEOF'
import sys

sys.path.insert(0, sys.argv[1])
from gate_partition_lib import parse_log  # noqa: E402

res = parse_log(sys.argv[2])
for name in sorted(res.all_names):
    print(name)
print("--- totals ---")
print(f"counts={res.counts} names={len(res.all_names)} wall={res.wall}s")
if not res.tally_ok:
    print(f"EXTRACTION BROKEN: {res.tally_note}")
    print("The failure set above is NOT the verdict -- it is what could be read.")
    raise SystemExit(3)
PYEOF
tally_rc=$?
grep -a 'GATE_WALL_SECONDS' "$OUT" || true
if [[ $tally_rc -ne 0 ]]; then
  exit $tally_rc
fi
exit $rc
