#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #631/#656 capacity ladder step, successor 25.
#
# WHY THIS EXISTS RATHER THAN REUSING s24_green_run.sh. That recipe is
# tuned to be COMPARABLE to the 68-min bench row, so its prefill leg is
# 12000 tokens and its occupancy stays low. Occupancy is exactly what a
# capacity step has to drive: the seam's staging term scales with the
# LIVE SET, so a load that never fills the pool cannot test the term that
# decides the ceiling. Successor 25's 14-minute run at pool 410000 held
# the corridor at 2707 MiB on the binding card -- a pass for the step,
# but no evidence about the ceiling, because the live set never got near
# 410000.
#
# This leg pushes occupancy instead: long prefills issued faster than
# they retire, so several coexist and the pool fills toward
# max_running_requests x prefill-tokens.
#
# Usage: bash scripts/s25_capacity_step.sh <pool> <minutes> [outdir]
set -uo pipefail

POOL="${1:?pool size, e.g. 500000}"
MINS="${2:-16}"
OUT="${3:-/spinning/evidence-631/s25/cap${POOL}}"
PY=/spinning/htsglang-gpu/.venv/bin/python
export PYTHONPATH=/spinning/wt-631-routea/python

mkdir -p "$OUT"
GRACE=$(python3 -c "print(int($MINS*60)+240)")

echo "capacity step: pool=$POOL ${MINS} min -> $OUT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/started_at"
echo "$POOL" > "$OUT/pool"

# The corridor must cover every other leg including its ramp. ONE csv per
# configuration -- a second step must not append to a previous step's file
# or the minimum becomes unattributable.
setsid nohup bash scripts/corridor_sample.sh "$GRACE" "$OUT/corridor.csv" \
    > "$OUT/corridor.stderr" 2>&1 < /dev/null &
echo "corridor pid $!"

# High-occupancy leg: long prefills issued faster than they retire.
setsid nohup $PY scripts/soak_631_mixed_load.py \
    --minutes "$MINS" --decode-streams 2 \
    --prefill-tokens 110000 --prefill-period 3 \
    > "$OUT/soak.log" 2>&1 < /dev/null &
echo "soak pid $!"

echo "legs launched; corridor -> $OUT/corridor.csv"
