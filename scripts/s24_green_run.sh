#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #631/#656 green run: the mixed acceptance load, held for MINUTES, with
# the corridor sampled throughout.
#
# THE RECIPE IS THE 68-MIN ROW'S (PROD_BRINGUP_BENCH 2d) so the two are
# comparable: a bs=4 soak, a long-prefill ladder, a decode probe, and
# real agent traffic through the router, all concurrently. That run held
# the corridor but logged 51 FLIP ABANDONED lines (17 events x 3 ranks);
# this one exists to show those gone once the seam is waved.
#
# Agent traffic is NOT started here -- it is launched from the operator
# session so the requests carry a real workload rather than a synthetic
# one, and so "did it carry traffic" is answered from the serving log
# rather than from this script's intentions.
#
# Usage: bash scripts/s24_green_run.sh <minutes> <outdir>
set -uo pipefail

MINS="${1:-65}"
OUT="${2:-/spinning/evidence-631/s24/green}"
PORT="${PORT:-30030}"
PY=/spinning/htsglang-gpu/.venv/bin/python
export PYTHONPATH=/spinning/wt-631-routea/python

mkdir -p "$OUT"
SECS=$(python3 -c "print(int($MINS*60))")
GRACE=$(python3 -c "print(int($MINS*60)+300)")

echo "green run: ${MINS} min -> $OUT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/started_at"

# Corridor first: it must cover every other leg, including their ramp.
setsid nohup bash scripts/corridor_sample.sh "$GRACE" "$OUT/corridor.csv" \
    > "$OUT/corridor.stderr" 2>&1 < /dev/null &
echo "corridor pid $!"

# The bs=4 soak: decode streams plus a periodic prefill, the steady load
# the flip policy actually oscillates against.
setsid nohup $PY scripts/soak_631_mixed_load.py \
    --minutes "$MINS" --decode-streams 2 \
    --prefill-tokens 12000 --prefill-period 8 \
    > "$OUT/soak.log" 2>&1 < /dev/null &
echo "soak pid $!"

# The long-prefill ladder, repeated for the whole window. Each pass puts a
# six-figure token prefill through the PP phase, which is what makes the
# flip's staging demand large enough to be interesting.
setsid nohup bash -c "
  end=\$(( \$(date +%s) + $SECS ))
  i=0
  while [ \"\$(date +%s)\" -lt \"\$end\" ]; do
    i=\$((i+1))
    $PY scripts/route_a_631_prefill_ladder.py --port $PORT \
        --rungs 2048,32768,111405 --draws 1 \
        --out '$OUT/ladder_\$i.json' >> '$OUT/ladder.log' 2>&1
    sleep 20
  done
" > "$OUT/ladder.stderr" 2>&1 < /dev/null &
echo "ladder pid $!"

# The decode probe, repeated: it is the only leg that reports accept
# length and the decode batch-size histogram.
setsid nohup bash -c "
  end=\$(( \$(date +%s) + $SECS ))
  i=0
  while [ \"\$(date +%s)\" -lt \"\$end\" ]; do
    i=\$((i+1))
    $PY scripts/s22_decode_probe.py --port $PORT --concurrency 4 \
        --rounds 2 --max-new 400 >> '$OUT/decode_probe.log' 2>&1
    sleep 30
  done
" > "$OUT/decode.stderr" 2>&1 < /dev/null &
echo "decode pid $!"

echo "all legs launched; corridor -> $OUT/corridor.csv"
