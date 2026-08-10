#!/usr/bin/env bash
# #631: the LOADED corridor run at B=16 -- the gate for defaulting the
# commit chunk on.
#
# WHY --max-tokens IS THE OCCUPANCY KNOB HERE, and the first attempt got
# it wrong. s26_fill_load holds K streams that each prefill a long unique
# context and then decode. With a SHORT decode the stream retires almost
# as soon as its prefill lands, and under strict purity the PP prefills
# serialise, so at any instant roughly ONE stream's KV is resident --
# measured 105,757 slots of a 500,000 pool, 21%, with four streams asked
# for. The context length sets how much a resident stream holds; the
# DECODE LENGTH sets how long it stays resident, and occupancy needs
# both. A long decode keeps every stream's context alive while its peers
# prefill, which is what makes the four overlap.
set -u
D=${D:-/spinning/evidence-631/s27/loaded_b16_run2}
MINUTES=${MINUTES:-16}
STREAMS=${STREAMS:-4}
CTX=${CTX:-150000}
MAXTOK=${MAXTOK:-8000}
TUNE=${TUNE:-/spinning/evidence-631/s27/seam_tune.json}
LOG=${LOG:-/spinning/serving-30030.boot.log}
PY=/spinning/htsglang-gpu/.venv/bin/python

mkdir -p "$D"
echo '{"row_blocks": 16}' > "$TUNE"
grep -c "PHASE-FLIP DONE" "$LOG" > "$D/flips_before"
grep -c "FLIP ABANDONED" "$LOG" > "$D/abandons_before"

"$PY" scripts/s26_fill_load.py --minutes "$MINUTES" --streams "$STREAMS" \
  --context-tokens "$CTX" --max-tokens "$MAXTOK" > "$D/load.log" 2>&1 &
LOADPID=$!

# Let the streams get their first contexts resident before the corridor's
# minimum starts counting; a cold sampler measures the ramp, not the run.
sleep 120
"$PY" scripts/route_a_631_corridor.py --interval-ms 100 --seconds 720 \
  --floor 1024 --out "$D/corridor.csv" > "$D/corridor.log" 2>&1

wait $LOADPID 2>/dev/null || true
{
  echo "=== flips ==="
  echo "before=$(cat "$D/flips_before") after=$(grep -c 'PHASE-FLIP DONE' "$LOG")"
  echo "abandons before=$(cat "$D/abandons_before") after=$(grep -c 'FLIP ABANDONED' "$LOG")"
  echo "=== occupancy (live slots at flip time) ==="
  grep -oE "[0-9]+ live slots" "$LOG" | awk '{print $1}' | sort -n | tail -3
  echo "=== staging reserved, last flip ==="
  grep "PHASE-FLIP DONE" "$LOG" | tail -3 \
    | sed -E 's/.*(PP[0-9]).*staging reserved ([0-9.]+) MiB.*/  \1 staging=\2/'
  echo "=== corridor ==="
  grep -E "per-card MINIMUM|CORRIDOR HELD" "$D/corridor.log"
} | tee "$D/summary.txt"
echo "LOADED-RUN-COMPLETE"
