#!/usr/bin/env bash
# #656: the loaded corridor run that can say WHICH PHASE binds.
#
# WHY THIS EXISTS AND s27_loaded_b16_run.sh DOES NOT SUFFICE. That script
# reports a per-card corridor MINIMUM, which is a worst-TIME quantity over
# both phases at once. Every spill design in this chain is phase-scoped --
# the drafter is spillable during PP because it is idle during PP, the PP
# weight shard is spillable during TP for the mirror reason -- and a
# phase-scoped spill is worth its FULL payload if the binding minimum falls
# in the phase where the asset is cold, and worth EXACTLY ZERO MiB if it
# falls in the other one. The aggregate minimum cannot tell those apart.
# Pricing a spill before knowing which phase binds is how this chain
# produced six capacity headlines that did not survive.
#
# So this run differs from s27's in exactly one way: the corridor sampler
# also writes its RAW per-sample series, which s21_phase_corridor.py then
# cuts into PP and TP windows using the serving log's own re-dispatch
# lines.
#
# Occupancy knob unchanged and it is load-bearing: context length sets how
# MUCH a stream holds, decode length sets how LONG it stays resident. Short
# decodes retire each stream right after its prefill and, under strict
# purity, the PP prefills serialise -- so roughly ONE stream is resident and
# the pool never fills. Hence --max-tokens 8000.
set -u
D=${D:-/spinning/evidence-631/s28/loaded_phase}
MINUTES=${MINUTES:-16}
STREAMS=${STREAMS:-4}
CTX=${CTX:-150000}
MAXTOK=${MAXTOK:-8000}
BLOCKS=${BLOCKS:-16}
TUNE=${TUNE:-/spinning/evidence-631/s27/seam_tune.json}
LOG=${LOG:-/spinning/serving-30030.boot.log}
PY=/spinning/htsglang-gpu/.venv/bin/python

mkdir -p "$D"
echo "{\"row_blocks\": $BLOCKS}" > "$TUNE"
grep -c "PHASE-FLIP DONE" "$LOG" > "$D/flips_before"
grep -c "FLIP ABANDONED" "$LOG" > "$D/abandons_before"

"$PY" scripts/s26_fill_load.py --minutes "$MINUTES" --streams "$STREAMS" \
  --context-tokens "$CTX" --max-tokens "$MAXTOK" > "$D/load.log" 2>&1 &
LOADPID=$!

# Let the streams get their first contexts resident before the corridor's
# minimum starts counting; a cold sampler measures the ramp, not the run.
sleep 120
"$PY" scripts/route_a_631_corridor.py --interval-ms 100 --seconds 720 \
  --floor 1024 --out "$D/corridor.json" --series "$D/series.csv" \
  > "$D/corridor.log" 2>&1

wait $LOADPID 2>/dev/null || true

{
  echo "=== flips ==="
  echo "before=$(cat "$D/flips_before") after=$(grep -c 'PHASE-FLIP DONE' "$LOG")"
  echo "abandons before=$(cat "$D/abandons_before") after=$(grep -c 'FLIP ABANDONED' "$LOG")"
  echo "=== occupancy (live slots at flip time) ==="
  grep -oE "[0-9]+ live slots" "$LOG" | awk '{print $1}' | sort -n | tail -3
  echo "=== corridor (both phases together) ==="
  grep -E "per-card MINIMUM|headroom above|CORRIDOR HELD" "$D/corridor.log"
} | tee "$D/summary.txt"

echo "=== PHASE ATTRIBUTION ===" | tee -a "$D/summary.txt"
"$PY" scripts/s21_phase_corridor.py --corridor "$D/series.csv" \
  --log "$LOG" --margin 1.5 2>&1 | tee "$D/phase.txt"

echo "PHASE-RUN-COMPLETE"
