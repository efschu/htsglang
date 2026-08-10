#!/usr/bin/env bash
# #631 section 2.1: the CLEAN row-block A/B -- same direction, same boot.
#
# The first sweep alternated directions between arms, so the floor arm and
# the blocked arms were not the same measurement and some arms picked up
# the previous flip's DONE lines. Here every measured flip is pp_to_tp:
# each arm first returns the layout to PP (unmeasured), then runs the one
# flip that counts. Same boot throughout, floor arm first.
set -u
LOG=${LOG:-/spinning/serving-30030.boot.log}
TUNE=${TUNE:-/spinning/evidence-631/s27/seam_tune.json}
OUT=${OUT:-/spinning/evidence-631/s27/rowblock_clean.txt}
PORT=${PORT:-30030}
BLOCKS=${BLOCKS:-"1 4 16 32"}

flip () {  # $1 direction; waits for all three ranks to report DONE
  local before after
  before=$(grep -c "PHASE-FLIP DONE" "$LOG")
  curl -s -m 5 "http://127.0.0.1:$PORT/phase_flip" \
       -H 'Content-Type: application/json' -d "{\"direction\":\"$1\"}" >/dev/null
  for _ in $(seq 45); do
    after=$(grep -c "PHASE-FLIP DONE" "$LOG")
    [ $((after - before)) -ge 3 ] && return 0
    sleep 1
  done
  return 1
}

: > "$OUT"
for b in $BLOCKS; do
  # Return to PP with the FLOOR setting so the reset never varies between
  # arms -- a reset that changes with the knob is part of the measurement.
  echo '{"row_blocks": 1}' > "$TUNE"
  flip tp_to_pp || true
  echo "{\"row_blocks\": $b}" > "$TUNE"
  mark=$(grep -c "PHASE-FLIP DONE" "$LOG")
  spans=$(grep -c "backing_restore_span" "$LOG")
  if flip pp_to_tp; then
    {
      echo "=== blocks=$b pp_to_tp ==="
      grep "PHASE-FLIP DONE" "$LOG" | tail -n +$((mark + 1)) \
        | sed -E 's/.*(PP[0-9]).*in ([0-9.]+) ms.*staging reserved ([0-9.]+) MiB.*/  \1 ms=\2 staging=\3/'
      echo "  span_marks_delta=$(( $(grep -c backing_restore_span "$LOG") - spans ))"
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | sed 's/^/  free /'
    } >> "$OUT"
  else
    echo "=== blocks=$b pp_to_tp: NO FLIP COMPLETED ===" >> "$OUT"
  fi
done
echo "CLEAN-AB-COMPLETE"
cat "$OUT"
