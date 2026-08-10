#!/usr/bin/env bash
# #631 section 2.1: same-boot A/B of the row-blocked seam.
#
# One boot, every block count, floor arm FIRST -- the tune file
# (SGLANG_FLIP_SEAM_TUNE_FILE) is re-read per flip, so the curve is not
# assembled across boots where boot-to-boot variance would be
# indistinguishable from the knob.
#
# Records per arm: staging reserved per rank (the gate's own number),
# flip wall time, and the seam census marks that PROVE the streamed path
# engaged. Without the engagement proof a silent fallback to whole-wave
# commits reads as "blocking changed nothing".
set -u
LOG=${LOG:-/spinning/serving-30030.boot.log}
TUNE=${TUNE:-/spinning/evidence-631/s27/seam_tune.json}
OUT=${OUT:-/spinning/evidence-631/s27/rowblock_ab.txt}
PORT=${PORT:-30030}
BLOCKS=${BLOCKS:-"1 2 4 8 16 32"}

: > "$OUT"
dir=pp_to_tp
for b in $BLOCKS; do
  echo "{\"row_blocks\": $b}" > "$TUNE"
  before=$(grep -c "PHASE-FLIP DONE" "$LOG")
  spans_before=$(grep -c "backing_restore_span" "$LOG")
  curl -s -m 5 "http://127.0.0.1:$PORT/phase_flip" \
       -H 'Content-Type: application/json' -d "{\"direction\":\"$dir\"}" >/dev/null
  for _ in $(seq 40); do
    now=$(grep -c "PHASE-FLIP DONE" "$LOG")
    [ "$now" -gt "$before" ] && break
    sleep 1
  done
  spans_after=$(grep -c "backing_restore_span" "$LOG")
  {
    echo "=== blocks=$b direction=$dir ==="
    grep "PHASE-FLIP DONE" "$LOG" | tail -3 \
      | sed -E 's/.*(PP[0-9]).*DONE ([a-z_]+) \(epoch ([0-9]+)\) in ([0-9.]+) ms.*staging reserved ([0-9.]+) MiB.*/  \1 \2 epoch=\3 ms=\4 staging=\5/'
    echo "  span_marks_delta=$(( spans_after - spans_before ))"
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | sed 's/^/  free /'
  } >> "$OUT"
  [ "$dir" = pp_to_tp ] && dir=tp_to_pp || dir=pp_to_tp
done
echo "AB-COMPLETE"
cat "$OUT"
