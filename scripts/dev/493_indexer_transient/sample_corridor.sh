#!/bin/bash
# Corridor sampler for a transient, not for a steady state (#493).
#
# Window 3 sampled `nvidia-smi --query-gpu=memory.free` in a shell loop with
# `sleep 1`. The transient it was trying to catch lasts well under a second and
# recurs once per prefill chunk (~12 s apart on that recipe), so the 1 Hz trace
# caught it only occasionally and at random points on its rise and fall: the
# reported 602 MiB excursion is a LOWER bound on the peak, and the apparent
# "growth" of the dip over the first 700 s of prefill is extreme-value
# statistics of undersampling, not a real ramp. Do not repeat that.
#
# This drives nvidia-smi's own internal loop (-lms), which does not pay process
# startup per sample, and defaults to 100 ms. The authoritative peak still comes
# from SGLANG_FORWARD_PEAK_PATH (per rank, per forward, driver-side
# `nvml_free_bytes_min` alongside torch's counter) -- this trace is the shape,
# forward_peak is the number.
#
# Usage:  sample_corridor.sh <out.csv> [interval_ms]
# Stop:   touch <out.csv>.stop
set -u

OUT="${1:?usage: sample_corridor.sh <out.csv> [interval_ms]}"
INTERVAL_MS="${2:-100}"

rm -f "$OUT.stop" "$OUT.done"
echo "timestamp,index,memory_free_mib" > "$OUT"

nvidia-smi \
  --query-gpu=timestamp,index,memory.free \
  --format=csv,noheader,nounits \
  -lms "$INTERVAL_MS" >> "$OUT" &
SMI_PID=$!
echo "$SMI_PID" > "$OUT.pid"

# Bounded wait, re-checked -- never an unbounded block inside one call.
while [ ! -f "$OUT.stop" ]; do
  kill -0 "$SMI_PID" 2>/dev/null || break
  sleep 1
done

kill "$SMI_PID" 2>/dev/null
wait "$SMI_PID" 2>/dev/null
echo done > "$OUT.done"
