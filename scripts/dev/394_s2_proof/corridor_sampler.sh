#!/usr/bin/env bash
# Sample per-card free VRAM at a fixed interval into a CSV.
#
# WHY THIS FILE IS IN THE REPO (#458). The corridor rule is ">= 400 MiB free
# per card DURING serving", and until the 2026-08-03 window nothing in this
# directory measured that: run_arm.sh records a single post-boot nvidia-smi
# line, and a post-boot snapshot overstates free VRAM by 250-330 MiB on this
# recipe. Judged on the snapshot the #439 arms looked green at ~515 MiB; judged
# on the serving minimum they were at 211-251 MiB, i.e. outside the corridor in
# every arm. The window wrote this sampler by hand, which is how two windows
# stop being comparable -- so it lives here now, like run_arm.sh.
#
# Sample at 1 Hz unless there is a reason not to. The 5 s used in 2026-08-03
# takes ~186 samples over a ~15 min serving window; the minimum of a sampled
# series is biased HIGH, and a transient allocation peak between two samples is
# exactly the event the corridor rule exists to catch.
#
# Stop it by touching <out>.stop -- never by pkill (shared box: broad kills
# take out other sessions' servers).
#
#   usage: corridor_sampler.sh <out-csv> [poll-sec]
#
# Read the result with the minimum per column, over the SERVING window only:
#   awk -F, 'NR>1 {for(i=2;i<=NF;i++) if(m[i]==""||$i<m[i]) m[i]=$i}
#            END {for(i=2;i<=NF;i++) printf "col%d min %s MiB\n", i-1, m[i]}' out.csv
#
# The columns are NVML indices in nvidia-smi order, which is NOT the CUDA order
# --rank-gpu-id speaks. Map them with preflight.sh's NVML table before
# attributing a column to a rank.
set -u
OUT="${1:?output csv}"
POLL="${2:-1}"
echo $$ > "$OUT.pid"
{
  printf 'ts_utc'
  n=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | wc -l)
  for i in $(seq 0 $((n - 1))); do printf ',nvml%d_free_mib' "$i"; done
  printf '\n'
} > "$OUT"
while true; do
  [ -f "$OUT.stop" ] && break
  line=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | paste -sd,)
  echo "$(date -u +%FT%TZ),$line" >> "$OUT"
  sleep "$POLL"
done
echo "stopped" > "$OUT.done"
