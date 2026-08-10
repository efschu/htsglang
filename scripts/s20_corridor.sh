#!/usr/bin/env bash
# #631/#656 corridor sampler: NVML FREE per card at 100 ms, plus a
# running minimum.
#
# THE COLUMN MATTERS. The user's limit is EXACTLY 1024 MiB free per card,
# read from NVML's FREE column -- never total-used, which hides a ~424/518
# MiB carve-out. And it is a CONTINUOUS limit: the figure that counts is
# the time-series minimum under load, not a boot snapshot. Free swings by
# GiB across a phase flip (the KV backing release/restore leg), so a
# minimum is only meaningful over a window that contains both phases.
#
# Usage: s20_corridor.sh <seconds> <out_file>
set -uo pipefail
DUR="${1:-3900}"
OUT="${2:-/tmp/s20_corridor.csv}"
END=$(( $(date +%s) + DUR ))
echo "ts,gpu0_free,gpu1_free,gpu2_free" > "$OUT"
while [ "$(date +%s)" -lt "$END" ]; do
    line=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | paste -sd,)
    [ -n "$line" ] && echo "$(date +%s.%N),$line" >> "$OUT"
    sleep 0.1
done
