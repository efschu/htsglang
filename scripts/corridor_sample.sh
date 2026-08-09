#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #602 corridor sampler: per-card NVML FREE at 100 ms, with the running
# minimum per card and a breach count against the 1024 MiB floor.
#
# THE FREE COLUMN, NOT total-minus-used. NVML holds a per-card driver
# carve-out (~424 MiB on the 3080s, ~519 on the 5090, measured) back from
# BOTH total and used, so total-minus-used over-states free by exactly that
# amount. The corridor law is written on the free column.
#
# Usage:  bash scripts/corridor_sample.sh <seconds> [out.csv]
set -euo pipefail

SECS="${1:-600}"
OUT="${2:-/tmp/corridor_$(date -u +%H%M%S).csv}"
FLOOR="${FLOOR:-1024}"

echo "ts_ms,gpu0_free,gpu1_free,gpu2_free" > "$OUT"
END=$(( $(date +%s) + SECS ))

while [ "$(date +%s)" -lt "$END" ]; do
    LINE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | paste -sd,)
    echo "$(date +%s%3N),$LINE" >> "$OUT"
    sleep 0.1
done

awk -F, -v floor="$FLOOR" '
NR>1 {
    for (i = 2; i <= 4; i++) {
        if (min[i] == 0 || $i < min[i]) min[i] = $i
        if ($i < floor) breach[i]++
    }
    n++
}
END {
    printf "samples=%d floor=%d MiB\n", n, floor
    for (i = 2; i <= 4; i++)
        printf "  gpu%d  min_free=%6d MiB  breaches=%d\n", i-2, min[i], breach[i]+0
}' "$OUT"
echo "series: $OUT"
