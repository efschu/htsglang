#!/usr/bin/env bash
# #631 seam-peak measurement: what one PP<->TP cutover costs in VRAM.
#
# THE QUANTITY AND WHY IT IS THE RIGHT ONE
# ----------------------------------------
# Every headroom argument in this task had been denominated in the 1024
# MiB corridor floor, which is a STEADY-STATE budget. A cutover is not
# steady state: it stages KV backing, packs GDN slots and checksums the
# payload, and it does all of that transiently. Measured on 2026-08-09
# that transient was 1376-2980 MiB per card -- larger than the entire
# margin some ranks were being sized to keep. A pool sized against the
# floor alone is therefore sized to kill the next flip.
#
# So the observable is the TROUGH of NVML free across the cutover, and
# the reported peak is (pre-flip baseline - trough). NVML free, not
# torch's accounting: the driver-level number is what actually refuses a
# cuMemCreate, and defect N died on exactly such a refusal while torch
# still believed it had cache.
#
# Baseline is the MEDIAN of the pre-flip samples rather than the mean:
# one sample landing inside an unrelated allocation would drag a mean and
# silently inflate the peak.
#
# Usage: seam_peak_measure.sh <direction> [out.csv] [settle_s] [watch_s]
#   direction: pp_to_tp | tp_to_pp
set -uo pipefail

DIR="${1:-pp_to_tp}"
OUT="${2:-/tmp/seam_peak_${DIR}.csv}"
SETTLE="${3:-3}"
WATCH="${4:-12}"
PORT="${PORT:-30030}"
INTERVAL=0.1

sample() {
  local end=$1
  while [ "$(date +%s%3N)" -lt "$end" ]; do
    echo "$(date -u +%s.%3N),$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits | tr '\n' ',')" >> "$OUT"
    sleep "$INTERVAL"
  done
}

: > "$OUT"
now=$(date +%s%3N)
sample $(( now + SETTLE * 1000 )) &
SPID=$!
wait $SPID
PRE=$(wc -l < "$OUT")

echo "-- arming $DIR --"
curl -s -m 25 -X POST "http://127.0.0.1:${PORT}/phase_flip" \
  -H 'Content-Type: application/json' \
  -d "{\"direction\":\"${DIR}\"}" | head -c 200
echo

now=$(date +%s%3N)
sample $(( now + WATCH * 1000 )) &
SPID=$!
wait $SPID

awk -F, -v pre="$PRE" -v dir="$DIR" '
NF>=4 { n++; for (g=0; g<3; g++) f[g,n]=$(g+2) }
END {
  if (n < pre+5) { print "TOO FEW SAMPLES"; exit 1 }
  # median of the pre-flip window per card
  for (g=0; g<3; g++) {
    for (i=1; i<=pre; i++) v[i]=f[g,i]
    # insertion sort, pre is tens of samples
    for (i=2; i<=pre; i++) { x=v[i]; j=i-1; while (j>0 && v[j]>x) { v[j+1]=v[j]; j-- } v[j+1]=x }
    base[g] = (pre % 2) ? v[(pre+1)/2] : (v[pre/2]+v[pre/2+1])/2
    m=1e9; for (i=1; i<=n; i++) if (f[g,i]<m) m=f[g,i]
    trough[g]=m
  }
  printf "direction %s, %d samples over %.1f s (%d pre-flip)\n", dir, n, n*0.1, pre
  printf "card, baseline_free_MiB, trough_free_MiB, seam_peak_MiB, above_1024_floor_MiB\n"
  name[0]="gpu0_5090"; name[1]="gpu1_3080a"; name[2]="gpu2_3080b"
  for (g=0; g<3; g++)
    printf "%s, %d, %d, %d, %d\n", name[g], base[g], trough[g], base[g]-trough[g], trough[g]-1024
}' "$OUT"
