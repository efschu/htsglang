#!/usr/bin/env bash
# #631 successor17: measure the PP-phase and TP-phase memory PLATEAUS.
#
# WHY THIS REPLACES THE SEAM-PEAK SAMPLER. The previous instrument sampled
# NVML across a driven flip and reported baseline-minus-trough as "what a
# cutover costs". That reading cannot distinguish a transient from a phase
# hold, and on this rig it was measuring the latter: POLICY=auto returns an
# idle instance to PP ~1.5 s after a driven pp_to_tp, so the sampling window
# straddled a there-and-back PAIR and the "trough" was the TP plateau between
# the two flips.
#
# The separator is to hold the target layout under REAL WORK for much longer
# than one cutover (0.96-1.71 s measured) and see whether the level tracks the
# WORK or the CUTOVER. One generation long enough to decode for several
# seconds does it: the instance is in PP before, in TP for the whole decode,
# and back in PP after.
#
# Usage: bash scripts/phase_plateau_measure.sh [tag] [max_new_tokens]
set -uo pipefail

TAG="${1:-run}"
NTOK="${2:-1200}"
PORT="${PORT:-30030}"
CSV="/tmp/plateau_${TAG}.csv"

if ! curl -s -m 5 -o /dev/null -w '%{http_code}' \
     "http://127.0.0.1:${PORT}/health" | grep -q 200; then
  echo "REFUSE: no healthy instance on ${PORT}" >&2
  exit 1
fi

POOL=$(curl -s -m 8 "http://127.0.0.1:${PORT}/get_server_info" \
       | python3 -c 'import sys,json;print(json.load(sys.stdin).get("max_total_num_tokens"))' \
       2>/dev/null || echo "?")
echo "PP pool (id space, = serving capacity): ${POOL}"

: > "$CSV"
( while true; do
    printf '%s,%s\n' "$(date -u +%s.%3N)" \
      "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ',')"
    sleep 0.1
  done ) > "$CSV" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

sleep 2   # settle: establish the PP plateau before any work
curl -s -m 300 -X POST "http://127.0.0.1:${PORT}/generate" \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"Write a detailed technical explanation of virtual memory paging.\",\"sampling_params\":{\"max_new_tokens\":${NTOK},\"temperature\":0.7}}" \
  -o "/tmp/plateau_${TAG}.json" -w 'request http=%{http_code} wall=%{time_total}s\n'
sleep 3   # settle: let it return to PP

kill $SAMPLER 2>/dev/null
sleep 0.3

# The two plateaus are the two MODES of the free-memory series, not its
# endpoints: quantise to 32 MiB and report the most-occupied levels, so a
# cutover sample (a handful of rows) cannot be mistaken for a plateau.
echo
echo "=== per-card levels, by dwell (>=0.5 s only) ==="
awk -F, -v OFS='' 'NF>=4 {
  for (i = 2; i <= 4; i++) { b = int($i / 32) * 32; cnt[i "," b]++ }
  n++
} END {
  for (k in cnt) {
    split(k, p, ",")
    if (cnt[k] >= 5)
      printf "card %d: %6d MiB held for %5.1f s\n", p[1]-2, p[2], cnt[k]/10.0
  }
}' "$CSV" | sort -k2,2n -k4,4nr

echo
echo "=== spread per card (max-held minus min-held) ==="
awk -F, 'NF>=4 { for (i=2;i<=4;i++){ if(!seen[i]||$i>hi[i])hi[i]=$i; if(!seen[i]||$i<lo[i])lo[i]=$i; seen[i]=1 } }
END { for (i=2;i<=4;i++) printf "card %d: high %d MiB, low %d MiB, TP-phase hold %d MiB\n", i-2, hi[i], lo[i], hi[i]-lo[i] }' "$CSV"
echo "raw: $CSV"
