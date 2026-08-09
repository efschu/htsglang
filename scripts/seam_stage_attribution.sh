#!/usr/bin/env bash
# #631 successor17: drive flips on a LIVE instance and extract the seam
# census, i.e. which STAGE of the cutover spends the transient.
#
# WHY DRIVEN AND NOT AWAITED: an idle instance stops flipping (successor
# 16 burned a 40 s passive sample catching no cutover at all), so the
# flips have to be issued.
#
# Usage: bash scripts/seam_stage_attribution.sh [tag]
# Leaves: /tmp/seam_stage_<tag>.csv  (100 ms NVML samples, external check)
#         the census lines on stdout (per rank, per direction)
set -uo pipefail

TAG="${1:-run}"
PORT="${PORT:-30030}"
LOG="${SERVING_LOG:-/spinning/serving-30030.boot.log}"
CSV="/tmp/seam_stage_${TAG}.csv"

if ! curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" \
  | grep -q 200; then
  echo "REFUSE: no healthy instance on ${PORT}" >&2
  exit 1
fi

# Mark our starting point in the log so the extract below cannot pick up
# an earlier shift's flips -- three handoffs have quoted stale lines.
START_LINE=$(wc -l < "$LOG")
echo "log starts at line ${START_LINE}"

: > "$CSV"
( while true; do
    printf '%s,%s\n' "$(date -u +%s.%3N)" \
      "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ',')"
    sleep 0.1
  done ) > "$CSV" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

sleep 2
for dir in pp_to_tp tp_to_pp; do
  echo "--- driving ${dir} ---"
  curl -s -m 120 -X POST "http://127.0.0.1:${PORT}/phase_flip" \
    -H 'Content-Type: application/json' -d "{\"direction\":\"${dir}\"}" \
    | head -c 400
  echo
  sleep 4
done

kill $SAMPLER 2>/dev/null
sleep 0.3

echo
echo "=== SEAM CENSUS (this run only) ==="
tail -n +"$((START_LINE + 1))" "$LOG" | grep "seam-census" | cut -c1-600

echo
echo "=== external NVML check (baseline -> trough per card) ==="
awk -F, 'NF>=4 {
  for (i = 2; i <= 4; i++) {
    if (NR == 1 || $i < min[i]) min[i] = $i
    if (NR <= 10) base[i] = $i
  }
} END {
  for (i = 2; i <= 4; i++)
    printf "card %d: baseline %d MiB, trough %d MiB, transient %d MiB\n",
           i - 2, base[i], min[i], base[i] - min[i]
}' "$CSV"
echo "raw: $CSV"
