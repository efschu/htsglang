#!/usr/bin/env bash
# #631/#656 successor20 validation load.
#
# THE POINT OF THIS SHAPE: the resident-carry leak (defect R) needs
# QUEUEING PRESSURE to appear -- #queue-req > 0 against
# max_running_requests=4 -- because that is what leaves a slot resident
# while it runs nothing. A quiet trickle hides the defect for the better
# part of an hour (measured: 55 min under light traffic, 8 min under
# heavy), so this deliberately runs MORE concurrent streams than the
# server will admit, and mixes long prefills in, rather than pacing itself
# to the design point.
#
# Usage: s20_soak.sh <seconds> <out_dir> [concurrency]
set -uo pipefail

DUR="${1:-3900}"
OUT="${2:-/tmp/s20_soak}"
CONC="${3:-8}"
PORT="${PORT:-30030}"
mkdir -p "$OUT"
END=$(( $(date +%s) + DUR ))

# A long prompt so prefill is real work and lands in the PP layout.
LONG=$(python3 - <<'PY'
print(("The following is a detailed technical discussion of distributed "
       "inference scheduling, pipeline parallelism and speculative decoding. ") * 220)
PY
)

stream() {
    local id="$1" n=0 err=0
    while [ "$(date +%s)" -lt "$END" ]; do
        local body
        if [ $(( n % 4 )) -eq 0 ]; then
            body=$(python3 -c '
import json,sys
print(json.dumps({"model":"Qwen3.6-27B","max_tokens":180,
 "messages":[{"role":"user","content":sys.stdin.read()}]}))' <<<"$LONG")
        else
            body=$(python3 -c '
import json,random
print(json.dumps({"model":"Qwen3.6-27B","max_tokens":220,
 "messages":[{"role":"user","content":
   "Explain in detail, step by step, topic %d: how a pipeline-parallel "
   "scheduler carries resident decode requests across a layout change." % random.randint(1,10000)}]}))')
        fi
        if ! curl -s -m 180 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
                -H 'Content-Type: application/json' -d "$body" \
                -o /dev/null -w '%{http_code}\n' >> "$OUT/stream_${id}.codes" 2>/dev/null; then
            err=$(( err + 1 ))
        fi
        n=$(( n + 1 ))
    done
    echo "stream=$id requests=$n curl_errors=$err" >> "$OUT/summary"
}

for i in $(seq 1 "$CONC"); do
    stream "$i" &
done
wait
echo "SOAK DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/summary"
