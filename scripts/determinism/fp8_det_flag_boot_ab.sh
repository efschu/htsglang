#!/usr/bin/env bash
# #192 flag-verification A/B boot. One arm per invocation:
#   fp8_det_flag_boot_ab.sh off|on
# Boots Qwen3.6-27B-FP8 (block-scaled fp8 [128,128]) TP=2 across the two sm86
# 3080s -- the exact configuration #190 measured -- checks the startup log for
# which fp8 lane the dense linears took, and sends one greedy request twice to
# see whether the two completions are identical. Short by design: this verifies
# the FLAG, not throughput.
set -uo pipefail

ARM="${1:-off}"
PORT="${PORT:-30099}"
# The source tree is this script's checkout; VENV and MODEL_ROOT come from the
# environment (source your local rig env file). Their fallbacks are
# placeholders, so an unsourced run fails on a bogus path instead of booting
# against somebody else's venv or model cache.
REPO="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${VENV:-<VENV>}/bin/python"
MODEL="${MODEL_ROOT:-<MODEL_ROOT>}/Qwen3.6-27B-FP8"
LOG=/tmp/fp8det_boot_${ARM}.log

if [ "$ARM" = "on" ]; then export SGLANG_DETERMINISTIC_FP8_GEMM=1; fi

# PCI_BUS_ID order: 0 and 2 are the 3080s, 1 is the 5090. Torch's default
# FASTEST_FIRST order disagrees, which is why the order is pinned explicitly.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,2
export PYTHONPATH=$REPO/python

"$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp 2 \
  --context-length 8192 \
  --kv-cache-dtype fp8_e5m2 \
  --mem-fraction-static 0.83 \
  --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
echo "arm=$ARM pid=$SRV log=$LOG"

for _ in $(seq 1 120); do
  sleep 5
  kill -0 "$SRV" 2>/dev/null || { echo "SERVER DIED"; tail -30 "$LOG"; exit 1; }
  if curl -s -m 3 "http://127.0.0.1:$PORT/health_generate" >/dev/null 2>&1; then
    READY=1; break
  fi
done
[ "${READY:-0}" = 1 ] || { echo "TIMEOUT"; tail -30 "$LOG"; kill "$SRV"; exit 1; }

echo "--- fp8 lane evidence from the startup log ---"
grep -iE "DETERMINISTIC_FP8_GEMM|Marlin|deterministic inference" "$LOG" | sort -u | head

REQ='{"text":"Explain in two sentences why floating point addition is not associative.","sampling_params":{"temperature":0,"max_new_tokens":48}}'
A=$(curl -s -m 120 "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' -d "$REQ")
B=$(curl -s -m 120 "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' -d "$REQ")
echo "--- completion ---"
echo "$A" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["text"])'
if [ "$(echo "$A" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["text"])')" = \
     "$(echo "$B" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["text"])')" ]; then
  echo "repeat: IDENTICAL"
else
  echo "repeat: DIFFERENT"
fi

kill "$SRV" 2>/dev/null
wait "$SRV" 2>/dev/null
echo "arm=$ARM done"
