#!/usr/bin/env bash
# #287 KV pressure staircase -- THE card proof (one window, <= 35 min).
#
# Arm A (negative control, no ladder flags): boot, one short greedy request,
#   record the output text. Baseline for "no pressure => byte-identical".
# Arm B (ladder relief:dcp_ratio,relief:admission_cap on a deliberately tiny
#   KV pool): the same calm request first (no flip may fire, text must equal
#   arm A -- the GDN prefill nondeterminism bound keeps the prompt short),
#   then a concurrent long-generation burst to drive occupancy through the
#   ascend mark: expect the dcp_ratio flip, then the admission_cap flip, on
#   ALL THREE ranks with the same epoch (rank-uniform), zero DESYNC lines, a
#   healthy server throughout, and no retract once the admission stage holds.
#
# Artifacts land in $ART; server logs stay in files (grep only).
set -u
ART="${1:-/root/.claude/jobs/1481bb40/tmp/p287}"
mkdir -p "$ART"
VENV=/spinning/htsglang-gpu/.venv/bin/python
WT=/spinning/wt-287
MODEL=/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8
PORT=30287
NVRTC=/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib

export LD_LIBRARY_PATH="$NVRTC:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$WT/python
export SGLANG_UNEVEN_DCP=1 SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16

PROMPT="Name three rivers in Europe and one fact about each of them."

say() { echo "[p287 $(date -u +%H:%M:%S)] $*"; }

boot() { # boot <logfile> [extra args...]
  local log="$1"; shift
  setsid "$VENV" -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
    --rank-auto-reserve-mib 3000,2700,2700 \
    --kv-cache-dtype fp8_e4m3 --context-length 8192 --trust-remote-code \
    --max-total-tokens 6000 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --enable-metrics --host 127.0.0.1 --port $PORT \
    "$@" > "$log" 2>&1 &
  BOOT_PGID=$!
  say "booted pgid=$BOOT_PGID log=$log"
}

wait_health() { # wait_health <seconds>
  local deadline=$((SECONDS+$1))
  while [ $SECONDS -lt $deadline ]; do
    code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)
    [ "$code" = "200" ] && return 0
    sleep 5
  done
  return 1
}

gen() { # gen <max_new> <outfile>
  curl -s -m 120 "http://127.0.0.1:$PORT/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"text\": \"$PROMPT\", \"sampling_params\": {\"temperature\": 0, \"max_new_tokens\": $1}}" \
    > "$2"
}

kill_server() {
  local pids
  pids=$(pgrep -f "sglang.launch_serve[r].*port $PORT" || true)
  for p in $pids; do
    /spinning/htsglang-gpu/.venv/bin/py-spy dump --pid "$p" > "$ART/pyspy_$p.txt" 2>&1 || true
  done
  [ -n "${BOOT_PGID:-}" ] && kill -- -"$BOOT_PGID" 2>/dev/null
  sleep 3
  pids=$(pgrep -f "sglang.launch_serve[r].*port $PORT" || true)
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  local deadline=$((SECONDS+30))
  while [ $SECONDS -lt $deadline ] && pgrep -f "sglang.launch_serve[r].*port $PORT" >/dev/null; do sleep 2; done
  say "server down"
}

verdict() { echo "P287-$1"; echo "P287-$1" >> "$ART/verdicts.txt"; }

# ---- power bracket ---------------------------------------------------------
nvidia-smi --query-gpu=timestamp,index,power.draw,memory.used --format=csv -l 5 \
  > "$ART/power.csv" 2>/dev/null &
POWER_PID=$!
trap 'kill $POWER_PID 2>/dev/null' EXIT

# ---- Arm A: no ladder ------------------------------------------------------
say "ARM A boot (no ladder)"
boot "$ART/armA.log"
if ! wait_health 240; then
  verdict "FAIL armA-boot: no health within 240s"; kill_server; exit 1
fi
gen 96 "$ART/armA_calm.json"
grep -c "KV-PRESSURE-LADDER" "$ART/armA.log" > "$ART/armA_ladder_lines.txt" || true
kill_server

# ---- Arm B: ladder on a tiny pool -----------------------------------------
say "ARM B boot (ladder relief:dcp_ratio,relief:admission_cap)"
boot "$ART/armB.log" \
  --kv-pressure-ladder relief:dcp_ratio,relief:admission_cap \
  --max-running-requests-ceiling 12 --max-running-requests 12
if ! wait_health 240; then
  verdict "FAIL armB-boot: no health within 240s"; kill_server; exit 1
fi

# Phase 1: calm -- no flip allowed, text must match arm A.
gen 96 "$ART/armB_calm.json"
if grep -q "KV-PRESSURE-LADDER FLIP" "$ART/armB.log"; then
  verdict "FAIL armB-calm: a flip fired without pressure"
else
  verdict "PASS armB-calm-no-flip"
fi
# Compare the generated TEXT, not the raw response: meta_info carries
# per-request ids/timestamps/throughput that differ between any two boots.
if "$VENV" -c "
import json, sys
a = json.load(open('$ART/armA_calm.json'))['text']
b = json.load(open('$ART/armB_calm.json'))['text']
sys.exit(0 if a == b else 1)
"; then
  verdict "PASS calm-output-text-byte-identical"
else
  verdict "FAIL calm-output-text-differs (see armA_calm.json/armB_calm.json)"
fi

# Phase 2: pressure -- 10 concurrent long generations against a 6000-token pool.
say "ARM B pressure burst"
for i in $(seq 1 10); do
  curl -s -m 400 "http://127.0.0.1:$PORT/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"text\": \"Write a very long story about a mountain expedition. Part $i.\", \"sampling_params\": {\"temperature\": 0.7, \"max_new_tokens\": 1200, \"ignore_eos\": true}}" \
    > "$ART/armB_burst_$i.json" &
done
BURST_PIDS=$(jobs -p)

deadline=$((SECONDS+300))
flip2=0
while [ $SECONDS -lt $deadline ]; do
  if grep -q "FLIP rung 1 -> 2" "$ART/armB.log"; then flip2=1; break; fi
  sleep 5
done
sleep 10  # let post-flip rounds accumulate
FLIP01=$(grep -c "FLIP rung 0 -> 1" "$ART/armB.log" || true)
FLIP12=$(grep -c "FLIP rung 1 -> 2" "$ART/armB.log" || true)
DESYNC=$(grep -c "DESYNC" "$ART/armB.log" || true)
RETRACT=$(grep -ci "retract" "$ART/armB.log" || true)
say "flip 0->1: $FLIP01 lines, flip 1->2: $FLIP12 lines, desync: $DESYNC, retract mentions: $RETRACT"

[ "$FLIP01" -ge 1 ] && verdict "PASS flip-dcp-ratio ($FLIP01 rank lines)" \
  || verdict "FAIL flip-dcp-ratio never fired"
[ "$flip2" = 1 ] && verdict "PASS flip-admission-cap ($FLIP12 rank lines)" \
  || verdict "FAIL flip-admission-cap never fired within 300s"
[ "$DESYNC" = 0 ] && verdict "PASS no-desync" || verdict "FAIL desync-lines=$DESYNC"

# Rank-uniformity: every rank logs the same transition (3 scheduler ranks).
if [ "$FLIP01" = 3 ] && { [ "$FLIP12" = 3 ] || [ "$flip2" != 1 ]; }; then
  verdict "PASS rank-uniform-3-lines-per-flip"
else
  verdict "WARN flip-line-count FLIP01=$FLIP01 FLIP12=$FLIP12 (expect 3 per flip)"
fi

# Health under pressure + a served request after the flips.
code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)
[ "$code" = "200" ] && verdict "PASS health-under-pressure" || verdict "FAIL health=$code"

wait $BURST_PIDS 2>/dev/null
gen 32 "$ART/armB_after.json"
grep -q "text" "$ART/armB_after.json" && verdict "PASS serves-after-episode" \
  || verdict "FAIL no answer after the episode"

RETRACT_AFTER=$(awk "/FLIP rung 1 -> 2/{seen=1} seen && /[Rr]etract/{n++} END{print n+0}" "$ART/armB.log")
echo "retract-after-admission-flip=$RETRACT_AFTER" >> "$ART/verdicts.txt"
[ "$RETRACT_AFTER" = 0 ] && verdict "PASS no-retract-after-admission-stage" \
  || verdict "WARN retract-after-admission-stage=$RETRACT_AFTER"

kill_server
say "done; verdicts:"; cat "$ART/verdicts.txt"
