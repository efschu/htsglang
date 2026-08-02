#!/usr/bin/env bash
# Task #428 / #421 F1 boot check: --kv-pressure-ladder auto.
#
# Desk state: the auto table source is injected at
# managers/kv_pressure_runtime.py and hermetically tested. What only a boot
# can show:
#   1. the profile builds from the REAL NVML + rank->card vector, not a stub;
#   2. the resulting table passes KvPressureRuntime's own rung/actuator
#      checks with real actuators attached;
#   3. a rung flip actually fires when KV pressure crosses the ascend mark;
#   4. serving stays coherent across the flip.
#
# Exits non-zero on the first failed check. One PASS/FAIL line per check.

set -uo pipefail

source /root/rig-env.sh 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT or provide /root/rig-env.sh}"
VENV="${VENV:?set VENV}"
MODEL_ROOT="${MODEL_ROOT:?set MODEL_ROOT}"
WT="${WT:-$(dirname "$REPO_ROOT")/wt-428-wiring}"
PORT="${PORT:-30428}"
LOG="${LOG:-/tmp/428_f1_ladder_auto.log}"
PIDFILE=/tmp/428_f1.pid
MODEL="${MODEL:-$MODEL_ROOT/Qwen3.6-27B-FP8}"

fail=0
say() { printf '%-6s %s\n' "$1" "$2"; [ "$1" = FAIL ] && fail=1; return 0; }

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  say FAIL "port $PORT is already in use; pick another PORT"
  exit 1
fi

export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16

cd "$WT" || exit 1
: > "$LOG"

# --max-running-requests-ceiling arms the admission_cap actuator, which is
# what makes the auto table inventory a relief rung at all -- without it the
# ladder is base-only and nothing can flip. --enable-kv-session-offload arms
# the second one, so the run exercises both wired reliefs.
setsid "$VENV/bin/python" -u -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
  --rank-auto-reserve-mib 5500,3800,3800 \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --max-running-requests 16 \
  --max-running-requests-ceiling 16 \
  --enable-kv-session-offload \
  --kv-pressure-ladder auto \
  --kv-pressure-ascend-window 3 --kv-pressure-descend-window 6 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

cleanup() {
  if [ -f "$PIDFILE" ]; then
    kill -TERM "-$(cat "$PIDFILE")" 2>/dev/null
    sleep 5
    kill -KILL "-$(cat "$PIDFILE")" 2>/dev/null
    rm -f "$PIDFILE"
  fi
}
trap cleanup EXIT

# --- check 1: it boots at all (the whole point -- it used to ValueError) ----
for _ in $(seq 1 180); do
  curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then break; fi
  sleep 5
done
if curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  say PASS "server is up with --kv-pressure-ladder auto"
else
  say FAIL "server did not come up; grep the log for 'kv-pressure-ladder auto'"
  grep -m3 -i "kv.pressure\|Traceback\|ValueError" "$LOG" | head -5
  exit 1
fi

# --- check 2: the table came from the rig profile --------------------------
if grep -q "kv-pressure.*auto: table computed from the rig profile" "$LOG"; then
  say PASS "auto table computed from the rig profile"
  grep -m1 "table computed from the rig profile" "$LOG" | tail -c 300
  echo
else
  say FAIL "no 'table computed from the rig profile' line -- the auto path did not run"
fi

# --- check 3: only wired reliefs were inventoried --------------------------
if grep -q "armed:.*rungs" "$LOG"; then
  say PASS "KvPressureRuntime armed (its rung/actuator checks accepted the table)"
  grep -m1 "armed:.*rungs" "$LOG" | tail -c 240
  echo
else
  say FAIL "KvPressureRuntime never armed"
fi

# --- check 4: drive KV pressure until a rung flips -------------------------
# Long prompts, concurrent, no early stop: the KV pool fills and the sensor
# crosses the ascend mark. Bounded by TIME, not by request count.
PROMPT=$(python3 -c "print('Summarize the following log line by line. ' + 'lorem ipsum dolor sit amet consectetur adipiscing elit. ' * 400)")
end=$(( $(date +%s) + ${DRIVE_SECONDS:-90} ))
while [ "$(date +%s)" -lt "$end" ]; do
  for _ in 1 2 3 4 5 6 7 8; do
    curl -s -m 60 "http://127.0.0.1:$PORT/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"default\",\"prompt\":$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$PROMPT"),\"max_tokens\":256,\"temperature\":0}" \
      > /dev/null 2>&1 &
  done
  wait
done

if grep -qE "kv-pressure.*(flip|ascend to rung|rung [1-9])" "$LOG"; then
  say PASS "a rung flip fired under KV pressure"
  grep -m3 -E "kv-pressure.*(flip|ascend to rung|rung [1-9])" "$LOG" | tail -c 500
  echo
else
  say FAIL "no rung flip observed in ${DRIVE_SECONDS:-90}s of pressure -- raise DRIVE_SECONDS, lower --max-running-requests-ceiling, or check the wired-relief list above"
fi

# --- check 5: serving is still coherent after the flip ---------------------
ANSWER=$(curl -s -m 60 "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","prompt":"Name the capital of France in one word:","max_tokens":8,"temperature":0}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'].strip())" 2>/dev/null)
if printf '%s' "$ANSWER" | grep -qi "paris"; then
  say PASS "serving coherent after the flip (answer: $ANSWER)"
else
  say FAIL "post-flip answer is not coherent: ${ANSWER:-<no answer>}"
fi

if grep -qiE "Traceback|CUDA error|KvLadderError" "$LOG"; then
  say FAIL "the log contains a traceback / CUDA error"
  grep -m2 -iE "Traceback|CUDA error|KvLadderError" "$LOG"
fi

exit "$fail"
