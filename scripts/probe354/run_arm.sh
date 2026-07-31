#!/usr/bin/env bash
# One measurement ARM for task #354: boot the TP=3 server with a given MLP
# weight vector, then walk prefill s=1..8 and decode bs=1..8 against it.
#
# The phase-optimal recipe needs TWO weight splits per checkpoint, and a
# weight split is a BOOT property: #297 /kv_reshard moves the KV TOKEN
# vector at runtime, #330 --enable-vram-dial moves the VRAM budget, but
# neither relocates MLP weight units. Hence one boot per phase, not one
# boot with a reshard in the middle.
#
# Usage: run_arm.sh <arm-name> <model-path> <mlp-vector|auto> <reserve-list>
set -uo pipefail

ARM="${1:?arm name}"
MODEL="${2:?model path}"
MLP="${3:?mlp vector or 'auto'}"
RESERVE="${4:?rank-auto-reserve-mib list}"

WT=/spinning/wt-354
VENV=/spinning/htsglang-gpu/.venv
PY="$VENV/bin/python"
OUT=/spinning/gpu-battery-results/2026-07-31_354_phase_optimal
PORT="${PORT:-31354}"
LOG="/tmp/probe354.$ARM.log"
PIDF="/tmp/probe354.$ARM.pid"
POINT_S="${POINT_S:-12}"
SESSIONS="${SESSIONS:-1 2 3 4 5 6 7 8}"
BATCHES="${BATCHES:-1 2 3 4 5 6 7 8}"

mkdir -p "$OUT"

export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16

MLP_FLAG=()
[ "$MLP" != "auto" ] && MLP_FLAG=(--rank-mlp-ratio "$MLP")

cd "$WT"
setsid "$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
  --rank-perf-tune enc \
  --rank-auto-reserve-mib "$RESERVE" \
  "${MLP_FLAG[@]}" \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --max-running-requests 16 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "arm=$ARM pid=$(cat "$PIDF") log=$LOG"

# Wait for readiness, bounded. Never an unbounded wait in one call.
UP=0
for _ in $(seq 1 150); do
    if curl -s -m 3 "http://127.0.0.1:$PORT/health_generate" >/dev/null 2>&1; then
        UP=1; break
    fi
    if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then break; fi
    sleep 4
done
if [ "$UP" != 1 ]; then
    echo "BOOT FAILED arm=$ARM"
    grep -nE "out of memory|OutOfMemoryError|NCCL error|Traceback|Error:" "$LOG" | tail -20
    tail -30 "$LOG"
    exit 2
fi
echo "arm=$ARM UP"

# The vector the boot actually installed, for the record.
grep -E "auto-performance|MLP vector PINNED|CHOSEN|derived memory budgets|materialized MLP units|candidate MLP vector|tune=enc" "$LOG" \
    > "$OUT/plan_$ARM.txt"

SEQ=0
for N in $SESSIONS; do
    SEQ=$((SEQ + 1))
    "$PY" "$WT/scripts/gpu_battery/s12_prefill_kurve.py" --mode messen \
        --port "$PORT" --out-dir "$OUT" --arm "$ARM" --sessions "$N" \
        --folge "$SEQ" --point-seconds "$POINT_S" --warmup-seconds 6 \
        --prompt-tokens 2048 --with-decode 0 --server-log "$LOG" \
        >> "$OUT/messen_$ARM.log" 2>&1
    echo "  prefill s=$N rc=$?"
done
SEQ=0
for B in $BATCHES; do
    SEQ=$((SEQ + 1))
    "$PY" "$WT/scripts/gpu_battery/s14_decode_punkt.py" \
        --port "$PORT" --out-dir "$OUT" --arm "$ARM" --bs "$B" \
        --folge "$SEQ" --context-tokens 2048 --model-context-tokens 32768 \
        --ramp-seconds 6 --window-seconds "$POINT_S" --server-log "$LOG" \
        >> "$OUT/messen_$ARM.log" 2>&1
    echo "  decode bs=$B rc=$?"
done

# VRAM corridor evidence while the server still holds its pools.
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free \
    --format=csv,noheader > "$OUT/vram_$ARM.txt"
cat "$OUT/vram_$ARM.txt"

"$VENV/bin/py-spy" dump --pid "$(cat "$PIDF")" > "$OUT/pyspy_$ARM.txt" 2>&1 || true
kill -TERM -- "-$(cat "$PIDF")" 2>/dev/null
for _ in $(seq 1 30); do kill -0 "$(cat "$PIDF")" 2>/dev/null || break; sleep 2; done
kill -KILL -- "-$(cat "$PIDF")" 2>/dev/null
rm -f "$PIDF"
echo "arm=$ARM DONE"
