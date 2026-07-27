#!/usr/bin/env bash
# Boot one arm of the MLP-split campaign (task #216 follow-up).
# $1 = arm label, $2 = rank-mlp-ratio ("none" for plain auto), $3 = port
#
# The source tree is this script's checkout; VENV and MODEL_ROOT come from the
# environment (source your local rig env file). Their fallbacks are
# placeholders so an unsourced run fails on a bogus path rather than booting
# against somebody else's venv or model cache.
set -u
ARM="$1"; MLP="$2"; PORT="${3:-30000}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOGDIR="$REPO_ROOT/bench216/logs"
LOG="$LOGDIR/${ARM}.log"
mkdir -p "$LOGDIR"

# #188: the measured-KV-budget cache persists per config hash. Keep the
# feature OFF and pin the ownership vector so the KV axis is identical in
# every arm and the only free variable is the MLP split.
rm -f /root/.cache/sglang/kv_budget-*.json
export SGLANG_MEASURED_KV_BUDGET=0
export SGLANG_UNEVEN_TOKEN_VECTOR=2,3,3

VENV="${VENV:-<VENV>}"
NV="$VENV/lib/python3.12/site-packages/nvidia"
# deep_gemm/_C.so links libnvrtc.so.13, which only resolves via the venv's
# bundled nvidia libs; without this every TP rank aborts at scheduler start.
export LD_LIBRARY_PATH="$NV/cu13/lib:$NV/cuda_nvrtc/lib:$NV/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT/python"
MODEL="${MODEL_ROOT:-<MODEL_ROOT>}/Qwen3.6-27B-FP8"
ARGS=(
  --model-path "$MODEL"
  --tp-size 3
  --rank-gpu-id 0,1,2
  --rank-tp-ratio auto
  --kv-cache-dtype fp8_e5m2
  --context-length 32768
  --max-total-tokens 16384
  --speculative-algorithm NEXTN
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
  --host 127.0.0.1 --port "$PORT"
)
[ "$MLP" != "none" ] && ARGS+=(--rank-mlp-ratio "$MLP")

"$VENV/bin/python" -m sglang.launch_server "${ARGS[@]}" \
  > "$LOG" 2>&1 &
echo $! > "$LOGDIR/${ARM}.pid"
echo "arm=$ARM mlp=$MLP port=$PORT pid=$(cat "$LOGDIR/${ARM}.pid") log=$LOG"
