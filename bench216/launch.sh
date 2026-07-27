#!/usr/bin/env bash
# Boot one arm of the MLP-split campaign (task #216 follow-up).
# $1 = arm label, $2 = rank-mlp-ratio ("none" for plain auto), $3 = port
set -u
ARM="$1"; MLP="$2"; PORT="${3:-30000}"
LOG="/spinning/wt-knee-guard/bench216/logs/${ARM}.log"
mkdir -p "$(dirname "$LOG")"

# #188: the measured-KV-budget cache persists per config hash. Keep the
# feature OFF and pin the ownership vector so the KV axis is identical in
# every arm and the only free variable is the MLP split.
rm -f /root/.cache/sglang/kv_budget-*.json
export SGLANG_MEASURED_KV_BUDGET=0
export SGLANG_UNEVEN_TOKEN_VECTOR=2,3,3

NV=/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia
# deep_gemm/_C.so links libnvrtc.so.13, which only resolves via the venv's
# bundled nvidia libs; without this every TP rank aborts at scheduler start.
export LD_LIBRARY_PATH="$NV/cu13/lib:$NV/cuda_nvrtc/lib:$NV/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/spinning/wt-knee-guard/python
MODEL=/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8
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

/spinning/htsglang-gpu/.venv/bin/python -m sglang.launch_server "${ARGS[@]}" \
  > "$LOG" 2>&1 &
echo $! > "/spinning/wt-knee-guard/bench216/logs/${ARM}.pid"
echo "arm=$ARM mlp=$MLP port=$PORT pid=$(cat /spinning/wt-knee-guard/bench216/logs/${ARM}.pid) log=$LOG"
