#!/usr/bin/env bash
# Handover test, arm A: TP=1 on the 5090 (CUDA device 0 -- verified by name,
# never assumed), deliberately small KV, HiCache file store as the park tier
# that survives the process. See docs/rig-runbook.md for the environment.
set -euo pipefail

source /root/rig-env.sh 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:-/spinning/htsglang}"
VENV="${VENV:-/spinning/htsglang-gpu/.venv}"
MODEL_ROOT="${MODEL_ROOT:-/spinning/llm_stuff/club-3090/models-cache}"

WT=/spinning/wt-handover
GGUF_DIR="${HANDOVER_GGUF_DIR:-$MODEL_ROOT/Qwen3.6-27B-MTP-Q3_K_M-GGUF}"
GGUF="${HANDOVER_GGUF:-$GGUF_DIR/Qwen3.6-27B-Q3_K_M.gguf}"
STORE="${HANDOVER_STORE_A:-/spinning/handover-store/a}"
PORT="${HANDOVER_PORT_A:-30071}"
LOG="${HANDOVER_LOG_A:-/tmp/handover.boot_a.log}"

mkdir -p "$STORE"

export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$STORE"
# The 5090 in CUDA device order (checked by the caller against the device name).
export CUDA_VISIBLE_DEVICES="${HANDOVER_CVD_5090:-0}"

cd "$WT"
setsid "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$GGUF" \
  --tokenizer-path "$GGUF_DIR" \
  --load-format gguf --quantization gguf \
  --tp-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static "${HANDOVER_MEM_FRACTION_A:-0.82}" \
  --context-length 8192 --max-total-tokens 8192 \
  --max-running-requests 4 \
  --trust-remote-code --disable-custom-all-reduce \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-write-policy write_through \
  --hicache-mem-layout page_first_direct --hicache-io-backend direct \
  --hicache-storage-backend file \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 &
echo $! > /tmp/handover.boot_a.pid
echo "boot A pid $(cat /tmp/handover.boot_a.pid) port $PORT store $STORE log $LOG"
