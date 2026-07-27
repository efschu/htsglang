#!/usr/bin/env bash
# Handover test, arm B: TP=3 uneven, weighted uneven DCP, one rank per card.
# Reads the MIGRATED HiCache store (see hicache_migrate.py) -- a fresh boot
# whose only possible source of cached prefix tokens is that store.
set -euo pipefail

source /root/rig-env.sh 2>/dev/null || true
VENV="${VENV:-/spinning/htsglang-gpu/.venv}"
MODEL_ROOT="${MODEL_ROOT:-/spinning/llm_stuff/club-3090/models-cache}"

WT=/spinning/wt-handover
GGUF_DIR="${HANDOVER_GGUF_DIR:-$MODEL_ROOT/Qwen3.6-27B-MTP-Q3_K_M-GGUF}"
GGUF="${HANDOVER_GGUF:-$GGUF_DIR/Qwen3.6-27B-Q3_K_M.gguf}"
STORE="${HANDOVER_STORE_B:-/spinning/handover-store/b}"
PORT="${HANDOVER_PORT_B:-30072}"
LOG="${HANDOVER_LOG_B:-/tmp/handover.boot_b.log}"
RATIO="${HANDOVER_RATIO:-auto-performance}"

mkdir -p "$STORE"

export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$STORE"
# --rank-gpu-id addresses the full device view; never set CUDA_VISIBLE_DEVICES.
unset CUDA_VISIBLE_DEVICES

cd "$WT"
setsid "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$GGUF" \
  --tokenizer-path "$GGUF_DIR" \
  --load-format gguf --quantization gguf \
  --tp-size 3 --rank-gpu-id 0,1,2 \
  --rank-tp-ratio "$RATIO" \
  --rank-auto-reserve-mib 3000,2700,2700 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 8192 \
  --max-running-requests 4 \
  --trust-remote-code --disable-custom-all-reduce \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-write-policy write_through \
  --hicache-mem-layout page_first_direct --hicache-io-backend direct \
  --hicache-storage-backend file \
  --hicache-storage-prefetch-policy wait_complete \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 &
echo $! > /tmp/handover.boot_b.pid
echo "boot B pid $(cat /tmp/handover.boot_b.pid) port $PORT store $STORE log $LOG"
