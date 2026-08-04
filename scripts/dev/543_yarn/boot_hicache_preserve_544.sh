#!/usr/bin/env bash
# #544 boot: current serving layout + disk HiCache tier + preserve_thinking
# default + #540 thinking budget. NOT the #543 YaRN layout (deferred), and NOT
# kv-session-offload -- kvso and --enable-hierarchical-cache are mutually
# exclusive (server_args.py:6664-6667).
#
# Delta against the live boot (PID 1236):
#   + --enable-hierarchical-cache --hicache-storage-backend file
#   + SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/spinning/hicache
#   + --hicache-ratio 2 (host L2; NOT --hicache-size, see the note below)
#   + preserve_thinking server default        <- flag name from feat/hicache-runtime-544
#   + #540 thinking budget (merged into this tree)
# Everything else, including the parser flags, context 262144 and NEXTN MTP,
# is carried over unchanged.
set -euo pipefail

WT=${WT:-/spinning/wt-543-yarn-1m}
VENV=${VENV:-/spinning/htsglang-gpu/.venv}
MODEL=${MODEL:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8}
LOG=${LOG:-/tmp/w544_serving.log}
HICACHE_DIR=${HICACHE_DIR:-/spinning/hicache}
# NOTE: --hicache-size is PER POOL PER RANK (memory_pool_host.py:121-126). On TP=3
# with a hybrid model that is KV + Mamba x 3 ranks = 6x the number. 24 asked for
# 144 GB of pinned host RAM on a 98 GB box and the OOM killer took rank 0.
# Use the ratio instead: 2 x device pool, about 33 GB pinned node-wide.

mkdir -p "$HICACHE_DIR"

export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_BARLINK=1
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$HICACHE_DIR"

cd "$WT"
setsid "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name Qwen3.6-27B \
  --tp-size 3 --rank-gpu-id 0,1,2 \
  --rank-tp-ratio auto-performance --rank-perf-tune phase-decode \
  --rank-auto-reserve-mib 13000,4200,4200 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 262144 \
  --max-running-requests 4 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --kv-pressure-ladder auto \
  --enable-fast-lane --retraction-policy priority \
  --enable-hierarchical-cache \
  --hicache-storage-backend file \
  --hicache-ratio 2 \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend-extra-config '{"max_size": "100Gi", "min_free_space": "20Gi"}' \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --chat-template-default-kwargs '{"preserve_thinking": true}' \
  --enable-metrics --trust-remote-code \
  --host 127.0.0.1 --port 30030 \
  > "$LOG" 2>&1 &

echo $! > /tmp/w544_serving.pid
echo "launched, pid $(cat /tmp/w544_serving.pid), log $LOG"
