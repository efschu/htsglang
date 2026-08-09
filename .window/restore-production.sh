#!/usr/bin/env bash
# Restore the production serving stack on port 30030 exactly as it ran before
# the #631 Route A GPU window. Captured from /proc/936394 at 2026-08-07T16:38Z.
#
# The #631 strand stops production to get all three cards. Bringing it back is
# this strand's own obligation, so the restore is a script and not a memory of
# a command line.
set -euo pipefail

PROD_WT=/spinning/wt-530-serving
VENV=/spinning/htsglang-gpu/.venv/bin/python
MODEL=/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8

if curl -fsS -m 3 http://127.0.0.1:30030/health >/dev/null 2>&1; then
    echo "production already healthy on 30030; nothing to do"
    exit 0
fi

export PYTHONPATH="$PROD_WT/python"
export LD_LIBRARY_PATH=/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_BARLINK=0
export SGLANG_BARLINK_BAR1_CAP_CYCLES=300000000000
export SGLANG_BOOT_COMMIT=6c1e5cafb7
export SGLANG_COLLECTIVE_CENSUS_INTERVAL=1
export SGLANG_MAMBA_PIN_TRACE=50
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/spinning/hicache
export SGLANG_VRAM_FLIGHT_DIR=/spinning/flight_605

setsid "$VENV" -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name Qwen3.6-27B \
    --tp-size 3 \
    --rank-gpu-id 0,1,2 \
    --rank-tp-ratio auto-performance \
    --rank-perf-tune phase-decode \
    --rank-auto-reserve-mib 5500,3800,3800 \
    --kv-cache-dtype fp8_e4m3 \
    --context-length 262144 \
    --max-running-requests 4 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --kv-pressure-ladder auto \
    --enable-fast-lane \
    --retraction-policy priority \
    --enable-hierarchical-cache \
    --hicache-ratio 2 \
    --hicache-write-policy write_through \
    --hicache-mem-layout page_first_direct \
    --hicache-io-backend direct \
    --chat-template-default-kwargs '{"preserve_thinking": true}' \
    --enable-cache-report \
    --enable-metrics \
    --trust-remote-code \
    --host 127.0.0.1 --port 30030 \
    > /tmp/route-a-631/prod_restore.log 2>&1 &

echo "production relaunched; waiting for health on 30030"
for _ in $(seq 1 180); do
    if curl -fsS -m 3 http://127.0.0.1:30030/health >/dev/null 2>&1; then
        echo "production healthy on 30030"
        exit 0
    fi
    sleep 5
done
echo "FATAL: production did not become healthy; see /tmp/route-a-631/prod_restore.log" >&2
exit 1
