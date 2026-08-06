#!/bin/bash
# #602 acceptance, arm B: identical to boot_baseline.sh except for the one
# flag under test. Everything else -- model, context, reserve, ledger, spec
# config -- is held fixed so the delta is attributable to the mode.
set -euo pipefail
WT=/spinning/wt-602-fill
NVRTC=/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH="$NVRTC:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$WT/python
export SGLANG_UNEVEN_DCP=1 SGLANG_UNEVEN_DCP_WEIGHTED=1 SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_VRAM_FLIGHT_DIR=$WT/flight602
exec /spinning/htsglang-gpu/.venv/bin/python -m sglang.launch_server \
  --model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8 \
  --tp 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
  --rank-kv-ratio corridor \
  --enable-vram-ledger --rank-user-reserve-mib 1024 \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --max-running-requests 16 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics --host 127.0.0.1 --port 30030
