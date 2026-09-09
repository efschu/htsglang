#!/bin/bash
# #602 acceptance, arm B: identical to boot_baseline.sh except for the one
# flag under test. Everything else -- model, context, reserve, ledger, spec
# config -- is held fixed so the delta is attributable to the mode.
# #1257c (2026-09-09): --rank-user-reserve-mib 1024 REMOVED from this recipe.
# It was written when 1024 was the DEFAULT reserve, so passing it explicitly
# restated the default and cost nothing. Under the corridor law the default is
# 0 and an explicit reserve is an ACTUATING term: it raises the floor and
# stacks on the measured transient (this rig's 5090 would go 1055 + 1024 =
# 2079 MiB), so the same literal now means something the arm-B/arm-A
# comparison was never controlling for. Dropped from BOTH arms, which keeps
# them identical to each other and to the shipped default.
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
  --enable-vram-ledger \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --max-running-requests 16 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics --host 127.0.0.1 --port 30030
