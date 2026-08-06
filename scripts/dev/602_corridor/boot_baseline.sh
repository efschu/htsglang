#!/bin/bash
# #602 acceptance, arm A (baseline): the ledger boot exactly as #594 left it,
# with the default 'coupled' token vector. This is the number the corridor arm
# is compared against.
#
# The recipe is deliberately the FP8 / 32768-context / max-running-requests-16
# one: the ledger's activation + graph-capture calibration is keyed on the
# activation profile digest, and that is the recipe the cached digest was
# measured for. A different recipe leaves the terms UNBOUNDED and the boot is
# refused -- which is correct, but is not an acceptance run.
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
  --enable-vram-ledger --rank-user-reserve-mib 1024 \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --max-running-requests 16 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics --host 127.0.0.1 --port 30030
