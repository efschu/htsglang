#!/bin/bash
# #651 item 3: live PP=2 mixed-device smoke -- CPU stage 0 + iGPU stage 1.
#
# WHY A NON-GGUF MODEL. The GGUF checkpoint cannot be used for this test yet:
# the laptop serving tree carries ROCm-GGUF ENABLEMENT patches (it adds "gguf"
# to rocm_supported_quantization, plus the standalone-binding wiring in
# gguf.py) that the #651 branch does not have, while the #651 branch carries
# the PP slice (W1/W2/W2b/W3) that the laptop tree does not have. Two divergent
# patch sets on different bases; unifying them is an integration task, not a
# boot flag.
#
# So this boot answers the question that CAN be answered today: does the
# mixed-device pipeline itself run on this hardware -- does a world with a CPU
# stage and a ROCm stage form, exchange activations, and produce correct
# tokens? That is W1/W2/W2b/W3 under real conditions rather than in unit tests.
# It uses the bf16 2B checkpoint already on the laptop.
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export SGLANG_NUM_THREADS=12
source /root/lh/venv/bin/activate
export PYTHONPATH=/root/651-p2/sglang_rig/python

PYTHONPATH=/root/lh/ggufbuild python /root/651-p2/scripts/gpu_sanity_guard_v2.py || {
  echo "GPU sanity guard v2 failed - dequantize is not fit to serve"; exit 1; }

MODEL=${MODEL:-/root/lh/models-2b}
MEMFRAC=${MEMFRAC:-0.45}
PORT=${PORT:-31651}
PPMAP=${PPMAP:-cpu,cuda}

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --device cuda \
  --tp-size 1 \
  --pp-size 2 \
  --pp-device-map "$PPMAP" \
  --context-length 4096 \
  --max-running-requests 1 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --page-size 1 \
  --mem-fraction-static "$MEMFRAC" \
  --chunked-prefill-size 256 \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  --log-level info
