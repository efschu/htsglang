#!/bin/bash
# NEXTN spec retest on the no-Q6K derived checkpoint (#651 spec-fault arm).
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export SGLANG_NUM_THREADS=12
source /root/lh/venv/bin/activate

# Refuse to serve on a poisoned GPU (HANDOFF section 12.3).
PYTHONPATH=/root/lh/ggufbuild python /root/651-p2/scripts/gpu_sanity_guard.py || {
  echo "GPU sanity guard failed - reboot required"; exit 1; }

MODEL=/root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf
MEMFRAC=${MEMFRAC:-0.97}

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --tokenizer-path /root/lh/models \
  --load-format gguf \
  --quantization gguf \
  --device cuda \
  --tp-size 1 \
  --context-length 2048 \
  --max-running-requests 1 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --disable-radix-cache \
  --mamba-radix-cache-strategy no_buffer \
  --disable-overlap-schedule \
  --page-size 1 \
  --mem-fraction-static "$MEMFRAC" \
  --chunked-prefill-size 1024 \
  --max-total-tokens 2048 \
  --enable-metrics \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path "$MODEL" \
  --speculative-draft-model-quantization gguf \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --host 127.0.0.1 --port 31651 \
  --log-level info
