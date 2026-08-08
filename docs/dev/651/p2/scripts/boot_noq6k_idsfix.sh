#!/bin/bash
# #651 phase 2 -- coherence boot: Q4_K_M, no speculation, ctx 8192.
# Proven recipe from /root/lh/boot_q4.sh (q4_f run), own port and paths.
# HSA_OVERRIDE_GFX_VERSION=11.0.0 is mandatory: torch ROCm carries no gfx1103
# code objects and the sglang_gguf_rocm extension is built for gfx1100.
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export SGLANG_NUM_THREADS=12
source /root/lh/venv/bin/activate
export PYTHONPATH=/root/651-p2/sglang_src/python

# Refuse to serve on a poisoned GPU (suspend/resume or load-widened defect
# family, HANDOFF section 12.3): cheap determinism probe, ~2 s.
PYTHONPATH=/root/lh/ggufbuild python /root/651-p2/scripts/gpu_sanity_guard.py || {
  echo "GPU sanity guard failed - reboot required"; exit 1; }

MODEL=/root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf
MEMFRAC=${MEMFRAC:-0.95}
PORT=${PORT:-31651}

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --tokenizer-path /root/lh/models \
  --load-format gguf \
  --quantization gguf \
  --device cuda \
  --tp-size 1 \
  --context-length 8192 \
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
  --max-total-tokens 8192 \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  --log-level info
