#!/bin/bash
# #651 item 3: PP=2 with the CPU as stage 0 and the iGPU as stage 1.
#
# This is the target configuration of the strand: CPU and iGPU both ACTIVE as
# pipeline-parallel prefill workers. Decode-after-flip is a separate concern
# (Route A / #631); this boot is about getting the mixed-device world to form
# and answer at all.
#
# Settings carried over from the working single-device boot:
#  - guard v2 armed (ground rule: every serving boot),
#  - --chunked-prefill-size 256, which is what fixed the prefill crash: the
#    GGUF large-batch path does one bf16 GEMM whose M equals the prefill chunk,
#    and that GEMM fails with HIP "unspecified launch failure" at M=1024 when
#    memory is nearly exhausted (reproduced standalone at 3% free).
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export SGLANG_NUM_THREADS=12
source /root/lh/venv/bin/activate
export PYTHONPATH=/root/651-p2/sglang_src/python

PYTHONPATH=/root/lh/ggufbuild python /root/651-p2/scripts/gpu_sanity_guard_v2.py || {
  echo "GPU sanity guard v2 failed - dequantize is not fit to serve"; exit 1; }

MODEL=/root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf
# The GPU stage now holds only its share of the layers, so it needs far less
# than the single-device 0.97.
MEMFRAC=${MEMFRAC:-0.60}
PORT=${PORT:-31651}

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --tokenizer-path /root/lh/models \
  --load-format gguf \
  --quantization gguf \
  --device cuda \
  --tp-size 1 \
  --pp-size 2 \
  --pp-device-map cpu,cuda \
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
  --chunked-prefill-size 256 \
  --max-total-tokens 8192 \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  --log-level info
