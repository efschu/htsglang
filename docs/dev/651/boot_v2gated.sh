#!/bin/bash
# #651 phase 2 -- coherence boot: Q4_K_M (noQ6K requant), no speculation, ctx 8192.
# Same recipe as boot_noq6k_idsfix.sh, gated by the v2 sanity guard.
#
# v1's guard gated on 8-run bit-determinism of Q5_K dequantize and refused
# roughly a third to a half of all boots of a healthy machine, prescribing a
# reboot for a condition that is not a machine state at all (measured
# 2026-08-08: failures uncorrelated with load, idle, suspend, runtime PM or
# GFXOFF; a rare per-launch kernel-output fault instead). v2 gates on
# CORRECTNESS against the numpy oracle and merely reports the transient.
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export SGLANG_NUM_THREADS=12
source /root/lh/venv/bin/activate
export PYTHONPATH=/root/651-p2/sglang_src/python

PYTHONPATH=/root/lh/ggufbuild python /root/651-p2/scripts/gpu_sanity_guard_v2.py || {
  echo "GPU sanity guard v2 failed - dequantize is not fit to serve"; exit 1; }

# #651: refuse the amdgpu MES-wedge regime (gfx1103). The prefill chunk is the
# M of the GGUF large-batch bf16 GEMM, and M=1024 wedges this GPU in firmware.
CHUNKED_PREFILL=${CHUNKED_PREFILL:-256}
python /root/651-p2/scripts/wedge_policy.py "$CHUNKED_PREFILL" || {
  echo "Wedge policy refused this configuration"; exit 1; }

MODEL=/root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf
# 0.95 is NOT viable for this checkpoint: the loader computes a minimum of
# 0.963 (weights, plus a 0.95 GiB GGUF dequant-scratch reservation) and aborts
# below it with "Loaded weights leave no GPU memory for the KV cache". 0.97 is
# the value the coherent boots of 2026-08-08 actually ran at.
MEMFRAC=${MEMFRAC:-0.97}
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
  --chunked-prefill-size "$CHUNKED_PREFILL" \
  --max-total-tokens 8192 \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  --log-level info
