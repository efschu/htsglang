#!/bin/bash
# #651 final leg: PP=2 with the CPU as stage 0 and the iGPU as stage 1, on a
# DENSE-ATTENTION checkpoint.
#
# WHY DENSE. The GDN hybrid vehicles (Qwen3.5/3.6) cannot run a CPU stage: the
# causal conv and the chunked delta-rule recurrence exist only as Triton
# kernels, with no torch implementation anywhere in the tree, and Triton has no
# CPU backend. A plain attention+MLP model needs none of that -- its layers
# already have torch paths -- so it isolates the QUESTION THIS BOOT ASKS:
# does the mixed-device pipeline machinery itself (W1/W2/W2b/W3 plus the six
# device-routing fixes) run end to end on this hardware?
#
# The wedge policy stays armed: --chunked-prefill-size is capped at 256 on
# gfx1103 because that chunk is the M of the large-batch bf16 GEMM, and M=1024
# wedges the GPU in amdgpu firmware.
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export SGLANG_NUM_THREADS=12
source /root/lh/venv/bin/activate
export PYTHONPATH=/root/651-p2/sglang_rig/python

PYTHONPATH=/root/lh/ggufbuild python /root/651-p2/scripts/gpu_sanity_guard_v2.py || {
  echo "GPU sanity guard v2 failed - dequantize is not fit to serve"; exit 1; }

CHUNKED_PREFILL=${CHUNKED_PREFILL:-256}
python /root/651-p2/scripts/wedge_policy.py "$CHUNKED_PREFILL" || {
  echo "Wedge policy refused this configuration"; exit 1; }

# Per-stage round accounting for the co-run split measurement (#651). Off by
# default; set to 1 for measurement boots.
export SGLANG_PP_ROUND_TRACE=${SGLANG_PP_ROUND_TRACE:-0}
export SGLANG_PP_ROUND_TRACE_EVERY=${SGLANG_PP_ROUND_TRACE_EVERY:-20}

# Capability scores per stage, highest = most layers. Empty means the default
# even split. Driven by the CO-RUN measurement, never by solo numbers: on an
# APU both stages share DDR5 bandwidth and package power, so solo speed does
# not predict co-run speed.
STAGE_RATIO=${STAGE_RATIO:-}

MODEL=${MODEL:-/root/651-p2/models-dense/Qwen2.5-1.5B-Instruct}
# The GPU stage holds only its layer share of a ~3 GB bf16 model, so it needs
# very little of the 24 GiB GTT budget.
MEMFRAC=${MEMFRAC:-0.35}
PORT=${PORT:-31651}
PPMAP=${PPMAP:-cpu,cuda}

RATIO_ARGS=()
if [ -n "$STAGE_RATIO" ]; then
  RATIO_ARGS=(--pp-stage-ratio "$STAGE_RATIO")
fi

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --device cuda \
  --tp-size 1 \
  --pp-size 2 \
  --pp-device-map "$PPMAP" \
  "${RATIO_ARGS[@]}" \
  --context-length 4096 \
  --max-running-requests 1 \
  --attention-backend torch_native \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --disable-radix-cache \
  --disable-overlap-schedule \
  --page-size 1 \
  --mem-fraction-static "$MEMFRAC" \
  --chunked-prefill-size "$CHUNKED_PREFILL" \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  --log-level info
