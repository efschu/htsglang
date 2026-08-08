#!/bin/bash
# #651 TARGET: Qwen3.6-35B-A3B Q4 GGUF with NEXTN speculation (draft), on the
# laptop iGPU. This is the strand's actual objective -- the dense 1.5B vehicle
# was only a proof of the PP machinery, not the target.
#
# The checkpoint carries its own draft weights (blk.40.nextn.*), so the draft
# model IS the same GGUF file; no separate draft checkpoint is needed.
#
# Differences from the earlier boot_noq6k_spec.sh, which ran at 09:11 BEFORE
# the coherence fix (int32 topk_ids, b7a46481c3) and therefore measured an
# incoherent model:
#   - sanity guard v2 (correctness-gated, not the bit-determinism canary that
#     refused a third to a half of all healthy boots);
#   - the gfx1103 wedge policy, which caps the prefill chunk at 256 because
#     that chunk is the M of the large-batch bf16 GEMM and M=1024 wedges the
#     GPU in amdgpu firmware;
#   - PYTHONPATH pinned to the laptop tree, which is the one carrying the
#     ROCm-GGUF enablement.
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
export SGLANG_NUM_THREADS=12
source /root/lh/venv/bin/activate
export PYTHONPATH=/root/651-p2/sglang_src/python

PYTHONPATH=/root/lh/ggufbuild python /root/651-p2/scripts/gpu_sanity_guard_v2.py || {
  echo "GPU sanity guard v2 failed - dequantize is not fit to serve"; exit 1; }

CHUNKED_PREFILL=${CHUNKED_PREFILL:-256}
python /root/651-p2/scripts/wedge_policy.py "$CHUNKED_PREFILL" || {
  echo "Wedge policy refused this configuration"; exit 1; }

MODEL=/root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf
MEMFRAC=${MEMFRAC:-0.97}
PORT=${PORT:-31651}

# GRAPHS=0 keeps the eager path every floor so far was measured on.
# GRAPHS=1 attempts HIP graph capture for decode (and, under speculation, the
# draft/verify path) -- the user's second deliverable.
GRAPHS=${GRAPHS:-0}
GRAPH_ARGS=()
if [ "$GRAPHS" = "0" ]; then
  GRAPH_ARGS=(--disable-cuda-graph)
fi

# SPEC=0 boots the same checkpoint without speculation, as the A/B baseline the
# accept-length and decode numbers have to be read against.
SPEC=${SPEC:-1}
SPEC_ARGS=()
if [ "$SPEC" = "1" ]; then
  SPEC_ARGS=(
    --speculative-algorithm NEXTN
    --speculative-draft-model-path "$MODEL"
    --speculative-draft-model-quantization gguf
    --speculative-num-steps "${SPEC_STEPS:-1}"
    --speculative-eagle-topk "${SPEC_TOPK:-1}"
    --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS:-2}"
  )
fi

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
  "${GRAPH_ARGS[@]}" \
  --disable-radix-cache \
  --mamba-radix-cache-strategy no_buffer \
  --disable-overlap-schedule \
  --page-size 1 \
  --mem-fraction-static "$MEMFRAC" \
  --chunked-prefill-size "$CHUNKED_PREFILL" \
  --max-total-tokens 2048 \
  --enable-metrics \
  "${SPEC_ARGS[@]}" \
  --host 127.0.0.1 --port "$PORT" \
  --log-level info
