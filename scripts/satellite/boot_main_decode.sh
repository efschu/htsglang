#!/usr/bin/env bash
# Main-rig decode arm (#212) -- runs ON THE HYPERVISOR HOST, in the serving
# image, on the host network.
#
# Counterpart of boot_satellite_prefill.sh. This side never prefills a request
# that arrives with bootstrap information: it pre-allocates the KV rows (and,
# for a hybrid GDN model, a mamba slot), tells the satellite where to write,
# and resumes at the handoff token.
#
# WHY THE HOST AND NOT THE DEV CONTAINER. The container has no interface on
# the cross-rig subnet, so a decode arm running there can only reach the
# satellite over the slow LAN, and the measurement then reports that LAN
# rather than the feature. MAIN_HOST_IP below is what sglang's
# get_local_ip_auto returns (SGLANG_HOST_IP) and therefore the address
# mooncake advertises -- it decides which wire the KV bulk rides. Set it to
# this host's address on the fast line.
#
# MODE=mono boots the same model WITHOUT disaggregation, which is path (a) of
# the comparison. Same flags otherwise, and in particular no speculative
# decoding -- the PD arms force it off, so a monolithic baseline with a draft
# model would be measuring the draft, not the satellite.
set -euo pipefail

MAIN_IMAGE="${MAIN_IMAGE:-<SERVING_IMAGE>}"
# Checkpoint on the host, and the path it is mounted at INSIDE the container.
# The mount path must equal the satellite's --model-path: the PD handshake
# does not compare models at all, and any HiCache-store handover hashes the
# normalized model_path into its key. Aligning it costs nothing and closes
# both gaps at once.
MAIN_MODEL_HOST="${MAIN_MODEL_HOST:-<MODEL_ROOT>/<MODEL>}"
MAIN_MODEL_IN="${MAIN_MODEL_IN:-/root/models/<MODEL>}"
MAIN_WTPY="${MAIN_WTPY:-<WORKTREE>/python}"
MAIN_PORT="${MAIN_PORT:-31213}"
# docker --gpus counts in NVML order, which is NOT CUDA order on this rig.
# Picking the wrong index boots cleanly on the wrong card.
MAIN_GPU_DEV="${MAIN_GPU_DEV:-device=1}"
MAIN_MEM_FRAC="${MAIN_MEM_FRAC:-0.45}"
MAIN_CTX="${MAIN_CTX:-16384}"
MAIN_HOST_IP="${MAIN_HOST_IP:-<RDMA_R1>}"
MODE="${MODE:-decode}"
MAIN_SERVED_NAME="${MAIN_SERVED_NAME:-satellite-pair}"
# fp16, not the checkpoint's bfloat16: the satellite is sm75 and has no
# bfloat16, so it casts on its own. Letting this side keep bfloat16 would put
# KV of two different widths on the wire -- and the bootstrap handshake checks
# kv_cache_dtype ("auto" on both), not the resolved element type, so nothing
# would complain. Keep the pair on one dtype, and keep the monolithic baseline
# (MODE=mono) on the SAME one or the comparison is between two numeric
# regimes rather than between two placements.
MAIN_DTYPE="${MAIN_DTYPE:-float16}"
NAME="${NAME:-t212_$MODE}"

EXTRA=""
if [ "$MODE" = "decode" ]; then
  EXTRA="--disaggregation-mode decode --disaggregation-transfer-backend mooncake_tcp"
  # --disaggregation-decode-enable-radix-cache is a hard ValueError for
  # Mamba/SSM models (mem_cache/kv_cache_builder.py), and every hybrid GDN
  # model is one. The decode arm runs a chunk cache instead, which costs
  # nothing here: the prefix does not come from this side's cache, it comes
  # over the wire. Add the flag back only for a dense model.
  if [ "${MAIN_DECODE_RADIX:-0}" = "1" ]; then
    EXTRA="$EXTRA --disaggregation-decode-enable-radix-cache"
  fi
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

# Two gaps in the serving image, both patched at container start rather than
# baked in: no mooncake wheel, no libibverbs, and the mooncake wheel links
# CUDA 12 while the image's torch is cu13 -- without the cu12 runtime on the
# loader path the transfer engine import dies on libcudart.so.12 and the arm
# comes up with no transport at all.
docker run -d --name "$NAME" --network host --ipc host \
  --gpus "\"$MAIN_GPU_DEV\"" \
  -v "$MAIN_MODEL_HOST":"$MAIN_MODEL_IN":ro \
  -v "$MAIN_WTPY":/wtpy:ro \
  -e PYTHONPATH=/wtpy \
  -e SGLANG_HOST_IP="$MAIN_HOST_IP" \
  -e SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-float32}" \
  -e MC_FORCE_TCP=1 \
  --entrypoint bash \
  "$MAIN_IMAGE" -lc "
    (apt-get update -qq && apt-get install -y -qq libibverbs1 librdmacm1) >/dev/null 2>&1
    pip install -q mooncake-transfer-engine==0.3.11.post1 nvidia-cuda-runtime-cu12 2>&1 | tail -2
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:\${LD_LIBRARY_PATH:-}
    exec python3 -u -m sglang.launch_server \
      --model-path $MAIN_MODEL_IN \
      --served-model-name $MAIN_SERVED_NAME \
      --dtype $MAIN_DTYPE \
      --tp-size 1 --base-gpu-id 0 \
      --mem-fraction-static $MAIN_MEM_FRAC \
      --context-length $MAIN_CTX \
      --max-running-requests 8 --page-size 1 \
      --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
      --trust-remote-code --enable-metrics \
      --host 0.0.0.0 --port $MAIN_PORT $EXTRA
  "
echo "started $NAME on $MAIN_PORT (gpu $MAIN_GPU_DEV, mode $MODE)"
