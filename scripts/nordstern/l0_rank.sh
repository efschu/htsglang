#!/bin/bash
# Nordstern L0 -- ONE rank of a flat TP=5 group spanning two hosts.
#
# L0 is a FEASIBILITY step, not a performance one: flat gloo over Ethernet,
# no RDMA, no hierarchical transport. It exists to prove group formation,
# uneven ratios over five heterogeneous cards in four architecture classes,
# and rank-uniform execution paths. It will be slow. That is expected and is
# not a result.
#
# Every rank is its own "node" (--nnodes 5 --node-rank R), which is how this
# fork already splits ranks across two different venvs on one host. The only
# new thing here is that the nodes are on two different machines.
#
# Required env: RANK (0-4), SIDE (main|second)
# Optional:     MODEL, CTX, RATIO, MASTER, PORT, EXTRA, GRAPHFLAG
set -u
RANK=${RANK:?set RANK=0..4}
SIDE=${SIDE:?set SIDE=main|second}
MASTER=${MASTER:-192.168.0.101}     # main rig LAN address; must be reachable from BOTH hosts
PORT=${PORT:-31900}
CTX=${CTX:-4096}
RATIO=${RATIO:-3,2,2,1,1}
GRAPHFLAG=${GRAPHFLAG---disable-cuda-graph}
SRVPORT=${SRVPORT:-31095}

# ---- rank -> GPU/venv map -------------------------------------------------
# rank 0 : RTX 5090   (sm120)  main rig,   CUDA index resolved below
# rank 1 : RTX 3080   (sm86)   main rig
# rank 2 : RTX 3080   (sm86)   main rig
# rank 3 : RTX 2080Ti (sm75)   second host
# rank 4 : Vega 64    (gfx900) second host
# NOTE: torch's CUDA order is NOT nvidia-smi's on the main rig (measured:
# CUDA 0 = 5090, CUDA 1/2 = 3080). Indices below are TORCH indices.
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export TORCHDYNAMO_DISABLE=1
export SGLANG_HTCCL=1
export SGLANG_HTCCL_TRANSPORT=${TRANSPORT:-gloo}   # L0 = flat gloo, cross-host
export MAX_JOBS=4

if [ "$SIDE" = main ]; then
  MODEL=${MODEL:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8}
  case "$RANK" in
    0) export CUDA_VISIBLE_DEVICES=0 ;;   # 5090
    1) export CUDA_VISIBLE_DEVICES=1 ;;   # 3080
    2) export CUDA_VISIBLE_DEVICES=2 ;;   # 3080
    *) echo "rank $RANK is not a main-rig rank" >&2; exit 2 ;;
  esac
  PY=/spinning/htsglang-gpu/.venv/bin/python
  export PYTHONPATH=/spinning/wt-htccl/python
else
  MODEL=${MODEL:-/root/models/qwen3.6-27b-fp8}
  cd /root
  case "$RANK" in
    3) export CUDA_VISIBLE_DEVICES=0
       export CPATH=/root/venv-cuda/lib/python3.12/site-packages/nvidia/cu13/include:${CPATH:-}
       PY=/root/venv-cuda/bin/python
       export PYTHONPATH=/root/sglang-src ;;
    4) export HIP_VISIBLE_DEVICES=0
       export PYTORCH_ROCM_ARCH=gfx900
       export TRITON_HIP_LLD_PATH=/root/walld/venv-rocm63/lib/python3.12/site-packages/triton/backends/amd/llvm/bin/ld.lld
       PY=/root/walld/venv-rocm63/bin/python
       export PYTHONPATH=/root/tritoncompat:/root/triton-gcn5/python:/root/sglang-src ;;
    *) echo "rank $RANK is not a second-host rank" >&2; exit 2 ;;
  esac
fi

exec $PY -u -m sglang.launch_server \
  --model-path "$MODEL" --dtype float16 \
  --tp-size 5 --nnodes 5 --node-rank "$RANK" --dist-init-addr "$MASTER:$PORT" \
  --rank-tp-ratio "$RATIO" \
  --mamba-radix-cache-strategy no_buffer --page-size 1 --disable-overlap-schedule \
  --attention-backend triton $GRAPHFLAG \
  --max-total-tokens $CTX --context-length $CTX --max-running-requests 1 \
  --mem-fraction-static ${MEMFRAC:-0.80} \
  ${EXTRA:-} \
  --host 0.0.0.0 --port $SRVPORT
