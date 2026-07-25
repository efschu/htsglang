#!/bin/bash
# Nordstern L0 -- ONE rank of a flat TP=5 group spanning two hosts.
#
# L0 is a FEASIBILITY step: flat gloo over Ethernet, no RDMA, no hierarchical
# transport. It proves group formation, uneven ratios over five heterogeneous
# cards in four architecture classes, and rank-uniform execution paths. It is
# slow by construction; that is not a result.
#
# CUDA GRAPHS ARE OUT FOR L0 BY DESIGN. Flat gloo is a CPU transport and the
# fork's own startup guard rejects graph capture on it. Not to be bypassed --
# see the report; the two ways to get graphs later are L2 (hierarchical
# transport, device intra-rig) and a piecewise-capture port.
#
# Model: Llama-3.1-8B-Instruct. Chosen because q=32/kv=8 means kv >= tp=5, so
# the REPLICATED-KV geometry does NOT engage: no mandatory DCP, and the triton
# backend's normal path is correct. (Any kv<5 model needs flashinfer's weighted
# owner rule, which does not exist on gfx900 -- task #173.)
#
# Required env: RANK (0-4), SIDE (main|second)
# Optional:     STAGE (s1|s2), MODEL, CTX, RATIO, MASTER, PORT, EXTRA
set -u
RANK=${RANK:?set RANK=0..4}
SIDE=${SIDE:?set SIDE=main|second}
STAGE=${STAGE:-s1}
MASTER=${MASTER:-192.168.0.101}
PORT=${PORT:-31900}
CTX=${CTX:-4096}
# 4,3,3,2,1 -> kv heads [2,2,2,1,1], q heads [8,8,8,4,4]: monotonic with
# capability and >= 6.5 GB headroom on every card. (3,2,2,1,1 is NOT monotonic
# here -- it gives kv [2,1,1,2,2], i.e. the weakest cards the most heads.)
RATIO=${RATIO:-4,3,3,2,1}
# Every node-rank starts its own HTTP server in this version, and ranks 0-2 are
# co-located on the main rig -- one shared port means "[Errno 98] address
# already in use" on whichever rank loses the race (measured: rank 0 lost, and
# its warmup then hit ANOTHER rank's app and got a 404). One port per rank;
# only rank 0's is ever queried.
SRVPORT=${SRVPORT:-$((31095 + RANK))}

export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export TORCHDYNAMO_DISABLE=1
export SGLANG_HTCCL=1
export SGLANG_HTCCL_TRANSPORT=${TRANSPORT:-gloo}
export MAX_JOBS=4
# The shm message-queue broadcaster is a NODE-LOCAL shared-memory optimisation.
# Across two hosts its handle is broadcast with dist.broadcast_object_list and
# comes back as a plain str on the far side (the two hosts run different torch
# builds), so create_from_handle dies with
#   AttributeError: 'str' object has no attribute 'local_reader_ranks'
# Measured on the second L0 attempt. Shared memory cannot span hosts anyway.
export SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0
# Cross-host gloo MUST be told which interface to use. Without this, gloo
# resolves the local hostname for its advertised endpoint, and on Debian that
# is 127.0.1.1 (/etc/hosts maps the hostname to loopback). Rank 0 then
# publishes 127.0.1.1 as its address and every remote rank tries to connect to
# its OWN loopback: "Connection refused, remote=[127.0.1.1]" -> connectFullMesh
# fails. Measured on the first L0 attempt. Pin the real LAN interface instead.

# S2 adds uneven DCP on top of S1. Explicit --dcp-size (not the env auto-set)
# so the validation handlers actually run: they are ordered before the
# auto-set, so with the env route the spec/tree guards are silently skipped.
DCPFLAGS=""
if [ "$STAGE" = s2 ]; then
  DCPFLAGS="--dcp-size 5 --rank-kv-ratio capacity"
fi

if [ "$SIDE" = main ]; then
  MODEL=${MODEL:-/spinning/llm_stuff/club-3090/models-cache/Llama-3.1-8B-Instruct}
  DRAFT=${DRAFT:-/spinning/llm_stuff/club-3090/models-cache/EAGLE3-LLaMA3.1-Instruct-8B}
  case "$RANK" in
    0) export CUDA_VISIBLE_DEVICES=0 ;;   # RTX 5090  (torch index, NOT nvidia-smi's)
    1) export CUDA_VISIBLE_DEVICES=1 ;;   # RTX 3080
    2) export CUDA_VISIBLE_DEVICES=2 ;;   # RTX 3080
    *) echo "rank $RANK is not a main-rig rank" >&2; exit 2 ;;
  esac
  export GLOO_SOCKET_IFNAME=${GLOO_IFNAME:-eth0}
  export TP_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
  PY=/spinning/htsglang-gpu/.venv/bin/python
  export PYTHONPATH=/spinning/wt-htccl/python
else
  MODEL=${MODEL:-/root/models/llama-3.1-8b}
  DRAFT=${DRAFT:-/root/models/eagle3-llama31-8b}
  export GLOO_SOCKET_IFNAME=${GLOO_IFNAME:-enp9s0}
  export TP_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
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

# S3 = EAGLE3 drafter, standard (split) placement: the draft model is TP-sharded
# over all five ranks like the target. Llama-3.1-8B has no native MTP head, so a
# drafter is required; yuhuili/EAGLE3-LLaMA3.1-Instruct-8B matches this fork's
# loader (legacy names midlayer/aux_norm_* and d2t/t2d are mapped in
# llama_eagle3.py) and its q=32/kv=8 geometry survives TP=5 like the target's.
# topk 1 keeps the draft a linear chain (no tree), which is the only shape S4
# could ever mirror.
#
# S4 (--speculative-draft-placement solo) is REJECTED BY DESIGN in this
# topology: server_args rejects solo for nnodes > 1, and L0 runs five ranks as
# five nodes. Kept here so the attempt is reproducible, not because it boots.
SPECFLAGS=""
if [ "$STAGE" = s3 ] || [ "$STAGE" = s4 ]; then
  SPECFLAGS="--speculative-algorithm EAGLE3 --speculative-draft-model-path $DRAFT \
    --speculative-num-steps ${SPEC_STEPS:-3} --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens ${SPEC_DRAFT_TOKENS:-4}"
  [ "$STAGE" = s4 ] && SPECFLAGS="$SPECFLAGS --speculative-draft-placement solo"
fi


exec $PY -u -m sglang.launch_server \
  --model-path "$MODEL" --dtype float16 \
  --tp-size 5 --nnodes 5 --node-rank "$RANK" --dist-init-addr "$MASTER:$PORT" \
  --rank-tp-ratio "$RATIO" $DCPFLAGS $SPECFLAGS \
  --page-size 1 --disable-overlap-schedule \
  --attention-backend triton --disable-cuda-graph \
  --max-total-tokens $CTX --context-length $CTX --max-running-requests 1 \
  --mem-fraction-static ${MEMFRAC:-0.80} \
  ${EXTRA:-} \
  --host 0.0.0.0 --port $SRVPORT
