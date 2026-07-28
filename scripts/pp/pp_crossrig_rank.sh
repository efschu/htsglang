#!/bin/bash
# #201 slice 2 -- ONE node of a cross-rig pipeline: stage 0 on rig 1, stage 1 on
# rig 2, the stage boundary carried over the 40G RoCE link.
#
#   node 0  PVE HOST 10.10.10.1  RTX 3080   20 GB  sm86   pp_rank 0
#   node 1  RIG2     10.10.10.2  RTX 2080Ti 11 GB  sm75   pp_rank 1
#
# Why this needs no HTCCL, unlike the cross-rig TP=4 recipe next to it: both
# cards are NVIDIA, and PP's transport is plain torch.distributed isend/irecv on
# the NCCL device_group with gloo for the pickled metadata
# (parallel_state.send_tensor_dict). HTCCL exists for cross-VENDOR groups. The
# consequence that matters for the numbers: nothing here is host-staged, so CUDA
# graphs stay on -- section 6.3's eager requirement does not apply.
#
# _calculate_rank_ranges(nnodes=2, pp_size=2, tp_size=1) already puts pp_rank 0
# on node 0 and pp_rank 1 on node 1, so no placement code is involved.
#
# --rank-gpu-id is single-node only by construction (a world-length vector
# cannot describe a device on another host), so each node picks its own card
# with CUDA_VISIBLE_DEVICES and --base-gpu-id 0. That also keeps every process
# at exactly one visible device, which is what the mixed-architecture PDL
# constant needs (runbook 4.7).
#
# Required env: NODE (0|1), SIDE (main|second)
set -u
NODE=${NODE:?set NODE=0|1}
SIDE=${SIDE:?set SIDE=main|second}
MASTER=${MASTER:-10.10.10.1}
PORT=${PORT:-31960}
SRVPORT=${SRVPORT:-$((31160 + NODE))}
CTX=${CTX:-16384}
MEMFRAC=${MEMFRAC:-0.85}
RATIO=${RATIO:-20,12}
MODEL_NAME=${MODEL_NAME:-Qwen3.5-4B}
MAXREQ=${MAXREQ:-4}
BOUNDARY_STATS=${BOUNDARY_STATS:-200}

# The 2080 Ti has no bfloat16, so the pair is fp16 -- the weakest member sets
# the dtype for both stages (runbook 4.8).
DTYPE=${DTYPE:-float16}

# head_dim is 256 on every Qwen3.5 size, and flashinfer's prefill asks for
# 65616 B of shared memory against Turing's 65536 -- rejected at the first real
# prefill, after a clean weight load. triton on the rig-2 stage is mandatory;
# the rig-1 stage is free to differ because the attention backend is chosen per
# process and the two stages share no KV pool.
ATTN_MAIN=${ATTN_MAIN:-triton}
ATTN_SECOND=${ATTN_SECOND:-triton}

export SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION=1
export SGLANG_PP_BOUNDARY_STATS=$BOUNDARY_STATS
export SGLANG_MAMBA_SSM_DTYPE=${SGLANG_MAMBA_SSM_DTYPE:-float32}
export TORCHDYNAMO_DISABLE=1
export MAX_JOBS=4
export PP_CROSSRIG_RUN_TAG=${PP_CROSSRIG_RUN_TAG:-pp-crossrig-$$}

# Every plane on the 40G line: the gloo rendezvous and metadata (GLOO/TP socket
# ifname), and NCCL's own bootstrap and data path. Without these, gloo picks the
# default route -- the 1 GbE LAN -- and the measurement silently describes the
# wrong wire.
#
# NCCL_IB=0 (sockets on the RoCE interface) is the DEFAULT, and not for lack of
# trying: NCCL's own verbs path dies on this fabric with
# `IBV_WC_REM_INV_REQ_ERR(9) ... req_type=Send ... hca rocep1s0f1` on the first
# 5120-byte proxy tensor, while UCX drives the same two HCAs fine (the cross-rig
# TP=4 recipe). Sockets on the same interface measure 2.0 GB/s and 142 us
# one-way on the 10 KiB bs=1 payload -- for a boundary that moves 10 KiB per
# decode microbatch, verbs would buy latency, not bandwidth, and this is a
# foreign bug in NCCL's RoCE setup, not one of ours. NCCL_IB=1 re-arms the verbs
# path for whoever wants to chase it.
NCCL_IB=${NCCL_IB:-0}
if [ "$NCCL_IB" = 1 ]; then
  export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
else
  export NCCL_IB_DISABLE=1
fi
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

if [ "$SIDE" = main ]; then
  R=/spinning/subvol-999-disk-0/spinning
  MODEL=${MODEL_MAIN:-$R/llm_stuff/club-3090/models-cache/$MODEL_NAME}
  # TORCH index, not NVML: on the PVE host 0=5090, 1=3080 (NVML 0), 2=3080
  # (NVML 2). Confirm by UUID before every run -- the two orders diverge.
  export CUDA_VISIBLE_DEVICES=${CVD_MAIN:-1}
  export GLOO_SOCKET_IFNAME=${GLOO_IFNAME_MAIN:-enp4s0f1np1}
  export TP_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
  export NCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
  [ "$NCCL_IB" = 1 ] && export NCCL_IB_HCA=${NCCL_IB_HCA_MAIN:-rocep4s0f1}
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  # The host's default nvcc is 12.2 and its headers collide with Debian 13's
  # glibc; pin the pip cu13 toolkit that matches torch 2.11.0+cu130 (#263).
  export CUDA_HOME=$R/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13
  export PATH=$CUDA_HOME/bin:$PATH
  export CPATH=$CUDA_HOME/include:${CPATH:-}
  PY=/spinning/miniforge3_local_install/bin/python3.12
  # Worktree FIRST so it wins over the venv's editable sglang install.
  export PYTHONPATH=$R/wt-201/python:$R/htsglang-gpu/.venv/lib/python3.12/site-packages
  ATTN=$ATTN_MAIN
else
  MODEL=${MODEL_SECOND:-/root/models/$(echo "$MODEL_NAME" | tr 'A-Z' 'a-z')}
  export CUDA_VISIBLE_DEVICES=0
  export GLOO_SOCKET_IFNAME=${GLOO_IFNAME_SECOND:-enp1s0f1np1}
  export TP_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
  export NCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
  [ "$NCCL_IB" = 1 ] && export NCCL_IB_HCA=${NCCL_IB_HCA_SECOND:-rocep1s0f1}
  export CPATH=/root/venv-cuda/lib/python3.12/site-packages/nvidia/cu13/include:${CPATH:-}
  PY=/root/venv-cuda/bin/python
  export PYTHONPATH=/root/sglang-src
  ATTN=$ATTN_SECOND
  cd /root
fi

# Do NOT exec and do NOT orphan: sglang's scheduler signals its PARENT with
# SIGQUIT on a crash. Orphaned, that parent is PID 1.
"$PY" -u -m sglang.launch_server \
  --model-path "$MODEL" --dtype "$DTYPE" \
  --tp-size 1 --pp-size 2 --pp-layer-ratio "$RATIO" \
  --nnodes 2 --node-rank "$NODE" --dist-init-addr "$MASTER:$PORT" \
  --base-gpu-id 0 \
  --disable-overlap-schedule \
  --attention-backend "$ATTN" \
  --context-length "$CTX" --max-running-requests "$MAXREQ" \
  --mem-fraction-static "$MEMFRAC" \
  --trust-remote-code --enable-metrics \
  ${EXTRA:-} \
  --host 0.0.0.0 --port "$SRVPORT" &
CHILD=$!
echo "pp_crossrig_rank: node $NODE server pid $CHILD, supervisor $$ (tag $PP_CROSSRIG_RUN_TAG)"

_fwd() { kill -TERM "$CHILD" 2>/dev/null; }
trap _fwd TERM INT
trap 'echo "pp_crossrig_rank: node '"$NODE"' caught SIGQUIT from its own child" >&2; _fwd' QUIT

(
  while kill -0 "$CHILD" 2>/dev/null; do
    _ppid=$(awk '{print $4}' "/proc/$CHILD/stat" 2>/dev/null)
    if [ "$_ppid" = "1" ]; then
      echo "pp_crossrig_rank: node $NODE server $CHILD REPARENTED TO INIT -- aborting it" >&2
      kill -TERM "$CHILD" 2>/dev/null; sleep 5; kill -KILL "$CHILD" 2>/dev/null
      exit 0
    fi
    sleep 2
  done
) &
WATCHDOG=$!

rc=0
while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD"; rc=$?; done
kill -TERM "$WATCHDOG" 2>/dev/null
echo "pp_crossrig_rank: node $NODE server pid $CHILD exited rc=$rc"
exit $rc
