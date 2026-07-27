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
#
# Site-specific paths and addresses come from the environment (MASTER_ADDR,
# MODEL_ROOT, VENV, REPO_ROOT on the main side; RIG2_MODEL_DIR, RIG2_VENV,
# RIG2_SGLANG_SRC, RIG2_ROCM_VENV, RIG2_TRITON_PATH on the second side).
# Source your local rig env file first; unset variables fall back to
# placeholders so the rank fails on a visibly bogus path instead of picking up
# somebody else's tree.
set -u
RANK=${RANK:?set RANK=0..4}
SIDE=${SIDE:?set SIDE=main|second}
STAGE=${STAGE:-s1}
MASTER=${MASTER:-${MASTER_ADDR:-<MASTER_ADDR>}}
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
# CAPACITY IS PART OF EVERY RESULT (harness rule #4): a tok/s number without
# the context it was reached at is not a result. MAXTOK=0 leaves the KV pool
# UNCAPPED so the boot log's `max_total_num_tokens` reports what the
# configuration actually affords; any other value caps it and must be labelled
# "capped" in the table. S1's numbers were taken CAPPED at 4096.
MAXTOK=${MAXTOK:-}
MAXTOK=${MAXTOK:-$CTX}
MAXTOK_FLAG=""
[ "$MAXTOK" != "0" ] && MAXTOK_FLAG="--max-total-tokens $MAXTOK"

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
  MODEL=${MODEL:-${MODEL_ROOT:-<MODEL_ROOT>}/Llama-3.1-8B-Instruct}
  DRAFT=${DRAFT:-${MODEL_ROOT:-<MODEL_ROOT>}/EAGLE3-LLaMA3.1-Instruct-8B}
  case "$RANK" in
    0) export CUDA_VISIBLE_DEVICES=0 ;;   # RTX 5090  (torch index, NOT nvidia-smi's)
    1) export CUDA_VISIBLE_DEVICES=1 ;;   # RTX 3080
    2) export CUDA_VISIBLE_DEVICES=2 ;;   # RTX 3080
    *) echo "rank $RANK is not a main-rig rank" >&2; exit 2 ;;
  esac
  export GLOO_SOCKET_IFNAME=${GLOO_IFNAME:-eth0}
  export TP_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
  PY=${VENV:-<VENV>}/bin/python
  # The main-side rank runs straight out of a checkout, so the source tree is
  # derived from this script's location unless REPO_ROOT overrides it.
  export PYTHONPATH=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/python
else
  MODEL=${MODEL:-${RIG2_MODEL_DIR:-<RIG2_MODEL_DIR>}/llama-3.1-8b}
  DRAFT=${DRAFT:-${RIG2_MODEL_DIR:-<RIG2_MODEL_DIR>}/eagle3-llama31-8b}
  export GLOO_SOCKET_IFNAME=${GLOO_IFNAME:-enp9s0}
  export TP_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
  cd /root
  case "$RANK" in
    3) export CUDA_VISIBLE_DEVICES=0
       R2_VENV=${RIG2_VENV:-<RIG2_VENV>}
       export CPATH=$R2_VENV/lib/python3.12/site-packages/nvidia/cu13/include:${CPATH:-}
       PY=$R2_VENV/bin/python
       export PYTHONPATH=${RIG2_SGLANG_SRC:-<RIG2_SGLANG_SRC>} ;;
    4) export HIP_VISIBLE_DEVICES=0
       export PYTORCH_ROCM_ARCH=gfx900
       R2_ROCM=${RIG2_ROCM_VENV:-<RIG2_ROCM_VENV>}
       export TRITON_HIP_LLD_PATH=$R2_ROCM/lib/python3.12/site-packages/triton/backends/amd/llvm/bin/ld.lld
       PY=$R2_ROCM/bin/python
       # RIG2_TRITON_PATH: the gfx900 triton shim tree, ':'-separated, ahead of
       # the sglang source on PYTHONPATH.
       export PYTHONPATH=${RIG2_TRITON_PATH:-<RIG2_TRITON_PATH>}:${RIG2_SGLANG_SRC:-<RIG2_SGLANG_SRC>} ;;
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


# --------------------------------------------------------------------------
# SUPERVISED LAUNCH -- do NOT exec, and do NOT detach the server.
#
# This is the guardrail for the container kill of 2026-07-25 20:48:34. sglang
# signals its PARENT when a scheduler dies (scheduler.py:
# parent_process.send_signal(SIGQUIT) via os.getppid()). If the server has been
# orphaned, that parent is PID 1: systemd caught the QUIT, dumped core, and the
# whole LXC container went down with every agent on it.
#
# `exec` would make this script BE the server, so the server's parent would be
# whatever started the script -- and when the launcher returns (it must, once
# the group is up) the server is reparented to init. Supervising instead keeps
# a parent alive for the server's entire life. If THIS script is later
# reparented to init that is harmless: a bash script's default SIGQUIT action
# is to die, not to take the container with it.
#
# Note SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION does NOT remove the hazard on its
# own: in scheduler.py the SIGQUIT to the parent is sent BEFORE the optional
# killpg. It is set below because it stops sibling ranks spewing tracebacks,
# not because it fixes this.
export SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION=1

# Tag: every process this run starts carries it in its ENVIRONMENT, so the
# launcher can find and kill exactly its own ranks. Never pattern-kill on this
# shared box.
export L0_RUN_TAG=${L0_RUN_TAG:-l0-standalone-$$}

"$PY" -u -m sglang.launch_server \
  --model-path "$MODEL" --dtype float16 \
  --tp-size 5 --nnodes 5 --node-rank "$RANK" --dist-init-addr "$MASTER:$PORT" \
  --rank-tp-ratio "$RATIO" $DCPFLAGS $SPECFLAGS \
  --page-size 1 --disable-overlap-schedule \
  --attention-backend triton --disable-cuda-graph \
  $MAXTOK_FLAG --context-length $CTX --max-running-requests 1 \
  --mem-fraction-static ${MEMFRAC:-0.80} \
  ${EXTRA:-} \
  --host 0.0.0.0 --port $SRVPORT &
CHILD=$!
echo "l0_rank: rank $RANK server pid $CHILD, supervisor pid $$ (tag $L0_RUN_TAG)"

_l0_forward() { kill -TERM "$CHILD" 2>/dev/null; }
trap _l0_forward TERM INT
# If the server crashes while orphaned it would signal init. Supervised, the
# signal lands HERE. Log it as what it is, then take the server down cleanly.
trap 'echo "l0_rank: rank '"$RANK"' caught SIGQUIT from its own child -- this is
      the crash signal that killed the container on 2026-07-25 when it reached
      PID 1 instead of a supervisor." >&2; _l0_forward' QUIT

# LIVE ORPHAN CHECK. The invariant is "the server''s PPID is this supervisor".
# It can only break if the supervisor is SIGKILLed, and the point of the check
# is that the server then dies with a clear message instead of surviving as a
# process whose next fault signals init.
(
  while kill -0 "$CHILD" 2>/dev/null; do
    _ppid=$(awk '{print $4}' "/proc/$CHILD/stat" 2>/dev/null)
    if [ "$_ppid" = "1" ]; then
      echo "l0_rank: rank $RANK server $CHILD was REPARENTED TO INIT (PPID 1)." >&2
      echo "         Its supervisor is gone, so its crash path would SIGQUIT" >&2
      echo "         PID 1 and kill the container. Aborting the rank instead." >&2
      kill -TERM "$CHILD" 2>/dev/null; sleep 5; kill -KILL "$CHILD" 2>/dev/null
      exit 0
    fi
    sleep 2
  done
) &
WATCHDOG=$!

rc=0
while kill -0 "$CHILD" 2>/dev/null; do
  wait "$CHILD"; rc=$?
done
kill -TERM "$WATCHDOG" 2>/dev/null
echo "l0_rank: rank $RANK server pid $CHILD exited rc=$rc"
exit $rc
