#!/bin/bash
# Drive link_collective_cost.py across both rigs at the real TP=4 placement:
# ranks 0-2 on rig 1 (the Proxmox host, which owns the RoCE port), rank 3 on
# rig 2. Hosts, keys and interpreters come from the rig env file -- source it
# first; the fallbacks are placeholders so an unsourced run fails at the first
# ssh instead of reaching for some other machine.
#
# Usage: run_collective_cost.sh [iters] [tag]
# Env:   RNDV=<bytes|inf>  sets UCX_RNDV_THRESH on both sides (default: unset)
#        FP32=<0|1>        sets SGLANG_HTCCL_FP32_REDUCE
#        RING=<MiB>        sets SGLANG_HTCCL_UCX_RING_MIB (0 = always ring)
#        CHUNK=<MiB>       sets SGLANG_HTCCL_UCX_CHUNK_MIB
#        RING_KIB=<KiB>    sets SGLANG_HTCCL_UCX_RING_KIB
#        AG_RING_KIB=<KiB> sets SGLANG_HTCCL_UCX_AG_RING_KIB (0 = never ring)
#        OP=<all_reduce|all_gather>  collective for the SIZES sweep
#        GRAIN=<elems>     sets SGLANG_HTCCL_UCX_GRAIN_ELEMS (0 = unchunked
#                          host passes, the pre-#263 A/B control)
#        GATE=1            exactness gate across the threshold, not timing
set -u
ITERS="${1:-200}"
TAG="${2:-default}"
PVEKEY="${RIG1_KEY:-<RIG1_SSH_KEY>}"
RIG2KEY="${RIG2_KEY:-<RIG2_SSH_KEY>}"
RIG1_ADDR="${RIG1_HOST:-<RIG1_IP>}"
RIG2_ADDR="${RIG2_HOST:-<RIG2_IP>}"
SSHOPT="-n -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"
PVE="ssh $SSHOPT -i $PVEKEY root@$RIG1_ADDR"
RIG2="ssh $SSHOPT -i $RIG2KEY root@$RIG2_ADDR"

R="${RIG1_REPO_ROOT:-<RIG1_REPO_ROOT>}"
PVE_PY="${PVE_MINIFORGE:-<PVE_MINIFORGE>}/bin/python3.12"
RIG2_PY="${RIG2_VENV:-<RIG2_VENV>}/bin/python"
SCRIPT=$R/scripts/r3val/link_collective_cost.py
RIG2_SCRIPT="${RIG2_COST_SCRIPT:-/root/link_collective_cost.py}"
COMM=$R/python/sglang/srt/distributed/device_communicators
RIG2_COMM="${RIG2_SGLANG_SRC:-<RIG2_SGLANG_SRC>}/sglang/srt/distributed/device_communicators"

MASTER="${RDMA_R1:-<RDMA_R1>}"
PORT=${PORT:-29581}
# WORLD ranks in total, RIG2_RANKS of them on rig 2 (the far side of the
# link). The production shape is WORLD=4 RIG2_RANKS=1. WORLD=4 RIG2_RANKS=0
# is the no-wire control: the same code path with every peer on one host.
WORLD=${WORLD:-4}
RIG2_RANKS=${RIG2_RANKS:-1}

EXTRA=""
[ -n "${RNDV:-}" ] && EXTRA="$EXTRA UCX_RNDV_THRESH=$RNDV"
[ -n "${FP32:-}" ] && EXTRA="$EXTRA SGLANG_HTCCL_FP32_REDUCE=$FP32"
[ -n "${RING:-}" ] && EXTRA="$EXTRA SGLANG_HTCCL_UCX_RING_MIB=$RING"
[ -n "${CHUNK:-}" ] && EXTRA="$EXTRA SGLANG_HTCCL_UCX_CHUNK_MIB=$CHUNK"
[ -n "${RING_KIB:-}" ] && EXTRA="$EXTRA SGLANG_HTCCL_UCX_RING_KIB=$RING_KIB"
[ -n "${AG_RING_KIB:-}" ] && EXTRA="$EXTRA SGLANG_HTCCL_UCX_AG_RING_KIB=$AG_RING_KIB"
[ -n "${GRAIN:-}" ] && EXTRA="$EXTRA SGLANG_HTCCL_UCX_GRAIN_ELEMS=$GRAIN"

COMMON="MASTER_ADDR=$MASTER MASTER_PORT=$PORT UCX_TLS=rc,self,sm UCX_IB_GID_INDEX=3 $EXTRA"
PVE_ENV="$COMMON UCX_NET_DEVICES=rocep4s0f1:1 SGLANG_HTCCL_UCX_LIB=/opt/ucx116/lib/libucp.so.0 GLOO_SOCKET_IFNAME=enp4s0f1np1"
RIG2_ENV="$COMMON UCX_NET_DEVICES=rocep1s0f1:1 GLOO_SOCKET_IFNAME=enp1s0f1np1"

LOGS="${R3VAL_LOGS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs}"
mkdir -p "$LOGS"

FIRST_RIG2=$((WORLD - RIG2_RANKS))
for r in $(seq $FIRST_RIG2 $((WORLD - 1))); do
  echo "== rank $r on rig 2 =="
  $RIG2 "cd /root && $RIG2_ENV $RIG2_PY $RIG2_SCRIPT --rank $r --world $WORLD --iters $ITERS ${SIZES:+--sizes $SIZES} ${OP:+--op $OP} ${GATE:+--gate} --comm-dir $RIG2_COMM" \
    > "$LOGS/cost_${TAG}.r${r}.log" 2>&1 &
done
for r in $(seq 1 $((FIRST_RIG2 - 1))); do
  $PVE "cd /tmp && $PVE_ENV $PVE_PY $SCRIPT --rank $r --world $WORLD --iters $ITERS ${SIZES:+--sizes $SIZES} ${OP:+--op $OP} ${GATE:+--gate} --comm-dir $COMM" \
    > "$LOGS/cost_${TAG}.r${r}.log" 2>&1 &
done
sleep 2
echo "== rank 0 on rig 1 (Proxmox host) =="
$PVE "cd /tmp && $PVE_ENV $PVE_PY $SCRIPT --rank 0 --world $WORLD --iters $ITERS ${SIZES:+--sizes $SIZES} ${OP:+--op $OP} ${GATE:+--gate} --comm-dir $COMM --out $R/scripts/r3val/logs/cost_${TAG}.json" \
  2>&1 | tee "$LOGS/cost_${TAG}.r0.log"
wait
echo "== done -> $LOGS/cost_${TAG}.json =="
