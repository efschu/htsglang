#!/bin/bash
# Drive one link-latency configuration across both rigs (world=2).
# Usage: run_lat.sh <gloo-1g|gloo-roce|ucx> [iters]
#
# Hosts, keys, interpreters and the two source trees come from the environment
# (RIG1_HOST, RIG1_KEY, RIG2_HOST, RIG2_KEY, RDMA_R1, PVE_MINIFORGE,
# RIG2_VENV, RIG1_REPO_ROOT, RIG2_SGLANG_SRC); source your local rig env file
# first. The fallbacks are placeholders, so an unsourced run fails at the first
# ssh instead of reaching for some other machine.
set -u
MODE="$1"
ITERS="${2:-10}"
PVEKEY="${RIG1_KEY:-<RIG1_SSH_KEY>}"
RIG2KEY="${RIG2_KEY:-<RIG2_SSH_KEY>}"
RIG1_ADDR="${RIG1_HOST:-<RIG1_IP>}"
RIG2_ADDR="${RIG2_HOST:-<RIG2_IP>}"
PVE="ssh -n -i $PVEKEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@$RIG1_ADDR"
RIG2="ssh -n -i $RIG2KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@$RIG2_ADDR"

# RIG1_REPO_ROOT: this checkout as the PVE host sees it (its mount of the same
# storage), so rank 0 runs the very same harness and transport modules.
R="${RIG1_REPO_ROOT:-<RIG1_REPO_ROOT>}"
PVE_PY="${PVE_MINIFORGE:-<PVE_MINIFORGE>}/bin/python3.12"
RIG2_PY="${RIG2_VENV:-<RIG2_VENV>}/bin/python"
PVE_SCRIPT=$R/scripts/r3val/link_lat.py
RIG2_SCRIPT="${RIG2_R3VAL_SCRIPT:-/root/link_lat.py}"
COMM=$R/python/sglang/srt/distributed/device_communicators
RIG2_COMM="${RIG2_SGLANG_SRC:-<RIG2_SGLANG_SRC>}/sglang/srt/distributed/device_communicators"

RDMA1="${RDMA_R1:-<RDMA_R1>}"
case "$MODE" in
  gloo-1g)   MASTER=$RIG1_ADDR; PVE_IF=vmbr0;        RIG2_IF=enp7s0 ;;
  gloo-roce) MASTER=$RDMA1;     PVE_IF=enp4s0f1np1; RIG2_IF=enp1s0f1np1 ;;
  ucx)       MASTER=$RDMA1;     PVE_IF=enp4s0f1np1; RIG2_IF=enp1s0f1np1 ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
PORT=${PORT:-29577}

COMMON="MASTER_ADDR=$MASTER MASTER_PORT=$PORT"
if [ "$MODE" = ucx ]; then
  UCXENV="UCX_TLS=rc,self,sm UCX_IB_GID_INDEX=3"
  PVE_ENV="$COMMON $UCXENV UCX_NET_DEVICES=rocep4s0f1:1 SGLANG_HTCCL_UCX_LIB=/opt/ucx116/lib/libucp.so.0 GLOO_SOCKET_IFNAME=$PVE_IF"
  RIG2_ENV="$COMMON $UCXENV UCX_NET_DEVICES=rocep1s0f1:1 GLOO_SOCKET_IFNAME=$RIG2_IF"
  PVE_ARGS="--comm-dir $COMM"
  RIG2_ARGS="--comm-dir $RIG2_COMM"
else
  PVE_ENV="$COMMON GLOO_SOCKET_IFNAME=$PVE_IF"
  RIG2_ENV="$COMMON GLOO_SOCKET_IFNAME=$RIG2_IF"
  PVE_ARGS=""
  RIG2_ARGS=""
fi

# Logs land beside the harness unless R3VAL_LOGS points elsewhere. The PVE
# host writes rank 0's json through its own view of the same tree, so OUT and
# the --out path below are the same file seen from the two sides.
LOGS="${R3VAL_LOGS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs}"
mkdir -p "$LOGS"
OUT=$LOGS/lat_${MODE}.json
echo "== $MODE: launching rank 1 on rig2 =="
$RIG2 "cd /root && $RIG2_ENV $RIG2_PY $RIG2_SCRIPT --mode $MODE --rank 1 --world 2 --iters $ITERS $RIG2_ARGS" \
  > "$LOGS/lat_${MODE}.r1.log" 2>&1 &
R1PID=$!
sleep 2
echo "== $MODE: rank 0 on the PVE host =="
$PVE "cd /tmp && $PVE_ENV $PVE_PY $PVE_SCRIPT --mode $MODE --rank 0 --world 2 --iters $ITERS $PVE_ARGS --out $R/scripts/r3val/logs/lat_${MODE}.json" \
  2>&1 | tee "$LOGS/lat_${MODE}.r0.log"
wait $R1PID 2>/dev/null
echo "== $MODE done -> $OUT =="
