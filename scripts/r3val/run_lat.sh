#!/bin/bash
# Drive one link-latency configuration across both rigs (world=2).
# Usage: run_lat.sh <gloo-1g|gloo-roce|ucx> [iters]
set -u
MODE="$1"
ITERS="${2:-10}"
PVEKEY=/root/.ssh/id_root@proxmox
RIG2KEY=/root/.ssh/id_ed25519_192.168.0.89
PVE="ssh -n -i $PVEKEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@192.168.0.1"
RIG2="ssh -n -i $RIG2KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@192.168.0.89"

R=/spinning/subvol-999-disk-0/spinning
PVE_PY=/spinning/miniforge3_local_install/bin/python3.12
RIG2_PY=/root/venv-cuda/bin/python
PVE_SCRIPT=$R/r3val/link_lat.py
RIG2_SCRIPT=/root/link_lat.py
COMM=$R/wt-crossrig-tp4/python/sglang/srt/distributed/device_communicators
RIG2_COMM=/root/sglang-src/sglang/srt/distributed/device_communicators

case "$MODE" in
  gloo-1g)   MASTER=192.168.0.1; PVE_IF=vmbr0;        RIG2_IF=enp7s0 ;;
  gloo-roce) MASTER=10.10.10.1;  PVE_IF=enp4s0f1np1; RIG2_IF=enp1s0f1np1 ;;
  ucx)       MASTER=10.10.10.1;  PVE_IF=enp4s0f1np1; RIG2_IF=enp1s0f1np1 ;;
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

OUT=/spinning/r3val/logs/lat_${MODE}.json
echo "== $MODE: launching rank 1 on rig2 =="
$RIG2 "cd /root && $RIG2_ENV $RIG2_PY $RIG2_SCRIPT --mode $MODE --rank 1 --world 2 --iters $ITERS $RIG2_ARGS" \
  > /spinning/r3val/logs/lat_${MODE}.r1.log 2>&1 &
R1PID=$!
sleep 2
echo "== $MODE: rank 0 on the PVE host =="
$PVE "cd /tmp && $PVE_ENV $PVE_PY $PVE_SCRIPT --mode $MODE --rank 0 --world 2 --iters $ITERS $PVE_ARGS --out $R/r3val/logs/lat_${MODE}.json" \
  2>&1 | tee /spinning/r3val/logs/lat_${MODE}.r0.log
wait $R1PID 2>/dev/null
echo "== $MODE done -> $OUT =="
