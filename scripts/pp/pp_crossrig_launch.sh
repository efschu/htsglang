#!/bin/bash
# #201 slice 2 -- bounded cross-rig PP=2 boot, driven from the dev container.
#
# The container has neither the GPUs nor an address on the 40G subnet, so both
# nodes are started over ssh on a host that has both: node 0 on the PVE host,
# node 1 on rig 2. Always returns within WAIT seconds.
#
# Reads the local env file for host/key/path values; nothing here is committed
# with a real address.
set -u
source /root/rig-env.sh 2>/dev/null || true

WAIT=${WAIT:-900}
KEEP=${KEEP:-1}
CTX=${CTX:-16384}
MEMFRAC=${MEMFRAC:-0.85}
RATIO=${RATIO:-20,12}
MODEL_NAME=${MODEL_NAME:-Qwen3.5-4B}
PORT=${PORT:-31960}
MAXREQ=${MAXREQ:-4}
NCCL_IB=${NCCL_IB:-0}
ATTN_MAIN=${ATTN_MAIN:-triton}
ATTN_SECOND=${ATTN_SECOND:-triton}
TAG=${TAG:-pp-crossrig-$(date +%H%M%S)}

PVEKEY=${RIG1_KEY:-/root/.ssh/id_root@proxmox}
RIG2KEY=${RIG2_KEY:-/root/.ssh/id_ed25519_192.168.0.89}
SSHOPT="-o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"
PVE="ssh -n -i $PVEKEY $SSHOPT root@${RIG1_HOST:-192.168.0.1}"
RIG2="ssh -n -i $RIG2KEY $SSHOPT root@${RIG2_HOST:-192.168.0.89}"
PVEF="ssh -f -n -i $PVEKEY $SSHOPT root@${RIG1_HOST:-192.168.0.1}"
RIG2F="ssh -f -n -i $RIG2KEY $SSHOPT root@${RIG2_HOST:-192.168.0.89}"

R=/spinning/subvol-999-disk-0/spinning
RANKSH=$R/wt-201/scripts/pp/pp_crossrig_rank.sh
LOGDIR=/spinning/subvol-999-disk-0/root/ppcrossrig   # host-side log dir
LOCAL_LOGDIR=/root/ppcrossrig                        # same dir seen from the container

ENVCOMMON="CTX=$CTX MEMFRAC=$MEMFRAC RATIO=$RATIO MODEL_NAME=$MODEL_NAME PORT=$PORT \
MAXREQ=$MAXREQ NCCL_IB=$NCCL_IB ATTN_MAIN=$ATTN_MAIN ATTN_SECOND=$ATTN_SECOND \
MASTER=${RDMA_R1:-10.10.10.1} PP_CROSSRIG_RUN_TAG=$TAG EXTRA='${EXTRA:-}' ${EXTRAENV:-}"

mkdir -p $LOCAL_LOGDIR; rm -f $LOCAL_LOGDIR/n*.log
$RIG2 "mkdir -p /root/ppcrossrig; rm -f /root/ppcrossrig/n*.log"

scp -q -i "$RIG2KEY" $SSHOPT /spinning/wt-201/scripts/pp/pp_crossrig_rank.sh \
  root@"${RIG2_HOST:-192.168.0.89}":/root/ppcrossrig/pp_crossrig_rank.sh
$RIG2 "chmod +x /root/ppcrossrig/pp_crossrig_rank.sh"

# ssh -f -n, not a backgrounded ssh: the remote command redirects its own
# output, but a foreground ssh still waits on the channel and the launcher hangs
# for the whole life of the server.
echo "=== launching node 1 (stage 1) on rig 2 / 2080 Ti ==="
$RIG2F "cd /root && NODE=1 SIDE=second $ENVCOMMON \
   setsid /root/ppcrossrig/pp_crossrig_rank.sh > /root/ppcrossrig/n1.log 2>&1 < /dev/null"
sleep 5

echo "=== launching node 0 (stage 0) on the PVE host ==="
$PVEF "mkdir -p $LOGDIR; NODE=0 SIDE=main $ENVCOMMON \
   setsid $RANKSH > $LOGDIR/n0.log 2>&1 < /dev/null"

echo "launched tag=$TAG ratio=$RATIO model=$MODEL_NAME ctx=$CTX memfrac=$MEMFRAC; bound ${WAIT}s"

res=TIMEOUT
for i in $(seq 1 $((WAIT/5))); do
  if grep -q "fired up and ready to roll" $LOCAL_LOGDIR/n0.log 2>/dev/null; then res=READY; break; fi
  if grep -qE "^Traceback|Received sigquit|CUDA out of memory" $LOCAL_LOGDIR/n0.log 2>/dev/null; then res=CRASH_NODE0; break; fi
  if $RIG2 "grep -qE '^Traceback|Received sigquit|CUDA out of memory' /root/ppcrossrig/n1.log 2>/dev/null"; then res=CRASH_NODE1; break; fi
  sleep 5
done
echo "RESULT: $res  (after $((i*5))s)"

echo "--- pipeline placement / layer split ---"
grep -hE "pipeline layer split|Pipeline placement|max_total_num_tokens|KV Cache is allocated" \
  $LOCAL_LOGDIR/n0.log 2>/dev/null | head -6
$RIG2 "grep -hE 'pipeline layer split|Pipeline placement|max_total_num_tokens|KV Cache is allocated' /root/ppcrossrig/n1.log 2>/dev/null | head -6"

echo "--- node 0 tail ---"
grep -v "server_args=ServerArgs" $LOCAL_LOGDIR/n0.log 2>/dev/null | tail -16
echo "--- node 1 tail ---"
$RIG2 "grep -v 'server_args=ServerArgs' /root/ppcrossrig/n1.log | tail -16"

if [ "$res" != READY ] || [ "$KEEP" != 1 ]; then
  echo "--- killing (own patterns only) ---"
  $PVE 'pkill -f "launch_serve[r]"; pkill -f "pp_crossrig_ran[k]"'
  $RIG2 'pkill -f "launch_serve[r]"; pkill -f "pp_crossrig_ran[k]"'
fi
[ "$res" = READY ]
