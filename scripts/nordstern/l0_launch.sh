#!/bin/bash
# Nordstern L0 -- start all five ranks: 0-2 here, 3-4 on the second host.
# Run l0_preflight.sh first and get GREEN. Env: STAGE (s1|s2), RATIO, CTX.
set -u
SECOND=${SECOND:-192.168.0.89}
KEY=${KEY:-/root/.ssh/id_ed25519_192.168.0.89}
SSH="ssh -i $KEY -o IdentitiesOnly=yes"
STAGE=${STAGE:-s1}; RATIO=${RATIO:-4,3,3,2,1}; CTX=${CTX:-4096}
LOGDIR=${LOGDIR:-/tmp/nordstern}; mkdir -p "$LOGDIR"; rm -f "$LOGDIR"/r*.log
R=/spinning/wt-htccl/scripts/nordstern/l0_rank.sh
$SSH root@$SECOND "mkdir -p /root/nordstern && rm -f /root/nordstern/r*.log"
# ranks 4 and 3 first: slowest to import; a late joiner is cheaper than a
# rendezvous master that has already timed out.
for r in 4 3; do
  $SSH root@$SECOND "RANK=$r SIDE=second STAGE=$STAGE RATIO='$RATIO' CTX=$CTX \
     nohup setsid /root/nordstern/l0_rank.sh > /root/nordstern/r$r.log 2>&1 < /dev/null &" &
done
sleep 6
for r in 0 1 2; do
  RANK=$r SIDE=main STAGE=$STAGE RATIO="$RATIO" CTX=$CTX \
    nohup setsid "$R" > "$LOGDIR/r$r.log" 2>&1 < /dev/null &
  sleep 2
done
echo "five ranks launched (stage=$STAGE ratio=$RATIO); watching rank 0"
for i in $(seq 1 150); do
  grep -qE "fired up and ready to roll" "$LOGDIR/r0.log" 2>/dev/null && { echo READY; exit 0; }
  grep -qE "^Traceback|SIGQUIT|Error:" "$LOGDIR"/r*.log 2>/dev/null && { echo CRASHED; exit 1; }
  sleep 5
done
echo TIMEOUT; exit 1
