#!/bin/bash
# Nordstern L0 -- start all five ranks: 0-2 here, 3-4 on the second host.
# DOES NOT RUN PREFLIGHT FOR YOU. Run l0_preflight.sh first and get GREEN.
#
# Rank 4 (Vega) and rank 3 (2080 Ti) are started FIRST: they are the slowest
# to import and build, and the rendezvous tolerates late joiners better than
# it tolerates a master that has already timed out waiting.
set -u
SECOND=${SECOND:-192.168.0.89}
KEY=${KEY:-/root/.ssh/id_ed25519_192.168.0.89}
SSH="ssh -i $KEY -o IdentitiesOnly=yes"
RATIO=${RATIO:-3,2,2,1,1}
LOGDIR=${LOGDIR:-/tmp/nordstern}
mkdir -p "$LOGDIR"
R=/spinning/wt-htccl/scripts/nordstern/l0_rank.sh

for r in 4 3; do
  $SSH root@$SECOND "RANK=$r SIDE=second RATIO='$RATIO' nohup setsid /root/nordstern/l0_rank.sh \
     > /root/nordstern/r$r.log 2>&1 < /dev/null &" &
done
sleep 5
for r in 0 1 2; do
  RANK=$r SIDE=main RATIO="$RATIO" nohup setsid "$R" > "$LOGDIR/r$r.log" 2>&1 < /dev/null &
  sleep 2
done
echo "all five ranks launched; watching rank 0"
for i in $(seq 1 120); do
  grep -qE "fired up and ready to roll" "$LOGDIR/r0.log" 2>/dev/null && { echo READY; exit 0; }
  grep -qE "^Traceback|SIGQUIT" "$LOGDIR"/r*.log 2>/dev/null && { echo CRASHED; exit 1; }
  sleep 5
done
echo TIMEOUT; exit 1
