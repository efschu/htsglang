#!/bin/bash
# usage: restart_generic.sh <boot-script> <log-tag>
set -u
SELF=$$
BOOT=$1; TAG=$2
for p in $(pgrep -f "sglang.launch_server" || true); do
  [ "$p" = "$SELF" ] && continue
  kill "$p" 2>/dev/null || true
done
sleep 8
TS=$(date +%H%M%S)
LOG=/root/651-p2/logs/boot_${TAG}_$TS.log
ln -sfn "$LOG" /root/651-p2/logs/current.log
nohup setsid bash "$BOOT" > "$LOG" 2>&1 < /dev/null &
sleep 8
echo "log=$LOG"
