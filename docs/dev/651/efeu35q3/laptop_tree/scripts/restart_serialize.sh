#!/bin/bash
# #651: restart serving with AMD_SERIALIZE_KERNEL=3 so an async HIP fault is
# attributed to the kernel that actually raised it, not to the next error check.
set -u
SELF=$$
for p in $(pgrep -f "sglang.launch_server" || true); do
  [ "$p" = "$SELF" ] && continue
  kill "$p" 2>/dev/null || true
done
sleep 6
TS=$(date +%H%M%S)
LOG=/root/651-p2/logs/boot_serialize_$TS.log
ln -sfn "$LOG" /root/651-p2/logs/current.log
nohup setsid bash /root/651-p2/scripts/boot_v2gated_serialize.sh > "$LOG" 2>&1 < /dev/null &
sleep 8
echo "log=$LOG"
