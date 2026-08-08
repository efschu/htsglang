#!/bin/bash
# #651: stop any previous serving instance on port 31651 and start a fresh
# guard-v2-gated boot.
#
# This exists as a FILE rather than as an inline ssh command on purpose. Run
# inline, `pkill -f "sglang.launch_server.*31651"` matches the remote shell's
# own command line -- the pattern is literally in it -- so the session kills
# itself before it can launch anything. Keeping the patterns inside a script
# keeps them out of the invoking command line. The self-PID exclusion below is
# the second belt: this script's own name must never be its own victim.
set -u
SELF=$$

stop_pattern() {
  for p in $(pgrep -f "$1" || true); do
    [ "$p" = "$SELF" ] && continue
    [ "$p" = "$PPID" ] && continue
    kill "$p" 2>/dev/null || true
  done
}

stop_pattern "sglang.launch_server.*31651"
sleep 5
stop_pattern "sglang.launch_server.*31651"   # SIGKILL stragglers next round
sleep 2

TS=$(date +%H%M%S)
LOG=/root/651-p2/logs/boot_gated_a_$TS.log
ln -sfn "$LOG" /root/651-p2/logs/current.log
nohup setsid bash /root/651-p2/scripts/boot_v2gated.sh > "$LOG" 2>&1 < /dev/null &
sleep 8
echo "log=$LOG"
pgrep -f "launch_server" > /dev/null && echo "server process present" \
  || echo "no server process yet (guard may still be running)"
