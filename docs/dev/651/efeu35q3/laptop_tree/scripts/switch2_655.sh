#!/bin/bash
# Stop the finalize/agent run and start a prompt-DEPTH ladder at CTX=32768.
#
# Why: the warm 10k probe wedged at CTX=32768, while a 1k probe at the same
# setting was clean. So context size is not the only axis -- prompt depth is a
# second one, and the usable envelope is somewhere between 1k and 10k. The
# coding agent's prompt lands in exactly that gap, so the gap has to be
# measured before the agent can be sized.
#
# In a file, not on an ssh command line: cmdline patterns match the caller's
# own shell and kill the session.
set -u
SELF=$$

for p in $(ls /proc | grep -E '^[0-9]+$'); do
  [ "$p" = "$SELF" ] && continue
  [ -r "/proc/$p/cmdline" ] || continue
  CMD=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
  case "$CMD" in
    *finalize655*|*accept_omp655*|*needle655*|*"omp --model"*)
      kill -TERM "$p" 2>/dev/null && echo "stopped $p: ${CMD:0:60}"
      ;;
  esac
done

sleep 3
setsid nohup /root/651-p2/scripts/ladder655.sh > /dev/null 2>&1 < /dev/null &
sleep 4
echo "started: $(ls -t /root/651-p2/results/ladder655_*.log | head -1)"
