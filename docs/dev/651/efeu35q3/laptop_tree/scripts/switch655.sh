#!/bin/bash
# Stop the ladder run and start the CTX-wall run. Lives in a FILE on purpose:
# a pattern typed into an interactive ssh command line is also present in that
# shell's own /proc/<pid>/cmdline, so `pkill -f` and cmdline scans match the
# caller and kill the session. Running from a file keeps the patterns out of
# the caller's cmdline.
set -u
SELF=$$

for p in $(ls /proc | grep -E '^[0-9]+$'); do
  [ "$p" = "$SELF" ] && continue
  [ -r "/proc/$p/cmdline" ] || continue
  CMD=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
  case "$CMD" in
    *ladder655*|*needle655*|*drive655*|*accept_omp655*)
      kill -TERM "$p" 2>/dev/null && echo "stopped $p: ${CMD:0:70}"
      ;;
  esac
done

sleep 2
chmod +x /root/651-p2/scripts/ctxwall655.sh
setsid nohup /root/651-p2/scripts/ctxwall655.sh > /dev/null 2>&1 < /dev/null &
sleep 4
echo "started: $(ls -t /root/651-p2/results/ctxwall655_*.log | head -1)"
