#!/bin/bash
# Stop the hung post-cleanup smoke run. In a file, not on an ssh command line:
# cmdline patterns match the caller's own shell and kill the session.
set -u
SELF=$$
for p in $(ls /proc | grep -E '^[0-9]+$'); do
  [ "$p" = "$SELF" ] && continue
  [ -r "/proc/$p/cmdline" ] || continue
  CMD=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
  case "$CMD" in
    *smoke655*|*SMOKE_OK*|*hello.py*)
      kill -TERM "$p" 2>/dev/null && echo "stopped $p: ${CMD:0:50}"
      ;;
  esac
done
sleep 2
echo "remaining omp: $(ps -eo args --no-headers | grep -c '[o]mp --model')"
echo "resets: $(dmesg -T | grep -c 'GPU reset(')"
echo "status: $(curl -s -m 8 localhost:31651/ondemand/status)"
