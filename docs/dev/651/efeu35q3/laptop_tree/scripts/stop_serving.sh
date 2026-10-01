#!/bin/bash
# Stop any serving instance. Lives in a FILE because an inline pkill pattern
# matches the invoking ssh command line and kills the session itself (#651).
set -u
SELF=$$
for p in $(pgrep -f "sglang.launch_server" || true); do
  [ "$p" = "$SELF" ] && continue
  kill "$p" 2>/dev/null || true
done
sleep 15
pgrep -f "[s]glang.laun""ch_server" >/dev/null && echo "still running" || echo "stopped"
