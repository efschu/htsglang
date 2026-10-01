#!/bin/bash
# efeu-TP14: run idle_probe.sh only in a CLEAN window: model loaded (state up),
# inflight 0, idle 45..540 s, and the SAME backend pid still up afterwards.
# Passive: sends no request, restarts nothing. Retries until one clean window.
#   probe_when_idle.sh <label> [seconds]
L=$1; S=${2:-30}
st() { curl -s -m5 localhost:31651/ondemand/status; }
while true; do
  J=$(st)
  ok=$(echo "$J" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["pid"] if d["state"]=="up" and d["inflight"]==0 and 45<d["idle_seconds"]<540 else "")' 2>/dev/null)
  if [ -n "$ok" ]; then
    OUT=$(/root/efeu35q3/energy/idle_probe.sh $S $L 2>&1)
    J2=$(st)
    ok2=$(echo "$J2" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["pid"] if d["state"]=="up" and d["inflight"]==0 else "")' 2>/dev/null)
    if [ "$ok2" = "$ok" ]; then echo "$OUT"; echo "clean window, backend pid $ok"; exit 0; fi
    echo "window disturbed (pid $ok -> '$ok2'), retrying" >&2
  fi
  sleep 20
done
