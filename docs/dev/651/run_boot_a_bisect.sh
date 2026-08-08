#!/bin/bash
# #651 boot A (guard-gated): coherence probe, then prefill-length crash bisection.
#
# The bisection is destructive by design -- after `HIP error: unspecified launch
# failure` the HIP context is dead -- so it is the LAST thing this boot does.
# Floors are measured on a separate boot (run_boot_b_floors.sh) at lengths the
# bisection proved safe.
set -u
cd /root/651-p2
TS=$(date +%H%M%S)
LOG=/root/651-p2/logs/boot_gated_a_$TS.log
ln -sfn "$LOG" /root/651-p2/logs/current.log

source /root/lh/venv/bin/activate

echo "[$(date +%T)] booting guard-gated server (boot A)"
nohup setsid bash /root/651-p2/scripts/boot_v2gated.sh > "$LOG" 2>&1 < /dev/null &

# Wait for readiness. A guard failure aborts the boot script, so a server that
# never comes up within the window is reported as such rather than waited on
# forever.
for i in $(seq 1 180); do
  code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:31651/health || true)
  if [ "$code" = "200" ]; then echo "[$(date +%T)] server ready after ${i}0s"; break; fi
  # The boot script runs the sanity guard BEFORE exec'ing the server, so for
  # the first seconds neither the server process nor the port exists yet while
  # the boot is perfectly healthy. Liveness therefore means "the boot script OR
  # the server is still alive", not "the server is already up".
  if ! pgrep -f "boot_v2gated.sh|sglang.launch_server.*31651" > /dev/null; then
    echo "[$(date +%T)] boot process gone before readiness -- guard refusal or crash"
    grep -E "GUARD:|Error|error" "$LOG" | tail -20
    exit 1
  fi
  sleep 10
done

code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:31651/health || true)
if [ "$code" != "200" ]; then echo "[$(date +%T)] server never became ready"; exit 1; fi

echo "[$(date +%T)] === coherence probe on the guard-gated boot ==="
python -u scripts/probe.py 31651 2>&1 | tee results/probe_gated_a_$TS.txt
echo "[$(date +%T)] probe exit=$?"

echo "[$(date +%T)] === prefill-length crash bisection (destructive) ==="
python -u scripts/crash_bisect_prefill.py --port 31651 \
  --lengths 128,256,512,768,1024,1152,1408,1792,2048,2560 --repeats 3 \
  2>&1 | tee results/crash_bisect_$TS.txt

echo "[$(date +%T)] boot A done; server state:"
curl -s -m 5 -o /dev/null -w "health=%{http_code}\n" http://127.0.0.1:31651/health || true
