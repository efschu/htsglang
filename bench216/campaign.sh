#!/usr/bin/env bash
# Interleaved cold-boot campaign over the MLP split.
# Order gives both the boot-to-boot A-vs-A noise floor (the _A/_B pairs of the
# same arm) and interleaving between the two arms under test.
set -u
D=/spinning/wt-knee-guard/bench216
PORT=30000
run_arm () {
  local ARM="$1" MLP="$2"
  echo "=== $(date +%T) arm=$ARM mlp=$MLP ==="
  "$D/launch.sh" "$ARM" "$MLP" "$PORT" >/dev/null
  local PID; PID=$(cat "$D/logs/${ARM}.pid")
  local L="$D/logs/${ARM}.log" ok=0
  for i in $(seq 1 150); do
    grep -q "The server is fired up and ready to roll" "$L" 2>/dev/null && { ok=1; break; }
    grep -qE "Traceback|CUDA out of memory" "$L" 2>/dev/null && break
    kill -0 "$PID" 2>/dev/null || break
    sleep 4
  done
  if [ "$ok" != 1 ]; then
    echo "BOOT-FAILED $ARM"; grep -nE "Error|Exception|RuntimeError" "$L" | tail -5
    kill "$PID" 2>/dev/null; sleep 8; return 1
  fi
  # clock/temperature per point, before and after the measurement
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw \
     --format=csv,noheader > "$D/logs/${ARM}.clocks_before.csv"
  /spinning/htsglang-gpu/.venv/bin/python "$D/measure.py" \
     --port "$PORT" --arm "$ARM" --out "$D/logs/${ARM}.json" --reps 3
  nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw \
     --format=csv,noheader > "$D/logs/${ARM}.clocks_after.csv"
  grep -oE "#new-token: [0-9]+, #cached-token: [0-9]+" "$L" \
     | sort | uniq -c > "$D/logs/${ARM}.cachedtok.txt"
  kill "$PID" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
  sleep 6
}
for spec in "$@"; do
  ARM="${spec%%:*}"; MLP="${spec##*:}"
  run_arm "$ARM" "$MLP" || echo "SKIP $ARM"
done
echo "=== $(date +%T) CAMPAIGN DONE ==="
