#!/usr/bin/env bash
# #340 posten 1 -- card proof for the lane device guard.
#
# Two steps, in this order, because the second one cannot show a SUCCESS on
# this rig and the first one can:
#
#   MICRO  guard_micro.py: an ARCHITECTURE-NEUTRAL triton kernel launched on a
#          foreign card. Case A (no guard) must reproduce the pointer error,
#          case B (guard) and case C (the real LaneColumnParallelShell over two
#          cards) must succeed AND be numerically right. This is the proof that
#          the guard is the fix, with the fp8e4nv limit held out of the frame.
#
#   FP8    the #336 arm-C repro boot, unchanged except for the worktree, WITH a
#          driven lane job (the repro trap: serving traffic never enters
#          LaneColumnParallelShell.forward, so a boot without a lane job proves
#          nothing). Expected AFTER the fix: the pointer ValueError at
#          dual_group_lane.py forward is GONE and the first exception is the
#          named architecture limit "type fp8e4nv not supported in this
#          architecture" on the sm86 part -- a different wall, one card behind.
#
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
source "$WT/scripts/dual_group/r7c/common.sh"
# shellcheck source=../dcp_report.sh
source "$HERE/../dcp_report.sh"

STEP="${1:-all}"
BUDGET_S="${BUDGET_S:-900}"
T_START=$(date +%s)
left() { echo $(( BUDGET_S - ( $(date +%s) - T_START ) )); }

# --- MICRO ------------------------------------------------------------------
if [ "$STEP" = "all" ] || [ "$STEP" = "micro" ]; then
  echo "--- posten1/micro: arch-neutral triton launch on a foreign card"
  timeout -k 20 300 "$PY" "$HERE/guard_micro.py" \
    >"$OUT/posten1/guard_micro.json" 2>"$OUT/posten1/guard_micro.err"
  echo "micro rc=$? (stdout json in $OUT/posten1/guard_micro.json)"
  "$PY" - "$OUT/posten1/guard_micro.json" <<'PY'
import json
import sys

# guard_micro.py prints one human line per case first and the JSON blob last,
# so the report is readable while the run happens; the machine part is the
# LAST line, not the whole file.
try:
    lines = [ln for ln in open(sys.argv[1]).read().splitlines() if ln.strip()]
    r = json.loads(lines[-1])
except Exception as exc:  # noqa: BLE001
    print(f"  micro: unreadable json ({exc})")
    raise SystemExit(0)
print(f"  visible={r.get('visible')} host={r.get('host_device')} "
      f"foreign={r.get('foreign_device')}")
for c in r.get("cases", []):
    print(f"  {c['case']:26s} {c['result']:8s} "
          f"delta={c['max_abs_delta']} {c['exc'] or ''}")
PY
fi

# --- FP8 --------------------------------------------------------------------
if [ "$STEP" = "all" ] || [ "$STEP" = "fp8" ]; then
  if [ "$(left)" -lt 420 ]; then
    echo "--- posten1/fp8: SKIPPED, only $(left)s of budget left"
    exit 0
  fi
  MODEL="$MODEL_ROOT/Qwen3.6-27B-FP8"
  PORT=30395
  LOG="$OUT/logs/p1_fp8_lane.server.log"
  PIDF=/tmp/340gpu-fp8.pid
  BOOT_TIMEOUT_S=420

  CARDS="$("$PY" - <<'PY'
import torch

n = torch.cuda.device_count()
big = max(range(n), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
small = [i for i in range(n) if i != big]
print(f"{big} {small[0]}")
PY
)"
  read -r BIG SMALL0 <<< "$CARDS"
  echo "--- posten1/fp8: two-card FP8 lane, host cuda:$BIG part cuda:$SMALL0"

  cd "$WT" || exit 1
  setsid "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --tokenizer-path "$MODEL" \
    --tp-size 2 --rank-gpu-id "$BIG,$SMALL0" \
    --rank-tp-ratio 6,1 --rank-gpu-memory-mib 27000,9500 \
    --attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 --context-length 4096 --trust-remote-code \
    --max-running-requests 1 \
    --dual-group-lane --dual-group-lane-budget-mib 600 \
    --dual-group-lane-part-gpu-id "$BIG,$SMALL0" \
    --enable-metrics --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
  PID=$!; echo "$PID" > "$PIDF"

  t0=$(date +%s); up=0
  while [ $(( $(date +%s) - t0 )) -lt "$BOOT_TIMEOUT_S" ]; do
    curl -sf -m 10 "http://127.0.0.1:$PORT/health_generate" >/dev/null 2>&1 \
      && { up=1; break; }
    kill -0 "$PID" 2>/dev/null || break
    sleep 5
  done
  echo "  fp8 boot: up=$up after $(( $(date +%s) - t0 ))s"
  # HARNESS DUTY (#345): this arm carries --rank-tp-ratio, so the env DCP
  # flags DO bite here and not in a control without the ratio. Say so.
  report_dcp "p1_fp8_lane" "$LOG" echo

  if [ "$up" = 1 ]; then
    # DRIVE A LANE JOB. Without this the forward under test never runs.
    timeout -k 20 240 "$PY" "$WT/scripts/dual_group/fam2/family_gate.py" \
      --port "$PORT" --tokenizer "$MODEL" --tokens 12 --deadline-s 200 \
      --out "$OUT/posten1/fp8_gate.json" >"$OUT/posten1/fp8_gate.txt" 2>&1
    tail -6 "$OUT/posten1/fp8_gate.txt"
  fi

  grep -n -E "Pointer argument|cannot be accessed from Triton|illegal memory|AcceleratorError|dual_group_lane.py|w8a8_block_fp8|fp8e4nv|not supported in this architecture" \
    "$LOG" > "$OUT/posten1/fp8_exceptions.txt" 2>/dev/null
  grep -E "lane spans cards|forcing EAGER|part rank .* loaded on cuda|model assembled|lane 0 ready" \
    "$LOG" > "$OUT/posten1/fp8_contract_lines.txt" 2>/dev/null
  echo "  --- first exception lines ---"
  head -14 "$OUT/posten1/fp8_exceptions.txt"

  kill -TERM "-$(ps -o pgid= "$PID" | tr -d ' ')" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
  for _ in $(seq 1 40); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -KILL "$PID" 2>/dev/null; rm -f "$PIDF"
fi

echo "posten1 done after $(( $(date +%s) - T_START ))s"
