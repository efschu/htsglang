#!/usr/bin/env bash
# #328: r8-E under the corrected window, several windows per arm in one boot,
# with the content gate applied. Prepared on the desk; see RUNSHEET.md for
# what is already measured and what a window actually buys.
#
# Card discipline: claim through /spinning/gpu-arb/ BEFORE running and release
# after. This script does not arbitrate.

set -euo pipefail

PORT="${PORT:-${1:-30081}}"
WT="${WT:-/spinning/wt-328}"
VENV="${VENV:-/spinning/htsglang-gpu/.venv}"
MODEL_ROOT="${MODEL_ROOT:-/spinning/llm_stuff/club-3090/models-cache}"
TARGET="${TARGET:-$MODEL_ROOT/Qwen3.6-27B-FP8}"
WINDOW_S="${WINDOW_S:-45}"
REPEATS="${REPEATS:-3}"          # interleaved windows per arm (the n>1 the
                                 # #328 open item asks for)
OUT="${OUT:-/spinning/gpu-battery-results/$(date +%Y-%m-%d)_328_r8redo}"

export PYTHONPATH="$WT/python"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT"
cd "$WT"

echo "=== boot (r8 recipe: NEXTN, dual-group lane + lane-spec) ==="
PORT="$PORT" OUT="$OUT" TARGET="$TARGET" \
  scripts/dual_group/r8/boot_lane_spec.sh 2>&1 | tee "$OUT/boot.log"

# Interleaved windows: arm A, arm B, arm A, ... in ONE boot, so a drift in
# the machine hits both arms alike instead of loading one of them. This is
# the n>1 the "#328 Posten 1" open item names; a single window per arm cannot
# separate the policy's E effect from window-to-window spread.
for i in $(seq 1 "$REPEATS"); do
  for arm in nochain chain; do
    echo "=== window $i / arm $arm ==="
    "$VENV/bin/python" scripts/dual_group/r8/lane_spec_window.py \
      --port "$PORT" --window-s "$WINDOW_S" --phases "$arm" \
      --out "$OUT/window_${i}_${arm}.json" 2>&1 | tee -a "$OUT/windows.log"
  done
done

echo "=== content controls (for the chain-quality gate) ==="
for arm in nospec spec; do
  "$VENV/bin/python" scripts/dual_group/r12/stock_spec_control.py \
    --port "$PORT" --"$arm" --out "$OUT/$arm.json" 2>&1 | tee -a "$OUT/control.log"
done

echo "=== chain-quality gate ==="
set +e
"$VENV/bin/python" scripts/dual_group/chain_quality_gate.py \
  --reference "$OUT/nospec.json" --candidate "$OUT/spec.json" \
  --json --out "$OUT/gate.json" | tee "$OUT/GATE.txt"
rc=$?
set -e
case "$rc" in
  0) echo "gate GREEN" ;;
  1) echo "gate RED -- the chain changed the content beyond the measured band" ;;
  2) echo "gate VOID -- instrument missing; NOT a pass, fix and re-run the arm" ;;
esac

curl -sf -m 30 "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json" || true
echo "artifacts in $OUT"
exit "$rc"
