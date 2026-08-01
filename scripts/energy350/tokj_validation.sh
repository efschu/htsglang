#!/usr/bin/env bash
# #350 phase 2: turnkey tok/J validation of the energy solver objective.
#
# Prepared on the desk, NOT run here (no cards). It boots the two plans the
# solver produces for the SAME rig -- the throughput optimum and the energy
# optimum -- measures each with the #146 energy harness, and prints whether
# the measured ranking agrees with the solver's prediction.
#
# The claim under test is a TRADE, not a win: the energy plan should measure
# FEWER J/token and FEWER tok/s. A run where the energy plan wins both, or
# loses both, falsifies the solver's energy model on this rig.
#
# Usage:  scripts/energy350/tokj_validation.sh [PORT]
# Card discipline: claim the cards through /spinning/gpu-arb/ BEFORE running
# this and release them after; this script does not arbitrate.

set -euo pipefail

PORT="${1:-31350}"
WT="${WT:-/spinning/wt-350-p2}"
VENV="${VENV:-/spinning/htsglang-gpu/.venv}"
MODEL_ROOT="${MODEL_ROOT:-/spinning/llm_stuff/club-3090/models-cache}"
MODEL="${MODEL:-$MODEL_ROOT/Qwen3.6-27B-FP8}"
OUT="${OUT:-/spinning/gpu-battery-results/$(date +%Y-%m-%d)_350_tokj}"

export PYTHONPATH="$WT/python"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT"
cd "$WT"

boot_and_measure() {
  local arm="$1" objective="$2"
  local log="$OUT/$arm.server.log"
  echo "=== [$arm] --objective $objective ==="
  setsid "$VENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
    --objective "$objective" \
    --rank-auto-reserve-mib 3000,2700,2700 \
    --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
    --max-running-requests 16 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --enable-metrics --host 127.0.0.1 --port "$PORT" \
    > "$log" 2>&1 &
  echo $! > "$OUT/$arm.pid"

  # Wait for READY (bounded: a boot that does not come up is a result too).
  for _ in $(seq 1 180); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 5
  done
  curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || {
    echo "[$arm] server did not come up; see $log"; return 1; }

  # The installed vector, read off the log rather than derived from flags.
  grep -iE "rank-mlp-ratio|mlp unit|installed vector" "$log" | tail -3 \
    | tee "$OUT/$arm.vector.txt"

  # #146 harness: tok/s + J/token, per card via NVML.
  "$VENV/bin/python" -m sglang.srt.planner.energy \
    --base-url "http://127.0.0.1:$PORT" \
    > "$OUT/$arm.energy.txt" 2>&1 || true

  # py-spy before the kill (standing rule), then stop only our own group.
  "$VENV/bin/python" -m py_spy dump --pid "$(cat "$OUT/$arm.pid")" \
    > "$OUT/$arm.pyspy.txt" 2>&1 || true
  kill -- "-$(cat "$OUT/$arm.pid")" 2>/dev/null || true
  sleep 20
}

boot_and_measure throughput throughput
boot_and_measure energy energy

echo
echo "=== VERDICT ==="
"$VENV/bin/python" "$WT/scripts/energy350/compare_tokj.py" \
  --throughput "$OUT/throughput.energy.txt" \
  --energy "$OUT/energy.energy.txt" | tee "$OUT/VERDICT.txt"
