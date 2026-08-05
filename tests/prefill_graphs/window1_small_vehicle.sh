#!/bin/bash
# Window 1: prefill CUDA graph content gate on the small vehicle.
#
# Vehicle is Qwen3.5-2B, which shares the production model's architecture
# (Qwen3_5ForConditionalGeneration, vision tower present, hybrid linear
# attention), so it hits exactly the same config-time gate that makes
# production run prefill eagerly:
#   "Breakable CUDA graph is incompatible with multimodal model"
#   (server_args.py _disable_breakable_cudagraph_if_incompatible)
#
# Passing --cuda-graph-backend-prefill explicitly LOCKS the prefill phase and
# so bypasses that auto-disable cascade. No source change is needed to run
# this experiment -- that is the point: arm B is the unmodified tree with one
# extra flag.
#
# Arm A: default resolution (prefill graph disabled -> eager)
# Arm B: --cuda-graph-backend-prefill breakable
#
# Each arm records the prompt set TWICE. A-vs-A must be identical inside each
# arm before A-vs-B means anything.
set -u

WT=/spinning/wt-prefill-graphs
VENV=/spinning/htsglang-gpu/.venv
MODEL=/spinning/llm_stuff/club-3090/models-cache/Qwen3.5-2B
OUT=${OUT:-/spinning/gpu-battery-results/2026-08-05_prefill_graphs_w1}
PORT=30040
# Address the 5090 by UUID, not by index: torch's device order and NVML's
# are known to diverge on this box, so an index would be a guess.
GPU_UUID=GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d

mkdir -p "$OUT"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"

boot() {
  local arm="$1"; shift
  local log="$OUT/boot_${arm}.log"
  echo "=== booting arm $arm -> $log"
  setsid "$VENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name default \
    --tp-size 1 \
    --context-length 8192 \
    --chunked-prefill-size 2048 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.60 \
    --max-running-requests 4 \
    `# radix cache off: the gate replays the SAME prompts, and a prefix hit` \
    `# would skip the very prefill under test on the second pass` \
    --disable-radix-cache \
    --trust-remote-code \
    --host 127.0.0.1 --port $PORT \
    "$@" > "$log" 2>&1 &
  echo $! > "$OUT/${arm}.pgid"
  local waited=0
  until curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200; do
    sleep 5; waited=$((waited+5))
    if [ $waited -gt 420 ]; then echo "ARM $arm FAILED TO BOOT after ${waited}s"; tail -30 "$log"; return 1; fi
    if ! kill -0 "$(cat "$OUT/${arm}.pgid")" 2>/dev/null; then echo "ARM $arm DIED"; tail -40 "$log"; return 1; fi
  done
  echo "arm $arm healthy after ${waited}s"
}

stop_arm() {
  local arm="$1"
  local pg; pg=$(cat "$OUT/${arm}.pgid" 2>/dev/null) || return 0
  kill -TERM -- -"$pg" 2>/dev/null
  local w=0
  while kill -0 "$pg" 2>/dev/null && [ $w -lt 60 ]; do sleep 2; w=$((w+2)); done
  kill -KILL -- -"$pg" 2>/dev/null
  sleep 5
}

run_arm() {
  local arm="$1"; shift
  boot "$arm" "$@" || return 1
  "$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" record \
      --port $PORT --out "$OUT/${arm}_run1.json" || return 1
  "$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" record \
      --port $PORT --out "$OUT/${arm}_run2.json" || return 1
  stop_arm "$arm"
}

run_arm A || { echo "ARM A ABORTED"; stop_arm A; exit 1; }
run_arm B --cuda-graph-backend-prefill breakable || { echo "ARM B ABORTED"; stop_arm B; exit 1; }

echo
echo "############ STATE PROBE: did each arm really do what it claims? ############"
echo "--- arm A: expect the multimodal auto-disable to have fired"
grep -c "Disable prefill CUDA graph because" "$OUT/boot_A.log" || true
grep -m1 "incompatible with multimodal model" "$OUT/boot_A.log" || echo "  (no multimodal line in A)"
echo "--- arm B: expect NO disable line, and real prefill capture activity"
grep -c "Disable prefill CUDA graph because" "$OUT/boot_B.log" || true
grep -iE "prefill.*(captur|graph)" "$OUT/boot_B.log" | head -5

echo
echo "############ A-vs-A FLOOR (oracle validity) ############"
echo "--- arm A floor:"
"$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" compare "$OUT/A_run1.json" "$OUT/A_run2.json"
echo "--- arm B floor:"
"$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" compare "$OUT/B_run1.json" "$OUT/B_run2.json"

echo
echo "############ CONTENT GATE: eager vs prefill-graph ############"
"$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" compare "$OUT/A_run1.json" "$OUT/B_run1.json"
