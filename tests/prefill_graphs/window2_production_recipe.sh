#!/bin/bash
# Window 2: the production recipe, prefill graph OFF vs ON.
#
# Same geometry as /root/bin/start-serving-30030.sh (TP=3 uneven DCP,
# INT8-W8A8, NEXTN spec 3, chunked prefill 2048, mamba pool 96), on port
# 30041 so it never collides with the real serving port.
#
# Arm PA: recipe as-is  -> prefill graph auto-disabled by the multimodal rule
# Arm PB: recipe + --cuda-graph-backend-prefill breakable
#
# Both arms get the content gate (short prompts, greedy) and the prefill
# throughput probe (unique ~1900-token prompts, max_tokens=1).
set -u

WT=/spinning/wt-prefill-graphs
VENV=/spinning/htsglang-gpu/.venv
OUT=${OUT:-/spinning/gpu-battery-results/2026-08-05_prefill_graphs_w2}
PORT=30041
RESERVE="5500,3800,3800"

mkdir -p "$OUT"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16

boot() {
  local arm="$1"; shift
  local log="$OUT/boot_${arm}.log"
  echo "=== booting arm $arm -> $log"
  setsid "$VENV/bin/python" -m sglang.launch_server \
    --model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8 \
    --served-model-name default \
    --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
    --rank-perf-tune phase-decode \
    --rank-auto-reserve-mib "$RESERVE" \
    --kv-cache-dtype fp8_e4m3 --context-length 262144 \
    --max-running-requests 4 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --kv-pressure-ladder auto \
    --max-mamba-cache-size 96 \
    --enable-fast-lane --retraction-policy priority \
    `# radix cache OFF for this experiment only: the content gate replays the` \
    `# same prompts and the perf probe must not be served from a prefix hit.` \
    `# hicache is off here for the same reason (production keeps both on).` \
    --disable-radix-cache \
    --trust-remote-code \
    --host 127.0.0.1 --port $PORT \
    "$@" > "$log" 2>&1 &
  echo $! > "$OUT/${arm}.pgid"
  local waited=0
  until curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200; do
    sleep 5; waited=$((waited+5))
    if [ $waited -gt 600 ]; then echo "ARM $arm FAILED TO BOOT after ${waited}s"; tail -40 "$log"; return 1; fi
    if ! kill -0 "$(cat "$OUT/${arm}.pgid")" 2>/dev/null; then echo "ARM $arm DIED"; tail -50 "$log"; return 1; fi
  done
  echo "arm $arm healthy after ${waited}s"
}

stop_arm() {
  local pg; pg=$(cat "$OUT/${1}.pgid" 2>/dev/null) || return 0
  kill -TERM -- -"$pg" 2>/dev/null
  local w=0
  while kill -0 "$pg" 2>/dev/null && [ $w -lt 90 ]; do sleep 3; w=$((w+3)); done
  kill -KILL -- -"$pg" 2>/dev/null
  sleep 8
}

run_arm() {
  local arm="$1"; shift
  boot "$arm" "$@" || return 1
  "$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" record \
      --port $PORT --out "$OUT/${arm}_run1.json" || return 1
  "$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" record \
      --port $PORT --out "$OUT/${arm}_run2.json" || return 1
  "$VENV/bin/python" "$WT/tests/prefill_graphs/prefill_perf.py" \
      --port $PORT --tokens 1900 --n 12 --seed 4242 \
      --out "$OUT/${arm}_perf.json" || return 1
  stop_arm "$arm"
}

run_arm PA || { echo "ARM PA ABORTED"; stop_arm PA; exit 1; }
run_arm PB --cuda-graph-backend-prefill breakable || { echo "ARM PB ABORTED"; stop_arm PB; exit 1; }

echo
echo "############ STATE PROBE ############"
echo "--- PA (expect the multimodal auto-disable):"
grep -m1 "incompatible with multimodal model" "$OUT/boot_PA.log" || echo "  (absent!)"
grep -c "cuda graph: True" "$OUT/boot_PA.log" || true
echo "--- PB (expect capture, and prefill batches reporting cuda graph: True):"
grep -m1 "Capture target prefill CUDA graph begin" "$OUT/boot_PB.log" || echo "  (NO PREFILL CAPTURE -- arm B did not engage!)"
grep -m1 "Capture target prefill CUDA graph end" "$OUT/boot_PB.log" || true
grep -c "incompatible with multimodal model" "$OUT/boot_PB.log" || true

echo
echo "############ A-vs-A FLOOR ############"
"$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" compare "$OUT/PA_run1.json" "$OUT/PA_run2.json"
"$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" compare "$OUT/PB_run1.json" "$OUT/PB_run2.json"

echo
echo "############ CONTENT GATE eager vs graph ############"
"$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" compare "$OUT/PA_run1.json" "$OUT/PB_run1.json"

echo
echo "############ PREFILL THROUGHPUT ############"
"$VENV/bin/python" - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
a = json.load(open(f"{out}/PA_perf.json"))
b = json.load(open(f"{out}/PB_perf.json"))
print(f"eager  : {a['prefill_tok_s_median']:8.1f} tok/s median  "
      f"({a['seconds_median']*1000:.0f} ms, cached={a['cached_tokens_total']})")
print(f"graphs : {b['prefill_tok_s_median']:8.1f} tok/s median  "
      f"({b['seconds_median']*1000:.0f} ms, cached={b['cached_tokens_total']})")
d = (b['prefill_tok_s_median'] / a['prefill_tok_s_median'] - 1) * 100
print(f"delta  : {d:+.1f}%")
sa, sb = sorted(a['prefill_tok_s_all']), sorted(b['prefill_tok_s_all'])
print(f"eager  spread min/max: {sa[0]:.1f} / {sa[-1]:.1f}")
print(f"graphs spread min/max: {sb[0]:.1f} / {sb[-1]:.1f}")
PY
