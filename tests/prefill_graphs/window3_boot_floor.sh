#!/bin/bash
# Window 3: the boot-to-boot floor, which windows 1 and 2 both lacked.
#
# Window 2 compared an eager BOOT against a graph BOOT and found 4/8 text
# divergences. That number only means something if two EAGER boots agree with
# each other. NOTE_452 §2 experiment 2 flags exactly this floor as missing.
# If E1 vs E2 already diverges, the graph-vs-eager result is boot noise and
# must be withdrawn.
#
# Arms: E1 eager, E2 eager (identical flags), G graphs.
# Each arm also runs the prefill probe at three prompt sizes, so the perf
# question gets its own boot-to-boot noise floor for free.
set -u

WT=/spinning/wt-prefill-graphs
VENV=/spinning/htsglang-gpu/.venv
OUT=${OUT:-/spinning/gpu-battery-results/2026-08-05_prefill_graphs_w3}
PORT=30042
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
  echo "=== booting arm $arm"
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
    --kv-pressure-ladder auto --max-mamba-cache-size 96 \
    --enable-fast-lane --retraction-policy priority \
    `# pin the seed so boot-to-boot RNG cannot be blamed for a divergence` \
    --random-seed 12345 \
    --disable-radix-cache --trust-remote-code \
    --host 127.0.0.1 --port $PORT \
    "$@" > "$log" 2>&1 &
  echo $! > "$OUT/${arm}.pgid"
  local waited=0
  until curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200; do
    sleep 5; waited=$((waited+5))
    [ $waited -gt 600 ] && { echo "ARM $arm BOOT TIMEOUT"; tail -40 "$log"; return 1; }
    kill -0 "$(cat "$OUT/${arm}.pgid")" 2>/dev/null || { echo "ARM $arm DIED"; tail -50 "$log"; return 1; }
  done
  echo "arm $arm healthy after ${waited}s"
}

stop_arm() {
  local pg; pg=$(cat "$OUT/${1}.pgid" 2>/dev/null) || return 0
  kill -TERM -- -"$pg" 2>/dev/null
  local w=0; while kill -0 "$pg" 2>/dev/null && [ $w -lt 90 ]; do sleep 3; w=$((w+3)); done
  kill -KILL -- -"$pg" 2>/dev/null; sleep 8
}

run_arm() {
  local arm="$1"; shift
  boot "$arm" "$@" || return 1
  "$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" record \
      --port $PORT --out "$OUT/${arm}_gate.json" || return 1
  if [ "${PERF:-1}" = "1" ]; then
    for sz in 256 900 1900; do
      "$VENV/bin/python" "$WT/tests/prefill_graphs/prefill_perf.py" \
          --port $PORT --tokens $sz --n 12 --seed 4242 \
          --out "$OUT/${arm}_perf_${sz}.json" || return 1
    done
    # Agent-like arrivals: short prompts, 4 in flight, so the scheduler
    # actually forms bs>1 prefill batches. This is the last regime where a
    # captured prefill could plausibly pay -- launch-train bound, not GEMM
    # bound (the 68-75% collective share of the prefill window, #252).
    "$VENV/bin/python" "$WT/tests/prefill_graphs/prefill_perf.py" \
        --port $PORT --tokens 256 --n 24 --seed 777 --concurrency 4 \
        --out "$OUT/${arm}_perf_256c4.json" || return 1
  fi
  stop_arm "$arm"
}

run_arm E1 || { stop_arm E1; exit 1; }
run_arm E2 || { stop_arm E2; exit 1; }
run_arm G --cuda-graph-backend-prefill breakable || { stop_arm G; exit 1; }

# Determinism question: can this recipe ever be byte-strict under graphs, or
# only distribution-level? --enable-deterministic-inference is NOT in the
# breakable rule list (server_args.py:8487) -- only tc_piecewise rejects it --
# so BCG + deterministic is a legal combination. Content gate only, no perf.
PERF=0 run_arm ED --enable-deterministic-inference || { stop_arm ED; exit 1; }
PERF=0 run_arm GD --enable-deterministic-inference \
    --cuda-graph-backend-prefill breakable || { stop_arm GD; exit 1; }

echo
echo "########## BOOT-TO-BOOT CONTENT FLOOR (eager vs eager) ##########"
echo "If this FAILS, the window-2 graph-vs-eager divergence is boot noise."
"$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" compare "$OUT/E1_gate.json" "$OUT/E2_gate.json"
echo
echo "########## GRAPHS vs EAGER (same gate) ##########"
"$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" compare "$OUT/E1_gate.json" "$OUT/G_gate.json"

echo
echo "########## DETERMINISTIC INFERENCE: byte-strict or only distribution? ##########"
echo "--- eager-det vs graphs-det:"
"$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" compare "$OUT/ED_gate.json" "$OUT/GD_gate.json"
echo "--- eager-det vs eager (does determinism change eager's own output?):"
"$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" compare "$OUT/E1_gate.json" "$OUT/ED_gate.json"

echo
echo "########## PREFILL THROUGHPUT, with boot-to-boot noise floor ##########"
"$VENV/bin/python" - "$OUT" <<'PY'
import json, sys, statistics as st
out = sys.argv[1]
print(f"{'size':>6} {'eager E1':>10} {'eager E2':>10} {'graphs G':>10} "
      f"{'noise E1E2':>11} {'G vs E-mean':>12}")
for sz in ("256", "900", "1900", "256c4"):
    v = {}
    for arm in ("E1", "E2", "G"):
        d = json.load(open(f"{out}/{arm}_perf_{sz}.json"))
        v[arm] = st.median(d["prefill_tok_s_all"])
        if d["cached_tokens_total"]:
            print(f"  !! {arm} size {sz}: cache hits, contaminated")
    noise = (v["E2"] / v["E1"] - 1) * 100
    emean = (v["E1"] + v["E2"]) / 2
    delta = (v["G"] / emean - 1) * 100
    print(f"{sz:>6} {v['E1']:>10.1f} {v['E2']:>10.1f} {v['G']:>10.1f} "
          f"{noise:>+10.1f}% {delta:>+11.1f}%")
print()
print("A delta smaller in magnitude than the E1/E2 noise column is NOT a result.")
print()
print("Concurrent point (256 tok x4 in flight), AGGREGATE throughput --")
print("per-request rates understate this regime, so read this line instead:")
agg = {}
for arm in ("E1", "E2", "G"):
    d = json.load(open(f"{out}/{arm}_perf_256c4.json"))
    agg[arm] = d["aggregate_tok_s"]
    print(f"  {arm:>3}: {d['aggregate_tok_s']:8.1f} tok/s aggregate "
          f"over {d['wall_seconds']:.1f}s")
n = (agg["E2"] / agg["E1"] - 1) * 100
dl = (agg["G"] / ((agg["E1"] + agg["E2"]) / 2) - 1) * 100
print(f"  eager boot-to-boot noise {n:+.1f}%  |  graphs vs eager {dl:+.1f}%")
PY
