#!/bin/bash
# Window 4: interleaved eager/graphs, because window 3's block order was not
# a recoverable design.
#
# Window 3 ran E1, E2, G in blocks, so any slow drift between the start and
# the end of the window landed entirely on the treatment: the graph arm ran
# last every time. That is the defect this window fixes.
#
# Here the arms alternate -- E, G, E, G, E, G -- so each treatment gets early,
# middle and late positions. Drift then loads on BOTH arms rather than on one,
# and what is left of it shows up in the eager replicate spread, which is the
# floor. Drift elimination is interleaving's whole job.
#
# Scoring is ms per FIXED unit of work: an identical prompt set, same count,
# same order, in every arm, paired E_i against G_i. That is valid because the
# power limit is identical across every run (200/400/200 W).
#
# Clock and power draw are recorded together as DIAGNOSTIC ANNOTATION ONLY.
# At a fixed power limit a lower clock often means MORE work per cycle, not
# less -- a power-limited card downclocks under heavy load, and low clock with
# high power is a busy card while low clock with low power is an idle one.
# Nothing here is rejected on clock, and nothing is ever normalised by it.
set -u

WT=/spinning/wt-prefill-graphs
VENV=/spinning/htsglang-gpu/.venv
OUT=${OUT:-/spinning/gpu-battery-results/2026-08-05_prefill_graphs_w4}
PORT=30043
# Matches the production default as of 2026-08-05. The 3080 term was raised
# 3800 -> 4200 because the boot's own demand model derives 4160 MiB
# (activation + graph capture + GDN prefill scratch) and warned that 3800 was
# 360 MiB short with the 96-slot mamba pool. Booting the arms at the stale
# 3800 would measure a differently-sized KV pool than production runs.
RESERVE="${RESERVE:-5500,4200,4200}"
REPS="${REPS:-3}"
# Extra flags appended to BOTH arms of every pair, so the treatment stays the
# prefill backend alone. Used for the determinism variant
# (EXTRA=--enable-deterministic-inference), which must go to eager and graphs
# alike or the pair is measuring two things at once.
EXTRA="${EXTRA:-}"

mkdir -p "$OUT"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16

# TRANSPORT is set EXPLICITLY, never inherited. Production now exports
# SGLANG_BARLINK=1 (user decision 2026-08-05: production runs barlink, usage is
# the soak), so a script that merely stayed silent would pick the transport up
# from whatever shell launched it -- and an arm that silently switched
# transport would not be comparable with window 3, nor necessarily with its own
# pair partner. Both failure modes are silent, which is why this is explicit.
#
# Default is nccl: window 4 answers the PREFILL-GRAPH question, and it must
# hold the transport fixed at window 3's value to be comparable with it.
# TRANSPORT=barlink runs the other half of the 2x2 (see NOTE_515 section 6a).
TRANSPORT="${TRANSPORT:-nccl}"
case "$TRANSPORT" in
  nccl)    unset SGLANG_BARLINK ;;
  barlink) export SGLANG_BARLINK=1 ;;
  *) echo "TRANSPORT must be nccl or barlink, got '$TRANSPORT'"; exit 2 ;;
esac
export PREFILL_GRAPHS_TRANSPORT="$TRANSPORT"   # stamped into every artifact
echo "transport=$TRANSPORT (SGLANG_BARLINK=${SGLANG_BARLINK:-unset})"
echo "$TRANSPORT" > "$OUT/transport.txt"

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
  # Content gate on the FIRST pair only. Window 3 already answered the content
  # question with a passing boot-to-boot floor; repeating it on every rep would
  # spend a minute per arm re-confirming it.
  if [ "${GATE:-0}" = "1" ]; then
    "$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" record \
        --port $PORT --out "$OUT/${arm}_gate.json" || return 1
  fi
  # TWO workload points only, each sized to land in the 5-20 s band where the
  # spread is already saturated. More points, or longer points, buy nothing
  # and turn a few-minute window into a battery.
  #   1900   : long single-stream prefill, GEMM-bound
  #   256c4  : short prompts, 4 in flight, launch-train bound (the regime the
  #            barlink hypothesis is actually about)
  "$VENV/bin/python" "$WT/tests/prefill_graphs/prefill_perf.py" \
      --port $PORT --tokens 1900 --prompts 8 --passes 1 \
      --seconds 5 --seed 4242 \
      --out "$OUT/${arm}_perf_1900.json" || return 1
  "$VENV/bin/python" "$WT/tests/prefill_graphs/prefill_perf.py" \
      --port $PORT --tokens 256 --prompts 24 --passes 2 --concurrency 4 \
      --seconds 5 --seed 777 \
      --out "$OUT/${arm}_perf_256c4.json" || return 1
  stop_arm "$arm"
}

# E, G, E, G, E, G -- alternating, so neither treatment owns the cool end.
for i in $(seq 1 "$REPS"); do
  G=0; [ "$i" = "1" ] && G=1   # content gate on the first pair only
  GATE=$G run_arm "E${i}" ${EXTRA} || { stop_arm "E${i}"; exit 1; }
  GATE=$G run_arm "G${i}" ${EXTRA} --cuda-graph-backend-prefill breakable \
      || { stop_arm "G${i}"; exit 1; }
done

if [ -f "$OUT/E1_gate.json" ] && [ -f "$OUT/G1_gate.json" ]; then
  echo
  echo "########## CONTENT GATE (pair 1) ##########"
  "$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" \
      compare "$OUT/E1_gate.json" "$OUT/G1_gate.json" || true
fi

echo
echo "########## REPORT HEADER ##########"
echo "Enforced GPU power caps at report time (measured, not assumed):"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit \
           --format=csv,noheader 2>/dev/null | sed 's/^/  /'
cat <<'HDR'
  Caps on this rig are 200 W / 400 W / 200 W (3080 / 5090 / 3080), reduced from
  320 / 525 / 320. All arms share them, so A/B deltas and floors are valid;
  comparisons against pre-change archive numbers are CONFOUNDED.
  Arm order is interleaved E,G,E,G,E,G. Interleaving is here to eliminate slow
  drift between arms -- that is its whole job. It is NOT a clock argument:
  clock and power are diagnostic annotation only (see report notes).
HDR
echo "  TRANSPORT for every arm in this run: $TRANSPORT (SGLANG_BARLINK=${SGLANG_BARLINK:-unset})"
echo "  RESERVE for every arm in this run:   $RESERVE"
echo "  EXTRA flags on every arm:            ${EXTRA:-none}"
echo "  Production as of 2026-08-05 runs TRANSPORT=barlink; an nccl run answers"
echo "  the prefill-graph question in isolation and is comparable with window 3,"
echo "  but is NOT by itself a production rollout argument."

"$VENV/bin/python" "$WT/tests/prefill_graphs/report_interleaved.py" "$OUT"
