#!/usr/bin/env bash
# Window 10 -- one measurement ARM.
#
# Every arm runs the SAME two instruments, because the 126.8 the campaign is
# asked to account for and the 55-64 it is compared against were NOT taken
# with the same one:
#   * s14_decode_punkt.py bs=1  -> the record's "decode tok/s bs=1" tick number
#   * club-3090 bench.sh        -> the record's narrative/code decode_TPS
# A-vs-A floor first (3 identical s14 draws), then the measured draw, then
# bench.sh, then a 20-sample py-spy leaf census.
#
# Usage: run_arm.sh <arm> <worktree> <flagset: record|today> <envfile>
set -uo pipefail

ARM="${1:?arm name}"
WT="${2:?worktree}"
FLAGSET="${3:?record|today}"
ENVFILE="${4:?env file to source}"

OUT=/spinning/gpu-battery-results/2026-08-05_window10
VENV=/spinning/htsglang-gpu/.venv
MODEL=/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8
C3090=/spinning/llm_stuff/club-3090
PORT="${PORT:-30605}"
SLOG="$OUT/raw/server_$ARM.log"
POINT_S="${POINT_S:-12}"

mkdir -p "$OUT/raw"
echo "=== $(date -u +%H:%M:%SZ) ARM=$ARM wt=$WT flagset=$FLAGSET env=$ENVFILE ==="

# ---- environment ------------------------------------------------------------
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
# Arm-specific env (transport, gate knobs) comes from the env file, which may
# also UNSET things -- sourcing, not exporting, so an arm can say "no barlink".
# shellcheck disable=SC1090
source "$ENVFILE"

# ---- flags ------------------------------------------------------------------
# RECORD = the #424 int8_decode arm's flags verbatim (RUNSHEET "Flags"):
#   ctx 131072, mrr 16, reserve auto, phase-decode, NEXTN 3/1/4, no HiCache,
#   no ladder, no fast-lane, no mamba pin, no preserve_thinking.
# TODAY  = /root/bin/start-serving-30030.sh verbatim, minus the port.
if [ "$FLAGSET" = "record" ]; then
  FLAGS=(
    --tp-size 3 --rank-gpu-id 0,1,2
    --rank-tp-ratio auto-performance --rank-perf-tune phase-decode
    --rank-auto-reserve-mib auto
    --kv-cache-dtype fp8_e4m3 --context-length 131072
    --max-running-requests 16
    --tool-call-parser qwen3_coder --reasoning-parser qwen3
    --speculative-algorithm NEXTN --speculative-num-steps 3
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
    --enable-metrics --trust-remote-code
  )
else
  FLAGS=(
    --tp-size 3 --rank-gpu-id 0,1,2
    --rank-tp-ratio auto-performance --rank-perf-tune phase-decode
    --rank-auto-reserve-mib auto
    --kv-cache-dtype fp8_e4m3 --context-length 262144
    --max-running-requests 4
    --speculative-algorithm NEXTN --speculative-num-steps 3
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder
    --kv-pressure-ladder auto
    --max-mamba-cache-size 96
    --enable-fast-lane --retraction-policy priority
    --enable-hierarchical-cache
    --hicache-ratio 2 --hicache-write-policy write_through
    --hicache-mem-layout page_first_direct --hicache-io-backend direct
    --chat-template-default-kwargs '{"preserve_thinking": true}'
    --enable-cache-report
    --sleep-on-idle
    --enable-metrics --trust-remote-code
  )
  export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/spinning/hicache
fi

# ---- boot -------------------------------------------------------------------
rm -f "$SLOG"
cd "$WT" || exit 2
setsid "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name Qwen3.6-27B \
  "${FLAGS[@]}" --host 127.0.0.1 --port "$PORT" > "$SLOG" 2>&1 &
PGID=$!
echo "$PGID" > "$OUT/raw/pgid_$ARM"
echo "boot pgid=$PGID log=$SLOG"

UP=0
for _ in $(seq 1 90); do
  if curl -s -m 5 -o /dev/null "http://127.0.0.1:$PORT/health_generate"; then UP=1; break; fi
  kill -0 "$PGID" 2>/dev/null || { echo "PROCESS GONE"; break; }
  sleep 10
done
if [ "$UP" != 1 ]; then
  echo "BOOT FAILED arm=$ARM -- tail:"; tail -30 "$SLOG"
  kill -TERM -- -"$PGID" 2>/dev/null; sleep 8; kill -KILL -- -"$PGID" 2>/dev/null
  exit 2
fi
echo "arm=$ARM UP $(date -u +%H:%M:%SZ)"

# ---- transport + plan gate --------------------------------------------------
grep -aE "barlink (enabled for group|group)|ACHIEVED=|PyNccl|transport" "$SLOG" \
  | head -30 > "$OUT/gate_$ARM.txt" 2>&1
grep -aE "derived memory budgets|MLP vector|CHOSEN|token sizing|max_total_num_tokens=" "$SLOG" \
  | head -40 > "$OUT/plan_$ARM.txt" 2>&1
grep -aE "max_total_num_tokens=" "$SLOG" | head -1

# ---- A-vs-A floor: three identical s14 bs=1 draws ---------------------------
for I in 1 2 3; do
  "$VENV/bin/python" scripts/gpu_battery/s14_decode_punkt.py --port "$PORT" \
    --out-dir "$OUT/raw" --arm "${ARM}_floorD$I" --bs 1 --folge 1 \
    --context-tokens 2048 --model-context-tokens 131072 --ramp-seconds 6 \
    --window-seconds "$POINT_S" --server-log "$SLOG" >> "$OUT/floor_$ARM.log" 2>&1
  echo "  floor draw $I rc=$?"
done
grep -h "bs=1" "$OUT/floor_$ARM.log" | tail -3

# ---- measured s14 bs=1 ------------------------------------------------------
"$VENV/bin/python" scripts/gpu_battery/s14_decode_punkt.py --port "$PORT" \
  --out-dir "$OUT/raw" --arm "$ARM" --bs 1 --folge 1 \
  --context-tokens 2048 --model-context-tokens 131072 --ramp-seconds 6 \
  --window-seconds "$POINT_S" --server-log "$SLOG" >> "$OUT/messen_$ARM.log" 2>&1
echo "  measured s14 bs=1 rc=$?"
tail -1 "$OUT/messen_$ARM.log"

# ---- py-spy leaf census on the TP0 worker (during a live decode load) -------
# Load first, then sample: a census of an idle server measures the idle loop.
( "$VENV/bin/python" scripts/gpu_battery/s14_decode_punkt.py --port "$PORT" \
    --out-dir "$OUT/raw" --arm "${ARM}_pyspyload" --bs 1 --folge 1 \
    --context-tokens 2048 --model-context-tokens 131072 --ramp-seconds 4 \
    --window-seconds 30 --server-log "$SLOG" > /dev/null 2>&1 ) &
LOADPID=$!
sleep 12
TP0=$(pgrep -f "sglang::scheduler_TP0" | head -1)
[ -z "$TP0" ] && TP0=$(pgrep -f "sglang" | head -1)
"$VENV/bin/python" "$OUT/scripts/pyspy_leaves.py" --pid "$TP0" --samples 20 \
  --label "$ARM" > "$OUT/pyspy_$ARM.txt" 2>&1
head -12 "$OUT/pyspy_$ARM.txt"
wait $LOADPID 2>/dev/null

# ---- club-3090 bench.sh -----------------------------------------------------
( cd "$C3090" && URL="http://127.0.0.1:$PORT" MODEL="Qwen3.6-27B" CONTAINER=none PP=1 \
    timeout -k 30 1500 bash scripts/bench.sh ) > "$OUT/bench_$ARM.txt" 2>&1
echo "bench.sh rc=$? arm=$ARM"
grep -A4 "=== summary" "$OUT/bench_$ARM.txt" | grep -E "summary|decode_TPS|wall_TPS|PP tok"

# ---- corridor + teardown ----------------------------------------------------
nvidia-smi --query-gpu=index,name,memory.used,memory.free,power.draw,utilization.gpu \
  --format=csv,noheader > "$OUT/vram_final_$ARM.txt" 2>&1
kill -TERM -- -"$PGID" 2>/dev/null; sleep 10
kill -KILL -- -"$PGID" 2>/dev/null; sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/vram_after_$ARM.txt"
cat "$OUT/vram_after_$ARM.txt"
echo "arm=$ARM DONE $(date -u +%H:%M:%SZ)"
