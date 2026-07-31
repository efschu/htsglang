#!/usr/bin/env bash
# One V4 arm: boot -> readiness -> coherence -> prefill point -> decode
# point(s) -> teardown. Same shape and the same two measuring scripts as the
# 2026-07-31 NVFP4 beleg, so every number here lands in the same JSONL schema
# as the anchors it is read against.
#
#   CVD=<uuid[,uuid...]> arm.sh <tag> <model> <decode-bs-list> -- <launch args>
set -uo pipefail
source /spinning/gpu-battery-results/2026-07-31_332_fam_beleg/env.sh

TAG="$1"; MODEL="$2"; BS_LIST="$3"; shift 3
[[ "${1:-}" == "--" ]] && shift

LOG="$OUT/logs/${TAG}.server.log"
PIDF="$OUT/logs/${TAG}.pid"
PROG="$OUT/logs/${TAG}.progress"
VRAM="$OUT/proofs/${TAG}.vram_sample.csv"

say() { echo "$(date -u +%H:%M:%S) $*" >> "$PROG"; }
: > "$PROG"

# CVD pins the arm to a subset of cards, by UUID (never by index:
# CUDA_VISIBLE_DEVICES is read in CUDA enumeration order, which is not NVML
# order on this rig). An arm that uses every card leaves it unset and selects
# with --rank-gpu-id instead, which is the flag that owns placement here --
# setting both would map the cards twice.
if [ -n "${CVD:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$CVD"
fi
say "BOOT ${TAG} model=$(basename "$MODEL") cvd=${CVD:-<all cards, --rank-gpu-id decides>}"

# Sampled during load as well as at steady state: the peak is what decides
# whether the corridor rule (>= 400 MiB free) holds, and warmup hides it.
echo "ts_utc,gpu_index,name,mem_used_mib,mem_total_mib,util_pct,power_w" > "$VRAM"
(
  while true; do
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits | sed "s/^/$(date -u +%Y-%m-%dT%H:%M:%SZ),/" >> "$VRAM"
    sleep 5
  done
) &
SAMPLER=$!

cd "$WT"
setsid "$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name Qwen3.6-27B-NVFP4 \
  --trust-remote-code --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  --decode-log-interval 1 \
  "$@" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDF"
say "pid=$PID"

# Bounded readiness poll: fixed attempt count, never an open-ended wait.
READY=0
for _ in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 \
         "http://127.0.0.1:${PORT}/health_generate" 2>/dev/null)
  if [[ "$code" == "200" ]]; then READY=1; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then say "DIED before ready"; break; fi
  sleep 5
done

if [[ "$READY" == "1" ]]; then
  say "READY"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
    > "$OUT/proofs/${TAG}.vram_resident.csv"
  curl -s -m 30 "http://127.0.0.1:${PORT}/get_server_info" \
    > "$OUT/proofs/${TAG}.server_info.json" 2>/dev/null

  say "COHERENCE"
  "$PY" "$OUT/coherence.py" --port "$PORT" --tag "$TAG" \
    --out "$OUT/coherence_${TAG}.jsonl" >> "$PROG" 2>&1

  say "PREFILL"
  "$PY" "$WT/scripts/gpu_battery/s12_prefill_kurve.py" \
    --mode messen --port "$PORT" --out-dir "$OUT" \
    --point-seconds 12 --warmup-seconds 6 \
    --prompt-tokens 2048 --with-decode 0 \
    --arm "$TAG" --sessions 1 --folge 1 --server-log "$LOG" >> "$PROG" 2>&1

  for BS in ${BS_LIST//,/ }; do
    say "DECODE bs=${BS}"
    "$PY" "$WT/scripts/gpu_battery/s14_decode_punkt.py" \
      --port "$PORT" --out-dir "$OUT" --arm "$TAG" --bs "$BS" \
      --context-tokens 2048 --model-context-tokens 32768 \
      --ramp-seconds 6 --window-seconds 12 --drain-seconds 4 \
      --folge 1 --server-log "$LOG" >> "$PROG" 2>&1
  done
  curl -s -m 30 "http://127.0.0.1:${PORT}/metrics" \
    > "$OUT/proofs/${TAG}.metrics.txt" 2>/dev/null
fi

# --- the #332 posten-1 and posten-2 readouts, taken while the log is fresh ---
{
  echo "DEQUANTISED_LINES=$(grep -c 'DEQUANTISED' "$LOG" 2>/dev/null || echo 0)"
  echo "DEQUANT_IN_PROJ_BA=$(grep -c 'DEQUANTISED.*in_proj_ba' "$LOG" 2>/dev/null || echo 0)"
  echo "UNLOADED_DRAFT=$(grep -ci 'unloaded' "$LOG" 2>/dev/null || echo 0)"
  echo "--- backend / fp4 lane lines ---"
  grep -iE 'nvfp4|fp4|marlin|backend' "$LOG" 2>/dev/null | sort | uniq -c | sort -rn | head -40
  echo "--- sizing lines ---"
  grep -iE 'max_total_num_tokens|reserve|weight|avail mem|KV Cache is allocated' "$LOG" 2>/dev/null | tail -30
  echo "--- unloaded / draft lines ---"
  grep -iE 'unloaded|draft|speculat' "$LOG" 2>/dev/null | tail -30
} > "$OUT/proofs/${TAG}.readout.txt" 2>&1

say "TEARDOWN"
kill "$SAMPLER" 2>/dev/null
"$VENV/bin/py-spy" dump --pid "$PID" > "$OUT/logs/${TAG}.pyspy.txt" 2>&1
# Only our own process group -- other sessions share this box.
kill -TERM -- "-${PID}" 2>/dev/null
for _ in $(seq 1 40); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
kill -KILL -- "-${PID}" 2>/dev/null
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
  > "$OUT/proofs/${TAG}.after_vram.csv"
say "DONE ready=${READY}"
[[ "$READY" == "1" ]] || exit 1
exit 0
