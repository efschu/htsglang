#!/usr/bin/env bash
# #631 A-vs-A REGRESSION GATE boot: the NON-FLIP DEFAULT PATH, on either tree.
#
# WHY THIS SCRIPT EXISTS AT ALL. The gate asks one question: does the
# phase-flip build regress the path a user gets when they do NOT ask for the
# flip? So BOTH boots must run WITHOUT --enable-phase-flip, and
# route_a_631_prod_boot.sh always passes it. This is that script with the
# flip surface removed and nothing else changed, parameterised by WT so the
# same bytes boot the flip tree and the baseline tree.
#
# EVERY knob below is deliberately IDENTICAL across the two boots. The only
# permitted difference between the two invocations is WT (which also sets
# PYTHONPATH and the boot provenance).
#
# THE PER-RANK INSTRUMENT. SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 is set here
# and is load-bearing for the gate, not decoration: under pp_size > 1 the
# "Prefill rank batch ... gpu-ms (compute, wait)" split is NOT installed
# (metrics_reporter._install_rank_prefill_timer returns early for
# pp_size != 1, identically in both trees), so the ms-per-round compute/wait
# canon cannot be served by that line here. What IS available per PP rank is
# the "Decode batch ... fwd occupancy: X%" field, which is the device
# timer's GPU-busy time over the wall window -- i.e. the compute fraction of
# each rank's round, with the remainder being wait plus host. All three PP
# ranks are stats-logging ranks (tp_size 1 => attn_tp_rank 0 on each), so
# the field lands once per rank.
set -euo pipefail

WT="${WT:?WT must name the tree to boot}"
PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"
MODEL="${MODEL:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8}"
# NOT 30030. The local router (30099) forwards to 30030, and a benchmark
# instance must never silently receive agent traffic mid-measurement.
PORT="${PORT:-30031}"
HOST="${HOST:-127.0.0.1}"
RANK_MIB="${RANK_MIB:-22700,11920,11970}"
CTX="${CTX:-262144}"
MAX_RUNNING="${MAX_RUNNING:-4}"
MAMBA_SLOTS="${MAMBA_SLOTS:-20}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-500000}"
# OFF, and it is not a choice. check_server_args REFUSES speculation under
# pipeline parallelism unless --enable-phase-flip is set ("Pipeline
# parallelism is not compatible with speculative decoding"), because the
# draft worker only exists on the flip's TP stack. The non-flip default path
# at pp_size 3 therefore CANNOT speculate, in either tree, so the gate
# measures the shape that path actually has.
SPEC="${SPEC:-off}"
BARLINK="${BARLINK:-1}"
LOG="${SERVING_LOG:?SERVING_LOG must be set explicitly for a gate boot}"

if [ "$SPEC" = "off" ]; then
  SPEC_FLAGS=""
else
  SPEC_FLAGS="--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
fi

if pgrep -f "sglang.launch_server.*--port $PORT" >/dev/null 2>&1; then
  echo "REFUSE: a serving instance for port $PORT is already running." >&2
  exit 1
fi

export PYTHONPATH="$WT/python"
export LD_LIBRARY_PATH="/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
# The gate's per-rank compute/wait instrument (see header).
export SGLANG_ENABLE_METRICS_DEVICE_TIMER=1
export SGLANG_COLLECTIVE_CENSUS_INTERVAL="${SGLANG_COLLECTIVE_CENSUS_INTERVAL:-50}"

if [ "$BARLINK" = "1" ]; then
  if [ ! -e /dev/dmabuf_holder ]; then
    echo "REFUSE: bar1 transport requested but /dev/dmabuf_holder is missing." >&2
    exit 1
  fi
  export SGLANG_BARLINK=1
  export SGLANG_BARLINK_TRANSPORT=bar1
  export SGLANG_BARLINK_BAR1_CAP_CYCLES=300000000000
  # NON-FLIP topology: only world:0 and pp:0 carry a window here, so the
  # flip-group budget of the production script does not apply. pp:0 keeps
  # the 96 MiB it needs for chunked-prefill activations.
  export SGLANG_BARLINK_BAR1_WINDOW_MIB="${SGLANG_BARLINK_BAR1_WINDOW_MIB:-32}"
  export SGLANG_BARLINK_BAR1_WINDOW_MIB_PP_0="${SGLANG_BARLINK_BAR1_WINDOW_MIB_PP_0:-96}"
else
  export SGLANG_BARLINK=0
  unset SGLANG_BARLINK_TRANSPORT || true
fi

# --- resolve cards by NAME -> UUID, 5090 FIRST (same rule as production) ----
BIG_UUID=""; SMALL_UUID=()
while IFS=, read -r _idx name uuid _tot; do
    name="$(echo "$name" | xargs)"; uuid="$(echo "$uuid" | xargs)"
    case "$name" in *5090*) BIG_UUID="$uuid" ;; *3080*) SMALL_UUID+=("$uuid") ;; esac
done < <(nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader)
[[ -n "$BIG_UUID" && ${#SMALL_UUID[@]} -ge 2 ]] || {
    echo "FATAL: expected one 5090 and two 3080s from NVML" >&2; exit 1; }

: > "$LOG"
BOOT_COMMIT="$(git -C "$WT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
{
  printf '=== A-vs-A GATE BOOT %s ===\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'tree=%s commit=%s barlink=%s rank_mib=%s ctx=%s spec=%s port=%s\n' \
         "$WT" "$BOOT_COMMIT" "$BARLINK" "$RANK_MIB" "$CTX" "$SPEC" "$PORT"
} >> "$LOG"

cd "$WT"
CUDA_VISIBLE_DEVICES="$BIG_UUID,${SMALL_UUID[0]},${SMALL_UUID[1]}" \
setsid "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --trust-remote-code \
    --served-model-name Qwen3.6-27B \
    --tp-size 1 --pp-size 3 \
    --pp-stage-ratio 2,1,1 \
    --rank-gpu-id 0,1,2 \
    --rank-gpu-memory-mib "$RANK_MIB" \
    --disable-overlap-schedule \
    --kv-cache-dtype fp8_e4m3 --context-length "$CTX" \
    --max-running-requests "$MAX_RUNNING" \
    --max-mamba-cache-size "$MAMBA_SLOTS" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    $SPEC_FLAGS \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --chat-template-default-kwargs '{"preserve_thinking": true}' \
    --enable-cache-report \
    --enable-metrics --host "$HOST" --port "$PORT" \
    "$@" >> "$LOG" 2>&1 &
echo "gate boot pgid $!  tree $WT  commit $BOOT_COMMIT  log $LOG"
