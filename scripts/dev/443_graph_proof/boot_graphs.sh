#!/usr/bin/env bash
# #443 -- the V4-Flash decode-graph proof window.
#
# Recipe is scripts/dev/394_s2_proof/boot_ab.sh (itself boot394.sh from the
# 2026-08-02 battery) with ONE treatment: CUDA graphs on for decode, served by
# the capturable expert-offload path instead of the eager one. Two arms, one
# boot each, SAME residency so the comparison is not confounded by which
# experts happen to be hot:
#
#   ARM=eager   --disable-cuda-graph, the published BENCH_394 configuration.
#               7.15 decode TPS on the code prompt, 7.10 on the narrative one.
#               This arm exists to re-measure that floor on the SAME boot day,
#               because a delta against a number from another day is a delta
#               against that day's clocks.
#   ARM=graphs  graphs on, SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1. The port under
#               test.
#
# WHAT THIS WINDOW PROVES, AND WHAT IT CANNOT
#
# Desk-proven already (tests/moe_offload/test_capture_desync_port.py, 63 tests,
# four executed can-fail arms): the ported step performs no host read, and it
# moves the same expert rows into the same scratch slots as the eager fetch.
# Nothing there involves a card.
#
# BOOT-PENDING, i.e. exactly what this window is for:
#
#   B1  CAPTURE SUCCEEDS. The eager arm's boot log must contain the
#       --disable-cuda-graph requirement being satisfied; the graph arm's must
#       show DecodeCudaGraphRunner capturing the bucket list without
#       cudaErrorStreamCaptureUnsupported / ...Invalidated. A capture that
#       refuses is the primary negative result and is worth reporting as such.
#   B2  REPLAY IS CORRECT. Greedy decode from a fixed prompt/seed, graphs arm
#       vs eager arm, SAME boot day and SAME residency. The port is a data
#       movement change, so the expected result is identical text; a divergence
#       means the gather sourced a row the eager path would not have.
#   B3  THE UVA VIEW HOLDS UNDER CAPTURE. device_view_of_pinned() asserts the
#       alias at build time (it raises if torch.as_tensor copied). What it
#       cannot assert is that a CAPTURED index_select over that address still
#       reads host memory after replay. B2 covers it observationally; a direct
#       check is to run the graphs arm with one expert's pinned row poisoned
#       between capture and replay and confirm the output changes.
#   B4  PERF. ms/round for decode, per rank, both arms. The 2.6x gap vs the
#       club-3090 reference has this sync as its ranked-#2 cause; #2 is not #1,
#       so a partial recovery is the honest expectation and the number to
#       report is the measured one, not the gap.
#
# The shared cold tier (#394 slice 2) is DELIBERATELY OFF here. Combining it
# with capture is refused by name (refuse_capturable_cold_tier): the captured
# gather has no peer source, and the peer UVA pointer is itself unverified.
# Opening that seam in the same window would make a failure un-attributable.
# SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE=1 is the switch for a LATER window, and
# past it the replay-boundary gate (offload_capture_gate) names any delegated
# expert that gets routed instead of letting it read row 0.
#
# Corridor discipline: per-card free VRAM >= 400 MiB, no registered posting
# wasting > 1.5 GiB net. Graph capture itself costs VRAM (one pool per bucket),
# which is why the bucket list is capped rather than left at the default.

set -uo pipefail

source /root/rig-env.sh 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT or provide /root/rig-env.sh}"
VENV="${VENV:?set VENV}"
MODEL_ROOT="${MODEL_ROOT:?set MODEL_ROOT}"
WT="${WT:-$(dirname "$REPO_ROOT")/wt-desync-port}"
ARM="${ARM:-eager}"
PORT="${PORT:-30443}"
RUN="${RUN:-/spinning/gpu-battery-results/$(date +%F)_443_graphs}"
GGUF_DIR="${GGUF_DIR:-$MODEL_ROOT/DeepSeek-V4-Flash-0731-GGUF/UD-IQ3_XXS}"
# Derive from scripts/dev/394_s2_proof/preflight.sh's NVML table. NEVER a fixed
# assumption: NVML order and CUDA order differ on this rig.
RANK_GPU_ID="${RANK_GPU_ID:-0,1,2}"
# Captured decode buckets. bs * top_k is what the scratch region must serve, so
# this and SGLANG_MOE_SCRATCH_SLOTS move together -- run scratch_preflight.py.
# Defaults are the pair the 2026-08-02 battery already ran (6 slots) at the
# recipe's own operating point (--max-running-requests 1 -> bs 1), so the graph
# arm costs no extra resident VRAM. Raising MAX_GRAPH_BS costs top_k more slots
# per unit and is a corridor decision -- see README.md, and run
# scratch_preflight.py before changing either.
MAX_GRAPH_BS="${MAX_GRAPH_BS:-1}"
SCRATCH_SLOTS="${SCRATCH_SLOTS:-6}"

mkdir -p "$RUN"

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "FAIL port $PORT is already in use" >&2
  exit 1
fi

export PYTHONPATH="$WT/python"
export SGLANG_MOE_SCRATCH_SLOTS="$SCRATCH_SLOTS"
export SGLANG_FORWARD_PEAK_PATH="$RUN/peak_$ARM"
export SGLANG_GGUF_STREAM_TRIM_SOFT_GIB=88
export SGLANG_GGUF_STREAM_TRIM_TARGET_GIB=78
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_EXPERT_STATS=1
export SGLANG_EXPERT_STATS_PATH="$RUN/expert_stats_$ARM"
# Interval dumps, never SIGUSR2 -- see INCIDENT_394_sigusr2.md.
export SGLANG_EXPERT_STATS_INTERVAL_SEC=45
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_TOPK_V2=0
# Equal byte ownership, no shared cold tier: the treatment is capture alone.
export SGLANG_MOE_HOST_SHARD_RATIO=1,1,1
unset SGLANG_MOE_COLD_TIER_SHM || true
unset SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE || true
# Residency must be FROZEN before capture and IDENTICAL across the two arms.
# Live calibration is refused under graph mode on purpose; static [0,R) is the
# default and is what both arms get unless a hot-set file is supplied.
unset SGLANG_MOE_HOT_RESIDENCY || true

ARM_ARGS=()

case "$ARM" in
  eager)
    unset SGLANG_MOE_OFFLOAD_CUDA_GRAPH || true
    ARM_ARGS+=(--disable-cuda-graph)
    ;;
  graphs)
    export SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1
    export SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS="$MAX_GRAPH_BS"
    ARM_ARGS+=(--cuda-graph-max-bs "$MAX_GRAPH_BS")
    ;;
  *)
    echo "FAIL unknown ARM=$ARM (eager|graphs)" >&2
    exit 1
    ;;
esac

setsid "$VENV/bin/python" -u -m sglang.launch_server \
  --model-path "$GGUF_DIR/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf" \
  --tp-size 3 --rank-gpu-id "$RANK_GPU_ID" --rank-tp-ratio auto \
  --rank-auto-reserve-mib 2200,1400,1400 \
  --rank-moe-resident-fraction 0.485,0.42,0.42 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 8192 --max-running-requests 1 \
  --chunked-prefill-size 512 \
  --trust-remote-code --enable-metrics \
  "${ARM_ARGS[@]+"${ARM_ARGS[@]}"}" \
  --host 127.0.0.1 --port "$PORT" \
  > "$RUN/boot_$ARM.log" 2>&1 &
echo $! > "$RUN/boot_$ARM.launchpid"
echo "launched ARM=$ARM pid=$(cat "$RUN/boot_$ARM.launchpid") port=$PORT run=$RUN"
echo "next: bounded curl -m readiness loop, then read the B1..B4 readouts in README.md"
