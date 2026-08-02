#!/usr/bin/env bash
# #394 slice 2 proof window -- the A/B arm.
#
# Recipe is boot394.sh from the 2026-08-02 battery verbatim, except for the
# worktree, the run dir, and ARM. Two arms, one boot each:
#
#   ARM=equal         SGLANG_MOE_HOST_SHARD_RATIO=1,1,1 -> is_equal -> no
#                     ColdShardContext at all, i.e. the pre-#394 plan field for
#                     field. Named explicitly rather than left unset, so the
#                     baseline identifies itself in the #390 dump.
#   ARM=proportional  the launcher-published rank->card vector reaches
#                     _gguf_cold_shard_context, the ratio resolves from the
#                     MEASURED card-probe H2D, and SGLANG_MOE_COLD_TIER_SHM=1
#                     makes the delegated experts REACHABLE. Arm A of the
#                     2026-08-02 battery could not run this: the delegated
#                     experts were absent and the boot was refused.
#
# WHAT THIS ARM DOES AND DOES NOT TEST -- read before quoting a delta.
#
# Slice 2 moves cold-expert BYTE OWNERSHIP, not compute. A rank that must run
# expert e still pulls e across its OWN link, whether the bytes came from its
# private pinned pool or from a peer's shared segment. So the PREDICTION for
# this arm is:
#
#   * per-rank H2D volume: UNCHANGED (within noise) vs the equal arm;
#   * decode ms/round:     UNCHANGED (within noise);
#   * the x4 rank stays the clock;
#   * what changes is that the arm RUNS AT ALL, plus host-DRAM placement.
#
# The ~27.5 -> ~20.2 s decode marker from ANALYSE_393 §7.3 belongs to Path A',
# which additionally moves WHICH RANK COMPUTES WHICH EXPERT (the #82 expert
# range). This arm does not do that, and reporting the marker as its target
# would be reporting a number the mechanism cannot produce. The arm is
# instrumented to SHOW that: the primary readout is per-rank H2D, and a null
# delta is the expected, publishable result. If per-rank H2D DOES move, the
# analysis above is wrong and that is the finding worth having.
#
# Corridor discipline: per-card free VRAM must stay >= 400 MiB and no single
# registered posting may waste > 1.5 GiB net. Both arms run at the reserve
# validated for this recipe; do not tighten it to buy the tier headroom -- the
# tier lives in host DRAM, not VRAM.

set -uo pipefail

source /root/rig-env.sh 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT or provide /root/rig-env.sh}"
VENV="${VENV:?set VENV}"
MODEL_ROOT="${MODEL_ROOT:?set MODEL_ROOT}"
WT="${WT:-$(dirname "$REPO_ROOT")/wt-394-s2}"
ARM="${ARM:-equal}"
PORT="${PORT:-30394}"
RUN="${RUN:-/spinning/gpu-battery-results/$(date +%F)_394_s2}"
GGUF_DIR="${GGUF_DIR:-$MODEL_ROOT/DeepSeek-V4-Flash-0731-GGUF/UD-IQ3_XXS}"
# Derive from preflight.sh's NVML table. NEVER a fixed assumption: NVML order
# and CUDA order differ on this rig and can shift between boots.
RANK_GPU_ID="${RANK_GPU_ID:-0,1,2}"

mkdir -p "$RUN"

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "FAIL port $PORT is already in use" >&2
  exit 1
fi

export PYTHONPATH="$WT/python"
export SGLANG_MOE_SCRATCH_SLOTS=6
export SGLANG_FORWARD_PEAK_PATH="$RUN/peak_$ARM"
export SGLANG_GGUF_STREAM_TRIM_SOFT_GIB=88
export SGLANG_GGUF_STREAM_TRIM_TARGET_GIB=78
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_EXPERT_STATS=1
export SGLANG_EXPERT_STATS_PATH="$RUN/expert_stats_$ARM"
# Interval dumps, never SIGUSR2. SIGUSR2 has no handler in the FRONTEND
# process (no MoE layers there, so no collector, so no handler installed) and
# its default action is terminate -- that is how the first arm-A boot of the
# 2026-08-02 battery died. See INCIDENT_394_sigusr2.md.
export SGLANG_EXPERT_STATS_INTERVAL_SEC=45
export SGLANG_MOE_STAGING_TRACE=1
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_TOPK_V2=0

case "$ARM" in
  equal)
    export SGLANG_MOE_HOST_SHARD_RATIO=1,1,1
    unset SGLANG_MOE_COLD_TIER_SHM || true
    ;;
  proportional)
    unset SGLANG_MOE_HOST_SHARD_RATIO || true
    # Refuse a nameplate-weighted arm: an A/B whose treatment is a datasheet
    # derivation is not the experiment anybody intended to run.
    export SGLANG_MOE_HOST_SHARD_MIN_PROVENANCE=measured
    export SGLANG_MOE_COLD_TIER_SHM=1
    ;;
  *)
    echo "FAIL unknown ARM=$ARM (equal|proportional)" >&2
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
  --disable-cuda-graph \
  --trust-remote-code --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  > "$RUN/boot_$ARM.log" 2>&1 &
echo $! > "$RUN/boot_$ARM.launchpid"
echo "launched ARM=$ARM pid=$(cat "$RUN/boot_$ARM.launchpid") port=$PORT run=$RUN"
echo "next: wait for readiness with a BOUNDED curl -m loop, then read_arm.py"
