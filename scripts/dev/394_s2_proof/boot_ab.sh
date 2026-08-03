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
#   ARM=compute       #394 slice 3: --rank-moe-ratio link moves the COMPUTE
#                     assignment, i.e. the #82 expert range itself, so a rank's
#                     streamed mass is proportional to its own link. This is
#                     the arm whose per-rank H2D delta is predicted NON-null.
#   ARM=compute-cal   FALSIFIED (2026-08-03), kept only to test a better
#                     hit-rate model. --rank-moe-ratio link-calibrated with the
#                     per-rank cold-traffic coefficients measured on the equal
#                     arm supplied through SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS.
#                     It is falsified on TWO legs, and the transfer term is not
#                     one of them: (1) END-TO-END, -0.94 % against -7.67 % for
#                     plain 'compute', i.e. inside the same-window floor; and
#                     (2) MECHANISM, a per-rank hit rate tracks the SIZE of the
#                     owned expert range, not the rank, so the coefficient is
#                     not the rank property the solve assumes. The old
#                     "1.439x against 1.496x" transfer-term argument is
#                     WITHDRAWN -- it divided two dumps sampled at different
#                     work points, and work-matched the calibrated arm reads
#                     1.4573x against 1.4253x. Do not run it as a serving
#                     recommendation; see ARM3_COMPUTE.md.
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
# range). The equal/proportional arms do not do that, and reporting the marker
# as their target would be reporting a number those mechanisms cannot produce.
# They are instrumented to SHOW that: the primary readout is per-rank H2D, and
# a null delta is the expected, publishable result. If per-rank H2D DOES move,
# the analysis above is wrong and that is the finding worth having.
#
# ARM=compute / compute-cal ARE that marker's arm, and the sign of their
# prediction is the opposite: a null per-rank H2D delta FALSIFIES them. Their
# predicted per-rank figures are tabulated in ARM3_COMPUTE.md; read it before
# quoting anything.
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
# The arm comes from the environment, or from $1 for a caller that passes it
# positionally. It is NOT defaulted to "equal" any more (#439): a driver that
# set ARM without exporting it ran four arms of a battery and booted the
# baseline every time, and the only symptom was a delta of zero between two
# baselines. An unnamed arm is a refusal, and every arm names itself in the
# facts file below.
ARM="${ARM:-${1:-}}"
if [ -z "$ARM" ]; then
  echo "FAIL ARM is not set. Pass it in the environment (ARM=compute bash" >&2
  echo "     boot_ab.sh) or as \$1. It is deliberately not defaulted: a" >&2
  echo "     silent fall back to the baseline is how an A/B reports a null" >&2
  echo "     that was never tested." >&2
  exit 2
fi
PORT="${PORT:-30394}"
RUN="${RUN:-/spinning/gpu-battery-results/$(date +%F)_394_s2}"
GGUF_DIR="${GGUF_DIR:-$MODEL_ROOT/DeepSeek-V4-Flash-0731-GGUF/UD-IQ3_XXS}"
# CORRIDOR-REPAIRED DEFAULT (#458). Arms 1 and 2 were measured at
# 2200,1400,1400, and the 2026-08-03 window proved that recipe breaches the
# corridor: sampled DURING load, both 3080s bottom out at 211-251 MiB free
# against the 400 MiB floor (the ~515 MiB previously quoted was a post-boot
# snapshot, which overstates free by 250-330 MiB). +400 MiB on each 3080 lifts
# the measured minimum to ~611-651 MiB; the 5090 was never near the floor
# (1262 MiB minimum) and keeps its 2200. This changes the derived budgets and
# therefore the base plan -- see ARM3_COMPUTE.md, "Green-corridor window".
# 'auto' is NOT usable on this recipe: it derives 3968 MiB uniformly from the
# stock activation heuristic and the resulting 16512 MiB budget is below what
# weights + runtime state already need on a 3080 (measured 17.59 GiB).
# Whatever is chosen, every arm in one window must use the SAME value -- the
# reserve moves the KV pool, and a reserve that differs between arms is a
# second treatment.
RESERVE_MIB="${RESERVE_MIB:-2200,1800,1800}"
# The solve holds the resident mass fixed against the BASE plan, so this value
# is part of the treatment definition: changing it between arms changes what is
# being compared.
RESIDENT_FRACTION="${RESIDENT_FRACTION:-0.485,0.42,0.42}"
# Derive from preflight.sh's NVML table. NEVER a fixed assumption: NVML order
# and CUDA order differ on this rig and can shift between boots.
RANK_GPU_ID="${RANK_GPU_ID:-0,1,2}"

mkdir -p "$RUN"

# DRY_RUN=1 resolves the arm and PRINTS the launch command instead of running
# it. That makes "does this arm's treatment actually reach the server?" a
# question a test can ask without a GPU -- see test_expert_compute_placement_439
# (TestTheArmHarnessCarriesTheArm), whose can-fail arm drops the export.
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" != "1" ] && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
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

# Extra launch arguments the arm adds. Empty for the two slice-2 arms, so their
# command line is byte-identical to the one the 2026-08-02 battery ran.
ARM_ARGS=()

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
  compute|compute-cal)
    # Slice 3. Byte ownership stays at the baseline on purpose -- this arm's
    # treatment is the COMPUTE assignment, and moving both at once would
    # produce a delta neither mechanism could claim. That is achieved by
    # leaving the shared cold tier OFF: without it _gguf_cold_shard_context
    # returns None and no expert is delegated, whatever the ratio says.
    unset SGLANG_MOE_COLD_TIER_SHM || true
    # NOT set to 1,1,1 the way the equal arm does. That variable is the FIRST
    # source of the link weights this arm's solve is built on (resolve_host_
    # shard_ratio), so pinning it equal here would hand the solver a uniform
    # link profile and quietly turn the treatment into the baseline. Byte
    # ownership is held at the baseline by the line above instead.
    unset SGLANG_MOE_HOST_SHARD_RATIO || true
    # Same refusal as the proportional arm: the placement must be weighted by a
    # timed transfer, not by a datasheet.
    export SGLANG_MOE_HOST_SHARD_MIN_PROVENANCE=measured
    if [ "$ARM" = "compute-cal" ]; then
      if [ -z "${SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS:-}" ]; then
        echo "FAIL ARM=compute-cal needs SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS" >&2
        echo "     derive them from the EQUAL arm's per-rank h2d_bytes; see" >&2
        echo "     scripts/dev/394_s2_proof/ARM3_COMPUTE.md" >&2
        exit 1
      fi
      # The FALSIFIED arm, and it names itself on the command line (#458). The
      # coefficients are read only under this symbol; plain 'link' refuses
      # while they are exported, so a leftover export from this arm cannot
      # turn the next arm into this one.
      ARM_ARGS+=(--rank-moe-ratio link-calibrated)
    else
      unset SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS || true
      ARM_ARGS+=(--rank-moe-ratio link)
    fi
    ;;
  *)
    echo "FAIL unknown ARM=$ARM (equal|proportional|compute|compute-cal)" >&2
    exit 1
    ;;
esac

LAUNCH=(
  "$VENV/bin/python" -u -m sglang.launch_server
  --model-path "$GGUF_DIR/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
  --tp-size 3 --rank-gpu-id "$RANK_GPU_ID" --rank-tp-ratio auto
  --rank-auto-reserve-mib "$RESERVE_MIB"
  --rank-moe-resident-fraction "$RESIDENT_FRACTION"
  --kv-cache-dtype fp8_e4m3
  --context-length 8192 --max-running-requests 1
  --chunked-prefill-size 512
  --disable-cuda-graph
  --trust-remote-code --enable-metrics
  "${ARM_ARGS[@]+"${ARM_ARGS[@]}"}"
  --host 127.0.0.1 --port "$PORT"
)

if [ "$DRY_RUN" = "1" ]; then
  echo "ARM=$ARM"
  printf '%s\n' "${LAUNCH[*]}"
  exit 0
fi

setsid "${LAUNCH[@]}" > "$RUN/boot_$ARM.log" 2>&1 &
echo $! > "$RUN/boot_$ARM.launchpid"
echo "launched ARM=$ARM pid=$(cat "$RUN/boot_$ARM.launchpid") port=$PORT run=$RUN"
echo "next: wait for readiness with a BOUNDED curl -m loop, then read_arm.py"
