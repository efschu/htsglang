#!/bin/bash
# Boot one recipe of the spill-night matrix.
#
# A boot is the expensive unit of this window, so recipes are grouped to cover
# as many matrix cells as one process can. Every recipe here is a DEVIATION
# from the production serving recipe (/root/bin/start-serving-30030.sh) and each
# deviation is justified in a comment -- silent divergence from the production
# recipe is how a matrix result stops describing production.
#
# Port 30041 throughout: never 30030 (production) and never 30099 (router).
#
# Usage:  boot.sh <recipe>            recipe in K0 K1 K2 K3 K4 L1 C1
# Env:    MTT   -- --max-total-tokens (pressure knob; default 8192)
#         PORT  -- default 30041
#         DRY   -- 1 = print the command and exit without launching
set -u

WT=/spinning/wt-spill-matrix
VENV=/spinning/htsglang-gpu/.venv
MODEL_INT8=/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8
MODEL_GGUF=/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-MTP-Q3_K_M-GGUF

RECIPE=${1:?usage: boot.sh <K0|K1|K2|K3|K4|L1|C1>}
PORT=${PORT:-30041}
MTT=${MTT:-8192}
DRY=${DRY:-0}
LOG=/spinning/spill-matrix-${RECIPE}.boot.log

export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16

# --- common spine (shared by every recipe) --------------------------------
# Matches production on: model, TP=3 + rank-gpu-id, auto-performance ratio,
# the solo reserve vector, fp8 KV, context length, metrics, trust-remote-code.
COMMON=(
    --model-path "$MODEL_INT8"
    --served-model-name Qwen3.6-27B
    --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance
    --rank-auto-reserve-mib 5500,3800,3800
    --kv-cache-dtype fp8_e4m3 --context-length 262144
    --max-running-requests 4
    --enable-metrics --trust-remote-code
    --host 127.0.0.1 --port "$PORT"
)

# Pressure knob. Production leaves the KV pool at whatever the reserve permits;
# here we cap it so a spill is reachable in seconds instead of needing a
# 262k-token prompt. This is the ONLY thing that makes the window fit its box.
PRESSURE=(--max-total-tokens "$MTT")

# kvso spine. NOTE: no --enable-hierarchical-cache anywhere in the K recipes --
# server_args.py:6679 refuses the pair outright (matrix S1). This is why the
# HOT arm cannot reuse the production recipe.
KVSO=(
    --enable-kv-session-offload
    --kv-session-offload-host-ram-gib 16
    --kv-session-offload-max-spills 3
)

case "$RECIPE" in
  K0)
    # A-vs-A floor: the K1 recipe with the SAME flags, driven without forcing a
    # spill. Any HOT delta is measured against this, not against production.
    ARGS=("${COMMON[@]}" "${PRESSURE[@]}" "${KVSO[@]}")
    ;;
  K1)
    # kvso base mechanics, no speculation: spill, host decode, wave back,
    # restore, multi-session FCFS victim order.
    #
    # The ladder rides along deliberately. On the production recipe #287 has NO
    # actuator at all -- wired_relief_features() (kv_ladder_auto.py:85-109)
    # returns empty unless one of --kv-reshard-vectors / --enable-kv-session-
    # offload / --max-running-requests-ceiling is set, and production sets
    # none, so its rung flips log "no actuator declared" and move nothing.
    # Here kvso IS on, so the ladder's `session_offload` relief has a real
    # actuator: this boot is the one place the ladder genuinely actuates.
    ARGS=("${COMMON[@]}" "${PRESSURE[@]}" "${KVSO[@]}"
          --kv-pressure-ladder auto --max-running-requests-ceiling 16)
    ;;
  K2)
    # Headline cell: kvso x NEXTN speculation. Needs KVSO_ALLOW_SPEC=1
    # (server_args.py:6620) and, for resume-under-spec, KVSO_RESUME=1
    # (kv_session_offload.py:503). Both are opt-in bring-up gates, not refusals.
    export KVSO_ALLOW_SPEC=1
    export KVSO_RESUME=1
    ARGS=("${COMMON[@]}" "${PRESSURE[@]}" "${KVSO[@]}"
          --speculative-algorithm NEXTN --speculative-num-steps 3
          --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
          --kv-session-offload-spec-in-tick
          --kv-pressure-ladder auto --max-running-requests-ceiling 16)
    ;;
  K3)
    # Spill-graph path on top of the K2 recipe.
    export KVSO_ALLOW_SPEC=1
    export KVSO_RESUME=1
    export SGLANG_KVSO_SPILL_GRAPH=1
    ARGS=("${COMMON[@]}" "${PRESSURE[@]}" "${KVSO[@]}"
          --speculative-algorithm NEXTN --speculative-num-steps 3
          --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    ;;
  K4)
    # Budget / cadence / fast-lane knobs. Budgets are BOOT flags, hence a
    # separate boot rather than a knob turned inside K1.
    ARGS=("${COMMON[@]}" "${PRESSURE[@]}" "${KVSO[@]}"
          --kv-session-offload-tick-adaptive
          --kv-session-offload-budget-total-tokens 65536
          --kv-session-offload-budget-session-tokens 16384
          --kv-session-offload-budget-max-sessions 2
          --kv-session-offload-spill-cooldown-seconds 2
          --enable-fast-lane --retraction-policy priority)
    ;;
  L1)
    # LEITER head cell: the four dynamic-step features ARMED at once. Note what
    # this recipe must drop to exist at all: --enable-hierarchical-cache and its
    # storage backend, because vram_dial.py:1046 refuses the dial when a hicache
    # storage backend is configured (matrix S3). So this recipe is deliberately
    # NOT the production recipe -- it is the nearest bootable neighbour of it.
    # --kv-reshard-vectors is REQUIRED for #297 to be attempted at all, and it
    # also gives the ladder its `dcp_ratio` relief actuator. Default pair is
    # derived from the recipe's own rank_kv_speed_weights [7,3,3]; override
    # with RESHARD_VEC once the boot log prints the resolved grid.
    #
    # kvso is deliberately ABSENT here: vram_dial.py:1052 refuses the dial when
    # kv-session-offload is on, so the dial and the HOT arm can never share a
    # boot. That is why #287's session_offload relief is exercised in K1/K2 and
    # its dcp_ratio/admission_cap reliefs here.
    #
    # --regime-trace is not optional decoration: the REGIME-OBSERVE summary log
    # ends "NOT ACTUATED (observe-only)." unconditionally, in BOTH modes, so the
    # trace file's "actuated" field is the only trustworthy signal.
    ARGS=("${COMMON[@]}" "${PRESSURE[@]}"
          --speculative-algorithm NEXTN --speculative-num-steps 3
          --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
          --kv-pressure-ladder auto --max-running-requests-ceiling 16
          --kv-reshard-vectors "${RESHARD_VEC:-7,3,3;5,4,4}"
          --enable-vram-dial
          --regime-controller observe
          --regime-trace /spinning/spill-matrix-L1.regime.jsonl
          --enable-fast-lane --retraction-policy priority)
    ;;
  C1)
    # COLD: hibernate-to-disk round trip. GGUF-scoped by a hard refusal in
    # server_args.py, so this recipe swaps the model -- an INT8 checkpoint
    # cannot reach the hibernate path at all.
    mkdir -p /spinning/hibernate-matrix
    # --load-format gguf is MANDATORY here and not redundant: the hibernate
    # gate reads server_args.load_format, and 'auto' does not resolve to 'gguf'
    # before that check runs, so an auto load of a GGUF checkpoint is refused
    # with "got load_format='auto'". Found by smoke.sh step 6 at the desk.
    ARGS=(--model-path "$MODEL_GGUF"
          --load-format gguf
          --served-model-name Qwen3.6-27B-GGUF
          --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance
          --rank-auto-reserve-mib 5500,3800,3800
          --kv-cache-dtype fp8_e4m3
          --max-running-requests 4
          --enable-weights-disk-backup --hibernate-dir /spinning/hibernate-matrix
          --enable-metrics --trust-remote-code
          --host 127.0.0.1 --port "$PORT")
    ;;
  *) echo "unknown recipe $RECIPE" >&2; exit 2 ;;
esac

if [ "$DRY" = "1" ]; then
    printf 'recipe=%s port=%s log=%s\n' "$RECIPE" "$PORT" "$LOG"
    printf '%s\n' "${VENV}/bin/python -m sglang.launch_server ${ARGS[*]}"
    exit 0
fi

cd "$WT"
setsid "$VENV/bin/python" -m sglang.launch_server "${ARGS[@]}" > "$LOG" 2>&1 &
echo "recipe=$RECIPE pgid=$! port=$PORT log=$LOG"
