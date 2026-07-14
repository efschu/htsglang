#!/usr/bin/env bash
# Entrypoint for the htsglang (uneven-TP sglang fork) runtime image.
#
# Every launch_server flag is exposed as an environment variable with the
# coordinator-verified production defaults baked in. Empty ENV => the flag is
# omitted, so you can drop a default cleanly (e.g. RANK_GPU_ID="" to disable
# the uneven-TP mapping for a single-GPU smoke test). Any extra arguments
# passed to `docker run` are appended verbatim AFTER the generated arg list,
# so they win for single-value argparse flags and can add new flags.
#
# Escape hatches:
#   docker run ... bash                 -> interactive shell
#   docker run ... python -m sglang...  -> fully custom command (first token is
#                                          an executable that exists on PATH)
set -euo pipefail

# --- debug / custom-command escape hatch ---------------------------------
if [ "$#" -gt 0 ]; then
    case "$1" in
        bash|sh|/bin/bash|/bin/sh)
            exec "$@"
            ;;
        python|python3|/usr/bin/python3)
            exec "$@"
            ;;
    esac
fi

# --- model + identity -----------------------------------------------------
: "${MODEL_PATH:=/model}"
: "${SERVED_MODEL_NAME:=qwen3.6-27b-aeon-fp8}"

# --- (uneven) tensor parallelism -----------------------------------------
: "${TP_SIZE:=3}"
: "${RANK_GPU_ID:=0,1,2}"
# Two mutually exclusive memory modes:
#   1) auto uneven split (default): RANK_TP_RATIO=auto + RANK_AUTO_RESERVE_MIB
#      fills each GPU's free VRAM, one rank per physical GPU.
#   2) absolute per-rank budget: RANK_GPU_MEMORY_MIB=<MiB>. REQUIRED for
#      multi-rank-per-GPU co-location (duplicate ids in RANK_GPU_ID, e.g.
#      RANK_GPU_ID=0,0,1,2). It disables the ratio/reserve flags, which the
#      fork rejects alongside an absolute budget.
: "${RANK_GPU_MEMORY_MIB:=}"
if [ -n "$RANK_GPU_MEMORY_MIB" ]; then
    RANK_TP_RATIO=""
    RANK_AUTO_RESERVE_MIB=""
else
    : "${RANK_TP_RATIO:=auto}"
    : "${RANK_AUTO_RESERVE_MIB:=2048}"
fi
# BASE_GPU_ID is only meaningful when RANK_GPU_ID is empty (stock sglang path).
: "${BASE_GPU_ID:=}"

# --- kv cache / speculative decode ---------------------------------------
: "${KV_CACHE_DTYPE:=fp8_e4m3}"
: "${SPECULATIVE_ALGORITHM:=NEXTN}"
: "${SPECULATIVE_NUM_STEPS:=3}"
: "${SPECULATIVE_EAGLE_TOPK:=1}"
: "${SPECULATIVE_NUM_DRAFT_TOKENS:=4}"

# --- chat template / parsers ---------------------------------------------
: "${CHAT_TEMPLATE:=/etc/htsglang/chat_template.jinja}"
: "${REASONING_PARSER:=qwen3}"
: "${TOOL_CALL_PARSER:=qwen3_coder}"

# --- hierarchical (HiCache) cache ----------------------------------------
: "${ENABLE_HIERARCHICAL_CACHE:=1}"
: "${HICACHE_STORAGE_BACKEND:=file}"
: "${HICACHE_RATIO:=0.5}"
: "${HICACHE_MEM_LAYOUT:=page_first_direct}"

# --- server sizing --------------------------------------------------------
: "${MAX_RUNNING_REQUESTS:=8}"
: "${CONTEXT_LENGTH:=32768}"
: "${TRUST_REMOTE_CODE:=1}"
: "${HOST:=0.0.0.0}"
: "${PORT:=30000}"

# --- uneven-TP / NCCL runtime knobs (passed through to the process) -------
# SGLANG_UNEVEN_MLP_VECTOR and friends are read from the environment by the
# fork; export whatever the caller set so they reach the workers unchanged.
export SGLANG_UNEVEN_MLP_VECTOR="${SGLANG_UNEVEN_MLP_VECTOR:-}"

args=(python3 -m sglang.launch_server)

add() { # add <flag> <value> : append only when value is non-empty
    [ -n "$2" ] && args+=("$1" "$2")
    return 0  # never fail under `set -e` when the value is empty (the
              # `&&` short-circuit would otherwise exit the whole script)
}
add_flag() { # add_flag <flag> <bool-ish> : append flag when value is truthy
    case "$2" in 1|true|TRUE|yes|on) args+=("$1") ;; esac
    return 0
}

add --model-path "$MODEL_PATH"
add --served-model-name "$SERVED_MODEL_NAME"
add --tp-size "$TP_SIZE"
add --rank-gpu-id "$RANK_GPU_ID"
add --rank-tp-ratio "$RANK_TP_RATIO"
add --rank-auto-reserve-mib "$RANK_AUTO_RESERVE_MIB"
add --rank-gpu-memory-mib "$RANK_GPU_MEMORY_MIB"
add --base-gpu-id "$BASE_GPU_ID"
add --kv-cache-dtype "$KV_CACHE_DTYPE"
add --speculative-algorithm "$SPECULATIVE_ALGORITHM"
add --speculative-num-steps "$SPECULATIVE_NUM_STEPS"
add --speculative-eagle-topk "$SPECULATIVE_EAGLE_TOPK"
add --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS"
add --chat-template "$CHAT_TEMPLATE"
add --reasoning-parser "$REASONING_PARSER"
add --tool-call-parser "$TOOL_CALL_PARSER"
add_flag --enable-hierarchical-cache "$ENABLE_HIERARCHICAL_CACHE"
add --hicache-storage-backend "$HICACHE_STORAGE_BACKEND"
add --hicache-ratio "$HICACHE_RATIO"
add --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
add --max-running-requests "$MAX_RUNNING_REQUESTS"
add --context-length "$CONTEXT_LENGTH"
add_flag --trust-remote-code "$TRUST_REMOTE_CODE"
add --host "$HOST"
add --port "$PORT"

# Append any extra docker-run args (override single-value flags / add flags).
args+=("$@")

echo "[htsglang-entrypoint] exec: ${args[*]}" >&2
exec "${args[@]}"
