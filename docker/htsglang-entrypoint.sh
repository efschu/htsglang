#!/usr/bin/env bash
# Entrypoint for the htsglang (uneven-TP sglang fork) runtime image.
#
# Every launch_server flag that this fork adds, plus the stock flags a
# deployment normally touches, is exposed as an environment variable. Empty ENV
# => the flag is omitted entirely, so a variable can be cleared to fall back to
# sglang's own default (e.g. RANK_GPU_ID="" disables the uneven-TP mapping).
# Any extra arguments passed to `docker run` are appended verbatim AFTER the
# generated arg list, so they win for single-value argparse flags and can add
# flags this script does not know about.
#
# Defaults here are deliberately NEUTRAL — a bare `docker run` behaves like a
# stock sglang image (TP=1, no speculation, no HiCache, no rank mapping). The
# rig-specific production profile lives in docker/htsglang.yml and
# docker/htsglang.env.example, not in this file.
#
# Modes (MODE, default "server"):
#   server    python3 -m sglang.launch_server   (the OpenAI-compatible server)
#   planner   python3 -m sglang.planner --serve (the web UI; it starts sglang
#                                                itself, inside this container)
#
# Escape hatches:
#   docker run ... --help               -> print this ENV surface and exit
#   docker run ... bash                 -> interactive shell
#   docker run ... python -m sglang...  -> fully custom command
set -euo pipefail

usage() {
    cat <<'EOF'
htsglang runtime image — configuration is by environment variable.

  MODE=server|planner            what to start (default: server)

Anything after the image name is appended to the generated command line, so
`docker run htsglang --enable-metrics` just works, and repeating a flag that is
already generated overrides it.

--- model and identity -----------------------------------------------------
  MODEL_PATH                     model dir, or the .gguf file for GGUF
  SERVED_MODEL_NAME              name reported by /v1/models
  TOKENIZER_PATH                 needed for GGUF: the sibling dir with
                                 config.json + tokenizer files
  LOAD_FORMAT                    auto|gguf|hibernate|...
  QUANTIZATION                   e.g. gguf, fp8, compressed-tensors
  DTYPE                          auto|bfloat16|float16 (float16 on sm75)
  CONTEXT_LENGTH                 max model len
  CHAT_TEMPLATE                  path or builtin name. The froggeric v21.3
                                 template ships at
                                 /etc/htsglang/chat_template.jinja
  REASONING_PARSER               e.g. qwen3
  TOOL_CALL_PARSER               e.g. qwen3_coder
  TRUST_REMOTE_CODE              1 to pass --trust-remote-code

--- tensor parallelism, uneven and co-located (fork) -----------------------
  TP_SIZE                        number of TP ranks
  RANK_GPU_ID                    per-rank physical GPU ids, e.g. 0,1,2 —
                                 duplicates co-locate ranks on one GPU and
                                 require NCCL >= 2.30 (bundled)
  RANK_TP_RATIO                  auto | explicit ratio, e.g. 4,3,3
  RANK_AUTO_RESERVE_MIB          headroom left per GPU in ratio mode
  RANK_GPU_MEMORY_MIB            absolute per-rank budget; REQUIRED for
                                 co-location, mutually exclusive with the
                                 ratio/reserve pair
  RANK_MLP_RATIO                 override the MLP split
  RANK_MOE_RATIO                 override the expert split
  RANK_VOCAB_RATIO               override the vocab/lm_head split
  RANK_KV_RATIO                  capacity|speed|explicit — KV token split
  RANK_PERF_TUNE                 auto-performance planner
  BASE_GPU_ID                    stock sglang path; only when RANK_GPU_ID=""
  DCP_SIZE                       decode context parallel size
  SWA_POOL_SIZING                cap|ratio (required =cap for SWA-DCP)
  SGLANG_UNEVEN_MLP_VECTOR       MLP self-calibration vector; the first boot
                                 suggests one, set it and restart

--- second node ------------------------------------------------------------
  NNODES                         total nodes (>1 enables multi-node)
  NODE_RANK                      0 on the head node
  DIST_INIT_ADDR                 <head-ip>:<port>, reachable from both nodes
  SGLANG_BARLINK                   1 = host-staged cross-vendor collectives
  SGLANG_BARLINK_TRANSPORT         ucx|shm|gloo
  SGLANG_BARLINK_UCX_LIB           path to a matching libucp.so.0
  UCX_TLS UCX_IB_GID_INDEX UCX_NET_DEVICES
                                 passed through unchanged
  Note: barlink synchronises with the host inside every collective, so it
  cannot be captured — ENFORCE_EAGER=1 is required with it.

--- kv cache and speculative decoding --------------------------------------
  KV_CACHE_DTYPE                 auto|fp8_e4m3|...
  MEM_FRACTION_STATIC            stock global memory fraction; ignored when
                                 RANK_GPU_MEMORY_MIB / RANK_TP_RATIO is set
  SPECULATIVE_ALGORITHM          NEXTN|EAGLE|EAGLE3|DFLASH|...
  SPECULATIVE_NUM_STEPS
  SPECULATIVE_EAGLE_TOPK
  SPECULATIVE_NUM_DRAFT_TOKENS
  SPECULATIVE_DRAFT_MODEL_PATH   the draft model — often a SEPARATE path from
                                 MODEL_PATH; for Qwen3.5/3.6 GGUF the MTP
                                 layer lives in the same .gguf, so point it at
                                 the same file
  SPECULATIVE_DRAFT_PLACEMENT    where the draft model lives
  SPECULATIVE_CROSS_ALGORITHM    NEXTN<->DFLASH ladder
  DISABLE_CUDA_GRAPH             1 disables CUDA graphs (sglang's flag name)
  ENFORCE_EAGER                  1, alias for the same thing

--- caches that survive a restart ------------------------------------------
  ENABLE_HIERARCHICAL_CACHE      1 to enable HiCache
  HICACHE_STORAGE_BACKEND        file|nixl|mooncake|hf3fs
  HICACHE_RATIO
  HICACHE_MEM_LAYOUT
  HICACHE_STORAGE_DIR            L3 directory for the file backend. MOUNT IT,
                                 otherwise the L3 tier is lost on restart
                                 (default /var/lib/htsglang/hicache)
  HIBERNATE_DIR                  suspend-to-disk of the per-rank weight
                                 shards; a matching manifest turns the next
                                 boot into a fast restore. MOUNT IT
  ENABLE_WEIGHTS_DISK_BACKUP     1, required together with HIBERNATE_DIR
  ENABLE_KV_SESSION_OFFLOAD      1 to spill idle sessions' KV to host RAM
  KV_SESSION_OFFLOAD_HOST_RAM_GIB
  ENABLE_FAST_LANE               1 for the short-request fast lane

--- server sizing and observability ----------------------------------------
  HOST PORT                      bind address (default 0.0.0.0:30000)
  MAX_RUNNING_REQUESTS
  MAX_TOTAL_TOKENS
  ATTENTION_BACKEND              flashinfer|triton|torch_native
  DISABLE_CUSTOM_ALL_REDUCE      1 on rigs without P2P
  DISABLE_RADIX_CACHE            1
  ENABLE_METRICS                 1 exposes /metrics
  LOG_LEVEL                      info|debug|warning
  EXTRA_ARGS                     word-split and appended, for anything else

--- planner / GUI mode (MODE=planner) --------------------------------------
  PLANNER_HOST PLANNER_PORT      web UI bind (default 0.0.0.0:8780)
  PLANNER_ARGS                   extra `python -m sglang.planner` arguments
  SGLANG_MODEL_ROOTS             os.pathsep-separated dirs the UI scans for
                                 models (default ~/.cache/huggingface/hub and
                                 ./models, the latter CWD-relative)
  SGLANG_PLANNER_PROFILES        saved config profiles JSON
  SGLANG_PLANNER_GRAPH_ANCHORS   CUDA-graph memory anchors JSON
  The planner starts sglang itself, in this container, as a subprocess group.
  It has NO authentication: anyone who reaches the port can start, stop and
  download models. Publish it to 127.0.0.1 only, or put it behind a proxy.
  The planner reads NVML for its short hardware probe. Clock and power control
  (nvidia-smi -pm/-lgc/-lmc/-pl) is REFUSED by the driver from inside a
  container even as root with full capabilities — run those on the host.

--- state directories (bind-mount these) -----------------------------------
  /root/.cache/sglang            hw_profile-*.json (rig probe) and
                                 kv_budget-*.json (measured KV budget). Losing
                                 it costs a re-probe on every boot
  /root/.cache/flashinfer        flashinfer JIT cubins. Losing it costs
                                 minutes of recompilation on every boot
  /root/.cache/torch_extensions  the HiCache native hash extension
  /root/.triton                  Triton kernel cache
EOF
}

# --- debug / custom-command escape hatch ---------------------------------
if [ "$#" -gt 0 ]; then
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        bash|sh|/bin/bash|/bin/sh)
            exec "$@"
            ;;
        python|python3|/usr/bin/python3)
            exec "$@"
            ;;
    esac
fi

add() { # add <flag> <value> : append only when value is non-empty
    [ -n "$2" ] && args+=("$1" "$2")
    return 0  # never fail under `set -e` when the value is empty (the
              # `&&` short-circuit would otherwise exit the whole script)
}
add_flag() { # add_flag <flag> <bool-ish> : append flag when value is truthy
    case "$2" in 1|true|TRUE|yes|on) args+=("$1") ;; esac
    return 0
}

# --- state directories ----------------------------------------------------
# Defaults point at paths the image creates, so an unmounted run still works
# and only loses the cache when the container is removed.
: "${HICACHE_STORAGE_DIR:=/var/lib/htsglang/hicache}"
# The HiCache file backend reads this env var and falls back to /tmp/hicache
# when it is unset, which is INSIDE the container's writable layer and dies
# with the container. Exporting it here is what makes the mount effective.
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="${SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR:-$HICACHE_STORAGE_DIR}"
mkdir -p "$SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR" 2>/dev/null || true

# --- planner / GUI mode ---------------------------------------------------
: "${MODE:=server}"
if [ "$MODE" = "planner" ] || [ "$MODE" = "gui" ]; then
    : "${PLANNER_HOST:=0.0.0.0}"
    : "${PLANNER_PORT:=8780}"
    : "${PLANNER_ARGS:=}"
    # The planner spawns `python3 -m sglang.launch_server` in its own process
    # group and reaps it itself, but a crashed worker can still orphan onto
    # PID 1. Run the container with an init process (compose: `init: true`,
    # docker run: `--init`) so orphans get reaped.
    if [ "$PLANNER_HOST" != "127.0.0.1" ] && [ "$PLANNER_HOST" != "localhost" ]; then
        echo "[htsglang-entrypoint] WARNING: the planner UI has no authentication" >&2
        echo "[htsglang-entrypoint] and it can start, stop and download models." >&2
        echo "[htsglang-entrypoint] Publish this port to 127.0.0.1 only." >&2
    fi
    args=(python3 -m sglang.planner --serve --host "$PLANNER_HOST" --port "$PLANNER_PORT")
    # shellcheck disable=SC2206  # deliberate word splitting
    [ -n "$PLANNER_ARGS" ] && args+=($PLANNER_ARGS)
    args+=("$@")
    echo "[htsglang-entrypoint] mode=planner exec: ${args[*]}" >&2
    exec "${args[@]}"
fi

if [ "$MODE" != "server" ]; then
    echo "[htsglang-entrypoint] FATAL: unknown MODE='$MODE' (server|planner)" >&2
    exit 2
fi

# --- model + identity -----------------------------------------------------
: "${MODEL_PATH:=}"
if [ -z "$MODEL_PATH" ]; then
    echo "[htsglang-entrypoint] FATAL: MODEL_PATH is unset." >&2
    echo "[htsglang-entrypoint] Run with --help for the full variable list." >&2
    exit 2
fi
: "${SERVED_MODEL_NAME:=}"
# GGUF: point MODEL_PATH at the .gguf file, TOKENIZER_PATH at the sibling dir
# (config.json + tokenizer live there), and set LOAD_FORMAT/QUANTIZATION=gguf.
: "${TOKENIZER_PATH:=}"
: "${LOAD_FORMAT:=}"
: "${QUANTIZATION:=}"
: "${DTYPE:=}"
: "${CONTEXT_LENGTH:=}"

# --- (uneven) tensor parallelism -----------------------------------------
: "${TP_SIZE:=1}"
: "${RANK_GPU_ID:=}"
# Two mutually exclusive memory modes:
#   1) auto uneven split: RANK_TP_RATIO=auto + RANK_AUTO_RESERVE_MIB fills each
#      GPU's free VRAM, one rank per physical GPU.
#   2) absolute per-rank budget: RANK_GPU_MEMORY_MIB=<MiB>. REQUIRED for
#      multi-rank-per-GPU co-location (duplicate ids in RANK_GPU_ID, e.g.
#      RANK_GPU_ID=0,0,1,2). It disables the ratio/reserve flags, which the
#      fork rejects alongside an absolute budget.
: "${RANK_GPU_MEMORY_MIB:=}"
if [ -n "$RANK_GPU_MEMORY_MIB" ]; then
    RANK_TP_RATIO=""
    RANK_AUTO_RESERVE_MIB=""
else
    : "${RANK_TP_RATIO:=}"
    : "${RANK_AUTO_RESERVE_MIB:=}"
fi
: "${RANK_MLP_RATIO:=}"
: "${RANK_MOE_RATIO:=}"
: "${RANK_VOCAB_RATIO:=}"
: "${RANK_KV_RATIO:=}"
: "${RANK_PERF_TUNE:=}"
: "${DCP_SIZE:=}"
: "${SWA_POOL_SIZING:=}"
# BASE_GPU_ID is only meaningful when RANK_GPU_ID is empty (stock sglang path).
: "${BASE_GPU_ID:=}"

# --- second node ----------------------------------------------------------
: "${NNODES:=}"
: "${NODE_RANK:=}"
: "${DIST_INIT_ADDR:=}"
# barlink and UCX are read from the environment by the fork / by libucp; export
# whatever the caller set so the values reach the worker processes unchanged.
for v in SGLANG_BARLINK SGLANG_BARLINK_TRANSPORT SGLANG_BARLINK_UCX_LIB \
         SGLANG_BARLINK_UCX_OVERLAP UCX_TLS UCX_IB_GID_INDEX UCX_NET_DEVICES; do
    if [ -n "${!v:-}" ]; then export "${v?}"; fi
done

# --- kv cache / speculative decode ---------------------------------------
: "${KV_CACHE_DTYPE:=}"
: "${MEM_FRACTION_STATIC:=}"
: "${SPECULATIVE_ALGORITHM:=}"
: "${SPECULATIVE_NUM_STEPS:=}"
: "${SPECULATIVE_EAGLE_TOPK:=}"
: "${SPECULATIVE_NUM_DRAFT_TOKENS:=}"
# The draft model usually lives somewhere else than the main model. For
# Qwen3.5/3.6 GGUF the MTP layer is in the SAME .gguf, so set it to MODEL_PATH.
: "${SPECULATIVE_DRAFT_MODEL_PATH:=}"
: "${SPECULATIVE_DRAFT_PLACEMENT:=}"
: "${SPECULATIVE_CROSS_ALGORITHM:=}"
: "${ENFORCE_EAGER:=}"
: "${DISABLE_CUDA_GRAPH:=}"

# --- chat template / parsers ---------------------------------------------
: "${CHAT_TEMPLATE:=}"
: "${REASONING_PARSER:=}"
: "${TOOL_CALL_PARSER:=}"

# --- hierarchical (HiCache) cache / persistence ---------------------------
: "${ENABLE_HIERARCHICAL_CACHE:=}"
: "${HICACHE_STORAGE_BACKEND:=}"
: "${HICACHE_RATIO:=}"
: "${HICACHE_MEM_LAYOUT:=}"
: "${HIBERNATE_DIR:=}"
: "${ENABLE_WEIGHTS_DISK_BACKUP:=}"
: "${ENABLE_KV_SESSION_OFFLOAD:=}"
: "${KV_SESSION_OFFLOAD_HOST_RAM_GIB:=}"
: "${ENABLE_FAST_LANE:=}"

# --- server sizing / observability ----------------------------------------
: "${MAX_RUNNING_REQUESTS:=}"
: "${MAX_TOTAL_TOKENS:=}"
: "${ATTENTION_BACKEND:=}"
: "${DISABLE_CUSTOM_ALL_REDUCE:=}"
: "${DISABLE_RADIX_CACHE:=}"
: "${ENABLE_METRICS:=}"
: "${TRUST_REMOTE_CODE:=}"
: "${LOG_LEVEL:=}"
: "${HOST:=0.0.0.0}"
: "${PORT:=30000}"
: "${EXTRA_ARGS:=}"

# --- uneven-TP runtime knobs (passed through to the process) --------------
export SGLANG_UNEVEN_MLP_VECTOR="${SGLANG_UNEVEN_MLP_VECTOR:-}"

args=(python3 -m sglang.launch_server)

add --model-path "$MODEL_PATH"
add --served-model-name "$SERVED_MODEL_NAME"
add --tokenizer-path "$TOKENIZER_PATH"
add --load-format "$LOAD_FORMAT"
add --quantization "$QUANTIZATION"
add --dtype "$DTYPE"
add --context-length "$CONTEXT_LENGTH"
add --tp-size "$TP_SIZE"
add --rank-gpu-id "$RANK_GPU_ID"
add --rank-tp-ratio "$RANK_TP_RATIO"
add --rank-auto-reserve-mib "$RANK_AUTO_RESERVE_MIB"
add --rank-gpu-memory-mib "$RANK_GPU_MEMORY_MIB"
add --rank-mlp-ratio "$RANK_MLP_RATIO"
add --rank-moe-ratio "$RANK_MOE_RATIO"
add --rank-vocab-ratio "$RANK_VOCAB_RATIO"
add --rank-kv-ratio "$RANK_KV_RATIO"
add --rank-perf-tune "$RANK_PERF_TUNE"
add --dcp-size "$DCP_SIZE"
add --swa-pool-sizing "$SWA_POOL_SIZING"
add --base-gpu-id "$BASE_GPU_ID"
add --nnodes "$NNODES"
add --node-rank "$NODE_RANK"
add --dist-init-addr "$DIST_INIT_ADDR"
add --kv-cache-dtype "$KV_CACHE_DTYPE"
add --mem-fraction-static "$MEM_FRACTION_STATIC"
add --speculative-algorithm "$SPECULATIVE_ALGORITHM"
add --speculative-num-steps "$SPECULATIVE_NUM_STEPS"
add --speculative-eagle-topk "$SPECULATIVE_EAGLE_TOPK"
add --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS"
add --speculative-draft-model-path "$SPECULATIVE_DRAFT_MODEL_PATH"
add --speculative-draft-placement "$SPECULATIVE_DRAFT_PLACEMENT"
add --speculative-cross-algorithm "$SPECULATIVE_CROSS_ALGORITHM"
# sglang has no --enforce-eager (that is vLLM's spelling); the equivalent is
# --disable-cuda-graph. ENFORCE_EAGER is kept as an alias because the fork's
# own notes use that word for the barlink requirement. Either variable sets the
# same flag, and the flag is emitted at most once.
case "${ENFORCE_EAGER}${DISABLE_CUDA_GRAPH}" in
    *1*|*true*|*TRUE*|*yes*|*on*) args+=(--disable-cuda-graph) ;;
esac
add_flag --disable-custom-all-reduce "$DISABLE_CUSTOM_ALL_REDUCE"
add_flag --disable-radix-cache "$DISABLE_RADIX_CACHE"
add --chat-template "$CHAT_TEMPLATE"
add --reasoning-parser "$REASONING_PARSER"
add --tool-call-parser "$TOOL_CALL_PARSER"
add_flag --enable-hierarchical-cache "$ENABLE_HIERARCHICAL_CACHE"
add --hicache-storage-backend "$HICACHE_STORAGE_BACKEND"
add --hicache-ratio "$HICACHE_RATIO"
add --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
add --hibernate-dir "$HIBERNATE_DIR"
add_flag --enable-weights-disk-backup "$ENABLE_WEIGHTS_DISK_BACKUP"
add_flag --enable-kv-session-offload "$ENABLE_KV_SESSION_OFFLOAD"
add --kv-session-offload-host-ram-gib "$KV_SESSION_OFFLOAD_HOST_RAM_GIB"
add_flag --enable-fast-lane "$ENABLE_FAST_LANE"
add --max-running-requests "$MAX_RUNNING_REQUESTS"
add --max-total-tokens "$MAX_TOTAL_TOKENS"
add --attention-backend "$ATTENTION_BACKEND"
add_flag --enable-metrics "$ENABLE_METRICS"
add --log-level "$LOG_LEVEL"
add_flag --trust-remote-code "$TRUST_REMOTE_CODE"
add --host "$HOST"
add --port "$PORT"

# EXTRA_ARGS is word-split on purpose: it is the escape hatch for flags this
# script does not model. Quoting inside it is not supported — use the trailing
# `docker run` arguments for values containing spaces.
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && args+=($EXTRA_ARGS)

# Append any extra docker-run args (override single-value flags / add flags).
args+=("$@")

echo "[htsglang-entrypoint] exec: ${args[*]}" >&2
exec "${args[@]}"
