#!/usr/bin/env bash
# Start the model-aware Anthropic split proxy.
#
# What it buys you over scripts/dev/local_model_agent.sh: that script points a
# WHOLE claude process at the local server. This router lets ONE process keep
# its parent turns on api.anthropic.com while only the subagents whose
# `"model"` matches a local id are served by htsglang. Claude Code has no
# per-subagent endpoint key -- ANTHROPIC_BASE_URL is process-global -- so the
# split has to happen in front of both endpoints, which is what this is.
#
# Usage:
#   claude_local_router.sh [-p PORT] [-l LOCAL_BASE] [-u UPSTREAM_BASE]
#                          [-m MODEL_ID]...
#
# With no -m the local model id is resolved from the live server, so the router
# follows a model switch instead of pinning a name.
#
# Then, in a SEPARATE process (the running session is untouched):
#
#   ANTHROPIC_BASE_URL=http://127.0.0.1:30099 \
#   claude -p "your task" --agents "$(cat <<'JSON'
#   {"local-model": {
#      "description": "Runs on the local htsglang server",
#      "prompt": "You are a local worker. Use the tools you are given.",
#      "tools": ["Read", "Grep", "Glob"],
#      "model": "Qwen3.6-27B"}}
#   JSON
#   )"
#
# Routing counters (how many requests went where) are at
# http://127.0.0.1:PORT/__router/stats -- that is the evidence the parent stayed
# on the Anthropic API while the subagent ran locally.

set -euo pipefail

PORT="${CLAUDE_ROUTER_PORT:-30099}"
HOST="${CLAUDE_ROUTER_HOST:-127.0.0.1}"
LOCAL_BASE="${LOCAL_MODEL_BASE_URL:-http://127.0.0.1:30030}"
UPSTREAM_BASE="${CLAUDE_ROUTER_UPSTREAM:-https://api.anthropic.com}"
PYTHON="${PYTHON:-python3}"
MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port)          PORT="$2"; shift 2 ;;
        -H|--host)          HOST="$2"; shift 2 ;;
        -l|--local-base)    LOCAL_BASE="$2"; shift 2 ;;
        -u|--upstream-base) UPSTREAM_BASE="$2"; shift 2 ;;
        -m|--model)         MODELS+=("$2"); shift 2 ;;
        -h|--help)          sed -n '1,33p' "$0"; exit 0 ;;
        *)                  echo "claude_local_router: unknown argument $1" >&2; exit 2 ;;
    esac
done

# Resolve from the live server when no id was pinned: the router should follow
# a model switch rather than carry a hardcoded name.
if [[ ${#MODELS[@]} -eq 0 ]]; then
    resolved="$(curl -sS -m 10 "$LOCAL_BASE/v1/models" \
        | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
    if [[ -z "$resolved" ]]; then
        echo "claude_local_router: could not resolve a model id from $LOCAL_BASE/v1/models" >&2
        exit 3
    fi
    MODELS=("$resolved")
fi

ARGS=()
for m in "${MODELS[@]}"; do
    ARGS+=(--local-model "$m")
done

echo "claude_local_router: ${MODELS[*]} -> $LOCAL_BASE, everything else -> $UPSTREAM_BASE"
echo "claude_local_router: listening on http://$HOST:$PORT (stats: /__router/stats)"

exec "$PYTHON" -m sglang.srt.entrypoints.anthropic.router \
    --host "$HOST" \
    --port "$PORT" \
    --local-base "$LOCAL_BASE" \
    --upstream-base "$UPSTREAM_BASE" \
    "${ARGS[@]}"
