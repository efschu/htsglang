#!/usr/bin/env bash
# Run a Claude Code agent loop whose INFERENCE runs on the local htsglang
# server instead of the Anthropic API.
#
# Why this exists as a wrapper script rather than an agent frontmatter key:
# Claude Code has no per-subagent endpoint binding. The subagent frontmatter
# schema (name/description/tools/disallowedTools/model/permissionMode/maxTurns/
# skills/mcpServers/hooks/memory/background/effort/isolation/color/
# initialPrompt) carries no baseUrl/provider/env field, and every routing
# variable (ANTHROPIC_BASE_URL and its Bedrock/Vertex/Foundry siblings) is
# process-global. CLAUDE_CODE_SUBAGENT_MODEL only remaps the model NAME on the
# already-configured endpoint. So the only way to keep the parent session on
# the Anthropic API while an agent runs locally is a SEPARATE claude process
# with its own environment -- which is exactly what this script starts.
#
# No translating proxy is involved: htsglang serves the Anthropic Messages
# wire format natively at POST /v1/messages (+ /v1/messages/count_tokens),
# entrypoints/http_server.py:2578 and :2588.
#
# Usage:
#   local_model_agent.sh [-b BASE_URL] [-m MODEL] [-t MAX_TOKENS] [-T TIMEOUT]
#                        [--allow-tools LIST] -- <prompt...>
#   echo "prompt" | local_model_agent.sh
#
# Defaults come from the generated config written by register_local_model.sh.

set -euo pipefail

CONF_DEFAULT="${SGLANG_LOCAL_AGENT_CONF:-$HOME/.config/htsglang/local_model_agent.env}"
if [[ -r "$CONF_DEFAULT" ]]; then
    # shellcheck disable=SC1090
    source "$CONF_DEFAULT"
fi

BASE_URL="${LOCAL_MODEL_BASE_URL:-http://127.0.0.1:30030}"
MODEL="${LOCAL_MODEL_ID:-}"
MAX_TOKENS="${LOCAL_MODEL_MAX_OUTPUT_TOKENS:-4096}"
TIMEOUT="${LOCAL_MODEL_TIMEOUT:-600}"
ALLOW_TOOLS="${LOCAL_MODEL_ALLOWED_TOOLS:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--base-url)     BASE_URL="$2"; shift 2 ;;
        -m|--model)        MODEL="$2"; shift 2 ;;
        -t|--max-tokens)   MAX_TOKENS="$2"; shift 2 ;;
        -T|--timeout)      TIMEOUT="$2"; shift 2 ;;
        --allow-tools)     ALLOW_TOOLS="$2"; shift 2 ;;
        --)                shift; break ;;
        -h|--help)         sed -n '1,30p' "$0"; exit 0 ;;
        *)                 break ;;
    esac
done

PROMPT="$*"
if [[ -z "$PROMPT" ]]; then
    PROMPT="$(cat)"
fi
if [[ -z "$PROMPT" ]]; then
    echo "local_model_agent: empty prompt" >&2
    exit 2
fi

# Resolve the model from the live server when it was not pinned. This is the
# "follows the model switch" property: the id is never hardcoded here.
if [[ -z "$MODEL" ]]; then
    MODEL="$(curl -sS -m 10 "$BASE_URL/v1/models" \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["data"][0]["id"])')"
fi
if [[ -z "$MODEL" ]]; then
    echo "local_model_agent: could not resolve a model id from $BASE_URL/v1/models" >&2
    exit 3
fi

# MAX_THINKING_TOKENS=0 is load-bearing: Claude Code requests an Anthropic
# `thinking` block by default, and the server refuses it unless the boot
# carries --reasoning-parser (serving.py rejects with "Anthropic thinking is
# not supported for models without a reasoning parser"). Zero suppresses the
# request entirely, so the agent works against either kind of boot.
#
# CLAUDE_CODE_MAX_OUTPUT_TOKENS is load-bearing too: the default completion
# request is 32000 tokens, which alone overruns a 32k-context boot once the
# ~20k-token system prompt is counted.
#
# ANTHROPIC_* are set on THIS PROCESS ONLY (env prefix, not settings.json), so
# the calling Claude Code session keeps talking to the Anthropic API.
exec timeout "$TIMEOUT" env \
    ANTHROPIC_BASE_URL="$BASE_URL" \
    ANTHROPIC_AUTH_TOKEN="${LOCAL_MODEL_AUTH_TOKEN:-local-no-auth}" \
    ANTHROPIC_API_KEY="${LOCAL_MODEL_AUTH_TOKEN:-local-no-auth}" \
    MAX_THINKING_TOKENS=0 \
    CLAUDE_CODE_MAX_OUTPUT_TOKENS="$MAX_TOKENS" \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    claude -p "$PROMPT" --model "$MODEL" --allowed-tools "$ALLOW_TOOLS"
