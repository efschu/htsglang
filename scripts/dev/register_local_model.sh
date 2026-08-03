#!/usr/bin/env bash
# Re-generate the "local model" Claude Code agent entry from the RUNNING
# htsglang server. This is the mechanism that makes the entry follow a model
# switch: nothing about the checkpoint is hardcoded -- the id, the context
# length and the residency come out of GET /v1/models on the live server.
#
# Run it after every serving switch (INT8 -> NVFP4 -> GGUF -> whatever):
#
#   scripts/dev/register_local_model.sh                     # default endpoint
#   scripts/dev/register_local_model.sh -b http://127.0.0.1:30030
#   scripts/dev/register_local_model.sh --agent-dir /path/to/.claude/agents
#
# Writes:
#   <agent-dir>/local-model.md                      the agent definition
#   ~/.config/htsglang/local_model_agent.env        defaults for the wrapper
#
# The agent definition is a NAMED FALLBACK, and the file says so in its own
# body: Claude Code cannot bind a subagent's own inference to a foreign
# endpoint (no baseUrl/provider/env key in the subagent frontmatter schema;
# every ANTHROPIC_*_BASE_URL is process-global). The agent therefore drives
# the local model through scripts/dev/local_model_agent.sh, which starts a
# separate `claude` process with a process-scoped environment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_URL="http://127.0.0.1:30030"
# USER-GLOBAL, and deliberately NOT repo-relative (#531 follow-up, user-caught).
# The first version defaulted to "$REPO_ROOT/.claude/agents", which for a
# WORKTREE checkout is a directory no Claude Code session ever reads: project
# agents are loaded from the SESSION's own project directory, so a file written
# under /spinning/wt-<something>/ registers nothing. The agent type simply never
# appeared in any agent list, while the script printed "wrote ..." and the
# wrapper round-trip still passed -- the wrapper reads $CONF_DIR, not the agent
# file, so it could never have caught this. Hence the canonical user-global path
# plus the read-path verification at the end of this script.
AGENT_DIR="${HOME:-/root}/.claude/agents"
CONF_DIR="${HOME:-/root}/.config/htsglang"
MAX_TOKENS=4096
ALLOW_UNREAD_AGENT_DIR=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--base-url)   BASE_URL="$2"; shift 2 ;;
        # Override only for a test harness or a genuinely different project
        # dir. Anything that is not a path Claude Code READS is warned about
        # below rather than silently accepted.
        --agent-dir)     AGENT_DIR="$2"; shift 2 ;;
        --conf-dir)      CONF_DIR="$2"; shift 2 ;;
        -t|--max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --allow-unread-agent-dir) ALLOW_UNREAD_AGENT_DIR=1; shift ;;
        -h|--help)       sed -n '1,25p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

echo "register_local_model: querying $BASE_URL/v1/models"
MODELS_JSON="$(curl -sS -m 15 "$BASE_URL/v1/models")"

read -r MODEL_ID CTX_LEN RESIDENCY < <(
    printf '%s' "$MODELS_JSON" | python3 -c '
import json, sys
d = json.load(sys.stdin)
entries = d.get("data") or []
if not entries:
    sys.exit("register_local_model: server reported no models")
m = entries[0]
x = m.get("x-htsglang") or {}
print(m["id"], m.get("max_model_len", "unknown"), x.get("residency", "unknown"))
'
)

# The Anthropic Messages front is what a Claude Code process needs. Probe it
# rather than assume it: a boot that does not expose it cannot back an agent,
# and finding that out here beats finding it out inside an agent run.
# Retried with a generous timeout: a healthy server that is mid-prefill on a
# long request answers this LATE, not never (observed at a 55k-token prefill,
# ~40 s of chunked prefill ahead of the probe in the queue). A single short
# probe turned that into a refusal to update the config, i.e. a transient load
# spike looked identical to a missing endpoint. Distinguish the two by asking
# /health when the probe fails.
MSG_CODE=000
for attempt in 1 2 3; do
    MSG_CODE="$(curl -sS -m 60 -o /dev/null -w '%{http_code}' \
        -X POST "$BASE_URL/v1/messages" \
        -H 'content-type: application/json' \
        -H 'anthropic-version: 2023-06-01' \
        -d '{"model":"probe","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}' \
        2>/dev/null || echo 000)"
    [ "$MSG_CODE" = "200" ] && break
    echo "register_local_model: /v1/messages probe attempt $attempt -> $MSG_CODE" >&2
    sleep 5
done
if [[ "$MSG_CODE" != "200" ]]; then
    HEALTH_CODE="$(curl -sS -m 10 -o /dev/null -w '%{http_code}' \
        "$BASE_URL/health" 2>/dev/null || echo 000)"
    echo "register_local_model: POST /v1/messages returned HTTP $MSG_CODE" >&2
    if [ "$HEALTH_CODE" = "200" ]; then
        echo "  /health answers 200, so the SERVER is alive: this is either a" >&2
        echo "  boot without the Anthropic front, or a server too loaded to" >&2
        echo "  answer within 3x60 s. Re-run when the queue drains; refusing" >&2
        echo "  rather than recording an unverified endpoint." >&2
    else
        echo "  /health also failed ($HEALTH_CODE) -- the endpoint is down." >&2
    fi
    echo "  The local-model agent needs the Anthropic Messages front; refusing" >&2
    echo "  to write a config that points at an endpoint which cannot serve it." >&2
    exit 4
fi

# #531: read the ACTUAL parser settings off the running boot instead of
# guessing from the model listing. /server_info returns the live ServerArgs,
# so reasoning_parser / tool_call_parser are authoritative here.
#
# Why this is worth a warning rather than a note: a boot without
# --reasoning-parser leaks raw </think> markers into the answer text and
# refuses Anthropic `thinking` blocks outright; a boot without
# --tool-call-parser returns a tool call as a JSON-looking STRING inside
# `content` instead of a structured `tool_calls` entry. Both degrade an
# agentic client while every response is still HTTP 200 -- observed on this
# rig's own FP8 boot, which reported both as None.
SERVER_INFO="$(curl -sS -m 15 "$BASE_URL/server_info" 2>/dev/null || echo '{}')"
read -r REASONING TOOLCALL < <(
    printf '%s' "$SERVER_INFO" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    d = {}
print(d.get("reasoning_parser") or "NONE", d.get("tool_call_parser") or "NONE")
' 2>/dev/null || echo "UNKNOWN UNKNOWN"
)
[ -z "${REASONING:-}" ] && REASONING=UNKNOWN
[ -z "${TOOLCALL:-}" ] && TOOLCALL=UNKNOWN

MISSING=""
[ "$REASONING" = "NONE" ] && MISSING="reasoning parser"
if [ "$TOOLCALL" = "NONE" ]; then
    if [ -n "$MISSING" ]; then MISSING="$MISSING and tool-call parser"
    else MISSING="tool-call parser"; fi
fi
PARSER_WARNING=""
if [ -n "$MISSING" ]; then
    PARSER_WARNING="current boot lacks ${MISSING}; agentic tool use degraded"
    {
        echo "register_local_model: WARNING -- $PARSER_WARNING"
        echo "  reasoning_parser=$REASONING tool_call_parser=$TOOLCALL at $BASE_URL"
        echo "  Re-boot with the parsers for this model family (runbook 4.1;"
        echo "  Qwen3.x = qwen3 / qwen3_coder, DeepSeek-V4 = deepseek-v4 / deepseekv4)."
    } >&2
fi

mkdir -p "$AGENT_DIR" "$CONF_DIR"

cat > "$CONF_DIR/local_model_agent.env" <<EOF
# Generated by scripts/dev/register_local_model.sh -- do not edit by hand.
# Regenerate after every serving switch; nothing here is a hand-picked value.
# Source server: $BASE_URL
# Generated:     $(date -u +%Y-%m-%dT%H:%M:%SZ)
LOCAL_MODEL_BASE_URL=$BASE_URL
LOCAL_MODEL_ID=$MODEL_ID
LOCAL_MODEL_MAX_OUTPUT_TOKENS=$MAX_TOKENS
LOCAL_MODEL_CONTEXT_LENGTH=$CTX_LEN
LOCAL_MODEL_AUTH_TOKEN=local-no-auth
LOCAL_MODEL_TIMEOUT=600
LOCAL_MODEL_ALLOWED_TOOLS=
EOF

SHORT_NAME="$(basename "$MODEL_ID")"

cat > "$AGENT_DIR/local-model.md" <<EOF
---
name: local-model
description: Runs a prompt on the LOCAL htsglang server ($SHORT_NAME) instead of the Anthropic API. Use for bulk/cheap/offline reasoning that should not leave the rig, for dogfooding the served checkpoint, and for any determined-answer probe against the live serving instance.
tools: Bash, Read, Grep, Glob
model: haiku
---

Generated by \`scripts/dev/register_local_model.sh\` from the live server at
$BASE_URL on $(date -u +%Y-%m-%dT%H:%M:%SZ). Do not hand-edit: re-run the
script after a serving switch and this file follows the new checkpoint.

Currently served:

| field | value |
|---|---|
| model id | \`$MODEL_ID\` |
| endpoint | $BASE_URL |
| max_model_len | $CTX_LEN |
| residency | $RESIDENCY |
| Anthropic Messages front | present (HTTP $MSG_CODE on POST /v1/messages) |
| reasoning parser | $REASONING |
| tool-call parser | $TOOLCALL |
${PARSER_WARNING:+
> **WARNING: $PARSER_WARNING.** This boot returns the chain-of-thought as raw
> text inside \`content\` and/or hands back tool calls as unparsed strings, so
> structured tool use will not work through this agent. Every response is
> still HTTP 200 -- the degradation is silent. Fix the boot recipe rather than
> working around it here.
}

## What you are

You are a THIN DRIVER. Your own inference does not run on the local model and
cannot be made to: Claude Code has no per-subagent endpoint binding. The
subagent frontmatter schema is closed
(\`name/description/tools/disallowedTools/model/permissionMode/maxTurns/skills/
mcpServers/hooks/memory/background/effort/isolation/color/initialPrompt\`) and
carries no \`baseUrl\`/\`provider\`/\`env\` key; \`CLAUDE_CODE_SUBAGENT_MODEL\` only
remaps the model NAME on the endpoint the session already uses; and every
\`ANTHROPIC_*_BASE_URL\` is a process-global environment variable, so setting
one in \`settings.json\` would reroute the PARENT session too. This agent is the
named fallback for that gap, not a workaround pretending to be the feature.

The real local-model inference happens in a SEPARATE \`claude\` process that
this agent starts, with a process-scoped environment. That process is a full
Claude Code agent loop -- it is not a single completion call.

## How to run the local model

\`\`\`bash
$REPO_ROOT/scripts/dev/local_model_agent.sh -- "<your prompt>"
\`\`\`

Options: \`-b BASE_URL\`, \`-m MODEL\`, \`-t MAX_OUTPUT_TOKENS\`, \`-T TIMEOUT_SECONDS\`,
\`--allow-tools "Read,Grep"\` (default: no tools, pure reasoning).

Rules:
- Always pass a bounded \`-T\`; never wait unbounded on the rig.
- Quote the local model's answer VERBATIM in your report and name the model id
  you got it from. A success message is not evidence -- the answer text is.
- If the wrapper fails, report the exact HTTP status/stderr. Do not fall back
  to answering from your own model and present it as the local model's answer.
- For a raw single-shot completion without an agent loop, call the endpoint
  directly instead:
  \`curl -sS $BASE_URL/v1/messages -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' -d '{"model":"$MODEL_ID","max_tokens":256,"messages":[{"role":"user","content":"..."}]}'\`
EOF

echo "register_local_model: wrote $CONF_DIR/local_model_agent.env"
echo "register_local_model: model=$MODEL_ID ctx=$CTX_LEN residency=$RESIDENCY reasoning=$REASONING toolcall=$TOOLCALL"

# ---------------------------------------------------------------------------
# Closing probe: "wrote the file" is not "registered the agent". Verify the
# file EXISTS at the path that will actually be READ, is non-empty, parses as
# frontmatter with the expected `name:`, and name that absolute path in the
# output -- the #531 follow-up defect was invisible precisely because the
# script reported a write to a path nobody loads.
# ---------------------------------------------------------------------------
AGENT_FILE="$AGENT_DIR/local-model.md"
USER_AGENT_DIR="${HOME:-/root}/.claude/agents"

if [ ! -s "$AGENT_FILE" ]; then
    echo "register_local_model: FAILED -- $AGENT_FILE is missing or empty" >&2
    exit 5
fi
AGENT_NAME="$(sed -n 's/^name:[[:space:]]*//p' "$AGENT_FILE" | head -1)"
if [ "$AGENT_NAME" != "local-model" ]; then
    echo "register_local_model: FAILED -- $AGENT_FILE has no usable 'name:'" >&2
    echo "  frontmatter name read as: ${AGENT_NAME:-<empty>}" >&2
    exit 6
fi

echo "register_local_model: VERIFIED agent '$AGENT_NAME' at $AGENT_FILE ($(wc -c < "$AGENT_FILE") bytes)"

if [ "$AGENT_DIR" != "$USER_AGENT_DIR" ] && [ "$AGENT_DIR" != "$PWD/.claude/agents" ]; then
    {
        echo "register_local_model: UNREADABLE TARGET -- $AGENT_DIR is neither"
        echo "  the user-global agent dir ($USER_AGENT_DIR) nor the current"
        echo "  project's ($PWD/.claude/agents). Claude Code will NOT load an"
        echo "  agent from there, so this run registered nothing usable."
        echo "  This is exactly the #531 defect; re-run without --agent-dir."
        echo "  (Pass --allow-unread-agent-dir if you are a test harness and"
        echo "  meant to write somewhere inert.)"
    } >&2
    # Non-zero on purpose: the failure this guards against is a SILENT no-op,
    # so a caller that does not read stderr must still be able to detect it.
    [ "$ALLOW_UNREAD_AGENT_DIR" = "1" ] || exit 7
fi

echo "register_local_model: NOTE -- a session loads its agent list at START."
echo "  The 'local-model' type appears in NEW sessions; the current one keeps"
echo "  the list it booted with. Existing sessions can still drive the model"
echo "  directly via scripts/dev/local_model_agent.sh."
