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
# The agent definition binds the subagent's OWN inference loop to the local
# checkpoint, via the frontmatter `model:` field. An earlier version of this
# script asserted that this was impossible; that claim was REFUTED on the wire
# (2026-08-04, Claude Code 2.1.221): a subagent declaring a non-Claude model
# string sends that string in the `model` field of its own request bodies,
# while the parent session's requests in the same process still carry
# `claude-*`. A model-routing proxy at ANTHROPIC_BASE_URL then splits the two
# legs. See /root/claude-code-qwen-patch/README.md; the evidence command is
# /root/claude-code-qwen-patch/check_cli_version.sh.
#
# The proxy is a PRECONDITION, not an optimisation: without it, the subagent's
# requests go to api.anthropic.com carrying a model id Anthropic does not
# serve. This script probes for the proxy and says so in the generated file.
#
# scripts/dev/local_model_agent.sh remains as the no-proxy fallback: it starts
# a separate `claude` process whose whole environment points at the local
# server.

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
# The model-routing proxy that makes the frontmatter `model:` field work.
# Probed, never assumed: a missing proxy is the one failure mode that turns
# this agent from "runs locally" into "sends an unknown model id to Anthropic".
PROXY_URL="http://127.0.0.1:8787"
PROXY_DIR="/root/claude-code-qwen-patch"

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
        --proxy-url)     PROXY_URL="$2"; shift 2 ;;
        -h|--help)       sed -n '1,32p' "$0"; exit 0 ;;
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

# The routing proxy is what makes the frontmatter `model:` field reach the
# local server instead of api.anthropic.com. Probe it; never assume it. A
# session running WITHOUT the proxy sends the local model id to Anthropic and
# gets an error, which is a confusing failure to debug from inside an agent.
PROXY_STATE="absent"
PROXY_JSON="$(curl -sS -m 5 "$PROXY_URL/__ccqp/health" 2>/dev/null || true)"
if printf '%s' "$PROXY_JSON" | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'; then
    if printf '%s' "$PROXY_JSON" | grep -qF "\"$MODEL_ID\""; then
        PROXY_STATE="ready"
    else
        PROXY_STATE="running-without-this-model"
        {
            echo "register_local_model: WARNING -- proxy at $PROXY_URL is up but"
            echo "  does not list '$MODEL_ID' in local_models, so this agent's"
            echo "  requests would be forwarded to Anthropic. Set CCQP_LOCAL_MODELS"
            echo "  (or config.json) to '$MODEL_ID' and restart it."
        } >&2
    fi
else
    {
        echo "register_local_model: WARNING -- no routing proxy at $PROXY_URL."
        echo "  The agent file is still written, but the subagent's own inference"
        echo "  will NOT reach $BASE_URL until the proxy runs and the session is"
        echo "  started with ANTHROPIC_BASE_URL=$PROXY_URL."
        echo "  Start it with: $PROXY_DIR/ccqp.sh start"
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
description: Runs on the LOCAL htsglang server ($SHORT_NAME) instead of the Anthropic API -- this agent's OWN inference loop executes on the rig. Use for bulk/cheap/offline reasoning that should not leave the rig, for dogfooding the served checkpoint, and for any determined-answer probe against the live serving instance.
tools: Bash, Read, Grep, Glob
model: $MODEL_ID
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
| routing proxy | $PROXY_URL (\`$PROXY_STATE\`) |
${PARSER_WARNING:+
> **WARNING: $PARSER_WARNING.** This boot returns the chain-of-thought as raw
> text inside \`content\` and/or hands back tool calls as unparsed strings, so
> structured tool use will not work through this agent. Every response is
> still HTTP 200 -- the degradation is silent. Fix the boot recipe rather than
> working around it here.
}

## What you are

YOUR OWN inference runs on \`$MODEL_ID\` on this rig. Not a driver, not a
wrapper: the frontmatter \`model:\` field above puts that id into the \`model\`
field of every request your agent loop makes, and the routing proxy at
$PROXY_URL forwards exactly those requests to $BASE_URL while the parent
session's \`claude-*\` requests still go to Anthropic.

An earlier version of this file claimed this was impossible. That claim was
REFUTED on the wire on 2026-08-04 against Claude Code 2.1.221: two model ids
were observed leaving one process, \`$MODEL_ID\` from the subagent and
\`claude-*\` from the parent. No binary patch is involved. The mechanism, the
proxy and the evidence command live in $PROXY_DIR/README.md.

PRECONDITION: the session must run with \`ANTHROPIC_BASE_URL=$PROXY_URL\` and
that proxy must list \`$MODEL_ID\` in its local model map. This file records the
proxy state at generation time as \`$PROXY_STATE\` in the table above. If it
says anything other than \`ready\`, or if your first request fails with an
unknown-model error from Anthropic, the proxy is the thing to fix:

\`\`\`bash
$PROXY_DIR/ccqp.sh start && $PROXY_DIR/ccqp.sh health
\`\`\`

Rules for your own work:
- Report what you actually produced. A success message is not evidence.
- Never present an answer as coming from the local model unless it did.
- If the endpoint fails, report the exact HTTP status. Do not silently
  substitute reasoning from another model.
- You are a 27B-class local model: take bounded, well-specified tasks. Long
  agentic chains and image work are not yours (the checkpoint is text-only;
  image content blocks return HTTP 500).

## Driving the local model from ANOTHER agent

Any agent -- including one running on Anthropic -- can reach this checkpoint
without the proxy, in a separate process with a process-scoped environment:

\`\`\`bash
$REPO_ROOT/scripts/dev/local_model_agent.sh -- "<your prompt>"
\`\`\`

Options: \`-b BASE_URL\`, \`-m MODEL\`, \`-t MAX_OUTPUT_TOKENS\`, \`-T TIMEOUT_SECONDS\`,
\`--allow-tools "Read,Grep"\` (default: no tools, pure reasoning). Always pass a
bounded \`-T\`; never wait unbounded on the rig.

For a raw single-shot completion without an agent loop, call the endpoint
directly:

\`\`\`bash
curl -sS $BASE_URL/v1/messages -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' -d '{"model":"$MODEL_ID","max_tokens":256,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"..."}]}'
\`\`\`

\`thinking:{"type":"disabled"}\` is not optional on the current build: without
it the \`$REASONING\` reasoning parser emits a thinking block first and can spend
the whole \`max_tokens\` budget on it.
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
