# NOTE #540 — routing ONE Claude Code subagent to the local server

## The problem this solves

Claude Code binds its endpoint per PROCESS. `ANTHROPIC_BASE_URL` and every
sibling (`ANTHROPIC_BEDROCK_BASE_URL`, `ANTHROPIC_VERTEX_BASE_URL`, …) are read
once at startup, and the subagent frontmatter schema carries no `baseUrl` /
`provider` / `env` key. `CLAUDE_CODE_SUBAGENT_MODEL` remaps only the model NAME
on the already-configured endpoint. So there is no in-client way to keep a
session's parent turns on api.anthropic.com while one subagent runs on the rig.

`scripts/dev/local_model_agent.sh` answers this by starting a SEPARATE `claude`
process pointed entirely at the local server — a fallback, not the feature. The
router is the feature: a proxy in front of both endpoints that decides per
request, so a single process can be mixed.

The lever that makes it work: Claude Code passes the `"model"` string from
`--agents` to the wire UNVALIDATED. Naming a local model id in an agent
definition is enough to make that agent's requests identifiable at the proxy.

## Start the router

```
scripts/dev/claude_local_router.sh                 # resolves the model id live
scripts/dev/claude_local_router.sh -m Qwen3.6-27B  # or pin it
```

Defaults: listens on `127.0.0.1:30099`, local base `http://127.0.0.1:30030`,
upstream `https://api.anthropic.com`. With no `-m` the id comes from
`GET /v1/models` on the local server, so the router follows a checkpoint switch
instead of carrying a hardcoded name. Implementation:
`python/sglang/srt/entrypoints/anthropic/router.py` (also runnable directly as
`python -m sglang.srt.entrypoints.anthropic.router`).

Requests are forwarded verbatim — path, query, method, every header including
`Authorization` / `x-api-key` / `anthropic-version` / `anthropic-beta`, and the
response body byte-for-byte (`auto_decompress=False`), streaming included. No
header value is ever logged, at any level; the log line carries the method,
path, decision and model id only.

## The `--agents` recipe

In a SEPARATE process from any running session:

```
ANTHROPIC_BASE_URL=http://127.0.0.1:30099 \
claude -p "read /etc/hostname and tell me line 1" \
  --agents '{"local-model": {
      "description": "Runs on the local htsglang server. Use for local file work.",
      "prompt": "You are a local worker. Use the tools you are given.",
      "tools": ["Read", "Grep", "Glob"],
      "model": "Qwen3.6-27B"}}'
```

The parent turns carry the session's real model name and go to Anthropic; only
the `local-model` subagent's requests carry `"model":"Qwen3.6-27B"` and land on
30030. `CLAUDE_CODE_MAX_OUTPUT_TOKENS` is still load-bearing for the local side
on a 32k-context boot (the default 32000-token completion request plus the ~20k
system prompt overruns it) — set it on the process when the local agent's
context is tight.

## The thinking shim, and when it stops mattering

The router fills in `"thinking":{"type":"disabled"}` on locally-routed
`/v1/messages` requests that omit the field. It never rewrites an explicit
value, never touches upstream traffic, and never touches `count_tokens`.

This is a COMPATIBILITY SHIM for a server that predates the #540 front fix.
Pre-#540, a boot with `--reasoning-parser` answered a request without a
`thinking` field with a leading thinking block that consumed the whole
`max_tokens` budget, so an agent loop on a tight budget never reached its
`tool_use`. #540 fixes that in the front itself (absent == disabled,
`anthropic/serving.py:694-719`). Once the serving process carries that commit
the shim is a NO-OP by construction: it only ever writes the value the front
would have defaulted to anyway. `--no-thinking-shim` turns it off; there is no
reason to before the live server is restarted onto #540.

## Evidence

`test/registered/unit/entrypoints/anthropic/test_router.py` — 17 tests, fully
hermetic: two mock aiohttp backends stand in for api.anthropic.com and the
local front, each recording what it received. Pinned both ways: local model to
the local backend / any other model upstream, no-model bodies upstream, path and
query preserved, auth headers forwarded intact, credentials absent from the log
at DEBUG, shim applied / explicit value untouched / upstream body unmodified /
`count_tokens` untouched, SSE pass-through from both backends, backend error
status and body survived, unreachable backend gives an Anthropic-shaped 502,
and the `/__router/stats` counters separate the two paths.

Mutation-checked rather than assumed green: removing the shim assignment fails
1 test, forcing `to_local = False` fails 10.

`/__router/stats` is also the live evidence instrument — it is what shows that
the parent kept talking to Anthropic (`upstream` counter climbing) while the
subagent ran locally (`local` counter climbing), alongside the
`sglang:generation_tokens_total` delta on 30030.
