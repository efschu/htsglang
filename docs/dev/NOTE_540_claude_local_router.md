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

One header is NOT purely forwarded, and the reason is the byte-for-byte
response: `Accept-Encoding` is pinned to `identity` when the client omitted it.
aiohttp otherwise adds its own `gzip, deflate`, and since we do not decompress,
the proxy would hand a gzipped body to a client that never advertised gzip.
This was found by probing rather than by reading — a plain `curl` through the
first version got binary garbage where the error envelope should have been. A
client that DOES send `Accept-Encoding` has its value forwarded untouched and
gets the encoded bytes it asked for.

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

## The thinking ALIAS — asking a local agent to think (#541)

The shim's mirror image. Claude Code cannot ask a subagent for extended
thinking either: the frontmatter schema carries `model` and no thinking key, so
once again the model string is the only client-side lever. For every id in
`--local-model` the router therefore also answers to `<id>-think`:

| the client sends | routed to | `model` on the wire | `thinking` on the wire |
|---|---|---|---|
| `Qwen3.6-27B` | local | unchanged | `{"type":"disabled"}` (shim, only if absent) |
| `Qwen3.6-27B-think` | local | rewritten to `Qwen3.6-27B` | `{"type":"adaptive"}` (forced) |
| `claude-*-think` | upstream | unchanged | untouched |

Three properties worth stating because each one is a decision:

* **Adaptive, not enabled.** `{"type":"enabled"}` fails protocol validation
  without `budget_tokens >= 1024`; `adaptive` means "thinking on, the model
  decides how much", which is the arm an A/B against the no-thinking default
  actually wants.
* **Forced, not filled in.** Unlike the shim the alias OVERWRITES an explicit
  client value. Naming the alias is itself the request for thinking, and a
  client default winning here would silently turn the thinking arm back into
  the default arm with nothing in the log to say so.
* **Rewritten on every path.** The local server has no `-think` checkpoint, so
  `count_tokens` and friends get the id un-aliased too — but only
  `/v1/messages` gets the `thinking` field, exactly like the shim.

`/__router/stats` lists the aliases next to `local_models`, so the live
instance can be checked for the feature without reading its code.

The agent definition that uses it is `~/.claude/agents/local-model-think.md`
(`model: Qwen3.6-27B-think`), the thinking twin of `local-model`. It needs a
router that carries the alias; against an older router it fails with an
unknown-model error rather than silently falling back.

Requires a serving boot with a reasoning parser. The live INT8 boot has
`--reasoning-parser qwen3`, verified in `/get_server_info`; without one the
front answers 400 for any thinking-enabled request.

## Evidence

`test/registered/unit/entrypoints/anthropic/test_router.py` — 27 tests, fully
hermetic: two mock aiohttp backends stand in for api.anthropic.com and the
local front, each recording what it received. Pinned both ways: local model to
the local backend / any other model upstream, no-model bodies upstream, path and
query preserved, auth headers forwarded intact, credentials absent from the log
at DEBUG, shim applied / explicit value untouched / upstream body unmodified /
`count_tokens` untouched, SSE pass-through from both backends, backend error
status and body survived, unreachable backend gives an Anthropic-shaped 502,
and the `/__router/stats` counters separate the two paths. The alias adds
eight: it reaches the local backend, rewrites the model, forces adaptive,
overrides an explicit client value, un-aliases `count_tokens` without adding
`thinking`, survives `--no-thinking-shim`, counts as local in the stats, and a
`-think` suffix on an UNKNOWN model still goes upstream unmodified.

Mutation-checked rather than assumed green: removing the shim assignment fails
1 test, forcing `to_local = False` fails 10, dropping the alias's forced
`thinking` assignment fails 3, and dropping its model rewrite fails 2.

`/__router/stats` is also the live evidence instrument — it is what shows that
the parent kept talking to Anthropic (`upstream` counter climbing) while the
subagent ran locally (`local` counter climbing), alongside the
`sglang:generation_tokens_total` delta on 30030.

## Live acceptance, 2026-08-04

Not hermetic-only. A real `claude -p` (2.1.221) was driven through the router
against the live 30030 boot, in a separate process; the running session and the
server were untouched. The `--agents` payload was the recipe above, the task
was "read /tmp/acceptance_540.txt and report line 1 verbatim".

* The subagent returned the marker `HTSGLANG_ACCEPTANCE_LINE_ONE_MARKER_7Q4Z`,
  which it could only obtain through a `Read` tool round trip.
* `sglang:generation_tokens_total{priority="0"}` went 14820 -> 14926 on 30030
  (+106), so the local server really did the generating.
* The router's decision log is the split itself, in order: two parent turns on
  `claude-fable-5` -> upstream, then two turns on `Qwen3.6-27B` -> local (the
  request that returned `tool_use`, and the one carrying the `tool_result`),
  then the parent's closing turn -> upstream. Counters `local: 3, upstream: 5`
  including the two hand probes.

The same run also demonstrated WHY the shim exists, against a server that
predates the front fix. Identical body, no `thinking` field, sent directly to
30030 (bypassing the router): the entire 40-token budget went to a `thinking`
block, `stop_reason: max_tokens`, no text content at all. Through the router:
clean text. That is the agent-loop blocker in one pair of requests. It
disappears on its own once 30030 is restarted onto the front fix.

## Live acceptance of the alias, 2026-08-04 (#541)

Same discipline as above: real requests, live 30030 boot, nothing restarted on
the serving side. The boot carries `--reasoning-parser qwen3`
(`/get_server_info`), which is the precondition for any thinking arm.

Three bodies, identical apart from the field under test, straight at 30030:

| `thinking` sent | `stop_reason` | output tokens | blocks returned |
|---|---|---|---|
| field absent | `max_tokens` | 200 (budget) | `thinking` only, no text |
| `{"type":"disabled"}` | `end_turn` | 4 | `text` (`391`) |
| `{"type":"adaptive"}` | `end_turn` | 200 | `thinking` + `text` |

The first row is the pre-#540 front behaviour this router shims around, still
present on the live boot — so the shim is NOT yet a no-op here.

Then the same question through the router (30099), which is the alias proof:

* `model: Qwen3.6-27B` → 4 output tokens, blocks `['text']`
* `model: Qwen3.6-27B-think` → 300 output tokens, blocks `['thinking', …]`,
  and the local front saw `model: Qwen3.6-27B`

And inside the real agent harness (`claude -p --output-format stream-json`,
identical prompt and tools, model id the only difference): arm A returned 0
thinking tokens, arm B 63, both `rc=0`. Counted with the served checkpoint's
own tokenizer, not estimated.
