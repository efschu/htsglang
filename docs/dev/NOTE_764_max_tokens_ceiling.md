# NOTE 764 — `max_tokens` is a ceiling, not a demand

## The failure

Claude Code agents routed to the local alias (ANTHROPIC_BASE_URL → the split
router on 30099 → the Anthropic front on 30030, NOTE_540) died with:

```
API Error: 400 max_completion_tokens is too large: 32000.
This model supports at most 8192 completion tokens.
```

The second sentence is not a model property. `8192` was the value of
`--context-length` on the launch command of the serving process at that
moment; the checkpoint (Qwen3.8-27B-INT8-yarn1.5) is served with far more when
the flag is not set. The same server generated 10000 tokens through native
`/generate` the same day (NOTE_762), so this was never a capability limit.

## Why one agent worked and the next two did not

Nothing about the agents or the route differed. claude-cli sends the *same*
`max_tokens` (32000) on every turn of every request — it is a per-model-family
budget baked into the client, which has no way to learn a particular server's
context length. What changed was the server under them: port 30030 was being
rebooted repeatedly that evening by the #665-f1 experiment with different
`--context-length` values.

| time (2026-08-18) | serving `context_length` | agent outcome |
|---|---|---|
| ≤ 21:05 | 327680 | agent A ran to completion (~130 s, 3 tool uses) |
| 21:05 – 21:43 | 12288 | — |
| 22:16 onward | 327680 → 8192 | agent B died mid-run, agent C died on its first call |

The router access log carries exactly three local-route 400s that day
(316-byte Anthropic error envelope, distinct from the ~830-byte upstream
ones): 22:16:11 ×2 and 22:17:46 — inside that reboot window, and nowhere
earlier. Agent B's earlier turns had landed on the 327680 server, so it failed
"mid-run"; agent C started after the reboot and failed immediately.

So: a serving **configuration** change under running agents. Not per-turn
escalation by the harness, not a route difference.

## The fix

`max_tokens` is a stop condition — "generate at most N" — in both the
Anthropic and the OpenAI Messages/Chat semantics. A ceiling above what fits
does not make a request unsatisfiable; it makes it satisfiable with a lower
ceiling. Two gates rejected it instead:

* `python/sglang/srt/entrypoints/openai/serving_chat.py` — `max_tokens >
  context_length` → 400 (this produced the message above).
* `python/sglang/srt/managers/tokenizer_manager.py` — `input +
  max_new_tokens > context_len` → `ValueError`.

The second already knows how to shrink, but only under
`--allow-auto-truncate`, which also drops **input** tokens: a lossy, global
switch that is the wrong trade for this.

So the opt-in is per request and off by default:

```
ChatCompletionRequest.clamp_max_tokens  →  GenerateReqInput.clamp_max_new_tokens
```

The Anthropic front sets it on every request it converts, because that is
precisely Anthropic's documented `max_tokens` semantics. Gate 1 then stops
rejecting those; gate 2 performs the shrink, in the one place that knows the
real input length. Every other caller keeps the reject-by-default behaviour.

### What is deliberately *not* done

* The input is never truncated. An input that alone exceeds the context is
  still a hard error — clamping a ceiling is not a licence to drop prompt
  tokens.
* The shrink is not silent. When a lowered budget is actually reached, the
  completion ends with `finish_reason: "length"`, which the Anthropic front
  already reports as `stop_reason: "max_tokens"`, so the caller sees a
  truncated completion for what it is. One INFO line records the shrink
  server-side, naming both the requested and the fitted value.
* The requested value is forwarded verbatim; the front labels it a ceiling,
  it does not rewrite what the caller asked for.

## Scope of the relief

This removes the `max_tokens` rejection for good, at any context length. It
does **not** manufacture context: while 30030 runs with `--context-length
8192`, a full Claude Code agent prompt (system prompt plus tool definitions)
still exceeds 8192 on its own and is correctly rejected by the input-length
check with a different, accurate message. Agents are usable again on any boot
whose context length holds their prompt — which is every normal boot; the
8192 arm is an experiment configuration.

## Tests

`test/registered/unit/entrypoints/anthropic/test_max_tokens_clamp_764.py`
(13 cases, CPU-only). Red-first: 11 of the 13 fail on the pre-change tree;
the 2 that pass there assert the default path still rejects, which must not
change. Covers the front setting the flag, both gates with and without it,
the boundaries at exactly `context_length` and one over, the reserved-token
arithmetic, and that an over-long input still raises.
