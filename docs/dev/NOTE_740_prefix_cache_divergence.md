# NOTE 740: where the prefix-cache misses come from — measured, not argued

**Verdict: the adapter and the chat template are NOT inserting divergence, and
neither is the router. Through the real chain, turn N is a 99.7 % token prefix
of turn N+1, and the only divergent tokens are the generation prompt that turn
N+1 legitimately replaces.** The remaining mechanism is serving-side, and it is
specific to this model being a HYBRID: prefix reuse needs a resident recurrent
state, not only KV.

The premise of the ticket is sound — 4 hits in 121 requests cannot be explained
by "genuinely divergent agent prompts", and it was right to treat it as a defect
until proven otherwise. It just is not an adapter defect.

## 1. Method

Fully desk. No serving request was issued; the two live queries used were
`/get_model_info` and `/get_server_info`, both read-only.

1. Streamed a session transcript (never `grep -r` over transcript trees; RAM
   discipline per the operator rule) and rebuilt the Anthropic message array,
   MERGING consecutive assistant entries — a transcript stores `thinking`,
   `text` and `tool_use` as separate rows, while a real request carries them as
   one assistant message with three content blocks.
2. Built request N (ending at an assistant `tool_use`) and request N+1 (the same
   plus its `tool_result` and the next assistant turn).
3. Converted both through the SHIPPED
   `AnthropicServing._convert_to_chat_completion_request`
   (`entrypoints/anthropic/serving.py:315`), driven unbound over a carrier, with
   the real `wrap_reasoning_history` and a real `Qwen3Detector`.
4. Applied serving's own pre-template normalization —
   `msg.model_dump()` then `normalize_assistant_tool_call_arguments`
   (`entrypoints/openai/serving_chat.py:772-774`).
5. Rendered with the LIVE model's own template
   (`Qwen3.8-27B-INT8-yarn1.5`), tokenized, and took the first divergent index.

## 2. Result

| pair | mode | turn N | turn N+1 | common prefix |
| --- | --- | --- | --- | --- |
| N -> N+1 | template default | 1180 | 1450 | **1176 (99.7 %)** |
| N -> N+1 | `preserve_thinking=True` (as live) | 1180 | 1450 | **1176 (99.7 %)** |
| N+1 -> N+2 | template default | 1450 | 2084 | **1446 (99.7 %)** |
| N+1 -> N+2 | `preserve_thinking=True` | 1450 | 2084 | **1446 (99.7 %)** |

The first divergent token is the generation prompt:

```
N    ... </tool_call><|im_end|>\n<|im_start|>assistant\n<think>\n
N+1  ... </tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{...
```

Those four tokens are `add_generation_prompt`. The cache stores turn N's prompt
PLUS the model's continuation, and turn N+1 re-sends that continuation as
history — so the reusable prefix is everything before the generation prompt, and
it matches exactly. **There is no adapter-injected divergence at position 0, at
the system block, or at the first assistant turn.**

`preserve_thinking` made NO difference, and the reason is worth recording: the
Anthropic front does not rely on the template to preserve history thinking. It
re-wraps prior-turn thinking into the detector's own tokens and appends it as an
ordinary text part (`serving.py:544-546`, via `wrap_reasoning_history` at
`serving_chat.py:1722`), so by the time the template runs, the thinking is
literal assistant content that no template arm can strip. NOTE_542 §4.3's
first-assistant-turn breakage does not apply to this front.

## 3. The router is clean

Live unit `claude-local-router` runs
`sglang.srt.entrypoints.anthropic.router` from `/spinning/wt-anthropic-front`
(NOT this worktree — the running code is a different lineage, and reading the
wrong copy would have made this whole check meaningless).

It mutates exactly two payload fields: `payload["model"] = target_model`
(`router.py:412`) and `payload["output_config"]` (`router.py:348`). Neither
appears in the rendered prompt. `messages` and `system` are read only, for
`count_tokens`. Effort normalization is a pure function of the client's own
value, so it cannot vary between two turns of one agent. The body is
re-serialized with `json.dumps`, which preserves key order from `json.loads`.

No timestamp, request id, or effort echo reaches the prompt. Suspect (1) is
eliminated on the code, and suspects (2) and (3) on the measurement above.

## 4. What is left, and why it is not mine to fix

The served model is a hybrid (GDN + full attention), and the live boot reports:

```
disable_radix_cache      = False        # caching IS enabled
uses_mamba_radix_cache   = True
mamba_radix_cache_strategy = 'no_buffer'
max_mamba_cache_size     = 12
max_running_requests     = 4
```

A KV prefix can be reused by slicing. A recurrent state cannot: reuse requires a
CHECKPOINTED mamba state at that exact prefix position. With 12 mamba slots, the
number of distinct prefixes that can stay reusable is small and bounded by
eviction, so a long agent tool loop can render a perfect 99.7 % prefix and still
find no state to resume from.

**This is a hypothesis with a mechanism, not a measured cause.** It is
consistent with every number above, but I have not shown that the misses are
mamba-slot evictions. Deciding it needs `mem_cache` ownership:
`mem_cache/mamba_radix_cache.py` (~1470 lines, with `no_buffer` handling at
`:793` and `:1468`) is the prior art — this is a tuning/behaviour question
inside existing machinery, not a missing feature.

Suggested next probe, cheap and read-only: correlate hit/miss against
`max_mamba_cache_size` by raising it on a TEST boot and re-measuring the same
121-request shape. If hits track the slot count, the cause is established.

## 5. Honest scope

* The system block and tool schemas were STUBBED with a fixed string, identical
  in both requests. This test therefore proves turn-to-turn stability within one
  agent; it does NOT test the cross-agent scaffold-sharing question. That second
  diff is still open.
* The transcript's stored `thinking` blocks are empty, so deterministic thinking
  text was injected to exercise the rewrap path. Without that, the rewrap
  correctly returns `None` and the test would have silently measured nothing —
  which is exactly what it did on the first attempt.
