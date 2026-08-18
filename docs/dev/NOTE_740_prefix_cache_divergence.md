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

## 4a. CORRECTION to section 4, from the #745 gate

Section 4 said the mechanism was "no resident state at that position". **That
is wrong as stated, and it is my claim to retract.**

`MambaRadixCache` already implements resume-from-nearest-checkpoint:
`_match_prefix_helper` tracks the deepest node on the matched path that still
carries a state (`mamba_radix_cache.py:1513-1517`), and
`_match_post_processor` truncates the returned KV prefix to exactly that point
(`:1685`, `value = value[:best_value_len]`) and COWs the state into a fresh
slot (`:1611-1654`). The tail is then re-prefilled. So a prefix hit is NOT
discarded for lack of a state at the exact position — it falls back to the
nearest checkpoint at-or-below and replays the remainder.

A full re-prefill therefore requires that NO node on the matched path carries a
state at all (the root never does). The corrected mechanism is **checkpoint
EVICTION, not checkpoint absence**: the mamba pool holds
`max_mamba_cache_size = 12` slots total, shared between the running requests
and every retained tree checkpoint, under LRU (`evict_mamba`, `:1090`). With
`max_running_requests = 4` and many concurrent agent sessions, the retained
checkpoints are evicted long before the next turn of the same session arrives.

This does not change section 4's conclusion — it is still serving-side, still
hybrid-specific, and still not the adapter — but it names the cause more
precisely, and it changes the fix: the states need somewhere to LIVE beyond 12
VRAM slots, which is exactly what #745 proposes.

## 5. Honest scope

* The system block and tool schemas were STUBBED with a fixed string, identical
  in both requests. This test therefore proves turn-to-turn stability within one
  agent; it does NOT test the cross-agent scaffold-sharing question. That second
  diff is CLOSED in §5a below.
* The transcript's stored `thinking` blocks are empty, so deterministic thinking
  text was injected to exercise the rewrap path. Without that, the rewrap
  correctly returns `None` and the test would have silently measured nothing —
  which is exactly what it did on the first attempt.

## 5a. The cross-agent scaffold diff — measured (the #740 residual)

Method, same standard as §1 but with REAL scaffolds instead of stubs. Two
fresh same-type agent sessions (`claude -p "hi" --model sonnet`, identical
invocations, same directory) were pointed at a local capture sink via a
`--settings` env override — necessary because `settings.json` pins every
session's `ANTHROPIC_BASE_URL` to the router, so a plain env var never reaches
the request path. The sink recorded the CLIENT-BUILT first-request bodies
byte-for-byte: 3 system blocks (27,578 chars), 25 tool schemas (102,607 chars
of JSON), 2 messages. Both bodies were then rendered through the real chain of
the serving lineage the 30030 unit's drop-in points at (`wt-merge-r4`):
`AnthropicServing._convert_to_chat_completion_request` with
`_merge_inline_system` derived by the shipped `detect_inline_system_support`
against the live template (=True), the REAL `apply_reasoning_enabled` bound
over the live boot's `--reasoning-parser qwen3` with a real `ReasoningParser`
detector, serving's `normalize_assistant_tool_call_arguments`, the live
model's own `chat_template.jinja` (`Qwen3.8-27B-INT8-yarn1.5`), tokenized.

Raw-body layer: **everything prompt-relevant is byte-identical across the two
sessions.** System blocks identical (including `cache_control`), tools
identical WITH stable ordering, even the messages identical (same prompt, same
context attachment). The only differing field is `metadata.user_id` (embeds
the session id), which §3 already showed never reaches the prompt. The only
launch-variable element inside the system text is the trailing
`<total_tokens>` budget (char 27,532 of 27,580 — the last line).

Rendered layer (33,703-token first request):

| pair                                        | common token prefix |
| ---                                         | --- |
| identical invocations (A vs B)              | **33,703 (100 %, byte-identical render)** |
| same dir, tasks differ                      | 33,695 |
| same dir, per-agent context AND task differ | 33,552 |
| different working directory                 | **30,960 (91.9 %)** |

The different-cwd row is the load-bearing one, and it survives for a reason
worth recording: the live template renders the TOOL SCHEMAS FIRST — `# Tools`
begins at ~token 41 — and the environment facts (working directory, git-repo
flag) sit near the END of the system prose, AFTER all ~20k tool-schema
tokens. So even two same-type agents in different worktrees share a ~31k-token
identical prefix through the render chain. Tool-schema ordering, the suspect
the residual named, is stable and contributes zero divergence.

**Verdict line for #743: the scaffold prefix EXCEEDS the mamba-anchor reach
requirement — cross-agent hits should occur, and their absence is an
anchor-survival fact, not an adapter fact.** Concretely: with the live
`chunked_prefill_size = 512`, chunk-end checkpoint donations can sit as deep
as token 30,720 inside the 30,960-token shared prefix (60 grid positions), so
ONE surviving anchor anywhere on that path turns a same-type agent's first
request into a prefix hit with a bounded tail re-prefill. 4 hits in 121
requests against a ~31k-token shared prefix therefore points at §4a's
eviction mechanism (12 LRU slots shared by all requests and checkpoints) plus
the split-node rule (`_split_node`: "mamba cache can not be split", the new
branch-point node carries `mamba_value = None`) — the #743/#745 world. Not
the adapter, not the template, not tool ordering.

Honest limits of this measurement:

* The captured pair is claude-model (sonnet) main-type sessions; a qwen-lane
  pair could not be captured (serving down; spawning local-model agents is
  barred by the sub-agent policy). The finding transfers structurally — the
  same client code builds system/tools for every agent type, and the chain
  under test is the one local-model requests traverse — but the qwen-lane
  scaffold is smaller (4-tool set), so its absolute prefix length is shorter
  than 30,960. The stability finding (byte-identical construction, stable
  tool order, env facts after the tools) is type-independent.
* Renders used `claude-sonnet-5` as the model string; the router would
  rewrite it for a local target (§3: `payload["model"]`), which does not
  enter the rendered prompt.
* The `<total_tokens>` budget line diverges at token 31,354 (measured by
  mutating the budget value and re-rendering). It sits AFTER the env facts in
  render order, so for a different-cwd pair it is beyond the 30,960 boundary
  and irrelevant; for a same-dir pair with differing consumed budgets it caps
  the shared prefix at 31,354 instead of 33,552. Either way the shared prefix
  stays ~31k tokens and the verdict is unchanged.
