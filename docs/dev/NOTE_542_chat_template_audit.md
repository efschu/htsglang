# NOTE 542: Qwen3.6 chat template audit (original vs community "fixed" template)

Question: is the original Qwen3.6 chat template defective for our usage, and
should the community-maintained replacement be adopted? Two in-flight efforts
depend on the answer, because both make assumptions about think-token
boundaries: #540 (per-request thinking budget) and #541 (thinking-off vs
thinking-adaptive A/B).

Answer up front: **do not adopt the community template.** The original template
does contain real defects, but every defect that our stack can actually reach is
either already neutralised in our own front or is not fixed by the community
template either. The community template additionally introduces prompt drift,
content injection, and a silent thinking-off override that would invalidate #540
and #541. The one verified, actionable win is unrelated to swapping templates:
the `preserve_thinking` kwarg that the **original** template already exposes.

## 1. Effective template

| Item | Value |
|---|---|
| Live server | `sglang.launch_server --model-path .../Qwen3.6-27B-INT8-W8A8 ... --port 30030` |
| `--chat-template` override | none — the model-dir template is effective |
| `.../Qwen3.6-27B-INT8-W8A8/chat_template.jinja` | `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` |
| `.../Qwen3.6-27B-AWQ-BF16-INT4/chat_template.jinja` | same sha256 (byte-identical) |
| `Qwen/Qwen3.6-27B` HF `chat_template.jinja` (main) | same sha256 |

The INT8 dir carries no `chat_template` key in `tokenizer_config.json`; the
`.jinja` file is the only source. The AWQ dir carries both and they agree. Both
local copies are the current upstream original — we are not running a stale or
locally patched template.

## 2. Candidates examined

| Candidate | Source | sha256 (first 16) | Size |
|---|---|---|---|
| original | https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/chat_template.jinja | `e84f32a23fdda276` | 7764 B |
| froggeric v21.3 | https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates | `d203f3342d8a7f84` | 16289 B |
| unsloth | https://huggingface.co/unsloth/Qwen3.6-27B/blob/main/chat_template.jinja | `55d4931433fe502b` | 8057 B |
| allanchan339 | https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix | `5d7b7fbc6ec106b8` | 11084 B |

Upstream issue tracker for the original template:

* https://huggingface.co/Qwen/Qwen3.6-27B/discussions/20 — "Update
  chat_template.jinja" (open) — `continue_final_message` / assistant-prefill.
* https://huggingface.co/Qwen/Qwen3.6-27B/discussions/16 — "Qwen3.6 27b has
  template issue in ralph loop on opencode: `No user query found in messages`"
  (open) — the same defect reproduced below as F1.

froggeric v21.3 is the template the request referred to. It targets Qwen 3.5 and
3.6 jointly in a single unified file, so it is not mis-targeted at another
variant — but it is a rewrite (16.3 kB vs 7.8 kB), not a patch.

## 3. Method

Hermetic rendering, no GPU and no contact with the live server:
`tokenizer.apply_chat_template(messages, chat_template=<candidate>,
tokenize=False, ...)` against the tokenizer of the served INT8 model, i.e. the
exact production render path (`serving_chat.py` calls the same function). Every
row below is a rendered-output comparison, not a claim taken from a README.

Acceptance criterion applied to any candidate: on healthy inputs it must render
**byte-identical** to the original, and differ only where the original crashes
or emits malformed text. Anything else is prompt drift, which silently
re-baselines #541 and invalidates any earlier measurement.

## 4. Findings

### 4.1 Defects in the original template

| # | Hunk | Class | Verified defect | Reachable in our stack | Evidence |
|---|---|---|---|---|---|
| O1 | `for args_name, args_value in tool_call.arguments\|items` | tool-call formatting | Yes — hard render failure | **No** | `TypeError: Can only get item pairs from a mapping` when `arguments` is a JSON string (case C2) |
| O2 | `raise_exception('System message must be at the beginning.')` | system-prompt handling | Yes — hard render failure | **No** | `TemplateError: System message must be at the beginning.` (case E1) |
| O3 | `raise_exception('No user query found in messages.')` | system/history handling | Yes — hard render failure | Degenerate histories only | `TemplateError: No user query found in messages.` (case F1); upstream discussion #16 |
| O4 | `{%- if loop.previtem and loop.previtem.role != "tool" %}` | im_start handling | Yes — malformed output, no exception | Degenerate histories only | tool message in first position renders a `<tool_response>` with **no** `<\|im_start\|>user` header (case G1) |
| O5 | unclosed `<think>` in an echoed assistant turn | thinking handling | Yes — duplicated think block | **No** (Anthropic front), yes on raw OpenAI path | case D2 below |
| O6 | history mutation across user turns | thinking handling / prefix cache | Yes — measurable prefix break | **Yes** | §4.3 |

O1 is neutralised by our own fork: `normalize_assistant_tool_call_arguments`
(`python/sglang/srt/entrypoints/openai/serving_chat.py:119`, called at `:733`)
parses string `arguments` into a dict before `apply_chat_template`. The
Anthropic front emits `"arguments": json.dumps(block.input or {})`
(`entrypoints/anthropic/serving.py:564`), i.e. exactly the string form that
would trip the original template, and it is converted back before rendering.

O2 is neutralised by `detect_inline_system_support`
(`python/sglang/srt/parser/template_detection.py:520`), which probes the loaded
template with a mid-conversation system message and sets
`self._merge_inline_system` (`entrypoints/anthropic/serving.py:263`) so inline
system turns are merged into the leading block for templates that raise. The
original template fails that probe, so merging is active.

O5, verified (case D2 — assistant turn with an unclosed `<think>` inside the
current tool round). The original emits a spurious empty think block and then
leaks the raw opener:

```
<|im_start|>assistant
<think>

</think>

<think>
cut off

<tool_call>
```

This is directly in #540's blast radius: a budget mechanism that truncates
reasoning without emitting `</think>` produces exactly this history. It is not
reachable through the Anthropic front, because prior-turn thinking is
re-wrapped with a closing tag by `wrap_reasoning_history`
(`serving_chat.py:1683`, `<think>` / `</think>` from the qwen3 detector,
`parser/reasoning_parser.py:257`). It **is** reachable for any client talking
to port 30030 directly. Note that froggeric does not fix this either: it drops
the spurious empty block but still emits the unclosed `<think>`, so its
"auto-injected closing tags" claim does not hold in this repro.

### 4.2 froggeric v21.3 — verified against claims

| # | Behaviour | Class | Assessment |
|---|---|---|---|
| F1 | Tool-instruction block is whitespace-mangled under Python Jinja2 | whitespace | **Regression.** Affects every tool-enabled request. |
| F2 | Preserves prior-turn thinking by default (`preserve_thinking` defaults to `true`) | thinking handling | Behaviour change, not a fix — the original already supports the kwarg (case B2 renders identically) |
| F3 | Drops the canonical `<think>\n\n</think>\n\n` block on in-round assistant turns | thinking handling | **Regression.** Contradicts its own KV-cache claim (§4.3) |
| F4 | Parallel tool calls separated by `\n\n` instead of `\n` | tool-call formatting | Drift from the trained format |
| F5 | String `arguments` render as raw JSON inside `<function=...>` with no `<parameter=>` wrapper | tool-call formatting | Does not crash, but emits history the `qwen3_coder` parser never produces |
| F6 | `<\|think_on\|>` / `<\|think_off\|>` intercepted in user text | thinking handling | **Hazard.** See below |
| F7 | Injects a warning string into tool responses after heuristic error detection | content injection | **Blocker** for #541 |
| F8 | Forces thinking off after 2 consecutive detected tool errors | thinking handling | **Blocker** for #540 and #541 |

F1, verbatim, as rendered by `apply_chat_template` on our tokenizer:

```
If you choose to call a function ONLY reply in the following format with NO suffix:<think>
Brief explanation of tool call
</think>
<tool_call>
```

and

```
- ALL explanation and reasoning MUST be placed strictly inside the <think></think> block.- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags.- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after thinking, with NO conversational text before it.- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with NO spaces or indentation before them.- To call multiple functions, output a separate, completely closed <tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.
```

Five separate instruction bullets are concatenated onto one line and the format
example is glued to the sentence introducing it. The template's `{%- set ... %}`
block combined with transformers' `trim_blocks`/`lstrip_blocks` environment eats
the newlines. This is the prompt every Claude-Code request would carry.

F6, verbatim diff for a user message that merely mentions the literal token:

```
--- official
+++ froggeric
@@ -1,4 +1,7 @@
 <|im_start|>user
-Explain what the <|think_off|> marker does.<|im_end|>
+Explain what the  marker does.<|im_end|>
 <|im_start|>assistant
 <think>
+
+</think>
+
```

The token is deleted from the user's text and thinking is silently switched
off. In a coding agent that pastes file contents and documentation verbatim
(this note contains the literal token), that is a live prompt-injection surface,
not a theoretical one.

F7/F8, verbatim, after two tool results whose first 80 characters match the
error heuristic:

```
 <|im_start|>user
 <tool_response>
 Error: no such file
+
+[WARNING] SYSTEM WARNING: 2 consecutive tool errors detected. Your previous approach is incorrect. You MUST use a fundamentally different approach or corrected arguments.
 </tool_response><|im_end|>
```

(the injected line carries an emoji in the original template; it is transcribed
here without it). The final generation prompt then becomes
`<|im_start|>assistant\n<think>\n\n</think>\n\n` — thinking forced **off**,
overriding `enable_thinking`. A benchmark arm that requests thinking would
silently stop thinking after two tool errors, and a thinking-budget mechanism
would have its boundary moved out from under it. Both #540 and #541 would
measure something other than what they intend.

### 4.3 Prefix-cache impact — the one claim worth acting on

froggeric's headline claim is a "100% Prefix KV Cache hit rate" from not
mutating history. Measured directly: build the token stream the server actually
holds after turn 1 (turn-1 prompt + the model's continuation), then render the
turn-2 prompt and count the common token prefix.

```
thinking ON  | original (default)                 | cached= 21 tok  turn2= 26 tok  common prefix= 10 tok
thinking ON  | original + preserve_thinking=True  | cached= 21 tok  turn2= 33 tok  common prefix= 21 tok
thinking ON  | froggeric v21.3 (default)          | cached= 21 tok  turn2= 33 tok  common prefix= 21 tok

thinking OFF | original (default)                 | cached= 18 tok  turn2= 28 tok  common prefix= 10 tok
thinking OFF | original + preserve_thinking=True  | cached= 18 tok  turn2= 32 tok  common prefix= 18 tok
thinking OFF | froggeric v21.3 (default)          | cached= 18 tok  turn2= 28 tok  common prefix= 10 tok
```

Three results, all verified rather than claimed:

1. The original template's default does break the prefix, and it breaks it at
   the **first** assistant turn of the conversation (10 of 21 / 10 of 18 tokens
   reused — the common prefix ends right after `<|im_start|>assistant\n`).
   Every new user turn in a long session re-renders the whole history without
   its think blocks, so the radix cache can only serve the part before the first
   assistant turn.
2. `preserve_thinking=True` on the **original** template reproduces the cached
   stream exactly (21/21, 18/18). No template swap is required to get this.
3. froggeric's claim does not hold in the thinking-off arm: it scores 10/18,
   identical to the unfixed original, because it drops the canonical
   `<think>\n\n</think>\n\n` block (F3). For #541's thinking-off arm it is no
   better than the original.

Scope note: inside a single user turn — the Claude-Code subagent shape, one
prompt followed by a long tool loop — the original template is already fine.
Measured on a one-call tool round: 273 of 273 cached tokens reused (100%).
The defect is confined to conversations with multiple real user turns.

### 4.4 Candidate comparison matrix

Byte-identity against the original across the 16 rendered cases. `renders*`
means the original raised and the candidate produced output.

```
case                                   unsloth     froggeric   allanchan
A1_single_think_on                     identical   identical   identical
A2_single_think_off                    identical   identical   identical
A3_no_system                           identical   identical   identical
B1_multiturn_prior_thinking            identical   DIFFERS     identical
B2_multiturn_preserve_thinking_true    identical   identical   identical
C1_tool_roundtrip_dict_args            identical   DIFFERS     DIFFERS
C2_tool_roundtrip_STRING_args          renders*    renders*    ERROR
C3_parallel_tool_calls                 identical   DIFFERS     DIFFERS
D1_unclosed_think                      identical   identical   identical
D2_unclosed_think_current_round        identical   DIFFERS     DIFFERS
E1_mid_conversation_system             renders*    renders*    identical
F1_no_user_query                       renders*    renders*    renders*
G1_tool_message_first                  identical   DIFFERS     DIFFERS
H1_thinking_off_multiturn_echo         identical   identical   identical
I1_think_off_token_in_user_text        identical   DIFFERS     identical
J1_two_consecutive_tool_errors         identical   DIFFERS     DIFFERS
```

Only the unsloth template meets the acceptance criterion: identical everywhere
the original works, different only where it crashes. It is a surgical patch
(+293 bytes) that merges leading system/developer messages, guards
`arguments` with `is mapping`, and removes both `raise_exception` calls.

It is still not recommended, because two of its three "fixes" convert a loud
failure into silent data loss:

* string `arguments` — the original raises; unsloth renders the call with the
  arguments **dropped entirely**:
  `<tool_call>\n<function=Read>\n</function>\n</tool_call>`. For a coding agent
  that turns a retryable 400 into a wrong action.
* mid-conversation system message — unsloth silently omits it rather than
  rendering it (case E1: `S-mid` does not appear in the output).

Neither failure mode is reachable through our front today (§4.1), so adopting
unsloth buys nothing and adds a silent-loss path if the front's normalisation
is ever bypassed. It also does not fix O4 (tool message in first position),
which it renders identically to the original.

## 5. Impact on the dependent efforts

* **Claude-Code subagent tool calling through the Anthropic front** — no defect
  reachable. O1 and O2 are handled in our code, verified by reading the call
  sites, not by assumption. Adopting froggeric would actively hurt: the mangled
  instruction block (F1), the drift in parallel-call separators (F4), and the
  malformed string-argument rendering (F5) all sit on the hot path.
* **#540 thinking budget / token boundaries** — the original's think-token
  boundaries are sound for our path; the O5 double-think case cannot be produced
  through the Anthropic front because `wrap_reasoning_history` always closes the
  tag. Guard for #540: if the budget mechanism ever truncates reasoning without
  emitting `</think>`, the raw OpenAI path on 30030 will produce the malformed
  history in §4.1. froggeric's F8 (forced thinking-off after two tool errors)
  would break the budget mechanism outright.
* **#541 A/B validity** — no template-induced invalidity today; the original's
  thinking-on and thinking-off generation prompts are exactly as documented
  (cases A1/A2). Swapping to froggeric mid-programme would invalidate the
  benchmark twice over: prompt drift changes the baseline (F1/F4), and F8
  silently changes the arm being measured. The prefix-cache finding in §4.3 is
  worth folding in as a variable: throughput in a multi-user-turn scenario
  differs measurably between `preserve_thinking` on and off, and #541 should
  either pin it or measure it rather than leave it unstated.
* **Cosmetic only** — none of the diffs examined were purely cosmetic.

## 6. Recommendation

**Adopt the community template: no.** Neither froggeric v21.3, nor unsloth, nor
allanchan339. Keep the original template, unchanged, from the model directory.

Reasons, in order: the reachable defect set is empty for our front; the
community template's claimed KV-cache advantage does not reproduce in the
thinking-off arm; and it introduces two behaviours (content injection into tool
responses, forced thinking-off after tool errors) that would corrupt #540 and
#541 rather than support them.

**Follow-up that is worth doing, and does not involve a template swap:** treat
`preserve_thinking` as a candidate setting for the original template. It is the
only variant measured that reproduces the generated token stream byte-exactly
across user turns (§4.3), in both the thinking-on and thinking-off arms. It is
not free — prior-turn reasoning stays in context permanently, so prompt length
grows, and the quality effect of keeping prior reasoning in context is
unmeasured. It should be an arm in #541, not a blind default.

Nothing in this note has been applied.

### Apply procedure, if a template swap is ever decided

Recorded for completeness; not recommended today.

1. Place the candidate at a path outside the model cache, for example
   `docs/dev/candidate_chat_template_542.jinja`, and never edit the files under
   `/spinning/llm_stuff/club-3090/models-cache/` — both quant dirs share the
   same template and both would drift.
2. Add `--chat-template <path>` to the boot command for port 30030. The live
   command currently passes no such flag, so the model-dir template is in
   effect; adding the flag is the whole switch.
3. Restart is required — the template is compiled once at startup by the
   tokenizer manager, and `detect_inline_system_support` probes it at init, so
   the front's system-merge decision is only re-evaluated on boot.
4. Before restarting, re-run the byte-identity matrix in §4.4 against the exact
   candidate file. A candidate that is not identical on the healthy cases
   re-baselines #541 and every earlier throughput measurement.
5. Re-run the #541 baseline arm after the swap regardless, and record the new
   template sha256 alongside the results.

For `preserve_thinking` specifically no template file is involved: it is passed
as `chat_template_kwargs: {"preserve_thinking": true}` on the request, or set
server-side; the original template already honours it (case B2).
