# NOTE 540 - per-request thinking budget

Status: implemented on `feat/thinking-budget-540`, hermetic tests green, live
validation on Qwen3.6-27B pending the next bundled serving restart.

## What existed before

* `ThinkingBudgetLogitProcessor` (`python/sglang/srt/sampling/custom_logit_processor.py`)
  forced a newline plus the thinking-end token once the thinking section
  exceeded `custom_params["thinking_budget"]`. The marker token ids were class
  constants per model family (`Qwen3...` 151667/151668, `Glm4Moe...`
  151350/151351, `DeepSeekR1...` 128798/128799).
* Using it required the client to serialize the processor into
  `custom_logit_processor` **and** the server to run with
  `--enable-custom-logit-processor`.
* The Anthropic front accepted `thinking.budget_tokens` and logged a WARNING
  saying it is not enforced.
* Per-request thinking ON/OFF already worked (chat template toggle plus the
  Anthropic `thinking.type` mapping) and is unchanged.

## Motivating defect

Hardcoded marker ids are wrong as soon as the deployed checkpoint tokenizes the
markers differently. The membership check `THINKING_START_TOKEN_ID in cur_ids`
then never fires and the budget is silently ignored - upstream
sgl-project/sglang#25536 (Qwen3.6) and #20274 (GLM-5).

This reproduces on our own checkpoint: `Qwen3.6-27B-INT8-W8A8` tokenizes
`<think>` to **248068** and `</think>` to **248069**, while
`Qwen3ThinkingBudgetLogitProcessor` hardcodes 151667/151668. The documented
usage would have produced an unbudgeted answer with no error.

## What changed

New module `python/sglang/srt/sampling/thinking_budget.py`:

* `validate_thinking_budget` - non-negative int, `-1`/`None` mean no budget,
  `bool`/`float` rejected. Same contract as vLLM's
  `validate_thinking_token_budget` (`vllm/sampling_params.py`).
* `resolve_thinking_budget_token_ids(tokenizer, reasoning_parser, model_name)` -
  takes the marker *strings* from the configured reasoning parser's detector
  (`think_start_token` / `think_end_token`) and encodes them with the serving
  tokenizer. Each marker must encode to exactly one token; the separating
  newline is optional. Anything else raises
  `ThinkingBudgetUnsupportedError` naming model and markers. This mirrors
  vLLM's `ReasoningConfig.initialize_token_ids` (`vllm/config/reasoning.py`).
* `attach_thinking_budget(...)` - called from
  `TokenizerManager._create_tokenized_object`. Injects the derived ids into
  `custom_params` and, when the client sent no processor of its own, attaches
  the built-in one. Server-owned `custom_params` keys are stripped from client
  input so the internal marker cannot be forged.

Enforcement path:

```
request field / thinking.budget_tokens
  -> ChatCompletionRequest.thinking_budget            (openai/protocol.py)
  -> SamplingParams.thinking_budget                   (sampling/sampling_params.py)
  -> attach_thinking_budget                           (managers/tokenizer_manager.py)
       resolve ids from tokenizer  | 400 on failure
       custom_params += ids, budget, internal marker
       custom_logit_processor = built-in processor
  -> Req.custom_logit_processor_internal              (managers/schedule_batch.py)
  -> uses_custom_logit_processor gate                 (sampling/sampling_batch_info.py)
  -> ThinkingBudgetLogitProcessor                     (sampling/custom_logit_processor.py)
       applied in layers/sampler.py (non-spec)
       and in eagle_sample (EAGLE/MTP verify)
```

The `--enable-custom-logit-processor` flag still gates *client-supplied*
processors. It does not gate the internal attachment, whose payload is produced
from an in-tree class.

Per-request API:

```bash
# OpenAI front
curl .../v1/chat/completions -d '{"model":"Qwen3.6-27B",
  "messages":[{"role":"user","content":"..."}], "thinking_budget": 256}'
```

```python
client.chat.completions.create(..., extra_body={"thinking_budget": 256})
```

```json
// Anthropic front
{"model": "...", "max_tokens": 4096,
 "thinking": {"type": "enabled", "budget_tokens": 2048}}
```

## Speculative decoding

`eagle_sample` (EAGLE / MTP verify) samples without going through
`layers/sampler.py`, so custom logit processors were inert there - the budget
would have been a silent no-op on exactly the boot we run. The processors are
now applied in `eagle_sample` the same way DFlash already did, and
`apply_custom_logit_processor` repeats each request's params across its
`draft_token_num` rows so a row-indexed processor covers the whole draft
instead of only its first row. State-based processors see the committed prefix
only, so a budget can overshoot by at most `draft_token_num` tokens before the
close is forced.

## Semantics decisions

* **Anthropic `adaptive`** stays enabled-without-budget. The SDK forbids
  `budget_tokens` on `adaptive`, and the local backend has no auto-throttle to
  emulate; reasoning runs to its natural end. Documented in
  `AnthropicThinkingParam`.
* **`budget_tokens` on a model without a reasoning parser** stays
  accept-and-log. There are no markers to cap, and 400-ing a request the
  Anthropic SDK would have accepted is worse than the warning. The `>= 1024`
  SDK validation is unchanged.
* **`-1`** means unlimited and normalizes to "no budget", matching vLLM.
* **Legacy `custom_params["thinking_budget"]` form** (docs/basic_usage/glm45.md,
  deepseek_v3.md) keeps working and now also receives derived ids, so it is
  immune to the same mismatch.
* **Conflict**: a first-class `thinking_budget` together with a client-supplied
  `custom_logit_processor` is a 400 - a request carries exactly one processor,
  and silently enforcing neither is the failure mode this task exists to kill.

## Tests

Hermetic, `CUDA_VISIBLE_DEVICES=99`, no server:

* `test/registered/unit/sampling/test_thinking_budget.py`
* `test/registered/unit/managers/test_thinking_budget_wiring.py`
* `test/registered/unit/entrypoints/openai/test_thinking_budget_request.py`
* `test/registered/unit/entrypoints/anthropic/test_serving.py` (extended)

The tokenizer fixture carries the real Qwen3.6 ids but does not read the model
path at runtime. Can-fail was proven by mutation: relaxing the single-token
check, dropping the id injection, dropping the per-row param repetition and
dropping the internal-processor gate each turn the corresponding tests red.

## Open

* Live validation on the running Qwen3.6-27B boot (budget honored end to end,
  reasoning token counts in `usage`, spec-decode overshoot within
  `draft_token_num`) is pending the next bundled serving restart. No server was
  restarted for this work.
