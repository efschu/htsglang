# TICKET #540 — Anthropic Messages front: seven protocol-conformance gaps

Status: **DESK COMPLETE, hermetically validated.** Everything below ran on
CPU only (`CUDA_VISIBLE_DEVICES=99`) against the real FastAPI app with a
mocked `OpenAIServingChat`. No GPU window was taken, no model was loaded, and
nothing here is a live-boot claim. The one statement that needs a live boot is
named in §6.

Branch `feat/anthropic-front-conformance`, cut from `integration/r3-probe-next2`
(tip `785458e161`).

---

## 1 — What was wrong

An audit of `python/sglang/srt/entrypoints/anthropic/` against the real
Anthropic Messages API found seven divergences. Three of them break a Claude
Code agent loop outright; the rest are protocol-shape defects that strict SDK
clients can trip over.

| id | gap | consequence |
|---|---|---|
| G1 | extended thinking defaulted to the SERVER's reasoning-parser setting | a boot with `--reasoning-parser qwen3` answered every plain request with a thinking block that consumed the whole `max_tokens` budget; the tool round trip never got emitted |
| G2 | content blocks were a CLOSED discriminated union over 8 tags | one `document` / `mcp_tool_use` / `server_tool_use` / `web_search_tool_result` block 400'd the ENTIRE conversation |
| G3 | `stop_reason:"stop_sequence"` was unreachable and `stop_sequence` never populated | a caller's `stop_sequences` hit was reported as a plain `end_turn` with no indication which string fired |
| G4 | `message_start` withheld until the first backend chunk; no `ping` frames | the client saw nothing for the whole prefill; idle connections had no keep-alive |
| G5 | outgoing `tool_use.id` was the backend's `call_...` | clients (and replayed transcripts) that validate the `toolu_` prefix reject it |
| G6 | `redacted_thinking` in history raised | a client replaying its own transcript ships that block every turn, so the conversation 400s permanently |
| G7 | `message_start.message` omitted `stop_reason` / `stop_sequence` | strict SDK parsers type them as required-but-nullable; a missing key is a schema violation |

Out of scope by instruction and correctly still absent: `/v1/messages/batches`,
`anthropic-beta` header handling, Anthropic server-tool execution.

## 2 — What shipped

`python/sglang/srt/entrypoints/anthropic/protocol.py`
- `UnknownContentBlock` (`extra="allow"`) plus a callable
  `Discriminator(_content_block_discriminator)` replacing
  `Field(discriminator="type")`. Pydantic's field discriminator cannot express
  a catch-all member; a callable discriminator can, because it maps the raw
  value onto a tag we control. Known tags still route to their precise model,
  so a malformed `text` block keeps its exact validation error and does NOT
  silently degrade to "unknown". `KNOWN_CONTENT_BLOCK_TAGS` is the one list
  both the protocol and the serving layer read.

`python/sglang/srt/entrypoints/anthropic/serving.py`
- G1: an `else` branch on the `thinking` check calls
  `apply_reasoning_enabled(chat_request, False)` (`serving.py:708`).
  Deliberate override of the
  server-level default, Anthropic front only — the OpenAI front reads its own
  `chat_template_kwargs`/`reasoning_effort` and never passes through here.
- G2/G6: the message-block loop skips unmodelled tags with one warning naming
  the type; `redacted_thinking` counts and warns instead of raising.
- G3: `_resolve_stop_sequence()` maps the backend's `matched_stop`
  (`openai/protocol.py:1107` non-streaming, `:1169` streaming) onto the
  Anthropic pair, in both paths. An `int` `matched_stop` is a stop TOKEN and
  is correctly ignored; a `str` only counts when it is one the CALLER asked
  for, so a chat-template stop stays `end_turn`.
- G4: `message_start` is emitted before the OpenAI stream is iterated, and
  `message_delta` now carries `input_tokens` as the correction channel;
  `PING_INTERVAL_SECONDS` (5.0 s) drives `ping` frames through a task-based
  read (`_iter_lines_with_pings`) that neither polls nor drops chunks and
  cancels the pending read on client disconnect.
- G5: `_anthropic_tool_use_id()` normalises outgoing ids to `toolu_`;
  INBOUND ids are never rewritten, so the emitted id is the accepted id and
  arbitrary legacy id strings keep working.
- G7: `_emit_message_start()` writes the two explicit nulls for that event
  only. The non-streaming response keeps `exclude_none=True` (it drops the
  cache/usage fields the local backend never populates) with ONE targeted
  `payload.setdefault("stop_sequence", None)` — the Message object documents
  that field as always present.

## 3 — Deliberate divergence, stated rather than hidden

"Absent `thinking` behaves identically to `{"type":"disabled"}`" holds except
on an ALWAYS-ON reasoning parser, where `apply_reasoning_enabled(..., False)`
raises by design (`serving_chat.py:1722`). An explicit `disabled` still
surfaces that as a 400 — the caller asked for something the model cannot do.
An ABSENT field is not a request, so there the failure is downgraded to a
WARNING and the request is served. Making them bit-identical would 400 every
plain message on such a model, which is a regression, not conformance.
Pinned by `TestThinkingDefaultsOff::test_always_on_model_still_serves_plain_requests`.

## 4 — Before/after matrix (the can-fail proof)

New file `test/registered/unit/entrypoints/anthropic/test_conformance_http.py`
drives the REAL FastAPI `app` through `TestClient` with a mocked backend.
Procedure: `git checkout -- python/sglang/srt/entrypoints/anthropic/` to
restore the pre-change source, run, restore the patch, run again.

| test | gap | pre-change | post-change |
|---|---|---|---|
| `test_absent_thinking_disables_reasoning` | G1 | FAIL (`[] != [False]`) | PASS |
| `test_absent_thinking_matches_explicit_disabled` | G1 | FAIL (`[] != [False]`) | PASS |
| `test_always_on_model_still_serves_plain_requests` | G1 | FAIL (no warning) | PASS |
| `test_explicit_enabled_still_turns_reasoning_on` | G1 | PASS | PASS — regression guard, intentionally unchanged behaviour |
| `test_unknown_block_type_does_not_reject_the_conversation` | G2 | FAIL (no warning; 400) | PASS |
| `test_several_unknown_tags_all_pass` (4 subtests) | G2 | 4/4 SUBFAIL (`400 != 200`) | 4/4 PASS |
| `test_malformed_known_block_still_reports_a_precise_error` | G2 | PASS | PASS — guard that the fallback did NOT weaken validation |
| `test_redacted_thinking_does_not_reject_the_conversation` | G6 | FAIL (`400 != 200`) | PASS |
| `test_non_streaming_reports_the_matched_stop_sequence` | G3 | FAIL (`end_turn`) | PASS |
| `test_template_stop_is_not_reported_as_a_stop_sequence` | G3 | FAIL (`KeyError: stop_sequence`) | PASS |
| `test_streaming_message_delta_reports_the_stop_sequence` | G3 | FAIL (`end_turn`) | PASS |
| `test_message_start_reaches_the_wire_before_the_backend_speaks` | G4a | FAIL (`1 not less than 0`) | PASS |
| `test_ping_frames_are_emitted_while_the_backend_is_silent` | G4b | FAIL (no `PING_INTERVAL_SECONDS`) | PASS |
| `test_message_start_precedes_the_first_content_event` | G4 | PASS | PASS — ordering regression guard, NOT a can-fail for G4a |
| `test_message_start_carries_explicit_stop_nulls` | G7 | FAIL (key absent) | PASS |
| `test_non_streaming_response_carries_explicit_stop_sequence_null` | G7 | FAIL (key absent) | PASS |
| `test_non_streaming_tool_use_id_is_anthropic_shaped` | G5 | FAIL (`call_abc123`) | PASS |
| `test_streaming_tool_use_id_is_anthropic_shaped` | G5 | FAIL | PASS |
| `test_emitted_tool_id_is_accepted_back_on_tool_result` | G5 | PASS | PASS — round-trip guard; the point is that it must STAY passing |
| `test_arbitrary_inbound_tool_ids_are_never_rewritten` | G5 | PASS | PASS — same, inbound ids must stay untouched |

Aggregate: pre-change **17 failed / 7 passed**, post-change **20 passed
(+4 subtests)**. Every gap G1-G7 has at least one arm that fails on the old
tree. The four rows marked "guard" pass on both versions on purpose and are
labelled as such rather than counted as evidence.

**The G4a proof needed a second attempt, and the first attempt was worthless.**
A latency assertion through `TestClient.stream(...)` measured nothing:
Starlette's test transport buffers the whole streaming response before
`iter_lines()` returns, so time-to-first-frame was 0.000 s on BOTH versions
(measured: `enter=1.817s firstline=0.000s` against a backend sleeping
0.6 s/chunk). Replaced with `_drive_asgi()`, which calls the ASGI app directly
and records wire sends interleaved with backend yields into one ordered trace.
That version fails on the old tree with `1 not less than 0`.

Existing suite: `test_serving.py` went 56 → 56 passing. Four of its cases were
RETARGETED, not deleted, because the intended behaviour changed:
`test_redacted_thinking_history_is_rejected` →
`..._is_skipped_with_warning`; the three usage cases now read the corrected
totals from `message_delta` instead of `message_start` (that IS the G4a
correction channel) and one was renamed
`test_stream_usage_subtracts_cache_read_and_corrects_in_message_delta`.

## 5 — Reproduce

```bash
cd /spinning/wt-anthropic-front
CUDA_VISIBLE_DEVICES=99 PYTHONPATH=$PWD/python \
  /spinning/htsglang-gpu/.venv/bin/python -m pytest \
  test/registered/unit/entrypoints/anthropic/ -q

/spinning/htsglang-gpu/.venv/bin/python -m ruff check \
  python/sglang/srt/entrypoints/anthropic/ \
  test/registered/unit/entrypoints/anthropic/
```

Before/after matrix (destructive to the working tree, restore afterwards):

```bash
git diff -- python/sglang/srt/entrypoints/anthropic/ > /tmp/anthropic_fix.patch
git checkout -- python/sglang/srt/entrypoints/anthropic/
CUDA_VISIBLE_DEVICES=99 PYTHONPATH=$PWD/python \
  /spinning/htsglang-gpu/.venv/bin/python -m pytest \
  test/registered/unit/entrypoints/anthropic/test_conformance_http.py -v
git apply /tmp/anthropic_fix.patch
```

## 6 — Remaining protocol risk (honest list)

1. **No live-boot evidence.** Everything is hermetic against a mocked
   backend. The claim "a real `claude` CLI now gets its tool round trip on a
   `--reasoning-parser qwen3` boot" is PREDICTED by G1, not measured. It needs
   one serving window plus a `claude` invocation WITHOUT
   `MAX_THINKING_TOKENS=0` (which is exactly the client-side setting the
   catalog currently documents as load-bearing).
2. **The stop-sequence suffix fallback is nearly inert.** SGLang strips the
   matched stop string from the output unless `no_stop_trim` is set, so the
   `accumulated_text.endswith(...)` probe normally finds nothing and returns
   None. It rescues only the no-trim configuration; the `matched_stop` path is
   what actually carries the feature. Documented at the helper.
3. **Ping cadence is a shipped constant with no operating-point proof.**
   5.0 s is a plausible keep-alive, not a measured one. No proxy/LB timeout on
   this rig was surveyed to derive it.
4. **`stop_sequence` on a MULTI-choice completion.** Only `choices[0]` is read
   — unchanged from before, and `n>1` is not exercised by the Anthropic front,
   but it is an assumption, not a guarantee.
5. **Unknown blocks nested inside `tool_result.content`** are dropped by the
   pre-existing dict-based walk without the new warning. Non-fatal already, so
   no behaviour change, but the log line does not name them.
6. **Unknown block types in the `system` array** are skipped silently by
   `_extract_system_text`'s existing non-text branch — again pre-existing and
   non-fatal, but not covered by the new warning.
7. **Count-tokens path** goes through the same conversion, so it inherits G1's
   default-off. That is consistent, but it means a `count_tokens` estimate for
   a thinking-enabled request is only accurate if the caller passes `thinking`
   there too.

---

## Effort-collapse verification (2026-08-18, second pass — prior-art verdict)

The re-issued remedy task found the remedy SHIPPED and contained:

* `57b04b2434` ("xhigh is passed through: the collapse rewrote a supported
  request", branch `fix/540-effort-collapse`, held by its owner's worktree)
  is an ancestor of comp4 `921d63defc`. Default: `xhigh` passes through
  verbatim — and the OpenAI protocol Literal gained `xhigh`, so the value
  reaches the Qwen3.8 template, which ACCEPTS it explicitly (the 500s are
  on explicit `high`/`max` only). The collapse survives strictly as the
  opt-in `SGLANG_ANTHROPIC_XHIGH_EFFORT` for templates whose top tier is
  named `max`, logged, never silent. `test_effort_passthrough_540.py`:
  8 tests, re-run green on comp4 here.
* The LIVE router (30099, wt-anthropic-front) independently normalizes
  client efforts onto the omit-encoding — its module doc states the
  reality and the running code carries it — so live traffic was already
  protected at the router layer; the front-side fix removes the hazard for
  direct-to-serving clients and rides the next composite boot.
* The live SERVING lineage (wt-merge-r4, currently DOWN) still carries the
  OLD collapse at its `serving.py:730-739`; it is retired by deployment,
  not by an edit to a running tree.

**Deferred, serving-down:** the one-real-request-per-arm probe through
30099 (explicit `xhigh` → 200 with thinking; omit → 200; explicit `max`
against Qwen3.8 → the template's refusal, surfaced honestly). Belongs in
the harvest boot's idle phase; a WINDOW_LADDER_0818 line carries it.
