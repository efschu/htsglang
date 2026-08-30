# DESIGN #502 — Live interview copilot

Status: P1 (design + skeleton) — no GPU work, no live exposure.
Line: `integration/r3-probe-next2` @ `3531da3f30`.

A browser app for online conversations. It captures two audio sources at once
(the user's own microphone and the far side via tab capture), streams both to
the runtime's ASR surface, and shows the user a continuously updated column of
**keywords and short explanations to READ** about whatever is currently being
discussed. The user is briefed in advance; the briefing is extended during the
conversation by a background job. Several topic contexts are kept pre-prefilled
so a hint decode starts inside an existing context instead of re-reading the
briefing.

Everything the app needs from a model it gets from the htsglang runtime that is
already serving. The app process owns no weights, no VRAM, no CUDA context.

---

## 1. What already exists (read from code, not assumed)

This section is the evidence base. Every later decision points back into it.

### 1.1 Streaming ASR is already IN the runtime

The input side of speech is not missing from this tree — it is a first-class
part of the OpenAI-compatible surface:

* `POST /v1/audio/transcriptions` — `entrypoints/http_server.py:2316`.
* `WEBSOCKET /v1/realtime` — `entrypoints/http_server.py:2357`, dispatching to
  `openai_serving_transcription.handle_websocket(ws)` (`:2365`).
* Handler: `entrypoints/openai/realtime/handler.py:64`
  (`handle_realtime_transcription`), event loop in
  `entrypoints/openai/realtime/session.py`.
* Client events accepted: `input_audio_buffer.append`,
  `input_audio_buffer.commit`, `input_audio_buffer.clear`
  (`realtime/session.py:111-113`), plus `session.update`.
  Server events: `session.created` (`:212`), `session.updated` (`:414`),
  `input_audio_buffer.committed` (`:506`), `conversation.item.created`
  (`:514`), and the transcription `delta` / `completed` events imported at
  `:34-38`.
* Chunked streaming with prefix rollback:
  `entrypoints/openai/streaming_asr.py:27` (`StreamingASRState`).
* Adapters registered for Whisper, Qwen3-ASR, MiMo-V2-ASR
  (`entrypoints/openai/transcription_adapters/`).

Three properties of that surface constrain this design and are quoted here so
no later reader re-derives them:

1. **Audio goes over JSON, not binary frames.** `realtime/session.py:249-253`:
   `"OpenAI Realtime is base64 PCM in JSON; binary frames aren't supported."`
   — a binary frame on `/v1/realtime` is answered with an error telling the
   caller to `use input_audio_buffer.append with base64 audio`.
2. **There is no server-side VAD.** `realtime/session.py:315-321`:
   `"Server-side VAD is not implemented; set audio.input.turn_detection: null
   and commit explicitly."` Utterance boundaries are the caller's job.
3. **One WebSocket is one transcription stream.** `RealtimeConnection` holds a
   single audio buffer and a single item (`realtime/session.py:143-164`), and
   concurrency is capped by `--asr-max-concurrent-sessions` (default 32,
   `server_args.py:2718`) with an `asr_max_buffer_seconds` overflow guard
   (default 60, `server_args.py:2714`).

Consequence for the two-source requirement: two tracks means **two
`/v1/realtime` connections**, one per track. There is no track/channel field in
that protocol to multiplex on, and mixing the two sources into one stream would
destroy exactly the self/other attribution the app exists to show.

### 1.2 The LLM seam

`entrypoints/openai/protocol.py:464-469` and `:868-873` carry, on both the
completion and chat-completion request schemas:

```
priority: Optional[int] = None
# Fast-lane scheduling class (Variant C Stage 0): "fast" or None.
# OpenAI clients send it via extra_body={"lane": "fast"}. Only takes
# effect when the server runs with --enable-fast-lane; forwarded to
# GenerateReqInput.lane exactly like the native /generate endpoint.
lane: Optional[str] = None
```

validated at `:492-496` / `:907-911` (`lane must be "fast" or omitted`). So the
priority class of a request is expressible **from an ordinary OpenAI client**,
which is the seam the #466 translator already uses for its MT calls
(`translator/mt.py:5-9`: "we do not reach into the engine, we call it like a
stranger would").

`usage.prompt_tokens_details.cached_tokens` is populated from
`meta_info["cached_tokens"]` (`entrypoints/openai/usage_processor.py:42`), but
with a trap that this design depends on:
`usage_processor.py:15` returns `PromptTokensDetails(...) if count > 0 else
None` — **a zero hit is reported as an absent details object, not as 0.** Any
residency probe that reads `details.cached_tokens` without treating `None` as
zero will silently read "unknown" where the truth is "nothing was cached".

### 1.3 What does NOT exist — and the predicates that say so

The briefing named "#410 checkpoints" as a candidate mechanism for holding
pre-prefilled topic sessions. **#410 is not a session-checkpoint feature in
this tree.** The only occurrence of the number anywhere under `docs/` or
`python/` is a roadmap row about pricing the breakable-graph route
(`docs/dev/ROADMAP_456_matrix_execution.md:36`), and `git log --grep='#410'`
returns nothing. There is no checkpoint API to build on. The mechanisms that
DO exist are enumerated below, with the predicate that bounds each.

**(a) Sessions are replay + prefix cache, not held KV.**
`session/session_controller.py:203-332` (`Session.create_req`) builds a *new*
`Req` per turn whose `origin_input_ids` is the previous turn's input + output +
the new input (`_concat_token_arrays`, `:177-201`). The restored context comes
from the ordinary radix prefix match, not from a session-owned allocation.
HTTP surface: `/open_session` `http_server.py:1867`, `/close_session` `:1881`.

**(b) The radix-native session mode does not pin either.**
`--enable-session-radix-cache` (`server_args.py:2757`) is gated on
`server_args.py:13759-13761`
(`if self.enable_session_radix_cache and self.radix_eviction_policy !=
"priority": raise ValueError(...)`), and its own docstring states the reach
verbatim (`mem_cache/session_radix_cache.py:23-27`):

> `Tags radix KV by session id; release_session (close) frees a session's
> tagged chains. A node holds the set of sessions on it, so a node shared by
> several sessions is freed only when its last holder closes. Tagged KV is
> ordinary LRU radix -- no pinning, no open.`

**(c) There is no pin API at all.** `RadixCache.evict()`
(`mem_cache/radix_cache.py:569-593`) walks evictable leaves; the only
protection is `lock_ref`, incremented while a request holds the node as its
`last_node` (`radix_cache.py:598-625`, `:492`, `:545-546`) — held
automatically for the duration of an in-flight request and not settable from
outside. Eviction *order* among unlocked nodes is
`PriorityStrategy.get_priority()` = `(node.priority, node.last_access_time)`
(`mem_cache/evict_policy.py:41-47`), with `node.priority` seeded from the
request priority (`radix_cache.py:241`). **Priority moves a prefix later in the
victim order; it does not exempt it.** No `pin` / `keep_warm` / `protect`
method exists on `BasePrefixCache` (`mem_cache/base_prefix_cache.py:244-345`)
or any subclass.

**(d) Prefill-only priming is legal but has a spec caveat.**
`SamplingParams` permits `max_new_tokens == 0`
(`sampling_params.py:207-211`). `Req.is_prefill_only`
(`schedule_batch.py:1097-1103`) is:

```
spec_alg = get_server_args().speculative_algorithm
return self.sampling_params.max_new_tokens == 0 and spec_alg is None
```

so under the standing MTP/NEXTN recipe the *prefill-only optimisation* is off,
while the request itself still runs prefill and still populates the radix tree.
The priming path therefore works under speculation; only its cheapness claim
does not transfer. There is **no dedicated warmup endpoint** — `entrypoints/
warmup.py` is a fixed boot-time registry, not a per-topic API.

**(e) #261 session handover is the opposite of warm.**
`POST /session_handover` (`http_server.py:1172`) *parks* a prefix and refuses
further extension of it (`managers/session_handover.py:327-336`, and
`scheduler.py:2599-2622` aborts a new request extending a parked prefix). Its
enforced scope at this tip: `tp_size == 1 and pp_size == 1` source
(`session_handover.py:416-429`), `page_size == 1` (`:425`),
`hicache_storage_backend == "file"` plus hierarchical cache enabled
(`:385-396`). It is a migration tool, not a warm-keeping tool.

**(f) #274 dual-group lane cannot be the background lane on this rig's
homogeneous case, and is not the right shape anyway.**
`server_args.py:9496-9503`:

```
if self.dual_group_lane:
    if not isinstance(self.rank_tp_ratio, list):
        raise ValueError(
            "--dual-group-lane requires an EXPLICIT --rank-tp-ratio "
            "integer list (...)  'auto' is not accepted here (...)"
```

and the *same* validation function `_handle_uneven_tp`
(`server_args.py:9377-9997`) later rejects a uniform vector at `:9619-9623`:

```
if len(set(self.rank_tp_ratio)) == 1:
    raise ValueError(
        "--rank-tp-ratio with identical entries is the even "
        "split — omit the flag instead."
    )
```

Executed proof (CPU only, `CUDA_VISIBLE_DEVICES=99`): calling
`_handle_uneven_tp()` on a `ServerArgs` with `dual_group_lane=True`,
`tp_size=2`, `rank_tp_ratio=[1,1]` raises exactly
`--rank-tp-ratio with identical entries is the even split — omit the flag
instead.` **`FEATURE_CATALOG.md` §5's claim that "a UNIFORM vector is accepted,
so the lane is reachable on a homogeneous rig" is false at this tip**; the
catalog is corrected in this change (code wins, CLAUDE.md MECHANISM REACH).

**(g) #347 idle workbench is idle-only.** `workbench/service.py:110-115`
refuses when not enabled, and the scheduler runs tenants only when the rig is
judged idle (`--workbench-idle-grace-seconds`, `server_args.py:5582-5587`);
the shipped tenants are `training`, `fp8_tuner`, `card_probe`
(`docs/dev/DESIGN_347_idle_workbench.md:87-91`) — none of them an inference
tenant. A background job that must run *while a live stream is being served* is
outside its contract by construction.

**(h) Fast lane is the mechanism that fits.**
`server_args.py:15056-15082`: `--enable-fast-lane` force-enables priority
scheduling in a two-tier mode, refuses to combine with
`--schedule-low-priority-values-first`, defaults the heavy tier to
`default_priority_value = 0`, and requires
`fast_lane_priority > default_priority_value`. Requests are tagged at
`scheduler.py:2704-2708` (`req.is_fast_lane = getattr(recv_req, "lane", None)
== "fast"`).

---

## 2. Architecture

```
 browser (PWA, desktop Chrome)
   mic  ──getUserMedia────► AudioWorklet ─┐
   tab  ──getDisplayMedia──► AudioWorklet ─┤ 16 kHz PCM16, 20 ms frames
                                           │ tagged binary frames
                                           ▼
                              ┌────────── copilot app ──────────┐
                              │  WS /api/copilot/stream         │
                              │  session, journal, transcript   │
                              │  topic registry (warm state)    │
                              │  hint scheduler / expander      │
                              └───┬──────────────────┬──────────┘
                two /v1/realtime  │                  │  /v1/chat/completions
                (self + other)    ▼                  ▼  lane=fast | default
                              ┌──────── htsglang runtime ────────┐
                              │ ASR adapters | LLM | radix cache │
                              └──────────────────────────────────┘
```

### 2.1 Why a separate app process, and why that is not a second engine

The copilot app is a FastAPI process under `python/sglang/srt/copilot/`,
launched with `python -m sglang.srt.copilot.launch`, in the same shape as the
#466 translator (`translator/server.py:612`, `translator/launch.py:180`).

It differs from the translator in the one respect that matters to the
ONE RUNTIME law: **the translator's process loads ASR and TTS weights onto a
pinned card** (`translator/asr_backends.py:59-80`, `translator/
inprocess_tts.py:84`), the copilot's process loads **nothing**. It has no
model, no CUDA context, no ledger entry, no GPU. It is a protocol adapter and a
state machine in front of the runtime's own public surface — the same category
as a browser, not the category of a serving engine. Both of its upstream calls
(`/v1/realtime`, `/v1/chat/completions`) are the runtime's own endpoints.

Rejected alternative: mounting these routes directly in `http_server.py`. That
file is the audited auth/CORS surface (#510, `utils/auth.py:149-159`); adding a
browser-facing app with its own static assets and its own WS lifecycle to it
widens that surface for no gain, and it would couple the app's restart to the
runtime's.

### 2.2 Answer to design question (a): how topic sessions stay warm

Given §1.3(a)-(e) — no checkpoint API, no pin API, session = replay — the only
honest construction is:

> A "warm topic" is a **prefix that is currently resident in the radix tree**,
> kept resident by periodic priming and by eviction *order*, and whose warmth
> is **measured, never assumed**.

Mechanics:

1. **Prime.** Per topic, the app sends one prefill-only request
   (`max_new_tokens = 0`, legal per `sampling_params.py:207-211`) carrying the
   topic's full context prefix: briefing header + topic section + the rolling
   conversation summary for that topic. It records `primed_tokens` = the prompt
   token count the runtime reports.
2. **Rank.** The same requests carry a high `priority`
   (`protocol.py:464`), which under `--radix-eviction-policy priority` puts
   those nodes late in the victim order (`evict_policy.py:41-47`,
   `radix_cache.py:241`). This is explicitly **not** a pin (§1.3(c)); the
   design does not claim one.
3. **Touch.** A cadence re-issues the priming request per topic, refreshing
   `last_access_time` (the second key of the priority strategy) and re-inserting
   anything already evicted.
4. **Probe — the load-bearing part.** Every real hint request reports
   `usage.prompt_tokens_details.cached_tokens`. The app compares it to the
   topic's `primed_tokens` and derives a verdict WARM / PARTIAL / COLD.
   `None` details are read as **zero cached**, per the `count > 0 else None`
   trap at `usage_processor.py:15`. The topic's displayed state comes from this
   measurement, never from the fact that a prime was sent. This is the
   SUCCESS-CLAIMS-ARE-NOT-EVIDENCE law applied to cache warmth: "we primed it"
   is a success message about state and needs an independent probe.

Trade-offs considered and rejected, each with its excluding predicate:

| Option | Verdict | Excluding predicate |
| --- | --- | --- |
| `#410` checkpoint API | does not exist | no occurrence in `python/` or `docs/` except `ROADMAP_456_matrix_execution.md:36`; `git log --grep='#410'` empty |
| pin the prefix | no such API | `radix_cache.py:598-625` `lock_ref` is in-flight-only; no pin method on `base_prefix_cache.py:244-345` |
| `--enable-session-radix-cache` as a warm-holder | tagging only | `session_radix_cache.py:23-27` "Tagged KV is ordinary LRU radix -- no pinning, no open" |
| `/session_handover` park | opposite semantics | `session_handover.py:327-336` refuses extension of a parked prefix |
| hold an open streaming request per topic to keep `lock_ref` > 0 | rejected on cost | it occupies a scheduler slot and a running sequence per topic for the whole conversation; the residency it buys is the same residency priming+touch buys without occupying the scheduler |

`--enable-session-radix-cache` is still **useful** here, just not as a pin: it
gives a clean bulk release of a topic's KV at close (`scheduler.py:5733-5736`),
which matters when a conversation ends and four topics should stop competing
for pool space. It is recorded as a P3 opt-in, and it drags
`--radix-eviction-policy priority` in with it (`server_args.py:13759`) — which
this design wants anyway for step 2.

### 2.3 Answer to design question (b): the background briefing expander

The expander runs as an ordinary request class **on the same engine**, one
priority tier below the live hints:

* live hint request → `lane: "fast"` (`protocol.py:469`, effective under
  `--enable-fast-lane`, tagged at `scheduler.py:2704-2708`).
* expander request → no lane, default priority tier (`default_priority_value`,
  forced to a concrete `0` at `server_args.py:15074-15075`).

Not #274 (excluded by the executed proof in §1.3(f)), not #347 (excluded by
§1.3(g)). Inside the app the expander is a single asyncio task per session with
a **concurrency of one and a hard cancel point**: when a live hint is due and
an expansion is in flight, the expansion is not aborted server-side (there is
no such contract for a plain completion) but the app stops awaiting it and
drops the result if it lands after the topic has moved on. The cost of a
wasted expansion is bounded by the expander's own `max_tokens`.

The expander's product is written back into the briefing as an *appended,
attributed* section — never an edit of user-written text. A generated section
carries its provenance so the reading user can tell what the model added.

### 2.4 Answer to design question (c): two audio sources in one desktop tab

Target platform is **desktop Chrome** (stated in the briefing), which is the
only place tab audio capture is available.

* Own voice: `getUserMedia({audio: {echoCancellation: true,
  noiseSuppression: true, autoGainControl: true, channelCount: 1}})` — the
  same parameters the translator settled on (`translator/client/
  index.html:242-252`), where echo cancellation is what keeps loudspeaker
  output out of the mic.
* Far side: `getDisplayMedia({video: true, audio: true})`. Chrome requires
  `video` to be requested even when only audio is wanted, and only offers the
  "share tab audio" checkbox for a **tab** share (window shares carry no audio;
  screen shares carry system audio only on Windows/ChromeOS). The app therefore
  must handle `stream.getAudioTracks().length === 0` as a named, explained
  failure — a share without audio is the single most likely user error, and it
  fails silently unless checked. The video track is stopped immediately after
  the stream is obtained; only audio is kept.
* Both streams go through their own `AudioWorklet` → box-average decimation to
  16 kHz → PCM16LE → 20 ms frames, i.e. the translator's proven capture chain
  (`translator/client/index.html:191-394`), duplicated per track rather than
  mixed.
* Frames reach the app over **one** WebSocket, tagged. The translator sends
  bare binary frames because it has exactly one source
  (`translator/server.py:387-394`); with two sources that is ambiguous, so this
  protocol prefixes every binary frame with a 4-byte header
  (`track`, `codec`, `seq16`). This is the one deliberate protocol divergence
  from #466 and it is documented at the encoder
  (`copilot/protocol.py`, `encode_audio_frame`).
* The app fans the two tracks out to **two `/v1/realtime` connections**
  (§1.1), tagging every resulting transcript line `self` or `other`. Track
  identity is carried by the connection, so attribution needs no diarization
  and cannot drift — which is the whole reason for capturing two sources
  instead of one mixed room.
* Named residual risk: the far side's audio also reaches the microphone
  acoustically. Echo cancellation attenuates it but does not remove it, so the
  `self` track can contain a faint echo of the `other` track and produce
  duplicate transcript lines. The mitigation (cross-track duplicate suppression
  on a short time window) is specified as P3 work, not hidden.

### 2.5 Answer to design question (d): latency budget, hearing to hint on screen

The chain, per link, with what is measurable today and what is not:

| # | Link | Budget | Basis |
| --- | --- | --- | --- |
| 1 | capture → 20 ms frame assembled | 20 ms | frame size, client |
| 2 | frame → app (LAN/loopback WS) | < 5 ms | local |
| 3 | app → `/v1/realtime` append | < 5 ms | loopback |
| 4 | utterance boundary decision | 300-800 ms | client-side endpointing; **no server VAD exists** (`realtime/session.py:315-321`), so this is the app's own silence timer and it dominates links 1-3 |
| 5 | ASR chunk inference | UNMEASURED | depends on the adapter and card; `streaming_asr.py` emits deltas per chunk, so a partial arrives before the boundary |
| 6 | hint decode TTFT | UNMEASURED | this is what the warm-topic machinery exists to shorten; the whole point is that the prefix is resident so TTFT is decode-bound, not prefill-bound |
| 7 | render | < 16 ms | one DOM update |

The budget is deliberately incomplete. Links 5 and 6 carry **UNMEASURED**, not
an estimate: no number for either exists in this tree for this workload, and
inventing one would be a desk figure with no falsifier. P2 measures 6 first
(warm vs cold TTFT on the same boot, A-vs-A floor first), because that is the
link the design's central mechanism claims to move — if warm and cold TTFT are
indistinguishable above the noise floor, §2.2 is refuted and the topic registry
should be deleted rather than tuned.

A design consequence of link 4 dominating: hints must be driven by **partial**
transcripts, not only by committed utterances. The app therefore feeds the hint
scheduler on transcription deltas and re-evaluates on commit, rather than
waiting for a clean sentence.

---

## 3. Phase cut, with a falsifier per phase

Every phase names the observation that would prove it wrong. A phase whose
falsifier fires is reported, not repaired silently.

**P1 — protocol, app skeleton, topic state machine, client.**
No GPU, no model, no exposure. Backends are desk fakes that differ from their
real counterparts at named places.
*Falsifier:* the hermetic suite cannot drive a full round trip
(two tracks in → transcript lines out → hint frames out → briefing update) over
the real ASGI stack, or a gate cannot be shown to fail when its condition is
violated.

**P1.5 — the stub backend set, and the app running on it (this change).**
Still no GPU, no model, no exposure. The difference from P1 is that the app is
now USABLE: one narrow seam (`backends.py`), a stub set behind it that speaks a
scripted conversation with human timing and partials, server-initiated events,
and a browser client that renders a continuously updating read pane with the
line→suggestion latency measured on two independent clocks. See §6.
*Falsifier:* the six acceptance items cannot be executed end to end against the
stub set — in the hermetic suite AND in a real browser — or the app's own
latency figure cannot be reproduced by an independent clock.

**P2 — real backends, measured.** Wire the realtime ASR client and the chat
client to a booted runtime; measure warm vs cold hint TTFT with an A-vs-A floor
on the same boot.
*Falsifier:* warm-topic TTFT is not distinguishable from cold TTFT above the
same-boot A-vs-A floor. Then §2.2's mechanism has reach zero on this rig and
the registry is removed, not tuned. (This is the REACH-INCLUDES-PARAMETERS law
applied in advance: a mechanism that never binds at the served geometry is
worth nothing regardless of its correctness.)

**P3 — conversation quality.** Cross-track echo suppression, topic switching
from the live transcript, expander write-back policy, optional
`--enable-session-radix-cache` bulk release.
*Falsifier:* the topic classifier picks a different topic than a human reader
would on a recorded conversation, often enough that the warm prefix is the
wrong one more than it is the right one — in which case warmth buys nothing
because the resident prefix is never the one used.

**P4 — exposure.** nginx template on the #466 pattern
(`scripts/translator/nginx-translator.conf.template`, including its
`location / { return 404; }` catch-all), auth keys, PWA install.
*Falsifier:* the template exposes any route beyond the app's own prefix.

---

## 4. Module map

```
python/sglang/srt/copilot/
  config.py     CopilotConfig — the backend switch, ports, cadences, limits
  protocol.py   frame kinds, event envelope, journal, audio frame codec
  briefing.py   markdown briefing loader (STATUS.md-shaped anchors)
  topics.py     TopicRegistry: prepare / touch / probe, WARM|PARTIAL|COLD
  backends.py   THE SEAM: AsrBackend + HintBackend + SessionPrep protocols,
                and build_backend_set(config) -> the stub set or the rig set
  stubs.py      the STUB set: scripted ASR with partials, canned hints with a
                configured latency, a capacity-bounded prepared-context store
  deskfakes.py  the adversarial UNIT doubles, deliberately degenerate
  asr_client.py /v1/realtime protocol conformance state machine (no transport)
  hints.py      prompt discipline, request builders, ChatHintBackend
  session.py    CopilotSession: two tracks, transcript, hints, expander, and
                the event stream every connection subscribes to
  server.py     FastAPI app: REST + WS (reader + writer) + PWA
  launch.py     python -m sglang.srt.copilot.launch
  client/index.html   PWA: two capture chains, read pane, latency instrument
test/registered/copilot/    hermetic suite, incl. test_acceptance_stub.py
```

## 5. Doubles, and how they are marked

Per the desk-fake law, a fake indistinguishable from the real thing is a trap.
There are two KINDS of double here and conflating them is the trap:

**`deskfakes.py` — adversarial unit doubles.** Deliberately degenerate, so a
component quietly depending on the good case fails a unit test.

* `DeskFakeAsrBackend` — instant, deterministic transcripts. **Named
  difference:** it never produces a `partial` delta stream; one final line per
  commit, the instant `commit` is called, with no relation to how much audio
  arrived. A component that only works because partials arrive passes here and
  starves against `/v1/realtime`.
* `DeskFakeHints` — **Named difference:** it reports `cached_tokens` equal to
  the full primed prefix on *every* call, i.e. always claims a perfect cache
  hit. The residency probe is therefore also tested against
  `DeskFakeHints(always_warm=False)`, which reports the miss in the runtime's
  own absent-details shape.
* `DeskFakePrep` — **Named difference:** `report()` omits the `held` key, which
  models the RIG: a radix prefix is not addressable by topic from outside, so
  there an eviction is only ever discovered by the warmth probe. The session's
  "the backend cannot tell me what it holds" path is exercised by this double,
  the other path by the stub.

**`stubs.py` — the app's stand-in for the rig.** Realistic, which is what makes
it dangerous.

* `StubAsrBackend` — **Named difference:** it speaks a FIXED SCRIPT, gated on
  audio (no audio, no words), and it never REVISES a partial; a real adapter can
  replace the whole partial text of an item.
* `StubHints` — **Named difference:** latency is a configured constant plus
  jitter, so any latency measured against it measures the harness, not a model.
  A cold topic additionally pays `stub_cold_penalty_ms`, and the request that
  paid it leaves the prefix behind, exactly as ordinary traffic populates a
  radix tree.
* `StubSessionPrep` — **Named difference:** it evicts deterministically by LRU
  within its own stated capacity, while the real tree also loses a prefix to
  OTHER tenants at unpredictable times. "Prepared" is stickier here than it will
  ever be on a rig.

Everything either kind produces carries a flag that reaches the browser, which
paints a permanent STUB banner and marks every synthetic transcript line.

---

## 6. P1.5: the seam, the stub set, and server-initiated events

### 6.1 One seam, two implementations, config-only switch

`backends.py` declares three protocols and one factory. The app never imports a
concrete backend; `CopilotConfig.backend` selects the set. A stub reached
through a different code path than the real thing proves nothing about the real
thing, so there is exactly one path.

`AsrStream` is **push-shaped**: deltas and errors leave through an `AsrEvents`
sink rather than being returned from `append`. That is the shape of the real
endpoint — a WebSocket that emits transcription deltas and `error` frames with
no relation to the arrival of any particular audio frame. A request/response
seam would have fit the desk fake and misfit the endpoint, and the mismatch
would only have surfaced on a booted rig.

`--backend rig` REFUSES at launch and names the missing piece rather than
booting an app whose transcript pane can never fill. The chat side of the rig
set is complete; the `/v1/realtime` transport is P2 and deliberately unbuilt.

### 6.2 The session owns an event stream

Everything a reader sees arrives unasked: partials while someone is still
talking, a suggestion when a decode finishes, a briefing addendum on a
background cadence. So `CopilotSession.emit` journals an event and fans it out
to every subscribed connection, and a connection is a reader task plus a writer
task over ONE outbound queue — never two tasks calling `send_text`.

Two consequences are load-bearing:

* A hint decode never blocks the audio path: it runs as its own task, and while
  it is in flight the client has already been told (`hint.pending`). A read pane
  that stops moving must always be able to say whether it is thinking or broken.
* An outbound queue that overflows drops the CONNECTION, loudly, instead of
  falling behind silently. The journal retains more events than the queue holds,
  so the reconnect replays exactly what the dropped connection missed.

### 6.3 The latency instrument, and its two clocks

§2.5 left links 5 and 6 UNMEASURED. P1.5 does not measure them — no model is
involved — but it does close the instrument around them, so that when a model
is wired in, the number has a denominator:

* the SERVER reports `pipeline_ms`, from the moment the app first had the text
  (a final line or a partial, named per hint by `source_kind` and
  `source_line_id`/`source_item_id`) to the moment the suggestion was emitted;
* the CLIENT independently measures from the moment it RENDERED that source to
  the moment it rendered the suggestion, and displays both with their sample
  counts.

Two independent clocks over the same interval is the whole point: they agreed to
about 1 ms in the browser run, which is what makes either of them worth
quoting. A replayed suggestion is excluded from both samples — its source was
rendered milliseconds earlier during the same replay, so sampling it would
report the speed of the recovery and flatter the pipeline after every
reconnect.

### 6.4 Preparing no more than the backend holds

Learned from a browser run against a bounded backend. With five briefing
sections and a capacity of three, preparing every section means each cadence
tick evicts what the previous tick prepared: steady churn, every topic measured
cold, and the prepared-context mechanism reduced to noise. So:

* when a backend STATES a capacity, the app prepares at most that many contexts
  and leaves the surplus honestly unprepared — a switch to one of those costs
  exactly one slow suggestion, which is then measured and displayed;
* the FOCUSED topic is prepared LAST, so it is the most recently used and the
  last to be evicted. Preparing in briefing order systematically evicted the one
  topic the conversation was about;
* a completed prepare clears the previous warmth verdict to UNKNOWN. It still
  never CLAIMS warmth — but leaving a stale COLD in place made the UI report a
  topic as cold immediately after preparing it, which is a claim about a state
  that no longer existed. The cumulative miss count is what carries the history.

A backend that cannot state a capacity (the rig) is treated as UNKNOWABLE, not
as unlimited: nothing is held back, and eviction is discovered by the probe.
