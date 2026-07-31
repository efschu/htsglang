# #344 — Universal client liveness and cleanup

Status: implemented (#344b) on `feat/universal-liveness-344`. #344a built the
detector for one tenant; this note covers the generalization to every endpoint
class, the per-class timeout flags, and the grace window.

## 1. The problem, restated once

A client that closes its connection is already handled everywhere: Starlette
throws into the response generator and the `finally` runs. The case that was
handled almost nowhere is the client that **neither closes nor reads**. The
socket stays open, the TCP window stays full, the response generator is
suspended at its `yield`, and back-pressure — working exactly as designed —
holds everything behind it: KV blocks, a running-batch slot, a decoder
session, a VRAM lease, a job slot.

From the server's side that is indistinguishable from a very slow consumer.
The only thing separating them is **how long**, which is why the duration is
the policy and why the policy is per endpoint class: a paused film is normal,
a preview tap that has not taken a frame in ten seconds has no viewer.

Two lessons from #339 that generalize and are now written into the shared
component's docstring:

1. **A suspended generator never reaches its own `finally`.** Whatever must
   happen on death has to happen in the release callback. The stream's own
   cleanup path is exactly what is stuck.
2. **Cancelling an executor does not unblock a stalled pipeline.** A producer
   parked in `ring.put` on a full ring only observes the cancel flag after
   that await returns, which on a stalled pipeline is never. The rings have to
   be closed.

**Not this layer.** #312's bounded peer liveness inside collectives watches
*ranks* on NCCL/HTCCL; its failure mode is a hung all-reduce. This watches
*clients* on a socket. They share no code and never interact.

## 2. Audit: every long-lived attachment in the tree

Before / after, honest about the ones where the answer was "nothing".

| # | Attachment | Detected before | Leaked if nothing fired | After |
|---|---|---|---|---|
| 1 | `POST /generate` (stream) | `create_abort_task`: `sleep(2)` then abort, but it runs **after the body ends** — never, for a stalled client | KV blocks + running-batch slot until EOS | `llm_stream` watchdog, abort at 90 s of silence |
| 2 | `/v1/chat/completions` (stream) | same 2 s background task, same hole | same | `llm_stream` watchdog via `OpenAIServingBase.handle_request` |
| 3 | `/v1/completions` (stream) | same | same | same choke point |
| 4 | `/v1/messages` (Anthropic, stream) | same (wraps chat) | same | inherits via chat |
| 5 | `/v1/responses` (stream) | **nothing at all** — no background task, `# TODO: 1. Handle disconnect` in the generator | KV + slot until EOS | `llm_stream` watchdog on `request.request_id` |
| 6 | `POST /api/chat`, `POST /api/generate` (Ollama) | **nothing at all** — no background task, no disconnect check, no `finally` | KV + slot until EOS | `llm_stream` watchdog |
| 7 | `/v1/audio/transcriptions` (chunked ASR) | per-chunk `is_disconnected()` — the best pre-existing pattern in the tree | one bounded chunk request | kept; plus `audio_transcription` watchdog on the stream |
| 8 | `/v1/audio/transcriptions` (full-model stream) | 2 s background task, same hole as #1 | KV + slot | `audio_transcription` watchdog |
| 9 | `/v1/realtime` (WS ASR) | `WebSocketDisconnect` + `async with session_semaphore` — correct | — | unchanged; class `realtime_session` defined, **not wired** (designed-only) |
| 10 | `/v1/images/generations`, `/v1/images/edits` | **nothing** — plain handler, no `is_disconnected()`, 900 s `aiohttp` timeout | diffusion-lane GPU job runs to completion for a client that left | `await_with_liveness` polls disconnect, cancels the lane call, answers 499 |
| 11 | `/v1/audio/speech` | **nothing**, 300 s timeout | speech-lane job, same shape | same treatment, `audio_speech` class |
| 12 | `/v1/embeddings`, `/v1/rerank`, `/v1/score` | nothing; one write, nothing to watch | nothing meaningful | class `embedding` defined with a documented default; **deliberately not wired** |
| 13 | `POST /v1/video/enhance` (#339) | `ConsumerWatchdog`, 300 s — the reference implementation | — | unchanged behaviour; now also declares grace claims |
| 14 | preview tap (`preview_tap`) | class + default exist, **no route exists yet** | — | unchanged; still a policy slot without an endpoint |
| 15 | `/v1/fine_tuning/jobs/{id}/events` (#341-M1) | `ConsumerWatchdog`, 120 s, keepalives | — | unchanged behaviour; import moved, claims declared |
| 16 | Registry ledger lease holders (#305-M1) | lease TTL 120 s, but `_reap_unlocked` needs **lease expired AND pid gone**, and reaping is lazy (only on `read`/`acquire`, no sweeper) | a live-pid tenant with a lapsed lease keeps its bytes — correct, but invisible | class `registry_lease` restated against `DEFAULT_LEASE_SECONDS`; entries now carry `in_grace` (§5) |
| 17 | gRPC bridge streaming (`grpc_bridge.py:293-324`) | abort only on the 300 s back-pressure branch; an **immediately closed** channel returns without calling `abort_request` at all | KV + slot until EOS | **not fixed** — designed-only, §7 |
| 18 | Planner/dashboard SSE (`planner/webui.py` `_sse_stream`) | `BrokenPipeError` on the *next* write only; no timeout, no heartbeat | drives real benchmark load against the target server with nobody watching | class `dashboard_sse` defined; **not wired** (sync `http.server`, not asyncio) — §7 |
| 19 | `/v1/responses` with `background=True` | detached by protocol | — | **deliberately excluded**: fire-and-forget is the documented contract |

## 3. The unified component, and where it lives

`python/sglang/srt/liveness/` — a package, not a module in `video_enhance`.

* `classes.py` — `EndpointClass`, `DEFAULT_TIMEOUTS_S`, `DEFAULT_TIMEOUT_RATIONALE`.
* `watchdog.py` — `ConsumerWatchdog`, `LivenessPolicy`, `LivenessState`,
  `LivenessConfig`, `ConsumerGone`, the process-global config install.
* `grace.py` — `AttachmentRegistry`, `Attachment`, `ResourceClaim`, `ClaimKind`,
  `AttachmentPhase`, the process-global registry.
* `stream.py` — `guarded_stream`, `guard_streaming_response`,
  `guard_generate_stream`, `await_with_liveness`.
* `ledger_bridge.py` — the one wired grace consumer.

**Why `srt/liveness` and not `srt/video_enhance/liveness`.** The consumers are
`srt/entrypoints/*` and `srt/registry/*`, and neither may import from a tenant
package — video enhance is a Class-3 tenant that a text-only deployment does
not install. `sglang.srt.video_enhance.liveness` is now a re-export shim with
the identical public surface, so #339's imports and any external ones keep
working. `EndpointClass` gained members; no existing member, default or
signature changed.

**Why the guards wrap rather than the handlers being rewritten.** Every
long-lived endpoint already returns either an async iterator of frames or one
long await. `guard_streaming_response` swaps `StreamingResponse.body_iterator`
for a wrapper that stamps progress *after* each yield, so a handler gains
liveness in one line and keeps its status code, headers and background tasks.
`OpenAIServingBase.handle_request` is a single choke point for every
OpenAI-shaped streaming endpoint, so #2, #3, #7 and #8 were one edit.

**Why progress is measured after the yield.** Bytes accepted by the transport,
never bytes produced by the chain. A stalled client makes the chain stop
producing, so "the pipeline is idle" is a *consequence* of the stall and
cannot be its evidence.

**Cost on the serving path.** The streaming loop does one `time.monotonic()`
and two integer adds per frame. The watchdog is a separate task that wakes on
`--client-liveness-poll-interval-s` (default 1 s) per attached client. No
liveness code runs inside the scheduler, the model runner or a collective.

## 4. Flags, defaults, and which numbers are guesses

| Flag | Default | Meaning |
|---|---|---|
| `--client-liveness-timeouts` | unset | `<class>=<seconds>,...`; zero or negative disables a class |
| `--client-liveness-poll-interval-s` | `1.0` | watchdog wakeup interval |
| `--client-liveness-teardown-timeout-s` | `30.0` | release budget before a hard cancel |
| `--client-liveness-grace-fraction` | `0.25` | when grace starts, as a fraction of the class timeout |
| `--training-event-stream-timeout-s` | `120.0` | pre-existing (#341-M1); shorthand for `training_events=`, loses to the general flag |

Per-class defaults. **None of these is measured.** Each is an argument about
what silence means for that consumer, recorded verbatim in
`DEFAULT_TIMEOUT_RATIONALE` and asserted non-empty by a test:

| Class | Default (s) | Basis |
|---|---|---|
| `llm_stream` | 90 | judgement — survives a long prefill and a slow link; least patient stream class because KV is the most contended resource |
| `embedding` | 60 | never fires (one write); present so the table is total |
| `video_stream` | 300 | judgement (#339) — a paused player is normal |
| `preview_tap` | 15 | judgement (#339) — a preview has no reason to pause |
| `control` | 60 | judgement — a poller that stopped polling stopped caring |
| `training_events` | 120 | judgement (#341-M1) — costs a queue, not a card |
| `image_generation` | 900 | **derived, not guessed**: equals the image lane's own `aiohttp` total timeout, so liveness never fires before the request it guards |
| `audio_speech` | 300 | derived the same way from the speech lane's timeout |
| `audio_transcription` | 120 | judgement — a couple of chunk intervals |
| `realtime_session` | 60 | judgement |
| `registry_lease` | 120 | **derived**: asserted equal to `ledger.DEFAULT_LEASE_SECONDS` by a test, so the two cannot drift |
| `dashboard_sse` | 60 | judgement |

The `0.25` grace fraction is also unmeasured. A much smaller value would put
healthy but bursty consumers into grace constantly and make the signal
useless; a much larger one leaves too little of the window for a reclaimer to
act in.

## 5. Grace: reclaimable, not pinned

The directive is that during the grace window the bound resources must join
the reclamation ladder rather than sit idle. Three phases:

* `ACTIVE` — the transport accepted bytes recently. Hands off.
* `GRACE` — silent past `grace_after` but not past the timeout. Claims are
  published as **reclaimable**. Nothing is dropped.
* `DEAD` — declared; the release is running. Deliberately **not** reclaimable:
  a second reclaimer racing the watchdog's own teardown would tear the same
  objects down from two directions.

`note_progress()` during grace moves the attachment straight back to `ACTIVE`,
before a reclaimer can act on a claim that is live again.

A claim is `(kind, key, nbytes, tenant_id)` with `kind` in
`vram_lease | kv | job_slot | pipeline | subscriber`. KV claims carry no byte
figure: the block count lives in the scheduler process and asking across the
ZMQ boundary per attachment would put a round trip on the serving path to
improve a reporting field. The claim still answers *which* requests are held.

**Query path.** `global_attachment_registry()` →
`reclaimable()`, `reclaimable_bytes(kind, key)`, `describe()`.

**The one wired consumer: the VRAM ledger.** `ReservationEntry` gained
`in_grace` / `grace_since_ts`; `ReservationStore.set_grace()` writes it;
`CardLedger.grace_bytes` and `render()` report it. `LedgerGraceBridge` is a
registry observer that mirrors phase changes for `vram_lease` claims onto the
ledger file. Chosen over the in-process registry because the ledger is the
**cross-process** view: the arbiter, a second serving process on the same
card, and `registry` CLI output all read the file and none of them can see
this process's Python objects.

The bridge is one-directional and advisory. It writes a flag; it never
releases a reservation, never shortens a lease, never rejects an acquisition.
In particular the lease is untouched on purpose — a tenant in grace is still
running and still holds its device memory, and shortening its lease would let
the reaper hand the same bytes to somebody else while they are in use, which
is exactly the failure `_reap_unlocked` refuses to make. Reclamation stays
with the ladder that owns the policy.

`grace_bytes` is distinct from `waste_bytes`: waste is memory a tenant
declared and has not touched, which it may still need. Grace is memory a
tenant is genuinely using on behalf of a consumer that appears to have left —
the more honest first target under pressure.

Ledger files written before #344 read back as ordinary active entries
(defaulted fields, covered by a test).

## 6. Tests

`test/registered/liveness/test_universal_liveness.py`, 40 tests, hermetic, CPU
only, ~12 s. Plus the pre-existing 22 in
`test/registered/video_enhance/test_liveness.py`.

The three shapes of dead client, each on a real stream, each with a **second
healthy stream on the same event loop** that must keep receiving frames while
the first is declared dead and released — the "nothing blocks unnecessarily"
gate:

* stops reading, never closes;
* closes hard (must **not** trigger the release path — that is the generator's
  own `finally`, and a double teardown is the regression to watch for);
* vanishes mid-stream (consumer task cancelled while parked between frames, so
  the generator stays suspended at its `yield` and nobody will ever close it).

Timing tests use real short durations (400 ms), not a fake clock: the property
under test is *concurrency*, and a clock the test advances by hand cannot show
that two streams progressed independently. Policy arithmetic uses a fake clock
where wall time would buy nothing — including a sweep asserting every one of
the twelve classes can be constructed, started and released at its own
default.

## 7. Designed only, not built

Listed rather than faked.

1. **gRPC bridge immediate-close abort** (audit #17). A closed Rust channel
   returns from `_send_with_backpressure` without calling `abort_request`.
   Same class of leak as `/v1/responses` was, in otherwise mature abort
   infrastructure. The fix is small and belongs in a gRPC-focused change.
2. **Dashboard/planner SSE** (audit #18). `planner/webui.py` is a synchronous
   `ThreadingHTTPServer`, so the asyncio watchdog does not apply to it; it
   needs a thread-shaped variant of the same policy. The class and its default
   exist so the eventual wire has somewhere to read from.
3. **Realtime websocket sessions** (audit #9). Correctly handled by
   `WebSocketDisconnect` today; the open case is a socket that stays open and
   goes silent. Class defined, not wired.
4. **Second and third grace consumers.** Only the ledger is wired. #287's
   pressure staircase and #341's idle tenant can read
   `reclaimable_bytes()` / `CardLedger.grace_bytes` but do not yet act on
   them; acting requires each ladder to decide how a grace-held claim is
   weighed against what wants the bytes, which is their policy, not this
   component's.
5. **Byte figures on KV claims** (§5).
6. **Measured defaults** (§4). Every judgement figure in the table is a
   candidate for a measurement window; none has had one.
