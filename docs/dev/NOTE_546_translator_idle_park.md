# NOTE #546 — the translator gives its VRAM back while nobody is talking

Branch `feat/translator-idle-park-546`, base `09db875dbb` (integration/r3-probe-next2).

Status: **desk-complete, GPU-pending.** Everything below is hermetically
tested except the two claims a card must settle — that `nvidia-smi` really
drops by ~5.9 GiB, and what the wake actually costs in milliseconds. §6 is the
GPU step that retires them; nothing in §1–§5 depends on their outcome.

---

## 1. What was wrong

The #466 translator held **~5916 MiB on the RTX 5090 while idle**, with no
compute running, around the clock. Its duty cycle is a conversation at a time
with hours of nothing in between, and it shares the card with the serving
engine — which, per the runbook's coexistence rule (§4.1), had to be sized
against the tenant's *declared* budget (`{asr 3000, tts 4000, diarization 500}`
= 7500 MiB) rather than what it happened to be using. So the idle tenant cost
the serving engine ~135k KV tokens permanently.

User order, verbatim: *"wenn es vorgehalten werden soll, dann soll es
gefälligst den VRAM freigeben und in den Systemram gespillt werden."*

## 2. What carries it — no bespoke spill code

Everything that moves a byte already existed. The slice adds the decision, the
staging and the announcement.

| Layer | Module | What #546 did to it |
|---|---|---|
| Asset class | `model_executor/short_term_offload_register.py` `ASSET_CLASSES["audio_modules"]` (rank `COLD_SECOND_MODEL`, `RECONSTRUCTABLE`, `va_stable_required=False`) | unchanged — the class was already declared and wired by rung B |
| Mover | `translator/ledger.py` `AudioAssetLedger.park` / `.restore` | added park ROUTES for non-torch assets, pinned host copies, a driver-level cache release, need-ranked restore |
| Register accounting | `model_executor/offload_register.py` | added `OffloadRegister.mark_parked` + `maybe_mark_parked` (see §2.1) |
| Decision | `translator/idle_park.py` (new) `IdleParkController` | the WHEN — the only genuinely new policy |
| Announcement | `translator/residency.py` (new) | park/wake events for #553 |
| Recognizer as an asset | `translator/asr_backends.py` `CtranslateWhisperParkRoute` (new) | the ~1.5 GiB the ledger could not see |

The register itself was not re-implemented, not forked, and not bypassed. The
`audio_modules` descriptor is untouched, which is the point: #488's native
lane replaces the BACKEND, and a backend joins by supplying a three-method
route (`park` / `restore` / `size_bytes` + a `device`), not by touching the
state machine. `TestAssetClassProvenance` pins exactly that.

### 2.1 One accounting bug found on the way

`AudioAssetLedger.park` updated its own flags and touched the register's
access clock, but never set the register item's `parked` flag. So
`OffloadRegister._parked_bytes_locked` — what the class fraction cap and every
planner pricing this class read — reported an idle, fully host-resident
translator as fully device-resident. Fixed with `mark_parked`, which is
accounting only: it moves nothing and applies no policy, because for a
`wired=True` class the decision and the movement belong to the owner and only
the books belong to the register. Pinned by `TestRegisterAccounting`.

## 3. The decision: why not a timer

A fixed "idle for N seconds" park is wrong for this tenant in the expensive
direction. Inside a live conversation the gap between requests is *seconds*.
Any N small enough to reclaim memory promptly after a conversation is also
small enough to fire inside one, and every false park costs a park+restore in
front of a person who is mid-sentence.

So the threshold is derived from the traffic, in the shape
`kv_session_offload.SpillTickController` established for a measured control
signal — trailing window, margin as deadzone, dwell as an independent rate
limit:

```
threshold = clamp( max( floor,
                        p95(recent inter-arrival gaps) x gap_margin,
                        break_even x measured_restore_s ),
                   floor, ceiling )
```

| term | default | answers |
|---|---|---|
| `floor_s` | 120 s | "how eager may this get at most?" — binds on a fresh process |
| `p95 x gap_margin` | margin 4.0 | "is this silence unusual FOR THIS CONVERSATION?" |
| `break_even x restore_s` | 20x | "is parking worth it at all?" — self-calibrating on the measured wake |
| `ceiling_s` | 900 s | one long mid-conversation pause may not push the bar out to hours |
| `dwell_s` | 180 s | anti-flap: no park within this long of a wake, whatever the threshold says |

Two subtleties that are load-bearing rather than decorative:

* **Gaps that span a park are not recorded.** The gap that ends a parked
  period is by definition not conversational. Folding it in would raise the
  percentile by the length of the idle period — i.e. the feature would
  disable itself the first time it worked. Pinned by
  `test_the_gap_that_ends_a_park_is_not_a_conversational_gap`.
* **Fewer than `min_gap_samples` (4) gaps leaves the inter-arrival term
  undefined**, and the floor governs. A percentile guessed from two samples is
  how a controller learns the wrong thing confidently.

The margin IS the deadzone, and it is one-sided on purpose: the only
controller-initiated transition is the park. The wake is driven by a request
arriving, not by the signal crossing back, so there is no second edge to
debounce.

### Worked example 1 — live conversation

Gaps 2–9 s, p95 ≈ 8 s, margin 4 → 32 s. Measured restore 0.9 s, break-even 20
→ 18 s. Floor 120 s binds; nothing parks. Note the tenant would not park here
even with the floor at 40 s: the inter-arrival term alone already refuses.

### Worked example 2 — overnight

Conversation ends 23:10. The ring still holds the conversation's gaps, so the
threshold stays at the 120 s floor. 23:12 the sweeper sees 120 s of silence,
no turn in flight, dwell expired → park, ~5.9 GiB back to the driver, one
`park_complete` event. 08:30 the first audio frame arrives: `wake_start` fires
*before* any byte moves, the recognizer comes back first, the turn proceeds.
**One park and one wake for nine hours of idle.**

## 4. The wake: staged, in pipeline need order

A wake needs SPACE, and under #553 that space may first have to be made by the
serving engine spilling something of its own. So the wake is a sequence, never
a monolithic barrier:

1. **`wake_start` is emitted FIRST**, before a byte moves, carrying the
   per-card MiB needed and the ranks still parked. A consumer that has to free
   room can only act while there is still time to act.
2. **Ranks restore in pipeline need order**, and each waiter is released at
   its own rank:

   | rank | assets | why here |
   |---|---|---|
   | 0 | `asr` | the first thing a turn touches |
   | 1 | `talker_trunk`, `code_predictor`, `speaker_encoder` | reachable only after recognition + the MT hop, which is seconds of cover |
   | 2 | `codec` | decodes the talker's output — needed last, and the largest single module (229 MB) |

   The drain gate is `ensure_awake(stage="asr")`, so a turn is released as
   soon as the recognizer is resident while the talker and codec keep
   restoring behind it.
3. **`wake_complete`** follows with the split latency. It is a report, not a
   signal to act on, and is emitted after the waiters are released so a slow
   telemetry consumer cannot sit inside a user's latency.

**What waits for #553:** nothing here asks anyone to make room. In v1 the room
already exists — the co-tenant's reserve was sized against this tenant's
declared budget — so the wake always succeeds immediately. The sequence and
the event contract are already the ones a room-granting consumer needs, so
#553 adds a reaction, not a redesign.

### The residency event contract (`translator/residency.py`)

Log marker `RESIDENCY_EVENT ` + JSON, always written; in-process sinks via
`add_sink`; optional cross-process POST via `--residency-event-url`
(fire-and-forget, daemon thread, 1 s timeout — a tenant must never block a
turn on a telemetry consumer, nor die because one is down).

```
RESIDENCY_EVENT {"tenant_id": "translator", "event": "park_complete",
  "at_s": ..., "total_mib": 5916.0,
  "cards": [{"card_uuid": "...", "nvml_index": 1, "card_name": "NVIDIA GeForce RTX 5090",
             "card_resolved": true, "mib": 5916.0}],
  "detail": {"reason": "...", "freed_mib": 5916.0}}
```

Card identity is the NVML UUID via the ONE identity map
(`registry.nvml.IdentityMap`, #331), never a torch ordinal — the two diverge
on this rig (cuda:0 = 5090 = nvml:1), and an event naming the wrong card would
have a consumer free memory on a card that never needed it. When the map
cannot place the card the event says `card_resolved: false` rather than
guessing.

## 5. How the VRAM is actually released

Two allocators, two mechanisms, one destination.

* **Torch modules** (TTS): the ledger's existing route detaches, copies to
  host and replaces the parameters with meta placeholders. That returns the
  pages to torch's **caching allocator**, where they stay reserved by this
  process — `nvidia-smi` would not move. `release_device_cache()`
  (`torch.cuda.synchronize()` + `empty_cache()`) is what hands them to the
  driver, and `park_all` calls it by default.
* **CTranslate2** (ASR): allocates entirely outside torch, so neither the
  tensor route nor `empty_cache` can reach it. It has its own first-class host
  spill — `Whisper.unload_model(to_cpu=True)` / `load_model(keep_cache=True)`
  / `model_is_loaded` (CTranslate2 4.8.1, verified against the installed
  wheel) — which is exactly the ledger's three-question contract, so the
  recognizer joins the same asset class through `CtranslateWhisperParkRoute`.
  `to_cpu=True` is not optional: a bare unload would make every wake re-read
  1.5 GiB from disk and re-quantize it.

**Why not the #330 VMM decommit.** #330's dial reaches the driver through
`KvVmmArena.decommit_range` (`cuMemUnmap` + `cuMemRelease`) because the memory
it dials is a VMM arena it reserved itself, and chunk-wise unmapping is the
only way to shrink one without moving virtual addresses a captured graph
holds. The translator has no such arena: its weights are ordinary caching-
allocator blocks and nothing captures their addresses (the `audio_modules`
descriptor says so with `va_stable_required=False`). Building a second VMM
layer under weights that never needed one would be the wrong tool. Same
destination — pages with the driver, visible in `nvidia-smi`, allocatable by
another process — reached through the allocator that owns the pages.

**What does NOT come back:** the CUDA context (a few hundred MiB) and
cuDNN/cuBLAS workspaces. Those belong to the process; only exiting it returns
them. Expect ~5.9 GiB → a few hundred MiB residual, not zero. A parked tenant
showing 300–500 MiB in `nvidia-smi` is a successful park.

**Pinned host copies.** The park page-locks its host buffers
(`pin_host_copies=True`, falling back to pageable memory with a warning if the
host is short of lockable pages). The asymmetry is deliberate: pinning is
expensive and happens while nobody waits; the DMA path it buys is cashed in on
the restore, which is the leg a user sits through. The restore's pinned copies
are asynchronous, so it ends with `torch.cuda.synchronize()` — both a
correctness barrier before the first kernel and the reason the measured
latency is a completion time rather than a launch time.

Host RAM cost while parked: ~5.9 GiB of the 98 GB box (~23–25 GB in use).

## 6. GPU step — BOOT-PENDING, the two claims a desk cannot make

Not yet run: the serving-restart window belonged to another agent, and the
translator process (PID 30439) was not to be touched until the operator
confirms. **This section is the acceptance procedure, not a result.**

Vehicle: stop the running translator, start the same command with the park
flags, against the live 30030 backend. Hold `/spinning/gpu-arb/`.

1. **Baseline.** `nvidia-smi --query-compute-apps=pid,used_memory` for the
   translator PID; expect ~5916 MiB. Record.
2. **Park.** Idle it past the threshold (or `--idle-park-floor-s 30` for the
   test). Gate: the PID's `used_memory` drops to the CUDA-context residual;
   free memory on the card rises by ≈ the recorded figure minus that residual;
   one `RESIDENCY_EVENT ... park_complete` in the journal with the 5090's
   NVML index and the MiB.
3. **Wake, split.** One real translation roundtrip against 30030. Record BOTH
   `translator_last_first_serve_ms` (what the user paid) and
   `translator_last_wake_ms` (full stack) from `/metrics`, and the per-asset
   breakdown from `/api/translator/health`. Target: first-serve well under a
   second, full stack under a few seconds. **Report both; a single number
   here would be either flattering or alarmist depending on which one it is.**
4. **Correctness of the roundtrip.** The turn must produce audible, correct
   output — a wake that returns fast and synthesizes noise is a failed wake.
5. **Re-park completeness (addendum 2).** Idle it again past the dwell and the
   threshold. Gate: `used_memory` returns to the SAME parked baseline as step
   2, not merely "some MiB freed". Anything lingering is a leak in the
   park/wake cycle and is the finding, not a rounding error.
6. **Co-tenancy.** Confirm the serving engine on 30030 is unaffected across
   all of the above, and that the freed memory is genuinely allocatable by
   another process (the whole point).
7. **Regression.** `--never-park` boot: `nvidia-smi` must stay at ~5916 MiB
   indefinitely, and the health endpoint must say `never_park: true`.

**No latency figure is claimed until step 3 runs.** The nearest measured
analogue in the tree is #102's graph-state swap at 40–85 ms per ~1 GB
(remap+zeroing dominated) on this rig, which would put ~5.9 GiB somewhere in
the hundreds of milliseconds; that is an extrapolation across a different
mechanism and is written here as context, not as a prediction to be confirmed.

## 7. Config surface

All on `python -m sglang.srt.translator.launch`; defaults are the translator
deployment's defaults.

| flag | default | |
|---|---|---|
| `--idle-park` / `--no-idle-park` | **ON** | the feature |
| `--never-park` | off | hard override, beats `--idle-park` |
| `--idle-park-floor-s` | 120.0 | absolute minimum silence |
| `--idle-park-ceiling-s` | 900.0 | upper bound on the whole threshold |
| `--idle-park-gap-margin` | 4.0 | multiple of the recent-gap p95 |
| `--idle-park-break-even` | 20.0 | multiple of the measured restore |
| `--idle-park-dwell-s` | 180.0 | anti-flap after a wake |
| `--residency-event-url` | "" | optional POST target for park/wake events |

Observability: `/api/translator/health` gains an `idle_park` block (state,
threshold with its terms, recent gaps, dwell remaining, park/wake counts,
split latency). `/metrics` gains `translator_assets_parked`,
`translator_parked_mib`, `translator_idle_seconds`,
`translator_park_threshold_seconds`, `translator_last_wake_ms`,
`translator_last_first_serve_ms`.

`InProcessTtsConfig.park_when_idle` (park after EVERY turn) is superseded and
stays off: it pays a restore inside every conversation — the exact thrash the
inter-arrival threshold exists to avoid — and it bypasses this controller's
state machine. The launcher never sets it.

## 8. Tests

Hermetic, `CUDA_VISIBLE_DEVICES=99`, three new files:

| file | tests | what it pins |
|---|---|---|
| `test/registered/translator/test_idle_park.py` | 67 | the controller: the decision, the state machine, and real byte movement |
| `test/registered/translator/test_idle_park_app.py` | 7 | the WIRING through the HTTP/WebSocket surface |
| `test/registered/translator/test_mt_thinking_off.py` | 9 | §9's rider |

Full run of `test/registered/translator/` plus both offload-register suites:
**694 passed**, up from 666 on the base commit with nothing pre-existing
changed.

The arrival tests drive a FAKE CLOCK. Sleeping through a 120 s threshold is
not a test, it is a delay, and a controller whose only proof is a sleep can
never be tested at the timescale it actually runs at (nine hours of idle).
The two tests whose waiters must be able to TIME OUT use a real clock instead,
because a wait that can only hang is not a can-fail proof.

Groups: arrival patterns (burst conversation never parks / long silence parks
exactly once / stray request wakes once and the dwell holds the re-park),
threshold terms, overrides, the state machine (park/wake ordering, double-park
idempotence, request-during-park queuing, concurrent wakes, driver handoff,
wake timeout, failed-wake attribution, failed-park repair), real byte movement
through the ledger (meta placeholders, bit-identical restore, per-device
accounting, complete re-park), the staged wake (need order, rank-0 early
release while the codec is held hostage, split latency), the park route
(CTranslate2 contract at its real API surface), the residency events, the
register accounting, the config surface, and the #488 provenance pin.

**One defect the app-level file caught that every controller test passed
over**: `prefetch_wake` wrapped the executor Future in `asyncio.create_task`,
which raises `TypeError: a coroutine was expected`. In production that fires
on the first audio frame after a park — i.e. on every wake. Only the
end-to-end turn saw it, which is the whole reason that test drives a real
WebSocket instead of calling the controller.

### Falsifier results (executed, CUDA_VISIBLE_DEVICES=99)

**24 arms, all RED.** Each neuter applied, run, and reverted; the baseline was
re-run green after the batch.

| # | neuter | result |
|---|---|---|
| F1 | inter-arrival term dropped | 2 red |
| F2 | park-spanning gap allowed into the percentile | 1 red |
| F3 | anti-flap dwell dropped | 1 red |
| F4 | break-even term dropped | 1 red |
| F5 | `min_gap_samples` ignored (percentile from 1 sample) | 1 red |
| F6 | ceiling clamp dropped | 1 red |
| F7 | busy probe ignored | 1 red |
| F8 | park re-entrancy allowed | 1 red |
| F9 | failed park claims RESIDENT | 1 red |
| F10 | mid-park caller cannot become the driver | 3 red |
| F11 | rank wait dropped (waiter released before its rank lands) | 13 red |
| F12 | wake staged as one monolithic barrier | 1 red |
| F13 | `wake_start` never emitted | 2 red |
| F14 | residency sink failure propagates out of `emit` | 1 red |
| F15 | wake order back to alphabetical | 1 red |
| F16 | need-order table flattened (`asr` no longer first) | 1 red |
| F17 | register parked-accounting dropped | 1 red |
| F18 | driver cache release dropped from `park_all` | 1 red |
| F19 | `parked_bytes` not cleared on restore (stale figure) | 1 red |
| F20 | recognizer unloaded WITHOUT `to_cpu` | 2 red |
| F21 | route park not idempotent | 1 red |
| F22 | handle without the CTranslate2 API accepted as parkable | 1 red |
| F23 | wake failure not reported to the waiters | 1 red |
| F24 | wake failure inherited by later, unrelated calls | 1 red |
| F25 | wake worker never started | hangs (see below) |

Three notes on how the arms were run, because they changed the tests:

* **F2 was GREEN on the first attempt.** The test called `ensure_awake`
  without the `notify_activity` that the server does first, so the nine-hour
  gap never reached the ring and the guard under test was never exercised.
  Fixed by making the test follow the production call order.
* **F14 was GREEN on the first attempt.** The controller has its own guard
  around `_emit`, so a controller-level test passes even when `emit` itself
  propagates — defence in depth hides which layer defends. Added a test
  against `residency.emit` directly.
* **F19 was GREEN on the first attempt.** `parked_bytes_by_device` filters on
  `parked`, so a stale per-asset figure hides there; it surfaces in the health
  endpoint as "5916 MiB parked" beside a fully resident tenant. The assertion
  now reads the asset.
* **F25 (`if drive:` never starts the worker) HANGS rather than failing.** The
  neuter means no wake ever completes, so every waiting test sits until its
  budget expires and the pytest process wedges on daemon threads. Recorded as
  a hang, not as a pass: F11 and F12 already bite on the same code path from
  the other side (waiters released too early / restore not staged), so the
  path is pinned even though this particular arm cannot report cleanly.

## 9. Rider: MT thinking is now sent explicitly off (same commit)

Not part of the park, carried here because it lands in the same file set and
the same latency argument.

`OpenAiMt` now stamps `chat_template_kwargs.enable_thinking` on **every**
request — the translation path and the name-extractor `ask` path — from
`MtConfig.enable_thinking` (default `False`), instead of relying on the
launcher to inject an `extra_body` key that was simply absent when
`--mt-thinking` was passed.

Why explicit rather than a template default: the default belongs to whichever
chat template the SERVED CHECKPOINT ships, and this tenant does not own that.
The checkpoint behind `--mt-model default` can be swapped at any restart
(runbook §14) and the translator would never notice. A field that is always
on the wire cannot be flipped by a change to someone else's file.

Two independent reasons for the value being `False`, each sufficient alone:
a reasoning model's chain of thought has no marker this stage could reliably
strip and whatever survives is read aloud in the speaker's cloned voice
(measured: Qwen3.6-27B-FP8 answered a de->es request with "Here's a thinking
process: 1. Analyze User Input..." and hit the token limit before producing
a translation); and the #541 A/B put thinking at 2.5x wall and 3.7x tokens
for equal quality, on a path where a person is waiting mid-sentence.

The flag is applied AFTER `extra_body`, so `--mt-extra-body` cannot turn
thinking back on by accident, and it MERGES into any other
`chat_template_kwargs` rather than replacing them. `--mt-thinking` still
opts in — it now sends `true` rather than omitting the key.

Tests: `test/registered/translator/test_mt_thinking_off.py`, 9 tests.

## 10. What this does not do

* No serving-side reaction. #553 owns that; this slice only emits.
* No park under memory PRESSURE from another tenant. The #286 register can
  already plan one (`plan_spill` ranks `audio_modules` at
  `COLD_SECOND_MODEL`), and now its accounting is right, but nothing calls it
  for this class yet. The idle park is time-driven, not pressure-driven.
* No partial park. The tenant parks all its assets or none. Partial spill at
  module grain is expressible (the ledger parks per asset and the wake ranks
  are per asset) but there is no consumer for it: the whole tenant is idle or
  it is not.
* The preset voice pool is host-side audio, not device memory, so it is not a
  ledgered asset and nothing about it changes here.
