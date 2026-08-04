# DESIGN #488 — TRT lane, talker pool, and the revision guard

Written 2026-08-04, after the precursor measurement
(`/spinning/gpu-battery-results/2026-08-04_488_precursor/RESULTS.md`:
OVERHEAD-BOUND confirmed, 84.7 % launch gap, recoverable factor 6.53x).

Server-side specification. The client halves named in §4 belong to agent 8.

## 1. TRT is the set lane; graphs are the control arm

User directive: build it with TensorRT. Accepted as the target, and the
precursor supports it with numbers rather than deference — kernel time is
**21.85 ms per frame against a 2.2 ms bandwidth floor**. That 10x is kernel
*shape*, not traffic: hundreds of tiny launches each far below peak. CUDA
graphs remove the 84.7 % launch gap and nothing else; only fusion touches the
10x. Expected ladder, to be measured not asserted:

| stage | frame ms | RTF | what it removes |
|---|---|---|---|
| today | 142.8 | 1.71 | — |
| raw loop, no `generate()` | ~110 est. | ~1.3 | 602 syncs/frame |
| + CUDA graphs | ~22 | **~0.26** | the launch gap |
| + TRT fused engines | ~4-9 est. | **~0.05-0.10** | tiny-kernel shape |

**Both arms get built and the crossover is reported either way.** If TRT loses,
that is a result with numbers, not a reason to have skipped it.

### 1a. MEASURED 2026-08-04 — the graphs rung, and what it leaves for TRT

`/spinning/gpu-battery-results/2026-08-04_488_slice2_graphs/`. The table above
was the hypothesis; this is the measurement, all rungs in one window on the
5090 (400 W limit against a 575 W default — a lowered power target, so these
numbers do not compare across power states):

| stage | predicted | **measured** frame ms | **measured** RTF |
|---|---|---|---|
| today | 1.71 | 142.18 | **1.706** |
| raw loop, no `generate()` | ~1.3 | 136.84 | **1.642** |
| + CUDA graphs | ~0.26 | **14.64** | **0.176** |
| + TRT fused engines | 0.05-0.10 | — | not yet built |

Two corrections fall out, and both move the plan:

* **The raw loop alone bought 1.04x, not the estimated ~1.3 RTF.** The 602
  syncs per frame were not the cost. Slice 1's value was *enabling* capture — a
  region that syncs cannot be captured at all — not speed. The estimate is
  corrected on the record rather than dropped.
* **The graphs rung beat its prediction (0.176 against ~0.26)**, because the
  static cache also removes device work the eager path was doing (`DynamicCache`
  concatenations and per-step mask construction): kernel time per frame fell
  21.26 → 13.85 ms, not just the gap.

**What this leaves for the TRT half, stated precisely.** The graphed frame is
**94.6 % kernel** and the launch gap is 4.6-5.9 %. The 6.53x the precursor
called recoverable has been collected in full; there is no overhead left to
attack. What remains is exactly the residual §1 named: **13.85 ms of kernel
against the 2.2 ms bandwidth floor**, ~6.3x of kernel *shape*. So the
0.05-0.10 band stays arithmetically reachable, and fusion is now the **sole**
remaining lever rather than one of two — which raises the value of the TRT lane
relative to when both arms were open.

Engine discipline, carried over from the video chain (#484 family):

* **Two engines**, matching the two graphs already specified in
  `ANALYSE_488_talker_lane_layout.md` §7.3: the **predictor step** (5 layers,
  17-slot static scratch, one head) and the **vocoder chunk**
  (`chunked_decode`, fixed chunk + 25-frame left context). The 28-layer trunk
  step is a third candidate but is captured by graphs first — it is one engine
  per frame against fifteen for the predictor, so the predictor is where fusion
  pays.
* **I/O signature pinned** per engine: name, dtype, shape, and layout, recorded
  next to the engine file. A silent I/O rename is the failure mode that makes a
  wrong-but-running engine.
* **Dynamic-shape profiles**: predictor is fully static (batch 1, seq 1, scratch
  17) — no profile needed, which is why it is the good first engine. The vocoder
  needs a profile over chunk length (min = 1 frame for the tail, opt = the
  chosen streaming chunk, max = 300 to keep `chunked_decode`'s own default
  reachable).
* **sha256 pins + a consumer matrix**: engine file hash, the checkpoint hash it
  was built from, the TRT version, and the arch. An engine is loaded only when
  all four match; a mismatch refuses by name rather than falling back silently.
* **Separate engines per arch**: `sm120` for the 5090, `sm86` for any 3080 pool
  member. TRT engines are not portable across compute capability, and §2 shows
  the 3080s are exactly where the pool wants to go.

## 1b. CORRECTION 2026-08-04 — TRT is built as a fork lane (#337), not a talker side path

§1's engine discipline stands; its *placement* was wrong. The standing user
order is **#337** (static TRT rungs: precompiled engines as a registry in the
fork, granularly selectable, offloadable through the **#407** tier registry,
hot/cold ± compression, graphs-vs-TRT crossover per regime, sm86 honesty) plus
**#469** (RIFE-TRT as a second backend track). The talker engines are the
**first production consumer** of that lane, not a private path beside it.

**The foundation already exists and must be promoted, not duplicated.**
`video_enhance/engine_cache.py` (261 lines) already implements exactly the
discipline §1 listed, and implements it well:

* `EngineKey` = `nvml_uuid | device_name | driver | trt_version | onnx_sha256 |
  precision | ShapeTriplet(min/opt/max) | builder flags` (`:80-111`) — the
  arch-separation §1 asked for is already structural, not a convention;
* `Provenance` manifests with source-artifact url/sha256/bytes/fetched_at
  (`:123-132`), and **an engine file with no manifest is treated as absent,
  not as usable** (`:19-23`);
* atomic writes (`_atomic_write_bytes`, `:247`).

Adjusted plan for the TRT half:

1. **Promote** the engine cache to a fork-level module (`srt/trt_lane/`), with
   `video_enhance` becoming its first *existing* consumer rather than its
   owner. Byte-identical behaviour for the video chain is the acceptance
   condition — this is a move plus a seam, not a rewrite.
2. **Add what #337 needs on top**: a selectable rung registry (granular
   per-consumer engine choice), **#286 ledger registration** so a loaded engine
   is a parkable asset class like every other VRAM tenant, and **#407 tier
   registry** integration for hot/cold placement ± compression.
3. **Crossover as a first-class output**, per regime: graphs vs TRT measured
   and reported, not asserted. §1's ladder is the hypothesis under test.
4. **sm86 honesty**: separate engines per compute capability, and where an
   engine cannot be built or does not pay on sm86, that is reported by name
   rather than silently falling back.
5. The talker's two engines (predictor step, vocoder chunk) are then declared
   as consumers of this lane; RIFE/SR (#469) attach to the same machinery
   afterwards.

Ordering is unchanged: **graphs first** — the 6.53x is on the table and needs
none of this. The TRT half is cut as lane infrastructure from its first line.

## 2. Talker pool — replication across cards, with an honest per-card verdict

The user's proposal composes exactly with the refutation already on record:
chunks are **independent**, so replication is the right axis where tensor
parallelism was the wrong one. Same three cards, opposite decomposition.

### 2.1 Measured instance footprint — not estimated

The precursor run gives this directly. Free VRAM on the 5090 went
**3605 → 903 MiB** while one standalone talker was resident, of which 24 MiB was
the calibration transient:

> **one bf16 talker instance = 2678 MiB** (checkpoint 1745 + CUDA context +
> cuBLAS workspaces + the 12 Hz codec + the speaker encoder).

### 2.2 Fixed-cost table against the 400 MiB corridor

| card | free now | usable (free − 400) | bf16 (2678) | fp8 (~1806) | verdict |
|---|---|---|---|---|---|
| 5090 (sm120), rank 0 + tenant resident | 3605 | 3205 | **fits** (527 spare) | fits | 1 extra instance, bf16 OK |
| 3080 #0 (sm86), INT8 TP rank | 2951 | 2551 | **DOES NOT FIT** (−127) | fits (745 spare) | fp8/TRT only |
| 3080 #2 (sm86), INT8 TP rank | 2951 | 2551 | **DOES NOT FIT** (−127) | fits (745 spare) | fp8/TRT only |

fp8 figure: the checkpoint's 1745 MiB halves to ~873, everything else unchanged
→ ~1806 MiB. It is an estimate; the 2678 is measured.

**The honest refusal, per the brief:** a **bf16** talker does not fit on either
3080 — it misses by 127 MiB, and taking it would breach the corridor on a card
carrying an INT8 serving rank. So the pool is **1 instance today** (the tenant's
own, on the 5090), or 2 if a second 5090 instance is wanted.

**This re-ranks fp8.** It was lever (3), a speed lever whose gain is unreadable
until the gap closes. It is now a **prerequisite for the pool**: without fp8 (or
an equivalently sized TRT engine) the 3080s are unreachable and the pool cannot
exist. That is a stronger reason to build it than the speed argument was, and it
should be scheduled as such.

### 2.2b CORRECTION 2026-08-04 — the 127 MiB is a rebooking, and fp8 is NOT the gate

The user is right that the miss is not a wall: the card is filled by *our own*
server, so the question is which knob hands back 300-400 MiB. §2.2's "fp8 is
the gate" is withdrawn. What replaces it is narrower than the brief hoped,
because two of the named levers do not reach this boot — read at source rather
than assumed.

**#330 runtime VRAM dial: NOT reachable on the running server.** Two
independent blockers:

1. `--enable-vram-dial` is a **boot flag**, and the deployed launch line does
   not carry it (`/proc/3953200/cmdline`, read 2026-08-04). Without it
   `build_kv_capacity_runtime` is never called and `POST /vram_budget` has no
   runtime behind it. So the "no restart" property does not apply *to this
   instance* — enabling it is itself a restart.
2. Even with the flag, the dial requires **weighted uneven DCP**:
   `if not uneven_dcp_active(dcp_size): raise KvCapacityError`
   (`managers/vram_dial.py:1110`), i.e. `--rank-kv-ratio` non-`coupled` or the
   `SGLANG_UNEVEN_DCP` + `_WEIGHTED` env pair. The boot has
   `--rank-tp-ratio auto-performance --rank-gpu-id 0,1,2` and **no
   `--rank-kv-ratio`**, so the predicate is false.

It also refuses under 11 named combinations (`vram_dial.py:1045-1086`), of
which **dual-group lane** is one — worth noting because §4 of ANALYSE_488
already found the lane is not the pool host either.

**The knob that does reach it is already in the launch line.**
`--rank-auto-reserve-mib 13000,3800,3800` is the #332 per-card reserve, and the
two 3080 entries are the two cards in question. Raising them
**3800 → 4200** hands back ~400 MiB per 3080 at the next serving restart, with
no new flag, no new code and no dependence on the dial's DCP precondition.
That is the cheapest correct form of the rebooking.

**The honest price, in tokens.** Served model geometry: 64 layers of which
**16 are full attention** (`full_attention_interval 4`), 4 kv heads, head_dim
256, `--kv-cache-dtype fp8_e4m3` = 1 byte:

> 16 x 4 x 256 x 2 (K+V) x 1 B = **32 KiB per token** on a rank's own shard
> (heads are replicated under the DCP token-shard geometry; the 48 GDN layers
> carry per-sequence state, not per-token KV, and are unaffected).

| rebooked | tokens lost on that rank |
|---|---|
| 300 MiB | **9,600** |
| 400 MiB | **12,800** |

For scale: 12,800 tokens is ~4.9 % of one 262144-token context, on a boot with
`--max-running-requests 4`. The global effect depends on the token vector, so
the per-rank figure is the honest one to quote.

**Consequence for the plan.** bf16 talker instances (2678 MiB) fit on the 3080s
(2551 MiB usable → 2951 after a 400 MiB rebooking) **as soon as the reserve
change ships with the next serving restart**. The pool does not wait for fp8.
fp8 and TRT engine size return to being what they were before this section
overstated them: **performance and quality levers, not existence conditions**.

### 2.2c STATE CHECK 2026-08-04 (post-reboot boot) — the 400 MiB did NOT show up as free VRAM

§2.2b predicted that raising `--rank-auto-reserve-mib` 3800 → 4200 on the two
3080s would take their free VRAM 2951 → 3351 MiB (usable 2551 → 2951) and
thereby admit a 2678 MiB bf16 talker. The reserve change **did** ship: the
running boot carries `--rank-auto-reserve-mib 13000,4200,4200` (read from
`/proc/1236/cmdline`). The free VRAM did not follow.

Measured on the running boot, 2026-08-04:

| card | free | usable (free − 400 corridor) | bf16 talker 2678 | verdict |
|---|---|---|---|---|
| 3080 #0 | **2957 MiB** | 2557 | −121 | **still does not fit** |
| 3080 #2 | **2959 MiB** | 2559 | −119 | **still does not fit** |

That is 6-8 MiB above the pre-change baseline, not 400. The shortfall is
essentially unchanged from §2.2's original −127 MiB, so **the pool gate is not
open** and the claim that it is should not be carried forward.

What this does NOT establish is *why*: this is a different boot (post-reboot,
KV pool 333254 tokens, `--rank-tp-ratio auto-performance`), so a clean
before/after on the reserve knob alone was never taken. Two candidate readings
— the reserve does not convert to free VRAM the way §2.2b assumed, or the pool
sizing reclaimed it elsewhere — and picking between them is the first task of
the pool work, ahead of any instance placement. Until then, item (3) of the
queue is **blocked on 121 MiB**, not unblocked.

Still to check honestly before any fp8 quantisation effort, per the brief: an
sm86 TRT engine at bf16/fp16 may already land under 2551 MiB on its own, in
which case the 3080s are reachable with neither fp8 nor a reserve change. That
is a measurement on the built engine, not a prediction.

Secondary and complementary levers, unchanged from the brief and NOT priced
here because the reserve route is cheaper: `--rank-kv-ratio` shifting KV mass
toward the 5090 at the next restart, #364 GDN idle vacate, #286 parking.

### 2.3 Pool mechanics

* **Chunk queue**, not turn queue: the unit of work is the clause that
  `session.py:1858` already enqueues. No new segmentation.
* **N instances, one claim each.** An instance takes the head of the queue.
* **IN-ORDER ASSEMBLY is mandatory and is the whole risk.** Chunk *n+1* must
  never reach the client before *n*. The queue hands out a monotonic
  `unit_index` (it exists — `session.py:1856`), and a reorder buffer releases
  chunk *n+1* only once *n* has been fully emitted. Out-of-order audio is worse
  than slow audio: it is unintelligible.
* **Reference-voice sharing**: the speaker embedding is computed once per turn
  and is small; it is passed with the work item. Each instance holds its own
  speaker encoder anyway (part of the 2678 MiB), so nothing is shared by
  pointer across processes.
* **Straggler rule**: a 3080 instance is ~2.4x slower than the 5090 on weight
  bandwidth alone. In-order assembly means a slow instance holding chunk *n*
  stalls everything behind it. So the dispatcher must be **head-biased**: the
  fastest free instance takes the lowest outstanding index, never a later one.
  Without that rule the pool can be slower than one instance.

## 3. Chunk policy — always eager, with a revision guard

Final user design: always synthesize clause-wise; a guard detects a
meaning-break against the final MT and repairs it. The self-stabilising
argument is accepted and is implemented **explicitly**, not hoped for.

### 3.1 Context-adaptive clause choice (the mechanism, made explicit)

The insight: when a repair is injected, playback falls further behind the
speaker, and that lag is itself an analysis buffer — later clauses can be
synthesized against more settled text.

Implemented as a rule, not a side effect:

```
lag = (audio queued but not yet played) in seconds
if lag >= LAG_SETTLED_S:      synthesize against the FINAL MT text
elif lag >= LAG_PARTIAL_S:    synthesize against the stream, but only clauses
                              already followed by a further clause
else:                          synthesize the newest clause eagerly
```

The lag regulates aggressiveness by itself: a repair raises the lag, which moves
the next clauses onto the settled path, which makes a second break unlikely.
Both thresholds are server parameters with defaults, exposed for tuning.

### 3.2 The guard — two stages, asynchronous, never in front of playback

Hooked on the **MT-final event** (`session.py:1944`, where `accumulator.flush()`
produces the tail and `mt_total_ms` is stamped), running **beside** playback:

1. **Deterministic first, no LLM call.** If the concatenation of already-spoken
   clauses is a prefix of the final translation (after normalisation:
   whitespace, casing, terminal punctuation), there is nothing to repair. This
   is the overwhelmingly common case and it must cost nothing.
2. **Only on divergence, ask Qwen on the fast lane** whether the difference is
   *material*. Word reordering, a synonym, a moved adverb — not a break. A
   changed subject, a negation, a reversed clause relation — a break. The prompt
   returns a strict verdict token plus the corrected text; anything unparsable
   is treated as "no break" (a false repair is worse than a missed one: it
   interrupts audio the user is understanding).

### 3.3 Repair path

On a material break:

1. **Abort playback for that turn** — cut the server-side chunk queue for the
   turn and signal the client. `stopAll` exists client-side; what is missing is
   a **per-turn abort channel** (§4).
2. **Play the correction prefix from RAM, immediately** — see §3.4.
3. **Re-synthesize the corrected sentences** in parallel with the prefix
   playing, so the repair is audible in ~0 ms and the real content follows
   without a second gap.

### 3.4 Pre-synthesized correction prefix cache

The prefix ("Entschuldige, ich meinte:" / "Perdona, quería decir:") is fixed
text, so it is synthesized **once per (target language × active voice)** and
held as PCM in RAM:

* built as soon as a cloned voice is established, and at boot for each preset
  voice;
* keyed `(voice_id, target_lang)`;
* **invalidated on roster events** — voice switch, voice merge, voice delete.
  Those events already exist (`speakers.py` / the roster path); the cache
  subscribes rather than polling.
* cost: ~1.5 s of 24 kHz float32 mono per entry ≈ 144 KiB. Negligible in RAM,
  and it is RAM, not VRAM, so it does not touch the corridor.

### 3.5 Measurement — observability, not a mode selector

Per the correction: revision rate and repair latency are **measured and
reported**, but they do not drive mode selection. Sentence-final remains a plain
user option with **no recommendation label**. Logged per turn on the existing
decision/turn log: `clauses_streamed`, `prefix_match` (stage-1 outcome),
`guard_invoked`, `guard_verdict`, `repair_latency_ms`, `lag_at_clause_s`.

## 4. Server interfaces for agent 8 (client halves are his)

| interface | server side (mine) | client side (agent 8) |
|---|---|---|
| chunk policy | `chunk_policy: eager \| sentence_final` parameter on the synthesis path, plus `lag_settled_s` / `lag_partial_s` | the selector UI, no recommendation label |
| per-turn abort | new event `turn.speech.abort {turn_id}`; server cuts the turn's chunk queue and stops the in-flight unit | handle the event by dropping queued audio for that `turn_id` only — **not** a global `stopAll` |
| correction turn | `turn.speech` unit carrying `kind: "correction"` and `prefix_cached: true` | render the turn as repaired (visual marker), keep the original text visible struck-through or greyed |
| guard telemetry | the six fields in §3.5 on the turn log | nothing required |

The abort channel is the one genuinely new protocol element; everything else
extends fields on events that already exist.

## 5. Ordering, with the dependency that is easy to miss

1. **Raw 15-step predictor loop** — DONE (slice 1). Worth 1.04x on its own; its
   real job was making the region capturable.
2. **CUDA graphs** — DONE (slice 2), **RTF 0.176, 9.71x**, gates green. Both
   the control arm and, at 94.6 % kernel, the thing that leaves TRT nothing to
   attack except kernel shape.
3. **TRT engines** (predictor, then vocoder) — the set lane.
4. **fp8** — no longer optional: §2.2 makes it the gate on the 3080s.
5. **Talker pool** — needs 4 to reach more than one card.
6. **Revision guard + repair** — independent of 1-5 and can proceed in parallel;
   it is latency *perception*, not throughput.

Streaming synthesis is not a separate item: it falls out of 1, and it must not
ship before RTF < 1 or an intra-clause stream underruns the player.
