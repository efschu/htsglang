# ANALYSE #488 — the talker lane, and which layout actually wins

Status: slice 1 (config + model file + geometry) landed and hermetically
proven. The layout verdict below is a **fixed-cost calculation plus executed
geometry**, not a GPU measurement — the cards were held by the #466 translator
serving window for the whole of this session (`/spinning/gpu-arb/holder`:
`session=operator cards=0,1,2 purpose=#466 translator SERVING`). The GPU arm is
specified in §6 and is the next thing to run.

## 1. The task's premise, and where it inverts

The order was: *"cut the talker and compute it on all three cards at once"*,
against an in-process baseline of **RTF 1.23**. The implicit model is that the
talker is compute- or bandwidth-bound, so dividing the work by three divides
the time.

For this workload that premise inverts, and the arithmetic is not close.

## 2. What one second of audio actually costs

Read from the checkpoint, not from the model card
(`/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base`, 478 tensors,
1744.5 MiB total):

| block | bytes | touched |
|---|---|---|
| `talker.model.layers.*` (28 layers) | 840.12 MiB | once per **frame** |
| `talker.model.text_embedding` | 593.50 MiB | prefill only, one row per token |
| `talker.code_predictor.model.layers.*` (5 layers) | 150.02 MiB | once per **residual group** |
| `talker.code_predictor.lm_head.{0..14}` | 4.00 MiB each | one per group |
| `talker.codec_head` | 6.00 MiB | once per frame |
| `speaker_encoder.*` | 9.06 MiB | once per **turn** |

The loop shape is fixed by the architecture and confirmed at the reference
source: `modeling_qwen3_tts.py:1671` runs a full
`self.code_predictor.generate(max_new_tokens=num_code_groups - 1)` **inside**
each trunk decode step, and the trunk's next input is
`codec_hiddens.sum(1) + text_step` (`:1684-1688`) — so the 15 residual codes of
frame *N* are required before frame *N+1* can start. Strictly sequential; there
is no pipeline to exploit between the two.

At the checkpoint's 12 Hz codec (`tokenizer_type: qwen3_tts_tokenizer_12hz`),
one second of audio is:

* 12 trunk steps × ~846 MiB = **9.9 GiB**
* 12 × 15 = 180 predictor steps × ~154 MiB = **27.1 GiB**
* **≈ 37.0 GiB of weight reads per audio-second**, in **192 forward passes**.

## 3. The bandwidth floor, per placement

| placement | effective BW | weight-read time / audio-second | RTF floor |
|---|---|---|---|
| solo 5090 | ~1669 GiB/s | 22.2 ms | **0.022** |
| solo 3080 | ~708 GiB/s | 52.2 ms | **0.052** |

Measured baseline: **1.23**. So between 24x (3080) and 55x (5090) of the
current cost is **not** weight traffic. At 192 forwards per audio-second,
1.23 s / 192 = **6.4 ms per forward** — against a bandwidth term of 0.50 ms
(trunk, 5090) and 0.09 ms (predictor, 5090). The talker is **launch- and
Python-overhead bound at batch 1**, with an HF `generate()` call re-entered
twelve times per audio-second on top.

Corroboration from an independent source in this repo: DESIGN_466 §11.6 records
a warm streaming implementation of the same model class at **RTF 0.16 / 71 ms
first-audio on a 3080** — 3x above the 0.052 floor computed here, which is what
a partially optimised implementation should look like, and 7.7x better than the
current 1.23 on a card with *less* bandwidth than the 5090.

## 4. What TP=3 does to that

TP=3 divides the 37.0 GiB and **multiplies the collective count**. Per trunk
step the sharded trunk needs 2 all-reduces per layer × 28 layers = **56**; the
predictor needs 2 × 5 = **10** per residual group. Per audio-second:

`12 × 56 + 180 × 10 = 2472 all-reduces`, each of a 1024-wide bf16 vector
(2 KiB).

This rig has **no P2P and no NVLink** — every rank pair is PHB, negotiated
PCIe x4/x8/x8 (FEATURE_CATALOG §7, "Rig facts"). A 2 KiB all-reduce there is
pure round-trip latency. Sensitivity, because the exact figure does not change
the verdict:

| per-collective latency | added time / audio-second | TP=3 RTF (incl. sharded BW term) |
|---|---|---|
| 10 µs (optimistic floor) | 24.7 ms | ≈ 0.043 |
| 25 µs | 61.8 ms | ≈ 0.080 |
| 40 µs | 98.9 ms | ≈ 0.117 |

Against **solo 5090 = 0.022**. TP=3 is worse across the entire plausible range,
by 2x at the optimistic end and 5x at the realistic end — and it is worse than a
**solo 3080** (0.052) from ~20 µs upward. The reason is structural and specific
to this workload: 192 steps per audio-second at batch 1 over a 0.6 B model is
the regime where the per-step latency term dominates and the per-step bandwidth
term is already sub-millisecond. This is the PER-FAMILY × PER-PHASE law reading
out: the talker is a **decode-only, batch-1, tiny-model** phase, where TP is at
its worst.

**This is not a "not partitionable" finding.** The geometry works — §5 proves
it by execution. It is a priced trade-off that comes out negative.

## 5. The geometry, executed (not desk-derived)

16 q heads / 8 kv heads / head_dim 128 / intermediate 3072 do **not** divide by
3, so the classic even branch cannot represent this checkpoint at tp=3 at all
(`models/qwen3.py:128`, `assert self.total_num_heads % attn_tp_size == 0`).
The uneven-TP plan can. Run against the fork's own partition functions
(`distributed/utils.py`), reproduced in
`test/registered/unit/models/test_qwen3_tts_talker_lane_488.py`:

| `--rank-tp-ratio` | q heads/rank | kv heads/rank | MLP intermediate/rank |
|---|---|---|---|
| `1,1,1` | `[6, 6, 4]` | `[3, 3, 2]` | `[1152, 960, 960]` |
| `4,3,3` | `[6, 6, 4]` | `[3, 3, 2]` | `[1224, 936, 912]` |
| `5,3,3` | `[8, 4, 4]` | `[4, 2, 2]` | `[1392, 840, 840]` |
| `2,1,1` | `[8, 4, 4]` | `[4, 2, 2]` | `[1536, 768, 768]` |

Note that even the *uniform* vector produces an uneven head split here. The
`--rank-tp-ratio 5,3,3` row is the layout a 5090 + 2×3080 rig would want if the
collective term were free. It is not free, which is the point of §4.

## 6. What to do instead, in priority order

1. **Solo-5090 native lane with CUDA graphs.** Floor RTF 0.022; a first
   implementation landing anywhere near the §3 corroboration (0.16 on a 3080)
   already clears the acceptance criterion by 6x. This is where the 55x of
   overhead lives, and it is the only lever that attacks it.
2. **Use the three cards for CONCURRENCY, not for one utterance.** The real
   three-card win for this workload is one talker instance per card serving
   independent sessions — which directly removes the documented head-of-line
   block (`translator/inprocess_tts.py:124-134`: a turn arriving while another
   synthesizes "pays the whole of that synthesis, a median 4.8 s"). Same three
   cards, same user intent, positive instead of negative.
3. **TP=2 on 5090+3080** is not a middle ground: it keeps 2/3 of the collective
   count and lands the clock on the slower card. Not recommended.

### 6a. Reach check on "#274 lanes carry multi-instance naturally" — they do NOT, yet

Read before acting on it, per MECHANISM REACH. The #274 lane is **not** a host
for three independent talker instances, and the shorthand is wider than the
code in three separate ways:

* `model_executor/dual_group_lane.py:35` — *"the scheduler holds a LIST of
  lanes. **Slice B instantiates exactly one lane.**"* The N-lane form is
  designed for and explicitly not built.
* `dual_group_lane.py:16-25` — a lane is a **full-width (weight-TP=1) module
  tree whose parallel linears are SHELLS** over the serving group's own
  sharded parts plus freshly loaded complements. It reconstitutes *the serving
  model* at full width; it is not a slot a *different* model occupies.
* `server_args.py:9524-9531` — `--dual-group-lane` requires an explicit
  `--rank-tp-ratio` integer list because *"the lane shares the plan's rank-0
  segment; without a plan there is nothing to nest"*, plus a mandatory
  `--dual-group-lane-budget-mib` with no fallback.

The shape #488's "one instance per card" actually needs is #305 multi-model,
which `DESIGN_305_multi_model_serving.md:4-5` states is design-only ("no
implementation") — the same finding the #466 feasibility cut recorded.

What slice 1 does about it, which is the part that costs nothing now: the model
file holds **no process-global state**. Every geometry decision is taken from
the parallel context at construction (`get_parallel()` in each attention
`__init__`), which is exactly the axis #274's context overlays already
substitute per lane. So N instances become a host problem, never a model-file
rewrite. That is the whole of "designed multi-instance-capable" that can be
honestly claimed today.

### 6b. The falsifying window

The GPU arm that would falsify §4 directly, once a window opens: boot the lane
solo on the 5090 and at `--rank-tp-ratio 5,3,3 --rank-gpu-id <5090>,<3080a>,
<3080b>`, measure ms/frame on identical clause lengths (fixed cost + slope
decomposition), against a same-window A-vs-A floor. The prediction under test is
that the TP=3 arm is **slower**, by the §4 table.

## 7. The tenant-module cut, priced against the live turn (2026-08-04)

New measurement from the live tenant: **TTS start → first audio 5511 ms of a
6076 ms turn (91 %)**; ASR 124, diarisation 60, MT first token 240. Over seven
turns TTS ran 4814-7620 ms.

Two desk findings change the shape of the plan, both read from code rather than
assumed.

### 7.1 Clause-wise overlap with the MT stream is ALREADY BUILT

`translator/session.py:1852-1871` already runs a FIFO `speech_worker`: the MT
delta stream is regrouped by `SentenceAccumulator`, each unit is announced
`queued` (`:1858`) and handed to `speak()` (`:1793`), which starts its own
`tts_start` clock per unit. So the 5511 ms is **one clause**, not the turn.

The remaining latency is therefore INTRA-clause, and its cause is named in the
module's own docstring (`inprocess_tts.py:22-27`): *"no true incremental
streaming. The reference generates a whole utterance and then decodes it. We
chunk the finished waveform ... the first-audio latency is whole-utterance, not
first-frame."*

Lever (2) as briefed is thus already discharged at the clause boundary. What is
left is emitting audio DURING a clause — and that is not an independent lever:

### 7.2 Levers (1) and (2) are one piece of work, and the codec already supports it

`core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:886` —
`chunked_decode(codes, chunk_size=300, left_context_size=25)`, with the left
context discarded on emit (`:894`). Incremental vocoding is a supported
operation on this codec today; nothing needs porting.

What blocks it is the same thing that blocks graph capture: the reference drives
the frame loop inside `generate_voice_clone`, so there is no point at which a
caller can (a) replay a captured graph or (b) take partial codes. **One
hand-written decode driver buys both**, and that is the cut:

* build the prompt with the reference's OWN `generate_icl_prompt`
  (`modeling_qwen3_tts.py:1968`) — unchanged, so the hardest-to-verify 1596-line
  surface is not reimplemented;
* prefill eager, once per clause;
* then per frame: trunk graph replay → sample → 15 predictor graph replays →
  sum embeddings → next frame;
* every K frames, `chunked_decode` the accumulated codes with 25 frames of left
  context and yield the new tail.

**Coupling worth stating: streaming only helps once RTF < 1.** At today's 1.23
an intra-clause stream underruns — the player would starve. So the graphs must
land first, or land together. They are not orderable independently.

### 7.3 The two graphs, concretely

| # | graph | shape | contents |
|---|---|---|---|
| 1 | trunk decode step | `(1,1,1024)` embeds in, `(1,1,1024)` out, position as a device tensor | all 28 layers over a **static** KV cache |
| 2 | predictor step | `(1,1,1024)` in, `(1,2048)` logits out, 17-slot static scratch | all 5 layers + one head |

Graph 2 is captured **15 times, once per residual group** — shape-identical, but
each group uses its own `lm_head[g]` and `codec_embedding[g]`, and a graph
cannot switch which `nn.Linear` runs. The alternative (stack the 15 heads and
embeddings into indexed tables, one graph) is rejected on cost: it needs a
120 MiB re-layout of tensors that are already resident, against ~60 MiB of extra
graph pools. Simpler and cheaper to capture fifteen.

**Stays eager**, all of it off the per-frame path: prompt build and prefill
(once per clause, dynamic length), speaker encoder and mel (once per turn), and
`chunked_decode` (once per emitted chunk, and convolutional).

**The one hard constraint**: a captured region may not sync to the host, so the
sampled token must never leave the device. Today's `do_sample=True, top_p=0.9`
goes through HF's warper; it has to be replaced with a hand-written top-p that
consumes a **pre-drawn uniform tensor**. That is the main correctness surface of
the cut, and §7.5 gates it.

### 7.4 VRAM fixed costs on the 5090, measured 2026-08-04

Card total 32607 MiB. Resident: rank 0 (pid 3953294) **22436**, translator
tenant (pid 3954713) **5910**, driver/context remainder ~139. **Free: 3605 MiB.**
The standing corridor keeps 400 MiB free, so the budget for this cut is
**3205 MiB**.

| line item | MiB | note |
|---|---|---|
| static trunk KV cache, 1024 positions | 117.4 | 28 layers x 8 kv x 128 x 1024 x 2 x 2 B; replaces a DynamicCache of the same order at full length, so most of this is not new |
| predictor scratch cache, 17 slots | 0.3 | 5 x 8 x 128 x 17 x 2 x 2 B |
| trunk graph pool | 64 | generous; batch-1/seq-1 intermediates are ~2 MiB, the rest is rounding and workspace |
| 15 predictor graph pools | 60 | ~4 MiB each |
| pre-drawn uniforms for on-device sampling | 0.1 | 2048 frames x 16 groups x 4 B |
| codec chunked decode working set | 0 | a 24-frame chunk is SMALLER than today's whole-utterance decode; counted as zero rather than as the saving it is |
| **total** | **~242** | **7.5 % of the 3205 MiB budget** |

No new weights: the driver reuses the modules the tenant already holds. The
242 MiB is the whole ask, and it leaves ~2963 MiB of corridor untouched.

### 7.5 The quality gate — voice is the product

A listening test is the final gate, not the only one, because "sounds fine" is
exactly the signal that failed this project before (a degenerate encoder scored
0.986 from garbage). The cut admits a **stronger, machine-checkable** gate,
because the graph path and the eager path are the same weights driven two ways:

1. **Code-token identity.** Same prompt, same pre-drawn uniform tensor → the
   graph driver must emit the **identical codec token sequence** as the eager
   reference, all 16 groups, every frame. Not "close": identical. This catches
   the sampling rewrite, the static-cache swap and the graph capture in one
   assertion, before any audio exists.
2. **Waveform equivalence across the chunk seam.** Whole-utterance decode vs
   `chunked_decode` at the chosen chunk size must agree to the codec's own
   left-context tolerance. Falsifier: drop `left_context_size` to 0 and the
   gate must go red at the seams.
3. **Then** the listening arm against the eager path, artefacts under
   `/spinning/gpu-battery-results/`.

Gate 1 is what makes this cut safe to ship quickly; without it the whole thing
rests on an ear.

#### 7.5a CORRECTION 2026-08-04 — gate 1 as written is NOT achievable, and the cut is still right

Measured, not reasoned:
`/spinning/gpu-battery-results/2026-08-04_488_slice2_graphs/`. Gate 1 asked for
**identical** codec tokens against the eager reference. Against that pair,
**23.3 % of tokens differ**. The gate was conflating two changes; decomposed:

| pair | isolates | result |
|---|---|---|
| eager-dynamic vs eager-static | the padded cache + explicit mask | **differs**, 2.4e-2 relative (bf16) |
| eager-static vs **graphed** | the capture | **bit-exact, 0.000000** |

Capture changes nothing. The divergence is entirely the `StaticCache` + 4-D
mask that capture *requires* — and `check_static_cache_semantics.py` settles
what kind of difference that is by rerunning the same pair in float32:
**1.4e-6 – 2.4e-6 relative** against fp32 eps 1.2e-7, i.e. 12-20 eps
accumulated over 28 layers. **Reduction order, not semantics.** An explicit
mask makes sdpa pick a different backend; at bf16 (eps 7.8e-3) a few ulp flips
the code predictor's frequently near-tied argmax.

The gate is split, and the replacement is **stronger** than the original for
what capture can break:

1. **capture gate** — graphed vs eager over the *same* static cache, and it
   must be **bit-exact**. Exactness, not a tolerance.
2. **semantics gate** — static+mask vs dynamic, in fp32, where the question
   "same computation?" is answerable at all.
3. **then** the listening arm, told in advance that the shipped path emits a
   *different valid sample* at bf16, not a degraded one.

The one-sentence lesson: a static cache is a **prerequisite** of graph capture
and perturbs bf16 rounding on its own, before any graph exists. Any future gate
that compares a graphed path to a dynamic-cache reference will fail for this
reason and not for a real one.

### 7.6 fp8 (lever 3), priced — and correctly ranked third

fp8 halves the per-step weight read: trunk 840 → 420 MiB, predictor 150 → 75 MiB,
moving the RTF **floor** from 0.022 to ~0.012 on the 5090. But the floor is not
where we will be: after the graphs cut the path still runs reference modules,
so expect the kernel term to be a minority of the remaining time. fp8 then buys
roughly 10-20 % end to end, not 2x — and it costs a mandatory quality gate on a
0.6 B model whose product IS prosody.

Verdict: correctly third, and **do not start it before the precursor and the
graphs cut have landed** — its gain is unreadable until the overhead gap is
closed, and if the graphs cut reaches the target it may never be needed.

### 7.7 SM contention against rank 0 (lever 5) — already instrumented

No separate instrument needed. The precursor's `calib_gpu_bound` arm IS a
contention meter: a known-kernel-bound region whose gap fraction rises only when
something else holds the SMs. It is already wired as a refusal
(`_MAX_GPU_BOUND_GAP = 0.35`), so every profile run reports the contention
number as a side effect of deciding whether it may testify at all.

## 8. Scope of slice 1 (this change)

Landed and hermetically proven (`CUDA_VISIBLE_DEVICES=99`, 13/13):

* `configs/qwen3_tts.py` — three model_type-keyed configs (#497 canon), with
  the M-RoPE `interleaved` → `mrope_interleaved` gate firing at construction.
* `models/qwen3_tts.py` — uneven-TP-aware trunk, code predictor over a private
  scratch cache (the residual codes must never enter the paged KV sequence),
  per-family embedding split, and a `load_weights` that **refuses** an
  unaccounted checkpoint name instead of dropping it.
* Registry resolution asserted (#497 trap: an unresolved architecture silently
  falls back to `TransformersForCausalLM`).

**Not** in slice 1, and each is a real gate: the scheduler unblock for the
embeds channel (`schedule_batch.py:3006-3008`, DESIGN_466 §11.2), the decode
regime that drives the nested predictor, graph capture over it, the prompt/
embeds builder (1596 lines in the reference), the vocoder, and the native
`/v1/audio/speech` path (`entrypoints/openai/serving_speech.py` is a proxy stub
today, with no lane behind it). DESIGN_466 §11.3 prices the remainder at ~16
days total against the reference implementation; nothing here contradicts that
estimate.
