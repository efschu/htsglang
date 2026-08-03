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

## 7. Scope of slice 1 (this change)

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
