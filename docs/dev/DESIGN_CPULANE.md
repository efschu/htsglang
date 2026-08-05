# DESIGN — cold-expert CPU compute lane

Slice 1 (foundation + proof), branch `feat/cpu-expert-lane`, 2026-08-05.
Measurements in this document were taken on this rig, hermetically, CPU-only,
while it was serving. Load average is recorded with each run.

---

## 0. The prior verdict, and why this branch is not a re-run of it

`ANALYSE_CPU_EXPERT_LANE.md` (2026-08-04) probed this idea and returned **DO NOT
BUILD**. That verdict stands for the variant it tested, and this design does not
overturn it — it takes the branch that analysis explicitly left open.

What it rejected: a lane that **dequantises** a cold expert per event. Measured,
§6a: Int4 -> fp32 dequant of one expert costs **6.177 ms**, against ~0.63 ms for
the H2D fetch it was supposed to replace and ~0.36 ms of available headroom.
Ten times the cost of the thing it was replacing. It also showed why the obvious
escape fails: an fp32 host tier is 129 GB for the 35B class against 98 GB of
RAM, so the masters cannot be pre-dequantised either. Its own closing sentence:

> The DSV4F/K3 vehicle is untouched by §6a, because its natural route is the
> integer kernel of §3 option 2, which has no dequant step. That remains open.

**This design is that route.** The lane computes directly on int8 bytes and
never widens a weight. Two measurements make it a different proposition:

| | fp32 variant (rejected) | int8 variant (this design) |
|---|---|---|
| per-event dequant | 6.177 ms | **none — zero** |
| host tier, 35B class, all 10240 experts | 129 GB (does not fit) | **32 GB** (fits, and is HALF the bf16 pool it displaces) |
| CPU kernel | oneDNN sgemm, 432 GF/s peak | fbgemm AVX2 int8, **1097 GF/s peak** |

The RAM wall that forced the per-event dequant is not a wall at int8: an int8
expert is 3.15 MB against bf16's 6.29 MB. The tier is *cheaper* than the pinned
pool it replaces, which is the fact the earlier analysis did not reach because
it only ever priced fp32.

I also re-measured the int8 -> fp32 widening step, since that is the tempting
shortcut for a batched kernel: **2.831 ms per expert**. Still far above the
entire fetch. The "never widen" rule is therefore load-bearing, not stylistic,
and `test_no_weight_is_widened_to_float` pins it.

---

## 1. Kernel selection — measured, and one earlier reading corrected

Four CPU paths were priced on real expert shapes. The box is an AMD 5950X
(16C/32T, AVX2, **no** AVX-512 and no VNNI), 98.5 GiB RAM, swapless.

Qwen3.5-35B-A3B shapes (h=2048, i=512), 16 threads, rotating DRAM-resident pool:

| path | M=1 | M=64 | verdict |
|---|---|---|---|
| **fbgemm dynamic int8 (W8A8)** | **0.219 ms** | **0.560 ms** (719 GF/s) | the lane |
| torch fp32 (oneDNN sgemm) | 0.507 ms | 1.215 ms (428 GF/s) | 2x slower |
| `aten::_weight_int8pack_mm` (W8A32) | 0.236 ms | 23.4 ms | GEMV only |
| `torch._int_mm` | 0.518 ms | 46.4 ms | unusable |

The fp32 428 GF/s at M=64 reproduces the earlier analysis's 432 GF/s, which is
the harness validating against a number it did not produce.

`torch._int_mm` and `aten::_weight_int8pack_mm` are *not* batched GEMMs on CPU
despite being the obvious-looking int8 entry points; both collapse for M>1. The
usable AVX2 integer GEMM is reached through `quantize_dynamic`, i.e. fbgemm.
No custom kernel and no vendored llama.cpp code is needed for Slice 1.

### The correction

An intermediate reading of mine had W8A32 "roughly free at M=1" and therefore
proposed it as the accurate decode default. **That was an artefact of measuring
one L3-resident expert.** Against a rotating pool that forces DRAM reads — which
is what production does, cycling thousands of experts — W8A32 loses at M=1 too:

| M | W8A8 | W8A32 | ratio |
|---|---|---|---|
| 1 | 0.183 ms | 0.330 ms | 1.8x |
| 4 | 0.196 ms | 1.271 ms | 6.5x |
| 64 | 0.514 ms | 20.292 ms | 39x |

W8A8 is the throughput mode at **every** row count. W8A32 survives only as an
explicit accuracy option, and only at M<=2 where it still undercuts the fetch.
This is exactly the fooling mode the earlier analysis warned about, caught by
its own anti-fooling rule.

---

## 1a. Shape table — read from the shipped configs, not assumed

Every geometry used below was read from the checkpoint `config.json` on this
box. Nothing here is inferred from a model card.

| | Qwen3.5-35B-A3B | Qwen3.5-122B-A10B | DeepSeek-V4-Flash |
|---|---|---|---|
| checkpoint | `Qwen3.5-35B-A3B-GPTQ-Int4` | `Qwen3.5-122B-A10B-GPTQ-Int4` | `...-GGUF/UD-IQ3_XXS` |
| `hidden_size` | 2048 | 3072 | 4096 |
| `moe_intermediate_size` | 512 | 1024 | 2048 |
| routed experts | 256 | 256 | 256 (`n_routed_experts`) |
| `num_experts_per_tok` | 8 | 8 | **6** |
| shared experts | — | — | 1 |
| `num_hidden_layers` | 40 | 48 | 43 |
| `num_nextn_predict_layers` | — | — | **1** (MTP) |
| one expert, bf16 / int8 | 6.29 / 3.15 MB | 18.87 / 9.44 MB | 50.33 / 25.17 MB |

Expert bytes are computed as `3 x hidden x moe_intermediate x dtype_size`
(gate, up, down). The 35B geometry is additionally asserted at test time against
the real checkpoint tensors by `test_shapes_match_the_documented_geometry`, so a
drift breaks the suite rather than silently invalidating these tables.

Note DSV4F routes **6** experts per token, not 8 — the per-token expert-call
counts in §3 use the per-model value, not a shared assumption.

---

## 2. The crossover table (the go/no-go instrument)

`scripts/dev/cpu_expert_lane_microbench.py`. Rotating expert pool >= 400 MB so
weights come from DRAM, warmup discarded, median of 50 reps, load average
recorded. A-vs-A noise floor measured first; nothing below it is reported as a
result.

Streaming side uses the **measured** per-rank H2D rates from
`BENCH_394_v4flash_club3090.md`: 14.42 GB/s (5090 x8), 6.45 GB/s (3080 **x4**),
13.41 GB/s (3080 x8). The x4 rank is the rig's clock at 1.40x the mean.

**Run: threads=8, load average 4.09, noise floor 6.7 %.**

### Qwen3.5-35B-A3B (h=2048, i=512) — expert: 6.29 MB bf16 / 3.15 MB int8
Fetch: 0.436 ms (5090 x8) | 0.975 ms (3080 x4) | 0.469 ms (3080 x8)

| M | W8A8 ms | W8A32 ms | fp32 ms | best | vs x4 link | vs x8 link |
|---|---|---|---|---|---|---|
| 1 | 0.183 | 0.330 | 0.524 | W8A8 | **5.34x** | **2.57x** |
| 2 | 0.187 | 0.662 | 0.402 | W8A8 | 5.21x | 2.51x |
| 4 | 0.196 | 1.271 | 0.474 | W8A8 | 4.98x | 2.40x |
| 8 | 0.224 | 2.587 | 0.490 | W8A8 | 4.35x | 2.09x |
| 16 | 0.238 | 5.192 | 0.644 | W8A8 | 4.10x | 1.97x |
| 32 | 0.354 | 10.808 | 0.700 | W8A8 | 2.75x | 1.32x |
| 64 | 0.514 | 20.292 | 1.091 | W8A8 | 1.90x | **0.91x** |

### Qwen3.5-122B-A10B (h=3072, i=1024) — expert: 18.87 MB bf16 / 9.44 MB int8
Fetch: 1.309 ms (5090 x8) | 2.926 ms (3080 x4) | 1.407 ms (3080 x8)

| M | W8A8 ms | W8A32 ms | fp32 ms | best | vs x4 link | vs x8 link |
|---|---|---|---|---|---|---|
| 1 | 0.337 | 0.857 | 1.348 | W8A8 | **8.68x** | **4.17x** |
| 2 | 0.354 | 1.775 | 1.325 | W8A8 | 8.28x | 3.98x |
| 4 | 0.350 | 3.640 | 1.382 | W8A8 | 8.37x | 4.03x |
| 8 | 0.359 | 7.360 | 1.547 | W8A8 | 8.15x | 3.92x |
| 16 | 0.441 | 15.195 | 1.885 | W8A8 | 6.63x | 3.19x |
| 32 | 0.644 | 30.171 | 2.130 | W8A8 | 4.54x | 2.18x |
| 64 | 0.955 | 59.532 | 2.984 | W8A8 | 3.06x | 1.47x |

### DSV4F-class (h=4096, i=2048) — expert: 50.33 MB bf16 / 25.17 MB int8
Fetch: 3.490 ms (5090 x8) | 7.803 ms (3080 x4) | 3.753 ms (3080 x8)

| M | W8A8 ms | W8A32 ms | fp32 ms | best | vs x4 link | vs x8 link |
|---|---|---|---|---|---|---|
| 1 | 0.819 | 2.412 | 3.789 | W8A8 | **9.53x** | **4.58x** |
| 2 | 0.793 | 4.824 | 4.738 | W8A8 | 9.84x | 4.74x |
| 4 | 0.782 | 10.428 | 3.757 | W8A8 | **9.97x** | **4.80x** |
| 8 | 0.843 | 20.332 | 4.767 | W8A8 | 9.26x | 4.45x |
| 16 | 1.066 | 41.054 | 5.420 | W8A8 | 7.32x | 3.52x |
| 32 | 1.364 | 79.043 | 5.383 | W8A8 | 5.72x | 2.75x |
| 64 | 2.526 | 159.701 | 8.359 | W8A8 | 3.09x | 1.49x |

### Reading the table

1. **The lane wins across the whole practical range, and by more the bigger the
   expert.** The DSV4F class — the one the earlier analysis said had the margin
   and the expensive build — wins by ~10x on the x4 link at decode-to-verify row
   counts. The build turned out not to be expensive, because fbgemm is already
   in torch.
2. **The earlier per-expert-token crossover moved decisively.** The fp32
   analysis put break-even at ~40 tokens/expert on the 35B class. At int8 the
   lane still wins 1.90x at M=64 on the x4 link. The only losing cell in the
   whole table is 35B/M=64 against the *fast* x8 link (0.91x), i.e. small
   experts, wide batches, best link.
3. **Cost is nearly flat in M up to ~8.** The lane is DRAM-bandwidth-bound
   there, not FLOP-bound: reading the expert dominates. This is what makes MTP
   verify batches almost free (§5).
4. **This was measured on a contended box** (load 4-8 from concurrent agents),
   so these are lower bounds on an idle rig, not upper bounds. The earlier
   analysis's anti-fooling note asked for exactly this direction of error.

---

## 3. The aggregate ceiling — the number that decides the feature

Per-expert cost is not the ceiling; **DRAM bandwidth is**. Measured aggregate
throughput with concurrent expert calls (load average 8.59):

| workers x intra-op | Qwen35B calls/s | DRAM | DSV4F calls/s | DRAM |
|---|---|---|---|---|
| 1 x 8 | 3440 | 10.8 GB/s | 878 | 22.1 GB/s |
| **2 x 4** | **5389** | **17.0 GB/s** | 986 | 24.8 GB/s |
| 4 x 2 | 5229 | 16.4 GB/s | 970 | 24.4 GB/s |
| **8 x 1** | 4846 | 15.2 GB/s | **1115** | **28.1 GB/s** |
| 16 x 1 | 4481 | 14.1 GB/s | 1063 | 26.7 GB/s |
| 32 x 1 | 4356 | 13.7 GB/s | 1062 | 26.7 GB/s |

Three consequences, and they shape the whole design:

* **The lane saturates at 2-8 workers.** Past that, more threads make it
  *slower*. It is memory-bound, and this box is dual-channel DDR4.
* **Therefore the lane is cheap to co-host.** It needs 2-8 of 32 hardware
  threads. It does not need to fight the serving process for cores, which was
  the open worry in the earlier analysis's anti-fooling note.
* **Therefore the budget is a rate, not a core count.** ~5400 expert-calls/s on
  the 35B class, ~1100/s on the DSV4F class.

### What that rate buys, honestly

Qwen3.5-35B-A3B has 40 MoE layers x top_k 8 = **320 expert-calls per token** if
every routed expert were cold. At 5389 calls/s that is 59.4 ms/token — so a lane
covering *all* experts would cap the model at ~17 tok/s. **The lane cannot be
the whole MoE path.** It is a partial-coverage device, and the honest framing is:

> per decode step of GPU-trunk duration `T_gpu` ms, the CPU can hide
> `5.389 * T_gpu` expert-calls (35B class) or `1.115 * T_gpu` (DSV4F class)
> **for free**, provided they overlap.

At a 25 ms decode step that is ~135 of 320 expert-calls on the 35B class, i.e.
**~42 % of the routed experts absorbed at zero added latency**. Streaming those
same 135 experts would move 849 MB/token, which at the x4 rank's 6.45 GB/s is
132 ms — five times the whole step. That gap is the feature.

DeepSeek-V4-Flash is the harder case and must be stated separately rather than
folded in: 43 layers x top_k **6** = **258 expert-calls per token**, against a
measured 1115 calls/s. All-CPU would be 231 ms/token, i.e. ~4.3 tok/s. At a
25 ms step the lane absorbs ~28 of 258 calls, **~11 %** — a much thinner slice
than the 35B class's 42 %, because its experts are 8x larger while host DRAM
bandwidth is unchanged.

That is the honest shape of the feature: **the per-expert win is largest on
DSV4F (~10x) while the coverage fraction is smallest there.** The two pull in
opposite directions, and only a Slice-2 end-to-end measurement resolves which
dominates. Per-expert speedup alone must not be quoted as the feature's value.

This is a *ceiling*, not a projection: it assumes perfect overlap and ignores
the transport round trips priced in §4.3. Slice 2 must measure the realised
fraction against it, not restate it.

---

## 4. Architecture

### 4.1 Three-way placement, plugged into #439

`expert_compute_placement.py` (#439) already decides per expert where it comes
from. This lane adds a third category rather than a parallel mechanism:

| category | weights live | computed by | cost driver |
|---|---|---|---|
| GPU-resident (hot) | VRAM slot | GPU grouped-GEMM | VRAM bandwidth |
| RAM-streamed (cold) | pinned host bf16 pool | GPU, after H2D | **PCIe link** |
| **CPU-computed (cold)** | **pinned host int8 tier** | **CPU threads** | **host DRAM** |

The three consume *different* resources — VRAM bandwidth, PCIe, host DRAM — which
is the reason the third category is worth having at all: it is the only one that
adds throughput rather than redistributing it. Under #439's link-proportional
assignment the natural rule is that a rank on a **slow link** should push more
of its cold set to the CPU, because that is precisely where the fetch is most
expensive. On this rig that means the x4 3080 — the rank that is the clock at
1.40x the mean — is the biggest beneficiary. That composition is Slice 2 work;
Slice 1 only makes sure the category is expressible.

### 4.2 Overlap model

The only arrangement that pays is CPU and GPU on **disjoint expert sets of the
same layer**, so the layer costs `max(GPU_hot, CPU_cold)` rather than the sum.
This is inherited unchanged from the earlier analysis §4 and is the design's
starting constraint, not a later optimisation: 40 layers x 0.18 ms of *serial*
CPU time would be 7.2 ms/token of pure addition and would eat the entire win.

The scheduling rule that follows:

> Dispatch the CPU jobs for layer L, then run the GPU's hot experts for layer L,
> then join. Never dispatch and immediately join.

### 4.3 Result transport — payload analysis

The whole economic case, at real shapes. For one layer with `T` tokens:

| | weight streaming (today) | CPU lane |
|---|---|---|
| crosses the link | `n_cold x 3 x h x i x 2` bytes H2D | `2 x T x h x 4` bytes (D2H acts + H2D results) |
| scales with | **number of cold experts** | number of *tokens* |
| 35B, T=1, 8 cold experts | 8 x 6.29 MB = **50.3 MB** | 2 x 8 KB = **16 KB** |
| DSV4F, T=1, 8 cold experts | 8 x 50.3 MB = **402 MB** | 2 x 16 KB = **32 KB** |

Ratio at T=1: **3100:1** (35B) and **12800:1** (DSV4F). Critically, the activation
payload is *per layer, not per expert* — one D2H of `[T, h]` feeds every
CPU-computed expert in that layer, and one H2D returns all their results. Adding
more CPU-computed experts to a layer costs **zero** additional transport.

**The honest cost is latency, not bandwidth.** 40 layers x 2 round trips x ~15 us
of PCIe/sync latency is ~1.2 ms/token, against 640 KB of actual payload that
moves in ~0.1 ms. So the transport term must be engineered as *round-trip count*,
not byte count — which argues for batching the D2H of several layers where the
dependency structure allows, and is a named Slice-2 risk (§9).

### 4.4 The graph seam (#462)

#462's discipline: the captured graph addresses **slots** at fixed device
addresses; the **eager** phase decides what occupies them, materialises the
bytes, and publishes the mapping, all before the replay that reads them.
`BreakableBridge` (`breakable_offload.py:163-182`) holds exactly the pair this
lane needs — `buf`, the fixed-address device buffer the captured segment reads,
and `stage`, its pinned host mirror.

A CPU-computed expert is the same shape with a different **producer**:

| | normal breakable fetch | CPU lane |
|---|---|---|
| what fills `stage` | the expert's weights, read from the host pool | the expert's **output rows**, computed on CPU |
| what `buf` holds at replay | weights in a slot | **results** in a fixed-shape output bridge |
| captured segment then | runs grouped-GEMM over the slot | **reads the result and skips the GEMM** |

`CpuLaneSlotFeed` in `cpu_expert_lane.py` is that contract, testable without a
GPU. It enforces three things:

1. `stage` is CPU/pinned, `buf` is the fixed-address device buffer; allocated
   together, never reallocated — matching `BreakableBridge`'s reasoning about
   the shared-buffer family this fork keeps rediscovering.
2. **All CPU compute completes before `publish()`.** `publish()` before
   `mark_compute_done()` raises, naming the failure mode: a replay reading a
   half-written bridge returns a wrong result, not a slow one.
3. `publish()` is the only writer of `buf` and performs **exactly one** H2D copy
   per layer per step. Double-publish raises.

Note the arena's existing `BreakableScratchOverflow` discipline applies in
reverse here and *relaxes*: a CPU-computed expert consumes no weight slot, so
routing an expert to the CPU lane **reduces** scratch-slot pressure. That is a
second-order benefit worth measuring in Slice 2, not claiming now.

---

## 5. MTP / NEXTN verify batches

A verify step multiplies expert calls: each of `num_draft_tokens + 1` positions
routes independently, so rows-per-expert rises from 1 to `M = num_draft + 1`
(more when draft tokens collide on an expert, which is common — they are
adjacent positions in one sequence).

**The lane handles this almost for free, because it is memory-bound in exactly
that range.** Measured on the 35B shapes:

| M | cost | vs M=1 |
|---|---|---|
| 1 (plain decode) | 0.183 ms | — |
| 2 | 0.187 ms | +2 % |
| 4 (3 draft tokens) | 0.196 ms | **+7 %** |
| 8 (7 draft tokens) | 0.224 ms | +22 % |

A 3-token MTP verify costs the CPU lane 7 % more than a single decode token
while doing 4x the work. On the DSV4F class M=4 is actually the *best* cell in
the table (9.97x vs the x4 link). Speculative decoding and this lane are
complementary rather than competing.

**This only holds for W8A8.** W8A32 grows linearly (0.330 -> 1.271 ms from M=1 to
M=4), so an accuracy-preferring configuration loses the verify-batch advantage
entirely and hands back to W8A8 above 2 rows. `Int8ExpertShard.select_mode`
encodes that, and `ExpertJob.mode` allows a per-job override so a verify step and
a decode step in the same layer can choose differently.

---

## 6. Numerics and the quality gate

**This lane is lossy by arithmetic and must never be default-on.** CPU int8 will
not be bit-identical with the GPU's grouped GEMM. Measured relative error
against an fp32 reference (256-deep reduction, random Gaussian weights):

| mode | M=1 | M=4 | M=33 | band |
|---|---|---|---|---|
| W8A32 (weight-only) | 1.57e-2 | 1.01e-2 | 1.14e-2 | flat in M |
| W8A8 (dynamic) | 3.42e-2 | 3.62e-2 | 6.71e-2 | **grows with M** |

Two findings that must not be buried:

* **W8A32 sits in the fork's already-accepted band.** GPTQ/AWQ-marlin offload is
  intrinsically ~1e-2 and that was accepted; FP8 offload is byte-identical.
  W8A32 belongs to the accepted class.
* **W8A8 is ~3x worse, and degrades with batch width.** The cause is specific and
  fixable: torch's dynamic path picks **one activation scale per tensor**, so a
  wider batch spans a wider range at a coarser step. Per-**token** activation
  scales would flatten this. That is a named Slice-2 item, not a mystery.
  `test_w8a8_error_grows_with_batch_width` pins the effect and simultaneously
  asserts W8A32 does *not* show it, so a future fix has a falsifier.

Per-channel *weight* quantisation was tested and does **not** help
(3.62e-2 -> 3.70e-2 at M=4), confirming activations are the dominant term.

### On real checkpoint weights

Random Gaussian weights are a proxy, not evidence. Repeated against a genuine
expert read out of the shipped `Qwen3.5-35B-A3B-GPTQ-Int4` checkpoint (layer 5,
expert 0, all three projections dequantised once to fp32 as the reference):

| M | W8A32 | W8A8 |
|---|---|---|
| 1 | 0.0149 | 0.0564 |
| 2 | 0.0057 | 0.0221 |
| 4 | 0.0069 | 0.0379 |
| 8 | 0.0046 | 0.0209 |
| 32 | 0.0081 | 0.0478 |

**Real weights quantise better than the random proxy** — W8A32 lands at
0.005-0.015 against 0.010-0.016 for Gaussian. So the bands quoted above are
conservative, which is the right direction for a tolerance. The checkpoint also
independently confirms the geometry these tables assume (hidden 2048,
moe_intermediate 512); `test_shapes_match_the_documented_geometry` fails if that
ever drifts.

The real-weight suite carries the same control as the synthetic one — a
deliberately perturbed expert that the tolerance must reject — plus a
non-degeneracy check on the dequantised reference, because a dequant bug
producing zeros would otherwise make every error test pass trivially.

### Quality-gate plan (must pass before the lane is recommended, Slice 2+)

1. **Unit band** (Slice 1, done): relative error against fp32 reference inside
   the declared band, on random and on real checkpoint weights, with a control
   proving the tolerance rejects a genuinely wrong expert.
2. **Layer-level**: end-to-end MoE layer output, CPU-lane vs all-GPU, on real
   checkpoint weights and real routing, reported as relative error per layer —
   the error compounds over 40 layers and a per-expert band does not bound that.
3. **Token-level divergence**: greedy decode, identical prompt, lane on vs off.
   Report first divergent token index over a corpus. Per the fork's
   hetero-spec-determinism experience this is where a numerics change actually
   shows up.
4. **Task quality**: the fork's standard quality gate at the model level. The
   INT8-W8A8 designation already passed 41/42, so there is a precedent bar for
   an int8 lane to clear.
5. **Composition**: gates 2-4 re-run with MTP armed, since the verify path
   multiplies the number of lossy expert calls per accepted token.

Until gates 2-4 exist, the flag help must state the lane is experimental and
numerics-changing. Per the quality-last rule this lane is built *after*
byte-identical wins, which is why it is opt-in with the lane off by default —
`test_lane_is_off_by_default` pins that.

---

## 7. Flag surface (proposed; wired in Slice 2)

```
--cpu-expert-lane                  off | auto | <fraction>
      Compute a share of COLD MoE experts on the CPU instead of streaming
      their weights host->device. Changes numerics (int8 CPU arithmetic, not
      bit-identical with GPU compute) - experimental, off by default.

--cpu-expert-lane-workers  N       default 4
      Experts computed concurrently. The lane is DRAM-bandwidth-bound and
      saturates at 2-8 on this class of host; higher values measured SLOWER.

--cpu-expert-lane-threads  N       default 4
      Intra-op threads per expert. Low on purpose: at M=1, 4 threads beat 16.

--cpu-expert-lane-precision  speed | accuracy      default speed
      speed    = W8A8, fastest at every batch width, ~3.4-6.7e-2 relative error.
      accuracy = W8A32 while it still beats the link (<=2 rows/expert), ~1.3e-2,
                 the same band as the accepted marlin offload. Costs ~1.8x.
```

`auto` should defer to #439's link-proportional assignment: a rank behind a
slower link earns a larger CPU share, because that is where a fetch costs most.

---

## 8. Honest ceiling per phase

| phase | verdict | basis |
|---|---|---|
| **decode (M=1-2)** | **strong** — 5.3x (35B) to 9.8x (DSV4F) vs the x4 link | §2 |
| **MTP verify (M=4-8)** | **strong** — 4.4x-10x, and only +7 % over plain decode | §2, §5 |
| **prefill (M>=32)** | **weak to negative** — 0.91x-1.9x on the 35B class against a fast link; #254's expert-major wave order already makes streaming win here | §2 |
| **aggregate** | **partial coverage only** — ~42 % of routed experts at a 25 ms step (35B), not the whole MoE path | §3 |

The earlier analysis called this "a decode-only lane, and any build should say
so in its flag help rather than presenting it as a general offload mode." At
int8 the range extends through MTP verify, but the shape of the conclusion is
unchanged and the flag help must still say it.

**Where this rig is weak.** The earlier analysis noted the local Qwen class has
only 0-28 % residual margin after #394, and that remains the honest caveat: the
vehicle with the largest CPU-lane win (DSV4F, ~10x) is not the one the rig runs
day to day. What has changed is that the build is no longer expensive, so the
asymmetry it flagged — "the vehicle with the margin has the expensive build" —
no longer holds.

---

## 9. Slice 2 cut list

Slice 1 delivers the executor, the two kernels, the placement category, the seam
contract, and the measurements. It is deliberately **not wired into the runtime**.
Slice 2, in dependency order:

1. **Host tier construction from real checkpoints.** Build `CpuExpertPool` from
   GPTQ-Int4 / GGUF shards at load time — int4 -> int8 requantisation happens
   **once, at load**, never per event. Needs the GGUF per-expert reader.
   *Risk: load-time cost and peak RAM during conversion.*
2. **Placement integration (#439).** Add the CPU category to
   `expert_compute_placement.py`; implement the link-proportional rule that a
   slower link earns a larger CPU share. *Risk: this is the composition the
   whole feature's value depends on.*
3. **Real bridge binding (#462).** Replace `CpuLaneSlotFeed`'s mock `buf` with
   the real `BreakableOffloadArena` bridge; add the fixed-shape zero-padded
   output buffer and the captured-side mask that skips the GEMM for
   CPU-produced rows. *Risk: highest — it is the only part that touches capture.*
4. **Overlap scheduling.** Dispatch-then-GPU-then-join per layer (§4.2). Prove
   with per-rank timing that CPU time is hidden, not added. *Risk: a wrong join
   point silently serialises and costs 7 ms/token.*
5. **Transport round-trip batching.** §4.3: the cost is ~1.2 ms/token of
   round-trip latency, not bandwidth. Investigate batching D2H across layers.
6. **Per-token activation scales for W8A8.** Removes the batch-width error growth
   (§6) and would narrow the accuracy gap that currently forces the
   speed/accuracy split.
7. **Quality gates 2-5** (§6). Blocking for any recommendation.
8. **MPS/co-existence check.** Confirm 2-8 lane workers do not disturb the
   serving process's CPU-side scheduling.

Explicitly **out of scope** and not to be revisited without new evidence:
per-event dequantisation of any kind (§0), an fp32 host tier (RAM wall), and
`torch._int_mm` / `aten::_weight_int8pack_mm` as batched kernels (§1).

---

## 10. What Slice 1 shipped

* `python/sglang/srt/layers/moe/cpu_expert_lane.py` — `Int8ExpertShard`
  (dual-mode, no dequant), `CpuExpertPool`, `CpuExpertExecutor` (across-expert
  thread pool), `build_jobs`, `CpuLaneSlotFeed` (the #462 seam contract),
  `CpuExpertLaneConfig` (off by default).
* `test/registered/unit/layers/moe/test_cpu_expert_lane.py` — 32 tests, hermetic,
  CPU-only. Includes falsifier controls: a tolerance that provably rejects a
  wrong expert, a pin that no weight is ever widened, a pin on the W8A8
  batch-width degradation, and a pin that the accuracy flag actually reaches the
  kernel rather than merely validating.
* `test/registered/unit/layers/moe/test_cpu_expert_lane_real_weights.py` —
  5 tests against a genuine expert from the shipped GPTQ-Int4 checkpoint, with a
  non-degeneracy guard on the reference and a geometry assertion. Skips cleanly
  when the checkpoint is absent.
* `scripts/dev/cpu_expert_lane_microbench.py` — the crossover instrument, with
  rotating-pool / noise-floor / load-average anti-fooling discipline built in.
