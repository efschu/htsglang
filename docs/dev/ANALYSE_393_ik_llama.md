# ANALYSE 393 — ik_llama.cpp vs. our GGUF stack: what is theirs, what do we take

Desk survey, no card time. Reads the current state of
`github.com/ikawrakow/ik_llama.cpp` (fetched 2026-08-01) against this fork on
`integration/r3-probe-next2`, and against mainline `ggml-org/llama.cpp` where a
feature was since backported.

Builds on and does not repeat: the #372 type inventory (appendix of
`docs/dev/ANALYSE_334_club3090_coverage.md`), the #389 NVMe-tier analysis
(`docs/dev/ANALYSE_389_nvme_expert_tier.md`), and the #66/#72/#73/#163 GGUF
kernel lineage in `python/sglang/srt/layers/quantization/gguf.py` +
`sgl-kernel/csrc/quantization/gguf/`.

---

## 1. Verdict

**ik_llama.cpp is a CPU-inference project with a competent CUDA backend
attached. We are a GPU-inference project. The overlap where they are ahead of
us is almost exactly the overlap where we do not currently compete.**

Three findings carry the whole survey:

1. **Their headline advantage is CPU, and they say so.** Every speedup claim
   in their own announcement threads is AVX2/NEON prompt processing (4.19x,
   6.45x vs. mainline for IQ2_K/IQ3_K on AVX2). Their DeepWiki feature page
   claims "150–350% faster prompt processing" for the IQK matmul engine — on
   CPUs. For pure-GPU serving, community comparisons put mainline llama.cpp
   *ahead* of ik, and neither is our reference point: our #73 tuned K-quant
   MMVQ already beats llama.cpp at TP=2 on sm86. **No GPU kernel technique in
   ik was found that we lack.** Their core IQK-engine trick — "unpack a
   quantized row once and reuse it across columns" — is precisely our #66
   batched MMVQ (`mul_mat_vec_q_nc`, one weight-block load dotted against up
   to 8 activation columns in registers).

2. **The one real type gap is still exactly one type, and the #372 verdict
   survives re-checking.** ik today ships ~29 CUDA-MMVQ-dispatched formats
   that we do not have (`IQ2_K…IQ6_K`, `IQ2_KS/IQ3_KS/IQ4_KS/IQ5_KS`,
   `IQ4_KSS`, `IQ2_KL`, `IQ1_KT…IQ4_KT` trellis, `IQ1_BN/IQ2_BN`, plus `_R4`
   repacks). Of these, exactly one appears in a recipe the user actually
   holds: `IQ4_KS`. The rest are demand-driven, per-type, linear-cost work.
   New information since #372: ik's own July-2026 benchmark thread (#2213)
   finds the trellis `_KT` family **loses to `IQ4_KS` on both quality and
   speed on CUDA for Qwen3.6-27B** — the exact model class we serve. That
   retires trellis as an adoption candidate rather than deferring it.

3. **The strategic item does not resolve the way the framing assumed.** CPU
   compute for cold experts is genuinely better than PCIe streaming on this
   box — but only by ~2.0x, and **~84% of that gap is closable by a placement
   policy change that costs a fraction of the effort.** Host DRAM read
   bandwidth (~38 GB/s) is the invariant floor for *any* RAM-resident expert
   scheme, because a PCIe DMA out of pinned host memory consumes the same DRAM
   reads that a CPU core would. Our current 145 ms/token cold-tier bound is
   not set by "weights must cross PCIe" — it is set by **sharding cold experts
   equally across three PCIe links of unequal width**, one of which is x4.
   Full arithmetic in §7.

**Overall: adopt narrowly and on demand. There is no large, cheap, GPU-side
win sitting in that repo for us.** The valuable output of this survey is not
a port list; it is (a) a confirmed "we are not behind on GPU kernels", (b) one
cheap ergonomic fix they formulate better than we do (`-amb`), and (c) the
DRAM-floor result in §7, which reprioritises #389/#391 work away from the
CPU-lane ambition and toward cold-expert placement.

**License: MIT** (verified at `LICENSE`, `Copyright (c) 2023-2024 The ggml
authors` / `The llama.cpp authors`). Compatible with our Apache-2.0 tree under
the repo-publication rule; any adopted file keeps its MIT header.

---

## 2. Ranked adoption table

Ranked by yield/effort, not by yield. Per the standing rule there is no kill
threshold — cheap small wins are taken.

| # | Item | Yield on our stack | Effort | Verdict |
|---|---|---|---|---|
| 1 | **`-amb`-style memory-budget formulation** for the chunked-prefix/attention scratch knob | Replaces a flat token-count threshold carrying an explicit "design a finer way" TODO with a MiB budget the user can reason about; removes a per-model tuning trap | **S** (one knob, one formula, no kernels) | **Take** |
| 2 | **`IQ4_KS` weight type** (dequant + MMVQ + MoE-vec path) | Unlocks the ubergarm MTP-`IQ4_KS` Qwen3.6-27B build; one more quantization point on a model we already serve six ways | **M** (one type; M not S because of the #109 MMQ-OOB corner at uneven-TP shard boundaries) | **Take on demand** — when the GGUF is actually on disk, read the block layout off the file |
| 3 | **On-demand expert page-faulting at load** (their MoE cold-start feature) | Cuts MoE cold-start latency for storage-resident experts; composes with #89 hibernate and with the #389 NVMe tier if it is ever built | **S–M** (mmap + madvise policy, no kernels) | **Take** when #389 becomes real; standalone value is load-time only |
| 4 | **Regex tensor-placement override** (`-ot` ergonomics, not mechanism) | Our per-expert residency planner is strictly more capable than their regex, but has no declarative user-facing surface. A regex override flag is a UX delta, not a capability delta | **S** | **Optional** — take if a user ever needs to pin a specific tensor family |
| 5 | **Remaining IQK types** (`IQ2_K…IQ6_K`, `IQ2_KS/KL`, `IQ5_KS`, `IQ4_KSS`) | One checkpoint each, none currently on disk | **M per type**, linear | **Defer** — same rule as #372: when a checkpoint arrives |
| 6 | **CPU execution lane for cold experts** (§7) | Bound: 145 ms → 73 ms/token on the cold tier (2.0x). But 145 → 86 ms (1.7x) is reachable by placement alone | **L** (new execution lane, determinism gate, core partitioning under TP, GGUF-MoE offload exclusion must be lifted first) | **Do not build yet** — do item 7 first, re-measure, then decide |
| 7 | *(derived here, not ik's)* **Link-proportional cold-expert sharding** | 145 ms → 86 ms/token bound (1.7x) on the cold tier | **S–M** (placement policy in `ExpertResidencyPlanner`; no kernels, no new lane) | **Take** — highest yield/effort item in this document |
| 8 | **`-ser` smart expert reduction** | Speed for quality; drops experts below the model's `top_k` | **S** | **Back of the queue** — lossy, per the quality-last rule. Note it, do not rank it. |
| 9 | **Trellis `IQ*_KT`** | Negative: their own data has it losing to `IQ4_KS` on quality *and* speed on CUDA for Qwen3.6-27B | **M per type** | **Reject** (see §3.2) |
| 10 | **`_R4/_R8/_R16` repacking + `-rtr`** | Zero on a GPU engine — see §3.3 | n/a | **Not adoptable** (hard reason) |

Items 11+ are "no delta" and live in §8.

---

## 3. Axis 1 — quantization types

### 3.1 What ik has today that we do not

Our supported set (verified in `python/sglang/srt/layers/quantization/gguf.py`
L199–240 and `sgl-kernel/csrc/quantization/gguf/gguf_kernel.cu`):

* 5 legacy (`Q4_0/Q4_1/Q5_0/Q5_1/Q8_0`) and 5 K-quants (`Q2_K…Q6_K`) — full
  MMQ + MMVQ + tiled-MoE + dequant.
* 9 mainline IQ types (`IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
  IQ4_NL, IQ4_XS`) — MMVQ + MoE-vec + dequant, **no tiled MMQ** (deliberate:
  above M=8 we route to dequant+cuBLAS, measured up to 13x faster at 2048
  tokens on a 3080/Q6_K).

ik's `ggml/src/ggml-cuda/iqk_mmvq.cu` dispatches 29 additional formats:
`IQ1_BN, IQ2_BN, IQ1_KT, IQ2_KT, IQ3_KT, IQ4_KT, IQ2_K, IQ3_K, IQ4_K, IQ5_K,
IQ6_K, IQ2_KS, IQ3_KS, IQ4_KS, IQ5_KS, IQ4_KSS, IQ2_KL`, plus `_R4` variants
(`IQ1_S_R4, IQ1_M_R4, IQ2_K_R4, IQ3_K_R4, IQ4_K_R4, IQ4_KS_R4, IQ5_K_R4,
IQ5_KS_R4`). So the KT trellis family and several `_R4` variants **do** have
CUDA MMVQ — the "CPU-only" label often attached to them is wrong at the
kernel level (it is right at the *usefulness* level; see §3.3). `MXFP4` and
`MXFP4_R8` and `Q8_KV` are also theirs, with `MXFP4` now upstream in mainline
too.

**Re-check of the #372 finding against today's repo: it holds.** The type list
grew (`IQ3_KS`, `MXFP4_R8` are new since that inventory) but the artifact
situation did not: still no ik-format GGUF on disk, still exactly one
ik-specific type (`IQ4_KS`) named in a recipe the user keeps. The #372 rule —
"do it when the user asks for that checkpoint, not before, and download the
GGUF first so the block layout is read off the real file" — is unchanged.

The #372 appendix contains one small inaccuracy worth correcting in place if
that file is ever touched: it lists our IQ coverage as including `IQ2_M` and
`IQ3_M`, which are **not** in `IMATRIX_QUANT_TYPES`. The real set is the nine
types listed above.

### 3.2 Trellis (`IQ*_KT`) — reject, on their own evidence

ik discussion **#2213** (30 Jul 2026, "KT (trellis) quants lose to IQ4_KS on
both quality and speed for Qwen3.6-27B (hybrid SSM) on CUDA", 5 replies,
ikawrakow participating): at 4.0 vs 4.25 bpw, `IQ4_KT` was worse than `IQ4_KS`
on this model class, and slower on an RTX 3090. ikawrakow's own explanation:
"the advantage of Trellis quants decreases with bits-per-weight", ambiguous at
4-bit. `IQ4_KT` does beat `IQ4_KSS` at matched 4.0 bpw, but the conclusion in
thread is "if it fits, don't go lower".

We serve Qwen3.6-27B. We have VRAM headroom at 4-bit on the 5090. Both
conditions of their negative result apply to us directly. **Reject**, and
record the reason so it is not revisited without new evidence (candidate for
the Verworfenes-Register).

### 3.3 `_R4`/`_R8`/`_R16` row-interleaving and `-rtr` — not adoptable

Row-interleaved repacking reorders weights so that a CPU SIMD register can
consume four/eight/sixteen rows at once. It is a **CPU data-layout
optimisation with no GPU analogue**: a warp-per-row MMVQ kernel already gets
its coalescing from the block layout, and interleaving rows across the K axis
actively fights it. ik's own documentation states the caveat plainly — avoid
`-rtr` in hybrid setups because K-quants have no CUDA row-interleaved
implementation, and their README warns `-rtr` "can bottleneck hybrid CPU/GPU
inference for MoE models by forcing CPU computation".

**Hard reason for non-adoption: the optimisation targets a compute device we
do not use for weights.** This is the one item whose verdict flips if — and
only if — §7's CPU lane is ever built; at that point `_R4` becomes the correct
storage format for the CPU-resident expert tier, and this row moves from
"not adoptable" to "prerequisite".

---

## 4. Axis 2 — kernel techniques

**Result: no GPU technique found in ik that we lack.**

| ik technique | Our equivalent | Delta |
|---|---|---|
| IQK matmul engine: "unpack a quantized row once, reuse across columns" | #66 batched MMVQ, `mmvq.cuh` `mul_mat_vec_q_nc` — `float tmp[ncols_dst]`, columns unrolled *inside* the block loop, specialised for `ncols_dst` 1..8 | **none** — same idea, ours is the CUDA expression of it |
| Per-type CUDA MMVQ for a large type zoo | Same structure, smaller zoo | type coverage only (§3) |
| Fused MoE `-fmoe` ("fused ffn_up and ffn_gate", default on) | Fused grouped-GEMM MoE / `ggml_moe_a8` tiled + `ggml_moe_a8_vec` | **none** |
| MMQ tile kernels for IQ types | We deliberately have none; above M=8 dequant+cuBLAS wins by up to 13x on our hardware | **not a gap** — a measured design choice, not an omission |
| CPU flash attention, AVX2/AVX512/NEON fused ops | n/a | not our device |

Their advantage is real and it is on the CPU: 150–350% faster prompt
processing vs. mainline llama.cpp on CPU, 4–6x on specific IQK types on AVX2.
Any claim that ik is "faster" must name the workload; for GPU decode and
prefill at bs 1–8, which is our operating point, no such claim was found —
and community GPU-only comparisons run the other way (mainline ahead of ik).

Note for honesty: ik also carries open performance *regressions* against
mainline in some hybrid configurations (their issue #1699 reports ~2x slower
PP, ~1.5x slower TG for Qwen3 MoE `IQ4_XS` CPU-MoE). Their CPU lead is
workload-specific, not uniform.

---

## 5. Axis 3 — MLA modes

**No adoptable delta.** Three reasons, in order of decisiveness:

1. **MLA is upstream now.** Mainline llama.cpp merged MLA for DeepSeek V2/V3
   (PR #12801, April 2025) with Metal/FA follow-ups; DeepSeek V4 landed
   upstream as PR #24162. ik's `-mla 1/2/3` was first, but it is no longer a
   fork-only feature, so "ik has MLA" is not by itself a reason to look at ik.

2. **Their load-time trick is already our load-time trick.** `-mla 2/3`
   amounts to attending over the compressed KV latent rather than a
   materialised K/V, i.e. weight absorption. We do exactly that, once, at load
   time: `deepseek_weight_loader.py:472` (`post_load_weights`, L628–685)
   splits `kv_b_proj` into static `w_kc`/`w_vc` tensors, and the decode path
   (`attention_forward_methods/forward_mla.py`, `forward_absorb_prepare` /
   `forward_absorb_core`) runs MQA over a `kv_lora_rank + qk_rope_head_dim`
   latent cache with `num_kv_heads=1`. There is no per-forward absorption to
   eliminate.

3. **Our DSV4 path is not the same shape as theirs, and is further along.**
   `models/deepseek_v4.py` uses a bespoke `MQALayer` with low-rank query *and*
   output projections, a separate `Compressor` / `C4Indexer` with
   `compress_ratio ∈ {0,4,128}`, and a dedicated `DeepSeekV4SingleKVPool`
   packing nope-FP8 + rope-BF16 with hierarchical compress-state ring buffers.
   ik's DSV4 support (checkpoints, `GGML_OP_LATENT_ATTN`, `dsa_attn.cu`) is a
   ggml-graph reimplementation of the same architecture; nothing in it maps
   onto a gap in ours. We already expose `dsa` and `nsa` attention backends.

One item flagged as *look-later, not adopt*: ik's 1 Aug 2026 commit "Faster
indexer top_k for very long context" is a CPU optimisation of the DSA/lightning
indexer. The *algorithmic* idea behind it may or may not transfer to our
indexer at long context; that would need reading their diff, and it is not a
port.

---

## 6. Axis 4 — runtime features

| ik feature | Our state | Verdict |
|---|---|---|
| `-ot` regex tensor overrides | `expert_offload.py` `ExpertResidencyPlanner` — per-expert residency with hot-set freezing, LRU scratch, wave splitting. Strictly more capable, but with no declarative surface | Adopt the **ergonomics** only (table item 4), S |
| `--cpu-moe` / `--n-cpu-moe N` | `SGLANG_MOE_RESIDENT_EXPERT_FRACTION` + planner is finer-grained (per expert, not per layer) | No delta |
| `-rtr` runtime repack | §3.3 | Not adoptable |
| `-ser` smart expert reduction (`Kmin,t`) | None | Lossy — back of the queue. Cheap to build (a `top_k` clamp with a routing-weight threshold) but it trades quality for speed, so per the quality-last rule it is behind every byte-identical win. Our #390 `expert_stats.py` peakedness metrics (normalised entropy, top-1/8/16/32 share) are, incidentally, exactly the instrument needed to decide whether `-ser` would be safe on a given model — that pairing is worth remembering if it ever comes up |
| `-amb` (max K*Q scratch in MB) | `SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD`, a flat token count, with `forward_mha.py:120` `# TODO: Design a finer way to determine the threshold` | **Take** — see below |
| KV-cache quant defaults (`-ctk/-ctv q4_0/q8_0`, `Q8_KV`) | `--kv-cache-dtype`, fp8 default in our recipes | No delta. This is the axis that produced 84 of the 121 false "type gaps" in #372; it needs no kernels |
| On-demand expert page faulting | Eager load | Take when #389 is real (table item 3) |
| NUMA mirroring (their discussion, in progress) | Single NUMA node on this box (`numactl -H`: 1 node) | Not applicable here; would matter only on a dual-socket successor |

**Why `-amb` is worth the S:** their knob is "the maximum K·Q size in MB I am
willing to tolerate", and the engine derives the chunking from it. Ours is
"switch strategies above N tokens", where the right N depends on head count,
head dim, and rank shard width — all of which differ per model and, under
**uneven TP, per rank**. A MiB budget is invariant across those; a token count
is not. This is the same class of reformulation as choosing MiB over a
fraction for `--rank-gpu-memory-mib`: it removes the "fraction of what?"
ambiguity. It also closes a TODO that is already written into our source.

---

## 7. THE STRATEGIC ITEM — CPU compute for cold experts

The idea, stated fairly: ik/llama.cpp keep RAM-resident experts in RAM and
compute them on the CPU, shipping only activations across PCIe. We keep
RAM-resident experts in pinned host RAM and ship the *weights* across PCIe to
compute them on the GPU. Activations are kilobytes; weights are megabytes.
That asymmetry is the entire argument.

Our design comment states the counter-position explicitly
(`expert_offload.py:60-61`):

> "Cold experts are FETCHED and computed on GPU (this rig's AMD CPU has no
> AMX, so ktransformers-style CPU compute via kt_ep_wrapper is not viable
> here)."

**The arithmetic below shows that reasoning is right for the wrong reason.**
At bs=1 the CPU lane is not compute-bound, so the absence of AMX does not
disqualify it — DRAM bandwidth does the disqualifying, and it disqualifies the
*streaming* path just as hard. The AMX argument becomes correct only above the
batch crossover.

### 7.1 Provenance of the numbers — read this before quoting any of them

The framing figure "~2.8 GB/token worst case for DSV4-Flash, projection band
5–8 tok/s" **is not in this repository.** `ANALYSE_389_nvme_expert_tier.md`
states the opposite in §2/§5/§9: "V4 Flash's exact total/active params and
published quant sizes are not on this box and I did not invent them"; its
concrete numbers (0.106–0.182 tok/s, 9.4 s/token) are a **Kimi-K3 NVMe
analog**, not DSV4-Flash and not the RAM tier. The one measured RAM-tier
anchor in the fork is **122B-A10B at 6.97 tok/s**.

So the model below is built from the *shape* given in the commission —
per-expert 3 × [4096 × 2048] at Q3_K, 258 expert-hits/token — which is
internally consistent and reproduces 2.79 GB/token. It is a **parameterised
model, not a measurement**, and every conclusion is stated as a ratio between
two paths evaluated under the same parameters, so that it survives the
parameters being wrong. Absolute tok/s figures are transfer-bound *ceilings*;
the 6.97 tok/s anchor implies a realised/ceiling ratio around 0.4–0.6 once
attention, dense layers, sampling and Python overhead are paid.

### 7.2 Fixposten

**This box** (`/proc/cpuinfo`, `numactl`, `nvidia-smi`, all read this session):

| Fixpost | Value |
|---|---|
| CPU | AMD Ryzen 9 5950X, 16 cores / 32 threads, Zen 3 |
| ISA | **AVX2 + FMA + F16C only** — no AVX-512, no AVX-VNNI, no AMX |
| L3 | 64 MiB (2 × 32 MiB, one per CCD) — far below the working set |
| NUMA | 1 node |
| RAM | 128 GB installed, 103.3 GB `MemTotal`, **no swap** |
| DRAM peak | dual-channel DDR4-3200 assumed = 51.2 GB/s (SPD hidden by the VM; `dmidecode -t 17` returns nothing) |
| DRAM sustained read | **32–45 GB/s, central 38 GB/s** (62–88% of peak; the low end is what quantized-inference loops actually see) |
| GPU 0 | RTX 3080 20 GB, gen4 **x4** |
| GPU 1 | RTX 5090 32.6 GB, gen4 **x8** |
| GPU 2 | RTX 3080 20 GB, gen4 **x8** |
| PCIe achieved H2D, pinned | x4 ≈ **6.4 GB/s**, x8 ≈ **13 GB/s**; aggregate over all three links ≈ **32.4 GB/s** |

**Workload per decoded token** (the parameterised model):

| Quantity | Derivation | Value |
|---|---|---|
| Params per expert | 3 × 4096 × 2048 | 25.17 M |
| Bytes per expert @ Q3_K (3.4375 bpw) | 25.17 M × 3.4375 / 8 | **10.81 MB** |
| Expert bytes per token | 10.81 MB × 258 hits | **2.79 GB** |
| FLOPs per token (expert FFNs) | 2 × 25.17 M × 258 | **12.99 GFLOP** |
| Arithmetic intensity | 12.99 GFLOP / 2.79 GB | **4.65 FLOP/byte** |
| Activation bytes per token, naive per-hit | 258 × (4096 in + 4096 out) × 2 B | **4.13 MB** |
| Activation bytes per token, per-layer batched | 32 MoE layers × 16 KiB | **0.5 MB** |
| **Weight : activation ratio** | 2.79 GB : 4.13 MB | **675 : 1** |

CPU effective quantized-GEMM throughput on 16 Zen-3 AVX2 cores: fp32 FMA peak
is 16 cores × 32 FLOP/cycle × ~4.0 GHz ≈ **2.05 TFLOP/s**; realised quantized
GEMM lands at **0.5–1.2 TFLOP/s, central 0.8** (anchored on published
llama.cpp `pp512` for 7B Q4_K on this CPU class, uplifted for ik's IQK engine).
Machine balance: 800 GFLOP/s ÷ 38 GB/s = **21 FLOP/byte**. The workload is at
**4.65 FLOP/byte**. At bs=1 this is a **memory-bound** problem by a factor of
4.5 — **compute, and therefore the missing AMX, is not the constraint.**

### 7.3 The three paths

**Path A — today: stream weights, equal expert shards across TP=3.**
Each rank moves 2.79 / 3 = 0.93 GB. The x4-slotted rank sets the clock:

    0.93 GB / 6.4 GB/s = 145 ms/token  →  6.9 tok/s (ceiling)

This reproduces the "5–8 tok/s" band in the commission's framing from first
principles, which is the best available evidence that the band came from an
equal-shard streaming model.

**Path A′ — stream weights, cold-expert shards proportional to link width.**
The three links absorb 32.4 GB/s together:

    2.79 GB / 32.4 GB/s = 86 ms/token  →  11.6 tok/s (ceiling)

**Path B — CPU computes cold experts, only activations cross PCIe.**

    DRAM read:    2.79 GB / 38 GB/s   = 73 ms
    CPU compute:  12.99 GFLOP / 0.8 TFLOP/s = 16 ms   (same loop — not additive)
    Activations:  4.13 MB / 32.4 GB/s = 0.13 ms       (0.18% of Path A's transfer)
    ------------------------------------------------------------------
    token time = max(73, 16) + 0.13   = 73 ms/token  →  13.7 tok/s (ceiling)

### 7.4 The result that changes the priority

**A PCIe DMA out of pinned host memory consumes host DRAM read bandwidth —
the same bandwidth a CPU core would consume reading the same weights.**
Therefore:

> For any RAM-resident expert tier, **73 ms/token (2.79 GB ÷ 38 GB/s) is a
> hard floor that no compute-device choice can beat.** Streaming is bounded by
> `min(DRAM, PCIe)`; CPU compute is bounded by DRAM alone. The CPU lane's
> entire benefit is *removing the PCIe term*, and the size of that benefit is
> exactly how far PCIe sits below DRAM.

On this box:

| Path | Bound | tok/s ceiling | vs. today | Effort |
|---|---|---|---|---|
| A — equal shards (today) | 145 ms | 6.9 | 1.00x | — |
| A′ — link-proportional shards | 86 ms | 11.6 | **1.69x** | **S–M**, placement policy only |
| B — CPU lane | 73 ms | 13.7 | **2.00x** | **L**, new execution lane |
| floor (any scheme) | 73 ms | 13.7 | 2.00x | — |

**A′ captures 84% of the total available gain — (145−86)/(145−73) — for
roughly a tenth of the effort of B.** That is the operative finding of this
section.

Sensitivity, since the two inputs that matter are estimates:

| DRAM sustained | Path B bound | Path A′ bound | B advantage over A′ |
|---|---|---|---|
| 32 GB/s | 87 ms | 86 ms | **0%** (PCIe aggregate ≈ DRAM) |
| 38 GB/s | 73 ms | 86 ms | 15% |
| 45 GB/s | 62 ms | 86 ms | 28% |

Once link-proportional placement exists, the CPU lane's remaining margin is
**0–28%**, and at the pessimistic end of the DRAM band it is **zero** —
because our three PCIe links in aggregate (32.4 GB/s) are of the same order as
this box's DRAM read bandwidth. That is a narrow, uncertain margin for an L
item.

### 7.5 The hybrid, and how small it can be

The two channels can run at once, and the arithmetic says the CPU side does
not have to be large. Total demand is 2.79 GB of DRAM reads per token
regardless of consumer. If PCIe absorbs up to 32.4 GB/s of it and CPU cores
absorb the remainder to reach the 38 GB/s ceiling, the CPU side carries only

    (38 − 32.4) / 38 = 15% of the bytes  ≈ 0.41 GB/token
                                          ≈ 1.9 GFLOP  ≈ 2.4 ms of CPU work

**To reach the DRAM floor on this box you do not need a full CPU MoE engine —
you need one that absorbs ~15% of cold-expert bytes.** That is a materially
smaller thing to build than "CPU expert execution". Caveat: DMA and core
traffic contend in the memory controller, so the two rates do not add cleanly
at saturation; treat 38 GB/s as optimistic when both are running.

### 7.6 Batch behaviour — where the CPU lane loses

At batch B the weight bytes are read once and reused across B tokens, so the
streaming paths are flat in B while CPU compute scales linearly:

    CPU:  max(73 ms, B × 16.2 ms)
    A  :  145 ms (flat)
    A′ :   86 ms (flat)

| B | CPU lane | A (today) | A′ | winner |
|---|---|---|---|---|
| 1 | 73 ms | 145 | 86 | CPU (2.0x / 1.2x) |
| 4 | 73 ms | 145 | 86 | CPU |
| 5 | 81 ms | 145 | 86 | CPU, marginal vs A′ |
| 8 | 130 ms | 145 | 86 | **A′** |
| 9 | 146 ms | 145 | 86 | A′, and A catches up |
| 16 | 259 ms | 145 | 86 | A′ |

**Crossover: B ≈ 9 against today's placement, B ≈ 5 against link-proportional
placement.** Above that, streaming wins on aggregate throughput and the CPU
lane is a latency-only tool. This is where the missing AMX finally bites: an
AMX box would push the crossover out by roughly an order of magnitude, which
is precisely why ktransformers-style CPU MoE is an Intel-server technique. On
this AMD box it is a **bs≤5 technique**, i.e. a single-user-latency technique,
not a serving technique.

The #389 batching table points the same way from the other side: at bs=8 the
union of routed experts already covers 22% of every layer's experts, so
batching saves only 11% of the reads — the weights get read almost regardless,
which is exactly why the streaming paths are flat in B.

### 7.7 What it would take in our stack, and what breaks

**Where it would go.** A CPU lane belongs inside `MoEExpertOffloadCache`
(`python/sglang/srt/layers/moe/expert_offload.py`) as a third disposition
alongside "resident" and "fetch to scratch": *compute in place, host-side*.
The wave planner already resolves, per forward, exactly which experts are cold
and which token rows route to them; that is the input a CPU lane needs, and it
already exists. The `expert`-major wave order (#254, `SGLANG_MOE_OFFLOAD_WAVE_ORDER=expert`)
is the right substrate, since it already guarantees each cold expert is touched
once per forward and already defers a `combine_topk_partials` reduction — a
CPU-computed partial slots into that reduction the same way a GPU-computed one
does.

**Prerequisite that is not optional:** `_OFFLOAD_UNSUPPORTED_QUANT_METHOD_NAMES`
(`expert_offload.py:511-520`) hard-aborts for `GGUFMoEMethod`. GGUF is the
*natural* format for a CPU lane — ggml block layouts are what ik's CPU kernels
consume, and MIT-licensed `iqk_mul_mat` code would drop straight onto them —
but GGUF MoE is precisely the path currently excluded from the offload cache.
**Lifting that exclusion is on the critical path and is itself non-trivial**
(the documented reason is that `GGUFUninitializedParameter` only materialises
at post-load time and per-expert slicing was never validated against GGUF's
quant layout).

**CUDA graphs — survivable, not free.** A host round trip per MoE layer means
~32 graph breaks per token. Our fork already has the machinery
(`eager_on_graph`, `bcg_*` breakable graphs, `tc_piecewise_cuda_graph` in
`deepseek_v4.py`), and 32 × ~20 µs ≈ 0.6 ms is under 1% of a 73 ms token. The
Stage-3 capturable offload path (`prepare_capturable`, fixed-shape device-side
planning) would however **not** cover the CPU lane — it exists specifically to
avoid host syncs, and a CPU lane reintroduces them by construction. So the CPU
lane and `SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1` are alternatives, not companions.

**Determinism — the real blocker.** Two distinct problems:
1. A CPU GEMM has a different reduction order than the CUDA kernel, so a
   CPU-computed expert is not bit-identical to the same expert computed on
   GPU. Under our hetero-determinism rules this is a byte-identity break.
2. Worse, *which* experts land on the CPU depends on residency/LRU state,
   which depends on request history — so the numeric profile of a given token
   varies run to run. That is the "self-determinism" failure mode, not merely
   a precision-class change.
   **Mitigation:** freeze the hot set (`SGLANG_MOE_HOTSET_FILE`, which the code
   already supports and already requires for graph capture) so the CPU/GPU
   split is static and history-independent. Then (1) remains — a fixed,
   reproducible numeric profile that differs from the pure-GPU path. That is
   gateable, but it is a gate, and it belongs behind the byte-identical wins.

**Uneven TP and DCP — a shared resource that does not shard.** PCIe shards
three ways; **DRAM does not.** All three rank processes contend for one 38 GB/s
memory controller and one pool of 16 cores. Consequences:
* The CPU lane's benefit does *not* scale with TP degree, while the streaming
  path's cost does divide by it — this is already visible in the A vs. A′ vs. B
  table and is why B's margin is narrow.
* Cores must be partitioned per rank explicitly, or three ranks each spawning
  16 OpenMP threads will thrash. Under **uneven** TP the shards differ in size,
  so the core split must follow the shard split — and the slowest rank sets the
  clock (the standing "langsamster-Rang-Taktgeber" effect applies unchanged).
* Those same cores currently run the scheduler, detokenizer and per-step Python.
  On a swapless box, saturating 16 cores is a real risk to step launch latency
  — a second-order cost the arithmetic above does not price.

**NVMe interaction (#389): the CPU lane does not help there at all.** If cold
experts live on NVMe, the binding constraint is 1.8 GB/s measured cold-read on
this box — 2.79 GB/token = 1.55 s/token — and it makes no difference whether a
CPU core or a GPU consumes those bytes. The CPU lane is a **RAM-tier**
technique only. Given ANALYSE_389's decision rule ("≤ ~150 GB of 4-bit weights
fits the existing 72 GB VRAM + 98 GB RAM tier"), that is the tier that matters
for anything we can actually run here.

### 7.8 Recommendation

1. **Build link-proportional cold-expert sharding (table item 7).** S–M,
   placement policy inside `ExpertResidencyPlanner`, no new lane, no
   determinism question, no CPU contention, and it composes with everything.
   Bound: 1.69x on the cold tier. It also fixes an architectural mismatch we
   would otherwise carry forever: cold-expert shards are currently sized by
   VRAM/compute capacity, but their cost is paid in *link width*, and on this
   box those two orderings disagree (the x4 slot holds a 20 GB card).
2. **Then re-measure**, with #390's `expert_stats.py`, on a real DSV4-Flash or
   122B-A10B boot — and fill in ANALYSE_389's admittedly empty hit-rate row
   while doing it. The whole of §7 rests on parameters, and one boot replaces
   them with facts.
3. **Only then decide on the CPU lane.** Its post-A′ margin is 0–28% depending
   on true DRAM bandwidth, it costs an L, it needs the GGUF-MoE offload
   exclusion lifted first, it gives up the capturable-graph offload path, and
   it needs a determinism gate. If the re-measurement lands at the top of the
   DRAM band and the user's operating point is bs≤4 single-stream, it becomes
   defensible; at the bottom of the band it is worth nothing.
4. **If it is ever built, adopt `_R4`/`-rtr` with it** (§3.3) — row-interleaved
   layouts are the correct storage format for a CPU-computed tier, and ik's
   MIT `iqk_mul_mat` kernels are the reference implementation.

---

## 8. Not adoptable / no delta — with the hard reason

| Item | Reason |
|---|---|
| `_R4`/`_R8`/`_R16` repacking, `-rtr` | Targets CPU SIMD register width. No GPU analogue; ik's own docs warn against it in hybrid setups. Flips only if §7's CPU lane is built. |
| Trellis `IQ*_KT` | Loses to `IQ4_KS` on quality *and* CUDA speed for Qwen3.6-27B in ik's own July-2026 data (#2213). Adopting it would be adopting a measured regression. |
| CPU flash attention, AVX2/AVX-512/NEON fused ops | Wrong device. |
| NUMA mirroring | One NUMA node on this box. |
| `-fmoe` fused MoE | We already fuse gate+up in both the tiled and vector MoE kernels. |
| `-mla 1/2/3` | We absorb at load time already; MLA is upstream in mainline; our DSV4 attention is a different and more elaborate design. |
| FlashMLA CUDA | We ship a `flashmla` backend plus nine other MLA-family backends. |
| `Q8_KV`, `-ctk/-ctv` quant defaults | Our `--kv-cache-dtype` axis. This is the axis that generated 84 of #372's 121 false type "gaps". |
| MMQ tile kernels for IQ types | We measured dequant+cuBLAS beating MMQ by up to 13x above M=8 on our hardware. Not an omission. |
| `MXFP4` | Upstream in mainline as well; we have NVFP4/INT8/FP8 lanes. No user demand recorded. |
| ik's server/API surface (`/models`, `/responses`, samplers, function calling) | sglang's OpenAI-compatible surface is ahead. |
| `-ser` | Not "not adoptable" — cheap and buildable, but lossy, so it sits behind every byte-identical win per the quality-last rule. |

---

## 9. What would change these verdicts

* **An ik-format GGUF actually landing on disk.** `IQ4_KS` moves from "on
  demand" to "do it", and its block layout becomes readable off a real file
  rather than a description (the #372 rule, unchanged).
* **A measured DRAM read bandwidth at the top of the 32–45 GB/s band**, plus a
  confirmed bs≤4 single-stream operating point. That is the only combination in
  which the CPU lane's post-A′ margin justifies an L.
* **A dual-socket or AMX successor box.** Both of the CPU lane's structural
  weaknesses here — one memory controller, crossover at B≈5 — are properties of
  this CPU, not of the technique. On such a box the verdict flips, and ik's
  NUMA-mirroring work becomes relevant at the same moment.
* **PCIe getting *worse* relative to DRAM** (e.g. more ranks per link, or a
  card moving to the x4 slot). Every step in that direction widens the CPU
  lane's margin; the current x4 slot is already the single largest contributor
  to Path A's 145 ms.
* **ik publishing a CUDA-side result we cannot match.** Nothing in the current
  repo qualifies; their GPU backend exists to serve hybrid inference, not to
  win at it.

---

## 10. Sources

Fetched 2026-08-01 unless noted.

* `github.com/ikawrakow/ik_llama.cpp` — README, `LICENSE`, `docs/parameters.md`,
  `ggml/src/ggml-cuda/` file listing, `ggml/src/ggml-cuda/iqk_mmvq.cu`,
  commit list for `main`
* ik discussions: **#2213** (trellis vs `IQ4_KS`, Qwen3.6-27B, CUDA, 30 Jul
  2026), **#8** (IQ2_K…IQ6_K announcement, AVX2/NEON speedups), **#201** (NUMA),
  the discussions index (expert streaming from Optane; NUMA mirroring;
  MiniMax-M3 MSA; GLM-DSA indexer)
* ik issue **#1699** (ik slower than mainline for Qwen3 MoE `IQ4_XS` CPU-MoE)
* `ikawrakow-ik_llama-cpp.mintlify.app` — quickstart, quantization overview,
  hybrid CPU/GPU inference, GPU offloading
* `deepwiki.com/ikawrakow/ik_llama.cpp/1.1-key-features-and-performance-improvements`
* `github.com/ggml-org/llama.cpp` PRs **#12801** (MLA merged), **#24162**
  (DeepSeek V4)
* This tree: `python/sglang/srt/layers/quantization/gguf.py`,
  `sgl-kernel/csrc/quantization/gguf/{gguf_kernel.cu,mmvq.cuh,mmq.cuh,moe.cuh}`,
  `python/sglang/srt/layers/moe/{expert_offload.py,expert_stats.py}`,
  `python/sglang/srt/models/{deepseek_v2.py,deepseek_v4.py}`,
  `python/sglang/srt/models/deepseek_common/attention_forward_methods/`,
  `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`,
  `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`,
  `docs/dev/ANALYSE_334_club3090_coverage.md` (#372 appendix),
  `docs/dev/ANALYSE_389_nvme_expert_tier.md`
* Host facts read this session: `/proc/cpuinfo`, `lscpu`, `numactl -H`,
  `/proc/meminfo`, `nvidia-smi --query-gpu=pcie.link.*`

---

## 11. §7.8 item 1 as built (#394) — the policy is live, the number is not yet

Recommendation 1 above asked for link-proportional cold-expert sharding. The
apportionment itself landed earlier as an **inert** build: `HostShardRatio`,
`plan_proportional_shares` (largest remainder, whole experts), and
`partition_cold_experts` were written and tested, but nothing ever handed them
a ratio. Two things were missing, and both are now built.

### 11.1 The weights are MEASURED, and the label is load-bearing

The inert build derived weights from NVML's *max PCIe link generation ×
width* — a nameplate. §7.2's numbers are not: 6.4 / 13 / 13 GB/s came off a
timed pinned transfer. The chain is now explicit and ordered by provenance, in
the `planner.cost_model.Provenance` vocabulary the rest of the fork prices
with:

| Rank | Source | Provenance | What it is |
|---|---|---|---|
| 1 | `SGLANG_MOE_HOST_SHARD_RATIO` | MEASURED | the vector an operator typed, i.e. the measurement they took |
| 2 | `card-probe-h2d` | MEASURED | `rigmon/card_probe.py`'s 64 MiB pinned H2D, best-of wall clock, per card by UUID |
| 3 | `nvml-pcie` | ESTIMATE | width × generation. A formula over a measured width, not a transfer anybody timed |
| 4 | `equal` | ABSENT | the shape a **refusal** takes |

`SGLANG_MOE_HOST_SHARD_MIN_PROVENANCE` sets the floor (`measured` or
`estimate`); `absent` is not selectable in either setting, so a split is never
weighted by a number nobody has. An absent link produces an equal ratio, and an
equal ratio produces **no `ColdShardContext` at all** — "nothing is known" and
"#394 is not in this build" are deliberately the same code path.

The card probe is read through `load_card_probe()`, which never triggers a
measurement, and memoized once per process: 40+ MoE layers must not each parse
the same JSON, and a weight load must never turn into a multi-second GPU probe
as a side effect. Reusing that artifact rather than timing a second H2D is the
point — two opinions about the same link measured by two kernels are
indistinguishable after the fact.

Measured against nameplate on this rig: 6.4 : 13 : 13 normalizes to
0.198 / 0.401 / 0.401, the nameplate x4 : x8 : x8 to 0.200 / 0.400 / 0.400. The
difference is ~1 % — which is the honest finding, and the reason the nameplate
stays admissible as a labelled ESTIMATE rather than being rejected.

### 11.2 The rank → card vector, published without a collective (#407 cut 2)

`_gguf_cold_shard_context` could not supply `card_uuids` because a rank knows
only its own card: the launcher narrows `CUDA_VISIBLE_DEVICES` to one GPU per
scheduler process. The obvious fix — `all_gather` the UUID — is refused here.
That call would sit **inside the weight-load loop**, which is the
rank-local-before-group hazard: a rank that reaches the load path on a
different schedule hangs the group with no diagnosis.

`python/sglang/srt/registry/rank_cards.py` inverts it. The datum is not a
runtime measurement at all — it is a launch decision the parent already made
when it computed `gpu_id_for_rank` for every scheduler it spawned. So the
launcher enumerates that same formula, resolves each CUDA ordinal through the
#331 IdentityMap, and writes the **UUID** vector into the environment before
the spawn loop. Workers read it. No peer is contacted.

Properties that are tested rather than assumed:

* **UUIDs, not ordinals.** Each child's CUDA view is narrowed, so an ordinal
  means something different in every process; a UUID does not (#392, #397).
* **All-or-nothing.** One unresolvable ordinal publishes nothing. A partial
  vector would be completed by the consumer with the index it already has,
  which is exactly the substitution being removed.
* **Length-checked by the consumer.** A vector whose length does not match the
  asking group describes a *different* group (a MoE-TP subgroup, a pipeline
  stage) and is refused, not truncated.
* **The CUDA context is not paid silently.** Building the CUDA side of the
  identity map costs a few hundred MiB on every visible card, in the process
  about to spawn workers onto them. It is used only when already paid — with
  `--rank-gpu-id` (whose validation resolves the cards anyway), when a context
  exists for other reasons, or on explicit `SGLANG_RANK_CARD_PROBE_CUDA=1`.
  Otherwise the vector is ABSENT with that reason attached.
* **Multi-node is refused by name.** This launcher sees only its own cards; a
  per-node prefix is not a world-length vector. #394 is single-node by
  construction.
* **A hand-set vector is never overwritten.** Same contract as
  `SGLANG_MOE_HOST_SHARD_RATIO`.

### 11.3 What did not move

Residency is decided before the ratio is consulted, so `resident_ids`,
`resident_count` and `buffer_slots` — the three numbers every VRAM figure and
every #400 ledger entry derives from — are identical with and without a ratio.
Only HOST-side ownership of the cold pool moves. The `StreamingStagingLedger`
already separates `pinned_bytes` from `delegated_bytes`, so the changed shard
sizes are accounted without a new post. The capturable decode path is
untouched: this is a placement change, not a compute-lane change.

### 11.4 The instrument names its own arm (#390)

An A/B whose two dumps cannot be told apart a week later is not a measurement.
Every staged layer now publishes a `host_shard` row — policy
(`equal` / `link-proportional`), the ratio string including its provenance,
cold-pool size, owned and delegated counts, and the realized `owned_share` so
whole-expert rounding is visible rather than assumed away. It rides into the
#390 JSON dump per layer and is lifted to `totals`, where disagreement between
layers surfaces as `mixed` rather than being averaged away.

The baseline arm emits a row too, saying `policy=equal`. A missing row and a
baseline row are different findings and must not look alike.

**Blind spot, unchanged and inherited:** captured decode under
`SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1` takes the host-sync-free path and is not
counted. The A/B is therefore run on the offload's default eager path, which is
also where the merged baseline (146.3 ms/token) was taken.

### 11.5 The card window (2026-08-02): #394 does not run, and the model was optimistic

Two findings, both from the merged V4-Flash recipe (UD-IQ3_XXS, TP=3 uneven,
fraction 0.485/0.42/0.42, eager, bs=1, chunk 512).

**Finding 1 — the proportional arm loads and then refuses to serve.** With the
rank->card vector published and the ratio resolved (0.400/0.200/0.400), all 43
MoE layers staged correctly: rank 0 owned 40.7 % of its cold pool, rank 1 (the
x4 card) 19.5 %, rank 2 39.0 %. The server reached `Application startup
complete` and died on the first forward, on all three ranks, in #394's own
precondition guard:

> experts [80, 83, 94] were delegated to a peer rank's host tier by the #394
> link-proportional cold shard, but this rank's router asked for them.

**This is not a wiring bug; it is the design premise.** `partition_cold_experts`
splits *this rank's own* cold experts and drops the ones apportioned to peers.
Under the #82 GGUF expert-dim shard the ranks hold **disjoint** expert ranges,
so no peer holds rank 0's expert 80 — a "delegated" expert is not relocated, it
is *absent*, and the first token routed to it has nowhere to go. Delegation is
sound only if a delegated expert stays reachable, which requires one of:

* the pinned host pools live in **shared** memory, so any rank can DMA an
  expert out of a peer's pool (all three pools are host DRAM already, so this
  is a mapping question, not a new tier); or
* the experts are **replicated** across ranks plus an EP-style dispatch that
  sends the token to whichever rank owns the copy.

`_gguf_expert_shard` — the eligibility test the wiring uses — is TRUE exactly
when experts are sharded *disjointly*, i.e. exactly the case where delegation
is unsound. **The eligibility test is inverted with respect to the precondition
it claims to enforce.** The guard the inert build shipped did its job: it turned
a silent wrong-output into a named refusal at the first forward.

**Finding 2 — the recoverable gain is 1.36x, not 1.69x/1.77x.** §7.3 derived
Path A from an equal 1/3 byte split per rank. The equal-shard arm's *measured*
split is not equal, because uneven TP already hands the x4 rank fewer experts:

| rank | H2D moved | share | measured link | transfer time |
|---|---|---|---|---|
| tp0 (5090, x8) | 258.0 GiB | 40.0 % | 14.42 GB/s | 19.2 s |
| tp1 (3080, **x4**) | 165.2 GiB | 25.6 % | 6.45 GB/s | **27.5 s** |
| tp2 (3080, x8) | 222.4 GiB | 34.4 % | 13.41 GB/s | 17.8 s |

The x4 rank is still the clock, at **1.28x the mean** — the effect is real and
now measured rather than modelled. But perfect proportional placement moves
27.5 s to 20.2 s, i.e. **1.36x on the transfer term**, and the transfer term is
only part of a 135 ms/token decode. The 1.77x figure (and §7.3's 1.69x) assumed
a 33/33/33 split that this configuration never had.

**Consequence for §7.8.** Recommendation 1 keeps its direction and loses most of
its size. It is now: an **M-to-L** item (a shared-memory host tier or an EP
dispatch, not "placement policy only"), for a **1.36x** ceiling on the transfer
term rather than 1.69x end-to-end. That materially changes its position against
recommendation 3 (the CPU lane), whose own margin §7.4 put at 0-28 % *over* a
working A'. With A' unbuilt and smaller than advertised, the two should be
re-ranked together rather than sequentially.

**Baseline, for the record.** Equal shards, same recipe: **135.1 / 140.1 / 135.4
ms/token** across three passes over two boots (mean 136.9). The A-vs-A floor on
this box today is **~5 % CV within a pass and 3.6 % between passes** — wider
than the 2.55 % the merged baseline was judged on, so nothing below ~4 % would
have been callable from this window even if the proportional arm had run.

| what | value |
|---|---|
| hit rate, activation grain | 0.820 aggregate (0.772 / 0.843 / 0.841) |
| hit rate, unique grain | 0.622 aggregate |
| vs WASTE's 0.14 | **5.9x** |
| decode, equal shards | 136.9 ms/token mean, floor ~5 % CV |
| decode, proportional shards | **not measurable — arm does not serve** |
