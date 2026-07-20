# htsglang — Features and Bugfixes over Upstream sglang

`htsglang` is a fork of [sglang](https://github.com/sgl-project/sglang) built for
**heterogeneous single-node rigs with mismatched GPUs** (the reference machine:
1× RTX 5090 32 GB + 2× RTX 3080 20 GB, no NVLink, NCCL over PCIe host-staging).
The core idea: split **tensor-parallel weights, KV-cache, and MoE experts
*proportionally* across differently-sized/differently-fast cards** instead of the
uniform, equal-shard split that upstream assumes.

This edition is **reorganized by who benefits**. Every capability falls into one of two
buckets:

- **Category A — advantages specific to heterogeneous systems.** The value only
  materializes on mismatched hardware; on matched cards these features degenerate to the
  standard even baseline (no benefit, but no penalty — see
  [*How much worse on homogeneous hardware?*](#how-much-worse-on-homogeneous-hardware)).
- **Category B — advantages for both heterogeneous and homogeneous systems.**
  Hardware-agnostic wins; a matched server gets these fully.

The detailed inventory below is grouped into **Part A** and **Part B** accordingly. The
numbered sections keep **stable numbers** (§1…§11) that are referenced elsewhere
(notably by TOPOLOGIES.md and by cross-references inside this file), so the numbers are
*identifiers, not reading order* — grouping by beneficiary means Part A holds §1, §9,
§10, §11 and Part B holds §2–§8.

See also: [TOPOLOGIES.md](TOPOLOGIES.md) — for each major capability, the canonical
multi-GPU constellation where it is the decisive advantage, with VRAM/host-RAM partition
diagrams and the stock-sglang contrast.

This document tracks work on **two events** — when a feature *lands* (the numbered
sections) and when one is *accepted/planned* (the **Roadmap** section at the end) — so the
"planned vs shipped" line is always explicit.

## How to read this document

### About the impact figures

Impact numbers here are deliberately **rough** (rounded to ≈5-10% or to x-factor
granularity), measured on the heterogeneous reference rig above. They are directional,
not benchmarks: interconnect on this rig is PCIe (no NVLink), which penalizes any
cross-GPU collective, so cost/benefit will differ on better-connected hardware.
Correctness/enablement fixes carry **no throughput claim** — they make something work
or make it deterministic; they are labelled as such rather than given a fabricated
percentage. Where a figure is an **estimate** or a **capability milestone** (something
*boots and stays coherent*) rather than a measured throughput result, it is labelled as
such and not dressed up as a performance claim.

**Direction tags.** Every ratio/percentage below is tagged right at the number so the sign's
meaning is unambiguous at a glance: **[better]** (a win), **[worse]** (a real cost/regression),
**[neutral]** (break-even / roughly unchanged). A single feature can carry both — e.g. more
context (`[better]`) at a small decode cost (`[worse]`).

### On "bugfixes" and robustness

A **bugfix over upstream** here means a defect present in vanilla sglang (or in a
kernel/library it uses) that this fork fixes — it would bite an upstream user too. Those are
listed on their own merit.

This doc does **not** enumerate the fork's own development bugs — a feature that was wrong in
an early revision and then corrected is just the feature working; the fix is not a selling
point. Robustness is only called out where the problem it solved was **not obvious on first
(or second) glance** — a subtle failure mode a careful reader wouldn't expect the feature to
have already handled. Those non-obvious robustness properties are noted at the relevant
feature; routine "we implemented it correctly" fixes are not.

### Status convention

A shipped, integrated + validated feature carries **no marker** (that is the default for the
numbered sections). Anything else is flagged inline: **[in progress]**, **[untested]**
(code complete, not yet validated), **[guarded]** (implemented but gated off — see that section),
**[planned]** (roadmap, not built).

## The two categories: who benefits

### Category A — advantages specific to heterogeneous systems

These are the fork's unique differentiator. Their value comes from **asymmetry** — a fast
card next to slow cards, or more ranks than physical GPUs — so it only exists on mismatched
hardware. On matched cards they collapse to the standard even split (details in the next
section).

- **Uneven Tensor Parallelism + Decode Context Parallelism** (`--rank-tp-ratio`,
  proportional weight shards; uneven-DCP proportional KV token-split; `auto` VRAM-optimal
  splitting) — §1 (Part A).
- **Multi-rank co-location / TP > physical card count** (`--rank-gpu-id` with duplicates) —
  §9 (Part A).
- **Weightless-KV fast lane (Variant C)** — the fast card holds the full model, the slow
  cards become pure KV workers; plus the tiered-KV fabric extension toward long context
  **[in progress]** — §10 (Part A).
- **Cross-vendor host-staging collectives (HTCCL)** — mixing NVIDIA + AMD in one TP group
  **[planned]** — §11 (Part A).
- **MoE expert-offload combined with uneven-TP + uneven-DCP** — the 122B-A10B-on-3-mixed-cards
  end-state. The *combination* is heterogeneous-specific; the base offload capability itself is
  Category B. Detailed in §6 (Part B), cross-referenced here.

### Category B — advantages for both heterogeneous and homogeneous systems

Hardware-agnostic wins. A matched, homogeneous server gets every one of these in full; the
heterogeneous rig gets them too, but they are not the *reason* to run mismatched cards.

- **GGUF new-arch adapters + K-quant kernel perf overhaul** — on top of upstream sglang's GGUF
  loader + MMVQ/MMQ K-quant kernels: bespoke hybrid-GDN adapters, uneven-TP GGUF sharding, the MMQ
  token-cutoff perf overhaul, Q8 lm_head (TP=2 beats llama.cpp on decode). The GGUF loader and the
  base kernels are **upstream**; the adapters, uneven-TP sharding + tuning are the fork's — §4
  (Part B).
- **MoE expert-offload** — run models larger than VRAM on any box; CUDA-graph compatible,
  hot-set residency, quality-validated — §6 (Part B).
- **HiCache under uneven-TP/DCP** — *extends* upstream sglang's hierarchical KV caching (GPU →
  host → disk, graph-safe) to the fork's non-uniform layouts + hybrid-Mamba (GDN) state tiering.
  The base HiCache is **upstream**; the fork's slice is the uneven-layout safety + concurrency
  hardening — §8 (Part B).
- **Speculative decoding under uneven-DCP + mixed-GPU reproducibility** — *extends* upstream
  spec-decode (MTP/NEXTN, EAGLE, EAGLE3, **and** the adaptive draft-length controller — all
  upstream) with the fork's rank-0-broadcast reproducibility across mismatched GPUs, uneven-DCP
  correctness, and EAGLE3-for-Gemma-4. The spec engines and the adaptive controller themselves are
  **upstream** — §3 (Part B).
- **Prefill/Decode disaggregation, single-node hetero** — *extends* upstream sglang's
  PD-disaggregation (the mooncake transfer stack) into a single-node solo-prefill-on-the-fast-card +
  uneven-TP/DCP decode layout. The PD framework is **upstream**; the fork adds the single-node
  hetero role split + GDN-state handoff — §2 (Part B).
- **TP > num_kv_heads** (replicated + token-sharded KV) — §1 / §9.
- **Quantization correctness** (AWQ/Marlin g32 MoE fix, GPTQ-MoE TP>1 upstream bugfix,
  compressed-tensors, AutoRound-int4) — §5 (Part B).
- **Spec-decode reproducibility across mismatched GPUs, CUDA-graph / uneven-DCP correctness fixes,
  and Mamba/GDN self-determinism** — three *distinct* properties, kept separate (not lumped under one
  loose "determinism") — §8 (Part B).
- **Vision / multimodal GGUF and broad model support** — §7 (Part B).
- **Long-context KV-spill** **[in progress]** — §10 (Part A, but the underlying spill
  mechanism is hardware-agnostic).

## How much worse on homogeneous hardware?

A natural question is "how much worse do these features run on homogeneous instead of
heterogeneous hardware?" The naive reading is misleading, so this section answers it
head-on.

**1. The Category-A features do not run *worse* on homogeneous hardware — they
*degenerate* to the standard even baseline.** On matched cards there is simply no asymmetry
to exploit, so the special path collapses to even-TP / even-DCP with no benefit **and no
penalty**:

- Proportional split **==** even split when the cards are equal. `--rank-tp-ratio auto`
  returns a uniform vector on identical GPUs (the equal case is handled as plain even TP;
  an early bug where `auto` produced a uniform *list* instead of collapsing to the scalar
  even-split was fixed — confirming this degeneration is the intended, exercised path).
- The weightless-KV lane's "fast card holds the weights, slow cards hold only KV" design is
  **pointless when all cards are equal** — there is no fast/slow split to exploit, so you
  would just run ordinary even-TP.
- Multi-rank co-location only matters when **TP exceeds the physical GPU count**; with
  enough matched cards you place one rank per card and co-location adds nothing.

Where forcing a Category-A feature onto matched cards would add *pure overhead* rather than
nothing, that is stated plainly: uneven-DCP adds extra per-step cross-GPU collectives
(measured **[worse] ≈-10-25% decode** over PCIe). Those collectives buy extra context only
when capacity is *asymmetric*; on matched cards there is no extra context to win, so you
simply do not enable it — it degenerates to even-DCP / no-DCP. The honest answer is
therefore: **no benefit, no penalty — you run the standard even path.**

**2. The real "how much worse" comparison is running heterogeneous hardware *without* the
fork** (or on a stock engine). There the mismatched capacity is wasted, you are forced to
the smallest-common-denominator card, or you must buy a matched set (capital cost).
Measured on the 5090 + 2×3080 rig:

- **Uneven-TP (§1).** Stock even-TP caps every rank to the smallest card's shard, wasting
  the 5090's extra ~12 GB; and for Qwen3.6-27B, TP=3 is *structurally impossible* on stock
  (24 query / 4 KV heads not divisible by 3). With the fork, `--rank-tp-ratio 2,1,1` runs
  27B at TP=3, **262k context** (capability milestone — boots + coherent, not a throughput
  claim), decode **68/97 tok/s** (≈93% of a TP=4-MPS layout). `auto` yields
  **[better] ≈+2.5-3× KV context** vs a naive equal split.
- **122B-A10B GPTQ-Int4 end-state (§6 combined with §1).** ~61 GB of experts: does not fit
  any single card, and even-TP=3 is impossible. Stock can at best run it offloaded on the
  single 5090 at **4.83 tok/s** (mechanism proof, poor hit-rate — nearly everything spills
  to one card). With the fork (per-expert offload × uneven-TP × uneven-DCP across all three
  cards): **6.97 tok/s eager → 16.34 tok/s** with CUDA-graph + hot-set (**[better] +134%**).
  So the mismatched rig goes from "4.83 tok/s on one card, or cannot run at all" to
  16.34 tok/s. The alternative without the fork is to buy a matched multi-GPU set.
- **Weightless-KV lane (§10).** On the same rig the slow cards' layer-weight VRAM drops to
  **≈0** (~14 GB freed per worker), context lifts **≈4×** on the 27B hybrid, and CUDA-graph
  decode over 512 tokens is **[better] +385%** vs the eager lane. Without the fork every
  rank must hold the full weights, so the slow cards spend VRAM on weights instead of KV.

All of the above are measured on the heterogeneous rig; the 262k figure is a **capability
milestone** (boots + coherent), not a throughput claim — decode over PCIe stays
bandwidth/latency-bound (the PCIe wall), which is exactly why the raw tok/s numbers are
modest even where the capability is decisive.

**3. Flip side, stated fairly: a homogeneous user still gets *all* of Category B in full.**
GGUF + tuned K-quant kernels (TP=2 beats llama.cpp), MoE offload to run models larger than
VRAM on any box, HiCache, speculative decoding, PD-disaggregation, and the CUDA-graph
correctness / reproducibility fixes all apply unchanged on matched cards. The fork is
genuinely valuable on homogeneous hardware too — just not for its unique heterogeneous
reason. The honest summary: **don't oversell A on matched cards, don't undersell B.**

## Performance-per-watt (theoretical, ~constant board power)

### Why throughput gains ≈ perf/Watt gains

A GPU under active inference draws **~constant board power**, close to its power envelope —
it does *not* draw meaningfully fewer watts when the software computes "less efficiently".
The direct consequence: a throughput gain of X% at ~constant power ≈ a **performance-per-watt
gain of ~X%**. Adaptive MTP is the clean example — it costs no extra electricity and lifts
throughput directly, so tokens/Watt rises deterministically. This is not an open measurement
dispute; it is a first-order derivation.

**Label it honestly.** These are **theoretical / first-order estimates** that assume
~constant board power. Real per-kernel power varies — memory-bound kernels draw less than the
envelope — so the actual figure differs from the nominal throughput ratio. The **measured
Dashboard energy is the ground truth**; the numbers below are an analytical derivation, not a
wattmeter reading. They are stated to keep the same measured-only discipline the Dashboard
enforces: a derived quantity, clearly flagged as derived.

### Applied to the throughput features

Per feature, the theoretical perf/Watt gain ≈ the already-**measured** throughput gain (all at
~constant board power; each baseline named precisely):

| Feature | Measured throughput gain | Theoretical perf/Watt (prefill / decode) |
|---|---|---|
| Adaptive MTP / spec-decode (EAGLE/EAGLE3/MTP) — §3 | **net** decode gain (acceptance-adjusted): EAGLE3 ≈+15-45% (code/JSON), high-accept k=4 ≈+8% / k=5 ≈+16% vs k=3 | decode ≈ **net** gain (never above it — see split 2); workload-conditional |
| GGUF K-quant kernel — batched MMVQ, Q8 lm_head — §4 | prefill ≈5-8× vs the always-MMQ path; TP=2 beats llama.cpp on decode | prefill ≈ +5-8× (same power, faster kernels); decode: up vs llama.cpp, ~neutral vs FP8 (bandwidth-bound) |
| CUDA-graph decode #133 (weightless lane) — §10 | **≈+385% decode** vs the **eager-weightless** baseline | decode ≈ +385% — the eager path wasted power on the per-collective gloo guard-handshake / launch overhead; the graph does the same work with less waste |
| MoE-offload Graph+Hotset — §6 | **≈+134% decode** vs the **eager-offload** baseline (TP=3, fraction=0.25) | decode ≈ +134% **within the offload path** — see split 1; this is *not* an absolute efficiency gain vs a fits-in-VRAM run |
| Solo-prefill PD-disagg — §2 | ≈2-5× TTFT (prefill), decode ~unchanged | prefill ≈ +2-5× (prefill runs alone on the fast card, zero cross-GPU traffic); decode ~neutral |
| `--rank-kv-ratio capacity` — §1 | break-even decode (≈±1%) at +25% context | decode ≈ neutral (a capacity win, not a perf/Watt win) |

### Hard honesty splits (so this is not an overclaim)

1. **This holds for *throughput* features at ~constant power.** **Capacity / enabling** features —
   expert offload to run a model that does not fit VRAM (§6), host-tier KV-spill for long context
   (§10, #134) — **add streaming overhead**, so their tokens/Watt can be **lower** than a
   hypothetical fits-in-VRAM run. Their value is **"it runs at all"**, not "more tps/Watt". The
   **+134% Graph+Hotset** figure is a perf gain **within the offload path, relative to the
   eager-offload baseline** — not an absolute efficiency gain, and it must not be read as one.
2. **Spec-decode perf/Watt is capped at the *net* throughput gain** (which already charges for the
   draft's discarded work). At low acceptance the extra draft compute can even **lower** tps/Watt —
   which is precisely why **adaptive MTP** is the good example: it adapts k to keep the net positive.
   Never claim a spec-decode perf/Watt gain above the measured net decode gain.
3. **All of the above is theoretical**, assumes ~constant board power, and is an analytical
   derivation. The **Dashboard's measured energy is the ground truth**; per-kernel power varies
   (memory-bound kernels draw below the envelope), so treat these as first-order estimates only.

---

# Part A — Advantages specific to heterogeneous systems

These sections deliver value only on mismatched hardware; on matched cards they degenerate
to the even baseline (see above).

## 1. Uneven Tensor Parallelism & Decode Context Parallelism (DCP)

**Benefits:** heterogeneous systems (the foundational differentiator). Degenerates to even-TP /
even-DCP on matched cards. The **TP > num_kv_heads** sub-item is hardware-agnostic (Category B).

The foundational feature set: let each TP rank own a share of the model proportional to
its GPU, instead of an equal split.

- **`--rank-tp-ratio`** — proportional TP weight shards across unequal GPUs.
  Attention / GDN are split in whole KV-head units per rank.
  *Impact: enables TP across mismatched GPUs that upstream cannot do at all (the big
  card would otherwise be capped to the small card's shard). No throughput claim —
  it's the enablement primitive everything else builds on.*
- **`--rank-tp-ratio auto`** — VRAM-optimal automatic split: fills every card,
  maximizing total usable context. Derives weights from NVML budgets (gcd-reduced),
  sets DCP = TP automatically. On identical GPUs it collapses to the uniform even split.
  *Impact: **[better] ≈+2.5-3x KV-cache context** on the reference rig vs a naive equal split (the
  equal split is bounded by the smallest card).*
- **`--rank-tp-ratio auto-performance`** — measurement-based split that optimizes
  for throughput rather than maximum context.
  *Impact: trades some context for decode throughput; workload-dependent.*
- **`--rank-mlp-ratio` / `--rank-moe-ratio` / `--rank-vocab-ratio`** — per-family
  shard vectors for MLP, MoE expert-intermediate, and vocabulary, so each dimension can
  be split independently of the attention split.
  *Impact: correctness/flexibility feature — lets indivisible dimensions land cleanly
  per rank. No throughput claim.*
- **Named-family shard plans + maximin solver** — MLP and MoE units are jointly
  distributed across ranks by a solver rather than ad-hoc rounding.
  *Impact: correctness — guarantees valid, balanced partitions for every sharded
  dimension.*
- **Uneven-DCP token sharding** — KV-cache follows the **token axis** instead of the
  KV-head axis: proportional KV-token split with a weighted owner rule + LSE merge.
  Removes the `sum(ratios) ∈ {2,4}` KV-head constraint entirely.
  *Impact: **[better] ≈+60-80% context** over the first (head-axis) DCP version by decoupling KV
  capacity from the weight split; **[worse] ≈-10-25% decode throughput** vs no-DCP baseline, the
  honest cost of the extra per-step collectives over PCIe. (On matched cards this cost is pure
  overhead with no context to win — so you run even-DCP / no-DCP instead.)*
- **`--rank-kv-ratio {coupled|capacity|auto|LIST}`** — KV-token ownership **decoupled**
  from the weight split; `capacity` mode installs the measured optimal vector in a single
  boot (no iterate-and-reboot).
  *Impact: **[better] ≈+25% context** on 27B-FP8 at **[neutral] break-even decode** (≈±1%).*
- **TP > num_kv_heads (replicated KV)** — KV heads are replicated + token-sharded +
  LSE-merged so TP can exceed the KV-head count. Coherent across FP8 / GGUF / AWQ, including
  GQA re-grouping for small head counts (down to gqa=1 ranks). **This sub-item is
  hardware-agnostic (Category B)** — it unlocks TP degrees that were structurally impossible
  on any hardware.
  *Impact: enables TP degrees that were structurally impossible before (applies under uneven-DCP
  too). No throughput claim.*
  **The price is small:** because TP exceeds `num_kv_heads`, the few KV heads must be **replicated**
  across the ranks that share a KV-head group instead of sharded. But under GQA there are only a
  handful of KV heads to begin with (the query heads, which dominate, are still sharded normally),
  and under uneven-DCP the KV *tokens* are still split across ranks — so what's duplicated is just a
  small KV slice, not the bulk of the cache. In practice this is a **[worse] minor, roughly single-digit-%
  KV overhead** (a real but small cost), easily paid for by the extra context / higher TP degree it
  unlocks — not a heavy tax.
- **Rank-uniform collective guards** — guards in front of `all_gather` that require
  rank-uniform shapes, converting a would-be NCCL hang into a clean, early error.
  *Impact: robustness — turns a silent distributed hang into a fail-fast error.*
- **Auto-sizing Mamba/GDN state pool, rank-aware GDN shapes, per-rank head counts
  (flashinfer + triton)** — including per-rank workspace sizing.
  *Impact: correctness — hybrid (GDN) models run correctly under uneven TP. No throughput
  claim.*
- **`max_total_num_tokens` caps** — physically-reachable ceilings for GDN hybrids and
  SWA hybrids, computed deterministically so the pool never overshoots into OOM.
  *Impact: correctness/robustness — prevents deterministic-looking OOM from over-sized
  pools.*
- **CPU unit tests** — coverage for family plans, the solver, calibration, and
  partition invariants (fast, no GPU needed).

## 9. Multi-rank-per-GPU & Tooling

**Benefits:** heterogeneous / limited-hardware systems (co-location matters only when TP exceeds
the physical GPU count). The replicated-KV correctness it validates is itself Category B.

- **`--rank-gpu-id` (with duplicates) + NCCL auto-config + physical-impossibility check** —
  explicit rank→physical-GPU placement, NCCL auto-configuration when co-location is detected, and a
  fail-fast check that `(ranks on a GPU) × per-rank-MiB ≤ NVML total`.
  *Impact: enablement — multiple ranks per physical GPU (e.g. TP=4 on 3 cards), with early, clear
  errors for impossible mappings.*
- **Replicated-KV-head correctness for TP > num_kv_heads, proven at TP=5 (#62)** — the replicated-KV
  path (§1: when the tensor-parallel degree exceeds the model's KV-head count, the KV heads are
  replicated across the extra ranks and the KV is token-sharded) is validated to run correctly at
  **TP=5** — a TP degree above both the KV-head count and the physical GPU count. That 5-way
  validation was carried out on only 3 physical GPUs by using **multi-rank co-location** as the test
  vehicle: 5 ranks emulated across 3 cards (2 ranks share the 5090, 1 on each 3080) via NCCL
  multi-rank, with the co-location env auto-set when duplicate rank→GPU mappings are detected. So the
  high-TP KV-head-copy behavior was proven without owning 5 GPUs.
  *Impact: **[better] confirms replicated-KV correctness at a TP degree above the card count** (correctness,
  no throughput claim). Multi-rank co-location — running TP=N with N above the physical GPU count — is
  the enabling method here, and is also usable on its own for finer-grained sharding. Honest caveat:
  **[neutral]/[worse] co-located ranks share one piece of silicon**, so they contend for SMs and add no memory
  bandwidth (they are the same GPU); co-location buys capability/flexibility (fitting or emulating
  larger-TP layouts on limited hardware), not raw throughput. Pairs with the fork's uneven-TP so the
  co-located card can also carry a proportionally larger shard.*
- **Rig dashboard, Docker image, falsifier levers** — operational tooling: a live rig dashboard,
  a reproducible container image, and env-gated falsifier probes used to hunt the determinism bugs
  in §8.

## 10. Serving & Scheduling (weightless-KV lane + fast-lane priority)

**Benefits:** the **weightless-KV lane** is heterogeneous-specific (fast card holds weights, slow
cards hold only KV — pointless when all cards are equal). The **fast-lane priority scheduler** is
hardware-agnostic (Category B) — it improves interactive TTFT under load on any box.

- **Fast-lane priority scheduling (Stage 0, `--enable-fast-lane`)** *(Category B — hardware-agnostic)* —
  an opt-in fast lane built on
  sglang's priority subsystem. Tag a request `lane="fast"` (or high priority) and, under heavy
  concurrent load, it **preempts into the running batch** instead of being head-of-line-blocked
  behind a full heavy batch. A **reserved-heavy-slots floor** plus **heavy-aging** guarantees the
  heavy batch keeps making progress, so there is no starvation of the background load. Opt-in and
  default OFF → the default scheduling path stays byte-identical.
  *Impact: **[better] interactive TTFT under saturated load dramatically better** — first token in
  milliseconds instead of tens of seconds, i.e. near-unloaded responsiveness even while a heavy
  batch runs; **[better] heavy-batch throughput largely retained** under the fast preemption; **[neutral] zero extra
  VRAM**.*
  **Honest scope (don't over-read it):** once admitted, the fast request **decodes at the shared
  batched rate, not at solo speed**. Stage 0 fixes **responsiveness / TTFT** under load — it does
  **not** give sustained **single-stream decode** speed under load. That needs a dedicated compute
  lane, i.e. the Weightless-KV Fast Lane below (stages B1+B2a now landed), not just priority
  preemption.

- **Weightless-KV Fast Lane (Variant C, stages B1 + B2a; + chunked prefill #131)** *(Category A —
  heterogeneous-specific)* — a single-process
  asymmetric-TP mode for heterogeneous GPUs. The **fast card** (here the 5090) holds the **full model as collective-free
  TP=1** and is the sole Q/K/V producer + attention dispatcher; the **slow cards** (the 3080s) become
  **"weightless KV workers"** — they hold **only a DCP token-shard of the attention KV cache** and run
  a stripped attention-only forward, with **no layer weights at all**. This inverts the usual problem:
  instead of the slow cards limiting capacity, they contribute pure KV headroom.
  *Impact: **[better] slow-card layer-weight VRAM drops to ≈0** — a worker materializes a meta-model, only
  the KV pool + attention backend are real (here ≈14 GB freed per worker), so the workers stop being
  the KV bottleneck and **[better] context capacity lifts ≈4×** on the test model (hybrid-GDN 27B). **[better]
  Correctness proven byte-identical** to a full-TP=1 baseline: the prefill/extend path is bit-identical
  (Δ=0), and decode differs only by benign decode-kernel fp-order — smaller than the model's own
  intrinsic decode-vs-extend kernel variance (the single observed trajectory flip lands on a perfect
  50/50 tie); self-deterministic. **[better] Built-in anti-hang guard** (bounded per-step handshake) —
  asymmetric-rank collective divergences fail loud in seconds instead of a silent NCCL hang.*
  **Chunked prefill / long prompts (#131) [landed, unpushed on `feat/weightless-kv-chunked`]:** the lane
  now handles chunked prefill, not just single-shot prefill — lifting the previous short-prompt-only
  limit. Correctness is stated at the lane's true two-class byte-identity bar (the honest framing):
  **head-local paths** (single-shot prefill, empty prefix) are machine-zero vs the TP=1-solo baseline;
  **cross-rank-merge paths** (decode, and now chunked prefill's sharded-prefix read via the LSE merge)
  are decode-class — argmax-identical to solo with divergence only inside the intrinsic fp-reassociation
  band (a 48/48 argmax-match decode trajectory with bounded, non-compounding delta), **not** bit-exact,
  exactly like decode. Rank-uniform collective lock-step verified on hardware (zero head-vs-worker
  mismatches across chunks, no NCCL hang).
  **CUDA-graph decode (#133) [landed, unpushed on `feat/weightless-kv-chunked`]:** the lane now supports
  CUDA-graph decode (prefill still runs eager by design — decode-graph-only scope). The earlier
  suspected root cause (a GGUF lazy-init param) was wrong: the real issue is that the weightless
  **workers** are meta-device models (no layer weights), so decode-graph capture ran the full
  meta-forward on them and crashed on the uninitialized meta embedding. The fix captures each worker's
  stripped KV+attention dispatch as its own decode graph, **symmetrically with the head** (the head
  cannot capture alone — its capture-time DCP collectives require worker co-participation), and
  completes the previously-unwired guard-in-graph path (the rank-uniform anti-hang guard is disabled
  inside the captured region and kept on for eager prefill).
  *Impact: on the reference rig (5090 + 2×3080, TP=3 DCP=3, Qwen3.6-27B Q3_K_M): **[better] ≈+385%
  decode throughput** (roughly 5×, graph 63.5 vs eager 13.1 tok/s) — the eager path paid a per-collective
  gloo guard-handshake that the graph bakes out — sustained 512-token decode with no lock-step hang,
  self-deterministic 5/5. Notably **[better] decode-graph vs eager-weightless is bit-identical** here
  (max |Δ| = 0, argmax-clean): this path has no capture-gated dual-stream, so graph == eager exactly
  (unlike the general graph/marlin floor elsewhere in the fork).*
  **Honest downsides:** **[neutral] prefill still runs eager** by design (decode-graph-only scope);
  **[worse] the fast card now holds the full weights**, so once the workers are freed **it** becomes the
  context limiter for big models (addressed by the load-time MoE offload in §6); **[neutral] TP-only,
  single-node** — PP/DP/EP/spec are rejected by design (hard fail-fast).

- **Tiered-KV fabric toward long context (Variant C next slice, #134 B1/B2) [in progress]** — the next
  step for the weightless lane: spill the DCP KV token-shards to a **host-RAM tier** so total usable
  context exceeds the combined VRAM of the rig (the ~262k target on the 27B hybrid). Same principle as
  the lane above — the slow cards contribute KV headroom — extended down the memory hierarchy.
  *Status: [in progress] — the KV-spill mechanism itself is hardware-agnostic, but the tiered fabric on
  the weightless lane is heterogeneous-specific. Expect the same PCIe wall on the host-tier hop: this is
  a **capacity/capability** extension, not a decode-throughput win. No throughput claim.*

## 11. Cross-vendor host-staging collectives (HTCCL) [planned]

**Benefits:** heterogeneous systems, specifically **cross-vendor** TP groups (NVIDIA + AMD in one
group). Not applicable on a single-vendor homogeneous box.

Host-staged tensor-parallel collectives that bypass NCCL/RCCL, so a single TP group could mix
NVIDIA and AMD GPUs — neither vendor's collective library can form a shared communicator, yet on
P2P-less rigs both already stage reductions through host RAM anyway. HTCCL replicates exactly that
host-RAM data movement vendor-neutrally.

*Status (honest): **[planned]** for this fork.* The NVIDIA-only path is the *identical* code path
(the reference rig has no P2P, so NCCL itself stages over host RAM here), and a prototype of that
host-staging collective reached **NCCL parity** in measurement — decode at parity, 16 MB all-reduce
≈0.8-1.2× NCCL, greedy output bit-identical. The actual **cross-vendor AMD bring-up is pending
hardware and untested**; the `all_gather` / `reduce_scatter` device-group paths and the uneven-DCP
all-to-all still need routing through the vendor-neutral transport. Listed here because cross-vendor
TP is the differentiator it targets — this is not a shipped htsglang capability yet.

---

# Part B — Advantages for both heterogeneous and homogeneous systems

These sections are hardware-agnostic. A matched, homogeneous server gets every one of them in
full; the heterogeneous rig benefits too, but not for a heterogeneity-specific reason.

## 2. Prefill/Decode Disaggregation (Single-Node)

**Benefits:** all hardware. On a heterogeneous rig it lets the fastest card do prefill solo; on a
homogeneous rig it still separates prefill and decode roles to protect TTFT under load.

**Provenance:** PD-disaggregation itself — the role split and the mooncake transfer stack — is
**upstream sglang**. This section is the fork's *extension* of it: solo-prefill on the fastest card,
distributed uneven-TP+DCP decode, the single-node `mooncake_tcp` loopback, and the GDN/hybrid-state
handoff. The transfer code is reused, not re-authored (see the `mooncake_tcp` entry).

Run prefill and decode as separate roles on the same box, so the fastest card does prefill
with zero cross-GPU communication while the slower cards handle distributed decode.

- **Solo-prefill on the fastest card** — prefill runs TP=1 (zero inter-GPU comm), decode
  runs distributed TP=3 uneven+DCP.
  *Impact: **[better] ≈2-5x faster TTFT** across context lengths (largest win in the mid-context range) —
  because prefill now runs alone on the fast card with zero cross-GPU traffic. Decode is a
  separate story: it stays distributed TP=3+DCP (not on the solo card), so it is essentially
  unchanged — a **[worse] negligible ≈-2% at long context** from the decode-side collectives, not a
  regression worth worrying about. Net: a clear, honest win for time-to-first-token at no real
  decode cost.*
- **Token-vector KV re-scatter** — global→owned-compact translation + ordinal filter; one
  rule-agnostic path serving both even and weighted DCP, with no head re-cutting (KV heads are
  replicated).
  *Impact: correctness/architecture — a single code path for the KV handoff instead of
  per-mode special cases.*
- **GDN / hybrid state transfer** — Mamba-state handoff from prefill to decode, validated
  bit-identical.
  *Impact: correctness — hybrid models survive the prefill→decode handoff losslessly. No
  throughput claim.*
- **`local_proxy`** — a thin two-server front (health, `/metrics_summary`, dashboard-ready).
- **`mooncake_tcp` loopback** — reuses the existing transfer stack (no new transfer code);
  prefill and decode ranks co-exist stably on one card via a budget protocol
  (`expandable_segments` + fake-warm).
- **Crash robustness** — hard-kill of the prefill mid-transfer: decode survives, re-registers
  cleanly, tears down to 0 MiB.
  *Impact: robustness — no orphaned VRAM or wedged decode after a prefill crash.*

## 3. Speculative Decoding (MTP / NEXTN / EAGLE / EAGLE3)

**Benefits:** all hardware. The mixed-arch *reproducibility* work is heterogeneity-motivated
correctness, but the speed of speculation itself is a win on any box.

**Provenance:** the spec-decode engines (MTP/NEXTN, EAGLE, EAGLE3) **and** the adaptive
draft-length controller (`--speculative-adaptive`, the EMA/hysteresis `AdaptiveStepSlot`, the
method-agnostic runtime-state/param machinery, and its topk=1 / not-multi-layer-EAGLE /
no-DP-attn-TBO-pdmux constraints) are **upstream sglang** — verified present at the fork's
merge-base and untouched by any fork commit. This section covers the fork's *extensions* on top of
that base: mixed-GPU reproducibility (rank-0 broadcast), uneven-DCP correctness, EAGLE3-for-Gemma-4,
and draft-path uneven-TP. The speed/adaptivity of speculation is upstream's; the fork makes it
*reproducible and correct under mismatched GPUs + uneven-DCP*.

- **MTP/NEXTN under uneven-DCP** — including heterogeneous-GPU reproducibility (verify-sync,
  draft broadcast from rank 0, workspace zeroing).
  *Impact: correctness — the **emitted greedy token sequence** is reproducible on mixed-arch GPUs
  (run-to-run, cold==warm, independent of which card holds which rank). This does **not** mean the
  MTP heads' internal layers are bit-identical across archs — they aren't, and don't need to be:
  spec decode is output-preserving (emitted tokens = the target model's argmax chain, so a
  numerically "noisy" draft only changes how many tokens are accepted per step, not which tokens
  come out), and the accept/argmax decision is taken once on rank 0 and broadcast so ranks can't
  commit different tokens near a tie. See §8 for the three roots. No throughput claim beyond
  enabling spec at all under DCP.*
- **Adaptive draft length (controller is UPSTREAM; the fork makes its picks rank-deterministic)** —
  the adaptive-spec controller is **upstream sglang**, not a fork feature: `--speculative-adaptive`,
  the EMA + hysteresis `AdaptiveStepSlot` that picks k∈{1,2,3} at runtime, the `candidate_steps`
  config, and the **shared, method-agnostic** `AdaptiveController` / runtime-state / param machinery
  that drives **EAGLE, EAGLE3, and NEXTN/MTP** (the `FROZEN_KV_MTP` worker) identically all exist at
  the merge-base and were **not modified by the fork**. The fail-fast constraints — **topk=1 only**
  (chain speculation; tree spec excluded), **not** the multi-layer-EAGLE worker, DP-attention /
  two-batch-overlap / pdmux rejected — are likewise upstream (`adaptive_unsupported_reason`).
  **The fork's contribution here is narrow:** the per-step draft picks are made **rank-deterministic**
  (broadcast from rank 0, §8) so upstream's adaptive spec stays *reproducible* under the fork's
  uneven-DCP + mismatched-GPU layouts. Pre-captured graph states per k are also upstream infra.
  *Impact (upstream's, restated for context): roughly matches the best fixed k on any workload without
  hand-tuning across EAGLE/EAGLE3/MTP alike. The fork adds no throughput here — only reproducibility
  under mismatched GPUs. No fork throughput claim.*
- **Graph-state offload to system RAM (offload mechanism, stages 1+2)** — the core enabler for
  adaptive / high-k speculation. Each draft length k needs its own pre-captured CUDA graph, and each
  such graph pins a block of VRAM. "Stages 1+2" refers to the two implementation stages of the
  *offload mechanism* — not the k values; the k range itself goes **up to 5**. That high end is where
  the feature earns its keep: with k∈{1..5} only the **currently-active** k's graph stays resident on
  the GPU, and the **other up to four k-graphs sit offloaded in system RAM**, streaming back onto the
  card on demand when the runtime switches k. Without this, keeping five k-graphs resident at once
  would multiply the graph VRAM and eat straight into the KV-cache / context budget. Mechanism:
  pause/resume, private MemPools, tagged capture-pools + int-workspaces. Modes: resident / offload /
  offload-scratch.
  *Impact: this is what makes adaptive **and** high-k (up to 5) spec **[better] free in VRAM** — **[better] KV loss:
  zero**, full context preserved at standard reserve even with all five k-rungs available, because
  four of them live in host RAM rather than on the card. Without it, multi-k (let alone k=5) spec
  would either not boot at standard reserve or would have to give back context. Enablement, no
  throughput claim.*
- **High-accept profile [1..5]** — k=4/5 rungs for repetitive workloads; opt-in, boots at
  standard reserve thanks to the offload above.
  *Impact: **[better] ≈+8% (k=4) to ≈+16% (pinned k=5) decode** on repetitive/structured loads vs k=3;
  workload-conditional — neutral-to-negative on diverse prose.*
- **EAGLE3 for Gemma-4** — speculators / vLLM-format support (aux-id off-by-one translation,
  `norm_before_residual` flag, converter).
  *Impact: **[better] ≈+15-45% decode** on code/JSON with an instruction-tuned head vs the non-spec baseline;
  recommendation documented as workload-conditional (weaker on free-form prose).*
- **Draft embed / lm_head sharing, draft-extend replay, ratio-weighted draft vocab** —
  *Impact: the draft path works correctly under uneven TP. No throughput claim.*

## 4. GGUF

**Benefits:** all hardware. The tuned K-quant kernel and perf overhaul help any GGUF model on any
box; the uneven-TP GGUF sharding is the heterogeneous slice within it.

**Provenance:** GGUF loading itself and the base K-quant kernels (`gguf_kernel.cu`, `mmvq.cuh`,
`mmq.cuh`, `layers/quantization/gguf.py`) are **upstream sglang** — present at the merge-base. The
fork's contribution is *on top of* that base: the bespoke new-architecture adapters (Qwen3.5/3.6
hybrid-GDN, Gemma-4 dense), uneven-TP GGUF sharding, and the MMQ-token-cutoff / batched-MMVQ / Q8
lm_head perf overhaul. "GGUF support" as such is not a fork addition; the adapters + uneven-TP +
kernel tuning are.

There are two kinds of GGUF work here: (1) **new architecture support** — bespoke loader adapters
that run families upstream cannot load on its fast path, on the fork's own fast GGUF path rather than
the slow generic transformers GGUF loader; and (2) **architecture-general GGUF improvements**
(uneven-TP sharding, the tuned K-quant kernel, the perf overhaul, the vec alignment below) that
benefit **any** GGUF model. The bespoke adapters (currently Qwen3.5/3.6 and Gemma-4 dense) have been
generalized into a registry + per-family mapping tables (see the loader-generalization entry below),
so a new family is now a small table module rather than a whole adapter. Note that Gemma-4 is also
available via AutoRound-int4 / Marlin (see §5 and §7) — the GGUF path below is a separate, additional
way to run it.

- **Qwen3.5/3.6 hybrid-GDN GGUF** — including NEXTN/MTP head, unsloth UD-quants, mmproj vision.
  *Impact: enables GGUF for a model family/architecture upstream sglang could not load (hybrid-GDN).
  No throughput claim — enablement.*
- **Gemma-4 dense GGUF (S1) [landed, unpushed on `feat/gemma4-gguf`]** — a bespoke GGUF loader adapter
  for the Gemma-4 dense family (the second family beside the Qwen3.5 path), verified on the 31B-it
  Q4_K_M. Runs Gemma-4 GGUF checkpoints on the fork's fast bespoke path, not the slow generic
  transformers GGUF loader. Correctness of the norm handling is proven two independent ways: all norms
  (including q/k-norm) load as **identity** — the unsloth export bakes no `+1` — confirmed both by
  byte-identity of every norm gamma / layer_scalar against the bf16 safetensors and by coherent live
  output; an env escape hatch exists for exports that *do* bake `+1`.
  *Impact: capability-add — a new model family on the fast GGUF path; throughput comparable to the
  existing bespoke GGUF path (single-GPU TP=1 in the tens-of-tok/s range for the 31B Q4). No effect on
  existing paths — gated to the gemma4 GGUF arch, the Qwen3.5 GGUF path is untouched.*
- **GGUF loader generalization — registry + family tables (#129 S2) [landed, unpushed on `feat/gguf-loader-registry`]** —
  an internal-architecture / extensibility change, not a perf or behavior change. The fast bespoke
  loaders (Qwen3.5/3.6 + Gemma-4 dense) were refactored onto a shared `GGUFAdapterBase` that
  generalizes only the quality/speed-neutral scaffolding — arch resolution, the tensor-name-map
  skeleton, the F32 carve-out — while each family keeps its exact per-family transform body verbatim.
  Adding a new GGUF family is now a small table+hooks module plus one registry line, instead of a whole
  bespoke adapter.
  *Impact: loader extensibility only — no capability or speed claim. Verified **byte-identical** for
  both existing families (Qwen3.5/3.6 dense + Gemma-4): CPU exact (name-map + ordered transform-stream
  digest unchanged) and GPU machine-zero (token / logprob identical to the pre-refactor loader).*
  **Honest limit:** the base currently carries the trunk's older-style qwen35-MoE mapping; the
  collective-stacked MoE variant re-applies later as a table edit, not a re-architecture.
  **Scope / limits (plainly):** dense only; Q4_K_M is the verified quant, other quants are structurally
  supported but unverified; MoE Gemma-4 (`_exps` tensors) is not yet supported and fails fast.
  **Launch requirements (avoid the traps):** Gemma-4 requires `--attention-backend triton` (flashinfer
  is rejected by the model); the default CUDA device order is fastest-first, so reaching a specific
  physical GPU needs `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **GGUF-MoE expert-dim sharding, uneven** — whole experts per rank + zero-pad + remap,
  with GDN block coarsening and an MMQ fallback for misaligned blocks; A3B TP=3 coherent.
  *Impact: enables uneven-TP GGUF-MoE. Correctness/enablement.*
- **Tuned K-quant dispatch (upstream MMVQ/MMQ kernels, fork-tuned)** — the MMVQ/MMQ kernels are
  upstream; the fork tunes the dispatch (batched MMVQ, token-cutoff — see the perf overhaul below).
  TP=2 beats llama.cpp on decode.
  *Impact: **[better] faster than llama.cpp** at TP=2 on this rig; a decode-throughput win for the GGUF path.*
- **GGUF perf overhaul** — flat layout, batched MMVQ, Q8 lm_head, shape-aware dispatch,
  graph capture within budget. K-quant MMQ is now capped to small token counts, above which it
  dequantizes to cuBLAS.
  *Impact: **[better] ≈5-8x prefill throughput** on batched/long prompts vs the always-MMQ path (which was
  flat/slow); **[neutral] decode roughly unchanged** (bandwidth-limited).*
- **GGUF uneven-TP vec alignment [in progress]** — 16-element MLP units for indivisible intermediate sizes
  (e.g. 17408 at TP=3/5); unlocks dense GGUF under uneven TP (the last remaining old GGUF blocker).
  *Impact: enablement for dense GGUF under uneven TP. Correctness, no throughput claim.*

> **Honest downside for the whole GGUF path:** GGUF K-quant decode is bandwidth-limited and
> generally **slower than native FP8/AWQ** on the same hardware; the value is model/quant
> availability and llama.cpp compatibility, not peak speed.

## 5. Quantization

**Benefits:** all hardware. These are correctness fixes (some genuine upstream bugs) plus uneven-TP
alignment; the correctness half helps any user, matched or mixed.

- **AWQ / Marlin uneven-TP** — group/tile alignment, K-mask in the MoE Triton kernel, and a
  **Marlin zero-point staging fix for g=32** (a CUDA-kernel bug that affected every AWQ-g32 MoE).
  *Impact: correctness — AWQ-g32 MoE was silently wrong before; also unlocks AWQ under uneven TP.
  No throughput claim.*
- **compressed-tensors uneven group alignment** — quant-block-aligned units applied only to the
  layers that are actually quantized.
  *Impact: correctness/enablement under uneven TP.*
- **AutoRound-int4** — `weight_block_size` convention + a Marlin repack fix (Gemma vision
  geometry).
  *Impact: enables AutoRound-int4 (Gemma-4) that failed to load before. Enablement.*
- **Upstream bugfix — GPTQ-MoE does not load at TP>1 (stock sglang defect)** — in stock sglang, a
  GPTQ MoE model fails to load under **either even or uneven** tensor parallelism: `create_weights`
  allocates `w2_scales` at full width for the `desc_act=False` case, but the TP loader then shards it,
  producing an out-of-bounds narrow. The offending lines are from a pre-fork upstream commit, so this
  hits **anyone** running GPTQ MoE at TP>1 — a genuine upstream defect, not a fork-only issue, and a
  real upstream contribution candidate. Fix: shard `w2_scales` to `intermediate_size_per_partition`
  (the `desc_act=True` path is untouched) plus a per-rank group-alignment guard.
  *Impact: correctness — GPTQ MoE at TP>1 loads at all where stock sglang errored out. No throughput
  claim.*

## 6. MoE / Expert Offload

**Benefits:** all hardware for the **base capability** — running a model whose experts exceed VRAM on
any box. The **122B-A10B on 3 mismatched cards** end-state (offload combined with uneven-TP +
uneven-DCP) is the heterogeneous showcase (Category A) and is the concrete number cited in
*How much worse on homogeneous hardware?* above.

- **Per-expert MoE offload (pinned-host pool + wave prefetch) [landed, unpushed on `feat/weightless-kv-fastlane`]** —
  env-driven (`SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0`): all experts live in a pinned-RAM pool, the
  GPU keeps only `ceil(fraction · num_experts)` resident experts, and each forward LRU-prefetches the
  misses + remaps slots on-device over the unchanged grouped-GEMM. Wave processing is done **over
  tokens, not experts** — each token is computed whole in a single wave, so there is no *cross-wave*
  partial-sum accumulation (the remaining within-wave slot-remap reduction-order effect is covered
  under *Correctness* below). It composes
  with **uneven-TP + uneven-DCP** and is now **CUDA-graph compatible** — a decode-graph + eager-prefill
  hybrid. On this rig it runs the **122B-A10B GPTQ-Int4** (~61 GB of experts) across the 3 mismatched
  cards (5090 32 GB + 2×3080 20 GB, no NVLink) — a model that does not otherwise fit.
  *Impact: **[better] enables models that otherwise would not fit** (the base capability, any hardware);
  **[better] single-stream decode more than doubled** vs eager on the 122B end-state — from
  **6.97 tok/s eager to 16.34 tok/s** with CUDA-graph capture (+52%) plus hot-expert residency on top
  (≈+134% overall; TP=3, fraction=0.25). The win is launch-overhead elimination: the offload decode was
  launch-bound. On the 35B-A3B path the fixed-resident+scratch redesign measured **8.64 → 11.47 tok/s
  (+33%)**. The cost is throughput only, not quality (see below). Env levers:
  `SGLANG_MOE_HOT_RESIDENCY`, `SGLANG_MOE_OFFLOAD_CUDA_GRAPH`, `SGLANG_MOE_HOTSET_FILE`; default path
  unchanged.*
  - *Quality — validated, no measurable loss (#120): on Qwen3.6-35B-A3B-FP8 (TP=3 uneven, eager,
    temp=0, tiered fraction=0.5 → half the experts resident, half host-spilled) vs the fully-resident
    baseline: **[better] ≈+0.15% perplexity** (well inside FP8 reduction-order noise), **[better] needle-in-haystack
    100%** at 8k and 30k, **[better] correctness batteries 15/15 identical**. Run-to-run it is
    **self-deterministic** (byte-identical across repeats at the same fraction, verified at 96 and 256
    tokens); cross-config (fraction 0.25 vs 1.0) it agrees on confident tokens and diverges only at
    near-ties — the compacted resident+scratch slot layout re-associates the FP reduction vs the full-N
    layout, a sub-ULP effect (same class as the marlin floor below, far smaller magnitude), which is
    also what the +0.15% perplexity registers. Verdict: offloading experts does **not** measurably
    degrade quality — you pay in throughput, not accuracy.*
  - *Correctness of both quant paths (stated precisely, no overclaim): **neither offload path is
    bit-identical to the no-offload run** — both are coherent, self-deterministic and argmax-identical
    with divergence only at near-ties. They differ only in magnitude: the **FP8-triton** path's
    reduction-order delta is sub-ULP (agrees with the no-offload run on all but the tightest ties),
    while the **GPTQ/AWQ-marlin-Int4** path sits at marlin's intrinsic ≈1e-2 tiling floor. Both stay
    within their format's normal reduction-order noise — not a correctness bug. The
    CUDA-graph path adds nothing on top of the platform's normal capture-vs-eager floor that every
    graph path in the fork already carries. In short: the offload adds zero on top of the existing
    graph/marlin floor — this is not a claim that graph == eager bit-exact, nor that offload == no-offload
    bit-exact.*
  - *Throughput cost — [worse] still PCIe-bandwidth-bound, not compute-bound: **[worse] prefill ≈an order
    of magnitude slower** single-stream (eager; the spill cache fragments prefill into many tiny
    fetch-gated GEMMs). Decode used to be the other soft spot but is now much closer to resident after
    the CUDA-graph + hot-residency work above. Levers that shrink the rest: bigger spill cache, hot-set
    tuning, NVLink/P2P, more PCIe lanes.*
  - *Robustness (non-obvious): a single forward can legitimately need more unique experts than
    fit in the resident slots (a prefill batch easily touches all of them) — not obvious up
    front. Waving over tokens keeps each token whole in one wave, so this overflow case is **computed
    correctly (each token exactly once, with all its experts resident)** instead of silently evicting a
    still-needed expert.*
  - *Honest follow-ups (not built, listed for completeness): a **load-time streaming loader** —
    materialize only resident+cushion experts per layer during load and stream the cold tier straight
    into pinned host RAM, so the host never holds the full ~61 GB expert set at once — is **unbuilt**;
    it was not needed once the box had 108 GB of host RAM, and it remains the correct fix only for
    models larger than this (or tighter multi-process TP aggregates). **Per-rank offload sizing** (a
    different fraction per card) is **low value** here — the big card fills at the same fraction-pace,
    so a single global fraction is already near-optimal on this rig.*

## 7. Model Support

**Benefits:** all hardware. Each model runs on a matched box too; the uneven-TP=3 variants are the
heterogeneous slice.

- **Qwen3.6-27B hybrid-GDN** — GGUF + AWQ-INT4 + FP8, uneven TP=3, MTP.
- **Qwen3.6-35B-A3B (MoE)** — FP8/GGUF/AWQ coherent under uneven TP=3; the vehicle for
  PD-disaggregation testing.
- **Gemma-4 31B dense** — int4-AutoRound, TP=1 and uneven TP=3 (plan-aware vision tower, Triton
  head fix); EAGLE3 speculation.
- **Gemma-4 26B-A4B (MoE, SWA hybrid)** — boots (vision-ignore mapper, gated-GeLU Marlin);
  **`--swa-pool-sizing` cap: [better] ≈6x long-context** (50k needle proven), using Stage-A rather than
  SWA-DCP.
  *Impact: **[better] ≈6x more usable long-context** for this SWA-hybrid model vs the un-capped default.*
- **Small models under uneven-TP** — the replicated-KV GQA handling protects mini geometries
  (2B models + future draft models).
  *Impact: enablement/correctness for small-head-count models under uneven TP.*

## 8. Reproducibility, CUDA-Graph Correctness, Mamba/GDN & HiCache

**Benefits:** all hardware. The mixed-arch reproducibility work is heterogeneity-motivated, but the
CUDA-graph correctness fixes and HiCache safety are wins on any box.

Three **distinct** properties live in this section — kept separate rather than lumped under one loose
word "determinism", because they mean different things:

1. **Self-determinism** — run == run at a *fixed* config: the emitted output is byte-identical across
   repeats on the same GPUs with the same flags. It is a per-config property; it does **not** imply
   equality across different GPUs, TP degrees, or offload fractions.
2. **Reproducibility across mismatched GPUs** (spec-decode) — the *emitted* token sequence is
   identical regardless of which physical card holds which rank. This is **output-preserving
   reproducibility, not bit-determinism of the activations**: different silicon produces different
   low-order bits, and spec decode tolerates that because it is output-preserving (emitted tokens =
   the target model's argmax chain) and the accept/token decision is taken **once on rank 0 and
   broadcast**, so a numerically "noisy" draft only changes *how many* tokens accept, not *which*
   ones come out.
3. **CUDA-graph / uneven-DCP correctness fixes** — crash- and silent-corruption fixes for the
   captured-graph × uneven-DCP combination. These are correctness/robustness, not "determinism".

- **Reproducible spec-decode output across mismatched GPUs (property 2)** — NEXTN/MTP + EAGLE/EAGLE3
  speculative decoding emits the **same greedy token sequence** even when the ranks are physically
  different cards (run-to-run, cold == warm, independent of which card holds which rank). It is the
  *emitted output* that is reproducible, not the intermediate activations. Worth highlighting because
  the sources of divergence are genuinely **not obvious** — they only appear once ranks are different
  silicon (upstream never runs that way) and hide in three independent places, all fixed:
  (a) a greedy verify that silently relied on every rank's argmax matching — the verify result **and**
  every per-step draft pick are now **broadcast from rank 0** to cover the greedy branch, not just the
  sampling branch, so a near-tie can't make ranks accept different tokens and desync KV/recurrent
  state; (b) CUDA-graph pad/tail rows leaking stale tokens from a prior replay into the MoE
  grouped-GEMM on draft-extend replay — the input_ids / hidden_states tails are reset before replay;
  (c) a shared, wrapper-persistent flashinfer float workspace whose split-KV path reads regions the
  current forward never wrote — which made the output a pure function of request *order* — now zeroed
  at request boundaries (≈0.5 ms/step, env opt-out `SGLANG_FLASHINFER_ZERO_WORKSPACE_PER_REQUEST=0`).
  *Cost sub-millisecond per step. No throughput claim.*
- **Mixed-arch sampling divergence — precise scope, no overclaim** — the same near-tie hazard exists
  for *any* redundant per-rank sampling on different architectures (sm120 vs sm86 reduce in slightly
  different order, so an independent per-rank argmax can flip ≈1/1000 and silently diverge into
  word-salad / loops). In this fork it is the **spec-decode** path that closes it, via the rank-0
  broadcast above. For the **non-speculative** path there is **no fork-specific fix and none is
  claimed**: stock sglang already provides an env-gated token-id TP sync (an `all_reduce(MIN)`,
  default off) that a mixed-arch rig can turn on, and torch/inductor already keys its compiled
  artifacts on device properties (including compute capability), so there is no foreign-arch-artifact
  trap to fix in this codebase. Stated explicitly so this fork is **not** credited with a
  general-sampling or compile-cache fix it does not contain (that AOT-cache-keying work lives in a
  different codebase, not here).
- **CUDA-graph correctness under uneven-DCP (property 3, non-obvious robustness)** — the captured-graph
  × uneven-DCP combination surfaced two failure modes that stock sglang never hits (its verify path is
  pure paged and it has no DCP-in-graph), both fixed: (a) an MTP verify graph that ran the draft-chain
  attention through a **ragged flashinfer wrapper whose plan() froze a raw pointer to a transient
  qo_indptr** — the tensor was freed and its block reused, so replaying any bs>1 bucket read garbage
  indptr → illegal memory access (crashed hard at bs>1, while bs=1 survived only by allocator luck);
  and (b) captured decode/verify wrappers that **read fixed capture-time buffers the post-capture
  fast-decode plan skips writing** — so replay silently corrupted long outputs for **all**
  quantizations (first ~5-15 tokens correct, then repetition loops / token-id-0 garbage) while short
  probes passed. These are the "graph-replay / numeric-corruption fixes" the Category-B index refers
  to.
  *Impact: correctness/robustness — the CUDA-graph decode path is safe under the fork's uneven-DCP
  layouts. No throughput claim.*
- **Mamba/GDN self-determinism (property 1)** — checkpoint grid (`--mamba-checkpoint-interval`),
  deterministic resume/eviction on that grid, the flush == fresh-boot invariant, and an fp32 beta-gate
  in the fused GDN decode kernel.
  *Impact: correctness (self-determinism) — a resumed recurrent state and a freshly-booted one produce
  identical output. This is run-to-run stability of the GDN state cache at a fixed config, a **distinct**
  property from the cross-GPU reproducibility of property 2, not the same thing under a shared word.*
- **HiCache under uneven-TP / DCP** — index translation and layout normalization make the
  KV-offload cache safe under the fork's non-uniform layouts (upstream assumes uniform shards).
  *Impact: correctness/robustness. The non-uniform layouts also exercise the offload paths
  differently enough to surface two non-obvious concurrency hazards — an L3 write-back race and a
  prefetch deadlock — that the feature is hardened against. No throughput claim.*

---

## Guarded / descoped (kept, not shipped)

These were implemented and evaluated but deliberately gated off — either because they violate a
correctness invariant or because they are net-negative on this rig's interconnect. The code is kept
dormant behind a guard for a possible future fix, not removed.

- **Tree-spec `--speculative-eagle-topk > 1` under uneven-weighted-DCP [guarded] (#76)** —
  GPU-validated and found **silently wrong**: topk=1 (chain) is bit-deterministic and greedy-correct,
  but topk=4 is non-deterministic within a single boot and diverges from the topk=1 oracle, violating
  the lossless-greedy invariant. Root cause: the tree-masked verify-attention under weighted-DCP
  produces tree-topology-dependent verify logits (a draft node does not see exactly
  committed-prefix + true tree ancestors). **Also [worse] perf-negative on this rig** (≈-15% decode: tree
  compute overhead > accept-length gain over PCIe x4, serial). Restored as a hard fail-fast guard at
  arg validation, with a CPU unit test. *Reactivation needs both an audit of the draft→draft ancestor
  semantics under DCP and hardware with a better interconnect that makes trees net-positive.*
- **SWA-DCP Stage B [descoped]** — evaluated at only [better] ≈+6-10% (a modest win) for a large
  implementation cost — not worth it;
  reactivation criteria documented. Gemma-4 SWA long-context is served by the Stage-A pool cap in §7
  instead.

## Roadmap (planned / not yet built)

Planned, **not shipped** — listed here because the doc tracks features on two events: when they
*land* and when they are *accepted/planned*. Everything below is planned only.

### Weightless fast-lane roadmap (accepted, ordered)

Deliberately ordered by risk to output: **byte-identical speed/capacity gains first, quality-reducing
(lossy) ones last** — and every lossy item ships behind its own quality gate.

- **Byte-identical performance [planned]** — hide the offload's cost without changing outputs: **[better]
  double-buffered expert prefetch with compute-overlap** (hide the PCIe fetch behind compute; targets
  the ≈order-of-magnitude prefill slowdown); **[better] DCP-collective/compute overlap on the weightless
  lane** (hide the PCIe attention collectives behind head compute); **[better] bandwidth-aware DCP
  token-split** (balance the LSE-merge barrier); **[better] importance/frequency-based expert residency**
  (fewer PCIe fetches under skewed routing). All lossless — no throughput claim beyond "reduces the
  existing [worse] costs".
- **Spec-decode synergy [planned] (lossless)** — **[better] speculative expert prefetch driven by the draft router**
  (prefetch the experts the draft predicts the target will need); **[better] draft model on the idle
  weightless-worker cards** (put the slow cards' otherwise-idle compute to work on speculation).
- **Determinism / byte-identity CI harness [planned] (seed committed)** — boots the weightless lane against a
  full-TP baseline and asserts extend Δ==0 + benign decode fp-order; a regression guard, not a perf
  feature.
- **Quality-tradeoff features [planned] (deferred to LAST — each gated on its own quality check)** — these trade
  some accuracy for capacity/bandwidth and are intentionally deferred behind the lossless work:
  **fp8/int4 KV on the weightless workers** ([better] ≈2-4× KV capacity on small cards, [worse] lossy KV);
  **harder-quantized cold-expert spill tier** ([better] fewer PCIe bytes, [worse] lossy cold experts);
  **attention-sink + sliding-window KV eviction** ([better] long context in a fixed budget, [worse] drops
  mid-context); **compressed/fp8 KV on the PD/HiCache transfer path** ([better] ≈half the transfer bytes,
  [worse] lossy transfer).

### Other directional items

- **Tiered-KV fabric on the weightless lane (#134 B1/B2) [in progress]** — see §10; host-RAM KV tier so
  usable context exceeds combined VRAM (the ~262k target). Capability extension, PCIe-wall-bound, no
  throughput claim.
- **Cross-vendor HTCCL AMD bring-up [planned]** — see §11; the NVIDIA-only host-staging path is a
  NCCL-parity prototype, the actual NVIDIA+AMD mixed TP group is pending hardware.
- **PD v2 "lane scheduler" [planned]** — mixed-mode decode server + routing matrix: length routing,
  overflow prefill, sticky fast-lane decode + lazy migration, cache-affinity scheduling; end state
  is HiCache as a session medium (sessions live in the RAM tier, engines check them in/out,
  cross-lane prefix caching).
- **Draft-KV-pool DCP layout (`--draft-kv-layout`) [planned]** — unlocks speculation in disaggregated
  operation + granular draft-VRAM control.
- **Suspend-to-disk [planned]** — persist VRAM state (weights + KV + GDN) to disk to free/reclaim GPUs in
  seconds without re-capture.
- **GDN-state RAM offload [planned]** — under investigation; converges with the HiCache session medium.
- **Web frontend [planned]** — HF repo in → quant selection with a VRAM preview, config proposal
  (TP/DCP/PD/spec per model family), live max-KV estimator; hardware-generic (NVML profile, no rig
  constants).
- **122B-A10B-Int4 real run — done** — no longer roadmap: the 122B-A10B GPTQ-Int4 now runs on the
  3-card rig via the per-expert offload in §6 (composed with uneven-TP + uneven-DCP, CUDA-graph decode).
  Kept here only as a pointer to the shipped entry.
- **GGUF loader generalization (registry / family tables) — done** — no longer roadmap: shipped as
  #129 S2 (see §4), byte-identity-verified for both existing families. Kept here only as a pointer.
- **MoE Gemma-4 GGUF [planned]** — extend the Gemma-4 GGUF adapter to the MoE variant (`_exps` tensors),
  which the dense S1 loader currently rejects fail-fast.
- **Upstream adaptive-spec PR review/port, tree-spec topk>1 on better interconnect, uneven-TP over
  nodes/RDMA [planned]** — longer-horizon items.
- **Future — new-Mistral support (conditional, pending external release) [planned]** — support Mistral's
  next-generation "fat but sparse" MoE family (announced for July 2026 early access; open weights not
  yet public) **once it ships publicly** and **only if it fits this class of rig** — i.e. its smallest
  usable quant (Int4/Q4) must fit the VRAM + host-RAM-offload budget (roughly ≤65 GB via the
  load-time expert offload in §6). **[neutral] Conditional:** a ≈119B-scale sparse MoE at Int4 (≈60 GB) fits
  tightly via offload; a 675B-class model does not. Foundation already exists (parked): the
  config/format subsystem is complete, dense/Mixtral GGUF is already generic, plus three small
  execution fixes (lazy MLA/Pixtral import decouple, LlamaMLP `tp_family` for uneven-TP, Mixtral
  uneven-TP attention). Strictly forward-looking — current Mistral models were dropped as test targets
  (too old; nothing newer from Europe than Gemma 4), so this is **not** a shipped feature.
