# htsglang — GPU Topologies and the Setting That Wins Each One

Companion to [FEATURES_VS_UPSTREAM.md](FEATURES_VS_UPSTREAM.md). That document is the
capability inventory; this one is the *placement* guide. It has two parts:

1. **The hero config (granular):** the measured **122B-A10B-Int4 on TP=3** end-state on the
   reference rig (1× RTX 5090 32 GB + 2× RTX 3080 20 GB, PCIe, no NVLink), drawn down to
   individual decoder layers, KV heads, per-token KV bands, the MoE expert-offload flow, and
   where MTP and the CUDA-graph boundary sit — each paired side-by-side with what **upstream
   sglang** does on the same three cards.
2. **The capability atlas (general):** the same ideas across a range of 2-8 GPU
   constellations of mixed VRAM / compute / interconnect.

Every capability is grounded in FEATURES_VS_UPSTREAM.md (section numbers in parentheses); the
concrete 122B numbers are the measured end-state run of the per-expert offload path (which is
still §6 **[in progress]**). Anything not yet shipped is labelled **[in progress]** or
**[planned]** exactly as in the parent doc. Impact/throughput figures are the parent doc's
rounded, directional numbers on this PCIe rig — not benchmarks.

## How to read the diagrams

GPU boxes are drawn with **height scaled to VRAM**; a separate horizontal bar is **pinned
host RAM (DDR)**. Interconnect lines: solid = PCIe x16, thin dashed brown = PCIe x4; this rig
has **no NVLink**, so collectives run over PCIe host-staging. Colour meaning is consistent
across every diagram:

- blue — model-weight shard (Q-heads, FFN, dense weights)
- green — KV cache / KV heads (on-GPU)
- amber — resident MoE experts (kept on the GPU)
- light amber — expert scratch / prefetch staging slots
- red — MoE experts spilled to pinned host RAM
- teal — host-staged / transferred KV (PD handoff)
- grey-blue — free / reserve headroom; grey — CUDA context + overhead
- steel — full-attention layer (holds KV); violet — GDN / linear-attention layer (holds state)
- light violet — GDN recurrent state; pink — MTP / NEXTN draft head
- pale red-grey — a region an upstream even-TP layout leaves unused on this hardware, or a
  configuration its divisibility constraints do not admit here

Region sizes inside a card are illustrative partitionings, **except** where a figure is quoted
from the measured run or the parent doc (per-rank GB, expert counts, tok/s). Every "our-mode"
diagram is paired with its **upstream sglang** counterpart, drawn as a real layout — a concrete
default configuration, not a bare "can't".

---

# Part 1 — The hero: 122B-A10B-Int4 on TP=3 (measured end-state)

Model: `Qwen3.5-122B-A10B-GPTQ-Int4` — 48 decoder layers, hybrid-GDN (~3:1), 256 experts per
layer, top-8 routing, ~10B active, **2 GQA KV heads**, 32 Q heads, FFN intermediate 1024, GPTQ
`group_size` 128. Launch (measured): `--tp-size 3 --dcp-size 3 --rank-tp-ratio auto
--rank-gpu-id 0,1,2 --rank-gpu-memory-mib 28000,19000,19000 --dtype float16 --attention-backend
flashinfer --disable-cuda-graph`, `SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.25`. The per-rank MiB
budgets are **[28000, 19000, 19000]** (the vector `--rank-tp-ratio auto` derives as its raw
weights); on this GPTQ-Int4 model the group-alignment constraint (each rank's FFN shard a whole
multiple of `group_size` 128) resolves the operative shard ratio to **2:1:1**. Measured: per-rank
GPU **25.5 / 16.6 / 15.4 GB**, host floor **24.4 GB**, **6.97 tok/s** (eager bring-up run of the
§6 offload path, measured on the offload validation battery), coherent, self-det 5/5.

> Correctness bar for this GPTQ-Int4 (marlin) path is **coherence + self-determinism +
> divergence only at near-ties** — a ~1e-2 argmax delta is intrinsic to marlin-Int4 tiling and
> is not a regression. The FP8 path in §6 shares the same *class* (self-deterministic, near-tie
> divergence only) but at a far smaller, sub-ULP magnitude; neither offload path is bit-identical to
> the no-offload run. No bit-identity is claimed here.

## 1.1 The model both stacks run

<img src="topologies/10-model-layer-stack.svg" alt="48-layer hybrid-GDN stack: full-attention (KV) layers vs GDN state layers, 3:1, plus the MTP head" width="100%">

The 48 layers alternate **GDN / linear-attention** (36 layers, a fixed recurrent state, no
per-token KV growth) with **full-attention** layers (12 layers, ~1 in 4, carrying the KV
cache). The **MTP / NEXTN** draft head (§3, shipped) is one extra layer on rank0. This
structure is identical under htsglang and upstream sglang — the entire difference is in **how each
layer is split** across three mismatched cards, which is what the rest of Part 1 shows.

## 1.2 One layer's tensors across 3 cards — uneven-TP vs even-TP

<img src="topologies/11-tp-tensor-split.svg" alt="Per-layer tensor split: htsglang uneven 16/8/8 Q and 512/256/256 FFN vs upstream even-TP=3 non-divisible on these dimensions" width="100%">

**htsglang** (`--rank-tp-ratio auto`, §1 shipped) sizes each rank's shard to its VRAM budget.
On this GPTQ-Int4 model the shard ratio group-aligns to **2:1:1**: **Q heads 16/8/8** (of 32),
**FFN intermediate 512/256/256** (of 1024) — a whole multiple of `group_size` 128 on every rank,
i.e. **4/2/2 GPTQ groups**. The **2 KV heads** are below the rank count, so they are replicated
(next diagram).

**Upstream sglang** splits each dimension equally across ranks. On these exact dimensions an even
TP=3 does not divide: 32 Q, 1024 FFN and 2 KV heads are not multiples of 3, and 1024/3 = 341.3 is
not `group_size`-128-aligned, so the GPTQ group-alignment check does not admit an even TP=3 here.
Per-rank uneven sharding is not part of the upstream path, so a TP=3 on this model uses the fork's
ratio-based split.

## 1.3 2 KV heads across 3 ranks — the DCP token-shard (the crux)

<img src="topologies/12-attention-dcp.svg" alt="2 KV heads replicated on each rank, KV token-sharded, Q broadcast, partial attention, LSE-merge; upstream head-shard needs kv_heads>=tp" width="100%">

This is the decisive idea. With only **2 GQA KV heads** but **TP=3**, the KV cache cannot be
head-sharded (2 < 3). htsglang instead **replicates the 2 KV heads on every rank** and shards
the KV cache along the **token axis** (`--dcp-size 3`, uneven-DCP, §1 shipped): each rank owns a
token band of the sequence's KV, the new token's **Q is broadcast** to all ranks, each rank
computes a **partial attention + local LSE** over its band, and an **LSE-merge** collective
stitches the partial softmaxes into the final output. What is duplicated is only the small
2-KV-head slice; the KV *tokens*, and the query heads, are still split — so aggregate KV
capacity scales with the rank count.

**Upstream sglang** shards KV along the head axis, which requires `num_kv_heads ≥ tp`: with 2
heads and 3 ranks the head-axis split does not cover rank2. Replicating the full KV on every rank
is possible and stores the whole cache on each card (no token-capacity gain from adding cards);
the token-axis KV split is specific to the fork.

## 1.4 KV-cache arrangement across ranks, and where MTP sits

<img src="topologies/13-kv-token-layout.svg" alt="Per-token KV bands across ranks for the 12 full-attention layers, MTP KV band on rank0, GDN state blocks; upstream replicated 1x" width="100%">

Only the **12 full-attention layers** hold a KV cache; the fork lays it out as **token bands**
per rank (rank0 gets the largest band, matching its shard). The **MTP draft head keeps its own
small KV band on rank0**. The **36 GDN layers** hold a fixed recurrent state instead (a small
per-rank block, independent of sequence length). Per-rank KV memory is `12 layers × 2 KV heads
× that rank's token band`, so growing the context adds tokens split three ways — aggregate KV
capacity scales with ranks (§1).

**Upstream sglang** either head-shards the KV (requires `num_kv_heads ≥ tp`, i.e. ≥3 here) or
replicates the whole KV on every rank (1× capacity, no token-capacity gain from adding cards).

## 1.5 Per-layer MoE expert offload — 256 → 64 resident + 16 scratch + 176 host

<img src="topologies/14-expert-offload-flow.svg" alt="Expert offload flow: 64 resident + 16 scratch on GPU, 176 spilled to host RAM, top-8 gather, PCIe H2D into scratch, grouped GEMM; eager; upstream keeps all experts resident" width="100%">

The model's experts do not fit VRAM, so `SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.25` (§6 **[in
progress]**) keeps only **ceil(0.25·256) = 64** experts resident per layer, plus a **16-slot
scratch** buffer (buffer = 80); the other **176** live in **pinned host RAM** (per layer × 48 →
host floor 24.4 GB). Per token-wave the router's **top-8** experts are gathered: resident ones
read directly, **spilled ones are fetched H2D over PCIe into scratch**, then the grouped GEMM
runs. Waving over tokens (not experts) keeps it coherent (§6).

Because the expert set is **data-dependent per wave**, the decode path is **not
CUDA-graph-capturable** → this run uses **`--disable-cuda-graph` (eager)**, consistent with §6's
"eager-only by design". Graph-capture for the offload path is **[planned]**.

**Upstream sglang** does not offer a per-expert, quant-aware host spill. The model's weights are
~65 GB; the three cards total 72 GB of VRAM, but that VRAM cannot be pooled for a single shard,
and per-card CUDA context plus KV and activations must also fit — so keeping all 256 experts
resident does not leave room for a working KV cache on these cards (and, as in 1.2, an even TP=3
does not divide these dimensions). The generic `--cpu-offload-gb` path is layer-granular and
dequantizes on CPU (no per-wave expert prefetch).

## 1.6 The whole rig, measured — vs the upstream default path

<img src="topologies/15-endstate-rig-map.svg" alt="Full 122B TP=3 rig map: 3 cards with measured GB, host spill bar, eager banner, 6.97 tok/s; upstream side lists the structural constraints" width="100%">

Everything above, on the real config: uneven-TP (budgets [28000,19000,19000] MiB, group-aligned
shard ratio 2:1:1), 2 KV heads replicated + DCP token-shard, per-layer expert offload **f=0.25**
(64 resident + 16 scratch, 176 host), MTP on rank0, **eager** decode. Measured per-rank GPU
**25.5 / 16.6 / 15.4 GB**, host floor **24.4 GB**, **6.97 tok/s** (above the single-card
solo-offload run's 4.8 tok/s).

On the upstream path, a TP=3 configuration on these three cards meets several structural
constraints: even-TP=3 does not divide these dimensions (1.2); head-axis KV sharding requires
`num_kv_heads(2) ≥ tp(3)` (1.3); and without a per-expert host spill the ~65 GB of weights leaves
no room for KV once per-card context is counted (1.5). An even-TP=2 fallback sizes both ranks to
the 3080's shard, which leaves ~12 GB of the 5090 unused and still does not fit the full model.

## 1.7 The same machinery at other scales

<img src="topologies/16-solo-and-35b.svg" alt="Solo 1-card 122B at f=0.15 (39 resident) with host spill, and 35B-A3B TP=3 with 2/1/1 groups" width="100%">

The same offload + uneven-TP path runs at other scales. **Solo 122B on one 5090** uses **f=0.15
(39 resident experts/layer**, host floor ~28 GB) — it boots the 122B on a single 32 GB card
(§6). The **35B-A3B-GPTQ-Int4** runs uneven-TP=3 (**2/1/1** GPTQ groups) with offload across mixed
architectures and was validated stage(a) **32/32 token-identical to TP=1** under uneven-TP+DCP,
self-det 5/5, 11.83 tok/s (§6/§7).

---

# Part 2 — Capability atlas (general topologies)

The same ideas across a range of 2-8 GPU constellations. For each, the canonical hardware where
that setting is the decisive advantage, and what the upstream default path does on it.

## Summary — topology class → winning setting → what the upstream default path does instead

| Topology class | Decisive fork setting | Source | What the upstream default path does |
|---|---|---|---|
| 2 mismatched GPUs, PCIe | `--rank-tp-ratio` (proportional shards) | §1 | Even TP only; both ranks sized to the smaller card, leaving the larger card's extra VRAM unused |
| 3 mismatched GPUs, more KV wanted | `--rank-tp-ratio auto` + uneven-DCP token KV | §1 | Even TP + head-axis KV; equal shards, so the larger card is filled to the smaller card's size (≈2.3-2.8× less token-axis KV context, measured on the 27B run) |
| Model with fewer KV heads than ranks | TP > num_kv_heads (replicate KV + token-shard) | §1, §9 | Head-axis KV bounds TP at `num_kv_heads`; additional ranks are not covered by the head split |
| MoE with more experts than fit VRAM | resident-fraction expert offload **[in progress]** | §6 | Generic `--cpu-offload-gb` (layer-granular) or EP (experts must fit aggregate VRAM) |
| Model too big for any single card | load-time MoE offload to host RAM **[in progress]** | §6 | No host-spill load path; the experts must fit aggregate VRAM (EP) or a single card |
| Long-context priority | weightless-KV fast lane (Variant C, landed) | §10 | Every rank holds its assigned layer weights; each card's VRAM is shared between weights and KV |
| More ranks than physical cards | multi-rank co-location (`--rank-gpu-id` duplicates) | §9 | TP bounded by physical GPU count; one rank per physical card |
| A slow PCIe x4 link in the rig | single-node PD-disaggregation | §2 | A single fused TP group runs prefill collectives over every link in the group, including the x4 lane |
| 8-GPU mixed fleet | several of the above combined | §1/§6/§9/§10 | Even TP sizes the whole group to the smallest card's shard; the token-KV / weightless / host-offload paths are fork-specific |

## 2.1 Two mismatched GPUs, PCIe — uneven Tensor Parallelism

<img src="topologies/01-uneven-tp.svg" alt="Uneven TP across a 32 GB and a 20 GB card, versus upstream even-TP sizing both ranks to the smaller card" width="100%">

`--rank-tp-ratio` (§1) sizes the TP shards in proportion to each card (8:5), so the 5090 carries
a bigger shard and KV slice than the 3080. Upstream even-TP sizes both ranks to the 3080's shard,
leaving ~12 GB of the 5090 unused (the dashed region).

## 2.2 Three mismatched GPUs — uneven-TP auto + uneven-DCP

<img src="topologies/02-uneven-dcp.svg" alt="Three mismatched cards with token-sharded KV proportional to each card" width="100%">

`--rank-tp-ratio auto` fills every card and sets DCP = TP; KV follows the **token axis**, so KV
capacity is decoupled from the weight split (**≈+2.3-2.8× token-axis KV context** vs an equal
head-axis split — measured 2.27-2.81× on the 27B uneven-DCP runs, §1; measured cost ≈-10-25%
decode from the PCIe collectives, i.e. 75-89% of the no-DCP decode rate). Upstream even-TP +
head-axis KV uses equal shards and requires `num_kv_heads` divisible by the rank count, so the
larger card is filled to the smaller card's size.

## 2.3 Fewer KV heads than ranks — TP > num_kv_heads

<img src="topologies/03-tp-gt-kvheads.svg" alt="TP=3 on a model with 2 KV heads: replicate the KV heads and token-shard" width="100%">

The general form of 1.3: replicate the few KV heads, token-shard + LSE-merge (§1/§9). The
duplicated slice is a single-digit-% KV overhead; upstream head-axis TP is bounded by
`num_kv_heads`, so additional ranks are not covered by the head split.

## 2.4 MoE bigger than VRAM — per-expert host offload + uneven-TP

<img src="topologies/04-moe-expert-offload.svg" alt="Resident experts on-GPU, cold experts spilled to pinned host RAM, fetched per token-wave" width="100%">

The general form of 1.5: a resident fraction on-GPU, the rest in pinned host RAM, prefetched per
token-wave (§6 **[in progress]**). The FP8 path is quality-neutral (#120: ≈+0.15% ppl within FP8
reduction-order noise, 15/15 batteries, needle 100%) and self-deterministic — not bit-identical to the
no-offload run, but diverging only at near-ties (sub-ULP); the cost is throughput rather than
quality (the offload path is slower than no-offload; the exact FP8 decode factor is
config-dependent and not separately benchmarked). Upstream offers the generic
`--cpu-offload-gb` path or Expert Parallelism (experts must fit aggregate VRAM).

## 2.5 Model too big for any single card — load-time offload to host RAM

<img src="topologies/05-122b-host-spill.svg" alt="122B-A10B on one 32 GB card with experts pinned in host RAM" width="100%">

The general form of 1.7 (solo): materialise only resident+cushion experts on the GPU and stream
the cold tier straight to host RAM at load (§6 head-rank load-time offload **[in progress]**) —
boots what would otherwise not fit at load. The upstream path has no load-time host-spill, so a
122B does not fit a single 32 GB card without it.

## 2.6 Long-context priority — weightless-KV fast lane

<img src="topologies/06-weightless-kv-lane.svg" alt="Fast card holds the full model TP=1; slow cards hold only KV token-shards" width="100%">

The fast card holds the full model as collective-free TP=1; the slow cards become **weightless
KV workers** holding only a KV token-shard (§10, Variant C, landed; CUDA-graph decode landed in
#133/#136). **≈14 GB of weights freed per worker** (measured) becomes available for KV; the
host-spill KV tier was measured to **262k tokens** on the 27B test model (#134). Single-shot
prefill on this lane is bit-identical to full-TP=1; decode/extend is argmax-identical (decode-class,
not bit-0). The resulting context multiple depends on the KV budget and is not separately
benchmarked. Upstream holds the layer weights on every rank.

## 2.7 More ranks than cards — multi-rank co-location

<img src="topologies/07-multi-rank-colocation.svg" alt="Multi-rank co-location: two ranks sharing the 5090 (illustrative)" width="100%">

`--rank-gpu-id 0,0,1,2` places two ranks on the 5090 as two processes (§9, co-location #82); NCCL
multi-rank is auto-set and a physical-impossibility check guards it. The first booted co-location
on this rig was **TP=4** with two ranks sharing the 5090 (via MPS + a side-loaded NCCL 2.30.7);
larger co-located layouts such as TP=5 (the illustrative diagram) remain emulation-stage and have
not been booted here. Replicated-KV itself (kv<TP, #62) is proven separately at TP=3 (§2.3, the
hero run). Co-located ranks share silicon — capability, not extra bandwidth. Upstream bounds TP at
the physical card count (one rank per card).

## 2.8 A slow PCIe x4 link — PD-disaggregation placement

<img src="topologies/08-pd-disagg-slow-pcie.svg" alt="Prefill solo on the x16 fast card; decode distributed on the x4 cards" width="100%">

Put prefill solo on the fast x16 card (zero cross-GPU traffic); decode distributed on the x4
cards, KV handed off via `mooncake_tcp` loopback (§2). Faster TTFT is expected because prefill
avoids the x4-lane collectives, and decode is largely unchanged — **the TTFT factor here is an
estimate, not benchmarked on this rig**. Upstream fuses one TP group, so prefill collectives run
over every link in the group including the x4 lane.

## 2.9 Eight-GPU mixed fleet — several capabilities combined

<img src="topologies/09-eight-gpu-fleet.svg" alt="Eight mixed GPUs combining uneven-TP, uneven-DCP, expert offload, weightless workers and PD placement" width="100%">

On a fleet spanning 11-32 GB and three PCIe speeds the settings **compose**: uneven-TP shards,
uneven-DCP token-KV, weightless-KV workers on the x4 cards, per-expert offload to host RAM, and
prefill on the fast x16 5090. Upstream even-TP sizes the whole group to the 11 GB card's shard (or
excludes that card); the token-KV, weightless, and host-offload paths are fork-specific.

---

## Caveats

- Diagrams are schematic. VRAM box heights are to scale; internal region splits are illustrative
  unless a figure is quoted from the measured 122B run or the parent doc.
- The 122B TP=3 numbers (per-rank GB, host floor, 6.97 tok/s, self-det 5/5) are the measured
  end-state run of the per-expert offload path, which is itself §6 **[in progress]**; this is a
  bring-up/validation run of an in-progress feature, not a shipped production mode.
- Neither offload path is bit-identical to the no-offload run: the GPTQ-Int4/marlin path carries a
  ~1e-2 argmax delta intrinsic to marlin tiling, and the FP8 path a sub-ULP reduction-order delta
  from the compacted slot layout. Both are self-deterministic with divergence only at near-ties; §6
  states the FP8 path as quality-neutral, not byte-identical.
- Impact figures are the parent doc's rounded, directional numbers on the PCIe reference rig (no
  NVLink). A better interconnect changes the cost side of every cross-GPU feature.
- `[in progress]` (expert offload, load-time MoE offload) and `[planned]` (graph-capture for the
  offload path) are labelled as in the parent doc — not shipped.
- The SVGs are regenerated by [`topologies/gen.py`](topologies/gen.py); they are self-contained
  (inline shapes + text only, no external fonts or images).
