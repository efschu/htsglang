# htsglang Roadmap

Planned and in-investigation features for this sglang fork.

## Tensor parallelism wider than the KV-head count (TP > num_kv_heads)

Status: unimplemented, approach undecided.

### Goal

Today both normal and uneven Tensor Parallelism assume every rank receives at
least one whole KV (key/value) head. When `tp_size` exceeds the model's number
of KV heads, the current sharding cannot hand each rank a KV head. This is
increasingly common with GQA models that carry very few KV heads (e.g. 4-8).
The goal is to support `TP > num_kv_heads` while keeping the existing (uneven)
per-rank KV-cache split intact. The same limit applies to the GDN heads on the
linear-attention path.

Two candidate approaches, both open:

### Approach 1 - Split a KV head along head_dim

Partition a single KV head across ranks along its feature dimension
(`head_dim`), so multiple ranks share one KV head's channels.

Open questions:
- Does the attention backend (flashinfer / FlashAttention, and the GDN /
  linear-attention path) permit a KV head to be split along its feature dim,
  and does this force a cross-rank reduction/gather of partial attention
  results?
- What is the correctness and performance cost of that extra communication?

### Approach 2 - Synced KV-head clones

Replicate whole KV heads (and GDN heads) onto the extra ranks and keep the
clones in sync: each cloned KV head produces identical K/V, while query heads
are still split across ranks as usual.

Open questions:
- Memory overhead of replicating KV / GDN heads across ranks.
- How and where to keep the clones bit-identical (broadcast of computed K/V vs.
  independent recompute on each rank)?
- Interaction with the existing per-rank uneven KV-cache split.

### Testing / hardware note

Exercising `TP > num_kv_heads` on the available hardware (this box: 1x RTX 5090
+ 2x RTX 3080 = 3 physical GPUs) requires MORE ranks than GPUs, i.e. multiple
ranks per GPU. That path already exists via `--rank-gpu-id` with duplicate ids
(and the NCCL >= 2.30 multi-rank-per-GPU support). Co-locating several ranks on
one physical GPU may need the CUDA MPS server (`nvidia-cuda-mps-control`) for
acceptable concurrency.

Open question: is MPS actually required for co-located ranks, or is plain
process-level co-location (duplicate `--rank-gpu-id` entries) sufficient? This
is itself undecided and must be measured.

Cross-reference: existing multi-rank-per-GPU support via `--rank-gpu-id`
duplicates and NCCL >= 2.30 multi-rank communicators.

## Additional model families on this 72 GB heterogeneous rig

Status: not started - candidate screening planned.

### Goal

Identify current-generation model families beyond Qwen3.5/3.6-27B that
(a) fit the total VRAM budget of this rig - 72 GB across 1x RTX 5090 (32 GB)
+ 2x RTX 3080 (20 GB) under uneven TP=3 - with usable context headroom, and
(b) can be made compatible with the fork's feature set: uneven TP, weighted
uneven-DCP token sharding, MTP/NEXTN speculative decoding, CUDA graphs, the
GGUF/AWQ/FP8 quant paths, and mmproj vision loading.

### Named candidates (specs to verify at investigation time, not assumed)

- **Gemma 4 family** - the variants whose quantized weights fit ~72 GB minus
  per-rank reserves. Verify: attention geometry (kv-head count vs. TP=3
  whole-head splits, sliding-window layers), availability of FP8 / AWQ /
  GGUF(+mmproj) checkpoints, and whether any MTP/EAGLE-style draft head
  exists for spec decode.
- **Qwen3.6-35B-A3B** (MoE, ~3B active parameters) - attractive because the
  small active set promises fast decode on this rig while total weights
  (~35B) still fit quantized. The open question is the **expert layout under
  UNEVEN TP**: Expert Parallelism is out of scope for this fork, so experts
  must run TP-sharded (per-expert weight split across ranks, divisibility vs.
  the uneven per-rank ratios) or replicated - to be derived from the
  fork's split math. Family relationship to Qwen3.5/3.6-27B should make the
  text-stack port cheap; verify the MoE router + shared-expert handling.
- **Screen of other currently "valuable" architectures** at investigation
  time (releases move fast): anything with strong quality-per-VRAM that fits
  72 GB quantized. Explicitly out: models whose smallest useful quant exceeds
  the budget.

### Compatibility checklist per candidate

1. Weight footprint under FP8 / AWQ-INT4 / GGUF quants vs. 72 GB minus
   per-rank reserves (weights + activations + draft head).
2. Attention geometry: query/kv-head counts vs. whole-head uneven splits over
   TP=3 (GQA group preservation; kv-heads < 3 needs the "TP > num_kv_heads"
   work above).
3. Hybrid layers (Mamba/GDN/linear attention/SWA): pool-configurator support
   and the CP-free path under DCP.
4. MoE: expert weight layout under uneven ratios (TP-shard vs. replicate; EP
   explicitly out of scope).
5. Spec decode: MTP/EAGLE/NEXTN head availability (native or community
   checkpoints).
6. Vision: HF vision tower and/or GGUF mmproj availability.
7. Quant availability: FP8 checkpoints, AWQ, unsloth-style GGUF dynamic
   quants.
8. Estimated max context via the calibrated uneven-DCP pool math
   (C = min_r(P_r/ratio_r) x S).

### Deliverable

A compatibility matrix over the candidates, a ranked implementation-gap list
(what each model needs from the fork), and per-model bring-up tasks for the
winners.

## Uneven Tensor Parallelism across nodes over RDMA NICs (full feature set)

Status: not started - investigation planned.

### Goal

Extend uneven TP beyond a single machine: ranks distributed across multiple
nodes connected via RDMA-capable NICs (InfiniBand or RoCE), with the FULL
feature set working - uneven per-rank ratios, weighted uneven-DCP token
sharding, MTP/NEXTN speculative decoding, CUDA graphs, the FP8/AWQ/GGUF
quant paths, and mmproj vision. Not a degraded "multi-node but even split
only" mode: the same heterogeneous math, spanning nodes.

### Why this is harder than in-node

- **Every decode step pays the network.** The per-step collectives (TP
  all-reduce per layer, the DCP LSE all-gather) move from PCIe (~2-16 GB/s,
  microsecond-scale) to the NIC. Inter-node round-trip latency enters the
  per-token critical path; the split math must treat NIC latency/bandwidth
  as a first-class cost, not an afterthought.
- **GPUDirect RDMA caveat on this hardware class.** GeForce cards (3080,
  5090) do not support GPUDirect RDMA - NCCL falls back to host-staging
  (GPU -> host memory -> NIC). That doubles PCIe crossings per transfer and
  adds CPU copy cost. Cross-reference: the HTCCL work (branch feature/htccl)
  already built a vendor-neutral HOST-STAGING collective layer for
  NVIDIA+AMD TP - its staging path is the natural transport candidate here,
  or NCCL's net transport with careful tuning.
- **Rank mapping needs a node dimension.** --rank-gpu-id (and the ratio
  vector) must address node:gpu pairs; process launch, NCCL rendezvous and
  the NVML-based device identity checks must work across hosts.
- **Calibration gains a new axis.** The self-calibrating token vector today
  measures per-rank VRAM. Across nodes, the effective per-rank throughput
  also depends on which side of the NIC a rank sits on; the perf-split
  ("auto performance") objective and the RDMA topology interact - a rank
  behind a slow link should attract fewer tokens/heads even if its VRAM is
  large (same principle as the PCIe Gen2-x4 finding in-node, amplified).
- **DCP ownership across nodes.** The weighted owner rule is
  position-based and node-agnostic in principle, but prefix gathers for
  long-context extend cross the NIC; chunking/overlap strategies need
  investigation.

### Investigation order

1. Transport baseline: NCCL over the actual NICs (IB/RoCE), host-staging
   penalty measured (all-reduce latency/bandwidth table inter-node vs
   in-node, small-to-large sizes) - the decode-step viability check.
2. Decide transport: NCCL net path vs HTCCL host-staging extension.
3. Rank/node mapping + launch plumbing (two-host bring-up, even split
   first as a stepping stone, then uneven ratios).
4. Full feature matrix on two nodes: DCP, MTP, graphs (graph capture with
   inter-node collectives!), quants, vision.
5. Fold NIC cost into calibration and the perf-split objective.

### Hardware note

Requires a second RDMA-capable machine; current rig is single-node
(1x 5090 + 2x 3080). Test plan must name the second node's GPUs/NICs before
implementation starts.

## Performance-oriented uneven split ("auto performance" mode)

Status: feasibility CONFIRMED (M22 measurements, 2026-07-16) - build as a
PREFILL/THROUGHPUT mode. Key findings from the measured ratio matrix
(AWQ-INT4, MTP, CUDA graphs, weighted DCP, TP=3 auto as reference):

- **Decode is flat across splits (+-2%)**: the VRAM-auto split already sits
  near the bandwidth optimum for decode; over-concentrating makes the 5090
  itself the bottleneck (-6-8%). Decode is NOT the lever.
- **Prefill/aggregate is the lever**: in-ring MLP concentration
  (`--rank-mlp-ratio 5,1,1` on top of auto) is a STRICT win: +10% prefill,
  +7% concurrent, +20% active max_total, -1.3% converged context. Dropping
  the narrow-link card entirely (TP=2 @ 5:2 or 32:8) buys +55-76% prefill /
  +25-30% concurrent at -72% context.
- **Attention/GDN shifting is the wrong knob**: TP=3 attention can ONLY
  materialize [2,1,1] permutations (4 GQA units); GDN shifting drags the
  SSM pool with it (4.68 MiB/req/GDN-unit) and can collapse context
  (5:1:1 -> -92%). The optimizer's real degrees of freedom: MLP units
  (544, fine-grained, ~free), DCP token vector (64 units), TP degree.
  TP=4 attention is forcibly even; TP>=5 uneven does not boot.
- **TP-degree reduction must be an optimizer candidate**: TP=2 @ 32:8
  dominated the extreme TP=3 ratio on both axes.
- The `p x max-KV` objective parametrizes the measured Pareto front
  cleanly (p=1.0 -> auto + MLP concentration; p~0.28 -> TP=2 5:2;
  p~0.19 -> 32:8).
- Cost-model must include: SSM pool ∝ GDN-units x max-running-requests,
  BF16 weight families inside INT4 checkpoints, the draft's transient
  embed/lm_head duplicates (made MTP@TP=1 impossible on 32 GB), and
  --max-total-tokens as the headroom valve against prefill OOM.
- Link-map correction: the narrow 3080 is **Gen4 x4** (~2x penalty vs the
  x8 cards), not Gen2 x4 (~8x) as earlier notes claimed.

### Goal

Today the uneven split (`--rank-tp-ratio auto` + uneven-DCP KV token-sharding)
optimizes for **maximum VRAM / context** — it fills every card and hands the KV
pool as many tokens as the hardware allows. Add an alternative **"auto
performance"** mode that instead distributes the split for **maximum decode /
prefill throughput**, accepting a bounded loss of context.

The idea: shift more of the model/work toward the **faster** card (higher
compute + memory bandwidth) **and** the **better-connected** card (more PCIe
lanes / higher link generation), and leave **more free space on the slower or
slower-linked** card. Crucially, compute/VRAM-bandwidth and **link speed (PCIe
lanes)** are **separate** factors: a card can be fast in compute/VRAM yet be
starved by a narrow PCIe link (lanes limit it, not compute/VRAM). The split
should account for both independently.

Two independent gain sources, both to be weighed:
1. **Off-loading** a slow / link-limited card (as above).
2. **Splitting across fast cards even when it is not memory-forced.** If both
   cards are fast, splitting the model may raise decode/prefill throughput
   (extra compute + memory bandwidth via TP) **even at a small context that
   would fit entirely on one card** — as long as the TP compute/bandwidth gain
   outweighs the cross-rank communication overhead. So the mode must be able to
   "split for speed" without any memory need, not only to relieve a slow card.

### Parameters

- **`auto performance`** — select this split objective (max speed) instead of
  the default max-VRAM objective.
- **`max auto loose ctx in percent`** — the precondition/constraint: how much
  max-available context may be sacrificed, in percent, relative to the
  VRAM-optimal split. This defines a **hard floor**: the optimizer shifts load
  toward the fast/well-linked card only until the max-available context would
  fall below `(100 - X)%` of the VRAM optimum. There is an **automatic, natural
  lower bound** here — the floor is derived from the maximum-available context
  itself, not chosen arbitrarily; at `X = 0` only speed gains that do not reduce
  context are taken.
- **`max auto tune both|dec|enc`** — the tuning target: `both` (decode +
  prefill jointly), `dec` (decode throughput only), or `enc` (encode / prefill
  throughput only).

### Feasibility investigation (do this FIRST)

Before implementing, confirm the win is real on this hardware:
- Load the slow cards (the 3080s) only **very little** (split heavily toward the
  5090 + the well-linked card) and measure decode/prefill throughput vs. the
  balanced VRAM-optimal split.
- Quantify the role of the PCIe link. On this box (NVML order 0=3080, 1=5090,
  2=3080) one 3080 sits on **PCIe Gen4 x4** (corrected by M22 load
  measurement; earlier notes wrongly said Gen2 x4) while the 5090 and the
  other 3080 run **Gen4 x8 (~16 GB/s)** — an ~8x slower link on that one card.
  Measure per-card all-reduce / comm cost and the slow link's share of the step
  time; that card is the prime candidate to off-load under this mode.
- Decide whether the speed gained (more work on the fast/well-linked card,
  slow/link-limited card off-loaded) justifies the context lost.
- **More-than-3-card scenarios via code analysis.** This box has only 3 GPUs,
  so the >3-card behavior must be studied by **analyzing the code / the split
  math**, not on hardware. Investigate whether, for example, 4 cards split
  `2:2:2:1` or `3:2:1:1` would raise decode/prefill throughput — and how the
  best split shifts as a function of the desired maximum KV-cache size (a larger
  target context pushes toward a more even, capacity-first split; a
  speed-first target pushes toward concentrating work on the fast/well-linked
  cards). Derive the relationship from the code rather than measurement.

### Hardware note

Because compute-fast and link-fast are independent, the mode must read both
`pcie.link.width/gen` (or measured link bandwidth) and compute/VRAM-bandwidth
per physical GPU, not just VRAM size.

### Calibration & weighting design (how the weights are DERIVED)

For a trustworthy automatic weighting, the calibration must briefly MEASURE
each card - not read spec sheets - before deriving a distribution:

1. **Stage 0 - hardware probe (once per rig, not per boot).** A short
   (~2-3 s/card) micro-probe: max memory bandwidth (large-copy/triad), max
   compute throughput (GEMM burst at model-relevant shapes), and the full
   pairwise link matrix (collective latency + bandwidth per card pair - this
   is what exposes a Gen2-x4 straggler). Results are cached in a local
   hardware-profile file keyed by the GPU-UUID set + driver version, so the
   probe runs ONCE per hardware configuration and later boots reuse it
   silently; any hardware change invalidates the cache automatically.
   Measurement building blocks already exist (m20 GEMV + NCCL benches).
2. **Stage 1 - cost model.** Derive per-rank weights from the probe: decode
   is VRAM-bandwidth-bound, prefill is compute-bound, and the link matrix
   adds a per-rank communication penalty (a rank behind a slow link attracts
   fewer heads/tokens even if its VRAM is large). The tuning target
   (`both|dec|enc`) selects the blend of the three factors.
3. **Stage 2 - in-situ refinement (the part synthetic probes cannot do).**
   Synthetic numbers miss the actual workload mix: quant-kernel choice
   (MMVQ/MMQ vs marlin vs cutlass), the mamba/GDN share, the MTP draft loop.
   So after boot, measure the REAL per-rank step times over the first N
   decode steps and refine the vector once - the same converge-in-one-
   feedback-step mechanic the uneven-DCP token-vector calibration already
   uses.
4. **Pinning (skip calibration entirely).** Every derived weighting is
   printed as a pinnable startup hint (exactly the existing
   SGLANG_UNEVEN_TOKEN_VECTOR UX): pass the vector at launch and neither the
   probe nor the refinement runs again. Cache + pin together mean the full
   calibration cost is paid once per rig, ever.

### Benchmark-methodology notes (from the in-session design discussion)

- Ratio experiments (e.g. 5:1:1 vs 3:2:2) should use the AWQ16-INT4 model
  rather than FP8: the halved weight footprint keeps extreme ratios from
  starving the small cards' KV, so the speed effect of the ratio is measured
  rather than a memory cliff.
- A TP=2 @ 5:2 layout (dropping the third card entirely) eliminates the slow
  x4 link from the collective ring altogether and is a legitimate
  perf-split candidate - only feasible with the smaller INT4 weights.
