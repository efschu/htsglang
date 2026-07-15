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

## Performance-oriented uneven split ("auto performance" mode)

Status: unimplemented, feasibility-gated (investigate first, see below).

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
  2=3080) one 3080 sits on **PCIe Gen2 x4 (~2 GB/s)** while the 5090 and the
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
