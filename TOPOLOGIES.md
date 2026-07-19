# htsglang — GPU Topologies and the Setting That Wins Each One

Companion to [FEATURES_VS_UPSTREAM.md](FEATURES_VS_UPSTREAM.md). That document is the
capability inventory; this one is the *placement* guide. For a set of concrete multi-GPU
constellations (2 to 8 cards, mixed VRAM / compute / interconnect) it names the single fork
setting that is the decisive advantage on that hardware, shows how VRAM and pinned host RAM
get partitioned, and states plainly what stock sglang is forced to do on the same rig.

Every capability referenced here is grounded in FEATURES_VS_UPSTREAM.md; the section number
in parentheses points at the source entry, and anything not yet shipped is labelled
**[in progress]** or **[planned]** exactly as it is there. Nothing here is a benchmark: the
impact figures are the same rounded, PCIe-rig, directional numbers as the parent doc.

## How to read the diagrams

Each diagram draws every GPU as a box whose **height is scaled to its VRAM** (a 32 GB card
is taller than a 20 GB card by that ratio), partitioned into labelled regions. A separate
horizontal bar, where relevant, is **pinned host RAM (DDR)** holding spilled experts or
staged KV. Interconnect is drawn as lines between cards and down to host RAM: a solid medium
line is PCIe x16, a thin dashed brown line is a slow PCIe x4 lane. This reference rig has
**no NVLink** — collectives run over PCIe host-staging, which is why every cross-GPU step
carries a real cost in the parent doc.

Colour meaning is **identical in every diagram**:

- blue — model-weight shard (TP)
- green — KV cache held on the GPU
- amber — resident MoE experts (kept on the GPU)
- light amber — expert scratch / prefetch staging buffer
- red — MoE experts spilled to pinned host RAM
- teal — host-staged / transferred KV (PD handoff)
- grey-blue — free / reserve headroom
- grey — CUDA context + framework overhead

Region *sizes* inside a card are illustrative partitionings of that card's VRAM, not measured
allocations, except where a number is quoted from the parent doc (e.g. the ≈14 GB freed per
weightless worker, the ≈61 GB of 122B experts). The wins and costs in prose are the parent
doc's figures.

The canonical heterogeneous rig used for most scenarios is the reference machine: **1× RTX
5090 (32 GB, fast, sm120) + 2× RTX 3080 (20 GB each, sm86), no NVLink, PCIe**. Scenarios 8
and 9 generalise to other card mixes and interconnects.

---

## Summary — topology class → winning setting → what stock sglang does instead

| Topology class | Decisive fork setting | Source | What stock sglang is forced to do |
|---|---|---|---|
| 2 mismatched GPUs, PCIe | `--rank-tp-ratio` (proportional shards) | §1 | Even TP only → both ranks capped to the small card; the big card's extra VRAM is wasted |
| 3 mismatched GPUs, more KV wanted | `--rank-tp-ratio auto` + uneven-DCP token KV | §1 | Even TP + head-axis KV; can't fill the big card, ≈2.5-3× less usable context |
| Model with fewer KV heads than ranks | TP > num_kv_heads (replicated KV + token-shard) | §1, §9 | TP capped at `num_kv_heads`; extra cards cannot join the group |
| MoE with more experts than fit VRAM | resident-fraction expert offload **[in progress]** | §6 | Only generic `--cpu-offload-gb` (layer-granular, not quant/MoE-aware, slow) or EP (experts must fit aggregate VRAM) |
| Model too big for any single card | load-time MoE offload to host RAM **[in progress]** / 122B run **[planned]** | §6 | Cannot run — OOMs at load; EP needs the experts to fit aggregate VRAM |
| Long-context priority | weightless-KV fast lane (Variant C, landed) | §10 | Every rank must hold layer weights; slow cards spend VRAM on weights instead of KV |
| More ranks than physical cards | multi-rank co-location (`--rank-gpu-id` duplicates) | §9 | TP bounded by physical GPU count; can't place two ranks on one card |
| A slow PCIe x4 link in the rig | single-node PD-disaggregation | §2 | One TP group forces prefill collectives over the x4 lane too, throttling TTFT |
| 8-GPU mixed fleet | several of the above combined | §1/§6/§9/§10 | Even TP drops the whole group to the smallest card's shard; none of the token-KV / weightless / offload paths exist |

---

## 1. Two mismatched GPUs, PCIe — uneven Tensor Parallelism

<img src="topologies/01-uneven-tp.svg" alt="Uneven TP across a 32 GB and a 20 GB card, versus stock even-TP wasting the big card" width="100%">

**Rig:** 1× RTX 5090 (32 GB) + 1× RTX 3080 (20 GB), PCIe, no NVLink.

**Winning setting:** `--rank-tp-ratio` (§1). The tensor-parallel weight shards are sized in
proportion to each card (here 8:5), so the 5090 carries a bigger shard and a bigger KV slice
than the 3080. Attention/GDN are split in whole KV-head units per rank. This is the
enablement primitive the rest of the fork builds on — it makes TP across mismatched GPUs
possible at all.

**Physical constraint it resolves:** an equal split forces every rank to the size that the
*smallest* card can hold. With a 32 GB and a 20 GB card, an equal split is bounded by 20 GB,
stranding ~12 GB on the 5090.

**Stock sglang:** TP is even-only. Both ranks are capped to the 3080's shard size; the extra
~12 GB on the 5090 (drawn as the dashed "WASTED" region) is unusable. In practice this pushes
upstream users toward homogeneous GPUs.

## 2. Three mismatched GPUs, more context wanted — uneven-TP auto + uneven-DCP

<img src="topologies/02-uneven-dcp.svg" alt="Three mismatched cards with token-sharded KV proportional to each card" width="100%">

**Rig:** 1× RTX 5090 (32 GB) + 2× RTX 3080 (20 GB), PCIe.

**Winning setting:** `--rank-tp-ratio auto` (§1). It derives VRAM-optimal weights from NVML
budgets, fills every card, and sets DCP = TP automatically. The KV cache follows the **token
axis** (uneven-DCP token sharding), so KV capacity is decoupled from the weight split and
each card owns a KV-token slice proportional to its free VRAM.

**Physical constraint it resolves:** on a naive equal split the usable context is bounded by
the smallest card. Filling the 5090 proportionally and splitting KV by token yields the
parent doc's **≈+2.5-3× KV-cache context** over that equal split. The honest cost is stated in
§1: **≈-10-25% decode** from the extra per-step collectives over PCIe.

**Stock sglang:** even TP plus head-axis KV. It needs equal shards *and* `num_kv_heads`
divisible by the rank count, cannot fill the 5090, and offers no way to grow KV along the
token axis.

## 3. Model with fewer KV heads than ranks — TP > num_kv_heads

<img src="topologies/03-tp-gt-kvheads.svg" alt="TP=3 on a model with 2 KV heads: replicate the KV heads and token-shard" width="100%">

**Rig:** 1× RTX 5090 + 2× RTX 3080, TP=3, a GQA model with only **2 KV heads**.

**Winning setting:** the replicated-KV path (§1, validated at TP=5 in §9). When the TP degree
exceeds the model's KV-head count, the few KV heads are **replicated** across the ranks that
share a KV-head group and the KV is **token-sharded + LSE-merged**. Query heads (which
dominate) are still sharded normally.

**Physical constraint it resolves:** you cannot head-shard 2 KV heads across 3 ranks. Without
replicated KV, TP is structurally capped at `num_kv_heads`. The duplicated part is only the
small KV-head slice (the salmon region), which the doc rates as a **minor, single-digit-% KV
overhead** — the bulk of the KV is still split by token.

**Stock sglang:** head-axis TP caps the degree at `num_kv_heads` (= 2 here); the third card
simply cannot join the group.

## 4. MoE with more experts than fit in VRAM — per-expert host offload

<img src="topologies/04-moe-expert-offload.svg" alt="Resident experts on-GPU, cold experts spilled to pinned host RAM, fetched per token-wave" width="100%">

**Rig:** 1× RTX 5090 + 2× RTX 3080 running a 35B-A3B MoE whose experts exceed the per-card
budget.

**Winning setting:** resident-fraction expert offload, `SGLANG_MOE_RESIDENT_EXPERT_FRACTION <
1.0` (§6, **[in progress]**). A fixed set of experts stays resident in VRAM (amber); the rest
live in a pinned host-RAM pool (red) and are LRU-prefetched per forward, remapped over the
unchanged grouped-GEMM. Waving is done **over tokens, not experts**, which is the key to
byte-identity.

**Physical constraint it resolves:** the experts do not fit VRAM at the desired context. The
doc reports this is **byte-identical** (#120: ≈+0.15% perplexity, needle 100%, 15/15
batteries) — you pay in throughput, not quality: **decode ≈1.4× slower**, prefill an order of
magnitude slower single-stream, PCIe-bandwidth-bound.

**Stock sglang:** only `--cpu-offload-gb`, which is generic, layer-granular, not quant/MoE
-aware, and slow; or Expert Parallelism, which requires all experts to fit the aggregate VRAM
across the GPUs. Neither is a per-expert, quant-native, byte-identical spill.

## 5. Model too big for any single card — load-time offload to host RAM

<img src="topologies/05-122b-host-spill.svg" alt="122B-A10B on one 32 GB card with ~61 GB of experts pinned in host RAM" width="100%">

**Rig:** a single RTX 5090 (32 GB) serving a 122B-A10B Int4 whose experts alone are ~61 GB.

**Winning setting:** head-rank **load-time** MoE offload (§6, Variant C B2b, **[in
progress]**). Instead of materialising all experts on the GPU and then slicing — which OOMs
*during load* — it materialises only the resident+cushion experts per layer and streams the
cold tier straight into pinned host RAM at load time. Pairs with the weightless fast lane
(§10) where the fast card holds the full model.

**Physical constraint it resolves:** the model cannot be brought up on 32 GB at all with the
naive load path. The doc notes the mechanism is validated on 35B-A3B; the full **122B-A10B
Int4 run is [planned]** (download-gated — Int4 only on this rig, since an FP8 pinned pool
would exceed host RAM). Steady-state resident compute is byte-identical; throughput stays
PCIe-bound.

**Stock sglang:** cannot run this at all on one 32 GB card — it OOMs at load; EP needs the
experts to fit aggregate VRAM; generic offload is not load-time-aware.

## 6. Long-context priority — weightless-KV fast lane

<img src="topologies/06-weightless-kv-lane.svg" alt="Fast card holds the full model TP=1; slow cards hold only KV token-shards" width="100%">

**Rig:** 1× RTX 5090 (full model) + 2× RTX 3080 (KV-only), PCIe.

**Winning setting:** the Weightless-KV Fast Lane (§10, Variant C stages **B1 + B2a —
landed**, eager-only). The **fast card holds the full model as collective-free TP=1** and is
the sole Q/K/V producer + attention dispatcher; the slow cards become **weightless KV
workers** that hold only a DCP token-shard of the KV cache and run a stripped
attention-only forward — **no layer weights at all** (the thin blue sliver is ≈0).

**Physical constraint it resolves:** normally the slow cards limit capacity because they must
also hold weights. Freeing their weight VRAM (**≈14 GB per worker** per the doc) turns them
into pure KV headroom, lifting context **≈4×** on the 27B test model. Correctness is
byte-identical on extend (Δ=0 vs full-TP=1); decode differs only by benign kernel fp-order.

**Stock sglang:** every rank must hold layer weights; the slow cards spend their VRAM on
weight shards/replicas instead of contributing pure KV capacity, so context stays bottlenecked
by them.

## 7. More ranks than physical cards — multi-rank co-location

<img src="topologies/07-multi-rank-colocation.svg" alt="TP=5 on 3 cards: two ranks co-located on the 5090" width="100%">

**Rig:** 1× RTX 5090 + 2× RTX 3080, but a **TP=5** layout is wanted.

**Winning setting:** multi-rank co-location via `--rank-gpu-id` with duplicates, e.g.
`--rank-gpu-id 0,0,1,2` (§9). Two ranks are placed on the 5090 as two independent processes;
NCCL multi-rank config is auto-set when duplicates are detected, and a
physical-impossibility check enforces `(ranks on a GPU) × per-rank-MiB ≤ NVML total`. In the
diagram the 5090 shows two stacked (weights + KV) blocks — two real ranks sharing one card.

**Physical constraint it resolves:** TP degree is otherwise bounded by the physical GPU
count. Co-location let the fork prove **replicated-KV correctness at TP=5 on only 3 GPUs**
(§9, #62). Honest caveat from the doc: co-located ranks share one piece of silicon, so they
contend for SMs and add no memory bandwidth — this buys capability/flexibility, not raw
throughput. It pairs with uneven-TP so the shared card can also carry a proportionally larger
shard.

**Stock sglang:** TP is bounded by the physical GPU count; there is no supported way to place
two ranks on one card, so TP=5 on 3 cards is impossible.

## 8. A slow PCIe x4 link in the rig — PD-disaggregation placement

<img src="topologies/08-pd-disagg-slow-pcie.svg" alt="Prefill solo on the x16 fast card; decode distributed on the x4 cards" width="100%">

**Rig:** RTX 5090 on PCIe x16 + 2× RTX 3080 on PCIe x4 (thin dashed brown links).

**Winning setting:** single-node Prefill/Decode disaggregation (§2). **Prefill runs solo on
the fast x16 card as TP=1** with zero cross-GPU traffic; decode runs distributed TP=2
uneven+DCP on the slower cards. The KV handoff uses the existing transfer stack
(`mooncake_tcp` loopback); the teal region on the decode cards is the handed-off KV.

**Physical constraint it resolves:** a single TP group would force every prefill collective
across the slow x4 lane, and interconnect penalises collectives hardest on this rig. Running
prefill alone on the fast card removes that traffic entirely — the doc reports **≈2-5× faster
TTFT**, with decode essentially unchanged (**≈-2%** at long context). The handoff is
crash-robust (decode survives a prefill hard-kill and tears down to 0 MiB).

**Stock sglang:** a single fused TP group runs prefill collectives over the x4 link too, so
the slow lane throttles time-to-first-token; there is no single-node prefill/decode role split
to place prefill on the fast link.

## 9. Eight-GPU mixed fleet — several capabilities combined

<img src="topologies/09-eight-gpu-fleet.svg" alt="Eight mixed GPUs combining uneven-TP, uneven-DCP, expert offload, weightless workers and PD placement" width="100%">

**Rig:** 1× RTX 5090 (32 GB, x16) + 2× RTX 4090 (24 GB, x16) + 1× RTX 3090 (24 GB, x8) +
2× RTX 3080 (20 GB; one x8, one x4) + 1× RTX 3080 (20 GB, x4) + 1× RTX 2080 Ti (11 GB, x4),
mixed PCIe, no NVLink.

**Winning combination:** no single setting — the fork's value on a fleet like this is that
the settings **compose**. Uneven-TP (§1) sizes each card's weight shard to its VRAM; uneven-DCP
(§1) token-shards the KV; the two x4-attached 3080s run as **weightless-KV workers** (§10)
contributing pure KV headroom rather than weights; per-expert offload (§6) spills cold MoE
experts to pinned host RAM (bottom bar); and the fast x16 5090 is the natural home for
prefill and the hot expert tier (§2/§6).

**Physical constraint it resolves:** the fleet spans 11 GB to 32 GB per card and three PCIe
speeds. There is no single homogeneous shard size that uses this hardware well.

**Stock sglang:** even TP drops the whole group to the smallest card's shard (the 11 GB
2080 Ti), or excludes it entirely; there is no token-axis KV, no weightless workers, and no
quant-aware per-expert offload — so most of the fleet's aggregate VRAM is either wasted or
unusable for one model.

---

## Caveats

- Diagrams are schematic. VRAM box heights are to scale; the internal region splits are
  illustrative unless a figure is quoted from FEATURES_VS_UPSTREAM.md.
- Impact figures are the parent doc's rounded, directional numbers on the PCIe reference rig
  (no NVLink). A better interconnect changes the cost side of every cross-GPU feature.
- Items marked **[in progress]** (expert offload, load-time MoE offload) or **[planned]** (the
  full 122B-A10B run) are labelled as in the parent doc — do not read them as shipped.
- The SVGs are regenerated by [`topologies/gen.py`](topologies/gen.py); they are
  self-contained (inline shapes + text only, no external fonts or images).
