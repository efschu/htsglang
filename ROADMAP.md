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
