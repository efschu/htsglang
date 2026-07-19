# htsglang — Features and Bugfixes over Upstream sglang

`htsglang` is a fork of [sglang](https://github.com/sgl-project/sglang) built for
**heterogeneous single-node rigs with mismatched GPUs** (the reference machine:
1× RTX 5090 32 GB + 2× RTX 3080 20 GB, no NVLink, NCCL over PCIe host-staging).
The core idea: split **tensor-parallel weights, KV-cache, and MoE experts
*proportionally* across differently-sized/differently-fast cards** instead of the
uniform, equal-shard split that upstream assumes.

This document is a reference inventory of everything the fork adds or changes over
vanilla sglang, grouped by area. Each entry gives a short description and a rough
impact estimate.

## About the impact figures

Impact numbers here are deliberately **rough** (rounded to ≈5-10% or to x-factor
granularity), measured on the heterogeneous reference rig above. They are directional,
not benchmarks: interconnect on this rig is PCIe (no NVLink), which penalizes any
cross-GPU collective, so cost/benefit will differ on better-connected hardware.
Correctness/enablement fixes carry **no throughput claim** — they make something work
or make it deterministic; they are labelled as such rather than given a fabricated
percentage.

**Direction tags.** Every ratio/percentage below is tagged right at the number so the sign's
meaning is unambiguous at a glance: **🟢 better** (a win), **🔴 worse** (a real cost/regression),
**⚪ neutral** (break-even / roughly unchanged). A single feature can carry both — e.g. more
context (🟢) at a small decode cost (🔴).

## On "bugfixes" and robustness

A **bugfix over upstream** here means a defect present in vanilla sglang (or in a
kernel/library it uses) that this fork fixes — it would bite an upstream user too. Those are
listed on their own merit.

This doc does **not** enumerate the fork's own development bugs — a feature that was wrong in
an early revision and then corrected is just the feature working; the fix is not a selling
point. Robustness is only called out where the problem it solved was **not obvious on first
(or second) glance** — a subtle failure mode a careful reader wouldn't expect the feature to
have already handled. Those non-obvious robustness properties are noted at the relevant
feature; routine "we implemented it correctly" fixes are not.

Status legend: ✅ integrated + validated · 🟠 in progress · 🟡 code complete, untested
· 🔴 guarded/descoped · 📋 planned

---

## 1. Uneven Tensor Parallelism & Decode Context Parallelism (DCP)

The foundational feature set: let each TP rank own a share of the model proportional to
its GPU, instead of an equal split.

- **`--rank-tp-ratio` ✅** — proportional TP weight shards across unequal GPUs.
  Attention / GDN are split in whole KV-head units per rank.
  *Impact: enables TP across mismatched GPUs that upstream cannot do at all (the big
  card would otherwise be capped to the small card's shard). No throughput claim —
  it's the enablement primitive everything else builds on.*
- **`--rank-tp-ratio auto` ✅** — VRAM-optimal automatic split: fills every card,
  maximizing total usable context. Derives weights from NVML budgets (gcd-reduced),
  sets DCP = TP automatically.
  *Impact: **🟢 ≈+2.5-3x KV-cache context** on the reference rig vs a naive equal split (the
  equal split is bounded by the smallest card).*
- **`--rank-tp-ratio auto-performance` ✅** — measurement-based split that optimizes
  for throughput rather than maximum context.
  *Impact: trades some context for decode throughput; workload-dependent.*
- **`--rank-mlp-ratio` / `--rank-moe-ratio` / `--rank-vocab-ratio` ✅** — per-family
  shard vectors for MLP, MoE expert-intermediate, and vocabulary, so each dimension can
  be split independently of the attention split.
  *Impact: correctness/flexibility feature — lets indivisible dimensions land cleanly
  per rank. No throughput claim.*
- **Named-family shard plans + maximin solver ✅** — MLP and MoE units are jointly
  distributed across ranks by a solver rather than ad-hoc rounding.
  *Impact: correctness — guarantees valid, balanced partitions for every sharded
  dimension.*
- **Uneven-DCP token sharding ✅** — KV-cache follows the **token axis** instead of the
  KV-head axis: proportional KV-token split with a weighted owner rule + LSE merge.
  Removes the `sum(ratios) ∈ {2,4}` KV-head constraint entirely.
  *Impact: **🟢 ≈+60-80% context** over the first (head-axis) DCP version by decoupling KV
  capacity from the weight split; **🔴 ≈-10-25% decode throughput** vs no-DCP baseline, the
  honest cost of the extra per-step collectives over PCIe.*
- **`--rank-kv-ratio {coupled|capacity|auto|LIST}` ✅** — KV-token ownership **decoupled**
  from the weight split; `capacity` mode installs the measured optimal vector in a single
  boot (no iterate-and-reboot).
  *Impact: **🟢 ≈+25% context** on 27B-FP8 at **⚪ break-even decode** (≈±1%).*
- **TP > num_kv_heads (replicated KV) ✅** — KV heads are replicated + token-sharded +
  LSE-merged so TP can exceed the KV-head count. Coherent across FP8 / GGUF / AWQ, including
  GQA re-grouping for small head counts (down to gqa=1 ranks).
  *Impact: enables TP degrees that were structurally impossible before (applies under uneven-DCP
  too). No throughput claim.*
  **The price is small:** because TP exceeds `num_kv_heads`, the few KV heads must be **replicated**
  across the ranks that share a KV-head group instead of sharded. But under GQA there are only a
  handful of KV heads to begin with (the query heads, which dominate, are still sharded normally),
  and under uneven-DCP the KV *tokens* are still split across ranks — so what's duplicated is just a
  small KV slice, not the bulk of the cache. In practice this is a **🔴 minor, roughly single-digit-%
  KV overhead** (a real but small cost), easily paid for by the extra context / higher TP degree it
  unlocks — not a heavy tax.
- **Rank-uniform collective guards ✅** — guards in front of `all_gather` that require
  rank-uniform shapes, converting a would-be NCCL hang into a clean, early error.
  *Impact: robustness — turns a silent distributed hang into a fail-fast error.*
- **Auto-sizing Mamba/GDN state pool, rank-aware GDN shapes, per-rank head counts
  (flashinfer + triton) ✅** — including per-rank workspace sizing.
  *Impact: correctness — hybrid (GDN) models run correctly under uneven TP. No throughput
  claim.*
- **`max_total_num_tokens` caps ✅** — physically-reachable ceilings for GDN hybrids and
  SWA hybrids, computed deterministically so the pool never overshoots into OOM.
  *Impact: correctness/robustness — prevents deterministic-looking OOM from over-sized
  pools.*
- **CPU unit tests ✅** — coverage for family plans, the solver, calibration, and
  partition invariants (fast, no GPU needed).

## 2. Prefill/Decode Disaggregation (Single-Node)

Run prefill and decode as separate roles on the same box, so the fastest card does prefill
with zero cross-GPU communication while the slower cards handle distributed decode.

- **Solo-prefill on the fastest card ✅** — prefill runs TP=1 (zero inter-GPU comm), decode
  runs distributed TP=3 uneven+DCP.
  *Impact: **🟢 ≈2-5x faster TTFT** across context lengths (largest win in the mid-context range) —
  because prefill now runs alone on the fast card with zero cross-GPU traffic. Decode is a
  separate story: it stays distributed TP=3+DCP (not on the solo card), so it is essentially
  unchanged — a **🔴 negligible ≈-2% at long context** from the decode-side collectives, not a
  regression worth worrying about. Net: a clear, honest win for time-to-first-token at no real
  decode cost.*
- **Token-vector KV re-scatter ✅** — global→owned-compact translation + ordinal filter; one
  rule-agnostic path serving both even and weighted DCP, with no head re-cutting (KV heads are
  replicated).
  *Impact: correctness/architecture — a single code path for the KV handoff instead of
  per-mode special cases.*
- **GDN / hybrid state transfer ✅** — Mamba-state handoff from prefill to decode, validated
  bit-identical.
  *Impact: correctness — hybrid models survive the prefill→decode handoff losslessly. No
  throughput claim.*
- **`local_proxy` ✅** — a thin two-server front (health, `/metrics_summary`, dashboard-ready).
- **`mooncake_tcp` loopback ✅** — reuses the existing transfer stack (no new transfer code);
  prefill and decode ranks co-exist stably on one card via a budget protocol
  (`expandable_segments` + fake-warm).
- **Crash robustness ✅** — hard-kill of the prefill mid-transfer: decode survives, re-registers
  cleanly, tears down to 0 MiB.
  *Impact: robustness — no orphaned VRAM or wedged decode after a prefill crash.*

## 3. Speculative Decoding (MTP / NEXTN / EAGLE3)

- **MTP/NEXTN under uneven-DCP ✅** — including heterogeneous-GPU reproducibility (verify-sync,
  draft broadcast from rank 0, workspace zeroing).
  *Impact: correctness — the **emitted greedy token sequence** is reproducible on mixed-arch GPUs
  (run-to-run, cold==warm, independent of which card holds which rank). This does **not** mean the
  MTP heads' internal layers are bit-identical across archs — they aren't, and don't need to be:
  spec decode is output-preserving (emitted tokens = the target model's argmax chain, so a
  numerically "noisy" draft only changes how many tokens are accepted per step, not which tokens
  come out), and the accept/argmax decision is taken once on rank 0 and broadcast so ranks can't
  commit different tokens near a tie. See §8 for the three roots. No throughput claim beyond
  enabling spec at all under DCP.*
- **Adaptive draft length ✅** — EMA + hysteresis + debounce picks k∈{1,2,3} at runtime, with
  pre-captured graph states per k, rank-deterministic.
  *Impact: roughly matches the best fixed k on any given workload without hand-tuning; avoids the
  throughput cliff of a badly-chosen fixed k.*
- **Graph-state offload to system RAM (offload mechanism, stages 1+2) ✅** — the core enabler for
  adaptive / high-k speculation. Each draft length k needs its own pre-captured CUDA graph, and each
  such graph pins a block of VRAM. "Stages 1+2" refers to the two implementation stages of the
  *offload mechanism* — not the k values; the k range itself goes **up to 5**. That high end is where
  the feature earns its keep: with k∈{1..5} only the **currently-active** k's graph stays resident on
  the GPU, and the **other up to four k-graphs sit offloaded in system RAM**, streaming back onto the
  card on demand when the runtime switches k. Without this, keeping five k-graphs resident at once
  would multiply the graph VRAM and eat straight into the KV-cache / context budget. Mechanism:
  pause/resume, private MemPools, tagged capture-pools + int-workspaces. Modes: resident / offload /
  offload-scratch.
  *Impact: this is what makes adaptive **and** high-k (up to 5) spec **🟢 free in VRAM** — **🟢 KV loss:
  zero**, full context preserved at standard reserve even with all five k-rungs available, because
  four of them live in host RAM rather than on the card. Without it, multi-k (let alone k=5) spec
  would either not boot at standard reserve or would have to give back context. Enablement, no
  throughput claim.*
- **High-accept profile [1..5] ✅** — k=4/5 rungs for repetitive workloads; opt-in, boots at
  standard reserve thanks to the offload above.
  *Impact: **🟢 ≈+8% (k=4) to ≈+16% (pinned k=5) decode** on repetitive/structured loads vs k=3;
  workload-conditional — neutral-to-negative on diverse prose.*
- **EAGLE3 for Gemma-4 ✅** — speculators / vLLM-format support (aux-id off-by-one translation,
  `norm_before_residual` flag, converter).
  *Impact: **🟢 ≈+15-45% decode** on code/JSON with an instruction-tuned head vs the non-spec baseline;
  recommendation documented as workload-conditional (weaker on free-form prose).*
- **Draft embed / lm_head sharing, draft-extend replay, ratio-weighted draft vocab ✅** —
  *Impact: the draft path works correctly under uneven TP. No throughput claim.*

## 4. GGUF

There are two kinds of GGUF work here: (1) **new architecture support** — a custom adapter for the
Qwen3.5/3.6 hybrid-GDN family, which upstream cannot load at all; and (2) **architecture-general
GGUF improvements** (uneven-TP sharding, the tuned K-quant kernel, the perf overhaul, the vec
alignment below) that benefit **any** GGUF model, not only Qwen. Note that the *new-model* GGUF
adapter is Qwen3.5/3.6-specific — other families in this fork (e.g. **Gemma-4**) are supported via
**AutoRound-int4 / Marlin**, not GGUF (see §5 and §7), so they don't appear here.

- **Qwen3.5/3.6 hybrid-GDN GGUF ✅** — including NEXTN/MTP head, unsloth UD-quants, mmproj vision.
  *Impact: enables GGUF for a model family/architecture upstream sglang could not load (hybrid-GDN).
  No throughput claim — enablement.*
- **GGUF-MoE expert-dim sharding, uneven ✅** — whole experts per rank + zero-pad + remap,
  with GDN block coarsening and an MMQ fallback for misaligned blocks; A3B TP=3 coherent.
  *Impact: enables uneven-TP GGUF-MoE. Correctness/enablement.*
- **Tuned K-quant MMVQ kernel ✅** — TP=2 beats llama.cpp on decode.
  *Impact: **🟢 faster than llama.cpp** at TP=2 on this rig; a decode-throughput win for the GGUF path.*
- **GGUF perf overhaul ✅** — flat layout, batched MMVQ, Q8 lm_head, shape-aware dispatch,
  graph capture within budget. K-quant MMQ is now capped to small token counts, above which it
  dequantizes to cuBLAS.
  *Impact: **🟢 ≈5-8x prefill throughput** on batched/long prompts vs the always-MMQ path (which was
  flat/slow); **⚪ decode roughly unchanged** (bandwidth-limited).*
- **GGUF uneven-TP vec alignment 🟠** — 16-element MLP units for indivisible intermediate sizes
  (e.g. 17408 at TP=3/5); unlocks dense GGUF under uneven TP (the last remaining old GGUF blocker).
  *Impact: enablement for dense GGUF under uneven TP. Correctness, no throughput claim.*

> **Honest downside for the whole GGUF path:** GGUF K-quant decode is bandwidth-limited and
> generally **slower than native FP8/AWQ** on the same hardware; the value is model/quant
> availability and llama.cpp compatibility, not peak speed.

## 5. Quantization

- **AWQ / Marlin uneven-TP ✅** — group/tile alignment, K-mask in the MoE Triton kernel, and a
  **Marlin zero-point staging fix for g=32** (a CUDA-kernel bug that affected every AWQ-g32 MoE).
  *Impact: correctness — AWQ-g32 MoE was silently wrong before; also unlocks AWQ under uneven TP.
  No throughput claim.*
- **compressed-tensors uneven group alignment ✅** — quant-block-aligned units applied only to the
  layers that are actually quantized.
  *Impact: correctness/enablement under uneven TP.*
- **AutoRound-int4 ✅** — `weight_block_size` convention + a Marlin repack fix (Gemma vision
  geometry).
  *Impact: enables AutoRound-int4 (Gemma-4) that failed to load before. Enablement.*

## 6. MoE / Expert Offload

- **MoE expert offload (pinned-host pool + wave prefetch) 🟠** — env-driven
  (`SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0`): all experts live in a pinned-RAM pool, the GPU
  keeps only `ceil(fraction · num_experts)` resident experts, and each forward LRU-prefetches the
  misses + remaps slots on-device over the unchanged grouped-GEMM. Wave processing is done **over
  tokens, not experts**, which is the key to byte-identity (no cross-wave partial sums → no
  floating-point re-association).
  *Impact: lets a MoE model run with a fraction of its experts resident in VRAM — enables models
  that otherwise would not fit. **🔴 Honest cost: ≈-35% decode tok/s** on short prompts (PCIe H2D
  traffic). Validated byte-identical (fraction 0.25 vs 1.0 → identical outputs). **Eager-only by
  design** (data-dependent routing is not graph-capturable → fail-fast guard if graphs are on).*
  - *Robustness (non-obvious): a single forward can legitimately need more unique experts than
    fit in the resident slots (a prefill batch easily touches all of them) — not obvious up
    front. Waving over tokens keeps each token whole in one wave, so this overflow case stays
    byte-identical instead of silently evicting a still-needed expert.*

## 7. Model Support

- **Qwen3.6-27B hybrid-GDN ✅** — GGUF + AWQ-INT4 + FP8, uneven TP=3, MTP.
- **Qwen3.6-35B-A3B (MoE) ✅** — FP8/GGUF/AWQ coherent under uneven TP=3; the vehicle for
  PD-disaggregation testing.
- **Gemma-4 31B dense ✅** — int4-AutoRound, TP=1 and uneven TP=3 (plan-aware vision tower, Triton
  head fix); EAGLE3 speculation.
- **Gemma-4 26B-A4B (MoE, SWA hybrid) ✅** — boots (vision-ignore mapper, gated-GeLU Marlin);
  **`--swa-pool-sizing` cap: 🟢 ≈6x long-context** (50k needle proven), using Stage-A rather than
  SWA-DCP.
  *Impact: **🟢 ≈6x more usable long-context** for this SWA-hybrid model vs the un-capped default.*
- **Small models under uneven-TP ✅** — the replicated-KV GQA handling protects mini geometries
  (2B models + future draft models).
  *Impact: enablement/correctness for small-head-count models under uneven TP.*

## 8. Determinism, Mamba/GDN & HiCache

- **Robustness — reproducible spec-decode output across mismatched GPUs ✅** — NEXTN+GDN speculative
  decoding emits the **same greedy token sequence** even when the ranks are physically different
  cards. (To be precise: it's the *emitted output* that is reproducible, not the intermediate
  activations — different silicon produces different low-order bits, and spec decode tolerates that
  because it is output-preserving and the accept decision is shared from rank 0; see §3.) Worth
  highlighting because the sources of divergence here are genuinely **not obvious**: they only
  appear once ranks are different silicon (upstream never runs that way), and they hide in places
  you don't normally look — a greedy verify that silently relies on every rank's argmax matching,
  CUDA-graph pad rows leaking stale tokens into the MoE grouped-GEMM, and a flashinfer float
  workspace that reads back regions the current forward never wrote (persistent across forwards, so
  the output became a function of request *order*). The feature is robust against all three.
  *Cost sub-millisecond per step. No throughput claim.*
- **Robustness — stable sampling on mixed-arch TP ✅** — with ranks on different architectures
  (sm120/sm86) the per-rank reduction order differs slightly, so the common shortcut of sampling
  independently on every rank (safe on identical GPUs) can pick different tokens (≈1/1000) and
  silently diverge into word-salad/loops. Non-obvious because it looks correct on any homogeneous
  rig. The feature is robust to it by taking token IDs from a single rank, and by keying the
  compile cache on GPU identity so ranks never load a foreign-arch artifact.
  *Cost < 1 ms/step. No throughput claim.*
- **Mamba/GDN determinism ✅** — checkpoint grid, deterministic resume/eviction, flush == fresh-boot,
  fp32 beta-gate.
  *Impact: correctness — resumed and freshly-booted state produce identical output.*
- **HiCache under uneven-TP / DCP ✅** — index translation and layout normalization make the
  KV-offload cache safe under the fork's non-uniform layouts (upstream assumes uniform shards).
  *Impact: correctness/robustness. The non-uniform layouts also exercise the offload paths
  differently enough to surface two non-obvious concurrency hazards — an L3 write-back race and a
  prefetch deadlock — that the feature is hardened against. No throughput claim.*

## 9. Multi-rank-per-GPU & Tooling

- **`--rank-gpu-id` (with duplicates) + NCCL auto-config + physical-impossibility check ✅** —
  explicit rank→physical-GPU placement, NCCL auto-configuration when co-location is detected, and a
  fail-fast check that `(ranks on a GPU) × per-rank-MiB ≤ NVML total`.
  *Impact: enablement — multiple ranks per physical GPU (e.g. TP=4 on 3 cards), with early, clear
  errors for impossible mappings.*
- **TP=5+ emulation via co-location 🟠** — 3+1+1 topology on 3 cards, MPS, Q4 GGUFs; validates the
  >3-card code path on a 3-card rig.
  *Impact: test-coverage enablement for higher TP degrees than the rig physically has.*
- **Rig dashboard, Docker image, falsifier levers ✅** — operational tooling: a live rig dashboard,
  a reproducible container image, and env-gated falsifier probes used to hunt the determinism bugs
  above.

---

## Guarded / descoped (kept, not shipped)

These were implemented and evaluated but deliberately gated off — either because they violate a
correctness invariant or because they are net-negative on this rig's interconnect. The code is kept
dormant behind a guard for a possible future fix, not removed.

- **Tree-spec `--speculative-eagle-topk > 1` under uneven-weighted-DCP 🔴 GUARDED (#76)** —
  GPU-validated and found **silently wrong**: topk=1 (chain) is bit-deterministic and greedy-correct,
  but topk=4 is non-deterministic within a single boot and diverges from the topk=1 oracle, violating
  the lossless-greedy invariant. Root cause: the tree-masked verify-attention under weighted-DCP
  produces tree-topology-dependent verify logits (a draft node does not see exactly
  committed-prefix + true tree ancestors). **Also 🔴 perf-negative on this rig** (≈-15% decode: tree
  compute overhead > accept-length gain over PCIe x4, serial). Restored as a hard fail-fast guard at
  arg validation, with a CPU unit test. *Reactivation needs both an audit of the draft→draft ancestor
  semantics under DCP and hardware with a better interconnect that makes trees net-positive.*
- **SWA-DCP Stage B 🔴 descoped** — evaluated at only 🟢 ≈+6-10% (a modest win) for a large
  implementation cost — not worth it;
  reactivation criteria documented. Gemma-4 SWA long-context is served by the Stage-A pool cap in §7
  instead.

## Roadmap (planned / not yet built)

Directional only — not part of the current feature surface.

- **PD v2 "lane scheduler" 📋** — mixed-mode decode server + routing matrix: length routing,
  overflow prefill, sticky fast-lane decode + lazy migration, cache-affinity scheduling; end state
  is HiCache as a session medium (sessions live in the RAM tier, engines check them in/out,
  cross-lane prefix caching).
- **Draft-KV-pool DCP layout (`--draft-kv-layout`) 📋** — unlocks speculation in disaggregated
  operation + granular draft-VRAM control.
- **Suspend-to-disk 📋** — persist VRAM state (weights + KV + GDN) to disk to free/reclaim GPUs in
  seconds without re-capture.
- **GDN-state RAM offload 📋** — under investigation; converges with the HiCache session medium.
- **Web frontend 📋** — HF repo in → quant selection with a VRAM preview, config proposal
  (TP/DCP/PD/spec per model family), live max-KV estimator; hardware-generic (NVML profile, no rig
  constants).
- **122B-A10B-Int4 real run 📋** — the MoE expert-offload mechanism is validated on 35B-A3B; the
  full 122B run is gated on a model download decision (Int4 only on this rig — an FP8 pinned pool
  would exceed host RAM).
- **Upstream adaptive-spec PR review/port, tree-spec topk>1 on better interconnect, uneven-TP over
  nodes/RDMA 📋** — longer-horizon items.
