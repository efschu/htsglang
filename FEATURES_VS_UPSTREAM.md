# htsglang — Features and Bugfixes over Upstream sglang

`htsglang` is a fork of [sglang](https://github.com/sgl-project/sglang) built for
**heterogeneous single-node rigs with mismatched GPUs** (the reference machine:
1× RTX 5090 32 GB + 2× RTX 3080 20 GB, no NVLink, NCCL over PCIe host-staging).
The core idea: split **tensor-parallel weights, KV-cache, and MoE experts
*proportionally* across differently-sized/differently-fast cards** instead of the
uniform, equal-shard split that upstream assumes.

See also: [TOPOLOGIES.md](TOPOLOGIES.md) — for each major capability below, the canonical
multi-GPU constellation where it is the decisive advantage, with VRAM/host-RAM partition
diagrams and the stock-sglang contrast.

This document is a reference inventory of everything the fork adds or changes over
vanilla sglang, grouped by area. Each entry gives a short description and a rough
impact estimate. It tracks work on **two events** — when a feature *lands* (the numbered
sections below) and when one is *accepted/planned* (the clearly-separated **Roadmap** section
at the end) — so the "planned vs shipped" line is always explicit.

## About the impact figures

Impact numbers here are deliberately **rough** (rounded to ≈5-10% or to x-factor
granularity), measured on the heterogeneous reference rig above. They are directional,
not benchmarks: interconnect on this rig is PCIe (no NVLink), which penalizes any
cross-GPU collective, so cost/benefit will differ on better-connected hardware.
Correctness/enablement fixes carry **no throughput claim** — they make something work
or make it deterministic; they are labelled as such rather than given a fabricated
percentage.

**Direction tags.** Every ratio/percentage below is tagged right at the number so the sign's
meaning is unambiguous at a glance: **[better]** (a win), **[worse]** (a real cost/regression),
**[neutral]** (break-even / roughly unchanged). A single feature can carry both — e.g. more
context (`[better]`) at a small decode cost (`[worse]`).

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

Status convention: a shipped, integrated + validated feature carries **no marker** (that is the
default for the numbered sections). Anything else is flagged inline: **[in progress]**, **[untested]**
(code complete, not yet validated), **[guarded]** (implemented but gated off — see that section),
**[planned]** (roadmap, not built).

---

## 1. Uneven Tensor Parallelism & Decode Context Parallelism (DCP)

The foundational feature set: let each TP rank own a share of the model proportional to
its GPU, instead of an equal split.

- **`--rank-tp-ratio`** — proportional TP weight shards across unequal GPUs.
  Attention / GDN are split in whole KV-head units per rank.
  *Impact: enables TP across mismatched GPUs that upstream cannot do at all (the big
  card would otherwise be capped to the small card's shard). No throughput claim —
  it's the enablement primitive everything else builds on.*
- **`--rank-tp-ratio auto`** — VRAM-optimal automatic split: fills every card,
  maximizing total usable context. Derives weights from NVML budgets (gcd-reduced),
  sets DCP = TP automatically.
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
  honest cost of the extra per-step collectives over PCIe.*
- **`--rank-kv-ratio {coupled|capacity|auto|LIST}`** — KV-token ownership **decoupled**
  from the weight split; `capacity` mode installs the measured optimal vector in a single
  boot (no iterate-and-reboot).
  *Impact: **[better] ≈+25% context** on 27B-FP8 at **[neutral] break-even decode** (≈±1%).*
- **TP > num_kv_heads (replicated KV)** — KV heads are replicated + token-sharded +
  LSE-merged so TP can exceed the KV-head count. Coherent across FP8 / GGUF / AWQ, including
  GQA re-grouping for small head counts (down to gqa=1 ranks).
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

## 2. Prefill/Decode Disaggregation (Single-Node)

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

## 3. Speculative Decoding (MTP / NEXTN / EAGLE3)

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
- **Adaptive draft length** — EMA + hysteresis + debounce picks k∈{1,2,3} at runtime, with
  pre-captured graph states per k, rank-deterministic.
  *Impact: roughly matches the best fixed k on any given workload without hand-tuning; avoids the
  throughput cliff of a badly-chosen fixed k.*
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
- **Tuned K-quant MMVQ kernel** — TP=2 beats llama.cpp on decode.
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

- **Per-expert MoE offload (pinned-host pool + wave prefetch) [landed, unpushed on `feat/weightless-kv-fastlane`]** —
  env-driven (`SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0`): all experts live in a pinned-RAM pool, the
  GPU keeps only `ceil(fraction · num_experts)` resident experts, and each forward LRU-prefetches the
  misses + remaps slots on-device over the unchanged grouped-GEMM. Wave processing is done **over
  tokens, not experts** (no cross-wave partial sums → no floating-point re-association). It composes
  with **uneven-TP + uneven-DCP** and is now **CUDA-graph compatible** — a decode-graph + eager-prefill
  hybrid. On this rig it runs the **122B-A10B GPTQ-Int4** (~61 GB of experts) across the 3 mismatched
  cards (5090 32 GB + 2×3080 20 GB, no NVLink) — a model that does not otherwise fit.
  *Impact: **[better] enables models that otherwise would not fit**; **[better] single-stream decode more
  than doubled** vs eager — CUDA-graph capture adds ≈+50% and hot-expert residency adds more on top,
  roughly ≈2.3× over eager overall (TP=3, fraction=0.25). The win is launch-overhead elimination: the
  offload decode was launch-bound. The cost is throughput only, not quality (see below). Env levers:
  `SGLANG_MOE_HOT_RESIDENCY`, `SGLANG_MOE_OFFLOAD_CUDA_GRAPH`, `SGLANG_MOE_HOTSET_FILE`; default path
  unchanged.*
  - *Quality — validated, no measurable loss (#120): on Qwen3.6-35B-A3B-FP8 (TP=3 uneven, eager,
    temp=0, tiered fraction=0.5 → half the experts resident, half host-spilled) vs the fully-resident
    baseline: **[better] ≈+0.15% perplexity** (well inside FP8 reduction-order noise), **[better] needle-in-haystack
    100%** at 8k and 30k, **[better] correctness batteries 15/15 identical**. Also byte-identical at
    fraction 0.25 vs 1.0 (identical output IDs). Verdict: offloading experts does **not** degrade
    quality — you pay in throughput, not accuracy.*
  - *Correctness of the Int4 path (stated precisely, no overclaim): the FP8-triton offload is
    byte-identical. The GPTQ/AWQ-marlin-Int4 path is coherent, self-deterministic and argmax-identical
    within marlin's intrinsic ≈1e-2 reduction floor (near-ties only) — **not** bit-exact. The
    CUDA-graph path adds nothing on top of the platform's normal capture-vs-eager floor that every
    graph path in the fork already carries. In short: the offload adds zero on top of the existing
    graph/marlin floor — this is not a claim that graph == eager bit-exact.*
  - *Throughput cost — [worse] still PCIe-bandwidth-bound, not compute-bound: **[worse] prefill ≈an order
    of magnitude slower** single-stream (eager; the spill cache fragments prefill into many tiny
    fetch-gated GEMMs). Decode used to be the other soft spot but is now much closer to resident after
    the CUDA-graph + hot-residency work above. Levers that shrink the rest: bigger spill cache, hot-set
    tuning, NVLink/P2P, more PCIe lanes.*
  - *Robustness (non-obvious): a single forward can legitimately need more unique experts than
    fit in the resident slots (a prefill batch easily touches all of them) — not obvious up
    front. Waving over tokens keeps each token whole in one wave, so this overflow case stays
    byte-identical instead of silently evicting a still-needed expert.*
  - *Honest follow-ups (not built, listed for completeness): a **load-time streaming loader** —
    materialize only resident+cushion experts per layer during load and stream the cold tier straight
    into pinned host RAM, so the host never holds the full ~61 GB expert set at once — is **unbuilt**;
    it was not needed once the box had 108 GB of host RAM, and it remains the correct fix only for
    models larger than this (or tighter multi-process TP aggregates). **Per-rank offload sizing** (a
    different fraction per card) is **low value** here — the big card fills at the same fraction-pace,
    so a single global fraction is already near-optimal on this rig.*

## 7. Model Support

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

## 8. Determinism, Mamba/GDN & HiCache

- **Robustness — reproducible spec-decode output across mismatched GPUs** — NEXTN+GDN speculative
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
- **Robustness — stable sampling on mixed-arch TP** — with ranks on different architectures
  (sm120/sm86) the per-rank reduction order differs slightly, so the common shortcut of sampling
  independently on every rank (safe on identical GPUs) can pick different tokens (≈1/1000) and
  silently diverge into word-salad/loops. Non-obvious because it looks correct on any homogeneous
  rig. The feature is robust to it by taking token IDs from a single rank, and by keying the
  compile cache on GPU identity so ranks never load a foreign-arch artifact.
  *Cost < 1 ms/step. No throughput claim.*
- **Mamba/GDN determinism** — checkpoint grid, deterministic resume/eviction, flush == fresh-boot,
  fp32 beta-gate.
  *Impact: correctness — resumed and freshly-booted state produce identical output.*
- **HiCache under uneven-TP / DCP** — index translation and layout normalization make the
  KV-offload cache safe under the fork's non-uniform layouts (upstream assumes uniform shards).
  *Impact: correctness/robustness. The non-uniform layouts also exercise the offload paths
  differently enough to surface two non-obvious concurrency hazards — an L3 write-back race and a
  prefetch deadlock — that the feature is hardened against. No throughput claim.*

## 9. Multi-rank-per-GPU & Tooling

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
  above.

## 10. Serving & Scheduling

- **Fast-lane priority scheduling (Stage 0, `--enable-fast-lane`)** — an opt-in fast lane built on
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

- **Weightless-KV Fast Lane (Variant C, stages B1 + B2a; + chunked prefill #131)** — a single-process
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
  **Honest downsides:** **[worse] prefill runs eager** — CUDA-graph capture on the weightless+GGUF path
  is blocked by a pre-existing limitation (a GGUF lazy-init param not materialized before capture); this
  is a tracked follow-up, not a chunked-prefill regression; **[worse] the fast card now holds the full
  weights**, so once the workers are freed **it** becomes the context limiter for big models (addressed
  by the load-time MoE offload in §6); **[neutral] TP-only, single-node** — PP/DP/EP/spec are rejected by
  design (hard fail-fast).

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
