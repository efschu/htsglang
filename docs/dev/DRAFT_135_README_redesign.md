# DRAFT — README and feature-docs redesign (#135)

> **NOT PUBLISHED. NOT A README.** This file is a draft held on a branch. It
> has not been posted anywhere, no repository was made public, and nothing here
> has been announced. Publishing is a USER gate — step 10 of
> `docs/dev/RELEASE_CHECKLIST.md`. Until that gate is answered this document is
> internal working material and should not be copied outward.
>
> Ordering rule applied throughout: **heterogeneous-hardware enablers first,
> then descending by usefulness on an ordinary homogeneous rig.** That is the
> standing documentation order for this fork, and it is a deliberate inversion
> of the usual "biggest number first" README: the fork's reason to exist is the
> mismatched-GPU case, and a reader with two identical cards should still find
> their answer without reading past features they cannot use.

---

## 1. What is wrong with the README today

Stating this first because the redesign is mostly a response to it.

1. **It is a flag dump.** Roughly seventy `--flags` in flat bullet lists,
   ordered by the subsystem they happen to belong to. A reader cannot tell
   which three matter to them, and the list does not distinguish a flag that
   changes what the fork *is* from a flag that tunes a corner of it.

2. **It opens with "under construction".** Two words, directly under the
   title, above everything else.

3. **It carries a stale caveat that describes a fixed bug.** The chat-template
   section warns:

   > *"the image entrypoint uses `${VAR:=default}` shell defaults, so an
   > empty env var (`-e RANK_GPU_ID=`) is treated as unset and the baked-in
   > default is re-applied"*

   and tells the reader to bypass env vars entirely. **This is no longer
   true.** Commit `25d3a5ded2` emptied those defaults; the flag variables are
   now `: "${VAR:=}"` and clearing one removes the flag, as
   `docker/htsglang-entrypoint.sh:13` states and
   `test/registered/unit/docker/test_entrypoint_empty_env_384.py` pins.
   Only genuinely non-flag variables (`MODE`, `HICACHE_STORAGE_DIR`,
   `PLANNER_HOST`, `PLANNER_PORT`) still carry non-empty defaults, which is
   intended. **The workaround must go with the bug**, or every reader keeps
   paying for a defect that was fixed.

4. **The split between "in the image" and "run from source" is presented as a
   flag property.** It is a property of *one particular published image*, and
   it will be wrong the moment the next image is built. It belongs in the
   release notes for a tag, not in the feature list.

5. **A feature list is not a reason to care.** Every entry says what a flag
   *sets*. Almost none says what it *buys*, and the ones with measured numbers
   behind them do not show the numbers where a reader would look.

---

## 2. Proposed structure

```
README.md
├── 1  What this fork is for            (three sentences, no flags)
├── 2  Does this apply to me?           (the mismatched-hardware test)
├── 3  Install / run                    (docker pull, one working command)
├── 4  The heterogeneous-hardware core  ← the ordering rule starts here
│      uneven TP, rank→GPU mapping, per-rank memory budgets, uneven KV
├── 5  Useful on any rig                ← descending by ordinary usefulness
│      speculation, scheduling lanes, KV offload, cache behaviour
├── 6  Experimental                     (clearly fenced, honestly labelled)
├── 7  Full flag reference              (the current list, moved here)
├── 8  Differences from upstream sglang (link to FEATURES_VS_UPSTREAM.md)
└── 9  Contributing / hardware wanted   (the community call, §5 below)
```

The flag dump is not deleted — it moves to §7 and stops being the first thing
a reader meets.

---

## 3. Section 4 — the heterogeneous core (drafted)

**Opening line, replacing "under construction":**

> Most inference servers assume every GPU in the box is the same. This one does
> not. If your machine has a 20 GB card next to a 32 GB card, upstream sglang
> will size everything to the smaller one and strand the difference. htsglang
> splits the model in proportion to what each card actually has.

**The reader test, up front, so the wrong reader can leave quickly:**

> **Use this fork if:** your GPUs differ in VRAM or speed, or you want more
> tensor-parallel ranks than you have cards.
> **You probably do not need it if:** your cards are identical and upstream
> sglang already fits your model — the fork tracks upstream and adds no speed
> on a symmetric rig by itself.

That second sentence is deliberate. A README that cannot say who should *not*
use the project is advertising.

**Features in this section**, hetero-enablers, in dependency order rather than
impressiveness order:

| Feature | What it buys | Flag |
|---|---|---|
| Rank→GPU mapping | Put ranks on chosen physical GPUs; duplicates co-locate two ranks on one card, so TP can exceed the card count | `--rank-gpu-id` |
| Per-rank memory budget | An absolute MiB budget per rank instead of one global fraction, because "80 % of the total" means two different things on two different cards | `--rank-gpu-memory-mib` |
| Uneven tensor parallelism | Shard weights in proportion to each rank's budget instead of equally, so the big card carries more | `--rank-tp-ratio` |
| Uneven KV ownership | Let the KV split differ from the weight split, so the cache is not pinned to the smallest rank | `--rank-kv-ratio` |
| Per-family ratios | Rebalance dense-MLP and MoE weight families separately, freeing bytes for KV | `--rank-mlp-ratio`, `--rank-moe-ratio` |

**The five rows above are a README-sized selection, not the inventory.**
`FEATURES_VS_UPSTREAM.md` documents **seventeen** Block 1 features. The full
list, in the source document's own order, is §3.1 below.

### 3.1 The ordering is inherited, not invented

The single most useful thing found while drafting this: **`FEATURES_VS_UPSTREAM.md`
already encodes the required order**, so the redesign must adopt it rather than
compose a new one.

* Its **Block 1 — heterogeneous-GPU enablers** is exactly the hetero-first
  group, defined as *"without these, three genuinely different GPUs cannot be
  combined into one usable TP group at all, or cannot be combined correctly."*
* Its **Block 2 — general fork deltas** is stated as *"ordered by how much a rig
  of identical GPUs (1/2/4/8 cards) gains from it"* — which **is** descending
  normal-rig usefulness, already applied.

So the standing documentation order is satisfied by following the source
document's sequence. Any README ordering that departs from it is a claim that
the source doc is wrong and needs to be argued, not assumed.

**Block 1, in source order** (§ numbers are the doc's feature ids):

3 rank-to-GPU mapping and co-location · 1 asymmetric tensor parallelism ·
2 asymmetric decode context parallelism · 10 measured VRAM budget ·
21 barlink cross-vendor collectives · 23 Turing/gfx900 without sgl-kernel ·
22 fp8 dequant fallback (W8A16) · 11 cross-architecture speculative determinism ·
8e asymmetric-TP × GGUF correctness · 15 asymmetric-TP quantization correctness ·
17 HiCache under asymmetric-TP/DCP · 24 SWA-DCP · 18 TP greater than
num_kv_heads · 19 broad model bring-up under asymmetric-TP · 14 single-node PD
disaggregation · 12 weightless-KV lane · 27 cross-rig uneven pipeline
parallelism.

Seventeen is too many for a README section. The proposal is to keep the first
four as the named core (they are the ones a user chooses with flags), and
present the rest as a "correctness under asymmetry" group with a link — because
features 8e, 15, 17, 18, 19 and 24 are not things a user turns on, they are
guarantees that other things keep working once the split is uneven. That
distinction is invisible in the current flat list and is most of what
distinguishes this fork from a patch.

---

## 4. Section 5 — useful on any rig (drafted, ordering pending)

**Correction to an earlier version of this draft.** This section previously
guessed an order — speculation, then scheduling, then KV offload — and that
guess was **wrong**. The source document's Block 2 is already sorted by gain on
an identical-GPU rig, and it puts the **GGUF family first**, not speculation.
Recording the error because it is the exact failure the ordering rule exists to
prevent: ranking by what sounds impressive rather than by what an ordinary rig
gains. The authoritative order is:

| Rank | Feature | Doc id | Status |
|---|---|---|---|
| 1 | Bespoke GGUF adapter framework | 8a | Boot-checked |
| 2 | Qwen3.5/3.6 GGUF | 8b | Boot-checked |
| 3 | Gemma-4 GGUF | 8c | Boot-checked |
| 4 | GGUF K-quant compute kernels | 8d | Boot-checked |
| 5 | Multimodal and dynamic-quant GGUF | 8f | Boot-checked |
| 6 | Solo drafter placement | 4 | Built |
| 7 | Cross-algorithm drafter routing | 5 | **WIP** |
| 8 | CUDA graph memory aliasing for spec branches | 6 | Boot-checked |
| 9 | MoE expert offload + asymmetric TP/DCP | 7 | Boot-checked |
| 10 | GPU-to-GPU collectives over small PCIe BARs | 28 | **Exp**, unmerged branch |
| 11 | Hibernate checkpoint/restore | 9 | Boot-checked |
| 12 | Session KV spill | 20 | **Exp** |
| 13 | Rig dashboard / planner UI | 13 | **Exp** |
| 14 | Fast-lane priority scheduling | 16 | Built |
| 15 | Dynamic concurrent-session limit | 29 | Built |
| 16 | Per-message-class link selection | 25 | Cross-checked |
| 17 | Prefill satellite (cross-host PD for hybrid GDN) | 26 | Boot-checked |

That GGUF leads is sensible on reflection: it is what lets an ordinary owner
*run a model at all* on a card that could not otherwise hold it. Speculation
makes an already-working setup faster, which is a smaller gift.

**Use the document's own status tiers, do not invent labels.** `Built` = merged,
fork tests only. `Boot-checked` = executed on hardware with a real model.
`Cross-checked` = validated against a named independent reference. Modifiers:
`WIP` incomplete, `Exp` experimental, trailing `*` lives only on an unmerged
branch. A README that flattens these into "supported" would be overclaiming
four different things at once — three of the entries above are `Exp` and one is
on an unmerged branch.

Each entry gets the same three lines: what it buys, the flag, and either a
measured number or an explicit "not measured on a symmetric rig".

**Experimental features are fenced, not mixed in.** `--weightless-kv-fastlane`
and `--enable-kv-session-offload` are labelled experimental in the current
README and must stay labelled, in their own section, with the caveat attached
to the feature rather than to a footnote.

---

## 5. Section 9 — the community call (drafted, ready for review)

The one section with no upstream equivalent, and the reason #135 is worth
doing at all.

> ## Hardware wanted
>
> This fork exists because of one machine: two RTX 3080s and a 5090 in a box
> that upstream sglang would have sized to the smallest card. Every uneven-TP
> ratio, every per-rank memory budget, and every co-location path in here was
> measured on exactly that hardware.
>
> **That is also the limitation.** Three consumer cards, no NVLink, no working
> peer-to-peer — every pair reports PHB, and one card sits on a four-lane
> link. So the fork knows a great deal about one shape of machine and very
> little about yours.
>
> Useful things to send, roughly in order of how much they would help:
>
> 1. **A machine that is mismatched differently.** AMD next to NVIDIA, three
>    generations apart, a laptop GPU carrying a rank. If uneven TP picks a bad
>    split on your hardware, that is the most valuable bug report available.
> 2. **A machine with working NVLink or P2P.** Several settings here are
>    deliberately *not* shipped as defaults because they were measured on a rig
>    with neither, and shipping them to you would be a regression rather than a
>    fix. Nobody has been able to check the other side of that.
> 3. **A failure, with the launch command.** Especially a hang: distributed
>    hangs are where an unusual rank layout shows up first.
>
> **What is honestly not ready:** anything marked experimental is experimental.
> Multi-node is a direction, not a feature. The published image is one build
> from one branch — if a flag is missing from it, that is the image, not the
> fork.
>
> No CLA, no contributor agreement, no roadmap voting. Issues and pull requests
> are read.

**Tone note for review.** This section deliberately leads with the fork's
limits rather than its numbers. A call for hardware from someone claiming their
setup is already complete reads as a request for free validation; one that
names what it cannot test reads as a request for help, which is what it is.

---

## 6. What this draft still needs before it can be published

| # | Item | Blocking? |
|---|---|---|
| 1 | ~~Feature inventory pass~~ — **DONE.** All 34 features enumerated in source order (17 Block 1, 17 Block 2), with the document's own status tier per row | no |
| 2 | ~~Confirm the descending order against the inventory~~ — **DONE, and the draft's guess was wrong.** The source document already sorts Block 2 by gain on an identical-GPU rig; the redesign inherits that order rather than composing one | no |
| 2b | ~~Per-row measured numbers from the detail sections~~ — **DONE, §8.** All 34 rows carry an evidence label (measured / estimate / absent) with the figure quoted verbatim. **14 of 34 are `absent`** and stay that way; 4 carry figures that are not wins; 1 is an estimate that must never be quoted as measured | no |
| 3 | Rewrite `README.md` itself in this shape | yes |
| 4 | Delete the stale `${VAR:=default}` caveat (§1.3) and the workaround with it | yes — it misinforms today |
| 5 | Move the "not in the published image" list into per-tag release notes | no |
| 6 | Decide whether §2's "you probably do not need it" survives review | no — a judgement call, flagged not settled |
| 7 | **USER go to publish anything at all** | **yes — the gate** |

---

## 7. How numbers must be presented (inherited from the source document)

`FEATURES_VS_UPSTREAM.md` is disciplined about evidence in a way a README
usually is not, and the redesign must not launder that discipline away while
"making it approachable".

**Every figure is a lower bound, and the document says so once rather than per
row.** The reference rig has no NVLink and no CUDA P2P, all cross-GPU traffic
is host-staged, one 3080 sits on PCIe Gen4 x4 (~6.5 GB/s against ~13-14 GB/s
for the other two), clock pinning is refused by the driver, and one card spent
part of the measurements in thermal slowdown at 85-87 C. The document's own
words: *"an unfavourable configuration on every interconnect axis; the figures
throughout are a lower bound for the features, not a projection of them."*
**The README must repeat that sentence, not drop it.** Dropping it converts a
conservative measurement into an advertised benchmark.

**Three identities do not count as validation**, and the README must not use
them as if they did:

1. **Byte-identity above ~109 prompt tokens on an RTX 3080 under fp8** —
   `gptq_marlin_gemm` is measured run-to-run nondeterministic there (0 of 1200
   mismatches through M=109, first at M=128). The 5090 is unaffected.
2. **Token identity between speculative and non-speculative decoding** — the
   verify round computes k+1 tokens in one forward, so the reduction order
   differs by construction. A valid reference carries the same speculative
   configuration.
3. **Text identity between two boots of the same checkpoint at temperature 0** —
   #360: two identical boots diverge on 12 of 42 graded answers. Text identity
   is not even self-consistent within one checkpoint and one config.

**The consequence for any claim the README makes:** a quality comparison needs a
graded rubric plus a same-arm A-vs-A pair to fix the noise band, and only a
delta clearing that band is evidence. #360 is the worked example — 12/42 texts
differ while the grades are 42/42 and 41/42, inside the band.

**Practical rule for the rewrite:** every number carries its unit, its hardware,
and its status tier. A feature with no measurement says "not measured" in the
same place a number would go. No adjective substitutes for either.

---

---

## 8. Per-row evidence — all 34 features (closes blocking item 2b)

Extracted from the detail sections, `FEATURES_VS_UPSTREAM.md` lines 342-1390.

**Evidence label**, applied strictly:

* **measured** — the document states a figure produced by a run.
* **estimate** — the document states a figure and labels it as *not* a
  measurement. Never promoted to "measured".
* **absent** — the document gives no defensible figure for this row. Written as
  `absent`, never filled with an invented, inferred, or borrowed number.

**Lower-bound framing applies to every figure below, without exception.** The
reference rig has no NVLink and no CUDA P2P, all cross-GPU traffic is
host-staged, one 3080 sits on PCIe Gen4 x4 (~6.5 GB/s against ~13-14 GB/s), the
driver refuses clock pinning, and one card spent part of the measurements in
thermal slowdown at 85-87 °C. The document's own words: *"an unfavourable
configuration on every interconnect axis; the figures throughout are a lower
bound for the features, not a projection of them."*

**And the detection limit is part of every figure.** Raw tok/s carries a
**2.6-4.2 % boot-to-boot spread** on this rig against **0.09-0.85 %** for ms per
verify round, so any claim finer than ~3.5 % belongs on the round-time axis, and
values inside the spread are **not gains**. Two rows below are inside their band
and are marked as such — they are not wins.

### Block 1 — heterogeneous-GPU enablers

| # | Feature | Tier | Evidence | Figure (verbatim) |
|---|---|---|---|---|
| 3 | Rank-to-GPU mapping / co-location | Boot-checked | **absent** | No perf figure. Exercised: TP=4 co-located on 3 cards, Qwen3.6-27B `UD-Q6_K_XL`; requires NCCL >= 2.30 |
| 1 | Asymmetric tensor parallelism | Boot-checked | **absent** | No perf figure. Correctness: TP=3 on 5090 + 2×3080, 27B FP8, "greedy decode byte-identical run-to-run and cold-vs-warm"; DFLASH shards green at MLP units `[68,34,34]` |
| 2 | Asymmetric decode context parallelism | Cross-checked | **absent** | No perf figure. `short_code` "byte-identical arm for arm"; the 11,650-token prompt is **excluded** from the tier |
| 10 | Measured VRAM budget | Boot-checked | **absent** | No figure given |
| 21 | barlink cross-vendor collectives | Cross-checked | **measured** | Raw link `ucx_perftest tag_bw` **3413 MB/s (~27.3 Gbit/s)**. world-4 `all_reduce`, median of 100 after 20 warmup: 20 KiB **101.1 µs** flat / 106.6 ring; 80 KiB **385.9 µs** flat / **195.0 µs** ring; 128 KiB 646.3 / 270.4. Rendezvous + wireup **0.11 s**. Ring threshold corrected to 24 KiB ("25× too high"). **Cross-vendor with CUDA graphs: "not demonstrated"** |
| 23 | Turing/gfx900 without sgl-kernel | Cross-checked | **measured** | 2080 Ti with `sgl_kernel` absent: all 11 core modules import, **608 unit tests pass**. `forward_native` sm75 vs gfx900 byte-identical; vs kernel path a **~4.8e-07-class** reduction-order difference |
| 22 | fp8 dequant fallback (W8A16) | Cross-checked | **measured** | Fused dequant-GEMV: **max diff 0.0** against `torch.to(float32)`; **mean relative error 0.0014 against 0.0133** for the path it replaces. End to end, 27B FP8, TP=3, forced dequant: **+35 % decode (27.59 → 37.26 tok/s)** |
| 11 | Cross-architecture speculative determinism | Boot-checked | **absent** | No figure. Explicitly: activations are *not* bit-identical; agreement comes from the rank-0 sampling broadcast, not an independent per-arch comparison |
| 8e | Asymmetric-TP × GGUF correctness | Boot-checked | **absent** | No figure; bugfix class with red-then-green tests |
| 15 | Asymmetric-TP quantization correctness | Boot-checked | **measured** (quality) | Mixed-arch fp8 MoE with Marlin W8A16 fallback on sm86 ranks at **kernel-level cosine >= 0.99998** |
| 17 | HiCache under asymmetric-TP/DCP | Boot-checked | **measured** (functional) | **8/8 concurrent requests hit**, restore deterministic |
| 24 | SWA-DCP | Cross-checked | **estimate** — do not promote | Throughput: **"none taken. The ~+6-10 % figure in the design note is an ex-ante estimate, not a measurement"**. Correctness is measured: a needle ~3k tokens beyond the 1024-token window retrieves byte-identical to a TP=1 solo-5090 oracle |
| 18 | TP greater than num_kv_heads | Boot-checked | **absent** | No perf figure. Exercised: 35B-A3B FP8 at TP=3 with 2 kv-heads, MTP, CUDA graphs; TP=4 co-located on 3 cards |
| 19 | Broad model bring-up under asymmetric-TP | Boot-checked | **absent** | Per-model bring-up work; no figures |
| 14 | Single-node PD disaggregation | Boot-checked | **absent** (in-section) | Section says "Numbers above" and gives none locally. Correctness: "byte-identical to the same build with disaggregation off" |
| 12 | Weightless-KV lane | Cross-checked | **absent** | No perf figure in the section. Oracle: TP=1 solo run. Open: "no correctness oracle for lane plus speculation" |
| 27 | Cross-rig uneven pipeline parallelism | Boot-checked | **measured** (a cost, not a gain) | **55.1 tok/s / 18.16 ms per token against a 67.6 tok/s / 14.80 ms monolithic control**; 8k-prompt TTFT **3.42 s against 1.35 s**; **A-vs-A noise floor 1.1-2.1 %** → −18 % decode, 2.5× prefill TTFT. Only **~0.4 ms of the 3.4 ms/token** is the stage boundary (NCCL 1-way **142 µs** at bs=1, p90 173 µs); shape metadata **249 µs against 142 µs — 64 % of the crossing**. NCCL over sockets **2.07 GB/s** vs 1 GbE fallback 0.105 GB/s. Doc's own framing: *"PP does not beat a card the model already fits on — it buys the capacity to run one that does not"* |

### Block 2 — general fork deltas (in the document's own descending order)

| # | Feature | Tier | Evidence | Figure (verbatim) |
|---|---|---|---|---|
| 8a | Bespoke GGUF adapter framework | Boot-checked | **absent** | Framework row; boot evidence comes from 8b-8f |
| 8b | Qwen3.5/3.6 GGUF | Boot-checked | **absent** | No perf figure. `Q4_K_M`…`Q8_0` "coherent and greedy-deterministic"; `Q6_K` validated at asymmetric TP=3 |
| 8c | Gemma-4 GGUF | Boot-checked | **measured** | Gemma-4-31B-it `Q4_K_M`, TP=1 on the 5090 at **~61 tok/s**, coherent and self-deterministic. Only `Q4_K_M` verified; MoE/MTP/vision fail fast |
| 8d | GGUF K-quant compute kernels | Boot-checked | **absent** (in-section) | "Numbers above"; none locally. Crossover is opt-in and **not byte-identical when on** |
| 8f | Multimodal and dynamic-quant GGUF | Boot-checked | **absent** | `UD-Q6_K_XL` + `mmproj` validated; `UD-Q8_K_XL` rejected without mixed-dtype handling |
| 4 | Solo drafter placement | Built | **absent** | "no dedicated hardware boot" |
| 5 | Cross-algorithm drafter routing | **WIP** | **measured — and negative** | "the bandit **loses** its regime cell against the static winner, **75.52 against 89.22 tok/s**; per-switch cost **~2.5 ms**". Also 542.0 MiB released. **Must not be presented as a speedup** |
| 6 | CUDA graph memory aliasing for spec branches | Boot-checked | **measured** | **542.0 MiB released** under CUDA graphs on the lazy-capture arm |
| 7 | MoE expert offload + asymmetric TP/DCP | Boot-checked / Cross-checked | **measured** (correctness) | 35B-A3B AWQ: **32/32 tokens identical** to a TP=1 run. Perf: "Numbers above", none locally. Not bit-identical to no-offload (Marlin-Int4 tiling order) |
| 28 | GPU-to-GPU collectives over small PCIe BARs | **Exp**, unmerged branch | **measured** | Transport vs NCCL interleaved: **1.13× / 1.34× / 1.15× / 1.04× / 1.30×** at 20 KiB / 80 KiB / 1 MiB / 4 MiB / 16 MiB; independent standalone probe agrees 1.48 / 1.45 / 1.13 / 1.04 / 1.27. Serving: prefill **+14.3 % (1469.0 vs 1285.6 tok/s at 1 session)**, decode **+3-4 %**. **Gain does not persist**: prefill ratio across 1/4/8/16 sessions **1.143 / 1.031 / 0.997 / 1.009**. Byte proof **`bad_bytes = 0` over 1.1 M rounds**. **Loses to 0.81× on the fast x8 pair between 1 and 8 MiB.** Needs an out-of-tree driver patch |
| 9 | Hibernate checkpoint/restore | Boot-checked | **measured** | **12.64 %** of a real rank image is zero pages; allocation win **1.1447×** on a filesystem that folds nothing and **"exactly zero on a ZFS pool with compression"**; write-time delta **inside the A-vs-A floor either way**. "Not yet boot-checked on a real park/restore round trip" |
| 20 | Session KV spill | **Exp**, Boot-checked | **measured — inside the band** | Armed vs off: **37.74 vs 37.92 ms per verify round, "inside the 0.09-0.85 % boot-to-boot band"** → **not a gain**. Functional: born-spilled **2 of 2** admissions (1829-token prompt against a 1306-token device budget); output 200 tokens coherent; **restores 3 of 3 boots**; host pool 1.00 GB of a 24 GiB node budget |
| 13 | Rig dashboard / planner UI | **Exp** | **absent** | "functional but under active development, not production-ready" |
| 16 | Fast-lane priority scheduling | Built | **absent** | "no hardware boot" |
| 29 | Dynamic concurrent-session limit | Built | **measured** (a capacity constraint) | Measured 2026-07-30, TP=3 uneven rig, 20 GB card: **ceiling 64 came out 559 MiB over the per-rank budget, ceiling 32 407 MiB over it, both before the first KV token, while 16 booted** |
| 25 | Per-message-class link selection | Cross-checked | **measured — and self-refuting on this rig** | Cross-rig world-2 `all_reduce`: **29.69 µs / 184.54 µs at 8 / 256 KiB**. A/B control on the legacy route: 26.33 / 24.14 / 38.39 µs at 8 KiB and 187.50 / 182.97 / 181.18 µs at 256 KiB — the new route lands **inside the old route's own repeat spread**. Negative control fails as required (`Destination is unreachable`). Doc's verdict: **"Worth it here? no"** (1.47 µs vs 1.58 µs for 8 B) |
| 26 | Prefill satellite (cross-host PD for hybrid GDN) | Boot-checked | **measured** (a trade, not a win) | Satellite pair **2.892 s TTFT against 0.604 s monolithic-under-load**, but the running decodes' worst inter-token time drops **6.54 ms → 3.22 ms**. **93.5 %** of the satellite's TTFT is the 2080 Ti's own prefill compute (**2385 against the 5090's 10850 tok/s** under the same load), **1.8 %** transport (98 MiB in ~53 ms). Handover proven by `cached_tokens=6464`. Doc's own caveat: *"this is a statement about this 2080 Ti, not the method"* |

### What this table changes about the README

**Seventeen of 34 rows are `absent`** — exactly half the fork. The full tally is
**16 measured, 1 estimate, 17 absent**. A README that gave every feature a number
would have to invent seventeen of them. The correct presentation is a status tier
and a plain description for those rows, with the number slot explicitly empty.

(Counted mechanically over the table, not by eye: an earlier draft of this
paragraph said "fourteen", which was wrong. Recording that because a
miscounted summary line sitting above a correct table is precisely how an
invented number enters a document that was trying to prevent them.)

**Four rows carry figures that are not wins**, and each must be presented as
what it is, because presenting them as gains would be the clearest possible
overclaim:

* **5** cross-algorithm routing — the bandit **loses** (75.52 vs 89.22 tok/s).
* **20** session KV spill — **inside** the boot-to-boot band; the win is
  functional (the session keeps decoding), not throughput.
* **25** per-message-class link selection — inside the control's own spread, and
  the document says it is not worth it on this rig.
* **27** cross-rig PP and **26** prefill satellite — both are **costs** bought
  deliberately for capacity and for undisturbed decode respectively.

**One row is an estimate and must never be quoted as measured:** 24 SWA-DCP's
"~+6-10 %" is explicitly *"an ex-ante estimate, not a measurement"*.

---

**Not done, deliberately:** no README was modified, no repository visibility
changed, nothing posted, no announcement drafted for any forum. This file is
the whole deliverable.
