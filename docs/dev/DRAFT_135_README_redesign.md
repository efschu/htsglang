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
| 2b | Per-row measured numbers, quoted verbatim with units, from the detail sections (`FEATURES_VS_UPSTREAM.md` lines 342-1390) | yes, for the evidence lines |
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

**Not done, deliberately:** no README was modified, no repository visibility
changed, nothing posted, no announcement drafted for any forum. This file is
the whole deliverable.
