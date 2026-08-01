# ANALYSE #389 — Kimi K3 and DeepSeek V4 Flash on our stack

Feasibility verdict and test plan. **No run** — the real tests are scheduled
late, during dashboard finalization and before the Docker build, per the user.
Discharges the feature-analysis-file duty for #389.

Reference: `sqliteai/waste` (Apache-2.0, C, streams MoE experts from NVMe;
K3 2.78T on a 64 GB laptop at ~0.5 tok/s).

---

## 1. Verdict first

**Kimi K3: NOT feasible on our stack today, and the blocker is not the tier.**
The architecture is **absent from the sglang registry** — we carry
`kimi_k25`, `kimi_linear`, `kimi_vl`, and no K3 — so there is no loader to
stream into. Even granting one, the honest floor is **~9.4 s/token
(0.106 tok/s) single-stream**, and **~0.12 tok/s at bs=8**, because our
storage delivers 1.8 GB/s against their 12.78. Add a hard disk blocker: the
container is 982 GiB plus ~1.42 TB staging against **729 GB free**.

**DeepSeek V4 Flash: the near-term vehicle, and it probably needs no NVMe tier
at all.** `DeepseekV4ForCausalLM` is in the registry with NextN and DSpark
variants, it is an actively maintained upstream target (PRs #33145/#33140 open
for reasoning-effort support), and our stack already has DSV4-specific
machinery — hybrid-SWA pools, HiSparse, the c4/c128 compression pools. If it
lands under ~150 GB in a 4-bit quant it fits **72 GB VRAM + 98 GB RAM** on the
existing #77/#123 tier with no new mechanism.

**So: V4 Flash near-term on the RAM tier; K3 is a late showcase gated on a
loader, ~1.9 TB of disk, and an NVMe tier that does not exist.**

---

## 2. Model support

| | Kimi K3 | DeepSeek V4 Flash |
| --- | --- | --- |
| in `srt/models/` | **no** — `kimi_k25.py`, `kimi_linear.py`, `kimi_vl.py` only | **yes** — `deepseek_v4.py` (+ `_nextn`, `_dspark`) |
| arch strings | — | `DeepseekV4ForCausalLM`, `…NextN`, `…DSpark` |
| upstream activity | not found | live: #33145, #33140 (reasoning effort), #31713 (SWA recompute), #32059 (shared KV prefill CP) |
| our fork extras | — | hybrid-SWA arch set, HiSparse, c4/c128 state pools |

**K3's missing loader is the first gap and it is not small.** WASTE converts
to its own container from the original weights; we would need the HF
architecture supported in sglang first. That is a model-enablement task of its
own, ahead of any streaming work.

**V4 Flash's exact total/active params and published quant sizes are not on
this box and I did not invent them** — that is the one input the test plan
must fetch first (§6). The decision rule is stated instead: **≤ ~150 GB of
4-bit weights fits the existing tier; above that it needs the NVMe tier.**

---

## 3. Fixposten (feasibility arithmetic before measurement)

### (a) Disk — a hard blocker for K3, today

| item | size |
| --- | --- |
| K3 container | 982 GiB |
| conversion staging | ~1.42 TB |
| `/spinning` free | **729 GB** |

Even the finished container does not fit, let alone the staging. **What must
happen before any K3 test**: free or add ~1.3 TB for the container alone, or
~2.4 TB to convert in place. Conversion **can** stage on another volume — it
is a read-original / write-container pass — so a second disk large enough for
staging plus a container on `/spinning` is the cheaper shape. **This is a
procurement/cleanup precondition, not an engineering one**, and it gates the
whole K3 line.

V4 Flash: unknown until §6 fetches the quant sizes, but a ≤150 GB checkpoint
fits today.

### (b) NVMe bandwidth — measured on this box

Cold reads, `iflag=direct`, no cards touched:

| probe | result |
| --- | --- |
| 4 GiB @1 MiB, offset | **1.8 GB/s** |
| 5.9 GiB @1 MiB | **1.8 GB/s** |
| 1.5 GiB @1 MiB, far offset | **1.8 GB/s** |
| 2 GiB @32 MiB (warm, same file) | 3.8 GB/s (ARC) |
| small cached file | 9.5 GB/s (ARC, not storage) |

**1.8 GB/s cold, reproduced three times.** Theirs: **12.78 GB/s** (rand 2
threads). We are at **0.141x their storage bandwidth**, and their engine is
I/O-bound by their own analysis — so this ratio, not compute, sets our floor.

### (c) RAM and the trunk — our structural advantage

Their 46.25 GB budget splits into a **27.5 GiB resident trunk** plus a
**17.56 GB expert cache**, and the trunk carries the dense/attention math on
laptop cores (~27 % of their arithmetic).

Ours: **72 GB VRAM across three cards** holds the trunk, and the dense math
runs on GPUs. That frees **essentially the whole 98 GB of host RAM for expert
cache** — call it ~80 GB usable, **4.5x their cache**. This is real and it is
our one large edge.

---

## 4. Ours vs WASTE, honestly

| | WASTE | us | net |
| --- | --- | --- | --- |
| storage bandwidth | 12.78 GB/s | **1.8 GB/s** | **theirs, 7.1x** — dominates everything |
| trunk compute | laptop cores, ~27 % of arithmetic | **72 GB VRAM, GPU** | ours, and it collapses their compute share |
| expert cache | 17.56 GB | **~80 GB** (trunk moved to VRAM) | ours, 4.5x |
| measured hit rate | **14 %** (3357/20195) | unmeasured | theirs — it is a number, ours is a hope |
| expert layout | purpose-built, one read per expert | safetensors, not expert-contiguous | **theirs** |
| read/compute overlap | measured, in-engine | #125 prefetch, VRAM<->RAM only | theirs at the disk tier; ours exists one tier up |
| NVMe tier | **exists** | **does not exist** | theirs — this is the build |
| K3 loader | its own converter | **absent** | theirs |
| batching | refused (single-stream engine) | native | ours, but see §5 — it buys little here |
| dependencies | zero, embeddable | full stack | theirs, irrelevant to us |

**Their edge is concentrated in exactly the thing we would have to build**
(container layout + disk-tier overlap) and in the one number we do not have
(a measured hit rate). **Our edge is the trunk on GPU and a 4.5x cache.**

### Does our edge close the gap? No — arithmetic

Their per-token read is **17.0 GB** at a 14 % hit rate. Holding the cold set
constant and improving only the hit rate:

| hit rate | cache | GB/token | s/token @1.8 GB/s | tok/s |
| --- | --- | --- | --- | --- |
| 14 % (theirs) | 17.56 GB | 17.00 | 9.44 | 0.106 |
| 30 % | ~80 GB | 13.84 | 7.69 | 0.130 |
| 40 % | ~80 GB | 11.86 | 6.59 | 0.152 |
| 50 % | ~80 GB | 9.88 | 5.49 | 0.182 |

A 4.5x cache lifting the hit rate from 14 % to even 50 % cuts bytes by 42 %,
against a storage deficit of **7.1x**. **The cache advantage cannot pay for
the bandwidth disadvantage**, and the honest single-stream figure is
**0.11–0.15 tok/s — three to five times slower than their laptop.**

---

## 5. Batching does not rescue it, and here is why

The hope is that with bs>1 the union of activated experts grows sublinearly.
For Qwen-style top-8 of 256 (the K3 analog; K3 is top-16, which is worse):

| bs | distinct experts/layer | expert-reads per token | aggregate gain |
| --- | --- | --- | --- |
| 1 | 8.0 | 8.00 | 1.00x |
| 4 | 30.5 | 7.63 | 1.05x |
| **8** | **57.4** | **7.18** | **1.11x** |
| 16 | 102.0 | 6.37 | 1.26x |
| 32 | 163.3 | 5.10 | 1.57x |

At bs=8 the union already covers 22 % of every layer's experts, so the saving
is **11 %** — 0.106 -> ~**0.12 tok/s**. bs=32 gives 1.57x and needs a 32-way
queue on a model emitting a token every several seconds, which is a latency
disaster rather than a throughput win. **The 4 % activation figure is per
token; it does not compound across a batch the way the hope requires.**

**We have no Qwen-side hit statistics to check this against.** I searched the
#77 offload work: the *mechanism* is there (`MoEExpertOffloadCache`, resident
fraction, pinned pool, async H2D) but **no recorded router distribution or
per-expert hit rate**. That absence is its own finding and it bears on #126 —
see the cut below.

---

## 6. Cuts, effort/yield

**Cut A — measure our router distribution and expert hit rate.** Env-gated
dump of renormalized top-8 weights and cache hit/miss for Qwen3.6-35B-A3B and
122B-A10B. **Effort S.** *Yield: high, and independent of #389* — it is the
precondition for #126, it is the only way to know whether WASTE's "the router
has no tail" verdict (measured on K3 top-16 and Kimi-Linear top-8, and
explicitly kept as a per-model instrument by its authors) applies to us, and
it turns the hit-rate row of §4 from a hope into a number. **Do this
regardless.**

**Cut B — fetch V4 Flash's real geometry and quant sizes**, then place it on
the ladder with the §3 arithmetic. **Effort S, desk.** *Yield: decides whether
the near-term vehicle needs any new mechanism at all.*

**Cut C — expert-contiguous container layout.** **Effort M.** *Yield:
prerequisite only; worthless without D.*

**Cut D — the NVMe tier** below #77/#123, reusing the #125 overlap
discipline. **Effort L.** *Yield: capability for >~150 GB models only, at the
floors in §4.*

**Not worth building if X**: if no model above ~150 GB is actually wanted
(everything we serve today fits 98 GB RAM, where #77 measured 6.97 tok/s); if
the target is speed rather than capability; or if cut B shows V4 Flash fits
the RAM tier and no other large vehicle is queued. **Rig-is-lower-bound**: the
1.8 GB/s is *this box*. On their 12.78 GB/s the same design gives their
0.5 tok/s, and this verdict must not be recorded as a verdict on the approach.

---

## 7. Test plan — scheduled LATE

Per the user: **during dashboard finalization, before the Docker build.**

**Preconditions, in order:**

1. **Disk procurement/cleanup** — ~1.3 TB free for the K3 container, ~2.4 TB
   if converting in place; staging may live on a second volume. *Blocks all
   K3 work.*
2. **A K3 loader in sglang** — the architecture is absent. *Blocks all K3
   work, and is a separate enablement task.*
3. Cut A (router/hit measurement) and cut B (V4 Flash geometry) — both desk
   or cheap, both should precede any card time.

**Vehicle 1 — V4 Flash (near-term, no tier needed if cut B confirms):**
boot on the existing #77/#123 RAM tier at TP=3 uneven; measure tok/s and
expert hit rate; compare against the 122B-A10B baseline (6.97 tok/s). One card
window. This is the useful test and it does not wait on disk or a loader.

**Vehicle 2 — K3 (late showcase):** only after preconditions 1 and 2.
Single-stream s/token and bs=8 aggregate against the §4 predictions; the
falsifier is whether the measured hit rate at ~80 GB of cache beats 14 %
enough to matter. Expect ~0.1 tok/s and treat anything above 0.2 as a finding
worth explaining.

**Reporting duty**: the #375 canon applies — first boot after a cache change
is a JIT outlier, and an A/B without a same-boot floor is not a measurement.

---

## 8. What transfers regardless

The **layout** (expert-contiguous, one read per expert) and the **streaming
discipline** (resident trunk, bounded cache, read-behind-compute) are sound
and portable; the C engine is not, and does not need to be. Their
read/compute overlap is the disk analog of our #125 prefetch, so the tier is
a third rung under machinery we already have — which is why cut D is L and
not XL. The reason to defer it is not difficulty; it is that **nothing we
currently want to serve needs it.**
