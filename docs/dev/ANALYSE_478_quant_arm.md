# ANALYSE 478 — moving the active DSV4F driver from UD-IQ3_XXS to UD-Q3_K_XL

Status: **DESK PHASE. No GPU number in this document.** Everything below is
derived from the checkpoints' GGUF tensor tables, from this rig's NVML/meminfo
state, and from code read at the cited file:line. The GPU window has not run.

Instrument: `scripts/dev/478_quant_footprint/gguf_footprint.py`. It parses the
GGUF tensor table only (header bytes; no tensor data is materialised), so it
costs no measurable time or RAM against a 120 GiB checkpoint. Its `--selftest`
passes a can-discriminate check on known-different inputs (expert vs shared
expert vs attention naming, per-type block costing, an over-subscribed card
reported infeasible against a roomy one reported feasible, repack growth vs no
growth) before any verdict below is allowed to count.

Power state at desk time (the user lowered all power targets today; every
number in this window carries its tag, and full-power anchors from previous
days are dead for comparison and quotable only as an order of magnitude with
a warning label):

| NVML idx | card | power.limit | default | max |
|---|---|---|---|---|
| 0 | RTX 3080 | 200 W | 320 W | 320 W |
| 1 | RTX 5090 | 400 W | 575 W | 600 W |
| 2 | RTX 3080 | 200 W | 320 W | 320 W |

---

## 1. Finding A — the device order has shifted since the proven recipe

The proven recipe (`/spinning/gpu-battery-results/2026-08-02_394_linkshards/boot394.sh`)
passes `--rank-gpu-id 0,1,2` together with per-rank vectors that are NOT
symmetric: `--rank-auto-reserve-mib 2200,1400,1400` and
`--rank-moe-resident-fraction 0.485,0.42,0.42`. The large entries belong to the
5090.

That boot's own launcher line records which card rank0 actually got:

```
rank->card vector (launcher placement (gpu_id_for_rank -> #331 IdentityMap)):
  rank0=GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d [00000000:0A:00.0], rank1=GPU-5c648f96-...
```
(`2026-08-02_394_linkshards/boot394_equal.log:25`)

`GPU-31d7ef41-...` is the RTX 5090. On 2026-08-02 it enumerated at **NVML index
0**. Today it enumerates at **NVML index 1**:

```
0, NVIDIA GeForce RTX 3080, GPU-5c648f96-be1d-42d5-0221-34d11ab137f7
1, NVIDIA GeForce RTX 5090, GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d
2, NVIDIA GeForce RTX 3080, GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4
```

**Replaying the recipe verbatim today would hand the 5090's reserve and
resident-expert budget to a 20 GiB 3080 and the 3080's budget to the 5090.**
The 3080 would be over-subscribed by roughly 9 GiB and the boot would die, at
best after the ~6 minute weight load. This is precisely the trap CLAUDE.md
names ("torch order != NVML order on this rig") and it has now bitten the
recorded recipe itself, because the recipe stores a *positional* answer to a
question whose answer moves.

**Action:** every boot script in this window resolves rank→card by **UUID** at
runtime and derives `--rank-gpu-id` from that. With today's enumeration the
recipe's semantics require `--rank-gpu-id 1,0,2`. The scripts must not contain
`0,1,2` as a literal. The recipe file in the results tree should be annotated,
not silently reused.

---

## 2. Finding B — UD-Q3_K_XL is 43 % MXFP4, and MXFP4 costs 29 % more in memory than on disk

Measured tensor-table composition (GiB, whole checkpoint, both tiers have the
same 129 expert / 1199 non-expert tensor split):

| type | UD-IQ3_XXS | UD-Q3_K_XL |
|---|---|---|
| IQ2_XS | 28.906 | — |
| IQ3_XXS | 57.422 | 64.312 |
| IQ3_S | 1.719 | — |
| **MXFP4** | **2.125** | **47.812** |
| Q6_K | 1.751 | 0.405 |
| Q8_0 | 4.877 | 6.621 |
| BF16 / F32 / I32 | 0.247 | 0.247 |
| **total** | **97.046** | **119.397** |
| of which routed experts | 90.172 | 112.125 |
| of which non-expert | 6.874 | 7.272 |

MXFP4 (ggml type 39) has **no kernel on this fork**. The loader repacks it
losslessly to Q5_0 inside `gguf_quant_weights_iterator`, and
`python/sglang/srt/model_loader/gguf_mxfp4_repack.py:39-41` states the price by
name:

> The price is bytes: 22 per block instead of 17, a factor 22/17 = 1.294 on the
> repacked tensors only. That is real RAM and real VRAM.

The same module's header independently confirms this measurement — it describes
exactly the tier analysed here: *"The published DeepSeek V4 Flash GGUF stores 45
of its expert tensors -- every routed `down` projection plus layer 26's
`gate`/`up` -- as MXFP4, 47.8 GiB of the 119.4 GiB file"*
(`gguf_mxfp4_repack.py:16-19`). Our independent tensor-table scan returns
47.812 GiB of 119.397 GiB. The docstring was written about **Q3_K_XL**, not
about the IQ3_XXS tier that is actually in service.

Consequence — the in-memory footprint, not the on-disk one, is what must fit:

| | UD-IQ3_XXS | UD-Q3_K_XL | delta |
|---|---|---|---|
| on disk | 97.05 GiB | 119.40 GiB | +22.35 |
| **in memory after repack** | **97.67 GiB** | **133.46 GiB** | **+35.79** |

**The naive disk-ratio estimate understates the real cost of this quant swap by
60 %.** Any planning that used +22 GiB was planning against the wrong number.

---

## 3. Finding C — where the bytes land, and the RAM wall

The two byte classes go to different places and only one of them is movable:

* **non-expert** tensors (attention, MLA projections, norms, embeddings, and
  the shared experts `ffn_*_shexp`, which run on every token and are never
  evicted) are unconditionally device-resident. No offload knob touches them.
* **routed expert** tensors (`*_exps`) are the only class the #77/#123 tier
  places: a fraction resident in VRAM, the remainder in the host cold pool.

With rank→card resolved correctly (rank0=5090), TP ratio 0.438/0.281/0.281,
the recipe's `--rank-auto-reserve-mib 2200,1400,1400`, the #493 corridor of
400 MiB free on **all** cards, and a KV+activation term of 2.0/1.2/1.2 GiB:

| | UD-IQ3_XXS | UD-Q3_K_XL |
|---|---|---|
| non-expert in VRAM | 6.87 GiB | 7.27 GiB |
| resident experts in VRAM | 54.51 GiB | 54.12 GiB |
| **host cold pool** | **36.28 GiB** | **72.07 GiB** |
| implied resident fraction vector | 0.611,0.592,0.592 | 0.436,0.423,0.423 |

VRAM is the binding constraint in both cases, and it is nearly saturated in
both — which is why the resident-expert total barely moves (54.5 → 54.1 GiB)
while **the entire +35.8 GiB of the quant swap lands in host RAM**.

### The cold pool is not reclaimable

`SwapTotal` is 0 on this host and `memory.max` is `max` (no cgroup limit), so
`MemTotal` = **104.0 GiB** is the wall. The stream-trim module states the
safety argument that also defines the wall
(`python/sglang/srt/model_loader/gguf_shards.py:479-486`):

> with no swap, cgroup reclaim cannot evict anonymous pages, so the pinned host
> expert pool, the CUDA host allocations and the Python heap are all
> structurally out of reach and only page cache can be taken

Whether the cold pool is the private `torch.empty(...).pin_memory()`
(`expert_offload.py:1430`, anonymous) or the #394 shared segment in `/dev/shm`
(`cold_tier_shm.py:130-132`, tmpfs, unswappable at swap 0), it is unreclaimable
either way; only the cgroup accounting bucket differs. This reconciles the
previous window's reading of `current=85.1GiB anon=14.1GiB file=70.2GiB`
(`2026-08-01_417_dsv4arch/ram417_w5.log`): the ~36 GiB cold pool sat in the
`file` bucket as tmpfs, ~34 GiB was genuinely reclaimable GGUF page cache, and
the 14.1 GiB `anon` is runtime overhead (Python heap + CUDA host allocations).

**The often-quoted "85.1 GiB" is therefore not a wall and never was.** The
binding quantity is `cold pool + runtime overhead`.

### The verdict for UD-Q3_K_XL

```
host cold pool             72.07 GiB   (unreclaimable)
runtime overhead          ~14.1  GiB   (measured, prior window)
                          ----------
unreclaimable floor       ~86.2  GiB
MemTotal                  104.0  GiB
                          ----------
left for page cache + OS   ~17.8 GiB
```

`/dev/shm` is 96 GiB and currently 4.2 GiB used, so the segment itself fits
(`cold_tier_shm.py:355` refuses if it would not).

**Verdict: UD-Q3_K_XL at TP=3 / context 8192 is FEASIBLE BUT MARGINAL — roughly
18 GiB of slack on a 104 GiB machine, and the slack is sensitive to a term this
desk analysis cannot pin (the true KV + activation VRAM at forward peak).**

Two consequences that must be built into the window rather than discovered in it:

1. **The recipe's stream-trim settings are actively wrong for this arm.**
   `SGLANG_GGUF_STREAM_TRIM_SOFT_GIB=88` / `TARGET_GIB=78` set a reclaim target
   of 78 GiB while the unreclaimable floor is ~86 GiB. `maybe_trim()`
   (`gguf_shards.py:529-540`) would then ask the kernel for a reclaim it can
   never satisfy, on every advice batch, for the whole load — reclaiming each
   page-cache page as fast as the loader creates it and turning a 6-minute load
   into a thrashing one. For the Q3_K_XL arm the marks must move **up**
   (SOFT ≈ 96, TARGET ≈ 90), not down. Replaying the recipe verbatim gets this
   backwards.
2. **The arm needs a pre-flight refusal.** The projected cold pool is
   computable before a single weight is read; the boot must refuse in seconds
   rather than after a 6+ minute load. The lever if it refuses is
   `--context-length` (8192 → 4096): every GiB of VRAM freed moves 1 GiB out of
   the host cold pool, on a roughly 1:1 exchange.

### Open, and decisive

The resident-fraction vector this model derives for UD-IQ3_XXS (0.611/0.592/
0.592) is **higher** than the vector the proven recipe actually used
(0.485/0.42/0.42, in that boot's rank order). The recipe is ground truth and
the model is an upper bound, so the real KV+activation term is larger than the
2.0/1.2/1.2 GiB assumed here — by enough to cost ~0.13 of resident fraction.
Carrying that same discount to Q3_K_XL gives a resident vector nearer
0.35/0.30/0.30, which pushes the cold pool from 72 GiB to roughly **86 GiB**
and the unreclaimable floor to ~100 GiB against 104 GiB of RAM — i.e. **it does
not fit.**

**This single unknown decides the arm.** It is cheap to close and must be
closed first in the window: boot the IQ3_XXS control arm (which we know boots),
read the actual per-rank KV+activation VRAM and the actual cold-pool size off
`SGLANG_FORWARD_PEAK_PATH` and the offload ledger, feed the measured term back
into `gguf_footprint.py`, and only then decide whether the Q3_K_XL arm is
launched at 8192 context, at 4096, or not at all.

That ordering is also free: the IQ3_XXS arm is required anyway as the
same-power-state comparison arm for #478.

---

## 4. What this changes about the window plan

* Arm 1 (#478) runs **IQ3_XXS first, not second.** It is the control arm, it is
  known-bootable, and it produces the one measurement that decides whether the
  Q3_K_XL arm is worth a 6-minute load. The briefing's order is inverted here
  for that reason.
* The `--rank-gpu-id` literal from the recipe must not be reused (Finding A).
* The stream-trim marks must be raised, not lowered, for the Q3_K_XL arm
  (Finding C).
* Quality comparison between the tiers is a genuine question and not a
  formality: the two tiers do not merely differ in bit-width, they differ in
  *type mix* — Q3_K_XL drops IQ2_XS entirely (28.9 GiB of the IQ3_XXS tier) and
  replaces it with MXFP4 that this fork cannot execute natively and must repack.
  The determined-answer probe is the gate.
