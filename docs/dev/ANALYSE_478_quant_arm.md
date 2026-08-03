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

## 1. Finding A — two device index spaces, permanently offset

Recorded here because this desk pass first got it **wrong**, and the wrong
version is the more intuitive one.

The proven recipe (`/spinning/gpu-battery-results/2026-08-02_394_linkshards/boot394.sh`)
passes `--rank-gpu-id 0,1,2` with per-rank vectors that are NOT symmetric:
`--rank-auto-reserve-mib 2200,1400,1400` and
`--rank-moe-resident-fraction 0.485,0.42,0.42`. The large entries belong to the
5090, and that boot's own launcher line confirms rank0 got it:

```
rank->card vector (launcher placement (gpu_id_for_rank -> #331 IdentityMap)):
  rank0=GPU-31d7ef41-...-e773fd938f6d [00000000:0A:00.0], rank1=GPU-5c648f96-...
```
(`2026-08-02_394_linkshards/boot394_equal.log:25`)

`GPU-31d7ef41` is the 5090. Today `nvidia-smi` reports that UUID at **index 1**,
which reads like the enumeration order moved and the recipe had gone stale. It
did not. The two libraries simply index differently, and always have:

| | idx 0 | idx 1 | idx 2 |
|---|---|---|---|
| NVML / `nvidia-smi` (PCI bus order) | 3080 `05:00.0` | **5090** `0A:00.0` | 3080 `0B:00.0` |
| CUDA / torch (`CUDA_DEVICE_ORDER` unset ⇒ FASTEST_FIRST) | **5090** | 3080 `05` | 3080 `0B` |

`--rank-gpu-id` is **CUDA**-indexed: `gpu_id_for_rank()` returns
`rank_gpu_id[world_rank]` (`server_args.py:8476-8477`) and the per-rank vectors
are zipped against it positionally (`server_args.py:9111`). So `0,1,2` puts
rank0 on cuda:0 = the 5090 — exactly what the log recorded, in 2026-08-02 and
today alike. **The recipe is correct as written and must not be "fixed".**

The real hazard is the inverse, and it is a silent one. `--speculative-draft-gpu`
is likewise a CUDA index — the code says so by name
(`server_args.py:7274-7276`): *"it is a CUDA DEVICE INDEX, resolved through
gpu_id_for_rank()"*. Deriving it from `nvidia-smi` output would yield 1 for the
5090. That does **not** raise: rank1 legitimately maps to cuda:1, so the DSpark
draft head would be placed on a 3080, where the MXFP4 Marlin path does not exist
at all (SM90/SM120 only, ANALYSE_447 §1.5). Wrong card, no error.

Independent cross-check: the #530 serving boot script already encodes exactly
this, and transposes its own thresholds because of it
(`/tmp/w530_boot.sh:34-37`) — *"NVML index 1 is the 5090 and carries rank 0
(CUDA order != NVML order on this rig), so the budgets transpose"*. The rig
knew; this analysis briefly forgot.

**Action:** resolve the 5090 by **UUID through CUDA device properties**, never
by NVML index equality. Any per-rank quantity read from `nvidia-smi` (the #493
free-VRAM corridor, power tags, per-rank VRAM) must be mapped NVML→CUDA by UUID
before it is lined up against a rank. Every arm writes both orderings to
`$RUN/device_order.json` so each artifact records which index space it used.

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

The cold pool is the private `torch.empty(...).pin_memory()` of
`expert_offload.py:1430`: neither `boot394.sh` nor `boot417_w5.sh` sets
`SGLANG_MOE_COLD_TIER_SHM`, so `cold_tier_enabled()`
(`cold_tier_fetch.py:136-148`) was False and the #394 shared `/dev/shm` segment
was never in play. Either backing is unreclaimable at swap 0 — pinned pages
cannot be paged out, and tmpfs has nowhere to go — so the floor below does not
depend on which one is used.

**Unresolved, and deliberately not guessed:** the previous window read
`current=85.1GiB anon=14.1GiB file=70.2GiB`
(`2026-08-01_417_dsv4arch/ram417_w5.log`), and a ~36-50 GiB pinned pool cannot
fit inside 14.1 GiB of `anon`. So the pool is landing in the `file` bucket —
plausibly because CUDA pinned host memory is mapped through the nvidia driver
character device and accounted as file-backed rather than anonymous — but this
desk pass did not prove that, and a plausible mechanism is not a measurement.
The window resolves it directly: `rammon` records `anon` and `file` separately
and the offload ledger prints `pinned(host)`, so one IQ3_XXS boot settles it.

The attribution matters for *diagnosis*, not for the budget: the floor below is
derived from the in-memory footprint and the VRAM ceiling, neither of which
depends on which cgroup bucket the pool is counted in.

**What is settled is that the often-quoted "85.1 GiB" is not a wall.** It is a
total that includes reclaimable page cache. The binding quantity is
`host-side weight bytes + runtime overhead`.

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
* `--rank-gpu-id 0,1,2` is kept verbatim; what must be resolved by UUID is
  `--speculative-draft-gpu` and every NVML-sourced per-rank reading (Finding A).
* The stream-trim marks must be raised, not lowered, for the Q3_K_XL arm
  (Finding C).
* Quality comparison between the tiers is a genuine question and not a
  formality: the two tiers do not merely differ in bit-width, they differ in
  *type mix* — Q3_K_XL drops IQ2_XS entirely (28.9 GiB of the IQ3_XXS tier) and
  replaces it with MXFP4 that this fork cannot execute natively and must repack.
  The determined-answer probe is the gate.
* **Precondition checked and met:** the comparison is only a *quant* swap if the
  prompt format is held constant. The sidecar `tokenizer_config.json` carries no
  `chat_template`, which initially read as "this family has none" — but the GGUF
  metadata itself does: `tokenizer.chat_template`, 13772 bytes, and
  **byte-identical between the two tiers** (sha256 `e643c31f…` for both, read
  directly from each tier's first shard). So the template is extracted from the
  checkpoint rather than hand-written, and arm 1 is not silently also a
  prompt-format swap.
