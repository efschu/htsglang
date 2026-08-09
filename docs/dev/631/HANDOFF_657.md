# #656 HANDOFF v17 — successor 14

Written 2026-08-09, tree `/spinning/wt-631-routea`, branch `feat/route-a-631`.

Read this, then the DESIGN LAW header of
`python/sglang/srt/managers/phase_flip_presence.py` (corpse table A–H, now
plus corpse I here). Do not re-walk any corpse.

---

## 1. State

- **HEAD**: `e9e2a7ff20`, **PUSHED** to the fork (origin) as a
  fast-forward `e7a15a193f..e9e2a7ff20`. No force-push, no upstream.
  Last CODE commit is `54b688aa95`; everything after it is docs.
- **Tests**: `bash scripts/run_631_flip_family.sh` → **583 passed**
  (was 565 at v16; +18 across three new files).
- **Serving**: port 30030 UP on the fixed build, POLICY=auto,
  pool 277468, ctx 393216. Booted **21:05:07Z**.
- **Soak**: **IN FLIGHT** — started 21:05:08Z, 65 min, ends ~22:10Z.
  See section 8.1 for the evidence paths and the four verdict criteria.
  **Its verdict was NOT captured before this handoff was written; read it
  from the files.**

### Commits this shift

| Commit | Carries |
|---|---|
| `7eaf003ca3` | **Corpse I**: the TP→PP seam left draft state reachable |
| `ceb1b6f720` | Armed tp_to_pp now stops prefilling (defect O's other half) |
| `ae38a0518f` | #652 falsified + soak/corridor tooling |
| `8d964b060f` | Phase-evidence extract (correlated, not rate-inferred) |
| `54b688aa95` | **Corpse J**: `int(getattr(req, "req_pool_idx", -1))` on a present-and-None attribute |

### CORPSE J — read this before trusting a soak verdict

The first soak on the corpse-I fix ran **5 minutes** and then killed all
three ranks at 20:59:45Z on the first line of `_cutover`:

```
resident_req_identity  ->  int(getattr(req, "req_pool_idx", -1))
TypeError: int() argument must be ... not 'NoneType'
```

`req_pool_idx` is `Optional[int] = None`; a Req is visible in
`last_batch`/`running_mbs`/`chunked_req` from ADMISSION, before its slot is
allocated. The `-1` default only fires when the attribute is absent, and it
is not — it is present and None.

**Honest attribution**: the bug is latent in the previous build too, but
the armed-park narrowing (`ceb1b6f720`) is what began letting flips commit
BETWEEN prefill chunks, which is the window where an
admitted-but-unallocated request is reachable. The change created the
reachability, not the bug. It is the same Optional-None trap that
`build_flip_live_slots_fn` documents at length one file over.

**This is why the 60-minute bar exists.** The corpse-I fix looked fully
proven at 4 minutes (219 flip records, 0 deaths, the seam clearing real
resident requests) and was still one minute away from a different death.
Do not call a soak green early.

---

## 2. THE CRASH, and why it is fixed at the producer

`AttributeError: 'NoneType' object has no attribute 'topk_index'`, PP0,
2026-08-09 20:31:48Z, one pass after a tp_to_pp cutover.

`ScheduleBatch.merge_batch` branches on `if self.spec_info:` — **the
truthiness of SELF alone** — and then dereferences `other.spec_info`
unconditionally. A carried TP decode batch (live `EagleDraftInput`) merged
with a prefill batch built after the cutover in a phase that has no
drafter (`spec_info=None`).

**The falsified inference**, which was written in the code and cost the
instance: *"the TP→PP leg is flipping speculation OFF … its spec_info is
simply not read there."* It is read there — not by a drafter, but by
`merge_batch`.

`retune_carried_batches_for_phase` was never enough: it rewrites the
`spec_algorithm` **field**, and nothing on the merge path reads that
field. The crash log shows the retune reporting success one line above the
traceback.

**Fix (producer)**: `clear_spec_info_for_unspeculated_phase` drops
`spec_info` from every batch the PP loop can reach. Its reach is
deliberately **wider** than `harvest_resident_batches` — that harvest skips
empty batches and never looks at `last_batch`, which is the handle that
held the fatal side.

**Mirror, fixed at the same time**: `arm_draft_bootstrap` seeded
`running_batch` alone, so a non-empty PP extend batch in `last_batch` at a
pp_to_tp cutover produced the same illegal pair with roles swapped →
`arm_draft_bootstrap_all_reachable`.

**Additional loud guard** in `merge_batch`, both directions, RAISING. It
does not skip the merge: the tempting None-guard turns a loud crash into a
silent one (requests merged, one side's draft state dropped, wrong tokens
from a healthy-looking server). The `self=None / other=set` direction is
the more dangerous one — the old code reached it **without** crashing.

### METAL PROOF (this is the part that matters)

The wider reach is not theoretical. Live log, this build:

```
PHASE-FLIP cleared TP spec_info from 1 reachable batch(es) entering the
PP phase (requests -)
PHASE-FLIP cleared TP spec_info from 2 reachable batch(es) entering the
PP phase (requests 4d371d7c…, 62b54e1c…)
```

The first line's `(requests -)` is an **empty** batch — exactly what
`harvest_resident_batches`'s residency filter skips. The second names two
real resident decodes carried across a tp_to_pp cutover: the identical
configuration that killed PP0.

Crash recipe reproduced: epochs 4→12 back to back in 46 s with 2 resident
decodes and arriving prefill. **Zero deaths.**

---

## 3. Prefill ran in TP while a flip was armed — fixed

Production 20:31:38→20:31:48Z: tp_to_pp armed *because* pending prefill was
12747 tok > N=7004, then the scheduler built the next 2048-token chunk
every round for ten seconds until all 12747 tokens had been prefilled **in
the TP layout at ~1500 tok/s** (PP does it at ~4200). The cutover committed
into PP with 459 tokens left, and the policy immediately re-armed pp_to_tp.
The flip raced its own reason for existing — and those back-to-back epochs
are the interleaving that exposed corpse I.

**The defect was two copies of one rule.** The armed park exempted any
in-flight chunked request; defect O had already retired that premise for
`ready_fn` in the same session, and the park was never updated. Now one
definition, `chunk_blocks_quiescence`, with two callers (announcement
decision + build-the-next-chunk decision). True only while the chunk is
**mid-admission** (no pool row yet — its KV has no home the carry could
move).

Tests pin the **agreement** as a biconditional over every chunk state, not
the value, because the defect was never a wrong value.

---

## 4. #652 IS A MYTH — measured, not argued

`scripts/probe_652_device_total.py` (new). Bare process, no serving:

```
cuda:0  RTX 5090   mem_get_info total = 32088 MiB   NVML total = 32607   shortfall +519
cuda:1  RTX 3080   mem_get_info total = 20054 MiB   NVML total = 20480   shortfall +426
cuda:2  RTX 3080   mem_get_info total = 20054 MiB   NVML total = 20480   shortfall +426
```

**The 5090's CUDA context sees 32088 MiB, not the 19.58 GiB §6g asserts.**
The only gap is the ~519/426 MiB driver carve-out already documented in the
corridor rule. There is no 13 GiB driver wall.

§6g of `PROD_BRINGUP_BENCH.md` attributes the sizing ceiling to "the #652
residual (the 5090's CUDA context seeing only 19.58 GiB total)". **That
attribution is wrong and must be corrected there.** The 19.58 GiB figure is
the *3080s'* `reachable` value, which appears two lines away in the same
`BUDGET-REACH` log family — the numbers were crossed.

Also confirms the enumeration trap: torch `cuda:0` is the 5090 at **NVML
index 1**. Never index by NVML order.

---

## 5. KV CAPACITY: the real picture, and why the spill route is mostly built

`BUDGET-REACH[nvml]`, reproduced on both boots today:

```
rank 0 (5090):  budget 22700 MiB, holds 17.72 GiB, free 13.54 -> reachable 31.27 GiB, shortfall 0.00
rank 1 (3080a): budget 11920 MiB, holds  9.73 GiB, free  9.81 -> reachable 19.55 GiB, shortfall 0.00
rank 2 (3080b): budget 11970 MiB, holds 10.90 GiB, free  8.63 -> reachable 19.53 GiB, shortfall 0.00
```

**Shortfall 0.00 on every rank.** The physical-availability check is NOT
binding. The binder is the hand-set `RANK_MIB` in
`scripts/route_a_631_prod_boot.sh`, chosen from corridor sampling — roughly
9.1 / 7.9 / 7.8 GiB below reach.

### What is already exclusive (do not rebuild it)

- **KV backing**: §6f. Log: `PHASE-FLIP-BOOT released the PP KV backing
  (4320.00 MiB) … boot peak is max(PP, TP), not PP + TP`.
- **Weights**: already a spill. `phase_flip_boot.snapshot_and_free` builds
  a **pinned host image** per layout and frees the device storages; ONE
  arena sized `max(pp_bytes, tp_bytes)` is refilled per flip
  (`phase_flip_runtime`: *"EXTRA MOVERS (weights arena, GDN state) then
  CUTOVER"*, *"one arena / mutually exclusive VMM backing sized max(PP,
  TP)"*).

So the two largest classes the amended order names are **already** spilled
or exclusive. The spill-depth ladder should therefore target what is
**still resident in both phases**:

| Rung | Asset | Measured size/rank | Note |
|---|---|---|---|
| 1 | draft (MTP) weights | **1.86 – 2.01 GB** | `Load weight end … Qwen3_5ForCausalLMMTP`. Explicitly *"stay resident across both phases"* — but **PP has no draft worker at all**, so this is pure dead weight in PP. Best ratio on the ladder. |
| 2 | draft graphs | **~0.55 GB** (verify 0.26–0.30 + decode 0.17 + extend 0.12) | Useless in PP. |
| 3 | mamba / GDN state sets | not yet measured | `--gdn-resident-state-slots 10`. |

Rungs 1+2 ≈ **2.4–2.5 GiB per rank**, and the 3080s are the min-reducing
ranks, so that converts directly into pool. It is not the whole 8 GiB gap;
the rest is the corridor-driven `RANK_MIB` choice, which is an empirical
boot-and-sample question, not a mechanism question.

**Status: METAL-UNPROVEN.** No spill rung implemented this shift.

### 5a. THE SPILL-DEPTH LADDER — design, ready to implement

Implement in this order; each rung is independently shippable and
independently measurable.

**Flag surface.** `--phase-flip-spill-depth {0,1,2,3}` (default **0** =
today's behaviour, so the default path is unchanged, per the backward-
compatibility law). Depth is cumulative: 2 implies 1. Also accept
`SGLANG_PHASE_FLIP_SPILL_DEPTH` for A/B without editing the boot script.

| Depth | Spills at the tp→pp seam | Restores at pp→tp | Measured size/rank |
|---|---|---|---|
| 0 | nothing (today) | — | — |
| 1 | draft (MTP) weights | before `arm_draft_bootstrap` | **1.86 – 2.01 GB** |
| 2 | + draft CUDA graphs | before first TP decode | **~0.55 GB** |
| 3 | + GDN/mamba state sets not owned by PP | at cutover | not yet measured |

**The primitive already exists — do not write a new one.**
`phase_flip_boot.snapshot_and_free(named, layout, pin=True)` builds a
pinned host image and rebinds every `param.data` to a 0-sized placeholder;
`bind_arena_views` + `arena_refill` put it back and verify a checksum. That
is exactly a spill/restore pair, already used at boot for the two model
layouts. Rung 1 is that pair applied to `draft_worker`'s model at the seam
instead of at boot.

**The exact handle** (already documented by `draft_kv_pool`'s docstring, so
do not re-derive it): the cutover's `draft_worker` is the `EAGLEWorkerV2`;
its `.draft_worker` is the `EagleDraftWorker`; that worker's
`.draft_runner` is the `ModelRunner`. Rung 1 therefore spills
`draft_runner.model` via `checkpoint_param_dict(...)` →
`plan_arena_layout` → `snapshot_and_free`. Reach it defensively: a
phase-flip instance may be built with no speculation at all, in which case
every rung is a no-op, not an `AttributeError` inside the no-return region.

**Why rung 1 is safe and why it is first.** The PP phase has *no draft
worker at all* — `build_flip_draft_worker` returns None there, and the
cutover already documents the PP phase as "bit-for-bit the state an
instance without speculation has". The draft weights are therefore
provably unreachable during PP, which is the strongest possible precondition
for a spill: there is no correctness question, only a cost question.
Note the boot comment that says the draft's weights "stay resident across
both phases … there is no second layout for them to flip between" — that
explains why they were never *arena*-backed; it is not an argument that
they must stay resident.

**Where the hooks go.** Both legs are already ordered and named in
`phase_flip_runtime._cutover` step 7b, which this shift edited:
- tp→pp branch (next to `clear_spec_info_for_unspeculated_phase`): spill.
- pp→tp branch (immediately before `arm_draft_bootstrap_all_reachable`,
  which needs the drafter live to scrub its pool): restore.

**Restore latency is the whole cost, and it is on the pp→tp path**, i.e.
in the flip the user feels before decode resumes. Budget it: ~2.0 GB over
PCIe from pinned host memory is roughly 150-250 ms/rank at realistic
speeds, against a current pp→tp cutover of 997-1720 ms/rank. Fill the
table on metal:

| Depth | flip pp→tp ms/rank | flip tp→pp ms/rank | pool tokens | corridor min free |
|---|---|---|---|---|
| 0 | (baseline, measure) | | 277468 @ ctx 393216 | |
| 1 | | | | |
| 2 | | | | |

**Do not spill on a flip that will abandon.** Spill only after the cutover
has actually committed, never while merely armed — an abandoned flip that
had already freed the draft weights would return to TP with no drafter.

### 5b. What budget the ranks can actually take, now that #652 is gone

Measured this shift, both boots, at PP sizing time:

| rank | card | budget | reachable | headroom |
|---|---|---|---|---|
| 0 | 5090 | 22700 MiB (22.17 GiB) | 31.27 GiB | **9.10 GiB** |
| 1 | 3080a | 11920 MiB (11.64 GiB) | 19.55 GiB | **7.91 GiB** |
| 2 | 3080b | 11970 MiB (11.69 GiB) | 19.53 GiB | **7.84 GiB** |

`shortfall 0.00 GiB` everywhere: the physical check is not binding, and
with #652 falsified there is no driver wall behind it either.

That headroom is not all spendable — the TP stack's non-exclusive assets
are built *after* PP sizing and must fit in it: draft weights ~2.0 GB +
draft graphs ~0.55 GB ≈ **2.5 GiB/rank**. Spilling them during PP (rungs
1+2) is precisely what releases that claim, so PP sizing can take it.

**Projection, and it is a projection — label it METAL-UNPROVEN.** Rank 1
holds 9.73 GiB of weights against an 11.64 GiB budget, so its KV portion is
≈ **1.9 GiB** and that yields the current 277468. The 3080s are the
min-reducing ranks. Adding 2.4 GiB of spilled drafter to the KV portion:

    1.9 GiB -> 4.3 GiB  =  2.26x  ->  ~627k tokens

which lands in the **>600k class the order asks for**, at ctx 393216. At
ctx-262144-class settings it should be higher again. Two caveats that must
be measured, not assumed: (a) the KV-portion figure is derived from a
sizing-time reading, not read out directly — get it from the pool census;
(b) the corridor floor of 1024 MiB free per card is evaluated at the
runtime PEAK (flip + graphs + speculating decode), and this projection
spends memory that is currently part of that headroom. Raise `RANK_MIB` in
`scripts/route_a_631_prod_boot.sh` in one step, boot, read
`/get_server_info`, and re-sample the corridor under the acceptance load
before quoting any number.

#### 5c. MEASURED CORRECTION — sizing-time headroom is NOT runtime slack

Do not read the 7.9-9.1 GiB of §5b "headroom" as spendable. Measured live
during the 21:05 soak, 2464 samples at 100 ms (`corridor_final.csv`):

```
  3080a  min_free = 1553 MiB     breaches(<1024) = 0
  5090   min_free = 3698 MiB     breaches        = 0
  3080b  min_free = 1631 MiB     breaches        = 0
```

So at the RUNTIME PEAK the 3080s hold only ~530-610 MiB above the corridor
floor, not 7.9 GiB. The sizing-time `reachable` figure is taken before the
TP stack allocates its pools and captures graphs; almost all of that
headroom is genuinely consumed later by graphs, activations and
fragmentation. Anyone who raises `RANK_MIB` by "the headroom" will breach
the corridor immediately.

**This is exactly why the spill ladder is the right lever and a budget
bump alone is not.** The pool that IS the serving capacity is the PP pool
(§6e of PROD_BRINGUP_BENCH), and KV backing is already exclusive per
phase — so the binding constraint is the **per-phase peak**, not the sum.
Spilling the drafter's ~2.5 GiB during PP raises the PP phase's peak
headroom specifically, which is the phase whose pool sets
`max_total_num_tokens`. The TP phase keeps its drafter and its current
peak unchanged.

Revised expectation for §5b's projection, therefore: the ~627k figure is
reachable only if rung 1+2 actually return ~2.4 GiB to the PP phase at
runtime. **Verify by measuring the corridor min in the PP phase
specifically (not the whole-run min) before and after the rungs** — the
whole-run minimum is dominated by the TP phase and will hide the gain.

---

## 6. CUDA GRAPHS (item 4)

**The draft already runs under CUDA graphs — there is no gap.** Boot log,
all `backend=full`:

```
Capture draft extend CUDA graph begin. backend=full
Capture draft decode CUDA graph begin. backend=full
Capture draft verify CUDA graph begin. backend=full   (num_tokens_per_bs=4, bs=[1,2,3,4])
```

**AMENDED ORDER (user, mid-shift): do NOT leave draft graphs on just
because they capture. Measure, and if they bring nothing, take them out
and record the number.** So the task is an A/B, not a gap-closure.

Levers found so far:

- `envs.SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH` — gates the draft EXTEND
  leg only (`eagle_worker_v2.py:965, 2528`;
  `multi_layer_eagle_worker_v2.py:284`). The decode and verify legs need
  their own gate identified before a clean three-way A/B.
- Graphs are captured at BOOT, so "same boot" A/B is impossible in the
  literal sense. Read the instruction as "same thermal state": this tree
  has a recorded trap where a boot measured from COLD decays 4106 → 3809
  tok/s across six reps of one build, which reads as an 8 % regression
  that does not exist. Interleave A/B/A/B across boots and discard warmup,
  or the number is worthless.
- Metric: decode tok/s AND accept-len together. Graphs cannot change
  accept-len; if accept-len moves, the A/B is contaminated.

**DFLASH x graphs** (also ordered): prior fork history has DFLASH as
solo-draft neutral-to-negative. A graph win could overturn that verdict;
a null result closes it. Not started.

**PP prefill graphs**: currently `prefill.backend='disabled'`, and the
reason is NOT a deliberate choice —
`Breakable CUDA graph is incompatible with multimodal model; disabling
prefill CUDA graph`. The model resolves as multimodal
(`Qwen3_5ForConditionalGeneration`). The PP stack additionally logs *"no
CUDA graphs captured by construction (eager prefill per the #625
recipe)"*. Measuring the gain therefore needs that auto-disable path
understood first. **METAL-UNPROVEN, not measured.**

---

## 7. FLIP THRASH — new, name it before tuning anything

Under the soak's 4 s prefill cadence the policy committed **epochs 4→12 in
46 seconds**, each cutover costing 1.0–1.7 s per rank. Two flips per
injected prompt. Stability held (0 deaths), but the instance spends a large
fraction of wall time in cutovers.

Measured consequence under that load: **no decode stream finished 512
tokens in four minutes** (`decode_tok=0` at 20:58 with `ok=29`), while
prefill sailed to 413k tokens. Roughly 39 % of wall clock went into
cutovers (73 real flips x ~1.3 s in ~4 min).

**DO NOT "FIX" THIS BY RAISING `min_dwell`.** I started to and stopped: the
boot script's `PHASE_POLICY_MIN_DWELL_S=3` is a *measured, deliberate*
choice, and its own comment carries the falsifier:

```
min dwell 15s -> ~14300 of 32768 tokens already computed in TP; 1485 tok/s
min dwell  2s -> policy arms after the FIRST chunk;              4524 tok/s
```

`min_dwell` gates ARMING, and the §3 admission hold only takes effect once
armed — so a longer dwell puts prefill straight back in TP at 1485 tok/s,
re-breaking the very thing this shift fixed. The code comment already names
the right lever: *"The structural thrash protection is the hysteresis band
around N, not this."*

**And the finding is partly an artifact of the driver.** A 12000-token
prompt every 4 s is far more prefill-dominant than real agent traffic;
each prompt genuinely wants PP and each decode genuinely wants TP, so at
that cadence the "thrash" is the design working as specified under a load
no agent produces. The honest next step is to measure the flip-overhead
fraction under the REAL agent traffic of the green criterion (§8.2) before
touching any policy constant.

---

## 7a. DEFECT K (NEW, TOP PRIORITY) — the instance sticks in PP and
## decodes there with no graphs and no speculation

**This is a consequence of my own commit `ceb1b6f720` and it is not fixed.
Do not read the soak's "0 deaths" as the feature working.**

Measured, 21:05-21:21Z soak, `scripts/phase_evidence_extract.sh`:

```
  PREFILL  in PP: 765 records, mean 4043.4 tok/s     <- the hold works
           in TP:   5 records, mean 1147.3 tok/s
  DECODE   in PP: 412 records, cuda graph True 0.0%, accept len 0.00
           in TP:   1 record
  flips: 6 tp_to_pp, 6 pp_to_tp  (12 total in 16 minutes)
```

Decode is running **in the PP phase**, eagerly, with no CUDA graphs and no
draft chain — i.e. at the ~16 tok/s PP decode rate instead of the ~100
tok/s TP+NEXTN rate. That is the exact opposite of the feature's purpose.

**Mechanism.** The policy arms `pp_to_tp` on "prefill down to X tok <= N".
Before the admission hold, the armed window drained pending prefill *in
TP*, so pending fell, `pp_to_tp` armed, and the instance thrashed (73 flips
in 4 minutes — see §7). With the hold, pending prefill is no longer
consumed during the armed window, so under a prefill-saturated load it
never falls below N and `pp_to_tp` never arms. The instance parks in PP and
every resident request decodes there.

**Both behaviours are the same policy defect seen from two sides**: the
policy decides on *pending prefill alone* and never weighs the cost it is
imposing on resident decodes. The hold did not create that; it moved which
end of it you fall off. §7's conclusion stands and is reinforced — the
lever is the hysteresis band and a decode-starvation term, NOT `min_dwell`.

**Suggested shape (unimplemented, do not land without metal):** give the
policy a decode-starvation term — if requests have been resident and
decoding in PP for longer than some bound, arm `pp_to_tp` regardless of
pending prefill, then let `min_dwell` bound the return. The correct
constant is a measurement: compare tokens-delivered-per-second across the
whole mix, not prefill throughput alone.

**Verdict impact.** Criterion (c) of §8.1 is **NOT met** in its intended
sense: flips do commit in both directions (6/6), but decode is not landing
in TP. Stability (a), (b) and the corridor (d) are green so far. A soak
that survives is not the same as a soak that serves.

## 8. Exact next steps

1. **THE SOAK IN FLIGHT.** Started **21:05:08Z**, 65 minutes, ends
   **~22:10Z**, on build `54b688aa95` (code identical to `e5e14be9e2`;
   later commits are docs only). Evidence files, all still being written:

   - driver: `/spinning/evidence-631/soak_final.log`
   - corridor 100 ms series: `/spinning/evidence-631/corridor_final.csv`,
     summary appears in `corridor_final.summary` when it ends
   - server: `/spinning/evidence-631/server_info_final.json`
     (pool 277468, ctx 393216)
   - serving log: `/spinning/serving-30030.boot.log`

   **Verdict criteria, all four required:**
   (a) `grep -c "Scheduler hit an exception" /spinning/serving-30030.boot.log`
       == 0 for the whole window;
   (b) `err=0` on the final soak line;
   (c) flips committed in BOTH directions
       (`bash scripts/phase_evidence_extract.sh` — prefill records in PP,
       decode records in TP with graphs);
   (d) corridor min free >= 1024 MiB on every card, 0 breaches.

   At T+2.5 min: `ok=6 err=0`, 0 exceptions, decode completing, health 200.
   **A 4-minute green means nothing here — corpse J appeared at minute 5.**
2. **Green criterion (user, item 5)**: real Qwen agents through the router
   (30099 → 30030) completing tasks, with a phase-evidence extract showing
   prefill tok/s tagged PP and decode tagged TP with graphs live. NOT DONE.
3. **Spill rung 1** (draft weights out of VRAM during PP) — best
   ratio/effort on the ladder; reuse `snapshot_and_free` + the arena refill
   pattern, gated by a user-selectable depth flag.
4. **Correct §6g** of `PROD_BRINGUP_BENCH.md`: the #652 attribution is
   falsified (§4 above).
5. **Flip thrash**: measure before tuning (§7).

## 9. Rules that bit this shift

- `pkill -f "<script name>"` matches **your own wrapper's command line**
  and kills your shell (exit 144). Kill by PID.
- A load generator against `--enable-prefix-caching` **must** vary its
  prompts. The first version reused one filler; the cache served it and
  prefill collapsed from ~13000 tokens to `#new-token: 1063`, so
  `pending prefill > N` never held and tp_to_pp was never armed for the
  reason the soak needs. Fixed with a per-request high-entropy prefix.
- A `Monitor` on `PHASE-FLIP DONE` is far too chatty during a flip-thrash
  soak. Filter to failures only.
