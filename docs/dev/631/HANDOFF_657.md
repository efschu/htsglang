# #656 HANDOFF v17 — successor 14

Written 2026-08-09, tree `/spinning/wt-631-routea`, branch `feat/route-a-631`.

Read this, then the DESIGN LAW header of
`python/sglang/srt/managers/phase_flip_presence.py` (corpse table A–H, now
plus corpse I here). Do not re-walk any corpse.

---

## 1. State

- **HEAD**: `ceb1b6f720` (two commits this shift, both on top of
  `e7a15a193f`). **NOT YET PUSHED** — push after the soak verdict.
- **Tests**: `bash scripts/run_631_flip_family.sh` → **581 passed**
  (was 565; +16 from the two new files below).
- **Serving**: port 30030 UP on the fixed build, POLICY=auto,
  pool 277468, ctx 393216. Booted 20:51:41Z.
- **Soak**: started 20:54:27Z, 65 minutes, `scripts/soak_631_mixed_load.py`.
  At 20:55: **48 flips committed, 0 scheduler exceptions.**

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

## 8. Exact next steps

1. **Finish the soak** (ends ~21:59Z), then push. Verdict criteria: 0
   scheduler exceptions, flips both directions, corridor floor 1024 held.
   Corridor series: `/spinning/evidence-631/corridor_soak.csv`, summary in
   `corridor_soak.summary`.
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
