# #656 HANDOFF v24 — successor 21

Written 2026-08-10, tree `/spinning/wt-631-routea`, branch `feat/route-a-631`.

Read this before HANDOFF_663. 663's two capacity closures were both wrong, and
663 says so itself in its §12 — but its replacement instruction ("build the
VMM spill, the inactive layout's weights are the missing mass") is **also
wrong**, for a reason that is now measured rather than argued. That correction
is the most useful thing in this document.

---

## 1. MY ERRORS, ranked — read these before my results

**0. I diagnosed a phantom bug, and the false step was comparing two
instants.** The spill rung logged `NVML free 10371 MiB` for a card
`nvidia-smi` showed with 3149, so I concluded it was reading the wrong device
— reasoning that every worker sees all three cards — and "fixed" it. **Both
halves were wrong.** `--rank-gpu-id` gives each worker a
`CUDA_VISIBLE_DEVICES` with exactly one physical GPU, so device 0 was always
this rank's card. And the two figures were never comparable: the rung runs at
the SEAM, just after the source pool hands its physical pages back, where free
memory is legitimately gigabytes higher than in either steady state. At pool
600000 that pool is ~8 GiB on rank 0, and the seam census independently shows
a seam maximum of 10974 MiB on that card. I only caught it because the
per-phase corridor table put the seam column next to the phase columns.
**Comparing an instrument's reading against a different instant is not a
cross-check**, and "the number looks impossible" is a hypothesis, not a
finding. The explicit device argument is kept — an absolute memory figure
should name its card — but the reason in my first commit message is wrong and
this supersedes it.

**1. I spent two boots optimising a layer split before checking that the
splitter did what I asked.** `PP_STAGE_RATIO=15,10,7` came back as **16/9/7**.
I only noticed because rank 0's arena was 14936 MiB — the 16-layer value —
when my model said 14209. `14,10,8` maps exactly; `15,10,7` does not. I never
found the rule. **Verify the actuator applied your setting before you spend a
measurement on it**, and see §6 for what is still unknown here.

**2. My desk arithmetic predicted the wrong binding phase, twice.** I
predicted PP would bind rank 0 at pool 190000; the measurement said TP bound
both 3080s and the SEAM bound the 5090. Then at pool 600000 the binding phase
on rank 0 genuinely did become PP — because the PP pool grows faster there.
**The binding phase is a function of the pool size, so it must be re-measured
at every pool you evaluate.** Every "this asset is cold in the non-binding
phase, therefore worth 0" argument in this corpus silently assumes a fixed
binding phase, and that assumption is false.

**3. I did not reach the acceptance target, and the reason is geometric, not
a missing spill.** See §4. I am stating it up front so no successor reads §3's
result as "600k works".

---

## 2. THE LEDGER CLOSES, AND IT CLOSES ON A TERM NOBODY HAD ITEMISED

Full write-up with every row: `PROD_BRINGUP_BENCH.md` §"SUCCESSOR 21 / 1".

The conservation identity said the flip setup holds 6.4 / 4.5 / 3.5 GiB per
card that plain TP3 does not. 663 §12 concluded that mass had to be the
inactive layout's resident bytes and ordered a VA-stable weight spill.

**It is not resident bytes. It is post-boot allocator growth.**

| | rank0 | rank1 | rank2 |
|---|---|---|---|
| NVML free at `boot_complete` | 7843 | 5764 | 5034 MiB |
| corridor minimum under load (663 §9) | 1646 | 1369 | 1037 MiB |
| **decay after boot sizing** | **6197** | **4395** | **3997** MiB |
| gap the identity demands | 6554 | 4608 | 3584 MiB |

Agreement within half a GiB on every card. The inactive layout occupies **zero
VRAM** — 663 §8 was right about that and wrong to then discard the finding:
the arena is `max(pp,tp)`, `snapshot_and_free` leaves a pinned host image, and
the two KV pools are provably never both backed (release-before-restore at
`phase_flip_runtime.py:1489-1491`, asserted at `phase_flip_boot.py:718-734`,
two disjoint page sets). 663's own unitemised leads — "two KV pools",
"duplicate graph pools" — are now closed at **0 MiB** each, by code.

### The coefficient

`scripts/s21_scratch_ladder.py`, one request per rung, short-to-long:

| prompt tokens | marginal drop (MiB) | cumulative floor |
|---|---|---|
| 2686 / 5370 / 10737 | 0 | 5017 / 6652 / 4347 |
| 21476 | 200 / 300 / 180 | 4817 / 6352 / 4167 |
| 42948 | 500 / 640 / 460 | 4317 / 5712 / 3707 |
| 85894 | 902 / 1320 / 924 | **3415 / 4392 / 2783** |

**~19-26 MiB of sticky allocator reserve per 1000 prompt tokens per card.**

**This is the mechanism behind successor 20's law.** 663 recorded "the corridor
decays in STEPS, not as a drift; a plateau of ANY length only means no larger
request has arrived yet" and noted the misread had cost three successors. That
is exactly a high-water indexed by the longest prefill seen. The law was
correct and its cause was never named, so it read as weather rather than as a
bug.

### Residue vs peak, separated, because the rung must not be credited with both

Re-running the ladder with a release before each rung:

| prompt tokens | concurrent PEAK (live) | RESIDUE (reclaimable) |
|---|---|---|
| 21476 | 1640 / 1042 / 1112 | 240 / 320 / 220 |
| 42948 | 2118 / 1562 / 1572 | 480 / 640 / 462 |
| 85894 | **3042 / 2624 / 2456** | 922 / 1322 / 882 |

## 3. WHAT I BUILT: spill ladder rung 1, and it works on metal

`--phase-flip-spill-depth {none,cache,draft,draft+graphs}`, cumulative,
integers 0..3 accepted, env `SGLANG_PHASE_FLIP_SPILL_DEPTH`, default `cache`
under `--enable-phase-flip`, refused without it.

Rung 1 returns the outgoing phase's cached allocator segments to the driver
**between** the source pool's release and the destination's restore. That
instant is not merely convenient: the outgoing layout's scratch is dead by
construction, the source's physical pages have just gone back, and the restore
is about to ask the driver for RAW pages — the allocation whose documented
failure mode (`phase_flip_runtime.py`, the comment above the swap) is torch
sitting on blocks it is not using.

Falsifier before the code: `/flush_cache` on the live instance took free from
**3911/4392/2911 to 6605/7846/5405 MiB** — full recovery to boot level. Before
this change, `torch.cuda.empty_cache()` appeared 4x in `phase_flip_boot.py`
and **0x** in `phase_flip_runtime.py`: the runtime flip had never reclaimed.

Metal: the rung fires at every cutover in both directions, returning
**120-200 MiB per flip**.

Tests: `test_phase_flip_spill_depth_631.py`, 14 cases, in the family.
Suite **662 passed / 0 failed** (649 + 13; a 14th was added after).
Can-fail proven: moving the reclaim after `restore_backing` — a mutation that
still calls it — fails the two ordering tests.

### Two defects found in the pre-existing dead module

`phase_flip_spill.py` was not a partial implementation of spec item 6; it was
a broken one.

* its docstring asserted call sites "in `phase_flip_runtime._cutover` step 7b"
  that **do not exist anywhere in the tree**. A whole-tree grep for
  `get_spill_ladder|on_enter_pp|on_enter_tp` matches only the file itself.
  I left the claim visible with a correction attached, because a docstring
  asserting an absent call site is precisely what let five successors believe
  item 6 was implemented.
* its `restore()` calls `allocate_arena` afresh, so the draft weights' device
  addresses **move** — and the TP decode graphs bake them. Wiring rung 2 as
  written would have corrupted the graphs. Rungs 2-3 are therefore **refused,
  not clamped**: a clamp would make a depth sweep report rung 3 as worth
  exactly what rung 1 is worth, which reads as a measurement.

## 4. THE ACCEPTANCE TARGET: 600000 BOOTS, AND WHAT STOPS IT HOLDING

**`max_total_num_tokens=600000` boots and serves.** The pool sizer was never
the obstacle, and the "pool >= 600000 is structurally unreachable" verdicts are
dead. What is not yet met is 600000 **with the 1024 MiB corridor held under
load**.

Boot flags: `PP_STAGE_RATIO=14,10,8`, `SGLANG_UNEVEN_TOKEN_VECTOR=14,10,8`,
`MAMBA_SLOTS=12`, `RANK_MIB=31800,17400,17450`, `MAX_TOTAL_TOKENS=600000`,
CTX 393216, purity strict, POLICY auto, spill depth `cache`.

Config search across four boots at pool 600000 (idle NVML free, MiB):

| geometry | rank0 (5090) | rank1 (3080) | rank2 (3080) | min |
|---|---|---|---|---|
| 2,1,1 + vector 28,26,20 + mamba 20 | 908 | 3043 | 1847 | 908 |
| 14,10,8 + vector 29,27,18 + mamba 20 | 3720 | 1467 | 1869 | 1467 |
| 15,10,7 (→ became 16/9/7) + mamba 20 | 908 | 2921 | 2711 | 908 |
| **14,10,8 + vector 14,10,8 + mamba 12** | **4280** | **1807** | **2149** | **1807** |

The bottleneck moves between cards as the split changes; the last row is the
best balance found. It leaves **783 MiB of transient budget** above the floor
on rank 1, against a measured concurrent peak of ~1042 MiB for a 21k-token
prefill. **So it holds for short prefills and breaches for long ones**, and
the breach length is computable from §2's coefficient rather than discovered.

### The geometry model, calibrated — use this instead of guessing

Per-rank PP weight bytes are linear in the layer count:

```
weights_pp(rank) = fixed_rank + 727 MiB x layers_rank
  fixed: rank0 3304 MiB (embedding)   rank1 874    rank2 3299 (lm_head)
arena(rank)      = max(weights_pp(rank), weights_tp(rank))
  weights_tp from --phase-flip-tp-vector 30,17,17: 13163 / 7924 / 7924 MiB
KV_pp(rank)      = 0.5334 GiB/layer x layers_rank      (at pool 600000)
KV_tp(rank)      = 17.05 GiB x token_share_rank        (at pool 600000)
resident(rank)   = arena + max(KV_pp, KV_tp) + other
  other (measured, incl. draft, mamba, graphs, context): 7756 / 4500 / 5124 MiB
```

Verified against four boots to within ~200 MiB. **There are THREE vectors and
they are independent**, which the boot log itself warns about: `pp_stage_ratio`
sets the PP layer split *and* the PP weight shards; `--phase-flip-tp-vector`
sets the TP weight shards; `SGLANG_UNEVEN_TOKEN_VECTOR` sets the TP KV token
split. Only the third is free of weight consequences.

### The structural finding that matters most, and it cuts against the spill

**Aligning the PP layer shares with the TP token shares maximises capacity and
simultaneously destroys the spill's opportunity.** With shares aligned, every
rank's `max(KV_pp, KV_tp)` equals both, the misalignment overhead (1.1216 of a
single layout, i.e. +12 %) goes to zero, and the arena tail
`arena - tp_bytes` goes to zero. But those two quantities *are* the cold
assets a phase spill reclaims. Misalignment creates the cold bytes that the
spill then partially recovers; alignment removes them outright.

So the ordered mechanism in spec item 6 ("spill everything cold of the
inactive layout") is worth **strictly less** than the permission the user gave
in spec item 2 ("the PP prefill phase's KV layout may be rebuilt for this").
A successor should spend the item-2 permission first and treat deeper spill
rungs as what is left over.

## 5. WHY DEEPER SPILL RUNGS ARE NOT THE NEXT MOVE (and when they would be)

The genuinely cold asset is the **draft (MTP) model**: 2058/1925/1925 MiB,
resident in both phases, and the PP phase has no drafter at all. Its worth is
entirely a function of which phase binds:

* pool 190000, misaligned: TP binds both 3080s → draft spill worth **0** there.
* pool 600000, `2,1,1`: PP binds rank 0 → draft spill worth its **full
  2058 MiB** on the card that was 116 MiB under the floor.
* pool 600000, aligned `14,10,8`: neither phase dominates on rank 1 → back
  toward **0**.

Which is the same lesson as §4: the aligned geometry that gets closest to the
target is also the one where the spill has least to take. Deeper rungs need a
VA-stable carrier (`KvVmmArena`, whose commit/decommit the KV pools already use
with addresses held fixed — `test_kv_vmm_dial.py` asserts a shrink/grow
round-trip with stable addresses). That is real work; do it only after the
geometry is exhausted, and only against a re-measured binding phase.

## 6. WHAT I DID NOT REACH

* **The `15,10,7 → 16/9/7` remap rule is unexplained.** `14,10,8` maps
  exactly. Until the rule is known, every stage-ratio experiment must verify
  the achieved split from the `TP stack built` arena figure (or the PP KV
  `K size`) before it means anything.
* **Rung 2/3 on a VA-stable carrier.** Designed and argued, not built. §5 says
  why it was deprioritised, which is a judgement a successor may reverse.
* **A hard runtime ceiling.** Confirmed by code audit: `--rank-gpu-memory-mib`
  is **ADVISORY ONLY** (it becomes `mem_fraction_static`, consumed once in
  `_profile_available_bytes`), `torch.cuda.set_per_process_memory_fraction` is
  called **nowhere** in the tree, and the VRAM dial resizes only the VMM-backed
  KV tail against a floor measured once and never re-checked. So nothing stops
  the allocator expanding into the corridor. A per-rank
  `set_per_process_memory_fraction` derived from `RANK_MIB` would convert the
  corridor from a measured outcome into an enforced invariant — the allocator
  would reclaim instead of expand. **This is the single highest-value unbuilt
  item I am aware of** and it is a small change.
* **Decode decomposition (program item 4), prefill chunk A/B (item 5).** Not
  started. Note for item 5: the transient coefficient in §2 is measured *at*
  `chunked_prefill_size=2048`, so a larger chunk is expected to raise it and
  should be disqualified on corridor before it is judged on speed.
* **The host-RAM lever.** `RssShmem` 34.2/17.5/25.8 GB on PP0/1/2 — the pinned
  weight images, 58.3 GiB, unswappable with no swap configured, against a
  117 GiB box. That is the mechanism of the `oom_kill 9` that ended 663's run.
  The images are READ-ONLY masters that also exist on disk; file-backing them
  converts 58 GiB of unreclaimable shmem into reclaimable page cache. Not
  taken — the VRAM axis was the ordered work.

## 7. INSTRUMENTS I ADDED, and the one that already existed

* `scripts/s21_scratch_ladder.py` — scratch high-water vs prefill length, with
  `--flush-between` to separate concurrent peak from accumulated residue. Run
  it BOTH ways; neither run alone can tell you what a release is worth.
* `scripts/s21_phase_corridor.py` — cuts the corridor series at the log's own
  `event loop re-dispatch` instants and reports the minimum per phase, with a
  settle margin so the cutover is neither phase. **The aggregate minimum every
  earlier handoff quotes cannot identify the binding phase**, and without the
  binding phase no spill claim can be evaluated.
* **Already in the tree and already armed, which nobody used:**
  `SGLANG_VRAM_FLIGHT_DIR=/spinning/flight_605` is set by the boot script, so
  `mem_ledger.flight_recorder` has been recording per-rank
  allocated/reserved/peak/non-torch/NVML marks at every boot phase for this
  entire chain. 663 believed it was unset — it read the log instead of
  `/proc/<pid>/environ`. There is also a complete by-name VRAM ledger engine
  (`sglang/srt/mem_ledger/`, 13 named terms) behind `--enable-vram-ledger`,
  with a calibration cached for this rig
  (`~/.cache/sglang/vram_calibration-a191a0712717.json`). **Turn it on before
  hand-building any itemisation.**

## 8. RESULT: 500000 HOLDS. That is +92 % on the shipped pool, measured.

Boot: `PP_STAGE_RATIO=14,10,8`, `SGLANG_UNEVEN_TOKEN_VECTOR=14,10,8`,
`MAMBA_SLOTS=12`, `RANK_MIB=31800,17400,17450`, `MAX_TOTAL_TOKENS=500000`,
CTX 393216, purity strict, POLICY auto, spill depth `cache`, HEAD 12d820fa8b.

Load history, stated with the row because the corridor is a property of it:
bs=4 soak (4 streams, mixed long prompts) for the whole window, plus real qwen
agent traffic through router 30099, plus a deliberate prefill ladder driving
21476 then 42948 then 85894 prompt tokens to force the sticky high-water
rather than wait for it.

| axis | result |
|---|---|
| pool | **500000** |
| corridor minimum, per phase, MiB | pp **1181 / 3676 / 1527**, tp 1483 / 4006 / 1793 |
| corridor floor | 1024 — **held on all three cards** (margin 157 / 2652 / 503) |
| binding phase | **PP on all three cards** at this pool |
| flips | 132, balanced 66 `pp_to_tp` / 66 `tp_to_pp` |
| spill rung fired | **132 of 132 flips** |
| prefill only in PP | 606 prefill batches, **0 with a CUDA graph** |
| decode only in TP | 144 decode batches, **144 carrying `accept len`** |
| purity gate active | 22 refusals of `prefill cannot run in tp` |
| accept length | 2.55 |
| tracebacks / exits | **0 / 0** |
| host RAM | `oom_kill` unchanged at 9 (no new kill) |

**Against the inherited state this is the headline: 663 shipped 190000 and
recorded 260000 as breaching. 500000 holds with graphs, speculation, strict
purity and agent traffic simultaneously.** 600000 boots and serves but breaches
(§4), so the acceptance target is approached and not met.

### The honest caveat, and it is the one that will bite a successor

Rank 1's margin is **157 MiB**, and it was measured with an 85894-token prefill
in flight. The ceiling is a function of the longest prefill admitted, not of
elapsed time: by §2's coefficient each additional 1000 prompt tokens costs
19-26 MiB per card. A deployment that admits materially longer prefills than
this window did will breach at 500000, and the pool must come down by
`breach_MiB / 9.10` thousand tokens on rank 1 to compensate. **Do not quote
500000 without the prefill length it was measured at.**

### What would buy the last 100000 tokens

In descending confidence:

1. **A hard runtime ceiling** (§6): `set_per_process_memory_fraction` per rank
   from `RANK_MIB`. Converts the corridor from an outcome into an invariant and
   makes the allocator reclaim instead of expand. Small change, unbuilt.
2. **Rank 2's arena.** It is PP-bound at 9115 MiB because rank 2 carries the
   lm_head (3299 MiB fixed) on top of 8 layers. Moving one layer off rank 2
   costs 727 MiB of arena and 533 MiB of PP KV there — but the layer has to go
   somewhere, and every destination was worse in the four boots tried. A
   non-uniform search over (layers, tp weight vector, token vector) jointly is
   the unexplored space; §4's calibrated model makes that a desk exercise
   before it is a boot.
3. **Spill rungs 2-3** on a VA-stable carrier — but read §5 first: at the
   aligned geometry that gets closest to the target, the draft is the only
   genuinely cold asset and PP now binds, so rung 2 is worth its full
   2058/1925/1925 MiB *in the binding phase*. That is the one configuration in
   which the ordered spill would pay, and it is this one. This reverses the
   priority I gave in §5 for the aligned case and is the strongest remaining
   lead.

## 9. DEFECT FOUND AT THE END, AND IT IS A LIVELOCK — read this first

At 07:48Z, pool 500000, with an **85894-token request resident**, the instance
stopped answering `/health` and never recovered. Not a crash: all three
schedulers stayed in state R, the log kept advancing, no traceback, no exit.

The log says exactly what happened, once per flip attempt, forever:

```
PHASE-FLIP FLIP ABANDONED (pool too small for the live set): pp_to_tp.
This rank: staging 2149 MiB needed but only 2136 MiB is spendable (driver ...)
```

**The three facts compose into a deadlock:**

1. strict purity forbids decode in the PP layout — the resident request can
   only make progress after a flip to TP;
2. `_staging_affordable` (`phase_flip_runtime.py:2693-2740`,
   `DEFAULT_STAGING_RESERVE_BYTES` = 1024 MiB) declines the flip because rank 1
   is 13 MiB short of the staging reserve;
3. the thing that would free that memory **is the flip**, because the resident
   request's KV is what makes staging unaffordable.

So the guard's refusal is itself the condition it is refusing on. **This is the
third time this chain has met that exact shape** — 663 §2 records the same
structure in the resident-carry guard and says "a detector that only declines
to act is not containment; it is now written into the code rather than into a
handoff". It was written into *that* catch site. This is a different one, and
it was not.

Margin: **13 MiB**. It is not a sizing accident that a smaller pool avoids; any
pool has a resident set that reaches it, and the larger the pool the longer the
admissible request that gets there.

**This is the single most important thing in this handoff.** It is a
correctness/availability bug, it is reachable at the shipped configuration, and
by the standing "bugs before features" rule it outranks the remaining capacity
work. Candidate fixes, in the order I would try them:

* make the abandon path **actionable**: if a flip is declined for staging and
  the live set can only drain in the other layout, the scheduler must do
  something that changes the condition — retract/preempt the largest resident
  request, or spill its KV — rather than re-deciding the same way forever;
* let rung 1 run **before** the affordability test, not only inside the swap.
  It returned 120-444 MiB per flip in this very run, and the shortfall was
  13 MiB. The reclaim currently sits *after* the decision that needed it;
* admission control: refuse to admit a request whose resident KV would make the
  next flip unaffordable, which is a boot-time computable bound.

**Consequence for section 8's result:** the 500000 row stands as measured — the
corridor held for the whole window and the purity, graph, flip and accept-len
axes were all green — but the run ENDED in this livelock rather than being
stopped cleanly, and no >=60-minute green run was achieved. Do not read section
8 as an acceptance pass.

## 10. USER NOTE (2026-08-10, during the run): the surplus free VRAM is a DEFECT, not slack

The user read section 8's minima (1181 / 3676 / 1527) and pointed out what they
leave unused: roughly 181 + 2676 + 527 MB above a 1000 MB reading, 157 + 2652 +
503 = **3312 MiB above the 1024 MiB floor**. Deferred by the user ("darum
koennen wir uns spaeter kuemmern"), but explicitly logged as owed work.

This is the OTHER half of the corridor law, which this chain has consistently
treated as one-sided. `[[vram-korridor-regel]]` says both: never breach 1024
**and** keep the cards as full as possible, free near 1024 and not more. Every
handoff in this corpus has reported the corridor as a floor to survive. A card
sitting 2.6 GiB above it is failing the rule just as a card at 900 MiB is.

**The structure of the surplus is the whole point, and it is not evenly spread:**

| card | margin above 1024 | share of surplus |
|---|---|---|
| rank0 (5090) | **2652 MiB** | 80 % |
| rank2 (3080) | 503 MiB | 15 % |
| rank1 (3080) | 157 MiB | 5 % |

**Raising the pool cannot spend it.** The pool is global and draws from every
card in proportion to that card's share, so the next token off the pool costs
rank 1 — the card with 157 MiB — at 9.10 MiB per 1000 tokens. Rank 0's 2652 MiB
is unreachable from the pool knob alone; it is reachable only by moving SHARE
onto rank 0, i.e. the joint search over (pp layers, `--phase-flip-tp-vector`,
`SGLANG_UNEVEN_TOKEN_VECTOR`) already listed as next move 3 in section 8.

So the user's note and that item are the same work, and section 4's calibrated
model (727 MiB/layer + fixed 3304/874/3299, accurate to ~200 MiB over four
boots) makes it a desk exercise before it costs a boot. Rough size of the
prize: if the surplus were levelled so all three cards sat near 1024, rank 1
would gain ~1100 MiB of the 3312, which at 9.10 MiB per 1000 tokens is on the
order of **+120000 tokens** — which is, to within the noise of this estimate,
exactly the gap between the measured 500000 and the 600000 acceptance target.

**Do not treat this as a tuning nicety.** It is the most likely route to the
acceptance number, and it is cheaper than spill rungs 2-3.
