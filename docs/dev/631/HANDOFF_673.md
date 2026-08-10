# HANDOFF 673 — #656 / #631 Route A, successor 30

Predecessor: HANDOFF_672 (successor 29). Read its section 1 for the payload
correction (a memory-usage delta is not a payload); this file does not repeat
it, but it applies the same law twice more below.

---

## 0. THE ONE-LINE VERSION

**CorridorGuard is wired at the seam and proven on metal in both directions:
an allocation that would cross the floor now spills FIRST (286 MiB returned to
the driver, re-probed), and when nothing can fund it the flip is refused and
abandoned cleanly instead of dying inside the no-return region.** Getting
there cost one self-inflicted wedge, and two of the three payload rungs the
briefing ordered turned out to be worth nothing or to need no code at all.

---

## 1. ERRORS FIRST

### 1a. The arming watermark and the corridor law were the same number

`CorridorGuard` judged its REFUSAL against `floor_mib` — the watermark it
arms at. Those answer different questions, and with the floor raised for a
proof run the gate refused seams the 1024 MiB law permitted easily:

    want 726 MiB, free 2306 -> 2306 MiB, reclaimed 0 MiB, floor 1600

2306 − 726 = 1580, which is 556 MiB **above** the law. Refused anyway.

On the `tp->pp` leg an over-eager refusal is survivable: the instance stays in
TP, decode keeps running, prefill defers. On `pp->tp` it is a **deadlock**.
Strict purity forbids decode in PP, so a persistently refused `pp->tp` means
decode never runs again — and nothing the PP phase holds can free the memory
that would end the refusal. It does not self-clear; waiting makes it worse.

Measured: **411 abandons, 0 requests completed in 6 minutes, `/health` 503**,
while all three ranks were alive and logging normal, clean, unanimous
abandons. Nothing in the flip machinery was broken. The gate was.

Fixed: `floor_mib` is a policy target (arm here, free to here);
`law_floor_mib` is the corridor law and the only thing a refusal may cite.
They coincide by default. **A gate may be made to work earlier; it may never
be made to refuse what the law allows.**

The repeated-`pp->tp`-refusal case is now named explicitly in the log at the
1st, 10th and every 100th occurrence, because the gate cannot fix that state
and a wall of identical abandons is not a diagnosis.

### 1b. Idle mamba/GDN slots are worth ZERO bytes, and they were rung 3a

The briefing ordered the TP-phase rungs as (a) idle mamba/GDN slot states,
(b) arena slack, (c) KV-to-host. **(a) is worth nothing as built** and should
not be attempted next:

* `MambaPool.__init__` allocates every state tensor with plain `torch.zeros`
  on device (`mem_cache/memory_pool.py:580-609`) — the caching allocator, not
  a `KvVmmArena`. There is no VMM path for the mamba pool.
* `torch_memory_saver` is **off** on this rig, so there is no second route.
* `MambaSlotAllocator.free` pushes an index onto a free list
  (`mem_cache/allocator/mamba.py:30`). Freeing a slot frees an **index**.

An index is not a payload. This is HANDOFF_672's ledger law a second time:
price a spill from bytes the payload exclusively owns *and can return to the
driver*. The tree already says so at
`model_executor/model_runner_kv_cache_mixin.py:2134-2140` — "a vacated slot is
reused by another session, not returned to the KV pool".

What IS reusable there: the idle-slot **detection** (`gdn_slot_runtime.py`)
and the **blob round-trip** (`export_state_blob`). The release path does not
exist, and bolting it on means re-hosting the pool on a VMM arena *and*
compacting slots first, because idle slots are only contiguous by accident.

### 1c. The 4-layer PP stage-cut grid does not exist

The coordinator asked me to price relaxing the PP cut off the
`full_attention_interval` grid, falsifier first. **There is nothing to
relax.** `derive_pp_layer_split` (`distributed/utils.py:1544-1558`) contains
no literal 4 and no interval rounding; it rounds the *full-attention count*
and clamps the layer target into the window between two full-attn positions —
four legal boundaries per group. Effective granularity is **2 layers, not 4**.
`--pp-layer-ratio` bypasses the planner entirely: any triple summing to 64
boots. Driven on the real functions, 140 of 245 reachable splits are already
non-aligned, and a deliberately mid-group `29,19,16` produces full map
`[7,5,4]`, identical to `[28,20,16]`, with `validate_layer_map` passing.

Fail-loud, not fail-silent: every global→local mapping is a **dict built from
a stage-filtered list** (`memory_pool.py:3628`, `:1415`), never `//` or `%` on
a global layer id. The only guard that exists refuses a stage with zero
full-attention layers, which is the thing that actually matters. Three
alignment-sensitive sites exist (MUSA, NPU, a SWA path) and none is reachable
for this model.

`15,9,8` snapped by **two** layers (30→32), not four — banker's rounding in
`round(n_full·cum/total)` on the `cum ≡ 3 (mod 4)` residue. The "quantum"
claim in HANDOFF_665 §5 was already retracted at
`PROD_BRINGUP_BENCH.md:2604-2606`; this is the fifth capacity closure in the
chain that did not survive contact.

**Do not build it.** The measured economics are unchanged: each layer moved
onto the 5090 destroys ~308 MiB of total corridor
(`PROD_BRINGUP_BENCH.md:2680-2683`). The named lever is the one at
`:2686-2688` — the layer split and the token vector are **independent knobs
currently pinned proportional to one another**, decoupling needs no new code,
and has never been tried. That is the cheap experiment, not a planner change.

### 1d. The gate has no provider at all for the pp->tp seam

The draft carrier is the only registered provider, and on `pp->tp` the
instance is *in PP*, where the drafter is already spilled by design. Every
`pp->tp` arm therefore reads `reclaimed 0 MiB from [nothing]`. It clears only
because the law is satisfied anyway.

So the gate is currently one-armed: it can fund `tp->pp` and nothing else.
The `pp->tp` seam is exactly where the deadlock of 1a lives, and it is the
direction that has no answer yet. That is what item 15c (kvso over the host
tier) is for.

---

## 2. WHAT IS BUILT AND PROVEN

`CorridorGuard` is consulted in `_execute`, before `_staging_affordable` and
before the wave loop. **Not** at `commit_range`: that is inside the no-return
region, there is no try/except on the path by design, and a raise there kills
the rank. The verdict travels as a string into `too_small`, which already
rides the `_collective_min` that makes the abandon unanimous.

Ordering is load-bearing and pinned by test: the gate runs first because its
providers hand pages to the **driver**, so its reclaim is money the
affordability check then sees.

Item 16 is folded in as a **tier above the cost**: `RELIEF_REBALANCE`,
`RELIEF_PARK`, `RELIEF_HOST`. Cost still orders within a tier; it may not
promote a host spill ahead of a rebalance. The host tier is gated on a
**fleet** predicate read over NVML — every card at the floor, or it stays
shut. NVML and not a collective, deliberately: this runs where ranks can
disagree, and a collective there deadlocks. No fleet probe also means the host
tier stays shut, because item 16 is a permission that must be proven.

### The metal proof

Arming floor 1600, law 1024, depth=draft, POLICY=auto, strict purity, MTP and
decode/verify/draft graphs on:

    CORRIDOR-GUARD cleared: want 820 MiB, free 2394 -> 2680 MiB,
                            reclaimed 286 MiB from [draft-weights]  (tp_to_pp)

The allocation that would have crossed the floor spilled first, and **NVML's
own free column rose by the payload** before it proceeded — re-probed, not
taken from the provider's return value. 26 arms, 0 refusals, 0 abandons,
health 200, corridor HELD.

The refusal path is proven by the run that found bug 1a: 197 refusals, 330
clean unanimous abandons, corridor HELD, **no breach**, every rank alive. The
gate declines rather than dying inside the no-return region. That is a
can-fail proof in both directions and it is what the whole step rested on.

---

## 3. ITEM 16, MEASURED: THE CARDS ARE NOT EVENLY FILLED

The sampler now books the per-card free-headroom **spread** per sample, next
to the minimum, because holding the floor everywhere says nothing about
whether one card carried the pressure alone.

| run | min free (0/1/2) | spread mean | spread worst |
|---|---|---|---|
| baseline, floor 1024 | 1720 / 3768 / 1960 | 2901 | 3551 |
| can-fail, floor 1600 | 2024 / 3984 / 2244 | 2901 | 4013 |

The 5090 never drops below ~3.8 GiB free while both 3080s bind near 1.7-2.2
GiB. That is the state item 16 forbids, it is stable across runs, and it now
has a number instead of an impression. **Roughly 2.9 GiB of 5090 headroom is
the prize** a levelling rebalance is playing for.

**Not capacity evidence**: peak occupancy 199453/500000 = **40%**, below the
50% bar. The corridor numbers above describe a lightly loaded instance.

---

## 4. THE NEXT RUNG IS THE ARENA TAIL, AND IT IS MEASURED

Re-measured on THIS boot from the existing `TP stack built` line, which
reports both layouts:

| rank | arena / pp | tp | **tail** |
|---|---|---|---|
| PP0 (5090) | 13482.18 | 13163.45 | **318.7 MiB** |
| PP1 | 8144.00 | 7923.95 | **220.1 MiB** |
| PP2 | 9114.95 | 7923.95 | **1191.0 MiB** |

This **confirms the 319/220/1191 record and refutes the 1773/0/1191 one** —
the two disagreed and the disagreement is now closed by measurement.

Why it is the right next rung: the arena is sized `max(pp, tp)` and `pp` is
the max on all three ranks, so the tail is committed-but-unused **in TP** —
the phase that now binds on every card. It needs **no host round trip** (the
content is rewritten by the refill that already runs), which makes it the
cheapest provider in the system. On one 3080 it is four times the drafter.

The mechanism is fully scouted:

* `allocate_arena` (`weights_arena.py:233`) is `torch.empty`, so the tail is
  **committed, not merely reserved**. Substituting a `KvVmmArena` reservation
  is the same move `allocate_carrier_tensor` already makes for the drafter.
* The active layout always occupies a **prefix** `[0, layout_X.total_bytes)`,
  so the tail above it is genuinely addressable by nothing in that phase.
* `PhaseFlipStacks.refill(direction)` (`phase_flip_boot.py:332`) is the hook:
  on `PP_TO_TP` decommit the tail **after** the refill+verify (the `restore=`
  arm touches the PP layout and must still be backed); on `TP_TO_PP` commit
  the tail **before** the refill.
* **The `TP_TO_PP` commit is an allocation inside the no-return region** and
  must be added to `_staging_bytes` the way `_draft_restore_bytes` was, or it
  recreates the exact `cuMemCreate` death this whole gate exists to prevent.

Suggested depth ladder: insert `DEPTH_ARENA_TAIL = 3` and push
`DEPTH_DRAFT_GRAPHS` to 4, rather than renumbering `draft`, so existing
integer evidence keeps its meaning.

---

## 5. NOT DONE

* Rung 3b (arena tail) — scouted and measured above, not built.
* A provider for the `pp->tp` seam (1d). This is item 15c, kvso over the host
  tier, and it is also the only cure for the 1a deadlock class.
* The TP rebalance provider. `distributed/corridor_vector.solve_corridor_vector`
  already solves the token vector against per-card corridor capacity — the
  stage-1 lever exists and is unwired.
* Occupancy ≥80% booking of the resident working set (step 3). Every corridor
  number in this handoff is a 40%-occupancy reading.
* Item-8 arms, threshold-purity arm, YaRN >262k leg, final all-axes acceptance
  under real router-30099 agent traffic.

## 6. IF YOU DO ONE THING

Build the **arena tail** (section 4). It is the only one of the three ordered
rungs that is committed bytes idle in the phase that binds, it is measured, it
needs no host round trip, and the entire mechanism is already in the tree
twice. Price the `TP_TO_PP` commit into `_staging_bytes` before you wire it —
that is the step that turns this rung from a win into an outage.

Do **not** spend a shift on the PP layer grid (1c) or on idle mamba slots
(1b). Both are already answered.

## 7. PROCESS

The can-fail proof is what found bug 1a. A gate that has never been observed
to FIRE is indistinguishable from a gate that is never reached, and the
cheapest way to make it fire on real metal was to raise its arming floor above
the resting headroom. That instrument is now `SGLANG_CORRIDOR_FLOOR_MIB`, it
logs loudly that it is not the law, and the fix in 1a is what makes it safe to
use. Every future provider should get the same treatment before it is believed.
