# DESIGN 584 -- one resource authority: cut, ledger, movement

Status: DESIGN, no code. Branch `feat/exact-vram-ledger` (design only; the
implementation gets its own branch per slice).

This document is written against the mandate plus six addenda. Addenda 3 and 4
are recorded as R6, R7 and R9 because each names an ERROR CLASS the design must
be structurally unable to commit -- not a behaviour it should prefer. That
distinction drives the falsifiers: several are deliberately PAIRS, because a
single case is passed by exactly the static implementation being excluded.

Requirements are numbered in the order they were established, not in reading
order (R9 arrived with addendum 4 and sits before R8 below). References
elsewhere use the numbers, so they are not renumbered.

**Provenance note.** The mandate and addenda 1-2 are summarised here from the
coordinator's messages, not quoted. Addenda 3 and 4 are recorded close to their
wording. Every mechanism this document builds on was checked in the tree before
being relied on; where a check contradicted the brief, the finding is in §5 and
the design follows the tree.

---

## 1. What this unifies, and what "one authority" means

Today the rig decides resource questions in several places that do not share a
denominator:

| Machinery | Decides | Sees |
|---|---|---|
| `mem_ledger` (#582, this branch) | per-card VRAM arithmetic | one boot's config |
| solver `key_solver` (#272) | co-resident budgets | instance specs |
| `regime_act` / `regime_stages` (#363) | stage/regime actuation | live traffic |
| `kv_ladder_auto` (#297) | KV vector at phase boundary | phase edges |
| `kv_pressure_ladder` (#272 runtime) | pool pressure response | pool occupancy |
| resharder / live handover (#261) | session movement | handover requests |
| tier registry (#407), hibernate (#89), kvso | where bytes live | their own tier |
| autoboot (#539) | what starts at all | recipe |

Each is correct in its own scope. The defect is that no one of them can answer
"given everything running right now and what is about to run, what is the best
whole-rig configuration, and is it worth moving to it?" -- so the answer today
is a human choosing a `RESERVE` vector.

**One authority** means: a single planner owns the *decision*, the ledger owns
the *arithmetic*, and every machinery above becomes an *actuator* the planner
drives or a *sensor* it reads. It does not mean rewriting them. A slice that
reimplements a carrier instead of driving it has failed the mandate.

---

## 2. Requirements

Requirements R1-R11. Each carries a falsifier: a concrete situation that must produce a
specific observable, and which the current design would fail if the
requirement were dropped.

### R1 -- The ledger is the only VRAM arithmetic

Every VRAM decision anywhere resolves through `mem_ledger` terms. No component
keeps a private constant, a private fraction, or a private reserve.

**This extends to HOST RAM where the stack parks things there.** Parked graph
stages (R9 iv), parked expert tiers and spilled KV all consume host bytes that
nothing currently accounts for. A stack that parks eight stages without a term
for their host footprint is running the same unledgered-demand defect #582
removed from VRAM, one tier down.

*Falsifier:* grep gate -- a new module-level MiB constant used in memory
arithmetic outside `mem_ledger` fails CI. (The three constants #582 already
replaced, 1280/1536/600, are the template for what must not recur.)

*Falsifier R1b.* Parking N pre-captured stages must move a HOST-RAM ledger term
by N x the per-stage footprint. A ledger that reports the same host total
before and after parking is not accounting for them.

### R2 -- Demand is computed, headroom is the user's

Carried forward from #582 and non-negotiable: `card_total = user_reserve +
computed_demand + kv_pool`, user reserve default 1024 MiB, nothing internal
ever funded from it.

*Falsifier:* already implemented and tested (#582); the gate is that no #584
slice reintroduces a path that sizes anything from the reserve.

### R3 -- Coverage comes from structure, not per-case code

A new tenant, a new lane, a new card, or a new phase must be handled by
declaring into existing interfaces, never by adding a branch. The #582 tenant
declaration (`declare_tenant_terms`, loud on omission) is the pattern.

*Falsifier:* adding a hypothetical fourth card class, or a new tenant type,
must require zero edits to planner logic -- only a declaration. A design review
that cannot demonstrate this on paper fails the slice.

### R4 -- The objective is JOINT and phase-weighted

*Answered by the user, 2026-08-05; no longer `[CONFIRM]`.*

The planner does not seek *a* configuration that fits; it seeks the one that
maximises the objective subject to fitting. Fitting is a constraint, the
objective is the goal.

The objective is **joint**: maximum compute AND maximum VRAM
throughput/latency, at all times. These are two axes, not one scalar, and the
design must not collapse them into a single number chosen on the user's behalf.

When the two disagree, **the running task's phase decides dominance**:

| Phase | Bound by | Dominant axis |
|---|---|---|
| decode | memory bandwidth | VRAM throughput -- bandwidth *is* throughput here |
| prefill | compute | compute |

So the scoring function is a phase-weighted combination whose weights come from
the same regime signal R7 uses, not a constant. A rig serving a decode-heavy
mix and one serving a prefill-heavy mix are answering different questions, and
a single fixed weighting would be wrong on at least one of them.

*Falsifier F4.* Two candidates, A compute-superior and B bandwidth-superior.
Under `REGIME_DECODE_HEAVY` the planner must choose B; under
`REGIME_PREFILL_HEAVY`, A. A planner with a fixed scalar objective passes
exactly one of these, which is why the falsifier is again the pair.

*Falsifier F4b.* A configuration that fits comfortably but is objective-inferior
to a tighter one must lose. If the planner ever prefers slack for its own sake,
the constraint has been confused with the goal.

### R5 -- Movement has a cost and the cost is in the objective

Evacuating to RAM, resharding, re-cutting and rebuilding graphs all cost time.
The planner compares the *discounted* benefit of a new configuration against
the movement cost of reaching it, over an expected horizon.

**Graph cost is TWO terms, never one** (see R9's correction). Collapsing them
is what makes a cheap flip look expensive:

| Term | When paid | Observed | Nature |
|---|---|---|---|
| graph **capture** | first entry into a stage, once | 3-6 s | one-time investment, schedulable during low load |
| graph **restore** | every flip to a parked, already-captured stage | 40-85 ms | per-flip, #464 improvement pending |

A cost model with a single "graph cost" line is wrong in one direction or the
other by roughly two orders of magnitude.

*Falsifier F5.* A 2% throughput gain that costs a 40 s reshard must be REFUSED
at short horizon and ACCEPTED at long horizon, with the horizon an explicit
input, not a constant.

*Falsifier F5b.* The same gain against a 60 ms restore must be ACCEPTED even at
short horizon. A model that refuses it has priced a restore as a capture.

### R6 -- (Addendum 3a) A state change re-solves the CUT, not only the pools

**Requirement.** On a state change the planner must recompute the
configuration itself -- layer/TP split, draft placement, KV/token vectors,
phase layouts -- under the new usage profile (compute demand, VRAM amount, and
VRAM throughput demand of each remaining task). Squeezing pools inside the
existing layout is one *candidate*, not the search space.

The planner may still CHOOSE pool-only adjustment; that is a cost calculation
under R5. What the design must exclude is the planner being structurally
*unable* to consider a re-cut.

**Carriers (verified present):**
- `#297` phase-boundary KV resharding -- `managers/kv_ladder_auto.py`
- `#485` phase matrix -- `uneven_perf.py`
- `#363` regime controller -- `managers/regime_act.py`, `regime_stages.py`
- `#261` resharder / live session handover -- `managers/scheduler.py:5063`

**Falsifier F6.** Construct a state change (a tenant departs, freeing a card's
worth of VRAM and compute) where:
- pool-only adjustment is *admissible* (everything still fits), and
- a re-cut (different TP split / draft placement) is strictly better under the
  R4 objective by a margin exceeding its movement cost under R5.

The planner MUST emit the re-cut. A planner that emits the pool-only
adjustment, or that never enumerated the re-cut, fails. The test asserts on the
*candidate set* as well as the choice, because a planner that picked correctly
by luck from a set of one is not implementing this.

**Second falsifier F6b.** The mirror case: a state change where the re-cut is
better in steady state but its movement cost exceeds the benefit over the
expected horizon. The planner must emit pool-only, and must record *why* --
naming the re-cut it rejected and the cost that rejected it. Silence here is
indistinguishable from not having considered it.

### R7 -- (Addendum 3b) Evacuation heat ordering is PHASE-DEPENDENT

**Requirement.** The movement plan's hot/cold ranking takes the current and
imminent phase as an input. Heat is not a static property of an item class.

Concretely: during prefill, decode-owned items -- decode CUDA graphs,
draft/spec state, decode workspaces -- are COLD and evacuate first, so nothing
hot is displaced. During decode the ranking inverts: prefill-owned scratch and
prefill graphs become the cold set.

**Carrier (verified present):** `managers/regime_stages.py` already classifies
`REGIME_PREFILL_HEAVY` / `REGIME_DECODE_HEAVY`. The ranking consumes that
signal; it does not introduce a second phase detector. A second detector would
be a new source of disagreement about what phase it is, which is the same
defect class as a second VRAM constant.

**Imminence matters, not only the current phase.** Evacuating decode graphs at
the *end* of a prefill burst is worse than useless -- they are about to be hot.
The ranking therefore takes `(current_phase, imminent_phase)`, where imminence
comes from the regime controller's own lookahead rather than from a new
predictor.

**Falsifier F7.** A prefill-burst event must rank decode CUDA graphs BELOW KV
in eviction order (i.e. graphs evacuate first, KV last). The decode-burst
mirror must rank them above. A static ranking passes exactly one of these two
and therefore fails the pair -- which is why the falsifier is the pair, never a
single case.

**Falsifier F7b.** At a prefill burst whose imminent successor is decode, the
plan must NOT evacuate decode graphs. This separates "phase-dependent" from
"reacts to the instantaneous phase", which is a different and worse thing.

### R9 -- (Addendum 4) Flip granularity is a PRICED AXIS, and an enforcer exists

*User requirement, 2026-08-05.*

**The requirement has two halves and both are load-bearing.**

**(i) Every prefill / draft / decode(verify) cut is itself a load change.** The
best layout for a prefill tick is not the best layout for a verify tick. The
planner must be able to use a different layout per load, and an ENFORCER must
actually cause that different layout to happen. A design where the planner
*could* choose differently but nothing actuates is not this requirement; it is
the #578 bug (see slice 0) written into the architecture.

**(ii) Tick-level versus regime-level flipping is PRICED PER KNOB, not decided
by a blanket rule.** For each control knob the design states its flip cost and
classifies it:

**CAPTURED GRAPHS ARE SPILLABLE. Recapture is a ONE-TIME cost per layout
stage, never a per-flip tax.** *(User correction, 2026-08-05 -- pinned here
because it has been forgotten more than once.)*

The stack can pre-capture N stages and park the inactive stages' graph state
**full-captured in host RAM**. Nothing has to be rebuilt to flip back to a
stage that was already captured. The carriers are present in tree, checked:

- `#93` physical aliasing / remap keeps virtual addresses stable across the
  park (`model_executor/offload_register.py`, `offload_movement.py`,
  `input_buffers.py`, `runtime_context.py`) -- this is what makes a parked
  capture still valid on return;
- `#286` offload register **already lists `graph_rungs` as a parkable item
  class** (`offload_register.py:17` "cold capture rungs of the K-/algo ladder",
  and in the class table at `:114`);
- `#89` hibernate machinery for the host-side staging.

So a flip between PRE-CAPTURED stages costs only the graph-state RESTORE:
**observed band 40-85 ms**, with `#464` targeting ~3 driver calls per contiguous
VA region. (`#464` has no marker in `python/sglang/srt/` -- it is a target, not
a mechanism. The design prices restore at the observed 40-85 ms and treats any
#464 gain as upside.)

**Revised classification.** "Costs a recapture" disqualifies a knob from fast
flipping ONLY when the target stage has never been captured:

| Class | Criterion | Flips at | Examples |
|---|---|---|---|
| **tick-flippable** | not graph-addressed AND zero data movement | every forward | #439 cold-expert compute assignment (verified VRAM-neutral: `expert_heat_migration.py:31` calls same-size-before-and-after "the #439 sizing latch's invariant"); candidate: token vector per prefill batch |
| **restore-flippable** | graph-addressed, but target stage ALREADY captured and parked | regime boundary, possibly coarse tick | any pre-captured stage of the #363 stair -- priced at restore (40-85 ms), NOT at capture |
| **capture-bound** | target stage NEVER captured | one-time, scheduled deliberately | first entry into a newly solved stage (3-6 s, `DESIGN_140`) |
| **movement-bound** | moves data | regime boundary, with hysteresis | KV resharding (#297); TP re-cut (#261) |

A knob is tick-flippable exactly when flipping it costs neither a recapture nor
a byte moved. The correction is that "graph-addressed" no longer implies
"expensive" -- it implies "expensive **once**".

**(iii) The planner pre-captures its own stair.** The stages the planner solves
(#363) are pre-captured at boot or lazily on first entry, then parked. First
entry is a **priced one-time investment the planner schedules deliberately**
(e.g. during low load), never a tax paid at the moment a flip is wanted. A
design that pays capture at flip time has turned a startup cost into a serving
cost.

**(iv) Holding N stages is nearly VRAM-free while parked** -- physical pages are
returned via the VMM dial (#330) and only the VA reservation remains. But it is
NOT free in host RAM, and therefore: **the ledger must carry the parked stages'
HOST-RAM footprint as a term** (R1 applies to host RAM here, not only VRAM). A
stack that parks eight stages and never accounts for their host bytes is
running the same unledgered-demand defect #582 removed from VRAM.

*Falsifier F9d.* A flip to an ALREADY-CAPTURED stage that triggers a recapture
is a defect, and the test asserts on the absence of recapture, not on elapsed
time (a fast machine could hide a recapture inside a generous threshold).

*Falsifier F9e.* A cost model that prices a flip to an already-captured stage at
CAPTURE cost must fail its test. This is the modelling error, distinct from the
runtime one in F9d, and it is the more dangerous of the two: it never
misbehaves, it just silently refuses flips that were actually cheap -- so the
gain is never observed and nothing surfaces the mistake.

**Known instance of exactly this error, in tree today.** `registry/rungs.py:93-95`
prices the `WARM` rung at "3-6 s" with basis *"graph recapture / weight flip"*,
i.e. it assumes promotion pays a recapture. Under this correction a WARM rung
whose stage is captured-and-parked should price at restore (40-85 ms), roughly
two orders of magnitude cheaper. The ladder is not wrong for a never-captured
stage; it is wrong as a blanket price. Slice 4b must split that entry the same
way R5 splits its graph terms, or the planner will inherit the very tax this
correction removes.

**Within a tick**, where the layout is fixed by definition, the answer is
overlap and behaviour adaptation rather than layout change: #128/#199 collective
overlap, #125 prefetch, #274 phase pairing, #156 adaptive draft. These are not
lesser substitutes for flipping; they are the correct tool at that timescale,
because at tick granularity the flip cost of anything structural exceeds a
tick.

**Every knob must carry its classification and its justification by cost.** A
knob placed on the regime side without a stated cost is an unpriced assumption,
which is the same defect class as an unledgered VRAM term (R1).

*Falsifier F9.* A knob that is not graph-addressed and needs zero data movement
must be flipped PER TICK by the planner. Concretely: give the planner a
workload whose optimal #439 cold-expert assignment differs between consecutive
prefill and verify ticks; it must emit a different assignment on each. A
planner that holds it until a regime boundary has mis-classified a zero-cost
knob as expensive -- and that failure is silent and permanent, because a
wrongly-regime-classified knob simply never demonstrates the gain it could have
made.

*Falsifier F9b.* The mirror, guarding against the opposite error: a
graph-addressed knob must NOT flip per tick. A planner that flips a knob
costing a 3-6 s recapture at tick granularity would spend the entire tick
budget on recapture and serve nothing.

*Falsifier F9c (enforcement, not intent).* After slice 0, a regime change must
produce an OBSERVABLE layout difference -- not merely a planner decision logged.
The test asserts on the actuated state, because #578 is precisely the case
where the decision existed and the actuation did not.

### R10 -- (Addendum 5) WEIGHT residency is phase-dependent, to layer-slice granularity

*User requirement, 2026-08-05.* R7 made **evacuation ordering** phase-dependent.
R10 says the same of **weights themselves**, and at a finer grain than "the
model": individual layers, and slices of layers.

**The named first-class case: the draft/target ping-pong.** During the DRAFT
phase the target's layers or layer slices may be spilled -- to ANY tier, for
ANY reason, including simply that the video and sound pipelines want the VRAM
and this is the only way everything is held at once. During PREFILL/VERIFY the
draft is spilled and the target's layers return. The two models take turns
owning VRAM because they take turns computing.

**(a) Every conceivable combination must be EXPRESSIBLE.** Coverage comes from
structure, not from enumerated cases (R3 applied to residency). The unit is
"(layer or slice) x (tier) x (phase)", and the planner must be able to express
any assignment over that product. A design that enumerates "draft spilled" and
"target spilled" as two supported modes has already failed: the third thing
somebody wants is a half-spilled target during a long prefill while the
diffusion lane holds a card.

**(b) The planner computes automatically WHAT moves WHEN.** This is a
movement-bound decision on R9's priced flip axis: movement cost against the
value of the VRAM held. It is never a structural "cannot". If the planner
declines to spill a target slice, that must be because the arithmetic said so
and the reason is recoverable under R8.

**(c) It is automatically EXECUTED, and the freed VRAM is automatically
REFILLED.** Freeing bytes and leaving them idle is a DEFECT, not a neutral
outcome -- it is the "surplus sits idle" failure #582 removed from the boot
path, reappearing at runtime. Whatever the solver yields takes the space: KV,
a tenant, a lane. An empty hole means the refill step did not run.

**Overlap is the reason this is cheap, and it must be PRICED.**

While the draft computes, the target's layers are idle -- so their reload can
hide behind draft compute, and symmetrically the draft's reload hides behind
target compute. The movement-cost model must therefore charge the **unhidden
remainder**, not raw bytes over bandwidth:

    effective_cost = max(0, transfer_time - overlappable_compute_time)

Charging raw transfer time would price every ping-pong as unaffordable and the
planner would never take a swap that is in fact nearly free. This is the same
error shape as pricing a restore at capture cost (F9e): a cost model that is
wrong upward simply never exercises the mechanism, and nothing surfaces it.

**Carriers -- verified, with an honest split.**

*Present in tree:* the #77/#123/#125 weight-streaming machinery is real and
does H2D fetch of weight rows from a host tier --
`layers/moe/expert_offload.py`, `expert_heat_migration.py`,
`fused_moe_triton/layer.py`, `cold_tier_fetch.py` (host DRAM segments,
identity-keyed by card UUID and PCI BDF, zero-copy peer views, the fetch path
issuing `copy_` over the rank's own PCIe link), plus
`model_executor/offload_register.py`, `short_term_offload_register.py`,
`offload_movement.py`.

*NOT present:* that machinery is **expert-granular**. Generalising it to
arbitrary layer / layer-slice granularity is the claim, and I did not find
layer-granular streaming in tree (`offload_movement.py` has no per-layer
addressing). Slice 5 therefore EXTENDS this carrier rather than merely driving
it -- the same honest posture as #553. The double-buffered
prefetch-behind-previous-layer-compute structure is likewise asserted in the
addendum but not something I located as a general mechanism; treat its
existence as a slice-5 finding, not a premise.

*Falsifier F10.* A co-tenant demand that fits ONLY via draft-phase target-slice
spill must be **planned AND executed**, not refused. A planner that declines it
for lack of expressiveness fails (a); one that plans it and does not execute
fails (c).

*Falsifier F10b.* The VERIFY phase following such a spill must produce
**byte-identical** output to a never-spilled reference run. This is the
full-captured doctrine applied to weights: a residency decision is a placement
decision, never a numerics decision. Any bit difference means spilling changed
the computation, which would make the whole mechanism unusable under the
determinism arm.

*Falsifier F10c.* After a spill frees VRAM, a subsequent ledger read must show
the freed bytes CLAIMED (KV pool, tenant, or lane). Bytes free and unassigned
is a defect per (c), and the test asserts on the claim, not on the free.

### R11 -- (Addendum 6) Necessity is COMPUTED per step; the rest moves, coldest first, only when free

*User requirement, 2026-08-05:* "It must effectively be COMPUTED what is
actually necessary in each step. The rest is moved elsewhere by a
coldest-to-hottest rule, when necessary, and when it costs no performance
(redistribution within the vram/ram tier should only take milliseconds)."

This is the rule that ties R6, R7, R9 and R10 together: they say what MAY move
and what it costs; R11 says what must STAY.

**(a) Necessity calculus per step.** For every step -- phase transitions and
finer steps within them -- the planner computes that step's actual minimal
working set. Everything outside it is a movement CANDIDATE. Necessity is
computed from the step, never inherited from a static class label: "weights are
necessary" is not a fact, it is a fact about a step, and R10 exists precisely
because a target layer is unnecessary during a draft step.

*Falsifier F11a.* Two steps whose minimal working sets differ must yield
different candidate sets. A planner that returns the same candidates for a
draft step and a verify step is not computing necessity, it is reading a class
table.

**(b) Eviction order: coldest to hottest.** The movement plan evicts in
ascending heat, where heat is R7's PHASE-DEPENDENT ordering. R7 says heat is
not static; R11 says what to do with the ranking once you have it. The two are
one mechanism: R7 supplies the order, R11 consumes it coldest-first.

*Falsifier F11b.* Given a heat ranking, the emitted eviction sequence must be
its ascending order. Any inversion -- evicting a hotter item while a colder one
remains resident -- is a defect, and the test asserts on the sequence, not on
the final residency set (two different orders can reach the same end state,
and only one of them is free).

**(c) TWO gates, both required.** A movement happens only if BOTH hold:

| Gate | Meaning | Fails when |
|---|---|---|
| **need-driven** | a trigger exists: demand, phase change, tenant arrival | eager reshuffling "to be tidy" |
| **performance-neutral** | hidden behind compute, or placed in an idle gap | the unhidden remainder is non-zero and nothing forces it |

Performance-neutrality is exactly R10's overlap pricing:
`effective_cost = max(0, transfer_time - overlappable_compute_time)`. A move
that cannot be hidden AND is not forced by need **does not happen**.

Note the asymmetry, because it matters: need without hiding is allowed (a
forced move pays its cost), and hiding without need is not (a free move is
still churn, and churn has second-order costs -- fragmentation, cache
disruption -- that the model does not price). So the gates are not symmetric
and must not be collapsed into a single score.

*Falsifier F11c.* A perfectly hideable movement with NO trigger must not be
emitted. This is the gate that a naive optimiser fails: it sees a free move
that marginally improves a metric and takes it, forever.

**(d) Latency expectation: milliseconds, not seconds.** The user expects
VRAM/RAM-tier redistribution to complete in MILLISECONDS. This is consistent
with what is already measured: the 40-85 ms graph-restore band (R9) and plain
PCIe transfer arithmetic.

**Any cost model that prices ordinary VRAM/RAM redistribution in SECONDS is
wrong until it justifies itself against a measurement.** This is the same
family as the blanket-recapture mispricing recorded in §5 (`registry/rungs.py`
charging 3-6 s for a promotion that costs a restore): a model that
over-prices a mechanism never exercises it, and never exercising it means the
over-price is never contradicted.

*Falsifier F11d.* A step-scoped movement whose transfer is fully overlappable
must be priced at ~0 effective cost. And the millisecond band itself must be
ASSERTED AGAINST MEASUREMENT, not against this document: slice 4c owns that
measurement, and until it lands the band is an expectation carried from the
restore observation, not an established input.

**(e) The gate is COMPARATIVE, never an absolute latency threshold.**
*(User refinement, 2026-08-05: "The proof that even a SECONDS window can be
worth it: when all VRAM and RAM are already occupied, you put it on disk
instead (or fetch from there) -- still faster than rebuilding and reloading the
whole model.")*

This corrects a misreading (d) invites. "Milliseconds" is the expected band for
the VRAM<->RAM tier; it is **not** a ceiling above which a move is refused. The
rule is:

    move iff effective_cost(move) < cost(best alternative)

and **the alternative set explicitly includes full re-materialization** -- model
reload, re-capture, re-prefill, i.e. minutes. A planner that rejects a
seconds-scale move "because seconds is too slow" while the alternative costs
minutes is wrong by construction, and it is wrong in the direction that looks
prudent, which is why it needs to be named.

**Per-tier bands, from the #407 registry, not one universal number:**

| Tier | Band | Source |
|---|---|---|
| VRAM <-> VRAM / RAM | milliseconds | 40-85 ms restore observation (R9), PCIe arithmetic |
| RAM <-> disk | seconds | #89 hibernate resume 8-14 s vs ~50 s cold start |

**Graceful degradation, never refusal:** VRAM full -> RAM; RAM also full ->
disk. As long as a deeper tier fits, the answer is a slower move, never a
refusal and never a teardown. Each tier carries its OWN measured band; the
ladder is the #305/#546 idle-park machinery and the #407 tier registry, and the
evidence line is already in tree -- `registry/rungs.py` prices the COLD rung at
"#89 resume 8-14 s at uneven TP=3 (DESIGN_201:1635) plus 3-6 s recapture"
against a boot it calls "effectively a boot".

*Falsifier F11e.* With VRAM **and** RAM saturated by co-tenants, an eviction
demand must produce a planned AND executed **disk** spill (and a later restore)
-- not a refusal, and not a rebuild.

*Falsifier F11f (structural, and the more important one).* The solver's
alternative-cost table must carry a **rebuild/re-materialize column**. Without
it F11e cannot even be expressed, because there is nothing to compare the
seconds-scale move against and "seconds" then looks unconditionally bad. That
column is a REQUIREMENT, not an optimisation, and a design review that finds it
missing fails the slice regardless of how the comparator behaves.

### R8 -- Every decision is explainable after the fact

Any configuration the planner chooses must be reconstructible: the candidates
considered, the objective values, the movement costs, and the horizon.

*Falsifier:* for any live configuration, a command prints why it is this and
not the runner-up. The #582 ledger render is the model -- itemization, not a
verdict.

---

## 3. Authority model

    sensors                 authority                 actuators
    -------                 ---------                 ---------
    regime controller  -->                       -->  #297 KV reshard
    ledger (#582)      -->   PLANNER              -->  #261 resharder
    probes #213/#513   -->   - enumerate cuts    -->  #363 regime actuation
    tier registry #407 -->   - price via ledger  -->  tier moves / kvso / #89
    tenant register    -->   - cost movement     -->  VMM dial #330
    live metrics       -->   - pick under R4/R5  -->  autoboot #539

The planner is the only component that *decides*. Everything left of it
reports; everything right of it executes. A carrier that both senses and
decides (the pressure ladder does today, within its scope) is refactored so
that its decision becomes a planner call -- its mechanism is kept, its autonomy
is not.

**The ledger is not in the loop as a gate; it is the denominator.** Candidates
are priced by it during enumeration, so an infeasible candidate is never
scored, never chosen, and never has to be rejected late.

---

## 4. Slice plan

Each slice is independently bootable and independently falsifiable. Ordering is
by "what unblocks the most" and by risk, cheapest proof first.

| # | Slice | Proves | Gate |
|---|---|---|---|
| **0** | **#578: bind the planner feed (`solve_fn`)** | **the authority has a runtime arm at all** | **hermetic: a non-empty stage table; F9c** |
| 1 | Boot path: ledger replaces the `RESERVE` vector | R1, R2 end to end on the production recipe | TICKET_582 gates (a)+(b) |
| 2 | Candidate enumeration (cuts, no movement) | R3, R6 candidate set | hermetic: F6 candidate-set assertion |
| 3 | Joint phase-weighted objective + movement cost | R4, R5 | hermetic: F4, F4b, F6, F6b |
| 4 | Phase-dependent heat ranking | R7 | hermetic: F7, F7b |
| 4b | Per-knob flip pricing; split capture vs restore; fix the WARM rung | R9 (i)-(iv), R5 two-term | hermetic: F9, F9b, F9d, F9e, F5b |
| 4c | Pre-capture + park the planner stair; host-RAM ledger term | R9 (iii), R1b | hermetic: F9d; GPU: measured restore band |
| 4d | Phase-dependent weight residency: express + price (no execution) | R10 (a), (b), overlap pricing | hermetic: F10 planning half |
| 5 | Actuation; BUILD elastic co-residency (#553); EXTEND weight streaming to layer/slice | R6 execution, tenant events, R10 (c) | GPU: live re-cut under load; F10 execution, F10b byte-identity, F10c refill |
| 6 | Explainability surface | R8 | hermetic + a live boot |

### Slice 0 is a hard prerequisite, and it is small

**Verified in code, not taken on report.** `regime_runtime.py:911` calls
`planner_candidates(server_args)` with `solve_fn` omitted, so
`regime_stages.py:383` takes the `solve_fn is None` branch, returns `[]`, and
the stage table permanently holds the booted stage alone. The docstring at
`regime_stages.py:357` states this outright: *"it is not broken, it is
unfed."*

Consequence for #584: the enforcer required by R9(i) **exists as code and
actuates never**. Every requirement in this document that ends in an actuated
layout change is unreachable until this seam is bound, so no later slice can be
honestly demonstrated first.

The work is named by that same docstring (tracked as #363/S8): `key_solver.solve`
already exists and is objective-aware, but its signature needs `plan_inputs`, a
base plan, per-rank budgets and `RigRates` -- i.e. a card probe and a measured
rate set -- and a `SolverAnswer` must be mapped back onto a `Stage`. It is
wiring plus a mapping, not a new solver, which is why it is slice 0 rather than
a project of its own.

Note the dependency this creates: binding the seam needs a card probe, which is
TICKET_582 gate (a). Slice 0 and slice 1 therefore share a GPU prerequisite and
should be scheduled into the same window.

Slice 1 is already most of the way there on this branch: #582 built the ledger
and the boot wiring behind `--enable-vram-ledger`. Slice 1 completes when
TICKET_582's GPU gates pass and the flag can default on.

**Slice 4c is the one addition that needs a card**, because the restore band
(40-85 ms) is a measured quantity and the whole flip economics rest on it. It
is a cheap measurement, not a full boot matrix: capture two stages, park one,
flip, time the restore.

**Slices 2-4b are desk-provable in full.** They are search, arithmetic,
ordering and pricing; none needs a card. This is deliberate -- slice 5 is the
only one that should ever wait on a GPU window.

---

## 5. Known gaps and risks

- **Elastic co-residency is a DESIGN, not a mechanism.** *Resolved:* task-list
  ID #551 is in-tree ID #553, and `docs/dev/ANALYSE_553_elastic_coresidence.md`
  is all that exists -- there is no code. Slice 5 therefore **BUILDS** it, as
  the planner's tenant-event actuator. It is not a carrier to be driven, and
  the slice plan must not be read as if it were: this is new construction, with
  the schedule that implies.
- **~~R4's objective is `[CONFIRM]`~~** -- *answered.* Joint compute + VRAM
  throughput, phase-weighted for dominance. See R4.
- **~~The enforcer~~** -- *promoted from unknown risk to slice 0 (#578).* This
  was the largest hidden hole in the first draft: every actuated requirement
  here depended on a seam that production never binds.
- **Re-cut cost is not currently measurable.** R5 needs a movement-cost model,
  and no component times a reshard today. Slice 3 must either measure it or
  declare it a calibrated term in the ledger's sense (measured once per rig,
  fingerprinted) -- not a literal. This applies to BOTH graph terms: "3-6 s"
  capture and "40-85 ms" restore are observed ranges, and a range is not a
  model. Restore in particular is the per-flip term, so its error enters every
  flip decision.
- **`registry/rungs.py` prices the WARM rung at recapture cost.** Named in R9:
  the LADDER entry at `:93-95` charges "3-6 s / graph recapture" for a
  promotion that, for a captured-and-parked stage, costs a 40-85 ms restore.
  Slice 4b must split it. Until then any planner reading that ladder inherits
  the per-flip tax this correction exists to remove.
- **`#464` is a target, not a mechanism.** No marker in `python/sglang/srt/`.
  Restore is priced at the observed 40-85 ms; a #464 gain is upside the design
  must not assume.
- **Host-RAM accounting does not exist yet.** R1's extension to host bytes is
  new work, not a carrier to drive. The #582 ledger is VRAM-only today.
- **Weight streaming is EXPERT-granular, not layer-granular.** R10 needs layer
  and layer-slice granularity; the #77/#123/#125 machinery streams expert rows
  (`expert_offload.py`, `cold_tier_fetch.py`, `offload_movement.py`) and
  `offload_movement.py` carries no per-layer addressing. Slice 5 EXTENDS it.
  The double-buffered prefetch-behind-previous-layer-compute structure is
  likewise a premise of the addendum I could not locate as a general mechanism
  -- slice 5 must confirm or build it, and the overlap pricing in R10 depends
  on it being real.
- **The reference boot's measured demand already contradicts one inherited
  term.** The 2026-08-05 window measured out-of-budget demand of 2434 MiB
  (3080) and 1305 MiB (5090) against a ledger prediction of >=4664 / >=5016,
  driven by the stock activation heuristic (3968 MiB). The ledger is
  conservative, not dangerous, but R2's "exact" is not yet met on that term.
  Fixing it is a prerequisite for trusting any objective computed from it.
- **R9's knob inventory is incomplete.** The classification table covers the
  knobs the addendum named. A sweep for every knob that changes a layout, with
  a cost for each, is part of slice 4b -- an unclassified knob is an unpriced
  assumption, and R9 forbids those.
- **The pressure ladder's autonomy overlaps the planner's authority.** Slice 5
  must resolve this or the two will fight over the same pool under load.
