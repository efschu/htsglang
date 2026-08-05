# DESIGN 584 -- one resource authority: cut, ledger, movement

Status: DESIGN, no code. Branch `feat/exact-vram-ledger` (design only; the
implementation gets its own branch per slice).

This document is written against the mandate plus four addenda. Addenda 3 and 4
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

Requirements R1-R9. Each carries a falsifier: a concrete situation that must produce a
specific observable, and which the current design would fail if the
requirement were dropped.

### R1 -- The ledger is the only VRAM arithmetic

Every VRAM decision anywhere resolves through `mem_ledger` terms. No component
keeps a private constant, a private fraction, or a private reserve.

*Falsifier:* grep gate -- a new module-level MiB constant used in memory
arithmetic outside `mem_ledger` fails CI. (The three constants #582 already
replaced, 1280/1536/600, are the template for what must not recur.)

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

*Falsifier:* a 2% throughput gain that costs a 40 s reshard must be REFUSED at
short horizon and ACCEPTED at long horizon, with the horizon an explicit input,
not a constant.

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

| Class | Criterion | Flips at | Examples |
|---|---|---|---|
| **tick-flippable** | not graph-addressed AND zero data movement | every forward | #439 cold-expert compute assignment (verified VRAM-neutral: `expert_heat_migration.py:31` calls same-size-before-and-after "the #439 sizing latch's invariant", `expert_compute_placement.py:566`); candidate: token vector per prefill batch |
| **regime-flippable** | graph-addressed OR moves data | regime boundary, with hysteresis | anything requiring CUDA-graph recapture (3-6 s, the general property in `DESIGN_140`); KV resharding (#297); TP re-cut (#261) |

The two criteria are the whole classification. A knob is tick-flippable exactly
when flipping it costs neither a recapture nor a byte moved -- and that is a
property of the knob that can be checked, not a judgement call.

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
| 4b | Per-knob flip pricing and classification | R9 (i), (ii) | hermetic: F9, F9b |
| 5 | Actuation; BUILD elastic co-residency (#553) | R6 execution, tenant events | GPU: a live re-cut under load |
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
  fingerprinted) -- not a literal. The same applies to the recapture cost R9
  prices knobs by: "3-6 s" is an observed range, and a range is not a model.
- **R9's knob inventory is incomplete.** The classification table covers the
  knobs the addendum named. A sweep for every knob that changes a layout, with
  a cost for each, is part of slice 4b -- an unclassified knob is an unpriced
  assumption, and R9 forbids those.
- **The pressure ladder's autonomy overlaps the planner's authority.** Slice 5
  must resolve this or the two will fight over the same pool under load.
