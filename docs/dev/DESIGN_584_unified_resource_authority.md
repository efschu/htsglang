# DESIGN 584 -- one resource authority: cut, ledger, movement

Status: DESIGN, no code. Branch `feat/exact-vram-ledger` (design only; the
implementation gets its own branch per slice).

This document is written against the mandate plus three addenda. Addendum 3 is
recorded verbatim in intent as R6 and R7 below, because both name an error
class the design must be structurally unable to commit -- not a behaviour it
should prefer.

**Provenance note.** The mandate and addenda 1-2 are summarised here from the
coordinator's messages, not quoted. Where a requirement below depends on
precise wording (the objective function in R4 especially), it is marked
`[CONFIRM]` and must be checked against the original before that slice starts.

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

Each requirement carries a falsifier: a concrete situation that must produce a
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

### R4 -- The objective is throughput/latency, not fit `[CONFIRM]`

The planner does not seek *a* configuration that fits; it seeks the one that
maximises the stated objective subject to fitting. Fitting is a constraint, the
objective is the goal.

*Falsifier:* a configuration that fits comfortably but is throughput-inferior
to a tighter one must lose. If the planner ever prefers slack for its own sake,
this requirement is not implemented.

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
| 1 | Boot path: ledger replaces the `RESERVE` vector | R1, R2 end to end on the production recipe | TICKET_582 gates (a)+(b) |
| 2 | Candidate enumeration (cuts, no movement) | R3, R6 candidate set | hermetic: F6 candidate-set assertion |
| 3 | Objective + movement cost model | R4, R5 | hermetic: F6, F6b |
| 4 | Phase-dependent heat ranking | R7 | hermetic: F7, F7b |
| 5 | Actuation through existing carriers | R6 execution | GPU: a live re-cut under load |
| 6 | Explainability surface | R8 | hermetic + a live boot |

Slice 1 is already most of the way there on this branch: #582 built the ledger
and the boot wiring behind `--enable-vram-ledger`. Slice 1 completes when
TICKET_582's GPU gates pass and the flag can default on.

**Slices 2-4 are desk-provable in full.** They are search, arithmetic and
ordering; none needs a card. This is deliberate -- the expensive slice (5) is
the only one that should ever wait on a GPU window.

---

## 5. Known gaps and risks

- **`#551` elastic co-residency has no marker in `python/sglang/srt/`.** The
  analysis exists (`docs/dev/ANALYSE_553_elastic_coresidence.md`) but the
  mandate named #551 as a carrier and I did not find one in code. Either the
  number is 553, or the carrier is a design and not yet a mechanism. MUST be
  resolved before slice 5 plans to drive it.
- **R4's objective is `[CONFIRM]`.** Throughput and latency are not the same
  objective and can disagree. The design cannot pick for the user.
- **Re-cut cost is not currently measurable.** R5 needs a movement-cost model,
  and no component times a reshard today. Slice 3 must either measure it or
  declare it a calibrated term in the ledger's sense (measured once per rig,
  fingerprinted) -- not a literal.
- **The pressure ladder's autonomy overlaps the planner's authority.** Slice 5
  must resolve this or the two will fight over the same pool under load.
