# DESIGN #363 — dynamic regime controller

Phase 1 deliverable: the design and the falsifier plan. No scheduler wiring
lands here. What does land is the part that can be settled at a desk without a
card — the classifier, the hysteresis, the stage-admission arithmetic — as
pure functions with hermetic tests, so phase 2 wires a component that is
already falsifiable rather than writing one inside the scheduler loop.

Source of the constraints: `ANALYSE_363_dynamic_regime_controller.md`. This
document conforms to it except in three places, argued with evidence in §7.

---

## 1. What the controller is, in one paragraph

The serving load changes shape. The rig has several pre-solved configurations,
each optimal for one shape and measurably worse for the others. A restart is
the only way to move between them today. The controller watches the load,
names its shape, and selects the configuration the planner already solved for
that shape — using the runtime actuators that exist. It never invents a
configuration, never optimises a rank's idle share, and never moves on a
signal it cannot distinguish from noise.

---

## 2. The central separation

The user's idea is "watch each card's per-round milliseconds and move
whichever cut is currently the brake". Taken literally that is one loop. It
has to be three questions, because they have three different instruments and
three different error modes:

| question | instrument | when |
|---|---|---|
| **What kind of work is arriving?** | replicated scheduler state: forward-mode mix, queue composition, KV occupancy | every round, free |
| **Which configuration is best for that work?** | the planner, offline, from measured per-card lane rates (#298a/#324/#353/#357) | at boot, once |
| **Does the live rig agree with the planner?** | per-rank ms/round and the compute/wait spread (#252) | continuously, as validation |

The regime is a **queue question**, not a timing question. Per-rank
milliseconds answer "which cut is the brake", and #264 already measured what
happens when that answer is acted on directly: prefill +8.2 %, decode
−13.7 %, KV capacity −47.9 %, net negative, because 69–75 % of the window is
collective floor that no cut reallocation touches. Wait time is a symptom.

So per-rank ms/round enters the controller in two roles, neither of them as
the classifier's primary axis:

1. **Validation.** The planner predicted a per-rank time split for the
   installed stage. If the live split disagrees beyond its own noise band, the
   plan is stale — log it, and in a later phase trigger a re-solve. This is
   also the entire output of the observe-only ship (§6).
2. **Veto.** A flip toward a concentrated stage is refused when the live
   compute/wait spread says the concentration the planner solved for is not
   the imbalance the rig currently shows.

That separation is what keeps the user's intent ("keep speed at the optimum
dynamically", "everything interlocks") while staying on the right side of the
#264 result.

---

## 3. The regime classifier

### 3.1 Inputs, and which tier they are in

The #287 runtime states the uniformity rule that any in-loop controller in
this fork inherits (`managers/kv_pressure_runtime.py:14-20`): replicated
inputs only, pure functions, so every rank reaches the same verdict without a
collective, and the only collective is the unconditional consensus reduction
at the cadence boundary.

Per-rank ms/round is **not** replicated. It differs per rank by construction —
that is the signal (`utils/collective_clock.py:27-29`: "the *spread* of `wait`
across ranks is the shard-imbalance signal"). The inputs therefore split into
two tiers, and the tier decides how a value may be used.

**Tier R — replicated, drives the classification.** Every one of these is
already identical on every rank, for the same reason the #287 occupancy sample
is:

| input | source | note |
|---|---|---|
| `prefill_rounds` / `decode_rounds` in the window | the scheduler's own forward-mode accounting | the mode is chosen by replicated policy |
| `held_tokens`, `capacity_tokens` | the admission limiter's sample, as #287 already uses | `kv_pressure_runtime.py:16-18` |
| `running_bs` | `ScheduleBatch` | replicated |
| `queued_reqs`, `queued_prompt_tokens`, `max_queued_prompt_tokens` | `scheduler.waiting_queue` (`managers/scheduler.py:1269`) | the admission decision is replicated; `gdn_slot_executor.on_round(running_batch, self.waiting_queue)` at `scheduler.py:3348` is the precedent for reading it from a per-round hook |
| `round_index` | replicated counter | the cadence gate, never a local verdict |

**Tier L — rank-local, may only be consumed after reduction.** Per-rank
forward device ms (`utils/device_timer.py`, `DeviceTimer`), and the
prefill-only compute/wait split (`utils/collective_clock.py`). A tier-L value
may never be tested before a group collective on the local rank. It is
quantized to an integer, packed into the consensus payload that already runs
unconditionally at every cadence boundary, and only the reduced summary
(min/max across ranks, hence the spread) is allowed to influence a decision.
This costs nothing: the reduction is already there, and the payload grows by a
few `int64`.

The packing convention is the fork's existing one — each field as `(v, -v)` so
one MIN-reduction yields min and max per field
(`kv_pressure_runtime.py:90-97`, `kv_reshard.py:76-82`). For tier L the
interesting output is exactly `max - min`: the spread.

### 3.2 The regimes

Four, mutually exclusive, one always holds.

| regime | entry condition (tier R only) | what it means |
|---|---|---|
| `PREFILL_HEAVY` | prefill share of window rounds ≥ `enter_prefill` **or** queued prompt tokens ≥ `burst_tokens` | the rig is ingesting |
| `DECODE_HEAVY` | decode share ≥ `enter_decode` and queue empty | the rig is emitting |
| `KV_PRESSURE` | occupancy ≥ the #287 ascend mark | the pool is the constraint |
| `MIXED` | none of the above held its window | the honest default |

`KV_PRESSURE` **outranks** the other two. It is not a fourth peer: it is the
constraint regime, and when it holds, stage selection is the #287 ladder's
decision, not the regime controller's. Two controllers must not both own the
KV axis (§7.3).

`MIXED` is a real answer, not a fallback. It is the regime under which the
controller does nothing, and on a general workload it should be the common
one. A controller that is rarely in `MIXED` on mixed traffic is misclassifying.

### 3.3 Hysteresis and dwell

Two independent mechanisms, because they stop two different failures.

**Hysteresis** stops chattering around a threshold. The fork already has the
shape and enforces it as a contract rather than a recommendation
(`model_executor/kv_pressure_ladder.py:1069-1125`): ordered marks, asymmetric
windows, and a constructor that refuses a configuration violating the
asymmetry. #363 inherits all three properties and the enforcement style:
entry and exit thresholds differ, exit windows are longer than entry windows,
and the constructor raises on any ordering that would let a regime enter and
leave on the same sample.

**Dwell** stops thrash that hysteresis cannot see: a load that genuinely
alternates faster than a flip pays for itself. Hysteresis is about the
*signal*; dwell is about the *actuator cost*. A configuration must be held
long enough that the move amortizes, however clean the signal is.

Dwell is therefore **derived, not typed**:

```
min_dwell_rounds(stage) = ceil(DWELL_AMORTIZATION * flip_cost_s / mean_round_s)
```

with `DWELL_AMORTIZATION = 20` as the one named policy constant: at 20x, the
flip consumes at most 5 % of the interval it bought. `flip_cost_s` comes from
the stage table (§4) and `mean_round_s` from the live window, so a 6 s weight
flip on a 40 ms decode round yields ~3000 rounds of dwell and a 0.3 s KV
reshard on the same round yields ~150. Both numbers fall out of measurement;
neither is a taste.

### 3.4 Proposed constants, and the measurement that sets each

Every number below is **provisional** and paired with the experiment that
replaces it. None of them may ship as a default until its own experiment has
run, and §5.3 is the gate: a threshold may not sit inside its signal's
measured A-vs-A band.

| constant | proposed | set by |
|---|---|---|
| `window_rounds` | 64 | F3 per-signal A-vs-A: the shortest window whose signal band is below the smallest threshold gap. Start at 64 because the #287 descend window is 64 and the two share a cadence. |
| `enter_prefill` | 0.35 | F3 on `prefill_share`. |
| `exit_prefill` | 0.15 | Asymmetry contract: gap must exceed 2x the measured `prefill_share` band. |
| `enter_decode` | 0.90 | F3 on `decode_share`. |
| `exit_decode` | 0.70 | as above |
| occupancy marks | **inherit #287 verbatim** (0.85 / 0.70 / 0.55) | already measured; inventing a second set would put two thresholds on one physical quantity |
| `consensus_interval` | 8 | inherit (`kv_pressure_runtime.py:509`) — one cadence for all in-loop controllers |
| `DWELL_AMORTIZATION` | 20 | policy, not measurement; stated as policy |
| `burst_tokens` | see §5 | derived per stage from the admissibility interlock, not a free parameter |
| `spread_veto_pct` | 25 | F3 on the tier-L spread; the #210 decode boot-to-boot floor is 1.07 % and the harness floor is 4.2 % (`planner/key_solver.py NOISE_FLOOR_PCT`), so 25 % is deliberately far above both until measured |

The two numbers already in the tree that bound this work: `_SPEED_TIE_TOL_PCT
= 1.0` in `planner/lever_profiles.py` (the #210 measured decode boot-to-boot
noise floor, 1.07 %) and `NOISE_FLOOR_PCT = 4.2` in `planner/key_solver.py`
(the benchmark harness floor). A live per-round signal will be noisier than
either. Assuming otherwise is the mistake this section exists to prevent.

---

## 4. The stage table

### 4.1 A stage is a tuple, not a knob

This is the sharpest correction the measured record forces (§7.3). The #354
table, measured over four boots at the 27B point
(`docs/rig-runbook.md:222-230`):

| arm | MLP vector | prefill s=1 | decode bs=1 | `max_total_num_tokens` |
|---|---|---|---|---|
| FP8 decode (auto) | none | 1256.7 tok/s | **122.2 tok/s** | 453 632 |
| FP8 prefill | `16,1,1` | **1540.3 tok/s** (+22.6 %) | 97.8 tok/s (−20.0 %) | 96 256 (−79 %) |
| INT8 decode (auto) | none | 1685.2 tok/s | **112.0 tok/s** | 464 256 |
| INT8 prefill | `10,1,1` | **1787.5 tok/s** (+6.1 %) | 119.6 tok/s (n=1, undecided) | 137 664 (−70 %) |

The prefill arm costs 79 % of the KV pool. A controller that flipped only the
weight cut would cut `max_total_num_tokens` by 4.7x underneath a live working
set. A stage is therefore the whole tuple:

```
Stage = (weight_vector, kv_token_vector, per_rank_vram_budget,
         reserve, max_total_num_tokens, measured_gain, measured_band)
```

and the flip moves all of it or none of it.

### 4.2 Admissibility — the interlock, as arithmetic

A stage may only be selected when the live working set fits its pool:

```
held_tokens <= ascend_mark * stage.max_total_num_tokens
```

With the #287 ascend mark of 0.85, the FP8 prefill stage is admissible only
below `0.85 * 96 256 ≈ 81 800` held tokens, against the decode stage's
`≈ 385 600`. That single inequality is the whole "everything interlocks"
intuition made checkable, and it is what turns the queue-aware trigger in §5
from a heuristic into a computation: the burst has to fit the stage it is
asking for.

It also settles a build question. `held_tokens` and `capacity_tokens` are the
#287 sensor's own inputs, so admissibility and pressure are the same
arithmetic on the same numbers. Running them in two controllers would let one
flip toward a stage the other is simultaneously relieving pressure away from.

### 4.3 A stage must earn its place in the table

Per the #360 standard (§5.3), a stage is admitted only if its measured gain
clears the band measured on its own arm. Applied to the table above:

* **FP8 prefill: admitted.** +22.6 % prefill against a 4.2 % harness floor.
* **INT8 prefill: candidate, not a stage.** +6.1 % prefill from one boot, and
  the decode column is explicitly undecided at n=1. 6.1 % against a 4.2 %
  floor is not a margin to move a live server on. It is re-measured or it
  stays out.

The table carries `measured_gain` and `measured_band` as fields, so this is a
constructor check, not a review convention — the same discipline the #287
runtime applies to actuator wiring, which refuses at boot rather than at the
first episode (`kv_pressure_runtime.py:160-177`).

### 4.4 Edges: which actuator moves which axis

| axis | actuator | entry point | cost | autonomy |
|---|---|---|---|---|
| KV token vector | #297 phase-boundary reshard | `KvReshardRuntime.arm(vector, source)` (`managers/kv_reshard.py:322`), commits at the next group-idle consensus boundary (`:364`) | instrumented per move, not a constant: `read_ms` / `exchange_ms` / `write_ms` / `total_ms` in `last_stats` (`kv_reshard.py:539-555`). ANALYSE quotes < 1 s; the controller reads the real number and feeds it back into `min_dwell_rounds` | full, within the declared ceiling set |
| VRAM budget / KV capacity | #330 dial + C re-raise | `KvCapacityRuntime.apply_budget_request(...)` (`managers/vram_dial.py:462`), `on_round()` (`:705`) | a **grow** commits pages into the boot VA reservation with no tensor move and no graph re-capture (`vram_dial.py:20-26`) | **grow: autonomous. shrink: never autonomous.** A shrink flushes the radix cache and "only an explicit dial authorizes that" (`vram_dial.py:44-46`). The controller may reclaim stranded VRAM; it may not throw away a prefix cache to do so |
| weight (MLP/GEMM) cut | none | — | restart. `uneven_perf.py` states it in the plan log: "Switching arms needs a RESTART: the MLP vector is a weight split and no runtime actuator moves weights" (`:5524-5531`) | n/a until phase 4 |
| spec algorithm / k | #156 ladder | existing | existing | out of #363's initial scope — it has its own controller and its own self-conditioning history |

Two consequences worth stating plainly. First, with today's actuators the
controller cannot reach the prefill stage at all: its defining axis is the
weight cut. What it *can* do in phase 3 is the KV and VRAM half of a stage —
which on the prefill arm is exactly the 14.3 GiB (FP8) / 12.3 GiB (INT8) the
runbook records as stranded on the 3080s with "nothing reclaims it
automatically". That is a real, reachable win with no weight mover.

Second, the stranded-VRAM reclaim is a **grow**, so it is autonomous under the
rule above. The controller's first shipped action is therefore the safest one
in the table.

---

## 5. Queue-aware pre-staging

### 5.1 Why it must be predictive

Every actuator commits at a **group-idle** boundary (`kv_reshard.py:364`,
`:429`). The moment a large prefill starts, the server is not idle, and the
window is gone until it finishes. A reactive controller therefore always
arrives one burst late. The queue is the only place where the burst is visible
before it costs anything.

### 5.2 The trigger, spelled out

At every consensus boundary, on tier-R state only:

```
burst_tokens   = sum(len(req.origin_input_ids) for req in waiting_queue)
burst_max      = max(len(req.origin_input_ids) for req in waiting_queue)
target         = the stage the classifier would pick for PREFILL_HEAVY

pre_stage IF ALL OF:
  (a) burst_tokens >= PRESTAGE_TOKEN_FRACTION * running window prefill volume
  (b) burst_max    >= PRESTAGE_SINGLE_PROMPT_TOKENS
  (c) held_tokens + burst_tokens <= ascend_mark * target.max_total_num_tokens
  (d) rounds_since_last_flip     >= min_dwell_rounds(current_stage)
  (e) the group is idle NOW, or projected idle within PRESTAGE_HORIZON_ROUNDS
```

(c) is §4.2's interlock applied to the *predicted* working set rather than the
current one — this is the clause that stops the controller from pre-staging
into a pool the burst it is preparing for cannot fit. (d) is the dwell gate,
checked before the move is armed rather than after. (e) mirrors the #287
sensor's trend-and-horizon shape: the pre-stage mark looks further ahead than
the flip mark (`kv_pressure_ladder.py:1082-1086`).

Constants: `PRESTAGE_SINGLE_PROMPT_TOKENS` starts at 8192 (the point at which
a single prompt's prefill exceeds a decode round by two orders on this rig)
and `PRESTAGE_TOKEN_FRACTION` at 1.0. Both are F3 subjects.

### 5.3 Abort

The #287 sensor pairs every pre-stage mark with a longer abort window
precisely so flapping does not "stage and discard forever"
(`kv_pressure_ladder.py:1077-1080`, enforced at `:1122`). #363 inherits that:
`abort_prestage_window > prestage_window`, enforced in the constructor. A
queue that drains before the flip commits aborts the arming, and the abort is
slower than the arming was.

---

## 6. Staged rollout

**v1 — observe-only. This is what ships.**
The classifier runs in the scheduler loop on tier-R state, packs its tier-L
summary into the existing consensus reduction, and emits one structured record
per boundary: regime, would-be stage, would-be actuator call, and the reason
it was or was not admissible. It calls no actuator. Its output is the input
trace that F2 and F4 need, so the observe-only ship is not a courtesy phase —
it is the instrument that earns the actuating phase.

The precedent is exact: the #287 ladder shipped Stage 1 as bookkeeping-only
with the wired/planned split printed at boot and named on every flip
(`kv_pressure_runtime.py:36-48`, `:369-374`). #363 uses the same
`WIRED`/`PLANNED-ONLY` inventory so a reader can never mistake a logged
intention for a move.

**v2 — the reachable half.** VRAM grow (stranded-VRAM reclaim) and KV token
vector, both already autonomous-safe per §4.4. No weight mover, so no stage in
the #354 sense is reached; what is reached is the KV/VRAM component of one.

**v3 — weight mover.** Separate decision, per ANALYSE §Sequencing phase 4.

Flag shape, matching the fork's existing spelling
(`--kv-reshard-vectors`, `--enable-vram-dial`):
`--regime-controller off|observe|on` with `observe` as the v1 default-on
candidate and `off` the shipped default until F4 passes.

---

## 7. Where this design departs from ANALYSE_363

### 7.1 The compute/wait split cannot classify the decode regime

ANALYSE §3 proposes classifying "prefill-heavy / decode-heavy / KV-pressure
from the existing per-rank sensing (#252 CollectiveClock, compute vs wait
split, graph-replay-honest)".

The split is graph-replay-honest in the sense that it refuses to lie — and
that is precisely why it cannot serve here. `collective_clock.py:40-45`:
collectives issued inside a replayed CUDA graph cannot record timing events,
"so a graph-covered forward reports no split at all rather than a wrong zero".
The reporter says the same (`metrics_reporter.py:126-130`) and the timer is
installed on plain-prefill forwards only ("only `ForwardMode.is_plain_prefill`
forwards are wrapped", `metrics_reporter.py:137`, installed at `:327-345`).

CUDA graphs are the fork's standing default, decode included (ANALYSE, "CUDA
graphs" section). So on every configuration this controller will actually run
in, the compute/wait split is a **prefill-only instrument** and is structurally
absent for decode. A decode-heavy classifier built on it would classify from
missing data.

Consequences adopted in this design: the classifier's primary axis is tier-R
scheduler state (§3.1); total per-forward device ms — which `DeviceTimer`
measures around the whole forward, replay included — is the tier-L timing
signal; and the compute/wait spread keeps the job it is actually an instrument
for, prefill imbalance, as the §2 veto.

### 7.2 Per-rank ms/round breaks #287 rule 1 as written

ANALYSE §3 says to generalise the #287 trigger onto the per-rank sensing. Rule
1 of that machinery is replicated inputs only, so that every rank reaches the
same verdict without a collective (`kv_pressure_runtime.py:14-20`), and the
whole value of per-rank timing is that it is *not* uniform
(`collective_clock.py:27-29`). Generalising the trigger onto a rank-local
input silently drops the property the module was built to hold, and the
failure mode is the #94/#194/#259/#312 hang, not a slowdown.

This is not a reason to drop the signal — it is a reason to route it. The
tier-R / tier-L split in §3.1 keeps every branch on replicated state and
carries the rank-local summary through the reduction that already runs
unconditionally at the cadence boundary. This is the house rule
[[rank-lokaler-test-vor-kollektiv]] applied before the fact instead of after
the fourth hang.

### 7.3 A stage is a tuple, and #363 must own the KV axis jointly with #287

ANALYSE §3 frames #363 as generalising the #287 trigger while #287 keeps
running. Two independent controllers would then both write the KV axis: the
regime controller selecting stages whose `max_total_num_tokens` differs by
4.7x (§4.1), and the pressure ladder relieving the occupancy that selection
just changed.

The loop is concrete, not hypothetical, and it is derivable from the measured
numbers. A flip to the FP8 prefill stage takes the pool from 453 632 to
96 256 tokens. A working set of 100 000 tokens sits at 22 % occupancy before
the flip and does not fit at all after it. Occupancy is the pressure ladder's
input, so the flip mechanically drives the ladder to ascend, and the ladder's
relief changes the state the regime controller reads next. That is the #156
self-conditioning trap with a mechanism attached — the controller reacting to
an effect it caused — and it is exactly what F2 exists to catch.

Adopted instead: `KV_PRESSURE` outranks the other regimes (§3.2), stage
admissibility is the same arithmetic on the same replicated sample as the
pressure sensor (§4.2), and the commit path is one arbiter. #363 subsumes the
#287 commit decision rather than sitting beside it. This is a change to the
build shape in ANALYSE §3, and it is the one place where "generalise, do not
fork" has to mean subsume rather than wrap.

---

## 8. Falsifier plan

Each falsifier states what would make it fail, before it is run.

### F0 — the interlock refusal (hermetic, no card)

A stage flip whose target pool cannot hold the current working set must be
refused, with the arithmetic in the message. Drive the admission predicate
with `held_tokens = 100 000` against the FP8 prefill stage's 96 256 and
require a refusal naming both numbers. **Fails if** the controller arms a flip
that cannot fit, or refuses without naming the numbers.

*Built in this branch.*

### F1 — oscillation under adversarial alternation (hermetic, no card)

Synthesise a trace that alternates across the hysteresis boundary at the worst
frequency for each mechanism: at the entry threshold ± epsilon, and at a period
just under `min_dwell_rounds`.

**Passes if** the flip count over the trace is bounded by
`floor(rounds / min_dwell_rounds)` and the regime sequence contains no
transition pair `A→B→A` inside one dwell interval. **Fails if** either bound
is exceeded — which is the definition of a controller that would thrash a live
server.

*Built in this branch.*

### F2 — self-conditioning replay (card, phase 2 gate)

The #156 pattern, with the mechanism from §7.3 as the named suspect.

1. Run the **do-nothing arm** with the controller in observe-only. Record the
   full tier-R/tier-L input trace and the regime sequence `R_open`. The
   controller took no action, so nothing in this trace is self-caused.
2. Run the **live arm** with the controller actuating on the same workload.
   Record `R_closed`.
3. Replay `R_open`'s input trace through the classifier offline.

**Passes if** every transition in `R_closed` also appears in the replay of
`R_open`, in the same order. **Fails if** `R_closed` contains a transition the
open-loop trace does not produce: that transition was a response to the
controller's own effect. The predicted failure is `PREFILL_HEAVY →
KV_PRESSURE` immediately after a stage flip, caused by the denominator change
rather than by the load.

### F3 — A-vs-A noise floor, per input signal (card, gates every constant)

The #360 standard, applied per signal rather than per benchmark. Two identical
runs, same workload, same seed, same boot recipe. For each classifier input
`s`, the band is what the arm measures on **itself**:

```
band(s) = max over windows of |s_runA(w) - s_runA'(w)|
```

This is the construction `scripts/dual_group/r12` uses for the lane gate:
`perturbation_band()` reads the band off the agreeing positions of the two
arms (`verdict.py:72-108`), and the graded gate's rule is
`abs(spec - no_spec) <= band` with the band measured as
`max(|no_spec - no_spec_repeat|, |spec - spec_repeat|)`
(`ANALYSE_284_lane_spec_divergence.md:130-135`).

**Passes if** every threshold in §3.4 sits outside its own signal's band by at
least the band itself (2x margin). **Fails** — and the constant is
re-derived, not re-argued — if any threshold sits inside the band. A threshold
inside its own noise band is a coin flip with a number written next to it.

Framing numbers (flip counts, regime dwell histograms) are reported next to
the verdict and are never criteria, per the same standard.

### F4 — do-nothing baseline (card, gates default-on)

Three arms on one workload: `off`, `observe`, `on`.

The metric is the house measure, ms/round per phase — ms/verify and
ms/prefill, per rank, not tok/s. The band is F3's, measured on the `off` arm
against itself.

**The controller earns default-on only if** it beats the `off` arm on ms/verify
**and** ms/prefill by more than the band, **at equal or better
`max_total_num_tokens`**. The capacity clause is not decoration: §4.1 shows
the prefill stage buying +22.6 % prefill with −79 % pool, and an arm that wins
on speed by giving away capacity has changed the question rather than answered
it.

**Fails** — and the ship stays observe-only — if the delta is inside the band.
The `observe` arm exists in this comparison to price the classifier itself: if
`observe` differs from `off` beyond the band, the instrument is not free and
its cost has to be paid down before the actuation is judged.

### F5 — consensus desync (hermetic, no card)

Inherited wholesale from #287/#297: inject a consensus channel that merges
different ranks' payloads and require the same loud error on every rank rather
than a hang (`kv_pressure_runtime.py:104-108`, `:279-288`). Any new field in
the packed payload — the tier-L summary above all — is covered by this.

*Deferred to phase 2 with the wiring, since it needs the real payload.*

---

## 9. What phase 1 builds

`python/sglang/srt/managers/regime_classifier.py`, stdlib-only, no torch, no
scheduler import, importable at desk speed:

* `RegimeSample` — one boundary's tier-R state plus the reduced tier-L summary.
* `Regime` constants and `classify_window()` — the pure classification.
* `RegimeSensor` — the windowed classifier with asymmetric hysteresis and
  constructor-enforced ordering, mirroring `KvPressureSensor`.
* `Stage` / `StageTable` — the tuple of §4.1, with the §4.3 gain-vs-band
  admission check and the §4.2 admissibility predicate, both as constructor
  and query-time refusals that name their arithmetic.
* `min_dwell_rounds()` — §3.3's derivation.
* `DwellGate` — the dwell mechanism, separate from hysteresis.
* `signal_band()` — the #360 A-vs-A band, so F3 computes its bands with the
  same code the controller compares against.
* `pack_proposal()` / `unpack_reduced()` — the `(v, -v)` MIN convention, so
  §3.1's tier-L routing is executable rather than described.

Not built, deliberately: any scheduler hook, any actuator call, any flag
parsing. Phase 1 ends at the boundary of the loop.

### 9.1 Two things building it changed in the design

**Two timelines, not one.** The classifier reports the regime every boundary,
truthfully, including when the load alternates faster than any flip could pay
for itself. The dwell gate decides what is affordable. Collapsing them — a
classifier that only reports what it may act on — would make the observe-only
mode useless exactly where it is most needed: a workload whose regime
alternates every 24 rounds is a finding, and it is the finding that says the
controller must not actuate on this workload at all. The F1 falsifier
therefore asserts on the *committed stage* timeline, not on the regime
timeline, and both are logged separately in v1.

**`MIXED` must not be absorbing.** The first implementation gave every regime
a symmetric exit window, including the resting state, and the resting state
then held forever: `MIXED` asserts nothing, so nothing ever falsified it, so
its exit streak never accumulated. Fixed by making the asymmetry explicit —
leaving `MIXED` needs only a challenger, falling back to it needs only the
incumbent to have failed, and every other transition needs both. Caught by
construction rather than by review, which is the argument for building the
classifier in phase 1 instead of describing it.

---

## 10. Phase 2 — falsifier results

All hermetic, CPU-only, no card. Suite:
`test/registered/unit/managers/test_regime_classifier.py` (F0, F1) and
`test_regime_observe.py` (F2–F5), **87 passed**.

### F0 — interlock refusal: PASS

A stage flip whose pool cannot hold the working set is refused with both
numbers in the message. Driven on the measured #354 pair: 100 000 held tokens
against the FP8 prefill arm's 96 256, refused; against the decode arm's
453 632, admitted.

### F1 — oscillation: PASS

Two adversarial traces, one per mechanism.

* Dither at the entry threshold, 512 rounds: **≤ 2 flips** (measured 0). The
  entry window requires consecutive samples, which a dither never supplies.
* Square wave with a 16-round half-period against a 64-round dwell, 1024
  rounds: flips bounded by `rounds / min_dwell_rounds`, and the dwell gate
  recorded refusals — i.e. it actually bound rather than being satisfied
  vacuously.
* No stage is returned to inside one dwell interval.

### F2 — self-conditioning replay: PASS, and the trap is confirmed real

The sharpest result of phase 2, because it demonstrates the mechanism before
crediting the guard with closing it. Workload: 90 000 held tokens, prefill-
heavy, shape constant for 240 rounds — so *any* regime change in the closed
loop came from the controller.

| arm | capacity source | regimes observed |
|---|---|---|
| open loop | fixed at the decode pool (453 632) | `PREFILL_HEAVY` only, occupancy 20 % |
| closed loop, **naive** (no interlock) | follows the committed stage | `PREFILL_HEAVY` → flip → `KV_PRESSURE` |
| closed loop, **guarded** | follows the committed stage | identical to its own open loop |

The naive arm manufactures `KV_PRESSURE` out of its own denominator: 90 000
tokens is 20 % of 453 632 and 93 % of 96 256, above the 85 % ascend mark. The
guarded arm refuses that flip (F0's arithmetic) and its transition sequence
equals the open-loop replay exactly.

Control on the control: at 40 000 held tokens the prefill pool is not the
constraint, the flip **does** commit, and the closed loop still matches its
replay. The guard is discriminating, not simply refusing.

### F3 — noise floor per input signal: PASS

Band computed the house way (#360): the largest difference an arm shows
against its own repeat. On the prefill-share traces used here the band is
**0.02**.

* The shipped threshold gap (`enter_prefill` 0.35 − `exit_prefill` 0.15 =
  0.20) clears 2× that band.
* Demonstrated, not asserted: a sensor whose marks are 0.01 apart — inside the
  band — tracks a pure-noise trace and transitions repeatedly; the shipped
  marks transition **zero** times on the identical trace.
* The occupancy marks are #287's, so they are not re-derived and cannot
  disagree with the pressure ladder about the same pool.

Constants in §3.4 remain provisional: this fixes the *method* and proves the
sensor is not thresholding inside noise, but the real per-signal bands are a
card measurement (phase 3).

### F4 — do-nothing baseline: hermetic half PASS, card half deferred

The card comparison (off / observe / on, judged on ms/verify and ms/prefill
against the off arm's own band, at equal or better capacity) is phase 3. What
is settled here is the property that makes it meaningful: the observe arm
reports `actuations: 0` by construction, calls no actuator (checked on the
syntax tree — imports and attribute calls, not a substring grep over prose),
and is `off` by default. It is therefore a legitimate do-nothing baseline that
differs from `off` only in what it writes down.

### F5 — consensus desync: PASS, with a deliberate departure from #287

An injected channel supplies a peer that classified differently while agreeing
on every other field. The observer **counts and logs** the disagreement at
WARNING and does **not** raise.

That is the departure and it is intentional. #287 raises because continuing
would run a collective under a geometry the ranks disagree about; nothing here
acts, so nothing here can hang, and an instrument that takes the server down
while proving the classifier is safe has failed at its own job. The count is
carried in `summary()["desyncs"]` because it is the **phase-3 gate**: a
non-zero desync count over a real workload blocks wiring any actuator.

---

## 11. The observe-only wiring contract

What phase 2 shipped into the loop, stated as obligations so phase 3 inherits
them explicitly.

### 11.1 Placement

One block in `managers/scheduler.py`, at the same between-tick boundary as the
#287 pressure block and the #364 GDN executor, after the #364 call and before
batch selection. Two reasons: the previous batch is retired, so the per-rank
device timing it reads is a completed measurement rather than a half-recorded
one; and the next batch is not selected yet, so nothing the observer reads can
be mid-mutation.

### 11.2 Cost when off

`SGLANG_REGIME_OBSERVE` unset (the default) resolves the mode once on the
first iteration and caches it; the per-round cost is one attribute compare and
the observer is never constructed. No import, no collective, no allocation.
Pinned by a source-level contract test that the build sits behind the gate.

The flag proper (`--regime-controller`) lands with phase 3, when there is an
action to authorize. Registering a server-args knob for a no-op would spread
the change across `server_args.py` and `environ.py` for nothing.

### 11.3 The tier split, as an obligation on the caller

Every keyword the scheduler passes is **tier R — replicated** except one:

```
prefill_active            #287's own phase definition, reused not re-derived
held_tokens               sum(req.seqlen) over the running batch
capacity_tokens           _global_kv_capacity_tokens()   (the #346 global span)
running_bs                running_batch.batch_size()
queued_reqs               len(waiting_queue)
queued_prompt_tokens      sum(len(req.origin_input_ids))
max_queued_prompt_tokens  max(len(req.origin_input_ids))
--------------------------------------------------------------------
rank_forward_ms           TIER L. This rank's own number.
```

A test asserts that exact keyword set, so adding an input to the hook forces a
tier decision rather than allowing one by default. The hook is keyword-only
for the same reason.

`rank_forward_ms` is accumulated, never branched on, quantized into the packed
proposal and read back only as a group spread. Reading it locally before the
collective is the #94/#194/#259/#312 hang class.

### 11.4 The collective

One bounded MIN all-reduce over the TP CPU group, every
`consensus_interval`-th round, gated by the **replicated round counter** and
never by local state — #287 rule 2 verbatim, using #287's own
`default_collective_min` rather than a second channel with a second idea of
what a dead peer looks like. A multi-rank group with no channel is recorded as
`uncoordinated` rather than refused: nothing acts on the verdict, so it is a
degraded observation, but it must not read as a clean run.

### 11.5 The one-boundary lag

The spread exists only after the reduction, and the classification that
produced the proposal ran before it. Each record therefore carries both
`sample_spread_pct` (what went in, from the previous boundary) and
`rank_ms_spread_pct` (what this reduction produced). In observe-only nothing
consumes it; when it becomes a veto input in phase 3 it is one boundary stale
**by construction**, which is a property of the reduction and not a shortcut.

### 11.6 The sensing adapter, and its documented absence

`rank_forward_ms_from(scheduler)` reads the #252 per-rank prefill timing
through a structured tap added to `RankPrefillLog` (three attribute writes
after the log line is already emitted; no behaviour change). Per §7.1 it
returns a number only when `last_split_known` is true — a graph-covered
forward reports nothing rather than a wrong zero, and that absence travels
into the payload as a sentinel so a blind rank cannot read as an infinitely
fast one.

### 11.7 What phase 3 must clear before wiring an actuator

1. `summary()["desyncs"] == 0` over a real workload (F5's gate).
2. F2 re-run against a live trace, not only the synthetic one.
3. F3's per-signal bands measured on the rig; every constant in §3.4 replaced
   or confirmed.
4. F4's card comparison passed at equal or better `max_total_num_tokens`.

---

## 12. Phase 3, desk half — the stage table, the flag, and the act path

Built behind an off flag. ENABLING it is still gated on §11.7, which needs
card windows; what is settled here is everything that can be settled without
one, so the card runs measure a finished mechanism instead of debugging it.

### 12.1 The boot stage table

`managers/regime_stages.py`. The #287 ladder generalized: N discrete operating
points, declared and validated at boot, selected at runtime, never invented
there (§1).

A stage's **reachability** is a property of the pair (booted, target),
computed rather than asserted:

| code | condition | actuator |
|---|---|---|
| `booted` | where the server is — and the normal flip-BACK target | none needed |
| `reshard` | same weights, target KV vector is in `--kv-reshard-vectors` | #297 |
| `vram_dial` | same weights and KV vector, budgets differ upward | #330 grow |
| `no_weight_mover` | the weight vector differs | **none — restart** |
| `undeclared_vector` | KV vector not declared, so no rows were reserved | none |

`reachable` includes the booted stage; `flip_targets` excludes it. That split
is the correction the build forced (§12.4). Unreachable stages stay **visible**
in the table with their reason — a planner-solved configuration this boot
cannot move to is information, not something to hide.

Refusals at construction, in the #287 style: two stages claiming one regime (a
tie broken by list order is not a decision), a candidate reusing the booted
name, a stage whose gain does not clear its own band (#360, inherited from
`StageTable`), a booted stage missing from its own table.

The planner feed is a seam (`planner_candidates(server_args, solve_fn=)`) so
the table logic needs no probe to test. With no feed bound the table holds the
booted stage alone and says so — the phase-2 "table absent" report becomes
"table present, 1 stage, 0 flip targets", which is a different and more useful
statement than an empty table pretending to be a full one.

### 12.2 `--regime-controller {off,observe,act}`

`off` is the default. `observe` is phase 2, now first-class.

`act` is **refused at parse time** until `--regime-gate-evidence` names a
complete set of measurements for the four §11.7 items. The refusal is in the
#350 style — it tells the operator what to run:

```
--regime-controller act is refused: the entry gate of DESIGN_363 section 11.7
is not clear. act moves a live server's KV placement, so it is authorized by
measurement, not by a flag.
  [ok]      desyncs_zero: observe run 2026-08-01
  [MISSING] f2_live_replay -- ...
            produce it: replay the observe run's record stream through the classifier
  ...
Until then use --regime-controller observe, ...
```

Two rules the evidence format enforces. A `passed: true` with no `source` is
refused — an unattributed pass is a claim, not evidence. And a **declared but
unparsable** file is an ERROR rather than a closed gate: a typo must not read
as "not measured yet".

A second refusal follows the gate: `act` with no `--kv-reshard-vectors` and no
`--enable-vram-dial` is rejected, because every proposal would then be refused
for want of an actuator and `act` would be an expensive `observe` under a
misleading name.

`SGLANG_REGIME_OBSERVE` survives as an override that can only ever select
`observe`, with a retirement note. It is refused by name if it asks for `act`:
an environment variable is not a place to put an authorization.

### 12.3 The act path, and why observe still cannot reach it

`managers/regime_act.py` is the **only** module that touches an actuator, and
it is a separate module for exactly that reason. `regime_runtime` imports it
inside the act branch of the builder, never at module scope, so the phase-2 F4
property survives as a fact about the import graph: a test walks both syntax
trees and asserts observe cannot reach #297 or #330 now that #363 can.

Two further guards make it structural rather than incidental: the observer
**refuses to be constructed** in observe mode with a `commit_fn`, and refuses
in act mode without one. Observe may not hold an unused path to an actuator,
so a future edit cannot quietly start calling it.

`RegimeActuator` refuses by named return, never by exception — a controller
that raises inside the scheduler loop turns a bad proposal into a dead server.
It applies a VRAM grow before a reshard (a bigger pool cannot make a reshard
fail; a reshard into a pool about to grow would be sized against the smaller
one) and refuses a shrink outright before arming anything, because a shrink
flushes the radix cache and #330 reserves that for an explicit dial.

### 12.4 Interlocks, all binding in act mode

Each was built in an earlier phase and reported by observe; act makes it bite.

1. **Selectability** — a stage the boot table judged unreachable is not a
   candidate, whatever the regime says.
2. **Dwell** (F1) — hysteresis says the signal is real, dwell says the move is
   affordable. Sized to the most expensive flip target in the table, not the
   cheapest, so an expensive flip cannot follow a cheap one inside its own
   amortization window.
3. **Admissibility** (F0, and the #350 lesson) — a stage whose pool cannot hold
   the live working set is refused with the arithmetic.
4. **Group agreement** — a flip under a disputed verdict is the
   #94/#194/#259 hang under an actuator, i.e. exactly what observe existed to
   rule out. A multi-rank group with no consensus channel may not move
   anything.
5. **The one-boundary-stale veto, now binding** — phase 2 built the lag and
   reported it; act requires the group timing to EXIST. Every rank blind means
   the planner's split has never been checked against the rig it is about to
   move.

### 12.5 What building it changed

**`reachable` and `flip_target` are two questions.** The first model made
`selectable` exclude the booted stage, on the reasoning that it is where we
are. That makes every flip a one-way door: when the prefill burst drains and
the controller wants the decode configuration back, the flip-back reads as an
unreachable stage. Returning to the booted stage is always legal — its KV
vector is by definition backed by the pool reserved for it — so `reachable`
includes it and `flip_targets` is the count that answers "does acting have
anywhere to go". Caught by the dwell falsifier, which vetoed a flip-back for
the wrong reason.

### 12.6 Test tally and what the card gates must still prove

59 hermetic cases in `test_regime_act.py`, plus the phase-1/2 suites still
green. The gate falsifier is symmetric and both halves pass: the gate CLOSES
(act refused, every missing item named) and the gate OPENS (a complete
evidence file lets a proposal reach the stubbed actuator with the right
vector and source tag).

Nothing here measures anything. §11.7 stands unchanged, and every item still
needs a card:

1. `summary()["desyncs"] == 0` over a real workload.
2. F2 re-run against a live observe trace, not the synthetic one.
3. F3's per-signal bands measured on-rig; every §3.4 constant replaced or
   confirmed against its own band.
4. F4's three-arm comparison passed at equal or better
   `max_total_num_tokens`.

Until all four are recorded in an evidence file, `--regime-controller act`
does not start a server.

---

## 13. First card window (2026-08-01) — neither gate passed, four findings

10 minutes on cards 0,1,2. Recorded here because two of the four are defects
in this design's own wiring, and one of those is the kind only a rig can find.

### 13.1 What held

FP8 TP=3 uneven, 26 requests, 0 errors. **93 603 verdicts across three ranks,
0 unparsable lines, `agreed=True` on every one — zero desyncs.** The tier-L
spread came back on 92 028 of them. The rank-uniformity property §3.1 could
only assert hermetically holds on three real ranks, and the consensus routing
of the rank-local tier works end to end. That is the substantive positive
result, and it is most of gate 1's content — it just cannot be *recorded* as
gate 1 for the reason in §13.3.

### 13.2 The headline defect: an idle round read as prefill

The regime histogram was `prefill_heavy` 93 600, `mixed` 3, on a rig that was
idle most of the window.

Cause: the scheduler hook computed a BOOLEAN by negating #287's "is this a
decode round", so an EMPTY batch landed on the prefill side. §3.2 says an idle
window must read `MIXED` — an idle window is no measurement, not 0 % prefill —
and `RegimeSample` already encodes that correctly by returning `None` shares
on an empty window. The bug was entirely in the mapping that produced the
input, and the hermetic tests never touched it because **they passed
`prefill_active` directly**. A test that supplies the value under test cannot
falsify the code that computes it.

Fixed: the hook passes a three-way `phase` (`prefill` / `decode` / `idle`),
idle counts in the round index and in neither share, and an unknown phase is
refused by name. The prefill/decode split for a round that *has* work is still
#287's, so the two controllers do not disagree about a working round. Four new
cases pin it, one of which drives the scheduler's own mapping rather than the
observer's parameter.

### 13.3 Gate 1 refused, correctly, on its own checks

No summary line: **nothing calls `observer.close_trace()` at shutdown**. The
writer landed in the phase-2 slice; the shutdown hook did not. `readout.py`
refuses a trace without a summary because "zero desyncs" and "zero desyncs so
far" are different claims — so the tooling was right and the wiring was
incomplete. Gate 2 was not run: it needs the same summary.

Also found in the reader itself: `judge()` `continue`d past a missing summary
and therefore reported "0 verdicts, regimes []" for a trace holding 93 603 of
them. The verdict was right and the diagnostics were misleading, which is the
failure mode the tool exists to prevent. Fixed: collect facts first, judge
second.

### 13.4 The vehicle, and three boots spent on it

* **INT8-W8A8 does not boot on this build**: `NotImplementedError: No
  implemented int8_scaled_mm for current compute capability` (sgl_kernel
  0.3.21), surfacing inside the cuda-graph cold-build window so the visible
  error is a `ColdBuildWindowError`. The runsheet's vehicle is now the FP8
  reference arm.
* `deep_gemm` hard-requires `libnvrtc.so.13`, which the venv ships under
  `nvidia/cu13/lib` and does not put on the loader path.
* The runsheet carried vLLM flag spellings, and `--speculative-algorithm
  NEXTN` auto-chooses all three spec params and asserts the others are unset.

All corrected in `RUNSHEET_363_card_gates.md` §0a.

### 13.5 Ready-wait discipline, three ways to get it wrong

Every one of these cost a boot: `Failed to` matches the benign `Ignore import
error when loading ...` lines; `sigquit` matches `custom_sigquit_handler=None`
inside the `server_args=ServerArgs(...)` dump; and any `pgrep -f` / `pkill -f`
pattern naming the server also matches the checking shell (exit 144,
self-kill). Terminal-state patterns need an exclusion list and an anchor, and
process cleanup needs PIDs captured at launch rather than a pattern.

### 13.6 Still outstanding for gates 1+2

The observe path is now believed correct, but the run must be repeated: the
idle mapping changed what the classifier reports, so the 93 603 verdicts
describe the OLD behaviour. The next window needs `close_trace()` wired to
shutdown first — that is a desk task, not a card one.

---

## 14. Second card window (2026-08-01) — gates 1 and 2 RECORDED

Evidence file holds `desyncs_zero` and `f2_live_replay`; `act` now refuses
naming only `f3_bands_measured` and `f4_card_comparison`.

### 14.1 The run, and the idle fix confirmed

104 862 verdicts over three ranks, 0 unparsable, **0 desyncs**, spread on
103 230. Regimes `mixed` 104 793 / `decode_heavy` 69 with **seven
transitions** — real regime returns, which is what hysteresis and dwell are
judged on. The §13.2 fix holds: the previous window read `prefill_heavy` on
93 600 of 93 603 verdicts on the same kind of idle rig.

### 14.2 The summary line could never have existed

§13.3 blamed a missing shutdown hook, and the hook landed. It still did not
produce a summary, because the launcher shuts schedulers down with
`kill_process_tree(..., include_parent=False)` — **`child.kill()`, SIGKILL**.
Nothing runs under SIGKILL: not `finally`, not `atexit`, not a SIGTERM
handler. The hook covers three real paths and not the one the server actually
takes, so the gate as specified was unsatisfiable.

The contract was wrong, not the hook. Completeness is now proved from the
verdicts: each rank emits one verdict per interval in order, so a contiguous
round sequence per rank means nothing was lost between the first and the last
— exactly what the summary stood in for. Without a rank stamp the same proof
runs on the round MULTIPLICITY (N ranks produce each round exactly N times,
so a constant multiplicity over contiguous rounds proves both completeness
and the rank count). A summary is still accepted as the strongest ending; a
torn file still refuses.

Generalisable: **a completeness proof must not depend on the process getting
to run code at the end.** Anything that can be SIGKILLed needs its evidence
in the stream, not in a trailer.

### 14.3 Two more tooling defects the live trace found

* **Per-rank paths read the wrong attribute.** `scheduler.tp_rank` is `None`;
  the rank lives on `scheduler.ps.tp_rank`. All three ranks appended to one
  file. Fixed, and every verdict now carries a `rank` stamp.
* **The F2 replay outpaced the run it replayed.** It fed all three ranks'
  copies of each boundary through one sensor, so the hysteresis windows were
  effectively 3x shorter: 13 transitions replayed against 7 recorded, which
  the tool reported as NON-DETERMINISM in the classifier. It was the replay's
  own doing. Fixed by replaying one entry per boundary.

That is the third and fourth time in this task that a falsifier fired for the
wrong reason and the finding was in the instrument. The pattern is worth
stating: **when a falsifier accuses the subject, check the instrument first
when the instrument is newer than the subject.**

### 14.4 What is still untested

Peak occupancy **16.5 %** (85 585 of 519 670), up from 6.2 % and still far
from the 0.85 ascend mark. `PREFILL_HEAVY` never appeared at all, despite
eight concurrent 12 k-token prompts: at `window_rounds = 64` the burst's
prefill rounds are diluted by the decode and idle rounds around them, so
`prefill_share` never reached the provisional 0.35.

Two consequences. The admissibility axis remains unexercised, and gate 2's
`interlock_was_load_bearing` came back **false** — it passed on weak evidence
(the workload never approached the trap) and its own output says so. And
`enter_prefill = 0.35` is probably too high for this rig's round mix; that is
a gate-3 calibration input, not something to re-tune by hand.

---

## 15. Gate 3's band script, and what the first traces already say

`scripts/regime_gates/bands.py`. Deferred until real traces existed, because
window alignment across two boots was the part that would otherwise have been
designed against an imagined trace. The real ones decided it.

### 15.1 Alignment, decided by the data

The re-run wrote 34 954 consensus boundaries of which **28 were active**. So
aligning two runs by boundary index would compare one run's idle stretch
against the other's workload and report the difference as noise. Alignment is
therefore per signal and over the boundaries where that signal EXISTS: drop
the absent ones (an idle window is no measurement, not a zero), resample both
subsequences onto `min(n_a, n_b)` positions of a normalised timeline, band via
the controller's own `signal_band`. Resampling is nearest-neighbour, never
interpolating: an interpolated value is one the run never produced, and the
band would then be partly a property of the interpolation.

### 15.2 A band is only a floor if the arms are comparable

Three refusals, kept apart because the fixes differ: `UNDERPOWERED` (under 8
paired samples), `ARMS_DISSIMILAR` (the band is as large as one arm's own
internal movement — the pairing lined a quiet stretch up against a busy one),
and `UNREACHED` (the signal never approached the constant, so the regime it
gates cannot be entered at all).

`ARMS_DISSIMILAR` came directly from the fixture: splitting the real trace in
half gives an idle first half and a working second one, and their occupancy
"band" equals the whole occupancy range. Compared against the WITHIN-ARM swing
rather than the pooled range, so two arms that each hold steady at different
levels still report their real, reproducible bias as the band instead of
having it thrown away as misalignment.

### 15.3 The calibration finding, sharper than expected

`enter_prefill = 0.35` was flagged as probably too high for this rig's round
mix. The trace says something stronger: **`prefill_share` peaked at 0.000**
across all 34 954 boundaries. The threshold is not high — the signal never
moved.

Root cause, upstream of the constant: the hook reads `is_prefill_only` off
`running_batch` at the between-tick boundary, and that is the batch just
RETIRED. During a prefill burst the retired batch is the decode one, and the
prefill batch is built after the hook returns. So the `prefill` phase
essentially never reports, and §13.2's three-way mapping — correct as far as
it goes — is reading the wrong batch.

Deliberately NOT fixed in the gates window or here. Changing it changes what
the classifier emits, and gates 1 and 2 were recorded under the current
reading; re-tuning underneath recorded evidence is how a gate stops meaning
anything. It is the first desk task after gate 3 confirms the reading on two
boots, and gates 1+2 will need re-recording when it lands.

`spread_veto_pct = 25` gets the same verdict for an ordinary reason: the
measured spread peaked at 12.5 %, so the veto never fires.

---

## 16. Gate 3 run (2026-08-01): NOT PASSED, four verdicts, three predicted

Two identical FP8 boots, identical workload flags. Preceded by the #384
permanent fork-wheel reinstall (runbook §2.1), verified in both directions
including the can-fail proof: `require_int8_arm(..., available=False)` refuses
and names the wheel. The shadowing pypi `sgl-kernel 0.3.21` dist is gone from
CT999; only `sglang-kernel 0.4.4` remains, provenance sha256 matching the pin.

### 16.1 The result

`decode_share` band **0** over 29 paired samples, so `enter_decode = 0.90`
CLEARS. `enter_prefill`, `kv_ascend_mark` and `spread_veto_pct` all UNREACHED
— the three §6 predicted in advance, confirmed on two real boots and recorded
as found. Nothing was chased into existence and no constant was re-tuned.

The arms were reproducible to a degree worth stating: **both produced exactly
29 active boundaries**, peak occupancy 0.1648 against 0.1649, and all three
ranks of arm A recorded zero desyncs.

### 16.2 The two ARMS_DISSIMILAR results are a method finding

`occupancy` and `queued_prompt_tokens` are present on EVERY boundary — an idle
window reports occupancy `0.0`, a real value and not an absence. So §15.1's
"drop the absent samples" does not drop idle windows for these two, and the
arms' different idle lengths (19 402 vs 15 504 boundaries) pair a quiet
stretch of one against a busy stretch of the other.

The guard worked exactly as designed: it refused to publish a `0.1649`
"occupancy noise floor" that is really the entire signal range. Without it the
report would have shipped a number that looks like a measurement and is an
alignment artifact.

The fix is one line of method — restrict these two signals to ACTIVE
boundaries, as the shares already are. **Deliberately not applied in the same
breath as reading the result**: it is a method change motivated by looking at
the data, and changing the instrument to improve the answer it just gave is
the failure this project keeps naming. It should be made as its own decision,
stated, and the same two traces re-analysed — no cards needed.

### 16.3 A signal that is not stable across boots

`rank_ms_spread_pct` peaked at **0.61 %** here against **12.5 %** in the
re-run of §14 — same recipe, same workload, different boot. Its within-pair
band is 0.449 on a max of 0.614, i.e. large relative to the signal itself.
Whatever `spread_veto_pct` eventually becomes has to respect that, and the
current 25 is not merely unreached but an order of magnitude off the observed
range.

---

## 17. #388 — the retired-batch attribution, and the alignment method

Desk only, no cards. Two items: the defect §15.3 named and the method change
§16.2 deferred. The two archived gate-3 traces were re-analysed with the fixed
method; nothing was re-recorded and no constant was re-tuned.

### 17.1 Root cause: `is_prefill_only` is a request kind, not a phase

`ScheduleBatch.is_prefill_only` is `all(req.is_prefill_only)`, and
`Req.is_prefill_only` is `max_new_tokens == 0 and speculative_algorithm is
None`. It marks embedding- and scoring-shaped requests. It is **False on every
batch of every generating workload**, and forced False whenever spec is on —
which is every recipe this rig runs.

So the hook's three-way mapping could only ever emit `idle` (empty running
batch) or `decode`. `prefill_share` was not low; it was unreachable. 0.000
across all 34 954 boundaries of §14 and across both gate-3 arms of §16, 29
active boundaries each — two independent boots agreeing on a structural zero.

The name is the trap, and the timing compounds it: by the time the hook runs,
the top of `get_next_batch_to_run` has already merged the finished extend
batch into `running_batch`, so during a prefill burst the batch the hook
inspects is the decode batch. Had the flag meant what it sounds like, it would
still have been read one merge too late.

### 17.2 The choice: attribute the boundary to the batch that RAN

`managers/regime_runtime.py:phase_of_last_batch`, called from
`managers/scheduler.py` with `last_batch`. Three reasons, one of them a hard
constraint:

* **The next batch does not exist yet.** `get_new_batch_prefill` runs after
  the hook. Reading the next batch's composition means moving the hook behind
  batch selection, which gives up the between-tick placement #287 and #364
  share — previous forward retired, next batch not chosen, no captured graph
  able to replay (#52/#53). That placement is *why* the device timing read
  here is a completed measurement.
* **`rank_forward_ms` is the just-retired forward's device time.** Phase and
  timing on one record now describe one event. Pairing this boundary's timing
  with the next batch's composition would be a new mismatch, not a fix.
* **Nothing is lost or double-counted.** The batch built after the hook is
  `last_batch` at the next boundary, so every dispatched forward is attributed
  exactly once, one boundary later. The classifier reads a *share over a
  window*; a uniform one-boundary shift does not move a share. Pinned by a
  test that drives a 27-tick mixed plan and counts attributions against
  dispatches.

`TARGET_VERIFY` is excluded from prefill even though `ForwardMode.is_extend()`
returns True for it: a spec verify is the decode lane's forward, and counting
it as prefill would report the reference NEXTN decode drain — the recipe both
gate-3 arms booted — as a prefill burst. `DLLM_EXTEND` is excluded for the
same reason in a different family: a denoising step emits, it does not ingest.

The falsifier keeps the **pre-#388 attribution transcribed in the test** and
drives both through one synthetic scheduler loop that reproduces the
merge-then-hook-then-select order. The old one reports 0 % prefill on a window
of nothing but prefill forwards; the new one reports 1.0. A mixed window reads
0.5 under the fix and 0.0 under the old code — the two shapes were
indistinguishable in every trace recorded so far.

### 17.3 The alignment method, applied and re-analysed

`scripts/regime_gates/bands.py`: `occupancy` and `queued_prompt_tokens` are
now restricted to ACTIVE boundaries (`ACTIVE_ONLY_SIGNALS`), the same
subsequence the shares already get. The report marks restricted signals with
`*` and prints each arm's active count, because a restriction the analysis
applied has to be visible in the analysis's own output.

`UNDERPOWERED` was added to the blocking set. The runsheet's table already
said only `CLEARS` is a pass, and the omission became reachable the moment
these two signals stopped drawing thousands of idle samples.

Re-analysis of the SAME two archived traces (no cards):

| signal | before | after |
|---|---|---|
| `occupancy` | band 0.164864, 15 504 paired, **ARMS_DISSIMILAR** | band **0.0823**, 29 paired, **OK** |
| `queued_prompt_tokens` | band 74 802, 15 504 paired, **ARMS_DISSIMILAR** | band **64 116**, 29 paired, **OK** |

`ARMS_DISSIMILAR` is gone: it was an alignment artifact, exactly as §16.2
predicted. The constants those two signals carry:

* `kv_ascend_mark = 0.85` → still **UNREACHED**, now for the honest reason —
  occupancy genuinely peaked at 0.1649, and the comparability failure is no
  longer standing in front of that reading.
* `PRESTAGE_SINGLE_PROMPT_TOKENS = 8192` → **NO_GAP** (reached: the queue mass
  peaked at 74 802). Worth carrying, though the report does not judge it: the
  signal's band is **64 116 on a peak of 74 802**, i.e. 86 % of its own range,
  and it clears the ARMS_DISSIMILAR guard by about five percent. A bare
  threshold on a signal that noisy is not something to set from these two
  boots.

Gate 3 is still **NOT PASSED**, and the blocking list is now exactly the three
the runsheet predicted in advance: `enter_prefill`, `kv_ascend_mark`,
`spread_veto_pct`, all UNREACHED. `enter_decode` still CLEARS on a band of 0.

`rank_ms_spread_pct` was deliberately left unrestricted. It is also present
through idle stretches — the #252 sensing reports the LAST measured forward,
so the value goes stale rather than absent — but that is a second method
change with its own argument to make, and it is not the one gate 3 identified.
Recorded so the omission is not mistaken for a claim that the signal is clean.
The bands script's own smoke now pins it: on the split-halves fixture it is
`rank_ms_spread_pct`, not `occupancy`, that trips the ARMS_DISSIMILAR guard.

### 17.4 What this invalidates

Everything recorded under the old attribution describes the old classifier.
Gates 1 and 2 were recorded in §14 and are now **stale**: `prefill_share` can
move for the first time, so the regime histogram, the transition count and the
F2 replay all have to be re-recorded before gate 4. §16's gate-3 numbers stand
for the four signals whose values the fix does not touch, but the
`enter_prefill` verdict has to be re-measured too — UNREACHED was a reading of
a signal that could not move.

`spread_veto_pct` stays at 25 in the tree. Per §16.3 the observed range is
0.61 % to 12.5 % across boots, so 25 is an order of magnitude off and the
signal is not boot-stable at that magnitude. Whatever replaces it must respect
both facts, and neither is a licence to set the number from two boots.

---

## 18. Gates re-recorded on the #388 classifier (2026-08-01)

Three boots, §1 recipe, only `--regime-trace` differing; workload identical on
all three at `--repeats 3`.

### 18.1 Early check passed, and gates 1+2 are re-recorded

`prefill_share` moves: the observe log reported 100 % / 50 % / 0 % where the
old attribution was 0.000 across 34 954 boundaries.

| | old (broken attribution) | new (#388) |
|---|---|---|
| verdicts | 104 862 | 147 105 (3 per-rank files) |
| desyncs | 0 | **0** |
| regimes | mixed, decode_heavy | mixed, decode_heavy, **prefill_heavy** |
| transitions | 7 | 36 |
| `prefill_share` max | 0.000 | **1.000** (n = 55) |

Zero desyncs holds on the fixed classifier, over more verdicts and a regime
the old attribution structurally could not produce. Gate 2 re-recorded on the
same trace and passes.

### 18.2 Gate 3 fails on every signal, and it is a method finding — again

Arm A 24 111 boundaries / **41 active**; arm B 24 762 / **56 active**. Every
signal came back `ARMS_DISSIMILAR`.

With the fixed attribution the share signals are near-BINARY at
`window_rounds = 64`: a boundary's window is essentially all-prefill or
all-decode, so `prefill_share` swings the full 0..1 and the within-arm
movement is 1. A pointwise A-vs-A band on a binary sequence is 1 whenever the
two runs' bursts do not land on the same boundary indices — and they cannot,
because the same workload produced 41 active boundaries in one arm and 56 in
the other, 37 % apart, which is real run-to-run variation in scheduler round
counts.

The guard is right that these arms are not comparable POINTWISE, and that is
not evidence the rig is noisy. **For a near-binary or bursty signal the
meaningful A-vs-A statistic is distributional** — the fraction of active
boundaries in each state, or the per-phase dwell distribution — not a
pointwise difference. §15.1's alignment was designed against traces in which
the shares barely moved; the fix that made them move also invalidated the
statistic chosen for them.

Not changed here, for the third time and the same reason: altering the
instrument in the same breath as reading its result is how a measurement
becomes a decision about itself. It needs no cards — four archived traces
re-analyse in seconds.

### 18.3 Carried unchanged

`spread_veto_pct = 25` against a measured max of **0.68** here, 0.61 and 12.5
in the two earlier windows: an order of magnitude off the observed range and
boot-unstable across all three. Recorded, not re-tuned.

### 18.4 One tooling bug, found and fixed in-window

`readout.py` aggregated `ranks_seen` as the MAX over trace files instead of
the UNION across them, so three per-rank files reported "1 rank" for a 3-rank
boot and gate 1 refused a complete group record. The per-rank layout is newer
than the check that consumes it — §14.3's instrument-first rule, fifth
instance.

---

## 19. The distributional band, and the four archived traces re-analysed

§18.2 left gate 3 blocked on a method finding for the third window running.
This is the fix, built at the desk on the four archived arms (the pre-#388
pair `2026-08-01_363_gate3_{a,b}` and the post-#388 pair
`2026-08-01_363_gate3_388_{a,b}`), no cards.

### 19.1 The statistic, chosen per signal

The pointwise A-vs-A band assumes the value at position `i` describes the same
thing in both runs. That holds for a slowly-moving signal and fails for a
bursty one, so the choice is made per signal against exactly that criterion —
**is the value at a given active boundary reproducible across boots, or only
its rate?** — and not blanket-replaced.

| signal | statistic | why |
|---|---|---|
| `prefill_share` | distributional | near-binary at `window_rounds = 64` |
| `decode_share` | distributional | near-binary, the complement |
| `occupancy` | distributional | near zero between bursts, a spike while one drains |
| `queued_prompt_tokens` | distributional | median 0, peak 74 802 |
| `rank_ms_spread_pct` | **pointwise, kept** | a within-boundary ratio across ranks, continuous, not a burst counter |

For a distributional signal the compared quantities are per-run SUMMARIES and
the band is `|summary_a - summary_b|` on those: the **peak** (the summary the
reachability check reads, and the signal-level band in the table), the **duty
cycle** at each constant's own value (the fraction of active boundaries at or
above it), and the p50/p75/p90 **quantiles** for shape. A constant is then
judged on its own crossing rate: it clears when
`mean(duty_a, duty_b) > 2 x |duty_a - duty_b|`, which is the direct form of
the question the gate asks — *would this signal cross the threshold
consistently across boots?*

Deliberately not used: the max quantile shift ("sort both series and
subtract"). On a near-binary signal it is maximally sensitive exactly where
the distribution is flat-then-jumps — on the #388 pair it reads 0.71 for
`prefill_share`, which measures the binariness, not the noise.

The hysteresis gap is still reported for a distributional signal, at both the
enter and the exit value, but does not decide the verdict. The gap lives in
signal units and the reproducible quantity does not; and converting it into a
duty difference would punish the good case, because a decisive signal puts
almost no windows inside the hysteresis interval and that is the hysteresis
being unnecessary rather than a finding.

### 19.2 Both guards survive, re-stated

`UNDERPOWERED` is unchanged in meaning: fewer than `MIN_PAIRED_SAMPLES = 8`
active boundaries in the smaller arm.

`ARMS_DISSIMILAR` keeps the within-arm rule on the pointwise path and gains a
distributional form: the two runs' duty cycles, **at the constants' own
thresholds**, differ by more than a two-proportion difference of
`DISSIMILAR_Z = 2.576` standard errors. Evaluated at those thresholds and
nowhere else on purpose — a sup-over-all-thresholds (Kolmogorov-Smirnov) form
flags two steady arms held at slightly different levels as totally separated,
and that offset is a real reproducible bias that IS the band. The gate asks
whether the constants' decisions reproduce; the guard asks the same question
of the same thresholds. The active boundaries are autocorrelated, so the
nominal level is optimistic; 1 % rather than 5 % takes some of that back.

### 19.3 One more signal restricted, and what it bought

`rank_ms_spread_pct` was the last signal read on ALL boundaries. It reports
the LAST measured forward, so it goes stale through an idle stretch rather
than absent — the same defect §17.3 fixed for `occupancy` and
`queued_prompt_tokens`, wearing a different hat, and recorded there as an
omission rather than a claim that the signal was clean. Restricted, its
pointwise band on the #388 pair drops from 0.6815 (the full observed range,
flagged `ARMS_DISSIMILAR`) to 0.5813 against a within-arm swing of 0.675.

### 19.4 The falsifier

`bands.py --falsify`, three cases, each able to fail:

| case | old statistic | new statistic |
|---|---|---|
| (i) same duty cycle (10/40), bursts shifted from indices 0-9 to 20-29 | band 1.0, `ARMS_DISSIMILAR` | band 0, duty 0.25 vs 0.25, `OK`, `enter_prefill` CLEARS |
| (ii) genuinely different duty cycles (10/40 vs 30/40) | — | duty 0.25 vs 0.75, z = 4.47 > 2.576, `ARMS_DISSIMILAR` |
| (iii) barely-moving signal (0.50 vs 0.53, flat) | band 0.03, `OK` | band 0.03, `OK` |

Case (i) is the false alarm of record and it is reproduced against code that
still runs: the pointwise path is live for `rank_ms_spread_pct`, so the
comparison is not against a reconstruction. Case (iii) is the regression pin
against the traces the first version was designed on.

### 19.5 The four arms, re-analysed

Per signal, active boundaries and band (`*` = pointwise):

| signal | pre-#388 A/B active | pre band | post-#388 A/B active | post band |
|---|---|---|---|---|
| `prefill_share` | 29 / 29 | 0 | 41 / 56 | 0 |
| `decode_share` | 29 / 29 | 0 | 41 / 56 | 0 |
| `occupancy` | 29 / 29 | 6.16e-05 | 41 / 56 | 1.19e-04 |
| `queued_prompt_tokens` | 29 / 29 | 0 | 41 / 56 | 0 |
| `rank_ms_spread_pct`* | 29 / 29 | 0.283 | 41 / 56 | 0.581 |

Every signal `OK` on both pairs: no `ARMS_DISSIMILAR`, no `UNDERPOWERED`.

Per constant:

| constant | pre-#388, old | pre-#388, new | post-#388, old | post-#388, new |
|---|---|---|---|---|
| `enter_prefill` | UNREACHED | UNREACHED | ARMS_DISSIMILAR | **CLEARS** |
| `enter_decode` | CLEARS | CLEARS | ARMS_DISSIMILAR | **CLEARS** |
| `kv_ascend_mark` | UNREACHED | UNREACHED | UNREACHED | UNREACHED |
| `spread_veto_pct` | UNREACHED | UNREACHED | UNREACHED | UNREACHED |
| `PRESTAGE_SINGLE_PROMPT_TOKENS` | NO_GAP | CLEARS | ARMS_DISSIMILAR | **CLEARS** |

The pre-#388 pair's blocking list is **unchanged** by the method change
(`enter_prefill`, `kv_ascend_mark`, `spread_veto_pct`, all UNREACHED) — which
is the regression pin on real data, since that pair is the barely-moving-share
regime the pointwise statistic was designed against. The post-#388 pair's
blocking list goes from five entries to **two**, and both survivors are
reachability findings already on the record rather than method artefacts.
Identical verdicts on all three per-rank trace files.

### 19.6 What is left, and what was deliberately not touched

Gate 3 does NOT pass. What blocks it:

* `kv_ascend_mark = 0.85` against a 0.1649 peak occupancy — #287's mark, and
  reported for information only.
* `spread_veto_pct = 25` against a 0.68 peak. §18.3's finding, still standing.

Neither is fixable by a better statistic; both need either a heavier workload
or a calibration decision. Calibration is a separate decision with its own
evidence rules, so no threshold constant was changed here — the observed
ranges are recorded (`rank_ms_spread_pct` 0.0018 .. 0.679 over the #388 pair,
0.0027 .. 0.614 over the pre-#388 pair) and the constants stand.

The two passing distributional verdicts are thin and are reported as such:
`enter_prefill` clears by 1.29x and `PRESTAGE_SINGLE_PROMPT_TOKENS` by 1.14x.
A duty cycle estimated from 41 active windows has a standard error near 0.07,
which is most of the disagreement that had to be cleared. The fix for that is
a longer or busier workload, not a constant.

---

## 20. 2026-08-03 decisions

Three operator decisions on the weight-mover half of the controller (§4.4,
"weight cut: none — restart"), recorded per the fork's standing rule that a
design decision lives in this file, not only in a chat transcript.

### 20.1 WORTH-IT AUTOCHECK

The controller is not authorized to attempt a layout switch by a flag; it is
authorized by a computation. For every (checkpoint format) x (model) x (rig)
point, the planner decides FROM THE MEASURED PHASE TABLE whether a stage flip
beats its own switch cost at all — the same measured-gain-vs-band discipline
§4.3 already applies to a single stage's admission, generalised to the pair.
No manual flag is a precondition for either answer, and a no-op verdict is
stated with its reason exactly as an acting one is: "one layout, checked, it
does not pay" is the autocheck's output on that point, not silence.

Canon examples, both read off the #424 phase-layout table (§4.1,
`comparison_table.md` / `RESULTS.md` §2 of
`/spinning/gpu-battery-results/2026-08-02_424_phase_record_bench/`):

* **INT8-27B: one layout.** The decode layout beats the prefill layout even ON
  PREFILL — 1890.6 vs 1847.2 tok/s at s=1, a -2.3 % cost to concentrate — so
  every axis measured there says do-not-switch and the autocheck returns that
  verdict with the comparison as its reason.
* **FP8-27B: a real divergence.** +24.1 % prefill-layout gain (1231.7 ->
  1528.9 tok/s s=1) against a -32.8 % decode-layout cost (125.1 -> 84.1 tok/s
  bs=1) on the same #424 window — the autocheck returns "switch", carrying
  both directions of the divergence in the reason.

**Evidence tension, flagged rather than resolved here.** The INT8 "one
layout" canon rests on #424's `10,1,1` prefill arm, and `NOTE_433_int8_prefill_vector.md`
(§1, §5) documents that this vector was a *manually re-pinned* pair borrowed
from a different task's corridor-safety decision — never re-solved by the
phase-prefill optimiser at #424's own context length and reserve. NOTE_433's
own desk re-solve found a real, if narrow, optimum at that point (`8,1,1`,
+8.5 % predicted) and its verdict is explicit: "the one-layout recommendation
... is not confirmed." The GPU confirmation this note called for then ran
(`/root/addendum_435.md`, `2026-08-02_435_coupling_fp8bar1`): the
optimiser-matched vector plus `--rank-kv-ratio capacity` produced a technical
flip on capacity (443 840 tokens, above the decode arm's 431 360) but an
explicitly **weak** flip on throughput — both prefill probes moved the wrong
way (-8.5 % / -4.3 %), the governing A-vs-A floor was that boot's own 13.0 %
rather than #424's 3.0 %, and the tighter `bench.sh` instrument disagreed
outright (-9.8 % to -11.5 % on decode). The addendum's own conclusion: "What
it does not license: retiring the decode layout as the one-layout default."
So the canon this section states (decode-only, do not switch) is the standing
decision and is not contradicted by #435 — but it rests on a weaker
foundation than "switching does not pay" alone suggests, and a reader
re-deriving the INT8 autocheck output from first principles should read
NOTE_433 and addendum_435 first rather than the #424 table alone.

### 20.2 WEIGHT MOVER = diff spill within the same TP group (not a #329 re-form)

A layout switch under a fixed rank set is **not** a #329 elastic-membership
event. #329's 12-20 s silence budget (`DESIGN_329_elastic_world_membership.md`
§1: "12-20 s of silence for a membership change") is scoped to membership
*changes* — quiesce, snapshot, RE-FORM, restore, a different member set. A
layout flip keeps the same ranks, the same process group, the same
communicators: the collective geometry does not change, so there is no NCCL
rebuild.

What moves is a **diff**, not a re-form: only the slabs whose ownership
changes between the two layouts move. For the 27B-INT8 class the estimate is
~1-2 s via host staging, overlappable with the round(s) around the switch.
Repack is charged only on the affected slabs and differs by format — INT8
per-channel requant is cheap, a Marlin-packed layout costs more per moved
slab. The KV vector moves through the existing #297 phase-boundary reshard
actuator (§4.4 already lists it as the KV axis's mover), whose own measured
target is < 1 s for the delta (`DESIGN_297_kv_resharding.md`: "move duration
(measured, target < 1 s for the delta)"); #363 reuses that actuator as-is, no
new KV mover is invented for the switch.

The #329 budget is unchanged by this: it applies only when the rank set
itself changes, never to a same-group layout flip.

### 20.3 GRAPH PRE-CAPTURE (primary route; lazy recapture demoted to fallback)

User directive, 2026-08-03: capture **all** layout families at program start
rather than recapturing on the fly. Boot time is where the ANALYSE §Sequencing
estimate of 3-6 s per flip (`ANALYSE_363_dynamic_regime_controller.md`:
"moving the MLP delta ... order 3-6 s including quantized repack and
re-capture from pre-staged pools (#102/#286)") was always going to be paid —
this decision moves the re-capture component of it off the live switch path
entirely, an option that document's own phrasing already named ("re-capture
from pre-staged pools").

The inactive family's graph-capture pools are offloaded as the #286
"graph rungs" offload class (`ANALYSE_363_dynamic_regime_controller.md:121`:
"#286: graph rungs, drafter, lane workspaces, cold lane") — VRAM -> host RAM
or disk, the same short-term offload register the fork already routes
graph-capture pools and IO buffers through. The VA reservation survives via
the #93 VMM remap (the same physical-remap machinery §"Physics of a flip"
already names as an existing piece: "VMM remap (#93)"), so a captured graph
for the inactive family stays **valid without recapture** when its family
reactivates — only its backing pages moved, not its addresses.

At switch time the visible cost is: weight diff spill (§20.2, ~1-2 s) plus a
graph-state reload. The per-state size comes from #102's own measured figure
for a full independent capture state — "1,5 -> 0,3 GB je State"
(`INTEGRATION_R3_VALIDATION.md:7601`) — carried over here by analogy from the
spec-ladder rung case to the layout-family case; the ~25 ms host-link figure
that follows from it is a **projection, not yet measured on this mechanism**,
and is flagged as provisional the same way §3.4's constants are, pending a
card window that actually times a reload. Visible switch time under this
route is therefore seconds-class with no eager gap, not the 3-6 s of a cold
recapture.

**Named precondition.** Capturing family B requires family-B's weights mapped
at their VAs at capture time — a VMM reservation plus a temporary mapping
during boot, held only long enough for that family's graphs to record.

**Residency ladder, coupled to the #287 pressure ladder.** Whether a family's
graphs and weights stay resident, get partially evicted, or fall back to lazy
recapture is not a fixed policy — it is driven by the same occupancy marks
§3.2 already gives `KV_PRESSURE` priority over:

* **RUNG 0 (KV pressure low, below the #287 descend mark).** Both weight
  layouts and both graph families stay fully resident. A switch is a pointer
  flip plus the §20.2 KV delta move (< 1 s via #297) — near-zero visible cost,
  no copying, because each layout lives at its own addresses with graphs
  already captured against them. This rung is cheaper than its own byte count
  suggests because the two layouts' shards **overlap** when the unit ordering
  is kept consistent: for the INT8-27B `10,1,1` <-> even-split pair, the big
  card's decode shard is a *prefix* of its prefill shard, so the 5090 carries
  zero extra bytes; only the two smaller cards hold disjoint ranges (union
  ~5/12 against ~4/12 of the MLP each). Total dual-residency overhead for that
  pair is ~3 GB, and it is paid only while the KV pool does not need the
  space back.
* **RUNG 1 (KV pressure rising).** The inactive layout's *non-shared* slabs —
  the disjoint ranges above, not the shared prefix — are evicted through the
  #286 graph-rungs/offload class **before** any KV admission is refused. A
  switch under this rung is the diff reload (~1-2 s, §20.2) plus the
  graph-state reload above (~25 ms, provisional).
* **RUNG 2 (fallback).** A family that was never pre-captured — outside the
  boot's declared set — falls back to lazy recapture, at the original 3-6 s
  cost. This is the only rung where the ANALYSE §Sequencing estimate still
  applies unamortised.

Rung transitions are driven by the #287 pressure ladder's own marks, not by a
second occupancy sensor, for the same reason §7.3 gives KV_PRESSURE joint
ownership rather than a second controller: one arbiter on one axis. Victim
selection *within* a rung — which slabs of the inactive family actually get
evicted first when RUNG 1 fires — is not a rung-1-local policy either: it
selects victims via the `DESIGN_407_memtier_registry.md` global importance
ladder ("Global eviction doctrine"), of which this residency ladder is one
named instance, not a parallel mechanism. And the §20.1 WORTH-IT AUTOCHECK
still gates whether any of this — pre-capture, dual residency, rung eviction —
is engaged for a given (format, model, rig) point at all; a point whose
autocheck says "one layout" never allocates the second family's capture pool
in the first place.

**Planner consequence, named explicitly.** Because RUNG 0's cost is set by
how much the layout pair's shards overlap, layout **pairs** should be solved
with maximal shard overlap (the prefix/nested property above) as a secondary
objective on top of each layout's own per-rank speed. This minimises the
dual-residency bytes and the switch diff size at once — the same overlap that
makes RUNG 0 cheap also makes a RUNG 1 diff smaller, because a diff is
exactly the non-overlapping remainder.

**Measurement duty when built.** Report the switch-time decomposition — diff
move / repack / KV move / graph-state reload — separately, A-vs-A on the
target layout, per the #360 standard already governing every other constant
in this document (§8 F3-F4). None of the numbers in this section are card
measurements; they are the physics estimate and the #102 analogy that the
next card window is expected to replace with real per-component numbers.

Cross-reference: `DESIGN_121_dual_group_runtime.md` §11.12 (the dual-group
lane's family-neutral vs. family-specific building blocks) is the nearest
existing account of what "the same mechanism across model families" already
means in this fork's runtime, and the #286 register (§2.6 of
`DESIGN_407_memtier_registry.md`) is the offload class this rung's eviction
routes through.

### 20.4 SLICE 1 BUILT (2026-08-03) — the decision layer, and only that

`planner/regime_switch.py` + `test/registered/unit/planner/test_regime_switch_363.py`
(65 hermetic tests, five executed can-fail arms). WAVE 1 item 4 of
`ROADMAP_456_matrix_execution.md`. What moved from decided-on-paper to built,
and what still cannot execute:

**What decides today.**

* `autocheck(table, ...)` returns a named verdict object — `NO_SWITCH`,
  `SWITCH_KV_ONLY`, `SWITCH_FULL`, `UNPRICEABLE` — with the reason and every
  number it used, from a `PhaseTable` whose cells are `cost_model.Rate`s and
  therefore carry `measured|estimate|absent`. Four steps in order:
  completeness (all four cells of the 2x2 priced, or `UNPRICEABLE` naming the
  missing arms), dominance against the table's own A-vs-A floor, the
  same-weight-vector `SWITCH_KV_ONLY` case, and finally a per-round pricing
  of the divergence against the switch cost at the rung the ledger reports.
  A no-op verdict is an output, exactly as §20.1 demands; the canon-INT8
  shape returns "one layout, checked ... there is nothing to pay for" and the
  canon-FP8 shape returns the +24.1 % / -32.8 % divergence in its reason.
  The autocheck knows nothing about the §20.1 canon sentence or about
  NOTE_433 / addendum_435: it decides from the table it is handed, which is
  the only way that evidence tension can be settled by re-running an arm
  rather than by editing a document.
* `layout_overlap()` computes the per-rank shard-range overlap of a pair,
  `residency_rung()` does the RUNG 0/1/2 ledger arithmetic, and
  `solve_layout_pair()` implements the "planner consequence" as a bounded
  secondary objective (default tolerance 2.0 %, deliberately below the
  reference rig's 4.2 % measured A-vs-A floor, so overlap may only break
  ties, never knowingly buy a slower layout).
* Surfaces: `planner.plan(..., regime_phase_table=...)` fills
  `PlanResult.regime` from the plan's OWN capacity report; the planner CLI
  gains `--regime-phase-table` / `--regime-workload` /
  `--regime-not-pre-captured` and prints the verdict in both text and
  `--json`; `solver_api.regime_switch_payload()` is the
  `POST /api/regime_switch` surface (webui binding left to the UI strand, per
  that module's convention). All opt-in: with no phase table the plan answer
  is unchanged field for field.

**Two corrections this section's own numbers needed, found by building it.**

1. **§20.3 mixes two baselines for "extra bytes".** Run on Qwen3.6-27B's real
   544-unit quant-group MLP grid, the `10,1,1` <-> even pair partitions to
   453/46/45 against 182/181/181. The big card's decode shard `[0,182)` IS a
   strict prefix of its prefill shard `[0,453)`, so it costs zero extra
   *against the larger of the two layouts* — but against the ACTIVE (decode)
   layout, which is what a ledger must charge, that same card owes 271 of 544
   units. The two readings differ by 317 vs 46 units, about 7x. The module
   therefore reports `extra_vs_active` (the ledger item) and
   `extra_vs_larger` (§20.3's "zero extra") separately and calls neither "the"
   cost.
2. **"the two smaller cards hold disjoint ranges ... each" over-counts by one
   card.** Only the MIDDLE card is disjoint — union 227/544 = 0.4173 against
   the decode layout's 181/544 = 0.3327, which is the section's "~5/12
   against ~4/12" to two decimals. The third card's prefill range `[499,544)`
   is a SUFFIX nested inside its decode range `[363,544)`, so it holds no
   extra bytes at all. The "~3 GB total" figure is consistent with the
   vs-active reading at some MLP byte masses and not at others; it is left
   standing as the estimate it was labelled as, now with the unit geometry
   pinned by a test instead.

**What still cannot execute — BOOT-PENDING in full.** Nothing in this build
flips a pointer, spills a diff, pre-captures a family or evicts a slab. The
verdict object carries `executes: false` and says so in its own text
rendering. Specifically still absent: the §20.2 diff-spill mover, the §20.3
boot-time pre-capture of all layout families (which needs #286, WAVE 1 item
5), the VMM remap wiring for an inactive family's pools, and the runtime
consumption of the verdict — today it is reported to a planner reader, not to
a controller. The switch-cost constants remain the §20.2 physics estimate and
the #102 analogy; only the KV delta inherits a measurement (#297). §20.3's
measurement duty is unchanged and unmet.
