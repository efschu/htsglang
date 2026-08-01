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
