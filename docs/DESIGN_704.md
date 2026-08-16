# DESIGN 704 — prefill layout ladder + PP-KV decoupling

Status: design settled; Slice 0 landed green; converged onto Slot-2's canonical
solver (rev5). Worktree: `/spinning/wt-704-ladder`, branch
`feat/704-prefill-ladder`, merged with `c5afff7a8d` (Slot-2 rev5).

---

## ERRATA — review gate, 2026-08-16. These supersede the body below.

The pre-boot review gate adjudicated the functional-form dispute **in this
document's favour** (the pool's token term lives on attention layers only,
proven white-box at the allocator and byte-exact against the boot log's K
sizes). It also found **four defects of mine**. Where the body still shows an
old number, this section wins.

**E1 — the KV cell was wrong, because it was FITTED.** §2.2 read the cell as
"4096 B (bf16)". The shipped config is `kv_cache_dtype='fp8_e4m3'`; the true
cell is **2048 B** for K+V (1024 B for K), byte-exact against the `[28,20,16]`
boot log. I derived 4096 by fitting against an observed pool, which also
produced the bogus "0.83 of observed" fudge factor. **Every pool number in the
body that predates this correction is wrong by up to 2x.** Fixed in code: the
cell is now consumed from config via
`pp_cut.kv_mib_per_token_per_attn_layer_from_config()`, unknown dtypes are
refused rather than defaulted, and `test_kv_cell_from_config_704.py`
reproduces the logged K sizes with zero free parameters.

**E2 — the binding rank was wrong.** I claimed the incumbent "binds on rank1".
The boot log shows **PP2 binds at 436,766** (PP1 463,406; PP0 clipped at
550,000). The functional form was right; my free-bytes vector was not — and
that error is the direct cause of E3.

**E3 — `[33,15,16]` is RETRACTED as a boot arm.** My claimed pool of 457,604
was **impossible**: `[33,15,16]` leaves rank2 byte-identical to the incumbent's
(layers 48-63), whose measured cap is 436,766, so no cut keeping that rank2 can
exceed it. I over-predicted an *unchanged* rank's measured capacity by 4.8%.
Expected actual is ~387k, about **-11% vs incumbent**. Metal confirmed the
class of error independently: `[32,16,16]` failed its pool gate at 416,796.
The "discriminating experiment" justification is also void — the white-box
adjudication settled the rule without a boot, so spending a window to re-prove
it was waste.

**E4 — GDN residency was under-charged 2.6x.** The full per-GDN-layer figure is
**50.85 MiB**, not 19.5: temporal_state 19.5 + **speculative
intermediate_ssm_state_cache 30.0** (5 spec slots x 4 draft tokens) +
conv_state 0.762 + intermediate_conv_window 0.586. This is material to every
rung that moves GDN layers, and it **falsified a structural claim of mine** —
see §3.7, where "same attention profile ⇒ identical pool ⇒ the deeper rung
strictly dominates" was an artifact of the missing 30 MiB term. Corrected, such
rungs differ by their GDN residency (~3,250 tokens at 8 attention layers): a
small real trade, not domination.

**The structural consequence for the ladder** (this is the load-bearing
finding, and it survives all four corrections): **no cut that keeps rank2 =
layers 48-63 can beat the incumbent pool.** Rank2's 436,766 is a hard ceiling
under the min-rule. Pool-positive rungs must *shrink rank2's attention count*
(e.g. a 12-layer rank2 = 3 attention layers), or wait for Part B. A
prefill-speed arm at pool-parity-or-below is a user trade decision, not a
pool-gate pass.

**The gate's binding lesson, which bit all three strands in one day:** a model
calibrated against the incumbent silently absorbs exactly the layout-varying
terms (attention counts, arming floors, GDN residency) that a new layout then
exposes. Consume every term from config or instruments; fit nothing.

### Convergence and ownership

Slot-2's rev5 (`c5afff7a8d`) is now the **single canonical pool solver** and is
merged into this branch. My duplicate `FamilyPoolModel` is **deleted**; rev5's
`PhasePoolModel` (with its per-layout `arming_floor_mib` and mamba terms) is
the only pool model. Division of ownership: **Slot-2 leads the solver; I own
`layout_ladder.py`, `ladder_controller.py` and the arena model**, which now
consume rev5 rather than reimplementing it. My two additions to `pp_cut.py` are
the config-derived KV cell (E1) and `decoupled_phase_pool()` (Part B's
projection), neither of which duplicates rev5.

### Consequence for every pool number below

Rev5's arming floor is **per layout**, and its own docstring records the gap:
the #676 solver derives the floor from a *measured seam draw*, so **a cut that
has never booted has no solved floor**. A ladder is mostly unbooted rungs, so
every rung's predicted pool carries roughly **±500 MiB → ±32,000 tokens (~7%)**
of floor uncertainty. `LadderInputs` therefore *requires* an
`arming_floor_for(counts)` provider and refuses to be constructed without one —
a constant floor is precisely the E3 error. **No rung's predicted pool in this
document is a boot gate on its own.**

---

## 0. The thesis in one paragraph

Under PP prefill the KV pool is layer-sharded, so a rank's capacity is divided
by its own attention-layer count. Deepening the cut (piling layers onto the
fast rank) shortens the pipeline and shrinks the pool: **speed and capacity
oppose along the cut axis** (#702, proven on metal by two `[42,11,11]` OOMs).
That opposition, however, only *binds when the pool is full*. At low fill the
capacity constraint is slack and the speed is free. Part A turns the cut into a
pressure ladder that harvests exactly that slack. Part B removes the
opposition itself: once the attention layers' KV is token-sharded across all
ranks, the pool stops depending on the cut at all, and the deepest rung becomes
permanently affordable. **Part A is the near-term, metal-provable win and the
control surface; Part B is what removes Part A's ceiling.**

> **WITHDRAWN TABLE (E1/E3).** A table stood here giving pool figures for
> `[31,17,16]` (518,433), `[41,11,12]` (254,687) and a decoupled `[44,10,10]`
> (539,017, "+23.9%"). Every one of those pool values was computed with the
> **fitted bf16 cell** and without per-layout arming floors or the correct GDN
> residency. They are wrong by up to 2x and are withdrawn rather than
> patched — re-deriving them is Slot-2's canonical solver's job, gated on the
> retro-prediction check in §6, and quoting a corrected-looking number here
> before that gate passes would repeat the exact mistake.
>
> The only pool figures in this document that are *evidence* are the measured
> ones: **436,766 / 435,822 / 434,878** for `[28,20,16]` (PP2 binding) and
> **416,796** for `[32,16,16]`.

What survives the withdrawal is the **shape**, which is what the design rests
on and which no pool constant changes:

| layout | pipelined prefill | pool, coupled | pool, decoupled |
|---|---|---|---|
| shallow rung | 1.00x | highest | *cut-independent* |
| deep rung | up to ~2x | collapses (÷ own attention count) | *cut-independent* |

Coupled, speed and capacity oppose along the cut because the deepest rank's
capacity is divided by its own attention count. Decoupled, the pool does not
depend on the cut at all, because total weight bytes are constant however the
layers are split. That asymmetry is the entire argument, and it is structural.

Decoupling is not a marginal improvement to the ladder. It is the difference
between a controller that must retreat under load and one that never does.

**Amendment after the actuator survey (§2.4, §3.7).** The table above prices
the ladder as if a rung's weights were resident only at that rung. No
cross-rank weight mover exists, so the actuator is an arena refill whose VRAM
is sized for the *deepest* rung and resident at *every* rung. Under that
constraint Part A alone yields 1.231x pipelined at a top-rung pool of 390,700
— i.e. **it costs ~10% pool to buy up to 23% prefill**, and standing alone it
is a marginal trade rather than a win. The Part B row is unaffected in kind:
its pool is cut-independent, so it absorbs the arena's fixed weight residency
and still lands near 478,000 live-equivalent with 2.0x prefill.

The order of the argument therefore inverts: **Part B is not the follow-on that
removes Part A's ceiling — Part B is what makes Part A worth doing at all.**

---

## 1. BINDING REQUIREMENT — everything ships as planner capability

Per the canonical plan's "PLANNER-SOLVED, UNIVERSAL" section, this is a
first-class constraint on every artifact below, not a compliance section.

Rungs, admission thresholds, hysteresis bands, move costs, the rank0 depth cap
and the decoupling decision are **outputs of `solve()` from measured inputs**.
This rig's numbers — `[32,16,16]`, 450.7 MiB/layer, a 42-layer rank0 cap,
1.757/7.740/7.275 ms per layer — are **calibration data only** and appear
nowhere in solver code. They live in tests, as one profile among several.

Measured inputs, and where each is probed:

| input | source | never |
|---|---|---|
| `free_mib` per rank | NVML free, minus corridor reserve | a typed VRAM number |
| `weight_mib_per_layer` | checkpoint census (`checkpoint_weight_terms`, `pp_cut.py:1907`) | a flat guess (rev 2's 724.3 was +61%) |
| `kv_mib_per_token_per_attn_layer` | model config geometry × KV dtype width | a fitted per-layer cost |
| `layer_families` | the checkpoint's own `layer_types` / `full_attention_interval` (`pp_cut.py:622`) | an assumed hybrid shape |
| `ms_per_layer` | per-rank prefill self-probe | a transferred vector |
| `link_mib_per_s` | pair-matrix bandwidth probe | a PCIe nameplate figure |
| `min_pool_tokens` | corridor floor for the deployment | a hardcoded floor |
| `prefill_tokens_per_s` | measured fill rate | an assumed arrival rate |

**The rank0 depth cap is derived, not typed.** It is not a parameter of the
ladder solver at all: a cut is admissible iff its weights plus GDN state fit
the rank's free bytes and its pool clears the corridor floor. The 42-layer cap
observed on this rig is what that arithmetic happens to yield here; on a bigger
card the same arithmetic yields a deeper cap, pinned by
`test_rank0_cap_is_derived_not_typed`.

**Generality proof (#434 canon)** — `test_layout_ladder_704.py` runs every
structural property against synthetic foreign profiles: a 4-card rig with
differing sizes and speeds, a 2-card **0-GDN pure-attention** model (the
degenerate family split, where every layer carries KV), and this rig. No solved
vector is transferred between them.

One property earned its place by failing first: on a **uniform** rig (identical
cards, identical per-layer cost) the Pareto frontier collapses to a single
rung, because piling layers onto a "fast" rank only buys pipeline time when
some rank is actually faster. The solver must report *no ladder* rather than
manufacture steps. Pinned by `test_a_uniform_rig_has_no_ladder`.

---

## 2. Verification pass — every mechanism checked at file:line

The counter-vs-actuator family produced five members today, so nothing below is
taken from a comment.

### 2.1 What is real and reusable

| claim | verdict | evidence |
|---|---|---|
| Phase-separated PP/TP pool rules exist | REAL | `planner/pp_cut.py:2430` `pp_phase_pool`, `:2450` `tp_phase_pool` |
| PP pool is the MIN rule, vector cannot relieve | REAL, metal-proven | `pp_cut.py:2437-2446` refuses a vector arg by construction |
| Weight census 450.7 MiB/layer | REAL, ±9 MiB on rank0 | `FINDING_702_weight_term.md` table |
| Layer-family vector from checkpoint | REAL | `pp_cut.py:622` `layer_families_from_config` |
| Per-stage attention counts | REAL | `pp_cut.py:673` `attention_counts` |
| Cut enumeration + dual-objective pricing | REAL | `pp_cut.py:2322`, `:2489` |

### 2.2 The correction this design rests on — a wrong functional form

**`pp_phase_pool` divides by the rank's TOTAL layer count, but only the
attention layers carry token-scaling KV.**

Ground truth is the allocator, not a fit. `HybridLinearKVPool`
(`mem_cache/memory_pool.py:3606`) sets
`self.full_layer_nums = len(full_attention_layer_ids)` at `:3609` and builds
its token-indexed pool with `layer_num=self.full_layer_nums` at `:3687`. The
linear (GDN) layers are served by `MambaPool` (`memory_pool.py:464`), whose
size follows the **sequence** count, not the token count.

The arithmetic that makes this a proof rather than a preference: this
checkpoint has `num_key_value_heads=4`, `head_dim=256`, so one token of K+V for
one attention layer is `2·4·256 = 2048` elements. The flat rule's calibrated
cost is 960 bytes per token per layer — **0.47 bytes per element**, and no
dtype has a fractional byte width. Priced on attention layers only, the same
census yields the cell **from config**: `2 x 4 x 256 x 1 B = 2048 B` under the
shipped `fp8_e4m3`.

> **Corrected (E1/E2).** This paragraph originally read "4096 B (bf16) … lands
> at 0.83 of the observed pool, binding on rank1". Both halves were wrong and
> for the same reason: I *fitted* the cell against an observed pool instead of
> reading it from config, so the 2x dtype error and a bogus 0.83 fudge factor
> cancelled into something that looked calibrated. The boot log settles it with
> zero free parameters — at 436,766 tokens it logs K sizes 2.92 / 2.08 / 1.67
> GB against attention counts 7 / 5 / 4, i.e. exactly `attn_i x 1024 B` per
> token. **PP2 binds at 436,766**, not rank1.

Both forms agree on *total* bytes per token, because both were fitted to the
same observed pool. They disagree on the **distribution across ranks**, which
is the only thing a cut solver produces. The flat form is a fit artifact — and
so, in its own smaller way, was mine.

Consequence, and it is not academic: the two forms diverge exactly in the
region the ladder operates in, which is why the divisor had to be settled. What
does *not* follow is any specific boot arm — see §6, where my `[33,15,16]`
recommendation is retracted.

### 2.3 Actuator survey — result

| component | verdict | evidence |
|---|---|---|
| `managers/kv_reshard.py` | **ACTUATOR**, the strongest in the tree | `_execute()` at `:789` — pack `read_rows` (`:816-830`), real P2P `dist.batch_isend_irecv` (`_dist_exchange`, `:939`), `write_rows` (`:865-883`), cutover (`:885-892`). Wired at `scheduler.py:4981-4985` / `:7390-7394` |
| `managers/corridor_guard.py` | **ACTUATOR** | `ensure_headroom()` `:601`; real `empty_cache()` `:978`, real `carrier.spill()` `:1033`; consumed at `regime_admission.py:344`, `phase_flip_runtime.py:2718/5376`, `vram_dial.py:1107`. Two-watermark band |
| `managers/regime_act.py` | **ACTUATOR**, 3 axes | `RegimeActuator.apply()` `:152`; `wired_axes` `:121-130` = `kv`, `vram`, `phase` |
| `model_executor/kv_pressure_ladder.py` | **COUNTER**, self-documented | `apply(plan)` `:1666` mutates only its own bookkeeping; every handover strategy but `NoHandover` raises `NotImplementedError` (`:273`, `:333-503`). Its hysteresis is real and reusable: asymmetric marks, ascend 0.85/window 4 vs descend 0.55/window 64, constructor-enforced (`:1088-1152`) |
| `managers/kv_ladder_auto.py` | **COUNTER** / boot-time config bridge | `auto_ladder_table_fn` `:272`; builds data objects only |
| `managers/regime_stages.py` | refusal logic, honestly so | see below |

### 2.4 R1 ANSWERED — there is no cross-rank weight mover, and its absence is deliberate

This is the finding that shapes Slice 1.

- `regime_stages.py:100` defines `REACH_NO_WEIGHT_MOVER`, whose reason string
  reads: *"NOT reachable: the weight (MLP/GEMM) vector differs and no runtime
  actuator moves weights — switching arms needs a restart (#354/#357)."*
- `StagePlan.reachable` (`:156`) excludes that code, so a stage needing a
  weight move **cannot be selected**; `regime_act.py:121-130` wires exactly
  three axes and a weight axis cannot appear in the dispatch table.
- Corroborated independently at `uneven_perf.py:7744-7751`.

So #363's "weight mover" is a **refusal code, not an actuator**. That is the
*opposite* of the counter-versus-actuator failure: a named unreachable state
that refuses itself, with the reason spelled out. It is honest, and it is not
something I can wire to.

**What does exist** is `model_executor/weights_arena.py` with
`managers/phase_flip_boot.py:361` (`PhaseFlipStacks.refill`): a fixed-address
VRAM arena per rank, refilled by a contiguous **host→device memcpy** from a
boot-baked pinned image (`arena_refill` `:539`, `dst.copy_(payload)` `:576`),
checksum-verified, with a `restore=` rollback arm. It contains **no `dist.*`
call** — nothing crosses a rank boundary. It toggles which *pre-loaded* layout
occupies a rank's own arena, and the tensor's VRAM address never changes, which
is what keeps captured CUDA graphs valid.

---

## 3. Part A — the prefill layout ladder

Implemented: `python/sglang/srt/planner/layout_ladder.py`.
Tests: `test/registered/unit/planner/test_layout_ladder_704.py` (20 green).

### 3.1 Rung definition

Rungs are the **Pareto frontier** of (pool tokens, pipelined prefill time) over
all feasible cuts, ordered most-pool/least-speed → least-pool/most-speed. A cut
beaten on both axes is not a rung. Feasibility is: weights + GDN state fit the
rank, every stage holds ≥1 attention layer, and the rung clears
`min_pool_tokens`.

A stage with **zero** attention layers is refused rather than priced as
infinite capacity (`pp_cut.py`, `stage_family_capacities`). Reporting unbounded
capacity for a stage that cannot serve a token is how a boot that cannot run
gets armed.

Solved ladder on this rig (census inputs, corridor floor 200k):

| rung | attn | pool | admit | pipelined |
|---|---|---|---|---|
| `[31,17,16]` | (7,5,4) | 429,537 | 408,060 | 1.000x |
| `[32,16,16]` | (8,4,4) | 393,562 | 373,884 | 1.062x |
| `[33,15,16]` | (8,4,4) | 379,139 | 360,182 | 1.130x |
| `[34,15,15]` | (8,4,4) | 364,717 | 346,481 | 1.133x |
| `[35,14,15]` | (8,4,4) | 350,294 | 332,780 | 1.206x |
| `[36,14,14]` | (9,3,4) | 298,553 | 283,625 | 1.214x |
| … | | | | |
| `[41,11,12]` | (10,3,3) | 211,008 | 200,458 | 1.507x |

**The attention plateau is the structure the ladder exploits.** From
`[32,16,16]` through `[35,14,15]` the attention profile stays (8,4,4): layers
32-34 are all linear under interval 4, so moving them onto rank0 buys pipeline
time while adding *no* per-token KV cost to rank0 — only their weights bite.
The flat rule cannot see this, which is why it under-rates the plateau. On a
0-GDN model no such plateau exists and the solver reflects that
(`test_pure_attention_model_has_no_gdn_plateau`).

### 3.2 Move cost, per rung — measured, not assumed

Weight movement dominates and is **cheap per step**, because adjacent frontier
rungs differ by one or two layers:

- 451 MiB (1 layer) or 901 MiB (2 layers) per step;
- at the pair-matrix reach used for the solve above, 150-300 ms;
- **8 of 15 frontier steps move ZERO attention layers**, hence zero KV.

That last fact is a design lever, not an observation: steps whose boundary
moves entirely within a run of linear layers are strictly cheaper, and the
solver already prefers them by construction because it prices the KV term at 0.

The 3000 MiB/s figure used above is a **placeholder pending the pair-matrix
probe** — rank0 sits on PCIe x4 on this rig and the effective reach under a
live serving load is not the nameplate. This is the first thing Slice 1
measures; §7 R2 tracks it.

### 3.3 Thresholds and hysteresis — derived, never typed

Two thresholds per adjacent pair, both consequences of measurement:

- **Ascend** (retreat to the roomier rung) must *start* while the current rung
  still has headroom for tokens arriving during the move:
  `ascend = admit(deeper) − fill_rate · move_time`. Solved by fixed-point
  iteration, because the KV half of the move cost itself depends on the fill at
  which it is evaluated.
- **Descend** (advance to the faster rung) fires only when the live set fits
  the deeper rung's ceiling with the same window to spare, doubled — the
  asymmetry *is* the hysteresis.

**Anti-oscillation is an invariant, not a tuning exercise.** If a pair's bands
cross, or land below zero fill, that pair is **dropped from the ladder**. Bands
below zero say the move consumes more headroom than the rung has at *any*
occupancy — there is no fill level at which the controller could sit there. A
link slow enough collapses the ladder to nothing, which is the honest answer
rather than a ladder that thrashes on metal
(`test_a_rung_whose_move_cannot_be_funded_is_pruned`).

This invariant caught a real bug in my own first implementation: with a
crawling link the thresholds went to −180M tokens yet still satisfied
`descend < ascend`, so nothing was pruned. The `>= 0` half of the guard exists
because the test failed without it.

### 3.4 Descend-only-at-low-fill

Descending (deeper, faster, tighter) is permitted only below the derived
descend threshold. This is doubly justified and both halves matter:

1. **Capacity** — the deeper rung physically cannot hold a larger live set.
2. **Move cost** — the KV that must follow its layer is `attn_moved ·
   c · live_tokens`. At 100k live tokens a 1-attention-layer step moves 324
   MiB; at 400k it moves 1,295 MiB. Descending only at low fill bounds KV
   movement to the regime where it is cheap.

The **ascent is the hard direction**, and this design does not pretend
otherwise: fill rises → a roomier rung is needed → but the ascent itself moves
KV precisely when KV is most expensive. The mitigation is the asymmetric
threshold above (leave early, while the move is still affordable). The *real*
fix is Part B, which drives the KV term to zero identically.

### 3.5 In-flight chunked requests at a rung change

A rung change relocates layers between ranks, so a chunked prefill in flight
would observe two different layer maps across its chunks.

**Decision: rung changes commit at a chunk boundary, never inside one.** The
controller raises a *pending rung* and the scheduler applies it at the next
boundary at which no chunked prefill is mid-sequence. Requests already admitted
keep the layout they started under. This is deliberately the same discipline
the flip controller already uses for seam waves, so the two share one
quiescence notion rather than inventing a second.

**D6 CORRECTION — as written, that rule can starve, and starves in exactly the
case that matters.** With a 327,680-token context at a 512 chunk size a single
prompt is 640 chunks, and with overlapping chunked admissions the "no chunked
prefill mid-sequence" boundary can be arbitrarily far away — possibly never.
**Ascent is needed precisely while fill is rising, i.e. while chunked prefills
are active**, so the pending ascent that can never commit is the ladder's own
#701-shaped wedge: the controller correctly decides to move and is structurally
prevented from doing so, right up until the pool it was trying to escape
overflows.

The quiescence rule is therefore paired with an **admission hold with a bounded
drain**:

1. When a rung change goes pending, **stop admitting new chunked prefills**.
   Already-admitted requests continue.
2. The in-flight set then drains monotonically, so the boundary is reached in
   bounded time — at most the longest in-flight remainder, which is a known
   quantity, not an open-ended wait.
3. The hold has a **deadline derived from the ascent's own urgency**: the
   headroom at the moment the ascent was raised, divided by the measured fill
   rate. If the drain would exceed it, the ascent is escalated ahead of
   throughput rather than missed.
4. Because the hold costs admission throughput, it is asymmetric: a **descent**
   (speed-seeking, taken at low fill) never holds admission — it simply waits
   for a natural boundary or is dropped. Only an **ascent** (safety-seeking)
   may hold.

The alternative — chunk-boundary-safe relocation, letting an in-flight request
survive a layout change mid-sequence — is strictly better and strictly harder,
and is not in scope for Slice 1. It is the right answer if the admission hold's
throughput cost measures badly.

**Not built:** per-request layout selection. One layout is regime-wide, per the
`phasen-layoutwechsel` law. A rung is a property of the instance, not a request.

### 3.7 The arena constraint, and the honest repricing of Part A

Since no cross-rank mover exists (§2.4), the ladder's actuator is an **arena
refill**: each rank keeps one fixed-address VRAM arena and one host-pinned
image covering the union of layers it may own across the ladder, and a rung
change is an H2D memcpy of the delta layers. On this rig that is the *better*
primitive, not a fallback — with no P2P (all PHB), a rank-to-rank weight
transfer would stage through host memory anyway, so the H2D refill costs the
same link time with no new collective, and the fixed address keeps captured
CUDA graphs valid across a rung change.

Host RAM cost is small: for a 2-rung ladder the union is 2 layer-images beyond
a single layout, about 0.9 GiB.

**The price is on the device side, and it is not small.** The arena is sized
for the *deepest* rung and is resident at *every* rung — a rank does not get
its weight bytes back when the ladder sits shallow. Modelled by
`solve_arena_ladder()`, and the consequences are:

1. **The shallow rungs get poorer.** A rank0 span that reaches deeper makes the
   top rung poorer, because the arena it implies is resident even when the
   ladder sits shallow. (The absolute figures previously quoted here — 518,433
   and 390,700 — were computed with the fitted bf16 cell and are withdrawn per
   E1; the *direction and mechanism* are unaffected, and are pinned by
   `test_a_deeper_ladder_costs_the_shallow_rungs_pool`.)
2. **Two rungs with the same attention profile differ only by GDN residency.**
   With free memory pinned by the arena, a rung's pool is set by its attention
   count plus its GDN state. So `[33,15,16]` and `[35,14,15]` — both (8,4,4) —
   price *nearly* the same, differing by the residency of the linear layers
   rank0 gained. **Under an arena the ladder's real axis is the attention-count
   vector, not the raw layer cut.**

> **Corrected (E4).** Point 2 originally claimed such rungs price *exactly* the
> same, so the deeper one **strictly dominates**. That was an artifact of
> charging GDN at 19.5 MiB/layer (temporal_state alone) instead of the true
> **50.85** — the missing 30.0 MiB is the *speculative*
> `intermediate_ssm_state_cache`. At 8 attention layers, 50.85 MiB is ~3,250
> tokens, so a same-profile deeper rung buys pipeline speed for a small but
> real pool cost. It is a weak trade, not a domination. The test that asserted
> exact equality now asserts the gap is explicable by GDN residency and nothing
> else — it failed the moment the correct constant landed, which is how the
> error surfaced.

Point 2 still invalidates my own earlier Slice 1 proposal, for the same reason
in weakened form. I had picked `[33,15,16] ↔ [35,14,15]` precisely because it
moves zero attention layers — which is what makes it nearly worthless *as a
ladder pair*: the pool gap is ~3k tokens, far too small for a controller to
trade against. It remains a fine pair for proving the **mover** mechanically
(weights move, no KV moves), and that is now its only stated purpose. A pair
that actually trades must cross an attention boundary,
e.g. `[35,14,15]` (attn0=8) ↔ `[38,13,13]` (attn0=9).

Solved arena ladder, rank0 span [31, 38], arena (38,16,17):

| rung | attn | pool (live-equiv) | pipelined |
|---|---|---|---|
| `[31,16,17]` | (7,4,5) | 390,700 | 1.000x |
| `[35,14,15]` | (8,4,4) | 370,568 | 1.135x |
| `[38,13,13]` | (9,3,4) | 329,394 | 1.231x |

**Verdict on Part A standing alone: marginal.** It buys up to 1.23x pipelined
prefill at low fill and costs ~10% pool at high fill, against an incumbent that
is not on the ladder at all. That is a real trade, not a free win, and it should
not be booted as an end in itself.

There is an alternative mover design worth naming rather than silently
discarding: **reallocate** weights per rung instead of holding an arena. That
restores the full per-rung pool (the `solve_layout_ladder()` model, top rung
518,433) at the cost of a free/alloc/load plus **CUDA graph recapture** on every
rung change — seconds, not milliseconds. Because rung changes are rare and
hysteresis-damped, this is not obviously the wrong trade, and the two solvers in
this branch price both. Choosing between them needs the measured recapture cost,
which Slice 2 should collect while it is measuring the link.

### 3.6 Interaction with the flip controller's seam funding

The ladder and the phase flip both move bytes over the same links, and they
must not bid against each other. **The flip owns the link during a seam wave;
the ladder is inhibited for its duration and may not raise a pending rung
mid-flip.** The ladder's own move is, in effect, a small seam wave and should
be funded from the same reserve machinery
(`managers/phase_flip_seam_reserve.py`) rather than a parallel budget.

Exact wiring is pending the explorer's actuator findings (§2.3), and is
deliberately left unspecified here rather than guessed.

---

## 4. Part B — PP-KV decoupling

### 4.1 The mechanism

Token-shard the 16 attention layers' KV across **all** ranks in proportion to
free bytes. The stage owning a layer computes attention over the distributed
pool by Q-broadcast → partial attention per shard → LSE merge. GDN states stay
with their layers (they are per-sequence and small, and moving them buys
nothing).

**Verification result: Part B is construction, not assembly.** The survey is
in, and it splits cleanly.

*Reusable as-is, production-wired:*

- **The cross-rank LSE merge** — `layers/dcp/comm.py:228-262`
  `cp_lse_ag_out_ar_mha_uneven()`. Genuinely **N-way** (all-gathers every rank's
  LSE in one collective at `comm.py:126-134`, then a single
  `torch.logsumexp` over the full stack at `:150-151`/`:251`). **Merge order is
  deterministic by construction** — `all_gather` places rank *i* at stack index
  *i*, so the order is rank order, never arrival order. **This retires risk R4.**
  Note this is *not* `layers/attention/merge_state.py`, which is strictly 2-way
  (`merge_state.py:26-46`) and merges KV blocks *within* one rank. Do not
  conflate them.
- **The owner-rule token-vector pipeline** — computed via `set_cp_token_ratios`
  (`distributed/utils.py:232-239`), turned into per-rank bounds at
  `layers/dcp/owner.py:348-360`, consumed by the attention backends at
  `flashinfer_backend.py:1369-1385` and `triton_backend.py:772,981-992`.
- **The weightless-rank forward path** — activated from real server args at
  `managers/scheduler.py:8910-8912`, consumed in the live decode/extend forward
  at `flashinfer_backend.py:5667/5733/5803/5943/6056`.

*Confirmed NOT evidence of a working path:*
`layers/dcp/test_weightless_kv_math.py` is **test-only**. It imports no torch
and no `comm.py`; it **reimplements** the LSE formula locally (`_merge_lse`,
lines 47-68). It proves the math, and nothing about wiring. My prior suspicion
is confirmed — it must not be counted as a working cross-rank path.

### 4.1a The two things that must actually be built

**B1 — a process group spanning PP stages does not exist.** DCP groups are, by
construction, subsets of a single PP stage: `parallel_state.py:3142-3159`
builds them by chunking *each TP group*, and each TP group's contiguous rank
block is exactly one PP stage's rank set (`:3101-3123` vs `:3354-3372`). Worse,
`distributed/dcp_group_guard.py:1-42` states that a PP prefill group runs with
`dcp_size == 1` and **no DCP group at all**. So under PP prefill the machinery
is not merely cross-stage-incapable — it is not instantiated.

The right template is `initialize_phase_flip_secondary_groups`
(`parallel_state.py:3422-3517`), which already builds groups over an arbitrary
rank layout behind a group-creation-manifest exchange and world-wide equality
check (`:3484-3497`). But the existing `_FLIP_*` groups themselves are not
reusable: they are semantically a *regime swap* (primary PP topology XOR
secondary TP topology, `:3435`), whereas Part B needs a group that **coexists
live** with an active PP-prefill forward pass. The pattern transfers; the
groups do not.

**B2 — the KV pool is dimensioned by weight ownership, not merely accessed by
it.** This is the tightest coupling in the tree and it is upstream of the
attention forward entirely:

```
model_executor/model_runner_kv_cache_mixin.py:2466-2470
    full_attention_layer_ids = [
        i for i in config.full_attention_layer_ids
        if self.start_layer <= i < self.end_layer
    ]
```

`start_layer`/`end_layer` are this rank's own weight-owned PP range. That
filtered list flows straight into pool sizing (`memory_pool.py:3637` →
`:3688`/`:3710`). **A rank's KV pool tensor today has no row-space for a layer
whose weights it does not own.** Decoupling therefore requires changing the
pool allocation path, not only the attention forward. The same filter recurs at
`:3453`, `:3633`, `:3893`.

This is the finding that reprices Part B, and it is worth stating plainly: the
prize in §4.2 is unchanged and still large, but the route to it runs through
allocation, not just collectives.

### 4.2 Why it is the strategic prize

**The pool becomes exactly cut-independent.** Total weight bytes are
`64 · 450.7` regardless of where the cut falls, so once KV is no longer pinned
to the layer owner, `decoupled_phase_pool` returns the same 446,592 tokens at
*every* rung. Verified in `pp_cut.py::decoupled_phase_pool`.

Three consequences, in descending order of value:

1. **The opposition dissolves.** The deepest cut `[44,10,10]` yields **2.000x
   pipelined prefill at 539,017 live-equivalent pool — +23.9% over the
   incumbent's observed 434,878.** Coupled, that same cut holds 160,358: a
   2.78x difference. Speed and capacity stop trading.
2. **Rung changes stop moving KV.** The ladder's hard direction (§3.4) goes
   away: a rung change moves weights only. The ladder becomes freely walkable
   in both directions at any fill, which is what makes Part A a real controller
   rather than a low-fill opportunist.
3. **Phase-uniform layout.** Choosing the **same token vector as the TP phase**
   makes the KV layout identical across both phases. The seam KV move at flip
   disappears — attacking #690's 2.0-4.2 s fixed flip cost, dissolving #635's
   handover complexity, and collapsing #703's two-geometry cache-key problem to
   one layout, one key.

Prize 3 is why the token vector must be chosen to match the TP phase and not
merely "by free bytes". Free-byte proportionality and TP-vector identity
coincide only if the TP vector is itself free-proportional.

**D5 CORRECTION — "TP identity wins" was stated unconditionally, and it cannot
be.** The measured TP vector puts ~43.8% of KV rows on rank0 (row counts
190,681 / 136,201 / 108,961 in the boot log), and rank0 is the rank with the
**least** free bytes at exactly the deep rungs Part B exists to unlock (e.g.
`[44,10,10]`, where rank0 carries 44 layers of weights). So prize 1
(cut-independent pool) and prize 3 (phase-uniform vector) can **contradict each
other**: forcing the TP vector at a deep rung can make that rung infeasible.

TP identity therefore carries a per-rung feasibility bound, checked before it
is adopted:

```
vector_share_i x total_rows x kv_cell x total_attn_layers  <=  free_i
```

Resolution where the bound fails, stated so the choice is not made ad hoc at
implementation time:

- The **rung** yields first, not the vector. Phase-uniformity is worth more
  than one extra rung of depth, because it is what deletes the seam move for
  *every* flip; so the ladder's reachable depth is clipped to the deepest rung
  at which the TP vector is feasible.
- If **no** rung admits the TP vector, phase-uniformity is unavailable on this
  hardware and Part B falls back to a free-proportional vector, keeping prizes
  1 and 2 and losing prize 3. The design must not pretend prize 3 is
  unconditional.
- Re-deriving the TP vector to be more rank0-friendly is a *third* option that
  trades decode balance for prefill depth. It is out of scope here and belongs
  to whoever owns the TP vector, but it is named so it is not rediscovered as
  novel.

### 4.3 Cost

~25 MiB of collective traffic per attention layer per 512-token chunk,
estimated at +10-20% chunk cost, overlappable behind GDN/FFN compute. On this
no-P2P rig (PCIe x4/x8/x8, all PHB) the collective floor already dominates
decode, so this estimate is the least trustworthy number in the document and is
the first thing Slice 3 measures.

### 4.4 Correctness gate — mandatory

**D4 CORRECTION — the gate as originally written could never pass.** I demanded
byte-identity "decoupled vs coupled". That is A-vs-**B**, not A-vs-A: an LSE
merge sums partial results in a different floating-point order than monolithic
attention, so bit-exact agreement is not a property correct code has. As
written the gate would fail forever on a correct implementation, and the
predictable outcome is that someone eventually waives it — which is worse than
having no gate. Respecified as two gates:

1. **Determinism (byte-identity, A-vs-A):** decoupled vs decoupled, identical
   output across repeated runs *and* across boots, with inputs sampled on
   **CPU** and moved to device — `torch.randn` on-GPU is not
   architecture-identical across the 3080s and the 5090, and this rig has been
   bitten by that before. This is the gate that catches a genuine
   non-determinism, e.g. an arrival-ordered merge.
2. **Agreement with the coupled reference, within a stated tolerance:**
   max-abs and max-rel bounds fixed *before* the run, plus an end-to-end
   greedy-decode token-sequence match over a fixed prompt set. Byte-identity is
   required here **only** against a coupled reference forced to the identical
   split schedule, which makes the summation orders match by construction; that
   variant is the stronger check and should be built if it is cheap.

Gate 1 is the one that must never be waived.

An LSE merge is not associative in floating point, so merge order must be
**fixed and rank-order-deterministic**, not arrival-order-dependent. This is
the single most likely source of a silent numerical divergence and the gate
exists to catch it.

Precedent for taking this seriously: tree-spec under uneven-DCP was
**silently wrong** and is permanently gated (#tree-spec-dcp-guarded). Part B
touches the same machinery. It does not ship without the gate green.

---

## 5. What I will NOT build

Named explicitly, so scope creep has to argue for itself:

- **No per-request layout choice.** One layout per regime (§3.5).
- **No PP+DP/EP combination.** Pure PP prefill / TP decode, single node.
- **No tree-spec under decoupled KV.** Permanently gated as silently wrong
  under uneven-DCP; Part B does not reopen it.
- **No new mover.** If a runtime weight mover does not exist, Slice 1 wires the
  ladder to the existing regime "weight mover" stage or blocks on it — it does
  not grow a second one alongside.
- **No host-tier KV for the ladder.** That is #703's axis. The ladder's answer
  to a full pool is a rung change, not a spill.
- **No safety margin invented on top of the corridor.** `min_pool_tokens` is a
  deployment input; the solver does not second-guess it.
- **No replacement of `pp_phase_pool`.** The family rule is added alongside;
  Slot-2's metal-calibrated backtest stays green and untouched.

---

## 6. Boot-arm implication — RETRACTED, and what replaces it

**I recommended `[33,15,16]` at pool 457,604. That number was impossible and
the arm is withdrawn.**

The arithmetic I failed to do: `[33,15,16]` leaves rank2 holding layers 48-63 —
**byte-identical to the incumbent's rank2**, whose measured capacity is 436,766
(436,766 / 435,822 / 434,878 across three boots, ~1.9k noise). Under the
min-rule the world pool cannot exceed the binding rank, so *no* cut keeping
that rank2 can exceed 436,766. My 457,604 overshot an **unchanged rank's
measured cap** by 4.8%. Expected actual is ~387k, about **-11% versus the
incumbent** — a pool regression, not an improvement. `[32,16,16]` had already
failed its gate on metal at 416,796, which is the same error class.

My "discriminating experiment" justification was also void. The divisor dispute
was settled **white-box** — allocator source plus byte-exact K-size log lines —
so a boot window spent re-proving it would have bought nothing.

**What actually follows, and it is a stronger result than the arm I lost:**

> **No cut that keeps rank2 = layers 48-63 can beat the incumbent pool.**
> Rank2's 436,766 is a hard ceiling. A pool-positive rung must **shrink rank2's
> attention count** — e.g. a 12-layer rank2 holds 3 attention layers instead of
> 4, lifting its cap by ~4/3 — or wait for Part B, which removes the per-rank
> attention divisor entirely.

That is the search direction for the next arm, and it is a *structural*
constraint, not a number that needs re-measuring. It also reinforces §0's
amended conclusion: the coupled ladder is boxed in by whichever rank owns the
tail, and decoupling is the way out.

**Before any further pool-gated boot**, the corrected model must clear a
retro-prediction gate that can fail: reproduce all four measured points —
434,878 / 435,822 / 436,766 (`[28,20,16]`) and 416,796 (`[32,16,16]`) — within
boot noise, **with the correct binding rank each time**. Slot-2 owns that gate
along with the canonical solver. Until it passes, no rung's predicted pool
gates a window, including mine.

Standing caveat, unchanged and now more relevant: `PrefillTiming.fixed_ms`
defaults to zero, the optimistic end of the family, so **every speedup in this
document is an upper bound** until a second measured cut pins the intercept.

---

## 7. Open risks

- **R1 — ANSWERED, and it reprices Part A downward.** No cross-rank weight
  mover exists (§2.4). The arena route avoids building one, but its price is
  the residency constraint in §3.7: the arena is sized for the deepest rung and
  is resident at every rung, so the top rung loses ~10% of its pool. Under that
  constraint **Part A alone is a marginal win** — see §3.7. It is Part B that
  makes the ladder worth having, which inverts the original slice ordering's
  implied priority.
- **R2 — link reach is a placeholder.** Move times assume 3000 MiB/s on rank0's
  x4. Under live serving load the real figure may be materially lower, which
  would widen the hysteresis bands and could prune the cheaper steps outright.
  The solver already handles this correctly (it prunes); the risk is that the
  ladder is shorter than advertised.
- **R3 — census pessimism (~17%) is a scale factor, not a proven constant.** All
  live-equivalent numbers carry it. Rankings are robust; absolutes are not.
- **R4 — LSE merge order. RETIRED.** The cross-rank merge
  (`layers/dcp/comm.py:228-262`) is deterministic by construction: `all_gather`
  places rank *i* at stack index *i*, so merge order is rank order and cannot
  depend on arrival. The byte-identity gate in §4.4 stays mandatory, but this
  particular failure mode is structurally excluded.
- **R5 — cross-PP-stage collectives. CONFIRMED, and worse than posed.** DCP
  groups are subsets of one PP stage by construction, and under PP prefill no
  DCP group is instantiated at all (§4.1a B1). A new group definition is
  required. Template exists; the groups themselves are not reusable.
- **R6 (NEW, and the largest in Part B) — KV pool allocation is keyed to weight
  ownership.** `model_runner_kv_cache_mixin.py:2466-2470` filters the attention
  layer ids to the rank's own PP range *before* the pool is built, so a rank
  has no row-space for a layer it does not own (§4.1a B2). Part B must change
  the allocation path. This moves Part B from "assemble existing collectives"
  to "rework pool construction", and Slice 3's estimate is revised accordingly.

---

## 8. Slice plan

| slice | content | effort | gate |
|---|---|---|---|
| **0 — landed** | Family-split pool rule + ladder solver, hermetic, 3 hardware profiles | done | 26 tests green |
| **0b — landed** | Rung controller (decision function) + oscillation falsifiers | done | 10 tests green, can-fail verified |
| **1a** | Multi-layout arena refill: extend `weights_arena.py`'s image/refill to N boot-baked layouts, `[33,15,16]` ↔ `[35,14,15]` (same (8,4,4), **zero KV moved**) — proves the MOVER only, not the ladder | 1 window | refill completes, checksum green, graphs survive, output byte-identical across a refill |
| **1b** | Wire the controller to 1a on a pair that actually trades: `[35,14,15]` ↔ `[38,13,13]` (attn0 8→9) | 1 window | hysteresis proven on metal, no oscillation under load |
| **2** | Measure real move cost on the pair matrix **and the CUDA-graph recapture cost**, to decide arena vs realloc (§3.7); re-solve | 0.5 window | move time within 20% of solved; the arena/realloc fork closed on numbers |
| **3a** | **B1**: build a cross-PP-stage process group at boot, using the manifest-verify-then-create pattern from `parallel_state.py:3422-3517` | 1-2 windows | group forms and survives a boot; collective round-trips |
| **3b** | **B2**: unpin KV pool allocation from `[start_layer, end_layer)`; size each rank's shard of the global attention pool from the free-byte vector | 2-3 windows | pool commits at the solved size; no OOM at the corridor floor |
| **3c** | Decoupled attention forward behind a flag, reusing `cp_lse_ag_out_ar_mha_uneven`; coupled fallback retained | 1-2 windows | **byte-identity A-vs-A green** |
| **4** | Phase-uniform token vector; retire the seam KV move | 1-2 windows | flip cost measurably below the 2.0-4.2 s baseline |

Slice 3 was a single 2-3 window line item before the survey; it is now 3a/3b/3c
at 4-7 windows, because R6 turned the forward-path change into a pool
allocation change. That is a real repricing, not padding — and 3b is the
critical path, not 3c.

Slice 1 is deliberately the cheapest metal-provable cut: a rung pair inside the
attention plateau, so it proves the mover and the hysteresis **without moving a
single KV byte** — the two mechanisms that everything else depends on, isolated
from the one that is hardest to get right.
