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

**Amendment after the actuator survey (§2.4, §3.7).** The shape above prices
the ladder as if a rung's weights were resident only at that rung. No
cross-rank weight mover exists, so the actuator is an arena refill whose VRAM
is sized for the *deepest* rung and resident at *every* rung. Part A alone
therefore **buys prefill speed at a real pool cost** at every rung, which makes
it a marginal trade standing alone rather than a win. (Specific figures
previously quoted here — 1.231x at 390,700, "~10% pool for 23% prefill" — came
from the withdrawn model and are not restated; the *sign* of the trade is what
the arena argument establishes, and that is unaffected.)

Part B is unaffected in kind: its pool is cut-independent, so it absorbs the
arena's fixed weight residency once and does not pay again per rung.

Two further findings tighten this, and both point the same way. The arena makes
a rung's pool depend on its attention count, so rungs inside one attention
plateau barely trade at all (E4). And no cut keeping rank2 = layers 48-63 can
beat the incumbent pool at all, so the coupled ladder is boxed in by whichever
rank owns the tail.

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

> **Confirmed on absolute grounds (2026-08-16, restored boot `bdd777a8cd`).**
> The running system's `cell_size` is `attn_layers × 2048` **exactly**, per rank
> — 7 / 5 / 4 × 2048. This is no longer a fit that reproduces a pool, nor an
> inference from allocator source: it is the live sizer agreeing with the rule
> term for term. The attention-layers divisor is settled.

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
infinite capacity (rev5 `pp_cut.stage_pp_capacities`). Reporting unbounded
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

> **HARD PRECONDITION, from the live system (2026-08-16).** The admission hold
> above must **not** be armed until the #701/#698 chunked-prefill admission
> deadlock is fixed. That deadlock is currently the dominant live defect —
> `#running-req: 0` on **90.6% of prefill rounds** with zero completions — and
> it is the same wedge D6 describes, already happening for another reason. An
> admission hold dropped into an admitter that is already starving would
> deepen the deadlock rather than bound a drain, and would then be
> indistinguishable from it in a log.
>
> So the hold ships **gated**: it may only fire when admission is demonstrably
> live (non-zero running requests), and it is disabled outright until #701
> lands. This is a case where my mechanism and an existing bug share a failure
> mode, and the ordering between them is not optional.

**Instrumentation warning for any acceptance work built on this design:** the
`cache_hit_rate` metric reports **0.0 despite real hits** (separate bug, filed).
Count hits from log lines and token counts instead. Any gate written against
that counter would pass or fail for reasons unrelated to what it measures.
Relatedly, acceptance of the "real cache hit across flip **and** reboot" kind is
**unfalsifiable until #701 is fixed**, because the cache is starved rather than
broken — such a test would fail for the wrong reason and must not consume a
boot window.

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

### 3.8 The union arena — a rung change copies ZERO weight bytes

Slice 1a's implementation found that §3.7's H2D refill is more work than the
existing primitives require. Landed as
`model_executor/weights_arena_union.py`, 13 hermetic tests.

`plan_arena_layout` fixes slot offsets by **sorted tensor name**, and
`bind_arena_views` rebinds parameters to arena views **without copying any
bytes** (`weights_arena.py:495-519`). Plan the arena over the **union** of the
rungs' tensors instead of per rung, and:

* every tensor shared by two rungs sits at the **same offset** in both, so
  changing rung cannot require moving it;
* a rung change becomes a rebind plus a PP-boundary change — **zero weight
  bytes on the wire**, replacing the 451-901 MiB per step I had specified.

**The union is not a restatement of a coincidence, and there is a can-fail
proof that it is doing real work.** With *per-rung* layouts, 14 of 56 shared
tensors land at **different offsets** — because `model.layers.28.*` sorts
lexically between `...27.*` and `...3.*`, displacing every slot after it. So a
per-rung arena would have to relocate weights that are already resident and
unchanged, purely because a name sorted differently. Under the union, zero move.

The cost moves from bandwidth to **residency**: a layer that changes hands must
be resident on *both* ranks, so the world holds `64 + (layers that move)`
layer-images. For the slice-1a pair that is one extra layer, ~451 MiB. This is
the same trade §3.7 already prices — pay in resident VRAM, not per-change link
time — and on this rig it strictly dominates the refill, because with no P2P a
rank-to-rank transfer would stage through host anyway.

`arena_refill` is **not** replaced. It remains correct for the *phase flip*,
where the two layouts are genuinely different tensor sets (PP weights vs TP
width-shards) with no useful union. The union is for the *ladder*, where
consecutive rungs differ by one or two whole layers.

**Scope limit, carried in the data structure rather than in a reader's
memory.** "Zero bytes" must not be read as "nothing to do". A GDN layer also
carries per-sequence recurrent state — temporal_state ~19.5 MiB/layer plus
conv_state ~0.762 — which lives with the layer. Slice 1a therefore flips **only
at quiescence**, where no live state exists to preserve; every `FlipDelta`
carries `requires_quiescence=True` and an `out_of_scope` string saying so, and
a test asserts both. Live-state transfer is slice 1b.

### 3.9 Slice 1a-ii — moving the boundary at runtime is a range mutation

Landed as `model_executor/layout_boundary.py`, 14 hermetic tests. The model
turned out to permit this almost directly, and three verified facts fix the
whole design:

* `make_layers` (`utils/common.py:1970-2010`) builds a ModuleList of length
  `num_hidden_layers` with `PPMissingLayer` placeholders outside the owned
  range — so **layer indices are GLOBAL on every rank** and a boundary change
  is not an index shift;
* `start_layer`/`end_layer` are **properties** over mutable `_start_layer` /
  `_end_layer` backing fields (`models/qwen3_5.py:1452-1457`);
* the decoder forward iterates `range(self.start_layer, self.end_layer)`
  (`qwen3_5.py:1483`), reading those properties **on every pass**.

A real layer module parked outside the active range is therefore simply not
executed. The boundary change is a **range mutation** — no module swapping, no
reallocation, no bytes. Combined with §3.8's union arena, a rung change moves
nothing at all; a test pins that every parameter keeps its `data_ptr()` across
a flip while the executed set changes.

**Load wide, run narrow.** At boot a rank builds and loads real modules for the
UNION of the ranges it may occupy, then runs whichever sub-range the ladder
selects. The union must be in force during *loading* too, because weight
loading is gated on the same range (`qwen3_5.py:1563-1564`, `:1707-1708`).

Two failure modes are enforced rather than documented:

1. **Entering a non-resident range is silently wrong, not loud.**
   `PPMissingLayer` is a pass-through (`layers/utils/common.py:109-127`), so
   executing a range whose weights never loaded yields *plausible output from a
   shallower model* rather than an error. Every flip verifies residency first.
2. **A half-applied boundary is worse than no flip.** Dependent structures
   (the KV pool's `full_attention_layer_ids` filter, GDN state maps) are keyed
   by the owned range. Observers run inside the flip; if one raises, the range
   is **rolled back** and the flip fails.

3. **CONTRACT — the actuator owns the active range; it never infers it.**
   Constructing a `LayoutBoundaryActuator` *applies* the declared current
   rung's range to the model. It does not read the model's range and assume
   agreement, and it does not accept the model's range as authoritative. There
   is exactly one writer of `_start_layer`/`_end_layer` once an actuator
   exists, and any other writer is a defect.

   The contract is stated positively because the alternative is not a warning
   but a silent split brain: under "load wide, run narrow" the model *arrives*
   at the union range, so an actuator that merely recorded a declared rung
   would disagree with the model from the very first forward, and the rank
   would execute a layer it does not own — producing plausible output, not an
   error. (This was found by test failure rather than foresight; it is written
   as a contract because the next reader needs the rule, not the anecdote.)
   Pinned by `test_construction_performs_the_narrowing_step`.

**Model-class correction, carried here because it misled me once:**
`models/qwen3_next.py` is **not** the pipeline-parallel model — it asserts
`is_first_rank and is_last_rank` at `qwen3_next.py:1165`, i.e. PP=1 only.
`models/qwen3_5.py` is the PP-capable class and is the one every citation in
this section refers to. Any note pointing at `qwen3_next` for PP layer handling
is wrong.

World-level tiling is validated separately (`validate_world_tiling`), because
it cannot be checked locally: a gap silently drops layers and an overlap
computes them twice, while every individual rank's range looks sensible.

### 3.11 The two dependent structures — both fail silently, so both are guards

A boundary change is not finished when the range changes. Two structures follow
the owned range, and neither raises on its own if it does not follow. Landed as
observers in `layout_boundary.py`, 7 further tests.

**(a) CUDA graphs bake the executed layer set.** The decode graph captures the
model's own `forward` (`decode_cuda_graph_runner.py:1770`), which iterates
`range(self.start_layer, self.end_layer)`. A CUDA graph records actual kernel
launches, so the executed set is fixed *at capture time* and a replay ignores
any later change to `_start_layer`/`_end_layer`. After an unguarded flip the
model reports the new range while every graph replay still runs the old one —
the layer count silently reverts **on exactly the fast path, and only under
replay**, so an eager smoke test would show the flip working. `cuda_graph_observer`
recaptures, or refuses and rolls back.

> **This corrects §3.7.** I argued there that the arena's advantage over
> reallocation was that it "keeps captured CUDA graphs valid across a rung
> change". That is wrong as stated. The arena keeps parameter *addresses*
> stable, which the graph's captured pointers do need — but the graph also
> bakes the executed layer *set*, so **a rung change requires recapture either
> way**. The arena-vs-realloc fork therefore narrows: recapture is common to
> both, and the arena's remaining advantages are avoiding the weight copy and
> avoiding reallocation. Slice 2 must measure recapture cost as a cost of
> *every* rung change, not as a penalty unique to the realloc design.

**(b) The KV and mamba pools are built for a span and indexed off its base.**
Their layer-id lists are filtered to the owned range at build time
(`model_runner_kv_cache_mixin.py:2460-2470`) and rows are addressed as
`layer_id - pool.start_layer` (`memory_pool.py:1576`, `:2889`). Two failures
follow, and the union answers both:

* a layer newly activated outside the built span has **no rows at all** — no KV
  for a full-attention layer, no mamba slot for a linear one;
* the indexing **base must not move**. If a downstream rank's pool were rebuilt
  with the new start (rank1: 28 → 29), every cached row would shift by one
  layer and be silently misattributed.

So **"load wide, run narrow" applies to the caches too**: build the pools over
the union, at a cost of one extra layer's rows per boundary that moves.
`pool_coverage_observer` refuses a range that escapes the built span.

Note which ranks are exposed in the slice-1a pair: rank0 keeps `start=0` and
rank2 keeps `start=48`, so only **rank1** moves its base (28 → 29). It is the
one rank where a naive rebuild would corrupt silently.

### 3.12 OPEN — the ladder enumerates rungs the canonical solver declines to price

Converging on Slot-2's `solve_rung_pool` (`planner/rung_pool.py`, `949a882d17`)
surfaced a tension worth stating plainly rather than working around.

His entry point requires `reserve_for` and `rest_for` **recovered from a boot of
that layout**, and its refusal message is explicit: the per-rank reserve tracks
CUDA-graph capture, does **not** transfer between layouts (measured
6.53 / 3.48 / 5.05 GiB within one boot), so it "cannot be defaulted or carried
from another rung — supply the reserve recovered from a boot of THIS layout, or
do not claim a pool."

That is exactly right for gating a boot. But a **ladder is mostly unbooted
rungs**: enumeration prices thousands of candidate cuts, none of which has ever
run. So the two uses need different provenance, and conflating them would
either paralyse the ladder or launder an extrapolation into a boot gate.

The split adopted: rung pools inside the ladder are **extrapolated and
self-labelled** (`measured=False`), and are never a boot gate. Only a cut that
has actually booted gets a measured pool, and only a measured pool may gate a
window. This is the same discipline already applied to the arming floor
(§3.3's ±32,000-token uncertainty on unbooted rungs) — one more term with the
same provenance problem.

**The consequence, and the premise behind it has since been refuted.** I
reasoned: *if* the reserve tracks CUDA-graph capture, and a rung change forces
recapture (§3.11a), *then* a rung change changes the reserve and therefore the
pool — a term my ladder treats as constant across rungs, and at 3.48–6.53 GiB
per rank far too large to assume away.

> **Refuted (Slot-2, `f55c1a8adf`).** The per-rank holdback is **not** a
> CUDA-graph-capture reserve. It is the delta across `_seam_adjusted_budget` —
> a **seam adjustment**. The "reserve tracks graph capture" premise came from
> the refusal message itself, and it is wrong; that is why every scaling
> hypothesis against it misfit.
>
> This is good news twice over. The graph-recapture coupling I feared does not
> exist, so §3.11a's recapture requirement does **not** drag a pool change
> behind it. And because seam funding is deterministic machinery (#676 arming
> floors, seam projection) rather than an empirical property of graph capture,
> the term is likely **computable per layout rather than needing a fitted
> model** — which would make ladder rung pools *exact* instead of extrapolated.

Until the confirming boot lands, every extrapolated rung pool in this document
keeps its **unquantified reserve term** on top of its arming-floor uncertainty,
self-labelled, with **no interim value folded into any number**. Slot-2 owns the
formula and a follow-up is expected once the boot confirms the form.

### 3.10 Slice 1a-i — what the timing pair can and cannot pin

Landed as `planner/timing_calibration.py`, 9 hermetic tests.

`PrefillTiming` models a stage as `fixed_ms[r] + ms_per_layer[r] * n_r`, and
one measured cut cannot separate the two — both fit the single point exactly.
`prefill_timing_from_measurement` therefore defaults `fixed_ms` to zero, the
**optimistic** end of the family, which is exactly why every speedup in this
document is an upper bound. Two cuts resolve it per rank by elimination.

Doing the arithmetic before the window rather than after changed what the boot
should ask for. The slice-1a pair is **three different calibrators at once**:

| rank | layers | dn | expected Δt | verdict |
|---|---|---|---|---|
| rank0 | 28 → 29 | +1 | ~1.76 ms | weak — the binding cost |
| rank1 | 20 → 19 | −1 | ~7.74 ms | strong |
| rank2 | 16 → 16 | 0 | — | **not calibrated at all** |

**rank2 is not calibrated by this pair**, and the solver refuses to report an
intercept for it rather than emitting the `fixed_ms=0` fallback, which at a
call site is indistinguishable from a measurement.

**Sample count is a precondition, not an afterthought.** The slope is a
difference of two means over the layer delta, so `stderr(s) = √2·σ/|dn|`, and
`dn = 1` puts the entire per-stage noise onto the slope. Because rank0's slope
is small, it is far more expensive to pin than rank1's:

| per-chunk SD | chunks for 10% on rank0 (1.757) | on rank1 (7.740) |
|---|---|---|
| 1 ms | 65 | 4 |
| 2 ms | 260 | 14 |
| 3 ms | 584 | 31 |
| 5 ms | 1620 | 84 |

Useful arithmetic: one max-length prompt is `327680 / 512 = 640` chunks, which
clears the 584 needed at SD = 3 ms — so **one full-length prefill per arm
suffices for rank0 at 10%**, provided the per-chunk SD is actually measured and
reported rather than assumed. `samples_needed()` answers this before the
window; discovering afterwards that the samples could never have supported the
claim is the expensive way to learn it.

Two further refusals, both because a confident-looking wrong number is worse
than none: a **negative intercept** is reported rather than clamped (it means
the linear model does not hold across the pair, most likely a non-constant
per-layer cost), and a **structural** refusal — this pair cannot calibrate that
rank — is raised before any data-dependent one, so the reader is not sent to
inspect timings when the pair was never capable of the answer.

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

### 4.2a COST MODEL — census numbers, replacing the plan's estimates

Landed as `planner/decoupled_kv.py`, 12 hermetic tests. Done before any
measurement (*Machbarkeit vor Messung*), and it moves two of the plan's numbers.

**Confirmed:** ~25 MiB per attention layer per 512-token chunk. From geometry:
Q 6.00 MiB + partial out 6.00 MiB + LSE 0.047 MiB, times 2 remote ranks =
**24.09 MiB/layer**, so **385.5 MiB per chunk** across the 16 layers.

**Corrected:** "+10-20% chunk cost" assumes a link rank0 does not have. The
same 385.5 MiB against a 320 ms serial chunk is:

| link | chunk cost | overhead |
|---|---|---|
| 12,000 MiB/s | 32.1 ms | +10.0% |
| 6,000 MiB/s | 64.2 ms | +20.1% |
| **3,000 MiB/s** (assumed rank0 PCIe x4 class) | **128.5 ms** | **+40.2%** |

> **WITHDRAWN (E5). The +40% row assumed the wrong card on the x4 link.**
> The authoritative mapping (user, 2026-08-17) is: **the 5090 is on x8, one
> 3080 on x8, the other 3080 on x4.** rank0 is the 5090 — it is the card
> measuring 1.757 ms/layer against the 3080s' 7.74/7.28 — so **rank0 is on
> x8**, and the 3,000 MiB/s row never applied to it. On measured links
> (x8 = 13 GB/s, x4 = 6.4 GB/s) the real overhead is **+6.3% to +27.5%**
> across the ladder (§4.2e), which makes the canonical plan's original
> "+10-20%" approximately right in mid-ladder and my "+40%" an artifact of a
> placeholder bound to a **rank index instead of a card identity**.
>
> The lesson generalises past this number: link data attached to a rank is not
> attached to anything, because torch device order and NVML order diverge on
> this rig. `overlap_schedule.CardLink` now keys reach by NVML UUID and PCI
> BDF and **refuses an unknown card rather than guessing positionally**.

> **Corrected (§4.2e).** This paragraph originally continued "the overlap
> argument is unchanged — there are 48 GDN layers of compute to hide behind".
> That is **wrong**, and it was wrong in the canonical plan's phrasing too. The
> GDN layers that follow an attention layer are *sequentially downstream of the
> very collective they were supposed to hide*: layer L+1 consumes layer L's
> attention output, so it cannot start until L's gather completes. See §4.2e
> for what actually hides and what does not.

**A third term, which the plan omits and which the coupled layout does not
pay: KV placement.** Under decoupling the two ownership axes are orthogonal — a
PP stage computes the layers it owns, but each token's rows are stored on the
rank owning that *token*. Every row a stage produces for a token it does not
own must therefore be shipped. Per attention layer a chunk produces
`512 × 2048` = 1.00 MiB, of which `1 − share_of_computing_stage` leaves; summed
over the 7/5/4 attention-layer split that is **11.1 MiB per chunk, ~2.9% on top
of the 385.5 MiB collective**. Small, but charged rather than waved away —
it is exactly the kind of term that is invisible in a design and obvious in a
profile. Solved by `kv_placement_bytes_per_chunk()`.

This also confirms #706's writer-side finding from my side: a canonical page
holds all 16 attention layers for one token, but those 16 slots are produced by
**three different PP stages** (7 + 5 + 4), so every page takes partial writes
from three writers regardless of which rank stores it. Decoupling does **not**
remove the need for his per-page completeness marker.

**Structural fact the estimate does not capture: traffic is driven by Q and the
partial OUTPUT, not by how much KV is remote.** Each remote participant
receives a full Q block and returns a full-size partial output plus LSE,
*regardless of whether it holds 1% or 99% of the shard*. So "shard less
aggressively to save bandwidth" does not work. The only levers are the number
of remote participants and chunk size relative to compute. Pinned by
`test_collective_traffic_is_independent_of_how_much_kv_is_remote`.

### 4.2e The overlap schedule — what hides, what does not, and what the fix is worth

Landed as `planner/overlap_schedule.py`, 10 hermetic tests. This is the
mitigation priced before a window, per *Machbarkeit vor Messung*.

"Overlappable behind GDN/FFN compute" is two claims, and they have opposite
answers:

| term | per attention layer | on the critical path? | hides behind the 3 following GDN layers? |
|---|---|---|---|
| Q-broadcast + partial gather + LSE merge | 24.09 MiB | **yes** | **no** |
| KV placement write | 1.00 MiB (× `1 − share`) | no | **yes, entirely** |

**Placement hides completely and stays hidden at every rung.** It is a write no
later layer in the chunk reads, so the ~37 ms of GDN compute following each
attention layer on rank0 absorbs its ~0.3 ms without effort.

**The gather does not hide at all within a chunk.** Layer L+1 consumes layer
L's attention output, so the compute that was supposed to hide L's gather
cannot begin until that gather has finished. With today's machinery the exposed
time *is* the gather time — not a fraction of it, all of it.

The only thing that hides the dominant term is **cross-chunk pipelining**:
overlapping chunk *c*'s gather with chunk *c+1*'s compute inside the same
stage, bounded by that stage's own per-chunk compute. It does not exist today.
**On MEASURED links and the authoritative card mapping** (x8 = 13 GB/s,
x4 = 6.4 GB/s; 5090 on x8, one 3080 on x8, the other 3080 on x4), overhead on
the pipelined chunk cost — `max_i(compute_i + exposed_i)` against
`max_i(compute_i)`:

| cut | compute only | with gather | overhead | with pipelining |
|---|---|---|---|---|
| `[28,20,16]` | 154.8 ms | 164.5 | **+6.3%** | +0.0% |
| `[32,16,16]` | 123.8 | 132.2 | +6.7% | +0.0% |
| `[35,14,15]` | 109.1 | 124.9 | +14.5% | +0.0% |
| `[38,13,13]` | 100.6 | 110.4 | +9.7% | +0.0% |
| `[41,11,12]` | 87.3 | 99.1 | +13.6% | +0.0% |
| `[44,10,10]` | 77.4 | 98.7 | **+27.5%** | +0.0% |

**Cross-chunk pipelining removes the exposure entirely at every rung**, because
each stage's own per-chunk compute (49–155 ms shallow, 72–77 ms deep) far
exceeds its gather (4–21 ms). But the quantity removed is **15.8–21.4 ms**, not
the 56–88 ms the refuted mapping implied — so it is a **deep-cut enabler, not
an incumbent necessity**: worth +27.5% at `[44,10,10]` and only +6.3% at the
incumbent.

> **Correction (E5): the "triple jeopardy" framing was wrong.** An earlier
> revision here read: *"rank0 is the worst stage under triple jeopardy — most
> attention layers, least compute to hide behind, and the slowest link."* The
> third disadvantage does not exist. rank0 is the 5090 and the 5090 is on x8,
> so the worst-rank exposure falls roughly fourfold and the deep cut stops
> looking unaffordable. Kept as `test_the_triple_jeopardy_framing_was_wrong`
> so it cannot be re-derived. The real lesson is not about links: **link data
> bound to a rank index rather than a card identity produced a confident,
> wrong analysis**, and torch/NVML order diverge on this rig.

**The placement rule, and its limit.** Putting the x4 card under the stage with
the *fewest attention layers* is worth a few ms **only while that stage binds**
— at the incumbent, 15.8 ms versus 19.7 ms for the wrong choice. By
`[38,13,13]` and deeper the **5090 binds on its own attention concentration**
(11 of 16 layers at `[44,10,10]`) whichever 3080 is slow, and the placement
choice stops mattering: both assignments give an identical 21.4 ms. So the rule
is real but shallow-rung-only, and it cannot be satisfied at both ends at once
— the fewest-attention stage is rank2 at the incumbent and rank1 at the deep
cut, while cards cannot be re-slotted at runtime.

Still self-labelled: stage compute away from the incumbent is **extrapolated**
with `fixed_ms = 0`, the optimistic end for hiding, so every "hidden" verdict is
an upper bound. Which *physical* 3080 sits on x4 remains to be resolved by
PCI-BDF/NVML identity — it is worth ~4 ms at shallow rungs and nothing at deep
ones, so it is low-stakes but must still be answered by identity, not by
ordinal.

## #699 — LIVENESS FROM PROGRESS, BECAUSE HEALTH 200 IS BLIND

On 2026-08-16 the server sat wedged for 52+ minutes while `/health` returned
200. **Two independent blind spots** produced that, and the second is the one
nobody had named. Landed as `managers/progress_liveness.py`, 16 hermetic tests.

**1. `/health` answers "is the process up", not "is work moving".** Known.

**2. The shipped watchdog is blind to admission wedges *by construction*.**
`create_scheduler_watchdog`
(`managers/scheduler_components/invariant_checker.py:536-540`) wires:

```
get_counter = lambda: scheduler.forward_ct
is_active   = lambda: (scheduler.is_initializing
                       or scheduler.cur_batch_for_debug is not None)
```

It only arms **while a batch exists**. An admission wedge is exactly the state
where *no batch exists while work is pending* (`#running-req: 0` on 90.6% of
prefill rounds in the specimen). So `is_active` is False for the entire wedge
and the timer never starts. **The watchdog was not slow — it was switched off.**
That is why a 52-minute wedge produced no alarm from either mechanism.

### The signal

Progress is **any one** of three counters advancing over a sliding window:

| counter | why it is required |
|---|---|
| `prefill_chunks` | the only one that moves in a long pure-prefill run — a 640-chunk prompt completes nothing for minutes |
| `decode_steps` | moves in pure decode, where no chunk is admitted |
| `completions` | moves when requests finish |

Requiring *all three* would call a healthy pure-prefill server wedged;
requiring only `completions` would do the same. That is precisely the trap, so
the rule is disjunctive.

### The refusals, which are what make the alarm worth listening to

* **An idle box is not a wedge.** Zero progress with zero pending work is a
  server with nothing to do. Only queue depth and pending tokens separate it
  from the wedge — the progress counters are *identical*. Alarming here is how a
  real alarm gets ignored.
* **A deliberate pause is not a wedge.** A flip is 2.0-4.2 s of legitimate
  silence (#690) and a maintenance hold is longer; an `inhibited` sample
  suppresses the verdict rather than tripping it.
* **A counter reset is not a wedge.** After a restart the counters return to
  zero, so a negative delta means *restarted*, not *stalled*. Reading it as a
  wedge would make every restart trigger another.
* **A cold start is not judged.** Fewer than two samples yields `unknown`.

### Wiring and policy

`build_liveness_is_active(scheduler)` replaces the shipped gate: it arms
whenever **work is pending**, not when a batch happens to exist — which is the
condition that actually distinguishes a wedge from an idle box, and the exact
inversion of the failure above.

The escalation is defined rather than implied: `confirmations` consecutive
wedged windows before an alarm (so one slow window cannot trip it),
`restart_after_alarms` before a restart, and a `restart_cooldown_s` so **a wedge
that survives a restart does not become a restart loop**. Restart can be
disabled outright.

`to_monitoring_dict()` exposes the **deltas and pending counts alongside the
verdict**, deliberately: an operator who sees only `wedged` cannot tell an
admission wedge from a stalled forward pass, and those need different responses.

Every threshold is a deployment input. The can-fail test is the real specimen
shape — health 200, zero progress, nonzero pending — which must alarm, and which
a health-200 check calls fine.

## #690 — WHAT THE FIXED FLIP COST IS MADE OF

The flip has been carried as a scalar (~2.0-4.2 s) with an unexplained
residual. It is not a scalar: the runtime already reports a five-way split on
every completed flip. Reading **765 unique `PHASE-FLIP DONE` lines** from the
boot captures decomposes it, with no boot required.

| component | median | share | what it is |
|---|---|---|---|
| read | 30.6 ms | 0.9% | seam read |
| exchange | 847.8 ms | 25.3% | rank-to-rank KV |
| write | 302.6 ms | 9.0% | seam write |
| **movers** | **1458.4 ms** | **43.4%** | GDN state + **weights arena refill** (`phase_flip_runtime.py:1843-1846`) |
| **cutover** | **670.7 ms** | **20.0%** | group routing, owner-rule refresh, component rebuild, scheduler swap (`:1147`) |
| residual | 99.2 ms | 3.0% | unattributed |

Totals: median 3357 ms, min 2401, max 4874 — the reported band. Both directions
agree (pp_to_tp 3298 ms, tp_to_pp 3417 ms).

**The premise is confirmed and then some.** `read + exchange + write` — the KV
seam move, the part everyone reaches for — is **35.3%**. Sixty-plus percent sits
outside it, and the **single largest component is `movers` at 43.4%**, which is
the H2D copy of the target layout's weight image. The residual is only ~3%, so
the "unexplained" part is small and the named parts are the story.

*(Medians do not sum, so the composite used by the solver adds to slightly more
than the median total; the per-event median of `(movers+cutover)/total` is
61.6%, against 63.4% for the composite. Both are honest and both are labelled.)*

### Ranked levers, priced in load rather than milliseconds

Per #677, flip cost sets the stability floor and the latency ceiling in
*opposite* directions, so a reduction **reopens refused configurations**. The
right unit for a lever is therefore the maximum feasible load, not the saving.
At a 10 s TTFT budget with an even phase split:

| lever | F after | max feasible ρ | reopens | price |
|---|---|---|---|---|
| baseline (measured) | 3.36 s | 0.66 | — | — |
| **L1 overlap movers behind the seam** (upper bound) | 2.21 s | **0.77** | **+0.11** | scheduling only |
| L1 (lower bound, fully contending) | 3.36 s | 0.66 | +0.00 | — |
| L2 phase-uniform vector (#704b) | 2.21 s | 0.77 | +0.11 | **costs ladder depth, n0 ≤ 37** |
| L3 cutover to its observed minimum | 2.73 s | 0.72 | +0.06 | mechanism unknown |
| **L1 + L3 (best realistic)** | **1.58 s** | **0.84** | **+0.18** | scheduling only |

**Three things fall out, and the first is the one to act on.**

1. **L1 and L2 deliver identical flip savings — 1150 ms, ρ 0.66 → 0.77 — but L1
   is free and L2 costs ladder depth.** For flip cost specifically, #704b's
   phase-uniform vector buys nothing that a pure scheduling change does not.
   (It still earns its keep on #703's cache-key problem, which L1 does not
   touch — so they are substitutes for *this* purpose only.) **Do L1 first.**
2. **L1's saving is bounded `[0, 1150] ms`, and the bound is not yet measured.**
   `movers` is H2D, and on this no-P2P rig the rank-to-rank `exchange` stages
   *through host* — so both legs traverse the same PCIe direction and may
   contend. If they fully contend the overlap saves nothing. Which end of the
   bound holds is a measurement, and it is the single highest-value thing the
   confirming window can capture.
3. **L1 + L3 makes ρ = 0.8 feasible.** #677 refuses ρ = 0.8 at *every* measured
   flip cost; at F = 1.58 s the ceiling rises to ρ = 0.84. That is the concrete
   form of the repricing: not "save 1.8 s" but "the load the rig currently
   refuses becomes servable".

**`cutover` deserves instrumentation before optimisation.** Its 24x spread
(43.5 to 1041.5 ms) is not the signature of a fixed cost — it looks like a wait
or a serialisation. Sub-step timestamps would say which, and the observed
minimum is the honest target rather than zero.

## #677 — PHASE WINDOW ECONOMICS: length solved, not set

A static window is wrong in both directions — too short at high load (the
backlog never clears) and too long at low load (decodes wait behind an empty
prefill window). Solved by `planner/phase_window.py`, 13 hermetic tests.

**The amortization argument.** Over a cycle `C = T_p + T_d + flips·F`, the work
arriving in `C` must clear in `C`, so `T_p = ρ_p·C`, `T_d = ρ_d·C` and

* **stability floor** `C ≥ flips·F / (1 − ρ)` — below it the backlog grows
  without bound however the windows are split;
* **latency ceiling** `C ≤ (budget − F) / ρ_d` — a request arriving just after
  the prefill window shuts waits out the whole decode window plus a flip.

Flip overhead is `flips·F / C`, which *falls* as the cycle lengthens. So
throughput always wants a longer window and **the economic choice is the
largest admissible cycle**, floored by stability — not a midpoint, and not a
constant.

Solved on this rig (10 s TTFT budget, 2 arrivals/s, batch queue 4):

| F | ρ | floor | ceiling | cycle | T_pre | T_dec | overhead | |
|---|---|---|---|---|---|---|---|---|
| 2.0 s | 0.30 | 5.7 | 53.3 | 53.3 | 8.0 | 8.0 | **7.5%** | |
| 2.0 s | 0.50 | 8.0 | 32.0 | 32.0 | 8.0 | 8.0 | 12.5% | |
| 2.0 s | 0.80 | 20.0 | 20.0 | — | — | — | — | **REFUSED** |
| 3.0 s | 0.50 | 12.0 | 28.0 | 28.0 | 7.0 | 7.0 | 21.4% | |
| 4.2 s | 0.50 | 16.8 | 23.2 | 23.2 | 5.8 | 5.8 | **36.2%** | |
| 4.2 s | 0.80 | 42.0 | 14.5 | — | — | — | — | **REFUSED** |

**The sharpest result, and it reprices #690.** The two constraints move in
**opposite** directions with `F`: the floor rises as `flips·F` while the ceiling
falls as `−F/ρ_d`. A dearer flip does not merely add an overhead line — **it
closes the feasible band from both ends**, and past some `F` the band shuts
entirely: *no window length works, at any split*. At ρ = 0.8 with a 10 s budget
this rig is already refused at every measured flip cost. So halving the flip
cost does not halve an overhead; it **reopens configurations that are currently
impossible**, which is a much stronger argument for #690 than "2-4 s is slow".

**Two refusals, because a policy that quietly does the impossible is worse than
one that stops.**

* **The seam must be able to arm.** If the layout's free column no longer clears
  its arming floor, there is no flip to schedule at any window length — this
  composes directly with #707's closed form and the n0 ≤ 51 depth bound it
  implies. Checked *before* any arithmetic.
* **The decode window must be worth entering.** Batch formation (#689) collapses
  toward size 1 below a queue threshold, so flipping early buys a fraction of
  the decode rate for a full flip cost. That is a floor on the cycle —
  `C ≥ q / (λ·ρ_p)` — and at light load it **binds instead of stability**, which
  is exactly the regime where a static window over-flips.

Every quantity is injected; the rig's figures are calibration data in the test,
with a foreign profile (flip 0.05 s, ρ = 0.8, 2 s budget, 50 arrivals/s) pinning
the generality.

## #702 — THE PP CUT SOLVED FOR PREFILL SPEED

The user's question, unowned since 2026-08-16: **more layers on the 5090 — what
does it cost?** The capacity solver answers "what cut holds the most context",
which is a different objective, so it never covered this. Solved by
`planner/prefill_frontier.py`, 9 hermetic tests.

Three prices are charged against every candidate, because quoting only the first
is how a cut gets recommended that cannot serve.

| cut | attn | compute | coupled pool | decoupled pool | ovh | **net now** | **net + lever** | lever | pool |
|---|---|---|---|---|---|---|---|---|---|
| `[28,20,16]` incumbent | (7,5,4) | 1.000x | 436,275 | 513,875 | 6.3% | 0.941 | 1.000 | | **MEASURED** |
| `[31,16,17]` | (7,4,5) | 1.250x | 397,957 (−9%) | 513,875 | 15.8% | 1.079 | 1.250 | | extrap |
| `[33,15,16]` | (8,4,4) | 1.330x | **415,859 (−5%)** | 513,875 | 13.6% | 1.171 | 1.330 | | extrap |
| `[34,15,15]` | (8,4,4) | 1.333x | 382,106 (−12%) | 513,875 | 7.6% | 1.239 | 1.333 | | extrap |
| `[36,14,14]` | (9,3,4) | 1.429x | 336,833 (−23%) | 513,875 | 8.6% | 1.316 | 1.429 | | extrap |
| `[38,13,13]` | (9,3,4) | 1.538x | 276,827 (−37%) | 513,875 | 9.7% | 1.403 | 1.538 | | extrap |
| `[40,12,12]` | (10,3,3) | 1.667x | 246,610 (−43%) | 513,875 | 6.7% | 1.561 | 1.667 | | extrap |
| **`[42,11,11]`** | (10,3,3) | 1.818x | 192,604 (−56%) | 513,875 | 9.5% | **1.660** ← best now | 1.818 | | extrap |
| `[43,10,11]` | (10,3,3) | 1.934x | 165,601 (−62%) | 513,875 | 18.7% | 1.630 | 1.934 | **YES** | extrap |
| **`[44,10,10]`** | (11,2,3) | **2.000x** ← peak | 172,791 (−60%) | 513,875 | 27.5% | 1.569 | **2.000** ← best w/ lever | **YES** | extrap |
| `[47,6,11]` | (11,2,3) | 1.915x | 99,146 (−77%) | 513,875 | 26.4% | 1.489 | 1.915 | **YES** | extrap |
| `[51,1,12]` | (12,1,3) | 1.727x | 43,767 (−90%) | 513,875 | 26.0% | 1.371 | 1.727 | **YES** | extrap |

Pool now comes from **Slot-2's #707 closed form**, not from an extrapolation of
mine: `allowed_tokens = id_space + (free_at_measure − arming_floor − margin) /
cell`, with `holdback_frac = 1 − allowed / (profiled/cell)`. It reproduces the
instrument boot's reported holdbacks (45.143 / 44.074 / 60.258 %) to **0.000 pp**.
Only the layout *shift* of `free_at_measure` is extrapolated — per family,
374.2 MiB per attention layer and 476.2 per linear, plus 51.20 per GDN layer.
Everything else is exact, and the incumbent row is **measured**.

**Three things the closed form changed, one of them a correction to me.**

1. **Coupled is far kinder than I reported.** My extrapolation put `[33,15,16]`
   at −17% and `[42,11,11]` at −85%. The truth is **−5%** and −56%, because
   `allowed_tokens` is floored at `id_space`, which does not shrink with the
   cut. So the first few layers onto the 5090 are nearly free in the regime that
   exists **today** — `[33,15,16]` buys 1.330x for 5% of context, without
   decoupling at all.
2. **The seam cap bounds the depth.** Past `n0 = 51` a rank's free column no
   longer clears its arming floor, so the layout **cannot arm a flip**. Those
   cuts are refused by the provider and never reach the frontier — absent, not
   priced as a tiny pool. An extrapolation cannot produce that boundary.
3. **Both optima are interior, for two different reasons.** Without the lever,
   overhead outgrows the compute gain past `[42,11,11]`. *With* it, the raw
   compute speedup itself peaks at `[44,10,10]` and then falls — piling layers
   onto the fast card eventually makes a **tail stage** the bottleneck. "More
   layers on the 5090" has a limit that is not about memory at all.

**Why the binder holds back most, and it is not waste.** PP2's bracket
(`free_at_measure − arming_floor − margin`) is **0.0 MiB** — it sits exactly at
its arming floor — against PP0's 2164.8 and PP1's 260.2. So its allowed tokens
collapse to `id_space`, i.e. it *sets* the pool. It simultaneously has the
smallest cell (4 attention layers), hence the largest raw capacity and therefore
the largest holdback *fraction*. Binding and holding back most co-occur **by
construction**. Reading PP2's 60.3% as waste is exactly backwards. And the TP
pass holds back 0.000% on every rank for the matching reason: there is no flip
to arm from.

**Two decisions fall out, and they are independent.**

**1. Coupled or decoupled — and coupled buys the first steps cheaply.** In the
regime that exists today `[33,15,16]` costs only **5%** of context for 1.330x,
and `[31,16,17]` costs 9% for 1.250x. The decline is real but not a collapse
until depth: `[42,11,11]` costs 56%. Under decoupling (#704b) the pool is **exactly
cut-independent at 514,034 — +17.7% over the incumbent's observed 436,766** —
because total weight bytes and total GDN state are invariant under a re-cut;
only their distribution moves. **So the pool price of depth is not merely
affordable under decoupling, it is negative.** That inverts the usual framing:
decoupling is not a cost centre bought for speed, it is what makes the speed
free of context loss.

**2. With or without the pipelining lever — and this one has a trap.** Net
speedup without cross-chunk pipelining **is not monotone in depth**. It peaks at
`[42,11,11]` (1.660x) and then *falls*: 1.630x at `[43,10,11]`, 1.569x at
`[44,10,10]`. Past the peak the collective overhead grows faster than the
compute gain, so a deeper cut is **actively worse**, not merely diminishing.
A frontier reporting only compute speedup would recommend exactly those cuts —
which is why they carry `needs_pipelining`.

**Recommendation, stated as a pick rather than a verdict:**

* **Today, decoupled, no lever:** `[42,11,11]` — **1.660x net at +17.7% pool.**
* **With the lever built:** `[44,10,10]` — **2.000x net at +17.7% pool.** The
  lever is worth the last 0.34x, and nothing else.
* **Coupled, available TODAY with no new machinery:** `[33,15,16]` — **1.330x
  for −5% pool**. This is the cheapest real win on the whole frontier and it
  needs neither decoupling nor the lever. `[34,15,15]` gives 1.333x for −12%.
  Past `[38,13,13]` (−37%) the context loss stops being a trade.
* **Do not pick `[28,17,19]`**: decoupled, it is **net 0.980x — slower than the
  incumbent**, because at incumbent depth the collective buys nothing and still
  costs its overhead. Decoupling does not pay for itself until roughly
  `[29,17,18]`.

**Caveats, all self-labelled and none folded into the numbers.** Every speedup
is an **upper bound** until slice 1a-i lands the timing intercept (`fixed_ms`
defaults to zero, the optimistic end). Pool figures use the four-boot gate's
metal `available_bytes` at the incumbent, extrapolated across cuts by weights
and GDN state, and carry the **unquantified reserve term** pending Slot-2's
instrument boot — the solver reports `measured=False` for exactly this reason.
The overhead column uses measured links with the authoritative card mapping
(§4.2e). No rig constant appears in the solver; this rig's figures are
calibration data in the test.

### 4.2g CROSS-CHUNK PIPELINING DESK SPEC — the exposure lever

Written against the code, no changes made. Removes 15.8–21.4 ms of exposure per
chunk (§4.2e), i.e. +27.5% at `[44,10,10]` and +6.3% at the incumbent — a
**deep-cut enabler, not an incumbent necessity**, which is how it should be
ranked against other work.

**Why it is the only lever left.** Within one chunk the gather cannot hide:
layer L+1 consumes layer L's attention output. The only independent work
available is the *next chunk*, whose layers 0…L−1 do not depend on chunk N at
all once the previous PP stage has delivered its activations.

**Schedule.** Interleave two chunks at attention-layer granularity. When chunk N
reaches attention layer L it issues its Q-broadcast and posts the gather on a
side stream; the stage then advances chunk N+1 until *it* reaches an attention
layer; then it returns to collect chunk N's merge. Steady state keeps exactly
two chunks resident and one gather in flight, which is sufficient — the gather
(4–21 ms) is smaller than a stage's per-chunk compute (49–155 ms shallow,
72–77 ms deep), so one chunk of lookahead already covers it. Deeper lookahead
buys nothing and costs buffers.

**Buffers and staging** (per stage, small): two hidden-state buffers and two
residual buffers at `512 × 5120 × 2 B` = 5 MiB each, plus per-in-flight-gather
Q/output staging at 6 MiB each — roughly **32 MiB total**, negligible against a
GiB-scale pool. This is a scheduling change, not a memory one.

**Interaction with the captured-graph route.** §3.11a established that a CUDA
graph bakes the executed layer *set*; a graph also bakes the launch *sequence*,
and an interleaved two-chunk schedule is a different sequence than a
single-chunk pass. So a graph captured for the single-chunk pass **cannot
replay** an interleaved one. Prefill already runs the **breakable** route
(`model_executor/runner_backend/breakable_cuda_graph_backend.py`), which exists
to be interrupted, and its replays already register with the abort gate
(`:462`, `barlink_abort_gate.note_replay("breakable", shape_key)`). The
attention layers are the natural break points, so the breakable route is the
host for this and the full-graph decode route is untouched.

**The 512-token chunk boundary.** Pipelining needs a successor chunk, so it
covers every chunk of a prompt except the last. A max-length prompt is 640
chunks, so 639 hide and one is exposed — immaterial. A **single-chunk prompt
gets no benefit at all**, which is worth stating because short-prompt latency is
exactly where an overhead is most visible.

**Failure and abort semantics — the sharpest constraint.** The existing
contract (`managers/scheduler.py:3821-3834`) is that `abort_request` only
*records* the target in `_pending_chunked_abort_req` because tearing down
mid-iteration is unsafe; `process_pending_chunked_abort` then clears
`chunked_req` at the top of the step so the **next** chunk does not launch,
while the **already-launched** chunk drains as its result resolves. The
docstring notes that under overlap the result lands a step later and
`inflight_middle_chunks` accounting keeps it straight. Cross-chunk pipelining
widens that window from one launched item to two, with three consequences:

1. **A collective cannot be aborted unilaterally.** The gather spans ranks, so a
   rank that skips it leaves its peers blocked in the collective — an abort
   becomes a hang. Abort must therefore be *collective-safe*: either every rank
   aborts at the same chunk boundary, or the in-flight gather is allowed to
   complete before teardown. The second is simpler and bounded by one gather
   (≤21 ms). `barlink_abort_gate.check_aborts` is the existing place that
   polices exactly this class.
2. **KV and metadata must not be freed under an in-flight gather.** Chunk N's
   rows may still be read by a peer computing its partial; releasing them
   (`release_kv_cache`, `maybe_release_metadata_buffer`) before the gather
   resolves is a use-after-free that would surface as wrong numbers, not a
   crash.
3. **`inflight_middle_chunks` accounting must count the deeper window**, or the
   drain bookkeeping under-counts by one and the aborted chunk is either double
   -dropped or never dropped.

**It invalidates the stage-1 timing calibration, and that ordering matters.**
§3.10 solves the per-layer slope and fixed per-stage intercept from measured
per-chunk stage times, which assumes a chunk's wall time *is*
`fixed + slope × layers`. Under pipelining a chunk's wall time includes
overlapped work from its neighbour and becomes a steady-state throughput
figure instead. **So slice 1a-i must be measured with pipelining OFF**, or the
solve is fitting a different quantity — and since 1a-i is the boot that converts
every speedup in this document from an upper bound into a prediction, that
ordering is not negotiable: calibrate first, pipeline second.

### 4.2f B1 DESK SPEC — the cross-PP-stage group

Written against the code, no changes made. Engine work does not start while the
boot cycle is owned elsewhere; this exists so it is ready-to-build when it
frees.

**The membership already exists; the TYPE does not.** With `tp_size=1`,
`pp_size=3` (confirmed from the restored boot: `tp_size=1 pp_size=3
dcp_size=1`), `world_size=3` and the PP group is built as
`range(pp_group_idx, world_size, world_size // pp_size)` =
`range(0, 3, 1)` = **`[0,1,2]`** (`parallel_state.py:3355-3372`). That is
already every stage. Generally, for `tp_size > 1` there is one PP group per TP
position and its members are that position's pipeline ranks — which is exactly
the set across which KV should be token-sharded. **So no new rank layout has to
be invented: the required membership is the PP group's, at every topology.**

**But `_PP` cannot be used as-is, and the reason is a silent-wrongness path.**
The attention backends read `attn_dcp_size` / `attn_dcp_rank` **once in the
constructor and cache them** on the instance; `dcp_enabled` is
`get_dcp_group_no_assert() is not None and dcp_size > 1`
(`dcp_group_guard.py:14-19`). Handed a communicator that is not a
**DCP-typed** group, they cache `dcp_size=1`, `uneven_dcp_owner_bounds()`
returns `None` on **every** rank, the owner rule is bypassed, and every rank
treats every global slot as its own local row — all ranks write the same token
to the same row, last write wins, each reads the whole sequence as local.
`dcp_group_guard.py:21-27` states the consequence exactly: *"The result is
silently wrong output. There is no error and no hang to follow."*

So B1 is: **a DCP-typed group whose rank sets come from the PP dimension
instead of from chunking TP groups.**

**Change point.** `parallel_state.py:3143-3159` builds `_DCP` by chunking each
TP group:
```
for tp_group in group_ranks:
    for start in range(0, len(tp_group), decode_context_parallel_size):
        dcp_group_ranks.append(tp_group[start : start + decode_context_parallel_size])
```
With `tp_size=1` each TP group is a single rank, so every DCP group is a single
rank — which is why PP prefill runs `dcp_size=1` with no usable group. The
decoupled path needs those rank sets derived from the PP layout.

**Ordering constraint, and it is load-bearing.** `_DCP` is built at `:3152`;
`_PP` is built at **`:3365`, afterwards**. So the new construction **must not
read `_PP`** — it must recompute the same pure arithmetic inline
(`range(idx, world_size, world_size // pp_size)`). Reading a group that does not
exist yet is the ordering class the DCP guard was written to catch.

**Creation point relative to the guard.** `assert_dcp_group_formed`
(`dcp_group_guard.py:63`) compares `server_args.dcp_size` against what a backend
constructed *now* would cache, and refuses on mismatch. So the group must exist
**before attention-backend construction**, and `server_args.dcp_size` must be
set to the number of participating stages. The guard is then a no-op — and if
the ordering is ever broken it names the construction step rather than failing
downstream.

**Collectives that run on it:** Q broadcast; partial-output + LSE all-gather via
the existing `cp_lse_ag_out_ar_mha_uneven` (`layers/dcp/comm.py:228-262`, N-way,
rank-ordered by the all_gather); and the KV placement writes (§4.2a).

**Refusal conditions, each traced to an existing guard:**

| condition | refuse because | citation |
|---|---|---|
| `dcp_size` ≠ participating stage count | backends would cache the wrong size and bypass the owner rule silently | `dcp_group_guard.py:63-100` |
| `page_size != 1` | the owner rule `L % S ∈ [lo,hi)` is defined on single-token slots; a paged allocator has no owner for a page | `dcp_group_guard.py:170-177` |
| `disaggregation_mode == "decode"` without uneven-TP + mooncake | stock head-sharded DCP receive is refused on first transfer | `dcp_group_guard.py:139-166` |
| `tp_size > 1` with an ambiguous shard set | membership must be one PP group per TP position, not a mixture | new — no existing guard |

Note `page_size == 1` is **already mandatory** for Option A independently
(#706: a multi-token page would span owner ranks), so the two constraints agree
rather than compete.

**Still open, routed to the survey in flight:** the #616 group-MIN-floor
interaction, whether an extra communicator carries a budgeted per-group cost,
and whether any census/registry must be told about a new group. Those are
recorded as unknown rather than assumed benign.

### 4.2b The prize and the purpose contradict — quantified

The phase-uniform vector is the TP vector `[14,10,8]`, i.e. shares
0.4375 / 0.3125 / 0.25. That puts **43.75% of all KV rows on rank0** — and
rank0 is exactly the rank a deep PP cut loads with weights. Against
free-for-KV measured at the `[28,20,16]` boot (`2 × K_size + available_gpu_mem`
= 10.01 / 6.19 / 5.74 GiB):

| cut | rank0 free | rank0 needs | verdict |
|---|---|---|---|
| `[28,20,16]` | 10.01 GiB | 5.83 GiB | fits |
| `[35,14,15]` | 6.93 | 5.83 | fits |
| `[38,13,13]` | 5.61 | 5.83 | **infeasible** |
| `[42,11,11]` | 3.85 | 5.83 | **infeasible** |
| `[44,10,10]` | 2.97 | 5.83 | **infeasible** |

**Depth ceiling under the phase-uniform vector: n0 ≤ 37**, solved by
`deepest_feasible_rank0_layers()`. So `[42,11,11]` and `[44,10,10]` — the cuts
decoupling exists to unlock — are precisely where the phase-uniform vector
fails. At `[44,10,10]` the free-proportional vector is `[0.135, 0.483, 0.382]`,
nearly the **inverse** of the TP vector on rank0, because rank0 is weight-rich
under a deep PP cut and weight-light under TP width-sharding. The two phases
genuinely want opposite vectors; this is not a tuning accident.

**Resolution: split the prize, because the halves have different prices.**

* **Structural uniformity** — both phases token-shard the same 16 layers, so
  the layout *kind* matches. This is what a content-addressed cache key needs,
  so it solves #703. It is **free**, and is taken unconditionally.
* **Vector identity** — both phases use the same *shares*, so no rows move at
  the seam. This additionally deletes #690's fixed flip cost. It **costs
  depth**, capping the ladder at n0 ≤ 37 (~1.5x pipelined instead of 2.0x).

Conflating the two would pay for both or get neither, which is why
`seam_rebalance_bytes()` reports them separately: identical shares return zero
(the #690 prize), differing shares return a rebalance rather than a re-layout
(the #703 prize retained regardless).

**OPERATOR DECISION (adopted).** Structural uniformity is taken
unconditionally. Vector identity stays **open** until slice 1a-i's timing
intercept gives the flip-frequency break-even — it is an empirical resolution,
not a values call, so it is not escalated yet; if the break-even lands
ambiguous it becomes a user-level fork (seam-free flips vs ladder depth).

**Consequence, and it inverts the default:** because vector identity may lose,
the mechanism is designed **free-proportional-primary**, with vector identity
as the special case — not the other way round. The structural half holds either
way.

### 4.2d Under free-proportional, the vector does NOT follow the cut

Free-proportional raises a question vector identity did not: the free vector
changes with the cut, so does the share vector follow it? If it did, every rung
change would rebalance KV rows — reintroducing exactly the cost decoupling was
meant to remove.

It does not have to. Solved by `fixed_vector_for_ladder()`:

| approach | rung-change cost | feasible at every rung? | pool range |
|---|---|---|---|
| vector follows each rung | **0.80–1.07 GiB per step**, 3.48 GiB across the ladder | yes | 492k–719k |
| **vector fixed once** | **zero rows moved** | **yes, all rungs** | 492k–719k |

Fixing it strictly dominates: identical pools, no rebalancing, and **every rung
exceeds the coupled incumbent's 436,766** — 492,393 at the shallowest rung,
718,930 at `[44,10,10]`. So a rung change moves no weights (§3.8) *and* no KV
rows, under the primary case as well as the special one.

The vector is built from the **per-rank minimum of free memory across rungs**,
not from any single rung's free vector, because there is no single tightest
rung: deepening starves rank0 while *freeing* rank1 and rank2, so each rank's
binding constraint comes from a different rung. That candidate is then verified
per rung — it is the natural construction, not a theorem, and the solver says
so.

### 4.2c Shared layout contract with #706 — **AGREED**

The same-key world must hold on **both** the host format (#706) and the device
layout (#704b), so the text below is carried **verbatim in both design docs**.
**Status: AGREED with the #706 strand.** He frames it as the stronger form of
his #241 invariant — canonical bytes depend on no geometry, therefore the key
carries none. The device-side layout code is unblocked on this basis.

> A cache key identifies CONTENT, never placement. The key is a function of the
> model identity, the token sequence, and semantic modifiers that change the
> values (RoPE scaling, quantisation of the stored KV). It MUST NOT encode any
> layout fact: not the PP cut, not the token-shard vector, not the page
> geometry, not which phase produced the bytes. Placement is carried separately
> as a LayoutDescriptor alongside the entry, used only to scatter and gather.
>
> The stored (host) form is CANONICAL and layout-neutral: one defined byte
> order — layer-major, then token, then kv-head, then head_dim — that any
> device layout can scatter from and gather into. A writer converts from its
> device layout into canonical form on store; a reader converts from canonical
> into its own device layout on load. Neither side stores its private device
> order.
>
> Consequence: bytes written by PP-prefill are readable by TP-decode and vice
> versa, at the same key, with no re-keying and no second geometry.
>
> **(a) Draft pages are excluded, by name.** `{hash}.draft` pages cannot be
> made geometry-neutral by any suffix rule: draft KV is the exact MIRROR of
> target KV — head-SHARDED and token-COMPLETE — and the draft worker exists
> only in the TP decode phase. Draft pages therefore stay phase- and
> rank-specific, and the draft pool starts **cold** after a flip or reboot. A
> partial `#cached-token` share is the DESIGNED shape, not a defect. Named and
> accepted rather than silently inherited.
>
> **(b) The mamba/GDN canonical form is DEFINED BY the #706 mamba spec, and is
> not restated here.** 48 of 64 layers are GDN, so the attention ordering above
> is silent on most of the model and must not be read as covering it. The
> definition is `MambaPool.get_conv_subblock_spec` (`memory_pool.py:1226`,
> returning `(sub_block_full_sizes, units, conv_dim)`) together with
> `layer_extents()` / `MambaBlobSpec.for_layers()` (#706, `e77a1e35a2`). Any
> implementation MUST call that spec rather than compute offsets locally.
>
> **The trap it exists to prevent:** a mamba layer range is **TWO DISJOINT byte
> ranges** — a temporal region and a conv region — and the conv region is three
> independently head-sharded `[query_key | query_key | value]` sub-blocks, so a
> rank's conv shard is three concatenated ranges, not one contiguous slice.
> **Cutting it as one flat range returns the right NUMBER of bytes and the
> wrong channels**, which is a silent-corruption class, not a crash.

If the key encoded the share vector we would lose structural uniformity for
free and gain nothing — hence "descriptor, not key".

**Page shape: OPTION A**, settled by #706. A page carries **all 16 attention
layers for its token range, layer-major within the page**. Grounds, all his:
`page_size == 1` is mandatory (required by `dcp_owner_mode`, since a
multi-token page would span owner ranks), so a page is ONE token — Option A is
a 32,768 B object where the per-(layer,token) alternative would be 2,048 B;
that alternative implies ~7.0M sub-4-KiB objects, below the filesystem block
size, giving ~2x space amplification and IOPS that would make disk-L3 slower
than re-prefilling; and `memory_pool_host.py:793` already allocates
`(num_host_pages, layer_num, item_bytes)`, so Option A only widens `layer_num`
to 16 and globalises `start_layer`.

Cost of Option A, stated honestly because **the writer pays**: each PP stage
partial-writes its own layer slots at byte offsets within a shared page, so a
per-page **completeness marker** (slot bitmap header, or rename-on-complete) is
required and **does not exist yet** — it must be built.

Sizing: a full 436,766-token pool is `16 × 2048 × 436,766` = **13.3 GiB** of
canonical attention bytes world-wide, independent of sharding.

**Known gap inherited from (b), and it is large.** The existing mamba spec cuts
by **heads**, for TP mismatch. The PP phase shards **layers**, so a stage's
`.mamba` blob is layer-partial rather than head-partial, and
`get_conv_subblock_spec` does not address that axis at all. A **layer** cut for
mamba that composes with the head cut is new work, and #706 records it as the
single largest piece in his design. #704b depends on it for any prefix that
must survive a phase change, because 48 of 64 layers are GDN and a prefix
cannot be resumed from attention KV alone.

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
| **1a-i — ticket + calibration harness ready** | Two STATIC boots, `[28,20,16]` ↔ `[29,19,16]`, one variable (`--pp-layer-ratio`). **Zero new code.** Both are (7,5,4), layer 28 is linear, so no attention layer and no KV row moves | 1 short window | A1 pool within 2.5k of 436,766; A2 PP2 binds; A3 K sizes unchanged. Plus **T1: the second timing cut** |
| **1a-ii — desk-complete** | Union arena (`weights_arena_union.py`, 13 tests) + boundary actuator (`layout_boundary.py`, 14 tests): load the union per rank, size the arena for the union, mutate `_start_layer`/`_end_layer` at quiescence with observer rollback. Remaining for a window: boot-time union load + the dependent-structure observers | 1 window | flip completes, output byte-identical across a flip and back, graphs survive |
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
