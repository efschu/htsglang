# DESIGN 704 — prefill layout ladder + PP-KV decoupling

Status: design settled, Slice 0 landed green, Slices 1-2 specified.
Worktree: `/spinning/wt-704-ladder`, branch `feat/704-prefill-ladder`,
based on `0f9e9993c4` (Slot-2's corrected phase-separated solver).

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

Measured consequence on this rig, from the solver in this branch:

| layout | pipelined prefill | pool (live-equiv) |
|---|---|---|
| incumbent `[28,20,16]` | 1.000x | 434,878 (observed) |
| best coupled rung `[31,17,16]` | 1.000x | 518,433 |
| deepest coupled rung `[41,11,12]` | 1.507x | 254,687 |
| **decoupled `[44,10,10]`** | **2.000x** | **539,017 (+23.9%)** |

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
census yields **4096 B (bf16)** and lands at 0.83 of the observed pool, binding
on rank1 — squarely inside the census's own known ~14-17% pessimism.

Both forms agree on *total* bytes per token, because both were fitted to the
same observed pool. They disagree on the **distribution across ranks**, which
is the only thing a cut solver produces. The flat form is a fit artifact.

Consequence, and it is not academic: the two forms diverge exactly in the
region the ladder operates in. Under the corrected rule `[33,15,16]` scores
**1.330x pipelined at pool 457,604** — better than `[32,16,16]` on *both* axes.
See §6 for the boot-arm implication.

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

1. **The shallow rungs get poorer.** For a rank0 span of [31, 38] the top
   rung's pool falls from 518,433 to **390,700** live-equivalent — *below* the
   incumbent's observed 434,878. Enabling the ladder costs ~10% pool at the top.
2. **Two rungs with the same attention profile become identical in pool.** With
   free memory pinned by the arena, a rung's pool depends only on its attention
   count. So `[33,15,16]` and `[35,14,15]` — both (8,4,4) — price *exactly the
   same*, and the faster one strictly dominates. **Under an arena the ladder's
   real axis is the attention-count vector, not the raw layer cut.**

Point 2 invalidates my own earlier Slice 1 proposal. I had picked
`[33,15,16] ↔ [35,14,15]` precisely because it moves zero attention layers —
which is exactly what makes it worthless *as a ladder pair*: the controller
would simply sit on the faster rung forever. It remains a fine pair for proving
the **mover** mechanically (weights move, no KV moves), and that is now its only
stated purpose. A pair that actually trades must cross an attention boundary,
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
coincide only if the TP vector is itself free-proportional; where they diverge,
**TP identity wins**, because a phase-uniform layout is worth more than a
locally optimal shard.

### 4.3 Cost

~25 MiB of collective traffic per attention layer per 512-token chunk,
estimated at +10-20% chunk cost, overlappable behind GDN/FFN compute. On this
no-P2P rig (PCIe x4/x8/x8, all PHB) the collective floor already dominates
decode, so this estimate is the least trustworthy number in the document and is
the first thing Slice 3 measures.

### 4.4 Correctness gate — mandatory

**Byte-identity A-vs-A for the decoupled attention path.** Same prompt, same
seed, decoupled vs coupled, byte-identical output. Test inputs sampled on
**CPU** and moved to device — `torch.randn` on-GPU is not architecture-identical
across the 3080s and the 5090, and this rig has already been bitten by that.

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

## 6. Boot-arm implication — for F4-r4, before the next window

`[32,16,16]` was selected under the flat all-layer rule. Under the corrected
family rule, **`[33,15,16]` dominates it on both axes** — 1.330x pipelined
versus 1.250x, at pool 457,604 versus 475,012 model-equivalent, with an
identical (8,4,4) attention profile. `[34,15,15]` is the same attention profile
again at 1.333x.

Two honest caveats, both of which argue for booting rather than re-deriving:

1. The corrected rule is verified from allocator code and dimensional analysis,
   **not yet on metal**. Both rules agree the arm-B class fails; they diverge in
   the plateau.
2. `PrefillTiming.fixed_ms` defaults to zero (`pp_cut.py:2206-2213`), the
   optimistic end of the family. **Every speedup in this document is an upper
   bound**, not a prediction, until a second measured cut pins the intercept.

Which makes the recommendation a single arm that does double duty: **boot
`[33,15,16]`**. It tests the corrected rule where it disagrees with the flat
one, and it supplies the second calibration point that converts every speedup
here from an upper bound into a measurement. A cut of `[32,15,17]` would
discriminate the two rules even more sharply (the rules disagree by ~15% and in
*opposite directions* there), but it is a worse operating point; `[33,15,16]`
is the better arm because it is worth running on its own merits.

This recommendation is **independent of everything in §3.7**: it is a single
static cut, no ladder, no arena, no mover. Nothing in the actuator survey
touches it. It is ready for a boot window now.

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
