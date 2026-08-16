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

### 2.3 What I have NOT yet verified

The actuator survey of the ladder/regime/reshard machinery (`kv_pressure_ladder`,
`corridor_guard`, `regime_stages`/`build_regime_stage_table`, `kv_reshard`) and
of the DCP/LSE/weightless attention path is in flight with two explorers. **The
single highest-risk unknown is whether a runtime weight mover exists at all**
(see §7 R1). Slice 1's shape depends on that answer and this document will be
amended, not quietly reinterpreted, when it lands.

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

Reuse, not reinvention: uneven-DCP owner-rule kernels (#62/#116), LSE merge and
overlap (#128), weightless-rank attention-over-remote-KV (#115), and the
cross-stage collectives already present in the flip runtime. The second
explorer is confirming which of these are production-wired versus test-only;
`layers/dcp/test_weightless_kv_math.py` in particular smells like a math check
with no runtime wiring, and that distinction decides how much of Part B is
assembly versus construction.

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

---

## 7. Open risks

- **R1 (highest) — does a runtime weight mover exist?** The ladder's entire
  actuation depends on relocating ~451-901 MiB of weights between ranks at
  runtime. #363's regime dial names a "weight mover" as its expansion stage; a
  named table entry is not an actuator. If it is a counter, Slice 1 grows by
  the cost of a real mover and the effort estimate below is wrong. **In flight.**
- **R2 — link reach is a placeholder.** Move times assume 3000 MiB/s on rank0's
  x4. Under live serving load the real figure may be materially lower, which
  would widen the hysteresis bands and could prune the cheaper steps outright.
  The solver already handles this correctly (it prunes); the risk is that the
  ladder is shorter than advertised.
- **R3 — census pessimism (~17%) is a scale factor, not a proven constant.** All
  live-equivalent numbers carry it. Rankings are robust; absolutes are not.
- **R4 — LSE merge order.** §4.4. The most likely silent-wrongness in Part B.
- **R5 — cross-PP-stage collectives.** Part B needs KV shards on ranks in a
  *different* PP stage than the layer owner. If the process groups are strictly
  intra-stage, group construction must change, which is a materially larger
  slice. **In flight with the second explorer.**

---

## 8. Slice plan

| slice | content | effort | gate |
|---|---|---|---|
| **0 — landed** | Family-split pool rule + ladder solver, hermetic, 3 hardware profiles | done | 26 tests green |
| **1** | Wire the ladder to the existing pressure-ladder/regime actuator; two-rung ladder (`[33,15,16]` ↔ `[35,14,15]`, same (8,4,4) profile, **zero attention layers moved**, weights only) | 1 window, **pending R1** | hysteresis proven on metal, no oscillation under load |
| **2** | Measure the real weight-mover cost on the pair matrix; replace the placeholder link figure; re-solve | 0.5 window | move time within 20% of solved |
| **3** | Decoupled attention path behind a flag: Q-broadcast + partial + LSE merge, coupled fallback | 2-3 windows, **pending R5** | byte-identity A-vs-A green |
| **4** | Phase-uniform token vector; retire the seam KV move | 1-2 windows | flip cost measurably below the 2.0-4.2 s baseline |

Slice 1 is deliberately the cheapest metal-provable cut: a rung pair inside the
attention plateau, so it proves the mover and the hysteresis **without moving a
single KV byte** — the two mechanisms that everything else depends on, isolated
from the one that is hardest to get right.
