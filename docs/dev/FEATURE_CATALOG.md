# Feature Catalog — what this fork already has

Read this BEFORE searching the tree or building anything: most capabilities you
are about to look for already exist. Rules: (1) never declare something
"impossible" or "missing" without checking this file, FEATURES_VS_UPSTREAM.md
and `git log`; (2) whoever merges a new feature updates the matching section in
the SAME merge. Last full refresh: 2026-08-02 (tip 33148dbe0f).

## 1. Uneven parallelism (core differentiator)
- **Uneven TP** `--rank-tp-ratio` + `--rank-gpu-id`: per-card weight shards.
  `auto` = byte-proportional from NVML totals minus auto reserve; with
  `--rank-perf-tune both|dec|enc|maxkv` the planner solves the vector.
  Unit system: `tp_units`/`tp_family` per layer class (16-element MLP family;
  coupled-dim rule: gate_up output and down_proj input partition the SAME
  intermediate dim and must coarsen identically); per-layer family table for
  `block_configs` models (Nemotron-Puzzle class).
- Sibling flags: `--rank-mlp-ratio`, `--rank-vocab-ratio`, `--rank-moe-ratio`
  (per-path meaning below — do not read this as "experts between ranks" in
  general), `--rank-moe-resident-fraction` (GPU/host split WITHIN a rank),
  `--rank-kv-ratio` (`coupled|speed|vector` — decouples KV split from weight
  split), `--rank-auto-reserve-mib`, `--rank-gpu-memory-mib` (absolute
  per-rank MiB budget with a line-item ledger incl. lane pools).
  Read `--rank-moe-ratio` precisely: under the **#82 GGUF expert-dim shard** it
  moves whole experts and therefore the COMPUTE assignment (owner runs the
  expert, foreign ids remap to a zero pad, the TP all-reduce sums the disjoint
  partials); on every other MoE path it splits the expert INTERMEDIATE dim, so
  every rank still computes every routed expert and only the weight slice
  moves. `--rank-moe-ratio link` (#394 slice 3) solves the vector instead of
  taking it: the GPU-resident expert mass stays exactly where the base plan put
  it (VRAM-neutral) and the STREAMED remainder is apportioned by the measured
  link weights, which equalises the per-rank transfer time the group waits on.
  Refused by name when offload is off, when the link provenance is `absent`, or
  under `ep_size>1`. Resolved ONCE in the launcher — a symbolic value that
  reaches a worker is a hard error there, never a silent fall back to the base
  plan.
  **CONFIRMED ON HARDWARE 2026-08-03** (first tokens through the slice-3 path,
  `/spinning/gpu-battery-results/2026-08-03_439_confirm/RESULTS.md`, DeepSeek-
  V4-Flash UD-IQ3_XXS TP=3 on 5090 + 2x 3080, 900 tok x 3 x 1 warmup): the
  clock moved off the x4 card (tp1 H2D 1157.6 → 672.7 GiB), transfer term
  192.7 → 128.8 s = **1.496x** against a re-derived prediction of 1.411x, and
  **-7.67 %** end-to-end ms/token against a same-window A-vs-A floor of
  CV 2.12 % / spread 4.09 %. All four gates PASS; the DESK-WRITTEN label is
  lifted for this path and the three 2026-08-02 defects are confirmed fixed on
  hardware. Still owed: ONE re-proof in a green corridor — every arm of that
  window ran with both 3080s at 211-251 MiB free against the 400 MiB floor, so
  the number is real but is not acceptance-evidence (`ARM3_COMPUTE.md`,
  "Green-corridor window", BOOT-PENDING).
  `link-calibrated` (per-rank cold-traffic coefficients from a prior boot's
  #390 dump, `SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS`, required and read ONLY
  under this symbol) is EXPERIMENTAL and **FALSIFIED** by the same window: the
  coefficient treats the cache hit rate as a property of the RANK and it
  tracks the SIZE of the owned expert range instead (tp1 0.8450 → 0.9050 as its
  range shrank 72 → 58; tp2 0.8474 → 0.7814 as it grew 72 → 89), so the solve
  overloaded tp2 and reached only 1.439x / -0.94 %, inside the floor. Plain
  `link` REFUSES while the coefficient variable is set rather than silently
  running the falsified solve — before #458 that env alone selected it.
  Registered in `planner/rejected.py` (`moe_link_calibrated_coefficients`).
- **Uneven DCP** (`dcp_size` + token vector): token/KV sharding across ranks,
  weighted owner rule, SWA-hybrid support, TP>kv_heads via replication+token
  shard. **Draft-KV-DCP**: draft KV token-sharded (−67 % draft KV; above
  TP>kv_heads, replicated is the DEGRADED layout — and the rule is two-sided:
  at TP <= kv_heads `replicated` is right and `dcp` costs 10-16 % accept,
  `planner/rejected.py::draft_kv_dcp_below_kv_threshold`). LSE log base
  follows the attention backend (FlashMLA = natural log).
- **THE ATTENTION/KV FAMILY HAS TWO DISTRIBUTION AXES (#492).** Read this
  before ever calling that family "pinned". (a) HEAD-partitioning, on the
  kv-head grid (`attn_units`), which on a checkpoint whose kv-head count does
  not divide across the ranks can be degenerate — Qwen3.6-27B at tp=3 admits
  only `[2,1,1]`. (b) REPLICATION + TOKEN-SHARDING: **kv heads are
  cloneable**. `uneven_dcp_kv_replicated` is gated on `dcp_size > 1 AND a base
  plan installed` — **not** on kv-heads vs ranks — so on every uneven-TP boot
  each rank already stores the FULL replicated kv-heads and only its own token
  shard (`_pool_kv_head_num`), and runs the attention core over the
  all-gathered head set (`cp_all_gather_heads_uneven`). The core's per-rank
  mass therefore follows the TOKEN vector, which is continuous (the owner rule
  takes any positive integer per rank — no grid, no >= 1-unit floor), and
  replicating the compute role costs NO extra KV bytes because the bytes are
  token-proportional already. A coarse kv-head grid therefore NEVER pins the
  family. Separately: PROJECTION-weight replication (`attn_kv_replicated`) is
  strictly `kv < tp`, the `<=` flip is measured-rejected, and at 24q/4kv over
  3 ranks it is structurally unrepresentable (`units % groups != 0` and
  `groups >= n` in the #116 alignment repair) — generalizing it is an unbuilt
  #169-family posten, named in `NOTE_492_attention_replication_axis.md` §2.2.
- **TPxPPxTP**: pipeline across rigs with per-stage TP groups. Slices 1+2
  merged (cross-rig PP=2 over 40G, full decode graphs on both stages incl.
  sm75). Slice 3 merged and cross-rig pp=2 validated: world-MIN
  `max_total_num_tokens` before the reduce, `--pp-stage-ratio`
  (score-proportional, snaps to full-attention boundaries), stage-local mamba
  slots, `auto` under PP with an agreement gate, `SGLANG_PP_SHAPE_CACHE` cuts
  boundary-send by −9.8/−9.2 % at bs=1 (0-1 % floor otherwise) — note the
  in-server counter reads 249 µs, which is not the standalone wire-transfer
  figure.
- **TP5+ emulation** via NCCL multi-rank co-location (several ranks per card).

## 2. Planner / solver
Key solver: water-filling over an affine cost model, pair-matrix collective
term, roles/nesting as box bounds, Pareto+knee, admissibility gates,
`coresident_budgets()`. Measured phase optima on the reference rig (1x RTX
5090 + 2x RTX 3080, Qwen3.6-27B-**FP8**, ctx 32768; RIG EXAMPLE, not a
portable default): prefill 10,1,1 (+ decoupled KV 2,11,10 keeps capacity),
decode ~3,2,2 — solve your own via
`--rank-perf-tune phase-prefill|phase-decode` and read the `CHOSEN` vector off
your boot's log. The FP8 qualifier is load-bearing: the same rig's INT8-W8A8
checkpoint has no prefill lever at all (#475 below). Under `--rank-perf-tune phase-*` the
solve now also OWNS the coupled KV token vector (#435): the chosen candidate's
matched `predict_capacity` vector is seeded into the boot instead of the
VRAM-budget split, so the pool the runtime sizes is the one the admissibility
gate accepted (#433 measured the gap: 125 504 vs a predicted 358 693 tokens).
An explicit `--rank-kv-ratio` still wins; the hand-paired
`--rank-mlp-ratio X + --rank-kv-ratio Y` of #354/#424 is no longer needed.
**The fundability gate prices the vector the boot runs (#437).** A FIXED KV
token vector keeps the relative base-plan pricing; a MATCHED one
(`--rank-kv-ratio capacity|speed`, and the phase arms since #435) has no
unused capacity to price, so every rank is checked ABSOLUTELY against the
derived reserve demand on ALL cards. Before #437 `capacity` mode accepted
16,1,1 at reserve 3000,2700,2700 -- #264's OOM config -- because it compared
a matched residual against an identical matched base; the capacity-directed
objective did not consult the gate at all. #330's 400 MiB corridor is priced
alongside the demand and REPORTED (`CORRIDOR-TIGHT`), never binding
(`SGLANG_PLANNER_CORRIDOR_MIB` overrides it; the number itself lives once in
`registry/ledger.py`).
**The prefill lockstep max is taken PER BARRIER (#475).** A prefill step is
two all-reduces per layer, not one barrier at the end, so the cost model's
compute term is `sum_family max_rank`, not `max_rank sum_family`; the Jensen
gap between them is `PerfCostModel.prefill_barrier_skew`, reported per
candidate in the plan log as a share of the base step. It is zero exactly when
one rank paces every family and large exactly when the phase-prefill arm does
its job — moving MLP onto the rank that is NOT the attention pacer. Measured
anchor: INT8-W8A8 `8,1,1` predicted 27.6 ms of extra lockstep per 1000 prompt
tokens with no fitted parameter against 27.9 ms of measured collective growth
(`#424` vs `#433` CollectiveClock windows). Consequence for reading a plan
log: **the prefill lever is a property of the checkpoint's GEMM lane spread,
not of the rig alone.** On FP8 the 3080s go through Marlin at ~1/10 the 5090's
native rate, pace every barrier, skew is 0, and 10,1,1 measured +15.2 / +18.0 %
of prefill window; on INT8-W8A8 both 3080s run the native int8 lane (3.7:1),
the whole concentration ladder is +3.0 to +4.6 % — inside the boots' own
3.0-3.5 % A-vs-A prefill floor — and INT8 has no prefill lever on this rig.
Details, backtest and the GPU confirmation ticket:
`NOTE_475_phase_prefill_prediction.md`.

**THE MATRIX DOCTRINE — rows are FAMILIES, columns are PHASES (#485), and a
row can have more than one AXIS (#492).** A family's optimum is not searched
until every axis it distributes on has been searched: finding one axis empty
and reporting the FAMILY as pinned is the #485 error. A
layout is not one vector. It is a table: every weight family (MLP / routed
experts, attention projections, GDN or mamba state, KV cache, vocab head,
vision tower, ...) has its own optimum in every phase or regime (prefill-class,
decode-class, and whatever else a workload adds), because the cards differ on
every resource axis at once. "One layout serves both phases" is a red flag,
never a default; where it appears measured, suspect an incomplete family cut
or the instrument floor first, and single-family/single-axis arms are
DIAGNOSTIC — never phrase their result as a phase-level verdict. The per-barrier
max of #475 is what makes the prefill column TRACTABLE: because the round is
`sum_family max_rank`, the prefill objective is SEPARABLE over families, so
each family's optimum is its own lane's rate-proportional split and nothing
else — and compensating one family's imbalance with another family's vector,
which is all a single-family solve can do, is precisely what manufactures the
barrier skew. At perfect per-family balance the skew is 0 and the lockstep time
is minimal at the same time. **Slice 1 delivers the prefill column's joint cut**
(`--rank-perf-tune phase-*` solves `(mlp_vector, attn_vector)` PAIRS, reports
per-candidate family pacers, and prints a `JOINT PREFILL LAYOUT` launch line):
the attention/GDN family is cut on its own #324 lane and its own grids, the GDN
state pool and the coupled KV vector follow it, and every candidate keeps >= 1
unit on the kv-head and GDN k-head grids (#62/#116). On the reference rig the
attention HEAD AXIS is grid-pinned (4 kv heads, 3 ranks -> only `[2,1,1]` is
representable), so on that axis the lever is the 16-unit GDN grid:
desk-predicted +1.0 points over the MLP-only cut on INT8-W8A8 and +6.9 on FP8,
both bracketed (see below). **#492 CORRECTION — that is NOT "the family is
pinned", and slice 1 said so wrongly (user-caught, 2026-08-03).** The family's
second axis (replication + token-sharding, §1) is continuous and live today;
the solve now prices it at both ends of a CORE-FREE / CORE-PACED bracket,
prints the geometric core/projection crossover depth (8,533 attended tokens on
Qwen3.6-27B — pure geometry, no fitted constant), and applies the same context
floor the main solve does. The corrected verdict on THIS rig: the axis is
blocked by **capacity, not the grid** — every token candidate is refused at
`--rank-perf-loose-ctx-percent 0` (the weighted owner rule funds
`min_r(P_r/v_r)` blocks, so concentrating the token vector discards the slow
cards' pools), and it only becomes reachable at loose 80-95 for +0.1 to +1.9
points against a 4-17x context loss. `NOTE_492_attention_replication_axis.md`.
The solve REPORTS the pair and does not install it — the only
runtime actuator for an attention vector is `--rank-tp-ratio`, since "mlp" is
the sole named family plan. Where the flash/scan core's per-token mass would be
needed the model BRACKETS instead of estimating: the same solve is run at the
pure-GEMM and the measured-#231-GEMV lane extremes and the plan log states
`LANE-INVARIANT` or `LANE-SENSITIVE`. ANALYSE_299's "attention lever = 0.01 %"
does NOT transfer — it was computed under the pre-#475 model, in which aligning
two families' pacers is worth zero by construction. DESK/PREDICTED; the GPU arm
is `TICKET_485_int8_joint_arm.md`, details in
`NOTE_485_joint_phase_vectors.md`.

`--objective energy` end to end with refusal over silent substitution. `planner/rejected.py` = machine-readable
register of discarded approaches — check it before re-proposing anything.

**#363 slice 1 — worth-it autocheck + layout-pair overlap + rung ledger**
(`planner/regime_switch.py`, `--regime-phase-table`, `PlanResult.regime`,
`solver_api.regime_switch_payload`). Given a per-phase layout table for one
(format, model, rig) triple, the planner returns a named verdict —
`NO_SWITCH` / `SWITCH_KV_ONLY` / `SWITCH_FULL` / `UNPRICEABLE` — with the
reason and every number it used; an absent cell yields UNPRICEABLE naming the
missing arm, never a guess. Also: per-rank shard-range overlap of a layout
pair with the dual-residency bytes (reported against BOTH baselines, since
`DESIGN_363` §20.3's "zero extra" and its ledger cost are different
quantities — 46 vs 317 units on the real 27B geometry), a pair-solving mode
that prefers maximal overlap among near-optimal candidates within a stated
tolerance (default 2.0 %, below the 4.2 % measured A-vs-A floor so it can only
break ties), and the §20.3 RUNG 0/1/2 feasibility arithmetic against the
plan's own capacity report. **DECISION LAYER ONLY — nothing in this build
executes a layout switch**: no pointer flip, no diff spill, no pre-capture
(#363 slices 2+, `ROADMAP_456` WAVE 4, gated on #286). Switch-cost constants
are the §20.2 physics estimate and the #102 graph-state analogy; only the KV
delta inherits a measurement (#297). 65 hermetic tests, five executed can-fail
arms (`test_regime_switch_363.py`); anything that decides at runtime whether a
path is worth its cost registers here rather than adding a flag (§17).

**Generality (#434 slice 1).** Plain `--rank-tp-ratio auto` is the documented
CAPACITY-FIRST default (byte-proportional to the VRAM budgets, no probe); it
now names the per-task optimizer and the flag that engages it in the CLI help
and in one boot log line, and calls out a hand-pinned `--rank-mlp-ratio` as the
solution of some earlier operating point. `--rank-perf-tune dec` no longer
returns the base split on the strength of M22's reference-rig "decode is flat"
finding: it SOLVES the bs=1 decode round time from the rig's own effective
bandwidth, and reports flatness as a result when that is what the profile says.
Every objective therefore solves from per-(rank, family) profile scores.
Constant audit: `docs/dev/AUDIT_434_planner_constants.md` (62 classified;
19 RIG-FITTED, 16 named follow-ups FU-434-1..16). The cost model now prints
which calibration scalars are BORROWED from the development rig rather than
only which were overridden. Standing hermetic proof suites on synthetic
foreign rigs: `test/registered/unit/planner/test_planner_generality_434.py`
(profile-follows, symmetry-has-no-lever, relabeling/scale/name invariance,
AST leak guard) and `test_borrowed_calibration_434.py` (a measurement may only
be applied to hardware it matches). Probe-first bootstrap on unknown hardware
is designed, not built: `docs/dev/DESIGN_434_probe_first_bootstrap.md`.

## 3. Memory tiers / offload / spill
- **KV-pool token-slot ledger** (`DESIGN_330_vram_dial.md` §3b, #486): every
  standing holder of `C_target` slots is a NAMED posten — committed KV, the
  per-decode reserve (`bs x get_alloc_reserve_per_decode()`, held under spec
  AND plain decode), radix inventory. Nothing that permanently occupies pool
  slots may stay an unnamed transient; that is how the spec reserve went
  uncounted until #486.
- **Expert offload**: MoE experts in a pinned host-RAM pool, streamed over
  PCIe on demand. **CUDA-graph-compatible path EXISTS** (decode-graph +
  eager-prefill hybrid): GPU kernels read the pinned pool via UVA zero-copy
  indexed by device-resident router ids — no host sync. **#443 ported the
  DeepSeek-V4 GGUF path onto it** (it was the last `tolist()`-syncing caller,
  and the ranked-#2 cause of the 2.6x decode gap in BENCH_394). The port added
  no mechanism: the capture-admission bound was tightened from `tokens x top_k`
  to `min(tokens x top_k, cold set)` — under the #82 expert-dim shard the loose
  half refused captures that are provably safe, because `forward_impl` has
  already collapsed foreign ids onto the resident zero-pad expert — and the
  #394 cold-tier seam grew a device breach counter read at the replay boundary
  (`moe/offload_capture_gate.py`, the #431 pattern) so its development switch
  is no longer a switch into an out-of-bounds gather. Desk-proven: no host read
  survives the ported step (interception over
  `item/tolist/cpu/numpy/nonzero/__bool__/__int__/__float__/__index__`), and
  the captured gather lands byte-identical scratch rows vs the eager `_fetch`
  (`tests/moe_offload/test_capture_desync_port.py`, 63 tests, four executed
  can-fail arms). **REFUTED-AT-BOOT (#452)**: the B1-B4 window of
  `scripts/dev/443_graph_proof/` ran 2026-08-02 (evidence
  `/spinning/gpu-battery-results/2026-08-02_desync_graph_proof/RESULTS.md`).
  **B1 PASS** — capture succeeds, 13 of 18 decode batches captured.
  **B2 FAIL** — the graph arm and the eager arm decode different text from the
  same greedy prompt (each arm internally deterministic over 3 runs, so
  systematic); the cause is NOT localised and the window lacked its control arm
  (graphs-vs-eager without the offload), so "the gather moved wrong rows" stays
  an unproven hypothesis — both arms' texts are fluent and on-topic, which a
  wrong-expert gather would not be.
  **B4 6.60x REGRESSION** — 984.4 ms/token captured vs 149.1 ms/token eager,
  localised to the captured decode step (prefill +1.5 %). B4 is STRUCTURAL: a
  graph cannot vary its work with the data, so the captured gather moves the
  worst-case scratch set every layer every step (2.128 GiB/token measured from
  the run's own expert-stats) where the eager fetch moves only the missed
  experts (0.366–0.535 GiB/token) — a 5.35x PCIe multiplier on a PCIe-bound
  step. **The gate is restored**: `SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1` now refuses
  by name at boot (`moe/offload_capture_gate.resolve_graph_mode`,
  `tests/moe_offload/test_capture_regate_452.py`, 15 tests, executed can-fail);
  `SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE=1` re-opens it for a card window. The
  mechanism stays in-tree behind the refusal so a candidate fix can be measured
  against these numbers. Verdict, repricing and what a real fix would require:
  `docs/dev/NOTE_452_desync_boot_refutation.md`.
  **#462 BUILT THE SURVIVING ROUTE** (`layers/moe/breakable_offload.py`,
  `SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable`, **OFF by default**): NOTE_452 §3's
  Option 3, "capture the compute, keep the fetch eager". The eager phase fetches
  the routed experts into the fixed slot arena and publishes the slot vector
  through a static bridge buffer BEFORE replay; the captured segment addresses
  SLOTS only. The key finding is that most of it already existed — `install()`
  builds the `[R+C]` arena and binds it into the layer's parameters, so a graph
  captured over that layer already addresses slots; what was missing was the
  bridge, a host-side remap, and the `eager_on_graph` break. Volume returns to
  the eager 0.366–0.535 GiB/token because the fetch never enters the graph.
  Cost, and it is a COUNT not a measurement: 1 D2H rendezvous + 1 pinned
  blocking copy per MoE layer per step, against the eager path's 1 rendezvous +
  2 PAGEABLE (hence host-blocking) `_build_lut` copies — 43 layers, 129
  host-blocking crossings/step → 86. The 43 rendezvous are IRREDUCIBLE: MoE
  routing is sequential across layers, so no point in a step has several
  layers' decisions available to batch, and removing them means the refuted
  in-graph fetch. Refuses by name at boot unless decode backend is `breakable`
  (`eager_on_graph` is a no-op otherwise → host reads inside a real capture) and
  prefill is eager (a prefill chunk overflows the arena and a captured segment
  cannot wave-split). Both spellings of the refuted path still refuse.
  **DESK-WRITTEN, NEVER EXECUTED — no boot, no replay, no ms/verify figure
  exists, and F1's 5.3–8.4x is a Qwen3.6-35B-A3B ceiling that is NOT a DSV4F
  number.** F2 (per-layer break cost, decomposed) is the first measurement of
  the next window and gates default-on. Tests:
  `tests/moe_offload/test_breakable_route_462.py`, 37 hermetic, SEVEN executed
  can-fail arms (pad marker dropped → 3 red; bridges aliased across capture
  shapes → 2; #286 capture gate removed → 1; overflow check moved after resolve
  → 2; prefill precondition dropped → 1; breakable silently downgraded to eager
  → 1; staged publish flipped to non_blocking → 1). The open finding it pinned
  — the `experts` descriptor's `va_stable_required=False` is FALSE under this
  route, since a captured graph holds the arena's addresses — is RESOLVED in
  the register by #468 (see the #286 entry above); nothing in
  `breakable_offload.py` changed, and no producer declares the reference until
  this route boots.
  Design: `docs/dev/DESIGN_462_breakable_route.md`; GPU ticket:
  `docs/dev/TICKET_462_f2_and_replay.md`.
  **The shipped offload path is the eager one** (`--disable-cuda-graph`,
  149.1 ms/token), unchanged and unaffected. Sizing note from that port: a
  captured decode
  bucket needs `bs x top_k` scratch slots, so the recipe's own `bs=1` operating
  point fits the battery's existing `SGLANG_MOE_SCRATCH_SLOTS=6` at no extra
  VRAM. Double-buffered prefetch with compute overlap; expert-major prefill
  waves (`SGLANG_MOE_OFFLOAD_WAVE_ORDER`, byte-identical proven); fp8 presplit;
  load-time-aware halves for fp8/GPTQ/AWQ (GGUF-MoE half missing — guarded).
- **#394 cold-shard chain** (slices 1+2 merged): measured H2D provenance chain
  (env > card-probe > nvml-negotiated > refusal; `absent` unselectable),
  `cold_tier_shm.py` shared-DRAM segments (UUID/BDF identity, manifest read
  lazily after load, header sealed last, PROT_READ views with kernel-enforced
  write protection). **Slice 2 wires the fetch path** (`cold_tier_fetch.py`):
  a rank-uniform owner map derived from the same `partition_cold_experts` the
  staging plan uses (plan `digest()` pins the uniformity), the cold pool
  ALLOCATED IN the segment rather than copied into it, and
  `MoEExpertOffloadCache._fetch` sourcing a delegated expert from the owner's
  `PROT_READ` view over this rank's own link. Behind
  `SGLANG_MOE_COLD_TIER_SHM=1`; with it off the slice-1 boot refusal for
  delegation on disjoint expert shards is unchanged, field for field.
  **Honest scope of slice 2**: byte ownership moves, COMPUTE does not, so
  per-rank H2D is predicted unchanged.
  **Slice 3 (#439) moves the compute assignment** and is where ANALYSE_393's
  Path A′ lives. It needed no new mechanism: the #82 expert range IS the "moe"
  family vector, so the slice is a SOLVE plus its wiring
  (`layers/moe/expert_compute_placement.py`, `--rank-moe-ratio link`, see §1).
  MEASURED on the reference recipe 2026-08-03 (base plan `30407,19080,19080`,
  solved vector `160,79,119`): clock rank 192.7 s → 128.8 s = **1.496x**, ahead
  of the 1.411x its own model predicted, and -7.67 % end-to-end. The calibrated
  variant reached 1.439x and is falsified (§1). Two findings the window added:
  `--rank-auto-reserve-mib auto` is INFEASIBLE on this recipe (it derives
  3968 MiB per card from the activation heuristic, leaving a 16512 MiB budget
  against 17.59 GiB of weights + runtime — the refusal now names the derivation
  and the pinned value that fits, `ServerArgs.derived_reserve_infeasible_note`),
  and the recipe is CORRIDOR-RED at `2200,1400,1400`; the repaired reserve is
  `2200,1800,1800` (base plan `30407,18680,18680`, vector `213,104,157`).
  BOOT-PENDING: the green-corridor re-proof of the 1.496x point
  (`ARM3_COMPUTE.md`, two boots), the eager arms 1+2 of
  `scripts/dev/394_s2_proof/`, and
  the graph seam, which refuses by name for TWO reasons (#443 named the nearer
  one): the captured gather sources only this rank's pinned pool, so a routed
  DELEGATED expert has no row — the eager path covers it with a host read of a
  Python set, which a capture cannot contain — and the UVA pointer for a
  `cudaHostRegister`'d peer mapping is unverified on hardware. Past
  `SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE=1` the first is no longer undefined
  behaviour: the remap clamps and counts on device, and
  `offload_capture_gate` raises by name at the replay boundary. Since #452 that
  seam is behind a SECOND refusal — the capturable decode path itself refuses at
  boot — so reaching it now takes both overrides. Graphs incl.
  CPU-MoE remain IMPLEMENTATION EFFORT, not blocked: UVA reads, cudaGraph host
  nodes, CUDA>=12.4 conditional nodes, and graphs pin ADDRESSES not CONTENTS
  (spill/restore under fixed buffers is legal). #452 adds the measured caveat
  that "capturable" and "offload" pull against each other on a PCIe-bound step:
  the graph must move the worst case, the offload's whole economy is moving only
  the miss. Conditional nodes are the mechanism that would reconcile them, and
  torch exposes none.
- **#302a dynamic expert heat migration** (`layers/moe/expert_heat_migration.py`,
  `SGLANG_MOE_HEAT_MIGRATION=1`, OFF by default): keeps re-ranking the resident
  expert set against a decayed window of live router traffic, instead of the
  load-time one-shot choice (or Stage-1 `SGLANG_MOE_HOT_RESIDENCY`'s one-shot
  freeze). Swaps are EQUAL-COUNT pairs — hot in, cold out — so residency size,
  and every VRAM figure derived from it, is invariant by construction; victims
  are coldest-first per `DESIGN_407` §8's ladder; the #82 pad expert is never
  demoted and a #394-delegated expert is never promoted. Two-sided hysteresis
  (relative margin plus an absolute `min_gain` floor, because a purely relative
  margin churns on sampling noise in the tail of the routing distribution).
  Eager path only: refused by name under `SGLANG_MOE_OFFLOAD_CUDA_GRAPH` and
  after `install_capturable_buffers()`, since a captured gather's LUTs pin the
  layout. Counters land in the #390 dump under `heat_migration` / `heat_*`
  totals.
  **DESK-PROVEN, BOOT-PENDING — it has never served a token.** The desk
  falsifier (`scripts/dev/302a_heat_desk/`, run over four independent boots'
  recorded `expert_stats_*.json`) reproduces the recorded static hit rates
  0.7623 / 0.8427 / 0.8463 exactly (delta 0.0000), puts the ORACLE ceiling at
  the same resident-set size at 0.9836 / 0.9844 / 0.9850, and shows a ranking
  learned on a DIFFERENT boot on a different day still capturing 40-83 % of that
  ceiling (+1.24 to +18.26 pp). Hit rate is a necessary condition for the H2D
  reduction, not a decode measurement: no tok/s claim exists until the A/B in
  `scripts/dev/302a_heat_desk/AB_SPEC.md` runs. Tests:
  `tests/moe_offload/test_heat_migration_302a.py`, 33 hermetic, two executed
  can-fail arms (policy neutered -> 12 fail; executor's D2H dropped -> 3 fail),
  output bit-identity across a migration pinned end-to-end through `run_waves`.
  Sub-cell **#302-lookahead is REFUTED at the aggregate grain**: adjacent-layer
  heat correlation is indistinguishable from zero (|mean Spearman| <= 0.03,
  top-R overlap on the chance line), which also settles that one shared ranking
  cannot serve several layers.
- **HiCache** L1-L3 prefix cache (validated with uneven DCP/TP; storage key
  includes kv-dtype; runtime attach/detach works on UnifiedRadixCache). The
  L2 host tier's `page_first_direct` transfer path was blocked on this rig by
  a segfault in `transfer_kv_all_layer_direct_lf_pf` (#436, cu12/cu13
  `cudaMemcpyBatchAsync` ABI split); unblocked by the cu13 `sgl_kernel`
  rebuild.
- **KV session offload (kvso)**: FCFS spill of youngest sessions to RAM (KV
  only, GDN stays resident), budgets (volume/rate/window, demote to HiCache),
  idle-first victim choice, decoupled from speculation.
- **Hibernate to disk** (weights+KV survive process exit; uneven-TP3 reload
  50s→8-14s) + suspend-to-RAM (memory saver; reaches the legacy hybrid-SWA
  `SWAKVPool` since upstream #32213 — before that it was silently a no-op
  there, while `UnifiedSWAKVPool` already honoured it).
  **#456 writes the image SPARSE by default** (`model_loader/sparse_write.py`,
  `SGLANG_HIBERNATE_DENSE_WRITE=1` to opt out): all-zero 4 KiB pages are
  `lseek`-ed over instead of written. This is the one mechanism that survived
  the #306 codec refutation — 12.64 % of a real rank image is zero pages,
  parked pre-allocated buffers. The format does NOT move: sparseness sits under
  the `torch.save` container, holes read back as zeros, `HIBERNATE_VERSION`
  stays 2, and there is no reader change. **Re-measured honestly, and the #306
  projection does not survive contact with this box's filesystem**
  (`DESIGN_456_sparse_image_write.md` §4, 3 GiB synthetic at the measured hole
  fraction, rotated arms, drain-before-timing, 11 reps): on `/spinning` ZFS the
  allocated bytes are **identical** dense vs sparse (2 816 098 816 both ways —
  ZFS compression had already taken the same 12.64 %, so the byte win here is
  ZERO). On tmpfs, which folds nothing, the allocation win is exactly the
  projected **1.1447x**. **No write-time win exists on either**: the point
  estimates are 0.897 / 0.855, both negative and both at or inside their A-vs-A
  floor (10.3 % / 16.9 %), and the detector is a second full pass over the image
  measured at 67 ms/GiB (≈0.45 s per 6.68 GiB rank) — not the "zero CPU" the
  ticket assumed. The ext4/xfs-on-a-real-device arm, where the win should be
  there to take, is UNMEASURED: this box has no such filesystem. It ships
  default-on because it is byte-identical (sha256 gate dense-vs-sparse) and the
  1.1447x is real on any non-folding target; on a compressing `hibernate_dir`
  the escape env is the better setting. BOOT-PENDING: a real park/restore round
  trip, recipe in DESIGN_456 §7. Second consumer identified but NOT wired:
  `hicache_migrate.execute_plan` (#297) does not share this writer.
- **Runtime VRAM dial** per card (VMM page return), **KV pressure ladder**
  (geometry stages instead of rejects; explicit ladders work; rung-dependency
  refusals exist and fire). `--kv-pressure-ladder auto` mode wired via
  rig-profile bridge (#428), boot validation pending — the table is
  computed from the rig profile by the #272 planner, rank-uniformly and
  UUID-keyed, and inventories only rungs whose actuator this configuration
  wires. Capacities are labelled placeholders until the measured figures
  arrive. BOOT-PENDING: `scripts/dev/428_boot_checks/`. **KV resharding**
  at phase boundaries (delta move <1 s, `kv_reshard_vectors`), **GDN slot
  ladder** (resident-state cap + idle vacate → VRAM back to KV pool).
  `--lane-offload-profile/-class-policy/-park-targets` are wired at runner
  init once-per-process (#428), boot validation pending; a typo now refuses
  there too. The park chain reaches the register and the movement layer's
  default reads it — but nothing in
  production constructs the movement backend yet, so the chain has a consumer
  PATH, not a consumer. Whole surface is behind `SGLANG_OFFLOAD_REGISTER=1`
  (dark launch). BOOT-PENDING: `scripts/dev/428_boot_checks/`.
- **#286 short-term offload register, asset-class layer**
  (`model_executor/short_term_offload_register.py`, desk slice): the half the
  existing `offload_register.py` did not have. (a) One
  `AssetClassDescriptor` per `OFFLOAD_CLASSES` member — ladder rank, memtier
  payload class, VA-stability, dimension presets, and the reason for each —
  with an IMPORT-TIME guard that a new class without a descriptor cannot exist.
  (b) `DESIGN_407` §8's global importance ladder as executable order
  (`LadderRank`, `plan_spill`): rank-ascending, coldest-first within a rank,
  partial spill (the walk stops at the first item covering the shortfall, so a
  class is never emptied for a request one item covers), and rank 5 (active
  work) never planned — that stays the scheduler's #273 FCFS call. Before this
  the ladder existed only as prose, so every consumer was free to invent a
  local victim list. (c) Graph-state FAMILIES as a wired asset class
  (`GraphFamilyRegister`) over the existing #102/#93 machinery in
  `speculative/adaptive_graph_memory.py` — this is what `DESIGN_363` §20.3
  RUNG 1 calls (`rung1_evict`), plan-only by default so the controller can
  price a rung before committing. (d) A NAMED refusal under active capture
  (`OffloadUnderCaptureRefused`, probing `get_is_capture_mode()`): a park
  unmaps pages, which is eager work, and #452 settled that eager work belongs
  BETWEEN replays. 55 hermetic tests, EIGHT executed can-fail arms (ladder key
  neutered → 3 red; capture gate removed → 4 red; provenance flags flipped
  → 1 red, and 2 red with the module's own defensive absent-check also removed;
  partial-spill `break` dropped → 1; origin exclusion dropped → 8; rank-5 guard
  dropped → 1; rung-1 class list widened → 1; derived move time relabelled
  MEASURED → 1). **DESK-ONLY — no page has ever moved.** The production mover
  (`AdaptiveGraphStateMover`) is desk-written and never executed; BOOT-PENDING
  list and the GPU ticket are in `docs/dev/DESIGN_286_short_term_register.md`
  §7. It states NO reload-latency figure: `DESIGN_363` §20.3's ~25 ms is a
  projection that #102's own measured 40-51 ms avg / 85 ms max per state swap
  already contradicts, and the derived link time this module computes is
  labelled a FLOOR (ESTIMATE) even off a MEASURED bandwidth.
  **Both open findings are now settled** (#461/#468, §8/§8b of the design note,
  69 hermetic tests, SEVEN further executed can-fail arms). (e) `gdn_state_sets`
  is classified per CONTENT STATE, not statically: LIVE it is `DEVICE_BOUND`
  (DESIGN_407 X2) and has no park target at all — and is not even a page range,
  since one set is a stride slice of the pool's `[layers, slots, ...]` tensors —
  while the EXPORTED blob (`MambaPool.export_state_blob`, name-keyed, restorable
  into any free slot, already carried to a #224 tier by #364) is an ordinary
  `EXPENSIVE_RECONSTRUCTABLE` byte payload. A park of this class is therefore
  always VACATE-then-move; `ContentState` is the axis, `SpillStep` marks the
  step, and the #407 doctrine text now says the class describes content in a
  state — the law that nothing device-bound travels is unchanged. (f) `experts`
  VA stability is ROUTE-acquired: under #462's breakable route a captured graph
  holds the slot arena's addresses, so a graph family declares
  `addresses_classes` and the register refuses the park by name
  (`ground=GROUND_GRAPH_ADDRESSED`) at the same gate that refuses under an
  active capture — one rule, two grounds, and `plan_spill` will not plan what
  the gate would refuse. Note a #93 family PARK preserves the VAs and so does
  NOT release the reference; the family must be unregistered.
- **memtier registry**: tier ids with volatility + payload class and
  provenance `measured|estimate|absent` (absent refuses use). HONEST STATE
  (audit #421, updated by #286): the FIRST production consumer is wired —
  `model_executor/short_term_offload_register.price_park_target` picks its park
  target through `TierRegistry.select` and refuses a tier whose bandwidth is
  ABSENT (and, by default, one that is only an ESTIMATE), so "an unmeasured
  path is never assumed usable" is now enforced code for the classes that
  module owns. It is ONE consumer, not the reconciliation: expert offload, the
  #394 cold tier and the rest of the #286 park-target ladder still carry their
  own target lists. The #421 F6 pin is retired accordingly and replaced by the
  positive `MemTierIsNowWiredTest`, which pins the module-scope import and
  pins that a priced target is a `TierId` and not one of the three
  hand-written `PARK_TARGETS` strings.
  Slice 1b (#407 / directive #434) made it hardware-general:
  `TierRegistry.for_machine()` fingerprints the box from NVML UUIDs (#397
  canon), applies a stored profile ONLY at the scope its hardware match
  licenses (`EXACT` = every tier; `MODEL` = card templates only, no host /
  filesystem / remote row), and otherwise bootstraps from live facts with
  measured sizes and every cost ABSENT naming its probe. `from_profile()` no
  longer defaults to the bundled rig profile — that default handed one
  development box's host RAM, ZFS pool and 40G peer to every machine.
  Measurements are ingested from the EXISTING artifacts (`card_probe` #213,
  rig artifact #271, `capability_matrix` #278) by `memtier/adapters.py`;
  #407 adds no probe of its own. `TierTransport.link_path` +
  `link_disjointness()` expose PATH identity for #423's striping gate, with
  `DISJOINT` requiring complete paths on both sides and `UNKNOWN` being a
  refusal. Design: `docs/dev/DESIGN_407_memtier_registry.md`.

## 4. Speculative decoding
NEXTN/MTP standard (steps 3, topk 1, draft 4); adaptive draft length (upstream
base, fork adds graph-offload, high-accept ladder, frozen-MTP, hetero
determinism); acceptance-driven DFLASH<->NEXTN switch + adaptive k; DFLASH solo
draft on the big card (vocab broadcast reclaims ~5 GB); chain-spec on the
weightless lane; multi-layer EAGLE fixes; spec-algo name validation (one
source, parse-time refusal); canonical `--speculative-draft-model-path`.
Tree-spec topk>1 under DCP is HARD-GATED (silently wrong + perf-negative — do
not re-attempt without new evidence; see rejected register).

**Per-decode KV reserve is derived, not a blanket 2x (#486).** The reserve
every running request holds ahead of `kv_committed_len` is `W + L`: the write
footprint of this step's draft+verify (`get_alloc_len_per_decode`) plus what
one in-flight verify can still commit while the host is one step behind
(`get_commit_lag_per_decode` = `max_speculative_num_draft_tokens` under
overlap, 0 with `--disable-overlap-schedule`). It is now a NAMED posten in the
pool ledger (`DESIGN_330_vram_dial.md` §3b) instead of an uncounted transient.
Honest result: on our NEXTN recipe (steps 3 / topk 1 / 4 draft) `W == L`, so
the old `2 x W` was already exactly the need and the fix saves nothing there —
upstream issue #32459's radix-collapse diagnosis does not transfer to this
shape. It tightens topk>1, page>1 trees, `steps > num_draft_tokens` chains and
all non-overlap runs, and it unifies the DFLASH solo lane's own hardcoded
`2 x block_size` onto the same derivation. NOT upstream PR #32574: that drops
the lag term entirely on the premise that `batch.seq_lens_cpu` is synchronous,
which is false in this tree (`resolve_seq_lens_cpu` runs inside `run_batch`,
after `prepare_for_decode`, and is `None` when a backend opts out of the D2H
mirror) — adopting it would under-reserve and let verify write past
`kv_allocated_len`. Both directions are pinned:
`test/registered/spec/test_alloc_reserve_need.py`.

Draft-solo placement now admits the whole DFLASH FAMILY (#470): DSPARK joins
DFLASH because it has the same shape — self-drafting block model, token-id
round output, post-all-reduce hidden-state input — and its one delta, the
confidence head's per-request block truncation, rides the SAME per-round
broadcast as one extra integer per request (`dspark_components/dspark_solo.py`
packs ids + lengths into one int64 tensor, so the round still costs one
collective). `FROZEN_KV_MTP` stays refused and is pinned by a test: its draft
reads the target KV in place, which no single rank holds. Solo DSpark v1 is
greedy-acceptance-only (a non-greedy round would need `[bs, gamma, vocab]`
corrected logits on every verifying rank) and switches the default-on
`SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD` off, with reasons logged rather than
inferred. `--speculative-moe-runner-backend` (the existing per-draft flag) now
actually reaches DFLASH/DSPARK draft builds, which is what puts an MXFP4
DSpark head on `Mxfp4MarlinMoEMethod` on sm120. **DESK-WRITTEN — no DSpark arm
has booted; `docs/dev/TICKET_470_dspark_boots.md` is the only evidence path,
and its Boot A prices the ~21 % rank-0 residency cut the arm costs.** Two
pre-boot blockers on any PACKED DSpark draft are cleared (#491, from the #490
upstream sweep, `NOTE_490_pr33271_abgleich.md` §C): the fused-KV-projection
support probe answers `False` for marlin/AWQ/GPTQ linears instead of raising
inside the branch whose job is to decline, and the draft's `.scale` rename is
suffix-anchored so `.scales`/`.scale_inv` are no longer mangled, dropped with
only a warning, and left as a silently zero accept rate.

## 5. Multi-group runtime (dual lane)
Slices A-D merged: lane-correct context overlays (~370 callsites), own thread +
high-priority stream, lend/reclaim in ms, SM-contention pairing rule,
lane-NEXTN head. Lane spec chain merged: rank-local draft-KV sizing, chain-spec
topk=1 on the lane, lane prefill chunking (`dual_group_lane_prefill_chunk`;
spec chunks carry the head primer — costs measured), Marlin LoRA workspace
keyed (lane,name). PD disaggregation: prefill satellite carries hybrid GDN
(KV+mamba slot via mooncake), default graph-covered.

## 6. Weightless KV lane
A card holds ONLY KV + attention (no weights): chunked prefill/extend, fp8/int4
worker KV, DCP comm fusion, graph-captured streaming decode, host-tier KV
spill, chain spec. **Live session handover without server stop** + draft
re-sharder as its own spec type: MERGED and GPU-gate-passed
(`POST /session_handover`, five-phase at session scope, hard GDN-blob gate
keyed on `BasePrefixCache.supports_mamba()`; proven byte-identical to a
never-moved reference via a real cached-tokens import, `cached_tokens=1152`
on resume, plus seven named-refusal negative controls) — the declared v1
limit stands unchanged: a booted TP>1 destination still needs the offline
manifest-scoped umsharder (`page_size == 1`, inherited from
`dcp_owner_mode`) to reshape into its geometry first, live handover does not
do that reshape in-process.

Also wired on the tip but easy to miss (audit #421): the regime-controller
gate machinery, KV-pressure rung-dependency refusals, the hibernate flag
contract (`hibernate_dir` + weights/draft CPU/disk backup flags), and a
118-name retired-env guard that refuses stale SGLANG_* variables loudly.

## 7. Collectives / transport
**barlink** (own vendor-neutral CCL): NCCL-parity device transport,
cross-vendor byte-exact, UCX transport (chunk pipelining, dual worker), tuned
all_gather ring, graph-capable direct mode. **Smallbar BAR1 direct path**:
peer VRAM over 256-MiB BARs, beats NCCL 1.13-1.34x in serving.
`--collective-net-small/-bulk` per message class with typo hard-reject.
dmabuf GPU-RDMA works on consumer cards with the stock driver. Rig facts: NO
P2P/NVLink here, negotiated PCIe x4/x8/x8 (NVML max-width reports x16
NAMEPLATE — always read negotiated width), NCCL-verbs broken on our RoCE.
**Collective-decision recorder** (`barlink_uniformity.py`, #431): per-rank
ordered log of every `(op, nbytes, path, rounds)` dispatch decision plus a
pure `first_divergence` comparator — the standing instrument for the
rank-local-condition-before-a-group-collective family (#94/#194/#312/#431).
Off by default (`SGLANG_BARLINK_RECORD_DECISIONS=1`, optional per-rank
on-disk dump via `SGLANG_BARLINK_RECORD_DUMP_DIR` for post-mortems on a
wedged run). **Scoped slow-boot warning**: barlink BAR1 × uneven weighted DCP × an
fp8-quantized checkpoint warns loudly at ModelRunner boot instead of refusing
(#438a). What #424 recorded as a wedge is a slow FIRST boot: on a cold JIT
kernel cache the first CUDA-graph capture batch spends ~190 s per rank
(184-197 s, three ranks concurrently) inside the JIT cold-build window, and
under the raw ~30 s BAR1 cap the peers' spin kernels tripped their deadline
about six times over inside it — which is the "~30-40 s per collective"
crawl. Two proofs, both 2026-08-02: capture (`.../2026-08-02_431_recheck/`,
12/12 in 4:58, READY after 6:05, 176/176 requests, no
`Bar1CollectiveAborted`) and full serving load
(`.../2026-08-02_435_coupling_fp8bar1/`, both FP8 layouts, 9× `ACHIEVED=bar1`
per arm, full probe sets, no abort/PeerLost/CollectiveTimeout in any log).
Warm boots are normal speed. Restore the old hard refusal with
`SGLANG_BARLINK_REFUSE_FP8_UNEVEN_DCP_BAR1=1`; the legacy
`SGLANG_BARLINK_ALLOW_FP8_UNEVEN_DCP_BAR1` is kept and is not a no-op — `=0`
still means "do not admit this arm" and still refuses, `=1` is still honoured
but now redundant and says so. INT8-W8A8 over BAR1 and fp8 over NCCL remain
untouched. See `docs/dev/ANALYSE_431_fp8_bar1_dcp_deadlock.md`. Still
unmeasured: a boot from a genuinely empty `extcache_docker`.
**BAR1 deadline + loud abort** (#431 fix slice): the three BAR1 kernel launch
sites go through `resolve_timeout_cycles`, so the documented 40x JIT
cold-build extension finally reaches the one transport whose kernels spin on
a device deadline (identity outside the window, so serving and the captured
graph are unchanged). A tripped spin kernel now raises
`Bar1CollectiveAborted` with rank/op/rounds instead of continuing over a
partially written buffer — checked after every host-path collective and, for
captured decode, at the CUDA-graph replay boundary
(`barlink_abort_gate.py`); never inside a stream capture, where the device
read would be illegal. Knobs:
`SGLANG_BARLINK_BAR1_ABORT_CHECK=0` (restore the old silence),
`..._CHECK_EVERY=N`, `..._CHECK_REPLAY=0`.

## 8. GGUF stack
Generalized loader (registry + family mapping tables), unsloth-UD, mixed-dtype
fused GDN qkvz, MoE tensor mapping, vision/mmproj, sibling-config validation,
DeepSeek-V2/3/4 class GGUF-safe (`.qweight` accessors, quantization_config
drop, tokenizer route). Perf: batched MMVQ, Q8 lm_head, K-quant MMVQ tuned to
Q8_0 efficiency (TP=2 beats llama.cpp), graph-replay numeric safety for ALL
quants, `gguf_mmq_decode_threshold`.
**MXFP4 (ggml type 39) is native since #398** — a complete kernel set
(dequantize, dense MMVQ, dense MMQ, MoE MMVQ, MoE MMQ, `moe_get_block_size`),
so the type is in all three GGUF type sets and the load-time MXFP4->Q5_0
repack that used to carry it (#391 blocker 1) is a no-op on such a wheel. That
is 4.25 bpw instead of 5.5, i.e. the repack's +29.4 % bytes returned. Block is
32 values in **17 bytes** (E8M0 byte + 16 split-half nibble pairs of doubled
E2M1), so it is the first GGUF block type with an ODD stride: every read of
`qs` must use the byte-granular `get_int_b1`, never the 2-byte-aligned
loaders. The scale helper `ggml_cuda_e8m0_to_fp32_half` returns 2^(e-128) —
already halved against the doubled lattice — and is bit-identical to the host
reference, so dequant is compared EXACTLY, not within a tolerance. Kernel
presence is a wheel property, probed via the `ggml_mxfp4_native` marker op
(the #73 pattern) and overridable with `SGLANG_GGUF_MXFP4_NATIVE=0`, which
hands the checkpoint back to the repack. GPU-pending:
`TICKET_398_mxfp4_validation.md`.

## 9. Quant lanes
FP8 (sm120 GEMM tuned; per-channel fused GEMV; opt-in deterministic
`SGLANG_DETERMINISTIC_FP8_GEMM`; e4m3 KV bit-exact on sm86), INT8-W8A8 (default
recommendation; sm86-native lane; beware the dual-dist wheel trap — pin by
sha256), NVFP4 (V4 class usable via dequant fallback for unpackable layers),
Marlin alignment family (EIGHT sibling bugs fixed — device-free fold predicate,
lcm=128 on coupled dims; alignment fixes must preserve cross-layer agreement).
The eighth (#444b) is MXFP8: its `weight_block_size [1, 32]` is the OCP scale
layout, not an alignment registration, so an asymmetric exposed block is now
coarsened by `lcm` of both axes before the marlin fold — latent, mxfp8 needs
capability 100.

## 10. Determinism / quality gates
Hetero-determinism roots fixed (verify sync, graph pads, flashinfer workspace,
rank-0 draft-pick broadcast, fp8 dequant pairing). GDN prefill beyond ~109
tokens is upstream-nondeterministic — byte gates only on short outputs. Canon:
no A/B without a same-boot A-vs-A floor; first boot after cache changes is a
JIT outlier; floor scales with measurement length. Determined-answer probes:
underdetermined text can only report "different", never "wrong".

## 11. Device identity (order trap)
torch order != NVML order on multi-vendor-generation rigs. The ONLY bridge is
the IdentityMap (registry/nvml.py) keyed by UUID/PCI-BDF. Never feed a CUDA
ordinal to NVML or vice versa; never key caches on the masked CUDA view.
This also covers the custom-group object exchange: it names the world gloo
cpu_group instead of letting torch pick a staging device.

## 12. Robustness canon
Rank-local condition BEFORE any group collective (hang family); bounded waits
with fixed pool universe; bounded peer-liveness instead of endless spin;
ColdBuild error unmasking (never substitute "lower mem-fraction" for a real
error); quant guards fail loudly instead of silently downgrading; JIT cache
poisoning family (stale batons, foreign-worktree kernels, cold-JIT =
capture-cost illusion); reference-twin drift family (#418 #425 #427 -- a torch
reference that disagrees with the kernel it validates, hidden by an oracle
that compares only the region where they agree; fix the reference AND widen
the comparison, and pin whether the reference is reachable from serving).
Quant-name-list family (#443 #446 -- packed-vs-dense and fused-a-proj cat_dim
decided by an `{awq, awq_marlin, moe_wna16}` enumeration instead of by the
layer: GGUF GLM-4-MoE blocks now construct instead of raising on
`.weight.dtype`, glm4_moe_lite can reach its fp8 shared-expert path at all,
GPTQ-family MLA fuses on its real output axis, and GPTQ `desc_act=True` is
refused by name rather than fused wrong -- see docs/dev/NOTE_446_gptq_cat_dim.md).
Pad-slot family (#444a #444e -- graph-padded verify batches carry
`PAD_SLOT_ID=-1` rows: #444a made the GDN TARGET_VERIFY conv run on a
request-private window unconditionally, but its `index_select`
(`gdn_backend.py:417`) crashed with a device-side assert on those rows
whenever a decode batch was graph-padded under GDN+spec; triton kernels
already skip PAD_SLOT_ID natively, so any torch-level indexing on
cache_indices must skip/mask it explicitly too -- fixed by #444e).
Third member, #472 (upstream sgl-project/sglang#33253): under the breakable /
piecewise graph the attention wrapper narrowed `out_cache_loc` but not
`positions`, and the EVEN DCP owner rule reads `positions`. The damage was not
the padded rows themselves (their ZERO `out_cache_loc` lands in the pool's
reserved slot) but the resulting LENGTH DISAGREEMENT: the upstream
`positions.numel() == loc.numel()` guard rejected the mask, the fallback
`forward_batch.dcp_kv_mask` is HIP-only, and a `None` mask degrades
`set_kv_buffer` to an UNMASKED write -- every rank then claims every real
token and the compact row keeps whichever write was last. #355's bound does
not eat this (it bounds the write to the buffer, and only inside the masked
kernel). Our WEIGHTED owner rule (#173) was immune throughout: it derives
ownership from `out_cache_loc`, the tensor the wrapper already narrows. Fixed
in two places rather than upstream's one -- `narrow_pcg_token_views` /
`restore_pcg_token_views` cover all THREE piecewise ops (upstream's diff
touches one and misses `unified_sparse_attention_with_output`), and
`dcp_even_write_mask` refuses a mask-less token-sharded write by name instead
of returning `None`. So the rule for this family: a token-axis FB field the
backend indexes alongside Q/K/V is narrowed with them, and an owner rule with
no usable mask fails loudly -- there is no correct maskless DCP write. See
docs/dev/NOTE_472_pad_positions_dcp.md.

**MERGE DUTY -- bookkeeping-mutation sites (#404 family).** The per-request
accounting clocks (`decode_batch_idx` / `extend_batch_idx`,
`kv_committed_len` / `kv_allocated_len`, `spec_verify_ct`) and the
`maybe_evict_swa()` call have a closed owner register:
`test/registered/unit/spec/test_decode_bookkeeping_ownership.py::_OWNER_SITES`.
It is an AST scan of the whole `srt/` tree, so ANY merge that adds, removes or
recounts such a mutation turns it red -- including merges that never touch a
scheduler file. **A merge that introduces a new mutation site pulls
`_OWNER_SITES` along in the SAME merge, and the entry carries the audit, not
just the count**: under which lock/owner the mutation runs, whether it can
advance a clock (a fast clock fires SWA eviction inside the overlap race
window and releases the SWA prefix lock early -- the hazard the register
exists for), and whether it shares a pool with a second worker (#444/#450).
A site that does not survive that audit is NOT registered: it stays red and
the defect is written up. Registering to silence the test is the one forbidden
resolution. #496 discharged the backlog this rule was written for -- the
dual-group lane (#274) and the kv-session-offload spill had both landed
mutation sites without an entry.

## 13. Serving surface
OpenAI-compatible with `--reasoning-parser qwen3 --tool-call-parser
qwen3_coder` (server-side fix, no template patches); fast lane, priority
scheduling, admission throttle, prefill delayer; training tenant + idle
workbench (ledger + pause rung); `/session_handover`; `/kv_reshard`.

Class-3 video enhance, adaptive chain planner (#451,
`video_enhance/chain_policy.py`): given an ffprobe source and a target, it
generates the chain shapes that reach it (`full`, `rife_only`,
`pre_downscale` with the SR entry point solved off the measured frontier,
opt-in `decimate_resynth`), prices each against the per-stage rate table AND
the §6.2 reservation through `plan_job`, and picks the least lossy feasible
one or refuses with every candidate's numbers. `full` wins automatically
whenever it fits. Absent stage rates make a candidate unpriceable rather than
optimistic; `allow_estimates` prices them by a labelled linear-in-pixels
extrapolation. `pre_downscale` and `decimate_resynth` are *recommendable but
not runnable* on the M2 executor (no scaled/strided decode) and say so;
`require_runnable` excludes them. Mode + one-line reason are in the job
status. Tipping points are correct for the 4.6/fp32-parity P1 table; the
fp16-TRT operating point is unmeasured.

Class-3 video enhance, RIFE version ladder (#460, `video_enhance/rife_ladder.py`):
eight selectable rungs (4.6, 4.15, 4.15.lite, 4.16.lite, 4.17, 4.17.lite, 4.18,
4.26) off four vendored IFNet files -- upstream ships several byte-identical
architecture files, so 4.15/4.17/4.18 share one module and the lite trio
another, with weights still pinned per version in `rife.KNOWN_WEIGHT_SHA256`.
Each rung carries a quality rank (configurable, and labelled **ASSUMPTION** in
every report -- no quality gate has graded RIFE output in this tree), a
frontier keyed `(version, card, resolution, scale)` with measured/estimate/absent
cells, a VRAM class, and a weight state; the registry *refuses* a rung that is
neither present on disk nor sha256-pinned. `auto_rife_version` walks the ladder
in quality order and takes the first version whose whole chain clears the
existing aggregate gate, so no per-pair budget has to be invented; a variant
with no measured frontier is never entered and surfaces as `measure_first`
instead. `pin_rife_version` overrides the ranking, the budget and an absent
frontier (that is how a GPU window runs what it is about to measure) and says
so. The version-keyed frontier is the authority for the interpolation stage
wherever the ladder knows the card -- the shared `StageRateTable` is keyed
`(stage, card, resolution)` and cannot tell 4.6 from 4.26 -- and the shared
table still stands for cards the ladder has never heard of.
`scripts/video_enhance/fetch_rife_weights.py` establishes a pin only with
`--record-new-pin` and refuses an unpinned re-download. Seeded with ticket V's
sixteen measured cells; six of the eight rungs are ABSENT and are TICKET_460's
work list.

Class-3 video enhance, stage-pipeline pricing (#457 desk half,
`video_enhance/stage_pipeline.py`): the Regime-B counterpart to the chain
planner's Regime A. Stages are placed on cards and throughput is
`1 / max(card load)` rather than `1 / sum(stage costs)`; an exhaustive sweep
over placements is cheaper than any heuristic at this size. Card crossings are
priced as a host bounce (D2H over the sender's link, H2D over the receiver's,
each charged to the card that carries it) because this rig has no NVLink and no
GPUDirect P2P; barlink BAR1 is the named alternative and carries an **absence**
for raw video frames, so a plan needing it is refused rather than guessed. Two
hard constraints are enforced, not reported: SR and the tail resize stay
co-resident, and a card may declare a `max_transfer_mib` above which it is
disqualified as an endpoint (the x4 card's default is the 8K fp16 frame size,
derived from geometry, expressed per card and never as an NVML index). Transfer
that prefetch can hide behind the receiving card's own compute is charged at
its unhidden remainder only, and an estimate that is fully hidden does not
degrade the verdict's provenance. `frames_in_flight` buys latency and nothing
else, bounded by `max_latency_s`. `replicated_throughput` prices Regime A off
the same table in two explicitly named readings -- strict (an absent stage
drops the card: lower bound) and `omit_absent_stages` (drops the term: upper
bound) -- so a comparison between the regimes cannot silently mix them. Verdict
for 1080p@25 -> 2160p@50 on ticket V's numbers is in
`TASK_333_M2_VIDEO_ENHANCE.md` §17.5 and pinned by a test. Stage-level
replication (splitting one stage's frames across cards) is not built.

Class-3 video enhance, fused SR tail resize (#457 build half,
`video_enhance/fused_tail.py`): the 8K->4K Lanczos-3 downscale is appended to
the pinned SR graph before the TensorRT build, so one engine emits 4K directly
and the 189.84 MiB 8K intermediate is never materialised (engine output drops
to 47.46 MiB/frame). Lanczos-3 is not an ONNX op and no substitute filter was
needed: for a resample ratio p/q in lowest terms the tap vector repeats with
period p, so at p=1 -- any exact integer decimation -- every output pixel
shares one vector and the resample **is** a stride-2 depthwise convolution
with edge padding. Four opset-16 nodes (Pad+Conv per axis, separable), no
opset bump, and the graph computes the reference filter rather than an
approximation: graded on the CPU provider against the existing
SR-fp32-then-`lanczos3_resize` path at **145 dB / max abs 3.6e-07**, with a
deliberately-wrong `nearest` arm **rejected at 17 dB** so the gate is shown to
be able to fail. The `bicubic_antialias` route that was NOT taken stays
buildable for comparison and is the loser's price in a number: 40.01 dB
against a 40 dB threshold, and it needs opset 18. Any non-halving geometry is
refused by name and keeps the separate stage; `apply_tail_torch` is the torch
twin (registered-test reference and per-arch fallback). Artifact provenance is
the existing derived-sidecar chain -- pinned -> fused -> fused fp16, each step
CPU-loaded before its sidecar is written -- and `derived_fused_tail_model`
returns the **net** scale (x2), so a stage built from it needs no knowledge
that a fusion happened. `stage_pipeline.fuse_stages` /`absorbed_tail_rates`
carry the consequence into the pricer as three coupled changes (cost,
handed-on geometry, discharged co-residency), which removes every refusal from
the placement sweep and turns the Regime-A comparison from a bound into a
value. Re-priced verdict and the honest negative result (21.24 pipeline /
23.55 replicated src-fps, both ESTIMATE, both short of 25 -- the ffmpeg encode
round trip is the wall behind it) in `TASK_333_M2_VIDEO_ENHANCE.md` §17.7,
pinned by `test_stage_pipeline.FusedVerdictTest`. The fused stage's ms/frame
is BOOT-PENDING (`TICKET_460_rife_frontier.md` §5).

Class-3 video enhance, in-process zero-copy NVENC (#484, `video_enhance/
codec.py`): the §8.1 sink, built and gated OFF. TASK_333 §9.5's open
"incorrect usage of CPU input buffer" is diagnosed at instruction level on the
installed 2.2.0 extension and it was **the earlier workaround's own doing**:
`PyNvEncoder::Encode` selects its device path with
`PyObject_HasAttrString(frame, "cuda")` -- not with
`__cuda_array_interface__` as the docs say -- and throws error 8 when the
attribute is absent under `usecpuinputbuffer=False`. The `__slots__` view
added to hide `__dlpack__` (whose stream argument the binding passes
positionally and torch declares keyword-only) hid `cuda` with it, so the
fix for symptom one created symptom two and both read as one wall.
`_NvencDeviceFrame` is the shape the binding actually consumes:
`frame.cuda()` returning the two NV12 plane views NVIDIA's own `AppFrame`
describes -- luma `(H,W,1)/(W,1,1)`, interleaved chroma
`(H/2,W/2,2)/(W,2,1)`, `version` 3, no `stream` key -- over the one
contiguous tensor `rgb_to_nv12` already produces, so no copy and no host
bounce. A SECOND defect only the card could show: NVENC reads that surface on
its own engine with **no dependency on the stream that wrote it** and does not
complain about a half-written frame -- it encodes it. 60 frames at 720p:
8.59 dB with no ordering, 15.65 dB with a current-stream sync, 9.43 dB when
the session's own `cudastream=` option is passed instead (not honoured),
against an ffmpeg baseline of 15.65 dB. Every arm delivered exactly 60 frames,
so none of it is visible in a count -- expect this from any zero-copy consumer
that is not a torch operator. `_StrictFakeEncoder` is the binding's branch in
Python and is a real falsifier: it reproduces the recorded error message
against the pre-#484 wrapper and passes against the current one (the old fake
accepted anything, which is how three windows of green tests coexisted with a
lane that had never encoded a frame). `auto` no longer flips the sink merely
because the package imports -- that is how the defect reached a preview lane
that had never asked for it -- but consults `SGLANG_VIDEO_INPROCESS_NVENC`,
defaulting to the **named ffmpeg bootstrap fallback**; an explicit
`backend="pynvvideocodec"` ignores the switch, and `auto` falls back with a
logged warning plus `EncodeStage.fell_back_to_ffmpeg` so a measurement cannot
be credited to the lane that did not run it. The encoder session is a LEDGER
post (`frame_math.nvenc_session_bytes`, `EncodeStage.session_bytes`,
`chain_reservation(inprocess_nvenc=True)`), zero on the ffmpeg lane because
that memory belongs to a process this runtime cannot see or evict -- which is
the VRAM half of the ONE-RUNTIME argument as a number. It is a FUNCTION of
geometry, not a constant: 51.8 MiB measured at 720p and 263.8 MiB at 2160p,
fitted affine (~25.3 MiB + ~30 B/px, about twenty NV12 frames); two points and
two unknowns, so a third geometry is its first real test. **EXECUTED on card
0, 2026-08-03**: the parity gate (`scripts/video_enhance/nvenc_parity.py`,
decode-roundtrip PSNR/SSIM per frame in both arms) PASSES at 15.65 dB against
15.65 dB, 60/60 frames, with the `wrong-chroma` can-fail arm rejected at
11.22 dB; a single-frame plane check at 200 Mbit/s puts the lanes at the same
pixels (luma mean|d| 0.12, chroma 0.13, identical in both arms). **1.72x** on
ms/frame at 720p (4.881 vs 8.376, three runs each, the 3.50 ms gap 2.6x the
noisier arm's whole spread). The default stays ffmpeg: the gate ran on one
card at one geometry and the chain's operating point is 2160p with two output
frames per source frame, which is the remaining open measurement
(TICKET_484 §4). `DESIGN_484_inprocess_nvenc.md`,
`TICKET_484_nvenc_window.md`.

Class-3 video enhance, stage-level replication (#484,
`video_enhance/stage_pipeline.py`): one stage's frames split across cards --
what §17.7.5 named as the next lever for the stage that binds after the
fusion, and what this catalog previously recorded as not built. A placement
value may now be a tuple of cards, and the split is a **water-fill, not a
halving**: shares bring every participating card to the same finishing time
(`sum_i max(0, (P - fixed_i)/stage_i) = 1`, solved in closed form by walking
the fixed-load breakpoints), so a card already loaded takes less and a card
past the period takes nothing. That is the per-family x per-phase law applied
to one stage, cut by that stage's own binding resource and reported per card.
Two limits are refused by name rather than approximated: TWO replicated
stages make the split a linear program instead of a water-fill (a water-fill
applied twice returns an answer indistinguishable from the optimum), and a
co-resident stage cannot be spread. The x4 taboo is unchanged and is judged
on the frame that crosses -- replication makes crossings fewer, not smaller.
Worth **21.24 -> 24.37 src-fps (+14.7 %)** on ticket V's fused table with
encode split across the two 3080s, moving the bind onto the 5090's own chain,
pinned by `StageReplicationTest`; both figures inherit that table's ESTIMATE
provenance. It is the SMALLER of #484's two levers and says so: §17.7.5's own
arithmetic gives 31.25 src-fps for encode at ~0, i.e. removing the host round
trip beats dividing it. `best_placement` only replicates stages the caller
names in `replicable=`, because a split encode emits two elementary streams
the executor must interleave back into output order and that wiring is not
built.

Class-3 video enhance, streaming-input admission (#448 desk half,
`video_enhance/streaming.py`): source kinds finished/growing/live with named
refusals (no growing source on the chunk executor — the split is verified
against a final frame count that does not exist; no live source under
`stall` — back-pressure cannot reach the feed, so the frames lost during a
stall would be uncounted), a bounded output buffer whose depth is a declared
seconds-deep watermark converted through the output rate, a growing-source
adapter that distinguishes "not yet" from "no more" (with an idle timeout),
and a sliding-window in/out fps accounting exposed on the job status for the
#344 live watch. Finished sources keep the depth-1 bridge unchanged.

## 14. Dashboard
Guided config wizard with honest refusals, comm benchmark suite with
anonymization gate, energy metering (tok/s + J/token), benchmark tiles with
measured/estimate/absent provenance, one-click knee-point probe, self-update
with auto-rollback, GitHub result posting (opt-in PAT).

## 15. Model bring-ups (boot-proven)
Qwen3.5/3.6 family (all quants), Gemma4 26/31B (+GGUF, quadratic-mask skip;
Gemma3RMSNorm runs the fused sgl-kernel path for 2-D and high-rank inputs,
adopted from upstream #32670 — do not re-add an eager-only forward_cuda),
Llama family, Mistral Small 24B FP8 + ministral3 SWA fix, Deckard-40B/Tess-27B,
122B-A10B offloaded, 35B-A3B, DeepSeek-V4-Flash-0731 GGUF TP=3 offloaded with
OWN sm86+sm120 attention paths (e4m3 bit-decode, f32 staging, indexer arch
dispatch, torch/triton reference-twin parity: indexer mask oracle, SWA
page-index wrap oracle, page-table rounding, top-k seq_len contract).
The torch paged-MQA indexer logits are chunked on BOTH axes — KV positions
(#426) and query rows under a per-rank MiB budget (#449,
`SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB`, converted with that rank's own head
geometry as #395 does) — bit-identical to the single-pass form, collective-free
inside the loops. This BOUNDS the per-query-token duplication of the KV gather
(one copy per query token, ANALYSE_447 L1); it does not remove it, and the
speed effect is unmeasured (no GPU window taken).
Nemotron-Puzzle class structurally covered, unbooted.

## 16. Measurement / window infrastructure
gpu-arb (UUID-based holder + heartbeat — stop the heartbeat BEFORE releasing),
forward_peak.py (VRAM corridor judged AT PEAK, not idle), cachetrim with
--ready-url self-retirement, expert_stats (router distribution + hit rate),
CollectiveClock (compute vs wait per rank), measured-KV-budget stale-boot trap.

## 17. META: combination matrix + eviction doctrine
Every "can asset X live at tier Y under primitive Z" question is a matrix-cell
lookup before it is a design question, and the cells are already enumerated:
`ANALYSE_456_dsv4f_matrix_sweep.md` is the asset x tier x primitive x control
sweep (§2.1 lists the occupied cells with their evidence, §2.2 names the empty
ones explicitly, so "nobody looked" and "somebody looked and rejected it" are
distinguishable rather than both reading as silence). Cell **#302b — cold
experts under CUDA graphs — is now BUILT and no longer empty**: the breakable
route (eager fetch into fixed slots before replay, compute captured, graphs
addressing SLOTS) exists behind `SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable`,
gated OFF and never booted (§3, `DESIGN_462_breakable_route.md`). The
neighbouring cell **#302c (per-expert runtime dispatch) stays empty**, but its
named prerequisite is discharged: DESIGN_462 §5 records how a foreign
contribution enters the combine, so #302c starts from a seam rather than from a
redesign. The in-graph fetch remains register-rejected. Anything that EVICTS
consumes `DESIGN_407_memtier_registry.md` §8's one global importance ladder
(cold second model, inactive layout/graph families, cold experts, idle sessions,
active work last and never out of FCFS order — coldest-first within a class)
instead of writing a local victim policy — that ladder is now EXECUTABLE rather
than prose (`model_executor/short_term_offload_register.LadderRank` /
`plan_spill`, #286), so a new consumer calls it instead of restating it, and a
class added without a ladder rank fails at import; and anything that decides at runtime
whether a path is worth its cost is an instance of `DESIGN_363_regime_controller.md`
§20.1's worth-it autocheck rather than a new flag. Read those three before
adding a cell, and register the answer back into them in the same merge.
