# Feature Catalog — what this fork already has

Read this BEFORE searching the tree or building anything: most capabilities you
are about to look for already exist. Rules: (1) never declare something
"impossible" or "missing" without checking this file, FEATURES_VS_UPSTREAM.md
and `git log`; (2) whoever merges a new feature updates the matching section in
the SAME merge; (3) **a conditional line here is a POINTER TO A PREDICATE, not
the predicate.** Since the #500 reach audit every "only when / gated on / not
combinable with" line carries the gate's `file:line`; before you act on one,
read that predicate. Where this file and the code disagree, the code wins and
this file is corrected in the same change (CLAUDE.md, MECHANISM REACH).
(4) There is a SECOND flag/env registry: `python/sglang/srt/planner/flags.py`
carries `requires` / `mutually_exclusive_with` / `allowed` / `model_compat` per
flag and drives the dashboard. It is authoritative for nothing on its own —
audit #500 found three of its curated edges NARROWER than the runtime — but it
is the only place the full flag surface is enumerated, and its own
"CRITICAL fork-capability note" (`flags.py:38-46`) stated the TP>kv_heads
capability correctly for as long as this file stated it as a special case.
Its uneven-TP edges are no longer maintained by reading: every declared
`requires` / `mutually_exclusive_with` / `requires_any` edge on those flags is
DRIVEN against the real `ServerArgs` validation by
`test/registered/unit/planner/test_flag_registry_contract_500.py`, so a
registry that forbids more than the server does fails a test instead of
silently greying a field. Add an edge there when you add one to `flags.py`.
Last full refresh: 2026-08-02 (tip 33148dbe0f); reach-audited 2026-08-03 (#500,
tip 3b7569f664 — see `docs/dev/AUDIT_500_mechanism_reach.md`).

## 1. Uneven parallelism (core differentiator)
- **Uneven TP** `--rank-tp-ratio` (+ `--rank-gpu-id` for placement): per-card
  weight shards. The two are INDEPENDENT — `--rank-tp-ratio` is a pure
  partition description and is legal with no placement flag at all (the
  cross-vendor two-launcher bring-up relies on it, `server_args.py:9644`);
  `--rank-gpu-id` conversely requires `--rank-gpu-memory-mib` or
  `--rank-tp-ratio auto` (`server_args.py:9434`). `auto` = byte-proportional
  from NVML totals minus auto reserve, needs `--rank-gpu-id`
  (`server_args.py:8971`) and collapses to the EVEN split on uniform budgets
  (`server_args.py:9157`, which then disarms the family flags). With
  `--rank-perf-tune both|dec|enc|maxkv|phase-prefill|phase-decode`
  (`server_args.py:723`) the planner solves the vector; the two `phase-*` arms
  are the #354/#357 phase-optimal recipe. The value that ENGAGES the solver is
  `--rank-tp-ratio auto-performance` (`server_args.py:433`) — plain `auto` is
  the capacity-first default and never solves for speed
  (`_CAPACITY_FIRST_DEFAULT_NOTICE`, `server_args.py:740`).
  `auto-performance` is refused under `--pp-size > 1` (`server_args.py:9394`);
  plain `auto` has the per-stage path.
  Unit system: `tp_units`/`tp_family` per layer class (16-element MLP family,
  boot-refused per rank at `distributed/utils.py:975`; coupled-dim rule:
  gate_up output and down_proj input partition the SAME intermediate dim and
  must coarsen identically — one SYMMETRIC block, `lcm(64,128)=128`,
  `layers/linear.py:181`). The `block_configs` (Nemotron-NAS/Puzzle) support is
  a PLANNER-side weight-byte census for the auto-performance cost model
  (`LayerFamilyCensus`, `uneven_perf.py:2894`), not a runtime shard-family
  table.
- Sibling flags (each needs a resolved non-uniform base plan,
  `server_args.py:10023`; `--rank-vocab-ratio` is additionally incompatible
  with `--enable-dp-lm-head`, :9911; `--rank-moe-resident-fraction` refuses a
  per-rank vector when `moe_tp_size != tp_size` and refuses env/flag
  disagreement, `moe/resident_fraction.py:132`/:107):
  `--rank-mlp-ratio`, `--rank-vocab-ratio`, `--rank-moe-ratio`
  (per-path meaning below — do not read this as "experts between ranks" in
  general), `--rank-moe-resident-fraction` (GPU/host split WITHIN a rank),
  `--rank-kv-ratio` (`coupled|capacity` (alias `auto`)`|speed|vector`,
  `server_args.py:484` — decouples KV split from weight split; `capacity` is
  the measured one-boot-convergence mode that `--rank-perf-tune maxkv`
  selects, `speed` degrades to it without bandwidth scores; only the EXPLICIT
  VECTOR is refused without a non-uniform base plan, `server_args.py:9608` —
  the derived modes degrade to `coupled` with a warning, :9615; every
  non-`coupled` value additionally requires `--rank-gpu-id`, because DCP is
  auto-engaged only on the placement path and the flag is otherwise inert,
  #500-B2),
  `--rank-auto-reserve-mib` (also usable WITHOUT any uneven flag: a pinned
  reserve then sizes the plain path as
  `mem_fraction_static = (NVML total − reserve)/total` exactly, #332,
  `server_args.py:9370`), `--rank-gpu-memory-mib` (absolute per-rank MiB
  budget with a line-item ledger incl. lane pools; the per-rank LIST form
  additionally needs no weight vector under the weightless-KV lane or PP>1,
  `server_args.py:9694`).
  Each of these vectors has an env twin that OVERRIDES the flag —
  `SGLANG_UNEVEN_MLP_VECTOR` / `_MOE_VECTOR` / `_VOCAB_VECTOR` /
  `_TOKEN_VECTOR` (`environ.py:573-588`, precedence at
  `distributed/utils.py:490`) — and they are what the planner emits to pin a
  solved vector into a launch (`planner/runner.py:411`).
  Read `--rank-moe-ratio` precisely: under the **#82 GGUF expert-dim shard** it
  moves whole experts and therefore the COMPUTE assignment (owner runs the
  expert, foreign ids remap to a zero pad, the TP all-reduce sums the disjoint
  partials); on every other MoE path it splits the expert INTERMEDIATE dim, so
  every rank still computes every routed expert and only the weight slice
  moves. `--rank-moe-ratio link` (#394 slice 3) solves the vector instead of
  taking it: the GPU-resident expert mass stays exactly where the base plan put
  it (VRAM-neutral) and the STREAMED remainder is apportioned by the measured
  link weights, which equalises the per-rank transfer time the group waits on.
  Refused by name, all five in the launcher: nothing to move (the mass test
  `cold_total <= 1e-9`, i.e. offload off, `expert_compute_placement.py:527`),
  link provenance `absent` (:951), `ep_size>1` (:916), no resolved uneven-TP
  base plan (:906), and no rank→physical-card vector (:941). Resolved ONCE in
  the launcher — a symbolic value that
  reaches a worker is a hard error there, never a silent fall back to the base
  plan.
  **ACCEPTANCE-EVIDENCE 2026-08-03** (green-corridor re-proof,
  `/spinning/gpu-battery-results/2026-08-03_439_green/RESULTS.md`, DeepSeek-
  V4-Flash UD-IQ3_XXS TP=3 on 5090 + 2x 3080, 900 tok x 3 x 1 warmup, repaired
  reserve `2200,1800,1800`, all five gates PASS): the clock moved off the x4
  card (tp1 H2D 1197.4 → 706.8 GiB), transfer term 199.3 → 139.3 s =
  **1.4307x** against a prediction of 1.427x, and **-6.42 %** end-to-end
  ms/token against a same-window A-vs-A floor of CV 0.223 % / spread 0.424 %,
  with per-card corridor minima 655-1318 MiB at 1 Hz against the 400 MiB floor.
  Those are the FINAL, WORK-MATCHED dump revision (163486 vs 163572 tokens
  across the two arms), which is the only basis on which two arms may be divided
  by each other. The night window
  (`/spinning/gpu-battery-results/2026-08-03_439_confirm/RESULTS.md`, first
  tokens through the path, corridor-red at 211-251 MiB free) re-reads to
  **1.4253x** on the same basis against its own 1.411x prediction, so the two
  windows agree to 0.4 %. The pre-teardown revision reads 1.5028x / 1.496x and
  is ~5 % high because the two arms' dumps land at different fractions of their
  runs (96.8 % vs 91.9 %); it is not quotable — see `ARM3_COMPUTE.md`, "Which
  revision to read". The DESK-WRITTEN label is lifted for this path and the
  three 2026-08-02 defects are confirmed fixed on hardware.
  `link-calibrated` (per-rank cold-traffic coefficients from a prior boot's
  #390 dump, `SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS`, required and read ONLY
  under this symbol) is EXPERIMENTAL and **FALSIFIED** on exactly TWO
  load-bearing legs. (1) END-TO-END, the economically decisive leg: it measured
  **-0.94 %** ms/token against the baseline, inside that window's own 4.09 %
  spread — a non-result, and probe-measured rather than dump-derived, so no
  revision choice can move it. (2) MECHANISM: the coefficient treats the cache
  hit rate as a property of the RANK and it tracks the SIZE of the owned expert
  range instead (night window tp1 0.8450 → 0.9050 as its range shrank 72 → 58,
  tp2 0.8474 → 0.7814 as it grew 72 → 89; independently reproduced in the green
  window, where the one rank whose range did not move did not move its hit
  rate), so the solve overloaded tp2 and made it the new clock. The
  transfer-term comparison is NOT a leg and is no longer cited: work-matched,
  `compute-cal` reads **1.4573x** against `compute`'s **1.4253x**, i.e. it
  slightly WINS that term, and the older "reached only 1.439x against 1.496x"
  sentence divided two counters sampled at different work points. Plain
  `link` REFUSES while the coefficient variable is set rather than silently
  running the falsified solve — before #458 that env alone selected it.
  Registered in `planner/rejected.py` (`moe_link_calibrated_coefficients`).
  **Propagated into code in #523** (#482 was the docs/harness half): the help
  text (`server_args.py:2450-2477`), the rejection verdict
  (`planner/rejected.py:657-672`) and the module docstring + resolver strings
  (`layers/moe/expert_compute_placement.py:74-89`, `:166`, `:847`, `:868`) now
  carry the WORK-MATCHED figures and rest the rejection on the end-to-end and
  mechanism legs only; the two test assertions that pinned `"1.496x"` /
  `"1.439x"` now pin the corrected strings AND assert the withdrawn pair does
  not reappear as a measurement. `ROADMAP_456_matrix_execution.md:32-33` and
  `ANALYSE_456_dsv4f_matrix_sweep.md:43-46` are corrected in the same change.
- **Uneven DCP** (`dcp_size` + token vector): token/KV sharding across ranks,
  weighted owner rule, SWA-hybrid support. The replication+token-shard axis is
  NOT kv-head-count-gated: the predicate is
  `dcp_size > 1 and get_tp_partition_ratios() is not None`
  (`distributed/utils.py:346`) — it never reads the kv-head count and does not
  require `dcp_size == tp_size`, so it is live for EVERY kv count. Three ways
  in: a `--rank-tp-ratio` plan, the weightless-KV lane
  (`... or self.weightless_kv`, `flashinfer_backend.py:689`), and a #274
  lane's context-local overlay (`utils.py:168`). The separate REPLICATED-KV
  attention geometry is `tp_plan_active(tp) and total_num_kv_heads < tp`
  (`utils.py:1048`; `kv == tp` deliberately excluded, reverted on
  measurement), and it refuses without DCP spanning the group
  (`flashinfer_backend.py:735`). MLA models have NO uneven-TP DCP combine and
  are refused by name (`layers/dcp/comm.py:81`). `dcp_size` is auto-set to
  `tp_size` only on the `--rank-gpu-id` path (`server_args.py:9845`); a
  non-`coupled` `--rank-kv-ratio` WITHOUT a placement is therefore refused by
  name rather than accepted-and-inert (#500-B2).
  **Draft-KV-DCP**: draft KV token-sharded (−67 % draft KV), admitted by
  `_reject_unsupported_draft_kv_dcp` (`server_args.py:7448`). Its gate is the
  INSTALLER's own predicate — `self.uneven_weighted_dcp_enabled()`
  (`server_args.py:8457` = `SGLANG_UNEVEN_DCP_WEIGHTED=1` OR any non-`coupled`
  `--rank-kv-ratio`), plus a non-uniform `--rank-tp-ratio` and
  `dcp_size == tp_size > 1` — so BOTH routes to the weighted owner rule are
  admitted, the same set the sibling speculation×DCP gate takes at :7628
  (#500-B3 fixed; it previously spelled the env pair literally and refused the
  flag route, putting the −67 % win out of reach for every `--rank-kv-ratio`
  boot). `SGLANG_UNEVEN_DCP` is not a separate condition: it only auto-sets
  `dcp_size` (:9845), which the gate checks directly. "Replicated is the DEGRADED layout above TP>kv_heads" is a
  measured recommendation in the CLI help (`server_args.py:1513`), not a code
  predicate — nothing compares TP to the kv-head count for the draft pool
  (audit #500, S1-34: NOT-FOUND, superseding the earlier "two-sided rule"
  reading of this line). LSE log base follows the attention backend
  (`NATURAL_LOG_LSE_ATTENTION_BACKENDS = {"flashmla"}`, `dcp/comm.py:44`).
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
  family. #503: this axis is REPORTED only where its own gate holds — a boot
  on the default `--rank-kv-ratio coupled` without `SGLANG_UNEVEN_DCP` never
  reaches `dcp_size > 1` (`server_args.py:9845-9853`), head-shards the KV
  cache and installs no token vector, and the planner now refuses the axis by
  name there instead of pricing a lever that boot cannot actuate. Separately: PROJECTION-weight replication (`attn_kv_replicated`) is
  strictly `kv < tp`, the `<=` flip is measured-rejected, and at 24q/4kv over
  3 ranks it is structurally unrepresentable (`units % groups != 0` and
  `groups >= n` in the #116 alignment repair) — generalizing it is an unbuilt
  #169-family posten, named in `NOTE_492_attention_replication_axis.md` §2.2.
- **TPxPPxTP**: pipeline across rigs with per-stage TP groups. Slices 1+2
  merged (cross-rig PP=2 over 40G, full decode graphs on both stages incl.
  sm75). Slice 3 merged and cross-rig pp=2 validated: world-MIN
  `max_total_num_tokens` before the reduce, `--pp-stage-ratio`
  (score-proportional, snaps to full-attention boundaries), stage-local mamba
  slots (per-stage GPU groups must be pairwise DISJOINT,
  `server_args.py:8669`), `auto` under PP with an agreement gate (all stages
  must derive the same vector, `server_args.py:9116`), `SGLANG_PP_SHAPE_CACHE` cuts
  boundary-send by −9.8/−9.2 % at bs=1 (0-1 % floor otherwise) — note the
  in-server counter reads 249 µs, which is not the standalone wire-transfer
  figure.
  **#481 closed three PP defects from the #445 window.** (a) The declared-depth
  probe joined `config.json` onto `--model-path`, which for EVERY GGUF launch
  is the `.gguf` FILE (rig-runbook §4.5.4b) — so `--pp-stage-ratio` refused
  every GGUF checkpoint with "hidden layer count is not readable". It now
  applies the #402/#414 sibling-directory canon (`declared_config_path`,
  `server_args.py`), the same one `_log_rank_plan_vectors` already used at
  `:9276-9280`. (b) `--rank-moe-resident-fraction` was the one rank vector in
  this family validated against `tp_size` alone; it now also accepts the
  world-length form (`pp_size x tp_size`, world-rank order
  `pp_rank * tp_size + tp_rank`) that `--rank-gpu-id` (`:9029-9036`) and
  `--rank-gpu-memory-mib` (`:9262`) already take, and the consumer indexes a
  world-length vector by world rank (`resident_fraction._rank_in_vector`). A
  tp-length vector keeps its old meaning (same pattern on every stage) and
  every non-PP launch is byte-identical. (c) `expert_stats` tagged its dump
  `tp{moe_tp_rank}ep{moe_ep_rank}`, which two stages' rank 0 share — both wrote
  `<path>.tp0ep0.json` and the later dump replaced the earlier. The tag is now
  built by `expert_stats.moe_rank_tag`, which prefixes `pp{rank}` ONLY when
  `pp_size > 1`, so existing dump filenames do not move. Tests:
  `test/registered/unit/test_pp_defects_481.py`, 16 hermetic, four executed
  can-fail arms (sibling-dir candidate removed → 3 red; world length dropped
  from the validator → 1; consumer re-indexed by the in-stage rank → 1; call
  site back to the pp-blind inline tag → 1).
- **TP5+ emulation** via multi-rank co-location (several ranks per card),
  gated ONLY on duplicate entries in `--rank-gpu-id`
  (`entrypoints/engine.py:1563`). The NCCL settings that follow
  (`NCCL_MULTI_RANK_GPU_ENABLE=1`, `NCCL_NVLS_ENABLE=0`,
  `NCCL_MAX_CTAS=max(1, 8//max_colocated)`) are opportunistic defaults, each
  skipped if the operator set it; MPS absence is a warning, not a refusal
  (:1598). Nothing excludes barlink — it is selected by
  `SGLANG_BARLINK and world_size > 1` (`parallel_state.py:687`), which cannot
  see the placement. The only forced consequence of co-location is
  `disable_custom_all_reduce = True` (`server_args.py:9820`).


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
token vector keeps the relative base-plan pricing (`unfunded = res[r] <
_fund_demand[r] <= _fund_base[r]`); a MATCHED one (`--rank-kv-ratio
capacity|speed`, and the phase arms since #435) has no unused capacity to
price, so every rank is checked ABSOLUTELY (`unfunded = res[r] <
_fund_demand[r]`) against the derived reserve demand on ALL cards — the branch
is `uneven_perf.py:5967-5981`, the two bases `:6027-6034`. The gate needs a
derived per-GPU auto reserve; without one (`if demand_by_gpu:`, `:5919`) it is
silently absent and every candidate is reported fundable (#500-B6). Before #437 `capacity` mode accepted
16,1,1 at reserve 3000,2700,2700 -- #264's OOM config -- because it compared
a matched residual against an identical matched base; the capacity-directed
objective did not consult the gate at all. #330's 400 MiB corridor is priced
alongside the demand and REPORTED (`CORRIDOR-TIGHT`), never binding — it
appears in no admissibility term (`admissible = floor_ok and (knee_ok or not
knee_binding) and unfundable is None`, `uneven_perf.py:6517`;
`_corridor_note` returns text, `:6051`).
`SGLANG_PLANNER_CORRIDOR_MIB` overrides it (`:5279`); the number lives once, in
`registry/ledger.py:69`.
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
state pool and the coupled KV vector follow it. The enumerator escapes the
kv-head grid ONLY when `attn_units < tp` (`grid = self.attn_units; if grid < n:
grid = max(self.q_heads, n)`, `uneven_perf.py:4133-4148`), and
`partition_units` then keeps >= 1 unit per rank (#62/#116; hard —
`distributed/utils.py:568`, `:589`). **That grid IS the runtime's, and #503
executed the check that says so** (audit finding #500-B1 is REFUTED —
`test_attn_replication_axis_492.py::TestWhatTheRuntimeCanActuallyDoToday`).
Two independent predicates govern this family and #500-B1 conflated them:
`attn_kv_replicated(tp_size, total_num_kv_heads)` is strictly `kv < tp`
(`distributed/utils.py:1081`; the `<=` flip is measured-rejected in its own
docstring) and is what `linear.py:1423` / `model_config.py:1332` read when they
build the q/k/v shards — so at `kv_heads >= tp` the k/v PROJECTIONS head-shard
and the enumerator's kv-head grid is correct. `uneven_dcp_kv_replicated`
(`dcp_size > 1 and get_tp_partition_ratios() is not None`,
`distributed/utils.py:346`) replicates the KV **POOL**, not the projections:
"the attention write gathers this rank's uneven projection shard up to
`get_total_num_kv_heads()`"
(`model_executor/model_runner_kv_cache_mixin.py:2721`). Pricing attention
WEIGHTS on the second predicate would model a layout the #105 ragged-kernel
guard refuses at the first forward. So "4 kv heads, 3 ranks -> only `[2,1,1]`"
is a property of the rig after all, the phase-prefill joint gains
(desk-predicted +1.0 points over the MLP-only cut on INT8-W8A8 and +6.9 on
FP8, both bracketed, see below) stand unchanged, and the #475 backtest
reproduces its four measured arms at rms 2.2 with `dcp_size` declared.
`planner/placement.py:813` (`replicated = kv_heads < tp or dcp_replicated`)
uses ONE flag for both mechanisms and therefore reports the k/v projection
heads as replicated under uneven DCP where the runtime shards them — an open
defect, narrower than #500-B1 claimed and in the placement report rather than
the solver. **#492 CORRECTION (the right empirical read on THIS rig, and #503
gated it on the predicate that installs it — the axis is reported only where
`uneven_dcp_kv_replicated` holds, i.e. not on a bare `--rank-kv-ratio
coupled` boot, which head-shards the KV cache and installs no token vector):
the axis is blocked by capacity, not the grid** — every token candidate is refused at
`--rank-perf-loose-ctx-percent 0` (the weighted owner rule funds
`min_r(P_r/v_r)` blocks, so concentrating the token vector discards the slow
cards' pools), and it only becomes reachable at loose 80-95 for +0.1 to +1.9
points against a 4-17x context loss; the solve prices the axis at both ends of
a CORE-FREE / CORE-PACED bracket and prints the geometric core/projection
crossover depth (8,533 attended tokens on Qwen3.6-27B — pure geometry, no
fitted constant). `NOTE_492_attention_replication_axis.md`. The solve REPORTS
the pair and does not install it (`if acand is None:` guards the argmax,
`uneven_perf.py:6558`) — the only runtime actuator for an attention vector is
`--rank-tp-ratio`, because none of the THREE named family plans is an
attention family (`("mlp", "rank_mlp_ratio"), ("moe", "rank_moe_ratio"),
("vocab", "rank_vocab_ratio")`, `managers/scheduler.py:5860`). Note the
mechanism itself has no whitelist: `_normalize_partition_plan`
(`distributed/utils.py:129`) validates only base presence, equal length and
positive ints, and layers opt in with an ordinary `tp_family=` argument
(`layers/linear.py:534`, honoured at `:657`) — today only `"mlp"` (8 sites)
and `"moe"` (1) declare one, so an attention family plan is a constructor
argument plus a flag, not a runtime redesign. The joint cut is additionally
OFF under `--objective energy` (`joint = tune in _PHASE_TUNES and not
decode_objective and not _objective_is_energy(server_args)`, `:6408`), and the
LANE-INVARIANT / LANE-SENSITIVE bracket is skipped entirely without a measured
per-rank bandwidth vector (`if not bw_scores …: return None`, `:5218`). Where
the flash/scan core's per-token mass would be
needed the model BRACKETS instead of estimating: the same solve is run at the
pure-GEMM and the measured-#231-GEMV lane extremes and the plan log states
`LANE-INVARIANT` or `LANE-SENSITIVE`. ANALYSE_299's "attention lever = 0.01 %"
does NOT transfer — it was computed under the pre-#475 model, in which aligning
two families' pacers is worth zero by construction. DESK/PREDICTED; the GPU arm
is `TICKET_485_int8_joint_arm.md`, details in
`NOTE_485_joint_phase_vectors.md`.

`--objective energy` with refusal over silent substitution at both ends (solver
`key_solver.py:1479`, boot `uneven_perf.py:5557`). Scope, not "end to end":
priceable goals are `("dec", "enc")` only (`ENERGY_PRICEABLE_GOALS`,
`key_solver.py:1421`), it refuses a second goal or constraints (`if
objective_is_energy and (goal_b is not None or constraints): raise`, `:2589`),
and on the boot path it disables the #485 joint pair space (`:6408`).
`planner/rejected.py` = machine-readable register of 24 discarded approaches —
check it before re-proposing anything. It is a READING surface first:
`check_combination` fires only on a full tag match (`set(e.tags).issubset(have)`,
`rejected.py:731`) and the two tag producers (`wizard.py:820/:1306`,
`rig_coupling.py:904/:951`) emit a vocabulary that reaches 5 of the 24 rows. The
other 19 are consulted by humans and by `register_json` (`/api/rejected`), not
by a gate.

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
plan's own capacity report. **The slice-1 pair solver is DECISION LAYER ONLY** —
`autocheck` has two call sites, both planner-side
(`planner/feasibility.py:503`, `planner/solver_api.py:590`), and
`--regime-phase-table` is a planner-CLI flag (`planner/cli.py:134`), not a
server flag. The RUNTIME half IS wired, though: `--regime-controller act`
builds a `RegimeActuator` (`managers/regime_runtime.py:721-727`, from
`managers/scheduler.py:3401`) that issues a #330 VRAM budget GROW and a #297 KV
reshard on the running scheduler (`managers/regime_act.py:182-189`), refusing a
shrink (`:151`) and a missing actuator (`:161`/`:169`), and refused at parse
time until `--regime-gate-evidence` names four measured items
(`server_args.py:4993`). What no build executes is the WEIGHT half: no pointer
flip, no diff spill, no pre-capture (#363 slices 2+, `ROADMAP_456` WAVE 4,
gated on #286). Switch-cost constants
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

**#483 is REFUTED: `check_regressions` already prices prefill on the RESOLVED
rates** (`resolved = model.rates`, `key_solver.py:4777-4780`, landed with #475).
What was missing was the PIN, not the fix: the only test that calls the function
is checkpoint-gated (`test_key_solver.py:433`, `skipUnless(_have(_FP8))`) and
skips in every hermetic run, so a revert was invisible.
`test_check_regressions_pricing_483.py` is the hermetic falsifier (synthetic fp8
checkpoint + lane profile; unresolved 3.54:1 vs resolved 9.70:1 rank spread,
2.85 % vs 10.70 % predicted prefill on the 2,1,1 -> 6,1,1 pair). Note what the
can-fail arm does NOT move: `gemm_format` / `gemm_lanes` come off `resolved` at
`:4795-4796` whatever priced the numbers, so the artifact named the right lane
while the arithmetic ran on the wrong one -- that is what kept the defect alive.

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
  in-graph fetch. Refuses by name at boot unless the RESOLVED decode backend is literally
  `breakable` (`if backend != "breakable":`, `offload_capture_gate.py:363` —
  `eager_on_graph` is a no-op otherwise → host reads inside a real capture) and
  the RESOLVED prefill backend is literally `disabled` (`if prefill is not None
  and prefill != "disabled":`, `:379` — a prefill chunk overflows the arena and
  a captured segment cannot wave-split). Both checks are SKIPPED when the
  backend cannot be resolved (`if backend is None: return`, `:358`, reached
  whenever `resolved_backend` swallows an exception, `:408-421`) — #500-B8. On
  DeepSeek-V4 the prefill half is already satisfied: the BCG-incompatibility
  rule set rewrites prefill to `disabled` (`server_args.py:8281-8309`). Both
  spellings of the refuted path still refuse.
  **DESK-WRITTEN, NEVER EXECUTED — no boot, no replay, no ms/verify figure
  exists, and F1's 5.3–8.4x is a Qwen3.6-35B-A3B ceiling that is NOT a DSV4F
  number.** F2 (per-layer break cost, decomposed) is the first measurement of
  the next window and gates default-on; **its instrument now exists (#494, §16
  break-cost probe)**, so F2 is a measurement ticket rather than an
  implementation one — the crossing cost is read off
  `scripts/dev/494_break_cost/summarise.py`, not reconstructed from logs.
  Tests:
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
  load-time-aware halves for fp8/GPTQ/AWQ and, since #123-GGUF, GGUF-MoE on
  CUDA — admitted per layer on the staging marker
  (`_OFFLOAD_CONDITIONAL_QUANT_METHOD_NAMES`, `expert_offload.py:2069`;
  `if getattr(layer, marker, False): continue`, `:2111`). The guard is a
  five-name DENYLIST (Ascend GGUF-MoE, MoeWNA16, three NVFP4 methods,
  `:2056-2064`) matched by class NAME, so every other quant method —
  unquantized, INT8, compressed-tensors non-NVFP4 — passes by default.
- **#394 cold-shard chain** (slices 1+2 merged): measured H2D provenance chain
  (env > card-probe > nvml-negotiated > refusal; `absent` unselectable as a MINIMUM
  (`SGLANG_MOE_HOST_SHARD_MIN_PROVENANCE`; `if raw not in ("measured",
  "estimate"): raise ValueError`, `expert_offload.py:800`; default `estimate`)
  and a hard refusal at the #439 door (`expert_compute_placement.py:951`) and in
  `ColdTierAssignment.__post_init__` (`cold_tier_fetch.py:218`) — but at the
  cold-tier door it DEGRADES instead: an absent provenance yields an equal ratio
  and `if ratio.is_equal: return None` (`expert_offload.py:1158`), i.e. the
  pre-#394 path, silently),
  `cold_tier_shm.py` shared-DRAM segments (UUID/BDF identity, manifest read
  lazily after load, header sealed last, PROT_READ views with kernel-enforced
  write protection). **Slice 2 wires the fetch path** (`cold_tier_fetch.py`):
  a rank-uniform owner map derived from the same `partition_cold_experts` the
  staging plan uses (plan `digest()` pins the uniformity), the cold pool
  ALLOCATED IN the segment rather than copied into it, and
  `MoEExpertOffloadCache._fetch` sourcing a delegated expert from the owner's
  `PROT_READ` view over this rank's own link. Behind
  `SGLANG_MOE_COLD_TIER_SHM=1`; with it off the GGUF streaming door warns once and falls back to the
  pre-#394 plan (`if not cold_tier_enabled() and not
  envs.SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE.get(): … return None`,
  `fused_moe_triton/layer.py:1450-1455`); the HARD refusal for delegation on
  disjoint expert shards is the separate marlin-repack door
  (`refuse_cold_shard_at_repack_door`, `expert_offload.py:3755-3782`). Two
  doors, two refusal shapes.
  **Honest scope of slice 2**: byte ownership moves, COMPUTE does not, so
  per-rank H2D is predicted unchanged.
  **Slice 3 (#439) moves the compute assignment** and is where ANALYSE_393's
  Path A′ lives. It needed no new mechanism: the #82 expert range IS the "moe"
  family vector, so the slice is a SOLVE plus its wiring
  (`layers/moe/expert_compute_placement.py`, `--rank-moe-ratio link`, see §1).
  MEASURED and re-proven in a green corridor 2026-08-03 at the repaired reserve
  `2200,1800,1800` (base plan `30407,18680,18680`, solved vector
  `213,104,157`): clock rank 199.3 s → 139.3 s = **1.4307x** work-matched,
  against the 1.427x its own model predicted, and -6.42 % end-to-end. The
  calibrated variant is falsified on its end-to-end and mechanism legs, not on
  the transfer term (§1). Two findings the night window added:
  `--rank-auto-reserve-mib auto` is INFEASIBLE on this recipe (it derives
  3968 MiB per card from the activation heuristic, leaving a 16512 MiB budget
  against 17.59 GiB of weights + runtime — the refusal now names the derivation
  and the pinned value that fits, `ServerArgs.derived_reserve_infeasible_note`),
  and the recipe is CORRIDOR-RED at `2200,1400,1400` (211-251 MiB free on the
  3080s) against 655-1318 MiB at the repaired reserve.
  BOOT-PENDING: the eager arms 1+2 of
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
  boot — so reaching it on the CAPTURABLE route takes both overrides. On the
  #462 BREAKABLE route it takes neither: `refuse_capturable_cold_tier` is armed
  only inside `install_capturable_buffers` (`expert_offload.py:2936-2940`),
  which that route never calls, and its eager `_fetch` keeps the peer branch
  (`if remote is not None and expert_id in self._remote_ids:`, `:2819`).
  Graphs incl.
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
  Refused under the CAPTURABLE route only — by name at boot (`if
  bool(envs.SGLANG_MOE_OFFLOAD_CUDA_GRAPH.get()):`,
  `expert_heat_migration.py:339`) and at migration time (`if
  self._capturable_ready:`, `expert_offload.py:3632`), since a captured
  gather's LUTs pin the layout. It RUNS under the #462 BREAKABLE route:
  `prepare_breakable` calls `_observe_routing` (`expert_offload.py:3107`) whose
  tail is `if self._heat.due(): self._migrate_heat()` (`:3197`), and
  `_capturable_ready` is False there — sound by construction (in-place swaps
  into address-stable arena slots, slot vector republished per replay), so §17's
  cell "#302a x graphs" is OCCUPIED. Same for Stage-1 `SGLANG_MOE_HOT_RESIDENCY`
  (`layer.py:702` gates on the capturable mode only). Note both refusals read
  only the legacy env, so `SGLANG_MOE_OFFLOAD_GRAPH_MODE=capturable` escapes
  them (#500-B9). Counters land in the #390 dump under `heat_migration` / `heat_*`
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
- **HiCache** L1-L3 prefix cache. Uneven DCP/TP is not a gated combination —
  there is NO refusal anywhere; the controller BRANCHES on it
  (`dcp_owner_mode = self._dcp_owner_ctx() is not None`,
  `managers/cache_controller.py:655`), which drops the rank suffix from the
  storage key (`mem_cache/hicache_storage.py:427-430`) so a token-sharded rank
  set shares one entry. Storage key includes kv-dtype; runtime attach/detach
  works on UnifiedRadixCache. The
  L2 host tier's `page_first_direct` transfer path was blocked on this rig by
  a segfault in `transfer_kv_all_layer_direct_lf_pf` (#436, cu12/cu13
  `cudaMemcpyBatchAsync` ABI split); unblocked by the cu13 `sgl_kernel`
  rebuild.
- **KV session offload (kvso)**: FCFS spill of youngest sessions to RAM (KV
  only — `bundle_spillable_sizes` returns `[("kv", kv)]` and nothing else,
  `managers/kv_session_offload.py:126`, so GDN stays resident), budgets
  (volume/rate/window, demote to HiCache), victim key
  `(spill_class_rank, fast_lane, -kv_arrival_seq)` — spill class, then
  non-fast-lane, then YOUNGEST arrival; there is NO idleness term (`:844`).
  It is NOT decoupled from speculation: `if self.speculative_algorithm is not
  None and os.environ.get("KVSO_ALLOW_SPEC", "0") != "1": raise ValueError`
  (`server_args.py:6580`), so with the standing NEXTN recipe the pair needs an
  explicit opt-in. #500-B10 verdict: the refusal is DELIBERATE, not stale —
  the mechanism exists (a spilled session decodes under MTP/NEXTN, the
  draft-KV share spills and restores with it, `--kv-session-offload-spec-in-tick`
  runs the drafter in the tick), and the gate stays for one named unobserved
  round: a spill landing in the same round as a drafter-in-tick step. Both
  facts are now IN the refusal text, and `KVSO_ALLOW_SPEC` is named in
  `--enable-kv-session-offload`'s CLI help (it appeared in no help text
  before). Also mutually exclusive with
  `--enable-hierarchical-cache` (:6620), PD disagg, `--weightless-kv-fastlane`,
  `--enable-unified-memory`, `--enable-hisparse`, `--enable-mixed-chunk`,
  `page_size > 1`, non-flashinfer backends, `pp_size > 1`, `dp_size > 1`
  (:6596-6641).
- **Hibernate to disk** (weights + module buffers survive process exit — NO KV
  is parked: `payload = {... "params", "static_state", "gguf_attrs" ...}`,
  `model_loader/hibernate.py:448-456`; uneven-TP3 reload 50s→8-14s, the uneven
  vector being part of the identity hash, `:96`). Scope, all hard refusals:
  GGUF checkpoints only (`if self.load_format != "gguf": raise`,
  `server_args.py:13204`), pure single-node TP (`hibernate.py:106-123`), and a
  per-rank NVML-UUID recheck on restore (`if live_uuid !=
  rank_meta["nvml_uuid"]: raise RuntimeError`, `:568`).
  `--enable-weights-disk-backup` and `--hibernate-dir` require each other in
  both directions (`server_args.py:13191-13199`).
  Plus suspend-to-RAM (memory saver; reaches the legacy hybrid-SWA `SWAKVPool`
  since upstream #32213 — `kwargs.setdefault("enable_memory_saver", False)`,
  `mem_cache/swa_memory_pool.py:54`, both call sites passing the server arg,
  `model_runner_kv_cache_mixin.py:3311`/`:3491` — before that it was silently a
  no-op there. `UnifiedSWAKVPool` honours it only INDIRECTLY: its own
  `enable_memory_saver` parameter (`mem_cache/unified_memory_pool.py:1014`) is
  never referenced; the saving happens in the shared `UnifiedKVPool` buffer,
  `:212-216`).
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
- **Runtime VRAM dial** per rank (VMM page return) — requires WEIGHTED uneven
  DCP (`if not uneven_dcp_active(dcp_size): raise KvCapacityError`,
  `managers/vram_dial.py:1110`), CUDA and a non-MLA VMM-backed pool, and refuses
  under 11 named combinations incl. memory-saver, PD disagg, hicache storage,
  kvso, dual-group, DP>1, kv-canary, hisparse, weightless-KV fastlane and the
  DFLASH lane (`:1045-1086`); its HTTP actuator is `POST /vram_budget`
  (`http_server.py:1149`). **KV pressure ladder**
  (geometry stages instead of rejects; explicit ladders work; rung-dependency
  refusals exist and fire — for EXPLICIT specs only, `if isinstance(spec,
  tuple):`, `server_args.py:6955`). `--kv-pressure-ladder auto` resolves to a
  real table at this tip (`ladder = build_ladder_from_server_args(server_args,
  table_fn=auto_ladder_table_fn(server_args))`,
  `managers/kv_pressure_runtime.py:467`), which closes AUDIT_421 F1; on a
  heterogeneous node it REQUIRES `--rank-gpu-id` (`if len(names) > 1: raise
  ValueError`, `managers/kv_ladder_auto.py:190`). Boot validation pending — the
  table is
  computed from the rig profile by the #272 planner, rank-uniformly and
  UUID-keyed, and inventories only rungs whose actuator this configuration
  wires. Capacities are labelled placeholders until the measured figures
  arrive. BOOT-PENDING: `scripts/dev/428_boot_checks/`. **KV resharding**
  at phase boundaries (delta move <1 s, `kv_reshard_vectors`) — also requires
  WEIGHTED uneven DCP and the hybrid-linear pool family only
  (`managers/kv_reshard.py:713-729`), **GDN slot ladder** (resident-state cap +
  idle vacate → VRAM back to KV pool; its flags validate at boot but are INERT
  without `SGLANG_OFFLOAD_REGISTER=1` — `if not offload_register_enabled():
  return []`, `model_executor/offload_gdn_states.py:344`, #500-B11).
  `--lane-offload-profile/-class-policy/-park-targets` are wired at runner
  init once-per-process (#428), boot validation pending; a typo now refuses
  there too. The park chain reaches the register and the movement layer's
  default reads it — but nothing in
  production constructs the movement backend yet, so the chain has a consumer
  PATH, not a consumer. `SGLANG_OFFLOAD_REGISTER=1` (dark launch) gates the
  process-global register ONLY (`if not offload_register_enabled(): return
  None`, `offload_register.py:1227`, `:1261`) — `get_global_register` and the
  `maybe_*` adapters. The `--lane-offload-*` parsers run at argument time
  regardless (`server_args.py:6663-6668`), and the #286 asset-class layer
  (`short_term_offload_register.py`) is not behind it at all: it is already
  called in production from `breakable_offload.py:216`/`:268-272`. The
  runner-init typo refusal (`offload_register.py:1239`) sits BEHIND the gate,
  so by default a typo refuses only at argument time.
  BOOT-PENDING: `scripts/dev/428_boot_checks/`.
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
  provenance `measured|estimate|absent`. ABSENT refuses on the BANDWIDTH axis
  only, and only under a floor or `allow_unmeasured_bandwidth=False`
  (`registry.py:407`, `:418`); absent LATENCY refuses nowhere; absent CAPACITY
  refuses only when bytes are requested (`:455`, `:458`). "Estimates refused by
  default" is the CALLER's default, not the registry's:
  `require_measured_bandwidth: bool = False` (`registry.py:199`) vs
  `price_park_target(..., require_measured: bool = True)`
  (`short_term_offload_register.py:787`). HONEST STATE
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
  licenses (`fingerprint.py:313` EXACT = every tier; `:325` MODEL = card
  templates only, no host / filesystem / remote row — the whole `tiers` list is
  dropped at `:416-419`; `:339` NONE), unless `SGLANG_MEMTIER_PROFILE_TRUST=1`
  promotes an explicit path's NONE verdict to full EXACT
  (`profile_store.py:208-213`), and otherwise bootstraps from live facts with
  measured sizes and every cost ABSENT naming its probe. `from_profile()` no
  longer defaults to the bundled rig profile — that default handed one
  development box's host RAM, ZFS pool and 40G peer to every machine.
  Measurements are ingested from the EXISTING artifacts (`card_probe` #213,
  rig artifact #271, `capability_matrix` #278) by `memtier/adapters.py`;
  #407 adds no probe of its own. `TierTransport.link_path` +
  `link_disjointness()` expose PATH identity for #423's striping gate, with
  `DISJOINT` requiring complete paths on both sides (`tiers.py:570-603`).
  `UNKNOWN` is RETURNED, not refused — no `raise`, and no production caller;
  with `link_path_complete=False` hardcoded in bootstrap
  (`memtier/bootstrap.py:241`) and in the bundled remote rows, DISJOINT is
  currently unreachable from real data (#500-B12).
  Design: `docs/dev/DESIGN_407_memtier_registry.md`.

## 4. Speculative decoding
NEXTN/MTP standard (steps 3, topk 1, draft 4); adaptive draft length (upstream
base, fork adds graph-offload, high-accept ladder, frozen-MTP, hetero
determinism); acceptance-driven DFLASH<->NEXTN switch + adaptive k (DFLASH only
— DSPARK is not a cross-algo rung, `cross_algo_utils.py:686`) — and in that
mode the DFLASH rung's solo host is PINNED TO RANK 0 with
`--speculative-draft-gpu` refused (`cross_algo_utils.py:733-739`), so "DFLASH
solo draft on the big card (vocab broadcast reclaims ~5 GB)" holds for plain
`--speculative-draft-placement solo`, not for the cross-algo ladder;
chain-spec on the weightless lane; multi-layer EAGLE fixes; spec-algo name
validation (one source, parse-time refusal); canonical
`--speculative-draft-model-path`.
Tree-spec topk>1 is HARD-GATED on the CROSS-RANK DCP VARIANTS ONLY — the gate
is `(rank_tp_ratio is not None or weightless_kv_fastlane) and
tree_verify_activation_reason() is not None` (`server_args.py:7407`, mirroring
`flashinfer_backend.py:931`'s `dcp_tree_mask`), reached only at `dcp_size > 1`
(`:7590`). It fires for a UNIFORM `--rank-tp-ratio` too, and has no `page_size`
term. DCP without a `--rank-tp-ratio` vector and without the weightless lane is
NOT gated: there `uneven_dcp` is False and topk>1 runs the stock correct EAGLE
tree path (live on HIP; on CUDA the separate spec×DCP gate at :7644 shuts that
door for its own reasons). Silently wrong + perf-negative under the gated
variants — do not re-attempt without new evidence; see rejected register.

**Per-decode KV reserve is derived, not a blanket 2x (#486).** The reserve
every running request holds ahead of `kv_committed_len` is `W + L`: the write
footprint of this step's draft+verify (`get_alloc_len_per_decode`) plus what
one in-flight verify can still commit while the host is one step behind
(`get_commit_lag_per_decode` = `max_speculative_num_draft_tokens` under
overlap, 0 with `--disable-overlap-schedule`). It is now a NAMED posten in the
pool ledger (`DESIGN_330_vram_dial.md` §3b) instead of an uncounted transient —
EXCEPT the hybrid-SWA / SWA-chunk-cap pool sizer, which still computes
`2 * get_alloc_len_per_decode(sa)` (`pool_configurator.py:628`) and therefore
disagrees with the allocator on every non-overlap and every topk>1/page>1 run.
Open (#500-B4).
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

Draft-solo placement admits `algo.is_eagle() or algo.is_dflash_family()`
(`server_args.py:7147`) — an ENUM MEMBERSHIP test, not a shape test:
EAGLE / EAGLE3 / NEXTN / DFLASH / DSPARK. A plugin algorithm registered via
`SpeculativeAlgorithm.register` is refused however DFLASH-shaped it is.
Solo also needs `tp_size >= 2` (:7282), refuses PD disaggregation (:7275) and
DP/PP/EP (:7196/:7203/:7209), but is NO LONGER single-node: only
`--speculative-draft-gpu` is refused across hosts (:7239, RELAXED-NOT-PROVEN).
DSPARK joined the family (#470) because it has the same shape — self-drafting
block model, token-id round output, post-all-reduce hidden-state input — and
its one delta, the
confidence head's per-request block truncation, rides the SAME per-round
broadcast as one extra integer per request (`dspark_components/dspark_solo.py`
packs ids + lengths into one int64 tensor, so the round still costs one
collective). `FROZEN_KV_MTP` stays refused — by NAME, one branch earlier
(`server_args.py:7111`), pinned by `test_draft_solo_args.py:168`: its draft
reads the target KV in place, which no single rank holds. Solo DSpark v1 is
greedy-acceptance-only (a non-greedy round would need `[bs, gamma, vocab]`
corrected logits on every verifying rank) and switches the default-on
`SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD` off, with reasons logged rather than
inferred. `--speculative-moe-runner-backend` reaches EVERY draft build — the
`speculative_moe_backend_context()` wrapper is applied in `eagle_worker_v2`,
`standalone_worker_v2`, `frozen_kv_mtp_worker_v2`, `multi_layer_eagle_worker_v2`,
`cross_algo_worker` and (new for #470) the shared DFLASH/DSPARK builder
`draft_worker_common.py:155`; unset, it defaults to `--moe-runner-backend`
(`overrides.py:2086`). Reaching the DFLASH/DSPARK builds is what puts an MXFP4
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
high-priority stream, lend/reclaim in ms, SM-contention pairing rule
(`--dual-group-lane-pairing` needs `--dual-group-lane-concurrent`,
`server_args.py:9462`), lane-NEXTN head. Admission: an EXPLICIT
`--rank-tp-ratio` integer list is mandatory (`'auto'` refused,
`server_args.py:9441`) — a UNIFORM vector is accepted, so the lane is reachable
on a homogeneous rig — plus `--dual-group-lane-budget-mib` (:9449). Only
`pp_size>1` and `--enable-dp-attention` are refused (:9455); `ep_size>1` and
plain `dp_size>1` are NOT. Lane spec chain merged: rank-local draft-KV sizing,
chain-spec topk=1 — the gate reads the SERVING GROUP's
`--speculative-eagle-topk` (`not in (None, 1)`, `server_args.py:9482`), so
enabling `--dual-group-lane-spec` also forbids a tree on the main group, and
the lane's own proposer has no topk knob at all (`dual_group_lane.py:2488`);
`--dual-group-lane-spec` additionally requires the serving group to speculate
(:9473) — lane prefill chunking (`dual_group_lane_prefill_chunk`; spec chunks
carry the head primer — costs measured), Marlin LoRA workspace keyed
(lane,name). PD disaggregation: prefill satellite carries hybrid GDN (KV+mamba
slot via mooncake, gated on `tree_cache.supports_mamba()`,
`disaggregation/prefill.py:911`). Graph coverage: a prefill satellite disables
only the DECODE graph (`server_args.py:8190`) and keeps the default BREAKABLE
prefill graph — but the breakable-compat sweep drops that prefill graph for
`attn_cp_size > 1` and for MLA attention (`server_args.py:8278-8288`), i.e.
every DCP plan runs the prefill satellite eager.

## 6. Weightless KV lane
A card holds ONLY KV + attention (no weights): chunked prefill/extend, fp8/int4
worker KV, DCP comm fusion, graph-captured streaming decode, host-tier KV
spill, chain spec — but SPEC AND THE STREAMING BLOCK LOOP DO NOT COMBINE:
`--weightless-kv-chunked-block-size` / `--weightless-kv-host-spill-tokens` are
refused together with a speculative algorithm (`server_args.py:6330`), two
capture axes nobody composed. Chain spec further requires the EAGLE family
(:6274), `--speculative-draft-placement solo` (:6288) with solo rank ==
`--weightless-kv-head-rank` (:6304), and no `--speculative-adaptive` (:6316).
Lane topology: `dcp_size == tp_size >= 2`, no PP/DP-attn/EP (:6069-6088).
**Live session handover without server stop** + draft
re-sharder as its own spec type: MERGED and GPU-gate-passed
(`POST /session_handover`, five-phase at session scope, hard GDN-blob gate
keyed on `BasePrefixCache.supports_mamba()`; proven byte-identical to a
never-moved reference via a real cached-tokens import, `cached_tokens=1152`
on resume, plus seven named-refusal negative controls). Endpoint at this tip:
`http_server.py:1126`. v1 limits, AS ENFORCED: live EXPORT requires a
TP=1 / PP=1 SOURCE (`session_handover.py:418`), `page_size == 1` (:425,
"inherited from `dcp_owner_mode`") and the `file` hicache storage backend
(:392). The DESTINATION carries NO `tp_size` predicate — `verify_import`
checks manifest version, model identity and blob presence only (:266-292); a
booted TP>1 destination simply misses the blobs and is told to run the offline
manifest-scoped umsharder for its geometry first, live handover does not do
that reshape in-process.

Also wired on the tip but easy to miss (audit #421): the regime-controller
gate machinery, KV-pressure rung-dependency refusals, the hibernate flag
contract (`hibernate_dir` + weights/draft CPU/disk backup flags), and a
118-name retired-env guard that refuses stale SGLANG_* variables loudly.

## 7. Collectives / transport
**barlink** (own vendor-neutral CCL): NCCL-parity device transport,
cross-vendor byte-exact, UCX transport (chunk pipelining, dual worker), tuned
all_gather ring, graph-capable direct mode — capture-safety is
TRANSPORT-NAME-keyed, not property-keyed:
`CAPTURABLE_BARLINK_TRANSPORTS = {"device","host"}` plus
`GRAPH_ENABLE_TRANSPORTS = {"bar1","matrix"}` through
`SGLANG_BARLINK_GRAPH_ENABLE` (default on) (`parallel_state.py:298`, `:303`,
`:352-362`); ucx/shm/gloo are refused at startup unless `--disable-cuda-graph`
(`:365-383`), and under an active capture there is no silent gloo fallback —
barlink aborts with the reason (`barlink.py:635-676`).
Op coverage per transport, from `BARLINK_OPS` at source: device
`{all_reduce, all_gather, reduce_scatter, broadcast}`
(`barlink_device.py:1152`); host the same plus send/recv
(`barlink_host.py:811`); ucx the same four (`barlink_ucx.py:376`); bar1
`{all_reduce, all_gather, all_to_all, all_to_all_single, broadcast}` — NO
reduce_scatter (`barlink_bar1.py:1450`); matrix only
`{all_reduce, all_to_all, all_to_all_single}`
(`barlink_matrix_transport.py:354`), a strict SUBSET of its own bar1 sub-path
(#500-B17). The communicator refuses four collectives outright:
`reduce_scatter(list)`, `reduce_scatterv`, `all_gather(output_tensor_list=)`,
`all_gatherv` (`parallel_state.py:1348-1371`); bar1 additionally caps the group
at 8 ranks (`MAX_RANGE`, `barlink_bar1.py:811`, `:1518`). **Smallbar BAR1 direct path**:
peer VRAM over the card's own BAR aperture — PROBED, not assumed (NVML
`bar1Free`, sysfs gross fallback, `barlink_matrix_transport.py:280-302`).
Requested window `SGLANG_BARLINK_BAR1_WINDOW_MIB` (default 96) with a per-group
override `..._MIB_<GROUP>` (`:113-120`); the group-wide MINIMUM governs
(`barlink_bar1.py:1953-1985`). Payload eligibility is checked against the
CONTIGUOUSLY mapped length, never the sysfs gross size
(`barlink_bar1.py:2385-2405`) — a larger BAR raises reachability directly.
Measured 1.13-1.34x over NCCL in serving (measurement, not a gate).
Undocumented sizing knobs on the same plane: `SGLANG_BARLINK_SLOT_MIB` (64,
`barlink.py:70`), `_HOST_SLOT_MIB` (`barlink_host.py:120`), `_CHUNK_MIB` (8,
`barlink.py:52`), `_PIPE_CHUNK_MIB` (4, `barlink_device.py:989`).
`--collective-net-small/-bulk` pin the NIC (not the transport) per message
class; typo hard-reject against sysfs, not a name list
(`server_args.py:14089-14098`, accepted set = dirs under
`/sys/class/infiniband` + `/sys/class/net`, plus `all`, optional `:port`). SMALL
reaches only the barlink UCX plane and pins BOTH small and large TP collectives
(one UCX context, `server_args.py:14176-14185`); BULK reaches PD-KV/HiCache via
`--disaggregation-ib-device`. BAR1 is not selectable here.
dma-buf EXPORT works on consumer cards with the stock driver — probed
(`cuMemGetHandleForAddressRange` first, `NV_ESC_EXPORT_TO_DMABUF_FD` ext as
fallback, `barlink_bar1.py:517-537`). The BAR1 PEER MAPPING on top of it is NOT
stock: it needs the widened driver guard (regkey
`BarlinkPeerBar1`/`RMSmallBarP2PPeerBar1`, `barlink_bar1.py:597`, `:2337-2342`),
the `dmabuf_holder` module (`:589`, `:644-653`) and a passing byte proof
(`:4644-4656`); `CAP_SYS_ADMIN` or `PeerMappingOverride=1` is the second hurdle
in a container (`:2352-2374`). Rig facts: NO
P2P/NVLink here, negotiated PCIe x4/x8/x8 (NVML max-width reports x16
NAMEPLATE — always read negotiated width), NCCL-verbs broken on our RoCE.
**Collective-decision recorder** (`barlink_uniformity.py`, #431): per-rank
ordered log of every `(op, nbytes, path, rounds)` dispatch decision plus a
pure `first_divergence` comparator — the standing instrument for the
rank-local-condition-before-a-group-collective family (#94/#194/#312/#431).
Off by default (`SGLANG_BARLINK_RECORD_DECISIONS=1`, read ONCE at import —
`barlink_uniformity.py:205` — so it must be exported before the process starts;
`SGLANG_BARLINK_RECORD_DUMP_DIR` adds the per-rank on-disk dump for post-mortems
on a wedged run but does nothing on its own, because recorders are only built
from `record_decision`, which returns early when recording is off, `:250`
— #500-B20).
**#279 path dispatcher**: flag-gated (`SGLANG_BARLINK_PATH_DISPATCHER=1`, read at
call time, `barlink_path_dispatcher.py:428`) and inert — a fresh dispatcher has
an EMPTY registry, so every decision is the status-quo #240 choice (`:431-443`).
`PathProfile.saturation_threshold` is permanently 1.0 (no writer) and no
production code attaches a saturation sensor, so `_utilization_locked` returns
0.0 and the `>= threshold` overflow tier at `:357` never fires today. It is not
dead code: the one production-intended sensor, `bus_saturation_sensor`, is
BINARY (`return 1.0 if stats.get("pending_demand") else 0.0`, `:415`), for which
threshold 1.0 is exactly right. AUDIT_421 §8's open question is CLOSED —
reachable by construction, correctly matched to its intended sensor, unreachable
until #279's measured slice wires both. **Scoped slow-boot warning**: barlink BAR1 × uneven weighted DCP × an
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
**Guard cost, and where it lands** (#476 measured, #517 named). The blocking
read cost -9.22 % of code decode_TPS against the same-tree NCCL baseline;
removing both seams gives +2.68 %, reproducing #424's pre-#431 BAR1 advantage.
Split: replay boundary 5.26 pp, host-path collectives **6.64 pp**. A DECODE
round is not free of host-path collectives — the model that said so is wrong:
per NEXTN round there are **3 replay boundaries** (the draft chain is ONE
captured graph, `eagle_draft_cuda_graph_runner.py:458` +
`eagle_worker_v2.py:1027-1030`, not one per step) and **5 host-path BAR1
broadcasts**, all of them the #50 speculative rank-agreement syncs: 3 in
`eagle_sample` (`eagle_utils.py:1149`, after the target-verify replay,
`eagle_worker_v2.py:2680`) and 2 in `_draft_extend_for_decode`
(`eagle_worker_v2.py:1583`, whose own comment says "runs every decode
iteration, outside any cuda graph"). They reach barlink because a barlink boot
does not CONSTRUCT pynccl (`parallel_state.py:440`, `:778-781`), so
`capture_safe_tp_broadcast`'s pynccl branch is dead and `spec_utils.py:138`
takes `tp_group.broadcast`; a BAR1 broadcast is issued as an `all_to_all`
(`barlink_bar1.py:3648-3651`), which is what the #476 §3 crash line
("all_to_all (8 bytes, 0 rounds)") names.
**Staged abort read** (#517): the status word is read asynchronously — a
non-blocking D2H onto the current stream plus a `cudaEventQuery`, returning
the value an earlier check staged — so a check costs no stream
synchronization. `ctlStatus` is sticky (only `*A.ctlStatus = 1u` in the ext,
only `torch.zeros` at bring-up), so deferral trades reporting LATENCY, never
detection, and `SGLANG_BARLINK_BAR1_ABORT_MAX_LAG` (default 4) forces one wait
after that many unresolved checks — the bound is what keeps it a guard, and it
has a can-fail proof (a never-ready event passes 200 boundaries over a tripped
word once the bound is raised out of the way).
`..._ABORT_DEFER=0` restores the pre-#517 blocking `.item()` exactly. A CPU
status word never defers (`barlink_abort_gate.should_defer_status`), so no
hermetic test changes meaning. `..._CHECK_EVERY` now also reaches the replay
boundary (before #517 a boundary was entered with `_unchecked_launches == 0`,
below the interval test, so the documented knob throttled Seam A only); K=1,
the default, is behaviour-identical. **Unmeasured: the benefit.** The saving
is a count of syncs (8 per decode round to 0 in the steady state), desk-only;
no arm has run. See `docs/dev/NOTE_517_bar1_guard_desk.md` for the named
collectives, the corrected round model and the GPU-window ticket.

## 8. GGUF stack
Generalized loader (registry + family mapping tables), unsloth-UD, mixed-dtype
fused GDN qkvz, MoE tensor mapping, vision/mmproj, sibling-config validation,
DeepSeek-V2/3/4 class GGUF-safe (`.qweight` accessors, quantization_config
drop, tokenizer route). Perf: batched MMVQ (default follows the WHEEL probe `_dequant_supports_out`,
`gguf.py:342-345`), Q8 lm_head, K-quant MMVQ tuned to Q8_0 efficiency (TP=2
beats llama.cpp; wheel probe `ggml_mmvq_kq_tuned` + `SGLANG_GGUF_KQ_KERNEL` kill
switch, `gguf.py:355-371` — when present it fully disables the #72 reroute,
`:442`), graph-replay numeric safety for ALL quants — literally type-agnostic
(`gguf.py:942-948`), but it needs the `ggml_dequantize(..., out=)` wheel schema;
on an older wheel the capture OOM returns. `gguf_mmq_decode_threshold` compares
the CUDA-graph decode BUCKET (raw M rounded UP) against a MEASURED
per-(capability, shape class) table `_MMQ_BUCKET_MIN` = {sm120, sm86} only
(`gguf.py:539-542`, `:665-672`) — silently inert on any other device
(#500-B19).
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
presence is a wheel property, probed via the `ggml_mxfp4_native` marker op (the
#73 pattern, `gguf.py:272`) and evaluated ONCE at import (`:277`).
`SGLANG_GGUF_MXFP4_NATIVE=0` hands the checkpoint back to the repack —
first-character test (`:265`), so `false`/`no`/`off` do NOT disable it. The
"no-op on a native wheel" is a short-circuit, not a cheap pass: `_type_map()`
returns `{}` before any tensor is read (`gguf_mxfp4_repack.py:113-115`). Second,
undocumented lever: `SGLANG_GGUF_MXFP4_REPACK=0` (default 1, `environ.py:1776`)
empties the same map (`:122-124`); combined with `NATIVE=0` or an old wheel it
turns the checkpoint into a loud load-time refusal by tensor name (`:127-135`) —
never a silent fallback. Native also widens MoE expert-offload coverage, since
`MOE_OFFLOAD_SUPPORTED_TYPES = MMVQ_QUANT_TYPES` (`gguf.py:292`). GPU-pending:
`TICKET_398_mxfp4_validation.md`.
**#479 traced the served checkpoint and found no untraced fallback.** The
active UD-IQ3_XXS driver carries exactly two type-39 tensors,
`blk.26`/`blk.42.ffn_down_exps` (2.125 GiB), and their gate/up siblings are
IQ3_S resp. IQ3_XXS — so those two layers are the ONLY mixed type pair
(`qweight_type != qweight_type2`) `fused_moe_gguf` ever sees, and nothing
tested that cell. Both arms take the same exit, the MoE MMVQ branch
(`gguf.py:1093`): native passes 39 straight into `ggml_moe_a8_vec`, the repack
arm passes 6. The MMQ branch is unreachable for these layers at ANY batch size
because the w13 side is an imatrix type with no MMQ kernel, and the slow
per-expert loop refuses an unknown type by name (`:958-963`) rather than
computing — so there is no silent path, only a priced one. The load never
materialises floats: with the kernels native both repack entry points are the
identity (`gguf_mxfp4_repack.py:113-115`, `:205-207`, `:224-226`), uint8 blocks
at 17 B/32 values, which settles TICKET_398 §2's open question — the native
delta is EXACTLY the repack's 0.625 GiB, nothing on top. Offload admission
checks BOTH expert tensors, not just w13 (`fused_moe_triton/layer.py:2520-2534`).
`NOTE_479_mxfp4_active_driver_path.md`;
`test/registered/unit/quantization/test_gguf_mxfp4_dsv4f_moe_479.py`, 8
hermetic, three executed can-fail arms.

## 9. Quant lanes
FP8 (sm120 GEMM tuned; per-channel fused GEMV; opt-in deterministic
`SGLANG_DETERMINISTIC_FP8_GEMM` — reaches dense fp8 linears and
`CompressedTensorsW8A16Fp8` on NVIDIA sm80..sm88 ONLY (`fp8_utils.py:313`,
`:321-323`), a no-op on sm89/90/120 and on ROCm, and deliberately NOT honoured
by fp8 MoE experts (`fp8.py:1245-1252`), FBGEMM fp8 (`fpgemm_fp8.py:56-67`) or
multimodal_gen, each of which logs the gap; it forces
`fp8_needs_dequant_fallback` on (`fp8_utils.py:361`), costing ~2.5-6x decode
throughput; e4m3 KV bit-exact on sm86 — a measurement, no sm86 code gate), INT8-W8A8 (default
recommendation; sm86-native lane; beware the dual-dist wheel trap — pin by
sha256), NVFP4 — "unpackable" is a pure SHAPE test, not a checkpoint class: Marlin
`output_size % 64` on the UNSHARDED width
(`compressed_tensors_w4a4_nvfp4.py:152-157`), native FP4 `width % 32` on the
SHARDED width (`:186-192`), routed per rank by the resolved lane (`:225-234`);
both verdicts land on `CompressedTensorsW4A4Fp4Dequant` (load packed,
materialise dense once, `F.linear`, exact numerics), so ANY NVFP4 checkpoint at
ANY `--rank-tp-ratio` is bootable, including layers no TP split can rescue,
Marlin alignment family (EIGHT sibling bugs fixed — device-free fold predicate,
lcm=128 on coupled dims; alignment fixes must preserve cross-layer agreement).
The eighth (#444b) is MXFP8: its `weight_block_size [1, 32]` is the OCP scale
layout, not an alignment registration, so an asymmetric exposed block is
coarsened by `lcm` of both axes before the marlin fold. The predicate is
STRUCTURAL — `asymmetric_block = bool(raw) and len(raw) == 2 and raw[0] !=
raw[1]` (`linear.py:289-291`) — so it covers every future asymmetric exposure,
not just MXFP8, and symmetric blocks are provably untouched. It is NOT latent:
the mxfp8 floor is `capability >= 100` (`loader.py:261` against
`fp8.py:315`), so sm120 (the 5090) CLEARS it; on ROCm the same config floors at
95 (gfx95) or 94 (gfx942 block-fp8 conversion) (`fp8.py:305-313`). The fold's family
predicate is a DECLARATION, not a name list (#500-B18): each backend sets
`QuantizationConfig.marlin_packable_linear` where its kernels are known, and
`_marlin_packable_family` reads it (`linear.py:209`). It previously carried
`_MARLIN_PACKABLE_CONFIGS = ("fp8config", "compressedtensorsconfig",
"fbgemmfp8config")` — the §12 quant-name-list family (#443/#446), second
instance. Reach of the fix, stated at the width of what was checked: of the
three configs the audit named, only `MarlinConfig` is genuinely marlin-served
and blockless, and it is neither in `QUANTIZATION_METHODS` nor concrete (no
`get_scaled_act_names`), so it is a LATENT hole, not a live boot failure;
`W8A8Fp8Config` and `QuarkConfig` reach no marlin repack entry point at all
(`apply_fp8_linear` does not repack, and `quark/` contains no marlin
reference), so they are correctly out. Every REGISTERED backend whose module
reaches a marlin repack either declares the capability or exposes a
`weight_block_size` — asserted per method in
`test_marlin_unit_coarsening.py::test_every_registered_marlin_repacking_backend_is_covered`.
**auto-round GPTQ MoE could never reach Marlin** (port of upstream
`00cdd4b85f`, PR #33271): `apply_gptq_quant_layer` delegates a FusedMoE layer
to `MoeWNA16Config`, which re-runs the eligibility check itself
(`moe_wna16.py:98`) — but the config it was handed omitted `desc_act`, and
`is_gptq_marlin_compatible` treats a MISSING key as ineligible (`if num_bits is
None or group_size is None or sym is None or desc_act is None: return False`,
`gptq/gptq.py:575`), so the delegation always landed on the Triton runner while
the same checkpoint's DENSE layers ran on Marlin. The `use_marlin=False` arm
was additionally a latent `NameError` (it built `GPTQMarlinMoEMethod` from a
variable bound only in the other branch), so neither arm worked. Now one
delegation with a complete config; auto-round has no act-order concept, so
`desc_act` is always False. Reach: auto-round checkpoints only — plain GPTQ
takes its own path. Tests:
`test/registered/unit/layers/quantization/test_auto_round_moe_delegation.py`,
7 hermetic, two executed can-fail arms (`desc_act` dropped → 3 red; the
pre-port split restored → 4 red). Unmeasured — no auto-round MoE boot exists on
this rig.
**The DSV4 `wq_a`+`wkv` fusion is gated on the built leaf inventory, not on a
quant-method name** (#526, own find; the same symptom class as the open upstream
issue #33245, which is unfixed there). `SGLANG_OPT_FUSE_WQA_WKV` defaults to
True (`environ.py:1733`) and joins exactly `_WQKV_A_LEAVES` = `weight`,
`weight_scale_inv`, `qweight`, `qweight_type` (`deepseek_v4.py:3330`), all by
`torch.cat(..., dim=0)` — the output-row axis for a dense or GGUF payload. A
packed integer format builds the linear from OTHER leaves, and both halves went
wrong quietly: GPTQ/auto-round `qzeros`/`scales`/`g_idx` and AWQ
`qzeros`/`scales` matched no fusion predicate and fell through to the
unmatched-name branch, which on a non-GGUF checkpoint only warns and continues
(`deepseek_v4.py:3147-3148`) — dropped tensors, fused parameter left
uninitialised; and `qweight` is not row-major there (GPTQ packs the INPUT dim
into dim 0, `(in//pack, out)`, so the concat joined shards along the pack
axis). A compressed-tensors packed checkpoint (`weight_packed`/`weight_scale`/
`weight_shape`/`weight_zero_point`) matched NOTHING at all, so the load ran to
completion having delivered zero tensors. Reach is wider than "packed integer":
per-tensor fp8 (`weight_scale` + `input_scale`) has the same hole. The gate is
`_unroutable_wqkv_a_leaves` (`deepseek_v4.py:3336`) reading
`named_parameters()` off the module the quant method ACTUALLY built — the §12
declaration-not-name-list shape, third instance — consumed in two places:
`_wqkv_a_fusion_survives_quant_format` (`:3388`) turns an inherited default-on
fusion OFF at construction with a named notice and REFUSES an explicit opt-in,
and `load_weights` (`:2781-2805`) derives `fuse_wqa_wkv` from the built topology
instead of re-reading the env (they diverge the moment the auto-off fires) plus
keeps a refusal backstop placed BEFORE the non-GGUF `list(weights)` drain.
Routable and byte-unchanged: bf16 (`weight`), fp8-block (`weight` +
`weight_scale_inv`), GGUF (`qweight` + `qweight_type`) — each asserted against
the real quant method. This protects the ANALYSE_463 R3 route (GPTQ-INT4 requant
of the DSpark experts) before it is built. Tests:
`test/registered/unit/model_loader/test_dsv4_wqkv_a_packed_formats_526.py`,
20 hermetic, three independently executed can-fail arms (leaf gate neutered →
8 red; env-driven fuse flag restored → 3 red; construction-time helper neutered
→ 3 red). Unmeasured — no packed-format DSV4 checkpoint exists on this rig yet.
**The same fusion also cuts a SCALE BLOCK, not a row** (#528, follow-on find
from the #526 review, where the axis was deliberately left byte-identical):
`weight_scale_inv` is routable and is joined by the same
`torch.cat(..., dim=0)` as the weight, but for a block-quantized checkpoint dim
0 of the scale governs `weight_block_size[0]` WEIGHT rows. The join is exact
only while `wq_a`'s width is a whole number of blocks; otherwise fused scale
block `q_lora_rank // b` spans wq_a's tail AND wkv's first rows under wq_a's
scale and every later block is shifted — and for the DSV4 shape
(`kv_rows = head_dim`, a multiple of 128) `ceil(q/b) + ceil(kv/b)` still equals
`ceil((q+kv)/b)`, so the wrong scales are simply copied in with no shape error.
REACH TODAY IS ZERO, established before the fix: `q_lora_rank` is 1024 in both
the config class defaults (`configs/deepseek_v4.py:78`) and the DSpark export
(`config.json`, `weight_block_size [128, 128]`), and wqkv_a is a
`ReplicatedLinear`, so no `--rank-tp-ratio` shard can move the cut. The #444b
eighth alignment sibling CROSSES this fusion — MXFP8 builds exactly
`weight`+`weight_scale_inv`, so the #526 gate lets it fuse — and is provably
immune: its `[1, 32]` dim-0 block IS a single row, so the two `cat`s are the
same axis at any split. Shipped as a REFUSAL rather than a block-aware concat,
because fused block `q // b` is fed by two independently quantized tensors and
no single scale row describes it — repairing the join would mean requantizing
the seam, i.e. changing the checkpoint's numbers. Two consumers of one
predicate `_misaligned_scale_block_axis` (`deepseek_v4.py:3414`): the
construction-time gate gained an optional `q_rows=` (`:3507`, the call site
passes `self.q_lora_rank` at `:499`) and turns an inherited fusion off / refuses
an explicit opt-in, and `load_weights` keeps a config-driven backstop next to
the #526 one, before the stream is touched (`:2824`). Tests:
`test/registered/unit/model_loader/test_dsv4_scale_inv_block_axis_528.py`,
21 hermetic, two executed can-fail arms (predicate neutered → 5 red; the
`q_rows=` argument dropped from the call site → 1 red), including a pin that
turns red the day a DSV4 geometry stops being block-aligned and an executed
falsifier that dequantizes both routes and shows the corrupted rows are exactly
the 4 × (128 − 104) wkv rows the shift predicts. Unmeasured — no misaligned
DSV4 geometry exists.

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
Unauthenticated-state-change family (#510, from audit #506): a
state-changing route is not protected by the mere existence of an auth
mechanism — it is protected by the level it is *registered at*. Four
independent instances shipped together: routes at the implicit NORMAL level
that `--admin-api-key` alone does not cover; two whole apps (registry,
video-enhance) with no auth wiring at all; a caller-supplied directory that
overrode its own configuration flag and reached `os.makedirs`; and a write
path missing the enabled-guard its sibling had. The rule this leaves behind:
a new state-changing route needs an explicit `@auth_level`, and a
caller-supplied path needs `utils/path_confinement.confine_to_root` against
the flag that configures it — both pinned by
`test/registered/unit/entrypoints/test_endpoint_security_510.py`, which
enumerates the routes rather than sampling them.

Incomplete-cache-key family (#241 #513, from audit #506): a persisted
artifact whose CONTENT depends on a dimension its KEY does not carry is a
silent wrong hit, not a miss -- and where the artifact is labelled
"measured", the wrongness inherits that label. #241 closed it for HiCache
pages (kv dtype); #513 closed four more, and the shape of each is worth more
than the fix. (a) `rigmon/card_probe.py` WROTE a correct key -- sorted UUIDs,
driver, probe version, with the rationale in the docstring -- and both
readers (`planner/solver_api.py`, `planner/rig_profile_source.py`) then took
the newest file by mtime; a key that the read path ignores is not a key.
`card_probe.matching_cached_probe_json()` is now the single reader and a
non-match is a miss, including when the live inventory cannot be read at all.
(b) HiCache page keys carried tp_rank/tp_size, which under uneven TP do not
determine a rank's kv-head count; `--rank-tp-ratio`/`--rank-kv-ratio` now
enter `compute_model_identity_hash`, appended only when set so an even-TP rig
keeps its persisted pages. (c) The measured-KV-budget fingerprint keys the
post-capture VRAM leftover and omitted `attention_backend`,
`disable_cuda_graph`, `dtype` and `enable_hierarchical_cache` -- all of which
change that quantity; added on the file's own only-when-non-default
convention, so existing digests stay valid. (d) `planner/graphmem.py` keyed
the NUMBER of capture batch sizes, so two different `--cuda-graph-bs` lists
of equal length shared an anchor and the second got the first's numbers with
provenance "measured"; the key is now versioned (`ANCHOR_KEY_VERSION`, v2)
and carries the list, the attention backend and the page size. Two existing
graphmem tests had been passing BECAUSE of (d) -- they looked up
`list(range(12))` against an anchor written for `[1,2,3,4,5,6,7,8,10,12,14,16]`
-- which is the general warning: a test written against an incomplete key
records the collision as expected behaviour. Reference implementations for a
complete key already in this tree:
`managers/kv_session_spill_destination.py:215-239` and
`video_enhance/engine_cache.py:79-121`.

Unreachable-registration family (#81 #518): an op registered for a DEVICE
dispatch key but taking no tensor argument cannot be routed at all -- the
dispatcher infers the backend from the tensors and there are none, so every
call raises "no tensor arguments ... no fallback function is registered",
identically on every arch, in the dispatcher before any kernel. Three GGUF
capability probes shipped that way (`ggml_moe_get_block_size`,
`ggml_mmvq_kq_tuned`, `ggml_mxfp4_native`); the serving path never noticed
because `layers/quantization/gguf.py` had grown a python mirror around the
raise, so the defect surfaced only when #398's Gate A called the op directly
(12/14 on BOTH sm86 and sm120, same two failures). Fix is the keyless
`m.impl` the same file already uses for `apply_token_bitmask_inplace_cuda`.
`test/registered/unit/quantization/test_no_tensor_op_dispatch_518.py` pins it
twice: the dispatcher behaviour on throwaway ops defined in-process (so the
claim is executed, off-GPU, no wheel) and a ratchet over every
`TORCH_LIBRARY_FRAGMENT` schema, so the next tensorless probe cannot
reintroduce it. General lesson: a python-side workaround around a raise keeps
serving alive AND keeps the defect out of sight -- the mirror stays (the wheel
is sha-pinned and can lag the tree), but it is a fallback, not the fix.

Byte-stride width family (#109 #112 #512, from audit #506 A1-1): the GGUF MoE
MMQ kernel reaches an expert with `(char*)vx + exp_idx * exp_stride` where
`exp_stride` is a BYTE stride, declared `int` while the call site already
passed `int64_t` -- so the product wrapped negative once one rank's per-layer
expert tensor passed 2 GiB (DSV4-Flash Q4_K, 256 experts: 2.416e9 B, first bad
local expert 227; TP=3 sharding is the only reason this rig never hit it).
Same read as the #109/#112 expert-id guard, from the other operand, which is
why the guard is pinned in the same test. Rule: a stride in BYTES needs 64
bits at every geometry this fork serves; a stride in BLOCK units has 16-32x
more headroom and `moe_vec.cuh` is left 32-bit WITH the bound written down
(9x at the widest served geometry) rather than silently.

Tolerance-that-cannot-fail (#380 #511, from audit #506 axis 4): a numeric gate
is only a gate if it rejects something. `atol=1.5, rtol=3e1` on outputs whose
RMS is 5.1e3 is the predicate `|a-b| <= 30|b|`, which an all-zeros and a
sign-flipped output both satisfy; a "tighter" ratio check denominated by
`a.abs().max().clamp_min(1e-6)` passes when both arms are zero. Replaced by a
tolerance DERIVED from the one physical error source (q8_1 activation
rounding: `sigma_ij^2 = sum_k d_k^2 w_jk^2 / 12`, extreme value over N
elements ~ sqrt(2 ln N) sigma) plus a spread precondition on the reference and
a magnitude precondition on each arm. Every gate in that file now has an
off-GPU can-discriminate test showing it reject a zeroed and a sign-flipped
output, including the refuted baseline executed as a test.

Superseded-premise family (#529, same audit axis, third instance): a test that
still RUNS but whose world changed. `test/registered/unit/model_loader/` stood
at a permanent **37 failed / 284 passed**, and the red block was not a product
defect in any of its three causes -- it was the harness. Standing red is not
neutral: it anaesthetises the regression signal, and the #526 merge had to diff
failure sets line by line to tell a new break from the noise. Causes and the
repair chosen per cause, since "skip it" is only right for one of them:
(1) **19 MXFP4 repack failures** -- #398 made the wheel execute ggml type 39
natively, so `gguf_mxfp4_repack` is the identity and every assertion about a
Q5_0 payload described bytes nobody rewrites. NOT skipped: the repack still
ships (the wheel is pinned separately from the source, and
`SGLANG_GGUF_MXFP4_NATIVE=0` is the standing A/B lever), so the state is forced
IN-PROCESS via `sglang/test/gguf_mxfp4_state.py` (env lever + module reload),
which is deterministic on every wheel and needs no capability skip that would
sleep forever on a native one. The measured value of that distinction: with an
off-by-one planted in the repack lattice offset, the pre-#529 file caught
**0** (15 red before, 15 red after -- pure noise), the forced-state file catches
**6**. Four of its tests had also been passing VACUOUSLY, e.g. "a slice of a
repack equals a repack of a slice" is trivially true of the identity.
(2) **5 Qwen3.8 forward-compat failures** -- a test stub that hand-listed the
`ServerArgs` methods it forwarded, which stopped matching when those methods
were refactored to share a `_read_declared_config` helper. Repaired by
delegating generically (`__getattr__` -> `functools.partial`) so a future
private helper travels with them; a planted off-by-one in
`full_attention_interval` now turns 2 tests red, where before the fix the file
caught nothing at all.
(3) **6 modelopt failures** -- a `register_cuda_ci` file that has no business
running on a hermetic CPU sweep. Skipped, but with the predicate derived from
`get_device()`, i.e. the SAME call the tests trip on, so the marker cannot
drift from its cause and lifts by itself on the GPU runner (proven: with the
probe reporting an accelerator, all 6 run instead of skipping).
The native MXFP4 path -- the one this rig actually serves -- had LESS
loader-level coverage than the dead repack path beside it, so
`test_gguf_mxfp4_native_path_529.py` pins it through the real
`gguf_quant_weights_iterator`: markers stay 39, payload stays 17 B/block, the
per-expert slice stays self-contained and decodable, the load-time line
announces the SAVING rather than a conversion, and the executability gate
accepts MXFP4 with the repack explicitly off. Its own falsifier runs both paths
over the same bytes and asserts they differ. After: **0 failed / 320 passed /
15 skipped**.
The red count was itself WHEEL-DEPENDENT, which is why two agents reported two
different numbers for the same tree on the same day: at the line tip the suite
is **37 failed** with the shipped native wheel and **11 failed** under
`SGLANG_GGUF_MXFP4_NATIVE=0` (both measured), the 26-failure difference being
exactly the MXFP4 repack block, which passes whenever the repack actually runs.
11 = 6 modelopt + 5 Qwen3.8; neither of those depends on the wheel. A suite
whose failure count moves with an env var cannot be used as a regression
signal at all. After the fix both states give the identical **0 failed / 320
passed / 15 skipped**, because each test now forces the state it means instead
of inheriting it.

Reach-before-fix (#487, the shape of a good negative result): the stock
even-DCP allocator branch (`model_runner_kv_cache_mixin.py`, the `else` of the
allocator chain) inflates BOTH the index space and the page granularity by
`dcp_size`, i.e. it assumes a token-sharded pool -- while a draft worker at
`--draft-kv-layout replicated` (the default) has the opposite geometry, which
the pool SIZING guards on (`draft_pool_is_replicated`) and the allocator
selection never mentions. #108 never audited the crossing. Answer, established
without a boot: **unreachable on CUDA**, by two predicates, one per producer of
`is_draft_worker=True`. Given `dcp_size > 1` the stock branch is taken exactly
when `rank_tp_ratio is None and not weightless_kv_active()`; (1) a SPECULATIVE
draft worker cannot exist in that shape because
`ServerArgs._handle_dcp_validation` refuses `dcp_size>1` + speculation on CUDA
unless the boot is uneven-weighted DCP (requires rank_tp_ratio) or the
weightless fast lane -- the gate's own two disjuncts; (2) a #274 DUAL-GROUP
LANE runner also sets `is_draft_worker=True` and is NOT speculative, so leg 1
misses it -- it is closed instead by `_lane_server_args_view` forcing
`view.dcp_size = 1`. Leg 2 exists only because the obvious premise ("a draft
pool implies a speculative algorithm") turned out FALSE when the producer set
was enumerated rather than assumed; the test now pins that set, so a third
family lands as a red test instead of a wrong address. Residual, named rather
than fixed: on HIP/ROCm leg 1 does not run (`if is_hip(): return` precedes the
CUDA branch), so there the crossing IS admitted -- not changed, because this
fork does not serve ROCm and a desk-guessed change to an address computation is
the #345 right-token/wrong-slot class waiting to happen. Corroboration that the
two layouts are not interchangeable comes from a guard written for another
feature: `_init_pools` refuses `--enable-kv-session-offload` on this branch
because "the stock even-DCP inflated-page layout re-interprets slot identity".

Forward-compatibility-by-model_type (#497, day-0 prep for a checkpoint that
does not exist yet): the load path must key on ``model_type`` and on config
FIELDS, never on a version string in the model name or on a geometry constant
-- then a new checkpoint that reuses an existing ``model_type`` loads with no
code change at all. Audited and CONFIRMED for both:
``gguf_registry.create_gguf_adapter`` reads exactly ``model_type`` and the
registry contains no ``name_or_path``; ``ServerArgs.declared_layer_kinds``
probes ``layer_types`` -> ``full_attention_interval`` -> all-attention, from
the top level or ``text_config``, so a different depth and interval need
nothing. Both are now ratcheted, including a literal sweep that ignores
docstrings (the family is documented in prose everywhere, so a naive grep
ratchet would be pure noise -- the sweep parses and looks only at string
literals the module evaluates). Two findings came out of it and are NOT fixed
on purpose: (1) an unregistered architecture does not refuse, it falls back to
``TransformersForCausalLM`` (``models/registry.py:61-78``), so a boot can
succeed on the generic backend with none of this fork's features -- assert the
resolved class at day 0 rather than trusting a green boot; (2) the M-RoPE
declaration gap vLLM PR #50068 closes on its side exists here too --
``model_runner.py:599-604`` decides ``model_is_mrope`` from the CONFIG alone
(true for text-only Qwen3.5/3.6, which carry ``mrope_section``) while
``models/qwen3_5.py`` sets ``is_mrope_enabled`` only on the two
``ForConditionalGeneration`` classes, so
``prefill_cuda_graph_runner.py:521-531`` computes mrope positions and then
discards them. Closing it changes which positions a captured graph replays and
therefore needs a boot; it is characterised by tests instead, which flip
together when the fix lands. `docs/dev/ANALYSE_495_qwen38_forward_compat.md`.

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

Reclaim-then-decline family (#501): a path that DECLINES and hands the caller
back to a named fallback must leave the request byte-untouched, and a one-shot
flag is the sharp edge. `try_spill` committed the speculative-overhang reclaim
(`allocator.free` + `kv_allocated_len` + `kv_overallocated_freed = True`)
before three later declines -- empty spill plan, #236 budget regler, tail wider
than one host region -- each of which returns False and leaves the victim
RUNNING, so the request's eventual `pop_overallocated_kv_cache()` asserted
(`schedule_batch.py:1141`) and killed the scheduler instead of falling back to
stock retraction. Fixed by PLANNING the reclaim where the state used to be
written and COMMITTING it past the last decline (`reclaim_overhang` /
`allocated_after_reclaim` in `managers/kv_session_offload.py`); the three
predicates in between are pure over the snapshot, the row head `[0, L)` and
replicated budget state, none of which the reclaim touches. So the rule: an
irreversible mutation belongs behind the LAST decline of its function, and a
comment claiming "a decline leaves no partial state" is a testable assertion,
not prose -- `test/registered/unit/test_kvso_reclaim_decline_501.py` pins the
ordering structurally so a decline added later cannot move in front of it
(4 tests, all four executed can-fail against the pre-fix file).

Two-stage-error family (#505-B-01, fixed in #514): when ONE `try` spans two
stages whose failures mean OPPOSITE things about where the bytes are, the
handler is forced to lie about one of them. `RealMovementBackend.wave_in`
wrapped the retrieval (`copy_in_tensors` + `wait`) AND the destination release
(`free_destination`) in one block whose handler wrote `state = STATE_PARKED`
unconditionally -- the state whose meaning is "the bytes are at the park
target". A `free_destination` failure happens AFTER the bytes are demonstrably
back, so the item was reported parked while resident, and because
`ledger.release` sat outside the `try`, the park-target booking was never
released: a permanent over-booking of the memtier ledger. The pre-fix test
injected only at `copy_in`, the one point where the shared handler's comment
was true, so the suite was green throughout. Split into STAGE 1 (retrieval:
failure -> PARKED, booking retained, `wave_in_failures`) and STAGE 2 (release:
failure -> RESIDENT, booking retained **and counted** as
`leaked_destination_bytes`, `destination_release_failures`, distinct error
text). The booking is deliberately NOT released on the leak path -- how much of
the destination came back is unknown and under-booking would let the next
booker allocate into bytes that may still be held; the requirement is that the
leak is NAMED and COUNTED, not that it is reclaimed. So the rule: a `try` whose
stages disagree about the meaning of failure gets one handler per stage, and a
counter split out of an existing one is added to every gate that read the old
one in the same change (`scripts/gpu_battery/checks/check_s07_offload_register_gpu.py`
gates all four counters, each with its own can-fail proof).

Group-agreement family (#505-A2-05, fixed in #514): a transport that MAY fall
back must fall back as a GROUP. `_build_transport` caught any bring-up
exception for the two transports outside `_NO_FALLBACK` (`bar1`, `matrix`),
warned, and returned `None` per rank, with nothing reconciling the outcome --
so a rank that failed after its transport's last bring-up collective ran gloo
while its peers ran barlink, and a barlink-default run was published as such
while being a gloo run. Fixed with a one-hot `all_reduce(SUM)` on the CPU group
that BOTH the success and the failure path reach: all-ok keeps the transport,
mixed sends every rank to gloo (the successful ones close theirs), and the
failing ranks are named. Shape taken from `parallel_state.py:975-992` and
`model_runner.py:1365-1369`. Deliberately unbounded, like the neighbouring
bring-up exchanges: ranks are legitimately minutes apart on a cold JIT cache
(#431/#438a) and a deadline would fire on a healthy group. Scope, measured
rather than assumed: the byte proof already reduces group-wide
(`barlink_bar1.py:2538`, `:2547`) and the `dmabuf_holder` / `_bind_region`
guards sit before a remaining collective, so those DESYNC into a hang rather
than split -- the residual hang class is NOT closed by this and is a named
follow-up.

Rank-local-verdict family (#505-A2-03, fixed in #514): a decline that hands the
caller to a NAMED FALLBACK must rest on replicated inputs, or the fallback is
taken by some ranks and not others. `try_spill`'s host-region check compared
THIS rank's owned tail rows (`n_own`, derived from the rank-local owner window
`cp_prefix[dcp_rank : dcp_rank+1]`, whose width differs per rank under uneven
DCP) against the replicated `region_tokens`, then returned False -- sending only
that rank down stock retraction while its peers spilled. The caller's own
comment claimed "no second collective, no branch-count divergence"
(`scheduler.py:4058-4059`), which the rank-local verdict falsified; the
prefill-spill twin RAISES on the identical condition (`:3906-3913`). Fixed by
taking the verdict on the WIDEST rank (`spill_tail_rows_max_over_ranks`) --
computable locally, with no new collective, because the `req_to_token` row
carries global slot ids and `cp_prefix` is replicated, the same trick `_restore`
already uses. `plain`/single-rank is byte-identical to the old path. Raising
like the twin was rejected deliberately: the decode path HAS a named fallback
the prefill path lacks. #501 ordering untouched -- still a `return False` ahead
of the reclaim commit.

Fabricated-identity family (#505-A2-04, fixed in #514): a bridge that cannot
answer must return UNKNOWN, never the index it already has.
`nvml_card_totals_mib` logged "assuming identical enumeration orders" and fell
back to the identity map, so on the reference rig (5090 = CUDA 0 / NVML 1) the
PD feasibility check compared each card's plan against ANOTHER card's capacity
and passed a plan that cannot fit -- warn-only downstream, so it booted with no
error. Two aggravations found while fixing it: the `except` arm was not the main
entrance (`registry/nvml.py:706-729` returns `{}` NON-exceptionally whenever
torch cannot place cards, so the fabrication was on the ordinary success path),
and an existing test PINNED the defect (`test_empty_mapping_is_identity`
asserted `reindex_totals_cuda_order(nvml, {}) == nvml`). It was the one caller
#397 never migrated -- `test_device_order_bridges_397.py::NoCudaBridgeTest`
pins the refusal for every other. Now routed through the IdentityMap
(`registry_nvml.identity_map`, PCI-BDF), an empty bridge yields `None` with a
named warning (the barlink `"sysfs-gross"` shape), a partial bridge keeps the
placeable cards and names the unplaced ones, and a single-GPU host stays
identity because `cuda:0` can only be that card. What BOOTS is unchanged: the
unverified state is loud, not fatal -- a strict mode belongs behind an explicit
flag rather than a silent default flip. So the rule: an unavailable bridge
degrades to a NAMED unknown, and a test asserting `f({}) == identity` is pinning
the defect.

**VALUE PINNING for bounding defaults (#505-C-05, convention adopted in #514).**
A numeric default that exists to BOUND something (cap/budget/threshold/limit/
reserve/margin/watermark/timeout) ships with a test that FAILS when the value
changes -- see `docs/dev/CONVENTION_bounding_defaults.md` and the reference
implementation `test/registered/unit/test_bounding_default_value_pins.py`.
Audit #505 enumerated 106 fork-added bounding defaults and found ZERO with such
a test, while 71 of them sit behind a gate that is off in the served
configuration. The anti-pattern is a test that READS the default and derives its
assertion from it: it proves the guard fires and passes for every possible
value, which is how #449's desk-picked 2048 MiB shipped above the real peak and
protected nothing for weeks. A green pin means the number is DELIBERATE -- not
that it is correct, and not that it BINDS; those need their own evidence and
are the axis-C backlog.

Resolution-ordering family (#499): a `server_args` field read on both sides of
`materialize_declarations` is TWO values. The #89 hibernate manifest identity
(`model_loader/hibernate.py:_model_identity`) is computed at the MATCH inside
`ServerArgs._handle_load_format` (`server_args.py:13332`, reached from
`__post_init__` at `:5939`) and again at the PARK in the worker
(`weight_updater._hibernate_park_weights` -> `hibernate.py:518`) -- the first
BEFORE `materialize_declarations(self)` (`server_args.py:5984`), the second
long after it. `quantization` is declared `"gguf"` by the `_gguf_quantization`
pass (`arg_groups/overrides.py:2091`), invoked at the HEAD of that same handler
(`:13300`), so every manifest was written with `"gguf"` while every subsequent
boot compared `None`: the identity could never match its own park, and #89's
fast restore was unreachable on GGUF -- the only checkpoint class #89 supports
(`if self.load_format != "gguf": raise`, `:13327`). The park kept reporting
success and the mismatch fell back to a cold load that works, so nothing
surfaced (SUCCESS-CLAIMS family). Fixed by computing the identity through
`resolved_view` (`overrides.py:230-235`), the mid-resolution equivalent of the
post-materialization fields, so both sides read the same value.
`HIBERNATE_VERSION` stays 2: the park side's bytes do not move, the match side
joins it, and manifests parked before the fix become matchable rather than
invalid. Reach of the hazard read off the metadata, not off prose: of the nine
identity fields only `quantization` and `dtype` are declarable at all
(`Arg(resolvable=True)`) -- `validate_declarations` (`overrides.py:2176`)
refuses the other seven by name, so they can only be written imperatively, and
every such writer runs before `:5939` (`rank_tp_ratio`/`dcp_size` in
`_handle_uneven_tp`, `:5785`). So the rule: **a value read during
`__post_init__` and again after it is two values unless it is read through
`resolved_view`**, and a fingerprint compared across processes is computed by
ONE function at ONE pipeline stage.
Falsifier: `test/registered/unit/model_loader/test_hibernate_identity_499.py`
(6 tests incl. a metadata-derived sibling sweep and an end-to-end "park, then
reboot the same command" arm; 4 executed red against the pre-fix file, the
sweep red on both declarable fields).

**#520 (the #499-B residual): a field the WORKER re-derives after the match.**
`resolved_view` cannot reach it -- the match runs in the launcher, before the
process that computes it exists. TWO live instances, both found and both fixed:
`ModelRunner._sm80_dtype_fallback` declaring `dtype="float16"` on a card
without bfloat16 (`model_executor/model_runner.py:1980`; sm75 hetero host,
gfx900-class ROCm) and -- the one the ticket did not predict --
`spec_worker.match_target_context_length` pinning `context_length` to the
target's `context_len` (`speculative/eagle_worker_v2.py:1764` plus the
standalone / frozen-KV-MTP / multi-layer workers, all four the SAME source
string). The spec one writes the SHARED `server_args`, which is exactly the
object the park reads (`BaseSpecWorker.model_runner` returns
`self.target_worker.model_runner`, `base_spec_worker.py:384`), so **#89's fast
restore was unreachable on every speculative boot as well** -- and nothing
refuses spec under hibernate (`server_args.py:13410-13439` gates only on GGUF).
Executed divergence, CPU-only: `{'dtype': ('auto','float16')}` and
`{'context_length': (None, 262144)}`.
Fix: the identity reads through `launch_view` (`arg_groups/overrides.py`) --
`resolved_view` plus the un-applied writes of `IDENTITY_TRANSPARENT_SOURCES`.
A source is identity-TRANSPARENT when it RE-DERIVES a field from state the
fingerprint already pins (the rank's card, re-checked by NVML UUID at
`hibernate.py:568` and presence-gated at match time in
`_manifest_cards_present`; the checkpoint at `model_path`). A source that
changes WHAT IS LOADED is deliberately absent and must keep showing through --
`model_runner.update_weights` is the case #499 argued must never be normalized
away. The superseded launch value is recorded at the single post-resolution
mutation point (`ServerArgs.override`, `server_args.py:14786`), first writer
wins; with no such source the register stays absent and the read is
byte-for-byte the #499 read. `HIBERNATE_VERSION` stays 2 for the same reason as
#499. Sweep is MAINTAINED, not by eye: an AST scan over `srt/` classifies every
literal-source post-resolution writer of an identity field as transparent or
explicitly opaque (8 sites at this tip), so a new worker-side writer turns red.
Falsifier: `test/registered/unit/model_loader/test_hibernate_identity_520.py`
(12 tests, 4 executed RED against the unfixed identity read AND separately
against a disabled recorder hook -- both halves proven load-bearing; the
`_needs_float16_fallback` arm fakes only the capability tuple and is asked in
both directions).
Residual, named and NOT fixable at the identity level: on heterogeneous
hardware only rank 0 writes the identity, so a per-rank re-derivation describes
rank 0 only -- the per-rank NVML-UUID re-check is what pins the others. And the
end-to-end proof (park on the sm75 host, relaunch, expect the 50s -> 8-14s
restore) is still owed; only a real mixed-hardware park can show it.

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

**Verified carve-outs (#505-A2-01, #514).** A size/budget carve-out written
onto an allocator is VERIFIED BY READ-BACK and refused by name if it did not
take. `kv_session_offload` did `self.allocator.size -= mtp_resident_slices`
inside `except Exception: pass` and then logged "reserved %d draft-read scratch
slots"; on `UnifiedMambaTokenToKVPoolAllocator` / `UnifiedSWATokenToKVPoolAllocator`
`size` is a COMPUTED property whose setter is literally `pass`
(`multi_ended_allocator.py:1764-1766`, `:1974-1976`), so the write vanished with
no exception -- the handler hid nothing and the success message was the only
evidence. The damage is a standing false leak report: the invariant checker
reads exactly that value (`invariant_checker.py:124`), and on the composite the
permanent `alloc()` moves `schedulable_available -> allocated_count` leaving
`size` unchanged, so the accounting is off by `mtp_resident_slices` forever. The
composite setters deliberately STAY no-op absorbers -- `BaseTokenToKVPoolAllocator.__init__`
and the inherited `resize()` (#330 VRAM dial) legitimately write there and a
raise would break them -- so the obligation is the caller's. This is the
SUCCESS-CLAIMS law as a code rule: where a write can be absorbed, read it back.

**An absent measurement is `None`, never `0.0` -- including in the harness
(#459).** The #359 rule for `RigRates` now also holds for the log parsers: a
spec-off tick has no accept length, and a `0.0` there survives every arithmetic
it enters and divides into ms/verify as if it had been measured.

**A procedural property must be visible in the artifact or it is not evidence
(#459).** The s12 warm-up draw ran for months and was discarded SILENTLY, so a
warmed-up point and a cold one wrote the same file. Any run property a later
verdict rests on (warm-up discarded, draws back-to-back, gap between draws) is
recorded as a measured NUMBER plus the verdict derived from it, so the claim can
be refuted from the file rather than taken on the harness's word.

**MERGE DUTY -- SITREP (#509).** A merge that changes what this fork can do,
how fast it does it, or which claim about it still holds also UPDATES the
matching head section of `STATUS.md` in the private dev-log repo
(`efschu/htsglang-dev-log`) and APPENDS a log paragraph -- two distinct
actions, not one. The head is maintained state and is corrected in place, so a
claim it supersedes is deleted rather than softened; the log is append-only and
is never rewritten. Every head claim carries an evidence anchor (catalog
section, task number, merge hash or RESULTS path), and a conditional claim
there is a pointer to its predicate under the same MECHANISM REACH rule this
file follows.

## 13. Serving surface
OpenAI-compatible with `--reasoning-parser qwen3 --tool-call-parser
qwen3_coder` (server-side fix, no template patches); fast lane, priority
scheduling, admission throttle, prefill delayer; training tenant
(`--enable-training-tenant`) + idle workbench (`--enable-idle-workbench`,
ledger + pause rung).

Fork-added HTTP surface, enumerated against upstream (audit #500) — the
catalog previously named two of these and the rest were discoverable only from
the router:
`POST /session_handover` (`http_server.py:1126`), `POST /kv_reshard` (:1109),
`POST /vram_budget` (:1149 — the runtime VRAM dial's actuator),
`POST /hibernate` (:1706 — POST-only since #510),
`POST /v1/images/{generations,edits,variations}` (:2013/:2022/:2057) and
`POST /v1/audio/speech` (:2067) — the OpenAI-compatible **diffusion** and
**speech** lane fronts, wired unconditionally at `http_server.py:461-462`,
routing to a registry-promoted class-2 lane and refusing with the registry's
own numbers when none is HOT (`serving_images.py:62-100`; `edits`/`variations`
are a named 501 at `:192-199`),
`/v1/files` + 4 sub-routes and `/v1/fine_tuning/*` + 6 (:2079-2170, #341-M1),
`/x-htsglang/workbench{,/events,/pause,/enqueue}` (:2191-2236).
Outside `http_server.py`: `video_enhance/server.py` serves
`/v1/video/{enhance,plan,capabilities,engines,tracks,liveness}` plus
`/v1/video/enhance/{job_id}` and `/v1/video/preview/{job_id}/{which}`, and
`registry/http_api.py` serves the engine control plane
`/registry{,/cards,/engines,/engines/{id},/engines/{id}/pin,/engines/{id}/state,/idle,/plan,/default_hot}`
— which is what the diffusion refusal above tells the operator to use.

Auth and CORS on that surface (#510, closing audit #506's endpoint axis). The
rule is the same in all three apps and it is **key-gated, not default-on**:
with no key configured nothing changes, and a configured key is what closes
anything.
* Every fork-added state-changing route is now decorated
  `@auth_level(AuthLevel.ADMIN_OPTIONAL)` — the level whose decision table
  (`utils/auth.py:149-159`) is "no keys: allow; `--api-key` only: require it;
  `--admin-api-key` set: require the admin key". Before #510 these routes sat
  at the implicit NORMAL level, where `--admin-api-key` **alone** protects
  nothing (`utils/auth.py:161-167`), so an operator who set only the admin key
  had `/v1/files`, `/v1/fine_tuning/*` and both workbench writers wide open.
  `/hibernate`, `/vram_budget`, `/kv_reshard` and `/session_handover` were
  already at ADMIN_OPTIONAL and are unchanged.
* `registry/http_api.py:build_app(registry, api_key=, admin_api_key=)` and
  `video_enhance/server.py:create_app(..., api_key=, admin_api_key=)` now
  accept the two keys and install the runtime's own middleware when either is
  set; their state-changing routes carry the same level. `python -m
  sglang.srt.registry` gained `--api-key` / `--admin-api-key`
  (`$SGLANG_REGISTRY_API_KEY`, `$SGLANG_REGISTRY_ADMIN_API_KEY`). This is what
  stands between an unauthenticated LAN peer and `launch.argv`, which
  `registry/adapters/class3_utility.py:171-186` hands to `subprocess.Popen`.
* `POST /hibernate` is POST-only (it accepted GET, so a link preview parked
  the model) and a `hibernate_dir` in the body is confined to the configured
  `--hibernate-dir` by realpath (`utils/path_confinement.py`), enforced both
  in the handler (400) and at the scheduler-side sink
  (`weight_updater._hibernate_park_weights`, which the Engine API reaches
  without the route). It used to override the flag outright.
* `--cors-allow-origins` (default `*`) replaces the hardcoded
  `allow_origins=["*"] + allow_credentials=True`. Credentials are sent only
  for an explicit origin list; the old pair is illegal per the Fetch standard
  and let any page the operator had open drive a loopback-bound runtime.
  `http_server.cors_policy()` / `configure_cors()` own it; configure replaces
  the import-time middleware rather than stacking a second one.
* `POST /v1/files` is behind the training-enabled guard `create_job` already
  had (`training/service.py:_require_tenant`), so a server without
  `--enable-training-tenant` no longer writes uploads to disk.
* ffprobe/ffmpeg stderr is logged, not reflected
  (`video_enhance/mux.py:subprocess_failure`) — reflected into a 422 it was a
  filesystem existence oracle for a caller who also chose the input path.

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

**The Anthropic Messages front is complete enough to back a Claude Code agent
loop** (#530): `POST /v1/messages` (`http_server.py:2578`) plus
`POST /v1/messages/count_tokens` (`:2588`), SSE deltas in Anthropic event
shape, and NO model-name validation — an unknown id is echoed back
(`{"model":"claude-sonnet-4-5"}` and `{"model":"default"}` both answer 200).
That last property is what makes a checkpoint switch invisible to clients that
pin a name, the #466 translator's `--mt-model default` included. Reach, stated
at the width of what was checked: a real `claude` CLI process (2.1.220) driven
against a live 27B boot returned correct determined answers, so no
LiteLLM-class OpenAI translation proxy is needed for this — but two client-side
settings are load-bearing and are NOT defaults: `MAX_THINKING_TOKENS=0`,
because Claude Code requests an Anthropic `thinking` block and a boot without
`--reasoning-parser` answers `400 Anthropic thinking is not supported for
models without a reasoning parser`; and `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, because
the default 32000-token completion request plus the ~20k-token system prompt
overruns a 32k-context boot. Claude Code itself has NO per-subagent endpoint
binding (the subagent frontmatter schema carries no `baseUrl`/`provider`/`env`
key; every `ANTHROPIC_*_BASE_URL` is process-global), so the fork ships the
named fallback rather than the feature: `scripts/dev/local_model_agent.sh`
starts a separate `claude` process with a process-scoped environment, and
`scripts/dev/register_local_model.sh` regenerates the USER-GLOBAL
`~/.claude/agents/local-model.md` from `GET /v1/models` so the entry follows a
serving switch, then VERIFIES the file at the path that is actually read and
prints it. The verification exists because the first version wrote to
`$REPO_ROOT/.claude/agents`, which for a worktree is read by no session: the
agent type appeared in no agent list while the script reported a successful
write, and the wrapper round-trip could not catch it because the wrapper reads
`~/.config/htsglang/`, never the agent file — it proved the WRAPPER, never the
REGISTRATION (§12 success-claims-are-not-evidence, and the reason the closing
probe re-reads the canonical path rather than trusting its own `cat`). An
`--agent-dir` that is neither user-global nor the current project exits 7.
Session-lifetime rule: an agent list is loaded at session START, so a
re-registration reaches NEW sessions only. Runbook §13.
**The usability trias — chat template + `--reasoning-parser` +
`--tool-call-parser` — is a STANDARD boot setting, not a tuning knob** (user
standing order 2026-08-03, #531). A boot missing them answers HTTP 200 while
degrading silently in three separate ways, all observed on this rig's FP8 boot:
chain-of-thought lands in `content` as raw `</think>` text, an Anthropic
`thinking` block is refused with `400 Anthropic thinking is not supported for
models without a reasoning parser`, and a tool call returns as a JSON-looking
STRING rather than a structured `tool_calls` entry. The mapping is CODE, not a
doc table: `planner/flags.py::usability_parsers` resolves the pair from the
checkpoint's `architectures` (path as fallback) over a squashed identity
string, so `Qwen3.6-27B`, `qwen3_5` and `Qwen3_5ForConditionalGeneration` all
resolve alike; the table is ordered specific-first because `v3` is a substring
of `v32`; and `validate_usability_parsers()` checks every emitted name against
the live `ReasoningParser.DetectorMap` / `FunctionCallParser.ToolCallParserEnum`
so a registry rename turns the mapping red instead of shipping a flag the
server rejects. Both planner command generators emit it
(`feasibility.py::_launch_flags`, `key_solver.py::_usability_launch_flags`) and
an unrecognised family emits a NAMED HINT instead of a bare command.
`register_local_model.sh` reads the live `/server_info` and writes
`current boot lacks <what>; agentic tool use degraded` into the generated agent
header. Scope is deliberately serving-only: the measurement arms under
`gpu_battery/`, `dual_group/`, `probe*/`, `determinism/`, `nordstern/` are NOT
patched, because a reasoning parser moves text out of `content` and would shift
the very token accounting those arms produce. Boot-proven on the INT8 instance:
template applied (system rule obeyed two turns later, zero marker leak),
reasoning split into `reasoning_content` (1103 chars, no `</think>` in
content), tool call parsed to `get_weather{"city":"Oslo"}`. Tests:
`test/registered/unit/test_usability_parsers_531.py`, 11 hermetic + 13
subtests, three executed can-fail arms (pre-fix token-set match → dotted
families go None; `v3` row moved ahead of `v32` → wrong point-release parser;
bogus name injected → registry check fires).
**Coexistence reserves come from a co-tenant's DECLARED budget, never from a
momentary observation** (#530, runbook §4.1): the #466 translator held 4204 MiB
on the 5090 while declaring 7500, so the INT8 boot reserved `13000,3800,3800`
(5500 long-prompt + 7500 declared) rather than sizing against what nvidia-smi
happened to show. Cost ~135k KV tokens, bought a coexistence that survives the
tenant growing into its own budget.
**Checkpoint switching is a RESTART, and the three live routes were priced at
their source before that verdict** (runbook §14): `update_weights_from_disk`
refills the EXISTING module tree and rebuilds neither the quant methods nor the
pool geometry (`model_runner.py:2536-2563`, with its own rollback branch at
`:2548-2556`); the #305 registry's HOT promotion boots a SEPARATE PROCESS on
its own port (`registry/adapters/class1_srt.py:220-241`, `build_argv` at
`:279-290`) and its demotion actuator refuses an engine it did not start
(`:411-415`); the #274 dual-group lane is a second GROUP over the SAME tensors
by `data_ptr` identity (`dual_group_lane.py:15-27`), not a second model.

## 14. Dashboard
Guided config wizard whose refusals each cite their source and which never emits
a flag it cannot explain (`planner/wizard.py:703-714`, `:1521`, plus
`wizard_islands/_lanes/_links/_offload/_tipping`). Comm benchmark suite; its
anonymization gate (`rig_artifact.assert_anonymized`, `rig_artifact.py:558`,
reachable only through `build_digest`, `:784-795`) covers **both** share routes
since #514: the #152 result-share route runs its payload through the same
`scrub_tree` + `assert_anonymized` inside `build_report`
(`github_share.py:144`, `:296`) before a single line is rendered, and `submit`
refuses a #152 body this process did not render (`:584`), so the previewed
string and the posted body are the same bytes (#505-D3 fixed). One field is
held out of the gate by necessity: the quality shot's `svg`, because the shared
path rule would rewrite its markup. Energy metering (tok/s + J/token) is
NVML board power integrated per phase (`energy.py:23-24`, `:383-412`) and is
therefore **GPU power only — not wall-socket energy** (`energy.py:278-279`).
Benchmark tiles carry measured/estimate/absent provenance with no "probably"
tier (`cost_model.py:142-146`); the decode-knee guard is **modelled**, and
reports ABSENT rather than "safe" when the membw scores it needs are missing
(`wizard_tipping.py:587-607`) — there is no knee-point probe: `power_limit_sweep`
(`energy.py:1217`) would measure one but has test callers only (#505-D6).
**Monitor tab, #522.** Three additions, all computed SERVER-side so `curl
/api/live_snapshot` is the same view the page draws (runbook §8's rule).
(1) **Four-state server diagnosis** (`planner/server_state.py`): the old page
printed "started without --enable-metrics" for ANY failed scrape, including
connection-refused — a guessed cause, not a diagnosis. The discriminator is a
second, metrics-independent API probe (`/get_model_info`, then `/health`), and
it is spent ONLY when the scrape failed (a 200 on /metrics already proves the
API, same HTTP server). `RUNNING_NO_METRICS` sits inside the `api.ok` branch of
`classify` and nowhere else (`server_state.py:194-208`), so the old claim
cannot return; `STARTING` requires NAMED evidence — the supervisor's own
`booting` state, or the port accepting TCP while the API does not answer —
and falls back to `NOT_RUNNING` where neither holds, never a heuristic
"probably coming up". (2) **Median badges** on the rate tiles
(`planner/rate_medians.py`): an idle tile reads `0 (median 43.2)`, where the
median covers the last `RATE_MEDIAN_WINDOW` = 30 PROCESSING windows only —
60 s of actual processing at the 2 s poll, the same horizon the per-card rings
use. Idle polls never enter the window (per-key activity predicates on the
counter DELTAS, not on the rate value), because a median over the idle zeros
would be 0 and the badge would carry nothing; an empty window renders no badge
rather than a zero. (3) **Spill/offload tier panel** under the per-card
placement (`planner/tier_occupancy.py` + `observability/spill_tiers.py`): one
row per tier = type x place, from local VRAM through host RAM and local disk
out to the paired rig. Occupancy comes from each consumer's OWN ledger through
new `sglang:spill_tier_used_bytes{spill_tier=…}` / `_total_bytes` gauges —
expert pool summed from the live per-layer pinned tensors, kvso from occupied
regions x region bytes, #224 park from parked-session rows x bytes-per-token,
HiCache-L3 from the file backend's evictor total. **Reach note, and it is the
point of the entry:** the #407 `memtier.TierRegistry` is NOT the data source
and cannot be — it is a capability/profile description with no production
consumer (`memtier/registry.py:34-36` "no consumer is wired to it yet",
`memtier/__init__.py:65` "No consumer reads any of this yet";
`TierCapacity.reserved` defaults to 0, `tiers.py:365`, and no `reserved=` is
ever passed in `bootstrap.py`), and the #286
short-term register's `CapacityLedger` is real but unreachable
(`OffloadRegister` always falls back to `CpuFakeMovementBackend`, which counts
ids and no bytes, `offload_register.py:563`). Both are therefore drawn as
ABSENT rows with that reason, as are the NVMe expert tier (#389, design only),
hibernate staging (a state, not a live gauge), and every unconfigured remote
tier. The cumulative ledgers are deliberately NOT used as occupancy
(`StreamingStagingLedger` only ever adds; `park_bytes_out` is bytes ever
moved). /proc is consulted for exactly one thing — MemTotal as the host-RAM
denominator — and the row says it is the DASHBOARD host's. Provenance is #218's
`measured`/`absent` with no "probably" tier; the token-valued HiCache-L2 row is
excluded from the byte sum by name rather than blended into it.
Self-update installs in any serve mode; **switching + auto-rollback need
`--serve-supervised`** (`webui.py:3632`, `self_update.py:659-688`), and the
health gate is HTTP 200 on `/` (`self_update.py:691-712`, #505-D8). GitHub
result posting is opt-in per-use PAT, redacted from every error path
(`github_share.py:97-105`); env-value redaction keys on five NAME suffixes
(`:89`) — since #505-D3 that is a second layer on top of the shared scrub, not
the only redaction.

## 15. Model bring-ups (boot-proven)
Qwen3.5/3.6 family (all quants), Gemma4 26/31B (+GGUF, quadratic-mask skip;
Gemma3RMSNorm runs the fused sgl-kernel path for 2-D and high-rank inputs,
adopted from upstream #32670 — do not re-add an eager-only forward_cuda),
Llama family, Mistral Small 24B FP8 + ministral3 SWA fix, Deckard-40B/Tess-27B,
122B-A10B offloaded, 35B-A3B, DeepSeek-V4-Flash-0731 GGUF TP=3 offloaded with
OWN sm86+sm120 attention paths (e4m3 bit-decode, f32 staging, indexer arch
dispatch, torch/triton reference-twin parity: indexer mask oracle, SWA
page-index wrap oracle, page-table rounding, top-k seq_len contract).
**The sm120 SWA page-split is MASKED since #471** (port of upstream #32320,
`204e0fbac0`): `_page_split_kernel` no longer rewrites the whole pbs=256 pool
into its pbs=64 view every decode step. `_page_mark_kernel` sets one int8 byte
per source page from the token indices (`mask[token // src_pbs] = 1`, -1
skipped) and the split kernel returns early for an unmarked page, so ~2*batch
pages are copied instead of the pool. Untouched destination pages keep stale
bytes by design — sound because the caller reads only the pages the same
indices address. Upstream's ITL/TPOT numbers are NOT reproduced here; the port
is desk-pinned only (`TICKET_471_masked_page_split.md` carries the window
recipe, incl. a one-line in-tree control arm). The page-split region is
byte-identical to upstream before and after the port; only `_flash_mla_flashinfer`
is fork-adapted (idx computed before the split). Tests:
`test/registered/unit/layers/attention/test_flash_mla_page_split_mask_471.py`,
10 hermetic, four executed can-fail arms — they run the REAL Triton kernels
through `TRITON_INTERPRET=1` on CPU, which is how an SM120-only kernel becomes
falsifiable at the desk; the compiled arm is
`test_flash_mla_backends.py::TestTouchedPageSplit` (SM120-gated, unrun).
**The sm120 decode entry point now buckets its topk width** (port of upstream
#33407, `86c2a34a45`, still open upstream): the CUTLASS decode kernels are
instantiated only for `topk in {128, 512, 1024}` (`(heads, topk)` table,
checked against the installed `flashinfer._DECODE_DSV4_DISPATCH`), while
DSpark's draft indexer emits **192** — a pre-boot crash on the 5090 the first
time a DSpark draft runs. `_flash_mla_flashinfer` right-pads the index tensor
with the kernels' `-1` skip sentinel to the next instantiated bucket and caps
the scan through `topk_length`, so the padding is allocated but never read;
anything still undispatchable (`d_qk != 512`, an uninstantiated head count, a
batch above `_DECODE_MAX_TOKENS`) routes to the existing Triton sparse-decode
kernel instead of dying. Fork deviation, stated: upstream gates the
dispatch test on `B <= _DECODE_MAX_TOKENS` because it also has a prefill
branch — this fork's entry point only ever calls the decode kernel, so the test
is unconditional here and the over-long batch becomes a fallback rather than a
crash. Tests:
`test/registered/unit/layers/attention/test_flash_mla_sm120_topk_buckets.py`,
15 hermetic, five executed can-fail arms (padding removed → 4 red; pad value 0
instead of -1 → 1; scan cap dropped → 1; dispatch check removed → 4; bucket
picks the widest instead of the smallest → 7). Unmeasured on a card.
The torch paged-MQA indexer logits are chunked on BOTH axes — KV positions
(#426) and query rows under a per-rank MiB budget (#449,
`SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB`, converted with that rank's own head
geometry as #395 does) — bit-identical to the single-pass form, collective-free
inside the loops. This BOUNDS the per-query-token duplication of the KV gather
(one copy per query token, ANALYSE_447 L1); it does not remove it, and the
speed effect is unmeasured (no GPU window taken).
**#493 made that budget BIND.** Reach note, and it is the point of the entry:
#449 shipped the knob at 2048 MiB, which is ABOVE the peak it bounds on the
geometry this fork serves — at `--chunked-prefill-size 256` one query row costs
2.27 MiB at `SEQ_CHUNK` 2048, so 2048 MiB permits 903 rows against 256 asked
for, and the cap returned the whole query axis. A present mechanism with zero
reach. The gate predicate is `rows = budget_bytes // step_bytes`
(`layers/attention/dsv4/indexer.py:347`); the default is now **256 MiB**, the
largest power of two that binds at both `SEQ_CHUNK` settings. Peak at the
window-3 geometry falls 588 → 262 MiB per rank. The transient is also NAMEABLE
at boot now (`indexer_prefill_scratch_bytes` +
`ServerArgs.dsv4_indexer_prefill_scratch_mib`, one formula shared with the loop),
and is itemized in `pinned_reserve_shortfall_note` as the sixth term no reserve
charges. GPU arm packaged, not run: `scripts/dev/493_indexer_transient/`,
falsifier = `peak_bytes_max` must fall ~326 MiB/rank between the arms.
`NOTE_493_indexer_prefill_transient.md`.
**Two DSpark pre-boot blockers closed by upstream ports (unbooted).** (1)
#33312: `DeepseekV4ForCausalLMDSpark` never resolved
`num_fused_shared_experts`, so its expert mapping covered `n_routed_experts`
alone and a checkpoint shipping the shared expert FUSED lost every such tensor
to the loader's "unexpected weight" drop — the #491 silent class, different
name family. The decision now lives in
`deepseek_v4._resolve_num_fused_shared_experts`, which the target model's
`determine_num_fused_shared_experts` delegates to, so the two cannot drift;
this fork's GGUF branch (a GGUF stream keeps the shared expert as separate
packed linears) survives the extraction, `--enforce-shared-experts-fusion`
refusal included. (2) #33098: `_fill_dp_moe_sync_metadata` filled the DP
vectors but not `num_token_non_padded`/`_cpu`, the fields the EP token
accounting reads (`layers/moe/topk.py`, `hash_topk.py`, `mega_moe.py`) — the
device tensor only under `moe_ep_size > 1` (`forward_batch_info.py:1527-1528`),
so a non-EP boot allocates nothing new. Tests:
`test/registered/unit/models/test_dspark_shared_expert_fusion_33312.py` (11
hermetic, four can-fail arms) and the extended
`test/registered/spec/dspark/test_dspark_dp_original_global_num_tokens_442.py`
(8 hermetic, four can-fail arms).
Nemotron-Puzzle class structurally covered, unbooted.

## 16. Measurement / window infrastructure
gpu-arb (UUID-based holder + heartbeat — stop the heartbeat BEFORE releasing).
It is a **convention, not an enforcement**: the code names it as such
(`registry/ledger.py:17`, `:607`, `registry/arbiter.py:1025`) and no path refuses
GPU work without a holder; the only enforced direction is `test/conftest.py:47-67`,
which fails a pytest run that WROTE the shared arb paths (#438).
forward_peak.py judges the VRAM corridor AT PEAK rather than idle — wired into
`model_runner.py:4060-4081` but **off unless `SGLANG_FORWARD_PEAK_PATH` is set**
(`forward_peak.py:150-155`), and that variable has no `environ.py` entry (#505-D11).
cachetrim with --ready-url self-retirement, which refuses a missing ready signal
with its own measured counter-number (`scripts/dsv4/cachetrim.sh:295`).
expert_stats (router distribution + hit rate). CollectiveClock gives compute vs wait
per rank for **plain-prefill forwards on the target runner only**, and only on cuda
with `pp_size == 1` (`managers/scheduler_components/metrics_reporter.py:341-352`,
`:134-136`, `:342-344`) — there is no decode/verify-round equivalent (#505-D14).
The measured-KV-budget stale-boot trap (`rigmon/kvbudget.py:16-22`, ~4x shifts from
boot order alone) applies only when the feature is switched on —
`SGLANG_MEASURED_KV_BUDGET` defaults to False (`environ.py:373`, consumed at
`uneven_perf.py:2617`); the benchmark harness clears the file per point regardless
(`planner/runner.py:203`, `:231-238`).
**Transients need a transient-rate sampler (#493):** a 1 Hz `nvidia-smi` loop
undersamples a sub-second prefill excursion by ~12x, so its minimum is a LOWER
bound and its apparent ramp is extreme-value statistics, not a signal — use
`scripts/dev/493_indexer_transient/sample_corridor.sh` (`nvidia-smi -lms`, 100 ms)
for the shape and `forward_peak`'s per-forward `nvml_free_bytes_min` for the
number (remember forward_peak itself is gated off by default per #505-D11
above — set `SGLANG_FORWARD_PEAK_PATH` first). And `--rank-auto-reserve-mib`
shapes the BUDGET, never a transient: it trades KV capacity for steady-state
free memory and moves a peak not at all (runbook §4.5.4 items 4-7 carry the
evidence, incl. "a corridor repair applies to every violating card, not the
one the briefing named").

**Work-matched counters are enforced, not advised (#523, rule from #482).**
An ACCUMULATING counter (`h2d_bytes` above all) may only be divided by another
arm's at a COMMON work point: each rank writes its #390 dump on its own 45 s
timer, so a pre-teardown read catches an arm at whatever fraction of its run the
last tick landed on, and the #439 green window's two arms sat at 96.8 % against
91.9 % — a ~5 % gap that inflated the published transfer term from 1.4307x to
1.5028x. `scripts/dev/394_s2_proof/read_arm.py --against <arm>` is now the ONLY
path in that harness to a cross-arm number (per-rank H2D delta, group delta,
transfer term, speedup); it REFUSES by name with a non-zero exit and prints no
number at all on `non-final-revision` (an arm's own ranks disagree),
`work-mismatch` (the arms sit at different work points, default tolerance 0.5 %,
which is the window's own 0.424 % A-vs-A floor rounded once and BINDS in both
directions), `missing-counter`, `rank-count-mismatch` and `link-count-mismatch`.
The single-arm readout is unchanged and prints no ratio, so a silent comparison
is unreachable. `test_work_matched_counters_523.py` (33 hermetic) pins the gate,
the refusals, and the falsifier: the same real dumps with the gate disarmed
reproduce the withdrawn 1.5028x to four decimals. What it does NOT cover, and
the negative finding is deliberate: s12's cross-arm columns are per-batch
medians and self-normalised shares, i.e. intensive, so the rule does not bind
there — what can diverge silently is the WINDOW BASIS (`punkt_fenster` falls
through to "the whole log incl. warmup" when punkte.jsonl has no request count),
and `s12_log_analyse.fenster_basis_pruefen` now names that on stderr.

**A floor round is one invocation, not three (#459, from #475 §6).**
`s12_prefill_kurve.py --floor-draws N` runs the N A-vs-A draws back to back in
ONE process after one explicitly discarded warm-up draw, and the artifact
carries the evidence for both properties rather than the claim:
`floor_series.warmup_draw` (which draw was discarded, why, and ITS OWN rate),
`discarded_draws`, per-draw `gap_before_s`, `max_gap_s`, `back_to_back` judged
against the named `BACK_TO_BACK_MAX_GAP_S`, and a summary `floor[]` with
`spread_pct` AND `monotone`. Default `1` = the pre-#459 single point,
byte-for-byte. Why: #435 sub-arm B2's three draws sat 48-51 s apart around ~12 s
of work and reported a monotone 13.0 % as a noise floor where #424 reported
3.0 % -- drift reported as noise, and loose in the direction that lets a real
regression pass as "within the floor" (#435 sub-arm B: -8.5 % scored as parity).
`monotone: true` is the flag that says the spread may not be used as a floor at
all. The GPU re-measurement is not yet run.

**The s12 decode parser reads both spec modes (#459).** `RE_DECODE` required the
accept block the scheduler writes only when speculation is ON
(`metrics_reporter.py:968-972`, `:1018`) and allowed nothing between it and the
graph flag, so a spec-off boot parsed ZERO decode ticks and s12 reported 0/0 --
and so did a spec-ON CAP_ACCEPT boot, whose `cap len:` (`:1020`) sits in
between, and any boot under `LOG_FORWARD_ITERS` (`:962`). Four missed shapes,
three of them not in the ticket. `accept_len` / `accept_rate` are now `None`,
never `0.0`, and `spec: bool` names the shape; all three tick aggregators
(s12/s14/s16) report `ticks_with_accept`.

**Break-cost probe (#494)**, `srt/utils/break_cost_clock.py`: prices ONE
CUDA-graph crossing of a breakable capture -- `segment_end -> eager slot ->
segment_start` -- with a CUDA event pair around every segment and every break
slot in `BreakableCUDAGraph.replay()`, read `SGLANG_BREAK_COST_DEFER_ROUNDS`
(default 2) rounds LATE through `query()` only, so the read never synchronises
the round it measures. Per crossing: `gap_in_ms` / `slot_ms` / `gap_out_ms`
(device wait, device compute, device wait) plus `host_ms` split into the four
terms the #462 F2 ticket names (`rendezvous`, `planning`, `publish`, `fetch`,
bracketed in `MoEExpertOffloadCache.prepare_breakable`); per round:
`compute_ms` / `wait_ms` / `span_ms` / `residual_ms`, per rank, as JSONL.
Crossings carry the break function's name, so MoE breaks are separable from
any other break point. `SGLANG_BREAK_COST_PROBE=1`, **OFF by default and
byte-neutral off** (the disabled `replay()` loop is the original one, statement
for statement: no event created, none recorded, nothing allocated per round --
pinned by a call-count spy). Reader: `scripts/dev/494_break_cost/summarise.py`.
It is route-agnostic: any `eager_on_graph` break point is priced, not only the
MoE one. `tests/moe_offload/test_break_cost_probe_494.py`, 21 hermetic, three
executed can-fail arms (crossing->segment mapping shifted by one -> 3 red;
probe default flipped ON -> 4 red incl. the neutrality spy; one host phase
bracket renamed -> 1 red). No GPU number exists yet -- the probe has never run
on a card; it is the instrument F2 was missing, not a result.

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
redesign. Cell **"#302a heat migration x CUDA graphs" is OCCUPIED, not empty**
(audit #500): both refusals key on the CAPTURABLE mode only
(`expert_heat_migration.py:339`, `expert_offload.py:3632`), while the breakable
route's `prepare_breakable` reaches `_observe_routing` →
`if self._heat.due(): self._migrate_heat()` (`expert_offload.py:3107`, `:3197`)
with `_capturable_ready` False — the swaps are in-place into address-stable
arena slots, which is exactly what a captured graph permits. The in-graph fetch remains refuted (#452) and is enforced at runtime
by `refuse_capturable_offload_decode` — both spellings
(`SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1`, `SGLANG_MOE_OFFLOAD_GRAPH_MODE=capturable`)
reach the same refusal (`layers/moe/offload_capture_gate.py:257`, `:284`). It
is NOT in `planner/rejected.py` — #500-B7 registers it there. Anything that EVICTS
consumes `DESIGN_407_memtier_registry.md` §8's one global importance ladder
(cold second model, inactive layout/graph families, cold experts, idle sessions,
active work last and never out of FCFS order — coldest-first within a class)
instead of writing a local victim policy — that ladder is now EXECUTABLE rather
than prose (`model_executor/short_term_offload_register.LadderRank` /
`plan_spill`, #286), and a class added without a ladder rank fails at import
(`if _MISSING_DESCRIPTORS: raise RuntimeError(...)`,
`short_term_offload_register.py:465-472`). `describe_class` already has a
production consumer (`layers/moe/breakable_offload.py:216`); `plan_spill`
itself has none outside the module's own `rung1_evict` (`:1486`) — a new
consumer SHOULD call it instead of restating it, but none does yet; and anything that decides at runtime
whether a path is worth its cost is an instance of `DESIGN_363_regime_controller.md`
§20.1's worth-it autocheck rather than a new flag. Read those three before
adding a cell, and register the answer back into them in the same merge.
