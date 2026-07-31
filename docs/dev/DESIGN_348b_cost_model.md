# DESIGN #348b — one cost model, many consumers

Task source: `ANALYSE_347_cross_feature_optimizations.md` §3. Three placement
planners priced the same physics — per-card compute rate × pair-matrix hop
cost — with separate code and separate assumptions. This is the refactor that
puts them on one library. **Same plans, one source of truth, discrepancies
surfaced.** Nothing here is a re-tune: every reference plan is byte-identical
before and after (§5).

Desk work, no cards (`CUDA_VISIBLE_DEVICES=99`).

---

## 1. Audit — who read what, before

`grep`-complete over the three named planners plus every module reachable from
them. Classification: **shared** = already through a common source;
**duplicate** = own reader, same intent; **divergent** = two planners would
produce different numbers for the same physical fact.

### 1.1 Per-card compute rate

| # | Site | What it reads | Source | Absent → | Class |
|---|---|---|---|---|---|
| C1 | `key_solver.py:695-699` | `membw_read_gbs or membw_gbs or 0.0` | card probe | **silent `0.0`, not in `absent`** | divergent |
| C2 | `key_solver.py:700-704` | `membw_gemv_gbs` | card probe | whole vector dropped, absence named | duplicate |
| C3 | `key_solver.py:707-712` | `gemm_bf16_tflops or gemm_tflops or gemm_fp8_tflops or 0.0` | card probe | **silent `0.0`, not in `absent`** | divergent |
| C4 | `key_solver.py:618-641` `resolve_gemm_dtype` | picks fp8 vs bf16 per rank | card probe | falls back to bf16 (documented) | duplicate |
| C5 | `key_solver.py:427-458` `gemm_dtype_for_checkpoint` | binary `"fp8"`/`"bf16"` classifier | own `config.json` read | int8/nvfp4/W4A16/MIXED → `"bf16"` | duplicate |
| C6 | `uneven_perf.py:2199` `rank_gemm_scores` | per-card lane resolution | hardware profile | dense bf16 + **loud** warning | shared (the good one) |
| C7 | `uneven_perf.py:2324` `rank_gemm_family_scores` | #324 (rank, family) widening | hardware profile | inherits C6 | shared |
| C8 | `uneven_perf.py:4874/4971` | C6/C7 inside the `--rank-perf-tune` recommender | hardware profile | as C6 | shared |
| C9 | `shard_plan.py:116-128` `RateTable.ms` | measured ms per (stage, card, resolution) | P1 bench table | **loud** `MissingRateError` | shared (own axis) |
| C10 | `shard_plan.py:565` | `1000 / full_ms` as the card weight | derived from C9 | n/a | duplicate |
| C11 | `shard_plan.py:497-503` | `rate_scale` under an LLM co-tenant | declared, default `1.0` | warns, then **plans at full rate** | duplicate |
| C12 | `sp_shard_utils.py:135-160` | `SGLANG_SP_CAPACITY_WEIGHTS` env vector | opaque, trusted | unset → silent equal split (by design) | duplicate |
| C13 | `class2_diffusion.py:244` `_capacity_weights` | `load_measured_registry(None)["<uuid>"]["gemm_tflops"]` | **wrong artifact** | always raises | **divergent (bug)** |

Note what C13 is: the brief's premise was that `sp_shard_utils` already
consumes `uneven_perf.rank_gemm_scores`. It does not, and neither does its
actual weight producer. See §3.1.

Also note what C6-C8 versus C1-C5 means: `key_solver.py` **never calls**
`rank_gemm_scores`, `rank_gemm_family_scores`, `family_prefill_tflops` or
`effective_prefill_tflops` (grep: zero hits). The #324 per-family widening
reaches the boot-time recommender and not the offline solver.

### 1.2 Hop / collective pricing

| # | Site | What it prices | Source | Absent → | Class |
|---|---|---|---|---|---|
| H1 | `key_solver.py:751-780` | narrowest / worst ordered pair | `probe["pairs"]` (ordered) | named absence, no invention | duplicate |
| H2 | `key_solver.py:392` `_ring_factor` | `2(R-1)/R` | formula | n/a | duplicate |
| H3 | `key_solver.py:989-994` `collective_decode_s` | `(R-1)·lat + ring·payload/BW` | H1 × `COLLECTIVE_EFFICIENCY=1.0` | returns `None` — correct | duplicate |
| H4 | `key_solver.py:1428/1431/4328` | link for the prefill **ratio** | `link_bw_gbs or 1e-3` | **fictitious `1e-3` GB/s** | divergent |
| H5 | `uneven_perf.py:4489` `_prefill_sharded_time` | `t_comm` | `max(min_link_gbs, 0.1)` | **floor `0.1` GB/s** | divergent |
| H6 | `uneven_perf.py:4911` | `min(pair_bws) if pair_bws else 8.0` | `profile["links"]` | **`8.0` GB/s assumed** | divergent |
| H7 | `lever_profiles.py:302/382` `_FALLBACK_LINK_GBS` | same question | card probe → profile → assumed | **`8.0` GB/s assumed** | divergent |
| H8 | `uneven_perf.py:4709` `pair_bw` | one pair | `profile["links"][a\|b]["p2p_gbs"]` | `None`, filtered | duplicate |
| H9 | `spread.py:493` `pair_bandwidth_from_probe` | ordered pairs by gpu index | `probe["pairs"]` | row skipped | duplicate |
| H10 | `rig_profile_source.py:260` | pairs → artifact measurements | `probe["pairs"]` | row skipped | duplicate |
| H11 | `shard_plan.py` — **whole file** | nothing | — | — | **absent by design** |
| H12 | `sp_shard_utils.py:255-293` `gather_seq` | max-pad all-gather | — | **cost documented in prose, never priced** | absent |

Six independent readers over two on-disk shapes (H1/H9/H10 read
`probe["pairs"]`; H6/H8 read `profile["links"]`), and no code reconciles them.

---

## 2. Divergences found

### D1 — `_capacity_weights` reads the wrong artifact (real bug, fixed)

`class2_diffusion.py:244` called `uneven_perf.load_measured_registry(None)`.
That function is the measured **KV-budget** registry: gated behind
`SGLANG_MEASURED_KV_BUDGET` (off by default, → `None`), shaped
`{"components": [...], "mlp_vector": [...]}` rather than keyed by card UUID,
and it dereferences `server_args.tp_size` on the argument this caller passes as
`None`. `registry.get(card)` could not return a per-card `gemm_tflops` under
any configuration.

Effect: the measured branch raised `EstimateError` on **every** call. Uneven SP
was reachable only by declaring `launch.capacity_weights` by hand, while the
docstrings in `class2_diffusion.py:220` and `sp_shard_utils.py:36` both claimed
`rank_gemm_scores` was the source. No test covered it
(`test_multi_card_diffusion_needs_opt_in_and_says_so` stops at the opt-in
check).

Fixed: routed through `cost_model.compute_rates_for_cards`, which reads
`get_cached_hardware_profile()["gpus"][uuid]` and resolves the lane through
`rank_gemm_scores` — the source the docstring always claimed. Covered by three
new cases in `test_adapters.py`.

This is the **only intentional behaviour change** in #348b. It cannot move a
plan, because the path it repairs produced no plan: it raised.

### D2 — three different link rates for one absent measurement

For the identical condition "no pair matrix was measured", the fork substitutes:

| Value | Site | Feeds |
|---|---|---|
| `1e-3` GB/s | `key_solver.py:1428/1431/4328` | prefill ratio → `raw["enc"]` → `value_of()` ranking |
| `0.1` GB/s | `uneven_perf.py:4489` (floor) | `_prefill_sharded_time` |
| `8.0` GB/s | `lever_profiles.py:302`, `uneven_perf.py:4911` | lever ranking + auto-performance |

**80x apart.** Whether that is safe under the #216/#264 guard depends on the
consumer, and the two cases differ:

* **argmax / Pareto dominance: safe.** The collective term is
  `n_layers · hidden · ranks` — none of which a candidate split varies — so it
  is an additive constant across candidates and cannot reorder them. Verified
  as arithmetic, not asserted, in
  `test_the_absent_link_placeholder_cannot_reorder_candidates`.
* **ratio against a threshold: NOT safe.** `lever_profiles._speed_ratios`
  (`lever_profiles.py:573-591`) divides two predicted times and compares the
  result against a move threshold. An additive constant does not survive a
  division: at 8.0 GB/s and at 0.1 GB/s the same two candidates yield different
  ratios on identical measured inputs, and a lever can clear the threshold in
  one case and not the other. Pinned by
  `test_a_ratio_is_not_invariant_and_that_is_the_open_risk`.

The library named both values (`ABSENT_LINK_RANKING_PLACEHOLDER_GBS`,
`ABSENT_LINK_ASSUMED_GBS`) rather than picking a winner: unifying them would
have changed plans on an unprobed rig, which is a re-tune. **Closed by #359**
— see §6.1: no stand-in rate exists any more, an absent link means the
collective term is not priced, and a compute-only figure may order candidates
but may not settle a threshold.

### D3 — two on-disk pair-matrix shapes, never reconciled

`probe["pairs"]` is a list of **ordered** rows with bandwidth, latency and
transport, produced by the card probe's pinned-transfer measurement.
`profile["links"]["<uuidA>|<uuidB>"]` is a single **unordered** row with
`p2p_gbs` only, produced by an NCCL send/recv, plus a `__group__` all-reduce
timing that is not per-pair at all. Different transports, different methods,
same wire. Nothing compared them.

The reference rig makes the asymmetry concrete: `GPU-3080a → GPU-5090` measures
4.52 GB/s and `GPU-3080b → GPU-5090` 6.88 GB/s — x4 against x8 (EVAL_272 §1.1).
The unordered shape cannot express that at all.

`reconcile_pair_matrices` now merges both, prefers the ordered artifact (it
measures the direction a collective actually takes), and reports every
disagreement beyond 10 % instead of averaging it away.

### D4 — silent `0.0` against the module's own contract

`RigRates`' docstring (`key_solver.py:584-586`): *"Nothing here is ever
defaulted to a plausible number."* `rates_from_probe` nevertheless defaulted a
missing `membw` (C1) and a missing `gemm_tflops` (C3) to `0.0` without
recording them in `absent`, unlike gemv/link/h2d which are properly named. The
`0.0` is then clamped to `1e-9` at `uneven_perf.py:4238` and raised to a
fractional exponent — so a genuinely absent measurement reads downstream as an
extremely slow but *valid* card. Both now produce a named absence.

### D5 — two rounding rules, deliberately kept apart

`shard_plan._weighted_boundaries` rounds cumulatively; `sp_shard_utils._apportion`
apportions by largest remainder. They disagree:

| weights | total | Hamilton | cumulative |
|---|---|---|---|
| `(1, 3)` | 10 | `[3, 7]` | `[2, 8]` |
| `(1, 1, 1)` | 10 | `[4, 3, 3]` | `[3, 4, 3]` |

Both are correct for their consumer — the video planner needs contiguous
boundaries on a shared timeline, the SP split needs counts — and both preserve
the total exactly. Unifying them would move a shipped plan. Both now live in
the library, side by side, with the divergence pinned as a fact
(`test_the_two_rounding_rules_disagree_and_the_difference_is_pinned`) so a
future change to it is a decision rather than an accident.

### D6 — two planners price no hop cost at all

`shard_plan.py` has no communication concept in its type system. That is
defensible: cards never exchange frame data, each re-decodes its own overlap
frame, and the overlap is priced as recompute. Recorded, not changed.

`sp_shard_utils.gather_seq` is different — it runs a real max-padded all-gather
whose byte volume grows with the skew of the very split `_apportion` chooses,
documents that cost in prose (`sp_shard_utils.py:270-273`), and never prices
it. The compute-optimal split can be communication-suboptimal and nothing
would know. Deferred to M4 by `DESIGN_333_M3_diffusion_lane.md:123`; **open
item, §6**, now with a library primitive (`allreduce_seconds` + a measured
`PairMatrix`) ready for it.

---

## 3. The library

`python/sglang/srt/planner/cost_model.py`. Stdlib-only at module scope;
`uneven_perf` (and through it torch) is imported lazily inside the functions
that need a measured profile. Measured on this box: **4 ms** on top of a warm
`sglang` import, and `sglang.srt.uneven_perf` stays unloaded.

That required one supporting change: `planner/__init__.py` re-exported
`PlanInputs` eagerly, which pulled torch into every
`import sglang.srt.planner.<anything>` (~7 s). It is now a PEP 562
`__getattr__`; the public name is unchanged and nothing in the tree imported it
from there.

### 3.1 Public surface

```
Provenance   Provenance{MEASURED,ESTIMATE,ABSENT}, Rate, AbsentRate
             Rate.value is None iff provenance is ABSENT (enforced in __post_init__)
             Rate.require(name) -> float, else AbsentRate carrying the reason
             Rate.or_none()     -> float | None

Compute      ComputeRates{rates, keys, fmt, families, warnings}
               .values(family=None) -> [float]   (raises on an absent card)
               .weights(family=None)-> [float]   (normalised to 1.0)
               .for_family(family)  -> (Rate,)   (#302 entry)
               .absences()          -> [str]
               .mixed               -> bool
             compute_rates_from_entries(entries, keys, fmt=, family_formats=)
             compute_rates_for_cards(uuids, fmt=, family_formats=, profile=)
             memory_rates_from_entries(entries, keys, "membw"|"gemv"|"h2d"|"d2h")

Hop          Hop{src, dst, bandwidth_gbs: Rate, latency_us: Rate, transport}
             PairMatrix{hops, keys, source, rejected, notes}
               .hop(src, dst) / .transports()
               .narrowest_bandwidth_gbs() -> Rate
               .worst_latency_us()        -> Rate
               .absences()                -> [str]
             pair_matrix_from_card_probe(probe, keys, uuid_of_key=)
             pair_matrix_from_hardware_profile(profile, keys, uuid_of_key=)
             reconcile_pair_matrices(*matrices, tolerance=0.10) -> (PairMatrix, [str])

Composition  ring_factor(ranks)
             allreduce_seconds(payload_bytes, ranks, bw_gbs, lat_us, efficiency=1.0)
             apportion_largest_remainder(total, weights, min_one=True)
             cumulative_boundaries(total, weights)
             apportion_cumulative(total, weights)
             ABSENT_LINK_COMPUTE_ONLY_REASON    (#359: no stand-in rate exists)

Bundle       CostSources{compute, links, keys, divergences}.absences()
             load_cost_sources(card_keys, fmt=, family_formats=,
                               card_probe=, hardware_profile=, uuid_of_key=)
```

### 3.2 Rules the library enforces

1. **Measured or named.** No fourth tier. An absent rate has no value and
   carries the reason; `require()` raises with it. Compute rates are never
   re-derived — `rank_gemm_family_scores` (#324) is the resolver, wrapped for
   provenance, not reimplemented.
2. **The roofline never ranks a split** (#216/#264). One documented
   split-invariant placeholder exists for ratios; absolute numbers stay absent.
3. **A hop is between two different cards.** Same-card rows are rejected in
   both on-disk shapes and counted in `PairMatrix.rejected`. The card probe
   structurally cannot emit one (`card_probe.py:775` skips `a.uuid == b.uuid`),
   but the old reader had no guard against a hand-edited or foreign artifact,
   and a device-local copy rate would win `min(bandwidth)` and make every
   collective look free.

### 3.3 #302 entry point (documented, not implemented)

Expert placement needs exactly the two axes and nothing else:

* per-expert compute rate — `ComputeRates.for_family(uneven_perf.GEMM_FAMILY_MOE)`.
  The #324 widening already resolves a MoE family onto its own lane per card,
  so a rig where the experts run on native fp8 on one card and Marlin on the
  next scores them apart without #302 deriving anything;
* the all-to-all hop — `PairMatrix.hop(src, dst)` for a directed route or
  `narrowest_bandwidth_gbs()` for the group bound, composed with
  `allreduce_seconds` (dispatch + combine are two collectives of the token
  payload).

`load_cost_sources()` returns both, resolved and provenance-tagged. #302 adds
only its own objective. If it reaches into `profile["gpus"]` or
`probe["pairs"]` directly, the library is missing a primitive and should grow
one. Exercised by `test_the_302_entry_point_resolves_a_moe_family_and_a_hop`.

---

## 4. Migration

| Consumer | Before | After |
|---|---|---|
| `key_solver._ring_factor` | own `2(R-1)/R` | `= cost_model.ring_factor` |
| `key_solver.collective_decode_s` / `_prefill_s` | inlined `(R-1)·lat + ring·payload/BW` | `cost_model.allreduce_seconds(...)` |
| `key_solver.rates_from_probe` pair block | own 25-line parser over `probe["pairs"]` | `pair_matrix_from_card_probe` + `narrowest/worst` |
| `key_solver.rates_from_probe` membw/gemm | silent `0.0` | absence named via `memory_rates_from_entries`; carried as `None` since #359 |
| `key_solver` link fallback | bare `1e-3` ×3 | none — the collective term is not priced (#359) |
| `shard_plan._weighted_boundaries` | own cumulative rounding | `cost_model.cumulative_boundaries` |
| `sp_shard_utils._apportion` | own Hamilton apportionment | `cost_model.apportion_largest_remainder` |
| `class2_diffusion._capacity_weights` | `load_measured_registry` (dead) | `compute_rates_for_cards` (D1 fix) |

---

## 5. Byte-identical-plan proof

Harness: every reference plan surface dumped as canonical sorted JSON, run once
against HEAD `5fa03e1664` and once against the branch, diffed.

**158 plan keys, zero differences.** Covered:

* `rates_from_probe` over 4 rank mappings × 2 probes (full pair matrix and
  none) × 3 dtype resolutions — every rate vector, both link figures;
* `solve()` over 2 real checkpoints on disk (`Qwen3.6-27B-FP8`,
  `Qwen3.6-27B-Q3_K_M.gguf`) × 2 probes × every goal in `GOALS`: units, launch
  flags, feasibility, all `raw` values to 12 dp and all `predictions` to 9 dp;
* the 2-goal **Pareto front** including the knee pick, on both probes — the one
  place the absent-link constant could plausibly have moved a choice;
* `check_regressions` on both checkpoints — the committed measured anchors,
  end to end through rates → cost model → prediction;
* `compare_plans` + `capacity_weighted_plan` over 4 card sets × 2 fps
  multipliers × 7 frame counts: strategy, every chunk `[card, start, stop,
  lead, tail]`, rate scales, and the predicted makespan to 9 dp;
* `_weighted_boundaries` swept over 9 weight vectors × 11 totals;
* `_ring_factor` over 0..8.

Per consumer:

* **K1 key solver** — the 56 `solve`/`front`/`regressions` keys above are
  identical, and `test_key_solver.py` (unmodified, 145 cases) stays green.
* **Video `shard_plan`** — the 112 `compare_plans`/`capacity_weighted_plan`
  keys are identical, and `test_shard_plan.py` (unmodified) stays green.
* **Diffusion `sp_shard_utils`** — its own suite cannot run in this venv
  (`diffusers` and `addict` are absent), so the proof is structural: the
  migrated `_apportion` is a pure one-line delegation (see the diff), and the
  library function is checked against a frozen verbatim copy of the pre-#348b
  algorithm over 187 (weights, total) combinations in
  `test_the_hamilton_rule_matches_the_code_it_replaced`. Same for
  `_weighted_boundaries` (198 combinations) and `_ring_factor` (33).

The absence naming added in D4 does **not** move the reference plans: the
reference probe measures every card, so both new `absent` lists come back
empty. The dump records `absent_n` per configuration and it is unchanged.

## 5.1 Test counts

| Suite | Before | After |
|---|---|---|
| `test/registered/unit/planner/` + `video_enhance/` + `registry/` | 2294 passed, 72 skipped | **2329 passed, 72 skipped** |
| of which new: `test_cost_model.py` | — | 32 |
| of which new: `test_adapters.py` (Class-2 capacity weights) | — | 3 |

Zero regressions, zero newly-skipped. `ruff check` clean, `black`/`isort`
clean, `codespell` clean, `mypy --ignore-missing-imports` clean on
`cost_model.py`.

---

## 6. Open items

Items 1-4 were closed by **task #359**; item 5 is unchanged, and two new
remainders are recorded honestly at the end.

1. **D2 — the ratio consumer ranks through an unmeasured constant. RESOLVED
   (#359).** There is no fallback link rate left in the tree. `1e-3`
   (`ABSENT_LINK_RANKING_PLACEHOLDER_GBS`), the `0.1` floor inside
   `PerfCostModel._prefill_sharded_time` and `8.0`
   (`ABSENT_LINK_ASSUMED_GBS` / `lever_profiles._FALLBACK_LINK_GBS`,
   `apply_auto_performance`) are all deleted. `prefill_time_model` now takes
   `min_link_gbs: Optional[float]`, where `None` means the collective term is
   not priced and the result is a compute-only time, and a non-positive rate
   raises instead of being floored. A compute-only figure may settle an argmax
   (the omitted term is split-invariant — proved as arithmetic in
   `test_omitting_the_collective_term_cannot_reorder_candidates`) and may not
   settle a ratio against a threshold: `lever_profiles._speed_ratios` returns
   `prefill=None` with `cost_model.ABSENT_LINK_COMPUTE_ONLY_REASON`, the
   `max prefill` objective reports itself unresolved, and `key_solver`'s
   `enc` cell reports the absence rather than a magnitude. Sites:
   `cost_model.py` (constant block), `uneven_perf.py:4433-4550, 4949-4966`,
   `lever_profiles.py:355-410, 553-640, 810-820`,
   `key_solver.py:1565-1615, 4494-4535`.
2. **`key_solver` does not use the #324 lane resolution. RESOLVED (#359).**
   `build_cost_model` calls `uneven_perf.checkpoint_compute_format_families`
   and `RigRates.resolve_gemm_format`, which routes through
   `cost_model.compute_rates_from_entries` and thus through
   `rank_gemm_family_scores`. `rates_from_probe` gained an optional
   `hardware_profile` argument: the card probe measures only dense bf16 and
   native fp8, so the v3 `gemm_lanes` map is what lets an int8 / nvfp4 /
   W4A16 checkpoint be priced on the lane it dispatches to.
   `solver_api.cached_hardware_profile()` supplies it on the live path.
   Measured before/after on the #264/#265 fixture: with the card probe alone,
   every plan for bf16, fp8 and int8 is byte-identical (int8 takes the loud
   dense fallback — same number, now labelled). With the profile's lanes,
   bf16 is unchanged; fp8 keeps its key (`47,8,13` / `1,0,0`) and only its
   reported prefill ratio moves (1.078439 → 1.087020, 1.234660 → 1.252042)
   because both 3080s move from the dense fallback to their measured
   `fp8_marlin` lane; int8 moves the prefill key `104,15,17` → `108,15,13`
   because the `int8_native` lane separates the two 3080s by 10 % (183.78 vs
   164.77) where dense bf16 had them 0.03 % apart. Pinned by
   `test_cost_model_open_items.TestSolverUsesTheLaneResolution`.
   `gemm_dtype_for_checkpoint` and `resolve_gemm_dtype` remain exported and
   tested; they no longer price a plan.
3. **D3 — the two pair-matrix shapes had a reconciler and no caller. RESOLVED
   (#359).** `cost_model.load_pair_matrix` is the single boundary: it reads
   both artifacts, prefers the ordered card probe, and returns every
   disagreement beyond 10 % as a line the caller surfaces.
   `key_solver.rates_from_probe`, `lever_profiles._min_link_rate` and
   `load_cost_sources` all go through it; `RigRates.divergences` carries the
   disagreements, which are not absences and are not averaged away. Both
   readers now record a malformed row in `PairMatrix.rejected` instead of
   dropping it with a bare `continue`.
4. **D4 — silent `0.0` for a missing membw / GEMM rate. RESOLVED (#359).**
   `RigRates.membw_gbs` and `RigRates.gemm_tflops` are
   `List[Optional[float]]`; a card the probe never scored carries `None` all
   the way to the consumer. Every consumer goes through
   `require_membw_gbs()` / `require_gemm_tflops()`, which raise
   `cost_model.AbsentRate` naming the rank. On a complete probe both lists are
   full and nothing moves.
5. **`sp_shard_utils` cannot be tested in the desk venv** (`diffusers`,
   `addict`). Its 34 existing cases were not run for this change; the
   equivalence proof is structural (§5).

Still open, unchanged by #359:

6. **D6 — the diffusion all-gather is still unpriced.** `allreduce_seconds`
   plus a measured `PairMatrix` is what M4 needs; the split objective currently
   optimises compute only.
7. **`shard_plan` plans at full rate under a flagged co-tenant**
   (`shard_plan.py:497-503` warns, then uses `rate_scale = 1.0`). The honest
   shape is a named absence; it belongs with #348a's co-tenancy measurement.
8. **The nvfp4 lanes have no probe.** `_FORMAT_LANES` registers
   `nvfp4_native` / `nvfp4_marlin` for dispatch order only, so an NVFP4 or
   W4A16 checkpoint still resolves to the loud dense fallback on every card.
   Item 2 routes it correctly; there is nothing measured at the other end yet.
9. **`gemm_lane_entries` merges two artifacts by UUID.** The card probe wins
   the lanes it measures (its numbers are what the K1 regression anchors were
   fitted against) and the hardware profile contributes the rest. A >10 %
   disagreement on the shared `fp8_native` lane is reported in
   `RigRates.divergences`, not resolved — the two probes ran in different
   thermal windows and picking one silently would be a re-tune.
