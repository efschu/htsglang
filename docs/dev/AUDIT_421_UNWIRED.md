# Audit #421 — built but not wired

Desk audit, no GPU. Base commit `33148dbe0f` (integration/r3-probe-next2),
audit branch `audit/unwired-sweep-421`. Upstream reference for every
fork-vs-inherited decision: ref `upstream/main` = `7e509f690e` (2026-08-01),
one day older than the audited head, so the fork delta is genuinely the
fork's own code and not upstream drift.

The class of defect under audit: code that exists, compiles, and passes its
unit tests, but that no production call path reaches — or that is reached
only with defaults because the wiring that would supply real values was never
written. The fork has shipped this twice knowingly (#197, #394) and the test
suite did not see either, because the tests called the feature directly.

## 1. Method

Four detectors, applied to fork-delta code only.

**A — flag/env consumer classification.** Enumerate the fork's own CLI
surface as a set difference against upstream rather than by grepping for
`add_argument` (only 27 of the fork's flags are declared that way; the rest
are generated from the `ServerArgs` dataclass by
`arg_groups.arg_utils.add_cli_args_from_dataclass`). Same for `SGLANG_*`
names in `python/sglang/srt/environ.py`. Then count references per item,
partitioned into declaration site / production / test / docs.

- `ServerArgs` fields: fork 597, upstream 457 → **165 fork-added**.
- `environ.py` entries: fork 556, upstream 525 → **96 fork-added**.
- `SGLANG_*` referenced in `srt/` but absent from `environ.py`: **426**
  (mostly inherited; triaged separately, see §6).

**B — always-default parameter.** For a callable whose signature carries a
feature parameter, collect every call site and classify it as "omits",
"passes literal None", or "passes a real expression". No production call site
in the third class means the feature parameter is inert. This is the #394
shape.

**C — asymmetric gate.** For an escape-hatch flag with real consumers, check
whether the entities its own help text claims to cover actually consult it.
A consumer *count* cannot see this defect, which is the point. This is the
#197 shape.

**D — module reachability.** For each fork-only production module, resolve
every top-level public definition against a token index of the whole tree,
partitioned production/test, and rank modules by the fraction of their public
API that nothing in production references.

Two refinements are load-bearing and were only found by running the
calibration:

- **Forwarding taint (B).** A call site that passes `foo=foo` proves nothing
  when `foo` is the enclosing function's own always-default parameter; it
  moves the question one frame up. B iterates a fixed point over that
  relation. Without it, #394 is classified WIRED.
- **Docstring stripping (B2).** Matched against raw source,
  `resolve_host_shard_ratio` reads as `has-measured-source` purely because its
  docstring says "an explicit, measured vector always wins" while every actual
  source was a nameplate query. B2 unparses the AST with docstrings removed.

Everything here is static analysis. Nothing was executed, no test of the
audited code was run, no server was booted; the only tests run are this
audit's own pins.

The four detectors are committed under `scripts/dev/audit_421/` with a README
covering invocation and the calibration contract, so this table can be
regenerated against a later tip. A finding list without a runnable detector
ages into folklore.

### 1.1 Verification discipline

Per the #381 lesson (a first-pass name-grep produced 113 false hits), no
detector output is reported as a finding until the call path has been traced
by hand. The raw-to-verified ratio is recorded in §5 precisely so the next
person knows not to trust the raw numbers.

### 1.2 Known blind spots of these detectors

Found while eliminating false positives, and worth writing down because each
one is a place a future defect can hide from this audit:

- **Computed attribute names.** `SpillBudgetConfig.from_server_args`
  (`managers/kv_session_offload.py:1306`) reads its ten fields through
  `getattr(sa, "kv_session_offload_" + name, default)`. Ten flags therefore
  look test-only to any literal-name search while being correctly wired.
  Detector A cannot see through string concatenation.
- **Validation-only consumption inside `server_args.py`.** Several flags are
  consumed entirely at argument time and never appear downstream — sometimes
  legitimately (they mutate another field, e.g. `--enable-weights-disk-backup`
  flips `load_format` to `"hibernate"`), sometimes not (F2 below). The
  distinction requires reading, not counting.
- **Env-var bridges.** `--collective-net-small` reaches
  `barlink_ucx.py` only by `server_args` writing `os.environ`. A
  flag→consumer search that does not follow the env hop reports a false
  positive.
- **Dynamic registration.** `registry/adapters/class2_diffusion.py` is loaded
  by `registry/adapter.py:load_builtin_adapters()`; Triton kernels are invoked
  as `kernel[grid](...)`. Both defeat plain call-graph reasoning.
- **Standalone operator tools.** `mem_cache/hicache_migrate.py` has no
  importer by design — it is a `python -m` CLI. Not a finding.

## 2. Calibration verdict

The two known cases are fixed at the audited head, so the honest test is
whether the method rediscovers the *pattern* independently rather than
whether it can cite the issue numbers.

**Calibration 2 (#394, always-default / degenerate derivation): PASSED, and
passed twice over.** Detector D surfaced `layers/moe/cold_tier_shm.py`
unprompted — 17 public definitions, 14 with no production reference, no
production importer at all. That is the #394 reachability-slice-1 primitive,
recovered from the code alone. More convincingly, the same detector pair then
found *two previously unrecorded* instances of the identical shape (F1 and F2
below), which is the stronger evidence: the method is not merely
re-recognising a case it was told about.

**Calibration 1 (#197, asymmetric gate): PASSED — but only by Detector C.**
`SGLANG_GGUF_DENSE_VOCAB` had, and has, many production consumers, so every
consumer-count and module-reachability method classifies it WIRED: **the
Detector A/D sweep recorded in §3–§6 would not have found #197.** Detector C,
run against `ef6f8bc0a2^`, does find it, and finds it without being told the
answer:

1. The gate's own docstring (`model_loader/gguf_qwen35.py:49-62`) names both
   halves on one line — "embed_tokens/lm_head are built with the GGUF
   quant_config". The detector harvests `{embed_tokens, lm_head}` from that
   text because both also exist as assignment targets in the tree.
2. `embed_tokens` reaches the gate (`models/qwen3_5.py:1391-1399` branches on
   `gguf_dense_vocab()` before constructing it).
3. `lm_head` does not: `models/qwen3_vl.py:1280/1288` constructs
   `ParallelLMHead(quant_config=quant_config)` and never mentions the gate.
   The site is in scope because `Qwen3_5ForConditionalGeneration` inherits
   that constructor — the inheritance edge is exactly what makes the two sites
   structurally parallel.

At HEAD the same command reports no strong candidate: `ef6f8bc0a2` added the
`lm_head_quant_config = None if gguf_dense_vocab()` branch, so `lm_head` now
has an honoured site. Correct negative.

**Calibration 2 (#394): PASSED on both halves, by three independent routes.**

- Detector D surfaced `layers/moe/cold_tier_shm.py` unprompted (§3, F4).
- Detector B at `d71e7133d2` classifies the apportionment chain TEST-ONLY: the
  only two callers of `cold_shard_context()` in the whole tree are two lines
  of a unit test, so every `cold_shard=` argument in production is absent.
  **The forwarding-taint pass is what makes this work** — a naive per-call-site
  counter calls `plan_load_time_staging.cold_shard` WIRED because
  `expert_offload.py:2358` does pass `cold_shard=cold_shard`, but that
  argument is the enclosing function's own always-default parameter, so it
  only moves the question one frame up. The untainted version was run first
  and printed WIRED: without the fixed point over the forwarding relation the
  detector would have missed #394.
- Detector B2 catches the *degenerate derivation* half. At `d71e7133d2` the
  only non-env source of the per-rank weight vector was
  `nvmlDeviceGetMaxPcieLinkGeneration` / `…MaxPcieLinkWidth` — the card's
  nameplate maximum, not the trained link. Every element is the same pure
  function of a per-device capability constant, so on any rig whose cards
  share a nameplate the vector is constant, `_normalize_weights` yields
  `(1/n, …)`, `is_equal` is true, `cold_shard_context()` returns `None`, and
  the feature disables itself. B2 reports the *precondition* ("constant on any
  rig whose devices share a nameplate"), not a runtime value. At HEAD:
  0 rows, both reclassified `has-measured-source` — `a2b21c2880` put the
  measured H2D probe ahead of the nameplate.

More convincing than either calibration: the same detector pair then found
*previously unrecorded* instances of both shapes (F1, F2, F8, F9). The method
is not merely re-recognising cases it was told about.

**The transferable lesson: consumer counting finds absent wiring, never
partial wiring.** A codebase with escape-hatch flags needs the doc-coverage
check (C) and the forwarding-taint check (B) as separate passes; neither falls
out of a reference count.

The general lesson: **consumer counting finds absent wiring, never partial
wiring.** A codebase with escape-hatch flags needs the doc-coverage check as
a separate pass.

## 3. Findings

| # | Item | Class | Evidence | Sev | Fix |
|---|------|-------|----------|-----|-----|
| F1 | `--kv-pressure-ladder auto` | INERT (hard-fails) | `build_ladder_from_server_args(sa, *, table_fn=None)` raises on `auto` when `table_fn is None`; the sole production caller `managers/kv_pressure_runtime.py:461` omits it. `planner/kv_ladder_table.build_ladder_table` exists and is tested but has zero production callers. | High | S |
| F2 | `--lane-offload-profile` / `--lane-offload-class-policy` / `--lane-offload-park-targets` | INERT | `server_args._handle_lane_offload_register` validates then discards ("recomputed at configure time"). The configure-time entry `configure_global_register` has no production caller — only `test_offload_movement.py:640`. `get_global_register()` instead builds a bare `OffloadRegister()` on the default latency profile. | High | S–M |
| F3 | #309 runtime drafter attach/detach | DEAD | `speculative/runtime_draft.py` (9 public defs, 65 tests, `docs/dev/TASK_309_RUNTIME_DRAFT.md`) has no production importer. Sibling `speculative/draft_selection.arms_from_server_args` likewise test-only. | High | L |
| F4 | #394 cold expert tier | INERT (declared) — **FIXED**, see §7 | `layers/moe/cold_tier_shm.py`: no production importer. Self-declared in the merge message as "inert pending reachability slice 2" — recorded here so the state is visible to the suite, not only to git log. | Medium | L |
| F5 | `layers/fused_qk_norm.py` | DEAD | Triton fused Q/K RMSNorm ported from ATOM. No importer anywhere in the tree. The similarly-named live path is `layers/fused_qk_norm_rope_store.py`, used by `models/deepseek_v4.py`; the two are unrelated. | Low | S (delete) |
| F6 | #407 memtier registry | **FIXED (#286 + #410)** — was INERT, see §7 | `srt/memtier/` (registry, tiers, probe, profile, reservations) had zero production importers and zero production symbol references outside its own package. `SGLANG_MEMTIER_PROFILE` is read at `memtier/profile.py:361`, on a path a serving process never executes. Found independently by module reachability and by the env triage. | Medium | L |
| F7 | `SGLANG_BARLINK_PP_TRANSPORT` | DEAD | Present only as a successor value in the retired-name guard (`barlink_env_guard.py:122`) and in a design doc. Not among the `_e()` suffixes read by `barlink_matrix.load_config`. The retired predecessor `SGLANG_HTCCL_PP_TRANSPORT` was introduced by the rename commit `6a5f307260` itself and never had a reader either. | Low | S |
| F8 | `presplit_expert_offload_after_repack(cold_shard=…)` — the #394 policy's *second* load-time door | TEST-ONLY | `layers/moe/expert_offload.py:3086`, 14 call sites; the only one passing a real value is a unit test. The four production callers (`fp8.py`, `gptq_moe.py`, `awq_moe.py`, internal) all omit it, and the docstring concedes it: *"With no `cold_shard` (every caller today: fp8.py, gptq_moe.py, awq_moe.py)"*. `a2b21c2880` wired the GGUF door (`fused_moe_triton/layer.py:1460,2505`) and not this one, so link-proportional cold sharding is live for GGUF MoE only — while the merge message claims both halves "take their layout from ONE plan object". | Medium | S, with a caveat: the docstring says `cold_shard` is only legal on a layer that shards experts on dim 0, so the right fix may be an explicit refusal rather than a wiring. |
| F9 | `PathProfile.transport_hint` | INERT | `distributed/device_communicators/barlink_path_dispatcher.py:92`. Ten construction sites, four of them production (`barlink_path_rates.py:81,217,304,400`); none passes `transport_hint`. It *is* read in production (`:222`, `:456`), so the barlink `_select` actuation hook is always handed `None`. The field comment concedes it: *"Paths without a hint can win a comparison but not (yet) be acted on there."* | Low–Medium | M |

Severity is "what does an operator lose": F1 and F2 are advertised in CLI
help text and silently do nothing (F2) or crash late (F1), which is why they
outrank F4 even though F4 is the larger body of unreached code.

## 4. Detail

### F1 — `--kv-pressure-ladder auto` cannot be used

`server_args.py:4715` advertises: *"'auto' = the step table is computed once
from the rig/model profile by the #272 planner."* `parse_kv_pressure_ladder`
accepts `auto`, and `_handle_kv_pressure_ladder` applies its extra checks only
to tuple specs, so argument time passes.

`model_executor/kv_pressure_ladder.py:1926` then documents the injection that
was intended:

> `table_fn` is the `auto` path's table source (the #272 planner's
> `build_ladder_table`, injected so this module never imports the planner).
> `auto` without a table source is a hard error rather than a silent fallback
> to a placeholder ladder.

The refusal is deliberate and correct. What is missing is the injection:
`managers/kv_pressure_runtime.py:461` calls
`build_ladder_from_server_args(server_args)` with no `table_fn`, and it is the
only production construction site. So `--kv-pressure-ladder auto` is a
guaranteed `ValueError` at scheduler construction.

Fix (S): import `planner.kv_ladder_table.build_ladder_table` at that call
site and pass it as `table_fn`. The indirection the docstring asks for is
preserved — `kv_pressure_ladder.py` still never imports the planner.

### F2 — the lane-offload profile never reaches the register

`server_args.py:6525` is explicit that validation throws its results away:

> Pure validation — the register itself is built at runner init (and only
> with `SGLANG_OFFLOAD_REGISTER=1`). […] the resolved values are discarded
> here (recomputed at configure time).

`configure_global_register(profile, class_policy_overrides, …)` is that
configure-time entry, and says so: *"Build (or rebuild) the process-global
register from the server-args knobs. Called once at runner init when the
register is enabled."* Its only caller in the entire tree is
`test/registered/unit/model_executor/test_offload_movement.py:640`.

The failure is silent rather than loud because `get_global_register()` has a
fallback: *"When enabled but not yet configured, a default (latency-profile)
register is created so early adapters can already book their items."* So with
`SGLANG_OFFLOAD_REGISTER=1` the register exists, works, and ignores the
operator's `--lane-offload-profile capacity` entirely.

`parse_park_target_order` is in the same state: the only production call is
the argument-time syntax check at `server_args.py:6542`; the movement layer
re-exports the symbol but never takes the operator's order.

Fix (S–M): call `configure_global_register` at runner init from the three
`ServerArgs` fields. S if the runner-init site is unambiguous; M if the park
order needs threading into the movement layer as well.

### F3 — #309 runtime drafter lifecycle

`speculative/runtime_draft.py` is a complete, deliberately pure state machine
(no torch, no scheduler, no device) with a clear intended boundary — its
docstring states that the weight load and VRAM return "are executed by the
scheduler at the boundary this machine hands them", reusing the #364
between-tick window. Nothing in `managers/` or `entrypoints/` imports it.
There is no code path by which an operator can attach or detach a drafter on
a running server.

This is the largest single body of unreached fork code found: full
implementation, 65 hermetic tests, a design document. Fix is L — it needs the
scheduler-side transition driver and an entrypoint, which is the half that was
never written, not a wiring line.

### F4 — #394 cold expert tier

Recorded for visibility rather than as a surprise: the merge message for
`f62073ca81` already states "placement policy inert pending reachability
slice 2", and `d71e7133d2` says "DELIBERATELY UNWIRED […] branch inert by
construction". The audit's contribution is that Detector D found it from the
code, and that the state is now pinned by a test instead of living only in a
commit message.

**Resolution (task #394 slice 2).** The routing half
(`layers/moe/cold_tier_fetch.py`) now reaches the storage half from three
production sites: the launcher mints the launch id
(`entrypoints/engine.py`), the GGUF streaming door builds the owner handle
and stashes the rank-uniform owner map
(`layers/moe/fused_moe_triton/layer.py`), and `MoEExpertOffloadCache` turns
the slice-1 refusal into a fetch route (`layers/moe/expert_offload.py`).
The absence pin in `test_unwired_features_421.py` is retired per this
document's own rule and replaced by a CALL-SITE pin,
`test/registered/unit/layers/moe/test_cold_tier_wiring_394.py` — consumer
counting finds absent wiring, only a call-site pin finds wiring a later
refactor removes. The chain's behaviour is covered hermetically by
`tests/moe_offload/test_cold_tier_fetch.py`.

Two things stay honestly open, and neither is a wiring gap: the eager path is
BOOT-PENDING (`scripts/dev/394_s2_proof/`), and the capturable path refuses by
name until the UVA device pointer for a `cudaHostRegister`'d peer mapping is
verified on hardware.

### F5 — dead ATOM port

`layers/fused_qk_norm.py` has no importer. Deleting it is zero-risk; the
alternative is a comment saying why it is kept.

## 5. Precision

| Detector | Raw candidates | Verified findings | Notes |
|---|---|---|---|
| A (ServerArgs fields) | 1 DEAD + 16 test-only, of 165 | 3 (F2's three flags) | The rest were computed-name getattr, env bridges, or legitimate validation-only consumption. |
| A (environ.py) | 0 DEAD, 0 test-only, of 96 | 0 | Every declared env has a real consumer; the interesting env work is in the *undeclared* set (§6). |
| A (undeclared envs) | 426 raw → 284 fork-added | 2 (F6 env, F7) | 118 turned out to be a live rejection list; 19 artefacts; 9 apparent orphans were prefix-concatenated reads. |
| D (module reachability) | 508 no-external-ref + 619 test-only names; 1 module at 100 % | 5 (F1, F3, F4, F5, F6) | Raw name-level output is dominated by Triton kernels, `_ref` test oracles, exception classes and re-exports. The per-module aggregation is what made it usable. |
| B (always-default param) | HEAD runtime areas: 184 rows → 12 printed | 3 (F8, F9, one deliberate test hook) | **25 %** on the printed tier. Without the injection-seam and `**dict`-builder filters it was 45 printed → 3, i.e. 6.7 %. The `planner/` run (936 rows) was left unverified. |
| B2 (degenerate derivation) | HEAD `layers/moe`: 0 | 0 (correct negative) | Fired 2/2 at the #394 pre-fix tree. |
| C (asymmetric gate) | HEAD, `SGLANG_GGUF_DENSE_VOCAB`: 9 weak, 0 strong | 0 (correct negative) | At the pre-fix tree: 10 weak → 10 % precision, but **strong tier 2 → 100 %**. The doc-line co-mention ranking is what separates the tiers. |

The raw name-level numbers are not findings and should not be quoted as
such — that is the #381 failure mode repeating. Only the per-module
aggregation plus hand tracing produced anything real.

Explicitly cleared as correctly wired after tracing (a non-exhaustive record,
so the next audit does not re-litigate them): `--enable-weights-disk-backup`
(sets `load_format="hibernate"`), the ten `kv_session_offload_budget_*` /
`spill_*` fields (computed-name `getattr`), `--collective-net-small|bulk` (env
bridge to `barlink_ucx`), `--pp-layer-ratio`, `--regime-gate-evidence` (boot
gate that can refuse), `--disaggregation-topology` (`scheduler.py:657`),
`disaggregation/topology.py` (module-level orchestrator),
`mem_cache/gdn_slot_executor.py` (via `gdn_slot_runtime` from
`scheduler.py:3348`), `registry/adapters/class2_diffusion.py` (dynamic
`register_adapter`), `planner/placement.py` and `planner/live_metrics.py`
(used by `planner/webui.py` and `bench_suite.py`),
`mem_cache/hicache_migrate.py` (`python -m` operator tool),
`ReservationStore(critical_section_probe=…)` (`registry/ledger.py:320` — a
deliberate test hook that injects a `time.sleep` to widen a race window;
recorded so it is not re-flagged).

Detector B's own limitations, which bound what §3 can claim: callees are
resolved **by simple name only** (two functions named `available_bytes` share
a bucket — one candidate was dropped for exactly this reason rather than
reported), there is no import resolution or type inference, and a parameter
passed through a dict, a `partial`, a decorator or `getattr` is invisible. A
`**kwargs` call site yields an `unknown` class that is neither positive nor
negative; `planner/` is full of them and was not verified.

A distinct class worth naming but *not* counted as a finding: features that
are wired but gated off by an env default, e.g. the GDN state-set ladder and
the whole offload register behind `SGLANG_OFFLOAD_REGISTER` (default false).
That is a deliberate dark launch, not missing wiring — but it is also what
let F2 go unnoticed, because nobody exercises the configure path.

## 6. Undeclared env vars

426 `SGLANG_*` names are referenced in `srt/` without a declaration in
`environ.py`; **284 are fork-added** (142 inherited, out of scope).

| Class | Count | Meaning |
|---|---|---|
| WIRED | 143 | reachable production read |
| GUARDED | 118 | retired name in a live rejection list |
| ARTEFACT | 19 | 8 concatenation prefixes + 11 regex false positives |
| INERT | 2 | `SGLANG_MEMTIER_PROFILE` (F6), `SGLANG_MOE_HOST_SHARD_RATIO` (#394) |
| DEAD | 1 | `SGLANG_BARLINK_PP_TRANSPORT` (F7) |
| TEST-ONLY | 1 | `SGLANG_MLX_TEST_MODEL` (pytest module inside the shipped package tree) |

`GUARDED` is a class this audit had to add: those names are neither knobs nor
dead. Setting one produces a live, production-reachable startup error. They
are functionally wired and **must not be "fixed" by declaring them in
`environ.py`**.

**The HTCCL → barlink rename is correct.** `RETIRED_ENV_VARS`
(`barlink_env_guard.py:25-147`, 118 rows) plus `check_retired_env_vars()`
raises `RetiredEnvVarError` — no alias, no copy-onto-successor, no ignore
path. It is reachable twice: at import (via `parallel_state.py:57`) and again
on every `parallel_state.graph_enable_set()` call, which gates
`capturable_transports()` and therefore sits on the live collective path. Set
symmetry is exact: the 117 `SGLANG_HTCCL*` names present anywhere in the tree
are precisely the 117 table keys.

Ten successors initially looked guarded-with-no-counterpart; **nine were false
alarms** — `SGLANG_BARLINK_{ALGORITHM, DOMAINS, LEAF_THRESHOLD,
MEASURE_REPEATS, MEASURE_SIZES_KIB, MEASURE_WARMUP, PLANNER,
SATURATION_SHARE, TIER_RATIO}` are all read through `_e(name)` =
`_ENV_PREFIX + name` in `barlink_matrix.py`, so the full literal never appears
in source and every name-grep misses them. Only F7 is genuine, and its impact
is low: the behaviour was not obtainable under the old name either, so the
rename lost nothing — the guard row is merely over-broad.

`SGLANG_BAR1EP_SELBSTTEST` / `SELFTEST` is a correctly wired pair, not a
leftover: `SELFTEST` is live (`bar1ep.py:852`), and the German spelling is
deliberately retained as a guard row and pinned by an existing test.

Two structural observations:

- An env var that bypasses `environ.py` is invisible to the central registry,
  and therefore to any future sweep that starts from the registry. The
  registry is only a useful audit surface if new envs are required to go
  through it.
- The prefix-concatenation idiom (`_e(name)`, `g("prefix_" + name)`) appears
  in at least three independent subsystems and defeats every literal search.
  Any future audit must resolve it or it will report the same false
  positives.

### 6.1 Screens that found nothing

Recorded so nobody re-runs them expecting a yield: an AST screen resolving the
enclosing function of every production read of all 284 names, then counting
call sites repo-wide, found **zero** names whose reads all sit inside an
uncalled function. The two INERT cases are package-level (F6) and gate-level
(#394), so a future sweep needs both screens, not the function-level one
alone.

## 7. Cross-check against `FEATURE_CATALOG.md`

Checked every catalog claim that names a flag, env or endpoint against the
reachability data. Two discrepancies, both in the direction of the catalog
overstating what the tip can do.

| Catalog claim | Reality at `33148dbe0f` | Direction |
|---|---|---|
| §13 lists `/session_handover` under the shipped serving surface | The string `session_handover` does not occur anywhere in `python/`. The work is on the unmerged branch `feat/live-handover-261`. `/kv_reshard` (also §13) **is** shipped (`http_server.py:1107`). | Catalog overstates |
| §3 "memtier registry […] All new spill/offload consumers must pick targets from it" | Was stated as an active constraint while being aspirational: zero production consumers (F6). Now honoured by TWO consumers — `model_executor/short_term_offload_register.price_park_target` (#286) and #410's session checkpoints through `memtier/consumers.py` — while the PRE-EXISTING offload/spill paths (#286's other park-target classes, #394 cold tier) still carry their own target lists (memtier cuts 4/5). The rule is *partly* enforced, not enforced, and §3 says so. | Catalog overstated; corrected and scoped to "new" |
| §3 "KV pressure ladder (geometry stages instead of rejects)" | True for an explicit spec; the advertised `auto` mode hard-fails (F1). | Catalog overstates in part |
| §3 "#394 cold-shard chain (slice 1 merged) […] Fetch-path wiring (slice 2) open" | Accurate — the catalog names the gap itself. | Catalog correct |
| §1 TPxPPxTP slice 3, `--pp-stage-ratio`, `SGLANG_PP_SHAPE_CACHE` | Absent from the tip; catalog correctly labels them as branch `feat/tpxppxtp-slice3-201`. | Catalog correct |
| §13 training tenant + idle workbench | Wired (`http_server.py:350,470`). | Catalog correct |
| §7 `--collective-net-small/-bulk` | Wired via an env bridge to `barlink_ucx`. | Catalog correct |

### 7.1 Catalog gaps (wired capabilities the catalog omits)

- `--kv-pressure-external-hysteresis-rounds`, `--kv-pressure-pre-stage` and
  the rung-dependency refusals (a ladder naming `admission_cap` without
  `--max-running-requests-ceiling`, or `session_offload` without
  `--enable-kv-session-offload`, is refused at argument time). §3 mentions the
  ladder but not that its rungs are dependency-checked.
- `--regime-controller` / `--regime-gate-evidence`: a boot gate that refuses
  `act` unless a runtime actuator (`--kv-reshard-vectors` or
  `--enable-vram-dial`) is wired. Not in the catalog at all.
- The `--lane-offload-*` surface (§3 area) — worth listing **with** its F2
  status rather than silently.
- `--enable-weights-disk-backup` / `--hibernate-dir` mutual requirement and
  the manifest-match auto-detect that flips `load_format` to `"hibernate"`.
  §3 mentions hibernate but not the flag contract.
- The retired-env guard itself (§7 area, 118 names): an operator upgrading
  across the rename gets a hard error with a named successor. That is a
  user-visible robustness feature and belongs in §12.

Structural note: the catalog interleaves tip state with branch state
(§1 slice 3, §6 handover). Both are labelled, but §13 then repeats a
branch-only endpoint as shipped. If the catalog is to be usable as an audit
baseline, tip-state and branch-state should be visually separated rather than
distinguished only by a parenthetical.

## 8. What this audit did NOT cover

Stated explicitly, because an audit's silence is otherwise read as a clean
bill of health:

- **Detector C was never swept.** It is confirmed against the one calibration
  env (`SGLANG_GGUF_DENSE_VOCAB`, fires pre-fix, correct negative at HEAD) and
  nothing else — the all-envs run did not finish. **Partial-wiring defects of
  the #197 shape are therefore essentially unaudited at HEAD.** This is the
  single largest remaining gap and the obvious next slice: the detector
  exists and is calibrated, it just needs to be run.
- **`planner/` was not verified.** Detector B produced 936 raw rows there,
  dominated by `**kwargs` call sites that the detector classes as `unknown`.
  Left alone on the grounds that the planner is an offline CLI/webui rather
  than serving runtime, which is a judgement call, not a proof.
- **Detector C cannot fire on an undocumented gate.** Its entity harvest comes
  from the flag's own docstring. A flag with a terse or absent docstring
  produces no entities and the detector silently declines — it cannot find an
  asymmetry that no comment ever claimed.
- **One open question, deliberately not claimed as a finding.**
  `PathProfile.saturation_threshold` has no writer anywhere in the tree, so it
  is permanently `1.0`. Whether that makes the `>= 1.0` re-route at
  `barlink_path_dispatcher.py:341` unreachable depends on the range of the
  injected `_saturation_sensor`, which was not traced.
- **The 142 inherited undeclared envs** were not examined (out of scope).
- **Nothing was executed.** No test of the audited code was run, no server
  booted, no GPU touched. Every classification above is static analysis plus
  hand tracing, and inherits that method's blind spots (§1.2).

## 9. Pins

`test/registered/unit/test_unwired_features_421.py` — 8 hermetic CPU tests
(AST over the repo tree, no torch, no CUDA). They assert that F1–F4 and F6 were
*still* unwired, so that wiring any of them turns the pin red and forces the
audit entry to be retired. Each failure message says so explicitly.

Can-fail proof (required — a pin that cannot fail is decoration): a temporary
production module importing and calling `DrafterLifecycle`, `ColdTierLayout`,
`arms_from_server_args`, `build_ladder_table`, `configure_global_register`,
`parse_park_target_order` and `TierRegistry`, plus a one-line `table_fn=None`
kwarg at `kv_pressure_runtime.py:461`, turned **every pin red**. All edits
were reverted and the suite is green again on the unmodified tree.

One pin was wrong on its first run and the can-fail discipline is what caught
it: the memtier pin initially counted `memtier/__init__.py`'s own re-exports
as a production consumer, so it failed on a clean tree. `_production_importers_of`
now takes `exclude_package` — a package `__init__.py` that re-exports its
submodules is part of the feature, not a consumer of it. Any future
reachability check on a *package* (rather than a module) needs the same
exclusion, or every packaged feature looks wired.

The pins deliberately do not assert "feature is broken" — they assert a
reachability fact with a named remedy. Relaxing one instead of deleting it
would re-create exactly the blindness this audit exists to remove.

---

# 2026-08-02 — task #428: fixes, named decisions, and the Detector C sweep

Desk follow-up on the audit above. Base `14d0675bbc`, branch
`fix/unwired-421-fixes-428`. No GPU was touched; the two HIGH fixes are
therefore reported **BOOT-PENDING** and carry a turnkey validation script
each (§B.6).

## B.1 Per-finding disposition

| # | Sev | Disposition | Where |
|---|-----|-------------|-------|
| F1 | High | **FIXED (boot-pending)** — `auto` gets the #272 planner's table injected | `managers/kv_ladder_auto.py` (new), `managers/kv_pressure_runtime.py` |
| F2 | High | **FIXED (boot-pending)** — the three `--lane-offload-*` flags reach the register at runner init | `model_executor/offload_register.py`, `model_executor/model_runner.py`, `model_executor/offload_movement.py` |
| F3 | High | open (L: needs the scheduler-side transition driver, not a wiring line) | pinned, unchanged |
| F4 | Medium | **FIXED (boot-pending)** — #394 slice 2 wires the fetch path; the tier has production importers and call-site pins | `layers/moe/cold_tier_fetch.py` (new), `layers/moe/expert_offload.py`, `layers/moe/fused_moe_triton/layer.py`, `entrypoints/engine.py` |
| F5 | Low | **CLOSED — deleted** | `layers/fused_qk_norm.py` removed |
| F6 | Medium | **FIXED (#286 + #410)** — the first production consumer is wired (`price_park_target`) and #410's session checkpoints resolve their tier through `memtier/consumers.py`; the inverted pin is retired and replaced by positive call-site pins. The PRE-EXISTING consumers (#286 register, #394 cold tier) still carry their own target lists — memtier cuts 4/5 — so the catalog rule is honoured by two consumers, not enforced across all of them | `MemTierIsNowWiredTest`, `memtier/consumers.py` |
| F7 | Low | **CLOSED — kept, as a reservation, with a pointer** | `barlink_env_guard.py:122` |
| F8 | Medium | **CLOSED — explicit named refusal, not a wiring** | `layers/moe/expert_offload.py` |
| F9 | Low–Med | **DOCUMENTED — belongs to #279; open question answered** | `barlink_path_dispatcher.py` |

## B.2 F1 — the audit's one-line fix was wrong, and how

The audit proposed: *"import `planner.kv_ladder_table.build_ladder_table` at
that call site and pass it as `table_fn`."* Checked against the code, that
does not compile into anything that works: `table_fn` is called as
`table_fn()` with no arguments, while `build_ladder_table(profile,
*, …)` requires a `RigModelProfile` — and **no production code path
constructs one**. Every existing `RigModelProfile` in the tree is built by a
unit test or by the planner's own offline CLI. The missing piece was not an
import, it was the profile bridge.

`managers/kv_ladder_auto.py` is that bridge. Three constraints shaped it, and
each one rules out an easier implementation:

- **Rank-uniformity.** The ladder's rung index is min-reduced across the TP
  group; two ranks with different tables would agree on an integer that means
  different things. So every input must be identical on every rank *by
  construction*. That excludes `torch.cuda.mem_get_info` (a rank under
  `CUDA_VISIBLE_DEVICES` narrowing sees only its own card) and it excludes
  any collective (the bridge runs in the scheduler constructor, the
  rank-local-before-group window).
- **Card identity by UUID.** The rank → card mapping comes from the
  launcher-published vector (`registry.rank_cards`, an environment read, the
  same source #394 uses), and card facts from NVML by UUID. Without the
  vector the mapping is only recoverable on a rig whose cards are
  indistinguishable; on a mixed rig the bridge **refuses and names the two
  remedies** rather than guessing an enumeration order (#392/#397).
- **Only rungs whose actuator exists.** `server_args`' actuator-dependency
  refusals apply to an explicit tuple spec only — `auto` had no equivalent
  gate — while `KvPressureRuntime.__init__` refuses an `admission_cap` rung
  without an armed limiter. The profile therefore inventories a relief
  feature only when this configuration wires its actuator, so `auto` cannot
  produce a table the runtime rejects.

Deliberately **not** supplied: `kv_bytes_per_token` and
`weight_bytes_total`. Both are per-rank runtime facts under uneven DCP /
uneven TP, so deriving a model-wide figure from one rank's pool is the one
input capable of making two ranks' tables disagree. Their absence means every
rung carries the planner's `placeholder` provenance and says so in its own
`source` field. That is the honest state; the measured figures are a
follow-up off the ms/round chain, not an estimate inserted here.

## B.3 F2 — configure before the first read, once per process

`configure_global_register_from_server_args` is called at the top of
`ModelRunner.__init__`, which is the earliest point that is also *after*
`server_args` exists: the pools, input buffers and lane workspaces whose
adapters call `get_global_register()` are all created later in the same
`__init__`. Two properties the audit's one-line suggestion does not mention
and that the fix has to get right:

- **Once per process.** A draft runner and a #274 dual-group lane runner each
  construct a further `ModelRunner` in the same process. `configure_*`
  rebuilds the register, which would drop every item the first runner's
  adapters already booked, so the entry point is idempotent and records that
  it has run.
- **The park order needed a home.** `configure_global_register` had no
  parameter for it. It now takes one, the register exposes it, and
  `offload_movement.park_target_order_from_register()` is where the movement
  layer picks it up — an omitted `target_order` on `RealMovementBackend` now
  defaults to the operator's chain instead of the module constant. Nothing in
  production constructs that backend yet (a separate, GPU-phase gap), so this
  is the honest reach of the fix: the flag now has a runtime consumer path,
  not a runtime consumer.

Values are **re-resolved** at configure time rather than passed down from
argument time, so a typo refuses loudly at runner init as well — including on
the direct-`ServerArgs`-construction path that never runs the argument-time
validator.

## B.4 F8 — refusal, not wiring, and the reason is a measurement

The audit left the choice open between threading the cold-shard context into
the fp8/GPTQ/AWQ presplit callers and an explicit refusal. Checked against
#394's own work, it has to be the refusal, and threading would have been
actively wrong:

The parameter's documented precondition is "only legal on a layer that shards
experts on dim 0 and remaps foreign ids away". That is exactly the
configuration in which delegation is **unsound** — `partition_cold_experts`
drops the experts it delegates to a peer's host tier, and under a disjoint
expert shard no peer holds them. The GGUF door found this by booting: on the
reference rig (V4-Flash UD-IQ3_XXS, TP=3, 2026-08-02) every rank died on the
first forward with *"experts [80, 83, 94] were delegated to a peer rank's
host tier … but this rank's router asked for them"*, which is why
`_gguf_cold_shard_context` returns `None` by default today.

So the repack door is shut in both directions and says which one applies: an
intermediate-dim TP MoE has no whole expert to delegate; an EP or GGUF
expert-shard layer could delegate structurally but the delegated experts
would be unreachable. `SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE` — the same
escape hatch the GGUF door carries — opens the eligible case for developing
the missing reachability mechanism against a real boot.

Note for the next sweep: a `cold_shard=refusal_helper(layer)` at the four
call sites would have made Detector B report the parameter WIRED while it is
still always `None`. That is the #394 forwarding-taint shape, deliberately
avoided.

## B.5 F5 and F7 — the two named decisions

**F5 `layers/fused_qk_norm.py`: DELETED.** Provenance is the interesting
part. `git log --diff-filter=A` attributes the file to `3f5e2c7688`
(*"[AMD] Dsv4/pr2 compressor opt (#26208)"*), an upstream-titled commit — but
that commit is **not** an ancestor of `upstream/main`, `upstream/main` has no
commit touching the path, and the file is absent from the upstream tree. The
consistent reading is a rebase artefact: the fork's own ATOM port
(the docstring says *"Ported from ATOM (atom/model_ops/layernorm.py)"*) was
folded into a rebased upstream commit. It has zero importers anywhere in the
tree, no design doc and no catalog entry; the similarly named live path
(`fused_qk_norm_rope_store.py`, used by `models/deepseek_v4.py`) is
unrelated. Deleting it costs nothing and removes a name that reads like a
live kernel.

**F7 `SGLANG_BARLINK_PP_TRANSPORT`: KEPT as a reservation, with a pointer
comment.** Cross-checked against #201 slice 3 (`feat/tpxppxtp-slice3-201`,
unmerged) as instructed: slice 3 does **not** consume it — its only mention
on that branch is the same retired-name guard row and the same DESIGN_201
table line that exist at this tip. The row is therefore not a rename of a
working knob; neither name ever had a reader. It is kept anyway because
DESIGN_201 still lists the successor as the stage-boundary P2P lever, so
whoever implements it inherits the retired-name protection for free. The
comment at `barlink_env_guard.py:122` says all of this, so nobody reads the
row as evidence the knob exists.

## B.6 Boot-pending — what the desk cannot show

`scripts/dev/428_boot_checks/` holds one turnkey script per HIGH fix. Neither
fix is DONE until they pass in a GPU window.

- `f1_kv_pressure_ladder_auto.sh` — boots the standard 27B recipe with
  `--kv-pressure-ladder auto`, asserts the table was computed from the real
  rig profile, drives KV pressure for a bounded time, asserts a rung flip
  fires and that serving is still coherent after it.
- `f2_lane_offload_register.sh` — boots with all three `--lane-offload-*`
  flags under `SGLANG_OFFLOAD_REGISTER=1`, asserts the operator's profile and
  park chain reached the register, asserts exactly one configure per process
  with a NEXTN draft runner present, then asserts a planted typo refuses the
  boot instead of degrading to the default preset.

## B.7 Detector C: the full sweep the audit did not run

§8 named this as *"the single largest remaining gap"*. It is now run.

**Detector change (performance only).** `run_env` recomputed
`defined_symbols` and `attribute_assign_sites` — two full walks of ~5.5k
trees — for every target, which put the 381-target sweep at ~2 h. Both are
now memoised per `Index` (the index is immutable once built), and the sweep
runs in ~35 min. The README's calibration contract was re-verified against
the patched detector: at `ef6f8bc0a2^` it still fires STRONG on `lm_head`
(`models/qwen3_vl.py:1280,1288`) with `embed_tokens` honoured, and it is
still a correct negative at HEAD.

**Scope.** `--auto` enumerates envs, not flags: Detector C's entity harvest
starts from a gate name that appears literally in function bodies, which is
what an `SGLANG_*` read looks like and is not what a `ServerArgs` field
access looks like. The **flag half of the sweep was not run** and is a named
remaining gap, not a silent omission.

**Result: 381 fork-scoped env names, 36 STRONG fires raw, 0 verified
findings.**

| tier | count | disposition |
|---|---|---|
| detector blowup (≥ 15 strong entities) | 20 | ARTEFACT, not reported |
| hand-traced | 10 | all false positives |
| tractable, not traced | 6 | **unverified**, named below |

**The blowup tier is a detector limit worth recording.** Twenty fires carry
73–147 strong entities against 552–6414 honoured sites — an "asymmetric gate"
spanning most of the tree is not a finding, it is a broken gate set. The
cause is `env_reads`: it treats any function under 40 lines whose body
mentions the env as a predicate helper, and when the enclosing function
happens to be named `ledger`, `all_gather`, `recv`/`send`, `get_path` or
`process_weights_after_loading`, the gate set becomes a tree-wide token and
every `self.X = Call()` in the codebase enters the comparison. A future
Detector C run should require a helper name to be *unique* to the gate's
module, or cap the gate set.

**The traced tier is one recurring false-positive shape, plus two subshapes.**
Verbatim, so the next run does not re-litigate them:

| env | strong entity | why it is not a finding |
|---|---|---|
| `SGLANG_DFLASH_SOLO_POOL_CAP` | `num_draft_slots` | a constructor *parameter* of `DraftKVSlotMapper`, assigned in its own `__init__`; the "honoured" sibling is the site that calls it |
| `SGLANG_DUAL_GROUP_LANE_STREAM_PRIORITY` | `device_clock` | one object, two lifecycle phases: created in `__init__`, bound to the stream in the honoured `start_worker` |
| `SGLANG_EXPERT_STATS`, `…_PATH`, `…_INTERVAL_SEC` | `graph_mode`, `num_experts` | constructor parameters assigned to `self` inside the collector the gate governs |
| `SGLANG_UNEVEN_DCP`, `…_WEIGHTED`, `SGLANG_UNEVEN_MLP_VECTOR` | `cmd`, `intermediate` | `cmd` is a subprocess argv builder in `planner/split_probe.py`; `intermediate` is an HF-config field parsed in `uneven_perf.py`. Prose collision, no relation to the gate |
| `SGLANG_BARLINK_TRANSPORT` | `transport`, `host` | `barlink.py:526` assigns `self.transport = _build_transport(_TRANSPORT, …)` — `_TRANSPORT` **is** the gate value, read into a module constant. The detector matches the literal env name in the function body and cannot see the constant. This is the §1.2 prefix/indirection blind spot in a new dress |
| `SGLANG_HACK_FLASHMLA_BACKEND` | `unified` | the def site is in `deepseek_v4_backend_hip_radix.py`, which exists in `upstream/main` — inherited, outside the audit's fork-delta scope |

**Unverified, and named so the next slice starts here:**
`SGLANG_BARLINK_SLOT_MIB`, `SGLANG_BARLINK`,
`SGLANG_ADAPTIVE_SERVING_MARGIN_MIB`, `SGLANG_MOE_RESIDENT_EXPERT_FRACTION`,
`SGLANG_CACHE_DIR`, `SGLANG_USE_HND_KVCACHE`. The last two have def sites in
files that are in `upstream/main` and are very likely out of scope for the
same reason as `SGLANG_HACK_FLASHMLA_BACKEND`; the first four each carry 2–6
strong entities with generic names (`device`, `host`, `lock`, `planner`,
`kernel`, `runner`) and look like the same prose-collision shape, but they
were **not** traced and are not claimed either way.

**What this does and does not license.** The #197 shape is now swept across
every fork-scoped env at HEAD and produced nothing verified. It does *not*
mean the fork has no partial wiring: the detector cannot fire on an
undocumented gate (§8), cannot see a gate read through a module constant (the
`SGLANG_BARLINK_TRANSPORT` row above proves that on live code), and the flag
half was not run at all.

## B.8 Pins

Three of the eight #421 pins are retired, per their own instruction ("delete
the pin, do NOT widen it"): `TestKvPressureLadderAutoIsUnreachable` (F1),
`TestOffloadRegisterProfileIsUnreachable` (F2) and
`TestColdTierShmIsUnreachable` (F4), and `TestMemTierRegistryHasNoConsumers`
(F6, retired by #286 — the memtier registry now has a production consumer,
`model_executor/short_term_offload_register.py`; the positive replacement is
`test/registered/unit/model_executor/test_short_term_offload_register.py::MemTierIsNowWiredTest`).
Two remain green (F3 ×2).
`test_unwired_features_421.py` carries a comment naming what replaced each
retired pin.

Positive replacements, each with a documented can-fail proof:

| file | covers |
|---|---|
| `test/registered/unit/managers/test_kv_ladder_auto_421.py` | F1: the runtime builds under `auto`; the profile is rank-uniform, UUID-keyed and refuses a mixed rig without a card vector; only wired reliefs are inventoried; the default path never evaluates the table source |
| `test/registered/unit/layers/moe/test_cold_tier_wiring_394.py` | F4: the three call sites that carry the cold-tier chain (launcher launch id, GGUF owner handle, cache fetch route), an inverted importer pin, and the default path taking no cold-tier branch |
| `test/registered/unit/model_executor/test_offload_register_wiring_421.py` | F2: all three flags observable in the register; planted typos refuse; once-per-process; and an AST pin on the **call site**, so a refactor cannot move the configure step after the first adapter read |
| `test/registered/unit/layers/test_expert_offload_repack_door_421.py` | F8: eligibility classification, both refusal messages, the escape hatch, and the real entry point refusing before doing any work |
| `test/registered/unit/distributed/test_barlink_dispatcher_inert_421.py` | F9: the overflow tier is reachable at exactly 1.0 and works with a measured threshold; `transport_hint` is pinned as an absence, and the hook is shown to work once a hint exists |

## B.9 F-S8 — the #363 stage feed: a seam whose docstring claimed its own wiring

Found while auditing spill composability (2026-08-04), same detector class as
this audit's, and worth recording here because it is the sharpest instance yet
of the failure mode the audit was built for.

`regime_stages.planner_candidates(server_args, *, solve_fn=None)` is the
production feed of the #363 stage table. Its docstring stated:

> ``solve_fn`` is the seam. Production binds it to the key solver
> (``planner.key_solver.solve`` through ``planner.solver_api``) ...

Nothing binds it, under any configuration. The one production caller,
`regime_runtime.build_regime_stage_table`, invokes it as
`planner_candidates(server_args)` — the argument omitted — so the
`solve_fn is None` branch is the only branch production ever takes. The stage
table therefore permanently holds the booted stage alone, and act mode has
nothing to select between: not broken, UNFED.

**Why the usual detectors missed it.** The seam has tests, and they pass —
three of them, all injecting a stub, which is exactly the shape section 1
warns about ("the tests called the feature directly"). The runtime also
behaves correctly and says so: the `None` branch emits an honest note ("no
planner feed bound: the stage table holds the booted stage only"). So the
absence was reported at runtime while the docstring asserted the opposite, and
a reader who read either one alone came away with a consistent — and, in the
docstring's case, wrong — picture.

**It is a wiring gap, not a missing solver.** `key_solver.solve` exists and is
#348b/#350-aware by construction. But its signature takes `plan_inputs`, a base
plan, per-rank budgets and `RigRates` — a card probe and a measured rate set —
and a binding also has to map a `SolverAnswer` back onto a `Stage`. That is a
boot-time construction job, not a two-line call, which is the most likely
reason it never happened and the reason it is tracked (#363/S8) rather than
fixed in passing.

**Disposition.** Docstring corrected to state that the seam is unbound, what
binding requires, and that the runtime note is the honest report until then.
The ratchet sat in `test_regime_act.py::TestPlannerFeed`, over both the call
site and the docstring: it goes red when the seam is bound, which forces the
wiring and the prose to move together.

**Resolution (task #578).** The ratchet fired and the seam is now BOUND.
`regime_runtime.build_regime_stage_table` calls
`planner_candidates(server_args, solve_fn=_planner_solve_fn(scheduler))`
(`managers/regime_runtime.py`), and `planner_candidates`
(`managers/regime_stages.py`) records the binding in its own docstring. Per
this document's rule the absence ratchet was INVERTED rather than deleted:
`test/registered/unit/managers/test_regime_act.py::TestPlannerFeed::test_the_seam_is_bound_and_refuses_without_measurement`
pins that the call site still passes `solve_fn` and that planner-produced
stages stay `unmeasured` and are refused by name.

**The general lesson, for the next sweep.** A docstring that describes its own
production wiring is an assertion about a call site somewhere else, and nothing
checks it. Prefer a test that reads the call site (`inspect.getsource`) over a
sentence that describes it — and when a seam's default branch exists to report
an absence, that branch being the only reachable one is the thing to detect.
