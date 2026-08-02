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

Detectors B and C, and the formal calibration runs, were executed by a
dedicated sub-audit; A and D are recorded here.

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

**Calibration 1 (#197, asymmetric gate): NOT caught by Detectors A or D, by
construction.** `SGLANG_GGUF_DENSE_VOCAB` had — and has — many production
consumers, so every consumer-count and module-reachability method classifies
it WIRED. This is the honest limitation that motivated Detector C, and it is
stated here as a negative result rather than glossed: **the sweep recorded in
this document would not have found #197.** The Detector C run against
`ef6f8bc0a2^` is a separate sub-audit; until its result is folded in, treat
partial-wiring defects as uncovered by this audit.

The general lesson: **consumer counting finds absent wiring, never partial
wiring.** A codebase with escape-hatch flags needs the doc-coverage check as
a separate pass.

## 3. Findings

| # | Item | Class | Evidence | Sev | Fix |
|---|------|-------|----------|-----|-----|
| F1 | `--kv-pressure-ladder auto` | INERT (hard-fails) | `build_ladder_from_server_args(sa, *, table_fn=None)` raises on `auto` when `table_fn is None`; the sole production caller `managers/kv_pressure_runtime.py:461` omits it. `planner/kv_ladder_table.build_ladder_table` exists and is tested but has zero production callers. | High | S |
| F2 | `--lane-offload-profile` / `--lane-offload-class-policy` / `--lane-offload-park-targets` | INERT | `server_args._handle_lane_offload_register` validates then discards ("recomputed at configure time"). The configure-time entry `configure_global_register` has no production caller — only `test_offload_movement.py:640`. `get_global_register()` instead builds a bare `OffloadRegister()` on the default latency profile. | High | S–M |
| F3 | #309 runtime drafter attach/detach | DEAD | `speculative/runtime_draft.py` (9 public defs, 65 tests, `docs/dev/TASK_309_RUNTIME_DRAFT.md`) has no production importer. Sibling `speculative/draft_selection.arms_from_server_args` likewise test-only. | High | L |
| F4 | #394 cold expert tier | INERT (declared) | `layers/moe/cold_tier_shm.py`: no production importer. Self-declared in the merge message as "inert pending reachability slice 2" — recorded here so the state is visible to the suite, not only to git log. | Medium | L |
| F5 | `layers/fused_qk_norm.py` | DEAD | Triton fused Q/K RMSNorm ported from ATOM. No importer anywhere in the tree. The similarly-named live path is `layers/fused_qk_norm_rope_store.py`, used by `models/deepseek_v4.py`; the two are unrelated. | Low | S (delete) |
| F6 | #407 memtier registry | INERT | `srt/memtier/` (registry, tiers, probe, profile, reservations): zero production importers and zero production symbol references outside its own package. `SGLANG_MEMTIER_PROFILE` is read at `memtier/profile.py:361`, on a path a serving process never executes. Found independently by module reachability and by the env triage. | Medium | L |
| F7 | `SGLANG_BARLINK_PP_TRANSPORT` | DEAD | Present only as a successor value in the retired-name guard (`barlink_env_guard.py:122`) and in a design doc. Not among the `_e()` suffixes read by `barlink_matrix.load_config`. The retired predecessor `SGLANG_HTCCL_PP_TRANSPORT` was introduced by the rename commit `6a5f307260` itself and never had a reader either. | Low | S |

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
`mem_cache/hicache_migrate.py` (`python -m` operator tool).

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
| §3 "memtier registry […] All new spill/offload consumers must pick targets from it" | Stated as an active constraint; it is aspirational. Zero production consumers (F6), and the two offload consumers audited here (#286 register, #394 cold tier) both pick targets without it. | Catalog overstates |
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

## 8. Pins

`test/registered/unit/test_unwired_features_421.py` — 8 hermetic CPU tests
(AST over the repo tree, no torch, no CUDA). They assert that F1–F4 and F6 are
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
