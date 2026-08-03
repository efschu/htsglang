# Audit #500 — mechanism reach: every catalog condition against its code predicate

Desk audit, no GPU, nothing executed. Base commit `3b7569f664`
(`origin/integration/r3-probe-next2`), audit branch `docs/reach-audit-500`.
Upstream reference for the fork-delta half: `upstream/main` = `ec741e4161`
(2026-08-02), one day older than the audited tip, so every delta reported here
is fork code and not upstream drift.

## 1. The class of defect under audit

CLAUDE.md's MECHANISM REACH law (user law, 2026-08-03, after the fifth incident
of this class): *the catalog NAMES mechanisms; their actual reach is defined by
CODE.* This audit is its first systematic application.

The occasion, restated because it is the template for everything below.
`FEATURE_CATALOG.md` §1 read "TP>kv_heads via replication+token shard". The gate
that actually governs that machinery is

```python
def uneven_dcp_kv_replicated(dcp_size: int) -> bool:
    return dcp_size > 1 and get_tp_partition_ratios() is not None
```

`python/sglang/srt/distributed/utils.py:346`. It never reads a kv-head count.
The mechanism was live for EVERY kv-head count, and two solver generations left
the axis out because they read the catalog shorthand as the condition. The
damage was not a missing feature — it was a **registry disagreement**: a
one-line summary treated as a specification.

Four classes are used throughout:

- **[WIDER]** — the code is MORE general than the catalog line. A hidden
  capability. These are the finds worth acting on.
- **[NARROWER]** — the code is MORE restrictive; the catalog over-promises.
  Split into DOC-CANDIDATE (the restriction is intended and argued at its site,
  the catalog line is just short of it) and BUG-CANDIDATE (the restriction
  causes real user harm).
- **[EXACT]** — predicate and catalog line agree.
- **[NOT-FOUND]** — no gate located. Recorded honestly with what was searched;
  a catalog claim with no predicate behind it is itself a finding.

Every row cites `file:line` of the predicate and quotes the operative condition
verbatim. A row that cites only a docstring, a comment or CLI help text says so
— that distinction is the whole point of the law.

## 2. Method

**Direction 1 — catalog to code.** Every conditional claim in §§1-17 was
extracted by pattern ("only when", "gated on", "requires", "refuses by name",
"never", "always", "not combinable with", "off by default", "only after", "must
equal", "is a hard error"), numbered, and then resolved to its predicate by
grepping the named symbol/flag/env, opening the file and reading the branch —
not the docstring above it. 190 items across the seven section groups.

**Direction 2 — code to catalog** (the inverse of audit #421, which asked
"built but not wired"; this asks "wired but not written down"). Three surfaces
enumerated as a set difference against `upstream/main`: HTTP routes, `ServerArgs`
fields (AST over the dataclass), `environ.py` entries (AST). Each fork-added
name matched against `FEATURE_CATALOG.md` and `FEATURES_VS_UPSTREAM.md` in both
spellings. The matcher is literal, so compact catalog forms
(`--lane-offload-profile/-class-policy/-park-targets`) read as misses for the
elided members; those are marked as matcher artefacts and not claimed as gaps.

**What this audit did not do.** Nothing was executed: no test run, no server
booted, no GPU touched. §14 (dashboard) and §16 (instruments) were not swept for
predicates — the context went to §§1-7, which the task ranked first. That is a
stated coverage gap, not a clean bill of health.

## 3. The structural finding

**There are two capability registries in this tree and they disagree.**

`python/sglang/srt/planner/flags.py` describes itself as "the single source of
truth for EVERY sglang `ServerArgs` flag plus EVERY fork-specific flag/env var"
(`flags.py:15-21`), carries `requires` / `mutually_exclusive_with` / `allowed` /
`model_compat` per flag, and drives the dashboard's Runner tab. `FEATURE_CATALOG.md`
never mentions it.

Its module docstring has carried the CORRECT general statement of the #492
capability the entire time (`flags.py:38-46`):

> The fork CAN do uneven Tensor Parallelism with `tp_size` GREATER than the
> model's KV-head count (KV heads are replicated and the KV cache is
> token-sharded on the DCP axis) … Neither of those is marked
> model-incompatible here.

So the knowledge that the replication axis is unconditional was in the tree, in
a file that calls itself authoritative, while the catalog stated it as a special
case and the solver skipped it. The lesson generalises past the one incident:
**neither registry is authoritative, only the predicate is** — which is what the
MECHANISM REACH law already says, and what §0 of the corrected catalog now
points at.

The disagreement runs both ways. Three of `flags.py`'s own curated edges are
NARROWER than the runtime and block real configurations through the dashboard
(I-2, I-3, I-4 in §6 below).

## 4. Results

| group | items | WIDER | NARROWER | EXACT | NOT-FOUND | bug candidates |
|---|---|---|---|---|---|---|
| §1 uneven parallelism | 39 | 9 | 10 | 18 | 2 | 3 |
| §2 planner + §17 | 44 | 4 | 12 | 22 | 4 | 4 |
| §3 memory tiers / offload / spill | 69 | 9 | 23 | 36 | 1 | 9 |
| §4/§5/§6 spec + lanes | 36 | 9 | 8 | 21 | 2 | 2 |
| §7/§8/§9 collectives, GGUF, quant | 40 | 6 | 10 | 20 | 0 | 4 |
| §10-§16 (partial) | 11 | 2 | 0 | 9 | 0 | 0 |
| **total** | **239** | **39** | **63** | **126** | **9** | **22** |

Three §7-§9 items are measurement or environment claims ("beats NCCL
1.13-1.34x", the rig interconnect facts, "e4m3 KV bit-exact on sm86") and are
excluded from the gate taxonomy rather than forced into it. Bug candidates
consolidate to twenty distinct tasks in §5 after de-duplication across sections.

**One number is the headline: 39 WIDER against 63 NARROWER.** The catalog errs
in both directions at a similar rate, which is the argument against treating
either registry as a specification. But the WIDER column is where the cost sits,
because a hidden capability is not merely undocumented — it is actively designed
around, as #492 and #500-B1 both show.

The per-section tables follow in §7. Bug candidates are consolidated as named
task proposals in §5.

## 5. Bug candidates — proposed tasks

**Status, 2026-08-03 (#504, branch `fix/audit-bundle-504`).** Six of the rows
below are FIXED, with a falsifier each: **B3** (draft-KV-DCP now keys on the
installer's own predicate `uneven_weighted_dcp_enabled()`, so the
`--rank-kv-ratio` route is admitted), **B8** (`validate_breakable_boot`'s
`None` arm split into a `NO_SERVER_ARGS` sentinel that skips and an
unresolvable backend that refuses by name), **B10** (verdict: the refusal is
DELIBERATE, not stale — the reason and the one unobserved round are now in the
message, and `KVSO_ALLOW_SPEC` is surfaced in the CLI help, which it never
was), **B2** (`--rank-kv-ratio` without a placement refuses by name instead of
being accepted-and-inert), **B18** (the marlin fold reads a per-backend
`marlin_packable_linear` declaration instead of a class-name list), and
**I-2/I-3** (`planner/flags.py`'s two inverted edges, plus a contract test that
drives every declared uneven-TP edge against the runtime).

**B1 is separately REFUTED, not fixed** — recorded here for one place to read
the whole board: `fix/enum-geometry-503` (merged onto this line ahead of
#504) executed the check and found the audit itself wrong, not the code.
Two independent predicates govern the family and B1 conflated them:
`attn_kv_replicated` (`kv < tp`, strictly) shards the k/v PROJECTIONS
whenever `kv_heads >= tp`, so the phase-prefill enumerator's kv-head grid was
the runtime's grid all along; `uneven_dcp_kv_replicated` only replicates the
KV POOL, never the projection weights. The real (narrower) defect this
uncovered is in `planner/placement.py:813`, which still reports projection
heads as replicated under uneven DCP where the runtime shards them — see
FEATURE_CATALOG.md §1 and `NOTE_485_joint_phase_vectors.md` for the full
correction chain. Combined board after both merges: **B1 REFUTED (#503)**;
**B2/B3/B8/B10/B18 + the flags.py registry edges FIXED (#504)**; **B4-B7,
B9, B11-B17, B19, B20 still OPEN**.

Two corrections to rows below, from executing rather than reading. **B18**:
of the three configs named, only `MarlinConfig` is genuinely marlin-served and
blockless, and it is neither registered in `QUANTIZATION_METHODS` nor concrete
(no `get_scaled_act_names`) — a LATENT hole, not a live boot failure;
`W8A8Fp8Config` and `QuarkConfig` reach no marlin repack entry point at all, so
they were not gaps. **I-2**: the `rank_gpu_id` x `dp_size`/`ep_size`/`nnodes`
edges the audit recorded as declared-but-unverified are REAL runtime refusals
(driven in the contract test), so those edges stay.


Sixteen [NARROWER] rows carry real user harm rather than a sloppy catalog line.
Ranked by leverage. Every one is a named, self-contained task; none of them was
fixed in this audit (the audit changed documentation only).

**#500-B1 — the #485 phase-prefill solve prices an attention geometry the boot
does not run.** `uneven_perf.py:4133-4148` grids the attention family on
`attn_units` (kv heads) and escapes to the q-head grid ONLY when
`attn_units < tp`. But every configuration `--rank-perf-tune phase-*` can run in
is an uneven-TP boot with a non-uniform base plan, which is exactly the condition
under which `uneven_dcp_kv_replicated` (`distributed/utils.py:346`) REPLICATES
the kv heads and shards the token axis. The planner's own `placement.py:813`
models this correctly (`replicated = kv_heads < tp or dcp_replicated`) and this
same cost model already prices the KV cache as replicated
(`uneven_perf.py:3852`) while pricing attention weights as head-sharded — one
class, two contradictory assumptions. Consequence: on the reference rig the
whole attention ladder is discarded as duplicates of `[2,1,1]`
(`_attn_partition_key`, `:5157`), so "the attention family is grid-PINNED" and
"the lever is the 16-unit GDN grid" are artefacts of the enumerator, and the
reported +1.0 / +6.9 points are an optimum over the wrong feasible set. **This
is the #492 defect recurring inside the code that #492 was supposed to teach.**
*Task:* `phase-prefill: price the attention family on the replicated-KV geometry (q-head grid + token vector), not the kv-head grid`

**#500-B10 — kvso is refused under speculation, and the catalog says the
opposite.** `if self.speculative_algorithm is not None and
os.environ.get("KVSO_ALLOW_SPEC", "0") != "1": raise ValueError`
(`server_args.py:6580`). §3 read "decoupled from speculation". With the standing
NEXTN recipe the documented pair does not boot at all.
*Task:* `kvso x speculation: decide whether KVSO_ALLOW_SPEC is the supported route or the refusal is stale`

**#500-B2 — `--rank-kv-ratio` without `--rank-gpu-id` passes validation and is
then silently inert.** Its validation was deliberately hoisted above the
`if self.rank_gpu_id is None: … return` at `server_args.py:9644`, but the
uneven-DCP auto-engage `self.dcp_size = self.tp_size` sits BELOW it at `:9845`.
So `--rank-tp-ratio 2,1,1 --rank-kv-ratio speed` with no placement flag keeps
`dcp_size == 1`, never installs a token vector (`managers/scheduler.py:5952`),
and nothing warns — `reject_silently_inert_dcp` is itself gated on `dcp_size`.
The user asks for the #210 decode lever and serves the coupled layout.
*Task:* `uneven-DCP auto-engage must run on the no-placement path too, or refuse --rank-kv-ratio there by name`

**#500-B3 — `--draft-kv-layout dcp` refuses the supported flag route.**
`_reject_unsupported_draft_kv_dcp` requires `SGLANG_UNEVEN_DCP=1` AND
`SGLANG_UNEVEN_DCP_WEIGHTED=1` literally (`server_args.py:7448-7456`), while the
sibling speculation×DCP gate accepts the same configuration expressed through
the flag (`… or self.uneven_kv_flag_active()`, `:7628`). A boot with
`--rank-kv-ratio speed|capacity|<vector>` IS on the weighted owner rule, and #108
refuses it anyway — so the −67 % draft-KV win is unreachable for every boot that
uses the flag instead of the legacy env pair.
*Task:* `#108 draft-KV-DCP gate must accept the --rank-kv-ratio route to weighted DCP, not only the env pair`

**#500-B8 — `validate_breakable_boot` has a total-bypass arm.**
`if backend is None: return` (`offload_capture_gate.py:358`) skips BOTH
preconditions whenever `resolved_backend` swallows an exception (`:408-421`).
The failure the gate exists to prevent — an illegal host read inside a real
capture — is reachable through the gate's own `None` arm.
*Task:* `validate_breakable_boot: an unresolvable backend must refuse, not return`

**#500-B9 — the heat-migration and hot-residency boot refusals read only the
legacy env.** Both check `SGLANG_MOE_OFFLOAD_CUDA_GRAPH`
(`expert_heat_migration.py:339`, `layer.py:702`), so
`SGLANG_MOE_OFFLOAD_GRAPH_MODE=capturable` — the other spelling of the same
refuted path, which `offload_capture_gate.py:281-284` does refuse — walks past
them into the configuration whose LUTs the migration would invalidate.
*Task:* `heat migration / hot residency: gate on the RESOLVED graph mode, not the legacy env spelling`

**#500-B11 — the GDN slot ladder validates at boot and is then inert.**
`--gdn-state-set-ladder` parses and validates, but the executor returns early
without `SGLANG_OFFLOAD_REGISTER=1` (`if not offload_register_enabled(): return
[]`, `model_executor/offload_gdn_states.py:344`). Same root suppresses the
`--lane-offload-*` runner-init typo refusal (`offload_register.py:1239` sits
behind the gate).
*Task:* `GDN slot ladder + lane-offload typo refusal: accepted-then-inert without the dark-launch env`

**#500-B4 — two derivations of the per-decode KV reserve disagree.** #486
derived the reserve as `W + L` and made it a named pool posten, but the
hybrid-SWA / SWA-chunk-cap pool sizer still computes
`2 * get_alloc_len_per_decode(sa)` (`model_executor/pool_configurator.py:628`).
Identical only on the NEXTN recipe where `W == L`; they diverge on every
non-overlap run and every topk>1 / page>1 tree.
*Task:* `route SWA/hybrid pool decode_alloc through get_alloc_reserve_per_decode (#486 follow-up)`

**#500-B13 — the cross-algo ladder pins the DFLASH solo host to rank 0.**
`cross_algo_utils.py:733-739` refuses `--speculative-draft-gpu` and hard-codes
rank 0, so "DFLASH solo draft on the big card" is false in the cross-algo mode
on any heterogeneous rig — i.e. on this one, where rank 0 is not necessarily the
5090.
*Task:* `let --speculative-cross-algorithm honour --speculative-draft-gpu for the DFLASH rung`

**#500-B5 — `--objective energy` silently switches off the #485 joint cut.**
`joint = tune in _PHASE_TUNES and not decode_objective and not
_objective_is_energy(server_args)` (`uneven_perf.py:6408`). The energy objective
is otherwise loud about everything it cannot price; this one absence prints
nothing at all.
*Task:* `--objective energy: price the #485 joint pairs or name the drop in the plan log`

**#500-B6 — the #437 fundability gate vanishes instead of failing when the
derived per-GPU reserve is missing.** `if demand_by_gpu:`
(`uneven_perf.py:5919`) — with no derived reserve the gate yields `None` for
every candidate, so the exact configuration class #264 OOM'd on is admitted with
no verdict and no `fundability basis:` line. That is the silence failure mode
audit #421 exists to catch.
*Task:* `fundability gate: state UNPRICED when the derived per-GPU reserve is missing instead of admitting everything`

**#500-B14 — `planner/flags.py` declares `rank_gpu_id` mutually exclusive with
`pp_size`; the runtime REQUIRES it there.** `flags.py:594` vs
`server_args.py:9592` ("`--rank-tp-ratio` with `--pp-size > 1` … requires
`--rank-gpu-id`") and `:8973` / `:9655`, which validate the world-length
`pp_size * tp_size` form. The dashboard therefore cannot express cross-rig
PP × uneven TP — the §1 TPxPPxTP feature. `tuple_len_flag="tp_size"` is wrong
under a pipeline for the same reason.
*Task:* `planner/flags.py: rank_gpu_id is not exclusive with pp_size — fix the edge and the world-length rule`

**#500-B15 — `planner/flags.py` makes `rank_tp_ratio` require `rank_gpu_id`;
the runtime deliberately decouples them.** `flags.py:627` vs
`server_args.py:9540-9548`, which names the configuration the decoupling exists
for: the cross-vendor two-launcher bring-up (one CUDA venv + one ROCm venv,
`--nnodes 2`), where `--rank-gpu-id` cannot describe the AMD rank at all because
it resolves devices through NVML.
*Task:* `planner/flags.py: drop the rank_gpu_id requirement from rank_tp_ratio (blocks the cross-vendor arm)`

**#500-B7 — the in-graph MoE expert fetch is refuted at runtime and absent from
`planner/rejected.py`.** CLAUDE.md and §17 both send readers to the register
before re-proposing an approach; `offload_capture_gate.py:284` refuses the
approach by name (#452, with a measured 6.60x counter-number) and the register
has no row for it.
*Task:* `planner/rejected.py: register the in-graph MoE expert fetch (#452) with its counter-number`

**#500-B12 — `link_disjointness()` UNKNOWN is documented as a refusal and is
not one.** No `raise`, no production caller, and `link_path_complete=False` is
hardcoded in bootstrap (`memtier/bootstrap.py:241`), so `DISJOINT` is
unreachable from real data — #423's striping gate has no working input.
*Task:* `memtier: make link_disjointness UNKNOWN an actual refusal and give link_path_complete a real source`

**#500-B16 — `--rank-tp-ratio auto` collapsing to the even split disarms the
family flags with a misleading message.** `auto` sets `rank_tp_ratio = None` on
uniform budgets (`server_args.py:9157`) with one INFO line;
`_handle_uneven_mlp_ratio` then raises "`{flag} … requires an active uneven-TP
base plan`" (`:10023`) naming a flag the operator DID pass. Loud rather than
silent, so lowest severity — but the text sends the reader to the wrong flag.
*Task:* `name the auto->even-split collapse in the family-vector refusal message`

**#500-B17 — the `matrix` transport advertises fewer ops than its own bar1
sub-path, and one of the missing ones is capturable.** `matrix` declares
`{all_reduce, all_to_all, all_to_all_single}`
(`barlink_matrix_transport.py:354`), a strict SUBSET of the bar1 set it wraps
(`{…, all_gather, broadcast}`, `barlink_bar1.py:1450`), contradicting the
"strictly more than bar1" comment at `barlink.py:302`. Because `matrix` is in
`GRAPH_ENABLE_TRANSPORTS`, a captured `all_gather` hard-aborts at
`barlink.py:660` instead of taking the path that exists underneath it.
*Task:* `barlink matrix: declare the ops its bar1 sub-path already serves (all_gather, broadcast)`

**#500-B18 — the marlin uneven-TP coarsening is gated on a quant-config CLASS
NAME.** `type(quant_config).__name__.lower() in _MARLIN_PACKABLE_CONFIGS` with
`("fp8config", "compressedtensorsconfig", "fbgemmfp8config")` (`linear.py:206`,
`:236`). `MarlinConfig`, `W8A8Fp8Config` and `QuarkConfig` are marlin-served,
expose no `weight_block_size`, and therefore receive NO coarsening — which is
the #377/#383 mid-tile abort reachable again through a different class. This is
the §12 quant-name-list family (#443/#446), second confirmed instance, in the
alignment code that family's own doctrine covers.
*Task:* `marlin coarsening: decide packability from the layer, not from the quant-config class name`

**#500-B19 — `--gguf-mmq-decode-threshold` is silently inert off two
architectures.** The measured table `_MMQ_BUCKET_MIN` has exactly two entries,
`(12,0)` and `(8,6)` (`gguf.py:539-542`, `:667`). On any other capability the
flag parses, validates and does nothing, with no log line.
*Task:* `gguf_mmq_decode_threshold: log (or refuse) on a capability with no measured bucket table`

**#500-B20 — the collective-decision recorder looks armed and is not.**
`SGLANG_BARLINK_RECORD_DECISIONS` is read ONCE at import
(`barlink_uniformity.py:205`), so setting it after start does nothing, and
`SGLANG_BARLINK_RECORD_DUMP_DIR` alone builds no recorder (`:250`). The standing
instrument for the rank-local-condition-before-a-collective family
(#94/#194/#312/#431) can therefore be enabled in a way that produces silence
during exactly the wedged run it exists for.
*Task:* `barlink decision recorder: read the enable env at call time and let DUMP_DIR imply recording`

## 5b. Debt discharged

**AUDIT_421 §8's one open question is CLOSED.** `PathProfile.saturation_threshold`
is permanently 1.0 (no writer anywhere), and no production code attaches a
saturation sensor, so `_utilization_locked` returns the no-sensor constant 0.0
(`barlink_path_dispatcher.py:387`) and the `>= 1.0` overflow re-route at `:357`
never fires today. It is not an oversight and not dead code: the one
production-intended sensor, `bus_saturation_sensor`, is BINARY (`return 1.0 if
stats.get("pending_demand") else 0.0`, `:415`), for which threshold 1.0 is
exactly the right value — the tier fires the moment #279's measured slice
attaches it. Correctly parked. (The dispatcher is additionally behind
`SGLANG_BARLINK_PATH_DISPATCHER=1` with an empty registry, so the status quo
holds regardless, `:428-443`.)

**AUDIT_421 F1 is FIXED at this tip and must not be carried forward.**
`--kv-pressure-ladder auto` resolves to a real table — the planner step-table
source is supplied in production (`ladder = build_ladder_from_server_args(
server_args, table_fn=auto_ladder_table_fn(server_args))`,
`managers/kv_pressure_runtime.py:467`); the `raise` at
`model_executor/kv_pressure_ladder.py:1956` now only catches direct callers.

**AUDIT_421's "`/session_handover` does not occur anywhere in `python/`" is
STALE.** The endpoint is at `http_server.py:1126` at this tip. So are
`--pp-stage-ratio` (`server_args.py:1437`) and `SGLANG_PP_SHAPE_CACHE`
(`environ.py:531`), both of which AUDIT_421 recorded as absent.

**AUDIT_421 §7.1's four catalog gaps were NOT discharged by the 2026-08-02
refresh** — `--kv-pressure-external-hysteresis-rounds`, `--kv-pressure-pre-stage`,
`--regime-gate-evidence` and the `--enable-weights-disk-backup` /
`--hibernate-dir` contract were all still absent from the catalog when this
audit started. Two are fixed here; the flag-surface gap generally is I-9.

## 6. Direction 2 — wired but uncatalogued

Method: fork-delta against `upstream/main` = `ec741e4161` (2026-08-02), one day
older than the audited tip `3b7569f664`, so every delta below is fork code and
not upstream drift. Three surfaces enumerated: HTTP routes (AST-free route-path
extraction over `@app.*` decorators incl. `api_route`), `ServerArgs` fields (AST
over the dataclass), `environ.py` entries (AST over the module's `NAME = Env*()`
assignments). Each name then matched against `docs/dev/FEATURE_CATALOG.md` and
`FEATURES_VS_UPSTREAM.md` in both underscore and `--dash` spelling.

Counts at the audited tip:

| surface | fork-added | absent from FEATURE_CATALOG | absent from BOTH docs |
|---|---|---|---|
| HTTP routes | 21 | 16 | 16 |
| `ServerArgs` fields | 166 | 145 | 123 |
| `environ.py` entries | 115 | — | 86 |

The matcher is literal, so a catalog line that writes a flag family in compact
form (`--lane-offload-profile/-class-policy/-park-targets`) reads as MISS for
the elided members. Those are noted as matcher artefacts below and are not
claimed as gaps.

### I-1 — a SECOND capability registry exists and disagrees with the catalog

`python/sglang/srt/planner/flags.py` calls itself "Authoritative flag/env
catalog … the single source of truth for EVERY sglang `ServerArgs` flag plus
EVERY fork-specific flag/env var" (`flags.py:15-21`). It carries `requires`,
`mutually_exclusive_with`, `allowed`, `model_compat` per flag and drives the
dashboard Runner tab. It is not referenced anywhere in FEATURE_CATALOG.md.

Its module docstring §"CRITICAL fork-capability note" (`flags.py:38-46`) has
carried the CORRECT general statement of the #492 capability all along:

> The fork CAN do uneven Tensor Parallelism with `tp_size` GREATER than the
> model's KV-head count (KV heads are replicated and the KV cache is token-
> sharded on the DCP axis) … Neither of those is marked model-incompatible
> here.

So the knowledge that the replication+token-shard axis is unconditional was in
the tree, in a file that calls itself authoritative, while the catalog
shorthand ("TP>kv_heads via replication+token shard") and two solver
generations read it as a special case. **The reach defect was a registry
disagreement, not missing knowledge.** Any future "is X possible" question must
consult `planner/flags.py` alongside the catalog — and where the two disagree,
neither wins: the code predicate does.

Three of `flags.py`'s own curated constraints are themselves NARROWER than the
runtime and are listed as bug candidates below (I-2, I-3, I-4).

### I-2 [NARROWER / BUG-CANDIDATE] — `flags.py` forbids `--rank-gpu-id` under PP, the runtime REQUIRES it

`planner/flags.py:594` declares
`mutually_exclusive_with=("mem_fraction_static", "pp_size", "dp_size", "ep_size", "nnodes", "base_gpu_id", "gpu_id_step")`
for `rank_gpu_id`.

The runtime says the opposite for `pp_size`:

- `server_args.py:9592` — `if self.pp_size > 1 and self.rank_gpu_id is None:` → raises
  "`--rank-tp-ratio with --pp-size > 1 … requires --rank-gpu-id to give each pipeline stage its own group of physical GPUs`".
- `server_args.py:8973` — `if self.pp_size > 1 and len(self.rank_gpu_id) != self.pp_size * self.tp_size:` → the world-length form is a validated, supported shape.
- `server_args.py:9655-9664` — `placed_ranks = self.tp_size * self.pp_size`, with a
  PP-specific error naming world-rank order `pp_rank * tp_size + tp_rank`.

So the dashboard catalog marks as impossible exactly the combination §1's
TPxPPxTP feature is built on, and its `tuple_len_flag="tp_size"` is also wrong
under a pipeline (the runtime wants `pp_size * tp_size`). A user configuring
cross-rig PP × uneven TP through the planner UI cannot express it.
No runtime refusal was found for `rank_gpu_id` × `dp_size` / `ep_size` /
`nnodes` either — those three are unverified in this audit, not confirmed.

Proposed task: *"planner/flags.py: rank_gpu_id is not exclusive with pp_size — fix the edge and the world-length rule (#500-I2)"*.

### I-3 [NARROWER / BUG-CANDIDATE] — `flags.py` makes `--rank-tp-ratio` require `--rank-gpu-id`; the runtime deliberately decouples them

`planner/flags.py:627` — `requires=("rank_gpu_id",)`.

`server_args.py:9540-9548` states and implements the opposite, and names the
configuration the decoupling exists for:

> `--rank-tp-ratio` … describes the PARTITION, not the placement. It therefore
> does NOT require `--rank-gpu-id`. Decoupling the two is what lets a rank be
> placed by something other than this process — e.g. the cross-vendor
> bring-up, where two launchers (`--nnodes 2 --node-rank 0/1`, one CUDA venv +
> one ROCm venv on ONE host) each place their own rank, and `--rank-gpu-id`
> could not describe the AMD rank anyway because it resolves devices through
> NVML.

Enforced by the control flow: the whole `--rank-tp-ratio` / `--rank-kv-ratio`
validation block is hoisted ABOVE the `if self.rank_gpu_id is None: … return`
early return at `server_args.py:9644`.

FEATURE_CATALOG §1 writes the pair as `**Uneven TP** `--rank-tp-ratio` + `--rank-gpu-id``,
which reads the same way. Under the MECHANISM REACH law that phrasing is a
reach claim and it is wrong: **uneven TP is available without any placement
flag**, which is the only way the cross-vendor (CUDA + ROCm) arm can be
expressed at all.

Proposed task: *"planner/flags.py: drop the rank_gpu_id requirement from rank_tp_ratio (blocks the cross-vendor two-launcher arm) (#500-I3)"*.

### I-4 [NARROWER / DOC+FLAGS] — `--rank-kv-ratio`: accepted values and its real requirement

- Accepted set in code: `coupled | capacity | auto | speed | <explicit vector>`
  (`planner/flags.py:731` `allowed=("coupled", "capacity", "auto", "speed")`;
  the vector form is validated at `server_args.py:9625-9642`).
  FEATURE_CATALOG §1 lists only `coupled|speed|vector` — it omits
  `capacity`/`auto`, which §2 then uses by name. §1 is the incomplete one.
- The requirement is NOT uniform across the modes. Only the EXPLICIT VECTOR is
  refused without a non-uniform base plan (`server_args.py:9608-9614`); the
  derived modes degrade to `coupled` with a warning
  (`server_args.py:9615-9624`, "the derived modes degenerate gracefully").
  `flags.py:730` states the flat `requires=("rank_gpu_id", "rank_tp_ratio")`,
  and `rank_gpu_id` is not required at all (see I-3 — this validation is hoisted
  above the `rank_gpu_id` early return).
- Asymmetry worth recording, because it is the opposite of `--rank-tp-ratio`'s
  rule: an ALL-EQUAL `--rank-kv-ratio` vector is LEGAL and meaningful
  (`server_args.py:9636-9642`, "uniform token ownership … under uneven
  weights"), whereas an all-equal `--rank-tp-ratio` is hard-refused
  (`server_args.py:9563-9567`, "`--rank-tp-ratio` with identical entries is the
  even split — omit the flag instead").

### I-5 [WIDER] — `--rank-tp-ratio auto-performance` is a value the catalog never names

`server_args.py:431-433` — `if value.strip() in ("auto", "auto-performance"):`.
`flags.py:632` — `allowed=("auto", "auto-performance")`.
`_CAPACITY_FIRST_DEFAULT_NOTICE` (`server_args.py:740-752`) exists precisely
because the default "reads as 'this IS the optimum'", and names
`auto-performance` as the per-task optimizer, per `--rank-perf-tune` target.

FEATURE_CATALOG §1 names only `auto` and then attributes the solve to
`--rank-perf-tune`; §2 names the optimizer in prose but not the value that
engages it. So the catalog describes a solver that a reader cannot switch on.
`_RANK_PERF_TUNE_CHOICES` (`server_args.py:723-730`) is
`both | dec | enc | maxkv | phase-prefill | phase-decode` — six, and §1 lists four.

### I-6 [WIDER — the §2 premise is false] — three named family plans exist, and the mechanism has no family whitelist

FEATURE_CATALOG §2 justifies not installing a solved attention vector with:

> the only runtime actuator for an attention vector is `--rank-tp-ratio`, since
> "mlp" is the sole named family plan.

Both halves fail against the predicates.

1. **Three family plans are installed, not one.** `uneven_family_plans`
   (`managers/scheduler.py:5844-5875`) loops over
   `(("mlp", "rank_mlp_ratio"), ("moe", "rank_moe_ratio"), ("vocab", "rank_vocab_ratio"))`
   and installs each as a named family vector. Consumers read them by name:
   `get_tp_partition_ratios("moe")` at `layers/moe/expert_stats.py` and
   `layers/moe/expert_compute_placement.py`, the vocab vector at
   `distributed/utils.py:901`.
2. **The mechanism accepts ANY family name.** `_normalize_partition_plan`
   (`distributed/utils.py:129-160`) validates exactly three things — a base plan
   must exist, the vector length must equal the base's, entries must be positive
   ints — and applies no name whitelist, no enumeration, no membership test.
   `get_tp_partition_ratios(family)` (`distributed/utils.py:162`) is a plain
   dict lookup with base-plan fallback.
3. What actually pins attention to the base plan is much narrower than "a
   runtime change": no attention layer passes `tp_family=`. Across the tree the
   only declared values are `tp_family="mlp"` (8 sites) and `tp_family="moe"`
   (1 site); `tp_family` is an ordinary optional argument on the linear layers
   (`layers/linear.py:534, 847`) that is honoured through
   `tp_plan_active(self.tp_size, self.tp_family)` (`layers/linear.py:657`).

So the #485 joint prefill solve's "REPORTS the pair and does not install it" is
gated by a missing constructor argument plus a flag, not by the absence of a
mechanism — the same shape as #492, in the sentence that justifies leaving the
axis unsolved. This is the highest-leverage find of the inverse sweep.

### I-7 [gap] — the four `SGLANG_UNEVEN_*` vector actuators are in no catalog

`environ.py:573, 574, 580, 588` declare `SGLANG_UNEVEN_MLP_VECTOR`,
`SGLANG_UNEVEN_MOE_VECTOR`, `SGLANG_UNEVEN_VOCAB_VECTOR`,
`SGLANG_UNEVEN_TOKEN_VECTOR`. They are not tuning trivia — each OVERRIDES its
CLI flag (`flags.py:696, 709, 722, 740`; `uneven_perf.py:5650-5653` prints
which of the two pinned the vector), they are the pinning mechanism the planner
emits into a launch (`planner/runner.py:411, 1175`), and `planner/flags.py`
registers them as first-class entries (`flags.py:861-895`). FEATURE_CATALOG
mentions none of the four.

### I-8 [gap] — 16 fork HTTP routes are absent from §13's "Serving surface"

Fork-only routes at this tip (21), against §13:

- Listed by §13: `/session_handover`, `/kv_reshard`, and the training tenant /
  idle workbench in prose.
- **Absent from the catalog entirely:**
  `POST /vram_budget` (`http_server.py:1149`) — the runtime VRAM dial's actuator;
  §3 describes the dial in prose and names no way to drive it.
  `GET|POST /hibernate` (`http_server.py:1706`) — §3 describes hibernate, not the endpoint.
  `POST /v1/images/generations` (`http_server.py:2013`), `/v1/images/edits` (2022),
  `/v1/images/variations` (2057) — OpenAI-compatible **diffusion lane** routing.
  `POST /v1/audio/speech` (`http_server.py:2067`) — OpenAI-compatible **speech lane**.
  `/v1/files` + 4 sub-routes, `/v1/fine_tuning/jobs` + 5 sub-routes,
  `/v1/fine_tuning/tenant` (`http_server.py:2079-2170`, "#341-M1").
  `/x-htsglang/workbench{,/events,/pause,/enqueue}` (`http_server.py:2191-2236`).

The image/speech pair is unconditionally wired — `fast_api_app.state.openai_serving_images = OpenAIServingImages()`
and `…_speech = OpenAIServingSpeech()` at `http_server.py:461-462`, no flag —
and it is an honest surface rather than a stub: `serving_images.py:62-100`
refuses with the registry's own numbers when no class-2 lane is HOT
(`LaneUnavailable`, "promote a diffusion engine to HOT via the registry control
plane"), and `edits`/`variations` return a named 501
(`serving_images.py:192-199`). Given the ONE-RUNTIME law this is exactly the
surface the catalog should be advertising.

Two further route groups live outside `http_server.py` and are in neither doc:
`video_enhance/server.py` — `/v1/video/{enhance,plan,capabilities,engines,tracks,liveness}`
plus `/v1/video/enhance/{job_id}` and `/v1/video/preview/{job_id}/{which}` (§13
describes the video chain at length and lists no endpoint), and
`registry/http_api.py` — `/registry{,/cards,/engines,/engines/{id},/engines/{id}/pin,/engines/{id}/state,/idle,/plan,/default_hot}`,
the control plane the diffusion refusal above tells the operator to use.

### I-9 [gap] — capability families with a flag surface and no catalog entry

Absent from BOTH docs, and not matcher artefacts (the family's own ENABLING
flag is missing, not just its tuning knobs):

- `--enable-deepep-waterfill` (`server_args.py:3741`) — no mention anywhere.
- `--mamba-checkpoint-interval` (`server_args.py:3812`) + `SGLANG_MAMBA_CKPT_WINDOW`,
  `SGLANG_MAMBA_CKPT_STRICT_RESUME`, `SGLANG_MAMBA_CKPT_DEBUG` — a whole
  mamba/GDN state-checkpointing capability with zero catalog presence.
- `--client-liveness-timeouts` and its three siblings (`server_args.py:5636-5661`)
  — `DESIGN_344_liveness.md` exists; the catalog does not.
- `--attn-scratch-budget-mib` (`server_args.py:1331`).
- `--draft-kv-layout` (`server_args.py:1499`) — §1 describes Draft-KV-DCP and
  never names its flag.
- `--determinism-logits-dump-dir` (`server_args.py:5366`) — §10's determinism
  work has an instrument the catalog omits.
- `--disaggregation-topology` + `--disaggregation-prefill-{gpus,layer-split,budget-mib,lane-interval}`
  (`server_args.py:4237-4305`) — §5 describes PD disaggregation and names no flag.
- `--enable-vram-dial` / `--vram-budget-mib` (`server_args.py:5067, 5086`),
  `--enable-training-tenant`, `--enable-idle-workbench` (`5453, 5537`),
  `--enable-weights-disk-backup` (`5392`), `--gdn-state-set-ladder` /
  `--gdn-resident-state-slots` (`4780, 4802`) — in every case the catalog has
  the prose and not the switch.
- `--speculative-draft-gpu`, `--speculative-drafter-policy`,
  `--speculative-adaptive-graph-memory`, `--speculative-cross-algorithm-{force,ctx-gate,lazy-capture,retire-ctx}`
  (`server_args.py:3380-3555`) — §4 describes all of this behaviour in prose
  and names none of the flags.
- `--rank-kv-capacity-seed`, `--rank-kv-speed-weights`,
  `--rank-perf-loose-ctx-percent` (`server_args.py:2521, 2528, 2274`).
- `--weightless-kv-{head-rank,chunked-block-size,host-spill-tokens,spill-device-cap}`
  (`server_args.py:1655-1750`) — §6 in prose only.

AUDIT_421 §7.1 already named `--kv-pressure-external-hysteresis-rounds`,
`--kv-pressure-pre-stage`, `--regime-gate-evidence` and the
`--enable-weights-disk-backup` / `--hibernate-dir` contract as catalog gaps.
**All four are still absent at this tip** — the 2026-08-02 "full refresh" did
not discharge them.

### I-10 [gap] — 86 fork env vars in neither doc

115 fork-added `environ.py` entries, 29 documented, 86 absent. Beyond I-7, the
ones that carry capability rather than tuning:

- `SGLANG_MOE_COMPUTE_POLICY`, `SGLANG_MOE_COMPUTE_BASE_PLAN` — the #439
  expert-compute-placement path §1/§3 describe at length.
- `SGLANG_MOE_HOST_SHARD_MIN_PROVENANCE`, `SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE`
  — the #394 provenance chain's own overrides.
- `SGLANG_MEASURED_KV_BUDGET{,_SAFETY_MIB,_CTX_ALLOWANCE_MIB}` — §16 names the
  "measured-KV-budget stale-boot trap" and not the knob that arms it.
- `SGLANG_RANK_CARD_UUIDS`, `SGLANG_RANK_CARD_PROBE_CUDA` — §11 device identity.
- `SGLANG_POISON_GRAPH_PAD`, `SGLANG_POISON_POOL_DATA` — falsifier instruments
  for the §12 pad-slot family.
- `SGLANG_SPEC_STATE_HASH{,_MAX_MB}`, `SGLANG_SPEC_RESET_PROBE{,_FILTER}`,
  `SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC` — the §10 hetero-determinism
  instruments.
- `SGLANG_EXPERT_STATS{,_INTERVAL_SEC,_PATH}` — §16 names `expert_stats`, not
  how to turn it on.
- twelve `SGLANG_BARLINK_*` (chunk/slot/ring/timeout/path-dispatcher/fp32-reduce)
  against §7's three.
- `SGLANG_WL_GRAPH_MAX_BS`, `SGLANG_WL_H2D_PREFETCH` — §6 weightless lane.
- `SGLANG_GGUF_MXFP4_REPACK` — §8 names `SGLANG_GGUF_MXFP4_NATIVE=0` and not
  its counterpart.
- `SGLANG_PERF_{REPROBE,PROBE_TIMEOUT_S,PROBE_LINK_TIMEOUT_S,PROBE_SKIP_LINKS}`
  and four `SGLANG_PERF_*` cost-model exponents — the §2 generality story's own
  probe controls.

An operator cannot discover any of these from the catalog. The recommendation
is NOT to inline 86 names into FEATURE_CATALOG: it is to make
`planner/flags.py` the named env/flag registry the catalog points at (I-1), and
to close the capability-level gaps in I-9 by name.

## 7. Per-section tables


### §1 — Uneven parallelism

Tree: `/spinning/wt-500-reach`. Read: `CLAUDE.md` (full), `FEATURE_CATALOG.md` §1 + §17.
Method: every conditional claim in §1 extracted, then the REAL gate located and its
predicate read at source (not docstring, not CLI help). All paths relative to
`/spinning/wt-500-reach/`.

Counts: **WIDER 9 · NARROWER 10 · EXACT 18 · NOT-FOUND 2** = 39 items, of which
**3 are bug-candidates** (S1-14, S1-33, S1-04 → B1/B2/B3 below).

| ID | Catalog claim (short) | Class | Gate predicate (verbatim) | file:line | Note |
|---|---|---|---|---|---|
| S1-01 | "Uneven TP `--rank-tp-ratio` + `--rank-gpu-id`" — reads as one coupled pair | WIDER | `if self.rank_gpu_id is None:` → `self._handle_uneven_mlp_ratio(); return` — every `--rank-tp-ratio` check sits ABOVE this early return | python/sglang/srt/server_args.py:9644 | `--rank-tp-ratio` is a pure PARTITION description and is legal with NO `--rank-gpu-id` at all (comment 9538-9549 states it, checks 9552-9567 enforce it). The converse is NOT true: `--rank-gpu-id` alone is refused (S1-02). |
| S1-02 | (same pair, other direction) | EXACT | `if self.rank_gpu_memory_mib is None and self.rank_gpu_id is not None: raise ValueError(...)` | python/sglang/srt/server_args.py:9434 | `--rank-gpu-id` DOES require `--rank-gpu-memory-mib` (or `--rank-tp-ratio auto`). |
| S1-03 | "`auto` = byte-proportional from NVML totals minus auto reserve" | NARROWER (doc) | `if self.rank_gpu_id is None: raise ValueError("--rank-tp-ratio auto requires --rank-gpu-id.")` | python/sglang/srt/server_args.py:8971 | `auto` (unlike an explicit vector) DOES require placement, because it reads NVML per named card. Catalog omits this asymmetry. |
| S1-04 | `auto` always yields an uneven vector | NARROWER (bug: message) | `if len(set(weights)) == 1: ... self.rank_tp_ratio = None` | python/sglang/srt/server_args.py:9157-9162 | On uniform budgets `auto` collapses to the EVEN split (`rank_tp_ratio=None`), which then disables every family flag that "requires an active plan" (S1-10/11). Silent except for one INFO line. |
| S1-05 | "with `--rank-perf-tune both\|dec\|enc\|maxkv` the planner solves the vector" | WIDER | `_RANK_PERF_TUNE_CHOICES: Tuple[str, ...] = ("both","dec","enc","maxkv","phase-prefill","phase-decode")` | python/sglang/srt/server_args.py:723-730 | Two more targets exist: the #354/#357 PHASE-OPTIMAL arms. Catalog's four-value list under-reports the axis. |
| S1-06 | (perf flags are part of auto-performance) | EXACT | `if not ratio_was_perf: if loose != 0.0 or self.rank_perf_tune != "both": raise ValueError(...)` | python/sglang/srt/server_args.py:8904-8909 | Also: `dec` auto-selects `--rank-kv-ratio speed`, `maxkv` auto-selects `capacity`, only while kv-ratio is at its default (8933-8950). |
| S1-07 | auto-performance is available generally | NARROWER (doc) | `if ratio_was_perf and self.pp_size > 1: raise ValueError(...)` | python/sglang/srt/server_args.py:9394-9406 | `auto-performance` is refused by name under PP; only plain `auto` has the per-stage path. Catalog states the PP agreement gate for `auto` but not the refusal for `auto-performance`. |
| S1-08 | "16-element MLP family" | EXACT | `bad = [r for r, s in enumerate(sizes) if s % ACTIVATION_VEC_ELEMS]` (`ACTIVATION_VEC_ELEMS = 16`), guarded by `if not tp_plan_active(tp_size, family): return` | python/sglang/srt/distributed/utils.py:975-988, 920 | Boot-time refusal only under an installed plan; even-split path keeps the runtime-only kernel check. |
| S1-09 | "coupled-dim rule: gate_up output and down_proj input … must coarsen identically" | EXACT | `n, k = _marlin_min_thread_pair(); return math.lcm(n, k)` (= 128, SYMMETRIC on both axes) | python/sglang/srt/layers/linear.py:181-182 | Same rule re-applied for asymmetric exposed blocks (MXFP8 `[1,32]` → lcm → 32 → 128) at linear.py:271-285. |
| S1-10 | "per-layer family table for `block_configs` models (Nemotron-Puzzle class)" | NARROWER (doc) | `blocks = text.get("block_configs")` → `if not blocks: return LayerFamilyCensus(...uniform...)` | python/sglang/srt/uneven_perf.py:2894-2896 | This is the PLANNER's weight-byte census (`LayerFamilyCensus`, #371) feeding the auto-performance cost model — NOT a runtime per-layer `tp_family` shard table. Catalog files it under "Unit system", which over-reads it. |
| S1-11 | `--rank-mlp-ratio` sibling (implicitly needs a plan) | EXACT | `if not isinstance(self.rank_tp_ratio, list): raise ValueError(f"{flag} / {env_name} requires an active uneven-TP base plan …")` | python/sglang/srt/server_args.py:10023-10028 | Applies to mlp/moe/vocab uniformly; env var wins over flag (10001-10018). |
| S1-12 | `--rank-vocab-ratio` sibling | NARROWER (doc) | `if value is not None and self.enable_dp_lm_head: raise ValueError("--rank-vocab-ratio is not compatible with --enable-dp-lm-head …")` | python/sglang/srt/server_args.py:9911-9915 | Second, unnamed refusal. Also `auto` is CACHE-ONLY (never probes) and falls back to the resolved tp-ratio (9960-9962). |
| S1-13 | "`--rank-kv-ratio` (`coupled\|speed\|vector`)" | **WIDER** | `if mode in ("coupled", "capacity", "speed"): return mode` / `if mode == "auto": return "capacity"` | python/sglang/srt/server_args.py:484-487 | Catalog omits the `capacity` mode entirely (and its alias `auto`) — the measured capacity-optimal one-boot-convergence vector, and the mode `--rank-perf-tune maxkv` selects. Five accepted forms, not three. |
| S1-14 | `--rank-kv-ratio` non-coupled "requires `--rank-gpu-id` with a non-uniform `--rank-tp-ratio` plan" (CLI help) | WIDER + BUG | raise fires only when `self.rank_gpu_id is None and self.rank_gpu_memory_mib is None and self.rank_tp_ratio is None` … `if self.uneven_kv_flag_active(): raise ValueError(...)` | python/sglang/srt/server_args.py:9365-9383 | Reach: `--rank-tp-ratio 2,1,1 --rank-kv-ratio speed` with NO `--rank-gpu-id` passes validation. BUT the uneven-DCP auto-set (`self.dcp_size = self.tp_size`) sits BELOW the `rank_gpu_id is None` early return at 9644, so that boot keeps `dcp_size=1` and the flag is SILENTLY INERT. See bug-candidates. |
| S1-15 | `--rank-kv-ratio speed` "requires --rank-tp-ratio auto-performance; without it degrades to capacity" | EXACT | `bw = self.server_args.rank_kv_speed_weights; if not bw or len(bw) != self.dcp_size or any(w <= 0 for w in bw):` → warn + fall back | python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:4963-4972 | The real predicate is "are bandwidth scores present", written only by `apply_auto_performance` (uneven_perf.py:7004). Degradation is a warning, not a refusal. |
| S1-16 | `--rank-moe-resident-fraction` "GPU/host split WITHIN a rank" | NARROWER (doc) | `if moe_size is not None and tp_size is not None and moe_size != tp_size: raise ValueError("a per-rank resident-fraction vector is not supported when the MoE parallel group differs from the attention-TP group …")` | python/sglang/srt/layers/moe/resident_fraction.py:132 | Two unnamed restrictions: the vector form is refused when `moe_tp_size != tp_size`, and env/flag disagreement is refused outright (resident_fraction.py:107). |
| S1-17 | `--rank-auto-reserve-mib` listed as a `--rank-gpu-id` sibling | **WIDER** | `if str(self.rank_auto_reserve_mib) != str(ServerArgs.AUTO_RANK_MEMORY_RESERVE_MIB): self._apply_reserve_based_mem_fraction()` — inside the branch where all three rank flags are None | python/sglang/srt/server_args.py:9370-9377 | (#332) On a plain one-rank-per-GPU boot with NO uneven flags at all, a pinned reserve sizes `mem_fraction_static = (NVML total - reserve)/total` exactly. That is a stock-topology capability the catalog does not mention. |
| S1-18 | `--rank-gpu-memory-mib` "absolute per-rank MiB budget" | WIDER | `if (self.rank_tp_ratio is None and not self.weightless_kv_fastlane and self.pp_size == 1): raise ValueError("--rank-gpu-memory-mib as a list requires --rank-tp-ratio …")` | python/sglang/srt/server_args.py:9694-9703 | The per-rank LIST form is also legal without any weight vector under the weightless-KV fast lane and under PP>1 — two exemptions the catalog's "per-rank MiB budget" line does not carry. |
| S1-19 | "physical impossibility" budget check | EXACT | `if required_mib > total_mib: raise ValueError(f"Physical impossibility: …")` where `required_mib = sum(budgets[r] for r in ranks)` over `ranks = [r for r,g in enumerate(self.rank_gpu_id) if g == gpu_id]` | python/sglang/srt/server_args.py:9773-9784 | NVML totals resolved through the #331 identity map (the card the ordinal BINDS, not the same NVML index). |
| S1-20 | `--rank-auto-reserve-mib auto` infeasibility note | EXACT | `if str(self.rank_auto_reserve_mib) != self.AUTO_RANK_MEMORY_RESERVE_MIB: return None` then `fits = max(derived - short_mib * colocated, 0)` | python/sglang/srt/server_args.py:10605, 10625 | The note is ADVISORY text attached to an existing shortfall refusal (consumed at model_runner_kv_cache_mixin.py:720); it is not itself a gate, and it is silent under a pinned reserve. |
| S1-21 | "#82 GGUF expert-dim shard moves whole experts … every other MoE path splits the intermediate dim" | EXACT | `self._gguf_expert_shard = (quant_config is not None and quant_config.get_name() == "gguf" and tp_plan_active(self.moe_tp_size, self.moe_tp_family))` | python/sglang/srt/layers/moe/fused_moe_triton/layer.py:440-444 | |
| S1-22 | `--rank-moe-ratio link` "Refused by name when offload is off" | EXACT | `if cold_total <= 1e-9: raise NoComputeLever("link-proportional expert compute placement has nothing to move …")` where `cold_total = 1.0 - sum(f[r]*b[r])` | python/sglang/srt/layers/moe/expert_compute_placement.py:527 | It is a MASS test on the resident-fraction vector, not a test of an "offload enabled" flag — but since offload IS `resident fraction < 1` (no separate switch), the two coincide. Runs in the LAUNCHER (engine.py:691). |
| S1-23 | `link` refused "when the link provenance is `absent`" | EXACT | `if ratio.provenance == "absent": raise NoComputeLever(...)` | python/sglang/srt/layers/moe/expert_compute_placement.py:951 | Plus a WARNING (not a refusal) when the ratio came from `SGLANG_MOE_HOST_SHARD_RATIO` and is equal (960-979). |
| S1-24 | `link` refused "under `ep_size>1`" | EXACT | `if ep_size > 1: raise NoComputeLever(f"--rank-moe-ratio {symbol} is a TP-group placement, but ep_size={ep_size} …")` | python/sglang/srt/layers/moe/expert_compute_placement.py:916 | |
| S1-25 | (catalog names exactly THREE `link` refusals) | NARROWER (doc) | `if not isinstance(base_plan, list) or len(base_plan) != world: raise NoComputeLever(...)` and `if not cards.present: raise NoComputeLever(...)` | python/sglang/srt/layers/moe/expert_compute_placement.py:906, 941 | Two further named refusals: no resolved uneven-TP base plan, and no rank→physical-card vector (#392/#397 chain). Five refusals in total, all in the launcher. |
| S1-26 | "Resolved ONCE in the launcher — a symbolic value that reaches a worker is a hard error there" | EXACT | `if isinstance(vector, str): raise ValueError(f"{field}={vector!r} is a symbolic placement that must be resolved before the workers are spawned …")` | python/sglang/srt/managers/scheduler.py:5866-5872 | Launcher entry point: entrypoints/engine.py:691. Env channel additionally refuses symbols: server_args.py:9986-10000. |
| S1-27 | `link-calibrated` coefficients "required and read ONLY under this symbol"; plain `link` refuses while the variable is set | EXACT | `if symbol == COMPUTE_PLACEMENT_LINK: if env_coefficients is not None: raise NoComputeLever(...)` / `if env_coefficients is None: raise NoComputeLever(...)` | python/sglang/srt/layers/moe/expert_compute_placement.py:831-855 | Exactly the #458 split the catalog describes. |
| S1-28 | **"Uneven DCP … TP>kv_heads via replication+token shard"** | **WIDER** | `return dcp_size > 1 and get_tp_partition_ratios() is not None` | python/sglang/srt/distributed/utils.py:354 | THE OCCASION. The predicate never reads a kv-head count. Replication+token-shard is live for ALL kv-head counts, at any `dcp_size > 1` (not only `dcp_size == tp_size`), and covers both the even-modulo and the weighted owner rule. |
| S1-29 | (same, "requires `--rank-tp-ratio`") | **WIDER** | flashinfer: `self.uneven_dcp = (uneven_dcp_kv_replicated(self.dcp_size) or self.weightless_kv) and not _draft_replicated` | python/sglang/srt/layers/attention/flashinfer_backend.py:689-691 | SECOND PATH: the weightless-KV fast lane reaches the identical token-shard/LSE-merge machinery with NO `--rank-tp-ratio`. THIRD PATH: `get_tp_partition_ratios` reads the context-local overlay first (`overlay = _TP_PARTITION_OVERLAY.get()`, utils.py:168-172), so a #274 dual-group lane's scoped plan satisfies the predicate too. The triton backend has only the first path (triton_backend.py:700-704). |
| S1-30 | (TP>kv_heads geometry itself) | NARROWER (doc) | `return tp_plan_active(tp_size) and total_num_kv_heads < tp_size` | python/sglang/srt/distributed/utils.py:1048 | `kv == tp` is deliberately EXCLUDED (a `<`→`<=` flip was tried and reverted on measurement, docstring 1028-1047). And a target model needing this geometry without DCP spanning the TP group is refused: `attn_kv_replicated(...) and not is_draft_worker and not self.uneven_dcp` → raise (flashinfer_backend.py:735-748). |
| S1-31 | "Uneven DCP … SWA-hybrid support" (no MLA caveat) | NARROWER (doc) | `ratios = get_tp_partition_ratios(); if not ratios or len(set(ratios)) == 1: return` … else `raise NotImplementedError(f"{fn} does not support uneven tensor parallelism …")` | python/sglang/srt/layers/dcp/comm.py:81-93 | The MLA DCP collectives (`cp_lse_ag_out_rs_mla`, `all_gather_q_for_mla_decode`) have NO uneven-TP variant and refuse by name. §1 never states that uneven TP is MHA/GQA-only on the DCP combine. |
| S1-32 | "LSE log base follows the attention backend (FlashMLA = natural log)" | EXACT | `NATURAL_LOG_LSE_ATTENTION_BACKENDS = frozenset({"flashmla"})` → `return attention_backend in NATURAL_LOG_LSE_ATTENTION_BACKENDS` | python/sglang/srt/layers/dcp/comm.py:44, 54 | Single consumer wired at models/deepseek_common/attention_forward_methods/forward_mla.py:817. |
| S1-33 | "**Draft-KV-DCP**: draft KV token-sharded" | NARROWER + BUG | `_uneven_weighted_dcp = (… and os.environ.get("SGLANG_UNEVEN_DCP","0") == "1" and os.environ.get("SGLANG_UNEVEN_DCP_WEIGHTED","0") == "1" and self.rank_tp_ratio is not None and len(set(self.rank_tp_ratio)) > 1 and self.dcp_size > 1 and self.dcp_size == self.tp_size)` | python/sglang/srt/server_args.py:7448-7456 | Hard-requires the ENV PAIR. The sibling gate in `_handle_dcp_validation` (which runs EARLIER in `__post_init__`: :5750 vs :5933) accepts the flag route for the same shape (`… or self.uneven_kv_flag_active()`, server_args.py:7628-7630), so `--rank-kv-ratio speed --draft-kv-layout dcp` is refused although that boot IS on the weighted path. See bug-candidates. |
| S1-34 | "above TP>kv_heads, replicated is the DEGRADED layout" | NOT-FOUND | — | — | No predicate anywhere compares `tp_size` to `num_kv_heads` to select, switch or warn about `draft_kv_layout`. The threshold exists only as CLI help prose (server_args.py:1513-1528). Greps: `draft_kv_layout`, `DEGRADED`/`degraded`, `attn_kv_replicated`, `num_kv_heads` × `draft`, `draft_pool_is_replicated`. |
| S1-35 | "**TPxPPxTP** … `--pp-stage-ratio` … `SGLANG_PP_SHAPE_CACHE`" (AUDIT_421: absent at an older tip) | EXACT — **both PRESENT at this tip** | `if self.pp_layer_ratio is not None: raise ValueError("--pp-stage-ratio derives the layer split that --pp-layer-ratio spells out explicitly …")`; `if not self._pp_shape_cache_enabled: return self.send_object(...)` | python/sglang/srt/server_args.py:12871 (flag def 1437), python/sglang/srt/distributed/parallel_state.py:2122 (env `SGLANG_PP_SHAPE_CACHE` at environ.py:531) | Rechecked at THIS tip, not copied from AUDIT_421. `--pp-stage-ratio` also refuses when depth is unreadable (12879-12884). |
| S1-36 | "pipeline across rigs with per-stage TP groups" | EXACT | `shared = sorted(set(groups[stage_a]) & set(groups[stage_b])); if shared: raise ValueError(…)` and `if any(vec != stage_vectors[0] for vec in stage_vectors[1:]): raise ValueError(…)` | python/sglang/srt/server_args.py:8669-8680, 9116-9131 | Disjoint per-stage card groups are the admission condition; `auto` under PP additionally requires all stages to derive the SAME vector (the agreement gate the catalog names). |
| S1-37 | "**TP5+ emulation** via **NCCL** multi-rank co-location" | **WIDER** | `rank_gpu_id = server_args.rank_gpu_id; if rank_gpu_id is None or len(rank_gpu_id) == len(set(rank_gpu_id)): return` | python/sglang/srt/entrypoints/engine.py:1563 | Co-location is gated ONLY on duplicate entries in `--rank-gpu-id`. What follows is env TUNING, never a requirement: `NCCL_MULTI_RANK_GPU_ENABLE=1`, `NCCL_NVLS_ENABLE=0`, `NCCL_MAX_CTAS=max(1, 8//max_colocated)` — each only `if os.environ.get(...) is None`. MPS is a WARNING (`if not _mps_control_daemon_responsive(): logger.warning`, engine.py:1598). No predicate anywhere excludes barlink: it is selected by `if envs.SGLANG_BARLINK.get() and self.world_size > 1:` (parallel_state.py:687), which cannot see the placement. |
| S1-38 | (co-location's only transport consequence) | EXACT | `colocated = len(set(self.rank_gpu_id)) < len(self.rank_gpu_id)` → `if (heterogeneous or colocated or uneven_plan) and not self.disable_custom_all_reduce: self.disable_custom_all_reduce = True` | python/sglang/srt/server_args.py:9820-9832 | The ONLY thing co-location forces is turning custom all-reduce OFF. |
| S1-39 | (uneven-DCP auto-engage) | NOT-FOUND (as stated) | `if (uneven_plan and self.dcp_size == 1 and (os.environ.get("SGLANG_UNEVEN_DCP","0") == "1" or self.uneven_kv_flag_active())): self.dcp_size = self.tp_size` | python/sglang/srt/server_args.py:9845-9853 | §1 says "Uneven DCP (`dcp_size` + token vector)" without saying WHO sets `dcp_size`. It is auto-set here — but only inside the `--rank-gpu-id` branch (below the 9644 early return), which is the S1-14 defect. |

---

#### WIDER finds — what they unlock

**S1-28 (the occasion, re-confirmed).** `uneven_dcp_kv_replicated` at
`distributed/utils.py:354` is `dcp_size > 1 and get_tp_partition_ratios() is not None`.
Two things the catalog's "TP>kv_heads via replication+token shard" hides: the kv-head
count is never read (so replication+token-shard is available at ANY kv count, including
kv ≥ tp, and is the second distribution axis for the attention family everywhere), and
the gate is `dcp_size > 1`, not `dcp_size == tp_size` — a DCP group smaller than the TP
group already engages the replicated-KV token-shard layout. Any solver that treated
attention as grid-pinned by kv-heads was excluding configurations this predicate admits.

**S1-29 (two more paths to the same machinery).** The weighted-DCP token-shard/LSE-merge
machinery is NOT reachable only through `--rank-tp-ratio`. In the flashinfer backend
`self.uneven_dcp = (uneven_dcp_kv_replicated(...) or self.weightless_kv) and not
_draft_replicated` (`flashinfer_backend.py:689-691`) — the weightless-KV fast lane runs the
identical owner rule with no weight vector at all. And `get_tp_partition_ratios` consults
the context-local overlay BEFORE the process-global plan (`utils.py:168-172`), so a #274
dual-group lane running under `scoped_tp_partition_ratios` satisfies the predicate inside
its own context. A design that needs token-sharded KV therefore has three entry points,
not one.

**S1-13 (`--rank-kv-ratio capacity` is missing from the catalog).** The parser accepts
`coupled | capacity | auto(→capacity) | speed | <vector>` (`server_args.py:484-487`).
`capacity` is the measured, one-boot-convergence mode that installs the capacity-optimal
token vector after the post-weight-load profiling and maximises `max_total_num_tokens` —
and it is what `--rank-perf-tune maxkv` selects (`server_args.py:8944-8949`). Anyone
reading §1's `coupled|speed|vector` would conclude that maximising context requires a
hand-pinned vector, when a derived mode exists.

**S1-37 (co-location is not NCCL-bound).** The only co-location predicate in the tree is
`len(rank_gpu_id) == len(set(rank_gpu_id))` (`engine.py:1563`); everything downstream is
opportunistic env tuning that respects an operator override, and MPS absence is a warning
only. `SGLANG_BARLINK` is selected by `envs.SGLANG_BARLINK.get() and self.world_size > 1`
(`parallel_state.py:687`) with no visibility into placement, so the barlink transports are
reachable under multi-rank co-location — the catalog's "via NCCL" reads as a hard
dependency that no predicate expresses. (Whether barlink's bar1/device path performs well
with two ranks on one card is unmeasured; the point is that nothing REFUSES it, and per
the barlink-standard order the barlink arm is the one to try first.)

**S1-17 (`--rank-auto-reserve-mib` works on a stock topology).** §1 lists it as one of the
`--rank-gpu-id` siblings, but `server_args.py:9370-9377` takes a pinned reserve on a boot
with none of the uneven flags set and turns it into an exact
`mem_fraction_static = (NVML total − reserve)/NVML total`. That gives even, one-rank-per-GPU
rigs an absolute-MiB sizing knob without opting into heterogeneous placement at all.

**S1-01 / S1-18 / S1-05 (smaller ones).** `--rank-tp-ratio` alone (no placement flags) is a
legal, validated launch — the cross-vendor two-launcher bring-up depends on it
(`server_args.py:9538-9549`). A per-rank `--rank-gpu-memory-mib` LIST is legal without any
weight vector under the weightless-KV lane and under PP>1 (`9694-9703`). And
`--rank-perf-tune` has six targets, not four (`723-730`).

---

#### NARROWER — bug candidates

**B1 — `--rank-kv-ratio` without `--rank-gpu-id` is accepted and then silently inert (S1-14/S1-39).**
Validation of `--rank-kv-ratio` was deliberately hoisted above the `if self.rank_gpu_id is
None: … return` at `server_args.py:9644` (it describes the partition, not the placement),
but the uneven-DCP auto-engage `self.dcp_size = self.tp_size` sits BELOW it at
`server_args.py:9845-9853`. So `--rank-tp-ratio 2,1,1 --rank-kv-ratio speed` with no
`--rank-gpu-id` passes every check, keeps `dcp_size == 1`, never installs a token vector
(the scheduler gate is `_dcp_size > 1 and base_plan is not None and
uneven_weighted_dcp_enabled()`, `managers/scheduler.py:5952-5956`), and nothing warns —
`reject_silently_inert_dcp` is itself gated on `dcp_size`. Harm: a user asks for the #210
decode lever and serves the coupled layout, and the boot log looks normal.
Task title: **"Uneven-DCP auto-engage must run on the no-placement path too (or refuse `--rank-kv-ratio` there by name)"**

**B2 — `--draft-kv-layout dcp` refuses the flag route to weighted DCP (S1-33).**
`_reject_unsupported_draft_kv_dcp` requires `SGLANG_UNEVEN_DCP=1` AND
`SGLANG_UNEVEN_DCP_WEIGHTED=1` literally (`server_args.py:7448-7456`), while the
speculation×DCP admission gate in `_handle_dcp_validation` — which runs earlier in
`__post_init__` (:5750 vs :5933) — accepts the same configuration expressed
through the flag (`… or self.uneven_kv_flag_active()`, `server_args.py:7628-7630`). A boot
with `--rank-kv-ratio speed|capacity|<vector>` IS on the weighted owner rule with
`dcp_size == tp_size`, and #108 refuses it anyway. Harm: the −67 % draft-KV win is
unreachable for every boot that uses the supported flag instead of the legacy env pair.
Task title: **"#108 draft-KV-DCP gate must accept the `--rank-kv-ratio` route to weighted DCP, not only the env pair"**

**B3 — `--rank-tp-ratio auto` collapsing to the even split silently disarms the family flags (S1-04).**
`auto` sets `self.rank_tp_ratio = None` on uniform budgets (`server_args.py:9157-9162`)
with one INFO line. `_handle_uneven_mlp_ratio` then raises
`"{flag} … requires an active uneven-TP base plan"` (`10023-10028`) for any
`--rank-mlp-ratio`/`--rank-moe-ratio`/`--rank-vocab-ratio` the same command line carried —
so a homogeneous-rig launch that worked with an explicit vector fails with a message that
names a flag the operator DID pass. Lower severity than B1/B2 (it is a loud error, not a
silent one), but the error text is misleading.
Task title: **"Name the `auto`→even-split collapse in the family-vector refusal message"**

The remaining NARROWER items (S1-03, S1-07, S1-10, S1-12, S1-16, S1-25, S1-30, S1-31) are
DOC-CANDIDATES: each restriction is intended, argued at its site, and correct — the
catalog line is simply short of it.

---

#### NOT-FOUND

* **S1-34 — "above TP>kv_heads, replicated is the DEGRADED layout".** Searched for a
  predicate that reads the kv-head count when deciding/validating `draft_kv_layout`:
  `grep draft_kv_layout` (server_args, boot_matrix, mem_cache), `grep -i degraded`
  (whole `python/sglang/srt/`), `grep attn_kv_replicated` (all call sites),
  `grep num_kv_heads` filtered on `draft`/`replicat`/`tp_size`, plus
  `draft_pool_is_replicated` in `layers/dcp/owner.py`. Nothing compares `tp_size` with
  `num_kv_heads` for the draft pool. The threshold is CLI help prose only
  (`server_args.py:1513-1528`), backed by measurements, and the operator gets no
  boot-time signal either way. A real gate would be cheap (the count is readable at
  model-config time) and would turn a documented footgun into a warning.
* **S1-39 — who sets `dcp_size` for uneven DCP.** Found, but not where §1 implies:
  it is auto-set only inside the placement branch (see B1). Listed as NOT-FOUND against
  the catalog's phrasing rather than against the code.

---

#### Catalog corrections

Copy-paste replacements for `docs/dev/FEATURE_CATALOG.md` §1.

**1. Uneven TP / placement coupling + auto (lines 10-13)**

OLD:
```
- **Uneven TP** `--rank-tp-ratio` + `--rank-gpu-id`: per-card weight shards.
  `auto` = byte-proportional from NVML totals minus auto reserve; with
  `--rank-perf-tune both|dec|enc|maxkv` the planner solves the vector.
```
NEW:
```
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
  (`server_args.py:723`) the planner solves the vector; the two `phase-*`
  arms are the #354/#357 phase-optimal recipe. `auto-performance` is refused
  under `--pp-size > 1` (`server_args.py:9394`); plain `auto` has the
  per-stage path.
```

**2. Unit system / Nemotron census (lines 13-16)**

OLD:
```
  Unit system: `tp_units`/`tp_family` per layer class (16-element MLP family;
  coupled-dim rule: gate_up output and down_proj input partition the SAME
  intermediate dim and must coarsen identically); per-layer family table for
  `block_configs` models (Nemotron-Puzzle class).
```
NEW:
```
  Unit system: `tp_units`/`tp_family` per layer class (16-element MLP family,
  boot-refused per rank at `distributed/utils.py:975`; coupled-dim rule:
  gate_up output and down_proj input partition the SAME intermediate dim and
  must coarsen identically — one SYMMETRIC block, `lcm(64,128)=128`,
  `layers/linear.py:181`). The `block_configs` (Nemotron-NAS/Puzzle) support is
  a PLANNER-side weight-byte census for the auto-performance cost model
  (`LayerFamilyCensus`, `uneven_perf.py:2894`), not a runtime shard-family
  table.
```

**3. `--rank-kv-ratio` modes and `--rank-auto-reserve-mib` reach (lines 17-22)**

OLD:
```
  `--rank-kv-ratio` (`coupled|speed|vector` — decouples KV split from weight
  split), `--rank-auto-reserve-mib`, `--rank-gpu-memory-mib` (absolute
  per-rank MiB budget with a line-item ledger incl. lane pools).
```
NEW:
```
  `--rank-kv-ratio` (`coupled|capacity` (alias `auto`)`|speed|vector`,
  `server_args.py:484` — decouples KV split from weight split; `capacity` is
  the measured one-boot-convergence mode that `--rank-perf-tune maxkv`
  selects, `speed` degrades to it without bandwidth scores),
  `--rank-auto-reserve-mib` (also usable WITHOUT any uneven flag: a pinned
  reserve then sizes the plain path as
  `mem_fraction_static = (NVML total − reserve)/total` exactly, #332,
  `server_args.py:9370`), `--rank-gpu-memory-mib` (absolute per-rank MiB
  budget with a line-item ledger incl. lane pools; the per-rank LIST form
  additionally needs no weight vector under the weightless-KV lane or PP>1,
  `server_args.py:9694`).
```

**4. `--rank-moe-ratio link` refusals (lines 32-33)**

OLD:
```
  Refused by name when offload is off, when the link provenance is `absent`, or
  under `ep_size>1`. Resolved ONCE in the launcher — a symbolic value that
```
NEW:
```
  Refused by name, all five in the launcher: nothing to move (the mass test
  `cold_total <= 1e-9`, i.e. offload off, `expert_compute_placement.py:527`),
  link provenance `absent` (:951), `ep_size>1` (:916), no resolved uneven-TP
  base plan (:906), and no rank→physical-card vector (:941). Resolved ONCE in
  the launcher — a symbolic value that
```

**5. Uneven DCP reach (lines 58-62) — the MECHANISM REACH correction**

OLD:
```
- **Uneven DCP** (`dcp_size` + token vector): token/KV sharding across ranks,
  weighted owner rule, SWA-hybrid support, TP>kv_heads via replication+token
  shard. **Draft-KV-DCP**: draft KV token-sharded (−67 % draft KV; above
  TP>kv_heads, replicated is the DEGRADED layout). LSE log base follows the
  attention backend (FlashMLA = natural log).
```
NEW:
```
- **Uneven DCP** (`dcp_size` + token vector): token/KV sharding across ranks,
  weighted owner rule, SWA-hybrid support. The replication+token-shard axis is
  NOT kv-head-count-gated: the predicate is
  `dcp_size > 1 and get_tp_partition_ratios() is not None`
  (`distributed/utils.py:354`) — it never reads the kv-head count and does not
  require `dcp_size == tp_size`, so it is live for EVERY kv count. Three ways
  in: a `--rank-tp-ratio` plan, the weightless-KV lane
  (`... or self.weightless_kv`, `flashinfer_backend.py:689`), and a #274
  lane's context-local overlay (`utils.py:168`). The separate REPLICATED-KV
  attention geometry is `tp_plan_active(tp) and total_num_kv_heads < tp`
  (`utils.py:1048`; `kv == tp` deliberately excluded, reverted on
  measurement), and it refuses without DCP spanning the group
  (`flashinfer_backend.py:735`). MLA models have NO uneven-TP DCP combine and
  are refused by name (`layers/dcp/comm.py:81`). `dcp_size` is auto-set to
  `tp_size` only on the `--rank-gpu-id` path (`server_args.py:9845`).
  **Draft-KV-DCP**: draft KV token-sharded (−67 % draft KV), admitted by
  `_reject_unsupported_draft_kv_dcp` (`server_args.py:7448`), which today
  requires the `SGLANG_UNEVEN_DCP`+`_WEIGHTED` env pair and does NOT accept
  the `--rank-kv-ratio` route the sibling gate at :7628 accepts (open defect).
  "Replicated is the DEGRADED layout above TP>kv_heads" is a measured
  recommendation in the CLI help (`server_args.py:1513`), not a code
  predicate — nothing compares TP to the kv-head count for the draft pool.
  LSE log base follows the attention backend
  (`NATURAL_LOG_LSE_ATTENTION_BACKENDS = {"flashmla"}`, `dcp/comm.py:44`).
```

**6. TP5+ emulation (line 72)**

OLD:
```
- **TP5+ emulation** via NCCL multi-rank co-location (several ranks per card).
```
NEW:
```
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
```

**7. TPxPPxTP (lines 63-71) — one-line accuracy note**

OLD:
```
  stage-local mamba slots, `auto` under PP with an agreement gate,
```
NEW:
```
  stage-local mamba slots (per-stage GPU groups must be pairwise DISJOINT,
  `server_args.py:8669`), `auto` under PP with an agreement gate (all stages
  must derive the same vector, `server_args.py:9116`),
```

**8. Sibling-flag caveats (after line 19) — append one sentence**

OLD:
```
- Sibling flags: `--rank-mlp-ratio`, `--rank-vocab-ratio`, `--rank-moe-ratio`
```
NEW:
```
- Sibling flags (each needs a resolved non-uniform base plan,
  `server_args.py:10023`; `--rank-vocab-ratio` is additionally incompatible
  with `--enable-dp-lm-head`, :9911; `--rank-moe-resident-fraction` refuses a
  per-rank vector when `moe_tp_size != tp_size` and refuses env/flag
  disagreement, `moe/resident_fraction.py:132`/:107):
  `--rank-mlp-ratio`, `--rank-vocab-ratio`, `--rank-moe-ratio`
```


### §2 + §17 — Planner / solver, combination matrix

Scope: `FEATURE_CATALOG.md` §2 (Planner / solver) and §17 (META: combination
matrix + eviction doctrine). Static read only; no boot, no GPU, no edits
outside this file. Every row cites the predicate that actually decides, not the
docstring next to it.

Tip read: worktree `/spinning/wt-500-reach`, branch as checked out 2026-08-03.

Class counts: **WIDER 4 · NARROWER 12 (4 bug-candidates, 8 doc-candidates) ·
EXACT 22 · NOT-FOUND 4** (42 items).

---

#### Table

| ID | Catalog claim (short) | Class | Gate predicate (verbatim) | file:line | Note |
|----|----------------------|-------|---------------------------|-----------|------|
| S2-01 | "`--rank-perf-tune both\|dec\|enc\|maxkv` the planner solves the vector" (§1 l.12, the only enumeration in the catalog) | NARROWER (doc) | `_RANK_PERF_TUNE_CHOICES: Tuple[str, ...] = (`<br>`    "both", "dec", "enc", "maxkv", "phase-prefill", "phase-decode",`<br>`)` | `python/sglang/srt/server_args.py:723` | Six accepted values, not four. `phase-prefill`/`phase-decode` are named in §2 prose but never in the enumeration a reader copies. No undocumented mode: the argparse `choices=list(_RANK_PERF_TUNE_CHOICES)` (`server_args.py:2327`) and the validator (`:8916`) both read the same tuple. |
| S2-02 | the perf flags are auto-performance-only | EXACT | `if loose != 0.0 or self.rank_perf_tune != "both":`<br>`    raise ValueError("--rank-perf-loose-ctx-percent / --rank-perf-tune only apply with --rank-tp-ratio auto-performance.")` | `server_args.py:8905` | Fail-fast, parse time. |
| S2-03 | "`--rank-perf-tune dec` … also selects `--rank-kv-ratio speed`" | EXACT | `if self.rank_kv_ratio == "coupled":`<br>`    if self.rank_perf_tune == "dec":`<br>`        self.rank_kv_ratio = "speed"` | `server_args.py:8933` | Only from the `coupled` default; an explicit flag wins. |
| S2-04 | "`maxkv` … also selects `--rank-kv-ratio capacity`" | EXACT | `elif self.rank_perf_tune == "maxkv":`<br>`    self.rank_kv_ratio = "capacity"` | `server_args.py:8944` | |
| S2-05 | "`phase-decode` … leaves `--rank-kv-ratio` alone" | EXACT | (absence) the `if/elif` chain at `server_args.py:8933-8950` names only `dec` and `maxkv` | `server_args.py:8933` | Confirmed by omission; the comment at `:8928` states the intent. |
| S2-06 | §2 describes the solve unconditionally | NARROWER (doc) | `if not isinstance(server_args.rank_tp_ratio, list):`<br>`    lines.append("base VRAM-auto split collapsed to the even split (uniform budgets); the MLP family vector requires an uneven base plan -- keeping the classic even split unchanged.")`<br>`    emit(); return` | `python/sglang/srt/uneven_perf.py:5605` | The ENTIRE optimizer (all six targets, the #485 joint cut, the energy objective, the #435 seed) is skipped when the derived base plan is uniform — i.e. on a homogeneous rig. §2 never says the planner has no lever on equal cards. |
| S2-07 | "prefill 10,1,1 (+ decoupled KV 2,11,10), decode ~3,2,2" measured optima | NOT-FOUND | — | — | Rig data, not a predicate; grepped `10,1,1`/`3,2,2` under `python/sglang/srt/` — only in log/doc strings. Catalog already labels it RIG EXAMPLE. No code claim to check. |
| S2-08 | "Under `--rank-perf-tune phase-*` the solve now also OWNS the coupled KV token vector (#435)" | EXACT | `return (`<br>`    tune in _PHASE_TUNES`<br>`    and not model.solo_active`<br>`    and getattr(server_args, "rank_kv_ratio", "coupled") == "coupled"`<br>`)` | `uneven_perf.py:5318` (`_phase_solve_owns_kv_ratio`) | Two conditions §2 does not mention: draft-solo placement is excluded, and any explicit `--rank-kv-ratio` (including the literal string `coupled`… which is indistinguishable from the default) disables the ownership. |
| S2-09 | "An explicit `--rank-kv-ratio` still wins" | WIDER | `env_vec = envs.SGLANG_UNEVEN_TOKEN_VECTOR.get()`<br>`if env_vec: … return [v // g for v in parsed]` — **above** `kv_flag = getattr(server_args, "rank_kv_ratio", None)` | `python/sglang/srt/distributed/utils.py:490` / `:508` / seed `:526` | Real precedence is 4-deep: `SGLANG_UNEVEN_TOKEN_VECTOR` > explicit `--rank-kv-ratio` vector > `rank_kv_capacity_seed` (#435) > budget estimate. The env var outranks the flag the catalog calls the winner. |
| S2-10 | "A FIXED KV token vector keeps the relative base-plan pricing; a MATCHED one … is checked ABSOLUTELY" | EXACT | `kv_ratio = getattr(server_args, "rank_kv_ratio", "coupled")`<br>`if isinstance(kv_ratio, list):  _fund_token_vec = [max(int(v), 1) for v in kv_ratio]`<br>`elif server_args.uneven_kv_derived_mode():  _fund_token_vec = None; _fund_matched = True`<br>`elif _phase_solve_owns_kv_ratio(server_args, model, tune):  _fund_token_vec = None; _fund_matched = True`<br>`else:  _fund_token_vec = partition_units(_PREDICT_TOKEN_UNITS, base_plan)` | `uneven_perf.py:5967-5981` | This is THE branch that distinguishes FIXED from MATCHED. Three MATCHED sources: `capacity`, `speed`, and the phase arms. |
| S2-11 | the fundability gate is described as always on | NARROWER (doc) | `demand_by_gpu = getattr(server_args, "_derived_rank_auto_reserve_per_gpu", None)`<br>`if demand_by_gpu:` … `if _fund_demand is not None:` | `uneven_perf.py:5919` / `:5935` | Without a derived per-GPU auto reserve (no `--rank-gpu-id` / NVML profile miss) `_fund_demand` stays `None`, `_residual` returns `None`, `_unfundable_reason` returns `None` and **every candidate is fundable by default**. The #437 gate is silently absent, not failed. |
| S2-12 | MATCHED ⇒ "every rank is checked ABSOLUTELY … on ALL cards" | EXACT | `if _fund_matched:`<br>`    unfunded = res[r] < _fund_demand[r]`<br>`else:`<br>`    unfunded = res[r] < _fund_demand[r] <= _fund_base[r]` | `uneven_perf.py:6027-6034` | The `<= _fund_base[r]` term is exactly the relative/absolute switch. |
| S2-13 | "#330's 400 MiB corridor is … REPORTED (`CORRIDOR-TIGHT`), never binding" | EXACT | `admissible = (`<br>`    floor_ok and (knee_ok or not knee_binding) and unfundable is None`<br>`)` — no corridor term; `_corridor_note` only concatenates into `verdict` text (`verdict = f"{verdict}; {note}"`) | `uneven_perf.py:6517` (admissibility) / `:6609` (note) / `:6051` (`_corridor_note`) | Proof of non-binding: the corridor never appears in the admissibility conjunction, and `_corridor_note` returns a string, never a rejection. |
| S2-14 | "`SGLANG_PLANNER_CORRIDOR_MIB` overrides it; the number itself lives once in `registry/ledger.py`" | EXACT | `override = envs.SGLANG_PLANNER_CORRIDOR_MIB.get()`<br>`if override is not None:  return max(int(override), 0)`<br>`return int(DEFAULT_CORRIDOR_BYTES // MIB)` + `DEFAULT_CORRIDOR_BYTES = 400 * MIB` | `uneven_perf.py:5279` / `python/sglang/srt/registry/ledger.py:69` | Single definition confirmed: `grep -rn "DEFAULT_CORRIDOR_BYTES\s*=" python/` → one hit. |
| S2-15 | "the cost model's compute term is `sum_family max_rank`, not `max_rank sum_family`" | EXACT | `"""The compute time one lockstep prefill round really costs (s).`<br>`   ``sum_family max_rank``, not ``max_rank sum_family`` (#475).` + call site `t_comp = PerfCostModel.prefill_lockstep_compute_time(self, mlp_vector, gemm_tflops, family_tflops, attn_vector)` | `uneven_perf.py:4885` / `:4829` | The per-barrier max is the only compute term `_prefill_sharded_time` uses. |
| S2-16 | "the Jensen gap … is `PerfCostModel.prefill_barrier_skew`" | EXACT | `return PerfCostModel.prefill_lockstep_compute_time(...) - max(PerfCostModel.per_rank_prefill_compute_times(...), default=0.0)` | `uneven_perf.py:4936` | |
| S2-17 | "every candidate keeps >= 1 unit on the kv-head and GDN k-head grids (#62/#116)" | EXACT (hard bound) | `if units < n:`<br>`    raise ValueError(f"Cannot give each of {n} ranks at least one of {units} units.")` … `sizes = [max(s, 1) for s in sizes]` … `assert sum(sizes) == units and all(s >= 1 for s in sizes)` | `python/sglang/srt/distributed/utils.py:568`, `:576`, `:589` | Hard: a raise plus an assert. It forecloses a 0-unit rank **on the head grid**. It does NOT foreclose the replication axis — replication bypasses the head grid entirely (see S2-18); the solver simply never takes that route. |
| S2-18 | **"the attention family is grid-PINNED (4 kv heads, 3 ranks -> only `[2,1,1]` is representable)"** | **NARROWER — BUG-CANDIDATE** | `if shard == "attn":`<br>`    grid = self.attn_units`<br>`    if grid < n:`<br>`        grid = max(self.q_heads, n)`<br>`    units = partition_units(grid, attn_plan)`<br>`    return [u / grid for u in units]` | `uneven_perf.py:4133-4148` | **The #492 class, repeated.** The enumerator never reads `uneven_dcp_kv_replicated`, `cp_token_prefix`, `_CP_TOKEN_RATIOS` or `rank_kv_ratio` — grepped all four across `uneven_perf.py`: zero hits. It escapes the kv-head grid **only** when `attn_units < tp`. On the reference rig `attn_units = 4 >= tp = 3`, so the grid stays 4 and the ladder collapses. But the runtime replicates kv heads on TWO paths, and the planner's own placement module says so: `replicated = kv_heads < tp or dcp_replicated` with `dcp_replicated = _uneven_dcp_active(flags, base_plan)` — "every rank keeps the FULL kv-head set, EVEN when kv_heads >= tp" (`planner/placement.py:812-813`, mirroring `distributed/utils.uneven_dcp_kv_replicated:346`). The same cost model already prices the KV **cache** as replicated (`#: Full-kv-head KV bytes per token (weighted DCP replicates heads)`, `uneven_perf.py:3852`) while pricing the attention **weights** as head-sharded. Under every uneven-TP + DCP boot — the only configuration `--rank-perf-tune phase-*` runs in at all (S2-06) — Q/O shard on the q-head grid (32 units on this checkpoint) and K/V are replicated, so the representable attention-weight ladder is ~32-wide, not 3-wide; and the attention family's second axis (token ownership, `--rank-kv-ratio`) is continuous and is not enumerated at all. |
| S2-19 | "the ladder … deduplication to the materialized partitions" | EXACT | `key = _attn_partition_key(model, c)`<br>`if key in seen:  continue` with `_attn_partition_key = (tuple(model._shard_fractions("attn", …)), tuple(model._shard_fractions("gdn", …)))` | `uneven_perf.py:5157` / `:5113` | Correct as written — and it is the mechanism that *executes* the pinning of S2-18: every wish vector that rounds to `[2,1,1]` on the 4-unit grid is discarded as a duplicate. |
| S2-20 | "The solve REPORTS the pair and does not install it" | EXACT | `elif admissible and gain > best_gain + 1e-9:`<br>`    if acand is None:`<br>`        best_gain = gain`<br>`        chosen = cand` | `uneven_perf.py:6550-6560` | A joint pair can never become `chosen`. The launch line is printed instead (`JOINT PREFILL LAYOUT (#485, DESK/PREDICTED -- SOLVED, NOT INSTALLED)`, `:6837`). |
| S2-21 | "since 'mlp' is the sole named family plan" | NARROWER (doc) | `for family, field in (("mlp", "rank_mlp_ratio"), ("moe", "rank_moe_ratio"), ("vocab", "rank_vocab_ratio")):` | `python/sglang/srt/managers/scheduler.py:5860` | Three named family plans exist, not one. The *conclusion* holds (none of them is an attention family), but the stated reason is wrong and would mislead anyone estimating the cost of adding an `attn` family. |
| S2-22 | "Slice 1 delivers the prefill column's joint cut (`--rank-perf-tune phase-*` solves PAIRS)" | NARROWER (doc) | `joint = (`<br>`    tune in _PHASE_TUNES`<br>`    and not decode_objective`<br>`    and not _objective_is_energy(server_args)`<br>`)` | `uneven_perf.py:6408` | `--objective energy` **silently disables** the joint per-family cut: no pairs are built, no `JOINT PREFILL LAYOUT` line, no lane bracket. Not stated anywhere in §2 or in the `--objective` help. |
| S2-23 | "the plan log states `LANE-INVARIANT` or `LANE-SENSITIVE`" | NARROWER (doc) | `if not bw_scores or len(bw_scores) != model.tp_size:`<br>`    return None` (then `if joint and lane_bracket is not None and results:`) | `uneven_perf.py:5218` / `:6693` | With no measured per-rank bandwidth vector in the profile the bracket is skipped entirely and NEITHER label is printed — the log is silent rather than saying it could not bracket. |
| S2-24 | "phase-prefill … treats the decode-knee guard as ADVISORY … fundability and the context floor still reject" | EXACT | `knee_binding = tune not in _PHASE_TUNES` + `admissible = (floor_ok and (knee_ok or not knee_binding) and unfundable is None)` | `uneven_perf.py:6275` / `:6517` | Both halves confirmed in one conjunction. |
| S2-25 | "`--objective energy` end to end with refusal over silent substitution" | EXACT | `if energy_model is None:`<br>`    return ("--objective energy needs per-card power anchors and none were supplied: … Planning by throughput under an energy objective would be a silent substitution, so the solve is refused instead.")` | `python/sglang/srt/planner/key_solver.py:1479` (solver) and `raise ValueError("--objective energy: the boot planner has no per-rank card identities, so it cannot look up power anchors. …")` `uneven_perf.py:5557` (boot) | Both ends refuse. |
| S2-26 | "end to end" (no stated exclusions) | NARROWER (doc) | `if objective_is_energy and (goal_b is not None or constraints):`<br>`    raise ValueError("--objective energy applies to the single-goal solve; it cannot be combined with a second goal (Pareto front) or with constraints …")` and `ENERGY_PRICEABLE_GOALS: Tuple[str, ...] = ("dec", "enc")` | `key_solver.py:2589` / `:1421` | Two undocumented exclusions: energy × Pareto-front and energy × constraints are hard errors, and only the `dec`/`enc` goals are priceable in joules at all. Combined with S2-22, `--objective energy` is materially narrower than "end to end". |
| S2-27 | energy needs a `j_per_work` rate | EXACT | `if j_per_work is None:`<br>`    raise ValueError("the ENERGY objective needs a j_per_work rate; … None is a wiring bug, not an absence")` | `python/sglang/srt/planner/objective.py:161` | Absence (`Rate.absent`) propagates to `unscorable`; `None` is a crash. Deliberate. |
| S2-28 | "a named verdict — `NO_SWITCH` / `SWITCH_KV_ONLY` / `SWITCH_FULL` / `UNPRICEABLE`" | EXACT | `class Verdict(str, enum.Enum):`<br>`    """The four answers §20.1 authorises. There is no "maybe" tier."""`<br>`    NO_SWITCH = "NO_SWITCH"` / `SWITCH_KV_ONLY` / `SWITCH_FULL` / `UNPRICEABLE` | `python/sglang/srt/planner/regime_switch.py:99-112` | Exactly four members, no fifth. |
| S2-29 | "within a stated tolerance (default 2.0 %)" | EXACT | `DEFAULT_PAIR_TOLERANCE_PCT = 2.0` used as `tolerance_pct: float = DEFAULT_PAIR_TOLERANCE_PCT` | `regime_switch.py:856` / `:918` | |
| S2-30 | "below the 4.2 % measured A-vs-A floor so it can only break ties" | EXACT | `keep = [c for c in priced if (best - float(c.score.value)) / best * 100.0 <= tolerance_pct + 1e-9]` | `regime_switch.py:952-956` | The predicate is a pure admission filter on the phase's own optimum; "below the floor" is the rationale for the constant, not a second check. |
| S2-31 | "an absent cell yields UNPRICEABLE naming the missing arm, never a guess" | EXACT | `missing = [f"({name}, {phase}): {rates[(name, phase)].source}" for (name, phase) in wanted if rates[(name, phase)].provenance is Provenance.ABSENT]`<br>`if missing:  return AutocheckResult(verdict=Verdict.UNPRICEABLE, …)` | `regime_switch.py:1144-1151` | Missing cells are returned in `missing_cells`; `AutocheckResult` docs pin "Non-empty exactly under UNPRICEABLE" (`:1056`). |
| S2-32 | **"DECISION LAYER ONLY — nothing in this build executes a layout switch"** | **WIDER (catalog under-claims)** | `if mode == MODE_ACT:`<br>`    from sglang.srt.managers.regime_act import build_regime_actuator`<br>`    commit_fn = build_regime_actuator(scheduler, table_plan).apply` | `python/sglang/srt/managers/regime_runtime.py:721-727`, actuator body `python/sglang/srt/managers/regime_act.py:182-189` (`ok, msg = self._vram_apply(want_vram)` … `ok, msg = self._reshard_arm(want_kv, ARM_SOURCE)`), wired from `python/sglang/srt/managers/scheduler.py:3401` | `--regime-controller act` DOES execute the reachable half of a layout switch on a live server: a #330 VRAM-budget GROW and a #297 KV reshard. Refusals are named, not silent: shrink refused (`regime_act.py:151`), missing actuator refused (`:161`, `:169`), and `act` itself is refused at parse time until `--regime-gate-evidence` names four measured gate items (`server_args.py:4993`). The catalog sentence is true only of the #363 *slice 1* pair-solver, and reads as build-wide. |
| S2-33 | "`--regime-phase-table`" listed beside `PlanResult.regime` | EXACT (clarify) | `r = p.add_argument_group("#363 regime autocheck (decision only, nothing switches)")` … `r.add_argument("--regime-phase-table", …)` | `python/sglang/srt/planner/cli.py:133-144` | It is a **planner CLI** flag (`python -m sglang.srt.planner.cli`), not a `ServerArgs` flag — `grep -n "regime" server_args.py` returns only the `regime_controller`/`regime_trace`/`regime_gate_evidence` trio. Four sibling flags exist that §2 does not name: `--regime-prefill-layout`, `--regime-decode-layout`, `--regime-workload`, `--regime-not-pre-captured`. |
| S2-34 | "`planner/rejected.py` = machine-readable register of discarded approaches — check it before re-proposing anything" | NARROWER (doc) | `return [e for e in REGISTER if e.tags and set(e.tags).issubset(have)]` | `python/sglang/srt/planner/rejected.py:731` (`check_combination`) | 24 registered entries, all tagged, all readable via `register_json` (`webui.py:4660`, `/api/rejected`). But only two call sites feed tags: `wizard.py:820`/`:1306` and `rig_coupling.py:904`. The tag universe those produce is `{solo-tp, uneven-tp, uneven-dcp, pd, rank-reuse, co-residence, satellite, pipeline-parallel, spill}` ∪ `{gguf, moe, moe-offload, <quant kind>, <arch: sm75/sm86/…>}` (`webui.py:4096-4137`) ∪ `{crossrig-tp-push}` (`rig_coupling.py:951`). Auto-consulted in practice: `gguf_on_sm75`, `gguf_moe_expert_offload`, `spill_with_pd`, `spill_with_pp_dp`, `crossrig_tp_push` — **5 of 24**. The other 19 (incl. `tree_spec_uneven_dcp`, `int8_mamba_sizing`, `vocab_ratio_7_3_3`, `mlp_concentration_611`, `reserve_2200_3080`, `spec_k4_mixed`, `pp_with_spec`, `moe_link_calibrated_coefficients`, …) carry tags no caller ever emits (`tree-spec`, `speculation`, `vocab-ratio`, `int8-mamba`, `reserve-2200`, `cuda-ipc-weights`, `collective-overlap`, `dcp-overlap`, `path-bundling`, `ucc`, `gdn-autotune`, `c4-indexer-head-fold`, `verify-intermediate-rows`, `moe-link-calibrated`, `triton-fp8`, `barlink-ring-bidir`, `spec-k4`/`mixed-cards`, `mlp-concentration`, `dual-group-lane`), so they are documentation for a human reader, never a predicate. |
| S2-35 | "Plain `--rank-tp-ratio auto` is the documented CAPACITY-FIRST default … names the per-task optimizer … in the CLI help and in one boot log line" | EXACT | `_CAPACITY_FIRST_DEFAULT_NOTICE = ("--rank-tp-ratio auto is the CAPACITY-FIRST default: … The per-task optimizer is a different flag: --rank-tp-ratio auto-performance … per --rank-perf-tune target (" + "\|".join(_RANK_PERF_TUNE_CHOICES) + ")…")` | `server_args.py:740-753` | The notice enumerates all six targets from the same tuple, so the help string is right where §1 l.12 is stale. |
| S2-36 | "`--rank-perf-tune dec` no longer returns the base split … it SOLVES the bs=1 decode round time" | EXACT | `decode_objective = tune == "dec"` … `for cand in _mlp_candidates(model, list(dec_scores), base_plan): …` with `dec_scores = model.effective_decode_bw(rank_scores_bw, rank_scores_gemv)` | `uneven_perf.py:6379-6393` | Union of the compute ladder and the bandwidth ladder; no early return. |
| S2-37 | "Every objective therefore solves from per-(rank, family) profile scores" | NARROWER (doc) | `family_tflops` is `Optional`; `"``None`` (every single-scheme checkpoint) keeps the scalar arithmetic byte-identical"` — per-rank time falls back to `sum_fam(p_fam) / r` | `uneven_perf.py:4811-4822` | On a single-scheme checkpoint the solve is per-(rank) with a summed-params proxy, not per-(rank, family). The per-family lane split is a MIXED_PRECISION / #324 feature. |
| S2-38 | §17 "a class added without a ladder rank fails at import" | EXACT | `_MISSING_DESCRIPTORS = tuple(k for k in OFFLOAD_CLASSES if k not in ASSET_CLASSES)`<br>`if _MISSING_DESCRIPTORS:`<br>`    raise RuntimeError(f"offload classes without an asset-class descriptor: {_MISSING_DESCRIPTORS}. Every class needs a ladder rank and a payload class, or it cannot take part in the DESIGN_407 §8 eviction order and would be silently unplannable.")` | `python/sglang/srt/model_executor/short_term_offload_register.py:465-472` | Module-scope, so it is a genuine import-time guard. One-directional: a descriptor for a class NOT in `OFFLOAD_CLASSES` is not caught. |
| S2-39 | §17 "that ladder is now EXECUTABLE rather than prose … so a new consumer calls it instead of restating it" | NARROWER (doc) | `plan = plan_spill(self._reg(), bytes_needed, classes=("kv_shadow", "graph_rungs", "drafter_heads", "lane_workspaces"))` — the ONLY caller, inside the same module | `short_term_offload_register.py:1486` (in `rung1_evict`) | `grep -rn "plan_spill" python/ test/ tests/` → definition, the in-module `rung1_evict` call, and test files. **Zero production consumers outside the module.** `describe_class` does have one real consumer (`layers/moe/breakable_offload.py:216`), so the register is not dead — but "a new consumer calls it" describes an intent, not a wiring that exists. |
| S2-40 | §17 "the breakable route … gated OFF and never booted" | EXACT | `raw = (os.environ.get(ENV_GRAPH_MODE) or "").strip().lower()`<br>`if not raw:  return MODE_EAGER` | `python/sglang/srt/layers/moe/offload_capture_gate.py:237-239` | Default is `eager`; `breakable` requires `SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable`, and an unknown value raises `BreakableModeRefused` (`:241`). |
| S2-41 | §17 "The in-graph fetch remains register-rejected" | NOT-FOUND (as a register entry) | — no `RejectedEntry` matches; `grep -rn "in-graph\|in_graph\|capturable" python/sglang/srt/planner/rejected.py` → zero hits | refusal actually lives at `python/sglang/srt/layers/moe/offload_capture_gate.py:284` (`refuse_capturable_offload_decode(layer_id)`) and `:257` (`"``capturable`` -- the in-graph fetch. REFUTED (#452)"`) | The refusal is real and enforced at runtime, but it is NOT in `planner/rejected.py` — which is precisely the file CLAUDE.md tells every agent to check before declaring something rejected. An agent grepping the register for "in-graph fetch" finds nothing and would conclude the cell is open. |
| S2-42 | §17 "anything that decides at runtime whether a path is worth its cost is an instance of DESIGN_363 §20.1's worth-it autocheck rather than a new flag" | NOT-FOUND (unenforced doctrine) | — no import guard, no registry, no test enforcing it; `regime_switch.autocheck` has exactly two call sites (`planner/feasibility.py:503`, `planner/solver_api.py:590`), neither in the serving path | `regime_switch.py:1044ff` | Prose policy. Note the mild self-contradiction: the runtime worth-it decision that DOES exist is reached through a flag (`--regime-controller`, `server_args.py:4979`). |
| S2-43 | §17 register semantics: an entry fires only on a full tag match | EXACT | `have = set(tags)`<br>`return [e for e in REGISTER if e.tags and set(e.tags).issubset(have)]` | `rejected.py:730-731` | Untagged entries never match automatically — by design (`:722`). |
| S2-44 | §2 "`coresident_budgets()`" named as a solver primitive | EXACT | `def coresident_budgets(specs, estimates, gpu_total_mib, *, reserve_mib=None, process_post_mib=FIXED_PROCESS_POST_MIB, shared_process=False) -> Optional[Dict[str, List[int]]]:` | `python/sglang/srt/planner/key_solver.py:3765` | Thin wrapper over `coresident_budget_plan`; returns `None` when no plan fits. |

---

#### WIDER finds — what they unlock

**S2-32 — `--regime-controller act` really does switch a live layout.**
The catalog's "DECISION LAYER ONLY — nothing in this build executes a layout
switch" is false at this tip. `regime_runtime.py:721-727` builds a
`RegimeActuator` in `act` mode and `regime_act.apply` issues a #330 VRAM
budget GROW and a #297 KV reshard on the live scheduler. Practically: a boot
with `--regime-controller act --regime-gate-evidence <file> --kv-reshard-vectors
… --enable-vram-dial` already reshapes KV ownership between regimes without a
restart — the "switching arms needs a RESTART" line the phase-prefill arm
prints (`uneven_perf.py:6881`) is true only for the WEIGHT half. Anyone
planning "#363 slices 2+" should start from the wired half, not from zero.

**S2-09 — `SGLANG_UNEVEN_TOKEN_VECTOR` outranks the flag.**
The token-vector precedence is 4-deep and the env var wins over an explicit
`--rank-kv-ratio a,b,c`. That is the self-calibration feedback path: a boot can
emit its measured optimal token vector and the next boot converges onto it
without editing the launch line — useful for A-vs-A token-vector sweeps where
the rest of the command must stay byte-identical.

**S2-21 — three named family plans, not one.**
`mlp`, `moe` and `vocab` are all installable per-rank weight vectors
(`scheduler.py:5860`). Adding a fourth (`attn`) is a table entry plus the
layers opting in with `tp_family="attn"` — not the architectural change the
"sole named family plan" wording implies. This is the cheapest route to
actuating the #485 attention half.

**S2-17 read correctly is permissive, not restrictive.**
The `>= 1 unit` rule binds only the head-PARTITION route. Under
`uneven_dcp_kv_replicated` there is no head partition at all, so the rule
imposes nothing on the replication+token-shard layout — the axis is free, the
solver just never enumerates it (S2-18).

---

#### NARROWER — bug candidates

**BUG-1 (S2-18) — the #485 attention enumerator prices a head-sharded
attention block that the boot does not run.**
`uneven_perf.py:4133-4148` grids the attention family on `attn_units`
(kv heads / o_groups) and escapes to the q-head grid only when
`attn_units < tp`. Every configuration `--rank-perf-tune phase-*` can run in is
an uneven-TP boot with a non-uniform base plan (S2-06), which is exactly the
condition `uneven_dcp_kv_replicated` (`distributed/utils.py:346`) uses to
REPLICATE kv heads and shard the token axis — the planner's own
`placement.py:812-813` models this correctly, and the same cost model already
prices the KV cache as replicated (`uneven_perf.py:3852`). User harm: on the
reference rig the whole attention ladder is discarded as duplicates of
`[2,1,1]` (`_attn_partition_key`, `:5157`), so the phase-prefill arm reports a
joint layout solved over a 3-point space when the real space is the 32-unit
q-head grid crossed with a continuous token vector. The reported "+1.0 / +6.9
points over the MLP-only cut" is an optimum over the wrong feasible set, and
the barrier-skew term #475 added is minimized against a family whose true
optimum was never a candidate.
*Task title:* `[#500] phase-prefill: price the attention family on the replicated-KV geometry (q-head grid + token vector), not the kv-head grid`

**BUG-2 (S2-22) — `--objective energy` silently switches off the joint
per-family cut.**
`joint = tune in _PHASE_TUNES and not decode_objective and not
_objective_is_energy(server_args)` (`uneven_perf.py:6408`). An operator who
launches `--rank-perf-tune phase-prefill --objective energy` gets an
MLP-only solve with no `JOINT per-family cut (#485)` line, no pacer notes and
no lane bracket — and nothing in the log says the pair space was dropped. The
energy objective is otherwise loud about everything it cannot price (S2-25);
this one absence is silent.
*Task title:* `[#500] --objective energy: either price the #485 joint pairs or name the drop in the plan log`

**BUG-3 (S2-11) — the #437 fundability gate disappears instead of failing when
the derived reserve is unavailable.**
`if demand_by_gpu:` (`uneven_perf.py:5919`) — with no
`_derived_rank_auto_reserve_per_gpu` the gate produces `None` everywhere and
`_unfundable_reason` returns `None` for every candidate, so the exact
configuration class #264 OOM'd on is admitted with no verdict at all. The
plan log prints neither `fundability basis:` line. That is the "silence"
failure mode #421 was written to catch.
*Task title:* `[#500] fundability gate: state UNPRICED when the derived per-GPU reserve is missing instead of admitting every candidate`

**BUG-4 (S2-41) — the in-graph expert fetch is refuted at runtime but absent
from `planner/rejected.py`.**
CLAUDE.md and §17 both send readers to the register before re-proposing an
approach; the register has no entry for the in-graph fetch, while
`offload_capture_gate.py:284` refuses it by name (#452) and DESIGN_462 §"The
in-graph fetch stays refuted" documents it. The register's whole purpose is
that the next person does not repeat the work.
*Task title:* `[#500] planner/rejected.py: register the in-graph MoE expert fetch (#452) with its counter-number`

---

#### NOT-FOUND

- **S2-07** measured phase optima `10,1,1` / `2,11,10` / `~3,2,2`: grepped
  `10,1,1`, `3,2,2`, `2,11,10` across `python/sglang/srt/` — only in log and
  docstring text, never as a constant or a default. Correctly labelled a rig
  example in the catalog; nothing in code to verify or contradict.
- **S2-41** in-graph fetch as a register entry: grepped
  `in-graph|in_graph|capturable|graph` across `planner/rejected.py` — the only
  `graph` hits are `dcp_overlap_fusion`'s prose. Refusal located elsewhere
  (cited above).
- **S2-42** "anything that decides at runtime … is an instance of the worth-it
  autocheck": grepped `autocheck` repo-wide — `planner/feasibility.py:503`,
  `planner/solver_api.py:590`, `planner/cli.py:723` only. No import guard, no
  registry, no test that would fail if a new runtime worth-it decision added a
  flag instead.
- **`SGLANG_PLANNER_CORRIDOR_MIB` second definition**: grepped
  `DEFAULT_CORRIDOR_BYTES\s*=` across `python/` — exactly one assignment
  (`registry/ledger.py:69`). The catalog's "lives once" claim is verified, so
  this is a NOT-FOUND in the good sense: no duplicate constant exists.

---

#### Catalog corrections

Copy-pasteable, in the catalog's own dense style. Each corrected conditional
carries its gate predicate and `file:line`.

**§1, line 12**

```
OLD: `auto` = byte-proportional from NVML totals minus auto reserve; with
  `--rank-perf-tune both|dec|enc|maxkv` the planner solves the vector.
NEW: `auto` = byte-proportional from NVML totals minus auto reserve; with
  `--rank-perf-tune both|dec|enc|maxkv|phase-prefill|phase-decode` the planner
  solves the vector (`_RANK_PERF_TUNE_CHOICES`, `server_args.py:723`; the
  argparse choices and the validator both read that tuple). The whole solve is
  skipped when the derived base plan is uniform — a homogeneous rig has no
  weight lever here (`if not isinstance(server_args.rank_tp_ratio, list): …
  return`, `uneven_perf.py:5605`).
```

**§2, the #437 fundability sentence**

```
OLD: **The fundability gate prices the vector the boot runs (#437).** A FIXED KV
  token vector keeps the relative base-plan pricing; a MATCHED one
  (`--rank-kv-ratio capacity|speed`, and the phase arms since #435) has no
  unused capacity to price, so every rank is checked ABSOLUTELY against the
  derived reserve demand on ALL cards.
NEW: **The fundability gate prices the vector the boot runs (#437).** A FIXED KV
  token vector keeps the relative base-plan pricing (`unfunded = res[r] <
  _fund_demand[r] <= _fund_base[r]`); a MATCHED one (`--rank-kv-ratio
  capacity|speed`, and the phase arms since #435) has no unused capacity to
  price, so every rank is checked ABSOLUTELY (`unfunded = res[r] <
  _fund_demand[r]`) — the branch is `uneven_perf.py:5967-5981`, the two bases
  `:6027-6034`. The gate needs a derived per-GPU auto reserve; without one
  (`if demand_by_gpu:`, `:5919`) it is silently absent and every candidate is
  reported fundable.
```

**§2, the corridor sentence** — no change needed; verified. Optionally tighten
the evidence:

```
OLD: #330's 400 MiB corridor is priced alongside the demand and REPORTED
  (`CORRIDOR-TIGHT`), never binding (`SGLANG_PLANNER_CORRIDOR_MIB` overrides it;
  the number itself lives once in `registry/ledger.py`).
NEW: #330's 400 MiB corridor is priced alongside the demand and REPORTED
  (`CORRIDOR-TIGHT`), never binding — it appears in no admissibility term
  (`admissible = floor_ok and (knee_ok or not knee_binding) and unfundable is
  None`, `uneven_perf.py:6517`; `_corridor_note` returns text, `:6051`).
  `SGLANG_PLANNER_CORRIDOR_MIB` overrides it (`:5279`); the number lives once,
  in `registry/ledger.py:69`.
```

**§2, THE MATRIX DOCTRINE — the grid-PINNED claim (the #492-class defect)**

```
OLD: the attention/GDN family is cut on its own #324 lane and its own grids, the
  GDN state pool and the coupled KV vector follow it, and every candidate keeps
  >= 1 unit on the kv-head and GDN k-head grids (#62/#116). On the reference rig
  the attention family is grid-PINNED (4 kv heads, 3 ranks -> only `[2,1,1]` is
  representable), so in practice the lever is the 16-unit GDN grid
NEW: the attention/GDN family is cut on its own #324 lane and its own grids, the
  GDN state pool and the coupled KV vector follow it. The enumerator escapes the
  kv-head grid ONLY when `attn_units < tp` (`grid = self.attn_units; if grid <
  n: grid = max(self.q_heads, n)`, `uneven_perf.py:4133-4148`), and
  `partition_units` then keeps >= 1 unit per rank (#62/#116; hard —
  `distributed/utils.py:568`, `:589`). **That grid is not the runtime's.** Under
  every uneven-TP boot with DCP the kv heads are REPLICATED and the split rides
  the token axis — `uneven_dcp_kv_replicated` is `dcp_size > 1 and
  get_tp_partition_ratios() is not None` (`distributed/utils.py:346`), no
  kv-head term — which the planner's placement report already models
  (`replicated = kv_heads < tp or dcp_replicated`, `planner/placement.py:813`)
  and the cost model already assumes for the KV cache (`uneven_perf.py:3852`).
  So the reference rig's "4 kv heads, 3 ranks -> only `[2,1,1]`" is an artefact
  of the enumerator, not a property of the rig: the real attention axes are the
  q-head grid plus a continuous token vector. #500/BUG-1 tracks the fix; until
  then the phase-prefill arm's joint gains are an optimum over an incomplete
  candidate set.
```

**§2, the "solved, not installed" sentence**

```
OLD: The solve REPORTS the pair and does not install it — the only runtime
  actuator for an attention vector is `--rank-tp-ratio`, since "mlp" is the sole
  named family plan.
NEW: The solve REPORTS the pair and does not install it (`if acand is None:`
  guards the argmax, `uneven_perf.py:6558`) — the only runtime actuator for an
  attention vector is `--rank-tp-ratio`, because none of the three named family
  plans is an attention family (`("mlp", "rank_mlp_ratio"), ("moe",
  "rank_moe_ratio"), ("vocab", "rank_vocab_ratio")`,
  `managers/scheduler.py:5860`). The joint cut is additionally OFF under
  `--objective energy` (`joint = tune in _PHASE_TUNES and not decode_objective
  and not _objective_is_energy(server_args)`, `:6408`), and the LANE-INVARIANT /
  LANE-SENSITIVE bracket is skipped entirely without a measured per-rank
  bandwidth vector (`if not bw_scores …: return None`, `:5218`).
```

**§2, the `--objective energy` sentence**

```
OLD: `--objective energy` end to end with refusal over silent substitution.
NEW: `--objective energy` with refusal over silent substitution at both ends
  (solver `key_solver.py:1479`, boot `uneven_perf.py:5557`). Scope, not "end to
  end": priceable goals are `("dec", "enc")` only
  (`ENERGY_PRICEABLE_GOALS`, `key_solver.py:1421`), it refuses a second goal or
  constraints (`if objective_is_energy and (goal_b is not None or constraints):
  raise`, `:2589`), and on the boot path it disables the #485 joint pair space
  (`:6408`).
```

**§2, the #363 slice-1 paragraph**

```
OLD: **DECISION LAYER ONLY — nothing in this build executes a layout switch**:
  no pointer flip, no diff spill, no pre-capture (#363 slices 2+, `ROADMAP_456`
  WAVE 4, gated on #286).
NEW: **The slice-1 pair solver is DECISION LAYER ONLY** — `autocheck` has two
  call sites, both planner-side (`planner/feasibility.py:503`,
  `planner/solver_api.py:590`), and `--regime-phase-table` is a planner-CLI flag
  (`planner/cli.py:134`), not a server flag. The RUNTIME half is wired,
  though: `--regime-controller act` builds a `RegimeActuator`
  (`managers/regime_runtime.py:721-727`) that issues a #330 VRAM budget GROW and
  a #297 KV reshard (`managers/regime_act.py:182-189`), refusing a shrink
  (`:151`) and refused at parse time until `--regime-gate-evidence` names four
  measured items (`server_args.py:4993`). What no build executes is the WEIGHT
  half: no pointer flip, no diff spill, no pre-capture (#363 slices 2+,
  `ROADMAP_456` WAVE 4, gated on #286).
```

**§2, the rejected-register sentence**

```
OLD: `planner/rejected.py` = machine-readable register of discarded approaches —
  check it before re-proposing anything.
NEW: `planner/rejected.py` = machine-readable register of 24 discarded
  approaches — check it before re-proposing anything. It is a READING surface
  first: `check_combination` fires only on a full tag match
  (`set(e.tags).issubset(have)`, `rejected.py:731`) and the two tag producers
  (`wizard.py:820/:1306`, `rig_coupling.py:904/:951`) emit a vocabulary that
  reaches 5 of the 24 rows. The other 19 are consulted by humans and by
  `register_json` (`/api/rejected`), not by a gate.
```

**§17, the eviction-ladder sentence**

```
OLD: that ladder is now EXECUTABLE rather than prose
  (`model_executor/short_term_offload_register.LadderRank` / `plan_spill`,
  #286), so a new consumer calls it instead of restating it, and a class added
  without a ladder rank fails at import
NEW: that ladder is now EXECUTABLE rather than prose
  (`model_executor/short_term_offload_register.LadderRank` / `plan_spill`,
  #286), and a class added without a ladder rank fails at import
  (`if _MISSING_DESCRIPTORS: raise RuntimeError(...)`,
  `short_term_offload_register.py:465-472`). `describe_class` already has a
  production consumer (`layers/moe/breakable_offload.py:216`); `plan_spill`
  itself has none outside the module's own `rung1_evict` (`:1486`) — a new
  consumer SHOULD call it instead of restating it, but none does yet
```

**§17, the in-graph fetch sentence**

```
OLD: The in-graph fetch remains register-rejected.
NEW: The in-graph fetch remains refuted (#452) and is enforced at runtime by
  `refuse_capturable_offload_decode` — both spellings
  (`SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1`, `SGLANG_MOE_OFFLOAD_GRAPH_MODE=capturable`)
  reach the same refusal (`layers/moe/offload_capture_gate.py:257`, `:284`). It
  is NOT in `planner/rejected.py` — #500/BUG-4 registers it there.
```


### §3 — Memory tiers / offload / spill

Scope: `FEATURE_CATALOG.md` §3 (Memory tiers / offload / spill), read against
§17. Static read only; no boot, no GPU, no edits outside this file. Every row
cites the predicate that actually decides, not the docstring next to it.

Tip read: worktree `/spinning/wt-500-reach`, 2026-08-03.

Class counts: **WIDER 9 · NARROWER 23 (9 bug-candidates, 14 doc-candidates) ·
EXACT 36 · NOT-FOUND 1 in-table (+3 searched-and-absent, listed below)**
(69 items).

---

#### Table

| ID | Catalog claim (short) | Class | Gate predicate (verbatim) | file:line | Note |
|----|----------------------|-------|---------------------------|-----------|------|
| S3-01 | "the per-decode reserve (`bs x get_alloc_reserve_per_decode()`, held under spec AND plain decode)" | EXACT | `write_footprint = (`<br>`    get_alloc_len_per_decode(server_args) if alloc_len is None else alloc_len`<br>`)`<br>`commit_lag = get_commit_lag_per_decode(server_args)`<br>`return write_footprint + commit_lag` | `python/sglang/srt/mem_cache/common.py:375-379` | Unconditional `W + L`; no spec-only branch. The no-spec case is `1 + 1`, i.e. the reserve is genuinely held under plain decode. |
| S3-02 | **"`SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1` now refuses by name at boot"** | EXACT | `if mode == MODE_CAPTURABLE or bool(opt_in):`<br>`    if not offloading:`<br>`        return MODE_EAGER`<br>`    refuse_capturable_offload_decode(layer_id)` | `python/sglang/srt/layers/moe/offload_capture_gate.py:281-284` | `opt_in` is `envs.SGLANG_MOE_OFFLOAD_CUDA_GRAPH.get()` (`fused_moe_triton/layer.py:671`). **Both spellings really are refused**: the legacy bool and `SGLANG_MOE_OFFLOAD_GRAPH_MODE=capturable` land on the same `refuse_capturable_offload_decode`. |
| S3-03 | the refusal is a boot refusal | NARROWER (doc) | `offloading = float(offload_fraction) < 1.0` … `if not offloading: return MODE_EAGER` | `offload_capture_gate.py:270`, `:282-283` | Refusal fires **only when the layer actually offloads** (`--rank-moe-resident-fraction`/`SGLANG_MOE_RESIDENT_EXPERT_FRACTION < 1.0`). With the offload off, `SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1` is silently ignored, not refused. Intended ("nothing on the default path reaches this function") but §3 does not say it. |
| S3-04 | "`SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE=1` re-opens it for a card window" | WIDER | `if graph_override_enabled():`<br>`    logger.warning(…)`<br>`    return` | `offload_capture_gate.py:206-216`, flag read at `:190` (`_env_flag(ENV_GRAPH_REFUTED_OVERRIDE, False)`) | The escape re-opens **only gate 1**. It does NOT re-open the cold-tier graph seam (`SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE`, S3-16) — §3 states this correctly ("reaching it now takes both overrides"). What §3 does not say: `_env_flag` treats *any* string outside `("0","false","no","off","")` as true, so `SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE=please-dont` opens it. |
| S3-05 | **"Refuses by name at boot unless decode backend is `breakable`"** | **NARROWER — BUG-CANDIDATE** | `backend = resolved_backend("decode")`<br>`if backend is None:`<br>`    # Server args not wired (unit / test context): nothing to validate`<br>`    # against. The runtime scratch guard still applies.`<br>`    return`<br>`if backend != "breakable":` | `offload_capture_gate.py:358-364` | The check is a plain string equality — but it is **preceded by a total bypass**. `resolved_backend` returns `None` on *any* exception importing `runtime_context`, and on a missing/`None` `cuda_graph_config` or `phase_config.backend` (`:408-421`, `try/except Exception: return None`). Under that bypass the decode-backend AND prefill preconditions are both skipped and the breakable route boots under **any** backend — where `eager_on_graph` is a pass-through and the route's `topk_ids.tolist()` executes inside a real stream capture. The named failure mode the gate exists to prevent is reachable through the gate's own `None` arm. Falsifier: construct `ServerArgs` without `cuda_graph_config` (or make `get_server_args()` raise) and set `SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable`. |
| S3-06 | "… and prefill is eager" | NARROWER (doc) | `prefill = resolved_backend("prefill")`<br>`if prefill is not None and prefill != "disabled":` | `offload_capture_gate.py:378-379` | Not "eager" — literally `disabled`. `tc_piecewise` prefill (arguably still eager for the MoE forward) is refused too. Practical note in the other direction: the **default** prefill backend on CUDA is `breakable` (`cuda_graph_config.py:101`, `default_prefill_backend()`), so the route always needs a flag — **except on DeepSeek-V4**, where `("DeepSeek-V4 (heavy capture-pool memory pressure)", lambda: is_deepseek_v4(self.get_model_config().hf_config))` rewrites prefill to `Backend.DISABLED` automatically (`server_args.py:8281-8309`). On the recipe this route targets, the prefill precondition is satisfied without the operator asking. |
| S3-07 | "Both spellings of the refuted path still refuse" (under breakable) | EXACT | `if _env_flag("SGLANG_MOE_OFFLOAD_CUDA_GRAPH", False):`<br>`    raise BreakableModeRefused(` | `offload_capture_gate.py:343-344` | Mutual exclusion is checked on the legacy bool; the new spelling cannot co-occur because `env_graph_mode()` returns exactly one mode. |
| S3-08 | breakable is "OFF by default" | EXACT | `raw = (os.environ.get(ENV_GRAPH_MODE) or "").strip().lower()`<br>`if not raw:`<br>`    return MODE_EAGER` | `offload_capture_gate.py:237-239` | Unknown values refuse by name (`:240-244`), so a typo does not silently downgrade. |
| S3-09 | "`SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable` … DESK-WRITTEN, NEVER EXECUTED" | EXACT | wiring exists: `if self._moe_offload_breakable and get_is_capture_mode():`<br>`    return self._run_moe_core_offload_breakable(…)` | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:2184-2187` | The route IS wired into the forward (not dead code) — `breakable_moe_offload_fetch` → `cache.prepare_breakable` (`expert_offload.py:3070`). "Never executed" is a boot claim, not a wiring claim, and the wiring is complete. |
| S3-10 | **"load-time-aware halves for fp8/GPTQ/AWQ (GGUF-MoE half missing — guarded)"** | **WIDER — headline** | `marker = _OFFLOAD_CONDITIONAL_QUANT_METHOD_NAMES.get(name)`<br>`if marker is not None:`<br>`    if getattr(layer, marker, False):`<br>`        continue  # the half staged this layer -> covered` | `python/sglang/srt/layers/moe/expert_offload.py:2109-2112`, table at `:2069-2071` | **The GGUF-MoE half EXISTS and is admitted.** `GGUFMoEMethod` moved out of the unsupported set into `_OFFLOAD_CONDITIONAL_QUANT_METHOD_NAMES = {"GGUFMoEMethod": "_moe_offload_gguf_staged"}` (#123-GGUF). It passes on any layer the materialization-time stager marked. The error text itself lists "GGUF-MoE (CUDA, ggml types with a MoE kernel)" as supported (`:2147`). The catalog line is stale by a whole quant lane. |
| S3-11 | the quant guard is an enumeration of supported formats | **WIDER** (and the #443/#446 name-list family) | `elif name in _OFFLOAD_UNSUPPORTED_QUANT_METHOD_NAMES:` … `else:`<br>`    continue` | `expert_offload.py:2125`, `:2139-2140` | It is a **denylist of five class names**, not an allowlist: `("GGUFMoEAscendMethod", "MoeWNA16Method", "ModelOptNvFp4FusedMoEMethod", "ModelOptNvFp4OnlineFusedMoEMethod", "CompressedTensorsW4A4Nvfp4MoE")` (`:2056-2064`). *Every* other quant method — unquantized, INT8/W8A8, compressed-tensors non-NVFP4 schemes, any future method — passes by default. Reach: expert offload is available on far more than "fp8/GPTQ/AWQ". Matching is by `type(candidate).__name__` (`:2108`), not `isinstance`, so a subclass or a renamed class escapes the denylist — the same string-name-list fragility as #443/#446. |
| S3-12 | "`SGLANG_MOE_HEAT_MIGRATION=1` … Eager path only: refused by name under `SGLANG_MOE_OFFLOAD_CUDA_GRAPH`" | EXACT (for that env) | `if not cfg.enabled:`<br>`    return`<br>`if bool(envs.SGLANG_MOE_OFFLOAD_CUDA_GRAPH.get()):`<br>`    raise RuntimeError(` | `python/sglang/srt/layers/moe/expert_heat_migration.py:337-340`, called at `expert_offload.py:2531` | Boot-time, at cache construction. |
| S3-13 | **"Eager path only"** (heat migration) | **WIDER — headline** | (absence) the refusal at `expert_heat_migration.py:339` names only `SGLANG_MOE_OFFLOAD_CUDA_GRAPH`; the runtime guard is `if self._capturable_ready:` | `expert_heat_migration.py:339`; `expert_offload.py:3632` | `_capturable_ready` is set only by `install_capturable_buffers()`, which runs only under `if self._moe_offload_graph_mode:` (`layer.py:2362-2363`), i.e. the CAPTURABLE mode. **Under `SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable` neither gate fires**, and `prepare_breakable` calls `self._observe_routing(ids_list)` (`expert_offload.py:3107`) whose tail is `if self._heat is not None: … if self._heat.due(): self._migrate_heat()` (`:3193-3198`). So **#302a heat migration runs live under a CUDA-graph route**. It is structurally sound there — the arena's device addresses are stable (swaps are in-place `buf[i].copy_(tmp)`, `:3653`) and the slot vector is republished through the bridge every replay — which is exactly why this is a reach find, not a bug: cell "#302a × graphs" is occupied and §3/§17 record it as empty. |
| S3-14 | Stage-1 `SGLANG_MOE_HOT_RESIDENCY` is "one-shot freeze", refused under capture | **WIDER** | `if self._moe_offload_graph_mode and envs.SGLANG_MOE_HOT_RESIDENCY.get():` and `if self._graph_mode and self._hot_enabled:` | `layer.py:702`; `expert_offload.py:2605` | Both refusals key on the CAPTURABLE mode only. Live hot calibration + `SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable` is **allowed**, and `_freeze_hotset()` runs from `_observe_routing` inside the break (`expert_offload.py:3179-3186`). Same reasoning as S3-13: address-stable in-place rearrange, so it works — but `SGLANG_MOE_HOTSET_FILE` is no longer the only way to get hot residency under graphs. |
| S3-15 | heat-migration refusal is complete | **NARROWER — BUG-CANDIDATE (spelling escape)** | `if bool(envs.SGLANG_MOE_OFFLOAD_CUDA_GRAPH.get()):` | `expert_heat_migration.py:339`; mirrored at `expert_offload.py:2604` (`self._graph_mode = bool(envs.SGLANG_MOE_OFFLOAD_CUDA_GRAPH.get())`) | `SGLANG_MOE_OFFLOAD_GRAPH_MODE=capturable` + `…_UNSAFE=1` selects the capturable route via `resolve_offload_graph_mode` but leaves the legacy env unset, so **neither the boot refusal nor `_graph_mode` fires**. The second gate (`_capturable_ready`, `:3632`) still catches it, so the harm is a late runtime abort instead of the boot abort the design promises — and the S3-14 hot-residency boot refusal at `expert_offload.py:2605` is bypassed entirely. Two flags now spell one mode and only one of them is read here. |
| S3-16 | "the graph seam … refuses by name for TWO reasons"; "`SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE=1`" | EXACT | `if envs.SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE.get():`<br>`    logging.getLogger(__name__).warning(…)`<br>`    return`<br>`raise RuntimeError(` | `expert_offload.py:2278-2290` (`refuse_capturable_cold_tier`), armed at `:2936-2944` (`if self._cold_tier is not None:`) | Both grounds are in one message; the arming site is the capturable installer only. |
| S3-17 | cold tier × graphs "takes both overrides" | **WIDER** | (absence) `prepare_breakable` → `self._fetch(fetch_plan)`; `_fetch` branches `if remote is not None and expert_id in self._remote_ids:` | `expert_offload.py:3146`; `:2819` | The two-override statement is true of the **capturable** route only. The **breakable** route never calls `install_capturable_buffers`, so `refuse_capturable_cold_tier` is never reached, and its eager `_fetch` carries the peer branch the capturable gather lacks. `SGLANG_MOE_COLD_TIER_SHM=1` + `SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable` is a **legal, unrefused combination** — cold tier under CUDA graphs with no UNSAFE flag at all. Boot-unproven like the rest of #462, but the code path is open by construction, not by omission. |
| S3-18 | "a device breach counter read at the replay boundary" | EXACT (and wider than "the decode graph") | `offload_capture_gate.check_after_graph_replay()` | `model_executor/runner_backend/full_cuda_graph_backend.py:178` **and** `runner_backend/breakable_cuda_graph_backend.py:270` | Wired into both graph backends, so the seam is instrumented on the breakable route too. Default cost is `if not _caches: return` (`offload_capture_gate.py:496`). |
| S3-19 | "capture-admission bound tightened from `tokens x top_k` to `min(tokens x top_k, cold set)`" | EXACT | `slots = max(0, int(routed_slots))`<br>`cold = max(0, int(num_local_experts) - int(resident_count))`<br>`return min(slots, cold)` | `expert_offload.py:2231-2233`, applied at `layer.py:2273-2276` | |
| S3-20 | (undocumented) `SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS` is offload-scoped | **WIDER** | `_moe_offload_graph_bs = envs.SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS.get()`<br>`if _moe_offload_graph_bs > 0:`<br>`    capped = [bs for bs in self.capture_bs if bs <= _moe_offload_graph_bs]` | `model_executor/runner/decode_cuda_graph_runner.py:379-387` | No offload test anywhere in the branch. The env caps the **decode capture-bucket list of every launch**, offload or not — i.e. it is a general decode-graph bucket cap wearing a MoE name. Refuses loudly if it filters everything out (`:382-386`). |
| S3-21 | "`SGLANG_MOE_SCRATCH_SLOTS=6`" is a slot count | NARROWER (doc) | `env = os.environ.get("SGLANG_MOE_SCRATCH_SLOTS", "")`<br>`if env.strip():`<br>`    try:`<br>`        return max(1, int(env))`<br>`    except ValueError:`<br>`        pass`<br>`return max(8, resident_count // 4)` | `expert_offload.py:589-595` | Accepts any int; `<= 0` clamps to 1; a **non-integer silently falls back to the default** — the one env on this surface that does not refuse a typo (contrast `resolve_wave_order`, S3-22, and `env_graph_mode`, S3-08). Default is `max(8, R//4)`, not 6; 6 is the battery's setting. |
| S3-22 | "`SGLANG_MOE_OFFLOAD_WAVE_ORDER`, byte-identical proven" | EXACT | `order = (value or "token").strip().lower()`<br>`if order not in ("token", "expert"):`<br>`    raise RuntimeError(` | `expert_offload.py:357-361` | Exactly two accepted values, default `token`; consumed once at cache construction (`:2598`) and read at `if self._wave_order == "expert":` (`:3223`). Decode is single-wave and unaffected. |
| S3-23 | "`SGLANG_MOE_HOT_RESIDENCY`'s one-shot freeze" | EXACT | `if self._hot_enabled and not self._hot_frozen:` … `if self._hot_seen >= self._hot_calib_steps:`<br>`    self._freeze_hotset()` | `expert_offload.py:3179-3186`; `_hot_calib_steps = max(1, int(envs.SGLANG_MOE_HOT_CALIB_STEPS.get()))` at `:2512` | Boolean env; the tunable is `SGLANG_MOE_HOT_CALIB_STEPS` (default 1), which §3 never names. |
| S3-24 | "#394 … measured H2D provenance chain (env > card-probe > nvml-negotiated > refusal; `absent` unselectable)" | EXACT | `if raw not in ("measured", "estimate"):`<br>`    raise ValueError(f"{HOST_SHARD_MIN_PROVENANCE_ENV}={raw!r} must be 'measured' or 'estimate'. 'absent' is not selectable: …")` | `expert_offload.py:800-805` (`_min_provenance`), chain at `:1019-1058` | Default minimum is `estimate`, so the NVML-derived estimate is admitted by default; `SGLANG_MOE_HOST_SHARD_MIN_PROVENANCE=measured` tightens it. §3 does not name that env. |
| S3-25 | "refusal" is what an absent provenance gets | NARROWER (doc) | `if ratio.is_equal:`<br>`    return None` | `expert_offload.py:1158-1159` (`cold_shard_context`) | At the **cold-tier door** an absent provenance becomes `equal_host_shard_ratio(...)`, which `is_equal`, which returns `None` — a silent degrade to the pre-#394 path, not a refusal. It is a hard refusal only at the **#439 door**: `if ratio.provenance == "absent": raise NoComputeLever(` (`expert_compute_placement.py:951-952`) and in `ColdTierAssignment.__post_init__`: `if self.ratio_provenance == "absent": raise ValueError(` (`cold_tier_fetch.py:218-219`). |
| S3-26 | "with it off the slice-1 boot refusal for delegation on disjoint expert shards is unchanged, field for field" | NARROWER (doc) | `if (`<br>`    not cold_tier_enabled()`<br>`    and not envs.SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE.get()`<br>`):`<br>`    _warn_host_shard_unreachable_once()`<br>`    return None` | `layer.py:1450-1455` | At the GGUF streaming door the "refusal" is a **warn-once + fall back to the pre-#394 plan**, not a raise. The hard `raise ValueError` lives at the *other* door: `refuse_cold_shard_at_repack_door` (`expert_offload.py:3755-3782`), whose own escape is `if eligible and envs.SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE.get(): return` (`:3756-3757`). Two doors, two different refusal shapes; §3 describes one. |
| S3-27 | "a rank-uniform owner map … plan `digest()` pins the uniformity" | EXACT | `payload = "|".join((str(self.world_size), ",".join(str(e) for e in self.cold_ids), ",".join(str(o) for o in self.owners), self.ratio_source, ",".join(f"{w:.12g}" for w in self.weights)))`<br>`return hashlib.sha256(payload.encode()).hexdigest()[:16]` | `python/sglang/srt/layers/moe/cold_tier_fetch.py:262-270` | `rank` is deliberately excluded so cross-rank comparison is not trivially true. |
| S3-28 | `SGLANG_MOE_COLD_TIER_SHM=1` gates the tier | NARROWER (doc) | `try:`<br>`    from sglang.srt.environ import envs`<br>`    return bool(envs.SGLANG_MOE_COLD_TIER_SHM.get())`<br>`except Exception:`<br>`    return os.environ.get("SGLANG_MOE_COLD_TIER_SHM", "") in ("1", "true", "True")` | `cold_tier_fetch.py:141-146` | Two different truth tests. The fallback accepts only three exact strings (`TRUE`, `yes`, `on` are false there) while `EnvBool` is broader. Desk-only divergence, but it means a hermetic context can disagree with a boot about whether the tier is on. Additional hard precondition §3 omits: `if not card_uuids or len(card_uuids) != int(world_size): raise ColdTierUnavailable(` (`:435-436`) — the tier needs a published rank→card vector (`--rank-gpu-id`). |
| S3-29 | "the #82 pad expert is never demoted and a #394-delegated expert is never promoted" | EXACT | `victims = sorted((e for e in resident if e not in pinned), key=lambda e: (h(e), e))` and `if cand in delegated:`<br>`    …`<br>`    continue` | `expert_heat_migration.py:198`, `:212-215`; sources at `expert_offload.py:3585-3594` (`pinned` from `layer._moe_offload_pinned_experts`, `delegated` from `self.planner.delegated_ids`) | `pinned` is populated only on the load-time-staging path (`if plan.pinned_ids:`, `expert_offload.py:1512`); on non-GGUF lanes there is no pad expert to protect. |
| S3-30 | "Two-sided hysteresis (relative margin plus an absolute `min_gain` floor)" | EXACT | `if h(cand) <= h(victim) * (1.0 + hysteresis) or h(cand) - h(victim) < min_gain:`<br>`    …`<br>`    break` | `expert_heat_migration.py:219-223` | Both must hold; the `break` (not `continue`) is justified by the two opposing sort orders. |
| S3-32 | "#286 (d) A NAMED refusal under active capture (`OffloadUnderCaptureRefused`, probing `get_is_capture_mode()`)" | NARROWER (doc) | `if capture_is_active():`<br>`    raise OffloadUnderCaptureRefused(`<br>`        operation, subject, where, ground=GROUND_CAPTURE_ACTIVE`<br>`    )` | `python/sglang/srt/model_executor/short_term_offload_register.py:700-703` | `capture_is_active()` returns False on **any** exception importing the probe (`except Exception: return False`, `:617-618`), and an injected test probe wins outright (`if probe is not None: return bool(probe())`, `:611-612`). The contextvar is context-local (`:600-603`), so a park issued from a different thread than the capturing one is not refused. |
| S3-33 | "#286 (f) `experts` VA stability is ROUTE-acquired … `ground=GROUND_GRAPH_ADDRESSED`" | EXACT (and narrow) | `desc = ASSET_CLASSES.get(offload_class)`<br>`if desc is None or desc.va_stable_required:`<br>`    return None`<br>`if not desc.va_stable_when_graph_addressed:`<br>`    return None`<br>`holders = sorted(k for k, v in _addressing_families().items() if offload_class in v)`<br>`if not holders:`<br>`    return None` | `short_term_offload_register.py:638-645` | Dynamic on the family side (`register_family(..., addresses_classes=...)`, `:1302`) but static on the class side: `va_stable_when_graph_addressed=True` is set on **exactly one** class, `experts` (`:388`). Classes declaring permanent VA stability (`graph_rungs`, `gdn_state_sets`) are excluded by the first conjunct. |
| S3-34 | "#286 (b) `plan_spill` … rank 5 (active work) never planned; partial spill; coldest-first" | EXACT | (i) `if desc.ladder_rank in _UNPLANNABLE_RANKS:` (`_UNPLANNABLE_RANKS = frozenset({LadderRank.ACTIVE_WORK})`, `:193`)<br>(ii) `if running >= bytes_needed:`<br>`    plan.satisfied = True`<br>`    break`<br>(iii) `return (int(rank), float(item.last_access_s), item.item_id)` | `short_term_offload_register.py:1039`, `:1072-1074`, `:963` (applied at `:1031`) | Rank 5 is derived from the enum, not a literal. Four further skips §3 does not mention, all before the ladder walk: `item.parked` (`:1046`), `item.prio_protected` (`:1049`), `item.hot()` (`:1052`), `item.size_bytes <= 0` (`:1055`), and the #468 graph refusal at `:1034`. |
| S3-35 | "an IMPORT-TIME guard that a new class without a descriptor cannot exist" | EXACT | `_MISSING_DESCRIPTORS = tuple(k for k in OFFLOAD_CLASSES if k not in ASSET_CLASSES)`<br>`if _MISSING_DESCRIPTORS:`<br>`    raise RuntimeError(` | `short_term_offload_register.py:465-467` | One-directional; the reverse is caught by `if self.offload_class not in OFFLOAD_CLASSES:` (`:272`). |
| S3-36 | "#286 (e) `gdn_state_sets` is classified per CONTENT STATE … the EXPORTED blob is `EXPENSIVE_RECONSTRUCTABLE`" | EXACT (name drift) | `if state is ContentState.SUSPENDED and self.suspended_payload is not None:`<br>`    return self.suspended_payload`<br>`return self.payload` | `short_term_offload_register.py:285-287`; declaration at `:410-411` | The enum member is `SUSPENDED`, not "EXPORTED". `park_requires_suspend` is `return self.suspended_payload is not None` (`:297`) and the consumer default is the conservative `content_state: ContentState = ContentState.LIVE` (`:788`). |
| S3-37 | **"Whole surface is behind `SGLANG_OFFLOAD_REGISTER=1` (dark launch)"** | **NARROWER — BUG-CANDIDATE (claim inverted)** | `if not offload_register_enabled():`<br>`    return None` | `python/sglang/srt/model_executor/offload_register.py:1227-1228` and `:1261-1262`; flag at `:1176` (`return bool(envs.SGLANG_OFFLOAD_REGISTER.get())`), default `EnvBool(False)` at `environ.py:1079` | The flag gates **only the process-global holder** (`get_global_register`, `configure_global_register_from_server_args`, hence the `maybe_*` adapters). Everything else runs with the flag off: `parse_park_target_order`, `parse_class_policy_overrides`, `resolve_class_policies` (called at argument time from `ServerArgs._handle_lane_offload_register`, `server_args.py:6663-6668`), `configure_global_register` itself (no gate, `:1179`), and the whole of `short_term_offload_register.py` — which never calls `offload_register_enabled()` at all and IS reached in production: `breakable_offload.py:216` imports `describe_class` at arena construction and `:268-272` calls `refuse_if_capture_active` on every arena park. Harm: an operator reading "dark launch" will assume the #286 asset-class layer is inert with the flag off; it is not. |
| S3-38 | "a typo now refuses there too" (`--lane-offload-*` at runner init) | **NARROWER — BUG-CANDIDATE** | `if not offload_register_enabled():`<br>`    return None` — **before** `order = parse_park_target_order(park_spec, "--lane-offload-park-targets")` | `offload_register.py:1227-1228` vs `:1239` | The runner-init half is behind the env gate, so with `SGLANG_OFFLOAD_REGISTER` unset (the default) a typo refuses **only** at argument time (`server_args.py:6663-6668`). The docstring at `:1222-1225` states the opposite motivation — "including on the direct-`ServerArgs`-construction path that never runs `__post_init__`'s validator" — which is exactly the path the gate above it suppresses. |
| S3-39 | "The park chain reaches the register and the movement layer's default reads it — but nothing in production constructs the movement backend yet" | EXACT | `self._backend = backend or CpuFakeMovementBackend()` with `backend: Optional[MovementBackend] = None` and no caller passing one | `offload_register.py:563`, `:1182`, `:1240-1244`; read side `if target_order is None:`<br>`    target_order = park_target_order_from_register()` at `model_executor/offload_movement.py:663-664` | `RealMovementBackend` (`offload_movement.py:631`) is constructed at six test sites and two `scripts/gpu_battery/` sites; zero under `python/sglang/`. Even with the flag on, the global register moves bytes through the CPU fake. Consumer PATH, no consumer — confirmed. |
| S3-40 | "memtier registry … provenance `measured|estimate|absent` (absent refuses use)" | **WIDER** | `if bandwidth.is_absent:`<br>`    if query.min_bandwidth_gbs is not None:` … `    if not query.allow_unmeasured_bandwidth:` | `python/sglang/srt/memtier/registry.py:407-408`, `:418` | `absent` refuses on the **bandwidth axis only**, and only when a floor was asked for or `allow_unmeasured_bandwidth=False`. An absent **latency** never refuses anything (no `latency` predicate exists in `registry.py`); absent **capacity** refuses only when bytes are actually requested (`if query.bytes_needed <= 0: return None`, `:455`, before `if headroom.is_absent:` at `:458`). |
| S3-41 | "refuses a tier whose bandwidth is ABSENT (and, by default, one that is only an ESTIMATE)" | NARROWER (doc) | `if (`<br>`    query.require_measured_bandwidth`<br>`    and bandwidth.provenance is not Provenance.MEASURED`<br>`):` | `registry.py:428-431`; default `require_measured_bandwidth: bool = False` at `:199` | The **registry's** default admits an ESTIMATE. "By default refused" is the *caller's* default: `price_park_target(..., require_measured: bool = True)` (`short_term_offload_register.py:787`, passed at `:837`). The override is therefore `require_measured=False` — a kwarg, not an env or a flag. The other caller, `memtier/consumers.expert_offload_host_targets`, defaults to `require_measured_bandwidth: bool = False` (`consumers.py:86`) and sets `allow_unmeasured_bandwidth=not require_measured_bandwidth` (`:117`), i.e. admits ABSENT by default. |
| S3-42 | "`TierRegistry.for_machine()` … applies a stored profile ONLY at the scope its hardware match licenses (`EXACT` = every tier; `MODEL` = card templates only)" | EXACT (+ undocumented escape) | `if stored_hardware and stored_hardware == fingerprint.hardware_key:` → EXACT<br>`if stored_model and stored_model == fingerprint.model_key:` → MODEL<br>else NONE | `python/sglang/srt/memtier/fingerprint.py:313`, `:325`, `:339` | Two hard pre-gates first: `if not isinstance(hardware, Mapping):` (`:287`) and `if version != FINGERPRINT_VERSION:` (`:300`). EXACT needs every card row complete: `if not uuid or not model or total is None:` (`:377`). MODEL filter is total, not selective: `reduced = {k: v for k, v in document.items() if k not in ("tiers", "device_models")}` then `reduced["tiers"] = []` (`:416-419`). **Undocumented escape §3 does not mention**: `if (trust_explicit and explicit and Path(explicit) == Path(path) and match.scope is MatchScope.NONE):` promotes NONE → full EXACT under `SGLANG_MEMTIER_PROFILE_TRUST=1` (`memtier/profile_store.py:208-213`). |
| S3-43 | "`from_profile()` no longer defaults to the bundled rig profile" | EXACT | `def from_profile(`<br>`    cls,`<br>`    profile: RigProfile,`<br>`    facts: Optional[LocalFacts] = None,`<br>`) -> TierRegistry:` | `registry.py:243-248` | Positional, no default. The bundled profile is still *reachable* — `for directory in (profile_store_dir(),) + ((_BUNDLED_DIR,) if include_bundled else ()):` with `include_bundled: bool = True` (`profile_store.py:153-161`) — but only through a hardware match, which is the guarantee actually claimed. |
| S3-44 | "`link_disjointness()` … `UNKNOWN` being a refusal" | **NARROWER — BUG-CANDIDATE (unenforced)** | `if not pa or not pb:` → UNKNOWN (`tiers.py:570`); `if shared:` → SHARED (`:581`); `if incomplete:` → UNKNOWN (`:591`); fallthrough → DISJOINT (`:602-603`) | `python/sglang/srt/memtier/tiers.py:570-603` | UNKNOWN is **returned, never refused**: `grep -rn "link_disjointness\|LinkVerdict" python/ test/` finds no production caller — only `tiers.py`, the `__init__` re-export and one test. "UNKNOWN is a refusal" is a docstring claim (`tiers.py:566-567`) with no `raise` behind it. Additionally every bundled remote tier ships `link_path_complete: false` (`memtier/profiles/rig1.json:165,195`) and bootstrap hardcodes `link_path_complete=False` (`memtier/bootstrap.py:241`), so on this rig DISJOINT is unreachable from real data. |
| S3-45 | "`--rank-auto-reserve-mib auto` … the refusal now names the derivation and the pinned value that fits" | EXACT | `if str(self.rank_auto_reserve_mib) != self.AUTO_RANK_MEMORY_RESERVE_MIB:`<br>`    return None` | `python/sglang/srt/server_args.py:10605-10606` | The note is emitted only when the reserve really was derived; under a pinned reserve it returns `None` and the standing advice stands. |
| S3-46 | "a per-rank resident-fraction vector" (implied by `SGLANG_MOE_RESIDENT_EXPERT_FRACTION` being a vector env) | NARROWER (doc) | `if moe_size is not None and tp_size is not None and moe_size != tp_size:`<br>`    raise ValueError(` | `python/sglang/srt/layers/moe/resident_fraction.py:132-133` | A per-rank vector is refused whenever the MoE group differs from the attention-TP group; a scalar still broadcasts. Also refused: env and flag disagreeing (`if env_v is not None and flag_v is not None and env_v != flag_v:`, `:107`) and a length mismatch (`:141`). §3 never mentions that the offload fraction is a per-rank vector at all. |
| S3-47 | "#439 … `--rank-moe-ratio link`" preconditions | EXACT | `if symbol not in COMPUTE_PLACEMENT_SYMBOLS:`<br>`    return None` … `if ep_size > 1:`<br>`    raise NoComputeLever(` … `if not cards.present:`<br>`    raise NoComputeLever(` … `if ratio.provenance == "absent":`<br>`    raise NoComputeLever(` | `python/sglang/srt/layers/moe/expert_compute_placement.py:895-896`, `:916-917`, `:941-942`, `:951-952` | Four refusals; also requires a resolved uneven-TP base plan of length `tp_size` (`:906-909`). `link` vs `link-calibrated` is the only thing selecting the calibrated solve. |
| S3-48 | "#456 writes the image SPARSE by default … `SGLANG_HIBERNATE_DENSE_WRITE=1` to opt out" | EXACT (+3 spellings) | `raw = (env if env is not None else os.environ).get(DENSE_WRITE_ENV, "")`<br>`return raw.strip().lower() in ("1", "true", "yes", "on")` | `python/sglang/srt/model_loader/sparse_write.py:269-270`, read per call at `:282-283` | Four accepted spellings, not one. Reach of the opt-out is complete: `torch_save_sparse` has exactly one non-test caller, `sparse_stats = torch_save_sparse(payload, tmp)` (`model_loader/hibernate.py:465`). The manifest is always written densely (`json.dump`, `:524-526`). |
| S3-49 | **"Hibernate to disk (weights+KV survive process exit)"** | **NARROWER — BUG-CANDIDATE (payload overstated)** | `payload = {`<br>`    "version": HIBERNATE_VERSION,`<br>`    "tp_rank": tp_rank,`<br>`    "nvml_uuid": nvml_uuid,`<br>`    "params": params_cpu,`<br>`    "static_state": static_cpu,`<br>`    "gguf_attrs": gguf_attrs,`<br>`    "byte_hash": byte_hash,`<br>`}` | `python/sglang/srt/model_loader/hibernate.py:448-456` | **No KV is parked.** `params` is `named_parameters`, `static_state` is `named_buffers`. `grep -n "kv\|KV" hibernate.py` returns only comments about KV *profiling*. A reader planning a "KV survives restart" feature on this line would be planning on a mechanism that does not exist. |
| S3-50 | (undocumented) hibernate scope | NARROWER (doc) | `if getattr(server_args, "pp_size", 1) not in (1, None): problems.append(...)` … `if getattr(server_args, "ep_size", 1) not in (1, None): …` … `if problems:`<br>`    raise ValueError("#89 hibernate V1 is scoped to pure single-node Tensor Parallelism; refusing to combine with: " …)` | `model_loader/hibernate.py:106-123`; GGUF scope `if self.load_format != "gguf": raise ValueError(` at `server_args.py:13204-13208`; per-rank card recheck `if live_uuid != rank_meta["nvml_uuid"]: raise RuntimeError(` at `hibernate.py:568-574` | §3 names none of this. Hibernate is **GGUF-only**, single-node pure-TP, and refuses if a rank's NVML UUID moved. Uneven TP is genuinely covered: `rank_tp_ratio`/`rank_gpu_id` are in the identity hash (`:96-97`). |
| S3-51 | `--enable-weights-disk-backup` / `--hibernate-dir` mutual requirement (AUDIT_421 §7.1 gap) | EXACT | `if self.enable_weights_disk_backup and self.hibernate_dir is None:`<br>`    raise ValueError("--enable-weights-disk-backup requires --hibernate-dir.")`<br>`if self.hibernate_dir is not None and not self.enable_weights_disk_backup:`<br>`    raise ValueError("--hibernate-dir requires --enable-weights-disk-backup.")` | `python/sglang/srt/server_args.py:13191-13199` | Genuinely mutual, both directions raise, argument time. `enable_weights_disk_backup` is read **nowhere outside `server_args.py`** — every runtime site keys off `hibernate_dir` / `load_format == "hibernate"`. |
| S3-52 | "suspend-to-RAM … reaches the legacy hybrid-SWA `SWAKVPool` since upstream #32213" | EXACT | `kwargs.setdefault("enable_memory_saver", False)` (no longer a hardcoded `False`) | `python/sglang/srt/mem_cache/swa_memory_pool.py:54`; both call sites pass `enable_memory_saver=self.server_args.enable_memory_saver` (`model_executor/model_runner_kv_cache_mixin.py:3311`, `:3491`) | Confirmed at this tip. |
| S3-53 | "`UnifiedSWAKVPool` already honoured it" | EXACT (by delegation, dead param) | `enable_memory_saver: bool = False,` — never referenced again in the class body | `python/sglang/srt/mem_cache/unified_memory_pool.py:1014` | The saving happens one level down: `self.memory_saver_adapter = TorchMemorySaverAdapter.create(enable=enable_memory_saver)` / `with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):` (`:212-216`, fed from `:1311`). Net behaviour correct; the class's own parameter is inert — a trap for a direct constructor. No pool allocates outside a saver region. |
| S3-54 | **"`--kv-pressure-ladder auto` mode wired via rig-profile bridge (#428)"** — AUDIT_421 F1 recheck | EXACT — **F1 is fixed at this tip** | `if spec == LADDER_SPEC_AUTO:`<br>`    if table_fn is None:`<br>`        raise ValueError("--kv-pressure-ladder auto needs the planner's step-table source; none was supplied. …")`<br>`    table = table_fn()` | `python/sglang/srt/model_executor/kv_pressure_ladder.py:1954-1962`; production `table_fn` supplied at `managers/kv_pressure_runtime.py:467-469` (`ladder = build_ladder_from_server_args(server_args, table_fn=auto_ladder_table_fn(server_args))`), reached from `managers/scheduler.py:3343-3349` | `table_fn` is non-None in production, so `auto` resolves to a real table. The `raise` survives only for direct/hermetic callers. Do not carry AUDIT_421 F1 forward. |
| S3-55 | `auto` "is computed from the rig profile … rank-uniformly and UUID-keyed" | EXACT | `if not devices:`<br>`    raise ValueError("--kv-pressure-ladder auto needs the rig profile, and NVML reported no devices. …")` and `if len(names) > 1:`<br>`    raise ValueError("--kv-pressure-ladder auto cannot map ranks to cards on this rig: the node holds different card models …")` | `python/sglang/srt/managers/kv_ladder_auto.py:184-189`, `:190-200`; co-located-rank budget mismatch at `:153-161` / `:207-214` | Loud everywhere except the homogeneous-node-without-a-vector case, which falls back to CUDA ordinals (`_cards_from_homogeneous_node`, `:172`). On this mixed rig `auto` therefore **requires `--rank-gpu-id`**; §3 does not say so. |
| S3-56 | "rung-dependency refusals exist and fire" | EXACT (argument time) | `if "admission_cap" in names and self.max_running_requests_ceiling is None:`<br>`    raise ValueError(…)`<br>`if "session_offload" in names and not self.enable_kv_session_offload:`<br>`    raise ValueError(…)` | `server_args.py:6961-6974`, guarded by `if isinstance(spec, tuple):` at `:6955` | Applies to **explicit specs only, never to `auto`** — correct by construction, but it means the typo-refusal §3 credits does not exist on the `auto` path. Fires again at runtime with a stricter test: `if admission_limiter is None or not admission_limiter.auto:` (`managers/kv_pressure_runtime.py:161-167`). |
| S3-57 | "inventories only rungs whose actuator this configuration wires" | NARROWER (doc) | `if getattr(server_args, "kv_reshard_vectors", None) is not None:`<br>`    reliefs.append("dcp_ratio")`<br>`if getattr(server_args, "max_running_requests_ceiling", None) is not None:`<br>`    reliefs.append("admission_cap")`<br>`if getattr(server_args, "enable_kv_session_offload", False):`<br>`    reliefs.append("session_offload")` | `python/sglang/srt/managers/kv_ladder_auto.py:99-108` | Three **flag-presence** checks, not an actuator probe. `kv_spill` / `weightless_rank` are excluded deliberately. Stale constant: `WIRED_RELIEFS = ("admission_cap", "session_offload")` (`kv_pressure_runtime.py:87`) omits `dcp_ratio` and is **never read** (`grep -rn WIRED_RELIEFS --include=*.py python/` → definition + one docstring). The real runtime inventory is the loop at `kv_pressure_runtime.py:157-183`; unrecognised relief names land in `planned_only.append(feature)` (`:182`) — logged, not refused. |
| S3-58 | **"KV session offload (kvso) … decoupled from speculation"** | **NARROWER — BUG-CANDIDATE (claim inverted)** | `if self.speculative_algorithm is not None and (`<br>`    os.environ.get("KVSO_ALLOW_SPEC", "0") != "1"`<br>`):`<br>`    …`<br>`    raise ValueError("--enable-kv-session-offload does not yet support speculative decoding …")` | `python/sglang/srt/server_args.py:6580-6595` | The opposite of the catalog. `--enable-kv-session-offload` + **any** `--speculative-algorithm` is a **boot failure** unless the undocumented env `KVSO_ALLOW_SPEC=1` is set. Even past the env, spec constrains the victim: `return spec_active and idx != n_reqs - 1` (`managers/kv_session_offload.py:1041`, only the LAST request may spill) and `return spec_active and enable_overlap` (`:1074`). `--draft-kv-layout dcp` + kvso is refused outright (`server_args.py:7558-7566`). On this rig, where NEXTN/MTP is the standing production path, the documented-as-compatible pair does not boot. |
| S3-59 | "idle-first victim choice" | **NOT-FOUND / contradicted** | `return (spill_class_rank(req), fast, -seq)` | `python/sglang/srt/managers/kv_session_offload.py:844` (`session_priority_key`), used by `select_spill_victim` at `:908-919` | The key is **spill-class → fast-lane → youngest arrival (`kv_arrival_seq`)**. No idleness / last-token-time term exists anywhere. Greps: `grep -n "idle" kv_session_offload.py` → 15 hits, all the tick regulator's device-idle measurement (`:1801-1932`, `:4312-4339`), unrelated to victim choice; `grep -n "victim\|_pick"` → 50 hits, none idle-keyed. "FCFS spill of youngest sessions" is right; "idle-first" is not. |
| S3-60 | "(KV only, GDN stays resident)" | EXACT (by absence) | `kv = max(0, int(seq_len) - int(req.cache_protected_len or 0))`<br>`return [("kv", kv)]` | `python/sglang/srt/managers/kv_session_offload.py:126-127` (`bundle_spillable_sizes`) | GDN is un-spillable structurally — there is no `("gdn_state", …)` entry — not by a guard. Related dtype refusal: `raise ValueError("GDN/Mamba state must stay at its native dtype …")` (`:1272-1276`). |
| S3-61 | "budgets (volume/rate/window, **demote to HiCache**)" | **NARROWER — BUG-CANDIDATE (mutually exclusive)** | `if self.enable_hierarchical_cache: raise ValueError(…)` in the kvso validator | `server_args.py:6620` (kvso × hicache mutual exclusion, inside the `enable_kv_session_offload` block at `:6496`) | kvso and HiCache **cannot be enabled together**, so "demote to HiCache" cannot describe a reachable configuration as written. Other scope refusals in the same validator (`:6596-6641`): flashinfer backend only, `page_size == 1`, no PD disagg, no `--weightless-kv-fastlane`, no `--enable-unified-memory`, no `--enable-hisparse`, `pp_size == 1 and dp_size == 1`, no `--enable-mixed-chunk`. |
| S3-62 | **"HiCache … validated with uneven DCP/TP"** — is there a RESTRICTING gate? | **WIDER (no restriction exists; HiCache adapts)** | `dcp_owner_mode=self._dcp_owner_ctx() is not None,` (branch, not refusal) | `python/sglang/srt/managers/cache_controller.py:655`, `_dcp_owner_ctx` → `uneven_dcp_owner_bounds()` at `:800` | No gate anywhere refuses HiCache under uneven DCP/TP. Instead the mode drops the rank suffix from KV keys: `if not is_mla_model:`<br>`    self.config_suffix += f"_{tp_rank}_{tp_size}"`<br>`    if not self.dcp_owner_mode:`<br>`        self.kv_config_suffix += f"_{tp_rank}_{tp_size}"` (`mem_cache/hicache_storage.py:427-430`, applied at `:517-519`). The only HiCache refusals in `server_args.py` are `enable_hierarchical_cache and disable_radix_cache` (`:13545`) and the kvso exclusion (`:6620`). |
| S3-63 | "storage key includes kv-dtype" | EXACT | `identity_parts = [`<br>`    os.path.normpath(server_args.model_path) if server_args.model_path else "",`<br>`    server_args.revision or "",`<br>`    str(server_args.dtype or "auto").lower(),`<br>`    server_args.quantization or "",`<br>`    str(server_args.kv_cache_dtype or "auto").lower(),`<br>`]` | `python/sglang/srt/mem_cache/hicache_storage.py:42-50`, folded in at `:423-426` | Five identity fields, not one — model path, revision, dtype, quantization and kv-dtype. |
| S3-64 | "**Runtime VRAM dial** per card (VMM page return)" | NARROWER (doc) | `if server_args.device != "cuda": problems.append(…)` … 10 further `problems.append` arms … `raise ValueError("--enable-vram-dial is refused with: " + "; ".join(problems) …)` | `python/sglang/srt/managers/vram_dial.py:1045-1086` | Eleven refusal arms; the argparse help (`server_args.py:5078-5082`) lists six. **Undocumented refusals: `device != cuda`, `--kv-canary`, `--enable-hisparse`, `--weightless-kv-fastlane`, the DFLASH spec lane.** Four more at build time: `if not uneven_dcp_active(dcp_size): raise KvCapacityError(` (`:1110-1116`) — **the dial requires WEIGHTED uneven DCP**, which §3's "per card" phrasing does not convey — plus `if not participants:` (`:1118`), `if getattr(p.pool, "use_mla", False):` (`:1132`), and the HND-layout refusal at `model_executor/model_runner_kv_cache_mixin.py:3706-3711`. No driver-version check and no VMM support probe exists. |
| S3-65 | the dial is "per card" | EXACT | `if len(budgets) != self.tp_size: raise ValueError(…)` and request field `device: str = "all"` (`rank:N` / `cuda:N` / NVML UUID / `all`) | `model_runner_kv_cache_mixin.py:5197-5202`, `vram_dial.py:1163-1167`; `managers/io_struct.py:1473-1485` | Per-rank on the request side; the resulting KV capacity is min-reduced group-wide at the consensus boundary (`managers/scheduler.py:4997`). |
| S3-66 | "**KV resharding** at phase boundaries (delta move <1 s, `kv_reshard_vectors`)" | NARROWER (doc) | `if not uneven_dcp_active(dcp_size):`<br>`    raise KvReshardError("--kv-reshard-vectors requires WEIGHTED uneven DCP …")` and `if not isinstance(pool, HybridLinearKVPool):`<br>`    raise KvReshardError("--kv-reshard-vectors Stage A supports the hybrid-linear KV pool family only …")` | `python/sglang/srt/managers/kv_reshard.py:713-729`; enable gate `if self.server_args.kv_reshard_vectors is not None:` at `managers/scheduler.py:3317-3322` | Two hard preconditions §3 omits: weighted uneven DCP, and the hybrid-linear pool family only (SWA-hybrid and dense-DCP are named follow-ups). |
| S3-67 | "**GDN slot ladder** (resident-state cap + idle vacate → VRAM back to KV pool)" | **NARROWER — BUG-CANDIDATE (inert by default)** | `if not offload_register_enabled():`<br>`    return []` | `python/sglang/srt/model_executor/offload_gdn_states.py:344-345` | The ladder attach (`_maybe_attach_ladder_from_server_args`, `:367`) sits downstream of that early return, so `--gdn-state-set-ladder` / `--gdn-resident-state-slots` are **validated at boot but produce zero effect** unless `SGLANG_OFFLOAD_REGISTER=1`. Their validators do fire (`if self.gdn_resident_state_slots is not None and self.gdn_resident_state_slots < 1: raise ValueError(`, `server_args.py:6841-6849`), which makes the inertness harder to notice, not easier. §3 lists the ladder outside the `SGLANG_OFFLOAD_REGISTER` sentence. |
| S3-68 | "`--lane-offload-profile/-class-policy/-park-targets` are wired at runner init once-per-process (#428)" | NARROWER (doc) | `if not offload_register_enabled():`<br>`    return None` (before any parsing) | `offload_register.py:1227-1228` vs the parse at `:1239` | Same root as S3-38/S3-67. Argument-time validation always runs (`server_args.py:6663-6668`); runner-init wiring never does by default. |
| S3-69 | "`rank_moe_resident_fraction`" per-rank vector | EXACT | `if len(vec) not in (1, self.tp_size): raise ValueError(…)` and `if not (0.0 < f <= 1.0): raise ValueError(…)` | `server_args.py:9347`, `:9354` (parser `_parse_rank_moe_resident_fraction` at `:438`) | Complements S3-46: length and range refuse at argument time; the moe-group/tp-group mismatch refuses at read time. |
| S3-70 | "#456 … `hicache_migrate.execute_plan` (#297) does not share this writer" | EXACT | `grep -rn "torch_save_sparse" --include=*.py python/` → one non-test caller, `model_loader/hibernate.py:465` | `model_loader/sparse_write.py` / `model_loader/hibernate.py:465` | Confirmed: every other `torch.save(` under `python/sglang/srt/` belongs to a different feature (`dumper.py:1097`, `layer_fingerprint.py:187`, `dual_group_lane.py:3683`, `model_runner.py:4331`, `spec_verify_dump.py:161`, `expert_distribution.py:954`, `dspark_sts.py:67`) and was never sparse. |

---

#### WIDER finds — what they unlock

**S3-10/S3-11 — expert offload is not a three-format feature.** The quant guard
is a five-name *denylist* (`expert_offload.py:2056-2064`), so every quant method
not named there is admitted by default, and `GGUFMoEMethod` has been admitted
conditionally since #123-GGUF via `_OFFLOAD_CONDITIONAL_QUANT_METHOD_NAMES`
(`:2069-2071`, `:2109-2112`). Practically: GGUF-MoE on CUDA (any ggml type with a
MoE kernel) *and* unquantized/INT8/compressed-tensors-non-NVFP4 checkpoints can
run `--rank-moe-resident-fraction < 1.0` today. Any planner or solver that
excluded a model from the offload lane because "the catalog lists fp8/GPTQ/AWQ"
excluded a lane the code admits — the exact #492 shape.

**S3-13/S3-14 — #302a heat migration and Stage-1 hot residency run under CUDA
graphs on the breakable route.** Both refusals key on the *capturable* mode only
(`expert_heat_migration.py:339`, `expert_offload.py:3632` via `_capturable_ready`,
`layer.py:702`), and `prepare_breakable` routes through `_observe_routing`
(`expert_offload.py:3107` → `:3179-3198`), which contains both. This is sound by
construction — swaps are in-place into address-stable arena slots and the
slot vector is republished each replay — so the §17 matrix cell "#302a × graphs"
is occupied, not empty. It also means a #462 F2 window measures heat migration
by default unless the operator turns it off, which would confound the break-cost
number the ticket is after.

**S3-17 — cold tier × graphs needs zero override on the breakable route.**
`refuse_capturable_cold_tier` is armed only inside `install_capturable_buffers`
(`expert_offload.py:2936-2940`), which the breakable route never calls, and the
breakable fetch is the eager `_fetch` with its `expert_id in self._remote_ids`
peer branch intact (`:2819`, reached from `:3146`). So
`SGLANG_MOE_COLD_TIER_SHM=1` + `SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable` is a
legal combination with neither UNSAFE flag — §3's "reaching it now takes both
overrides" is true of the capturable route only.

**S3-20 — `SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS` is a general decode-bucket cap.**
No offload test guards it (`decode_cuda_graph_runner.py:379-387`). Any launch can
use it to trim the captured decode bucket list, which is a VRAM lever
(capture-pool size) independent of MoE entirely.

**S3-37 — the #286 asset-class layer is live with `SGLANG_OFFLOAD_REGISTER` off.**
The flag gates the process-global holder only (`offload_register.py:1227`,
`:1261`). `short_term_offload_register.py` — descriptors, the import-time guard,
`refuse_if_capture_active`, `plan_spill`, `price_park_target` — is reachable and
is already executed in production through `breakable_offload.py:216/268-272`.

**S3-40/S3-41 — memtier admits more than "measured".** Absent bandwidth passes
whenever `allow_unmeasured_bandwidth=True` and no floor is set
(`registry.py:418`); the registry's own `require_measured_bandwidth` default is
`False` (`:199`). An absent *latency* is never refused at all.

**S3-06 (secondary) — on DeepSeek-V4 the breakable route's prefill precondition
is met without a flag**, because `_disable_breakable_cudagraph_if_incompatible`
rewrites prefill to `disabled` for DSV4 (`server_args.py:8281-8309`). Only
`--cuda-graph-backend-decode=breakable` has to be typed.

**S3-62 — HiCache is unconditional with respect to uneven DCP/TP.** There is no
refusal anywhere; the cache *branches* on `dcp_owner_mode`
(`cache_controller.py:655`, `:800`) and drops the rank suffix from the KV key
(`hicache_storage.py:427-430`). Practically: an uneven-DCP boot writes
rank-agnostic KV entries, so the L2/L3 tiers are shareable across a rank set
that a rank-suffixed key would have partitioned — and no planner needs to
exclude HiCache from an uneven-DCP candidate.

---

#### NARROWER — bug candidates

**S3-05 — `validate_breakable_boot` has a total-bypass arm.**
`if backend is None: return` (`offload_capture_gate.py:358-362`) skips *both*
preconditions whenever `get_server_args()` raises or `cuda_graph_config` /
`phase_config.backend` is absent (`resolved_backend`, `:408-421`, catches
`Exception`). The breakable route then boots under `full`, where `eager_on_graph`
is a pass-through and `topk_ids.tolist()` executes inside a live stream capture —
the illegal-D2H failure the gate exists to prevent.
→ *Task title: "#462: validate_breakable_boot must refuse, not skip, when the resolved CUDA-graph backend is unknown"*

**S3-15 — the capturable refusals read only the legacy env spelling.**
`expert_heat_migration.py:339` and `expert_offload.py:2604` both test
`envs.SGLANG_MOE_OFFLOAD_CUDA_GRAPH`, while the mode is now selected by
`resolve_offload_graph_mode`, which also honours
`SGLANG_MOE_OFFLOAD_GRAPH_MODE=capturable`. With the new spelling the boot
refusals for heat migration and live hot residency do not fire; only the later
`_capturable_ready` guard catches it, and the hot-residency one is bypassed
outright. One mode, two spellings, one of them unread.
→ *Task title: "MoE offload: route the capturable-mode refusals through resolve_offload_graph_mode instead of the legacy SGLANG_MOE_OFFLOAD_CUDA_GRAPH bool"*

**S3-37 — "whole surface is behind `SGLANG_OFFLOAD_REGISTER=1`" is false and the
error is load-bearing.** An operator or a future consumer that trusts the
dark-launch framing will assume `short_term_offload_register` is inert with the
flag off; it is reached on every breakable arena construction and park
(`breakable_offload.py:216`, `:268-272`) and its import-time guard fires
regardless.
→ *Task title: "Correct the SGLANG_OFFLOAD_REGISTER dark-launch scope in the catalog and add the gate to short_term_offload_register's production entry points"*

**S3-38 — `--lane-offload-*` typo refusal at runner init is suppressed by the
very gate it sits behind.** `configure_global_register_from_server_args` returns
before parsing when the env flag is off (`offload_register.py:1227-1228`), so the
"refuses LOUDLY at runner init too, including on the direct-`ServerArgs` path"
contract in its own docstring (`:1222-1225`) holds only when the dark-launch flag
is on — i.e. never by default, and never on the path the docstring names.
→ *Task title: "#428: parse --lane-offload-* park targets before the SGLANG_OFFLOAD_REGISTER gate so a typo refuses on the direct-ServerArgs path"*

**S3-44 — `link_disjointness` UNKNOWN is documented as a refusal and enforced
nowhere.** No production caller exists (`grep -rn "link_disjointness\|LinkVerdict"
python/ test/`), and with `link_path_complete=False` hardcoded in bootstrap
(`memtier/bootstrap.py:241`) and in every bundled remote row
(`memtier/profiles/rig1.json:165,195`), DISJOINT cannot be produced from real
data on this rig. #423's striping gate would inherit a predicate that always
answers UNKNOWN.
→ *Task title: "#423 prerequisite: make link_path_complete derivable from probe data, or state that link_disjointness is desk-only until it is"*

**S3-58 — kvso is not "decoupled from speculation"; it refuses to boot with it.**
`if self.speculative_algorithm is not None and (os.environ.get("KVSO_ALLOW_SPEC",
"0") != "1"): raise ValueError(...)` (`server_args.py:6580-6595`). NEXTN/MTP is
the standing production path on this rig, so the catalog line describes a
configuration that aborts at argument time unless an undocumented env is set.
→ *Task title: "kvso x speculation: either land the spill+MTP gate removal or correct the catalog and surface KVSO_ALLOW_SPEC in --enable-kv-session-offload's help"*

**S3-61 — "demote to HiCache" names a configuration the validator forbids.**
kvso and `--enable-hierarchical-cache` are mutually exclusive
(`server_args.py:6620`), so the demotion target in §3's kvso line is unreachable
as written.
→ *Task title: "kvso budgets: name the actual demotion destinations (--kv-session-offload-destinations) or lift the hicache mutual exclusion"*

**S3-67 — the GDN slot ladder is validated at boot and inert at runtime.**
`if not offload_register_enabled(): return []`
(`offload_gdn_states.py:344-345`) precedes the ladder attach, so
`--gdn-state-set-ladder` / `--gdn-resident-state-slots` accept values, range-check
them (`server_args.py:6841-6857`) and then do nothing without
`SGLANG_OFFLOAD_REGISTER=1`. A flag that validates and no-ops is worse than one
that refuses.
→ *Task title: "#286: refuse --gdn-state-set-ladder / --lane-offload-* at argument time when SGLANG_OFFLOAD_REGISTER is off, instead of validating and no-opping"*

**S3-49 — "weights+KV survive process exit" overstates the hibernate payload.**
The parked dict is params + buffers only (`hibernate.py:448-456`); no KV pool,
radix tree or host tier is written.
→ *Task title: "Correct the #89 hibernate catalog line to weights+buffers, and name the GGUF-only / pure-TP / NVML-UUID scope"*

---

#### NOT-FOUND

* **A gate restricting HiCache under uneven DCP/TP** — none exists (S3-62).
  Grepped: `hierarchical_cache|hicache` × `dcp|rank_tp_ratio|rank_kv_ratio|tp_partition|uneven`
  in `server_args.py` (0 hits); `uneven|dcp|rank_tp_ratio` in
  `mem_cache/hiradix_cache.py` (0 hits); `hierarchical` × `raise|refus` across
  `python/sglang/srt/` (0 hits); all 13 `enable_hierarchical_cache` sites in
  `server_args.py` read individually.
* **An "idle-first" victim term in kvso** — does not exist (S3-59). Grepped
  `idle` and `victim|_pick` across `managers/kv_session_offload.py`; the key is
  `(spill_class_rank(req), fast, -seq)` at `:844`.
* **A driver-version check or VMM support probe for `--enable-vram-dial`** —
  none. Grepped `vram_dial|#330` × `raise|refus|not support|incompatible` across
  `python/sglang/srt/`, plus the full body of `validate_vram_dial_compat`. The
  only capability test is `server_args.device != "cuda"` (`vram_dial.py:1046`).
* **A quant-format *allowlist* for the expert offload** — grepped
  `EXPERT_TENSOR_ATTRS`, `presplit_expert_offload_after_repack`,
  `_OFFLOAD_*_QUANT_METHOD_NAMES` across `python/sglang/srt/layers/`. Only the
  denylist (S3-11) and the one-entry conditional table exist; there is no
  positive enumeration anywhere.
* **An env or CLI override for `require_measured_bandwidth` /
  `allow_unmeasured_bandwidth`** — grepped both names plus `min_bandwidth_gbs`
  across `python/`. Three call sites, all Python kwargs; no operator-facing knob.

---

#### Catalog corrections

```
OLD: load-time-aware halves for fp8/GPTQ/AWQ (GGUF-MoE half missing — guarded).
NEW: load-time-aware halves for fp8/GPTQ/AWQ and, since #123-GGUF, GGUF-MoE on
     CUDA — admitted per layer on the staging marker
     (`_OFFLOAD_CONDITIONAL_QUANT_METHOD_NAMES`, `expert_offload.py:2069`;
     `if getattr(layer, marker, False): continue`, `:2111`). The guard is a
     five-name DENYLIST (Ascend GGUF-MoE, MoeWNA16, three NVFP4 methods,
     `:2056-2064`) matched by class NAME, so every other quant method —
     unquantized, INT8, compressed-tensors non-NVFP4 — passes by default.
```

```
OLD: Refuses by name at boot unless decode backend is `breakable` … and
     prefill is eager …
NEW: Refuses by name at boot unless the resolved decode backend is literally
     `breakable` and the resolved prefill backend is literally `disabled`
     (`if backend != "breakable":` / `if prefill is not None and prefill !=
     "disabled":`, `offload_capture_gate.py:363`, `:379`). Both checks are
     SKIPPED when the backend cannot be resolved (`if backend is None: return`,
     `:358`). On DeepSeek-V4 the prefill half is already satisfied: the
     BCG-incompatibility rule set rewrites prefill to `disabled`
     (`server_args.py:8281-8309`).
```

```
OLD: Eager path only: refused by name under `SGLANG_MOE_OFFLOAD_CUDA_GRAPH` and
     after `install_capturable_buffers()`, since a captured gather's LUTs pin the
     layout.
NEW: Refused under the CAPTURABLE route only — by name at boot
     (`if bool(envs.SGLANG_MOE_OFFLOAD_CUDA_GRAPH.get()):`,
     `expert_heat_migration.py:339`) and at migration time
     (`if self._capturable_ready:`, `expert_offload.py:3632`). It RUNS under the
     #462 breakable route: `prepare_breakable` calls `_observe_routing`
     (`expert_offload.py:3107`), whose tail is `if self._heat.due():
     self._migrate_heat()` (`:3197-3198`), and `_capturable_ready` is False there.
     Sound by construction (in-place swaps into address-stable arena slots,
     slot vector republished per replay); §17 cell "#302a x graphs" is occupied.
     Same for Stage-1 `SGLANG_MOE_HOT_RESIDENCY` (`layer.py:702` gates on the
     capturable mode only).
```

```
OLD: Since #452 that seam is behind a SECOND refusal … so reaching it now takes
     both overrides.
NEW: … so reaching it on the CAPTURABLE route takes both overrides. On the #462
     breakable route it takes neither: `refuse_capturable_cold_tier` is armed
     only inside `install_capturable_buffers` (`expert_offload.py:2936-2940`),
     which that route never calls, and its eager `_fetch` keeps the peer branch
     (`if remote is not None and expert_id in self._remote_ids:`, `:2819`).
```

```
OLD: `absent` unselectable
NEW: `absent` is unselectable as a MINIMUM
     (`SGLANG_MOE_HOST_SHARD_MIN_PROVENANCE`; `if raw not in ("measured",
     "estimate"): raise ValueError`, `expert_offload.py:800`), and is a hard
     refusal at the #439 door (`if ratio.provenance == "absent": raise
     NoComputeLever`, `expert_compute_placement.py:951`) and in
     `ColdTierAssignment.__post_init__` (`cold_tier_fetch.py:218`). At the
     cold-tier door it DEGRADES instead: an absent provenance yields an equal
     ratio and `if ratio.is_equal: return None` (`expert_offload.py:1158`) —
     the pre-#394 path, silently. Default minimum is `estimate`.
```

```
OLD: with it off the slice-1 boot refusal for delegation on disjoint expert
     shards is unchanged, field for field.
NEW: with it off the GGUF streaming door warns once and falls back to the
     pre-#394 plan (`if not cold_tier_enabled() and not
     envs.SGLANG_MOE_HOST_SHARD_UNSAFE_DELEGATE.get(): … return None`,
     `fused_moe_triton/layer.py:1450-1455`); the HARD refusal is the separate
     marlin-repack door (`refuse_cold_shard_at_repack_door`,
     `expert_offload.py:3755-3782`). Two doors, two refusal shapes.
```

```
OLD: Whole surface is behind `SGLANG_OFFLOAD_REGISTER=1` (dark launch).
NEW: The env gates the process-global register only (`if not
     offload_register_enabled(): return None`, `offload_register.py:1227`,
     `:1261`) — i.e. `get_global_register` and the `maybe_*` adapters. The
     `--lane-offload-*` parsers run at argument time regardless
     (`server_args.py:6663-6668`), and the #286 asset-class layer
     (`short_term_offload_register.py`) is not behind it at all: it is already
     called in production from `breakable_offload.py:216` / `:268-272`. Note the
     runner-init typo refusal (`offload_register.py:1239`) sits BEHIND the gate,
     so by default a typo refuses only at argument time.
```

```
OLD: provenance `measured|estimate|absent` (absent refuses use)
NEW: provenance `measured|estimate|absent`. ABSENT refuses on the BANDWIDTH axis
     only, and only under a floor or `allow_unmeasured_bandwidth=False`
     (`registry.py:407`, `:418`); absent LATENCY refuses nowhere; absent CAPACITY
     refuses only when bytes are requested (`:455`, `:458`). "Estimates refused
     by default" is the CALLER's default, not the registry's:
     `require_measured_bandwidth: bool = False` (`registry.py:199`) vs
     `price_park_target(..., require_measured: bool = True)`
     (`short_term_offload_register.py:787`).
```

```
OLD: `TierTransport.link_path` + `link_disjointness()` expose PATH identity for
     #423's striping gate, with `DISJOINT` requiring complete paths on both sides
     and `UNKNOWN` being a refusal.
NEW: … with `DISJOINT` requiring complete paths on both sides
     (`tiers.py:570-603`). `UNKNOWN` is RETURNED, not refused: there is no
     production caller and no `raise` (`grep -rn "link_disjointness|LinkVerdict"
     python/ test/`). With `link_path_complete=False` hardcoded in bootstrap
     (`memtier/bootstrap.py:241`) and in the bundled remote rows, DISJOINT is
     currently unreachable from real data.
```

```
OLD: applies a stored profile ONLY at the scope its hardware match licenses
NEW: … at the scope its hardware match licenses (`fingerprint.py:313` EXACT /
     `:325` MODEL / `:339` NONE; MODEL drops the whole `tiers` list,
     `:416-419`) — unless `SGLANG_MEMTIER_PROFILE_TRUST=1` promotes an explicit
     path's NONE verdict to full EXACT (`profile_store.py:208-213`).
```

```
OLD: **Hibernate to disk** (weights+KV survive process exit; uneven-TP3 reload
     50s→8-14s)
NEW: **Hibernate to disk** (weights + module buffers survive process exit — NO
     KV is parked: `payload = {... "params", "static_state", "gguf_attrs" ...}`,
     `model_loader/hibernate.py:448-456`; uneven-TP3 reload 50s→8-14s, the
     uneven vector being part of the identity hash, `:96-97`). Scope, all hard
     refusals: GGUF checkpoints only (`if self.load_format != "gguf": raise`,
     `server_args.py:13204`), pure single-node TP (`hibernate.py:106-123`), and
     a per-rank NVML-UUID recheck on restore (`if live_uuid !=
     rank_meta["nvml_uuid"]: raise RuntimeError`, `:568`).
     `--enable-weights-disk-backup` and `--hibernate-dir` require each other in
     both directions (`server_args.py:13191-13199`).
```

```
OLD: **KV session offload (kvso)**: FCFS spill of youngest sessions to RAM (KV
     only, GDN stays resident), budgets (volume/rate/window, demote to HiCache),
     idle-first victim choice, decoupled from speculation.
NEW: **KV session offload (kvso)**: FCFS spill of youngest sessions to RAM (KV
     only — `bundle_spillable_sizes` returns `[("kv", kv)]` and nothing else,
     `managers/kv_session_offload.py:126-127`), budgets (volume/rate/window),
     victim key `(spill_class_rank, fast_lane, -kv_arrival_seq)` — spill class,
     then non-fast-lane, then YOUNGEST arrival; there is no idleness term
     (`:844`). REFUSED with speculative decoding unless `KVSO_ALLOW_SPEC=1`
     (`if self.speculative_algorithm is not None and
     os.environ.get("KVSO_ALLOW_SPEC", "0") != "1": raise ValueError`,
     `server_args.py:6580`), and mutually exclusive with
     `--enable-hierarchical-cache` (`:6620`), PD disagg, `--weightless-kv-fastlane`,
     `--enable-unified-memory`, `--enable-hisparse`, `--enable-mixed-chunk`,
     `page_size > 1`, non-flashinfer backends, `pp_size > 1`, `dp_size > 1`
     (`:6596-6641`).
```

```
OLD: `--kv-pressure-ladder auto` mode wired via rig-profile bridge (#428), boot
     validation pending
NEW: `--kv-pressure-ladder auto` resolves to a real table at this tip — the
     planner step-table source is supplied in production (`ladder =
     build_ladder_from_server_args(server_args,
     table_fn=auto_ladder_table_fn(server_args))`,
     `managers/kv_pressure_runtime.py:467`), so AUDIT_421 F1 is closed; the
     `raise` at `model_executor/kv_pressure_ladder.py:1956` now only catches
     direct callers. On a heterogeneous node `auto` REQUIRES `--rank-gpu-id`:
     without a rank→card vector it refuses (`if len(names) > 1: raise
     ValueError`, `managers/kv_ladder_auto.py:190`). Boot validation still
     pending. Rung-dependency refusals apply to EXPLICIT specs only
     (`if isinstance(spec, tuple):`, `server_args.py:6955`).
```

```
OLD: **Runtime VRAM dial** per card (VMM page return), **KV pressure ladder** …
     **KV resharding** at phase boundaries (delta move <1 s,
     `kv_reshard_vectors`), **GDN slot ladder** (resident-state cap + idle
     vacate → VRAM back to KV pool).
NEW: **Runtime VRAM dial** per rank (VMM page return) — requires WEIGHTED uneven
     DCP (`if not uneven_dcp_active(dcp_size): raise KvCapacityError`,
     `managers/vram_dial.py:1110`), CUDA, a non-MLA VMM-backed pool, and refuses
     under 11 named combinations incl. memory-saver, PD disagg, hicache storage,
     kvso, dual-group, DP>1, kv-canary, hisparse, weightless-KV fastlane and the
     DFLASH lane (`:1045-1086`). **KV pressure ladder** … **KV resharding** at
     phase boundaries (`kv_reshard_vectors`) — also requires WEIGHTED uneven DCP
     and the hybrid-linear pool family only (`managers/kv_reshard.py:713-729`).
     **GDN slot ladder** (resident-state cap + idle vacate) — its flags validate
     at boot but are INERT without `SGLANG_OFFLOAD_REGISTER=1`
     (`if not offload_register_enabled(): return []`,
     `model_executor/offload_gdn_states.py:344`).
```

```
OLD: reaches the legacy hybrid-SWA `SWAKVPool` since upstream #32213 — before
     that it was silently a no-op there, while `UnifiedSWAKVPool` already
     honoured it
NEW: … since upstream #32213 (`kwargs.setdefault("enable_memory_saver", False)`,
     `mem_cache/swa_memory_pool.py:54`, with both call sites passing the server
     arg, `model_runner_kv_cache_mixin.py:3311`/`:3491`). `UnifiedSWAKVPool`
     honours it only INDIRECTLY: its own `enable_memory_saver` parameter
     (`mem_cache/unified_memory_pool.py:1014`) is never referenced; the saving
     happens in the shared `UnifiedKVPool` buffer (`:212-216`).
```


### §4 / §5 / §6 — Speculation, multi-group lane, weightless lane

Scope: FEATURE_CATALOG.md §4 (Speculative decoding), §5 (Multi-group runtime /
dual lane), §6 (Weightless KV lane). Read-only, static. Every row cites the
predicate at its source; comments/help text are NOT evidence.

Counts: WIDER 9 · NARROWER 8 (2 bug-candidates) · EXACT 21 · NOT-FOUND 2.

| ID | Catalog claim (short) | Class | Gate predicate (verbatim) | file:line | Note |
|---|---|---|---|---|---|
| S4-01 | "Tree-spec topk>1 under DCP is HARD-GATED" | **WIDER** | `if (\n    self.rank_tp_ratio is not None or self.weightless_kv_fastlane\n) and tree_reason is not None:` | `python/sglang/srt/server_args.py:7407` | Not "under DCP". Gate = (a **--rank-tp-ratio vector exists**) OR (weightless lane), reached only when `dcp_size > 1`. Plain even DCP with NO `--rank-tp-ratio` and no lane never fires it. No `page_size` term at all. Fires for a **uniform** ratio vector too (`[1,1]`), which is even-modulo DCP. |
| S4-02 | same, caller condition | EXACT | `if not self.dcp_size > 1:\n    return` | `server_args.py:7590` | Guard is unreachable at `dcp_size == 1`; `weightless_kv_fastlane` alone cannot trigger it (the lane forces `dcp_size == tp_size >= 2`, so in practice it always does). |
| S4-03 | the door list ("HARDENED, not read as topk>1") | EXACT | `topk = self.speculative_eagle_topk\nif topk is not None and topk > 1:\n    return f"--speculative-eagle-topk={topk} > 1 (EAGLE tree draft)"\nif getattr(self, "speculative_dflash_tree_verify", False):` | `server_args.py:7335-7338` | Second door armed for a flag that does not exist yet (`getattr` default False). |
| S4-04 | backend mirror of the same condition | EXACT | `self.uneven_dcp = (\n    uneven_dcp_kv_replicated(self.dcp_size) or self.weightless_kv\n) and not _draft_replicated` / `self.dcp_tree_mask = bool(\n    self.uneven_dcp\n    and getattr(_sa, "speculative_eagle_topk", None) is not None\n    and _sa.speculative_eagle_topk > 1\n)` | `layers/attention/flashinfer_backend.py:689-691`, `:931-935` | Backend adds a `not _draft_replicated` term the ServerArgs guard does not have → ServerArgs is the strict superset. Defensive `RuntimeError("#76 guard hole: …")` at `:947`. |
| S4-05 | the DCP-replication base condition | EXACT | `return dcp_size > 1 and get_tp_partition_ratios() is not None` | `distributed/utils.py:354` | `get_tp_partition_ratios()` reads a **context-local overlay first** (`_TP_PARTITION_OVERLAY`, the #274 lane plan) before the process plan (`utils.py:168-177`) — i.e. the runtime value is lane-scoped while the ServerArgs guard reads a static field. |
| S4-06 | spec+DCP admitted configs (CUDA) | NARROWER (doc-candidate) | `if (\n    self.speculative_algorithm is not None\n    and not uneven_weighted_dcp\n    and not self.weightless_kv_fastlane\n):` → raise | `server_args.py:7644-7659` | On CUDA, spec × DCP is refused unless (weighted uneven DCP with `len(set(rank_tp_ratio)) > 1`) or the weightless lane. On HIP the whole block is skipped (`if is_hip(): return`, `:7612`) — spec × even DCP × topk>1 is allowed there and runs the stock (correct) EAGLE tree path. |
| S4-07 | `FROZEN_KV_MTP` "stays refused" (draft-solo) | EXACT, **name-keyed** | `if algo.is_frozen_kv_mtp():` → raise | `server_args.py:7111` | `is_frozen_kv_mtp()` is `self == SpeculativeAlgorithm.FROZEN_KV_MTP` (`spec_info.py:134-135`) — pure enum identity, not a structural property. Any future in-place-target-KV drafter is admitted unless someone remembers to add its name. |
| S4-08 | "…and is pinned by a test" | EXACT | `with self.assertRaisesRegex(ValueError, "FROZEN_KV_MTP"):` | `test/registered/unit/server_args/test_draft_solo_args.py:168-170` | Test asserts the message, i.e. it pins the NAME-keyed refusal, not a property. |
| S4-09 | Draft-solo "admits the whole DFLASH FAMILY (#470)" | **WIDER + name-list** | `if not (algo.is_eagle() or algo.is_dflash_family()):` → raise | `server_args.py:7147` | Admission is enum membership: `is_eagle()` = {EAGLE, EAGLE3, **FROZEN_KV_MTP**} (`spec_info.py:125-129`), `is_dflash_family()` = `is_dflash() or is_dspark()` (`spec_info.py:143-144`). So EAGLE3 is admitted (catalog names only EAGLE/EAGLE3/NEXTN/DFLASH/DSPARK — consistent), and any **plugin** algorithm registered via `SpeculativeAlgorithm.register` returns a `CustomSpecAlgo` that is neither → silently refused, whatever its shape. |
| S4-10 | solo is "pure single-node TP" | **WIDER** | `if self.nnodes > 1 and self.speculative_draft_gpu is not None:` → raise | `server_args.py:7239` | The blanket `nnodes > 1` refusal is **gone**. Multi-node solo is admitted; only `--speculative-draft-gpu` (a per-node device index) is refused across hosts. DP/PP/EP still refused (`:7196`, `:7203`, `:7209`). Marked RELAXED-NOT-PROVEN in the source. |
| S4-11 | solo (implicit: any TP) | EXACT | `if self.tp_size < 2:` → raise; `if self.disaggregation_mode != "null":` → raise | `server_args.py:7282`, `:7275` | Solo needs TP>=2 and refuses PD disaggregation. Neither is in the catalog. |
| S4-12 | Solo DSpark v1 "greedy-acceptance-only" | EXACT (runtime, not boot) | `if sampling_info is None or sampling_info.is_all_greedy:\n    return\nraise ValueError(...)` | `speculative/dspark_components/dspark_solo.py:405-413` | Per-ROUND check (called from `dspark_worker_v2.py:677`), so a non-greedy request fails mid-serving, not at boot. `sampling_info is None` passes through. |
| S4-13 | "switches `SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD` off, with reasons logged" | EXACT | `if not getattr(draft_model, "_opt_markov_w2_tp_shard", False):\n    return\ndraft_model._opt_markov_w2_tp_shard = False` | `dspark_solo.py:431-433` | Disabled, not refused; warning from rank 0 only (`:438`). |
| S4-14 | `--speculative-moe-runner-backend` "now actually reaches DFLASH/DSPARK draft builds" | **WIDER (catalog silent)** | `with speculative_moe_backend_context():` | `speculative/draft_worker_common.py:155` (DFLASH/DSPARK), `speculative/eagle_worker_v2.py:349,487,503,1850,1915,2020,2062,2089,2163`, `standalone_worker_v2.py:140,146`, `frozen_kv_mtp_worker_v2.py`, `multi_layer_eagle_worker_v2.py:186,235,242,884`, `cross_algo_worker.py:811,1569,1744` | It reaches **every** draft-building worker family, not just DFLASH/DSPARK. The context manager itself is unconditional: `moe.runner_backend = get_speculative_moe_runner_backend()` (`layers/moe/utils.py:546`), and the value defaults to `moe_runner_backend` when unset (`arg_groups/overrides.py:2086-2087`). |
| S4-15 | ditto, per-rank sm90/sm120 refusal | EXACT | `if not backend.is_marlin():\n    return\n…\nif is_sm90_supported() or is_sm120_supported():\n    return` | `draft_worker_common.py:80-85` | Deliberately per-rank; a solo SHADOW builds on `meta` and never reaches it. Only fires for marlin backends. |
| S4-16 | per-decode reserve is `W + L` | EXACT | `write_footprint = (\n    get_alloc_len_per_decode(server_args) if alloc_len is None else alloc_len\n)\ncommit_lag = get_commit_lag_per_decode(server_args)\nreturn write_footprint + commit_lag` | `mem_cache/common.py:375-379` | |
| S4-17 | `W = get_alloc_len_per_decode` | EXACT | `if page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():\n    return max(spec_steps * spec_topk, spec_tokens)` | `mem_cache/common.py:278-287` | NGRAM escapes the page-tree branch via `has_draft_kv()` (`spec_info.py:161-165`). |
| S4-18 | "`L` = 0 with `--disable-overlap-schedule`" | EXACT | `if server_args.disable_overlap_schedule:\n    # No overlap: …\n    return 0` | `mem_cache/common.py:308-311` | Yes, exactly 0, and it is the FIRST branch — it short-circuits before the no-spec (`return 1`) and spec (`return max_speculative_num_draft_tokens`) branches. |
| S4-19 | "unifies the DFLASH solo lane's own hardcoded `2 x block_size`" | EXACT | `reserve = get_alloc_reserve_per_decode(alloc_len=block_size)` | `speculative/dflash_info_v2.py:154` | |
| S4-20 | "It is now a NAMED posten in the pool ledger … instead of an uncounted transient" | **NARROWER (bug-candidate)** | `decode_alloc = 2 * get_alloc_len_per_decode(sa)` | `model_executor/pool_configurator.py:628` | The hybrid-SWA / SWA-chunk-cap pool sizing still uses the **old blanket `2 x W`**, not `get_alloc_reserve_per_decode`. Under non-overlap it over-reserves by the whole of W per request; under topk>1/page>1 it under- or over-shoots relative to the allocator's actual `W + L`. Two derivations of the same quantity now coexist. |
| S4-21 | "Both directions are pinned" | EXACT | `test_shaved_reserve_is_caught`, `test_dropping_the_commit_lag_term_under_reserves` | `test/registered/spec/test_alloc_reserve_need.py:210,239` | Also pins the widest-rung ceiling (`:150`). |
| S4-22 | "spec-algo name validation (one source, parse-time refusal)" | EXACT | `try:\n    SpeculativeAlgorithm.from_string(name)\nexcept ValueError:\n    known = SpeculativeAlgorithm.known_names()\n    raise ValueError(...)` | `server_args.py:6759-6771` | Single source is `known_names()` = enum members ∪ `registered_names()` ∪ `SPECULATIVE_ALGORITHM_ALIASES` (`spec_info.py:80-85`). Aliases bypass `from_string` earlier (`server_args.py:6750-6757`). |
| S4-23 | "acceptance-driven DFLASH<->NEXTN switch + adaptive k" | **NARROWER (bug-candidate)** | `if server_args.speculative_draft_gpu is not None:\n    _fail("owns the solo rank (rank 0); leave --speculative-draft-gpu unset.")` | `speculative/cross_algo_utils.py:738-739` (see also `:733`) | The cross-algo ladder **hardcodes the DFLASH rung's solo host to rank 0** and refuses `--speculative-draft-gpu`. On a heterogeneous rig the big card is not necessarily rank 0, so the catalog's "DFLASH solo draft on the big card" does NOT hold in the cross-algo configuration. |
| S4-24 | cross-algo scope | NARROWER (doc-candidate) | `if server_args.speculative_algorithm != "EAGLE":\n    _fail("requires --speculative-algorithm NEXTN (or EAGLE) as the MTP rung; …")` | `cross_algo_utils.py:686-690` | DFLASH only — **DSPARK is not a cross-algo rung** even though §4 says "DSPARK joins DFLASH". Also refuses PP/DP/EP/nnodes>1/PD/multi-layer-eagle/rejection-sampling/topk>1/draft-window (`:697-732`). |
| S4-25 | "#491: fused-KV-projection support probe answers `False` … instead of raising" | EXACT | `quant_method = getattr(qkv_proj, "quant_method", None)\nif not isinstance(quant_method, UnquantizedLinearMethod):\n    return (\n        False,\n        "quantized qkv_proj is not supported for this path "\n        f"(quant_method={type(quant_method).__name__})",\n    )` | `speculative/dflash_utils.py:525-531` | Predicate is `isinstance(..., UnquantizedLinearMethod)` — a **structural** check that covers marlin/AWQ/GPTQ *and every other* quant method, not a name list. Consumed at `dflash_worker_v2.py:872`. |
| S4-26 | "the draft's `.scale` rename is suffix-anchored" | EXACT | `if rest.endswith(".scale"):\n    rest = rest[: -len(".scale")] + ".weight_scale_inv"` | `models/deepseek_v4_dspark.py:896-897` | `.scales` / `.scale_inv` no longer match. |
| S4-27 | "NEXTN/MTP standard (steps 3, topk 1, draft 4)" | NOT-FOUND as a gate | — | — | Descriptive recipe, not a predicate. What IS enforced: DFLASH/DSPARK **override** `num_steps` and `topk` to 1 with a warning (`arg_groups/speculative_hook.py:217-233`, `:363-379`); DSpark additionally forces `num_draft_tokens == gamma + 1` (`:413-421`). |
| S4-28 | multi-layer EAGLE | EXACT | `if not _algo_is_dflash and self.enable_multi_layer_eagle:` → raise | `server_args.py:7159` | Multi-layer EAGLE is refused under solo placement, and under `--draft-kv-layout dcp` (`:7513`). |
| S5-01 | dual lane (no stated hardware precondition) | **NARROWER (doc-candidate)** | `if not isinstance(self.rank_tp_ratio, list):` → raise | `server_args.py:9441-9448` | `--dual-group-lane` **requires an explicit `--rank-tp-ratio` integer list**; `'auto'` is refused by name. A **uniform** list (`1,1`) satisfies it, so the lane IS reachable on a homogeneous rig — but never without the flag. Catalog §5 says nothing about this. |
| S5-02 | lane budget | EXACT | `if not self.dual_group_lane_budget_mib:` → raise | `server_args.py:9449-9454` | No fallback to `--mem-fraction-static`. |
| S5-03 | lane is "single-node pure TP" | **WIDER** | `if self.pp_size > 1 or self.enable_dp_attention:` → raise | `server_args.py:9455-9459` | Only PP and DP-**attention** are refused. `dp_size > 1` without `--enable-dp-attention`, and **`ep_size > 1`**, are NOT refused — expert parallelism is reachable with the dual-group lane. (Contrast weightless lane `:6069`, which refuses `ep_size > 1` explicitly.) |
| S5-04 | "chain-spec topk=1 on the lane" | **WIDER** | `if self.speculative_eagle_topk not in (None, 1):` → raise | `server_args.py:9482-9487` | The predicate reads the **SERVING group's** `speculative_eagle_topk`, not a lane-local value. So enabling `--dual-group-lane-spec` also forbids a topk>1 tree on the *main* serving group — broader than "topk=1 on the lane". The lane's own proposer is unconditionally a chain (`model_executor/dual_group_lane.py:2488-2493`, `_propose`, no topk parameter at all). |
| S5-05 | lane NEXTN head | EXACT | `if self.speculative_algorithm is None:` → raise | `server_args.py:9473-9481` | Lane spec requires the serving group to speculate (head shards are nested). |
| S5-06 | lane adaptive ladder | EXACT | `if rungs is None and self.dual_group_lane_spec_adaptive:` → raise; `if len(rungs) < 2:` → raise | `server_args.py:9505-9511`, `:9499-9504` | |
| S5-07 | "SM-contention pairing rule" | EXACT | `if self.dual_group_lane_pairing and not self.dual_group_lane_concurrent:` → raise | `server_args.py:9462-9468` | Plus orphan-flag refusals for spec/budget/part-gpu-id/pairing without the lane (`:9518-9533`). |
| S5-08 | "Marlin LoRA workspace keyed (lane,name)" | EXACT | routed through the lane-keyed buffer accessor | `lora/lora_moe_runner_marlin.py:41-49` | Docstring states the mechanism; the accessor is the lane-scoped buffer helper. Not a refusal, no gate. |
| S5-09 | PD disagg: prefill satellite "default graph-covered" | **NARROWER (doc-candidate)** | `if self.disaggregation_mode == "prefill":\n    if (Phase.DECODE, "backend") not in self._cuda_graph_config_locked:\n        self.cuda_graph_config.decode.backend = Backend.DISABLED` | `server_args.py:8189-8192` | Confirmed: on a prefill satellite only the DECODE graph is disabled; PREFILL keeps its default backend (**breakable**, not eager). BUT the breakable-compat sweep then disables the prefill graph for `("context parallel (attn_cp_size > 1)", lambda: self._resolved().attn_cp_size > 1)` and `("MLA attention", …)` (`server_args.py:8278-8300`) — i.e. **any DCP plan drops the prefill graph**, which is exactly the geometry this fork exists for. "Default graph-covered" holds only off the DCP path. |
| S5-10 | PD prefill satellite "carries hybrid GDN (KV+mamba slot via mooncake)" | EXACT | `if req.req_pool_idx is not None or self.tree_cache.supports_mamba():` | `disaggregation/prefill.py:911` (payload built at `:1059-1061`, `:1125`) | Same `supports_mamba()` capability the handover gate uses. |
| S6-01 | `POST /session_handover` MERGED | EXACT — **present at this tip** | `@app.api_route("/session_handover", methods=["POST"])` | `entrypoints/http_server.py:1126` (handler `:1128`, dispatch `:1138`) | AUDIT_421's "string does not occur in python/" is stale: it now occurs in `http_server.py`, `managers/tokenizer_control_mixin.py`, `managers/scheduler.py`, `mem_cache/hicache_migrate.py`. Auth: `@auth_level(AuthLevel.ADMIN_OPTIONAL)`. |
| S6-02 | "hard GDN-blob gate keyed on `BasePrefixCache.supports_mamba()`" | EXACT | `hybrid_gdn = bool(tree.supports_mamba())\nmamba_key = f"{kv_keys[-1]}.mamba" if hybrid_gdn else None` | `managers/session_handover.py:556-557`; enforcement `if manifest["hybrid_gdn"]:` → refuse on missing/absent mamba key at `:231-244` | Capability-keyed (structural), not class-sniffed — the comment at `:548-555` names the exact failure it replaces. |
| S6-03 | "`page_size == 1`, inherited from `dcp_owner_mode`" | EXACT | `if args.page_size != 1:\n    raise SessionHandoverError(\n        f"page_size == 1 is required (got {args.page_size}); the "\n        "umsharder inherits this limit from dcp_owner_mode"\n    )` | `managers/session_handover.py:425-429` | Checked on the **source/export** side only. `dcp_owner_mode` itself is only recorded into the manifest (`:579`), never compared. |
| S6-04 | "a booted TP>1 **destination** still needs the offline umsharder" | **NARROWER (doc-candidate)** | `if args.tp_size != 1 or args.pp_size != 1:` → raise | `managers/session_handover.py:418-424` | The only `tp_size` predicate in the feature is on the **SOURCE**: live export requires TP=1, PP=1. The **destination** has NO tp_size check at all — `verify_import` is "presence + identity only" (`:266-292`), and a TP>1 destination fails indirectly as "N manifest blob(s) absent from this rank's store … run the manifest-scoped umsharder for this geometry first" (`:288-292`). So the catalog understates the limit on one side (source must be TP=1) and overstates it as an enforced predicate on the other. |
| S6-05 | (catalog silent) storage backend | **NARROWER (doc-candidate)** | `if self.scheduler.server_args.hicache_storage_backend != "file":` → raise | `managers/session_handover.py:392-396` | Also requires `tree.enable_storage` (`:387`). Handover works with the `file` backend only. |
| S6-06 | "draft re-sharder as its own spec type" | EXACT, name-keyed **with a completeness audit** | `verdict = DRAFT_RESHARD_CAPABILITIES.get(canonical)\nif verdict is None:\n    … DraftReshardCapability.REFUSE` | `mem_cache/draft_migrate.py:190-199`; table `:78-125`; audit `:142-153` | Only **EAGLE** is `RESHARD`; NGRAM/NONE are `NO_DRAFT_KV`; EAGLE3, STANDALONE, DFLASH, DSPARK, FROZEN_KV_MTP are all `REFUSE`. The `audit_capability_names()` check forces every enum member to carry a row, so this name list cannot silently drift — the right shape for a name-keyed gate. |
| S6-07 | §6 lists "chunked prefill/extend … host-tier KV spill, chain spec" | **NARROWER (doc-candidate)** | `if self.weightless_kv_chunked_block_size:` → raise | `server_args.py:6330-6345` | Weightless **spec and the streaming block loop / host spill are mutually exclusive**: two capture axes nobody composed. The §6 one-liner reads as if all six features compose. |
| S6-08 | weightless "chain spec" admission | EXACT, name-keyed | `if algo.is_frozen_kv_mtp() or not algo.is_eagle():` → raise | `server_args.py:6274-6287` | EAGLE-family only; DFLASH/DSPARK/NGRAM/STANDALONE and all plugin algos refused. |
| S6-09 | weightless spec: placement | EXACT | `if self.speculative_draft_placement != "solo":` → raise; `if solo_rank != self.weightless_kv_head_rank:` → raise | `server_args.py:6288-6302`, `:6304-6315` | "THE load-bearing condition" per the source. |
| S6-10 | weightless spec: adaptive | EXACT | `if self.speculative_adaptive:` → raise | `server_args.py:6316-6329` | One captured verify shape per boot. |
| S6-11 | weightless lane: topology | EXACT | `if self.pp_size > 1 or self.enable_dp_attention or self.ep_size > 1:` → raise; `if self.dcp_size != self.tp_size:` → raise; `if self.tp_size < 2:` → raise | `server_args.py:6069`, `:6083`, `:6078` | Stricter than the dual-group lane (S5-03), which permits EP. |
| S6-12 | weightless lane: topk>1 | EXACT (independent site) | `if self.speculative_eagle_topk is not None and self.speculative_eagle_topk > 1:` → raise | `server_args.py:6104-6112` | Fires even without an algorithm, deliberately redundant with S4-01. |
| S6-13 | "118-name retired-env guard that refuses stale SGLANG_* variables loudly" | EXACT | `found = sorted(\n    n for n in env if n.startswith(RETIRED_PREFIX) or n in RETIRED_ENV_VARS\n)` | `distributed/device_communicators/barlink_env_guard.py:188-190`; table `:25`; import-time call `:204` | Table has exactly 118 entries. Reach is WIDER than "118 names": the `RETIRED_PREFIX` arm catches any name with the retired prefix, table or not. |
| S6-14 | "the hibernate flag contract / regime-controller gate machinery" | NOT-FOUND in scope | — | — | Both are wired (regime-controller mode validation at `server_args.py:6784-6799`), but they are §12/§13 machinery; not audited here beyond confirming the validator exists. |

---

#### WIDER finds — what they unlock

**S4-01 / S4-06 — tree spec is not gated "under DCP".** The guard predicate is
`rank_tp_ratio is not None or weightless_kv_fastlane` (`server_args.py:7407`),
reached only at `dcp_size > 1`. A DCP run with **no `--rank-tp-ratio` vector and
no weightless lane** never touches it, `uneven_dcp` is False, `dcp_tree_mask` is
False, and `--speculative-eagle-topk > 1` runs the **stock, correct** single-wrapper
EAGLE tree path. On HIP that combination is live today (the CUDA-only spec×DCP
refusal at `:7644` sits behind `if is_hip(): return`). Practical consequence:
"topk>1 is unavailable whenever DCP is on" is false and must not be used to
exclude tree-spec candidates from a solver. Conversely the gate is *broader* than
"uneven": a **uniform** `--rank-tp-ratio` (`1,1`) also trips it. And there is no
`page_size` term anywhere — page>1 neither arms nor disarms the gate.

**S4-09 — draft-solo admission is `is_eagle() or is_dflash_family()`.** That is
enum identity, not shape. It admits EAGLE, EAGLE3, NEXTN (alias→EAGLE) and
DFLASH+DSPARK, and it refuses **every plugin algorithm** registered through
`SpeculativeAlgorithm.register` regardless of whether it has exactly the
DFLASH shape the catalog names (self-drafting block model, token-id round output,
post-all-reduce hidden input). A future block drafter is refused by omission,
which is the #443/#446 name-list family again.

**S4-10 — solo placement is no longer single-node.** The blanket `nnodes > 1`
refusal was replaced by a narrow one on `--speculative-draft-gpu`
(`server_args.py:7239`). Solo draft over a TP group spanning two hosts is
admitted today — relevant to the Nordstern TP=5 ladder, where solo placement was
previously assumed out of reach. Source flags it RELAXED-NOT-PROVEN (no
multi-node solo boot yet), so it is a candidate to smoke-test, not a claim.

**S4-14 — `--speculative-moe-runner-backend` reaches every draft family.** The
threading is a context manager applied in EAGLE/EAGLE3/NEXTN, STANDALONE,
FROZEN_KV_MTP, multi-layer-EAGLE and cross-algo workers as well as the
DFLASH/DSPARK shared builder. A per-lane MoE runner backend for a NEXTN draft is
therefore already available; nothing needs building for that axis.

**S5-03 — the dual-group lane permits expert parallelism.** Only `pp_size > 1`
and `enable_dp_attention` are refused (`server_args.py:9455`). `ep_size > 1` and
plain `dp_size > 1` pass. That is a genuinely reachable combination the catalog's
"single-node pure TP" shorthand hides, and it is the only lane variant that could
carry an MoE tenant with EP dispatch.

**S5-04 — the lane's topk gate is group-wide.** `speculative_eagle_topk not in
(None, 1)` reads the serving group's value, so `--dual-group-lane-spec` costs the
main group its tree option too. The lane's own proposer never had a topk knob
(`dual_group_lane.py:2488`), so the flag is doing double duty.

**S6-13 — the retired-env guard is prefix-based, not just a 118-name table.**
`n.startswith(RETIRED_PREFIX)` catches names built at runtime and names a launch
script grew after the table was written.

---

#### NARROWER — bug candidates (real user harm)

**S4-20 — the #486 reserve derivation was not applied to the SWA/hybrid pool
sizer.** `pool_configurator.py:628` still computes `decode_alloc = 2 *
get_alloc_len_per_decode(sa)` while the allocator reserves `W + L`. Two
independent derivations of the same quantity now disagree on every non-overlap
run (sizer over-reserves by the whole of W per request) and on every topk>1 /
page>1 tree. On a hybrid-SWA model with `--disable-overlap-schedule` this silently
spends SWA pool capacity that the allocator never asks for.
→ *Task title:* "Route SWA/hybrid pool decode_alloc through get_alloc_reserve_per_decode (#486 follow-up)".

**S4-23 — the cross-algo ladder pins the DFLASH solo rung to rank 0 and refuses
`--speculative-draft-gpu`** (`cross_algo_utils.py:733-739`). On a heterogeneous
rig where rank 0 is not the big card, the DFLASH rung's unsharded draft lands on
the *small* card, which is the opposite of the placement §4 advertises ("DFLASH
solo draft on the big card, vocab broadcast reclaims ~5 GB"). The refusal is
phrased as "owns the solo rank", i.e. deliberate, but it makes the cross-algo
bandit unusable in the configuration the fork exists for.
→ *Task title:* "Let --speculative-cross-algorithm honour --speculative-draft-gpu for the DFLASH rung".

Doc-candidates (no code change needed, catalog wording is wrong): S4-06, S4-24,
S5-01, S5-09, S6-04, S6-05, S6-07.

---

#### NOT-FOUND

- **S4-27**: no predicate enforces "steps 3, topk 1, draft 4" for NEXTN. Grepped
  `speculative_num_steps`, `speculative_eagle_topk`, `speculative_num_draft_tokens`
  across `arg_groups/speculative_hook.py`, `server_args.py` and
  `speculative/adaptive_spec_params.py`. What exists is the opposite direction:
  DFLASH/DSPARK *force* steps=1 / topk=1 with a warning. The NEXTN triple is a
  runbook recipe, not a gate.
- **S4-19 / §4 "vocab broadcast reclaims ~5 GB"**: no predicate — a measurement
  claim. Grepped `vocab`, `broadcast` in `speculative/dflash_solo_pool.py`,
  `dspark_solo.py`, `eagle_worker_v2.py`; found the broadcast mechanism
  (`get_tp_group().broadcast(payload, src=solo_rank)`) but no reclaim accounting
  reachable from a gate.
- **S6-14**: hibernate flag contract + regime-controller gate machinery are
  present but out of the §4/§5/§6 predicate surface; only the mode validator
  (`server_args.py:6794-6799`) was read.

---

#### Catalog corrections

**§4, tree-spec line**

```
OLD: Tree-spec topk>1 under DCP is HARD-GATED (silently wrong + perf-negative — do
     not re-attempt without new evidence; see rejected register).
NEW: Tree-spec topk>1 is HARD-GATED on the CROSS-RANK DCP VARIANTS ONLY — the gate
     is `(rank_tp_ratio is not None or weightless_kv_fastlane) and
     tree_verify_activation_reason() is not None`, reached only at dcp_size > 1
     (server_args.py:7407, mirroring flashinfer_backend.py:931 dcp_tree_mask).
     It fires for a UNIFORM --rank-tp-ratio too, and has no page_size term. DCP
     without a --rank-tp-ratio vector and without the weightless lane is NOT gated:
     there uneven_dcp is False and topk>1 runs the stock correct EAGLE tree path
     (live on HIP; on CUDA the separate spec×DCP gate at server_args.py:7644 shuts
     that door for its own reasons). Second door armed for --speculative-dflash-
     tree-verify before it exists (server_args.py:7338).
```

**§4, draft-solo family line**

```
OLD: Draft-solo placement now admits the whole DFLASH FAMILY (#470): DSPARK joins
     DFLASH because it has the same shape …
NEW: Draft-solo placement admits `algo.is_eagle() or algo.is_dflash_family()`
     (server_args.py:7147) — an ENUM MEMBERSHIP test, not a shape test: EAGLE /
     EAGLE3 / NEXTN / DFLASH / DSPARK. A plugin algorithm registered via
     SpeculativeAlgorithm.register is refused however DFLASH-shaped it is.
     FROZEN_KV_MTP is refused by name one branch earlier (server_args.py:7111,
     pinned by test_draft_solo_args.py:168). Solo also needs tp_size >= 2
     (:7282), refuses PD disaggregation (:7275) and DP/PP/EP (:7196/:7203/:7209),
     but is NO LONGER single-node: only --speculative-draft-gpu is refused across
     hosts (:7239, RELAXED-NOT-PROVEN).
```

**§4, cross-algo line**

```
OLD: acceptance-driven DFLASH<->NEXTN switch + adaptive k; DFLASH solo draft on the
     big card (vocab broadcast reclaims ~5 GB)
NEW: acceptance-driven DFLASH<->NEXTN switch + adaptive k (DFLASH only — DSPARK is
     not a cross-algo rung, cross_algo_utils.py:686) — and in that mode the DFLASH
     rung's solo host is PINNED TO RANK 0 with --speculative-draft-gpu refused
     (cross_algo_utils.py:733-739), so "solo draft on the big card" holds for plain
     --speculative-draft-placement solo, not for the cross-algo ladder.
```

**§4, moe-runner line**

```
OLD: `--speculative-moe-runner-backend` (the existing per-draft flag) now actually
     reaches DFLASH/DSPARK draft builds
NEW: `--speculative-moe-runner-backend` reaches EVERY draft build — the
     speculative_moe_backend_context() wrapper is applied in eagle_worker_v2,
     standalone_worker_v2, frozen_kv_mtp_worker_v2, multi_layer_eagle_worker_v2,
     cross_algo_worker and (new for #470) the shared DFLASH/DSPARK builder
     draft_worker_common.py:155. Unset it defaults to --moe-runner-backend
     (overrides.py:2086).
```

**§4, reserve line (append)**

```
OLD: It is now a NAMED posten in the pool ledger (`DESIGN_330_vram_dial.md` §3b)
     instead of an uncounted transient.
NEW: It is now a NAMED posten in the pool ledger (`DESIGN_330_vram_dial.md` §3b)
     instead of an uncounted transient — EXCEPT the hybrid-SWA / SWA-chunk-cap
     pool sizer, which still computes `2 * get_alloc_len_per_decode(sa)`
     (pool_configurator.py:628) and therefore disagrees with the allocator on
     every non-overlap and every topk>1/page>1 run. Open.
```

**§5, first line**

```
OLD: Slices A-D merged: lane-correct context overlays (~370 callsites), own thread +
     high-priority stream, lend/reclaim in ms, SM-contention pairing rule,
     lane-NEXTN head.
NEW: Slices A-D merged: lane-correct context overlays (~370 callsites), own thread +
     high-priority stream, lend/reclaim in ms, SM-contention pairing rule
     (--dual-group-lane-pairing needs --dual-group-lane-concurrent,
     server_args.py:9462), lane-NEXTN head. Admission: an EXPLICIT --rank-tp-ratio
     integer list is mandatory ('auto' refused, server_args.py:9441) — a uniform
     vector is accepted, so the lane is reachable on a homogeneous rig — plus
     --dual-group-lane-budget-mib (:9449). Only pp_size>1 and --enable-dp-attention
     are refused (:9455); ep_size>1 and plain dp_size>1 are NOT.
```

**§5, lane spec line**

```
OLD: chain-spec topk=1 on the lane
NEW: chain-spec topk=1 — the gate reads the SERVING GROUP's --speculative-eagle-topk
     (`not in (None, 1)`, server_args.py:9482), so enabling --dual-group-lane-spec
     also forbids a tree on the main group; the lane's own proposer has no topk knob
     at all (dual_group_lane.py:2488). --dual-group-lane-spec additionally requires
     the serving group to speculate (:9473).
```

**§5, PD line**

```
OLD: PD disaggregation: prefill satellite carries hybrid GDN (KV+mamba slot via
     mooncake), default graph-covered.
NEW: PD disaggregation: prefill satellite carries hybrid GDN (KV+mamba slot via
     mooncake, gated on tree_cache.supports_mamba(), disaggregation/prefill.py:911).
     Graph coverage: a prefill satellite disables only the DECODE graph
     (server_args.py:8190) and keeps the default BREAKABLE prefill graph — but the
     breakable-compat sweep drops that prefill graph for attn_cp_size > 1 and for
     MLA attention (server_args.py:8278-8288), i.e. every DCP plan runs the prefill
     satellite eager.
```

**§6, handover line**

```
OLD: … the declared v1 limit stands unchanged: a booted TP>1 destination still needs
     the offline manifest-scoped umsharder (`page_size == 1`, inherited from
     `dcp_owner_mode`) to reshape into its geometry first …
NEW: … v1 limits, as enforced: live EXPORT requires a TP=1 / PP=1 SOURCE
     (session_handover.py:418) and page_size == 1 (:425, "inherited from
     dcp_owner_mode"), and the 'file' hicache storage backend (:392). The
     DESTINATION carries NO tp_size predicate — verify_import checks manifest
     version, model identity and blob presence only (:266-292); a TP>1 destination
     simply misses the blobs and is told to run the manifest-scoped umsharder for
     its geometry first. Endpoint present at this tip: http_server.py:1126.
```

**§6, feature list line**

```
OLD: A card holds ONLY KV + attention (no weights): chunked prefill/extend, fp8/int4
     worker KV, DCP comm fusion, graph-captured streaming decode, host-tier KV
     spill, chain spec.
NEW: A card holds ONLY KV + attention (no weights): chunked prefill/extend, fp8/int4
     worker KV, DCP comm fusion, graph-captured streaming decode, host-tier KV
     spill, chain spec — but SPEC AND THE STREAMING BLOCK LOOP DO NOT COMBINE:
     --weightless-kv-chunked-block-size / --weightless-kv-host-spill-tokens are
     refused together with a speculative algorithm (server_args.py:6330), two
     capture axes nobody composed. Chain spec further requires the EAGLE family
     (:6274), --speculative-draft-placement solo (:6288) with solo rank ==
     --weightless-kv-head-rank (:6304), and no --speculative-adaptive (:6316).
     Lane topology: dcp_size == tp_size >= 2, no PP/DP-attn/EP (:6069-6088).
```


### §7 / §8 / §9 — Collectives, GGUF, quant lanes

Scope: FEATURE_CATALOG.md §7 (Collectives / transport), §8 (GGUF stack),
§9 (Quant lanes). Every row's predicate was read at source; docstrings and
help text were never accepted as the gate. Items the orchestrator already
verified EXACTLY (#438a four-condition notice, its env pair, `_is_fp8_weight_quant`,
the abort-gate knob names, the 118-entry `RETIRED_ENV_VARS` + `RETIRED_PREFIX`)
are carried forward as EXACT without re-derivation.

| ID | Catalog claim (short) | Class | Gate predicate (verbatim) | file:line | Note |
|----|----------------------|-------|---------------------------|-----------|------|
| S7-01 | barlink is the collective transport "wherever the combination supports it" | EXACT | `if envs.SGLANG_BARLINK.get() and self.world_size > 1:` | `python/sglang/srt/distributed/parallel_state.py:687` | Two conditions only — no model, quant, TP/DCP/PP/EP or vendor term. barlink is admitted for EVERY group with world_size>1 once the env is on. |
| S7-02 | which collectives barlink genuinely REFUSES | EXACT | `raise NotImplementedError(f"barlink does not implement {op!r}. ...")` reached from `if self.barlink_comm is not None: self._barlink_unsupported("all_gatherv")` | `parallel_state.py:1348-1371`; call sites `:1499`, `:1515`, `:1667`, `:1746` | Exactly four refusals: `reduce_scatter(output, input_list)`, `reduce_scatterv`, `all_gather(output_tensor_list=...)`, `all_gatherv`. Everything else (all_reduce, all_gather-into-tensor, reduce_scatter_tensor, all_to_all_single incl. the v form, broadcast) is admitted. |
| S7-03 | `--collective-net-small/-bulk` "per message class with typo hard-reject" | NARROWER (DOC) | `if entry.rsplit(":", 1)[0] not in known: unknown.append(entry)` … `raise ValueError(f"{flag}={value!r} names device(s) this host does not have: ...")` | `server_args.py:14089-14098`; accepted set built at `:14053-14064` (`for root in ("/sys/class/infiniband", "/sys/class/net")`), `"all"` wildcard at `:14087` | These flags select a NIC, not a transport. **No transport is selectable per class and BAR1 is not selectable there at all.** The reject is structural (sysfs membership), not a name list. Reach is further narrowed in code itself: `--collective-net-small` reaches only the barlink UCX plane (warning at `:14136` when `SGLANG_BARLINK_TRANSPORT != "ucx"`), and `:14176-14185` logs that LARGE TP collectives cannot be routed separately. |
| S7-04 | "graph-capable direct mode" | EXACT (but name-keyed) | `if graph_enable_set(): return CAPTURABLE_BARLINK_TRANSPORTS \| GRAPH_ENABLE_TRANSPORTS` over `CAPTURABLE_BARLINK_TRANSPORTS = frozenset({"device", "host"})` / `GRAPH_ENABLE_TRANSPORTS = frozenset({"bar1", "matrix"})` | `parallel_state.py:352-362`, `:298`, `:303` | **Transport-NAME-keyed, not property-keyed.** A fifth transport is capture-unsafe by default no matter what its data path does. The in-capture question is separately property-keyed: `return bool(torch.cuda.is_current_stream_capturing())` at `barlink.py:437`. Release switch `SGLANG_BARLINK_GRAPH_ENABLE` default ON: `os.environ.get(_GRAPH_ENABLE_ENV, "1") not in _OFF_VALUES` (`:349`). |
| S7-05 | "Smallbar BAR1: peer VRAM over 256-MiB BARs" | WIDER | requested size `int(os.environ.get("SGLANG_BARLINK_BAR1_WINDOW_MIB", str(WINDOW_MIB_DEFAULT))) * 1024 * 1024` with `WINDOW_MIB_DEFAULT = 96`; actual size `cap = free - reserve` from NVML, else `cap = gross - reserve - already` from sysfs | `barlink_matrix_transport.py:116-120`, `:67`, `:280-302`; gross read at `barlink_bar1.py:784-805` | **256 MiB is nowhere a constant** — it is a comment about the 3080's aperture. The window is PROBED (`nvmlDeviceGetBAR1MemoryInfo` → `bar1Free`, fallback `/sys/bus/pci/devices/<bdf>/resource` line 1) and configurable globally AND per group (`SGLANG_BARLINK_BAR1_WINDOW_MIB_<GROUP>`, `_requested()` at `:113-120`). A larger BAR raises reachability directly: `check_window_requirement` refuses per *actually mapped contiguous* length, not per nominal size. |
| S7-06 | what refuses BAR1 at boot | EXACT | `if not os.path.exists(path): raise Bar1Unavailable(f"{path} is missing. ...")` (holder); `if not ps["peer_bar1_regkey"]` branch after `self._cuda.register_io(...)` raises; `if not t._proofs_hold:` → `report["holds_space"] = True` | `barlink_bar1.py:644-653` (`HOLDER_PATH` at `:589`), `:2331-2374`, `build_bar1` `:4644-4656`; `PEER_BAR1_REGKEYS = ("BarlinkPeerBar1", "RMSmallBarP2PPeerBar1")` at `:597` | Four independent refusals: `/dev/dmabuf_holder` absent, dma-buf export unavailable, `cudaHostRegister(IoMemory)` on the peer BAR denied (regkey OR `CAP_SYS_ADMIN`), byte proof failed. Plus a structural cap `if self.world > MAX_RANGE: raise Bar1Unavailable(...)` with `MAX_RANGE = 8` (`:1518-1523`, `:811`) and `if self.world < 2: raise Bar1Unavailable("fewer than two ranks -- nothing to do")` (`:1926`). |
| S7-07 | BAR1/transport sizing knobs (undocumented set) | NOT-IN-CATALOG (WIDER) | `_CHUNK_BYTES = int(os.environ.get("SGLANG_BARLINK_CHUNK_MIB", "8")) * 1024 * 1024`; `_SLOT_BYTES = int(os.environ.get("SGLANG_BARLINK_SLOT_MIB", "64")) * 1024 * 1024`; `_SLOT_MIB_ENV = "SGLANG_BARLINK_HOST_SLOT_MIB"`; `int(os.environ.get("SGLANG_BARLINK_PIPE_CHUNK_MIB", "4")) * 1024 * 1024` | `barlink.py:52`, `:70`; `barlink_host.py:120`; `barlink_device.py:989` | None of the four is in any doc. `SLOT_MIB` sizes device/shm/host slots (NOT bar1 — bar1 uses `WINDOW_MIB`); `HOST_SLOT_MIB` overrides it for the host transport only; `CHUNK_MIB` is the inline gloo pipeline chunk; `PIPE_CHUNK_MIB` is the device transport's reduce-scatter/all-gather pipe chunk. Raising `SLOT_MIB` directly widens `shm`'s `handles` (`barlink_shm.py:207`: `return op in self.BARLINK_OPS and nbytes <= self.slot_bytes`). |
| S7-08 | "dmabuf GPU-RDMA works on consumer cards with the stock driver" | NARROWER (DOC) | `fn = getattr(self.drv, "cuMemGetHandleForAddressRange", None)` … `if rc == 0: return int(fd.value), [], "cuMemGetHandleForAddressRange"` … `ext = barlink_bar1_ext.load_dmabuf_ext(); if ext is None: raise Bar1Unavailable("dma-buf export not possible. ...")` | `barlink_bar1.py:517-537` | The dma-buf EXPORT is genuinely probed and does work on GeForce with the stock libcuda/ioctl route (the code documents `DMA_BUF_SUPPORTED = 0` yet a working `nv->dma_buf_supported = 1`, `:502-506`). What is NOT stock is the consumer of that fd: the BAR1 peer mapping needs the patched-driver regkey `BarlinkPeerBar1` (`:2337-2342`) and the out-of-tree `dmabuf_holder` module. Catalog conflates the two halves. |
| S7-09 | recorder "off by default", two env gates | NARROWER (DOC) | `_RECORDING = os.environ.get(ENV_RECORD, "0") not in ("0", "", "false", "False")` (read ONCE at import) and `directory = os.environ.get(ENV_DUMP_DIR, ""); if not directory: return None` | `barlink_uniformity.py:205`, `:230-232`; names at `:81-82` | Off by default: confirmed. Two undocumented facts: (a) `_RECORDING` is an IMPORT-TIME read — exporting the var after `barlink_uniformity` is imported has no effect (`set_recording_for_test` is the only in-process flip, `:213-221`); (b) `SGLANG_BARLINK_RECORD_DUMP_DIR` alone does nothing — recorders are only constructed from `record_decision`, which returns at `if not _RECORDING: return` (`:250`). |
| S7-10 | `PathProfile.saturation_threshold` — AUDIT_421 open question | EXACT (settled) | `if self._utilization_locked(best.name) >= best.saturation_threshold:` with `saturation_threshold: float = 1.0` and `if self._saturation_sensor is None: return 0.0` | `barlink_path_dispatcher.py:357`, `:126`, `:385-388` | **Settled — see the dedicated section below.** No production writer for the field AND no production caller of `set_saturation_sensor` (`grep` over `python/` finds only the definition; the only callers are `test/registered/unit/distributed/*` and `scripts/gpu_battery/s08_dispatcher_tables.py:167,211`). The one named production-intended sensor, `bus_saturation_sensor`, is BINARY: `return 1.0 if stats.get("pending_demand") else 0.0` (`:415`). |
| S7-11 | "beats NCCL 1.13-1.34x in serving" | MEASUREMENT CLAIM — not a gate | (no predicate) | — | Nothing in code reads or asserts a ratio. Not classifiable as a gate; kept out of the gate taxonomy deliberately. |
| S7-12 | Rig facts (no P2P/NVLink, x4/x8/x8 negotiated, NCCL-verbs broken on RoCE) | MEASUREMENT/ENVIRONMENT CLAIM | (no predicate) | — | Not gates. The only code-side companion is the identity map's negotiated-width rule, out of scope for §7. |
| S7-13 | "graph-capable direct mode" + fallback honesty | EXACT | `if chosen is None and graph_capture_running(): ... raise RuntimeError(f"barlink: {op!r} with {nbytes} bytes during a CUDA graph capture, but {reason}. ...")` | `barlink.py:635-676` | Under capture there is no silent gloo fallback — the abort is unconditional on the FINAL choice, after the #279 dispatcher hook. Outside capture, a once-per-(op,size-class) warning (`:600-634`). |
| S7-14 | `matrix` is "strictly more than bar1" (code comment `barlink.py:302`) | NARROWER (BUG) | `BARLINK_OPS: frozenset = frozenset({"all_reduce", "all_to_all", "all_to_all_single"})` and `if op not in self.BARLINK_OPS or self.bar1 is None: return False` | `barlink_matrix_transport.py:354-356`, `:452-454` vs `barlink_bar1.py:1450-1453` (`{"all_reduce", "all_gather", "all_to_all", "all_to_all_single", "broadcast"}`) | `matrix`'s op set is a strict SUBSET of its own sub-path's. Under `SGLANG_BARLINK_TRANSPORT=matrix`, `all_gather` and `broadcast` are handed to the gloo plane even though `bar1` handles them — and `matrix` is in `GRAPH_ENABLE_TRANSPORTS`, so those two ops hit the capture abort at `barlink.py:660` instead. See bug candidates. |
| S7-15 | (bar1 op coverage vs the device transport) | NARROWER (DOC) | bar1: `{"all_reduce","all_gather","all_to_all","all_to_all_single","broadcast"}`; device: `{"all_reduce","all_gather","reduce_scatter","broadcast"}` with `def handles(self, op, nbytes): return op in self.BARLINK_OPS` | `barlink_bar1.py:1450`; `barlink_device.py:1152-1157` | bar1 has NO `reduce_scatter`; device has no `all_to_all`. `reduce_scatter_tensor` under bar1 therefore always rides the gloo plane, and under capture aborts. Not stated anywhere in §7. |
| S7-16 | UCX transport (chunk pipelining, dual worker, tuned all_gather ring) | EXACT | `BARLINK_OPS = frozenset({"all_reduce", "all_gather", "broadcast", "reduce_scatter"})`, `return op in self.BARLINK_OPS` | `barlink_ucx.py:376`, `:668` | Size-unconditional: ucx declares no aperture/slot ceiling in `handles`. It is NOT in `capturable_transports()`, so `_enforce_cpu_transport_needs_eager` rejects it at startup unless `--disable-cuda-graph` (`parallel_state.py:365-383`). |
| S7-17 | #438a fp8 × uneven weighted DCP × BAR1 → warn, not refuse | EXACT (orchestrator-verified) | four AND'ed conditions | `barlink_uniformity.py:568-575`; env pair `:461-468`; `_is_fp8_weight_quant` `:393-407` | Carried forward unchanged. |
| S7-18 | BAR1 abort-gate knobs | EXACT (orchestrator-verified) | — | `barlink_abort_gate.py:60, 67, 75` | Carried forward unchanged. |
| S7-19 | retired-env guard | EXACT (orchestrator-verified) | 118 entries + `RETIRED_PREFIX = "SGLANG_HTCCL"` | `barlink_env_guard.py:25, 162, 188` | Carried forward unchanged. |
| S8-01 | MXFP4 kernel presence "probed via the `ggml_mxfp4_native` marker op" | EXACT | `return hasattr(torch.ops.sgl_kernel, "ggml_mxfp4_native")` | `python/sglang/srt/layers/quantization/gguf.py:272` (op defined `sgl-kernel/csrc/quantization/gguf/gguf_kernel.cu:119`) | Existence probe, deliberately not device-gated (`:268-271`). |
| S8-02 | "overridable with `SGLANG_GGUF_MXFP4_NATIVE=0`" | NARROWER (DOC) | `if os.environ.get("SGLANG_GGUF_MXFP4_NATIVE", "1")[:1] == "0": return False` | `gguf.py:265-266` | FIRST-CHARACTER test, evaluated ONCE at import (`MXFP4_NATIVE = _mxfp4_kernels_present()`, `:277`). `=false`, `=no`, `=off` do NOT disable it; `=0x1` DOES. A late export has no effect. |
| S8-03 | "the type is in all three GGUF type sets" | EXACT | `DEQUANT_TYPES = DEQUANT_TYPES \| MXFP4_QUANT_TYPES` / `MMVQ_QUANT_TYPES = MMVQ_QUANT_TYPES \| MXFP4_QUANT_TYPES` / `MMQ_QUANT_TYPES = MMQ_QUANT_TYPES \| MXFP4_QUANT_TYPES` under `if MXFP4_NATIVE:` | `gguf.py:278-281`; the three sets defined `:244-246` | The three sets are `DEQUANT_TYPES`, `MMVQ_QUANT_TYPES`, `MMQ_QUANT_TYPES`. A fourth consequence the catalog does not mention: `MOE_OFFLOAD_SUPPORTED_TYPES = MMVQ_QUANT_TYPES` (`:292`), so a #398 wheel also makes MXFP4 experts eligible for the #123/#268 MoE expert-offload. |
| S8-04 | "the repack … is a no-op on such a wheel" | WIDER / mechanism differs | `if native_mxfp4_kernels(): return {}` inside `_type_map()`, and separately `if not repack_enabled(): return set()` inside `repack_source_types()` | `python/sglang/srt/model_loader/gguf_mxfp4_repack.py:113-115`, `:122-124` | It is NOT "the repack runs and does nothing" — a SEPARATE predicate empties the type map, so every entry point becomes the identity. Two independent routes to the empty map; either alone suffices. |
| S8-05 | `SGLANG_GGUF_MXFP4_REPACK` (undocumented) | NOT-IN-CATALOG (NARROWER when set) | `return bool(envs.SGLANG_GGUF_MXFP4_REPACK.get())`, default `EnvBool(True)`; consequence `raise RuntimeError(f"GGUF tensor {tensor_name!r} is {type_name}, which no GGUF kernel in this build dispatches on, and {_ENV_VAR}=0 disabled the lossless load-time repack ...")` | `gguf_mxfp4_repack.py:82-86`, `:127-135`; default `environ.py:1776` | Interaction, all four states: NATIVE=1 → repack irrelevant either way. NATIVE=0 + REPACK=1 (default) → MXFP4→Q5_0 repack, +29.4 % bytes. NATIVE=0 + REPACK=0 → **hard load-time refusal by name**, no fallback. NATIVE=1 + REPACK=0 → native, no effect. So `REPACK=0` is only ever reachable as a behaviour change together with `NATIVE=0` or an old wheel. |
| S8-06 | "graph-replay numeric safety for ALL quants" | EXACT (literally all), one caveat | `if (_is_cuda and shape[0] * shape[1] * x.dtype.itemsize > _dequant_ws_cap_bytes() and _dequant_supports_out() and torch.cuda.is_current_stream_capturing()): y = _mul_mat_dequant_chunked(...)` | `gguf.py:942-948` | Type-agnostic — no quant name list; every type in `DEQUANT_TYPES` reaches it. "ALL quants" is literally true. It is NOT "all wheels": `_dequant_supports_out()` (`:783-792`, `"out" in schema`) is a wheel probe; on an old wheel the capture falls back to a fresh full-size allocation and the #70 capture OOM returns. |
| S8-07 | `gguf_mmq_decode_threshold` | NARROWER (DOC) | `min_bucket = table.get(shape_class); if min_bucket is None or bucket < min_bucket: return False` after `table = _MMQ_BUCKET_MIN.get(cap); if not table: return False` | `gguf.py:665-672`; table `:539-542`; bucket rounding `_decode_bucket_for` `:576-585`; enable gate `:596-617` | Comparison is `bucket >= min_bucket` on the CUDA-graph decode BUCKET (raw M rounded UP to the next captured bucket), not raw token count. Below → MMVQ (byte-identical to default). Above → MMQ, plus a one-shot INFO. **`_MMQ_BUCKET_MIN` is a device-capability ENUMERATION `{(12,0), (8,6)}`** — on any other GPU (sm89, sm90, sm100, MI300, …) the flag is silently inert. sm86 is present but maps to `None` for all three shape classes, i.e. measured never-wins. |
| S8-08 | "batched MMVQ" | EXACT | `if _BATCHED_MMVQ_ENV is not None: return _BATCHED_MMVQ_ENV == "1"; return _dequant_supports_out()` | `gguf.py:342-345` | Default follows a WHEEL probe, not a device. |
| S8-09 | "K-quant MMVQ tuned to Q8_0 efficiency" | EXACT | `_kq_tuned_cached = has_op and (len(env) == 0 or env[0] != "0")` with `has_op = hasattr(torch.ops.sgl_kernel, "ggml_mmvq_kq_tuned")` | `gguf.py:355-371`; consumed at `:442-443` (`if _kq_kernel_tuned(): return False`) | Wheel probe + `SGLANG_GGUF_KQ_KERNEL` kill switch (first-character test again). When present it fully disables the #72 shape reroute — the tuned kernel makes MMVQ win at every measured M<=8 on sm86. |
| S8-10 | GGUF MoE expert-offload coverage | EXACT | `MOE_OFFLOAD_SUPPORTED_TYPES = MMVQ_QUANT_TYPES` + `gguf_moe_offload_covered_type` | `gguf.py:292`, `:295-…` | Structural (kernel-set membership), not a name list. Widens automatically with the wheel (see S8-03). |
| S8-11 | (uneven-TP GGUF output-dim rule) | NAME-KEYED GATE (found, correct but fragile) | `if block_idx == 0 and getattr(quant_config, "get_name", lambda: "")() == "gguf": return units` | `python/sglang/srt/layers/linear.py:267-268` | Keyed on the config's string name. Any second GGUF-flavoured config class (or a rename) loses the #82 output-dim exemption silently. Recorded, not proposed for change — the rule is genuinely GGUF-specific. |
| S9-01 | "opt-in deterministic `SGLANG_DETERMINISTIC_FP8_GEMM`" | NARROWER (DOC) | `if not _environ.envs.SGLANG_DETERMINISTIC_FP8_GEMM.get(): return False` … `sm = major * 10 + minor; if not (80 <= sm < 89): return False` | `python/sglang/srt/layers/quantization/fp8_utils.py:313`, `:321-323` | Reach: dense `Fp8LinearMethod` and `CompressedTensorsW8A16Fp8`, on NVIDIA sm80..sm88 ONLY. It is a NO-OP on sm89/sm90/sm120 and on ROCm. It does NOT reach fp8 MoE experts (`fp8.py:1245-1252` logs the gap), FBGEMM fp8 (`fpgemm_fp8.py:56-67`), or the multimodal_gen runtime, and it deliberately does not gate `can_auto_enable_marlin_fp8` (`fp8_utils.py:2340-2347`). **Not the #197 shape** — every uncovered consumer emits its own warning, so the asymmetry is announced rather than hidden. |
| S9-02 | "e4m3 KV bit-exact on sm86" | MEASUREMENT CLAIM — no gate | (no predicate; grepped `sm86`, `(8, 6)`, `== 86`, `e4m3` across `layers/`, `mem_cache/`, `server_args.py`) | — | There is no sm86-keyed code path for the KV dtype. `--kv-cache-dtype fp8_e4m3` is accepted for "CUDA 11.8+" generally (`server_args.py:987`). |
| S9-03 | "NVFP4 (V4 class usable via dequant fallback for unpackable layers)" | WIDER | Marlin lane: `if output_size is None or output_size % GPTQ_MARLIN_MIN_THREAD_N == 0: return None`; native lane: `if width % NVFP4_NATIVE_MIN_N == 0: return None`; router: `if get_fp4_gemm_runner_backend().is_marlin(): return nvfp4_marlin_unpackable_reason(layer)` | `compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:152-157`, `:186-192`, `:225-234`; routing `compressed_tensors/compressed_tensors.py:1025`, `:993-…`, `:1189` | **Pure SHAPE test, no name list anywhere.** Marlin is judged `% 64` on the UNSHARDED width, native FP4 `% 32` on the SHARDED width — two lanes, two widths, two moments (`get_quant_method` and `create_weights`). Nothing restricts this to the "V4 class": ANY NVFP4 checkpoint, any model, any TP ratio whose layer width misses its lane's tile is routed to `CompressedTensorsW4A4Fp4Dequant` automatically, per layer, with exact numerics. |
| S9-04 | "mxfp8 needs capability 100" | WIDER | `if self.use_mxfp8 and _is_hip and _is_gfx95_supported: return 95` / `if self.use_mxfp8 and _mxfp8_to_block_fp8_required: return 94` / `return 100 if self.use_mxfp8 else 80`, enforced as `if capability < quant_config.get_min_capability(): raise ValueError(...)` | `python/sglang/srt/layers/quantization/fp8.py:305-315`; enforcement `model_loader/loader.py:257-267` | Comparison is `>= 100`, **not `== 100`** — sm120 (RTX 5090, capability 120) passes the floor. And 100 is not the only answer: ROCm gfx95 floors at 95 and a gfx942-class part that needs `mxfp8_block_convert_required()` floors at 94. The floor is also skipped entirely when `supports_current_device()` answers True/False first (`loader.py:224-238`) and is unenforceable on ROCm (`:242-255`). |
| S9-05 | Marlin "lcm=128 on coupled dims" | EXACT | `n, k = _marlin_min_thread_pair(); return math.lcm(n, k)` with `(GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MIN_THREAD_K) = (64, 128)` | `python/sglang/srt/layers/linear.py:157-182`, `:185-200` | Symmetric by construction (#385 correction): both dims carry 128 so gate_up's OUTPUT and down's INPUT coarsen identically. Siblings agree: `modelopt_fp4_uneven_tp_block` `lcm(32,128,16)=128` (`modelopt_quant.py:223-226`), `CompressedTensorsConfig._group_size_block` `lcm(groups…,128)` (`compressed_tensors.py:187-198`), INT8-W8A8 `lcm(8,16)=16` (`w8a8_int8.py:195`, deliberately NOT 128 — its path is not marlin). |
| S9-06 | #444b MXFP8 `weight_block_size [1, 32]` coarsened by `lcm` of both axes | WIDER | `asymmetric_block = bool(raw) and len(raw) == 2 and raw[0] != raw[1]` → `block = math.lcm(int(raw[0]), int(raw[1]))` | `python/sglang/srt/layers/linear.py:289-291` (the `[1, 32]` pin itself: `fp8.py:291-296`) | **Structural, not keyed on "mxfp8".** ANY quant config exposing a two-element asymmetric `weight_block_size` gets the lcm fold plus the marlin fold at `:318-319`. MXFP8 is merely the first instance. Symmetric exposures ([128,128], [256,256], group-size siblings) are untouched because `raw[0] == raw[1]`. |
| S9-07 | (the predicate that decides "can be marlin-packed") | NARROWER (BUG) | `return type(quant_config).__name__.lower() in _MARLIN_PACKABLE_CONFIGS` with `_MARLIN_PACKABLE_CONFIGS = ("fp8config", "compressedtensorsconfig", "fbgemmfp8config")` | `python/sglang/srt/layers/linear.py:236`, `:206` | **The §12 name-list family, second instance in this tree.** A CLASS-NAME enumeration decides whether the marlin 128-fold is applied. `MarlinConfig`, `GPTQMarlinConfig`, `AWQMarlinConfig`, `QuarkConfig`, `QuarkInt4Fp8Config`, `W8A8Fp8Config`, `W4AFp8Config`, `PetitNvFp4Config` are all absent; of these `MarlinConfig`, `W8A8Fp8Config` and `QuarkConfig` also expose no `weight_block_size` (grepped), so `block` stays `None` and `if not block: return units` (`:320-321`) means **no coarsening at all under `--rank-tp-ratio`** — exactly the #377/#383 failure mode the fold exists to prevent. See bug candidates. |
| S9-08 | FP8 "per-channel fused GEMV" | EXACT | scale factored out of the k loop, raw-byte e4m3 decode: `Triton's ``fp8e4nv`` does not exist on sm75/sm70/sm86/gfx900` | `python/sglang/srt/layers/quantization/fp8_dequant_gemv.py:231-244` | Structural (per-channel scale shape), not device-gated; the byte-decode exists precisely so it runs where `fp8e4nv` does not. |
| S9-09 | INT8-W8A8 "sm86-native lane" | EXACT | `return [math.lcm(INT8_SCALED_MM_ALIGN_N, INT8_SCALED_MM_ALIGN_K)] * 2` = `[16, 16]` | `python/sglang/srt/layers/quantization/w8a8_int8.py:195` (rationale `:178-194`) | The block carries no quantization meaning — it exists only for the CUTLASS alignment check under an uneven split. Deliberately not folded to 128. |
| S9-10 | fp8 "dequant fallback" reachability | EXACT | `if get_bool_env_var("SGLANG_FORCE_FP8_DEQUANT"): return True` / `if deterministic_fp8_marlin_disabled(device_id): return True` / `return not (fp8_native_gemm_available(device_id) or cutlass_fp8_supported(device_id) or can_auto_enable_marlin_fp8(device_id))` | `python/sglang/srt/layers/quantization/fp8_utils.py:359-367` | Functional (kernel-existence), not a capability number. `SGLANG_FORCE_FP8_DEQUANT=1` is an undocumented exerciser for the fallback on hardware that does not need it. |

Class counts: **WIDER 5** (S7-05, S7-07, S8-04, S9-03, S9-04, plus S9-06) — 6 including S9-06;
**NARROWER 9** (S7-03, S7-08, S7-09, S7-14, S7-15, S8-02, S8-05, S8-07, S9-01, S9-07) — 10 counting S9-07;
**EXACT 20**; **NOT-FOUND 0**; **measurement/environment claims (not gates) 3** (S7-11, S7-12, S9-02).

---

#### WIDER finds — what they unlock

**S7-05 — BAR1 is not a 256-MiB mechanism.** The window is probed per rank
(`nvmlDeviceGetBAR1MemoryInfo` → `bar1Free`, sysfs gross as fallback) and clipped
to `free - reserve`, with a 96-MiB default REQUEST and per-group override
`SGLANG_BARLINK_BAR1_WINDOW_MIB_<GROUP>`. On a card with a resized/large BAR — the
5090 here — the group can carry a much larger window, and since `handles` fails
per size against the *mapped* length, a bigger window keeps more of the prefill
all_reduce range on BAR1 instead of dropping it to gloo. Concretely: raising
`SGLANG_BARLINK_BAR1_WINDOW_MIB` for the `tp` group while pinning `..._DCP` low is
a supported, already-implemented way to buy back the 2457-token prefill fallback.

**S7-07 — four sizing knobs nobody has swept.** `SGLANG_BARLINK_SLOT_MIB` (64),
`SGLANG_BARLINK_HOST_SLOT_MIB`, `SGLANG_BARLINK_CHUNK_MIB` (8),
`SGLANG_BARLINK_PIPE_CHUNK_MIB` (4) are live and undocumented. `SLOT_MIB` is the
direct lever on `shm`'s size ceiling and on the device/host transports' staging
geometry; `PIPE_CHUNK_MIB` is the device transport's pipeline grain. These are the
cheapest untried per-phase knobs in §7 (prefill wants large chunks, decode wants
small ones) and they cost nothing but a boot to sweep.

**S9-03 — the NVFP4 dequant fallback is model-agnostic.** It is a `% 64` /
`% 32` shape test on the layer width, evaluated per layer and per lane, with the
Marlin verdict on the unsharded width and the native verdict on the sharded one.
Nothing scopes it to DeepSeek-V4. Any NVFP4 checkpoint on this rig — including
`Qwen3.6-27B-NVFP4` at any `--rank-tp-ratio`, and the gated-delta-net gate whose
width no TP split can rescue — boots, with the offending layers served exactly by
`F.linear` over dense weights. That makes NVFP4 a usable lane for uneven TP, not a
V4-only curiosity.

**S9-04 — mxfp8 already runs on the 5090.** The floor is `capability >= 100`, and
sm120 = 120. The catalog's "needs capability 100" reads as "Blackwell datacenter
only" and has been treated that way; in fact an MXFP8 checkpoint clears the loader
floor on the 5090 rank today. On ROCm the same config floors at 95 (gfx95) or 94
(gfx942 via block-fp8 conversion). Any solver that excluded the mxfp8 lane on this
rig excluded it on a misread.

**S9-06 — the #444b coarsening is not an MXFP8 patch.** The predicate is
`raw[0] != raw[1]`. Every future quant config that exposes an asymmetric two-axis
block inherits the lcm fold plus the marlin fold automatically, and every symmetric
one is provably untouched. The catalog line reads like a one-checkpoint fix; the
code is a general rule for the coupled-dimension family.

**S8-04 — the MXFP4 "no-op" is a short-circuit, not a cheap pass.** On a native
wheel `_type_map()` returns `{}` before any tensor is touched, so `repack_source_types()`
is empty and the family adapters' executability gates see the same empty set. That
means the native path costs literally zero load-time work, and it also means an
executability gate built on `repack_source_types()` refuses exactly what it refused
before the repack existed — worth knowing before adding a new GGUF family adapter.

---

#### NARROWER — bug candidates

**S7-14 — `matrix` transport silently drops `all_gather` and `broadcast` to gloo,
and aborts under capture.**
`barlink_matrix_transport.py:354-356` declares `{"all_reduce","all_to_all","all_to_all_single"}`
while its own sub-path `barlink_bar1.py:1450` declares
`{"all_reduce","all_gather","all_to_all","all_to_all_single","broadcast"}`, and
`handles` refuses on the composite's set first (`:452`). The module comment at
`barlink.py:302` claims matrix is "strictly more than bar1" — it is strictly less
on op coverage. Because `matrix` IS in `GRAPH_ENABLE_TRANSPORTS`, a captured decode
graph that issues `all_gather` (uneven TP does) hits the hard abort at
`barlink.py:660-676` rather than degrading. User harm: `SGLANG_BARLINK_TRANSPORT=matrix`
is unusable with CUDA graphs on the fork's own uneven-TP path, and the planner arm
it exists to serve cannot be measured against `bar1` on equal footing.
→ *Task title:* `barlink matrix transport: delegate all_gather/broadcast to its bar1 sub-path instead of refusing`

**S9-07 — the marlin fold is gated on a class-name enumeration, so marlin-format
checkpoints get no uneven-TP coarsening.**
`linear.py:236` tests `type(quant_config).__name__.lower() in ("fp8config", "compressedtensorsconfig", "fbgemmfp8config")`.
`MarlinConfig`, `W8A8Fp8Config` and `QuarkConfig` are marlin-served and expose no
`weight_block_size`, so `_quant_block_aligned_units` returns the element-granular
unit family unchanged (`:320-321`) and an uneven `--rank-tp-ratio` split lands
mid-tile — the exact `"size_n = 17888 is not divisible by tile_n_size = 64"` abort
#383 was built to prevent, now reachable again through a different config class.
The docstring even argues the predicate must be a property of the CHECKPOINT rather
than the device, and then implements it as a list of three strings.
→ *Task title:* `_marlin_packable_family: replace the class-name list with a config-declared marlin-packable property`

**S8-07 — `--gguf-mmq-decode-threshold` is silently inert on every GPU except
sm120 and sm86.** `_MMQ_BUCKET_MIN` (`gguf.py:539-542`) has two entries; an absent
capability returns `False` at `:667-668` with no log line at all. A user on sm89/sm90
who passes the flag gets no reroute, no warning, and no way to tell from the logs
that the flag did nothing (the one-shot INFO only fires on the *active* path).
Harm is limited to a silently ignored flag, not wrong numbers.
→ *Task title:* `gguf mmq decode threshold: log once when the flag is set but this device has no measured table`

**S7-09 — `SGLANG_BARLINK_RECORD_DECISIONS` is read at import.** The catalog
presents it as a switch for post-mortems on a wedged run; in practice it must be in
the environment before `barlink_uniformity` is first imported, and
`SGLANG_BARLINK_RECORD_DUMP_DIR` alone never creates a recorder. An operator who
exports it against a running or partially-initialised process gets silence. Harm is
a debugging instrument that appears armed and is not.
→ *Task title:* `barlink decision recorder: state the import-time read in the docs (or re-read the env at recorder construction)`

Deliberately NOT filed as bugs: S7-03, S7-08, S7-15, S8-02, S8-05, S9-01. Each is a
catalog wording problem with no user-visible misbehaviour — the code in every case
warns or errors clearly at the point of use.

---

#### NOT-FOUND

Nothing in §7/§8/§9 came out unlocatable. Searches that returned empty and are
therefore reported as *absence of a gate* rather than a failed search:

* **sm86-keyed e4m3 KV gate (S9-02).** Grepped `sm86`, `(8, 6)`, `== 86`, `86` in a
  capability context, and `e4m3` across `python/sglang/srt/layers/`,
  `python/sglang/srt/mem_cache/`, `configs/model_config.py` and `server_args.py`.
  Every `sm86` hit is a comment or a measurement note; the only `(8, 6)` key is
  `_MMQ_BUCKET_MIN`. There is no arch gate on the KV dtype.
* **A saturation-sensor production writer (S7-10).** Grepped
  `set_saturation_sensor`, `bus_saturation_sensor`, `saturation_threshold`,
  `_saturation_sensor` over `python/`, `test/`, `scripts/`. Every hit under
  `python/` is either the definition itself or `OffloadRegister`'s unrelated
  same-named hook.
* **A ratio assertion behind "beats NCCL 1.13-1.34x" (S7-11).** No code reads,
  stores or checks that ratio; it is a measurement statement only.
* **`--collective-net-*` transport selection (S7-03).** Grepped
  `collective_net_small`, `collective_net_bulk`, `SGLANG_COLLECTIVE_NET_*` across
  `python/`. The only consumers are the UCX context (`barlink_ucx.py:189`) and
  `--disaggregation-ib-device`. No transport name is ever derived from them.

---

#### Catalog corrections

**§7 — `--collective-net-small/-bulk`**

```
OLD: `--collective-net-small/-bulk` per message class with typo hard-reject.
NEW: `--collective-net-small/-bulk` pin the NIC (not the transport) per message
     class; typo hard-reject against sysfs, not a name list
     (`server_args.py:14089-14098`, accepted set = dirs under
     /sys/class/infiniband + /sys/class/net, plus `all`, optional `:port`).
     SMALL reaches only the barlink UCX plane and pins BOTH small and large TP
     collectives (one UCX context — `server_args.py:14176-14185`); BULK reaches
     PD-KV/HiCache via --disaggregation-ib-device. BAR1 is not selectable here.
```

**§7 — Smallbar BAR1 window**

```
OLD: **Smallbar BAR1 direct path**: peer VRAM over 256-MiB BARs, beats NCCL
     1.13-1.34x in serving.
NEW: **Smallbar BAR1 direct path**: peer VRAM over the card's own BAR aperture —
     probed, not assumed (NVML bar1Free, sysfs gross fallback;
     `barlink_matrix_transport.py:280-302`). Requested window
     SGLANG_BARLINK_BAR1_WINDOW_MIB (default 96) with a per-group override
     SGLANG_BARLINK_BAR1_WINDOW_MIB_<GROUP> (`:113-120`); the group-wide MINIMUM
     governs (`barlink_bar1.py:1953-1985`). Payload eligibility is checked against
     the CONTIGUOUSLY mapped length, never the sysfs gross size
     (`barlink_bar1.py:2385-2405`). A larger BAR raises reachability directly.
     Measured 1.13-1.34x over NCCL in serving (measurement, not a gate).
```

**§7 — dmabuf / driver requirements**

```
OLD: dmabuf GPU-RDMA works on consumer cards with the stock driver.
NEW: dma-buf EXPORT works on consumer cards with the stock driver — probed:
     cuMemGetHandleForAddressRange first, NV_ESC_EXPORT_TO_DMABUF_FD ext as
     fallback (`barlink_bar1.py:517-537`). The BAR1 PEER MAPPING on top of it is
     NOT stock: it needs the widened driver guard (regkey BarlinkPeerBar1 /
     RMSmallBarP2PPeerBar1, `barlink_bar1.py:597`, `:2337-2342`), the
     dmabuf_holder module (`:589`, `:644-653`) and a passing byte proof
     (`:4644-4656`); CAP_SYS_ADMIN or PeerMappingOverride=1 is the second hurdle
     in a container (`:2352-2374`).
```

**§7 — graph-capable direct mode**

```
OLD: ... tuned all_gather ring, graph-capable direct mode.
NEW: ... tuned all_gather ring, graph-capable direct mode — capture-safety is
     TRANSPORT-NAME-keyed, not property-keyed:
     CAPTURABLE_BARLINK_TRANSPORTS = {"device","host"} plus
     GRAPH_ENABLE_TRANSPORTS = {"bar1","matrix"} through
     SGLANG_BARLINK_GRAPH_ENABLE (default on)
     (`parallel_state.py:298`, `:303`, `:352-362`). ucx/shm/gloo are refused at
     startup unless --disable-cuda-graph (`parallel_state.py:365-383`). Under an
     active capture there is no silent gloo fallback: barlink aborts with the
     reason (`barlink.py:635-676`).
```

**§7 — op coverage (new line; currently absent)**

```
NEW: barlink op coverage per transport, from BARLINK_OPS at source: device
     {all_reduce, all_gather, reduce_scatter, broadcast}
     (`barlink_device.py:1152`); host the same plus send/recv
     (`barlink_host.py:811-812`); ucx the same four (`barlink_ucx.py:376`);
     bar1 {all_reduce, all_gather, all_to_all, all_to_all_single, broadcast} —
     NO reduce_scatter (`barlink_bar1.py:1450`); matrix only
     {all_reduce, all_to_all, all_to_all_single} (`barlink_matrix_transport.py:354`),
     i.e. a strict SUBSET of its own bar1 sub-path — open defect. The
     communicator itself refuses exactly four collectives outright:
     reduce_scatter(list), reduce_scatterv, all_gather(output_tensor_list=),
     all_gatherv (`parallel_state.py:1348-1371`). bar1 additionally caps the
     group at 8 ranks (MAX_RANGE, `barlink_bar1.py:811`, `:1518`).
```

**§7 — collective-decision recorder**

```
OLD: Off by default (`SGLANG_BARLINK_RECORD_DECISIONS=1`, optional per-rank
     on-disk dump via `SGLANG_BARLINK_RECORD_DUMP_DIR` for post-mortems on a
     wedged run).
NEW: Off by default (`SGLANG_BARLINK_RECORD_DECISIONS=1`, read ONCE at import —
     `barlink_uniformity.py:205` — so it must be exported before the process
     starts; `SGLANG_BARLINK_RECORD_DUMP_DIR` adds the per-rank on-disk dump but
     does nothing on its own, because recorders are only built from
     record_decision, which returns early when recording is off, `:250`).
```

**§7 — path dispatcher (new line; currently absent, closes AUDIT_421 §8)**

```
NEW: #279 path dispatcher: flag-gated (SGLANG_BARLINK_PATH_DISPATCHER=1, read at
     call time, `barlink_path_dispatcher.py:428`) and inert — a fresh dispatcher
     has an EMPTY registry, so every decision is the status-quo #240 choice
     (`:431-443`). PathProfile.saturation_threshold is permanently 1.0 (no
     writer) and no production code attaches a saturation sensor, so
     _utilization_locked returns 0.0 and the `>= threshold` overflow tier at
     `:357` never fires today. It is not dead code: the one named
     production-intended sensor, bus_saturation_sensor, is BINARY
     (`return 1.0 if stats.get("pending_demand") else 0.0`, `:415`), for which
     threshold 1.0 is exactly right. AUDIT_421 §8's open question is closed:
     reachable by construction, correctly matched to its intended sensor,
     unreachable until #279's measured slice wires both.
```

**§8 — MXFP4 native probe and its overrides**

```
OLD: Kernel presence is a wheel property, probed via the `ggml_mxfp4_native`
     marker op (the #73 pattern) and overridable with
     `SGLANG_GGUF_MXFP4_NATIVE=0`, which hands the checkpoint back to the repack.
NEW: Kernel presence is a wheel property, probed via the `ggml_mxfp4_native`
     marker op (the #73 pattern, `gguf.py:272`) and evaluated ONCE at import
     (`:277`). `SGLANG_GGUF_MXFP4_NATIVE=0` hands the checkpoint back to the
     repack — first-character test (`:265`), so `false`/`no`/`off` do NOT
     disable it. The "no-op on a native wheel" is a short-circuit, not a cheap
     pass: `_type_map()` returns `{}` before any tensor is read
     (`gguf_mxfp4_repack.py:113-115`). Second, undocumented lever:
     `SGLANG_GGUF_MXFP4_REPACK=0` (default 1, `environ.py:1776`) empties the same
     map (`:122-124`); combined with NATIVE=0 or an old wheel it turns the
     checkpoint into a loud load-time refusal by tensor name (`:127-135`) —
     never a silent fallback. Native also widens MoE expert-offload coverage,
     since MOE_OFFLOAD_SUPPORTED_TYPES = MMVQ_QUANT_TYPES (`gguf.py:292`).
```

**§8 — perf line**

```
OLD: Perf: batched MMVQ, Q8 lm_head, K-quant MMVQ tuned to Q8_0 efficiency
     (TP=2 beats llama.cpp), graph-replay numeric safety for ALL quants,
     `gguf_mmq_decode_threshold`.
NEW: Perf: batched MMVQ (default follows the WHEEL probe `_dequant_supports_out`,
     `gguf.py:342-345`), Q8 lm_head, K-quant MMVQ tuned to Q8_0 efficiency
     (wheel probe `ggml_mmvq_kq_tuned` + SGLANG_GGUF_KQ_KERNEL kill switch,
     `gguf.py:355-371`; when present it fully disables the #72 reroute, `:442`),
     graph-replay numeric safety for ALL quants — literally type-agnostic
     (`gguf.py:942-948`), but it needs the `ggml_dequantize(..., out=)` wheel
     schema; on an older wheel the capture OOM returns. `gguf_mmq_decode_threshold`
     compares the CUDA-graph decode BUCKET (raw M rounded UP) against a MEASURED
     per-(capability, shape class) table `_MMQ_BUCKET_MIN` = {sm120, sm86} only
     (`gguf.py:539-542`, `:665-672`) — silently inert on any other device.
```

**§9 — deterministic fp8**

```
OLD: opt-in deterministic `SGLANG_DETERMINISTIC_FP8_GEMM`
NEW: opt-in deterministic `SGLANG_DETERMINISTIC_FP8_GEMM` — reaches dense fp8
     linears and CompressedTensorsW8A16Fp8 on NVIDIA sm80..sm88 ONLY
     (`fp8_utils.py:313`, `:321-323`); a no-op on sm89/90/120 and on ROCm, and
     deliberately NOT honoured by fp8 MoE experts (`fp8.py:1245-1252`), FBGEMM
     fp8 (`fpgemm_fp8.py:56-67`) or multimodal_gen — each has no fallback on
     Ampere and each logs the gap. It forces fp8_needs_dequant_fallback on
     (`fp8_utils.py:361-362`), costing ~2.5-6x decode throughput.
```

**§9 — NVFP4**

```
OLD: NVFP4 (V4 class usable via dequant fallback for unpackable layers)
NEW: NVFP4 — "unpackable" is a pure SHAPE test, not a checkpoint class: Marlin
     `output_size % 64` on the UNSHARDED width
     (`compressed_tensors_w4a4_nvfp4.py:152-157`), native FP4 `width % 32` on the
     SHARDED width (`:186-192`), routed per rank by the resolved lane
     (`:225-234`). Both verdicts land on CompressedTensorsW4A4Fp4Dequant
     (load packed, materialise dense once, F.linear, exact numerics). Any NVFP4
     checkpoint at any --rank-tp-ratio is therefore bootable, including layers no
     TP split can rescue (Qwen3.6-27B-NVFP4's merged b/a gate, n = 6g).
```

**§9 — mxfp8 / #444b**

```
OLD: The eighth (#444b) is MXFP8: its `weight_block_size [1, 32]` is the OCP
     scale layout, not an alignment registration, so an asymmetric exposed block
     is now coarsened by `lcm` of both axes before the marlin fold — latent,
     mxfp8 needs capability 100.
NEW: The eighth (#444b) is MXFP8: its `weight_block_size [1, 32]` is the OCP
     scale layout, not an alignment registration, so an asymmetric exposed block
     is coarsened by `lcm` of both axes before the marlin fold. The predicate is
     STRUCTURAL — `raw[0] != raw[1]` (`linear.py:289-291`) — so it covers every
     future asymmetric exposure, not just MXFP8; symmetric blocks are provably
     untouched. The mxfp8 floor is `capability >= 100` (`loader.py:261`), so
     sm120 (5090) CLEARS it; on ROCm the same config floors at 95 (gfx95) or 94
     (gfx942 block-fp8 conversion) (`fp8.py:305-315`).
     CAVEAT (open defect): the marlin fold that follows is gated on a CLASS-NAME
     list `_MARLIN_PACKABLE_CONFIGS = ("fp8config","compressedtensorsconfig",
     "fbgemmfp8config")` (`linear.py:206`, `:236`), so MarlinConfig /
     W8A8Fp8Config / QuarkConfig — marlin-served and exposing no
     weight_block_size — get NO uneven-TP coarsening at all.
```


### §10-§16 — Determinism, device identity, robustness, serving surface (partial)

Lower-leverage group per the task's own ranking, so this part is a targeted
pass over the conditional claims that name a checkable predicate, not an
exhaustive extraction.

| ID | Catalog claim (short) | Class | Gate predicate (verbatim) | file:line | Note |
|---|---|---|---|---|---|
| S12-01 | §12 "GPTQ `desc_act=True` is refused by name rather than fused wrong" | **WIDER** | `if q_a_proj_weight.shape != kv_a_proj_weight.shape or not torch.equal(q_a_proj_weight, kv_a_proj_weight):` | `models/deepseek_common/utils.py:170-172` | The predicate never reads `desc_act`. It is a STRUCTURAL value-equality check on the shared per-input-channel parameter; `desc_act=True` is only what the message names as the usual cause. See below. |
| S12-02 | §12 #472 "`dcp_even_write_mask` refuses a mask-less token-sharded write by name instead of returning `None`" | EXACT | `def dcp_even_write_mask(positions, num_rows, dcp_size, dcp_rank, precomputed=None) -> torch.Tensor` — returns a tensor, never `Optional`; both candidate sources must agree with `num_rows` | `layers/dcp/owner.py:363-386` | The docstring states the same reasoning the catalog does, and the signature (non-Optional return) is what enforces it. |
| S12-03 | §12 #472 "the fallback `forward_batch.dcp_kv_mask` is HIP-only" | EXACT | `assert dcp_kv_mask is None, (...)` on the non-HIP path | `mem_cache/swa_memory_pool.py:207`; `mem_cache/memory_pool.py:2412` | Confirmed by the assert plus the `extra = {} if dcp_kv_mask is None else {...}` shim at `swa_memory_pool.py:226`. |
| S12-04 | §12 "our WEIGHTED owner rule (#173) was immune throughout — it derives ownership from `out_cache_loc`" | EXACT | `dcp_weighted_write_slots` derives from `out_cache_loc`, not `positions` | `layers/dcp/owner.py:387-390` (stated), impl in same module | |
| S6-R1 | §6 "a 118-name retired-env guard that refuses stale SGLANG_* variables loudly" | **WIDER** | `n for n in env if n.startswith(RETIRED_PREFIX) or n in RETIRED_ENV_VARS` with `RETIRED_PREFIX = "SGLANG_HTCCL"` | `distributed/device_communicators/barlink_env_guard.py:188`, `:162`, `:25` | The name count is exactly 118 (verified by AST over `RETIRED_ENV_VARS`), but the guard is not limited to them: ANY variable whose name starts with `SGLANG_HTCCL` is refused, and `:171` auto-derives the successor name by rewriting the prefix to `SGLANG_BARLINK`. The reach is "118 names PLUS the whole retired prefix". |
| S13-01 | §13 video "`pre_downscale` / `decimate_resynth` … `require_runnable` excludes them" | EXACT | `if request.require_runnable and candidate.requires:` | `video_enhance/chain_policy.py:1301` | Exclusion is on the candidate's `requires` set, not on the two mode names — so any future mode carrying an executor requirement is excluded by the same predicate (mildly wider than the catalog's two-name phrasing). |
| S13-02 | §13 RIFE "the registry *refuses* a rung that is neither present on disk nor sha256-pinned" | EXACT | `RifeLadder.__post_init__` raises `LadderError` | `video_enhance/rife_ladder.py:458-486` (`:148` states the rule) | |
| S13-03 | §13 "TWO replicated stages … refused by name" | EXACT | refusal text "linear program rather than a water-fill and is not built" | `video_enhance/stage_pipeline.py:513-520` | |
| S13-04 | §13 "`best_placement` only replicates stages the caller names in `replicable=`" | EXACT | `replicable: Sequence[str] = ()` default empty; `for target in replicable:` | `video_enhance/stage_pipeline.py:717, 741, 757, 775` | Default `()` means "an empty `replicable` yields exactly what it always did" (`:729`) — the opt-in is real. |
| S11-01 | §11 "the ONLY bridge is the IdentityMap … never feed a CUDA ordinal to NVML" | EXACT (mechanism present) | `class IdentityMap` + `raise DeviceOrderUnresolvedError(...)` when the order cannot be resolved | `registry/nvml.py:412, 506, 50` | The refusal exists and is imported by `planner/flags.py:60`, so the canon has an enforcing failure mode rather than only a rule. Not audited: whether every ordinal-consuming callsite routes through it (that is a Detector-C-shaped sweep, out of scope here). |

#### WIDER finds — what they unlock

**S12-01 (`desc_act`).** The fused-`a_proj` guard is keyed on whether the two
projections' per-input-channel parameter actually differs, not on the quant
config's `desc_act` flag. Two consequences the catalog line hides: a GPTQ
`desc_act=True` checkpoint whose two `g_idx` permutations happen to agree
FUSES and serves — it is not refused — and conversely any other format that
grows a disagreeing per-input-channel parameter is caught by the same guard
without a code change. This is §12's own quant-name-list doctrine (#443/#446)
being followed correctly, and the catalog states it as if it were a name check,
which is the failure mode the doctrine exists to prevent.

**S6-R1 (retired-env guard).** An operator upgrading across the HTCCL→barlink
rename is protected for every `SGLANG_HTCCL*` variable that ever existed,
including names nobody enumerated, and gets the successor name computed by
prefix rewrite. The catalog's "118-name" phrasing understates it as a fixed
list and would let a reader conclude an unlisted stale name passes silently.

#### NARROWER — bug candidates

None in this group. S12-01 and S6-R1 are the opposite direction, and the
remaining items are exact.

#### NOT-FOUND

- §10's determinism canon ("no A/B without a same-boot A-vs-A floor", "first
  boot after cache changes is a JIT outlier") is process discipline with no
  code gate, by design. Not a reach claim; not classified.
- §14 (dashboard: anonymization gate, opt-in PAT, self-update with
  auto-rollback) and §16 (instruments) were not swept for predicates in this
  pass — the task ranked them last and the audit's context went to §1-§7.
  Recorded as an explicit coverage gap rather than as a clean bill of health.

## 8. What changed in this branch, and what did not

**Changed — documentation only.** `docs/dev/FEATURE_CATALOG.md`: rule (3) and
(4) added at the head (a conditional line is a POINTER to a predicate; the
second registry is named), and every corrected conditional line in §§1-9, §13
and §17 now carries its gate predicate with `file:line` as evidence.
`docs/dev/NOTE_485_joint_phase_vectors.md`: a correction banner plus two
in-place corrections, because the catalog's "grid-PINNED" line points there and
a reader following the pointer would otherwise land on the same stale claim.

**Not changed — no code, no behaviour.** No file under `python/`, `sgl-kernel/`,
`test/` or `scripts/` was touched. The twenty bug candidates in §5 are proposals;
none was fixed here, deliberately, so that each lands as its own falsifier-first
change rather than inside a documentation commit.

**Coverage this audit does not claim.** §14 and §16 were not swept. Direction 2
enumerated three surfaces (routes, `ServerArgs`, `environ.py`); it did not
enumerate undeclared `SGLANG_*` names used in `srt/` but absent from
`environ.py` (AUDIT_421 counted 426, mostly inherited), nor the planner's own
CLI. Nothing was executed.

**The standing rule this audit produces.** A design or solver task that EXCLUDES
an option must quote the excluding predicate verbatim with `file:line`. "The
catalog says so" is not a citation — and after this audit, neither is
"`planner/flags.py` says so". #500-B1 is what it costs when a shorthand is read
as a specification twice in the same class.
