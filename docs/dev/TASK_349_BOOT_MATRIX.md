# Task #349 — the integration boot matrix, a standing bug net

ANALYSE #347 item 4. Cross-feature bugs are invisible to git and to
single-feature tests: the #132 x weightless NCCL hang, and the #340 arm matrix
that silently carried `SGLANG_UNEVEN_DCP=1` in the shared harness environment
and published a wrong verdict from it. This is a small, time-boxed matrix of
FEATURE-CROSS boots, each printing its EFFECTIVE (resolved) configuration and
gated on coherence rather than identity, so that class is caught systematically.

Status: desk work complete and unit-green. The multi-boot sweep is a later
card ticket (estimate below).

## What was built, and the standalone-before-tenant rule

Two consumers of one component, in that order of dependence:

1. **The component** — `python/sglang/srt/boot_matrix/`, importable and
   unit-tested on a card-less host:
   - `arms.py` — the arm list AS DATA. One dataclass, one tuple. The single
     place the matrix is defined.
   - `effective.py` — `report_effective()`, the resolved config read FROM THE
     SERVER LOG.
   - `coherence.py` — the gate: short byte-exact + long graded.
   - `check.py` — `check_arm()`, hermetic, file-only: artifacts → one verdict.
   - `sweep.py` — the standalone runner (`python -m sglang.srt.boot_matrix.sweep`).
2. **The tenant** — `workbench/tenants/boot_matrix.py`, a thin wrapper over the
   SAME `ARMS`. It restates nothing; it teaches the #347 workbench to run one
   arm as one preemptible segment.

The order is the point (the component-before-composition rule): the checking
logic is provable in isolation before it goes into anything, so a tenant bug
can never be confused with a matrix bug.

## The arm list (settled)

11 boot arms + 6 reject arms. Spec is "the r3-probe A-J set plus the crossings
#108 refused" — a deliberate covering subset, not the full 5-axis cross (which
is the truncation-vs-comprehensiveness trap: a matrix nobody runs is no net).

| arm | kind | axis | catches |
|---|---|---|---|
| A_default | boot | baseline, no new flags | the default path regressing; makes every delta meaningful |
| B_offload | boot | kv-session-offload × spec | host spill breaking the resident chain |
| C_crossalgo | boot | cross-algorithm × lazy capture | rung swap desyncing draft graphs |
| D_offload_x_crossalgo | boot | spec-in-tick offload × cross-algo | two features writing draft KV on one tick |
| E_barlink | boot | barlink device transport × spec | barlink collectives diverging from NCCL under verify |
| G_all_axes | boot | barlink × cross-algo × offload × spec-in-tick, graphs | **the #132×weightless class** — the highest-value arm |
| H_ps2_prefill_spill | boot | PS2 prefill-spill, no spec | born-spilled prefill sizing wrong with spec OFF |
| I_dflash_shards | boot | DFLASH per-rank shards × spill | uneven-TP draft MLP shards misaligning during a spill |
| J_waveback_ps2 | boot | wave-back × PS2 | two spill state machines racing on one pool |
| K_bar1_graphs | boot | bar1 transport × CUDA graphs (#369) | bar1 peer-VRAM staying graph-captured |
| L_video_cotenancy | boot | video / dual-lane co-tenancy | a second lane corrupting shared input buffers (#121) |
| reject_dcp_draftextend | reject | #108 dcp on its own lane | the v1 draft-extend guard going stale |
| reject_dcp_topk | reject | #108 dcp × tree topk>1 | #76 + #108 topk guards both failing |
| reject_dcp_multilayer | reject | #108 dcp × multi-layer EAGLE | multi-runner draft reaching the single-owner pool |
| reject_dcp_offlane | reject | #108 dcp off the weighted lane | dcp admitted with no vector to shard by |
| reject_dcp_crossalgo | reject | #108 dcp × cross-algorithm | runtime rung swap invalidating the boot chain guarantee |
| reject_dcp_offload | reject | #108 dcp × kv-session-offload | raw-global-slot draft writes on a sharded pool (#60) |

A REJECT arm is a PASS when the server refuses the configuration by name, at
arg resolution, before weight load. `reject_dcp_draftextend` deliberately
encodes the #108 v1 state: the day the draft-extend DCP split lands, this arm
must be reclassified to a boot arm, and a still-firing reject tells us the
guard text went stale — the matrix guards its own guards.

## report_effective() shape

An `EffectiveConfig` (dataclass) with one field per axis: `tp_size`,
`dcp_size`, `dcp_engaged`, `rank_tp_ratio`, `token_vector`, `spec_algorithm`,
`eagle_topk`, `cross_algorithm`, `draft_kv_layout`, `offload`,
`dual_group_lane`, `barlink`, `graphs`, `ready`, plus free-form `extra`. Every
field Optional: `None` means "the log did not say", which the check treats
differently from a value that disagrees.

The one load-bearing rule, inherited from `dcp_report.sh`: **read the resolved
value from the server's own log, never re-derive it from the launch flags.**
`dcp_engaged` in particular comes from the scheduler's token-sizing line, not
from `SGLANG_UNEVEN_DCP` being in the environment — which is exactly the
inference #340 got wrong. Every arm's `render()` prints its full effective line,
always, so the report itself is the audit trail.

## The check design: short-byte + long-graded

The gate is two tiers because Qwen GDN prefill is not reproducible past ~109
tokens on any backend (registered fact), so there is no byte-identity gate on
long output — that is the exact false-red a bug net must not have.

- **BYTE tier** — a short forced continuation whose leading bytes must match a
  reference exactly. Valid only inside the reproducibility window; a byte probe
  whose reference is too long is a mis-designed probe and the check says STOP,
  never FAIL.
- **GRADED tier** — anything longer is scored by the #274/#284 house grader
  (`scripts/dual_group/r12/graded.py`, reused verbatim, not reinvented) against
  the empirical A-vs-A floor. Text identity is reported as framing only, never
  the criterion — the #360 standard. A trajectory that diverged at a numeric
  tie and still emitted the correct sequence scores the same and stays green.

The band and the byte reference are MEASURED from the baseline arm A-vs-A, not
guessed — the #103 noise-floor discipline.

## Verdict vocabulary (the battery's)

- **PASS** — the crossing is sound; a clean reject is a PASS.
- **FAIL** — a real cross-feature defect: a boot that should have come up
  crashed or hung (the #132 silent-hang class: no ready marker, no fatal,
  timed out → FAIL, not STOP); a boot that resolved a config it did not declare
  (#340); a config that must refuse booting anyway; a coherence score under the
  A-vs-A floor.
- **STOP** — nothing was learned: a missing artifact, an absent grader, a
  declared fact the log did not carry, a byte probe past the repro window.

## The tenant wrapper

`BootMatrixTenant(IdleWorkTenant)`, priority 70 (after training/tuning/probe;
the matrix is a pre-release net, not a continuous consumer). One arm per
segment (the preemption granularity — a preempted arm loses one boot, not a
sweep, and `SubprocessSegment` signals the child's whole process group so a
half-booted TP server never keeps a CUDA context). `cards_wanted=0` (every
visible card: a full TP boot), so it runs only on a fully idle rig. Model path
is input (`--workbench-boot-matrix-model`), never hardcoded — the rig-only
assumption ANALYSE #347 excludes; unavailable by name until configured. NOT in
the default tenant set: opt-in via `--workbench-tenants boot_matrix`.

## Test tally

`test/registered/unit/boot_matrix/`, 63 tests, CPU-only, hermetic:
`test_arms` (arm-list invariants), `test_effective` (log parsing incl. the
#340 falsifier), `test_coherence` (both tiers + the real grader located and
scoring), `test_check` (every verdict path against synthetic artifacts),
`test_sweep` (pure command composition), `test_tenant` (availability, pricing,
argv, and the service-registry wiring). Falsifier-checked: neutering the #340
mismatch catch turns `test_declared_config_mismatch_is_fail` red.

## Card-time estimate for the eventual sweep

~69 min for the full 17-arm set (11 boot ≈ 240–360 s each; 6 reject ≈ 60 s
each, arg-resolution refusals with no weight load) — inside the 60–90 min
window. **Caveat, folded in from #366:** `K_bar1_graphs` is budgeted at 1200 s
because bar1 + NEXTN draft-graph capture ran to 18 min in the #366 window with
a cold graph cache and nothing wedged. Confirm the cache warms across boots
before trusting a shorter estimate; do not let capture blow the window
silently. If the covering set does not fit a given window, drop arms by
ascending cross-feature bug-catching value (a reject arm is cheapest to keep;
G_all_axes is the last to drop) and LOG what was dropped — never silently cap.

## Honest remainder

1. The sweep's GPU layer (`run_arm`, `_run_probes`, `_wait_for_boot`) is
   written but unexercised on a card — that is the later card ticket. The
   hermetic core it feeds is fully tested.
2. The A-vs-A band/reference wiring (`run_arm`'s `reference_probes`/`band`
   arguments) is threaded but the baseline-first ordering that fills them is
   not yet driven by `_main`; the first real sweep must boot `A_default` first
   and seed the band from it.
3. `report_effective` parses the fields the current scheduler prints; a new
   axis (or a log-line rename) needs a parser line here — by design, so the
   effective report can never silently drift from the flags.
