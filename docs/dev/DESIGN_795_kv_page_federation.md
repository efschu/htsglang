# DESIGN #795 — KV page federation (world pool from min-bound to sum-bound)

Branch `fix/784-corridor-arming-credit`, base `d2c5be2dee`. Step 0 of the
user's D-first decision: inventory before design, design as COMPOSITION.
Build decision belongs to the user and this document does not preempt it.

## Recommendation (read this first)

On this rig's own vector-independent capacity measurement (§5):

- **Stage 0 — consume the advisory the runtime already prints (#797).**
  **+10.1 %** global tokens (1 198 400 → 1 318 912), **~87 % of the entire
  available gain**, at the smallest effort of any option: the number is already
  computed and logged on every boot, and nothing consumes it. Planner-side, not
  a hand-pinned vector.
- **Stage D — full page federation — fails its own aufwand/ertrag test on page
  arithmetic: +1.45 %** over a correctly solved static vector (1 318 912 →
  1 338 080), against a large change touching four owner-rule consumers, both
  attention backends and the token-axis gather.
- **D's justification, if it is built, is the capital axis, not the pages.**
  MEASURED: arming capital and KV-backing recovery cannot cross a rank
  boundary — a 127–164 MiB shortfall on one rank stood the phase flip down
  permanently while ~4.7 GB sat stranded on the other two, and long-context
  requests are now refused outright (§7). UNQUANTIFIED: per-sequence and
  runtime variance that no static vector can track. The measured half is real;
  the unquantified half must not be asserted without numbers.

Two of the three published loss estimates for this area — including one of my
own — were circular and are **withdrawn** (§4). The numbers above rest on the
one vector-independent measurement in the log.

*Naming note: the advisory-consumer gap is filed as **#797**, which collides
with the in-tree `#797` PP-admission void fix (`scheduler_pp_mixin.py`). Same
digits, unrelated work; the collision is recorded in the task.*

---

## 1. The diagnosis, restated exactly

`cp_token_context_budget` (`python/sglang/srt/distributed/utils.py:792-802`):

    return min(capacities[r] // vector[r] for r in range(n)) * sum(vector)

Rank `r` owns `vector[r]` of every `sum(vector)` context tokens, so the unit
every rank can fund is `min_r(capacities[r] // vector[r])`. One rank binds; the
others' surplus is **constructively unreachable**.

The docstring of that very function already names the remedy: *"Maximised when
the vector is proportional to the capacities — which is exactly what
`--rank-kv-ratio capacity` installs."* That sentence is the reason §6 exists.

The owner rule itself is a **pure position function**, stated once
(`layers/dcp/owner.py:24-29`):

    owner(L) = rank r with (L % cp_S) in [cp_lo, cp_hi)
    row(L)   = (L // cp_S) * cp_ratio + (L % cp_S - cp_lo)

`cp_lo/cp_hi` come from `cp_token_prefix` (`distributed/utils.py:407-418`) over
a static gcd-coarse integer vector from `resolve_cp_token_ratios`
(`utils.py:534-668`). It is applied identically to **every sequence**; the only
existing non-uniformity is the SWA-hybrid *layer-type* predicate
(`owner.py:275-345`), which is not per-page.

---

## 2. Inventory — building block → file:line → role

Verified with file:line. `EXISTS` means reachable code, not a doc claim.

| # | Building block | file:line | Role for federation | Status |
|---|---|---|---|---|
| 1 | Owner rule, single definition | `layers/dcp/owner.py:24-29`, impl `:348-466` | The one place a table would replace | EXISTS |
| 2 | Owner-bounds refresh registry | `owner.py:58-89` (`register_owner_bounds_consumer`, `refresh_all_owner_bounds`) | Graph-safe rebind seam; a table hooks here | EXISTS |
| 3 | `req_to_token` / radix / allocator | store GLOBAL ids only, `kv_reshard.py:11-14` | Need **no** change under federation | EXISTS |
| 4 | Runtime reshard executor (#297) | `managers/kv_reshard.py`, `on_round` `:586-675`, cutover `:1156-1174` | Moves real bytes under a changed vector | EXISTS |
| 5 | Uneven head gather (#173) | `layers/dcp/comm.py:166-217`, combine `:228-262` | Irregular ownership already solved on the HEAD axis | EXISTS |
| 6 | Token-axis KV gather | `comm.py:314-327` (`all_gather_into_tensor`) | Requires identical per-rank shapes | **GATE** |
| 7 | MLA uneven refusal | `comm.py:57-93` (`_reject_uneven_tp_mla`) | Hard refusal, explicit | **GATE** |
| 8 | Capacity/corridor solver | `distributed/corridor_vector.py:51,149,232` | Solves proportional vectors already | EXISTS |
| 9 | Bandwidth regulator (#705) | `utils.py:805-844`, consumed `model_runner_kv_cache_mixin.py:5720-5758` | Capacity-vs-speed pricing | EXISTS |
| 10 | Flip vector install | `phase_flip_boot.py:85-143`, `phase_flip_runtime.py:1409-1410` | Re-points owner bounds each cutover | EXISTS |
| 11 | #706 canonical read-cut | `mem_cache/hicache_storage.py:1038-1067` / `:1069-1108` | Cross-geometry read, both directions | EXISTS |
| 12 | HiCache owner mask | `managers/cache_controller.py:2187-2192` | 3rd consumer of the owner rule | EXISTS |
| 13 | Prefill spill owner split | `managers/kv_session_offload.py:474-494` | 4th consumer of the owner rule | EXISTS |
| 14 | `page_size == 1` requirement | `cache_controller.py:486-499` (`NotImplementedError`) | Per-page ownership needs it; we ship it | **CONSTRAINT** |
| 15 | KV-head alignment (#116) | `utils.py:748-789` (`_partition_units_kv_aligned`) | Split must not straddle a kv-head group | **CONSTRAINT** |
| 16 | `kv_store_bound` (#355) | `mem_cache/memory_pool.py:175-193` | Index bound on every store | **CONSTRAINT** |
| 17 | VMM elastic backing (#93/#330) | `mem_cache/kv_vmm_backing.py:452-568` | Single-device; `location.id = device_id`, no peer flag `:484-491` | **GATE** |
| 18 | Determinism requirement | `utils.py:559-560` | Vector is a pure function of args so all ranks agree | **CONSTRAINT** |

### Two corrections to the register's inventory

- **#637 "capture lines" is not substantiated.** No `[#637]` commit and no code
  comment ties #637 to CUDA-graph capture. The real graph-safety precedent is
  `kv_reshard.py:46-49` — *"no growth, no address change, no CUDA-graph
  recapture (graph metadata is rebuilt host-side per replay from the refreshed
  backend bounds)"* — wired through the `owner.py:58-89` registry. Any table
  must hook that same refresh point. Use this, not #637.
- **`page_size == 1` is a hard gate, not a preference** (`cache_controller.py:486-499`:
  *"the per-page owner rule needs page_size == 1 (a multi-token page would span
  owner ranks)"*). Our shipped recipe passes `--page-size 1`, so the constraint
  is satisfied today, but it is inherited, not removed, by federation.

---

## 3. The real delta

The register's three hypotheses, resolved:

**(a) Allocator branch for foreign-rank marginal pages — ABSENT.**
Refusing gates: `owner.py:429-437` and `:462-466` compute a *local* compact row
and mask everything not owned; every consumer
(`triton_backend.py:2392`, `flashinfer_backend.py:2552`) indexes only its own
buffer. Cross-rank access exists **only** as explicit byte movement over a
collective (`layers/dcp/comm.py`, `kv_reshard.py`), never as indexing into a
peer pool. `KvVmmArena` cannot help: it is single-device by construction
(`kv_vmm_backing.py:484-491`, `location.id = self.device_id`, peer access never
requested).

**(b) Owner rule position function → rank-uniform table — ABSENT, but the seam
is unusually good.** The rule is defined *once* (item 1) and every backend
imports it rather than copying it; a refresh registry already exists (item 2)
and a cutover already rebinds every consumer (item 4). The work is to make
`owned`/`compact` a **lookup** rather than arithmetic, keeping the result a
deterministic rank-uniform function (item 18) so no collective is needed to
agree on ownership. Note the consumer count is **four**, not two — read, write,
HiCache (item 12) and spill (item 13). Missing one of them is the #60 L3
zero-page corruption class: silently wrong output, not a crash.

**(c) Gather generalisation — PARTIALLY EXISTS.** The head axis already
tolerates irregular ownership (item 5: pad to `max`, all-gather, slice the true
count back; combine uses all-reduce + slice *because* reduce-scatter cannot
express an uneven split). The token/KV-cache axis does not (item 6), and MLA is
hard-refused (item 7). So (c) is "port a solved pattern onto a second axis",
not new invention — but it must be done for **both** backends.

**Out of scope** (register): GDN/Mamba states, rank-local, #745/#773 path.

---

## 4. Metric pin (mandatory, and it invalidated two earlier estimates)

Per-rank token counts are **not comparable** unless `cell_size` matches.
On `boot_restore_797.log`: `:391` PP0 `cell_size=16384` → 431858; `:394` PP1
`cell_size=8192` → 613046; `:971-992` the FINAL GLOBAL pool, 431858, reported
identically by all three ranks.

- Baseline is always the **global** figure.
- Per-rank figures are quoted **only with their `cell_size`**.
- **Federation must account in bytes or cell-normalised units, never in
  tokens** — cell_size differs *between ranks* in the PP stack (16384/8192/8192).

**Circularity warning, applying to two published estimates including one of my
own.** A capacity measured *under* a vector cannot be used to judge that
vector. The rung-4 figure (37.9 % loss) and the row-derived figure (0.074 %
loss) are both circular: pools were sized *from* the vector, so `rows // vec`
coming out balanced measures only that the allocator obeyed. Both are
**withdrawn** as evidence.

---

## 5. The non-circular measurement

The runtime already computes a vector-independent per-rank capacity at uniform
`cell_size=32768` (`boot_restore_797.log:571`, `:586-588`):

    profiled per-rank capacity = [611344, 355784, 370952]

| Configuration | Binding unit | Global tokens | vs current |
|---|---|---|---|
| current vector `29,19,16` | 18725 | 1 198 400 | — |
| runtime's own advisory `29,17,18` | 20608 | 1 318 912 | **+10.1 %** |
| full federation (sum-bound) | n/a | 1 338 080 | **+11.7 %** |

    FEDERATION OVER BEST STATIC VECTOR: +19 168 tokens = +1.45 %

About **87 % of the entire available gain is reachable by changing a static
vector.** Page federation buys the remaining 1.45 %.

And the runtime prints the advice on every boot, `:571` verbatim:

> `Uneven DCP: restart with SGLANG_UNEVEN_TOKEN_VECTOR=29,17,18 to raise
> max_total_num_tokens from 1198400 to ~1318912 (per-rank profiled capacity
> [611344, 355784, 370952]; active vector [29, 19, 16] lea…`

We ship `29,19,16` — traceable to the **retracted** #602 investigation — while
our own boot log recommends better. The advisory exists; nothing **consumes**
it. That gap is **Stage 0, filed as #797**: the planner consumes the `:571`
advisory / profiled capacity inside the solve path. Planner is the single VRAM
authority; a hand-pinned vector is the recovery-pin anti-pattern and is
explicitly **not** proposed here.

A provenance rule falls out of this and belongs in the solver, not in a
comment: **the active vector must never originate from a retracted
investigation.** `29,19,16` does, and nothing currently notices.

---

## 6. Why the shipped recipe never reaches the existing machinery

Read from the live instance's own `server_args` line:

    rank_tp_ratio=None   rank_kv_ratio='coupled'   regime_controller='off'
    kv_reshard_vectors=None   enable_vram_dial=False
    uneven_token_vector='29,19,16'

**Every adaptive capacity mechanism in the fork is unarmed on the recipe we
ship**, and the one capacity flag we do pass is precisely the one that disables
the others. Three independent refusals, each with a line:

1. **Silent downgrade.** `_handle_uneven_tp` (`server_args.py:11083`, called
   `:6741`): with no `--rank-tp-ratio` and a *mode string*, it takes the
   `logger.warning` branch `:11398-11404` and sets `rank_kv_ratio = "coupled"`
   at `:11405` — warning, not raise.
2. **Install suppressed.** `_handle_corridor_kv_ratio` (`:13042`) early-returns
   at `:13051-13052`; and the install is gated at
   `model_runner_kv_cache_mixin.py:5702-5707` on `not
   envs.SGLANG_UNEVEN_TOKEN_VECTOR`, which `--uneven-token-vector` publishes at
   `server_args.py:17412-17418`. Corridor computes, logs, never installs.
3. **Flip overwrites.** `parse_flip_token_vector`
   (`phase_flip_boot.py:85-143`) reads only the env (`:118`) with a
   `--phase-flip-tp-vector` fallback (`:119`) and never consults
   `rank_kv_ratio`; re-installed every cutover at `phase_flip_runtime.py:1409`.

**No guard and no test covers `--enable-phase-flip` + `--rank-kv-ratio
corridor`.** A boot can log a corridor solve, look protected, and route on the
flip vector. That is the single largest unaddressed risk in this area.

Keep two distinct "corridor" concepts apart: `rank_kv_ratio == "corridor"`
(`server_args.py:10181-10193`) versus the spill-planner
`corridor_guard.CORRIDOR_LAW_MIB` (`:5670-5679`). Likewise `#705` in
`boot_layout.py` is an unrelated PP-cut note — ticket-number reuse.

`FEATURE_CATALOG.md:77` omits `corridor` entirely and carries stale line refs.

---

## 7. The capital axis — the actual case for D

Page federation pools **pages**. It does not pool **capital**, and capital is
what visibly failed on this rig.

**Specimen #796, measured `boot_restore_797.log`, 21:19:04Z–21:25:49Z.** Eight
consecutive `tp_to_pp` arm attempts abandoned; PP0 asks staging 2040–2077 MiB
against 1863–1913 MiB spendable (driver free 2682–2732, allocator cache
311–319, reserve 819). Shortfall **127–164 MiB**, stable, never drifting. At
the same moments PP1 reports 1826 MiB free and PP2 2866 MiB free — **~4.7 GB
stranded**. Terminal state at 21:25:49Z: `PHASE-FLIP SEAM UNFUNDABLE — PHASE
FLIP STOOD DOWN (tp_to_pp). 8 consecutive group abandons reached the cap.`

Consequences observed, not inferred: the instance is pinned in TP, the
pool-binding phase can never be sampled (corridor verdict `NO-VERDICT`), and
long-context requests are refused outright — *"Out of memory 9 times in a row
as the sole remaining request in the decode batch. The pool cannot currently
fit this request even alone"* (#679/#694 graceful refusal).

**A second instance of the same shape:** all three ranks logged `KV-BACKING
recovery deferred` (`:21:18:42`–`:21:18:52`), each unable to re-commit rows
because *its own* free VRAM sits at the corridor law — PP0 1091 MiB, PP1 1826,
PP2 2866. Three ranks, three pools, each pinned by a local figure.

So the min-binding is not one defect at the pool level. It is a **pattern**:
capacity, arming capital and backing recovery are each funded per rank out of
that rank's own free memory, with no borrowing. #330's VRAM dial is grow-only
and rank-local (`server_args.py:5340`, `:5405-5406`); #297's reshard moves only
between a pre-declared, boot-reserved ceiling set at an idle boundary
(`kv_reshard.py:46-49`).

**Therefore: a federation design that pools KV pages but leaves arming capital
per-rank-bound leaves the observed failure standing.** Pooling capital is a
mandatory requirement of this design, not a follow-up.

---

## 8. B and C derived from D

**Stage B — re-solve at the flip. Largely BUILT.** `KvReshardRuntime.on_round`
(`kv_reshard.py:586-675`) is a genuine runtime re-solve and re-home, consensus
driven, reachable via `--kv-reshard-vectors` + `--regime-controller act`
(`server_args.py:5411-5429`, `:5339-5347`) or `POST /kv_reshard`. Limits are
the delta: pre-declared ceiling set only, idle boundary only.

The smallest honest B slice is **not** a new subsystem: teach
`parse_flip_token_vector` (`phase_flip_boot.py:118-119`) and the cutover
install (`phase_flip_runtime.py:1409`) to consult `rank_kv_ratio`, and add the
missing guard for `--enable-phase-flip` + `corridor`. Measured in lines.

**Stage C — overflow pages.** Should compose with the #287 pressure ladder and
the #423 striped spill rather than introduce a parallel path; both already
reuse the same position-function owner rule on a different axis, so neither
gives ownership generality for free.

**Ordering that follows from §5, stated plainly:** Stage 0 (#797) first, then
B. Stage 0 captures ~87 % of the measurable gain at essentially no cost, and —
decisively — a boot under a *different* vector produces the **first
non-circular capacity measurement**, which is precisely the input D's benefit
case currently lacks. Doing D first means costing a large change against
numbers we already know to be contaminated.

---

## 9. Aufwand / Ertrag

| Stage | Effort | Measured benefit | Confidence |
|---|---|---|---|
| **0 (#797)**: consume the existing advisory vector (planner) | ~0 (restart) | **+10.1 % — ~87 % of all available gain** | HIGH — runtime's own number |
| B: flip consults `rank_kv_ratio` + guard | small, localized | keeps the solve across cutovers | HIGH — gates named |
| C: overflow pages | medium | unquantified | LOW |
| D: full page federation | large, 4 consumers + both backends + token-axis gather | **+1.45 %** over best static vector, plus the capital axis | page arithmetic HIGH and small; capital axis MEASURED but not yet costed |

D is not justified by its token arithmetic. It is justified, if at all, by §7.

---

## 10. Falsifiers

Each would refute a load-bearing claim; none is expensive.

1. **Vector-independence.** Boot without `--uneven-token-vector`, with
   `--rank-kv-ratio capacity`. If the profiled capacities move materially, the
   §5 numbers are themselves vector-contaminated and the +1.45 % is wrong.
   *This is the named falsifier item the circularity in §4 demands.*
2. **The +10.1 % is real.** Boot with `29,17,18`. If the global pool does not
   reach ~1 318 912, the runtime's advisory is miscomputed and §5's ordering
   collapses.
3. **Capital pooling is the true binder.** If a boot that fixes only the
   arming floor restores the flip and lifts the long-context refusal, then §7
   is the whole story and D's page arithmetic is decoration.
4. **Four consumers, not two.** Convert the owner rule to a table behind a flag
   and run with HiCache and spill enabled. Silent wrongness (not a crash) in
   `cache_controller.py:2187-2192` or `kv_session_offload.py:474-494` confirms
   the consumer count; a #60-class zero-page read is the signature.
5. **Token-axis gather.** Attempt an uneven token-axis all-gather via the head
   axis' pad-and-slice pattern (`comm.py:166-217`). If it is not
   cuda-graph-safe there, (c) is harder than "port a solved pattern".
6. **#602's rule applies here.** Never auto-arm a redistribution from a modeled
   number: census boots showed a 23–29 % modeled-vs-measured gap (471303 vs
   361566). Any federated sum-bound capacity must be checked against a measured
   per-rank transient before it arms anything.

---

## 11. Open items

- Reachability of `KvReshardRuntime` end-to-end on the shipped recipe is
  **unverified** (it is unarmed today, §6).
- Whether the pre-declared ceiling-set restriction (`kv_reshard.py:46-49`) is
  architectural or merely current scope.
- Whether `_all_gather_dcp_kv_cache`'s equal-shape requirement is unconditional
  at all four call sites, or whether some caller pre-pads.
- Whether `mem_cache/dsa_cache_layer_split.py`'s `_get_layer_owner_rank` is a
  second, independent layer-owner axis.
- `FEATURE_CATALOG.md:77` needs `corridor` added and its stale line refs fixed.
