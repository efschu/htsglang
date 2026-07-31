# DESIGN #330 — Runtime per-card VRAM budget dial + KV capacity re-raise

Branch `feat/vram-dial-330`, base `5db315ae5c` (integration/r3-probe-next2).
Two halves, one mechanism: a running server can (a) lower or raise one
card's VRAM budget at runtime, with released memory REALLY returned to the
driver (visible in nvidia-smi, allocatable by another process), and (b) GROW
`max_total_num_tokens` after a #297 reshard whose new vector geometry allows
far more KV (the C re-raise, #320: 6.18x prognosis on `2,11,10`).

## 1. Mechanism: one VMM arena, chunked commit, tail decommit

The #93/#102 machinery already in-tree (`mem_cache/kv_vmm_backing.py`:
`KvVmmArena` + `KvVmmBufferOwner`) reserves a CUDA virtual-address range once
(`cuMemAddressReserve`), exposes it as a `torch.cuda.MemPool` bump allocator,
and maps physical pages underneath on demand (`cuMemCreate`/`cuMemMap`).
Addresses never change, so decode CUDA graphs captured over these tensors
keep replaying without re-capture. #330 extends it, it does NOT build a
second VMM layer:

* `commit_range` gains an optional `commit_chunk_bytes`: physical handles
  are created in fixed-size chunks (default 16 MiB, granularity-aligned)
  instead of one arbitrary-size handle per extension. Chunks are what makes
  release possible: `cuMemUnmap` operates on whole mappings only.
* new `decommit_range(offset, keep_bytes)`: unmaps and `cuMemRelease`s every
  whole chunk above `keep_bytes` (rounded up to the next chunk boundary, so
  live rows below the keep point are never lost). This is DRIVER-level
  release — the freed bytes leave the process, show up in nvidia-smi, and
  are `cudaMalloc`-able by another process. Freeing inside torch's caching
  allocator would be none of those things.
* `KvVmmBufferOwner.set_capacity_tokens(n)`: converge every buffer's backing
  to the span of `n` tokens — commit upward, decommit downward. Reports
  actually-released bytes (decommit is chunk-granular, so actual backing may
  exceed the model by < 1 chunk per buffer; the reported number is the real
  one).

## 2. What is dialable

The dialable portion of a rank's VRAM is exactly the VMM-backed KV pool
tail: the target model's full-attention KV pool (DCP row-sharded) plus the
NEXTN/EAGLE draft KV pool (token-complete, raw slot space). Weights, CUDA
graphs, the GDN/Mamba state pool, activation workspaces and allocator
metadata are the PINNED FLOOR — measured once per rank at runtime build as

    floor_r = nvml_process_used_bytes(rank's card) - vmm_backed_bytes_r

and all-gathered so every rank knows every floor (replicated inputs, rule 1
of the #287 pattern). A dial below `floor_r + minimum KV` is rejected at the
API with the exact numbers (floor MiB, requested MiB, minimum viable MiB).
The floor is a boot-time snapshot; post-boot non-KV allocations (there are
none on the supported lane) would show up as honest OOM, not silent capping.

## 3. Capacity model (pure function of replicated state)

Per rank r, with per-row KV bytes `trb_r` (target pool, all full layers,
K+V) and per-token draft bytes `drb_r`, current vector v (S = sum(v)):

    kv_budget_r  = budget_r - floor_r
    unit_max_r   = (kv_budget_r - page_slack_r) // (v_r * trb_r + S * drb_r)
    unit_max_r   = min(unit_max_r, va_reserve_cap_r)         # VA reservation
    C_target     = min_r(unit_max_r) * S                     # group ceiling
    C_target     = min(C_target, user --max-total-tokens cap)

Budgets and floors are replicated (broadcast RPC + one-time gather), the
vector is the process-global `get_cp_token_ratios()`, so every rank computes
the same `C_target` locally; the consensus reduction merely VERIFIES
equality (desync = loud error on every rank, never a hang).

The VA reservation is sized at boot for the best case: rank r reserves
`max_v dcp_compact_pool_rows(C_v, S_v, v_r)` rows where `C_v` is the
per-vector achievable ceiling from the #297 fitted-ceiling min-reduce in
`_apply_token_constraints` (stashed as the boot capacity plan). VA is free
until committed; physical backing at boot stays exactly the fitted ceiling,
so a dial-less boot commits the same bytes as today.

## 4. Rank-uniform protocol (the #287/#297 consensus pattern)

`KvCapacityRuntime` (managers/vram_dial.py) runs in the scheduler loop of
every TP rank, ABOVE the #297 reshard block. Flag unset = not constructed,
zero collectives, byte-identical behavior.

Every `consensus_interval`-th round — gated by the replicated round counter,
never by local state — every rank enters ONE bounded MIN-reduction
(#312 `bounded_collective`; dead peer = loud `PeerLostError`) with payload
`(armed, ready, epoch, op_seq, C_target)` packed as `(x, -x)` pairs:

* `epoch` (completed capacity ops): equality-checked always.
* `op_seq` (budget RPCs applied): min != max is LEGAL delivery skew ->
  uniform hold, wait for the next boundary.
* `armed`, `ready`: MIN semantics, disagreement = uniform wait.
* `C_target`: equality-checked once every rank is armed at the same op_seq;
  mismatch raises the same `KvCapacityError` on every rank.

Arming:
* GROW (C_target > C_cur) arms AUTOMATICALLY within the budgets — this is
  the C re-raise: after a #297 cutover installs a better vector, every
  rank's next boundary computes a higher C_target from the same replicated
  state and the group grows in lock step. `C_target` is additionally
  clamped by the CURRENT vector's boot achievable ceiling (`_caps` from the
  #297 min-reduce, which folds in the hybrid mamba physical ceiling and the
  profiled physical capacity) — the runtime's byte math must never outgrow
  what the boot sizing knows to be physically addressable.
* Budget stance: the default boot budget is the NATURAL footprint
  (floor + backed KV), so capacity never grows beyond the boot allocation
  without an operator action — one dial-up (`budget_mib` is clamped to the
  effective VA ceiling) or `--vram-budget-mib` authorizes the headroom, and
  from then on the re-raise is automatic at every reshard. This is
  deliberate: auto-claiming all free VRAM would silently swallow the
  operator's activation reserve (the 2700-MiB GDN-prefill-scratch lesson)
  and would squat on co-tenant memory.
* SHRINK (C_target < C_cur) arms only from an explicit dial RPC (it flushes
  the radix cache; the runtime never destroys cache spontaneously).

Commit (all ranks, synchronously, inside the same scheduler round, fully
idle — `ready = is_fully_idle()` for both directions; growth under load is a
named follow-up, kept off to avoid racing the overlap worker thread):

* GROW: pool backing rows -> `rows(C_new, v, r)` (newly committed rows are
  zeroed — a fresh boot's pools are zeros and flushed state must match),
  draft pool -> `C_new` tokens, allocator free list appended with slot ids
  `(C_old, C_new]`, every capacity snapshot refreshed (section 6).
* SHRINK: flush (tree reset, req_to_token clear, allocator resize+clear,
  draft cache pool clear — the `flush_cache` recipe), pool backing
  decommitted to `rows(C_new, v, r)` on every rank, ceilings refreshed.
  Shrink is allowed to release on every rank; only the dialed rank MUST
  (its budget binds); the others' backing target still follows `rows(C_new)`
  so the group model stays a pure function.

Address stability: growth maps pages inside the boot VA reservation, shrink
unmaps tail pages below the new allocator ceiling — no tensor moves, no
re-capture. A configuration that cannot keep addresses stable does not
exist on this path by construction; configurations outside the supported
lane are rejected at boot (section 7).

## 5. Interplay with #297 resharding (fit guard)

Growth breaks the boot fitted-ceiling invariant on purpose: after growing
to `C_v1`, another vector v2 with `C_v2 < C_v1` may need more rows at the
CURRENT C than a rank has backed. The reshard runtime therefore gains an
optional `fit_check(target_vector)` (wired to the capacity runtime): rows
needed at `C_cur` under the target vector are compared against the
REPLICATED backing model on every rank; a target that does not fit turns
`ready` off uniformly with a loud hold message naming the required dial-down
— the reshard waits instead of crashing or corrupting. Dial down (shrink),
then reshard, then the auto re-raise grows into the new geometry.

The #287 pressure ladder composes: a dial-down raises occupancy over the
new (smaller) capacity, and the ladder's admission/spill rungs see it
through the same replicated sample. The pause rung (#305) is NOT built here
and nothing in this design blocks it.

## 6. Snapshot refresh (growth/shrink must reach every reader)

`max_total_num_tokens` snapshot holders refreshed at commit, in the same
round: `scheduler.max_total_num_tokens`, every `ModelRunner` in
`model_runner_list` (target + draft) incl. `memory_pool_config`, the
allocator (`size`/free list), `pool_stats_observer`, `invariant_checker`,
`load_inquirer`, `kv_events_publisher`. The DP controller's cached copy is
out of scope (DP is rejected at boot on this lane); the Prometheus constant
`sglang:max_total_num_tokens` is re-emitted when the collector allows it.

## 7. Supported lane and refusals (Stage 1)

Supported: CUDA + weighted uneven DCP (`SGLANG_UNEVEN_DCP=1`, weighted,
`--rank-kv-ratio`) + hybrid-linear pool family (`HybridLinearKVPool`) —
the fork's serving lane, same as #297 Stage A. `--kv-reshard-vectors` is
optional (without it the plan holds only the boot vector: dial works,
cross-vector re-raise has nothing to raise to).

Hard boot-time refusals (each names its reason): non-CUDA, MLA, hybrid-SWA,
memory saver, PD disaggregation, hierarchical cache storage,
kv-session-offload (host rows sized from boot C), weightless-KV ranks,
dual-group lane, DP > 1, hisparse. These snapshot or derive from the boot
capacity in ways Stage 1 does not migrate; each is a named follow-up, not a
physical impossibility.

## 8. Ledger integration (#305-M1)

Every rank holds one entry in the VRAM ledger
(`registry/ledger.py`, lease files under `/run/htsglang/vram/<uuid>.json`):
tenant `<HTSGLANG_TENANT_ID | srt-<port>>:rank<r>` on its card's NVML UUID,
`reserved_bytes = budget_r`, HOT, heartbeated at consensus boundaries.
A dial-down re-acquires with the smaller budget in the SAME commit, so an
external tenant (diffusion job, training tenant) sees the freed bytes
through `available_bytes()` immediately and can claim them honestly.
Ledger I/O failures log loudly but never take serving down.

## 9. API surface

* `POST /vram_budget` (ADMIN_OPTIONAL, like /kv_reshard): body
  `{"device": "<uuid | cuda:N | rank:N>", "budget_mib": B}` or
  `{"device": ..., "release_mib": X}` / `{"release_fraction": f}` (fraction
  of the dialable span `budget - floor`), or `{"query": true}` for state.
  Response: verdict + per-rank `{budget, floor, backed, C, epoch}` MiB
  numbers. Below-floor requests are rejected HERE with exact numbers.
* Server args: `--enable-vram-dial` (constructs the VMM lane + runtime),
  `--vram-budget-mib r0,r1,...` (initial per-rank budgets; profile-time
  clamp + exact reconcile at the first boundary),
  `--vram-dial-consensus-interval N` (default 8).

## 9b. Known limit (card-run finding, open)

Slot ids ABOVE the boot fitted ceiling carry an open defect: sustained
10-way 30k-token concurrency after growth (held tokens ~0.88 of the grown
ceiling) hits a `store_kvcache` `index < size_limit` device assert on a
2-kv-head rank. The bound passed is live (`pool.size + page`), and the
owner-rule row/draft bounds are arithmetically covered, so the escaping
index implicates a path first exercised when allocation reaches the grown
id region (low ids are handed out first; a 260k-token sequential fill with
peak 80k concurrent was clean). Needs a dedicated debug window
(compute-sanitizer, SGLANG_POISON_POOL_DATA). Until resolved: dial-down/up
WITHIN the boot ceiling is fully validated; growth beyond it carries this
known limit. Registered in INTEGRATION_R3_VALIDATION (#330 section).

## 10. Deliberately NOT built

* The #305 pause rung (suspend a tenant wholesale) — separate task; this
  design only guarantees not to block it (the dial's floor is exactly what
  a pause would evacuate).
* GPU hotplug / changing the card set at runtime.
* Growth under load (commit is idle-boundary only, see section 4).
* Plain-MHA (non-DCP) dial lane — the mechanism generalizes (the VMM owner
  is pool-family-agnostic), but Stage 1 validates one lane end to end.
* Uncommitting the pinned floor (weights/graphs) — that is #305 territory.
