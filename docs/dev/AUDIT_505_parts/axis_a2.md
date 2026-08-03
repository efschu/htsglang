# Axis A2 — warning-instead-of-error: mem_cache / managers / model_executor / distributed+barlink / dcp / memtier / disagg

Desk audit, nothing executed, no GPU. Base commit `d653405223`, branch
`docs/silent-wrongness-505`, worktree `/spinning/wt-505-silent`. Method copied
from `AUDIT_500_mechanism_reach.md` §§1-6: every row cites `file:line` and
quotes the operative line verbatim; a row that rests on a comment or docstring
rather than executed code says so.

Scope split: sub-auditor A1 holds `model_loader/`, `speculative/`,
`layers/moe/`. This part holds `mem_cache/`, `managers/`, `model_executor/`,
`distributed/` (incl. the barlink family), `layers/dcp/`, `memtier/`,
`disaggregation/`.

## The class of defect

A site DETECTS an anomaly, logs a WARNING (or nothing at all), and CONTINUES,
where the resulting state is silently WRONG rather than merely degraded. The
bar for **DANGEROUS** is a concrete answer to: *what exactly would be silently
wrong, and through which observable would a user NOT notice it?* Sites that
cannot answer that concretely are **BENIGN** — the list below is deliberately
short rather than padded.

- **BENIGN** — genuine compat shim, optional feature absent, cosmetic, or a
  path with no production reach.
- **DEGRADED-LOUD** — behaviour changes, but visibly, and the warning names it
  accurately.
- **DANGEROUS** — the wrongness stays silent.

## Coverage

| dir | grep total | reviewed at message level | opened at source |
|---|---|---|---|
| `mem_cache/` | 96 | 96 | 8 |
| `managers/` | 84 | 84 | 11 |
| `model_executor/` | 59 | 59 | 6 |
| `distributed/` | 86 (77 in `device_communicators/`) | 86 | 19 |
| `layers/dcp/` | 1 | 1 | 1 (+ `collective_guard.py` read whole) |
| `memtier/` | 1 | 1 | 1 |
| `disaggregation/` | 83 | 83 | 4 |
| **total** | **410** | **410** | **50** |

Extraction: `grep -rn "logger\.warning\|warning_once\|warnings\.warn\|logger\.warn(" --include=*.py <dir>`
(counts above), each re-run with `-A3/-A4` so all 410 were read with their
message text and the immediately following statement. 50 were then opened in
their surrounding logic. A second sweep extracted the *silent* shapes
(`except Exception:` followed by `pass`/`continue`/`return None`) across the
same directories: 51 hits, 8 opened.

**Not reached** (stated as a gap, not a clean bill of health):
- `mem_cache/storage/{flexkv,eic,hf3fs,nixl,aibrix_kvcache,simm,umbp,lmcache}`
  — ~40 warning sites, third-party storage backends, read at message level
  only. They are I/O-failure→cache-miss shapes, which is why they were
  deprioritised, but none was opened.
- `disaggregation/{mooncake,mori,nixl,ascend,fake}` — 79 of the 83
  disaggregation hits. Only `topology.py` was opened. The mooncake/mori KV
  transfer failure paths are the plausible place for a second finding of the
  A2-04 shape and were NOT audited.
- `managers/` multimodal, tokenizer, and detokenizer sites (~30) — read at
  message level, not opened.
- `model_executor/` NPU/Ascend and CPU paths.
- The `except: pass` sweep covered only bare `pass/continue/return None`;
  fallbacks that substitute a *default value* were followed only where a
  warning already pointed at them.

## Table

| file:line | site (symbol) | class | what would be silently wrong | fix pattern |
|---|---|---|---|---|
| `managers/kv_session_offload.py:2204-2207` | `KVSessionOffloadManager.__init__`, spec-in-tick scratch reservation | **DANGEROUS** | `self.allocator.size -= int(self.mtp_resident_slices)` inside `try: … except Exception: pass`. On the composite allocators the setter is a NO-OP (`@size.setter … pass`, `mem_cache/multi_ended_allocator.py:1764-1766` and `:1974-1976`), so the subtraction silently does nothing — no exception, nothing to catch. The slots ARE out of circulation (`self.allocator.alloc(...)` at `:2195` succeeded) but the pool still counts them, so the invariant the comment at `:2198-2203` names ("available + evictable + protected + session_held + uncached == total, total = allocator.size") is permanently off by `mtp_resident_slices`. `logger.info("… reserved %d draft-read scratch slots …")` at `:2208` then reports success for the half that failed. | (iii) independent state probe: read `allocator.size` back after the write and refuse by name when it did not move — `if self.allocator.size != _before - self.mtp_resident_slices: raise ValueError(...)`. Drop the bare `except`. The composite allocators must either grow a real carve-out API or be refused for this feature by name. |
| `model_executor/model_runner_kv_cache_mixin.py:1208-1209` | `note_post_capture_leftover` / `foreign_residue_warning` | **DANGEROUS** | `if foreign_msg is not None: logger.warning("%s", foreign_msg)` — and then the boot persists the correction anyway. The warning text itself states the consequence: *"the correction persisted for the next boot will under-book the KV pool"* (`:1078-1079`). Nothing in the written cache records the pollution (`grep -n foreign` over the file returns only `:1198-1209`), and the reader (`:896-910`) validates only shape and `mlp_vector`. So the wrongness materialises in the NEXT boot, in a different log, as a smaller `max_total_num_tokens` with provenance "cached" and no reason attached. | (i) named hard error / refusal at the site: do not persist when `foreign_b > 0` unless `SGLANG_MEASURED_KV_BUDGET_CTX_ALLOWANCE_MIB` was raised deliberately; alternatively write `"provenance": "polluted"` into the cache and have the reader refuse it by name the way `:908` already refuses `malformed`. |
| `managers/kv_session_offload.py:3426-3434` | `KVSessionOffloadManager.try_spill` | **DANGEROUS** | `if n_own > self.region_tokens:` warns and `return False` → the caller falls back to stock retraction (`managers/scheduler.py:4051`). `n_own` is `int(dev_idx.numel())` from `owned_device_indices(..., lo=self.lo, hi=self.hi, dcp_rank=self.dcp_rank)` (`:3415-3425`) — a RANK-LOCAL quantity under the weighted uneven-DCP owner rule, which is the whole point of `--rank-kv-ratio`. Every neighbouring decline in the same function is annotated rank-uniform (`":3344 Rank-uniform: the overhang count is replicated"`, `":3400 Replicated inputs -> rank-uniform verdict"`); this one is not, and it is the only one whose input is per-rank. A divergent spill/retract verdict means the ranks build different `ScheduleBatch`es — the file's own doctrine at `:501` is *"RANK-UNIFORMITY (this fork: divergence == NCCL hang, not a wrong number)"*. The identical condition in the sibling prefill-spill path RAISES: `raise RuntimeError("… needs {n_own} host rows > region {self.region_tokens}; the admission gate should have rejected it.")` (`:3874-3880`). | (i) raise like the `:3874` twin — the region is sized `max_ratio`-wide precisely so this cannot happen (`:3013 "every region is max_ratio-sized (holds any rank's per-rank shard of a full-context session)"`), so a hit is a broken invariant, not a runtime condition. If a soft decline is wanted, min-reduce it exactly as the sibling module already does: `reduce_extra()` → `transfer_verdict(min_done, min_ok, abandoned)` (`managers/kv_session_spill_destination.py:621-626`, `:309-330`). |
| `disaggregation/topology.py:421-427` | `_card_totals_mib` (CUDA→NVML bridge) | **DANGEROUS** | `except Exception as exc: logger.warning("PD topology: CUDA->NVML index bridge unavailable (%s); assuming identical enumeration orders.", exc)` then `mapping = {}` → `reindex_totals_cuda_order(nvml_totals, {})` is the identity. CLAUDE.md: *"Device identity strictly via the IdentityMap … torch order != NVML order on this rig"*. With the identity assumption, each card's VRAM total is attributed to the wrong rank on any rig where the orders differ — which is this rig. The only downstream guard is itself warn-only (`for warning in check_vram_feasibility(plan): logger.warning("PD topology: %s", warning)`, `:742-743`), so a plan is admitted whose feasibility was computed against the wrong card. Observable: none — the PD servers boot and OOM later, or silently get a smaller pool. The function's own docstring demands the opposite for the sibling case: *"Returns None when NVML is unusable; the caller must then say so instead of silently skipping"* (`:389-391`). | (i) return `None` here too (the docstring's contract), so the caller reports "no card totals" instead of fabricating them. AUDIT_331 is the precedent; `registry/nvml.py`'s IdentityMap is the sanctioned source. |
| `distributed/device_communicators/barlink.py:384-400` | `_build_transport` | **DANGEROUS** (unproven direction; see note) | `except Exception as e: … logger.warning("barlink: group %r does NOT get the requested transport %r …")` then `return None` → the gloo plane. The decision is per-rank and there is NO group agreement step: `_NO_FALLBACK = frozenset({"device", "host"})` (`:327`) deliberately excludes `bar1` and `matrix`, so those two are exactly the transports that may fall back on one rank alone. A one-rank fallback is the `#94/#194/#312/#431` family. Honest qualifier: bar1 bring-up itself contains collectives (`dist.all_gather_object(...)` at `barlink_bar1.py:1953`), so a failure raised BEFORE that point desyncs into a hang rather than a wrong number; a failure after it, or in `matrix`'s own probe path, leaves the group split with no diagnostic. Nothing in the module reconciles the outcome. | (i)+(iii) the in-tree pattern already exists twice: `parallel_state.py:975-992` all-gathers the local availability of custom all-reduce and disables it on EVERY rank when it diverges; `model_executor/model_runner.py:1365-1369` does the same for CUDA-graph plans. Apply that shape to `_build_transport`'s result before the communicator is used. |
| `distributed/device_communicators/barlink_path_dispatcher.py:468-478` | `refine_transport_choice` | **DANGEROUS-latent, reach zero today** | `if hint == HINT_GLOO: return None` demotes one message class to the gloo plane on a per-rank decision; the module docstring only *asserts* the saturation sensor "MUST be group-uniform" (`:246-248`) — a comment, not a predicate. Reach today is zero: `dispatcher_enabled()` reads `SGLANG_BARLINK_PATH_DISPATCHER` default `"0"` (`:428`) and a fresh dispatcher has an empty registry, so every decision is `status_quo` (`:431-443`, and catalog §7 says the same). Recorded because the wiring slice would activate it. | (i) when #279's measured slice lands, the decision must be reduced across the group (or derived only from replicated inputs) before it can change a transport; the `barlink_uniformity.first_divergence` recorder (`barlink_uniformity.py`) is the standing instrument for exactly this and is OFF by default (`:205`, #500-B20). |
| `model_executor/runner/decode_cuda_graph_runner.py:1243-1245`, `:1263-1265` | kvso spill-graph / C4 attention selftests | **DANGEROUS (instrument)** | `logger.warning("kvso spill-graph selftest raised (ignored): %r", _sess_e)` — a self-test whose exception is swallowed. CLAUDE.md rule (2): an instrument's verdict counts only after it passes a can-discriminate check. Here a *crashing* instrument is indistinguishable from a *passing* one in every downstream consumer: capture proceeds, and the only trace is one WARNING in a worker log — and `model_executor/forward_peak.py:14-17` records that worker `logger.warning` lines provably do not reach the server log on this rig. | (iii) the selftest must either raise (it is a gate) or record a machine-readable NOT-RUN verdict next to its PASS/FAIL, on the channel that has evidence behind it (the per-rank JSON dump named in `forward_peak.py:16-17`), so "no verdict" is never read as "passed". |
| `mem_cache/multi_ended_allocator.py:297-313` | `SubPoolAllocator.restore_state` | **BENIGN today (comment-asserted invariant, no production caller)** | `logger.warning("MultiEndedAllocator.restore_state: %d relocation(s) recorded inside a backup window … Eager compaction is not fully reversible; SGLang's spec path should not produce a free() inside a backup window.")` then `del self._inverse_history[n_inverse:]` — the records are DISCARDED and the relocations are not undone, so the rolled-back watermark would describe pages that moved. The invariant is asserted in a comment only (`:289-290 "Spec-decode allocates only inside a backup window (no free), so `_inverse_history` doesn't grow under correct usage."`). Reach: `grep -rn "backup_state=True"` over `srt/` returns nothing and the only `restore_state` callers outside `mem_cache/` are `hardware_backend/npu/dsv4/dsv4_allocator.py:727-735`, so no CUDA production path reaches it. | Per CLAUDE.md rule (3) the comment is a TESTABLE CLAIM: pin it with a unit test (a `free()` inside a backup window must be impossible), or convert the warning into a raise now that nothing can trip it. |
| `mem_cache/unified_radix_cache.py:2044-2055` | `write_backup_storage` (uneven-DCP owner mask) | **DEGRADED-LOUD** | `if device_value is None: logger.warning("[uneven-dcp hicache] skipping storage backup of node %d: device indices gone, page ownership unknown."); return`. The rank-local skip is argued at the site: *"Skipping is safe: batch_exists prefix semantics just truncate at the first missing page, identically on every rank (rank-shared file names)"* (`:2046-2049`). The whole node is skipped, so no partially-written rank-shared page is produced — the argument holds as written. | none; keep. (Rests on the comment for the `batch_exists` truncation semantics, which was not verified in the storage backend — flagged.) |
| `managers/kv_session_spill_destination.py:740-749` | `SpillDestinationController._worker_loop` | **DEGRADED-LOUD (exemplar)** | `logger.warning("kv-session-offload destinations: %s transfer of rid=%s failed on rank %d: %r", …); ok = False` — a per-rank I/O failure, but `ok` is published into `_io_ok` (`:756`) and MIN-reduced group-wide before any verdict (`reduce_extra`, `:621-626`; `transfer_verdict`, `:309-330`). A single rank's failure fails the transfer on every rank deterministically. | none — this is the pattern the two DANGEROUS rank-local rows above should copy. |
| `distributed/parallel_state.py:983-992` | `_reconcile_custom_allreduce` | **DEGRADED-LOUD (exemplar)** | *"Custom allreduce enablement diverges across group '%s' … disabling it on every rank so no collective is gated on a rank-local state."* Preceded by `torch.distributed.all_gather_object(gathered, bool(local_ok), group=self.cpu_group)` (`:976-978`). | none — cite as fix pattern. |
| `model_executor/model_runner.py:1365-1369` | CUDA-graph plan reconciliation | **DEGRADED-LOUD (exemplar)** | *"CUDA graph plan differs across the group for the %s phase (ranks resolved %s); disabling it on every rank so the collective sequence stays rank-uniform."* | none — cite as fix pattern. |
| `layers/dcp/collective_guard.py:111-122`, `:131-136` | `guard_dcp_step` | **DEGRADED-LOUD (exemplar)** | A count divergence raises `RuntimeError("weightless-KV anti-hang guard: DCP collective COUNT divergence …")` behind a bounded `monitored_barrier`; an order/type divergence raises after an `all_gather` of the step signature. Enabled only for the weightless lane and turned off around weightless decode graphs (`model_runner.py:4121 set_guard_enabled(not _wl_decode_graphs)`). | none — the in-tree instrument for the rank-local family; the A2-03/A2-05 sites are outside its reach. |
| `distributed/device_communicators/barlink.py:644-654` | `BarlinkCommunicator._select` fallback notice | **DEGRADED-LOUD** | Once per `(op, size class)`: *"… does NOT cover %r at %d bytes -- falling back to the host-staged layer … this run's numbers for this size are NOT %s numbers."* `handles()` is argued rank-uniform at `barlink_bar1.py:794` ("depends only on group-uniform state") and the a2a size check is explicitly widened to the group max (`barlink.py:1134-1137`). | none. (The rank-uniformity of `handles()` is a comment; `barlink_bar1.py:689-691` concedes *"rank-uniform, and nothing enforces that"* for the neighbouring case.) |
| `distributed/device_communicators/barlink_matrix_transport.py:316-325` | `window_for` clipping | **DEGRADED-LOUD** | Clips a group's BAR1 window and says payloads above it "fall back to the gloo layer without further notice". Per-rank clipping is reconciled by the group-wide MINIMUM at `barlink_bar1.py:1948-1978` (`common = min(proposals)`, `raise Bar1Unavailable` at 0). | none. |
| `distributed/device_communicators/barlink_matrix_transport.py:218-224` | `bar1_free_for` NVML identity | **DEGRADED-LOUD** | `except DeviceOrderUnresolvedError` → window sized from the sysfs GROSS aperture, provenance string `"sysfs-gross"` returned so the caller can see it. Contrast `disaggregation/topology.py:422`, which fabricates an identity mapping instead. | none — this is the correct shape of the A2-04 defect. |
| `distributed/parallel_state.py:735-743` | barlink ACHIEVED notice | **DEGRADED-LOUD** | Reports `requested` vs `ACHIEVED` and states the measurement is mixed. It is the *report* of the `barlink.py:393` fallback, not a second decision. | see A2-05 (the missing piece is agreement, not reporting). |
| `model_executor/model_runner_kv_cache_mixin.py:908-910` | measured-KV-budget cache read | **DEGRADED-LOUD** | `logger.warning("Measured KV-budget cache %s is malformed; ignoring.")`, `provenance = "malformed"`, `return 0` — a named provenance the rest of the path can read. | none — this is the shape A2-02's writer should adopt. |
| `model_executor/model_runner_kv_cache_mixin.py:4966-4972` | `--rank-kv-ratio speed` degradation | **DEGRADED-LOUD** | Falls back when no bandwidth scores exist; catalog §1 documents the degradation (`server_args.py:9615`). | none. |
| `model_executor/pool_configurator.py:1018-1023` | `--swa-pool-sizing` no-op notice | **DEGRADED-LOUD** | A user-set cap is ignored, and the line says so and names the selected configurator. | none. |
| `memtier/profile_store.py:225-229` | `PROFILE_TRUST_ENV` override | **DEGRADED-LOUD** | *"the profile's numbers are being read on hardware that did not produce them"* — the operator asked for it via an env var; the line names the consequence. | none. |
| `mem_cache/mamba_radix_cache.py:577-584`, `:730-737` | mamba checkpoint interval | **DEGRADED-LOUD** | Off-grid tracked state is dropped (`cache_len = 0`) / the request is not cached. Both refuse to *store* rather than storing a mismatched pair; `:596-600` argues why rounding down would be wrong. | none. |
| `mem_cache/hiradix_cache.py:1188-1192`, `:1301-1307` | `write_back` drop / `load_back` failure | **DEGRADED-LOUD** | Host-pressure drop and load-back failure both degrade to a cache MISS; correctness of the served tokens is unaffected. | none. |
| `managers/kv_session_offload.py:2219-2231` | spec-in-tick scratch unavailable | **DEGRADED-LOUD** | Spilled sessions fall back to the plain host tick; the reason and the flag to set are named. | none. |
| `managers/kv_session_offload.py:3311-3321`, `:3372-3382` | `try_spill` snapshot / bookkeeping declines | **DEGRADED-LOUD** | Both decline on REPLICATED inputs (annotated at `:3344`, `:3400`) and fall back to stock retraction. | none. |
| `managers/kv_pressure_runtime.py:500-504` | dcp_ratio rung stays planned-only | **DEGRADED-LOUD** | The rung is not armed and the operator is told which vectors to declare. A refusal, not a silent flip. | none. |
| `distributed/parallel_state.py:817-820`, `833` | custom / quick all-reduce setup failure | **BENIGN** | Optional accelerator absent; `:983` reconciles the group afterwards. | none. |
| `distributed/parallel_state.py:2636-2639`, `:2671-2677`, `distributed/utils.py:49-52` | global TCPStore | **BENIGN** | NIXL-only coordination; consumers check for `None`. | none. |
| `distributed/parallel_state.py:1656-1659`, `:3460-3462` | world-size-1 in-place all-gather, torch<2.5 host cache | **BENIGN** | Efficiency note; version shim. | none. |
| `distributed/device_communicators/{custom_all_reduce_utils,quick_all_reduce,pymscclpp,torch_symm_mem,triton_symm_mem_ag}.py` (17 sites) | import/probe failures | **BENIGN** | Optional dependency or capability absent; the feature disables itself uniformly (import-time, hence identical on every rank). | none. |
| `distributed/device_communicators/barlink_{bar1,ucx,host,shm,device}.py` teardown sites (`bar1:746`, `:4111`, `:4225`, `ucx:1983`, `host:1172`, `shm:72/79`, `device:822`) | close/release/unregister failures | **BENIGN** | Teardown after the real error; `bar1:465` says so verbatim (*"teardown must never mask the real error"*). | none. |
| `distributed/device_communicators/barlink_liveness.py:142`, `barlink_abort_gate.py:106` | env parse | **BENIGN** | Malformed env value → documented default, value echoed. | none. |
| `mem_cache/storage/**` (~40 sites) | third-party backends | **BENIGN (message level only)** | I/O failure → cache miss. Not opened — see coverage gap. | none. |
| `disaggregation/topology.py:398-399`, `:407-408` | NVML unavailable / query failed | **BENIGN** | Returns `None`; the caller must report it (docstring `:389-391`). | none. |
| `layers/dcp/comm.py:97-101` | deprecated-name shim | **BENIGN** | `DeprecationWarning` for a renamed symbol. | none. |

## Top findings

**#505-A2-01 — the kvso draft-scratch carve-out silently does nothing on hybrid
(mamba/SWA) allocators, and then logs success.**
`managers/kv_session_offload.py:2204-2207` writes `self.allocator.size -=
int(self.mtp_resident_slices)` inside `try: … except Exception: pass`. On
`UnifiedMambaAlloc` / `UnifiedSWAAlloc` — the allocators for exactly the hybrid
GDN models this fork serves — `size` is a computed property whose setter is
`pass` (`mem_cache/multi_ended_allocator.py:1764-1766`, `:1974-1976`). The
write is therefore a no-op with no exception, the reserved slots stay counted
in the pool's advertised capacity while being permanently out of circulation,
and `:2208` logs *"reserved %d draft-read scratch slots"* either way. The
comment at `:2198-2203` states the intent explicitly ("so the
SchedulerInvariantChecker … stays balanced"), so the failure is a broken stated
invariant, not an unstated assumption. Three of this project's standing laws
converge here: a success message treated as evidence, a returned/derived value
whose contract lives in a comment, and pool accounting that stays wrong
forever. Reach qualifier, stated honestly: the whole spec-in-tick arm sits
behind `KVSO_ALLOW_SPEC=1` (`server_args.py:6580`, recorded as #500-B10), so a
stock speculative boot does not reach it today.
*Task:* `#505-A2-01: kvso draft-scratch reservation must verify the pool carve-out took effect (composite allocators no-op the size setter)`

**#505-A2-02 — a foreign-residue-polluted KV budget correction is persisted and
is unmarked on read.**
`model_executor/model_runner_kv_cache_mixin.py:1208-1209` warns with a message
that itself predicts the harm — *"the correction persisted for the next boot
will under-book the KV pool"* (`:1078-1079`) — and then persists it. Nothing is
written into the cache to mark the reading as polluted, and the reader
(`:896-910`) checks only shape and `mlp_vector`. The damage lands in the NEXT
process, in a different log, as an unexplained smaller KV pool. The same file
already has the right shape one screen up: a malformed cache is refused by name
with `provenance = "malformed"`.
*Task:* `#505-A2-02: refuse to persist (or mark "polluted") a measured KV-budget correction taken over foreign device residue`

**#505-A2-03 — `try_spill` declines on a rank-LOCAL quantity where its own
sibling raises.**
`managers/kv_session_offload.py:3426` compares `n_own` — derived from this
rank's owner window `lo/hi` under weighted uneven DCP — against the replicated
`region_tokens`, warns, and returns `False`, which sends only this rank down
the stock-retraction path (`managers/scheduler.py:4051`). Every other decline
in the function is annotated as resting on replicated inputs; this one is not.
The prefill-spill twin raises a `RuntimeError` on the identical condition
(`:3874-3880`), and the module header states the doctrine that makes the
asymmetry a defect: *"RANK-UNIFORMITY (this fork: divergence == NCCL hang, not
a wrong number)"* (`:501`). Cheap fix, exact in-tree precedent: raise like the
twin, or min-reduce the verdict the way `kv_session_spill_destination.py`
already does for its I/O flags.
*Task:* `#505-A2-03: try_spill must not decline on a per-rank owned-row count — raise like the PS2 twin or min-reduce the verdict`

**#505-A2-04 — PD topology fabricates a CUDA→NVML identity mapping when the
bridge is unavailable.**
`disaggregation/topology.py:421-427` logs *"assuming identical enumeration
orders"* and sets `mapping = {}`, i.e. the identity — on a rig where CLAUDE.md
records that torch order != NVML order. Card VRAM totals are then attributed to
the wrong ranks, and the only downstream check is warn-only
(`check_vram_feasibility`, `:742-743`), so an infeasible plan boots. The
function's own docstring demands `None` + a caller-side report for the sibling
NVML failure, and `barlink_matrix_transport.py:218-224` shows the correct shape
(degrade to a NAMED provenance, `"sysfs-gross"`, never to a fabricated
identity).
*Task:* `#505-A2-04: PD topology must return None when the CUDA→NVML bridge is unavailable instead of assuming identical order`

**#505-A2-05 — barlink `bar1`/`matrix` fall back to gloo per rank with no group
agreement.**
`distributed/device_communicators/barlink.py:384-400` catches any bring-up
exception, warns, and returns `None` for the two transports deliberately
excluded from `_NO_FALLBACK` (`:327`). No step reconciles the outcome across
the group, so a probe that fails on one card only (BAR1 peer mapping, the
`dmabuf_holder` guard, the byte proof — `barlink_bar1.py:589`, `:597`,
`:4644-4656`) leaves the group split. Direction of failure is unproven and is
reported as such: a failure before `dist.all_gather_object` at
`barlink_bar1.py:1953` desyncs into a hang, one after it does not. The fix is
one collective, and the tree already contains it twice for precisely this class
(`parallel_state.py:975-992`; `model_runner.py:1365-1369`). This also touches
the CLAUDE.md barlink-default order: a group silently on gloo is a
NCCL/gloo run reported as a barlink run.
*Task:* `#505-A2-05: reconcile the barlink transport outcome across the group before use (all-gather like custom-allreduce does)`

**#505-A2-06 — two selftests in the decode graph runner swallow their own
exceptions.**
`model_executor/runner/decode_cuda_graph_runner.py:1243-1245` and `:1263-1265`
log *"selftest raised (ignored)"* and continue. A crashing instrument is
indistinguishable from a passing one downstream, and worker-process WARNING
lines are documented in this tree as not reaching the server log
(`model_executor/forward_peak.py:14-17`).
*Task:* `#505-A2-06: kvso spill-graph / C4 selftests must record a NOT-RUN verdict on an evidenced channel, not a swallowed warning`

**#505-A2-07 (registration, not a bug) — `MultiEndedAllocator.restore_state`
carries a comment-asserted invariant with a warn-and-continue arm and no
production caller.**
`mem_cache/multi_ended_allocator.py:288-313`. Per CLAUDE.md rule (3) the
comment at `:289-290` is a testable claim; the arm is currently unreachable
(`backup_state=True` has no caller in `srt/`), which is itself worth pinning
before some future spec path reaches it.
*Task:* `#505-A2-07: pin "no free() inside a backup window" with a test, or make restore_state raise now that nothing trips it`

## Fix-pattern references used above

- (i) named hard error at the site — precedent in this scope:
  `managers/kv_session_offload.py:3874-3880`,
  `layers/dcp/collective_guard.py:116-122`.
- (ii) completeness check in the #496-(b) shape —
  `_assert_required_params_loaded` is defined at
  `python/sglang/srt/models/deepseek_v4_dspark.py:868` and called at `:864`. It
  is the model-loader analogue: enumerate what MUST have been written and raise
  naming the missing entries. The A2-01 row is the same defect in the pool
  ledger and wants the same shape (assert the ledger moved by the amount that
  was carved out).
- (iii) independent state probe — `kv_session_spill_destination.py:621-626` +
  `:309-330` (min-reduced flags instead of a per-rank success belief);
  `distributed/parallel_state.py:975-992` (all-gather the local capability
  before letting it gate a collective).

## Catalog sections read

`CLAUDE.md` in full (MECHANISM REACH, REACH INCLUDES PARAMETERS, SUCCESS CLAIMS
ARE NOT EVIDENCE, the rank-uniformity and device-identity rules, the barlink
default order). `docs/dev/FEATURE_CATALOG.md` §1 (uneven parallelism, incl. the
two attention/KV distribution axes), §3 (memory tiers / offload / spill), §5
(multi-group runtime), §6 (weightless KV lane), §7 (collectives / transport),
§12 (robustness canon, incl. the #404 bookkeeping-owner register), §17
(combination matrix + eviction doctrine).
`docs/dev/AUDIT_500_mechanism_reach.md` §§1-6 (classification method, verbatim
predicate quoting, honest coverage statement, bug-candidate write-up shape).
