# DESIGN #297 — Phase-boundary KV resharding (dcp_ratio as a real actuator)

Branch `feat/kv-resharding`, base `eda4f03b28`. Turns the `dcp_ratio` rung of
the #287 pressure ladder from PLANNED-ONLY into a physical actuator: existing
KV-cache bytes are redistributed across the DCP ranks when the token-ratio
vector flips (e.g. `7,3,3 -> 2,11,10`, the #320-proven redistribution), at a
true phase boundary (fully idle scheduler), rank-uniformly, byte-identically.

## 1. Why no metadata rewrite is needed (the core simplification)

Under the weighted owner rule (`layers/dcp/owner.py`), the physical pool row
of a global KV slot `L` is a PURE FUNCTION of `L` and the vector:

    owner(L)  =  rank r with (L % S) in [prefix[r], prefix[r+1])
    row(L)    =  (L // S) * ratio_r + (L % S - prefix[r])

`req_to_token`, the radix tree and the allocator all store GLOBAL slot ids.
None of them reference physical rows. Therefore a reshard moves BYTES ONLY:
for every live slot `L`, the bytes must be at `row'(L)` on `owner'(L)` under
the new vector, and every translation site must use the new vector afterwards.
No request, radix or allocator state is rewritten — the cutover is
`set_cp_token_ratios(new)` plus a refresh of the few caches that snapshot the
vector (section 5).

## 2. Stage A scope (this task) and guards

Stage A moves KV at a TRUE phase boundary: the scheduler is fully idle
(`is_fully_idle()` — empty running batch, empty waiting queue, no chunked req,
no in-flight async KV ops, no spilled sessions). At that moment the only live
KV is the radix tree's content (evictable + protected), enumerated via
`tree_cache.all_values_flatten()` — replicated across ranks by the same
argument as `replicated_pool_usage`.

Stage B (move under load with delta catch-up) is NOT built; it is a named
follow-up and only worthwhile once A is proven.

Hard guards (arming refuses loudly, reshard stays planned-only):
- weighted uneven DCP must be active and `dcp_size == tp_size`;
- target vector length == dcp_size, all entries >= 1;
- target vector must be in the declared ceiling set (section 4);
- refused while incompatible features are enabled: hierarchical-cache
  storage backup, PD disaggregation, weightless-KV ranks, dual-group lane
  (each caches or encodes owner state we do not migrate in Stage A).
  kv-session-offload is compatible at the boundary itself: `is_fully_idle()`
  already implies `has_spilled() == False`, and the spill manager derives its
  sentinel arithmetic from the live vector; it is nevertheless guard-refused
  in Stage A because its host-pool row layout is sized from the boot vector.

## 3. Two-phase cutover and the "silence" answer

At a consensus tick where every rank is armed AND ready (idle), the move runs
SYNCHRONOUSLY inside the scheduler round, on every rank, in lock step:

1. PACK phase (reads only, pool untouched): stage outgoing rows per
   destination rank into device payloads (+ 8-byte checksum trailer).
2. EXCHANGE (pool still untouched): one `batch_isend_irecv` batch over the
   TP DEVICE (NCCL) group, every work polled through the #312 bounded
   primitive; then checksum verification. A failure up to and including
   this point aborts with the pool byte-identical -- a later boundary may
   retry. Transport decision (card-run finding): torch's GLOO p2p works do
   not implement `is_completed`, so a bounded poll can never observe
   completion -- the first card run ended in a clean 120 s
   `CollectiveTimeoutError` on all three ranks (the #312 family catching
   exactly its hang class). NCCL p2p works poll truthfully and keep the
   payloads on-device.
3. WRITE phase (the only no-return region): per layer, gather retained rows
   into a temp (read), then scatter retained and received rows into the NEW
   rows. Reads of a layer strictly precede its writes, which makes the
   old/new row overlap in the same physical buffer harmless (the aliasing
   falsifier pins this); an error here is fatal and loud on every rank.
4. CUTOVER: install the new vector, refresh every snapshot cache, bump the
   reshard epoch, log `KV-RESHARD` with duration and bytes.

Scheduler interplay (card-run finding): an armed reshard must keep the
scheduler loop TICKING -- `maybe_sleep_on_idle` parks rank 0 in a zmq poll
and the request broadcast parks every other rank behind it, so a parked
loop never reaches a consensus boundary. While a target is pending
(replicated state), the sleep is skipped; the busy spin is bounded by the
consensus interval plus the move itself.

Because the server is fully idle, there is no serving to silence: the only
"Stille" is that a request arriving mid-move waits in the input queue for the
move duration (measured, target < 1 s for the delta). There are no deltas to
chase — nothing allocates or writes KV between the ready check and the
cutover (same scheduler iteration, single-threaded). This is why Stage A
chooses "short silence at an idle boundary" over "shadow copy with delta
catch-up": at an idle boundary the delta is structurally empty, so the
simpler design is also the correct one.

Atomicity: no rank leaves the collective with a half-installed state. Errors
before the WRITE phase abort the attempt with the pool untouched (retry at a
later boundary is legal). Errors after writes began are FATAL and loud
(`KvReshardError` on every rank via the bounded collective family) — the
server never serves from a mixed layout because the failure path raises
before the scheduler round continues.

## 4. Capacity: fitted-ceiling reservation (decision vs #93 VMM)

A vector flip changes each rank's required physical rows
(`dcp_compact_pool_rows(C, S_v, r_v)`). Decision: **fitted ceiling at boot**,
not runtime VMM growth. `--kv-reshard-vectors` declares the vector set; the
uneven-DCP min-rule in `_apply_token_constraints` is extended to pick the
largest C feasible under EVERY declared vector, and the hybrid full-KV pool
is sized to each rank's row maximum over the set. Costs VRAM headroom up
front, buys: stable pool addresses and sizes (decode CUDA graphs are keyed on
batch size/stream/variant and refill `cuda_graph_kv_indices` — sized
`max_num_tokens * max_context_len`, vector-independent — host-side per
replay, so NO recapture and NO per-geometry graph set is needed). #93 VMM
append-growth (`kv_vmm_backing.py`) remains the named follow-up for
memory-tight configurations; it composes with this design (grow to the
ceiling lazily instead of at boot) without changing the cutover.

## 5. Ownership-transition matrix (every reader of the vector)

| Reader | How it holds the vector | Cutover action |
|---|---|---|
| `distributed/utils._CP_TOKEN_RATIOS` | process-global source of truth | `set_cp_token_ratios(new)` |
| flashinfer backend `cp_S/cp_lo/cp_hi/cp_ratio` | snapshot in `__init__` | `refresh_dcp_owner_bounds()` on every registered backend |
| triton backend, same fields | snapshot in `__init__` | same refresh |
| spec-verify (`dcp_verify_*`) | reads backend fields | covered by backend refresh |
| CUDA-graph replay metadata | rebuilt per replay from backend fields | covered by backend refresh (no recapture; section 4) |
| `cache_controller._dcp_owner_ctx_cache` | memoized | invalidated at cutover (feature also guard-refused in Stage A) |
| allocator / `req_to_token` / radix | global slot ids only | nothing to do (section 1) |
| hybrid pool sizing | ceiling over the declared set | nothing to do at cutover |
| scheduler `max_total_num_tokens` | vector-independent (fixed C) | nothing to do |
| kv-session-offload sentinels / PD disagg / hicache storage / weightless | own owner encodings | Stage A guard-refused (section 2) |
| draft/NEXTN KV pool | token-complete per rank (not DCP-sharded) | unaffected |
| GDN/Mamba state pool | per-request, not token-sharded | unaffected |

All refreshes happen inside the same synchronous, all-ranks move; the next
forward pass on any rank sees only the new state. Cross-rank uniformity comes
from the consensus commit plus the collective exchange itself.

## 6. Consensus protocol (the #287 pattern, extended)

`KvReshardRuntime.on_round(fully_idle)` runs every scheduler iteration when
`--kv-reshard-vectors` is set (flag off = not constructed, zero collectives,
byte-identical). Every `consensus_interval`-th round, UNCONDITIONALLY, every
rank enters one bounded MIN-reduction with payload
`[armed, ready, epoch, v_0..v_{n-1}]` packed as `(x, -x)` pairs:

- `epoch` (completed reshards) and — once `min(armed) == 1` — the target
  vector elements are EQUALITY-checked: any mismatch raises the same loud
  `KvReshardError` on every rank (desync falsifier).
- `armed` and `ready` are MIN-semantics: disagreement is LEGAL and resolves
  to "wait for a later boundary" uniformly (arming arrives via broadcast RPC
  or a ladder flip and may skew by an iteration; idleness may skew while
  async queues drain). This is the [[rank-lokaler-test-vor-kollektiv]]
  discipline: the byte-moving collective is entered only from a
  group-agreed state, never from a rank-local predicate.

Arming sources (both replicated): the #287 ladder's `dcp_ratio` FLIP (the
runtime passes the operating point's `kv_vector` to `arm()` — the rung is now
WIRED), and the `POST /kv_reshard` control RPC (flush_cache-style broadcast)
for operator- and test-driven resharding.

## 7. Verification

- Hermetic (CPU, threads + barrier channel, `test/registered/scheduler/
  test_kv_reshard.py`): plan covers every live slot exactly once against a
  brute-force owner reference; byte-identity of the reassembled global KV
  after a threaded multi-rank move; aliasing falsifier (heavy old/new row
  overlap); desync falsifier (poisoned target vector -> every rank raises,
  none hang, no pool byte was written); readiness skew holds uniformly;
  guard refusals; flag-off inertness.
- Per-pair payload checksums travel with every exchange (cheap, always on) —
  they pin the sender/receiver packing-order contract at runtime.
- Card window: boot `7,3,3` with ceiling `{7,3,3; 2,11,10}`, generate, go
  idle, trigger reshard, measure move duration (< 1 s target), same session
  continues coherently from the moved prefix (radix hit), negative control
  without the flag byte-identical.
