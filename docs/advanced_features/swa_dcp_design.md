# Uneven DCP for SWA-hybrid models (Gemma4 class) — design note (Task #91)

Status: CHECKPOINT RULED — **Stage A only** (constant-swa sizing, no DCP).
**Update 2026-07-26 (#96): Stage B was subsequently built** on branch
`feat/swa-dcp-triton` (section 3's design, on top of #173's Triton weighted
owner rule) — see `docs_new/swa_dcp_stage_b_triton.md` for what landed, what it
refuses, and the GPU recipe. It is CPU-pinned and NOT GPU-validated; the
+6-10% below is still an estimate. The rest of this note is the original
Stage-A checkpoint text and is left as the historical record.
Stage B (the triton weighted-DCP port, sections 3(a-c)) is DESCOPED to a
backlog task; recorded reactivation criteria: Stage-A measurements revealing
a bigger stranded-VRAM gap than estimated, strategic desire for a uniform
qwen/gemma DCP operational mode, or future SWA models with more global
layers. The Stage-B backlog entry additionally depends on merging #88
(`--rank-kv-ratio`, branch `feat/kv-split-ratio`, diverges at `5e4fbed68`)
into the feature branch — deferred to a dedicated integration round, NOT
needed for Stage A (no token sharding here).
Branch `feat/swa-dcp`, base `715ca09fb` (tip of `feat/gemma-bringup`).

## 0. Stage A as implemented: `--swa-pool-sizing {ratio,cap}`

A checkpoint correction to section 3(d) as originally written: the rule
"swa = ratio x min(C, #90 cap)" does NOT deliver the section-5 win — at a
raised `--context-length` the #90 cap value grows proportionally, so
ratio x cap blows the swa pool up again. The section-5 scenario numbers
implicitly (and correctly) assumed the swa pool pinned at its
**window-bounded worst case** (`swa_pool_token_cap`: window + eviction lag +
decode over-allocation per request, plus in-flight prefill chunks). That is
the implemented rule:

- New flag `--swa-pool-sizing`, default `ratio` = byte-identical today
  (every new branch is behind `== "cap"` checks).
- `cap`: selects `SWAChunkCapPoolConfigurator` (which already implements
  pin-swa-give-rest-to-full) **with the radix cache allowed**. Safety
  argument: scheduler admission counts `swa_available + swa_evictable`
  (schedule_policy.py) and eviction is demand-driven
  (mem_cache/common.py `evict_from_tree_cache` evicts the swa side when
  `swa_available_size < need`), so cached in-window prefixes are reclaimed
  under pressure; the pin bounds SWA-side cache RETENTION (hit-rate
  tradeoff), never correctness. The legacy auto route (radix disabled) is
  condition-for-condition unchanged.
- Preconditions (fail fast in server_args): explicit
  `--max-running-requests` and chunked prefill enabled.
- #90 cap in `cap` mode: the reachability ceiling on max_total becomes
  `full_need` alone — the `ceil(swa_need / ratio)` term exists only to
  back-derive the max_total that makes a RATIO-sized swa pool big enough,
  which no longer applies. Ratio mode keeps the #90 formula byte-identical.
- Measured shape (gemma-4-31B launch config, 4 reqs, window 1024, chunked
  2048, overlap off): `swa_pool_token_cap = 6665` tokens -> r1 (5090) swa
  pool 2.39 GB vs 9.40 GB ratio-sized at the 8k operating point — ~7 GB
  moved to the full pool on the largest card alone.

Touched files: `server_args.py` (flag + validation),
`pool_configurator.py` (`is_applicable` cap route + factory no-effect
warning), `model_runner_kv_cache_mixin.py` (#90 cap `full_need` branch).

### Stage-A measured results (2026-07-18, full tables in job tmp t91/)

Gemma4-31B TP=3 uneven, ctx 65536, 4 reqs: ratio (today) 42,856 total
tokens (memory-limited, 47.6k-token input rejected) -> cap mode
**262,160** (#90 reachability cap binds; profiled memory ceiling 388,157;
5.22 GB min free) = **6.1x served capacity**; the estimate in section 5
(~3.7-4x) was conservative — measured swa decoupling freed more than
modeled. Needle at 50,168 prompt tokens retrieved on both cap boots
(baseline rejects it); sizing deterministic line-for-line across 2 boots;
temp-0 outputs bit-identical to the ratio oracle; shallow decode 19.4
tok/s in both modes. Regressions byte-identical: gemma TP=3 ratio @8k
(32784/26227 = #90 values), TP=1 fp8 (26767/21413), Qwen3.6-27B
weighted-DCP (443,904, vector [30,17,17], hint 563,456 = T88 reference).
The Stage-B descope criterion re-check: with the #90 cap now binding at
4 x ctx and ~5 GB min free, the stranded-VRAM gap is SMALLER than the
section-5 estimate — Stage B stays descoped.

## 1. Problem

The fork's uneven DCP (weighted KV token-sharding: owner rule per virtual
block, compact per-rank pools, LSE merge) assumes one growing full-attention
KV pool. Gemma4 is a hybrid: 50 of 60 layers are sliding-window (window 1024,
constant-size ring KV with full->swa loc translation), 10 are global
full-attention. Today DCP is simply OFF for gemma4 (HANDOFF_M41 gap 1):
`SWAKVPool.set_kv_buffer` does not even accept `dcp_kv_mask`, and the
window-vs-ownership semantics are undesigned.

Gemma4-31B geometry (measured from the checkpoint config):
60 layers = 50 SWA + 10 full; 32 q / 16 kv heads; head_dim 256; window 1024.
bf16 KV: 1 KiB per token per layer per kv head (K+V).
Uneven TP=3 plan (83ff3cdb1): q sliding [8,14,10] / full [8,16,8]
=> kv shards sliding [4,7,5] / full [4,8,4] (GQA units of 2).

## 2. Code reality (read before designing — findings)

| Piece | Where | State today |
|---|---|---|
| SWA/full pool split | `mem_cache/swa_memory_pool.py` `SWAKVPool` (two `MHATokenToKVPool` sub-pools), `mem_cache/allocator/swa.py` `SWATokenToKVPoolAllocator` (full index space + `full_to_swa_index_mapping`) | No DCP awareness anywhere; `set_kv_buffer` lacks `dcp_kv_mask` |
| Pool sizing | `model_executor/pool_configurator.py` `HybridSWAPoolConfigurator` (swa = ratio x full, default ratio 0.8), `SWAChunkCapPoolConfigurator` (constant swa from `swa_pool_token_cap`, requires `--disable-radix-cache` + explicit `--max-running-requests` + chunked prefill) | `HybridSWAPoolConfigurator` has NO uneven-DCP cell-size branch (`DefaultPoolConfigurator` line ~266 has one) |
| #90 physical ceiling | `model_runner_kv_cache_mixin.py` `_swa_hybrid_kv_token_cap` (cap = max(full_need, ceil(swa_need/ratio))) | Caps C at reachability; measured 32784/26227 @ ctx 8192, 4 reqs |
| Weighted owner rule | `distributed/utils.py` (`cp_token_prefix`, `uneven_dcp_owner_bounds`, `cp_token_split_factor`) | Backend-agnostic, fine as-is |
| Weighted DCP attention | `flashinfer_backend.py` only: `cp_lo/cp_hi` snapshot, `_dcp_masked_write` (kv-head all-gather + owner-masked compact store), `_build_dcp_weighted_kv_indices`, `_forward_{extend,decode}_dcp` with `cp_all_gather_heads_uneven` / `cp_lse_ag_out_ar_mha_uneven` | **Gemma runs the TRITON backend** (`arg_groups/overrides.py` `_gemma4_overrides`: default triton off sm100) |
| Triton DCP | `triton_backend.py` + `layers/dcp/kernels.py`, `layout.py` | EVEN-modulo only (`loc // dcp_size`, `pos % dcp_size == rank`), EVEN-head `group.all_gather(dim=1)` + `cp_lse_ag_out_rs_mha`; `_forward_extend_dcp` raises NotImplementedError for sliding-window layers and custom masks; DCP applied unconditionally to ALL layers when `dcp_size > 1` |
| LSE-merge comm helpers | `layers/dcp/comm.py` | Backend-agnostic (used by flashinfer); reusable from triton |
| Per-rank workspace sizing | `triton_backend.py` `_plan_aware_num_q_heads` (83ff3cdb1) | Plan-aware; DCP path must size by GATHERED total heads |
| #88 `--rank-kv-ratio` | commit `428d3cd19` on `feat/kv-split-ratio` | **NOT an ancestor of this branch** (diverges at `5e4fbed68`); design assumes merging it in (3 commits, one duplicated testfix, trivial merge) |

## 3. Chosen design: replicate SWA per rank, DCP-shard only the global layers

Every rank already stores only its swa kv-head shard; the swa pool is
window-bounded (constant in context length). All the *growing* long-context
KV lives in the 10 full layers. So:

- **SWA layers (50): unchanged local path.** Each rank stores ALL token
  positions (within window semantics) for its kv-head shard, writes via the
  existing pre-translated `swa_loc`, reads via `window_kv_indices`. No owner
  rule, no mask, no merge. Byte-identical to today's validated path.
- **Full layers (10): weighted-DCP token sharding**, exactly the qwen
  semantics: kv heads REPLICATED (all 16) per rank, tokens owner-ruled,
  compact per-rank storage, q all-gather + LSE merge per layer.

### (a) Pool split / construction

`SWAKVPool` gains the same gated treatment `HybridLinearKVPool` got
(`model_runner_kv_cache_mixin.py` ~1470-1516), applied ONLY to the full
sub-pool:

- `size` (full sub-pool) = `(C // S) * ratio_r` (compact owned share);
  `head_num` for the full sub-pool = total kv heads (16) under
  `uneven_dcp_kv_replicated`; requires splitting the currently-shared
  `head_num` kwarg into full vs swa values (swa keeps the per-rank shard).
- `size_swa` = per-rank constant (see (d)); swa sub-pool untouched.
- `set_kv_buffer` accepts and forwards `dcp_kv_mask` to the full sub-pool
  only (the trivial signature fix from gap 1); swa branch ignores it.
- `SWATokenToKVPoolAllocator` is constructed with the GLOBAL virtual index
  space C for the full side (same rule as the existing weighted-DCP paged
  allocator branch, mixin ~1698: index space = C, natural page size);
  `full_to_swa_index_mapping` is indexed by full-pool indices, so it is
  sized C+1 (int32; ~4 B/slot — negligible). Translation full->swa is
  position-keyed, not ownership-keyed, so it works unchanged: every rank
  translates every global loc to its LOCAL swa slot.

### (b) dcp_kv_mask / owner rule per-pool

Ownership attaches to the FULL sub-pool only. The owner rule and vector
(`cp_token_prefix` etc.) are process-global and stay untouched; what changes
is *which writes/reads consult it*:

- `_set_kv_buffer` (triton): branch on `layers_mapping[layer_id]` — swa
  layer -> plain write with `swa_loc`; full layer -> kv-head all-gather
  (`cp_all_gather_heads_uneven`, kv shards [4,8,4]) + owner-masked compact
  write (port of flashinfer `_dcp_masked_write`, weighted formula
  `(L // S) * ratio + (L % S - lo)`).
- HiCache `cache_controller` already maps device<->host through
  `uneven_dcp_owner_bounds()` lazily; for SWA-hybrid it must apply that
  mapping only to the full sub-pool buffers and treat swa buffers as fully
  local. v1: **HiCache + swa-dcp explicitly rejected at validation**
  (fail fast), lifted later.

### (c) Attention path: per-layer mixing in the triton backend

Verified feasible but NOT free — this is the main implementation cost:

- `init_forward_metadata` already builds BOTH the DCP kv indices and the
  window buffers side by side (lines ~730/865: window buffers are built
  regardless of `dcp_size`), so metadata coexistence is already there.
  The DCP kv-index build must switch from the even-modulo kernel to the
  weighted rule — reuse/relocate flashinfer's `_build_dcp_weighted_kv_indices`
  into `layers/dcp/` and call it from both backends.
- Forward dispatch: today `if self.dcp_size > 1:` routes ALL layers into
  `_forward_extend_dcp` / the decode DCP branch. Change to
  `if self.dcp_size > 1 and not is_swa_layer:`. SWA layers fall through to
  the existing (validated) window path — the current
  `NotImplementedError("DCP Triton extend does not support sliding window")`
  becomes unreachable by construction.
- The triton DCP paths are today EVEN-only in a second way: they use
  `group.all_gather(q, dim=1)` (equal per-rank head counts) and
  `cp_lse_ag_out_rs_mha` (even reduce-scatter). Under the uneven plan the
  q-head counts differ per rank AND per layer type; the full-layer path must
  use `cp_all_gather_heads_uneven` + `cp_lse_ag_out_ar_mha_uneven`
  (backend-agnostic, already in `layers/dcp/comm.py`) with the FULL-layer
  q partition [8,16,8]. Workspace sizing: gathered-total heads (32), the
  83ff3cdb1 plan-aware sizing extended to the gathered case.
- CUDA graphs: out of scope for v1 (gemma bring-up runs eager /
  `--disable-cuda-graph`); the DCP graph path stays fenced off.

### (d) Sizing

- **Full pool becomes the DCP-shardable budget.** Per-rank:
  `B_full_r = available_bytes_r - swa_fixed_bytes_r`;
  `P_r = B_full_r // full_cell_repl` where `full_cell_repl` uses TOTAL kv
  heads (16 x 10 layers x 1 KiB = 160 KiB/token, bf16) — i.e.
  `HybridSWAPoolConfigurator` gains the same `uneven_dcp_kv_replicated`
  branch `DefaultPoolConfigurator` already has, but applied to
  `_full_per_token` only. Then the existing weighted-DCP capacity math runs
  unchanged: `C = min_r(P_r // ratio_r) * S`, per-rank compact share, #90's
  reachability cap (`concurrency x (ctx + headroom)`) applied to global C
  (formula unchanged — it is already expressed in context tokens).
- **SWA pool per rank is a CONSTANT, decoupled from C.** It must not scale
  as `ratio x C` (C can exceed any single rank's reach — the exact pre-#90
  OOM disease). v1 rule: `swa_tokens = ratio x min(C, swa_hybrid_cap)` where
  `swa_hybrid_cap` is #90's reachability value — i.e. identical to today's
  DCP-off swa sizing at the same launch settings (26227 @ 8192/4reqs). This
  is deterministic, radix-compatible, and never larger than the DCP-off
  value. (A tighter `swa_pool_token_cap`-based bound is a follow-up knob.)
- **#88 integration** (`--rank-kv-ratio capacity`): the measured-install in
  `_resolve_memory_pool_config` derives `P_r` from
  `calculate_pool_sizes(...)`; with the configurator changes above that
  number automatically reflects the full-pool-only budget, so capacity mode
  works with no extra logic — but #88 must first be MERGED into this branch
  (see topology note, section 8). The rank-uniform collective guard
  (904102a8c) applies unchanged.

### (e) Prefix cache / radix / HiCache

Radix (and swa_radix_cache) operate on full-pool indices, which under
weighted DCP are GLOBAL allocator indices — the same invariant the qwen
weighted path already validated (#60/#88): install-before-first-cache, never
changes. The swa side of the radix tree keys the same global indices through
the (globally-sized) translation table, and swa eviction
(`ScheduleBatch._evict_swa`, `free_swa`) frees LOCAL swa slots identically on
every rank (same positions evicted everywhere — rank-uniform because
scheduling is rank-uniform). HiCache: rejected in v1 (see (b)).

### (f) MTP / draft

Gemma4-31B has no draft/MTP tensors (M41 handoff) — the draft-worker
non-DCP pool branch and verify paths are simply never exercised. The
`_draft_non_dcp` gates stay as they are.

## 4. Alternative considered: window-aware ownership (shard SWA tokens too)

Shard the sliding window's tokens across ranks (owner rule applied within
the ring buffer, LSE merge on every layer).

- Benefit: removes the per-rank swa replication cost — bounded by
  `swa_tokens x kv_swa_r x 50 KiB` ≈ 2.0-3.6 GB/rank at the #90 operating
  point; sharing saves at most ~2/3 of that (~1.4-2.4 GB/rank).
- Cost: LSE merge + q all-gather on 60 layers per step instead of 10 (6x
  the per-step DCP comm), kv-head replication x50 layers of write traffic,
  and genuinely new semantics: ring-buffer slots are position-recycled, so
  ownership must follow the *slot*, not the token position, or eviction lag
  makes owner sets diverge across ranks. That is a new correctness surface
  (the #83ff/#60 bug class) for a bounded, small win.
- Verdict: rejected. The replicated-SWA design captures the growing-KV win
  at constant window cost; window sharding trades a fixed few GB for 6x
  merge traffic and the hardest new semantics in the space.

## 5. Expected capacity win — honest numbers

Anchors: #90 measured (ctx 8192, 4 reqs, bf16 KV, TP=3 uneven): full pool
32784, swa 26227; per-rank pool bytes [6.7, 12.1, 8.0] GB (r1 = 5090);
3.9-9.8 GB free per rank after pools. Approximate per-rank KV budgets
(pool + free; to be re-measured in validation):
r0 ≈ 10.6 GB, r1 ≈ 21.9 GB, r2 ≈ 12.0 GB.

Per-token cell sizes (bf16): full per kv head 10 KiB (10 layers), swa per kv
head 50 KiB (50 layers). Shards: full [4,8,4], swa [4,7,5] kv heads.

**At the validated operating point (ctx 8192, 4 reqs): the win is ZERO.**
The #90 reachability cap binds at 32784 tokens; every rank has GB of
headroom. DCP only matters at memory-limited settings (longer context /
more requests).

Long-context scenario (4 reqs, illustrative swa constant ≈ 16k tokens):

| Config | Binding math | C (tokens) | ~ctx/req |
|---|---|---|---|
| (0) today: ratio-coupled swa, no DCP | min over cell_r = [200, 360, 240] KiB/token | ≈ 49k | ≈ 12k |
| (1) constant-swa sizing, NO DCP | swa fixed [3.3, 5.7, 4.1] GB; full cell [40, 80, 40] KiB; min(178k, 198k, 193k) | ≈ 178k | ≈ 44k |
| (2) = (1) + DCP full-pool sharding (capacity vector) | sum(remaining)/160 KiB = 31.4 GB / 160 KiB | ≈ 192k | ≈ 48k |

Two honest conclusions fall out:

1. **~3.7x of the long-context win comes from decoupling the swa pool from
   C — which does NOT require DCP at all.** It is a sizing change (the
   `SWAChunkCapPoolConfigurator` machinery already exists behind
   `--disable-radix-cache`; the ratio-path variant needs a bounded-swa
   extension of #90).
2. **The DCP-specific increment is ≈ +6-10%** (178k -> 192k here; the exact
   number depends on measured budgets). Compare qwen's +26.9% (#88): gemma
   shards only 10/60 layers, and its full-layer kv shards [4,8,4] =
   [25/50/25]% already track the VRAM distribution [~26/47/27]% closely, so
   pooling recovers little. The aggregate full-layer cost is identical with
   and without DCP (replication x sharding cancel: 160 KiB/token either
   way); the win is purely mismatch recovery.

Secondary DCP effects, for completeness: the flashinfer-style dequant-gather
would NOT automatically unblock fp8-KV on the 3080s (gap 2) — the triton
extend kernel still reads the KV buffer directly; fp8 for the full pool only
would need explicit dequant staging, and the swa pool stays fp8-blocked on
sm86 regardless. So gap 2 is NOT a justification for this design.

## 6. Recommendation

Given section 5, the honest recommendation is to **split the task and put a
descope checkpoint between the halves**:

- **Stage A (recommended, small): swa-decoupled long-context sizing, no
  DCP.** Bound the swa pool per rank (reachability/window cap) and give the
  full pool the remaining budget, per rank, DCP off. Touches only
  `pool_configurator.py` + the #90 cap plumbing; no distributed semantics,
  no kernel work. Expected ~3-4x long-context capacity on 31B TP=3.
- **Stage B (this task's nominal core, expensive): the triton weighted-DCP
  port + per-layer mixing of section 3.** Expected +6-10% on top of Stage A
  on this rig, for the largest implementation surface in this design
  (weighted index kernel port, uneven-head gathers in the triton DCP paths,
  masked-write port, per-pool sizing branches). Recommend descoping to a
  follow-up unless the measured Stage-A numbers reveal a bigger stranded-VRAM
  gap than estimated, or the strategic value (uniform DCP operational mode
  across qwen and gemma fleets) outweighs the ratio.

If main wants Stage B regardless, the design in section 3 is the one to
build; nothing in Stage A is throwaway (Stage B strictly builds on it).

## 7. Risk list

1. Triton weighted-DCP port is new kernel-adjacent work (even-modulo index
   kernel -> weighted; even all_gather/reduce-scatter -> uneven helpers);
   the flashinfer helpers are reusable but the triton decode/extend
   integration is hand-written. Bug class: loc-space confusion
   (virtual/compact/swa) — exactly the #83ff smoking-gun class.
2. SWA-pool constant sizing under radix: bound too tight -> alloc failure /
   eviction livelock; too loose -> wasted GBs. v1 rule (sec 3d) is
   deliberately conservative (never below today's DCP-off size).
3. `head_num` split (full=16 replicated vs swa=shard) threads through
   `SWAKVPool` kwargs, kv_cache_builder, and host-pool mirrors; a missed
   consumer silently mis-sizes buffers.
4. Determinism: P_r measurement now subtracts swa_fixed bytes; the #88
   collective-install must stay rank-uniform (904102a8c guard pattern).
5. CUDA-graph and overlap-schedule interactions untested (v1: eager,
   overlap allowed but validated explicitly).
6. HiCache + swa-dcp combination rejected in v1; needs a fail-fast gate so
   it cannot be enabled silently.
7. Branch topology: #88 must be merged from `feat/kv-split-ratio` (diverges
   at `5e4fbed68`, 3 commits, one duplicated testfix) before Stage-B
   capacity-mode work; a sloppy merge would double-apply the testfix.

## 8. Validation plan

Vehicle: gemma-4-31B-it-int4-AutoRound, TP=3 uneven (5090 resolved by NAME),
bf16 KV, eager. Thermal-gate <= 80C, 0 MiB before/after each boot.

1. **Regression first**: DCP-off gemma TP=3 boot byte-identical sizing +
   temp-0 coherence vs the #90-recorded values; Qwen3.6-27B weighted-DCP
   boot unchanged (non-SWA path untouched).
2. **Stage A**: capacity table (max_total, per-pool sizes, per-rank free) at
   ctx {8192, 32768} x reqs {1, 4}, sizing-only vs baseline; deterministic
   across 2 boots; needle 8k + 20k; coherence vs DCP-off oracle.
3. **Stage B (if approved)**: same table with DCP on (+ `--rank-kv-ratio
   capacity` combined), coherence vs DCP-off oracle, needle 8k + 20k,
   deep-decode tok/s shallow/8k/24k vs Stage A, zero rank divergence in
   logs, 2-boot deterministic sizing.
4. Results recorded before any commit (job tmp + commit bodies).
