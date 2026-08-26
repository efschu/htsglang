# Decoupled KV-token ownership ratio (`--rank-kv-ratio`) — design note (Task #88)

Status: IMPLEMENTED + VALIDATED (see §9). Branch `feat/kv-split-ratio`, base `5e4fbed68`.

## 1. The idea

Under uneven DCP (token-axis KV sharding across an uneven-TP group), a context
token's *home rank* determines only **where that token's attention math runs**.
The query broadcast and the LSE merge exchange KB-sized per-head tensors whose
size is independent of how many tokens each rank owns. KV-token ownership is
therefore a **free placement knob**, decoupled from the projection/GEMM weight
split:

- **Weight/projection ratio** stays *speed-weighted* (concentrate GEMM mass on
  the fast card — the existing `--rank-tp-ratio` / auto-performance machinery).
- **KV-token ownership ratio** becomes *capacity-weighted* (concentrate context
  tokens on whichever cards have spare VRAM after their weight shards — on this
  rig the 3080s, whose VRAM is large relative to their compute share).

Result: **more max context from the same rig**. The known, honest cost: at deep
context, a larger fraction of per-token attention FLOPs/bandwidth runs on the
slower cards, so deep-context decode slows down. It is a capacity-vs-depth-speed
slider, not a free lunch.

## 2. What already exists (read before designing — findings)

The fork's uneven-DCP v3 already carries a token vector that is *internally*
separate from the weight vector:

| Piece | Where | Behavior today |
|---|---|---|
| Token vector storage | `distributed/utils.py` `_CP_TOKEN_RATIOS`, `set/get_cp_token_ratios` | process-global, installed once at boot |
| Owner rule | `cp_token_prefix` / `uneven_dcp_owner_bounds`; consumed by `flashinfer_backend` (`cp_lo/cp_hi` snapshot at backend init), triton backend, HiCache `cache_controller` | global slot `L` owned by rank r iff `L % S ∈ [lo_r, hi_r)`, `S = sum(ratios)`; compact physical slot `(L//S)*(hi-lo) + (L%S-lo)` |
| Resolution chain | `resolve_cp_token_ratios` (`distributed/utils.py`) | `SGLANG_UNEVEN_TOKEN_VECTOR` env → budget-estimate (`budget_mib − checkpoint_share − 1536 MiB`, 64-unit largest-remainder, gcd-reduced) → gcd-reduced `--rank-tp-ratio` weights |
| Install gate | `scheduler.py` `configure_scheduler_process` | requires `dcp_size>1` + base plan + `SGLANG_UNEVEN_DCP_WEIGHTED=1` (env) |
| DCP auto-set | `server_args.py` `_handle_uneven_tp` | `dcp_size := tp_size` when uneven plan + `SGLANG_UNEVEN_DCP=1` (env) |
| Sizing | `model_runner_kv_cache_mixin.py` `_apply_token_constraints` | context `C = min_r(P_r // ratio_r) * S` (min-reduce over ranks); per-rank pool `C * ratio_r / S`; allocator index space `C` |
| Self-calibration | `_maybe_suggest_dcp_token_vector` | measures per-rank `P_r` post-weight-load, computes the **optimal** vector `gcd_reduce(partition_units(64, P_r))`, but only logs a **restart hint** (`SGLANG_UNEVEN_TOKEN_VECTOR=...`) — two-boot convergence |
| Auto-performance | `uneven_perf.py` `predict_capacity` | already predicts `ctx = min(sum P, 64·min P)` — i.e. it *assumes* the converged capacity-optimal token vector; the MLP/GEMM vector choice is independent of it |

So the delta for #88 is **not** new sharding machinery. It is: (a) a first-class
knob, (b) one-boot convergence to the *measured* capacity optimum, (c) removing
the env-var gymnastics for the decoupled mode, (d) documenting the interaction
points.

### Why not derive 'capacity' from `PerfCostModel`?

`PerfCostModel` has no MoE family; on Qwen3.6-A3B (`intermediate_size: null`,
`moe_intermediate_size: 512`, 256 experts) it cannot even parse the config.
That is a pre-existing limitation of auto-performance on MoE (reported to main,
not silently fixed here). The measured-`P_r` route below is model-agnostic
(FP8 / AWQ / GGUF / MoE / hybrid-mamba alike) and strictly more accurate.

## 3. The knob

```
--rank-kv-ratio {coupled | capacity | auto | R0,R1,...,Rn-1}     default: coupled
```

- **`coupled`** (default): exactly today's behavior, **byte-identical**. All new
  code paths are inert; the env-gated chain above runs unchanged (including the
  restart-hint self-calibration).
- **`capacity`** (alias **`auto`**): decoupled, capacity-weighted ownership.
  - implies the weighted-DCP path — no `SGLANG_UNEVEN_DCP` /
    `SGLANG_UNEVEN_DCP_WEIGHTED` env needed (they remain honored for `coupled`);
  - phase 1 (scheduler configure): install today's pre-boot budget *estimate*
    vector so all gates (`uneven_dcp_active`, pool/page decisions) are stable;
  - phase 2 (model runner, post-weight-load profiling): **install** the measured
    optimal vector `gcd_reduce(partition_units(64, P_r))` instead of logging a
    restart hint — one-boot convergence. See §4 for why this point is safe.
- **explicit list** (e.g. `5,4,4`): pinned ownership vector (gcd-reduced;
  length = `tp_size`; positive ints). Phase 2 stays hint-only (a pin is a pin).
  `1,1,1` = uniform token ownership (even-modulo owner rule) under uneven
  weights — a legitimate point on the slider.
- Precedence (matches the mlp/moe/vocab family convention): env
  `SGLANG_UNEVEN_TOKEN_VECTOR` > flag list > capacity/coupled derivation.
  The env wins on **presence**, not on value, and it is never compared to the
  flag. #897: that loss is announced once per process by
  `distributed/utils.py` `announce_superseded_rank_kv_ratio`, called from the
  boot-time install site in `configure_scheduler_process` — the resolver
  itself stays a silent pure function. Unlike the role and provenance
  variables, `SGLANG_UNEVEN_TOKEN_VECTOR` is published from
  `--uneven-token-vector` only when that flag is set
  (`server_args.py` `_publish_promoted_781_flags`), so a value from an earlier
  process survives instead of being overwritten. To let the flag govern,
  REMOVE the variable — never blank it (`server_args.py:5607`).
- Validation (fail fast in `_handle_uneven_tp`): non-`coupled` requires
  `--rank-gpu-id` and an uneven `--rank-tp-ratio` plan; explicit list with a
  collapsed-to-even auto plan is a hard error; `capacity` on a collapsed plan
  degenerates to a no-op with a warning (nothing to rebalance).

## 4. Where the measured install happens (and why it is the only safe point)

The token vector must be **frozen before** anything snapshots it. Snapshot
consumers: `FlashInferAttnBackend.__init__` (`cp_lo/cp_hi`), pool construction
(per-rank pool size `C·ratio_r/S`), allocator sizing, CUDA-graph capture.

Order inside `ModelRunner`: load weights → `_profile_available_bytes` →
`_resolve_memory_pool_config` → pool build → attention backend init → capture.

The install therefore goes into `_resolve_memory_pool_config`, immediately
after profiling and **before** `_config_from_budget`:

1. every rank computes `P_r` = `calculate_pool_sizes(available_bytes).max_total_num_tokens`
   (exactly what `_maybe_suggest_dcp_token_vector` already computes);
2. `all_gather_object` of `(dcp_rank, P_r)` on the CPU group (already exists);
3. every rank derives the identical `optimal = gcd_reduce(partition_units(64, P_by_rank))`
   and calls `set_cp_token_ratios(optimal)`;
4. `_apply_token_constraints` then min-reduces `C = min_r(P_r // ratio_r) · S`
   with the *installed* vector — pools, allocator, backends, radix/HiCache all
   read the same vector afterwards.

Determinism (requirement 5): the vector is a pure function of the all-gathered
`P_r` list (ordered by `attn_dcp_rank`) + config — same invariant class as the
draft-length broadcast. Every rank runs the same collective and the same
arithmetic; there is no rank-local branch after the gather (the pre-gather
`local_p <= 0` early-return is the pre-existing divergence bug reported to
main; the capacity path must not replicate it).

Post-capture KV sizing (`post_capture_resize_kv_pool`): the vector is already
frozen (backends + graphs captured), so that path keeps hint-only semantics in
every mode; the pre-capture measurement is what capacity mode converges on.
Residual pre/post-capture deltas remain visible as the usual restart hint.

## 5. Interaction points (requirement 4)

- **(a) owner rule**: `cp_token_prefix`/`uneven_dcp_owner_bounds` take any
  positive vector; the virtual block is `S = sum(ratios)` after gcd reduction
  (derived vectors come from a 64-unit largest-remainder partition, so `S ≤ 64`).
  No kernel changes.
- **(b) per-rank KV pool sizing**: already follows the *token* vector, not the
  weight vector (`C·ratio_r/S` in `HybridLinearKVPool` sizing and
  `_apply_token_constraints`). Installing a different vector is sufficient;
  no sizing code forks.
- **(c) LSE merge**: `cp_lse_ag_out_ar_mha_uneven` merges per-head partials by
  LSE weights; its shapes depend on the *head* partition only. Token-ownership
  asymmetry changes which tokens contribute to which rank's partial, and the
  log-sum-exp merge is exact for any partition of the context. Verified
  empirically via temp-0 coherence + needle tests (§7).
- **(d) MTP/draft KV + mamba state**: unchanged by design, documented:
  the draft worker keeps a plain uneven-TP pool — LOCAL head-sharded KV, FULL
  token context, raw allocator index space (not DCP-token-sharded; see the
  draft gate in `FlashInferAttnBackend.__init__` and the `_draft_non_dcp`
  branch in pool construction). Mamba/GDN state is per-request, placed by the
  GDN *unit* partition (follows `--rank-tp-ratio` on the GDN grid), never by
  the token vector.
- **(e) prefix-cache/radix + HiCache**: both operate on GLOBAL allocator
  indices; device↔host transfers map through `uneven_dcp_owner_bounds()`
  (cache_controller caches it lazily at first transfer, long after install).
  Because the vector is installed before any cache exists and never changes
  afterwards, ownership is consistent for the whole process lifetime.

## 6. Auto-performance integration (requirement 3)

`apply_auto_performance` chooses the MLP/GEMM family vector from the measured
hardware profile; `predict_capacity` scores candidates under the *assumption*
of a converged capacity-optimal token vector (`ctx = min(sum P, 64·min P)`).

- Under `--rank-kv-ratio capacity`, that assumption becomes **true on the first
  boot** — the context floor (`--rank-perf-loose-ctx-percent`) and the tune
  targets (`--rank-perf-tune both|dec|enc`) keep their exact semantics and
  become *accurate* rather than aspirational. The solver needs no change; the
  KV ratio is chosen independently of the GEMM ratio by construction.
- Under `coupled`, nothing changes (hint-based two-boot convergence).
- The auto-performance log block gains one line naming the active KV-ratio mode
  so the decision inputs stay auditable.

## 7. Validation plan (deliverable table)

Rig: 5090 32G + 2×3080 20G (one on PCIe x4), TP=3, 5090 resolved by name.
Models: Qwen3.6-27B-FP8 (dense) and Qwen3.6-35B-A3B-FP8 (MoE).

1. **Correctness first** (27B, TP=3): capacity mode vs coupled-mode oracle,
   temp-0 coherence; needle-in-haystack at 8k and 20k. Zero rank divergence.
2. **Capacity gain**: `max_total_num_tokens` coupled vs capacity, 27B + A3B.
3. **Cost**: decode tok/s shallow, deep-8k, deep-24k, coupled vs capacity.
4. **Regression**: default (coupled) boot on the feature branch vs base commit:
   byte-identical sizing lines + tok/s within noise.

Results land in the commit body and the accompanying results notes.

## 8. Touch list (small, gated)

| File | Change |
|---|---|
| `server_args.py` | `rank_kv_ratio` field + parser + help; validation in `_handle_uneven_tp`; helper `uneven_kv_flag_active()` / `uneven_weighted_dcp_requested()`; extend the two env gates (dcp auto-set, spec-decode validation) to accept the flag |
| `distributed/utils.py` | `resolve_cp_token_ratios`: flag-list branch between env and estimate |
| `managers/scheduler.py` | weighted-install gate: env **or** flag |
| `model_runner_kv_cache_mixin.py` | factor the measured-vector math out of `_maybe_suggest_dcp_token_vector`; capacity-mode install in `_resolve_memory_pool_config` (both post-capture-planned and plain paths); hint path byte-identical for `coupled` |
| `uneven_perf.py` | one log line naming the KV-ratio mode |

Default-path guarantee: with `--rank-kv-ratio` unset, every new branch is
behind `!= "coupled"` checks; no existing line's behavior changes.

## 9. Measured results (2026-07-18, this rig)

Capacity gain (`max_total_num_tokens`, KV pool budget):

| Model | coupled | capacity | gain |
|---|---|---|---|
| Qwen3.6-27B-FP8 | 443,904 ([30,17,17]) | 563,456 ([33,13,18]) | **+26.9%** |
| Qwen3.6-35B-A3B-FP8 | 1,911,488 pre-cap | 2,187,648 pre-cap | +14.4% |

27B pool-end free memory: coupled 5.21/2.33/3.58 GB (stranded on the
3080s) → capacity 2.71/2.46/2.33 GB (balanced). A3B: the #79 mamba
ceiling (16 reqs × ctx = 655,440) binds first at these settings in both
modes; the pool gain becomes usable with more requests or longer ctx.

Cost (decode tok/s, bs=1): within noise (±1%) at shallow, 8k and 24k
depth on both models — decode is weight-streaming-bound here, so the
slider costs ~nothing at these depths on this rig.

Correctness: needle retrieval (~7.1k and ~17.7k tokens) correct in every
mode on both models; temp-0 outputs bit-identical vs the coupled oracle
except at known near-tie loci (think/no-think flips) that flip across
boots even on identical pre-feature code (endemic baseline
nondeterminism, verified by double-booting the base commit); within-boot
reruns bit-deterministic; zero rank divergence in any log. Coupled-path
regression: sizing byte-identical pre/post feature, tok/s within noise.
Full tables: T88_RESULTS.md (job tmp) and the feature commit bodies.
