# Task #96 — SWA-DCP Stage B: uneven DCP for the GLOBAL layers of an SWA-hybrid model (Triton)

Design note for putting the fork's weighted (uneven) DCP token split under the
~10 **global full-attention** layers of a Gemma-4-class hybrid, while the ~50
**sliding-window** layers keep their existing local path.

Base: `integration/r3-probe` @ `1ff5178fd5` (carries #169.1-.4, #173.1-.2,
#172a/b). Branch `feat/swa-dcp-triton`. No GPU was available for this change —
§8 is the recipe, §9 the honest status.

---

## 0. Two premises checked before designing (one of them was wrong)

**(a) "Stage A (#91) is the flashinfer reference for how an SWA window and a
weighted token split fit together — read it as the template."**

It is not, and no such reference exists anywhere in the tree. What #91 actually
shipped (`docs/advanced_features/swa_dcp_design.md` §0) is
`--swa-pool-sizing {ratio,cap}`: a **pure pool-sizing change with DCP OFF**
(`server_args.py`, `pool_configurator.py`,
`model_runner_kv_cache_mixin.py::_swa_hybrid_kv_token_cap`). Measured win 6.1x
served capacity on Gemma-4-31B TP=3 — all of it from decoupling the SWA pool
from the context budget `C`, none of it from token sharding.

Checked against the code, not the doc: `grep -i 'swa\|sliding_window'` over
`flashinfer_backend.py` has **zero** hits in any DCP context, and
`flashinfer_backend.py`'s DCP lane never looks at a window. So there are no
"per-rank window lengths" to port. Stage A is not a template for Stage B; it is
the *prerequisite* (§4.1) for it.

**(b) "The core problem is the window cut per rank: the last W positions are
spread over several owner slices, so each rank must compute
(owner slice ∩ window)."**

That is the core problem of the design #91 §4 **considered and rejected**
("window-aware ownership": shard the SWA tokens too). It costs an LSE merge on
60 layers instead of 10 (6x per-step DCP comm), replicates kv-head write
traffic across 50 layers, and needs ownership to follow the *ring slot* rather
than the token position (or eviction lag makes owner sets diverge across
ranks). It buys at most ~1.4-2.4 GB/rank.

The design that was chosen — and that this note implements — is
**replicate-SWA / shard-global-only**. Under it there is no window ∩ owner-slice
intersection anywhere:

* an SWA layer's KV is **not** token-sharded, so every rank holds every
  in-window position of its own kv-head shard;
* the window read stays `update_sliding_window_buffer()`: global
  `req_to_token` rows for `[seq_len - min(seq_len, W), seq_len)`, translated
  `full -> swa` through `full_to_swa_index_mapping`, which is **position**-keyed
  and not ownership-keyed;
* the SWA slot assignment is rank-uniform because allocation is rank-uniform
  (the full-side allocator index space is the *global* `C` on every rank, so
  every rank makes the identical allocation decisions).

So Stage B is a **layer-type split**, not a window-slice intersection. Writing
an `owner.py` window ∩ slice helper now would be dead code for the shipped
design; the honest place for that math is the day window sharding is revisited,
and #91 §4 is the record of why that day has not come. What Stage B *does*
need from the shared module is the compact-row sizing expression (§3.4) — added
there, used by both hybrid pool families.

---

## 1. What Stage B actually is, in one table

| layer type | count (31B) | kv heads per row | token axis | collectives per layer |
|---|---|---|---|---|
| sliding window | 50 | this rank's SWA shard (`get_swa_num_kv_heads`) | **not sharded** (all positions, window-bounded) | none |
| global / full | 10 | ALL `num_key_value_heads` (replicated rows) | weighted owner rule (#173) | kv-head all-gather on write (only if `kv >= tp`), q all-gather + LSE merge on read |

The SWA half is therefore **exactly** the already-validated non-DCP uneven-TP
SWA path, condition for condition. The full half is **exactly** #173's weighted
Triton lane. Stage B is the wiring that lets both live in one model, plus the
sizing that makes it fit.

The expected win remains the ex-ante estimate from #91 §5: **+6-10%** context
on top of Stage A on this rig (10/60 layers shard, and their `[4,8,4]` kv shards
already track the VRAM distribution `[~26/47/27]%` closely, so pooling recovers
little). Stage B is *not* justified by capacity alone on this rig — it is
justified by making Gemma and Qwen share one operational DCP mode, and by
sharding the only KV that grows with context.

---

## 2. Gemma-4's geometry, measured from the code (this is where the traps are)

`models/gemma4_causal.py:296-401`: **each layer type carries its own kv-head
base and its own q partition.**

```
sliding layer: total_num_kv_heads = swa_num_key_value_heads (fallback: num_key_value_heads)
full    layer: total_num_kv_heads = num_key_value_heads
per type:      _attn_q_units  = attn_q_partition_units(total_q, total_kv_of_this_type, tp)
               _attn_q_groups = attn_q_partition_groups(total_kv_of_this_type, tp)
               num_heads      = tp_partition_size(total_q, tp, rank, units, groups=groups)
```

Consequences that Stage B has to respect:

1. **The per-rank q shard differs between the two layer types.** #91 measured
   Gemma-4-31B TP=3 uneven: q sliding `[8,14,10]`, q full `[8,16,8]` (both
   exhaustive, both summing to 32).
2. **One layer type can be REPLICATED-KV while the other is head-sharded.**
   Gemma-4-26B: sliding kv=8 (head-sharded at tp=3), full kv=2 (`kv < tp` ->
   replicated). `attn_kv_replicated()` must therefore be evaluated on the
   **full** base for everything DCP does, because the full layers are the only
   ones DCP touches. `TritonAttnBackend.__init__` already reads
   `_total_kv = get_total_num_kv_heads()` (the full base) for
   `dcp_kv_replicated_heads` / `dcp_kv_head_counts` / `num_kv_head`, so that
   part is right by accident of construction — this note pins it as intent.
3. **The head *dims* can differ too** (`swa_head_dim` / `swa_v_head_dim`), which
   is why `ForwardMetadata.swa_attn_logits` and `self.swa_v_head_dim` exist.
   Untouched here.

### 2.1 A real bug found in #169.3's max()-over-bases, and why Stage B is where it bites

`_plan_aware_dcp_group_q_head_counts()` (triton_backend.py:199) takes
`max()` over both kv bases per rank. For **workspace sizing**
(`_plan_aware_num_q_heads`, `_plan_aware_dcp_gathered_q_heads`) that is the
correct direction to be wrong in — over-allocating is harmless. For the
**collective head counts** it is not: those must be *exact and exhaustive*.

With the two vectors of §2.1 above, `max()` yields `[8,16,10]` — **sum 34
against a total of 32**. `cp_all_gather_heads_uneven` asserts
`counts[rank] == local_heads` (rank 2 would see 10 vs its real 8) and
`_dcp_merge_q_heads` asserts `sum(counts) == out.shape[1]`. So the first
gathered-q forward of a hybrid model fails loudly — or, on a plan where the
assertion happens to pass, slices the wrong heads out of the merge.

CPU-reproducible instance (used as the regression test): `total_q=32`,
bases `{16, 8}`, tp=dcp=3, ratio vector `[5,3,2]`:

```
base 16 -> units 16 -> [8,5,3] units -> q [16,10,6]   (exhaustive)
base  8 -> units  8 -> [4,2,2] units -> q [16, 8, 8]  (exhaustive)
max()   ->                              [16,10,8]      sum 34 != 32   <- the bug
```

**Fix:** the group's q-head counts come from the **full-attention base only**,
because under Stage B the DCP collectives run on full-attention layers only.
Plus an exhaustiveness assertion (`sum(counts) == total_q` when `dcp == tp`), so
a future second base cannot re-introduce a non-partition silently. Models with
a single kv base are byte-identical (the set has one element).

---

## 3. The change, part by part

### 3.1 One lane predicate, one dispatch predicate (shared, pure)

Added to `layers/dcp/owner.py` (re-exported from `layers/dcp/__init__.py`):

```python
swa_hybrid_dcp_lane(*, is_hybrid_swa, uneven_plan, is_draft_worker,
                    num_full_layers, num_swa_layers, swa_pool_sizing) -> bool
dcp_token_sharded_layer(is_swa_layer: bool, *, swa_hybrid_lane: bool) -> bool
dcp_compact_pool_rows(global_tokens: int, cp_S: int, cp_ratio: int) -> int
```

`swa_hybrid_dcp_lane` is the single definition of "this process is serving
Stage B", consumed by the pool sizing (`model_runner_kv_cache_mixin`), the
attention backend, and the geometry guard. Being a pure function of named
booleans/ints, it is unit-tested without a device, and — crucially — it is a
function of **process-global configuration only**, never of anything derived
from a batch or a rank's data. Every rank of the group computes the same value.

`dcp_token_sharded_layer` is the per-layer dispatch:
`not (swa_hybrid_lane and is_swa_layer)`.

**Why this is the [[rank-lokaler-test-vor-kollektiv]] shape and not a violation
of it** (5 prior sightings; the #173 D5 case and the #180 D5 second-door lesson
are the two nearest): the predicate's inputs are `layer.sliding_window_size`
(model config) and the lane flag (server args + model config). Both are
identical on every rank. A rank can therefore never enter the q all-gather /
LSE merge for a layer another rank skips. The forbidden shape would be deciding
per layer from something rank-local — e.g. "does this rank own any window
slots" — and that is exactly what the rejected design of §0(b) would have
forced. The dispatch is **layer-type-first**, the same discipline #180
established for the verify split (`forward_mode`-first).

### 3.2 `triton_backend.py`

| site | change |
|---|---|
| `__init__` | compute `self.swa_hybrid_dcp` from `swa_hybrid_dcp_lane(...)`; pass it to the geometry guard |
| `_plan_aware_dcp_group_q_head_counts` | full-attention base only + exhaustiveness assert (§2.1) |
| `_dcp_layer_token_sharded(layer)` | new one-line helper over `dcp_token_sharded_layer` |
| `_set_kv_buffer` | SWA layer under the lane -> the plain non-DCP write (`KVWriteLoc` -> `swa_loc`, no kv-head gather, no owner mask, no `dcp_kv_mask`) |
| `forward_extend` | `if self.dcp_size > 1 and self._dcp_layer_token_sharded(layer)` -> `_forward_extend_dcp`; SWA layers fall through to the existing 2-stage window path. The `NotImplementedError("...sliding window")` inside `_forward_extend_dcp` becomes unreachable **by construction** and is kept as the assertion it now is |
| `forward_decode` | same gate around the DCP gather/merge branch; SWA layers take the stock decode call with `window_kv_indptr/indices` |
| `reject_unsupported_dcp_geometry` | the `sliding_window` refusal becomes conditional on the lane (§5) |

Metadata needs **no** new field: `init_forward_metadata` already builds the
window buffers and the DCP-sharded `kv_indptr/kv_indices` side by side
(:1295-1314 eager, `_update_decode_kv_buffers` :969-1004 for cuda-graph), for
every mode. Stage B only changes which of the two a given layer reads.

Two deliberate non-changes:

* `num_kv_splits` for SWA layers stays the DCP-owned-length-derived tensor
  (`forward_decode` passes `forward_metadata.num_kv_splits` for every layer,
  as upstream does — `window_num_kv_splits` is built and carried but has no
  consumer). This is **correctness-neutral**: the decode kernel derives
  `kv_len_per_split` from `cur_batch_seq_len` read out of `kv_indptr`
  (`decode_attention.py:158-163`), so any split count >= 1 covers the whole
  range; only the split *granularity* (a perf heuristic) is off. Wiring
  `window_num_kv_splits` in would change the reduction order on the default
  SWA path, i.e. break byte-identity for a perf guess. Left as a follow-up.
* `attn_logits` / `attn_lse` are sized for the **gathered** head count under
  DCP (`self.num_head = _plan_aware_dcp_gathered_q_heads`). An SWA layer writes
  only its local heads into that buffer — over-allocated, never under.

### 3.3 `SWAKVPool` (`mem_cache/swa_memory_pool.py`)

Only one signature change is needed: `set_kv_buffer(..., dcp_kv_mask=None)`,
forwarded to the **full** sub-pool only, with a hard assert that no mask ever
arrives on the SWA branch (an SWA write is unsharded by construction, so a mask
there would mean the dispatch of §3.1 leaked).

The **head-count split needs no new plumbing**: `MHATokenToKVPool` already
honours `swa_head_num` / `swa_head_dim` / `swa_v_head_dim`
(`memory_pool.py:1686-1692`) and `SWAKVPool.__init__` already passes kwargs to
the SWA sub-pool before popping them for the full one
(`swa_memory_pool.py:60-74`). Today only `is_hybrid_swa_compress` models (which
include every Gemma-4 arch) pass them. Under the lane the mixin passes:

* `head_num = get_total_num_kv_heads()` — the FULL sub-pool's replicated rows;
* `swa_head_num = get_swa_num_kv_heads(attn_tp_size)` — this rank's SWA shard,
  unchanged semantics, now passed for *any* hybrid model under the lane and not
  only the compress list.

`finalize_backing()` (post-capture VA resize) would rewrite the full sub-pool
back to the global `C`. It cannot be reached here:
`post_capture_kv_sizing_planned()` requires `dcp_size == 1`
(`server_args.py:6361`). Pinned with a comment + a test asserting that
precondition, rather than a second sizing path.

### 3.4 Sizing chain (`pool_configurator.py`, `model_runner_kv_cache_mixin.py`)

The chain already has the right shape; Stage B changes exactly two numbers in
it. With `S = sum(token_ratios)` and `ratio_r` this rank's share:

```
SWAChunkCapPoolConfigurator.calculate_pool_sizes(available_bytes):
    swa_tokens      = ceil_align(swa_pool_token_cap, page)          # rank-local CONSTANT
    fixed_swa_bytes = swa_tokens * swa_per_token * swa_layers       # unsharded, per-rank kv shard
    full_cell       = full_per_token * (full_layers + draft_full_layers)
       ^ (1) under the lane full_per_token uses get_total_num_kv_heads()
             (replicated rows), not get_num_kv_heads(tp)
    full_tokens     = (available_bytes - fixed_swa_bytes) // full_cell        # = P_r

_apply_token_constraints(P_r):                        # unchanged, already DCP-aware
    C = min_r(P_r // ratio_r) * S                     # all-reduce MIN over the world
    C = min(C, user_limit, swa_hybrid_kv_token_cap)   # #90 reachability, cap mode -> full_need

calculate_pool_sizes_from_max_tokens(C):
    full_max_total_num_tokens = C                     # GLOBAL: allocator index space
    swa_max_total_num_tokens  = swa_cap               # LOCAL constant

SWAKVPool(size=..., size_swa=swa_cap, head_num=..., swa_head_num=...):
       ^ (2) under the lane size = dcp_compact_pool_rows(C, S, ratio_r)
             = (C // S + 1) * ratio_r    -- this rank's OWNED rows only
```

`(2)` is the same expression the mambaish `HybridLinearKVPool` branch already
uses inline (`model_runner_kv_cache_mixin.py:2410`), including the `+ 1`
ceil-to-a-whole-owner-block that fixed the out-of-bounds scatter on an
unaligned `--max-total-tokens`. It is **extracted** into
`dcp_compact_pool_rows()` and both call sites now share it — no second copy of
a sizing rule whose off-by-one has already cost one debugging round.

`SWATokenToKVPoolAllocator` needs **no** change: it is constructed with
`(full_max_total_num_tokens, swa_max_total_num_tokens)`, i.e. the full side's
index space is already the global `C` (exactly what the weighted owner rule
requires), the SWA side is already the local pool, and
`full_to_swa_index_mapping` is already sized off the full space (`C + page + 1`
int64 ≈ 2 MB per 256k context — negligible, and it must be that big because
every rank translates every global loc).

### 3.5 Why `--swa-pool-sizing cap` is a hard precondition of the lane

In `ratio` mode `swa_tokens = ratio * full_tokens = ratio * C`. Under DCP `C`
is the *global* budget, several times any single rank's reach, and the SWA pool
is **not** sharded — so a ratio-sized SWA pool is precisely the pre-#90 OOM
disease, with the multiplier the token split just introduced. The lane
therefore requires `cap` (or the legacy `--disable-radix-cache` route, which
selects the same configurator). Anything else fails fast at boot, naming the
flag. This also means Stage B *strictly* builds on Stage A, as #91 predicted.

---

## 4. Preconditions, and what is refused

### 4.1 Required for the lane to activate

* hybrid SWA model with `len(full_attention_layer_ids) > 0` (a pure-SWA model
  has nothing to shard: DCP would be a no-op with extra collectives);
* an installed `--rank-tp-ratio` plan with `dcp_size == tp_size`
  (`uneven_dcp_kv_replicated`), target worker (not the draft worker);
* `--swa-pool-sizing cap` (or `--disable-radix-cache`), i.e. Stage A active.

### 4.2 Refused, loudly, at boot

| combination | why |
|---|---|
| SWA + uneven DCP **without** cap sizing | §3.5 |
| SWA + uneven DCP + TREE-masked speculative verify (`--speculative-eagle-topk > 1`, or the DFLASH tree-verify door) | unchanged #173 refusal. CHAIN verify (topk == 1) is **served** here since the #96 x #180 rebase — see §10 |
| SWA + uneven DCP + MLA / weightless-KV | unchanged #173 refusals |
| SWA + uneven DCP + HiCache | `cache_controller._dcp_kv_transfer_pairs` compacts *both* device index streams through the owner rule; the SWA stream must stay local. #91 §3(b) already scoped this out of v1 — now it is a gate instead of a silent hole |
| pure-SWA model (no global layers) + uneven DCP | nothing to shard (§4.1) |
| unified memory pool + DCP | pre-existing assert (`unified_memory_pool.py:441`, `server_args.py:8882`) |

---

## 5. What the geometry guard now lets through

`reject_unsupported_dcp_geometry()` keeps its three branches. Inside branch 1
(the uneven lane) the `sliding_window` reason becomes conditional:

```
sliding_window and not swa_hybrid_dcp   -> refuse (unchanged message)
sliding_window and     swa_hybrid_dcp   -> serve  (Stage B, this change)
```

`swa_hybrid_dcp` is computed by the constructor from `swa_hybrid_dcp_lane(...)`
and passed by value, so the rule stays a pure function of its inputs and the
test file can enumerate the matrix without a device.

Newly served: **SWA-hybrid + uneven plan + (weighted or even-modulo owner rule)
+ cap sizing + global layers present**.
Still refused, with the same words as before: a window model on the uneven lane
**without** the Stage-B preconditions (no cap sizing / no global layers /
draft worker / HiCache), and every non-window refusal of #173 (weightless, MLA,
speculative). Branch 2 (token vector without a plan) and branch 3 (even DCP
replication arithmetic over `max(full, swa)` kv bases, #169.3) are untouched.

---

## 6. Files touched

| file | what |
|---|---|
| `python/sglang/srt/layers/dcp/owner.py` | `swa_hybrid_dcp_lane`, `dcp_token_sharded_layer`, `dcp_compact_pool_rows` |
| `python/sglang/srt/layers/dcp/__init__.py` | re-exports |
| `python/sglang/srt/layers/attention/triton_backend.py` | §3.2 |
| `python/sglang/srt/mem_cache/swa_memory_pool.py` | `dcp_kv_mask` on the full branch |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | lane sizing + head split + fail-fast gates; mambaish reuses `dcp_compact_pool_rows` |
| `python/sglang/srt/model_executor/pool_configurator.py` | replicated full cell under the lane |
| `test/registered/unit/distributed/test_swa_dcp_stage_b.py` | new, CPU |
| `test/registered/unit/distributed/test_triton_dcp_geometry_guard.py` | the window refusal is now conditional |

---

## 7. CPU test matrix (what is actually pinned without a GPU)

1. `dcp_compact_pool_rows`: window/edge cases — `C < S`, `C % S != 0`,
   `ratio == 1`, `ratio == S` (single rank owns everything), and the property
   that the compact row of the **largest** allocator slot `C` is inside the
   sized pool for every rank (the #2410 off-by-one, as an assertion rather than
   a comment).
2. `swa_hybrid_dcp_lane`: full truth table over its six inputs; in particular
   OFF for the draft worker, OFF without a plan, OFF in ratio sizing, OFF for a
   pure-SWA model.
3. `dcp_token_sharded_layer`: off-lane every layer is sharded (byte-identical
   #173 behaviour); on-lane exactly the SWA layers are not.
4. Group q-head counts (§2.1): the `[5,3,2]` two-base instance — old max()
   gives a non-partition, the fix gives the exhaustive full-base vector; a
   single-base model is unchanged; the exhaustiveness assert fires on a
   deliberately non-exhaustive stub.
5. Guard matrix (§5), including that a window model *without* the lane still
   gets the old message.
6. Source-pinned invariants (the shape #173 already pins this way): the SWA
   write branch cannot pass `dcp_kv_mask`; the per-layer dispatch is spelled
   with `dcp_token_sharded_layer` in all three call sites, so a later edit
   cannot re-route SWA layers into the DCP path silently.
7. The full pre-existing DCP/guard CPU suite, diffed: **37 passed before, and
   the same 37 after** (one expectation intentionally rewritten, see §9).

## 8. GPU validation recipe (nothing below was run — no cards in this window)

Vehicle: `gemma-4-31B-it-int4-AutoRound`, TP=3 uneven on 5090 + 2x3080, 5090
resolved **by NVML name at runtime** (never a hardcoded index), thermal gate
<= 80C, 0 MiB before/after each boot, `--attention-backend triton`.
Cheapest-first, each step a falsifier for the next ("Einzelteil vor Verbund").

**H1 — the CPU suite on the box** (seconds): §7, plus the #173 G1 index-math
GPU test (`test_triton_weighted_dcp_gpu.py`) unchanged — Stage B must not
perturb it.

**H2 — the guard opens, and only where it should** (~2 min, one card): boot the
Stage-B config and confirm the process gets past `TritonAttnBackend.__init__`.
Then confirm each refusal of §4.2 still fires by adding it to the same command:
`--swa-pool-sizing ratio`, `--enable-hierarchical-cache`, a speculative config.
Each must abort at boot naming itself.

**H3 — DCP-off regression, byte-identical** (~5 min): Gemma-4-31B TP=3 uneven
**without** `--dcp-size`, `--swa-pool-sizing cap`. Sizing lines and temp-0
output must equal the #91 Stage-A recorded values line for line
(`full/swa = 262160 / <cap>`, needle at 50,168 prompt tokens). This is the
"default path unchanged" gate: Stage B must be inert here.

**H4 — Stage B boots and sizes as designed** (~10 min): add `--dcp-size 3` +
the token vector. Record `full_max_total_num_tokens` (global C),
`swa_max_total_num_tokens` (the cap), and the per-rank SWA-pool log line
`SWAKVPool mem usage ... full size: <rows>`; assert
`rows == (C // S + 1) * ratio_r` per rank and that the SWA size is identical on
all three ranks. Two boots, deterministic line for line.

**H5 — the coherence anchor** (the point of the task): same model, same plan,
greedy/temp-0, no speculative decoding.

* A: `--attention-backend flashinfer`, DCP off (Stage A) — the oracle.
* B: `--attention-backend triton`, DCP off — separates "triton SWA" from
  "triton SWA under DCP".
* C: `--attention-backend triton`, Stage B on.

Bar: C coherent on its own; C vs B token-identical for as long as possible with
**first divergence late, never at token 1**; CJK/mojibake or a token-1
divergence means the owner rule or the layer dispatch is wired wrong, not that
kernels differ. Prompt set: one short (single chunk: current-chunk ragged stage
only) and one long enough to chunk (paged owned-prefix read + LSE merge), plus
one needle > window so the answer *requires* a global layer to carry it — that
last one is the specific falsifier for "the global layers lost context in the
merge", which a short prompt cannot see because the window alone answers it.

**Determinism control, designed around the G4 finding that flashinfer is NOT
self-deterministic:** do not use flashinfer as the byte-identity reference.
Run the self-determinism check on the **triton** arms only (B and C, 3x each,
same seed, same prompt -> byte-identical token ids), and use flashinfer only as
a *semantic* oracle. A B-vs-B or C-vs-C divergence is a real defect; an A-vs-C
byte divergence is expected.

**H6 — with CUDA graphs, not only eager** ("Full-Perf-Testen"): repeat H5-C
with graph decode enabled. This is what exercises #173's D3 capture-stable
buffer contract *while* the same graph also contains the window reads of 50 SWA
layers; a wrong-context replay shows up only here. Gemma bring-up historically
ran `--disable-cuda-graph`, so expect this step to be the one that finds
something.

**H7 — the empty-shard provocation** (#173 G6, re-run under the hybrid): a
1-2 token prefix under a strongly uneven vector (`[13,30,21]`), so at least one
rank owns zero prefix rows *while all 50 SWA layers have work*. Must complete,
not hang.

**H8 — capacity table, the actual claim**: Stage A vs Stage B, ctx {8192,
32768, 65536} x reqs {1,4}: `max_total_num_tokens`, per-rank pool bytes,
per-rank free MiB (>= 400 MiB per the VRAM corridor rule), decode tok/s
shallow/8k/24k. The +6-10% estimate is confirmed or corrected here — and if it
lands at or below the measurement noise, that is the number that goes into
`FEATURES_VS_UPSTREAM.md`, not the estimate.

## 9. Open points / risks

1. **Zero GPU evidence.** Everything in §3 is CPU-pinned index/geometry math
   plus wiring; the numerics of a Gemma-4 forward with 10 sharded and 50
   unsharded layers has never run. §8 H5/H6 are the gates.
2. **The head-sharded write gather (`kv >= tp`) is still the untested sub-lane**
   from #173 G5 — and Gemma-4-31B (full kv=16, tp=3) *is* that sub-lane, so
   Stage B validates it for the first time. Gemma-4-26B is the mixed case
   (sliding head-sharded, full replicated) and is a second, independent test
   vehicle worth booting.
3. **`_plan_aware_dcp_group_q_head_counts` change touches #169.3's contract.**
   Single-base models are byte-identical; hybrid models were previously unable
   to reach it at all (the guard refused them), so nothing that works today
   changes. Recorded here because the "second kv-head base" reasoning is now
   split: **max() for workspaces, full base for collectives.**
4. **HiCache and post-capture sizing are gated, not solved** (§3.3, §4.2).
5. **The `num_kv_splits` granularity for SWA layers under DCP** is a known
   left-on-the-table perf item (§3.2), deliberately not taken to keep the
   default SWA path byte-identical.
6. **Global-index consumers of the full sub-pool, audited.** Anything that
   indexes the full sub-pool's *buffers* with GLOBAL allocator ids is wrong
   under a compact pool. Checked, with the result that none of them is reachable
   on this lane:
   * `SWAKVPool.move_kv_cache` — called only from the speculative paths
     (`spec_utils.py::move_accept_tokens_to_target_kvcache`,
     `base_spec_worker.py::duplicate_prefix_tail_to_draft_branches`). The
     original reason given here ("speculative decoding is refused on the lane")
     went stale with the #96 x #180 rebase, which serves the CHAIN verify. The
     conclusion survives on a different and narrower fact: **both call sites are
     tree-only.** `_finalize_accept_tree_path` is entered under
     `self.topk > 1` (`eagle_worker_v2.py:2707`) and the branch duplicator is
     built out of `topk - 1` branches, while the tree verify is still refused
     at boot. `ngram_worker.py:473` calls the mover unconditionally, but
     `NGRAM_VERIFY` is not in `_DCP_VERIFY_SPEC_INPUT_TYPES`, so that lane
     raises in the metadata build before the mover runs — loudly, at the first
     verify rather than at boot, which is a guard gap worth closing but not a
     silent one. Re-audit this bullet if the tree refusal is ever lifted: the
     mover hands GLOBAL allocator ids to a COMPACT full sub-pool;
   * `--enable-kv-session-offload` — requires a mambaish model, so a hybrid-SWA
     model raises before reaching it;
   * HiCache — gated (§4.2); unified memory pool — pre-existing assert.
   The one that is NOT lane-specific: `ScheduleBatch.offload_kv_cache` /
   `get_cpu_copy(global token indices)` is equally wrong under the *existing*
   qwen weighted-DCP lane, so it is a DCP-wide pre-existing gap and deliberately
   not "fixed" here (fixing it blind, with no way to run it, is how a working
   lane gets broken).
7. **Deterministic mode** (`--enable-deterministic-inference`) sends non-DCP
   layers through `_forward_extend_unified` and DCP layers through
   `_forward_extend_dcp`; on the lane a model would therefore mix the two extend
   kernels across layer types. DCP x deterministic was already unvalidated
   before #96 and stays so — worth an explicit refusal if it is ever asked for.
8. **One existing test expectation was intentionally rewritten**:
   `test_the_lane_still_refuses_what_has_no_triton_twin` asserted that a window
   under the uneven lane is always refused. It now asserts the *conditional*
   refusal (refused without the Stage-B preconditions, served with them). That
   is the guard change, made visible in the test, not a weakened test.

---

## 10. Task #191 — the `custom_mask` drop on a window layer under target-verify

The seam the #96 x #180 rebase left open, and its resolution. Settled on
hardware (5090 + 2x 3080, single card, Triton).

### 10.1 Which case can actually arise

Only one of the two. The mask is dropped, never wrongly kept:

* `init_forward_metadata`'s target-verify branch sets
  `custom_mask = mask_indptr = None` keyed on **`self.dcp_size > 1` alone**
  (`triton_backend.py:1567-1580`), and the cuda-graph twin does the same in two
  places (`:1176-1179` in `_update_target_verify_buffers`, `:1827-1833` in
  `_build_cuda_graph_forward_metadata`). No per-layer condition appears in any
  of them, and nothing re-installs a mask later.
* `forward_extend` (`:2242`) then sends only the token-sharded GLOBAL layers
  into `_forward_extend_dcp`. A sliding-window layer falls through to the
  ordinary 2-stage window path (`:2254-2334`) and reads
  `self.forward_metadata.custom_mask` at `:2321` — now `None`, where the
  non-DCP twin would hand it the EAGLE chain mask.

So the question is exactly: **is a window layer's chain verify correct without
the mask?** ("Mask wrongly present" cannot occur: the drop is unconditional
under DCP, and off the lane the mask is upstream's own behaviour.)

The seam is Triton-only. FlashInfer's DCP lane never looks at a window at all,
so the pairing does not exist there.

### 10.2 Why it is correct

A property of the kernel, not of the dispatch. In
`kernels/ops/attention/extend_attention.py`:

* **Paged stage.** The window mask at `:381-387` is applied under
  `if SLIDING_WINDOW_SIZE > 0`, with no reference to `USE_CUSTOM_MASK`, and it
  is per-query and absolute: `cur_seq_len_prefix + m <= n + W`, where
  `cur_seq_len_prefix` is `min(seq_len, W)` (the clipped window buffer) so the
  expression re-bases correctly to `q_abs <= k_abs + W`. Moreover the custom
  mask was **never** consulted in this stage: `skip_prefix_custom_mask`
  defaults to True (`:640`) and the Triton backend never overrides it. The two
  arms are bit-identical here by construction.
* **Ragged stage.** With the mask gone, `elif IS_CAUSAL` (`:514-519`) supplies
  `m >= n`, which for a topk == 1 chain is exactly the mask's d x d
  draft->draft block. The window mask at `:524-529` is again applied
  independently of `USE_CUSTOM_MASK`.

Hence the window is neither widened (no out-of-window key becomes visible) nor
narrowed (no in-window key is lost). `window_kv_offsets` is loaded only inside
the `USE_CUSTOM_MASK` branches, so dropping the mask makes it inert rather than
wrong.

`verify_splitkv_fwd` cannot take this case: its `can_handle` returns False for
any `sliding_window_size > 0`, so a window layer always reaches
`extend_attention_fwd`.

### 10.3 How it was falsified

The needle test of §8 H5 would have worked but is the expensive instrument, and
comparing tokens between spec-on and spec-off is not a valid gate anyway —
speculation breaks token identity at temperature 0 on the standard path. The
cheaper and stronger instrument is a direct structural comparison at the kernel:
`test/registered/attention/unittests/swa/test_triton.py`,
`test_chain_verify_on_a_window_layer_needs_no_custom_mask` and its cuda-graph
twin.

Each geometry is run in **both arms** — mask installed and mask dropped — with
the spec input carrying `custom_mask=None` for the dropped arm, which is the
exact kernel-level configuration Stage B hands the window layer. The expected
output is not relaxed for the dropped arm: the dense reference keeps the full
prefix + draft-causal + window mask in absolute positions. Geometries: prefix
below the window, prefix above it (paged-stage re-basing), a window shorter
than the draft block (ragged-stage window), and a GQA shape.

Mutation-checked, because a green test that cannot fail proves nothing:

| mutation | result |
|---|---|
| ragged stage: `elif IS_CAUSAL` -> `elif IS_CAUSAL and USE_CUSTOM_MASK` (remove the causal fallback) | 6 failures, **all in the dropped arm, none in the installed arm** — the coverage is drop-specific |
| ragged stage: neutralise the window mask | 4 failures, both arms, on the window-shorter-than-draft-block geometry |
| paged stage: drop the `cur_seq_len_prefix` re-basing | 12 failures, both arms |

The second mutation also found a hole in the first draft of the test: at
`W == 2` with a 3-token draft block the kernel's inclusive rule (`q <= k + W`)
still admits the whole block, so the ragged-stage window mask was vacuous. The
case now uses `W == 1`.

### 10.4 Result

**No defect.** The drop is correct on window layers for the chain verify, for
the reason in §10.2, and the seam is now pinned by GPU coverage in both arms
plus a CPU source-pin
(`test_swa_dcp_stage_b.py::test_the_verify_mask_drop_is_per_forward_not_per_layer`)
that makes the per-forward shape of the decision explicit — so the plausible
"tightening" of making the drop per-layer, which would re-install a mask the
window kernel is not indexed for, cannot happen silently.

Scope of this result: **topk == 1 only.** The tree verify remains refused at
boot and nothing here weakens that.
