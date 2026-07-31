# Task #108 — `--draft-kv-layout`: a DCP-token-sharded draft KV pool

Status: **v1 lands the geometry, NOT a usable flag.** `--draft-kv-layout dcp`
refuses at boot, by name, because the draft-EXTEND forward has no uneven-DCP
metadata split (§"The gap that stopped v1"). Everything up to that point is
implemented and boot-proven: the pool is token-sharded on-card, the draft
DECODE cuda graphs capture, and the owner-rule write runs for the draft.
Scope is the MTP/NEXTN chain; multi-layer EAGLE is out and refused by name.

## What the flag does

`--draft-kv-layout {replicated,dcp}`, default `replicated` (unchanged).

Under `dcp` the speculative DRAFT KV pool is token-sharded across the TP
ranks by the SAME machinery the TARGET pool already uses:

- the weighted DCP owner rule (`layers/dcp/owner.py`), slot `L` owned by the
  rank with `(L % cp_S) in [cp_lo, cp_hi)`, stored compactly at
  `(L // cp_S) * cp_ratio + (L % cp_S - cp_lo)`,
- replicated kv-heads in the pool (the write gathers this rank's uneven
  projection shard up to `get_total_num_kv_heads()`),
- the cross-rank LSE merge in the decode path.

There is no new kernel and no draft-specific branch past the geometry gate.
That reuse is the design: the draft runner simply stops being excluded from
the target's path.

## Why it is not a config flip

Before this task the draft pool was excluded from the DCP path at four
independent sites, each with its own copy of the condition
`self.is_draft_worker`. Turning the exclusion off means the draft's KV
indices must go through the owner rule — which is what the exclusion was
protecting them from.

The four sites are now ONE predicate,
`layers/dcp/owner.py: draft_pool_is_replicated(is_draft_worker, server_args)`:

| site | file | what it decides |
|---|---|---|
| pool head count | `model_runner_kv_cache_mixin._pool_kv_head_num` | per-rank shard vs full replicated kv-heads |
| pool row count | `model_runner_kv_cache_mixin._dcp_token_sharded_pool_rows` | global context C vs this rank's `C*ratio_r/S` |
| hybrid pool | `model_runner_kv_cache_mixin` (`_draft_non_dcp`) | same two, for the mamba/GDN full-attention sub-pool |
| attention path | `flashinfer_backend.__init__` (`self.uneven_dcp`) | owner-rule masked write, kv-head all-gather, LSE merge |

Pool geometry and backend geometry MUST agree. A pool shaped for a token
split the backend does not perform is #345's failure mode: the store kernel
writes at `loc * H * D` with `H` from the cache tensor, so a wrong row stride
makes every owned slot land at an address that drifts with the slot id —
silent, request-order-dependent corruption, not a crash. One predicate, read
by both, is the structural defence.

## Admitted shape, and every refusal

`ServerArgs._reject_unsupported_draft_kv_dcp()`. Admitted: weighted uneven
DCP + EAGLE (the alias NEXTN resolves to) + `topk == 1` + one draft KV layer.

| refused | why it is not merely unsupported |
|---|---|
| off the weighted lane (no `SGLANG_UNEVEN_DCP` / `_WEIGHTED` / non-uniform `--rank-tp-ratio` / `dcp_size != tp_size`) | there is no token weight vector to shard BY; the flag would be an expensive no-op |
| `--speculative-eagle-topk > 1` | branching draft KV chain the owner rule does not describe; #76 measured tree verify under uneven DCP emitting non-greedy, run-to-run-varying output at temperature 0 |
| EAGLE3 / STANDALONE / DFLASH / DSPARK / NGRAM / FROZEN_KV_MTP | draft KV access patterns outside the linear-append chain |
| `--enable-multi-layer-eagle` | one draft ModelRunner per chain position; needs per-layer owner-rule kernels (#76-scale) |
| `--speculative-cross-algorithm` | the active draft rung changes at runtime, so the boot-time chain guarantee does not hold |
| draft-solo placement | the draft lives on one rank; no peer group to shard across |
| `--enable-kv-session-offload` | `spec_in_tick_draft_pre` writes RAW GLOBAL allocator slot ids straight into `draft_full_pool.k_buffer/v_buffer` and `req_to_token`, bypassing `set_kv_buffer` and therefore the owner rule. Against a compact pool those ids address other tokens' rows — the #60 L3 zero-page class |
| a draft checkpoint with > 1 KV layer | tier 2, in the draft `ModelRunner` (`reject_multi_layer_draft_kv_dcp`): not knowable from the CLI |

### Where the gate runs, and why that is load-bearing

The gate does NOT live in `_handle_dcp_validation` next to its siblings. Its
inputs are all products of earlier resolution passes:

- `--speculative-algorithm`: `NEXTN -> EAGLE` (`handle_speculative_decoding`)
- `--speculative-eagle-topk`: defaulted, not necessarily user-supplied
- `--rank-tp-ratio`: `auto-performance` -> a concrete vector (`_handle_uneven_tp`)
- `dcp_size`: auto-set to `tp_size` under `SGLANG_UNEVEN_DCP` (`_handle_uneven_tp`)

Placed early it reads every one of them RAW. Measured on the rig: a correct
TP=3 `--speculative-algorithm NEXTN --rank-tp-ratio auto-performance
--draft-kv-layout dcp` boot was refused with
`rank_tp_ratio=auto-performance, dcp_size=1` — a false rejection of the exact
configuration the feature exists for. The gate now runs right after
`_handle_speculative_draft_placement`, the same slot the weightless × spec
gate uses for the same reason, and the placement is pinned by
`TestGateRunsAfterArgResolution`.

## Interaction with the KV budget accounting

`pool_configurator` inflates the target's per-token cell by
`1 + L_draft / L_target` — a charge levied per LOCAL token, i.e. per
`C * ratio_r / S` tokens. Its docstring already recorded that under split
placement + token-sharded DCP the draft pool is nonetheless sized to the
GLOBAL C, so the draft is UNDER-charged by `S / ratio_r`, and that fixing the
charge upward was declined because it would shrink every validated arm.

`--draft-kv-layout dcp` resolves the same mismatch from the other side: it
shrinks the POOL to `C * ratio_r / S`, which is exactly what the existing
charge assumes. No change to `pool_configurator` was needed — the accounting
was always written for the sharded shape.

## The gap that stopped v1: draft-EXTEND has no DCP metadata split

Measured on the rig, not predicted. With `--draft-kv-layout dcp` the boot gets
all the way through pool construction and draft-DECODE graph capture, then
dies in draft-EXTEND capture:

```
Capture draft decode CUDA graph end. elapsed=2.17 s      <- fine
Capture draft extend CUDA graph begin...
AttributeError: 'BatchPrefillWithRaggedKVCacheWrapper' object
  has no attribute '_cached_q_data_type'
  flashinfer_backend._forward_extend_dcp -> _dcp_ragged_current
  -> self.active_ragged_wrapper.forward_return_lse(...)
```

The ragged wrapper was never planned. `init_forward_metadata` sets
`use_ragged = True` in exactly two places: the non-spec extend branch
(`spec_info is None and uneven_dcp`) and the target-verify split
(`uneven_dcp and spec_input_type in _DCP_VERIFY_SPEC_INPUT_TYPES`, i.e.
`{EAGLE_VERIFY, DFLASH_VERIFY}`). A draft-extend carries an
`EAGLE_DRAFT_EXTEND` spec input, so it matches neither and falls into the
generic spec branch, which leaves `use_ragged` False — while
`_forward_extend_dcp` runs the current-chunk ragged stage unconditionally.

This is the identical failure shape the ngram/DFLASH guards already document,
which is what makes it recognisable rather than mysterious.

### The likely fix, for the next pass

Adding `EAGLE_DRAFT_EXTEND` to `_DCP_VERIFY_SPEC_INPUT_TYPES` is the WRONG
door: that branch is the verify split, which assumes a uniform
`draft_token_num` query block per request over a committed prefix, and
draft-extend's query block is the per-request accept length.

The RIGHT door is almost certainly the OTHER branch — the non-spec extend
DCP split (`spec_info is None and uneven_dcp`). That branch is already shaped
for exactly this: it builds owned-prefix kv indices with
`_build_dcp_weighted_kv_indices(..., prefix_lens, ...)` and derives
`qo_indptr` from `cumsum(seq_lens - prefix_lens)` — i.e. it handles
PER-REQUEST RAGGED query lengths natively, which is what varying accept
lengths need. Draft-extend is structurally a normal extend of the draft's own
chain: paged owned prefix + local ragged current chunk + LSE merge.

So the hypothesis to test first is: widen that branch's condition from
`spec_info is None` to also admit `EAGLE_DRAFT_EXTEND` on a DCP draft worker,
provided the draft-extend call site actually supplies `prefix_lens`
(`seq_lens - accept_len`) rather than only `spec_info`. Both the condition
and the `prefix_lens` availability need checking against
`eagle_draft_extend_cuda_graph_runner`, and the result needs a boot — a wrong
guess here is the silent right-token/wrong-slot class, not a crash, so it was
not worth guessing at inside this pass's remaining GPU window.

## Measured: what the byte win actually is (and is not)

Reference rig, Qwen3.6-27B-FP8, TP=3 uneven DCP, `--rank-tp-ratio
auto-performance` -> vector `[30, 17, 17]`, `S = 64`, NEXTN k=3 topk=1,
kv-cache fp8, full CUDA graphs. Draft = 1 layer; total kv-heads 8, split
`[4, 2, 2]` by the uneven-TP plan.

| | rank 0 (5090) | rank 1 (3080) | rank 2 (3080) |
|---|---|---|---|
| draft pool rows, `replicated` | 453 632 | 453 632 | 453 632 |
| draft pool rows, `dcp` | 212 670 | 120 513 | 120 513 |
| draft kv-heads, `replicated` | 4 | 2 | 2 |
| draft kv-heads, `dcp` | 8 | 8 | 8 |

The two factors move in opposite directions. Token sharding multiplies rows
by `ratio_r / S`; head replication multiplies the row width by
`total_heads / local_heads`. The net byte change per rank is their product:

```
rank 0:  (30/64) * (8/4) = 0.94     ->  -6%
rank 1:  (17/64) * (8/2) = 1.06     ->  +6%
rank 2:  (17/64) * (8/2) = 1.06     ->  +6%
```

**On this rig the draft pool bytes are a wash.** That is not a defect and it
is not hidden: it follows from the model being head-shardable at TP=3
(8 kv-heads over 3 ranks). State it as the rule:

> `dcp` reduces draft KV bytes exactly when `ratio_r / S` beats
> `local_heads / total_heads`. When the kv-heads are freely shardable across
> the TP group the two cancel and the layout is byte-neutral. The win appears
> when the heads CANNOT be sharded any further — the replicated-KV geometry
> `TP > num_kv_heads` (task #62), where every rank already holds all `H`
> heads. There `replicated` costs `C * H` per rank and `dcp` costs
> `(C * ratio_r / S) * H`, i.e. the full shard factor, up to `N`x on `N`
> ranks.

So the v1 deliverable is correctly described as **token-addressability of the
draft KV**, not as a byte saving on every configuration. That addressability
is what spec-in-PD-disaggregation and granular draft-VRAM control need; the
byte win is a configuration-dependent bonus.

## Verify path and collectives

No new collective and no new rank-local branch were introduced. The draft
runner takes the target's existing `uneven_dcp` path, whose collectives
(`cp_all_gather_heads_uneven` + `cp_lse_ag_out_ar_mha_uneven`) are already
rank-uniform per layer. The audit of `eagle_worker_v2`'s own collectives
(`_broadcast_draft_picks`, the solo broadcast pair, `eagle_sample`) found
every conditional gated on boot-fixed per-rank roles or on a
single-scheduler-cohort batch predicate — unchanged by this task, since the
new gate is a geometry decision taken once at construction, not per step.

The pool numbers in the table above are read from the `dcp` arm's boot log,
which reaches pool construction before the draft-extend failure — so they are
measured, not projected. The accept length and completions come from the
`replicated` arm (accept 2.42 over 38 verifies, two coherent completions,
temperature 0); the `dcp` arm never served a request.

## Honest remainder

0. **Draft-extend DCP metadata split** — the blocker above. Until it lands the
   flag refuses at boot and nothing downstream of it can be exercised.
1. Triton backend: the draft gate is wired in `flashinfer_backend` only.
   `triton_backend` has the owner rule (#173) and chain verify (#180) but its
   own `uneven_dcp` derivation still reads `is_draft_worker` directly, so
   `--draft-kv-layout dcp --attention-backend triton` is not covered.
2. Multi-layer EAGLE: refused, needs per-layer owner-rule kernels.
3. kv-session-offload: refused; the fix is to compact `spec_in_tick_draft_pre`'s
   indices through the owner rule and route them via `set_kv_buffer` so they
   inherit the `kv_store_bound` clamp.
4. Draft-solo: refused; a sharded draft on a single rank is not meaningful,
   but a solo draft over a SUBSET group would be.
5. Converting the freed bytes into context: on a configuration where `dcp`
   does shrink the draft pool, `pool_configurator` does not yet raise
   `max_total_num_tokens` to spend the difference.
6. The `_set_kv_buffer_impl` naive fallback (`k_cache[indices] = k`, taken
   when `can_use_store_cache` is false) carries no `kv_store_bound` clamp,
   relying on torch advanced indexing's own device assert. Pre-existing, same
   class as the registered #355 open items, not draft-specific.
