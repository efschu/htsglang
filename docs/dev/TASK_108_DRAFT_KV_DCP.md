# Task #108 — `--draft-kv-layout`: a DCP-token-sharded draft KV pool

Status: **slice 1 (geometry) + slice 2 (draft-extend split) built; GPU
validation pending.** Slice 1 unified the draft-exclusion predicate and
token-sharded the draft pool (boot-proven on-card), but refused the layout at
boot because the draft-EXTEND forward had no uneven-DCP metadata split. Slice 2
built that split, so the blanket refusal is gone and the covered shape is
admitted. Nothing on this path has run on a card yet -- see "What the GPU
ticket must validate". Scope is the MTP/NEXTN chain; multi-layer EAGLE is out
and refused by name.

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

## Slice 2: the draft-EXTEND DCP metadata split

### What slice 1 hit

With `--draft-kv-layout dcp` the boot got through pool construction and
draft-DECODE graph capture, then died in draft-EXTEND capture:

```
Capture draft decode CUDA graph end. elapsed=2.17 s      <- fine
Capture draft extend CUDA graph begin...
AttributeError: 'BatchPrefillWithRaggedKVCacheWrapper' object
  has no attribute '_cached_q_data_type'
```

The ragged wrapper was never planned. `init_forward_metadata` set
`use_ragged = True` in exactly two places -- the non-spec extend branch and the
target-verify split (`_DCP_VERIFY_SPEC_INPUT_TYPES`) -- and an
`EAGLE_DRAFT_EXTEND` matched neither, so it fell into the generic spec branch
while `_forward_extend_dcp` ran the ragged stage unconditionally.

### The split contract

Draft-extend is now decomposed exactly like target-verify, in its own branch in
`call_begin_forward`:

| stage | reads | heads | mask | collectives |
|---|---|---|---|---|
| paged | this rank's OWNED slots of the committed prefix | full replicated | non-causal | Q all-gather + LSE merge |
| ragged | the `num_tokens_per_req` tokens this step appends | LOCAL shard | causal | none |

combined by the cross-rank LSE merge. Same owner rule, same index builder
(`_build_dcp_weighted_kv_indices`), same kernels, same collectives as the
target side. No new kernel and no new collective kind.

**The one thing not shared with verify, and the whole correctness content of
the branch.** For verify, `paged_kernel_lens` IS the committed prefix -- the
draft tokens are not in `seq_lens`. For draft-extend, `seq_lens` ALREADY counts
the tokens this step appends: they are written into the pool by the owner-rule
masked write at the top of the same forward. So the paged read must cover
`seq_len - num_tokens_per_req` (`lockstep.draft_extend_prefix_lens`, clamped at
zero). Reading the full `seq_len` would let every query attend its OWN key
through the non-causal paged stage as well as through the causal ragged stage,
i.e. count it twice in the LSE merge -- a wrong answer, not a crash. That is
also why this is a separate branch and not a new member of
`_DCP_VERIFY_SPEC_INPUT_TYPES`.

The slice-1 note here previously guessed the fix was the non-spec extend branch
("it handles ragged per-request query lengths"). That guess was wrong in its
premise: draft-extend's qo layout is a CONSTANT `num_tokens_per_req` stride, not
the per-request accept length, precisely so it can be cuda-graph captured
(`EagleDraftExtendInput.generate_attn_arg_prefill`). The verify-shaped branch is
the right template; only the prefix derivation differs.

### Rank-uniformity (#94 family)

`has_prefix` decides whether the Q all-gather and the LSE merge are issued at
all, so it must be identical on every rank. A draft-extend batch carries NO
prefix vector -- `ForwardBatch` fills `extend_prefix_lens` from
`batch.prefix_lens`, which the draft-extend batch does not populate -- so a
length-based test answers False and skips the prefix stage entirely: the draft
would attend only this step's tokens and silently ignore the whole context, and
where the answer differed per rank the owner of a short prefix would sit alone
in an all-gather nobody joins.

Forward-mode first, unconditionally, as #180 established for verify. The rule is
one shared function, `lockstep.dcp_forces_prefix(is_target_verify,
is_draft_extend)`, read by the flashinfer forward and pinned on CPU. No new
collective was introduced; no new rank-local branch guards an existing one.

### CUDA-graph capture safety

The captured draft-extend graph runs its ragged stage INSIDE the capture, which
is the same hazard `_get_verify_ragged_cg_wrapper` exists for: the shared
`prefill_wrapper_ragged` is not in cuda-graph mode, so its `plan()` keeps a bare
reference to the caller's transient `torch.arange` indptr, the captured kernel
freezes that pointer, and the allocator later reuses the block.

Draft-extend therefore gets its own per-bucket graph-mode wrapper,
`_get_draft_extend_ragged_cg_wrapper`, from a SEPARATE dict. Separate on
purpose: draft-extend and verify are captured at the same `bs` values but are
different graphs with different qo strides, and flashinfer latches
`_max_total_num_rows` on a wrapper's first plan (the #274 round-7a failure one
level up). No mask buffers -- a draft-extend chain is plain causal and topk > 1
cannot reach this layout at all (boot-refused), and a wrapper created with a
mask buffer would silently run CUSTOM mode on every replay.

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

0. **Nothing on the dcp path has run on a card since slice 2.** The split is
   built and hermetically tested; correctness is argued, not measured. See
   the GPU ticket below.
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


## What the GPU ticket must validate

The split is desk work: its numeric behaviour has never executed. The ticket
needs one TP=3 uneven-DCP window (the runbook 4.1 recipe, NEXTN k=3 topk=1,
full CUDA graphs) and must answer these, in this order, because each one makes
the next meaningful:

1. **It boots at all.** `--draft-kv-layout dcp` must reach "fired up" --
   specifically past "Capture draft extend CUDA graph", the step that died in
   slice 1 with `AttributeError: '_cached_q_data_type'`. That single log line
   is the whole slice-2 claim.
2. **No hang under the forced prefix.** The prefix stage now runs on every
   draft-extend, so every rank issues the Q all-gather and the LSE merge on
   every such step. A hang here is the #94 family and would mean
   `dcp_forces_prefix` is not in fact rank-uniform in some batch shape (idle
   batches and bs=1 buckets are the ones to watch).
3. **The answers are coherent, and the accept length is not degraded.** Two
   completions at temperature 0, plus `meta_info.spec_accept_length` read
   against the `replicated` arm on the same prompt. A double-counted prefix
   (the failure mode the subtraction prevents) would most likely show as a
   DROP in accept length rather than as garbage, because the draft would still
   be fluent while diverging from what the target verifies -- so accept length
   is the sensitive instrument here, not eyeballing the text.
4. **bs > 1 replay.** The per-bucket ragged wrapper exists because bs=1 alone
   survived by allocator luck in the verify case. Exercise several decode
   bucket sizes so more than one captured draft-extend bucket is replayed.
5. **Per-rank draft-KV pool bytes, both layouts**, to confirm the slice-1
   measurement still holds after the split (it should be unchanged: the split
   touches attention metadata, not pool sizing).

Not in scope for the first window: a byte-win claim. On the reference rig the
kv-heads are freely shardable (8 over 3 ranks) so the layout is byte-neutral
there by the rule in the section above; demonstrating the full shard factor
needs a `TP > num_kv_heads` configuration.

---

# Slice 3 desk pass: the GPU verdict, and what the code actually says

The 2026-08-01 window (`/spinning/gpu-battery-results/2026-08-01_108-dcp-validation/`)
returned four green points and one split: the split works, output quality is
byte-clean, and **accept length regresses on the dcp arm** (alphabet
4.000 -> 3.368, squares 3.368 -> 3.048 at `draft_tokens=4`; no gap at
`draft_tokens=2`, where the arm sits at the 2.0 ceiling).

## The window's hypothesis is FALSIFIED

The window suspected the prefix subtraction under-read by
`(padded - accept_len)`, because the qo layout is a padded constant while
acceptance is variable. Reading the code settles it the other way.

`base_spec_worker.prepare_for_draft_extend`:

```python
batch.prefix_lens  = batch.seq_lens            # the committed history
batch.extend_lens  = [num_draft_tokens] * bs   # PADDED
extend_num_tokens  = bs * num_draft_tokens     # PADDED
# Forward sees post-write length (draft extend writes num_draft_tokens slots)
forward_batch.seq_lens = forward_batch.seq_lens + num_draft_tokens
```

and `eagle_worker_v2._draft_extend_for_decode` sets
`num_tokens_per_req=self.speculative_num_draft_tokens` (padded) while carrying
`num_accept_tokens=batch_result.accept_lens` (actual) alongside it.

So `seq_lens` at draft-extend is inflated by the **padded** count, not the
accepted one, and **the write is padded too** — the same constant on both
sides. Subtracting `num_tokens_per_req` therefore lands exactly on
`committed`. The subtraction is correct; the write and the read agree about
padding; there is no sibling defect on the write side. The pads are scratch
rows the next round overwrites, and because both stages agree on where the
padding starts, the paged and ragged ranges stay disjoint and complete.

**Graph capture and the real prefix coexist by construction**, which answers
the design question: the padded stride lives entirely in the RAGGED stage (the
current chunk), and the paged read stops at `committed`. No mask is needed to
reconcile them because no padded row is ever inside the paged range. If the
append count ever became genuinely variable per request, that would no longer
hold and the paged stage would need a per-request length — but today it does
not.

## Consequence: the regression has another cause, not yet identified

Ruled out by this pass: the subtraction, the write/read padding agreement, and
an arm-level confound — arms C and D resolved to the **identical** effective
configuration (vector `[30,17,17]`, ratio `[29607,17780,17780]`,
`max_total_num_tokens=453632`), so the flag is the only difference.

Still open, in the order worth checking:

1. The draft's **cross-rank LSE merge and owned-slot indices**. The draft pool
   is compacted by the owner rule while `req_to_token` (shared with the target)
   holds global allocator ids; the compaction must agree between the draft
   pool's row space and the index builder. A subtly wrong merge degrades the
   draft's predictions without touching the target verify — which is exactly
   the observed shape (accept down, quality perfect).
2. The **ragged stage's head geometry** for the draft. The current chunk is
   attended on LOCAL head-sharded q/kv while the paged prefix uses gathered
   full heads; `_replicated_kv_ragged_reindex` corrects the GQA grouping only
   when `dcp_kv_replicated_heads` is set, which it is not at 8 kv heads over 3
   ranks.
3. Plain numeric drift from reading a sharded prefix through an LSE merge
   instead of one local read. Benign if so, but it must be shown, not assumed.

Note the padding-width scaling that looked like evidence for the falsified
hypothesis is **also** consistent with (1)-(3): at `draft_tokens=2` acceptance
was at its ceiling, so there was no headroom in which a degraded draft could
show up. The discriminator localized the effect to a regime, not to a cause.

## Test-design fix

`test/registered/unit/distributed/test_draft_extend_prefix_contract.py` pins
the CONTRACT rather than the arithmetic:

* C1 prefix == committed, C2 paged and ragged disjoint and complete, C3 no
  committed token outside the paged range — over a grid of padded widths
  (1,2,3,4,8) crossed with committed lengths.
* The padded constant is varied **independently** of the accept count, so the
  k=2-vs-k=4 discriminator is now a CPU case.
* The falsified hypothesis is pinned as a counterfactual: subtracting an
  accept-like count reads PAST the commit.

Falsifier-checked: a wrong subtrahend reds 45 assertions across the grid. The
slice-2 test, which asserted `seq_lens - k` directly, caught none of them —
it pinned the arithmetic chosen rather than the property it had to satisfy.

## Re-run scope (next window, ~10 min)

The prepared runsheet unchanged, plus the k=2/4 pair as a permanent arm. But
note that **the fix this pass was expected to produce does not exist**, because
the defect it assumed is not there. The re-run should therefore be deferred
until one of (1)-(3) is either confirmed at the desk or turned into an on-card
discriminator — re-measuring the same regression without a changed hypothesis
would spend a window to reproduce a number already in hand.

## Cause pass: verdicts

**Read the discriminator as a REGIME, not a cause.** This is the canonical
framing; the window over-read it and the next reader should not. The k=2/k=4
pair showed *where the effect has headroom to become visible* — at
`draft_tokens=2` acceptance was already at its 2.0 ceiling, so a slightly worse
draft had nowhere to show. It localized a regime. It said nothing about
mechanism, and every hypothesis below is compatible with it.

### Cause (1) — owner-rule compaction vs the draft pool's row space: CLEAN

`build_dcp_weighted_kv_indices` (`layers/dcp/owner.py:448`) reads the global
allocator ids out of `req_to_token` and then compacts every one of them through
`dcp_weighted_read_slots` to `(loc // cp_S) * cp_ratio + (loc % cp_S - cp_lo)`
— "the exact inverse of the `_dcp_masked_write` packing, so a token is read
from the slot it was written to". Raw global ids never reach the pool.

Draft and target **share** `req_to_token` and the allocator (handed in by
`alloc_memory_pool`), so a token at global slot `L` is written to
`draft_pool[compact(L)]` and read from `compact(L)` under the same
`(cp_S, cp_lo, cp_hi, cp_ratio)`. Slice 1 sized the draft pool with the same
`dcp_compact_pool_rows(C, S, ratio_r)`, so every compacted row is in range.

The draft **decode** path was checked too, since slice 2 only built extend: its
DCP branch is gated on `spec_info.kv_indptr is None`, and nothing in
`speculative/` ever assigns that field for the EAGLE draft inputs — so decode
takes the compaction branch as well. No bypass.

### Cause (2) — ragged head geometry: NOT APPLICABLE here

`_replicated_kv_ragged_reindex` fires only under `dcp_kv_replicated_heads =
attn_kv_replicated(attn_tp_size, total_kv)`, i.e. `TP > num_kv_heads`. This rig
runs 3 ranks over 8 kv heads, so it is False and the reindex is correctly
skipped: the aligned split gives whole GQA groups per rank
(`gqa_local == gqa_global`), which is the condition the reindex exists to
repair when it does not hold. Its docstring symptom ("corrupting short-prompt /
first-chunk generation while the gathered-q paged prefix/decode path stayed
correct") is a near-match in shape, which is why it was checked — but its
trigger is absent. **It becomes live on `TP > num_kv_heads` hardware, which is
exactly the hardware where this layout is worth enabling**, so it must be
re-checked there rather than considered closed.

### Cause (3) — LSE-merge numeric drift: the live hypothesis, and benign-shaped

The decisive evidence was already in the window's artifacts and was not read at
the time. **Both arms emit byte-identical output token ids** — 64/64 on both
prompts, no divergence. The whole difference is the verify count:

| prompt | C verifies | D verifies | accept C | accept D |
|---|---|---|---|---|
| alphabet | 16 | 19 | 4.000 | 3.368 |
| squares | 19 | 21 | 3.368 | 3.048 |

An indexing or compaction defect makes the draft read *wrong rows*; that
degrades its predictions erratically and content-dependently, and it would not
leave the accepted trajectory untouched at 64/64. A uniform handful of extra
rounds with identical output is instead the signature of the draft's **logits
differing in their last bits**: the sharded prefix is read as per-rank partials
and recombined by LSE, which reassociates the reduction relative to one local
read. A draft proposal at a near tie flips, the target rejects it, and the
round costs one more verify. That is the #274/#360 reassociation story one
level down — in the DRAFT rather than in the verify.

## What this means for the feature

Correct, and on THIS rig not worth enabling:

* VRAM: byte-neutral here (kv-heads freely shardable at 8 over 3 ranks; the
  token-shard factor and the head-replication factor cancel — see the byte-win
  rule above).
* Accept: **-10 to -16 %**, at identical output.

So the applicability rule is sharper than "it works": `--draft-kv-layout dcp`
pays only where the heads cannot be sharded further (`TP > num_kv_heads`),
where the VRAM win is the full shard factor. This rig is the configuration
where it costs and returns nothing. That is a property of the hardware, not a
defect, and it belongs in the flag's help text.

## Re-run proposal — one falsifier per hypothesis

Deferred until a hypothesis is worth falsifying; this is what would make it so.
Two boots, ~10 min, only if the answer is wanted before the `TP > num_kv_heads`
work.

**H3 (drift, live).** Falsifier: **context-length sensitivity.** Drift flips
near-tie draft positions at a roughly constant per-position rate, so the
RELATIVE accept gap should stay about the same as context grows. A latent
indexing defect that only bites when the prefix spans many owner blocks would
make the gap GROW with context. Run both arms at ~4k prompt context and compare
the relative gap against the 64-token measurement already in hand. Gap constant
-> H3 confirmed, feature correct, close it. Gap grows -> reopen (1) with a
concrete regime to bisect.

**H2 (head geometry).** Not falsifiable on this rig — its trigger is absent.
Falsifier belongs to the `TP > num_kv_heads` window: run the same two arms on a
model with fewer kv heads than ranks and check that `dcp_kv_replicated_heads`
is True and the reindex path is exercised.

**H1 (compaction).** Reads clean; no on-card falsifier proposed. It would be
re-opened only by H3's gap-grows outcome.

Recommended order: do the `TP > num_kv_heads` window FIRST (it tests the
feature where it is supposed to pay, and it exercises H2), and treat H3's
context sweep as optional confirmation rather than a blocker.

---

# TP > num_kv_heads: vehicle inventory and window proposal

## Why this window, and why it is not a re-run

The validation arc's durable outcome is an applicability rule, not a bug fix:
`dcp` pays only where the heads cannot be sharded further. Every number so far
comes from a rig where they can, so the feature has never been measured where
it is supposed to work. This window measures that, and it is also the only
place H2's reindex is reachable.

## Vehicle candidates (scan of the model cache)

The trigger is `attn_kv_replicated(attn_tp_size, total_kv)`, i.e. strictly
`TP > num_key_value_heads`. **Two vehicles reach it at NATIVE TP=3 — no #82
co-location emulation needed**, which removes a whole axis from the window.

| model | kv | q | size | format | MTP head | TP=3 trigger |
|---|---|---|---|---|---|---|
| **Qwen3.5-2B** | **2** | 8 | 3.5 G | bf16 dense | yes (15 tensors) | **YES** (3 > 2) |
| **Qwen3.6-35B-A3B-FP8** | **2** | 16 | 31 G | FP8 MoE | yes (1560) | **YES** (3 > 2) |
| Qwen3.5-35B-A3B-GPTQ-Int4 | 2 | 16 | 20 G | GPTQ MoE | yes (785) | YES |
| Qwen3.6-35B-A3B-AWQ-4bit | 2 | 16 | 22 G | AWQ MoE | yes (2321) | YES |
| Qwen3.5-122B-A10B-GPTQ-Int4 | 2 | 32 | — | GPTQ MoE | — | YES, but needs expert offload (#77) — too heavy for this question |
| Qwen3.6-27B family | 4 | 24 | 20-31 G | several | yes | no at TP=3; would need TP=5 co-location |
| Qwen3.5-4B | 4 | 16 | 7.1 G | bf16 dense | yes (15) | no at TP=3 |

Recommended pair: **Qwen3.5-2B as the mechanism vehicle** (3.5 G, boots in
seconds, leaves the whole rig free for the measurement) and
**Qwen3.6-35B-A3B-FP8 as the realistic one** (31 G over 72 G aggregate, the
configuration a deployment would actually run). The 27B family is deliberately
NOT used: reaching the trigger there needs TP=5 co-location, which adds the
#82 axis to a window that already has one lever.

Note both recommended vehicles are MoE-or-dense Qwen3.5/3.6, so the
uneven-weighted-DCP path and the NEXTN chain are the same machinery already
validated — only the head geometry changes.

## The prediction to test

Under `attn_kv_replicated` every rank holds ALL `total_kv` heads because they
cannot be split. So:

* `replicated` draft pool = `C` rows x `total_kv` heads
* `dcp` draft pool = `C * ratio_r / S` rows x `total_kv` heads

The head factor is 1 on both sides — it no longer cancels the token shard — so
the saving is the **full shard factor**. For a `[30,17,17]` vector that is
**47 % / 27 % / 27 %** of the replicated draft pool, per rank.

## Window proposal — 2 questions, 4 boots, ~25 min

Minimal by construction: only the two questions the applicability rule turns on.

**Q1 — does the VRAM win materialize at the predicted factor?**
Both layouts, both vehicles, read the draft-pool rows and K/V sizes out of the
`KV Cache is allocated` lines exactly as the previous window did. GREEN iff the
`dcp` draft pool is the replicated one scaled by `ratio_r / S` per rank, within
the ceil-to-a-whole-owner-block slack of `dcp_compact_pool_rows`.

**Q2 — does accept hold, and does H2 behave?**
The same instruments, unchanged: `stock_spec_control.py` for accept length and
the in-boot A-vs-A, `chain_quality_gate.py` for the verdict. Two sub-checks
that only exist on this hardware:
  * `dcp_kv_replicated_heads` must be **True** — assert it from the boot log /
    server info, because if the trigger is not actually active the window
    measured nothing.
  * `_replicated_kv_ragged_reindex` is now on the live path. Its failure shape
    is documented as "corrupting short-prompt / first-chunk generation while
    the gathered-q paged prefix/decode path stayed correct" — so the probe set
    must include a SHORT prompt, and the gate is the instrument.

Arms: `{2B, 35B-A3B-FP8} x {replicated, dcp}` = 4 boots. 2B boots are seconds;
budget ~25 min total, standard reserve, no bar1, local CT999. Abort rules and
process discipline exactly as the prepared runsheet.

**Explicitly out of scope:** the H3 context sweep (optional confirmation of a
benign explanation, and not a blocker), throughput comparison, and TP=5
co-location.

## Expected outcomes and what each would mean

* Win materializes AND accept holds -> the applicability rule is confirmed;
  `dcp` is a real feature on this class of hardware and the flag text stands.
* Win materializes AND accept drops similarly (~10-16 %) -> the drift cost is
  hardware-independent; the rule becomes "trade N % accept for the shard
  factor", which is still a good trade where VRAM is the binding constraint.
* Win does NOT materialize -> the sizing path has a second head-geometry
  assumption that only shows under replication; reopen at the pool sizing, not
  at the attention split.
* Gate goes RED on the short prompt -> H2 is live and the reindex condition is
  wrong for the draft. This is the outcome that would make H2 a real defect
  rather than an inapplicable one, and it is the reason a short prompt is
  mandatory in the probe set.
