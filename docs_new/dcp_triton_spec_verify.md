# #180 — the M4 verify split, twinned into the Triton backend

Companion to `dcp_triton_weighted_owner_rule.md` (#173), which brought the weighted
uneven-DCP owner rule to Triton but left **speculative decoding refused** with this
reason:

> speculative decoding is on, and the Triton target-verify metadata builds FULL
> un-sharded kv indices (global allocator slot ids) which index a compact
> token-sharded pool out of bounds; flashinfer has a dedicated verify split (M4),
> this backend has none

This document is the design for that twin. Scope is **chain spec only** — EAGLE /
NEXTN / MTP at `--speculative-eagle-topk 1`. Tree verify (`topk > 1`, and the
`--speculative-dflash-tree-verify` door) stays refused; it is measured-wrong under
uneven DCP and perf-negative on this class of interconnect (#76).

Nordstern relevance: the draft has to be able to sit on ranks that have no
flashinfer (sm75 Turing, gfx900 Vega). Until the *target*'s verify runs on Triton
under DCP, that whole arm is unreachable.

---

## 1. What flashinfer actually does (the reference)

Read out of `layers/attention/flashinfer_backend.py` at `763ddef0d2`.

### 1.1 The split

The verify step appends `d = num_draft_tokens` query positions per request. The
committed prefix is `seq_lens` — **`eagle_prepare_for_verify` does not advance
`seq_lens`**, it only allocates `out_cache_loc` for the draft tokens. flashinfer
splits the verify attention along exactly that seam:

| stage | rows attended | source | owner-sharded? | mask |
|---|---|---|---|---|
| paged prefix | `[0, seq_len)` | the KV pool, through the owner rule | **yes** | none (non-causal) |
| ragged current | the `d` draft tokens | this rank's freshly projected `k`/`v` | **no** — locally complete | causal |

`call_begin_forward`'s verify branch (`flashinfer_backend.py:6725-6836`) calls the
*same* `build_dcp_weighted_kv_indices` as decode and extend, over
`paged_kernel_lens == seq_lens`. `qo_indptr` is the uniform block
`arange(0, (bs+1)*d, step=d)` (`:6810`). `custom_mask = None` (`:6820`).

The load-bearing property is that **the draft block is not read from the pool at
all during verify**. It is written there (owner-sharded, for later decodes) but
attended out of the local `k`/`v` tensors, which every rank holds in full for its
own head shard. That is why the split needs no cross-rank coordination on the
draft tokens: they behave exactly like a chunked-prefill current chunk.

### 1.2 The merge

Per attention layer, in this order (invariant issue order on the communicator):

```
A  fused all_gather of cat(k, v)        -- write side; NO collective at all when
                                           kv < tp (replicated kv heads, the
                                           27B/35B case)
B  all_gather of q heads (uneven counts)
C  all_gather of LSE
D  all_reduce of the output
```

then two *local* merges that must not be confused:

* cross-rank prefix combine — `cp_lse_ag_out_ar_mha_uneven` (`layers/dcp/comm.py:209`),
  a `logsumexp` over the gathered LSE stack. C and D live inside it.
* current-chunk ⊕ prefix combine — `_dcp_extend_final_merge`
  (`flashinfer_backend.py:5871`), a `torch.logaddexp`. Its docstring warns that
  flashinfer's `merge_state` uses a different internal convention and must **not**
  be used across these two sources.

Shapes: with `T = bs*d`, every verify collective carries `T` rows where decode
carries `bs`. Nothing else differs.

### 1.3 How the accepts stay group-uniform

This is the correctness question, and the answer is **not** "the logits come out
identical". The chain is:

1. `cp_lse_ag_out_ar_mha_uneven` ends in `all_reduce(out)` — every rank holds the
   same attention output *up to floating point*.
2. Each decoder layer's RowParallel `o_proj` all-reduces — hidden states replicated.
3. `LogitsProcessor` all-gathers the vocab shards
   (`layers/logits_processor.py:992-1000`, uneven variant at `:1086`) — every rank
   holds the full `[bs*d, vocab]` tensor.
4. `eagle_sample` (`speculative/eagle_utils.py:798`) computes argmax + the accept
   kernel **redundantly on every rank**.
5. `speculative/eagle_utils.py:983-996` then **overwrites** the result with an
   explicit pynccl broadcast from rank 0:

   ```python
   capture_safe_tp_broadcast(
       tp_group, (predict, accept_index, num_correct_drafts), src=0
   )
   ```

So identity is *approximate* by construction and *exact* by collective. The
comment there names the reason directly: on heterogeneous GPUs per-rank argmax
flips on near-ties are common, and a single divergent accept desynchronizes
KV/recurrent state for the rest of the sequence.

**Consequence for this port: nothing in the Triton twin has to reproduce the
accept sync.** It lives in the speculative worker, above the attention backend,
and is backend-agnostic. What the Triton twin owes is only that its attention
output be *the same function of the same inputs* as flashinfer's — the broadcast
then makes the rest exact. That also means a Triton-vs-flashinfer accept-rate
comparison is a real signal: it is measuring the attention math, not the sync.

### 1.4 The draft worker is exempt, deliberately

The draft/NEXTN worker's KV pool is **not** token-sharded — it keeps the full token
context with local heads. Running the owner rule over it would compact indices that
were never compacted. Three coordinated places:

* `flashinfer_backend.py:606-608` — `uneven_dcp = ... and not is_draft_worker`.
* `flashinfer_backend.py:514` — `reject_silently_inert_dcp` short-circuits on
  `is_draft_worker` (#169.2: flashinfer refuses a `--dcp-size` it would silently
  ignore, *except* for the draft, whose `dcp_size > 1` is an artifact of the shared
  parallel context).
* `model_runner_kv_cache_mixin.py:2370-2400` — the draft pool skips both the head
  replication and the ratio-proportional token capacity.

Triton already has its twin from #173 (`triton_backend.py:586-600`), and goes one
step further: it sets `dcp_size = 1` outright, because every DCP branch in that
file keys on `dcp_size > 1`. **No change needed here for #180** — the exemption is
already installed and is what keeps the draft's multi-step backend untouched.

---

## 2. The Triton twin: construct-by-construct mapping

| flashinfer construct | Triton twin | status before #180 |
| --- | --- | --- |
| verify branch of `call_begin_forward` (`:6725`) | `init_forward_metadata`'s `is_target_verify` arm (`triton_backend.py:1357`) | **DCP-blind** — the only forward mode in the function with no `if self.dcp_size > 1` |
| `build_dcp_weighted_kv_indices(paged_kernel_lens=seq_lens)` (`:6774`) | `self._dcp_kv_indices(req_pool_indices, seq_lens, self.kv_indptr)` (`:831`) | exists, unused by verify |
| `qo_indptr = arange(0, (bs+1)*d, d)` (`:6810`) | already identical (`:1359-1365`) | ok |
| `custom_mask = None` (`:6820`) | `custom_mask = spec_info.custom_mask` (`:1398`) | must be dropped on the DCP lane |
| `_forward_extend_dcp(force_prefix=True)` (`:1999`) | `_forward_extend_dcp` (`:2151`) + `_dcp_batch_has_prefix` (`:2117`) | gate falls through to a **rank-local** test for verify |
| `_dcp_ragged_current(causal=True)` (`:5834`) | the `if k.numel() > 0` current-chunk stage (`:2224-2243`), `skip_prefix=True` | ok as-is |
| paged prefix, `causal=False` (`:5641`) | the second `extend_attention_fwd`, `skip_extend=True`, `is_causal=False` (`:2296`) | ok as-is |
| `_dcp_extend_final_merge` `logaddexp` (`:5871`) | `torch.logaddexp` at `:2308-2314` | ok as-is |
| `_dcp_masked_write` (`:2212`) | `_set_kv_buffer` (`:1890`) — keys on `out_cache_loc`, not on forward mode | **already correct for verify** |
| per-bucket verify ragged CG wrappers (`:1836`) | *not needed* — `extend_attention_fwd` is stateless | n/a |
| tree mask `_build_dcp_ragged_tree_mask` (`:5904`, dormant) | *not ported* | stays refused |
| `_DCP_VERIFY_SPEC_INPUT_TYPES` (`:79`) | same frozenset, pinned EQUAL to flashinfer's by test | added in #180.3 |

### 2.0 Which verify layouts the split is valid for

Four `SpecInputType`s end in `_VERIFY` (`EAGLE`, `FROZEN_KV_MTP`, `DFLASH`,
`NGRAM`), and flashinfer's M4 admits exactly two: `EAGLE_VERIFY` and
`DFLASH_VERIFY`. Both present a uniform `draft_token_num` query block per request
and a **linear** draft chain, which is what the split assumes — the draft block is
attended ragged on local heads with a plain causal mask while the committed prefix
is read owner-sharded.

Keying the Triton branch on `forward_mode.is_target_verify()` alone would admit the
other two silently. So the same frozenset gates it, and a non-member **raises**
rather than falling through to the non-DCP branch — falling through would rebuild
the FULL un-sharded indices against a compact pool, which is the out-of-bounds read
this whole change removes. The test pins Triton's set *equal to* flashinfer's
rather than restating its members, so the two cannot drift.

**No new kernel, no new collective kind, no second copy of the owner rule.** The
verify read is `build_dcp_weighted_kv_indices` over a different length vector; the
forward is `_forward_extend_dcp` with the prefix gate forced.

### 2.1 Why dropping `custom_mask` at topk==1 is sound

At `topk == 1` the EAGLE draft is a chain, so its `d x d` draft→draft mask block is
exactly lower-triangular — the causal mask. The prefix columns of the mask are never
consumed anyway: `extend_attention_fwd` defaults `SKIP_PREFIX_CUSTOM_MASK=True`
(`kernels/ops/attention/extend_attention.py:640,677`) and stage 1 only reads the
mask when that flag is off.

This equivalence is already asserted twice in-tree, so #180 is not inventing it:

* `kernels/ops/attention/verify_splitkv.py:624-633` — "The kernel always computes
  pure-causal attention, which equals the tree mask ONLY at speculative topk == 1.
  The caller therefore MUST gate enablement on topk == 1."
* `flashinfer_backend.py:6738-6748` / `:6818-6820` — flashinfer's verify branch
  sets `custom_mask = None` for the same reason.

Dropping it is also *necessary*, not merely convenient: the mask's row stride is
`(prefix_len + extend_len)` with stage 2 offsetting by the **global** prefix length
(`extend_attention.py:507`). Under an owner-sharded prefix that stride no longer
describes the rows the kernel walks, so a retained mask would be indexed wrong.
`_forward_extend_dcp` already refuses a non-None `custom_mask` outright; #180 keeps
that refusal as the backstop and makes the metadata not produce one.

### 2.2 The prefix gate — the D5 trap, second sighting

`_dcp_batch_has_prefix` (`triton_backend.py:2117`) decides whether the q-head
all-gather (B) and the LSE merge (C+D) run. It is documented as group-uniform, and
it is — *for extend*. It reads `extend_prefix_lens_cpu`, then `extend_prefix_lens`,
then falls back to `kv_indices.numel() > 0`.

A **target-verify batch carries no prefix lengths**: `forward_batch_info.py:663-668`
fills them from `batch.prefix_lens`, which a verify batch built out of a decode
batch does not set. So verify lands on the fallback — and that fallback is
`kv_indices.numel() > 0`, a **rank-local** quantity under the owner rule. A short
committed prefix over an uneven token vector is routinely owned entirely by the
high-ratio rank; the others would return early and the owner would sit alone in an
all_gather. That is [[rank-lokaler-test-vor-kollektiv]] verbatim — the same defect
D5 fixed for extend, arriving through a different door.

flashinfer avoids it structurally with `force_prefix = forward_mode.is_target_verify()`
(`:1999`, consumed at `:5580-5586`), hoisted above the kernels precisely so the
overlap lane knows up front whether B will be issued. The Triton twin does the same:
`forward_mode` is replicated, so keying on it is uniform by construction. A verify
batch *always* has a committed prefix (`seq_lens >= 1`), and even if it did not,
forcing the branch costs two collectives over an empty read, which the LSE merge
already handles as `out=0 / lse=-inf`.

Note the existing wiring test
`test_the_prefix_gate_falls_back_to_the_local_test_without_lengths` asserts the old
premise in its docstring ("Target-verify style batches carry no extend_prefix_lens;
the old local test is then equivalent"). That premise becomes false the moment
verify plans a paged read, and the test is updated accordingly.

### 2.3 CUDA graph

`extend_attention_fwd` is stateless, so unlike flashinfer there is no wrapper plan
to bucket. The whole graph story is routing the two index buffers through the DCP
builder:

* `_update_target_verify_buffers` (`:1007`) must pass `self.cuda_graph_kv_indices`
  into `_dcp_kv_indices`, exactly as `_update_decode_kv_buffers` (`:970-980`) does.
  `_dcp_weighted_kv_indices` already implements the **D3 buffer contract** — copy
  into the caller's buffer, return *that object*, raise on overflow — so this is a
  call-site change, not new machinery.
* `_build_cuda_graph_forward_metadata`'s verify arm (`:1639`) must report
  `custom_mask=None` / `mask_indptr=None` on the DCP lane, matching eager.
* `get_verify_buffers_to_fill_after_draft` keeps returning
  `[self.cuda_graph_custom_mask, None]`. The spec worker keeps filling a buffer the
  captured graph no longer reads. Harmless, and changing the return shape would
  reach into the worker for no gain.

### 2.4 The guard, narrowed

`reject_unsupported_dcp_geometry` (`triton_backend.py:279`) branch (1) currently
refuses on `speculative`. It is narrowed to refuse on **tree** verify only. The
three siblings in that same branch — `weightless_kv`, `mla`, `sliding_window` —
are untouched.

`server_args.py::_handle_dcp_validation` needs **no change**: `:4661-4681` already
refuses tree verify on the weighted lane via `tree_verify_activation_reason()`, and
`:4682-4692` already exempts `uneven_weighted_dcp` from the general spec refusal —
that is how flashinfer's M4 got through. Triton was the only thing in the way.

To keep the "two doors" problem from reopening (topk>1 *and*
`--speculative-dflash-tree-verify`), the backend does not re-derive the tree
predicate; it is expressed once, in `layers/dcp/owner.py::dcp_verify_mask_mode`,
and both the guard call site and any future reader go through it.

---

## 3. Collective audit for the new path

Per [[rank-lokaler-test-vor-kollektiv]], every entry condition on the verify path,
after the change:

| # | site | collective | guard | verdict |
|---|---|---|---|---|
| A | `_set_kv_buffer` → `_dcp_write_gather` (`:1846`) | fused `cat(k,v)` all_gather | `dcp_size > 1` and `uneven_dcp`; inner early return on `dcp_kv_replicated_heads` (**no collective issued** for kv < tp) | group-uniform — all three are boot-time geometry constants |
| — | current-chunk `extend_attention_fwd` (`:2226`) | *none* | `k.numel() > 0` | rank-local, but guards **no collective**; the draft token count is replicated anyway |
| B | `_dcp_gather_q_heads` (`:2252`) | q-head all_gather | `_dcp_batch_has_prefix` | group-uniform **after this change** — `forward_mode.is_target_verify()` short-circuits to True before any rank-local source is consulted |
| C+D | `_dcp_merge_q_heads` (`:2305`) | LSE all_gather + output all_reduce | same gate as B | group-uniform, same condition |
| E | `eagle_utils.py:994` | accept broadcast from rank 0 | `tp_group.world_size > 1`, plus an early return on `forward_mode.is_idle()` | group-uniform; unchanged by #180, and it is what makes the accepts exact |

Nothing on the path is guarded by an owned-slot count, a per-rank kv length, or a
per-rank head count.

---

## 4. What is pure and CPU-testable

Three functions in `layers/dcp/owner.py`, in the established style (pure, no device,
no group), so the decisions that make verify correct are pinned without a GPU:

* `dcp_verify_paged_lens(seq_lens, num_draft_tokens)` — the paged read length vector
  for verify. Returns `seq_lens`, and the *point* is that it is not
  `seq_lens + num_draft_tokens`: reading the draft block from the pool as well would
  double-count it against the ragged stage and, worse, dereference slots this rank
  may not own. This is the single most likely way to get the port wrong, because the
  non-DCP verify branch it replaces reads `seq_lens` for a *different* reason (the
  draft K/V are handed in as tensors) and looks the same.
* `dcp_verify_window_is_disjoint(seq_lens, num_draft_tokens, qo_stride)` — the
  invariant: paged rows + ragged rows == the full attended context, exactly once.
* `dcp_verify_mask_mode(topk, dflash_tree_verify)` — `"causal"` or `"tree"`, the one
  place the topk==1 equivalence of §2.1 is expressed.

The verify **index build itself needs no new math**: it is
`build_dcp_weighted_kv_indices` over `seq_lens`, the same expression flashinfer
calls. The parity test therefore checks that the Triton verify call site produces
byte-identical slices/lengths/slots to the flashinfer verify expression on the same
inputs, rather than re-deriving a second rule.

Edge cases the tests must carry:

1. **1-token prefix** (the D5 class): `seq_lens = [1]` over `dcp_size = 3` with an
   uneven vector — one rank owns the single slot, the others own nothing, and every
   rank must still enter B/C/D.
2. **Draft length exceeding a rank's owned tokens**: `d = 4` with an owned prefix of
   1 or 0 rows. The ragged stage is unaffected (it is local); the paged stage runs
   over an empty index and contributes `lse = -inf`.
3. **A rank owning nothing in the verify window**: the index tensor must still be
   dereferenceable (the `max(n, 1)` dummy row from #173), the indptr all-zero.
4. **Uniform token vector** degenerating to the even rule — byte-identical.

---

## 5. GPU validation recipe

Ordered cheapest-first. `V1-V3` are rank-local and need no model.

* **V1** — index parity: on device, the verify build over `seq_lens` vs the numpy
  reference and vs the flashinfer expression, over the same token plans the #173
  GPU test uses.
* **V2** — guard: `topk=1` boots on the uneven Triton lane; `topk=2` still raises,
  naming the tree; `weightless / mla / sliding_window` still raise.
* **V3** — graph buffer: `_update_target_verify_buffers` fills
  `cuda_graph_kv_indices` in place and the metadata's `kv_indices` **is** that
  object; `custom_mask is None` on the lane.
* **V4 — THE PARITY ANCHOR.** 27B, uneven DCP, `TP=3`, `--dcp-size 3`, **with MTP**,
  greedy (`temperature 0`), the same prompt set, CUDA graphs **on** and spec on (per
  [[full-perf-testen]] — eager hides graph-replay bugs):
  - arm F: `--attention-backend flashinfer`
  - arm T: `--attention-backend triton`
  - **Gate 1 (bytes/token ids):** arm T's output token id sequence must match arm F's.
    Greedy + the rank-0 accept broadcast make this a hard equality, not a similarity.
  - **Gate 2 (accepts):** `meta_info.spec_accept_length` (**not** `spec_ema_accept_len`
    — see [[spec-acceptance-messfalle]]) must land in the same band as arm F's over
    the same prompts. A verify that reads the wrong slots still *runs*; it shows up
    as a collapsed accept rate long before it shows up as a crash. Accept rate is the
    correctness instrument here, which is why V4 cannot be replaced by a smoke test.
  - **Gate 3 (self-determinism):** arm T 3x byte-identical on the same prompt.
* **V5** — the empty-shard case, provoked: a 1-token prompt (D5 class) through arm T.
  Must answer, not hang. Run it *before* V4's long prompts — it is the cheapest way
  to catch a reintroduced rank-local gate, and a hang there is diagnosable while a
  hang inside a 262k-context run is not.
* **V6** — `dcp-size 1` baseline on Triton with spec, to confirm the default path is
  byte-unchanged.
* **V7** — sm75 / gfx900: the actual Nordstern reason for the port.

Order: V6 (baseline first) → V1 → V2 → V3 → V5 → V4 → V7.

---

## 6. What #180 deliberately does not do

* **Tree verify.** Still refused, now by name. #76 measured it non-deterministic
  *within a boot* under uneven DCP and net-negative on tok/s. The dormant flashinfer
  tree machinery is not ported.
* **The accept sync.** It is backend-agnostic and already correct
  (`eagle_utils.py:983-996`). Reproducing it in the backend would be a second copy of
  a rule that has exactly one.
* **The draft worker.** Already exempt from #173; #180 changes nothing there.
* **`num_kv_splits` for verify.** Verify runs `extend_attention_fwd`, which does not
  consume it (`num_kv_splits = None`, `:1406`). D4 stays a decode-only concern.
* **Sliding window under verify.** Still refused by the same guard branch.
