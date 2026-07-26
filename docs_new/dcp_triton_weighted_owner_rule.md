# Task #173 — the weighted (uneven) DCP owner rule in the Triton attention backend

Design note for the port of the fork's uneven-DCP machinery from the
FlashInfer backend into `layers/attention/triton_backend.py`.

**Why.** The Nordstern target is TP across ranks that have no FlashInfer
(sm75 Turing, gfx900 Vega). Today the weighted owner rule exists only in
`flashinfer_backend.py`, and `reject_unsupported_dcp_geometry()` refuses
every uneven-DCP geometry on Triton (#169.1). A `kv < tp` model of the
27B/35B class therefore cannot run uneven DCP on a rank that has to use
the Triton backend. This note records what the FlashInfer path actually
does, which pieces are backend-agnostic already, and where Triton has to
diverge.

---

## 1. The mechanism, as it exists in the FlashInfer path

### 1.1 The owner rule itself is already backend-agnostic

`python/sglang/srt/layers/dcp/owner.py` (relocated verbatim out of
`flashinfer_backend.py`) is pure torch plus the shared Triton kernel
`create_flashinfer_kv_indices_triton`:

| symbol | side | what it computes |
|---|---|---|
| `dcp_weighted_owner_bounds(dcp_size, rank)` | both | `(cp_S, cp_lo, cp_hi, cp_ratio)` from `cp_token_prefix()` |
| `dcp_weighted_write_slots(cache_loc, ...)` | WRITE | `(compact_loc, owned_mask)` for a batch of global slots |
| `build_dcp_weighted_kv_indices(...)` | READ | paged `kv_indptr` / `kv_indices` over this rank's owned slots |

The rule, stated once:

```
rank owns global cache slot L   <=>   (L % cp_S) in [cp_lo, cp_hi)
compact physical slot of L       =    (L // cp_S) * cp_ratio + (L % cp_S - cp_lo)
```

with `cp_S = sum(token_ratios)`, `cp_lo/cp_hi` the rank's prefix-sum range
and `cp_ratio = cp_hi - cp_lo`. All-ones ratios reduce it exactly to the
even modulo layout (`L % N == rank`, stored at `L // N`).

Ownership is a function of **`out_cache_loc`**, never of the sequence
position. That is what keeps the compact slot injective across concurrent
requests. The Triton *even* path is the other way round on the write side
(`positions % dcp_size == dcp_rank`, compacted by `loc // dcp_size`) — it
gets away with it because the even allocator's index space guarantees
`loc % N == pos % N`. The weighted rule does not have that property and
must stay loc-based on both sides.

### 1.2 What FlashInfer wires around the rule

`FlashInferAttnBackend.__init__` (≈ lines 606-790):

* `uneven_dcp = uneven_dcp_kv_replicated(dcp_size) or weightless_kv` **and
  not a draft worker**. The draft/NEXTN worker keeps a plain uneven-TP pool
  (local heads, FULL token context) — it is not token-sharded, so DCP is
  simply *off* for that backend instance.
* `uneven_dcp_weighted = uneven_dcp and uneven_dcp_active(dcp_size)`, then
  `cp_S / cp_lo / cp_hi / cp_ratio` cached once.
* `dcp_kv_replicated_heads = attn_kv_replicated(tp, total_kv)` — true when
  `kv < tp`, i.e. every rank projects **all** kv heads itself.
* `dcp_kv_head_counts`, `dcp_q_head_counts` — per-rank head vectors for the
  uneven collectives.
* `dcp_full_qo_heads = num_attention_heads`, `dcp_full_kv_heads = total_kv`.

The KV **pool** under this lane stores the FULL `total_num_kv_heads` per
slot on every rank (`model_runner_kv_cache_mixin.py` ≈ 2660) and is sized
`(max_total // S + 1) * ratio_r` rows — ratio-proportional, this rank's
owned share only. The allocator's index space is the global context `C`.

### 1.3 Write path (`_dcp_masked_write` = `_dcp_write_gather` + `_dcp_write_scatter`)

1. **gather** (`_dcp_write_gather`): if `dcp_kv_replicated_heads` -> no-op.
   Else one fused all-gather of `cat(k, v)` along the head dim via
   `cp_all_gather_heads_uneven(..., dcp_kv_head_counts)`, so the row written
   into the pool always carries the full replicated kv-head set.
2. **scatter** (`_dcp_write_scatter`): weighted -> `dcp_weighted_write_slots`;
   even -> `loc // dcp_size` + `positions % dcp_size`. Then
   `set_kv_buffer(..., dcp_kv_mask=mask)`.

### 1.4 Read path

* decode (`call_begin_forward`, ≈ 6163): `build_dcp_weighted_kv_indices` on
  the weighted branch, `get_dcp_lens` + `create_triton_kv_indices_for_dcp_triton`
  on the even one.
* extend/prefill (`call_begin_forward` ragged/paged split, ≈ 6657): identical
  index build over `prefix_lens` (with `pad=256`), plus the structural rule
  that DCP **always** splits the forward into
  - ragged **current chunk**, LOCAL heads, causal, no collective, and
  - paged **prefix**, gathered q heads, non-causal, cross-rank LSE merge.
* verify (M4, ≈ 6725): same split with the committed prefix as the paged part.

### 1.5 Compute path

decode: `q_full = cp_all_gather_heads_uneven(q_local, ...)` ->
`forward_return_lse` -> `cp_lse_ag_out_ar_mha_uneven(o, lse, ...)`.

extend: ragged current chunk (local heads) merged with the LSE-combined
paged prefix by `torch.logaddexp` in `_dcp_extend_final_merge`.

`_replicated_kv_ragged_reindex` (#105) is the one non-obvious piece: under
REPLICATED-KV the ragged current-chunk kernel derives its GQA grouping from
the LOCAL counts (`local_q // local_kv`), which is *not* the global grouping
whenever the rank does not hold every q head of its kv head. FlashInfer
re-indexes the local kv slots so a uniform-GQA kernel reproduces the global
q->kv mapping, and raises if this rank's q heads straddle a global kv-head
boundary.

---

## 2. What the Triton backend already has (after #169)

* `cp_all_gather_heads_uneven` / `cp_lse_ag_out_ar_mha_uneven` are wired
  through `_dcp_gather_q_heads` / `_dcp_merge_q_heads` with per-rank counts
  from `_plan_aware_dcp_group_q_head_counts` (#169.4). The head geometry is
  therefore uneven-capable already, and the per-rank count comes from the
  model.
* `_forward_extend_dcp` already has exactly the FlashInfer *structure*:
  current chunk (local heads, causal, `skip_prefix=True`) merged by
  `logaddexp` with the gathered-q paged prefix (`skip_extend=True`).
* `_dcp_kv_indices` / `_set_kv_buffer` implement only the EVEN modulo rule.
* `reject_unsupported_dcp_geometry` refuses the whole uneven lane.

So the port is **index math + head bookkeeping**, not a new attention kernel.

---

## 3. Where Triton has to diverge from FlashInfer

| # | Concern | FlashInfer | Triton |
|---|---|---|---|
| D1 | pool kv-head count | wrappers planned with `dcp_full_kv_heads` | `self.num_kv_head` feeds the split-KV schedule; must become `total_kv` under the uneven lane (pool holds the full set) |
| D2 | prefix-stage GQA grouping | wrapper is planned with explicit head counts | `extend_attention_fwd` derives `kv_group_num = q_extend.shape[1] // k_extend.shape[1]`. The prefix stage passes an EMPTY `k_extend`, so that empty tensor's head count decides how `k_buffer` is indexed — it must carry the **pool's** kv-head count, not this rank's projection count |
| D3 | cuda-graph replay | `fast_decode_plan` reads the wrapper's fixed buffers; FlashInfer copies into them | Triton's captured kernels read `self.cuda_graph_kv_indices`. `build_dcp_weighted_kv_indices` allocates a FRESH tensor -> must be copied into the capture-stable buffer, with a capacity check |
| D4 | `num_kv_splits` sizing | n/a | Triton sizes the split-KV schedule from the per-rank kv length; under the weighted rule that length is `diff(kv_indptr)`, not `get_dcp_lens` |
| D5 | empty-shard early return | `has_prefix` is computed from the GLOBAL `extend_prefix_lens_cpu` | Triton returns early on `kv_indices.numel() == 0`, i.e. on a **rank-local** quantity. Under the weighted rule a low-ratio rank can own zero prefix slots while its peers own some -> it would skip the q-gather and the LSE merge and hang the group. Must be re-based on the global prefix length |
| D6 | draft worker | `uneven_dcp` excludes it; DCP branches simply do not run | Triton branches on `self.dcp_size > 1` everywhere -> the draft backend must set `dcp_size = 1` for itself under the uneven lane |
| D7 | ragged current chunk | `_replicated_kv_ragged_reindex` | same rule, applied to the `extend_attention_fwd` current-chunk stage |

D5 is a **pre-existing latent hang in the even Triton path** as well
(prefix length 1 over dcp 3: rank 0 owns the slot, ranks 1-2 own none), so
it is fixed for both rules; no currently-working configuration changes
behaviour.

---

## 4. Scope of this port (V1)

**Served after this change** (`reject_unsupported_dcp_geometry` opens):

* `uneven_dcp_kv_replicated` lane (a `--rank-tp-ratio` plan installed,
  `dcp_size == tp_size`), with **either** owner rule:
  * weighted (non-uniform token vector) — the new code;
  * even-modulo under the plan (uniform/absent token vector) — same wiring,
    weighted branch inert.
* both kv-head geometries: REPLICATED-KV (`kv < tp`, the 27B/35B Nordstern
  case, no kv collective at all) and head-sharded (`kv >= tp`, one fused
  kv-head all-gather per layer on the write).
* decode, prefill/extend, and chunked prefill.

**Still refused, loudly, at construction time:**

| geometry | reason |
|---|---|
| weightless-KV fast lane (`[all,0,0]` heads) | its own dispatch (block decode, host spill, broadcast K/V) has no Triton twin |
| MLA under the uneven lane | Triton's MLA decode is a different kernel family, unvalidated here |
| speculative decoding under the uneven lane | Triton's `is_target_verify` metadata builds FULL (un-sharded) kv indices — global slot ids into a compact pool. Already unusable under DCP today; refusing is the loud version |
| sliding window under the uneven lane | `_forward_extend_dcp` cannot mask a sparse owned-slot subset; was a per-layer `NotImplementedError`, now a boot-time refusal |
| a token vector WITHOUT a `--rank-tp-ratio` plan | mixed pool state (pool sized weighted, heads not replicated) |
| even DCP without kv-head replication (no plan) | unchanged #169.2 rule |

One further refusal is raised at the first forward rather than at boot,
because it depends on the layer and not on the server args: a head-sharded
uneven lane (`kv >= tp`) on a model with `qk_head_dim != v_head_dim`. The
fused `cat(k, v)` kv-head all-gather cannot stack two different head dims,
and doing two separate gathers would double the per-layer collective count
rather than silently work. `kv < tp` never reaches it (no gather at all).

## 4b. What actually landed

* `layers/dcp/owner.py`: `dcp_weighted_read_slots` and
  `dcp_weighted_owned_lengths` extracted; `build_dcp_weighted_kv_indices`
  now composes them and takes an optional `req_to_token_stride`.
* `triton_backend.py`:
  * constructor: `uneven_dcp` / `uneven_dcp_weighted` / `cp_S,cp_lo,cp_hi,
    cp_ratio` / `dcp_kv_replicated_heads` / `dcp_kv_head_counts` /
    `dcp_full_{qo,kv}_heads`, `num_kv_head = total_kv` for the lane, and
    `dcp_size = 1` for a draft worker under the lane (D6);
  * `_dcp_weighted_kv_indices` (read side, D3 buffer contract, D4 lengths);
  * `_dcp_write_gather` + the weighted branch of `_set_kv_buffer`;
  * `replicated_kv_reindex` / `_replicated_kv_ragged_reindex` (D7);
  * `_dcp_batch_has_prefix` (D5);
  * prefix-stage placeholder head count (D2);
  * `reject_unsupported_dcp_geometry` restructured into three branches.

---

## 5. Test strategy

Triton kernel numerics need a GPU. The **owner rule and the index math do
not** — they are pure integer tensor arithmetic. So:

* `layers/dcp/owner.py` grows two extracted pure functions,
  `dcp_weighted_read_slots` and `dcp_weighted_owned_lengths`, which
  `build_dcp_weighted_kv_indices` then calls. The Triton-kernel part
  (`create_flashinfer_kv_indices_triton`, which only materialises
  `req_to_token[req, kv_start : kv_start+len]`) is the only GPU-bound step,
  and it is *shared* with FlashInfer.
* CPU parity tests compare those pure functions against an independent
  numpy reference **and** against the flashinfer-side expression, over
  several token plans and ranks, asserting: read/write inverse, partition,
  no collision, per-request owned lengths, and full-set reconstruction.
* A GPU-marked test file exercises the Triton wrapper end to end
  (`req_to_token` -> `kv_indptr`/`kv_indices`) against the same numpy
  reference, and the `_dcp_kv_indices` cuda-graph buffer contract.

---

## 6. GPU validation recipe (to run in a GPU window)

Nothing below was run for this change: the implementation phase had no cards.
Ordered cheapest-first, each step is a falsifier for the one after it
("Einzelteil vor Verbund").

### G1 — index math on a device, no model, no collectives (~1 min, any 1 GPU)

```
PYTHONPATH=<worktree>/python .venv/bin/python -m pytest \
  test/registered/unit/distributed/test_triton_weighted_dcp_gpu.py -v
```
Covers: the shared kv-index kernel composed with the owner rule; the
`kv_start_idx` (chunked-prefill) offset; read rows == write rows; the
capture-stable buffer contract; the buffer-overflow refusal; the
owns-nothing case. If this fails, nothing further is worth booting.

### G2 — the guard actually opens (~2 min, main rig)

Boot the 27B uneven-DCP config with `--attention-backend triton` and confirm
the process gets past `TritonAttnBackend.__init__` instead of raising
"Triton attention cannot serve --dcp-size". Then confirm the four refusals
still fire, by adding each of `--speculative-*`, an MLA model, a
sliding-window model, and the weightless-KV flag to that same command: each
must abort at boot naming itself.

### G3 — single-rank coherence before any TP (~5 min, 5090 alone)

Same model, `--attention-backend triton`, `--dcp-size 1`. This is the
BASELINE the parity anchor is compared against and it must be unchanged from
before the branch (the whole weighted path is inert at `dcp_size <= 1`).

### G4 — the parity anchor: 27B uneven DCP, triton vs flashinfer (main rig)

The point of the task. Same model, same `--rank-tp-ratio`, same token
vector, same seed, greedy, **no** speculative decoding (refused under this
lane, see §4):

* run A: `--attention-backend flashinfer`
* run B: `--attention-backend triton`

Compare on the same prompt set: (a) coherence of B on its own; (b) A vs B
token sequences under greedy decoding. Exact token identity is NOT expected
across two different attention kernel families — the acceptance bar is
"coherent, and the same answer semantically, with the first divergence late
rather than at token 1". A first-token divergence, or CJK/mojibake output,
means the owner rule is wired wrong, not that the kernels differ.

Run this in BOTH eager and CUDA-graph decode: G4 with graphs is what
exercises the D3 buffer contract, and a wrong-context replay shows up only
there ("Full-Perf-Testen": validate with graphs, not just eager).

Cover both prefill shapes: one short prompt (single chunk, exercises the
current-chunk ragged stage alone) and one long enough to chunk (exercises
the paged owned-prefix read + the LSE merge).

### G5 — the head-sharded sub-lane (`kv >= tp`)

G4's 27B is `kv < tp` (REPLICATED-KV: no kv collective, `_dcp_write_gather`
is a no-op). The other sub-lane — a model with `kv >= tp` under a
`--rank-tp-ratio` plan — is the only path that issues the fused kv-head
all-gather on the write, and it is untested by G4. Boot one such model
(e.g. a Qwen3.5 class with 8 kv heads at TP=3) and repeat G4's comparison.

### G6 — the empty-shard hang (D5), deliberately provoked

Send a request whose prefix is 1-2 tokens under a strongly uneven token
vector (e.g. `[13,30,21]`), so at least one rank owns ZERO prefix rows.
Before this change the group would hang there; it must now complete. Do this
on the flashinfer run too — the fix is in shared structure, and flashinfer
already gets it right, so the two must agree.

### G7 — sm75 / gfx900, the actual Nordstern reason

Everything above can run on the main rig. The point of the port is a rank
with no flashinfer. Repeat G3 and then G4 on the hetero host
(2080Ti = sm75, Vega64 = gfx900) once a rank there can hold the model at
all. Note the known gfx900 limits (stock Triton rejects gfx900; torch-native
falls back to MATH SDPA and OOMs on long prefill) — expect this step to be
gated on those, not on #173.

### Hosts

| step | host | why |
|---|---|---|
| G1 | any single GPU | no collectives, seconds |
| G2, G3 | main rig, one card | boot-only |
| G4, G6 | main rig, 5090 + 2x3080, TP=3 | the parity anchor |
| G5 | main rig | needs a `kv >= tp` model |
| G7 | hetero host 192.168.0.89 | the sm75/gfx900 target |
