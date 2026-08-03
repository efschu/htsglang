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
Per the "Rig ist Untergrenze" rule this is not a reason to downgrade the
finding: the kernel ships for every rig, and a single-GPU host with enough
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

### 1.4 What Axis 1 did NOT cover (honest coverage)

- `sgl-kernel/csrc` was swept for **byte-pointer arithmetic** (`(char*)p + x`
  / `(uint8_t*)p + x` without an int64 operand): exactly one hit, A1-1. It
  was **not** exhaustively swept for `int` element-index arithmetic — that is
  thousands of sites, and the ones that matter are those whose stride is in
  bytes or whose extent is a whole tensor. The named families read:
  gguf (moe, moe_vec, mmvq, mmq, dequantize, gguf_kernel), kvcacheio,
  elementwise/copy, elementwise/pos_enc, gemm/per_token_group_quant (v1+v2),
  moe/moe_align_kernel, barlink BAR1 inline CUDA.
- Triton: `layers/dcp/kernels.py` read in full; `layers/attention/dsv4/*`
  read for offset dtype (the heavy indexer math is chunked torch, not a
  custom flat-offset kernel — `indexer.py:237,552,585,826,1081` are id
  tensors bounded by page/topk counts). The ~40 other Triton files with
  `tl.arange`/`tl.int32` hits (fla/, mamba/, moe/ep_moe, quantization/*) were
  **not** read line-by-line.
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
