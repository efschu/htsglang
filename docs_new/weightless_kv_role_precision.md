# Task #127 — per-role KV precision on the weightless-KV lane (fp8 on the workers)

Design note for `--weightless-kv-worker-cache-dtype`: letting the weightless
KV workers of the Variant-C lane store their KV token-shard at a different
precision than the head rank.

**Why the idea is attractive.** A weightless worker's entire VRAM budget is
KV cache — it holds no layer weights, no optimizer state, no activations
beyond a staging region. Per-token bytes are therefore the *only* capacity
knob it has, and halving them is an exact 2× on its slot count. The head
rank is a different animal: its card is mostly weights, and its KV shard is
whatever is left over. Two roles, two cost structures, no reason to force one
storage format on both.

**What this note establishes.** (1) fp8 KV already works on this lane through
the group-wide `--kv-cache-dtype`; the delta being built is the per-role
split, not fp8 itself. (2) The split is cheap and safe because KV bytes never
cross the role boundary in *storage* format. (3) The capacity win is **not**
unconditional — under the lane's even-modulo owner rule it depends on which
rank binds the group capacity, and that has to be measured, not assumed.
Section 5 is the honest arithmetic; it is the most important section here.

---

## 1. Inventory — what already exists

### 1.1 fp8 KV is not new territory on this fork

`--kv-cache-dtype fp8_e5m2` is established practice: it is in the standing
uneven-DCP validation launchers (`r3val/dcp_launch.sh`, `r3val/a_launch.sh`)
and in the r3 integration evidence. The engine-side machinery is complete and
none of it is lane-specific:

| stage | where | behaviour under fp8 |
|---|---|---|
| spec → dtype | `ModelRunner.configure_kv_cache_dtype()` | `fp8_e5m2` → `torch.float8_e5m2`, `fp8_e4m3` → `torch.float8_e4m3fn` (HIP remaps both) |
| pool storage | `KVCache.__init__`, `mem_cache/memory_pool.py` | `store_dtype = torch.uint8` for every fp8 dtype; `self.dtype` stays the logical fp8 view (`index_put` has no fp8 kernel) |
| write | `MHATokenToKVPool.set_kv_buffer` | `div_(k_scale)` → `.to(fp8)` → `.view(uint8)`, then the fork's masked scatter kernel, which is dtype-transparent |
| read | `get_kv_buffer` | returns `.view(self.dtype)`; **no dequant here** |
| attention | `flashinfer_backend` | `plan(kv_data_type=model_runner.kv_cache_dtype, q_data_type=model_runner.dtype)` + `k_scale=layer.k_scale_float` per call; flashinfer dequantizes |
| capacity | `MemoryPoolConfigurator._compute_cell_size` | `kv_size = element_size(kv_cache_dtype)` as a single multiplicative factor — fp8 is a **pure 2× slot win**, with no scale-buffer term (unlike fp4) |
| host tiers | `pool_host/base.py`, `pool_host/mha.py` | `HostKVCache.dtype = device_pool.store_dtype`; `token_stride_size` follows; every transfer is a raw byte copy |
| CUDA graphs | the graph runners | capture no KV-shaped tensor; the pool is captured by pointer, so the pool dtype simply has to be final before capture (it is — resolution happens during `load_model`) |

So the lane inherits a working fp8 path. **Nothing in this task is a new
numerical format.** What did not exist is any notion of a *role*.

### 1.2 The lane has no per-role dtype anywhere

`grep -n "is_weightless" model_runner_kv_cache_mixin.py` returns nothing. The
mixin only ever consults the process-global `weightless_kv_active()`, which is
identical on every rank. One `model_runner.kv_cache_dtype` flows to the head
and to every worker, from one `--kv-cache-dtype` string.

### 1.3 The load-bearing fact: KV bytes cross roles in COMPUTE dtype

This is what makes the split cheap. Per attention layer the head projects
K/V and both roles enter the same fused all-gather:

```
FlashInferAttnBackend._dcp_write_gather   ->  cp_all_gather_heads_uneven(cat(k, v), group, [total_kv, 0, 0, ...])
worker side (forward_*_weightless_worker) ->  contributes an empty [T, 0, D] slice of dtype self._wl_dtype
                                          ->  _dcp_owner_write(...) == the head's _dcp_write_scatter
```

and `self._wl_dtype = mc.dtype` — the **model compute dtype**, chosen so the
padded all-gather's shapes and dtypes agree between roles. The wire therefore
carries bf16/fp16 K/V, and each rank quantizes into its own pool locally
inside `set_kv_buffer`. Consequences:

* a per-role storage dtype needs **no conversion point on the wire**;
* the collective count, order and payload are untouched, so the Lock-Step
  contract (#131/#133) is untouched;
* `_wl_dtype` must **never** be re-wired to `kv_cache_dtype`. That is now
  pinned by a source-level test (§7), because the failure mode is an NCCL
  fault or a double-quantized value, not an exception.

### 1.4 The read path needs no fork kernel work

The lane is flashinfer-only (`server_args` hard-rejects any other backend;
`triton_backend.reject_unsupported_dcp_geometry` refuses the lane in both of
its branches). Every worker-side read — the monolithic decode, the B0 block
loop `_wl_blockwise_decode_return_lse`, its #136a graph twin, and the extend
prefix stream — goes through a flashinfer wrapper planned with
`kv_data_type=kv_cache_dtype` and run with `k_scale`/`v_scale` floats. There
is no fork-local dequant to write. (For the record, had the lane been on
Triton: `kernels/ops/attention/decode_attention.py` folds K's descale into
`sm_scale` and applies V's descale to the accumulator — also no explicit
dequant, but it is not the lane's path.)

### 1.5 Two defects found during the inventory

**(a) The per-token cell size under-charges on the lane.** The pool
allocation site charges the FULL kv-head count for the lane —

```python
if (uneven_dcp_kv_replicated(self.dcp_size) or weightless_kv_active()) and not _draft_non_dcp:
    _hybrid_kv_head_num = self.model_config.get_total_num_kv_heads()
```

— because the head projects all kv-heads and broadcasts them, so every rank
writes all of them into its owned slots. But `_compute_cell_size` tested only
`uneven_dcp_kv_replicated(...)`, and the lane runs with `rank_tp_ratio=None`,
so that predicate is False. The cell was therefore charged
`get_num_kv_heads(attn_tp_size)` while the pool was built at
`get_total_num_kv_heads()` — **under-charged by the head ratio**, inflating
`max_total_num_tokens` by the same factor and sizing the pool past the rank's
own budget. Post-capture KV resizing hides it when it runs; with graphs off,
or at a large context, it surfaces as the previously-observed
weightless-worker OOM that was worked around with a manual
`--max-total-tokens`. Fixed here (§6), with a red-without-the-fix unit test.

The same rule was missing from `_pool_kv_head_num()`, which shapes the
**plain-MHA** pool (non-hybrid models on the lane) — that pool would have been
built for a per-rank shard the head's broadcast does not match. Also fixed.

**(b) No dtype agreement check between a device pool and its host tier.**
Every host↔device move on the weightless spill tier (#134 B1) and on the
kv-session-offload tier is a raw byte copy driven by
`host_pool.token_stride_size`. Nothing downstream re-derives an element type.
A host tier built from a different dtype than its device pool would not fail
— it would reinterpret KV bytes at the wrong stride and return plausible
garbage. That seam was theoretical while one dtype was group-wide. It is not
theoretical with per-role precision.

---

## 2. The design

### 2.1 One flag, one direction

```
--weightless-kv-worker-cache-dtype {auto,fp8_e5m2,fp8_e4m3,bf16,bfloat16}
```

* Default `None` (and `auto`) = **inherit** `--kv-cache-dtype`. Identity. The
  default path and every existing lane boot are byte-identical.
* Applies **only on weightless worker ranks**. The head always keeps
  `--kv-cache-dtype`.
* Requires `--weightless-kv-fastlane` (hard error otherwise: without the lane
  there are no worker ranks to address).
* `fp4_e2m1` is refused. An fp4 pool carries a separate per-block scale buffer
  in the cell and its own pool class; neither has been exercised on this lane.
  That is a scope statement, not a capability claim.

Only the worker direction is exposed. A head-only override would be the same
mechanism with the roles swapped, and §5 shows it is often the more valuable
direction — but the head runs the head-local paths whose machine-zero byte
contract (#124 row `head_local_prefill`) is the lane's numerical anchor, so
changing the head's KV format is a separate decision with a separate gate.
Today that direction is reachable by simply setting `--kv-cache-dtype` for
the whole group, which is the honest way to ask for it.

### 2.2 Where the split happens: one input, one resolver

The override resolves a **spec string**, not a torch dtype:

```python
kv_spec = effective_kv_cache_dtype_spec(
    server_args.kv_cache_dtype,
    server_args.weightless_kv_worker_cache_dtype,
    self.is_weightless_worker,
)
```

and that string is fed through the existing, otherwise-unmodified
`configure_kv_cache_dtype()` branch chain. There is deliberately **no second
resolver**: HIP remapping, checkpoint auto-detection and the fp4 fallback all
stay in one place, so the per-role path cannot drift from the global one
about what `fp8_e5m2` means. A unit test asserts that no dispatch branch in
that method still reads the raw group spec.

`ModelRunner.is_weightless_worker` is computed in `__init__` from server args;
`configure_kv_cache_dtype()` runs later, during `load_model`. No ordering work
was needed.

### 2.3 Everything downstream follows for free

Because the role decision lands on `model_runner.kv_cache_dtype` itself, every
consumer is already rank-local and already correct:

| consumer | why it is fine |
|---|---|
| pool construction (`HybridLinearKVPool` / `MHATokenToKVPool`) | takes `dtype=self.kv_cache_dtype` |
| `_compute_cell_size` | takes `mr.kv_cache_dtype`; a worker's cell is half a bf16 cell |
| flashinfer `plan(kv_data_type=...)` | reads `model_runner.kv_cache_dtype` per rank |
| host spill tier | `per_token_bytes` derives from `full_pool.store_dtype.itemsize`; the pinned tier halves too |
| B0/B1 staging region | carved out of the pool in **slots**, not bytes — dtype-agnostic by construction |
| #136a/#136b graph capture | captures the pool by pointer; dtype is fixed before capture |

### 2.4 What is explicitly *not* touched

* **The wire.** `_wl_dtype` stays `mc.dtype` (§1.3).
* **The Lock-Step.** dtype is not a runtime decision; it is fixed per rank at
  boot from replicated arguments. No collective negotiates it, no branch
  depends on it, so the per-step collective count and order are identical.
  Rank-local test before the collective, per the standing rule.
* **The LSE merge.** `cp_lse_ag_out_ar_mha_uneven` merges attention *outputs*
  in fp32, already dequantized by flashinfer. Storage format is invisible to
  it. A head partial computed over bf16 KV and a worker partial computed over
  fp8 KV are both fp32 (out, lse) pairs on the same scale and merge exactly as
  before.

---

## 3. Grenz-assertions instead of silent reinterpretation

Three places could reinterpret bytes rather than fail. Each now raises.

**(a) Host tier vs device pool.** `host_tier_stride_mismatch()` compares the
host tier's element size and `token_stride_size` against the device pool's
`store_dtype.itemsize × head_num × head_dim`, and is called where both
objects first exist (`_wl_attach_spill_host_pool`). A mismatch names both
sides.

**(b) KV scales across roles.** KV scaling factors are per-tensor floats that
load into *model weights*. A weightless worker holds a meta-device model and
deliberately skips that load (`kv_cache_dtype == "fp8_e4m3" and not
is_weightless_worker` in `load_model`). With a scale file present, the head
would quantize against a loaded scale and the worker against the 1.0 default —
two shards on different scales, merged as if comparable, with nothing to
notice it. So: `--quantization-param-path` is refused together with a role
split at arg-parse time, and `_wl_verify_role_kv_scales()` re-checks the
actual layers once per process, refusing any non-unit `k_scale_float` /
`v_scale_float` on a worker under a split. Inheriting (the default) keeps the
existing group-wide behaviour with scale files fully available.

**(c) fp4.** Refused up front (§2.1) rather than allowed to build a pool whose
cell arithmetic does not match its class.

---

## 4. Accuracy budget and the lossy-class label

fp8 KV storage is lossy. Its class and treatment:

* **Opt-in, default off.** `None` inherits; the lane's default path is
  unchanged and byte-identical.
* **Not new risk.** This is the same fp8 KV storage the engine already
  supports group-wide and that the fork's own uneven-DCP validation runs on
  (`--kv-cache-dtype fp8_e5m2` in the standing launchers). The per-role flag
  changes *which ranks* store fp8, not *how* fp8 is stored. The novel part is
  a mixed-format group, and §1.3/§2.4 establish that the two formats never
  meet in a shared buffer.
* **Where the error enters.** Exactly one place: `set_kv_buffer`'s
  `.to(torch.float8_*)` round-to-nearest on the worker's owned slots. Tokens
  owned by the head keep the head's precision. Under the even-modulo rule the
  worker-owned fraction is `(dcp_size − 1) / dcp_size` of all tokens — 2/3 at
  TP=3 — so a role split is *not* a small perturbation of full-fp8; it is
  most of it, with the head's third kept exact.
* **Byte-gate class.** A role split cannot be a machine-zero row. Expect
  `DECODE_CLASS` at best against a bf16-KV reference, and note that the
  head-local prefill row stays machine-zero precisely *because* the head's KV
  format is untouched — that is a real, checkable benefit of splitting by role
  rather than switching the group.
* **Gate policy.** Token identity is only a meaningful gate on short prompts
  (<109 tokens on GDN models, which drift above that regardless); longer
  prompts get semantic checks plus an accept-length band if speculation is in
  play.

---

## 5. The capacity arithmetic — read before expecting a win

This is the part that changes how the feature should be sold.

Under the lane's **even-modulo** owner rule, token `L` is owned by rank
`L % dcp_size` and stored at compacted slot `L // dcp_size` — the *same* slot
index on every rank. The slot space is rank-uniform, so
`_apply_token_constraints` MIN-reduces the per-rank token capacities into one
number, and

```
global_capacity = dcp_size * min_r(per_rank_capacity_r)
```

Halving the workers' per-token bytes raises `per_rank_capacity` on the workers
only. Therefore:

* if a **worker** is the binding (minimum) rank → the group gains, up to 2×;
* if the **head** is the binding rank → the group gains **exactly nothing**,
  and the only effect of the flag is the accuracy cost.

Which case you are in is a property of the deployment, not of the design.
Worked example on the reference rig (head = 5090 26000 MiB, workers = 3080
17000 MiB, per the #136a/#136b boots), writing `b` for bf16 bytes/token:

| arrangement | head slots | worker slots | binding | global |
|---|---|---|---|---|
| all bf16, ~13 GB of weights on the head | ~11 GB / b | ~16 GB / b | head | `3 × 11/b` |
| **workers fp8**, head bf16 | ~11 GB / b | ~32 GB / b | head | `3 × 11/b` — **no change** |
| group-wide fp8 (`--kv-cache-dtype`) | ~22 GB / b | ~32 GB / b | head | `3 × 22/b` — 2× |
| workers fp8, small model / roomy head | ~30 GB / b | ~32 GB / b | worker | `3 × 32/b` — 2× |

So on a rig where the head's weights dominate its card, the lever is the
group-wide flag, not the per-role one; the per-role flag is for the
configuration the lane was *named* for — small worker cards carrying the KV,
a head with slack. Rather than guess, the boot now **prints the binding rank**
for every rank in the min-sync branch:

```
KV token sizing: rank 1 local capacity 5908290 tokens, min-reduced across ranks
to 196943 (another rank binds; 5711347 stranded on this rank).
Global addressable KV = 196943 x dcp_size(3).
```

`even_modulo_global_capacity()` encodes this as a pure function, and the unit
tests assert both the "worker fp8 buys nothing when the head binds" case and
the "worker fp8 doubles the group when a worker binds" case, so the caveat
cannot quietly rot out of the docs.

**Escape hatch, not built here.** The *weighted* uneven-DCP owner rule
(`--rank-tp-ratio` + `--rank-kv-ratio`) sizes capacity as
`min_r(P_r / ratio_r) * S`, i.e. it exploits genuinely unequal per-rank pools —
under which worker fp8 does pay even with a head-bound rig, because the token
vector can shift ownership toward the fp8 ranks. The weightless host-spill
tier currently hard-rejects `--rank-tp-ratio` (its static slot→tier map is
defined on the even compaction). Combining per-role precision with weighted
DCP is the natural follow-on and is deliberately out of scope here.

Note also that on hybrid GDN models the `_apply_hybrid_kv_token_cap` ceiling
(`max_running_requests × (context_len + headroom)`) can bind before either
KV budget does, in which case no KV precision change moves anything. The boot
log names that cap too.

---

## 6. Change inventory

| file | change |
|---|---|
| `python/sglang/srt/layers/dcp/role_kv_dtype.py` | **new.** Pure decision functions: `effective_kv_cache_dtype_spec`, `worker_dtype_is_role_split`, `even_modulo_global_capacity`, `host_tier_stride_mismatch`; the worker choice list and the lossy-spec label. |
| `python/sglang/srt/server_args.py` | new `weightless_kv_worker_cache_dtype` arg; requires the lane; refuses fp4 and `--quantization-param-path` under a split. |
| `python/sglang/srt/model_executor/model_runner.py` | `configure_kv_cache_dtype()` resolves the role spec first and branches on it; `_wl_verify_role_kv_scales()` grenz-assertion wired into `_weightless_attn_layers()`. |
| `python/sglang/srt/model_executor/pool_configurator.py` | **bugfix** — `_compute_cell_size` charges the full kv-head count on the lane (§1.5a). |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | **bugfix** — `_pool_kv_head_num()` likewise; host-tier stride grenz-assertion; binding-rank sizing log. |
| `test/registered/unit/model_executor/test_weightless_role_kv_dtype.py` | **new**, CPU: 24 falsifiers (§7). |
| `test/registered/unit/distributed/test_uneven_tp_memory.py` | fake gains `tp_rank` for the new sizing log. |

---

## 7. Test surface (CPU)

`test/registered/unit/model_executor/test_weightless_role_kv_dtype.py`:

* **resolution** — default path is the identity; the head never takes the
  worker spec; `auto` means *inherit*, not *re-detect independently*; the
  role-split predicate; lossy specs are labelled and unreachable by default.
* **arg validation** — flag without the lane; fp4; `--quantization-param-path`
  under a split refused, and still available when inheriting; a valid split
  parses.
* **capacity** — `dcp_size × min`; the head-binds case buying nothing; the
  worker-binds case doubling; malformed vectors.
* **host tier** — matching tiers pass; an itemsize mismatch and a stride
  mismatch are both *named*, not silent.
* **wire dtype** — source-level: `_wl_dtype` follows the compute dtype and is
  never assigned from a KV dtype.
* **resolver reachability** — source-level: no dispatch branch in
  `configure_kv_cache_dtype` still reads the raw group spec.
* **cell size** — stock path unchanged (byte-identity guard for every
  non-weightless deployment); the lane charges full kv-heads (**red without
  the §1.5a fix**); fp8 halves the cell with no scale term.
* **pool storage** — fp8 pools are uint8-backed and half-width.

---

## 8. GPU recipe (to run at the next window; no boots were taken for this task)

Vehicle and topology from the #124 harness (`tests/determinism/`), which is
the validated lane configuration: Qwen3.6-27B Q3_K_M GGUF, `--dtype float16`,
`--attention-backend flashinfer`, TP=3/DCP=3, `--rank-gpu-id 0,1,2`,
`--rank-gpu-memory-mib 26000,17000,17000` (CUDA order — 5090 is 0 on this
rig; resolve, never assume). Head rank 0 = 5090, workers = the two 3080s.

**A/B capacity.** Same boot twice, differing only in
`--weightless-kv-worker-cache-dtype` (`auto` vs `fp8_e5m2`). Record from the
boot log: each rank's `KV token sizing: rank R local capacity ...` line, the
binding rank, and the final `max_total_num_tokens`.

* Expected on this rig: worker local capacity doubles; **the global may not
  move** because the head binds (§5). That is a PASS for the mechanism and a
  measurement of the caveat — the falsifier is a worker capacity that does
  *not* double, or a global that moves while the head still binds.
* To exercise the case the feature is for, add a third arm with the head
  given a much larger budget (e.g. `--rank-gpu-memory-mib 30000,10000,10000`,
  or a smaller model) so a worker binds; the global should then double.
* Cross-check the §1.5a fix separately: with `auto` on both arms, the new cell
  size should lower `max_total_num_tokens` relative to the pre-fix branch and
  the pool should now fit without a manual `--max-total-tokens` workaround at
  a context where the worker previously OOM'd.

**Quality gate.** Short low-entropy prompt (the harness's counting/copy
prompts, <109 tokens): greedy token identity, worker-fp8 arm vs `auto` arm.
Expect flips only at genuine near-ties; the discriminator is argmax-clean and
non-compounding, not bit equality. Long prompt: semantic check (needle
retrieval on the spill path). Self-determinism 5/5 on the fp8 arm.

**Path coverage.** The fp8 worker arm must be re-run across the lane's four
read paths, since each plans its own flashinfer wrapper:
monolithic decode; B0 block decode (`--weightless-kv-chunked-block-size
1024`); B1 host spill (`--weightless-kv-host-spill-tokens 16384
--weightless-kv-spill-device-cap 2048` — this also exercises the halved host
tier and the new stride assertion); and the #136a/#136b graph ladder (graphs
on, which is the default and the full-perf discipline).

**Harness rows.** The natural additions to `tests/determinism/matrix.py` are a
`weightless_worker_fp8` row (test = lane + worker fp8, reference = lane
inheriting, class `DECODE_CLASS` with a band) and a `head_local_prefill_under_
role_split` row asserting the head-local path stays `MACHINE_ZERO` — the
latter is the concrete claim §4 makes about splitting by role.

**Follow-on measurement.** #143 (speculation on the lane) is currently
hard-rejected, so accept-length bands are not measurable yet; note the
combination as a later slice rather than a gate here.

---

## 9. int4 KV — design only, not built

The task frames int4 as "perspektivisch". It is a different problem, and the
difference is worth writing down before anyone treats it as "fp8 but smaller".

**What carries over.** The role plumbing is format-agnostic: one spec string
per role through one resolver, no wire conversion, host tier follows
`store_dtype`, staging carved in slots. An int4 worker pool would reuse all of
it.

**What does not.**

1. **Per-tensor scales stop being enough.** fp8 works with a single float per
   layer per K/V (`BaseKVCacheMethod` hard-rejects anything else) because
   e4m3/e5m2 have real exponents. int4 has 16 levels total and needs
   *group-wise* scales (per head, per 32/64-element block) to be usable at
   all. That means a second buffer in the pool, a second term in
   `_compute_cell_size` (fp4 already shows the shape of this: `cell // 2 +
   scale_bytes`), and it erodes the nominal 4× — a 4-bit value with an fp16
   scale per 32 elements is 4.5 bits/element, so ~3.5× not 4×.
2. **No kernel reads it.** flashinfer's paged wrappers take
   `kv_data_type ∈ {fp16, bf16, fp8_e4m3, fp8_e5m2}` (and fp4 via a separate
   pool class). There is no int4 KV entry point, so unlike fp8 this is *not*
   a configuration change — it requires either a dequant-into-staging step
   before every wrapper call (killing the point on the block-streaming path,
   which is already copy-bound) or a genuine fork kernel.
3. **The block-streaming lane makes option (a) less bad than it sounds.**
   B0/B1 already stage blocks through a bounded device region. An int4 tier
   could store int4 *on the host* and dequantize during the H2D staging step
   into an fp8/bf16 device staging region — halving PCIe traffic on the tier
   that is bandwidth-bound, with device-side attention unchanged. That is a
   much smaller and much more targeted change than an int4 device pool, and
   it is the shape an int4 slice should probably take on this lane.
4. **Accuracy.** int4 KV without group scales is not competitive; with them it
   is roughly "fp8-class quality at ~0.6× the bytes", which on a
   min-reduced, head-bound rig may buy nothing at all (§5 applies unchanged).

**Precondition for opening it.** fp8-on-workers has to show a measured
capacity gain on a worker-bound configuration first. If §5's caveat turns out
to dominate every configuration we actually run, an int4 pool is more bytes
saved on a dimension that is not binding, and the effort belongs on the
weighted-DCP combination instead.
