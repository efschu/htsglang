# ANALYSE #854 — lued W8A16 vs. our W8A8-INT8 standard checkpoint

Desk analysis, 2026-08-24. No boots, no GPU claims, no cards touched.
Trigger: a Reddit KLD chart showing `lued/Qwen3.8-27B-INT8-W8A16-MTP` with notably
low KLD. Question: what is different, what would it cost to replace our INT8
standard model with it, pros/cons, recommendation.

## 0. Executive summary

1. **The headline KLD is not measured against us.** The lued card compares W8A16
   against Qwen's **official FP8** (0.000894 vs 0.004396 nats/token). There is no
   W8A8-INT8 arm in that comparison. "5x better than FP8" says nothing yet about
   "better than our INT8".
2. **The chart's numbers do not certify the serving path.** The card states the
   KLD was computed on the *HuggingFace decompressed path*, not through the Marlin
   kernel that would actually serve it. That is an unvalidated-indicator situation
   (INDIKATOR-GESETZ): the measured quantity is not the served quantity.
3. **The user's objection ("W8A16 is bigger in layout, so it costs KV cache") is
   correct on the like-for-like axis and wrong on the gross axis** — and the two
   axes are separable to the MiB. Pure W8A16 tax = **+355 MiB = -11.4k KV tokens
   = -2.5 % of pool**. But the lued checkpoint is nonetheless **4.42 GiB smaller
   than our active one**, because it quantizes 48 GDN layers that ours leaves in
   BF16. That second term is **not a W8A16 property** — we already own a W8A8
   checkpoint with the same coverage.
4. **The execution lane is clean.** Marlin `wNa16` admits sm86 **and** sm120, is
   JIT-compiled per arch, keeps weights packed in VRAM, and our uneven-TP
   coarsening already folds `group_size=128` correctly. **No dequant fallback is
   involved, and none exists for INT8 in this tree** — the lane is either Marlin
   or a hard error.
5. **The real cost of W8A16 is compute**, not VRAM: Marlin computes in BF16, so
   the ~77 % of GEMM parameters that today run on INT8 tensor cores lose their 2x.
   Modelled prefill-GEMM penalty vs. our active checkpoint: **~1.6x**. And the
   prize W8A16 was supposed to buy — dropping activation quant — was already
   measured by #368 under CUDA graphs at **11 % of the fused op, with a
   perfect-fusion ceiling of 12 % on sm120 and 2 % on sm86**. We would be paying
   BF16 compute for a 2-12 % prize.
6. **The cheap version of the KV win is dead on arrival.** The GDN-coverage win is
   not a W8A16 property and is also carried by
   `Qwen3.8-27B-SmoothQuant-W8A8-INT8`, already on disk — but that checkpoint was
   **boot-tested on 2026-08-15 and emits `!` from token 0 / token salad**
   (`/spinning/evidence-qwen38/CANDIDATE_VERDICT_2026-08-15.md`). An earlier draft
   of this document recommended it; that recommendation is retracted.
7. **Recommendation: do not adopt lued as the standard model.** Spend a
   ~30-minute microbench first (Marlin-W8A16 vs int8_scaled_mm on sm86/sm120, no
   download, no boot); if it survives, the right long-run answer is to **requant
   the incumbent ourselves with GDN coverage**, which is the only path that takes
   the 4.9 GiB / +156k-token KV win *and* keeps the measured 2x INT8 lane.

## 1. Prior-art gate

Search sets executed:
- `git log --all --grep=` in `/spinning/htsglang-gpu` for `#854`, `w8a16`, `lued` —
  no `#854`, no `lued`. W8A16 appears only in FP8/Marlin lane contexts
  (`cc8e666496`, `df189f015e`, `30592b3207`, `20a52751c9`, `23302643bc`).
- `/spinning/htsglang/docs/dev/` (166 entries, 32 `ANALYSE_*`) — no W8A16-vs-W8A8
  checkpoint analysis, no `#854` file.
- `/spinning/gpu-arb/` (`WINDOW-QUEUE.md`, boot scripts, holder/progress files) —
  no INT8/W8A16 quant window queued.
- Local checkpoint inventory `/spinning/llm_stuff/club-3090/models-cache/`.

**Prior art FOUND — this question has been partly answered before.** Two documents
matter more than any ticket number, and both were missed by a filename-level search:

- **`docs/dev/ANALYSE_319_int8_lane.md:49-57`** already identified this exact
  checkpoint by name and classified it correctly:
  > "`lued/...-INT8-W8A16-MTP` — `format: "pack-quantized"`, no
  > `input_activations` block at all. These are **weight-only INT8 (W8A16)**:
  > activations stay bf16/fp16, weight is dequantized before the matmul. They do
  > not exercise `sgl_kernel.int8_scaled_mm` ... they are architecturally the INT8
  > analogue of `fp8_w8a16`, not of `fp8_native`."
  It was *rejected for #319's thesis* (which was about the native INT8 lane), not
  evaluated as a serving checkpoint.
- **`/spinning/evidence-qwen38/CANDIDATE_VERDICT_2026-08-15.md`** — a candidate
  survey for **this exact model**, nine days old, which already ranked lued:

  | rank | candidate | scheme | lane | size | verdict |
  |---|---|---|---|---|---|
  | 1 | Freaksterz SmoothQuant-W8A8-INT8 | W8A8 per-channel + dynamic per-token, SmoothQuant-smoothed | W8A8 prefill lane | 29.1 GiB | "the one challenger worth an A/B" |
  | 2 | Minachist INT8-AutoRound | weight-only int8, `group_size=-1` (per-channel), sym | marlin | 28.3 GiB | "secondary; trades the prefill lane away" |
  | 3 | **lued INT8-W8A16-MTP** | weight-only, pack-quantized, group 128 | marlin | 29.4 GiB | "already ticketed with the shift; do not duplicate" |
  | — | lokeshe09 INT8 (incumbent) | W8A8 per-channel + dynamic per-token | W8A8 lane | 33.9 GiB | "running today; GDN path entirely bf16" |

  Its size column (29.4 / 33.9 GiB) matches this document's independently
  reconstructed footprints to the decimal, which cross-validates §3.2.

**The decisive prior-art fact — and it kills the obvious recommendation.** That
same document carries an amendment, *"theory verdict FALSIFIED on metal"*: the
rank-1 SmoothQuant candidate was actually boot-tested and **failed** —
> "as shipped (fp16 per its own config) it emits `!` from token 0 on every
> prompt; forced bf16 it emits multilingual token salad."

with the generalized lesson:
> "Runnability of a third-party checkpoint is NOT desk-provable — every future
> candidate verdict must label the runnability leg UNVERIFIED until a
> 1-generation smoke boot passes."

That checkpoint is the `Qwen3.8-27B-SmoothQuant-W8A8-INT8` sitting on our disk.
**It is not a shortcut to the KV win; it is a known-bad artifact.** This closes
open item O2 and invalidates the recommendation an earlier draft of this document
made. The same lesson applies with full force to lued: its "loadable: yes" means
*loads*, not *emits coherent tokens*, and there is no record of it ever being
smoke-booted.

**Absence claim (narrowed):** no prior *measurement* of a W8A16 checkpoint against
our W8A8 incumbent exists — no KLD arm, no throughput arm, no smoke boot. The
"already ticketed with the shift" pointer for lued does not resolve to any
locatable ticket file (searched `/spinning/evidence-qwen38/*.md` incl.
`SHIFT_NOTES.md`, `docs/dev/*.md`, `gpu-arb/WINDOW-QUEUE.md`); "the shift" is an
operational work-shift, not a ticket scheme, so that tracking is informal and did
not persist. This document is therefore the first written evaluation, not a
duplicate.

**One stale prior-art item corrected:** #319 recorded that `int8_scaled_mm` has a
closed SM dispatch sm75-90, so sm120 hard-crashes with no dequant fallback. In the
current tree that is **no longer true in source**:
`sgl-kernel/csrc/gemm/int8_gemm_kernel.cu:774-786` adds an sm120 branch forwarding
to `sm89_dispatch_shape`; the `TORCH_CHECK_NOT_IMPLEMENTED` at `:788` now only
catches sm<75 and sm100-119. Whether the *installed wheel* carries `compute_120a`
cubin is a build-artifact question this desk read cannot settle
(`sgl-kernel/CMakeLists.txt:211,247-251`); the boot gate `require_int8_arm`
(`w8a8_int8.py:99-145`) only checks that the symbol *imports*, not that the arch
has cubin. See §6 fix item F4.

## 2. What the lued checkpoint actually is

`lued/Qwen3.8-27B-INT8-W8A16-MTP`, pinned from its `config.json` and HF API:

| Property | Value |
|---|---|
| `quant_method` | `compressed-tensors` |
| `format` | `pack-quantized` |
| weights | `num_bits: 8`, `strategy: "group"`, `group_size: 128`, `symmetric: true`, `dynamic: false` |
| `input_activations` | **`null`** — this is what makes it W8A16 |
| `kv_cache_scheme` | `null` |
| calibration | **none** — data-free symmetric RTN, `llmcompressor QuantizationModifier` |
| `ignore` | `lm_head`, `re:.*visual.*`, `re:^mtp.*`, `re:.*linear_attn[.]in_proj_a$`, `re:.*linear_attn[.]in_proj_b$` |
| storage | `usedStorage` = 31,636,252,213 B (29.46 GiB) |
| architectures | `Qwen3_5ForConditionalGeneration` (same as ours) |

**Independent proof of quantization coverage.** The HF dtype histogram reports
`I32: 24,326,963,200` logical weight elements. Reconstructing from the model
geometry (hidden 5120, intermediate 17408, 64 layers = 16 full-attention + 48 GDN):

```
MLP      64 x (2*17408*5120 + 5120*17408) = 17,112,760,320
FA       16 x (12288+1024+1024)*5120 + 5120*6144 =  1,677,721,600
GDN dense 48 x (10240*5120 + 6144*5120 + 5120*6144) = 5,536,481,280
                                          total = 24,326,963,200   <- exact match
```

The match is exact to all ten digits **only if the 144 GDN dense projections
(`in_proj_qkv`, `in_proj_z`, `out_proj`) are quantized**. That settles the
coverage question without trusting the model card's prose.

### Delta vs. our checkpoint

Our active model is `/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8`
(confirmed as the live boot target at `/root/bin/start-serving-30030.sh:242`),
`quantization_config`: `format: int-quantized`, weights `strategy: "channel"`,
`num_bits: 8`, symmetric; `input_activations` `strategy: "token"`, `dynamic: true`
(per-token dynamic INT8) — i.e. **W8A8**. Its `ignore` list:

```
["re:.*(vision|visual).*", "lm_head", "re:.*embed_tokens.*",
 "re:.*norm.*", "re:.*conv1d.*", "re:.*linear_attn.*"]
```

| Axis | ours (Qwen3.8-27B-INT8) | lued W8A16 |
|---|---|---|
| activations | INT8, per-token dynamic | **BF16 (none)** |
| weight granularity | per-channel | **group of 128** |
| GDN dense projections (48 layers) | **BF16, not quantized** | INT8 |
| MTP head | **INT8-quantized** | BF16 |
| lm_head / embeddings / vision | BF16 | BF16 (same) |
| quantized GEMMs | 256 (192 MLP + 64 FA) + MTP | 400 (192 + 64 + 144 GDN) |
| calibration | `memoryless_minmax` RTN | RTN, data-free |

Four differences, of which **only the first two are the "W8A16" part**. The GDN
and MTP differences are orthogonal recipe choices that could be made in either
scheme.

## 3. KV-cache cost — the exact numbers

### 3.1 KV geometry (verified from `config.json`, not assumed)

`num_hidden_layers: 64`, `full_attention_interval: 4` -> 16 full-attention layers
carry KV; the 48 `linear_attention` (GDN) layers carry a per-sequence recurrent
state, **not** per-token KV. `num_key_value_heads: 4`, `head_dim: 256`.

```
KV bytes/token = 16 layers x 2 (K,V) x 4 heads x 256 dim x 1 B (fp8_e4m3)
               = 32,768 B = 32.0 KiB/token
```

This confirms the operator desk figure exactly (the desk assumed 8x128; the real
4x256 has the identical product). Conversion constants: **1 GiB = 32,768 tokens**,
**1 GB = 30,518 tokens** (the desk's "~30k tokens/GB" is right).
KV dtype is an independent axis and is unaffected by the switch — fp8 stays fp8,
`kv_cache_scheme: null` in both checkpoints.

### 3.2 In-VRAM weight footprint (the headline number)

Under the actual lane (Marlin `wNa16`, §4) weights stay **packed** in VRAM, so the
in-VRAM footprint equals the on-disk tensor footprint. Three checkpoints,
reconstructed tensor-by-tensor and validated against measured file sizes:

| Component | active INT8 | SmoothQuant W8A8 | lued W8A16 |
|---|---:|---:|---:|
| MLP + FA weights (INT8) | 17,920.0 | 17,920.0 | 17,920.0 |
| GDN dense projections | 10,560.0 *(BF16)* | 5,280.0 | 5,280.0 |
| MTP head | 405.1 *(INT8)* | 810.0 *(BF16)* | 810.0 *(BF16)* |
| embed + lm_head (BF16) | 4,850.0 | 4,850.0 | 4,850.0 |
| vision tower (BF16) | 878.8 | 878.8 | 878.8 |
| GDN gates a/b, norms, conv1d | 50.1 | 50.1 | 50.1 |
| **weight scales** | **5.5** | **7.4** | **362.5** |
| **total (MiB)** | **34,669.5** | **29,797.6** | **30,151.4** |
| total (GiB) | 33.86 | 29.10 | 29.44 |

Validation against measured bytes:
- active: index `total_size` = 36,353,564,128 B = 34,669.5 MiB — **exact**.
- SmoothQuant: index `total_size` = 31,243,691,488 B = 29,796.9 MiB — model off by
  0.7 MiB.
- lued: `usedStorage` 31,636,252,213 B minus ~20 MB non-tensor files ≈ 30,147 MiB
  — model off by ~4 MiB.

All three reconstructions land within ~4 MiB of measurement, so the component
attribution below is trustworthy.

### 3.3 The two axes, separated

**(a) Pure W8A16 tax — the user's objection, isolated.** Compare at *equal*
coverage (lued W8A16 vs. SmoothQuant W8A8, both GDN-quantized, both BF16 MTP):

```
30,151.4 - 29,797.6 = +353.8 MiB   (measured delta ~350 MiB)
```

This is **entirely the scale-granularity term**: group-128 scales
(255.0 MiB MLP + 25.0 FA + 82.5 GDN = 362.5 MiB) vs. per-channel scales (7.4 MiB).
Converted:

```
355 MiB / 32 KiB per token = 11,360 KV tokens
11,360 / 457,000                = 2.5 % of pool
```

**The user is right.** W8A16 at group-128 costs ~2.5 % of the KV pool. The
operator's earlier desk estimate (~420 MiB, ~13k tokens, ~3 %) was close; the
exact figure is 355 MiB / 11.4k tokens / 2.5 %.

**(b) Gross delta vs. what we run today.** Our active checkpoint leaves the 48 GDN
layers in BF16:

```
34,669.5 - 30,151.4 = -4,518.1 MiB  freed
-4,518.1 MiB / 32 KiB = +144,579 KV tokens = +31.6 % of a 457k pool
per card at TP=3: ~1,506 MiB freed
```

So the lued checkpoint is **4.42 GiB smaller** than ours despite the scale tax —
the -5,280 MiB GDN term and +405 MiB MTP term swamp the +355 MiB scale term.

**(c) The decisive comparison the user did not ask for.** The GDN win is not a
W8A16 property. `Qwen3.8-27B-SmoothQuant-W8A8-INT8`, **already on disk**, has the
same GDN coverage in W8A8 form:

```
34,669.5 - 29,797.6 = -4,872.6 MiB = +155,923 KV tokens = +34.1 % of pool
```

i.e. **11.3k tokens MORE KV than lued**, with no lane change and no loss of the
INT8 prefill path.

**This is an existence proof about the recipe, not an available shortcut.** That
specific artifact is falsified on metal (§1: emits `!` from token 0 / token
salad), so it cannot be adopted. What the number proves is that **the entire KV
win is reachable inside W8A8** — it is a consequence of covering the 48 GDN
layers, which our incumbent's producer simply chose not to do, and which nothing
in our stack forbids. That is why §8.3 (requant the incumbent ourselves with GDN
coverage) is the recommendation with the best effort/benefit ratio: it is the only
path that takes the full 4.87 GiB *and* keeps the measured 2x INT8 lane.

### 3.4 Minor terms

- **Dropped per-token activation-quant buffers (W8A16 gains):** the W8A8 path
  allocates an INT8 activation buffer + per-token scale per quantized linear.
  These are transient, sized by `max_num_batched_tokens x hidden`, and are pool-
  allocated rather than reserved — at 1600-8192 batched tokens this is single-digit
  MiB and does not move the KV number.
- **Marlin workspace (W8A16 costs): quantified, negligible.**
  `marlin_make_workspace` (`marlin_utils.py:291-299`) allocates
  `SM_count x max_blocks_per_sm(=1) x 4 B` per linear layer — 680 B on the 5090
  (170 SMs), 272 B on a 3080 (68 SMs). Allocated once per layer at
  `wNa16.py:232`, held on the per-layer scheme instance and never freed (it is the
  kernel's threadblock semaphore array, so that is by design). At ~450 linears:
  **~0.3 MB per rank**, i.e. 0.08 % of the scale tax. Not a factor.
- **Transient load-time spike (W8A16):** `gptq_marlin_repack` (`wNa16.py:250`)
  holds the pre- and post-repack copy of a single layer simultaneously. Per-layer,
  not cumulative — it raises the load-time high-water mark by roughly one MLP
  tensor (~85 MiB unsharded, ~28 MiB at TP=3), not the steady-state footprint.
- **Excluded-layer footprint:** identical in both (embed 2,425 MiB + lm_head
  2,425 MiB + vision 878.8 MiB BF16). Neither checkpoint quantizes them; our
  `vocabint8-*` experiments (#724/#727/#763) are a separate axis and are **not**
  in the live boot path (`start-serving-30030.sh:242` points at plain
  `Qwen3.8-27B-INT8`).
- **GDN recurrent state** is per-sequence, not per-token, and is unaffected.

## 4. Execution lane — proven from code

The decisive question was whether a W8A16 checkpoint engages Marlin natively
(weights stay packed) or falls back to dequant-at-load (BF16 in VRAM, ~2x, which
would be KV-disqualifying).

**Answer: Marlin, natively, on both architectures. There is no dequant fallback
for INT8 anywhere in this tree — the lane is Marlin or a hard error.**

- Dispatch: `_get_scheme_from_parts`,
  `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py:721-817`.
  The wNa16 test runs **before** the activation-quant block.
- Selector `_is_wNa16_group_channel` (`:670-683`) requires
  `input_quant is None` **and** strategy in {CHANNEL, GROUP} **and** not dynamic.
  lued satisfies all three. `:727-730` additionally requires
  `quant_format == pack_quantized` and `num_bits in WNA16_SUPPORTED_BITS = [4,8]`
  (`schemes/compressed_tensors_wNa16.py:55-60`). lued satisfies both.
- Kernel: `apply_gptq_marlin_linear` -> `sglang.jit_kernel.gptq_marlin`
  (`schemes/compressed_tensors_wNa16.py:327`, `:47`). Weights are Marlin-repacked
  at load and **remain packed**; there is no BF16 materialization.
- **Arch:** `CompressedTensorsWNA16.get_min_capability() -> 80`
  (`schemes/compressed_tensors_wNa16.py:93-96`). `query_marlin_supported_quant_types`
  (`marlin_utils.py:84-122`) has floor `< 80 -> []` (`:99-100`) and **no upper
  bound**. Kernel guard is only `__CUDA_ARCH__ < 800`
  (`jit_kernel/csrc/gemm/marlin/gptq_marlin.cuh:37`), and the kernel is
  **JIT-compiled per arch** (`jit_kernel/gptq_marlin.py:18-26`,
  `@cache_once_per_arch`) rather than shipped against a fixed gencode list.
  **sm86 ✓ and sm120 ✓.** 8-bit symmetric is instantiated:
  `COMMON_GET_IF(host::kU8B128)` / `ACT_GET_IF(host::kU8B128)`
  (`gptq_marlin.cuh:412,419`).
- **No INT8 dequant fallback exists**: `_maybe_dequantize_unpackable`
  (`compressed_tensors.py:997-1045`) is NVFP4-only and returns every other scheme
  untouched (`:1026-1027`). The W8A8 INT8 scheme raises `NotImplementedError`
  outright when its symbol is missing
  (`schemes/compressed_tensors_w8a8_int8.py:56-64`). So the failure mode of a
  malformed W8A16 checkpoint is a **loud abort**, never a silent VRAM doubling.

### 4.1 Uneven TP x group_size=128 — already correct, no fix required

Per standing doctrine (Uneven-Verteilung NIE abgeschaltet), a group-size violation
would be a bug to fix, not a reason to route around. **It does not arise:**

- Partitioner: `_quant_block_aligned_units`,
  `python/sglang/srt/layers/linear.py:282-367` (called at `:637`, `:2061`);
  shard math `block_aligned_units` / `tp_partition_size` in
  `distributed/utils.py:954,976,1041-1050`.
- Granularity source: `quant_config.weight_block_size`, `linear.py:312-313`. For
  compressed-tensors that property is `compressed_tensors.py:328-352`, whose
  `_group_size_block` (`:184-204`) takes the lcm of the config group sizes and
  folds in `GPTQ_MARLIN_MIN_THREAD_K = 128` (`marlin_utils.py:57`):
  `lcm(128,128) = 128`, returned as `[128,128]` (equal on both dims, mandatory for
  the coupled gate_up/down pair, `:156-162`).
- Consequence: every rank's shard is a multiple of 128, which simultaneously
  satisfies the group grid, `verify_marlin_supports_shape`
  (`marlin_utils.py:191-208`, `min_thread_k=128`, `min_thread_n=64`) and wNa16's
  own `assert input_size_per_partition % group_size == 0`
  (`schemes/compressed_tensors_wNa16.py:127-129`).
- **Both** parallel directions coarsen, and deliberately so: `ColumnParallelLinear`
  uses `block_idx=0` (`linear.py:637`), `RowParallelLinear` `block_idx=1`
  (`linear.py:2061`). The symmetry is required because gate_up's **output** and
  down_proj's **input** are the same intermediate dimension and must split
  identically — that is the #385 argument, written down at `linear.py:192-217`.
  (An earlier draft of this note claimed column-parallel was untouched because
  groups run along K. That is wrong, and the coupled-dim rule is the reason.)
- Worked example, ratio 12:10:10 (sum 32), via `partition_sizes`
  (`distributed/utils.py:901-942`, largest-remainder over whole units):
  - `down_proj K=17408` = 136 units of 128 -> 51/43/42 units =
    **6528 / 5504 / 5376**, all %128; the `wNa16.py:128` assert passes.
  - `gate_up out=17408`: the 16-element MLP family coarsens to `lcm(16,128)=128`
    -> same 136 units, so gate_up and down_proj agree on the intermediate split.
  - `o_proj` / GDN `out_proj K=6144` = 48 units -> 18/15/15 exactly, no
    remainder -> **2304 / 1920 / 1920**.
  - GDN: `_quant_block_aligned_units(value_dim=6144, num_k_heads, ..., 1)`
    (`models/qwen3_5.py:209-211`) has unit_elems 384, `384 % 128 == 0` -> early
    pass-through, 16 units -> 6/5/5 -> the same 2304/1920/1920.
  - Ignored layers (`in_proj_a`/`in_proj_b`) keep raw units: `linear.py:631-637`
    and `:2055-2061` null the quant_config once the layer resolved to
    `UnquantizedLinearMethod` (hazard comment at `linear.py:624-630`).
  Worst-case load imbalance from the rounding: ~1.2 % on one rank.
- Even a **channel**-strategy W8A16 (`weight_block_size` would be `None`) is
  covered: `linear.py:361-362` folds in `_marlin_uneven_tp_block()` = `lcm(64,128)`
  = 128 when `_marlin_packable_family` holds, and `CompressedTensorsConfig`
  declares `marlin_packable_linear = True` at `compressed_tensors.py:114`. The
  fold is deliberately device-free so all ranks agree (`linear.py:241-258`).

**Ratio-granularity note (a genuine, small con):** W8A16 coarsens uneven-TP shards
to 128, W8A8 to 16 (`w8a8_int8.py:169-195`, and `linear.py:351-358` documents that
INT8-W8A8 deliberately folds only 16 because its path is not Marlin). So W8A16
gives us an **8x coarser ratio grid** for `--rank-tp-ratio`. On the shapes that
matter this costs ~1.2 % of balance in the worst case — worth recording, not a
blocker, and not a fallback.

### 4.2 #384 sgl_kernel dual-distribution trap — W8A16 is the *more* robust lane

- W8A8 INT8 depends on `sgl_kernel.int8_scaled_mm` (an "arm"). Under an armless
  pypi wheel it does not degrade — `CompressedTensorsW8A8Int8.__init__` raises
  `NotImplementedError` (`schemes/compressed_tensors_w8a8_int8.py:56-64`), i.e. a
  `pip -U` takes serving **down**.
- W8A16 goes through `sglang.jit_kernel.gptq_marlin`
  (`schemes/compressed_tensors_wNa16.py:47`), which is **in-tree and JIT-compiled**,
  not an `sgl_kernel` symbol. It **survives** an armless wheel.

This is a real, if secondary, argument in W8A16's favour.

## 5. Speed axis — the actual cost of W8A16

GEMM parameter split (from §2): MLP+FA = 18,790,481,920 params (**77.2 %** of the
quantizable set), GDN dense = 5,536,481,280 (**22.8 %**).

### 5.1 Prefill (compute-bound, large M) — W8A16 loses

Marlin dequantizes to BF16 and computes on **BF16** tensor cores. INT8 tensor
cores are ~2x BF16 on both sm86 (GA102) and sm120. Modelling GEMM time as
`params / throughput`, normalized to an all-BF16 baseline of 1.0:

| checkpoint | INT8-computed share | modelled prefill-GEMM time |
|---|---:|---:|
| W8A8 + GDN coverage (§8.3, to be built) | 100 % | **0.50** |
| active INT8 (GDN in BF16) | 77.2 % | **0.61** |
| lued W8A16 | 0 % | **1.00** |

So lued is **~1.63x slower than our active checkpoint** and **~2.0x slower than a
GDN-covered W8A8** on the linear-layer part of prefill. End-to-end the penalty is
smaller (attention, GDN scan, norms, sampling are unaffected), plausibly
1.3-1.5x — but it is the dominant cost of this switch and it lands on exactly the
phase our agent workload stresses. HiCache/radix prefix caching mitigates it to
the extent prefixes hit; it does not remove it for cold context.

Caveat on the 1.0: Marlin at large M typically reaches ~90 % of dense BF16, and on
sm86/sm120 with BF16 the path logs "consider fp16" and disables atomic-add reduce
(`marlin_utils.py:451-456,494-498`). An FP16 serving profile could recover part of
this — the lued card in fact reports its FP16 arm as the better one.

### 5.2 Decode (memory-bound, M=1..8) — near-wash, both beat us

Per-step weight traffic (total minus embed rows and the unused vision tower):

| checkpoint | traffic (MiB) | vs. active |
|---|---:|---:|
| active INT8 | 31,365.7 | 1.000 |
| W8A8 + GDN coverage (§8.3) | 26,493.1 | **0.845** |
| lued W8A16 | 26,847.6 | **0.856** |

Both GDN-covering checkpoints cut ~15 % of decode weight traffic — again a GDN
effect, not a W8A16 effect. Between them, W8A16 reads 1.3 % more (scales).

Against this, W8A8 pays the per-token activation-quant kernels that W8A16 does not
have. **This is the axis on which W8A16 was supposed to win, and #368 already
settled it — against W8A16.**

The ~25.6 us activation-quant constant is an **eager launch artifact**, and the
graph-mode re-measure *did* happen (commit `5df91f62fb` / `19055521f4`, "#368
graph-mode + sm86 windows: FUSION CLOSED (NO)"), verbatim:
> "the eager quant-dominance verdict was a launch constant and graph replay
> removes it (`int8_quant` 0.0266 -> 0.0012 ms, ~21x; quant share of fused 61% ->
> 11%; 11.06 -> 2.56 ms/token M=1 replay; perfect-fusion ceiling 12% sm120 / 2%
> sm86) ... sm86 verdict gemm-slow not quant-dominant (GEMM 58-99% of fused, 2-5x
> the 5090's; **INT8 still the right lane, 0.50 median vs bf16 under graph**)"

Three consequences, all adverse to W8A16:

1. **The activation-quant cost W8A16 removes is ~11 % of the fused op under
   graphs, not 61 %.** The prize is small, and the measured perfect-fusion ceiling
   (12 % sm120 / 2 % sm86) bounds it: even *perfectly* eliminating activation quant
   buys at most 12 % on the 5090 and 2 % on the 3080s. W8A16 does not get that for
   free — it pays BF16 compute for it.
2. **sm86 at M=1 is GEMM-dominated (58-99 % of the fused op), not
   quant-dominated.** The pure-bandwidth model above therefore understates the risk
   on the two 3080s: if the GEMM, not the weight traffic, is the M=1 bottleneck
   there, a BF16-compute Marlin kernel is exposed at decode too, not only at
   prefill.
3. **INT8 measured 0.50 median vs BF16 under graph on sm86** — i.e. the 2x is real
   and measured on our own cards, not a datasheet claim.

Where Marlin W8A16 lands between "int8 traffic" and "bf16 compute" on sm86 at
M=1..8 is the single remaining unknown of the speed axis — and per the #368
precedent it is answerable by **microbench, without a model download or a full
boot** (see §9, step 0). Quoting the 25.6 us eager figure as if it applied to our
graph-mode serving would be exactly the error the INDIKATOR-GESETZ warns about;
it is recorded here only to retire it.

## 6. Compatibility — named fix items, no accepted fallbacks

For the lued checkpoint specifically, the loader path is clean. The items below are
either satisfied or are loud aborts to fix, never silent degradations:

| # | Item | Status for lued | file:line |
|---|---|---|---|
| C1 | `format` must be `pack-quantized` | ✓ satisfied | `compressed_tensors.py:727-730` |
| C2 | `input_activations: null` | ✓ satisfied | `compressed_tensors.py:673` |
| C3 | symmetric 8-bit (asym `kU8` is **not** instantiated) | ✓ symmetric | `gptq_marlin.cuh:410-425` |
| C4 | group_size in Marlin set, uneven-TP coarsening | ✓ 128, folds to 128 | `linear.py:312-313`, `compressed_tensors.py:184-204` |
| C5 | BF16 MTP head must be named in `ignore:` | ✓ `re:^mtp.*` | `compressed_tensors/utils.py:25-51` |
| C6 | vision tower unquantized | ✓ unconditional | `models/qwen3_vl.py:1246` (`quant_config=None`) |
| C7 | lm_head / embed never quantized | ✓ unconditional | `compressed_tensors.py:311` |

**Fix items (would need work if we requant ourselves, F1-F3; F4 regardless):**

- **F1** — an INT8 weight-only checkpoint serialized as `int-quantized` (rather
  than `pack-quantized`) dies at config parse with a bare `AssertionError`:
  `assert target_scheme_map[target]["weights"].type == QuantizationType.FLOAT`,
  `compressed_tensors.py:443-454`. Any self-made W8A16 must be written as
  `pack-quantized`. **Fix: emit the correct format** (llmcompressor does this by
  default for W8A16); optionally improve the assert into a named error. Size: XS.
- **F2** — the non-pack-quantized `ImportError` at `compressed_tensors.py:739-741`
  reports *"CompressedTensorsW4A16Sparse24 is not supported"*, a misleading
  diagnosis for an INT8 W8A16 config. **Fix: correct the message.** Size: XS.
- **F3** — an **asymmetric** 8-bit W8A16 passes every Python gate
  (`_is_wNa16_group_channel` explicitly admits asym, `:680-683`;
  `WNA16_ZP_SUPPORTED_TYPES_MAP[8] = uint8`, `wNa16.py:59`;
  `CompressedTensorsWNA16` never calls `check_marlin_supported`, only
  `check_marlin_supports_shape`, `wNa16.py:221-228`) and then panics in CUDA at
  first forward (`gptq_marlin.cuh:724-735`). **Fix: call `check_marlin_supported`
  in the wNa16 scheme so asym-8bit is refused at load with a named error.**
  Size: S. Not triggered by lued (symmetric), but it is a live trap for any
  self-made checkpoint.
- **F4** — `require_int8_arm` (`w8a8_int8.py:99-145`) validates only that
  `sgl_kernel.int8_scaled_mm` **imports**, not that the installed wheel carries
  `compute_120a` cubin for the 5090. This is the #384 trap wearing an INT8 hat and
  it affects our **current** W8A8 serving, independent of this decision.
  **Fix: probe-execute a tiny int8 GEMM per arch at the gate.** Size: S.

## 7. Pros / cons

| Axis | lued W8A16 vs. our active W8A8-INT8 |
|---|---|
| **Quality (weights)** | **+** group-128 vs. per-channel scales is strictly finer |
| **Quality (activations)** | **+** removes per-token INT8 activation quant entirely — our one extra loss source |
| **Quality (evidence)** | **-** KLD measured vs. **FP8**, not vs. our INT8; measured on the **HF decompressed path**, not the Marlin serving kernel; no behavioural evals (tool use, JSON, coding, long-context recall) |
| **Quality (coverage)** | **-** quantizes 48 GDN layers we currently keep in BF16 — more of the model is lossy, not less |
| **VRAM / KV (gross)** | **+** -4.42 GiB -> +144.6k KV tokens (+31.6 %) — but this is the GDN term, obtainable in W8A8 |
| **VRAM / KV (like-for-like)** | **-** +355 MiB scale tax -> -11.4k KV tokens (-2.5 %) |
| **Prefill speed** | **--** ~1.63x slower on the GEMM part (Marlin computes BF16, loses the INT8 2x) |
| **Decode speed** | **?** ~15 % less weight traffic vs. active on a pure-bandwidth model (GDN term) — **but #368 measured sm86 M=1 as GEMM-dominated (58-99 % of the fused op), so a BF16-compute kernel is exposed at decode too on the 3080s. Unresolved; step 0 of §9 settles it** |
| **Activation-quant saving** | **-** the prize W8A16 exists to collect is 11 % of the fused op under graphs, ceiling 12 % sm120 / 2 % sm86 (#368 graph-mode) — small, and paid for in BF16 compute |
| **Lane robustness** | **+** in-tree JIT Marlin survives the #384 armless-wheel trap; W8A8 goes down |
| **Arch coverage** | **+** one lane spans sm86 + sm120 cleanly; W8A8 sm120 depends on wheel cubin (F4) |
| **Uneven TP** | **-** 8x coarser shard grid (128 vs 16), ~1.2 % worst-case imbalance |
| **MTP / spec** | **+** BF16 draft head (ours is INT8-quantized) should help acceptance; unmeasured |
| **Operational** | **-** upstream card documents a `>=2 concurrent request` engine crash in the GDN spec-decode path (validated only at `max-num-seqs 1`) and prefix-cache corruption without a specific patch. Both are **vLLM-side** claims and may not transfer to our tree — but they must be actively falsified here, not assumed away |
| **Control** | **-** community checkpoint, data-free RTN, no provenance over the recipe |

## 8. Recommendation

**Do not adopt `lued/Qwen3.8-27B-INT8-W8A16-MTP` as the standard model.** Not
because it is bad — its recipe is sane and our lane serves it natively — but
because the win the user actually asked about (KV cache) is 97 % attributable to
GDN coverage, which we can have **without** paying the ~1.6x prefill penalty, and
the quality claim that motivated the question is not yet evidence about us.

Sequenced, cheapest falsifier first:

0. **Microbench before anything else — no download, no boot, ~30 min desk/GPU-lite.**
   Reuse the #368 harness (which already measures `int8_scaled_mm` and dense BF16
   under CUDA-graph replay on both archs) and add a **Marlin wNa16 8-bit** arm at
   the real shapes: `(17408, 5120)`, `(5120, 17408)`, `(12288, 5120)`,
   `(5120, 6144)`, `(10240, 5120)`, at M = 1, 4, 8 and one prefill M (1600-8192).
   This settles the entire speed axis for the cost of a microbench, because it
   needs no checkpoint at all — random int8 weights suffice. **If Marlin comes out
   worse than 0.75x of the W8A8 lane at decode M, stop here**: no quality gain
   justifies that on a rig whose 3080s are already GEMM-slow (#368: GEMM 58-99 %
   of the fused op at M=1 on sm86). Sample inputs on CPU and move them
   (CUDA-randn cross-arch rule).
1. **Only if step 0 survives — smoke-boot lued before any A/B.**
   One generation, one prompt, coherent tokens. This is not a formality: the
   rank-1 candidate from the same survey passed every desk check and then emitted
   `!` from token 0. Runnability is UNVERIFIED until this passes.
2. **Then, and only then, the quality arm.** Our KLD/quality suite with a **W8A8
   arm in the comparison**, measured **through the Marlin serving path**. That is
   precisely the number the Reddit chart does not contain: it compares against FP8,
   and it measures the HF decompressed path.
3. **The likely right answer regardless of W8A16: requant the incumbent ourselves,
   with GDN coverage, staying W8A8.**
   The 4.87 GiB / +156k-token KV win is a *recipe* property, not a scheme property.
   Our incumbent's producer simply excluded `re:.*linear_attn.*`; nothing in our
   stack requires that. Requanting the same base with GDN dense projections
   included keeps the measured 2x INT8 lane (#368: 0.50 median vs BF16 under graph
   on sm86) **and** takes the whole VRAM win — the only path that gets both. It
   also gives us provenance, which neither community checkpoint does. This should
   be opened as its own ticket independently of the W8A16 question.
4. **If W8A16 does win on quality — do not adopt lued; take the channel-strategy
   route.** Either the already-surveyed `Minachist INT8-AutoRound` (weight-only
   int8, `group_size=-1`, 28.3 GiB, marlin lane — **calibrated AutoRound beats
   lued's data-free RTN on method, and per-channel scales carry no scale tax**), or
   requant ourselves with llmcompressor `scheme: W8A16` + `strategy: channel`. The
   wNa16 lane accepts CHANNEL (`compressed_tensors.py:670-683`), Marlin accepts
   `group_size = -1` (`marlin_utils.py:60`), and uneven-TP still coarsens to 128 via
   the `marlin_packable_linear` fold (`linear.py:361-362`). Scale tax
   362.5 MiB -> ~7.4 MiB, i.e. the user's 11.4k tokens come straight back.
   On present evidence **lued is not even the best W8A16 candidate we know of.**

**Explicitly rejected option:** keeping our W8A8 weights and merely switching off
activation quantization at runtime. **Not possible in this tree, from code**: the
gate is `input_quant_none` at `compressed_tensors.py:673`; a populated
`input_activations` can never reach the wNa16 branch, and there is no `SGLANG_*`
override (the only force/dequant switches are FP8-only: `fp8_utils.py:348,359`,
`fp8.py:502,1237`). Editing `config.json` to drop `input_activations` does not
work either — it trips the `QuantizationType.FLOAT` assert at `:451-454`, and the
weights are not Marlin-packed on disk regardless.

### Effort verdict

| Path | Size | Reason |
|---|---|---|
| (0) Marlin-vs-INT8 microbench | **XS** | no checkpoint, no boot, random int8 weights suffice; reuses the #368 harness; decides the speed axis outright |
| (1) smoke-boot lued | **S** | 31.6 GB download + one short GPU window; loader gates C1-C7 all satisfied, **no coarsening fix needed** (§4.1) |
| (2) lued as a measurement arm (quality + throughput) | **S-M** | (1) plus one full same-boot-floor window with three arms |
| (2') lued as the *standard* model | **M-L** | (2) plus: falsify the concurrency and prefix-cache caveats in our tree (both hit HiCache directly), re-tune the planner for the 8x coarser shard grid, accept or mitigate the ~1.6x prefill regression |
| (3) **self-requant W8A8 with GDN coverage** | **M** | base BF16 download (~54 GB) + one llmcompressor run + validation; takes the entire KV win while keeping the measured 2x INT8 lane; **best effort/benefit ratio here** |
| (4) self-requant W8A16-channel | **M** | as (3) but weight-only; only worth it if (2) shows a real quality gap; F1/F3 become live and should be fixed first |
| ~~switch to on-disk SmoothQuant W8A8~~ | **dead** | falsified on metal 2026-08-15 (emits `!` / token salad) — see §1 |
| F1-F4 loader hardening | **S** total | XS + XS + S + S; F4 is worth doing regardless of this decision |

## 9. Measurement plan (one window, desk-prepared)

Same-boot floor per #375 canon; full feature set per Full-Feature-Default-Serving
(graphs + spec + uneven-DCP/TP), never eager (Full-Perf-Testen). Arms:

**Step 0 first, and it is not a boot:** the Marlin-vs-INT8 microbench of §8.0.
It needs no checkpoint and no serving stack, and it can veto the whole exercise.
Only if it passes does the window below get scheduled.

- **A** — active `Qwen3.8-27B-INT8` (baseline, GDN BF16, W8A8)
- **B** — ~~`Qwen3.8-27B-SmoothQuant-W8A8-INT8`~~ **dropped**: falsified on metal
  2026-08-15 (emits `!` from token 0 / token salad). If a GDN-covered W8A8 arm is
  wanted, it must be produced by us (§8.3), not taken off the disk.
- **C** — `lued/Qwen3.8-27B-INT8-W8A16-MTP` (GDN covered, W8A16)

Per arm, one boot each, same commit (Patchstand vor Last):

1. **VRAM ledger at steady state** — NVML-free per card under load, verify the
   819-1229 MiB corridor (VRAM-Korridor-Regel), and read the **actual** KV pool
   size. This is the direct falsification of §3.3: expect B ≈ A + 156k tokens,
   C ≈ A + 145k, C ≈ B - 11k. If the measured deltas disagree with the table,
   the footprint model is wrong and the rest of this document is suspect.
   Also read the Marlin workspace off the ledger for arm C (open item O1).
2. **Quality** — club-3090 quality suite, plus per-token KLD vs. the BF16 teacher
   **through the serving path**, with A as an arm. This is the number the Reddit
   chart lacks.
3. **Prefill tok/s** at bs=1 and bs=8, cold prefix (prefix caching off for this
   measurement only) — tests the §5.1 prediction of ~1.63x C-vs-A on the GEMM part.
4. **Decode tok/s** at bs=1 and bs=8 — tests §5.2 (~15 % traffic cut for B and C).
   Runs >= 10 s, ms/round per worker with COMPUTE vs WAIT split
   (ms-pro-Runde-als-Messlatte), so the activation-quant question (#368 graph-mode,
   §5.2 / M4) is answered as a by-product: the A-vs-C decode gap *is* the graph-mode
   activation-quant cost.
5. **Spec acceptance** from `meta_info` (never `spec_ema_accept_len`,
   Spec-Acceptance-Messfalle) — C carries a BF16 MTP head, A an INT8 one, so this
   is where that difference shows.
   **Correction to the brief:** there is no "#779 gate threshold 2.0". #779
   (`82b6ec5fdc`, "A gate nobody has watched fail is not a gate") adds a
   *can-fail proof* (`SGLANG_774_NEUTER_VOCAB_COMPANIONS=1`), not a numeric
   threshold. The usable reference band comes from #774 instead: the defect drove
   accept length to **1.02** tokens/verify, healthy is **3.625-4.0**
   (`4e1b940900`). Use that band, and do not cite a 2.0 gate — the several `2.0`
   constants in the test tree belong to unrelated sanity tests for other models.
6. **Concurrency falsification for C** — drive >= 2 concurrent requests through the
   GDN spec-decode path and try to reproduce the card's `cudaErrorIllegalAddress`.
   A clean run here is required before C could ever be a default.
7. **Load** via HiCache/radix session load, not manual fill batteries
   (Lastprobe-via-Session-Load); at least one full flip cycle under load before any
   "stable" claim (WEDGE-RECOVERY abnahme-lehre).

An A-vs-A noise floor first (Benchmark-Harness-Pflichten) — the predicted C-vs-B
decode gap is 1.3 %, which is below the noise floor of most harnesses, so without
it that comparison is unreportable.

## 10. Open items

- **O1** — *closed.* Marlin workspace is `SM_count x 4 B` per linear
  (`marlin_utils.py:291-299`) ≈ 0.3 MB per rank. Negligible; no ledger read needed.
- **O6** — the one arithmetic not closable by desk reading: `q_proj` out 12288 vs
  `o_proj` in 6144. If `o_proj`'s input split is coupled to the q-head split via
  `tp_q_groups` (#116, `linear.py:2065-2068`, `distributed/utils.py:928-932`),
  that group constraint composes with the 128 block. With head_dim 128 both are
  128-multiples so no conflict is expected, but this should be asserted in the
  partitioner corpus rather than reasoned about.
- **O7** — verify that `re:^mtp.*` actually resolves against sglang's layer
  prefixes (draft built with prefix `mtp` at `qwen3_5_mtp.py:147`, fc as `mtp.fc`
  at `:49`). An anchored `^mtp` would miss if the resolved name carried a leading
  `model.`. If it missed, the head would be built quantized while the checkpoint
  ships no `mtp.*.weight_scale` — a loud load error, not silent, but it would
  abort arm C's boot.
- **O2** — *closed.* `Qwen3.8-27B-SmoothQuant-W8A8-INT8` never became the default
  because it was boot-tested on 2026-08-15 and emits `!` from token 0 (fp16 as
  shipped) / multilingual token salad (forced bf16) —
  `/spinning/evidence-qwen38/CANDIDATE_VERDICT_2026-08-15.md`.
- **O3** — *closed.* The #368 graph-mode re-measure **did** run
  (`5df91f62fb`/`19055521f4`): `int8_quant` 0.0266 -> 0.0012 ms (~21x), quant
  share of fused 61 % -> 11 %, perfect-fusion ceiling 12 % sm120 / 2 % sm86, sm86
  GEMM-dominated at 58-99 %. Folded into §5.2. The earlier "re-measure was gated"
  belief was stale.
- **O8** — the tokens/GiB conversion has an unreconciled factor. This document
  derives 32.0 KiB/token from the config; the runbook records that a 7500 MiB
  declared tenant budget cost "~135k KV tokens", which implies ~56.9 KiB/token
  (1.78x). Candidate explanations: spec/draft KV allocated on top, or the 135k
  figure being per-rank against a DCP-sharded pool. **Until reconciled, treat this
  document's token figures as an upper bound on the KV gain** and read the true
  pool size directly off the ledger in measurement 1. The MiB figures are unaffected.
- **O9** — the #725 "NInfer crossover table (activation quant pays from
  T>=10/11/25)" cited in the brief could not be corroborated for INT8: the
  harvested table in `planner/activation_quant_crossover.py` is keyed `FP8_SM120A`
  (FP8, sm120), `FP8_SM86` is empty, and it is corroborating data **not wired into
  any planner decision**. No INT8 crossover table exists. Nothing in this analysis
  rests on it.
- **O4** — whether the installed `sgl_kernel` wheel carries `compute_120a` cubin
  (F4) cannot be settled by reading source; it needs a probe execution.
- **O5** — the `Qwen3.8-27B-INT8-vocabint8-embed` index reports a `total_size`
  byte-identical to plain `Qwen3.8-27B-INT8` while carrying an extra
  `embed_tokens.weight_scale` tensor. Either the index was not regenerated or the
  embedding was not actually re-serialized. Not on the live path
  (`start-serving-30030.sh:242`), so out of scope here, but it makes any
  footprint claim about the `vocabint8-*` family unreliable until checked.
