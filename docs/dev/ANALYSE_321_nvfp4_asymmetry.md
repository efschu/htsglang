# Task #321 -- NVFP4 as a placement lever on the mixed rig

CPU-only investigation (`CUDA_VISIBLE_DEVICES=99`), 2026-07-31. Branch
`analysis/321-nvfp4-asymmetry`, base `a990bc6990`. No card window taken, no
checkpoint downloaded (config/metadata only, via `huggingface_hub`). Written to
be read without any of the surrounding chat context.

**The thesis under test.** NVFP4 runs accelerated on the 5090 (sm_120, native
FP4 tensor cores) and only through a 16-bit dequant detour on the 3080s
(sm_86). An NVFP4 checkpoint would therefore *increase* the pressure to place
compute on the 5090 and shift the split optima 5090-wards -- the mirror image
of the INT8 lane (#319, `docs/dev/ANALYSE_319_int8_lane.md`), where the
advantage sat on the 3080s.

---

## 0. Verdict up front

1. **The direction is right and the magnitude is nil.** Every realistic NVFP4
   variant moves the prefill optimum to the corner (`MLP 136,0,0` -- the whole
   dense-MLP family on the 5090) but lands within **-3.1 % … +13.0 %** of the
   FP8 baseline window, i.e. inside or on the wrong side of the **3.18 %** s=8
   prefill noise floor (§5). The MLP axis was already 94 % exhausted at FP8
   (#299's optimum is `128,4,4`); there was almost nothing left for a format
   lever to harvest.
2. **The pacer flips, and that is the whole story.** Above a 5090 speed-up of
   **phi0 = 1.33** the interior optimum disappears and the 3080s' *non-MLP*
   residual becomes the binding term -- ~200-209 ms of GDN scan, conv1d,
   softmax and norms, work that carries **no weights at all** and that no
   weight format can touch (§5.2).
3. **The premise about the 3080s is factually wrong.** The sm_86 path is not a
   slow dequant detour: it is **gptq_marlin with in-kernel E2M1 dequant**, the
   same kernel family that is already the 3080s' *best* FP8 lane
   (`fp8_marlin` 58.44/59.15 TFLOPS vs `fp8_w8a16` 53.43/53.78). At prefill it
   is at parity with today's lane; at decode it is strictly better because it
   reads 4.5 bits per weight instead of 8 (§4).
4. **The money is on the decode/VRAM axis, and it is symmetric, not
   asymmetric.** A 4-bit checkpoint cuts the per-rank weight read that
   dominates bs=1 decode: modelled **10.19 -> 7.43 ms/token (-27 %)**, against
   a measured 10.93 ms/token anchor and a 2.72 % ms/Verify noise floor. Context
   rises **1.48x-1.57x** (§6). Both effects help the 3080s at least as much as
   the 5090.
5. **A format scissors blocks the best variant.** The NVFP4 flavour that is
   VRAM-positive, decode-positive and prefill-neutral (all-Linear NVFP4,
   compressed-tensors) **cannot boot on sm_86** (`get_min_capability() == 100`,
   no fallback branch). The flavour that boots everywhere
   (`nvidia/Qwen3.6-27B-NVFP4`, `W4A16_NVFP4`) drags the **5090** off
   `fp8_native` (568 TFLOPS) onto Marlin (216 TFLOPS) and costs **+13 %
   prefill**. The flavour with native FP4 compute on the 5090
   (`mmangkad`, `AEON-*-MTP`) leaves attention/GDN in **bf16** and is a **VRAM
   regression** (§2, §6).
6. **#291 verdict**: the Marlin W4A16 path on sm_86 is *not* a compatibility
   minimum to be tolerated -- it is the good lane, and #291's slice **S3**
   (teach `CompressedTensorsW4A4Fp4` the Marlin branch, `min_capability`
   100 -> 80) is promoted from "nice to have" to **the single highest-value
   item in the whole task**, because it is the only thing standing between this
   rig and variant V4 (§9).

---

## 1. Anchors this document is built on

Everything below is derived from measurements already in the tree. Nothing is
assumed that could have been measured.

### 1.1 Lane table (probe shape `M,K,N = 2048, 5120, 17408`)

Cached v3 profile `/root/.cache/sglang/hw_profile-9a5e9b49b7dc.json`, identical
to `/spinning/gpu-battery-results/2026-07-30_lane_reprobe/hw_profile_after.json`;
original in `docs/dev/INTEGRATION_R3_VALIDATION.md:12092-12134` (#298b) and
`:13428-13440` (post-#304 reprobe).

| card | dense bf16 | `fp8_native` | `fp8_marlin` | `fp8_w8a16` | dispatch lane today |
|---|---:|---:|---:|---:|---|
| RTX 5090 (sm_120) | 232.97 | **568.48** | 216.34 | 181.43 | `fp8_native` |
| RTX 3080 #1 (sm_86) | 62.72 | -- (no fp8 tensor path) | **58.44** | 53.43 | `fp8_marlin` |
| RTX 3080 #2 (sm_86) | 62.98 | -- | **59.15** | 53.78 | `fp8_marlin` |

Format-pure spread **9.3-9.7 : 1 : 1**. Marlin is **0.93-0.96x** of the same
card's dense bf16 on sm_86 and **1.09-1.13x** of `fp8_w8a16`
(`INTEGRATION_R3_VALIDATION.md:13443-13466`).

Bandwidth (#213, `/root/.cache/sglang/card_probe-20ae5edfc9b2.json`): streaming
read **1660.4 / 717.0 / 717.1 GB/s**, decode-shaped GEMV **1529.7 / 717.8 /
717.8 GB/s**. VRAM totals **32607 / 20480 / 20480 MiB**. No GPUDirect P2P
(`peer_access: false` on every pair, host-staged 4.3-6.9 GB/s).

### 1.2 Prefill cost model (#299 §1.2, refit here)

`comp_r = a_r * m_r + b_r * g_r`, per rank, for one 2048-token prefill chunk at
8 sessions, from `/spinning/gpu-battery-results/2026-07-30_phasen_optima/s15_phasen_optima/wait/`:

```
a = [ 193.2, 1202.8, 1282.4 ] ms   100 % of the dense-MLP family on rank r
b = [ 128.0,  835.6,  820.0 ] ms   100 % of the attention+GDN family on rank r
window = max_r(comp_r) + collective_floor ,  floor = 975 … 1000 ms
```

TP0 = 5090, TP1/TP2 = 3080. The floor is **64-76 % of the whole window** and is
transport, not skew (#252/#264). Baseline optimum: `max-comp 245.3 ms` at MLP
units `[128, 4, 4]`, window **1245 ms = 1645 tok/s**.

**New in this document -- the GEMM / non-GEMM decomposition of `b`.** The
attention+GDN family's projection weights are 7.240e9 params
(GDN 5.562e9 + full-attn 1.678e9), i.e. `2 * 7.240e9 * 2048 = 2.966e13` FLOP.
Divided by each rank's *implied achieved rate* from `a` (`362.8 / 58.3 / 54.7`
TFLOPS -- note how exactly rank 1's 58.3 reproduces the measured `fp8_marlin`
58.44, confirming the #299 data was taken on the Marlin lane):

```
b_GEMM     = [  81.7, 508.9, 542.5 ] ms   scales with the weight format
b_non-GEMM = [  46.3, 326.7, 277.5 ] ms   GDN scan, conv1d, softmax, norms --
                                           WEIGHT-FREE, format-invariant
```

That second row is the finding that decides §5.

### 1.3 Noise floors in force

s=1 prefill **2.71 %**, s=8 prefill **3.18 %**
(`INTEGRATION_R3_VALIDATION.md:11305`, restated `:11510`); ms/Verify **2.72 %**,
tok/s (Tick) **7.53 %** (`:11794-11816`, six-sample A-vs-A at bs=16, #294).
Nothing below these is reported as a finding.

---

## 2. Checkpoint landscape -- and the format scissors

HF Hub search (`huggingface_hub.HfApi.list_models`,
`/spinning/shvllm/.venv/bin/python`; queries `Qwen3.6-27B NVFP4`, `… FP4`,
`… W4A4`, `… modelopt`) returns **40+ NVFP4 checkpoints for Qwen3.6-27B**.
Availability is emphatically not the constraint. What matters is that they fall
into four structurally different classes, and the class decides everything.

Weight-format census taken from `HfApi.model_info(expand=["safetensors"])`
dtype counts (packed FP4 lives in `U8` at 2 values/byte; the per-16 E4M3 block
scale appears either as typed `F8_E4M3` or, for ModelOpt, folded into the same
`U8` blob -- both cases were disambiguated arithmetically against the model's
17.113e9 MLP / 7.240e9 attention+GDN / 2.543e9 embed+lm_head parameter counts).

| class | representative | `quant_method` | what is 4-bit | what the rest is | 3080 boots? | 5090 lane |
|---|---|---|---|---|---|---|
| **V1** MIXED_PRECISION, weight-only | `nvidia/Qwen3.6-27B-NVFP4` (2.14 M dl) | `modelopt`, `quant_algo: MIXED_PRECISION` | MLP + lm_head, `W4A16_NVFP4` | attn/GDN **FP8**, embed bf16, KV fp8 | **yes**, Marlin | **Marlin too** (see below) |
| **V2** W4A4, MLP only | `mmangkad/Qwen3.6-27B-NVFP4` (17.9 k dl), `AEON-7/…-Multimodal-NVFP4-MTP` (26.2 k dl), `PassingByPixels/…-NVFP4-MTP` | `modelopt`, `quant_algo: NVFP4`, `group_size 16` | MLP only (`exclude_modules` lists every `linear_attn*`/`self_attn*`) | attn/GDN **bf16**, embed/lm_head bf16 | **yes**, Marlin (as W4A16) | native FP4 |
| **V4** all-Linear W4A4 | `ocicek/Qwen3.6-27B-NVFP4`, `AEON-7/…-NVFP4-MTP-XS`, `llmfan46/…-NVFP4`, `sakamakismile/Qwen3.6-27B-NVFP4` (192 k dl) | **compressed-tensors**, `format: nvfp4-pack-quantized`, `tensor_group`/16 | MLP + GDN + full-attn (24.35e9 params exactly) | embed/lm_head/MTP bf16 | **NO** -- `NotImplementedError` | native FP4 |
| **V5** mixed nvfp4+bf16 per layer | `rdtand/…-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` (224 k dl), `unsloth/Qwen3.6-27B-NVFP4` (2.9 M dl, nvfp4 MLP + **W8A8-channel** fp8 rest) | compressed-tensors, `format: mixed-precision` | per-layer selection | bf16 / fp8-channel | **NO** | native FP4 |

Two further observations worth carrying forward:

* **The MTP drafter is handled differently by every publisher.**
  `ocicek/Qwen3.6-27B-NVFP4` ships `model-mtp-bf16.safetensors` as a separate
  shard (drafter stays bf16 -- the safe pattern);
  `AEON-7/…-Multimodal-NVFP4-MTP` lists `mtp.fc`, `mtp.layers.0.mlp.*`,
  `mtp.layers.0.self_attn.*` explicitly in `exclude_modules`;
  `llmfan46/…-NVFP4-MLP-Only` names no `mtp.*` at all, so its MTP MLP *is*
  quantised. Per the #318 lesson (draft-namespace verification), the drafter's
  target/ignore namespace has to be read out of `hf_quant_config.json` per
  checkpoint -- it is not implied by the `-MTP` suffix in the repo name.
* `unsloth/Qwen3.6-27B-NVFP4`'s non-FP4 half is **W8A8 channel-strategy**
  compressed-tensors, which is precisely the shape #319 §2d identified as
  returning `weight_block_size == None` and silently disabling the uneven-TP
  coarsening. Two independent alignment traps in one checkpoint.

### 2.1 The scissors, stated plainly

```
              boots on sm_86    native FP4 on 5090   VRAM vs FP8    prefill vs FP8
V1 nvidia          yes                 NO             -6457 MiB       +13.0 %
V2 mmangkad        yes                 yes            +2189 MiB      -2.3 … +1.0 %
V4 all-Linear      NO                  yes            -7736 MiB      -3.1 … -3.4 %
```

No currently published Qwen3.6-27B NVFP4 checkpoint is in the top-right corner
of all three columns. V4 is one code change away (§9).

---

## 3. Fork stand: what exists, per architecture

### 3.1 There are no FP4 kernels in `sgl-kernel/`

`sgl-kernel/csrc/` contains zero FP4 sources; `csrc/common_extension.cc:117-129`
registers only `int8_scaled_mm` / `fp8_scaled_mm` / `fp8_blockwise_scaled_mm`.
The arch list in `sgl-kernel/CMakeLists.txt:197-236` (`sm_80, sm_89, sm_90a,
sm_100a, sm_120a, …`, plus the fork's `SGL_KERNEL_LIMIT_CUDA_ARCHS` escape at
`:246-264`, task #66) is therefore **irrelevant to FP4**. All NVFP4 CUDA lives
in the JIT tree `python/sglang/jit_kernel/csrc/gemm/nvfp4/` and
`…/moe/nvfp4_blockwise_moe.cuh`.

### 3.2 sm_120 *is* covered by the JIT CUTLASS kernels

`python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_kernels.cuh:122-145`:

```cpp
const int sm_version = getSMVersion(A.device().device_id);
if (sm_version >= 120) {
  ... cutlass_fp4_bf16_gemm_dispatch_sm120(...);   // ArchTag = cutlass::arch::Sm120
} else {
  ... cutlassFp4GemmDispatchSm100<cutlass::bfloat16_t>(...);
}
```

guarded by `#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)`
(`nvfp4_scaled_mm_sm120.cuh:20,228`). The grouped-MoE sibling
(`nvfp4_blockwise_moe.cuh:758-838`) is a *closed* dispatch -- `{100,103}` or
`>= 120`, everything else `RuntimeCheck(false, "Unsupported SM version")` --
and its sm120 arm is **bf16-output only**.

The whole JIT tree is fenced at the Python level
(`python/sglang/jit_kernel/nvfp4.py:38-51`):

```python
major, minor = torch.cuda.get_device_capability()
if major < 10:
    raise RuntimeError(f"NVFP4 JIT kernels require compute capability >= 10.0, got {major}.{minor}.")
return override_jit_cuda_arch(major, minor, suffix="a")   # -> sm_120a
```

sm_86 never reaches nvcc. sm_120 compiles `sm_120a`. Kernel alignment
requirements: `k % 32 == 0`, `n % 32 == 0`, scale tensors rounded to
`(round_up(m,128), round_up(k/16,4))` (`nvfp4_scaled_mm_kernels.cuh:79-120`);
the quant kernel additionally needs `n % 16 == 0` (`nvfp4_quant_kernels.cuh:224`).

### 3.3 sm_86 already boots ModelOpt NVFP4 -- through Marlin

`python/sglang/srt/layers/quantization/fp4_utils.py:149-162`:

```python
if backend == "auto":
    if is_sm100_supported():                                     # major == 10
        backend = "flashinfer_cutedsl"
    elif is_cuda() and (10, 0) > get_device_capability() >= (8, 0):
        backend = "marlin"
    else:
        backend = "flashinfer_cutlass"
```

Called **per scheduler process = per rank**
(`python/sglang/srt/managers/scheduler.py:862`), so a mixed-arch rig already
resolves a *different* backend per rank without any new code. On this rig:

| rank | card | `auto` resolves to | arithmetic actually performed |
|---|---|---|---|
| TP0 | 5090 (sm_120) | `flashinfer_cutlass` -- **not** the fork's own JIT sm120a kernel, because `is_sm100_supported()` is false (major 12 ∉ [10]) and `(10,0) > (12,0)` is false | W4A4: activations quantised to E2M1 |
| TP1/TP2 | 3080 (sm_86) | `marlin` | W4**A16**: activations stay bf16, weight dequantised in-kernel |

The fork's own sm120a CUTLASS kernel is reachable only via the explicit
`--fp4-gemm-backend cutlass` (`server_args.py:2548-2552`,
`modelopt_quant.py:148`). **Worth flagging as a defect in its own right:**
`auto` on sm_120 routes to an external dependency and silently falls back to
`cutlass_fp4_gemm(...)` at `modelopt_quant.py:163` only when flashinfer is
absent -- the fork's kernel is never *chosen*, only *defaulted into*.

Marlin path details: `marlin_utils_fp4.py` (`apply_fp4_marlin_linear:77`,
`prepare_nvfp4_layer_for_marlin:127`, requires `group_size == 16`), calling
`gptq_marlin_gemm(..., b_q_type=scalar_types.float4_e2m1f, ...)`; in-kernel
E2M1 dequant at `jit_kernel/csrc/gemm/marlin/dequant.h:358-425`
(`MASK = 0x70007000`, pure bit-twiddling, four half2/bf162 specialisations).
Marlin's own floor is `device_capability >= 80` (`marlin_utils.py:99-100`).

**`ModelOptNvFp4A16LinearMethod` (`modelopt_quant.py:1737`) is unconditionally
Marlin on every architecture** -- `process_weights_after_loading` calls
`prepare_nvfp4_layer_for_marlin(layer)` at `:1842` and `apply` returns
`apply_fp4_marlin_linear(...)` at `:1850`, with no capability test anywhere in
the class. This is what makes V1 a 5090 pessimisation (§4.2).

### 3.4 compressed-tensors NVFP4 hard-refuses sm_86

`compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:37-38` returns
`get_min_capability() == 100`, and unlike the FP8 case two blocks below it
(which degrades to `CompressedTensorsW8A16Fp8`), the NVFP4 branch has **no
fallback**:

```python
# compressed_tensors.py:683-692
if self._is_fp4a4_nvfp4(weight_quant, input_quant):
    if self._check_scheme_supported(CompressedTensorsW4A4Fp4.get_min_capability(), error=False):
        return CompressedTensorsW4A4Fp4()
    else:
        raise NotImplementedError("Current platform does not support w4a4 nvfp4 quantization.")
```

`_check_scheme_supported` compares `DeviceCapability(*torch.cuda.get_device_capability()).to_int()`:
sm_120 -> 120 >= 100 admitted, sm_86 -> 86 < 100 refused. The MoE sibling
throws even earlier (`compressed_tensors_w4a4_nvfp4_moe.py:42-43`).

### 3.5 Two latent gaps that would bite before any perf question

* **`ModelOptFp4Config` exposes no `weight_block_size`.** Confirmed by grep:
  zero occurrences in `modelopt_quant.py`. `_quant_block_aligned_units`
  (`python/sglang/srt/layers/linear.py:156-188`, invoked at `:451-453`) reads
  `getattr(quant_config, "weight_block_size", None)` and short-circuits at
  `if not block: return units`, so **the fork's uneven-TP shard coarsening
  never fires for exactly the checkpoint family that does boot on sm_86.**
  With an uneven `--rank-tp-ratio` / `--rank-mlp-ratio` this lands in
  `verify_marlin_supports_shape` (`marlin_utils.py:191-205`,
  `MIN_THREAD_N = 64`, `MIN_THREAD_K = 128`) mid-first-forward. Same bug class
  as #289/#300 and #319 §2d.
  The compressed-tensors path, by contrast, is already correct: its
  `weight_block_size` property (`compressed_tensors.py:277-291`) falls through
  to `_group_size_block` (`:133-199`), which does
  `g = lcm(group_sizes…); g = lcm(g, GPTQ_MARLIN_MIN_THREAD_K)` and returns
  **`[128, 128]`** for an NVFP4 scheme -- a multiple of both the group 16 and
  the kernels' 32. **Fix for ModelOpt: return the same `[128, 128]`.** Five
  lines, and a hard prerequisite for any uneven-TP NVFP4 boot.
* **Expert-offload silently mis-installs on NVFP4.**
  `presplit_expert_offload_after_repack` is called from `fp8.py:2295`,
  `gptq_moe.py:139,413`, `awq_moe.py:172` -- **never** from
  `modelopt_quant.py` / `nvfp4_online.py` / `compressed_tensors_w4a4_nvfp4_moe.py`.
  `EXPERT_TENSOR_ATTRS` (`layers/moe/expert_offload.py:713-741`) lists no NVFP4
  tensor names (`w13_weight_scale_2`, `w13_blockscale_swizzled`, `w13_alphas`, …),
  and the fail-fast guard `assert_expert_offload_quant_supported` (`:505`) works
  by *exclusion list* (`_OFFLOAD_UNSUPPORTED_QUANT_METHOD_NAMES` at `:498-502`,
  three GGUF/WNA16 names), so `ModelOptNvFp4FusedMoEMethod` **passes the guard**
  and installs an offload with no offload half. Not on Qwen3.6-27B's path
  (dense model), but it is the exact hazard #268 was built to catch and NVFP4
  walks straight through it.

### 3.6 The lane table does not know about FP4 at all

`_FORMAT_LANES` (`python/sglang/srt/uneven_perf.py:714-717`) has exactly two
keys:

```python
_FORMAT_LANES: Dict[str, Tuple[str, ...]] = {
    "bf16": (LANE_BF16,),
    "fp8":  (LANE_FP8_NATIVE, LANE_FP8_MARLIN, LANE_FP8_W8A16),
}
```

`checkpoint_compute_format` (`:1733-1766`) would classify a ModelOpt NVFP4
checkpoint as `("modelopt", …)` and a compressed-tensors one as
`("compressed-tensors", …)`; neither has a `_FORMAT_LANES` entry, so
`rank_gemm_scores` takes the **warned dense-bf16 fallback** (`:1792-1809`) and
the planner scores all three cards on bf16 -- **3.79 : 1 instead of the real
9.3-9.7 : 1**. This is the same distortion #296 measured for the AWQ path.
Design for closing it in §8.

---

## 4. The throughput asymmetry, quantified

### 4.1 The 3080 side: parity at prefill, a real win at decode

The user's premise -- "sm_86 only via a 16-bit dequant detour, very slow" --
does not survive contact with the measured lane table. The sm_86 NVFP4 path is
`gptq_marlin_gemm` with `b_q_type = float4_e2m1f`, structurally identical to the
`fp8_marlin` lane that the same cards already dispatch to today. Both dequantise
in-kernel and run the multiply on bf16 tensor cores; the only difference is the
number of bytes fetched per weight and the shift/mask sequence.

| what | at prefill (M=2048, compute-bound) | at decode (M=1, bandwidth-bound) |
|---|---|---|
| `fp8_marlin`, measured | **58.44 / 59.15 TFLOPS** (0.93-0.96x of the same card's dense bf16) | 8 bits/weight of traffic |
| `nvfp4_marlin`, expected | same bf16-tensor-core ceiling; +0-5 % from halved weight traffic -> **58-62 TFLOPS** | **4.5 bits/weight** -- 1.78x less traffic |

So on the 3080s, **phi1 = phi2 ≈ 1.00 ± 0.05 at prefill and ≈ 1.7 at decode**.
This is the opposite sign from the thesis at prefill and strongly positive at
decode. The corroborating prior is in the tree: the fork already measured that
the *bad* 3080 lane costs `-69.9 %` at decode against `-8.1 %` at prefill
(`python/sglang/srt/layers/quantization/fp8_utils.py:356-357`, restated at
`uneven_perf.py:884-885`) -- decode on these cards is a bandwidth question, not
a kernel-quality question.

### 4.2 The 5090 side: two possible lanes, 5x apart

| variant | lane on sm_120 | rate |
|---|---|---|
| **W4A4** NVFP4 (`quant_algo: NVFP4`) | flashinfer cutlass / fork JIT `sm_120a` CUTLASS block-scaled GEMM | **phi0 x 568.48 TFLOPS**, phi0 unmeasured |
| **W4A16** NVFP4 (`quant_algo: W4A16_NVFP4`) | Marlin, unconditionally (`modelopt_quant.py:1737-1850`) | **216-230 TFLOPS = 0.38-0.40x** of today's `fp8_native` |

**phi0 is the one number this analysis cannot supply and must not invent.**
Bounding it honestly: Blackwell's 5th-gen tensor cores run dense FP4 at 2x dense
FP8, so the ceiling is `2 x 568.48 ≈ 1137 TFLOPS` -> `phi0 = 2.0`. Against that,
the sm_120 block-scaled CUTLASS path is younger than the FP8 path and pays a
group-16 scale-swizzle epilogue that the per-tensor FP8 path does not. A first
measurement in the **1.3-1.7** band would be unsurprising. **Band used
throughout: phi0 ∈ [1.3, 2.0].**

The conclusion turns out to be insensitive to phi0 over that whole band (§5.2) --
which is why measuring it is the *cheap* next step and not a blocker.

---

## 5. Split consequence: re-running the #299 machinery with format factors

Method: the calibrated model of §1.2, with each rank's family cost divided by
that rank's format factor, then `min max_r comp_r` over the simplex (exact
water-filling; corner solutions handled). Collective floor held at 1000 ms --
NVFP4 changes nothing about it, because the all-reduce operand is always full
`hidden_size` in the **activation** dtype, which stays bf16 in every variant.

### 5.1 The four variants

Baseline (FP8, today): `max-comp 245.3 ms`, MLP units `[128, 4, 4]`,
window **1245 ms**, 1645 tok/s.

| variant | MLP factor per rank | attn/GDN factor | fixed residual `n` | opt max-comp | MLP units | pacer | window | vs FP8 |
|---|---|---|---|---:|---|---|---:|---:|
| **V1** nvidia MIXED_PRECISION | `[0.38, 1.00, 1.00]` | `[1,1,1]` (stays fp8) | `[64.0, 208.9, 205.0]` | 407.6 | `[92, 22, 21]` | rank 0 | 1408 | **+13.0 %** |
| **V2** mmangkad / AEON-MTP, phi0 = 1.3 | `[1.30, 1.02, 1.02]` | `[0.41, 1.07, 1.06]` (attn/GDN drop to bf16!) | `[122.9, 200.2, 196.8]` | 257.2 | `[123, 7, 7]` | rank 1 | 1257 | +1.0 % |
| **V2**, phi0 = 1.6 | | | | 236.1 | `[128, 4, 4]` | rank 0 | 1236 | -0.7 % |
| **V2**, phi0 = 2.0 | | | | 216.6 | `[132, 2, 2]` | rank 1 | 1217 | -2.3 % |
| **V3** (hypothetical: W4A4 MLP **+ fp8 attn/GDN**) phi0 = 1.3 | `[1.30, 1.02, 1.02]` | `[1,1,1]` | `[64.0, 208.9, 205.0]` | 211.5 | `[135, 0, 1]` | rank 0 | 1212 | -2.7 % |
| **V3**, phi0 >= 1.6 | | | | **208.9** | `[136, 0, 0]` | **rank 1** | 1209 | **-2.9 %** |
| **V4** all-Linear W4A4, phi0 = 1.3 | `[1.30, 1.02, 1.02]` | `[1.30, 1.02, 1.02]` | `[54.6, 206.4, 202.3]` | 206.4 | `[136, 0, 0]` | rank 1 | 1206 | -3.1 % |
| **V4**, phi0 = 2.0 | | | | 206.4 | `[136, 0, 0]` | rank 1 | 1206 | -3.1 % |

**Every entry is inside the 3.18 % s=8 prefill floor except V1, which is a
+13.0 % regression well outside it.**

Two structural facts drive the table:

* **V1 (`nvidia/Qwen3.6-27B-NVFP4`) is a 5090 pessimisation.** Because
  `W4A16_NVFP4` is unconditionally Marlin, the 5090's dense-MLP GEMM falls from
  568.48 to ~216 TFLOPS -- a 2.6x slowdown on the rank that carries 46 % of the
  MLP family. The optimum responds by pushing MLP *back onto the 3080s*
  (`[92, 22, 21]`, the opposite of the thesis' direction) and the window still
  grows 13 %. The official NVIDIA checkpoint is, in this fork on this rig, the
  worst of the four.
* **V2 (the actually-available W4A4 checkpoints) de-quantises attention and GDN
  to bf16**, which costs the 5090 the `568.48 -> 232.97` factor on 2.966e13 FLOP
  of projection work -- `b_GEMM[0]` rises 81.7 -> 199.6 ms and rank 0's fixed
  residual nearly doubles (64.0 -> 122.9 ms). That single side effect consumes
  most of what the FP4 MLP lane gains. (It does help the 3080s slightly:
  bf16 62.72 > `fp8_marlin` 58.44, so their residual *falls* to ~200/197 ms.)

### 5.2 The tipping point, and why it is the real answer

The brief asks how extreme the vector may become before the slowest-rank clock
flips. Closed form, from the model: the MLP corner `[136, 0, 0]` binds as soon as

```
a_0 / phi0 + n_0  <=  max(n_1, n_2)
193.2  / phi0 + 64.0  <=  208.9        ->        phi0 >= 1.333
```

**Above phi0 = 1.33 the interior optimum ceases to exist.** There is no
"how far can we push it" question left: the answer is *all the way*, and the
binding term becomes `max(n_1, n_2)` -- the 3080s' attention+GDN residual.

That residual is **200-209 ms** and it is **326.7 / 277.5 ms of weight-free
work** plus a projection GEMM (§1.2). No weight format can move the weight-free
part. Consequently:

```
absolute floor of max-comp under ANY 4-bit weight format:  ~200 ms
baseline max-comp today:                                    245.3 ms
maximum recoverable:                                        ~45 ms of a 1245 ms window = 3.6 %
```

**3.6 % is the theoretical ceiling of the entire "NVFP4 as a placement lever"
thesis on this rig, against a 3.18 % noise floor.** That number is independent
of phi0, of which checkpoint is chosen, and of how clever the split is.

### 5.3 What is left for the 3080s

In every variant at the optimum the 3080s carry **0-7 of 136 MLP units**. Their
role reduces to exactly three things:

1. **KV / DCP token carriers.** This is where their 2 x 20 GiB actually pays.
2. **Attention + GDN scan carriers** -- not by choice but because the state pool
   moves with the GDN units at ~4.7 MiB/req/unit (`uneven_perf.py:31-33`,
   `DESIGN_201:1198-1200`). §7 revisits this: NVFP4 does relax the VRAM half of
   that objection.
3. **Bandwidth-decode contributors** -- and here they benefit from 4-bit weights
   in exact proportion to the 5090 (§6).

The #299 connection the brief asks about holds and strengthens: with the MLP
vector pinned at the corner `136,0,0`, the capacity-matched KV vector is no
longer `2,11,10` but essentially "everything the 3080s have", because rank 0's
weight footprint at 4 bits leaves it far more KV headroom than at fp8 (§7.1).
The `--rank-kv-ratio capacity` recommendation from #299 §8.2 becomes *more*
important under NVFP4, not less.

### 5.4 Both families free (the #299 attention/GDN vector, under NVFP4)

For completeness, allowing an independent attention+GDN family vector on top
(the feature #299 rejected):

| | max-comp | MLP units | attn/GDN shares | window | vs FP8 |
|---|---:|---|---|---:|---:|
| FP8 baseline, both free | 243.9 | `[108, 28, 0]` | `[0.702, 0.000, 0.297]` | 1244 | -0.1 % |
| V4, phi0 = 1.3 | 203.4 | `[136, 0, 0]` | `[0.502, 0.246, 0.251]` | 1203 | -3.4 % |
| V4, phi0 = 1.6 | 176.1 | `[136, 0, 0]` | `[0.569, 0.213, 0.218]` | 1176 | **-5.6 %** |
| V4, phi0 = 2.0 | 151.4 | `[136, 0, 0]` | `[0.629, 0.183, 0.187]` | 1151 | **-7.5 %** |

This is the one place where the thesis produces a number above the floor -- and
only under V4 (which does not boot on sm_86 today) *and* only with an
attention/GDN family vector that does not exist (#299 §3.1: the divergence is
the bare `partition_sizes(...)` call at
`python/sglang/srt/configs/model_config.py:1313`). Note the direction is mild:
the optimum wants the 5090 at 50-63 % of attention/GDN, i.e. barely above
today's 50 %. #299's verdict ("register as discarded; new reason required to
retry") named the retry condition as *"a card pair where one side lacks fp8 for
GEMM but matches on bandwidth"* -- V4 is not that condition, and -5.6 % at
phi0=1.6 under an unbootable checkpoint is not enough to reopen it. Recorded
here so the next reader does not have to recompute it.

---

## 6. Where the money actually is: VRAM, context, decode

Weight budget from the model geometry (`hidden 5120`, 64 layers = 48
`linear_attention` + 16 `full_attention`, `intermediate_size 17408`, 24 q heads /
4 kv heads / `head_dim 256`, `linear_num_key_heads 16` / `value 48`, vocab
248320, untied lm_head; per #299 §4.1 at fp8: MLP 16320, GDN 5304, full-attn
1600, embed 1213, lm_head 1212 MiB). NVFP4 = **4.5 effective bits/weight**
(4 bits + one E4M3 scale per 16), i.e. 0.5625 B/param.

Per-rank shares use the anchor split (MLP `[0.463, 0.272, 0.265]`, attn/GDN
`[0.5, 0.25, 0.25]`, vocab even). Decode time = per-rank weight bytes /
decode-shaped GEMV bandwidth (`1529.7 / 717.8 / 717.8 GB/s`).

| variant | total MiB | vs FP8 | per-rank weights MiB | decode ms/token | max |
|---|---:|---:|---|---|---:|
| **FP8 baseline (today)** | 25649 | -- | `[11820, 6975, 6854]` | `[8.10, 10.19, 10.01]` | **10.19** |
| **V1** nvidia MIXED_PRECISION | 19192 | **-6457** | `[8740, 5260, 5192]` | `[5.99, 7.68, 7.58]` | **7.68** |
| **V2** mmangkad / AEON-MTP | 27838 | **+2189** | `[12773, 7567, 7499]` | `[8.76, 11.05, 10.95]` | 11.05 |
| **V4** all-Linear NVFP4 | 17914 | **-7736** | `[7811, 5085, 5017]` | `[5.35, 7.43, 7.33]` | **7.43** |

**Model validation.** The FP8 row predicts 10.19 ms/token as the bs=1 decode
step; the rig's own measured anchor on the good FP8 lane is **10.93 ms/token**
(`fp8_utils.py:356`). 7 % optimistic -- decode at bs=1 on this rig really is
almost pure weight-read bandwidth, which is what makes the other rows
trustworthy as *differences*.

* **V4: -27 % decode step time. V1: -25 %.** Against a **2.72 %** ms/Verify
  noise floor that is a 9-10x margin -- by a wide distance the largest,
  most-certain effect in this entire analysis, and it is **symmetric**: the
  3080s gain the same 27 % their weight shard shrinks by.
* **V2 is a decode regression (+8 %)**, because bf16 attention/GDN weights
  outweigh the 4-bit MLP saving.

### 6.1 Context

At the anchor's `max_total_num_tokens = 433,017` (KV 32.0 KiB/token, identical
on every rank under DCP; `proofs/anchor.txt:269-284`), the freed VRAM converts
directly:

| variant | added KV tokens | resulting context |
|---|---:|---|
| V1 | +206,632 | **~640,000 (1.48x)** |
| V2 | -70,048 | ~363,000 (0.84x) |
| V4 | +247,536 | **~681,000 (1.57x)** |

Three orders above any noise floor. Combined with #299 §7 (`--rank-kv-ratio`
capacity-matching, worth 6.18x on the prefill-optimal arm), this is the axis on
which NVFP4 pays.

### 6.2 The option the thesis does not consider: drop TP

The lockstep window is `max_r(comp_r) + floor` with `floor ≈ 1000 ms`, i.e.
**76-80 % of the best achievable TP=3 window is transport** on a rig with no
GPUDirect P2P and no NVLink (all PHB, `peer_access: false` on every pair, #213).
A single 5090 running the whole model pays none of it:

| variant | weights on one 5090 | free after 4500 MiB reserve | KV tokens | prefill (2048 tok) | decode ms/token |
|---|---:|---:|---:|---|---:|
| FP8 | 25649 | 2458 | 78,656 | 321 ms | 17.58 |
| V1 | 19192 | 8915 | 285,288 | ~845 ms (Marlin) | 13.16 |
| **V4, phi0 = 1.6** | **17914** | **10194** | **326,192** | **~161 ms** | 12.28 |

V4 solo-5090 prefill is ~161 ms for the chunk that costs 1206 ms under the best
TP=3 plan -- a **7.5x** prefill difference, entirely because the collective
disappears. Decode is roughly a wash (12.28 ms solo vs 7.43 ms weight read +
~4.1 ms of decode all-reduce at `ar_10kb_us = 32.4 µs` x 128 collectives/token
= ~11.5 ms for TP=3).

**This is the largest single consequence of NVFP4 on this specific rig: it is
the first weight format under which Qwen3.6-27B fits on one 5090 with a real KV
pool** (FP8 leaves 78 k tokens; V4 leaves 326 k). Two caveats, both important:

* Per the standing "the rig is a lower bound" rule, this says nothing general.
  On a rig with NVLink or working P2P the collective floor is an order of
  magnitude smaller and TP wins. It is a statement about *this* interconnect.
* It is not a dead end for the fork's heterogeneous work -- it is the enabling
  condition for the **weightless KV lane** (Variant C, #131/#133: the fast head
  holds the weights, meta-device workers hold only KV + attention). At 17.9 GiB
  of weights the "fast head holds everything" premise stops being aspirational,
  and the 3080s' 40 GiB become pure KV capacity, which is precisely the role
  §5.3 independently derives for them. Named here as a second use, not costed.

### 6.3 Quality -- literature and experience, explicitly flagged as such

**No quality measurement was made for this document. Everything in this
subsection is published/experiential prior and must be treated as a hypothesis
to be tested, not as a result.**

* NVFP4 (E2M1 values, per-16 E4M3 block scale, per-tensor FP32 global scale) is
  the highest-fidelity 4-bit format currently shipping. Its two advantages over
  MXFP4 are structural rather than empirical: a group of 16 instead of 32, and a
  real E4M3 scale instead of a power-of-two E8M0. Published NVIDIA ModelOpt PTQ
  results typically report sub-1 % benchmark deltas against BF16.
* Against **INT4 AWQ/GPTQ g128**: NVFP4 costs ~0.25 bpw more (4.5 vs ~4.25) and
  buys an 8x finer group plus a non-uniform value spacing that matches weight
  distributions better. Expect parity-to-better, not a step change.
* Against **FP8**: FP8 W8A8 is effectively lossless for this model class. The
  4-bit *weight* step is the small risk; the 4-bit **activation** step (W4A4) is
  the real one. That is almost certainly why NVIDIA's own reference checkpoint
  for this model is `W4A16_NVFP4` for the MLP with FP8 attention -- the publisher
  with the best tooling chose *not* to quantise activations to 4 bits here.
* **Mixed-arch determinism is a separate, harder problem** -- §7 (e).

---

## 7. Net conditions: when does NVFP4 win

### (a) VRAM -- **yes, unconditionally, for V1 and V4**
-6457 / -7736 MiB, 1.48x / 1.57x context, -25 % / -27 % decode step time. All
far above their floors. **V2 loses on this axis** (+2189 MiB) and should be
rejected on that ground alone regardless of its compute story.

### (b) Compute asymmetry vs the collective floor -- **no**
The floor is 64-80 % of the prefill window and is transport of bf16 activations;
no weight format touches it. The addressable remainder is `max_r comp_r`, whose
irreducible term under any 4-bit format is the 3080s' weight-free attention/GDN
residual (~200 ms). Ceiling **3.6 %**, floor **3.18 %**. Not addressable.

### (c) Quality -- **conditional, and the condition is W4A16 vs W4A4**
Weight-only NVFP4 is the safe choice and is what the best-resourced publisher
shipped. W4A4 is where both the 5090 speed-up and the quality risk live. They
cannot be separated: **the knob that buys the asymmetry is the knob that costs
the fidelity.**

### (d) Checkpoint availability -- **solved, but not in the shape wanted**
40+ candidates. Best picks, in order:

1. `ocicek/Qwen3.6-27B-NVFP4` -- **V4**, compressed-tensors all-Linear NVFP4,
   19.17 GiB, MTP cleanly separated as `model-mtp-bf16.safetensors`. The right
   target; **needs #291-S3 to boot on the 3080s.**
2. `nvidia/Qwen3.6-27B-NVFP4` -- **V1**, boots today on all three cards with no
   code change, 20.43 GiB, KV-cache quant algo FP8. The correct *first boot
   proof*, with the +13 % prefill cost understood in advance.
3. `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS` --
   V4-shaped (24.35e9 params at fp4) with 31 k downloads;
   `llmfan46/…-Native-MTP-Preserved-NVFP4` is byte-for-byte the same census.
4. Rejected: everything `mlx*` (MLX packing, not CUDA-loadable), everything
   `*-GGUF` (different loader), `mmangkad`/`AEON-…-NVFP4-MTP` (V2, VRAM
   regression), `unsloth/Qwen3.6-27B-NVFP4` (two alignment traps, §2).

### (e) Spec / MTP compatibility -- **two distinct risks**

1. **Drafter format.** Verify per checkpoint, out of `hf_quant_config.json`,
   whether `mtp.*` appears in `exclude_modules` -- the `-MTP` repo suffix does
   not imply it (§2). Prefer a bf16 drafter: the MTP layer runs at tiny shapes
   where a 4-bit GEMM has no arithmetic advantage and the quant epilogue is pure
   overhead, and accept length is directly quality-sensitive, so a 4-bit drafter
   is doubly bad. `ocicek`'s separate `model-mtp-bf16.safetensors` is the pattern
   to copy.
2. **Cross-rank arithmetic divergence -- the sharper risk.** Under a W4A4
   checkpoint on this rig, rank 0 quantises activations to E2M1 and ranks 1/2
   (Marlin, W4A16) do not. The ranks are computing partial sums of the *same*
   matmul with *different* arithmetic. This is the #50 / hetero-spec-determinism
   family, and unlike FP8 there is **no shared fallback lane** to force parity
   onto: the #291 study already records that no
   `SGLANG_DETERMINISTIC_NVFP4_GEMM` analogue to #192 is constructible without
   new code (`INTEGRATION_R3_VALIDATION.md:9582-9591`). The one knob that does
   exist is `--fp4-gemm-backend marlin`, which puts the **5090 onto Marlin too**
   -- restoring arithmetic parity at the price of the entire asymmetry
   (568 -> 216 TFLOPS). **The determinism knob and the asymmetry knob are the
   same knob, pointing opposite ways.** Any W4A4 bring-up must budget a
   determinism battery (the #50 methodology) as a first-class item, not an
   afterthought.

### (f) Uneven TP -- **blocked until `ModelOptFp4Config.weight_block_size` exists**
§3.5. The format that boots on sm_86 has no coarsening; the format with
coarsening does not boot on sm_86. Both halves of the scissors are ~5-line fixes.

---

## 8. Staircase coupling (#287) and the lane design (#298a / #319 infrastructure)

### 8.1 What the depth-aware staircase needs, and what the profile can give it

#287's staircase already carries two axes (phase x context depth, each with its
own layer + KV optimum). A format asymmetry adds a third, and the brief's
requirement is correct and non-negotiable: **the per-card format factor must
come out of the measured profile, never out of a hardcoded arch table.** The
fork is already disciplined about this -- `uneven_perf.py` contains **no**
`get_device_capability()` call at all; every `sm86`/`sm_120` string in it is a
comment, and the design note at `:770-778` states the principle outright
(*"Asked FUNCTIONALLY … because the integer is ambiguous across vendors --
gfx900 reports (9, 0), the same value as Hopper"*). §3.3's `is_sm100_supported()`
dispatch in `fp4_utils.py` is a live counterexample of what happens when the
integer is trusted: on sm_120 it silently routes past the fork's own kernel.

### 8.2 `nvfp4_*` lanes in `gemm_lanes` v3 -- concrete design

**Format-key detection.** `checkpoint_compute_format` (`uneven_perf.py:1733-1766`)
gains a predicate parallel to `_is_fp8_like` (`:1723-1731`), reading the same
`quantization_config` (and `text_config.quantization_config`, already handled by
`_quant_config` at `:1712-1721`, which matters because every Qwen3.6-27B NVFP4
checkpoint is a VL config):

```python
def _is_nvfp4_like(method: str, fmt: str, qc: dict) -> Optional[str]:
    """Returns "nvfp4_a4" (activations also 4-bit -> native FP4 lane is
    reachable), "nvfp4_a16" (weight-only -> Marlin on EVERY arch, including
    Blackwell), or None.

    The distinction is NOT cosmetic and NOT inferable from the repo name: for
    W4A16_NVFP4 the fork's ModelOptNvFp4A16LinearMethod is unconditionally
    Marlin (modelopt_quant.py:1737-1850), so a 5090 scored on a native-FP4
    lane number would be scored ~2.6x too fast."""
    if method in ("modelopt", "modelopt_fp4"):
        algo = str(qc.get("quant_algo", "")).upper()
        if algo == "NVFP4":
            return "nvfp4_a4"
        if algo == "W4A16_NVFP4":
            return "nvfp4_a16"
        if algo == "MIXED_PRECISION":
            return _mixed_precision_dominant_algo(qc)   # see 8.3
    if "nvfp4" in fmt:                     # nvfp4-pack-quantized
        groups = (qc.get("config_groups") or {}).values()
        for g in groups:
            w, ia = g.get("weights") or {}, g.get("input_activations") or {}
            if int(w.get("num_bits", 0)) == 4 and w.get("type") == "float":
                return "nvfp4_a4" if int(ia.get("num_bits", 0)) == 4 else "nvfp4_a16"
    return None
```

**Lane table.**

```python
_FORMAT_LANES["nvfp4_a4"]  = (LANE_NVFP4_NATIVE, LANE_NVFP4_MARLIN)
_FORMAT_LANES["nvfp4_a16"] = (LANE_NVFP4_MARLIN,)
```

Order mirrors the serving dispatch, exactly as the fp8 triple does. Note there
is **no `w8a16`-style third rung**: no nvfp4 -> bf16 weight-dequant path exists
anywhere in the tree (grep over `dequant.*fp4|e2m1|unpack.*fp4` finds only the
FP4 *KV-cache* dequant and the Marlin in-kernel bit-twiddle). If a rank can run
neither lane, `rank_gemm_scores`' existing "no lane measured on this card"
branch (`:1817-1829`) already produces the correct loud bf16 fallback -- which
is exactly the right, honest behaviour for a compressed-tensors NVFP4 checkpoint
on an sm_86 rank *before* #291-S3 lands, because that rank genuinely cannot
serve it either.

**Probes.** Two new `(Optional[float], note)` functions with the same shape and
warmup/iteration contract as their fp8 siblings, at the same
`_PROBE_GEMM_M/K/N = 2048, 5120, 17408`:

* `_bench_gemm_nvfp4_native_tflops(dev)` -- build packed E2M1 `mat_a`/`mat_b`
  plus swizzled E4M3 scale tensors at
  `(round_up(M,128), round_up(K/16,4))` / `(round_up(N,128), round_up(K/16,4))`
  and call `sglang.jit_kernel.nvfp4.cutlass_scaled_fp4_mm` **directly** -- not
  through `fp4_gemm`, so the measurement is of the fork's own kernel and is not
  silently redirected to flashinfer by the `auto` dispatch of §3.3. On sm_86
  this surfaces as the note `"NVFP4 JIT kernels require compute capability
  >= 10.0, got 8.6"` -- a fact in the profile, not a crash at first forward.
* `_bench_gemm_nvfp4_marlin_tflops(dev)` -- mirror `_bench_gemm_fp8_marlin_tflops`
  (`:811`) using the real serving helpers `prepare_nvfp4_layer_for_marlin` +
  `apply_fp4_marlin_linear`, so the number measured is the number served.
  Reminder for whoever writes it: `prepare_nvfp4_layer_for_marlin` demands
  `group_size == 16` (`marlin_utils_fp4.py:129-131`) and the scale-inverse
  convention differs from the FP8 helper (already flagged at
  `INTEGRATION_R3_VALIDATION.md:9608-9625`).

Both go into `_LANE_PROBES` (`:977-981`) behind the existing
`_check_lane_probe_environment()` gate (`:942`) so a missing `sgl_kernel` or a
`MockScalarTypes` stub cannot be cached as a card fact.

**No `PROFILE_VERSION` bump** -- identical reasoning to #319 §3c, and for the
same measured reason. The new keys go **inside** the already-declared v3 fields
`gemm_lanes` / `gemm_lane_notes` (`_PROFILE_VERSION_FIELDS[3]`,
`uneven_perf.py:133-136`); `migrate_profile` (`:611-629`) tests field-name
presence, not sub-key completeness. Bumping the version changes the cache key
itself (`profile_cache_path`, `:575-587`) and forces every rig to re-probe the
pairwise NCCL link matrix -- the phase that charged **600 s per boot** in the
#303 incident. Cost of not bumping: a rig with a pre-existing v3 profile needs
one `SGLANG_PERF_REPROBE=1`. That is the correct trade.

### 8.3 The third axis proper: one score per rank is not enough

The load-bearing structural gap for #287 is *not* the lane table -- it is that
`rank_gemm_scores(entries, fmt) -> List[float]` returns **one scalar per rank**,
and the format only selects *which measured number* is read
(`uneven_perf.py:1768-1836`; consumed flat at `:4152-4153`, `:4380`, `:4506`,
`_prefill_sharded_time` at `:3759-3786`). That is exactly correct for FP8, where
one lane serves every family on a card. It is **wrong for a MIXED_PRECISION
checkpoint**, where on the *same* card the MLP family runs Marlin at 216 TFLOPS
while attention/GDN runs `fp8_native` at 568 -- a 2.6x intra-card divergence that
the current planner cannot represent at all. (`_mixed_precision_dominant_algo`
in §8.2 is a stopgap that picks the family carrying the most FLOP; it is not a
fix.)

The minimal correct shape:

```python
rank_gemm_scores(entries, fmt) -> Dict[str, List[float]]   # family -> per-rank
_FORMAT_LANES: Dict[Tuple[str, str], Tuple[str, ...]]      # (format, family) -> lanes
```

with `"mlp"`, `"attn"`, `"moe"`, `"vocab"` as the family keys already registered
by `scheduler.py:5367-5393`. Only one consumer needs widening today
(`enc_scores = list(rank_scores_gemm)` at `:4380`/`:4506`), because only the MLP
family currently has its own vector -- which makes this cheap *now* and
progressively more expensive after #287 adds axes on top. **Recommendation: do
the widening as part of #287's planner work, not as part of an NVFP4 bring-up.**

---

## 9. Recommendation, effort classes, and the #291 verdict

### 9.1 Relation to #291

#291 (`INTEGRATION_R3_VALIDATION.md:9335-9675`, commit `74da172417`) already
established, by code reading, that ModelOpt NVFP4 reaches sm_80-sm_89 through
Marlin as **upstream inheritance** (upstream #19491 -> PR #19652 -> revert #22047
-> relanded in #25655 / `b8d7351a74` as SM80+, compressed-tensors not restored),
that `initialize_fp4_gemm_config` is per-rank, that `sgl-kernel/csrc` has no FP4
kernels, that the CT `min_capability 100` with no Marlin branch is
*"die groesste konkrete Luecke"*, and that `docs/advanced_features/quantization.md:41`
still wrongly documents `modelopt_fp4` as Blackwell-only. Its open question --
*"does NVFP4 actually run on a 3080?"* -- is still unanswered by boot. **This
document does not re-derive any of that; it prices it.**

**Verdict on the framing question -- is the 16-bit detour on sm_86 still worth
pursuing as a goal in its own right, or only as a compatibility minimum?**

**As a goal in its own right, and it is the best item in the task.** The reason
is the measurement, not the intention: the sm_86 "detour" is Marlin at
**58.44/59.15 TFLOPS**, which is **0.93-0.96x of the card's own dense bf16** and
**1.09-1.13x of `fp8_w8a16`** -- it is the 3080s' *best* quantised lane, not a
degraded one, and at 4 bits it fetches 1.78x fewer weight bytes than the FP8
lane it replaces, which is where the -27 % decode number in §6 comes from.
Calling it a compatibility minimum undersells it by the entire payoff.

Concretely this promotes **#291-S3** (`CompressedTensorsW4A4Fp4`:
`get_min_capability()` 100 -> 80, plus a Marlin branch reusing
`prepare_nvfp4_layer_for_marlin`, mirroring `modelopt_quant.py:1511-1521`) from
the middle of #291's slice list to **the highest-value item**, because it is the
only thing between this rig and V4 -- the sole variant that is simultaneously
VRAM-positive (-7736 MiB), context-positive (1.57x), decode-positive (-27 %) and
prefill-neutral (-3.1 %).

### 9.2 Ordered plan

| # | item | effort | gate / expected |
|---|---|---|---|
| **1** | **Lane microbench only.** Land `_bench_gemm_nvfp4_native_tflops` + `_bench_gemm_nvfp4_marlin_tflops` (§8.2) as a standalone script mirroring `lane_probe_only.py` (the tool #298b used to sidestep the hanging link-matrix phase, `INTEGRATION_R3_VALIDATION.md:12094-12098`). Run on all three cards, NVML-resolved indices. **No model, no download, seconds of card time.** | **XS** | Produces **phi0**, the one number this analysis had to band. Compare in the same run against the cached 568.48 / 58.44 / 59.15. Also settles whether the 5090's `auto` really lands on flashinfer (§3.3). |
| **2** | **`ModelOptFp4Config.weight_block_size -> [128, 128]`** (§3.5), same value the CT path already computes at `compressed_tensors.py:187-199`, with a comment noting it encodes the *kernel's* `MIN_THREAD_K = 128` alignment and not a quantisation-semantic block. | **XS** | Hard prerequisite for any uneven-TP NVFP4 boot. Independent of step 1's outcome. |
| **3** | **Boot proof: `nvidia/Qwen3.6-27B-NVFP4` (V1) at TP=3**, even split, no `--rank-mlp-ratio`. Closes #291's open question. Expect **-25 % decode, 1.48x context, +13 % prefill**; correctness battery (5-prompt coherence + accept length, the #289 set) before any perf number counts. | **S** | The +13 % prefill is *predicted*; if it does not appear, the §5 model is wrong and everything downstream needs re-deriving. Good falsifier. |
| **4** | **#291-S3**: CT NVFP4 Marlin branch + `min_capability` 100 -> 80. Then boot `ocicek/Qwen3.6-27B-NVFP4` (V4) at TP=3. | **S-M** | The actual payoff arm: -27 % decode, 1.57x context, prefill neutral. |
| **5** | Determinism battery for W4A4 under mixed-arch TP (§7 e), and a decision on whether `--fp4-gemm-backend marlin` should be forced when a mixed-arch plan is detected. | **M** | Only if step 4 boots. This is the #50 family and must not be skipped. |
| -- | **Not recommended**: the attention/GDN family vector (§5.4), the expert-offload x NVFP4 wiring (§3.5, no MoE model on this path), a native sm_120 FP4 kernel effort (one already exists and is unreached by `auto`, which is a dispatch bug, not a kernel gap). | | |

**Stop rule.** If step 1 returns `phi0 < 1.33`, the entire placement thesis is
dead on arithmetic (§5.2) and steps 3-5 should be re-justified purely on the
VRAM/decode axis of §6 -- which they can be, because that axis does not depend
on phi0 at all.

---

## 10. Bottom line, as a condition list

**NVFP4 helps when:**

* the goal is **VRAM, context or bs=1 decode latency**, not prefill throughput
  (-25 to -27 % decode step, 1.48-1.57x context, both far above their floors);
* the checkpoint keeps **attention and GDN at 8 bits or 4 bits, never bf16**
  (V1 or V4; V2 loses on VRAM *and* decode);
* the target rig's **collective is not the bottleneck** -- on a rig with NVLink
  or working P2P the 64-80 % transport floor of §1.2 shrinks and the compute
  lever regains its leverage. Nothing here should be read as a verdict on NVFP4
  in general; this rig is a lower bound, not a jury;
* on **this** rig specifically: it is the first format under which the model
  fits on the 5090 alone with a usable KV pool (17.9 GiB + 326 k tokens),
  which is both a serving option and the enabling condition for the weightless
  KV lane (§6.2).

**NVFP4 does not help when:**

* the goal is **prefill wall time**. Ceiling **3.6 %** against a **3.18 %**
  floor, independent of phi0, checkpoint and split -- because the binding term
  becomes the 3080s' weight-free GDN/attention residual (§5.2);
* the goal is a **bigger 5090-wards split**. The optimum does move to the corner
  `136,0,0` above `phi0 = 1.33`, but the move is worth nothing: the MLP axis was
  already 94 % exhausted at FP8 (#299's `128,4,4`);
* the checkpoint is `W4A16_NVFP4`-for-MLP (`nvidia/Qwen3.6-27B-NVFP4`) **and**
  prefill matters -- it drags the 5090 off `fp8_native` onto Marlin for a
  **+13 %** window;
* the checkpoint is **compressed-tensors** and the rig has any pre-Blackwell
  rank -- it does not boot at all until #291-S3 lands;
* an **uneven `--rank-tp-ratio` / `--rank-mlp-ratio`** is in play on a ModelOpt
  checkpoint -- the coarsening is silently dead (§3.5) and the failure is a
  late Marlin shape abort;
* **bit-exact cross-rank behaviour** is required under W4A4 -- rank 0 quantises
  activations, the Marlin ranks do not, and the only parity knob costs the
  entire asymmetry (§7 e).

**Net:** the thesis is directionally correct and quantitatively empty on the
compute axis, and quietly correct on an axis it did not claim -- 4 bits is a
bandwidth and capacity story, and on this rig that story is symmetric across
cards rather than 5090-weighted.

---

## 11. Reproducing this

```bash
# checkpoint census (CPU only, metadata only, no download)
CUDA_VISIBLE_DEVICES=99 /spinning/shvllm/.venv/bin/python -c "
from huggingface_hub import HfApi, hf_hub_download; import json
api=HfApi()
for r in ['nvidia/Qwen3.6-27B-NVFP4','mmangkad/Qwen3.6-27B-NVFP4','ocicek/Qwen3.6-27B-NVFP4']:
    i=api.model_info(r, expand=['safetensors']);  print(r, dict(i.safetensors.parameters))
    print('  ', json.dumps(json.load(open(hf_hub_download(r,'hf_quant_config.json'))))[:300])
"
# U8 bytes -> fp4 params:  W = 2 * U8   (typed F8_E4M3 scales)  or  W = U8 / 0.5625  (scales folded into U8)
# cross-check against  MLP 17.113e9 | attn+GDN 7.240e9 | embed+lm_head 2.543e9

# split model (pure arithmetic, no GPU) -- inputs are the a/b vectors of section 1.2
# a = [193.2,1202.8,1282.4]  b = [128.0,835.6,820.0]  floor = 1000 ms
# b_GEMM_r = 2*7.240e9*2048 / achieved_r ,  achieved_r = 2*17.113e9*2048 / a_r
# then min over the simplex of max_r ( a_r/phi_r * m_r + b_GEMM_r/psi_r + b_nonGEMM_r ) * share_r
```

**Measurement inputs.** Lane table:
`/root/.cache/sglang/hw_profile-9a5e9b49b7dc.json`,
`/spinning/gpu-battery-results/2026-07-30_lane_reprobe/hw_profile_after.json`,
`docs/dev/INTEGRATION_R3_VALIDATION.md:12092-12134` (#298b), `:13428-13466`
(post-#304). Card probe: `/root/.cache/sglang/card_probe-20ae5edfc9b2.json`
(#213), `INTEGRATION_R3_VALIDATION.md:11487-11500`. Prefill cost model:
`/spinning/gpu-battery-results/2026-07-30_phasen_optima/s15_phasen_optima/{wait/,proofs/}`
via `docs/dev/ANALYSE_299_attention_gdn_split.md` §1-§4. Decode anchor:
`python/sglang/srt/layers/quantization/fp8_utils.py:356-357`. Noise floors:
`INTEGRATION_R3_VALIDATION.md:11305`, `:11794-11816`. Prior NVFP4 study:
`INTEGRATION_R3_VALIDATION.md:9335-9675` (#291). Sibling format analysis:
`docs/dev/ANALYSE_319_int8_lane.md`.
