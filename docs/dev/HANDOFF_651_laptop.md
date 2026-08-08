# Handoff #651 — Qwen3.6-35B-A3B GGUF bring-up on the APU laptop

Branch: `feat/gguf-q4-bringup-651`
Revised 2026-08-08 by the #651 strand. Revision 1 folded in the real target
hardware (an AMD APU, not a discrete-NVIDIA laptop). Revision 2 — this one —
folded in the prior session's working setup found on the laptop itself, which
inverts revision 1's "cannot run" conclusion. The geometry, the #647 fix and
the parallelism analysis are unchanged throughout.

---

## 0. The headline, before anything else

**It already works.** Qwen3.6-35B-A3B **Q4_K_M GGUF serves on the Radeon 780M
iGPU** through htsglang at **~12.5 tok/s decode and ~165 tok/s prefill**, bs=1,
ctx 8192, TP=1, no speculation. Measured on the machine, 2026-08-07 — see §1.5.

That is true **despite** this tree having no ROCm GGUF path at all. The gap was
closed *outside* the repository: the laptop runs a modified sglang copy plus a
purpose-built `sglang_gguf_rocm` extension. So there are two different true
statements, and confusing them is the main hazard in this area:

| | Status |
|---|---|
| **This tree (`feat/gguf-q4-bringup-651`)** | GGUF has **no ROCm path**. Kernels absent from the ROCm build, ops never registered, `sgl-kernel` refuses gfx1103 outright. Fully evidenced in §2. |
| **The laptop, right now** | **Works** for Q4_K_M without speculation, via an out-of-tree port that supplies exactly what §2 says is missing. §1.5 |
| **CPU compute, via this fork's only current path** | Dense materialization needs ~67 GiB against 29.5 GiB. But that is a **missing-kernel gap, not physics** — llama.cpp computes on quantized blocks and Q4 stays ~22 GB. §3 |

**The two real problems now are:**

1. **Speculation (NEXTN/MTP) hard-faults the HIP context.** It boots, serves a
   batch, then dies with `hipErrorLaunchFailure` within 10-40 s — twice, at two
   context lengths, with a same-config no-spec control that runs clean. §1.5.3
2. **The working code is unversioned and its provenance is unrecoverable.**
   `/root/lh/sglang_src` is **not a git repo** and carries no baked commit id.
   The port that makes this machine work cannot currently be diffed against any
   rig commit. §1.5.4

So the next step is **not** "port the kernels" — that is done. It is: localize
the speculation fault with `AMD_SERIALIZE_KERNEL=3`, and get the out-of-tree work
back into the repository before it is lost. §7.

*(The 2026-08-08 first revision of this file concluded "cannot run". That was
right about the tree and wrong about the machine, because it was written before
the laptop was inspected. Corrected here.)*

### Verification markers used throughout

| Marker | Meaning |
|---|---|
| **[MEASURED]** | Executed on the real target machine or the real checkpoint. Command given. |
| **[CODE]** | Verified by reading this tree, with file:line. |
| **[UNVERIFIED]** | Reasoning or arithmetic only. Treat as hypothesis. |

**Correction to the previous revisions:** the model *has* now generated tokens on
this hardware (§1.5.2). What is still **[UNVERIFIED]** is its **coherence** — no
content-checked probe run survives. `docs/dev/651/probe.py` exists for exactly
this and has never been run against the laptop. Serving 200s and a plausible
tok/s are not evidence of correct output, and this checkpoint's specific failure
mode (§9.1) is fluent, grammatical, wrong text.

---

## 1. The target machine [MEASURED 2026-08-08]

Lenovo ThinkPad P14s Gen 5 AMD. **It is reachable from the rig box** — the
2026-08-07 note that it was unreachable is stale:

```
ssh -i "/root/.ssh/id_ed25519_root@192.168.0.116" root@192.168.0.116   # efeu-TP14
```

| Property | Value | Source |
|---|---|---|
| CPU | Ryzen 7 PRO 8840HS, 8C/16T | user |
| iGPU | Radeon 780M, RDNA3, **gfx1103** | `rocminfo` |
| RAM | 32 GB DDR5, **shared** CPU/iGPU | user |
| `MemTotal` | **30,211 MiB** (29.50 GiB) | `/proc/meminfo` |
| `MemAvailable` | 28,568 MiB (27.90 GiB), desktop idle | `/proc/meminfo` |
| Swap | 8,192 MiB | `/proc/meminfo` |
| **VRAM (BIOS UMA)** | **1,024 MiB** | `mem_info_vram_total` |
| **GTT** | **24,576 MiB** | `mem_info_gtt_total` |
| kernel cmdline | `amdgpu.gttsize=24576 ttm.pages_limit=6291456` | `/proc/cmdline` |
| OS / kernel | Ubuntu 26.04, **7.0.0-29-generic** | `uname -r` |
| ROCm | **7.1** from Ubuntu packages (no `/opt/rocm`) | `dpkg -l` |

Two facts that shape every number later:

- **The 1 GiB UMA carve-out is the BIOS minimum and cannot be reduced** (user,
  2026-08-08). This is the right setting: a small dedicated carve-out leaves the
  most memory to the OS and pushes the weights into GTT instead.
- **GTT is backed by the same DDR5 as system RAM.** It is a cap on how much the
  iGPU may pin, *not* memory in addition to `MemTotal`. Somebody has already
  raised it deliberately to 24 GiB on the kernel command line, which is the
  single most important tuning already done on this machine.

**GPU-addressable ceiling = 1,024 + 24,576 = 25,600 MiB.**

### 1.1 Checkpoints, on the laptop [MEASURED]

In `/root/lh/models/` — note these are **not** the rig's file:

| File | Bytes | MiB |
|---|---:|---:|
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 22,663,387,424 | **21,614** |
| `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` | 12,574,128,416 | **11,992** |

The rig carries `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (21,795 MiB) instead. The
2026-08-07 claim that "there is no smaller fallback without a new download" is
**false for the laptop**: a Q2_K_XL at 11,992 MiB is already there, and it
halves the weight budget. **But per the checkpoint policy (§6.0), Q4_K_M is the
target and Q2 is "zu dumm" — Q2_K_XL is a DEBUG vehicle only** (kernel
validation, spec-fault hunting, fast iteration). No Q2 number may be quoted as a
result. If a configuration stops fitting once the CPU stage takes memory, the
fallback is a **Q3 variant (UD-Q3_K_XL class), which is NOT on the laptop and
would need a download** — not Q2.

`config.json`, `tokenizer.json` and `chat_template.jinja` are present beside
them, which is what `--tokenizer-path` needs.

---

## 1.5 What is ALREADY RUNNING on the laptop [MEASURED 2026-08-07/08]

A prior session left a complete working setup in `/root/lh/`. Nobody had
recorded it here. This section is the single most important part of this
handoff, because it inverts the conclusion of §2.

### 1.5.1 The out-of-tree stack

| Component | State |
|---|---|
| venv | `/root/lh/venv/` (Python 3.12.13, uv-managed); every boot script sources it |
| torch | **2.10.0+rocm7.0**, `torch.version.hip = 7.0.51831`, `torch.version.cuda = None` |
| device | `torch.cuda.is_available() = True`, **`AMD Radeon 780M Graphics`**, `gcnArchName='gfx1103'`, `total_memory=24576MB` (that is GTT, not the 1 GiB carve-out), 6 CUs, 2 MB L2 |
| sglang | **editable from `/root/lh/sglang_src/python`** via a `_htsglang_fork.pth`; version `0.0.0.dev0` |
| `sgl_kernel` | **NOT installed** |
| GGUF kernels | **`sglang_gguf_rocm.cpython-312-x86_64-linux-gnu.so`**, a standalone extension built for **gfx1100** in `/root/lh/ggufbuild` |

The extension exports `ggml_dequantize`, `ggml_mul_mat_a8`,
`ggml_mul_mat_vec_a8`, `ggml_mmvq_kq_tuned`, `ggml_moe_a8`, `ggml_moe_a8_vec`,
`ggml_moe_get_block_size`, `ggml_mxfp4_native`. The modified `gguf.py` imports
it at `:99` behind an **`elif _is_hip:`** arm at `:75` — i.e. exactly the missing
binding that §2.3 and §7 item 3 describe, already written, just not in this repo.

**`HSA_OVERRIDE_GFX_VERSION=11.0.0` is load-bearing and verified.** It is set in
8 boot scripts, not in the ambient environment. Without it a 64x64 matmul fails
with `hipErrorInvalidDeviceFunction`; with it, `gcnArchName` reports `gfx1100`
and the matmul returns. The script comment states the reason: *"torch ROCm
carries no gfx1103 code objects and the sglang_gguf_rocm extension is built for
gfx1100"* (`boot_q4_spec.sh:3-4`).

### 1.5.2 It serves, and the numbers

Eight logs contain `The server is fired up and ready to roll!`. The clean runs
are Q4_K_M **without** speculation (`q4_d`, `q4_e`, `q4_f`, 15:22-16:17, all
ended by operator SIGTERM, no exception), answering real `POST /generate 200 OK`.

```
q4_f.log  Load weight begin. avail mem=24.76 GB
          Load weight end. elapsed=74.33 s, avail mem=3.38 GB, mem usage=21.38 GB
          max_total_num_tokens=8192, context_len=8192, available_gpu_mem=2.86 GB
```

| Metric | Measured |
|---|---|
| decode, bs=1, steady state | **~12.3-12.8 tok/s** |
| prefill, pure compute (`#new-token: 841`, gpu-ms 4880-5347) | **~157-172 tok/s** |
| weight load | 74.33 s, **21.38 GB** resident |
| free for KV after weights, ctx 8192 | 2.86 GB |

**This validates §4's budget model.** Predicted room under the 25,600 MiB
ceiling was 3,137 MiB; measured free-after-weights was 3.38 GB. The model is
sound and can be trusted for the remaining planning.

Flags actually used (`boot_q4_lean_nospec.sh`): `--device cuda` (HIP), `--tp-size
1`, `--load-format gguf --quantization gguf`, `--attention-backend triton`,
`--sampling-backend pytorch`, `--disable-cuda-graph`, `--disable-radix-cache`,
`--mamba-radix-cache-strategy no_buffer`, `--disable-overlap-schedule`,
`--page-size 1`, `--mem-fraction-static 0.95/0.97`, `--chunked-prefill-size 1024`,
`--max-running-requests 1`. Env: `HSA_OVERRIDE_GFX_VERSION=11.0.0`,
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (torch warns it is
unsupported on this platform), `OMP_NUM_THREADS=12`. Wrapped in
`systemd-run --scope -p CPUQuota=1200%`. **`--device cpu` was never used.**

### 1.5.3 The live blocker: speculation faults the HIP context

The decisive evidence is an A/B four minutes apart with **identical flags except
speculation**:

| Run | Config | Outcome |
|---|---|---|
| `q4speclean_a` 16:33 | NEXTN, ctx 2048, memfrac 0.97 | ready 16:33:22, **dead 16:33:31** |
| `q4ctrl` 16:37 | **same, spec removed** | **ran fine**, produced tokens, SIGTERM |

```
[16:33:28] Prefill rank batch, #new-token: 2, #chunks: 1, gpu-ms: 215.2
scheduler.py:5206: UserWarning: HIP warning: unspecified launch failure
[16:33:31] Scheduler hit an exception:
    batch.seq_lens_cpu = batch_result.new_seq_lens.to("cpu")
torch.AcceleratorError: HIP error: unspecified launch failure
```

The context is then unrecoverable — even `torch.cuda.set_stream` in the
`__exit__` handler raises, and the process aborts. The same fault hit
`q4spec_fix647` at ctx 8192 (ready 16:22:17, dead 16:22:53). **[UNVERIFIED]**
the crash *site* is meaningless: `.to("cpu")` is merely where an async fault
surfaces. The offending kernel is upstream in the draft/verify step.

Two earlier speculation problems were found and fixed on the way, and are worth
keeping:

- **Draft checkpoint quantization**: `ValueError: Draft checkpoint left 2
  parameter(s) of Qwen3_5ForCausalLMMTP unloaded:
  ['model.layers.0.mlp.gate.weight', 'model.layers.0.mlp.shared_expert_gate.weight']`
  — i.e. **exactly the #647 pair of §9.1, observed live**. Fixed by passing
  **`--speculative-draft-model-quantization gguf`**. Carry this flag.
- **Mamba state cache at ctx 8192 under spec**: `max_mamba_cache_size=0
  (total_rest_memory=0.56 GB, mamba_cache_per_req=61.41 MB)`. The spec arm needs
  headroom the no-spec arm does not — which is precisely where the 818 MiB
  vision tower (§9.4) would pay for itself.

Also fixed earlier: `ModuleNotFoundError: No module named 'sgl_kernel'` reached
from `kernels/selector.py:64` on the MoE path (`fused_moe_triton/layer.py:2202`).

### 1.5.4 A real correctness finding: Q6_K dequant is broken on gfx11

Contained, **not fixed**. From the modified `gguf.py:334-345`:

> *"CONTAINMENT, NOT A FIX. Q6_K dequantise returns non-deterministically wrong
> values on gfx1103 (Radeon 780M): eight runs on one fixed input tensor differ
> from each other, worst max|d| 5.8e-01 against the numpy reference, up to 75
> non-finite values in 262144, and it is wrong on the FIRST call in a fresh
> process... Q4_K and Q5_K are byte-identical across the same eight runs."*

Eight hypotheses are recorded as falsified, including that it is an
`HSA_OVERRIDE` artefact — *"a native gfx1103 build is worse — Q5_K becomes
affected too"*. Scope is narrow: **Q6_K is 4 tensors of 753**, but one of them is
the **lm_head**, which this checkpoint ships as Q6_K. Handling:

- `ggml_mul_mat_a8` (MMQ) is **validated correct for Q6_K on gfx1103**, max|d|
  5.5e-04 against the numpy reference on real weights, so Q6_K is pinned to MMQ
  at any token count.
- The load-time path is rescued by an exact one-time **CPU** dequantise:
  `q4_f.log` — *"GGUF ROCm containment: dequantised a Q6_K layer (248320 x 2048)
  once on the CPU at load; its GPU dequant kernel is known-bad on gfx11."*

This is an open root-cause question and a good candidate for the first real
kernel investigation on this hardware.

### 1.5.5 What is NOT evidence

`boot_pd631.sh` runs `MODEL=/root/lh/models-2b`, a **safetensors 2B model, not
the GGUF** (the script says so itself). Its success proves PD plumbing only.
**Do not cite it as GGUF or as 35B evidence.**

Likewise: `bench.py`, `bench_spec.py` and `bandwidth_floor.py` exist and are
correctly written — `bench_spec.py` even takes accept length as
`completion_tokens/spec_verify_ct` rather than the `spec_ema_accept_len` trap —
but **no result file exists anywhere and no log contains their output**. Their
stdout went to a lost session. **There is no bandwidth-floor number and no
accept-rate number.** The ~12.5 tok/s decode figure therefore has no denominator:
it implies ~20-22 GB/s effective read bandwidth, plausible for dual-channel
DDR5-5600 on an APU, but until `bandwidth_floor.py` is run and *saved* it cannot
be called good or bad.

### 1.5.6 Machine state as of this writing

No server running; no listening port beyond sshd/resolved/cupsd. GPU idle
(`mem_info_vram_used` 359.7 MiB of 1024, `mem_info_gtt_used` 59.9 MiB of 24576),
so the full 22 GB footprint is free to re-take. `free -m`: 30210 total, 1674
used, 28536 available.

Housekeeping left behind, harmless but untidy: `laws_sampler.py` (pid 13802) is
still appending to a **43.9 MB and growing** `laws_boot.jsonl`, and **12 orphaned
`triple.sh` samplers** are still writing `q4_*.csv` (which is why those files
carry current mtimes).

---

## 2. Backend verdict: GGUF has no ROCm path in this tree [CODE]

**Scope note:** everything in this section is about **this repository**. It is
all verified and all still true — and it is exactly the gap the laptop's
out-of-tree port (§1.5.1) fills. Read it as the specification of what must be
brought back into the tree, not as a statement that the laptop cannot run.

This is the question the 2026-08-07 revision could not answer because it
predates knowing the GPU is AMD. It is now answered, and the answer is
unambiguous.

### 2.1 The kernels are not in the ROCm build

The GGUF K-quant sources **do exist** — 8,328 lines under
`sgl-kernel/csrc/quantization/gguf/`: `gguf_kernel.cu` (entry points +
`quantize_q8_1`), `vecdotq.cuh` (the K-quant dot products), `mmq.cuh`,
`mmvq.cuh`, `dequantize.cuh`, `moe.cuh`, `moe_vec.cuh`, `ggml-common.h`.

They are compiled **only** for CUDA. `sgl-kernel/CMakeLists.txt:323` lists
`csrc/quantization/gguf/gguf_kernel.cu` in the CUDA `SOURCES`; the file contains
no `USE_ROCM` or `HIP` token anywhere — it is the CUDA build exclusively.

ROCm builds instead through `sgl-kernel/setup_rocm.py`, whose complete source
list at **`setup_rocm.py:43-60`** is 17 entries ending at
`csrc/elementwise/pos_enc.cu`. It contains **no `csrc/quantization/` entry of
any kind**. There is no hipify step either — a repo-wide grep for
`hipify`/`HIPIFY` under `sgl-kernel/` returns nothing; `.cu`/`.hip` files go
straight to hipcc (`setup_rocm.py:106-117`).

The omission is deliberate rather than an oversight in a shared list:
**`setup_musa.py:91` does list `csrc/quantization/gguf/gguf_kernel.cu`.** MUSA
got the port; ROCm did not.

### 2.2 The ops are never even registered on ROCm

`sgl-kernel/csrc/common_extension.cc:428-480` is the only place the ggml ops are
bound, and every binding names the CUDA dispatch key:

```cpp
m.impl("ggml_dequantize",     torch::kCUDA, &ggml_dequantize);   // :433
m.impl("ggml_mul_mat_vec_a8", torch::kCUDA, &ggml_mul_mat_vec_a8); // :438
m.impl("ggml_mul_mat_a8",     torch::kCUDA, &ggml_mul_mat_a8);   // :441
m.impl("ggml_moe_a8",         torch::kCUDA, &ggml_moe_a8);       // :448
m.impl("ggml_moe_a8_vec",     torch::kCUDA, &ggml_moe_a8_vec);   // :454
```

The ROCm build compiles `csrc/common_extension_rocm.cc` instead
(`setup_rocm.py:47`), which registers 21 ops across elementwise / allreduce /
moe / speculative / kvcacheio / grammar / memory and has **no quantization
section and zero `ggml_*` symbols**. On ROCm
`torch.ops.sgl_kernel.ggml_dequantize` does not exist as a *schema*, never mind
as a kernel.

### 2.3 The Python gate fails silently, and worse than expected

`gguf.py:33-37` computes `_is_cuda = is_cuda()` and `_is_hip = is_hip()`, where
(`utils/common.py:146-148`, `:130-131`):

```python
def is_cuda(): return torch.cuda.is_available() and torch.version.cuda is not None
def is_hip():  return torch.version.hip is not None
```

On a ROCm PyTorch build `torch.version.cuda is None`, so **`_is_cuda` is False
and `_is_hip` is True**.

Now read `gguf.py:41-62` carefully — verified in this tree. The ggml imports
**and their `None` fallbacks** are all nested inside `if _is_cuda:`:

```python
if _is_cuda:
    try:
        from sgl_kernel import moe_sum
        from sgl_kernel.quantization import (ggml_dequantize, ggml_moe_a8, ...)
        _has_sgl_gguf_kernels = True
    except ImportError:
        moe_sum = None
        ggml_dequantize = ggml_moe_a8 = ggml_moe_a8_vec = None      # <- inside
        ggml_moe_get_block_size = ggml_mul_mat_a8 = ... = None      # <- inside
    from sglang.jit_kernel.activation import gelu_and_mul, silu_and_mul
elif _is_musa: ...
elif _is_npu:  ...
else:
    if not _is_hip:
        warnings.warn("Only CUDA, MUSA and NPU support GGUF quantization currently.")
```

Three consequences, in ascending order of nastiness:

1. On ROCm none of those branches run, so `ggml_dequantize`,
   `ggml_mul_mat_a8`, `ggml_mul_mat_vec_a8`, `ggml_moe_a8`, `ggml_moe_a8_vec`,
   `ggml_moe_get_block_size`, `moe_sum`, `silu_and_mul` and `gelu_and_mul` are
   **never bound as module globals at all**. The first forward pass dies with a
   bare **`NameError`**, not a clean `NotImplementedError` — at `gguf.py:840`,
   `:927`, `:932`, `:1010`, `:1092`, `:1107`.
2. **The warning is actively suppressed on exactly this hardware.** `:78-79`
   fires the "Only CUDA, MUSA and NPU support GGUF" message only `if not
   _is_hip`. ROCm gets a differently-worded warning later from
   `GGUFConfig.__init__` (`:88-90`), and neither raises. `_is_hip` is otherwise
   cosmetic — its only other use is a block-size constant, `:995`,
   `return 8 if _is_hip else 4`.
3. **The fail-fast hook abstains.** `GGUFConfig.supports_current_device()`
   (`gguf.py:134-152`) returns `_has_sgl_gguf_kernels` only `if _is_cuda:` and
   otherwise returns `None`, so `_enforce_capability_floor`
   (`model_loader/loader.py:225-250`) gets no opinion — and its numeric fallback
   is confined to the NVIDIA namespace (`loader.py:203-204`, `:218`, `:241`).

**Net effect: the model LOADS cleanly on ROCm and then crashes with a `NameError`
deep inside the matmul.** Do not read a successful load, or the absence of the
warning, as support. This is the single most misleading behaviour in the whole
area.

The predecessor's framing — "`_is_cuda` is a BUILD probe so the warning never
fires" — is confirmed, with one precision correction worth carrying: it is a
build-**plus-device-visibility** probe. With `CUDA_VISIBLE_DEVICES=` empty on a
CUDA box, `_is_cuda` goes False and the warning *does* fire.

### 2.3.1 `sgl-kernel` will not even build for this GPU

Independent of GGUF, **`setup_rocm.py:77-81`**:

```python
if amdgpu_target not in ["gfx942", "gfx950"]:
    print(f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. ...")
    sys.exit(1)
```

gfx942/gfx950 are CDNA datacenter parts. This laptop is **gfx1103**, so the
ROCm build of `sgl-kernel` **exits 1 before compiling anything**. `AMDGPU_TARGET`
is read from the environment at `:65` but is validated by the same gate, so it
is not an escape hatch. This is a prior and separate blocker from the missing
GGUF sources, and it must be fixed first.

### 2.4 The good news: the port is mechanically shallow

Before anyone prices this as a rewrite — the kernels are unusually portable:

- **No blocking intrinsics.** Greps across the gguf directory for `__ldg`,
  `cp.async`, `asm volatile`, `mma`, `wmma`, `__shfl`, `__ballot`, `__reduce`,
  `__funnelshift` return **zero hits**. No tensor-core paths, no inline PTX.
  These are plain integer-SIMD dot products.
- **The AMD shims already exist**, inherited from upstream.
  `ggml-common.h:1019` opens an `#if defined(USE_ROCM)` block supplying
  `__vsubss4` (`:1025`), **`__dp4a` via `__builtin_amdgcn_sdot4`**
  (`:1046-1048`) and `__vcmpeq4` (`:1057`) — which covers the many `__dp4a`
  call sites in `vecdotq.cuh`.
- **`moe.cuh` already carries ~20 `USE_ROCM` tuning branches** (e.g. `:170-182`,
  and again at `:298`, `:426`, `:554`, `:682`, `:810`, `:938`, `:1066`, `:1194`,
  `:1322`).
- **`ggml-common.h:6` hardcodes `WARP_SIZE_GGUF 32`** — which is *correct* for
  RDNA wave32, i.e. for this laptop. It would be wrong for CDNA wave64, which is
  plausibly why nobody ported it for the gfx942 target the build gate names.

Remaining friction is header plumbing: `gguf_kernel.cu:3-5` includes
`<c10/cuda/CUDAGuard.h>`, `<cuda_fp16.h>`, `<cuda_runtime.h>`, which need hipify
or the torch HIP header shims. Untested: numerical correctness of the
`__builtin_amdgcn_sdot4` paths on RDNA, and whether the CDNA-chosen `moe.cuh`
tuning constants are sane on an APU.

### 2.5 The tree already documents this, in two places

Both agree with the code, and both were missed on 2026-08-07:

- `docs/advanced_features/quantization.md:39` — compatibility matrix, columns
  CUDA / ROCm / Ascend: `| gguf | Yes | No | Yes | CUDA-only kernels in
  sgl-kernel; Pre-dequantized on Ascend |`
- `docs/platforms/amd_gpu.md:119` — "Methods that depend on Marlin or
  NVIDIA-specific kernels (`awq_marlin`, `gptq_marlin`, **`gguf`**,
  `modelopt_fp8`, `modelopt_fp4`) do not [work on AMD]."

### 2.6 The decisive check the laptop operator runs first

No GPU compute; it only asks whether the schema was ever registered.

```bash
python -c "import torch, sgl_kernel; \
  print('hip=', torch.version.hip, 'cuda=', torch.version.cuda); \
  print('ggml op registered:', hasattr(torch.ops.sgl_kernel, 'ggml_dequantize'))"
```

**Actually run on the laptop, 2026-08-08 [MEASURED]:**

```
hip= 7.0.51831 cuda= None
sgl_kernel: NOT INSTALLED -> No module named 'sgl_kernel'
sglang_gguf_rocm: /root/lh/venv/lib/python3.12/site-packages/sglang_gguf_rocm.cpython-312-x86_64-linux-gnu.so
  exports: ['ggml_dequantize', 'ggml_mmvq_kq_tuned', 'ggml_moe_a8',
            'ggml_moe_a8_vec', 'ggml_moe_get_block_size', 'ggml_mul_mat_a8',
            'ggml_mul_mat_vec_a8', 'ggml_mxfp4_native']
```

This is the whole story in six lines. `torch.version.cuda is None` confirms
`_is_cuda` is False and `_is_hip` is True, exactly as §2.3 predicts. **`sgl_kernel`
is not installed at all** — it cannot be, because of the gfx gate (§2.3.1) — so
the in-tree question "is the ggml op registered" is moot on this machine. The
ops are supplied instead by the **out-of-tree `sglang_gguf_rocm` extension**
(§1.5.1), which exports every operator the GGUF path needs, plus a tuned
`ggml_mmvq_kq_tuned` and `ggml_mxfp4_native` that have no in-tree counterpart.

So on a machine running **this tree unmodified**, the check would report
`ggml op registered: False` and GGUF would be unusable. On the laptop as it
stands, it is usable — through code that lives outside the repository.

Supporting checks, in order:

```bash
rocminfo | grep -m2 gfx                     # expect gfx1103
python -c "import torch; print(torch.version.hip, torch.cuda.is_available(),
           torch.cuda.get_device_name(0))"
echo $HSA_OVERRIDE_GFX_VERSION              # gfx1103 is not officially ROCm-
                                            # supported; 11.0.0 is the usual
                                            # override. Needed for rocBLAS etc.
```

---

## 3. The 67 GiB figure is a property of THIS FORK, not of CPUs [MEASURED]

**Read this section carefully — its headline was wrong in revisions 1 and 2 and
the wrong version is easy to quote.**

The ~67 GiB is the cost of **the fork's only currently existing CPU path**:
dense bf16 materialization at the measured 3.17x, forced because **this tree has
no CPU K-quant kernels**. It is *not* an inherent cost of running this model on
a CPU. **llama.cpp computes directly on quantized blocks on CPU, where Q4 stays
~22 GB.** So the correct statement is:

| Claim | Status |
|---|---|
| "A CPU stage costs 3.17x memory" | True **only** of the dense-materialization stopgap |
| "This model cannot run on CPU in 32 GB" | **FALSE.** llama.cpp is the existence proof |
| "This *fork* cannot today run a CPU stage without dense materialization" | **True**, and it is a missing-kernel gap, not physics |

The arithmetic below therefore prices **one implementation route among two**
(§6.0), and it is the route we do *not* intend to ship.

There is no CPU K-quant kernel (§2.3 — `get_quant_method`, `gguf.py:165-192`,
branches only on `_is_npu`; a grep of the whole file for `is_cpu` or
`device.type == "cpu"` returns zero hits). The only CPU-adjacent route is the
Ascend helper `ggml_dequantize_ascend` (`gguf.py:1557-1581`), which dequantizes
with the numpy reference at **load time** — i.e. the dense weights become
**resident**, not transient.

Dense cost, measured from the checkpoint by `docs/dev/651/dequant_cost.py`:
**506.2 MiB packed -> 1604.2 MiB bf16 per decoder layer, a 3.17x expansion.**

| Quantity | MiB | GiB |
|---|---:|---:|
| dense bf16 weights (40 layers x 1604.2 + vocab) | **~68,500** | **~66.9** |
| `MemTotal` on this machine | 30,211 | 29.50 |
| **overshoot** | | **2.27x** |

**Starting from Q2_K_XL does not help.** Dense bf16 is *quant-independent* — the
same weights, unpacked — so the Q2 file also lands at ~67 GiB. The smaller file
buys nothing on this path.

Two clarifications that prevent a misreading:

- The 3.17x figure is a **static file-footprint ratio** over GGUF tensor
  metadata (`dequant_cost.py:6-11`), weights only. No KV cache, no activations.
- Do not confuse it with the **transient** per-matmul dequant scratch, a
  separate grow-only workspace `_DEQUANT_WS` at `gguf.py:706-835` with its own
  accounting in `gguf_dequant_scratch_residual_bytes` (`:753`).

A *streaming* CPU path — dequantize one layer, matmul, free — would fit in
memory (~1.6 GiB transient) but does not exist in this tree and would be
brutally slow. Note for calibration: llama.cpp runs this class of model on this
class of laptop precisely because it *has* CPU K-quant kernels. The hardware is
not the problem; this tree's kernel coverage is.

---

## 4. Memory budget [MEASURED, and validated against a real load in section 1.5.2]

Reproduce with `docs/dev/651/apu_budget.py`. Ceiling is the 25,600 MiB
GPU-addressable window of §1, not `MemTotal` — that is the binding constraint.

Per-token and per-sequence costs, measured from the checkpoint (§9.2):
KV **20.00 KiB/token** at fp16, **10.00 KiB/token** at fp8_e4m3 (only 10 of 40
layers are full attention); GDN state **61.9 MiB per sequence** at fp32,
**31.9 MiB** at bf16; vision tower **818 MiB** of never-used dense weights
(§9.4).

| Checkpoint | vision tower | weights | room under 25,600 | fp16 ctx | fp8 ctx |
|---|---|---:|---:|---:|---:|
| Q4_K_M | present (today) | 21,614 | **3,137** | ~82k tok | ~164k tok |
| Q4_K_M | dropped (#651b) | 21,614 | 3,955 | ~123k tok | ~245k tok |
| Q2_K_XL | present (today) | 11,992 | 12,758 | ~563k tok | ~1.1M tok |
| Q2_K_XL | dropped (#651b) | 11,992 | 13,576 | ~604k tok | ~1.2M tok |

```
room = 25,600 - weights - vision - GDN(31.9 MiB, 1 seq, bf16)
ctx  = (room - 1,500 MiB runtime reserve) / KV-per-token
```

The 1,500 MiB reserve covers activations, allocator slack and the **unpriced**
dequant scratch (§9.3). It is a working assumption, not a measurement.

Read the table this way:

- **Context is not the binding constraint — the weights are.** Even Q4_K_M at
  fp16 clears 82k tokens, far past anything this machine will chew through at
  APU decode speed. Start at **8k** and raise; do not chase the ceiling figure
  on a first boot, because the scratch term is unknown.
- **Q4_K_M fits, but with only ~3.1 GiB of slack** inside the GTT window. That
  is tight enough that the 818 MiB vision tower is real money — dropping it
  (#651b) is worth ~26% more headroom.
- **Q2_K_XL is roomy** and is the right first vehicle.
- `MemTotal` (30,211 MiB) is not the ceiling; GTT (24,576 + 1,024) is. Raising
  `amdgpu.gttsize` further would trade OS memory for headroom, but at 24 GiB of
  29.5 GiB it is already aggressive.

---

## 5. Shared memory: the reshard is free here [user-confirmed]

The user confirmed the strand's earlier reasoning (2026-08-08): on this machine
a PP-prefill/TP-decode reshard is **logical only — physically everything already
lives in the same RAM**, so a layout flip is an ownership/view reinterpretation
and **bytes never move**.

This is the laptop-side simplification of Route A's rig design. On a discrete-GPU
rig the prefill->decode handover is genuine data movement: KV rows are filtered
by `owned_ordinals` and pushed over RDMA/TCP by mooncake
(`disaggregation/mooncake/conn.py:1502-1558`), and that transfer shaped the whole
PD design. Here its bandwidth cost approaches zero.

**Route A's hardest problem does not exist on this machine.** That problem is the
*weights*: the two layouts want different bytes per rank, so a flip is
regime-wide rather than per-request. With one memory pool and one process there
is no per-rank weight partition to disagree about — the weights are simply
*there*. What remains is scheduling and graph paths, not data placement.

**A latent optimization, unbuilt and worth its own ticket:** on unified memory
the PD KV transfer could be **zero-copy** — hand over ownership rather than
memcpy. **Today the code copies unconditionally.** Out of scope for #651, but
this hardware is the natural place to motivate it.

What shared memory does **not** change, and these are the objections that
survive:

1. **The compute ceiling is untouched.** A CPU prefill stage is capped at
   `1 + R_cpu/R_gpu` — the CPU's share of total FLOPs. That is arithmetic about
   compute, not bytes; a free handover does not move it. (Full derivation:
   `docs/dev/FINDING_651c_cpu_gpu_pp_feasibility.md` §1.)
2. **The 3.17x dequantization penalty gets *worse*, not better.** Those dense
   GiB now come out of the *same* pool the iGPU draws from (§3).
3. **The code blockers are refusals in argument handling**, unaffected by memory
   topology (§6).

---

## 6. THE GOAL (user, 2026-08-08) — CPU+iGPU PP prefill, iGPU-only decode

This supersedes every earlier "achievable goal" in this file, including
revision 1's "CPU as a memory tier, not a compute stage". The user's design,
near-verbatim:

> Prefill runs **pipeline-parallel across the APU with two workers**: the CPU
> part and the iGPU part are the two PP stages. Decode: **only the iGPU
> computes**. **Reshard** between the phases. Rationale: prefill is
> compute-bound, so PP **adds** the APU's two compute pools; decode is
> memory-bandwidth-bound, where two workers on the *same* DDR5 buy nothing —
> the compute decode needs is done by the iGPU alone.

**The CPU is explicitly NOT a fallback and NOT a memory tier. It is an active
PP-prefill stage.**

### 6.0 Checkpoint policy (user, 2026-08-08) — Q4 is the target

**Q2 is "zu dumm" and is not the quality target.** Order of preference:

1. **Q4_K_M — primary.** It already serves on the iGPU (21,614 MiB, ~3.1 GiB
   slack). Everything meant to *stand* is measured here.
2. **A Q3 variant** (UD-Q3_K_XL class) — the fallback if a configuration stops
   fitting once the CPU stage takes memory. **Would need a download; not on the
   laptop today.**
3. Anything smaller, only after that.

**Q2_K_XL stays useful as a cheap DEBUG vehicle** — kernel validation,
spec-fault hunting, fast iteration — because it halves load time and leaves
huge headroom. But **every measurement meant to stand must be re-run on Q4**
(or Q3 if that becomes the shipping fallback). Do not quote a Q2 number as a
result.

### 6.0.1 Pricing the CPU stage — quant-compute is the PRIMARY route

§3 priced a CPU stage holding **all 40 layers** *dense*. A PP stage holds a
**layer share k**, and the dense assumption is the wrong one to build on.
Reproduce with `docs/dev/651/cpu_stage_k.py`.

**ROUTE B (PRIMARY) — CPU-native quantized compute.** Port or adapt ggml's CPU
quant kernels, or write an equivalent minimal CPU quant-matmul. **No 3.17x at
all**: the CPU stage costs `~k x quant-size`, memory the machine already holds
(5,263 MiB spare on Q4_K_M). **k is bounded purely by the balance point, not by
memory, and Q4_K_M stays the target.**

Three things make this much smaller than "port the GGUF surface to CPU":

- **The CPU stage needs far fewer ops than the full GGUF surface** — only the
  quant types this checkpoint actually uses, on the layers the CPU will own.
- llama.cpp's `ggml-cpu` is the existence proof and the reference
  implementation, and the 8840HS is **Zen4**, so the AVX-512 paths apply.
- **CPU compute sidesteps the gfx1103 Q6_K bug entirely** for the tensors it
  owns — and the CPU dequant is already the *validated-correct reference* the
  containment in §1.5.4 leans on.

**ROUTE A (STOPGAP COMPARATOR ONLY) — dense-materialize the CPU stage's layers.**
Nearly free to try (generalize the Ascend load-time helper,
`gguf.py:1557-1581`), and useful as a cheap way to get real co-run numbers
before committing to B. But it caps k hard and forces the debug checkpoint:

| Checkpoint | packed/layer | **max k** | max CPU layers | iGPU MiB |
|---|---:|---:|---:|---:|
| Q4_K_M (target) | 502.0 MiB | **0.119** | **4.8** | 19,216 |
| Q2_K_XL (debug) | 278.5 MiB | **0.281** | **11.2** | 8,864 |

Against the balanced split `L_cpu = 40/(R+1)` and ceiling `1 + 1/R`, route A is
viable on the debug checkpoint for R >= ~3, but **on the Q4 target only for
R >= ~8, where the payoff is already down to ~+12%**. That is precisely why it
is a comparator and not the destination.

**Decision: build B. Use A only if it buys co-run numbers sooner.**

### 6.0.2 Measurement discipline — solo numbers do NOT predict co-run

The user's explicit expectation: **the CPU part will be much slower and take far
fewer layers than the iGPU, and the optimum must be measured.** Critically:

> **Solo measurements of each part do not predict co-run behaviour — under
> parallel operation both parts throttle each other.**

On this APU the coupling is **threefold** — shared TDP/thermal envelope (CPU load
steals the iGPU's power budget and vice versa), shared DDR5 bandwidth, and shared
memory-controller contention, which is distinct from raw bandwidth.

### The only quantity that counts: co-run **ms per round, per stage**

**User, 2026-08-08, verbatim:**

> *"es muss nicht der takt mitgeschrieben werden sondern die ms zeiten der karten
> selbst. der takt sagt nichts aus über die verteilung die wir später brauchen.
> einzig und allein ms 'runden' zeiten."*

**Do NOT record clocks or power as the measurement basis.** Clock and power
telemetry says nothing about the distribution we actually need, and it is
**redundant**: the entire throttling interaction — all three coupling mechanisms
above — is **already fully captured in the measured ms times**. That redundancy
is precisely *why* co-run ms is the number that matters and why solo ms and
derived clock models are not substitutes.

This is the rig's standing **ms-per-round doctrine** applied to the APU:

- Measure **ms per round per worker** — for each stage, the wall-clock ms to
  forward *its own layer share* for one pipeline round.
- Split each worker's round into **compute time vs wait time**. The wait
  component is what exposes an unbalanced split.
- **Runs >= 10 s**, warmup discarded.
- Never tok/s. Note that the one existing iGPU figure (~157-172 tok/s prefill,
  §1.5.2) is in the wrong unit for this purpose and **must be restated as
  ms/round** before it can feed a split decision.

### The optimum condition

For a 2-stage pipeline the optimum is exactly:

```
co-run ms per round (CPU stage)  ==  co-run ms per round (iGPU stage)
```

Derive `--pp-layer-ratio` so those two come out **equal**. Nothing else — not
core counts, not solo throughput, not FLOP estimates — is a valid basis.

### Plan, in order

- **(a) Solo baselines** — ms/round per stage on *real* layers, each alone.
  Diagnostic only: it brackets the search and exposes gross errors. It does
  **not** set the ratio.
- **(b) CO-RUN measurement** — both stages active simultaneously at
  representative shares, ms/round captured per stage. **These are the numbers
  that feed `--pp-layer-ratio`.**
- **(c) Sweep 2-3 candidate splits** around the co-run-derived optimum, since
  contention moves the optimum away from wherever the solo ratio pointed.

`docs/dev/651/bench_prefill.py` supplies the A-vs-A noise floor, warmup discard,
unique prompts and time-bounded runs. **But it measures TTFT, not per-stage
ms/round** — the per-worker round instrumentation is the piece that does not
exist yet and is the thing to build.

**If the co-run measurement shows the CPU stage's net contribution is negligible
or negative after throttling, that is a reportable verdict, not a failure.** The
user wants the real optimum, whatever it turns out to be.

### 6.0.3 R is a derived quantity, not the measurement

`--pp-layer-ratio` exists and sums to backbone depth **40**. Set it by the
**equal-co-run-ms condition of §6.0.2** — *not* proportional to core counts, and
not proportional to a throughput number.

`R` as used in the §6.0.1 tables is a **sweep parameter for pricing the routes**,
and at most a *derived* summary of a completed measurement
(`R = ms_cpu_per_layer / ms_igpu_per_layer` under co-run). It is not itself the
thing to measure, and a solo-derived `R` is not valid input to a split decision.

Known so far: the iGPU prefill figure is ~157-172 tok/s (§1.5.2) — **the wrong
unit**, and it must be restated as ms/round before use. **CPU prefill on real
layers has never been measured at all**, so every speedup figure in §6.0.1 is
conditional. On this APU the iGPU:CPU gap is far smaller than on a dGPU rig, so
the ceiling is plausibly meaningful — but that is a hypothesis, not a result.

`docs/dev/651/bench_prefill.py` supplies the A-vs-A noise floor and the
`--split-from` balance arithmetic; the **per-stage ms/round instrumentation it
lacks** is the piece to build (§6.0.2).

### 6.0.4 The phase flip, and what PP+spec exclusivity actually binds

Decode after the flip is a **single-worker** regime, so the PP-vs-speculation
conflict is a property of the **prefill** phase's world shape only — *in
principle*. **In this tree it is not, and this is the crux:**

`server_args.py:16264-16269` asserts `speculative_algorithm is None` whenever
`pp_size > 1`. That is a **server-level argument check evaluated once at
startup**, not a per-phase condition. A single server booted `pp_size=2` for
prefill therefore cannot have speculation *at all*, however single-worker its
decode phase is. So the flip design needs either the assert made phase-aware, or
two world shapes inside one process.

That is precisely the **#297 envelope / Route A (#631)** question, and its
**laptop degenerate case may be trivial exactly because bytes never move**
(§5) — the rig's hard part is moving weights and KV between layouts, which does
not exist here. **Check what Route A's work already gives you before building
any flip machinery.** Do not re-derive it.

### 6.0.5 Engineering unknowns to map before building

Findings from reading this tree, marked honestly — these are **blockers to
verify**, not assumptions:

- **Mixed-device world (one PP rank on CPU, one on ROCm) is not expressible
  today.** `--device` is a single string for the whole server
  (`server_args.py:1623-1626`); the only per-rank value is a GPU *index*. Worse,
  `GroupCoordinator` picks its device from a platform probe, not from
  `server_args.device` (`parallel_state.py:656-668`:
  `if is_cuda_alike(): self.device = torch.device(f"cuda:{device_id}")`). So
  `--device cpu` puts **both** stages on CPU. This needs a per-rank device
  vector threaded through `ModelRunner.device` -> `GroupCoordinator.device` ->
  `get_default_distributed_backend`.
- **The p2p wire itself is fine.** A gloo `cpu_group` is created unconditionally
  for every group (`parallel_state.py:701-717`) and the PP send/recv already
  branches on tensor residency (`:2334-2364`,
  `comm_group = metadata_group if tensor.is_cpu else group`).
- **The concrete break is the receive side**: the receiver allocates on the
  *sender's* device type (`:238-246`, `:2395`), so a GPU stage receiving from a
  CPU stage gets a CPU tensor and **nothing inserts the `.to(device)`**. Small
  fix, but it will not work until someone writes it.
- **Zero-copy stage-to-stage over shared DDR5 is unbuilt.** gloo will memcpy.
  True handover would need shared-memory tensors or HIP host-registered memory.
  This is the same latent optimization as the PD KV copy in §5.
- **`cpu_graph_runner.py:609-610` asserts `pp_size == 1`**, so a CPU stage would
  have to run eager.
- **PP stages must occupy disjoint GPU groups** (`server_args.py:9066-9098`).

### 6.0.6 Staging

Prerequisite #1 is **the ROCm port for the iGPU part, regardless of everything
above** — decode is iGPU-only, so nothing serves without it. It is already
working out-of-tree (§1.5.1) and needs recovering, not rebuilding.

Then, incrementally: **iGPU-only boot -> verify coherence -> spec stable ->
measure both stages' prefill floors -> add the CPU stage as PP=2 -> the phase
flip.**

---

## 6.1 What was achievable before this correction (superseded, kept for context)

**GGUF Q4 35B, TP=1, PP=1, iGPU-only, with NEXTN speculation.** Stages a and b
of `docs/dev/651/boot.sh`. Specifically:

- **TP is impossible** — TP needs >=2 GPUs; this machine has one iGPU.
- **PP is impossible** — the only candidate second stage is the CPU, blocked by
  §3 and by the device-type blocker below; and even unblocked it is capped at
  +2-20% prefill while costing GiB of the shared pool.
- **Speculation is achievable, but only WITHOUT PP.**
- **CPU as a memory TIER, not a compute stage**: `--cpu-offload-gb`
  (`server_args.py:4273`, `utils/offloader.py:98-140`) parks parameters in host
  RAM and copies them back per forward, so compute stays on the GPU. On shared
  memory that copy is within one physical pool — genuinely favourable here.
  **Whether it composes with GGUF is UNVERIFIED and is the first thing to
  test.** Do not assume it works.
- **MoE expert-offload is walled off for GGUF by #123** — the expert parameter
  is a `GGUFUninitializedParameter` that only takes shape in loader postprocess,
  so the offload machinery has nothing to grab (`expert_offload.py:1245`,
  `:1447`, `:2130`; `planner/rejected.py:289`). That is a rebuild, not a flag.

### 6.2 PP + speculation: closed tree-wide (see 6.0.4 for what this binds)

Both routes are shut, on **every** machine:

- **One server**: `server_args.py:16264-16269` asserts
  `speculative_algorithm is None` when `pp_size > 1`. `DESIGN_625.md:81-91`
  names this **B1**.
- **Two processes (PD pair)**: both arms reject `--speculative-algorithm`
  (`arg_groups/pd_disaggregation_hook.py:194-229`); the only env escape
  `SGLANG_PD_AUTO_DISABLE_SPEC=1` *disables* spec rather than enabling it.
  `DESIGN_631b_draft_kv_wiring.md:7` — "Status: specification only."

**This gap is owned by Route A / #631** and is a prerequisite for the eventual
PP+spec goal. **Do not build it here.**

---

## 7. The actual work queue

Rewritten after §1.5. The port is **done**; it is just not in the repository.

### Immediate, on the laptop

1. **Localize the speculation fault.** Re-run `boot_q4_spec_lean.sh` with
   **`AMD_SERIALIZE_KERNEL=3`** (and ideally `TORCH_USE_HIP_DSA`) to turn the
   async `unspecified launch failure` into a synchronous fault at the real
   kernel. The previous session never did this, so the recorded crash site
   (`scheduler.py:5206`, a `.to("cpu")`) is meaningless. **Prime suspect: the
   NEXTN draft path reusing a GGUF op that was only ever validated on the target
   path** — which is doubly plausible given §1.5.4, where one op is already known
   to be silently wrong on this GPU. Keep
   `--speculative-draft-model-quantization gguf`.
2. **Run `bandwidth_floor.py` and `bench.py` against a no-spec boot and SAVE the
   output to a file.** Cheap, and currently the whole perf picture rests on a
   single unanchored 12.5 tok/s. `bench_spec.py`'s accept-length instrument has
   never once executed.
3. **Give the spec arm its headroom** — at ctx 8192 it could not fit the mamba
   state cache (§1.5.3). Dropping the 818 MiB vision tower (#651b, §9.4) is the
   obvious source, and this is now a concrete motivation rather than a
   nice-to-have.

### Repository hygiene, and it is urgent

4. **Recover `/root/lh/sglang_src` into version control.** It is **not a git
   repo**, has no baked commit id, and came from a `.tgz`. The work that makes
   this machine run — the `elif _is_hip:` arm, the `sglang_gguf_rocm` extension,
   the Q6_K containment, the MoE-path `sgl_kernel` fix — exists in exactly one
   unversioned copy on one laptop. **This is the largest risk in the whole
   ticket.** Diff it against this branch and land it.
5. **Bring the kernel build in-tree**: widen the ROCm gfx allow-list
   (`setup_rocm.py:77-81`) to admit gfx110x, add
   `csrc/quantization/gguf/gguf_kernel.cu` to the ROCm source list, and register
   the ops in `common_extension_rocm.cc`. §2.4 says the kernel bodies should
   largely survive as-is. The laptop's standalone extension is the working
   reference for what this must produce.
6. **Bind the Python path for HIP in-tree** — `gguf.py:41-62` needs the
   `elif _is_hip:` arm, and `supports_current_device()` (`:134-152`) must stop
   returning `None` on ROCm so a missing kernel fails loudly at load instead of
   as a `NameError` mid-forward. Worth landing on its own merits regardless.

### Open kernel question

7. **Root-cause the Q6_K dequant non-determinism on gfx11** (§1.5.4). Eight
   hypotheses are already falsified. It is contained, not fixed, and it is the
   kind of defect that will resurface somewhere else on this GPU.

---

## 8. Staged boot

`docs/dev/651/boot.sh` is **NVML/CUDA-only** — it resolves devices via
`pynvml` UUIDs, exports `CUDA_VISIBLE_DEVICES` and points `LD_LIBRARY_PATH` at
`nvidia/cu13/lib`. **None of that applies to an APU** and it must be rewritten
for ROCm (`rocminfo` / `HIP_VISIBLE_DEVICES` / `HSA_OVERRIDE_GFX_VERSION`)
before use. Its *staging logic* is still the right shape:

| Stage | Config | Answers |
|---|---|---|
| **a** | TP=1, no spec, eager, Q2_K_XL (debug) | loader / kernels / checkpoint |
| **b** | TP=1, NEXTN spec, Q2_K_XL (debug) | the #647 router-gate fix (§9.1), spec fault |
| **c** | TP=1, NEXTN spec, **Q4_K_M** — **counts** | the real memory budget (§4) |
| **d** | **Q4_K_M** + co-run CPU stage | the PP split (§6.0.2) — **counts** |

Q2_K_XL may be used to isolate "does anything work at all" from "does it fit",
because it halves load time and leaves huge headroom. **But it is a debug
vehicle only (§6.0): every stage whose result is meant to stand must be re-run
on Q4_K_M.** Stages c and d below are the ones that count.

Flags that still apply: `--model-path` must be the **`.gguf` FILE**, not its
directory (`_prepare_weights` requires `os.path.isfile`, commit `d274bbe9ce`);
`--tokenizer-path <dir>`; both `--load-format gguf` and `--quantization gguf`;
`--context-length 8192` to start; `--max-running-requests 1`;
`SGLANG_MAMBA_SSM_DTYPE=bfloat16`; and for speculation
`--speculative-algorithm NEXTN --speculative-draft-model-path <the same .gguf>`,
which is **mandatory** for this family — there is no auto-default (§9.2).

Judge **content, never HTTP 200** — `docs/dev/651/probe.py` (`python probe.py
<port>`) runs determined-answer probes twice at temperature 0 and exits non-zero
unless every answer is correct in both rounds and the rounds are identical. The
failure mode of this checkpoint is fluent, grammatical, wrong text with a happy
200. Read acceptance from `meta_info`, **not** `spec_ema_accept_len` (known
measurement trap). Turn CUDA/HIP graphs on before believing any stability claim.

---

## 9. Facts carried forward unchanged from 2026-08-07

These were established against the rig's CUDA tree and the real checkpoint, and
none of them is invalidated by the hardware correction.

### 9.1 The load-bearing fix: #647, commit `0155ff2c00`

"[#651/#647] GGUF: restore the dense name for non-F32 MoE router gates."

`weight_utils.py:1517` renames `.weight` -> `.qweight` keyed on **tensor
dtype**, treating dtype as a proxy for "the destination module is quantized".
Those are different statements: a MoE router gate is never quantized
(`qwen2_moe.py:408`, `:459` build `mlp.gate` and `mlp.shared_expert_gate` with
`quant_config=None`), so it owns a dense `.weight` and has no `.qweight` at all.

Two things hid it: published GGUFs almost always store router gates F32 (the one
type the rename skips), and BF16 arrives from `gguf-py` as raw `uint8` with the
last dimension doubled, which also fails `Tensor.is_floating_point()` and so
slips past the dense-shard rescue in `gguf.py:_cast_dense_qweight`. **F16
survives the misroute; BF16 does not.** That asymmetry is the sting.

On this checkpoint, of **753 tensors exactly two are BF16**, and both are router
gates in the MTP block `blk.40` — so **the NEXTN draft's router never loads**.
A garbage router still routes every token *somewhere*, so the model stays fluent
and is quietly wrong. On the draft side `raise_on_unloaded_draft_parameters`
would likely abort; on the target side an unloaded parameter is only a
`logger.warning`.

Proof [MEASURED, CPU]: new
`test/registered/unit/quantization/test_gguf_qwen35_router_gate_dtype.py` 7/7
pass; **can-fail proof** — with the suffix table neutralized 3 of 7 fail, the
other 4 being inertness cases that correctly stay green; GGUF unit cluster 33
passed; and on the **real checkpoint** through the real adapter both gates
arrive dense and correct, `mtp.layers.0.mlp.gate.weight` bf16 `(256, 2048)` and
`...shared_expert_gate.weight` bf16 `(1, 2048)`, all finite, std 0.0096 / 0.0020.

**[UNVERIFIED]** that the fixed draft improves acceptance on-card.

### 9.2 Geometry, cross-checked [MEASURED]

GGUF header and `config.json` agree on **every** axis — this mattered, because
an unvalidated sibling config once produced a 64-of-96-layer build that booted
and served nonsense.

`block_count` 41 = 40 backbone + 1 MTP; hidden 2048; 16 attn / 2 KV heads; head
dim 256; 256 experts, 8 active; expert FFN 512; full-attention every 4th layer
(10 of 40); SSM inner 4096, state 128, groups 16; `nextn_predict_layers` 1.
`general.architecture = qwen35moe`, `model_type = qwen3_5_moe`.

The MTP class unwraps `config.text_config` (`qwen3_5_mtp.py:64-66`), whose
`model_type` is `qwen3_5_moe_text`, so the draft block builds **MoE** layers
matching `blk.40`, not a dense MLP. The GGUF vocab is quantized-resident
(`embed_tokens`/`lm_head` arrive as uint8 `.qweight`), so the draft shares the
target's embedding/head as **modules** (`set_embed_and_head_modules`) — which is
why a packed head works.

Weight totals: **21,784 MiB** on the rig's Q4_K_XL, of which routed experts are
**18,726 MiB (86.0%)**, vocab 1,031, full attention 787, GDN 529, MTP block 504,
shared expert 128, gates 80. **86% of the model is routed experts** — so any
plan that must shed more than ~3 GiB has to shed experts, which is the #123 wall
(§6.1).

The file is **TEXT-ONLY**: `config.json` declares a 27-layer vision tower but
the GGUF contains zero vision tensors and there is no `mmproj*.gguf`.
`model_config.py` force-disables multimodal for GGUF without an mmproj, which
closes the #52 NaN-contamination path.

Full weight-stream audits [MEASURED, CPU]: target delivers 63,841 tensors of
which exactly **301** are dense `.weight`, closing arithmetically as
`30 GDN x 4 + 10 attn x 2 + 40 x 4 + 1 final norm = 301`, **zero orphans**.
Draft delivers 63,000, all **1,560** `mtp`-bound names present; the 61,440
dropped are the main model's experts (`40 x 256 x 3 x 2`), harmless but it means
**the draft load reads the whole 22 GB file to throw most of it away**.

### 9.3 Dequant scratch is still unpriced [UNVERIFIED]

For K-quants this fork runs MMQ only up to `SGLANG_GGUF_MMQ_MAX_TOKENS`
(default **8**) and dequantizes to cuBLAS above that. The workspace is a
**prefill-batch-dependent transient**, not a constant. If you OOM during prefill
but not at load, this is the first suspect — lower `--max-num-batched-tokens` or
`SGLANG_GGUF_MMQ_MAX_TOKENS` before touching anything else.

### 9.4 The vision tower costs 818 MiB for nothing

`qwen3_vl.py:1242` constructs `self.visual` **unconditionally** with
`quant_config=None` (dense bf16), ~429M params = **818 MiB per rank**, never fed
on this text-only checkpoint. Filed as
`docs/dev/FINDING_651b_vision_tower_vram.md`. On a machine with 3.1 GiB of slack
(§4) this is worth ~26% more headroom.

### 9.5 Parallelism facts, for whoever revisits PP

- `moe_intermediate_size = 512` against a 256-byte K-quant block gives exactly
  **two** aligned shards, so **TP=2 is the clean width** and TP=3 degrades to a
  REPLICATED shared expert (`qwen2_moe.py:_shared_expert_uneven_misaligned`,
  whose docstring uses `512 -> [229,141,141]` as the GGUF example).
  `num_key_value_heads = 2` points the same way independently.
- **`--pp-layer-ratio` must sum to the backbone depth 40, NOT the GGUF
  `block_count` 41** — the extra block is the MTP head.
- **`max_total_num_tokens` is min-reduced across the world group**, so the
  tightest stage caps capacity for every stage (DESIGN_625 B5).
- Keep hierarchical/disk HiCache **off** on a PP arm (#630 is a silent wedge at
  warmup, health 503 forever), and ensure the tree carries the #633 PP
  weight-update-group deadlock fix (upstream sglang#33934).
- Uneven PP split is merged and tested — `--pp-layer-ratio`, `--pp-stage-ratio`,
  `SGLANG_PP_LAYER_PARTITION` — and PP p2p has a working gloo/CPU path. The
  weighting idea is expressible; it just cannot address a CPU stage.

---

## 10. Reproducing the desk checks

CPU-only, no GPU. From the worktree root — **the `PYTHONPATH` matters**, without
it you silently test a different tree:

```bash
PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES="" python -m pytest \
  test/registered/unit/quantization/test_gguf_qwen35_router_gate_dtype.py -q

python docs/dev/651/apu_budget.py      # section 4, from measured laptop constants
python docs/dev/651/vram_budget.py     # section 9.2, from the checkpoint
python docs/dev/651/dequant_cost.py    # the 3.17x of section 3
python docs/dev/651/layer_split.py     # PP split arithmetic
```

`docs/dev/651/bench_prefill.py` measures TTFT with `max_tokens=1`, unique prompts
so prefix caching cannot serve them, an A-vs-A noise floor before any
cross-device claim, warmup discarded, time-bounded runs, and a length sweep.

---

## 11. Open questions

1. **What kernel actually faults under NEXTN?** (§1.5.3, §7 item 1) — the one
   question standing between this machine and the stated #651 goal.
2. **What is the memory-bandwidth floor**, and is 12.5 tok/s decode near it or
   far from it? (§1.5.2, §7 item 2) — without this the perf picture has no
   denominator.
3. **Why is Q6_K dequant non-deterministic on gfx11?** (§1.5.4) — contained, root
   cause unknown, eight hypotheses already falsified.
4. **Can `/root/lh/sglang_src` be reconciled with this branch at all**, given it
   is not a git repo and carries no commit id? (§7 item 4) — if not, the port
   may have to be re-derived from the extension source in `/root/lh/ggufbuild`.
5. **Does `--cpu-offload-gb` compose with GGUF?** (§6.1) — now testable, since a
   working backend exists. Q4_K_M runs with only ~3 GiB of slack, so this is the
   difference between comfortable and tight.
6. **Dequant scratch at the intended prefill batch size** (§9.3) — still the one
   budget term never priced.
7. **Is the vision tower worth fixing now** (#651b, §9.4)? §1.5.3 turned this from
   a nice-to-have into the concrete source of the headroom the spec arm lacks.

---

## 12. Phase 2 findings (2026-08-08, the boot session) [MEASURED]

### 12.1 COHERENCE VERDICT: the "working" Q4_K_M serving was NOT coherent

`probe.py` against the proven no-spec recipe: **2/6 then 1/6 content-correct,
greedy rounds differ** — fluent, grammatical, wrong, exactly the predicted
failure class. First-token top-4 logprobs differ across 8 identical greedy
requests (8/8 distinct), run-to-run logit wobble ~1e-1 — large enough to flip
argmax. The ~12.5 tok/s serving of §1.5.2 was serving noise. All evidence in
`/root/651-p2/results/` on the laptop.

### 12.2 Root cause: the Q6_K defect family is wider than the containment

8-run fixed-input harnesses (`det_mm.py`, `det_moe.py`, `det_moe_gemm.py` in
`/root/651-p2/scripts/`), all on real block fixtures:

| op | Q4_K | Q5_K | Q6_K |
|---|---|---|---|
| dequantize | clean | clean* | **broken** (known) |
| MMVQ (GEMV) | clean | clean | **non-det, spread 2.3e-02** |
| MMQ (GEMM) | clean | clean | clean (the one validated op) |
| ggml_moe_a8 | clean | clean | **non-det, spread 3.4e-02** |
| ggml_moe_a8_vec | clean | clean | **non-det, spread 1.1e-01** |

The checkpoint's four Q6_K tensors are the lm_head (MMQ-pinned, contained) and
**three MoE expert down-proj stacks (blk.34/38/39)** — those three run through
the defective MoE kernels on EVERY forward. That is the incoherence.

Debug battery re-run (predecessor's q6k3/q6k4/q6k5, outputs now saved):
kernel writes ALL output elements (not a coverage bug); corruption is in
device memory, not readback; syncs/fixed-buffer/chunked launches don't help;
**AMD_SERIALIZE_KERNEL=3 does not help** (intra-kernel, not inter-kernel).
O-level dependence: the O1 build breaks Q5_K too; predecessor's native gfx1103
build likewise. Codegen-sensitive → instruction-pattern-level misexecution.

### 12.3 (*) Suspend/resume poisons the GPU: reboot before believing anything

The laptop suspended overnight and resumed 06:42; post-resume, **Q5_K
dequantize went non-deterministic too (2e-02) even idle**, and yesterday's
clean Q5_K baseline could not be reproduced — until a REBOOT restored it
(3x8 runs byte-identical again). Operational law for this machine: **after
any suspend/resume, reboot before serving or measuring.** The defect family
is marginal-execution-like; resume state widens it.

### 12.4 Containment shipped: a Q6_K-free derived checkpoint

`docs/dev/651/requant_no_q6k.py` requantizes exactly the four Q6_K tensors to
Q8_0 (validated byte-identical on this GPU; +~250 MiB; requant error ~5e-4,
two orders below the removed noise). Derived file on the laptop:
`/root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf`. Prediction it must
satisfy: coherent AND greedy-deterministic serving. If NEXTN's HIP fault was
Q6_K-driven (the draft shares the Q6_K lm_head), the spec arm may heal too.

### 12.5 Recovery completed: 7 more unversioned laptop files secured

The first recovery captured only `gguf.py`. mtime sweep + git-blob provenance
found seven more unique laptop-authored files — including the load-bearing
early expert-stack materialization (without it Q4_K_M cannot load at all on
unified memory) and the malloc_trim arena release. All verbatim + SHA-256 in
`docs/dev/651/recovered/laptop_sglang_delta/` (commit 0200388983).

### 12.6 DESIGN DIRECTIVE (user, 2026-08-08): disk-park across the phase flip

For the PP/reshard phase flip on the APU: anything PHASE-EXCLUSIVE that must
survive the flip (inactive phase's graph pools/workspaces, CPU-stage buffers
during decode, drafter state during prefill, unused vision tower) parks on
**DISK, not RAM** — GTT and system RAM are the same DDR5 here, so every byte
parked is direct headroom for the resharding; small contiguous blobs re-read
fast. Per item decide by MEASURED ms among (a) keep resident, (b) disk-park +
sequential reload, (c) drop + reconstruct (e.g. re-capture graphs); write the
table down; flip budget = park-write ms + reload ms. Reuse the #286 offload
register classes, #89 hibernate VRAM-to-disk with #456 sparse-write, #407
registry doctrine — no new spill path. Explicit named park files, never the
8 GB swap. Flips are regime-wide and rare, so write volume is negligible.
