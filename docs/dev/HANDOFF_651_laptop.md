# Handoff #651 — Qwen3.6-35B-A3B GGUF bring-up on the APU laptop

Branch: `feat/gguf-q4-bringup-651`
Revised 2026-08-08 by the #651 strand, after the user supplied the real target
hardware. The 2026-08-07 revision of this file assumed a discrete-NVIDIA laptop
and is wrong in its conclusions; the geometry, the #647 fix and the parallelism
analysis survive unchanged and are kept below.

---

## 0. The headline, before anything else

**The bring-up as specified cannot run on this laptop with this tree today.**
Both possible backends are closed, each for an independent and fully-evidenced
reason:

| Path | Status | Why |
|---|---|---|
| **iGPU via ROCm/HIP** | **Closed** | The GGUF K-quant kernels are not in the ROCm build at all — not broken, absent. And `sgl-kernel`'s ROCm build refuses this GPU architecture outright. §2 |
| **CPU-only** | **Closed** | No CPU K-quant kernel exists, so a CPU stage must materialize weights dense: **~67 GiB against 29.5 GiB of RAM**, a 2.3x overshoot. §3 |

This is not a tuning problem and no runtime flag changes it. What *is* true, and
is the constructive half of this handoff: the ROCm port is **mechanically
shallow** (§2.4) — the kernels contain no CUDA-only hardware intrinsics, the
AMD shims are already written and inherited from upstream, and the hardcoded
wave size is already correct for this GPU. The blocker is build-system
plumbing and an architecture allow-list, not kernel physics.

**So the honest next step is not "boot it". It is "port the GGUF kernels to
ROCm and widen the gfx allow-list" (§7).**

### Verification markers used throughout

| Marker | Meaning |
|---|---|
| **[MEASURED]** | Executed on the real target machine or the real checkpoint. Command given. |
| **[CODE]** | Verified by reading this tree, with file:line. |
| **[UNVERIFIED]** | Reasoning or arithmetic only. Treat as hypothesis. |

The model has still **never generated a token** on this hardware. Nothing below
claims it is coherent.

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
halves the weight budget. It is the obvious vehicle for proving a pipeline works
before spending the full Q4 budget on it — quality is materially worse, so it is
a bring-up instrument, not the destination.

`config.json`, `tokenizer.json` and `chat_template.jinja` are present beside
them, which is what `--tokenizer-path` needs.

---

## 2. Backend verdict: GGUF has no ROCm path in this tree [CODE]

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

Expected on any ROCm build of this tree: `hip= 7.x  cuda= None` and
`ggml op registered: False`. `False` means GGUF is unusable on that machine and
no flag will change it — the kernel is not in the binary.

If `sgl-kernel` cannot even be installed, that is `setup_rocm.py:77-81` exiting
on the gfx gate (§2.3.1) — a prior blocker with a different fix.

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

## 3. The CPU-only fallback is physically impossible here [MEASURED]

Not "slow" — impossible, by arithmetic that needs no hardware.

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

## 4. Memory budget, if and when a backend exists [MEASURED]

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

## 6. Achievable-goal restatement

Given §2 and §3, the goal must be restated in two tiers.

### 6.1 What is achievable once a backend exists

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

### 6.2 PP + speculation is closed tree-wide, and is not this strand's job

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

## 7. What would actually unblock this laptop, in order

1. **Widen the `sgl-kernel` ROCm gfx allow-list** (`setup_rocm.py:77-81`) to
   admit RDNA3 / gfx110x, and get a build to complete. Smallest item; blocks
   everything else. Verify with §2.6.
2. **Add `csrc/quantization/gguf/gguf_kernel.cu` to `setup_rocm.py`'s source
   list**, hipify the three CUDA headers in `gguf_kernel.cu:3-5`, and register
   the ops in `common_extension_rocm.cc`. §2.4 says the kernel bodies themselves
   should largely survive: the `USE_ROCM` shims and `__dp4a` are already there
   and `WARP_SIZE_GGUF 32` is already right for wave32.
3. **Bind the Python path for HIP** — `gguf.py:41-62` needs an `elif _is_hip:`
   import arm, and `supports_current_device()` (`:134-152`) must stop returning
   `None` on ROCm so the failure is loud at load instead of a `NameError`
   mid-forward. **Do this even if nothing else is done**, because the current
   silent-load-then-`NameError` behaviour will waste somebody's day.
4. **Validate numerics on RDNA** — `__builtin_amdgcn_sdot4` paths and the
   CDNA-tuned `moe.cuh` constants (§2.4).
5. Only then: the staged boot of §8.

Item 3 is cheap and is worth landing on its own merits regardless of whether
anyone finishes the port.

---

## 8. Staged boot, for when there is a backend

`docs/dev/651/boot.sh` is **NVML/CUDA-only** — it resolves devices via
`pynvml` UUIDs, exports `CUDA_VISIBLE_DEVICES` and points `LD_LIBRARY_PATH` at
`nvidia/cu13/lib`. **None of that applies to an APU** and it must be rewritten
for ROCm (`rocminfo` / `HIP_VISIBLE_DEVICES` / `HSA_OVERRIDE_GFX_VERSION`)
before use. Its *staging logic* is still the right shape:

| Stage | Config | Answers |
|---|---|---|
| **a** | TP=1, no spec, eager, **Q2_K_XL** | loader / kernels / checkpoint |
| **b** | TP=1, NEXTN spec, Q2_K_XL | the #647 router-gate fix (§9.1) |
| **c** | TP=1, NEXTN spec, **Q4_K_M** | the real memory budget (§4) |
| **d** | + `--cpu-offload-gb` | does offload compose with GGUF (§6.1) |

Start on **Q2_K_XL**, not Q4 — it halves the budget and isolates "does anything
work at all" from "does it fit". This inverts the 2026-08-07 staging, which had
no smaller checkpoint available to it.

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

1. **Does anyone want to do the ROCm port (§7)?** That is the real decision.
   Until it is answered, #651 has no path on this machine.
2. **Does `--cpu-offload-gb` compose with GGUF?** (§6.1) — cheap to test once a
   backend exists, and it is the difference between Q4 fitting comfortably and
   fitting with 3.1 GiB of slack.
3. **Dequant scratch at the intended prefill batch size** (§9.3) — the one
   budget term still unpriced.
4. **Is `HSA_OVERRIDE_GFX_VERSION=11.0.0` needed** for the rest of the ROCm
   stack on gfx1103, and does it interact with a gfx1103-targeted `sgl-kernel`
   build? (§2.6)
5. **Is the vision tower worth fixing now** (#651b, §9.4) rather than later, given
   the tight Q4 budget?
