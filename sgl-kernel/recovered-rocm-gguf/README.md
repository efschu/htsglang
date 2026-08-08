# Recovered: standalone ROCm build of the GGUF K-quant kernels

**Status: RECOVERED VERBATIM, NOT YET RECONCILED.** This directory is a rescue
copy, not an integrated build target. Nothing in the normal `sgl-kernel` build
references it yet. Reconciliation into `setup_rocm.py` /
`common_extension_rocm.cc` is tracked as #651 work-queue items 5-6
(`docs/dev/HANDOFF_651_laptop.md` §7).

## Provenance

Recovered 2026-08-08 from **unversioned** working directories on the laptop
`efeu-TP14` (192.168.0.116), authored during the session of **2026-08-07** by
the prior #651 strand, which was killed by the token outage before it could
commit anything.

`/root/lh/` on that machine is **not a git repository** and carries no commit
id, so this was the only copy of the work. Sources came from
`/root/lh/ggufbuild/` and `/root/lh/ggufmod/` (two iterations of the same tree;
they differ only in the extension name). Laptop originals were **copied out
read-only and left untouched**.

### SHA-256 of the laptop originals, so reconciliation stays possible

```
18c116449712c08e0b332cfba1630a25bbc55d03f502b3db6302848f8f9df5bf  ggufbuild/binding.cpp
d3ae9a8d4d3cca8896d0e064711c98b216b0d58feb9b987e0b2966e7b6ed4554  ggufbuild/setup.py
65942c0e89ee223ba91f153ea1cb93c91d59623c78279ecb84a1dda98bdd41ba  csrc/quantization/gguf/dequantize_hip.cuh
698fe064800e0630d293f4ff4e88c992a4d9bb79bdec83e53b32e6c1ffdc606e  csrc/quantization/gguf/ggml-common_hip.h
613d5471176ab721b16be80871292804841bd100eb39836deda1d0010744a964  csrc/quantization/gguf/mmq_hip.cuh
613191ae2e570540064262f34b320733434511e3b048031d430b3c3e29693560  csrc/quantization/gguf/mmvq_hip.cuh
4330103d5d12ad41681991cbcca2566715543a0a9da8ae7c56871566e15ff0af  csrc/quantization/gguf/moe_hip.cuh
b661608fa1596b8b9d7d097fe8d053a2309c0948f0d02ed56bc9af5bd4bc6ed5  csrc/quantization/gguf/moe_vec_hip.cuh
e7bcd8989ebdf6741c5c4f6ca8eef2c6078c9c4977543e380b21df815b70e382  csrc/quantization/gguf/vecdotq_hip.cuh
7e7152bc4e1113568770e41415786dd96b4fc6446289eb151732bae8021aea81  csrc/quantization/gguf/gguf_kernel.hip
550798f65fa64f421138f6a5177be3dc83a14f3a3763eaa6d7953bfe0ef95b8d  sglang_src/.../quantization/gguf.py
```

The `gguf.py` delta is kept separately as
`docs/dev/651/recovered/gguf.py.laptop.patch` (6 hunks, +233/-8 against this
branch's `python/sglang/srt/layers/quantization/gguf.py`).

## What this proves

It **works**: built against ROCm 7.1 / torch 2.10.0+rocm7.0 on a Radeon 780M,
this extension is what lets Qwen3.6-35B-A3B Q4_K_M GGUF serve on that iGPU at
~12.5 tok/s decode (HANDOFF §1.5.2).

The exclusion of the GGUF sources from `setup_rocm.py` was **missing build
wiring, not a genuine porting gap** — which is exactly the question
`setup.py`'s docstring set out to answer. The sources already carry `USE_ROCM`
branches (`vecdotq.cuh`'s `__dp4a` guards, `mmq.cuh`/`moe.cuh` tile sizes and
`__launch_bounds__`) and torch's HIP extension path defines `-DUSE_ROCM=1`.

## Build

```bash
# name is the only difference between the two recovered iterations:
#   gguf_rocm_probe   (ggufbuild)  -- the feasibility probe
#   sglang_gguf_rocm  (ggufmod)    -- the one gguf.py actually imports
AMDGPU_TARGET=gfx1103 python setup.py build_ext --inplace
```

`AMDGPU_TARGET` defaults to `gfx1103` in the recipe, but the shipped artifact
was built for **gfx1100** and run under `HSA_OVERRIDE_GFX_VERSION=11.0.0`.

**`HSA_OVERRIDE_GFX_VERSION=11.0.0` is load-bearing, not cosmetic.** Without it
a bare 64x64 matmul fails with `hipErrorInvalidDeviceFunction`, because torch's
ROCm wheels carry no gfx1103 code objects. With it, `gcnArchName` reports
`gfx1100`. Cap build parallelism (`MAX_JOBS=4-6`) — the laptop is 8C/16T with
only 8 GiB of swap.

## KNOWN CORRECTNESS BUG: Q6_K dequant is wrong on gfx11

**Do not drop the containment.** From the recovered `gguf.py`:

> Q6_K dequantise returns non-deterministically wrong values on gfx1103
> (Radeon 780M): eight runs on one fixed input tensor differ from each other,
> worst max|d| 5.8e-01 against the numpy reference, up to 75 non-finite values
> in 262144, and it is wrong on the FIRST call in a fresh process. Q4_K and
> Q5_K are byte-identical across the same eight runs.

Eight hypotheses were falsified, including that it is an `HSA_OVERRIDE`
artefact — **a native gfx1103 build is worse, Q5_K becomes affected too**.

Containment, which is in the recovered `gguf.py` and must survive
reconciliation:
- `ggml_mul_mat_a8` (MMQ) **is** validated correct for Q6_K on gfx1103
  (max|d| 5.5e-04 vs the numpy reference on real weights), so Q6_K is pinned to
  MMQ at any token count.
- The load-time path is rescued by an exact one-time **CPU** dequantise. This
  matters concretely: Qwen3.6-35B-A3B ships a **Q6_K `lm_head`**.

Scope is narrow — 4 tensors of 753 — but root cause is **unknown**.

`tests/` carries the investigation harness that established all of the above:
`test_kquant.py`, `test_correctness.py`, `determinism_all.py`,
`oneshot_q6k.py`, `debug_q6k{,2,3,4,5}.py`, plus real GGUF block fixtures
`slice_q4_K.bin` / `slice_q5_K.bin` / `slice_q6_K.bin` and `slices.json`.
Those fixtures are real blocks from the checkpoint, which is what makes the
comparison against the numpy reference meaningful.
