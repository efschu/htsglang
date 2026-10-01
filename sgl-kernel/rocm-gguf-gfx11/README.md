# sglang_gguf_rocm for gfx11 (Radeon 780M / gfx1103) — the fixed build

Source of the standalone GGUF kernel extension that serves Qwen3.8-35B-A3B on
efeu-TP14 (2026-10-01). Byte-identical to `../recovered-rocm-gguf/` (the laptop's
`/root/lh/ggufmod`, recovered 2026-08-08) except for the fixes below.

Build on the laptop (ROCm 7.1, Ubuntu clang 21, torch 2.10+rocm7.0):

    AMDGPU_TARGET=gfx1100 PYTORCH_ROCM_ARCH=gfx1100 python setup.py build_ext --inplace

and run with `HSA_OVERRIDE_GFX_VERSION=11.0.0` (torch ships no gfx1103 code objects).

## Fixes

1. **real-true16 miscompile (setup.py).** clang 21 compiles gfx11 with real-true16:
   16-bit values live in VGPR halves. In the Q6_K dequantize kernel the scale load
   `global_load_d16_b16 v0` (writes v0.l) is still outstanding when
   `v_lshlrev_b16 v0.h, ...` writes the other half; when the load returns in that
   window the VALU write is lost. Result: Q6_K dequant/MMVQ/MoE-vec
   nondeterministically wrong (20/20 launches, up to 1.6 and Inf), Q5_K rarely.
   This was the #651 "per-launch fault". setup.py passes
   `-Xclang -target-feature -Xclang -real-true16`; the gfx11-generic target (which
   has no true16) is equally clean. The override (11.0.0 vs 11.0.2) is NOT the cause.
2. **WARP_SIZE host/device mismatch (mmvq.cuh, moe_vec.cuh).** `include/utils.h`
   defines WARP_SIZE as 64 in host passes on ROCm and 32 in gfx11 device passes.
   The tuned K-quant MMVQ (#73, nvecs 2..8) therefore launched 64*nwarps threads
   against `__launch_bounds__(nwarps*32)` -> launch failure for every dense K-quant
   linear at M=2..8; legacy MMVQ and MoE-vec ran a second, discarded wave per row.
   Both files now use WARP_SIZE_GGUF (32 on host and device; equal to WARP_SIZE on
   CUDA, so CUDA builds are unchanged).
3. **moe_vec skips expert id < 0.** `mask_cpu_expert_ids` (kt CPU experts) sets
   CPU-owned ids to -1; the vec kernel indexed the expert table with it
   (hipErrorIllegalAddress in decode-graph capture, #655). It now contributes 0.
4. **Capability marker `ggml_rocm_true16_safe` (binding.cpp).** gguf.py drops its
   ROCm Q6_K dequant containment (CPU-dequantised dense lm_head) only when this op
   exists, so an older build keeps the containment.

Evidence and harnesses: `docs/dev/651/efeu35q3/` (q6k_rootcause.py,
dequant_types_oracle.py, small_rows_repro.py, results/).
