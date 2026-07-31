# Task #327 pre-stage: INT8 on sm120 -- port, wheel, verification

> **Status after #353.** The port is fork SOURCE and stays in the tree; the
> wheel below is one build of it, kept for its provenance record, not a
> deliverable. No wheel is shipped or vendored, and the container keeps
> installing sgl-kernel from PyPI on purpose. Reasoning:
> `ANALYSE_319_int8_lane.md` section 5d. Build/install/rollback recipe for a
> rig: `docs/rig-runbook.md` section 6.6.

Closes the gap recorded in `ANALYSE_319_int8_lane.md` section 2b: `int8_scaled_mm`
had no dispatch arm for the 5090, so a W8A8 INT8 checkpoint crashed that rank on
its first forward. Code change: commit `7da6f0cb2f`. Everything below was done
without a GPU (`CUDA_VISIBLE_DEVICES=99`); numerical correctness and speed on the
5090 belong to the #327 boot window.

## 1. What vLLM actually has

vLLM has **no** CUTLASS INT8 kernel for any Blackwell part. Both Blackwell c3x
dispatchers pass a null int8 functor:

```cpp
// vllm/csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu
dispatch_scaled_mm(c, a, b, a_scales, b_scales, bias,
                   vllm::cutlass_scaled_mm_sm120_fp8,
                   nullptr,  // int8 not supported on SM120
                   vllm::cutlass_scaled_mm_blockwise_sm120_fp8);
```

`c3x/scaled_mm_helper.hpp` turns that into `"Int8 not supported on SM<n>. Use FP8
quantization instead, or run on older arch (SM < 100)."`, and `scaled_mm_entry.cu`
routes everything `>= 120` there, so vLLM's INT8 c2x path is unreachable on
Blackwell. There was no sm120 INT8 config to port.

What *is* portable is the CUTLASS 2.x IMMA path vLLM uses for Ada
(`scaled_mm_c2x_sm89_int8_dispatch.cuh`), which this fork already carries as
`sm89_dispatch_shape` -- the file's own comment names that file as its source.
Retargeting it at sm120 holds because:

* sm120 keeps the classic warp-level IMMA (`mma.sync.aligned.m16n8k32.s8`) that
  those kernels emit under the `Sm80` arch tag; only the sm100 tcgen05 path
  dropped it, which is exactly what vLLM's null functor encodes.
* sm120's per-block shared memory ceiling (100 KB) matches sm86/sm89, not sm80's
  160 KB, so the sm89 tile/stage table is the right budget fit.

sm100/sm103 stay unimplemented on purpose.

## 2. The correction to ANALYSE_319 section 2c

That section priced the native branch as "a materially bigger and higher-risk
undertaking" than a dequant fallback lane. It is not. The **SASS was already
shipping**: the installed `sm100/common_ops.abi3.so` contains 120
`GemmWithEpilogueVisitor` (INT8 c2x) kernels = 30 per arch across its four
arches, sm_120 included, because the templates are instantiated from the sm86
arm regardless of which arch the host code will dispatch to. The gap was a
missing host-side `else if`. Total change: one forwarding template plus a branch.

## 3. Wheel provenance

| item | value |
| --- | --- |
| wheel | `/spinning/wt-327a-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl` |
| sha256 | `e7b16e1d74527ba070afeaf7bab58ed5df0fadbeb344d0fb372ff334f7e15b54` |
| size | 21 358 348 B |
| source | branch `feat/int8-sm120-port`, tree at commit `7da6f0cb2f` |
| build script | `/spinning/wt-327a-build.sh` (log: `/spinning/wt-327a-build.log`) |
| arch list | `SGL_KERNEL_LIMIT_CUDA_ARCHS=86;120` -- rig cards only (3080, 5090) |
| variants | `SGL_KERNEL_SKIP_SM90_VARIANT=ON`, `SGL_KERNEL_ENABLE_FA3=OFF` |
| nvcc | 12.9.86 (`/usr/local/cuda-12.9`) |
| torch | 2.11.0+cu130, from `/spinning/htsglang-gpu/.venv` (read-only) |
| cmake | 4.4.0 from that venv -- system cmake 3.28 is too old for `CMP0169`/`CMP0177` |
| ccache | `/spinning/wt-327a-ccache`, created empty for this build: 91 cacheable calls, 0 hits, 91 misses -- no foreign cache contributed an object (task #304 lesson) |
| parallelism | `-j4` / `MAX_JOBS=4`, one nvcc thread per TU (swapless host) |
| build time | 44 min compile+package (configure done 10:42:47, wheel 11:26:44) |

The build venv `/spinning/wt-327a-buildvenv` holds only `scikit-build-core`
(plus pytest and clang-format for the checks); torch, cmake and ninja come from
the GPU venv via `PYTHONPATH`. The GPU venv itself was not modified.

## 4. Verification without a GPU

| check | result |
| --- | --- |
| single-TU probe compile for `sm_120a` | clean (advisory ptxas notes only, from the pre-existing sm90 c3x code) |
| arch coverage of `sm100/common_ops.abi3.so` | 52 TUs x {sm_86, sm_120}, nothing else |
| INT8 c2x kernels in the wheel | 60 = 30 per arch |
| IMMA instructions in the sm_120 SASS | 2000 |
| dispatch discriminator | new `.so` carries `"No implemented int8_scaled_mm for compute capability sm"`, the installed one carries the old `"... for current compute capability"`; neither carries the other |
| registered op schemas vs installed wheel | 79 vs 79, no op added or lost -- drop-in |
| Python import, no GPU | `import sgl_kernel` and `torch.ops.sgl_kernel.int8_scaled_mm` load; schema `(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype, Tensor? bias) -> Tensor` |
| `tests/test_int8_gemm_dispatch.py` | 5 passed; 3 of them fail against the pre-change source (falsified) |
| ruff check / ruff format / codespell | clean on the touched Python; clang-format (`--style=file`) applied to the `.cu` |

The `.so` is 61.7 MB against the installed 15.1 MB. Both are stripped and the op
schema sets are identical; the delta comes from the different arch sets and the
kernels added to the tree since the installed wheel was built on 2026-07-17.

## 5. Install recipe for the #327 boot window

The main venv is deliberately left untouched until then. In the window:

```bash
V=/spinning/htsglang-gpu/.venv
SO=$V/lib/python3.12/site-packages/sgl_kernel/sm100/common_ops.abi3.so
cp "$SO" /spinning/wt-327a-wheel/common_ops.abi3.so.pre327   # rollback copy

$V/bin/pip install --force-reinstall --no-deps \
  /spinning/wt-327a-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl
```

The wheel is package `sglang-kernel` 0.4.4 and owns exactly the four objects the
currently installed 0.4.4 owns (`sm100/common_ops`, `flashmla_ops`, `infllm_ops`,
`spatial_ops`). `flash_ops.abi3.so` (FA3) belongs to `sgl_kernel` 0.3.21 and is
not touched -- which is why the build keeps `SGL_KERNEL_ENABLE_FA3=OFF`.

Post-install smoke check:

```bash
$V/bin/python -c "
import torch, sgl_kernel
print(torch.cuda.get_device_capability())
print(torch.ops.sgl_kernel.int8_scaled_mm.default._schema)"
```

Rollback: reinstall the previous wheel, or copy `common_ops.abi3.so.pre327` back
over the installed object.

Then, on the 5090 rank:
`pytest sgl-kernel/tests/test_int8_gemm.py` -- it no longer skips there.
