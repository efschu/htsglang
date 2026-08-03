# TICKET #511 — wheel build for the #512/#518 kernel bundle

**Status: NOT BUILT. Deliberately.** The translator serving boot owns the
cards and the RAM; §"Build-Parallelität begrenzen" and the RAM/page-cache rule
forbid a CUDA build next to a RAM-near GPU load. This ticket is the command
block for the next quiet slot (window 5).

Branch: `fix/kernel-bundle-511` off `96c7dc5d2b`.

## 1. What needs a rebuild, and what does not

| task | change | rebuild? |
| --- | --- | --- |
| #511 | test-only: derived MXFP4 tolerances + the off-GPU instrument | **no** |
| #512 | `moe.cuh` — `exp_stride` widened to `int64_t` (23 declarations) | **yes** |
| #518 | `common_extension.cc` / `common_extension_musa.cc` — three no-tensor GGUF probes re-registered catch-all | **yes** |

So the #511 half is testable and CI-visible today; the other two are inert in
the installed wheel until this ticket runs. Until then:

* #512 stays latent exactly as before — the wrap needs a >2 GiB per-rank
  per-layer expert tensor, which TP=3 expert sharding keeps out of reach on
  this rig (86 local experts x 9.44 MB = 0.81e9 B against the 2.147e9 ceiling).
* #518 stays worked around by the Python mirror
  `_ggml_moe_get_block_size` (`layers/quantization/gguf.py`), which is why the
  serving path never saw it and only the Gate-A test did. **The mirror stays
  after the rebuild** — the wheel ships prebuilt and sha-pinned, so a tree
  carrying #518 can still be running a pre-#518 wheel.

## 2. Command block

Copy `/spinning/wt-398-build.sh` and change only `SRC`, `OUT`, `BUILD_DIR`,
`CCACHE_DIR` — the knobs must stay identical to #398/#436 or the wheel is not
comparable with the pinned one.

```bash
# Preconditions, all three:
#   - no serving boot holds the cards or the RAM (this is a CPU build, but it
#     is a RAM-heavy one on a swapless host)
#   - nothing maps sgl_kernel:  grep -l sgl_kernel /proc/*/maps   -> empty
#   - the GPU arb is free (not needed for the build, needed for the Gate-A run)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=99

SRC=/spinning/wt-511-kernel-bundle/sgl-kernel
GPU_VENV=/spinning/htsglang-gpu/.venv
GPU_SP="${GPU_VENV}/lib/python3.12/site-packages"
CU13="${GPU_SP}/nvidia/cu13"
BUILD_VENV=/spinning/wt-327a-buildvenv
OUT=/spinning/wt-511-wheel
BUILD_DIR=/spinning/wt-511-build
export CCACHE_DIR=/spinning/wt-511-ccache
mkdir -p "${CCACHE_DIR}" "${OUT}" "${BUILD_DIR}"
ccache -M 20G >/dev/null 2>&1 || true

# Swapless host: hard cap on build parallelism (project rule MAX_JOBS/-j4).
export MAX_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4

export CMAKE_EXECUTABLE="${GPU_SP}/cmake/data/bin/cmake"
export CMAKE_GENERATOR=Ninja
export CUDA_HOME="${CU13}"
export CUDAToolkit_ROOT="${CU13}"
export PATH="${CU13}/bin:${PATH}"
export LD_LIBRARY_PATH="${CU13}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${GPU_SP}"
cd "${SRC}"

"${BUILD_VENV}/bin/python" - <<'PY'
import sys
from scikit_build_core.build import build_wheel

gpu_sp = "/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages"
cu13 = f"{gpu_sp}/nvidia/cu13"
config_settings = {
    "build-dir": "/spinning/wt-511-build",
    "cmake.define.CMAKE_CUDA_COMPILER": f"{cu13}/bin/nvcc",
    "cmake.define.CUDAToolkit_ROOT": cu13,
    "cmake.define.CMAKE_PREFIX_PATH": f"{gpu_sp}/torch/share/cmake",
    "cmake.define.SGL_KERNEL_LIMIT_CUDA_ARCHS": "86;120",
    "cmake.define.SGL_KERNEL_SKIP_SM90_VARIANT": "ON",
    "cmake.define.SGL_KERNEL_ENABLE_FA3": "OFF",
    "cmake.define.SGL_KERNEL_COMPILE_THREADS": "1",
}
name = build_wheel("/spinning/wt-511-wheel", config_settings)
print(f"BUILT_WHEEL={name}", file=sys.stderr)
PY
echo "BUILD_DONE rc=$? $(date -Is)"
```

## 3. Pre-install checks (no GPU needed)

```bash
W=/spinning/wt-511-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl
sha256sum "$W"                       # -> record; replaces 67f03cfa... in the runbook
stat -c %s "$W"                      # -> record; #398 was 16 638 372 B
unzip -l "$W" | wc -l                # -> same 39-file set as the pinned wheel
python3 -c "import zipfile;z=zipfile.ZipFile('$W');print(len(z.namelist()))"
# CUDA major must stay 13 (the #436 ABI trap):
mkdir -p /tmp/w511 && unzip -oq "$W" -d /tmp/w511
objdump -p /tmp/w511/sgl_kernel/common_ops.abi3.so | grep -E 'libcud|libcublas'
# expect .so.13 only, no .so.12
```

## 4. Install

Follow rig-runbook "Making it durable" verbatim. Check `/proc/*/maps` for
`sgl_kernel` FIRST — the #398 install was deferred once for exactly this.

Then update the runbook §2.1 pin table in the same change:

| field | new value |
| --- | --- |
| wheel | `/spinning/wt-511-wheel/sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl` |
| sha256 | *(from step 3)* — replaces `67f03cfa755efa01498c7732bd6ae015ec5673feffe9a51452fefdbe0dcd4664` |
| source | `sgl-kernel/` at `fix/kernel-bundle-511` |
| carries | #398 MXFP4 kernels + #512 int64 expert stride + #518 catch-all probes |

## 5. Acceptance: Gate A 14/14, per arch

Gate A is `test/registered/unit/quantization/test_gguf_mxfp4_cuda.py`. Window 4
ran it for the first time against a real wheel and got **12/14 on both sm86 and
sm120**; the two failures were the same on both, and both are #518:

* `TestMXFP4MoE::test_moe_block_size_is_registered` — calls
  `ggml_moe_get_block_size(MXFP4)` directly.
* `TestMXFP4MoE::test_moe_mmq_matches_a_per_expert_reference` — calls it to get
  the tile width before `moe_align_block_size`.

Both raised `NotImplementedError: There were no tensor arguments to this
function ... but no fallback function is registered`, in the dispatcher, before
any kernel. #518 removes that. **Expected after this wheel: 14/14 on each
arch**, with no change to the 12 that already passed.

```bash
# per arch, holding the GPU arb, one card visible at a time
CUDA_VISIBLE_DEVICES=<3080_idx> PYTHONPATH=/spinning/wt-511-kernel-bundle/python \
  python3 -m pytest -q test/registered/unit/quantization/test_gguf_mxfp4_cuda.py
CUDA_VISIBLE_DEVICES=<5090_idx> PYTHONPATH=/spinning/wt-511-kernel-bundle/python \
  python3 -m pytest -q test/registered/unit/quantization/test_gguf_mxfp4_cuda.py
```

Record both in `docs/dev/TICKET_398_mxfp4_validation.md`.

**The #511 risk to watch, stated in advance.** The 12 numerical passes were
green under `atol=1.5, rtol=3e1` (and `rtol=1e4` on the MMQ arms), a predicate
so wide that an all-zeros output satisfied it. They are now gated by
`SIGMA_MULTIPLIER * max sigma` from `activation_quant_sigma()` — about
1270x tighter on the m=1 shape. The tolerance is derived, not guessed:
the model predicts `err / max|ref| ~= 4.8e-3`, against the ~5e-3 this file's
own docstring recorded for MMVQ-vs-fp32 before any of this, and the off-GPU
`TestToleranceInstrument` shows a correct kernel passing at every shape the
file uses with >=1.5x headroom. If an arm still reddens, the number to report
is `|out - ref|.max() / activation_quant_sigma(x, w).max()` — if that ratio is
above 8 the model is missing a term (look at MMQ first: its old tolerance was
loosened separately to `rtol=1e4`, which may mean someone once saw more than
the activation noise there), and if it is below 8 the failure is real.

Two quick one-liners for the same window, both new since #518:

```bash
python3 -c "import torch,sgl_kernel;print(torch.ops.sgl_kernel.ggml_moe_get_block_size(12))"   # expect 4 (CUDA), not a raise
python3 -c "import torch,sgl_kernel;print(torch.ops.sgl_kernel.ggml_mxfp4_native())"           # expect 1, not a raise
```

## 6. What this ticket does NOT cover

* No serving boot, no perf number. #512 and #518 are correctness-only; neither
  changes any hot path (a widened 64-bit multiply per CTA, and a dispatch key).
* The #512 falsifier is static + arithmetic
  (`test_gguf_moe_stride_width_512.py`) and stays valid without the wheel. An
  end-to-end demonstration would need a >2 GiB per-rank expert tensor, i.e. a
  256-expert Q4_K GGUF at TP=1 — not reachable on this rig and explicitly not
  worth a synthetic allocation.
