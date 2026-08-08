#!/usr/bin/env python
"""#651: does hipBLAS bf16 GEMM fail on this APU at prefill shapes?

HOW WE GOT HERE. Three serving crashes reported `unspecified launch failure` at
`moe_align_kernel.cu:530`. Re-running with `AMD_SERIALIZE_KERNEL=3`, so an
async HIP fault is attributed to the kernel that actually raised it, moved the
blame somewhere else entirely:

    srt/layers/quantization/gguf.py:1122, in fused_mul_mat_gguf
    RuntimeError: CUDA error: HIPBLAS_STATUS_INTERNAL_ERROR when calling
      hipblasGemmEx(..., HIP_R_16BF, ..., HIPBLAS_GEMM_DEFAULT)

moe_align was the messenger, not the culprit. Line 1122 is `y = x @ weight.T`,
the LARGE-BATCH branch of the GGUF path: dequantize the whole weight once, then
one big GEMM. Small batches take the MMQ branch (`ggml_mul_mat_a8`) instead.

That predicts the exact pattern observed all day: decode (M=1) survived a full
60 s bench, while every prefill sweep died within seconds -- prefill is what
takes the GEMM branch. It also explains why an earlier rocBLAS check looked
clean: it tested **fp16**, and serving runs **bf16** (HIP_R_16BF).

THE TEST. Same shapes, same operation, fp16 vs bf16, M swept from decode-sized
to prefill-sized. If bf16 fails where fp16 succeeds, the blocker is the bf16
GEMM path and the fix is to route this GEMM through fp16 (or another backend)
on this device -- not to touch the MoE kernel at all.

Each dtype runs in a SEPARATE process (`--dtype`), because a HIP internal error
poisons the context and anything measured afterwards is meaningless.
"""

import argparse
import sys
import traceback

import torch

# (N, K) pairs taken from this checkpoint's serving shapes.
SHAPES = [
    (2048, 4096),
    (512, 2048),
    (2048, 5120),
]
M_VALUES = [1, 8, 16, 64, 256, 512, 1024, 2048]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"dtype={args.dtype}  device={torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).gcnArchName})")

    failures = 0
    for (N, K) in SHAPES:
        # weight as stored: [N, K]; the call under test is x @ weight.T
        weight = torch.randn(N, K, device="cuda", dtype=dtype)
        for M in M_VALUES:
            x = torch.randn(M, K, device="cuda", dtype=dtype)
            try:
                for _ in range(args.repeats):
                    y = x @ weight.T
                torch.cuda.synchronize()
                ok = tuple(y.shape) == (M, N)
                print(f"  [ok ] M={M:5d} N={N:6d} K={K:6d} -> {tuple(y.shape)}"
                      f"{'' if ok else '  SHAPE MISMATCH'}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  [FAIL] M={M:5d} N={N:6d} K={K:6d}: "
                      f"{type(exc).__name__}: {str(exc)[:160]}")
                traceback.print_exc(limit=1)
                # The HIP context is dead after an internal error.
                print(f"\nVERDICT: {args.dtype} GEMM FAILS at M={M}, "
                      f"N={N}, K={K}")
                return 1

    print(f"\nVERDICT: {args.dtype} GEMM clean over "
          f"{len(SHAPES) * len(M_VALUES)} shape/size combinations")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
