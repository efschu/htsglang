"""Numerical correctness of the gfx1103 GGUF kernels against the format's own reference.

A green build proves nothing about arithmetic. The oracle here is the `gguf`
python package's numpy dequantiser, which IS the definition of the on-disk
format -- not another GPU implementation that could share a bug.

Covers only the four ggml types the target checkpoint contains:
Q8_0 (8), Q4_K (12), Q5_K (13), Q6_K (14).
"""

import numpy as np
import torch
import gguf
from gguf.constants import GGMLQuantizationType as QT

import gguf_rocm_probe as K

TYPES = [("Q8_0", QT.Q8_0, 8), ("Q4_K", QT.Q4_K, 12),
         ("Q5_K", QT.Q5_K, 13), ("Q6_K", QT.Q6_K, 14)]

DEV = "cuda"
rng = np.random.default_rng(0)


def make_quantized(qtype, rows, cols):
    """Quantise random data with the reference implementation."""
    # CPU-sampled inputs on purpose: on-GPU RNG is not architecture-identical.
    data = rng.standard_normal((rows, cols), dtype=np.float32) * 0.1
    packed = gguf.quants.quantize(data, qtype)          # numpy, reference
    ref = gguf.quants.dequantize(packed, qtype)         # numpy, reference oracle
    return data, packed, ref.astype(np.float32)


def report(tag, got, ref, tol):
    err = np.abs(got - ref)
    denom = np.abs(ref).mean() + 1e-9
    maxe, meane = err.max(), err.mean()
    ok = maxe <= tol
    print(f"    {tag:<28} max|d| {maxe:.3e}  mean|d| {meane:.3e}  "
          f"rel {meane/denom:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print(f"device: {torch.cuda.get_device_properties(0).name} "
          f"({torch.cuda.get_device_properties(0).gcnArchName})")
    print(f"ggml_mmvq_kq_tuned={K.ggml_mmvq_kq_tuned()} "
          f"ggml_mxfp4_native={K.ggml_mxfp4_native()}")
    all_ok = True

    rows, cols = 512, 1024
    for name, qtype, tid in TYPES:
        print(f"\n=== {name} (ggml type {tid}) ===")
        _, packed, ref = make_quantized(qtype, rows, cols)
        W = torch.from_numpy(packed.reshape(rows, -1)).to(DEV)
        print(f"    packed {tuple(W.shape)} uint8, block size "
              f"{K.ggml_moe_get_block_size(tid)}")

        # 1. dequantize
        got = K.ggml_dequantize(W, tid, rows, cols, torch.float16, None)
        all_ok &= report("dequantize", got.float().cpu().numpy(), ref, 2e-2)

        # 2. GEMV (decode path): X @ W^T for a single row
        X = torch.from_numpy(
            (rng.standard_normal((1, cols), dtype=np.float32) * 0.1)
        ).to(DEV).half()
        got_v = K.ggml_mul_mat_vec_a8(W, X, tid, rows)
        ref_v = X.float().cpu().numpy() @ ref.T
        # a8 quantises the ACTIVATION to int8, so tolerance is activation-driven
        all_ok &= report("mul_mat_vec_a8 (GEMV)", got_v.float().cpu().numpy(), ref_v, 5e-1)

        # 3. GEMM (prefill path): 16 rows
        Xb = torch.from_numpy(
            (rng.standard_normal((16, cols), dtype=np.float32) * 0.1)
        ).to(DEV).half()
        got_m = K.ggml_mul_mat_a8(W, Xb, tid, rows)
        ref_m = Xb.float().cpu().numpy() @ ref.T
        all_ok &= report("mul_mat_a8 (GEMM)", got_m.float().cpu().numpy(), ref_m, 5e-1)

    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
