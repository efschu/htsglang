"""K-quant correctness on gfx1103 using REAL tensor bytes from the target model.

The gguf python package can dequantise K-quants but cannot quantise to them,
so synthetic test data is not available for Q4_K/Q5_K/Q6_K. These slices are
the actual first 512 rows of real expert tensors from
unsloth/Qwen3.6-35B-A3B-MTP-GGUF Q4_K_M, fetched by HTTP range request. The
oracle is gguf.quants.dequantize (numpy), which defines the format.
"""

import json

import numpy as np
import torch
import gguf
from gguf.constants import GGMLQuantizationType as QT

import gguf_rocm_probe as K

QT_BY_ID = {8: QT.Q8_0, 12: QT.Q4_K, 13: QT.Q5_K, 14: QT.Q6_K}
rng = np.random.default_rng(1)


def report(tag, got, ref, tol):
    err = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    maxe, meane = err.max(), err.mean()
    denom = np.abs(ref).mean() + 1e-12
    ok = maxe <= tol
    print(f"    {tag:<26} max|d| {maxe:.3e}  mean|d| {meane:.3e}  "
          f"rel {meane/denom:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} ({p.gcnArchName})  [real model tensors as oracle]")
    slices = json.load(open("slices.json"))
    all_ok = True

    for s in slices:
        rows, cols, tid = s["rows"], s["cols"], s["type"]
        raw = np.fromfile(s["path"], dtype=np.uint8)
        assert raw.size == rows * s["row_bytes"], (raw.size, rows * s["row_bytes"])
        print(f"\n=== {s['name']} (ggml type {tid}) "
              f"{rows}x{cols} from real weights, {raw.size} B ===")

        ref = gguf.quants.dequantize(
            raw.reshape(rows, s["row_bytes"]), QT_BY_ID[tid]
        ).astype(np.float32)
        assert ref.shape == (rows, cols), ref.shape
        print(f"    ref stats: mean {ref.mean():+.5f} std {ref.std():.5f} "
              f"absmax {np.abs(ref).max():.5f}")

        W = torch.from_numpy(raw.reshape(rows, s["row_bytes"])).cuda()

        got = K.ggml_dequantize(W, tid, rows, cols, torch.float16, None)
        all_ok &= report("dequantize", got.float().cpu().numpy(), ref, 2e-2)

        X = torch.from_numpy(
            (rng.standard_normal((1, cols), dtype=np.float32) * 0.1)
        ).cuda().half()
        gv = K.ggml_mul_mat_vec_a8(W, X, tid, rows)
        rv = X.float().cpu().numpy() @ ref.T
        all_ok &= report("mul_mat_vec_a8 (GEMV)", gv.float().cpu().numpy(), rv, 5e-1)

        Xb = torch.from_numpy(
            (rng.standard_normal((16, cols), dtype=np.float32) * 0.1)
        ).cuda().half()
        gm = K.ggml_mul_mat_a8(W, Xb, tid, rows)
        rm = Xb.float().cpu().numpy() @ ref.T
        all_ok &= report("mul_mat_a8 (GEMM)", gm.float().cpu().numpy(), rm, 5e-1)

    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
