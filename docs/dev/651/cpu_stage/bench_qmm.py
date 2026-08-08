"""Throughput probe for the ggml-cpu quantized MUL_MAT shim (#651 CPU stage).

ALL NUMBERS ARE RIG CPU (AMD Ryzen 9 5950X, 16C/32T) -- NOT the APU laptop.
The recipe is the portable part; the laptop re-measures with the same script.

Shapes (35B-relevant, T=1024 prefill chunk):
  dense-eq : one Q4_K [4096 x 2048] matmul (dense-equivalent aggregate)
  moe-8exp : 8 active experts x Q4_K [512 x 2048] (per-expert down-proj
             stack; 8-of-256 routing worth of work per token batch)
Weights are REAL packed bytes cut from the Q4_K_XL checkpoint.

Run: /spinning/htsglang-gpu/.venv/bin/python bench_qmm.py [gguf_path]
"""
import sys
import time

import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as QT

from test_qmm import lib, shim_qmm, SRC

T = 1024
THREADS = (1, 8, 16)
MIN_SECONDS = 10.0
WARMUP = 2
Q4K_ID = int(QT.Q4_K)


def cut_q4k_rows(reader, rows, cols):
    """First `rows` rows of a Q4_K tensor with row length `cols`."""
    for tsr in reader.tensors:
        if tsr.tensor_type != QT.Q4_K:
            continue
        d = tsr.data
        if d.ndim == 3:
            d = d.reshape(-1, d.shape[-1])
        if d.ndim != 2 or d.shape[0] < rows:
            continue
        deq_cols = int(tsr.shape[0])  # ne[0] = row width in elements
        if deq_cols == cols:
            return tsr.name, np.ascontiguousarray(d[:rows].copy())
    raise RuntimeError(f"no Q4_K tensor with cols={cols} and >= {rows} rows")


def bench(fn, min_seconds=MIN_SECONDS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    times = []
    t_end = time.perf_counter() + min_seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return times.mean() * 1e3, times.min() * 1e3, len(times)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    reader = GGUFReader(src)
    rng = np.random.default_rng(7)

    name_d, w_dense = cut_q4k_rows(reader, 4096, 2048)   # [4096 x 2048]
    name_e, w_exp = cut_q4k_rows(reader, 8 * 512, 2048)  # 8 x [512 x 2048]
    x = (rng.standard_normal((T, 2048)) * 0.1).astype(np.float32)

    flops_dense = 2.0 * T * 4096 * 2048
    flops_moe = 2.0 * T * 8 * 512 * 2048

    print("RIG CPU (AMD Ryzen 9 5950X 16C/32T) -- NOT laptop numbers")
    print(f"model: {src}")
    print(f"dense-eq  W: {name_d} first 4096 rows, Q4_K [4096x2048], T={T}")
    print(f"moe-8exp  W: {name_e} first 4096 rows as 8x[512x2048], T={T}")
    print(f"{'shape':<10} {'threads':>7} {'ms/call':>10} {'ms best':>10} "
          f"{'calls':>6} {'GFLOP/s':>9}")

    for nt in THREADS:
        def call_dense():
            shim_qmm(Q4K_ID, w_dense, x, 4096, 2048, n_threads=nt)
        mean_ms, best_ms, n = bench(call_dense)
        print(f"{'dense-eq':<10} {nt:>7} {mean_ms:>10.2f} {best_ms:>10.2f} "
              f"{n:>6} {flops_dense / (mean_ms * 1e6):>9.1f}")

    experts = [np.ascontiguousarray(w_exp[i * 512:(i + 1) * 512])
               for i in range(8)]
    for nt in THREADS:
        def call_moe():
            for we in experts:  # 8 active experts, one qmm each
                shim_qmm(Q4K_ID, we, x, 512, 2048, n_threads=nt)
        mean_ms, best_ms, n = bench(call_moe)
        print(f"{'moe-8exp':<10} {nt:>7} {mean_ms:>10.2f} {best_ms:>10.2f} "
              f"{n:>6} {flops_moe / (mean_ms * 1e6):>9.1f}")

    print("\nAll numbers RIG CPU (not laptop). ms/call for moe-8exp covers "
          "all 8 expert matmuls (one routed MoE layer pass at capacity 8).")


if __name__ == "__main__":
    main()
