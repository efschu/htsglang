"""Falsifier for the ggml-cpu quantized MUL_MAT shim (#651 CPU stage).

Real packed slices from the Q4_K_XL checkpoint, one tensor per quant type;
oracle is gguf.quants.dequantize (numpy) -> fp64 matmul. For each type x
T in {1, 16, 512}: relative max error vs oracle (gate 5e-2), 3-run
byte-identical determinism, and a can-fail proof (one flipped byte in W
must trip the error gate).

Run: /spinning/htsglang-gpu/.venv/bin/python test_qmm.py [gguf_path]
"""
import ctypes
import os
import sys

import numpy as np
import gguf
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as QT

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else (
    "/spinning/llm_stuff/club-3090/models-cache/unsloth/"
    "Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
ROWS = 512
TS = (1, 16, 512)
REL_GATE = 5e-2
N_DET = 3
N_THREADS = 8

lib = ctypes.CDLL(os.path.join(HERE, "libqmatmul_shim.so"))
lib.qmm_supported.restype = ctypes.c_int
lib.qmm_supported.argtypes = [ctypes.c_int]
lib.qmm_row_size.restype = ctypes.c_longlong
lib.qmm_row_size.argtypes = [ctypes.c_int, ctypes.c_longlong]
lib.qmm.restype = ctypes.c_int
lib.qmm.argtypes = [
    ctypes.c_int, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong,
    ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p,
    ctypes.c_int,
]


def shim_qmm(tid, raw, x, rows, cols, n_threads=N_THREADS):
    """raw: packed (rows, row_bytes) uint8; x: (t, cols) f32 -> (t, rows) f32."""
    raw = np.ascontiguousarray(raw)
    x = np.ascontiguousarray(x, dtype=np.float32)
    t = x.shape[0]
    out = np.full((t, rows), np.nan, dtype=np.float32)
    rc = lib.qmm(
        tid, raw.ctypes.data_as(ctypes.c_void_p), rows, raw.shape[1],
        x.ctypes.data_as(ctypes.c_void_p), t, cols,
        out.ctypes.data_as(ctypes.c_void_p), n_threads)
    if rc != 0:
        raise RuntimeError(f"qmm rc={rc}")
    return out


def pick_tensors(reader, rows):
    """One representative 2D-sliceable tensor per quant type
    (same pattern as docs/dev/651/p2/scripts/audit_all_types.py)."""
    picked = {}
    for tsr in reader.tensors:
        tt = tsr.tensor_type
        if tt.name in ("F32", "F16", "BF16") or tt.name in picked:
            continue
        d = tsr.data
        if d.ndim == 3:
            d = d.reshape(-1, d.shape[-1])
        if d.ndim != 2 or d.shape[0] < rows:
            continue
        picked[tt.name] = (tsr.name, int(tt), d[:rows].copy())
    return picked


def rel_err(out, ref, scale):
    a = np.nan_to_num(out.astype(np.float64), nan=0.0, posinf=1e30, neginf=-1e30)
    return np.abs(a - ref).max() / scale


def main():
    rng = np.random.default_rng(23)
    reader = GGUFReader(SRC)
    picked = pick_tensors(reader, ROWS)

    failures = []
    print(f"model: {SRC}")
    print(f"types found: {sorted(picked)}  gate rel<{REL_GATE}  threads={N_THREADS}")
    for name, (tname, tid, raw) in sorted(picked.items()):
        ref_w = gguf.quants.dequantize(raw, QT(tid)).astype(np.float64)
        rows, cols = ref_w.shape
        assert lib.qmm_supported(tid) == 1, f"{name}: not supported by ggml-cpu"
        assert lib.qmm_row_size(tid, cols) == raw.shape[1], f"{name}: row_bytes mismatch"
        print(f"=== {name} (id {tid}) tensor {tname} {rows}x{cols} "
              f"row_bytes {raw.shape[1]} ===")
        for T in TS:
            x = (rng.standard_normal((T, cols)) * 0.1).astype(np.float32)
            ref = x.astype(np.float64) @ ref_w.T
            scale = np.abs(ref).max() + 1e-12

            outs = [shim_qmm(tid, raw, x, rows, cols) for _ in range(N_DET)]
            det = all(np.array_equal(outs[0], o) for o in outs[1:])
            rel = rel_err(outs[0], ref, scale)
            nonfin = int((~np.isfinite(outs[0])).sum())
            ok = rel < REL_GATE and det and nonfin == 0

            # CAN-FAIL PROOF: flip ONE byte in a copy of W and require that
            # the error gate then actually FAILS (rel >= gate). A byte in a
            # low-order quant nibble can be numerically invisible at this
            # gate, so target block-scale bytes: fp16 d sits at the block
            # HEAD for Q8_0/Q4_K/Q5_K (byte 1 = sign/exponent) and at the
            # block TAIL for Q6_K (byte bs-1/bs-2). One tripped gate proves
            # the falsifier is capable of failing.
            bs = gguf.constants.GGML_QUANT_SIZES[QT(tid)][1]  # bytes/block
            candidates = [1, bs - 1, 0, bs - 2] + \
                [int(rng.integers(0, raw.size)) for _ in range(4)]
            canfail = False
            for pos in candidates:
                bad = raw.copy()
                bad.reshape(-1)[pos] ^= 0xFF
                out_bad = shim_qmm(tid, bad, x, rows, cols)
                # same criteria as the real gate: error bound AND finiteness
                # (a flipped fp16 scale byte can turn d into NaN, which the
                # nonfin check catches even where nan->0 masks the error)
                if (rel_err(out_bad, ref, scale) >= REL_GATE or
                        not np.isfinite(out_bad).all()):
                    canfail = True
                    break

            status = "PASS" if (ok and canfail) else "FAIL"
            if status == "FAIL":
                failures.append((name, T))
            print(f"  T={T:<4} rel {rel:.3e}  det={det!s:<5} nonfin {nonfin}  "
                  f"canfail={canfail!s:<5} {status}")

    print()
    if failures:
        print(f"FALSIFIED: {failures}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
