"""Is the Q6_K non-determinism a race, or stale allocator memory showing through?

dequantize_block_q6_K has no shared memory and no cross-thread communication:
each of the 64 threads writes 4 disjoint offsets and the block covers exactly
QK_K=256 elements. A race is structurally impossible. The competing hypothesis
is that the kernel does not write every output element, so torch's caching
allocator (which returns DIRTY memory) shows different garbage each call.

Decisive test: hand the op a pre-zeroed output buffer via out=. If the garbage
vanishes, it is a coverage bug, not a race.
"""
import json
import numpy as np
import torch
import gguf
from gguf.constants import GGMLQuantizationType as QT
import gguf_rocm_probe as K

s = [x for x in json.load(open("slices.json")) if x["name"] == "q6_K"][0]
r, c = s["rows"], s["cols"]
raw = np.fromfile(s["path"], dtype=np.uint8).reshape(r, s["row_bytes"])
ref = gguf.quants.dequantize(raw, QT.Q6_K).astype(np.float32)
W = torch.from_numpy(raw).cuda()

print("--- A: fresh allocation each call (as the earlier test did) ---")
for i in range(3):
    a = K.ggml_dequantize(W, 14, r, c, torch.float16, None).float().cpu().numpy()
    print(f"  run {i}: non-finite {int((~np.isfinite(a)).sum()):4d}  "
          f"max|d| {np.abs(np.nan_to_num(a,posinf=0,neginf=0)-ref).max():.3e}")

print("--- B: caller-supplied buffer, ZEROED before each call ---")
for i in range(3):
    out = torch.zeros((r, c), dtype=torch.float16, device="cuda")
    a = K.ggml_dequantize(W, 14, r, c, torch.float16, out).float().cpu().numpy()
    nz = int((a == 0).sum())
    print(f"  run {i}: non-finite {int((~np.isfinite(a)).sum()):4d}  "
          f"max|d| {np.abs(np.nan_to_num(a,posinf=0,neginf=0)-ref).max():.3e}  "
          f"exact-zero outputs {nz}")

print("--- C: caller-supplied buffer, POISONED with a sentinel before each call ---")
for i in range(3):
    out = torch.full((r, c), 12345.0, dtype=torch.float16, device="cuda")
    a = K.ggml_dequantize(W, 14, r, c, torch.float16, out).float().cpu().numpy()
    untouched = int((a == 12345.0).sum())
    print(f"  run {i}: untouched-by-kernel elements {untouched} / {a.size}  "
          f"max|d| {np.abs(np.nan_to_num(a,posinf=0,neginf=0)-ref).max():.3e}")
    if untouched:
        idx = np.argwhere(a == 12345.0)
        print(f"     first untouched at row {idx[0][0]} col {idx[0][1]}; "
              f"unique cols {sorted(set(idx[:,1].tolist()))[:12]}")
