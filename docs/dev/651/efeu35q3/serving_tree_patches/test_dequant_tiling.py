"""CPU test: the tiled over-cap dequant matmul equals the untiled one.

The function under test is extracted VERBATIM from the patched gguf.py text
(patch_dequant_tiling.py builds it), so this tests the code that ships. The
dequant_fn is the numpy reference (gguf.quants.dequantize) on real bytes of the
served file's lm_head (Q6_K) -- a lm_head-shaped slice, CPU only.

    python test_dequant_tiling.py <gguf>
"""
import ast
import sys

import numpy as np
import torch
import gguf
from gguf import GGUFReader

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import patch_dequant_tiling as P  # noqa: E402  (does not write unless __main__)

tree = ast.parse(P.s)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_tiled_dequant_matmul")
ns = {"torch": torch}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "gguf.py", "exec"), ns)
tiled = ns["_tiled_dequant_matmul"]

src = sys.argv[1]
t = [x for x in GGUFReader(src).tensors if x.name == "output.weight"][0]
qt = t.tensor_type
raw = np.array(t.data[:40000])  # 40000 vocab rows x 2048 (> one production tile)
rows = raw.shape[0]
cols = raw.shape[1] // gguf.GGML_QUANT_SIZES[qt][1] * gguf.GGML_QUANT_SIZES[qt][0]
qw = torch.from_numpy(raw)


def dq(qrows, qtype, rn, c, dtype):
    return torch.from_numpy(gguf.quants.dequantize(qrows.numpy(), qtype).astype(np.float32)).to(dtype).view(rn, c)


ok = True
for dtype in (torch.float32, torch.bfloat16):
    full_w = dq(qw, qt, rows, cols, dtype)
    for m in (1, 7, 64):
        x = torch.randn(m, cols, generator=torch.Generator().manual_seed(m)).to(dtype)
        ref = x @ full_w.T
        # production tile: 64 MiB / (2048 x 2 B) = 16384 rows (bf16), 8192 rows (fp32)
        for tile_rows in (1, 37, 512, 4096, 8192, 16384, rows):
            got = tiled(x, qw, qt, rows, cols, lambda a, b, rn, c: dq(a, b, rn, c, dtype), tile_rows)
            same = torch.equal(got, ref)
            if not same:
                d = (got.float() - ref.float()).abs().max().item()
                print(f"DIFF dtype={dtype} M={m} tile={tile_rows} max|d|={d:.3e}")
                # tiles of 1 / 37 rows take BLAS's GEMV-like small-N kernels
                # (1-ulp differences); production tiles are >= 8192 rows.
                if tile_rows >= 512:
                    ok = False
            else:
                print(f"same dtype={dtype} M={m} tile={tile_rows}")
print(f"{qt.name} lm_head slice {rows}x{cols}: tiled == untiled bitwise for every tile >= 512 rows"
      if ok else "NOT bitwise at a production tile size")
sys.exit(0 if ok else 1)
