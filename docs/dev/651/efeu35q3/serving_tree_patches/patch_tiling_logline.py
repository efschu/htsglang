"""efeu-TP14: one log line per over-cap dequant target at its first tiled use.

The kernel-level call site only sees tensors, not module names, so the target
is named by its (rows x cols, ggml type); the only over-cap targets of the
served model are the 248320-row vocab matrices (lm_head / embedding).
Staged into the serving tree; takes effect at the next (gated) restart.
"""
import ast
import os
import shutil

P = "/root/efeu35q3/sglang_src/python/sglang/srt/layers/quantization/gguf.py"
s = open(P).read()
old = """            y = _tiled_dequant_matmul(x, qweight, qweight_type, shape[0], shape[1], _dq, tile_rows)
"""
new = """            _tkey = (shape[0], shape[1], int(qweight_type), x.dtype)
            if _tkey not in _TILE_LOGGED:
                _TILE_LOGGED.add(_tkey)
                logger.info(
                    "GGUF dequant tiling: target %dx%d %s (%s%.0f MiB untiled) -> %d rows per tile, %.0f MiB tile",
                    shape[0], shape[1], WeightType(qweight_type).name,
                    "vocab-shaped, " if shape[0] >= 100000 else "",
                    shape[0] * shape[1] * x.dtype.itemsize / 2**20,
                    tile_rows, tile_rows * shape[1] * x.dtype.itemsize / 2**20,
                )
            y = _tiled_dequant_matmul(x, qweight, qweight_type, shape[0], shape[1], _dq, tile_rows)
"""
assert s.count(old) == 1, "call site"
s = s.replace(old, new, 1)
old2 = "def _dequant_tile_overcap() -> bool:"
assert s.count(old2) == 1
s = s.replace(old2, "_TILE_LOGGED: set = set()\n\n\n" + old2, 1)
ast.parse(s)
if __name__ == "__main__":
    if not os.path.exists(P + ".orig-efeu-tilelog"):
        shutil.copy(P, P + ".orig-efeu-tilelog")
    open(P + ".new", "w").write(s)
    os.replace(P + ".new", P)
    print("tiling log line staged (active at next restart)")
