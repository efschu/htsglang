"""efeu-TP14 2026-10-01: over-cap GGUF dequant is TILED, never a fresh full-size
allocation, and the KV budget is charged for one tile instead of the full target.

Why: the scratch post reserved 0.95 GiB = the Q6_K lm_head (248320 x 2048
bf16) "largest over-cap dequant target", although the lm_head runs through
MMVQ (M = number of requests <= 2) on every normal decode/prefill step and is
only dequantized for prompt-logprob requests. On this APU those 0.95 GiB are
the difference between a 124941- and a >=131076-token KV pool.

What changes (gguf.py of the laptop serving tree only):
  * _tiled_dequant_matmul(x, qweight, type, rows, cols, dequant_fn, tile_rows):
    x @ dequant(qweight).T computed over OUTPUT-row tiles; every output element
    keeps its full K dot product. dequant_fn is injected so the identical code
    is unit-tested on CPU against the untiled product.
  * fused_mul_mat_gguf: an over-cap dequant target (> SGLANG_GGUF_DEQUANT_WS_CAP_MIB)
    takes the tiled path whenever SGLANG_GGUF_DEQUANT_TILE_OVERCAP=1 (default 1),
    not only during CUDA-graph capture; the tile is <= SGLANG_GGUF_DEQUANT_TILE_MIB
    (default 64) and lives in one cached buffer.
  * _reserve_dequant_workspace records min(target, tile) as the peak for
    over-cap targets, so gguf_dequant_scratch_residual_bytes charges one tile.
  * the dequant call works with the standalone ROCm extension (sglang_gguf_rocm
    has the out= argument; torch.ops.sgl_kernel does not exist on this host).
"""
import ast
import os
import shutil

P = "/root/efeu35q3/sglang_src/python/sglang/srt/layers/quantization/gguf.py"
s = open(P).read()

# 1) helpers after _dequant_ws_cap_bytes
old = '''def _dequant_ws_cap_bytes() -> int:
    return int(os.environ.get("SGLANG_GGUF_DEQUANT_WS_CAP_MIB", "512")) * (1 << 20)
'''
new = old + '''

def _dequant_tile_overcap() -> bool:
    """efeu-TP14: tile EVERY over-cap dequant (not only under graph capture)."""
    return os.environ.get("SGLANG_GGUF_DEQUANT_TILE_OVERCAP", "1") == "1"


def _dequant_tile_bytes() -> int:
    return min(
        int(os.environ.get("SGLANG_GGUF_DEQUANT_TILE_MIB", "64")) * (1 << 20),
        _dequant_ws_cap_bytes(),
    )


def _dequant_into(qweight, qweight_type, rows, cols, dtype, out):
    """Dequantize into a caller buffer with whichever GGUF op this host has."""
    if _dequant_supports_out():
        return torch.ops.sgl_kernel.ggml_dequantize.default(
            qweight, qweight_type, rows, cols, dtype, out
        )
    return ggml_dequantize(qweight, qweight_type, rows, cols, dtype, out)


def _tiled_dequant_matmul(x, qweight, qweight_type, rows, cols, dequant_fn, tile_rows):
    """x @ dequant(qweight).T over OUTPUT-row tiles of tile_rows rows.

    dequant_fn(qweight_rows, qweight_type, rn, cols) -> [rn, cols] weight tile.
    Each output column j depends only on weight row j, so the tiles compute
    exactly the elements they own (full K dot product each); the result is
    assembled by concatenation along the output dimension.
    """
    tile_rows = max(int(tile_rows), 1)
    outs = []
    for r0 in range(0, rows, tile_rows):
        rn = min(tile_rows, rows - r0)
        w = dequant_fn(qweight.narrow(0, r0, rn), qweight_type, rn, cols)
        outs.append(x @ w.T)
    return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]
'''
assert s.count(old) == 1, "helpers"
s = s.replace(old, new, 1)

# 2) peak recording: over-cap targets are charged one tile
old = '''    nbytes = numel * dtype.itemsize
    # Record the peak target BEFORE either early return (#257): both of them
    # mean "this dequant fresh-allocates its full size at forward time",
    # which is precisely the demand the KV budget has to know about.
    if nbytes > _DEQUANT_PEAK_TARGET.get(key, 0):
        _DEQUANT_PEAK_TARGET[key] = nbytes'''
new = '''    nbytes = numel * dtype.itemsize
    # Record the peak target BEFORE either early return (#257): both of them
    # mean "this dequant fresh-allocates its full size at forward time",
    # which is precisely the demand the KV budget has to know about.
    # efeu-TP14: an over-cap target is TILED (_tiled_dequant_matmul), so its
    # forward-time demand is one tile, not the whole weight.
    charged = nbytes
    if _dequant_tile_overcap() and nbytes > _dequant_ws_cap_bytes():
        charged = _dequant_tile_bytes()
    if charged > _DEQUANT_PEAK_TARGET.get(key, 0):
        _DEQUANT_PEAK_TARGET[key] = charged'''
assert s.count(old) == 1, "peak"
s = s.replace(old, new, 1)

# 3) dispatch: tiled path for over-cap targets outside capture too
old = '''        if (
            _is_cuda
            and shape[0] * shape[1] * x.dtype.itemsize > _dequant_ws_cap_bytes()
            and _dequant_supports_out()
            and torch.cuda.is_current_stream_capturing()
        ):
            y = _mul_mat_dequant_chunked(x, qweight, qweight_type, *shape)
        else:'''
new = '''        if (
            _is_cuda
            and shape[0] * shape[1] * x.dtype.itemsize > _dequant_ws_cap_bytes()
            and _dequant_supports_out()
            and torch.cuda.is_current_stream_capturing()
        ):
            y = _mul_mat_dequant_chunked(x, qweight, qweight_type, *shape)
        elif (
            _dequant_tile_overcap()
            and shape[0] * shape[1] * x.dtype.itemsize > _dequant_ws_cap_bytes()
        ):
            # efeu-TP14: tile through ONE cached buffer of <= TILE_MIB instead of
            # a fresh full-size allocation (the KV budget is charged one tile).
            key = ("tile",) + _dequant_ws_key(x.dtype, qweight.device)
            tile_rows = max(_dequant_tile_bytes() // (shape[1] * x.dtype.itemsize), 1)
            buf = _DEQUANT_WS.get(key)
            if buf is None or buf.numel() < tile_rows * shape[1]:
                buf = torch.empty(tile_rows * shape[1], dtype=x.dtype, device=qweight.device)
                _DEQUANT_WS[key] = buf

            def _dq(qw, qt, rn, cols, _buf=buf, _dt=x.dtype):
                return _dequant_into(qw, qt, rn, cols, _dt, _buf.narrow(0, 0, rn * cols).view(rn, cols))

            y = _tiled_dequant_matmul(x, qweight, qweight_type, shape[0], shape[1], _dq, tile_rows)
        else:'''
assert s.count(old) == 1, "dispatch"
s = s.replace(old, new, 1)
ast.parse(s)

if __name__ == "__main__":
    if not os.path.exists(P + ".orig-efeu-tiling"):
        shutil.copy(P, P + ".orig-efeu-tiling")
    open(P + ".new", "w").write(s)
    os.replace(P + ".new", P)
    print("dequant tiling staged")
