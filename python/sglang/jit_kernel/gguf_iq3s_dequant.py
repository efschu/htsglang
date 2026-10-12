"""GGUF-NF G12 (2026-10-12): format-exact IQ3_S dequantisation in plain torch ops (CPU or CUDA), for the dense dequant + GEMM path.

Why it exists: the installed sgl-kernel wheel's ``ggml_dequantize`` decodes IQ3_S with ggml's pre-final ``iq3xs_grid`` (grid value 15 -> 15.5,
+3.3 % on ~1/8 of the weights; csrc/quantization/gguf/iq3s_grid_ggml.cuh has the story). A wheel built from the G12 source is fixed and carries the
marker op ``ggml_iq3s_grid_ggml``; until the image runs such a wheel, ``srt/layers/quantization/gguf.py`` uses this function instead of
``ggml_dequantize`` for IQ3_S (dense M > 16, embedding rows, the per-token MoE fallback).

The arithmetic is gguf-py's ``IQ3_S.dequantize_blocks`` (ggml ``dequantize_row_iq3_s``): ``y = d * (1 + 2 * s) * grid * sign`` with the grid
1,3,...,15 taken from gguf-py itself (``IQ3_S.grid``), float32 throughout, one cast to the output dtype at the end -- the same single rounding as the
CUDA kernel's ``dst_t`` store. Block layout (110 bytes): d (fp16) | qs[64] | qh[8] | signs[32] | scales[4].
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

QK_K = 256
IQ3_S_BLOCK_BYTES = 110

_GRID_CACHE: Dict[Tuple[str, int], torch.Tensor] = {}


def ggml_grid() -> np.ndarray:
    """gguf-py's IQ3_S grid, int values 1,3,...,15, shape (512, 4)."""
    import gguf

    gguf.quants.IQ3_S.init_grid()
    return np.asarray(gguf.quants.IQ3_S.grid).reshape(512, 4).astype(np.float32)


def _grid_on(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index if device.index is not None else -1)
    g = _GRID_CACHE.get(key)
    if g is None:
        g = torch.from_numpy(ggml_grid()).to(device)
        _GRID_CACHE[key] = g
    return g


def prepare(device: torch.device) -> None:
    """Build the per-device grid table. Call at weight-load time so a CUDA-graph capture never meets the first host-to-device copy."""
    _grid_on(torch.device(device))


def dequantize_iq3_s(qweight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``[rows, nb * 110]`` raw uint8 IQ3_S blocks -> ``[rows, nb * 256]`` weights of ``dtype``."""
    if qweight.dtype != torch.uint8 or qweight.dim() != 2:
        raise ValueError(f"IQ3_S dequant: expected a 2-D uint8 tensor of raw blocks, got {qweight.dtype} {tuple(qweight.shape)}")
    rows, row_bytes = qweight.shape
    if row_bytes % IQ3_S_BLOCK_BYTES:
        raise ValueError(f"IQ3_S dequant: {row_bytes} bytes per row is not a multiple of {IQ3_S_BLOCK_BYTES}")
    nb = row_bytes // IQ3_S_BLOCK_BYTES
    n = rows * nb
    b = qweight.contiguous().view(n, IQ3_S_BLOCK_BYTES)
    d = b[:, 0:2].contiguous().view(torch.float16).to(torch.float32)  # [n, 1]
    qs = b[:, 2:66].to(torch.int64)  # [n, 64]
    qh = b[:, 66:74].to(torch.int64)  # [n, 8]
    signs = b[:, 74:106].to(torch.int64)  # [n, 32]
    scales = b[:, 106:110].to(torch.int64)  # [n, 4]

    shifts = torch.arange(8, device=b.device, dtype=torch.int64)
    # 4-bit scales: low nibble = even sub-block, high nibble = odd sub-block (gguf-py: scales >> [0, 4], & 0x0F)
    sc = torch.stack((scales & 0xF, (scales >> 4) & 0xF), dim=-1).reshape(n, 8)
    db = d * (1 + 2 * sc).to(torch.float32)  # [n, 8]
    # sign bits: one bit per weight, 8 per byte, 32 bytes per block (gguf-py: signs >> 0..7, & 1 -> -1 / +1)
    sgn = torch.where(((signs.unsqueeze(-1) >> shifts) & 1) == 0, 1.0, -1.0).to(torch.float32).reshape(n, 8, 4, 8)
    # the ninth index bit of each grid entry lives in qh (one byte per sub-block, bit j = entry j)
    idx = qs | (((qh.unsqueeze(-1) >> shifts) & 1).reshape(n, 64) << 8)  # [n, 64]
    grid = _grid_on(b.device)[idx].reshape(n, 8, 4, 8)  # [n, 64, 4] -> [n, 8, 4, 8]
    out = db.reshape(n, 8, 1, 1) * grid * sgn
    return out.reshape(rows, nb * QK_K).to(dtype)
