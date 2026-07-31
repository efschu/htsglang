# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Lanczos-3 resize on the device.

This is the §8.1 resize stage, and it is not optional:
``realesr-general-wdn-x4v3`` is a fixed x4 network, so a 1080p source leaves
SR at 7680x4320. The resize is what makes the chain terminate at a sane
resolution, and placing it after SR and before RIFE is what lets the RIFE
engine be built at the target size instead of at 8K.

The implementation is a separable Lanczos-3 filter in torch, matching the
sampling convention of ``efschu/vs-mlrt``'s ``lanczos3_kernel.cu`` (half-pixel
centres, support scaled by the sampling ratio when downscaling, normalised
taps). It accumulates one tap at a time rather than materialising an
``[N, C, H, W_out, taps]`` gather, because at 8K to 4K that intermediate is
over a gigabyte on its own and would dominate the stage's memory post.
"""

from __future__ import annotations

import math
from functools import lru_cache

from sglang.srt.video_enhance.frame_math import PixelFormat, Resolution
from sglang.srt.video_enhance.frames import StageBase

LANCZOS_A = 3


def _lanczos(x: float, a: int = LANCZOS_A) -> float:
    if x == 0.0:
        return 1.0
    if abs(x) >= a:
        return 0.0
    px = math.pi * x
    return a * math.sin(px) * math.sin(px / a) / (px * px)


@lru_cache(maxsize=64)
def _taps(in_size: int, out_size: int, a: int = LANCZOS_A):
    """Per-output-pixel source indices and weights, as plain Python lists.

    Cached because a stage resizes thousands of frames with identical
    geometry, and rebuilding the tap table per frame would show up as CPU
    time in the ms/frame measurement.
    """
    ratio = out_size / in_size
    # Downscaling widens the filter support in source space; upscaling does
    # not narrow it below the reconstruction kernel.
    support_scale = 1.0 / ratio if ratio < 1.0 else 1.0
    support = a * support_scale
    n_taps = int(math.ceil(support) * 2)

    indices: list[list[int]] = []
    weights: list[list[float]] = []
    for i in range(out_size):
        center = (i + 0.5) / ratio - 0.5
        start = int(math.floor(center - support + 0.5))
        row_idx: list[int] = []
        row_w: list[float] = []
        total = 0.0
        for t in range(n_taps):
            src = start + t
            w = _lanczos((src - center) / support_scale, a)
            row_idx.append(min(max(src, 0), in_size - 1))
            row_w.append(w)
            total += w
        if total == 0.0:
            # Can only happen for a degenerate ratio; fall back to the nearest
            # sample rather than emitting NaNs.
            row_w = [0.0] * n_taps
            row_w[n_taps // 2] = 1.0
        else:
            row_w = [w / total for w in row_w]
        indices.append(row_idx)
        weights.append(row_w)
    return indices, weights


def _tap_tensors(in_size: int, out_size: int, device, dtype):
    import torch

    indices, weights = _taps(in_size, out_size)
    idx = torch.tensor(indices, dtype=torch.long, device=device)
    w = torch.tensor(weights, dtype=dtype, device=device)
    return idx, w


def _resize_axis(x, out_size: int, axis: int):
    """Resample one spatial axis of an NCHW tensor."""
    import torch

    in_size = x.shape[axis]
    if in_size == out_size:
        return x
    # Weights are accumulated in fp32 even for an fp16 chain: a 12-tap
    # normalised sum in fp16 loses enough precision to be visible against the
    # fp32 reference the parity gate compares to.
    acc_dtype = torch.float32
    idx, w = _tap_tensors(in_size, out_size, x.device, acc_dtype)
    n_taps = idx.shape[1]

    out_shape = list(x.shape)
    out_shape[axis] = out_size
    out = torch.zeros(out_shape, dtype=acc_dtype, device=x.device)
    xf = x.to(acc_dtype)
    w_shape = [1] * x.dim()
    w_shape[axis] = out_size
    for t in range(n_taps):
        gathered = torch.index_select(xf, axis, idx[:, t])
        out += gathered * w[:, t].reshape(w_shape)
    return out.to(x.dtype)


def lanczos3_resize(x, target: Resolution):
    """Resize an NCHW device tensor to ``target`` with separable Lanczos-3."""
    if x.dim() != 4:
        raise ValueError(f"expected NCHW, got shape {tuple(x.shape)}")
    # Horizontal first: it is the pass that shrinks 7680 to the target width,
    # so the vertical pass then runs on the smaller intermediate. The memory
    # post in frame_math.resize_footprint assumes this order.
    x = _resize_axis(x, target.width, axis=3)
    x = _resize_axis(x, target.height, axis=2)
    return x


class ResizeStage(StageBase):
    name = "resize"

    def __init__(
        self, source: Resolution, target: Resolution, fmt: PixelFormat
    ) -> None:
        self.source = source
        self.target = target
        self.format = fmt

    def process(self, frames):
        out = []
        for frame in frames:
            if frame.end_of_stream:
                out.append(frame)
                continue
            if frame.resolution != self.source:
                raise ValueError(
                    f"resize stage configured for {self.source}, got {frame.resolution}"
                )
            resized = lanczos3_resize(frame.data, self.target)
            out.append(
                frame.with_data(resized, resolution=self.target).require_device(
                    self.name
                )
            )
        return out
