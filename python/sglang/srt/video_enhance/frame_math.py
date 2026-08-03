# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Exact frame and stage arithmetic for the Class-3 video-enhance tenant.

Everything in this module is pure arithmetic with no device access, so the
fixed-cost calculation that has to precede any GPU booking (DESIGN #333 §8.3,
§10 M2 feasibility gate) can be run on CPU, in a unit test, or from the
planner.

Two numbers are deliberately kept apart:

*   ``*_tensor_bytes``  -- the exact tensor arithmetic. No fudge factor.
*   ``*_reserved_bytes`` -- the tensor arithmetic plus a declared allocator
    overhead fraction. This is what goes into the ledger reservation.

Measurement post P3 compares the second against measured peak device bytes.
A large miss is a design failure of the Class-3 estimator, not a tuning
issue, which is why the overhead fraction is a single named constant rather
than being sprinkled through the call sites.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


class PixelFormat(str, Enum):
    """Frame layouts that appear at a stage boundary in the chain."""

    NV12 = "nv12"  # 8-bit 4:2:0, the codec-side layout
    RGB_UINT8 = "rgb_uint8"
    RGB_FP16 = "rgb_fp16"
    RGB_FP32 = "rgb_fp32"


_BYTES_PER_PIXEL: dict[PixelFormat, float] = {
    # 4:2:0 carries one luma byte plus half a chroma byte per pixel.
    PixelFormat.NV12: 1.5,
    PixelFormat.RGB_UINT8: 3.0,
    PixelFormat.RGB_FP16: 6.0,
    PixelFormat.RGB_FP32: 12.0,
}

_DTYPE_BYTES: dict[str, int] = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}


@dataclass(frozen=True, order=True)
class Resolution:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"resolution must be positive, got {self}")

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def scaled(self, factor: int) -> "Resolution":
        return Resolution(self.width * factor, self.height * factor)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.width}x{self.height}"

    @classmethod
    def parse(cls, text: str) -> "Resolution":
        """Parse ``WIDTHxHEIGHT``, e.g. ``1920x1080``."""
        parts = text.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            raise ValueError(f"resolution must look like 1920x1080, got {text!r}")
        return cls(int(parts[0]), int(parts[1]))


# The resolutions the design tabulates. Named so tests can assert against the
# published table rather than against re-derived numbers.
R540P = Resolution(960, 540)
R720P = Resolution(1280, 720)
R1080P = Resolution(1920, 1080)
R4K = Resolution(3840, 2160)
R8K = Resolution(7680, 4320)


def frame_bytes(res: Resolution, fmt: PixelFormat) -> int:
    """Exact byte size of one frame. NV12 requires even dimensions."""
    if fmt is PixelFormat.NV12 and (res.width % 2 or res.height % 2):
        raise ValueError(f"NV12 requires even dimensions, got {res}")
    return int(res.pixels * _BYTES_PER_PIXEL[fmt])


def dtype_bytes(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError:
        raise ValueError(
            f"unknown dtype {dtype!r}, expected one of {sorted(_DTYPE_BYTES)}"
        ) from None


# --------------------------------------------------------------------------
# Super-resolution stage
# --------------------------------------------------------------------------

# realesr-general-wdn-x4v3 is SRVGGNetCompact: 64 feature channels, 32
# convolutions, operating at *input* resolution throughout, with the x4
# upscale done only by the final pixel-shuffle. So the activation tensor is
# W*H*64*sizeof(dtype) and two must be live for the convolution ping-pong.
SRVGG_FEATURE_CHANNELS = 64
SRVGG_LIVE_ACTIVATIONS = 2
SR_SCALE = 4

# Allocator, reformat and workspace overhead on top of the exact tensor
# arithmetic. Calibrated so the design's published per-stream budgets
# (~1.0 GiB at 1080p input, ~0.25 GiB at 540p input) come out of the formula
# rather than being hard-coded next to it. Measurement post P3 either
# confirms this fraction or replaces it; it is not a safety margin and must
# not be treated as one.
SR_ALLOCATOR_OVERHEAD_FRACTION = 0.45


@dataclass(frozen=True)
class StageFootprint:
    """Per-in-flight-work-unit device bytes of one stage, itemised."""

    stage: str
    posts: dict[str, int]

    @property
    def tensor_bytes(self) -> int:
        return sum(self.posts.values())

    def reserved_bytes(self, overhead_fraction: float) -> int:
        if overhead_fraction < 0:
            raise ValueError("overhead fraction must not be negative")
        return int(math.ceil(self.tensor_bytes * (1.0 + overhead_fraction)))


def sr_footprint(
    source: Resolution, dtype: str = "fp16", scale: int = SR_SCALE
) -> StageFootprint:
    """Device bytes for one in-flight frame in the SR stage.

    ``scale`` is the model's output factor: 4 for realesr-general-wdn-x4v3, 1
    for a same-size denoise or restoration model. The activation term does not
    depend on it -- SRVGGNetCompact works at input resolution throughout and
    upscales only in the final pixel-shuffle -- but the output frame does.
    """
    esize = dtype_bytes(dtype)
    activation = source.pixels * SRVGG_FEATURE_CHANNELS * esize
    rgb_fmt = PixelFormat.RGB_FP16 if esize == 2 else PixelFormat.RGB_FP32
    return StageFootprint(
        stage="sr",
        posts={
            "input_frame": frame_bytes(source, rgb_fmt),
            "activations": activation * SRVGG_LIVE_ACTIVATIONS,
            "output_frame": frame_bytes(source.scaled(scale), rgb_fmt),
        },
    )


def sr_reserved_bytes(
    source: Resolution, dtype: str = "fp16", scale: int = SR_SCALE
) -> int:
    return sr_footprint(source, dtype, scale).reserved_bytes(
        SR_ALLOCATOR_OVERHEAD_FRACTION
    )


# --------------------------------------------------------------------------
# Resize stage
# --------------------------------------------------------------------------


def resize_footprint(
    source: Resolution, target: Resolution, dtype: str = "fp16"
) -> StageFootprint:
    """Separable Lanczos-3 keeps one intermediate at ``target.width x source.height``."""
    esize = dtype_bytes(dtype)
    rgb_fmt = PixelFormat.RGB_FP16 if esize == 2 else PixelFormat.RGB_FP32
    intermediate = Resolution(target.width, source.height)
    return StageFootprint(
        stage="resize",
        posts={
            "input_frame": frame_bytes(source, rgb_fmt),
            "horizontal_pass": frame_bytes(intermediate, rgb_fmt),
            "output_frame": frame_bytes(target, rgb_fmt),
        },
    )


# --------------------------------------------------------------------------
# RIFE stage
# --------------------------------------------------------------------------

# RIFE 4.x holds optical-flow pyramids at several scales and has no tiling.
# The design registers its per-frame-pair footprint as measurement post P4 and
# asserts no number. This module therefore refuses to invent one: a caller
# that wants a RIFE reservation must pass a measured or operator-declared
# bytes-per-frame-pair value.


class UnprobedFootprintError(RuntimeError):
    """Raised when a reservation needs a value that only measurement can supply."""


def rife_footprint(
    frame: Resolution,
    dtype: str = "fp16",
    measured_bytes_per_pair: int | None = None,
) -> StageFootprint:
    esize = dtype_bytes(dtype)
    rgb_fmt = PixelFormat.RGB_FP16 if esize == 2 else PixelFormat.RGB_FP32
    if measured_bytes_per_pair is None:
        raise UnprobedFootprintError(
            "RIFE per-frame-pair device bytes is measurement post P4 and has no "
            "predicted value in DESIGN #333 §8.3. Pass measured_bytes_per_pair "
            "from a P4 run or from an operator declaration."
        )
    return StageFootprint(
        stage="rife",
        posts={
            "input_pair": 2 * frame_bytes(frame, rgb_fmt),
            "flow_pyramids_and_output": measured_bytes_per_pair,
        },
    )


# --------------------------------------------------------------------------
# Codec stages
# --------------------------------------------------------------------------


def codec_pool_bytes(res: Resolution, pool_depth: int, fmt: PixelFormat) -> int:
    if pool_depth <= 0:
        raise ValueError("pool depth must be positive")
    return pool_depth * frame_bytes(res, fmt)


# --------------------------------------------------------------------------
# The tenant reservation (DESIGN #333 §6.2)
# --------------------------------------------------------------------------

# Per-card CUDA context and library overhead for a Class-3 executor tenant.
# §6.2 gives the range 0.5-1.0 GiB; the reservation takes the top of the range
# because under-reserving is what produces a runtime OOM.
TENANT_CTX_OVERHEAD_BYTES = 1 * GIB

# Driver-side device memory one in-process NVENC session holds beyond the
# input surfaces the chain hands it: the encoder's reconstructed-picture and
# reference buffers, its bitstream output buffers, and the libnvidia-encode
# context itself (#484).
#
# This post exists because moving the encode in-process moves that memory INTO
# our ledger. Under the ffmpeg subprocess fallback the same allocation exists
# on the same card and is invisible to every accounting this runtime does --
# it lives in a foreign process. That invisibility is the VRAM half of the
# ONE-RUNTIME argument, and the post is what discharges it.
#
# MEASURED, and it is NOT one number: it moves with geometry, so it is a
# function rather than a constant. Card 0 (RTX 3080), h264 P4/high_quality,
# 2026-08-03, read as the device-wide NVML free delta around session creation
# plus one encode with the input surface subtracted (NVML, not the torch
# allocator -- the #333 P3 lesson):
#
#     1280x720    51.8 MiB
#     3840x2160  263.8 MiB
#
# Two points, two unknowns, so the affine fit below REPRODUCES them rather
# than being validated by them -- a third geometry would be the first real
# test of the shape. The per-pixel term works out at ~30 B/px, which is about
# twenty NV12 frames: the reconstructed-picture and reference buffers scale
# with the frame, the libnvidia-encode context does not. Not a safety margin.
NVENC_SESSION_FIXED_BYTES = int(25.3 * MIB)
NVENC_SESSION_BYTES_PER_PIXEL = 30.15


def nvenc_session_bytes(resolution: "Resolution") -> int:
    """Device memory one in-process NVENC session holds at this geometry.

    Beyond the input surfaces the chain hands it: reconstructed-picture and
    reference buffers, bitstream output buffers, and the encode context.

    This post exists because moving the encode in-process moves that memory
    INTO our ledger. Under the ffmpeg subprocess fallback the same allocation
    exists on the same card and is invisible to every accounting this runtime
    does -- it lives in a foreign process. That invisibility is the VRAM half
    of the ONE-RUNTIME argument, and this function is what discharges it.
    """
    pixels = resolution.width * resolution.height
    return NVENC_SESSION_FIXED_BYTES + int(NVENC_SESSION_BYTES_PER_PIXEL * pixels)


@dataclass(frozen=True)
class ReservationBreakdown:
    """The §6.2 formula, itemised so a mismatch is attributable to one post."""

    posts: dict[str, int]

    @property
    def total_bytes(self) -> int:
        return sum(self.posts.values())

    @property
    def total_mib(self) -> int:
        return self.total_bytes // MIB

    def render(self) -> str:
        width = max(len(k) for k in self.posts)
        lines = [
            f"  {name.ljust(width)}  {value / MIB:10.1f} MiB"
            for name, value in sorted(self.posts.items(), key=lambda kv: -kv[1])
        ]
        lines.append(f"  {'TOTAL'.ljust(width)}  {self.total_bytes / MIB:10.1f} MiB")
        return "\n".join(lines)


def chain_reservation(
    *,
    source: Resolution,
    target: Resolution,
    streams_in_flight: int,
    dtype: str = "fp16",
    engine_device_bytes: int = 0,
    decoder_pool_depth: int = 4,
    encoder_pool_depth: int = 4,
    rife_measured_bytes_per_pair: int | None = None,
    with_rife: bool = True,
    with_sr: bool = True,
    sr_scale: int = SR_SCALE,
    with_resize: bool = True,
    ctx_overhead_bytes: int = TENANT_CTX_OVERHEAD_BYTES,
    inprocess_nvenc: bool = False,
) -> ReservationBreakdown:
    """Compute ``reserved(C)`` exactly as DESIGN #333 §6.2 specifies.

    ``reserved(C) = ctx_overhead + sum(engine bytes)
                    + sum over stages: streams_in_flight * peak_intermediate
                    + codec surface pools [+ in-process encoder session]``

    ``inprocess_nvenc`` adds the encoder session's own device memory (#484).
    It defaults to False so the reservation of a chain that still encodes
    through the ffmpeg subprocess is unchanged: that memory is real either
    way, but under the subprocess it belongs to a process this ledger does not
    own, and reserving for it here would double-count against a card the
    subprocess is already charging.
    """
    if streams_in_flight <= 0:
        raise ValueError("streams_in_flight must be positive")

    posts: dict[str, int] = {
        "tenant_ctx_overhead": ctx_overhead_bytes,
        "trt_engine_device_bytes": engine_device_bytes,
        "nvdec_surface_pool": codec_pool_bytes(
            source, decoder_pool_depth, PixelFormat.NV12
        ),
        "nvenc_surface_pool": codec_pool_bytes(
            target, encoder_pool_depth, PixelFormat.NV12
        ),
    }
    if inprocess_nvenc:
        posts["nvenc_session"] = nvenc_session_bytes(target)
    if with_sr:
        posts["stage_sr"] = streams_in_flight * sr_reserved_bytes(
            source, dtype, sr_scale
        )
    resize_input = source.scaled(sr_scale) if with_sr else source
    if with_resize and resize_input != target:
        posts["stage_resize"] = streams_in_flight * resize_footprint(
            resize_input, target, dtype
        ).reserved_bytes(SR_ALLOCATOR_OVERHEAD_FRACTION)
    if with_rife:
        posts["stage_rife"] = streams_in_flight * rife_footprint(
            target, dtype, rife_measured_bytes_per_pair
        ).reserved_bytes(SR_ALLOCATOR_OVERHEAD_FRACTION)
    return ReservationBreakdown(posts=posts)


def max_in_flight_for_budget(
    *,
    source: Resolution,
    target: Resolution,
    budget_bytes: int,
    dtype: str = "fp16",
    **kwargs,
) -> int:
    """Largest ``streams_in_flight`` whose reservation fits ``budget_bytes``.

    §8.4 rule 5: the in-flight cap is derived from the reservation, never
    configured independently, so a depth that does not fit cannot be
    configured in the first place.
    """
    depth = 0
    while True:
        candidate = depth + 1
        total = chain_reservation(
            source=source,
            target=target,
            streams_in_flight=candidate,
            dtype=dtype,
            **kwargs,
        ).total_bytes
        if total > budget_bytes:
            return depth
        depth = candidate
        if depth > 1024:  # pragma: no cover - guards a pathological budget
            return depth
