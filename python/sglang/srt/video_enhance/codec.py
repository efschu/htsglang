# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The decode, colour-conversion and encode ends of the enhance chain.

DESIGN #333 §8.1 fixes the chain as

    decode -> [colour] -> SR -> [resize] -> RIFE -> [colour] -> encode

and states that every arrow between ``decode`` and ``encode`` is a
device-to-device pointer hand-off. This module owns the two ends of that
sentence: NVDEC on the way in, NVENC on the way out, and the YUV<->RGB
conversions that sit immediately inside them.

Backends
--------
``PyNvVideoCodec`` is the primary backend on both ends. Version 2.2.0 exports
``DecodedFrame.__dlpack__``/``__dlpack_device__``, so a decoded surface
becomes a torch CUDA tensor through ``torch.from_dlpack`` without touching
host memory, and ``PyNvEncoder.Encode`` accepts any object exposing
``__cuda_array_interface__``, which a torch CUDA tensor does. That is the
zero-copy seam §8.1 requires and it exists in the installed version.

An ffmpeg subprocess backend covers containers and codecs PyNvVideoCodec
cannot open. It is a *coverage* fallback, not a performance path, and it
breaks the §8.1 property -- see the class docstrings for the exact reason.

Back-pressure
-------------
§8.4 rule 2 puts the stall at the decoder. :class:`DecodeStage` therefore has
no internal queue and no reader thread: it decodes only inside a
:meth:`DecodeStage.pull` call. When the ring downstream of it is full nobody
calls ``pull`` and the decoder simply stops touching the source.

Optional imports
----------------
``torch`` and ``PyNvVideoCodec`` are imported lazily so that importing this
module on a CPU-only host -- which is what the planner and the §8.3
arithmetic do -- costs nothing and fails nowhere.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Literal, Sequence

from sglang.srt.video_enhance.frame_math import (
    PixelFormat,
    Resolution,
    codec_pool_bytes,
    frame_bytes,
)
from sglang.srt.video_enhance.frames import Frame, StageBase

logger = logging.getLogger(__name__)

BackendName = Literal["auto", "pynvvideocodec", "ffmpeg"]

#: Decoder surface pool depth. Counted in the §6.2 reservation as
#: ``nvdec_surface_pool_bytes``; the default matches
#: ``chain_reservation``'s ``decoder_pool_depth``.
DEFAULT_DECODER_POOL_DEPTH = 4
DEFAULT_ENCODER_POOL_DEPTH = 4

#: Frames per emitted byte segment. §8.4 makes one HTTP chunk per encoded
#: muxed segment the response contract, so this is the chunk granularity the
#: client sees: 30 frames is about a second of 30 fps output, small enough
#: that the client starts rendering early and large enough that the chunk
#: framing overhead is irrelevant.
DEFAULT_SEGMENT_FRAMES = 30


class CodecError(RuntimeError):
    """A codec backend could not be opened, or produced something unusable."""


class CodecBackendUnavailable(CodecError):
    """The requested codec backend is not installed or not applicable."""


# --------------------------------------------------------------------------
# Lazy optional imports
# --------------------------------------------------------------------------


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise CodecError(
            "torch is required for the codec and colour stages; the frame "
            "arithmetic in frame_math stays importable without it"
        ) from exc
    return torch


def _import_pynvvideocodec():
    """Import PyNvVideoCodec.

    Kept as a named function rather than an inline import so backend
    selection can be exercised without the package present.
    """
    try:
        import PyNvVideoCodec as nvc
    except Exception as exc:  # noqa: BLE001 - the package raises non-ImportError
        # PyNvVideoCodec raises RuntimeError, not ImportError, when the driver
        # is too old or libnvidia-encode is missing, so a bare ImportError
        # guard would let that escape as a crash at chain-build time.
        raise CodecBackendUnavailable(f"PyNvVideoCodec unavailable: {exc}") from exc
    return nvc


def pynvvideocodec_available() -> bool:
    try:
        _import_pynvvideocodec()
    except CodecBackendUnavailable:
        return False
    return True


# --------------------------------------------------------------------------
# Colour: matrices and ranges
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class YuvMatrix:
    """A YCbCr matrix given by its two independent luma coefficients.

    Everything else is derived. Writing the nine matrix entries out by hand
    is how a transcode ends up with a green cast that survives review, so the
    only free parameters here are ``kr`` and ``kb`` as the standards define
    them.
    """

    name: str
    kr: float
    kb: float

    @property
    def kg(self) -> float:
        return 1.0 - self.kr - self.kb

    def to_rgb_coefficients(self) -> tuple[tuple[float, float, float], ...]:
        """3x3 matrix mapping (Y, U, V) -> (R, G, B), full-swing units."""
        kr, kb, kg = self.kr, self.kb, self.kg
        return (
            (1.0, 0.0, 2.0 * (1.0 - kr)),
            (1.0, -2.0 * kb * (1.0 - kb) / kg, -2.0 * kr * (1.0 - kr) / kg),
            (1.0, 2.0 * (1.0 - kb), 0.0),
        )

    def to_yuv_coefficients(self) -> tuple[tuple[float, float, float], ...]:
        """3x3 matrix mapping (R, G, B) -> (Y, U, V), full-swing units."""
        kr, kb, kg = self.kr, self.kb, self.kg
        return (
            (kr, kg, kb),
            (-kr / (2.0 * (1.0 - kb)), -kg / (2.0 * (1.0 - kb)), 0.5),
            (0.5, -kg / (2.0 * (1.0 - kr)), -kb / (2.0 * (1.0 - kr))),
        )


#: ITU-R BT.709-6 Table 3 luma coefficients. The HD default.
BT709 = YuvMatrix("709", kr=0.2126, kb=0.0722)
#: ITU-R BT.601-7 luma coefficients. Standard-definition sources.
BT601 = YuvMatrix("601", kr=0.299, kb=0.114)

_MATRICES: dict[str, YuvMatrix] = {
    "709": BT709,
    "bt709": BT709,
    "601": BT601,
    "bt601": BT601,
}


@dataclass(frozen=True)
class ColorRange:
    """Where the 8-bit code values for full-swing 0.0-1.0 signals land."""

    name: str
    luma_offset: float
    luma_scale: float
    chroma_offset: float
    chroma_scale: float


#: "MPEG"/"TV" range: Y in 16-235, chroma in 16-240. What NVDEC hands back for
#: essentially every real-world source.
LIMITED_RANGE = ColorRange("limited", 16.0, 219.0, 128.0, 224.0)
#: "JPEG"/"PC" range: the full 0-255 code space.
FULL_RANGE = ColorRange("full", 0.0, 255.0, 128.0, 255.0)

_RANGES: dict[str, ColorRange] = {
    "limited": LIMITED_RANGE,
    "tv": LIMITED_RANGE,
    "mpeg": LIMITED_RANGE,
    "full": FULL_RANGE,
    "pc": FULL_RANGE,
    "jpeg": FULL_RANGE,
}


def resolve_matrix(matrix: str | YuvMatrix) -> YuvMatrix:
    if isinstance(matrix, YuvMatrix):
        return matrix
    try:
        return _MATRICES[matrix.lower()]
    except KeyError:
        raise ValueError(
            f"unknown colour matrix {matrix!r}, expected one of {sorted(_MATRICES)}"
        ) from None


def resolve_range(color_range: str | ColorRange) -> ColorRange:
    if isinstance(color_range, ColorRange):
        return color_range
    try:
        return _RANGES[color_range.lower()]
    except KeyError:
        raise ValueError(
            f"unknown colour range {color_range!r}, expected one of {sorted(_RANGES)}"
        ) from None


def rgb_format_for_dtype(dtype: str) -> PixelFormat:
    """Mirror of VSGAN's format naming: fp16 is RGBH, fp32 is RGBS."""
    if dtype in ("fp16", "half", "float16"):
        return PixelFormat.RGB_FP16
    if dtype in ("fp32", "float", "float32"):
        return PixelFormat.RGB_FP32
    raise ValueError(f"colour stages produce fp16 (RGBH) or fp32 (RGBS), got {dtype!r}")


def _torch_dtype(dtype: str):
    torch = _import_torch()
    if rgb_format_for_dtype(dtype) is PixelFormat.RGB_FP16:
        return torch.float16
    return torch.float32


# --------------------------------------------------------------------------
# Colour: the conversions themselves
# --------------------------------------------------------------------------


def nv12_to_rgb(
    nv12,
    *,
    matrix: str | YuvMatrix = BT709,
    color_range: str | ColorRange = LIMITED_RANGE,
    dtype: str = "fp16",
):
    """NV12 -> planar RGB, NCHW, values in [0, 1].

    ``nv12`` is a ``(height * 3 // 2, width)`` uint8 tensor: the NVDEC surface
    layout, luma plane followed by an interleaved chroma plane at half
    resolution in both axes. The output is ``(1, 3, height, width)`` in fp16
    (VSGAN's RGBH) or fp32 (RGBS), which is the layout the SR, resize and RIFE
    stages expect.

    Chroma is upsampled nearest-neighbour. That is what NVDEC's own built-in
    conversion does, and it keeps the operation a pure gather; a bilinear
    chroma upsample would shift the chroma siting by half a chroma sample
    relative to the decoder's own output and break parity against a
    NVDEC-converted reference.

    The arithmetic runs in fp32 regardless of the output dtype. The
    intermediate is transient and the alternative -- accumulating the
    three-term matrix product in fp16 -- costs about an LSB against the fp32
    parity reference for no measurable time.
    """
    torch = _import_torch()
    mat = resolve_matrix(matrix)
    rng = resolve_range(color_range)

    if nv12.dim() != 2:
        raise ValueError(
            f"NV12 must be a 2-D (rows, width) plane, got {tuple(nv12.shape)}"
        )
    rows, width = int(nv12.shape[0]), int(nv12.shape[1])
    if rows % 3 or width % 2:
        raise ValueError(
            f"NV12 plane {rows}x{width} is not a valid 4:2:0 layout: rows must be "
            "divisible by 3 (height * 3 / 2) and width must be even"
        )
    height = rows // 3 * 2

    luma = nv12[:height, :].to(torch.float32)
    chroma = nv12[height:, :].to(torch.float32).reshape(height // 2, width // 2, 2)

    y = (luma - rng.luma_offset) / rng.luma_scale
    uv = (chroma - rng.chroma_offset) / rng.chroma_scale
    u = _upsample2x(uv[..., 0], height, width)
    v = _upsample2x(uv[..., 1], height, width)

    (_, _, c_rv), (_, c_gu, c_gv), (_, c_bu, _) = mat.to_rgb_coefficients()
    r = y + c_rv * v
    g = y + c_gu * u + c_gv * v
    b = y + c_bu * u

    rgb = torch.stack((r, g, b), dim=0).clamp_(0.0, 1.0)
    return rgb.unsqueeze(0).to(_torch_dtype(dtype))


def rgb_to_nv12(
    rgb,
    *,
    matrix: str | YuvMatrix = BT709,
    color_range: str | ColorRange = LIMITED_RANGE,
):
    """Planar RGB in [0, 1] -> NV12, the layout NVENC takes as input.

    Accepts ``(1, 3, H, W)`` or ``(3, H, W)`` and returns a
    ``(H * 3 // 2, W)`` uint8 tensor on the same device. Chroma is box-filtered
    to half resolution -- averaging the 2x2 block before quantisation rather
    than after, so the downsample does not inherit the rounding error of four
    separate 8-bit values.
    """
    torch = _import_torch()
    mat = resolve_matrix(matrix)
    rng = resolve_range(color_range)

    if rgb.dim() == 4:
        if int(rgb.shape[0]) != 1:
            raise ValueError(
                f"the chain carries one frame per work unit, got batch {rgb.shape[0]}"
            )
        rgb = rgb[0]
    if rgb.dim() != 3 or int(rgb.shape[0]) != 3:
        raise ValueError(f"expected planar RGB (3, H, W), got {tuple(rgb.shape)}")
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    if height % 2 or width % 2:
        raise ValueError(f"NV12 requires even dimensions, got {width}x{height}")

    x = rgb.to(torch.float32).clamp(0.0, 1.0)
    (c_yr, c_yg, c_yb), (c_ur, c_ug, c_ub), (c_vr, c_vg, c_vb) = (
        mat.to_yuv_coefficients()
    )
    r, g, b = x[0], x[1], x[2]
    y = c_yr * r + c_yg * g + c_yb * b
    u = c_ur * r + c_ug * g + c_ub * b
    v = c_vr * r + c_vg * g + c_vb * b

    y8 = y * rng.luma_scale + rng.luma_offset
    u8 = _downsample2x(u) * rng.chroma_scale + rng.chroma_offset
    v8 = _downsample2x(v) * rng.chroma_scale + rng.chroma_offset

    luma = _quantize_u8(y8)
    uv = _quantize_u8(torch.stack((u8, v8), dim=-1)).reshape(height // 2, width)
    return torch.cat((luma, uv), dim=0)


def _upsample2x(plane, height: int, width: int):
    """Replicate each chroma sample over its 2x2 luma footprint."""
    return plane[:, None, :, None].expand(-1, 2, -1, 2).reshape(height, width)


def _downsample2x(plane):
    h, w = int(plane.shape[0]), int(plane.shape[1])
    return plane.reshape(h // 2, 2, w // 2, 2).mean(dim=(1, 3))


def _quantize_u8(x):
    torch = _import_torch()
    return x.round().clamp_(0.0, 255.0).to(torch.uint8)


class ColorToRgbStage(StageBase):
    """§8.1's first bracketed stage: NVDEC's NV12 surface into the RGB chain."""

    name = "color_to_rgb"

    def __init__(
        self,
        *,
        matrix: str | YuvMatrix = BT709,
        color_range: str | ColorRange = LIMITED_RANGE,
        dtype: str = "fp16",
    ) -> None:
        self.matrix = resolve_matrix(matrix)
        self.color_range = resolve_range(color_range)
        self.dtype = dtype
        self.out_format = rgb_format_for_dtype(dtype)

    def process(self, frames: Sequence[Frame]) -> tuple[Frame, ...]:
        out: list[Frame] = []
        for frame in frames:
            if frame.end_of_stream:
                out.append(frame)
                continue
            if frame.format is not PixelFormat.NV12:
                raise ValueError(f"{self.name} reads NV12, got {frame.format.value}")
            rgb = nv12_to_rgb(
                frame.data,
                matrix=self.matrix,
                color_range=self.color_range,
                dtype=self.dtype,
            )
            out.append(
                frame.with_data(rgb, format=self.out_format).require_device(self.name)
            )
        return tuple(out)


class ColorToYuvStage(StageBase):
    """§8.1's second bracketed stage: the RGB chain back into NVENC's NV12.

    ``dtype`` names the precision of the *incoming* RGB (RGBH for fp16, RGBS
    for fp32); the output is always 8-bit NV12. It is checked rather than
    applied, so a chain whose two colour stages were configured with different
    precisions fails here with the mismatch named instead of inside NVENC.
    """

    name = "color_to_yuv"

    def __init__(
        self,
        *,
        matrix: str | YuvMatrix = BT709,
        color_range: str | ColorRange = LIMITED_RANGE,
        dtype: str = "fp16",
    ) -> None:
        self.matrix = resolve_matrix(matrix)
        self.color_range = resolve_range(color_range)
        self.dtype = dtype
        self.in_format = rgb_format_for_dtype(dtype)

    def process(self, frames: Sequence[Frame]) -> tuple[Frame, ...]:
        out: list[Frame] = []
        for frame in frames:
            if frame.end_of_stream:
                out.append(frame)
                continue
            if frame.format is not self.in_format:
                raise ValueError(
                    f"{self.name} configured for {self.in_format.value}, got "
                    f"{frame.format.value}"
                )
            nv12 = rgb_to_nv12(
                frame.data, matrix=self.matrix, color_range=self.color_range
            )
            out.append(
                frame.with_data(nv12, format=PixelFormat.NV12).require_device(self.name)
            )
        return tuple(out)


# --------------------------------------------------------------------------
# Source metadata
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceInfo:
    """What the chain needs to know about a source before it opens it."""

    resolution: Resolution
    #: None when the container does not carry a frame count and no full scan
    #: was requested. Callers must not treat it as zero.
    frame_count: int | None
    fps: Fraction
    pixel_format: PixelFormat
    codec: str
    container: str = ""
    #: Which decode backend produced or will consume this. Recorded so an E2E
    #: run can attribute its numbers to a backend.
    backend: str = "unopened"

    def with_backend(self, backend: str) -> "SourceInfo":
        return SourceInfo(
            resolution=self.resolution,
            frame_count=self.frame_count,
            fps=self.fps,
            pixel_format=self.pixel_format,
            codec=self.codec,
            container=self.container,
            backend=backend,
        )


def _parse_rational(text: str | None, default: Fraction) -> Fraction:
    if not text:
        return default
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            num_i, den_i = int(num), int(den)
        except ValueError:
            return default
        if den_i == 0:
            return default
        return Fraction(num_i, den_i)
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return default


def parse_ffprobe_output(payload: str) -> SourceInfo:
    """Turn ``ffprobe -of json`` output into a :class:`SourceInfo`.

    Split out from the subprocess call so the parsing -- which is where the
    surprises are, not in ``Popen`` -- is testable without ffmpeg installed.
    """
    doc = json.loads(payload)
    streams = [s for s in doc.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise CodecError("source has no video stream")
    stream = streams[0]

    width, height = stream.get("width"), stream.get("height")
    if not width or not height:
        raise CodecError("source video stream reports no resolution")

    fps = _parse_rational(
        stream.get("avg_frame_rate") or stream.get("r_frame_rate"), Fraction(0)
    )
    if fps <= 0:
        fps = _parse_rational(stream.get("r_frame_rate"), Fraction(0))

    frame_count: int | None = None
    raw_count = stream.get("nb_frames")
    if raw_count not in (None, "", "N/A"):
        try:
            frame_count = int(raw_count)
        except (TypeError, ValueError):
            frame_count = None
    if frame_count is None and fps > 0:
        duration = stream.get("duration") or doc.get("format", {}).get("duration")
        if duration not in (None, "", "N/A"):
            try:
                frame_count = int(Fraction(str(duration)) * fps)
            except (ValueError, ZeroDivisionError):
                frame_count = None

    return SourceInfo(
        resolution=Resolution(int(width), int(height)),
        frame_count=frame_count,
        fps=fps,
        # The chain decodes to NV12 whatever the source's own subsampling is;
        # a 10-bit source is rejected at open time, not silently truncated.
        pixel_format=PixelFormat.NV12,
        codec=str(stream.get("codec_name", "")),
        container=str(doc.get("format", {}).get("format_name", "")),
        backend="ffprobe",
    )


def probe_source(source: str | Path, *, ffprobe_path: str = "ffprobe") -> SourceInfo:
    """Read source metadata with ffprobe.

    ffprobe rather than the demuxer because the answer must be available
    before a backend is chosen -- backend selection depends on the codec --
    and because it costs no GPU.
    """
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, check=True, text=True)
    except FileNotFoundError as exc:
        raise CodecError(f"{ffprobe_path} not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise CodecError(f"ffprobe failed for {source}: {exc.stderr.strip()}") from exc
    return parse_ffprobe_output(completed.stdout)


#: ffprobe codec names that NVDEC can take, mapped to the
#: ``nvc.cudaVideoCodec`` member. Anything outside this table goes to the
#: ffmpeg fallback, which is the reason the fallback exists.
NVDEC_CODECS: dict[str, str] = {
    "h264": "H264",
    "hevc": "HEVC",
    "av1": "AV1",
    "vp8": "VP8",
    "vp9": "VP9",
    "mpeg1video": "MPEG1",
    "mpeg2video": "MPEG2",
    "mpeg4": "MPEG4",
    "vc1": "VC1",
    "mjpeg": "JPEG",
}

#: NVENC codecs, mapped to the ffmpeg encoder name used by the fallback.
NVENC_CODECS: dict[str, str] = {
    "h264": "h264_nvenc",
    "hevc": "hevc_nvenc",
    "av1": "av1_nvenc",
}

_CONTENT_TYPES: dict[tuple[str, str], str] = {
    ("annexb", "h264"): "video/H264",
    ("annexb", "hevc"): "video/H265",
    ("annexb", "av1"): "video/AV1",
    ("mpegts", "h264"): "video/mp2t",
    ("mpegts", "hevc"): "video/mp2t",
    ("mpegts", "av1"): "video/mp2t",
}


def select_decode_backend(
    codec: str, requested: BackendName = "auto", *, nvc_available: bool | None = None
) -> str:
    """Resolve ``auto`` to a concrete decode backend, or validate an explicit one.

    Pure, so the policy is testable without either backend installed.
    """
    if requested == "ffmpeg":
        # Probing availability first would import PyNvVideoCodec -- and with
        # it the driver check -- for a caller that has already ruled it out.
        return "ffmpeg"
    available = pynvvideocodec_available() if nvc_available is None else nvc_available
    if requested == "pynvvideocodec":
        if not available:
            raise CodecBackendUnavailable(
                "backend='pynvvideocodec' requested but the package is unavailable"
            )
        if codec.lower() not in NVDEC_CODECS:
            raise CodecBackendUnavailable(
                f"NVDEC has no decoder for {codec!r}; use backend='ffmpeg'"
            )
        return "pynvvideocodec"
    if requested != "auto":
        raise ValueError(f"unknown decode backend {requested!r}")
    if available and codec.lower() in NVDEC_CODECS:
        return "pynvvideocodec"
    return "ffmpeg"


def select_encode_backend(
    codec: str,
    container: str,
    requested: BackendName = "auto",
    *,
    nvc_available: bool | None = None,
) -> str:
    """Resolve ``auto`` to a concrete encode backend, or validate an explicit one.

    ``mpegts`` forces ffmpeg: PyNvVideoCodec's encoder emits an elementary
    stream and its muxer writes a file, neither of which is a transport-stream
    chunk.
    """
    if codec.lower() not in NVENC_CODECS:
        raise ValueError(
            f"unknown encode codec {codec!r}, expected one of {sorted(NVENC_CODECS)}"
        )
    if container not in ("annexb", "mpegts"):
        raise ValueError(
            f"unknown container {container!r}, expected 'annexb' or 'mpegts'"
        )
    if requested == "ffmpeg":
        # See select_decode_backend: do not import the package to rule it out.
        return "ffmpeg"
    available = pynvvideocodec_available() if nvc_available is None else nvc_available
    if requested == "pynvvideocodec":
        if not available:
            raise CodecBackendUnavailable(
                "backend='pynvvideocodec' requested but the package is unavailable"
            )
        if container != "annexb":
            raise CodecBackendUnavailable(
                f"container {container!r} needs a muxer; use backend='ffmpeg'"
            )
        return "pynvvideocodec"
    if requested != "auto":
        raise ValueError(f"unknown encode backend {requested!r}")
    if available and container == "annexb":
        return "pynvvideocodec"
    return "ffmpeg"


# --------------------------------------------------------------------------
# ffmpeg command construction
# --------------------------------------------------------------------------


def build_ffmpeg_decode_command(
    source: str | Path,
    resolution: Resolution,
    *,
    ffmpeg_path: str = "ffmpeg",
    device_id: int = 0,
    start_frame: int = 0,
    frame_rate: Fraction | None = None,
    frame_limit: int | None = None,
) -> list[str]:
    """The fallback decode command: CUVID decode, host-staged NV12 out.

    ``-hwaccel cuda -hwaccel_output_format cuda`` keeps the decode itself on
    NVDEC, but ffmpeg has no way to hand a CUDA frame to another process, so
    ``hwdownload`` is unavoidable and the frames cross PCIe twice on their way
    into the chain.

    ``start_frame`` is what makes a multi-card shard possible: each card needs
    only its own stretch of the timeline. It is expressed as an *input* seek
    (``-ss`` before ``-i``), which decodes forward from the preceding keyframe
    and discards, rather than as a ``select`` filter, which would decode the
    entire prefix on every card and give the last card in the plan the whole
    clip to chew through before it starts working.

    The seek target is placed half a frame interval early, so the frame that
    should be first is unambiguously the first one at or after the seek point
    rather than a coin flip on floating-point rounding. ``frame_rate`` is
    required whenever ``start_frame`` is non-zero, because a frame index
    cannot be converted to a seek time without it.
    """
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-hwaccel_device",
        str(device_id),
        "-hwaccel_output_format",
        "cuda",
    ]
    if start_frame:
        if frame_rate is None or frame_rate <= 0:
            raise ValueError(
                "start_frame needs frame_rate: a frame index cannot be turned "
                "into a seek time without the rate it was counted at"
            )
        seek = (Fraction(start_frame) - Fraction(1, 2)) / frame_rate
        cmd += ["-ss", f"{float(seek):.6f}"]
    cmd += [
        "-i",
        str(source),
        "-vf",
        f"hwdownload,format=nv12,scale={resolution.width}:{resolution.height}",
    ]
    if frame_limit is not None:
        if frame_limit < 1:
            raise ValueError("frame_limit must be positive")
        cmd += ["-frames:v", str(frame_limit)]
    cmd += [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "nv12",
        "-",
    ]
    return cmd


def build_ffmpeg_encode_command(
    resolution: Resolution,
    fps: Fraction,
    *,
    codec: str = "h264",
    container: str = "annexb",
    bitrate: int | None = None,
    preset: str = "p4",
    ffmpeg_path: str = "ffmpeg",
    device_id: int = 0,
) -> list[str]:
    """The fallback encode command: raw NV12 on stdin, muxed bytes on stdout."""
    try:
        encoder = NVENC_CODECS[codec.lower()]
    except KeyError:
        raise ValueError(f"unknown encode codec {codec!r}") from None
    muxer = "h264" if container == "annexb" and codec.lower() == "h264" else container
    if container == "annexb" and codec.lower() == "hevc":
        muxer = "hevc"
    if container == "annexb" and codec.lower() == "av1":
        muxer = "obu"
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "nv12",
        "-s",
        f"{resolution.width}x{resolution.height}",
        "-r",
        f"{fps.numerator}/{fps.denominator}",
        "-i",
        "-",
        "-c:v",
        encoder,
        "-gpu",
        str(device_id),
        "-preset",
        preset,
    ]
    if bitrate:
        cmd += ["-b:v", str(bitrate)]
    cmd += ["-f", muxer, "-"]
    return cmd


# --------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------


class _PyNvDecodeBackend:
    """Demuxer plus NVDEC, driven one packet at a time.

    The packet loop is the reason this is the primary backend: nothing decodes
    until :meth:`read` asks for a frame, which is exactly the §8.4 rule-2
    behaviour. ``ThreadedDecoder`` would be simpler to call and is rejected
    for the opposite reason -- it keeps a decoder thread running ahead of the
    consumer.
    """

    name = "pynvvideocodec"

    def __init__(
        self,
        source: str | Path,
        info: SourceInfo,
        *,
        device_id: int = 0,
        copy_surfaces: bool = True,
    ) -> None:
        nvc = _import_pynvvideocodec()
        self._nvc = nvc
        self._info = info
        self._copy_surfaces = copy_surfaces
        self._demuxer = nvc.CreateDemuxer(str(source))
        self._packets = iter(self._demuxer)
        self._decoder = nvc.CreateDecoder(
            gpuid=device_id,
            codec=self._demuxer.GetNvCodecId(),
            usedevicememory=True,
        )
        # Frames a single packet produced beyond what the caller asked for.
        # Bounded by one packet's output, so it is a carry-over, not a queue.
        self._carry: list[object] = []
        self._exhausted = False
        self._checked_format = False

    def read(self, count: int) -> list[object]:
        out: list[object] = []
        while len(out) < count:
            if self._carry:
                out.append(self._to_tensor(self._carry.pop(0)))
                continue
            if self._exhausted:
                break
            try:
                packet = next(self._packets)
            except StopIteration:
                self._exhausted = True
                break
            decoded = list(self._decoder.Decode(packet))
            if not decoded:
                continue
            self._carry.extend(decoded)
        return out

    def _to_tensor(self, decoded):
        torch = _import_torch()
        if not self._checked_format:
            self._check_pixel_format()
            self._checked_format = True
        tensor = torch.from_dlpack(decoded)
        tensor = _as_nv12_plane(tensor, self._info.resolution)
        if self._copy_surfaces:
            # NVDEC recycles surfaces from a pool of fixed depth. Copying into
            # chain-owned device memory decouples frame lifetime from that
            # pool at the cost of one device-to-device copy (2.97 MiB at
            # 1080p NV12). With copy_surfaces=False the caller must consume
            # the frame before the pool wraps, which the ring depth does not
            # guarantee on its own.
            tensor = tensor.clone()
        return tensor

    def _check_pixel_format(self) -> None:
        get_format = getattr(self._decoder, "GetPixelFormat", None)
        if get_format is None:
            return
        fmt = get_format()
        name = getattr(fmt, "name", str(fmt))
        if name != "NV12":
            raise CodecError(
                f"decoder produced {name}; the chain's frame arithmetic (§8.3) is "
                "8-bit NV12 and a higher bit depth would have to be budgeted "
                "separately"
            )

    def close(self) -> None:
        self._carry.clear()
        self._decoder = None
        self._packets = iter(())
        self._demuxer = None


def _as_nv12_plane(tensor, resolution: Resolution):
    """Normalise whatever shape the DLPack export used to ``(H*3//2, W)``."""
    rows = resolution.height * 3 // 2
    if tuple(tensor.shape) == (rows, resolution.width):
        return tensor
    if tensor.numel() != rows * resolution.width:
        raise CodecError(
            f"decoded surface has {tensor.numel()} elements, expected "
            f"{rows * resolution.width} for NV12 {resolution}"
        )
    return tensor.reshape(rows, resolution.width)


class _FfmpegDecodeBackend:
    """Coverage fallback: an ffmpeg subprocess emitting raw NV12 on stdout.

    This backend violates the §8.1 no-host-round-trip property. ffmpeg decodes
    on NVDEC, then ``hwdownload`` copies each frame to host memory, this
    process reads it off a pipe and uploads it again. At 1080p that is 2.97
    MiB down and 2.97 MiB up per frame plus a pipe copy. It exists so that a
    container or codec PyNvVideoCodec cannot open still runs, not because it
    is fast, and a chain that lands on it should say so in its report.
    """

    name = "ffmpeg"

    def __init__(
        self,
        source: str | Path,
        info: SourceInfo,
        *,
        device_id: int = 0,
        device: str | None = None,
        ffmpeg_path: str = "ffmpeg",
        start_frame: int = 0,
        frame_limit: int | None = None,
    ) -> None:
        if shutil.which(ffmpeg_path) is None:
            raise CodecBackendUnavailable(f"{ffmpeg_path} not found on PATH")
        self._info = info
        self._device = device if device is not None else f"cuda:{device_id}"
        self._bytes_per_frame = frame_bytes(info.resolution, PixelFormat.NV12)
        self._rows = info.resolution.height * 3 // 2
        self._proc = subprocess.Popen(
            build_ffmpeg_decode_command(
                source,
                info.resolution,
                ffmpeg_path=ffmpeg_path,
                device_id=device_id,
                start_frame=start_frame,
                frame_rate=info.fps,
                frame_limit=frame_limit,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._exhausted = False

    def read(self, count: int) -> list[object]:
        torch = _import_torch()
        out: list[object] = []
        while len(out) < count and not self._exhausted:
            raw = self._read_exactly(self._bytes_per_frame)
            if raw is None:
                self._exhausted = True
                break
            host = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
                self._rows, self._info.resolution.width
            )
            out.append(host.to(self._device))
        return out

    def _read_exactly(self, size: int) -> bytes | None:
        assert self._proc.stdout is not None
        buf = bytearray()
        while len(buf) < size:
            chunk = self._proc.stdout.read(size - len(buf))
            if not chunk:
                return None if not buf else bytes(buf).ljust(size, b"\x00")
            buf += chunk
        return bytes(buf)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        # The chain routinely stops early (client disconnect, cancellation),
        # so killing a still-writing ffmpeg is the normal path, not an error.
        proc.kill()
        proc.communicate()


class DecodeStage(StageBase):
    """The chain's source: a video file in, device-resident NV12 frames out.

    Not a 1-in-1-out stage. It produces frames on demand through
    :meth:`pull` or iteration and holds no queue of its own, which is what
    makes §8.4 rule 2 -- back-pressure stops the decoder -- fall out of the
    structure instead of needing a mechanism.

    A stage instance is bound to one source, because it owns that source's
    demuxer and decoder session. ``source`` may be left unset at construction
    -- a tenant builds its stage set from the chain, before any request has
    arrived -- and supplied later with :meth:`set_source`; ``resolution``
    then carries the geometry the chain was planned for.
    """

    name = "decode"

    def __init__(
        self,
        source: str | Path | None = None,
        *,
        resolution: Resolution | None = None,
        backend: BackendName = "auto",
        device_id: int = 0,
        device: str | None = None,
        pool_depth: int = DEFAULT_DECODER_POOL_DEPTH,
        info: SourceInfo | None = None,
        copy_surfaces: bool = True,
        emit_eos: bool = True,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        start_frame: int = 0,
        frame_limit: int | None = None,
    ) -> None:
        self.source = source
        self.resolution = resolution
        self.requested_backend = backend
        self.device_id = device_id
        self.device = device
        self.pool_depth = pool_depth
        self.copy_surfaces = copy_surfaces
        self.emit_eos = emit_eos
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        # Multi-card sharding (``multicard.py``): a chunk decodes only its own
        # stretch of the timeline. ``start_frame`` is a real seek on the
        # ffmpeg backend and a decode-and-discard on the NVDEC backend, which
        # has no seek of its own -- the cost is named in ``_open``.
        if start_frame < 0:
            raise ValueError("start_frame cannot be negative")
        if frame_limit is not None and frame_limit < 1:
            raise ValueError("frame_limit must be positive")
        self.start_frame = start_frame
        self.frame_limit = frame_limit

        self._info = info
        self._backend: object | None = None
        # Frame indices are absolute in the source timeline, not relative to
        # the shard. A shard's ``encode_filter`` is written against the index
        # of the seam frame in the source, and interpolated frames inherit the
        # index of the earlier frame of their pair, so a shard-relative
        # numbering would make the two disagree.
        self._index = start_frame
        self._cancelled = False
        self._eos_emitted = False
        self.frames_decoded = 0

    # -- lifecycle ---------------------------------------------------------

    def set_source(self, source: str | Path, *, info: SourceInfo | None = None) -> None:
        """Bind a source to a stage that was built without one.

        Rebinding after decoding has started would silently splice two
        sources into one output timeline, so it is refused.
        """
        if self._backend is not None:
            raise CodecError(
                "the decode stage is already open on "
                f"{self.source!r}; build a new stage per source"
            )
        self.source = source
        self._info = info

    def warmup(self) -> None:
        self._open()

    def _open(self):
        if self._backend is not None:
            return self._backend
        if self.source is None:
            raise CodecError(
                "the decode stage has no source; pass one to the constructor or "
                "call set_source() before pulling frames"
            )
        if self._info is None or self._info.backend == "unopened":
            self._info = probe_source(self.source, ffprobe_path=self.ffprobe_path)
        if self.resolution is not None and self._info.resolution != self.resolution:
            raise CodecError(
                f"source is {self._info.resolution} but the chain was planned for "
                f"{self.resolution}; the engine shapes and the §6.2 reservation "
                "are both derived from the planned size"
            )
        chosen = select_decode_backend(self._info.codec, self.requested_backend)
        if chosen == "pynvvideocodec":
            self._backend = _PyNvDecodeBackend(
                self.source,
                self._info,
                device_id=self.device_id,
                copy_surfaces=self.copy_surfaces,
            )
            if self.start_frame:
                # NVDEC through PyNvVideoCodec is driven packet by packet and
                # exposes no seek, so the only way to reach frame N is to
                # decode and throw away the N before it. That is correct but
                # it is not free: the last chunk of a multi-card plan pays for
                # the whole prefix. The ffmpeg backend, which seeks, is the
                # one to use for sharded work -- said here rather than
                # discovered from a flat speedup curve.
                logger.warning(
                    "decode backend %s cannot seek; reaching frame %d costs a "
                    "decode-and-discard of every frame before it",
                    chosen,
                    self.start_frame,
                )
                remaining = self.start_frame
                while remaining > 0:
                    got = self._backend.read(min(remaining, 32))
                    if not got:
                        raise CodecError(
                            f"source ended before frame {self.start_frame}; a "
                            "shard was planned against a frame count the source "
                            "does not have"
                        )
                    remaining -= len(got)
        else:
            self._backend = _FfmpegDecodeBackend(
                self.source,
                self._info,
                device_id=self.device_id,
                device=self.device,
                ffmpeg_path=self.ffmpeg_path,
                start_frame=self.start_frame,
                frame_limit=self.frame_limit,
            )
        self._info = self._info.with_backend(chosen)
        return self._backend

    def probe(self) -> SourceInfo:
        """Source metadata with the resolved backend filled in.

        Opens the backend, because "which backend did this run use" is only
        answerable once the choice has actually been made.
        """
        self._open()
        assert self._info is not None
        return self._info

    @property
    def info(self) -> SourceInfo:
        """Source metadata without forcing the backend open."""
        if self._info is None:
            if self.source is None:
                raise CodecError("the decode stage has no source to describe")
            self._info = probe_source(self.source, ffprobe_path=self.ffprobe_path)
        return self._info

    @property
    def backend_name(self) -> str:
        return self.info.backend

    @property
    def pool_bytes(self) -> int:
        """``nvdec_surface_pool_bytes`` for the §6.2 reservation.

        Answerable before a source is bound, from the planned resolution: the
        reservation has to be computable at configuration time (§6.2), which
        is before any request has named a file.
        """
        resolution = self.resolution if self.source is None else self.info.resolution
        if resolution is None:
            raise CodecError("pool_bytes needs either a source or a planned resolution")
        return codec_pool_bytes(resolution, self.pool_depth, PixelFormat.NV12)

    def cancel(self) -> None:
        """Stop producing frames. Safe to call from another thread."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def close(self) -> None:
        self._cancelled = True
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.close()

    # -- production --------------------------------------------------------

    def pull(self, n: int = 1) -> list[Frame]:
        """Decode at most ``n`` frames. Returns fewer at end of source.

        Returns an empty list once the source is exhausted and the
        end-of-stream sentinel has been handed out, and immediately after
        :meth:`cancel`.
        """
        if n <= 0:
            raise ValueError("pull count must be positive")
        if self._cancelled:
            return []
        backend = self._open()
        assert self._info is not None

        if self.frame_limit is not None:
            n = min(n, self.frame_limit - self.frames_decoded)
            if n <= 0:
                # The shard's range is complete. The sentinel still has to go
                # out once, or the downstream stages never see end of stream.
                if self._eos_emitted or not self.emit_eos:
                    return []
                self._eos_emitted = True
                return [Frame.eos(self._index)]

        tensors = backend.read(n)
        frames = [
            Frame(
                data=tensor,
                resolution=self._info.resolution,
                format=PixelFormat.NV12,
                index=self._index + offset,
                pts=None,
            )
            for offset, tensor in enumerate(tensors)
        ]
        self._index += len(frames)
        self.frames_decoded += len(frames)

        if len(frames) < n and not self._eos_emitted and self.emit_eos:
            self._eos_emitted = True
            frames.append(Frame.eos(self._index))
        return frames

    def __iter__(self) -> Iterator[Frame]:
        while True:
            batch = self.pull(1)
            if not batch:
                return
            yield from batch

    def process(self, frames: Sequence[Frame] = ()) -> Sequence[Frame]:
        """Source-stage form of the protocol: input is ignored."""
        if frames:
            raise ValueError("the decode stage is a source and takes no input frames")
        return self.pull(1)


# --------------------------------------------------------------------------
# Encode
# --------------------------------------------------------------------------


class _PyNvEncodeBackend:
    """NVENC through PyNvVideoCodec, taking device memory directly.

    ``PyNvEncoder.Encode`` accepts any object exposing
    ``__cuda_array_interface__``; a contiguous uint8 torch CUDA tensor of
    shape ``(H*3//2, W)`` is exactly the NV12 input NVENC wants, so the frame
    goes from the colour stage into the encoder without leaving VRAM.
    """

    name = "pynvvideocodec"

    def __init__(
        self,
        resolution: Resolution,
        *,
        codec: str = "h264",
        device_id: int = 0,
        bitrate: int | None = None,
        preset: str = "P4",
        tuning_info: str = "high_quality",
        extra_options: dict[str, str] | None = None,
    ) -> None:
        nvc = _import_pynvvideocodec()
        options: dict[str, object] = {
            "codec": codec,
            "preset": preset,
            "tuning_info": tuning_info,
            "gpu_id": device_id,
        }
        if bitrate:
            options["bitrate"] = bitrate
        options.update(extra_options or {})
        self._encoder = nvc.CreateEncoder(
            resolution.width,
            resolution.height,
            "NV12",
            False,  # usecpuinputbuffer: the input is a device pointer
            **options,
        )

    def encode(self, tensor) -> list[bytes]:
        # PyNvVideoCodec 2.2.0 probes the input for __dlpack__ first and calls
        # it with the stream as a POSITIONAL argument. torch's
        # Tensor.__dlpack__ takes the stream keyword-only, so handing it a
        # torch tensor raises TypeError from inside the encoder and leaves the
        # NVENC session in a state whose next use faults. The documented
        # alternative input form is __cuda_array_interface__, so the tensor is
        # wrapped in an object that exposes only that.
        return _packets_to_bytes(self._encoder.Encode(_CudaArrayView.wrap(tensor)))

    def flush(self) -> list[bytes]:
        if self._encoder is None:
            return []
        return _packets_to_bytes(self._encoder.EndEncode())

    def close(self) -> None:
        self._encoder = None


class _CudaArrayView:
    """Exposes only ``__cuda_array_interface__`` over a device tensor.

    Hiding ``__dlpack__`` is deliberate: it forces PyNvVideoCodec down the
    array-interface path, which is the one that accepts a foreign allocation
    without negotiating a stream. The view keeps a reference to the tensor so
    the allocation cannot be freed while the encoder holds the pointer.
    """

    __slots__ = ("_tensor", "__cuda_array_interface__")

    def __init__(self, tensor) -> None:
        self._tensor = tensor
        self.__cuda_array_interface__ = tensor.__cuda_array_interface__

    @classmethod
    def wrap(cls, tensor):
        """Wrap a device tensor; pass anything else through untouched.

        Only a CUDA tensor exposes ``__cuda_array_interface__``. Objects that
        do not (a host tensor, or a test double) are handed to the encoder as
        they are, so the wrapper narrows the encoder input form without
        becoming a second type check.
        """
        if getattr(tensor, "is_cuda", False):
            return cls(tensor)
        return tensor


def _packets_to_bytes(packets) -> list[bytes]:
    """Normalise NVENC's return shape.

    2.2.0 returns a list of dicts with a ``data`` key. Earlier releases
    returned bytearrays directly, and pinning to one of the two would make
    this file version-locked for no benefit.
    """
    out: list[bytes] = []
    for packet in packets or ():
        if isinstance(packet, dict):
            payload = packet.get("data", b"")
        else:
            payload = packet
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            payload = bytes(payload)
        if len(payload):
            out.append(bytes(payload))
    return out


class _FfmpegEncodeBackend:
    """Coverage fallback: NVENC through an ffmpeg subprocess.

    Like the decode fallback it breaks §8.1 -- the NV12 frame is copied to
    host memory and written down a pipe. It earns its place by being the only
    path to a real muxer: ``-f mpegts`` gives genuinely muxed, directly
    chunkable output, which the elementary stream from the primary backend is
    not.
    """

    name = "ffmpeg"

    def __init__(
        self,
        resolution: Resolution,
        fps: Fraction,
        *,
        codec: str = "h264",
        container: str = "mpegts",
        device_id: int = 0,
        bitrate: int | None = None,
        preset: str = "p4",
        ffmpeg_path: str = "ffmpeg",
    ) -> None:
        if shutil.which(ffmpeg_path) is None:
            raise CodecBackendUnavailable(f"{ffmpeg_path} not found on PATH")
        self._proc = subprocess.Popen(
            build_ffmpeg_encode_command(
                resolution,
                fps,
                codec=codec,
                container=container,
                bitrate=bitrate,
                preset=preset,
                ffmpeg_path=ffmpeg_path,
                device_id=device_id,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._chunks: list[bytes] = []
        self._lock = threading.Lock()
        # ffmpeg writes the bitstream while we are still writing frames, so
        # stdout has to be drained concurrently or the pipe buffer deadlocks
        # the pair of us.
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        assert self._proc.stdout is not None
        while True:
            chunk = self._proc.stdout.read(65536)
            if not chunk:
                return
            with self._lock:
                self._chunks.append(chunk)

    def _take(self) -> list[bytes]:
        with self._lock:
            chunks, self._chunks = self._chunks, []
        return chunks

    def encode(self, tensor) -> list[bytes]:
        assert self._proc.stdin is not None
        payload = tensor.detach().to("cpu").contiguous().numpy().tobytes()
        self._proc.stdin.write(payload)
        return self._take()

    def flush(self) -> list[bytes]:
        if self._proc is None:
            return []
        if self._proc.stdin is not None and not self._proc.stdin.closed:
            self._proc.stdin.close()
        self._proc.wait()
        self._reader.join()
        return self._take()

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.kill()
        proc.communicate()


class EncodeStage(StageBase):
    """The chain's sink: device-resident NV12 frames in, HTTP chunks out.

    ``process`` returns ``bytes`` segments rather than frames -- the encode
    stage terminates the frame chain, and §8.4 makes one HTTP chunk per
    encoded segment the response contract, so the segment is the natural unit
    to hand back. ``segment_frames`` is the chunk granularity.
    """

    name = "encode"

    def __init__(
        self,
        resolution: Resolution,
        *,
        fps: Fraction | int = 30,
        codec: str = "h264",
        container: str = "annexb",
        backend: BackendName = "auto",
        device_id: int = 0,
        segment_frames: int = DEFAULT_SEGMENT_FRAMES,
        pool_depth: int = DEFAULT_ENCODER_POOL_DEPTH,
        bitrate: int | None = None,
        preset: str = "P4",
        tuning_info: str = "high_quality",
        extra_options: dict[str, str] | None = None,
        ffmpeg_path: str = "ffmpeg",
    ) -> None:
        if segment_frames < 1:
            raise ValueError("segment_frames must be at least 1")
        self.resolution = resolution
        self.fps = Fraction(fps)
        self.codec = codec.lower()
        self.container = container
        self.requested_backend = backend
        self.device_id = device_id
        self.segment_frames = segment_frames
        self.pool_depth = pool_depth
        self.bitrate = bitrate
        self.preset = preset
        self.tuning_info = tuning_info
        self.extra_options = extra_options
        self.ffmpeg_path = ffmpeg_path

        self.backend_name = select_encode_backend(self.codec, container, backend)
        self._backend: object | None = None
        self._pending: list[bytes] = []
        self._frames_in_segment = 0
        self.frames_encoded = 0
        self.segments_emitted = 0
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def warmup(self) -> None:
        self._open()

    def _open(self):
        if self._backend is not None:
            return self._backend
        if self.backend_name == "pynvvideocodec":
            self._backend = _PyNvEncodeBackend(
                self.resolution,
                codec=self.codec,
                device_id=self.device_id,
                bitrate=self.bitrate,
                preset=self.preset,
                tuning_info=self.tuning_info,
                extra_options=self.extra_options,
            )
        else:
            self._backend = _FfmpegEncodeBackend(
                self.resolution,
                self.fps,
                codec=self.codec,
                container=self.container,
                device_id=self.device_id,
                bitrate=self.bitrate,
                preset=self.preset.lower(),
                ffmpeg_path=self.ffmpeg_path,
            )
        return self._backend

    @property
    def content_type(self) -> str:
        """``Content-Type`` for the chunked response (§8.4)."""
        return _CONTENT_TYPES[(self.container, self.codec)]

    @property
    def pool_bytes(self) -> int:
        """``nvenc_surface_pool_bytes`` for the §6.2 reservation."""
        return codec_pool_bytes(self.resolution, self.pool_depth, PixelFormat.NV12)

    # -- consumption -------------------------------------------------------

    def submit(self, frame: Frame) -> tuple[bytes, ...]:
        """Encode one frame; return any segment it completed."""
        if self._closed:
            raise CodecError("encode stage is closed")
        if frame.end_of_stream:
            return ()
        if frame.format is not PixelFormat.NV12:
            raise ValueError(f"{self.name} takes NV12, got {frame.format.value}")
        if frame.resolution != self.resolution:
            raise ValueError(
                f"encoder configured for {self.resolution}, got {frame.resolution}"
            )
        backend = self._open()
        self._pending.extend(backend.encode(frame.data))
        self.frames_encoded += 1
        self._frames_in_segment += 1
        if self._frames_in_segment >= self.segment_frames:
            return self._cut_segment()
        return ()

    def process(self, frames: Sequence[Frame]) -> tuple[bytes, ...]:
        out: list[bytes] = []
        for frame in frames:
            out.extend(self.submit(frame))
        return tuple(out)

    def _cut_segment(self) -> tuple[bytes, ...]:
        if not self._pending:
            self._frames_in_segment = 0
            return ()
        segment = b"".join(self._pending)
        self._pending.clear()
        self._frames_in_segment = 0
        self.segments_emitted += 1
        return (segment,)

    def flush(self) -> tuple[bytes, ...]:
        """Drain NVENC's internal reorder queue and emit the trailing bytes."""
        if self._backend is not None:
            self._pending.extend(self._backend.flush())
        return self._cut_segment()

    def close(self) -> tuple[bytes, ...]:
        """Flush the encoder, release it, and return the trailing segment.

        The return value is the final HTTP chunk. A caller that discards it
        truncates the output, so the executor must write it. Idempotent:
        a second call returns ``()``.
        """
        if self._closed:
            return ()
        trailing = self.flush()
        self._closed = True
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.close()
        return trailing


# --------------------------------------------------------------------------
# Deterministic synthetic test clip
# --------------------------------------------------------------------------

#: Constant colour of the flat region. Flat in space and in time, so a
#: parity gate can separate "the codec preserved a flat area" from "the codec
#: preserved detail" instead of averaging the two into one number.
FLAT_REGION_RGB = (128, 64, 192)
#: Fraction of the frame width the flat region occupies.
FLAT_REGION_WIDTH_FRACTION = 0.25


def synthetic_frame_rgb(index: int, resolution: Resolution):
    """One deterministic RGB24 frame as a ``(H, W, 3)`` uint8 numpy array.

    Purely procedural -- integer arithmetic on the pixel coordinates and the
    frame index, no RNG anywhere -- because the M2 acceptance gate requires
    byte-stable output across two identical runs and a seeded RNG would still
    tie the content to the numpy version.

    The right three quarters carry high-frequency detail that translates with
    the frame index, which is what gives the SR and RIFE stages something to
    do and what makes a rate-control difference visible. The left quarter is
    flat.
    """
    import numpy as np

    height, width = resolution.height, resolution.width
    y = np.arange(height, dtype=np.int64)[:, None]
    x = np.arange(width, dtype=np.int64)[None, :]
    t = int(index)

    detail = ((x * 7 + y * 11 + t * 5) ^ ((x * x + y * y) // 3)) % 256
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[..., 0] = detail
    frame[..., 1] = (detail * 3 + t * 7) % 256
    frame[..., 2] = (((x + t * 2) // 4) * 37 + y * 5) % 256

    # A moving diagonal bar gives the interpolator coherent large-scale motion
    # on top of the per-pixel noise; without it RIFE has nothing to track.
    bar = np.abs((x + y - t * 9) % (2 * width) - width) < max(4, width // 32)
    frame[np.broadcast_to(bar, (height, width))] = (255, 255, 255)

    flat_width = max(2, int(width * FLAT_REGION_WIDTH_FRACTION))
    frame[:, :flat_width] = FLAT_REGION_RGB
    return frame


def make_test_clip(
    path: str | Path,
    resolution: Resolution,
    frames: int,
    fps: Fraction | int = 30,
    *,
    ffmpeg_path: str = "ffmpeg",
    crf: int = 18,
    preset: str = "medium",
    gop: int = 12,
) -> Path:
    """Write a deterministic synthetic clip, CPU-encoded with libx264.

    The encode is CPU because the clip is chain *input*: producing it on NVENC
    would make the test content depend on the same hardware the test is
    measuring. ``threads=1`` and the bitexact flags are what make two runs
    produce identical files -- x264's frame-level threading is not
    bit-reproducible, and the muxer otherwise writes an encoder string and a
    creation time into the container.
    """
    import numpy as np

    if frames < 1:
        raise ValueError("frames must be positive")
    fps = Fraction(fps)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which(ffmpeg_path) is None:
        raise CodecError(f"{ffmpeg_path} not found on PATH")

    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{resolution.width}x{resolution.height}",
        "-r",
        f"{fps.numerator}/{fps.denominator}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(gop),
        "-x264-params",
        "threads=1:sliced-threads=0:deterministic=1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-map_metadata",
        "-1",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for i in range(frames):
            proc.stdin.write(np.ascontiguousarray(synthetic_frame_rgb(i, resolution)))
        proc.stdin.close()
    except BrokenPipeError:  # pragma: no cover - surfaces as the stderr below
        pass
    finally:
        # communicate() flushes self.stdin unconditionally, which raises on a
        # pipe this function has already closed.
        proc.stdin = None
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise CodecError(
            f"ffmpeg failed to write {path}: {stderr.decode(errors='replace').strip()}"
        )
    return path


@dataclass(frozen=True)
class ClipSpec:
    """A test clip's identity, so a parity gate can name its input."""

    resolution: Resolution
    frames: int
    fps: Fraction
    codec: str = "h264"
    extras: dict[str, str] = field(default_factory=dict)

    def as_manifest(self) -> dict:
        return {
            "resolution": str(self.resolution),
            "frames": self.frames,
            "fps": f"{self.fps.numerator}/{self.fps.denominator}",
            "codec": self.codec,
            **self.extras,
        }
