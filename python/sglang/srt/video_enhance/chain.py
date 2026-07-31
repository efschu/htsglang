# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The enhance chain as a validated graph.

DESIGN #333 §8.1 fixes the stage order:

    decode -> [colour] -> SR -> [resize] -> RIFE -> [colour] -> encode

and fixes two structural properties that this module enforces rather than
documents:

1.  Resize sits *after* SR and *before* RIFE. That is what lets RIFE's engine
    be built at the target size instead of at 8K, and it is why pre-resizing
    before SR is rejected (it throws away the detail SR exists to recover).
2.  Every boundary between decode and encode is a device-to-device hand-off.
    A stage that cannot accept a device tensor is a chain error, not a
    silently host-staged copy.

``build_chain`` resolves each stage's input and output geometry from the
request, so the engine shape triplets, the ring depths and the reservation
all read the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from sglang.srt.video_enhance.frame_math import (
    PixelFormat,
    ReservationBreakdown,
    Resolution,
    chain_reservation,
    dtype_bytes,
)


class StageKind(str, Enum):
    DECODE = "decode"
    COLOR_TO_RGB = "color_to_rgb"
    SR = "sr"
    RESIZE = "resize"
    RIFE = "rife"
    COLOR_TO_YUV = "color_to_yuv"
    ENCODE = "encode"


#: The canonical order. A chain is a subsequence of this list; nothing else is
#: a valid chain.
CANONICAL_ORDER: tuple[StageKind, ...] = (
    StageKind.DECODE,
    StageKind.COLOR_TO_RGB,
    StageKind.SR,
    StageKind.RESIZE,
    StageKind.RIFE,
    StageKind.COLOR_TO_YUV,
    StageKind.ENCODE,
)

#: Stages that may be omitted. Decode, encode and the colour conversions
#: around them are structural; SR, resize and RIFE are what the request asks
#: for.
OPTIONAL_STAGES: frozenset[StageKind] = frozenset(
    {StageKind.SR, StageKind.RESIZE, StageKind.RIFE}
)


class ChainError(ValueError):
    """Raised for a chain that cannot be built, before anything is allocated."""


@dataclass(frozen=True)
class StageSpec:
    """One resolved stage: what it reads, what it writes, at what size."""

    kind: StageKind
    in_res: Resolution
    out_res: Resolution
    in_format: PixelFormat
    out_format: PixelFormat
    #: Frames consumed per invocation (RIFE reads a pair).
    arity_in: int = 1
    #: Frames produced per invocation (RIFE produces multiplier-1 extra frames).
    arity_out: int = 1
    options: dict[str, object] = field(default_factory=dict)

    @property
    def is_device_resident(self) -> bool:
        """Decode output and encode input stay in VRAM; so does everything between."""
        return True

    def describe(self) -> str:
        return (
            f"{self.kind.value:<13} {self.in_res}/{self.in_format.value}"
            f" -> {self.out_res}/{self.out_format.value}"
        )


@dataclass(frozen=True)
class ChainRequest:
    """What a caller asks for. Resolutions are of the *source* and the *target*."""

    source: Resolution
    target: Resolution
    #: Frame-rate multiplier. 1 disables RIFE; 2 inserts one frame per pair.
    fps_multiplier: int = 1
    dtype: str = "fp16"
    #: Stages are request-level configuration, not a fixed pipeline. A 4K
    #: source is typically RIFE-only (``enable_sr=False``), because a x4 SR
    #: model on 4K produces 8K and that is only ever done on request; a small
    #: source typically runs the full chain. A 1x denoise or restoration model
    #: is expressed as ``enable_sr=True`` with ``sr_scale=1``.
    enable_sr: bool = True
    #: SR output scale factor of the selected model. 4 for
    #: realesr-general-wdn-x4v3, 2 for the x2 variants, 1 for a same-size
    #: denoise or restoration model.
    sr_scale: int = 4
    #: Resize is normally derived (it appears when SR does not land on the
    #: target). Setting this False asserts that no resize is wanted, and a
    #: chain that would need one is then a configuration error rather than a
    #: silent insertion.
    enable_resize: bool = True
    #: RIFE optical-flow scale. HolyWu/vs-rife accepts exactly these values.
    rife_scale: float = 1.0
    rife_version: str = "4.6"
    sr_model: str = "realesr-general-wdn-x4v3"
    streams_in_flight: int = 2

    def __post_init__(self) -> None:
        if self.fps_multiplier < 1:
            raise ChainError("fps_multiplier must be >= 1")
        if self.rife_scale not in (0.25, 0.5, 1.0, 2.0, 4.0):
            raise ChainError(
                "rife_scale must be one of 0.25, 0.5, 1.0, 2.0, 4.0 "
                "(HolyWu/vs-rife semantics, mirrored exactly)"
            )
        dtype_bytes(self.dtype)  # raises on an unknown dtype
        if self.streams_in_flight < 1:
            raise ChainError("streams_in_flight must be >= 1")
        if self.sr_scale < 1:
            raise ChainError("sr_scale must be >= 1")
        if (
            self.fps_multiplier == 1
            and not self.enable_sr
            and self.source == self.target
        ):
            raise ChainError(
                "the request asks for no SR, no resize and no interpolation; "
                "there is nothing to enhance"
            )


@dataclass(frozen=True)
class Chain:
    request: ChainRequest
    stages: tuple[StageSpec, ...]

    def stage(self, kind: StageKind) -> StageSpec | None:
        for spec in self.stages:
            if spec.kind is kind:
                return spec
        return None

    @property
    def kinds(self) -> tuple[StageKind, ...]:
        return tuple(s.kind for s in self.stages)

    def describe(self) -> str:
        return "\n".join(s.describe() for s in self.stages)

    def reservation(
        self,
        *,
        engine_device_bytes: int = 0,
        decoder_pool_depth: int = 4,
        encoder_pool_depth: int = 4,
        rife_measured_bytes_per_pair: int | None = None,
    ) -> ReservationBreakdown:
        return chain_reservation(
            source=self.request.source,
            target=self.request.target,
            streams_in_flight=self.request.streams_in_flight,
            dtype=self.request.dtype,
            engine_device_bytes=engine_device_bytes,
            decoder_pool_depth=decoder_pool_depth,
            encoder_pool_depth=encoder_pool_depth,
            rife_measured_bytes_per_pair=rife_measured_bytes_per_pair,
            with_rife=StageKind.RIFE in self.kinds,
            with_sr=StageKind.SR in self.kinds,
            sr_scale=self.request.sr_scale,
            with_resize=StageKind.RESIZE in self.kinds,
        )


def _rgb_format(dtype: str) -> PixelFormat:
    return PixelFormat.RGB_FP16 if dtype_bytes(dtype) == 2 else PixelFormat.RGB_FP32


def build_chain(request: ChainRequest) -> Chain:
    """Resolve a request into an ordered, validated stage list.

    Raises :class:`ChainError` before any device work happens. This is the
    fail-fast point: a chain that cannot be built must not reach the executor.
    """
    rgb = _rgb_format(request.dtype)
    stages: list[StageSpec] = []

    cursor = request.source
    stages.append(
        StageSpec(
            kind=StageKind.DECODE,
            in_res=cursor,
            out_res=cursor,
            in_format=PixelFormat.NV12,
            out_format=PixelFormat.NV12,
        )
    )
    stages.append(
        StageSpec(
            kind=StageKind.COLOR_TO_RGB,
            in_res=cursor,
            out_res=cursor,
            in_format=PixelFormat.NV12,
            out_format=rgb,
        )
    )

    if request.enable_sr:
        sr_out = cursor.scaled(request.sr_scale)
        stages.append(
            StageSpec(
                kind=StageKind.SR,
                in_res=cursor,
                out_res=sr_out,
                in_format=rgb,
                out_format=rgb,
                options={"model": request.sr_model, "scale": request.sr_scale},
            )
        )
        cursor = sr_out

    if cursor != request.target:
        if not request.enable_resize:
            raise ChainError(
                f"resize is disabled but the chain lands on {cursor} while the "
                f"target is {request.target}; enable resize or ask for a target "
                "the selected model produces directly"
            )
        if not request.enable_sr and (
            request.target.width > request.source.width
            or request.target.height > request.source.height
        ):
            raise ChainError(
                "upscaling target requested with SR disabled; a Lanczos-3 resize "
                "cannot recover the detail SR exists to produce. Enable SR or "
                "ask for a target at or below the source resolution."
            )
        stages.append(
            StageSpec(
                kind=StageKind.RESIZE,
                in_res=cursor,
                out_res=request.target,
                in_format=rgb,
                out_format=rgb,
                options={"filter": "lanczos3"},
            )
        )
        cursor = request.target

    if request.fps_multiplier > 1:
        stages.append(
            StageSpec(
                kind=StageKind.RIFE,
                in_res=cursor,
                out_res=cursor,
                in_format=rgb,
                out_format=rgb,
                arity_in=2,
                arity_out=request.fps_multiplier - 1,
                options={
                    "version": request.rife_version,
                    "scale": request.rife_scale,
                    # The engine is built at the post-resize size. This is the
                    # whole reason resize precedes RIFE (§8.1).
                    "trt_max_shape": (cursor.width, cursor.height),
                },
            )
        )

    stages.append(
        StageSpec(
            kind=StageKind.COLOR_TO_YUV,
            in_res=cursor,
            out_res=cursor,
            in_format=rgb,
            out_format=PixelFormat.NV12,
        )
    )
    stages.append(
        StageSpec(
            kind=StageKind.ENCODE,
            in_res=cursor,
            out_res=cursor,
            in_format=PixelFormat.NV12,
            out_format=PixelFormat.NV12,
        )
    )

    chain = Chain(request=request, stages=tuple(stages))
    validate_chain(chain)
    return chain


def validate_chain(chain: Chain) -> None:
    """Structural invariants. Cheap, and run on every build."""
    kinds = chain.kinds
    if not kinds:
        raise ChainError("empty chain")
    if kinds[0] is not StageKind.DECODE or kinds[-1] is not StageKind.ENCODE:
        raise ChainError("a chain must start at decode and end at encode")

    order = {kind: i for i, kind in enumerate(CANONICAL_ORDER)}
    positions = [order[k] for k in kinds]
    if positions != sorted(positions):
        raise ChainError(
            f"stage order violates the canonical order: {[k.value for k in kinds]}"
        )
    if len(set(kinds)) != len(kinds):
        raise ChainError("a stage may appear at most once in a chain")

    for missing in set(CANONICAL_ORDER) - set(kinds):
        if missing not in OPTIONAL_STAGES:
            raise ChainError(f"structural stage {missing.value} is missing")

    if StageKind.RESIZE in kinds and StageKind.RIFE in kinds:
        if kinds.index(StageKind.RESIZE) > kinds.index(StageKind.RIFE):
            raise ChainError(
                "resize must precede RIFE so the RIFE engine is built at the "
                "target size rather than at the x4 SR output size"
            )

    for left, right in zip(chain.stages, chain.stages[1:]):
        if left.out_res != right.in_res:
            raise ChainError(
                f"geometry break between {left.kind.value} and {right.kind.value}: "
                f"{left.out_res} != {right.in_res}"
            )
        if left.out_format != right.in_format:
            raise ChainError(
                f"format break between {left.kind.value} and {right.kind.value}: "
                f"{left.out_format.value} != {right.in_format.value}"
            )
