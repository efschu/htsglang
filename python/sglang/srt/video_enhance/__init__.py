# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Class-3 video-enhance stream server (DESIGN #333 §8, build stage M2).

A process-isolated utility tenant that decodes into VRAM, runs
super-resolution, Lanczos-3 resize and frame interpolation with
GPU-persistent intermediates, encodes with NVENC, and streams the result over
chunked HTTP with real back-pressure.

Nothing here imports ``sglang.srt`` scheduler internals. The tenant is its own
process with its own CUDA context, pinned to one physical GPU through
``CUDA_VISIBLE_DEVICES``; that process boundary is what makes its residency
ladder and its memory accounting tractable without allocator changes.

Import layering: ``frame_math``, ``chain``, ``ring``, ``engine_cache``,
``shard_plan`` and ``reservation`` are pure Python and import on a CPU-only
host with no torch. Everything that touches a device is imported lazily by
the module that needs it.
"""

from sglang.srt.video_enhance.chain import (
    Chain,
    ChainError,
    ChainRequest,
    StageKind,
    build_chain,
)
from sglang.srt.video_enhance.frame_math import (
    GIB,
    MIB,
    PixelFormat,
    ReservationBreakdown,
    Resolution,
    chain_reservation,
    frame_bytes,
    max_in_flight_for_budget,
)
from sglang.srt.video_enhance.frames import Frame, HostResidencyError, Stage, StageBase
from sglang.srt.video_enhance.ring import BoundedRing, OverloadPolicy, RingSet

__all__ = [
    "BoundedRing",
    "Chain",
    "ChainError",
    "ChainRequest",
    "Frame",
    "GIB",
    "HostResidencyError",
    "MIB",
    "OverloadPolicy",
    "PixelFormat",
    "ReservationBreakdown",
    "Resolution",
    "RingSet",
    "Stage",
    "StageBase",
    "StageKind",
    "build_chain",
    "chain_reservation",
    "frame_bytes",
    "max_in_flight_for_budget",
]
