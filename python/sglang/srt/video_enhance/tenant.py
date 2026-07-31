# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Assembling a Class-3 executor tenant: config, budget, stages.

A Class-3 tenant is a process that owns TensorRT engines and codec contexts
and exposes work-unit submission -- not an ``srt`` scheduler and not coupled
to one. Nothing in this package imports ``srt`` internals, which is what lets
the tenant run as its own process with its own CUDA context and its own
``CUDA_VISIBLE_DEVICES``.

The budget is the important part. Under M2 the tenant runs ahead of the
registry (§10), so its reservation is a static configured number rather than
a registry slot. The number is not free-floating: ``max_in_flight`` is
*derived* from it through the §8.3 arithmetic, so a depth that does not fit
cannot be configured (§8.4 rule 5), and the tenant refuses to start when the
configured budget cannot hold even one frame in flight.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from sglang.srt.video_enhance.chain import Chain, ChainRequest, StageKind, build_chain
from sglang.srt.video_enhance.engine_cache import EngineCache
from sglang.srt.video_enhance.frame_math import (
    MIB,
    ReservationBreakdown,
    Resolution,
    max_in_flight_for_budget,
)
from sglang.srt.video_enhance.ring import OverloadPolicy


class TenantConfigError(ValueError):
    """A configuration that cannot fit, detected before anything is allocated."""


@dataclass
class TenantConfig:
    """Static configuration of one Class-3 video-enhance tenant."""

    #: Absolute device budget in MiB. Same semantics as ``--rank-gpu-memory-mib``
    #: for Class 1: the whole budget, no implicit ceiling, no safety factor, no
    #: rounding down. Leaving headroom is the operator's responsibility.
    budget_mib: int
    #: Physical GPU this tenant owns, as an NVML UUID. The process is pinned to
    #: exactly this card via CUDA_VISIBLE_DEVICES, so inside the process
    #: ``cuda:0`` is unambiguous.
    card_uuid: str | None = None
    engine_cache_dir: Path = Path("/spinning/llm_stuff/k3-models/engines")
    model_dir: Path = Path("/spinning/llm_stuff/k3-models")
    precision: str = "fp16"
    provider: str = "tensorrt"
    decoder_pool_depth: int = 4
    encoder_pool_depth: int = 4
    #: Measured RIFE bytes per frame pair (measurement post P4). None means the
    #: value has not been measured and a RIFE chain cannot be budgeted.
    rife_measured_bytes_per_pair: int | None = None
    default_policy: OverloadPolicy = OverloadPolicy.STALL
    tenant_id: str = "k3-video-enhance"
    extra: dict = field(default_factory=dict)

    @property
    def budget_bytes(self) -> int:
        return self.budget_mib * MIB


@dataclass(frozen=True)
class PlannedJob:
    """A request that has passed the fixed-cost check and may be started."""

    chain: Chain
    reservation: ReservationBreakdown
    max_in_flight: int
    ring_depth: int


def plan_job(config: TenantConfig, request: ChainRequest) -> PlannedJob:
    """Resolve a request into a runnable plan, or refuse it with the arithmetic.

    This is the fixed-cost calculation that precedes any GPU work. It runs in
    microseconds and needs no device, which is the whole point: "does this
    configuration fit on this card" must be answerable without booking a GPU
    window.
    """
    chain = build_chain(request)
    with_rife = StageKind.RIFE in chain.kinds
    if with_rife and config.rife_measured_bytes_per_pair is None:
        raise TenantConfigError(
            "a RIFE chain cannot be budgeted without a measured per-frame-pair "
            "footprint (measurement post P4). Run the P4 probe and set "
            "rife_measured_bytes_per_pair, or request fps_multiplier=1."
        )

    fits = max_in_flight_for_budget(
        source=request.source,
        target=request.target,
        budget_bytes=config.budget_bytes,
        dtype=request.dtype,
        decoder_pool_depth=config.decoder_pool_depth,
        encoder_pool_depth=config.encoder_pool_depth,
        rife_measured_bytes_per_pair=config.rife_measured_bytes_per_pair,
        with_rife=with_rife,
    )
    if fits < 1:
        one = chain_reservation_for(config, request, streams=1)
        raise TenantConfigError(
            f"budget of {config.budget_mib} MiB cannot hold a single frame in "
            f"flight for {request.source} -> {request.target}. One in-flight "
            f"frame needs {one.total_mib} MiB:\n{one.render()}"
        )

    depth = min(request.streams_in_flight, fits)
    reservation = chain_reservation_for(config, request, streams=depth)
    return PlannedJob(
        chain=chain,
        reservation=reservation,
        max_in_flight=depth,
        ring_depth=max(1, depth),
    )


def chain_reservation_for(
    config: TenantConfig, request: ChainRequest, streams: int
) -> ReservationBreakdown:
    chain = build_chain(replace(request, streams_in_flight=streams))
    return chain.reservation(
        decoder_pool_depth=config.decoder_pool_depth,
        encoder_pool_depth=config.encoder_pool_depth,
        rife_measured_bytes_per_pair=config.rife_measured_bytes_per_pair,
    )


def pin_process_to_card(index: int) -> None:
    """Isolate this process to exactly one physical GPU.

    Process-level isolation rather than an in-process logical-to-physical
    mapping table: two tenants sharing a card are two processes with the same
    ``CUDA_VISIBLE_DEVICES``, and inside each ``cuda:0`` is unambiguous. This
    must run before torch initialises CUDA.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)


def build_stages(
    config: TenantConfig,
    chain: Chain,
    *,
    device_id: int = 0,
) -> dict[StageKind, object]:
    """Instantiate the concrete stage objects for a chain.

    Imports are local because the codec and RIFE stages pull optional heavy
    dependencies; a host that only plans (the CPU-side feasibility check, the
    planner, the tests) must not need them.
    """
    from sglang.srt.video_enhance import codec, rife
    from sglang.srt.video_enhance.resize import ResizeStage
    from sglang.srt.video_enhance.sr import SuperResolutionStage

    cache = EngineCache(config.engine_cache_dir)
    stages: dict[StageKind, object] = {}
    for spec in chain.stages:
        if spec.kind is StageKind.DECODE:
            stages[spec.kind] = codec.DecodeStage(
                resolution=spec.in_res, device_id=device_id
            )
        elif spec.kind is StageKind.COLOR_TO_RGB:
            stages[spec.kind] = codec.ColorToRgbStage(dtype=config.precision)
        elif spec.kind is StageKind.SR:
            stages[spec.kind] = SuperResolutionStage.build(
                source=spec.in_res,
                fmt=spec.in_format,
                model_dir=Path(config.model_dir) / "sr",
                provider=config.provider,
                precision=config.precision,
                device_id=device_id,
                cache=cache,
            )
        elif spec.kind is StageKind.RESIZE:
            stages[spec.kind] = ResizeStage(spec.in_res, spec.out_res, spec.in_format)
        elif spec.kind is StageKind.RIFE:
            stages[spec.kind] = rife.RifeStage(
                version=str(spec.options["version"]),
                scale=float(spec.options["scale"]),  # type: ignore[arg-type]
                resolution=spec.in_res,
                multiplier=spec.arity_out + 1,
                dtype=config.precision,
                device=f"cuda:{device_id}",
            )
        elif spec.kind is StageKind.COLOR_TO_YUV:
            stages[spec.kind] = codec.ColorToYuvStage(dtype=config.precision)
        elif spec.kind is StageKind.ENCODE:
            stages[spec.kind] = codec.EncodeStage(
                resolution=spec.in_res, device_id=device_id
            )
    return stages


def default_target_for(source: Resolution) -> Resolution:
    """A x4 SR output resized back to a sane target.

    1080p source leaves SR at 7680x4320. The first build-out targets 4K, which
    is a halving of the SR output rather than an arbitrary number.
    """
    return Resolution(source.width * 2, source.height * 2)
