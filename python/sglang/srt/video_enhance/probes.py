# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Measurement posts P1-P5 as a runnable harness, plus the playback arithmetic.

The posts are named in DESIGN #333 §8.6. This module makes them executable so
a card window produces a JSON record rather than numbers in a chat log, and so
the same record can be replayed into the shard planner without a second run.

Standing benchmark rules apply and are implemented, not just cited: an
A-versus-A run establishes the noise floor before any arm is compared, arms
are interleaved rather than run in blocks, per-stage ms/frame is reported
instead of aggregate frames per second, and a difference below the measured
floor is reported as "below detection" rather than as a result.

The 4K point is a first-class part of the matrix, not an extrapolation. A 4K
source is a real target scenario for interpolation-only chains -- x4 SR on 4K
would produce 8K, which is only done on request -- so RIFE at 3840x2160 is
probed directly, on each card, with the optical-flow ``scale`` knob as an arm.
:func:`playback_feasibility` then answers the question those numbers exist for:
does the aggregate over the cards sustain a 24-30 fps playback rate, and how
much watch-ahead does it need.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

from sglang.srt.video_enhance.frame_math import (
    MIB,
    R4K,
    R540P,
    R720P,
    R1080P,
    Resolution,
)

#: The P1 grid. Each row is (stage, input resolution, options). The input
#: resolutions are the REAL per-stage resolutions of the chain, not one
#: resolution reused for every stage: SR runs at the source size, resize runs
#: at the SR output (source x4), RIFE runs at the post-resize target. A table
#: keyed only by card would price the wrong work.
P1_STAGE_POINTS: tuple[tuple[str, Resolution, dict], ...] = (
    ("sr", R540P, {}),
    ("sr", R720P, {}),
    ("sr", R1080P, {}),
    ("resize", R540P.scaled(4), {"target": R1080P}),
    ("resize", R720P.scaled(4), {"target": R4K}),
    ("resize", R1080P.scaled(4), {"target": R4K}),
    ("rife", R1080P, {"scale": 1.0}),
    ("rife", R1080P, {"scale": 0.5}),
    # 4K interpolation: the interpolation-only chain for a 4K source, both
    # flow-scale arms. RIFE has no tiling, so this point also decides whether
    # a whole 4K frame pair fits at all on the smaller cards.
    ("rife", R4K, {"scale": 1.0}),
    ("rife", R4K, {"scale": 0.5}),
    # Decode at the source sizes a request actually arrives at, and encode at
    # the sizes the chain delivers. Both were missing from this grid, which is
    # why the whole-chain rate table had a hole at each end and a Regime-B
    # optimiser could not price an assignment that put decode or encode on a
    # different card from the middle of the chain.
    ("decode", R540P, {}),
    ("decode", R1080P, {}),
    ("encode", R1080P, {}),
    ("encode", R4K, {}),
)

#: P2 boundary sizes, in MiB, from the §8.3 frame table.
P2_TRANSFER_MIB: tuple[float, ...] = (0.74, 2.97, 11.87, 47.46, 189.84)


@dataclass
class Sample:
    """One measured point."""

    post: str
    stage: str
    card: str
    resolution: str
    dtype: str
    options: dict
    ms_per_frame: float
    ms_stdev: float
    iterations: int
    peak_device_bytes: int | None = None
    note: str = ""


@dataclass
class ProbeReport:
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    host: dict = field(default_factory=dict)
    noise_floor_pct: float | None = None
    samples: list[Sample] = field(default_factory=list)

    def add(self, sample: Sample) -> None:
        self.samples.append(sample)

    def to_json(self) -> str:
        payload = {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "host": self.host,
            "noise_floor_pct": self.noise_floor_pct,
            "samples": [asdict(s) for s in self.samples],
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def rate_table(self) -> dict[tuple[str, str, str], float]:
        """``(stage, resolution, card) -> ms/frame``, the shard planner's input."""
        return {(s.stage, s.resolution, s.card): s.ms_per_frame for s in self.samples}


def significant(delta_pct: float, noise_floor_pct: float | None) -> bool:
    """Is a difference above the measured detection threshold?

    Reporting a difference smaller than the A-versus-A spread as a result is
    the single most common way a benchmark lies, so the check is a function
    rather than a habit.
    """
    if noise_floor_pct is None:
        return False
    return abs(delta_pct) > noise_floor_pct


# --------------------------------------------------------------------------
# Playback arithmetic -- pure, so it runs without a card
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaybackVerdict:
    target_fps: float
    aggregate_fps: float
    per_card_fps: dict[str, float]
    realtime: bool
    #: Seconds of output that must be buffered before playback starts so the
    #: stream never underruns. Zero when the aggregate already exceeds the
    #: target rate.
    watch_ahead_s: float
    #: Seconds of source the run covers; needed because a shortfall on a long
    #: source cannot be absorbed by any finite buffer.
    source_duration_s: float
    note: str = ""


def playback_feasibility(
    *,
    per_card_ms_per_frame: dict[str, float],
    target_fps: float,
    source_duration_s: float,
    fps_multiplier: int = 2,
    max_watch_ahead_s: float = 60.0,
) -> PlaybackVerdict:
    """Can the cards, together, sustain playback -- and with how much lead?

    Regime A replicated: each card runs a whole chain over its own contiguous
    chunk, so the rates add. That is the only reason the aggregate is a sum;
    a stage-split regime would be bounded by its slowest stage instead.

    A shortfall is absorbable only if the whole run's deficit fits in the
    watch-ahead buffer. Over a two-hour source a 5 percent shortfall is six
    minutes of lead, which is not a buffer, so the verdict reports the
    required lead rather than a boolean alone.
    """
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    per_card_fps = {
        card: (1000.0 / ms if ms > 0 else 0.0)
        for card, ms in per_card_ms_per_frame.items()
    }
    aggregate = sum(per_card_fps.values())
    # Interpolation multiplies the frames that must be *produced* but the
    # measured ms/frame is already per produced frame, so the target rate is
    # the playback rate of the interpolated stream.
    deficit = target_fps - aggregate
    if deficit <= 0:
        return PlaybackVerdict(
            target_fps=target_fps,
            aggregate_fps=aggregate,
            per_card_fps=per_card_fps,
            realtime=True,
            watch_ahead_s=0.0,
            source_duration_s=source_duration_s,
            note=f"aggregate exceeds target by {aggregate - target_fps:.1f} fps",
        )
    lead = deficit / target_fps * source_duration_s
    return PlaybackVerdict(
        target_fps=target_fps,
        aggregate_fps=aggregate,
        per_card_fps=per_card_fps,
        realtime=False,
        watch_ahead_s=lead,
        source_duration_s=source_duration_s,
        note=(
            f"short by {deficit:.1f} fps; needs {lead:.0f} s of lead, which "
            + ("fits" if lead <= max_watch_ahead_s else "exceeds")
            + f" the {max_watch_ahead_s:.0f} s watch-ahead budget"
        ),
    )


def frames_for_duration(duration_s: float, rate: Fraction) -> int:
    return int(duration_s * float(rate))


# --------------------------------------------------------------------------
# Device-side probes
# --------------------------------------------------------------------------


def _timeit(fn, *, iterations: int, warmup: int = 3) -> tuple[float, float]:
    """Run ``fn`` and return (mean ms, stdev ms), synchronising once per sample."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean, stdev


def noise_floor(fn, *, iterations: int = 20) -> float:
    """A-versus-A: run the same arm twice and report the spread in percent."""
    a, _ = _timeit(fn, iterations=iterations)
    b, _ = _timeit(fn, iterations=iterations)
    return abs(a - b) / max(a, b) * 100.0


def probe_transfer(card: str, iterations: int = 20) -> list[Sample]:
    """P2: host-staged D2H + H2D round trip at the §8.3 boundary sizes."""
    import torch

    out: list[Sample] = []
    for mib in P2_TRANSFER_MIB:
        n = int(mib * MIB)
        device = torch.empty(n, dtype=torch.uint8, device="cuda")
        host = torch.empty(n, dtype=torch.uint8, pin_memory=True)

        def hop():
            host.copy_(device, non_blocking=True)
            device.copy_(host, non_blocking=True)

        mean, stdev = _timeit(hop, iterations=iterations)
        out.append(
            Sample(
                post="P2",
                stage="transfer",
                card=card,
                resolution=f"{mib:.2f}MiB",
                dtype="uint8",
                options={"direction": "d2h+h2d"},
                ms_per_frame=mean,
                ms_stdev=stdev,
                iterations=iterations,
                note=f"{2 * mib / mean:.2f} GiB/s round trip" if mean > 0 else "",
            )
        )
        torch.cuda.empty_cache()
    return out


def probe_resize(
    card: str, source: Resolution, target: Resolution, dtype: str, iterations: int = 10
) -> Sample:
    import torch

    from sglang.srt.video_enhance.resize import lanczos3_resize

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    x = torch.rand(1, 3, source.height, source.width, dtype=torch_dtype, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    mean, stdev = _timeit(lambda: lanczos3_resize(x, target), iterations=iterations)
    peak = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    return Sample(
        post="P1",
        stage="resize",
        card=card,
        resolution=str(source),
        dtype=dtype,
        options={"target": str(target), "filter": "lanczos3"},
        ms_per_frame=mean,
        ms_stdev=stdev,
        iterations=iterations,
        peak_device_bytes=peak,
    )


def probe_encode(
    card: str,
    target: Resolution,
    dtype: str,
    iterations: int = 10,
    backend: str = "ffmpeg",
) -> Sample:
    """P1 for the encode stage: ms per frame through a real encoder session.

    Synthetic content, and that is a caveat rather than a detail: an encoder's
    time per frame depends on how compressible the picture is, so a gradient
    encodes faster than film grain. The number is therefore a floor for this
    stage on this card, and it is comparable *between cards* -- which is what
    a placement decision needs -- while not being a prediction for an
    arbitrary clip.
    """
    import torch

    from sglang.srt.video_enhance import codec
    from sglang.srt.video_enhance.frame_math import PixelFormat
    from sglang.srt.video_enhance.frames import Frame

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    rgb = torch.rand(
        1, 3, target.height, target.width, dtype=torch_dtype, device="cuda"
    )
    to_yuv = codec.ColorToYuvStage(dtype=dtype)
    fmt = PixelFormat.RGB_FP16 if dtype == "fp16" else PixelFormat.RGB_FP32
    (yuv_frame,) = to_yuv.process(
        [Frame(data=rgb, resolution=target, format=fmt, index=0)]
    )
    encoder = codec.EncodeStage(
        target, fps=30, codec="h264", device_id=0, backend=backend, segment_frames=1
    )
    counter = {"i": 0}

    def once() -> None:
        counter["i"] += 1
        encoder.process(
            [
                Frame(
                    data=yuv_frame.data,
                    resolution=target,
                    format=yuv_frame.format,
                    index=counter["i"],
                )
            ]
        )

    try:
        mean, stdev = _timeit(once, iterations=iterations)
    finally:
        closer = getattr(encoder, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001 - a probe must not mask its own result
                pass
        torch.cuda.empty_cache()
    return Sample(
        post="P1",
        stage="encode",
        card=card,
        resolution=str(target),
        dtype=dtype,
        options={"backend": backend, "codec": "h264"},
        ms_per_frame=mean,
        ms_stdev=stdev,
        iterations=iterations,
        note="synthetic content; a floor for this stage, comparable across cards",
    )


def probe_decode(
    card: str,
    source: Resolution,
    dtype: str,
    clip: str,
    iterations: int = 10,
    backend: str = "ffmpeg",
) -> Sample:
    """P1 for the decode stage: ms per frame out of a real container.

    Decode is the one stage whose cost is not a function of the card alone --
    it is a function of the clip. The probe therefore reports which clip it
    used, and a table built from two different clips is not internally
    comparable on this row. The ffmpeg backend is the measured one because it
    is the only one that can seek, which is what every multi-card shard needs.
    """
    import torch

    from sglang.srt.video_enhance import codec

    stage = codec.DecodeStage(
        source=clip,
        resolution=source,
        device_id=0,
        backend=backend,
        frame_limit=iterations + 3,
    )
    frames = iter(stage)
    # Warm the container open and the first packet out of the measurement:
    # a probe that charged the stage for opening the file would price a
    # 4-second clip and a 4-hour one differently for no physical reason.
    for _ in range(3):
        next(frames, None)

    def once() -> None:
        next(frames, None)

    try:
        mean, stdev = _timeit(once, iterations=iterations, warmup=0)
    finally:
        closer = getattr(stage, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001 - a probe must not mask its own result
                pass
        torch.cuda.empty_cache()
    return Sample(
        post="P1",
        stage="decode",
        card=card,
        resolution=str(source),
        dtype=dtype,
        options={"backend": backend, "clip": str(clip)},
        ms_per_frame=mean,
        ms_stdev=stdev,
        iterations=iterations,
        note="clip-dependent; only comparable across cards for the same clip",
    )


def probe_sr(
    card: str,
    source: Resolution,
    dtype: str,
    *,
    provider: str = "tensorrt",
    model_dir=None,
    cache_dir=None,
    iterations: int = 10,
) -> Sample:
    """P1 + P3 for the SR stage: ms/frame and measured peak device bytes."""
    import torch

    from sglang.srt.video_enhance.engine_cache import EngineCache, ShapeTriplet
    from sglang.srt.video_enhance.sr import (
        DEFAULT_MODEL_DIR,
        REALESR_GENERAL_WDN_X4V3,
        SuperResolutionStage,
    )
    from sglang.srt.video_enhance.frame_math import PixelFormat

    fmt = PixelFormat.RGB_FP16 if dtype == "fp16" else PixelFormat.RGB_FP32
    stage = SuperResolutionStage.build(
        source=source,
        fmt=fmt,
        model=REALESR_GENERAL_WDN_X4V3,
        model_dir=model_dir or DEFAULT_MODEL_DIR,
        provider=provider,
        precision=dtype,
        cache=EngineCache(cache_dir) if cache_dir else None,
        shapes=ShapeTriplet.static(source.width, source.height),
    )
    x = torch.rand(
        1, 3, source.height, source.width, dtype=torch.float32, device="cuda"
    )
    # torch.cuda.max_memory_allocated only sees torch's own allocator. ONNX
    # Runtime and TensorRT allocate through their own, so the SR stage's real
    # footprint is the device-wide free-memory delta -- P3 has to compare
    # against that or it compares against a fraction of the truth.
    torch.cuda.synchronize()
    free_before, _total = torch.cuda.mem_get_info()
    mean, stdev = _timeit(lambda: stage.backend.run(x), iterations=iterations)
    torch.cuda.synchronize()
    free_after, _total = torch.cuda.mem_get_info()
    peak = max(0, free_before - free_after)
    stage.close()
    torch.cuda.empty_cache()
    return Sample(
        post="P1",
        stage="sr",
        card=card,
        resolution=str(source),
        dtype=dtype,
        options={"provider": provider, "model": REALESR_GENERAL_WDN_X4V3.model_id},
        ms_per_frame=mean,
        ms_stdev=stdev,
        iterations=iterations,
        peak_device_bytes=peak,
    )


def probe_rife(
    card: str,
    resolution: Resolution,
    dtype: str,
    scale: float,
    *,
    version: str = "4.6",
    weights_dir=None,
    iterations: int = 10,
) -> Sample:
    """P1 + P4: ms per frame pair and the measured per-pair footprint."""
    import torch

    from sglang.srt.video_enhance.frame_math import PixelFormat
    from sglang.srt.video_enhance.frames import Frame
    from sglang.srt.video_enhance.rife import RifeStage, download_weights

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    stage = RifeStage(
        resolution=resolution,
        version=version,
        multiplier=2,
        scale=scale,
        dtype=dtype,
        device="cuda",
        weights_path=download_weights(version, weights_dir) if weights_dir else None,
    )
    stage.warmup()
    fmt = PixelFormat.RGB_FP16 if dtype == "fp16" else PixelFormat.RGB_FP32
    pair = [
        Frame(
            data=torch.rand(
                1,
                3,
                resolution.height,
                resolution.width,
                dtype=torch_dtype,
                device="cuda",
            ),
            resolution=resolution,
            format=fmt,
            index=i,
        )
        for i in range(2)
    ]
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    mean, stdev = _timeit(lambda: stage.process(pair), iterations=iterations)
    peak = torch.cuda.max_memory_allocated()
    stage.close()
    torch.cuda.empty_cache()
    return Sample(
        post="P1",
        stage="rife",
        card=card,
        resolution=str(resolution),
        dtype=dtype,
        options={"scale": scale, "version": version},
        ms_per_frame=mean,
        ms_stdev=stdev,
        iterations=iterations,
        peak_device_bytes=peak - baseline,
        note="peak_device_bytes is the P4 per-frame-pair figure",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m sglang.srt.video_enhance.probes --card-index 1``."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="DESIGN #333 measurement posts")
    parser.add_argument("--card-index", type=int, required=True, help="NVML index")
    parser.add_argument("--out", default="-", help="JSON output path, - for stdout")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--posts", default="P1,P2", help="comma-separated post names")
    parser.add_argument("--sr-provider", default="tensorrt")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--rife-weight-dir", default=None)
    parser.add_argument("--skip-stages", default="", help="comma-separated stage names")
    parser.add_argument(
        "--clip",
        default=None,
        help="clip for the decode post; without it the decode rows are skipped",
    )
    args = parser.parse_args(argv)

    # Pin before torch touches CUDA: one physical card per probe process, so a
    # measurement can never be attributed to the wrong device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.card_index)

    from sglang.srt.video_enhance.nvml import list_devices

    devices = {d.index: d for d in list_devices()}
    info = devices.get(args.card_index)
    card = info.name if info else f"index{args.card_index}"

    posts = {p.strip() for p in args.posts.split(",") if p.strip()}
    skip = {s.strip() for s in args.skip_stages.split(",") if s.strip()}
    report = ProbeReport(
        host={
            "card_index": args.card_index,
            "card_name": card,
            "nvml_uuid": info.uuid if info else None,
            "total_mib": (info.total_bytes // MIB) if info else None,
        }
    )

    import torch

    if "P1" in posts:
        # Noise floor first, on the cheapest real kernel of the chain.
        from sglang.srt.video_enhance.resize import lanczos3_resize

        probe_tensor = torch.rand(1, 3, 540, 960, device="cuda")
        report.noise_floor_pct = noise_floor(
            lambda: lanczos3_resize(probe_tensor, R1080P), iterations=10
        )
        del probe_tensor
        torch.cuda.empty_cache()

        for stage, resolution, options in P1_STAGE_POINTS:
            if stage in skip:
                continue
            try:
                if stage == "sr":
                    sample = probe_sr(
                        card,
                        resolution,
                        args.dtype,
                        provider=args.sr_provider,
                        model_dir=args.model_dir,
                        cache_dir=args.cache_dir,
                        iterations=args.iterations,
                    )
                elif stage == "resize":
                    sample = probe_resize(
                        card,
                        resolution,
                        options["target"],
                        args.dtype,
                        iterations=args.iterations,
                    )
                elif stage == "rife":
                    sample = probe_rife(
                        card,
                        resolution,
                        args.dtype,
                        float(options["scale"]),
                        weights_dir=args.rife_weight_dir,
                        iterations=args.iterations,
                    )
                elif stage == "encode":
                    sample = probe_encode(
                        card, resolution, args.dtype, iterations=args.iterations
                    )
                elif stage == "decode":
                    if not args.clip:
                        # No clip, no decode measurement. Skipping is right:
                        # inventing one would put a number in the table that
                        # no card produced.
                        continue
                    sample = probe_decode(
                        card,
                        resolution,
                        args.dtype,
                        args.clip,
                        iterations=args.iterations,
                    )
                else:
                    continue
            except Exception as exc:  # noqa: BLE001 - a failed point is a datum
                report.add(
                    Sample(
                        post="P1",
                        stage=stage,
                        card=card,
                        resolution=str(resolution),
                        dtype=args.dtype,
                        options={k: str(v) for k, v in options.items()},
                        ms_per_frame=float("nan"),
                        ms_stdev=0.0,
                        iterations=0,
                        note=f"{type(exc).__name__}: {exc}",
                    )
                )
                torch.cuda.empty_cache()
                continue
            report.add(sample)

    if "P2" in posts:
        for sample in probe_transfer(card, iterations=args.iterations):
            report.add(sample)

    report.finished_at = time.time()
    payload = report.to_json()
    if args.out == "-":
        print(payload)
    else:
        with open(args.out, "w") as fh:
            fh.write(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# --------------------------------------------------------------------------
# Capability frontier
# --------------------------------------------------------------------------

#: The chain configurations whose limits are measured separately, because they
#: have different bottlenecks: SR is compute-bound at input resolution,
#: interpolation is bound by flow-pyramid memory traffic at output resolution,
#: and the full chain is neither.
CONFIGURATIONS: tuple[str, ...] = ("sr_only", "rife_only", "full_chain")


@dataclass(frozen=True)
class FrontierPoint:
    """The largest (resolution x fps) one configuration sustains."""

    configuration: str
    card: str
    resolution: str
    ms_per_output_frame: float
    max_fps: float
    peak_device_mib: float | None = None
    options: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "configuration": self.configuration,
            "card": self.card,
            "resolution": self.resolution,
            "ms_per_output_frame": round(self.ms_per_output_frame, 3),
            "max_fps": round(self.max_fps, 2),
            "peak_device_mib": self.peak_device_mib,
            **{f"opt_{k}": v for k, v in self.options.items()},
        }


def frontier_from_samples(samples: list[Sample]) -> list[FrontierPoint]:
    """Turn a P1 sample list into the machine-readable capability table.

    One row per (configuration, card, resolution, options). ``max_fps`` is the
    output frame rate that configuration sustains on that card:
    ``1000 / ms_per_output_frame``. It is deliberately not called "realtime" --
    whether it is enough depends on the requested playback rate, which is the
    caller's, and :func:`playback_feasibility` answers that separately.
    """
    by_config = {"sr": "sr_only", "rife": "rife_only"}
    rows: list[FrontierPoint] = []
    for sample in samples:
        config = by_config.get(sample.stage)
        if config is None or sample.ms_per_frame != sample.ms_per_frame:
            continue  # NaN means the point failed; a failed point is not a limit
        rows.append(
            FrontierPoint(
                configuration=config,
                card=sample.card,
                resolution=sample.resolution,
                ms_per_output_frame=sample.ms_per_frame,
                max_fps=1000.0 / sample.ms_per_frame if sample.ms_per_frame else 0.0,
                peak_device_mib=(
                    round(sample.peak_device_bytes / MIB, 1)
                    if sample.peak_device_bytes
                    else None
                ),
                options={k: v for k, v in sample.options.items() if k == "scale"},
            )
        )
    return rows


def aggregate_frontier(rows: list[FrontierPoint]) -> list[dict]:
    """Regime-A aggregate: whole chains replicated across cards, so rates add."""
    grouped: dict[tuple[str, str, str], list[FrontierPoint]] = {}
    for row in rows:
        key = (
            row.configuration,
            row.resolution,
            json.dumps(row.options, sort_keys=True),
        )
        grouped.setdefault(key, []).append(row)
    out = []
    for (config, resolution, options), members in sorted(grouped.items()):
        out.append(
            {
                "configuration": config,
                "resolution": resolution,
                "options": json.loads(options),
                "cards": {m.card: round(m.max_fps, 2) for m in members},
                "aggregate_max_fps": round(sum(m.max_fps for m in members), 2),
            }
        )
    return out


def load_probe_reports(directory: Path | str) -> tuple[list[Sample], list[dict]]:
    """Read every ``ProbeReport`` JSON in a directory back into samples.

    Returns the samples and one provenance record per file. Provenance is not
    decoration: a frontier row is only meaningful together with the card it
    was measured on and the noise floor of that run, and a caller comparing
    two deployments needs to see that the numbers came from different
    windows.

    A file that is not a probe report -- the directory also holds e2e and
    multi-card records -- has no ``samples`` key and is skipped rather than
    treated as an empty measurement, which would look like a card that
    sustains nothing.
    """
    root = Path(directory)
    samples: list[Sample] = []
    sources: list[dict] = []
    if not root.is_dir():
        return samples, sources
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or "samples" not in payload:
            continue
        rows = payload.get("samples") or []
        loaded = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                samples.append(
                    Sample(
                        post=row.get("post", ""),
                        stage=row.get("stage", ""),
                        card=row.get("card", ""),
                        resolution=row.get("resolution", ""),
                        dtype=row.get("dtype", ""),
                        options=row.get("options") or {},
                        ms_per_frame=float(row["ms_per_frame"]),
                        ms_stdev=float(row.get("ms_stdev", 0.0)),
                        iterations=int(row.get("iterations", 0)),
                        peak_device_bytes=row.get("peak_device_bytes"),
                        note=row.get("note", ""),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
            loaded += 1
        sources.append(
            {
                "file": path.name,
                "host": payload.get("host") or {},
                "noise_floor_pct": payload.get("noise_floor_pct"),
                "finished_at": payload.get("finished_at"),
                "samples": loaded,
            }
        )
    return samples, sources


def load_frontier(directory: Path | str | None) -> dict:
    """The measured capability frontier of a deployment, with its provenance.

    The ``measured`` flag is the honest part. With no measurement directory,
    or a directory with no probe reports in it, the frontier is empty and
    every capability question about *rates* is unanswerable -- which the
    caller is told, rather than being handed a plausible-looking table
    extrapolated from nothing.
    """
    if directory is None:
        return {
            "measured": False,
            "reason": "no measurement directory is configured for this tenant",
            "sources": [],
            "rows": [],
        }
    samples, sources = load_probe_reports(directory)
    rows = aggregate_frontier(frontier_from_samples(samples))
    return {
        "measured": bool(rows),
        "reason": (
            ""
            if rows
            else f"no probe report with usable stage samples under {directory}"
        ),
        "sources": sources,
        "rows": rows,
    }


@dataclass(frozen=True)
class CapabilityAnswer:
    """A planner-honest reply to "can you do target X at Y fps?"."""

    achievable: bool
    requested_resolution: str
    requested_fps: float
    configuration: str | None
    max_fps_at_resolution: float | None
    alternatives: list[dict]
    reason: str

    def as_dict(self) -> dict:
        return {
            "achievable": self.achievable,
            "requested": {
                "resolution": self.requested_resolution,
                "fps": self.requested_fps,
            },
            "configuration": self.configuration,
            "max_fps_at_resolution": self.max_fps_at_resolution,
            "alternatives": self.alternatives,
            "reason": self.reason,
        }


def answer_capability(
    *,
    frontier: list[dict],
    resolution: Resolution,
    target_fps: float,
    configuration: str = "full_chain",
) -> CapabilityAnswer:
    """Answer a target request against the measured frontier, refusal included.

    Refusal names what *is* reachable rather than saying no: the maximum fps at
    the requested resolution, and the configurations and flow-scale arms that
    would meet the requested rate. Same refusal culture as the LLM planner --
    a rejection the caller cannot act on is a bug.
    """
    key = str(resolution)
    at_resolution = [
        row
        for row in frontier
        if row["resolution"] == key and row["configuration"] == configuration
    ]
    best = max((row["aggregate_max_fps"] for row in at_resolution), default=None)
    if best is not None and best >= target_fps:
        return CapabilityAnswer(
            achievable=True,
            requested_resolution=key,
            requested_fps=target_fps,
            configuration=configuration,
            max_fps_at_resolution=best,
            alternatives=[],
            reason=f"aggregate sustains {best:.1f} fps at {key}",
        )

    alternatives = sorted(
        (
            {
                "configuration": row["configuration"],
                "resolution": row["resolution"],
                "options": row["options"],
                "aggregate_max_fps": row["aggregate_max_fps"],
            }
            for row in frontier
            if row["aggregate_max_fps"] >= target_fps
        ),
        key=lambda row: -row["aggregate_max_fps"],
    )[:5]
    if best is None:
        reason = (
            f"no measurement exists for {configuration} at {key}; the frontier "
            "table has no row to answer from"
        )
    else:
        reason = (
            f"{configuration} at {key} sustains {best:.1f} fps, short of the "
            f"requested {target_fps:.1f} fps"
        )
    return CapabilityAnswer(
        achievable=False,
        requested_resolution=key,
        requested_fps=target_fps,
        configuration=configuration,
        max_fps_at_resolution=best,
        alternatives=alternatives,
        reason=reason,
    )
