#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""What a live preview tap costs the main chain, measured against a noise floor.

The §8.1 rule is that a preview must never *stall* the chain, and that is a
structural property proven on CPU in
`test/registered/video_enhance/test_preview.py`. This script answers the other
question, which no structural argument can settle: the tap's downscale and its
NVENC side-encode run on the same device as the chain, so they **compete** with
it. Competition is a throughput cost, and a throughput cost is a number.

Method, following the standing harness rules:

1.  **Noise floor first, by A-vs-A.** Two runs with taps *off*, interleaved
    like the real arms. Whatever spread they show is the floor, and nothing
    below it may be reported as an effect. A harness that skips this step can
    "measure" a 3 percent tap cost on a rig whose run-to-run spread is 4.
2.  **Interleaved, not blocked.** off, on, off, on, ... rather than all the
    taps-off runs and then every taps-on run, so a card that warms up or clocks down over the
    session biases both arms equally instead of only the second.
3.  **The output is the same bytes either way.** Each arm hashes its
    elementary stream; the arms must agree, because a tap that changed the
    deliverable would be a defect and not a cost.

    PYTHONPATH=python python scripts/video_enhance/preview_tap_bench.py \\
        --card 1 --frames 120 --reps 4 --out /tmp/taps
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path

from sglang.srt.video_enhance.chain import ChainRequest, StageKind, build_chain
from sglang.srt.video_enhance.frame_math import Resolution
from sglang.srt.video_enhance.mux import retimed_rate
from sglang.srt.video_enhance.pipeline import PipelineExecutor
from sglang.srt.video_enhance.preview import PreviewConfig, build_preview_lanes
from sglang.srt.video_enhance.ring import OverloadPolicy


async def _as_async(decode):
    for frame in decode:
        yield frame
        await asyncio.sleep(0)


def build_stages(chain, request: dict, source_url: str, device_id: int = 0):
    """The same chain the chunk worker builds, on ``cuda:0``."""
    from sglang.srt.video_enhance import codec
    from sglang.srt.video_enhance.resize import ResizeStage
    from sglang.srt.video_enhance.rife import RifeStage, download_weights
    from sglang.srt.video_enhance.sr import SuperResolutionStage

    source_res = Resolution.parse(request["source_resolution"])
    dtype = request.get("dtype", "fp16")
    model_dir = Path(request.get("model_dir", "/spinning/llm_stuff/k3-models"))
    out_fps = retimed_rate(
        Fraction(request["source_frame_rate"]), request["fps_multiplier"]
    )
    decode = codec.DecodeStage(
        source=source_url,
        resolution=source_res,
        device_id=device_id,
        backend="ffmpeg",
        frame_limit=request.get("frame_limit"),
    )
    stages: dict = {}
    for spec in chain.stages:
        if spec.kind is StageKind.DECODE:
            stages[spec.kind] = decode
        elif spec.kind is StageKind.COLOR_TO_RGB:
            stages[spec.kind] = codec.ColorToRgbStage(dtype=dtype)
        elif spec.kind is StageKind.SR:
            stages[spec.kind] = SuperResolutionStage.build(
                source=spec.in_res,
                fmt=spec.in_format,
                model_dir=model_dir / "sr",
                provider=request.get("sr_provider", "cuda"),
                precision="fp32"
                if request.get("sr_provider", "cuda") == "cuda"
                else dtype,
                device_id=device_id,
            )
        elif spec.kind is StageKind.RESIZE:
            stages[spec.kind] = ResizeStage(spec.in_res, spec.out_res, spec.in_format)
        elif spec.kind is StageKind.RIFE:
            stages[spec.kind] = RifeStage(
                resolution=spec.in_res,
                version=request.get("rife_version", "4.6"),
                multiplier=spec.arity_out + 1,
                scale=request.get("rife_scale", 1.0),
                dtype=dtype,
                device="cuda",
                weights_path=download_weights(
                    request.get("rife_version", "4.6"), model_dir / "rife"
                ),
            )
        elif spec.kind is StageKind.COLOR_TO_YUV:
            stages[spec.kind] = codec.ColorToYuvStage(dtype=dtype)
        elif spec.kind is StageKind.ENCODE:
            stages[spec.kind] = codec.EncodeStage(
                spec.in_res,
                fps=out_fps,
                codec=request.get("video_codec", "h264"),
                device_id=device_id,
                backend=request.get("encode_backend", "ffmpeg"),
                bitrate=request.get("bitrate"),
            )
    return stages, decode


async def one_run(chain, request: dict, source: Path, *, taps_on: bool, config) -> dict:
    """One arm: the whole clip through the chain, with or without the taps."""
    stages, decode = build_stages(chain, request, str(source))
    digest = hashlib.sha256()
    written = {"bytes": 0}

    async def sink(payload: bytes) -> None:
        digest.update(payload)
        written["bytes"] += len(payload)

    lanes = None
    taps = None
    if taps_on:
        lanes = build_preview_lanes(
            chain,
            fps=retimed_rate(
                Fraction(request["source_frame_rate"]), request["fps_multiplier"]
            ),
            dtype=request.get("dtype", "fp16"),
            device_id=0,
            config=config,
        )
        taps = dict(lanes.by_stage)
        lanes.start()
        # A viewer that never reads. This is the adversarial case on purpose:
        # the byte ring fills, the preview encoder stalls, and the tap drops on
        # ingress -- which is exactly the state the §8.1 rule is about, and the
        # state in which a badly built tap would show up as lost throughput.

    executor = PipelineExecutor(
        job_id="taps" if taps_on else "notaps",
        chain=chain,
        stages=stages,
        source=_as_async(decode),
        sink=sink,
        ring_depth=request.get("ring_depth", 2),
        policy=OverloadPolicy.STALL,
        taps=taps,
    )
    started = time.perf_counter()
    try:
        stats = await executor.run()
    finally:
        elapsed = time.perf_counter() - started
        if lanes is not None:
            await lanes.close()
        for stage in stages.values():
            closer = getattr(stage, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - one bad stage must not mask
                    continue

    out = {
        "taps_on": taps_on,
        "wall_seconds": round(elapsed, 4),
        "frames_encoded": stats.frames_encoded,
        "fps": round(stats.frames_encoded / elapsed, 3) if elapsed else None,
        "bytes_out": written["bytes"],
        "digest": digest.hexdigest()[:16],
        "stage_ms_per_frame": stats.stage_ms_per_frame,
    }
    if lanes is not None:
        out["previews"] = lanes.snapshot()
    return out


def summarise(label: str, arm: list[dict]) -> dict:
    fps = [r["fps"] for r in arm if r["fps"]]
    return {
        "label": label,
        "runs": len(fps),
        "fps_mean": round(statistics.mean(fps), 3) if fps else None,
        "fps_stdev": round(statistics.pstdev(fps), 3) if len(fps) > 1 else 0.0,
        "fps_min": round(min(fps), 3) if fps else None,
        "fps_max": round(max(fps), 3) if fps else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="#344a preview tap A/B")
    parser.add_argument("--card", default="1", help="NVML index")
    parser.add_argument("--out", default="/tmp/taps")
    parser.add_argument("--source", default=None)
    parser.add_argument("--source-resolution", default="960x540")
    parser.add_argument("--target", default="1920x1080")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--fps-multiplier", type=int, default=2)
    parser.add_argument("--reps", type=int, default=4, help="runs per arm")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--enable-sr", action="store_true", default=True)
    parser.add_argument("--no-sr", dest="enable_sr", action="store_false")
    parser.add_argument("--sr-scale", type=int, default=4)
    parser.add_argument("--sr-provider", default="cuda")
    parser.add_argument("--rife-version", default="4.6")
    parser.add_argument("--rife-scale", type=float, default=1.0)
    parser.add_argument("--encode-backend", default="ffmpeg")
    parser.add_argument("--bitrate", type=int, default=150_000_000)
    parser.add_argument("--ring-depth", type=int, default=2)
    parser.add_argument("--preview-width", type=int, default=480)
    parser.add_argument("--preview-bitrate", type=int, default=1_500_000)
    parser.add_argument("--preview-fps-divisor", type=int, default=1)
    parser.add_argument("--model-dir", default="/spinning/llm_stuff/k3-models")
    args = parser.parse_args(argv)

    workdir = Path(args.out)
    workdir.mkdir(parents=True, exist_ok=True)
    source_res = Resolution.parse(args.source_resolution)

    if args.source:
        source = Path(args.source)
    else:
        from sglang.srt.video_enhance.codec import make_test_clip

        source = workdir / "source.mp4"
        if not source.is_file():
            make_test_clip(source, source_res, args.frames, args.fps)

    chain = build_chain(
        ChainRequest(
            source=source_res,
            target=Resolution.parse(args.target),
            fps_multiplier=args.fps_multiplier,
            dtype=args.dtype,
            enable_sr=args.enable_sr,
            sr_scale=args.sr_scale,
            enable_resize=args.enable_sr,
            rife_scale=args.rife_scale,
            rife_version=args.rife_version,
            streams_in_flight=1,
        )
    )
    request = {
        "source_resolution": args.source_resolution,
        "dtype": args.dtype,
        "fps_multiplier": args.fps_multiplier,
        "source_frame_rate": str(args.fps),
        "model_dir": args.model_dir,
        "sr_provider": args.sr_provider,
        "rife_version": args.rife_version,
        "rife_scale": args.rife_scale,
        "encode_backend": args.encode_backend,
        "bitrate": args.bitrate,
        "ring_depth": args.ring_depth,
        "frame_limit": args.frames,
    }
    config = PreviewConfig(
        width=args.preview_width,
        bitrate=args.preview_bitrate,
        fps_divisor=args.preview_fps_divisor,
    )

    report: dict = {
        "args": vars(args),
        "chain": [s.describe() for s in chain.stages],
    }

    # -- phase 1: the noise floor, A vs A -----------------------------------
    # Both arms taps-off. Whatever they differ by is what this rig's
    # run-to-run spread is, and no smaller difference may be called an effect.
    print("== noise floor: taps off vs taps off ==", flush=True)
    floor_a: list[dict] = []
    floor_b: list[dict] = []
    for rep in range(args.reps):
        floor_a.append(
            asyncio.run(one_run(chain, request, source, taps_on=False, config=config))
        )
        floor_b.append(
            asyncio.run(one_run(chain, request, source, taps_on=False, config=config))
        )
        print(
            f"  rep {rep}: A {floor_a[-1]['fps']} fps   B {floor_b[-1]['fps']} fps",
            flush=True,
        )

    a_sum, b_sum = summarise("floor_a", floor_a), summarise("floor_b", floor_b)
    # The floor is the larger of "how far the two A-vs-A means sat apart" and
    # "how much one arm wandered on its own". Taking only the first would call
    # a noisy rig quiet whenever its two halves happened to agree.
    floor_pct = max(
        abs(a_sum["fps_mean"] - b_sum["fps_mean"]) / a_sum["fps_mean"] * 100.0,
        (a_sum["fps_stdev"] + b_sum["fps_stdev"]) / a_sum["fps_mean"] * 100.0,
    )
    report["noise_floor"] = {"a": a_sum, "b": b_sum, "floor_pct": round(floor_pct, 3)}
    print(f"  noise floor = {floor_pct:.3f}%", flush=True)

    # -- phase 2: the real arms, interleaved --------------------------------
    print("== taps off vs taps on, interleaved ==", flush=True)
    off: list[dict] = []
    on: list[dict] = []
    for rep in range(args.reps):
        off.append(
            asyncio.run(one_run(chain, request, source, taps_on=False, config=config))
        )
        on.append(
            asyncio.run(one_run(chain, request, source, taps_on=True, config=config))
        )
        print(
            f"  rep {rep}: off {off[-1]['fps']} fps   on {on[-1]['fps']} fps",
            flush=True,
        )

    off_sum, on_sum = summarise("taps_off", off), summarise("taps_on", on)
    delta_pct = (off_sum["fps_mean"] - on_sum["fps_mean"]) / off_sum["fps_mean"] * 100.0
    verdict = "MEASURABLE" if abs(delta_pct) > floor_pct else "below noise"

    # The deliverable must be byte-identical with and without the tap. A tap
    # that changed the output would be a defect, and no throughput number
    # would be worth reading.
    digests_off = {r["digest"] for r in off}
    digests_on = {r["digest"] for r in on}
    output_unchanged = len(digests_off | digests_on) == 1

    report["arms"] = {"off": off_sum, "on": on_sum}
    report["result"] = {
        "throughput_cost_pct": round(delta_pct, 3),
        "noise_floor_pct": round(floor_pct, 3),
        "verdict": verdict,
        "output_byte_identical_with_and_without_taps": output_unchanged,
        "digests_off": sorted(digests_off),
        "digests_on": sorted(digests_on),
    }
    report["preview_delivery"] = [r.get("previews") for r in on]
    report["runs"] = {"floor_a": floor_a, "floor_b": floor_b, "off": off, "on": on}

    (workdir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["result"], indent=2))
    if not output_unchanged:
        print(
            "FAILED: the encoded output differs with taps on; a preview tap must "
            "not touch the deliverable",
            file=sys.stderr,
        )
        return 1
    print(
        f"taps cost {delta_pct:+.2f}% of main-chain throughput "
        f"(noise floor {floor_pct:.2f}%) -- {verdict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
