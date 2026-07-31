#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""End-to-end functional proof of the Class-3 enhance chain on one card.

Builds a multi-track source (video + audio + subtitles), runs it through the
real executor with the real stages, remuxes, and then validates the output
against the properties that matter:

*   the enhanced video has the geometry and frame count the retiming
    arithmetic predicts,
*   every non-video track survives bit-identically (hash of the demuxed
    packets, before and after),
*   audio duration is unchanged, so A/V sync is preserved,
*   two identical runs produce byte-identical output.

Run inside a card window, pinned to one physical GPU:

    CUDA_VISIBLE_DEVICES=<nvml index> \\
    PYTHONPATH=python python scripts/video_enhance/e2e_chain.py --out /tmp/e2e
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from sglang.srt.video_enhance.chain import ChainRequest, StageKind, build_chain
from sglang.srt.video_enhance.frame_math import Resolution
from sglang.srt.video_enhance.mux import (
    StreamRemuxer,
    TrackSelection,
    alignment_report,
    build_remux_command,
    expected_frame_count,
    probe,
    retimed_rate,
    strip_empty_mov_text,
)
from sglang.srt.video_enhance.pipeline import PipelineExecutor
from sglang.srt.video_enhance.ring import OverloadPolicy


def build_source(path: Path, resolution: Resolution, frames: int, fps: int) -> Path:
    """A deterministic clip with one video, two audio and one subtitle track."""
    from sglang.srt.video_enhance.codec import make_test_clip

    video = path.with_name("src_video.mp4")
    make_test_clip(video, resolution, frames, fps)

    subs = path.with_name("src.srt")
    subs.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nmarker one\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nmarker two\n\n"
    )
    duration = frames / fps
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        # A click on a known beat: an audible marker for the sync check.
        "-f",
        "lavfi",
        "-t",
        f"{duration}",
        "-i",
        "sine=frequency=1000:sample_rate=48000:duration=" + str(duration),
        "-f",
        "lavfi",
        "-t",
        f"{duration}",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=" + str(duration),
        "-i",
        str(subs),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "2:a:0",
        "-map",
        "3:s:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-c:s",
        "mov_text",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:1",
        "language=deu",
        "-metadata:s:s:0",
        "language=eng",
        "-movflags",
        "+faststart",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


async def _as_async(decode):
    """Adapt the synchronous pull-source decoder to the executor's async source.

    The decoder is deliberately a pull iterator (§8.4 rule 2), so the adapter
    yields one frame per await and never runs ahead: back-pressure still stops
    at the decoder.
    """
    for frame in decode:
        yield frame
        await asyncio.sleep(0)


def stream_bytes(path: Path, selector: str) -> bytes:
    """The demuxed packet payload of one stream, concatenated."""
    out = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            selector,
            "-c",
            "copy",
            "-f",
            "data",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return out.stdout


def stream_hash(path: Path, selector: str) -> str:
    """SHA-256 of one demuxed stream's packets: the bit-identity check."""
    return hashlib.sha256(stream_bytes(path, selector)).hexdigest()


def subtitle_content_hash(path: Path, selector: str) -> str:
    """SHA-256 of a subtitle track with mov_text gap samples removed.

    The raw-byte comparison used for audio is the wrong instrument for
    mov_text: the codec has no gap representation, so the muxer is obliged to
    emit an empty sample wherever no cue is on screen, and an output that runs
    longer than the source necessarily gains one at the tail. Filtering the
    empty samples is what separates "the text survived" from "the container
    added padding it had to add".
    """
    return hashlib.sha256(
        strip_empty_mov_text(stream_bytes(path, selector))
    ).hexdigest()


async def run_chain(source: Path, out_path: Path, args) -> dict:
    from sglang.srt.video_enhance import codec
    from sglang.srt.video_enhance.resize import ResizeStage
    from sglang.srt.video_enhance.rife import RifeStage, download_weights
    from sglang.srt.video_enhance.sr import SuperResolutionStage

    info = probe(str(source))
    src_res = Resolution(info.track(0).width, info.track(0).height)
    request = ChainRequest(
        source=src_res,
        target=Resolution.parse(args.target),
        fps_multiplier=args.fps_multiplier,
        dtype=args.dtype,
        enable_sr=args.enable_sr,
        streams_in_flight=1,
    )
    chain = build_chain(request)
    print("chain:\n" + chain.describe(), flush=True)

    source_rate = info.track(0).frame_rate()
    out_fps = retimed_rate(source_rate, args.fps_multiplier)
    stages: dict = {}
    decode = codec.DecodeStage(source=str(source), resolution=src_res, device_id=0)
    for spec in chain.stages:
        if spec.kind is StageKind.DECODE:
            stages[spec.kind] = decode
        elif spec.kind is StageKind.COLOR_TO_RGB:
            stages[spec.kind] = codec.ColorToRgbStage(dtype=args.dtype)
        elif spec.kind is StageKind.SR:
            stages[spec.kind] = SuperResolutionStage.build(
                source=spec.in_res,
                fmt=spec.in_format,
                model_dir=Path(args.model_dir) / "sr",
                provider=args.sr_provider,
                precision="fp32" if args.sr_provider == "cuda" else args.dtype,
                device_id=0,
            )
        elif spec.kind is StageKind.RESIZE:
            stages[spec.kind] = ResizeStage(spec.in_res, spec.out_res, spec.in_format)
        elif spec.kind is StageKind.RIFE:
            stages[spec.kind] = RifeStage(
                resolution=spec.in_res,
                version=args.rife_version,
                multiplier=spec.arity_out + 1,
                scale=args.rife_scale,
                dtype=args.dtype,
                device="cuda",
                weights_path=download_weights(
                    args.rife_version, Path(args.model_dir) / "rife"
                ),
            )
        elif spec.kind is StageKind.COLOR_TO_YUV:
            stages[spec.kind] = codec.ColorToYuvStage(dtype=args.dtype)
        elif spec.kind is StageKind.ENCODE:
            stages[spec.kind] = codec.EncodeStage(
                spec.in_res,
                fps=out_fps,
                codec=args.video_codec,
                device_id=0,
                backend=args.encode_backend,
            )

    out_rate = out_fps
    remuxer = StreamRemuxer(
        build_remux_command(
            source_url=str(source),
            info=info,
            selection=TrackSelection(),
            enhanced_codec=args.video_codec,
            output_rate=out_rate,
            container="mp4",
        )
    )
    await remuxer.start()

    # Tee the encoded elementary stream before it reaches the muxer. Two runs
    # that differ only after this point implicate the muxer; two runs that
    # already differ here implicate the encoder or a stage above it. Without
    # the tee "the output is not byte-stable" names no layer.
    es_digest = hashlib.sha256()
    es_bytes = {"n": 0}

    async def sink(payload: bytes) -> None:
        es_digest.update(payload)
        es_bytes["n"] += len(payload)
        await remuxer.feed(payload)

    executor = PipelineExecutor(
        job_id="e2e",
        chain=chain,
        stages=stages,
        source=_as_async(decode),
        sink=sink,
        ring_depth=2,
        policy=OverloadPolicy.STALL,
    )

    written = {"bytes": 0}

    async def drain() -> None:
        with open(out_path, "wb") as fh:
            async for chunk in remuxer.read_chunks():
                fh.write(chunk)
                written["bytes"] += len(chunk)

    reader = asyncio.create_task(drain())
    started = time.perf_counter()
    stats = await executor.run()
    await remuxer.close_input()
    await reader
    await remuxer.wait()
    elapsed = time.perf_counter() - started

    for stage in stages.values():
        close = getattr(stage, "close", None)
        if callable(close):
            close()

    return {
        "elapsed_s": round(elapsed, 3),
        "bytes_out": written["bytes"],
        "elementary_sha256": es_digest.hexdigest(),
        "elementary_bytes": es_bytes["n"],
        "stats": stats.snapshot(),
        "source_rate": str(source_rate),
        "output_rate": str(out_rate),
    }


def validate(source: Path, out: Path, args, result: dict) -> dict:
    src = probe(str(source))
    dst = probe(str(out))
    dst_video = dst.video_tracks[0]

    src_frames = int(
        json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-print_format",
                    "json",
                    str(source),
                ],
                check=True,
                capture_output=True,
            ).stdout
        )["streams"][0]["nb_read_frames"]
    )
    dst_frames = int(
        json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-print_format",
                    "json",
                    str(out),
                ],
                check=True,
                capture_output=True,
            ).stdout
        )["streams"][0]["nb_read_frames"]
    )

    checks: dict = {
        "video_geometry": {
            "expected": args.target,
            "actual": f"{dst_video.width}x{dst_video.height}",
            "pass": f"{dst_video.width}x{dst_video.height}" == args.target,
        },
        "frame_count": {
            "source": src_frames,
            "expected": expected_frame_count(src_frames, args.fps_multiplier),
            "actual": dst_frames,
            "pass": dst_frames == expected_frame_count(src_frames, args.fps_multiplier),
        },
        "track_count": {
            "source": len(src.tracks),
            "output": len(dst.tracks),
            "pass": len(dst.tracks) == len(src.tracks),
        },
    }

    # Bit-identity of every passthrough track. Audio must be byte-for-byte
    # equal; a subtitle track is compared on content, because mov_text's gap
    # samples are muxer output rather than source content -- see
    # ``subtitle_content_hash``.
    passthrough = {}
    for kind, count in (("a", 2), ("s", 1)):
        for i in range(count):
            selector = f"0:{kind}:{i}"
            digest = subtitle_content_hash if kind == "s" else stream_hash
            try:
                before = digest(source, selector)
                after = digest(out, selector)
                raw_before = stream_hash(source, selector)
                raw_after = stream_hash(out, selector)
            except subprocess.CalledProcessError as exc:
                passthrough[selector] = {
                    "pass": False,
                    "error": exc.stderr.decode()[:200],
                }
                continue
            passthrough[selector] = {
                "sha256_source": before[:16],
                "sha256_output": after[:16],
                "compared": "content" if kind == "s" else "raw_bytes",
                "raw_bytes_identical": raw_before == raw_after,
                "pass": before == after,
            }
    checks["passthrough_bit_identical"] = passthrough

    # Inter-track start offsets. The duration delta alone cannot see a
    # constant lag of every copied track against the enhanced video, which is
    # exactly what a fragmented container that drops edit lists produces.
    try:
        report = alignment_report(str(source), str(out))
        # 1 ms, not one frame. Container timescales quantise a start offset by
        # tens of microseconds, so exact zero is not reachable; a frame-sized
        # tolerance would have accepted the 21.333 ms edit-list loss this check
        # exists to catch.
        tolerance_s = 0.001
        checks["av_alignment"] = {
            **report.as_dict(),
            "tolerance_s": tolerance_s,
            "pass": report.max_drift_s <= tolerance_s,
        }
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        checks["av_alignment"] = {
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    checks["av_sync"] = {
        "source_duration_s": src.duration_s,
        "output_duration_s": dst.duration_s,
        "delta_s": (
            round(dst.duration_s - src.duration_s, 4)
            if (src.duration_s and dst.duration_s)
            else None
        ),
    }
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="#333 M2 end-to-end functional proof")
    parser.add_argument("--out", default="/tmp/e2e")
    parser.add_argument("--source-resolution", default="960x540")
    parser.add_argument("--target", default="1920x1080")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--fps-multiplier", type=int, default=2)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--enable-sr", action="store_true", default=True)
    parser.add_argument("--no-sr", dest="enable_sr", action="store_false")
    parser.add_argument("--sr-provider", default="cuda")
    parser.add_argument("--rife-version", default="4.6")
    parser.add_argument("--rife-scale", type=float, default=1.0)
    parser.add_argument("--video-codec", default="h264")
    parser.add_argument("--encode-backend", default="auto")
    parser.add_argument("--model-dir", default="/spinning/llm_stuff/k3-models")
    parser.add_argument("--repeat", type=int, default=2, help="runs for byte stability")
    args = parser.parse_args(argv)

    workdir = Path(args.out)
    workdir.mkdir(parents=True, exist_ok=True)
    source = build_source(
        workdir / "source.mp4",
        Resolution.parse(args.source_resolution),
        args.frames,
        args.fps,
    )
    print(f"source: {source} ({source.stat().st_size} bytes)", flush=True)

    digests = []
    elementary = []
    result = {}
    for run in range(args.repeat):
        out = workdir / f"out{run}.mp4"
        result = asyncio.run(run_chain(source, out, args))
        digests.append(hashlib.sha256(out.read_bytes()).hexdigest())
        elementary.append(result["elementary_sha256"])
        print(f"run {run}: {json.dumps(result)}", flush=True)

    checks = validate(source, workdir / "out0.mp4", args, result)
    # Two layers, reported separately. The muxer is byte-stable given an
    # identical elementary stream (proved on CPU in test_mux.py), so a
    # container digest that moves while the elementary digest holds is a muxer
    # bug, and one where both move is upstream of the muxer.
    checks["byte_stable_elementary"] = {
        "digests": [d[:16] for d in elementary],
        "pass": len(set(elementary)) == 1,
    }
    checks["byte_stable_across_runs"] = {
        "digests": [d[:16] for d in digests],
        "layer": (
            "muxer"
            if len(set(elementary)) == 1 and len(set(digests)) > 1
            else "chain"
            if len(set(digests)) > 1
            else "stable"
        ),
        "pass": len(set(digests)) == 1,
    }
    report = {"args": vars(args), "result": result, "checks": checks}
    (workdir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(checks, indent=2, sort_keys=True))

    failures = [
        name
        for name, value in checks.items()
        if isinstance(value, dict)
        and (
            value.get("pass") is False
            or any(
                isinstance(v, dict) and v.get("pass") is False for v in value.values()
            )
        )
    ]
    if failures:
        print(f"FAILED: {failures}", file=sys.stderr)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
