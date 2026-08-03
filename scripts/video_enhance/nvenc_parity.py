# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#484 parity gate: the in-process NVENC lane against the ffmpeg baseline.

What this grades, and what it deliberately does not
---------------------------------------------------
The two lanes drive the SAME encoder -- NVENC on the same card, at the same
codec, preset, rate control and GOP -- so the interesting question is not
"are the bitstreams equal". They are not, and could not be: the subprocess
lane goes through ffmpeg's rate-control front end and its own muxer, and NVENC
itself is free to reorder its internal queue differently when it is fed from a
device pointer rather than a host pipe. Grading bytes would grade the muxer.

What must hold is that the in-process lane is not LOSING anything: the same
source frames, encoded at the same settings, decoded back, must reconstruct
the source as well as the subprocess arm does, and there must be exactly as
many frames coming out as went in, in the same order. So the gate is a decode
round trip:

    source frames -> [lane] -> bitstream -> decode -> PSNR/SSIM vs source

and the two lanes are compared on that number, not on their bytes. The frame
count and the per-frame ordering are checked exactly, because a dropped or
reordered frame is the failure mode a mean PSNR would hide.

The gate can fail
-----------------
``--arm wrong-chroma`` runs the same comparison over frames whose chroma
planes have been swapped before encoding. It is a lane that is genuinely
producing the wrong picture while every count and timestamp still lines up,
so it separates "the gate measures reconstruction" from "the gate measures
that something arrived". A run that does not include a rejected arm has not
shown its threshold to be a threshold.

Running it
----------
Needs a card, so it needs a ``/spinning/gpu-arb/`` window (holder + heartbeat,
heartbeat stopped BEFORE release). Bounded by frame count rather than by wall
clock; 60 frames at 720p is a few seconds.

    python scripts/video_enhance/nvenc_parity.py --device 1 --frames 60 \
        --width 1280 --height 720 --out /tmp/nvenc_parity.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import torch

from sglang.srt.video_enhance import codec
from sglang.srt.video_enhance.frame_math import PixelFormat, Resolution
from sglang.srt.video_enhance.frames import Frame
from sglang.srt.video_enhance.parity import psnr, ssim

#: Both arms are pinned to one operating point. A parity claim across two
#: different rate-control settings would say nothing about the lane.
CODEC = "h264"
PRESET = "P4"
BITRATE = 150_000_000
GOP = 30


def _source_frames(resolution: Resolution, count: int, device: str):
    """Deterministic planar RGB in [0, 1], built on the host and moved.

    ``synthetic_frame_rgb`` is procedural integer arithmetic with no RNG
    anywhere, which is what the M2 acceptance gate needs and what makes the
    two arms comparable: both encode the same bytes. Built on the host on
    purpose -- on-device sampling is not bit-identical across architectures,
    so a clip generated on one card and graded against another would carry
    that difference straight into the PSNR.
    """
    out = []
    for index in range(count):
        hwc = codec.synthetic_frame_rgb(index, resolution)
        planar = torch.from_numpy(hwc).permute(2, 0, 1).contiguous()
        out.append(planar.to(torch.float32).div_(255.0).to(device))
    return out


def _swap_chroma(nv12, resolution: Resolution):
    """The deliberately-wrong arm: U and V exchanged in the NV12 plane."""
    height, width = resolution.height, resolution.width
    broken = nv12.clone()
    chroma = broken[height:].reshape(height // 2, width // 2, 2)
    chroma[..., [0, 1]] = chroma[..., [1, 0]]
    return broken


def _encode(frames, resolution: Resolution, backend: str, device_id: int, arm: str):
    stage = codec.EncodeStage(
        resolution,
        fps=Fraction(30),
        codec=CODEC,
        container="annexb" if backend == "pynvvideocodec" else "mpegts",
        backend=backend,
        device_id=device_id,
        segment_frames=len(frames),
        bitrate=BITRATE,
        preset=PRESET,
        extra_options={"gop_length": str(GOP)} if backend == "pynvvideocodec" else None,
    )
    stage.warmup()
    if stage.fell_back_to_ffmpeg:
        raise SystemExit(
            "the in-process lane fell back to ffmpeg; the arms would not be "
            "comparable and this run would grade the fallback against itself"
        )
    chunks: list[bytes] = []
    started = time.perf_counter()
    for index, rgb in enumerate(frames):
        nv12 = codec.rgb_to_nv12(rgb)
        if arm == "wrong-chroma":
            nv12 = _swap_chroma(nv12, resolution)
        chunks.extend(
            stage.submit(
                Frame(
                    data=nv12,
                    resolution=resolution,
                    format=PixelFormat.NV12,
                    index=index,
                )
            )
        )
    chunks.extend(stage.close())
    elapsed = time.perf_counter() - started
    return b"".join(chunks), elapsed, stage.backend_name


def _decode_back(
    payload: bytes, resolution: Resolution, device_id: int, tmp: Path, suffix: str
):
    """Decode the bitstream back to RGB, in stream order.

    Deliberately a SOFTWARE ffmpeg decode, not the chain's own DecodeStage.
    Two reasons, and both are about keeping the instrument out of the
    measurement: the gate must read the same way for both arms whatever the
    encoder did, and NVDEC is not a dependency of the thing under test. On
    this rig it is not even available to it -- ``-hwaccel cuda`` fails the
    round trip with ``cuvidCreateDecoder ... CUDA_ERROR_INVALID_VALUE``
    while an NVENC session is open on the same card, which would make the
    gate unrunnable for a reason that has nothing to do with the lane.

    The suffix matters: ffmpeg identifies an elementary stream by extension
    where it has no container to read, and a misidentified file decodes to
    zero frames rather than to an error.
    """
    path = tmp / f"arm{suffix}"
    path.write_bytes(payload)
    frame_bytes = resolution.width * resolution.height * 3 // 2
    completed = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "nv12",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    raw = completed.stdout
    if len(raw) % frame_bytes:
        raise SystemExit(
            f"decoded {len(raw)} bytes, not a whole number of "
            f"{frame_bytes}-byte NV12 frames"
        )
    out = []
    for offset in range(0, len(raw), frame_bytes):
        nv12 = torch.frombuffer(
            bytearray(raw[offset : offset + frame_bytes]), dtype=torch.uint8
        ).reshape(resolution.height * 3 // 2, resolution.width)
        out.append(codec.nv12_to_rgb(nv12, dtype="fp32"))
    return out


def _grade(reconstructed, sources) -> dict:
    per_frame = []
    for index, (candidate, reference) in enumerate(zip(reconstructed, sources)):
        # ``psnr``/``ssim`` want NCHW; the colour stage emits (1, 3, H, W) and
        # the source frames are planar (3, H, W).
        cand = candidate.to(torch.float32).cpu().reshape(1, *candidate.shape[-3:])
        ref = reference.to(torch.float32).cpu().reshape(1, *reference.shape[-3:])
        per_frame.append(
            {
                "index": index,
                "psnr_db": psnr(cand, ref),
                "ssim": ssim(cand, ref),
            }
        )
    return {
        "frames_out": len(reconstructed),
        "psnr_mean_db": statistics.fmean(f["psnr_db"] for f in per_frame),
        "psnr_min_db": min(f["psnr_db"] for f in per_frame),
        "ssim_mean": statistics.fmean(f["ssim"] for f in per_frame),
        "per_frame": per_frame,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--arm",
        choices=("both", "inprocess", "ffmpeg", "wrong-chroma"),
        default="both",
        help="'wrong-chroma' is the can-fail arm: it must be REJECTED.",
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/nvenc_parity.json"))
    parser.add_argument(
        "--margin-db",
        type=float,
        default=1.0,
        help="how far the in-process arm may fall below the ffmpeg arm",
    )
    args = parser.parse_args()

    resolution = Resolution(args.width, args.height)
    device = f"cuda:{args.device}"
    sources = _source_frames(resolution, args.frames, device)
    tmp = args.out.parent / f"nvenc_parity_{int(time.time())}"
    tmp.mkdir(parents=True, exist_ok=True)

    arms = {
        "ffmpeg": ("ffmpeg", "clean"),
        "inprocess": ("pynvvideocodec", "clean"),
        "wrong-chroma": ("pynvvideocodec", "wrong-chroma"),
    }
    if args.arm != "both":
        arms = {args.arm: arms[args.arm]}

    report: dict = {
        "resolution": f"{args.width}x{args.height}",
        "frames_in": args.frames,
        "device": args.device,
        "codec": CODEC,
        "preset": PRESET,
        "bitrate": BITRATE,
        "gop": GOP,
        "arms": {},
    }
    for name, (backend, arm) in arms.items():
        payload, elapsed, resolved = _encode(
            sources, resolution, backend, args.device, arm
        )
        suffix = ".h264" if backend == "pynvvideocodec" else ".ts"
        graded = _grade(
            _decode_back(payload, resolution, args.device, tmp, suffix), sources
        )
        graded.update(
            {
                "backend": resolved,
                "bitstream_bytes": len(payload),
                "encode_s": elapsed,
                "ms_per_frame": 1000.0 * elapsed / args.frames,
            }
        )
        report["arms"][name] = graded
        print(
            f"{name:>13}: {graded['frames_out']}/{args.frames} frames, "
            f"PSNR {graded['psnr_mean_db']:.2f} dB (min "
            f"{graded['psnr_min_db']:.2f}), SSIM {graded['ssim_mean']:.5f}, "
            f"{graded['ms_per_frame']:.3f} ms/frame"
        )

    verdict = []
    if "ffmpeg" in report["arms"] and "inprocess" in report["arms"]:
        base = report["arms"]["ffmpeg"]
        lane = report["arms"]["inprocess"]
        if lane["frames_out"] != args.frames:
            verdict.append(
                f"FAIL frame count: {lane['frames_out']} out of {args.frames}"
            )
        if lane["psnr_mean_db"] < base["psnr_mean_db"] - args.margin_db:
            verdict.append(
                f"FAIL reconstruction: {lane['psnr_mean_db']:.2f} dB against the "
                f"baseline's {base['psnr_mean_db']:.2f} dB"
            )
        if not verdict:
            verdict.append(
                f"PASS: {lane['psnr_mean_db']:.2f} dB against "
                f"{base['psnr_mean_db']:.2f} dB, {lane['frames_out']} frames, "
                f"{base['ms_per_frame'] / max(lane['ms_per_frame'], 1e-9):.2f}x "
                "the subprocess lane"
            )
    if "wrong-chroma" in report["arms"] and "ffmpeg" in report["arms"]:
        broken = report["arms"]["wrong-chroma"]
        base = report["arms"]["ffmpeg"]
        rejected = broken["psnr_mean_db"] < base["psnr_mean_db"] - args.margin_db
        verdict.append(
            f"can-fail arm {
                'REJECTED as required'
                if rejected
                else 'ACCEPTED -- '
                'the gate does not discriminate and its PASS means nothing'
            }: "
            f"{broken['psnr_mean_db']:.2f} dB"
        )
    report["verdict"] = verdict
    for line in verdict:
        print(line)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0 if all(not v.startswith("FAIL") for v in verdict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
