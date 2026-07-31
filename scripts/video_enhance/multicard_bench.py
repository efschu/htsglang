#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Multi-card chunk execution, measured against the single-card baseline.

Three phases, in this order, because the third is only worth reading if the
second passed:

1.  **Calibrate.** A short chunk on every offered card, one at a time, to fill
    the measurement post P1 rate table with real per-(stage, card) numbers.
    The capacity weighting is only as good as this table, and inventing it
    would make the split look balanced on paper and be crooked in practice.
2.  **Correctness.** Baseline and multi-card runs of the *same* clip, then a
    frame-count check and a PSNR comparison of the two decoded outputs. A
    seek that lands one frame off, or a seam that duplicates a frame, shows
    up here as a PSNR cliff and nowhere else -- both produce a file that
    plays.
3.  **Throughput.** Wall clock end to end and per-stage ms/frame per card,
    against the single-card baseline. The baseline runs through the same
    executor with a one-chunk plan, so the only difference between the two
    numbers is the number of cards.

Card discipline: every chunk worker is its own process with
``CUDA_VISIBLE_DEVICES`` set to exactly one NVML index. Nothing here holds a
CUDA context in the parent.

    PYTHONPATH=python python scripts/video_enhance/multicard_bench.py \\
        --cards 1,0,2 --frames 240 --out /tmp/mc
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

from sglang.srt.video_enhance.chain import ChainRequest, build_chain
from sglang.srt.video_enhance.frame_math import MIB, Resolution
from sglang.srt.video_enhance.multicard import (
    ChunkSpec,
    MultiCardExecutor,
    PersistentChunkRunner,
    SubprocessChunkRunner,
    chunk_specs_from_plan,
    pull_queue_chunks,
    pull_queue_size,
    resolve_cards,
    verify_chunk_arithmetic,
)
from sglang.srt.video_enhance.mux import (
    StreamRemuxer,
    TrackSelection,
    build_remux_command,
    expected_frame_count,
    probe,
    retimed_rate,
)
from sglang.srt.video_enhance.shard_plan import (
    CardAvailability,
    RateTable,
    ReservationInputs,
    StageRate,
    capacity_weighted_plan,
    predict_makespan,
)


def sh(cmd: list[str]) -> bytes:
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def count_frames(path: Path) -> int:
    payload = json.loads(
        sh(
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
                str(path),
            ]
        )
    )
    return int(payload["streams"][0]["nb_read_frames"])


def psnr_between(a: Path, b: Path) -> float | None:
    """ffmpeg's own PSNR between two decoded videos, in dB.

    Used as the correctness gate for the stitch: a frame duplicated or lost at
    a seam shifts every frame after it, which collapses PSNR from ~infinity
    to the low twenties. It is a much sharper instrument than "the file
    plays".
    """
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(a),
            "-i",
            str(b),
            "-lavfi",
            "[0:v][1:v]psnr",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
    )
    for line in proc.stderr.decode(errors="replace").splitlines()[::-1]:
        if "average:" in line and "PSNR" in line:
            for token in line.split():
                if token.startswith("average:"):
                    value = token.split(":", 1)[1]
                    return float("inf") if value == "inf" else float(value)
    return None


def cross_card_psnr(value: float | None, compared: str) -> dict:
    """A cross-architecture PSNR: recorded, deliberately not graded.

    It was a 40 dB gate, and that was a category error the numbers exposed
    rather than an assertion anyone could defend. 40 dB is the floor
    ``parity.py`` derives for **fp16 against fp32 on one card** -- the point
    where mean squared error drops below the 8-bit output step. This
    comparison is a different thing: identical chunk boundaries and identical
    GOP structure, but the frames were produced on different architectures,
    whose convolution kernels are not bit-identical. TASK_333 §9.4 already
    recorded 37.4 dB here and described it, correctly, as "cross-architecture
    arithmetic, measured, not passed" -- so the code was failing a run on a
    threshold the document it implements had already disowned.

    What grades the stitch is ``*_seam_is_exact``: a sha256 per frame taken
    immediately before the encoder, against the un-chunked run on the same
    card. That is exact, and it is a gate. This is context next to it.
    """
    return {
        "psnr_db": value,
        "compared": compared,
        "note": (
            "measurement, not a gate: the two arms ran on different "
            "architectures, whose convolutions are not bit-identical. The "
            "40 dB figure in parity.py is a same-card fp16-vs-fp32 floor and "
            "does not apply here; the seam gate is the digest comparison"
        ),
    }


def without_digests(arm: dict) -> dict:
    """An arm's record with the per-frame hash lists replaced by their counts.

    A 480-frame run carries 959 digests per arm and their value is entirely in
    the comparison, which lives in ``checks``. Printing them makes the log
    unreadable and hides the numbers a reader came for.
    """
    trimmed = {k: v for k, v in arm.items() if k != "frame_digests"}
    if "frame_digests" in arm:
        trimmed["frame_digest_count"] = len(arm["frame_digests"])
    trimmed["chunks"] = [
        {k: v for k, v in entry.items() if k != "frame_digests"}
        for entry in arm.get("chunks", [])
    ]
    return trimmed


def request_payload(args, source_rate: Fraction) -> dict:
    return {
        "source_resolution": args.source_resolution,
        "target": args.target,
        "dtype": args.dtype,
        "enable_sr": args.enable_sr,
        "sr_scale": args.sr_scale,
        "enable_resize": args.enable_sr,
        "rife_scale": args.rife_scale,
        "rife_version": args.rife_version,
        "streams_in_flight": 1,
        "source_frame_rate": str(source_rate),
        "model_dir": args.model_dir,
        "sr_provider": args.sr_provider,
        "video_codec": args.video_codec,
        "encode_backend": args.encode_backend,
        # ffmpeg, not NVDEC: only the ffmpeg backend can seek, and a shard
        # that cannot seek pays for the whole prefix of the clip. Forced for
        # both arms so the baseline and the multi-card run differ in exactly
        # one thing.
        "decode_backend": "ffmpeg",
        "bitrate": args.bitrate,
        "frame_digests": args.frame_digests,
        "ring_depth": args.ring_depth,
    }


async def run_chunks(
    chunks,
    *,
    source: Path,
    request: dict,
    out_es: Path,
    job_id: str,
    spool: Path,
    cards: list[str] | None = None,
    persistent: bool = False,
) -> dict:
    """One arm. ``cards`` selects pull scheduling, ``persistent`` the runner.

    The two are separable on purpose. Pull scheduling and the persistent
    worker are one change in intent and two in mechanism, and a measurement
    that could not separate them would not be able to say which of the two
    produced whatever the numbers show.
    """
    if persistent:
        runner: object = PersistentChunkRunner(source_url=str(source), request=request)
    else:
        runner = SubprocessChunkRunner(source_url=str(source), request=request)
    handle = open(out_es, "wb")

    async def sink(payload: bytes) -> None:
        handle.write(payload)

    executor = MultiCardExecutor(
        job_id=job_id,
        chunks=chunks,
        runner=runner,
        sink=sink,
        spool_dir=spool,
        cards=cards,
    )
    started = time.perf_counter()
    try:
        stats = await executor.run()
    finally:
        handle.close()
    snapshot = stats.snapshot()
    snapshot["wall_seconds"] = round(time.perf_counter() - started, 3)
    if persistent:
        snapshot["worker_spawn_seconds"] = dict(runner.spawn_seconds)
    return snapshot


async def remux(source: Path, es: Path, out: Path, rate: Fraction, codec: str) -> None:
    info = probe(str(source))
    remuxer = StreamRemuxer(
        build_remux_command(
            source_url=str(source),
            info=info,
            selection=TrackSelection(),
            enhanced_codec=codec,
            output_rate=rate,
            container="mp4",
        )
    )
    await remuxer.start()

    async def pump() -> None:
        with open(out, "wb") as handle:
            async for block in remuxer.read_chunks():
                handle.write(block)

    reader = asyncio.create_task(pump())
    with open(es, "rb") as handle:
        while True:
            block = handle.read(256 * 1024)
            if not block:
                break
            await remuxer.feed(block)
    await remuxer.close_input()
    await reader
    await remuxer.wait()


def calibrate(args, source: Path, chain, request: dict, workdir: Path) -> RateTable:
    """Per-(stage, card) ms/frame from a short real run on each card.

    Sequential on purpose: a calibration run that shared the cards with
    another calibration run would measure contention, not capacity.
    """
    rows: list[StageRate] = []
    per_card: dict[str, dict] = {}
    for card in args.cards:
        chunk = ChunkSpec(
            index=0,
            card=card,
            start=0,
            stop=args.calibration_frames,
            pulls_successor=False,
            multiplier=args.fps_multiplier,
        )
        spool = workdir / f"cal-{card}"
        spool.mkdir(parents=True, exist_ok=True)
        report = asyncio.run(
            run_chunks(
                [chunk],
                source=source,
                request=request,
                out_es=workdir / f"cal-{card}.h264",
                job_id=f"cal{card}",
                spool=spool,
            )
        )
        stage_ms = report["chunks"][0]["stage_ms_per_frame"] if report["chunks"] else {}
        per_card[card] = {"wall_seconds": report["wall_seconds"], "stages": stage_ms}
        for spec in chain.stages:
            ms = stage_ms.get(spec.kind.value)
            if not ms or ms <= 0:
                # A stage the timer never saw cannot be priced. Refuse rather
                # than substitute a number: a zero cell hands that card an
                # unbounded share of the timeline.
                raise SystemExit(
                    f"calibration produced no rate for stage {spec.kind.value} on "
                    f"card {card}; measured stages were {sorted(stage_ms)}"
                )
            rows.append(StageRate(spec.kind, card, spec.in_res, float(ms)))
    print(json.dumps({"calibration": per_card}, indent=2), flush=True)
    return RateTable(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="#339 multi-card executor benchmark")
    parser.add_argument("--cards", default="1", help="comma-separated NVML indices")
    parser.add_argument("--baseline-card", default=None, help="default: the first card")
    parser.add_argument("--out", default="/tmp/mc")
    parser.add_argument("--source", default=None, help="existing clip; else generated")
    parser.add_argument("--source-resolution", default="960x540")
    parser.add_argument("--target", default="1920x1080")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--fps-multiplier", type=int, default=2)
    parser.add_argument("--calibration-frames", type=int, default=24)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--enable-sr", action="store_true", default=True)
    parser.add_argument("--no-sr", dest="enable_sr", action="store_false")
    parser.add_argument("--sr-scale", type=int, default=4)
    parser.add_argument("--sr-provider", default="cuda")
    parser.add_argument("--rife-version", default="4.6")
    parser.add_argument("--rife-scale", type=float, default=1.0)
    parser.add_argument("--video-codec", default="h264")
    parser.add_argument("--encode-backend", default="ffmpeg")
    # Near-transparent by default, and that is a correctness requirement,
    # not a quality preference. The PSNR gate below compares two independently
    # encoded arms; at a rate where the encoder is lossy, the two are distorted
    # differently and PSNR measures the rate controller rather than the stitch.
    # Measured on the synthetic clip at the ffmpeg h264_nvenc default rate:
    # baseline against a single un-concatenated chunk of the same frames scored
    # 6-25 dB with the pixels provably identical going in.
    parser.add_argument("--bitrate", type=int, default=150_000_000)
    parser.add_argument(
        "--frame-digests",
        action="store_true",
        default=True,
        help="hash every frame before the encoder; the exact seam gate",
    )
    parser.add_argument(
        "--no-frame-digests", dest="frame_digests", action="store_false"
    )
    parser.add_argument(
        "--same-card-control",
        action="store_true",
        default=True,
        help=(
            "also run the multi-card chunking with every chunk on the baseline "
            "card. Compared against the baseline it isolates the seam from the "
            "cards: same chunk boundaries, same architecture, so any difference "
            "is the stitch and nothing else."
        ),
    )
    parser.add_argument(
        "--no-same-card-control", dest="same_card_control", action="store_false"
    )
    parser.add_argument(
        "--schedule",
        choices=("weighted", "pull", "both"),
        default="both",
        help=(
            "weighted: the capacity-weighted plan, one chunk per card. "
            "pull: card-agnostic items taken from a shared queue. "
            "both: run each as its own arm against the same baseline, which is "
            "the only way the two are comparable"
        ),
    )
    parser.add_argument(
        "--chunks-per-card",
        type=int,
        default=4,
        help="pull queue length, as a multiple of the card count",
    )
    parser.add_argument(
        "--persistent-worker",
        action="store_true",
        default=True,
        help=(
            "keep one worker process per card and feed it items, instead of "
            "launching a process per item. Only meaningful under pull "
            "scheduling, where a process per item would pay the ~8 s torch and "
            "ONNX Runtime import once per queue item"
        ),
    )
    parser.add_argument(
        "--no-persistent-worker", dest="persistent_worker", action="store_false"
    )
    parser.add_argument("--ring-depth", type=int, default=2)
    parser.add_argument("--budget-mib", type=int, default=14000)
    parser.add_argument("--model-dir", default="/spinning/llm_stuff/k3-models")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args(argv)
    args.cards = [c.strip() for c in args.cards.split(",") if c.strip()]

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

    info = probe(str(source))
    source_rate = info.track(0).frame_rate() or Fraction(args.fps)
    total_frames = count_frames(source)
    out_rate = retimed_rate(source_rate, args.fps_multiplier)
    request = request_payload(args, source_rate)

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

    report: dict = {
        "args": vars(args),
        "cards": resolve_cards(args.cards),
        "source": {
            "path": str(source),
            "frames": total_frames,
            "rate": str(source_rate),
            "resolution": str(source_res),
        },
        "chain": [s.describe() for s in chain.stages],
        "expected_output_frames": expected_frame_count(
            total_frames, args.fps_multiplier
        ),
    }
    print(json.dumps({"setup": report}, indent=2), flush=True)

    rates = calibrate(args, source, chain, request, workdir)

    availability = [
        CardAvailability(card=c, reserved_bytes=args.budget_mib * MIB)
        for c in args.cards
    ]
    reservation = ReservationInputs(
        rife_measured_bytes_per_pair=int(1185.4 * MIB),
    )
    plan = capacity_weighted_plan(
        chain=chain,
        rates=rates,
        cards=availability,
        total_frames=total_frames,
        reservation=reservation,
    )
    chunks = chunk_specs_from_plan(plan, multiplier=args.fps_multiplier)
    verify_chunk_arithmetic(chunks, total_frames, args.fps_multiplier)
    report["plan"] = {
        "describe": plan.describe(),
        "chunks": [c.describe() for c in chunks],
        "predicted_makespan_ms": plan.predicted_makespan_ms,
        "predicted": predict_makespan(plan, rates).render(),
    }
    print(plan.describe(), flush=True)

    # -- baseline: the same executor, one chunk, one card ------------------
    baseline_out = workdir / "baseline.mp4"
    if not args.skip_baseline:
        baseline_card = args.baseline_card or args.cards[0]
        baseline_chunk = [
            ChunkSpec(
                index=0,
                card=baseline_card,
                start=0,
                stop=total_frames,
                pulls_successor=False,
                multiplier=args.fps_multiplier,
            )
        ]
        es = workdir / "baseline.h264"
        spool = workdir / "spool-baseline"
        spool.mkdir(parents=True, exist_ok=True)
        report["baseline"] = asyncio.run(
            run_chunks(
                baseline_chunk,
                source=source,
                request=request,
                out_es=es,
                job_id="baseline",
                spool=spool,
            )
        )
        asyncio.run(remux(source, es, baseline_out, out_rate, args.video_codec))
        report["baseline"]["output_frames"] = count_frames(baseline_out)
        report["baseline"]["card"] = baseline_card
        print(
            json.dumps({"baseline": without_digests(report["baseline"])}, indent=2),
            flush=True,
        )

    run_weighted = args.schedule in ("weighted", "both")
    run_pull = args.schedule in ("pull", "both")

    # -- multi-card, capacity-weighted --------------------------------------
    multicard_out = workdir / "multicard.mp4"
    if run_weighted:
        es = workdir / "multicard.h264"
        spool = workdir / "spool-multicard"
        spool.mkdir(parents=True, exist_ok=True)
        report["multicard"] = asyncio.run(
            run_chunks(
                chunks,
                source=source,
                request=request,
                out_es=es,
                job_id="multicard",
                spool=spool,
            )
        )
        asyncio.run(remux(source, es, multicard_out, out_rate, args.video_codec))
        report["multicard"]["output_frames"] = count_frames(multicard_out)

    # -- same-card control --------------------------------------------------
    # The identical chunking, every chunk on the baseline card. Against the
    # baseline this isolates the seam: same boundaries, same architecture, so
    # a difference can only be the stitch. Against the multi-card run it
    # isolates the cards: same boundaries, same GOP structure, so a difference
    # can only be per-architecture arithmetic.
    if run_weighted and args.same_card_control and not args.skip_baseline:
        control_chunks = [
            ChunkSpec(
                index=c.index,
                card=baseline_card,
                start=c.start,
                stop=c.stop,
                pulls_successor=c.pulls_successor,
                multiplier=c.multiplier,
            )
            for c in chunks
        ]
        es_ctl = workdir / "control.h264"
        spool_ctl = workdir / "spool-control"
        spool_ctl.mkdir(parents=True, exist_ok=True)
        report["same_card_control"] = asyncio.run(
            run_chunks(
                control_chunks,
                source=source,
                request=request,
                out_es=es_ctl,
                job_id="control",
                spool=spool_ctl,
            )
        )
        control_out = workdir / "control.mp4"
        asyncio.run(remux(source, es_ctl, control_out, out_rate, args.video_codec))
        report["same_card_control"]["output_frames"] = count_frames(control_out)

    # -- pull scheduling ----------------------------------------------------
    # The same clip, the same seam convention, and no plan: the timeline is cut
    # into more items than there are cards and every card takes the next one
    # when it is free. Two arms, because they answer two questions that a
    # single arm would conflate -- the multi-card one is a throughput number
    # against the same baseline, and the one-card one is the exact seam gate,
    # where "exact" means a sha256 per frame taken immediately before the
    # encoder and compared against the un-chunked run on that same card.
    pull_out = workdir / "pull.mp4"
    if run_pull:
        queue_len = pull_queue_size(
            total_frames, cards=len(args.cards), chunks_per_card=args.chunks_per_card
        )
        pull_items = pull_queue_chunks(
            total_frames,
            multiplier=args.fps_multiplier,
            has_rife=any(s.kind.value == "rife" for s in chain.stages),
            chunks=queue_len,
        )
        verify_chunk_arithmetic(pull_items, total_frames, args.fps_multiplier)
        report["pull_plan"] = {
            "items": len(pull_items),
            "chunks_per_card": args.chunks_per_card,
            "persistent_worker": args.persistent_worker,
            "describe": [c.describe() for c in pull_items],
        }

        es_pull = workdir / "pull.h264"
        spool_pull = workdir / "spool-pull"
        spool_pull.mkdir(parents=True, exist_ok=True)
        report["pull"] = asyncio.run(
            run_chunks(
                pull_items,
                source=source,
                request=request,
                out_es=es_pull,
                job_id="pull",
                spool=spool_pull,
                cards=args.cards,
                persistent=args.persistent_worker,
            )
        )
        asyncio.run(remux(source, es_pull, pull_out, out_rate, args.video_codec))
        report["pull"]["output_frames"] = count_frames(pull_out)
        print(
            json.dumps({"pull": without_digests(report["pull"])}, indent=2), flush=True
        )

        if args.same_card_control and not args.skip_baseline:
            es_pctl = workdir / "pull-control.h264"
            spool_pctl = workdir / "spool-pull-control"
            spool_pctl.mkdir(parents=True, exist_ok=True)
            report["pull_same_card_control"] = asyncio.run(
                run_chunks(
                    pull_items,
                    source=source,
                    request=request,
                    out_es=es_pctl,
                    job_id="pullctl",
                    spool=spool_pctl,
                    cards=[baseline_card],
                    persistent=args.persistent_worker,
                )
            )
            pull_control_out = workdir / "pull-control.mp4"
            asyncio.run(
                remux(source, es_pctl, pull_control_out, out_rate, args.video_codec)
            )
            report["pull_same_card_control"]["output_frames"] = count_frames(
                pull_control_out
            )

    # -- verdict ------------------------------------------------------------
    checks: dict = {}
    if run_weighted:
        checks["multicard_frame_count"] = {
            "expected": report["expected_output_frames"],
            "actual": report["multicard"]["output_frames"],
            "pass": report["multicard"]["output_frames"]
            == report["expected_output_frames"],
        }
    if run_pull:
        checks["pull_frame_count"] = {
            "expected": report["expected_output_frames"],
            "actual": report["pull"]["output_frames"],
            "pass": report["pull"]["output_frames"] == report["expected_output_frames"],
        }

    def seam_check(base_digests, control_digests, label):
        """The exact gate: chunked and un-chunked, same card, hashed pre-encode.

        It is a gate rather than a tolerance because the comparison is taken
        before the encoder. Two independently encoded H.264 streams of
        identical pixels are not the same stream, so an encoded-output PSNR
        here measures rate control; these digests measure the stitch.
        """
        first_diff = next(
            (
                i
                for i, (a, b) in enumerate(zip(base_digests, control_digests))
                if a != b
            ),
            None,
        )
        return {
            "compared": f"pre-encode frame digests, {label} vs whole clip, same card",
            "frames": len(base_digests),
            "chunked_frames": len(control_digests),
            "first_differing_frame": first_diff,
            "pass": len(base_digests) == len(control_digests) and first_diff is None,
        }

    if args.frame_digests and not args.skip_baseline:
        base_digests = report["baseline"].get("frame_digests", [])
        control_digests = report.get("same_card_control", {}).get("frame_digests", [])
        if control_digests:
            checks["seam_is_exact"] = seam_check(
                base_digests, control_digests, "capacity-weighted chunking"
            )
        pull_control_digests = report.get("pull_same_card_control", {}).get(
            "frame_digests", []
        )
        if pull_control_digests:
            # The re-proof the pull-queue change owes. A finer queue means more
            # seams -- one per item boundary instead of one per card -- so the
            # convention is under more load here than it ever was under the
            # weighted plan, and this is where that shows up if it is wrong.
            checks["pull_seam_is_exact"] = seam_check(
                base_digests, pull_control_digests, "pull queue"
            )
        for arm, key in (("multicard", "multicard"), ("pull", "pull")):
            digests = report.get(arm, {}).get("frame_digests", [])
            if digests and base_digests:
                same = sum(1 for a, b in zip(base_digests, digests) if a == b)
                # Not a pass/fail. Frames produced on a different architecture
                # are not expected to be bit-identical to the 5090's -- the
                # convolution kernels differ -- so this is recorded as a
                # measured fraction and the encoded-output PSNR grades it.
                checks[f"{key}_frames_bit_identical_to_baseline"] = {
                    "identical": same,
                    "of": len(digests),
                    "note": (
                        "cross-architecture convolution results are not "
                        "bit-identical; this is a measurement, not a gate"
                    ),
                }
    if not args.skip_baseline:
        checks["baseline_frame_count"] = {
            "expected": report["expected_output_frames"],
            "actual": report["baseline"]["output_frames"],
            "pass": report["baseline"]["output_frames"]
            == report["expected_output_frames"],
        }
        if run_weighted:
            # Informational, not a gate. It compares a three-card run against a
            # one-card run, so it carries two differences at once: the cards,
            # and the GOP structure -- chunk boundaries force an IDR, which the
            # baseline does not have there, and two encodes of identical pixels
            # with different GOP structure already differ. The gate below
            # removes the second of those.
            checks["multicard_vs_single_card_baseline"] = {
                "psnr_db": psnr_between(baseline_out, multicard_out),
                "note": (
                    "cards and GOP structure both differ; use "
                    "multicard_vs_same_card_control for the cards alone"
                ),
            }
            if args.same_card_control:
                checks["multicard_vs_same_card_control"] = cross_card_psnr(
                    psnr_between(workdir / "control.mp4", multicard_out),
                    "same chunking, 3 cards vs 1 card",
                )
        if run_pull and args.same_card_control:
            checks["pull_vs_same_card_control"] = cross_card_psnr(
                psnr_between(workdir / "pull-control.mp4", pull_out),
                "same queue, 3 cards vs 1 card",
            )

        base_wall = report["baseline"]["wall_seconds"]
        # Every arm against the one baseline, so the arms are comparable to
        # each other and not only to themselves.
        report["speedup"] = {
            "baseline_wall_s": base_wall,
            "cards": len(args.cards),
        }
        for arm in ("multicard", "pull"):
            wall = report.get(arm, {}).get("wall_seconds")
            if wall:
                report["speedup"][f"{arm}_wall_s"] = wall
                report["speedup"][f"{arm}_speedup_x"] = round(base_wall / wall, 3)
    report["checks"] = checks
    # The digest lists are long and their value is the comparison, which is
    # already in "checks". Keep the counts, drop the lists.
    for arm in (
        "baseline",
        "multicard",
        "same_card_control",
        "pull",
        "pull_same_card_control",
    ):
        if arm in report and isinstance(report[arm], dict):
            digests = report[arm].pop("frame_digests", None)
            if digests is not None:
                report[arm]["frame_digest_count"] = len(digests)
            for entry in report[arm].get("chunks", []):
                entry.pop("frame_digests", None)
    (workdir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"checks": checks, "speedup": report.get("speedup")}, indent=2))

    failed = [k for k, v in checks.items() if not v.get("pass")]
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
