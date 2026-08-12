# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""One chunk of a multi-card job, in its own process on its own card.

Launched by :class:`~sglang.srt.video_enhance.multicard.SubprocessChunkRunner`
with ``CUDA_VISIBLE_DEVICES`` already set to exactly one physical GPU, so
inside this process ``cuda:0`` is the only device there is. That is the same
process-level isolation Class 1 uses for ``--rank-gpu-id``: two chunks on one
card are two processes with identical ``CUDA_VISIBLE_DEVICES``, never one
process juggling a logical-to-physical map.

Two modes, one body of work:

*   ``--spec`` runs exactly one item and exits. The child writes the chunk's
    encoded elementary stream to the spool path it was given and prints one
    JSON line on stdout; the parent parses the last line.
*   ``--serve`` reads items from stdin until EOF and reports each on a
    prefixed line, keeping the chain between them. This is the pull-scheduling
    worker, and keeping the chain is the point: the ~8 s of torch and ONNX
    Runtime import, the CUDA context, the SR session and the RIFE weights are
    paid once per card instead of once per item, which is what makes a queue
    finer than one item per card worth having.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

from sglang.srt.video_enhance.asset_root import default_model_root
from sglang.srt.video_enhance.chain import ChainRequest, StageKind, build_chain
from sglang.srt.video_enhance.frame_math import Resolution
from sglang.srt.video_enhance.multicard import REPORT_PREFIX, ChunkSpec
from sglang.srt.video_enhance.mux import retimed_rate
from sglang.srt.video_enhance.pipeline import PipelineExecutor
from sglang.srt.video_enhance.ring import OverloadPolicy


async def _as_async(decode):
    """The synchronous pull decoder as the executor's async source.

    One frame per await and never running ahead, so back-pressure still stops
    at the decoder inside a shard exactly as it does on the single-card path.
    """
    for frame in decode:
        yield frame
        await asyncio.sleep(0)


#: Stages that depend on the chunk and must be rebuilt for every item.
#:
#: ``DECODE`` is seeked to the item's first frame and limited to its pulled
#: range. ``ENCODE`` has to be a fresh session per item because each item's
#: elementary stream is concatenated with its neighbours', which is only
#: decodable if every item begins with its own parameter sets and an IDR --
#: the property the executor's docstring relies on. Everything else in the
#: chain is a pure function of the request, so a serving worker builds it once
#: and keeps it.
PER_CHUNK_STAGES = frozenset({StageKind.DECODE, StageKind.ENCODE})


def build_chunk_stages(
    chunk: ChunkSpec,
    request: dict,
    source_url: str,
    shared: dict | None = None,
) -> tuple:
    """Instantiate the chain for one chunk on ``cuda:0``.

    Kept separate from ``tenant.build_stages`` because a chunk's decode stage
    is the one thing that differs: it is seeked to the chunk's first frame and
    limited to the chunk's pulled range, and everything downstream is the same
    chain the single-card path builds.

    ``shared`` is the serving worker's stage cache. When given, the stages
    outside :data:`PER_CHUNK_STAGES` are built on the first item and reused on
    every later one; that is where the ONNX Runtime session build and the RIFE
    weight load stop being paid per item. It is safe because those stages hold
    no cross-frame state: the sliding pair RIFE consumes lives in the
    executor's ``_ArityWindow``, which is constructed fresh per ``run()``, so
    no frame of item ``k`` can reach item ``k+1``.
    """
    from sglang.srt.video_enhance import codec
    from sglang.srt.video_enhance.resize import ResizeStage
    from sglang.srt.video_enhance.rife import RifeStage, download_weights
    from sglang.srt.video_enhance.sr import SuperResolutionStage

    source_res = Resolution.parse(request["source_resolution"])
    chain_request = ChainRequest(
        source=source_res,
        target=Resolution.parse(request["target"]),
        fps_multiplier=chunk.multiplier,
        dtype=request.get("dtype", "fp16"),
        enable_sr=request.get("enable_sr", True),
        sr_scale=request.get("sr_scale", 4),
        enable_resize=request.get("enable_resize", True),
        rife_scale=request.get("rife_scale", 1.0),
        rife_version=request.get("rife_version", "4.6"),
        streams_in_flight=request.get("streams_in_flight", 1),
    )
    chain = build_chain(chain_request)
    # #251: the request still wins; only the fallback moved behind
    # SGLANG_VIDEO_MODEL_ROOT (unset -> the previous literal).
    model_dir = Path(request.get("model_dir") or default_model_root())
    out_fps = retimed_rate(Fraction(request["source_frame_rate"]), chunk.multiplier)

    decode = codec.DecodeStage(
        source=source_url,
        resolution=source_res,
        device_id=0,
        backend=request.get("decode_backend", "ffmpeg"),
        start_frame=chunk.start,
        frame_limit=chunk.pulled_frames,
    )
    stages: dict = {}
    for spec in chain.stages:
        if shared is not None and spec.kind in shared:
            stages[spec.kind] = shared[spec.kind]
            continue
        if spec.kind is StageKind.DECODE:
            stages[spec.kind] = decode
        elif spec.kind is StageKind.COLOR_TO_RGB:
            stages[spec.kind] = codec.ColorToRgbStage(dtype=chain_request.dtype)
        elif spec.kind is StageKind.SR:
            stages[spec.kind] = SuperResolutionStage.build(
                source=spec.in_res,
                fmt=spec.in_format,
                model_dir=model_dir / "sr",
                provider=request.get("sr_provider", "cuda"),
                precision=(
                    "fp32"
                    if request.get("sr_provider", "cuda") == "cuda"
                    else chain_request.dtype
                ),
                device_id=0,
            )
        elif spec.kind is StageKind.RESIZE:
            stages[spec.kind] = ResizeStage(spec.in_res, spec.out_res, spec.in_format)
        elif spec.kind is StageKind.RIFE:
            stages[spec.kind] = RifeStage(
                resolution=spec.in_res,
                version=chain_request.rife_version,
                multiplier=spec.arity_out + 1,
                scale=chain_request.rife_scale,
                dtype=chain_request.dtype,
                device="cuda",
                weights_path=download_weights(
                    chain_request.rife_version, model_dir / "rife"
                ),
            )
        elif spec.kind is StageKind.COLOR_TO_YUV:
            stages[spec.kind] = codec.ColorToYuvStage(dtype=chain_request.dtype)
        elif spec.kind is StageKind.ENCODE:
            stages[spec.kind] = codec.EncodeStage(
                spec.in_res,
                fps=out_fps,
                codec=request.get("video_codec", "h264"),
                device_id=0,
                backend=request.get("encode_backend", "auto"),
                bitrate=request.get("bitrate"),
            )
    if shared is not None:
        for kind, stage in stages.items():
            if kind not in PER_CHUNK_STAGES:
                shared[kind] = stage
    return chain, stages, decode


async def run_chunk(payload: dict, shared: dict | None = None) -> dict:
    chunk = ChunkSpec(**payload["chunk"])
    request = payload["request"]
    output_path = Path(payload["output_path"])
    chain, stages, decode = build_chunk_stages(
        chunk, request, payload["source_url"], shared
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(output_path, "wb")
    written = {"bytes": 0}

    async def sink(chunk_bytes: bytes) -> None:
        handle.write(chunk_bytes)
        written["bytes"] += len(chunk_bytes)

    # Optional verification tap. Hashing the frame the instant before it
    # reaches the encoder is the only comparison of two chunkings that the
    # encoder cannot confound: two independently encoded H.264 streams of
    # identical pixels are not identical streams, and their PSNR is a
    # statement about rate control, not about the seam. Off by default -- it
    # costs a device-to-host copy per frame.
    digests: list[str] = []
    want_digests = bool(request.get("frame_digests"))

    def encode_filter(frame) -> bool:
        keep = chunk.encodes(frame.index, frame.sub_index)
        if keep and want_digests:
            payload = frame.data.detach().to("cpu").contiguous().numpy().tobytes()
            digests.append(hashlib.sha256(payload).hexdigest()[:16])
        return keep

    executor = PipelineExecutor(
        job_id=f"chunk{chunk.index}",
        chain=chain,
        stages=stages,
        source=_as_async(decode),
        sink=sink,
        ring_depth=max(1, request.get("ring_depth", 2)),
        policy=OverloadPolicy.STALL,
        encode_filter=encode_filter,
    )

    started = time.perf_counter()
    try:
        stats = await executor.run()
    finally:
        # Only the stages this item owns. Closing a shared stage would tear
        # down the session the next item is about to reuse, and the failure
        # would arrive one item later than its cause.
        for kind, stage in stages.items():
            if shared is not None and kind not in PER_CHUNK_STAGES:
                continue
            close = getattr(stage, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - one bad stage must not mask
                    continue
        handle.close()
    elapsed = time.perf_counter() - started

    return {
        "index": chunk.index,
        "card": chunk.card,
        "frames_decoded": stats.frames_decoded,
        "frames_encoded": stats.frames_encoded,
        "frames_skipped": stats.frames_skipped,
        "bytes_out": written["bytes"],
        "wall_seconds": round(elapsed, 3),
        "stage_ms_per_frame": stats.stage_ms_per_frame,
        "expected_output_frames": chunk.output_frames,
        "frame_digests": digests,
    }


async def serve(source_url: str, request: dict, reports) -> int:
    """Take items from stdin until EOF, one report line each.

    The pull-scheduling worker. The process, the CUDA context, the ONNX
    Runtime session and the RIFE weights are built on the first item and kept
    for every later one, which is the whole reason the queue can be finer than
    one item per card.

    Reads and runs strictly one item at a time. That is not a limitation being
    accepted, it is the contract: a card is one device and a second concurrent
    item on it would contend for exactly the memory the reservation sized for
    one. The parent's :class:`~sglang.srt.video_enhance.multicard.PersistentChunkRunner`
    holds a per-card lock that makes the same statement from its side.
    """
    loop = asyncio.get_running_loop()
    shared: dict = {}
    items = 0
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        payload.setdefault("source_url", source_url)
        payload.setdefault("request", request)
        try:
            report = await run_chunk(payload, shared)
        except Exception as exc:  # noqa: BLE001 - reported, not raised at the pipe
            # A failed item must come back as a report. Dying here would leave
            # the parent reading a pipe that will never produce a line, which
            # is the same symptom as a very slow card and a much worse one to
            # diagnose.
            import traceback

            report = {
                "index": payload.get("chunk", {}).get("index"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
        items += 1
        reports.write(REPORT_PREFIX + json.dumps(report) + "\n")
        reports.flush()
    # Shared stages outlive the items by design; they are closed here, at the
    # one moment there is provably no next item.
    for stage in shared.values():
        close = getattr(stage, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - one bad stage must not mask
                continue
    print(f"worker finished after {items} item(s)", file=sys.stderr, flush=True)
    return 0


def _report_stream():
    """Take the real stdout for reports and point everything else at stderr.

    The one-shot protocol got away with "the parent parses the last line",
    because there was exactly one report and it came last. A serving worker
    emits a report per item interleaved with whatever torch, ONNX Runtime and
    ffmpeg decide to print, so the report channel has to be a channel and not
    a convention. Duplicating fd 1 and then pointing fd 1 at fd 2 gives the
    parent a stdout pipe that carries reports and nothing else, without asking
    every library in the process to be quiet.
    """
    import os

    report_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(report_fd, "w")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="#333 M2 multi-card chunk worker")
    parser.add_argument("--spec", help="JSON job description for a single item")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="read items from stdin until EOF, reusing the chain between them",
    )
    parser.add_argument("--source-url", help="serving mode: the clip, once")
    parser.add_argument("--request", help="serving mode: the request JSON, once")
    args = parser.parse_args(argv)

    if args.serve:
        if not args.source_url or not args.request:
            parser.error("--serve needs --source-url and --request")
        reports = _report_stream()
        try:
            return asyncio.run(
                serve(args.source_url, json.loads(args.request), reports)
            )
        finally:
            reports.close()

    if not args.spec:
        parser.error("one of --spec or --serve is required")
    payload = json.loads(args.spec)
    report = asyncio.run(run_chunk(payload))
    # One JSON line, last line of stdout. The parent reads exactly this.
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
