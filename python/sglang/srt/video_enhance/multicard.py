# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Running one job across several cards: the executor for ``shard_plan``.

``shard_plan.py`` decides *what* each card should do -- contiguous chunks of
the source timeline, weighted by measured per-card chain rate, with an
explicit one-frame overlap at every seam so RIFE can interpolate across it.
This module is what carries that out, and it is the follow-on post #3 that
TASK_333 §7 registered as "the planner exists; the executor is single-card".

Three things have to be true for a multi-card run to be a correct run, and
each of them is arithmetic rather than hope:

1.  **The seam pair is interpolated exactly once.** A chunk pulls one frame
    past its own range so RIFE has both inputs for the pair that straddles
    the boundary, and it emits the interpolated frames -- but *not* that
    trailing frame itself, which belongs to the next chunk. Encoding it in
    both places would duplicate one source frame per seam;  encoding it in
    neither would drop one.
2.  **The frame counts add up to the single-card answer.** With ``N`` source
    frames, ``m`` as the interpolation multiplier and any chunking whatsoever,
    :func:`total_output_frames` returns ``N + (N-1)(m-1)`` -- the same
    ``expected_frame_count`` the muxer retimes against. A chunking that
    changed the output length would put audio out of sync by construction.
3.  **Chunks are emitted in timeline order.** Cards finish at different
    instants; the response is a single stream. Chunk ``k`` is forwarded only
    once every chunk before it has been forwarded.

**What multi-card costs, stated plainly.** The single-card chain has one
unbroken back-pressure path from the socket to the decoder (§8.4). Here, a
chunk that has finished while an earlier chunk is still streaming has to go
somewhere, and that somewhere is a spool file. Back-pressure therefore
reaches the workers late rather than immediately: a stalled client stops the
run after at most ``spool_chunks`` completed-but-unforwarded chunks, not
within one ring depth. The bound is explicit and configurable rather than
absent, and with the default of one spool slot per card it is the smallest
bound that still lets every card work at once.

**Why concatenating elementary streams is legal.** Each chunk is encoded
independently and begins with its own parameter sets and an IDR, which is
what makes an annex-B concatenation decodable: an IDR resets ``frame_num``
and the picture-order count, so the discontinuity at a seam is exactly the
discontinuity a decoder is required to handle. The muxer is fed the
concatenation with the output rate declared on the command line and no
container timestamps, so it regenerates one monotonic PTS ladder over the
whole stream -- which is also why a per-chunk timestamp error cannot survive
into the output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Sequence

from sglang.srt.video_enhance.chain import StageKind
from sglang.srt.video_enhance.mux import expected_frame_count
from sglang.srt.video_enhance.shard_plan import ShardPlan

logger = logging.getLogger(__name__)

ByteSink = Callable[[bytes], Awaitable[None]]

#: Read granularity when forwarding a spooled chunk to the muxer. Matches the
#: muxer's own read size so a forwarded chunk costs the same number of writes
#: as a directly muxed one.
SPOOL_READ_BYTES = 256 * 1024


class MultiCardError(RuntimeError):
    """A multi-card run that cannot be started, or that produced the wrong
    number of frames. Both are refusals, never silent corrections."""


# --------------------------------------------------------------------------
# Chunk specs: the plan, resolved into what one worker actually does
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkSpec:
    """One card's unit of work, with the seam convention already resolved.

    ``shard_plan`` prices both a lead and a tail overlap at every interior
    seam, because its cost model is symmetric and does not know which side of
    a seam will do the work. The executor has to choose, and it chooses the
    *tail*: chunk ``k`` pulls frame ``stop`` and interpolates the pair
    ``(stop-1, stop)``. The lead overlap is then not pulled at all, so the
    real cost of a seam is half what the planner charged -- the planner is
    pessimistic, which is the safe direction for an admission check.
    """

    index: int
    card: str
    #: First source frame this chunk owns and encodes.
    start: int
    #: One past the last source frame this chunk owns.
    stop: int
    #: True when a successor chunk exists, so frame ``stop`` must be pulled as
    #: RIFE's second input for the seam pair but must not be encoded here.
    pulls_successor: bool
    multiplier: int

    def __post_init__(self) -> None:
        if self.stop <= self.start:
            raise MultiCardError(
                f"chunk {self.index} on {self.card} owns no frames "
                f"([{self.start}:{self.stop}]); a card with nothing to do must be "
                "dropped by the planner, not handed an empty range"
            )
        if self.multiplier < 1:
            raise MultiCardError("multiplier must be at least 1")

    @property
    def owned_frames(self) -> int:
        return self.stop - self.start

    @property
    def pulled_frames(self) -> int:
        """Source frames this chunk decodes, including the seam frame."""
        return self.owned_frames + (1 if self.pulls_successor else 0)

    @property
    def successor_index(self) -> int | None:
        return self.stop if self.pulls_successor else None

    @property
    def output_frames(self) -> int:
        """Frames this chunk encodes.

        With ``n`` owned frames and a successor, the pulled range has ``n``
        adjacent pairs, so the chain emits ``n+1`` originals and ``n(m-1)``
        interpolated frames, of which the trailing original is withheld:
        ``n * m``. Without a successor the range has ``n-1`` pairs and nothing
        is withheld: ``n + (n-1)(m-1)``, which is ``expected_frame_count``.
        """
        if self.pulls_successor:
            return self.owned_frames * self.multiplier
        return expected_frame_count(self.owned_frames, self.multiplier)

    def encodes(self, frame_index: int, sub_index: int) -> bool:
        """The ``encode_filter`` predicate, as a pure function.

        Only the *original* seam frame is withheld. The interpolated frames
        that carry ``frame_index == successor_index`` do not exist -- an
        interpolated frame is keyed to the earlier frame of its pair -- but
        the check is written against ``sub_index`` anyway so a future arity
        change cannot silently start dropping invented frames.
        """
        successor = self.successor_index
        if successor is None:
            return True
        return not (frame_index == successor and sub_index == 0)

    def describe(self) -> str:
        seam = f" +1 seam frame ({self.stop})" if self.pulls_successor else ""
        return (
            f"chunk {self.index} on {self.card}: source [{self.start}:{self.stop}]"
            f"{seam} -> {self.output_frames} encoded frames"
        )

    def as_dict(self) -> dict:
        return asdict(self)


def chunk_specs_from_plan(plan: ShardPlan, *, multiplier: int) -> tuple[ChunkSpec, ...]:
    """Resolve a :class:`ShardPlan` into executable chunks.

    Only contiguous plans are executable. The modulo baseline in
    ``shard_plan`` exists to be priced, not to be run -- under a stride the
    successor of every owned frame lives on another card, so there is no
    "the seam frame" to pull, and the executor says so instead of producing
    something that decodes but is wrong.
    """
    strided = [a.card for a in plan.assignments if a.stride != 1]
    if strided:
        raise MultiCardError(
            f"plan strategy {plan.strategy.value} assigns strided ranges to "
            f"{strided}; the executor runs contiguous chunks only. A stride puts "
            "the successor of every owned frame on another card, which is the "
            "reason shard_plan keeps the modulo shape as a costed baseline "
            "rather than a runnable plan."
        )
    ordered = sorted(plan.assignments, key=lambda a: a.start)
    covered = 0
    for assignment in ordered:
        if assignment.start != covered:
            raise MultiCardError(
                f"chunks do not tile the timeline: expected the next chunk to "
                f"start at {covered}, {assignment.card} starts at "
                f"{assignment.start}"
            )
        covered = assignment.stop
    if covered != plan.total_frames:
        raise MultiCardError(
            f"chunks cover {covered} of {plan.total_frames} frames; a partial "
            "cover would silently shorten the output"
        )

    has_rife = StageKind.RIFE in plan.chain.kinds
    return tuple(
        ChunkSpec(
            index=i,
            card=assignment.card,
            start=assignment.start,
            stop=assignment.stop,
            # Without RIFE nothing reads across frames, so no seam frame is
            # needed and every chunk is independent.
            pulls_successor=has_rife and i < len(ordered) - 1,
            multiplier=multiplier,
        )
        for i, assignment in enumerate(ordered)
    )


def total_output_frames(chunks: Sequence[ChunkSpec]) -> int:
    return sum(c.output_frames for c in chunks)


def verify_chunk_arithmetic(
    chunks: Sequence[ChunkSpec], total_frames: int, multiplier: int
) -> None:
    """Refuse a chunking whose frames do not add up to the single-card answer.

    This runs before any card is touched. A chunking that produces one frame
    too many or too few does not fail visibly -- it produces a video whose
    audio is progressively out of sync -- so it has to be impossible rather
    than unlikely.
    """
    expected = expected_frame_count(total_frames, multiplier)
    produced = total_output_frames(chunks)
    if produced != expected:
        raise MultiCardError(
            f"chunking produces {produced} output frames but {total_frames} "
            f"source frames at multiplier {multiplier} must produce {expected}. "
            f"Chunks: {[c.describe() for c in chunks]}"
        )
    owned = sum(c.owned_frames for c in chunks)
    if owned != total_frames:
        raise MultiCardError(
            f"chunks own {owned} source frames, the source has {total_frames}"
        )


# --------------------------------------------------------------------------
# Running a chunk
# --------------------------------------------------------------------------


@dataclass
class ChunkResult:
    """What one worker reports back."""

    index: int
    card: str
    #: Spool file holding this chunk's encoded elementary stream.
    path: Path
    frames_encoded: int
    frames_skipped: int
    bytes_out: int
    wall_seconds: float
    stage_ms_per_frame: dict[str, float] = field(default_factory=dict)
    #: Per-frame digests taken immediately before the encoder, when the run
    #: asked for them. The encoder-independent way to compare two chunkings.
    frame_digests: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "card": self.card,
            "frames_encoded": self.frames_encoded,
            "frames_skipped": self.frames_skipped,
            "bytes_out": self.bytes_out,
            "wall_seconds": round(self.wall_seconds, 3),
            "stage_ms_per_frame": self.stage_ms_per_frame,
            "error": self.error,
        }


class ChunkRunner:
    """How a chunk is turned into an elementary stream on a spool file.

    The protocol exists so the scheduling, ordering and arithmetic above can
    be tested with no card and no subprocess, while the real runner keeps the
    one property the design insists on: a worker is a *process* pinned to one
    physical GPU through ``CUDA_VISIBLE_DEVICES``, not a thread sharing a CUDA
    context with its neighbours.
    """

    async def run(self, chunk: ChunkSpec, spool: Path) -> ChunkResult:
        raise NotImplementedError


class SubprocessChunkRunner(ChunkRunner):
    """One worker process per chunk, pinned to that chunk's physical GPU.

    ``card`` is an NVML index. It is placed in the child's
    ``CUDA_VISIBLE_DEVICES`` before the child starts, so inside the child
    ``cuda:0`` is the only device there is -- the same process-level isolation
    Class 1 uses for ``--rank-gpu-id``, and for the same reason: an in-process
    logical-to-physical mapping table is a bug generator, a one-device process
    is not.
    """

    def __init__(
        self,
        *,
        source_url: str,
        request: dict,
        python: str | None = None,
        env: dict[str, str] | None = None,
        module: str = "sglang.srt.video_enhance.chunk_worker",
    ) -> None:
        self.source_url = source_url
        self.request = dict(request)
        self.python = python or sys.executable
        self.env = dict(env or {})
        self.module = module

    def _child_env(self, chunk: ChunkSpec) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.env)
        # CUDA_DEVICE_ORDER first, and it is not optional. CUDA's default
        # ordering is FASTEST_FIRST, which is not NVML's -- and NVML's is what
        # the planner, the reservation, nvidia-smi and every card window in
        # the runbook are expressed in. On this rig the 5090 is NVML index 1
        # and CUDA ordinal 0, so a plan that hands "card 1" the largest chunk
        # because it measured that card as the fastest would put that chunk on
        # a 3080 and put the 5090's chunk on a 3080 as well.
        #
        # It does not fail: every card runs, every frame is produced, the seam
        # is exact and the output is correct. It just measures and schedules
        # the wrong cards. Caught here by a per-stage rate table in which the
        # "3080" at NVML index 0 ran super-resolution at 36 ms/frame and the
        # "5090" at NVML index 1 ran it at 93.
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(chunk.card)
        return env

    async def run(self, chunk: ChunkSpec, spool: Path) -> ChunkResult:
        payload = {
            "source_url": self.source_url,
            "output_path": str(spool),
            "chunk": chunk.as_dict(),
            "request": self.request,
        }
        started = time.perf_counter()
        process = await asyncio.create_subprocess_exec(
            self.python,
            "-m",
            self.module,
            "--spec",
            json.dumps(payload),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._child_env(chunk),
        )
        stdout, stderr = await process.communicate()
        elapsed = time.perf_counter() - started

        if process.returncode != 0:
            return ChunkResult(
                index=chunk.index,
                card=chunk.card,
                path=spool,
                frames_encoded=0,
                frames_skipped=0,
                bytes_out=0,
                wall_seconds=elapsed,
                error=(
                    f"chunk worker on card {chunk.card} exited "
                    f"{process.returncode}: {stderr.decode(errors='replace')[-2000:]}"
                ),
            )
        try:
            report = json.loads(stdout.decode().strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            return ChunkResult(
                index=chunk.index,
                card=chunk.card,
                path=spool,
                frames_encoded=0,
                frames_skipped=0,
                bytes_out=0,
                wall_seconds=elapsed,
                error=f"chunk worker produced no parseable report: {exc}",
            )
        return ChunkResult(
            index=chunk.index,
            card=chunk.card,
            path=spool,
            frames_encoded=report.get("frames_encoded", 0),
            frames_skipped=report.get("frames_skipped", 0),
            bytes_out=report.get("bytes_out", 0),
            wall_seconds=elapsed,
            stage_ms_per_frame=report.get("stage_ms_per_frame", {}),
            frame_digests=tuple(report.get("frame_digests", ())),
        )


# --------------------------------------------------------------------------
# The executor
# --------------------------------------------------------------------------


@dataclass
class MultiCardStats:
    job_id: str
    chunks: list[dict] = field(default_factory=list)
    #: Every chunk's pre-encode frame digests, concatenated in timeline order.
    #: Two chunkings of the same clip on the same card must produce the same
    #: sequence; that equality is the exact statement of a correct stitch.
    frame_digests: list[str] = field(default_factory=list)
    bytes_out: int = 0
    frames_encoded: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    state: str = "pending"
    error: str | None = None

    @property
    def wall_seconds(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    def snapshot(self) -> dict:
        per_card: dict[str, float] = {}
        for chunk in self.chunks:
            per_card[chunk["card"]] = per_card.get(chunk["card"], 0.0) + chunk.get(
                "wall_seconds", 0.0
            )
        return {
            "id": self.job_id,
            "state": self.state,
            "cards": sorted(per_card),
            "chunks": self.chunks,
            "bytes_out": self.bytes_out,
            "frames_encoded": self.frames_encoded,
            "wall_seconds": round(self.wall_seconds, 3),
            # The makespan is the slowest card; the sum is the work done. Both
            # are reported because the ratio is what says whether the
            # capacity weighting actually balanced the cards.
            "busiest_card_seconds": round(max(per_card.values(), default=0.0), 3),
            "card_seconds": {k: round(v, 3) for k, v in per_card.items()},
            "frame_digests": list(self.frame_digests),
            "error": self.error,
        }


class MultiCardExecutor:
    """Runs the chunks of one plan concurrently and stitches them in order."""

    def __init__(
        self,
        *,
        job_id: str,
        chunks: Sequence[ChunkSpec],
        runner: ChunkRunner,
        sink: ByteSink,
        spool_dir: str | Path | None = None,
        spool_chunks: int | None = None,
        total_frames: int | None = None,
        multiplier: int = 1,
    ) -> None:
        if not chunks:
            raise MultiCardError("no chunks to run")
        self.job_id = job_id
        self.chunks = tuple(chunks)
        self.runner = runner
        self.sink = sink
        self.stats = MultiCardStats(job_id=job_id)
        self._cancelled = asyncio.Event()

        if total_frames is not None:
            verify_chunk_arithmetic(self.chunks, total_frames, multiplier)

        cards = {c.card for c in self.chunks}
        # One completed-but-unforwarded chunk per card is the smallest bound
        # that still lets every card work while chunk 0 is streaming out.
        self._slots = asyncio.Semaphore(
            max(1, spool_chunks if spool_chunks is not None else len(cards))
        )
        self._owns_spool = spool_dir is None
        self._spool_dir = Path(
            spool_dir or tempfile.mkdtemp(prefix=f"k3-multicard-{job_id}-")
        )
        self._spool_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def _spool_path(self, chunk: ChunkSpec) -> Path:
        return self._spool_dir / f"chunk{chunk.index:05d}.es"

    async def run(self) -> MultiCardStats:
        self.stats.state = "running"
        # One event per chunk, set when that chunk's spool file is complete.
        done: list[asyncio.Event] = [asyncio.Event() for _ in self.chunks]
        results: list[ChunkResult | None] = [None] * len(self.chunks)
        # One event per chunk, set once it has been forwarded and its spool
        # slot released. Chunk k's worker waits for chunk k-slots to be
        # forwarded before it starts, which is what bounds the spool.
        forwarded: list[asyncio.Event] = [asyncio.Event() for _ in self.chunks]

        async def work(chunk: ChunkSpec) -> None:
            await self._slots.acquire()
            try:
                if self.cancelled:
                    return
                result = await self.runner.run(chunk, self._spool_path(chunk))
                results[chunk.index] = result
            finally:
                done[chunk.index].set()

        workers = [asyncio.create_task(work(chunk)) for chunk in self.chunks]
        try:
            for chunk in self.chunks:
                await done[chunk.index].wait()
                result = results[chunk.index]
                if result is None:
                    if self.cancelled:
                        break
                    raise MultiCardError(
                        f"chunk {chunk.index} on card {chunk.card} produced no result"
                    )
                if not result.ok:
                    raise MultiCardError(result.error or "chunk failed")
                if result.frames_encoded != chunk.output_frames:
                    raise MultiCardError(
                        f"chunk {chunk.index} on card {chunk.card} encoded "
                        f"{result.frames_encoded} frames, the seam arithmetic says "
                        f"{chunk.output_frames}. A chunk that is short or long by "
                        "one frame desynchronises the whole output, so the run is "
                        "refused rather than stitched."
                    )
                await self._forward(result)
                self.stats.chunks.append(result.as_dict())
                self.stats.frame_digests.extend(result.frame_digests)
                self.stats.frames_encoded += result.frames_encoded
                forwarded[chunk.index].set()
                self._slots.release()
            self.stats.state = "cancelled" if self.cancelled else "done"
        except Exception as exc:  # noqa: BLE001 - surfaced through the stats
            self.stats.state = "failed"
            self.stats.error = f"{type(exc).__name__}: {exc}"
            self.cancel()
            for task in workers:
                task.cancel()
            raise
        finally:
            await asyncio.gather(*workers, return_exceptions=True)
            self.stats.finished_at = time.time()
            self._cleanup()
        return self.stats

    async def _forward(self, result: ChunkResult) -> None:
        """Stream one completed chunk to the sink, in timeline order.

        Awaiting the sink here is what remains of the back-pressure chain: a
        stalled client stops the forwarding loop, the spool slots are not
        released, and the workers stop when they run out of slots.
        """
        with open(result.path, "rb") as handle:
            while True:
                block = handle.read(SPOOL_READ_BYTES)
                if not block:
                    break
                self.stats.bytes_out += len(block)
                await self.sink(block)
        result.path.unlink(missing_ok=True)

    def _cleanup(self) -> None:
        for chunk in self.chunks:
            self._spool_path(chunk).unlink(missing_ok=True)
        if self._owns_spool:
            shutil.rmtree(self._spool_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Card identity
# --------------------------------------------------------------------------


def resolve_cards(names: Iterable[str]) -> dict[str, str]:
    """Map NVML index to device name, for a run record that can be read later.

    Physical indices move between boots and driver states, so a measurement
    that records "card 1" and nothing else cannot be compared against a later
    one. Recording the name next to the index is the cheapest thing that makes
    the record self-describing; the engine cache keys on the UUID, which is
    the stronger identity where one is needed.
    """
    out: dict[str, str] = {}
    for name in names:
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                    "-i",
                    str(name),
                ],
                capture_output=True,
                timeout=30,
                check=True,
            )
            out[str(name)] = probe.stdout.decode().strip()
        except (OSError, subprocess.SubprocessError):
            out[str(name)] = "unknown"
    return out


__all__ = [
    "ChunkResult",
    "ChunkRunner",
    "ChunkSpec",
    "MultiCardError",
    "MultiCardExecutor",
    "MultiCardStats",
    "SubprocessChunkRunner",
    "chunk_specs_from_plan",
    "resolve_cards",
    "total_output_frames",
    "verify_chunk_arithmetic",
]
