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

None of the three mentions which card runs which chunk, and that is what
makes the second scheduling mode possible. Under **pull scheduling** the
timeline is cut into more items than there are cards, the items name no card,
and each card takes the next one whenever it is free
(:func:`pull_queue_chunks`, :class:`WorkQueue`). The balance is then an
outcome rather than a prediction: a card that is twice as fast comes back
twice as often, with no rate table, no calibration run, and no exposure to a
rate that changed after the plan was made. The pre-weighted mode remains, and
both run through the same executor and the same gates.

Fine-grained items are only affordable because the worker is not restarted
for each one -- :class:`PersistentChunkRunner` keeps one process per card and
feeds it items over a pipe, which is also what removes the per-chunk import
cost that #339 measured as the gap between the 1.64x compute-only speedup and
the 1.44x end-to-end one.

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
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Mapping, Sequence

from sglang.srt.video_enhance.chain import StageKind
from sglang.srt.video_enhance.mux import expected_frame_count
from sglang.srt.video_enhance.shard_plan import ShardPlan

logger = logging.getLogger(__name__)

ByteSink = Callable[[bytes], Awaitable[None]]

#: Read granularity when forwarding a spooled chunk to the muxer. Matches the
#: muxer's own read size so a forwarded chunk costs the same number of writes
#: as a directly muxed one.
SPOOL_READ_BYTES = 256 * 1024

#: The card of a work item that no card has taken yet. A pull-scheduled item
#: is card-agnostic by construction -- which card runs it is decided when a
#: card becomes free, not when the plan is built.
UNASSIGNED_CARD = "-"

#: Default number of queue items per card under pull scheduling. One item per
#: card is not a queue: the last item a card takes runs alone, so the tail of
#: the job is as long as the slowest single item. More items make that tail
#: shorter and the balance finer, and the only thing pushing back is the fixed
#: per-item cost, which :class:`PersistentChunkRunner` is what removes.
DEFAULT_CHUNKS_PER_CARD = 4

#: A chunk shorter than this is not worth being an item: with RIFE in the
#: chain it pulls a seam frame it then does not encode, so its overhead
#: fraction is ``1/len``, and it pays a decode seek either way.
MIN_PULL_CHUNK_FRAMES = 8


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

    @property
    def is_assigned(self) -> bool:
        return self.card != UNASSIGNED_CARD

    def assigned_to(self, card: str) -> "ChunkSpec":
        """The same work item, placed on a card.

        Late binding is the whole of pull scheduling, and it is safe because
        nothing that makes a chunk *correct* reads the card. ``start``,
        ``stop``, ``pulls_successor``, ``output_frames`` and :meth:`encodes`
        are functions of the timeline split alone, so the seam convention --
        chunk ``k`` interpolates the pair that straddles its boundary and
        withholds the trailing original -- holds identically whichever card
        ends up taking the item, and holds when two adjacent items land on the
        same card or on different ones.
        """
        return replace(self, card=card)

    def describe(self) -> str:
        seam = f" +1 seam frame ({self.stop})" if self.pulls_successor else ""
        where = "unassigned" if not self.is_assigned else f"on {self.card}"
        return (
            f"chunk {self.index} {where}: source [{self.start}:{self.stop}]"
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


def pull_queue_size(
    total_frames: int,
    *,
    cards: int,
    chunks_per_card: int = DEFAULT_CHUNKS_PER_CARD,
    min_chunk_frames: int = MIN_PULL_CHUNK_FRAMES,
) -> int:
    """How many items to cut the timeline into for pull scheduling.

    The count is a trade with two ends and no measurement in the middle, so
    both ends are named rather than tuned. More items balance better -- the
    worst case is one item's runtime of imbalance at the tail, so halving the
    item size halves the tail -- and cost more, because each item pays a
    decode seek, an encoder session, and with RIFE in the chain one seam frame
    pulled through the pre-RIFE prefix and then not encoded.

    ``min_chunk_frames`` is the floor that keeps the second term from
    dominating, and a short clip therefore gets fewer items than
    ``cards * chunks_per_card`` rather than a queue of two-frame slivers.
    """
    if cards < 1:
        raise MultiCardError("pull scheduling needs at least one card")
    if chunks_per_card < 1:
        raise MultiCardError("chunks_per_card must be at least 1")
    wanted = cards * chunks_per_card
    affordable = max(1, total_frames // max(1, min_chunk_frames))
    return max(1, min(wanted, affordable))


def pull_queue_chunks(
    total_frames: int,
    *,
    multiplier: int,
    has_rife: bool,
    chunks: int,
) -> tuple[ChunkSpec, ...]:
    """Cut the timeline into ``chunks`` card-agnostic work items.

    This is the pull-scheduling counterpart to :func:`chunk_specs_from_plan`,
    and the difference between them is the whole point: that function asks the
    planner which card gets which frames and bakes the answer in, this one
    does not ask. The items are equal-length because there is nothing to
    weight them by -- under a pull queue the balancing is done by the cards
    themselves, at run time, by a fast card coming back for more work sooner.

    What that buys, stated as the properties rather than as a speedup:

    *   It needs no P1 rate table. A capacity-weighted plan is only as good as
        its measurement, and :func:`shard_plan.capacity_weighted_plan` refuses
        to run without one.
    *   It is correct under a rate the measurement could not have known: a
        thermal cap, an LLM co-tenant that arrives mid-job, a card that is
        slower on *this* clip's content than on the calibration clip. A
        pre-weighted plan commits to its split before the first frame and
        cannot revise it.
    *   Its worst-case imbalance is bounded by one item, not by the error in
        the rate estimate.

    The seam is untouched by any of it. ``pulls_successor`` is
    ``index < chunks - 1``, a fact about the timeline split, so
    :func:`verify_chunk_arithmetic` holds on the queue before a card is
    touched and holds whatever order the cards then take the items in.
    """
    if total_frames <= 0:
        raise MultiCardError(f"total_frames must be positive, got {total_frames}")
    if chunks < 1:
        raise MultiCardError(f"need at least one chunk, got {chunks}")
    if chunks > total_frames:
        raise MultiCardError(
            f"cannot cut {total_frames} frames into {chunks} chunks; an empty "
            "chunk is refused rather than silently dropped"
        )

    # Equal lengths, remainder spread over the earliest items, so no two items
    # differ by more than one frame and the boundaries tile the timeline
    # exactly. Cumulative arithmetic rather than repeated rounding: rounding
    # each boundary independently is how a chunking comes to cover 479 of 480
    # frames.
    boundaries = [round(total_frames * i / chunks) for i in range(chunks + 1)]
    return tuple(
        ChunkSpec(
            index=i,
            card=UNASSIGNED_CARD,
            start=boundaries[i],
            stop=boundaries[i + 1],
            pulls_successor=has_rife and i < chunks - 1,
            multiplier=multiplier,
        )
        for i in range(chunks)
    )


class WorkQueue:
    """The shared list every card pulls its next item from.

    A single cursor over an ordered list, and that is the entire mechanism.
    No lock is needed because :meth:`take` contains no ``await``: an asyncio
    task cannot be preempted between reading the cursor and advancing it. The
    invariant that matters for the executor's deadlock argument is that items
    leave the queue in index order, which a single monotonic cursor gives for
    free.
    """

    def __init__(self, items: Sequence[ChunkSpec]) -> None:
        self._items = tuple(items)
        self._cursor = 0
        #: Which card took which item, in the order they were taken. The
        #: measured record of how the queue actually balanced, as opposed to
        #: how a plan predicted it would.
        self.pull_order: list[tuple[int, str]] = []

    def __len__(self) -> int:
        return len(self._items)

    @property
    def remaining(self) -> int:
        return len(self._items) - self._cursor

    def take(self, card: str) -> ChunkSpec | None:
        """The next unclaimed item, placed on ``card``. ``None`` when empty."""
        if self._cursor >= len(self._items):
            return None
        item = self._items[self._cursor]
        self._cursor += 1
        self.pull_order.append((item.index, card))
        return item.assigned_to(card)

    def items_per_card(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _index, card in self.pull_order:
            counts[card] = counts.get(card, 0) + 1
        return counts


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


def pinned_child_env(
    card: str, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The environment of a worker pinned to one physical card.

    ``CUDA_DEVICE_ORDER`` first, and it is not optional. CUDA's default
    ordering is FASTEST_FIRST, which is not NVML's -- and NVML's is what the
    planner, the reservation, nvidia-smi and every card window in the runbook
    are expressed in. On this rig the 5090 is NVML index 1 and CUDA ordinal 0,
    so a plan that hands "card 1" the largest chunk because it measured that
    card as the fastest would put that chunk on a 3080, and put the 5090's
    chunk on a 3080 as well.

    It does not fail: every card runs, every frame is produced, the seam is
    exact and the output is correct. It just measures and schedules the wrong
    cards. Caught by a per-stage rate table in which the "3080" at NVML index
    0 ran super-resolution at 36 ms/frame and the "5090" at NVML index 1 ran
    it at 93.

    Set after anything the caller passes in, because a caller that has
    inherited a wrong ``CUDA_DEVICE_ORDER`` from its own environment is
    exactly the case this defends against.
    """
    env = dict(os.environ)
    env.update(extra or {})
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(card)
    return env


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
        return pinned_child_env(chunk.card, self.env)

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


#: Every report line the serving worker emits starts with this. The parent
#: reads whole lines and matches on it, so a library that writes to the
#: worker's stdout cannot be mistaken for a report -- and the worker also
#: redirects its own stdout to stderr, so both ends of that hazard are shut.
REPORT_PREFIX = "@@CHUNK_REPORT@@ "

#: Stderr lines kept per worker for the error message when one dies.
STDERR_TAIL_LINES = 200


class PersistentChunkRunner(ChunkRunner):
    """One long-lived worker process per card, fed items over a pipe.

    :class:`SubprocessChunkRunner` launches a process per chunk, which was the
    right shape when a card ran exactly one chunk: the cost is paid once per
    card either way. Under pull scheduling it is the wrong shape, and
    measurably so -- #339 recorded roughly 8 s of torch and ONNX Runtime
    import before the first frame moves, and that is the whole reason the
    measured end-to-end speedup (1.44x) sat below the compute-only one
    (1.64x). A queue of four items per card would pay that eight seconds four
    times per card and hand back more than pull scheduling could win.

    So the worker is started once per card and kept. What it saves is the
    import, the CUDA context, the ONNX Runtime session build and the RIFE
    weight load; what it necessarily rebuilds per item is the decoder, which
    is seeked to a different frame, and the encoder, which must open its own
    session so that each chunk's elementary stream begins with its own
    parameter sets and an IDR. That division is the worker's, not this
    class's -- see ``chunk_worker.serve``.

    One item at a time per card, enforced by a lock rather than assumed: the
    protocol is one request line and one report line on a single pipe pair, so
    two concurrent items on one card would interleave into nonsense. The pull
    executor already runs one loop per card, so the lock never contends; it is
    there because a future caller cannot see that from here.
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
        self._workers: dict[str, asyncio.subprocess.Process] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._stderr: dict[str, deque[str]] = {}
        self._drains: list[asyncio.Task] = []
        #: Wall seconds to *spawn* each worker, per card -- which is about a
        #: millisecond and is deliberately not called startup. The cost this
        #: class exists to amortise is the import, the CUDA context and the
        #: session build, and all of that happens inside the child after
        #: ``create_subprocess_exec`` has already returned; it lands in the
        #: first item's ``wall_seconds``, where a run record can see it as the
        #: gap between that item and the card's later ones. Naming this field
        #: "startup" would invite reading ~1 ms as evidence the fixed cost is
        #: gone.
        self.spawn_seconds: dict[str, float] = {}

    async def _drain_stderr(self, card: str, stream: asyncio.StreamReader) -> None:
        """Keep the worker's stderr moving and keep its tail.

        Not optional. A worker whose stderr pipe fills stops writing, and a
        worker that has stopped writing stops working -- a deadlock that looks
        exactly like a slow card.
        """
        tail = self._stderr[card]
        while True:
            line = await stream.readline()
            if not line:
                return
            tail.append(line.decode(errors="replace").rstrip("\n"))

    async def _worker(self, card: str) -> asyncio.subprocess.Process:
        existing = self._workers.get(card)
        if existing is not None and existing.returncode is None:
            return existing
        started = time.perf_counter()
        process = await asyncio.create_subprocess_exec(
            self.python,
            "-m",
            self.module,
            "--serve",
            "--source-url",
            self.source_url,
            "--request",
            json.dumps(self.request),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=pinned_child_env(card, self.env),
        )
        self._workers[card] = process
        self._stderr[card] = deque(maxlen=STDERR_TAIL_LINES)
        assert process.stderr is not None
        self._drains.append(
            asyncio.create_task(self._drain_stderr(card, process.stderr))
        )
        self.spawn_seconds[card] = round(time.perf_counter() - started, 3)
        return process

    def _tail(self, card: str) -> str:
        return "\n".join(self._stderr.get(card, ()))

    async def run(self, chunk: ChunkSpec, spool: Path) -> ChunkResult:
        card = chunk.card
        lock = self._locks.setdefault(card, asyncio.Lock())
        started = time.perf_counter()

        def failed(message: str) -> ChunkResult:
            return ChunkResult(
                index=chunk.index,
                card=card,
                path=spool,
                frames_encoded=0,
                frames_skipped=0,
                bytes_out=0,
                wall_seconds=time.perf_counter() - started,
                error=message,
            )

        async with lock:
            try:
                process = await self._worker(card)
            except OSError as exc:
                return failed(f"could not start a worker on card {card}: {exc}")
            assert process.stdin is not None and process.stdout is not None

            payload = {
                "chunk": chunk.as_dict(),
                "output_path": str(spool),
            }
            try:
                process.stdin.write((json.dumps(payload) + "\n").encode())
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                return failed(
                    f"worker on card {card} closed its input before the item was "
                    f"sent; its stderr tail:\n{self._tail(card)[-2000:]}"
                )

            while True:
                line = await process.stdout.readline()
                if not line:
                    await process.wait()
                    return failed(
                        f"worker on card {card} exited {process.returncode} while "
                        f"item {chunk.index} was in flight; its stderr tail:\n"
                        f"{self._tail(card)[-2000:]}"
                    )
                text = line.decode(errors="replace")
                if text.startswith(REPORT_PREFIX):
                    break

            elapsed = time.perf_counter() - started
            try:
                report = json.loads(text[len(REPORT_PREFIX) :])
            except ValueError as exc:
                return failed(f"worker on card {card} sent an unparsable report: {exc}")
            if report.get("error"):
                return failed(
                    f"item {chunk.index} on card {card} failed inside the worker: "
                    f"{report['error']}"
                )
            return ChunkResult(
                index=chunk.index,
                card=card,
                path=spool,
                frames_encoded=report.get("frames_encoded", 0),
                frames_skipped=report.get("frames_skipped", 0),
                bytes_out=report.get("bytes_out", 0),
                wall_seconds=elapsed,
                stage_ms_per_frame=report.get("stage_ms_per_frame", {}),
                frame_digests=tuple(report.get("frame_digests", ())),
            )

    async def close(self) -> None:
        """Shut every worker down. Idempotent; called from the executor's
        ``finally``, so it runs on the failure and cancellation paths too."""
        for process in self._workers.values():
            if process.returncode is not None:
                continue
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except (BrokenPipeError, ConnectionResetError, RuntimeError):
                pass
        for card, process in self._workers.items():
            if process.returncode is not None:
                continue
            try:
                await asyncio.wait_for(process.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "worker on card %s did not exit on EOF within 30 s; killing it. "
                    "A worker that outlives its job holds a CUDA context and the "
                    "card's share of the reservation.",
                    card,
                )
                process.kill()
                await process.wait()
        for task in self._drains:
            task.cancel()
        await asyncio.gather(*self._drains, return_exceptions=True)
        self._drains.clear()
        self._workers.clear()


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
    #: ``"pull"`` or ``"pinned"``. Recorded because the two produce the same
    #: output and different card-second distributions, so a record that does
    #: not say which one ran cannot be compared against another.
    schedule: str = "pinned"
    #: ``(chunk index, card)`` in the order the cards took the items. Empty
    #: under pinned scheduling, where the order was decided by the planner.
    pull_order: list[tuple[int, str]] = field(default_factory=list)
    items_per_card: dict[str, int] = field(default_factory=dict)
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
            "schedule": self.schedule,
            "pull_order": [list(entry) for entry in self.pull_order],
            "items_per_card": dict(self.items_per_card),
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
    """Runs the chunks of one job concurrently and stitches them in order.

    Two scheduling modes, and which one is in use is an explicit argument
    rather than an inference:

    *   **Pinned** (``cards`` not given). Every chunk already names its card,
        because ``shard_plan`` weighted the split by a measured rate table and
        :func:`chunk_specs_from_plan` baked the answer in. One task per chunk.
    *   **Pull** (``cards`` given). The chunks are card-agnostic items in a
        shared :class:`WorkQueue` and one task per *card* takes the next one
        whenever it is free. The split is not decided in advance at all; a
        card that turns out to be twice as fast simply comes back twice as
        often.

    The correctness gates do not distinguish between the two, and that is the
    load-bearing claim rather than a convenience: :func:`verify_chunk_arithmetic`
    runs on the item list before a card is touched, the per-chunk
    ``output_frames`` check runs on every result, and both read only the
    timeline split. Late-binding a card cannot move a seam.
    """

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
        cards: Sequence[str] | None = None,
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

        self._cards = tuple(cards) if cards is not None else None
        if self._cards is not None:
            if not self._cards:
                raise MultiCardError("pull scheduling needs at least one card")
            if len(set(self._cards)) != len(self._cards):
                raise MultiCardError(
                    f"a card is offered twice to the pull queue: {list(self._cards)}. "
                    "Two workers on one physical card is a co-tenancy decision, not "
                    "a scheduling one, and this executor does not make it silently."
                )
            pinned = [c.index for c in self.chunks if c.is_assigned]
            if pinned:
                raise MultiCardError(
                    f"pull scheduling was asked for, but chunks {pinned} already "
                    "name a card. A pre-weighted plan and a pull queue are two "
                    "answers to the same question; running one list through the "
                    "other's scheduler would silently discard the plan."
                )
            self.queue: WorkQueue | None = WorkQueue(self.chunks)
            self.stats.schedule = "pull"
        else:
            unassigned = [c.index for c in self.chunks if not c.is_assigned]
            if unassigned:
                raise MultiCardError(
                    f"chunks {unassigned} name no card and no card list was given; "
                    "pass cards=[...] to schedule them from a pull queue"
                )
            self.queue = None
            self.stats.schedule = "pinned"

        card_count = (
            len(self._cards) if self._cards else len({c.card for c in self.chunks})
        )
        # One completed-but-unforwarded chunk per card is the smallest bound
        # that still lets every card work while chunk 0 is streaming out.
        self._slots = asyncio.Semaphore(
            max(1, spool_chunks if spool_chunks is not None else card_count)
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

        async def pinned_work(chunk: ChunkSpec) -> None:
            await self._slots.acquire()
            try:
                if self.cancelled:
                    return
                result = await self.runner.run(chunk, self._spool_path(chunk))
                results[chunk.index] = result
            finally:
                done[chunk.index].set()

        async def pull_work(card: str) -> None:
            """One card, taking items until the queue is empty.

            The slot is acquired *before* the item is taken, and that order is
            what makes the spool bound deadlock-free rather than merely
            bounded. Items then leave the queue in exactly the order slots
            were acquired, so a card blocked on a slot is always waiting for
            items with a *lower* index than the one it is about to take --
            and those are precisely the ones the forwarding loop is working
            through and releasing. A card can never be blocked behind work
            that is itself blocked behind that card.
            """
            assert self.queue is not None
            while True:
                await self._slots.acquire()
                if self.cancelled:
                    self._slots.release()
                    return
                chunk = self.queue.take(card)
                if chunk is None:
                    self._slots.release()
                    return
                try:
                    result = await self.runner.run(chunk, self._spool_path(chunk))
                    results[chunk.index] = result
                finally:
                    done[chunk.index].set()
                if not result.ok:
                    # A failed item stops this card. The forwarding loop turns
                    # it into the run's error; carrying on would burn the rest
                    # of the queue on cards for a run that is already lost.
                    #
                    # Items still in the queue are simply never taken, and that
                    # cannot strand the forwarding loop: items leave the queue
                    # in index order, so every item before the failed one was
                    # taken and will complete, and the loop raises on the
                    # failed one before it can wait on an untaken successor.
                    return

        if self._cards is not None:
            workers = [asyncio.create_task(pull_work(card)) for card in self._cards]
        else:
            workers = [asyncio.create_task(pinned_work(chunk)) for chunk in self.chunks]

        try:
            for chunk in self.chunks:
                await done[chunk.index].wait()
                result = results[chunk.index]
                # Under pull scheduling the card is not known until the item
                # has been taken, so the messages below name the card that ran
                # it rather than the card a plan intended.
                where = result.card if result is not None else chunk.card
                if result is None:
                    if self.cancelled:
                        break
                    raise MultiCardError(
                        f"chunk {chunk.index} on card {where} produced no result"
                    )
                if not result.ok:
                    raise MultiCardError(result.error or "chunk failed")
                if result.frames_encoded != chunk.output_frames:
                    raise MultiCardError(
                        f"chunk {chunk.index} on card {where} encoded "
                        f"{result.frames_encoded} frames, the seam arithmetic says "
                        f"{chunk.output_frames}. A chunk that is short or long by "
                        "one frame desynchronises the whole output, so the run is "
                        "refused rather than stitched."
                    )
                await self._forward(result)
                self.stats.chunks.append(result.as_dict())
                self.stats.frame_digests.extend(result.frame_digests)
                self.stats.frames_encoded += result.frames_encoded
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
            if self.queue is not None:
                self.stats.pull_order = list(self.queue.pull_order)
                self.stats.items_per_card = self.queue.items_per_card()
            await asyncio.gather(*workers, return_exceptions=True)
            close = getattr(self.runner, "close", None)
            if callable(close):
                await close()
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
    "DEFAULT_CHUNKS_PER_CARD",
    "MIN_PULL_CHUNK_FRAMES",
    "REPORT_PREFIX",
    "UNASSIGNED_CARD",
    "ChunkResult",
    "ChunkRunner",
    "ChunkSpec",
    "MultiCardError",
    "MultiCardExecutor",
    "MultiCardStats",
    "PersistentChunkRunner",
    "SubprocessChunkRunner",
    "WorkQueue",
    "chunk_specs_from_plan",
    "pinned_child_env",
    "pull_queue_chunks",
    "pull_queue_size",
    "resolve_cards",
    "total_output_frames",
    "verify_chunk_arithmetic",
]
