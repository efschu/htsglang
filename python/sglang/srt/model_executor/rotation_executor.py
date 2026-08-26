# SPDX-License-Identifier: Apache-2.0
"""#809/W28 slice 2: the DEVICE SIDE that executes a chunk-rotation plan.

Slice 1 (:mod:`rotation_plan`) fixed the arithmetic -- how large the overshoot
must be, and in what order the chunks travel. This module runs that plan
against the real arena, and it exists because the plan alone is not sufficient
on THIS arena.

THE HAZARD THE ARITHMETIC CANNOT SEE. The weights arena is ONE contiguous
device tensor sized ``max(pp, tp)`` (``weights_arena.allocate_arena``), and the
refill overwrites ``arena[: layout.total_bytes]`` IN PLACE
(``weights_arena.py:1184,1192``). Under this scheme the host image is likewise
ONE buffer, reused by whichever layout is resting. So at every chunk offset k
the two directions are CIRCULARLY dependent:

    the H2D wants to write ``arena[k]``  -- which the D2H still has to read
    the D2H wants to write ``image[k]``  -- which the H2D still has to read

Serialising them throws away the duplex the whole scheme is for; running them
concurrently corrupts the image, and it corrupts it in the direction a
checksum on THIS flip would not catch -- the damage lands in the image the
NEXT flip will stream in.

THE RING IS THEREFORE LOAD-BEARING, not an optimisation. It holds the incoming
chunk while the outgoing one is placed into the pages that chunk vacated:

    save  image[k] -> ring slot          (host, off both links)
    D2H   arena[k] -> image[k]           (lane D)
    H2D   ring slot -> arena[k]          (lane H, gated on that D2H)

Chunk k+1's D2H is enqueued before chunk k's H2D is waited on, so the two
lanes genuinely run together -- a full-duplex pipeline with no aliasing
anywhere inside it.

AND THE RAM ARITHMETIC THEN AGREES WITH SLICE 1 BY CONSTRUCTION. The host cost
is one max-sized image buffer (= one image PLUS the size asymmetry) plus
``depth * chunk`` of ring, which is exactly ``rotation_overshoot_bytes``. The
two slices are not merely consistent; they are the same number reached twice.

THE COPY-BACK IS NOT WRITE-BACK. The weights are immutable and nothing is being
saved. It is residency PLACEMENT for the next flip, which is what a
single-layout RAM budget requires. A reader who mistakes it for a write-back
will "optimise" it away and break the following flip.

PRIOR ART, REUSED RATHER THAN REBUILT: the pinned ring is #720's
``ReadBufferPool``, which charges its bytes to the pinned-host registry BEFORE
allocating them (#729's register-then-allocate discipline), exactly as
``weights_arena._refill_staging_pool`` already composes it. The acceptance
readout is #856(a)'s ``RefillLegTiming`` / ``refill_bound_phrase``; no new
telemetry is introduced here.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from sglang.srt.model_executor.rotation_plan import (
    plan_rotation,
    rotation_overshoot_bytes,
)
from sglang.srt.model_executor.weights_arena import (
    _CHECKSUM_BYTES,
    RefillLegTiming,
    uint8_checksum,
)

logger = logging.getLogger(__name__)

LOG_PREFIX = "#809 ROTATION"

#: Ring identity. One name, so the pinned-host ledger sees ONE post however
#: many flips run: the ring is registered once at first use and reused.
_RING_NAME = "phase_flip_rotation_staging"
_RING_FLAG = "--phase-flip-image-rotation"

_ring_lock = threading.Lock()
_rotation_ring = None


class RotationHazard(RuntimeError):
    """A rotation that cannot be executed safely as configured.

    Distinct from :class:`~sglang.srt.model_executor.rotation_plan.
    RotationPlanError`, which refuses a plan that cannot be SCHEDULED. This one
    refuses a plan that schedules fine but would alias its own buffers.
    """


@dataclass
class RotationStats:
    """A PURE RECORD of what one rotation did. Every judgement lives elsewhere.

    ``overlapped_steps`` is the falsifier for the duplex premise, and it is
    measured on the EXECUTOR rather than on the plan: a step counts as
    overlapped when, at the instant its D2H was enqueued, an earlier chunk's
    H2D had not yet been waited on. A plan-shaped counter would report overlap
    for a serialised implementation, which is the reading this number exists to
    make impossible.
    """

    steps: int = 0
    overlapped_steps: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    ring_saves: int = 0
    aliased_steps: int = 0
    priming: bool = False
    elapsed_s: float = 0.0

    @property
    def overlap_share(self) -> float:
        return (self.overlapped_steps / self.steps) if self.steps else 0.0


@dataclass
class RotationPhases:
    """WHERE THE ROTATION'S WALL TIME ACTUALLY WENT (#809/W28 follow-up).

    THE DEFECT THIS CLOSES, measured in the W28 window. The leg reported
    `LINK-BOUND (read 0.000s / h2d-wait 0.114s ... drain 0.001s)` on a 4.833 s
    rotation -- read + wait + drain account for 2.4 % of it. The phrase named a
    bound nobody had observed, which is the #851/indicator class: one number
    with several meanings sitting inside the dominant term of the seam.

    HOST-SIDE PHASES ARE THE ONES THAT MUST RECONCILE, and that follows from
    the measurement rather than from taste: `h2d_wait_s` was NEAR ZERO, so the
    host was not blocked on the device. Whatever consumed the wall clock was
    work the host thread itself did, so timing the host side of each call is
    what can add up to the whole. The GPU spans below answer a DIFFERENT
    question (did the two directions actually overlap on the device) and are
    deliberately not part of the reconciliation -- adding a device span to a
    host sum is how a reconciliation is made to "pass" without meaning
    anything.

    ``residual_s`` exists so the unexplained mass has to land somewhere NAMED.
    A phase set whose parts do not add up to its whole is the #846 class, so
    the residual is reported rather than distributed.
    """

    save_s: float = 0.0
    d2h_issue_s: float = 0.0
    h2d_issue_s: float = 0.0
    wait_s: float = 0.0
    ring_s: float = 0.0
    checksum_s: float = 0.0
    plan_s: float = 0.0
    total_s: float = 0.0
    #: Device-side spans, read ONCE after the drain (never synchronised inside
    #: the loop, per the ms-per-round canon). 0.0 when there is no device.
    gpu_d2h_s: float = 0.0
    gpu_h2d_s: float = 0.0

    @property
    def accounted_s(self) -> float:
        return (
            self.save_s
            + self.d2h_issue_s
            + self.h2d_issue_s
            + self.wait_s
            + self.ring_s
            + self.checksum_s
            + self.plan_s
        )

    @property
    def residual_s(self) -> float:
        return max(0.0, float(self.total_s) - self.accounted_s)

    @property
    def residual_share(self) -> float:
        return (self.residual_s / self.total_s) if self.total_s > 0 else 0.0

    def dominant(self) -> Tuple[str, float]:
        """(name, seconds) of the largest term, residual included.

        The residual COMPETES with the named phases on purpose: if the
        unexplained mass is the biggest term, the honest answer is that the
        instrument still does not know, and it must say so rather than crown
        the largest thing it happens to measure.
        """
        terms = {
            "save": self.save_s,
            "d2h_issue": self.d2h_issue_s,
            "h2d_issue": self.h2d_issue_s,
            "wait": self.wait_s,
            "ring": self.ring_s,
            "checksum": self.checksum_s,
            "plan": self.plan_s,
            "UNACCOUNTED": self.residual_s,
        }
        name = max(terms, key=lambda k: terms[k])
        return name, terms[name]


#: The share of a leg the phase set may leave unexplained and still be called
#: an instrument. Chosen, not derived: W28 left 97.6 % unaccounted, so anything
#: that still admits a majority-unexplained leg would pass the very case this
#: was built for.
PHASE_RECONCILE_TOLERANCE = 0.10


def phases_reconcile(
    phases: RotationPhases, tolerance: float = PHASE_RECONCILE_TOLERANCE
) -> bool:
    """Do the parts add up to the whole, within ``tolerance``?"""
    if phases.total_s <= 0:
        return False
    return phases.residual_share <= float(tolerance)


def rotation_phase_report(phases: RotationPhases) -> str:
    """One line naming every term AND the leftover. Never hides the residual."""
    name, secs = phases.dominant()
    verdict = "RECONCILED" if phases_reconcile(phases) else "UNRECONCILED"
    return (
        f"phases {verdict} total {phases.total_s:.3f}s = "
        f"save {phases.save_s:.3f} + d2h-issue {phases.d2h_issue_s:.3f} + "
        f"h2d-issue {phases.h2d_issue_s:.3f} + wait {phases.wait_s:.3f} + "
        f"ring {phases.ring_s:.3f} + checksum {phases.checksum_s:.3f} + "
        f"plan {phases.plan_s:.3f} + UNACCOUNTED {phases.residual_s:.3f} "
        f"({phases.residual_share * 100:.1f} %); dominant={name} {secs:.3f}s; "
        f"gpu-span d2h {phases.gpu_d2h_s:.3f}s / h2d {phases.gpu_h2d_s:.3f}s"
    )


class TorchRotationOps:
    """The two copy lanes, over torch.

    On CUDA these are two real streams with events between them, so the H2D of
    a chunk cannot start before that chunk's D2H has landed while the NEXT
    chunk's D2H is already running. On CPU the copies are eager and the handles
    are bookkeeping only -- which is deliberate: the overlap counter then still
    measures the executor's PIPELINING DECISIONS, the property that determines
    whether the CUDA lanes can overlap at all, and it does so without a device.
    """

    def __init__(
        self,
        save_slices: Optional[int] = None,
        save_slice_min_bytes: Optional[int] = None,
    ) -> None:
        self._d2h_stream = None
        self._h2d_stream = None
        self._outstanding_h2d = 0
        self._save_slices = (
            _save_slices_default() if save_slices is None else max(1, int(save_slices))
        )
        #: #875: INJECTABLE SO A TEST CAN REACH THE SLICED PATH. The production
        #: threshold is 1 MiB against 32 MiB chunks, so a unit test written at a
        #: natural size falls entirely into the serial branch and asserts
        #: nothing about the slicing -- which is what the first version of
        #: test_rotation_save_slices_875 did, and three of its four mutants
        #: survived because of it.
        self._save_slice_min_bytes = (
            _SAVE_SLICE_MIN_BYTES
            if save_slice_min_bytes is None
            else max(0, int(save_slice_min_bytes))
        )

    # -- lane setup -------------------------------------------------------
    def _streams(self):
        if self._d2h_stream is None and torch.cuda.is_available():
            self._d2h_stream = torch.cuda.Stream()
            self._h2d_stream = torch.cuda.Stream()
        return self._d2h_stream, self._h2d_stream

    @staticmethod
    def _is_device(*tensors) -> bool:
        return any(getattr(t, "is_cuda", False) for t in tensors)

    @property
    def outstanding_h2d(self) -> int:
        """H2D copies enqueued and not yet waited on. The overlap witness."""
        return self._outstanding_h2d

    # -- the three primitives --------------------------------------------
    def save(self, dst_buf: torch.Tensor, src: torch.Tensor) -> None:
        """Host-to-host: hold the incoming chunk while its pages are reused.

        #875: SPLIT ACROSS PYTHON THREADS, because this copy is the seam's
        largest single term and it was running on one core.

        WHY IT EXISTS AT ALL, so nobody deletes it looking for the win. The
        rotation is an in-place transform whose two streams share ONE
        coordinate system: on the host the H2D READS ``host_image[off:off+len]``
        while the D2H WRITES it; on the arena the D2H READS
        ``arena[off:off+len]`` while the H2D WRITES it -- and
        ``d2h_offset == h2d_offset`` on every interleaved step (498 of 512 on
        PP0, 268 of 281 on PP1, 268 of 297 on PP2, against the real layout
        vectors). So each region must be read before it is written AND written
        before it is read. That cycle is what ``RotationHazard`` below names,
        and it cannot be broken by lagging the copy-back (a lagging D2H would
        read arena bytes the H2D has already overwritten) nor by reordering
        (both streams run sequentially from 0, so there is no better
        permutation). The staging copy is structural; only its RATE was not.

        WHY THREADS AND NOT ``torch.set_num_threads``. Every CUDA worker runs
        under a process-global ``torch.set_num_threads(1)``
        (model_runner.py:2224), set for weight loading. Raising it here would
        change it under every other thread in the process for the duration of
        the seam, and restoring it in a ``finally`` still leaves it wrong in
        between. ``Tensor.copy_`` releases the GIL, so N Python threads over N
        disjoint slices get the same parallelism and touch no global.

        THE MEASURED GAIN IS 1.42x, NOT 8x, and the difference is the whole
        reason this docstring carries numbers. On this box (5950X, 16 cores,
        one NUMA node), streaming 3-4 GiB through a 32 MiB ring:

            1 rank,  serial     4957-5109 MiB/s
            1 rank,  sliced x4      40860 MiB/s   8.0x  -- and a mirage
            3 ranks, serial     3757-3869 MiB/s         -- reproduces the
                                                           flip's own 2687-3768
            3 ranks, sliced x4  5353-5701 MiB/s   1.42x -- the real one

        Three concurrent ranks are DRAM-bandwidth-bound, not core-bound:
        aggregate copy rises from ~11.5 to ~16.5 GB/s and stops. PP0's ``save``
        therefore goes 4.342 s -> ~3.06 s, not to ~0.5 s. Benchmarking a
        three-rank operation on one rank overstates it by 5.6x.

        Slices are computed so every byte lands exactly once for any length and
        any slice count, including a short final chunk (the two layouts differ
        in size, so the tails never line up) and a length that does not divide
        by the slice count.
        """
        n = int(src.numel())
        if n <= 0:
            return
        workers = min(int(self._save_slices), n)
        if workers <= 1 or n < self._save_slice_min_bytes:
            dst_buf[:n].copy_(src)
            return
        # Ceiling division. COVERAGE does not depend on it -- `range(0, n, step)`
        # with `min(o + step, n)` tiles [0, n) exactly once for any positive
        # step -- so this is about the slice COUNT, not correctness: a floor
        # split yields `workers + 1` slices, the last one a sliver, which costs
        # a thread hand-off for a few bytes. Said precisely because the earlier
        # wording here claimed a floor split "leaves the remainder uncopied",
        # and a mutation test proved that claim false.
        step = -(-n // workers)
        bounds = [(o, min(o + step, n)) for o in range(0, n, step)]
        pool = _save_pool(len(bounds))
        if pool is None:
            dst_buf[:n].copy_(src)
            return
        list(pool.map(lambda b: dst_buf[b[0] : b[1]].copy_(src[b[0] : b[1]]), bounds))

    def d2h(self, src: torch.Tensor, dst: torch.Tensor) -> Any:
        """Device -> host, on the copy-back lane."""
        d2h_stream, _ = self._streams()
        if d2h_stream is not None and self._is_device(src):
            with torch.cuda.stream(d2h_stream):
                dst.copy_(src, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(d2h_stream)
            return ev
        dst.copy_(src)
        return None

    def h2d(self, src: torch.Tensor, dst: torch.Tensor, after: Any = None) -> Any:
        """Host -> device, gated on ``after`` so it cannot outrun the D2H."""
        _, h2d_stream = self._streams()
        if h2d_stream is not None and self._is_device(dst):
            if after is not None:
                h2d_stream.wait_event(after)
            with torch.cuda.stream(h2d_stream):
                dst.copy_(src, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(h2d_stream)
            self._outstanding_h2d += 1
            return ev
        dst.copy_(src)
        self._outstanding_h2d += 1
        return _EAGER

    def wait(self, handle: Any, is_h2d: bool = True) -> None:
        """Block until ``handle`` has landed.

        ``is_h2d`` keeps the overlap witness honest: draining a copy-back-only
        tail must not decrement the count of outstanding H2D copies, or the
        tail would silently erase the evidence of the overlap that preceded it.
        """
        if handle is None:
            return
        if handle is not _EAGER:
            handle.synchronize()
        if is_h2d:
            self._outstanding_h2d = max(0, self._outstanding_h2d - 1)


class _Eager:
    """Sentinel for a completed eager copy: a handle that is already done."""

    __slots__ = ()


_EAGER = _Eager()


def rotation_ring(chunk_bytes: int, depth: int):
    """The pinned ring, allocated ONCE per process and reused by every flip.

    Reuses #720's ``ReadBufferPool`` rather than adding a second ring: it
    charges its bytes to the pinned-host registry BEFORE allocating them, which
    is what keeps this host post inside the ledger on a swapless box instead of
    beside it. Registered here, in the worker that actually allocates the
    memory; the LAUNCHER only prices it (``joint_pinned_host_error``), because a
    planner-side registration of a post nothing allocates is the mistake
    commit 272d0d9d8c removed and must not be reintroduced.
    """
    global _rotation_ring
    from sglang.srt.mem_cache.read_buffer_pool import ReadBufferPool

    with _ring_lock:
        if _rotation_ring is not None:
            if _rotation_ring.page_bytes == int(
                chunk_bytes
            ) and _rotation_ring.capacity == int(depth):
                return _rotation_ring
            _rotation_ring.close()
            _rotation_ring = None
        _rotation_ring = ReadBufferPool(
            name=_RING_NAME,
            flag=_RING_FLAG,
            capacity=int(depth),
            page_bytes=int(chunk_bytes),
            factory=lambda: torch.empty(
                int(chunk_bytes), dtype=torch.uint8, pin_memory=True
            ),
        )
        return _rotation_ring


def allocate_rotation_image(
    pp_bytes: int, tp_bytes: int, pin: bool = True
) -> torch.Tensor:
    """ONE host buffer, sized for the LARGER layout plus its trailer.

    THIS REPLACES THE TWO LIFETIME IMAGES, and replacing rather than joining
    them is the point: W26 OOM-killed BOTH arms of the dual pin in the LAUNCH
    phase, before any flip ever ran. RAM holds one layout image plus the
    overshoot, and the resting layout rotates through this single buffer.

    Sized from the MAXIMUM and never from the resting layout: the buffer has to
    receive whichever layout the next flip copies back, and on a rank where the
    outgoing layout is the larger (PP0: 16362.7 MiB tp against 15925.8 MiB pp)
    a buffer sized for the resting one is 436.9 MiB short at the worst possible
    moment -- inside the seam's no-return region.
    """
    from sglang.srt.model_executor.weights_arena import _alloc_host_image

    span = max(int(pp_bytes), int(tp_bytes)) + _CHECKSUM_BYTES
    return _alloc_host_image(span, pin)


def rotation_host_bytes(pp_bytes: int, tp_bytes: int, chunk_bytes: int, depth: int):
    """(image buffer bytes, ring bytes) -- the host post this scheme costs.

    ONE max-sized image buffer, not two: the whole point of the rotation is
    that RAM never holds both layouts. The trailer rides on the buffer because
    the image is payload plus an int64 checksum (weights_arena.py:1182-1183).
    """
    span = max(int(pp_bytes), int(tp_bytes)) + _CHECKSUM_BYTES
    return span, int(chunk_bytes) * int(depth)


def _overlaps(a_off: int, a_len: int, b_off: int, b_len: int) -> bool:
    return a_len > 0 and b_len > 0 and a_off < b_off + b_len and b_off < a_off + a_len


#: #875: below this, thread hand-off costs more than the copy. Chunks are 32 MiB
#: by default (SGLANG_PHASE_FLIP_REFILL_CHUNK_MIB), so the real path is never
#: near this bound; it exists so a small-chunk configuration degrades to the
#: serial copy instead of paying scheduling overhead per slice.
_SAVE_SLICE_MIN_BYTES = 1 << 20

#: One pool for the process, built on first use and reused by every flip.
#: Per-chunk construction would dominate: a leg is 281-512 chunks.
_SAVE_POOL = None
_SAVE_POOL_WIDTH = 0


def _save_slices_default() -> int:
    """How many slices the staging copy is split into. 1 restores the old path.

    Default 4, which is where the three-rank measurement flattens (5353-5701
    MiB/s per rank against 3757-3869 serial); 8 buys nothing once the memory
    controller is the bound. Env-settable because the right value is a property
    of the box's memory system, not of this code.
    """
    try:
        from sglang.srt.environ import envs

        return max(1, int(envs.SGLANG_PHASE_FLIP_REFILL_SAVE_SLICES.get()))
    except Exception:  # noqa: BLE001 - a knob may never break a flip
        return 4


def _save_pool(width: int):
    """The shared slice pool, grown but never shrunk. None if threads are
    unavailable, in which case the caller copies serially -- a rotation may not
    fail because a thread could not be started."""
    global _SAVE_POOL, _SAVE_POOL_WIDTH
    if _SAVE_POOL is not None and _SAVE_POOL_WIDTH >= width:
        return _SAVE_POOL
    try:
        from concurrent.futures import ThreadPoolExecutor

        _SAVE_POOL = ThreadPoolExecutor(
            max_workers=width, thread_name_prefix="flip-save"
        )
        _SAVE_POOL_WIDTH = width
        return _SAVE_POOL
    except Exception:  # noqa: BLE001 - fall back to the serial copy
        return None


def rotate_arena(
    *,
    arena: torch.Tensor,
    host_image: torch.Tensor,
    incoming_bytes: int,
    outgoing_bytes: int,
    chunk_bytes: int,
    depth: int,
    ring: Optional[Any],
    ops: Optional[TorchRotationOps] = None,
    timing: Optional[RefillLegTiming] = None,
    priming: bool = False,
    verify_incoming: bool = True,
    phases: Optional[RotationPhases] = None,
) -> RotationStats:
    """Rotate ``host_image`` into ``arena`` while placing ``arena`` back into it.

    ``incoming_bytes`` is the layout resident in RAM and about to enter VRAM;
    ``outgoing_bytes`` is the layout resident in VRAM and about to be placed
    back into the RAM the incoming one vacates.

    THE CHECKSUM CONTRACT, and it is deliberately asymmetric. The outgoing
    trailer is computed from the ARENA before a byte moves -- that is the
    source of truth, and it is the same quantity ``arena_image`` writes
    (weights_arena.py:1129-1144). Verification of those returned bytes is NOT
    done here: it happens on the next flip, when this image streams back in and
    ``arena_refill``'s existing device-side ``uint8_checksum(dst)`` checks it.
    Re-checksumming a ~16 GiB host buffer on the seam's critical path to learn
    the same fact one flip earlier is not a trade this seam can afford, and
    adding a second clock over the same bytes is what #856(a) exists to stop.

    ``priming`` marks the first-ever rotation in a direction, whose image is not
    yet in RAM. It is recorded on the stats and never folded into a warm mean
    (P4): a steady-state figure averaged over the priming flip is the specific
    measurement error this ticket is most likely to make.
    """
    t0 = time.perf_counter()
    ops = ops if ops is not None else TorchRotationOps()
    incoming_bytes = int(incoming_bytes)
    outgoing_bytes = int(outgoing_bytes)
    chunk_bytes = int(chunk_bytes)
    depth = int(depth)

    need = max(incoming_bytes, outgoing_bytes) + _CHECKSUM_BYTES
    if int(host_image.numel()) < need:
        raise RotationHazard(
            f"{LOG_PREFIX} host image holds {int(host_image.numel())} B but the "
            f"rotation needs {need} B -- ONE buffer sized for the LARGER layout "
            f"plus its {_CHECKSUM_BYTES}-byte trailer. Sizing it from the "
            f"incoming layout, or from a mean of the two, is the OOM."
        )
    if int(arena.numel()) < max(incoming_bytes, outgoing_bytes):
        raise RotationHazard(
            f"{LOG_PREFIX} arena holds {int(arena.numel())} B, short of the "
            f"{max(incoming_bytes, outgoing_bytes)} B this rotation touches."
        )

    overshoot = rotation_overshoot_bytes(
        incoming_bytes, outgoing_bytes, chunk_bytes, depth
    )
    _t_plan = time.perf_counter()
    steps = plan_rotation(
        incoming_bytes=incoming_bytes,
        outgoing_bytes=outgoing_bytes,
        chunk_bytes=chunk_bytes,
        overshoot_bytes=overshoot,
    )

    if phases is not None:
        phases.plan_s += time.perf_counter() - _t_plan
    _t_ck = time.perf_counter()
    want_in = None
    if verify_incoming and incoming_bytes:
        want_in = int(
            host_image[incoming_bytes : incoming_bytes + _CHECKSUM_BYTES]
            .clone()
            .view(torch.int64)
            .item()
        )
    # From the ARENA, before a byte of it moves. See the docstring.
    out_sum = uint8_checksum(arena[:outgoing_bytes]) if outgoing_bytes else 0
    if phases is not None:
        phases.checksum_s += time.perf_counter() - _t_ck

    stats = RotationStats(priming=bool(priming))
    inflight: deque = deque()

    def _retire_one() -> None:
        buf, handle, is_h2d = inflight.popleft()
        t_wait = time.perf_counter()
        ops.wait(handle, is_h2d=is_h2d)
        _w = time.perf_counter() - t_wait
        if timing is not None:
            timing.h2d_wait_s += _w
        if phases is not None:
            phases.wait_s += _w
        if buf is not None and ring is not None:
            ring.release(buf)

    for step in steps:
        stats.steps += 1
        aliased = _overlaps(
            step.h2d_offset, step.h2d_len, step.d2h_offset, step.d2h_len
        )
        buf = None
        src = None
        if step.h2d_len:
            if aliased:
                stats.aliased_steps += 1
                if ring is None:
                    raise RotationHazard(
                        f"{LOG_PREFIX} this rotation transforms the arena IN "
                        f"PLACE: at offset {step.h2d_offset} the H2D would "
                        f"overwrite the {step.d2h_len} B the copy-back has not "
                        f"read yet, and the copy-back would overwrite the host "
                        f"bytes the H2D reads. A staging ring is required to "
                        f"break that cycle; running the two lanes over aliased "
                        f"ranges corrupts the NEXT flip's image, which this "
                        f"flip's checksum would not catch."
                    )
                _t = time.perf_counter()
                buf = ring.acquire()
                if phases is not None:
                    phases.ring_s += time.perf_counter() - _t
                _t = time.perf_counter()
                ops.save(
                    buf,
                    host_image[step.h2d_offset : step.h2d_offset + step.h2d_len],
                )
                if phases is not None:
                    phases.save_s += time.perf_counter() - _t
                stats.ring_saves += 1
                src = buf[: step.h2d_len]
            else:
                src = host_image[step.h2d_offset : step.h2d_offset + step.h2d_len]

        d2h_handle = None
        if step.d2h_len:
            # Counted BEFORE anything is waited on: this is the moment that
            # decides whether the two lanes can run together.
            if ops.outstanding_h2d > 0:
                stats.overlapped_steps += 1
            _t = time.perf_counter()
            d2h_handle = ops.d2h(
                arena[step.d2h_offset : step.d2h_offset + step.d2h_len],
                host_image[step.d2h_offset : step.d2h_offset + step.d2h_len],
            )
            if phases is not None:
                phases.d2h_issue_s += time.perf_counter() - _t
            stats.d2h_bytes += step.d2h_len

        if step.h2d_len:
            _t = time.perf_counter()
            handle = ops.h2d(
                src,
                arena[step.h2d_offset : step.h2d_offset + step.h2d_len],
                after=d2h_handle,
            )
            if phases is not None:
                phases.h2d_issue_s += time.perf_counter() - _t
            stats.h2d_bytes += step.h2d_len
            inflight.append((buf, handle, True))
            if timing is not None:
                timing.chunks += 1
        elif step.d2h_len:
            # A copy-back-only tail still has to land before the trailer is
            # written, or the checksum would describe bytes still in flight.
            # Queued rather than waited on the spot, so the tail keeps
            # pipelining instead of turning into a per-chunk stall.
            inflight.append((None, d2h_handle, False))
            if timing is not None:
                timing.chunks += 1

        while len(inflight) >= depth:
            _retire_one()

    t_drain = time.perf_counter()
    while inflight:
        _retire_one()
    if timing is not None:
        timing.drain_s += time.perf_counter() - t_drain

    if outgoing_bytes:
        host_image[outgoing_bytes : outgoing_bytes + _CHECKSUM_BYTES] = torch.tensor(
            [out_sum], dtype=torch.int64
        ).view(torch.uint8)

    _t_ck2 = time.perf_counter()
    if want_in is not None:
        have = uint8_checksum(arena[:incoming_bytes])
        if have != want_in:
            raise RotationHazard(
                f"{LOG_PREFIX} rotated image checksum mismatch: arena sums to "
                f"{have}, image trailer says {want_in}. The arena content is "
                f"now undefined and the layout must not be served."
            )

    if phases is not None:
        phases.checksum_s += time.perf_counter() - _t_ck2
    stats.elapsed_s = time.perf_counter() - t0
    if phases is not None:
        phases.total_s = stats.elapsed_s
    return stats


def rotation_report(direction: str, stats: RotationStats) -> str:
    """One line, with the priming flip named as such so no mean hides it."""
    kind = "PRIMING" if stats.priming else "warm"
    return (
        f"{LOG_PREFIX} {direction} {kind} rotation: "
        f"{stats.h2d_bytes / (1024 * 1024):.1f} MiB in / "
        f"{stats.d2h_bytes / (1024 * 1024):.1f} MiB back over "
        f"{stats.steps} chunk(s), {stats.overlapped_steps} overlapped "
        f"({stats.overlap_share * 100:.1f} %), {stats.elapsed_s:.3f} s"
    )
