# SPDX-License-Identifier: Apache-2.0
"""VISION-WEIGHTS (plan PLAN-VISION-GEWICHTE-VERDRAENGEN-1009, AP1): the
transient tower BORROWS weight memory of the PP0 rank instead of KV pages.

``--pdflip-vision-place weights`` (``FLLIPER_PDFLIP_VISION_PLACE=weights``, default
off until the metal proof). The tower's parameters become VIEWS on the device
memory of VICTIM weights of this rank; the victim bytes are copied to a host
image first and written back after the encode, and a device checksum of the
victim bytes before and after proves the return. The addresses never change
(no unmap, no allocator, no cudaFree), so CUDA graphs that baked those
addresses read the same bytes again after the return.

This module is the MODEL-NEUTRAL core. Nothing here names a model or a
layer; WHICH tensors may be victims is a :class:`VictimSource`, registered
per line (27B: ``pdflip.vision_victim_27b``; NF builds its own over the resident
expert rows and returns them through its store instead of a host image).

Interface a source implements (``VictimSource``)::

    kind           "dense" | "pp_only" | "experts" | ... (the W102 field)
    inventory()    every eligible storage, deterministic: [VictimCandidate]
    views(segs)    the live uint8 device view of each planned segment
    stash(views)   make the bytes recoverable (host image D2H; NF: no-op)
    restore(views) the bytes back on the device (H2D; NF: refill from store)
    release()      drop what stash() holds (the host image)
    join()         R3: no async reader/writer of a victim during the stage
    shape_tower(module, ckpt, largest_run) -> split map (default: none)
    host_bytes     what stash() holds in host RAM right now

The ORDER and the INVARIANTS live here, in :class:`VictimLease`:

    open:  join -> inventory -> plan (W105b when short, nothing touched)
           -> live views (W111b when the plan does not match the storages)
           -> checksum -> stash (host image; a failed malloc is W105b,
              still before any device byte moved)
    close: stream sync -> restore -> checksum -> compare -> release
           restore failed or checksum differs = W110c, FATAL (crash-stop:
           the group must never serve with foreign bytes in its weights)

``close`` runs on EVERY way out of the stage (``run_rank_stage``'s finally),
before any teardown error escalates; the deadline is checked only between
legs, never before the return.

The encoder's ACTIVATIONS are not victim memory: a view cannot serve the
caching allocator, so they come from the allocator as before (the rank's
booked prefill transient, idle before the admission). Their size is computed
from the REAL image (patches; no own area cap, only the model's) by
:func:`encode_work_bytes` and checked against the card's air before anything
moves -- a named W105b with the numbers, never a silent cut.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import msgspec
import torch

from flliper.srt.pdflip.vision_rank_stage import (
    MIB,
    PLACE_WEIGHTS,
    SLAB_ALIGN,
    CkptTensor,
    VisionRankStageRefused,
    slab_bytes,
)

logger = logging.getLogger(__name__)

__all__ = ["PLACE_WEIGHTS", "VictimLease", "VictimSource", "HostImageVictims", "register_source"]

#: W-codes (vision family, collision-free: W105 tail, W110/W110b residue,
#: W111 arming are taken)
W_VICTIM_SHORT = "W105b PdFlipVisionVictimShort"
W_VICTIM_NOT_RESTORED = "W110c PdFlipVisionVictimNotRestored"
W_VICTIM_PLAN_REFUSED = "W111b PdFlipVisionVictimPlanRefused"
#: the arming line of the victim inventory (M0) and the host-RAM line
W_VICTIM_ARMED = "W102 PdFlipVisionStage VICTIM-ARMED"
#: the store census of the expert-row source (its own line: ARMED is the M0
#: arming line, one per arming -- review H2)
W_VICTIM_STORE = "W102 PdFlipVisionStage VICTIM-STORE"
W_VICTIM_HOST = "W102 PdFlipVisionStage VICTIM-HOST"


class VisionVictimShort(VisionRankStageRefused):
    """W105b: the victims cannot hold the tower (or the encode cannot get its
    work memory); nothing has moved, the request is aborted by name."""


class VisionVictimPlanRefused(VisionRankStageRefused):
    """W111b: the planned victims do not match the live storages; nothing
    has moved."""


class VisionVictimNotRestored(RuntimeError):
    """W110c, FATAL: victim bytes did not come back. The group stops (raised
    out of the scheduler pass on purpose) instead of serving wrong text."""


# ---------------------------------------------------------------------------
# 1. records (pure)
# ---------------------------------------------------------------------------


class VictimCandidate(msgspec.Struct, frozen=True):
    """One eligible storage: ``key`` = its base address, ``offset`` = the
    first ``SLAB_ALIGN``-aligned byte in it (0 for every CUDA allocation),
    ``nbytes`` = the usable bytes from there."""

    name: str
    key: int
    offset: int
    nbytes: int
    storage_nbytes: int


class VictimSegment(msgspec.Struct, frozen=True):
    """The used bytes ``[offset, offset + nbytes)`` of a victim storage."""

    name: str
    key: int
    offset: int
    nbytes: int
    storage_nbytes: int


class TowerSlot(msgspec.Struct, frozen=True):
    """Where one tower tensor (in tower order) sits: segment index, offset."""

    name: str
    nbytes: int
    seg: int
    offset: int


class VictimPlan(msgspec.Struct, frozen=True):
    segments: Tuple[VictimSegment, ...]
    slots: Tuple[TowerSlot, ...]
    inventory_bytes: int
    largest_run: int

    @property
    def victim_bytes(self) -> int:
        return sum(s.nbytes for s in self.segments)


def _align_up(n: int, align: int) -> int:
    return (int(n) + align - 1) // align * align


# ---------------------------------------------------------------------------
# 2. the choice: first-fit-decreasing over the largest runs (pure)
# ---------------------------------------------------------------------------


def ffd_place(sizes: Sequence[int], capacities: Sequence[int], *, align: int = SLAB_ALIGN
              ) -> Optional[List[Tuple[int, int]]]:
    """(segment, offset) per size (in the given order), placed in DESCENDING
    size order into the first segment with room; a tensor never straddles two
    segments. None when one does not fit."""
    order = sorted(range(len(sizes)), key=lambda i: (-int(sizes[i]), i))
    used = [0] * len(capacities)
    out: List[Optional[Tuple[int, int]]] = [None] * len(sizes)
    for i in order:
        n = int(sizes[i])
        for s, cap in enumerate(capacities):
            start = _align_up(used[s], align)
            if start + n <= int(cap):
                out[i] = (s, start)
                used[s] = start + n
                break
        else:
            return None
    return out  # type: ignore[return-value]


def plan_victims(candidates: Sequence[VictimCandidate], tower: Sequence[Tuple[str, int]], *,
                 align: int = SLAB_ALIGN) -> VictimPlan:
    """The victims for ``tower`` ((name, nbytes) in tower order).

    Rules (plan §2, R5): moved bytes = the tower's bytes (each victim is
    trimmed to its used prefix, 256 B grain), then the fewest storages
    (largest first), deterministic (size desc, then name). Raises W105b with
    the numbers when the inventory cannot hold the tower."""
    cands = sorted((c for c in candidates if c.nbytes > 0),
                   key=lambda c: (-c.nbytes, c.name, c.key))
    inventory = sum(c.nbytes for c in cands)
    largest = cands[0].nbytes if cands else 0
    sizes = [int(n) for _, n in tower]
    need = slab_bytes(sizes, align)
    big = max(sizes) if sizes else 0
    if big > largest:
        name = tower[sizes.index(big)][0]
        raise VisionVictimShort(
            f"{W_VICTIM_SHORT}: tower tensor {name} needs {big / MIB:.1f} MiB in one piece, the "
            f"largest victim run is {largest / MIB:.1f} MiB ({len(cands)} victims, "
            f"{inventory / MIB:.1f} MiB)")
    if need > inventory:
        raise VisionVictimShort(
            f"{W_VICTIM_SHORT}: the tower needs {need / MIB:.1f} MiB, the victims hold "
            f"{inventory / MIB:.1f} MiB ({len(cands)} victims)")
    lo, acc = 0, 0
    while lo < len(cands) and acc < need:
        acc += cands[lo].nbytes
        lo += 1
    caps = [c.nbytes for c in cands]
    if ffd_place(sizes, caps, align=align) is None:
        raise VisionVictimShort(
            f"{W_VICTIM_SHORT}: first-fit-decreasing places the tower ({need / MIB:.1f} MiB, largest "
            f"tensor {big / MIB:.1f} MiB) into none of the {len(cands)} victims "
            f"({inventory / MIB:.1f} MiB)")
    # the fewest victims that place it: first fit with one more (smaller)
    # victim at the END places every tensor as before, so success is monotone
    # in k -- a bisection, not a walk
    hi = len(cands)
    while lo < hi:
        mid = (lo + hi) // 2
        if ffd_place(sizes, caps[:mid], align=align) is None:
            lo = mid + 1
        else:
            hi = mid
    placed = ffd_place(sizes, caps[:hi], align=align)
    return _trimmed(cands[:hi], tower, placed, inventory, largest, align)


def _trimmed(cands, tower, placed, inventory, largest, align) -> VictimPlan:
    hi = [0] * len(cands)
    for (s, off), (_, n) in zip(placed, tower):
        hi[s] = max(hi[s], off + int(n))
    segs, remap = [], {}
    for s, c in enumerate(cands):
        if hi[s] == 0:
            continue
        remap[s] = len(segs)
        segs.append(VictimSegment(name=c.name, key=c.key, offset=c.offset,
                                  nbytes=min(c.nbytes, _align_up(hi[s], align)),
                                  storage_nbytes=c.storage_nbytes))
    slots = tuple(TowerSlot(name=name, nbytes=int(n), seg=remap[s], offset=off)
                  for (s, off), (name, n) in zip(placed, tower))
    return VictimPlan(segments=tuple(segs), slots=slots, inventory_bytes=inventory,
                      largest_run=largest)


def largest_adjacent_run(candidates: Sequence[VictimCandidate]) -> int:
    """M0 instrument: the longest run of victims that lie back to back in the
    device address space (key + nbytes == next key). Measured only -- the
    placement never spans two storages."""
    best = run = 0
    end = None
    for c in sorted(candidates, key=lambda c: c.key):
        start = c.key + c.offset
        run = run + c.nbytes if end == start else c.nbytes
        end = start + c.nbytes
        best = max(best, run)
    return best


# ---------------------------------------------------------------------------
# 3. the tower as pieces: a split map from the source (AP2), applied to the
#    checkpoint (pure)
# ---------------------------------------------------------------------------

#: ckpt module name -> ((piece module name, rows), ...) in row order
SplitMap = Dict[str, Tuple[Tuple[str, int], ...]]


def split_checkpoint(tensors: Sequence[CkptTensor], split: SplitMap,
                     map_name: Callable[[str], str]) -> Tuple[List[CkptTensor], Callable[[str], str]]:
    """Cut every checkpoint tensor whose module name is in ``split`` into its
    row pieces (dim 0 of a row-major tensor = contiguous file bytes), and the
    name mapper that sends a piece to its own module name."""
    if not split:
        return list(tensors), map_name
    out, piece_of = [], {}
    for ck in tensors:
        pieces = split.get(map_name(ck.name))
        if not pieces:
            out.append(ck)
            continue
        row_bytes = ck.nbytes // int(ck.shape[0])
        off, rows_seen = ck.file_offset, 0
        for i, (pname, rows) in enumerate(pieces):
            key = f"{ck.name}#{i}"
            out.append(CkptTensor(key, ck.dtype, (int(rows),) + tuple(ck.shape[1:]), off,
                                  int(rows) * row_bytes))
            piece_of[key] = pname
            off += int(rows) * row_bytes
            rows_seen += int(rows)
        if rows_seen != int(ck.shape[0]):
            raise VisionRankStageRefused(
                f"{ck.name}: split pieces cover {rows_seen} of {ck.shape[0]} rows")

    def mapped(name: str) -> str:
        return piece_of[name] if name in piece_of else map_name(name)

    return out, mapped


# ---------------------------------------------------------------------------
# 4. the device checksum and the host image (model-neutral legs)
# ---------------------------------------------------------------------------

CHECKSUM_CHUNK = 8 * MIB


def segment_checksum(view: torch.Tensor) -> Tuple[int, ...]:
    """Two sums per 8 MiB chunk, the tail bytes last (review S3): the sum of
    the int32 words (int64, wrapping) and the POSITION-WEIGHTED sum (each word
    times its index in the chunk, wrapping at 2^32). A changed word changes
    the first; two rows swapped inside one chunk leave the first equal and
    change the second. Callers only compare tuples for equality. One host
    sync; the weights are one int32 arange per call, freed with it."""
    n = int(view.numel())
    words = n // 4
    sums = []
    if words:
        w = view[:words * 4].view(torch.int32)
        step = CHECKSUM_CHUNK // 4
        pos = torch.arange(min(step, words), dtype=torch.int32, device=view.device)
        for s in range(0, words, step):
            c = w[s:s + step]
            sums.append(c.sum(dtype=torch.int64))
            sums.append((c * pos[:c.numel()]).sum(dtype=torch.int64))
    if n > words * 4:
        tail = view[words * 4:].to(torch.int64)
        sums.append(tail.sum())
        sums.append((tail * torch.arange(1, int(tail.numel()) + 1, dtype=torch.int64,
                                         device=view.device)).sum())
    return tuple(int(x) for x in torch.stack(sums).tolist()) if sums else ()


class HostImage:
    """The victims' bytes in ONE anonymous, pageable host buffer, for one
    stage (plan §6: never pinned per stage -- cudaHostAlloc of 1.5 GB cost
    8-10 s, draft_park; never /dev/shm -- a dead rank leaves nothing). The
    copies are synchronous: the D2H is complete before anything writes the
    victims, and the H2D before the checksum reads them."""

    def __init__(self, alloc: Callable[[int], torch.Tensor] = None):
        self._alloc = alloc or (lambda n: torch.empty(n, dtype=torch.uint8))
        self.host: Optional[torch.Tensor] = None
        self._spans: List[Tuple[int, int]] = []

    @property
    def nbytes(self) -> int:
        return 0 if self.host is None else int(self.host.numel())

    def stash(self, views: Sequence[torch.Tensor]) -> None:
        spans, off = [], 0
        for v in views:
            spans.append((off, int(v.numel())))
            off += int(v.numel())
        self.host = self._alloc(max(1, off))
        self._spans = spans
        for (o, n), v in zip(spans, views):
            self.host[o:o + n].copy_(v)

    def restore(self, views: Sequence[torch.Tensor]) -> None:
        if self.host is None:
            raise RuntimeError("no host image to restore from")
        for (o, n), v in zip(self._spans, views):
            v.copy_(self.host[o:o + n])

    def release(self) -> None:
        self.host = None
        self._spans = []


# ---------------------------------------------------------------------------
# 5. the source interface, and a base that restores from a host image
# ---------------------------------------------------------------------------


class VictimSource(Protocol):
    kind: str

    @property
    def host_bytes(self) -> int: ...

    def inventory(self) -> List[VictimCandidate]: ...

    def views(self, segments: Sequence[VictimSegment]) -> List[torch.Tensor]: ...

    def verify_before_move(self, views: Sequence[torch.Tensor]) -> None:
        """Raise ``VisionVictimPlanRefused`` (W111b) when the bytes the stage
        will give BACK are not the bytes that are on the device now -- called
        before anything moved (review S2). A source that returns the very
        bytes it stashed has nothing to compare."""
        ...

    def stash(self, views: Sequence[torch.Tensor]) -> None: ...

    def restore(self, views: Sequence[torch.Tensor]) -> None: ...

    def release(self) -> None: ...

    def join(self) -> None: ...

    def shape_tower(self, module: Optional[torch.nn.Module], tower: Sequence[Tuple[str, Tuple[int, ...], int]],
                    largest_run: int) -> SplitMap: ...


def storage_view(t: torch.Tensor) -> torch.Tensor:
    """A uint8 view of ``t``'s WHOLE storage (as ``draft_park.park_population``)."""
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(t.untyped_storage())


class HostImageVictims:
    """Base of a source whose victims return from a host image (D2H before,
    H2D after). A subclass names its tensors: ``_named_storages()`` yields
    (name, uint8 whole-storage view) of the eligible victims."""

    kind = "host"

    def __init__(self, image: Optional[HostImage] = None):
        self._image = image or HostImage()

    # -- to be provided by the line --
    def _named_storages(self) -> List[Tuple[str, torch.Tensor]]:
        raise NotImplementedError

    def shape_tower(self, module, tower, largest_run) -> SplitMap:
        return {}

    def join(self) -> None:
        return None

    def verify_before_move(self, views: Sequence[torch.Tensor]) -> None:
        return None  # the host image IS the device bytes, copied right after

    # -- the interface --
    @property
    def host_bytes(self) -> int:
        return self._image.nbytes

    def inventory(self) -> List[VictimCandidate]:
        out = []
        for n, v in self._named_storages():
            ptr, total = int(v.data_ptr()), int(v.numel())
            off = (-ptr) % SLAB_ALIGN
            if total - off > 0:
                out.append(VictimCandidate(name=n, key=ptr, offset=off, nbytes=total - off,
                                           storage_nbytes=total))
        return out

    def views(self, segments: Sequence[VictimSegment]) -> List[torch.Tensor]:
        live = {int(v.data_ptr()): (n, v) for n, v in self._named_storages()}
        out = []
        for seg in segments:
            hit = live.get(seg.key)
            if hit is None or int(hit[1].numel()) != seg.storage_nbytes:
                raise VisionVictimPlanRefused(
                    f"{W_VICTIM_PLAN_REFUSED}: victim {seg.name} at 0x{seg.key:x} "
                    f"({seg.storage_nbytes} B) is not a live storage of that size any more")
            out.append(hit[1][seg.offset:seg.offset + seg.nbytes])
        return out

    def stash(self, views: Sequence[torch.Tensor]) -> None:
        self._image.stash(views)

    def restore(self, views: Sequence[torch.Tensor]) -> None:
        self._image.restore(views)

    def release(self) -> None:
        self._image.release()


# ---------------------------------------------------------------------------
# 6. the lease: order + invariants
# ---------------------------------------------------------------------------


class PlannedSlab:
    """``SlabAllocator``'s ``take`` over a victim plan: the n-th call returns
    the n-th tower slot's view (tower order = ``named_parameters`` order)."""

    def __init__(self, views: Sequence[torch.Tensor], slots: Sequence[TowerSlot]):
        self._views = list(views)
        self._slots = list(slots)
        self._i = 0

    def take(self, nbytes: int) -> torch.Tensor:
        if self._i >= len(self._slots):
            raise VisionRankStageRefused(f"the victim plan holds {len(self._slots)} tensors, a "
                                         f"{self._i + 1}th was asked for")
        slot = self._slots[self._i]
        if int(nbytes) != slot.nbytes:
            raise VisionRankStageRefused(
                f"victim plan slot {self._i} ({slot.name}) holds {slot.nbytes} B, {nbytes} B asked")
        self._i += 1
        return self._views[slot.seg][slot.offset:slot.offset + slot.nbytes]


class HostProbe:
    """memory.current at the lease's moments, and -- where the kernel offers
    it (cgroup v2 ``memory.peak`` with per-fd reset, Linux >= 6.12) -- the
    exact peak between ``open`` and ``close`` (Release-Notes number: the
    host-RAM peak of the displacement)."""

    def __init__(self, cgroup_dir: Optional[str] = None):
        self._dir = cgroup_dir if cgroup_dir is not None else _own_cgroup_dir()
        self._peak_fd: Optional[int] = None
        self.marks: Dict[str, Tuple[float, Optional[int]]] = {}

    def start(self) -> None:
        if self._dir is None:
            return
        try:
            fd = os.open(os.path.join(self._dir, "memory.peak"), os.O_RDWR)
        except OSError:
            return
        try:
            os.write(fd, b"reset\n")
            self._peak_fd = fd
        except OSError:
            os.close(fd)

    def mark(self, name: str) -> None:
        self.marks[name] = (time.time(), self._current())

    def peak(self) -> Optional[int]:
        if self._peak_fd is None:
            return None
        try:
            return int(os.pread(self._peak_fd, 64, 0).strip())
        except (OSError, ValueError):
            return None
        finally:
            os.close(self._peak_fd)
            self._peak_fd = None

    def _current(self) -> Optional[int]:
        if self._dir is None:
            return None
        try:
            with open(os.path.join(self._dir, "memory.current")) as fh:
                return int(fh.read())
        except (OSError, ValueError):
            return None


def _own_cgroup_dir() -> Optional[str]:
    try:
        with open("/proc/self/cgroup") as fh:
            rel = fh.read().strip().split("::", 1)[-1]
        d = "/sys/fs/cgroup" + rel
        return d if os.path.isdir(d) else None
    except OSError:
        return None


def tower_of(module: torch.nn.Module) -> List[Tuple[str, Tuple[int, ...], int]]:
    """(name, shape, nbytes) of every parameter, in ``named_parameters`` order."""
    return [(n, tuple(int(x) for x in p.shape), int(p.numel() * p.element_size()))
            for n, p in module.named_parameters()]


class VictimLease:
    """One stage's borrow of victim weight memory. Build with :meth:`open`,
    end with :meth:`close` on every path."""

    def __init__(self, source: VictimSource, plan: VictimPlan, views: List[torch.Tensor],
                 sums0: List[Tuple[int, ...]], probe: HostProbe, split: SplitMap, legs: Dict[str, float]):
        self.source = source
        self.plan = plan
        self.views = views
        self.sums0 = sums0
        self.probe = probe
        self.split = split
        self.legs_ms = legs
        self.stashed_bytes = source.host_bytes
        self.host_peak: Optional[int] = None
        self.closed = False

    @classmethod
    def open(cls, source: VictimSource, module: torch.nn.Module, *,
             checksum: Callable[[torch.Tensor], Tuple[int, ...]] = segment_checksum,
             clock: Callable[[], float] = time.perf_counter,
             probe: Optional[HostProbe] = None) -> "VictimLease":
        """Borrow victims for the (meta-built) ``module``. The source may
        reshape the tower first (``shape_tower``: AP2's row split of a
        tensor larger than every victim run); the plan follows the module's
        parameter order afterwards, which is the order ``place_parameters``
        takes the slots in."""
        legs: Dict[str, float] = {}
        t0 = clock()
        source.join()
        inv = source.inventory()
        largest = max((c.nbytes for c in inv), default=0)
        split = source.shape_tower(module, tower_of(module), largest)
        plan = plan_victims(inv, [(n, b) for n, _s, b in tower_of(module)])
        views = source.views(plan.segments)
        legs["victim_plan"] = (clock() - t0) * 1e3
        t0 = clock()
        sums0 = [checksum(v) for v in views]
        legs["checksum"] = (clock() - t0) * 1e3
        t0 = clock()
        try:
            source.verify_before_move(views)  # W111b BEFORE a byte moves (review S2)
        except Exception:
            source.release()
            raise
        legs["store_check"] = (clock() - t0) * 1e3
        probe = probe or HostProbe()
        probe.start()
        probe.mark("before")
        t0 = clock()
        try:
            source.stash(views)
        except (MemoryError, RuntimeError) as exc:
            # plan §6.8: before any device byte moved -- W105b, rig intact
            source.release()
            probe.peak()
            raise VisionVictimShort(
                f"{W_VICTIM_SHORT}: the host image of {sum(int(v.numel()) for v in views) / MIB:.1f} MiB "
                f"could not be written ({type(exc).__name__}: {exc})") from exc
        legs["stash"] = (clock() - t0) * 1e3
        probe.mark("after_stash")
        return cls(source, plan, views, sums0, probe, dict(split or {}), legs)

    def slab(self) -> PlannedSlab:
        return PlannedSlab(self.views, self.plan.slots)

    def close(self, *, sync: Optional[Callable[[], None]] = None,
              checksum: Callable[[torch.Tensor], Tuple[int, ...]] = segment_checksum,
              clock: Callable[[], float] = time.perf_counter) -> Optional[str]:
        """Return the victims. None = restored and verified; else the W110c
        detail (the caller makes it fatal). Never raises."""
        if self.closed:
            return None
        self.closed = True
        why = None
        t0 = clock()
        try:
            if sync is not None:
                sync()  # nothing reads the tower views any more
            self.source.restore(self.views)
        except Exception as exc:  # noqa: BLE001 -- named, fatal at the caller
            why = f"restore raised {type(exc).__name__}: {exc}"
        self.legs_ms["restore"] = (clock() - t0) * 1e3
        t0 = clock()
        if why is None:
            try:
                bad = [self.plan.segments[i].name for i, v in enumerate(self.views)
                       if checksum(v) != self.sums0[i]]
            except Exception as exc:  # noqa: BLE001
                bad, why = [], f"verify raised {type(exc).__name__}: {exc}"
            if bad:
                why = (f"checksum MISMATCH on {len(bad)} of {len(self.views)} victims "
                       f"({', '.join(bad[:3])}{' ...' if len(bad) > 3 else ''})")
        self.legs_ms["verify"] = (clock() - t0) * 1e3
        try:
            self.source.release()
        except Exception as exc:  # noqa: BLE001 -- a host buffer, never fatal
            logger.warning("%s host image release raised %s: %s", W_VICTIM_HOST, type(exc).__name__, exc)
        self.probe.mark("after_release")
        self.host_peak = self.probe.peak()
        self.views = []
        return None if why is None else f"{W_VICTIM_NOT_RESTORED}: {why}"

    def fields(self) -> str:
        """The W102 additions of a weights stage."""
        segs = self.plan.segments
        return (f"victim={self.source.kind} victim_mib={self.plan.victim_bytes / MIB:.1f} "
                f"segments={len(segs)} largest_run_mib={self.plan.largest_run / MIB:.1f} "
                f"split_tensors={len(self.split)} host_image_mib={self.stashed_bytes / MIB:.1f}")

    def host_line(self, run: int) -> str:
        """Coordinator 09.10. (3): the host-RAM price of the displacement,
        read by the rank itself (cgroup memory.current; memory.peak fd-reset
        window where the kernel has it)."""
        def cur(k):
            v = self.probe.marks.get(k, (0.0, None))[1]
            return "n/a" if v is None else f"{v / MIB:.1f}"

        def ts(k):
            t = self.probe.marks.get(k)
            return "n/a" if t is None else str(int(t[0] * 1000))

        peak = "n/a" if self.host_peak is None else f"{self.host_peak / MIB:.1f}"
        return (f"{W_VICTIM_HOST} run={run} victim={self.source.kind} stashed_bytes={self.stashed_bytes} "
                f"host_current_mib=(before {cur('before')}, after_stash {cur('after_stash')}, "
                f"after_release {cur('after_release')}) host_peak_mib={peak} "
                f"t_unix_ms=(before {ts('before')}, after_stash {ts('after_stash')}, "
                f"after_release {ts('after_release')})")


# ---------------------------------------------------------------------------
# 7. the encoder's work memory, from the REAL image (pure)
# ---------------------------------------------------------------------------


def item_patches(item: Any) -> int:
    """Patches of one item: sum over its grid rows of t*h*w."""
    grid = torch.as_tensor(item.image_grid_thw).reshape(-1, 3)
    return int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())


def item_segments(item: Any) -> int:
    """Attention segments of one item: sum over its grid rows of t (the
    tower's cu_seqlens repeats h*w once per frame). A still image is one."""
    grid = torch.as_tensor(item.image_grid_thw).reshape(-1, 3)
    return int(grid[:, 0].sum().item())


def encode_work_bytes(patches: int, *, hidden: int, intermediate: int, heads: int, out_hidden: int,
                      merge: int, deepstack: int, in_dim: int, quadratic: bool, elem: int = 2) -> int:
    """Peak bytes the tower's forward allocates for ONE item of ``patches``
    patches (``encode_items`` runs one item at a time). A model of the
    Qwen3-VL-family forward (``models/qwen3_vl.py``), named term by term: what
    lives for the whole forward, plus the largest of its phases. The measured
    ``encode_peak_mib`` of W102 is what calibrates it (plan §9 M1; metal 10.10.
    before VISION-WORK: 1213/2150 MiB at 3072/4096 = exactly the old forward's
    live tensors, 12.47 hidden in the MLP phase + pixels + pos rows + rope).

    Held for the whole forward:

    * inputs   -- the pixel rows on the device (``in_dim`` = C*T*P*P);
                  ``encode_items`` holds them across the call
    * rope     -- the rope cos/sin rows (fp32, half the head width each)

    The phases (``h`` = one hidden-wide row block, ``i`` = intermediate):

    * embed    -- position interpolation: the patch rows, the four corner
                  lookups, their sum and its permuted copy = 7 h (the
                  pos rows are dropped once added, VISION-WORK)
    * attn     -- residual, norm1, fused qkv (3 h) and the contiguous q/k/v
                  copies (3 h) = 8 h; after ``del qkv`` the rotation holds
                  residual, norm1, q/k/v, rotated q/k (7 h) plus the
                  full-width fp32 cos/sin
    * mlp      -- residual, norm2, fc1 out, act out = 2 h + 2 i (fc1 is
                  dropped before fc2: residual, norm2, act, fc2 out =
                  3 h + i); norm1 and the attention output are dropped
                  before the MLP (VISION-WORK)
    * block    -- the larger of attn and mlp, plus the deepstack rows the
                  earlier mergers left
    * merger   -- residual, norm, fc1, act (4 h) with the deepstack rows,
                  or the merged rows and their concatenation, the larger
    * quad     -- ``quadratic`` (sdpa AND more than one segment, see
                  ``encode_work_for``): the block-diagonal mask on the device,
                  bool + its additive bf16 form = 3 B per pair. One segment
                  (a still image) runs sdpa without a mask
                  (``layers/attention/vision.py:_sdpa_single_segment``), so no
                  pair term
    """
    p = int(patches)
    head_dim = max(1, hidden // max(1, heads))
    h = p * hidden * elem
    inputs = p * in_dim * elem
    rope = 2 * p * (head_dim // 2) * 4
    embed = 7 * h
    attn = max(8 * h, 7 * h + 2 * p * head_dim * 4)
    mlp = p * elem * max(2 * hidden + 2 * intermediate, 3 * hidden + intermediate)
    merged = (p // max(1, merge * merge)) * out_hidden * elem
    block = max(attn, mlp) + deepstack * merged
    merger = max(4 * h + deepstack * merged, (1 + deepstack) * merged * 2)
    quad = 3 * p * p if quadratic else 0
    return int(inputs + rope + max(embed, block, merger) + quad)


def encode_work_for(vision_config: Any, items: Sequence[Any], backend: Optional[str]) -> int:
    """The largest per-item work of ``items`` under ``vision_config``. The
    pair term only for sdpa with more than one segment in the item: one
    segment drops the mask (``_sdpa_single_segment``)."""
    vc = vision_config
    in_dim = (int(getattr(vc, "in_channels", 3)) * int(getattr(vc, "temporal_patch_size", 2))
              * int(vc.patch_size) ** 2)
    kw = dict(hidden=int(vc.hidden_size), intermediate=int(vc.intermediate_size),
              heads=int(vc.num_heads), out_hidden=int(vc.out_hidden_size),
              merge=int(vc.spatial_merge_size), deepstack=len(getattr(vc, "deepstack_visual_indexes", ()) or ()),
              in_dim=in_dim)
    sdpa = (backend or "sdpa") == "sdpa"
    return max((encode_work_bytes(item_patches(it), quadratic=sdpa and item_segments(it) > 1, **kw)
                for it in items), default=0)


def encode_peak_start(device: torch.device) -> Optional[int]:
    """Open a private allocator-peak window around the encode (M1's
    ``encode_peak_mib``), folding the cumulative peak into the
    ``vram_peak_window`` shadow first so ``[vram-peak]`` keeps "since the
    pools". Returns the allocated bytes at the start (None off CUDA)."""
    if device.type != "cuda":
        return None
    try:
        from flliper.srt.model_executor.vram_peak_window import fold_and_rebase

        return fold_and_rebase(torch.cuda)
    except Exception as exc:  # noqa: BLE001 -- an instrument, never a stage failure
        logger.debug("encode peak window not opened: %s", exc)
        return None


def encode_peak_since(device: torch.device, start: int) -> Optional[int]:
    try:
        return int(torch.cuda.max_memory_allocated(device)) - int(start)
    except Exception:  # noqa: BLE001
        return None


def encode_air_refusal(work: int, card_free: int, cache_idle: int, patches: int) -> str:
    """'' when the encode's work memory fits the card's air, else the W105b
    text with the numbers (never a silent cut of the image)."""
    air = int(card_free) + max(0, int(cache_idle))
    if int(work) <= air:
        return ""
    return (f"{W_VICTIM_SHORT}: the encode of the largest item ({patches} patches) needs "
            f"{work / MIB:.0f} MiB of work memory, the card's air is {air / MIB:.0f} MiB "
            f"(free {card_free / MIB:.0f} + idle cache {max(0, int(cache_idle)) / MIB:.0f}); the "
            "image is not cut")


# ---------------------------------------------------------------------------
# 8. the sources per line, and the arming (M0)
# ---------------------------------------------------------------------------

#: modules that register a source (``register_source``) at import; a line
#: adds its own (NF: its expert-row source)
SOURCE_MODULES: Tuple[str, ...] = ("flliper.srt.pdflip.vision_victim_27b",)

#: (kind, applies(scheduler) -> bool, build(scheduler) -> VictimSource), in order
_SOURCES: List[Tuple[str, Callable[[Any], bool], Callable[[Any], VictimSource]]] = []


def register_source(kind: str, applies: Callable[[Any], bool], build: Callable[[Any], VictimSource]) -> None:
    for i, (k, _a, _b) in enumerate(_SOURCES):
        if k == kind:
            _SOURCES[i] = (kind, applies, build)
            return
    _SOURCES.append((kind, applies, build))


def resolve_source(scheduler) -> VictimSource:
    """The first registered source that applies to this rank, or W111b."""
    import importlib

    for mod in SOURCE_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            logger.warning("%s: victim source module %s not importable (%s)", W_VICTIM_PLAN_REFUSED, mod, exc)
    for kind, applies, build in _SOURCES:
        if applies(scheduler):
            return build(scheduler)
    raise VisionVictimPlanRefused(
        f"{W_VICTIM_PLAN_REFUSED}: no victim source applies to this rank "
        f"(registered: {[k for k, _a, _b in _SOURCES] or 'none'})")


def arming_line(source: VictimSource, ckpt: Sequence[CkptTensor],
                map_name: Callable[[str], str]) -> Tuple[str, str]:
    """(line, refusal): the M0 arming line -- the inventory and the plan the
    stage WILL make for the checkpoint's tower (sizes from the header, names
    mapped to the module's, the split the source asks for) -- or the
    W105b refusal."""
    inv = source.inventory()
    largest = max((c.nbytes for c in inv), default=0)
    adjacent = largest_adjacent_run(inv)
    tower = [(map_name(c.name), tuple(int(x) for x in c.shape), c.nbytes) for c in ckpt]
    split = source.shape_tower(None, tower, largest)
    pieces = []
    for name, shape, n in tower:
        if name in split:
            row = n // shape[0]
            pieces.extend((p, rows * row) for p, rows in split[name])
        else:
            pieces.append((name, n))
    head = (f"{W_VICTIM_ARMED} victim={source.kind} inventory_mib={sum(c.nbytes for c in inv) / MIB:.1f} "
            f"candidates={len(inv)} largest_run_mib={largest / MIB:.1f} "
            f"largest_adjacent_mib={adjacent / MIB:.1f} tower_mib={sum(n for _, n in pieces) / MIB:.1f} "
            f"tower_tensors={len(tower)} split_tensors={len(split)}")
    try:
        plan = plan_victims(inv, pieces)
    except VisionVictimShort as exc:
        return f"{head} plan=REFUSED", str(exc)
    # H2: only a host-image source stashes the victims; the expert rows are
    # already in the store (host_image_mib=0 at every stage)
    image = plan.victim_bytes if isinstance(source, HostImageVictims) else 0
    return (f"{head} planned_victim_mib={plan.victim_bytes / MIB:.1f} planned_segments={len(plan.segments)} "
            f"planned_host_image_mib={image / MIB:.1f}"), ""
