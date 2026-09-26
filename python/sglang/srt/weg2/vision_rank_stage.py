# SPDX-License-Identifier: Apache-2.0
"""The transient vision stage INSIDE the P rank (user design 2026-09-24).

Replaces the tokenizer-process stage of Task #58 (own CUDA context, card
census, band displacement into host RAM). Design, as approved by the user:

* The stage runs in P's PP0 rank process -- the rank whose forward embeds the
  tokens, so the rows land where they are consumed and no second CUDA context
  exists.
* PLACE: the context region. After a D->P flip P's KV region is free by
  construction (flush_cache before the pause at the sleep, allocator.clear at
  the wake), so the tower's parameters are VIEWS into a TAIL range of free KV
  slots -- taken out of the allocator's free list before the first admission
  and handed back afterwards. No displacement, no host-RAM post, no extra
  VRAM booking.
* SOURCE: the tower bytes are ALWAYS read from disk, O_DIRECT (no page cache,
  no ARC), through two 32 MiB pinned bounce buffers, H2D straight into the
  views; afterwards the bounce buffers are freed and the pinned host cache is
  emptied, so memory.current/anon return to their baseline.
* The encoder's activations come from the default allocator -- the rank's
  booked prefill transient, idle before the first prefill -- and are handed
  back with ``empty_cache`` after the encode.

This module is the MODEL-NEUTRAL half: tail reservation, KV byte segments, a
bump allocator over them, the meta-built module placed on the views, and the
direct reader. Nothing here names a model; the tower spec comes from the
model's own config and checkpoint header. The scheduler wiring lives with the
caller.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import struct
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

MIB = 1 << 20

#: O_DIRECT alignment on this rig (OpenZFS 2.3.4, real direct I/O): offset,
#: length and buffer address all on 4 KiB -- the same constant the existing
#: direct readers use (``weight_utils._DIRECT_ALIGN``).
DIRECT_ALIGN = 4096
#: Two bounce buffers of 32 MiB each (user design): one fills from disk while
#: the other drains to the device.
BOUNCE_BYTES = 32 * MIB
BOUNCE_COUNT = 2
#: Alignment of every tensor placed on the slab. 256 B covers every dtype and
#: the vectorised loads of the kernels that read the tower's weights.
SLAB_ALIGN = 256


class VisionRankStageRefused(RuntimeError):
    """A named reason the in-rank stage cannot run; the rig is intact."""


# ---------------------------------------------------------------------------
# 1. the TAIL of the KV slot space, out of the free list and back
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailReservation:
    """Pages ``[lo_page, hi_page]`` (inclusive) held out of the free list."""

    lo_page: int
    hi_page: int
    page_size: int

    @property
    def pages(self) -> int:
        return self.hi_page - self.lo_page + 1

    @property
    def slots(self) -> int:
        return self.pages * self.page_size


def _allocator_num_pages(allocator) -> int:
    n = getattr(allocator, "num_pages", None)
    if n is not None:
        return int(n)
    return int(allocator.size) // int(getattr(allocator, "page_size", 1) or 1)


def reserve_tail_pages(allocator, n_pages: int) -> Optional[TailReservation]:
    """Take the last ``n_pages`` pages out of ``allocator``'s free list, or
    return None when any of them is in use.

    The allocator hands out pages from the FRONT of its free list and the
    wake clears it to ``arange(1, num_pages + 1)``, so the tail is the last
    region a prefill reaches: free at a phase start by construction, and free
    mid-phase unless the pool is nearly full. Only whole-tail reservations
    are made -- a partially used tail is a named refusal of the caller, never
    a fragmented placement. Page 0 is the padded slot and never part of it.
    No free-listener is notified: nothing is allocated for a request and the
    pages go back unchanged, so the #1243 tail ring and the owner bias see no
    event.
    """
    n_pages = int(n_pages)
    num_pages = _allocator_num_pages(allocator)
    if n_pages <= 0 or n_pages > num_pages:
        return None
    free = allocator.free_pages
    rel = getattr(allocator, "release_pages", None)
    if rel is not None and rel.numel() > 0:
        free = torch.cat((free, rel))
    lo = num_pages - n_pages + 1
    in_tail = free >= lo
    if int(in_tail.sum().item()) != n_pages:
        return None
    allocator.free_pages = free[~in_tail]
    if rel is not None:
        allocator.release_pages = rel.new_empty((0,))
    if getattr(allocator, "need_sort", False):
        allocator.free_pages, _ = torch.sort(allocator.free_pages)
    return TailReservation(lo_page=lo, hi_page=num_pages,
                           page_size=int(getattr(allocator, "page_size", 1) or 1))


def return_tail_pages(allocator, res: TailReservation) -> None:
    """Hand the reserved pages back AT THE END of the free list, so the tail
    stays the last region a prefill reaches."""
    pages = torch.arange(res.lo_page, res.hi_page + 1, dtype=allocator.free_pages.dtype,
                         device=allocator.free_pages.device)
    allocator.free_pages = torch.cat((allocator.free_pages, pages))


# ---------------------------------------------------------------------------
# 2. the tail's bytes, per KV buffer
# ---------------------------------------------------------------------------


def attention_kv_buffers(pool) -> List[torch.Tensor]:
    """This rank's full-attention K and V buffers, walked through the hybrid
    wrappers (``full_kv_pool``). Refuses a pool that exposes none."""
    seen = pool
    for _ in range(4):
        k = getattr(seen, "k_buffer", None)
        v = getattr(seen, "v_buffer", None)
        if isinstance(k, (list, tuple)) and isinstance(v, (list, tuple)) and k:
            return [t for pair in zip(k, v) for t in pair]
        nxt = getattr(seen, "full_kv_pool", None)
        if nxt is None:
            break
        seen = nxt
    raise VisionRankStageRefused(
        f"{type(pool).__name__} exposes no full-attention k_buffer/v_buffer lists; "
        "the tower has no KV bytes to be placed on"
    )


def tail_segments(buffers: Sequence[torch.Tensor], res: TailReservation,
                  num_pages: int) -> List[torch.Tensor]:
    """One contiguous ``uint8`` view per buffer, covering the reserved pages.

    Layout-agnostic: dim 0 of a KV buffer is either rows (``size + page``) or
    pages (``num_pages + 1``, the vectorised layout); both have
    ``num_pages + 1`` equal dim-0 groups per page, so the tail pages are the
    dim-0 range ``[lo_page * g, (hi_page + 1) * g)``.
    """
    segs = []
    for buf in buffers:
        if not buf.is_contiguous():
            raise VisionRankStageRefused(f"a KV buffer of shape {tuple(buf.shape)} is not contiguous")
        d0 = int(buf.shape[0])
        if d0 % (num_pages + 1) != 0:
            raise VisionRankStageRefused(
                f"KV buffer dim 0 = {d0} is not a multiple of num_pages + 1 = {num_pages + 1}"
            )
        g = d0 // (num_pages + 1)
        part = buf[res.lo_page * g:(res.hi_page + 1) * g]
        segs.append(part.view(torch.uint8).reshape(-1))
    return segs


def bytes_per_slot(buffers: Sequence[torch.Tensor], num_pages: int, page_size: int) -> int:
    """KV bytes one slot spans over all ``buffers``."""
    total = 0
    for buf in buffers:
        total += buf.numel() * buf.element_size() // ((num_pages + 1) * page_size)
    return int(total)


class SlabAllocator:
    """Bump allocator over byte segments. A tensor never straddles two."""

    def __init__(self, segments: Sequence[torch.Tensor], align: int = SLAB_ALIGN):
        self.segments = list(segments)
        self.align = int(align)
        self._seg = 0
        self._off = 0

    def take(self, nbytes: int) -> torch.Tensor:
        nbytes = int(nbytes)
        while self._seg < len(self.segments):
            seg = self.segments[self._seg]
            start = (self._off + self.align - 1) // self.align * self.align
            if start + nbytes <= seg.numel():
                self._off = start + nbytes
                return seg[start:start + nbytes]
            self._seg += 1
            self._off = 0
        raise VisionRankStageRefused(
            f"the reserved KV tail cannot hold a {nbytes}-byte tensor "
            f"({len(self.segments)} segment(s) exhausted)"
        )


def slots_for(named_nbytes: Iterable[int], buffers: Sequence[torch.Tensor], num_pages: int,
              page_size: int, *, align: int = SLAB_ALIGN) -> int:
    """Pages the tail must span to hold tensors of these sizes, placed by
    :class:`SlabAllocator` (first fit, in order, never straddling). Computed
    by simulation, so the answer is exactly what the placement will do."""
    sizes = [int(n) for n in named_nbytes]
    per_page = [buf.numel() * buf.element_size() // (num_pages + 1) for buf in buffers]
    total = sum(sizes)
    pages = max(1, total // max(1, sum(per_page)))
    while pages <= num_pages:
        caps = [p * pages for p in per_page]
        seg, off, ok = 0, 0, True
        for n in sizes:
            while seg < len(caps):
                start = (off + align - 1) // align * align
                if start + n <= caps[seg]:
                    off = start + n
                    break
                seg, off = seg + 1, 0
            else:
                ok = False
                break
        if ok:
            return pages
        pages = int(pages * 1.05) + 1
    raise VisionRankStageRefused(
        f"the tower ({total / MIB:.1f} MiB) does not fit into this rank's whole KV pool"
    )


# ---------------------------------------------------------------------------
# 3. the module: built on meta, parameters placed on the slab
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def params_on_meta():
    """Every parameter registered inside this context lives on ``meta``.

    Buffers stay where the module puts them (a rope's cos/sin cache is
    computed at construction and must be real). A parameter is created by
    its layer first and moved to meta at registration, so the transient per
    parameter is one tensor at a time -- the largest tower tensor (the 27B
    merger's fc2, 45 MiB).
    """
    orig = torch.nn.Module.register_parameter
    owner = threading.get_ident()

    def register_parameter(module, name, param):
        # Scoped to the building thread: the class attribute is process-wide,
        # and a module another thread builds meanwhile must stay untouched.
        if threading.get_ident() == owner and param is not None and param.device.type != "meta":
            param = torch.nn.Parameter(param.to("meta"), requires_grad=False)
        return orig(module, name, param)

    torch.nn.Module.register_parameter = register_parameter
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = orig


def place_parameters(module: torch.nn.Module, slab: SlabAllocator) -> Dict[str, torch.Tensor]:
    """Replace every (meta) parameter of ``module`` by a typed view on the
    slab (``load_state_dict(assign=True)``). Returns name -> view."""
    views: Dict[str, torch.Tensor] = {}
    for name, p in module.named_parameters():
        nbytes = p.numel() * p.element_size()
        raw = slab.take(nbytes)
        views[name] = raw.view(p.dtype).view(p.shape)
    missing, unexpected = module.load_state_dict(views, strict=False, assign=True)
    params = {n for n, _ in module.named_parameters()}
    left_meta = [n for n, p in module.named_parameters() if p.device.type == "meta"]
    if left_meta or (set(missing) & params) or unexpected:
        raise VisionRankStageRefused(
            f"placement left parameters unplaced: meta={left_meta[:3]} "
            f"missing={[m for m in missing if m in params][:3]} unexpected={list(unexpected)[:3]}"
        )
    meta_bufs = [n for n, b in module.named_buffers() if b.device.type == "meta"]
    if meta_bufs:
        raise VisionRankStageRefused(f"buffers left on meta: {meta_bufs[:3]}")
    return views


# ---------------------------------------------------------------------------
# 4. the checkpoint: header offsets, and the direct read into the views
# ---------------------------------------------------------------------------

_ST_DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8, "I8": torch.int8,
}


@dataclass(frozen=True)
class CkptTensor:
    name: str
    dtype: torch.dtype
    shape: Tuple[int, ...]
    file_offset: int
    nbytes: int


def checkpoint_tensors(shard: str, selector: Callable[[str], bool]) -> List[CkptTensor]:
    """Header-only: the selected tensors with their absolute file offsets."""
    with open(shard, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
    base = 8 + hlen
    out = []
    for name, meta in header.items():
        if name == "__metadata__" or not selector(name):
            continue
        s, e = (int(x) for x in meta["data_offsets"])
        dt = _ST_DTYPES.get(str(meta["dtype"]))
        if dt is None:
            raise VisionRankStageRefused(f"{name}: unsupported safetensors dtype {meta['dtype']}")
        out.append(CkptTensor(name, dt, tuple(int(x) for x in meta["shape"]), base + s, e - s))
    out.sort(key=lambda t: t.file_offset)
    return out


def _alloc_bounce(nbytes: int, pinned: bool) -> torch.Tensor:
    """A bounce buffer whose ADDRESS is ``DIRECT_ALIGN``-aligned (O_DIRECT
    needs it; cudaHostAlloc is page-aligned anyway, a plain CPU tensor is
    not)."""
    raw = torch.empty(nbytes + DIRECT_ALIGN, dtype=torch.uint8, pin_memory=pinned)
    off = (-raw.data_ptr()) % DIRECT_ALIGN
    return raw[off:off + nbytes]


def _open_direct(path: str) -> Tuple[int, bool]:
    """(fd, is_direct). O_DIRECT first; a filesystem that refuses it (EINVAL)
    falls back to a buffered fd, NAMED in the log -- never silent."""
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECT), True
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.EOPNOTSUPP):
            raise
        logger.warning(
            "vision rank stage: O_DIRECT refused on %s (%s) -- BUFFERED read; the "
            "bytes may stay in the page cache/ARC", path, exc)
        return os.open(path, os.O_RDONLY), False


@dataclass
class DirectReadReport:
    bytes_read: int = 0
    chunks: int = 0
    direct: bool = False
    pinned: bool = False
    host_cache_emptied: bool = False


def read_into(shard: str, plan: Sequence[Tuple[CkptTensor, torch.Tensor]], *,
              bounce_bytes: int = BOUNCE_BYTES, bounce_count: int = BOUNCE_COUNT,
              stream: Optional[Any] = None, direct: bool = True) -> DirectReadReport:
    """Read the planned checkpoint tensors straight into their destination
    tensors (the slab views), through ``bounce_count`` bounce buffers.

    The union of the tensors' byte ranges is walked in aligned chunks; each
    chunk is read with ``preadv`` into a bounce buffer, and every destination
    byte range inside that chunk is copied out of it (asynchronously on
    ``stream`` when the destination is on a CUDA device). A bounce buffer is
    reused only after the copies out of it completed (an event per buffer).
    Afterwards the bounce buffers are dropped and, on CUDA, the pinned host
    cache is emptied -- the host footprint of the whole read is the bounce
    buffers, for the duration of the read.

    ``direct=False`` (H125, the RAM source): the file is a tmpfs image of the
    tower extent, which refuses O_DIRECT by construction; it is opened
    buffered on purpose and without the fallback warning, and the report
    says ``direct=False``.
    """
    rep = DirectReadReport()
    if not plan:
        return rep
    for ck, dst in plan:
        if dst.numel() * dst.element_size() != ck.nbytes:
            raise VisionRankStageRefused(
                f"{ck.name}: checkpoint holds {ck.nbytes} bytes, destination "
                f"{tuple(dst.shape)}x{dst.dtype} holds {dst.numel() * dst.element_size()}"
            )
    cuda = any(d.is_cuda for _, d in plan)
    rep.pinned = bool(cuda and torch.cuda.is_available())
    start = min(ck.file_offset for ck, _ in plan) // DIRECT_ALIGN * DIRECT_ALIGN
    end = max(ck.file_offset + ck.nbytes for ck, _ in plan)
    size = os.path.getsize(shard)
    bufs = [_alloc_bounce(bounce_bytes, rep.pinned) for _ in range(bounce_count)]
    events: List[Optional[Any]] = [None] * bounce_count
    if direct:
        fd, rep.direct = _open_direct(shard)
    else:
        fd, rep.direct = os.open(shard, os.O_RDONLY), False
    ordered = sorted(plan, key=lambda p: p[0].file_offset)
    try:
        pos, i = start, 0
        while pos < end:
            slot = i % bounce_count
            if events[slot] is not None:
                events[slot].synchronize()
            want = min(bounce_bytes, end - pos)
            if rep.direct:
                want = (want + DIRECT_ALIGN - 1) // DIRECT_ALIGN * DIRECT_ALIGN
                want = min(want, (size - pos + DIRECT_ALIGN - 1) // DIRECT_ALIGN * DIRECT_ALIGN)
            view = memoryview(bufs[slot].numpy())[:want]
            got = os.preadv(fd, [view], pos)
            if got <= 0:
                raise VisionRankStageRefused(f"{shard}: read returned {got} at {pos}")
            chunk_end = pos + got
            for ck, dst in ordered:
                a = max(ck.file_offset, pos)
                b = min(ck.file_offset + ck.nbytes, chunk_end)
                if a >= b:
                    continue
                src = bufs[slot][a - pos:b - pos]
                flat = dst.view(torch.uint8).reshape(-1)[a - ck.file_offset:b - ck.file_offset]
                if flat.is_cuda:
                    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                        flat.copy_(src, non_blocking=True)
                else:
                    flat.copy_(src)
            if cuda and torch.cuda.is_available():
                ev = torch.cuda.Event()
                ev.record(stream if stream is not None else torch.cuda.current_stream())
                events[slot] = ev
            rep.bytes_read += got
            rep.chunks += 1
            pos = chunk_end
            i += 1
    finally:
        os.close(fd)
        for ev in events:
            if ev is not None:
                ev.synchronize()
        del bufs
        if rep.pinned:
            try:
                torch._C._host_emptyCache()
                rep.host_cache_emptied = True
            except Exception as exc:  # noqa: BLE001 -- reported, never faked
                logger.warning("vision rank stage: pinned host cache not emptied (%s)", exc)
    return rep


def plan_checkpoint_into(views: Dict[str, torch.Tensor], tensors: Sequence[CkptTensor],
                         map_name: Callable[[str], str]) -> List[Tuple[CkptTensor, torch.Tensor]]:
    """Pair checkpoint tensors with the module's views. Refuses an unmapped
    checkpoint name and an unfilled view -- an unfilled block encodes garbage."""
    plan, filled = [], set()
    for ck in tensors:
        name = map_name(ck.name)
        dst = views.get(name)
        if dst is None:
            raise VisionRankStageRefused(f"checkpoint tensor {ck.name!r} -> {name!r} has no view")
        if tuple(dst.shape) != tuple(ck.shape) or dst.dtype != ck.dtype:
            raise VisionRankStageRefused(
                f"{ck.name}: checkpoint {ck.shape}x{ck.dtype} vs module {tuple(dst.shape)}x{dst.dtype}"
            )
        plan.append((ck, dst))
        filled.add(name)
    unfilled = sorted(set(views) - filled)
    if unfilled:
        raise VisionRankStageRefused(f"{len(unfilled)} parameter(s) have no checkpoint tensor: {unfilled[:3]}")
    return plan


# ---------------------------------------------------------------------------
# 5. H125 (NF): the SOURCE of the tower bytes -- disk or RAM, selectable
# ---------------------------------------------------------------------------
#
# User order 2026-09-24 10:15Z: the tower exists on NF only transiently and
# its source is selectable, RAM or disk. ``disk`` (default) is the 27B
# stage's reader unchanged: the checkpoint shard, O_DIRECT, no page cache.
# ``ram`` stages the tower's byte extent ONCE into a tmpfs file (one
# contiguous extent on both checkpoints: NF 897,862,112 B in
# model-00014-of-00014, 27B 921,460,192 B) and every stage reads that file
# buffered -- a memcpy from host RAM instead of a disk read. The price is
# named, never hidden: the extent stays in host RAM (shmem, charged to the
# writer's memory cgroup) for the life of the file. The file is keyed by
# the shard's path, size and mtime, so a later boot REUSES it instead of
# writing a second copy -- one image per checkpoint is the bound.

SOURCE_DISK = "disk"
SOURCE_RAM = "ram"
SOURCES = (SOURCE_DISK, SOURCE_RAM)
SOURCE_ENV = "SGLANG_WEG2_VISION_SOURCE"
RAM_DIR_ENV = "SGLANG_WEG2_VISION_RAM_DIR"
DEFAULT_RAM_DIR = "/dev/shm/weg2-vision"
#: staging read size (the shard is read once, sequentially)
RAM_STAGE_CHUNK = 64 * MIB


def vision_source(env: Optional[Dict[str, str]] = None) -> str:
    """``SGLANG_WEG2_VISION_SOURCE``: unset/empty = ``disk``. An unknown
    value is REFUSED -- a typo must not silently pick a source."""
    e = os.environ if env is None else env
    raw = (e.get(SOURCE_ENV, "") or "").strip().lower()
    if not raw:
        return SOURCE_DISK
    if raw not in SOURCES:
        raise VisionRankStageRefused(
            f"{SOURCE_ENV}={raw!r} is not one of {list(SOURCES)}")
    return raw


def ram_dir(env: Optional[Dict[str, str]] = None) -> str:
    e = os.environ if env is None else env
    return (e.get(RAM_DIR_ENV, "") or "").strip() or DEFAULT_RAM_DIR


@dataclass(frozen=True)
class TowerSource:
    """Where the stage reads the tower bytes from.

    ``shift`` is subtracted from every checkpoint offset (0 for the shard
    itself, the extent's first byte for a RAM image). ``host_bytes`` is what
    the source holds in host RAM for as long as it exists (0 for disk)."""

    kind: str
    path: str
    shift: int
    direct: bool
    host_bytes: int
    reused: bool = False


def disk_source(shard: str) -> TowerSource:
    return TowerSource(SOURCE_DISK, shard, 0, True, 0)


def ram_image_path(shard: str, directory: str) -> str:
    base = os.path.basename(os.path.dirname(os.path.abspath(shard))) or "model"
    return os.path.join(directory, f"{base}__{os.path.basename(shard)}.tower")


def _image_meta(shard: str, lo: int, hi: int) -> Dict[str, Any]:
    st = os.stat(shard)
    return {"shard": os.path.abspath(shard), "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns), "lo": int(lo), "hi": int(hi)}


def stage_tower_to_ram(shard: str, tensors: Sequence[CkptTensor], directory: str) -> TowerSource:
    """Copy the byte extent ``[min offset, max end)`` of ``tensors`` from the
    shard into ``directory`` (a tmpfs), once. Reuses a matching image (same
    shard path/size/mtime and extent); writes atomically (tmp + rename), so a
    killed writer never leaves a half image under the real name."""
    if not tensors:
        raise VisionRankStageRefused(f"{shard}: no tower tensors to stage")
    lo = min(t.file_offset for t in tensors)
    hi = max(t.file_offset + t.nbytes for t in tensors)
    meta = _image_meta(shard, lo, hi)
    path = ram_image_path(shard, directory)
    meta_path = path + ".json"
    try:
        with open(meta_path) as fh:
            old = json.load(fh)
        if old == meta and os.path.getsize(path) == hi - lo:
            return TowerSource(SOURCE_RAM, path, lo, False, hi - lo, reused=True)
    except (OSError, ValueError):
        pass
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        src = os.open(shard, os.O_RDONLY)
        try:
            with open(tmp, "wb") as out:
                pos = lo
                while pos < hi:
                    buf = os.pread(src, min(RAM_STAGE_CHUNK, hi - pos), pos)
                    if not buf:
                        raise VisionRankStageRefused(f"{shard}: short read at {pos} while staging")
                    out.write(buf)
                    pos += len(buf)
        finally:
            os.close(src)
        os.replace(tmp, path)
        with open(meta_path + ".tmp", "w") as fh:
            json.dump(meta, fh)
        os.replace(meta_path + ".tmp", meta_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return TowerSource(SOURCE_RAM, path, lo, False, hi - lo)


def remove_ram_image(src: TowerSource) -> None:
    """Drop a RAM image and its key file (atexit of the staging rank)."""
    if src.kind != SOURCE_RAM:
        return
    for p in (src.path, src.path + ".json"):
        with contextlib.suppress(OSError):
            os.unlink(p)


def shift_plan(plan: Sequence[Tuple[CkptTensor, torch.Tensor]], shift: int
               ) -> List[Tuple[CkptTensor, torch.Tensor]]:
    """The same plan against a file that starts ``shift`` bytes into the
    shard (a RAM image of the extent)."""
    if not shift:
        return list(plan)
    out = []
    for ck, dst in plan:
        if ck.file_offset < shift:
            raise VisionRankStageRefused(
                f"{ck.name} starts at {ck.file_offset}, before the image's first byte {shift}")
        out.append((CkptTensor(ck.name, ck.dtype, ck.shape, ck.file_offset - shift, ck.nbytes), dst))
    return out


# ---------------------------------------------------------------------------
# 6. H125 (NF): the PLACE on the card -- KV tail or free VRAM
# ---------------------------------------------------------------------------
#
# The order's two places, "freier VRAM oder ein kleiner temporaerer
# Offload": the KV tail IS the temporary offload that costs nothing (free KV
# pages at the phase start, handed back before the admission); free VRAM is
# the card's own air (cudaMemGetInfo + this process's idle allocator cache)
# when the tail is not wholly free -- e.g. a P group that still holds pages
# at its wake. ``auto`` (default) tries the tail first, then free VRAM, and
# refuses by name when neither fits. ``kvtail`` is the 27B stage exactly.

PLACE_ENV = "SGLANG_WEG2_VISION_PLACE"
PLACE_AUTO = "auto"
PLACE_KVTAIL = "kvtail"
PLACE_FREE = "free"
PLACES = (PLACE_AUTO, PLACE_KVTAIL, PLACE_FREE)
#: air kept on the card beyond the tower for the encoder's activations (one
#: image at a time: a 1024x1024 image is 4096 patches x 1152 wide, tens of
#: MiB per activation) and the allocator's rounding.
FREE_HEADROOM_MIN = 512 * MIB


def vision_place(env: Optional[Dict[str, str]] = None) -> str:
    e = os.environ if env is None else env
    raw = (e.get(PLACE_ENV, "") or "").strip().lower()
    if not raw:
        return PLACE_AUTO
    if raw not in PLACES:
        raise VisionRankStageRefused(f"{PLACE_ENV}={raw!r} is not one of {list(PLACES)}")
    return raw


def slab_bytes(named_nbytes: Iterable[int], align: int = SLAB_ALIGN) -> int:
    """One contiguous slab holding these tensors in order, each aligned."""
    off = 0
    for n in named_nbytes:
        off = (off + align - 1) // align * align + int(n)
    return off


def free_headroom(tower_bytes: int) -> int:
    return max(FREE_HEADROOM_MIN, int(tower_bytes) // 4)


def free_vram_verdict(need_bytes: int, card_free_bytes: int, cache_idle_bytes: int
                      ) -> Tuple[bool, str]:
    """May the tower go into free VRAM? ``card_free`` is cudaMemGetInfo's
    free, ``cache_idle`` this process's reserved-but-unallocated cache (the
    allocator serves the slab from it first)."""
    air = int(card_free_bytes) + max(0, int(cache_idle_bytes))
    head = free_headroom(need_bytes)
    want = int(need_bytes) + head
    why = (f"need {need_bytes / MIB:.0f} MiB + headroom {head / MIB:.0f} MiB "
           f"vs air {air / MIB:.0f} MiB (card free {card_free_bytes / MIB:.0f} + idle cache "
           f"{max(0, int(cache_idle_bytes)) / MIB:.0f})")
    return air >= want, why
