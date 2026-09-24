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
    parameter is one tensor at a time -- the largest tower tensor, ~10 MiB.
    """
    orig = torch.nn.Module.register_parameter

    def register_parameter(module, name, param):
        if param is not None and param.device.type != "meta":
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
              stream: Optional[Any] = None) -> DirectReadReport:
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
    fd, rep.direct = _open_direct(shard)
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
