# SPDX-License-Identifier: Apache-2.0
"""Shared weights arena for the #631 phase flip (DESIGN_631 section 3.3).

Both phase layouts' checkpoint parameters live -- one at a time -- inside
ONE boot-allocated byte arena per rank, at fixed offsets. A flip rewrites
the arena's CONTENTS from the other layout's host image; no tensor
address ever changes after boot, which is what keeps the decode CUDA
graphs valid without recapture (graphs bake weight ADDRESSES at capture).

Mechanics:

* ``plan_arena_layout`` walks a named-tensor set in DETERMINISTIC order
  (sorted names, 256-byte aligned slots) and returns the byte layout.
  Tensors sharing one storage (tied embeddings, GDN ``A_log``/``dt_bias``
  exposed under two names) get ONE slot -- aliasing is preserved, the
  hibernate-restore lesson (#89). Non-checkpoint device params (marlin
  workspace family) are excluded by name.
* ``pack_into_arena`` copies each tensor's bytes to its slot and REBINDS
  ``param.data`` to an arena view with the ORIGINAL shape and STRIDES --
  compressed-tensors INT8 finalizes weights as transposed views, so a
  contiguified copy would silently reorder bytes; ``as_strided`` on the
  arena segment reproduces the exact layout. Rebinding ``.data`` keeps
  the Parameter OBJECT identity (GDN modules capture Parameter objects at
  construction -- replacing them breaks those references; rebinding does
  not).
* ``arena_image`` snapshots the packed arena to ONE contiguous host
  tensor (optionally pinned); ``arena_refill`` is the flip: one
  contiguous copy back. A trailing 8-byte checksum keeps a stale or
  corrupted image falsifiable at runtime.

V1 scope, enforced loudly: every packed tensor must own its storage from
offset 0 (``storage_offset == 0``, view covers the whole storage). That
holds for loader-produced parameters; anything else is refused rather
than half-supported.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

import torch

from sglang.srt.layers.dcp.reshard_plan import KvReshardError

logger = logging.getLogger(__name__)

ARENA_ALIGN = 256
_CHECKSUM_BYTES = 8

# Chunk size for uint8_checksum. torch's sum(dtype=int64) over an integral
# tensor CASTS THE INPUT FIRST (measured: +8x the payload as one transient,
# CPU and CUDA alike), so the checksum walks the payload in chunks and the
# transient is bounded by 8x this constant (128 MiB) instead of 8x the
# payload. With the naive idiom the 27B vehicle allocated ~117 GB host RAM
# per rank at flip-boot; the host OOM killer SIGKILLed rank 2, reproduced
# twice with a memory trace (2026-08-08).
_CHECKSUM_CHUNK_BYTES = 16 * 1024 * 1024

# DEFECT N (2026-08-09 22:00:12Z): the bound above is a CONSTANT, and at a
# phase-flip seam the constant is the problem. 16 MiB x 8 = a 128 MiB
# transient, and the flip happens with the card deliberately full: the KV
# pool is sized to consume it. PP1 asked for exactly 128.00 MiB with
# 106.38 MiB free and the instance died inside
# gdn_flip_mover._pack_pp_side -> kv_reshard._checksum -> here.
#
# It was already primed to fail: the exclusive-KV-backing path had just
# been refused 161480704 bytes by the driver and had called
# empty_cache(), so torch's cache -- the reserve this allocation would
# otherwise have come from -- had been handed back moments earlier.
#
# So the chunk is now chosen against what is ACTUALLY FREE on the
# payload's device rather than against a number picked on a different rig
# for a different failure. The floor keeps it from degenerating into
# per-element launches on a card that is genuinely out of memory; at the
# floor the transient is 8 MiB.
#
# This matters beyond the crash: the >600k-token pool this task is
# working toward makes the seam TIGHTER, not looser, so a fixed transient
# that barely fits today does not fit at all after the pool grows.
_CHECKSUM_CHUNK_MIN_BYTES = 1024 * 1024

# sum(dtype=int64) casts the input first, so the transient is 8x the
# chunk. Spend at most this share of free device memory on it.
_CHECKSUM_FREE_SHARE = 0.25
_CHECKSUM_CAST_FACTOR = 8


def _checksum_chunk_bytes(payload: torch.Tensor) -> int:
    """Largest chunk whose 8x transient fits in free memory, bounded.

    Returns the module constant unchanged for CPU payloads and whenever
    the device cannot be queried -- the host case is what the 16 MiB bound
    was measured for, and it is still right there.
    """
    if not payload.is_cuda:
        return _CHECKSUM_CHUNK_BYTES
    try:
        free, _total = torch.cuda.mem_get_info(payload.device)
    except Exception:
        # Never let an accounting refinement be the thing that breaks a
        # checksum: fall back to the measured constant.
        return _CHECKSUM_CHUNK_BYTES
    affordable = int(free * _CHECKSUM_FREE_SHARE) // _CHECKSUM_CAST_FACTOR
    return max(_CHECKSUM_CHUNK_MIN_BYTES, min(_CHECKSUM_CHUNK_BYTES, affordable))


def uint8_checksum(payload: torch.Tensor) -> int:
    """Exact int64 sum of a uint8 payload without materializing a
    converted copy of it (see _CHECKSUM_CHUNK_BYTES). Device-agnostic;
    a single host sync at the end.

    The VALUE is independent of the chunk size -- an exact integer sum is
    associative -- so sizing the chunk to the device's free memory changes
    only the peak transient, never the checksum. That is what makes the
    adaptive bound safe to apply to payloads whose checksums are compared
    ACROSS ranks and across a flip: two ranks with different free memory
    still agree.
    """
    if payload.numel() == 0:
        return 0
    parts = [
        chunk.sum(dtype=torch.int64)
        for chunk in payload.split(_checksum_chunk_bytes(payload))
    ]
    return int(torch.stack(parts).sum().item())


def checksum_is_representable(value: int, nbytes: int) -> bool:
    """Can ``value`` be :func:`uint8_checksum` of SOME payload of ``nbytes``?

    An exact int64 sum of ``nbytes`` unsigned bytes lies in ``[0, 255 *
    nbytes]`` and nowhere else. That makes the range a free, exact
    discriminator between the two failures a checksum comparison can
    report, which otherwise look identical in a log line:

    * BOTH values in range -> the two ends framed the payload the same way
      and computed different sums over it. The DATA differs. This is what
      the guard is for.
    * A value OUT of range -> that field was never a checksum. The bytes
      read as one are something else (an unwritten buffer tail, payload
      bytes at the wrong offset), so the payload was not framed the way
      the reader expected and the data is not the thing that is wrong.

    #656 register C22 is the second case misreported as the first: the
    acceptance run's two events named "sender" checksums of
    4626949667419791296 (which would need an 18-petabyte payload) and
    -4450328002521349435 (negative, which no sum of unsigned bytes can
    be), and the instance was killed for a data corruption that had not
    happened.
    """
    return 0 <= int(value) <= 255 * int(nbytes)


class WeightsArenaError(KvReshardError):
    """Loud failure of the weights-arena family (#631)."""


@dataclass(frozen=True)
class ArenaSlot:
    name: str
    offset: int
    nbytes: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class ArenaLayout:
    """Deterministic byte layout of one phase's checkpoint tensors."""

    slots: Tuple[ArenaSlot, ...]
    #: alias name -> canonical slot name (same storage at load time).
    aliases: Tuple[Tuple[str, str], ...]
    total_bytes: int

    def slot_of(self, name: str) -> ArenaSlot:
        for a, canon in self.aliases:
            if a == name:
                name = canon
                break
        for s in self.slots:
            if s.name == name:
                return s
        raise WeightsArenaError(f"no arena slot for tensor {name!r}")


def _align(n: int) -> int:
    return (n + ARENA_ALIGN - 1) // ARENA_ALIGN * ARENA_ALIGN


def _check_v1_scope(name: str, t: torch.Tensor) -> int:
    if t.storage_offset() != 0:
        raise WeightsArenaError(
            f"{name!r} has storage_offset {t.storage_offset()}; the arena "
            f"packs whole storages only (V1 scope)"
        )
    storage_nbytes = t.untyped_storage().nbytes()
    want = t.numel() * t.element_size()
    if storage_nbytes != want:
        raise WeightsArenaError(
            f"{name!r} views {want} bytes of a {storage_nbytes}-byte "
            f"storage; a partial view would smuggle unowned bytes into "
            f"the arena (V1 scope)"
        )
    return want


def plan_arena_layout(
    named: Dict[str, torch.Tensor],
    exclude: Optional[Set[str]] = None,
    alias_of: Optional[Dict[str, str]] = None,
) -> ArenaLayout:
    """Deterministic slot layout for a named-tensor set.

    Order is sorted(name) -- replicated on every boot by construction.
    Tensors sharing one storage get one slot under the FIRST (sorted)
    name; later names become aliases.

    ``alias_of`` (#785) SUPPLIES THE ALIAS RELATION INSTEAD OF INFERRING IT,
    as ``{alias_name: canonical_name}``. Everything this function needs is
    metadata -- ``shape``, ``stride``, ``dtype``, ``storage_offset`` and the
    storage size -- and all of it is correct on a META tensor. Exactly one
    input is not: ``data_ptr()`` is 0 for every meta tensor, so the storage
    identity used to detect aliasing collapses the entire model into ONE slot
    and returns a total that is wrong by orders of magnitude while looking
    perfectly well-formed.

    That matters because #785 needs ``layout_tp.total_bytes`` at KV-SIZING
    time, and the TP stack does not exist yet then -- it is built later, in
    ``phase_flip_boot``, which is precisely why the flip seam has had to carry
    its arena tail between boots in a cached record (#782). Laying out a
    meta-device TP parameter set answers the same question without allocating
    a byte, provided the aliasing comes from the caller.

    So the inference path REFUSES on meta input rather than guessing: a silent
    wrong total here would be charged straight into the pool solve.
    """
    exclude = exclude or set()
    slots = []
    aliases = []
    by_storage: Dict[int, str] = {}
    offset = 0
    considered = [n for n in sorted(named) if n not in exclude]
    if alias_of is None:
        # #785: data_ptr() cannot separate "aliases" from "has no storage".
        meta = [n for n in considered if named[n].is_meta]
        if meta:
            raise WeightsArenaError(
                f"{len(meta)} tensor(s) are on the meta device "
                f"(e.g. {meta[0]!r}) and alias_of= was not supplied. Meta "
                f"tensors all report data_ptr()==0, so storage-identity "
                f"aliasing would fold them into a single slot and return a "
                f"total that is far too small but structurally valid. Pass "
                f"the alias relation explicitly to lay out a meta set."
            )
        # The same trap for any storage-less tensor with real extent.
        ghosts = [
            n
            for n in considered
            if named[n].numel() > 0 and named[n].untyped_storage().data_ptr() == 0
        ]
        if len(ghosts) > 1:
            raise WeightsArenaError(
                f"{len(ghosts)} non-empty tensor(s) report data_ptr()==0 "
                f"(e.g. {ghosts[0]!r}); they cannot be distinguished from "
                f"one another by storage identity. Pass alias_of= explicitly."
            )
    else:
        unknown = set(alias_of) - set(considered)
        if unknown:
            raise WeightsArenaError(
                f"alias_of names tensor(s) not in the set: {sorted(unknown)}"
            )
        bad_target = {
            a: c for a, c in alias_of.items() if c not in named or c in exclude
        }
        if bad_target:
            raise WeightsArenaError(
                f"alias_of points at absent canonical tensor(s): {bad_target}"
            )
        # A canonical that is itself an alias would leave the chain's real
        # slot unallocated, so the arena would be short by that tensor.
        chained = {a: c for a, c in alias_of.items() if c in alias_of}
        if chained:
            raise WeightsArenaError(
                f"alias_of is chained: {chained}. Point every alias directly "
                f"at the tensor that owns the slot; a chain leaves the owning "
                f"tensor unslotted and undersizes the arena."
            )
    for name in considered:
        t = named[name]
        if alias_of is not None:
            canon = alias_of.get(name)
            if canon is not None:
                canon_t = named[canon]
                if (
                    t.shape != canon_t.shape
                    or t.stride() != canon_t.stride()
                    or t.dtype != canon_t.dtype
                ):
                    raise WeightsArenaError(
                        f"{name!r} is declared an alias of {canon!r} but has a "
                        f"different view (shape/stride/dtype); the arena "
                        f"preserves aliasing only for identical views "
                        f"(V1 scope)"
                    )
                aliases.append((name, canon))
                continue
            nbytes = _check_v1_scope(name, t)
            slots.append(
                ArenaSlot(
                    name=name,
                    offset=offset,
                    nbytes=nbytes,
                    shape=tuple(t.shape),
                    stride=tuple(t.stride()),
                    dtype=t.dtype,
                )
            )
            offset += _align(nbytes)
            continue
        key = t.untyped_storage().data_ptr()
        if key in by_storage:
            canon = by_storage[key]
            canon_t = named[canon]
            if (
                t.shape != canon_t.shape
                or t.stride() != canon_t.stride()
                or t.dtype != canon_t.dtype
            ):
                raise WeightsArenaError(
                    f"{name!r} aliases {canon!r} storage with a different "
                    f"view (shape/stride/dtype); the arena preserves "
                    f"aliasing only for identical views (V1 scope)"
                )
            aliases.append((name, canon))
            continue
        nbytes = _check_v1_scope(name, t)
        by_storage[key] = name
        slots.append(
            ArenaSlot(
                name=name,
                offset=offset,
                nbytes=nbytes,
                shape=tuple(t.shape),
                stride=tuple(t.stride()),
                dtype=t.dtype,
            )
        )
        offset += _align(nbytes)
    return ArenaLayout(slots=tuple(slots), aliases=tuple(aliases), total_bytes=offset)


def allocate_arena(total_bytes: int, device) -> torch.Tensor:
    """One flat uint8 arena, sized max over all layouts by the caller."""
    return torch.empty(int(total_bytes), dtype=torch.uint8, device=device)


def _slot_view(arena: torch.Tensor, slot: ArenaSlot) -> torch.Tensor:
    seg = arena[slot.offset : slot.offset + slot.nbytes]
    typed = seg.view(slot.dtype)
    return torch.as_strided(typed, slot.shape, slot.stride)


def pack_into_arena(
    named: Dict[str, torch.Tensor],
    layout: ArenaLayout,
    arena: torch.Tensor,
    rebind: Iterable[Tuple[str, torch.nn.Parameter]] = (),
) -> Dict[str, torch.Tensor]:
    """Copy every layout tensor into its slot and return the view map.

    ``rebind`` optionally lists (name, Parameter) pairs whose ``.data``
    is rebound to the arena view -- Parameter object identity preserved.
    Values are copied element-wise through the strided view, so a
    transposed (non-contiguous) weight keeps its exact byte layout."""
    if layout.total_bytes > int(arena.numel()):
        raise WeightsArenaError(
            f"layout needs {layout.total_bytes} bytes but the arena holds "
            f"{int(arena.numel())}; the arena is sized max(all layouts) at "
            f"boot -- this is a sizing bug"
        )
    views: Dict[str, torch.Tensor] = {}
    for slot in layout.slots:
        view = _slot_view(arena, slot)
        view.copy_(named[slot.name])
        views[slot.name] = view
    for alias, canon in layout.aliases:
        views[alias] = views[canon]
    for name, param in rebind:
        param.data = views[name]
    return views


def _alloc_with_host_register(dims, dtype, device, pin_memory, allocator):
    """Indirection so the routing is spyable from a CPU unit test.

    Imported lazily: ``pool_host.common`` pulls in the mem_cache stack, and
    this module is imported by CPU-only tests that must not need it.
    """
    from sglang.srt.mem_cache.pool_host.common import (
        HostTensorAllocator,
        alloc_with_host_register,
    )

    if allocator is None:
        allocator = HostTensorAllocator()
    return alloc_with_host_register(dims, dtype, device, pin_memory, allocator)


def host_image_mode() -> str:
    """The host-image allocation mode boot logs report. Read per call."""
    from sglang.srt.environ import envs

    if envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.get():
        return "file-backed reclaimable"
    return "pinned"


#: O_DIRECT requires the file offset, the length and the buffer address to be
#: aligned to the device's logical block size. 4096 satisfies every device this
#: runs on, and CUDA's pinned host allocations are page-aligned already.
_DIRECT_ALIGN = 4096


@dataclass(frozen=True)
class _FileBackedImage:
    """The read side of a file-backed flip image: fds, and how big it is.

    Held so ``arena_refill`` can pull the bytes with ``preadv`` instead of
    walking the mapping and taking one synchronous fault per 4 KiB page.

    TWO fds, because the fast one cannot serve the whole file. ``fd_direct``
    is O_DIRECT -- it bypasses the ARC and reads at device rate (8 304 MiB/s
    measured, against 2 595 MiB/s for the same reads buffered, because the
    ARC copy is a second pass over every byte and the ARC is capped at 5 GiB
    against a 16 GiB image). But O_DIRECT can only serve block-aligned
    offsets and lengths, and the image is payload + an 8-byte checksum, so
    the last few bytes are not aligned. ``fd`` is the plain buffered fd that
    finishes the tail, and the fallback if O_DIRECT is refused outright.
    """

    fd: int
    nbytes: int
    path: str
    fd_direct: Optional[int] = None


#: Keyed by the image tensor's ``data_ptr()``. These images are allocated once
#: per rank and live for the process, so the key is stable; ``nbytes`` is
#: re-checked at use so a recycled address cannot silently read a wrong file.
_FILE_BACKED_IMAGES: Dict[int, _FileBackedImage] = {}

_staged_pool = None


def _register_file_backed_image(
    image: torch.Tensor,
    fd: int,
    total: int,
    path: str,
    fd_direct: Optional[int] = None,
) -> None:
    _FILE_BACKED_IMAGES[image.data_ptr()] = _FileBackedImage(
        fd=fd, nbytes=total, path=path, fd_direct=fd_direct
    )


def _file_backed_meta(image: torch.Tensor) -> Optional[_FileBackedImage]:
    """The read fd for ``image``, or None if it is not a file-backed image.

    None is the ordinary answer for the DEFAULT pinned-image path, which must
    keep taking the untouched ``dst.copy_`` below.
    """
    meta = _FILE_BACKED_IMAGES.get(image.data_ptr())
    if meta is None:
        return None
    if meta.nbytes != int(image.numel()):
        # Address reuse, not our image. Refusing here is cheap; reading the
        # wrong file would be caught by the checksum but only after the arena
        # had already been overwritten.
        logger.warning(
            "#802 file-backed image registry has %d bytes at this address but "
            "the image holds %d; ignoring the fd and using the fault path.",
            meta.nbytes,
            int(image.numel()),
        )
        return None
    return meta


def _refill_staging_pool(chunk_bytes: int, depth: int):
    """The pinned staging ring, allocated ONCE and reused by every flip.

    Reuses #720's ``ReadBufferPool`` rather than adding a second ring: it is
    a fixed ring that charges its bytes to the pinned-host registry BEFORE
    allocating them, which is what keeps this new host post inside the ledger
    on a swapless box instead of beside it.
    """
    global _staged_pool
    if _staged_pool is not None:
        if _staged_pool.page_bytes == chunk_bytes and _staged_pool.capacity == depth:
            return _staged_pool
        _staged_pool.close()
        _staged_pool = None
    from sglang.srt.mem_cache.read_buffer_pool import ReadBufferPool

    _staged_pool = ReadBufferPool(
        name="phase_flip_refill_staging",
        flag="--phase-flip-image-file-backed",
        capacity=depth,
        page_bytes=chunk_bytes,
        factory=lambda: torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True),
    )
    return _staged_pool


def _staged_file_refill(dst: torch.Tensor, meta: _FileBackedImage, nbytes: int) -> None:
    """Move ``nbytes`` of the image file into ``dst`` as READS, not faults.

    WHY THIS EXISTS (#802, measured on this rig 2026-08-22). The previous path
    was one ``dst.copy_`` over the whole file-backed mapping. That takes one
    synchronous major fault per 4 KiB page -- 4 077 045 of them for the
    16 699 408 904-byte PP0 image, confirmed by ``/proc/self/stat`` -- and
    reaches 1266 MiB/s on a pool that writes at ~3500 MiB/s. No DMA takes
    part on either side, which is why all three ranks converged on the same
    rate despite PCIe links differing by 1.80x.

    THE HINTS DO NOT WORK HERE, and that is measured, not assumed. On this
    OpenZFS pool ``madvise(MADV_WILLNEED)`` over the cold mapping returns 0
    and populates NOTHING: 12 564 ms against a 12 572 ms baseline, with
    4 077 052 faults against 4 077 045, and ``mincore`` residency 0.0 after
    the call. ``MADV_SEQUENTIAL`` per chunk is worse than useless -- 196 200
    ms, a 15.6x REGRESSION. An advisory hint that the filesystem ignores is
    the #738 class of defect (mechanism present, actuator absent), so this
    path does not rely on one.

    What it does instead: bounded ``preadv`` into a pinned staging ring, with
    the next read overlapping the previous chunk's H2D DMA. The reads are
    real reads, so the pool serves them at its own rate; the copies leave
    from PINNED memory, so they are real DMA.

    MEASURED, full cold sweep on the production-sized image, baseline taken
    both first and last so the floor is not assumed::

        mmap_copy (first)       14 377 ms   4 077 015 flt   1 108 MiB/s
        mmap_copy (last)        16 998 ms   4 077 005 flt     937 MiB/s
        willneed_ahead_16x4     20 472 ms   4 077 005 flt     778 MiB/s
        pread_staged_8           6 138 ms          20 flt   2 595 MiB/s
        pread_staged_32          7 103 ms           0 flt   2 242 MiB/s
        pread_staged_32_direct   1 918 ms           0 flt   8 304 MiB/s

    O_DIRECT is 3.2x the buffered read because the buffered path pays a
    second pass over every byte into the ARC -- an ARC capped at 5 GiB here,
    against a 16 GiB image it can never retain. It is also the only arm
    whose ``read_bytes`` (16 056 MiB) matches the file size; ZFS serves
    buffered reads from kernel threads that the caller is never charged for,
    which is why ``read_bytes`` is not a usable instrument on the other arms
    and the fault count is.
    """
    pool = _refill_staging_pool(_refill_chunk_bytes(), _refill_depth())
    chunk = pool.page_bytes
    depth = pool.capacity
    # O_DIRECT serves only block-aligned offsets and lengths. Chunks are a
    # multiple of the alignment and start at multiples of the chunk, so the
    # ONLY unaligned piece is the tail past the last aligned boundary -- the
    # image is payload + an 8-byte checksum, so a tail is the normal case,
    # not an edge case. The buffered fd finishes it.
    direct_limit = (nbytes // _DIRECT_ALIGN) * _DIRECT_ALIGN if meta.fd_direct else 0
    bufs = [pool.acquire() for _ in range(depth)]
    try:
        streams = [torch.cuda.Stream() for _ in range(depth)]
        events = [torch.cuda.Event() for _ in range(depth)]
        inflight = [False] * depth
        views = [b.numpy() for b in bufs]
        off = 0
        i = 0
        while off < nbytes:
            n = min(chunk, nbytes - off)
            if inflight[i]:
                # This buffer's previous H2D must land before it is refilled,
                # or the read would overwrite bytes still in flight.
                events[i].synchronize()
                inflight[i] = False
            got = 0
            while got < n:
                at = off + got
                use_direct = at < direct_limit and at % _DIRECT_ALIGN == 0
                if use_direct:
                    want = min(n - got, direct_limit - at)
                    want -= want % _DIRECT_ALIGN
                if not use_direct or want == 0:
                    fd, want = meta.fd, n - got
                else:
                    fd = meta.fd_direct
                r = os.preadv(fd, [memoryview(views[i])[got : got + want]], at)
                if r == 0:
                    raise WeightsArenaError(
                        f"#802 short read on flip image {meta.path!r}: got "
                        f"{at} of {nbytes} bytes"
                    )
                got += r
            with torch.cuda.stream(streams[i]):
                dst[off : off + n].copy_(bufs[i][:n], non_blocking=True)
                events[i].record(streams[i])
            inflight[i] = True
            off += n
            i = (i + 1) % depth
        for j in range(depth):
            if inflight[j]:
                events[j].synchronize()
        # The checksum below reads ``dst`` on the CURRENT stream, which never
        # saw these copies. Without this the verify could race the transfer.
        cur = torch.cuda.current_stream()
        for j in range(depth):
            cur.wait_event(events[j])
    finally:
        for b in bufs:
            pool.release(b)


def _staged_refill_enabled() -> bool:
    """Read per call, never frozen at import.

    A value captured at import survives a test override and yields a silently
    single-armed measurement -- the same reason #695's opt-out reads per call.
    """
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_PHASE_FLIP_REFILL_STAGED.get())


def _refill_chunk_bytes() -> int:
    from sglang.srt.environ import envs

    return max(1, int(envs.SGLANG_PHASE_FLIP_REFILL_CHUNK_MIB.get())) << 20


def _refill_depth() -> int:
    from sglang.srt.environ import envs

    return max(1, int(envs.SGLANG_PHASE_FLIP_REFILL_DEPTH.get()))


def _file_backed_image(total: int) -> torch.Tensor:
    """Flip host image as a FILE-BACKED shared mapping -- reclaimable, not
    pinned.

    WHY. The pinned images are the flip world's single largest host post
    (68.7 GiB on the 2026-08-18 review composition, non-reclaimable, against
    a 118-GiB swapless container -- SPECIMEN-2026-08-18T0516Z-*). The pin
    buys DMA speed for ``arena_refill``'s one contiguous H2D copy at flip
    time (#690: 9,614.9 MiB per rank). A file-backed mapping trades that
    for reclaimability: after write-back the pages are CLEAN page cache the
    kernel may drop under pressure and refault from disk at the next flip.
    Flip cost, honestly: a pageable H2D copy (roughly half DMA bandwidth)
    when the pages are still cached, plus a disk read bounded by the pool's
    sequential rate when they were reclaimed -- #89's hibernate restore
    (8-14 s for a full weight set on this same pool) is the same-medium
    cold anchor. That price is why this arm is OPT-IN and the default stays
    pinned and byte-identical.

    REFUSES rather than falls back: an enabled arm that quietly pinned
    would claim reclaimability the host ledger then plans on (the #742
    silently-inert-flag class). Volatile filesystems (tmpfs/ramfs) are
    refused by the same #407 verdict the hibernate dir uses -- RAM-backed
    files are exactly the non-reclaimable post this arm exists to remove.

    The file is UNLINKED after mapping: the inode (and with it write-back
    and refault) lives as long as the mapping, and a crash leaves no
    stale multi-GiB files behind. Fresh file pages read zero by the
    filesystem's own rule, which the image checksum contract requires of
    alignment gaps. NOT registered as a pinned host post -- the registry
    sums non-reclaimable bytes.
    """
    import os
    import uuid

    from sglang.srt.environ import envs

    image_dir = envs.SGLANG_PHASE_FLIP_IMAGE_DIR.get()
    if not image_dir:
        raise WeightsArenaError(
            "SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED=1 needs "
            "SGLANG_PHASE_FLIP_IMAGE_DIR pointing at persistent storage; "
            "refusing rather than silently allocating a pinned image the "
            "host ledger would then double-count as reclaimable."
        )
    if not os.path.isdir(image_dir):
        raise WeightsArenaError(
            f"SGLANG_PHASE_FLIP_IMAGE_DIR {image_dir!r} does not exist or "
            "is not a directory."
        )
    from sglang.srt.memtier.hibernate_tier import hibernate_dir_verdict

    verdict = hibernate_dir_verdict(image_dir)
    if verdict.known and not verdict.persistent:
        raise WeightsArenaError(
            f"SGLANG_PHASE_FLIP_IMAGE_DIR {image_dir!r} is on a "
            f"{verdict.fs_type} filesystem: RAM-backed files cannot be "
            "reclaimed to disk, which is the entire point of the "
            "file-backed image arm. Point it at persistent storage."
        )
    path = os.path.join(image_dir, f"flip-image-{os.getpid()}-{uuid.uuid4().hex}.img")
    image = torch.from_file(path, shared=True, size=total, dtype=torch.uint8)
    # #802: keep a read fd BEFORE the unlink, so the refill can READ these
    # bytes instead of faulting them. Without an fd the only way back to the
    # data is the mapping, and touching the mapping is precisely the 4 KiB
    # synchronous fault path that costs ~12.5 s per 16 GiB leg on this pool.
    # The unlink still happens: the inode stays alive for the mapping AND the
    # fd, and neither survives the process, so a crash leaves nothing behind.
    try:
        refill_fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        # Not fatal: the mapping is valid and the fault path still works. The
        # refill just does not get its fast route, and says so once.
        refill_fd = None
        logger.warning(
            "#802 could not open a read fd for flip image %s (%s); the refill "
            "falls back to the page-fault path.",
            path,
            exc,
        )
    direct_fd = None
    if refill_fd is not None:
        try:
            direct_fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        except OSError as exc:
            # Degrades, never kills: the buffered fd still beats the mapping
            # (2 595 MiB/s vs 1 108), it just does not reach device rate.
            logger.warning(
                "#802 O_DIRECT unavailable for flip image %s (%s); the refill "
                "reads buffered instead, which pays the ARC copy.",
                path,
                exc,
            )
    os.unlink(path)
    if refill_fd is not None:
        _register_file_backed_image(image, refill_fd, total, path, direct_fd)
    logger.info(
        "flip host image FILE-BACKED (reclaimable): %d bytes mapped from "
        "%s (unlinked; pages are page cache and may be written back and "
        "reclaimed under pressure; the next flip refaults them). Not "
        "charged to the pinned-host registry.",
        total,
        path,
    )
    return image


def _torch_pinned_zeros(total: int) -> torch.Tensor:
    """The pre-#695 allocation, kept as the fallback. See below."""
    return torch.zeros(total, dtype=torch.uint8, pin_memory=True)


def _torch_pinned_empty(total: int) -> torch.Tensor:
    """The pre-#695 allocation for the ``zero=False`` caller (``arena_image``).

    Separate from ``_torch_pinned_zeros`` because the difference is not
    cosmetic: ``torch.zeros`` faults the whole image in at allocation time,
    which for a 13 GiB image is boot-time cost paid for bytes that are
    overwritten immediately afterwards. The opt-out arm has to reproduce the
    pre-#695 path it is a comparand for, including this.
    """
    return torch.empty(total, dtype=torch.uint8, pin_memory=True)


#: Serial number for the host-image posts. The registry is keyed by NAME, so
#: two images in one process must not share one, or the second would silently
#: replace the first and the ledger would under-report by a whole image.
_image_post_seq = 0


def _register_image_post(nbytes: int) -> Optional[str]:
    """Record a pinned host image in the shared post registry. Never raises.

    Returns the NAME it registered, so the caller can take it back if the
    allocation this post describes then fails. Returning None means nothing
    was registered and there is nothing to undo.
    """
    global _image_post_seq
    try:
        from sglang.srt.mem_cache.pinned_host_budget import (
            PinnedHostPost,
            register_pinned_post,
        )

        _image_post_seq += 1
        name = f"phase-flip host weight image #{_image_post_seq}"
        register_pinned_post(
            PinnedHostPost(
                name=name,
                flag="--enable-phase-flip",
                nbytes=int(nbytes),
            )
        )
        return name
    except Exception as exc:  # noqa: BLE001 -- bookkeeping never kills a boot
        logger.debug("could not register host image post: %s", exc)
    return None


def _unregister_image_post(name: Optional[str]) -> None:
    """Take back a post whose allocation never happened. Never raises.

    THE WINDOW THIS CLOSES. Every producer of a pinned post declares it BEFORE
    allocating, deliberately -- the registry's job is to refuse an
    over-commitment at the declaration rather than discover it at the
    allocation. If the allocation then fails, the post describes bytes that do
    not exist, and #706's credit-back (d7d85b4e37) subtracts already-allocated
    posts from the next admission's demand. A post that never allocated is
    credited back anyway, so the next admission is charged too little: the
    registry itself would wave through the over-commitment it exists to refuse.
    """
    if not name:
        return
    try:
        from sglang.srt.mem_cache.pinned_host_budget import unregister_pinned_post

        unregister_pinned_post(name)
    except Exception as exc:  # noqa: BLE001 -- cleanup never kills a boot
        logger.debug("could not unregister host image post %s: %s", name, exc)


def _alloc_host_image(total: int, pin: bool, zero: bool = True) -> torch.Tensor:
    """``total`` host bytes, page-locked when ``pin``, EXACTLY sized.

    ``zero`` is not cosmetic and the default is not free. An UNPINNED image
    with ``zero=False`` is ``torch.empty``: pages stay untouched until written,
    so an image whose every byte is about to be overwritten costs no resident
    memory up front. Forcing ``torch.zeros`` there faults the whole image in,
    and on a CPU test run over the manager suite that alone was the difference
    between finishing and being SIGKILLed. Callers that overwrite the payload
    immediately (``arena_image``) pass ``zero=False``; callers that rely on
    alignment-gap bytes being zero (``image_from_tensors``) must not.

    #695 -- WHY THIS IS NOT ``torch.zeros(..., pin_memory=True)``.
    Every ``pin_memory=True`` allocation goes through PyTorch's pinned-host
    caching allocator, which rounds the request up to the next POWER OF TWO
    before it ever reaches ``cudaHostAlloc``::

        ATen/core/CachingHostAllocator.h:302  roundSize = PowerOf2Ceil(size);
        ATen/core/CachingHostAllocator.h:334  allocate_host_memory(roundSize, &ptr);

    For a transient buffer that is a bounded overshoot. For these images it is
    permanent: the flip holds two of them per rank for the life of the process
    (``image_pp``, ``image_tp``) plus a draft image, and ``PhaseFlipStacks
    .refill`` re-reads them on every flip, so nothing may free them.

    Measured on the live PP=3 boot of 2026-08-12, ``/proc/<pid>/smaps`` of the
    three scheduler ranks, against the payload figures this repo already
    records at ``phase_flip_spill.py:851-854``::

        PP0  13482.18 -> 16384 MiB   13163.45 -> 16384 MiB
        PP1   8144.00 ->  8192 MiB    7923.95 ->  8192 MiB
        PP2   9114.95 -> 16384 MiB    7923.95 ->  8192 MiB

    58.35 GiB of payload held in 72 GiB of mappings: 13.65 GiB of pure
    rounding, on a swapless box with a ~120 GiB ceiling that had by then taken
    nine cgroup OOM kills. Host shmem is not a free axis -- see
    ``memtier/profile.py:honest_host_memory_bytes``, which until #695 counted
    every one of those bytes as reclaimable page cache.

    ``alloc_with_host_register`` is the in-tree path the host KV pool already
    uses: an exact-size ``MAP_SHARED|MAP_ANONYMOUS|MAP_POPULATE`` mapping plus
    ``cudaHostRegister``. The pages are locked exactly as ``cudaHostAlloc``
    locks them, so the restore's H2D copy stays a DMA -- the size is decided in
    PAGES instead of in powers of two, and nothing else about the image
    changes.

    ZEROING is the mapping's own guarantee, not a memset: MAP_ANONYMOUS pages
    arrive zero-filled, which is what the caller's "alignment-gap bytes are
    zeroed, so the checksum is deterministic" contract needs. Memsetting 13 GiB
    per image to restate a kernel guarantee would cost boot time for nothing.

    FALLBACK. ``cudaHostRegister`` can refuse -- no CUDA context yet, or a
    driver unwilling to lock that many pages. A rank that cannot register must
    still boot, so the pre-#695 allocation is kept and used: the failure costs
    the rounding back, never the boot.
    """
    if not pin:
        if zero:
            return torch.zeros(total, dtype=torch.uint8)
        return torch.empty(total, dtype=torch.uint8)
    # File-backed opt-in arm (see _file_backed_image): reclaimable page
    # cache, so it is NOT a pinned post and must not reach the registry
    # below. Read per call, like the #695 opt-out, and checked BEFORE the
    # registration so a refused misconfiguration never leaves a phantom
    # post behind. Takes precedence over SGLANG_PHASE_FLIP_EXACT_PIN: both
    # of that flag's arms allocate pinned RAM, which is what this arm
    # replaces.
    from sglang.srt.environ import envs as _fb_envs

    if _fb_envs.SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED.get():
        return _file_backed_image(total)
    # #695: make the bytes VISIBLE to the one registry that sums host posts.
    # These images never registered, so `pinned_host_budget` -- which exists
    # precisely because "two independently plausible budgets can be jointly
    # impossible" -- was summing hicache and kvso while 72 GiB of pinned flip
    # images sat outside it. Registered, not CHECKED: a new refusal path here
    # could break a boot that works today, and the diagnosis this is for is
    # served by the number being present, not by a veto.
    _image_post = _register_image_post(total)
    # #695 risk 2 -- the opt-out arm. SGLANG_PHASE_FLIP_EXACT_PIN=0 restores the
    # pre-#695 allocation so the two allocators can be compared through ONE
    # harness on ONE binary, the arms differing by this variable and nothing
    # else. Read per call, not at import: a value frozen at import would also
    # survive a test override and yield a silently single-armed measurement.
    # The post above is registered on BOTH arms on purpose -- the ledger
    # visibility and the shmem pricing are separate defects from the same
    # commit and are correct under either allocator.
    from sglang.srt.environ import envs

    # ONLY THE FAILURE PATH IS NEW. The success path below is byte-identical:
    # same allocator, same fallback, same return. What changes is that an
    # allocation which raises out of here takes its post back with it, instead
    # of leaving the registry charging bytes that never existed. The original
    # exception is re-raised untouched -- cleanup must never substitute the
    # diagnosis (#386), so the operator still sees cudaHostRegister's own words.
    try:
        return _alloc_host_image_inner(total, zero, envs)
    except BaseException:
        _unregister_image_post(_image_post)
        raise


def _alloc_host_image_inner(total: int, zero: bool, envs) -> torch.Tensor:
    """The allocation itself, unchanged. Split out so the post-cleanup wrapper
    above has a single expression to guard rather than a duplicated body."""
    if not envs.SGLANG_PHASE_FLIP_EXACT_PIN.get():
        logger.warning(
            "#695 exact-size pinned host images DISABLED by "
            "SGLANG_PHASE_FLIP_EXACT_PIN=0: allocating %d bytes through "
            "torch's pinned caching allocator, which rounds it up to %d bytes "
            "and holds the difference for the life of the process.",
            total,
            1 << (total - 1).bit_length() if total > 1 else total,
        )
        return _torch_pinned_zeros(total) if zero else _torch_pinned_empty(total)
    try:
        return _alloc_with_host_register((total,), torch.uint8, "cpu", True, None)
    except Exception as exc:  # noqa: BLE001 -- any refusal degrades, none kills
        logger.warning(
            "#695 exact-size pinned host image unavailable (%s); falling back "
            "to torch pin_memory for %d bytes, which the caching host "
            "allocator rounds up to %d bytes. The boot proceeds; the "
            "difference is resident host RAM for the life of the process.",
            exc,
            total,
            1 << (total - 1).bit_length() if total > 1 else total,
        )
        return _torch_pinned_zeros(total)


def image_from_tensors(
    named: Dict[str, torch.Tensor], layout: ArenaLayout, pin: bool = False
) -> torch.Tensor:
    """Host image of a layout built DIRECTLY from live tensors.

    Same format as :func:`arena_image` (payload + 8-byte checksum), but
    without requiring the tensors to be packed in a device arena first.
    This is the boot-order enabler on VRAM-tight ranks: the PP layout is
    snapshotted to host and its device originals freed BEFORE the second
    layout loads and the arena is allocated -- packing both layouts
    through the arena would need weights x3 resident at once (PP originals
    + TP originals + arena), which does not fit the 5090
    (14.7 + 12.4 + 14.7 GB > 31.8). Alignment-gap bytes are zeroed, so
    the checksum is deterministic."""
    total = layout.total_bytes + _CHECKSUM_BYTES
    # #695: exact-size, page-locked. See _alloc_host_image for why this must
    # not be torch's pin_memory (power-of-two rounding, held for process life).
    host = _alloc_host_image(total, pin)
    for slot in layout.slots:
        t = named[slot.name]
        seg = host[slot.offset : slot.offset + slot.nbytes]
        view = torch.as_strided(seg.view(slot.dtype), slot.shape, slot.stride)
        view.copy_(t)
    payload = host[: layout.total_bytes]
    csum = uint8_checksum(payload)
    host[layout.total_bytes :] = torch.tensor([csum], dtype=torch.int64).view(
        torch.uint8
    )
    return host


def bind_arena_views(
    layout: ArenaLayout,
    arena: torch.Tensor,
    rebind: Iterable[Tuple[str, torch.nn.Parameter]],
) -> Dict[str, torch.Tensor]:
    """Rebind parameters to arena views WITHOUT copying any bytes.

    The pure-rebind sibling of :func:`pack_into_arena`: contents are
    expected to arrive via :func:`arena_refill` (before or after -- the
    views are address-stable either way). Used for the layout whose bytes
    live in a host image while the OTHER layout occupies the arena."""
    if layout.total_bytes > int(arena.numel()):
        raise WeightsArenaError(
            f"layout needs {layout.total_bytes} bytes but the arena holds "
            f"{int(arena.numel())}"
        )
    views: Dict[str, torch.Tensor] = {}
    for slot in layout.slots:
        views[slot.name] = _slot_view(arena, slot)
    for alias, canon in layout.aliases:
        views[alias] = views[canon]
    for name, param in rebind:
        param.data = views[name]
    return views


def arena_image(
    arena: torch.Tensor, layout: ArenaLayout, pin: bool = False
) -> torch.Tensor:
    """Host snapshot of the packed arena bytes + 8-byte checksum trailer."""
    used = arena[: layout.total_bytes]
    # #695: same exact-size pinned allocation as image_from_tensors, but
    # zero=False -- the payload and the checksum trailer below overwrite every
    # byte, so this keeps the pre-#695 torch.empty behaviour on the unpinned
    # path instead of faulting the whole image in for nothing.
    host = _alloc_host_image(layout.total_bytes + _CHECKSUM_BYTES, pin, zero=False)
    host[: layout.total_bytes].copy_(used)
    total = uint8_checksum(used)
    host[layout.total_bytes :] = torch.tensor([total], dtype=torch.int64).view(
        torch.uint8
    )
    return host


def arena_refill(
    arena: torch.Tensor,
    layout: ArenaLayout,
    image: torch.Tensor,
    restore: Optional[Tuple[ArenaLayout, torch.Tensor]] = None,
) -> None:
    """The flip: ONE contiguous copy of the other layout's image into the
    arena, verified AFTER the copy on the ARENA's device.

    Ordering rationale (flip-time economics, measured 2026-08-08): the
    old verify-BEFORE-copy summed the multi-GB image on the HOST -- a
    single-core uint8 reduction at ~0.8 GiB/s, worse with all ranks
    summing concurrently. It was the dominant flip leg: 22-33 s of the
    measured per-rank flip wall time against a ~2 s design estimate. On a
    CUDA arena the post-copy checksum runs at device bandwidth instead.

    Abort contract: size/shape violations still abort BEFORE any byte
    moves. A checksum mismatch is detected after the copy; with
    ``restore`` (the CURRENT phase's (layout, image), creation-time
    verified) the active layout's bytes are rewritten so every live view
    of the restoring layout is byte-identical again, then the same clean
    WeightsArenaError raises. Without ``restore`` the mismatch error is
    marked FATAL for the arena content."""
    want_numel = layout.total_bytes + _CHECKSUM_BYTES
    if image.numel() != want_numel:
        raise WeightsArenaError(
            f"image holds {image.numel()} bytes but layout expects "
            f"{want_numel} (payload + checksum); refusing to refill"
        )
    if layout.total_bytes > int(arena.numel()):
        raise WeightsArenaError(
            f"layout needs {layout.total_bytes} bytes but the arena holds "
            f"{int(arena.numel())}"
        )
    payload = image[: layout.total_bytes]
    want = int(image[layout.total_bytes :].clone().view(torch.int64).item())
    dst = arena[: layout.total_bytes]
    # #802: a FILE-BACKED image is refilled by reading the file, not by
    # faulting the mapping (see _staged_file_refill for the measurements).
    # The default PINNED path is untouched: _file_backed_meta returns None for
    # it and the original copy_ below runs unchanged, byte for byte.
    meta = _file_backed_meta(image) if _staged_refill_enabled() else None
    if meta is not None and dst.is_cuda:
        _staged_file_refill(dst, meta, layout.total_bytes)
    else:
        dst.copy_(payload)
    have = uint8_checksum(dst)
    if want != have:
        restored = ""
        if restore is not None:
            r_layout, r_image = restore
            arena[: r_layout.total_bytes].copy_(r_image[: r_layout.total_bytes])
            restored = (
                "; the current phase's layout was restored into the arena "
                "(its views are byte-identical again)"
            )
        else:
            restored = "; ARENA CONTENT IS NOW UNDEFINED (no restore image)"
        raise WeightsArenaError(
            f"arena image checksum mismatch (stored {want}, computed "
            f"{have}); refusing to serve from a corrupted image{restored}"
        )
