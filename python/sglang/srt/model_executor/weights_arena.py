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
    named: Dict[str, torch.Tensor], exclude: Optional[Set[str]] = None
) -> ArenaLayout:
    """Deterministic slot layout for a named-tensor set.

    Order is sorted(name) -- replicated on every boot by construction.
    Tensors sharing one storage get one slot under the FIRST (sorted)
    name; later names become aliases."""
    exclude = exclude or set()
    slots = []
    aliases = []
    by_storage: Dict[int, str] = {}
    offset = 0
    for name in sorted(named):
        if name in exclude:
            continue
        t = named[name]
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
    return ArenaLayout(
        slots=tuple(slots), aliases=tuple(aliases), total_bytes=offset
    )


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


def _torch_pinned_zeros(total: int) -> torch.Tensor:
    """The pre-#695 allocation, kept as the fallback. See below."""
    return torch.zeros(total, dtype=torch.uint8, pin_memory=True)


#: Serial number for the host-image posts. The registry is keyed by NAME, so
#: two images in one process must not share one, or the second would silently
#: replace the first and the ledger would under-report by a whole image.
_image_post_seq = 0


def _register_image_post(nbytes: int) -> None:
    """Record a pinned host image in the shared post registry. Never raises."""
    global _image_post_seq
    try:
        from sglang.srt.mem_cache.pinned_host_budget import (
            PinnedHostPost,
            register_pinned_post,
        )

        _image_post_seq += 1
        register_pinned_post(
            PinnedHostPost(
                name=f"phase-flip host weight image #{_image_post_seq}",
                flag="--enable-phase-flip",
                nbytes=int(nbytes),
            )
        )
    except Exception as exc:  # noqa: BLE001 -- bookkeeping never kills a boot
        logger.debug("could not register host image post: %s", exc)


def _alloc_host_image(total: int, pin: bool) -> torch.Tensor:
    """``total`` zeroed host bytes, page-locked when ``pin``, EXACTLY sized.

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
        return torch.zeros(total, dtype=torch.uint8)
    # #695: make the bytes VISIBLE to the one registry that sums host posts.
    # These images never registered, so `pinned_host_budget` -- which exists
    # precisely because "two independently plausible budgets can be jointly
    # impossible" -- was summing hicache and kvso while 72 GiB of pinned flip
    # images sat outside it. Registered, not CHECKED: a new refusal path here
    # could break a boot that works today, and the diagnosis this is for is
    # served by the number being present, not by a veto.
    _register_image_post(total)
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
    host[layout.total_bytes :] = (
        torch.tensor([csum], dtype=torch.int64).view(torch.uint8)
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
    # #695: same exact-size pinned allocation as image_from_tensors. Every
    # byte is overwritten below, so the zero-fill is redundant here -- it is
    # accepted because a second allocator variant that skips it would be a
    # second thing to keep page-locked correctly, for one memset at boot.
    host = _alloc_host_image(layout.total_bytes + _CHECKSUM_BYTES, pin)
    host[: layout.total_bytes].copy_(used)
    total = uint8_checksum(used)
    host[layout.total_bytes :] = (
        torch.tensor([total], dtype=torch.int64).view(torch.uint8)
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
    want = int(
        image[layout.total_bytes :].clone().view(torch.int64).item()
    )
    dst = arena[: layout.total_bytes]
    dst.copy_(payload)
    have = uint8_checksum(dst)
    if want != have:
        restored = ""
        if restore is not None:
            r_layout, r_image = restore
            arena[: r_layout.total_bytes].copy_(
                r_image[: r_layout.total_bytes]
            )
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
