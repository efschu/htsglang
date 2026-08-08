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

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

import torch

from sglang.srt.layers.dcp.reshard_plan import KvReshardError

ARENA_ALIGN = 256
_CHECKSUM_BYTES = 8


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


def arena_image(
    arena: torch.Tensor, layout: ArenaLayout, pin: bool = False
) -> torch.Tensor:
    """Host snapshot of the packed arena bytes + 8-byte checksum trailer."""
    used = arena[: layout.total_bytes]
    host = torch.empty(
        layout.total_bytes + _CHECKSUM_BYTES,
        dtype=torch.uint8,
        pin_memory=pin,
    )
    host[: layout.total_bytes].copy_(used)
    total = int(used.to(torch.int64).sum().item()) if used.numel() else 0
    host[layout.total_bytes :] = (
        torch.tensor([total], dtype=torch.int64).view(torch.uint8)
    )
    return host


def arena_refill(
    arena: torch.Tensor, layout: ArenaLayout, image: torch.Tensor
) -> None:
    """The flip: ONE contiguous copy of the other layout's image into the
    arena. Checksum-verified BEFORE the copy touches the arena -- a
    corrupted image aborts with the arena byte-identical."""
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
    have = int(payload.to(torch.int64).sum().item()) if payload.numel() else 0
    if want != have:
        raise WeightsArenaError(
            f"arena image checksum mismatch (stored {want}, computed "
            f"{have}); refusing to refill from a corrupted image"
        )
    arena[: layout.total_bytes].copy_(payload)
