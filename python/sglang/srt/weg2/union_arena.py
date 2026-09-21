"""The UNION weights arena across the two weg2 phase groups (2026-09-21).

WHAT THIS IS FOR, in the numbers that caused it. Boot fnFL2 v26/v28 runs the
flip with ``--flip-weights resident``, so the front skips the legs and BOTH
process groups keep their weights on the card for the whole boot:

===========  ==========  ====================  ==================
card         P holds     D holds               sum of a 31.3 GiB card
===========  ==========  ====================  ==================
5090         10.17 GB    12.89 + 3.25 GB MTP   26.31 GB
3080 (0)      6.29 GB     9.36 GB              15.65 of 19.6 GiB
3080 (2)      5.13 GB     9.36 GB              14.49 of 19.6 GiB
===========  ==========  ====================  ==================

v28 died of a CUDA OOM on the 5090 with 54.81 MiB free, the two processes
holding 16.08 and 14.99 GiB. The user's rule (2026-09-21) is the fix and it
is older than this module: *everything byte-identical between the phases is
CARRIED OVER at the phase change; only what differs parks in host RAM.* Today
neither happens -- the identical bytes are simply resident twice.

They really are identical. Under Form A the decode host replicates attention
(DP-attention) and shards experts by whole index (EP-MoE), so both groups hold
WHOLE tensors, not width shards: for a layer both phases carry, the parameter
is the same shape, the same dtype, the same strides and the same bytes.
``weights_arena_union`` already plans one arena over such a union and rebinds
without copying; its own docstring excludes the phase flip on the grounds that
the two layouts are "PP weights vs TP width-shards ... no useful union", and
THAT PREMISE DIED WITH FORM A. This module is the phase-flip case it excluded.

THE BLOCKER IN THE PRIOR ART DOES NOT APPLY HERE.
``REUSE_DESIGN/understand_prior-art.md`` §5 records the load-bearing open
question: TMS-owned pages cannot be exported cross-process, because
``tms_csrc/utils.h`` ``cu_mem_create`` never sets ``requestedHandleTypes``.
Under ``--flip-weights resident`` the weights are NOT TMS-owned (commit
06ebb5b32e: ``weights_region`` takes no ``adapter.region`` when the resident
arm is armed), so this arena creates its OWN physical handles and exports
them. No boot-wide change to the saver is needed for this path.

SCOPE OF THIS SLICE (slice 1 of 3): the PLAN and the CONTRACT between the two
processes -- which tensors may be shared, at which offsets, and the identity
check that makes sharing safe. No CUDA, no mapping, no rebinding: those are
slice 2 (``union_arena_vmm``: exportable handles + the /dev/shm rendezvous)
and slice 3 (the launcher flag and the bind at boot).

THE IDENTITY CHECK IS THE POINT, NOT A FORMALITY. Name plus shape plus dtype
is NOT enough to prove two tensors are the same bytes. Expert-index sharding
gives rank r experts ``[r*k, (r+1)*k)``: same name, same shape, same dtype,
DIFFERENT CONTENT. Sharing those would be silent corruption of exactly the
kind an arena exists to prevent, so the manifest carries a checksum per slot
and the peer verifies its own bytes BEFORE it binds anything. A mismatch is a
loud refusal, never a fallback.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from sglang.srt.model_executor.weights_arena import (
    ArenaLayout,
    ArenaSlot,
    WeightsArenaError,
    plan_arena_layout,
    uint8_checksum,
)
from sglang.srt.model_executor.weights_arena_union import (
    UnionArenaError,
    UnionArenaPlan,
    plan_union_arena,
)

MANIFEST_VERSION = 1

#: the two weg2 phase groups; the arena is planned per CARD over this pair.
PHASE_P = "P"
PHASE_D = "D"


class UnionShareError(UnionArenaError):
    """A refusal of the cross-process union: never a silent fallback."""


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".", 1)[-1]


def _dtype_of(name: str) -> torch.dtype:
    dt = getattr(torch, name, None)
    if not isinstance(dt, torch.dtype):
        raise UnionShareError(f"manifest names an unknown dtype {name!r}")
    return dt


def byte_view(t: torch.Tensor) -> torch.Tensor:
    """The tensor's whole storage as uint8 -- what a checksum must cover.

    ``view(torch.uint8)`` is not enough: it is defined on the VIEW, and the
    arena's V1 scope already guarantees view == whole storage, but a caller
    handing a partial view would silently checksum less than it shares.
    """
    if t.storage_offset() != 0:
        raise UnionShareError(
            f"a tensor with storage_offset {t.storage_offset()} cannot be "
            f"checksummed as a whole storage (V1 scope)"
        )
    storage = t.untyped_storage()
    want = t.numel() * t.element_size()
    if storage.nbytes() != want:
        raise UnionShareError(
            f"tensor views {want} of {storage.nbytes()} storage bytes; the "
            f"union shares whole storages only (V1 scope)"
        )
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(
        storage, 0, (want,), (1,)
    )


def checksums_for(named: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    """Exact uint8 sum per tensor, the content identity the peer verifies."""
    return {name: uint8_checksum(byte_view(t)) for name, t in named.items()}


@dataclasses.dataclass(frozen=True)
class UnionManifest:
    """The owner's published contract: one arena, one offset per name.

    The peer does NOT re-plan the layout. It cannot: it knows only its own
    half of the union, and a layout planned over half the names has different
    offsets for everything after the first missing one. The manifest is the
    single authority and the peer's only job is to prove it fits it.
    """

    version: int
    tag: str
    card: str
    owner_phase: str
    total_bytes: int
    slots: Tuple[ArenaSlot, ...]
    #: slot name -> uint8 checksum of its bytes, as published by the owner
    checksums: Tuple[Tuple[str, int], ...]
    #: phase -> the slot names that phase actually uses
    active: Tuple[Tuple[str, Tuple[str, ...]], ...]
    aliases: Tuple[Tuple[str, str], ...] = ()

    def slot_of(self, name: str) -> ArenaSlot:
        for alias, canon in self.aliases:
            if alias == name:
                name = canon
                break
        for slot in self.slots:
            if slot.name == name:
                return slot
        raise UnionShareError(f"the manifest has no slot for {name!r}")

    def checksum_of(self, name: str) -> Optional[int]:
        for slot_name, value in self.checksums:
            if slot_name == name:
                return value
        return None

    def active_for(self, phase: str) -> Tuple[str, ...]:
        for name, names in self.active:
            if name == phase:
                return names
        raise UnionShareError(
            f"the manifest covers phases {[p for p, _ in self.active]}, not {phase!r}"
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "tag": self.tag,
                "card": self.card,
                "owner_phase": self.owner_phase,
                "total_bytes": self.total_bytes,
                "slots": [
                    {
                        "name": s.name,
                        "offset": s.offset,
                        "nbytes": s.nbytes,
                        "shape": list(s.shape),
                        "stride": list(s.stride),
                        "dtype": _dtype_name(s.dtype),
                    }
                    for s in self.slots
                ],
                "checksums": [[n, v] for n, v in self.checksums],
                "active": [[p, list(names)] for p, names in self.active],
                "aliases": [[a, c] for a, c in self.aliases],
            },
            sort_keys=False,
        )

    @staticmethod
    def from_json(text: str) -> "UnionManifest":
        raw = json.loads(text)
        version = int(raw.get("version", -1))
        if version != MANIFEST_VERSION:
            raise UnionShareError(
                f"union manifest version {version} != {MANIFEST_VERSION}; the "
                f"two groups were started from different trees"
            )
        slots = tuple(
            ArenaSlot(
                name=str(s["name"]),
                offset=int(s["offset"]),
                nbytes=int(s["nbytes"]),
                shape=tuple(int(x) for x in s["shape"]),
                stride=tuple(int(x) for x in s["stride"]),
                dtype=_dtype_of(str(s["dtype"])),
            )
            for s in raw["slots"]
        )
        return UnionManifest(
            version=version,
            tag=str(raw["tag"]),
            card=str(raw["card"]),
            owner_phase=str(raw["owner_phase"]),
            total_bytes=int(raw["total_bytes"]),
            slots=slots,
            checksums=tuple((str(n), int(v)) for n, v in raw.get("checksums", [])),
            active=tuple(
                (str(p), tuple(str(n) for n in names))
                for p, names in raw.get("active", [])
            ),
            aliases=tuple((str(a), str(c)) for a, c in raw.get("aliases", [])),
        )


def plan_card_union(
    named_by_phase: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    alias_of_by_phase: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> UnionArenaPlan:
    """One arena layout over both phases' tensors on ONE card.

    Thin over :func:`plan_union_arena`; it exists so the refusal names the
    PHASES rather than "rungs", and so the phase pair is validated here
    instead of in three call sites.
    """
    if set(named_by_phase) != {PHASE_P, PHASE_D}:
        raise UnionShareError(
            f"a card union is planned over exactly the phases "
            f"{{{PHASE_P!r}, {PHASE_D!r}}}, got {sorted(named_by_phase)}"
        )
    if alias_of_by_phase:
        # plan_union_arena merges first and lays out once, so an alias
        # relation would have to be merged too. V1 refuses instead of
        # guessing: an alias that holds in one phase and not the other is
        # exactly the case a silent merge would get wrong.
        raise UnionShareError(
            "alias relations across phases are out of scope for V1; pass "
            "sets whose tensors own their storages"
        )
    return plan_union_arena(named_by_phase)


def union_saving(
    plan: UnionArenaPlan,
    named_by_phase: Mapping[str, Mapping[str, torch.Tensor]],
) -> Dict[str, int]:
    """What the union costs and what it saves ON THIS CARD, in bytes.

    ``separate`` is what the two groups allocate today (each its own layout),
    ``union`` is the one arena, ``saved`` is the difference. ``overhead_<phase>``
    is what that phase alone would have allocated less than the union -- the
    residency it pays so the flip copies nothing.
    """
    out: Dict[str, int] = {}
    separate = 0
    for phase, named in named_by_phase.items():
        own = int(plan_arena_layout(dict(named)).total_bytes)
        out[f"own_{phase}"] = own
        out[f"overhead_{phase}"] = int(plan.total_bytes) - own
        separate += own
    out["separate"] = separate
    out["union"] = int(plan.total_bytes)
    out["saved"] = separate - int(plan.total_bytes)
    return out


def build_manifest(
    plan: UnionArenaPlan,
    *,
    tag: str,
    card: str,
    owner_phase: str,
    checksums: Mapping[str, int],
) -> UnionManifest:
    """The owner's manifest for a planned card union.

    Every slot must carry a checksum. A slot without one could never be
    verified by the peer, and an unverifiable slot is precisely the
    same-shape-different-content trap this design exists to refuse.
    """
    if owner_phase not in plan.rung_names:
        raise UnionShareError(
            f"owner phase {owner_phase!r} is not one of {plan.rung_names}"
        )
    layout: ArenaLayout = plan.layout
    missing = [s.name for s in layout.slots if s.name not in checksums]
    if missing:
        raise UnionShareError(
            f"{len(missing)} slot(s) carry no checksum (e.g. {missing[0]!r}); "
            f"the peer could not verify them, and an unverified slot is the "
            f"same-shape-different-content trap"
        )
    return UnionManifest(
        version=MANIFEST_VERSION,
        tag=str(tag),
        card=str(card),
        owner_phase=str(owner_phase),
        total_bytes=int(layout.total_bytes),
        slots=tuple(layout.slots),
        checksums=tuple((s.name, int(checksums[s.name])) for s in layout.slots),
        active=tuple((p, tuple(names)) for p, names in plan.active.items()),
        aliases=tuple(layout.aliases),
    )


@dataclasses.dataclass(frozen=True)
class PeerBinding:
    """What the peer may bind, and what stays its own.

    ``shared`` are the slots it will rebind onto the owner's physical pages.
    ``private`` are tensors it holds that the manifest does not cover -- they
    keep their own allocation, which is the correct answer for anything the
    two phases do not hold identically (sharded experts, phase-local buffers).
    """

    shared: Tuple[ArenaSlot, ...]
    private: Tuple[str, ...]
    shared_bytes: int
    private_bytes: int


def verify_peer(
    manifest: UnionManifest,
    phase: str,
    named: Mapping[str, torch.Tensor],
    *,
    checksums: Optional[Mapping[str, int]] = None,
    require_checksums: bool = True,
) -> PeerBinding:
    """Prove this process's tensors fit the owner's manifest, or refuse.

    Checked per tensor the manifest covers: shape, stride, dtype, byte count
    and -- unless the caller explicitly waives it for a desk test -- the
    content checksum. Anything the manifest does not name is reported as
    private rather than refused: a phase legitimately holds bytes the other
    does not.
    """
    if phase not in [p for p, _ in manifest.active]:
        raise UnionShareError(
            f"this process is phase {phase!r}; the manifest covers "
            f"{[p for p, _ in manifest.active]}"
        )
    if checksums is None and require_checksums:
        checksums = checksums_for(
            {n: t for n, t in named.items() if _covered(manifest, n)}
        )
    shared: list = []
    private: list = []
    shared_bytes = 0
    private_bytes = 0
    for name in sorted(named):
        t = named[name]
        if not _covered(manifest, name):
            private.append(name)
            private_bytes += t.numel() * t.element_size()
            continue
        slot = manifest.slot_of(name)
        want = (tuple(t.shape), tuple(t.stride()), t.dtype)
        have = (slot.shape, slot.stride, slot.dtype)
        if want != have:
            raise UnionShareError(
                f"{name!r} is {want} here and {have} in the manifest. The two "
                f"phases do not hold the same tensor, so its bytes cannot be "
                f"shared -- this is a plan error, not a runtime condition."
            )
        nbytes = t.numel() * t.element_size()
        if nbytes != slot.nbytes:
            raise UnionShareError(
                f"{name!r} is {nbytes} bytes here and {slot.nbytes} in the "
                f"manifest"
            )
        if require_checksums:
            published = manifest.checksum_of(slot.name)
            mine = int((checksums or {}).get(name, -1))
            if published is None:
                raise UnionShareError(
                    f"the manifest publishes no checksum for {name!r}"
                )
            if mine != published:
                raise UnionShareError(
                    f"{name!r} has the same shape and dtype in both phases but "
                    f"different BYTES (checksum {mine} here, {published} in the "
                    f"manifest). Sharing it would corrupt one of the two phases; "
                    f"expert-index sharding is the known source of this shape."
                )
        shared.append(slot)
        shared_bytes += slot.nbytes
    return PeerBinding(
        shared=tuple(shared),
        private=tuple(private),
        shared_bytes=shared_bytes,
        private_bytes=private_bytes,
    )


def _covered(manifest: UnionManifest, name: str) -> bool:
    try:
        manifest.slot_of(name)
    except UnionShareError:
        return False
    return True
