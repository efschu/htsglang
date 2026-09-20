# SPDX-License-Identifier: Apache-2.0
"""R3 -- executing the seam-A plan: the VMM remap that CHANGES OWNER.

``flip_kv_remap.plan_kv_remap()`` decides WHAT happens to each of the 12
full-attention layers: seven are device-local on the 5090 (``remap``) and five
travel a link (``leg``). This module executes the remap half.

WHY A REMAP AND NOT A COPY, in the one sentence that matters: PP0's 7-layer
pool and the Form-A 12-layer pool are two DIFFERENT allocations of the same
card, and a copy needs both resident at once -- a transient 1.99 GB on a 5090
that has ``card free 0.24 GiB`` under extend (fnFA19:2195). Memory
``S6-REMAP-STATT-ALLOKATION``: the VMM remap hands the PHYSICAL HANDLES from
one virtual reservation to another. The pages never move, nothing is
duplicated, and the peak is the handle table, not the pool.

THE ORDERING THAT IS THE WHOLE CORRECTNESS ARGUMENT. A physical handle may be
mapped at exactly one virtual address at a time for our purposes: the source
pool still owns it until it is unmapped, and the destination cannot map it
before then. So per layer it is strictly

    unmap(src_va)  ->  map(dst_va, handle)  ->  set_access(dst)

and NEVER map-then-unmap. Map-first is the version that looks safer (the data
is reachable throughout) and is the one that actually doubles the residency
for the duration -- i.e. exactly the OOM the remap exists to avoid. The
executor enforces the order rather than documenting it, and
:class:`FakeVmmOps` fails the test if it is inverted.

AND THE WINDOW IN WHICH NOTHING OWNS THE PAGES IS REAL. Between the unmap and
the map the handle is owned by neither pool. A fault there leaves the KV pool
holed -- and an unmapped KV page is not a loud error, it is a page that reads
as zeros on the first decode (the same silent shape seam C's W115 exists for).
So the executor tracks every handle it has taken out and, on any failure,
reports exactly which layers are in that window instead of unwinding blindly:
a rollback that re-maps a handle the destination already took is a second
bug on top of the first.

Hermetic: the device layer is injected (:class:`VmmOps`), the real one
(:class:`CudaVmmOps`) is a thin wrapper that REFUSES BY NAME when the driver
bindings are absent rather than pretending to work, and every test runs
against :class:`FakeVmmOps`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.flip_kv_remap import (
    DISPOSITION_LEG,
    DISPOSITION_REMAP,
    KvRemapPlan,
    LayerMove,
)
from sglang.srt.flip_nextflash_plan import Weg2FlipKvRelayInfeasible

__all__ = [
    "VmmOps",
    "FakeVmmOps",
    "CudaVmmOps",
    "LayerHandle",
    "RemapResult",
    "execute_kv_remap",
]


@dataclass(frozen=True)
class LayerHandle:
    """One layer's physical backing, as the driver names it.

    ``handle`` is opaque -- a ``CUmemGenericAllocationHandle`` in the real
    path. ``src_va`` / ``dst_va`` are the two virtual reservations; the whole
    operation is moving ``handle`` from the first to the second.
    """

    layer_index: int
    handle: object
    src_va: int
    dst_va: int
    nbytes: int


class VmmOps:
    """Injectable device layer -- the ONLY place that may touch the driver.

    Deliberately four verbs and no more, in the shape
    ``offload_movement.DeviceOps`` established: every method is something the
    remap needs, nothing is a convenience.
    """

    def unmap(self, va: int, nbytes: int) -> None:
        """``cuMemUnmap(va, nbytes)`` -- the source releases the handle."""
        raise NotImplementedError

    def map(self, va: int, nbytes: int, handle: object) -> None:
        """``cuMemMap(va, nbytes, 0, handle, 0)`` -- the destination takes it."""
        raise NotImplementedError

    def set_access(self, va: int, nbytes: int, device: int) -> None:
        """``cuMemSetAccess`` -- a freshly mapped range is NOT accessible until
        this runs. Skipping it is a fault at first touch, not a slow path."""
        raise NotImplementedError

    def available(self) -> bool:
        """Can this layer actually reach the driver?"""
        raise NotImplementedError


class FakeVmmOps(VmmOps):
    """Hermetic stub. Models the one invariant the real driver enforces:
    a handle mapped at two addresses at once is a programming error."""

    def __init__(self, fail_on_map: Sequence[int] = ()):
        self.calls: List[Tuple[str, int]] = []
        self.mapped: Dict[int, object] = {}   # va -> handle
        self.accessible: set = set()
        self._owner: Dict[int, int] = {}      # id(handle) -> va
        self._fail_on_map = set(fail_on_map)

    def unmap(self, va: int, nbytes: int) -> None:
        self.calls.append(("unmap", va))
        if va not in self.mapped:
            raise RuntimeError(f"cuMemUnmap on an unmapped va {va}")
        handle = self.mapped.pop(va)
        self._owner.pop(id(handle), None)
        self.accessible.discard(va)

    def map(self, va: int, nbytes: int, handle: object) -> None:
        self.calls.append(("map", va))
        if va in self._fail_on_map:
            raise RuntimeError(f"cuMemMap refused at va {va}")
        owner = self._owner.get(id(handle))
        if owner is not None:
            raise RuntimeError(
                f"cuMemMap: handle is still mapped at va {owner}; a physical "
                f"handle cannot back two virtual ranges at once -- this is the "
                f"map-before-unmap inversion that doubles residency"
            )
        self.mapped[va] = handle
        self._owner[id(handle)] = va

    def set_access(self, va: int, nbytes: int, device: int) -> None:
        self.calls.append(("set_access", va))
        if va not in self.mapped:
            raise RuntimeError(f"cuMemSetAccess on an unmapped va {va}")
        self.accessible.add(va)

    def available(self) -> bool:
        return True


class CudaVmmOps(VmmOps):
    """The real path. Refuses BY NAME when the bindings are absent.

    Deliberately NOT a silent fallback to a copy: a remap that quietly became
    a copy is the 1.99 GB transient this whole seam exists to avoid, and it
    would surface as an OOM on the tightest card with no line saying why.
    """

    def __init__(self, device: int = 0):
        self.device = int(device)
        self._driver = None
        try:  # pragma: no cover -- exercised only on metal
            from cuda.bindings import driver as _driver  # type: ignore

            self._driver = _driver
        except Exception:  # noqa: BLE001
            try:
                from cuda import cuda as _driver  # type: ignore

                self._driver = _driver
            except Exception:  # noqa: BLE001
                self._driver = None

    def available(self) -> bool:
        return self._driver is not None

    def _check(self, rc) -> None:  # pragma: no cover -- metal only
        code = rc[0] if isinstance(rc, tuple) else rc
        if int(getattr(code, "value", code)) != 0:
            raise Weg2FlipKvRelayInfeasible(
                f"W113 Weg2FlipKvRelayInfeasible -- a VMM call failed with "
                f"driver status {code}. The seam-A remap cannot fall back to a "
                f"copy: PP0's 7-layer pool and the Form-A 12-layer pool are "
                f"different allocations, and copying needs a transient 1.99 GB "
                f"on a card that has 0.24 GiB free under extend."
            )

    def _require(self) -> None:
        if self._driver is None:
            raise Weg2FlipKvRelayInfeasible(
                "W113 Weg2FlipKvRelayInfeasible -- no CUDA driver bindings "
                "(cuda.bindings.driver / cuda.cuda) in this process, so the "
                "seam-A VMM remap cannot run. This is refused rather than "
                "degraded to a copy on purpose (Memory "
                "S6-REMAP-STATT-ALLOKATION): the copy needs both pools "
                "resident and the 5090 has 0.24 GiB free under extend."
            )

    def unmap(self, va: int, nbytes: int) -> None:  # pragma: no cover
        self._require()
        self._check(self._driver.cuMemUnmap(va, nbytes))

    def map(self, va: int, nbytes: int, handle: object) -> None:  # pragma: no cover
        self._require()
        self._check(self._driver.cuMemMap(va, nbytes, 0, handle, 0))

    def set_access(self, va: int, nbytes: int, device: int) -> None:  # pragma: no cover
        self._require()
        d = self._driver
        desc = d.CUmemAccessDesc()
        desc.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = int(device)
        desc.flags = d.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._check(d.cuMemSetAccess(va, nbytes, [desc], 1))


@dataclass
class RemapResult:
    remapped_layers: Tuple[int, ...] = ()
    remapped_bytes: int = 0
    legged_layers: Tuple[int, ...] = ()
    #: Layers whose handle is unmapped from the source and not yet mapped at
    #: the destination. Non-empty ONLY on a failure, and then it is the
    #: repair list -- never silently discarded.
    in_flight_layers: Tuple[int, ...] = ()

    def line(self) -> str:
        return (
            f"KV-REMAP remapped={len(self.remapped_layers)} "
            f"bytes={self.remapped_bytes >> 20}MiB "
            f"legged={len(self.legged_layers)} "
            f"in_flight={list(self.in_flight_layers)}"
        )


def execute_kv_remap(
    plan: KvRemapPlan,
    handles: Sequence[LayerHandle],
    ops: VmmOps,
    device: int = 0,
    logger=None,
) -> RemapResult:
    """Hand every device-local layer's physical pages to the destination pool.

    Only the ``remap`` moves are executed here; the ``leg`` moves are the
    transport half of seam A and belong to the link path. They are counted
    and reported so a caller cannot mistake a partial execution for a whole
    one -- "seven layers remapped" is only the truth if the other five were
    actually sent.

    Order per layer is enforced, not documented: unmap source, map
    destination, set access. On a failure the handles already taken out are
    named in ``in_flight_layers`` and the refusal says so, because a blind
    rollback that re-maps a handle the destination already holds is a second
    bug on top of the first.
    """
    if not ops.available():
        raise Weg2FlipKvRelayInfeasible(
            "W113 Weg2FlipKvRelayInfeasible -- the VMM device layer reports "
            "itself unavailable; the seam-A remap cannot run and must not "
            "degrade to a copy (Memory S6-REMAP-STATT-ALLOKATION)."
        )

    by_index: Dict[int, LayerHandle] = {h.layer_index: h for h in handles}
    remap_moves = [m for m in plan.moves if m.disposition == DISPOSITION_REMAP]
    leg_moves = [m for m in plan.moves if m.disposition == DISPOSITION_LEG]

    missing = sorted(m.layer_index for m in remap_moves if m.layer_index not in by_index)
    if missing:
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- the plan remaps layer(s) "
            f"{missing} but no physical handle was supplied for them. A remap "
            f"without a handle is not a smaller remap; it is a hole in the "
            f"destination pool that reads as zeros on the first decode after "
            f"the flip."
        )
    extra = sorted(set(by_index) - {m.layer_index for m in plan.moves})
    if extra:
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- handle(s) supplied for layer(s) "
            f"{extra} that the plan does not move. Either the plan is stale or "
            f"the pool inventory is; both make the remap a claim about pages "
            f"nobody scheduled."
        )

    done: List[int] = []
    in_flight: List[int] = []
    moved = 0
    try:
        for move in sorted(remap_moves, key=lambda m: m.layer_index):
            h = by_index[move.layer_index]
            if h.src_va == h.dst_va:
                raise Weg2FlipKvRelayInfeasible(
                    f"W113 Weg2FlipKvRelayInfeasible -- layer "
                    f"{move.layer_index} has src_va == dst_va ({h.src_va}). "
                    f"That is not a remap, it is a no-op wearing a remap's "
                    f"name, and the destination pool would end up with a "
                    f"reservation nothing backs."
                )
            # ORDER: source lets go FIRST. Map-first would hold the pages at
            # two addresses and double the residency -- the exact OOM this
            # seam exists to avoid.
            ops.unmap(h.src_va, h.nbytes)
            in_flight.append(move.layer_index)
            ops.map(h.dst_va, h.nbytes, h.handle)
            ops.set_access(h.dst_va, h.nbytes, device)
            in_flight.pop()
            done.append(move.layer_index)
            moved += h.nbytes
    except Weg2FlipKvRelayInfeasible:
        raise
    except Exception as exc:  # noqa: BLE001 -- re-raised by name below
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- the seam-A remap failed after "
            f"{len(done)} of {len(remap_moves)} layer(s): {exc}. Layer(s) "
            f"{in_flight} are IN THE WINDOW -- their handles are unmapped from "
            f"the source pool and not yet mapped at the destination, so those "
            f"pages are owned by neither and read as zeros rather than "
            f"faulting. Repair them explicitly; do NOT roll back blindly, "
            f"because re-mapping a handle the destination already took is a "
            f"second bug on top of this one. Layer(s) {done} are complete."
        ) from exc

    result = RemapResult(
        remapped_layers=tuple(done),
        remapped_bytes=moved,
        legged_layers=tuple(m.layer_index for m in leg_moves),
        in_flight_layers=(),
    )
    if logger is not None:
        try:
            logger.info("%s", result.line())
        except Exception:  # noqa: BLE001
            pass
    return result
