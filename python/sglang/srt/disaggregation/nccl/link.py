# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The cross-instance KV link seam (#111), with NCCL as today's only member.

WHY A SEAM AND NOT JUST AN NCCL PATH (#110). The byte-moving primitive under a
PD KV transfer is the thing most likely to be replaced: a P2P/NVLink fastpath
between co-located instances, a BAR1 direct path, an RDMA verb. Those differ in
exactly one respect -- how a set of (local buffer, offset, length) blocks
reaches a peer -- and in nothing else. Putting that one operation behind an
interface now means a later fastpath is a new :class:`KvLink`, not a second
copy of the sender/receiver/handshake stack.

The seam is deliberately NARROW. A link does four things:

    setup()     establish the peer connection for a (session, role) pair
    register()  make local memory addressable to the link, ALL OR NOTHING
    transfer()  move a block list to the peer and complete, under a bound
    close()     release

Everything else -- who the peer is, whether the payload is legal, which route
carries it, what the pages mean -- is decided above the link, in
:mod:`contract`, so a new link member cannot accidentally redecide policy.

WHAT IS AND IS NOT PROVEN HERE. :class:`LoopbackLink` is fully exercised: it is
what makes the byte-identity gate a real test rather than a mock assertion.
:class:`NcclLink` is written but has NEVER run on a card -- there is no
cross-instance NCCL group on a CPU host to run it against. Its contract is
pinned by the same tests through the shared :class:`KvLink` conformance suite,
which is a statement about its INTERFACE, not about its wire behaviour. See
the GPU ticket in docs/dev/TASK_111_PD_KV_NCCL.md.
"""

from __future__ import annotations

import abc
import dataclasses
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Default bound for any link wait, in seconds. Never unbounded: #259 found a
#: HiCache collective sitting on a 7200 s gloo default, and #312 made a dead
#: peer an error rather than a spin. A link that blocks forever is the same
#: defect wearing a different hat.
DEFAULT_LINK_TIMEOUT_S = 120.0


class LinkError(RuntimeError):
    """Any link-level failure. Loud by construction; never swallowed."""


class LinkRegistrationError(LinkError):
    """A memory region could not be made addressable to the link.

    Separate from the generic error because the #221 lesson is specifically
    about registration: ``batch_register``'s int status was ignored, so an
    unregistered region became a later transfer error or a silently wrong
    payload instead of a boot failure.
    """


class LinkTimeoutError(LinkError):
    """A link wait hit its bound. Names the peer and the elapsed time."""


@dataclasses.dataclass(frozen=True)
class MemoryRegion:
    """One contiguous local region a link must be able to address."""

    ptr: int
    length: int
    #: What this region is, for error messages that a human can act on
    #: ("KV data buffers", "state component 0 (mamba)", ...).
    what: str

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"region {self.what!r}: length must be positive")
        if self.ptr <= 0:
            raise ValueError(f"region {self.what!r}: ptr must be non-null")


@dataclasses.dataclass(frozen=True)
class TransferBlock:
    """One block to move: a source row range and its destination row range.

    Rows rather than bytes because every PD payload here is row-addressed (KV
    pages, mamba slots, aux entries), and because a row count is what the
    owner rule and the page tables speak in.
    """

    region_index: int
    src_offset_bytes: int
    dst_offset_bytes: int
    length_bytes: int

    def __post_init__(self) -> None:
        if self.length_bytes <= 0:
            raise ValueError("transfer block length must be positive")
        if self.src_offset_bytes < 0 or self.dst_offset_bytes < 0:
            raise ValueError("transfer block offsets must be non-negative")


class KvLink(abc.ABC):
    """One way of moving KV blocks to a peer instance.

    Implementations are constructed per (session, role); a link instance is
    never shared between two peers, so ``setup`` has no membership decision to
    make and nothing here needs a collective to agree on who is present.
    """

    #: Stable name, used in ``--disaggregation-kv-link`` and in errors.
    name: str = "abstract"

    @abc.abstractmethod
    def setup(
        self,
        *,
        session_id: str,
        is_sender: bool,
        peer: str,
        expected_world_size: Optional[int] = None,
    ) -> None:
        """Establish the connection. Bounded; raises rather than blocking.

        ``expected_world_size`` is the member count bootstrap agreed on. A
        member that forms a group MUST check what it was handed against it
        (#259's fixed universe): a short world rendezvouses successfully and
        then silently moves a subset of the rows.
        """

    @abc.abstractmethod
    def register(self, regions: Sequence[MemoryRegion]) -> None:
        """Make every region addressable, or raise.

        ALL OR NOTHING (#221). A partial success must raise
        :class:`LinkRegistrationError` naming which region failed; a link that
        registers three of four buffers and returns normally has produced a
        server that boots and corrupts one component's payload later.
        """

    @abc.abstractmethod
    def transfer(
        self,
        blocks: Sequence[TransferBlock],
        *,
        message_class: str,
        timeout_s: Optional[float] = None,
    ) -> int:
        """Move the blocks and complete. Returns bytes moved.

        Bounded by ``timeout_s`` (default :data:`DEFAULT_LINK_TIMEOUT_S`);
        exceeding it raises :class:`LinkTimeoutError` naming the peer.
        """

    def close(self) -> None:
        """Release. Idempotent; never raises on a already-closed link."""


def bounded_formation(
    factory: Callable[[], Any],
    *,
    peer: str,
    timeout_s: float = DEFAULT_LINK_TIMEOUT_S,
) -> Any:
    """Run a group-formation call under a wall-clock bound.

    FORMATION IS A WAIT -- the first one, and the one most likely to hang. A
    rendezvous blocks until every declared peer arrives, and torch's TCPStore
    carries its own default measured in tens of minutes, which silently wins
    over any bound this module declares elsewhere. Calling the factory bare
    made "every wait is bounded" false for exactly the wait that matters:
    #259's defect, one layer above where #259 found it.

    Why a thread and not ``bounded_collective``: the #312 helpers bound a
    collective ON an existing group, and here the group is what does not exist
    yet. There is nothing to poll, so the bound has to be wall-clock around
    the call. The worker thread is left daemon on timeout -- a rendezvous
    stuck in the store cannot be interrupted from outside, so the honest
    behaviour is to stop WAITING on it and say so, not to pretend it was
    cancelled.
    """
    import threading

    box: dict = {}

    def _run() -> None:
        try:
            box["value"] = factory()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
            box["error"] = exc

    t = threading.Thread(target=_run, daemon=True, name=f"kvlink-formation-{peer}")
    started = time.monotonic()
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise LinkTimeoutError(
            f"group formation with peer {peer} did not complete within "
            f"{timeout_s:.0f}s (waited {time.monotonic() - started:.1f}s). The "
            "rendezvous is still blocked in the store; this bound exists "
            "because the store's own default is tens of minutes and would "
            "otherwise decide the timeout (#259)."
        )
    if "error" in box:
        raise LinkError(
            f"group formation with peer {peer} failed: "
            f"{type(box['error']).__name__}: {box['error']}"
        ) from box["error"]
    return box.get("value")


def verify_world(
    group: Any,
    *,
    expected_world_size: Optional[int],
    peer: str,
    world_size_fn: Optional[Callable[[Any], int]] = None,
) -> None:
    """Check the world we were HANDED against the one bootstrap agreed on.

    The fixed-universe rule (#259) is stated in this module and was, until
    now, only stated: a factory that quietly discovers its members, or builds
    a smaller world because one peer was slow to arrive, produced a link that
    looked correct. A short world is not a formation error -- every present
    rank rendezvouses successfully -- so nothing raises, and the transfer
    later moves a subset of the rows with no indication that it did.

    ``expected_world_size=None`` means the caller did not carry a
    bootstrap-agreed count; the check is then skipped and says so, rather than
    inventing an expectation.
    """
    if expected_world_size is None:
        return
    if world_size_fn is None:

        def world_size_fn(g: Any) -> int:
            import torch.distributed as dist

            return int(dist.get_world_size(g))

    try:
        actual = int(world_size_fn(group))
    except Exception as exc:  # noqa: BLE001 - an unreadable world is a failure
        raise LinkError(
            f"cannot read the world size of the group formed with {peer}, so "
            f"the fixed-universe rule cannot be checked: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if actual != int(expected_world_size):
        raise LinkError(
            f"group formed with {peer} has world size {actual}, but bootstrap "
            f"agreed on {int(expected_world_size)}. A short world is not a "
            "formation error -- every present rank rendezvouses fine -- so "
            "this is checked rather than trusted: transferring into it would "
            "move a subset of the rows with nothing to indicate it."
        )


class LoopbackLink(KvLink):
    """An in-process link that copies between two local buffer sets.

    Not a mock and not a test double in the pejorative sense: it is a real
    implementation of the interface whose peer happens to be a set of local
    destination regions. That is what makes the byte-identity gate meaningful
    on a card-less host -- the same block list, the same offsets, the same
    all-or-nothing registration, and a real memcpy at the end.
    """

    name = "loopback"

    def __init__(self, dst_regions: Optional[Sequence[MemoryRegion]] = None) -> None:
        self._src: List[MemoryRegion] = []
        self._dst: List[MemoryRegion] = list(dst_regions or ())
        self._open = False
        self.bytes_moved = 0
        #: Every transfer, for tests that assert on the block plan itself.
        self.transfers: List[Dict[str, Any]] = []

    def setup(
        self,
        *,
        session_id: str,
        is_sender: bool,
        peer: str,
        expected_world_size: Optional[int] = None,
    ) -> None:
        # Loopback's universe is one local pair by construction, so an
        # expectation other than 1 is a caller error rather than a short world.
        if expected_world_size is not None and int(expected_world_size) != 1:
            raise LinkError(
                f"loopback link is a single local pair (world size 1) but the "
                f"caller expects {expected_world_size}; the fixed-universe "
                "check would silently pass on a link that has no peers"
            )
        self._open = True
        self.session_id = session_id
        self.peer = peer

    def register(self, regions: Sequence[MemoryRegion]) -> None:
        _validate_regions(regions, link=self.name)
        self._src = list(regions)

    def set_destination(self, regions: Sequence[MemoryRegion]) -> None:
        _validate_regions(regions, link=self.name)
        self._dst = list(regions)

    def transfer(
        self,
        blocks: Sequence[TransferBlock],
        *,
        message_class: str,
        timeout_s: Optional[float] = None,
    ) -> int:
        if not self._open:
            raise LinkError("loopback link used before setup()")
        import ctypes

        moved = 0
        for b in blocks:
            if b.region_index >= len(self._src) or b.region_index >= len(self._dst):
                raise LinkError(
                    f"block references region {b.region_index}, but only "
                    f"{min(len(self._src), len(self._dst))} region pair(s) exist"
                )
            src = self._src[b.region_index]
            dst = self._dst[b.region_index]
            if b.src_offset_bytes + b.length_bytes > src.length:
                raise LinkError(
                    f"block runs past the source region {src.what!r}: "
                    f"{b.src_offset_bytes}+{b.length_bytes} > {src.length}"
                )
            if b.dst_offset_bytes + b.length_bytes > dst.length:
                raise LinkError(
                    f"block runs past the destination region {dst.what!r}: "
                    f"{b.dst_offset_bytes}+{b.length_bytes} > {dst.length}"
                )
            ctypes.memmove(
                dst.ptr + b.dst_offset_bytes,
                src.ptr + b.src_offset_bytes,
                b.length_bytes,
            )
            moved += b.length_bytes
        self.bytes_moved += moved
        self.transfers.append(
            {"message_class": message_class, "blocks": len(blocks), "bytes": moved}
        )
        return moved

    def close(self) -> None:
        self._open = False


def order_blocks(blocks: Sequence[TransferBlock]) -> List[TransferBlock]:
    """The order both sides walk, derived rather than coordinated.

    NCCL send/recv pair up POSITIONALLY: the n-th send on the sender must meet
    the n-th recv on the receiver. The two instances build their block lists
    independently from their own page tables, so relying on "the order the
    planner happened to emit" makes correctness depend on two code paths
    staying in step -- and a mismatch does not fail, it lands the right bytes
    at the wrong offset (#345's class, one layer down).

    Sorting both sides by a key each can compute alone removes the need to
    agree at all. Region first, then the offsets, so the order is total.
    """
    return sorted(
        blocks,
        key=lambda b: (b.region_index, b.src_offset_bytes, b.dst_offset_bytes),
    )


def _validate_blocks(
    blocks: Sequence[TransferBlock],
    regions: Sequence[MemoryRegion],
    *,
    link: str,
) -> None:
    """Bounds-check a plan against the registered regions.

    The messages deliberately mirror LoopbackLink's, because the KvLink
    contract says a caller must not be able to tell the members apart by which
    errors they raise. Loopback keeps its own inline copy: rewriting it would
    put a refactor of a validated member inside an unvalidated member's cut.
    """
    if not regions:
        raise LinkError(
            f"{link} link has no registered regions; register() must run "
            f"before transfer()"
        )
    for b in blocks:
        if b.region_index >= len(regions):
            raise LinkError(
                f"block references region {b.region_index}, but only "
                f"{len(regions)} region(s) exist"
            )
        region = regions[b.region_index]
        for offset, side in (
            (b.src_offset_bytes, "source"),
            (b.dst_offset_bytes, "destination"),
        ):
            if offset + b.length_bytes > region.length:
                raise LinkError(
                    f"block runs past the {side} region {region.what!r}: "
                    f"{offset}+{b.length_bytes} > {region.length}"
                )


class LinkOps:
    """The two wire calls, isolated so the contract above is testable.

    Production is :class:`TorchDistributedOps`. A test supplies a double and
    then owns every property this module claims -- ordering, bounds, partial
    failure, accounting -- without a peer, which is the only way any of it can
    be checked before the GPU ticket runs.
    """

    def send(self, view: Any, *, group: Any, timeout_s: float) -> None:
        raise NotImplementedError

    def recv(self, view: Any, *, group: Any, timeout_s: float) -> None:
        raise NotImplementedError


class TorchDistributedOps(LinkOps):
    """torch.distributed point-to-point, bounded by the #312 helpers.

    NOT VALIDATED ON HARDWARE -- this is the half the GPU ticket exists to
    prove. It is written against the same bounded-wait helpers the rest of the
    fork uses rather than raw ``dist.send``/``dist.recv``, so a dead peer ends
    the wait with a named error instead of hanging a scheduler thread.
    """

    def _peer_rank(self, group: Any) -> int:
        import torch.distributed as dist

        rank = dist.get_rank(group=group)
        # A cross-instance link is a two-member group by construction (#259's
        # fixed universe), so the peer is simply the other rank.
        return 1 - rank

    def send(self, view: Any, *, group: Any, timeout_s: float) -> None:
        import torch.distributed as dist

        from sglang.srt.distributed.device_communicators.barlink_liveness import (
            bounded_collective,
        )

        bounded_collective(
            lambda: dist.send(view, dst=self._peer_rank(group), group=group),
            timeout_s=timeout_s,
            what="nccl kv send",
        )

    def recv(self, view: Any, *, group: Any, timeout_s: float) -> None:
        import torch.distributed as dist

        from sglang.srt.distributed.device_communicators.barlink_liveness import (
            bounded_collective,
        )

        bounded_collective(
            lambda: dist.recv(view, src=self._peer_rank(group), group=group),
            timeout_s=timeout_s,
            what="nccl kv recv",
        )


class _ArrayInterfaceView:
    """Describes a raw device range to torch, per weak_ref_tensor.py's pattern."""

    __slots__ = ("__cuda_array_interface__",)

    def __init__(self, interface: dict):
        self.__cuda_array_interface__ = interface


def default_region_view(region: MemoryRegion, offset: int, length: int) -> Any:
    """A byte range of a registered region, as a tensor torch can send.

    THE ONE PRIMITIVE THAT COULD NOT BE WRITTEN BLIND, and it is isolated here
    for that reason. Device memory goes through ``__cuda_array_interface__``
    (the same route ``compilation/weak_ref_tensor.py`` already uses to build a
    non-owning tensor over a pointer); host memory goes through ctypes. The
    region carries only a pointer, so which one applies cannot be read off the
    type -- the CUDA route is tried first and the host route is the fallback.

    The GPU ticket owns confirming this against a real KV buffer. Until then
    the wire path is opt-in, and a caller that wants a different materialisation
    passes ``region_view=`` rather than editing this.
    """
    import torch

    base = region.ptr + offset
    try:
        interface = {
            "shape": (length,),
            "typestr": "|u1",
            "data": (base, False),
            "version": 3,
            "strides": None,
        }
        return torch.as_tensor(_ArrayInterfaceView(interface))
    except Exception:  # noqa: BLE001 - host memory, or no CUDA at all
        import ctypes

        buffer = (ctypes.c_char * length).from_address(base)
        return torch.frombuffer(memoryview(buffer), dtype=torch.uint8)


class NcclLink(KvLink):
    """Cross-instance KV movement over an NCCL process group.

    NOT VALIDATED ON HARDWARE. Written to the interface above and reviewed
    against the mooncake path it parallels, but there is no cross-instance
    NCCL group on a CPU host, so nothing below this line has executed against
    a real peer. The GPU ticket is the gate; until it passes this member is
    opt-in and the manager says so at construction.

    DESIGN NOTES that the GPU ticket must confirm rather than assume:

    * **Group formation is a rendezvous, not a collective vote.** Both sides
      learn the peer set from the bootstrap exchange (host, port, ranks) and
      then join a ``TCPStore``-backed group of exactly that fixed universe.
      No rank asks the group who is present -- that would be a membership
      decision taken by a collective, which is the #94 family, and #259's
      "fixed pool universe" for the same reason.
    * **Every wait is bounded** through ``bounded_collective`` /
      ``bounded_barrier`` from ``distributed.device_communicators.
      barlink_liveness`` -- the #312 helpers, reused rather than re-derived,
      so a dead peer ends the wait with a named error.
    * **Registration is a no-op for NCCL** (it addresses tensors, not pinned
      MRs), but the method still validates the region list so a caller cannot
      tell the two links apart by which errors they raise.
    """

    name = "nccl"

    def __init__(
        self,
        *,
        group_factory: Optional[Callable[..., Any]] = None,
        timeout_s: float = DEFAULT_LINK_TIMEOUT_S,
        world_size_fn: Optional[Callable[[Any], int]] = None,
        wire_enabled: bool = False,
        ops: Optional["LinkOps"] = None,
        region_view: Optional[Callable[[MemoryRegion, int, int], Any]] = None,
    ) -> None:
        self._group = None
        self._group_factory = group_factory
        self._timeout_s = timeout_s
        #: OPT-IN. False keeps transfer() raising the pre-wire NotImplementedError
        #: verbatim. The two-instance GPU ticket is what flips this default; a
        #: transport nobody has run must not become reachable by upgrading.
        self._wire_enabled = bool(wire_enabled)
        #: The two seams the wire path is built on, both injectable so the
        #: CONTRACT (ordering, bounds, partial failure, accounting) is
        #: exercisable without a peer -- and so the one primitive that cannot
        #: be written blind, materialising a region as a tensor, is replaceable
        #: by whatever the GPU ticket proves correct.
        self._ops = ops if ops is not None else TorchDistributedOps()
        self._region_view = (
            region_view if region_view is not None else default_region_view
        )
        self.is_sender: bool = False
        #: Observability parity with LoopbackLink, so a harness can read the
        #: same fields off either member.
        self.bytes_moved: int = 0
        self.transfers: List[Dict[str, Any]] = []
        #: Injectable so the fixed-universe check is exercisable without a
        #: real process group; production leaves it None -> torch.distributed.
        self._world_size_fn = world_size_fn
        self._regions: List[MemoryRegion] = []
        self.peer: Optional[str] = None
        self.session_id: Optional[str] = None

    def setup(
        self,
        *,
        session_id: str,
        is_sender: bool,
        peer: str,
        expected_world_size: Optional[int] = None,
    ) -> None:
        self.session_id = session_id
        self.peer = peer
        self.is_sender = is_sender
        if self._group_factory is None:
            raise LinkError(
                "NcclLink needs a group factory: the cross-instance process "
                "group is built from the bootstrap exchange (a FIXED peer "
                "universe, #259), never discovered by asking the group who is "
                "present. Pass group_factory=... from the KV manager."
            )
        # BOUNDED: formation is a wait, and the store's own default would
        # otherwise decide the timeout. See bounded_formation.
        self._group = bounded_formation(
            lambda: self._group_factory(
                session_id=session_id, is_sender=is_sender, peer=peer
            ),
            peer=peer,
            timeout_s=self._timeout_s,
        )
        # CHECKED, not trusted: the world we were handed must be the world
        # bootstrap agreed on. See verify_world.
        verify_world(
            self._group,
            expected_world_size=expected_world_size,
            peer=peer,
            world_size_fn=self._world_size_fn,
        )

    def register(self, regions: Sequence[MemoryRegion]) -> None:
        # NCCL addresses tensors, so there is no MR to pin -- but the region
        # list is still validated here so that a malformed plan fails at the
        # same point on every link, and so the all-or-nothing contract in the
        # base class is not quietly weaker for this member (#221).
        _validate_regions(regions, link=self.name)
        self._regions = list(regions)

    def transfer(
        self,
        blocks: Sequence[TransferBlock],
        *,
        message_class: str,
        timeout_s: Optional[float] = None,
    ) -> int:
        if self._group is None:
            raise LinkError("NcclLink.transfer() before setup()")
        if not blocks:
            return 0
        if not self._wire_enabled:
            # The pre-wire refusal, PRESERVED VERBATIM. The wire path below is
            # opt-in until the two-instance GPU ticket passes, so a deployment
            # that has not asked for it gets exactly the message it got before
            # the path existed -- an unproven transport must not become
            # reachable by upgrading.
            raise NotImplementedError(
                "NcclLink.transfer is the GPU slice of #111: it needs a "
                "cross-instance NCCL group to run against and has therefore not "
                "been written blind. The seam, the contract, the route policy and "
                "the block planner are complete and tested; see "
                "docs/dev/TASK_111_PD_KV_NCCL.md for the ticket that lands this."
            )

        from sglang.srt.disaggregation.nccl.contract import MessageClass

        # Validated, not just carried: an unknown class here means the caller
        # and the route policy disagree, and the route policy is what decided
        # this link should carry the payload at all.
        message = MessageClass(message_class)
        timeout = self._timeout_s if timeout_s is None else float(timeout_s)
        ordered = order_blocks(blocks)
        _validate_blocks(ordered, self._regions, link=self.name)

        moved = 0
        for position, block in enumerate(ordered):
            region = self._regions[block.region_index]
            offset = (
                block.src_offset_bytes if self.is_sender else block.dst_offset_bytes
            )
            try:
                view = self._region_view(region, offset, block.length_bytes)
                if self.is_sender:
                    self._ops.send(view, group=self._group, timeout_s=timeout)
                else:
                    self._ops.recv(view, group=self._group, timeout_s=timeout)
            except LinkTimeoutError:
                raise
            except Exception as exc:  # noqa: BLE001 - re-raised as a link error
                # NEVER a partial success. The count of bytes already on the
                # wire is named because a caller that retries needs to know the
                # stream is dirty -- #221's all-or-nothing, one layer down.
                raise LinkError(
                    f"{self.name} link to peer {self.peer!r}: block "
                    f"{position + 1}/{len(ordered)} (region "
                    f"{block.region_index}, {block.length_bytes} B) failed "
                    f"after {moved} B had already moved: {exc!r}. The transfer "
                    f"is incomplete and the peer's stream is out of step; this "
                    f"session cannot be retried on the same link."
                ) from exc
            moved += block.length_bytes

        self.bytes_moved += moved
        self.transfers.append(
            {
                "message_class": message.value,
                "blocks": len(ordered),
                "bytes": moved,
            }
        )
        return moved

    def close(self) -> None:
        self._group = None


def _validate_regions(regions: Sequence[MemoryRegion], *, link: str) -> None:
    """All-or-nothing region validation, shared by every link (#221).

    Raises on the FIRST bad region and names it. The point is that a caller
    can never observe a link that accepted some regions and rejected others:
    either the whole set is addressable or the call raised.
    """
    if not regions:
        raise LinkRegistrationError(
            f"{link}: empty region list. A transport registered with no "
            "buffers would transfer nothing and report success."
        )
    seen: Dict[int, str] = {}
    for i, r in enumerate(regions):
        if not isinstance(r, MemoryRegion):
            raise LinkRegistrationError(
                f"{link}: region {i} is {type(r).__name__}, not MemoryRegion"
            )
        if r.ptr in seen:
            raise LinkRegistrationError(
                f"{link}: region {i} ({r.what!r}) has the same base pointer as "
                f"{seen[r.ptr]!r}; a duplicate registration means one of the "
                "two components would be written over the other"
            )
        seen[r.ptr] = r.what


#: The registry. A new fastpath (#110) adds one entry here and implements
#: KvLink; nothing else in the PD stack changes.
_LINKS: Dict[str, Callable[..., KvLink]] = {
    "nccl": NcclLink,
    "loopback": LoopbackLink,
}


def available_links() -> List[str]:
    return sorted(_LINKS)


def get_link(name: str, **kwargs: Any) -> KvLink:
    """Construct a link by name. An unknown name is a loud error, never a
    silent fallback to a different transport than the operator asked for."""
    try:
        factory = _LINKS[name]
    except KeyError:
        raise ValueError(
            f"unknown KV link {name!r}; available: {', '.join(available_links())}"
        ) from None
    return factory(**kwargs)


def register_link(name: str, factory: Callable[..., KvLink]) -> None:
    """Add a link member. Used by out-of-tree fastpaths (#110)."""
    if name in _LINKS:
        raise ValueError(f"KV link {name!r} is already registered")
    _LINKS[name] = factory
