# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""NCCL cross-instance KV transport for PD disaggregation (#111).

STATUS. The seam, the compatibility handshake, the route policy and the block
planner are complete and hermetically tested. :class:`~.link.NcclLink` itself
is written to the interface but its ``transfer`` is the GPU slice: there is no
cross-instance NCCL group on a card-less host, so it has not been written
blind. See ``docs/dev/TASK_111_PD_KV_NCCL.md`` for the two-instance ticket.

Layout, and why it is split this way:

    link.py      the ONE operation a future P2P/NVLink fastpath replaces
                 (#110): move a block list to a peer. Plus LoopbackLink, a
                 real member whose peer is local memory -- which is what makes
                 the byte-identity gate a test rather than a mock.
    contract.py  everything decided ABOVE the link, so no link member can
                 redecide it: the identity handshake (model + geometry +
                 KV DTYPE, #241), the message classes (#240), the direct-vs-
                 store route policy (#212), and the block planner.

Deliberately NOT registered in ``TransferBackend`` yet. Adding the enum member
would advertise a working backend, and the wire path has never run; the ticket
lands the member together with the proof.
"""

from sglang.srt.disaggregation.nccl.contract import (
    IncompatiblePeer,
    MessageClass,
    Route,
    RouteUnavailable,
    TransportIdentity,
    identity_from_args,
    net_for_class,
    plan_blocks,
    resolve_route,
)
from sglang.srt.disaggregation.nccl.link import (
    DEFAULT_LINK_TIMEOUT_S,
    KvLink,
    LinkError,
    LinkRegistrationError,
    LinkTimeoutError,
    LoopbackLink,
    MemoryRegion,
    NcclLink,
    TransferBlock,
    available_links,
    get_link,
    register_link,
)

__all__ = [
    "DEFAULT_LINK_TIMEOUT_S",
    "IncompatiblePeer",
    "KvLink",
    "LinkError",
    "LinkRegistrationError",
    "LinkTimeoutError",
    "LoopbackLink",
    "MemoryRegion",
    "MessageClass",
    "NcclLink",
    "Route",
    "RouteUnavailable",
    "TransferBlock",
    "TransportIdentity",
    "available_links",
    "get_link",
    "identity_from_args",
    "net_for_class",
    "plan_blocks",
    "register_link",
    "resolve_route",
]
