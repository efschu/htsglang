# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#753: the mid-loop crossing wire for gapped PP layer sets.

WHAT WAS MISSING. ``qwen3_5.py``'s forward loop exchanges ``pp_proxy_tensors``
exactly ONCE per rank, at the stage boundary. That is correct for contiguous
ownership and silently wrong for a gapped set: a rank owning ``{2, 4}`` runs
layer 2 straight into layer 4, and layer 3 -- computed on a peer -- is never
received. No exception. Fluent, confidently wrong output.

#735 delivered the DESK half of this: ``crossing_schedule`` says which
crossings exist, and ``route_schedule`` binds each to a transport. Neither
moved a tensor. This module is the forward wire that consumes both.

THE SHAPE, and why it is a separate module rather than inline in the model.
The model's loop is boot-only territory; a wire written inside it can only be
tested by booting. Here the same logic is a small object with an injectable
link, so the crossing algebra, the ordering, the counting and the
byte-identity of the contiguous path are all hermetic. The model file gains
two call sites and no logic.

BYTE-IDENTITY IS A NULL OBJECT, NOT A BRANCH. Contiguous ownership yields
:class:`NoCrossingWire`, whose methods return their arguments unchanged. There
is no ``if wire_enabled`` inside the hot loop deciding between two code paths
that could drift apart -- the disabled path executes the same two calls, and
they are identity functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "CrossingWire",
    "NoCrossingWire",
    "PpCrossingWireError",
    "PpGroupLink",
    "build_crossing_wire",
    "build_wire_for_model",
]


class PpCrossingWireError(RuntimeError):
    """A crossing could not be carried. Never silent, never skipped."""


class NoCrossingWire:
    """The contiguous path: both hooks are identity.

    Returned whenever ownership is contiguous, which is every configuration
    that worked before #753. The model calls the same two methods either way,
    so the contiguous path cannot drift from the gapped one by acquiring a
    branch that only one of them takes.
    """

    #: Kept so callers can report uniformly.
    crossings_sent = 0
    crossings_received = 0

    def __bool__(self) -> bool:
        return False

    def before_layer(self, layer_id: int, hidden, residual):
        return hidden, residual

    def after_layer(self, layer_id: int, hidden, residual) -> None:
        return None

    def describe(self) -> str:
        return "crossing wire: inactive (contiguous ownership)"


@dataclass
class CrossingWire:
    """Carries activations across ownership boundaries inside the loop.

    ``recv_before[layer]`` is the peer to receive from before computing
    ``layer``; ``send_after[layer]`` is the peer to send to after computing it.
    Both are derived ONCE from the schedule at construction, so the hot loop
    does two dict lookups and nothing else.
    """

    rank: int
    link: Any
    recv_before: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    send_after: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    timeout_s: float = 60.0
    crossings_sent: int = 0
    crossings_received: int = 0

    def __bool__(self) -> bool:
        return True

    def before_layer(self, layer_id: int, hidden, residual):
        """Receive the activations this rank needs to compute ``layer_id``."""
        peer = self.recv_before.get(layer_id)
        if peer is None:
            return hidden, residual
        src, slot = peer
        try:
            payload = self.link.recv(src, slot, self.timeout_s)
        except Exception as exc:  # noqa: BLE001 - re-raised as a named refusal
            raise PpCrossingWireError(
                f"rank {self.rank}: receiving the activations for layer "
                f"{layer_id} from rank {src} (slot {slot}) failed: {exc}. The "
                f"layer cannot be computed without them, and continuing would "
                f"silently skip the peer's layers -- which is the #753 defect."
            ) from exc
        if payload is None:
            raise PpCrossingWireError(
                f"rank {self.rank}: no activations arrived for layer "
                f"{layer_id} from rank {src} (slot {slot})."
            )
        self.crossings_received += 1
        return payload["hidden_states"], payload["residual"]

    def after_layer(self, layer_id: int, hidden, residual) -> None:
        """Send this layer's output to whoever owns the next one."""
        peer = self.send_after.get(layer_id)
        if peer is None:
            return None
        dst, slot = peer
        try:
            self.link.send(
                dst,
                slot,
                {"hidden_states": hidden, "residual": residual},
                self.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a named refusal
            raise PpCrossingWireError(
                f"rank {self.rank}: sending layer {layer_id}'s output to rank "
                f"{dst} (slot {slot}) failed: {exc}."
            ) from exc
        self.crossings_sent += 1
        return None

    def expected_sends(self) -> int:
        return len(self.send_after)

    def expected_receives(self) -> int:
        return len(self.recv_before)

    def describe(self) -> str:
        return (
            f"crossing wire rank {self.rank}: {len(self.send_after)} send(s), "
            f"{len(self.recv_before)} receive(s)"
        )


def build_crossing_wire(
    owned: Sequence[FrozenSet[int]],
    num_layers: int,
    rank: int,
    link: Any,
    *,
    timeout_s: float = 60.0,
    log: Optional[logging.Logger] = None,
    peer_map=None,
):
    """The wire for ``rank``, or :class:`NoCrossingWire` when none is needed.

    Derived from ``crossing_schedule`` -- the SAME function #735's pricing and
    routing consume -- so the wire cannot disagree with the cost model about
    which crossings exist. A schedule that says 31 crossings produces a wire
    whose per-rank sends and receives sum to 31.

    ``peer_map`` is optional. When given, the schedule is routed through it and
    ``log_routing`` is called, which is what finally makes crossings COUNTABLE:
    that function had zero callers, so "31 crossings observed" was a claim no
    run could evidence.
    """
    from sglang.srt.distributed.pp_crossing_schedule import crossing_schedule

    schedule = crossing_schedule(list(owned), num_layers)

    if peer_map is not None:
        # #753: the observable. route_schedule refuses an uncovered pair by
        # name, and log_routing prints one line per transport plus a WARNING
        # per degraded pair -- previously unreachable code.
        from sglang.srt.distributed.pp_crossing_transport import (
            log_routing,
            route_schedule,
        )

        routes = route_schedule(schedule, peer_map)
        log_routing(routes, log or logger)

    if not schedule:
        return NoCrossingWire()

    recv_before: Dict[int, Tuple[int, int]] = {}
    send_after: Dict[int, Tuple[int, int]] = {}
    for c in schedule:
        if c.src == rank:
            send_after[c.after_layer] = (c.dst, c.slot)
        if c.dst == rank:
            # The receiving rank needs it before the NEXT layer, which is the
            # first layer it owns after the boundary.
            recv_before[c.after_layer + 1] = (c.src, c.slot)

    if not recv_before and not send_after:
        return NoCrossingWire()

    wire = CrossingWire(
        rank=rank,
        link=link,
        recv_before=recv_before,
        send_after=send_after,
        timeout_s=timeout_s,
    )
    (log or logger).info("%s", wire.describe())
    return wire


class PpGroupLink:
    """``send``/``recv`` of one activation pair over the existing PP group.

    HOST-STAGED, and that is not a choice this class makes. `can_access_peer`
    is false for all six directed pairs on this rig, so NCCL falls back to host
    staging and every crossing is GPU -> host -> GPU
    (``ANALYSE_732_bar1_repricing.md:212-218``). #735 priced the schedule on
    exactly that path -- 24.68 ms/pass over 31 crossings -- so the wire and the
    cost model agree by construction rather than by coincidence.

    ``slot`` is carried by the schedule for a future transport that needs to
    match a send to its recv without a global sequence number. The PP group's
    tensor-dict path is already ordered per peer, so it is accepted and unused
    here rather than silently dropped somewhere less visible.
    """

    def __init__(self, pp_group):
        self.group = pp_group

    def send(self, dst: int, slot: int, payload, timeout_s: float) -> None:
        self.group.send_tensor_dict(payload, dst)

    def recv(self, src: int, slot: int, timeout_s: float):
        return self.group.recv_tensor_dict(src)


def build_wire_for_model(config, pp_group, log=None):
    """The model's entry point: a wire, or the null object.

    Returns :class:`NoCrossingWire` unless BOTH a layer set is configured and
    the #753 wire is switched on. The default path therefore constructs the
    null object and the forward loop's two calls are identity -- byte-identical
    to before this existed, with no branch in the loop to drift.
    """
    import os

    from sglang.srt.distributed.utils import (
        PP_LAYER_SET_ENV,
        parse_pp_layer_sets,
        pp_crossing_wire_enabled,
    )



    raw = os.getenv(PP_LAYER_SET_ENV, None)
    if raw is None or not raw.strip() or not pp_crossing_wire_enabled():
        return NoCrossingWire()

    owned = parse_pp_layer_sets(
        raw,
        config.num_hidden_layers,
        pp_group.world_size,
        allow_gapped=True,
    )
    return build_crossing_wire(
        owned,
        config.num_hidden_layers,
        pp_group.rank_in_group,
        PpGroupLink(pp_group),
        log=log,
    )
