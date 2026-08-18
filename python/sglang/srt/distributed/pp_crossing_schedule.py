# SPDX-License-Identifier: Apache-2.0
"""Crossing schedule for non-contiguous pipeline stages (the family full plan).

A contiguous pipeline sends once per stage boundary: `pp_size - 1` crossings
per forward, and the schedule is so obvious nobody writes it down. Under an
explicit layer SET (``SGLANG_PP_LAYER_SET``) ownership changes wherever the map
says it does, so the schedule becomes a real object: **send at every ownership
change**, in layer order.

For the family plan on this rig -- 48 linear-attention layers on the 5090, the
16 interleaved full-attention layers on the two 3080s -- that is 31 crossings
per token, not 2, and this module is what says exactly where they fall.

WHAT THIS MODULE IS NOT. It does not move bytes and does not know a transport.
It computes a schedule and drives it through a :class:`Link` SHAPE, so it can
be built and tested before any wire exists. The real transport (barlink
send/recv) is a separate build; a schedule written against a guessed interface
is how two halves end up disagreeing, so the interface here is deliberately the
smallest thing both sides must agree on: a per-pair slot, a bounded wait, and a
refusal that names itself.

THE TERMINAL-LAYER LEVER, worth knowing before choosing a map: the LAST layer
of the model has no return crossing -- its output goes to the head, not back
into another stage. So whichever stage owns the final layer is owed one fewer
crossing than its layer count suggests, and placing that layer on the SLOWEST
link is free. On this rig that is the x4-linked card.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, FrozenSet, List, Optional, Protocol, Sequence, Tuple

__all__ = [
    "Crossing",
    "CrossingScheduleError",
    "Link",
    "LoopbackLink",
    "crossing_schedule",
    "schedule_cost",
    "stage_of_layer",
]


class CrossingScheduleError(RuntimeError):
    """A schedule that cannot be built or driven. Never silent."""


@dataclasses.dataclass(frozen=True)
class Crossing:
    """One activation handoff.

    ``after_layer`` is the last layer computed on ``src`` before the handoff,
    so the crossing carries that layer's output. ``slot`` is the per-pair
    ordinal: the n-th crossing between this ordered pair within one forward,
    which is what lets a transport match a send to its recv without a global
    sequence number.
    """

    after_layer: int
    src: int
    dst: int
    slot: int

    @property
    def pair(self) -> Tuple[int, int]:
        return (self.src, self.dst)


def stage_of_layer(owned: Sequence[FrozenSet[int]], num_layers: int) -> List[int]:
    """``layer id -> owning stage``, validated.

    The validation duplicates ``parse_pp_layer_sets`` on purpose: this function
    is also reachable with a map built in code rather than parsed from the
    environment, and an unowned layer here would produce a schedule that skips
    a crossing rather than an obvious error.
    """
    out: List[int] = [-1] * num_layers
    for stage, layers in enumerate(owned):
        for layer in layers:
            if not 0 <= layer < num_layers:
                raise CrossingScheduleError(
                    f"layer {layer} is outside [0, {num_layers})"
                )
            if out[layer] != -1:
                raise CrossingScheduleError(
                    f"layer {layer} is claimed by stage {out[layer]} and stage "
                    f"{stage}; a layer has exactly one owner"
                )
            out[layer] = stage
    unowned = [i for i, s in enumerate(out) if s == -1]
    if unowned:
        raise CrossingScheduleError(
            f"layer(s) {unowned} have no owner, so the schedule would silently "
            f"skip the crossing into them"
        )
    return out


def crossing_schedule(
    owned: Sequence[FrozenSet[int]], num_layers: int
) -> List[Crossing]:
    """Every ownership change, in layer order.

    There is no crossing after the FINAL layer: its output leaves the pipeline
    for the head. That single exception is why a 64-layer 3-GDN-then-1-FA map
    yields 31 crossings and not 32.
    """
    owner = stage_of_layer(owned, num_layers)
    per_pair: Dict[Tuple[int, int], int] = {}
    out: List[Crossing] = []
    for layer in range(num_layers - 1):
        src, dst = owner[layer], owner[layer + 1]
        if src == dst:
            continue
        slot = per_pair.get((src, dst), 0)
        per_pair[(src, dst)] = slot + 1
        out.append(Crossing(after_layer=layer, src=src, dst=dst, slot=slot))
    return out


def schedule_cost(
    schedule: Sequence[Crossing],
    per_pair_us: Dict[Tuple[int, int], float],
    default_us: float,
) -> float:
    """Total microseconds for one pass, priced PER LINK.

    Per-link rather than one flat number because the rig's edges are not
    alike: the two x8 edges and the single x4 edge differ enough that a
    schedule's cost depends on WHICH pairs it uses, not only how many crossings
    it makes. A pair with no entry falls back to ``default_us`` rather than
    being priced at zero -- an unpriced edge is a modelling gap, and zero would
    hide it.
    """
    return sum(per_pair_us.get(c.pair, default_us) for c in schedule)


class Link(Protocol):
    """The smallest interface a transport must satisfy to drive a schedule.

    Deliberately not a byte mover: ``send``/``recv`` take an opaque payload so
    this module never assumes a tensor library, a device or a wire format.
    """

    def send(self, dst: int, slot: int, payload, timeout_s: float) -> None: ...

    def recv(self, src: int, slot: int, timeout_s: float): ...


class LoopbackLink:
    """In-process test double (the #111 KvLink precedent).

    Holds one queue per (src, dst, slot) so a schedule can be driven end to end
    with no devices and no process group. It enforces the two properties the
    real transport must also have, because a double that is more permissive
    than the thing it stands for tests nothing:

    * a recv that finds nothing REFUSES BY NAME rather than blocking forever --
      the bounded-wait requirement, made visible;
    * a slot may be delivered once; a second recv on the same slot refuses,
      which is what catches a schedule that double-consumes a crossing.
    """

    def __init__(self) -> None:
        self._slots: Dict[Tuple[int, int, int], object] = {}
        self.sent: List[Tuple[int, int, int]] = []
        self.received: List[Tuple[int, int, int]] = []

    def send(self, dst: int, slot: int, payload, timeout_s: float) -> None:
        key = (getattr(self, "_src", 0), dst, slot)
        if key in self._slots:
            raise CrossingScheduleError(
                f"slot {slot} for pair {key[:2]} is already occupied; a "
                f"schedule must not reuse a slot within one pass"
            )
        self._slots[key] = payload
        self.sent.append(key)

    def recv(self, src: int, slot: int, timeout_s: float):
        key = (src, getattr(self, "_dst", 0), slot)
        if key not in self._slots:
            raise CrossingScheduleError(
                f"no payload on slot {slot} for pair {key[:2]} within "
                f"{timeout_s}s; the sender never arrived or the schedules "
                f"disagree about the slot numbering"
            )
        self.received.append(key)
        return self._slots.pop(key)

    # -- driving a whole schedule, for tests -----------------------------

    def run(self, schedule: Sequence[Crossing], payload_of, timeout_s: float = 1.0):
        """Send and receive every crossing in order; return what was received.

        This is the property a real driver must also have: each crossing is
        sent by its src and consumed exactly once by its dst, in schedule
        order.
        """
        out = []
        for c in schedule:
            self._src, self._dst = c.src, c.dst
            self.send(c.dst, c.slot, payload_of(c), timeout_s)
            self._src, self._dst = c.src, c.dst
            out.append(self.recv(c.src, c.slot, timeout_s))
        return out
