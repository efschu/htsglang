# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""The junction: a PP crossing schedule driven over a per-peer transport map.

Two desk-proven halves meet here for the first time (#735).

* ``pp_crossing_schedule.py`` answers WHICH crossings happen and BETWEEN WHICH
  RANKS. It was built against the ``Link`` *shape* with an in-process
  ``LoopbackLink`` double, deliberately, so a schedule could be proved before a
  wire existed. It knows nothing about devices.
* ``device_communicators/barlink_peer_transport.py`` answers WHICH TRANSPORT a
  directed pair should use, resolved once at world build from NVML link
  identity. It knows nothing about layers.

Neither should learn the other's vocabulary, so the conversion between them --
rank pairs to device-keyed bindings -- happens here and only here.

WHAT THIS MODULE IS NOT
-----------------------
It still moves no bytes. The BAR1 p2p kernel does not exist (BAR1 owns three
collective kernels and no p2p kernel), and NCCL ``send``/``recv`` is not wired
into this seam either. What this module does is make every crossing resolve to
a NAMED transport, or refuse by name -- so that the day a wire lands, the
routing above it is already proved and the only new thing is the byte mover.

A ``RoutedLink`` built without a backend for the transport a pair resolved to
therefore refuses on first use, loudly and by name. That is the intended
shipping behaviour today, not a gap: the alternative -- quietly running every
crossing over whatever single transport happens to be available -- is exactly
the per-communicator assumption #732 was written to remove.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from sglang.srt.distributed.device_communicators.barlink_peer_transport import (
    PeerBinding,
    PeerTransport,
    PeerTransportMap,
)
from sglang.srt.distributed.pp_crossing_schedule import (
    Crossing,
    CrossingScheduleError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MEASURED_GBPS_BY_LANES",
    "CrossingRoute",
    "RoutedLink",
    "UnroutableCrossing",
    "log_routing",
    "per_pair_us_from_map",
    "route_schedule",
]


class UnroutableCrossing(CrossingScheduleError):
    """A crossing whose pair has no admissible transport.

    Subclasses the schedule's own error type on purpose: a caller already
    catching ``CrossingScheduleError`` around a pass keeps working, and the
    routing layer does not introduce a second exception vocabulary for the same
    failure mode.
    """


@dataclass(frozen=True)
class CrossingRoute:
    """One crossing, and the transport its pair resolved to."""

    crossing: Crossing
    binding: PeerBinding

    @property
    def transport(self) -> PeerTransport:
        return self.binding.transport

    @property
    def degraded(self) -> bool:
        return self.binding.degraded

    @property
    def refused(self) -> bool:
        return self.binding.refused

    def describe(self) -> str:
        c = self.crossing
        return (
            f"after layer {c.after_layer:>3}: rank {c.src} -> {c.dst} "
            f"slot {c.slot}  {self.transport.value}"
        )


def route_schedule(
    schedule: Sequence[Crossing], peer_map: PeerTransportMap
) -> Tuple[CrossingRoute, ...]:
    """Attach a transport binding to every crossing, or refuse by name.

    Every crossing must resolve. A schedule whose ranks the map does not cover
    is a world built from a different card set than the one being dispatched
    on, and the failure belongs at routing time -- not at the first send, half
    a forward pass later.
    """
    routes = []
    for c in schedule:
        try:
            binding = peer_map.for_ranks(c.src, c.dst)
        except KeyError as exc:
            raise UnroutableCrossing(
                f"crossing after layer {c.after_layer} (rank {c.src} -> "
                f"{c.dst}, slot {c.slot}) has no transport binding: {exc}"
            ) from None
        routes.append(CrossingRoute(crossing=c, binding=binding))
    return tuple(routes)


def log_routing(
    routes: Sequence[CrossingRoute], log: Optional[logging.Logger] = None
) -> None:
    """One INFO summary, and one WARNING per degraded or refused PAIR.

    Per pair, not per crossing: a 31-crossing schedule over three ranks would
    otherwise emit the same fallback line fifteen times and bury it. The count
    of affected crossings goes IN the line instead, because that count is the
    exposure -- it is what a degraded edge actually costs the pass.
    """
    log = log or logger
    by_transport: Dict[str, int] = {}
    for r in routes:
        by_transport[r.transport.value] = by_transport.get(r.transport.value, 0) + 1
    log.info(
        "pp crossing routing: %d crossings, %s",
        len(routes),
        ", ".join(f"{k}={v}" for k, v in sorted(by_transport.items())),
    )

    seen: Dict[Tuple[int, int], int] = {}
    for r in routes:
        if r.degraded or r.refused:
            seen[r.crossing.pair] = seen.get(r.crossing.pair, 0) + 1
    for pair, count in sorted(seen.items()):
        binding = next(r.binding for r in routes if r.crossing.pair == pair)
        kind = "REFUSED" if binding.refused else "FALLBACK"
        log.warning(
            "pp crossing %s: rank %d -> %d carries %d of %d crossings -- %s",
            kind,
            pair[0],
            pair[1],
            count,
            len(routes),
            binding.note or binding.reason,
        )


#: Measured NCCL point-to-point throughput, GB/s, keyed by the edge's
#: bottleneck lane count. From the hardware profile's pairwise link matrix
#: (``uneven_perf.py`` ``p2p_{a}_{b}_gbs``: ``dist.send``/``recv`` at exactly
#: 512K bf16 = 1 MiB), four independent boot caches agreeing within 1.5 %.
#:
#: x8 is 9.06. x4 takes the CONSERVATIVE of the two measured x4-bottleneck
#: edges -- 5.10 (5090 <-> x4 3080) rather than 5.82 (x8 3080 <-> x4 3080) --
#: because lane count alone does not determine throughput and picking the
#: faster of two would flatter the model.
#:
#: These are 1 MiB rows used for a ~5 MiB payload. Per
#: ``NOTE_732_transport_selection.md`` section 1.5(b) the curve is flat from
#: ~4 MiB and 1 MiB sits 1-6 % below asymptote, so the extrapolation is bounded
#: and its error has a KNOWN SIGN: costs come out mildly pessimistic, never
#: optimistic.
#:
#: Any lane count not in this table is deliberately absent rather than
#: interpolated -- see :func:`per_pair_us_from_map`.
MEASURED_GBPS_BY_LANES: Mapping[int, float] = {4: 5.10, 8: 9.06}


def per_pair_us_from_map(
    peer_map: PeerTransportMap,
    payload_bytes: int,
    *,
    gbps_by_lanes: Mapping[int, float] = MEASURED_GBPS_BY_LANES,
) -> Dict[Tuple[int, int], float]:
    """``{(src_rank, dst_rank): microseconds}`` for :func:`schedule_cost`.

    Priced from the SAME map the transport routing uses, so a schedule cannot
    be costed against one edge assignment and driven over another.

    A pair whose lane count has no measured entry is OMITTED, not interpolated
    and not zeroed. ``schedule_cost`` already treats a missing pair as
    ``default_us`` and its docstring says why -- "an unpriced edge is a
    modelling gap, and zero would hide it". Omitting is how this function hands
    that gap over intact instead of inventing a number to fill it.
    """
    out: Dict[Tuple[int, int], float] = {}
    for b in peer_map:
        if b.lanes is None:
            continue
        gbps = gbps_by_lanes.get(b.lanes)
        if gbps is None:
            continue
        out[(b.src_rank, b.dst_rank)] = payload_bytes / (gbps * 1e9) * 1e6
    return out


class RoutedLink:
    """A rank-local ``Link`` that dispatches per DIRECTED pair.

    This is the object the #732 recommendation needed and the seam did not
    have: ``barlink.py``'s ``_select`` picks one transport per communicator,
    refined by op and size but never by peer, while the measured BAR1 standing
    changes sign with edge width. Here the peer is the first thing consulted.

    ``backends`` maps a transport to the object that actually moves the bytes.
    It is deliberately incomplete today: with no BAR1 p2p kernel, a map that
    routes an x4 edge to ``bar1_p2p`` and a ``backends`` without that key means
    the first crossing on that edge refuses BY NAME. Compare the alternative --
    silently sending it over whatever single backend exists -- which is the
    per-communicator behaviour this whole line of work removes.
    """

    def __init__(
        self,
        rank: int,
        peer_map: PeerTransportMap,
        backends: Mapping[PeerTransport, object],
    ):
        self.rank = rank
        self.peer_map = peer_map
        self.backends = dict(backends)

    def _backend_for(self, src: int, dst: int):
        try:
            binding = self.peer_map.for_ranks(src, dst)
        except KeyError as exc:
            raise UnroutableCrossing(str(exc)) from None
        if binding.refused:
            raise UnroutableCrossing(
                f"rank {src} -> {dst} resolved to REFUSED: {binding.note}"
            )
        backend = self.backends.get(binding.transport)
        if backend is None:
            raise UnroutableCrossing(
                f"rank {src} -> {dst} resolved to {binding.transport.value}, "
                f"but no backend is registered for it "
                f"(have: {sorted(t.value for t in self.backends)}). "
                f"Refusing rather than sending over a transport nobody chose."
            )
        return backend

    def send(self, dst: int, slot: int, payload, timeout_s: float) -> None:
        # Direction matters: this rank is the SOURCE of a send.
        self._backend_for(self.rank, dst).send(dst, slot, payload, timeout_s)

    def recv(self, src: int, slot: int, timeout_s: float):
        # ...and the DESTINATION of a recv. Looking the pair up the other way
        # round would silently price and route the reverse edge, which on an
        # asymmetric map is a different transport.
        return self._backend_for(src, self.rank).recv(src, slot, timeout_s)
