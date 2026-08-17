# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Per-directed-pair transport binding for the PP crossing seam (#732).

The seam above this module picks a transport per COMMUNICATOR
(``barlink.py:554`` ``_select(op, nbytes)`` -- one ``self.transport`` for the
whole group, refined by op and size but never by peer). The #732 transport
survey concluded the opposite shape: the right choice is per PEER LINK,
because the measured BAR1 standing changes sign with edge width.

    "on the fast x8 pair (2 cards) the transport loses between 1 and 8 MiB,
     down to 0.81x. On the x4 pair and at three cards it wins everywhere.
     Pattern: the faster the edge, the worse the standing."
        -- FEATURES_VS_UPSTREAM.md:1349, "Honest weak spot"

On a rig whose crossing set straddles that boundary, one verdict for "the
crossing" is the wrong shape of answer. This module resolves the binding ONCE
at world build and hands the dispatch site an immutable, directed map.

WHAT THIS MODULE IS NOT
-----------------------
It moves no bytes and imports no CUDA. It is the desk half, like
``barlink_bar1_p2p.py`` beside it: resolution, refusal algebra and pricing,
wired into nothing, so existing layouts stay byte-for-byte. The BAR1 p2p
KERNEL does not exist yet (BAR1 owns three collective kernels and no p2p
kernel -- ``FEATURE_CATALOG.md`` section 7), so on today's tree every
BAR1-preferring edge degrades to NCCL. That degradation is the interesting
path, and it is loud by construction; see :class:`PeerBinding.degraded`.

DEVICE IDENTITY
---------------
Edges are keyed by GPU UUID, never by CUDA ordinal and never by NVML index.
CUDA enumerates FASTEST_FIRST and NVML in bus order, so on a mixed rig a
rank's CUDA ordinal is not its NVML index; #392 is what happens when those
are conflated. The bridge is the #397/#331 ``IdentityMap``
(``registry/nvml.py``) and nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "FAST_EDGE_LANES",
    "PeerTransport",
    "PeerBinding",
    "PeerTransportMap",
    "PeerTransportRefused",
    "lanes_by_uuid_via_nvml",
    "parse_peer_map_override",
    "resolve_peer_transports",
]


# The lane count at and above which an edge counts as FAST, and therefore as
# an edge where BAR1 is measured to LOSE. Not a tuning knob discovered by
# search: 8 is where the measured sign change sits on the reference rig
# (x4 pair -> BAR1 wins; x8 pair -> BAR1 loses to 0.81x). A rig with x16
# edges is "fast" by the same rule, which is the intended generalisation --
# the pattern in the source row is monotone ("the faster the edge, the worse
# the standing"), not a fact about the number 8.
FAST_EDGE_LANES = 8


class PeerTransport(str, Enum):
    """The transport bound to one directed pair."""

    NCCL_SENDRECV = "nccl_sendrecv"
    BAR1_P2P = "bar1_p2p"
    #: No transport is admissible for this pair. Recorded, not raised -- see
    #: :meth:`PeerTransportMap.require_no_refusals`.
    REFUSED = "refused"


class PeerTransportRefused(RuntimeError):
    """A pair resolved to :attr:`PeerTransport.REFUSED` and the caller asked."""


# ---------------------------------------------------------------------------
# Lane resolution
# ---------------------------------------------------------------------------


def lanes_by_uuid_via_nvml(uuid: str) -> Optional[int]:
    """Negotiated PCIe lane count for the SLOT this card sits in, or ``None``.

    WIDTH COMES FROM THE CURRENT LINK, NOT THE MAXIMUM, and the asymmetry is
    measured rather than stylistic. ``nvmlDeviceGetMaxPcieLinkWidth`` reports
    what the CARD can do, not what the SLOT gives it: on the reference rig it
    returns x16 for all three cards while the slots are wired x4 / x8 / x8.
    Deriving from it collapses every edge to "fast" and silently disables this
    whole feature on exactly the box it exists for.
    ``nvmlDeviceGetCurrPcieLinkWidth`` reports 4 / 8 / 8 and is the physical
    wiring.

    This canon is not invented here. It is the rule established at
    ``layers/moe/expert_offload.py:842`` (``_pcie_link_gbps_by_uuid``) for the
    expert-offload link weights, together with its #392 rationale. That
    function is private to another lineage and pulling ``layers.moe`` into
    world build would be the wrong dependency direction, so the canon is
    restated here and pinned by
    ``test_barlink_peer_transport.py::CanonTests``. Consolidating both onto a
    single helper in ``registry/nvml.py`` is filed, not done here.

    Unlike that function this one returns LANES, not GB/s: the policy below
    switches on physical width, and folding in a generation constant would
    turn an exact integer into an estimate for no gain.
    """
    try:
        from sglang.srt.registry.nvml import identity_map, nvml_session

        card = identity_map().require(uuid)
        with nvml_session() as pynvml:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(card.nvml_index))
            try:
                width = int(pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))
            except Exception:  # noqa: BLE001 - older binding: fall back below
                width = 0
            if width <= 0:
                # Deliberate last resort, and known to over-report (see above).
                # Better than None: an over-reported width picks NCCL, which is
                # never catastrophically wrong, only unoptimised.
                width = int(pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle))
    except Exception:  # noqa: BLE001 - absent driver/binding is not an error
        return None
    return width if width > 0 else None


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeerBinding:
    """The resolved transport for one DIRECTED pair, with its provenance."""

    src_uuid: str
    dst_uuid: str
    src_rank: int
    dst_rank: int
    #: Bottleneck lane count of the edge, ``None`` when unresolved.
    lanes: Optional[int]
    #: What the policy asked for, before availability was consulted.
    preferred: PeerTransport
    #: What the pair actually gets.
    transport: PeerTransport
    #: Why ``preferred`` was chosen.
    reason: str
    #: Set when ``transport != preferred``. The loud path.
    note: str = ""

    @property
    def degraded(self) -> bool:
        return self.transport is not self.preferred

    @property
    def refused(self) -> bool:
        return self.transport is PeerTransport.REFUSED

    def describe(self) -> str:
        lanes = "?" if self.lanes is None else f"x{self.lanes}"
        head = (
            f"rank {self.src_rank} -> {self.dst_rank}  {lanes:>3}  "
            f"{self.transport.value}"
        )
        if self.degraded:
            head += f"  (WANTED {self.preferred.value})"
        return head


class PeerTransportMap:
    """An immutable, directed pair -> transport binding, resolved once."""

    def __init__(self, bindings: Iterable[PeerBinding], *, source: str = "policy"):
        self._by_pair: Dict[Tuple[str, str], PeerBinding] = {}
        for b in bindings:
            key = (b.src_uuid, b.dst_uuid)
            if key in self._by_pair:
                raise ValueError(f"duplicate binding for directed pair {key}")
            self._by_pair[key] = b
        self.source = source

    def __len__(self) -> int:
        return len(self._by_pair)

    def __iter__(self):
        return iter(self._by_pair.values())

    def for_pair(self, src_uuid: str, dst_uuid: str) -> PeerBinding:
        """The binding for one DIRECTED pair.

        Raises ``KeyError`` on an unknown pair rather than returning a
        default. A missing pair means the world was built with a different
        card set than the one being dispatched on, and a silent default there
        is how a crossing ends up on the transport nobody chose.
        """
        try:
            return self._by_pair[(src_uuid, dst_uuid)]
        except KeyError:
            raise KeyError(
                f"no transport binding for directed pair {src_uuid} -> "
                f"{dst_uuid}; the map was resolved over "
                f"{len(self._by_pair)} pairs at world build"
            ) from None

    def degraded(self) -> Tuple[PeerBinding, ...]:
        return tuple(b for b in self._by_pair.values() if b.degraded and not b.refused)

    def refusals(self) -> Tuple[PeerBinding, ...]:
        return tuple(b for b in self._by_pair.values() if b.refused)

    def require_no_refusals(self) -> None:
        """Raise if any pair is :attr:`PeerTransport.REFUSED`.

        The refusal algebra deliberately RECORDS rather than raises, so the
        caller decides when a refused edge is fatal. This is the explicit
        ask.
        """
        bad = self.refusals()
        if not bad:
            return
        lines = "\n".join(f"  {b.describe()}: {b.note}" for b in bad)
        raise PeerTransportRefused(
            f"{len(bad)} directed pair(s) have no admissible transport:\n{lines}"
        )

    def describe(self) -> str:
        rows = sorted(
            self._by_pair.values(), key=lambda b: (b.src_rank, b.dst_rank)
        )
        out = [f"peer transport map ({self.source}, {len(rows)} directed pairs)"]
        out += [f"  {b.describe()}" for b in rows]
        return "\n".join(out)

    def log_decisions(self, log: Optional[logging.Logger] = None) -> None:
        """One INFO summary, and one WARNING per degraded or refused edge.

        The WARNING is the point. A BAR1-preferring edge that silently ran on
        NCCL would look exactly like a rig where the policy never applied, and
        the whole value of the per-link split is invisible in that case.
        """
        log = log or logger
        counts: Dict[str, int] = {}
        for b in self._by_pair.values():
            counts[b.transport.value] = counts.get(b.transport.value, 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        log.info(
            "barlink peer transport map (%s): %d directed pairs, %s",
            self.source,
            len(self._by_pair),
            summary,
        )
        for b in self._by_pair.values():
            if b.refused:
                log.warning("barlink peer transport REFUSED: %s -- %s", b.describe(), b.note)
            elif b.degraded:
                log.warning("barlink peer transport FALLBACK: %s -- %s", b.describe(), b.note)


# ---------------------------------------------------------------------------
# Override flag
# ---------------------------------------------------------------------------


def parse_peer_map_override(spec: Optional[str]) -> Dict[Optional[Tuple[int, int]], PeerTransport]:
    """Parse ``SGLANG_BARLINK_PEER_MAP`` into a rank-keyed override table.

    Grammar, comma separated::

        all=nccl_sendrecv          # every directed pair
        0>1=bar1_p2p               # one directed pair, by RANK
        all=bar1_p2p,2>0=nccl_sendrecv

    Keys are RANKS, not CUDA ordinals and not NVML indices: the rank -> UUID
    binding is the world list handed to :func:`resolve_peer_transports`, so a
    rank index here never touches the device-order trap. It is the friendlier
    handle for an A/B, which is what this flag is for.

    ``all`` is applied first regardless of position, so an explicit pair
    always wins over it.
    """
    table: Dict[Optional[Tuple[int, int]], PeerTransport] = {}
    if not spec or not spec.strip():
        return table
    valid = {t.value for t in PeerTransport}
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"peer map override {item!r}: expected '<key>=<transport>', "
                f"where key is 'all' or '<src>><dst>'"
            )
        key_s, _, val_s = item.partition("=")
        key_s, val_s = key_s.strip(), val_s.strip()
        if val_s not in valid:
            raise ValueError(
                f"peer map override {item!r}: unknown transport {val_s!r}; "
                f"known: {sorted(valid)}"
            )
        transport = PeerTransport(val_s)
        if key_s == "all":
            table[None] = transport
            continue
        if ">" not in key_s:
            raise ValueError(
                f"peer map override {item!r}: pair key must be '<src>><dst>', "
                f"e.g. '0>1'"
            )
        a, _, b = key_s.partition(">")
        try:
            pair = (int(a), int(b))
        except ValueError:
            raise ValueError(
                f"peer map override {item!r}: ranks must be integers"
            ) from None
        if pair[0] == pair[1]:
            raise ValueError(
                f"peer map override {item!r}: a rank has no edge to itself"
            )
        table[pair] = transport
    return table


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

#: Stated where the fallback is priced, so nobody reads the absence of a
#: number as an absence of cost. The share of crossings riding a degraded
#: edge is owned by the crossing schedule, not by this module.
_UNPRICED = (
    "BAR1-vs-NCCL margin on sub-x{lanes} edges is UNMEASURED (gap 8 of "
    "docs/dev/NOTE_732_transport_selection.md); the exposure is the share of "
    "crossings on this edge, which the crossing schedule owns"
)


def _policy_for(lanes: Optional[int]) -> Tuple[PeerTransport, str]:
    """Default binding for an edge of ``lanes`` width."""
    if lanes is None:
        return (
            PeerTransport.NCCL_SENDRECV,
            "edge width unresolved; NCCL is the choice that is never "
            "catastrophically wrong, only unoptimised",
        )
    if lanes >= FAST_EDGE_LANES:
        return (
            PeerTransport.NCCL_SENDRECV,
            f"fast edge (x{lanes} >= x{FAST_EDGE_LANES}): BAR1 is measured to "
            f"LOSE here, 1-8 MiB, down to 0.81x "
            f"(FEATURES_VS_UPSTREAM.md:1349)",
        )
    return (
        PeerTransport.BAR1_P2P,
        f"slow edge (x{lanes} < x{FAST_EDGE_LANES}): BAR1 is measured to win "
        f"everywhere on the x4 pair (FEATURES_VS_UPSTREAM.md:1349)",
    )


def resolve_peer_transports(
    uuids: Sequence[str],
    *,
    lanes_by_uuid: Callable[[str], Optional[int]] = lanes_by_uuid_via_nvml,
    bar1_p2p_available: bool = False,
    override: Optional[Mapping[Optional[Tuple[int, int]], PeerTransport]] = None,
) -> PeerTransportMap:
    """Resolve the directed transport map ONCE, at world build.

    ``uuids[rank]`` is the GPU UUID serving that rank -- the caller gathers it
    (a rank knows its own via ``registry.nvml.current_device_uuid``). Both
    ``lanes_by_uuid`` and ``bar1_p2p_available`` are injectable so the
    hermetic tests can construct a rig whose CUDA and NVML orders deliberately
    disagree without a driver, the same shape ``registry.nvml.identity_map``
    already supports.

    ``bar1_p2p_available`` defaults to ``False`` because on today's tree it IS
    false: BAR1 owns three collective kernels and no p2p kernel. Passing the
    default therefore reproduces the shipping behaviour -- every BAR1-preferring
    edge degrades to NCCL, loudly.
    """
    if len(set(uuids)) != len(uuids):
        # Two ranks on one card is a legitimate configuration elsewhere, but
        # it makes "the edge between rank a and rank b" ambiguous here.
        raise ValueError(
            f"peer transport map needs one card per rank; got {len(uuids)} "
            f"ranks over {len(set(uuids))} distinct UUIDs"
        )
    override = dict(override or {})
    lanes_cache = {u: lanes_by_uuid(u) for u in uuids}

    bindings = []
    for src, src_uuid in enumerate(uuids):
        for dst, dst_uuid in enumerate(uuids):
            if src == dst:
                continue
            a, b = lanes_cache[src_uuid], lanes_cache[dst_uuid]
            # The bottleneck of the path, not the card. An x8 card talking to
            # an x4 card gets x4 of usable width whichever way the bytes go.
            lanes = None if (a is None or b is None) else min(a, b)

            forced = override.get((src, dst), override.get(None))
            if forced is not None:
                scope = "all" if (src, dst) not in override else f"{src}>{dst}"
                preferred = forced
                reason = f"forced by SGLANG_BARLINK_PEER_MAP ({scope}=)"
            else:
                preferred, reason = _policy_for(lanes)

            transport, note = preferred, ""
            if preferred is PeerTransport.BAR1_P2P and not bar1_p2p_available:
                if forced is not None:
                    # An explicit A/B asked for a transport that cannot run.
                    # Degrading here would silently answer a different
                    # question than the one the operator posed.
                    transport = PeerTransport.REFUSED
                    note = (
                        "SGLANG_BARLINK_PEER_MAP forced bar1_p2p but the BAR1 "
                        "p2p kernel is unavailable; refusing rather than "
                        "answering a different A/B than the one asked"
                    )
                else:
                    transport = PeerTransport.NCCL_SENDRECV
                    note = (
                        "BAR1 p2p kernel unavailable (BAR1 owns three "
                        "collective kernels and no p2p kernel); falling back "
                        "to NCCL. "
                        + _UNPRICED.format(lanes=FAST_EDGE_LANES)
                    )

            bindings.append(
                PeerBinding(
                    src_uuid=src_uuid,
                    dst_uuid=dst_uuid,
                    src_rank=src,
                    dst_rank=dst,
                    lanes=lanes,
                    preferred=preferred,
                    transport=transport,
                    reason=reason,
                    note=note,
                )
            )

    source = "override" if override else "policy"
    return PeerTransportMap(bindings, source=source)
