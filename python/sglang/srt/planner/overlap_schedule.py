"""#704b: which collectives actually hide, and behind what.

DESIGN_704 sec 4.2a said the decoupling collectives are "overlappable behind
GDN/FFN compute", inheriting the canonical plan's phrasing. Pricing it turned
that into two different statements with very different values, because the
collectives are not one thing:

**Q-broadcast + partial-output gather + LSE merge is ON THE CRITICAL PATH.**
Attention layer ``L`` cannot produce its output until the merge completes, and
layer ``L+1`` consumes that output. So the 3 GDN layers that follow ``L`` cannot
hide ``L``'s gather -- they cannot start. Within one chunk this traffic is
simply exposed. The "48 GDN layers to hide behind" intuition counts compute
that is sequentially downstream of the very thing it is supposed to hide.

**KV placement is OFF the critical path.** The rows a stage ships to their
token-owner are a WRITE that no later layer in the same chunk reads, so it
overlaps the following GDN layers freely. On this rig that is ~0.3 ms of
traffic against ~37 ms of following GDN compute per attention layer: entirely
free, and it stays free at every rung.

So the only thing that hides the dominant term is **cross-chunk pipelining** --
overlapping chunk ``c``'s gather with chunk ``c+1``'s compute within the same
stage, bounded by that stage's own per-chunk compute time. That machinery does
not exist today, and this module prices what building it would buy.

SELF-LABELLED, per the measured/extrapolated discipline:

* link bandwidths are PLACEHOLDERS pending the pair-matrix probe (rank0 is x4,
  ranks 1-2 x8; the figures used are class estimates, not measurements);
* stage compute away from the incumbent cut is EXTRAPOLATED from a single
  measured cut with ``fixed_ms = 0``, which attributes all measured time to
  layers. That is the OPTIMISTIC end for hiding: a real fixed per-stage cost
  would mean less compute at deep cuts, hence less to hide behind. Every
  "hidden" verdict here is therefore an upper bound on how much hides.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

MIB = 1024.0 * 1024.0


class OverlapScheduleError(ValueError):
    """An overlap question that cannot be answered as posed."""


@dataclasses.dataclass(frozen=True)
class StageOverlap:
    stage: int
    attn_layers: int
    gather_mib: float
    gather_ms: float
    placement_mib: float
    placement_ms: float
    gdn_compute_ms: float
    placement_exposed_ms: float
    #: What survives with today's machinery: the gather is simply exposed.
    exposed_no_pipeline_ms: float
    #: What would survive if cross-chunk pipelining were built.
    exposed_pipelined_ms: float


@dataclasses.dataclass(frozen=True)
class OverlapSchedule:
    stages: tuple[StageOverlap, ...]
    worst_no_pipeline_ms: float
    worst_pipelined_ms: float
    worst_stage_no_pipeline: int

    @property
    def pipelining_speedup(self) -> float:
        """How much building cross-chunk pipelining is worth, as a ratio."""
        if self.worst_pipelined_ms <= 0.0:
            return float("inf")
        return self.worst_no_pipeline_ms / self.worst_pipelined_ms


def solve_overlap(
    attn_per_stage: Sequence[int],
    layers_per_stage: Sequence[int],
    ms_per_layer: Sequence[float],
    link_mib_per_s: Sequence[float],
    shares: Sequence[float],
    gather_mib_per_attn_layer: float,
    placement_mib_per_attn_layer: float,
) -> OverlapSchedule:
    """Per-stage exposed collective time for one chunk.

    ``gather_mib_per_attn_layer`` is the Q + partial-output + LSE volume for one
    attention layer (see ``decoupled_kv.collective_bytes_per_chunk``);
    ``placement_mib_per_attn_layer`` is the KV a stage produces for one
    attention layer before the ownership split is applied.
    """
    n = len(attn_per_stage)
    for name, seq in (
        ("layers_per_stage", layers_per_stage),
        ("ms_per_layer", ms_per_layer),
        ("link_mib_per_s", link_mib_per_s),
        ("shares", shares),
    ):
        if len(seq) != n:
            raise OverlapScheduleError(
                f"{name} covers {len(seq)} stages against {n} attention counts."
            )
    total_share = float(sum(shares))
    if total_share <= 0.0:
        raise OverlapScheduleError("the share vector sums to zero.")

    out: list[StageOverlap] = []
    for i in range(n):
        attn = int(attn_per_stage[i])
        layers = int(layers_per_stage[i])
        if attn > layers:
            raise OverlapScheduleError(
                f"stage {i} claims {attn} attention layers out of {layers} total."
            )
        bw = float(link_mib_per_s[i])
        if bw <= 0.0:
            raise OverlapScheduleError(f"stage {i} has a non-positive link.")
        share = float(shares[i]) / total_share

        gather_mib = attn * float(gather_mib_per_attn_layer)
        gather_ms = gather_mib / bw * 1000.0

        # Only the rows whose token-owner is another rank actually travel.
        placement_mib = attn * float(placement_mib_per_attn_layer) * (1.0 - share)
        placement_ms = placement_mib / bw * 1000.0

        # The GDN layers of this stage are what a WRITE can hide behind: they
        # run after each attention layer and do not read the shipped rows.
        gdn_ms = (layers - attn) * float(ms_per_layer[i])
        placement_exposed = max(0.0, placement_ms - gdn_ms)

        compute_ms = layers * float(ms_per_layer[i])
        no_pipe = gather_ms + placement_exposed
        pipelined = max(0.0, no_pipe - compute_ms)

        out.append(
            StageOverlap(
                stage=i,
                attn_layers=attn,
                gather_mib=gather_mib,
                gather_ms=gather_ms,
                placement_mib=placement_mib,
                placement_ms=placement_ms,
                gdn_compute_ms=gdn_ms,
                placement_exposed_ms=placement_exposed,
                exposed_no_pipeline_ms=no_pipe,
                exposed_pipelined_ms=pipelined,
            )
        )

    worst_np = max(s.exposed_no_pipeline_ms for s in out)
    worst_p = max(s.exposed_pipelined_ms for s in out)
    worst_i = max(range(n), key=lambda i: out[i].exposed_no_pipeline_ms)
    return OverlapSchedule(
        stages=tuple(out),
        worst_no_pipeline_ms=worst_np,
        worst_pipelined_ms=worst_p,
        worst_stage_no_pipeline=worst_i,
    )


# ---------------------------------------------------------------------------
# Link bandwidth binds to CARD IDENTITY, never to a rank index.
#
# This exists because getting it wrong cost a whole analysis. An earlier
# revision assumed "rank0 is on the x4 link" and concluded a triple jeopardy:
# rank0 carries the most attention layers, has the least compute to hide behind,
# AND sits on the slowest link. The authoritative mapping is the opposite -- the
# 5090 is on x8, one 3080 is on x8, the other 3080 is on x4 -- and with it the
# worst-rank exposure falls roughly fourfold and the "triple jeopardy" dissolves.
#
# Torch device order and NVML order diverge on this rig, so a rank index is not
# a card. The canon is registry/nvml.py's IdentityMap / CardIdentity (uuid,
# pci_bus_id, name, total_mib). A caller must therefore hand over a rank->UUID
# mapping explicitly; there is no positional shortcut, because the positional
# shortcut is the bug.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CardLink:
    """Measured host-link reach of ONE physical card, keyed by identity."""

    uuid: str
    pci_bus_id: str
    name: str
    total_mib: int
    lanes: int
    host_mib_per_s: float
    #: Provenance. False means estimated, and every number derived from it
    #: inherits that label rather than quietly becoming a measurement.
    measured: bool

    def __post_init__(self) -> None:
        if not self.uuid:
            raise OverlapScheduleError(
                "a CardLink without a UUID cannot be bound to a card; a rank "
                "index is not an identity (torch and NVML order diverge here)."
            )
        if self.host_mib_per_s <= 0.0:
            raise OverlapScheduleError(f"{self.uuid}: non-positive link reach.")


def links_for_stages(
    stage_card_uuids: Sequence[str], profile: Sequence[CardLink]
) -> tuple[float, ...]:
    """Resolve per-stage link reach through card identity.

    ``stage_card_uuids[i]`` is the NVML UUID of the card stage ``i`` runs on.
    Refuses an unknown UUID rather than falling back to a positional guess:
    a wrong card-to-link binding does not fail loudly, it just produces a
    confident and wrong schedule -- which is exactly what happened once.
    """
    by_uuid = {c.uuid: c for c in profile}
    if len(by_uuid) != len(profile):
        raise OverlapScheduleError("the link profile lists a UUID twice.")
    out: list[float] = []
    for i, uuid in enumerate(stage_card_uuids):
        card = by_uuid.get(uuid)
        if card is None:
            raise OverlapScheduleError(
                f"stage {i} runs on card {uuid!r}, which is not in the link "
                f"profile {sorted(by_uuid)}. Refusing to guess its link: "
                "binding link data by position is how a 4x-too-pessimistic "
                "schedule got published."
            )
        out.append(float(card.host_mib_per_s))
    return tuple(out)


def all_measured(profile: Sequence[CardLink]) -> bool:
    """True only if EVERY card's reach is measured; else results are estimates."""
    return all(c.measured for c in profile)
