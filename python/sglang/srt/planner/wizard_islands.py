# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Families for rigs whose cards are not all equally close to each other.

WHAT AN ISLAND IS
-----------------
A set of cards that talk to each other over a fast path -- NVLink, or PCIe
with peer access -- while talking to everything outside the set over a slow
one. Two NVLink-bridged pairs in a four-card box are two islands. Eight cards
with an NVSwitch are one. Three consumer cards on a consumer chipset with no
peer access at all are three islands of one card each, which is this rig.

The boundary matters because a TP collective pays for its slowest edge. A
group laid out inside an island and a group straddling two islands are not
the same configuration with different numbers; they are different
configurations, and a wizard that cannot express the difference cannot advise
anybody who owns such hardware.

WE CANNOT MEASURE THIS, AND THAT IS NOT A REASON TO SAY NOTHING
---------------------------------------------------------------
There is no NVLink on this rig and no GPUDirect P2P: every pair is PHB and
every collective is host-staged. The standing rule is that the local rig is a
LOWER BOUND and never a verdict about a general feature -- other people have
NVLink, and refusing to model their hardware would be turning our cabling
into everybody's ceiling.

So these families are offered as ESTIMATES with an origin that says exactly
what they are: modelled from the pair-matrix STRUCTURE (which edges are fast,
which are slow, how many cards a collective crosses) using the interconnect
discount ladder that already exists in :mod:`sglang.srt.planner.roofline`.
Three things follow, and all three are load-bearing:

* never ``absent``. "We cannot measure it here" is not the same as "nobody
  can know it", and refusing to answer is not honesty, it is unhelpfulness
  wearing honesty's clothes.
* never a made-up number. Every figure is a ratio between two rungs of a
  ladder that is in the tree, was calibrated (crudely, and it says so) against
  our own measured decode, and is quoted with the caveat it carries there.
* the RATIO is the answer, not the absolute. What these families can honestly
  say is "respecting the island boundary is worth about this much on the
  collective term", not "you will get N tok/s".

LANES
-----
An island topology is the natural place where more than one group appears --
one group per island is the obvious layout the moment a model fits inside
one. So these families produce :class:`~sglang.srt.planner.wizard_lanes.LaneSet`
values with as many lanes as the topology has, not with two.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sglang.srt.planner import wizard_lanes as lanes_mod
from sglang.srt.planner.bench_factors import ABSENT, ESTIMATE, MEASURED

__all__ = [
    "IslandTopology",
    "TIERS",
    "collective_discount",
    "islands_from_pairs",
    "island_families",
]

#: Transport tiers, fastest first. The names are the ones ``roofline``'s
#: ``_interconnect`` uses, so a reader can find the same word in both files.
TIERS: Tuple[str, ...] = ("nvlink", "pcie-p2p", "pcie-host-staging")

_LADDER_CAVEAT = (
    "The ladder is a crude heuristic and says so where it is defined: it was "
    "calibrated against this fork's own measured hetero-TP and weightless-lane "
    "decode, which ran far below the single-card roofline. It ranks "
    "interconnects; it does not predict a rate."
)


def _ladder() -> Dict[str, Any]:
    """The discount constants, read from ``roofline`` rather than copied.

    Read through ``getattr`` so a rename over there fails a test here instead
    of leaving two ladders that slowly disagree.
    """
    from sglang.srt.planner import roofline

    return {
        "nvlink": getattr(roofline, "_NVLINK_DISCOUNT"),
        "pcie_p2p": getattr(roofline, "_PCIE_P2P_DISCOUNT"),
        "nop2p_by_cards": getattr(roofline, "_PCIE_NOP2P_BY_CROSS_CARDS"),
        "nop2p_many": getattr(roofline, "_PCIE_NOP2P_MANY"),
    }


def collective_discount(tier: str, n_cross: int) -> float:
    """The TP-collective knock-down for ``n_cross`` cards over ``tier``.

    One card is one card: no collective, no discount. Everything else comes
    off the ladder in ``roofline``.
    """
    if n_cross <= 1:
        return 1.0
    lad = _ladder()
    if tier == "nvlink":
        return float(lad["nvlink"])
    if tier == "pcie-p2p":
        return float(lad["pcie_p2p"])
    by_cards: Dict[int, float] = dict(lad["nop2p_by_cards"])
    return float(by_cards.get(n_cross, lad["nop2p_many"]))


# ---------------------------------------------------------------------------
# The topology
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class IslandTopology:
    """Which cards are close to which, and how that was established."""

    #: Card ordinals, grouped. Always a partition of the card set.
    islands: Tuple[Tuple[int, ...], ...]
    #: Transport inside an island, and between islands.
    intra_tier: str
    inter_tier: str
    #: ``measured`` when a pair matrix established it, ``estimate`` when the
    #: reader described the hardware instead.
    provenance: str
    source: str
    #: True only when this rig could actually observe the fast edges. On a
    #: box with no peer access at all it is False, and every figure derived
    #: from the topology says so.
    measurable_here: bool = True
    note: str = ""

    @property
    def card_count(self) -> int:
        return sum(len(i) for i in self.islands)

    @property
    def island_count(self) -> int:
        return len(self.islands)

    @property
    def has_islands(self) -> bool:
        """More than one island, and at least one of them holds >1 card.

        A rig where every card is its own island has no island STRUCTURE to
        respect -- every edge is slow -- and saying "you have three islands"
        there would dress a flat topology up as a hierarchical one.
        """
        return self.island_count > 1 and any(len(i) > 1 for i in self.islands)

    def to_json(self) -> dict:
        return {
            "islands": [list(i) for i in self.islands],
            "island_count": self.island_count,
            "card_count": self.card_count,
            "intra_tier": self.intra_tier,
            "inter_tier": self.inter_tier,
            "provenance": self.provenance,
            "source": self.source,
            "measurable_here": self.measurable_here,
            "has_islands": self.has_islands,
            "note": self.note,
        }


def _tier_of(transport: str, peer_access: bool) -> str:
    t = (transport or "").lower()
    if "nvlink" in t:
        return "nvlink"
    if peer_access or "p2p" in t:
        return "pcie-p2p"
    return "pcie-host-staging"


def islands_from_pairs(
    cards: Sequence[dict], pairs: Sequence[dict]
) -> IslandTopology:
    """Partition the measured cards by the fast edges between them.

    An edge counts as fast when the driver reports peer access in that
    direction or the transport names NVLink. Everything else is a slow edge,
    and cards joined by nothing but slow edges end up in separate islands.
    The partition is a plain union over the fast edges -- islands are an
    equivalence, so a chain of fast edges is one island even where no single
    pair spans it.
    """
    uuid_by_ordinal: Dict[str, int] = {}
    for i, c in enumerate(
        sorted(cards, key=lambda x: str(x.get("uuid") or x.get("index") or ""))
    ):
        uuid_by_ordinal[str(c.get("uuid") or c.get("index") or i)] = i
    n = len(uuid_by_ordinal)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    fast_tiers: List[str] = []
    for p in pairs:
        s = uuid_by_ordinal.get(str(p.get("src_uuid") or ""))
        d = uuid_by_ordinal.get(str(p.get("dst_uuid") or ""))
        if s is None or d is None:
            continue
        tier = _tier_of(str(p.get("transport") or ""), bool(p.get("peer_access")))
        if tier in ("nvlink", "pcie-p2p"):
            fast_tiers.append(tier)
            union(s, d)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    islands = tuple(tuple(sorted(v)) for _, v in sorted(groups.items()))
    intra = "nvlink" if "nvlink" in fast_tiers else (
        "pcie-p2p" if fast_tiers else "pcie-host-staging"
    )
    measurable = bool(fast_tiers)
    return IslandTopology(
        islands=islands,
        intra_tier=intra,
        inter_tier="pcie-host-staging",
        provenance=MEASURED if pairs else ESTIMATE,
        source=(
            "the card probe's ordered pair matrix, read for peer access and "
            "transport per edge"
            if pairs
            else "no pair matrix on disk"
        ),
        measurable_here=measurable,
        note=(
            ""
            if measurable
            else "no pair on this host has peer access, so there are no fast "
            "edges to group by: every card is its own island. That is a "
            "finding about this cabling, not about the method"
        ),
    )


def described_topology(
    island_sizes: Sequence[int], *, intra_tier: str = "nvlink"
) -> IslandTopology:
    """A topology the reader described rather than one we measured.

    The entry point for "I have two NVLink-bridged pairs" -- hardware nobody
    here owns. Provenance is ``estimate`` from the first line, and every
    figure derived from it inherits that.
    """
    islands: List[Tuple[int, ...]] = []
    nxt = 0
    for size in island_sizes:
        islands.append(tuple(range(nxt, nxt + int(size))))
        nxt += int(size)
    return IslandTopology(
        islands=tuple(islands),
        intra_tier=intra_tier if intra_tier in TIERS else "nvlink",
        inter_tier="pcie-host-staging",
        provenance=ESTIMATE,
        source="described on the form; no pair matrix for this hardware",
        measurable_here=False,
        note=(
            "This machine is not the one running the dashboard, so nothing "
            "here was observed. The layout is taken as stated."
        ),
    )


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------


def _cell(value, basis: str, unit: str = "") -> dict:
    """An island figure. ``estimate`` when there is one, ``absent`` when the
    quantity does not apply -- a labelled estimate with no number would be a
    contradiction, and the third word exists for exactly this case. Note that
    an absent CELL is not an absent FAMILY: these families are always offered,
    and the one cell that says "not a collective question" says why."""
    return {
        "value": value,
        "available": value is not None,
        "provenance": ESTIMATE if value is not None else ABSENT,
        "basis": basis,
        "unit": unit,
        "study": None,
    }


def _origin(topo: IslandTopology) -> dict:
    """Where an island family's numbers come from, said once per family."""
    if topo.measurable_here:
        detail = (
            "the fast edges were read off this rig's own pair matrix, and the "
            "collective term is then modelled from the discount ladder"
        )
    elif topo.provenance == ESTIMATE:
        detail = (
            "NOT MEASURABLE FROM HERE: this layout was described rather than "
            "observed -- it is another machine's hardware. The structure is "
            "taken as stated and the collective term is modelled from the "
            "interconnect discount ladder in planner/roofline.py"
        )
    else:
        detail = (
            "NOT MEASURABLE ON THIS RIG: there is no NVLink here and no "
            "GPUDirect peer access, so no island boundary exists to observe. "
            "The figures are modelled from the pair-matrix structure using "
            "the interconnect discount ladder in planner/roofline.py"
        )
    return {
        "provenance": ESTIMATE,
        "source": "planner/roofline.py interconnect discount ladder",
        "detail": detail,
        "caveat": _LADDER_CAVEAT,
        "rule": (
            "The local rig is a lower bound, never a verdict about a general "
            "feature. Hardware we cannot measure is modelled and labelled, "
            "not refused."
        ),
    }


def _local_family(topo: IslandTopology, model_fits_in_island: Optional[bool]) -> dict:
    """One group per island. The layout the boundary suggests on its own."""
    sizes = [len(i) for i in topo.islands]
    inside = collective_discount(topo.intra_tier, max(sizes))
    across = collective_discount(topo.inter_tier, topo.card_count)
    lanes = lanes_mod.lanes_from_card_groups(
        topo.islands,
        labels=[f"island {i}" for i in range(topo.island_count)],
        roles=["main"] * topo.island_count,
    )
    return {
        "key": "island_local_tp",
        "label": "One group per island",
        "purpose": (
            "Every tensor-parallel group stays inside one island, so no "
            "collective ever crosses the slow edge."
        ),
        "explain": (
            "The islands become independent servers over the same weights. "
            "Nothing crosses the boundary during a forward pass at all -- the "
            "only traffic between islands is whatever the front end routes. "
            "The condition is that the model fits inside ONE island: if it "
            "does not, this family does not exist and the next one is the "
            "question."
        ),
        "lanes": lanes.to_json(),
        "collective_advantage": _cell(
            (inside / across) if across else None,
            f"the collective knock-down inside an island of {max(sizes)} "
            f"card(s) over {topo.intra_tier} (x{inside:.2f}) against the "
            f"knock-down of one group spanning all {topo.card_count} cards "
            f"over {topo.inter_tier} (x{across:.2f}). A ratio of the two "
            "rungs, which is what the ladder can honestly support -- not a "
            "predicted rate",
            "x on the collective term",
        ),
        "requires": (
            "the model must fit inside one island's VRAM"
            + (
                ""
                if model_fits_in_island is None
                else (
                    " -- which it does with this card set"
                    if model_fits_in_island
                    else " -- which it does NOT with this card set, so this "
                    "family is out"
                )
            )
        ),
        "feasible": model_fits_in_island is not False,
        "origin": _origin(topo),
    }


def _split_family(topo: IslandTopology) -> dict:
    """One group across the boundary, keyed so the slow edge carries least."""
    across_all = collective_discount(topo.inter_tier, topo.card_count)
    biggest = max(len(i) for i in topo.islands)
    naive = across_all
    # An island-aware split still pays the slow edge, but for a group whose
    # cross-island cardinality is the ISLAND COUNT rather than the card count.
    aware = collective_discount(topo.inter_tier, topo.island_count)
    lanes = lanes_mod.single_lane(
        [c for isl in topo.islands for c in isl],
        label="one group over every island",
    )
    return {
        "key": "island_split_tp",
        "label": "One group across the islands, split island-aware",
        "purpose": (
            "A single tensor-parallel group spanning the boundary, with the "
            "split chosen so the slow edge carries as little as possible."
        ),
        "explain": (
            "Where the model does not fit inside one island there is no "
            "choice but to cross, and then the question is how much crosses. "
            "The ladder charges a collective by how many cards it spans, so "
            "an arrangement in which the cross-island traffic behaves like "
            f"{topo.island_count} participants rather than {topo.card_count} "
            "is the shape to aim for -- the fast edges inside each island "
            "carry the rest. On this fork the mechanism that expresses it is "
            "the uneven split key; what is missing is a measurement of any "
            "rig where the boundary exists."
        ),
        "lanes": lanes.to_json(),
        "collective_advantage": _cell(
            (aware / naive) if naive else None,
            f"an island-aware arrangement charged as {topo.island_count} "
            f"participants (x{aware:.2f}) against a flat one charged as "
            f"{topo.card_count} (x{naive:.2f}), both on the {topo.inter_tier} "
            "rung. Both terms come off the same ladder, so the ratio is "
            "meaningful where the absolute values are not",
            "x on the collective term",
        ),
        "requires": (
            f"the largest island holds {biggest} card(s); the split key has "
            "to follow the island sizes rather than the card count"
        ),
        "feasible": True,
        "origin": _origin(topo),
    }


def _lane_family(topo: IslandTopology) -> dict:
    """Islands as lanes: one island foreground, the rest scavenging."""
    if topo.island_count < 2:
        return {}
    head, rest = topo.islands[0], topo.islands[1:]
    lanes = lanes_mod.LaneSet(
        (
            lanes_mod.Lane(
                lane_id="lane0",
                label="prefill lane (island 0)",
                cards=head,
                goal="max_prefill",
                priority_class=lanes_mod.FOREGROUND,
                role="pd",
                note="reserved budget; never lends without a recall guarantee",
            ),
            lanes_mod.Lane(
                lane_id="lane1",
                label="serving group (remaining islands)",
                cards=tuple(c for isl in rest for c in isl),
                goal="max_decode",
                priority_class=lanes_mod.SCAVENGER,
                role="main",
                note=(
                    "work-conserving follower: yields the shared cards at "
                    "chunk and decode-step boundaries when the foreground "
                    "lane has work"
                ),
            ),
        )
    )
    return {
        "key": "island_lanes",
        "label": "Islands as lanes (prefill lane on its own island)",
        "purpose": (
            "One island carries prefill, the rest carry decode, and the "
            "boundary that would have cost a collective becomes the handover "
            "instead."
        ),
        "explain": (
            "A slow edge between islands is an argument FOR splitting the "
            "work rather than the tensors: a KV handover crosses the "
            "boundary once per request, where a tensor-parallel collective "
            "crosses it every layer. This is the same trade the prefill "
            "satellite makes over a network, one level down. It inherits the "
            "satellite's arithmetic and its condition -- the prefill island "
            "has to be fast enough that its own compute beats what the "
            "serving islands would have spent, under load."
        ),
        "lanes": lanes.to_json(),
        "collective_advantage": _cell(
            None,
            "not a collective question: this family removes the cross-island "
            "collective instead of discounting it, and replaces it with a "
            "handover whose cost is the satellite arithmetic on the "
            "cross-island link. The link rate is the term that decides it, "
            "and no island link exists on this rig to measure",
        ),
        "requires": (
            "the multi-group runtime, and a handover across the boundary. "
            "Slice A of that runtime is merged; the second lane is slice B"
        ),
        "feasible": True,
        "origin": _origin(topo),
    }


def island_families(
    topo: IslandTopology, *, model_fits_in_island: Optional[bool] = None
) -> dict:
    """The island families for this topology, or the reason there are none."""
    if not topo.has_islands:
        return {
            "applies": False,
            "topology": topo.to_json(),
            "reason": (
                "This card set has no island structure: "
                + (
                    "there is only one island, so every collective is already "
                    "inside it"
                    if topo.island_count <= 1
                    else "every card is its own island, so no boundary "
                    "separates a fast group from a slow one. On this rig that "
                    "is the measured state -- no pair has peer access"
                )
            ),
            "families": [],
            "how_to_explore": (
                "Describe hardware with islands (two NVLink-bridged pairs, an "
                "NVSwitch box) and the families appear as estimates. They "
                "cannot become measurements here, and they say so."
            ),
        }
    fams = [
        _local_family(topo, model_fits_in_island),
        _split_family(topo),
    ]
    lane_fam = _lane_family(topo)
    if lane_fam:
        fams.append(lane_fam)
    return {
        "applies": True,
        "topology": topo.to_json(),
        "reason": "",
        "families": fams,
        "coverage": {
            "measured": 0,
            "estimate": sum(
                1 for f in fams if (f.get("collective_advantage") or {}).get("available")
            ),
            "absent": sum(
                1
                for f in fams
                if not (f.get("collective_advantage") or {}).get("available")
            ),
            "total": len(fams),
            "summary": (
                "Every figure on these families is an estimate off the "
                "discount ladder. None of them can become a measurement on "
                "this rig, and none of them is withheld for that reason."
            ),
        },
    }
