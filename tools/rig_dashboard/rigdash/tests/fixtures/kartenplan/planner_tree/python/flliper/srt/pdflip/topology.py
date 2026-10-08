"""HW-GENERIC 1002 Stage 2 (enabling): the pdflip group topology as a function
of the card count, instead of literals in the argv builders.

Release topology (proven on metal for N = 3 only): group P = TP1 x PP<N>
over every card, group D = TP<N> x PP1 over every card, one rank per card,
rank i on the i-th card of ``order_cards`` (biggest first); Form A (the D
attention host on ordinal 0, experts on the workers) comes from the profile
argv (``--rank-role host,worker,...``), not from here.

What this module decides today: the argv sizes and the rank map the launcher
prints -- byte-identical for N = 3 ("--pp-size 3", "--tp-size 3",
"--rank-gpu-id 0,1,2"). What it does NOT decide yet (Stage 2 proper, see
/spinning/gpu-arb/docs/HW-GENERISCH-SM86-SM120-1002.md): whether N != 3 is
servable at all -- the 3-shaped data paths (BAR1 group windows "PP_0=96",
the exchange lanes, the platztausch card, the census, the per-rank records)
are listed in :data:`N_NOT_3_BLOCKERS`, and :func:`plan_topology` names them
for an N outside :data:`PROVEN_CARD_COUNTS` instead of pretending.

PURE: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

#: Card counts a release boot has been proven on (metal).
PROVEN_CARD_COUNTS: Tuple[int, ...] = (3,)
#: barlink_bar1_ext.py BARLINK_BAR1_MAX_RANKS (#define 8): the BAR1
#: collectives cannot address more ranks than this.
MAX_CARDS_BAR1 = 8
#: A flip needs two groups that share the cards; one card cannot hold a PP
#: or TP group of more than one rank.
MIN_CARDS = 2

#: What still assumes three ranks (HW-KOPPLUNG-AUDIT-1002.md 2/4c, desk
#: audit 02.10.); each must become N-aware before an N != 3 boot.
N_NOT_3_BLOCKERS: Tuple[str, ...] = (
    "BAR1 group windows sized for 3 ranks with PP0 on a big BAR (P_WINDOWS 'PP_0=96', D 16+32+24)",
    "exchange lanes / platztausch card / census laid out for 3 ranks",
    "per-rank measured records (D_FIXED_MIB, P_OVERSHOOT_MIB, ...) are 3-vectors of one inventory",
    "PP-cut calibration (MEASURED_MS_PER_LAYER, P_CHUNK_BUILTIN, stage model) fitted on 3 stages",
    "Form A host/worker split chosen by the profile, not by a layout objective",
    "d_reshard speed model calibrated for TP3 = 5090 + 2x3080 (D_RANK_CARD_CLASSES)",
)


class TopologyRefused(RuntimeError):
    """No pdflip topology for this card count (named)."""


@dataclass(frozen=True)
class Topology:
    n_cards: int
    p_tp: int
    p_pp: int
    d_tp: int
    d_pp: int
    #: Form A attention host ordinal (the biggest card, ordinal 0 of order_cards)
    host_ordinal: int
    proven: bool
    blockers: Tuple[str, ...]

    @property
    def rank_gpu_id(self) -> Tuple[int, ...]:
        return tuple(range(self.n_cards))


def release_topology(n_cards: int) -> Topology:
    """The release layout for ``n_cards`` cards (P = PP<N>, D = TP<N>, one
    rank per card, host on ordinal 0) with its proof status. Raises
    :class:`TopologyRefused` outside [:data:`MIN_CARDS`, :data:`MAX_CARDS_BAR1`]."""
    n = int(n_cards)
    if n < MIN_CARDS or n > MAX_CARDS_BAR1:
        raise TopologyRefused(
            f"HW-TOPOLOGY: {n} card(s): the pdflip flip needs {MIN_CARDS}..{MAX_CARDS_BAR1} cards "
            f"(two groups over the same cards; BAR1 collectives address at most {MAX_CARDS_BAR1} ranks)")
    proven = n in PROVEN_CARD_COUNTS
    return Topology(n_cards=n, p_tp=1, p_pp=n, d_tp=n, d_pp=1, host_ordinal=0,
                    proven=proven, blockers=() if proven else N_NOT_3_BLOCKERS)


def plan_topology(n_cards: int) -> Topology:
    """:func:`release_topology`, refusing BY NAME an unproven card count
    (with the list of what still assumes three ranks)."""
    t = release_topology(n_cards)
    if not t.proven:
        raise TopologyRefused(
            f"HW-TOPOLOGY: {t.n_cards} cards would be P = PP{t.p_pp}, D = TP{t.d_tp} (host ordinal "
            f"{t.host_ordinal}); proven on metal only for N in {list(PROVEN_CARD_COUNTS)}. Still "
            "3-shaped: " + "; ".join(t.blockers))
    return t


def rank_gpu_id_csv(n_cards: int) -> str:
    """``--rank-gpu-id`` of the release topology ("0,1,2" for N = 3)."""
    return ",".join(str(i) for i in release_topology(n_cards).rank_gpu_id)
