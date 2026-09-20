# SPDX-License-Identifier: Apache-2.0
"""Slice 4 -- seam A as a PLAN: which KV pages move, from where, and how.

``flip_nextflash_plan.solve_kv_relay`` answers the FEASIBILITY question: can
the Form-A host pool hold 262144 tokens at all (yes, by 1.93x). This module
answers the one after it: for each of the 12 full-attention layers, WHERE do
its pages live now and what has to happen to them -- and the answer is not
one thing but three, which is the whole point.

THE ASYMMETRY THAT MAKES THIS CHEAP. PP stage 0 sits on the 5090
(``BUDGET-REACH`` card free 9728 of 32604, fn7s) and the Form-A attention
host is THE SAME CARD. Of the 12 full-attention layers PP holds [7, 3, 2]
(fn7s line 21), so SEVEN are already on the destination card. Only five cross
a link:

    PP1 -> host   3 layers x 1088 B/token x 262144 = 0.856 GB  over x4  ~0.13 s
    PP2 -> host   2 layers x 1088 B/token x 262144 = 0.570 GB  over x8  ~0.04 s
    PP0 -> host   0 GB over any link -- device-local

1.43 GB, not the full 3.42 GB, and the two legs run on disjoint lanes. This
is Memory ``FLIP-KARTE-ZUERST`` applied: the seam's cost is ~0.15 s and seam A
is explicitly NOT the flip's cost driver (design §3: the flip is
rebuild-bound, not transport-bound).

AND THE ONE TRAP IS DEVICE-LOCAL, NOT ON THE LINK. PP0's 7-layer pool and the
Form-A 12-layer pool are DIFFERENT ALLOCATIONS. A naive copy needs both
resident at once -- transient ``2 x 1.99 GB`` on a 5090 that has
``card free 0.24 GiB`` under extend (fnFA19:2195). So the seven local layers
must be REMAPPED, not copied: Memory ``S6-REMAP-STATT-ALLOKATION``, the VMM
remap changes the OWNER of the pages instead of duplicating them. A plan that
says "copy" for a local layer is an OOM with a schedule attached, which is
why :func:`plan_kv_remap` assigns the disposition rather than letting a
caller pass one in.

THE CELL IS PER FULL-ATTENTION LAYER AND IT IS THE SAME ON BOTH SIDES.
PP3's three cells sum to ``7616 + 3264 + 2176 = 13056`` and that is exactly
``12 x 1088``, so the split [7, 3, 2] is lossless and one layer costs 1088
B/token wherever it sits. The Form-A cell is ``14143 = 13056 + 1087``; the
1087 is the solo draft's KV term plus the undivided mamba amortisation --
terms that belong to the DESTINATION and have no source-side pages. Never
compare 13056 with 14143 and call the difference a rounding error; this
module keeps the two apart by construction (``moved_bytes`` is computed from
the per-layer cell, never from a pool cell).

Pure: no torch, no NVML, no device. Every default carries its boot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.flip_nextflash_plan import (
    FORM_A_KV,
    PP3_KV,
    KvLayout,
    Weg2FlipKvRelayInfeasible,
    solve_kv_relay,
)

__all__ = [
    "FULL_ATTENTION_LAYERS",
    "PP3_BYTES_PER_TOKEN_PER_LAYER",
    "LINK_GB_S",
    "LayerMove",
    "KvRemapPlan",
    "plan_kv_remap",
    "DISPOSITION_REMAP",
    "DISPOSITION_LEG",
]

#: Qwen3.8 Next Flash: 12 full-attention layers among the hybrid stack.
#: fn7s line 21, "full-attention per stage: [7, 3, 2] of 12 total".
FULL_ATTENTION_LAYERS = 12

#: 13056 / 12 -- the per-token cell of ONE full-attention layer, identical on
#: both sides because the PP split is lossless.
PP3_BYTES_PER_TOKEN_PER_LAYER = 1088

#: Memory ``RANG-LINK-ZUORDNUNG``: rank 1 is x4 (6.5 GB/s), ranks 0 and 2 are
#: x8 (14.4 / 13.3). Never hand-numbers -- these are the measured link rates.
LINK_GB_S: Dict[int, float] = {0: 14.4, 1: 6.5, 2: 13.3}

#: A layer already on the destination card: its pages change OWNER.
DISPOSITION_REMAP = "remap"
#: A layer on another card: its pages travel a link.
DISPOSITION_LEG = "leg"


@dataclass(frozen=True)
class LayerMove:
    """What happens to the KV pages of ONE full-attention layer."""

    layer_index: int
    src_rank: int
    dst_rank: int
    bytes_moved: int
    disposition: str
    link_gb_s: Optional[float]

    @property
    def seconds(self) -> float:
        """A remap costs no transport time; a leg costs bytes / link rate."""
        if self.disposition == DISPOSITION_REMAP or not self.link_gb_s:
            return 0.0
        return self.bytes_moved / (self.link_gb_s * 1e9)


@dataclass(frozen=True)
class KvRemapPlan:
    """The whole of seam A: per layer, per rank, per lane."""

    context_tokens: int
    moves: Tuple[LayerMove, ...]
    dst_rank: int
    remapped_bytes: int
    legged_bytes: int
    #: Lanes run in PARALLEL, so the seam's wall time is the SLOWEST lane, not
    #: the sum. Summing them would overstate seam A by ~2x and point the flip's
    #: optimisation at the transport, which §3 says is not where the time is.
    seam_seconds: float
    per_lane_seconds: Dict[int, float]

    @property
    def moved_bytes(self) -> int:
        return self.remapped_bytes + self.legged_bytes

    def report(self) -> str:
        _GB = 1e9
        lines = [
            f"KV REMAP seam A @ {self.context_tokens} tokens -> rank {self.dst_rank}",
            f"  remapped (device-local, no link) "
            f"{self.remapped_bytes / _GB:.3f} GB in "
            f"{sum(1 for m in self.moves if m.disposition == DISPOSITION_REMAP)} layer(s)",
            f"  legged   (over a link)          "
            f"{self.legged_bytes / _GB:.3f} GB in "
            f"{sum(1 for m in self.moves if m.disposition == DISPOSITION_LEG)} layer(s)",
            f"  lanes run in PARALLEL -> seam {self.seam_seconds * 1000:.0f} ms "
            f"(slowest lane, not the sum)",
        ]
        for rank, secs in sorted(self.per_lane_seconds.items()):
            lines.append(f"    rank {rank} lane: {secs * 1000:.0f} ms")
        return "\n".join(lines)


def plan_kv_remap(
    src: KvLayout = PP3_KV,
    dst: KvLayout = FORM_A_KV,
    context_tokens: int = 262144,
    bytes_per_token_per_layer: int = PP3_BYTES_PER_TOKEN_PER_LAYER,
    link_gb_s: Optional[Dict[int, float]] = None,
    total_full_attention_layers: int = FULL_ATTENTION_LAYERS,
) -> KvRemapPlan:
    """Layer -> token re-lay: which pages from which card, and how each moves.

    Runs :func:`solve_kv_relay` FIRST, so a destination that cannot hold the
    context is W113 before any page is assigned a disposition -- scheduling
    a move into a pool that does not fit is a plan for an OOM.

    The disposition is ASSIGNED, never accepted from a caller: a layer whose
    source rank IS the destination rank is a remap (Memory
    ``S6-REMAP-STATT-ALLOKATION``), everything else is a leg. That is the
    whole rule, and it is the rule because the 5090 has 0.24 GiB free under
    extend and a copy of PP0's 7 layers would need 1.99 GB of it.
    """
    # Feasibility first. W113 out of solve_kv_relay if the destination binds
    # below the demanded context.
    solve_kv_relay(src, dst, context_tokens)

    if bytes_per_token_per_layer <= 0:
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- a per-layer cell of "
            f"{bytes_per_token_per_layer} B/token is not a cell"
        )

    src_layers = tuple(int(n) for n in src.full_attn_per_rank)
    if sum(src_layers) != total_full_attention_layers:
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- the source layout {src.name!r} "
            f"splits {sum(src_layers)} full-attention layer(s) {list(src_layers)} "
            f"but the model has {total_full_attention_layers}. A re-lay that "
            f"does not account for every layer leaves the destination with a "
            f"hole it will read as zeros on the first decode after the flip -- "
            f"silently, because an unwritten KV page is a valid page."
        )

    dst_layers = tuple(int(n) for n in dst.full_attn_per_rank)
    carriers = [i for i, n in enumerate(dst_layers) if n > 0]
    if len(carriers) != 1:
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- the destination {dst.name!r} "
            f"carries full attention on {len(carriers)} rank(s) {carriers}. "
            f"Form A has exactly ONE attention host; this plan re-lays INTO a "
            f"single pool and has no rule for splitting across two."
        )
    dst_rank = carriers[0]
    if dst_layers[dst_rank] != total_full_attention_layers:
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- destination rank {dst_rank} "
            f"holds {dst_layers[dst_rank]} of {total_full_attention_layers} "
            f"full-attention layers. Under Form A the host holds ALL of them; "
            f"a destination that holds fewer is a different layout."
        )

    rates = dict(LINK_GB_S if link_gb_s is None else link_gb_s)
    per_layer_bytes = bytes_per_token_per_layer * context_tokens

    moves: List[LayerMove] = []
    layer_index = 0
    for src_rank, count in enumerate(src_layers):
        for _ in range(count):
            local = src_rank == dst_rank
            moves.append(
                LayerMove(
                    layer_index=layer_index,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    bytes_moved=per_layer_bytes,
                    disposition=DISPOSITION_REMAP if local else DISPOSITION_LEG,
                    link_gb_s=None if local else rates.get(src_rank),
                )
            )
            layer_index += 1

    remapped = sum(m.bytes_moved for m in moves if m.disposition == DISPOSITION_REMAP)
    legged = sum(m.bytes_moved for m in moves if m.disposition == DISPOSITION_LEG)

    per_lane: Dict[int, float] = {}
    for m in moves:
        if m.disposition != DISPOSITION_LEG:
            continue
        per_lane[m.src_rank] = per_lane.get(m.src_rank, 0.0) + m.seconds

    unrated = sorted(
        {m.src_rank for m in moves if m.disposition == DISPOSITION_LEG and not m.link_gb_s}
    )
    if unrated:
        raise Weg2FlipKvRelayInfeasible(
            f"W113 Weg2FlipKvRelayInfeasible -- rank(s) {unrated} must send KV "
            f"over a link but have no measured link rate. Memory "
            f"RANG-LINK-ZUORDNUNG: rank 1 is x4 (6.5 GB/s), ranks 0 and 2 are "
            f"x8 (14.4 / 13.3) -- never hand-numbers. An unrated lane would "
            f"make the seam look free."
        )

    return KvRemapPlan(
        context_tokens=context_tokens,
        moves=tuple(moves),
        dst_rank=dst_rank,
        remapped_bytes=remapped,
        legged_bytes=legged,
        seam_seconds=max(per_lane.values()) if per_lane else 0.0,
        per_lane_seconds=per_lane,
    )
