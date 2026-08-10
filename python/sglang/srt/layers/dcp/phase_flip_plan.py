# SPDX-License-Identifier: Apache-2.0
"""Pure planning arithmetic for the #631 PP<->TP phase flip.

The #297 reshard moves KV between two WEIGHTED TOKEN vectors on a fixed
layer set. The phase flip moves KV between two LAYOUTS of the same three
ranks:

* PP layout (prefill): full-attention layer ordinal ``f`` belongs to
  exactly one stage ``s`` (the replicated ``layer_map``); that stage holds
  ALL tokens of its layers, and the stage pool row of global slot ``L`` is
  ``L`` itself (dcp_size=1 -- the compact rule degenerates to identity).
* TP layout (decode): every rank holds ALL layer ordinals, token-sharded
  by the weighted DCP owner rule of ``owner.py``; the pool row is the
  compact row of ``reshard_plan.rows_of``.

Both layouts store rows in the SAME byte format (the fork's weighted DCP
replicates KV heads), so a flip is a pure row redistribution.

The moved cell set of the (stage ``s``, dcp rank ``r``) pair is
``layers(s) x slots_owned_by_r`` -- INDEPENDENT of the flip direction.
Direction only decides which end sends. Enumeration order, the shared
payload-layout convention on both ends (a checksum keeps it falsifiable
at runtime, as in #297): layer ordinals ascending, slots ascending within
a layer, K bytes then V.

Everything here is a pure function of replicated inputs (the live slot
set, the layer map, the token vector): every rank computes the SAME
transition, so pairwise transfer lists agree by construction and the
consensus round never carries plan metadata. Token ownership is
layer-independent, which is why one row list per pair, reused for every
layer of the pair, is enough.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch

from sglang.srt.layers.dcp.reshard_plan import KvReshardError, owner_of, rows_of

PP_TO_TP = "pp_to_tp"
TP_TO_PP = "tp_to_pp"
_DIRECTIONS = (PP_TO_TP, TP_TO_PP)


def validate_layer_map(
    layer_map: Sequence[Sequence[int]], n_layers: int
) -> Tuple[Tuple[int, ...], ...]:
    """Check that ``layer_map`` partitions ``range(n_layers)`` exactly.

    ``layer_map[s]`` lists the full-attention pool ordinals stage ``s``
    owns. Every ordinal must appear exactly once across all stages (a
    stage MAY be empty -- a degenerate split is legal, a hole or a
    duplicate is not). Returns the normalized tuple form.
    """
    # Ascending order per stage is part of the payload-layout convention
    # (module docstring); normalize here so callers cannot diverge on it.
    norm = tuple(tuple(sorted(int(x) for x in stage)) for stage in layer_map)
    seen: Dict[int, int] = {}
    for s, layers in enumerate(norm):
        for f in layers:
            if not (0 <= f < n_layers):
                raise KvReshardError(
                    f"layer map stage {s} names ordinal {f}, outside "
                    f"[0, {n_layers})"
                )
            if f in seen:
                raise KvReshardError(
                    f"layer map assigns ordinal {f} to both stage "
                    f"{seen[f]} and stage {s}; the PP layout owns every "
                    f"layer exactly once"
                )
            seen[f] = s
    missing = [f for f in range(n_layers) if f not in seen]
    if missing:
        raise KvReshardError(
            f"layer map covers no stage for ordinal(s) {missing}; a hole "
            f"in the partition would silently drop those layers' KV at "
            f"the flip"
        )
    return norm


def default_wave_count(layer_map: Sequence[Sequence[int]]) -> int:
    """The most waves the seam can be split into with every rank paying in.

    A wave is the unit at which the flip releases the source layout's
    backing and restores the destination's, so the two must NET OUT inside
    each wave -- see :func:`layer_waves`. That holds only while every rank
    contributes at least one of its OWN layers per wave, so the ceiling is
    the SMALLEST stage's layer count (16 on this rig's [28, 20, 16] split).
    """
    sizes = [len(stage) for stage in layer_map if len(stage)]
    return max(1, min(sizes)) if sizes else 1


def layer_waves(
    layer_map: Sequence[Sequence[int]], n_waves: int
) -> Tuple[Tuple[int, ...], ...]:
    """Split the global layer ordinals into ``n_waves`` balanced waves.

    THE BALANCE IS THE WHOLE POINT, and it is a memory argument rather
    than a load-balancing one. The flip's seam swaps physical backing
    between the two layouts: the source layout's pages for a layer may be
    released once that layer has been read, and the destination layout's
    pages for a layer must be committed before it is written. Doing that
    ONCE for the whole pool forces every byte that crosses the seam to be
    staged at the same instant, which is the unbounded term behind the
    one-request livelock (#631, HANDOFF_666): staging grew with the
    resident live set until no request past a certain length could ever
    flip, and under strict purity that wedges rather than degrades.

    Splitting the seam into waves bounds the staged bytes to ONE wave's
    share -- but only if a wave's releases pay for its commits. Rank ``r``
    releases the wave's layers that IT owns in the PP layout (``L_r/W`` of
    them, each spanning the full PP pool) and commits the wave's layers in
    the TP layout (``64/W`` of them, each spanning its ``w_r`` share of the
    rows). Those are equal exactly when each rank contributes a
    PROPORTIONAL slice of its own block to every wave, which is what this
    split produces: each stage's ascending ordinals are cut into ``n_waves``
    contiguous groups and group ``j`` of every stage forms wave ``j``.

    Integer floors leave at most one layer of drift between the two sides
    at a wave boundary, so the peak carries ONE layer-span of slack. That
    term is priced explicitly in ``_staging_bytes`` -- it is a constant of
    the pool geometry and does not grow with the request.
    """
    n_waves = max(1, int(n_waves))
    waves: list[list[int]] = [[] for _ in range(n_waves)]
    for stage_layers in layer_map:
        stage = list(stage_layers)
        total = len(stage)
        for j in range(n_waves):
            lo = (j * total) // n_waves
            hi = ((j + 1) * total) // n_waves
            waves[j].extend(stage[lo:hi])
    return tuple(tuple(sorted(w)) for w in waves)


@dataclass
class PhaseFlipTransition:
    """One rank's share of a PP<->TP layout flip.

    Row tensors are int64 on CPU and refer to rows in THIS rank's pool of
    the respective side: PP-side rows are global slot ids, TP-side rows
    are weighted compact rows. ``send_rows[p]`` / ``recv_rows[p]`` are
    reused for every layer in ``send_layers[p]`` / ``recv_layers[p]``
    (token ownership is layer-independent). Layer tuples are ascending;
    slot enumeration within a pair is ascending on both ends.
    """

    rank: int
    direction: str
    n_layers: int
    layer_map: Tuple[Tuple[int, ...], ...]
    tp_vector: Tuple[int, ...]
    #: my stage's layer ordinals (both directions: the local move covers
    #: my layers x my dcp-owned slots).
    local_layers: Tuple[int, ...]
    #: PP-pool rows (= slot ids) of my dcp-owned slots.
    local_pp_rows: torch.Tensor
    #: TP-pool compact rows of the same slots, same order.
    local_tp_rows: torch.Tensor
    #: peer -> layer ordinals covered by the payload I send to that peer.
    send_layers: Dict[int, Tuple[int, ...]]
    #: peer -> rows in MY (sending-side) pool, one list per pair.
    send_rows: Dict[int, torch.Tensor]
    #: peer -> layer ordinals covered by the payload that peer sends me.
    recv_layers: Dict[int, Tuple[int, ...]]
    #: peer -> rows in MY (receiving-side) pool the payload fills.
    recv_rows: Dict[int, torch.Tensor]
    total_slots: int

    @property
    def outgoing_cells(self) -> int:
        return sum(
            len(self.send_layers[p]) * int(r.numel())
            for p, r in self.send_rows.items()
        )

    @property
    def incoming_cells(self) -> int:
        return sum(
            len(self.recv_layers[p]) * int(r.numel())
            for p, r in self.recv_rows.items()
        )

    def max_pp_row(self) -> int:
        """Largest PP-pool row this rank touches (bounds-check input)."""
        mx = -1
        if self.local_pp_rows.numel():
            mx = max(mx, int(self.local_pp_rows.max().item()))
        rows = self.send_rows if self.direction == PP_TO_TP else self.recv_rows
        for r in rows.values():
            if r.numel():
                mx = max(mx, int(r.max().item()))
        return mx

    def max_tp_row(self) -> int:
        """Largest TP-pool row this rank touches (bounds-check input)."""
        mx = -1
        if self.local_tp_rows.numel():
            mx = max(mx, int(self.local_tp_rows.max().item()))
        rows = self.recv_rows if self.direction == PP_TO_TP else self.send_rows
        for r in rows.values():
            if r.numel():
                mx = max(mx, int(r.max().item()))
        return mx


def build_phase_flip_transition(
    slots: torch.Tensor,
    layer_map: Sequence[Sequence[int]],
    n_layers: int,
    tp_vector: Sequence[int],
    rank: int,
    direction: str,
) -> PhaseFlipTransition:
    """Derive this rank's flip transition from the replicated inputs.

    ``slots``: int64, sorted ascending, unique (checked -- the pairwise
    payload layout is derived from this order on both ends, as in #297).
    ``layer_map``: stage -> full-attention pool ordinals, a partition of
    ``range(n_layers)``. ``tp_vector``: the weighted DCP token vector of
    the TP layout. ``direction``: :data:`PP_TO_TP` or :data:`TP_TO_PP`.
    """
    if direction not in _DIRECTIONS:
        raise KvReshardError(
            f"unknown flip direction {direction!r}; expected one of "
            f"{_DIRECTIONS}"
        )
    vec = tuple(int(x) for x in tp_vector)
    norm_map = validate_layer_map(layer_map, n_layers)
    if len(norm_map) != len(vec):
        raise KvReshardError(
            f"layer map has {len(norm_map)} stages but the TP vector "
            f"{vec} has {len(vec)} ranks; the flip reuses the SAME "
            f"three ranks, so the counts must match"
        )
    n_ranks = len(vec)
    if not (0 <= rank < n_ranks):
        raise KvReshardError(f"rank {rank} out of range for {n_ranks} ranks")
    slots = slots.detach().to("cpu", torch.int64)
    if slots.numel():
        if not bool((slots[1:] > slots[:-1]).all()):
            raise KvReshardError(
                "live slot ids must be sorted ascending and unique -- the "
                "pairwise payload layout is derived from this order on "
                "both ends"
            )
        if int(slots[0].item()) < 0:
            raise KvReshardError("negative KV slot id in live set")

    owner = owner_of(slots, vec)
    my_slots = slots[owner == rank]
    my_layers = norm_map[rank]
    local_pp = my_slots.clone()  # PP row of slot L is L itself (dcp_size=1)
    local_tp = rows_of(my_slots, vec, rank)

    send_layers: Dict[int, Tuple[int, ...]] = {}
    send_rows: Dict[int, torch.Tensor] = {}
    recv_layers: Dict[int, Tuple[int, ...]] = {}
    recv_rows: Dict[int, torch.Tensor] = {}
    for peer in range(n_ranks):
        if peer == rank:
            continue
        peer_slots = slots[owner == peer]
        peer_layers = norm_map[peer]
        if direction == PP_TO_TP:
            # I am stage `rank` (sender side) and dcp rank `rank`
            # (receiver side): send my layers x peer's slots; receive
            # peer's layers x my slots into my compact rows.
            if my_layers and peer_slots.numel():
                send_layers[peer] = my_layers
                send_rows[peer] = peer_slots.clone()
            if peer_layers and my_slots.numel():
                recv_layers[peer] = peer_layers
                recv_rows[peer] = local_tp.clone()
        else:
            # TP_TO_PP: send peer's layers x my slots out of my compact
            # rows; receive my layers x peer's slots at their slot ids.
            if peer_layers and my_slots.numel():
                send_layers[peer] = peer_layers
                send_rows[peer] = local_tp.clone()
            if my_layers and peer_slots.numel():
                recv_layers[peer] = my_layers
                recv_rows[peer] = peer_slots.clone()

    return PhaseFlipTransition(
        rank=rank,
        direction=direction,
        n_layers=n_layers,
        layer_map=norm_map,
        tp_vector=vec,
        local_layers=my_layers,
        local_pp_rows=local_pp,
        local_tp_rows=local_tp,
        send_layers=send_layers,
        send_rows=send_rows,
        recv_layers=recv_layers,
        recv_rows=recv_rows,
        total_slots=int(slots.numel()),
    )
