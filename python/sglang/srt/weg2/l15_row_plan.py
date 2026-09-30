# SPDX-License-Identifier: Apache-2.0
"""L1.5 KV row plan, both directions (L15-08, pure).

Derives WHICH KV rows move between the D (TP/DCP) layout and the P (PP)
layout for the L1.5 line of ``L15-PLAN-0930``:

* :func:`plan_d_to_p` -- the hot D->P handover (plan 2.0): at a D->P flip,
  the prefix a follow-up prefill needs is hot in D's pool; every (layer,
  token) cell moves from the D rank that owns the token to the P stage that
  owns the layer, so P adopts the prefix as device nodes without a store
  read.
* :func:`plan_p_to_d` -- the phase-2 deposit (plan 2.5): a running P chunk
  writes its finished tokens into D's reserved deposit slots; the owner of
  every deposit slot is known up-front by the owner rule, "P guesses
  nothing".

Ownership is ``layers/dcp/owner.py``'s weighted rule, used, not restated:
rank ``r`` owns global slot ``L`` iff ``prefix[r] <= (L % S) < prefix[r+1]``
(``prefix`` = cumulative owner vector, ``S = prefix[-1]``), and its compact
pool row is ``(L // S) * ratio + (L % S - lo)`` -- computed with
``dcp_weighted_write_slots`` so the plan can never drift from the write
path (pinned by ``test_weg2_l15_row_plan_0930``).

Row semantics follow ``managers/phase_flip_runtime.py``'s payload
convention: one row list per (src, dst) pair, REUSED for every layer the
piece names, so ``RowPiece.src_rows`` / ``RowPiece.dst_rows`` are rows
inside the per-layer buffers and ``nbytes = len(tokens) * len(layers) *
row_bytes_per_layer``.

Routing (plan 2.0/2.5): a block whose D rank and P stage share a physical
card is a local D2D copy (``route="local"``); otherwise it rides the
existing flip lanes (``route="lane"``, ``weg2/bar1_lanes.py``, the pusher
pushes). In the deposit direction the 5090's rank (rank 0, "TP0") is never
a destination -- plan 2.1: "TP0 fills its rows of the slots of H and E in
bulk from L2"; with ``tp0_skip=True`` its blocks are still enumerated but
flagged ``route="skip"`` so the accounting sees what was NOT moved. A
zero-share rank (the NF form, token vector with a 0 entry) simply owns no
slot and produces no pieces at all.

Piece sizing mirrors ``weight_exchange_transport.batch_descs``: pieces are
cut so ``nbytes <= slot_bytes`` (the lane ring slot), splitting the
consecutive token run first and -- only when even one token's full-layer
block exceeds a slot -- the layer list. The smallest indivisible unit is
one row (one token, one layer): a ``slot_bytes`` below ``row_bytes_per_layer``
cannot be honored, and the plan says so by returning that single-row piece
rather than inventing sub-row copies.

Geometry is an input everywhere (27B: 16 attention layers, P attention cut
10/3/3, D token vector [7,4,5] over S=16; NF: [0,9,7]); nothing here reads
a runtime global, so the plan is computable on the front and on every rank
from the same arguments, and is deterministic: the returned list is sorted
by ``(dst, first dst row, layers)``.

Pure module: torch CPU tensors only, no CUDA, no process group, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from sglang.srt.layers.dcp.owner import dcp_weighted_write_slots

__all__ = [
    "RowPiece",
    "owner_prefix",
    "owned_rows",
    "plan_d_to_p",
    "plan_p_to_d",
]

ROUTE_LOCAL = "local"
ROUTE_LANE = "lane"
ROUTE_SKIP = "skip"


@dataclass(frozen=True)
class RowPiece:
    """One transport piece of an L1.5 KV move.

    ``src`` / ``dst`` name the endpoints (``tp<r>`` = D rank r, ``pp<s>`` =
    P stage s); ``layers`` are the global attention-layer ordinals this
    piece carries; ``tokens`` are the prefix/chunk token indices; the row
    tuples are pool rows in the sender's / receiver's per-layer buffer
    (payload convention: the same row list serves every layer named).
    ``nbytes = len(tokens) * len(layers) * row_bytes_per_layer``.
    """

    src: str
    dst: str
    layers: Tuple[int, ...]
    tokens: Tuple[int, ...]
    src_rows: Tuple[int, ...]
    dst_rows: Tuple[int, ...]
    nbytes: int
    route: str


def owner_prefix(vector: Sequence[int]) -> List[int]:
    """Cumulative owner vector -> the ``prefix`` of the weighted owner rule.

    ``[7,4,5] -> [0,7,11,16]``; a zero entry keeps a zero-width range, i.e.
    a rank that owns nothing (the NF form's rank 0). This is
    ``distributed.utils.cp_token_prefix``'s arithmetic without its runtime
    lane probe, so the front can build it for a layout that is not installed.
    """
    vec = [int(v) for v in vector]
    if any(v < 0 for v in vec):
        raise ValueError(f"owner vector entries must be non-negative, got {vec}")
    out = [0]
    for v in vec:
        out.append(out[-1] + v)
    return out


def owned_rows(
    slots: Sequence[int], prefix: Sequence[int], rank: int
) -> Tuple[List[int], List[int]]:
    """Which of ``slots`` rank ``rank`` owns, and its compact rows.

    Returns ``(token_idx, compact_rows)``: ``token_idx[i]`` is the index
    INTO ``slots`` of the i-th owned slot, ``compact_rows[i]`` that slot's
    row in the rank's compact pool. The mapping is
    ``dcp_weighted_write_slots`` verbatim -- the same expression the masked
    KV write uses -- so plan rows and write rows cannot drift.
    """
    if not slots:
        return [], []
    lo, hi = int(prefix[rank]), int(prefix[rank + 1])
    loc, mask = dcp_weighted_write_slots(
        torch.tensor([int(s) for s in slots], dtype=torch.int64),
        int(prefix[-1]),
        lo,
        hi,
        hi - lo,
    )
    idx = [int(i) for i in mask.nonzero(as_tuple=True)[0]]
    return idx, [int(r) for r in loc[mask]]


def _runs(
    n_tokens: int, n_layers: int, slot_bytes: int, row_bytes_per_layer: int
):
    """Yield ``((t0, t1), (l0, l1))`` index ranges with piece bytes <= slot.

    Token runs first (the normal case: one piece per block, plan 2.0's
    "block (D rank r's owned tokens x P stage s's layers)"); the layer list
    is split only when one token's full-layer block alone exceeds a lane
    slot. One row is the smallest indivisible unit (batch_descs' rule for
    a STRIDED2D run), so below ``row_bytes_per_layer`` the byte bound is
    physically unreachable and the single row is what comes out.
    """
    block = n_layers * row_bytes_per_layer
    if block <= slot_bytes:
        tokens_per = max(1, slot_bytes // block)
        layers_per = n_layers
    else:
        tokens_per = 1
        layers_per = max(1, slot_bytes // max(1, row_bytes_per_layer))
    t0 = 0
    while t0 < n_tokens:
        t1 = min(t0 + tokens_per, n_tokens)
        l0 = 0
        while l0 < n_layers:
            l1 = min(l0 + layers_per, n_layers)
            yield (t0, t1), (l0, l1)
            l0 = l1
        t0 = t1


def _sort_key(p: RowPiece):
    return (p.dst, p.dst_rows[0] if p.dst_rows else -1, p.layers)


def plan_d_to_p(
    d_slots: Sequence[int],
    prefix: Sequence[int],
    stage_layers: Sequence[Sequence[int]],
    stage_card: Sequence[int],
    rank_card: Sequence[int],
    p_row0: int,
    row_bytes_per_layer: int,
    slot_bytes: int,
) -> List[RowPiece]:
    """Hot D->P handover rows (plan 2.0).

    Prefix token ``i`` sits at D global slot ``d_slots[i]``; the D rank
    that owns that slot holds it at its compact row. The P stage owning
    layer ``l`` needs the cell at its dense P row ``p_row0 + i`` (in the PP
    layout pool row == slot id, plan: "P stage s rows [p0, p0+n)"). One
    block per (rank, stage); blocks are cut to lane slots by :func:`_runs`.
    A zero-share rank contributes nothing (it owns no slot).
    """
    pieces: List[RowPiece] = []
    slots = [int(s) for s in d_slots]
    for rank in range(len(prefix) - 1):
        idx, compact = owned_rows(slots, prefix, rank)
        if not idx:
            continue
        for stage, layers in enumerate(stage_layers):
            route = (
                ROUTE_LOCAL if rank_card[rank] == stage_card[stage] else ROUTE_LANE
            )
            for (t0, t1), (l0, l1) in _runs(
                len(idx), len(layers), slot_bytes, row_bytes_per_layer
            ):
                toks = tuple(idx[t0:t1])
                layers_sub = tuple(layers[l0:l1])
                pieces.append(
                    RowPiece(
                        src=f"tp{rank}",
                        dst=f"pp{stage}",
                        layers=layers_sub,
                        tokens=toks,
                        src_rows=tuple(compact[t0:t1]),
                        dst_rows=tuple(p_row0 + t for t in toks),
                        nbytes=len(toks) * len(layers_sub) * row_bytes_per_layer,
                        route=route,
                    )
                )
    pieces.sort(key=_sort_key)
    return pieces


def plan_p_to_d(
    a: int,
    b: int,
    e0: int,
    prefix: Sequence[int],
    stage: int,
    stage_layers: Sequence[Sequence[int]],
    stage_card: Sequence[int],
    rank_card: Sequence[int],
    row_bytes_per_layer: int,
    slot_bytes: int,
    tp0_skip: bool = True,
) -> List[RowPiece]:
    """Phase-2 deposit rows (plan 2.5): P chunk -> D owner rows.

    Chunk tokens ``[a, b)`` of P stage ``stage``; token ``i`` lands at D
    global slot ``e0 + i``, so its owner is the rank the owner rule names
    for that slot -- "P guesses nothing". The P-side row is ``i`` itself
    (the PP layout stores dense: pool row == slot id == chunk index), the
    D-side row the compact row from ``dcp_weighted_write_slots``. With
    ``tp0_skip`` the blocks bound for rank 0 carry ``route="skip"``: TP0's
    rows are filled from L2 at the wake (plan 2.1) and the lane plan must
    not spend a byte on them. The deposit region convention (which the
    wiring, L15-14, sizes against): the reserved slots span
    ``[e0 + a, e0 + b)``.
    """
    pieces: List[RowPiece] = []
    layers = tuple(stage_layers[stage])
    tokens = list(range(a, b))
    global_slots = [e0 + i for i in tokens]
    for rank in range(len(prefix) - 1):
        idx, compact = owned_rows(global_slots, prefix, rank)
        if not idx:
            continue
        if rank == 0 and tp0_skip:
            route = ROUTE_SKIP
        elif rank_card[rank] == stage_card[stage]:
            route = ROUTE_LOCAL
        else:
            route = ROUTE_LANE
        for (t0, t1), (l0, l1) in _runs(
            len(idx), len(layers), slot_bytes, row_bytes_per_layer
        ):
            toks = tuple(tokens[idx[j]] for j in range(t0, t1))
            layers_sub = tuple(layers[l0:l1])
            pieces.append(
                RowPiece(
                    src=f"pp{stage}",
                    dst=f"tp{rank}",
                    layers=layers_sub,
                    tokens=toks,
                    src_rows=toks,  # P row == chunk token index (identity)
                    dst_rows=tuple(compact[t0:t1]),
                    nbytes=len(toks) * len(layers_sub) * row_bytes_per_layer,
                    route=route,
                )
            )
    pieces.sort(key=_sort_key)
    return pieces
