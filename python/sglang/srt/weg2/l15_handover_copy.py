"""L15-10 S3a: the copy executor of the hot D->P handover.

The plans are pure and already exist: l15_row_plan.plan_d_to_p gives one
RowPiece per (D rank, P stage) block -- per layer the same source rows (D's
compact rows) and destination rows (P's dense rows) -- and
l15_anchor_plan.plan_anchor gives byte pieces of the canonical END-anchor blob.
This module EXECUTES them on buffers it is handed:

* :func:`apply_row_pieces` -- ``dst[layer][rows_dst] = src[layer][rows_src]``
  for K and V of every layer a piece names;
* :func:`apply_anchor_pieces` -- byte ranges between flat uint8 views.

The route filter runs the "local" pieces (same card, D2D) separately from the
"lane" pieces (BAR1 lanes, S5). Every buffer a piece needs is checked FIRST;
a missing one raises L15CopyError before any byte moves, so the caller's
fallback (today's store read) starts from an untouched destination.

On the metal the source buffers are D's KV pages mapped into P's address
space (S3b, exportable VMM extents); here they are any torch tensors, so the
executor is tested on CPU.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch


class L15CopyError(RuntimeError):
    """A planned piece cannot be executed (named); nothing was copied."""


def _rows(t, rows: Sequence[int]):
    return torch.tensor(list(rows), dtype=torch.int64, device=t.device)


def apply_row_pieces(pieces, src: Mapping[int, Tuple[torch.Tensor, torch.Tensor]],
                     dst: Mapping[int, Tuple[torch.Tensor, torch.Tensor]],
                     routes: Iterable[str]) -> int:
    """Copy the K/V cells of every piece whose route is in ``routes``.
    ``src``/``dst``: global attention layer -> (k_buffer, v_buffer). Returns
    the number of (layer, token) cells copied."""
    want = set(routes)
    todo = [p for p in pieces if p.route in want]
    for p in todo:
        if len(p.src_rows) != len(p.dst_rows):
            raise L15CopyError("piece %s->%s: %d src rows vs %d dst rows"
                               % (p.src, p.dst, len(p.src_rows), len(p.dst_rows)))
        for l in p.layers:
            if l not in src or l not in dst:
                raise L15CopyError("piece %s->%s: no buffer for layer %d"
                                   % (p.src, p.dst, l))
    cells = 0
    for p in todo:
        for l in p.layers:
            sk, sv = src[l]
            dk, dv = dst[l]
            si, di = _rows(sk, p.src_rows), _rows(dk, p.dst_rows)
            dk.index_copy_(0, di, sk.index_select(0, si).to(dk.device))
            dv.index_copy_(0, di, sv.index_select(0, si).to(dv.device))
            cells += len(p.dst_rows)
    return cells


def apply_anchor_pieces(pieces, src: Mapping[str, torch.Tensor],
                        dst: Mapping[str, torch.Tensor],
                        routes: Iterable[str]) -> int:
    """Copy the anchor byte pieces whose route is in ``routes``; ``src`` /
    ``dst`` map an endpoint name (``tp<r>`` / ``pp<s>``) to a flat uint8 view
    of its compact anchor buffer. Returns bytes copied."""
    want = set(routes)
    todo = [p for p in pieces if p.route in want]
    for p in todo:
        if p.src not in src or p.dst not in dst:
            raise L15CopyError("anchor piece %s->%s: no buffer" % (p.src, p.dst))
        if (p.src_off + p.length > src[p.src].numel()
                or p.dst_off + p.length > dst[p.dst].numel()):
            raise L15CopyError("anchor piece %s->%s: range past the buffer"
                               % (p.src, p.dst))
    n = 0
    for p in todo:
        s, d = src[p.src], dst[p.dst]
        d[p.dst_off:p.dst_off + p.length].copy_(
            s[p.src_off:p.src_off + p.length].to(d.device))
        n += p.length
    return n
