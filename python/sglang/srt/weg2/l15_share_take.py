"""L15-10 S4n-b: one P stage takes a hot prefix's KV from D's published hold.

Inputs: every D rank's (descriptor, fds) from l15_share_publish.fetch_share,
the follow-up's predecessor rid and the prefix length ``n``, this stage's
global attention layers, its own K/V buffers and the P rows it allocated for
the prefix. Steps:

1. the span of ``rid`` (the same in every rank's manifest) gives the prefix's
   global D slots ``slots[:n]``;
2. l15_row_plan.plan_d_to_p(...) -> the pieces bound for THIS stage (one per
   D rank), destination rows remapped onto the rows P allocated;
3. per D rank, the hold extent holding each (layer, k|v) view is mapped once
   (``map_extent(fd, size)`` -> uint8 tensor; vmm import on the metal) and
   viewed as that rank's compact rows;
4. l15_handover_copy.apply_row_pieces per rank, all as uint8 rows (row bytes
   must agree on both sides, else refused).

Every refusal raises L15TakeError before P's rows are written (all mapping
and checks first); the caller falls back to the store read and frees the rows.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import torch


class L15TakeError(RuntimeError):
    """The hot prefix cannot be taken (named); P's rows untouched."""


def _span_slots(descs: Mapping[int, dict], rid: str, n: int) -> List[int]:
    for d in descs.values():
        for s in d.get("spans", ()):
            if s.get("rid") == rid:
                slots = [int(x) for x in s.get("slots", ())]
                if len(slots) < n:
                    raise L15TakeError("span %s holds %d slots < prefix %d"
                                       % (rid, len(slots), n))
                return slots[:n]
    raise L15TakeError("rid %s is not held by D" % rid)


def _view_rows(mapped: Dict[int, torch.Tensor], base: dict, fd_index0: int,
               row_bytes: int, rows_needed: int) -> torch.Tensor:
    """The compact-row view of one (layer, role) base from its mapped hold
    extent(s): the extent that contains [view_off, view_off + rows*unit)."""
    off, unit = int(base["view_off"]), int(base["unit"])
    if unit != row_bytes:
        raise L15TakeError("layer %s %s: D row %d bytes != P row %d bytes"
                           % (base.get("layer"), base.get("role"), unit, row_bytes))
    lo, hi = off, off + rows_needed * unit
    for i, (eo, es) in enumerate(base["extents"]):
        eo, es = int(eo), int(es)
        if eo <= lo and hi <= eo + es:
            t = mapped[fd_index0 + i]
            return t[lo - eo: lo - eo + rows_needed * unit].view(rows_needed, unit)
    raise L15TakeError("layer %s %s: rows [0,%d) not inside one hold extent"
                       % (base.get("layer"), base.get("role"), rows_needed))


def take_kv(shares: Mapping[int, Tuple[dict, Sequence[int]]], *, rid: str, n: int,
            stage_layers: Sequence[int],
            p_buffers: Mapping[int, Tuple[torch.Tensor, torch.Tensor]],
            p_rows: Sequence[int],
            map_extent: Callable[[int, int], torch.Tensor]) -> int:
    """Copy the prefix's KV cells for this stage's layers into ``p_rows``.
    ``shares``: D rank -> (descriptor, fds); ``p_buffers``: global attention
    layer -> (k, v) of this stage. Returns cells copied."""
    from sglang.srt.weg2.l15_handover_copy import apply_row_pieces
    from sglang.srt.weg2.l15_row_plan import plan_d_to_p

    if len(p_rows) != n or n <= 0:
        raise L15TakeError("prefix %d vs %d allocated P rows" % (n, len(p_rows)))
    descs = {r: d for r, (d, _f) in shares.items()}
    slots = _span_slots(descs, rid, n)
    any_desc = next(iter(descs.values()))
    prefix = [int(x) for x in any_desc["prefix"]]
    R = len(prefix) - 1
    if set(range(R)) - set(shares):
        raise L15TakeError("hold share missing for D rank(s) %s"
                           % sorted(set(range(R)) - set(shares)))
    k0 = next(iter(p_buffers.values()))[0]
    row_bytes = int(k0[0].numel() * k0.element_size())
    pieces = plan_d_to_p(slots, prefix, [list(stage_layers)], [0], [0] * R,
                         0, row_bytes, 1 << 62)
    # map + view everything FIRST (all-or-nothing before any write)
    src_by_rank: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
    for r in range(R):
        rp = [p for p in pieces if p.src == "tp%d" % r]
        if not rp:
            continue
        need = max(max(p.src_rows) for p in rp) + 1
        d, fds = shares[r]
        mapped: Dict[int, torch.Tensor] = {}
        views: Dict[Tuple[int, str], torch.Tensor] = {}
        idx = 0
        for b in d["bases"]:
            n_ext = len(b["extents"])
            if b.get("role") in ("k", "v") and int(b.get("layer", -1)) in stage_layers:
                for i, (eo, es) in enumerate(b["extents"]):
                    if idx + i not in mapped:
                        mapped[idx + i] = map_extent(int(fds[idx + i]), int(es))
                views[(int(b["layer"]), b["role"])] = _view_rows(
                    mapped, b, idx, row_bytes, need)
            idx += n_ext
        src = {}
        for l in stage_layers:
            if (l, "k") not in views or (l, "v") not in views:
                raise L15TakeError("D rank %d publishes no k/v for layer %d" % (r, l))
            src[l] = (views[(l, "k")], views[(l, "v")])
        src_by_rank[r] = src
    dst = {l: (kv[0].view(torch.uint8).view(kv[0].shape[0], row_bytes),
               kv[1].view(torch.uint8).view(kv[1].shape[0], row_bytes))
           for l, kv in p_buffers.items()}
    remap = {t: int(p_rows[t]) for t in range(n)}
    cells = 0
    for r, src in src_by_rank.items():
        rp = []
        for p in pieces:
            if p.src != "tp%d" % r:
                continue
            rp.append(p.__class__(p.src, p.dst, p.layers, p.tokens, p.src_rows,
                                  tuple(remap[t] for t in p.tokens), p.nbytes,
                                  "local"))
        cells += apply_row_pieces(rp, src, dst, routes={"local"})
    return cells
