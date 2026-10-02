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
    for i, e in enumerate(base["extents"]):
        eo, es = int(e[0]), int(e[1])
        key = int(e[2]) if len(e) > 2 else fd_index0 + i
        if eo <= lo and hi <= eo + es:
            t = mapped[key]
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
                for i, e in enumerate(b["extents"]):
                    key = int(e[2]) if len(e) > 2 else idx + i
                    if key not in mapped:
                        mapped[key] = map_extent(int(fds[key]), int(e[1]))
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


def _rows_of(mapped_or_tensor, base: dict, fd_index0: int, row: int,
             map_extent, fds) -> torch.Tensor:
    """Bytes of row ``row`` of one published mamba (layer) view."""
    off, unit = int(base["view_off"]), int(base["unit"])
    lo, hi = off + row * unit, off + (row + 1) * unit
    for i, e in enumerate(base["extents"]):
        eo, es = int(e[0]), int(e[1])
        if eo <= lo and hi <= eo + es:
            key = int(e[2]) if len(e) > 2 else fd_index0 + i
            if key not in mapped_or_tensor:
                mapped_or_tensor[key] = map_extent(int(fds[key]), es)
            return mapped_or_tensor[key][lo - eo:hi - eo]
    raise L15TakeError("mamba %s layer %s: anchor row %d not inside a hold extent"
                       % (base.get("role"), base.get("layer"), row))


def take_anchor(shares: Mapping[int, Tuple[dict, Sequence[int]]], *, rid: str,
                spec, ratios: Sequence[int], stage: Tuple[int, int],
                p_temporal: torch.Tensor, p_conv: torch.Tensor, p_slot: int,
                map_extent: Callable[[int, int], torch.Tensor]) -> int:
    """Rebuild the END anchor of ``rid`` for THIS stage's linear layers
    ``stage=(lo, hi)`` into P mamba slot ``p_slot``. D rank r's compact blob
    is cat(temporal[l, a]) ++ cat(conv[l, a]) over all its layers (its head
    share, l15_anchor_plan.rank_ranges order); the stage's is the same over
    its own layers with all heads (stage_ranges order). Pieces from
    plan_anchor(d_to_p). ``p_temporal``/``p_conv``: (stage layers, slots, ...)
    tensors of this stage. Returns bytes copied."""
    from sglang.srt.weg2.l15_anchor_plan import plan_anchor
    from sglang.srt.weg2.l15_handover_copy import apply_anchor_pieces

    lo, hi = int(stage[0]), int(stage[1])
    R = len(ratios)
    blobs: Dict[str, torch.Tensor] = {}
    for r in range(R):
        if r not in shares:
            raise L15TakeError("hold share missing for D rank %d" % r)
        d, fds = shares[r]
        a = None
        for s in d.get("spans", ()):
            if s.get("rid") == rid:
                a = int(s.get("anchor_slot", -1))
        if a is None or a < 0:
            raise L15TakeError("rid %s has no held anchor on D rank %d" % (rid, r))
        temporal: Dict[int, torch.Tensor] = {}
        conv: Dict[int, torch.Tensor] = {}
        mapped: Dict[int, torch.Tensor] = {}
        idx = 0
        for b in d["bases"]:
            role = b.get("role")
            if role in ("mamba_temporal", "mamba_conv0"):
                row = _rows_of(mapped, b, idx, a, map_extent, fds)
                (temporal if role == "mamba_temporal" else conv)[int(b["layer"])] = row
            elif str(role).startswith("mamba_conv"):
                raise L15TakeError("D rank %d publishes %s: only one conv tensor "
                                   "is supported" % (r, role))
            idx += len(b["extents"])
        L = spec.num_layers
        if sorted(temporal) != list(range(L)) or sorted(conv) != list(range(L)):
            raise L15TakeError("D rank %d publishes %d/%d temporal and %d/%d conv "
                               "layers" % (r, len(temporal), L, len(conv), L))
        blob = torch.cat([temporal[l].reshape(-1) for l in range(L)]
                         + [conv[l].reshape(-1) for l in range(L)])
        rs = spec.shard_for_rank(list(ratios), r)
        if blob.numel() != rs.total_bytes:
            raise L15TakeError("D rank %d anchor blob %d bytes != spec %d"
                               % (r, blob.numel(), rs.total_bytes))
        blobs["tp%d" % r] = blob
    n_l = hi - lo
    pt = p_temporal.view(torch.uint8).reshape(p_temporal.shape[0], p_temporal.shape[1], -1)
    pc = p_conv.view(torch.uint8).reshape(p_conv.shape[0], p_conv.shape[1], -1)
    if pt.shape[0] != n_l or pc.shape[0] != n_l:
        raise L15TakeError("stage holds %d/%d layers, plan needs %d"
                           % (pt.shape[0], pc.shape[0], n_l))
    stage_blob = torch.cat([pt[l, p_slot].reshape(-1) for l in range(n_l)]
                           + [pc[l, p_slot].reshape(-1) for l in range(n_l)]).clone()
    if stage_blob.numel() != spec.for_layers(lo, hi).total_bytes:
        raise L15TakeError("stage blob %d bytes != spec %d"
                           % (stage_blob.numel(), spec.for_layers(lo, hi).total_bytes))
    pieces = plan_anchor(spec, list(ratios), [(lo, hi)], [0], [0] * R, "d_to_p")
    n = apply_anchor_pieces([p.__class__(p.src, p.dst, p.canon_off, p.length,
                                         p.src_off, p.dst_off, "local")
                             for p in pieces], blobs, {"pp0": stage_blob},
                            routes={"local"})
    t_bytes = pt[0, p_slot].numel()
    c_bytes = pc[0, p_slot].numel()
    for l in range(n_l):
        pt[l, p_slot].copy_(stage_blob[l * t_bytes:(l + 1) * t_bytes])
        base = n_l * t_bytes
        pc[l, p_slot].copy_(stage_blob[base + l * c_bytes: base + (l + 1) * c_bytes])
    return n
