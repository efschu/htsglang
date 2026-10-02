"""L15-14c: a P stage deposits finished chunk tokens into D's held deposit
region (the reverse of l15_share_take), plus the END anchor after the last
chunk. D's pages are mapped through the same hold share (D publishes it at its
sleep with the deposit region); cap-0 ranks are skipped by name (skip_ranks:
their rows return from L2 at the wake)."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Sequence, Tuple

import torch

from sglang.srt.weg2.l15_share_take import L15TakeError, _rows_of, _view_rows


def put_kv(shares: Mapping[int, Tuple[dict, Sequence[int]]], *, e_start: int,
           a: int, b: int, stage_layers: Sequence[int],
           p_buffers: Mapping[int, Tuple[torch.Tensor, torch.Tensor]],
           p_rows: Sequence[int], skip_ranks: Sequence[int],
           map_extent: Callable[[int, int], torch.Tensor]) -> int:
    """Chunk tokens ``[a, b)`` (P rows ``p_rows``, same order) of this
    stage's layers -> D global slots ``[e_start + a, e_start + b)`` on every
    non-skipped rank. Returns cells written."""
    from sglang.srt.weg2.l15_handover_copy import apply_row_pieces
    from sglang.srt.weg2.l15_row_plan import plan_p_to_d

    if len(p_rows) != b - a or b <= a:
        raise L15TakeError("chunk [%d,%d) vs %d P rows" % (a, b, len(p_rows)))
    descs = {r: d for r, (d, _f) in shares.items()}
    prefix = [int(x) for x in next(iter(descs.values()))["prefix"]]
    R = len(prefix) - 1
    k0 = next(iter(p_buffers.values()))[0]
    row_bytes = int(k0[0].numel() * k0.element_size())
    pieces = plan_p_to_d(a, b, int(e_start), prefix, 0, [list(stage_layers)],
                         [0], [0] * R, row_bytes, 1 << 62,
                         skip_ranks=set(skip_ranks))
    src = {l: (kv[0].view(torch.uint8).view(kv[0].shape[0], row_bytes),
               kv[1].view(torch.uint8).view(kv[1].shape[0], row_bytes))
           for l, kv in p_buffers.items()}
    prow = {a + i: int(p_rows[i]) for i in range(b - a)}
    dst_by_rank: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
    for r in range(R):
        rp = [p for p in pieces if p.dst == "tp%d" % r and p.route != "skip"]
        if not rp:
            continue
        if r not in shares:
            raise L15TakeError("hold share missing for D rank %d" % r)
        need = max(max(p.dst_rows) for p in rp) + 1
        d, fds = shares[r]
        mapped: Dict[int, torch.Tensor] = {}
        views: Dict[Tuple[int, str], torch.Tensor] = {}
        for idx_b, base in enumerate(d["bases"]):
            if base.get("role") in ("k", "v") and int(base.get("layer", -1)) in stage_layers:
                for i, e in enumerate(base["extents"]):
                    key = int(e[2]) if len(e) > 2 else i
                    if key not in mapped:
                        mapped[key] = map_extent(int(fds[key]), int(e[1]))
                views[(int(base["layer"]), base["role"])] = _view_rows(
                    mapped, base, 0, row_bytes, need)
        dst = {}
        for l in stage_layers:
            if (l, "k") not in views or (l, "v") not in views:
                raise L15TakeError("D rank %d publishes no k/v for layer %d" % (r, l))
            dst[l] = (views[(l, "k")], views[(l, "v")])
        dst_by_rank[r] = dst
    cells = 0
    for r, dst in dst_by_rank.items():
        rp = [p.__class__(p.src, p.dst, p.layers, p.tokens,
                          tuple(prow[t] for t in p.tokens), p.dst_rows, p.nbytes,
                          "local")
              for p in pieces if p.dst == "tp%d" % r and p.route != "skip"]
        cells += apply_row_pieces(rp, src, dst, routes={"local"})
    return cells


def put_anchor(shares: Mapping[int, Tuple[dict, Sequence[int]]], *, spec,
               ratios: Sequence[int], stage: Tuple[int, int],
               p_temporal: torch.Tensor, p_conv: torch.Tensor, p_slot: int,
               anchor_row: int, skip_ranks: Sequence[int],
               map_extent: Callable[[int, int], torch.Tensor]) -> int:
    """This stage's linear layers of P mamba slot ``p_slot`` -> every
    non-skipped D rank's head share at its mamba row ``anchor_row`` (the
    reverse of take_anchor: stage blob -> plan_anchor(p_to_d) -> rank blobs,
    written back into D's rows). Returns bytes written."""
    from sglang.srt.weg2.l15_anchor_plan import plan_anchor
    from sglang.srt.weg2.l15_handover_copy import apply_anchor_pieces

    lo, hi = int(stage[0]), int(stage[1])
    n_l = hi - lo
    pt = p_temporal.view(torch.uint8).reshape(p_temporal.shape[0], p_temporal.shape[1], -1)
    pc = p_conv.view(torch.uint8).reshape(p_conv.shape[0], p_conv.shape[1], -1)
    stage_blob = torch.cat([pt[l, p_slot].reshape(-1) for l in range(n_l)]
                           + [pc[l, p_slot].reshape(-1) for l in range(n_l)])
    if stage_blob.numel() != spec.for_layers(lo, hi).total_bytes:
        raise L15TakeError("stage blob %d bytes != spec %d"
                           % (stage_blob.numel(), spec.for_layers(lo, hi).total_bytes))
    R = len(ratios)
    pieces = [p for p in plan_anchor(spec, list(ratios), [(lo, hi)], [0], [0] * R,
                                     "p_to_d", skip_ranks=set(skip_ranks))
              if p.route != "skip"]
    rank_rows: Dict[int, Tuple[list, list]] = {}
    blobs: Dict[str, torch.Tensor] = {}
    for r in sorted({int(p.dst[2:]) for p in pieces}):
        if r not in shares:
            raise L15TakeError("hold share missing for D rank %d" % r)
        d, fds = shares[r]
        temporal, conv, mapped = {}, {}, {}
        for base in d["bases"]:
            role = base.get("role")
            if role in ("mamba_temporal", "mamba_conv0"):
                row = _rows_of(mapped, base, 0, int(anchor_row), map_extent, fds)
                (temporal if role == "mamba_temporal" else conv)[int(base["layer"])] = row
        order = sorted(temporal)
        if len(order) != spec.num_layers or sorted(conv) != order:
            raise L15TakeError("D rank %d publishes an incomplete mamba layer set" % r)
        t_rows = [temporal[g] for g in order]
        c_rows = [conv[g] for g in order]
        blob = torch.cat([x.reshape(-1) for x in t_rows]
                         + [x.reshape(-1) for x in c_rows]).clone()
        if blob.numel() != spec.shard_for_rank(list(ratios), r).total_bytes:
            raise L15TakeError("D rank %d anchor rows do not match the spec" % r)
        blobs["tp%d" % r] = blob
        rank_rows[r] = (t_rows, c_rows)
    n = apply_anchor_pieces([p.__class__(p.src, p.dst, p.canon_off, p.length,
                                         p.src_off, p.dst_off, "local")
                             for p in pieces], {"pp0": stage_blob}, blobs,
                            routes={"local"})
    for r, (t_rows, c_rows) in rank_rows.items():
        blob, off = blobs["tp%d" % r], 0
        for x in t_rows + c_rows:
            k = x.numel()
            x.copy_(blob[off:off + k].view(x.shape))
            off += k
    return n
