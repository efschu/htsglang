# SPDX-License-Identifier: Apache-2.0
"""L15-14c: a P stage deposits a chunk into D's deposit region (fake mapping:
D's buffers are the "extents"); the skipped (cap-0) rank gets nothing."""

from __future__ import annotations

import torch

from sglang.srt.weg2.l15_deposit_put import put_kv

UNIT, PREFIX, LAYERS = 8, [0, 2, 3], [5, 7]


def _compact(slot, rank):
    lo, hi = PREFIX[rank], PREFIX[rank + 1]
    return (slot // 3) * (hi - lo) + (slot % 3 - lo)


def _owner(slot):
    return 0 if slot % 3 < 2 else 1


def _shares():
    reg, shares, nfd = {}, {}, 0
    for r in (0, 1):
        bases, fds = [], []
        for l in LAYERS:
            for role in ("k", "v"):
                buf = torch.zeros(32, UNIT, dtype=torch.uint8)
                reg[nfd] = buf
                bases.append({"role": role, "layer": l, "view_off": 0, "unit": UNIT,
                              "extents": [[0, buf.numel(), len(fds)]]})
                fds.append(nfd)
                nfd += 1
        shares[r] = ({"prefix": PREFIX, "bases": bases}, fds)
    return shares, reg


def test_chunk_lands_on_the_owner_rows_and_skips_the_cap0_rank():
    shares, reg = _shares()
    p = {}
    for l in LAYERS:
        k = torch.zeros(20, UNIT, dtype=torch.uint8)
        v = torch.zeros(20, UNIT, dtype=torch.uint8)
        for row in range(20):
            k[row] = row + 10 * l
            v[row] = row + 10 * l + 1
        p[l] = (k, v)
    p_rows = [9, 2, 14, 4]          # chunk tokens 2..5 at these P rows
    e_start = 30
    cells = put_kv(shares, e_start=e_start, a=2, b=6, stage_layers=LAYERS,
                   p_buffers=p, p_rows=p_rows, skip_ranks=[0],
                   map_extent=lambda fd, size: reg[fd].view(-1))
    owned1 = [t for t in range(2, 6) if _owner(e_start + t) == 1]
    assert cells == len(owned1) * len(LAYERS)
    for t in owned1:
        row = _compact(e_start + t, 1)
        prow = p_rows[t - 2]
        k_fd = [b for b in shares[1][0]["bases"] if b["role"] == "k" and b["layer"] == 5][0]
        kbuf = reg[shares[1][1][k_fd["extents"][0][2]]]
        assert int(kbuf[row][0]) == prow + 50
    for fd in shares[0][1]:
        assert int(reg[fd].sum()) == 0, "the skipped rank got bytes"


def test_anchor_deposit_writes_each_ranks_head_share_and_skips_cap0():
    from sglang.srt.mem_cache.hicache_migrate import MambaBlobSpec
    from sglang.srt.weg2.l15_anchor_plan import rank_ranges
    from sglang.srt.weg2.l15_deposit_put import put_anchor

    spec = MambaBlobSpec(num_layers=2, num_heads=4, head_dim=1, state_size=2,
                         conv_dim=8, conv_width=1, key_dim=2, value_dim=4, units=2,
                         temporal_itemsize=1, conv_itemsize=1)
    ratios = [1, 1]
    canon = torch.arange(spec.total_bytes, dtype=torch.uint8) + 1
    tl, cl = spec.temporal_layer_bytes, spec.conv_layer_bytes
    pt = torch.zeros(2, 4, tl, dtype=torch.uint8)
    pc = torch.zeros(2, 4, cl, dtype=torch.uint8)
    for l in range(2):
        pt[l, 3] = canon[l * tl:(l + 1) * tl]
        pc[l, 3] = canon[2 * tl + l * cl: 2 * tl + (l + 1) * cl]
    reg, shares, ids = {}, {}, {}
    for r in range(2):
        rs = spec.shard_for_rank(ratios, r)
        temporal = torch.zeros(2, 6, rs.temporal_layer_bytes, dtype=torch.uint8)
        conv = torch.zeros(2, 6, rs.conv_layer_bytes, dtype=torch.uint8)
        bases, fds = [], []
        for name, t in (("mamba_temporal", temporal), ("mamba_conv0", conv)):
            fid = len(reg)
            reg[fid] = t
            ids[(r, name)] = fid
            for l in range(2):
                bases.append({"role": name, "layer": 10 + 2 * l,
                              "view_off": l * t.stride(0), "unit": t.stride(1),
                              "extents": [[0, t.numel(), len(fds)]]})
            fds.append(fid)
        shares[r] = ({"bases": bases}, fds)
    n = put_anchor(shares, spec=spec, ratios=ratios, stage=(0, 2), p_temporal=pt,
                   p_conv=pc, p_slot=3, anchor_row=4, skip_ranks=[0],
                   map_extent=lambda fd, size: reg[fd].reshape(-1))
    rs1 = spec.shard_for_rank(ratios, 1)
    want = torch.cat([canon[o:o + k] for o, k in rank_ranges(spec, ratios, 1)])
    got = torch.cat([reg[ids[(1, "mamba_temporal")]][l, 4] for l in range(2)]
                    + [reg[ids[(1, "mamba_conv0")]][l, 4] for l in range(2)])
    assert n == rs1.total_bytes and torch.equal(got, want)
    assert int(reg[ids[(0, "mamba_temporal")]].sum()) == 0
    assert int(reg[ids[(0, "mamba_conv0")]].sum()) == 0
