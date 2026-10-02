# SPDX-License-Identifier: Apache-2.0
"""L15-10 S4n-b: a P stage takes a hot prefix's KV from D's published hold
extents (fake mapping: the "extent" is the D buffer's own bytes)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang.srt.weg2.l15_share_take import L15TakeError, take_kv

UNIT = 8          # row bytes (uint8 rows)
PREFIX = [0, 2, 3]   # 2 D ranks over S=3: rank0 residues {0,1}, rank1 {2}
LAYERS = [0, 1]


def _compact(slot, rank):
    lo, hi = PREFIX[rank], PREFIX[rank + 1]
    return (slot // 3) * (hi - lo) + (slot % 3 - lo)


def _owner(slot):
    return 0 if slot % 3 < 2 else 1


def _d_side(slots):
    """Per rank: k/v buffers per layer filled with (layer, role, slot) bytes,
    the descriptor and the fd table (fd = index into a registry)."""
    reg = {}
    shares = {}
    nfd = 0
    for r in (0, 1):
        rows = 16
        bases = []
        fds = []
        for l in LAYERS:
            for role in ("k", "v"):
                buf = torch.zeros(rows, UNIT, dtype=torch.uint8)
                for s in slots:
                    if _owner(s) == r:
                        buf[_compact(s, r)] = 10 * l + (1 if role == "k" else 2) + 100 * (s % 2)
                reg[nfd] = buf.view(-1)
                fds.append(nfd)
                nfd += 1
                bases.append({"role": role, "layer": l, "view_off": 0,
                              "unit": UNIT, "extents": [[0, rows * UNIT]]})
        desc = {"prefix": PREFIX, "bases": bases,
                "spans": [{"rid": "r1", "depth": len(slots), "slots": slots,
                           "anchor_slot": 1}]}
        shares[r] = (desc, fds)
    return shares, reg


def test_stage_rows_receive_each_tokens_cells():
    slots = [0, 1, 2, 3, 4, 5]
    shares, reg = _d_side(slots)
    p = {l: (torch.zeros(20, UNIT, dtype=torch.uint8),
             torch.zeros(20, UNIT, dtype=torch.uint8)) for l in LAYERS}
    p_rows = [11, 3, 17, 5, 9, 2]   # non-contiguous on purpose
    cells = take_kv(shares, rid="r1", n=6, stage_layers=LAYERS, p_buffers=p,
                    p_rows=p_rows, map_extent=lambda fd, size: reg[fd])
    assert cells == 6 * len(LAYERS)
    for t, s in enumerate(slots):
        for l in LAYERS:
            want_k = 10 * l + 1 + 100 * (s % 2)
            assert int(p[l][0][p_rows[t]][0]) == want_k
            assert int(p[l][1][p_rows[t]][0]) == want_k + 1


def test_unknown_rid_refuses_before_writing():
    shares, reg = _d_side([0, 1, 2])
    p = {l: (torch.zeros(8, UNIT, dtype=torch.uint8),
             torch.zeros(8, UNIT, dtype=torch.uint8)) for l in LAYERS}
    try:
        take_kv(shares, rid="nope", n=3, stage_layers=LAYERS, p_buffers=p,
                p_rows=[1, 2, 3], map_extent=lambda fd, size: reg[fd])
    except L15TakeError as exc:
        assert "not held" in str(exc)
    else:
        raise AssertionError("unknown rid must refuse")
    assert int(p[0][0].sum()) == 0


def test_row_size_mismatch_refuses():
    shares, reg = _d_side([0, 1, 2])
    p = {l: (torch.zeros(8, 2 * UNIT, dtype=torch.uint8),
             torch.zeros(8, 2 * UNIT, dtype=torch.uint8)) for l in LAYERS}
    try:
        take_kv(shares, rid="r1", n=3, stage_layers=LAYERS, p_buffers=p,
                p_rows=[1, 2, 3], map_extent=lambda fd, size: reg[fd])
    except L15TakeError as exc:
        assert "row" in str(exc)
    else:
        raise AssertionError("row size mismatch must refuse")
