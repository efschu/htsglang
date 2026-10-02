# SPDX-License-Identifier: Apache-2.0
"""L15-10 S3a: the copy executor of the hot handover (pure torch).

Given the planned RowPieces (l15_row_plan.plan_d_to_p) and the per-layer K/V
buffers of the D side (source, compact rows) and the P side (destination,
dense rows), copy every (layer, token) cell exactly where the plan says --
the route filter lets the same executor run the local pieces (same card)
now and the lane pieces later. Anchor pieces (l15_anchor_plan) are byte
ranges of the canonical anchor blob. CPU tensors here; on the metal the
source buffers are D's KV pages imported into P's address space.
"""

from __future__ import annotations

import torch

from sglang.srt.weg2 import l15_handover_copy as hc
from sglang.srt.weg2.l15_anchor_plan import AnchorPiece
from sglang.srt.weg2.l15_row_plan import RowPiece


def _bufs(layers, rows, fill):
    return {l: (torch.full((rows, 4), float(fill(l, 0))).clone(),
                torch.full((rows, 4), float(fill(l, 1))).clone())
            for l in layers}


def test_row_pieces_land_on_the_planned_rows_per_layer():
    src = {}
    for l in (1, 3):
        k = torch.arange(10 * 4, dtype=torch.float32).reshape(10, 4) + 100 * l
        src[l] = (k, k + 0.5)
    dst = {l: (torch.zeros(8, 4), torch.zeros(8, 4)) for l in (1, 3)}
    pieces = [RowPiece("tp1", "pp0", (1, 3), (0, 1), (5, 7), (0, 1), 64, "local"),
              RowPiece("tp2", "pp0", (1,), (2,), (2,), (2,), 32, "lane")]
    n = hc.apply_row_pieces(pieces, src, dst, routes={"local"})
    assert n == 2 * 2                      # 2 layers x 2 tokens, lane skipped
    for l in (1, 3):
        assert torch.equal(dst[l][0][0], src[l][0][5])
        assert torch.equal(dst[l][1][1], src[l][1][7])
    assert torch.count_nonzero(dst[1][0][2]) == 0   # lane piece not run


def test_missing_layer_buffer_raises_before_any_copy():
    src = {1: (torch.ones(4, 2), torch.ones(4, 2))}
    dst = {1: (torch.zeros(4, 2), torch.zeros(4, 2))}
    pieces = [RowPiece("tp1", "pp0", (1, 9), (0,), (0,), (0,), 8, "local")]
    try:
        hc.apply_row_pieces(pieces, src, dst, routes={"local"})
    except hc.L15CopyError as exc:
        assert "layer 9" in str(exc)
    else:
        raise AssertionError("a missing layer must refuse")
    assert torch.count_nonzero(dst[1][0]) == 0      # nothing copied


def test_anchor_pieces_copy_byte_ranges():
    src = torch.arange(64, dtype=torch.uint8)
    dst = torch.zeros(64, dtype=torch.uint8)
    pieces = [AnchorPiece("tp1", "pp0", 0, 8, 16, 0, "local"),
              AnchorPiece("tp1", "pp1", 8, 4, 24, 0, "lane")]
    n = hc.apply_anchor_pieces(pieces, {"tp1": src}, {"pp0": dst},
                               routes={"local"})
    assert n == 8
    assert torch.equal(dst[:8], src[16:24])
    assert torch.count_nonzero(dst[8:]) == 0
