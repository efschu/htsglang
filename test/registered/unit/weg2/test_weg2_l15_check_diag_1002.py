# SPDX-License-Identifier: Apache-2.0
"""L15-CHECK-DIAG (N3r bad=64/64): a failing sample check names WHY.

N3r's "L15-CHECK rank=1 ok=0 bad=64" cannot tell a misaligned l2_slot<->token
map (the L2 row of token i holds token j's KV) from stale/foreign L2 bytes
(P claimed the slot) or zero pages. check_diag() compares each bad sampled
L2 row against EVERY live sampled row: a hit at another position is a
misalignment (offset printed), no hit with a zero L2 row is an empty page,
no hit otherwise is foreign bytes.
"""

from __future__ import annotations

import torch

from sglang.srt.weg2.l15_check import check_diag


def _rows(vals):
    return [torch.tensor([float(v), float(v) + 0.5]) for v in vals]


def test_misalignment_is_named_with_its_offset():
    live = _rows([1, 2, 3, 4])
    l2 = _rows([2, 3, 4, 9])         # each L2 row is the NEXT token's KV
    line = check_diag(live, l2, [(f"r{i}", i, 100 + i, 1) for i in range(4)])
    assert "misaligned=3" in line and "offsets=[1]" in line
    assert "foreign=1" in line


def test_zero_pages_are_named():
    live = _rows([1, 2])
    l2 = [torch.zeros(2), torch.zeros(2)]
    line = check_diag(live, l2, [("r", 0, 5, 1), ("r", 1, 6, 1)])
    assert "zero=2" in line


def test_all_equal_reports_nothing_bad():
    live = _rows([1, 2])
    line = check_diag(live, _rows([1, 2]), [("r", 0, 5, 1), ("r", 1, 6, 1)])
    assert "bad=0" in line
