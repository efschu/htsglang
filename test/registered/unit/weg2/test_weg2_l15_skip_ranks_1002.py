# SPDX-License-Identifier: Apache-2.0
"""L15 hardware-generic (user order 02.10. 07:05Z): the deposit plans skip the
ranks named by the caller (the cap-0 ranks), not rank 0 by position."""

from __future__ import annotations

from sglang.srt.mem_cache.hicache_migrate import MambaBlobSpec
from sglang.srt.weg2.l15_anchor_plan import plan_anchor
from sglang.srt.weg2.l15_row_plan import plan_p_to_d

PREFIX = [0, 7, 11, 16]
SPEC = MambaBlobSpec(num_layers=2, num_heads=8, head_dim=1, state_size=2,
                     conv_dim=24, conv_width=1, key_dim=8, value_dim=8, units=4,
                     temporal_itemsize=1, conv_itemsize=1)


def _routes(pieces):
    return {p.dst: p.route for p in pieces}


def test_row_plan_skips_the_named_rank_not_rank0():
    pcs = plan_p_to_d(0, 16, 0, PREFIX, 0, [[0, 1]], [0], [0, 0, 0], 8,
                      1 << 30, skip_ranks={2})
    r = _routes(pcs)
    assert r["tp2"] == "skip" and r["tp0"] != "skip"


def test_anchor_plan_skips_the_named_rank_not_rank0():
    pcs = plan_anchor(SPEC, [2, 1, 1], [(0, 2)], [0], [0, 0, 0], "p_to_d",
                      skip_ranks={1})
    r = _routes(pcs)
    assert r.get("tp1") == "skip" and r.get("tp0") != "skip"


def test_legacy_default_still_skips_rank0():
    pcs = plan_p_to_d(0, 16, 0, PREFIX, 0, [[0, 1]], [0], [0, 0, 0], 8, 1 << 30)
    assert _routes(pcs)["tp0"] == "skip"
