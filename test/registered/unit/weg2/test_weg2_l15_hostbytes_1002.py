# SPDX-License-Identifier: Apache-2.0
"""L15-HOSTBYTES: per wake and rank, the bytes the hold kept off the host."""

from __future__ import annotations

import inspect

from sglang.srt.weg2 import l15_restore as R
from sglang.srt.weg2.l15_manifest import HoldSpan, Manifest

PREFIX = [0, 2, 3]   # rank 0 owns slot%3 in {0,1}, rank 1 owns slot%3 == 2


def _m():
    a = HoldSpan(rid="a", depth=6, slots=(0, 1, 2, 3, 4, 5), anchor_slot=1,
                 l2_slots=(), l2_gens=())
    b = HoldSpan(rid="b", depth=4, slots=(0, 1, 2, 8), anchor_slot=2,
                 l2_slots=(), l2_gens=())
    return Manifest(epoch=7, pid=1, spans=(a, b), rows_by_rank=(4, 2), anchor_slots=3)


def test_owned_rows_count_shared_prefix_once():
    assert R.owned_held_rows(_m(), 0, PREFIX) == 4      # 0,1,3,4
    assert R.owned_held_rows(_m(), 1, PREFIX) == 3      # 2,5,8


def test_capped_rank_saves_cap0_rank_refills_fallback_neither():
    line = R.hostbytes_line(_m(), 1, PREFIX, cap_rows=10, cell_bytes=100,
                            anchor_bytes=7, verdict="hold", epoch=7)
    assert "h2d_saved=314 h2d_refill=0 d2h_saved=0" in line and "flip=7 rank=1" in line
    line0 = R.hostbytes_line(_m(), 0, PREFIX, cap_rows=0, cell_bytes=100,
                             anchor_bytes=7, verdict="hold", epoch=7)
    assert "h2d_saved=0 h2d_refill=414" in line0
    lf = R.hostbytes_line(_m(), 1, PREFIX, 10, 100, 7, "fallback", 7)
    assert "h2d_saved=0 h2d_refill=0" in lf
    assert "rows=0" in R.hostbytes_line(None, 1, PREFIX, 10, 100, 7, "none", 0)


def test_wake_logs_timing_and_hostbytes_after_the_restore_line():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu)
    i = src.index("l15_restore.restore_line(")
    tail = src[i:i + 4000]
    assert "L15-WAKE-TIMING rank=%d verdict=%s refill_ms" in tail
    assert "l15_restore.hostbytes_line(" in tail
