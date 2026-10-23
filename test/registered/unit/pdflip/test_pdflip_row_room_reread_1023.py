# SPDX-License-Identifier: Apache-2.0
"""ROW-ROOM-REREAD (27B INT4 flip, boot ..._a45112428e_1023_140032, P PP1 14:33:43Z, rid pdflip-23-212).

PP0 loaded the 78271-token host hit in the row's pass (``SF LOADBACK-ROOM SAME-PASS ... evictable=89453``). PP1, executing
the row, read ``evictable=1902`` (the retained 22-211 leaf was held by an in-flight write-through), evicted 1902, and the
eviction's own #1465 WRITE-BACK DRAIN (n=1, 3730 ms) released the 105k rows behind the leaf -- after the one read the
verdict was taken from: ``ROW-ROOM ok=False`` -> AgreedRoomShort -> group stop. ``row_room`` re-reads like ``local_pp_room``
(Q-640) so the follower pays the room the row needs whenever its tree can.
"""
import os
from types import SimpleNamespace
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import pp_loadback_verdict as LBV  # noqa: E402

KV = 78271
AVAIL = 38395
FIRST_READ = 1902       # PP1's evictable before the peel
HELD = 105728 - 1902    # behind the write-through-locked leaf until the drain
PP0_EVICTABLE = 89453


class _Alloc:
    def __init__(self, avail):
        self.avail = avail

    def available_size(self):
        return self.avail


class _Tree:
    """``held`` tokens become evictable when an eviction drains the in-flight write-through (the real evict() ends in
    ``writing_check(write_back=True)``, #1465)."""

    def __init__(self, avail, evictable, held=0):
        self.token_to_kv_pool_allocator = _Alloc(avail)
        self._evictable, self._held, self.evicted = evictable, held, []

    def evictable_size(self):
        return self._evictable

    def evict(self, params):
        n = min(int(params.num_tokens), self._evictable)
        self._evictable -= n
        self.token_to_kv_pool_allocator.avail += n
        self.evicted.append(n)
        self._evictable += self._held   # the drain released the held leaf's ancestors
        self._held = 0
        return SimpleNamespace(num_tokens_evicted=n)


def test_pp1_pays_the_row_pp0_loaded_after_the_drain_released_the_leaf():
    t = _Tree(AVAIL, FIRST_READ, held=HELD)
    ok, a0, ev0, evd, a1 = LBV.row_room(t, KV, "pdflip-23-212")
    assert ok is True and a1 >= KV
    assert (a0, ev0) == (AVAIL, FIRST_READ)              # the tuple still reports the FIRST read
    assert t.evicted == [FIRST_READ, KV - AVAIL - FIRST_READ]   # round 1 = all it could see, round 2 = the shortfall only
    assert evd == KV - AVAIL                            # never more than the shortfall in total


def test_a_peer_with_the_room_in_one_read_runs_one_round():
    t = _Tree(AVAIL, PP0_EVICTABLE)
    ok, *_ = LBV.row_room(t, KV, "pdflip-23-212")
    assert ok is True and t.evicted == [KV - AVAIL]


def test_a_true_residual_still_stops_by_name_without_spinning():
    t = _Tree(AVAIL, FIRST_READ, held=0)                 # nothing behind the leaf: genuinely short
    ok, a0, ev0, evd, a1 = LBV.row_room(t, KV, "pdflip-23-212")
    assert ok is False and a1 == AVAIL + FIRST_READ and t.evicted == [FIRST_READ]


def test_a_round_that_frees_and_releases_nothing_ends_the_loop():
    class _Stuck(_Tree):
        def evict(self, params):
            self.evicted.append(0)
            return SimpleNamespace(num_tokens_evicted=0)

    t = _Stuck(AVAIL, FIRST_READ)
    assert LBV.row_room(t, KV)[0] is False and t.evicted == [0]


def test_switch_off_is_the_single_round_of_before():
    with mock.patch.dict(os.environ, {LBV.ROW_ROOM_REREAD_ENV: "0"}):
        t = _Tree(AVAIL, FIRST_READ, held=HELD)
        assert LBV.row_room(t, KV)[0] is False and t.evicted == [FIRST_READ]


def test_the_line_names_the_reread(caplog):
    with caplog.at_level("INFO", logger=LBV.logger.name):
        LBV.row_room(_Tree(AVAIL, FIRST_READ, held=HELD), KV, "pdflip-23-212")
    line = [r.getMessage() for r in caplog.records if "ROW-ROOM" in r.getMessage()][0]
    assert "evictable=1902 evicted=%d avail_after=%d evictable_after=" % (KV - AVAIL, KV) in line and "rounds=2 ok=True" in line
