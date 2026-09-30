# SPDX-License-Identifier: Apache-2.0
"""sync_tables x RW-FINISH: a warm row reserved by ``reserve_warm_rows``
(row_key = SEAT_OFF_KEY until its copy lands) is OFF for an eager sync too.

Metal: y4u f50f51020f, D TP1 15:18:30Z -- the wake's RW WAKE-FIRST-TOKEN had
``finish_layers=40 finish_rows=1120 finish_landed=0`` on TP1 (TP0/TP2 logged
RW FINISH-LANDED), an eager extend (weg2-13-21, uncached=187) synced the pool
tables, ``sync_tables`` read ``old = 0x7FFFFFFF`` from a reserved row, passed
``old >= 0`` and indexed ``hot[old]``: IndexError index 2147483647 out of
bounds for size 128, scheduler_exception, W17 group D dead.

Pinned (CPU pool tables): both sync forms leave a reserved row reserved, free
no expert through it, keep the bijection for the other rows, count it as not
owned, and the landed warm still commits afterwards.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.layers.moe import expert_pool_device as epd  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

E, R, ROWS, STAGING = 16, 4, 12, 2  # residents 0..3, LRU rows 4..9, staging 10..11


def _tables():
    hot = {e: e for e in range(R)}
    host = [-1] * R + list(range(E - R))
    return epd.allocate_pool_tables("cpu", E, ROWS, R, STAGING, hot, host)


def _own(t, r, e, u=1):
    t.row_key[r] = e
    t.hot_phys[e] = r
    t.row_use[r] = u


class SyncKeepsReservedWarmRows(unittest.TestCase):
    def _run(self, keep):
        t = _tables()
        _own(t, 4, 11)
        _own(t, 5, 12)
        epd.reserve_warm_rows(t, torch.tensor([6, 7]))  # the warm rest has not landed
        # the eager pass wrote row 8 (expert 13); rows 4/5 were not written
        rep = epd.sync_tables(t, {8: 13}, keep_unwritten=keep)
        self.assertEqual([int(t.row_key[r]) for r in (6, 7)], [epd.SEAT_OFF_KEY] * 2)
        self.assertEqual([int(t.row_use[r]) for r in (6, 7)], [epd.ROW_USE_NEVER] * 2)
        self.assertEqual((int(t.hot_phys[13]), int(t.row_key[8])), (8, 13))
        self.assertEqual(epd.bijection_breaks(t), 0)
        return t, rep

    def test_keep_form(self):
        t, rep = self._run(keep=True)
        self.assertEqual((int(t.hot_phys[11]), int(t.hot_phys[12])), (4, 5))  # kept rows keep
        self.assertEqual(rep.owned, 3)  # 4, 5, 8 -- never the reserved 6, 7

    def test_clear_form(self):
        t, rep = self._run(keep=False)
        self.assertEqual((int(t.hot_phys[11]), int(t.hot_phys[12])), (-1, -1))  # cleared as before
        self.assertEqual(rep.owned, 1)

    def test_the_warm_commits_after_the_sync(self):
        t, _ = self._run(keep=True)
        t.clock[0] = 9
        epd.commit_warm_rows(t, torch.tensor([14, 15]), torch.tensor([6, 7]))
        self.assertEqual([int(t.hot_phys[e]) for e in (14, 15)], [6, 7])
        self.assertEqual(epd.bijection_breaks(t), 0)


if __name__ == "__main__":
    unittest.main()
