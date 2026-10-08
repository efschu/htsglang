# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 U2 glue: a process's KV stage bounded by the shared card ledger.

DANGER DIRECTIONS guarded here:
* a grow takes only whole stages the ledger granted and returns the rest at
  once (nothing held that is not mapped -- I1 stays exact);
* a shrink releases exactly the bytes above the new stage;
* pressure never pushes a process below the stage its running work needs.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import card_kv_ledger as K
from flliper.srt.pdflip import dual_kv_arbiter as A
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

STAGES = (0, 16384, 32768, 65536, 98304, 131072)
CELL = 32768  # D bytes per token
MIB = 1 << 20


class DualKvArbiter(CustomTestCase):
    def setUp(self):
        path = os.path.join(tempfile.mkdtemp(prefix="wkva"), "card")
        self.d = K.CardKvLedger(path, "D")
        self.p = K.CardKvLedger(path, "P")
        self.d.join(4000 * MIB)
        self.p.join(4000 * MIB)

    def test_grow_takes_whole_stages_and_returns_rest(self):
        self.p.request(2500 * MIB)                       # P holds most of the card
        g = A.grow_to(self.d, STAGES, 0, 5, CELL)         # D wants 131072 x 32 KiB = 4 GiB
        # 1500 MiB free: stage 2 (32768 tok = 1024 MiB) fits, stage 3 (2048 MiB) does not
        self.assertEqual(g.stage, 2)
        self.assertEqual(g.granted, A.stage_bytes(STAGES, 2, CELL))
        self.assertEqual(g.returned, 1500 * MIB - A.stage_bytes(STAGES, 2, CELL))
        st = self.d.state()
        self.assertEqual(st.committed["D"], A.stage_bytes(STAGES, g.stage, CELL))
        self.assertLessEqual(sum(st.committed.values()), st.budget)

    def test_shrink_releases_exactly(self):
        g = A.grow_to(self.d, STAGES, 0, 4, CELL)
        self.assertEqual(g.stage, 4)
        n = A.shrink_to(self.d, STAGES, 4, 1, CELL)
        self.assertEqual(n, A.stage_bytes(STAGES, 4, CELL) - A.stage_bytes(STAGES, 1, CELL))
        self.assertEqual(self.d.state().committed["D"], A.stage_bytes(STAGES, 1, CELL))

    def test_pressure_respects_running_floor(self):
        tgt, freed = A.stage_for_pressure(STAGES, 5, 3000 * MIB, CELL, floor_tokens=40000)
        self.assertEqual(tgt, 3)                          # 65536 >= 40000 is the lowest allowed
        self.assertEqual(freed, A.stage_bytes(STAGES, 5, CELL) - A.stage_bytes(STAGES, 3, CELL))
        tgt, freed = A.stage_for_pressure(STAGES, 5, 1 * MIB, CELL, floor_tokens=0)
        self.assertEqual(tgt, 4)                          # the smallest step that pays
        self.assertEqual(A.stage_for_pressure(STAGES, 5, 0, CELL, 0), (5, 0))
