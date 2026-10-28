# SPDX-License-Identifier: Apache-2.0
"""D-POOL-LEND on the 27B NVFP4 duo form (5090 + 3080, D = TP2 with UNEVEN DCP, 27.10.2026).

The duo's D pool is the one shape the base test (``test_pdflip_d_pool_lend_1027``, TOP 65536, one ratio) does not cover: boot
``dkr27bnvfp4duorebarbar1duo10250113`` (D log 01:17:02..03): EFFECTIVE max_total_num_tokens 536576, ownership vector [37, 27] over
S = 64 token blocks, K buffers of 310245 rows on TP0 and 226395 rows on TP1 (``KV Cache is allocated ... #tokens``), page 1, so a
tensor holds ``tokens + page`` rows. ``_tensor_geom`` answers None -- and the lend then arms NOTHING, named only by one WARNING at the
attach -- when a tensor's rows do not map the top through the owner block. That is the silent-failure path guarded here: the duo
would boot, never lend, and the width vote would stay at the old card-free line.
"""
from __future__ import annotations

import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from flliper.srt.pdflip import d_pool_lend as L
from flliper.srt.pdflip import d_seat_vram as SV
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

TOP = 536576                    # global tokens (boot log)
S = 64                          # token blocks of the uneven DCP (vector sum 37 + 27)
RATIOS = {"TP0 (5090)": 37, "TP1 (3080)": 27}
LOG_ROWS = {37: 310245, 27: 226395}          # "#tokens" of the K buffers in the boot log
STEP = L.resolve_step(L.knobs_from_env().step, TOP)   # the lend's lattice step (0 = auto: follows the pool size)


def _geom(ratio: int):
    compact = (TOP // S + 1) * ratio
    t = torch.zeros(compact + 1, 1)           # page 1: tokens + page rows
    pool = types.SimpleNamespace(size=compact, page_size=1)
    with mock.patch.object(SV, "kv_owner_block", return_value=(S, ratio)):
        return compact, L._tensor_geom(t, pool, TOP, (compact + 1) * 4 + 2 * (1 << 20), "k0")


class DuoGeometry(CustomTestCase):
    def test_the_boot_logs_row_counts_are_the_owner_block_compact_rows(self):
        for name, ratio in RATIOS.items():
            compact, geom = _geom(ratio)
            self.assertEqual(compact, LOG_ROWS[ratio], name)
            self.assertIsNotNone(geom, "%s: the lend would arm nothing (its rows do not map %d global tokens)" % (name, TOP))
            self.assertEqual(geom.slots_for(TOP), compact + 1, name)

    def test_the_lattice_step_is_a_whole_number_of_owner_blocks(self):
        self.assertEqual(STEP % S, 0)             # a cell edge inside an owner block would cut a block between two cells

    def test_a_128k_prefill_frees_most_of_the_pool_on_both_cards(self):
        t = L.need_tokens(live=0, incoming=131072, air=L.AIR_FLOOR_TOKENS, step=STEP)
        self.assertLessEqual(t, TOP - L.MIN_RELEASE_CELLS * STEP)           # size_pool releases (>= 2 cells go)
        for name, ratio in RATIOS.items():
            _, geom = _geom(ratio)
            kept, full = geom.slots_for(t), geom.slots_for(TOP)
            self.assertGreater((full - kept) / full, 0.55, name)            # BERECHNET: 4.4 GB on the 3080, 6.0 GB on the 5090
            self.assertLess(kept, full, name)
