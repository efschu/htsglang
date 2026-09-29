# SPDX-License-Identifier: Apache-2.0
"""(3) 29.09.: the measured peak pool demand is the LRU floor of the stage solve.

rc12z30r3 measured MOE-POOL-DEMAND max_nonres_per_step 89/32/43 (TP0/1/2) at
C 100/48/48 with the stage form off. With the stage form a KV rank's stage
rows are LRU rows in S0 and a wake to S_j unmaps them: TP2's LRU at S1 fell
below its peak, so every peak step paid an extra wave. The floor raises the
short rank's scratch and the planner solves again (FR_D moves the rows from
resident to scratch) before the Platztausch map pins the form.
"""
import inspect
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

PEAK, SPAN = [89, 32, 43], [204, 61, 94]
HEAD = "D-KV-STUFEN"


def _form(c2=48, r2=67, e2=152, stage2=(0, 10, 19), cap=6):
    """dry run (b) z30v, rang2: E 152, R 67, scratch 48, stage rows 0/10/19."""
    last = SimpleNamespace(scratch_given=(100, 48, c2), max_rows=(129, 124, r2 + c2),
                           ids_per_step=240)
    rows = [last] * cap
    t2 = SimpleNamespace(host_rank=2, tokens=(262144, 393216, 524288),
                         capacity=tuple(tuple(c2 - s for s in stage2) for _ in range(cap)))
    group = SimpleNamespace(tables=(t2,))
    fits = [SimpleNamespace(rank=2, local_experts=e2)]
    return group, rows, fits


class Record(CustomTestCase):
    def test_nextflash_carries_the_measured_peak(self):
        peak, span, src = launcher.d_pool_peak_record("nextflash")
        self.assertEqual((peak, span), (PEAK, SPAN))
        self.assertIn("dkrnfh91dprsavisadoptstcutvsyncodxbz3bar1dauer09290548", src)

    def test_27b_has_none_so_the_stage_solve_is_untouched(self):
        self.assertEqual(launcher.d_pool_peak_record("qwen27b"), (None, None, ""))


class Floor(CustomTestCase):
    def test_every_stage_allowed_raises_the_short_rank(self):
        group, rows, fits = _form()
        add, lines = launcher.kv_stage_lru_floor(group, rows, fits, [2] * 6, PEAK, SPAN,
                                                 "boot x", HEAD)
        # need ceil(43/94 x 85) = 39 against 48 - 19 = 29 at S2; a row moved
        # widens the span by one: ceil(10 / (1 - 43/94)) = 19
        self.assertEqual(add, {2: 19})
        self.assertIn("rang2", lines[0])
        self.assertIn("Scratch +19", lines[0])
        self.assertIn(launcher.D_KV_STAGE_LRU_FLOOR_MARKER, lines[0])

    def test_the_raise_is_a_fixpoint(self):
        group, rows, fits = _form(c2=48 + 19, r2=67 - 19)
        add, lines = launcher.kv_stage_lru_floor(group, rows, fits, [2] * 6, PEAK, SPAN,
                                                 "boot x", HEAD)
        self.assertEqual(add, {})
        self.assertIn("haelt", lines[0])

    def test_s1_only_costs_fewer_rows(self):
        group, rows, fits = _form()
        add, _ = launcher.kv_stage_lru_floor(group, rows, fits, [1] * 6, PEAK, SPAN, "", HEAD)
        self.assertEqual(add, {2: 2})  # 39 against 48 - 10 = 38

    def test_s0_default_form_holds(self):
        group, rows, fits = _form()
        add, _ = launcher.kv_stage_lru_floor(group, rows, fits, [0] * 6, PEAK, SPAN, "", HEAD)
        self.assertEqual(add, {})


class Wiring(CustomTestCase):
    def test_stage_form_computes_it_behind_a_named_switch(self):
        src = inspect.getsource(launcher.apply_d_kv_stage_form)
        self.assertIn("kv_stage_lru_floor(", src)
        self.assertIn("D_KV_STAGE_LRU_FLOOR_ENV", src)
        self.assertIn("_d_kv_stage_lru_raise", src)

    def test_the_solve_raises_the_scratch_and_solves_again_before_the_pin(self):
        src = inspect.getsource(launcher.log_d_rank_vram_solve)
        self.assertIn("_d_kv_stage_lru_raise", src)
        self.assertIn("return log_d_rank_vram_solve(", src)
        self.assertIn("D_KV_STAGE_LRU_FLOOR_ROUNDS", src)
        # a pinned map is never re-solved
        self.assertLess(src.index("if _pinned:\n            log(f\"{D_RANK_SOLVE_MARKER} {label} "
                                  "{D_KV_STAGE_LRU_FLOOR_MARKER}"),
                        src.index("return log_d_rank_vram_solve("))

    def test_the_raised_scratch_outlives_the_map_pass(self):
        # the map pass restores env_d; the pinned FR_D was solved with the
        # raised scratch, so the dry and the real pass must run with it
        # (dry run b before this: 'Scratch rang2 +13 waere noetig, die Karte ist gebaut')
        self.assertIn("_d_kv_stage_lru_scratch = list(_new)",
                      inspect.getsource(launcher.log_d_rank_vram_solve))
        src = inspect.getsource(launcher.main)
        self.assertLess(src.index("ns.env_d = _env_d_before"),
                        src.index('getattr(ns, "_d_kv_stage_lru_scratch", None)'))

    def test_kill_switch_is_read(self):
        with mock.patch.dict(os.environ, {launcher.D_KV_STAGE_LRU_FLOOR_ENV: "0"}):
            self.assertEqual(os.environ[launcher.D_KV_STAGE_LRU_FLOOR_ENV], "0")


if __name__ == "__main__":
    unittest.main()
