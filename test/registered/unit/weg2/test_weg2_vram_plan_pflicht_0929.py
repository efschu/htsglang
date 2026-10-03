# SPDX-License-Identifier: Apache-2.0
"""VRAM-VERTRAG M1 + Pflicht (29.09., Nutzer 12:42Z): the token-exact demand.

"wir wissen ja eigentlich schon vor dem flip zu P, wie viel kv gebraucht wird.
aufs token genau": the plan carries the cells that turn the KNOWN tokens of a
wake into its duty -- P per stage (KV cell, row price, booked cap), D per KV
rank (stage ladder, stage rows), the arena (slots) -- and ONE set of functions
in vram_plan evaluates them for the front, the ranks and the arena seat. The
kvs2 boot (bs2 x 240k) died on a full P host arena (4 GiB): the arena duty of
those two prompts is 7680 slots against 5461.
"""
import os
import sys
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_weg2_vram_plan_m1_0929 as M1  # noqa: E402

from sglang.srt.weg2 import launcher as L  # noqa: E402
from sglang.srt.weg2 import vram_plan as vp  # noqa: E402
from sglang.srt.weg2 import vram_plan_view as vpv  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

#: z30y DRY 12:46Z, PP-CUT P-KARTE: KV 1904/816/544 MiB at 262144, row price
#: over the reference 70.1/26.6/19.3 MiB
P_KV = (1904.0, 816.0, 544.0)
P_ROW = (70.1, 26.6, 19.3)
#: z30y env: SGLANG_HICACHE_ARENA_GIB=4, KV_PAGE_BYTES=786432, MAMBA_SLOTS=32
ARENA_ENV = ("SGLANG_HICACHE_ARENA_GIB=4;SGLANG_HICACHE_ARENA_KV_PAGE_BYTES=786432;"
             "SGLANG_HICACHE_ARENA_MAMBA_SLOTS=32")


def _pfit(stage):
    return SimpleNamespace(stage=stage, fraction=0.5, buffer_rows=200, layer_row_mib=P_ROW[stage],
                           row_card_mib=P_ROW[stage], expert_mib=8000.0, kv_mib=P_KV[stage],
                           transient_mib=3000.0, transient_source="RECORD(P_ACTIVATION_MIB)",
                           draft_mib=0.0, prompt_tokens=262144, growth_mib=0.0,
                           headroom_mib=100.0, near_oom_mib=400.0, ceiling_fraction=None)


def _table(rank, cell, rows, stage_rows, tokens, low=0):
    return SimpleNamespace(host_rank=rank, kv_cell_bytes=cell, row_mib=112.5, rows=rows,
                           stage_rows=tuple(stage_rows), tokens=tuple(tokens), low_rows=low)


#: DRY floor32k (12:46Z): TP0 9 rows, TP1 23, TP2 30 over 32k..512k
LADDER = (32768, 65536, 98304, 131072, 163840, 196608, 229376, 262144, 393216, 524288)
GROUP = SimpleNamespace(tables=(
    _table(0, 1855, 9, (0, 0, 1, 1, 2, 2, 3, 3, 6, 8), LADDER, low=3),
    _table(1, 5376, 23, (0, 2, 3, 5, 6, 8, 9, 10, 16, 22), LADDER, low=10),
    _table(2, 6912, 30, (0, 2, 4, 6, 8, 10, 12, 13, 21, 29), LADDER, low=13)))


def _ns():
    return SimpleNamespace(tag="t", extra_p="--page-size 64 --max-total-tokens 262144",
                           extra_d="--page-size 64", env_p=ARENA_ENV, env_d=ARENA_ENV)


def _plan(d=True):
    cards, v = M1._z30r3_view()
    v.note_p_card(cards, [_pfit(s) for s in range(3)], [0.33, 0.701, 0.9956], [32, 32, 32],
                  "fnFL2x163", num_experts=512)
    if d:
        v.note_d_stages(GROUP)
    v.note_inputs(_ns())
    return v.build("map", cards, d_label="D(Karte, Erwartung)")


class Functions(CustomTestCase):
    def test_duty_is_whole_pages_per_sequence(self):
        self.assertEqual(vp.pages(1, 64), 1)
        self.assertEqual(vp.pages(64, 64), 1)
        self.assertEqual(vp.pages(65, 64), 2)
        # two sequences never share a page
        self.assertEqual(vp.demand_tokens([65, 1], 64), 3 * 64)

    def test_rows_freed_rounds_down_and_stops_at_the_cap(self):
        cell = int(round(1904 * vp.MIB / 262144))  # 7616 B/token on PP0
        self.assertEqual(cell, 7616)
        # half the cap free on PP0: 131072 x 7616 B over 70.1 MiB rows = 13.58 -> 13
        self.assertEqual(vp.rows_freed(262144, 131072, cell, 70.1), 13)
        self.assertLessEqual(13 * 70.1 * vp.MIB, 131072 * cell)
        self.assertEqual(vp.rows_freed(262144, 300000, cell, 70.1), 0)

    def test_stage_is_the_smallest_holding_the_duty(self):
        self.assertEqual(vp.stage_of(1, LADDER), 0)
        self.assertEqual(vp.stage_of(32768, LADDER), 0)
        self.assertEqual(vp.stage_of(32769, LADDER), 1)
        self.assertEqual(vp.stage_of(262144, LADDER), 7)
        self.assertEqual(vp.stage_of(10 ** 7, LADDER), 9)  # beyond the top: the top


class Arena(CustomTestCase):
    def test_kvs2_bs2x240k_overflows_the_4gib_arena(self):
        dm = _plan()["demand"]
        self.assertEqual(dm["arena"]["kv_slots"], 4 * (1 << 30) // 786432)  # 5461
        ev = vp.evaluate_demand(dm, p_tokens=[245760, 245760])
        self.assertEqual(ev["arena"]["slots_needed"], 7680)
        self.assertFalse(ev["arena"]["fits"])
        self.assertTrue(vp.evaluate_demand(dm, p_tokens=[131072, 131072])["arena"]["fits"])

    def test_slots_are_the_arena_pools_own_count(self):
        from sglang.srt.mem_cache.pool_host import arena_pool

        old = {k: os.environ.get(k) for k in ("SGLANG_HICACHE_ARENA_GIB",
                                             arena_pool.ENV_ARENA_KV_PAGE_BYTES)}
        os.environ["SGLANG_HICACHE_ARENA_GIB"] = "4"
        os.environ[arena_pool.ENV_ARENA_KV_PAGE_BYTES] = "786432"
        try:
            self.assertEqual(arena_pool.planned_arena_slots(1),
                             _plan()["demand"]["arena"]["kv_slots"])
        finally:
            for k, val in old.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val


class Plan(CustomTestCase):
    def test_the_plan_carries_p_and_d_cells(self):
        dm = _plan()["demand"]
        self.assertEqual(dm["P"]["cap_tokens"], 262144)
        self.assertEqual(dm["P"]["page_tokens"], 64)
        self.assertEqual([dm["P"]["ranks"]["pp%d" % s]["kv_cell_bytes"] for s in range(3)],
                         [7616, 3264, 2176])
        self.assertEqual(dm["D"]["stage_tokens"], list(LADDER))
        self.assertEqual({r: c["rows"] for r, c in dm["D"]["ranks"].items()},
                         {"tp0": 9, "tp1": 23, "tp2": 30})
        self.assertEqual(dm["D"]["ranks"]["tp2"]["low_rows"], 13)

    def test_known_tokens_choose_stage_and_rows(self):
        ev = vp.evaluate_demand(_plan()["demand"], p_tokens=[20000], d_tokens=[20000])
        # one 20k session: D on the floor, every stage row ON
        self.assertEqual((ev["D"]["stage"], ev["D"]["stage_tokens"]), (0, 32768))
        self.assertEqual({r: v["rows_on"] for r, v in ev["D"]["ranks"].items()},
                         {"tp0": 9, "tp1": 23, "tp2": 30})
        # P: 20032 of 262144 booked -> 242112 x 7616 B / 70.1 MiB = 25.08 -> 25 rows on PP0
        self.assertEqual(ev["P"]["ranks"]["pp0"]["rows_freed"], 25)
        ev = vp.evaluate_demand(_plan()["demand"], d_tokens=[131072, 131072])
        self.assertEqual((ev["D"]["stage"], ev["D"]["ranks"]["tp1"]["rows_on"]), (7, 13))

    def test_a_plan_without_cells_is_todays_plan(self):
        cards, v = M1._z30r3_view()
        v.note_inputs(SimpleNamespace())
        plan = v.build("map", cards, d_label="D(Karte, Erwartung)")
        self.assertNotIn("demand", plan)  # same bytes, same plan_id as the z30y image
        vp.read_plan(plan)

    def test_unknown_demand_field_is_refused(self):
        p = _plan()
        p["demand"]["P"]["ranks"]["pp0"]["guess"] = 1
        p["plan_id"] = vp.compute_plan_id(p)
        with self.assertRaises(vp.VramPlanRefused) as cm:
            vp.read_plan(p)
        self.assertEqual(cm.exception.code, "VRAM_PLAN_FIELD")

    def test_undo_drops_the_stage_cells(self):
        cards, v = M1._z30r3_view()
        v.note_d_stages(GROUP)
        v.note_d_stages(None)
        v.note_inputs(_ns())
        self.assertNotIn("D", v.build("map", cards, d_label="D(Karte, Erwartung)")["demand"])


class Launcher(CustomTestCase):
    def test_emit_logs_the_duty_line(self):
        cards, v = M1._z30r3_view()
        v.note_p_card(cards, [_pfit(s) for s in range(3)], [0.33, 0.701, 0.9956], [32, 32, 32],
                      "fnFL2x163", num_experts=512)
        v.note_d_stages(GROUP)
        L._VRAM_VIEW = v
        lines = []
        old = os.environ.pop("WEG2_STATE_DIR", None)
        try:
            L.vram_plan_emit(_ns(), cards, "map", lines.append, d_label="D(Karte, Erwartung)")
        finally:
            L._VRAM_VIEW = None
            if old is not None:
                os.environ["WEG2_STATE_DIR"] = old
        duty = [x for x in lines if x.startswith("VRAM-PLAN-PFLICHT")]
        self.assertEqual(len(duty), 1, lines)
        self.assertIn("2x240k(245760+245760)", duty[0])
        self.assertIn("arena 7680/5461 VOLL", duty[0])

    def test_stage_form_notes_the_view(self):
        import inspect

        self.assertIn("vram_view().note_d_stages(group)", inspect.getsource(L.apply_d_kv_stage_form))
        self.assertIn("vram_view().note_d_stages(None)", inspect.getsource(L.d_kv_stage_undo))


if __name__ == "__main__":
    unittest.main()
