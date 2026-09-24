"""fnFL2 H29b (SGLANG_WEG2_LRU_WARM_FROM_HANDOFF): D's LRU after the wake from P's routing.

After the wake ``rearm_after_wake`` rewrites the pool tables and the whole
device LRU is free; under H24 (no extend on D) the first decode rounds fetch
every non-resident expert row by row (x138 TP0: pool.fetch 29-80 ms in rounds
1-5 against 9 ms stationary). P publishes its last tokens' routing per layer;
D fills ONLY the rows the reinit left free (no VRAM), most-routed first, with
the bytes of the same store rows a miss would copy, and the tables keep the
pool's bijection.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import logging
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.layers.moe import expert_pool_device as ep
from sglang.srt.weg2 import decode_warm_handoff as dwh
from sglang.test.test_utils import CustomTestCase

# residents 0,1 in rows 0,1; LRU rows 2..5; staging rows 6,7
E, ROWS, R, S = 10, 8, 2, 2
HOST = [-1, -1] + list(range(E - 2))


def _pool():
    t = ep.allocate_pool_tables("cpu", E, ROWS, R, S, {0: 0, 1: 1}, HOST)
    b = ep.allocate_step_buffers("cpu", E, 8)
    return t, b


def _assert_bijection(tc, t):
    hot, key = t.hot_phys.tolist(), t.row_key.tolist()
    for r in range(t.lru_start, t.pool_rows):
        if key[r] >= 0:
            tc.assertEqual(hot[key[r]], r)
    for e, r in enumerate(hot):
        if r >= t.lru_start:
            tc.assertEqual(key[r], e)
    for r in range(t.pool_rows, len(key)):
        tc.assertEqual(key[r], -1)


class TestRouteFile(CustomTestCase):
    def test_route_counts_most_routed_first_padding_and_foreign_dropped(self):
        rows = np.array([[3, 5, -1], [5, 7, 3], [5, 9, 42]])
        self.assertEqual(dwh.route_counts(rows, 10), [(5, 3), (3, 2), (7, 1), (9, 1)])

    def test_recorder_writes_once_per_forward_at_the_stage_last_layer(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(
            os.environ, {"SGLANG_HICACHE_ARENA_DIR": d}
        ):
            rec = dwh.RouteRecorder()
            for lid in (4, 5, 6):
                rec.register(lid)
            ids = np.arange(20).reshape(10, 2) % 8
            self.assertIsNone(rec.note(4, ids, 8, tokens=3))
            self.assertIsNone(rec.note(5, ids, 8, tokens=3))
            path = rec.note(6, ids, 8, tokens=3)
            self.assertTrue(path.endswith("lru_route.L4-6.json"))
            routes, files = dwh.load_routes()
            self.assertEqual(files, 1)
            self.assertEqual(sorted(routes), [4, 5, 6])
            # only the last 3 tokens: rows 7..9 = [14,15],[16,17],[18,19] % 8
            self.assertEqual(sorted(e for e, _ in routes[6]), [0, 1, 2, 3, 6, 7])

    def test_load_merges_stages_newer_wins_stale_and_torn_files_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            now = time.time()
            for name, t, layers in (
                ("lru_route.L0-1.json", now - 5, {"0": [[1, 2]], "1": [[2, 1]]}),
                ("lru_route.L1-2.json", now - 1, {"1": [[3, 4]], "2": [[4, 1]]}),
            ):
                with open(os.path.join(d, name), "w") as f:
                    json.dump({"t": t, "layers": layers}, f)
            old = os.path.join(d, "lru_route.L9-9.json")
            with open(old, "w") as f:
                json.dump({"t": now, "layers": {"9": [[5, 1]]}}, f)
            os.utime(old, (now - 5000, now - 5000))
            with open(os.path.join(d, "lru_route.Lx.json"), "w") as f:
                f.write("{torn")
            routes, files = dwh.load_routes(d, max_age_s=600)
            self.assertEqual(files, 2)
            self.assertEqual(routes, {0: [(1, 2)], 1: [(3, 4)], 2: [(4, 1)]})
            self.assertEqual(dwh.load_routes(os.path.join(d, "absent"), 600), ({}, 0))


class TestSeedLruRows(CustomTestCase):
    def test_free_rows_take_the_wanted_experts_and_the_next_step_hits_them(self):
        t, b = _pool()
        pairs = ep.seed_lru_rows(t, [5, 0, 5, 3, 99, -1, 7])
        # 0 is resident, the repeat of 5, 99 and -1 are skipped
        self.assertEqual(pairs, [(HOST[5], 2), (HOST[3], 3), (HOST[7], 4)])
        self.assertEqual(t.row_key.tolist()[2:6], [5, 3, 7, -1])
        _assert_bijection(self, t)
        missed, _ = ep.step_reference(t, torch.tensor([5, 3, 7, 0], dtype=torch.int32), b)
        self.assertEqual(missed, [])

    def test_limit_and_owned_rows_are_respected(self):
        t, b = _pool()
        ep.step_reference(t, torch.tensor([2, 3], dtype=torch.int32), b)  # rows 2,3 owned
        pairs = ep.seed_lru_rows(t, [3, 8, 9, 4], limit=1)
        self.assertEqual(pairs, [(HOST[8], 4)])  # 3 is owned already, row 4 is the first free
        self.assertEqual(ep.seed_lru_rows(t, [9, 4, 6]), [(HOST[9], 5)])  # one free row left
        self.assertEqual(ep.seed_lru_rows(t, [6]), [])
        _assert_bijection(self, t)


class _FakeLayer:
    def __init__(self, layer_id, local_of_global=None, num_experts=E):
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.num_local_experts = num_experts
        self._map = local_of_global

    def pool_prefetch_local_ids(self, g):
        if self._map is None:
            return g
        return torch.tensor([self._map.get(int(x), -1) for x in g.tolist()])


def _fake_cache(layer):
    c = eo.MoEExpertOffloadCache.__new__(eo.MoEExpertOffloadCache)
    c.layer = layer
    c.num_local_experts = E
    c._pool_ready = True
    c._pool_tables, _ = _pool()
    width = 6
    src = torch.arange((E - 2) * width, dtype=torch.float32).reshape(E - 2, width)
    c._pool_srcs = [src]
    c._pool_dsts = [torch.zeros(ROWS, width)]
    return c


class TestWarmLruFromRoute(CustomTestCase):
    def test_global_route_is_translated_and_the_bytes_land_in_the_seeded_rows(self):
        # this rank owns global 20..29 as local 0..9
        layer = _FakeLayer(3, {20 + i: i for i in range(E)})
        c = _fake_cache(layer)
        n = c.warm_lru_from_route([(24, 9), (41, 7), (26, 3), (20, 2)], limit=0)
        self.assertEqual(n, 2)  # 41 is foreign, 20 -> local 0 is resident
        key = c._pool_tables.row_key.tolist()
        self.assertEqual(key[2:4], [4, 6])
        src, dst = c._pool_srcs[0], c._pool_dsts[0]
        self.assertTrue(torch.equal(dst[2], src[HOST[4]]))
        self.assertTrue(torch.equal(dst[3], src[HOST[6]]))
        self.assertEqual(float(dst[4].abs().sum()), 0.0)

    def test_after_wake_every_pool_layer_is_warmed_and_the_line_is_logged(self):
        model = torch.nn.Module()
        mods = []
        for lid in (0, 1, 2):
            m = torch.nn.Module()
            m.layer_id = lid
            m._expert_offload = _fake_cache(_FakeLayer(lid))
            model.add_module(f"l{lid}", m)
            mods.append(m)
        mods[2]._expert_offload._pool_ready = False
        routes = {0: [(2, 5), (3, 1)], 1: [(4, 1)], 2: [(5, 1)]}
        with self.assertLogs("sglang.srt.layers.moe.expert_offload", "INFO") as cm:
            rows, layers = eo.warm_lru_after_wake(model, limit=1, routes=routes)
        self.assertEqual((rows, layers), (2, 2))
        line = [x for x in cm.output if "LRU-WARM" in x][0]
        self.assertIn("rows_filled=2 layers=2", line)
        self.assertIn("pool_layers=2 route_layers=3", line)

    def test_switches_off_touch_nothing(self):
        with envs.SGLANG_WEG2_LRU_WARM_FROM_HANDOFF.override(False), \
                envs.SGLANG_WEG2_PLE_DECODE_PREFETCH.override(False), \
                mock.patch.object(eo, "warm_lru_after_wake") as w:
            eo._warm_after_wake(torch.nn.Module())
            w.assert_not_called()


class TestRouteNoteOnP(CustomTestCase):
    def test_armed_only_on_p_with_every_expert_and_notes_reach_the_file(self):
        rec = dwh.RouteRecorder()
        with mock.patch.object(eo, "_ROUTE_RECORDER", rec), \
                envs.SGLANG_WEG2_LRU_WARM_FROM_HANDOFF.override(True), \
                tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"SGLANG_HICACHE_ARENA_DIR": d,
                                             "SGLANG_WEG2_GROUP": "P"}):
            self.assertFalse(eo._route_note_armed(_FakeLayer(1), E - 1))  # a shard
            self.assertTrue(eo._route_note_armed(_FakeLayer(1), E))
            c = _fake_cache(_FakeLayer(1))
            c._note_route([[2, 3], [3, 4]])
            routes, _ = dwh.load_routes(os.path.join(d, "handoff"), 600)
            self.assertEqual(routes, {1: [(3, 2), (2, 1), (4, 1)]})
            with mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "D"}):
                self.assertFalse(eo._route_note_armed(_FakeLayer(2), E))
        with envs.SGLANG_WEG2_LRU_WARM_FROM_HANDOFF.override(False):
            self.assertFalse(eo._route_note_armed(_FakeLayer(1), E))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
