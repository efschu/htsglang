"""fnFL2 H20b: expert-major prefill planned on a numpy array.

The list route of ``MoEExpertOffloadCache.run_waves`` turns the [T, K] routed
ids into a nested Python list (``tolist``), plans the waves element by element
(``plan_expert_waves``) and turns the list back into an array
(``np.asarray``). At T=16384 that is ~26 ms of host Python per MoE layer after
the device rendezvous -- idle GPU time inside every PP0 prefill forward. The
vector route must hand ``_run_waves_expert_major`` the SAME waves and the SAME
flat pair array, or the forward stops being byte-identical. The cases pin:

* ``plan_expert_waves_np`` == ``plan_expert_waves`` element for element
  (padding, prefix residency, scattered residency, scratch edges);
* the vector route's arguments into the expert-major pass equal the list
  route's, and a single-wave forward re-enters the list route with the list;
* the route names why it is shut (router stats, hot calibration, heat
  window, token-major order, decode size, switch off) and the NaN guard is
  NOT among them (H20c, x136: the guard shut the route on the Bestform P);
* 'MOE-PLAN-ROUTE' is logged on change only.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.layers.moe import expert_offload as eo
from sglang.test.test_utils import CustomTestCase

E, K = 512, 10


def _ids(T: int, seed: int, pad_every: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = np.argsort(rng.random((T, E)), axis=1)[:, :K].astype(np.int32)
    if pad_every:
        ids[::pad_every, -1] = -1
    return ids


class TestPlanExpertWavesNp(CustomTestCase):
    def test_equals_list_planner(self):
        scattered = frozenset(int(e) for e in np.random.default_rng(7).choice(E, 133, replace=False))
        cases = [
            (4096, 0, 133, 28, None),
            (4096, 3, 133, 28, None),
            (2048, 5, 0, 512, None),
            (1024, 0, 512, 1, None),
            (4096, 2, 133, 28, scattered),
            (512, 0, 133, 1, scattered),
            (8, 1, 133, 7, frozenset()),
        ]
        for T, pad, R, scr, res_ids in cases:
            ids = _ids(T, seed=T + R + scr, pad_every=pad)
            want = eo.plan_expert_waves(ids.tolist(), R, scr, res_ids)
            got = eo.plan_expert_waves_np(ids, R, scr, res_ids, num_experts=E)
            self.assertEqual(got, want, (T, pad, R, scr, res_ids is None))

    def test_all_padding(self):
        ids = np.full((16, K), -1, dtype=np.int32)
        self.assertEqual(eo.plan_expert_waves_np(ids, 4, 2), ([], []))


class _Stats:
    def __init__(self):
        self.overflow_forwards = 0
        self.lookahead_dropped = 0


def _bare_cache(*, resident_ids=None):
    c = eo.MoEExpertOffloadCache.__new__(eo.MoEExpertOffloadCache)
    c.layer = SimpleNamespace(layer_id=3)
    c.resident_count = 133
    c.scratch = 28
    c.num_local_experts = E
    c.planner = SimpleNamespace(resident_ids=resident_ids, stats=_Stats())
    c._plan_vector = True
    c._wave_order = "expert"
    c._router_stats = None
    c._hot_enabled = False
    c._hot_frozen = False
    c._heat = None
    c._nan_trace = None
    return c


def _dispatch(ids: np.ndarray):
    return SimpleNamespace(topk_output=SimpleNamespace(topk_ids=torch.from_numpy(ids)))


class TestVectorRoute(CustomTestCase):
    def test_multi_wave_hands_the_list_route_arguments(self):
        c = _bare_cache()
        seen = {}
        c._run_waves_expert_major = lambda d, f, flat, ru, sw: seen.update(flat=flat, ru=ru, sw=sw)
        ids = _ids(4096, seed=11, pad_every=4)
        c._run_waves_vector(_dispatch(ids), apply_fn=None, lookahead=None)
        ru, sw = eo.plan_expert_waves(ids.tolist(), 133, 28, None)
        want_flat = np.asarray(ids.tolist(), dtype=np.int64).reshape(-1)
        self.assertEqual(seen["flat"].dtype, np.int64)
        np.testing.assert_array_equal(seen["flat"], want_flat)
        self.assertEqual((seen["ru"], seen["sw"]), (ru, sw))
        self.assertEqual(c.planner.stats.overflow_forwards, 1)

    def test_single_wave_reenters_the_list_route(self):
        c = _bare_cache()
        c.scratch = 512
        seen = {}
        c._run_single_wave = lambda d, f, ids_list, prefetch: seen.update(ids_list=ids_list, prefetch=prefetch)
        ids = _ids(512, seed=3)
        c._run_waves_vector(_dispatch(ids), apply_fn=None, lookahead=None)
        self.assertEqual(seen["ids_list"], ids.tolist())
        self.assertIsNone(seen["prefetch"])

    def test_route_names_every_list_reader(self):
        big = torch.zeros((4096, K), dtype=torch.int32)
        self.assertIsNone(_bare_cache()._vector_plan_closed_by(big, None))
        shut = {
            "switch_off": dict(_plan_vector=False),
            "token_order": dict(_wave_order="token"),
            "router_stats": dict(_router_stats=object()),
            "hot_calibration": dict(_hot_enabled=True, _hot_frozen=False),
            "heat_window": dict(_heat=object()),
        }
        for why, attrs in shut.items():
            c = _bare_cache()
            for k, v in attrs.items():
                setattr(c, k, v)
            self.assertEqual(c._vector_plan_closed_by(big, None), why)
        self.assertEqual(_bare_cache()._vector_plan_closed_by(big, "token"), "token_order")
        small = torch.zeros((4, K), dtype=torch.int32)
        self.assertEqual(_bare_cache()._vector_plan_closed_by(small, None), "decode_size")
        frozen = _bare_cache()
        frozen._hot_enabled, frozen._hot_frozen = True, True
        self.assertIsNone(frozen._vector_plan_closed_by(big, None), "frozen hot set reads no list")


class TestNanGuardKeepsTheVectorRoute(CustomTestCase):
    """x136 (24.09.): SGLANG_MOE_OFFLOAD_PLAN_VECTOR=1 on P, and PP0's
    moe_plan_ms stayed at 1078-1696 ms per 16k chunk (x135 without the switch:
    1044-1669). The Bestform P env carries SGLANG_NAN_GUARD=1, and the H20b
    eligibility shut the route whenever the guard was on -- the NaN trace
    held the Python list. The trace reads rows by index, which a [T, K]
    array answers the same way; the guard must not close the route."""

    def setUp(self):
        from sglang.srt.layers import nan_guard

        self._saved = dict(nan_guard._STATE)
        nan_guard._STATE["on"] = True
        self.addCleanup(lambda: nan_guard._STATE.update(self._saved))

    def test_guard_on_route_open_and_trace_is_the_array(self):
        c = _bare_cache()
        big = torch.zeros((4096, K), dtype=torch.int32)
        self.assertIsNone(c._vector_plan_closed_by(big, None))
        c._run_waves_expert_major = lambda *a: None
        ids = _ids(4096, seed=5, pad_every=3)
        c._run_waves_vector(_dispatch(ids), apply_fn=None, lookahead=None)
        rows = c._nan_trace["ids_list"]
        self.assertEqual(rows.shape, (4096, K))
        np.testing.assert_array_equal(rows, ids.astype(np.int64))

    def test_disc2_reads_the_array_like_the_list(self):
        from sglang.srt.layers.moe import nan_disc2

        ids = _ids(256, seed=9, pad_every=5)
        arr = ids.astype(np.int64)
        bad, good = [3, 17, 200], [0, 1, 2, 250]
        self.assertEqual(
            nan_disc2.row_expert_sets(arr, bad), nan_disc2.row_expert_sets(ids.tolist(), bad)
        )
        waves = [{"wave": 0, "needed": [int(e) for e in ids[3][:2]]}, {"wave": 1, "needed": [511]}]
        self.assertEqual(
            nan_disc2.waves_of_rows(waves, arr, bad), nan_disc2.waves_of_rows(waves, ids.tolist(), bad)
        )
        cache = SimpleNamespace(
            _nan_trace={"ids_list": arr, "waves": waves, "partials": "table"},
            _scratch_holds={}, resident_count=0, _pinned={},
        )
        snap = nan_disc2.snapshot(None, cache, bad, good)  # raised on `array or []`
        self.assertEqual(snap["bad_rows"], bad)


class TestPlanRouteLine(CustomTestCase):
    def setUp(self):
        self._saved = dict(eo._PLAN_ROUTE_LAST)
        eo._PLAN_ROUTE_LAST["state"] = None
        self.addCleanup(lambda: eo._PLAN_ROUTE_LAST.update(self._saved))

    def test_logged_on_change_only_and_decode_never_moves_it(self):
        with self.assertLogs(eo.logger, level="INFO") as cap:
            self.assertTrue(eo.note_plan_route("switch_off", layer_id=0, pairs=163840))
            self.assertFalse(eo.note_plan_route("switch_off", layer_id=1, pairs=163840))
            self.assertFalse(eo.note_plan_route("decode_size", layer_id=0, pairs=10))
            self.assertTrue(eo.note_plan_route(None, layer_id=0, pairs=163840))
            self.assertFalse(eo.note_plan_route("decode_size", layer_id=0, pairs=10))
            self.assertFalse(eo.note_plan_route(None, layer_id=5, pairs=163840))
        self.assertEqual(len(cap.output), 2)
        self.assertIn("MOE-PLAN-ROUTE mode=list why=switch_off", cap.output[0])
        self.assertIn("MOE-PLAN-ROUTE mode=vector why=open", cap.output[1])


if __name__ == "__main__":
    unittest.main()
