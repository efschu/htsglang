"""#276 heat record of the device-planned expert pool.

WHAT MUST HOLD.
(1) Off (SGLANG_DEBUG_MOE_HEAT unset, the default): nothing is allocated and
    the captured step is the step before #276 -- the op sequence of
    ``prepare_pool`` is identical to that of a cache that has no heat
    attribute at all, and no op touches a heat tensor.
(2) On: the histogram counts, per local expert id, exactly the routed lanes
    of every step (lanes outside ``[0, E)`` in the not-local slot, one step
    count per call); the step's routes and tables are identical to off.
(3) Overflow waves (H95) count every lane once: wave 1 sees all of them.
(4) ``flush`` writes one JSON record per rank with the layer's global id
    window, zeroes the counters; nothing is written when off or without a
    step. ``reset`` zeroes.
(5) The D sleep flushes before any pause, the D wake resets after the rearm.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import inspect
import json
import random
import tempfile
import types
import unittest
from typing import NamedTuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_pool_device as ep
from sglang.srt.layers.moe import pool_heat
from sglang.test.test_utils import CustomTestCase


class _Topk(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


class _Disp(NamedTuple):
    hidden_states: torch.Tensor
    topk_output: _Topk


class _Comb(NamedTuple):
    hidden_states: torch.Tensor


def _cache(E=20, R=4, C=16, S=3, width=64, heat=False, attr=True, pad=None, lo=None):
    """A CPU pool layer: residents 0..R-1 (plus ``pad`` in row R-1), the rest
    in host row ``e``; the reference step and row copy run on the CPU."""
    from sglang.srt.layers.moe.expert_offload import MoEExpertOffloadCache

    hot = {e: e for e in range(R)}
    if pad is not None:
        hot = {e: e for e in range(R - 1)}
        hot[pad] = R - 1
    host_row = [(-1 if e in hot else e) for e in range(E)]
    t = ep.allocate_pool_tables("cpu", E, R + C, R, S, hot, host_row)
    b = ep.allocate_step_buffers("cpu", E, width)
    rows = int(t.row_key.shape[0])
    bank = torch.zeros(rows, 1)
    for e, r in hot.items():
        bank[r, 0] = float(e + 1)
    host = torch.tensor([[float(e + 1)] for e in range(E)])
    cache = object.__new__(MoEExpertOffloadCache)
    cache._pool_ready = True
    cache._pool_tables, cache._pool_buffers = t, b
    cache._pool_srcs, cache._pool_dsts = [host], [bank]
    cache._pool_pf_armed = False
    cache._pool_waves_seen = {}
    cache.num_local_experts = E
    cache.resident_count = R
    cache.planner = types.SimpleNamespace(resident_ids=None)
    if pad is not None:
        cache.layer = types.SimpleNamespace(
            layer_id=7, num_experts=512, num_local_experts=E, _gguf_expert_shard=True,
            _gguf_expert_range=(lo, lo + pad), _expert_shard_generic=True)
    else:
        cache.layer = types.SimpleNamespace(layer_id=7, num_experts=E, num_local_experts=E)
    cache._pool_layout = lambda: (hot, host_row)
    if attr:
        cache._pool_heat, cache._pool_heat_ones = None, None
    if heat:
        with envs.SGLANG_DEBUG_MOE_HEAT.override("/nonexistent-but-on"):
            cache._pool_heat, cache._pool_heat_ones = pool_heat.allocate("cpu", E, width)
    return cache


class _OpLog(TorchDispatchMode):
    def __init__(self, watch=()):
        super().__init__()
        self.ops, self.touched = [], 0
        self._watch = [w.data_ptr() for w in watch if w is not None]

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, torch.Tensor) and a.data_ptr() in self._watch:
                self.touched += 1
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def _ids(rows, k=10, E=20, seed=0, holes=True):
    rng = random.Random(seed)
    v = [rng.randrange(E) for _ in range(rows * k)]
    if holes:
        for i in range(0, len(v), 7):
            v[i] = -1  # an expert another rank owns
    return torch.tensor(v, dtype=torch.int32).view(rows, k)


class TestOff(CustomTestCase):
    def test_default_is_off_and_allocates_nothing(self):
        self.assertIsNone(envs.SGLANG_DEBUG_MOE_HEAT.get())
        self.assertIsNone(pool_heat.heat_dir())
        self.assertEqual(pool_heat.allocate("cpu", 20, 64), (None, None))
        with envs.SGLANG_DEBUG_MOE_HEAT.override(""):
            self.assertIsNone(pool_heat.heat_dir())

    def test_off_step_is_the_step_before_276(self):
        """The op sequence of prepare_pool with the heat off equals that of a
        cache that predates #276 (no heat attribute at all), step by step."""
        ids = [_ids(3, seed=s) for s in range(3)]
        base = _cache(attr=False)  # the shape of a pre-#276 cache
        off = _cache()
        self.assertFalse(hasattr(base, "_pool_heat"))
        for i in ids:
            with _OpLog() as a:
                ra = base.prepare_pool(i)
            with _OpLog() as b:
                rb = off.prepare_pool(i)
            self.assertEqual(a.ops, b.ops)
            self.assertTrue(torch.equal(ra, rb))
        self.assertTrue(torch.equal(base._pool_tables.row_key, off._pool_tables.row_key))

    def test_on_adds_only_heat_ops_and_changes_no_route(self):
        off, on = _cache(), _cache(heat=True)
        for s in range(3):
            i = _ids(3, seed=s)
            with _OpLog() as a:
                ra = off.prepare_pool(i)
            with _OpLog(watch=[on._pool_heat]) as b:
                rb = on.prepare_pool(i)
            self.assertTrue(torch.equal(ra, rb))
            self.assertGreater(b.touched, 0)
            self.assertIn("aten.index_add_.default", b.ops)
            self.assertNotIn("aten.index_add_.default", a.ops)
            # b = a with exactly the ops of one pool_heat.count spliced in
            scratch, flat = torch.zeros_like(on._pool_heat), i.reshape(-1)
            with _OpLog() as h:
                pool_heat.count(scratch, on._pool_heat_ones, flat, 20)
            self.assertTrue(any(
                b.ops == a.ops[:p] + h.ops + a.ops[p:] for p in range(len(a.ops) + 1)))
        for name in ("row_key", "row_use", "hot_phys", "miss_count"):
            self.assertTrue(torch.equal(getattr(off._pool_tables, name),
                                        getattr(on._pool_tables, name)), name)


class TestCount(CustomTestCase):
    def test_histogram_equals_the_routed_lanes(self):
        E = 20
        cache = _cache(E=E, heat=True)
        want = [0] * E
        not_local = 0
        steps = [_ids(r, E=E, seed=s) for s, r in enumerate((1, 3, 2, 4))]
        for i in steps:
            cache.prepare_pool(i)
            for v in i.reshape(-1).tolist():
                if 0 <= v < E:
                    want[v] += 1
                else:
                    not_local += 1
        got = cache._pool_heat.tolist()
        self.assertEqual(got[:E], want)
        self.assertEqual(got[E], not_local)
        self.assertEqual(got[E + 1], len(steps))

    def test_overflow_waves_count_every_lane_once(self):
        E, R, C, S, pad = 65, 9, 12, 4, 64
        cache = _cache(E=E, R=R, C=C, S=S, width=256, heat=True, pad=pad, lo=100)
        T, k = 24, 10
        rng = random.Random(3)
        ids = torch.tensor([rng.randrange(E - 1) for _ in range(T * k)], dtype=torch.int64)
        g = torch.Generator().manual_seed(0)
        disp = _Disp(hidden_states=torch.rand(T, 3, generator=g),
                     topk_output=_Topk(topk_weights=torch.rand(T, k, generator=g),
                                       topk_ids=ids.view(T, k)))
        waves = ep.pool_waves_for(T * k, E, R, C)
        self.assertGreaterEqual(waves, 2)
        cache.run_pool_waves(disp, lambda d: _Comb(hidden_states=d.hidden_states), waves)
        got = cache._pool_heat.tolist()
        self.assertEqual(got[:E], torch.bincount(ids, minlength=E).tolist())
        self.assertEqual(got[E], 0)
        self.assertEqual(got[E + 1], 1)


class TestRecord(CustomTestCase):
    def _model(self, caches):
        mods = [types.SimpleNamespace(_expert_offload=c) for c in caches]
        return types.SimpleNamespace(modules=lambda: iter(mods))

    def test_flush_writes_the_record_and_zeroes(self):
        E, pad, lo = 20, 19, 300
        cache = _cache(E=E, heat=True, pad=pad, lo=lo)
        for s in range(5):
            cache.prepare_pool(_ids(2, E=E, seed=s))
        before = cache._pool_heat.tolist()
        with tempfile.TemporaryDirectory() as d, envs.SGLANG_DEBUG_MOE_HEAT.override(d):
            path = pool_heat.flush([self._model([cache])], rank=1, group="D",
                                   reason="sleep", phase_index=12)
            self.assertIsNotNone(path)
            self.assertEqual(os.path.dirname(path), d)
            rec = json.load(open(path))
            self.assertEqual(os.listdir(d), [os.path.basename(path)])  # no .tmp left
        self.assertEqual((rec["kind"], rec["group"], rec["rank"], rec["reason"],
                          rec["phase_index"]), ("moe_heat", "D", 1, "sleep", 12))
        (layer,) = rec["layers"]
        self.assertEqual(layer["layer_id"], 7)
        self.assertEqual(layer["counts"], before[:E])
        self.assertEqual((layer["not_local"], layer["steps"]), (before[E], 5))
        self.assertEqual((layer["global_lo"], layer["pad"]), (lo, True))
        self.assertEqual(layer["resident_local"], [0, 1, 2, 3])
        self.assertEqual(cache._pool_heat.tolist(), [0] * (E + 2))

    def test_nothing_written_when_off_or_without_a_step(self):
        cache = _cache(heat=True)
        with tempfile.TemporaryDirectory() as d:
            with envs.SGLANG_DEBUG_MOE_HEAT.override(d):
                self.assertIsNone(pool_heat.flush([self._model([cache])], rank=0,
                                                  group="D", reason="sleep"))
            cache.prepare_pool(_ids(1))
            self.assertIsNone(pool_heat.flush([self._model([cache])], rank=0,
                                              group="D", reason="sleep"))  # off
            self.assertEqual(os.listdir(d), [])

    def test_reset_zeroes_and_an_off_cache_is_skipped(self):
        on, off = _cache(heat=True), _cache()
        on.prepare_pool(_ids(2))
        self.assertEqual(pool_heat.reset([self._model([on, off]), None]), 1)
        self.assertEqual(sum(on._pool_heat.tolist()), 0)


class TestPhaseBoundaryWiring(CustomTestCase):
    def test_d_sleep_flushes_before_the_pause_and_the_wake_resets(self):
        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu)
        rel = src[src.index("def release_memory_occupation"):]
        self.assertLess(rel.index("_heat.flush("), rel.index("tags = recv_req.tags"))
        self.assertLess(rel.index('_rw().snapshot([_m])'), rel.index("_heat.flush("))
        self.assertIn('group="D", reason="sleep"', rel[: rel.index("tags = recv_req.tags")])
        res = src[src.index("_rw().arm(_early)"):]
        self.assertIn("_heat.reset(list(_early) + list(_late))", res[:600])


if __name__ == "__main__":
    unittest.main()
