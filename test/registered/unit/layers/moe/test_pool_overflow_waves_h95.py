"""H95 A: the device-planned expert pool without the worst-case bound.

THE BOUND. A captured pool step is exact whenever its DISTINCT NON-RESIDENT
ids fit the rows above the residents, ``D_nr <= C`` (C = LRU + staging). Task
#40 / H91b booked the worst case at capture, ``min(bs x 4 x top-10, E - R) <=
C``: Form A D bs2 needed scratch 80 on the 3080 workers (residency 0.29/0.30
instead of 0.51/0.48), bs6 240 ids -- not representable.

WHAT MUST HOLD.
(1) SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES=N >= 2: the capture bound is
    ``min(ids, E - R) <= N x C``; wave 1 spills what it cannot hold instead of
    setting the sticky error, the next wave serves exactly those lanes; every
    (token, k) lane is computed in exactly one wave, from a row that holds its
    expert at that wave's GEMM; the summed output equals the true MoE output,
    and a step that fits one wave is bit-identical to the single step.
(2) The switch off (default) is the step before H95: no waves, no demand
    counters, the same refusal.
(3) The demand probe counts the per-step MAXIMUM of D_nr and the steps over C.
(4) The Triton kernel equals the reference in every mode (TRITON_INTERPRET=1).
(5) The planner prices the bound with the waves: the x177 worker scratch 48
    carries seats 1..6 with two waves.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import random
import subprocess
import sys
import types
import unittest
from typing import NamedTuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_pool_device as ep
from sglang.test.test_utils import CustomTestCase


def _tables(E, R, C, S, width=256, demand=False, pad=None):
    """Residents = experts 0..R-1 in rows 0..R-1 (plus ``pad`` in row R-1 when
    given), every other expert in the host store row ``e``."""
    hot = {e: e for e in range(R)}
    if pad is not None:
        hot = {e: e for e in range(R - 1)}
        hot[pad] = R - 1
    host_row = [(-1 if e in hot else e) for e in range(E)]
    kw = {"demand": True} if demand else {}
    t = ep.allocate_pool_tables("cpu", E, R + C, R, S, hot, host_row, **kw)
    b = ep.allocate_step_buffers("cpu", E, width)
    return t, b, hot


def _bank(t, hot, value):
    """A one-number-per-row bank and its host store: row r holds value[e]."""
    rows = int(t.row_key.shape[0])
    bank = torch.full((rows, 1), float("nan"))
    for e, r in hot.items():
        bank[r, 0] = value[e]
    host = torch.tensor([[value[e]] for e in range(t.num_experts)])
    return bank, host


def _distinct_nonres(ids, R, pad):
    """Residents are 0..R-2 plus the pad (row R-1), see ``_tables``."""
    return len({int(v) for v in ids if int(v) >= R - 1 and int(v) != pad})


class TestWaveBound(CustomTestCase):
    def test_waves_needed_is_the_worst_case_over_c(self):
        self.assertEqual(ep.pool_waves_for(40, 145, 74, 48), 1)
        self.assertEqual(ep.pool_waves_for(80, 145, 74, 48), 2)  # min(80, 71) = 71
        self.assertEqual(ep.pool_waves_for(240, 177, 85, 48), 2)  # min(240, 92) = 92
        self.assertEqual(ep.pool_waves_for(240, 193, 12, 118), 2)  # min(240, 181)
        self.assertEqual(ep.pool_waves_for(80, 40, 10, 30), 1)

    def test_two_waves_accept_what_one_refuses(self):
        t, b, _ = _tables(E=64, R=8, C=10, S=4)
        ids = torch.arange(8, 48, dtype=torch.int32)  # 40 distinct non-residents
        with self.assertRaisesRegex(ValueError, "LRU rows plus the staging rows"):
            ep.step_reference(t, ids, b)
        with self.assertRaisesRegex(ValueError, "LRU rows plus the staging rows"):
            ep.step_reference(t, ids, b, spill=True, waves=3)  # 40 > 3 x 10
        ep.step_reference(t, ids, b, spill=True, waves=4)
        self.assertEqual(int(t.error[0]), 0)

    def test_without_spill_an_overflow_is_still_the_sticky_error(self):
        t, b, _ = _tables(E=64, R=8, C=10, S=4)
        ids = torch.arange(8, 48, dtype=torch.int32)
        ep.step_reference(t, ids, b, waves=4)  # last wave semantics: no spill
        self.assertEqual(int(t.error[0]), 1)


class TestWavesServeEveryLaneOnce(CustomTestCase):
    def _run_waves(self, t, b, bank, host, ids, waves):
        """The wave schedule of MoEExpertOffloadCache.run_pool_waves on the
        reference step: returns per lane (wave, row value at that wave)."""
        n = ids.numel()
        got = {}
        cur = ids.clone()
        for w in range(1, waves + 1):
            last = w == waves
            ep.step_reference(t, cur, b, spill=not last, wave=w > 1, waves=waves)
            n_pairs = int(b.gather_count[0])
            ep.copy_rows(
                [host], [bank], b.gather_src[:n_pairs], b.gather_dst[:n_pairs],
                torch.tensor([n_pairs], dtype=torch.int32))
            r = b.routes[:n]
            for i in range(n):
                if int(cur[i]) >= 0 and int(r[i]) >= 0:
                    self.assertNotIn(i, got, "lane served twice")
                    got[i] = (w, float(bank[int(r[i]), 0]))
            cur = torch.where((cur >= 0) & (r < 0), cur, torch.full_like(cur, -1))
        self.assertEqual(int(t.error[0]), 0)
        return got

    def test_bs6_on_a_worker_with_bs1_scratch(self):
        # x177 TP2 in miniature: E 177 local, R 85 residents, scratch 48
        # (36 LRU + 12 staging); bs6 x 4 verify x top-10 = 240 ids.
        E, R, C, S = 177, 85, 48, 12
        t, b, hot = _tables(E, R, C, S)
        value = [float(e + 1) for e in range(E)]
        bank, host = _bank(t, hot, value)
        rng = random.Random(3)
        for step_i in range(6):
            ids = torch.tensor([rng.randrange(E) for _ in range(240)], dtype=torch.int32)
            waves = ep.pool_waves_for(240, E, R, C)
            self.assertEqual(waves, 2)
            got = self._run_waves(t, b, bank, host, ids, waves)
            self.assertEqual(sorted(got), list(range(240)))  # every lane once
            for i, (_w, v) in got.items():
                self.assertEqual(v, value[int(ids[i])])  # from its own expert

    def test_a_step_that_fits_never_reaches_wave_two(self):
        E, R, C, S = 64, 8, 20, 4
        t, b, hot = _tables(E, R, C, S)
        value = [float(e + 1) for e in range(E)]
        bank, host = _bank(t, hot, value)
        ids = torch.tensor([8 + (i % 12) for i in range(80)], dtype=torch.int32)  # 12 <= C
        got = self._run_waves(t, b, bank, host, ids, 3)  # min(80, 56) <= 3 x 20
        self.assertEqual({w for w, _v in got.values()}, {1})


class _Topk(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: object = None


class _Disp(NamedTuple):
    hidden_states: torch.Tensor
    topk_output: _Topk
    hidden_states_scale: object = None


class _Comb(NamedTuple):
    hidden_states: torch.Tensor


class TestRunPoolWaves(CustomTestCase):
    """``MoEExpertOffloadCache.run_pool_waves`` end to end on the CPU: the
    reference step, the reference row copy, and a toy apply that reads the
    bank the way the Marlin MoE does -- by route, weighted, summed over k."""

    def _cache(self, E, R, C, S, pad):
        from sglang.srt.layers.moe.expert_offload import MoEExpertOffloadCache

        t, b, hot = _tables(E, R, C, S, width=256, pad=pad)
        value = [float((e * 7) % 13 + 1) for e in range(E)]
        value[pad] = 0.0  # the expert-shard pad: all-zero weights
        bank, host = _bank(t, hot, value)
        cache = object.__new__(MoEExpertOffloadCache)
        cache._pool_ready = True
        cache._pool_tables, cache._pool_buffers = t, b
        cache._pool_srcs, cache._pool_dsts = [host], [bank]
        cache._pool_pf_armed = False
        cache._pool_waves_seen = {}
        cache.num_local_experts = E
        cache.layer = types.SimpleNamespace(
            layer_id=5, _gguf_expert_shard=True, _gguf_expert_range=(100, 100 + pad),
            _expert_shard_generic=False)
        host_row = [int(x) for x in t.host_row.tolist()]
        cache._pool_layout = lambda: (hot, host_row)
        return cache, bank, value

    def _apply(self, bank):
        def apply(disp):
            rows = disp.topk_output.topk_ids.long()
            w = disp.topk_output.topk_weights
            x = disp.hidden_states
            per = bank[rows.clamp(min=0), 0] * w  # [T, k]
            return _Comb(hidden_states=x * per.sum(dim=1, keepdim=True))
        return apply

    def _disp(self, ids, T, k, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.rand(T, 3, generator=g) + 0.5
        w = torch.rand(T, k, generator=g)
        return _Disp(hidden_states=x, topk_output=_Topk(topk_weights=w, topk_ids=ids.view(T, k)))

    def test_overflow_output_equals_the_true_moe(self):
        E, R, C, S, pad = 65, 9, 12, 4, 64
        cache, bank, value = self._cache(E, R, C, S, pad)
        T, k = 24, 10  # bs6 x 4 verify rows, top-10
        rng = random.Random(11)
        for seed in range(4):
            ids = torch.tensor([rng.randrange(E - 1) for _ in range(T * k)], dtype=torch.int64)
            self.assertGreater(_distinct_nonres(ids.tolist(), R, pad), C)  # overflow
            disp = self._disp(ids, T, k, seed)
            waves = ep.pool_waves_for(T * k, E, R, C)
            out = cache.run_pool_waves(disp, self._apply(bank), waves)
            true = torch.tensor([[value[int(e)] for e in row] for row in ids.view(T, k)])
            want = disp.hidden_states * (true * disp.topk_output.topk_weights).sum(1, keepdim=True)
            torch.testing.assert_close(out.hidden_states, want, rtol=1e-6, atol=1e-6)
            self.assertEqual(int(cache._pool_tables.error[0]), 0)

    def test_a_fitting_step_is_bit_identical_to_the_single_step(self):
        E, R, C, S, pad = 65, 9, 30, 4, 64
        cache, bank, _value = self._cache(E, R, C, S, pad)
        twin, bank2, _ = self._cache(E, R, C, S, pad)
        T, k = 3, 10  # 30 ids: the single step's own bound holds
        ids = torch.tensor([9 + (i % 20) for i in range(T * k)], dtype=torch.int64)
        disp = self._disp(ids, T, k, 0)
        out = cache.run_pool_waves(disp, self._apply(bank), 3)
        routes = twin.prepare_pool(disp.topk_output.topk_ids)
        one = self._apply(bank2)(disp._replace(topk_output=disp.topk_output._replace(topk_ids=routes)))
        self.assertTrue(torch.equal(out.hidden_states, one.hidden_states))

    def test_masked_lanes_point_at_the_zero_pad_row(self):
        E, R, C, S, pad = 65, 9, 12, 4, 64
        cache, _bank, _v = self._cache(E, R, C, S, pad)
        self.assertEqual(int(cache._pool_zero_row()[0]), R - 1)  # pad's row


class TestSwitch(CustomTestCase):
    def test_default_is_off(self):
        self.assertEqual(envs.SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES.get(), 0)
        self.assertEqual(envs.SGLANG_DEBUG_MOE_POOL_DEMAND.get(), 0)

    def test_pool_waves_off_is_one_and_on_is_the_ceiling(self):
        from sglang.srt.layers.moe.expert_offload import MoEExpertOffloadCache

        t, b, _ = _tables(E=177, R=85, C=48, S=12)
        cache = object.__new__(MoEExpertOffloadCache)
        cache._pool_ready, cache._pool_tables, cache._pool_buffers = True, t, b
        cache._pool_waves_seen = {}
        cache.layer = types.SimpleNamespace(layer_id=0)
        self.assertEqual(cache.pool_waves(240), 1)
        with envs.SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES.override(2):
            self.assertEqual(cache.pool_waves(240), 2)
            self.assertEqual(cache.pool_waves(40), 1)  # bs1 captures one wave
        with envs.SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES.override(1):
            self.assertEqual(cache.pool_waves(240), 1)

    def test_off_tables_carry_no_demand_counters(self):
        t, _b, _ = _tables(E=20, R=4, C=6, S=2)
        self.assertIsNone(t.demand)
        self.assertIsNone(ep.take_demand_report(t))


class TestDemandProbe(CustomTestCase):
    def test_max_nonresident_per_step_and_steps_over_c(self):
        t, b, _ = _tables(E=64, R=8, C=10, S=4, demand=True)
        steps = [
            [0, 1, 2, 8, 9, 9, 10],  # 3 non-resident
            list(range(8, 20)) + [0, 0],  # 12 > C
            [8, 9, 3],  # 2 (8, 9 now LRU hits: still non-resident)
        ]
        for s in steps:
            ep.step_reference(t, torch.tensor(s, dtype=torch.int32), b, spill=True, waves=2)
        self.assertEqual(ep.take_demand_report(t), (12, 1, 3))
        self.assertEqual(ep.take_demand_report(t), (0, 0, 0))  # reset

    def test_later_waves_and_prefetch_are_not_steps(self):
        t, b, _ = _tables(E=64, R=8, C=10, S=4, demand=True)
        pf = ep.allocate_step_buffers("cpu", 64, 256)
        ep.step_reference(t, torch.tensor([8, 9], dtype=torch.int32), pf, prefetch=True)
        ep.step_reference(t, torch.tensor([10, 11], dtype=torch.int32), b, wave=True)
        self.assertEqual(ep.take_demand_report(t), (0, 0, 0))

    def test_the_report_line(self):
        from sglang.srt.layers.moe import pool_demand_probe as p

        line = p.demand_line([(0, 20, 0, 64, 48, 2), (23, 51, 3, 64, 48, 2),
                              (47, 30, 0, 64, 48, 2)], replays=64)
        self.assertIn("MOE-POOL-DEMAND (H95)", line)
        self.assertIn("max_nonres_per_step=51 at layer 23 (C=48, 1.06 of C)", line)
        self.assertIn("over_C_steps=3 of 192", line)
        self.assertIn("waves=[2]", line)
        self.assertIn("steps=0", p.demand_line([(0, 0, 0, 0, 48, None)], replays=8))


_INTERP = r'''
import os, random, sys
os.environ["TRITON_INTERPRET"] = "1"
import torch
from sglang.srt.layers.moe import expert_pool_device as ep

def snap(t, b):
    out = {}
    for f in ("hot_phys", "row_key", "row_use", "clock", "error", "forwards", "miss_count",
              "misses_total", "pf_row", "pf_counts", "demand"):
        v = getattr(t, f)
        if v is not None:
            out[f] = v.tolist()
    n = int(b.gather_count[0])
    out["pairs"] = sorted(zip(b.gather_src[:n].tolist(), b.gather_dst[:n].tolist()))
    for f in ("routes", "staged_count", "promoted_count", "step_map"):
        out[f] = getattr(b, f).tolist()
    return out

def make(E, R, C, S):
    hot = {e: e for e in range(R)}
    host = [-1] * R + list(range(E - R))
    return (ep.allocate_pool_tables("cpu", E, R + C, R, S, hot, host, demand=True),
            ep.allocate_step_buffers("cpu", E, 64))

rng = random.Random(5)
bad = compared = 0
for case in range(int(sys.argv[1])):
    E = rng.choice([24, 40, 64]); R = rng.randint(1, E // 2)
    C = rng.randint(3, min(20, E - R)); S = rng.randint(1, C - 1)
    for mode in ({}, {"spill": True}, {"wave": True}, {"wave": True, "spill": True}):
        ta, ba = make(E, R, C, S); tb, bb = make(E, R, C, S)
        n = rng.randint(1, 64) if mode.get("spill") else rng.randint(1, C)
        for _ in range(3):
            ids = torch.tensor([rng.randint(-1, E - 1) for _ in range(n)], dtype=torch.int32)
            ep.step_reference(ta, ids, ba, waves=64 if mode.get("spill") else 1, **mode)
            ep._launch_step_kernel(tb, ids.clone(), bb, **mode)
            compared += 1
            if snap(ta, ba) != snap(tb, bb):
                bad += 1
                print("DIFF", case, mode)
                break
print("compared", compared, "mismatches", bad)
sys.exit(1 if bad or not compared else 0)
'''


class TestKernelEqualsReference(CustomTestCase):
    def test_triton_interpreter_matches_the_reference_in_every_mode(self):
        env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
        r = subprocess.run([sys.executable, "-c", _INTERP, "8"], env=env,
                           capture_output=True, text=True, timeout=900)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        self.assertIn("mismatches 0", r.stdout)


class TestPlannerWaves(CustomTestCase):
    def _fit(self, rank, E, R, S):
        return types.SimpleNamespace(rank=rank, local_experts=E, resident_rows=R, scratch_rows=S)

    def test_x177_scratch_carries_seats_one_to_six_with_two_waves(self):
        from sglang.srt.planner import expert_residency as er

        fits = [self._fit(0, 193, 12, 118), self._fit(1, 145, 74, 48), self._fit(2, 177, 85, 48)]
        for seats in range(1, 7):
            lines, refusal = er.pool_step_rows_check(
                fits, seats=seats, verify_tokens=4, top_k=10, pool_mode=True,
                marker="M", label="D", waves=2)
            self.assertIsNone(refusal, (seats, lines))
        self.assertIn("Wellen je Rang [2, 2, 2]", lines[0])
        # without waves bs2 is refused exactly as H91b
        _l, refusal = er.pool_step_rows_check(
            fits, seats=2, verify_tokens=4, top_k=10, pool_mode=True, marker="M", label="D")
        self.assertIsNotNone(refusal)
        # a wave cap too small is refused by name
        _l, refusal = er.pool_step_rows_check(
            [self._fit(0, 193, 12, 40)], seats=6, verify_tokens=4, top_k=10,
            pool_mode=True, marker="M", label="D", waves=2)
        self.assertIn("SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES", refusal)

    def test_the_wave_cap_is_read_off_group_d_env(self):
        from sglang.srt.planner import expert_residency as er

        self.assertEqual(er.pool_overflow_waves({}), 1)
        self.assertEqual(er.pool_overflow_waves({"SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES": "0"}), 1)
        self.assertEqual(er.pool_overflow_waves({"SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES": "2"}), 2)


if __name__ == "__main__":
    unittest.main()
