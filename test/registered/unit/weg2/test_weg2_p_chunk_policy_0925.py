"""--p-chunk-policy (27B + NF line, 25.09.): the dynamic prefill chunk plan.

Pinned without a GPU:
  * the RULE -- exact flow-shop pricing, the candidate family, the fixed
    baseline winning under the hysteresis, a large per-forward cost (the NF
    form a + b*M) pushing to large chunks;
  * the LIMITS -- sum, ceiling, floor, page, grid (no chunk crosses it),
    graph-bucket hits (a chunk inside the bucket range IS a bucket unless it
    is the final rest), eager=False keeps every chunk inside the buckets,
    a tail hook is applied and a hook that breaks the plan raises;
  * FIXED IS IDENTICAL -- the launcher's default adds no env, no line and
    keeps group P's --chunked-prefill-size; the scheduler without a planner
    takes the old path;
  * the per-request cursor and the scheduler hook.
"""

from __future__ import annotations

import json
import random
import unittest
from types import SimpleNamespace

import pytest

from sglang.srt.weg2 import p_chunk_policy as P
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _lin(a, b, attn=0.0, floor=0.0):
    return P.StageModel.linear(a, b, attn_ms_per_tok_1k=attn, eager_floor_ms=floor)


def _prop(ms512, attn=0.0, floor=0.0):
    return P.StageModel(((512, ms512),), attn, floor)


class TestStageModel(unittest.TestCase):
    def test_linear_and_interpolation(self):
        s = _lin(10.0, 0.1)
        self.assertAlmostEqual(s.base_ms(0), 10.0)
        self.assertAlmostEqual(s.base_ms(512), 10.0 + 51.2)
        pts = P.StageModel(((256, 30.0), (512, 50.0), (1024, 90.0)))
        self.assertAlmostEqual(pts.base_ms(384), 40.0)
        self.assertAlmostEqual(pts.base_ms(2048), 170.0)  # outer segment extended
        self.assertAlmostEqual(pts.base_ms(128), 20.0)
        self.assertAlmostEqual(_prop(40.0).base_ms(1024), 80.0)

    def test_attention_and_eager_floor(self):
        s = P.StageModel(((512, 40.0),), attn_ms_per_tok_1k=0.002, eager_floor_ms=60.0)
        # 512 tokens at prefix 10000: 0.002 * 512 * (10000 + 256) / 1000
        self.assertAlmostEqual(s.forward_ms(512, 512, 10000, False), 40.0 + 0.002 * 512 * 10.256)
        self.assertAlmostEqual(s.forward_ms(512, 512, 0, True), 60.0)  # host-paced
        self.assertAlmostEqual(s.forward_ms(512, 100, 0, False), 40.0 + 0.002 * 100 * 0.05)

    def test_invalid(self):
        for bad in ((), ((512, 1.0), (256, 1.0)), ((0, 1.0),), ((512, -1.0),)):
            with self.assertRaises(P.ChunkPolicyError):
                P.StageModel(bad)

    def test_json_roundtrip(self):
        s = P.StageModel(((256, 30.0), (512, 50.0)), 0.001, 12.0, "PP0")
        self.assertEqual(P.StageModel.from_json(json.loads(json.dumps(s.to_json()))), s)
        nf = P.StageModel.from_json({"a_ms": 1700, "b_ms_per_token": 0.05})
        self.assertAlmostEqual(nf.base_ms(4096), 1700 + 0.05 * 4096)


class TestLimits(unittest.TestCase):
    def test_ladder(self):
        lim = P.ChunkLimits(max_tokens=3000, min_tokens=512, fixed_tokens=512, graph_buckets=(16, 512, 768))
        self.assertEqual(lim.ladder(), (512, 768, 1024, 2048, 3000))
        self.assertEqual(lim.exec_shape(1), (16, False))
        self.assertEqual(lim.exec_shape(600), (768, False))
        self.assertEqual(lim.exec_shape(1024), (1024, True))

    def test_refusals(self):
        with self.assertRaises(P.ChunkPolicyError):
            P.ChunkLimits(1024, 2048, 512)  # min > max
        with self.assertRaises(P.ChunkPolicyError):
            P.ChunkLimits(1024, 512, 4096)  # fixed > max
        with self.assertRaises(P.ChunkPolicyError):
            P.ChunkLimits(8192, 1024, 4096, page=64, grid=4096)  # a chunk could cross the grid
        with self.assertRaises(P.ChunkPolicyError):
            P.ChunkLimits(1000, 512, 512, page=64)  # max not page-aligned
        with self.assertRaises(P.ChunkPolicyError):
            P.ChunkLimits(2048, 512, 512, eager=False)  # no buckets
        with self.assertRaises(P.ChunkPolicyError):
            P.ChunkLimits(2048, 512, 512, graph_buckets=(512,), eager=False)


class TestMakespan(unittest.TestCase):
    def test_one_stage_is_the_sum(self):
        lim = P.ChunkLimits(4096, 512, 512)
        st = [_lin(5.0, 0.01)]
        self.assertAlmostEqual(P.makespan_ms([512, 512, 100], 0, st, lim),
                               3 * 5.0 + 0.01 * 1124)

    def test_balanced_pipeline_is_n_plus_s_minus_1(self):
        lim = P.ChunkLimits(4096, 512, 512)
        st = [_prop(40.0)] * 3
        self.assertAlmostEqual(P.makespan_ms([512] * 16, 0, st, lim), (16 + 2) * 40.0)

    def test_inflight_cap_binds(self):
        st = [_prop(10.0), _prop(10.0), _prop(10.0)]
        free = P.makespan_ms([512] * 6, 0, st, P.ChunkLimits(4096, 512, 512, max_inflight=3))
        tight = P.makespan_ms([512] * 6, 0, st, P.ChunkLimits(4096, 512, 512, max_inflight=1))
        self.assertAlmostEqual(free, 80.0)
        self.assertAlmostEqual(tight, 6 * 30.0)

    def test_bubble_share(self):
        lim = P.ChunkLimits(4096, 512, 512)
        st = [_prop(40.0)] * 3
        self.assertAlmostEqual(P.bubble_share([512] * 16, 0, st, lim), 2.0 / 18.0)


def _check_limits(tc, chunks, n, lim, start=0):
    tc.assertEqual(sum(chunks), n)
    tc.assertTrue(all(0 < c <= lim.max_tokens for c in chunks), chunks)
    p = start
    for i, c in enumerate(chunks):
        final = i == len(chunks) - 1
        if lim.grid:
            tc.assertEqual(p // lim.grid, (p + c - 1) // lim.grid, f"chunk {p}+{c} crosses the grid")
        if not final:
            tc.assertEqual((p + c) % lim.page, 0, f"chunk end {p + c} off page {lim.page}")
            if lim.graph_buckets and c <= lim.graph_buckets[-1]:
                tc.assertIn(c, lim.graph_buckets, f"non-final chunk {c} misses every graph bucket")
        if not lim.eager:
            tc.assertLessEqual(c, lim.graph_buckets[-1])
        p += c


class TestRule(unittest.TestCase):
    def test_no_gain_means_fixed(self):
        """A model where every chunk costs exactly its tokens has no reason to
        deviate: the plan IS today's fixed plan (identical behaviour)."""
        lim = P.ChunkLimits(2048, 512, 512)
        st = [_prop(40.0)] * 3
        for n in (1, 300, 512, 513, 2048, 8191, 32768):
            res = P.plan_detail(n, 3, st, lim)
            fixed = [512] * (n // 512) + ([n % 512] if n % 512 else [])
            self.assertEqual(list(res.chunks), fixed, n)
            self.assertTrue(res.candidate.startswith("fixed"), res.candidate)

    def test_hysteresis(self):
        """A 0.5 % predicted gain does not move the width under min_gain 1 %."""
        lim = P.ChunkLimits(2048, 512, 512, min_gain=0.01)
        st = [P.StageModel(((512, 40.0), (1024, 79.6)))] * 1
        res = P.plan_detail(32768, 1, st, lim)
        self.assertEqual(set(res.chunks), {512})
        lim0 = P.ChunkLimits(2048, 512, 512, min_gain=0.0)
        self.assertNotEqual(set(P.plan_detail(32768, 1, st, lim0).chunks), {512})

    def test_never_predicted_worse_than_fixed(self):
        rng = random.Random(7)
        for _ in range(60):
            stages = []
            for _s in range(3):
                a = rng.uniform(1.0, 60.0)
                stages.append(P.StageModel(((512, a), (1024, a * rng.uniform(1.6, 2.1)),
                                            (2048, a * rng.uniform(3.0, 4.2))),
                                           rng.uniform(0.0, 0.003), rng.uniform(0.0, 80.0)))
            lim = P.ChunkLimits(2048, 512, 512, graph_buckets=(512,))
            n = rng.choice([700, 2048, 5000, 8192, 20000])
            res = P.plan_detail(n, 3, stages, lim)
            self.assertLessEqual(res.predicted_ms, res.fixed_ms + 1e-6)
            _check_limits(self, res.chunks, n, lim)

    def test_large_fixed_cost_prefers_large_chunks(self):
        """NF form: a_s ~ 1.7 s per forward, so every extra chunk costs a whole
        a_s -- the plan goes to the ceiling."""
        st = [_lin(1700.0, 0.05), _lin(600.0, 0.03), _lin(500.0, 0.03)]
        lim = P.ChunkLimits(4096, 1024, 1024, page=64, grid=4096)
        res = P.plan_detail(32768, 3, st, lim)
        self.assertEqual(set(res.chunks), {4096})
        self.assertGreater(res.gain, 0.3)
        _check_limits(self, res.chunks, 32768, lim)

    def test_ramps_fill_the_pipeline(self):
        """Heavy per-token cost and a light fixed cost: a head ramp starts the
        downstream stages early -- the chosen plan begins small."""
        st = [_lin(1.0, 0.1)] * 3
        lim = P.ChunkLimits(4096, 256, 4096)
        res = P.plan_detail(16384, 3, st, lim)
        self.assertLess(res.chunks[0], 4096)
        self.assertLess(res.chunks[-1], 4096)
        self.assertLess(res.predicted_ms, res.fixed_ms)
        _check_limits(self, res.chunks, 16384, lim)

    def test_limits_hold_across_a_grid_of_cases(self):
        st = [_lin(3.0, 0.08, 0.0015, 20.0), _lin(2.0, 0.07, 0.002, 15.0), _lin(2.5, 0.07, 0.002, 25.0)]
        cases = [
            P.ChunkLimits(2048, 512, 512, graph_buckets=(512,)),
            P.ChunkLimits(2048, 256, 512, graph_buckets=(256, 512)),
            P.ChunkLimits(2048, 512, 512, graph_buckets=(512, 1024, 2048), eager=False),
            P.ChunkLimits(4096, 1024, 4096, page=64, grid=4096),
            P.ChunkLimits(4096, 512, 512, page=64, grid=4096),
        ]
        for lim in cases:
            for start in (0, 300, 4096, 5000):
                for n in (1, 63, 511, 2048, 6000, 9999, 40000):
                    ch = P.chunk_plan(n, 3, st, lim, start=start)
                    _check_limits(self, ch, n, lim, start=start)

    def test_stage_count_must_match(self):
        with self.assertRaises(P.ChunkPolicyError):
            P.chunk_plan(1000, 2, [_prop(1.0)] * 3, P.ChunkLimits(2048, 512, 512))
        self.assertEqual(P.chunk_plan(0, 3, [_prop(1.0)] * 3, P.ChunkLimits(2048, 512, 512)), [])

    def test_tail_hook(self):
        def fold(chunks, start, end):
            # NF-style fold: a last chunk under 1024 joins its predecessor.
            if len(chunks) > 1 and chunks[-1] < 1024 and chunks[-2] + chunks[-1] <= 4096:
                return chunks[:-2] + [chunks[-2] + chunks[-1]]
            return chunks

        st = [_lin(1700.0, 0.05)] * 3
        # min_gain 0.99: the fixed plan wins, so the hook's effect is visible
        # on it: [1024, 1024, 252] -> [1024, 1276].
        lim = P.ChunkLimits(4096, 1024, 1024, min_gain=0.99, tail_hook=fold)
        self.assertEqual(P.chunk_plan(2300, 3, st, lim), [1024, 1276])
        # without the hook the rest stays its own chunk
        self.assertEqual(P.chunk_plan(2300, 3, st, P.ChunkLimits(4096, 1024, 1024, min_gain=0.99)),
                         [1024, 1024, 252])

        def breaker(chunks, start, end):
            return chunks + [1]

        with self.assertRaises(P.ChunkPolicyError):
            P.chunk_plan(5000, 3, st, P.ChunkLimits(4096, 1024, 4096, tail_hook=breaker))


class TestCursor(unittest.TestCase):
    def _planner(self, **kw):
        st = [_lin(1.0, 0.1)] * 3
        spec = P.PolicySpec(tuple(st), P.ChunkLimits(4096, 256, 1024, **kw), "t")
        seen = []
        return P.ChunkPlanner(spec, on_plan=lambda *a: seen.append(a)), seen

    def test_follows_the_plan_and_replans_off_it(self):
        pl, seen = self._planner()
        plan = P.chunk_plan(16384, 3, pl.spec.stages, pl.spec.limits)
        pos, got = 0, []
        while pos < 16384:
            w = pl.next_width("r1", pos, 16384)
            got.append(w)
            pos += w
        self.assertEqual(got, plan)
        self.assertEqual(len(seen), 1)
        # executed width narrowed (corridor): 100 tokens only -> replan, no head ramp
        pl2, seen2 = self._planner()
        w0 = pl2.next_width("r2", 0, 16384)
        w1 = pl2.next_width("r2", 100, 16384)
        self.assertEqual(len(seen2), 2)
        self.assertTrue(seen2[1][-1])  # replan flag
        self.assertEqual(pl2.next_width("r2", 100, 16384), w1)  # stable
        self.assertGreater(w0, 0)

    def test_padded_rest_is_priced(self):
        """With a 512 graph, a 1-token rest replays the whole 512 bucket: the
        model prices that, so 513 tokens go as ONE (eager) chunk."""
        lim = P.ChunkLimits(2048, 512, 512, graph_buckets=(512,))
        res = P.plan_detail(513, 3, [_prop(40.0)] * 3, lim)
        self.assertEqual(list(res.chunks), [513])

    def test_forward_budget_keeps_bundling_room(self):
        spec = P.PolicySpec((_lin(1700.0, 0.05),) * 3, P.ChunkLimits(4096, 256, 1024), "t")
        pl = P.ChunkPlanner(spec)
        self.assertEqual(pl.next_width("short", 0, 300), 300)
        self.assertEqual(P.forward_budget(pl, "short", 0, 300), 1024)  # finishes: >= fixed
        self.assertEqual(P.forward_budget(pl, "done", 50, 50), 0)


class TestShortPromptBypass(unittest.TestCase):
    """rc9j metal (26.09.): 2k/8k measured slower under dynamic although the
    plan was 512xn -- at or below --p-chunk-dynamic-min-tokens nothing is
    planned and every forward is the fixed width."""

    def _spec(self, thr):
        st = [_lin(1700.0, 0.05)] * 3  # a model that WOULD plan large chunks
        return P.PolicySpec(tuple(st), P.ChunkLimits(2048, 512, 512, dynamic_min_tokens=thr), "t")

    def test_bypass_gives_the_fixed_width_and_no_plan(self):
        seen = []
        pl = P.ChunkPlanner(self._spec(8192), on_plan=lambda *a: seen.append(a))
        for n in (1, 300, 2047, 8192):
            pos, widths = 0, []
            while pos < n:
                b = P.forward_budget(pl, f"r{n}", pos, n)
                widths.append(b)
                pos += min(b, n - pos)
            # the forward budget is EXACTLY what the fixed scheduler gives: 512
            self.assertEqual(set(widths), {512}, n)
        self.assertEqual(seen, [])

    def test_above_the_threshold_the_plan_is_unchanged(self):
        spec = self._spec(8192)
        pl = P.ChunkPlanner(spec)
        n = 8193
        want = P.chunk_plan(n, 3, spec.stages, spec.limits)
        pos, got = 0, []
        while pos < n:
            w = pl.next_width("long", pos, n)
            got.append(w)
            pos += w
        self.assertEqual(got, want)
        self.assertEqual(got, P.chunk_plan(n, 3, spec.stages, P.ChunkLimits(2048, 512, 512)))
        # the tail of a planned request stays on its plan below the threshold
        self.assertNotEqual(set(got), {512})

    def test_zero_is_off_and_roundtrip(self):
        pl = P.ChunkPlanner(self._spec(0))
        self.assertEqual(pl.next_width("s", 0, 2048), 2048)
        spec = self._spec(8192)
        back = P.PolicySpec.from_json(spec.to_json())
        self.assertEqual(back.limits.dynamic_min_tokens, 8192)
        self.assertEqual(P.ChunkLimits.from_json({"max_tokens": 2048, "min_tokens": 512,
                                                  "fixed_tokens": 512}).dynamic_min_tokens, 0)
        with self.assertRaises(P.ChunkPolicyError):
            P.ChunkLimits(2048, 512, 512, dynamic_min_tokens=-1)


class TestEnv(unittest.TestCase):
    def test_fixed_and_unset_are_none(self):
        self.assertIsNone(P.policy_from_env({}))
        self.assertIsNone(P.policy_from_env({P.POLICY_ENV: "fixed"}))
        self.assertIsNone(P.planner_from_env({P.POLICY_ENV: "fixed"}))

    def test_dynamic_without_spec_raises(self):
        with self.assertRaises(P.ChunkPolicyError):
            P.policy_from_env({P.POLICY_ENV: "dynamic"})
        with self.assertRaises(P.ChunkPolicyError):
            P.policy_from_env({P.POLICY_ENV: "sometimes"})

    def test_roundtrip_and_armed_line(self):
        spec = P.PolicySpec((_prop(40.0, 0.002, 60.0),) * 3,
                            P.ChunkLimits(2048, 512, 512, graph_buckets=(512,)), "unit")
        back = P.policy_from_env({P.POLICY_ENV: "dynamic", P.SPEC_ENV: spec.to_json()})
        self.assertEqual(back.stages, spec.stages)
        self.assertEqual(back.limits.key(), spec.limits.key())
        lines = []
        pl = P.planner_from_env({P.POLICY_ENV: "dynamic", P.SPEC_ENV: spec.to_json()}, log=lines.append)
        self.assertIn("P-CHUNK-POLICY armed policy=dynamic", lines[0])
        pl.next_width("rid", 0, 9000)
        self.assertTrue(lines[1].startswith("P-CHUNK-POLICY plan policy=dynamic key=rid start=0 tokens=9000"))


# ---------------------------------------------------------------------------
# launcher (27B line)

try:
    from sglang.srt.weg2 import launcher as L
except Exception as exc:  # pragma: no cover
    L = None
    _LERR = exc


@pytest.mark.skipif(L is None, reason="weg2 launcher unavailable")
class TestLauncher(unittest.TestCase):
    def setUp(self):
        self._g = dict(L._P_PREFILL_GRAPH)
        self._c = dict(L._P_CHUNK)

    def tearDown(self):
        L._P_PREFILL_GRAPH.clear()
        L._P_PREFILL_GRAPH.update(self._g)
        L._P_CHUNK.clear()
        L._P_CHUNK.update(self._c)

    def _apply(self, *extra):
        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", *extra])
        L.apply_p_prefill_graph(ns)
        L.apply_p_chunk_policy(ns)
        return ns

    def test_default_fixed_is_byte_identical(self):
        ns = self._apply("--p-prefill-graph", "512")
        self.assertEqual(ns.p_chunk_policy, "fixed")
        self.assertEqual(L.p_chunked_prefill_tokens(), 512)
        self.assertEqual(L.p_chunk_policy_env(), {})
        self.assertEqual(L.p_chunk_policy_lines(), [])
        self._apply()
        self.assertEqual(L.p_chunked_prefill_tokens(), L.CHUNKED_PREFILL_TOKENS)
        # the dynamic-only knobs are inert under fixed
        self._apply("--p-prefill-graph", "512", "--p-chunk-max", "4096")
        self.assertEqual(L.p_chunked_prefill_tokens(), 512)

    def test_dynamic_prices_the_ceiling_and_ships_the_spec(self):
        self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic")
        self.assertEqual(L.p_chunked_prefill_tokens(), L.P_CHUNK_MAX_DEFAULT)
        env = L.p_chunk_policy_env()
        self.assertEqual(env[P.POLICY_ENV], "dynamic")
        spec = P.PolicySpec.from_json(env[P.SPEC_ENV])
        self.assertEqual(len(spec.stages), 3)
        self.assertEqual(spec.limits.graph_buckets, (512,))
        self.assertEqual(spec.limits.fixed_tokens, 512)
        self.assertEqual(spec.limits.ladder(), (512, 1024, 2048))
        lines = L.p_chunk_policy_lines()
        self.assertEqual(len(lines), 1 + len(L.P_CHUNK_DRY_RUN_TOKENS))
        self.assertIn("ceiling 2048", lines[0])
        # default bypass 8192: 2k/8k are not planned, 32k/128k are
        self.assertEqual(spec.limits.dynamic_min_tokens, 8192)
        self.assertIn("dynamic_min_tokens=8192", lines[0])
        self.assertIn("BYPASS", lines[1])
        self.assertIn("BYPASS", lines[2])
        self.assertNotIn("BYPASS", lines[3])

    def test_bypass_flag_and_scheduler_width(self):
        self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic",
                    "--p-chunk-dynamic-min-tokens", "0")
        self.assertEqual(L.p_chunk_policy_spec().limits.dynamic_min_tokens, 0)
        self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic")
        spec = P.PolicySpec.from_json(L.p_chunk_policy_env()[P.SPEC_ENV])
        from sglang.srt.managers.scheduler import Scheduler

        class _S:
            _p_chunk_policy_width = Scheduler._p_chunk_policy_width

        s = _S()
        s.chunked_prefill_size = L.p_chunked_prefill_tokens()  # the 2048 ceiling
        s.waiting_queue = []
        s._p_chunk_planner = P.ChunkPlanner(spec)
        for n in (2047, 8191):
            for pos in range(0, n, 512):
                s.chunked_req = SimpleNamespace(rid=f"q{n}", full_untruncated_fill_ids=list(range(n)),
                                                origin_input_ids=[], output_ids=[],
                                                prefix_indices=list(range(pos)))
                self.assertEqual(s._p_chunk_policy_width(), 512, (n, pos))

    def test_dynamic_refusals_and_forced_probe(self):
        with self.assertRaises(SystemExit):
            self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic", "--p-chunk-max", "256")
        with self.assertRaises(SystemExit):
            self._apply("--p-chunk-policy", "dynamic", "--p-chunk-model", "/nonexistent.json")
        self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic", "--p-chunk-min", "2048",
                    "--p-chunk-max", "2048", "--p-chunk-fixed", "2048")
        spec = L.p_chunk_policy_spec()
        self.assertEqual(P.chunk_plan(8192, 3, spec.stages, spec.limits), [2048] * 4)

    def test_builtin_int8_is_calibrated_against_its_boot(self):
        """The builtin model's FIXED prediction against weg2rc7c's measured
        P ladder (8k 940 ms, 32k 4230 ms wall medians): within 5 %."""
        self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic")
        spec = L.p_chunk_policy_spec()
        for n, wall in ((8192, 940.0), (32768, 4230.0)):
            res = P.plan_detail(n, 3, spec.stages, spec.limits)
            self.assertLess(abs(res.fixed_ms - wall) / wall, 0.05, (n, res.fixed_ms))

    def test_graph_bucket_hits(self):
        self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic", "--p-chunk-max", "4096")
        spec = L.p_chunk_policy_spec()
        for n in (2048, 8192, 32768, 131072, 5000):
            ch = P.chunk_plan(n, 3, spec.stages, spec.limits)
            _check_limits(self, ch, n, spec.limits)

    def test_json_model(self):
        import tempfile

        spec = [P.StageModel(((512, 40.0), (2048, 150.0)), 0.002, 60.0).to_json()] * 3
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"stages": spec}, fh)
        with self.assertRaises(SystemExit):  # no graph: fixed 4096 above the 2048 ceiling
            self._apply("--p-chunk-policy", "dynamic", "--p-chunk-model", fh.name)
        self._apply("--p-chunk-policy", "dynamic", "--p-chunk-model", fh.name, "--p-chunk-fixed", "512")
        self.assertEqual(L.p_chunk_policy_spec().source, "json:" + fh.name.rsplit("/", 1)[-1])
        self.assertEqual(L.p_chunked_prefill_tokens(), 2048)


# ---------------------------------------------------------------------------
# scheduler hook


class TestSchedulerHook(unittest.TestCase):
    def _sched(self, planner, chunked_req=None, queue=(), static=2048):
        from sglang.srt.managers.scheduler import Scheduler

        class _S:
            dynamic_chunked_prefill_size = Scheduler.dynamic_chunked_prefill_size
            _p_chunk_policy_width = Scheduler._p_chunk_policy_width
            _log_dynamic_chunk_engagement = Scheduler._log_dynamic_chunk_engagement

        s = _S()
        s.chunked_prefill_size = static
        s.enable_dynamic_chunking = False
        s.chunked_req = chunked_req
        s.waiting_queue = list(queue)
        s._p_chunk_planner = planner
        return s

    def _req(self, rid, n, prefix):
        return SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(n)),
                               origin_input_ids=list(range(n)), output_ids=[],
                               prefix_indices=list(range(prefix)))

    def test_no_planner_is_the_old_path(self):
        s = self._sched(None, queue=[self._req("a", 9000, 0)])
        self.assertEqual(s.dynamic_chunked_prefill_size(), 2048)

    def test_head_and_chunked_request_follow_the_plan(self):
        st = [_lin(1700.0, 0.05)] * 3
        spec = P.PolicySpec(tuple(st), P.ChunkLimits(2048, 512, 512), "t")
        pl = P.ChunkPlanner(spec)
        head = self._req("h", 9000, 0)
        s = self._sched(pl, queue=[head])
        self.assertEqual(s.dynamic_chunked_prefill_size(), 2048)
        s.chunked_req = self._req("h", 9000, 2048)
        self.assertEqual(s.dynamic_chunked_prefill_size(), 2048)
        # capped by the static budget (the ceiling)
        s.chunked_prefill_size = 1024
        s.chunked_req = self._req("x", 9000, 0)
        self.assertEqual(s.dynamic_chunked_prefill_size(), 1024)
        # nothing to serve -> static
        s2 = self._sched(pl)
        self.assertEqual(s2.dynamic_chunked_prefill_size(), 2048)

    def test_a_failing_plan_falls_back_to_static(self):
        class Boom:
            spec = None

            def next_width(self, *a):
                raise RuntimeError("boom")

        s = self._sched(Boom(), queue=[self._req("a", 9000, 0)], static=512)
        self.assertEqual(s.dynamic_chunked_prefill_size(), 512)


if __name__ == "__main__":
    unittest.main()
