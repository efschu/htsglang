"""P cut model + prefill-graph policy (27B line, 26.09.).

User orders 26.09. ~05:15Z: "der eigentliche Hebel ist jetzt der
Schichtschnitt" / "cuda graphen im prefill nur anschalten wenn es was bringt,
wenn es negativ ist, ausschalten". Pinned without a GPU:

  * p_graph_policy -- the auto rule (slowest stage, hysteresis), the SAFE
    DEFAULT without a measurement (main bucket graph, extras eager), on / off;
  * the launcher -- auto without a table is BYTE-IDENTICAL to the pre-switch
    form (argv, env, pool post, chunk spec); a table that puts eager ahead
    drops the graph (P keeps its chunk); extra buckets and their VRAM gate;
  * consistency -- the chunk policy prices each width in the decided mode
    from the same numbers;
  * p_chunk_policy -- graph/eager points, width-dependent attention, the
    measurement sweep, old JSON unchanged;
  * p_stage_model -- contiguity, the formula, the fitter, and the committed
    rc9j models against the metal #PGAP sums they were fitted on.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import pytest

from sglang.srt.weg2 import p_chunk_policy as P
from sglang.srt.weg2 import p_graph_policy as G
from sglang.srt.weg2 import p_stage_model as M
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

DATA = os.path.join(os.path.dirname(M.__file__), "p_stage_model_data")


def _cal(widths, ref=16384):
    return G.Calibration.from_json({"ref_prefix": ref, "widths": widths, "source": "t"})


# ---------------------------------------------------------------------------
# p_graph_policy


class TestGraphPolicyRule(unittest.TestCase):
    def test_no_table_is_the_safe_default(self):
        cap, vs = G.decide("auto", 512, (), (1024, 2048), None, 3)
        self.assertEqual(cap, (512,))
        self.assertEqual([v.graph for v in vs], [True, False, False])
        self.assertIn("safe default", vs[1].reason)

    def test_incomplete_measurement_is_the_safe_default(self):
        cal = _cal({"512": {"graph_ms": [10, 10], "eager_ms": [20, 20, 20]},
                    "1024": {"graph_ms": [1, 1, 1]}})
        cap, _ = G.decide("auto", 512, (), (1024,), cal, 3)
        self.assertEqual(cap, (512,))

    def test_graph_kept_only_when_faster_at_the_slowest_stage(self):
        cal = _cal({"512": {"graph_ms": [40, 30, 35], "eager_ms": [45, 30, 35]},
                    # graph speeds up PP0 (idle) but slows the bottleneck PP2
                    "1024": {"graph_ms": [60, 70, 90], "eager_ms": [80, 70, 85]},
                    "2048": {"graph_ms": [150, 150, 150], "eager_ms": [170, 160, 165]}})
        cap, vs = G.decide("auto", 512, (), (1024, 2048), cal, 3)
        self.assertEqual(cap, (512, 2048))
        self.assertFalse(vs[1].graph)

    def test_negative_graph_is_switched_off_with_its_tiny_buckets(self):
        cal = _cal({"512": {"graph_ms": [46, 38, 40], "eager_ms": [45, 37, 39]}})
        cap, vs = G.decide("auto", 512, (16,), (), cal, 3)
        self.assertEqual(cap, ())
        self.assertFalse(vs[0].graph)

    def test_hysteresis(self):
        cal = _cal({"512": {"graph_ms": [44.8, 30, 30], "eager_ms": [45, 30, 30]}})
        self.assertEqual(G.decide("auto", 512, (), (), cal, 3)[0], ())
        self.assertEqual(G.decide("auto", 512, (), (), cal, 3, min_gain=0.0)[0], (512,))

    def test_attention_term_enters_at_the_reference_prefix(self):
        # equal base, graph attention 2x slower: eager wins once the prefix is deep
        w = {"512": {"graph_ms": [40, 40, 40], "eager_ms": [41, 41, 41],
                     "graph_attn": [0.0006, 0, 0], "eager_attn": [0.0003, 0, 0]}}
        self.assertEqual(G.decide("auto", 512, (), (), _cal(w, ref=0), 3)[0], (512,))
        self.assertEqual(G.decide("auto", 512, (), (), _cal(w, ref=65536), 3)[0], ())

    def test_on_and_off(self):
        self.assertEqual(G.decide("on", 512, (16,), (1024, 2048), None, 3)[0], (16, 512, 1024, 2048))
        self.assertEqual(G.decide("off", 512, (16,), (), None, 3)[0], (16, 512))
        with self.assertRaises(G.GraphPolicyError):
            G.decide("off", 512, (), (1024,), None, 3)
        with self.assertRaises(G.GraphPolicyError):
            G.decide("auto", 512, (), (256,), None, 3)
        with self.assertRaises(G.GraphPolicyError):
            G.decide("auto", 0, (), (1024,), None, 3)
        with self.assertRaises(G.GraphPolicyError):
            G.decide("maybe", 512, (), (), None, 3)
        self.assertEqual(G.decide("auto", 0, (), (), None, 3), ((), ()))

    def test_table_json(self):
        with self.assertRaises(G.GraphPolicyError):
            G.Calibration.from_json({"widths": {"x": {}}})
        with self.assertRaises(G.GraphPolicyError):
            G.Calibration.from_json({"widths": {"512": {"graph_ms": [-1, 1, 1]}}})
        cal = _cal({"512": {"graph_ms": [1, 2, 3], "eager_ms": [2, 3, 4], "graph_attn": [0.1, 0.2, 0.3]}})
        self.assertEqual(G.Calibration.from_json(cal.to_json()), cal)


class TestOverlayConsistency(unittest.TestCase):
    def test_overlay_prices_each_width_in_its_mode(self):
        base = tuple(P.StageModel(((512, 40.0), (2048, 160.0)), 0.0003, 60.0, f"PP{r}") for r in range(3))
        cal = _cal({"512": {"graph_ms": [38, 30, 33], "eager_ms": [44, 29, 32]},
                    "2048": {"graph_ms": [150, 140, 145], "eager_ms": [155, 150, 150],
                             "eager_attn": [0.0002, 0.0007, 0.0007]}})
        cap, _ = G.decide("auto", 512, (), (2048,), cal, 3)
        st = G.overlay_stage_models(base, cal, cap)
        lim = P.ChunkLimits(2048, 512, 512, graph_buckets=cap)
        for r in range(3):
            m, eager = lim.exec_shape(512)
            self.assertEqual((m, eager), (512, False))
            self.assertAlmostEqual(st[r].forward_ms(m, 512, 0, eager),
                                   cal.widths[512].graph_ms[r] + 0.0003 * 512 * 256 / 1000.0)
            self.assertAlmostEqual(st[r].base_ms(2048, True), cal.widths[2048].eager_ms[r])
            self.assertAlmostEqual(st[r].attn_coeff(2048, True), cal.widths[2048].eager_attn[r])
        self.assertEqual(G.overlay_stage_models(base, None, (512,)), base)


# ---------------------------------------------------------------------------
# launcher

try:
    from sglang.srt.weg2 import launcher as L
except Exception as exc:  # pragma: no cover
    L = None
    _L_ERR = exc

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
BUDGETS = [28904, 17704, 17672]


@unittest.skipIf(L is None, "weg2 launcher unavailable")
class TestLauncherGraphPolicy(unittest.TestCase):
    def setUp(self):
        self._g = dict(L._P_PREFILL_GRAPH)
        self._c = dict(L._P_CHUNK)
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        L._P_PREFILL_GRAPH.clear()
        L._P_PREFILL_GRAPH.update(self._g)
        L._P_CHUNK.clear()
        L._P_CHUNK.update(self._c)
        self._tmp.cleanup()

    def _apply(self, *extra):
        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", *extra])
        L.apply_p_prefill_graph(ns)
        L.apply_p_chunk_policy(ns)
        return ns

    def _snapshot(self, ns):
        argv = L.argv_p("py", MODEL, BUDGETS, 8, 1024, "8.0", [],
                        stage_ratio="42,11,11", attn_stage_ratio="10,3,3")
        return (argv, L.p_prefill_graph_pool_mib(ns),
                L.p_prefill_graph_env(L.p_prefill_graph_pool_mib(ns)),
                L.p_graph_split_env(ns), L.p_chunk_policy_env(), L.p_chunked_prefill_tokens())

    def _write(self, obj, name="cal.json"):
        path = os.path.join(self._tmp.name, name)
        with open(path, "w") as fh:
            json.dump(obj, fh)
        return path

    def test_parser_defaults(self):
        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
        self.assertEqual(ns.p_prefill_graph_policy, "auto")
        self.assertEqual((ns.p_prefill_graph_buckets, ns.p_prefill_graph_calibration, ns.p_chunk_sweep),
                         ("", "", ""))

    def test_auto_without_table_is_byte_identical_to_off(self):
        for base in (["--p-prefill-graph", "512", "--p-prefill-graph-split", "7"],
                     ["--p-prefill-graph", "512", "--p-prefill-graph-split", "7", "--p-chunk-policy", "dynamic"],
                     ["--p-prefill-graph", "512", "--p-prefill-graph-tiny", "16"],
                     []):
            a = self._snapshot(self._apply(*base))
            b = self._snapshot(self._apply(*base, "--p-prefill-graph-policy", "off"))
            self.assertEqual(a, b, base)
            if base:
                self.assertIn('"bs":[', " ".join(a[0]))

    def test_table_that_puts_eager_ahead_switches_the_graph_off(self):
        path = self._write({"widths": {"512": {"graph_ms": [46, 38, 40], "eager_ms": [44, 37, 39]}}})
        ns = self._apply("--p-prefill-graph", "512", "--p-prefill-graph-split", "7",
                         "--p-prefill-graph-calibration", path)
        argv, pool, env, split, cenv, chunk = self._snapshot(ns)
        self.assertNotIn("--cuda-graph-config", argv)
        self.assertEqual((pool, env, split), ((), {}, {}))
        self.assertEqual(chunk, 512)  # P keeps its chunk, it just runs eager
        self.assertEqual(argv[argv.index("--chunked-prefill-size") + 1], "512")
        self.assertIn("NOTHING captured", L.p_prefill_graph_line())
        self.assertTrue(any("-> eager" in x for x in L.p_prefill_graph_policy_lines()))

    def test_extra_buckets_calibrated_and_consistent_with_the_plan(self):
        table = {"ref_prefix": 16384, "source": "unit",
                 "widths": {"512": {"graph_ms": [41, 34, 38], "eager_ms": [45, 33, 37]},
                            "1024": {"graph_ms": [70, 66, 70], "eager_ms": [80, 73, 77]},
                            "2048": {"graph_ms": [170, 170, 175], "eager_ms": [161, 163, 168]}}}
        # the table rides in the --p-chunk-model JSON (graph_calibration key)
        stages = [P.StageModel(((512, 41.0), (1024, 80.0), (2048, 161.0)), 0.0028 * k, 0.0, f"PP{k}").to_json()
                  for k in (1, 1, 1)]
        path = self._write({"stages": stages, "graph_calibration": table}, "model.json")
        ns = self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic",
                         "--p-chunk-model", path, "--p-prefill-graph-buckets", "1024,2048")
        self.assertEqual(L.p_prefill_graph_captured(), [512, 1024])
        argv, pool, env, _, cenv, chunk = self._snapshot(ns)
        cfg = json.loads(argv[argv.index("--cuda-graph-config") + 1])
        self.assertEqual(cfg["prefill"]["bs"], [512, 1024])
        self.assertEqual(pool, (L.P_PREFILL_GRAPH_POOL_MIB_PER_512 * 3,) * 3)
        spec = P.PolicySpec.from_json(cenv[P.SPEC_ENV])
        self.assertEqual(spec.limits.graph_buckets, (512, 1024))
        self.assertIn("+graphcal:unit", spec.source)
        for r in range(3):
            # 1024 is captured -> the plan prices its GRAPH cost; 2048 eager -> its EAGER cost
            self.assertAlmostEqual(spec.stages[r].base_ms(1024, False), table["widths"]["1024"]["graph_ms"][r])
            self.assertAlmostEqual(spec.stages[r].base_ms(2048, True), table["widths"]["2048"]["eager_ms"][r])
            self.assertEqual(spec.limits.exec_shape(2048), (2048, True))
            self.assertEqual(spec.limits.exec_shape(1024), (1024, False))
        # 'on' captures all of them, without a table
        self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic",
                    "--p-prefill-graph-policy", "on", "--p-prefill-graph-buckets", "1024,2048")
        self.assertEqual(L.p_prefill_graph_captured(), [512, 1024, 2048])
        with self.assertRaises(SystemExit):
            self._apply("--p-prefill-graph", "512", "--p-prefill-graph-policy", "off",
                        "--p-prefill-graph-buckets", "1024")

    def test_vram_gate(self):
        from sglang.srt.planner import pp_cut as PC

        table = {"widths": {"512": {"graph_ms": [41, 34, 38], "eager_ms": [45, 33, 37]},
                            "2048": {"graph_ms": [150, 150, 150], "eager_ms": [161, 163, 168]}}}
        path = self._write(table)
        fam = tuple(PC.LAYER_FAMILY_ATTENTION if i % 4 == 3 else PC.LAYER_FAMILY_LINEAR for i in range(64))

        def pool_model(free0):
            return PC.PhasePoolModel(
                free_mib=(free0, 15744.0, 15472.0), weight_mib_per_layer=363.4,
                kv_mib_per_token_per_attn_layer=2048 / PC.MIB, arming_floor_mib=(1229.0,) * 3,
                stage_fixed_mib=(2342.0, 1105.5, 3518.0),
                prefill_graph_pool_mib=L.p_prefill_graph_pool_mib(ns), activation_reserve_mib=1024.0,
                corridor_holdback_mib=1800.0, mamba_mib_per_linear_layer_per_slot=1.5588, mamba_slots=3)

        logs = []
        ns = self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic",
                         "--p-prefill-graph-calibration", path, "--p-prefill-graph-buckets", "2048")
        self.assertEqual(L.p_prefill_graph_captured(), [512, 2048])
        # INT8 rc9j budgets: 42,11,11 clears 264192 by 5.6k tokens with the 512 post
        # alone; +4x160 MiB of 2048 capture pool on PP0 (10 attn layers) costs 32.8k
        mp = L.p_prefill_graph_vram_gate(ns, pool_model(26008.0), [42, 11, 11], fam, 264192, logs.append)
        self.assertEqual(L.p_prefill_graph_captured(), [512])
        self.assertEqual(mp.prefill_graph_pool_mib, (160.0,) * 3)
        self.assertIn("DROPPED", logs[-1])
        self.assertEqual(L.p_chunk_policy_spec().limits.graph_buckets, (512,))
        # with room it is carried
        ns = self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic",
                         "--p-prefill-graph-calibration", path, "--p-prefill-graph-buckets", "2048")
        mp = L.p_prefill_graph_vram_gate(ns, pool_model(30000.0), [42, 11, 11], fam, 264192, logs.append)
        self.assertEqual(L.p_prefill_graph_captured(), [512, 2048])
        self.assertIn("CARRIED", logs[-1])
        # an unpinned cut takes the safe side
        L.p_prefill_graph_vram_gate(ns, pool_model(30000.0), None, fam, 264192, logs.append)
        self.assertEqual(L.p_prefill_graph_captured(), [512])
        # no extras: a no-op returning the same object
        ns = self._apply("--p-prefill-graph", "512")
        m0 = pool_model(26008.0)
        self.assertIs(L.p_prefill_graph_vram_gate(ns, m0, [42, 11, 11], fam, 264192, logs.append), m0)

    def test_sweep_flag(self):
        self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic",
                    "--p-chunk-sweep", "512,1024,2048")
        spec = P.PolicySpec.from_json(L.p_chunk_policy_env()[P.SPEC_ENV])
        self.assertEqual(spec.limits.sweep, (512, 1024, 2048))
        self.assertIn("SWEEP=", L.p_chunk_policy_lines()[0])
        with self.assertRaises(SystemExit):
            self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic", "--p-chunk-sweep", "4096")


# ---------------------------------------------------------------------------
# p_chunk_policy extensions


class TestChunkPolicyExtensions(unittest.TestCase):
    def test_old_json_is_unchanged(self):
        st = P.StageModel(((512, 40.0),), 0.002, 60.0, "x")
        self.assertEqual(set(st.to_json()), {"points", "attn_ms_per_tok_1k", "eager_floor_ms", "name"})
        lim = P.ChunkLimits(2048, 512, 512, graph_buckets=(512,))
        self.assertNotIn("sweep", lim.to_json())
        self.assertEqual(P.StageModel.from_json(st.to_json()), st)

    def test_graph_eager_and_width_dependent_attention(self):
        st = P.StageModel(((512, 40.0), (2048, 160.0)), 0.0028, 0.0, "x",
                          eager_points=((512, 45.0), (2048, 150.0)),
                          attn_points=((512, 0.0028), (2048, 0.0020)),
                          eager_attn_points=((512, 0.0026), (2048, 0.0017)))
        self.assertEqual(st.base_ms(2048, False), 160.0)
        self.assertEqual(st.base_ms(2048, True), 150.0)
        self.assertAlmostEqual(st.attn_coeff(1280, False), 0.0024)
        self.assertAlmostEqual(st.attn_coeff(8192, True), 0.0017)
        self.assertAlmostEqual(st.forward_ms(2048, 2048, 4096, True), 150.0 + 0.0017 * 2048 * 5120 / 1000.0)
        rt = P.StageModel.from_json(json.loads(json.dumps(st.to_json())))
        self.assertEqual(rt, st)
        with self.assertRaises(P.ChunkPolicyError):
            P.StageModel(((512, 1.0),), eager_points=((0, 1.0),))

    def test_sweep_cursor(self):
        st = (P.StageModel(((512, 40.0),)),) * 3
        spec = P.PolicySpec(st, P.ChunkLimits(2048, 512, 512, graph_buckets=(512,),
                                               dynamic_min_tokens=8192, sweep=(512, 1024, 2048)))
        pl = P.ChunkPlanner(spec)
        got = {}
        for k, n in enumerate([2048, 5000, 9000, 2048]):
            key, pos, widths = f"r{k}", 0, []
            while pos < n:
                w = pl.next_width(key, pos, n)
                widths.append(w)
                pos += w
            got[k] = widths
        self.assertEqual(got[0], [512] * 4)
        self.assertEqual(got[1], [1024] * 4 + [904])     # below the bypass, still swept
        self.assertEqual(got[2], [2048] * 4 + [808])
        self.assertEqual(got[3], [512] * 4)              # cyclic
        rt = P.ChunkLimits.from_json(spec.limits.to_json())
        self.assertEqual(rt.sweep, (512, 1024, 2048))


# ---------------------------------------------------------------------------
# p_stage_model


class TestStageModel(unittest.TestCase):
    def _toy(self):
        cards = {"A": M.CardCost("A", {"graph": [(512, 1.0)], "eager": [(1024, 2.0), (2048, 4.0)]},
                                 {"graph": [(512, 0.28)], "eager": [(1024, 0.21), (2048, 0.17)]}),
                 "B": M.CardCost("B", {"graph": [(512, 3.0)], "eager": [(1024, 6.5), (2048, 14.0)]},
                                 {"graph": [(512, 0.67)], "eager": [(2048, 0.71)]},
                                 last_fixed_ms={"graph": [(512, 4.0)]})}
        return M.LayerCostModel(cards, ("A", "B", "B"), M.layer_types_every(64))

    def test_contiguity(self):
        m = self._toy()
        self.assertEqual(m.attn_counts((42, 11, 11)), (10, 3, 3))
        self.assertEqual(m.attn_counts((49, 8, 7)), (12, 2, 2))
        self.assertEqual(m.attn_counts((52, 6, 6)), (13, 1, 2))   # never 8 on a 52-layer stage 0
        with self.assertRaises(M.StageModelError):
            m.attn_counts((60, 4))

    def test_formula_and_stage_models_agree(self):
        m = self._toy()
        cut = (42, 11, 11)
        self.assertAlmostEqual(m.stage_ms(cut, 2, 512, 1000, "graph"),
                               11 * 3.0 + 3 * 0.67 * 512 * 1256 / 1e6 + 4.0)
        st = m.stage_models(cut)
        lim = P.ChunkLimits(2048, 512, 512, graph_buckets=(512,))
        for r in range(3):
            for w, p in ((512, 0), (512, 70000), (1024, 5000), (2048, 120000)):
                me, eager = lim.exec_shape(w)
                self.assertAlmostEqual(st[r].forward_ms(me, w, p, eager),
                                       m.stage_ms(cut, r, me, p, "eager" if eager else "graph"), places=3)
        rt = M.LayerCostModel.from_json(json.loads(json.dumps(m.to_json())))
        self.assertEqual(rt.stage_ms(cut, 0, 2048, 9999, "eager"), m.stage_ms(cut, 0, 2048, 9999, "eager"))

    def test_fitter_recovers_a_synthetic_boot(self):
        lines, fwd = [], {0: 0, 1: 0, 2: 0}
        a = {0: 42.0, 1: 34.5, 2: 38.7}
        s = {0: 2.81, 1: 2.0, 2: 2.05}
        for req in range(3):
            pos = 0
            while pos < 16384:
                for r in range(3):
                    fwd[r] += 1
                    lines.append(f"[t PP{r}] #969N ADMIT slot=1 fwd_ct={fwd[r] - 1} bs=1 extend=512 "
                                 f"input_ids=None rids=[req{req}]")
                    g = a[r] + s[r] * M.attn_work(512, pos)
                    lines.append(f"[2026-09-26 00:00:00 PP{r}] #PGAP pp_rank={r} fwd={fwd[r]} tokens=512 "
                                 f"gpu_gap_ms=0.1 gpu_fwd_ms={g:.3f} host[launch=2]")
                pos += 512
        lines.insert(0, "[t PP0] PREFILL-GRAPH captured backend=full pp_rank=0 buckets=[512] slots=1")
        smp = M.samples_from_lines(lines)
        self.assertEqual(len(smp), 3 * 3 * 32)
        self.assertTrue(all(x.mode == "graph" for x in smp))
        fit = M.fit_width_lines(smp)
        for r in range(3):
            self.assertAlmostEqual(fit[(r, 512, "graph")].a_ms, a[r], places=2)
            self.assertAlmostEqual(fit[(r, 512, "graph")].s, s[r], places=3)
        cc = M.card_costs_from_fit(fit, (42, 11, 11), (10, 3, 3), ("A", "B", "B"))
        self.assertAlmostEqual(cc["A"].gemm(512, "graph"), 1.0, places=3)
        self.assertAlmostEqual(cc["B"].gemm(512, "graph"), 34.5 / 11, places=3)
        self.assertAlmostEqual(cc["B"].last_fixed(512, "graph"), 38.7 - 34.5, places=2)
        self.assertAlmostEqual(cc["B"].attn_coeff(512, "graph"), (2.0 / 3 + 2.05 / 3) / 2, places=4)

    def test_drift_is_never_worse_than_the_best_static_cut(self):
        m = self._toy()
        cuts = [(40, 12, 12), (42, 11, 11), (44, 10, 10)]
        lim = P.ChunkLimits(2048, 512, 512, graph_buckets=(512,))
        ch = [2048] * 16
        ms, per, _ = M.best_drifting_makespan(m, cuts, ch, lim, grid=4)
        static = min(M.drifting_cut_makespan(m, [c], ch, lim)[0] for c in cuts)
        self.assertLessEqual(ms, static + 1e-9)
        self.assertEqual(len(per), len(ch))


@unittest.skipIf(L is None, "weg2 launcher unavailable")
class TestPchunkExport(unittest.TestCase):
    def test_exported_cut_model_drives_the_launcher_plan(self):
        m = M.load_model(os.path.join(DATA, "27b_int8_rc9j.json"))
        doc = M.pchunk_json(m, (41, 12, 11))
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(doc, fh)
        try:
            models, src = L.p_chunk_stage_model(fh.name, "int8", 3)
        finally:
            os.unlink(fh.name)
        self.assertTrue(src.startswith("json:"))
        self.assertEqual(models, m.stage_models((41, 12, 11)))
        lim = P.ChunkLimits(2048, 512, 512, graph_buckets=(512,))
        # on the rc9j model the 3080s are cheaper per token at 1024 than at 2048 eager
        self.assertEqual(set(P.chunk_plan(131072, 3, models, lim)[:100]), {1024})
        # the committed proposal files are exactly this export
        with open(os.path.join(DATA, "27b_int8_rc9j_cut41-12-11.pchunk.json")) as fh2:
            self.assertEqual(json.load(fh2)["stages"], json.loads(json.dumps(doc))["stages"])


class TestCommittedRc9jModels(unittest.TestCase):
    """The committed rc9j models against the metal #PGAP sums (abnahme_cu130.log
    'AUSWERTUNG ... chunkab #PGAP', median per run) -- within 4 % per stage."""

    METAL = {
        "int8": ((42, 11, 11), {32768: (4.19, 3.29, 3.57), 131072: (34.88, 25.97, 27.39)}),
        "nvfp4": ((49, 8, 7), {32768: (5.33, 4.54, 4.00), 131072: (42.86, 26.97, 25.26)}),
    }

    def test_fixed_512_ladder(self):
        for fmt, (cut, rungs) in self.METAL.items():
            m = M.load_model(os.path.join(DATA, f"27b_{fmt}_rc9j.json"))
            lim = P.ChunkLimits(512, 512, 512, graph_buckets=(512,))
            for n, metal in rungs.items():
                r = M.evaluate_cut(m, cut, lim, (n,))[0]
                for k in range(3):
                    self.assertLess(abs(r.stage_busy_ms[k] / 1000.0 / metal[k] - 1.0), 0.04, (fmt, n, k))

    def test_dynamic_2048_ladder(self):
        # metal plan of the NVFP4 dynamic boot at 128k: 2048x62,1024x2,512x3,510 -> 33.41/28.29/26.02 s
        m = M.load_model(os.path.join(DATA, "27b_nvfp4_rc9j.json"))
        lim = P.ChunkLimits(2048, 512, 512, graph_buckets=(512,))
        ch = [2048] * 62 + [1024] * 2 + [512] * 3 + [510]
        r = M.evaluate_cut(m, (49, 8, 7), lim, (sum(ch),), {sum(ch): ch})[0]
        for k, metal in enumerate((33.41, 28.29, 26.02)):
            self.assertLess(abs(r.stage_busy_ms[k] / 1000.0 / metal - 1.0), 0.04, k)


if __name__ == "__main__":
    unittest.main()
