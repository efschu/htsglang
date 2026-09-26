"""Prefill-graph calibration is valid only for its P cut (27B line, Agent PG2 26.09.).

Finding (Agent PG, 26.09.): graphcal_int8.json compared a graph arm at P cut
43,11,10 (attn 10,3,3) with an eager arm at 44,10,10 (attn 11,2,3) -- the
solver cut each arm anew -- so PP1 compared 3 against 2 full-attention layers
and the table measured the cut, not the graph. docker/graphcal_eval.py now
refuses G != E and writes ``pp_layer_ratio``; this file pins the runtime side:

  * p_graph_policy -- ``pp_layer_ratio`` parsed / written, ``cut_check``
    match / mismatch / unknown (older table) / gapped;
  * the launcher -- ``p_prefill_graph_cut_check`` runs ONCE for all P ranks:
    matching cut keeps the table, a different cut falls back to EXACTLY the
    table-less form (argv, env, pool post, chunk spec) with the loud
    CUT-MISMATCH line, an older table warns and stays, no flag = no line and
    byte-identical; main() calls it right after solve_p_cut and re-solves.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest

from sglang.srt.weg2 import p_graph_policy as G
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WIDTHS = {"512": {"graph_ms": [41, 34, 38], "eager_ms": [45, 33, 37]},
          "2048": {"graph_ms": [150, 150, 150], "eager_ms": [161, 163, 168]}}


class TestCutField(unittest.TestCase):
    def test_parse_and_round_trip(self):
        for raw in ("43,11,10", [43, 11, 10], " 43, 11 ,10"):
            cal = G.Calibration.from_json({"widths": WIDTHS, "pp_layer_ratio": raw})
            self.assertEqual(cal.pp_layer_ratio, (43, 11, 10))
            self.assertEqual(cal.to_json()["pp_layer_ratio"], "43,11,10")
            self.assertEqual(G.Calibration.from_json(cal.to_json()), cal)
        for raw in (None, "", []):
            cal = G.Calibration.from_json({"widths": WIDTHS, "pp_layer_ratio": raw})
            self.assertEqual(cal.pp_layer_ratio, ())
        old = G.Calibration.from_json({"widths": WIDTHS})
        self.assertEqual(old.pp_layer_ratio, ())
        self.assertNotIn("pp_layer_ratio", old.to_json())  # older tables stay byte-identical
        for bad in ("43,x,10", "43,0,10", {"a": 1}):
            with self.assertRaises(G.GraphPolicyError):
                G.Calibration.from_json({"widths": WIDTHS, "pp_layer_ratio": bad})

    def test_cut_check(self):
        cal = G.Calibration.from_json({"widths": WIDTHS, "pp_layer_ratio": "43,11,10", "source": "gc"})
        v, line = G.cut_check(cal, (43, 11, 10))
        self.assertEqual(v, G.CUT_MATCH)
        self.assertIn("CUT-MATCH table=43,11,10 boot=43,11,10", line)
        v, line = G.cut_check(cal, (44, 10, 10))
        self.assertEqual(v, G.CUT_MISMATCH)
        self.assertTrue(line.startswith(
            "PREFILL-GRAPH-CALIBRATION CUT-MISMATCH table=43,11,10 boot=44,10,10 -> fallback"), line)
        # a gapped map is not named by its counts: never a match
        v, line = G.cut_check(cal, (43, 11, 10), gapped=True, layer_set="0-42|43-53|54-63")
        self.assertEqual(v, G.CUT_MISMATCH)
        self.assertIn("boot=gapped(0-42|43-53|54-63)", line)
        v, line = G.cut_check(G.Calibration.from_json({"widths": WIDTHS}), (44, 10, 10))
        self.assertEqual(v, G.CUT_UNKNOWN)
        self.assertIn("CUT-UNKNOWN WARNING", line)
        self.assertIn("boot=44,10,10", line)


try:
    from sglang.srt.weg2 import launcher as L
except Exception as exc:  # pragma: no cover
    L = None
    _L_ERR = exc

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
BUDGETS = [28904, 17704, 17672]


def _cut(counts, gapped=False, layer_set=""):
    return L.PCutFacts(
        stage_ratio="" if gapped else ",".join(map(str, counts)),
        attn_stage_ratio="" if gapped else "10,3,3", pool_tokens=300000.0,
        attn_counts=(10, 3, 3), kv_mib_per_token_per_attn_layer=2048 / (1024 * 1024),
        hidden_size=5120, cap_tokens=262144, layer_set=layer_set, gapped=gapped,
        layer_counts=tuple(counts))


@unittest.skipIf(L is None, "weg2 launcher unavailable")
class TestLauncherCutCheck(unittest.TestCase):
    BASE = ("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic", "--p-prefill-graph-buckets", "2048")

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
        return (argv, L.p_prefill_graph_captured(), L.p_prefill_graph_pool_mib(ns),
                L.p_prefill_graph_env(L.p_prefill_graph_pool_mib(ns)),
                L.p_graph_split_env(ns), L.p_chunk_policy_env(), L.p_chunked_prefill_tokens())

    def _table(self, **extra):
        path = os.path.join(self._tmp.name, "cal.json")
        with open(path, "w") as fh:
            json.dump(dict({"ref_prefix": 16384, "widths": WIDTHS, "source": "unit"}, **extra), fh)
        return path

    def test_matching_cut_keeps_the_table(self):
        ns = self._apply(*self.BASE, "--p-prefill-graph-calibration", self._table(pp_layer_ratio="42,11,11"))
        self.assertEqual(L.p_prefill_graph_captured(), [512, 2048])
        before = self._snapshot(ns)
        logs = []
        self.assertFalse(L.p_prefill_graph_cut_check(ns, _cut((42, 11, 11)), logs.append))
        self.assertEqual(self._snapshot(ns), before)
        self.assertEqual(len(logs), 1)
        self.assertIn("CUT-MATCH table=42,11,11 boot=42,11,11", logs[0])
        self.assertIn("CUT-MATCH", L.p_prefill_graph_policy_lines()[-1])
        self.assertIn("+graphcal:unit", L.p_chunk_policy_spec().source)

    def test_different_cut_falls_back_to_the_tableless_form(self):
        ns0 = self._apply(*self.BASE)
        tableless = self._snapshot(ns0)
        self.assertEqual(tableless[1], [512])
        ns = self._apply(*self.BASE, "--p-prefill-graph-calibration", self._table(pp_layer_ratio="43,11,10"))
        self.assertNotEqual(self._snapshot(ns), tableless)  # the table did change the form
        logs = []
        self.assertTrue(L.p_prefill_graph_cut_check(ns, _cut((44, 10, 10)), logs.append))
        self.assertEqual(len(logs), 1)
        self.assertTrue(logs[0].startswith(
            "WEG2 PREFILL-GRAPH-CALIBRATION CUT-MISMATCH table=43,11,10 boot=44,10,10 -> fallback"), logs[0])
        # exactly what a boot without the table ships: argv, capture set, pool post, env, chunk spec
        self.assertEqual(self._snapshot(ns), tableless)
        self.assertIsNone(L._P_PREFILL_GRAPH["calibration"])
        lines = L.p_prefill_graph_policy_lines()
        self.assertIn("calibration=none", lines[0])
        self.assertIn("CUT-MISMATCH", lines[-1])
        self.assertNotIn("graphcal", L.p_chunk_policy_spec().source)
        # the fallback is final: a second check has no table and says nothing
        self.assertFalse(L.p_prefill_graph_cut_check(ns, _cut((44, 10, 10)), logs.append))
        self.assertEqual(len(logs), 1)

    def test_mismatch_restores_a_graph_the_table_had_switched_off(self):
        path = self._table(pp_layer_ratio="43,11,10",
                           widths={"512": {"graph_ms": [46, 38, 40], "eager_ms": [44, 37, 39]}})
        ns = self._apply("--p-prefill-graph", "512", "--p-prefill-graph-calibration", path)
        self.assertEqual(L.p_prefill_graph_captured(), [])
        self.assertTrue(L.p_prefill_graph_cut_check(ns, _cut((42, 11, 11)), lambda _l: None))
        self.assertEqual(L.p_prefill_graph_captured(), [512])
        # gapped boot against a counts table: fallback too
        ns = self._apply("--p-prefill-graph", "512", "--p-prefill-graph-calibration", path)
        self.assertTrue(L.p_prefill_graph_cut_check(
            ns, _cut((43, 11, 10), gapped=True, layer_set="x"), lambda _l: None))

    def test_old_table_without_the_field_warns_and_stays(self):
        for extra in ({}, {"pp_layer_ratio": None}):  # graphcal_eval writes null when no arm named its cut
            ns = self._apply(*self.BASE, "--p-prefill-graph-calibration", self._table(**extra))
            before = self._snapshot(ns)
            logs = []
            self.assertFalse(L.p_prefill_graph_cut_check(ns, _cut((44, 10, 10)), logs.append))
            self.assertEqual(self._snapshot(ns), before)
            self.assertEqual(before[1], [512, 2048])
            self.assertEqual(len(logs), 1)
            self.assertIn("CUT-UNKNOWN WARNING", logs[0])
            self.assertIn("boot=44,10,10", logs[0])

    def test_no_flag_is_silent_and_byte_identical(self):
        for base in ((), self.BASE, ("--p-prefill-graph", "512")):
            ns = self._apply(*base)
            before = self._snapshot(ns)
            lines = L.p_prefill_graph_policy_lines()
            logs = []
            self.assertFalse(L.p_prefill_graph_cut_check(ns, _cut((44, 10, 10)), logs.append))
            self.assertEqual((logs, self._snapshot(ns), L.p_prefill_graph_policy_lines()),
                             ([], before, lines), base)
        # a table without the switch is never consulted (no bucket, no overlay)
        ns = self._apply("--p-prefill-graph-calibration", self._table(pp_layer_ratio="43,11,10"))
        logs = []
        self.assertFalse(L.p_prefill_graph_cut_check(ns, _cut((44, 10, 10)), logs.append))
        self.assertEqual(logs, [])

    def test_main_checks_the_shipped_cut_and_resolves(self):
        src = inspect.getsource(L.main)
        i_solve = src.index("cut = solve_p_cut(")
        i_check = src.index("if p_prefill_graph_cut_check(ns, cut, log):")
        i_resolve = src.index("cut = solve_p_cut(", i_check)
        i_use = src.index("stage_ratio, attn_stage_ratio = cut.stage_ratio")
        self.assertLess(i_solve, i_check)
        self.assertLess(i_check, i_resolve)
        self.assertLess(i_resolve, i_use)


if __name__ == "__main__":
    unittest.main()
