"""--p-prefill-graph (27B line, 2026-09-24): the launcher half.

User goal: group P on 512-token chunks, and for the fixed chunk a prefill
CUDA graph ("ja unbedingt"). Measured motive (xsn422 #PGAP): one P forward's
host launch is 63/42/74 ms per stage, ~24 ms fixed + ~0.9 ms per layer and
independent of the token count, against ~48-71 ms of compute per 512 chunk.

Pinned here, all without a GPU or a launch:

* OFF (the default) is byte-identical: no graph term in argv_p, P's chunk
  stays the common constant, P's environment gains nothing, the cut's pool
  model gets no graph post.
* ON is ONE switch that states three coupled facts: the full prefill graph
  (--cuda-graph-config, backend 'full', one bucket, one request slot), P's
  chunk = the bucket (a smaller bucket would replay only the rest chunk), and
  the 'prefill graph pool' budget post -- ONE vector booked by the ranks
  (env) and by the P cut's pool model, never two numbers.
* Group D is untouched either way.
"""

import json
import types
import unittest

import pytest

try:
    from sglang.srt.weg2 import launcher as L  # noqa: F401
except Exception as exc:  # pragma: no cover - no weg2 launcher in this build
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)

from sglang.srt.model_executor.cuda_graph_config import parse_cuda_graph_config_arg
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
BUDGETS = [28904, 17704, 17672]


def _flag_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def _argv_p():
    return L.argv_p(
        "py", MODEL, BUDGETS, 8, 1024, "8.0", [],
        stage_ratio="42,11,11", attn_stage_ratio="10,3,3",
    )


def _argv_d():
    return L.argv_d("py", MODEL, BUDGETS, 8, 1024, "8.0", [])


class _SwitchCase(CustomTestCase):
    def setUp(self):
        self._saved = dict(L._P_PREFILL_GRAPH)

    def tearDown(self):
        L._P_PREFILL_GRAPH.clear()
        L._P_PREFILL_GRAPH.update(self._saved)

    def _apply(self, **kw):
        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
        for k, v in kw.items():
            setattr(ns, k, v)
        L.apply_p_prefill_graph(ns)
        return ns


class TestOffIsByteIdentical(_SwitchCase):
    def test_parser_default_is_off(self):
        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
        self.assertEqual(ns.p_prefill_graph, 0)
        self.assertEqual(ns.p_prefill_graph_pool_mib, "")

    def test_off_ships_no_graph_term_and_the_common_chunk(self):
        before = _argv_p()
        ns = self._apply()
        after = _argv_p()
        self.assertEqual(before, after)
        self.assertNotIn("--cuda-graph-config", after)
        self.assertEqual(_flag_value(after, "--chunked-prefill-size"), str(L.CHUNKED_PREFILL_TOKENS))
        self.assertEqual(L.p_prefill_graph_flags(), [])
        self.assertEqual(L.p_prefill_graph_pool_mib(ns), ())
        self.assertEqual(L.p_prefill_graph_env(()), {})


class TestOnStatesTheCoupledForm(_SwitchCase):
    def test_graph_config_is_full_one_bucket_one_slot(self):
        self._apply(p_prefill_graph=512)
        p = _argv_p()
        raw = _flag_value(p, "--cuda-graph-config")
        self.assertIsNotNone(raw)
        cfg = json.loads(raw)
        self.assertEqual(
            cfg, {"prefill": {"backend": "full", "bs": [512], "full_prefill_max_req": 1}}
        )
        # the runtime's own parser accepts exactly this token
        parsed = parse_cuda_graph_config_arg(raw)
        self.assertEqual(parsed["prefill"]["backend"], "full")

    def test_p_chunk_follows_the_bucket_and_d_does_not(self):
        self._apply(p_prefill_graph=512)
        self.assertEqual(_flag_value(_argv_p(), "--chunked-prefill-size"), "512")
        d = _argv_d()
        self.assertEqual(_flag_value(d, "--chunked-prefill-size"), str(L.CHUNKED_PREFILL_TOKENS))
        self.assertNotIn("--cuda-graph-config", d)

    def test_the_cut_solver_reads_the_same_chunk(self):
        """chunked_prefill_size_of(common_flags(..., 'P')) is what solve_p_cut
        and its armed re-check read; it must be the shipped chunk."""
        self._apply(p_prefill_graph=512)
        flags = L.common_flags(MODEL, 8, 1024, "8.0", 262144, "write_back", "P")
        self.assertEqual(L.chunked_prefill_size_of(flags), 512)

    def test_operator_extra_p_still_wins_the_graph_config(self):
        self._apply(p_prefill_graph=512)
        p = L.argv_p(
            "py", MODEL, BUDGETS, 8, 1024, "8.0",
            ["--cuda-graph-config", '{"prefill":{"backend":"disabled"}}'],
            stage_ratio="42,11,11", attn_stage_ratio="10,3,3",
        )
        # argparse keeps the LAST occurrence: the extra, after ours
        last = len(p) - 1 - p[::-1].index("--cuda-graph-config")
        self.assertEqual(json.loads(p[last + 1])["prefill"]["backend"], "disabled")

    def test_one_pool_vector_for_both_sides(self):
        ns = self._apply(p_prefill_graph=512)
        pool = L.p_prefill_graph_pool_mib(ns)
        self.assertEqual(pool, (L.P_PREFILL_GRAPH_POOL_MIB_PER_512,) * 3)
        env = L.p_prefill_graph_env(pool)
        self.assertEqual(
            env["SGLANG_KV_BUDGET_PREFILL_GRAPH_MIB"],
            ",".join("%.1f" % v for v in pool),
        )
        self.assertEqual(env["SGLANG_FULL_CG_PREFILL_SHARED_WORKSPACE"], "1")

    def test_measured_values_replace_the_estimate(self):
        ns = self._apply(p_prefill_graph=512, p_prefill_graph_pool_mib="131.5,120,118")
        self.assertEqual(L.p_prefill_graph_pool_mib(ns), (131.5, 120.0, 118.0))
        ns = self._apply(p_prefill_graph=512, p_prefill_graph_pool_mib="140")
        self.assertEqual(L.p_prefill_graph_pool_mib(ns), (140.0, 140.0, 140.0))
        for bad in ("1,2", "-5"):
            with self.assertRaises(SystemExit):
                L.p_prefill_graph_pool_mib(
                    self._apply(p_prefill_graph=512, p_prefill_graph_pool_mib=bad)
                )

    def test_the_estimate_scales_with_the_bucket(self):
        ns = self._apply(p_prefill_graph=1024)
        self.assertEqual(
            L.p_prefill_graph_pool_mib(ns), (2 * L.P_PREFILL_GRAPH_POOL_MIB_PER_512,) * 3
        )

    def test_negative_bucket_is_refused(self):
        with self.assertRaises(SystemExit):
            self._apply(p_prefill_graph=-1)

    def test_the_line_names_the_form(self):
        self._apply(p_prefill_graph=512)
        line = L.p_prefill_graph_line()
        self.assertIn("WEG2 P-PREFILL-GRAPH: on bucket=512", line)
        self.assertIn("--chunked-prefill-size 512", line)

    def test_the_form_key_moves(self):
        """The chunk and the graph config are device allocations: an ON boot
        must never price its ring off an OFF boot's image."""
        from sglang.srt.weg2 import ring_table

        off_key, _ = ring_table.p_form_key(_argv_p())
        self._apply(p_prefill_graph=512)
        on_key, _ = ring_table.p_form_key(_argv_p())
        self.assertNotEqual(off_key, on_key)


if __name__ == "__main__":
    unittest.main()
