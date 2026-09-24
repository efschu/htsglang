"""The `prefill graph pool` post (27B line, --p-prefill-graph).

The full prefill graph on group P is captured AFTER the KV pool is sized, so
its capture pool, static buffers and attention workspace need a budget post
or the KV pool takes those bytes and the capture pushes the rank past the
absolute budget the flip's credit ledger holds it to. The post lives on both
sides of one seam, and this file pins that they are the SAME post:

* the runtime (model_runner_kv_cache_mixin, absolute-budget branch) books
  ``prefill graph pool`` from ``SGLANG_KV_BUDGET_PREFILL_GRAPH_MIB``;
* the P cut's pool model (PhasePoolModel.prefill_graph_pool_mib) subtracts the
  same per-stage vector in the one post arithmetic every capacity reads
  (_stage_free_after_residency), and names it in RUNTIME_BUDGET_POSTS so the
  #1286 census can see it.

Empty = the switch is off: nothing booked, every price unchanged.
"""

import dataclasses
import os
import re
import unittest

from sglang.srt.planner import pp_cut
from sglang.srt.planner.pp_cut import PhasePoolModel, _stage_free_after_residency
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

COUNTS = (42, 11, 11)
ATTN = (10, 3, 3)


def _model(**kw):
    base = PhasePoolModel(
        free_mib=(26040.0, 15776.0, 15496.0),
        weight_mib_per_layer=340.0,
        kv_mib_per_token_per_attn_layer=2048.0 / (1024 * 1024),
        arming_floor_mib=(0.0, 0.0, 0.0),
        stage_fixed_mib=(2342.0, 1105.5, 3518.0),
        activation_reserve_mib=1024.0,
        corridor_holdback_mib=0.0,
        page_size=1,
    )
    return dataclasses.replace(base, **kw)


class TestGraphPoolPost(CustomTestCase):
    def test_off_changes_nothing(self):
        self.assertEqual(
            _stage_free_after_residency(COUNTS, ATTN, _model()),
            _stage_free_after_residency(COUNTS, ATTN, _model(prefill_graph_pool_mib=())),
        )

    def test_on_subtracts_exactly_the_per_stage_vector(self):
        base = _stage_free_after_residency(COUNTS, ATTN, _model())
        post = (160.0, 150.0, 140.0)
        charged = _stage_free_after_residency(
            COUNTS, ATTN, _model(prefill_graph_pool_mib=post)
        )
        for b, c, p in zip(base, charged, post):
            self.assertAlmostEqual(b - c, p)

    def test_a_broadcast_vector_is_refused(self):
        with self.assertRaises(ValueError):
            _stage_free_after_residency(
                COUNTS, ATTN, _model(prefill_graph_pool_mib=(160.0,))
            )

    def test_the_post_lowers_the_priced_pool(self):
        base = pp_cut.pp_phase_pool(COUNTS, ATTN, _model())
        charged = pp_cut.pp_phase_pool(
            COUNTS, ATTN, _model(prefill_graph_pool_mib=(160.0,) * 3)
        )
        self.assertLess(charged, base)

    def test_a_legitimate_zero_is_not_unfunded(self):
        self.assertNotIn("prefill_graph_pool_mib", _model().unfunded_posts)


class TestTheTwoSidesNameOnePost(CustomTestCase):
    def _runtime_src(self):
        here = os.path.dirname(os.path.abspath(pp_cut.__file__))
        path = os.path.join(
            here, "..", "model_executor", "model_runner_kv_cache_mixin.py"
        )
        with open(path) as f:
            return f.read()

    def test_the_runtime_books_it_under_this_name_from_this_env(self):
        src = self._runtime_src()
        self.assertRegex(src, r"budget_posts\.append\(\(\"prefill graph pool\"")
        self.assertIn('PREFILL_GRAPH_POOL_ENV = "SGLANG_KV_BUDGET_PREFILL_GRAPH_MIB"', src)
        # read from the env constant, per rank, in the absolute-budget branch
        self.assertRegex(src, r"os\.environ\.get\(PREFILL_GRAPH_POOL_ENV")

    def test_the_model_mirrors_the_runtime_name(self):
        names = {n for n, _f in PhasePoolModel.RUNTIME_BUDGET_POSTS}
        self.assertIn("prefill graph pool", names)
        fields = dict(PhasePoolModel.RUNTIME_BUDGET_POSTS)
        self.assertEqual(fields["prefill graph pool"], "prefill_graph_pool_mib")

    def test_the_launcher_hands_one_vector_to_both(self):
        from sglang.srt.weg2 import launcher as L

        src = open(L.__file__).read()
        # the pool model and the env are built from the same call
        self.assertIn("prefill_graph_pool_mib=p_prefill_graph_pool_mib(ns)", src)
        self.assertIn("env_p.update(p_prefill_graph_env(_pg_pool))", src)
        self.assertTrue(re.search(r"_pg_pool = p_prefill_graph_pool_mib\(ns\)", src))


if __name__ == "__main__":
    unittest.main()
