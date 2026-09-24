"""--p-prefill-graph-tiny (27B line, 24.09.): extra small prefill graph buckets.

Measured motive (xsn430/xsn433 #PGAP, graph bs [512]): every P prompt ends
with the 1-token END-OF-PREFILL ANCHOR chunk (schedule_policy.
_weg2_end_anchor_split), and the runner pads it to the only bucket -- 37.1 /
35.0 / 38.0 ms per stage for one token, against ~41 / 34 / 38 ms for a real
512 chunk. A 16-token bucket replays weight-bandwidth bound instead.

Pinned without a GPU: default off is byte-identical (bs [512], same pool
post); on, the capture list is [tiny..., bucket] ascending, the chunk stays
the main bucket, the budget post adds the proportional estimate, the runner's
own padding helper sends a 1-token batch to the tiny bucket, and bad values
are refused before any argv is built.
"""

import json
import unittest

import pytest

try:
    from sglang.srt.weg2 import launcher as L  # noqa: F401
except Exception as exc:  # pragma: no cover
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)

from sglang.srt.model_executor.cuda_graph_config import parse_cuda_graph_config_arg
from sglang.srt.model_executor.runner.base_cuda_graph_runner import BaseCudaGraphRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Case(CustomTestCase):
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

    def _bs(self):
        flags = L.p_prefill_graph_flags()
        cfg = json.loads(flags[flags.index("--cuda-graph-config") + 1])
        return cfg["prefill"]["bs"]


class TestDefaultOff(_Case):
    def test_parser_default_and_byte_identical_graph_form(self):
        ns = self._apply(p_prefill_graph=512)
        self.assertEqual(ns.p_prefill_graph_tiny, "")
        self.assertEqual(L.p_prefill_graph_buckets(), [512])
        self.assertEqual(self._bs(), [512])
        self.assertIn('"bs":[512]', L.p_prefill_graph_flags()[1])
        self.assertEqual(L.p_prefill_graph_pool_mib(ns), (160.0, 160.0, 160.0))

    def test_graph_off_means_no_buckets(self):
        self._apply()
        self.assertEqual(L.p_prefill_graph_buckets(), [])
        self.assertEqual(L.p_prefill_graph_flags(), [])


class TestTinyOn(_Case):
    def test_capture_list_chunk_and_post(self):
        ns = self._apply(p_prefill_graph=512, p_prefill_graph_tiny="16")
        self.assertEqual(L.p_prefill_graph_buckets(), [16, 512])
        self.assertEqual(self._bs(), [16, 512])
        self.assertEqual(L.p_chunked_prefill_tokens(), 512)
        self.assertAlmostEqual(L.p_prefill_graph_pool_mib(ns)[0], 160.0 * 528 / 512)
        self.assertIn("tiny bucket(s) [16]", L.p_prefill_graph_line())
        # the runtime parser accepts the list
        flags = L.p_prefill_graph_flags()
        cfg = parse_cuda_graph_config_arg(flags[flags.index("--cuda-graph-config") + 1])
        self.assertEqual(sorted(cfg["prefill"]["bs"]), [16, 512])

    def test_measured_pool_still_wins(self):
        ns = self._apply(p_prefill_graph=512, p_prefill_graph_tiny="16", p_prefill_graph_pool_mib="210")
        self.assertEqual(L.p_prefill_graph_pool_mib(ns), (210.0, 210.0, 210.0))

    def test_several_tiny_buckets_are_sorted_and_deduped(self):
        self._apply(p_prefill_graph=512, p_prefill_graph_tiny="128,16,16")
        self.assertEqual(L.p_prefill_graph_buckets(), [16, 128, 512])

    def test_the_anchor_token_replays_the_tiny_bucket(self):
        buckets = [16, 512]
        self.assertEqual(BaseCudaGraphRunner._pad_to_bucket(1, buckets), 16)
        self.assertEqual(BaseCudaGraphRunner._pad_to_bucket(16, buckets), 16)
        self.assertEqual(BaseCudaGraphRunner._pad_to_bucket(17, buckets), 512)
        self.assertEqual(BaseCudaGraphRunner._pad_to_bucket(512, buckets), 512)


class TestRefusals(_Case):
    def test_bad_values(self):
        for kw in (
            dict(p_prefill_graph_tiny="16"),  # no main bucket
            dict(p_prefill_graph=512, p_prefill_graph_tiny="512"),
            dict(p_prefill_graph=512, p_prefill_graph_tiny="600"),
            dict(p_prefill_graph=512, p_prefill_graph_tiny="0"),
            dict(p_prefill_graph=512, p_prefill_graph_tiny="x"),
        ):
            with self.assertRaises(SystemExit, msg=str(kw)):
                self._apply(**kw)


if __name__ == "__main__":
    unittest.main()
