"""--pp-cut-stage-fit, --pp-cut-depth-profile, --p-deep-split-from: the launcher half.

27B line, 24.09. All three default OFF (parser defaults, empty environment).
Pinned without a GPU or a launch:

* the POWER RECORD (user order 24.09.: the cards run power-limited, the
  limits may be raised, a measurement at one limit must not be reused at
  another silently): every boot prints its P cards' limits; a fit whose
  record matches the metal is used, a STALE one is refused (W40) unless
  --pp-cut-stage-fit-stale-ok, a record-less one is refused unless the
  operator asserts the limits (--pp-cut-stage-fit-power);
* a fit at another chunk size is refused (the attention cost is chunk-specific);
* --p-deep-split-from N gives group P the two wave-split variables, 0 none.
"""

import os
import tempfile
import types
import unittest

import pytest

try:
    from sglang.srt.weg2 import launcher as L  # noqa: F401
except Exception as exc:  # pragma: no cover - no weg2 launcher in this build
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

CARDS = [
    L.Card(1, "GPU-a", "NVIDIA GeForce RTX 5090", 32607),
    L.Card(0, "GPU-b", "NVIDIA GeForce RTX 3080", 20480),
    L.Card(2, "GPU-c", "NVIDIA GeForce RTX 3080", 20480),
]
LINES = [(41.4, 1.4358), (33.65, 1.0389), (37.86, 1.0568)]


def _p_log(dirpath, chunk=512):
    out = [
        "[t] server_args=ServerArgs(model_path='m', chunked_prefill_size=%d, "
        "pp_stage_ratio=[42, 11, 11], pp_attn_stage_ratio=[10, 3, 3], x=1)\n" % chunk
    ]
    fct = [0, 0, 0]
    for i, p in enumerate((40960, 81920)):
        pre = 0
        while pre < p - 1:
            ext = min(chunk, p - 1 - pre)
            for r in range(3):
                a, b = LINES[r]
                gpu = a + b * (pre + chunk / 2) / 1000.0 if ext == chunk else 9.0
                out.append(
                    "[t PP%d] #969N ADMIT slot=0 fwd_ct=%d bs=1 extend=%d input_ids=None "
                    "rids=[weg2-%d-%d]\n[t PP%d] #PGAP pp_rank=%d fwd=%d tokens=%d "
                    "gpu_gap_ms=0.2 gpu_fwd_ms=%.4f host[plan=1 launch=2 process=1] "
                    "plan_parts[anchor=0] overlap=1\n"
                    % (r, fct[r], ext, i, i, r, r, fct[r] + 1, ext, gpu)
                )
                fct[r] += 1
            pre += ext
    path = os.path.join(dirpath, "boot_weg2_weg2xsnT_abc_0924_000000.P.log")
    with open(path, "w") as fh:
        fh.write("".join(out))
    return path


def _front(p_log, limits):
    path = p_log[: -len(".P.log")] + ".front.log"
    with open(path, "w") as fh:
        fh.write("[t] WEG2-LAUNCH %s\n" % L.card_power_line(CARDS, limits))
    return path


def _ns(**kw):
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class TestDefaultsOff(CustomTestCase):
    def test_parser_defaults(self):
        ns = _ns()
        self.assertEqual(ns.pp_cut_stage_fit, "")
        self.assertEqual(ns.pp_cut_stage_fit_power, "")
        self.assertFalse(ns.pp_cut_stage_fit_stale_ok)
        self.assertEqual(ns.pp_cut_depth_profile, "")
        self.assertEqual(ns.p_deep_split_from, 0)

    def test_deep_split_env(self):
        self.assertEqual(L.p_deep_split_env(_ns()), {})
        env = L.p_deep_split_env(_ns(p_deep_split_from=10240))
        self.assertEqual(
            env,
            {"SGLANG_FI_PREFILL_WAVE_SPLIT": "1",
             "SGLANG_FI_PREFILL_WAVE_SPLIT_FROM_PREFIX": "10240"},
        )
        with self.assertRaises(SystemExit):
            L.p_deep_split_env(_ns(p_deep_split_from=-1))


class TestPowerRecord(CustomTestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.p_log = _p_log(self.dir)
        self.lines = []

    def _fit(self, now=(400.0, 230.0, 230.0), **kw):
        return L.stage_fit_family_cost(
            _ns(pp_cut_stage_fit=self.p_log, **kw), CARDS, 512, list(now),
            self.lines.append)

    def test_line_round_trips(self):
        front = _front(self.p_log, [400.0, 230.0, None])
        self.assertEqual(
            L.read_card_power_record(front),
            [("NVIDIA GeForce RTX 5090", 400.0), ("NVIDIA GeForce RTX 3080", 230.0),
             ("NVIDIA GeForce RTX 3080", None)],
        )
        self.assertIsNone(L.read_card_power_record(self.p_log))

    def test_matching_record_is_used(self):
        _front(self.p_log, [400.0, 230.0, 230.0])
        cost, prov, fitted = self._fit()
        self.assertIn("power MATCH", prov)
        self.assertEqual(fitted.chunk_tokens, 512)
        self.assertAlmostEqual(cost.layer_ms[1], 33.65 / 11, places=3)
        self.assertTrue(any("power MATCH" in x for x in self.lines))

    def test_stale_record_is_refused(self):
        _front(self.p_log, [600.0, 230.0, 230.0])
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            self._fit()
        self.assertIn("STALE", str(cm.exception))
        self.assertIn("600 W -> 400 W", str(cm.exception))

    def test_stale_ok_prices_and_says_so(self):
        _front(self.p_log, [600.0, 230.0, 230.0])
        _cost, prov, _ = self._fit(pp_cut_stage_fit_stale_ok=True)
        self.assertIn("power STALE", prov)

    def test_no_record_is_refused_unless_asserted(self):
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            self._fit()
        self.assertIn("no power record", str(cm.exception))
        _cost, prov, _ = self._fit(pp_cut_stage_fit_power="400,230,230")
        self.assertIn("ASSERTED", prov)
        self.assertIn("power MATCH", prov)
        with self.assertRaises(L.Weg2LaunchRefused):
            self._fit(pp_cut_stage_fit_power="400,320,320")

    def test_other_chunk_size_is_refused(self):
        _front(self.p_log, [400.0, 230.0, 230.0])
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.stage_fit_family_cost(
                _ns(pp_cut_stage_fit=self.p_log), CARDS, 4096,
                [400.0, 230.0, 230.0], self.lines.append)
        self.assertIn("chunk", str(cm.exception))


if __name__ == "__main__":
    unittest.main()


class TestGraphSplitFlag(CustomTestCase):
    """--p-prefill-graph-split: default off; inert without the prefill graph."""

    def setUp(self):
        self._saved = dict(L._P_PREFILL_GRAPH)

    def tearDown(self):
        L._P_PREFILL_GRAPH.clear()
        L._P_PREFILL_GRAPH.update(self._saved)

    def test_default_off_and_inert_without_the_graph(self):
        ns = _ns()
        self.assertEqual(ns.p_prefill_graph_split, 0)
        L.apply_p_prefill_graph(ns)
        self.assertEqual(L.p_graph_split_env(ns), {})
        ns = _ns(p_prefill_graph_split=7)
        L.apply_p_prefill_graph(ns)
        self.assertEqual(L.p_graph_split_env(ns), {})

    def test_on_with_the_graph(self):
        ns = _ns(p_prefill_graph=512, p_prefill_graph_split=7)
        L.apply_p_prefill_graph(ns)
        self.assertEqual(L.p_graph_split_env(ns), {"SGLANG_FI_PREFILL_GRAPH_SPLIT": "7"})
        with self.assertRaises(SystemExit):
            L.p_graph_split_env(_ns(p_prefill_graph=512, p_prefill_graph_split=1))
