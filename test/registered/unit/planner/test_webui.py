"""CPU smoke tests for the S3 web-UI backend (design §7-S3 / §2.6 / §6).

No GPU, no network bind, no server boot: the API functions are exercised
directly (``plan_from_payload`` / ``issue_from_payload`` / ``discover_knobs``)
plus one in-process HTTP round-trip on an ephemeral loopback port to prove
the handler wiring, then the server is torn down. The UI is a thin client, so
testing the API IS testing the UI's contract.
"""

import json
import os
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from sglang.srt.planner import webui
from sglang.srt.planner.hardware import hardware_from_manual  # noqa: F401
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

RIG = ["RTX 5090:32607", "RTX 3080:20480", "RTX 3080:20480"]

_CONFIG = {
    "architectures": ["Qwen3NextForCausalLM"],
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 48,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "vocab_size": 151936,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
    "quantization_config": {"group_size": 32},
}


def _make_model(tmpdir):
    path = os.path.join(tmpdir, "model")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(_CONFIG, f)
    with open(os.path.join(path, "model-00001.safetensors"), "wb") as f:
        f.truncate(int(14 * 2**30))
    return path


class WebUIFixture(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = _make_model(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _payload(self, **kw):
        p = {
            "model": self.model,
            "hardware": {"source": "manual", "gpus": RIG},
            "tp_size": 3,
            "quant": "compressed-tensors",
        }
        p.update(kw)
        return p


class TestKnobDiscovery(CustomTestCase):
    def test_feature_detect_intersects_server_args(self):
        d = webui.discover_knobs()
        ids = {k["id"] for k in d["knobs"]}
        # Present in this build's ServerArgs -> offered.
        for expected in (
            "rank_gpu_id",
            "rank_gpu_memory_mib",
            "rank_tp_ratio",
            "rank_mlp_ratio",
            "rank_moe_ratio",
            "rank_vocab_ratio",
            "dcp_size",
        ):
            self.assertIn(expected, ids, expected)
        # Planner-only external headroom is always offered.
        self.assertIn("plan_free_reserve_gb", ids)

    def test_absent_server_args_knob_is_not_offered(self):
        # rank_kv_ratio / weightless-KV live on other branches; the catalog
        # must not surface a knob whose ServerArgs field is missing.
        from sglang.srt.server_args import ServerArgs
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ServerArgs)}
        for k in webui.discover_knobs()["knobs"]:
            if k["server_arg"] is not None:
                self.assertIn(k["server_arg"], fields, k["id"])

    def test_not_expressible_notes_present(self):
        d = webui.discover_knobs()
        self.assertTrue(d["not_expressible"])
        joined = " ".join(d["not_expressible"]).lower()
        self.assertIn("per-layer", joined)


class TestPlanAPI(WebUIFixture):
    def test_auto_plan_fits(self):
        d = webui.plan_from_payload(self._payload())
        self.assertTrue(d["valid"])
        self.assertTrue(d["fits"], d.get("infeasible_reasons"))
        self.assertIn("--rank-gpu-id 0,1,2", " ".join(d["launch_flags"]))
        self.assertGreater(d["capacity"]["max_context_tokens"], 0)
        # honest advantage: stock cannot shard 4 KV heads across 3.
        self.assertFalse(d["advantage"]["stock"]["runs"])

    def test_concurrency_table_matches_headline_context(self):
        # Regression: the KV-vs-concurrency table must report the AGGREGATE
        # context (== the headline max_context_tokens and == the per-rank sum),
        # not the min per-rank capacity (which understated it ~4x and read as a
        # contradiction, e.g. 97k in the table vs 408k in the header).
        d = webui.plan_from_payload(self._payload())
        self.assertTrue(d["fits"], d.get("infeasible_reasons"))
        headline = d["capacity"]["max_context_tokens"]
        per_rank_sum = sum(rc["kv_tokens"] for rc in d["capacity"]["per_rank"])
        self.assertAlmostEqual(headline, per_rank_sum, delta=max(2, headline * 0.001))
        row1 = next(r for r in d["kv_by_concurrency"] if r["concurrency"] == 1)
        self.assertAlmostEqual(
            row1["kv_tokens"], headline, delta=max(2, headline * 0.001)
        )

    def test_fp8_kv_roughly_doubles_context(self):
        # KV-cache quantization must flow into sizing: fp8 (1 B/cell) yields
        # ~2x the max context of auto (~2 B/cell) for the same VRAM.
        auto = webui.plan_from_payload(self._payload(kv_cache_dtype="auto"))
        fp8 = webui.plan_from_payload(self._payload(kv_cache_dtype="fp8_e4m3"))
        if not (auto["fits"] and fp8["fits"]):
            self.skipTest("model does not fit on the test rig")
        self.assertGreater(
            fp8["capacity"]["max_context_tokens"],
            1.7 * auto["capacity"]["max_context_tokens"],
        )

    def test_manual_edit_reject_carries_reason(self):
        d = webui.plan_from_payload(
            self._payload(
                rank_gpu_id="0,1,2",
                rank_gpu_memory_mib="20000,15000,15000",
                rank_tp_ratio="4,3",  # wrong length
            )
        )
        self.assertFalse(d["valid"])
        self.assertTrue(any("--rank-tp-ratio length" in r for r in d["reasons"]))

    def test_free_reserve_lowers_capacity(self):
        base = webui.plan_from_payload(self._payload())
        carved = webui.plan_from_payload(
            self._payload(plan_free_reserve_gb="4,2,2")
        )
        self.assertLess(
            carved["capacity"]["max_context_tokens"],
            base["capacity"]["max_context_tokens"],
        )

    def test_scalar_free_reserve_applies_to_all_cards(self):
        d = webui.plan_from_payload(self._payload(plan_free_reserve_gb="2"))
        self.assertTrue(d["valid"])
        self.assertTrue(d["fits"])

    def test_no_estimated_throughput_field_in_response(self):
        # Honesty (design §3.4, refined by #145): the capacity + advantage
        # sections carry NO throughput/tok-s field (they are memory quantities).
        # A throughput ESTIMATE exists ONLY in the separate, clearly-labelled
        # roofline_estimate block, which is provenance-tagged planner-estimate
        # and never admissible into the measured store.
        d = webui.plan_from_payload(self._payload())
        for section in ("capacity", "advantage", "offload"):
            blob = json.dumps(d[section]).lower()
            for bad in ("tok_s", "tokps", "throughput", "decode_tok", "tok/s"):
                self.assertNotIn(bad, blob)
        # measured perf is absent (manual hardware, no cached profile).
        self.assertIsNone(d["advantage"]["measured"])
        # The roofline estimate IS present, and structurally separate/labelled.
        rf = d["roofline_estimate"]
        self.assertIsNotNone(rf)
        self.assertEqual(rf["provenance"], "planner-estimate")
        self.assertIn("decode_tok_s_low", rf)
        self.assertFalse(rf["measured_available"])


class TestIssueAPI(WebUIFixture):
    def test_results_issue_from_ui(self):
        d = webui.issue_from_payload(self._payload(kind="results", group_size=32))
        self.assertTrue(d["ok"])
        self.assertEqual(d["kind"], "results")
        self.assertIn("## htsglang result", d["markdown"])
        self.assertTrue(d["url"].startswith(
            "https://github.com/efschu/htsglang/issues/new?"
        ))
        # scrubbed: the model dir never leaks.
        self.assertNotIn("/tmp", d["markdown"])

    def test_bug_issue_from_ui(self):
        d = webui.issue_from_payload(
            self._payload(kind="bug", symptom="NCCL hang after boot")
        )
        self.assertTrue(d["ok"])
        self.assertIn("## htsglang bug", d["markdown"])
        self.assertIn("NCCL hang after boot", d["markdown"])


class TestHttpRoundTrip(WebUIFixture):
    """One real in-process HTTP round-trip on an ephemeral loopback port."""

    def setUp(self):
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), webui._Handler)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(timeout=5)

    def _get(self, path):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}{path}", timeout=10
        ) as r:
            return r.read().decode()

    def _post(self, path, obj):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def test_index_and_knobs_and_plan(self):
        self.assertIn("htsglang offline config planner", self._get("/"))
        knobs = self._get("/api/knobs")
        self.assertIn("rank_gpu_id", knobs)
        d = self._post("/api/plan", self._payload())
        self.assertTrue(d["valid"])
        self.assertTrue(d["fits"])
        issue = self._post("/api/issue", self._payload(kind="results"))
        self.assertTrue(issue["ok"])


if __name__ == "__main__":
    unittest.main()
