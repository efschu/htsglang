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
from unittest import mock

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

    def test_scenario_and_flush_routes(self):
        # #150 scenario expansion + cache-flush gate over real HTTP.
        d = self._post("/api/scenario", {
            "phases": "both", "concurrency": 2, "cold_prefill": True,
            "target_running_server": True})
        self.assertTrue(d["ok"])
        self.assertTrue(d["cache_flush_warning"]["mandatory"])  # running server
        self.assertEqual(d["summary"]["concurrency"], 2)
        w = self._post("/api/cache_flush_warning", {
            "will_flush": True, "target_running_server": False})
        self.assertTrue(w["warn"])
        self.assertFalse(w["mandatory"])            # fresh server = informative
        # live scrape of a nonexistent target fails gracefully (no raise).
        live = self._post("/api/live", {"target": "127.0.0.1:1"})
        self.assertFalse(live["ok"])
        self.assertIn("error", live)


class TestEnergyRoutePayloads(CustomTestCase):
    """The energy route adapters, called directly (no HTTP)."""

    def test_scenario_validation_error_is_carried(self):
        d = webui.scenario_payload({"phases": "nonsense"})
        self.assertFalse(d["ok"])
        self.assertIn("phases", d["error"])

    def test_index_has_energy_tab(self):
        self.assertIn("view_energy", webui.INDEX_HTML)
        self.assertIn("/api/gpu_state", webui.INDEX_HTML)
        self.assertIn("previewScenario", webui.INDEX_HTML)


class _FakeSupervisor:
    """A GPU-less stand-in for SglangSupervisor (no real sglang boot)."""

    def __init__(self):
        self.started = None
        self.stopped = False
        self.restarted = None
        self._running = False
        self.busy = False

    def is_running(self):
        return self._running

    def start(self, settings, wait_ready=False, argv=None):
        self.started = settings
        self._running = True
        return {"state": "booting", "pid": 4242, "port": settings.port}

    def stop(self, wait_vram=True):
        self.stopped = True
        self._running = False
        return {"vram_recovered": True, "group_gone": True}

    def restart(self, settings, wait_ready=False):
        if self.busy:
            from sglang.srt.planner.server_manager import SupervisorBusyError

            raise SupervisorBusyError("refusing restart: a live job is in-flight")
        self.restarted = settings
        self._running = True
        return {"state": "booting", "pid": 99, "port": settings.port}

    def status(self):
        return {
            "state": "ready" if self._running else "stopped",
            "pid": 4242 if self._running else None,
            "busy": self.busy,
            "port": self.started.port if self.started else None,
            "log_tail": "boot log line 1\nboot log line 2\n",
        }


class TestModelManagerRoutes(CustomTestCase):
    def setUp(self):
        self.fake = _FakeSupervisor()
        webui._set_supervisor(self.fake)

    def tearDown(self):
        webui._set_supervisor(None)

    def test_list_models_serializes(self):
        from sglang.srt.planner.server_manager import DiscoveredModel, GgufVariant

        fake_models = [
            DiscoveredModel(
                name="Qwen3.6-27B", path="/models/q", format="hf",
                quant_method="compressed-tensors", vision=False,
                size_bytes=14 * 2**30,
            ),
            DiscoveredModel(
                name="repo-gguf", path="/models/g", format="gguf",
                quant_method="gguf",
                gguf_variants=[
                    GgufVariant("Q4_K_M", "a.gguf", "/models/g/a.gguf", 2**30),
                    GgufVariant("Q6_K", "b.gguf", "/models/g/b.gguf", 2 * 2**30),
                ],
            ),
        ]
        with mock.patch(
            "sglang.srt.planner.server_manager.discover_models",
            return_value=fake_models,
        ):
            d = webui.list_models_payload()
        self.assertTrue(d["ok"])
        self.assertEqual(len(d["models"]), 2)
        gguf = d["models"][1]
        self.assertEqual(len(gguf["gguf_variants"]), 2)
        self.assertEqual(gguf["gguf_variants"][0]["quant"], "Q4_K_M")

    def test_server_start_builds_settings_and_calls_supervisor(self):
        d = webui.server_start_payload(
            {"model_path": "/models/q", "tp_size": 2, "rank_gpu_id": "0,1",
             "port": 31000, "spec_mode": "mtp"}
        )
        self.assertTrue(d["ok"], d.get("error"))
        self.assertIsNotNone(self.fake.started)
        self.assertEqual(self.fake.started.tp_size, 2)
        self.assertEqual(self.fake.started.rank_gpu_id, [0, 1])
        self.assertIn("launch_command", d)

    def test_server_start_validation_error_is_carried(self):
        # rank_gpu_id length != tp_size -> fail fast, no supervisor call.
        d = webui.server_start_payload(
            {"model_path": "/m", "tp_size": 4, "rank_gpu_id": "0,1"}
        )
        self.assertFalse(d["ok"])
        self.assertIn("rank-gpu-id length", d["error"])
        self.assertIsNone(self.fake.started)

    def test_restart_while_busy_is_refused(self):
        self.fake.busy = True
        d = webui.server_restart_payload({"model_path": "/m", "tp_size": 1})
        self.assertFalse(d["ok"])
        self.assertTrue(d["busy"])
        self.assertIsNone(self.fake.restarted)

    def test_stop_and_status(self):
        webui.server_start_payload({"model_path": "/m", "tp_size": 1})
        st = webui.server_status_payload()
        self.assertTrue(st["running"])
        d = webui.server_stop_payload({})
        self.assertTrue(d["ok"])
        self.assertTrue(self.fake.stopped)

    def test_download_gated_on_writability(self):
        with tempfile.TemporaryDirectory() as ro:
            # Writable temp dir -> button enabled.
            d = webui.download_targets_payload({"root": ro})
            self.assertTrue(d["writable"])
            self.assertIsNone(d["note"])
        # Nonexistent root -> not writable, note surfaced.
        d = webui.download_targets_payload({"root": "/no/such/mount"})
        self.assertFalse(d["writable"])
        self.assertIn("read-only", d["note"])

    def test_download_refused_when_not_writable(self):
        d = webui.model_download_payload(
            {"repo_id": "org/model", "root": "/no/such/mount"}
        )
        self.assertFalse(d["ok"])
        self.assertFalse(d.get("writable", True))


class TestPowerRoute(CustomTestCase):
    def test_measure_power_serializes_result(self):
        from sglang.srt.planner.power_calibration import (
            CardPowerMeasurement,
            PowerCalibrationResult,
        )

        result = PowerCalibrationResult(
            cards=[
                CardPowerMeasurement(
                    uuid="GPU-abc", name="RTX 5090", arch="sm120",
                    total_mib=32607, p_idle_w=25.0, p_membw_w=300.0,
                    p_gemm_w=450.0, membw_gbs=1400.0, gemm_tflops=180.0,
                    driver="580.00",
                )
            ],
            skipped=[{"name": "RTX 3080", "reason": "busy", "detail": "pid=123"}],
            driver="580.00", created="2026-07-21 10:00:00",
        )
        with mock.patch(
            "sglang.srt.planner.power_calibration.measure_all_cards",
            return_value=result,
        ) as m:
            d = webui.measure_power_payload({})
        self.assertTrue(m.called)
        self.assertTrue(d["ok"])
        self.assertEqual(len(d["cards"]), 1)
        self.assertEqual(d["cards"][0]["arch"], "sm120")
        self.assertEqual(len(d["skipped"]), 1)

    def test_power_profile_read(self):
        with mock.patch(
            "sglang.srt.planner.power_calibration.load_power_profile",
            return_value={},
        ):
            d = webui.power_profile_payload({})
        self.assertTrue(d["ok"])
        self.assertFalse(d["loaded"])


class TestQualityRoutes(CustomTestCase):
    _FAKE_SVG = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">'
        '<text x="0.5" y="0.5">K</text></svg>'
    )

    def _fake_resp(self, content):
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 200,
                      "total_tokens": 300},
        }

    def test_quality_run_calls_model_backend_side_only(self):
        # The model call goes through the backend _chat_completion seam; the
        # browser path (INDEX_HTML JS) must never POST to /chat/completions.
        self.assertNotIn("chat/completions", webui.INDEX_HTML)
        self.assertIn("/api/quality_run", webui.INDEX_HTML)

        captured = {}

        def fake_chat(endpoint, model, prompt, thinking, budget, **kw):
            captured["endpoint"] = endpoint
            captured["thinking"] = thinking
            captured["budget"] = budget
            return self._fake_resp("here:\n```svg\n" + self._FAKE_SVG + "\n```")

        with mock.patch.object(webui, "_chat_completion", side_effect=fake_chat), \
            mock.patch(
                "sglang.srt.planner.quality_chess.validate"
            ) as v:
            v.return_value.as_dict.return_value = {
                "verdict": "wrong-position", "report": "R", "piece_diff": [],
                "highlight_squares": ["h4"], "offer_download": False,
                "representation": "text-letter", "render_error": None,
            }
            d = webui.quality_run_payload(
                {"endpoint": "127.0.0.1:30000", "model": "m",
                 "thinking": True, "thinking_budget": 512}
            )
        self.assertTrue(d["ok"])
        self.assertTrue(v.called)  # graded by quality_chess.validate
        self.assertEqual(captured["thinking"], True)
        self.assertEqual(captured["budget"], 512)
        self.assertEqual(d["verdict"], "wrong-position")
        self.assertEqual(d["tokens"]["total"], 300)
        self.assertIn("<svg", d["svg"])

    def test_quality_run_no_svg_is_broken(self):
        with mock.patch.object(
            webui, "_chat_completion",
            return_value=self._fake_resp("I cannot draw that."),
        ):
            d = webui.quality_run_payload(
                {"endpoint": "e", "model": "m"}
            )
        self.assertTrue(d["ok"])
        self.assertIsNone(d["svg"])
        self.assertEqual(d["verdict"], "broken")
        self.assertTrue(d["offer_download"])

    def test_quality_run_requires_endpoint_and_model(self):
        self.assertFalse(webui.quality_run_payload({"model": "m"})["ok"])
        self.assertFalse(webui.quality_run_payload({"endpoint": "e"})["ok"])

    def test_quality_save_honors_toggle_and_shots_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shots.jsonl")
            # save toggle off -> nothing written.
            d = webui.quality_save_payload(
                {"save": False, "model": "m", "path": path, "verdict": "correct"}
            )
            self.assertFalse(d["saved"])
            self.assertFalse(os.path.exists(path))
            # save on -> appended.
            d = webui.quality_save_payload(
                {"save": True, "model": "m", "quant": "Q4_K_M", "path": path,
                 "verdict": "correct", "tokens": {"total": 5}, "svg": "<svg/>"}
            )
            self.assertTrue(d["saved"])
            shots = webui.quality_shots_payload({"path": path})
            self.assertTrue(shots["ok"])
            self.assertEqual(len(shots["shots"]), 1)
            self.assertEqual(shots["shots"][0]["verdict"], "correct")

    def test_quality_shots_missing_file_is_empty(self):
        d = webui.quality_shots_payload({"path": "/no/such/shots.jsonl"})
        self.assertTrue(d["ok"])
        self.assertEqual(d["shots"], [])


class TestNewTabsInIndex(CustomTestCase):
    def test_index_has_new_tabs_and_routes(self):
        for token in (
            "view_models", "view_quality", "/api/models", "/api/server_start",
            "/api/measure_power", "/api/quality_run", "/api/quality_shots",
            "/assets/quality_chess_reference.png",
            "1t53dhp",  # reddit reference link
        ):
            self.assertIn(token, webui.INDEX_HTML, token)


class TestNewHttpRoutes(WebUIFixture):
    """The new routes over a real in-process HTTP round-trip."""

    def setUp(self):
        self.fake = _FakeSupervisor()
        webui._set_supervisor(self.fake)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), webui._Handler)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        webui._set_supervisor(None)
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(timeout=5)

    def _get(self, path):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}{path}", timeout=10
        ) as r:
            return r.read(), r.headers.get("Content-Type")

    def _post(self, path, obj):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def test_server_status_route(self):
        body, ctype = self._get("/api/server_status")
        d = json.loads(body)
        self.assertTrue(d["ok"])
        self.assertFalse(d["running"])

    def test_reference_png_static_route(self):
        body, ctype = self._get("/assets/quality_chess_reference.png")
        self.assertEqual(ctype, "image/png")
        self.assertTrue(len(body) > 1000)
        self.assertEqual(body[:8], b"\x89PNG\r\n\x1a\n")

    def test_server_start_over_http(self):
        d = self._post("/api/server_start",
                       {"model_path": "/m", "tp_size": 1, "port": 31234})
        self.assertTrue(d["ok"], d.get("error"))
        self.assertIsNotNone(self.fake.started)


if __name__ == "__main__":
    unittest.main()
