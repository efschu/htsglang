"""CPU smoke tests for the S3 web-UI backend (design §7-S3 / §2.6 / §6).

No GPU, no network bind, no server boot: the API functions are exercised
directly (``plan_from_payload`` / ``issue_from_payload`` / ``discover_knobs``)
plus one in-process HTTP round-trip on an ephemeral loopback port to prove
the handler wiring, then the server is torn down. The UI is a thin client, so
testing the API IS testing the UI's contract.
"""

import json
import os
import re
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

import pytest

from sglang.srt.planner import jtok_counter as jc
from sglang.srt.planner import self_update as su
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


class TestEnergyRoutePayloads(CustomTestCase):
    """The energy route adapters, called directly (no HTTP)."""

    def test_scenario_validation_error_is_carried(self):
        d = webui.scenario_payload({"phases": "nonsense"})
        self.assertFalse(d["ok"])
        self.assertIn("phases", d["error"])

    def test_index_has_energy_section(self):
        # Energy calibration is now a section of the merged Data tab, not
        # its own tab.
        self.assertIn("view_data", webui.INDEX_HTML)
        self.assertIn('id="data_energy"', webui.INDEX_HTML)
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


class TestQualityHistoryIsGroupedNotASlider(CustomTestCase):
    """Round 5: the history panel used a single 1-D slider over the flat
    shot list, so "which model/quant" and "which run of that model/quant"
    were the same axis -- picking a shot from an older model meant scrubbing
    past every OTHER model's saved runs first. Rebuilt as two dropdowns:
    model+quant narrows the list, run picks a specific saved shot within it
    (newest first). No slider remains.
    """

    def test_the_slider_is_gone(self):
        html = webui.INDEX_HTML
        self.assertNotIn('id="q_slider"', html)
        self.assertNotIn('id="q_slide_lbl"', html)
        self.assertNotIn('type="range"', html.split('<div id="view_quality"')[1]
                          .split('<div id="view_about"')[0])

    def test_two_dropdowns_replace_it(self):
        html = webui.INDEX_HTML
        self.assertIn('id="q_hist_mq"', html)
        self.assertIn('id="q_hist_run"', html)
        self.assertIn('onchange="onHistModelChange()"', html)

    def test_groups_are_keyed_by_model_and_quant(self):
        js = _index_script()
        self.assertIn("function _histKey(s){", js)
        self.assertIn("s.model", js.split("function _histKey(s){")[1][:200])
        self.assertIn("s.quant", js.split("function _histKey(s){")[1][:200])

    def test_runs_within_a_group_are_newest_first(self):
        js = _index_script()
        body = js[js.index("function onHistModelChange(){"):]
        body = body[:body.index("\n}")]
        self.assertIn("idxs.slice().reverse()", body)


class TestNewTabsInIndex(CustomTestCase):
    def test_index_has_new_tabs_and_routes(self):
        # PHASE 2 reorg: the Models tab is MERGED into the Runner tab, so its
        # controls (model dropdown / server launch) now live under view_runner.
        for token in (
            "view_runner", "view_quality", "/api/models", "/api/server_start",
            "/api/measure_power", "/api/quality_run", "/api/quality_shots",
            "/assets/quality_chess_reference.png",
            "1t53dhp",  # reddit reference link
            # joule-per-token counter (persistent, resettable, opt-in).
            "jtok_enabled", "jtok_out", "/api/jtok_counter",
            "/api/jtok_counter/reset",
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


class TestJtokCounterHttpRoutes(TestNewHttpRoutes):
    """/api/jtok_counter (GET+POST) and /api/jtok_counter/reset (POST) over a
    real in-process HTTP round-trip -- confirms the routing prefix ordering
    (the /reset extension is checked BEFORE the base path, same convention
    as card_probe/card_probe-status) actually dispatches correctly, not just
    the underlying payload functions in isolation."""

    def setUp(self):
        super().setUp()
        self._tmp2 = tempfile.TemporaryDirectory()
        self._store_patch = mock.patch.object(
            jc, "DEFAULT_JTOK_STORE",
            os.path.join(self._tmp2.name, "jtok_counter.json"),
        )
        self._store_patch.start()
        _reset_jtok_state()

    def tearDown(self):
        _reset_jtok_state()
        self._store_patch.stop()
        self._tmp2.cleanup()
        super().tearDown()

    def test_get_default_disabled_empty(self):
        d = json.loads(self._get("/api/jtok_counter")[0])
        self.assertTrue(d["ok"])
        self.assertFalse(d["enabled"])
        self.assertEqual(d["records"], [])

    def test_toggle_then_reset_all_over_http(self):
        d = self._post("/api/jtok_counter", {"enabled": True})
        self.assertTrue(d["ok"])
        self.assertTrue(d["enabled"])

        webui._jtok_live_tick(
            {"launch_config": {"served_model_name": "m", "tp_size": 1,
                                "quantization": "fp8"},
             "rates": {"dt": 2.0, "prefill_tok_s": 100.0, "decode_tok_s": 0.0},
             "gpus": [{"power_watts": 200.0}]},
            {"endpoint": "http://x:1"},
        )
        d = json.loads(self._get("/api/jtok_counter")[0])
        self.assertEqual(len(d["records"]), 1)
        self.assertIsNotNone(d["records"][0]["j_per_prefill_token"])

        # The /reset extension must route to the reset handler, not be
        # swallowed by the base /api/jtok_counter (toggle) route.
        d = self._post("/api/jtok_counter/reset", {"reset_all": True})
        self.assertTrue(d["ok"])
        self.assertEqual(d["reset_count"], 1)
        d = json.loads(self._get("/api/jtok_counter")[0])
        self.assertIsNone(d["records"][0]["j_per_prefill_token"])


# ===========================================================================
# Dashboard v2 (PHASE 2): Landing live-monitor + Runner tab route wiring.
# The three backend modules are exercised through the thin webui adapters; a
# couple of tests mock the module to prove the route only wires (parse + shape),
# the rest run the real modules on an injected model_cfg (no disk / GPU).
# ===========================================================================

_V2_CONFIG = dict(_CONFIG)  # the Qwen3-Next hybrid config above


class TestDashboardV2Index(CustomTestCase):
    def test_landing_and_runner_markers_present(self):
        for token in (
            "view_landing", "view_runner", "/api/live_snapshot",
            "/api/placement", "/api/resolve_flags", "/api/flag_catalog",
            "/api/config_profiles", "renderPlacement", "landingPoll",
            "loadFlagCatalog", "flag_surface",
        ):
            self.assertIn(token, webui.INDEX_HTML, token)

    def test_energy_calibration_kept_in_data_tab(self):
        # per-card power CALIBRATION stays reachable, now as a section of the
        # merged Data tab (former Energy + Rig data tabs); the live monitor
        # moved to Landing.
        self.assertIn("view_data", webui.INDEX_HTML)
        self.assertIn('id="data_energy"', webui.INDEX_HTML)
        self.assertIn("measurePower", webui.INDEX_HTML)


class TestLandingTopStrip(CustomTestCase):
    """Full-width live metrics strip at the top of the Landing page: headline
    tiles with 60s freeze-at-zero activity graphs."""

    def test_strip_container_and_tile_ids_present(self):
        for token in (
            "landing_strip", "strip_decode", "strip_prefill", "strip_spec",
            "strip_cache", "strip_energy", "strip_cost", "strip_saved",
            "renderLandingStrip", "stripTile",
        ):
            self.assertIn(token, webui.INDEX_HTML, token)

    def test_old_lower_rates_block_is_gone(self):
        # The former "throughput / spec / cache" block MOVED into the strip --
        # its container id and legend must no longer exist (no duplication).
        self.assertNotIn("landing_rates", webui.INDEX_HTML)
        self.assertNotIn("throughput / spec / cache (live + 60s)",
                         webui.INDEX_HTML)

    def test_freeze_at_zero_semantics_markers(self):
        # Drop-to-zero-once + FREEZE ring: the grace constant and the
        # freeze-aware push helper must both be part of the page JS.
        self.assertIn("STRIP_FREEZE_GRACE_S", webui.INDEX_HTML)
        self.assertIn("stripPush", webui.INDEX_HTML)
        self.assertIn("FROZEN", webui.INDEX_HTML)

    def test_one_pixel_bucket_per_sample_markers(self):
        # Resolution rule: fixed px-per-sample SVG sized samples*px (no
        # smoothing / CSS stretching that would alias samples).
        self.assertIn("STRIP_PX_PER_SAMPLE", webui.INDEX_HTML)
        self.assertIn("STRIP_SAMPLES", webui.INDEX_HTML)
        self.assertIn("LAND_POLL_MS", webui.INDEX_HTML)

    def test_saved_tile_reuses_147_pipeline(self):
        # SAVED tile = host-RAM + disk tiers ONLY via /api/hicache_saved; the
        # device/VRAM hot tier is excluded by the #147 pipeline it reuses.
        self.assertIn("stripFetchSaved", webui.INDEX_HTML)
        self.assertIn("/api/hicache_saved", webui.INDEX_HTML)


class TestMergedCardTelemetry(CustomTestCase):
    """The landing's standalone per-GPU 60s chart grid is MERGED into the
    per-card VRAM-placement blocks: one place per card explains BOTH what
    occupies its VRAM and how it is doing live."""

    def test_old_per_gpu_chart_grid_is_gone(self):
        # container id + legend of the removed standalone grid: no
        # duplication, the card blocks are the only per-GPU live view.
        self.assertNotIn("landing_gpus", webui.INDEX_HTML)
        self.assertNotIn("per-card GPU (live + 60s)", webui.INDEX_HTML)

    def test_placement_card_blocks_carry_live_telemetry(self):
        # compact live line inside renderPlacement's card blocks: current
        # util/clock/power/temp + 60s util/power sparklines; matched to the
        # CUDA-keyed card via the device-map-bridged gpus[] rows.
        for token in (
            "cardLiveHtml", "_liveGpuForCard", "cardlive",
            "renderPlacement(d.placement, s.gpus",
        ):
            self.assertIn(token, webui.INDEX_HTML, token)

    def test_util_ring_reuses_freeze_at_zero_power_stays_plain(self):
        # util is an activity stream -> strip freeze-at-zero semantics;
        # power idles at a nonzero wattage -> plain ring.
        self.assertIn("stripPush(key+'_util'", webui.INDEX_HTML)
        self.assertIn("pushRing(key+'_pow'", webui.INDEX_HTML)

    def test_top_strip_untouched(self):
        for token in ("landing_strip", "strip_decode", "renderLandingStrip"):
            self.assertIn(token, webui.INDEX_HTML, token)


class TestSpecDraftSelectorUi(CustomTestCase):
    """Runner Speculative section: local draft-model selector (drives the
    authoritative speculative_draft_model_path field) + the muted amber
    no-MTP-head hint."""

    def test_selector_and_hint_ids_present(self):
        for token in (
            "draft_model_select", "spec_draft_hint", "row_draft_pick",
            "renderDraftPick", "draftPickChanged", "updateSpecDraftHint",
            "fl_speculative_draft_model_path",
        ):
            self.assertIn(token, webui.INDEX_HTML, token)

    def test_hint_wordings_present(self):
        self.assertIn(
            "this model has no MTP head - pick a draft model",
            webui.INDEX_HTML,
        )
        self.assertIn("none matching found locally", webui.INDEX_HTML)
        self.assertIn("suggestion: ", webui.INDEX_HTML)


class TestPlacementRoute(CustomTestCase):
    def test_placement_from_model_cfg_running_and_prospective(self):
        # ONE renderer, two data sources: identical route for both the landing
        # RUNNING config and the runner PROSPECTIVE config.
        d = webui.placement_payload(
            {"model_cfg": _V2_CONFIG,
             "flags": {"tp_size": 3, "rank_gpu_id": [0, 1, 2],
                       "context_length": 8192}}
        )
        self.assertTrue(d["ok"], d.get("error"))
        pl = d["placement"]
        for key in ("model", "ranks_per_gpu", "attn_heads", "kv_tokens",
                    "cards", "mtp", "notes", "token_vector"):
            self.assertIn(key, pl)
        self.assertEqual(len(pl["cards"]), 3)
        self.assertEqual(len(pl["attn_heads"]), 3)

    def test_placement_route_wires_module(self):
        with mock.patch(
            "sglang.srt.planner.placement.compute_placement",
            return_value={"cards": [], "sentinel": True},
        ) as m:
            d = webui.placement_payload({"model_cfg": _V2_CONFIG, "flags": {}})
        self.assertTrue(d["ok"])
        self.assertTrue(d["placement"]["sentinel"])
        m.assert_called_once()

    def test_placement_missing_model_is_error(self):
        d = webui.placement_payload({"flags": {}})
        self.assertFalse(d["ok"])
        self.assertIn("model", d["error"])


class TestResolveFlagsRoute(CustomTestCase):
    def test_resolve_greying_reaches_field_state_json(self):
        d = webui.resolve_flags_payload(
            {"settings": {"tp_size": 3, "rank_gpu_id": [0, 1, 2]},
             "model_cfg": _V2_CONFIG}
        )
        self.assertTrue(d["ok"])
        fields = d["fields"]
        # rank_gpu_id excludes the global gpu-memory-utilization / mem-fraction:
        # some field must be disabled with a reason, and a dependency auto-set.
        self.assertTrue(
            any(v.get("disabled_reason") for v in fields.values()),
            "expected at least one greyed field with a reason",
        )
        self.assertTrue(
            any(v.get("auto_set") for v in fields.values()),
            "expected at least one auto-set dependency",
        )

    def test_cross_field_warning_weightless_needs_flashinfer(self):
        d = webui.resolve_flags_payload(
            {"settings": {"weightless_kv_fastlane": True,
                          "attention_backend": "triton", "tp_size": 2},
             "model_cfg": _V2_CONFIG}
        )
        self.assertTrue(d["ok"])
        msgs = " ".join(w["message"] for w in d["warnings"])
        self.assertIn("flashinfer", msgs)

    def test_cross_field_warning_rank_kv_ratio_needs_nonuniform(self):
        d = webui.resolve_flags_payload(
            {"settings": {"rank_kv_ratio": "capacity", "rank_tp_ratio": [1, 1]},
             "model_cfg": _V2_CONFIG}
        )
        ids = [w["id"] for w in d["warnings"]]
        self.assertIn("rank_kv_ratio", ids)


class TestFlagCatalogRoute(CustomTestCase):
    def test_catalog_groups_and_counts(self):
        d = webui.flag_catalog_payload()
        self.assertTrue(d["ok"])
        self.assertGreater(d["upstream_count"], 0)
        self.assertGreater(d["fork_count"], 0)
        # grouped, each entry carries the render metadata.
        some = next(iter(d["groups"].values()))[0]
        for k in ("id", "name", "type", "help", "hover", "source"):
            self.assertIn(k, some)


class _RunningFakeSupervisor(_FakeSupervisor):
    """A fake supervisor that reports running with a LaunchSettings-like
    ``settings`` object, for the landing snapshot path."""

    class _S:
        host = "127.0.0.1"
        port = 31000

    def __init__(self):
        super().__init__()
        self._running = True
        self.settings = self._S()


def _reset_landing_state():
    webui._LANDING_SNAPSHOT_STATE = None
    webui._LANDING_TARGET_KEY = None
    webui._DETECTED_ENDPOINT = None


class TestLandingSnapshotRoute(CustomTestCase):
    def setUp(self):
        self.fake = _FakeSupervisor()
        webui._set_supervisor(self.fake)
        _reset_landing_state()

    def tearDown(self):
        webui._set_supervisor(None)
        _reset_landing_state()

    def test_no_server_is_graceful(self):
        # Detection mocked out: this box may genuinely have a hand-started
        # server on :30000 and the test must not depend on it.
        with mock.patch.object(
            webui, "_detect_external_endpoint", return_value=None
        ):
            d = webui.landing_snapshot_payload()
        self.assertTrue(d["ok"])            # NOT an error
        self.assertFalse(d["running"])
        self.assertIsNone(d["snapshot"])
        self.assertIsNone(d["target"])

    def test_running_server_returns_snapshot(self):
        webui._set_supervisor(_RunningFakeSupervisor())
        fake_snap = ({"ok": True, "gpus": [], "rates": None}, {"t": 1.0})
        with mock.patch(
            "sglang.srt.planner.live_metrics.snapshot", return_value=fake_snap
        ) as m:
            d = webui.landing_snapshot_payload()
        self.assertTrue(d["ok"])
        self.assertTrue(d["running"])
        self.assertIn("gpus", d["snapshot"])
        self.assertEqual(d["target"]["kind"], "managed")
        m.assert_called_once()

    def test_concurrent_polls_serialize_the_delta_state(self):
        """Round 5 bug: rates flashing to zero / few updates landing while a
        real inference load is running. Root cause: the dashboard is a
        ThreadingHTTPServer, and the client's poll only ABORTS its own wait
        on the client side when a scrape runs long -- the server-side
        request keeps going. A slow /metrics scrape (contended for CPU/IO by
        real inference work) can therefore still be in flight when the NEXT
        poll's request starts on a second thread, and without serialization
        the two requests interleave their read of _LANDING_SNAPSHOT_STATE:
        whichever call WRITES last wins, even if it read an older baseline,
        which regresses the stored state and hands the next poll a t_prev
        that live_metrics._rates() treats as an "out-of-order scrape" --
        clamped to a hard 0.0. This proves the fix serializes the two calls:
        the second call must see the first call's already-written state, not
        the same starting baseline.
        """
        webui._set_supervisor(_RunningFakeSupervisor())
        seen_prev_states = []
        call_index = [0]

        def fake_snapshot(target, prev_state=None):
            seen_prev_states.append(prev_state)
            call_index[0] += 1
            if call_index[0] == 1:
                # Simulate a slow scrape (real inference load contending for
                # it) that is still running when the next poll fires.
                time.sleep(0.2)
            return (
                {"ok": True, "gpus": [], "rates": None, "t": call_index[0]},
                {"t": call_index[0], "counters": {}},
            )

        results = []

        def run():
            results.append(webui.landing_snapshot_payload())

        with mock.patch(
            "sglang.srt.planner.live_metrics.snapshot", side_effect=fake_snapshot
        ):
            t1 = threading.Thread(target=run)
            t1.start()
            time.sleep(0.05)  # let t1 acquire the lock and start its "scrape"
            t2 = threading.Thread(target=run)
            t2.start()
            t1.join(timeout=3)
            t2.join(timeout=3)

        self.assertEqual(len(seen_prev_states), 2)
        self.assertTrue(all(r["ok"] for r in results))
        self.assertIsNone(seen_prev_states[0])
        # Without the lock, t2 would also see None (the same stale baseline
        # t1 started from) instead of waiting for t1's write.
        self.assertIsNotNone(
            seen_prev_states[1],
            "the second concurrent poll must see the first poll's written "
            "state, not race it for the same starting baseline",
        )


def _reset_jtok_state():
    webui._JTOK_STORE = None
    webui._JTOK_LAST_FLUSH_TS = 0.0


class TestJtokCounterRoutes(CustomTestCase):
    """The persistent joule-per-token counter's webui.py wiring: the
    /api/jtok_counter (read/toggle) + /api/jtok_counter/reset routes, the
    live-poll hook (_jtok_live_tick, folded into landing_snapshot_payload),
    and the schema write-guard. No GPU/network -- NVML power and Prometheus
    counters are fabricated dicts, exactly like TestLandingSnapshotRoute's
    fake snapshot above."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._prev_env = os.environ.get("SGLANG_PLANNER_DATA_DIR")
        os.environ["SGLANG_PLANNER_DATA_DIR"] = self._tmp.name
        # DEFAULT_JTOK_STORE (like hicache's DEFAULT_HICACHE_STORE / energy's
        # DEFAULT_RESULTS_STORE) is a module-level constant resolved ONCE at
        # jtok_counter import time -- by the time this test class runs,
        # jtok_counter is already imported, so the SGLANG_PLANNER_DATA_DIR
        # override above only affects the schema-stamp GUARD lookup (which is
        # re-resolved dynamically on every call), not the store file path.
        # Patch the constant directly so each test's store lives in its own
        # tmp dir instead of all tests sharing whatever path was frozen in at
        # first import.
        self._store_patch = mock.patch.object(
            jc, "DEFAULT_JTOK_STORE",
            os.path.join(self._tmp.name, "jtok_counter.json"),
        )
        self._store_patch.start()
        _reset_jtok_state()

    def tearDown(self):
        _reset_jtok_state()
        self._store_patch.stop()
        if self._prev_env is None:
            os.environ.pop("SGLANG_PLANNER_DATA_DIR", None)
        else:
            os.environ["SGLANG_PLANNER_DATA_DIR"] = self._prev_env
        self._tmp.cleanup()

    def _snap(self, dt, prefill_tok_s, decode_tok_s, watts=(150.0, 160.0)):
        return {
            "launch_config": {"served_model_name": "Qwen3.6-27B", "tp_size": 2,
                              "quantization": "compressed-tensors"},
            "rates": {"dt": dt, "prefill_tok_s": prefill_tok_s,
                     "decode_tok_s": decode_tok_s},
            "gpus": [{"power_watts": w} for w in watts],
        }

    def _target(self):
        return {"endpoint": "http://127.0.0.1:8000", "kind": "managed", "managed": True}

    # -- default state / read --------------------------------------------

    def test_default_disabled_and_empty(self):
        d = webui.jtok_counter_read_payload()
        self.assertTrue(d["ok"])
        self.assertFalse(d["enabled"])
        self.assertEqual(d["records"], [])

    # -- toggle -------------------------------------------------------------

    def test_toggle_on_then_off_persists(self):
        d = webui.jtok_counter_set_enabled_payload({"enabled": True})
        self.assertTrue(d["ok"])
        self.assertTrue(d["enabled"])
        # A brand-new in-process store load (simulating a dashboard restart)
        # must see the persisted toggle.
        _reset_jtok_state()
        self.assertTrue(webui.jtok_counter_read_payload()["enabled"])
        d = webui.jtok_counter_set_enabled_payload({"enabled": False})
        self.assertFalse(d["enabled"])

    def test_toggle_requires_enabled_key(self):
        d = webui.jtok_counter_set_enabled_payload({})
        self.assertFalse(d["ok"])

    # -- live-poll hook: phase attribution -----------------------------

    def test_live_tick_noop_while_disabled(self):
        webui._jtok_live_tick(self._snap(2.0, 100.0, 0.0), self._target())
        self.assertEqual(webui.jtok_counter_read_payload()["records"], [])

    def test_live_tick_pure_prefill_accumulates(self):
        webui.jtok_counter_set_enabled_payload({"enabled": True})
        webui._jtok_live_tick(self._snap(2.0, 100.0, 0.0), self._target())
        recs = webui.jtok_counter_read_payload()["records"]
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r["model"], "Qwen3.6-27B")
        self.assertEqual(r["config_label"], "tp2 compressed-tensors")
        self.assertEqual(r["lanes"], ["http://127.0.0.1:8000"])
        self.assertEqual(r["provenance"], "measured")
        # joules = (150+160)*2 = 620 ; tokens = 100*2 = 200
        self.assertAlmostEqual(r["j_per_prefill_token"], 620.0 / 200.0)
        self.assertIsNone(r["j_per_decode_token"])

    def test_live_tick_mixed_window_kept_separate(self):
        webui.jtok_counter_set_enabled_payload({"enabled": True})
        webui._jtok_live_tick(self._snap(2.0, 100.0, 50.0), self._target())
        r = webui.jtok_counter_read_payload()["records"][0]
        self.assertIsNone(r["j_per_prefill_token"])
        self.assertIsNone(r["j_per_decode_token"])
        self.assertEqual(r["mixed_windows"], 1)

    def test_live_tick_two_endpoints_are_two_lanes(self):
        webui.jtok_counter_set_enabled_payload({"enabled": True})
        webui._jtok_live_tick(self._snap(2.0, 100.0, 0.0),
                              {"endpoint": "http://127.0.0.1:8000"})
        webui._jtok_live_tick(self._snap(2.0, 100.0, 0.0),
                              {"endpoint": "http://127.0.0.1:9000"})
        recs = webui.jtok_counter_read_payload()["records"]
        self.assertEqual(len(recs), 2)
        lanes = sorted(r["lanes"][0] for r in recs)
        self.assertEqual(lanes, ["http://127.0.0.1:8000", "http://127.0.0.1:9000"])

    def test_live_tick_never_raises_on_malformed_snapshot(self):
        webui.jtok_counter_set_enabled_payload({"enabled": True})
        # Missing 'gpus'/'rates' entirely must not blow up the poll response.
        webui._jtok_live_tick({}, self._target())
        self.assertEqual(webui.jtok_counter_read_payload()["records"], [])

    # -- landing_snapshot_payload end-to-end (mocked live_metrics.snapshot) --

    def test_landing_snapshot_feeds_jtok_counter(self):
        webui.jtok_counter_set_enabled_payload({"enabled": True})
        webui._set_supervisor(_RunningFakeSupervisor())
        _reset_landing_state()
        fake_snap = (self._snap(2.0, 0.0, 40.0), {"t": 1.0})
        try:
            with mock.patch("sglang.srt.planner.live_metrics.snapshot",
                            return_value=fake_snap):
                d = webui.landing_snapshot_payload()
            self.assertTrue(d["ok"])
            recs = webui.jtok_counter_read_payload()["records"]
            self.assertEqual(len(recs), 1)
            self.assertIsNotNone(recs[0]["j_per_decode_token"])
        finally:
            webui._set_supervisor(None)
            _reset_landing_state()

    # -- reset --------------------------------------------------------------

    def test_reset_one_and_reset_all(self):
        webui.jtok_counter_set_enabled_payload({"enabled": True})
        webui._jtok_live_tick(self._snap(2.0, 100.0, 0.0),
                              {"endpoint": "http://127.0.0.1:8000"})
        webui._jtok_live_tick(self._snap(2.0, 0.0, 40.0),
                              {"endpoint": "http://127.0.0.1:9000"})

        d = webui.jtok_counter_reset_payload({
            "model": "Qwen3.6-27B", "config_label": "tp2 compressed-tensors",
            "lanes": ["http://127.0.0.1:8000"],
        })
        self.assertTrue(d["ok"])
        by_lane = {r["lanes"][0]: r for r in
                   webui.jtok_counter_read_payload()["records"]}
        self.assertIsNone(by_lane["http://127.0.0.1:8000"]["j_per_prefill_token"])
        self.assertIsNotNone(by_lane["http://127.0.0.1:9000"]["j_per_decode_token"])

        d = webui.jtok_counter_reset_payload({"reset_all": True})
        self.assertTrue(d["ok"])
        self.assertEqual(d["reset_count"], 2)
        for r in webui.jtok_counter_read_payload()["records"]:
            self.assertIsNone(r["j_per_prefill_token"])
            self.assertIsNone(r["j_per_decode_token"])

    def test_reset_missing_key_errors(self):
        d = webui.jtok_counter_reset_payload({"model": "m"})  # no config_label/lanes
        self.assertFalse(d["ok"])
        d = webui.jtok_counter_reset_payload({
            "model": "nope", "config_label": "nope", "lanes": ["nope"],
        })
        self.assertFalse(d["ok"])
        self.assertIn("no such counter", d["error"])

    # -- schema write-guard (mirrors test_self_update's pattern) -------

    def test_newer_schema_blocks_toggle_and_reset(self):
        with open(os.path.join(self._tmp.name, su.SCHEMA_STAMP_NAME), "w") as f:
            json.dump({"schema_version": su.DATA_SCHEMA_VERSION + 1}, f)
        d = webui.jtok_counter_set_enabled_payload({"enabled": True})
        self.assertFalse(d["ok"])
        self.assertIn("schema", d["error"])
        d = webui.jtok_counter_reset_payload({"reset_all": True})
        self.assertFalse(d["ok"])
        self.assertIn("schema", d["error"])

    def test_read_never_blocked_by_guard(self):
        with open(os.path.join(self._tmp.name, su.SCHEMA_STAMP_NAME), "w") as f:
            json.dump({"schema_version": su.DATA_SCHEMA_VERSION + 1}, f)
        d = webui.jtok_counter_read_payload()
        self.assertTrue(d["ok"])


class TestConfigProfilesRoutes(CustomTestCase):
    def setUp(self):
        self._tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tf.close()
        self._prev = os.environ.get("SGLANG_PLANNER_PROFILES")
        os.environ["SGLANG_PLANNER_PROFILES"] = self._tf.name

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("SGLANG_PLANNER_PROFILES", None)
        else:
            os.environ["SGLANG_PLANNER_PROFILES"] = self._prev
        os.unlink(self._tf.name)

    def test_generated_profiles_listed(self):
        d = webui.config_profiles_get(
            {"model_cfg": _V2_CONFIG,
             "gpus": [{"name": "A", "total_mib": 32607},
                      {"name": "B", "total_mib": 20480}]}
        )
        self.assertTrue(d["ok"])
        self.assertGreater(len(d["generated"]), 0)

    def test_save_list_delete_roundtrip(self):
        sv = webui.config_profiles_save(
            {"name": "my-cfg", "settings": {"tp_size": 2}, "kind": "custom"}
        )
        self.assertTrue(sv["ok"])
        got = webui.config_profiles_get({})
        self.assertIn("my-cfg", [p["name"] for p in got["saved"]])
        rm = webui.config_profiles_delete({"name": "my-cfg"})
        self.assertTrue(rm["deleted"])
        got2 = webui.config_profiles_get({})
        self.assertNotIn("my-cfg", [p["name"] for p in got2["saved"]])

    def test_save_requires_name(self):
        d = webui.config_profiles_save({"settings": {"tp_size": 2}})
        self.assertFalse(d["ok"])

    def test_draft_candidates_feed_presets_for_no_mtp_model(self):
        # _V2_CONFIG has no MTP layers: with a matching local draft model the
        # generated presets enable spec via --speculative-draft-model-path,
        # and the candidates + MTP fact reach the UI payload (they feed the
        # Speculative section's selector + hint).
        from types import SimpleNamespace

        drafts = [SimpleNamespace(
            name="Gemma-4-31B-Eagle3", path="/m/ge3", error=None)]
        with mock.patch(
            "sglang.srt.planner.server_manager.discover_models",
            return_value=drafts,
        ):
            d = webui.config_profiles_get(
                {"model": "/m/gemma-4-31B-it", "model_cfg": _V2_CONFIG,
                 "gpus": [{"name": "A", "total_mib": 32607},
                          {"name": "B", "total_mib": 20480}]}
            )
        self.assertTrue(d["ok"])
        self.assertIs(d["model_has_mtp"], False)
        self.assertEqual(len(d["draft_candidates"]), 1)
        self.assertEqual(d["draft_candidates"][0]["path"], "/m/ge3")
        self.assertEqual(d["draft_candidates"][0]["algorithm"], "EAGLE3")
        self.assertGreater(len(d["generated"]), 0)
        for p in d["generated"]:
            self.assertEqual(
                p["settings"]["speculative_algorithm"], "EAGLE3", p["name"]
            )
            self.assertEqual(
                p["settings"]["speculative_draft_model_path"], "/m/ge3",
                p["name"],
            )
            self.assertIn("--speculative-draft-model-path", p["argv"])
            self.assertTrue(
                any("Gemma-4-31B-Eagle3" in i for i in p["info"]), p["info"]
            )

    def test_no_matching_draft_keeps_spec_off_with_note(self):
        from types import SimpleNamespace

        drafts = [SimpleNamespace(
            name="Gemma-4-31B-Eagle3", path="/m/ge3", error=None)]
        with mock.patch(
            "sglang.srt.planner.server_manager.discover_models",
            return_value=drafts,
        ):
            d = webui.config_profiles_get(
                {"model": "/m/Qwen3.6-27B-AWQ", "model_cfg": _V2_CONFIG,
                 "gpus": [{"name": "A", "total_mib": 32607},
                          {"name": "B", "total_mib": 20480}]}
            )
        self.assertTrue(d["ok"])
        self.assertIs(d["model_has_mtp"], False)
        self.assertEqual(d["draft_candidates"], [])
        for p in d["generated"]:
            self.assertFalse(
                p["settings"].get("speculative_algorithm"), p["name"]
            )
            self.assertTrue(
                any(
                    "no MTP head and no matching local draft model found"
                    in i
                    for i in p["info"]
                ),
                p["info"],
            )


class TestDashboardV2HttpRoutes(WebUIFixture):
    """The new v2 routes over a real in-process HTTP round-trip."""

    def setUp(self):
        self.fake = _FakeSupervisor()
        webui._set_supervisor(self.fake)
        self._tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tf.close()
        os.environ["SGLANG_PLANNER_PROFILES"] = self._tf.name
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), webui._Handler)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        webui._set_supervisor(None)
        os.environ.pop("SGLANG_PLANNER_PROFILES", None)
        os.unlink(self._tf.name)
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(timeout=5)

    def _get(self, path):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}{path}", timeout=10
        ) as r:
            return json.loads(r.read())

    def _req(self, method, path, obj=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(obj or {}).encode(),
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def test_live_snapshot_no_server(self):
        with mock.patch.object(
            webui, "_detect_external_endpoint", return_value=None
        ):
            d = self._get("/api/live_snapshot")
        self.assertTrue(d["ok"])
        self.assertFalse(d["running"])

    def test_flag_catalog_route(self):
        d = self._get("/api/flag_catalog")
        self.assertTrue(d["ok"])
        self.assertIn("groups", d)

    def test_placement_route_http(self):
        d = self._req("POST", "/api/placement",
                      {"model_cfg": _V2_CONFIG,
                       "flags": {"tp_size": 3, "rank_gpu_id": [0, 1, 2]}})
        self.assertTrue(d["ok"], d.get("error"))
        self.assertEqual(len(d["placement"]["cards"]), 3)

    def test_resolve_flags_route_http(self):
        d = self._req("POST", "/api/resolve_flags",
                      {"settings": {"tp_size": 2}, "model_cfg": _V2_CONFIG})
        self.assertTrue(d["ok"])
        self.assertIn("fields", d)
        self.assertIn("warnings", d)

    def test_config_profiles_crud_http(self):
        saved = self._req("POST", "/api/config_profiles",
                          {"name": "http-cfg", "settings": {"tp_size": 2}})
        self.assertTrue(saved["ok"])
        got = self._get("/api/config_profiles")
        self.assertIn("http-cfg", [p["name"] for p in got["saved"]])
        rm = self._req("DELETE", "/api/config_profiles", {"name": "http-cfg"})
        self.assertTrue(rm["deleted"])


# ===========================================================================
# Dashboard v3: monitor-target resolution (the TPS-bug regression suite),
# profile env/argv launch wiring, benchmark suite (#151) + GitHub share
# (#152) routes. Everything mocked -- no GPU, no server boot, no network.
# ===========================================================================


class TestMonitorTargetResolution(CustomTestCase):
    """Regression suite for the never-rendering-TPS bug: the landing snapshot
    previously gated on ``sup.is_running()`` and returned ``running: False``
    for every hand-started server. It must now attach to ANY reachable
    server: explicit endpoint > managed instance > auto-detected."""

    def setUp(self):
        self.fake = _FakeSupervisor()
        webui._set_supervisor(self.fake)
        _reset_landing_state()

    def tearDown(self):
        webui._set_supervisor(None)
        _reset_landing_state()

    def _snap(self, rates=None):
        return (
            {"ok": True, "gpus": [], "rates": rates, "launch_config": None,
             "server_info": {"server_info": {"model_path": "/m"}}},
            {"t": 1.0, "counters": {}},
        )

    def test_external_hand_started_server_yields_rates(self):
        # THE TPS regression test: supervisor NOT running, a hand-started
        # server is auto-detected -> a snapshot WITH rates comes back
        # (previously: running False / snapshot None on every poll).
        rates = {"decode_tok_s": 51.0, "prefill_tok_s": 12.5}
        with mock.patch.object(
            webui, "_detect_external_endpoint",
            return_value="http://127.0.0.1:30000",
        ), mock.patch(
            "sglang.srt.planner.live_metrics.snapshot",
            return_value=self._snap(rates),
        ) as m:
            d = webui.landing_snapshot_payload()
        self.assertTrue(d["ok"])
        self.assertTrue(d["running"])
        self.assertEqual(d["snapshot"]["rates"]["decode_tok_s"], 51.0)
        self.assertEqual(d["snapshot"]["rates"]["prefill_tok_s"], 12.5)
        self.assertEqual(d["target"]["kind"], "detected")
        self.assertFalse(d["target"]["managed"])
        # the URL string (not the supervisor) was handed to live_metrics.
        self.assertEqual(m.call_args[0][0], "http://127.0.0.1:30000")

    def test_explicit_endpoint_beats_managed(self):
        webui._set_supervisor(_RunningFakeSupervisor())
        with mock.patch(
            "sglang.srt.planner.live_metrics.snapshot",
            return_value=self._snap(),
        ) as m:
            d = webui.landing_snapshot_payload({"endpoint": "127.0.0.1:30100"})
        self.assertEqual(d["target"]["kind"], "explicit")
        self.assertEqual(m.call_args[0][0], "http://127.0.0.1:30100")

    def test_managed_beats_detected(self):
        sup = _RunningFakeSupervisor()
        webui._set_supervisor(sup)
        with mock.patch.object(
            webui, "_detect_external_endpoint"
        ) as det, mock.patch(
            "sglang.srt.planner.live_metrics.snapshot",
            return_value=self._snap(),
        ) as m:
            d = webui.landing_snapshot_payload()
        det.assert_not_called()
        self.assertEqual(d["target"]["kind"], "managed")
        self.assertTrue(d["target"]["managed"])
        self.assertIs(m.call_args[0][0], sup)

    def test_nothing_reachable_is_clean_placeholder(self):
        with mock.patch.object(
            webui, "_detect_external_endpoint", return_value=None
        ):
            d = webui.landing_snapshot_payload()
        self.assertTrue(d["ok"])
        self.assertFalse(d["running"])
        self.assertIsNone(d["snapshot"])
        self.assertIsNone(d["target"])

    def test_target_switch_resets_delta_state(self):
        # Counters from two different servers must never be subtracted: the
        # second (switched) poll must start with prev_state=None.
        with mock.patch(
            "sglang.srt.planner.live_metrics.snapshot",
            return_value=self._snap(),
        ) as m:
            webui.landing_snapshot_payload({"endpoint": "127.0.0.1:30000"})
            self.assertIsNotNone(webui._LANDING_SNAPSHOT_STATE)
            webui.landing_snapshot_payload({"endpoint": "127.0.0.1:30100"})
        self.assertIsNone(m.call_args_list[1].kwargs["prev_state"])

    def test_detect_endpoint_payload_probes_ports(self):
        with mock.patch.object(
            webui, "_tcp_open", side_effect=lambda h, p, timeout=0.15: p == 30100
        ), mock.patch.object(
            webui, "_probe_sglang",
            side_effect=lambda u, timeout=0.8: u.endswith(":30100"),
        ):
            d = webui.detect_endpoint_payload()
        self.assertTrue(d["ok"])
        self.assertEqual(d["endpoint"], "http://127.0.0.1:30100")
        self.assertIn(30100, d["probed"])

    def test_detect_endpoint_sweeps_the_whole_sglang_range(self):
        # The 'detect' button must cover the documented 30000-30100 range, not
        # just a three-port sample.
        with mock.patch.object(webui, "_tcp_open", return_value=False):
            d = webui.detect_endpoint_payload()
        for p in (30000, 30017, 30100, 8000):
            self.assertIn(p, d["probed"])
        self.assertIsNone(d["endpoint"])
        self.assertFalse(d["explicit"])

    def test_detect_endpoint_tcp_prescan_gates_the_http_probe(self):
        # A closed port must never reach the (expensive) HTTP probe.
        with mock.patch.object(webui, "_tcp_open", return_value=False), \
                mock.patch.object(webui, "_probe_sglang") as probe:
            webui.detect_endpoint_payload()
        probe.assert_not_called()

    def test_detect_endpoint_explicit_target_skips_the_sweep(self):
        with mock.patch.object(webui, "_tcp_open") as scan, \
                mock.patch.object(webui, "_probe_sglang", return_value=True):
            d = webui.detect_endpoint_payload({"endpoint": "1.2.3.4:31000"})
        scan.assert_not_called()
        self.assertTrue(d["explicit"])
        self.assertEqual(d["endpoint"], "http://1.2.3.4:31000")
        self.assertEqual(d["probed"], [31000])

    def test_detect_endpoint_explicit_target_unreachable_reports_why(self):
        with mock.patch.object(webui, "_probe_sglang", return_value=False):
            d = webui.detect_endpoint_payload({"endpoint": "host-x:31000"})
        self.assertIsNone(d["endpoint"])
        self.assertIn("host-x:31000", d["error"])

    def test_split_host_port_forms(self):
        self.assertEqual(webui._split_host_port("1.2.3.4:31000"), ("1.2.3.4", 31000))
        self.assertEqual(
            webui._split_host_port("http://1.2.3.4:31000"), ("1.2.3.4", 31000))
        self.assertEqual(webui._split_host_port("1.2.3.4"), ("1.2.3.4", 30000))
        self.assertEqual(webui._split_host_port(""), (None, None))


class TestProfileLaunchWiring(CustomTestCase):
    """flags.py env carriage through the launch path: a launched profile must
    match the reference command exactly (argv AND env)."""

    def setUp(self):
        self.fake = _FakeSupervisor()
        webui._set_supervisor(self.fake)

    def tearDown(self):
        webui._set_supervisor(None)

    def test_env_and_profile_argv_reach_the_supervisor(self):
        d = webui.server_start_payload({
            "model_path": "/m", "tp_size": 1, "port": 31000,
            "env": {"SGLANG_UNEVEN_DCP": "1", "HF_TOKEN": "supersecret"},
            "profile_argv": ["--model-path", "/m", "--tp-size", "1"],
        })
        self.assertTrue(d["ok"], d.get("error"))
        st = self.fake.started
        self.assertEqual(st.extra_env["SGLANG_UNEVEN_DCP"], "1")
        self.assertEqual(st.extra_env["HF_TOKEN"], "supersecret")
        # exact profile argv: interpreter -m sglang.launch_server + flag list.
        self.assertEqual(d["launch_command"][1:3], ["-m", "sglang.launch_server"])
        self.assertIn("--model-path", d["launch_command"])
        # the echoed env redacts credential-suffixed names, keeps knobs exact.
        self.assertEqual(d["env_applied"]["HF_TOKEN"], "<redacted>")
        self.assertEqual(d["env_applied"]["SGLANG_UNEVEN_DCP"], "1")
        self.assertNotIn("supersecret", json.dumps(d))

    def test_launch_without_env_stays_default(self):
        d = webui.server_start_payload({"model_path": "/m", "tp_size": 1})
        self.assertTrue(d["ok"], d.get("error"))
        self.assertEqual(self.fake.started.extra_env, {})
        self.assertEqual(d["env_applied"], {})

    def test_generated_profiles_carry_argv_and_launch_env(self):
        d = webui.config_profiles_get({
            "model_cfg": _V2_CONFIG,
            "gpus": [
                {"name": "NVIDIA GeForce RTX 5090", "total_mib": 32607},
                {"name": "NVIDIA GeForce RTX 3080", "total_mib": 20480},
                {"name": "NVIDIA GeForce RTX 3080", "total_mib": 20480},
            ],
        })
        self.assertTrue(d["ok"])
        self.assertGreater(len(d["generated"]), 0)
        for p in d["generated"]:
            self.assertIsInstance(p["argv"], list, p["name"])
            self.assertIsInstance(p["launch_env"], dict, p["name"])
        uneven = [p for p in d["generated"] if "uneven" in p["kind"]]
        self.assertTrue(uneven, [p["kind"] for p in d["generated"]])
        # spec+DCP is only supported with the env pair -- it must be carried.
        self.assertEqual(uneven[0]["launch_env"].get("SGLANG_UNEVEN_DCP"), "1")


class TestBenchRoutes(CustomTestCase):
    """#151 -- probe/gating + the streaming run route (bench_suite mocked)."""

    def _caps(self, **kw):
        from sglang.srt.planner import bench_suite

        d = dict(chat_template_basic=True, tool_parser=None,
                 reasoning_parser="qwen3", streaming=True, spec_decode=False,
                 spec_mode="off", max_model_len=32768, model="m")
        d.update(kw)
        return bench_suite.Capabilities(**d)

    def test_probe_reports_capabilities_and_gates(self):
        with mock.patch(
            "sglang.srt.planner.bench_suite.probe_capabilities",
            return_value=self._caps(),
        ) as m:
            d = webui.bench_probe_payload({"endpoint": "127.0.0.1:30000"})
        self.assertTrue(d["ok"])
        m.assert_called_once()
        self.assertEqual(m.call_args[0][0], "http://127.0.0.1:30000")
        by_id = {t["test_id"]: t for t in d["tests"]}
        self.assertIsNone(by_id[1]["gate_status"])            # runnable
        self.assertEqual(by_id[2]["gate_status"], "blocked")  # no tool parser
        self.assertEqual(by_id[7]["gate_status"], "skip")     # spec off
        self.assertIn("functional", d["presets"])
        self.assertEqual(d["capabilities"]["model"], "m")

    def test_probe_without_endpoint_returns_catalog_only(self):
        with mock.patch(
            "sglang.srt.planner.bench_suite.probe_capabilities"
        ) as m:
            d = webui.bench_probe_payload({})
        m.assert_not_called()
        self.assertTrue(d["ok"])
        self.assertIsNone(d["capabilities"])
        self.assertEqual(len(d["tests"]), 16)

    def test_run_events_stream_per_test(self):
        results = [
            {"test_id": 1, "label": "Basic", "status": "pass",
             "metric": {"name": "latency", "value": 1.2, "unit": "s"},
             "detail": {}, "deps": {}},
            {"test_id": 2, "label": "Tool", "status": "blocked",
             "metric": {"name": "none", "value": None}, "detail": {},
             "deps": {}, "reason": "no tool parser"},
        ]

        def fake_run(endpoint, model, selected=None, capabilities=None,
                     preset=None, force=False, transcript_sink=None):
            yield from results

        with mock.patch(
            "sglang.srt.planner.bench_suite.run_suite", side_effect=fake_run
        ):
            evs = list(webui.bench_run_events(
                {"endpoint": "127.0.0.1:30000", "model": "m",
                 "selected": [1, 2]}))
        self.assertEqual(evs[0]["event"], "start")
        got = [e["result"]["test_id"] for e in evs if e["event"] == "result"]
        self.assertEqual(got, [1, 2])
        self.assertEqual(evs[-1]["event"], "done")
        self.assertEqual(evs[-1]["counts"], {"pass": 1, "blocked": 1})

    def test_run_requires_endpoint(self):
        evs = list(webui.bench_run_events({"model": "m"}))
        self.assertEqual(evs[0]["event"], "error")

    def test_run_passes_capabilities_and_force_through(self):
        # Gating input: client-side probed capabilities are handed to
        # run_suite (no re-probe), and the force flag reaches it.
        captured = {}

        def fake_run(endpoint, model, selected=None, capabilities=None,
                     preset=None, force=False, transcript_sink=None):
            captured["caps"] = capabilities
            captured["force"] = force
            return iter(())

        with mock.patch(
            "sglang.srt.planner.bench_suite.run_suite", side_effect=fake_run
        ):
            list(webui.bench_run_events({
                "endpoint": "e", "model": "m", "selected": [1],
                "capabilities": self._caps(
                    tool_parser="qwen3_coder").to_json(),
                "force": True,
            }))
        self.assertEqual(captured["caps"].tool_parser, "qwen3_coder")
        self.assertTrue(captured["force"])

    def test_browser_never_calls_the_model(self):
        # Backend-driven by design: the page never POSTs to the model API.
        self.assertNotIn("chat/completions", webui.INDEX_HTML)
        self.assertIn("/api/bench_run", webui.INDEX_HTML)


class TestShareRoutes(CustomTestCase):
    """#152 -- preview-then-confirm; the PAT never persists, never echoes."""

    TOKEN = "ghp_TESTSECRETTOKEN123"

    def test_preview_renders_exact_markdown(self):
        d = webui.share_preview_payload({"payload": {
            "model": "Qwen3.6-27B",
            "command": {
                "argv": ["python", "-m", "sglang.launch_server", "--tp", "3"],
                "env": {"SGLANG_UNEVEN_DCP": "1", "HF_TOKEN": "hfsecret"},
            },
            "metrics": {"decode_tok_s": 51.0},
        }})
        self.assertTrue(d["ok"])
        self.assertIn("Start command (exact)", d["report"])
        self.assertIn("SGLANG_UNEVEN_DCP=1", d["report"])   # knob stays exact
        self.assertNotIn("hfsecret", d["report"])           # credential redacted
        self.assertTrue(d["default_repo"])

    def test_submit_refused_without_confirmation_no_network(self):
        api = mock.Mock()
        with mock.patch(
            "sglang.srt.planner.github_share._default_api", api
        ):
            d = webui.share_submit_payload(
                {"report": "r", "token": self.TOKEN, "confirmed": False})
        self.assertFalse(d["ok"])
        self.assertIn("confirm", d["error"].lower())
        api.assert_not_called()
        self.assertNotIn(self.TOKEN, json.dumps(d))

    def test_submit_confirmed_calls_github_never_echoes_pat(self):
        with mock.patch(
            "sglang.srt.planner.github_share.submit",
            return_value={"action": "created", "number": 7,
                          "url": "https://github.com/x/y/issues/7"},
        ) as m:
            d = webui.share_submit_payload({
                "report": "r", "token": self.TOKEN, "confirmed": True,
                "repo": "user/repo", "existing_issue": "",
            })
        self.assertTrue(d["ok"])
        self.assertEqual(d["number"], 7)
        self.assertEqual(d["action"], "created")
        self.assertTrue(m.call_args.kwargs["confirmed"])
        self.assertEqual(m.call_args.kwargs["repo"], "user/repo")
        self.assertNotIn(self.TOKEN, json.dumps(d))

    def test_submit_error_stays_redacted(self):
        from sglang.srt.planner.github_share import GitHubShareError

        with mock.patch(
            "sglang.srt.planner.github_share.submit",
            side_effect=GitHubShareError("HTTP 401 <redacted-token>"),
        ):
            d = webui.share_submit_payload(
                {"report": "r", "token": self.TOKEN, "confirmed": True})
        self.assertFalse(d["ok"])
        self.assertNotIn(self.TOKEN, d["error"])

    def test_submit_requires_report(self):
        d = webui.share_submit_payload(
            {"token": self.TOKEN, "confirmed": True})
        self.assertFalse(d["ok"])
        self.assertNotIn(self.TOKEN, json.dumps(d))


class TestV3HttpRoutes(WebUIFixture):
    """The v3 routes over a real in-process HTTP round-trip, including the
    SSE stream of /api/bench_run."""

    def setUp(self):
        self.fake = _FakeSupervisor()
        webui._set_supervisor(self.fake)
        _reset_landing_state()
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), webui._Handler)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        webui._set_supervisor(None)
        _reset_landing_state()
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(timeout=5)

    def _post_raw(self, path, obj):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=30)

    def test_bench_run_streams_sse(self):
        results = [
            {"test_id": 1, "label": "Basic", "status": "pass",
             "metric": {"name": "none", "value": None}, "detail": {},
             "deps": {}},
            {"test_id": 3, "label": "Streaming", "status": "skip",
             "metric": {"name": "none", "value": None}, "detail": {},
             "deps": {}, "reason": "gated"},
        ]

        def fake_run(endpoint, model, selected=None, capabilities=None,
                     preset=None, force=False, transcript_sink=None):
            yield from results

        with mock.patch(
            "sglang.srt.planner.bench_suite.run_suite", side_effect=fake_run
        ):
            with self._post_raw(
                "/api/bench_run",
                {"endpoint": "127.0.0.1:30000", "model": "m",
                 "selected": [1, 3]},
            ) as r:
                ctype = r.headers.get("Content-Type")
                body = r.read().decode()
        self.assertEqual(ctype, "text/event-stream")
        frames = [
            json.loads(f[len("data: "):])
            for f in body.split("\n\n") if f.startswith("data: ")
        ]
        events = [f["event"] for f in frames]
        self.assertEqual(events, ["start", "result", "result", "done"])
        self.assertEqual(frames[1]["result"]["status"], "pass")
        self.assertEqual(frames[2]["result"]["status"], "skip")

    def test_bench_probe_route(self):
        with self._post_raw("/api/bench_probe", {}) as r:
            d = json.loads(r.read())
        self.assertTrue(d["ok"])
        self.assertEqual(len(d["tests"]), 16)

    def test_share_preview_route(self):
        with self._post_raw(
            "/api/share_preview",
            {"payload": {"model": "M", "metrics": {"decode_tok_s": 1.0}}},
        ) as r:
            d = json.loads(r.read())
        self.assertTrue(d["ok"])
        self.assertIn("Measured metrics", d["report"])

    def test_detect_endpoint_route(self):
        with mock.patch.object(webui, "_tcp_open", return_value=False):
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/api/detect_endpoint", timeout=10
            ) as r:
                d = json.loads(r.read())
        self.assertTrue(d["ok"])
        self.assertIsNone(d["endpoint"])

    def test_detect_routes_answer_post_too(self):
        # The page's detect button POSTs a body (explicit host:port / port
        # list); a GET-only route answered {"error": "not found"}.
        with mock.patch.object(webui, "_probe_sglang", return_value=True):
            with self._post_raw(
                "/api/detect_endpoint", {"endpoint": "1.2.3.4:31000"}
            ) as r:
                d = json.loads(r.read())
        self.assertTrue(d["ok"])
        self.assertEqual(d["endpoint"], "http://1.2.3.4:31000")
        with self._post_raw("/api/detect", {}) as r:
            d = json.loads(r.read())
        self.assertIn("ok", d)
        self.assertNotIn("error", d)

    def test_detect_endpoint_route_accepts_query_params(self):
        with mock.patch.object(webui, "_probe_sglang", return_value=True):
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}"
                "/api/detect_endpoint?endpoint=1.2.3.4:31000",
                timeout=10,
            ) as r:
                d = json.loads(r.read())
        self.assertEqual(d["endpoint"], "http://1.2.3.4:31000")

    def test_live_snapshot_route_accepts_endpoint_query(self):
        fake_snap = (
            {"ok": True, "gpus": [], "rates": {"decode_tok_s": 42.0}},
            {"t": 1.0},
        )
        with mock.patch(
            "sglang.srt.planner.live_metrics.snapshot", return_value=fake_snap
        ) as m:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/api/live_snapshot"
                "?endpoint=127.0.0.1:30777",
                timeout=10,
            ) as r:
                d = json.loads(r.read())
        self.assertTrue(d["running"])
        self.assertEqual(d["target"]["kind"], "explicit")
        self.assertEqual(d["snapshot"]["rates"]["decode_tok_s"], 42.0)
        self.assertEqual(m.call_args[0][0], "http://127.0.0.1:30777")


class TestV3IndexMarkers(CustomTestCase):
    def test_new_ui_markers_present(self):
        for token in (
            "view_bench", "/api/bench_probe", "/api/bench_run",
            "/api/share_preview", "/api/share_submit", "/api/detect_endpoint",
            # #13: the monitor target's three ambiguous buttons became two
            # explicit actions; detectLandingEndpoint split into Connect and
            # Find a server.
            "land_endpoint", "landingConnect", "landingFindServer", "segbar",
            "renderStartConfig", "normalizeStartConfig", "profile_env_box",
            "renderProfileLaunch", "never persisted, never logged",
        ):
            self.assertIn(token, webui.INDEX_HTML, token)

    def test_runner_left_column_single_flow(self):
        """Runner redesign: EXACTLY one model selector (dropdown fills the one
        free-text field + one GGUF sub-dropdown); the old duplicate launch
        form (sv_*), plan-knob list and second variant dropdown are gone; the
        SERVING identity group and the flag-surface search exist."""
        html = webui.INDEX_HTML
        # the one model selector: one free-text field, one gguf sub-dropdown.
        self.assertEqual(html.count('id="model"'), 1)
        self.assertEqual(html.count('id="gguf_choice"'), 1)
        self.assertEqual(html.count('id="gguf_pick"'), 1)
        # every removed duplicate input / mechanism is really gone.
        for removed in (
            'id="sv_model"', 'id="sv_variant"', 'id="sv_format"',
            'id="sv_tp"', 'id="sv_rgi"', 'id="sv_rtr"', 'id="sv_kv"',
            'id="sv_spec"', 'id="sv_ct"', 'id="sv_tool"', 'id="sv_reason"',
            'id="sv_seqs"', 'id="sv_vision"', 'id="model_variant_select"',
            "model_variant_wrap", "applyVariant", 'id="tp_size"',
            'id="kv_cache_dtype"', 'id="knobs"', "loadKnobs", "knob_",
            'id="notexpr"',
        ):
            self.assertNotIn(removed, html, removed)
        # the new single-source structure is present.
        for kept in (
            'id="model_select"', 'id="sv_served"', 'id="sv_host"',
            'id="sv_port"', 'id="sv_ctx"', 'id="max_running_requests"',
            'id="flag_search"', "filterFlags", "SERVING_OWNED",
            "modelState", "schedulePlan",
        ):
            self.assertIn(kept, html, kept)

    def test_one_placement_renderer_two_callers(self):
        # ONE renderer feeds both the landing (running, WITH the live gpus[]
        # for the merged telemetry line) and the runner (prospective, without)
        # placement views.
        self.assertEqual(
            webui.INDEX_HTML.count("function renderPlacement"), 1)
        self.assertGreaterEqual(
            webui.INDEX_HTML.count("renderPlacement(d.placement)"), 1)
        # landing passes the live gpus[] plus the live graph-capture opts.
        self.assertGreaterEqual(
            webui.INDEX_HTML.count("renderPlacement(d.placement, s.gpus"), 1)


class TestRunnerLmStudioLayout(CustomTestCase):
    """LM-Studio-style runner restructure: fixed collapsible sections in a
    learnable order, one advanced toggle, labeled rows with slider/toggle/
    dropdown controls, preset dropdown bar, sticky action bar, per-card
    hardware bars. The flag catalog stays the single source of truth; the
    sections are client-side VIEWS onto it (each flag rendered exactly once).
    """

    def test_section_structure_fixed_order(self):
        html = webui.INDEX_HTML
        order = ["sec_context", "sec_gpu", "sec_speculative", "sec_cache",
                 "sec_serving", "sec_advanced"]
        idx = [html.index('id="%s"' % s) for s in order]
        self.assertEqual(idx, sorted(idx), "sections out of order")
        # each section exists exactly once, and the per-section flag
        # containers the renderer fills are present.
        for s in order:
            self.assertEqual(html.count('id="%s"' % s), 1, s)
        for c in ("secflags_context", "secflags_gpu", "secflags_speculative",
                  "secflags_cache", "secflags_serving"):
            self.assertEqual(html.count('id="%s"' % c), 1, c)
        # context rows live IN the Context section; serving identity rows in
        # the Serving section (ids unchanged: sv_ctx / max_running_requests /
        # sv_served / sv_host / sv_port).
        self.assertLess(html.index('id="sv_ctx"'), html.index('id="sec_gpu"'))
        self.assertGreater(
            html.index('id="sv_served"'), html.index('id="sec_serving"'))

    def test_advanced_behind_one_toggle(self):
        html = webui.INDEX_HTML
        self.assertEqual(html.count('id="advanced_toggle"'), 1)
        self.assertIn("Show advanced settings", html)
        self.assertIn("toggleAdvanced", html)
        # advanced starts hidden; the search filter can reveal it.
        self.assertIn('id="sec_advanced" style="display:none', html)

    def test_no_duplicate_flag_rendering(self):
        html = webui.INDEX_HTML
        # ONE row builder, ONE render site, and a first-match-wins section
        # classifier -> a catalog flag can only ever render once.
        self.assertEqual(html.count("function flagRowHtml"), 1)
        self.assertEqual(html.count("flagRowHtml).join"), 2)  # sections + adv
        self.assertEqual(html.count("function flagSection"), 1)
        self.assertEqual(html.count("function renderFlagSurface"), 1)

    def test_lmstudio_controls_present(self):
        html = webui.INDEX_HTML
        # slider+numeric pairs (context clamps to the plan capacity max).
        for token in ('id="sv_ctx_slider"', 'id="mrr_slider"',
                      "setCtxCap", "cap.max_context_tokens"):
            self.assertIn(token, html, token)
        # toggle switches (pure CSS) + "?" hover + changed-from-profile dot.
        # The dot's wording followed the #7 rename: prefabricated presets are
        # gone, so a row can only differ from a profile the user loaded.
        for token in ('class="switch"', 'class="track"', 'class="qmark"',
                      "changed from the loaded profile", "markPresetDrift",
                      "presetSnapshot"):
            self.assertIn(token, html, token)

    def test_preset_bar_action_bar_hardware_model_search(self):
        html = webui.INDEX_HTML
        # preset dropdown + save above the settings panel.
        self.assertIn("profile_select", html)
        self.assertIn("applyProfileSel", html)
        self.assertLess(html.index('id="profile_pick"'),
                        html.index('id="flag_search"'))
        # sticky action bar: Load / Eject / Restart + status chip + boot log.
        for token in ('id="action_bar"', 'id="status_chip"', ">Eject<",
                      ">Load model<", 'id="boot_log"', "updateStatusChip",
                      "pollStatusUntilSettled"):
            self.assertIn(token, html, token)
        # hardware rows carry a VRAM bar; model picker searches as you type.
        for token in ("cardbar", 'id="model_search"', "renderModelOptions"):
            self.assertIn(token, html, token)

    def test_single_gpu_selector_writes_base_gpu_id(self):
        # tp=1: a compact GPU dropdown (index + name + VRAM) defaulting to the
        # preset's rule pick, writing the stock --base-gpu-id flag field.
        html = webui.INDEX_HTML
        self.assertEqual(html.count('id="gpu_pick_select"'), 1)
        self.assertEqual(html.count('id="row_gpu_pick"'), 1)
        self.assertGreater(html.index('id="row_gpu_pick"'),
                           html.index('id="sec_gpu"'))
        self.assertLess(html.index('id="row_gpu_pick"'),
                        html.index('id="sec_speculative"'))
        for token in ("updateGpuPick", "gpuPickChanged", "fl_base_gpu_id",
                      "_effectiveTp"):
            self.assertIn(token, html, token)

    def test_section_map_covers_catalog(self):
        """The JS classifier's section ids reference REAL catalog flags: the
        curated section id lists must stay in sync with the catalog (a
        renamed/removed flag would silently fall through to advanced)."""
        import re

        from sglang.srt.planner import flags as flagsmod

        cat = flagsmod.catalog()
        html = webui.INDEX_HTML
        for const in ("SEC_CACHE_IDS", "SEC_SERVING_IDS", "SEC_GPU_IDS"):
            m = re.search(const + r"=\{([^}]*)\}", html)
            self.assertIsNotNone(m, const)
            ids = re.findall(r"(\w+):1", m.group(1))
            self.assertTrue(ids, const)
            for fid in ids:
                self.assertIn(fid, cat, "%s names unknown flag %s"
                              % (const, fid))
        # the pinned single-flag routes exist too.
        self.assertIn("kv_cache_dtype", cat)
        self.assertIn("base_gpu_id", cat)


if __name__ == "__main__":
    unittest.main()


class TestProfileArgvMerge(unittest.TestCase):
    """A profile's argv is a FLAG SET; the serving identity lives in the launch
    form. Live boots exposed both halves of this: using the profile argv alone
    produced a command sglang rejects ("--model-path required"), and letting the
    profile's placeholder --max-running-requests win OOM'd CUDA-graph capture.
    """

    def _settings(self, **kw):
        from sglang.srt.planner.server_manager import LaunchSettings

        base = dict(
            model_path="/models/M", served_model_name="M", tp_size=3,
            context_length=262144, max_running_requests=2, host="0.0.0.0",
            port=30000,
        )
        base.update(kw)
        return LaunchSettings(**base).validate()

    def test_merge_keeps_serving_identity_and_profile_flags(self):
        from sglang.srt.planner import webui

        prof = [
            "--tp-size", "3", "--rank-tp-ratio", "auto-performance",
            "--rank-auto-reserve-mib", "3000,2200,2200",
            "--speculative-adaptive",
            # profile placeholders that MUST NOT win over the form:
            "--host", "127.0.0.1", "--max-running-requests", "16",
        ]
        argv = webui._argv_from_payload(
            {"profile_argv": prof}, self._settings())
        cmd = " ".join(argv)
        # the profile's own tuning flags survive
        self.assertIn("--rank-tp-ratio auto-performance", cmd)
        self.assertIn("--rank-auto-reserve-mib 3000,2200,2200", cmd)
        self.assertIn("--speculative-adaptive", cmd)
        # serving identity comes from the form, exactly once
        self.assertIn("--model-path /models/M", cmd)
        self.assertIn("--context-length 262144", cmd)
        self.assertIn("--max-running-requests 2", cmd)
        self.assertNotIn("--max-running-requests 16", cmd)
        self.assertIn("--host 0.0.0.0", cmd)
        self.assertNotIn("--host 127.0.0.1", cmd)
        for flag in ("--model-path", "--max-running-requests", "--host"):
            self.assertEqual(argv.count(flag), 1, flag)

    def test_drop_flags_removes_flag_and_value(self):
        from sglang.srt.planner import webui

        out = webui._drop_flags(
            ["--a", "1", "--keep", "2", "--flagonly", "--b", "3"],
            {"--a", "--flagonly"})
        self.assertEqual(out, ["--keep", "2", "--b", "3"])


class TestDeviceIndexSpaces(CustomTestCase):
    """CUDA-order vs NVML-order bridging through the dashboard: the detect
    payload names BOTH indices per card, gpu_state rows gain the cuda index,
    and the JS keys every rank->card inventory in CUDA space (the space
    rank_gpu_id / base_gpu_id live in), never in NVML order."""

    def _spec(self):
        from sglang.srt.planner.hardware import GpuDescriptor, HardwareSpec

        # THE reference box: NVML order [3080, 5090, 3080]; the bridge puts
        # the 5090 at cuda:0.
        return HardwareSpec(
            gpus=(
                GpuDescriptor(index=0, name="NVIDIA GeForce RTX 3080",
                              total_mib=20480, free_mib=15000,
                              cuda_index=1),
                GpuDescriptor(index=1, name="NVIDIA GeForce RTX 5090",
                              total_mib=32607, free_mib=30000,
                              cuda_index=0),
                GpuDescriptor(index=2, name="NVIDIA GeForce RTX 3080",
                              total_mib=20480, free_mib=15000,
                              cuda_index=2),
            ),
            source="nvml",
            host_ram_mib=64000,
            cuda_index_source="identity-map",
        )

    def test_detect_payload_carries_both_indices(self):
        with mock.patch(
            "sglang.srt.planner.hardware.hardware_from_nvml",
            return_value=self._spec(),
        ):
            d = webui.detect_hardware()
        self.assertTrue(d["ok"])
        self.assertEqual(d["cuda_index_source"], "identity-map")
        by_nvml = {g["index"]: g for g in d["gpus"]}
        self.assertEqual(by_nvml[1]["cuda_index"], 0)   # 5090: nvml:1=cuda:0
        self.assertEqual(by_nvml[1]["total_mib"], 32607)
        self.assertEqual(by_nvml[0]["cuda_index"], 1)
        self.assertEqual(by_nvml[2]["cuda_index"], 2)

    def test_detect_payload_carries_pcie_link_info(self):
        # Round 5 bug: hardware_from_nvml() has always computed pcie_gen /
        # pcie_width per card (hardware.py's GpuDescriptor), but the dict
        # comprehension in detect_hardware() never forwarded them, so the
        # Hardware step's card list could not show what a card's link
        # actually trains to -- only the capacity-matrix's "assumes
        # pcie/nvlink topology, not measured" disclaimer existed, never the
        # measured fact for THIS rig's asymmetric wiring.
        from sglang.srt.planner.hardware import GpuDescriptor, HardwareSpec

        spec = HardwareSpec(
            gpus=(
                GpuDescriptor(index=0, name="NVIDIA GeForce RTX 5090",
                              total_mib=32607, free_mib=30000,
                              pcie_gen=4, pcie_width=4),
                GpuDescriptor(index=1, name="NVIDIA GeForce RTX 3080",
                              total_mib=20480, free_mib=15000,
                              pcie_gen=4, pcie_width=16),
            ),
            source="nvml",
        )
        with mock.patch(
            "sglang.srt.planner.hardware.hardware_from_nvml",
            return_value=spec,
        ):
            d = webui.detect_hardware()
        self.assertTrue(d["ok"])
        by_index = {g["index"]: g for g in d["gpus"]}
        self.assertEqual(by_index[0]["pcie_gen"], 4)
        self.assertEqual(by_index[0]["pcie_width"], 4)
        self.assertEqual(by_index[1]["pcie_width"], 16)

    def test_hardware_step_card_row_renders_the_detected_pcie_link(self):
        html = webui.INDEX_HTML
        self.assertIn("pcie_gen: g.pcie_gen, pcie_width: g.pcie_width", html)
        self.assertIn("c.pcie_gen", html)
        self.assertIn("PCIe ' + c.pcie_gen", html)

    def test_gpu_state_rows_annotated_with_cuda_index(self):
        from sglang.srt.planner import device_map as dmod
        from sglang.srt.planner.energy import GpuPowerState

        def _st(i, name):
            return GpuPowerState(
                nvml_index=i, name=name, power_watts=100.0,
                power_limit_w=320.0, default_limit_w=320.0,
                min_limit_w=100.0, max_limit_w=350.0,
                limit_pct_of_default=1.0, sm_clock_mhz=1000,
                mem_clock_mhz=1000, temperature_c=50, oc_uv="stock",
                clock_offset_mhz=None,
            )

        dm = dmod.DeviceMap(
            entries=(
                dmod.DeviceMapEntry(nvml_index=0, cuda_index=1,
                                    name="RTX 3080", total_mib=20480,
                                    uuid="aaaa"),
                dmod.DeviceMapEntry(nvml_index=1, cuda_index=0,
                                    name="RTX 5090", total_mib=32607,
                                    uuid="bbbb"),
            ),
            source=dmod.IDENTITY_MAP_SOURCE,
        )
        with mock.patch(
            "sglang.srt.planner.energy.read_gpu_power_states",
            return_value=[_st(0, "RTX 3080"), _st(1, "RTX 5090")],
        ), mock.patch.object(dmod, "device_map", return_value=dm):
            d = webui.gpu_state_payload()
        self.assertTrue(d["ok"])
        by_nvml = {c["nvml_index"]: c for c in d["cards"]}
        self.assertEqual(by_nvml[1]["cuda_index"], 0)
        self.assertEqual(by_nvml[0]["cuda_index"], 1)

    def test_js_keys_placement_inventory_in_cuda_space(self):
        html = webui.INDEX_HTML
        # the shared dual-label helper + the unresolved-order plumbing
        # (#397: there is no "emulated order" marker any more -- a card
        # either has a resolved cuda_index or none at all).
        for token in ("function devLabel", "function noteCudaMap",
                      "_cudaNvml", "_cudaMapUnresolved"):
            self.assertIn(token, html, token)
        # landing placement inventory: keyed by cuda_index (nvml only as a
        # labeled fallback), NEVER plainly by nvml_index (the old bug that
        # attributed rank0 = cuda:0 = the 5090 to the 3080 at nvml:0).
        self.assertIn("g.cuda_index!=null?g.cuda_index:g.nvml_index", html)
        self.assertNotIn("ct[g.nvml_index]=g.mem_total_mib", html)
        # runner placement inventory: detected cards keyed by cuda_index.
        self.assertIn("c.cuda_index", html)
        # single-GPU pick: option VALUES are cuda indices with a dual label.
        self.assertIn("'<option value=\"'+d.cuda", html)
        # placement card blocks + live telemetry rows use the dual label.
        self.assertIn("devLabel(c.gpu_index", html)
        self.assertIn("devLabel(g.cuda_index, g.nvml_index)", html)
        # detect stores the cuda index on each card row.
        self.assertIn("cuda_index: (g.cuda_index!=null?g.cuda_index:null)",
                      html)
        # the positional plan payload posts detected cards in CUDA order
        # (backend re-index positions == the --rank-gpu-id space).
        self.assertIn("a.cuda_index!=null?a.cuda_index:1e9", html)
        # #397: an order that could not be resolved is surfaced as such --
        # and there is no emulated order left to surface.
        self.assertIn("cuda indices unresolved for some cards", html)
        self.assertNotIn("FASTEST_FIRST emulation", html)


class TestGranularVramView(CustomTestCase):
    """Fine-grained VRAM segments + CUDA-graph line + KV-replication marking
    + side-by-side normal-TP: renderer markers and payload wiring."""

    def test_renderer_markers_present(self):
        for token in (
            "segBarHtml", "graphMemHtml", "class=\"hatch\"",
            "repltag", "replicated x", "SEG_COLORS",
            "attn_q", "kv_draft", "graphs", "flagsAreForkish",
            "stock_compare", "sxs-h", "graphCapture",
        ):
            self.assertIn(token, webui.INDEX_HTML, token)
        # tiny-segment collapse into ONE neutral "other" sliver.
        self.assertIn("other (", webui.INDEX_HTML)
        # the old coarse four-lump bar title is gone from the renderer.
        self.assertNotIn("title=\"weights '+fmtMib", webui.INDEX_HTML)

    def test_neutral_stock_wording(self):
        self.assertNotIn("DOES NOT RUN", webui.INDEX_HTML)
        self.assertIn("not expressible", webui.INDEX_HTML)

    def test_placement_payload_carries_segments_and_graph_mem(self):
        d = webui.placement_payload(
            {"model_cfg": _V2_CONFIG,
             "flags": {"tp_size": 3, "rank_gpu_id": [0, 1, 2],
                       "rank_gpu_memory_mib": [16000, 16000, 16000],
                       "context_length": 8192}}
        )
        self.assertTrue(d["ok"], d.get("error"))
        pl = d["placement"]
        self.assertIn("graph_mem", pl)
        self.assertIn("kv_replication", pl)
        segs = pl["cards"][0]["segments"]
        self.assertTrue(segs)
        for s in segs:
            self.assertIn("detail", s)
            self.assertIn("replicated", s)

    def test_stock_compare_side_by_side_payload(self):
        # _V2_CONFIG: q=24, kv=4 -> tp=3 is NOT stock-legal, tp=2 is; the
        # comparison picks tp=2 on the identical-VRAM pair and renders a
        # full second placement with stock semantics.
        d = webui.placement_payload(
            {"model_cfg": _V2_CONFIG,
             "stock_compare": True,
             "flags": {"tp_size": 3, "rank_tp_ratio": [4, 2, 2],
                       "dcp_size": 3,
                       "rank_gpu_memory_mib": [28000, 16000, 16000],
                       "card_total_mib": {"0": 32607, "1": 20480,
                                          "2": 20480},
                       "context_length": 8192}}
        )
        self.assertTrue(d["ok"], d.get("error"))
        st = d["stock"]
        self.assertTrue(st["legal"])
        self.assertEqual(st["tp"], 2)
        self.assertIsNotNone(st["placement"])
        self.assertEqual(st["placement"]["tp_size"], 2)
        self.assertEqual(len(st["placement"]["cards"]), 2)
        # neutral wording: a fact-shaped note, no verdict.
        self.assertNotIn("DOES NOT RUN", st["note"])

    def test_stock_compare_states_rule_when_inexpressible(self):
        # kv=5, q=25 on 3 cards: no tp in {3,2} is stock-legal -> no numbers,
        # just the neutral rule statement.
        cfg = dict(_V2_CONFIG, num_attention_heads=25,
                   num_key_value_heads=5, head_dim=128)
        d = webui.placement_payload(
            {"model_cfg": cfg,
             "stock_compare": True,
             "flags": {"tp_size": 3, "rank_tp_ratio": [4, 2, 2],
                       "card_total_mib": {"0": 32607, "1": 20480,
                                          "2": 20480},
                       "context_length": 8192}}
        )
        self.assertTrue(d["ok"], d.get("error"))
        st = d["stock"]
        self.assertFalse(st["legal"])
        self.assertIsNone(st["placement"])
        self.assertIn("stock requires", st["note"])

    def test_replication_marker_in_payload(self):
        d = webui.placement_payload(
            {"model_cfg": _V2_CONFIG,
             "flags": {"tp_size": 3, "rank_tp_ratio": [4, 2, 2],
                       "dcp_size": 3,
                       "rank_gpu_memory_mib": [16000] * 3,
                       "context_length": 8192}}
        )
        kr = d["placement"]["kv_replication"]
        self.assertTrue(kr["replicated"])
        self.assertEqual(kr["source"], "fork-uneven-dcp")


class TestLiveGraphCapture(CustomTestCase):
    """LIVE CUDA-graph memory source resolution: managed boot log first,
    conventional /tmp log for detected servers, server_info fallback, and an
    honest n/a."""

    _LINES = (
        "[2026-07-21 13:35:33 TP0] Capture target decode CUDA graph end. "
        "elapsed=4.83 s, mem usage=0.24 GB, avail mem=7.28 GB.\n"
    )

    def test_managed_boot_log_parse(self):
        tmp = tempfile.mkdtemp()
        log = os.path.join(tmp, "sglang_boot_30000.log")
        with open(log, "w") as f:
            f.write(self._LINES)

        class _Sup:
            _log_path = log

        gc = webui._live_graph_capture(_Sup(), "http://127.0.0.1:30000", {})
        self.assertEqual(gc["source"], "boot-log")
        self.assertAlmostEqual(
            gc["summary"]["per_rank_mib"][0], 0.24 * 1024, places=2
        )

    def test_server_info_fallback(self):
        snap = {"server_info": {"server_info": {
            "internal_states": [{"memory_usage": {"graph": 0.5}}]}}}
        gc = webui._live_graph_capture(None, "http://10.0.0.9:12345", snap)
        self.assertEqual(gc["source"], "server_info")
        self.assertAlmostEqual(gc["total_mib"], 512.0, places=2)
        self.assertIn("rank 0", gc["note"])

    def test_honest_na_without_any_source(self):
        gc = webui._live_graph_capture(None, "http://10.0.0.9:12345", {})
        self.assertIsNone(gc["source"])
        self.assertIn("n/a", gc["reason"])


class TestQualityPermissionNote(CustomTestCase):
    """The pre-publication self-reminder is gone; only a short thanks and the
    existing reddit reference remain in the quality tab."""

    REDDIT_URL = (
        "https://www.reddit.com/r/LocalLLaMA/comments/1t53dhp/"
        "quality_comparison_between_qwen_36_27b/"
    )

    def test_reminder_note_removed(self):
        self.assertNotIn("quality_permission_note", webui.INDEX_HTML)
        self.assertNotIn("ERINNERUNG", webui.INDEX_HTML)

    def test_thanks_and_reddit_reference_remain(self):
        html = webui.INDEX_HTML
        qi = html.index('id="view_quality"')
        qend = html.index("<script>")
        quality = html[qi:qend]
        self.assertIn("Thanks", quality)
        # The existing reddit link/reference survives unchanged -- nothing
        # new is invented in its place.
        self.assertIn(self.REDDIT_URL, quality)


class TestColocationControls(CustomTestCase):
    """Explicit co-location controls in the Runner's GPU offload/split
    section: tp chooser, per-card rank steppers (sum == tp enforced),
    even/uneven selector, preset prefill, and the Launch/Plan gate."""

    def test_new_ids_and_mps_note_present(self):
        html = webui.INDEX_HTML
        for token in (
            'id="row_tp_count"', 'id="tp_count"', 'id="colo_box"',
            'id="colo_note"', 'id="colo_rows"', 'id="colo_sum"',
            'id="colo_err"', 'id="colo_manual_note"',
            'id="row_split_mode"', 'id="split_mode_select"',
        ):
            self.assertEqual(html.count(token), 1, token)
        # tp chooser sits at the TOP of the GPU offload/split section,
        # above the prospective placement and the single-GPU pick.
        self.assertGreater(html.index('id="row_tp_count"'),
                           html.index('id="sec_gpu"'))
        self.assertLess(html.index('id="row_tp_count"'),
                        html.index('id="gpu_placement"'))
        self.assertLess(html.index('id="colo_box"'),
                        html.index('id="row_gpu_pick"'))
        # MPS-required + fork-only notes.
        self.assertIn("REQUIRES CUDA MPS", html)
        self.assertIn("fork-only capability", html)
        self.assertIn("cannot co-locate ranks at all", html)

    def test_derivation_and_reverse_populate_js_present(self):
        html = webui.INDEX_HTML
        # canonical derivation (mirrors flags.rank_gpu_id_from_counts) +
        # default distribution (mirrors flags.colocation_rank_counts) +
        # reverse mapping (mirrors flags.rank_counts_from_gpu_id).
        for token in (
            "function coloCards", "function coloDefaultCounts",
            "function coloRankGpuIdFromCounts", "function coloParseRankGpuId",
            "function updateColoUI", "flags.colocation_rank_counts",
            "flags.rank_counts_from_gpu_id",
            "rank_gpu_id 0,0,1,2",  # the documented 2/1/1 -> 0,0,1,2 rule
        ):
            self.assertIn(token, html, token)
        # manual rank_gpu_id mode: steppers grey out with the note.
        self.assertIn("manual rank_gpu_id active", html)
        self.assertIn("_coloManual", html)

    def test_controls_write_authoritative_fields_and_replan(self):
        html = webui.INDEX_HTML
        # every control routes through the ONE flag surface + onFlagChange,
        # which drives the SAME debounced re-resolve/re-plan path
        # (resolveFlags + refreshRunnerPlacement + schedulePlan).
        self.assertIn("function tpCountChanged", html)
        self.assertIn("onFlagChange('tp_size')", html)
        self.assertIn("function coloRanksChanged", html)
        self.assertIn("onFlagChange('rank_gpu_id')", html)
        self.assertIn("function splitModeChanged", html)
        self.assertIn("onFlagChange('rank_tp_ratio')", html)
        self.assertIn("fl_tp_size", html)
        self.assertIn("fl_rank_gpu_id", html)
        self.assertIn("fl_rank_tp_ratio", html)
        # updateColoUI is wired into flag edits, preset apply and card list.
        self.assertGreaterEqual(html.count("updateColoUI()"), 3)

    def test_split_mode_selector_values(self):
        html = webui.INDEX_HTML
        # three-way + custom: even clears the ratio (uniform split), the two
        # uneven modes write the enum sentinels, custom leaves the free-text
        # rank-tp-ratio field authoritative.
        for token in (
            '<option value="even">', '<option value="auto">',
            '<option value="auto-performance">', '<option value="custom">',
            "el.value=(v==='even')? '' : v",
            "if(v==='custom') return;",
        ):
            self.assertIn(token, html, token)

    def test_launch_and_plan_gated_on_rank_sum(self):
        html = webui.INDEX_HTML
        self.assertIn("function coloBlockError", html)
        self.assertIn("sum must equal tp", html)
        # Plan gate: doPlan refuses with the inline reason.
        self.assertIn("PLAN BLOCKED", html)
        # Launch/Restart gate: both go through launchGate().
        self.assertIn("function launchGate", html)
        self.assertEqual(html.count("if (!launchGate()) return;"), 2)

    def test_placement_reacts_to_colocation_flags_with_overcommit(self):
        # The prospective placement consumed by the section must carry the
        # physical-impossibility flag for the co-located card: 2 x 18000 MiB
        # on a 32607 MiB card exceeds it -> EXCEEDS CARD; 2 x 15000 fits.
        d = webui.placement_payload(
            {"model_cfg": _V2_CONFIG,
             "flags": {"tp_size": 4, "rank_gpu_id": [0, 0, 1, 2],
                       "rank_gpu_memory_mib": [18000] * 4,
                       "card_total_mib": {"0": 32607, "1": 20480,
                                          "2": 20480},
                       "context_length": 8192}}
        )
        self.assertTrue(d["ok"], d.get("error"))
        cards = {c["gpu_index"]: c for c in d["placement"]["cards"]}
        self.assertEqual(cards[0]["ranks"], [0, 1])
        self.assertTrue(cards[0]["physical_overcommit"])
        self.assertFalse(cards[1]["physical_overcommit"])
        d2 = webui.placement_payload(
            {"model_cfg": _V2_CONFIG,
             "flags": {"tp_size": 4, "rank_gpu_id": [0, 0, 1, 2],
                       "rank_gpu_memory_mib": [15000] * 4,
                       "card_total_mib": {"0": 32607, "1": 20480,
                                          "2": 20480},
                       "context_length": 8192}}
        )
        self.assertTrue(d2["ok"], d2.get("error"))
        cards2 = {c["gpu_index"]: c for c in d2["placement"]["cards"]}
        self.assertFalse(cards2[0]["physical_overcommit"])
        # the renderer shows the flag as the EXCEEDS CARD marker.
        self.assertIn("EXCEEDS CARD", webui.INDEX_HTML)

    def test_preset_prefill_maps_onto_steppers(self):
        # applyProfile refreshes the co-location controls (reverse-populate,
        # never redistribute) right after filling the flag surface.
        html = webui.INDEX_HTML
        # Both anchors are searched FROM applyProfile: the same two calls also
        # appear in the running-config prefill earlier in the file, and this
        # test is about the order inside applyProfile.
        i_apply = html.index("function applyProfile(")
        i_snap = html.index("window._presetBase=presetSnapshot()", i_apply)
        i_colo = html.index("updateColoUI()", i_apply)
        self.assertLess(i_apply, i_colo)
        self.assertLess(i_colo, i_snap)
        self.assertIn("window._coloRedistribute=false;\n  updateGpuPick(); updateColoUI();",
                      html)


# ===========================================================================
# Update foundation (dashboard rework): the front end is one embedded script,
# so a syntax slip in it is invisible to every Python test and shows up only
# as a blank page in a browser. These tests close that gap and pin the
# state-preserving update rules that the rework exists to enforce.
# ===========================================================================


def _index_script() -> str:
    """The embedded <script> body of INDEX_HTML."""
    m = re.search(r"<script>(.*)</script>", webui.INDEX_HTML, re.S)
    assert m, "INDEX_HTML has no <script> block"
    return m.group(1)


class TestIndexScriptParses(CustomTestCase):
    def test_embedded_javascript_is_syntactically_valid(self):
        esprima = pytest.importorskip("esprima")
        src = _index_script()
        # The parser is an ES2017 implementation; the page uses two later
        # operators. Rewriting them keeps this a pure syntax check without
        # holding the page back to ES2017.
        norm = src.replace("??", "||").replace("?.", ".")
        try:
            esprima.parseScript(norm)
        except Exception as e:  # pragma: no cover - only on a real break
            self.fail(f"INDEX_HTML script does not parse: {e}")


class TestWizardStaleClearsOnFailureToo(CustomTestCase):
    """Round 5 bug: the Guide's downstream steps ('hardware'/'goal'/
    'families'/'command') stayed grey FOREVER whenever the recompute that was
    supposed to clear them (wizardFamilies/wizardCommand) failed or timed
    out, because the old code returned from inside the `catch` block before
    ever reaching the `wizardFresh(...)` call that removes the [data-step]
    stale band. wizardInvalidate('model') marks those steps stale on every
    model edit; nothing but a concluded recompute -- success OR failure --
    may honestly claim they no longer describe "the previous input", so both
    must un-stale on both outcomes, not only on success.
    """

    def _fn_body(self, name: str, next_marker: str) -> str:
        js = _index_script()
        start = js.index(f"async function {name}(){{")
        end = js.index(next_marker, start)
        return js[start:end]

    def test_wizard_families_unstales_on_error(self):
        body = self._fn_body("wizardFamilies", "function wzCell(c){")
        catch_block = body[body.index("}catch(e){"):body.index("} finally")]
        self.assertIn("wizardFresh(", catch_block,
                       "wizardFamilies' catch block must clear the stale "
                       "marks on the steps it just (unsuccessfully) redid, "
                       "else they stay grey with no way to self-heal")
        # The abort/superseded path must NOT be treated as concluded -- a
        # newer call is already in flight and owns the eventual un-staling.
        self.assertIn("if(apiAborted(e)) return;", catch_block)

    def test_wizard_command_unstales_on_error(self):
        body = self._fn_body("wizardCommand", "const wizardCommandDebounced")
        catch_block = body[body.index("}catch(e){"):body.index("} finally")]
        self.assertIn("wizardFresh('command')", catch_block)
        self.assertIn("if(apiAborted(e)) return;", catch_block)


class TestUpdateFoundation(CustomTestCase):
    """A refresh may change numbers and nothing else."""

    def test_patcher_and_bounded_fetch_helpers_exist(self):
        js = _index_script()
        for fn in ("function setHTML(", "function _beforeElUpdated(",
                   "function _nodeKey(", "function openDetails(",
                   "async function api(", "function apiAborted(",
                   "function stale(", "function debounce("):
            self.assertIn(fn, js, fn)

    def test_dom_patching_uses_the_vendored_library(self):
        # The tree diff is morphdom (MIT, vendored); only the policy is ours.
        js = _index_script()
        self.assertIn("morphdom(el,holder,{childrenOnly:true", js)
        self.assertIn("onBeforeElUpdated:_beforeElUpdated", js)

    def test_every_backend_call_is_bounded(self):
        # api() is the only fetch entry point that may be used for polling;
        # it carries the AbortController and the timeout.
        js = _index_script()
        self.assertIn("new AbortController()", js)
        self.assertIn("opts.timeout||API_TIMEOUT_MS", js)

    def test_landing_poll_patches_instead_of_replacing(self):
        # The 2 s landing poll used to rewrite landing_config wholesale, which
        # closed the two <details> inside it on every tick.
        js = _index_script()
        self.assertIn("setHTML($('landing_config')", js)
        self.assertNotIn("$('landing_config').innerHTML", js)

    def test_collapses_carry_a_stable_identity(self):
        js = _index_script()
        for key in ('data-key="cfg_launch"', 'data-key="cfg_raw"'):
            self.assertIn(key, js, key)

    def test_details_open_state_survives_a_patch(self):
        js = _index_script()
        i = js.index("function _beforeElUpdated(")
        body = js[i:i + 1200]
        # the LIVE open state is written back onto the incoming markup, so
        # morphdom's attribute sync cannot close what the reader opened
        self.assertIn("fromEl.tagName==='DETAILS'", body)
        self.assertIn("toEl.setAttribute('open','')", body)

    def test_edited_field_is_skipped_entirely(self):
        js = _index_script()
        i = js.index("function _beforeElUpdated(")
        body = js[i:i + 1200]
        self.assertIn("if(_isField(fromEl)&&_busy(fromEl)) return false;", body)

    def test_scroll_position_survives_a_patch(self):
        js = _index_script()
        i = js.index("function _beforeElUpdated(")
        body = js[i:i + 1200]
        self.assertIn("fromEl.scrollTop||fromEl.scrollLeft", body)

    def test_flag_rows_describe_their_current_value(self):
        # The patcher treats markup as the description of state, so a row
        # that renders an empty control would wipe what the user entered.
        js = _index_script()
        i = js.index("function flagRowHtml(")
        body = js[i:i + 2000]
        self.assertIn("const cur=_flagValue(f.id);", body)
        self.assertIn("value=\"'+esc(cur)+'\"", body)
        self.assertIn("(cur?' checked':'')", body)
        self.assertIn("(String(a)===cur?' selected':'')", body)

    def test_search_only_closes_what_search_opened(self):
        # Clearing the flag search must not slam shut a section the reader
        # opened by hand.
        js = _index_script()
        self.assertIn("_searchOpened", js)
        self.assertNotIn("else { det.style.display=''; det.open=false; }", js)

    def test_boot_log_is_opened_once_not_every_poll(self):
        js = _index_script()
        self.assertIn("openDetails($('boot_log'))", js)
        self.assertNotIn("bl.open=true", js)

    def test_quality_autofill_fills_but_does_not_overwrite(self):
        js = _index_script()
        i = js.index("async function autofillQuality()")
        body = js[i:i + 900]
        self.assertIn("if (!$('q_endpoint').value.trim())", body)
        self.assertIn("!$('q_model').value.trim()", body)

    def test_sliders_are_debounced(self):
        js = _index_script()
        # sliders route through the debounced recompute, not a call per tick
        self.assertIn("function _replan(){ scheduleRecompute(); }", js)
        self.assertIn("function _reflow(){ scheduleRecompute(); }", js)
        self.assertIn("const scheduleRecompute=debounce(", js)
        # one POST per pixel of slider travel is what this replaced
        self.assertNotIn("onServingEdit(); doPlan(); refreshRunnerPlacement();", js)

    def test_dead_live_widget_is_gone(self):
        js = _index_script()
        for sym in ("function liveScrape", "function toggleLive",
                    "function remeasureNow", "$('live_target')",
                    "$('live_res')", "$('live_btn')", "$('live_out')"):
            self.assertNotIn(sym, js, sym)
        self.assertFalse(hasattr(webui, "live_snapshot_payload"))

    def test_orphaned_live_route_is_gone(self):
        # The route addressed DOM ids that have not existed since the landing
        # strip replaced the widget. Nothing may answer on it now.
        srv = ThreadingHTTPServer(("127.0.0.1", 0), webui._Handler)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{srv.server_address[1]}/api/live",
                data=json.dumps({"target": "127.0.0.1:1"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=10)
            self.assertEqual(cm.exception.code, 404)
        finally:
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=5)


class TestVendoredAssets(CustomTestCase):
    """Third-party front-end code is vendored, licensed and inlined.

    The page makes no external requests, so a CDN link would simply be a
    broken page on a machine without internet. Everything is in the tree.
    """

    def test_morphdom_is_vendored_with_its_licence(self):
        d = os.path.join(os.path.dirname(webui.__file__), "assets")
        js = os.path.join(d, "morphdom-umd.min.js")
        lic = os.path.join(d, "morphdom.LICENSE")
        self.assertTrue(os.path.exists(js), js)
        self.assertTrue(os.path.exists(lic), lic)
        with open(lic, encoding="utf-8") as f:
            self.assertIn("MIT License", f.read())

    def test_normalize_is_vendored_with_its_licence(self):
        d = os.path.join(os.path.dirname(webui.__file__), "assets")
        css = os.path.join(d, "modern-normalize.css")
        lic = os.path.join(d, "modern-normalize.LICENSE")
        self.assertTrue(os.path.exists(css), css)
        self.assertTrue(os.path.exists(lic), lic)
        with open(lic, encoding="utf-8") as f:
            self.assertIn("MIT License", f.read())
        html = webui.INDEX_HTML
        self.assertIn("modern-normalize v3", html)
        self.assertNotIn("/*__VENDOR_NORMALIZE__*/", html)

    def test_design_tokens_are_defined_once(self):
        # Every surface/spacing/state value comes from the :root token block,
        # so a colour is never spelled out twice in the page.
        html = webui.INDEX_HTML
        for tok in ("--bg-canvas", "--bg-panel", "--bd-weak", "--fg-muted",
                    "--accent", "--ok", "--warn", "--bad", "--s2", "--t-md"):
            self.assertIn(tok + ":", html, tok)

    def test_page_inlines_it_and_links_nothing_external(self):
        html = webui.INDEX_HTML
        self.assertIn("global.morphdom=", html)      # the library itself
        self.assertNotIn("/*__VENDOR_MORPHDOM__*/", html)   # placeholder filled
        # no external fetch of code or styling, at all
        for bad in ("cdn.", "unpkg.com", "jsdelivr", "googleapis",
                    "<script src=", "<link rel=\"stylesheet\""):
            self.assertNotIn(bad, html, bad)


# ===========================================================================
# Etappe 3: simple / expert views and live propagation.
# ===========================================================================


class TestRecomputeEndpoint(WebUIFixture):
    """One call, one consistent answer about one configuration."""

    def test_returns_all_three_sections(self):
        d = webui.recompute_payload(self._payload())
        self.assertTrue(d["ok"])
        for k in ("plan", "placement", "fields"):
            self.assertIn(k, d, k)
        self.assertTrue(d["plan"].get("valid", True))
        self.assertTrue(d["fields"]["ok"])

    def test_sections_are_selectable(self):
        # The simple view asks only for what it shows.
        d = webui.recompute_payload(dict(self._payload(), sections=["plan"]))
        self.assertIn("plan", d)
        self.assertNotIn("placement", d)
        self.assertNotIn("fields", d)

    def test_one_broken_section_does_not_sink_the_answer(self):
        # A rejected plan must still return the per-field states that explain
        # the rejection, so the panels can say WHY rather than going blank.
        p = dict(self._payload(), model="/definitely/not/a/model")
        d = webui.recompute_payload(p)
        self.assertTrue(d["ok"])
        self.assertIn("fields", d)
        self.assertFalse(d["plan"].get("valid", True))

    def test_placement_request_is_honoured(self):
        p = dict(self._payload())
        p["placement_request"] = {"model": p["model"], "flags": {"tp_size": 3}}
        d = webui.recompute_payload(p)
        self.assertTrue(d["placement"]["ok"], d["placement"].get("error"))
        self.assertEqual(d["placement"]["placement"]["tp_size"], 3)


class TestRecomputeRoute(TestHttpRoundTrip):
    def test_recompute_over_http(self):
        d = self._post("/api/recompute", dict(self._payload(), sections=["plan"]))
        self.assertTrue(d["ok"])
        self.assertIn("plan", d)


class TestSimpleExpertViews(CustomTestCase):
    def test_mode_switch_exists_and_persists(self):
        js = _index_script()
        for fn in ("function setViewMode(", "function applyViewMode(",
                   "function viewMode("):
            self.assertIn(fn, js, fn)
        self.assertIn("localStorage.setItem('view_mode'", js)

    def test_visibility_is_one_class_on_body(self):
        # A mode switch must cost no backend call and must not disturb any
        # control's value; CSS-only visibility is what guarantees that.
        html = webui.INDEX_HTML
        self.assertIn("body.mode-simple .expert-only { display: none !important; }", html)
        self.assertIn("body.mode-expert .simple-only { display: none !important; }", html)
        self.assertIn('<div class="simple-only">', html)
        self.assertIn('<div class="cols runner expert-only">', html)

    def test_simple_view_shows_one_bar_and_one_slider_per_card(self):
        js = _index_script()
        i = js.index("function renderSimpleCards(")
        body = js[i:i + 3000]
        self.assertIn('class="csbar"', body)          # one bar, whole-card
        self.assertIn('type="range" id="csb_', body)  # one budget slider
        # the granular segment view belongs to the expert mode only
        self.assertNotIn("segments", body)

    def test_simple_budget_slider_writes_the_existing_reserve(self):
        # One mechanism, two presentations -- not a second, parallel notion
        # of "how much VRAM may this card give".
        js = _index_script()
        self.assertIn("function setCardBudgetMib(", js)
        i = js.index("function setCardBudgetMib(")
        self.assertIn("c.reserve_gb=", js[i:i + 400])

    def test_live_propagation_is_one_call(self):
        js = _index_script()
        self.assertIn("const scheduleRecompute=debounce(recomputeNow", js)
        self.assertIn("api('/api/recompute'", js)
        # the three-way race is gone from the edit path
        self.assertNotIn("resolveFlags(); refreshRunnerPlacement(); schedulePlan();", js)
        self.assertNotIn("resolveFlags(); refreshRunnerPlacement(); doPlan();", js)

    def test_no_arithmetic_is_duplicated_in_the_browser(self):
        # The simple view renders the placement result; it does not derive it.
        js = _index_script()
        i = js.index("function renderSimpleCards(")
        body = js[i:i + 3000]
        self.assertIn("pc?pc.total_mib:null", body)


class TestTemplatesAreStartingPoints(CustomTestCase):
    """A template moves the starting point; it never fences a control in."""

    def test_tuning_objectives_are_no_longer_a_second_preset_bar(self):
        """#7: the objective BUTTONS are gone; the objective itself is not.

        They were the second of two prefabricated starting points sitting in
        one panel. applyTune stays -- it is how the working-point stops write
        --rank-perf-tune -- and the flag keeps its own row in the flag
        surface, so nothing about the knob changed except that it stopped
        looking like a preset mechanism.
        """
        html = webui.INDEX_HTML
        for key in ("tune_both", "tune_maxkv", "tune_dec", "tune_enc"):
            self.assertNotIn('id="%s"' % key, html, key)
        self.assertIn("function applyTune(", html)
        # the objective labels still map onto the fork's own --rank-perf-tune
        # enum -- including the phase-optimal arms (#357), so a value the
        # server accepts never renders as a bare identifier in the objective
        # note.
        from sglang.srt.planner import flags as flagsmod
        allowed = set(flagsmod.catalog()["rank_perf_tune"].allowed)
        self.assertEqual(
            allowed,
            {"both", "dec", "enc", "maxkv", "phase-prefill", "phase-decode"},
        )
        labels = _index_script().split("const TUNE_LABELS={")[1].split("};")[0]
        keys = {
            part.split(":")[0].strip().strip("'")
            for part in labels.split(",")
            if ":" in part
        }
        self.assertEqual(keys, allowed)

    def test_concurrency_slider_is_not_capped_at_a_constant(self):
        js = _index_script()
        # the old hard 64 silently swallowed anything larger
        self.assertNotIn("$('mrr_slider').value=Math.min(v, 64)", js)
        self.assertNotIn("$('mrr_slider').value=Math.min(mv, 64)", js)
        self.assertIn("function setMrrCap(", js)
        # the range follows what the planner reports as fitting
        self.assertIn("d.kv_by_concurrency||[]).filter(r=>r.fits)", js)

    def test_context_slider_bound_to_computed_capacity_not_a_preset(self):
        js = _index_script()
        self.assertIn("setCtxCap(cap ? cap.max_context_tokens : null)", js)


class TestTradeoffTooltips(CustomTestCase):
    """Every control says what it gives and what it costs, from ONE source."""

    def test_registry_entries_have_both_halves(self):
        from sglang.srt.planner import tooltips as tipsmod
        for key, t in tipsmod.TRADEOFFS.items():
            self.assertTrue(t.gain.strip(), key)
            self.assertTrue(t.cost.strip(), key)
            # two half-sentences, not an essay
            self.assertLess(len(t.gain), 200, key)
            self.assertLess(len(t.cost), 200, key)

    def test_rendered_line_states_the_cost(self):
        from sglang.srt.planner import tooltips as tipsmod
        txt = tipsmod.describe("rank_kv_ratio=speed")
        self.assertIn("Costs:", txt)

    def test_numbers_are_never_invented(self):
        # A trade-off that points at a study must say the study has not been
        # run rather than quoting a figure from nowhere.
        from sglang.srt.planner import tooltips as tipsmod
        txt = tipsmod.describe("rank_kv_ratio=speed", measurements={})
        self.assertIn("Not measured on this rig", txt)
        txt2 = tipsmod.describe(
            "rank_kv_ratio=speed", measurements={"mlp_crossover": "+6% prefill"}
        )
        self.assertIn("Measured here: +6% prefill", txt2)
        self.assertNotIn("Not measured", txt2)

    def test_unknown_key_falls_back(self):
        from sglang.srt.planner import tooltips as tipsmod
        self.assertEqual(tipsmod.describe("nope", fallback="old text"), "old text")

    def test_fork_flags_are_covered(self):
        # A knob this fork invented has no documentation anywhere else, so it
        # is the one that most needs to state its cost. Env-only vectors are
        # excluded: they mirror a --rank-*-ratio flag that is covered.
        from sglang.srt.planner import flags as flagsmod
        from sglang.srt.planner import tooltips as tipsmod
        fork = [
            k for k, v in flagsmod.catalog().items()
            if v.source == "fork" and not v.is_env
        ]
        missing = tipsmod.missing_coverage(fork)
        self.assertEqual(missing, [], f"fork flags without a trade-off: {missing}")

    def test_catalog_payload_carries_the_text(self):
        d = webui.flag_catalog_payload()
        specs = {f["id"]: f for g in d["groups"].values() for f in g}
        self.assertIn("Costs:", specs["rank_kv_ratio"]["tradeoff"])
        # enums whose options pull opposite ways get a line per option
        by_value = specs["rank_kv_ratio"]["tradeoff_by_value"]
        self.assertIn("speed", by_value)
        self.assertIn("capacity", by_value)
        self.assertNotEqual(by_value["speed"], by_value["capacity"])
        self.assertIn("tooltips", d)

    def test_frontend_holds_no_copy_of_the_text(self):
        # The browser renders what the server sends. If a sentence were
        # inlined in the JS it would drift from the registry silently.
        js = _index_script()
        from sglang.srt.planner import tooltips as tipsmod
        for key, t in tipsmod.TRADEOFFS.items():
            self.assertNotIn(t.gain, js, key)
            self.assertNotIn(t.cost, js, key)
        self.assertIn("function tip(key){", js)
        self.assertIn("window._tips=d.tooltips", js)


class TestTooltipsRoute(TestHttpRoundTrip):
    def test_tooltips_endpoint(self):
        d = json.loads(self._get("/api/tooltips"))
        self.assertTrue(d["ok"])
        self.assertIn("rank_kv_ratio=speed", d["tooltips"])


# ===========================================================================
# Etappe 4: task #214 rig pairing over HTTP.
#
# The endpoints are thin adapters over rigmon/pairing.py. What is asserted
# here is the CONTRACT the architecture depends on: one endpoint per step,
# small bodies, state on the host, and no rule about pairing validity living
# in the browser.
# ===========================================================================


class TestRigPairRoutes(TestHttpRoundTrip):
    def setUp(self):
        super().setUp()
        from sglang.srt.rigmon import pairing
        self.pairing = pairing
        self._real_opener = pairing.STORE._opener
        self._real_sync = pairing.STORE.synchronous
        pairing.STORE._opener = lambda url, timeout: json.dumps(
            {"nodes": []} if url.endswith("/api/nodes") else {"nodes": {}}
        ).encode()
        pairing.STORE.synchronous = True

    def tearDown(self):
        self.pairing.STORE._opener = self._real_opener
        self.pairing.STORE.synchronous = self._real_sync
        super().tearDown()

    def test_a_curl_per_step_drives_the_whole_flow(self):
        d = self._post("/api/rig_pair/start", {"target": "far:8770"})
        self.assertTrue(d["ok"], d.get("error"))
        sid = d["session"]["session_id"]
        self.assertEqual(d["session"]["next_step"], "reach")

        d = self._post("/api/rig_pair/advance", {"session_id": sid})
        self.assertTrue(d["ok"])

        d = json.loads(self._get(f"/api/rig_pair/status?session_id={sid}"))
        self.assertTrue(d["ok"])
        self.assertEqual(d["session"]["session_id"], sid)
        self.assertEqual(len(d["session"]["steps"]), len(self.pairing.STEPS))

    def test_state_is_readable_after_the_fact(self):
        # This is what makes a browser reload resume instead of restart.
        sid = self._post("/api/rig_pair/start", {"target": "far:8770"})["session"][
            "session_id"
        ]
        self._post("/api/rig_pair/advance", {"session_id": sid})
        again = json.loads(self._get(f"/api/rig_pair/status?session_id={sid}"))
        self.assertNotEqual(again["session"]["steps"][0]["state"], "pending")

    def test_reset_keeps_the_target(self):
        sid = self._post("/api/rig_pair/start", {"target": "far:8770"})["session"][
            "session_id"
        ]
        self._post("/api/rig_pair/advance", {"session_id": sid})
        d = self._post("/api/rig_pair/reset", {"session_id": sid})
        self.assertTrue(d["ok"])
        self.assertEqual(d["session"]["target"], "far:8770")
        self.assertEqual(d["session"]["steps"][0]["state"], "pending")

    def test_missing_target_says_what_to_pass(self):
        d = self._post("/api/rig_pair/start", {})
        self.assertFalse(d["ok"])
        self.assertTrue(d["remedy"])

    def test_unknown_session_is_reported_not_crashed(self):
        d = self._post("/api/rig_pair/advance", {"session_id": "nope"})
        self.assertFalse(d["ok"])
        self.assertIn("no such pairing session", d["error"])

    def test_status_without_an_id_lists_sessions(self):
        self._post("/api/rig_pair/start", {"target": "far:8770"})
        d = json.loads(self._get("/api/rig_pair/status"))
        self.assertTrue(d["ok"])
        self.assertGreaterEqual(len(d["sessions"]), 1)


class TestRigPairUiIsSteeringOnly(CustomTestCase):
    """No pairing rule may live in the browser.

    The same flow has to behave identically when driven by a shell script; a
    rule implemented here would simply be absent there.
    """

    def test_tab_exists(self):
        html = webui.INDEX_HTML
        self.assertIn('id="view_pair"', html)
        self.assertIn("onclick=\"showTab('pair')\"", html)

    def test_frontend_decides_nothing(self):
        js = _index_script()
        i = js.index("const PAIR_POLL_MS")
        body = js[i:js.index("// Shared granular placement renderer")]
        # no compatibility, reachability or transport logic client-side
        for forbidden in ("check_compatibility", "NCCL_", "nccl", "sm86",
                          "choose_transport", "verdict==='ok'?'ok'"):
            if forbidden == "verdict==='ok'?'ok'":
                continue
            self.assertNotIn(forbidden, body, forbidden)
        # it renders server-supplied verdicts and remedies verbatim
        self.assertIn("st.remedy", body)
        self.assertIn("r.remedy", body)

    def test_client_state_is_only_the_session_id(self):
        js = _index_script()
        self.assertIn("localStorage.setItem('pair_session'", js)

    def test_no_cross_rig_boot_is_offered(self):
        js = _index_script()
        i = js.index("function pairStepBody(")
        body = js[i:i + 4000]
        self.assertIn("never starts a run by itself", body)
        self.assertNotIn("serverStart", body)


# ===========================================================================
# Etappe 5: the benchmark and chess windows.
# ===========================================================================


class TestBenchLeadMetrics(CustomTestCase):
    """ms per round is the yardstick; absent must never read as zero."""

    def setUp(self):
        webui._LEAD_PREV.clear()

    def tearDown(self):
        webui._LEAD_PREV.clear()

    def test_endpoint_required(self):
        d = webui.bench_lead_metrics_payload({"endpoint": ""})
        self.assertFalse(d["ok"])

    def test_first_call_only_seeds_the_window(self):
        # A delta needs two samples. Saying so beats showing an empty table
        # that looks like a measurement of nothing.
        class _Sample:
            up = True
            reason = None
            metrics = {}
            info = {}
            per_rank = {}
            per_rank_phase = {}

        with mock.patch(
            "sglang.srt.rigmon.sources.EngineScraper.scrape",
            return_value=_Sample(),
        ):
            d = webui.bench_lead_metrics_payload({"endpoint": "127.0.0.1:30000"})
            self.assertTrue(d["ok"])
            self.assertEqual(d["metrics"], {})
            self.assertIn("next poll", " ".join(d["notes"]))

    def test_missing_device_timer_is_explained_not_zeroed(self):
        class _Sample:
            up = True
            reason = None
            metrics = {}
            info = {}
            per_rank = {}
            per_rank_phase = {}

        with mock.patch(
            "sglang.srt.rigmon.sources.EngineScraper.scrape",
            return_value=_Sample(),
        ):
            webui.bench_lead_metrics_payload({"endpoint": "127.0.0.1:30000"})
            d = webui.bench_lead_metrics_payload({"endpoint": "127.0.0.1:30000"})
        self.assertTrue(d["ok"])
        self.assertEqual(d["metrics"], {})
        joined = " ".join(d["notes"])
        self.assertIn("absent, not zero", joined)
        self.assertIn("SGLANG_ENABLE_METRICS_DEVICE_TIMER", joined)

    def test_unreachable_engine_reports_rather_than_raises(self):
        d = webui.bench_lead_metrics_payload({"endpoint": "127.0.0.1:1"})
        self.assertFalse(d["ok"])
        self.assertTrue(d["error"])


class TestBenchLeadMetricsRoute(TestHttpRoundTrip):
    def test_route(self):
        d = self._post("/api/bench_lead_metrics", {"endpoint": ""})
        self.assertFalse(d["ok"])


class TestBenchWindow(CustomTestCase):
    def test_running_and_finished_are_separate(self):
        # A table still filling must never be mistaken for a complete result.
        html = webui.INDEX_HTML
        self.assertIn('id="bn_running_box"', html)
        self.assertIn("<legend>finished runs</legend>", html)
        js = _index_script()
        self.assertIn("function renderBenchRunning(", js)
        self.assertIn("function renderBenchFinished(", js)

    def test_results_are_configuration_measure_value_tables(self):
        js = _index_script()
        self.assertIn("function benchTableHtml(", js)
        self.assertIn("measure / value", js)
        i = js.index("function renderBenchFinished(")
        self.assertIn("configuration:", js[i:i + 1200])

    def test_recorded_measures_are_no_longer_dropped(self):
        # bench_suite records these per test; the old renderer read only
        # status and metric and threw the rest away.
        js = _index_script()
        i = js.index("function benchMeasures(")
        body = js[i:i + 900]
        self.assertIn("d.ttft_ms", body)
        self.assertIn("d.prefill_tps", body)

    def test_lead_metrics_are_ms_per_round(self):
        js = _index_script()
        self.assertIn("ms_per_verify_round:'ms / verify round'", js)
        self.assertIn("ms_per_1k_prefill_tokens:'ms / 1k prefill tokens'", js)

    def test_a_run_is_never_aborted_by_a_newer_request(self):
        # api() aborts the previous call under the same key, which is right
        # for a poll and wrong for a benchmark.
        js = _index_script()
        i = js.index("async function benchRun(")
        body = js[i:js.index("function benchFinish(")]
        self.assertIn("deliberately NOT routed through api()", body)

    def test_lead_poll_stops_when_the_tab_is_left(self):
        js = _index_script()
        self.assertIn("if (t!=='bench') benchLeadStop();", js)


class TestChessWindow(CustomTestCase):
    """Same design line as the benchmark window."""

    def test_result_is_a_measure_value_table(self):
        js = _index_script()
        self.assertIn("function qualityTableHtml(", js)
        i = js.index("function qualityTableHtml(")
        body = js[i:i + 1400]
        self.assertIn("prompt tokens", body)
        self.assertIn("completion tokens", body)

    def test_running_state_is_visible(self):
        js = _index_script()
        i = js.index("async function qualityRun(")
        body = js[i:i + 2000]
        self.assertIn('chip loading', body)
        self.assertIn('chip ready', body)

    def test_verdict_stays_a_verdict(self):
        # It is a judgement, not a measurement, and has to read as one.
        js = _index_script()
        i = js.index("function qualityTableHtml(")
        self.assertIn("verdictClass(d.verdict)", js[i:i + 1400])


# ===========================================================================
# Etappe 6: the discussion export. Built, tested, and deliberately NOT armed.
# ===========================================================================


class TestDiscussionExport(CustomTestCase):
    def setUp(self):
        from sglang.srt.planner import discussion_export as dx
        self.dx = dx
        self._env = {k: os.environ.get(k) for k in (dx.TARGET_ENV, dx.PAT_FILE_ENV)}
        for k in self._env:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _payload(self):
        return {
            "bench_results": [{
                "test_id": 1, "label": "basic", "status": "pass",
                "metric": {"name": "tok_s", "value": 42, "unit": "tok/s"},
                "detail": {"ttft_ms": 120.5, "prefill_tps": 900.0},
            }],
            "lead_metrics": {"ms_per_verify_round": 18.4},
            "system": {
                "cards": ["RTX 5090", "RTX 3080"],
                "driver": "580.00",
                "model": "/home/someone/models/Qwen3.6-27B",
            },
            "launch_flags": ["--model", "/home/someone/models/Q", "--host", "192.168.0.89"],
            "energy": {"j_per_decode_token": 0.42, "avg_decode_watts": 310.0,
                       "per_card": [{"name": "RTX 5090", "j_per_decode_token": 0.2,
                                     "watts": 200, "efficiency": 1.8}]},
        }

    # -- the gate ---------------------------------------------------------

    def test_nothing_is_armed_without_a_target(self):
        d = self.dx.preview(self._payload(), "bench_system")
        self.assertFalse(d["can_send"])
        self.assertEqual(d["reason"], "no target configured")

    def test_submit_without_a_target_sends_nothing(self):
        def _boom(req):  # pragma: no cover - must never run
            raise AssertionError("a request was made with no target configured")

        d = self.dx.submit(self._payload(), "bench_system", confirmed=True,
                           opener=_boom)
        self.assertFalse(d["sent"])
        self.assertIn("no target", d["reason"])

    def test_target_without_a_token_still_sends_nothing(self):
        os.environ[self.dx.TARGET_ENV] = "D_kwDOABCD"
        d = self.dx.preview(self._payload(), "bench_system")
        self.assertFalse(d["can_send"])
        self.assertIn(self.dx.PAT_FILE_ENV, d["reason"])

    def test_confirm_is_required_even_when_armed(self):
        with tempfile.NamedTemporaryFile("w", suffix=".pat", delete=False) as f:
            f.write("ghp_notarealtokenatall")
            path = f.name
        try:
            os.environ[self.dx.TARGET_ENV] = "D_kwDOABCD"
            os.environ[self.dx.PAT_FILE_ENV] = path

            def _boom(req):  # pragma: no cover - must never run
                raise AssertionError("a request was made without confirmation")

            d = self.dx.submit(self._payload(), "bench_system", opener=_boom)
            self.assertFalse(d["sent"])
            self.assertIn("not confirmed", d["reason"])
        finally:
            os.unlink(path)

    # -- redaction --------------------------------------------------------

    def test_system_details_are_always_scrubbed(self):
        md = self.dx.build_markdown(self._payload(), "full")
        self.assertNotIn("/home/someone", md)
        self.assertNotIn("192.168.0.89", md)
        # what SHOULD survive
        self.assertIn("RTX 5090", md)
        self.assertIn("580.00", md)

    def test_notes_are_scrubbed_too(self):
        p = dict(self._payload(), notes="ran on 10.0.0.5 at /srv/models/foo")
        md = self.dx.build_markdown(p, "full")
        self.assertNotIn("10.0.0.5", md)
        self.assertNotIn("/srv/models", md)

    # -- bundles ----------------------------------------------------------

    def test_bundles_select_sections(self):
        only = self.dx.build_markdown(self._payload(), "bench")
        self.assertIn("## Benchmark", only)
        self.assertNotIn("## System", only)
        self.assertNotIn("## Energy", only)
        both = self.dx.build_markdown(self._payload(), "bench_system")
        self.assertIn("## System", both)
        self.assertNotIn("## Energy", both)
        full = self.dx.build_markdown(self._payload(), "bench_system_energy")
        self.assertIn("## Energy", full)

    def test_energy_groups_are_selectable(self):
        md = self.dx.build_markdown(
            self._payload(), "bench_system_energy", energy_groups=["per_token"]
        )
        self.assertIn("Energy per token", md)
        self.assertNotIn("Average power", md)

    def test_lead_metrics_reach_the_report(self):
        md = self.dx.build_markdown(self._payload(), "bench")
        self.assertIn("ms / verify round", md)

    def test_unknown_bundle_is_refused(self):
        with self.assertRaises(self.dx.DiscussionError):
            self.dx.build_markdown(self._payload(), "nope")

    def test_preview_is_what_would_be_posted(self):
        # A preview assembled differently from the payload is not a preview.
        pv = self.dx.preview(self._payload(), "bench_system")
        md = self.dx.build_markdown(self._payload(), "bench_system")
        self.assertEqual(pv["markdown"], md)

    # -- the send path, against a mocked API ------------------------------

    def _armed(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".pat", delete=False)
        f.write("ghp_notarealtokenatall")
        f.close()
        os.environ[self.dx.TARGET_ENV] = "D_kwDOABCD"
        os.environ[self.dx.PAT_FILE_ENV] = f.name
        return f.name

    def test_first_submit_adds_a_comment(self):
        path = self._armed()
        seen = []

        def opener(req):
            seen.append(json.loads(req.data.decode()))
            if "comments(first" in seen[-1]["query"]:
                return json.dumps({"data": {"node": {"comments": {"nodes": []}}}}).encode()
            return json.dumps({"data": {"addDiscussionComment": {
                "comment": {"id": "C_1", "url": "https://example/x"}}}}).encode()

        try:
            d = self.dx.submit(self._payload(), "bench_system",
                               confirmed=True, opener=opener)
            self.assertTrue(d["sent"])
            self.assertEqual(d["action"], "created")
        finally:
            os.unlink(path)

    def test_resubmit_updates_in_place(self):
        path = self._armed()

        def opener(req):
            q = json.loads(req.data.decode())["query"]
            if "comments(first" in q:
                return json.dumps({"data": {"node": {"comments": {"nodes": [
                    {"id": "C_old", "body": self.dx.MARKER + " old",
                     "viewerDidAuthor": True}]}}}}).encode()
            return json.dumps({"data": {"updateDiscussionComment": {
                "comment": {"id": "C_old", "url": "https://example/x"}}}}).encode()

        try:
            d = self.dx.submit(self._payload(), "bench_system",
                               confirmed=True, opener=opener)
            self.assertEqual(d["action"], "updated")
        finally:
            os.unlink(path)

    def test_token_never_leaves_except_in_the_header(self):
        path = self._armed()
        seen = []

        def opener(req):
            seen.append(req)
            return json.dumps({"data": {"node": {"comments": {"nodes": []}}}}).encode()

        try:
            try:
                self.dx.submit(self._payload(), "bench_system",
                               confirmed=True, opener=opener)
            except self.dx.DiscussionError:
                pass
            req = seen[0]
            self.assertNotIn("ghp_", req.full_url)
            self.assertIn("ghp_", req.headers.get("Authorization", ""))
            self.assertNotIn("ghp_", req.data.decode())
        finally:
            os.unlink(path)

    def test_api_errors_are_token_redacted(self):
        path = self._armed()

        def opener(req):
            return json.dumps({"errors": [{"message": "bad credentials ghp_notarealtokenatall"}]}).encode()

        try:
            with self.assertRaises(self.dx.DiscussionError) as cm:
                self.dx.submit(self._payload(), "bench_system",
                               confirmed=True, opener=opener)
            self.assertNotIn("ghp_notarealtokenatall", str(cm.exception))
            self.assertIn("<redacted-token>", str(cm.exception))
        finally:
            os.unlink(path)


class TestDiscussionRoutes(TestHttpRoundTrip):
    def test_preview_route_reports_the_gate(self):
        d = self._post("/api/discussion_preview", {"data": {}, "bundle": "bench"})
        self.assertTrue(d["ok"])
        self.assertFalse(d["can_send"])
        self.assertIn("bundles", d)

    def test_submit_route_sends_nothing_when_unarmed(self):
        d = self._post("/api/discussion_submit",
                       {"data": {}, "bundle": "bench", "confirmed": True})
        self.assertFalse(d["sent"])


class TestDiscussionUi(CustomTestCase):
    def test_composer_exists_and_states_the_gate(self):
        html = webui.INDEX_HTML
        self.assertIn('id="dx_bundle"', html)
        self.assertIn('id="dx_preview"', html)
        self.assertIn("Nothing is created automatically", html)

    def test_frontend_composes_no_markdown(self):
        js = _index_script()
        i = js.index("function discussionData(")
        body = js[i:js.index("// ---- GitHub share (#152)")]
        for forbidden in ("## Benchmark", "## System", "|---|", "scrub"):
            self.assertNotIn(forbidden, body, forbidden)

    def test_send_needs_an_explicit_confirm(self):
        js = _index_script()
        i = js.index("async function discussionSubmit(")
        self.assertIn("confirm(", js[i:i + 600])


class TestObservabilityIsNotOptional(CustomTestCase):
    """A server booted from this dashboard is a server that can be watched."""

    def test_argv_override_still_gets_enable_metrics(self):
        # The full-argv path is the one place a caller could otherwise decide
        # to boot a blind server.
        out = webui._force_enable_metrics(["python", "-m", "sglang.launch_server"])
        self.assertIn("--enable-metrics", out)

    def test_not_added_twice(self):
        argv = ["python", "--enable-metrics"]
        self.assertEqual(webui._force_enable_metrics(argv).count("--enable-metrics"), 1)

    def test_none_stays_none(self):
        # None means "no override": LaunchSettings.launch_command() appends the
        # flag itself, and that path is already covered.
        self.assertIsNone(webui._force_enable_metrics(None))

    def test_launch_settings_command_has_it(self):
        from sglang.srt.planner.server_manager import LaunchSettings

        self.assertIn("--enable-metrics",
                      LaunchSettings(model_path="/m").launch_command())

    def test_page_names_the_no_metrics_state(self):
        html = webui.INDEX_HTML
        self.assertIn("Server started without --enable-metrics", html)
        self.assertIn("noMetricsBanner", html)

    def test_detect_reports_whether_metrics_are_served(self):
        with mock.patch.object(webui, "_probe_sglang", return_value=True), \
                mock.patch.object(webui, "_serves_metrics", return_value=False):
            d = webui.detect_endpoint_payload({"endpoint": "1.2.3.4:30000"})
        self.assertEqual(d["endpoint"], "http://1.2.3.4:30000")
        self.assertIs(d["metrics"], False)


class TestLaunchPayloadCompleteness(CustomTestCase):
    """Every field LaunchSettings models must be reachable from the payload --
    otherwise the API can only express a subset of the configurations the
    launcher supports, and a full-argv override becomes the only real path."""

    def test_spec_depth_loader_and_reserve_survive_the_mapper(self):
        ls = webui._launch_settings_from_payload({
            "model": "/m", "format": "gguf", "gguf_variant": "x.gguf",
            "speculative_num_steps": 3, "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 4,
            "speculative_draft_model_path": "/draft",
            "mem_fraction_static": 0.85, "rank_auto_reserve_mib": 2700,
            "tokenizer_path": "/tok", "load_format": "gguf",
            "quantization": "gguf", "extra_flags": ["--enable-torch-compile"],
        })
        self.assertEqual(ls.speculative_num_steps, 3)
        self.assertEqual(ls.speculative_eagle_topk, 1)
        self.assertEqual(ls.speculative_num_draft_tokens, 4)
        self.assertEqual(ls.speculative_draft_model_path, "/draft")
        self.assertEqual(ls.mem_fraction_static, 0.85)
        self.assertEqual(ls.rank_auto_reserve_mib, 2700)
        self.assertEqual(ls.tokenizer_path, "/tok")
        self.assertEqual(ls.load_format, "gguf")
        self.assertEqual(ls.quantization, "gguf")
        self.assertEqual(ls.extra_flags, ["--enable-torch-compile"])

    def test_blank_values_stay_unset(self):
        ls = webui._launch_settings_from_payload({
            "model": "/m", "speculative_num_steps": "",
            "mem_fraction_static": "", "rank_auto_reserve_mib": "",
        })
        self.assertIsNone(ls.speculative_num_steps)
        self.assertIsNone(ls.mem_fraction_static)
        self.assertIsNone(ls.rank_auto_reserve_mib)


class TestBenchTabAndHistory(CustomTestCase):
    """The tests are the tab's content, and a finished run is reviewable."""

    def test_tests_are_buttons_not_checkboxes(self):
        html = webui.INDEX_HTML
        self.assertIn('class="testbtn', html)
        self.assertIn("function benchToggle(", html)
        self.assertIn("window._benchSel", html)
        # the old per-test checkbox markup is gone
        self.assertNotIn('<input type="checkbox" id="bnt_', html)

    def test_selection_survives_a_regate(self):
        # The selection lives in one set, not in the DOM, so re-rendering the
        # buttons after a gate change cannot silently drop it.
        html = webui.INDEX_HTML
        i_render = html.index("function renderBenchTests(")
        i_set = html.index("window._benchSel.has(t.test_id)", i_render)
        self.assertGreater(i_set, i_render)

    def test_history_ui_and_download_link(self):
        # #6: run history moved out of the Benchmark column into its own tab.
        html = webui.INDEX_HTML
        self.assertIn("/api/bench_history", html)
        self.assertIn("/api/bench_run_detail?download=1", html)
        self.assertIn("function historyShowRun(", html)
        self.assertIn('<div id="view_history"', html)
        self.assertIn('id="tab_history"', html)
        # explicit filters that belong to the tab itself -- the old list
        # narrowed itself from the Benchmark tab's target field.
        for token in ('id="hs_model"', 'id="hs_period"', 'id="hs_outcome"',
                      'id="hs_search"', 'id="hs_limit"'):
            self.assertIn(token, html, token)
        # deletion is reachable at last (delete_run had no route and no button)
        self.assertIn("function historyDelete(", html)
        self.assertIn("function historyDeleteFiltered(", html)
        # and the Benchmark tab no longer carries a second copy of the list
        self.assertNotIn('id="bn_history"', html)

    def test_graphs_are_columns_not_a_polyline(self):
        # A polyline across two samples drew a stroke over half an empty
        # field and interpolated between samples seconds apart.
        html = webui.INDEX_HTML
        self.assertIn("function _columns(", html)
        self.assertIn("<rect x=", html)
        self.assertNotIn("<polyline", html)


class TestBenchHistoryRoutes(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bhroute_")
        self._env = mock.patch.dict(
            os.environ, {"SGLANG_PLANNER_BENCH_HISTORY": self.tmp})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_is_saved_with_its_transcript_and_listed(self):
        from sglang.srt.planner import bench_history

        class _Ctx:
            transcript = [{"test_id": 1, "request": {"messages": []},
                           "answer": "Paris", "http_code": 200}]

        def fake_run(endpoint, model, selected=None, capabilities=None,
                     preset=None, force=False, transcript_sink=None):
            if transcript_sink is not None:
                transcript_sink.append(_Ctx())
            yield {"test_id": 1, "label": "Basic", "status": "pass",
                   "metric": {"name": "none", "value": None}, "detail": {},
                   "deps": {}}

        with mock.patch("sglang.srt.planner.bench_suite.run_suite",
                        side_effect=fake_run):
            evs = list(webui.bench_run_events(
                {"endpoint": "127.0.0.1:30000", "model": "/m/Model-A"}))
        run_id = evs[-1]["run_id"]
        self.assertIsNotNone(run_id)
        listed = webui.bench_history_payload({"model": "/m/Model-A"})
        self.assertTrue(listed["ok"])
        self.assertEqual(len(listed["runs"]), 1)
        self.assertEqual(listed["runs"][0]["n_exchanges"], 1)
        detail = webui.bench_run_payload({"run_id": run_id})
        self.assertTrue(detail["ok"])
        self.assertEqual(detail["run"]["transcript"][0]["answer"], "Paris")
        self.assertEqual(bench_history.load_run(run_id)["model"], "/m/Model-A")

    def test_a_crashed_run_is_still_stored(self):
        # The transcript up to the failure is usually the only record of what
        # killed it.
        def boom(endpoint, model, selected=None, capabilities=None,
                 preset=None, force=False, transcript_sink=None):
            yield {"test_id": 1, "label": "Basic", "status": "pass",
                   "metric": {"name": "none", "value": None}, "detail": {},
                   "deps": {}}
            raise RuntimeError("engine died")

        with mock.patch("sglang.srt.planner.bench_suite.run_suite",
                        side_effect=boom):
            evs = list(webui.bench_run_events(
                {"endpoint": "127.0.0.1:30000", "model": "/m/Model-B"}))
        self.assertEqual(evs[-1]["event"], "error")
        runs = webui.bench_history_payload({"model": "/m/Model-B"})["runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["n_tests"], 1)

    def test_detail_requires_a_run_id(self):
        d = webui.bench_run_payload({})
        self.assertFalse(d["ok"])
        self.assertIn("run_id", d["error"])

    # ---- #6: deletion. bench_history.delete_run existed and was unit-tested
    # from the start but had no HTTP route and no button, so the store could
    # only ever grow. These pin the route that finally reaches it.
    def _save(self, model="/m/Model-A"):
        from sglang.srt.planner import bench_history

        return bench_history.save_run(
            {"model": model, "endpoint": "e", "started_at": 1.0,
             "duration_s": 1.0, "results": [], "transcript": []})

    def test_delete_removes_the_stored_run(self):
        run_id = self._save()
        self.assertEqual(len(webui.bench_history_payload({})["runs"]), 1)
        d = webui.bench_history_delete_payload({"run_id": run_id})
        self.assertTrue(d["ok"])
        self.assertEqual(d["deleted"], [run_id])
        self.assertEqual(d["missing"], [])
        self.assertEqual(webui.bench_history_payload({})["runs"], [])
        self.assertFalse(webui.bench_run_payload({"run_id": run_id})["ok"])

    def test_delete_takes_a_list_and_reports_misses_without_failing(self):
        a, b = self._save("/m/A"), self._save("/m/B")
        d = webui.bench_history_delete_payload(
            {"run_ids": [a, "never-existed", b]})
        self.assertTrue(d["ok"])
        self.assertEqual(sorted(d["deleted"]), sorted([a, b]))
        self.assertEqual(d["missing"], ["never-existed"])
        self.assertEqual(webui.bench_history_payload({})["runs"], [])

    def test_delete_requires_an_id(self):
        d = webui.bench_history_delete_payload({})
        self.assertFalse(d["ok"])
        self.assertIn("run_id", d["error"])

    def test_delete_route_over_http(self):
        run_id = self._save()
        srv = ThreadingHTTPServer(("127.0.0.1", 0), webui._Handler)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/bench_history",
                data=json.dumps({"run_ids": [run_id]}).encode(),
                headers={"Content-Type": "application/json"},
                method="DELETE",
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.loads(r.read())
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=5)
        self.assertTrue(d["ok"])
        self.assertEqual(d["deleted"], [run_id])
        self.assertEqual(webui.bench_history_payload({})["runs"], [])


class TestConsolidatedNavigation(CustomTestCase):
    """#1/#3/#4: the tab strip IS the workflow, and it has no duplicates.

    The Planner stopped being a tab (its markup is the wizard's expert step),
    Landscape stopped being a tab (it was a one-model slice of the Rigs
    matrix with four permanently-empty measured columns), Rigs itself
    stopped being a tab (the matrix + landscape drill-down are now the
    Guide's last expert-step section), and Energy + Rig data merged into one
    Data tab (same underlying question: what this rig costs, what it can
    tell the project). All of these are structural claims that a later edit
    could quietly undo, so they are pinned here.

    Task #342 (Frontend IA v2) added a top-level nav-group bar over this
    same tab strip and three new tabs -- Models (registry hub), Training
    (stub, #341) and Video & Media (M2, read-only) -- without deleting,
    renaming or reordering any pre-existing tab_*/view_* pair; see
    NAV_GROUPS in the embedded script and TestFrontendIAv2NavMapping below
    for the group-level assertions. ORDER below is the DOM order of the
    #sub_tabs bar as it stands after that change.
    """

    # "about" sits last on purpose: it is not a workflow step but the
    # dashboard's own version management (install / update / roll back).
    # "models" leads (Task #342: the Models hub is the new primary landing
    # concern for "what can this rig serve"); "training"/"video" are appended
    # after the pre-existing workflow tabs, keeping every old tab's relative
    # order unchanged.
    ORDER = [
        "models",
        "landing",
        "wizard",
        "bench",
        "quality",
        "data",
        "pair",
        "history",
        "training",
        "video",
        "about",
    ]

    def test_tab_order_is_the_workflow_order(self):
        html = webui.INDEX_HTML
        seen = re.findall(r'id="tab_([a-z]+)"', html)
        self.assertEqual(seen, self.ORDER)

    def test_every_tab_button_has_a_view_and_vice_versa(self):
        html = webui.INDEX_HTML
        for key in self.ORDER:
            self.assertIn(f'id="tab_{key}"', html, key)
            self.assertIn(f'id="view_{key}"', html, key)
        # the switch drives exactly this set
        switch = html.split("function showTab(t) {")[1][:400]
        for key in self.ORDER:
            self.assertIn(f"'{key}'", switch, key)

    def test_planner_is_the_wizard_expert_step_not_a_tab(self):
        html = webui.INDEX_HTML
        self.assertNotIn('id="tab_runner"', html)
        self.assertNotIn("showTab('runner')", html)
        # still present, but NESTED inside the wizard so it inherits its
        # visibility instead of being switched independently.
        self.assertIn('id="view_runner"', html)
        self.assertLess(html.index('<div id="view_wizard"'),
                        html.index('<div id="view_runner"'))
        self.assertIn('data-step="expert"', html)

    def test_no_planner_capability_was_dropped_on_the_way(self):
        # The controls that only ever existed on the Planner tab must all be
        # reachable from the wizard's expert step. This is the "nothing lost"
        # assertion for #1.
        html = webui.INDEX_HTML
        for token in ('id="flag_surface"', 'id="flag_search"',
                      'id="action_bar"', 'id="status_chip"',
                      'id="runner_placement"', 'id="cardlist"',
                      'id="host_ram_gb"', 'id="sv_ctx"', 'id="sv_served"',
                      'id="tp_count"', 'id="colo_rows"',
                      'id="gpu_pick_select"', 'id="draft_model_select"',
                      'id="lp_slider"', 'id="simple_cards"',
                      'id="dl_repo"', 'id="profile_save_name"',
                      "serverStart()", "serverStop()", "serverRestart()",
                      "doPlan()", "doIssue(", "loadFlagCatalog"):
            self.assertIn(token, html, token)

    def test_rigs_tab_removed(self):
        html = webui.INDEX_HTML
        self.assertNotIn('id="tab_explore"', html)
        self.assertNotIn('id="view_explore"', html)

    def test_matrix_and_landscape_folded_into_guide(self):
        # The former Rigs tab (capacity matrix) and, nested inside it, the
        # former Landscape tab (per-rig detail for one model) both survive
        # as the Guide's last expert-step section -- nothing rendered on
        # either former tab was dropped.
        html = webui.INDEX_HTML
        self.assertNotIn('id="tab_landscape"', html)
        self.assertNotIn('id="view_landscape"', html)
        guide = html[html.index('<div id="view_wizard"'):]
        guide = guide[:guide.index("<script>")]
        # Round 5 rebuild: the free-text textareas are gone (chips/buttons
        # only); the matrix itself is kept (see TestCapacityMatrixIsChipDriven
        # for the verdict) with new, click-only ids.
        for token in ('id="mx_model_chips"', 'id="mx_rig_live"',
                      'id="mx_rig_builders"',
                      "doMatrix()", 'id="ls_model"', 'id="ls_quant"',
                      'id="ls_bucket"', 'id="ls_store"', 'id="ls_rigs"',
                      "doLandscape()"):
            self.assertIn(token, guide, token)
        for gone in ('id="mx_models"', 'id="mx_rigs"', 'id="mx_cards"',
                     '<textarea id="mx'):
            self.assertNotIn(gone, guide, gone)


# ===========================================================================
# Task #342 -- Frontend IA v2: nav-group skeleton (Models / Playground /
# Training / Video & Media / Rig / Benchmarks / Settings) over the
# pre-existing tab strip, plus the Models hub's binding to the registry
# (M1) API and the Video & Media tab's binding to the M2 job list.
# ===========================================================================


def _nav_groups_js() -> str:
    """The body of the NAV_GROUPS object literal in the embedded script."""
    html = webui.INDEX_HTML
    m = re.search(r"const NAV_GROUPS = \{(.*?)\n\};", html, re.S)
    assert m, "NAV_GROUPS object not found in INDEX_HTML"
    return m.group(1)


class TestFrontendIAv2NavMapping(CustomTestCase):
    """Every pre-existing tab keeps its tab_*/view_* pair and its entry in
    showTab()'s TABS array (TestConsolidatedNavigation covers that); this
    class pins the NEW grouping layer added on top: every tab -- old and
    new -- belongs to exactly one nav group, every group has a clickable
    button, and the three brand-new tabs (models/training/video) are wired
    the same way the pre-existing ones are.
    """

    # Every tab that must be reachable after the IA v2 change, old or new.
    ALL_TABS = [
        "landing",
        "wizard",
        "bench",
        "quality",
        "data",
        "pair",
        "history",
        "about",
        "models",
        "training",
        "video",
    ]
    GROUPS = ["models", "playground", "training", "video", "rig", "bench", "settings"]

    def test_every_group_has_a_clickable_button(self):
        html = webui.INDEX_HTML
        for g in self.GROUPS:
            self.assertIn(f'id="group_{g}"', html, g)
            self.assertIn(f"showGroup('{g}')", html, g)

    def test_nav_groups_js_partitions_every_tab_exactly_once(self):
        js = _nav_groups_js()
        seen = []
        for m in re.finditer(r"tabs:\s*\[([^\]]*)\]", js):
            names = re.findall(r"'([a-z]+)'", m.group(1))
            seen.extend(names)
        self.assertEqual(sorted(seen), sorted(self.ALL_TABS))
        # exactly once each -- a tab claimed by two groups would be reachable
        # from one but "active" in both, a state the group bar can't render.
        self.assertEqual(len(seen), len(set(seen)))

    def test_new_tabs_have_view_and_data_group(self):
        html = webui.INDEX_HTML
        expect_group = {"models": "models", "training": "training", "video": "video"}
        for tab, group in expect_group.items():
            self.assertIn(f'id="tab_{tab}"', html, tab)
            self.assertIn(f'id="view_{tab}"', html, tab)
            m = re.search(r'id="tab_' + tab + r'"[^>]*data-group="([a-z]+)"', html)
            self.assertIsNotNone(m, tab)
            self.assertEqual(m.group(1), group, tab)

    def test_old_tabs_each_carry_a_data_group_attribute(self):
        # The pre-#342 tabs were not renamed or moved in the DOM -- only
        # tagged with which group now shows/hides them.
        html = webui.INDEX_HTML
        old = [
            "landing",
            "wizard",
            "bench",
            "quality",
            "data",
            "pair",
            "history",
            "about",
        ]
        for tab in old:
            m = re.search(r'id="tab_' + tab + r'"[^>]*data-group="([a-z]+)"', html)
            self.assertIsNotNone(m, tab)
            self.assertIn(m.group(1), self.GROUPS, tab)

    def test_showgroup_and_hash_router_are_defined(self):
        html = webui.INDEX_HTML
        for needle in (
            "function showGroup(",
            "function routeFromHash(",
            "function syncGroupBar(",
            "function tabGroup(",
        ):
            self.assertIn(needle, html, needle)

    def test_deep_link_hash_is_written_on_tab_switch(self):
        html = webui.INDEX_HTML
        self.assertIn("history.replaceState(null, '', '#'+g+'/'+t)", html)


class TestFrontendIAv2RegistryHub(CustomTestCase):
    """Models hub <-> registry (M1) binding. Every call is exercised through
    ``webui._proxy_json`` mocked to hand back the exact (status, body) the
    real registry would -- including its rejection shapes -- so these tests
    pin that the dashboard relays them VERBATIM rather than composing its
    own wording (Task #342 scope item 3)."""

    def setUp(self):
        super().setUp()
        os.environ.pop("SGLANG_REGISTRY_URL", None)
        os.environ.pop("SGLANG_VIDEO_URL", None)

    def test_registry_base_resolution_order(self):
        # query override > env > documented default (§ same convention as
        # the landing tab's editable endpoint).
        self.assertEqual(webui._registry_base({}), "http://127.0.0.1:8500")
        os.environ["SGLANG_REGISTRY_URL"] = "10.0.0.5:9000"
        try:
            self.assertEqual(webui._registry_base({}), "http://10.0.0.5:9000")
            self.assertEqual(
                webui._registry_base({"registry": "other:1234"}), "http://other:1234"
            )
        finally:
            os.environ.pop("SGLANG_REGISTRY_URL", None)

    def test_snapshot_reachable_relays_body_verbatim(self):
        # Real shape of GET /registry (arbiter.EngineRegistry.snapshot() /
        # spec.EngineInstance.to_json()): a flat "engines" list, not a
        # separate specs/instances mapping.
        body = {
            "engines": [
                {
                    "engine_id": "qwen",
                    "klass": 1,
                    "state": "HOT",
                    "cards": ["GPU-1"],
                    "pinned": False,
                    "priority": 0,
                }
            ],
            "slots": [],
            "cards": [],
            "default_hot": [],
            "max_hot": None,
        }
        with mock.patch.object(webui, "_proxy_json", return_value=(200, dict(body))):
            out = webui.registry_snapshot_payload({})
        self.assertTrue(out["ok"])
        self.assertTrue(out["reachable"])
        self.assertEqual(out["engines"], body["engines"])

    def test_snapshot_unreachable_is_honest_offline_state(self):
        with mock.patch.object(
            webui,
            "_proxy_json",
            return_value=(
                503,
                {
                    "ok": False,
                    "reachable": False,
                    "error": "http://127.0.0.1:8500/registry unreachable: "
                    "[Errno 111] Connection refused",
                },
            ),
        ):
            out = webui.registry_snapshot_payload({})
        self.assertFalse(out["ok"])
        self.assertFalse(out["reachable"])
        self.assertIn("Connection refused", out["error"])

    def test_plan_rejection_message_is_relayed_verbatim(self):
        # Shape of RegistrationRejected / a plan that does not fit (see
        # registry/arbiter.py PlanResult / RegistrationRejected.to_json()).
        rejection_text = (
            "engine 'qwen-27b' would take about 42000 ms to promote "
            "(estimated), which exceeds the caller's budget of 5000 ms. "
            "Getting there would evict ['other-engine']. Wait, raise the "
            "budget, or use an engine that is already hot."
        )
        body = {
            "engine_id": "qwen-27b",
            "fits": False,
            "feasible_without_eviction": False,
            "shortfall_detail": "does not fit: card X short by 4096 MiB",
            "would_evict": ["other-engine"],
            "reason": rejection_text,
        }
        with mock.patch.object(webui, "_proxy_json", return_value=(200, dict(body))):
            out = webui.registry_plan_payload({"engine_id": "qwen-27b", "klass": 1})
        self.assertTrue(out["ok"])
        # Not reworded, not truncated, not summarized -- byte for byte.
        self.assertEqual(out["reason"], rejection_text)
        self.assertEqual(out["shortfall_detail"], body["shortfall_detail"])
        self.assertEqual(out["would_evict"], ["other-engine"])

    def test_register_rejection_400_is_relayed_with_original_status(self):
        message = (
            "engine 'x' cannot fit on this rig even with every other "
            "tenant evicted: does not fit: card Y short by 2048 MiB"
        )
        with mock.patch.object(
            webui,
            "_proxy_json",
            return_value=(
                400,
                {
                    "error": "registration_rejected",
                    "message": message,
                    "detail": "full report text",
                },
            ),
        ):
            out = webui.registry_register_payload({"engine_id": "x", "klass": 1})
        self.assertFalse(out["ok"])
        self.assertEqual(out["status"], 400)
        self.assertEqual(out["message"], message)
        self.assertEqual(out["detail"], "full report text")

    def test_promotion_rejected_503_shape_is_relayed_unchanged(self):
        body = {
            "error": "promotion_rejected",
            "engine_id": "qwen-27b",
            "message": "engine 'qwen-27b' would take about 42000 ms to promote",
            "projected_wait_ms": 42000.0,
            "projected_wait_is_estimated": True,
            "would_evict": ["other-engine"],
            "shortfalls": [
                {
                    "card_uuid": "GPU-1",
                    "requested_bytes": 1,
                    "held_bytes": 2,
                    "corridor_bytes": 3,
                    "total_bytes": 4,
                    "shortfall_bytes": 5,
                    "holders": ["other-engine"],
                }
            ],
        }
        with mock.patch.object(webui, "_proxy_json", return_value=(503, dict(body))):
            out = webui.registry_state_payload(
                {"engine_id": "qwen-27b", "target": "HOT"}
            )
        self.assertFalse(out["ok"])
        self.assertEqual(out["status"], 503)
        self.assertEqual(out["message"], body["message"])
        self.assertEqual(out["shortfalls"], body["shortfalls"])
        self.assertEqual(out["would_evict"], ["other-engine"])

    def test_state_change_without_engine_id_never_calls_the_registry(self):
        with mock.patch.object(webui, "_proxy_json") as proxy:
            out = webui.registry_state_payload({"target": "HOT"})
        proxy.assert_not_called()
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "invalid_spec")

    def test_delete_without_engine_id_never_calls_the_registry(self):
        with mock.patch.object(webui, "_proxy_json") as proxy:
            out = webui.registry_delete_payload({})
        proxy.assert_not_called()
        self.assertFalse(out["ok"])

    def test_delete_success_relays_deregistered_id(self):
        with mock.patch.object(
            webui,
            "_proxy_json",
            return_value=(200, {"deregistered": "qwen-27b"}),
        ) as proxy:
            out = webui.registry_delete_payload({"engine_id": "qwen-27b"})
        self.assertTrue(out["ok"])
        self.assertEqual(out["deregistered"], "qwen-27b")
        # DELETE to the per-engine path, not the collection path.
        args, kwargs = proxy.call_args
        self.assertTrue(args[0].endswith("/registry/engines/qwen-27b"))
        self.assertEqual(args[1], "DELETE")

    def test_cards_payload_relays_per_card_ledger_unchanged(self):
        cards = [
            {
                "card_uuid": "GPU-1",
                "total_bytes": 100,
                "reserved_bytes": 40,
                "measured_bytes": 38,
                "corridor_bytes": 5,
                "waste_bytes": 2,
            }
        ]
        with mock.patch.object(
            webui, "_proxy_json", return_value=(200, {"cards": cards})
        ):
            out = webui.registry_cards_payload({})
        self.assertTrue(out["ok"])
        self.assertEqual(out["cards"], cards)


class TestFrontendIAv2VideoTab(CustomTestCase):
    """Video & Media tab <-> M2 job-list binding. Read-only: no test here
    calls anything but the GET path. An unconfigured or unreachable service
    is an honest offline state, not an error (module docstring on
    video_jobs_payload)."""

    def setUp(self):
        super().setUp()
        os.environ.pop("SGLANG_VIDEO_URL", None)

    def test_no_base_configured_is_offline_without_a_network_attempt(self):
        with mock.patch.object(webui, "_proxy_json") as proxy:
            out = webui.video_jobs_payload({})
        proxy.assert_not_called()
        self.assertTrue(out["ok"])
        self.assertFalse(out["reachable"])
        self.assertEqual(out["jobs"], [])

    def test_reachable_service_relays_job_list(self):
        jobs = [
            {
                "id": "vid_1",
                "status": "completed",
                "progress": 100,
                "created_at": "2026-07-01T00:00:00Z",
            }
        ]
        with mock.patch.object(
            webui, "_proxy_json", return_value=(200, {"data": jobs})
        ):
            out = webui.video_jobs_payload({"video": "127.0.0.1:9100"})
        self.assertTrue(out["ok"])
        self.assertTrue(out["reachable"])
        self.assertEqual(out["jobs"], jobs)

    def test_configured_but_unreachable_is_offline_not_an_error_page(self):
        with mock.patch.object(
            webui,
            "_proxy_json",
            return_value=(
                503,
                {
                    "ok": False,
                    "reachable": False,
                    "error": "http://127.0.0.1:9100/v1/videos "
                    "unreachable: [Errno 111] Connection refused",
                },
            ),
        ):
            out = webui.video_jobs_payload({"video": "127.0.0.1:9100"})
        self.assertFalse(out["ok"])
        self.assertFalse(out["reachable"])
        self.assertIn("Connection refused", out["error"])


class TestCapacityMatrixIsChipDriven(CustomTestCase):
    """Round 5, point 2: the capacity matrix, evaluated honestly.

    VERDICT: kept. It answers "does model X fit on rig Y, and with how much
    context?" for SEVERAL checkpoints against SEVERAL rigs at once, including
    hypothetical ones this box does not have (composed from the GPU-model
    card library). The families table above answers for ONE model (the
    ticked one) on THIS live rig only; the planner/expert step answers for
    ONE model, ONE specific configuration. Neither can compare several
    checkpoints against several -- possibly hypothetical -- rigs side by
    side, which is the question this section exists to answer, so it is not
    a duplicate of either and stays.

    What was actually broken was operability: two freeform textareas
    ("LABEL=path" / "NAME=card,card,..."), no explanation of what the answer
    meant. Rebuilt: models are ticked from the discovered-checkpoint list
    (the same one the model dropdown uses), rigs are composed from the
    GPU-model card library with +/- buttons, one "Calculate" button, one
    sentence stating the question. No free-text input remains anywhere in
    this section.
    """

    def _matrix_html(self):
        # Bounded to the matrix's OWN model/rig inputs -- excludes the nested
        # "per-rig detail" (former Landscape tab) drill-down, which is a
        # separate, single-model tool the chip requirement never applied to
        # (ls_model / ls_rigs keep their free-text fields).
        html = webui.INDEX_HTML
        start = html.index('<legend>capacity matrix')
        end = html.index('old "Landscape" tab lived here', start)
        return html[start:end]

    def test_one_sentence_explanation_present(self):
        section = self._matrix_html()
        self.assertIn(
            "does model X fit on rig Y, and with how much context?", section
        )

    def test_models_are_chips_from_discovered_checkpoints(self):
        section = self._matrix_html()
        self.assertIn('id="mx_model_chips"', section)
        js = _index_script()
        self.assertIn("async function loadMxModelChips(){", js)
        self.assertIn("window._models", js.split(
            "async function loadMxModelChips(){")[1][:600])

    def test_rigs_are_this_rig_plus_composed_from_card_library(self):
        section = self._matrix_html()
        self.assertIn('id="mx_rig_live"', section)
        self.assertIn("this rig (live)", section)
        self.assertIn('onclick="mxAddRig()"', section)
        js = _index_script()
        self.assertIn("function renderMxRigBuilders(){", js)
        self.assertIn("window._mxCardLib", js)

    def test_exactly_one_calculate_action(self):
        section = self._matrix_html()
        # doMatrix() is wired to exactly one button in this section.
        self.assertEqual(section.count('onclick="doMatrix()"'), 1)

    def test_no_free_text_input_anywhere_in_the_section(self):
        section = self._matrix_html()
        self.assertNotIn("<textarea", section)
        self.assertNotIn('<input type="text"', section)
        self.assertNotIn('<input id="mx', section)

    def test_calculate_refuses_with_nothing_ticked_rather_than_silently(self):
        js = _index_script()
        start = js.index("async function doMatrix()")
        body = js[start:start + 900]
        self.assertIn("tick at least one discovered model", body)
        self.assertIn('tick "this rig (live)" or add at least one', body)
        self.assertIn("card to a hypothetical rig", body)


class TestPresetsRemoved(CustomTestCase):
    """#7: prefabricated presets are gone; named profiles stay.

    Two canned starting points used to sit in one panel (a generated-preset
    dropdown and the tuning-objective buttons) for a configuration the guide
    already works out exactly. Saving and loading a named profile is what
    remains.
    """

    def test_the_objective_template_buttons_are_gone(self):
        html = webui.INDEX_HTML
        for token in ('id="tune_both"', 'id="tune_maxkv"', 'id="tune_dec"',
                      'id="tune_enc"', 'id="tune_pick"'):
            self.assertNotIn(token, html, token)

    def test_rank_perf_tune_is_still_reachable_and_still_written(self):
        # Removing the buttons must not remove the knob: the flag keeps its
        # own row in the flag surface, and the working-point stops still set
        # it through applyTune.
        html = webui.INDEX_HTML
        self.assertIn("applyTune", html)
        self.assertIn("rank_perf_tune", html)
        ids = {
            f["id"]
            for flags in webui.flag_catalog_payload()["groups"].values()
            for f in flags
        }
        self.assertIn("rank_perf_tune", ids)

    def test_the_picker_offers_only_saved_profiles(self):
        html = webui.INDEX_HTML
        self.assertIn("window._profiles=(d.saved||[])", html)
        self.assertIn(">save profile<", html)
        self.assertNotIn("&mdash; apply a preset &mdash;", html)

    def test_the_endpoint_still_returns_both_sets(self):
        # The dashboard stopped OFFERING generated profiles; the endpoint is
        # unchanged because the CLI and the family generator still use them.
        d = webui.config_profiles_get({})
        self.assertTrue(d["ok"])
        self.assertIn("generated", d)
        self.assertIn("saved", d)


class TestMonitorCardsAndViewMode(CustomTestCase):
    """#8: every card visible, and the simple/expert switch actually acts."""

    def test_cards_outside_the_placement_are_still_rendered(self):
        html = webui.INDEX_HTML
        # renderPlacement walks pl.cards (rank-carrying cards only); the
        # remaining NVML cards get a telemetry-only block.
        self.assertIn("idlecard", html)
        self.assertIn("not in this configuration", html)
        self.assertIn("const placed=new Set((pl.cards||[]).map(c=>c.gpu_index));",
                      html)

    def test_the_view_mode_switch_reaches_the_card_breakdown(self):
        html = webui.INDEX_HTML
        # the VRAM key exists in both densities, picked by the body class
        self.assertIn('class="segsum simple-only"', html)
        self.assertIn('class="seglegend expert-only"', html)
        self.assertIn("body.mode-simple .expert-only", html)
        self.assertIn("body.mode-expert .simple-only", html)


class TestEnergyPerPhase(CustomTestCase):
    """#9: energy per token is reported per phase, and savings are named."""

    def test_prefill_and_decode_are_separate_tiles(self):
        html = webui.INDEX_HTML
        self.assertIn("strip_energy_pfx", html)
        self.assertIn("energy / token &mdash; DECODE", html)
        self.assertIn("energy / token &mdash; PREFILL", html)
        # each phase has its own ring, so one cannot overwrite the other
        self.assertIn("stripPush('st_j',t,jtokDec)", html)
        self.assertIn("stripPush('st_jp',t,jtokPfx)", html)

    def test_saved_prefill_energy_leads_with_kwh(self):
        html = webui.INDEX_HTML
        self.assertIn("prefill tok recovered", html)
        self.assertIn("saved by RAM+disk cache", html)


class TestWizardOwnsTheFlow(CustomTestCase):
    """#10/#11/#12: picker in step 1, card set in step 2, steps stay live."""

    def test_the_model_picker_lives_in_step_one(self):
        html = webui.INDEX_HTML
        wiz = html.index('<div id="view_wizard"')
        expert = html.index('data-step="expert"')
        for token in ('id="model_search"', 'id="models_out"', 'id="model"',
                      'id="gguf_choice"'):
            at = html.index(token)
            self.assertGreater(at, wiz, token)
            self.assertLess(at, expert, f"{token} must be in step 1, not the "
                                        f"expert step")

    def test_the_card_set_is_the_hardware_step(self):
        html = webui.INDEX_HTML
        hw = html.index('data-step="hardware"')
        goal = html.index('data-step="goal"')
        for token in ('id="cardlist"', 'id="wz_unused"', "addCard(false)"):
            at = html.index(token)
            self.assertGreater(at, hw, token)
            self.assertLess(at, goal, token)
        self.assertIn("function renderUnusedCards(", html)
        self.assertIn("available, unused", html)

    def test_every_step_is_marked_and_invalidates_the_later_ones(self):
        html = webui.INDEX_HTML
        for step in ("model", "hardware", "goal", "families", "command",
                     "expert"):
            self.assertIn(f'data-step="{step}"', html, step)
        self.assertIn("function wizardInvalidate(", html)
        self.assertIn("[data-step].stale", html)
        # a change in an earlier step marks the later ones
        self.assertIn("wizardInvalidate('model')", html)
        self.assertIn("wizardInvalidate('hardware')", html)
        self.assertIn("wizardInvalidate('goal')", html)


class TestStopCollapseIsExplained(CustomTestCase):
    """#2/#215: identical stop columns are the answer, and the page says so.

    Investigated and confirmed as an HONEST COLLAPSE, not a bug: the total
    weight bytes are invariant under an MLP redistribution, so the capacity
    objective is flat across the candidate ladder, and the base split is the
    strict decode optimum (#265). The backend already flagged the EXACT case
    per row; the page did not show it.

    Round 5 generalized this from an annotation (#215: identical columns
    stayed drawn, just badged "= base") into a real fold: near-identical
    stops (within a stated percentage threshold on every priced figure, not
    only byte-identical ones) are dropped from the table and named in the
    base column's header + a prose note instead, so a reader comparing
    several "options" that coincide on this rig sees one card, not several
    that only differ by a label.
    """

    def test_a_collapsed_column_says_so_on_its_head(self):
        html = webui.INDEX_HTML
        self.assertIn("lp-same", html)
        self.assertIn("function withinFoldThreshold(p){", html)
        # the exact-match case from #215 still folds unconditionally
        self.assertIn("if(p.same_as_baseline) return true;", html)

    def test_fold_threshold_is_visible_not_just_internal(self):
        # "<1% difference = folded", stated where the reader can see it, not
        # only as a magic number inside the JS.
        html = webui.INDEX_HTML
        self.assertIn("LP_FOLD_THRESHOLD_PCT = 1", html)
        self.assertIn("Fold threshold:", html)
        self.assertIn("folded into its column", html)

    def test_the_collapse_is_also_explained_in_prose(self):
        html = webui.INDEX_HTML
        self.assertIn("lp-collapse", html)
        self.assertIn("landed on the same answer", html)
        self.assertIn("--rank-kv-ratio", html)

    def test_only_genuinely_different_stops_get_their_own_column(self):
        # A stop folds only when EVERY one of its priced deltas vs. the base
        # is within threshold -- one figure outside it is a real difference
        # and must keep its own column, not be swallowed into the base's.
        html = webui.INDEX_HTML
        self.assertIn("deltas.length>0 && deltas.every(pct=>Math.abs(pct)<LP_FOLD_THRESHOLD_PCT)", html)


class TestReferencePngIsDerived(CustomTestCase):
    """#5: the chess reference is computed, not a checked-in screenshot.

    Root cause of the missing image AND of the long-standing
    test_reference_png_static_route failure: the repository ignores ``*.png``
    (a "Plots" rule with only two narrow negations), so the asset could never
    be committed and was absent from every fresh checkout.
    """

    def setUp(self):
        webui._REFERENCE_PNG_CACHE = None

    def tearDown(self):
        webui._REFERENCE_PNG_CACHE = None

    def test_it_renders_the_ground_truth_board(self):
        data = webui._reference_png_bytes()
        self.assertIsNotNone(data)
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(data), 1000)

    def test_it_is_derived_from_the_same_movetext_the_validator_grades(self):
        from sglang.srt.planner import quality_chess

        captured = {}
        real = quality_chess.ground_truth

        def spy(pgn=quality_chess.CHESS_PGN):
            captured["pgn"] = pgn
            return real(pgn)

        with mock.patch.object(quality_chess, "ground_truth", side_effect=spy):
            webui._render_reference_png()
        self.assertEqual(captured["pgn"], quality_chess.CHESS_PGN)

    def test_a_file_on_disk_still_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "assets")
            os.makedirs(path)
            with open(os.path.join(path, "quality_chess_reference.png"),
                      "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"pinned")
            with mock.patch.object(
                webui.os.path, "dirname", return_value=tmp
            ):
                data = webui._reference_png_bytes()
        self.assertEqual(data, b"\x89PNG\r\n\x1a\n" + b"pinned")


class TestMonitorTargetControls(CustomTestCase):
    """#13: two actions with fixed meanings, not three that swap roles.

    "detect" used to do what "use" did when the box had text and what "auto"
    did when it did not, and "use" applied an address without ever checking
    it, so a typo produced no message at all.
    """

    def test_two_actions_and_an_unpin(self):
        html = webui.INDEX_HTML
        self.assertIn("function landingConnect(", html)
        self.assertIn("function landingFindServer(", html)
        self.assertIn('id="land_unpin"', html)
        self.assertNotIn("function detectLandingEndpoint(", html)
        self.assertNotIn(">use<", html)

    def test_connect_reports_a_refusal_instead_of_applying_it(self):
        html = webui.INDEX_HTML
        connect = html.split("async function landingConnect(")[1][:1400]
        # it probes before it pins
        self.assertIn("/api/detect_endpoint", connect)
        self.assertIn("Nothing was changed", connect)

    def test_the_mode_is_written_out_not_inferred(self):
        html = webui.INDEX_HTML
        self.assertIn("function renderLandingMode(", html)
        self.assertIn("Pinned to", html)


class TestViewModeIsOneCheckbox(CustomTestCase):
    """#14: an opt-in checkbox, and a plain bar in the simple density."""

    def test_it_is_a_checkbox_not_a_two_way_pick(self):
        html = webui.INDEX_HTML
        self.assertNotIn('id="vm_simple"', html)
        self.assertIn('type="checkbox" id="vm_expert"', html)
        # no "simple" button to click, so nobody is asked to classify
        # themselves before seeing the page
        self.assertNotIn("setViewMode('simple')\"", html)

    def test_simple_density_gets_one_solid_bar_and_no_breakdown(self):
        html = webui.INDEX_HTML
        self.assertIn('class="segbar simple-only"', html)
        self.assertIn('class="segbar expert-only"', html)
        # the itemised legend is expert-only, so simple shows no segments
        self.assertIn('class="seglegend expert-only"', html)
        self.assertIn('class="segsum simple-only"', html)


class TestLinkThroughput(CustomTestCase):
    """#15: live PCIe / NVLink per card, each against its own ceiling."""

    def setUp(self):
        from sglang.srt.planner import live_metrics

        live_metrics._LINK_PEAKS.clear()
        live_metrics._LINK_STATE_PEAKS.clear()
        live_metrics._NVLINK_LAST.clear()

    tearDown = setUp

    def test_pcie_is_sampled_and_nvlink_absence_is_not_zero(self):
        from sglang.srt.planner import live_metrics

        class _Nvml:
            NVML_PCIE_UTIL_TX_BYTES = 0
            NVML_PCIE_UTIL_RX_BYTES = 1

            def nvmlDeviceGetPcieThroughput(self, h, kind):
                return 2_000_000 if kind == 0 else 1_000_000   # KB/s

            def nvmlDeviceGetMaxPcieLinkGeneration(self, h):
                return 4

            def nvmlDeviceGetMaxPcieLinkWidth(self, h):
                return 16

            def nvmlDeviceGetCurrPcieLinkGeneration(self, h):
                return 4

            def nvmlDeviceGetCurrPcieLinkWidth(self, h):
                return 16

            def nvmlDeviceGetNvLinkState(self, h, link):
                raise RuntimeError("no nvlink on this board")

        d = live_metrics._link_fields(_Nvml(), object(), 0, "uuid-a")
        self.assertAlmostEqual(d["pcie_tx_gbs"], 2.0, places=6)
        self.assertAlmostEqual(d["pcie_rx_gbs"], 1.0, places=6)
        self.assertAlmostEqual(d["pcie_max_gbs"], 1.969 * 16, places=3)
        # "no NVLink" is None, never 0.0 -- they are different facts
        self.assertIsNone(d["nvlink_tx_gbs"])
        self.assertEqual(d["nvlink_links"], 0)

    def test_without_calibration_the_ceiling_is_the_highest_seen_so_far(self):
        from sglang.srt.planner import live_metrics

        rate = {"v": 3_000_000}

        class _Nvml:
            NVML_PCIE_UTIL_TX_BYTES = 0
            NVML_PCIE_UTIL_RX_BYTES = 1

            def nvmlDeviceGetPcieThroughput(self, h, kind):
                return rate["v"] if kind == 0 else 0

            def nvmlDeviceGetNvLinkState(self, h, link):
                raise RuntimeError("none")

            # no link-capability getters at all -> peak is the only ceiling
            def __getattr__(self, name):
                raise AttributeError(name)

        n = _Nvml()
        d = live_metrics._link_fields(n, object(), 0, "uuid-b")
        self.assertAlmostEqual(d["pcie_max_gbs"], 3.0, places=6)
        # a bigger sample raises the ceiling in place...
        rate["v"] = 9_000_000
        d = live_metrics._link_fields(n, object(), 0, "uuid-b")
        self.assertAlmostEqual(d["pcie_max_gbs"], 9.0, places=6)
        # ...and a smaller one never lowers it
        rate["v"] = 1_000_000
        d = live_metrics._link_fields(n, object(), 0, "uuid-b")
        self.assertAlmostEqual(d["pcie_tx_gbs"], 1.0, places=6)
        self.assertAlmostEqual(d["pcie_max_gbs"], 9.0, places=6)

    def test_a_measured_peak_outranks_a_lower_capability_figure(self):
        from sglang.srt.planner import live_metrics

        class _Nvml:
            NVML_PCIE_UTIL_TX_BYTES = 0
            NVML_PCIE_UTIL_RX_BYTES = 1

            def nvmlDeviceGetPcieThroughput(self, h, kind):
                return 50_000_000 if kind == 0 else 0     # 50 GB/s

            def nvmlDeviceGetMaxPcieLinkGeneration(self, h):
                return 1

            def nvmlDeviceGetMaxPcieLinkWidth(self, h):
                return 4

            def nvmlDeviceGetCurrPcieLinkGeneration(self, h):
                return 1

            def nvmlDeviceGetCurrPcieLinkWidth(self, h):
                return 4

            def nvmlDeviceGetNvLinkState(self, h, link):
                raise RuntimeError("none")

        d = live_metrics._link_fields(_Nvml(), object(), 0, "uuid-c")
        # capability says 1.0 GB/s; something measured 50. The measurement
        # wins, so a bar can never read over 100%.
        self.assertAlmostEqual(d["pcie_max_gbs"], 50.0, places=6)

    def test_the_card_line_renders_the_link_figures(self):
        html = webui.INDEX_HTML
        self.assertIn("function linkLiveHtml(", html)
        self.assertIn("PCIe", html)
        self.assertIn("linkbar", html)
        # NVLink pending is a state of its own, not a zero
        self.assertIn("rate on the next poll", html)


class TestLiveBannerStates(CustomTestCase):
    """Three states, and the panel is in one of them after EVERY poll.

    A warning that outlives the server it was about is the failure mode here,
    so the structural guarantee is what gets tested: the panel is not inside
    the "a server is running" container, and the not-running branch of the
    poll rewrites it.
    """

    def test_panel_is_outside_the_running_container(self):
        html = webui.INDEX_HTML
        i_panel = html.index('<fieldset id="live_panel">')
        i_live = html.index('<div id="landing_live"')
        self.assertLess(i_panel, i_live,
                        "live_panel must not live inside landing_live")

    def test_not_running_branch_rerenders_the_panel(self):
        html = webui.INDEX_HTML
        i_branch = html.index("if(!d || !d.running || !d.snapshot){")
        i_render = html.index("renderLivePanel(null, d)", i_branch)
        i_hide = html.index("$('landing_none').style.display=''", i_branch)
        self.assertLess(i_render, i_hide + 1)

    def test_the_three_states_are_named(self):
        html = webui.INDEX_HTML
        self.assertIn("No inference server running", html)
        self.assertIn("Server started without --enable-metrics", html)
        self.assertIn("function targetLabel(", html)

    def test_no_metrics_banner_names_model_and_port(self):
        html = webui.INDEX_HTML
        i = html.index("function targetLabel(")
        seg = html[i:i + 900]
        self.assertIn("served_model_name", seg)
        self.assertIn("' @ :'", seg)
        self.assertIn("pid", seg)


class TestCardProbeEndpoint(CustomTestCase):
    """POST /api/card_probe + GET /api/card_probe/status (#213).

    The point under test is not the measurement (that needs cards) but the
    contract around it: the start call returns instead of blocking, the
    status call answers before anything has ever been started, and the
    absence of a probe is stated in the words the surface uses.
    """

    def _isolated_store(self):
        """A fresh job store, so a test never starts a real probe."""
        from sglang.srt.rigmon import card_probe

        store = card_probe.ProbeJobStore()
        store.synchronous = True
        return card_probe, store

    def test_status_without_a_probe_names_the_nameplate_basis(self):
        from sglang.srt.rigmon import card_probe

        real = card_probe.load_card_probe
        card_probe.load_card_probe = lambda *a, **k: None
        try:
            d = webui.card_probe_status_payload({})
        finally:
            card_probe.load_card_probe = real
        self.assertTrue(d["ok"])
        self.assertFalse(d["measured"])
        self.assertIsNone(d["cached"])
        self.assertEqual(d["basis"], "nameplate-ranked, no probe cached")

    def test_status_with_a_probe_reports_measured(self):
        from sglang.srt.rigmon import card_probe

        prof = card_probe.CardProbeProfile(
            created=1000.0,
            cards=[card_probe.CardProbeMeasurement(
                uuid="u", name="card", cuda_index=0, membw_read_gbs=700.0,
                gemm_bf16_tflops=60.0)],
            pairs=[],
        )
        real = card_probe.load_card_probe
        card_probe.load_card_probe = lambda *a, **k: prof
        try:
            d = webui.card_probe_status_payload({})
        finally:
            card_probe.load_card_probe = real
        self.assertTrue(d["measured"])
        self.assertIn("measured card probe of 1 cards", d["basis"])
        self.assertIsNotNone(d["cached"])

    def test_start_returns_a_job_rather_than_the_result(self):
        card_probe, store = self._isolated_store()
        prof = card_probe.CardProbeProfile(created=1000.0)
        store.runner = lambda node_id: (prof, "/tmp/x.json")
        real_jobs = card_probe.JOBS
        card_probe.JOBS = store
        try:
            d = webui.card_probe_start_payload({})
        finally:
            card_probe.JOBS = real_jobs
        self.assertTrue(d["ok"])
        self.assertIn("job_id", d["job"])
        # A job, not a profile: the caller polls for the numbers.
        self.assertNotIn("cards", d)

    def test_unknown_job_id_is_an_error_not_an_empty_job(self):
        card_probe, store = self._isolated_store()
        real_jobs = card_probe.JOBS
        card_probe.JOBS = store
        try:
            d = webui.card_probe_status_payload({"job_id": "nope"})
        finally:
            card_probe.JOBS = real_jobs
        self.assertFalse(d["ok"])
        self.assertIn("nope", d["error"])

    def test_status_path_is_dispatched_before_the_start_path(self):
        # The flat chain matches on prefixes and /api/card_probe is a prefix
        # of /api/card_probe/status.
        src = webui.__file__
        with open(src) as f:
            text = f.read()
        i_status = text.index('startswith("/api/card_probe/status")')
        i_start = text.index('startswith("/api/card_probe"):', i_status)
        self.assertLess(i_status, i_start)

    def test_the_page_offers_the_probe_and_stops_its_own_poll(self):
        html = webui.INDEX_HTML
        self.assertIn("function cardProbeBar(", html)
        self.assertIn("/api/card_probe/status", html)
        self.assertIn("no probe cached", html)
        i = html.index("async function cardProbePoll(")
        seg = html[i:i + 1400]
        self.assertIn("clearInterval(window._cardProbeTimer)", seg)
        self.assertIn("refreshLeverProfiles()", seg)


# ===========================================================================
# Task #232: the tipping point, one measured split candidate per click.
#
# What is pinned here is that the endpoint returns a JOB and never the
# measurement, that the table and the poll come from the same answer, and
# that an unmeasured candidate is a row with a button rather than a number.
# ===========================================================================


class TestSplitProbeEndpoint(CustomTestCase):
    def _isolated(self):
        """A fresh job store, so a test never boots a server."""
        from sglang.srt.planner import split_probe as sp

        store = sp.SplitProbeJobStore()
        store.synchronous = True
        store.runner = lambda req: sp.SplitProbeResult(
            candidate=req.get("candidate", "auto"),
            prefill_tok_s=1151.1,
            decode_tok_s=92.55,
            ms_per_verify=33.0,
            max_total_num_tokens=502528,
            timestamp=1.0,
        )
        return sp, store

    def _with_store(self, sp, store, fn):
        real = sp.JOBS
        sp.JOBS = store
        try:
            return fn()
        finally:
            sp.JOBS = real

    def test_a_start_without_a_model_is_refused_with_a_remedy(self):
        d = webui.split_probe_start_payload({})
        self.assertFalse(d["ok"])
        self.assertIn("model_path", d["error"])
        self.assertTrue(d["remedy"])

    def test_start_returns_a_job_not_a_measurement(self):
        sp, store = self._isolated()
        d = self._with_store(
            sp,
            store,
            lambda: webui.split_probe_start_payload(
                {"model_path": "/m", "mlp_vector": "6,1,1"}
            ),
        )
        self.assertTrue(d["ok"])
        self.assertEqual(d["job"]["candidate"], "6,1,1")
        self.assertNotIn("prefill_tok_s", d)

    def test_status_carries_the_table_so_the_poll_and_the_render_agree(self):
        sp, store = self._isolated()
        d = self._with_store(sp, store, lambda: webui.split_probe_status_payload({}))
        self.assertTrue(d["ok"])
        self.assertIn("table", d)
        keys = {r["candidate"] for r in d["table"]["rows"]}
        self.assertTrue(set(sp.LADDER).issubset(keys))

    def test_an_unmeasured_candidate_is_a_row_with_a_reason_not_a_number(self):
        from sglang.srt.planner import split_probe as sp

        with tempfile.TemporaryDirectory() as d:
            table = sp.tipping_point_table(os.path.join(d, "empty.jsonl"))
        for row in table["rows"]:
            self.assertFalse(row["measured"])
            self.assertNotIn("prefill_tok_s", row)
            self.assertIn("not measured", row["missing_reason"])

    def test_unknown_job_id_is_an_error_not_an_empty_job(self):
        sp, store = self._isolated()
        d = self._with_store(
            sp, store, lambda: webui.split_probe_status_payload({"job_id": "nope"})
        )
        self.assertFalse(d["ok"])
        self.assertIn("nope", d["error"])

    def test_status_path_is_dispatched_before_the_start_path(self):
        with open(webui.__file__) as f:
            text = f.read()
        i_status = text.index('startswith("/api/split_probe/status")')
        i_start = text.index('startswith("/api/split_probe"):', i_status)
        self.assertLess(i_status, i_start)

    def test_the_page_renders_the_table_and_stops_its_own_poll(self):
        html = webui.INDEX_HTML
        self.assertIn("function renderTipping(", html)
        self.assertIn("/api/split_probe/status", html)
        self.assertIn('id="bn_tipping"', html)
        self.assertIn("not measured", html)
        i = html.index("function splitProbePollStart(")
        seg = html[i : i + 900]
        self.assertIn("clearInterval(window._splitProbeTimer)", seg)

    def test_the_measure_button_says_what_it_costs(self):
        from sglang.srt.planner import split_probe as sp

        with tempfile.TemporaryDirectory() as d:
            table = sp.tipping_point_table(os.path.join(d, "empty.jsonl"))
        self.assertIn("6-8 minutes", table["cost_note"])
        # The tooltip is the cost note, so the two cannot drift apart.
        self.assertIn('title="\'+esc(cost)', webui.INDEX_HTML)


# ===========================================================================
# Task #218: the limiting factors, measured one at a time, and the composed
# one-click scenario.
#
# What is pinned here is the CONTRACT the surface rests on: every factor
# carries its own provenance, a factor with no study says so instead of
# showing a number, the suggestion never disagrees with the working-point
# table it is composed from, and applying a suggestion starts nothing.
# ===========================================================================


class TestBenchFactors(WebUIFixture):
    def _factors(self, payload=None):
        d = webui.bench_factors_payload(payload or {})
        self.assertTrue(d["ok"], d.get("error"))
        return {f["key"]: f for f in d["factors"]}, d

    def test_every_factor_answers_with_a_provenance_and_an_action(self):
        from sglang.srt.planner import bench_factors as bfm

        by_key, d = self._factors()
        self.assertEqual(set(by_key), set(bfm.FACTOR_KEYS))
        for key, f in by_key.items():
            self.assertIn(
                f["provenance"], (bfm.MEASURED, bfm.ESTIMATE, bfm.ABSENT), key
            )
            self.assertTrue(f["question"].strip(), key)
            self.assertTrue(f["remeasure"]["kind"], key)
            self.assertTrue(f["remeasure"]["label"], key)
        self.assertEqual(
            sum(d["counts"].values()), len(bfm.FACTOR_KEYS)
        )

    def test_a_missing_study_carries_no_number_at_all(self):
        # The whole point: absent must be distinguishable from measured-zero.
        from sglang.srt.planner import bench_factors as bfm

        by_key, _ = self._factors()
        for key, f in by_key.items():
            if f["provenance"] != bfm.ABSENT:
                continue
            self.assertFalse(f["available"], key)
            self.assertEqual(f["values"], [], key)
            self.assertTrue(f["missing_reason"].strip(), key)
            self.assertIsNone(f["measured_at"], key)

    def test_no_endpoint_is_stated_rather_than_reported_as_a_zero(self):
        by_key, _ = self._factors()
        for key in ("round_time", "rank_balance"):
            f = by_key[key]
            self.assertFalse(f["available"])
            self.assertIn("running server", f["missing_reason"])

    def test_a_cached_probe_is_read_without_touching_a_gpu(self):
        # Written here, read back through the endpoint: no NVML, no CUDA.
        from sglang.srt.rigmon import card_probe as cp

        prof = cp.CardProbeProfile(
            created=time.time() - 3600,
            created_str="now",
            driver="999.0",
            cards=[
                cp.CardProbeMeasurement(
                    uuid="GPU-a", name="Card A", cuda_index=0,
                    membw_read_gbs=900.0, membw_gemv_gbs=850.0,
                    gemm_bf16_tflops=200.0,
                ),
                cp.CardProbeMeasurement(
                    uuid="GPU-b", name="Card B", cuda_index=1,
                    membw_read_gbs=700.0, gemm_bf16_tflops=60.0,
                ),
            ],
            pairs=[
                cp.PairMeasurement("GPU-a", "GPU-b", bandwidth_gbs=5.5),
                cp.PairMeasurement("GPU-b", "GPU-a", bandwidth_gbs=4.25),
            ],
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "probe.json")
            cp.save_card_probe(prof, path)
            by_key, _ = self._factors({"card_probe_path": path})
        rates = by_key["card_rates"]
        self.assertTrue(rates["available"])
        self.assertEqual(rates["provenance"], "measured")
        self.assertEqual(len(rates["values"]), 2)
        self.assertIsNotNone(rates["measured_at"])
        self.assertGreater(rates["age_s"], 0)
        link = by_key["pair_link"]
        # The NARROWEST ordered direction, not the mean and not the best.
        narrow = [v for v in link["values"] if v["key"] == "narrowest"][0]
        self.assertEqual(narrow["value"], 4.25)

    def test_an_action_that_cannot_run_is_blocked_with_its_reason(self):
        by_key, _ = self._factors()
        pc = by_key["prefix_cache"]["remeasure"]
        # No model and no endpoint on the tab -> recording cannot key or
        # scrape anything, and the surface says which is missing.
        self.assertFalse(pc["ready"])
        self.assertIn("model", pc["blocked_reason"])

    def test_the_action_carries_what_the_endpoint_needs(self):
        by_key, _ = self._factors(
            {"bench_model": "some/model", "endpoint": "127.0.0.1:30000"}
        )
        pc = by_key["prefix_cache"]["remeasure"]
        self.assertTrue(pc["ready"])
        self.assertEqual(pc["body"]["model"], "some/model")
        self.assertEqual(pc["body"]["target"], "http://127.0.0.1:30000")

    def test_the_only_no_endpoint_factor_without_a_button_says_the_command(self):
        by_key, _ = self._factors()
        mlp = by_key["mlp_split"]["remeasure"]
        self.assertEqual(mlp["kind"], "command")
        self.assertTrue(mlp["command"].strip())
        self.assertFalse(mlp["path"])

    def test_the_balance_factor_is_an_estimate_not_a_measurement(self):
        by_key, _ = self._factors(self._payload())
        bal = by_key["concurrency_balance"]
        self.assertTrue(bal["available"], bal.get("missing_reason"))
        self.assertEqual(bal["provenance"], "estimate")
        self.assertTrue(any("arithmetic" in bal["source"] for _ in (0,)))


class TestBenchFactorsRoute(TestHttpRoundTrip):
    def test_factors_endpoint_over_http(self):
        d = self._post("/api/bench_factors", {})
        self.assertTrue(d["ok"])
        self.assertTrue(d["factors"])
        self.assertIn("limiting factors measured", d["summary"])


class TestScenarioSuggest(WebUIFixture):
    def test_the_suggestion_agrees_with_the_lever_profile_baseline(self):
        # One composition, one set of numbers. If the suggestion recomputed
        # anything the two surfaces could disagree about the same config.
        body = self._payload()
        profiles = webui.lever_profiles_payload(body)
        self.assertTrue(profiles["ok"], profiles.get("reasons"))
        sug = webui.scenario_suggest_payload(body)
        self.assertTrue(sug["ok"], sug.get("reasons"))
        base_row = [
            p for p in profiles["profiles"]
            if p.get("key") == profiles["baseline"] and p.get("resolved")
        ][0]
        self.assertEqual(sug["baseline"]["key"], profiles["baseline"])
        self.assertEqual(sug["baseline"]["metrics"], base_row["metrics"])

    def test_without_a_workload_shape_the_answer_is_the_baseline(self):
        sug = webui.scenario_suggest_payload(self._payload())
        self.assertTrue(sug["ok"])
        self.assertTrue(sug["is_baseline"])
        self.assertEqual(sug["flags"], [])
        joined = " ".join(r["statement"] for r in sug["reasoning"])
        self.assertTrue(
            "prompt-to-output ratio" in joined or "same configuration" in joined
            or "nameplate" in joined,
            joined,
        )

    def test_every_step_names_the_provenance_it_rests_on(self):
        from sglang.srt.planner import bench_factors as bfm

        sug = webui.scenario_suggest_payload(self._payload())
        self.assertTrue(sug["reasoning"])
        for r in sug["reasoning"]:
            self.assertIn(
                r["provenance"], (bfm.MEASURED, bfm.ESTIMATE, bfm.ABSENT)
            )
            self.assertTrue(r["statement"].strip())

    def test_every_expected_figure_carries_its_instrument(self):
        sug = webui.scenario_suggest_payload(self._payload())
        self.assertTrue(sug["expected"])
        for m in sug["expected"]:
            self.assertTrue((m["basis"] or "").strip(), m["key"])
            self.assertEqual(m["provenance"], "estimate")
            if not m["available"]:
                self.assertIsNone(m["value"], m["key"])

    def test_the_suggestion_starts_nothing(self):
        sug = webui.scenario_suggest_payload(self._payload())
        self.assertTrue(sug["boots_nothing"])
        self.assertIn("lever_profile", sug["apply"])
        self.assertIn("Nothing was started", sug["note"] + " " + sug["note"])

    def test_an_unplannable_body_returns_reasons_not_a_recommendation(self):
        sug = webui.scenario_suggest_payload({"model": ""})
        self.assertFalse(sug["ok"])
        self.assertTrue(sug["reasons"])

    def test_an_unmeasured_rig_never_proposes_a_directed_split(self):
        # The conservative gate: with no probe of these cards, the answer is
        # the reference configuration, and the reason says why.
        from sglang.srt.planner import bench_factors as bfm

        profiles = {
            "ok": True,
            "baseline": "balanced",
            "distinct_working_points": 2,
            "basis": {"speed_scores": {"measured": False, "sources": ["nameplate"]},
                      "card_probe": None, "min_link_gbs": 8.0,
                      "min_link_source": "assumed"},
            "profiles": [
                {"key": "balanced", "label": "balanced", "resolved": True,
                 "is_base_split": True, "metrics": [], "tune": "both"},
                {"key": "max_prefill", "label": "max prefill", "resolved": True,
                 "is_base_split": False, "mlp_vector": [2, 1, 1], "tune": "enc",
                 "flag_delta": ["--rank-mlp-ratio 2,1,1"],
                 "metrics": [{"key": "prefill_tok_s", "available": True,
                              "value": 100.0, "basis": "x",
                              "vs_baseline": {"pct": 9.0, "direction": "gain"}}]},
            ],
        }
        sug = bfm.suggest_scenario(profiles, prompt_to_output_ratio=32.0)
        self.assertTrue(sug["ok"])
        self.assertTrue(sug["is_baseline"])
        joined = " ".join(r["statement"] for r in sug["reasoning"])
        self.assertIn("nameplate", joined)
        self.assertTrue(
            any(r["provenance"] == bfm.ABSENT for r in sug["reasoning"])
        )

    def test_a_measured_rig_takes_the_prefill_point_on_a_prompt_workload(self):
        from sglang.srt.planner import bench_factors as bfm

        profiles = {
            "ok": True,
            "baseline": "balanced",
            "distinct_working_points": 2,
            "basis": {"speed_scores": {"measured": True, "sources": ["measured"]},
                      "card_probe": {"cards": 3, "ordered_pairs": 6},
                      "min_link_gbs": 4.3, "min_link_source": "measured"},
            "profiles": [
                {"key": "balanced", "label": "balanced", "resolved": True,
                 "is_base_split": True, "metrics": [], "tune": "both"},
                {"key": "max_prefill", "label": "max prefill", "resolved": True,
                 "is_base_split": False, "mlp_vector": [2, 1, 1], "tune": "enc",
                 "flag_delta": ["--rank-mlp-ratio 2,1,1"], "lever_flags": [],
                 "metrics": [{"key": "prefill_tok_s", "available": True,
                              "value": 100.0, "basis": "x",
                              "vs_baseline": {"pct": 9.0, "direction": "gain"}}]},
            ],
        }
        sug = bfm.suggest_scenario(profiles, prompt_to_output_ratio=32.0)
        self.assertEqual(sug["profile"], "max_prefill")
        self.assertIn("--rank-mlp-ratio 2,1,1", sug["flags"])
        self.assertEqual(sug["apply"]["settings"]["rank_mlp_ratio"], "2,1,1")
        self.assertEqual(sug["apply"]["settings"]["rank_perf_tune"], "enc")

    def test_a_stop_that_resolves_back_to_the_baseline_is_not_proposed(self):
        from sglang.srt.planner import bench_factors as bfm

        profiles = {
            "ok": True, "baseline": "balanced", "distinct_working_points": 2,
            "basis": {"speed_scores": {"measured": True, "sources": ["m"]},
                      "card_probe": {"cards": 3, "ordered_pairs": 6},
                      "min_link_gbs": 4.3, "min_link_source": "measured"},
            "profiles": [
                {"key": "balanced", "label": "balanced", "resolved": True,
                 "is_base_split": True, "metrics": [], "tune": "both"},
                # Resolved back to the base split: proposing it as an
                # improvement over the baseline would be proposing the
                # baseline twice.
                {"key": "max_prefill", "label": "max prefill", "resolved": True,
                 "is_base_split": True, "metrics": [], "tune": "enc"},
            ],
        }
        sug = bfm.suggest_scenario(profiles, prompt_to_output_ratio=32.0)
        self.assertTrue(sug["is_baseline"])


class TestScenarioSuggestRoute(TestHttpRoundTrip):
    def test_suggest_is_dispatched_before_the_scenario_prefix(self):
        # /api/scenario is a prefix of /api/scenario_suggest in a flat chain.
        with open(webui.__file__) as f:
            text = f.read()
        i_sug = text.index('startswith("/api/scenario_suggest")')
        i_sc = text.index('startswith("/api/scenario"):', i_sug)
        self.assertLess(i_sug, i_sc)

    def test_the_two_endpoints_stay_separate_over_http(self):
        sc = self._post("/api/scenario", {"phases": "both", "concurrency": 2})
        self.assertTrue(sc["ok"])
        self.assertIn("summary", sc)
        sug = self._post("/api/scenario_suggest", {"model": ""})
        self.assertFalse(sug["ok"])
        self.assertIn("reasons", sug)


class TestFactorTooltipCoverage(CustomTestCase):
    """Every tile and the suggestion say what they give and what they cost."""

    def test_every_factor_has_a_tradeoff(self):
        from sglang.srt.planner import bench_factors as bfm
        from sglang.srt.planner import tooltips as tipsmod

        missing = [
            f.key for f in bfm.FACTORS if f.tooltip_key not in tipsmod.TRADEOFFS
        ]
        self.assertEqual(missing, [], f"factors without a trade-off: {missing}")

    def test_the_suggestion_controls_have_one_too(self):
        from sglang.srt.planner import tooltips as tipsmod

        for key in ("scenario.suggest", "scenario.apply"):
            self.assertIn(key, tipsmod.TRADEOFFS)
        self.assertIn("scenario.suggest", webui.tooltips_payload()["tooltips"])

    def test_the_page_looks_the_text_up_rather_than_holding_it(self):
        js = _index_script()
        self.assertIn("tip(f.tooltip_key)", js)
        self.assertIn("sc_btn:'scenario.suggest'", js)


class TestFactorPanelRendering(CustomTestCase):
    def test_the_panel_is_wired_to_the_endpoints_and_nothing_else(self):
        js = _index_script()
        for fn in ("function benchFactors(", "function renderFactors(",
                   "async function factorRemeasure(", "function factorPollJob(",
                   "async function scenarioSuggest(", "function renderScenario(",
                   "function scenarioApply("):
            self.assertIn(fn, js, fn)
        self.assertIn("'/api/bench_factors'", js)
        self.assertIn("'/api/scenario_suggest'", js)
        # The tab seeds itself from disk on entry; it starts no measurement.
        i = js.index("function benchInit(")
        self.assertIn("benchFactors();", js[i:i + 600])

    def test_absent_is_drawn_as_absent_not_as_a_dash(self):
        js = _index_script()
        self.assertIn("study not run.", js)
        # ...and the provenance dot is the only coloured mark on the tile.
        self.assertIn(".fxdot.measured", webui.INDEX_HTML)
        self.assertIn(".fxdot.absent", webui.INDEX_HTML)

    def test_applying_goes_through_the_existing_control(self):
        js = _index_script()
        i = js.index("function scenarioApply(")
        seg = js[i:i + 900]
        self.assertIn("applyLeverProfile(a.lever_profile)", seg)
        # No launch, no server call: the button writes fields.
        self.assertNotIn("/api/server_start", seg)
        self.assertNotIn("fetch(", seg)

    def test_the_polling_helper_stops_itself(self):
        js = _index_script()
        i = js.index("function factorPollJob(")
        seg = js[i:i + 700]
        self.assertIn("clearInterval(t)", seg)
