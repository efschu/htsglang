"""Named working points and their counter-reckoning (#215).

What is pinned here is the property that makes the control trustworthy rather
than merely present: a profile named after a quantity is, over the shared
candidate set, the best profile at that quantity. ``max context`` reports at
least as many context tokens as any other stop, ``max decode`` at least as
much decode throughput, ``max prefill`` at least as much prefill throughput.
That holds by construction -- one candidate set, one instrument per objective,
and the reported figure IS the ranked one -- and these tests keep it holding.

The second thing pinned is the refusal discipline. A figure that cannot be
computed comes back absent WITH a reason; it never comes back as zero, and it
never comes back as a plausible-looking constant. The registry convention
(:mod:`sglang.srt.planner.tooltips`) is the same one, and its coverage test
lives next door.

The rig and the model are the reference ones: Qwen3.6-27B FP8 geometry on the
5090 + 2x 3080 box, the same fixture ``test_mrr_balance`` uses, so the numbers
in the profile table can be read against the balance figures recorded there.
"""

import json
import os
import tempfile

from sglang.srt.planner import lever_profiles as lp
from sglang.srt.planner import webui
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

#: The reference rig: 5090 32 GiB + 2x 3080 20 GiB.
_RIG3 = ["RTX 5090:32760", "RTX 3080:20480", "RTX 3080:20480"]
#: A uniform rig -- no asymmetry for an MLP vector to exploit.
_RIG_UNIFORM = ["RTX 3080:20480", "RTX 3080:20480"]

#: Qwen3.6-27B geometry, verbatim from the checkpoint's ``text_config``.
_QWEN36_27B_TEXT = dict(
    model_type="qwen3_5",
    hidden_size=5120,
    num_hidden_layers=64,
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    intermediate_size=17408,
    vocab_size=248064,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    max_position_embeddings=262144,
    layer_types=[
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)
    ],
)


def _write_27b_config(tmpdir: str) -> str:
    cfg = dict(
        architectures=["Qwen3_5ForConditionalGeneration"],
        model_type="qwen3_5",
        text_config=dict(_QWEN36_27B_TEXT),
        quantization_config=dict(
            quant_method="fp8", fmt="e4m3", activation_scheme="dynamic"
        ),
    )
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(cfg, f)
    return tmpdir


class ProfileFixture(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model = _write_27b_config(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _payload(self, gpus=None, **kw):
        p = {
            "model": self.model,
            "hardware": {"source": "manual", "gpus": list(gpus or _RIG3)},
            "tp_size": len(gpus or _RIG3),
            "kv_cache_dtype": "fp8_e4m3",
            "max_running_requests": 16,
        }
        p.update(kw)
        return p

    def _report(self, **kw):
        d = webui.lever_profiles_payload(self._payload(**kw))
        self.assertTrue(d.get("ok"), d.get("reasons"))
        return d

    @staticmethod
    def _metric(row, key):
        return [m for m in row["metrics"] if m["key"] == key][0]

    @staticmethod
    def _row(d, key):
        return [p for p in d["profiles"] if p["key"] == key][0]


class TestShape(ProfileFixture):
    """The answer's shape is the contract the page and curl share."""

    def test_four_stops_in_the_documented_order(self):
        d = self._report()
        self.assertEqual(
            [p["key"] for p in d["profiles"]],
            ["max_context", "balanced", "max_decode", "max_prefill"],
        )
        self.assertEqual(
            [s["key"] for s in d["slider"]["steps"]], list(lp.PROFILE_KEYS)
        )
        self.assertEqual(d["baseline"], "balanced")

    def test_every_profile_carries_every_metric(self):
        d = self._report()
        for row in d["profiles"]:
            self.assertTrue(row["resolved"], row)
            keys = [m["key"] for m in row["metrics"]]
            self.assertEqual(keys, [m.key for m in lp.METRICS], row["key"])

    def test_each_metric_states_where_it_came_from(self):
        # A number without its basis travels further than it should.
        d = self._report()
        for m in self._row(d, "balanced")["metrics"]:
            self.assertTrue(m["basis"].strip(), m["key"])
            self.assertIn("higher_is_better", m)

    def test_the_slider_axis_is_stated(self):
        d = self._report()
        self.assertIn("context", d["slider"]["axis"])
        self.assertEqual(
            d["slider"]["default_index"], lp.PROFILE_KEYS.index("balanced")
        )

    def test_session_target_and_its_options_come_from_the_balance_module(self):
        from sglang.srt.planner import mrr_balance

        d = self._report()
        self.assertEqual(
            d["session_target_options"], list(mrr_balance.DEFAULT_TARGET_CONTEXTS)
        )
        self.assertEqual(d["session_target_context_tokens"], lp.DEFAULT_SESSION_CONTEXT)

    def test_a_chosen_session_target_is_honoured(self):
        d = self._report(session_target_context=32768)
        self.assertEqual(d["session_target_context_tokens"], 32768)
        m = self._metric(self._row(d, "balanced"), "sessions")
        if m["available"]:
            self.assertEqual(m["detail"]["target_context_tokens"], 32768)


class TestConsistency(ProfileFixture):
    """A profile named after a quantity is the best profile at it."""

    def _best_at(self, metric_key):
        d = self._report()
        vals = {}
        for row in d["profiles"]:
            m = self._metric(row, metric_key)
            if m["available"]:
                vals[row["key"]] = m["value"]
        self.assertTrue(vals, f"no profile reported {metric_key}")
        return vals

    def test_max_context_holds_the_most_context(self):
        vals = self._best_at("kv_tokens")
        top = vals["max_context"]
        for k, v in vals.items():
            # Equality is the normal case: total weight bytes do not change
            # when MLP units move, so the KV pool moves only when the
            # per-rank minimum binds.
            self.assertGreaterEqual(top + 1e-6, v, k)

    def test_max_decode_is_the_fastest_decode(self):
        vals = self._best_at("decode_tok_s")
        for k, v in vals.items():
            self.assertGreaterEqual(vals["max_decode"] * (1 + 1e-9), v, k)

    def test_max_prefill_is_the_fastest_prefill(self):
        vals = self._best_at("prefill_tok_s")
        for k, v in vals.items():
            self.assertGreaterEqual(vals["max_prefill"] * (1 + 1e-9), v, k)

    def test_the_baseline_is_its_own_reference(self):
        d = self._report()
        for m in self._row(d, "balanced")["metrics"]:
            self.assertNotIn("vs_baseline", m)
        for key in ("max_context", "max_decode", "max_prefill"):
            for m in self._row(d, key)["metrics"]:
                if m["available"]:
                    self.assertIn("vs_baseline", m, (key, m["key"]))
                    self.assertIn(
                        m["vs_baseline"]["direction"], ("gain", "cost", "same")
                    )

    def test_a_direction_is_a_gain_only_when_the_metric_improves(self):
        d = self._report()
        for row in d["profiles"]:
            for m in row["metrics"]:
                vb = m.get("vs_baseline")
                if not vb or vb["direction"] == "same":
                    continue
                better = vb["delta"] > 0 if m["higher_is_better"] else vb["delta"] < 0
                self.assertEqual(vb["direction"], "gain" if better else "cost")

    def test_the_balanced_row_matches_the_plan_the_verdict_shows(self):
        # One configuration, one answer: the profile table and the ordinary
        # plan must not describe the same thing differently.
        d = self._report()
        plan_d = webui.plan_from_payload(self._payload())
        self.assertTrue(plan_d["valid"], plan_d.get("reasons"))
        self.assertAlmostEqual(
            self._metric(self._row(d, "balanced"), "kv_tokens")["value"],
            plan_d["capacity"]["max_context_tokens"],
            places=3,
        )


class TestRefusals(ProfileFixture):
    """Absent is said out loud, with the missing input named."""

    def test_a_rejected_configuration_answers_with_reasons(self):
        d = webui.lever_profiles_payload(
            self._payload(rank_gpu_id="0,1,2,3")  # more ranks than cards
        )
        self.assertFalse(d["ok"])
        self.assertTrue(d["reasons"])
        self.assertEqual(d["profiles"], [])

    def test_a_missing_model_is_an_answer_not_an_exception(self):
        d = webui.lever_profiles_payload({"model": "/nope/not/a/model"})
        self.assertFalse(d["ok"])
        self.assertTrue(d["reasons"])

    def test_an_uncomputable_figure_names_what_is_missing(self):
        # Cards the library has no specs for and no probe covers: no roofline,
        # hence no level for a modelled move to be applied to. The context
        # figure is pure VRAM arithmetic and survives, which is exactly the
        # distinction a reader needs -- one number is missing, not all of them.
        d = self._report(
            gpus=[
                "Fantasy Accelerator XL:32768",
                "Fantasy Accelerator S:24576",
                "Fantasy Accelerator S:24576",
            ]
        )
        row = self._row(d, "balanced")
        self.assertTrue(self._metric(row, "kv_tokens")["available"])
        for key in ("decode_tok_s", "prefill_tok_s"):
            cell = self._metric(row, key)
            self.assertFalse(cell["available"], key)
            self.assertTrue(cell["reason"].strip(), key)
            self.assertNotIn("value", cell)
        self.assertIn("no per-rank speed figures", " ".join(d["caveats"]).lower())
        # ... and the profile says why it could not move, in its own words.
        self.assertIn("probe", self._row(d, "max_prefill")["selection_reason"].lower())

    def test_uniform_cards_resolve_every_stop_to_the_base_split(self):
        d = self._report(gpus=_RIG_UNIFORM)
        vectors = {tuple(p["mlp_vector"]) for p in d["profiles"] if p["resolved"]}
        self.assertEqual(len(vectors), 1, vectors)
        self.assertEqual(d["distinct_working_points"], 1)
        joined = " ".join(d["caveats"]).lower()
        self.assertIn("uniform", joined)
        for key in ("max_context", "max_decode", "max_prefill"):
            self.assertTrue(self._row(d, key)["same_as_baseline"])

    def test_a_stop_that_equals_the_baseline_says_so(self):
        d = self._report()
        for row in d["profiles"]:
            if row["key"] == "balanced":
                self.assertFalse(row["same_as_baseline"])
            else:
                self.assertEqual(row["same_as_baseline"], row["is_base_split"])

    def test_every_selection_carries_its_reason(self):
        d = self._report()
        for row in d["profiles"]:
            self.assertTrue(row["selection_reason"].strip(), row["key"])


class TestProvenance(ProfileFixture):
    """Modelled, measured and assumed stay distinguishable."""

    def test_the_speed_basis_names_both_instruments(self):
        d = self._report()
        dec = self._metric(self._row(d, "balanced"), "decode_tok_s")
        self.assertIn("roofline", dec["basis"])
        self.assertIn("cost model", dec["basis"])

    def test_nameplate_ranking_is_declared(self):
        # Without a probe of these cards the ranking runs on nameplate specs,
        # and that has to be said rather than implied.
        d = self._report()
        scores = d["basis"]["speed_scores"]
        if scores is not None and not scores["measured"]:
            self.assertIn("nameplate", " ".join(d["caveats"]).lower())

    def test_the_link_term_says_whether_it_was_measured(self):
        d = self._report()
        self.assertIn(d["basis"]["min_link_source"].split()[0], ("measured", "assumed"))

    def test_no_net_of_prefill_gain_and_decode_cost_is_reported(self):
        # crossover.MODELLED_NET_REFUSED: the two terms are each fitted on one
        # rig, so their net has no bound off it. No field may carry one.
        d = self._report()
        keys = {m["key"] for row in d["profiles"] for m in row["metrics"]}
        for forbidden in ("net", "net_ms_per_output_token", "net_gain"):
            self.assertNotIn(forbidden, keys)


class TestTooltips(CustomTestCase):
    """The control's hover text lives in the one registry, like every other."""

    def test_every_stop_and_the_slider_are_covered(self):
        from sglang.srt.planner import tooltips as tipsmod

        want = ["lever_profile.slider"] + [
            "lever_profile." + k for k in lp.PROFILE_KEYS
        ]
        self.assertEqual(tipsmod.missing_coverage(want), [])
        for key in want:
            txt = tipsmod.describe(key)
            self.assertIn("Costs:", txt)

    def test_a_stop_that_points_at_a_study_says_when_it_has_not_run(self):
        from sglang.srt.planner import tooltips as tipsmod

        txt = tipsmod.describe("lever_profile.max_prefill", measurements={})
        self.assertIn("Not measured on this rig", txt)


class TestFrontend(CustomTestCase):
    """The page renders the answer; it does not compute one."""

    @staticmethod
    def _js():
        import re

        m = re.search(r"<script>(.*)</script>", webui.INDEX_HTML, re.S)
        return m.group(1)

    def test_the_control_and_its_renderer_exist(self):
        js = self._js()
        for marker in (
            "function refreshLeverProfiles(",
            "function renderLeverProfiles(",
            "function applyLeverProfile(",
            "function leverProfileInput(",
            "/api/lever_profiles",
        ):
            self.assertIn(marker, js, marker)
        self.assertIn('id="lp_slider"', webui.INDEX_HTML)
        self.assertIn('id="lp_body"', webui.INDEX_HTML)

    def test_the_call_is_bounded_like_every_other(self):
        js = self._js()
        self.assertIn("api('/api/lever_profiles',{key:'lever_profiles'", js)

    def test_picking_a_stop_writes_the_state_the_expert_view_edits(self):
        # Not a second mode: the profile sets the ordinary MLP vector and the
        # tuning objective, so switching views shows one configuration.
        js = self._js()
        self.assertIn("window._flagSettings.rank_mlp_ratio=", js)
        self.assertIn("if(row.tune) applyTune(row.tune)", js)

    def test_the_page_holds_no_copy_of_the_profile_text(self):
        from sglang.srt.planner import tooltips as tipsmod

        js = self._js()
        for key in ["lever_profile.slider"] + [
            "lever_profile." + k for k in lp.PROFILE_KEYS
        ]:
            t = tipsmod.TRADEOFFS[key]
            self.assertNotIn(t.gain, js, key)
            self.assertNotIn(t.cost, js, key)
        for spec in lp.PROFILES:
            self.assertNotIn(spec.axis_note, js, spec.key)

    def test_the_wide_table_scrolls_inside_its_own_box(self):
        self.assertIn(".lp-wrap { overflow-x: auto; }", webui.INDEX_HTML)


class TestRoute(ProfileFixture):
    """One endpoint per step: the table is reachable with curl."""

    def setUp(self):
        import threading
        from http.server import ThreadingHTTPServer

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), webui._Handler)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(timeout=5)

    def test_post_route_answers_over_http(self):
        import urllib.request

        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/lever_profiles",
            data=json.dumps(self._payload()).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
        self.assertTrue(d["ok"], d.get("reasons"))
        self.assertEqual(len(d["profiles"]), len(lp.PROFILES))


if __name__ == "__main__":
    import unittest

    unittest.main()
