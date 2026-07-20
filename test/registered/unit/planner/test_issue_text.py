"""CPU unit tests for the S2 issue-text generators (design §4 + §5B.2).

No GPU, no network, no send: everything is pure text/URL rendering + the
privacy scrub. The measured-only honesty of the benchmark/energy fields is
enforced structurally (a fabricated-number path must not exist).
"""

import dataclasses
import json
import os
import tempfile
import unittest
import urllib.parse

from sglang.srt.planner import issue_text as it
from sglang.srt.planner import scrub
from sglang.srt.planner.hardware import hardware_from_manual
from sglang.srt.planner.issue_text import (
    BenchmarkFields,
    EnergyFields,
    HardwareFingerprint,
    bug_from_plan,
    bug_issue,
    hardware_fingerprint_from_spec,
    results_from_plan,
    results_issue,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

RIG = ("RTX 5090:32607", "RTX 3080:20480", "RTX 3080:20480")

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


def _fingerprint():
    return hardware_fingerprint_from_spec(hardware_from_manual(RIG))


# ---------------------------------------------------------------------------
# Scrubbing (design §4.5).
# ---------------------------------------------------------------------------


class TestScrub(CustomTestCase):
    def test_path_to_basename(self):
        self.assertEqual(
            scrub.scrub_path("/spinning/llm_stuff/x/Qwen3.6-27B-AWQ"),
            "Qwen3.6-27B-AWQ",
        )
        self.assertEqual(scrub.scrub_path("/a/b/dir/"), "dir")

    def test_secrets_dropped(self):
        s = scrub.scrub_text("HF_TOKEN=hf_ABCDEFGHIJKLMNOPQRSTUV foo")
        self.assertNotIn("hf_ABCDEFGHIJKLMNOPQRSTUV", s)
        self.assertIn("<redacted>", s)

    def test_bare_hf_token_dropped(self):
        s = scrub.scrub_text("using hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ now")
        self.assertNotIn("hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ", s)

    def test_uuid_dropped(self):
        s = scrub.scrub_text(
            "GPU-12345678-1234-1234-1234-123456789abc here"
        )
        self.assertNotIn("12345678", s)
        self.assertIn("<uuid>", s)

    def test_ip_dropped_but_loopback_kept(self):
        s = scrub.scrub_text("connect 192.168.1.42 and 127.0.0.1")
        self.assertNotIn("192.168.1.42", s)
        self.assertIn("127.0.0.1", s)

    def test_launch_flag_model_path_basenamed(self):
        flags = ["--model /spinning/secret/user/Qwen3.6-27B", "--tp-size 3"]
        out = scrub.scrub_launch_flags(flags)
        self.assertEqual(out[0], "--model Qwen3.6-27B")
        self.assertEqual(out[1], "--tp-size 3")

    def test_hotset_env_basenamed(self):
        out = scrub.scrub_launch_flags(["SGLANG_MOE_HOTSET_FILE=/home/x/hot.json"])
        self.assertEqual(out[0], "SGLANG_MOE_HOTSET_FILE=hot.json")

    def test_log_excerpt_windows_around_error_and_scrubs(self):
        log = "\n".join(
            ["noise %d" % i for i in range(100)]
            + ["Traceback (most recent call last):", "boom at /home/u/f.py"]
            + ["tail %d" % i for i in range(100)]
        )
        ex = scrub.scrub_log_excerpt(log, radius=5)
        self.assertIn("Traceback", ex)
        self.assertNotIn("/home/u/f.py", ex)
        self.assertIn("f.py", ex)
        # windowed: not the whole 200-line log
        self.assertLess(len(ex.splitlines()), 30)


# ---------------------------------------------------------------------------
# Hardware fingerprint (anonymous: card model + count).
# ---------------------------------------------------------------------------


class TestFingerprint(CustomTestCase):
    def test_grouping_and_summary(self):
        fp = _fingerprint()
        self.assertEqual(fp.gpu_count, 3)
        self.assertEqual(fp.summary(), "1x RTX 5090 32GB, 2x RTX 3080 20GB")

    def test_no_uuid_field_exists(self):
        # Anonymity is structural: the fingerprint type cannot carry a UUID
        # or a hostname (design §4.5).
        names = {f.name for f in dataclasses.fields(HardwareFingerprint)}
        self.assertNotIn("uuid", names)
        self.assertFalse(any("host" in n and n != "host_ram_mib" for n in names))
        self.assertFalse(any("uuid" in n or "serial" in n for n in names))


# ---------------------------------------------------------------------------
# RESULTS generator.
# ---------------------------------------------------------------------------


class TestResultsIssue(CustomTestCase):
    def _issue(self, **kw):
        base = dict(
            model_name="/spinning/llm_stuff/x/Qwen3.6-27B-AWQ-BF16-INT4",
            hardware=_fingerprint(),
            launch_flags=[
                "--tp-size 3",
                "--rank-gpu-id 0,1,2",
                "--rank-gpu-memory-mib 28591,16464,16464",
                "SGLANG_UNEVEN_DCP=1",
            ],
            fits=True,
            max_context_tokens=206748,
            quant="compressed-tensors",
            group_size=32,
        )
        base.update(kw)
        return results_issue(**base)

    def test_basic_shape_and_scrub(self):
        issue = self._issue()
        md = issue.markdown
        self.assertIn("## htsglang result", md)
        self.assertIn("1x RTX 5090 32GB, 2x RTX 3080 20GB", md)
        self.assertIn("Qwen3.6-27B-AWQ-BF16-INT4", md)
        # Path never leaks its directory.
        self.assertNotIn("/spinning/", md)
        self.assertIn("planner estimate", md)

    def test_prefilled_url_roundtrips(self):
        issue = self._issue()
        self.assertTrue(issue.url.startswith(
            "https://github.com/efschu/htsglang/issues/new?"
        ))
        q = urllib.parse.parse_qs(urllib.parse.urlparse(issue.url).query)
        self.assertEqual(q["body"][0], issue.markdown)
        self.assertTrue(issue.url_within_budget)

    def test_no_benchmark_section_without_measured_data(self):
        issue = self._issue()
        self.assertNotIn("### Benchmark", issue.markdown)
        self.assertNotIn("### Energy", issue.markdown)
        self.assertNotIn("tok/s", issue.markdown)

    def test_benchmark_section_only_with_measured_data(self):
        issue = self._issue(
            benchmark=BenchmarkFields(
                prefill_tok_s=512.0, decode_tok_s=48.0, batch=8, concurrency=8
            )
        )
        self.assertIn("### Benchmark (measured, opt-in)", issue.markdown)
        self.assertIn("512", issue.markdown)
        self.assertIn("Decode: 48", issue.markdown)

    def test_energy_section_only_with_measured_data(self):
        issue = self._issue(
            energy=EnergyFields(
                j_per_prefill_token="0.42",
                kwh_saved="2.6",
                conditions="sampling 20Hz, live",
            )
        )
        self.assertIn("### Energy (measured, opt-in)", issue.markdown)
        self.assertIn("2.6 kWh", issue.markdown)

    def test_capacity_range_only_when_stock_runs(self):
        # stock does not run -> no capacity % claimed
        issue = self._issue(stock_runs=False, stock_reason="kv_heads < tp")
        self.assertNotIn("Capacity vs stock", issue.markdown)
        # stock runs -> the range renders
        issue2 = self._issue(stock_runs=True, capacity_pct_range=(120, 135))
        self.assertIn("Capacity vs stock even-TP: **+120% .. +135%**", issue2.markdown)


# ---------------------------------------------------------------------------
# BUG generator.
# ---------------------------------------------------------------------------


class TestBugIssue(CustomTestCase):
    def test_bug_divergence_line_and_scrub(self):
        log = "\n".join(
            ["init"]
            + ["Traceback (most recent call last):", "at /home/u/mr.py line 88"]
            + ["torch.cuda.OutOfMemoryError: CUDA out of memory"]
        )
        issue = bug_issue(
            model_name="/x/y/Qwen3.6-27B",
            hardware=_fingerprint(),
            launch_flags=["--tp-size 2", "--rank-gpu-id 0,1"],
            symptom="OOM at load",
            planner_expected="planner said: fits",
            planner_max_context=90000,
            log_text=log,
        )
        md = issue.markdown
        self.assertIn("## htsglang bug", md)
        self.assertIn("### Planner expectation vs outcome", md)
        self.assertIn("planner: planner said: fits", md)
        self.assertIn("actual: OOM at load", md)
        self.assertIn("### Log excerpt", md)
        self.assertNotIn("/home/u/", md)
        self.assertIn("OutOfMemory", md)


# ---------------------------------------------------------------------------
# Bridges from an S1 PlanResult (the primary reuse).
# ---------------------------------------------------------------------------


class TestPlanBridges(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.planner.feasibility import plan

        cls._tmp = tempfile.TemporaryDirectory()
        path = os.path.join(cls._tmp.name, "model")
        os.makedirs(path)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(_CONFIG, f)
        with open(os.path.join(path, "model-00001.safetensors"), "wb") as f:
            f.truncate(int(14 * 2**30))
        cls.hw = hardware_from_manual(RIG)
        cls.result = plan(path, cls.hw, tp_size=3)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_results_from_plan_reuses_launch_flags(self):
        issue = results_from_plan(self.result, quant="compressed-tensors")
        self.assertIn("### Config that ran", issue.markdown)
        self.assertIn("--rank-gpu-id 0,1,2", issue.markdown)
        # Env token routed into the config block, not as a flag arg.
        self.assertIn("SGLANG_UNEVEN_DCP=1", issue.markdown)
        # stock cannot shard 4 KV heads across 3 -> verdict carries it, no
        # fabricated capacity %.
        self.assertIn("planner estimate", issue.markdown)
        self.assertNotIn("Capacity vs stock", issue.markdown)

    def test_results_from_plan_has_no_benchmark_by_default(self):
        issue = results_from_plan(self.result)
        self.assertNotIn("### Benchmark", issue.markdown)
        self.assertNotIn("### Energy", issue.markdown)

    def test_bug_from_plan_uses_planner_verdict_as_expected(self):
        issue = bug_from_plan(
            self.result, symptom="NCCL hang after boot"
        )
        self.assertIn("planner said: fits", issue.markdown)
        self.assertIn("NCCL hang after boot", issue.markdown)


# ---------------------------------------------------------------------------
# Honesty structure: measured-only benchmark/energy is enforced by the type
# system, not by convention (design §5B.2 / §3.4 mirror).
# ---------------------------------------------------------------------------


class TestHonestyStructure(CustomTestCase):
    def test_benchmark_energy_are_measured_only_optionals(self):
        # Every field of the measured dataclasses defaults to None: there is
        # no constructor path that produces a number without a caller
        # supplying a measured one.
        for cls in (BenchmarkFields, EnergyFields):
            obj = cls()
            self.assertFalse(obj.has_data())
            for f in dataclasses.fields(cls):
                self.assertIsNone(getattr(obj, f.name), f.name)

    def test_generators_expose_no_estimated_perf_parameter(self):
        # results_issue must not accept an *estimated* throughput/energy
        # argument — only the measured dataclasses, which are opt-in.
        import inspect

        sig = inspect.signature(results_issue)
        forbidden = ("tok", "tps", "throughput", "watt", "joule", "energy_est")
        for name in sig.parameters:
            low = name.lower()
            # max_context_tokens / max_total_num_tokens are KV/capacity
            # (memory) quantities, not throughput rates — same carve-out as
            # the S1 honesty test.
            if name in (
                "benchmark",
                "energy",
                "max_context_tokens",
                "max_total_num_tokens",
            ):
                continue
            self.assertFalse(
                any(bad in low for bad in forbidden),
                f"results_issue param {name} looks like an estimated-perf knob",
            )

    def test_no_module_level_fabricated_defaults(self):
        # The measured dataclasses carry only Optional fields; assert none is
        # a non-None numeric default that would render without measurement.
        for cls in (BenchmarkFields, EnergyFields):
            for f in dataclasses.fields(cls):
                self.assertIsNone(f.default, f"{cls.__name__}.{f.name}")


if __name__ == "__main__":
    unittest.main()
