"""CPU unit tests for the measured-KV-budget view and run provenance."""

import json
import os
import tempfile
import time
import unittest

from sglang.srt.rigmon.kvbudget import describe_budget, list_budget_files, reset_budget
from sglang.srt.rigmon.provenance import (
    RunRecord,
    capture_provenance,
    compare_runs,
    model_fingerprint,
    state_summary,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


GIB = 1024**3

#: Shaped after the real kv_budget-*.json on this rig.
BUDGET = {
    "safety_mib": 3072,
    "mlp_vector": [6, 1, 1],
    "components": [
        {
            "device_total_bytes": 33647230976,
            "ranks_on_gpu": 1,
            "residual_residency_bytes": 2127183360,  # 1.98 GiB not ours
            "weights_alloc_bytes": 18406912000,
            "kv_pool_bytes": 805601280,
            "kv_pool_tokens": 24584,
            "mamba_aux_pool_bytes": 1052447232,
            "graphs_ws_bytes": 789718528,
            "frag_bytes": 313202688,
            "free_bytes_at_measure": 11255087104,
            "max_total_num_tokens": 98328,
            "safety_mib": 3072,
        },
        {
            "device_total_bytes": 21029126144,
            "ranks_on_gpu": 1,
            "residual_residency_bytes": 104857600,  # 100 MiB, benign
            "weights_alloc_bytes": 6680857600,
            "kv_pool_bytes": 1208385536,
            "kv_pool_tokens": 36876,
            "free_bytes_at_measure": 10429530112,
            "max_total_num_tokens": 98328,
            "safety_mib": 640,
        },
    ],
}


def write_budget(d, data=BUDGET, name="kv_budget-6d2e86083a3d.json"):
    path = os.path.join(d, name)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestKvBudget(CustomTestCase):
    def test_listing_finds_budget_files_only(self):
        with tempfile.TemporaryDirectory() as d:
            write_budget(d)
            with open(os.path.join(d, "hw_profile-abc.json"), "w") as f:
                f.write("{}")
            found = list_budget_files(d)
        self.assertEqual(len(found), 1)
        self.assertIn("kv_budget-", found[0])

    def test_missing_cache_dir_is_not_an_error(self):
        self.assertEqual(list_budget_files("/nonexistent/path/xyz"), [])

    def test_describe_extracts_the_context_number_and_the_hash(self):
        with tempfile.TemporaryDirectory() as d:
            b = describe_budget(write_budget(d))
        self.assertEqual(b.config_hash, "6d2e86083a3d")
        self.assertEqual(b.max_total_num_tokens, 98328)
        self.assertEqual(b.ranks, 2)
        self.assertEqual(b.safety_mib, 3072)

    def test_foreign_residency_is_surfaced_as_the_explanation(self):
        """This is the answer to 'why is my context different today': a
        competing process was resident when the budget was measured, and the
        occupancy is carried into every later boot."""
        with tempfile.TemporaryDirectory() as d:
            b = describe_budget(write_budget(d))
        self.assertEqual(b.components[0]["foreign_residency_mib"], 2029)
        warn = " ".join(b.warnings)
        self.assertIn("rank 0", warn)
        self.assertIn("carries that occupancy forward", warn)
        # The small one must not raise a warning.
        self.assertNotIn("rank 1", warn)

    def test_per_rank_breakdown_is_in_mib(self):
        with tempfile.TemporaryDirectory() as d:
            b = describe_budget(write_budget(d))
        c = b.components[0]
        self.assertEqual(c["kv_pool_tokens"], 24584)
        self.assertEqual(c["device_total_mib"], 32088)
        self.assertGreater(c["weights_mib"], 17000)

    def test_disagreeing_ranks_are_flagged_and_the_min_wins(self):
        data = json.loads(json.dumps(BUDGET))
        data["components"][1]["max_total_num_tokens"] = 24576
        with tempfile.TemporaryDirectory() as d:
            b = describe_budget(write_budget(d, data))
        self.assertEqual(b.max_total_num_tokens, 24576)
        self.assertIn("disagree", " ".join(b.warnings))

    def test_old_budget_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_budget(d)
            old = time.time() - 60 * 86400
            os.utime(path, (old, old))
            b = describe_budget(path)
        self.assertIn("days old", " ".join(b.warnings))

    def test_reset_backs_up_and_explains_the_consequence(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_budget(d)
            res = reset_budget(path)
            self.assertTrue(res["removed"])
            self.assertFalse(os.path.exists(path))
            self.assertTrue(os.path.exists(res["backup"]))
            self.assertIn("re-measures", res["effect"])

    def test_reset_without_backup(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_budget(d)
            res = reset_budget(path, backup=False)
        self.assertIsNone(res["backup"])

    def test_reset_of_missing_file_is_reported_not_raised(self):
        res = reset_budget("/nonexistent/kv_budget-x.json")
        self.assertFalse(res["removed"])
        self.assertIn("no such file", res["reason"])


class TestProvenance(CustomTestCase):
    def test_secrets_never_enter_the_record(self):
        p = capture_provenance(
            ["--tp-size", "3"],
            env={
                "CUDA_VISIBLE_DEVICES": "0,1,2",
                "GITHUB_TOKEN": "ghp_secret",
                "HF_TOKEN": "hf_secret",
                "AWS_SECRET_ACCESS_KEY": "x",
            },
        )
        blob = json.dumps(p.to_json())
        self.assertIn("CUDA_VISIBLE_DEVICES", blob)
        self.assertNotIn("ghp_secret", blob)
        self.assertNotIn("hf_secret", blob)

    def test_command_line_is_copyable(self):
        p = capture_provenance(["--model", "/m", "--tp-size", "3"], env={})
        self.assertEqual(
            p.command_line(), "python -m sglang.launch_server --model /m --tp-size 3"
        )

    def test_model_fingerprint_changes_with_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "model.safetensors"), "wb") as f:
                f.write(b"x" * 100)
            a = model_fingerprint(d)
            with open(os.path.join(d, "model.safetensors"), "wb") as f:
                f.write(b"x" * 200)
            b = model_fingerprint(d)
        self.assertNotEqual(a["fingerprint"], b["fingerprint"])
        self.assertIn("not a content hash", a["method"])

    def test_missing_model_is_reported(self):
        self.assertFalse(model_fingerprint("/nope/xyz")["exists"])

    def test_repeatability_names_its_blockers(self):
        p = capture_provenance(["--tp-size", "3"], env={})
        p.git = {"commit": "abc", "dirty": True}
        p.model = {"fingerprint": "deadbeef"}
        rep = p.repeatable()
        self.assertFalse(rep["repeatable"])
        self.assertIn("dirty", " ".join(rep["blockers"]))

    def test_repeatable_when_everything_is_pinned(self):
        p = capture_provenance(["--tp-size", "3"], env={})
        p.git = {"commit": "abc", "dirty": False}
        p.model = {"fingerprint": "deadbeef"}
        p.kv_budget = {"present": False}
        self.assertTrue(p.repeatable()["repeatable"])

    def test_kv_budget_in_force_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            b = describe_budget(write_budget(d))
            p = capture_provenance(["--tp-size", "3"], kv_budget=b, env={})
        self.assertEqual(p.kv_budget["config_hash"], "6d2e86083a3d")
        self.assertEqual(p.kv_budget["max_total_num_tokens"], 98328)


class TestStateAndComparison(CustomTestCase):
    CLEAN = [
        {"index": 0, "sm_clock_mhz": 2900, "sm_clock_max_mhz": 3090, "temp_c": 62.0,
         "throttle": ["gpu_idle"]},
    ]
    #: The observed case: 88 C, thermal slowdown, 1695 of 1905 MHz.
    THROTTLED = [
        {"index": 2, "sm_clock_mhz": 1695, "sm_clock_max_mhz": 1905, "temp_c": 88.0,
         "throttle": ["sw_thermal_slowdown"]},
    ]

    def _run(self, value, cards, created, commit="abc", dirty=False):
        p = capture_provenance(["--tp-size", "3"], env={}, now=created)
        p.git = {"commit": commit, "dirty": dirty}
        return RunRecord(
            provenance=p, metrics={"tok_s": value}, state=state_summary(cards)
        )

    def test_state_summary_flags_throttling_and_low_clock(self):
        s = state_summary(self.THROTTLED)
        self.assertFalse(s["clean"])
        self.assertEqual(s["throttled"][0]["reasons"], ["sw_thermal_slowdown"])
        self.assertEqual(s["low_clock"][0]["mhz"], 1695)
        self.assertEqual(s["max_temp_c"], 88.0)

    def test_idle_is_not_a_throttle(self):
        self.assertTrue(state_summary(self.CLEAN)["clean"])

    def test_without_a_noise_floor_the_verdict_is_unknown(self):
        runs = [self._run(100.0, self.CLEAN, 1), self._run(94.0, self.CLEAN, 2)]
        out = compare_runs(runs, "tok_s")
        self.assertEqual(out["verdict"], "unknown")
        self.assertIn("noise floor", out["explanation"])

    def test_small_difference_is_within_noise(self):
        runs = [self._run(100.0, self.CLEAN, 1), self._run(97.0, self.CLEAN, 2)]
        out = compare_runs(runs, "tok_s", noise_floor_pct=5.0)
        self.assertEqual(out["verdict"], "within_noise")

    def test_thermal_snapshot_is_not_called_a_regression(self):
        """The whole point: a run taken while a card was thermally limited
        must not read as a code regression."""
        runs = [self._run(100.0, self.CLEAN, 1), self._run(78.0, self.THROTTLED, 2)]
        out = compare_runs(runs, "tok_s", noise_floor_pct=3.0)
        self.assertEqual(out["verdict"], "not_comparable")
        self.assertIn("thermal state", " ".join(out["caveats"]))

    def test_real_change_on_comparable_runs(self):
        runs = [self._run(100.0, self.CLEAN, 1), self._run(78.0, self.CLEAN, 2)]
        out = compare_runs(runs, "tok_s", noise_floor_pct=3.0)
        self.assertEqual(out["verdict"], "changed")
        self.assertAlmostEqual(out["delta_pct"], -22.0, places=6)

    def test_different_commits_block_the_verdict(self):
        runs = [
            self._run(100.0, self.CLEAN, 1, commit="aaa"),
            self._run(78.0, self.CLEAN, 2, commit="bbb"),
        ]
        out = compare_runs(runs, "tok_s", noise_floor_pct=3.0)
        self.assertEqual(out["verdict"], "not_comparable")
        self.assertIn("different commits", " ".join(out["caveats"]))

    def test_single_run_is_insufficient(self):
        out = compare_runs([self._run(100.0, self.CLEAN, 1)], "tok_s")
        self.assertEqual(out["verdict"], "insufficient")

    def test_runs_are_ordered_by_time(self):
        runs = [self._run(78.0, self.CLEAN, 9), self._run(100.0, self.CLEAN, 1)]
        out = compare_runs(runs, "tok_s", noise_floor_pct=3.0)
        self.assertEqual([p["value"] for p in out["points"]], [100.0, 78.0])


if __name__ == "__main__":
    unittest.main()
