"""BootMatrixTenant is a thin, well-behaved workbench tenant (#349).

Hermetic: exercises availability, pricing and the per-arm launch line without
spawning anything. start_segment's spawn is not driven here (it boots a
server); segment_argv, the pure command it would run, is.
"""

import tempfile
import unittest
from pathlib import Path

from sglang.srt.boot_matrix.arms import ARMS
from sglang.srt.workbench.tenants.boot_matrix import BootMatrixTenant
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBootMatrixTenant(CustomTestCase):
    def _tenant(self, model="/models/qwen"):
        return BootMatrixTenant(artifact_root=Path("/tmp/wb"), model_path=model)

    def test_unavailable_without_a_model(self):
        ok, why = BootMatrixTenant(artifact_root=Path("/tmp/wb")).available()
        self.assertFalse(ok)
        self.assertIn("model", why)

    def test_available_with_a_model(self):
        ok, _ = self._tenant().available()
        self.assertTrue(ok)

    def test_pending_starts_at_the_full_matrix(self):
        self.assertEqual(self._tenant().pending(), len(ARMS))

    def test_estimate_wants_every_card(self):
        est = self._tenant().estimate()
        self.assertEqual(est.cards_wanted, 0)  # 0 == every visible card
        self.assertGreater(est.per_card_bytes, 0)
        self.assertIn("serving_boot", est.posts)

    def test_estimate_uses_the_next_arms_time(self):
        t = self._tenant()
        first = ARMS[0]
        self.assertEqual(t.estimate().expected_seconds, first.expected_seconds)

    def test_segment_argv_targets_one_arm(self):
        t = self._tenant()
        argv = t.segment_argv(ARMS[0])
        self.assertIn("sglang.srt.boot_matrix.sweep", argv)
        self.assertIn("--only", argv)
        self.assertIn(ARMS[0].name, argv)
        self.assertIn("--model", argv)
        self.assertIn("/models/qwen", argv)

    def test_enqueue_rearms_a_known_arm(self):
        t = self._tenant()
        name = ARMS[0].name
        t._done.add(name)
        self.assertEqual(t.pending(), len(ARMS) - 1)
        t.enqueue({"arm": name})
        self.assertEqual(t.pending(), len(ARMS))

    def test_enqueue_rejects_an_unknown_arm(self):
        with self.assertRaises(ValueError):
            self._tenant().enqueue({"arm": "not_an_arm"})

    def test_snapshot_is_json_shaped(self):
        snap = self._tenant().snapshot()
        self.assertEqual(snap["name"], "boot_matrix")
        self.assertIn("pending", snap)
        self.assertTrue(snap["available"])

    def test_service_registry_knows_the_tenant(self):
        """build_tenants must accept 'boot_matrix' and wire the model through."""
        from types import SimpleNamespace

        from sglang.srt.workbench.scheduler import WorkbenchConfig
        from sglang.srt.workbench.service import build_tenants

        with tempfile.TemporaryDirectory() as d:
            cfg = WorkbenchConfig(enabled=True, artifact_root=d)
            sa = SimpleNamespace(
                workbench_tenants="boot_matrix",
                workbench_boot_matrix_model="/models/qwen",
            )
            tenants = build_tenants(sa, cfg)
            self.assertEqual([t.name for t in tenants], ["boot_matrix"])
            self.assertEqual(tenants[0].model_path, "/models/qwen")


if __name__ == "__main__":
    unittest.main()
