"""The pass-3 executor: every refusal fires, every mapped resolution lands.

Desk-written-never-executed rule: the real pass is gated on comp4's boot
proof and is NOT run here. What runs here, in synthetic throwaway repos:
the gate refusal, the standing-rule lineage abort, the unpredicted-conflict
loud abort (THE can-fail: a stale map must stop the train, never be
resolved ad hoc), the 'ours' and 'theirs'+markers resolutions, the
marker-loss abort, and the manual-stop resumable path.
"""

import json
import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../../scripts")
)

import merge_train_pass3 as m3  # noqa: E402


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=True
    )


def _mini_repo(tmp):
    """base -> (side branch edits a.txt) ; main edits a.txt differently."""
    repo = os.path.join(tmp, "repo")
    os.makedirs(repo)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@t")
    with open(os.path.join(repo, "a.txt"), "w") as f:
        f.write("base\n")
    with open(os.path.join(repo, "b.txt"), "w") as f:
        f.write("markers: 32607 4577\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "checkout", "-qb", "side")
    with open(os.path.join(repo, "a.txt"), "w") as f:
        f.write("side change\n")
    with open(os.path.join(repo, "b.txt"), "w") as f:
        f.write("markers survived: 32607 4577\n")
    _git(repo, "commit", "-qam", "side edit")
    side = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", "main")
    with open(os.path.join(repo, "a.txt"), "w") as f:
        f.write("main change\n")
    with open(os.path.join(repo, "b.txt"), "w") as f:
        f.write("markers gone from ours\n")
    _git(repo, "commit", "-qam", "main edit")
    return repo, side


class TestTheGate(CustomTestCase):
    def test_missing_marker_refuses(self):
        with self.assertRaises(m3.Abort) as ctx:
            m3.check_gate("/nonexistent/COMP4_ACCEPTED_757")
        self.assertIn("boot-proof-gated", str(ctx.exception))

    def test_present_marker_passes(self):
        import tempfile

        with tempfile.NamedTemporaryFile() as f:
            m3.check_gate(f.name)

    def test_cli_refuses_end_to_end_without_marker(self):
        """The refusal through the real entry point, not just the helper."""
        proc = subprocess.run(
            [
                sys.executable,
                m3.__file__,
                "--repo",
                "/tmp",
                "--marker",
                "/nonexistent/COMP4_ACCEPTED_757",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("GATE", proc.stderr)


class TestConflictDiscipline(CustomTestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp(prefix="pass3-test-")
        self.addCleanup(self._rm)
        self.repo, self.side = _mini_repo(self.tmp)

    def _rm(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unpredicted_conflict_aborts_without_resolving(self):
        """THE can-fail: a conflict outside the predicted set means the map
        is stale; the script must stop, not improvise."""
        step = m3.Step(name="t", kind="cherry", ref=self.side, predicted={})
        with self.assertRaises(m3.Abort) as ctx:
            m3.execute_step(self.repo, step, dry_run=True)
        self.assertIn("UNPREDICTED", str(ctx.exception))
        self.assertIn("a.txt", str(ctx.exception))
        # and it left the conflict standing for the operator's --abort:
        self.assertTrue(m3.conflicted_paths(self.repo))

    def test_ours_resolution_drops_the_incoming_hunk(self):
        step = m3.Step(
            name="t",
            kind="cherry",
            ref=self.side,
            predicted={"a.txt": "ours", "b.txt": "ours"},
        )
        out = m3.execute_step(self.repo, step, dry_run=True)
        self.assertIsNone(out)
        with open(os.path.join(self.repo, "a.txt")) as f:
            self.assertEqual(f.read(), "main change\n")

    def test_theirs_resolution_verifies_markers(self):
        step = m3.Step(
            name="t",
            kind="cherry",
            ref=self.side,
            predicted={"a.txt": "ours", "b.txt": "theirs"},
            markers={"b.txt": ["32607", "4577"]},
        )
        self.assertIsNone(m3.execute_step(self.repo, step, dry_run=True))
        with open(os.path.join(self.repo, "b.txt")) as f:
            self.assertIn("survived", f.read())

    def test_theirs_resolution_aborts_when_a_marker_is_lost(self):
        step = m3.Step(
            name="t",
            kind="cherry",
            ref=self.side,
            predicted={"a.txt": "ours", "b.txt": "theirs"},
            markers={"b.txt": ["THIS-STRING-DOES-NOT-EXIST"]},
        )
        with self.assertRaises(m3.Abort) as ctx:
            m3.execute_step(self.repo, step, dry_run=True)
        self.assertIn("load-bearing markers", str(ctx.exception))

    def test_manual_prediction_stops_resumably(self):
        step = m3.Step(
            name="t",
            kind="cherry",
            ref=self.side,
            predicted={"a.txt": "manual", "b.txt": "ours"},
        )
        out = m3.execute_step(self.repo, step, dry_run=True)
        self.assertIsNotNone(out)
        self.assertIn("manual resolution required", out)
        self.assertIn("a.txt", out)

    def test_skip_kind_is_inert(self):
        step = m3.Step(name="t", kind="skip", ref="deadbeef")
        self.assertIsNone(m3.execute_step(self.repo, step, dry_run=True))


class TestThePlanMirrorsTheLedger(CustomTestCase):
    """The executor IS section (c); pin the load-bearing identities so an
    edit to either artifact has to touch both."""

    def test_order_and_refs(self):
        names = [s.name for s in m3.PLAN]
        self.assertEqual(
            names,
            [
                "749-order-dependence",
                "751-preflight-boundary",
                "754-layerset-scope-guard",
                "735-arithmetic-docs",
                "727-requant-method",
                "727-lmhead-artifact",
                "727-ab-runner",
                "755-determination-docs",
                "740-scaffold-note",
                "740-scaffold-residual",
            ],
        )
        by_name = {s.name: s for s in m3.PLAN}
        self.assertEqual(by_name["754-layerset-scope-guard"].ref, "f384591531")
        self.assertEqual(
            by_name["754-layerset-scope-guard"].predicted,
            {
                "test/registered/unit/mem_cache/test_hicache_gdn_layer_counter_752.py": "ours"
            },
        )
        self.assertEqual(
            by_name["735-arithmetic-docs"].markers[
                "docs/dev/DESIGN_pp_layer_set.md"
            ],
            ["32607", "4577"],
        )

    def test_no_step_pushes(self):
        import inspect

        src = inspect.getsource(m3)
        self.assertNotIn('"push"', src)

    def test_state_roundtrip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            st = m3.load_state(p)
            self.assertEqual(st["status"], "fresh")
            st["done"].append("x")
            st["status"] = "manual-stop"
            m3.save_state(p, st)
            st2 = m3.load_state(p)
            self.assertEqual(st2["done"], ["x"])
            self.assertEqual(st2["status"], "manual-stop")
            self.assertEqual(json.load(open(p))["status"], "manual-stop")


if __name__ == "__main__":
    unittest.main()
