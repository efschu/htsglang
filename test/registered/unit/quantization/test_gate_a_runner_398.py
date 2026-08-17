# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The #398 Gate-A runner must still work when its window finally opens.

A turnkey script written weeks before the window it serves is the
desk-written-never-executed failure waiting to happen: nobody runs it until
the one night it has to work, and by then the tree has moved under it. So the
runner carries a hermetic ``--self-test`` and this pins that the self-test
still passes on today's tree.

It costs no GPU and no window -- that is the point.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
_RUNNER = os.path.join(_REPO, "bench", "398", "run_398_gate_a.py")


class TestTheRunnerSelfTestPasses(unittest.TestCase):
    def test_it_exists(self):
        self.assertTrue(os.path.exists(_RUNNER), _RUNNER)

    def test_self_test_is_green_hermetically(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = os.path.join(_REPO, "python")
        done = subprocess.run(
            [sys.executable, _RUNNER, "--self-test"],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        self.assertEqual(
            done.returncode,
            0,
            f"stdout:\n{done.stdout}\nstderr:\n{done.stderr}",
        )
        self.assertIn("self-test: OK", done.stdout)

    def test_the_self_test_reports_its_reject_checks(self):
        # A self-test with zero can-fail checks is decoration. The runner
        # counts them rather than claiming a number.
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = os.path.join(_REPO, "python")
        done = subprocess.run(
            [sys.executable, _RUNNER, "--self-test"],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        self.assertIn("rejects bad input", done.stdout)


class TestTheCorrectedFalsifierExpectation(unittest.TestCase):
    """#519's correction must not be re-broken by a future edit.

    The lever-off arm expects ``True/False/False``: the wheel marker is a
    property of what is on disk and the env lever cannot change it. An edit
    back to ``False/False/False`` would make the gate red on a correct build.
    """

    def _arms(self):
        sys.path.insert(0, os.path.dirname(_RUNNER))
        try:
            import run_398_gate_a as runner
        finally:
            sys.path.pop(0)
        return runner

    def test_lever_off_still_expects_the_marker_to_stay_true(self):
        runner = self._arms()
        lever_off = {a.name: a for a in runner.FALSIFIER_ARMS}["lever-off"]
        self.assertEqual(lever_off.expected, (True, False, False))

    def test_the_arm_rejects_the_pre_519_observation(self):
        runner = self._arms()
        lever_off = {a.name: a for a in runner.FALSIFIER_ARMS}["lever-off"]
        problems = runner.check_arm(runner.Observation(False, False, False), lever_off)
        self.assertTrue(problems)


if __name__ == "__main__":
    unittest.main()
