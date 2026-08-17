# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The #613 boot-validation runner must still work when its window opens.

Same reason as the #398 and #464 runners: a turnkey script written ahead of its
window is the desk-written-never-executed failure waiting to happen. This pins
the hermetic half so the window only has to buy the boots.

It also pins the judge's REJECT arms. A validator that cannot fail would report
a broken gate as validated, which is worse than not running it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_RUNNER = os.path.join(_REPO, "bench", "613", "run_613_regime_gate.py")


def _load():
    sys.path.insert(0, os.path.dirname(_RUNNER))
    try:
        import run_613_regime_gate as runner
    finally:
        sys.path.pop(0)
    return runner


class TestSelfTestIsGreen(unittest.TestCase):
    def test_runner_exists(self):
        self.assertTrue(os.path.exists(_RUNNER), _RUNNER)

    def test_self_test_passes_hermetically(self):
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
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("self-test: OK", done.stdout)


class TestTheJudgeRejects(unittest.TestCase):
    """The three failure modes the window must not report as a pass."""

    def _judge(self, **over):
        runner = _load()
        base = dict(
            long_gated_vs_eager=+0.05,
            long_graph_vs_eager=+10.25,
            short_gated_vs_eager=-4.60,
            short_graph_vs_eager=-4.62,
            long_floor=0.12,
            short_floor=1.99,
        )
        base.update(over)
        return runner.judge_gate(**base)

    def test_a_correct_gate_passes(self):
        self.assertTrue(self._judge().passed)

    def test_a_gate_that_did_not_refuse_the_long_point_fails(self):
        self.assertFalse(self._judge(long_gated_vs_eager=+10.10).passed)

    def test_a_gate_too_strict_to_keep_the_win_fails(self):
        self.assertFalse(self._judge(short_gated_vs_eager=-0.10).passed)

    def test_a_sub_floor_win_does_not_count(self):
        self.assertFalse(self._judge(short_gated_vs_eager=-1.00).passed)


class TestTheRunnerTestsTheShippingGate(unittest.TestCase):
    """The runner's workload points must match what the gate actually decides.

    If the runner drove shapes the shipping gate routes differently, the
    window would validate a gate that is not the one in the tree.
    """

    def test_the_points_agree_with_the_gate(self):
        runner = _load()
        from sglang.srt.model_executor.runner.prefill_graph_regime import (
            PREFILL_GRAPH_REGIME_ENV,
            regime_permits_graph,
        )

        prev = os.environ.get(PREFILL_GRAPH_REGIME_ENV)
        os.environ[PREFILL_GRAPH_REGIME_ENV] = "1"
        try:
            long_v = regime_permits_graph(
                batch_size=runner.POINT_LONG["concurrency"],
                num_tokens=runner.POINT_LONG["tokens"],
            )
            short_v = regime_permits_graph(
                batch_size=runner.POINT_SHORT["concurrency"],
                num_tokens=runner.POINT_SHORT["tokens"]
                * runner.POINT_SHORT["concurrency"],
            )
        finally:
            if prev is None:
                os.environ.pop(PREFILL_GRAPH_REGIME_ENV, None)
            else:
                os.environ[PREFILL_GRAPH_REGIME_ENV] = prev

        self.assertFalse(long_v.permits, "long point must be refused")
        self.assertTrue(short_v.permits, "short point must be permitted")


if __name__ == "__main__":
    unittest.main()
