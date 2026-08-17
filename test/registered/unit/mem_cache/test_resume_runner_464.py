# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The #464 measurement runner must still work when its window opens.

Same reason as the #398 gate runner: a turnkey script written ahead of the
window it serves is the desk-written-never-executed failure waiting to happen.
This pins the hermetic half so the window only has to buy the timing.

It also pins the two premise corrections the runner encodes, because both are
easy to "simplify" back out by someone reading only the ticket.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_RUNNER = os.path.join(_REPO, "bench", "464", "run_464_resume.py")


def _load():
    sys.path.insert(0, os.path.dirname(_RUNNER))
    try:
        import run_464_resume as runner
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


class TestTheCallCountModel(unittest.TestCase):
    """The count half is arithmetic and must stay exact."""

    def test_seam_default_chunk_is_257_calls_per_gib(self):
        runner = _load()
        off, on = runner.analytic_arms(1024 * runner.MIB, 8 * runner.MIB)
        self.assertEqual((off.extents, off.calls), (128, 257))
        self.assertEqual((on.extents, on.calls), (1, 3))

    def test_the_ticket_illustration_is_not_a_default(self):
        # 2 MiB is the granularity fallback; no default chunk sets it. Pinned so
        # the "~500 x 2 MiB" figure is not silently adopted as this path's shape.
        runner = _load()
        self.assertEqual(runner.KNOWN_CHUNKS_MIB["kv_seam (SGLANG_FLIP_SEAM_CHUNK_MIB default)"], 8)
        self.assertEqual(runner.KNOWN_CHUNKS_MIB["carriers (CARRIER_COMMIT_CHUNK)"], 64)


class TestTheBandIsContextNotAThreshold(unittest.TestCase):
    """Correction (1): 40-85 ms belongs to a different mechanism.

    It is the graph-state swap through torch_memory_saver
    (``adaptive_graph_memory.py:207-214``), not ``KvVmmArena.commit_range``.
    Using it as a pass threshold here would be judging this path by another
    path's number.
    """

    def test_the_report_labels_the_band_as_context(self):
        runner = _load()
        off, on = runner.analytic_arms(1024 * runner.MIB, 8 * runner.MIB)
        text = runner.Result(
            nbytes=1024 * runner.MIB, chunk_bytes=8 * runner.MIB, off=off, on=on
        ).render()
        self.assertIn("not a threshold", text)
        self.assertIn("torch_memory_saver", text)

    def test_no_saving_is_claimed_without_both_arms(self):
        runner = _load()
        self.assertEqual(runner.Result().calls_saved, 0)


if __name__ == "__main__":
    unittest.main()
