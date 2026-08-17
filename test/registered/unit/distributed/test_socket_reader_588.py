# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The #588 socket reader must keep parsing the instrument's real output.

The #252 per-rank prefill instrument is live by default on any CUDA boot, so
this reader is the only thing between that output and a verdict. If the log
format drifts the reader goes silently empty, which would read as "no socket"
rather than as "cannot parse" -- so the format is pinned against a VERBATIM
line from the window-8 record rather than against a synthetic one.

The floor-vs-skew classification is pinned too. It decides which lever class
can move a term at all -- overlap for a floor, balancing for skew -- and
getting it backwards is how a balancing lever came to be tried against a term
that was never skew.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_RUNNER = os.path.join(_REPO, "bench", "588", "run_588_socket.py")

#: Verbatim from /spinning/gpu-battery-results/2026-08-05_window8/
#: boot_A2_588_coverage.log -- do not tidy, the point is that it is real.
_TP0 = (
    "[2026-08-05 16:03:38 TP0] Prefill rank batch, #new-token: 2048, "
    "#cached-token: 0, #chunks: 1, gpu-ms: 1905.7 (compute 354.7, wait "
    "1551.0) (wait by family: tp.all_reduce 932.2/129x, dcp.all_gather "
    "366.2/48x, dcp.all_reduce 250.4/16x, tp.all_gather 0)"
)
_TP1 = (
    "[2026-08-05 16:04:01 TP1] Prefill rank batch, #new-token: 2004, "
    "#cached-token: 0, #chunks: 1, gpu-ms: 2185.8 (compute 857.4, wait "
    "1328.4) (wait by family: tp.all_reduce 932.2/129x, dcp.all_reduce "
    "263.3/16x, dcp.all_gather 132.7/48x, tp.all_gather 0)"
)


def _load():
    sys.path.insert(0, os.path.dirname(_RUNNER))
    try:
        import run_588_socket as reader
    finally:
        sys.path.pop(0)
    return reader


class TestSelfTestIsGreen(unittest.TestCase):
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


class TestTheRealFormatStillParses(unittest.TestCase):
    def test_a_verbatim_line_parses(self):
        s = _load().parse_line(_TP0)
        self.assertIsNotNone(s, "the instrument's real line no longer parses")
        self.assertEqual(s.rank, 0)
        self.assertEqual(s.compute_ms, 354.7)
        self.assertEqual(s.wait_ms, 1551.0)

    def test_families_carry_their_call_counts(self):
        s = _load().parse_line(_TP0)
        self.assertEqual(s.families["tp.all_reduce"]["calls"], 129)

    def test_a_zero_family_without_a_count_still_parses(self):
        s = _load().parse_line(_TP0)
        self.assertEqual(s.families["tp.all_gather"]["ms"], 0.0)

    def test_an_unparseable_line_is_none_not_a_crash(self):
        self.assertIsNone(_load().parse_line("Prefill rank batch, garbage"))


class TestFloorVersusSkew(unittest.TestCase):
    def test_the_all_reduce_is_a_floor(self):
        reader = _load()
        floors = reader.floor_families(
            [reader.parse_line(_TP0), reader.parse_line(_TP1)]
        )
        self.assertTrue(floors["tp.all_reduce"])

    def test_the_all_gather_is_skew(self):
        reader = _load()
        floors = reader.floor_families(
            [reader.parse_line(_TP0), reader.parse_line(_TP1)]
        )
        self.assertFalse(floors["dcp.all_gather"])

    def test_one_rank_cannot_establish_a_floor(self):
        reader = _load()
        self.assertEqual(reader.floor_families([reader.parse_line(_TP0)]), {})


class TestTheRatioIsNotAveraged(unittest.TestCase):
    """Per rank, never pooled: the ranks genuinely disagree."""

    def test_the_two_ranks_disagree_sharply(self):
        reader = _load()
        a = reader.parse_line(_TP0).wait_over_compute
        b = reader.parse_line(_TP1).wait_over_compute
        self.assertGreater(a / b, 2.0, "the asymmetry an average would hide")


if __name__ == "__main__":
    unittest.main()
