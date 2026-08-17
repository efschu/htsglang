# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""A sweep result must carry the tree it was taken against (#349).

WHY THIS EXISTS. The boot matrix is a STANDING bug net, and the only question
ever asked of a standing net is "is it still green at today's HEAD". Before
this, a run wrote ``summary.json`` (verdicts) and nothing else: the artifacts
recorded WHAT passed but not WHAT TREE it passed on. Reconstructing that meant
reading ``git log`` and directory mtimes and guessing -- which is exactly what
the #349 determination had to do to establish that the last real sweep (the
2026-08-01 artifact set) predated its own arm roster by two commits.

A green with no tree identity cannot be distinguished from a stale green. That
is the same disarmed-while-looking-armed class the net itself exists to catch,
so it gets pinned rather than left to convention.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest

from sglang.srt.boot_matrix import sweep


class TestProvenanceIsCollected(unittest.TestCase):
    def test_it_reports_the_current_head_inside_a_repo(self):
        prov = sweep.collect_provenance()
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(sweep.__file__)),
            capture_output=True,
            text=True,
        )
        self.assertEqual(prov["git_head"], head.stdout.strip())
        self.assertEqual(len(prov["git_head"]), 40)

    def test_it_records_which_arms_the_roster_held(self):
        # The 08-01 artifact set holds 17 arm dirs while today's roster is 20.
        # Recording the roster is what makes that difference readable from the
        # artifacts instead of inferred from a commit message.
        prov = sweep.collect_provenance()
        self.assertEqual(prov["arm_count"], len(prov["arm_names"]))
        self.assertIn("A_default", prov["arm_names"])

    def test_it_reports_a_dirty_tree_as_dirty(self):
        prov = sweep.collect_provenance()
        # A boolean either way, never absent: "unknown" dirtiness on a standing
        # net is the same defect as an unknown head.
        self.assertIsInstance(prov["git_dirty"], bool)

    def test_it_timestamps_in_utc(self):
        prov = sweep.collect_provenance()
        self.assertTrue(prov["collected_utc"].endswith("Z"))


class TestProvenanceNeverBreaksTheSweep(unittest.TestCase):
    def test_it_degrades_to_none_outside_a_repo(self):
        # Wheel install / exported tree: no .git. Provenance is unavailable,
        # which must be RECORDED as unavailable, never raised -- an hour of
        # card time must not be lost to a missing git binary.
        with tempfile.TemporaryDirectory() as tmp:
            prov = sweep.collect_provenance(repo_dir=tmp)
        self.assertIsNone(prov["git_head"])
        self.assertIn("arm_count", prov)


class TestProvenanceIsWritten(unittest.TestCase):
    def test_it_lands_beside_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            sweep.write_provenance(tmp)
            path = os.path.join(tmp, "provenance.json")
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                loaded = json.load(f)
        self.assertIn("git_head", loaded)
        self.assertIn("arm_names", loaded)

    def test_the_sweep_writes_it_on_the_run_path(self):
        # Pin the CALL, not just the helper: a provenance writer nothing calls
        # is the always-absent-marker defect this determination was sent to
        # find, wearing a different hat.
        import inspect

        src = inspect.getsource(sweep._main)
        self.assertIn("write_provenance", src)


if __name__ == "__main__":
    unittest.main()
