# SPDX-License-Identifier: Apache-2.0
"""Expert maps go to a writable directory, by name (29.09.).

A dry run against the container's evidence tree (owned by the container's user)
logged 'WEG2-EXPERT-MAP failed: PermissionError' and published no map. The
launcher now writes the map where it can and says where.
"""
import os
import stat
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class ExpertMapDir(CustomTestCase):
    def test_writable_evidence_dir_is_kept_silently(self):
        lines = []
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(launcher.expert_map_dir(d, lines.append), d)
        self.assertEqual(lines, [])

    def test_unwritable_evidence_dir_is_redirected_by_name(self):
        lines = []
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as alt:
            ev = os.path.join(d, "evidence")
            os.makedirs(ev)
            with mock.patch.dict(os.environ, {launcher.EXPERT_MAP_DIR_ENV: alt}), \
                    mock.patch.object(launcher.os, "access", lambda p, m: p != ev):
                got = launcher.expert_map_dir(ev, lines.append)
            self.assertEqual(got, alt)
        self.assertEqual(len(lines), 1)
        self.assertIn("WEG2-EXPERT-MAP-DIR", lines[0])
        self.assertIn(alt, lines[0])

    def test_both_writers_use_it(self):
        import inspect

        self.assertIn("expert_map_dir(", inspect.getsource(launcher.publish_expert_map))
        self.assertIn("expert_map_dir(", inspect.getsource(launcher.main))


if __name__ == "__main__":
    unittest.main()
