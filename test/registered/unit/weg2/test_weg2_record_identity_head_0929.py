"""The calibration identity's ``line`` term names an unreadable tree HEAD.

29.09. (x27ra): a dry run on a git-archive tree (no .git) could not prove any
record's boot commit as an ancestor of the tree HEAD, so every 27B record fell
silently to CONSTANT and P was priced 1040 MiB low against the real boot z30j.
The identity now prints ``RECORD-IDENTITY UNRESOLVABLE`` for exactly that case.
"""

import inspect
import tempfile
import unittest
from unittest import mock

from sglang.srt.weg2 import form as F
from sglang.srt.weg2 import launcher as L
from sglang.test.test_utils import CustomTestCase


def _identity(repo: str) -> "F.CalibrationIdentity":
    return F.CalibrationIdentity(model="/m/Qwen3.8-27B-INT8", evidence_dir="/nonexistent",
                                 fields=("checkpoint", "form", "line"), repo=repo,
                                 line_heads=("103712cdb2ab",))


class HeadUnresolvable(CustomTestCase):
    def test_tree_without_git_is_named(self):
        with tempfile.TemporaryDirectory() as d:
            line = _identity(d).head_unresolvable_line()
        self.assertIsNotNone(line)
        self.assertTrue(line.startswith("RECORD-IDENTITY UNRESOLVABLE: tree "))
        self.assertIn("103712cdb2", line)
        self.assertIn("CONSTANT", line)

    def test_readable_head_says_nothing(self):
        with mock.patch.object(F, "_repo_head", return_value="bd1bc7a505"):
            self.assertIsNone(_identity("/some/tree").head_unresolvable_line())

    def test_without_line_term_says_nothing(self):
        ident = F.CalibrationIdentity(model="/m/x", evidence_dir="/nonexistent",
                                      fields=("checkpoint", "form"), repo="")
        self.assertIsNone(ident.head_unresolvable_line())

    def test_launcher_prints_it_beside_the_identity(self):
        src = inspect.getsource(L.main)
        self.assertIn("calib_identity.head_unresolvable_line()", src)


if __name__ == "__main__":
    unittest.main()
