"""F0-I fix round 1: the catalog's cited source lines are only true if the cited files did not change since the build revision.

The checker (tools/release/catalog_rev_check_1008.py) is tested on a throw-away git repository (synthetic catalog); against the real tree it
runs only when FLLIPER_CATALOG_REV_CHECK=1 (release gate of F0-I; any later edit of launcher.py makes it red on purpose).
"""
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
_spec = importlib.util.spec_from_file_location("catalog_rev_check_1008", os.path.join(ROOT, "tools", "release", "catalog_rev_check_1008.py"))
CK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CK)


def git(d, *a):
    return subprocess.run(["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True, text=True, check=True).stdout.strip()


class Checker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        os.makedirs(os.path.join(self.d, "python", "flliper", "srt", "pdflip"))
        self.f = os.path.join(self.d, "python", "flliper", "srt", "pdflip", "launcher.py")
        with open(self.f, "w") as fh:
            fh.write("A = 1\nB = 2\n")
        git(self.d, "init", "-q")
        git(self.d, "add", "-A")
        git(self.d, "commit", "-q", "-m", "one")
        self.rev = git(self.d, "rev-parse", "HEAD")

    def tearDown(self):
        self.tmp.cleanup()

    def catalog(self, tree_rev):
        p = os.path.join(self.d, "cat.json")
        cat = {"tree_rev": tree_rev, "entries": {"X": {"source": {"file": "pdflip/launcher.py", "line": 2}, "lesestellen": ["pdflip/launcher.py:2"]}}}
        with open(p, "w") as fh:
            json.dump(cat, fh)
        return p

    def test_unchanged_file_is_no_drift(self):
        res, drift = CK.check(self.d, self.catalog("27b=%s+nf=0000000000" % self.rev))
        self.assertEqual(drift, [])
        self.assertEqual([(l, s) for l, _r, s, _f in res], [("27b", "checked"), ("nf", "unknown-rev")])

    def test_edit_of_a_cited_file_after_the_build_rev_is_drift(self):
        with open(self.f, "w") as fh:
            fh.write("# three lines\n# more\nA = 1\nB = 2\n")
        git(self.d, "commit", "-q", "-am", "two")
        res, drift = CK.check(self.d, self.catalog("27b=%s" % self.rev))
        self.assertEqual(drift, [("27b", self.rev, "python/flliper/srt/pdflip/launcher.py")])

    def test_edit_of_an_uncited_file_is_no_drift(self):
        other = os.path.join(self.d, "python", "flliper", "srt", "other.py")
        with open(other, "w") as fh:
            fh.write("x = 1\n")
        git(self.d, "add", "-A")
        git(self.d, "commit", "-q", "-m", "two")
        res, drift = CK.check(self.d, self.catalog("27b=%s" % self.rev))
        self.assertEqual(drift, [])

    def test_a_revision_that_is_not_an_ancestor_is_not_checked(self):
        git(self.d, "checkout", "-q", "-b", "side")
        with open(self.f, "w") as fh:
            fh.write("A = 9\n")
        git(self.d, "commit", "-q", "-am", "side")
        side = git(self.d, "rev-parse", "HEAD")
        git(self.d, "checkout", "-q", "-")
        res, drift = CK.check(self.d, self.catalog("27b=%s+nf=%s" % (self.rev, side)))
        self.assertEqual({l: s for l, _r, s, _f in res}, {"27b": "checked", "nf": "not-ancestor"})
        self.assertEqual(drift, [])


@unittest.skipUnless(os.environ.get("FLLIPER_CATALOG_REV_CHECK") == "1", "release gate: set FLLIPER_CATALOG_REV_CHECK=1")
class ShippedCatalog(unittest.TestCase):
    def test_no_cited_file_changed_since_the_build_revision(self):
        res, drift = CK.check(ROOT)
        self.assertTrue(any(s == "checked" for _l, _r, s, _f in res), res)
        self.assertEqual(drift, [])


if __name__ == "__main__":
    unittest.main()
