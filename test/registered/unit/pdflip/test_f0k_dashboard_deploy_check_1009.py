"""F0-K (RC1 acceptance 09.10.2026, M3): ``install.sh --check`` must accept the renamed integration head, and only for the right reason.

The acceptance found that the head was refused twice: ``<head> enthält das laufende Release 9d3f7bed11 nicht`` (27B) and ``kein Nachfahre der
Deploy-Linie desk/dashboard-ipc-0929`` (Flash-Next).  Porting the six dashboard commits (desk/dashboard-i18n-1008) puts their CONTENT into the tree,
but a commit of the old name is never an ancestor of a renamed tree.  The deploy check therefore reads ``deploy/CARRIED_FROM`` of the revision it is
asked about.  The cases run the REAL script against a throw-away repository:

* an old-name commit that is neither an ancestor nor listed is still refused (both rules);
* the same revision with the list is accepted;
* the list is read from the REVISION, not from the working directory or the environment;
* a listed commit that is an ancestor changes nothing.
"""
import os
import pathlib
import subprocess
import tempfile
import unittest

from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

ROOT = pathlib.Path(__file__).resolve().parents[4]
INSTALL = ROOT / "tools" / "rig_dashboard" / "rigdash" / "deploy" / "install.sh"
CARRIED = "tools/rig_dashboard/rigdash/deploy/CARRIED_FROM"
ENV = {**{k: v for k, v in os.environ.items() if not k.startswith("GIT_")}, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
       "GIT_COMMITTER_EMAIL": "t@x"}


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, env=ENV, check=True).stdout.strip()


class DeployCheck(CustomTestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="f0k-deploy-")
        self.addCleanup(self._td.cleanup)
        self.repo = pathlib.Path(self._td.name) / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        (self.repo / "a.txt").write_text("1")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "root")
        # the OLD line: two commits on a side branch (the deploy-line tip and the running release)
        git(self.repo, "checkout", "-q", "-b", "old")
        (self.repo / "a.txt").write_text("old1")
        git(self.repo, "commit", "-qam", "line tip")
        self.line_tip = git(self.repo, "rev-parse", "HEAD")
        (self.repo / "a.txt").write_text("old2")
        git(self.repo, "commit", "-qam", "running release")
        self.release = git(self.repo, "rev-parse", "HEAD")
        git(self.repo, "checkout", "-q", "main")
        git(self.repo, "branch", "-q", "line", self.line_tip)          # RIGDASH_DEPLOY_LINE target
        self.link = pathlib.Path(self._td.name) / "current"
        self.link.symlink_to("releases/" + self.release[:10])

    def _renamed(self, carried_lines):
        """a revision on main that CARRIES the old content (rewritten, no ancestry) and lists `carried_lines`"""
        git(self.repo, "checkout", "-q", "main")
        (self.repo / "a.txt").write_text("renamed content")
        if carried_lines is not None:
            p = self.repo / CARRIED
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# comment line\n" + "\n".join(carried_lines) + "\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "renamed head")
        return git(self.repo, "rev-parse", "HEAD")

    def _check(self, rev):
        env = {**ENV, "RIGDASH_REPO": str(self.repo), "RIGDASH_DEPLOY_LINE": "line", "RIGDASH_CURRENT_LINK": str(self.link)}
        return subprocess.run(["bash", str(INSTALL), "--check", rev], capture_output=True, text=True, env=env)

    def test_an_unlisted_renamed_head_is_refused_on_the_line_rule(self):
        p = self._check(self._renamed(None))
        self.assertEqual(p.returncode, 3, p.stdout + p.stderr)
        self.assertIn("kein Nachfahre der Deploy-Linie", p.stderr)

    def test_listing_the_line_tip_alone_moves_the_refusal_to_the_release_rule(self):
        p = self._check(self._renamed([self.line_tip[:10] + "  tip of the old deploy line"]))
        self.assertEqual(p.returncode, 3, p.stdout + p.stderr)
        self.assertIn("enthält das laufende Release", p.stderr)

    def test_listing_both_is_accepted(self):
        p = self._check(self._renamed([self.line_tip + "  line", self.release[:10] + "  release"]))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("deploy-linie ok", p.stdout)

    def test_a_wrong_hash_does_not_count(self):
        p = self._check(self._renamed([self.line_tip[:10], "0123456789  not it"]))
        self.assertEqual(p.returncode, 3, p.stdout + p.stderr)

    def test_the_list_is_read_from_the_revision_not_from_the_work_tree(self):
        rev = self._renamed(None)                                 # head WITHOUT the list ...
        p = self.repo / CARRIED
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("%s\n%s\n" % (self.line_tip, self.release))      # ... a list lying around uncommitted
        self.assertEqual(self._check(rev).returncode, 3)

    def test_an_ancestor_needs_no_list(self):
        git(self.repo, "checkout", "-q", "-b", "fast", self.release)
        (self.repo / "b.txt").write_text("x")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "on top of the old line")
        p = self._check(git(self.repo, "rev-parse", "HEAD"))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_the_shipped_list_names_exactly_what_the_acceptance_missed(self):
        txt = (ROOT / CARRIED).read_text(encoding="utf-8")
        heads = [ln.split()[0] for ln in txt.splitlines() if ln.strip() and not ln.startswith("#")]
        self.assertEqual(sorted(heads), ["9d3f7bed11", "d983311745"])


if __name__ == "__main__":
    unittest.main()
