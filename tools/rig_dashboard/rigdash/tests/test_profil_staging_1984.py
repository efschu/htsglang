"""Auftrag 1984 (B): ``deploy/stage_profil_modules.sh`` und ``--profil-tree``.

Das Skript wird gegen ein WEGWERF-Repo in einem Temp-Verzeichnis gefahren (RIGDASH_REPO, --root): nie gegen /opt/rigdash.  Gepinnt:

* --check (Voreinstellung) und --dry-run schreiben NICHTS (Ablage-Wurzel bleibt leer), nennen den Stand und --dry-run die Aktionen;
* eine Revision ohne die Editor-Module (die Dashboard-Linie) wird mit Exit 3 verweigert und nennt jede fehlende Datei;
* --apply stagt den vollen Baum, schaltet ``profil/current`` atomar, ist idempotent (zweiter Lauf = No-op, current wird nicht neu gesetzt),
  ein Wechsel auf eine neue Revision laesst die alte liegen und schaltet nur den Zeiger um;
* ohne Argument/mit relativem --root Aufruffehler (Exit 2), --unit-flags druckt die Unit-Zeilen und schreibt nichts;
* der Baum, den das Skript ablegt, genuegt den Findern des Dashboards (profil.find_tree, hwprofil._find_tree, modellprofil.find_tree);
* ``--profil-tree`` laeuft in App ein: Editor, Modellprofil, Hardwareprofil und Kopplungs-Worker nehmen denselben Baum.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import hwprofil, modellprofil, profil  # noqa: E402
from rigdash import server as S  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(HERE), "deploy", "stage_profil_modules.sh")
REQUIRED = [
    "pdflip/profile_json.py", "pdflip/refusals.py", "pdflip/profile_catalog.py", "pdflip/profile_catalog_curated.py", "pdflip/model_profile.py",
    "pdflip/card_identity.py", "pdflip/topology.py", "rigmon/hardware_profile.py", "planner/profile_couplings.py", "planner/expert_residency.py",
    "planner/pp_cut.py"]
ROOTS = ["python/flliper/__init__.py", "python/flliper/srt/planner/__init__.py", "python/flliper/srt/pdflip/__init__.py", "python/flliper/srt/rigmon/__init__.py"]   # srt/ ohne __init__ wie im echten Repo


def git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def write(repo, rel, text="x = 1\n"):
    p = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        fh.write(text)


class Staging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pf1984s_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = os.path.join(self.tmp, "repo")
        self.root = os.path.join(self.tmp, "stage")
        os.makedirs(self.repo)
        git(self.repo, "init", "-q")
        write(self.repo, "python/flliper/srt/other.py")           # Dashboard-Linie: kein Editor-Modul
        for r in ROOTS:
            write(self.repo, r, "")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "dash")
        self.dash = git(self.repo, "rev-parse", "--short=10", "HEAD")
        for f in REQUIRED:
            write(self.repo, "python/flliper/srt/" + f, "# %s\n" % f)
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "py")
        self.py1 = git(self.repo, "rev-parse", "--short=10", "HEAD")
        write(self.repo, "python/flliper/srt/pdflip/model_profile.py", "# neu\n")
        git(self.repo, "commit", "-q", "-am", "py2")
        self.py2 = git(self.repo, "rev-parse", "--short=10", "HEAD")

    def run_script(self, *args, root=True):
        cmd = [SCRIPT, *args]
        if root:
            cmd[1:1] = ["--root", self.root]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=dict(os.environ, RIGDASH_REPO=self.repo))

    def staged_files(self):
        return [os.path.join(dp, f) for dp, _d, fs in os.walk(self.root) for f in fs] if os.path.isdir(self.root) else []

    def test_check_is_the_default_and_writes_nothing(self):
        for args in ([self.py1], ["--check", self.py1], ["--dry-run", self.py1]):
            r = self.run_script(*args)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("ok: %s traegt alle" % self.py1, r.stdout)
            self.assertIn("gestagt=nein", r.stdout)
            self.assertIn("wuerde stagen=ja current-umschalten=ja", r.stdout)
            self.assertFalse(os.path.exists(self.root), "kein Schreiben ohne --apply: %s" % args)

    def test_dry_run_names_the_actions(self):
        r = self.run_script("--dry-run", self.py1)
        self.assertIn("DRY-RUN", r.stdout)
        self.assertIn("git -C %s archive %s python/flliper" % (self.repo, self.py1), r.stdout)
        self.assertIn("ln -sfn releases/%s" % self.py1, r.stdout)
        self.assertFalse(os.path.exists(self.root))

    def test_a_dashboard_revision_is_refused_and_every_missing_file_is_named(self):
        for mode in ("--check", "--dry-run", "--apply"):
            r = self.run_script(mode, self.dash)
            self.assertEqual(r.returncode, 3, (mode, r.stdout, r.stderr))
            for f in REQUIRED:
                self.assertIn("python/flliper/srt/" + f, r.stderr, f)
            self.assertIn("keine Python-Release-Revision", r.stderr)
            self.assertFalse(os.path.exists(self.root), "auch die Verweigerung schreibt nichts (%s)" % mode)

    def test_unknown_revision_and_usage_errors(self):
        self.assertEqual(self.run_script("--check", "deadbeef0000").returncode, 3)
        self.assertEqual(self.run_script().returncode, 2)
        r = subprocess.run([SCRIPT, "--root", "relativ", self.py1], capture_output=True, text=True, env=dict(os.environ, RIGDASH_REPO=self.repo))
        self.assertEqual(r.returncode, 2)
        self.assertEqual(self.run_script("--bogus", self.py1).returncode, 2)

    def test_apply_stages_the_full_tree_and_is_idempotent(self):
        r = self.run_script("--apply", self.py1)
        self.assertEqual(r.returncode, 0, r.stderr)
        base = os.path.join(self.root, "profil", "releases", self.py1)
        for f in REQUIRED:
            self.assertTrue(os.path.isfile(os.path.join(base, "python", "flliper", "srt", f)), f)
        self.assertTrue(os.path.isfile(os.path.join(base, "python", "flliper", "srt", "other.py")))   # der VOLLE Baum
        self.assertEqual(open(os.path.join(base, "REV")).read().strip(), self.py1)
        self.assertEqual(os.readlink(os.path.join(self.root, "profil", "current")), "releases/%s" % self.py1)
        self.assertFalse([n for n in os.listdir(os.path.join(self.root, "profil")) if n.startswith(".stage")])
        marker = os.stat(os.path.join(base, "STAGED_OK")).st_mtime_ns
        link_ino = os.lstat(os.path.join(self.root, "profil", "current")).st_ino
        again = self.run_script("--apply", self.py1)
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("aktuell (nichts zu tun", again.stdout)
        self.assertNotIn("gestagt:", again.stdout)
        self.assertNotIn("current ->", again.stdout)
        self.assertEqual(os.stat(os.path.join(base, "STAGED_OK")).st_mtime_ns, marker)
        self.assertEqual(os.lstat(os.path.join(self.root, "profil", "current")).st_ino, link_ino)
        chk = self.run_script("--check", self.py1)
        self.assertIn("gestagt=ja", chk.stdout)
        self.assertIn("aktuell", chk.stdout)

    def test_a_new_revision_keeps_the_old_one_and_only_moves_the_pointer(self):
        self.run_script("--apply", self.py1)
        r = self.run_script("--apply", self.py2)
        self.assertEqual(r.returncode, 0, r.stderr)
        rel = os.path.join(self.root, "profil", "releases")
        self.assertEqual(sorted(os.listdir(rel)), sorted([self.py1, self.py2]))
        self.assertEqual(os.readlink(os.path.join(self.root, "profil", "current")), "releases/%s" % self.py2)
        self.assertEqual(open(os.path.join(rel, self.py2, "python/flliper/srt/pdflip/model_profile.py")).read(), "# neu\n")
        self.assertEqual(open(os.path.join(rel, self.py1, "python/flliper/srt/pdflip/model_profile.py")).read(), "# pdflip/model_profile.py\n")
        back = self.run_script("--apply", self.py1)               # zurueck: nur der Zeiger, nichts neu ausgepackt
        self.assertNotIn("gestagt:", back.stdout)
        self.assertEqual(os.readlink(os.path.join(self.root, "profil", "current")), "releases/%s" % self.py1)

    def test_unit_flags_print_and_write_nothing(self):
        r = self.run_script("--unit-flags")
        self.assertEqual(r.returncode, 0, r.stderr)
        for needle in ("RIGDASH_PROFIL_TREE=%s/profil/current/python" % self.root, "RIGDASH_COUPLINGS_PYTHON=", "MemoryMax=2G", "--couplings-python",
                       "--hw-tree", "--hw-measure-tree", "--hw-python", "--hw-prefix", "--profile-dir", "--profiles-release-dir", "--model-root",
                       "--edition release", "--profil-tree"):
            self.assertIn(needle, r.stdout, needle)
        self.assertFalse(os.path.exists(self.root))

    def test_every_unit_flag_the_script_names_exists_in_the_server(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
            S.main(["--help"])
        for flag in ("--profil-tree", "--couplings-python", "--hw-tree", "--hw-measure-tree", "--hw-python", "--hw-prefix", "--profile-dir",
                     "--profiles-release-dir", "--model-root", "--edition"):
            self.assertIn(flag, buf.getvalue(), flag)

    def test_the_staged_tree_satisfies_the_dashboard_finders(self):
        # Findet der Dashboard-Prozess seine Dateien im gestagten Baum?  Die Test-Dateien sind Platzhalter, es zaehlt die Fundstelle.
        self.run_script("--apply", self.py1)
        tree = os.path.join(self.root, "profil", "current", "python")
        self.assertEqual(profil.find_tree(tree), tree)
        self.assertEqual(hwprofil._find_tree(tree), tree)
        self.assertEqual(modellprofil.find_tree(tree), tree)


class ProfileTreeFlag(unittest.TestCase):
    def test_the_app_hands_one_tree_to_editor_model_hardware_and_worker(self):
        with tempfile.TemporaryDirectory() as d:
            tree = os.path.join(d, "python")
            for rel in ("flliper/srt/pdflip/profile_json.py", "flliper/srt/pdflip/refusals.py", "flliper/srt/pdflip/model_profile.py",
                        "flliper/srt/rigmon/hardware_profile.py"):
                write(tree, rel)
            ns = argparse.Namespace(log_glob=[], docker_ssh="", docker_host_prefix="", front=[], gpuq="", state_dir="", release_profile=[],
                                    image_changes=os.path.join(d, "ic.json"), features=os.path.join(d, "f.json"), features_repo=d,
                                    edition="release", profile_tree=tree, hw_tree=None)
            app = S.App(ns)
            self.assertEqual(app.profil.tree, tree)
            self.assertEqual(app.modellprofil.tree, tree)
            self.assertEqual(app.hwprofil.tree, tree)
            self.assertEqual(app.couplings.tree_python, tree)

    def test_hw_tree_overrides_only_the_hardware_profile(self):
        with tempfile.TemporaryDirectory() as d:
            tree, hw = os.path.join(d, "python"), os.path.join(d, "hw")
            for rel in ("flliper/srt/pdflip/profile_json.py", "flliper/srt/pdflip/refusals.py"):
                write(tree, rel)
            write(hw, "flliper/srt/rigmon/hardware_profile.py")
            ns = argparse.Namespace(log_glob=[], docker_ssh="", docker_host_prefix="", front=[], gpuq="", state_dir="", release_profile=[],
                                    image_changes=os.path.join(d, "ic.json"), features=os.path.join(d, "f.json"), features_repo=d,
                                    edition="rig", profile_tree=tree, hw_tree=hw)
            app = S.App(ns)
            self.assertEqual(app.profil.tree, tree)
            self.assertEqual(app.hwprofil.tree, hw)


if __name__ == "__main__":
    unittest.main()
