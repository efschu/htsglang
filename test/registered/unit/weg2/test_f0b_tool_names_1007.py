"""F0-B (rename R5): the shell tools that ship WITH the tree read the old AND the renamed name.

Two of them live in the tree and are baked into the image:

* ``scripts/weg2/devtools/boot_deadman.sh`` -- tier 4 (PROGRESS-STALL) is armed by the launcher through the bare env names
  ``WEG2_DEADMAN_GROUP / WEG2_STATE_DIR / WEG2_STATE_FILE_PY / WEG2_PY`` (the rename tool turns them into ``PDFLIP_*``).  A deadman
  that knew one spelling would be silently OFF behind a launcher of the other generation.
* ``docker/htsglang-entrypoint.sh`` -- the August server/planner wrapper starts ``python -m <package>.launch_server``; the release
  entrypoint hands it either generation of the code stand (``PYTHONPATH=<stand>/python``).

Each case runs the REAL text of the tool (functions cut out of the file, nothing re-implemented) with the old and the renamed
spelling and requires the same answer.  The old token is written split (the mechanical rename must leave the readers alone):
``test_readers_survive_the_rename`` runs the files through the rename tool when it is on this box.
"""

import importlib.util
import os
import pathlib
import re
import subprocess
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

ROOT = pathlib.Path(__file__).resolve().parents[4]
DEADMAN = ROOT / "scripts" / "weg2" / "devtools" / "boot_deadman.sh"
ENTRY = ROOT / "docker" / "htsglang-entrypoint.sh"
OLD_P, NEW_P = "WE" "G2_", "PDFLIP_"


def _bash(script, env=None, timeout=60):
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, p.stdout, p.stderr


def _clean_env(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith((OLD_P, NEW_P))}
    env.update(extra)
    return env


class TestDeadmanTier4Names(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        text = DEADMAN.read_text()
        a, b = text.index("dm_env() {"), text.index("progress_check() {")
        cls.funcs = text[a:b]          # dm_env + the four readers + progress_on, exactly as shipped
        cls.tmp = tempfile.TemporaryDirectory()
        cls.sd = os.path.join(cls.tmp.name, "state")
        os.makedirs(cls.sd)
        open(os.path.join(cls.sd, "state.json"), "w").write("{}")
        cls.py = os.path.join(cls.tmp.name, "state_file.py")
        open(cls.py, "w").write("print('x')\n")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _armed(self, prefix, **over):
        env = {prefix + "DEADMAN_GROUP": "front", prefix + "STATE_DIR": self.sd, prefix + "STATE_FILE_PY": self.py}
        env.update(over)
        rc, out, err = _bash("PROGRESS_STALL_S=60; %s progress_on; echo rc=$?" % self.funcs, env=_clean_env(env))
        return out.strip(), err

    def test_syntax(self):
        self.assertEqual(_bash("bash -n %s" % DEADMAN)[0], 0)

    def test_old_names_arm_tier_4(self):
        self.assertEqual(self._armed(OLD_P)[0], "rc=0")

    def test_renamed_names_arm_tier_4(self):
        self.assertEqual(self._armed(NEW_P)[0], "rc=0")

    def test_no_name_leaves_tier_4_off(self):
        rc, out, _ = _bash("PROGRESS_STALL_S=60; %s progress_on; echo rc=$?" % self.funcs, env=_clean_env({}))
        self.assertEqual(out.strip(), "rc=1")

    def test_a_group_other_than_front_stays_off_in_both_spellings(self):
        self.assertEqual(self._armed(OLD_P, **{OLD_P + "DEADMAN_GROUP": "P"})[0], "rc=1")
        self.assertEqual(self._armed(NEW_P, **{NEW_P + "DEADMAN_GROUP": "D"})[0], "rc=1")

    def test_renamed_name_wins_when_both_are_set(self):
        env = {OLD_P + "PY": "oldpy", NEW_P + "PY": "newpy"}
        rc, out, _ = _bash("%s\ndm_py" % self.funcs, env=_clean_env(env))
        self.assertEqual(out, "newpy")
        rc, out, _ = _bash("%s\ndm_py" % self.funcs, env=_clean_env({OLD_P + "PY": "oldpy"}))
        self.assertEqual(out, "oldpy")
        rc, out, _ = _bash("%s\ndm_py" % self.funcs, env=_clean_env({}))
        self.assertEqual(out, "python3")

    def test_writer_pid_reaches_both_spellings(self):
        text = DEADMAN.read_text()
        m = re.search(r"line=\$\((\S+_WRITER_PID=\$\$ \S+_WRITER_PID=\$\$) ", text)
        self.assertIsNotNone(m, "progress_check must hand the writer pid to the state file in both spellings")


class TestEntrypointPackageModule(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        text = ENTRY.read_text()
        a = text.index("# F0B-PKGMOD-BEGIN")
        b = text.index("# F0B-PKGMOD-END")
        cls.block = text[a:b]
        cls.tmp = tempfile.TemporaryDirectory()
        cls.new_tree = os.path.join(cls.tmp.name, "new", "python")
        cls.old_tree = os.path.join(cls.tmp.name, "old", "python")
        os.makedirs(os.path.join(cls.new_tree, "flliper"))
        open(os.path.join(cls.new_tree, "flliper", "__init__.py"), "w").write("")
        os.makedirs(os.path.join(cls.old_tree, "sg" "lang"))
        open(os.path.join(cls.old_tree, "sg" "lang", "__init__.py"), "w").write("")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _pkgmod(self, pythonpath):
        env = {"PATH": os.environ.get("PATH", ""), "F0B_SITE_DIRS": "/nonexistent-site"}
        if pythonpath is not None:
            env["PYTHONPATH"] = pythonpath
        rc, out, err = _bash("%s\necho $PKGMOD" % self.block, env=env)
        self.assertEqual(rc, 0, err)
        return out.strip()

    def test_renamed_stand_starts_the_renamed_module(self):
        self.assertEqual(self._pkgmod(self.new_tree), "flliper")

    def test_old_stand_starts_the_old_module(self):
        self.assertEqual(self._pkgmod(self.old_tree), "sg" "lang")

    def test_no_stand_on_the_path_keeps_the_old_module(self):
        self.assertEqual(self._pkgmod(None), "sg" "lang")

    def test_renamed_stand_found_later_on_a_path_list(self):
        self.assertEqual(self._pkgmod(self.old_tree + ":" + self.new_tree), "flliper")

    def test_both_launch_lines_use_the_detected_module(self):
        text = ENTRY.read_text()
        self.assertIn('args=(python3 -m "$PKGMOD.launch_server")', text)
        self.assertIn('python3 -m "$PKGMOD.planner" --serve', text)
        self.assertEqual(_bash("bash -n %s" % ENTRY)[0], 0)


def _rename_tool():
    p = "/spinning/flliper/tools/rename_to_flliper.py"
    if not os.path.exists(p):
        return None
    spec = importlib.util.spec_from_file_location("_rename_to_flliper_f0b", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRenameSurvival(CustomTestCase):
    def test_readers_survive_the_rename(self):
        """After the mechanical rename (all rule sets on) the old-name readers still exist: the split spelling is not touched."""
        tool = _rename_tool()
        if tool is None:
            self.skipTest("rename tool not on this box")
        dm = DEADMAN.read_text()
        out, _rep, _skip = tool.rewrite_all(dm, False, True, {})
        self.assertIn('old=WE""G2_$1', out)
        ep = ENTRY.read_text()
        out2, _rep2, _skip2 = tool.rewrite_all(ep, False, True, {})
        self.assertIn('PKGMOD=sg""lang', out2)
        self.assertIn('flliper/__init__.py', out2)


if __name__ == "__main__":
    unittest.main()
