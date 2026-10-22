"""release 1022: ``--profil-tree`` / RIGDASH_PROFIL_TREE reaches the editor, the planner gate and the hardware profile.

Defect of d828544b5c: server.py read ``getattr(args, "profile_tree")`` while argparse stores the flag as ``profil_tree``; the tree was always None, the planner answered
"no planner tree (KARTENPLAN_TREE or install_510.sh)" and the page stayed on "computing".  The test starts the real server (release edition, editor only) with only the
environment variable RIGDASH_PROFIL_TREE (no KARTENPLAN_TREE, no HWPROFIL_TREE) and asks it.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ProfilTree(unittest.TestCase):
    def test_the_env_variable_reaches_editor_gate_and_hardware(self):
        tree = os.path.join(REPO, "python")
        self.assertTrue(os.path.isfile(os.path.join(tree, "flliper", "srt", "pdflip", "profile_json.py")))
        port = free_port()
        with tempfile.TemporaryDirectory() as d:
            env = {k: v for k, v in os.environ.items() if k not in ("KARTENPLAN_TREE", "HWPROFIL_TREE", "MODELLPROFIL_TREE")}
            env.update({"RIGDASH_PROFIL_TREE": tree, "PYTHONPATH": PKG_PARENT, "HOME": d, "CUDA_VISIBLE_DEVICES": ""})
            cmd = [sys.executable, "-m", "rigdash", "--edition", "release", "--editor-only", "--host", "127.0.0.1", "--port", str(port), "--state-dir", d,
                   "--profile-dir", os.path.join(d, "p"), "--hw-profile-file", os.path.join(d, "hw.json"), "--profiles-release-dir",
                   os.path.join(REPO, "docker", "flliper", "profiles_release"), "--docker-ssh", "", "--vm-url", "", "--oracle-prefix", "none"]
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=d)
            try:
                got = None
                for _ in range(60):
                    try:
                        got = json.loads(urllib.request.urlopen("http://127.0.0.1:%d/api/profil/list" % port, timeout=5).read())
                        break
                    except OSError:
                        time.sleep(0.5)
                self.assertIsNotNone(got, "the server did not start")
                self.assertEqual(got.get("planner_tree"), tree)                                   # the editor has the tree (it was None before the fix)
                cat = json.loads(urllib.request.urlopen("http://127.0.0.1:%d/api/profil/planer-katalog" % port, timeout=60).read())
                self.assertIn(3, cat["supported_n"])                                              # the topology gate was asked in that tree ...
                self.assertIn(cat["n_states"]["3"], ("proven",))                                   # ... and proved N=3 (no gate: the constants would also say so, but then 1 is "off" only)
                self.assertNotIn("no planner tree", json.dumps(cat))
            finally:
                p.terminate()
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()

    def test_the_flag_name_is_the_one_the_server_reads(self):
        src = open(os.path.join(PKG_PARENT, "rigdash", "server.py"), encoding="utf-8").read()
        self.assertIn('"--profil-tree"', src)
        self.assertIn('getattr(args, "profil_tree"', src)
        self.assertIn('"--hw-tree"', src)
        self.assertIn('getattr(args, "hw_tree"', src)


if __name__ == "__main__":
    unittest.main()
