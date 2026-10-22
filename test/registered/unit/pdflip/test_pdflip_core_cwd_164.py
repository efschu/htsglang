"""Ops #164 (22.10.): a crash core must never land in the code tree.

proxmox writes a crash core to "core.<pid>" relative to the crashing process's
cwd (kernel.core_pattern=core, core_uses_pid=1; docker's LimitCORE=infinity
reaches every container). The launcher spawned the groups and the front with
cwd=tree, so a core made the tree dirty and the container's next `docker start`
was REFUSED TREE by the entrypoint's preflight_tree (27B night 22.10. 00:47Z).

Pinned here:
  * P, D and the front of one boot get ONE directory <evidence>/core/<boot stem>/
    (one per boot: the kernel unlinks an existing core of the same name, and
    container pids restart low on every start);
  * launch_group really spawns with that cwd, not the tree;
  * FLLIPER_PDFLIP_CORE_CWD=0 keeps the tree; a directory that cannot be made
    falls back to the tree with a CORE-CWD FALLBACK line, never a refusal.

No GPU, no launch of a server.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

from flliper.srt.pdflip import launcher as L

_STEM = "boot_weg2_dkr27bbar1drei10220047_a4f4b8fec1_1022_004712"


class _Log:
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(msg)


class TestCoreCwd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ev = os.path.join(self.tmp.name, "evidence")
        self.tree = os.path.join(self.tmp.name, "src-27b")
        os.makedirs(self.ev)
        os.makedirs(self.tree)
        self._p = mock.patch.object(L, "EVIDENCE_DIR", self.ev)
        self._p.start()
        self._e = mock.patch.dict(os.environ, {}, clear=False)
        self._e.start()
        os.environ.pop(L.CORE_CWD_ENV, None)

    def tearDown(self):
        self._e.stop()
        self._p.stop()
        self.tmp.cleanup()

    def test_one_dir_per_boot_for_p_d_and_front(self):
        want = os.path.join(self.ev, "core", _STEM)
        got = {g: L.core_cwd_for(os.path.join(self.ev, f"{_STEM}.{g}.log"), self.tree, _Log())
               for g in ("P", "D", "front")}
        self.assertEqual(set(got.values()), {want})
        self.assertTrue(os.path.isdir(want))
        self.assertNotEqual(want, self.tree)

    def test_two_boots_never_share_a_dir(self):
        a = L.core_cwd_for(f"/x/{_STEM}.P.log", self.tree, _Log())
        b = L.core_cwd_for("/x/boot_weg2_dkr27bbar1drei10220106_a4f4b8fec1_1022_010612.P.log", self.tree, _Log())
        self.assertNotEqual(a, b)

    def test_switch_off_keeps_the_tree(self):
        os.environ[L.CORE_CWD_ENV] = "0"
        self.assertEqual(L.core_cwd_for(f"/x/{_STEM}.P.log", self.tree, _Log()), self.tree)
        self.assertFalse(os.path.exists(os.path.join(self.ev, "core")))

    def test_unmakeable_dir_falls_back_to_tree_loudly(self):
        # a FILE where the core directory would go
        with open(os.path.join(self.ev, "core"), "w") as f:
            f.write("x")
        log = _Log()
        self.assertEqual(L.core_cwd_for(f"/x/{_STEM}.D.log", self.tree, log), self.tree)
        self.assertTrue(any("CORE-CWD FALLBACK" in l for l in log.lines), log.lines)

    def test_launch_group_spawns_in_the_core_dir_not_the_tree(self):
        out = os.path.join(self.tmp.name, "cwd.txt")
        glog = os.path.join(self.ev, f"{_STEM}.P.log")
        spec = L.GroupSpec(
            name="P", port=0,
            argv=[sys.executable, "-c", f"import os; open({out!r}, 'w').write(os.getcwd())"],
            log=glog, env=dict(os.environ),
        )
        log = _Log()
        with mock.patch.object(L, "boot_state_write", lambda *a, **k: None), \
             mock.patch.object(L, "boot_state_dir", lambda: ""):
            L.launch_group(spec, self.tree, log, dry=False)
        spec.proc.wait(timeout=60)
        with open(out) as f:
            cwd = f.read()
        self.assertEqual(os.path.realpath(cwd), os.path.realpath(os.path.join(self.ev, "core", _STEM)))
        self.assertTrue(any(f"cwd {os.path.join(self.ev, 'core', _STEM)}" in l for l in log.lines), log.lines)
        # the tree stays clean: nothing was created in it
        self.assertEqual(os.listdir(self.tree), [])


if __name__ == "__main__":
    unittest.main()
