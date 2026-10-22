"""Ops #164 (22.10.): a crash core in the code tree must never refuse the next start.

The three fixed containers (flliper-nf[-dev], flliper-27b-int8, flliper-27b-nvfp4-dual)
keep their writable layer across stop/start. proxmox writes a crash core to
"core.<pid>" in the crashing process's cwd (kernel.core_pattern=core,
core_uses_pid=1), the cwd was the code tree, and the entrypoint's preflight_tree
refused the next `docker start`: "REFUSED TREE: Baum nicht sauber" (27B night
22.10. 00:47Z; flliper-27b-nvfp4-dual carries /opt/htsglang/src-27b/core.967 at
the time of writing).

The fix lives in two live rig scripts (not in this tree; the image ships the
host copy of entrypoint.sh, make_flat_ctx.sh "HOST copy"):
  * docker/drei/drei_init.sh -- the --entrypoint of all three containers,
    bind-mounted by path, so it acts at the very next start: CORE-SWEEP moves
    every UNTRACKED core / core.<n> to <evidence>/core/... before the tree check;
  * docker/entrypoint.sh preflight_tree -- the same block, for the next image.
The launcher half (cwd of groups/front = <evidence>/core/<boot>) is pinned in
test/registered/unit/pdflip/test_pdflip_core_cwd_164.py.

Pinned here, on a throwaway git tree, with no GPU and no container:
  * the stand-in tree check (the entrypoint's exact porcelain test) refuses a
    tree with a core -- the red control;
  * drei_init moves the core to the evidence volume, writes HERKUNFT.txt and a
    CORE-SWEEP line, and the stand-in check then passes (rc 0);
  * evidence not writable -> parked outside the tree, start still passes;
  * only cores move: another untracked file still refuses, a TRACKED file
    named "core" is never touched;
  * the real preflight_tree out of entrypoint.sh (set -Eeuo pipefail) passes
    with a core and moves it;
  * both copies of the block are identical.
"""

import os
import re
import subprocess
import tempfile
import unittest

DOCKER = os.environ.get("OPS164_DOCKER_DIR", "/spinning/gpu-arb/docker")
INIT = os.path.join(DOCKER, "drei", "drei_init.sh")
EP = os.path.join(DOCKER, "entrypoint.sh")
LIVE = os.path.isfile(INIT) and os.path.isfile(EP)

ELF_CORE = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8 + b"\x04\x00" + b"x" * 4000  # e_type 4 = ET_CORE

_GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "GIT_CONFIG_NOSYSTEM": "1"}

# The entrypoint's own refusal line, byte for byte in its condition (entrypoint.sh preflight_tree).
_STUB_EP = """#!/bin/bash
T=$1
[ -z "$(git -C "$T" status --porcelain)" ] || { echo "REFUSED TREE: Baum nicht sauber: $(git -C "$T" status --porcelain | head -3 | tr '\\n' ' ')" >&2; exit 3; }
echo "STUB-EP Baum sauber" >&2
exit 0
"""


def _block(path):
    with open(path, encoding="utf-8") as f:
        s = f.read()
    m = re.search(r"^# >>> CORE-SWEEP[^\n]*\n(.*?)^# <<< CORE-SWEEP\n", s, re.S | re.M)
    assert m, f"no CORE-SWEEP block in {path}"
    return m.group(1)


@unittest.skipUnless(LIVE, f"live rig scripts not here ({INIT}, {EP})")
class TestCoreSweep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        r = self.tmp.name
        self.env = dict(os.environ, **_GIT_ENV, HOME=r)
        self.tree = os.path.join(r, "opt", "src-27b")
        self.ev = os.path.join(r, "evidence")
        self.park = os.path.join(r, "parked")
        self.run_dir = os.path.join(r, "tmp-htsglang")
        os.makedirs(os.path.join(self.tree, "python"))
        os.makedirs(self.ev)
        os.makedirs(self.run_dir)
        with open(os.path.join(self.tree, "python", "a.py"), "w") as f:
            f.write("x = 1\n")
        self._git("init", "-q")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "base")
        self.stub = os.path.join(r, "stub_ep.sh")
        with open(self.stub, "w") as f:
            f.write(_STUB_EP)
        os.chmod(self.stub, 0o755)
        # the tag of the start before (drei_init wrote it there; the next start reads it before clearing)
        with open(os.path.join(self.run_dir, "tag"), "w") as f:
            f.write("dkr27bnvfp4dualbar1drei10220219\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _git(self, *a):
        return subprocess.run(["git", "-C", self.tree, *a], env=self.env, check=True,
                              capture_output=True, text=True).stdout

    def _core(self, rel="core.967", data=ELF_CORE):
        p = os.path.join(self.tree, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def _init(self, evidence=None):
        env = dict(self.env, DREI_TMP=self.run_dir, DREI_CORE_TREES=self.tree,
                   DREI_CORE_EVIDENCE=evidence or self.ev, DREI_CORE_PARK=self.park,
                   DREI_ENTRYPOINT=self.stub, DREI_NAME="flliper-27b-nvfp4-dual",
                   HTSGLANG_PROFILE="", FLLIPER_PROFILE="")
        return subprocess.run(["bash", INIT, self.tree], env=env, capture_output=True, text=True, timeout=120)

    def _moved(self, base):
        out = []
        for d, _, fs in os.walk(base):
            out += [os.path.join(d, f) for f in fs]
        return sorted(out)

    # -- red control -------------------------------------------------------
    def test_red_control_the_tree_check_refuses_a_core(self):
        self._core()
        r = subprocess.run(["bash", self.stub, self.tree], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("REFUSED TREE", r.stderr)
        self.assertIn("core.967", r.stderr)

    # -- drei_init ---------------------------------------------------------
    def test_init_moves_the_core_to_evidence_and_the_start_passes(self):
        src = self._core()
        r = self._init()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("STUB-EP Baum sauber", r.stderr)
        self.assertFalse(os.path.exists(src))
        files = self._moved(os.path.join(self.ev, "core"))
        cores = [f for f in files if f.endswith("/src-27b_core.967")]
        self.assertEqual(len(cores), 1, files)
        self.assertIn("/dkr27bnvfp4dualbar1drei10220219_flliper-27b-nvfp4-dual_sweep", cores[0])
        with open(cores[0], "rb") as f:
            self.assertEqual(f.read(), ELF_CORE)
        with open(os.path.join(os.path.dirname(cores[0]), "HERKUNFT.txt")) as f:
            h = f.read()
        self.assertIn("datei=core.967", h)
        self.assertIn("elf_core=ja", h)
        self.assertIn("voriger_tag=dkr27bnvfp4dualbar1drei10220219", h)
        self.assertIn(f"bytes={len(ELF_CORE)}", h)
        self.assertRegex(r.stderr, r"CORE-SWEEP verschoben .*core\.967 -> ")
        self.assertIn("CORE-SWEEP 1 Core-Datei(en)", r.stderr)
        self.assertEqual(self._git("status", "--porcelain"), "")

    def test_init_on_a_clean_tree_logs_zero_and_passes(self):
        r = self._init()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("CORE-SWEEP 0 Core-Datei(en)", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.ev, "core")))

    def test_nested_and_bare_core_both_move(self):
        self._core("core")
        self._core("python/core.12")
        r = self._init()
        self.assertEqual(r.returncode, 0, r.stderr)
        names = sorted(os.path.basename(f) for f in self._moved(os.path.join(self.ev, "core")) if "HERKUNFT" not in f)
        self.assertEqual(names, ["src-27b_core", "src-27b_python_core.12"])

    def test_unwritable_evidence_parks_outside_the_tree_and_passes(self):
        blocker = os.path.join(self.tmp.name, "ev-is-a-file")
        with open(blocker, "w") as f:
            f.write("x")
        src = self._core()
        r = self._init(evidence=blocker)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(os.path.exists(src))
        parked = [f for f in self._moved(self.park) if f.endswith("src-27b_core.967")]
        self.assertEqual(len(parked), 1)
        self.assertIn("CORE-SWEEP WARN", r.stderr)

    def test_only_cores_move_other_dirt_still_refuses(self):
        self._core()
        with open(os.path.join(self.tree, "stray.txt"), "w") as f:
            f.write("not a core\n")
        r = self._init()
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("REFUSED TREE", r.stderr)
        self.assertIn("stray.txt", r.stderr)
        self.assertNotIn("core.967", r.stderr.split("REFUSED TREE", 1)[1])
        self.assertTrue(os.path.exists(os.path.join(self.tree, "stray.txt")))

    def test_a_tracked_file_named_core_is_never_touched(self):
        self._core("docs/core", b"tracked text\n")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "tracked core")
        r = self._init()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(os.path.join(self.tree, "docs", "core")))
        self.assertIn("CORE-SWEEP 0 Core-Datei(en)", r.stderr)

    def test_a_name_that_only_starts_with_core_stays(self):
        self._core("core.py", b"x = 1\n")
        r = self._init()
        self.assertEqual(r.returncode, 3, r.stderr)  # stays dirty: not a crash core, not moved
        self.assertTrue(os.path.exists(os.path.join(self.tree, "core.py")))

    # -- entrypoint.sh preflight_tree ----------------------------------------
    def _preflight(self):
        with open(EP, encoding="utf-8") as f:
            s = f.read()
        m = re.search(r"^preflight_tree\(\) \{\n.*?^\}\n", s, re.S | re.M)
        self.assertIsNotNone(m, "preflight_tree not found in entrypoint.sh")
        head = self._git("rev-parse", "HEAD").strip()
        script = ("set -Eeuo pipefail\n"
                  "say(){ printf '[ep] %s\\n' \"$*\" >&2; }\n"
                  "refuse(){ say \"REFUSED $1: $2\"; exit 3; }\n"
                  + _block(EP) + m.group(0) + "preflight_tree\n")
        env = dict(self.env, TREE=self.tree, HTSGLANG_REVISION=head, STAND="", EVIDENCE_DIR=self.ev,
                   HTSGLANG_TAG="dkr27bbar1drei10220106", HTSGLANG_CORE_PARK=self.park, DREI_NAME="flliper-27b-int8")
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=120)

    def test_entrypoint_preflight_tree_moves_the_core_and_passes(self):
        src = self._core()
        r = self._preflight()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("sauber", r.stderr)
        self.assertFalse(os.path.exists(src))
        cores = [f for f in self._moved(os.path.join(self.ev, "core")) if f.endswith("src-27b_core.967")]
        self.assertEqual(len(cores), 1)
        self.assertIn("/vor-dkr27bbar1drei10220106_flliper-27b-int8_sweep", cores[0])

    def test_entrypoint_preflight_tree_still_refuses_other_dirt(self):
        with open(os.path.join(self.tree, "python", "a.py"), "a") as f:
            f.write("y = 2\n")
        r = self._preflight()
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("REFUSED TREE", r.stderr)

    def test_the_two_blocks_are_identical(self):
        self.assertEqual(_block(INIT), _block(EP))


if __name__ == "__main__":
    unittest.main()
