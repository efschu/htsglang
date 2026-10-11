"""#1158c root: the GROUP deadmen (P, D) never send /health_generate (10.10.).

Metal 10.10. 01:07:32Z (boot dkr27bggufrabar1fs10100058, D.log:4681-4750): the launcher armed the group deadmen with
PROBE_S=10000000 ("tier 1 only"), but boot_deadman.sh started its probe clock at ``last_probe=0`` -- so the first loop
pass after grace sent ONE /health_generate straight to :30032. It ran as D's first forward right after the flip's quiesce
/flush_cache had said idle, and the Release 0.7 s later met last_batch/overlap_result_queue -> W120/W29/W17.

Pinned here, CPU only:
  * the launcher hands ``TIER2=0`` to the P and D deadmen and NOT to the front's (whose tier 2 reaches the awake group
    through the front); the launch line names it;
  * the three call sites still arm exactly P, D and front (a renamed group would silently lose the switch);
  * the tree's boot_deadman.sh, run as the launcher arms it with curl replaced by a recorder, sends no /health_generate
    for P/D (switch, fallback, both) and still sends the front's first probe at grace end (``--selftest-tier2``).
"""

import inspect
import os
import pathlib
import re
import subprocess
import unittest

from flliper.srt.pdflip import launcher

ROOT = pathlib.Path(__file__).resolve().parents[4]
DEADMAN = ROOT / "scripts" / "pdflip" / "devtools" / "boot_deadman.sh"


def _armed_cmd(name, port, probe_s):
    lines = []
    launcher.arm_deadman(lines.append, f"/x/boot.{name}.log", port, f"launch_server.*--port {port}", probe_s,
                         "tag1158c", name, True)
    (cmd,) = [x for x in lines if "would arm deadman" in x]
    return cmd


class TestLauncherHandsTheSwitch(unittest.TestCase):
    def test_group_deadmen_get_tier2_off(self):
        for name, port in (("P", 30031), ("D", 30032)):
            with self.subTest(group=name):
                cmd = _armed_cmd(name, port, 10**7)
                self.assertRegex(cmd, r"(^|\s)TIER2=0\s")
                # the env prefix sits in front of setsid, i.e. it is the deadman PROCESS's environment
                self.assertLess(cmd.index("TIER2=0"), cmd.index("setsid"))
                self.assertIn(f"PDFLIP_DEADMAN_GROUP={name} ", cmd)

    def test_front_deadman_keeps_tier2(self):
        cmd = _armed_cmd("front", 30030, 120)
        self.assertNotIn("TIER2", cmd)
        self.assertIn("PROBE_S=120 ", cmd)

    def test_the_call_sites_arm_exactly_p_d_front(self):
        src = inspect.getsource(launcher)
        names = re.findall(r'arm_deadman\(log, [^\n]*?, ns\.tag, "(\w+)", dry\)', src)
        self.assertEqual(sorted(names), ["D", "P", "front"])
        self.assertEqual(tuple(launcher.GROUP_DEADMEN_TIER1_ONLY), ("P", "D"))

    def test_launch_line_names_the_switch(self):
        src = inspect.getsource(launcher.arm_deadman)
        self.assertIn("' TIER2=0' if tier1_only", src)


class TestDeadmanNeverProbesAGroup(unittest.TestCase):
    def test_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(DEADMAN)]).returncode, 0)

    def test_selftest_tier2(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("TIER2", "PROBE_S", "PDFLIP_DEADMAN_GROUP", "PDFLIP_DEADMAN_GROUP")}
        r = subprocess.run(["bash", str(DEADMAN), "--selftest-tier2"], capture_output=True, text=True,
                           timeout=180, env=env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("SELFTEST-TIER2 PASS", r.stdout)
        self.assertEqual(r.stdout.count("  ok   "), 7, r.stdout)
        self.assertIn("front, PROBE_S=120 (1 /health_generate", r.stdout)


if __name__ == "__main__":
    unittest.main()
