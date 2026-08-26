"""#871a: the window script decides, and it is exercised BEFORE metal.

`desk-written-never-executed`: a window script whose first run is on the metal
it was written for is not a ticket, it is a hope. So its self-test is wired into
the desk gate here, and it must come out right in BOTH directions -- a script
that can only say PASS is not a check.

IT ALSO SHIPPED A FALSE PASS FOR EXACTLY ONE COMMIT, and that is the reason the
regression case below exists. The first version summed `acked=` over the WHOLE
boot log and returned PASS on the W40 boot -- a boot in which every one of the
21 writeback fences reported `acked=0`. Seven lines of that log carry `acked=`
from an unrelated subsystem, three of them `acked=24`, so the script credited 72
storage acknowledgements that no fence ever made. A window script that reports a
false PASS is worse than no script, because it CLOSES a claim that is open.

Hermetic: pure text and exit codes. No CUDA, no boot, no card.
"""

import os
import subprocess
import sys
import unittest

from sglang.test.test_utils import CustomTestCase


def _find_script() -> str:
    """Walk up to the repo root rather than counting `dirname` calls.

    Counting them is how the first version of this file pointed at `test/`
    instead of the root and failed every case, including the one that only
    asks whether the file exists -- a green arm that is red for a reason
    unrelated to the thing under test.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    while True:
        candidate = os.path.join(here, "scripts", "window_871a_verify.py")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(here)
        if parent == here:
            return candidate  # not found; the existence test reports it
        here = parent


SCRIPT = _find_script()


def _run(*args):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


class TestWindow871aVerify(CustomTestCase):
    def test_the_script_exists_and_is_executable(self):
        self.assertTrue(os.path.exists(SCRIPT), SCRIPT)

    def test_its_self_test_passes_both_directions(self):
        """The script's own known-good / known-bad matrix."""
        r = _run("--self-test")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        self.assertIn("SELF-TEST PASSED", r.stdout)

    def test_a_missing_log_is_undecided_not_a_pass(self):
        """Exit 2 is the 'no evidence' code and must never read as success."""
        r = _run("--log", "/nonexistent/boot.log")
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("UNDECIDED", r.stdout)
        self.assertNotIn("RESULT: PASS", r.stdout)

    def test_no_log_argument_is_undecided(self):
        r = _run()
        self.assertEqual(r.returncode, 2, r.stdout)

    def test_it_reproduces_the_known_verdict_of_a_REAL_boot(self):
        """The strongest available desk check: a boot whose answer is known.

        W40 #857 is the boot this whole ticket was rooted on -- 21 fences, all
        `acked=0`, and the phase host tier armed with the full pool set. The
        script must reproduce exactly that split, or it is not measuring the
        boot it claims to read.
        """
        log = "/spinning/evidence-665-f1/boot_w40_857strict_0826_0516.log"
        if not os.path.exists(log):
            self.skipTest(f"evidence log absent: {log}")
        r = _run("--log", log)
        self.assertEqual(
            r.returncode, 1, "W40 must decide FAIL on store delivery\n" + r.stdout
        )
        self.assertIn("[PASS     ] #871 phase host tier", r.stdout)
        self.assertIn("[FAIL     ] #871a a byte has reached the store", r.stdout)

    def test_the_snapshot_mode_reports_cgroup_bytes(self):
        r = _run("--snapshot")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for key in ("current", "anon", "file"):
            self.assertIn(f"{key}=", r.stdout)


if __name__ == "__main__":
    unittest.main()
