"""The cross-session GPU arbitration must not be writable by accident (#438).

Two things meet here and both were wrong in the same way -- they treated the
mere EXISTENCE of a path as a claim:

* ``/tmp/gpu-card-N.lock`` is acquired with ``mkdir``, so it is a DIRECTORY. A
  plain file at that path is debris: it makes the atomic acquire fail for the
  next legitimate taker and it reads as "card held" to every tool that only
  tests for existence.
* ``/spinning/gpu-arb/holder`` is a claim line (session, cards, purpose,
  since). An EMPTY holder is not a claim, it is a leftover.

The observed leak planted all four as 0-byte regular files with identical
nanosecond mtimes -- the signature of one ``touch a b c d`` call, not of four
independent writers. The writer was a heartbeat loop touching them without
``-c``; ``arb-reaper.sh`` then found a stray lock, saw no holder, and
RECONSTRUCTED one, turning a stray touch into a permanent phantom hold.

The fix is symmetric: a heartbeat refreshes what exists (``touch -c``), and the
reaper deletes an empty holder instead of refreshing it and never creates one.
This file pins the reaper half hermetically, against a throwaway ARB directory
and a throwaway lock root, plus the guard that keeps the Python side out of the
real paths.

FALSIFIER: restore the reconstruct-from-lock branch in ``arb-reaper.sh`` and
``test_a_lock_without_a_holder_does_not_mint_one`` goes red; drop the
empty-holder deletion and ``test_an_empty_holder_is_deleted_not_refreshed``
goes red.
"""

import os
import subprocess
import sys
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

#: The live cross-session reaper. It is infrastructure shared by every session
#: on this box and deliberately lives outside the repo, so it is tested where
#: it actually runs rather than copied in (a copy would drift silently).
_REAPER = "/spinning/gpu-arb/arb-reaper.sh"

#: The four paths this whole file exists to keep untouched.
REAL_ARB_PATHS = (
    "/tmp/gpu-card-0.lock",
    "/tmp/gpu-card-1.lock",
    "/tmp/gpu-card-2.lock",
    "/spinning/gpu-arb/holder",
)


def _snapshot(paths=REAL_ARB_PATHS):
    """``{path: (exists, is_dir)}`` for the real arbitration paths.

    Deliberately NOT mtime, and deliberately NOT the inode. The inode was
    here on the claim that "creation, deletion and a change of type or
    inode cannot happen [from a foreign heartbeat]". That claim is false
    (#654): another session's heartbeat refreshes its holder by writing a
    temp file and renaming it over the target -- the correct atomic
    update, and one that mints a new inode each time. Asserting on the
    inode made this tearDown fail for something no test did, which is
    exactly the failure mode the mtime exclusion was already guarding
    against.

    Existence and type are the only foreign-safe invariants; a content
    hash would false-positive identically.

    The trade, named: a test REPLACING the holder via rename is no longer
    caught. Catching tests that CREATE or DELETE real arbitration files --
    the reason this guard exists -- is unaffected.
    """
    out = {}
    for path in paths:
        try:
            os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            out[path] = (False, False)
        else:
            out[path] = (True, os.path.isdir(path))
    return out


_HAVE_REAPER = os.path.isfile(_REAPER)


@unittest.skipUnless(_HAVE_REAPER, f"{_REAPER} is not present on this host")
class TestTheReaperNeverMintsAClaim(CustomTestCase):
    """``arb_operator_holder_hygiene``, run against a fake ARB directory.

    The function is sourced out of the real script, so this tests the text the
    running reapers will execute after their next restart -- not a copy.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.arb = os.path.join(self._tmp.name, "arb")
        self.locks = os.path.join(self._tmp.name, "locks")
        os.makedirs(self.arb)
        os.makedirs(self.locks)
        self.before = _snapshot()

    def tearDown(self):
        self.assertEqual(
            self.before,
            _snapshot(),
            "the hermetic reaper test touched the REAL arbitration paths",
        )
        self._tmp.cleanup()

    def _hygiene(self):
        """Source the reaper and run one hygiene iteration."""
        script = (
            f"ARB={self.arb!r} CARD_LOCK_ROOT={self.locks!r} "
            f"ARB_SIDE=operator; export ARB CARD_LOCK_ROOT ARB_SIDE; "
            f". {_REAPER}; arb_operator_holder_hygiene"
        )
        return subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "99"},
        )

    def _holder(self):
        return os.path.join(self.arb, "holder")

    def test_sourcing_the_reaper_does_not_enter_its_loop(self):
        """The main-guard is what makes this file testable at all: without it,
        sourcing the script would run the 2-minute reap loop forever."""
        result = self._hygiene()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_a_planted_regular_file_lock_is_not_occupancy(self):
        """A card lock is a directory. A regular file at that path is debris
        and must not make the reaper act as if a card were held."""
        open(os.path.join(self.locks, "gpu-card-0.lock"), "w").close()
        self._hygiene()
        self.assertFalse(
            os.path.exists(self._holder()),
            "a stray regular-file lock was read as a held card",
        )

    def test_a_lock_without_a_holder_does_not_mint_one(self):
        os.mkdir(os.path.join(self.locks, "gpu-card-0.lock"))
        self._hygiene()
        self.assertFalse(
            os.path.exists(self._holder()),
            "the reaper invented a claim no session made",
        )

    def test_an_empty_holder_is_deleted_not_refreshed(self):
        os.mkdir(os.path.join(self.locks, "gpu-card-0.lock"))
        open(self._holder(), "w").close()
        self._hygiene()
        self.assertFalse(
            os.path.exists(self._holder()),
            "a 0-byte holder is not a claim and must not survive a tick",
        )

    def test_a_real_holder_is_refreshed_and_kept(self):
        """The anti-over-deletion control: a claim with content is exactly
        what the heartbeat mirror is for, and it survives."""
        os.mkdir(os.path.join(self.locks, "gpu-card-0.lock"))
        with open(self._holder(), "w") as handle:
            handle.write("session=operator  cards=0,1,2  purpose=test\n")
        self._hygiene()
        self.assertTrue(os.path.exists(self._holder()))
        with open(self._holder()) as handle:
            self.assertIn("session=operator", handle.read())

    def test_no_lock_at_all_leaves_everything_alone(self):
        with open(self._holder(), "w") as handle:
            handle.write("session=other  cards=0  purpose=someone else\n")
        self._hygiene()
        self.assertTrue(os.path.exists(self._holder()))


class TestNothingWritesThePathsOnProcessExit(CustomTestCase):
    """The deferred-writer falsifier.

    One sighting of the leak came ~9 minutes after a window closed, with no
    pytest and no container running, which is the shape of an ``atexit`` /
    finalizer / timer path materializing the production defaults at teardown
    rather than honoring the redirect. A run-time guard would not see that, so
    this one watches a whole PROCESS LIFETIME: import the lock and arbitration
    harness, exit immediately, and check from outside that nothing appeared.

    The mtime of an already-present path is deliberately not asserted on --
    see ``_snapshot``. Creation is the observable that mattered in every
    sighting: all four paths were 0-byte REGULAR files, which is what a single
    ``touch a b c d`` leaves behind and what no correct writer here produces
    (a card lock is a ``mkdir``, a holder is a claim line).
    """

    IMPORTS = (
        "from sglang.srt.planner import comm_suite",
        "from sglang.srt.workbench import arb",
        "from sglang.srt.registry import ledger",
    )

    def test_importing_the_harness_and_exiting_touches_nothing(self):
        before = _snapshot()
        script = "\n".join(self.IMPORTS) + "\nraise SystemExit(0)\n"
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "99"},
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(
            before,
            _snapshot(),
            "importing the lock/arbitration harness and exiting created or "
            "replaced a real arbitration path",
        )


class TestThePythonLockPathsAreRedirectable(CustomTestCase):
    """The Python half of the same arbitration must be movable in one step.

    Before ``HTSGLANG_CARD_LOCK_ROOT`` a test had to patch three module
    attributes by hand to keep ``_CardWindow`` off the real paths, and the one
    it forgot wrote debris under /tmp that the rig then read as a live claim.
    """

    def test_the_lock_family_follows_one_env_var(self):
        import importlib

        from sglang.srt.planner import comm_suite

        with tempfile.TemporaryDirectory() as root:
            os.environ["HTSGLANG_CARD_LOCK_ROOT"] = root
            try:
                reloaded = importlib.reload(comm_suite)
                self.assertTrue(reloaded.LOCK_DIR_FMT.startswith(root))
                self.assertTrue(reloaded.LEGACY_LOCK_DIR.startswith(root))
                self.assertTrue(reloaded.QUIET_LOCK_DIR.startswith(root))
            finally:
                del os.environ["HTSGLANG_CARD_LOCK_ROOT"]
                importlib.reload(comm_suite)

    def test_the_production_default_is_still_the_rig_wide_name(self):
        """Five independent tools arbitrate on these exact names; the seam may
        not change what an unconfigured process uses."""
        from sglang.srt.planner import comm_suite

        self.assertEqual(comm_suite.LOCK_DIR_FMT, "/tmp/gpu-card-{}.lock")
        self.assertEqual(comm_suite.LEGACY_LOCK_DIR, "/tmp/gpu-owner.lock")
        self.assertEqual(comm_suite.QUIET_LOCK_DIR, "/tmp/gpu-quiet.lock")


if __name__ == "__main__":
    unittest.main()
