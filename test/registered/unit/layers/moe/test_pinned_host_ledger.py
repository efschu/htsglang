"""#537: the cross-process pinned-host-bytes ledger.

Its only consumer is the GGUF stream trim, which compares against the CGROUP's
``memory.current`` -- a figure that spans every rank process. The window of
2026-08-04 measured 20.78 + 14.44 + 14.44 = 49.66 GiB of pinned pool across a
TP=3 boot, so a per-process read would have corrected the trim's budget by less
than a third of the real unreclaimable floor.

What is pinned here: the SUM over live publishers, the ``None``-means-unknown
contract, and the requirement that a dead rank stops counting.
"""

import os
import tempfile
import unittest

from sglang.srt.layers.moe import pinned_host_ledger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ENV = "SGLANG_PINNED_HOST_LEDGER_DIR"


class PinnedHostLedgerTest(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._saved = os.environ.get(_ENV)
        self._tmp = tempfile.TemporaryDirectory()
        os.environ[_ENV] = os.path.join(self._tmp.name, "ledger")
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = self._saved
        self._tmp.cleanup()

    def _write_foreign(self, pid, nbytes):
        os.makedirs(pinned_host_ledger.ledger_dir(), exist_ok=True)
        with open(os.path.join(pinned_host_ledger.ledger_dir(), str(pid)), "w") as fh:
            fh.write(str(nbytes))

    def test_unknown_is_not_zero_before_anyone_publishes(self):
        """The #218 provenance rule: absent and empty are different states,
        and the trim keeps its pre-#537 arithmetic on 'absent'."""
        self.assertIsNone(pinned_host_ledger.total_pinned_bytes())

    def test_round_trip_of_this_process(self):
        pinned_host_ledger.publish_pinned_bytes(7 << 30)
        self.assertEqual(pinned_host_ledger.total_pinned_bytes(), 7 << 30)
        pinned_host_ledger.publish_pinned_bytes(9 << 30)
        self.assertEqual(
            pinned_host_ledger.total_pinned_bytes(),
            9 << 30,
            "a republish replaces, it does not accumulate",
        )

    def test_sums_over_live_ranks(self):
        """The reason the file exists: one rank's own pool is not the floor."""
        pinned_host_ledger.publish_pinned_bytes(20780 << 20)
        # A second live process: reuse this test's own parent, which is alive
        # for the duration of the run and is not this pid.
        self._write_foreign(os.getppid(), 14440 << 20)
        self.assertEqual(
            pinned_host_ledger.total_pinned_bytes(), (20780 + 14440) << 20
        )

    def test_a_dead_rank_stops_counting(self):
        pinned_host_ledger.publish_pinned_bytes(1 << 30)
        dead = 4_000_000_000  # far above /proc/sys/kernel/pid_max
        self._write_foreign(dead, 99 << 30)
        self.assertEqual(pinned_host_ledger.total_pinned_bytes(), 1 << 30)
        self.assertFalse(
            os.path.exists(os.path.join(pinned_host_ledger.ledger_dir(), str(dead))),
            "a stale entry is unlinked, not just skipped",
        )

    def test_can_fail_a_stale_entry_would_inflate_the_floor(self):
        """Falsifier for the liveness check: without it the dead rank's 99 GiB
        would be added to the floor and the trim would stop trimming at all."""
        pinned_host_ledger.publish_pinned_bytes(1 << 30)
        self._write_foreign(4_000_000_000, 99 << 30)
        with self.assertRaises(AssertionError):
            self.assertEqual(pinned_host_ledger.total_pinned_bytes(), 100 << 30)

    def test_garbage_entries_are_ignored_not_fatal(self):
        pinned_host_ledger.publish_pinned_bytes(2 << 30)
        os.makedirs(pinned_host_ledger.ledger_dir(), exist_ok=True)
        with open(os.path.join(pinned_host_ledger.ledger_dir(), "not-a-pid"), "w") as f:
            f.write("nonsense")
        self.assertEqual(pinned_host_ledger.total_pinned_bytes(), 2 << 30)

    def test_publish_never_raises_on_an_unusable_directory(self):
        """An optimisation must never be the reason a weight load fails."""
        os.environ[_ENV] = "/proc/self/cannot-create-here"
        pinned_host_ledger.publish_pinned_bytes(1 << 30)
        self.assertIsNone(pinned_host_ledger.total_pinned_bytes())

    def test_clear_drops_this_process(self):
        pinned_host_ledger.publish_pinned_bytes(3 << 30)
        pinned_host_ledger.clear_pinned_bytes()
        self.assertIsNone(pinned_host_ledger.total_pinned_bytes())

    def test_the_staging_layer_boundary_actually_publishes(self):
        """Execution smoke for the WIRING, not the module.

        A ledger nothing writes to reads as 'no publisher' forever, and the
        trim would then keep its pre-#537 arithmetic on exactly the boot the
        fix is for. So this drives the real producer hook -- and with the
        staging TRACE switch off, because the trim is a different consumer and
        must not depend on a debug flag being on.
        """
        from unittest import mock

        from sglang.srt.layers.moe import expert_offload

        expert_offload.reset_streaming_staging_ledger()
        self.addCleanup(expert_offload.reset_streaming_staging_ledger)
        expert_offload.streaming_staging_ledger().record(pinned=5 << 30)

        from sglang.srt.environ import envs

        with mock.patch.object(
            envs.SGLANG_MOE_STAGING_TRACE, "get", lambda: False
        ):
            expert_offload.log_streaming_staging_layer("layer0", mock.MagicMock())

        self.assertEqual(pinned_host_ledger.total_pinned_bytes(), 5 << 30)

    def test_can_fail_an_unpublished_ledger_reads_as_absent(self):
        """Spread precondition for the arm above: the assertion discriminates
        only because the same read returns None when nobody published."""
        self.assertIsNone(pinned_host_ledger.total_pinned_bytes())


if __name__ == "__main__":
    unittest.main()
