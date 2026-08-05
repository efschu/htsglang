"""#606 blast-radius: training tenant's reserved_bytes_per_card is no longer
silently defaulted to 0.

If a training job lacks the contractually required
``reserved_bytes_per_card`` field, the workbench must refuse to price it
(RuntimeError) rather than silently under-booking capacity.
"""

import types
import unittest

from sglang.srt.workbench.tenants.training import TrainingWorkTenant
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeJob:
    """Minimal training job stub."""

    def __init__(self, reserved_bytes_per_card=None):
        if reserved_bytes_per_card is not None:
            self.reserved_bytes_per_card = reserved_bytes_per_card
        # else: deliberately no attribute


class _FakeJobQueue:
    """Queue that returns a list of jobs."""

    def __init__(self, jobs):
        self._jobs = jobs

    def queued(self):
        return self._jobs


class _FakeService:
    def __init__(self, jobs):
        self.jobs = _FakeJobQueue(jobs)
        self.config = types.SimpleNamespace(poll_seconds=1.0)


def _make_tenant(jobs):
    """Build a TrainingWorkTenant wired to fake service."""
    service = _FakeService(jobs)
    service.config = types.SimpleNamespace(enabled=True, poll_seconds=1.0)
    return TrainingWorkTenant(
        service=service,
        idle_settle_s=5.0,
        clock=lambda: 0.0,
    )


class TestTrainingReservedBytesPerCard(unittest.TestCase):
    """reserved_bytes_per_card is contractually required."""

    def test_missing_reserved_bytes_per_card_raises(self):
        """A training job without reserved_bytes_per_card raises RuntimeError."""
        tenant = _make_tenant([_FakeJob()])  # no attribute

        with self.assertRaises(RuntimeError) as cm:
            tenant.estimate()
        self.assertIn("reserved_bytes_per_card", str(cm.exception))

    def test_can_fail_proof_restoring_default_zero_would_silence_error(self):
        """If someone restores ``getattr(jobs[0], 'reserved_bytes_per_card', 0)``
        this test goes red because no exception is raised.

        Mechanical proof that the RuntimeError path is active: the old code
        would silently return per_card_bytes=0 and under-book the machine.
        """
        tenant = _make_tenant([_FakeJob()])

        with self.assertRaises(RuntimeError):
            tenant.estimate()

    def test_present_reserved_bytes_per_card_passes_through(self):
        """A job that declares the field is priced normally."""
        tenant = _make_tenant([_FakeJob(reserved_bytes_per_card=4 * 1024 * 1024 * 1024)])
        est = tenant.estimate()
        self.assertEqual(est.per_card_bytes, 4 * 1024 * 1024 * 1024)

    def test_empty_queue_returns_zero_no_crash(self):
        """No queued jobs -> per_card=0, no crash, no RuntimeError."""
        tenant = _make_tenant([])
        est = tenant.estimate()
        self.assertEqual(est.per_card_bytes, 0)


if __name__ == "__main__":
    unittest.main()
