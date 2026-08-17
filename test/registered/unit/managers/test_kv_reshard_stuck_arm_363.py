"""#363 last mile: an armed reshard that never runs must SAY so.

The #656 audit (46399c045c) settled that ``_execute`` is gated on
``is_fully_idle()`` -- an empty running batch, an empty waiting queue and
drained PP microbatches -- while the flip seam deliberately requires none of
them, so "under continuous load a fully idle round may never arrive".

That makes one hold reason permanent, and the hold reporter deduplicated by
REASON: it logged once, at INFO, and then went silent for as long as the arm
sat there. So the #363 controller could arm a stage, count it as acted on, and
never learn that the vector did not move. The decision reached the actuator and
stopped.

These cases drive the REAL runtime -- real ``arm``, real ``on_round``, real
consensus reduction -- with ``ready_fn`` returning False, which is exactly the
loaded-server condition. Nothing here asserts on private counters: every case
reads the log the operator would read.
"""

import logging
import unittest

import torch

from sglang.srt.managers.kv_reshard import (
    HOLD_ESCALATE_AFTER_S,
    HOLD_REPORT_INTERVAL_S,
    HOLD_STUCK_MARKER,
    KvPoolView,
    KvReshardRuntime,
    logger as reshard_logger,
)
from sglang.test.test_utils import CustomTestCase

VECTORS = [(7, 3, 3), (2, 11, 10), (5, 4, 4)]
#: A third DECLARED target. VECTORS[0] is the CURRENT vector, and arming the
#: current vector is refused outright -- using it as the "different" target
#: would test the refusal path, not the age reset.
OTHER = VECTORS[2]


class _Clock:
    """A hand-advanced clock, so an age is a fact and not a sleep."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _identity_min(payload):
    """Stands in for a group that agrees: the reduction returns this
    rank's own proposal, which is what MIN yields when every rank proposes
    the same thing. The desync paths have their own suite."""
    return list(payload)


def _exchange_must_not_run(*args, **kwargs):
    """These cases never become a move. If the byte channel is ever entered,
    the test has stopped testing what it claims to."""
    raise AssertionError("the exchange ran: this case must never reach _execute")


class _StuckArmBase(CustomTestCase):
    def setUp(self):
        self.clock = _Clock()
        rows = 32
        k = [torch.zeros(rows, 8, dtype=torch.uint8)]
        v = [torch.zeros(rows, 8, dtype=torch.uint8)]
        self.rt = KvReshardRuntime(
            dcp_size=3,
            dcp_rank=0,
            allowed_vectors=VECTORS,
            current_vector=VECTORS[0],
            consensus_interval=1,
            collective_min=_identity_min,
            exchange=_exchange_must_not_run,
            pool_view=KvPoolView(k, v),
            live_slots_fn=lambda: torch.zeros(0, dtype=torch.int64),
            # The whole point: the server is never fully idle.
            ready_fn=lambda: False,
            cutover_fn=lambda vec: None,
            clock=self.clock,
        )

    def _rounds(self, n=1):
        for _ in range(n):
            self.rt.on_round()


class TestAStuckArmBecomesVisible(_StuckArmBase):
    def test_a_hold_that_never_changes_is_re_reported_with_its_age(self):
        """The defect: dedup by reason meant silence for the one reason that
        can last forever."""
        self.rt.arm(VECTORS[1], source="test")
        self._rounds()  # first report

        self.clock.advance(HOLD_REPORT_INTERVAL_S + 1.0)
        with self.assertLogs(reshard_logger, level=logging.INFO) as cm:
            self._rounds()

        self.assertTrue(
            any("hold" in m for m in cm.output), "the stuck hold was not re-reported"
        )
        self.assertTrue(
            any("61s" in m or "6" in m for m in cm.output),
            f"no age in the re-report: {cm.output}",
        )

    def test_it_escalates_to_a_warning_once_the_wait_is_not_plausibly_transient(self):
        self.rt.arm(VECTORS[1], source="test")
        self._rounds()

        self.clock.advance(HOLD_ESCALATE_AFTER_S + 1.0)
        with self.assertLogs(reshard_logger, level=logging.WARNING) as cm:
            self._rounds()

        joined = "\n".join(cm.output)
        self.assertIn(HOLD_STUCK_MARKER, joined)
        self.assertIn("fully-idle", joined)
        self.assertTrue(any(r.levelno == logging.WARNING for r in cm.records))

    def test_it_does_not_report_every_boundary(self):
        """A boundary-rate log would bury the line it is trying to surface."""
        self.rt.arm(VECTORS[1], source="test")
        self._rounds()

        with mock_no_logs(self, reshard_logger):
            self.clock.advance(1.0)
            self._rounds(5)


class TestTheAgeBelongsToTheTarget(_StuckArmBase):
    def test_re_arming_the_same_vector_does_not_hide_the_age(self):
        """A controller that re-proposes every boundary must not keep a
        permanently stuck arm looking fresh."""
        self.rt.arm(VECTORS[1], source="test")
        self._rounds()
        self.clock.advance(HOLD_ESCALATE_AFTER_S + 1.0)

        self.rt.arm(VECTORS[1], source="test-again")  # same target

        with self.assertLogs(reshard_logger, level=logging.WARNING) as cm:
            self._rounds()
        self.assertIn(HOLD_STUCK_MARKER, "\n".join(cm.output))

    def test_re_arming_a_different_vector_starts_a_fresh_age(self):
        """A new target is a new decision; charging it the old wait would
        report a stall that this move has not had."""
        self.rt.arm(VECTORS[1], source="test")
        self._rounds()
        self.clock.advance(HOLD_ESCALATE_AFTER_S + 1.0)

        ok, msg = self.rt.arm(OTHER, source="test")  # a real, declared change
        self.assertTrue(ok, msg)

        with self.assertLogs(reshard_logger, level=logging.INFO) as cm:
            self._rounds()
        self.assertNotIn(HOLD_STUCK_MARKER, "\n".join(cm.output))


class mock_no_logs:
    """Assert nothing is logged, which assertLogs cannot express directly."""

    def __init__(self, case, target_logger):
        self.case = case
        self.logger = target_logger

    def __enter__(self):
        self.records = []
        self.handler = logging.Handler()
        self.handler.emit = self.records.append
        self.logger.addHandler(self.handler)
        # Force the level, or this asserts nothing: with the root logger at
        # WARNING, logger.info() never creates a record and "silence" would be
        # observed even from a reporter logging every single boundary. The
        # can-fail run caught exactly that -- the mutation that reports every
        # boundary passed against the first version of this helper.
        self._prev_level = self.logger.level
        self._prev_propagate = self.logger.propagate
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self._prev_level)
        self.logger.propagate = self._prev_propagate
        if exc[0] is None:
            self.case.assertEqual(
                [r.getMessage() for r in self.records],
                [],
                "expected silence between report intervals",
            )
        return False


if __name__ == "__main__":
    unittest.main()
