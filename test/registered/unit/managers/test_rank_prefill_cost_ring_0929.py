"""X-COST-LINE (29.09.): the rank fills the D prefill cost ring.

``RankPrefillLog.flush`` hands its measured (new, cached, gpu-ms) to
``weg2.prefill_clock.note_batch_cost`` before it formats the log line; the
front reads the ring over ``/get_server_info`` (IPC, not the log). The front
side (fit, solve, records) is in
``test/registered/unit/weg2/test_weg2_x_cost_line_0929.py``.

Imported the way the reporter's own tests are (CPU default device at module
top): importing it inside a test, after a weg2 sibling module had hidden the
devices, died in dynamo's device table (IndexError, order-dependent).
"""

from __future__ import annotations

import logging

import pytest
import torch

torch.set_default_device("cpu")

from sglang.srt.managers.scheduler_components.metrics_reporter import (  # noqa: E402
    RankPrefillLog,
)
from sglang.srt.weg2 import prefill_clock as pc  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_ring():
    pc._reset_for_tests()
    yield
    pc._reset_for_tests()


def test_rank_fills_the_ring_from_its_own_measurement_with_the_logger_muted():
    """27B review (3): the value in the ring comes from the rank's measured
    numbers, never parsed back out of the log line -- a muted logger fills it
    all the same."""

    class _Timer:
        def __init__(self, log):
            self.log, self.completed = log, []

        def _report(self):
            while self.completed:
                self.log._on_duration(self.completed.pop(0))

    log = RankPrefillLog()
    log.timer = _Timer(log)
    mr = logging.getLogger("sglang.srt.managers.scheduler_components.metrics_reporter")
    was = mr.disabled
    mr.disabled = True
    try:
        log.record(1574, 55744, timed=True)
        log.timer.completed.append(3.8833)       # z30w 08:46:52 TP0: 3883.3 gpu-ms
        log.flush()
    finally:
        mr.disabled = was
    snap = pc.cost_snapshot()
    assert snap["seq"] == 1
    assert snap["recent"] == [{"seq": 1, "n": 1574, "cached": 55744, "ms": 3883.3, "chunks": 1}]


def test_an_untimed_line_is_no_cost_record():

    log = RankPrefillLog()                         # no timer: the untimed line
    log.record(624, 54592, timed=True)
    assert pc.cost_snapshot()["seq"] == 0          # never a zero-cost row
    pc.note_batch_cost(25, 0, 0.0)
    assert pc.cost_snapshot()["seq"] == 0
