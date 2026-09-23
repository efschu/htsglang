"""23.09. (fnFL2x43): D's first decode round dies on TP0 with an asynchronous
'illegal memory access' that is REPORTED one round late (at the next round's
stream wait) and leaves no GPU coredump. ``spec_stage_sync.checkpoint`` syncs
the stream after each stage of the first N rounds, so the first checkpoint
whose sync raises bounds the fault between itself and ``last_ok``.

What must hold for that to be usable on a boot and harmless on every other:
off by default (no sync at all), a per-stage budget, never inside a stream
capture, and a fault that is logged with its bound AND re-raised -- the probe
localizes, it must never swallow the error it found.
"""
import logging
import unittest
from unittest import mock

from sglang.srt.speculative import spec_stage_sync as sss
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-weg2-unit")


class _Stream:
    """Stands in for a CUDA stream; ``fault`` makes its sync raise."""

    def __init__(self, fault=None):
        self.fault = fault
        self.sync_ct = 0

    def synchronize(self):
        self.sync_ct += 1
        if self.fault is not None:
            raise self.fault


class StageSync(unittest.TestCase):
    def setUp(self):
        self._env = sss.os.environ.pop(sss.ENV, None)
        self._reset()
        self.stream = _Stream()
        self._patches = [
            mock.patch.object(sss.torch.cuda, "current_stream", return_value=self.stream),
            mock.patch.object(sss.torch.cuda, "is_current_stream_capturing", return_value=False),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        sss.os.environ.pop(sss.ENV, None)
        if self._env is not None:
            sss.os.environ[sss.ENV] = self._env
        self._reset()

    @staticmethod
    def _reset():
        sss._SEEN.clear()
        sss._STATE.update(budget=None, last_ok="none", checked_ct=0)

    def test_off_by_default_never_syncs(self):
        with self.assertNoLogs(sss.logger, level=logging.INFO):
            for _ in range(3):
                sss.checkpoint("verify-forward")
        self.assertEqual(self.stream.sync_ct, 0)

    def test_the_budget_is_per_stage(self):
        sss.os.environ[sss.ENV] = "2"
        with self.assertLogs(sss.logger, level=logging.INFO) as cm:
            for _ in range(3):
                sss.checkpoint("draft")
                sss.checkpoint("verify-forward")
        self.assertEqual(self.stream.sync_ct, 4)
        self.assertEqual(len(cm.output), 4)
        self.assertIn("SPEC-STAGE-SYNC ok stage=draft n=1 checked=1", cm.output[0])
        self.assertIn("stage=verify-forward n=2 checked=4", cm.output[3])

    def test_capture_is_never_synced(self):
        sss.os.environ[sss.ENV] = "4"
        with mock.patch.object(sss.torch.cuda, "is_current_stream_capturing", return_value=True):
            sss.checkpoint("verify-forward")
        self.assertEqual(self.stream.sync_ct, 0)

    def test_an_explicit_stream_is_the_one_synced(self):
        sss.os.environ[sss.ENV] = "1"
        plan = _Stream()
        sss.checkpoint("verify-prepare", stream=plan)
        self.assertEqual((plan.sync_ct, self.stream.sync_ct), (1, 0))

    def test_a_fault_is_bounded_by_last_ok_and_reraised(self):
        sss.os.environ[sss.ENV] = "8"
        sss.checkpoint("draft")
        self.stream.fault = RuntimeError(
            "CUDA error: an illegal memory access was encountered\nmore"
        )
        with self.assertLogs(sss.logger, level=logging.ERROR) as cm:
            with self.assertRaises(RuntimeError):
                sss.checkpoint("verify-sample")
        line = cm.output[0]
        self.assertIn("SPEC-STAGE-SYNC FAULT stage=verify-sample n=1 checked=2", line)
        self.assertIn("last_ok=draft#1", line)
        self.assertIn("illegal memory access was encountered", line)
        self.assertNotIn("more", line)


if __name__ == "__main__":
    unittest.main()
