"""#1468: the backup thread's bubble wait is skipped while ops queue behind it."""
import os
import queue
import threading
import time
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_bubble_publish as bp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")


def _ctl(depth: int, gate_set: bool):
    ev = threading.Event()
    if gate_set:
        ev.set()
    q = queue.Queue()
    for _ in range(depth):
        q.put(object())
    return SimpleNamespace(_weg2_bubble_gate=ev, backup_queue=q)


class Test1468(unittest.TestCase):
    def test_empty_queue_waits_for_the_bubble_up_to_the_timeout(self):
        c = _ctl(0, gate_set=False)
        t0 = time.perf_counter()
        self.assertFalse(bp.backup_wait_for_bubble(c, timeout_s=0.05))
        self.assertGreaterEqual(time.perf_counter() - t0, 0.045)

    def test_backlog_never_waits(self):
        c = _ctl(3, gate_set=False)
        t0 = time.perf_counter()
        self.assertFalse(bp.backup_wait_for_bubble(c, timeout_s=0.5))
        self.assertLess(time.perf_counter() - t0, 0.02)
        c2 = _ctl(1, gate_set=True)
        self.assertTrue(bp.backup_wait_for_bubble(c2, timeout_s=0.5))

    def test_env_zero_disables_the_wait_and_no_gate_means_no_wait(self):
        os.environ[bp.BUBBLE_WAIT_MS_ENV] = "0"
        try:
            c = _ctl(0, gate_set=False)
            t0 = time.perf_counter()
            self.assertFalse(bp.backup_wait_for_bubble(c, timeout_s=0.5))
            self.assertLess(time.perf_counter() - t0, 0.02)
        finally:
            os.environ.pop(bp.BUBBLE_WAIT_MS_ENV, None)
        self.assertFalse(bp.backup_wait_for_bubble(SimpleNamespace(), timeout_s=0.5))


if __name__ == "__main__":
    unittest.main()
