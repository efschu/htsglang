"""NUMPY-THP-SERVE (30.09.): a rank process serves with numpy's MADV_HUGEPAGE hint OFF.

WHY.  27B z30y11 (b4946aa966): flip epoch 6 P->D took 48.1 s -- PP1/PP2 waited 43.5 s at the
sleep leg's group fence for PP0, whose L3 write-behind pass ran 77.8 s (cpu_ms=77776.8,
slices=0) and ended in the same second PP0 reached the fence; four sleep flushes (epochs 14,
24, 28, 32) waited 10.3-12.9 s the same way. perf of the write-behind threads (P PP0, D TP0):
31.4 s / 18.8 s of sys in 152 s, top frames ``arena_complete_census`` ->
``do_huge_pmd_anonymous_page`` -> ``__alloc_pages_direct_compact`` -> ``migrate_pages`` ->
TLB shootdown. The census fills four fresh ``np.empty(slots)`` arrays (5.5 MiB each at 720896
slots, over numpy's 4 MiB hint threshold) per pass; under the host's THP ``defrag=madvise``
each 2 MiB first touch compacts in the faulting thread and migrates the shared arena's shmem
pages under all six ranks. The GGUF scope (3bc8f49909) switched the hint off only for the load
and gave numpy its default back for serving.

Proofs (hermetic):
a) ``numpy_hugepage_off_for_serving`` switches the hint off, says so in one named line, and
   the opt-out env leaves numpy alone;
b) the kernel sees it: a fresh 64 MiB numpy array carries no ``hg`` VmFlag afterwards;
c) the wiring: ``run_scheduler_process`` calls it, after ``configure_scheduler_process``
   (the line lands in the rank's configured log). RED on b4946aa966.
"""

import ast
import inspect
import logging
import os
import sys
import unittest
from unittest import mock

import numpy as np

from sglang.srt.model_loader import gguf_numpy_hugepage as H
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_SWITCH = H.numpy_madvise_hugepage_switch()


def _get_hint():
    try:
        from numpy._core import multiarray as ma
    except ImportError:  # numpy 1.x
        from numpy.core import multiarray as ma
    return bool(ma._get_madvise_hugepage())


@unittest.skipIf(_SWITCH is None, "this numpy has no _set_madvise_hugepage")
class _HintCase(CustomTestCase):
    """Every test starts with numpy's hint ON and leaves the process as found."""

    def setUp(self):
        self._orig = bool(_SWITCH(True))
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("SGLANG_WEG2_NUMPY_HUGEPAGE", None)

    def tearDown(self):
        _SWITCH(self._orig)


class TheServingSwitch(_HintCase):
    def test_off_for_the_process_and_named(self):
        with self.assertLogs(H.logger, level=logging.INFO) as cm:
            self.assertTrue(H.numpy_hugepage_off_for_serving("scheduler tp=0 pp=0"))
        self.assertFalse(_get_hint())
        line = "\n".join(cm.output)
        self.assertIn("[NUMPY-THP] scheduler tp=0 pp=0", line)
        self.assertIn("OFF for the whole rank process (was on;", line)

    def test_opt_out_env_keeps_numpys_setting(self):
        os.environ["SGLANG_WEG2_NUMPY_HUGEPAGE"] = "1"
        self.assertFalse(H.numpy_hugepage_off_for_serving("x"))
        self.assertTrue(_get_hint())

    def test_the_injected_switch_is_asked_for_off(self):
        calls = []
        self.assertTrue(H.numpy_hugepage_off_for_serving("x", switch=lambda v: calls.append(v) or True))
        self.assertEqual(calls, [False])


def _vmflags_of(addr: int):
    with open("/proc/self/smaps") as f:
        inside = False
        for line in f:
            head = line.split()[0] if line.strip() else ""
            if "-" in head and ":" not in head:
                lo, hi = (int(x, 16) for x in head.split("-"))
                inside = lo <= addr < hi
            elif inside and line.startswith("VmFlags:"):
                return line.split()[1:]
    return None


@unittest.skipUnless(
    sys.platform == "linux" and os.path.isdir("/sys/kernel/mm/transparent_hugepage"), "needs Linux THP"
)
class TheKernelSeesNoHugepageAdviceWhileServing(_HintCase):
    SIZE = 64 << 20  # > glibc's largest mmap threshold: the array's own mapping

    def _flags(self):
        a = np.empty(self.SIZE, dtype=np.uint8)
        flags = _vmflags_of(a.ctypes.data + self.SIZE // 2)
        del a
        self.assertIsNotNone(flags)
        return flags

    def test_default_advises_and_serving_does_not(self):
        self.assertIn("hg", self._flags())
        H.numpy_hugepage_off_for_serving("x")
        self.assertNotIn("hg", self._flags())


class TheRankProcessCallsIt(CustomTestCase):
    def test_run_scheduler_process_switches_after_configuring_logging(self):
        from sglang.srt.managers import scheduler as S

        tree = ast.parse(inspect.getsource(S.run_scheduler_process))
        order = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                if name in ("numpy_hugepage_off_for_serving", "configure_scheduler_process"):
                    order.append((node.lineno, name))
        names = [n for _, n in sorted(order)]
        self.assertIn("numpy_hugepage_off_for_serving", names)  # RED on b4946aa966
        self.assertLess(names.index("configure_scheduler_process"),
                        names.index("numpy_hugepage_off_for_serving"))


if __name__ == "__main__":
    unittest.main()
