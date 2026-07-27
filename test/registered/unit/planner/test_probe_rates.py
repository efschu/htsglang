"""What the memory-bandwidth micro-probe measures, and what survives it.

``_bench_membw_gbs`` runs three kernels over a working set well past L2: a
read-only reduction, a copy, and a decode-shaped GEMV weight read. Its own
docstring names the GEMV as the meaningful one --

    "The GEMV is what a bs=1 decode actually does (stream the weights once),
     so this number is the right divisor for the decode roofline"

-- and then returns ``max()`` of the three. On every card measured here the
streaming kernels are the faster ones, so the maximum is the streaming peak
and the GEMV rate never reaches a consumer. The reduction and the copy were
meant as a guard against a GEMV that fails to saturate on some architecture;
a maximum implements that guard by discarding the informative number in
exactly the case it was supposed to cover.

These tests pin that behaviour before it is replaced, so the replacement has
something to be a diff against.
"""

import unittest
from unittest import mock

import torch

from sglang.srt import uneven_perf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

#: Rigged per-kernel rates in the order the probe runs them: reduction, copy,
#: GEMV. The GEMV value is the lowest, which is the situation the probe's own
#: docstring describes as the one worth measuring.
_RIGGED = [1000.0, 900.0, 500.0]


class _RiggedProbe:
    """Replaces ``_time_best_gbs`` and records the calls."""

    def __init__(self, rates):
        self.rates = list(rates)
        self.moved_bytes = []

    def __call__(self, dev, fn, moved_bytes, iters=40):
        self.moved_bytes.append(moved_bytes)
        fn()  # run once so a kernel that cannot execute still fails loudly
        return self.rates[len(self.moved_bytes) - 1]


class TestProbeRates(CustomTestCase):
    """Runs on CPU: the probe shapes are shrunk and the timing is rigged, so
    what is under test is the aggregation, not the hardware."""

    def _run(self, rates=_RIGGED):
        probe = _RiggedProbe(rates)
        with mock.patch.object(uneven_perf, "_time_best_gbs", probe), \
                mock.patch.object(uneven_perf, "_PROBE_GEMV_ROWS", 64), \
                mock.patch.object(uneven_perf, "_PROBE_GEMV_K", 32):
            value = uneven_perf._bench_membw_gbs(torch.device("cpu"))
        return value, probe

    def test_all_three_kernels_run(self):
        _, probe = self._run()
        self.assertEqual(len(probe.moved_bytes), 3)
        n = 64 * 32 * 2
        self.assertEqual(probe.moved_bytes, [n, 2 * n, n])

    def test_the_probe_returns_a_single_number(self):
        """One float reaches the profile, so the three rates are not
        distinguishable by any consumer."""
        value, _ = self._run()
        self.assertIsInstance(value, float)

    def test_the_gemv_rate_is_discarded_by_the_maximum(self):
        """The number the docstring calls "the right divisor for the decode
        roofline" is 500; what comes out is the streaming peak."""
        value, _ = self._run()
        self.assertEqual(value, 1000.0)

    def test_a_gemv_that_fails_to_saturate_is_silently_replaced(self):
        """The reduction/copy guard fires without saying so: an unusable GEMV
        (10 % of the streaming rate, i.e. not bandwidth-bound at all) and a
        healthy one that is merely a little slower than streaming produce the
        SAME output, so no consumer can tell the two apart."""
        unusable, _ = self._run([1000.0, 900.0, 100.0])
        healthy, _ = self._run([1000.0, 900.0, 950.0])
        self.assertEqual(unusable, healthy)

    def test_the_maximum_only_yields_when_the_gemv_is_the_fastest(self):
        """The GEMV rate survives exactly when it is not the interesting
        case -- when the GEMV is measuring more than the streaming kernels
        can, e.g. because it fits in L2."""
        value, _ = self._run([1000.0, 900.0, 2200.0])
        self.assertEqual(value, 2200.0)


if __name__ == "__main__":
    unittest.main()
