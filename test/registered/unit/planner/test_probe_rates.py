"""What the memory-bandwidth micro-probe measures, and what survives it.

``_bench_membw_rates`` runs three kernels over a working set well past L2: a
read-only reduction, a copy, and a decode-shaped GEMV weight read. They answer
different questions -- the first two say what the DRAM path can do when
nothing else limits it, the third says what a bs=1 decode step actually
reaches reading weights -- so all three are returned.

The probe used to collapse them with ``max()``. On every card measured here
the streaming kernels are the faster ones, so the maximum WAS the streaming
peak and the decode number never reached a consumer. The reduction and the
copy were meant as a guard against a GEMV that fails to saturate on some
architecture; a maximum implements that guard by discarding the informative
number in exactly the case it was supposed to cover, and silently: a GEMV at
10 % of the streaming rate (not bandwidth-bound at all) and one at 95 %
produced identical output.

The guard survives as a NAMED condition in ``decode_bw_basis``, which reports
which divisor it used and, when it falls back, why.
"""

import unittest
from unittest import mock

import torch

from sglang.srt import uneven_perf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

#: Rigged per-kernel rates in the order the probe runs them: reduction, copy,
#: GEMV. The GEMV value is the lowest, which is the situation on every card
#: measured on the reference rig (5090: 1663 / 1524 / 1532 GB/s; 3080:
#: 717.8 / 686 / 717.4).
_RIGGED = [1000.0, 900.0, 500.0]


class _RiggedProbe:
    """Replaces ``_time_best_gbs`` and records the calls."""

    def __init__(self, rates):
        self.rates = list(rates)
        self.moved_bytes = []

    def __call__(self, dev, fn, moved_bytes, iters=40):
        self.moved_bytes.append(moved_bytes)
        fn()  # run once so a kernel that cannot execute still fails loudly
        # Cycles, so one instance can serve repeated probe calls; the kernel
        # order within a call is what the indices mean.
        return self.rates[(len(self.moved_bytes) - 1) % len(self.rates)]


class TestProbeRates(CustomTestCase):
    """Runs on CPU: the probe shapes are shrunk and the timing is rigged, so
    what is under test is the aggregation, not the hardware."""

    def _run(self, rates=_RIGGED):
        probe = _RiggedProbe(rates)
        with mock.patch.object(uneven_perf, "_time_best_gbs", probe), \
                mock.patch.object(uneven_perf, "_PROBE_GEMV_ROWS", 64), \
                mock.patch.object(uneven_perf, "_PROBE_GEMV_K", 32):
            rates_out = uneven_perf._bench_membw_rates(torch.device("cpu"))
            peak = uneven_perf._bench_membw_gbs(torch.device("cpu"))
        return rates_out, peak, probe

    def test_all_three_kernels_run(self):
        _, _, probe = self._run()
        n = 64 * 32 * 2
        # Three per _bench_membw_rates call, and _bench_membw_gbs makes one.
        self.assertEqual(probe.moved_bytes[:3], [n, 2 * n, n])

    def test_the_three_rates_come_back_apart(self):
        """The point of the change: the GEMV rate is no longer swallowed."""
        rates, _, _ = self._run()
        self.assertEqual(rates.read_gbs, 1000.0)
        self.assertEqual(rates.copy_gbs, 900.0)
        self.assertEqual(rates.gemv_gbs, 500.0)

    def test_the_streaming_score_is_unchanged_for_its_consumers(self):
        """``_bench_membw_gbs`` keeps its meaning and its signature -- the
        vocab / KV-speed weighting and the power calibration still get the
        card's streaming bandwidth score, and only that."""
        rates, peak, _ = self._run()
        self.assertEqual(peak, 1000.0)
        self.assertEqual(rates.streaming_peak_gbs, 1000.0)

    def test_the_streaming_score_ignores_a_cache_fed_gemv(self):
        """A GEMV that reads faster than a pure stream is not a DRAM rate, so
        it must not become the card's bandwidth score either."""
        rates, peak, _ = self._run([1000.0, 900.0, 2200.0])
        self.assertEqual(peak, 1000.0)
        self.assertEqual(rates.gemv_gbs, 2200.0)


class TestDecodeBwBasis(CustomTestCase):
    """The saturation guard, as a named condition rather than a maximum.

    ``decode_bw_basis`` is a pure function of the two rate vectors, so it is
    exercised on a bare instance -- no checkpoint, no device."""

    def setUp(self):
        self.m = uneven_perf.PerfCostModel.__new__(uneven_perf.PerfCostModel)

    def test_the_gemv_rate_is_the_divisor(self):
        rates, beta, basis = self.m.decode_bw_basis(
            [1663.0, 717.8, 717.8], [1532.3, 717.4, 717.4]
        )
        self.assertEqual(rates, [1532.3, 717.4, 717.4])
        self.assertEqual(beta, uneven_perf._PREDICT_DECODE_GEMV_RESIDUAL)
        self.assertIn("GEMV", basis)

    def test_no_gemv_rate_falls_back_and_says_so(self):
        """A profile from before PROFILE_VERSION 2 carries no GEMV rate."""
        rates, beta, basis = self.m.decode_bw_basis([1663.0, 717.8, 717.8])
        self.assertEqual(rates, [1663.0, 717.8, 717.8])
        self.assertEqual(beta, uneven_perf._PREDICT_DECODE_BW_COMPRESSION)
        self.assertIn("streaming peak", basis)
        self.assertIn("no GEMV rate", basis)

    def test_an_unsaturated_gemv_falls_back_and_names_the_rank(self):
        """The old max()'s job, done out loud: a GEMV at 10 % of the streaming
        rate is not bandwidth-bound, so it is not a weight-read rate."""
        rates, beta, basis = self.m.decode_bw_basis(
            [1663.0, 717.8, 717.8], [1532.3, 71.0, 717.4]
        )
        self.assertEqual(rates, [1663.0, 717.8, 717.8])
        self.assertEqual(beta, uneven_perf._PREDICT_DECODE_BW_COMPRESSION)
        self.assertIn("rank 1", basis)
        self.assertIn("saturation floor", basis)

    def test_a_cache_fed_gemv_falls_back_and_names_the_rank(self):
        """Not hypothetical: a 64 MiB matrix is L2-resident on a 5090 and
        reads at 2.2 TB/s, past that card's own DRAM peak."""
        rates, _, basis = self.m.decode_bw_basis(
            [1663.0, 717.8, 717.8], [2200.0, 717.4, 717.4]
        )
        self.assertEqual(rates, [1663.0, 717.8, 717.8])
        self.assertIn("rank 0", basis)
        self.assertIn("cache", basis)

    def test_a_slower_but_healthy_gemv_is_kept(self):
        """The floor must not fire on a real measurement. The widest gap
        measured on the reference rig is the 5090's 1532 vs 1663 (92 %); a
        card at 40 % would still be a genuine weight-read rate."""
        for frac in (0.92, 0.6, 0.4, 0.26):
            with self.subTest(frac=frac):
                rates, beta, basis = self.m.decode_bw_basis(
                    [1663.0, 717.8, 717.8],
                    [1663.0 * frac, 717.8 * frac, 717.8 * frac],
                )
                self.assertEqual(beta,
                                 uneven_perf._PREDICT_DECODE_GEMV_RESIDUAL)
                self.assertIn("GEMV", basis)
                self.assertAlmostEqual(rates[0], 1663.0 * frac)

    def test_one_bad_rank_falls_the_whole_rig_back(self):
        """The roofline consumes RATIOS between ranks, so mixing a GEMV rate
        on one rank with a streaming rate on another would compare two
        different quantities."""
        rates, _, _ = self.m.decode_bw_basis(
            [1663.0, 717.8, 717.8], [1532.3, 717.4, 10.0]
        )
        self.assertEqual(rates, [1663.0, 717.8, 717.8])


if __name__ == "__main__":
    unittest.main()
