"""#605 R2: the corridor trace must report the WORST instant and its own price.

THE DEFECT CLASS. A resting-level reading cannot answer a law about a floor.
#631 measured a cutover that entered at 3006 MiB free and sat at 940 MiB -- 84
MiB under the 1024 law -- for 1.5 s, while every boot-snapshot reading in that
window looked lawful. A summariser that reports a mean, or that quietly drops
samples it failed to take on time, reproduces exactly that blindness with more
decimal places.

So the tests below pin three things: the summary reduces on MIN and not mean,
a trough of one single sample is enough to declare a breach, and the
instrument's own cost is published beside its readings rather than assumed
negligible.
"""

import os
import tempfile
import unittest

from sglang.srt.mem_ledger import corridor_trace
from sglang.srt.mem_ledger.corridor_trace import MIB, CorridorTrace, Sample


def _sample(free_mib, backed_mib=0, t=0.0):
    return Sample(
        monotonic=t,
        nvml_free_bytes=free_mib * MIB,
        nvml_self_bytes=0,
        torch_reserved_bytes=0,
        torch_allocated_bytes=0,
        kv_arena_backed_bytes=backed_mib * MIB,
    )


class TestTheMinimumDecides(unittest.TestCase):
    def test_a_single_trough_sample_breaches_even_when_the_mean_is_lawful(self):
        trace = CorridorTrace()
        # 3006 free for a long while, one instant at 940, back up. The mean is
        # ~2900 MiB and entirely lawful; the law is broken all the same.
        for index in range(20):
            trace.samples.append(_sample(3006, t=index * 0.1))
        trace.samples.append(_sample(940, t=2.0))
        for index in range(20):
            trace.samples.append(_sample(3006, t=2.1 + index * 0.1))

        summary = trace.summary(corridor_mib=1024)
        self.assertEqual(summary["free_min_mib"], 940)
        self.assertTrue(
            summary["breach"],
            "a single sample under the law is a breach; the mean is not the law",
        )
        self.assertEqual(summary["margin_mib"], 940 - 1024)

    def test_a_run_that_never_dips_reports_no_breach_and_a_positive_margin(self):
        trace = CorridorTrace()
        for index in range(10):
            trace.samples.append(_sample(1123, t=index * 0.1))
        summary = trace.summary(corridor_mib=1024)
        self.assertFalse(summary["breach"])
        self.assertEqual(summary["margin_mib"], 99)


class TestTheInstrumentPublishesItsOwnCost(unittest.TestCase):
    def test_summary_carries_the_sample_cost_and_duty_cycle(self):
        trace = CorridorTrace()
        for index in range(10):
            trace.samples.append(_sample(2000, t=index * 0.1))
        trace.cost_total_us = 1000.0  # 10 samples x 100 us
        trace.cost_max_us = 250.0

        summary = trace.summary()
        self.assertEqual(summary["sample_cost_us_mean"], 100.0)
        self.assertEqual(summary["sample_cost_us_max"], 250.0)
        # 1000 us of work across a 0.9 s span.
        self.assertAlmostEqual(summary["duty_pct"], 0.1111, places=3)

    def test_missed_cadence_is_counted_not_hidden(self):
        trace = CorridorTrace()
        trace.samples.append(_sample(2000))
        trace.overruns = 7
        self.assertEqual(trace.summary()["overruns"], 7)


class TestOnByDefault(unittest.TestCase):
    """INVERTED 2026-08-15. This used to pin "no env, no thread".

    That default made a law nobody could see. This sampler is the only
    instrument here that answers the corridor law's own question -- the law is
    a continuous minimum and everything else takes snapshots -- and the
    self-correcting margin downstream is fed from it: the scheduler reports a
    breach only if this armed, and `record_corridor_shortfall` only ever
    writes a number that report produced.

    Measured on this rig over two boots: an external 100 ms NVML sampler saw
    57 and 15 breaches, minima 895 and 935 MiB, while "CORRIDOR LAW BREACHED"
    appeared ZERO times in either serving log and `corridor_shortfall_bytes`
    stayed 0. The price of the old default was one daemon thread; the price of
    the bug was a pool sizer with nothing pulling it back above the law.
    """

    def setUp(self):
        self._saved = os.environ.pop(corridor_trace.TRACE_ENV, None)

    def tearDown(self):
        if self._saved is not None:
            os.environ[corridor_trace.TRACE_ENV] = self._saved
        else:
            os.environ.pop(corridor_trace.TRACE_ENV, None)

    def test_no_env_still_samples_at_the_corridor_cadence(self):
        self.assertEqual(
            corridor_trace.requested_period_ms(), corridor_trace.DEFAULT_PERIOD_MS
        )

    def test_the_operator_can_still_turn_it_off(self):
        for off in ("0", "off", "false", "no"):
            os.environ[corridor_trace.TRACE_ENV] = off
            self.assertIsNone(corridor_trace.requested_period_ms(), off)
            self.assertIsNone(corridor_trace.start(), off)

    def test_a_negative_cadence_is_off_rather_than_a_busy_loop(self):
        os.environ[corridor_trace.TRACE_ENV] = "-5"
        self.assertIsNone(corridor_trace.requested_period_ms())

    def test_the_env_selects_the_corridor_cadence_by_default(self):
        os.environ[corridor_trace.TRACE_ENV] = "1"
        self.assertEqual(
            corridor_trace.requested_period_ms(), corridor_trace.DEFAULT_PERIOD_MS
        )
        os.environ[corridor_trace.TRACE_ENV] = "250"
        self.assertEqual(corridor_trace.requested_period_ms(), 250)


class TestTheRingIsBounded(unittest.TestCase):
    def test_a_long_leg_costs_constant_memory(self):
        trace = CorridorTrace(capacity=100)
        for index in range(10000):
            trace.samples.append(_sample(2000, t=index * 0.1))
        self.assertEqual(len(trace.samples), 100)
        # And the surviving window is the RECENT one, not the first 100.
        self.assertEqual(trace.samples[-1].monotonic, 9999 * 0.1)


class TestDumpRoundTrips(unittest.TestCase):
    def test_dump_writes_summary_and_samples(self):
        import json

        trace = CorridorTrace()
        trace.samples.append(_sample(1500, backed_mib=8000))
        with tempfile.TemporaryDirectory() as directory:
            path = trace.dump(os.path.join(directory, "trace.json"))
            payload = json.load(open(path))
        self.assertEqual(payload["summary"]["free_min_mib"], 1500)
        self.assertEqual(payload["summary"]["arena_backed_min_mib"], 8000)
        self.assertEqual(len(payload["samples"]), 1)


if __name__ == "__main__":
    unittest.main()
