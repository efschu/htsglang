"""fnFL2 H49: the scheduler HOST share of a decode round, of a PP pass and of
the per-chunk publish on P -- the measuring half of the 27B host-path fixes.

Hermetic, CPU only. The decode half runs through the real DecodeRoundLog
with the fake device clock of the H23 tests; the HiCache poll through the
real ``UnifiedRadixCache.check_hicache_events`` on a stub with a scripted
``perf_counter``; the PP half and the publish ledger directly. Numbers are
x153b's (24.09., D-TP0 and P-PP0) where the boot has them.
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import host_round_cost as hrc
from sglang.srt.managers.scheduler_components.decode_round_log import DecodeRoundLog
from sglang.srt.managers.scheduler_components.host_round_cost import (
    DecodeHostCost,
    HostCostCounters,
    PPHostPeriod,
    med_max,
)
from sglang.srt.utils.collective_clock import ClockBackend, CollectiveClock
from sglang.srt.weg2.publish_cost import PublishCostLedger, format_parts
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MOD_HRC = "sglang.srt.managers.scheduler_components.host_round_cost"


class _State:
    def __init__(self) -> None:
        self.now = 0.0
        self.syncs = 0


class _Event:
    def __init__(self, state: _State) -> None:
        self._state = state
        self.t = None

    def record(self) -> None:
        self.t = self._state.now

    def query(self) -> bool:
        return self.t is not None

    def elapsed_time(self, other: "_Event") -> float:
        return other.t - self.t

    def synchronize(self) -> None:  # pragma: no cover - must never run
        self._state.syncs += 1
        raise AssertionError("the instrument synchronized the device")


class _Backend(ClockBackend):
    def __init__(self, state: _State) -> None:
        self.state = state

    def event(self):
        return _Event(self.state)

    def is_capturing(self) -> bool:
        return False


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _capture(test: unittest.TestCase, *names: str) -> _Capture:
    cap = _Capture()
    for name in names:
        lg = logging.getLogger(name)
        lg.addHandler(cap)
        lg.setLevel(logging.INFO)
        test.addCleanup(lg.removeHandler, cap)
    return cap


class _Counters(HostCostCounters):
    """A private counter set, so the process singleton is never touched."""


class DecodeHostCostTest(unittest.TestCase):
    def _interval(self, c, *, hc=0.0, drain=0.0, dar=0.0, ar=0.0, arn=0,
                  wait=0.0, calls=1, writing=0.0, loading=0.0):
        c.hc_ms += hc
        c.hc_calls += calls
        c.drain_ms += drain
        c.drain_ar_ms += dar
        c.ar_ms += ar
        c.ar_n += arn
        c.result_wait_ms += wait
        c.writing_ms += writing
        c.loading_ms += loading

    def test_round_interval_is_open_to_next_open_and_splits_host(self):
        """x153b D-TP0 steady: wall ~ gpu 30 ms. Host = wall - result sync;
        host_other = host - hicache; the all_reduce is every HiCache CPU
        collective of the interval, the drain's own reported apart."""
        c = _Counters()
        hc = DecodeHostCost(rank=0, period=0, counters=c)
        hc.on_round_open(round_id=10, mono=1.000)
        self._interval(c, hc=2.5, drain=2.0, dar=1.6, ar=1.9, arn=2, wait=20.0,
                       writing=0.2, loading=0.1)
        hc.on_round_open(round_id=11, mono=1.030)
        r = hc.round_host(10)
        self.assertAlmostEqual(r.wall_ms, 30.0, places=6)
        self.assertAlmostEqual(r.host_ms, 10.0, places=6)
        self.assertAlmostEqual(r.hicache_ms, 2.5)
        self.assertAlmostEqual(r.host_other_ms, 7.5, places=6)
        self.assertAlmostEqual(r.hc_allreduce_ms, 1.9)
        self.assertEqual(r.hc_allreduce_n, 2)
        self.assertAlmostEqual(r.drain_ar_ms, 1.6)
        self.assertIsNone(hc.round_host(11))  # still open
        hc.on_round_end(mono=1.045)
        self.assertAlmostEqual(hc.round_host(11).wall_ms, 15.0, places=6)
        self.assertEqual(hc.round_host(11).hicache_calls, 0)

    def test_end_without_open_round_is_inert_and_prefill_polls_are_not_charged(self):
        """P's (or a prefill batch's) polls between a round's end and the
        next open are not charged to any decode round."""
        c = _Counters()
        hc = DecodeHostCost(rank=0, period=0, counters=c)
        hc.on_round_end(mono=0.5)
        hc.on_round_open(round_id=1, mono=1.0)
        hc.on_round_end(mono=1.01)
        self._interval(c, hc=500.0, calls=40)  # a prefill phase's polls
        hc.on_round_open(round_id=2, mono=5.0)
        hc.on_round_open(round_id=3, mono=5.02)
        self.assertEqual(hc.round_host(2).hicache_ms, 0.0)
        self.assertEqual(hc.round_host(1).hicache_ms, 0.0)

    def test_period_line_every_n_rounds_and_none_when_off(self):
        cap = _capture(self, MOD_HRC)
        c = _Counters()
        hc = DecodeHostCost(rank=1, period=4, counters=c)
        t = 0.0
        for rid in range(9):
            hc.on_round_open(round_id=rid, mono=t)
            self._interval(c, hc=1.0 + rid, ar=0.5, arn=1, wait=10.0)
            t += 0.030
        lines = [l for l in cap.lines if l.startswith("DECODE-HOST-PERIOD")]
        self.assertEqual(len(lines), 2, lines)
        self.assertRegex(
            lines[0],
            r"^DECODE-HOST-PERIOD rank=1 n=4 last_round=3 wall_ms=30\.0/30\.0 "
            r"host_ms=20\.0/20\.0 result_wait_ms=10\.0/10\.0 hicache_ms=3\.0/4\.0 "
            r"drain_ms=0\.0/0\.0 allreduce_ms=0\.5/0\.5 allreduce_n=4 "
            r"writing_ms=0\.0/0\.0 loading_ms=0\.0/0\.0 host_other_ms=18\.0/19\.0 "
            r"hicache_calls=4 \(",
        )
        cap.lines.clear()
        off = DecodeHostCost(rank=1, period=0, counters=c)
        for rid in range(200):
            off.on_round_open(round_id=rid, mono=rid * 0.03)
        self.assertEqual([l for l in cap.lines if "DECODE-HOST-PERIOD" in l], [])

    def test_env_drives_the_period(self):
        self.assertEqual(envs.SGLANG_DEBUG_DECODE_HOST_PERIOD.get(), 64)
        self.assertEqual(envs.SGLANG_DEBUG_PP_HOST_PERIOD.get(), 8)
        with envs.SGLANG_DEBUG_DECODE_HOST_PERIOD.override(0):
            self.assertEqual(DecodeHostCost.from_env(rank=0)._period, 0)
        with envs.SGLANG_DEBUG_PP_HOST_PERIOD.override(0):
            self.assertFalse(PPHostPeriod.from_env().on)

    def test_keep_bounds_memory(self):
        hc = DecodeHostCost(rank=0, period=0, counters=_Counters())
        for rid in range(500):
            hc.on_round_open(round_id=rid, mono=float(rid))
        self.assertLessEqual(len(hc._recent), DecodeHostCost.KEEP)
        self.assertIsNotNone(hc.round_host(498))
        self.assertIsNone(hc.round_host(10))

    def test_med_max(self):
        self.assertEqual(med_max([]), (0.0, 0.0))
        self.assertEqual(med_max([3.0, 1.0, 2.0]), (2.0, 3.0))
        self.assertEqual(med_max([4.0, 1.0]), (4.0, 4.0))


ROUND1 = dict(compute=10.0, waits=[("spec_verify:pool.fetch", 39.4),
                                   ("spec_verify:tp.all_reduce", 16.3)])
STEADY = dict(compute=10.1, waits=[("spec_verify:pool.fetch", 6.9),
                                   ("spec_verify:tp.all_reduce", 4.7)])


class DecodeRoundCostCarriesHostTest(unittest.TestCase):
    """Through the real DecodeRoundLog: the H23 line gains host fields."""

    def setUp(self) -> None:
        self.state = _State()
        self.clock = CollectiveClock(backend=_Backend(self.state))
        self.counters = _Counters()
        self.log = DecodeRoundLog(clock=self.clock, rank=0)
        self.log.host_cost = DecodeHostCost(rank=0, period=0, counters=self.counters)
        self.cap = _capture(
            self,
            "sglang.srt.managers.scheduler_components.decode_round_log",
            "sglang.srt.managers.scheduler_components.wake_round_census",
        )
        self.next_round = 100

    def _round(self, shape=STEADY, hc=0.0, ar=0.0, wait=0.0) -> int:
        rid = self.next_round
        self.next_round += 1
        self.log.begin_round(round_id=rid, bs=1, rows=4)
        with self.log.segment("target_verify", False):
            for family, ms in shape["waits"]:
                with self.clock.span(family.split(":", 1)[-1]):
                    self.state.now += ms
            self.state.now += shape["compute"]
        # the host work between this launch and the next round's open
        self.counters.hc_ms += hc
        self.counters.hc_calls += 1
        self.counters.ar_ms += ar
        self.counters.ar_n += 1 if ar else 0
        self.counters.result_wait_ms += wait
        return rid

    def _costs(self):
        return [l for l in self.cap.lines if l.startswith("DECODE-ROUND-COST")]

    def test_fields_appended_after_ple_ms_and_filled_per_round(self):
        self.log.arm_wake_census(wake_mono=None)
        self._round(shape=ROUND1, hc=3.25, ar=2.5, wait=0.0)
        for _ in range(4):
            self._round(hc=0.75, ar=0.5, wait=0.0)
        self.log.end_round()
        costs = self._costs()
        self.assertEqual(len(costs), 5, costs)
        # the H23 prefix is unchanged; host fields come last
        self.assertRegex(
            costs[0],
            r"gpu_ms=65\.7 compute_ms=10\.0 fetch_ms=39\.4 allreduce_ms=16\.3 "
            r"wall_ms=\S+ since_wake_ms=- graph=no cold=eager ple_ms=- "
            r"host_ms=\d+\.\d hicache_ms=3\.2 hc_allreduce_ms=2\.5$",
        )
        self.assertTrue(costs[4].endswith("hicache_ms=0.8 hc_allreduce_ms=0.5"), costs[4])
        # host <= wall on every line (wait 0 -> host == wall)
        for line in costs:
            wall = float(re.search(r"wall_ms=(\S+)", line).group(1))
            host = float(re.search(r"host_ms=(\S+)", line).group(1))
            self.assertAlmostEqual(host, wall, delta=0.11)
        self.assertEqual(self.state.syncs, 0)

    def test_unknown_host_interval_prints_dash_never_zero(self):
        from sglang.srt.managers.scheduler_components.wake_round_census import (
            WakeRoundCensus,
        )

        census = WakeRoundCensus(rank=0)
        census.arm(wake_mono=None)
        census.note_open(round_id=70, mono=1.0)
        cost = census.on_round(round_id=70, gpu_ms=22.4, compute_ms=10.1,
                               families={}, graphed=True, now_mono=1.1)
        self.assertTrue(cost.line().endswith("host_ms=- hicache_ms=- hc_allreduce_ms=-"))


class _FakeTime:
    """perf_counter advancing by a scripted step per call; monotonic fixed."""

    def __init__(self, steps):
        self._steps = list(steps)
        self.t = 100.0

    def perf_counter(self):
        self.t += self._steps.pop(0) if self._steps else 0.0
        return self.t

    def monotonic(self):
        return 0.0


class HicachePollCountsItsPartsTest(unittest.TestCase):
    def _stub(self, calls):
        return types.SimpleNamespace(
            _attn_reduce_world=lambda: 3,
            _drain_async_work=lambda: calls.append("async"),
            writing_check=lambda: calls.append("writing"),
            loading_check=lambda: calls.append("loading"),
            _pin_trace_every=0,
            enable_storage=True,
            drain_storage_control_queues=lambda: calls.append("drain"),
            enable_storage_metrics=False,
            storage_metrics_collector=None,
            ongoing_prefetch={},
        )

    def test_check_hicache_events_adds_total_and_named_parts(self):
        from sglang.srt.mem_cache import unified_radix_cache as urc

        counters = _Counters()
        calls = []
        # perf_counter calls in order: t0, before-writing, after-writing,
        # after-loading, before-drain, after-drain, end
        fake = _FakeTime([0.0, 0.001, 0.0002, 0.0001, 0.0, 0.0016, 0.0001])
        with mock.patch.object(urc, "_HOST_COST", counters), mock.patch.object(urc, "time", fake):
            urc.UnifiedRadixCache.check_hicache_events(self._stub(calls))
        self.assertEqual(calls, ["async", "writing", "loading", "drain"])
        self.assertEqual(counters.hc_calls, 1)
        self.assertAlmostEqual(counters.writing_ms, 0.2, places=6)
        self.assertAlmostEqual(counters.loading_ms, 0.1, places=6)
        self.assertAlmostEqual(counters.drain_ms, 1.6, places=6)
        self.assertAlmostEqual(counters.hc_ms, 3.0, places=6)

    def test_all_reduce_counted_only_when_a_collective_ran(self):
        from sglang.srt.mem_cache import unified_radix_cache as urc

        counters = _Counters()
        waited = []
        stub = types.SimpleNamespace(
            attn_cp_group=None,
            attn_tp_group="tp",
            tp_world_size=3,
            tp_group="tp",
            _wait_bounded=lambda work, label: waited.append(label),
        )
        with mock.patch.object(urc, "_HOST_COST", counters), \
                mock.patch.object(urc.torch.distributed, "get_world_size", lambda group=None: 3), \
                mock.patch.object(urc.torch.distributed, "all_reduce", lambda *a, **k: "work"):
            urc.UnifiedRadixCache._all_reduce_attn_groups(stub, None, None, label="drain")
        self.assertEqual(waited, ["drain/all_reduce/attn_tp"])
        self.assertEqual(counters.ar_n, 1)
        # a world of 1 (group P's PP stages): no collective, nothing counted
        stub1 = types.SimpleNamespace(attn_cp_group=None, attn_tp_group=None,
                                      tp_world_size=1, tp_group=None,
                                      _wait_bounded=lambda *a: None)
        with mock.patch.object(urc, "_HOST_COST", counters):
            urc.UnifiedRadixCache._all_reduce_attn_groups(stub1, None, None)
        self.assertEqual(counters.ar_n, 1)

    def test_result_wait_note_adds_to_the_singleton(self):
        before = hrc.COUNTERS.result_wait_ms
        hrc.note_result_wait(2.5)
        self.assertAlmostEqual(hrc.COUNTERS.result_wait_ms - before, 2.5)


class PPHostPeriodTest(unittest.TestCase):
    def test_sync_wait_host_work_and_starved(self):
        """P stage: launch, sync of the slot's result, host work to the next
        launch; a gap across a pass without a launch is starved, not timed."""
        cap = _capture(self, MOD_HRC)
        php = PPHostPeriod(every=3)
        php.note_launch(0.000)
        self.assertIsNone(php.note_sync(stage=1, entry=0.010, exit=0.910))  # waits 900
        php.note_launch(0.950)                                              # host 40
        self.assertIsNone(php.note_sync(stage=1, entry=0.960, exit=1.860))
        php.note_no_batch()                                                 # starved gap
        php.note_launch(5.000)
        line = php.note_sync(stage=1, entry=5.005, exit=5.905)
        self.assertIsNotNone(line)
        self.assertRegex(
            line,
            r"^PP-HOST-PERIOD stage=1 n=3 total=3 sync_wait_ms=900\.0/900\.0 "
            r"host_work_ms=40\.0/40\.0 launch_to_sync_ms=10\.0/10\.0 "
            r"host_work_n=1 starved=1 \(",
        )
        self.assertIn(line, cap.lines)

    def test_off_emits_nothing(self):
        cap = _capture(self, MOD_HRC)
        php = PPHostPeriod(every=0)
        self.assertFalse(php.on)
        for i in range(32):
            php.note_launch(float(i))
            self.assertIsNone(php.note_sync(stage=0, entry=i + 0.1, exit=i + 0.2))
        self.assertEqual([l for l in cap.lines if "PP-HOST-PERIOD" in l], [])

    def test_event_loop_pp_times_the_d2h_sync_it_already_does(self):
        """Structural: in event_loop_pp the slot's d2h_event.synchronize() is
        bracketed by perf_counter and handed to PP-HOST-PERIOD -- and no new
        synchronize() call was added (the instrument only times)."""
        from sglang.srt.managers import scheduler_pp_mixin as m

        text = open(m.__file__).read()
        a = text.index("    def event_loop_pp(self: Scheduler):")
        b = text.index("    def event_loop_pp_disagg_prefill(self: Scheduler):")
        src = textwrap.dedent(text[a:b])
        self.assertEqual(src.count(".synchronize()"), 1)
        self.assertIn("_h49_t0 = time.perf_counter()\n", src)
        self.assertIn("_h49_php.note_sync(", src)
        self.assertIn("_h49_php.note_launch(time.perf_counter())", src)
        self.assertIn("pp_host_period().note_no_batch()", src)


class PublishLedgerTest(unittest.TestCase):
    def test_chunk_lines_budget_over_and_request_sum(self):
        """x153b P-PP0 rid weg2-0-4: chunk publishes 113, 225 (stopped at the
        150 ms budget), 38, 17, 17, 15 ms -> sum 425, one over budget."""
        led = PublishCostLedger()
        lines = []
        for ms, stopped in ((113, None), (225, "budget"), (38, None), (17, None),
                            (17, None), (15, None)):
            lines.append(led.note_chunk(rid="weg2-0-4", ms=float(ms), budget_ms=150.0,
                                        stopped=stopped, issued=2, parts="-"))
        self.assertRegex(
            lines[1],
            r"^WEG2-PUBLISH-CHUNK rid=weg2-0-4 chunk=2 ms=225\.0 budget_ms=150 "
            r"over=yes stopped=budget issued=2 sum_ms=338\.0 parts=-$",
        )
        self.assertIn("chunk=1 ms=113.0 budget_ms=150 over=no stopped=-", lines[0])
        req = led.close("weg2-0-4", retain_ms=2.0)
        self.assertRegex(
            req,
            r"^WEG2-PUBLISH-REQ rid=weg2-0-4 chunks=6 chunk_sum_ms=425\.0 "
            r"chunk_max_ms=225\.0 over_budget=1 stopped_budget=1 budget_ms=150 "
            r"retain_ms=2\.0 total_ms=427\.0 \(",
        )
        # closed: a second close has nothing (no chunk, no retain)
        self.assertIsNone(led.close("weg2-0-4", retain_ms=None))

    def test_budget_zero_is_unbounded_never_over(self):
        led = PublishCostLedger()
        line = led.note_chunk(rid="r", ms=1411.0, budget_ms=0.0, stopped=None,
                              issued=1, parts="-")
        self.assertIn("over=no", line)

    def test_retain_only_request_and_bounded_open_set(self):
        led = PublishCostLedger()
        self.assertIn("chunks=0 chunk_sum_ms=0.0", led.close("x", retain_ms=4.0))
        for i in range(PublishCostLedger.MAX_OPEN + 10):
            led.note_chunk(rid="r%d" % i, ms=1.0, budget_ms=150.0, stopped=None,
                           issued=0, parts="-")
        self.assertEqual(len(led._open), PublishCostLedger.MAX_OPEN)

    def test_parts_from_a_finished_publish_clock(self):
        from sglang.srt.mem_cache import hicache_write_path as hwp

        clk = hwp.PublishClock()
        clk.finish(1)
        self.assertIsNotNone(clk.delta)
        self.assertRegex(format_parts(clk),
                         r"^issue:\d+\.\d,move:\d+\.\d,cpu:\d+\.\d,blocked:\d+\.\d,ops:\d+$")
        self.assertEqual(format_parts(object()), "-")


if __name__ == "__main__":
    unittest.main()
