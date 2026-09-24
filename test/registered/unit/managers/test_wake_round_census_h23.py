"""fnFL2 H23: DECODE-ROUND-COST, the first decode rounds after a Weg-2 wake.

Hermetic, CPU only: the device clock is the fake of the #1241 tests (events
are plain numbers, ``synchronize`` raises). Run with ``CUDA_VISIBLE_DEVICES=''``.

The numbers are x132's flip 1 on D-TP0 (24.09.): round 1 after the wake
66.5 gpu-ms = compute 10.0 + spec_verify:pool.fetch 39.4 +
spec_verify:tp.all_reduce 16.3 (+0.8 pool.step); a stationary round
22.4 = compute 10.1 + pool.fetch 6.9 + all_reduce 4.7 (+0.7).
"""

from __future__ import annotations

import logging
import types
import unittest

from sglang.srt.managers.scheduler_components.decode_round_log import DecodeRoundLog
from sglang.srt.managers.scheduler_components.wake_round_census import (
    WakeRoundCensus,
    cold_term,
    fold_families,
)
from sglang.srt.utils.collective_clock import ClockBackend, CollectiveClock


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
        raise AssertionError("the census synchronized the device")


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


ROUND1 = dict(compute=10.0, waits=[("spec_verify:pool.fetch", 39.4),
                                   ("spec_verify:tp.all_reduce", 16.3),
                                   ("spec_verify:pool.step", 0.8)])
STEADY = dict(compute=10.1, waits=[("spec_verify:pool.fetch", 6.9),
                                   ("spec_verify:tp.all_reduce", 4.7),
                                   ("spec_verify:pool.step", 0.7)])


class WakeRoundCensusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = _State()
        self.clock = CollectiveClock(backend=_Backend(self.state))
        self.log = DecodeRoundLog(clock=self.clock, rank=0)
        self.cap = _Capture()
        for name in (
            "sglang.srt.managers.scheduler_components.decode_round_log",
            "sglang.srt.managers.scheduler_components.wake_round_census",
        ):
            lg = logging.getLogger(name)
            lg.addHandler(self.cap)
            lg.setLevel(logging.INFO)
            self.addCleanup(lg.removeHandler, self.cap)
        self.next_round = 100

    def _round(self, shape=STEADY, graphed=False) -> int:
        rid = self.next_round
        self.next_round += 1
        self.log.begin_round(round_id=rid, bs=1, rows=4)
        compute = shape["compute"]
        # the segment's phase scope writes the spec_verify: prefix itself
        with self.log.segment("target_verify", graphed):
            for family, ms in shape["waits"]:
                with self.clock.span(family.split(":", 1)[-1]):
                    self.state.now += ms
            self.state.now += compute
        return rid

    def _costs(self):
        return [l for l in self.cap.lines if l.startswith("DECODE-ROUND-COST")]

    def _drain(self) -> None:
        self.log.end_round()

    def test_counts_exactly_the_first_five_rounds_opened_after_the_arm(self):
        """Rounds before the wake are not n=1; the sixth is not priced; a
        second wake numbers from 1 again. A census that started at the arm's
        drain, or kept counting, would name the wrong round as 'round 1'."""
        before = self._round()
        self.log.arm_wake_census(wake_mono=None)
        after = [self._round() for _ in range(7)]
        self._drain()
        costs = self._costs()
        self.assertEqual(len(costs), 5, costs)
        for n, (line, rid) in enumerate(zip(costs, after), start=1):
            self.assertIn("DECODE-ROUND-COST n=%d round=%d " % (n, rid), line)
        self.assertNotIn("round=%d " % before, " ".join(costs))
        self.assertFalse(self.log.wake_census.armed)
        # second wake
        self.log.arm_wake_census(wake_mono=None)
        again = self._round()
        self._round()
        self.assertIn("n=1 round=%d " % again, self._costs()[-1])

    def test_unarmed_emits_nothing_and_records_nothing(self):
        """Negative branch: without a wake the census neither logs nor keeps
        per-round state (the steady decode path pays one attribute test)."""
        for _ in range(8):
            self._round(shape=ROUND1)
        self._drain()
        self.assertEqual(self._costs(), [])
        self.assertEqual(self.log.wake_census._opens, {})
        self.assertEqual(self.state.syncs, 0)
        self.assertTrue(any(l.startswith("Decode rank batch") for l in self.cap.lines))

    def test_the_line_carries_the_rounds_own_split_folded(self):
        """x132's round 1 and a steady round through the real DecodeRoundLog:
        the spec_verify: prefix is folded and the terms are the round's own
        (an eager fake round -- the fake clock has no graph event nodes)."""
        self.log.arm_wake_census(wake_mono=None)
        self._round(shape=ROUND1)
        for _ in range(4):
            self._round(shape=STEADY)
        self._drain()
        c1, *_, c5 = self._costs()
        self.assertIn("gpu_ms=66.5 compute_ms=10.0 fetch_ms=39.4 allreduce_ms=16.3", c1)
        self.assertIn("gpu_ms=22.4 compute_ms=10.1 fetch_ms=6.9 allreduce_ms=4.7", c5)
        self.assertIn("graph=no cold=eager", c1)
        self.assertEqual(self.state.syncs, 0)

    def test_round_one_names_the_pool_fetch_and_a_steady_round_names_compute(self):
        """'n=1 near n=5' is read off cold=: x132's graphed round 1 must say
        pool.fetch, the steady round compute."""
        census = WakeRoundCensus(rank=0)
        census.arm(wake_mono=None)
        census.note_open(round_id=70, mono=1.0)
        census.note_open(round_id=71, mono=1.1)
        r1 = census.on_round(round_id=70, gpu_ms=66.5, compute_ms=10.0,
                             families=dict(ROUND1["waits"]), graphed=True, now_mono=1.2)
        r2 = census.on_round(round_id=71, gpu_ms=22.4, compute_ms=10.1,
                             families=dict(STEADY["waits"]), graphed=True, now_mono=1.2)
        self.assertEqual((r1.cold, r2.cold), ("pool.fetch", "compute"))
        self.assertIn("graph=yes cold=pool.fetch", r1.line())

    def test_wall_ends_at_the_next_open_and_since_wake_starts_at_the_wake(self):
        """Derived: a round's wall is open(n) -> open(n+1), not open -> emit
        (the emit is one round late by construction); since_wake is open(n)
        minus the wake, so n=1 is the post-wake host path to the first
        decode launch."""
        census = WakeRoundCensus(rank=2)
        census.arm(wake_mono=10.0)
        census.note_open(round_id=70, mono=13.150)
        census.note_open(round_id=71, mono=13.164)
        cost = census.on_round(round_id=70, gpu_ms=66.1, compute_ms=4.1,
                               families={"spec_verify:tp.all_reduce": 39.4},
                               graphed=True, now_mono=13.300)
        self.assertEqual(cost.n, 1)
        self.assertAlmostEqual(cost.wall_ms, 14.0, places=3)
        self.assertAlmostEqual(cost.since_wake_ms, 3150.0, places=3)
        self.assertEqual(cost.cold, "tp.all_reduce")
        # the last known round ends at the drain
        last = census.on_round(round_id=71, gpu_ms=38.9, compute_ms=4.3,
                               families={}, graphed=True, now_mono=13.200)
        self.assertAlmostEqual(last.wall_ms, 36.0, places=3)

    def test_fold_and_cold_term_contracts(self):
        folded = fold_families({"spec_verify:pool.fetch": 1.0, "pool.fetch": 2.0,
                                "spec_draft:tp.all_reduce": 3.0})
        self.assertEqual(folded, {"pool.fetch": 3.0, "tp.all_reduce": 3.0})
        self.assertEqual(cold_term(graphed=True, compute_ms=None, folded=folded), "unsplit")
        self.assertEqual(cold_term(graphed=True, compute_ms=5.0, folded=folded), "compute")


class SchedulerArmsOnTheFirstPassTest(unittest.TestCase):
    def test_pass_zero_arms_with_the_wake_time_and_later_passes_do_not(self):
        """The wake arms the post-wake pass counter (weight_updater); the
        scheduler's pass 0 arms the census with ``_weg2_last_wake_t``. Passes
        1..7 must not re-arm (that would restart n=1 in mid-burst)."""
        from sglang.srt.managers.scheduler import Scheduler

        arms = []
        drl = types.SimpleNamespace(arm_wake_census=lambda *, wake_mono: arms.append(wake_mono))
        stub = types.SimpleNamespace(
            _weg2_post_wake_pass_n=0,
            _weg2_last_wake_t=123.25,
            metrics_reporter=types.SimpleNamespace(decode_round_log=drl),
        )
        stub._weg2_arm_wake_round_census = (
            lambda: Scheduler._weg2_arm_wake_round_census(stub)
        )
        batch = types.SimpleNamespace(forward_mode=None, reqs=[])
        for _ in range(3):
            Scheduler._weg2_post_wake_pass_log(stub, batch)
        self.assertEqual(arms, [123.25])
        self.assertEqual(stub._weg2_post_wake_pass_n, 3)


if __name__ == "__main__":
    unittest.main()
