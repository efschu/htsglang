# SPDX-License-Identifier: Apache-2.0
"""H136 (NF): the storage-queue agreement cadence does not reach the sleep drain.

The 27B D-round gate (SGLANG_HICACHE_DRAIN_AGREE_EVERY=N, 0619f1280f, on the NF
line since H62 c977af7bbc) thins `drain_storage_control_queues`' gloo MIN over
the attention group to every N-th round while the group's MIN is zero. The NF
line (fnFL2x105) also polls `check_hicache_events` in a GROUP LOOP before every
sleep (`weg2_sleep_drain.drain_until_group_verdict`, 10 ms per poll). A backup
ack that lands while the gate is cold would wait there up to N-1 polls -- with
N=8 up to ~70 ms added to the flip. `UnifiedRadixCache.drain_gate_forced`
makes every poll inside that loop agree again, exactly as with the gate unset.

What must hold:
* unset (N=1) = the unchanged path: the flag is never read, nothing is gated;
* N>1 outside the block = the 27B cadence (1, 9, 17, ...);
* N>1 inside the block = every call agrees;
* three ranks that run the SAME call sequence (decode rounds, then the sleep
  drain's forced block, then decode rounds) post the agreement on IDENTICAL
  call indices with IDENTICAL MINs -- the Form A (TP0 host, workers without KV
  bytes) condition H93..H99 rest on: one rank alone in a gloo all_reduce is the
  wedge;
* the real `_weg2_drain_hicache_before_sleep` wraps its loop in the block, and
  a sleep drain with the gate at N=8 needs as many polls as unset.
"""

import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.mem_cache import unified_radix_cache as u

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ENVS = ("SGLANG_HICACHE_DRAIN_AGREE_EVERY", "SGLANG_HICACHE_ROUND_TIMING")


class _EnvCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._saved = {k: os.environ.pop(k, None) for k in _ENVS}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cache(tp_world_size=3):
    """What check_hicache_events reads (the test_hicache_dround_27b fixture)."""
    c = u.UnifiedRadixCache.__new__(u.UnifiedRadixCache)
    c.attn_cp_group = None
    c.attn_tp_group = None
    c.tp_world_size = tp_world_size
    c.enable_storage = True
    c.enable_storage_metrics = False
    c.storage_metrics_collector = None
    c._pin_trace_every = 0
    c._drain_async_work = lambda: None
    c.writing_check = lambda *a, **k: None
    c.loading_check = lambda: None
    return c


class UnsetIsTheUnchangedPath(_EnvCase):
    def test_force_block_is_inert_without_the_gate(self):
        c = _cache()
        c.drain_storage_control_queues = mock.Mock(return_value=False)
        c._gated_drain_storage_control_queues = mock.Mock()
        with c.drain_gate_forced():
            for _ in range(3):
                c.check_hicache_events()
        for _ in range(3):
            c.check_hicache_events()
        self.assertEqual(c.drain_storage_control_queues.call_count, 6)
        c._gated_drain_storage_control_queues.assert_not_called()
        self.assertFalse(hasattr(c, "_drain_gate_round"))

    def test_the_block_restores_the_previous_flag(self):
        c = _cache()
        self.assertFalse(getattr(c, "_drain_gate_force", False))
        with c.drain_gate_forced():
            self.assertTrue(c._drain_gate_force)
            with c.drain_gate_forced():
                self.assertTrue(c._drain_gate_force)
            self.assertTrue(c._drain_gate_force)
        self.assertFalse(c._drain_gate_force)
        with self.assertRaises(RuntimeError):
            with c.drain_gate_forced():
                raise RuntimeError("the poll raised")
        self.assertFalse(c._drain_gate_force)


class TheBlockAgreesEveryCall(_EnvCase):
    def _counting(self, c, hot=lambda n: False):
        calls = []

        def drain():
            calls.append(c._drain_gate_round)
            return hot(len(calls))

        c.drain_storage_control_queues = drain
        return calls

    def test_cold_cadence_outside_every_call_inside(self):
        os.environ["SGLANG_HICACHE_DRAIN_AGREE_EVERY"] = "8"
        c = _cache()
        calls = self._counting(c)
        for _ in range(10):  # rounds 1..10: agreements at 1 and 9
            c.check_hicache_events()
        with c.drain_gate_forced():  # rounds 11..14: every one
            for _ in range(4):
                c.check_hicache_events()
        for _ in range(12):  # rounds 15..26: last agreement at 14 -> due 22
            c.check_hicache_events()
        self.assertEqual(calls, [1, 9, 11, 12, 13, 14, 22])

    def test_p_takes_no_collective_and_is_never_gated(self):
        os.environ["SGLANG_HICACHE_DRAIN_AGREE_EVERY"] = "8"
        c = _cache(tp_world_size=1)
        c.drain_storage_control_queues = mock.Mock(return_value=False)
        with c.drain_gate_forced():
            c.check_hicache_events()
        for _ in range(5):
            c.check_hicache_events()
        self.assertEqual(c.drain_storage_control_queues.call_count, 6)


class ThreeRanksPostTheSameSequence(_EnvCase):
    """Form A: TP0 carries the host (its queues fill), the workers carry no KV
    bytes (theirs fill later or not at all) -- different LOCAL queues, the SAME
    call sequence. Every rank must enter the MIN on identical call indices."""

    def _run(self, every, arrivals, schedule):
        if every is not None:
            os.environ["SGLANG_HICACHE_DRAIN_AGREE_EVERY"] = str(every)
        ranks = [_cache() for _ in range(3)]
        pending = [0, 0, 0]
        entered = [[] for _ in range(3)]
        agreed = [[] for _ in range(3)]
        cur = {"call": 0, "min": None}

        def make_drain(i):
            def drain():
                if cur["min"] is None:
                    cur["min"] = min(pending)
                entered[i].append(cur["call"])
                agreed[i].append(cur["min"])
                pending[i] -= cur["min"]
                return cur["min"] > 0

            return drain

        for i, r in enumerate(ranks):
            r.drain_storage_control_queues = make_drain(i)
        call = 0
        for forced, n in schedule:
            for _ in range(n):
                call += 1
                cur["call"], cur["min"] = call, None
                for i in range(3):
                    pending[i] += arrivals(i, call)
                for r in ranks:
                    if forced:
                        with r.drain_gate_forced():
                            r.check_hicache_events()
                    else:
                        r.check_hicache_events()
        return entered, agreed, pending

    def _arrivals(self, i, call):
        # TP0 (host) gets acks early and alone; the workers join 2/4 calls later
        return 1 if call in (5 + 2 * i, 23 + i, 24 + 2 * i, 40) else 0

    def test_identical_indices_and_mins_across_a_forced_window(self):
        schedule = [(False, 20), (True, 8), (False, 30)]
        entered, agreed, pending = self._run(8, self._arrivals, schedule)
        self.assertEqual(entered[0], entered[1])
        self.assertEqual(entered[1], entered[2])
        self.assertEqual(agreed[0], agreed[1])
        self.assertEqual(agreed[1], agreed[2])
        # the forced window agrees on every call
        self.assertTrue(set(range(21, 29)) <= set(entered[0]))
        # and outside it the cadence still thins the collective
        self.assertLess(len(entered[0]), 58 // 2)

    def test_unset_is_every_call_on_every_rank(self):
        schedule = [(False, 20), (True, 8), (False, 30)]
        entered, agreed, _p = self._run(None, self._arrivals, schedule)
        for i in range(3):
            self.assertEqual(entered[i], list(range(1, 59)))
        self.assertEqual(agreed[0], agreed[2])


# ---------------------------------------------------------------- sleep drain
_TIMEOUT_S = 5.0


class _Group:
    """A 3-rank all_reduce over threads (the x105 test's gloo stand-in)."""

    def __init__(self, world):
        self._slots = [None] * world
        self._out = None
        self._b1 = threading.Barrier(world, timeout=_TIMEOUT_S)
        self._b2 = threading.Barrier(world, timeout=_TIMEOUT_S)

    def all_reduce(self, rank, values, op):
        self._slots[rank] = list(values)
        if self._b1.wait() == 0:
            self._out = [op(col) for col in zip(*self._slots)]
        self._b2.wait()
        return list(self._out)


def _rank(group, rank, land_at_poll):
    """A real check_hicache_events + the real cadence gate; the drain is the
    MIN rule over the group, one ack per rank that LANDS at call
    `land_at_poll` of this cache (0-based count of check_hicache_events)."""
    c = _cache()
    c.ongoing_backup = {"op": object()}
    c._landed = 0
    c._calls = 0
    c._agreements = 0

    real_check = u.UnifiedRadixCache.check_hicache_events

    def check():
        if c._calls == land_at_poll:
            c._landed = 1
        c._calls += 1
        real_check(c)

    def drain():
        c._agreements += 1
        (n,) = group.all_reduce(rank, [c._landed], min)
        if n:
            c.ongoing_backup.clear()
            c._landed = 0
        return n > 0

    c.check_hicache_events = check
    c.drain_storage_control_queues = drain
    c.hicache_group_max = lambda values, *, label: group.all_reduce(rank, values, max)
    return c


def _updater(tree):
    sch = SimpleNamespace(tree_cache=tree, enable_hierarchical_cache=True)
    sch.idle_blockers = lambda: ["hicache_backup(1)"] if tree.ongoing_backup else []
    wu = SchedulerWeightUpdaterManager.__new__(SchedulerWeightUpdaterManager)
    wu.scheduler = sch
    return wu


class TheSleepDrainIsNotGated(_EnvCase):
    def _sleep_drain(self, every, decode_rounds=3, land=5):
        """`decode_rounds` gated rounds first (the gate goes cold), then the
        sleep drain; each rank's ack lands at its call number `land`."""
        if every is not None:
            os.environ["SGLANG_HICACHE_DRAIN_AGREE_EVERY"] = str(every)
        group = _Group(3)
        trees = [_rank(group, r, land) for r in range(3)]
        out = [None] * 3

        def main(r):
            try:
                for _ in range(decode_rounds):
                    trees[r].check_hicache_events()
                _updater(trees[r])._weg2_drain_hicache_before_sleep(bound_s=2.0)
                out[r] = "returned"
            except Exception as exc:  # noqa: BLE001 -- the outcome is the finding
                out[r] = type(exc).__name__

        th = [threading.Thread(target=main, args=(r,)) for r in range(3)]
        for t in th:
            t.start()
        for t in th:
            t.join(timeout=30)
        return out, [t._calls for t in trees], [t._agreements for t in trees]

    def test_gate_on_drains_in_as_many_polls_as_unset(self):
        out_off, calls_off, agr_off = self._sleep_drain(None)
        out_on, calls_on, agr_on = self._sleep_drain(8)
        self.assertEqual(out_off, ["returned"] * 3)
        self.assertEqual(out_on, ["returned"] * 3)
        self.assertEqual(calls_on, calls_off)
        self.assertEqual(len(set(calls_on)), 1, calls_on)
        self.assertEqual(len(set(agr_on)), 1, agr_on)
        # the three decode rounds before it were gated (1 agreement, not 3)
        self.assertEqual(agr_off[0] - agr_on[0], 2)

    def test_without_the_block_the_gate_would_delay_the_drain(self):
        """The case the block exists for: the same drain with the flag stubbed
        away needs more polls (up to N-1) -- measured, not assumed."""
        with mock.patch.object(
            u.UnifiedRadixCache, "drain_gate_forced", None, create=True
        ):
            _out, calls_gated, _a = self._sleep_drain(8)
        _out2, calls_forced, _a2 = self._sleep_drain(8)
        self.assertGreater(calls_gated[0], calls_forced[0])

    def test_the_release_leg_wraps_the_loop(self):
        import inspect

        src = inspect.getsource(
            SchedulerWeightUpdaterManager._weg2_drain_hicache_before_sleep
        )
        i = src.index("drain_gate_forced")
        j = src.index("drain_until_group_verdict(")
        self.assertLess(i, j)


if __name__ == "__main__":
    unittest.main()
