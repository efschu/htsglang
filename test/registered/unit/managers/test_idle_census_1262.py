# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#1262: the idle pool census livelocked all six schedulers of boot weg2t2a.

THE SPECIMEN (record /spinning/gpu-arb/weg2/BOOT_weg2t2a_0908.md, six py-spy
dumps in BOOT_weg2t2a_killer_context.txt). First full three-group READY of
either Weg-2 train, every gate green, then::

    [2026-09-08 11:57:07,109] INFO weg2.front: WEG2-FLIP begin epoch=0
        sleep=D wake=P outstanding=0 queue=1

never completed. Seven minutes later: ``state=flipping epoch=0 flips=0``. All
SIX scheduler ranks, ``active+gil``, in ONE identical stack, unchanged across
two dumps 30 s apart::

    read_free_rows (kv_row_ownership.py:996)
    _check_full_pool (scheduler_components/invariant_checker.py:275)
    _check_all_pools (scheduler_components/invariant_checker.py:763)
    on_idle (scheduler.py:14123)

The hot line materialises two Python ``frozenset``s of one int per KV row, per
idle pass, per rank, under the GIL. PP2's pool had gone from 162 435 rows
(weg2tr2) to **410 857** (2.53x) because #1259's deferred drafter ``lm_head``
freed 2425.0 MiB -- i.e. the capacity fix succeeding is what made the
instrument unaffordable.

THE MEASURED PART AND THE UNMEASURED PART, kept apart. Measured: the stack, the
row counts, ``active+gil``, the never-completing flip. NOT measured: that the
set-building *cost* is what crossed the threshold (no per-call timing was taken
on that boot, and weg2rg6 ran 714 788 rows without wedging, under a different
group form). This file does not assume the attribution -- item 3 of the fix is
the instrument that will settle it on the next boot, and it is tested here.

WHAT IS FIXED, and why none of it is a threshold
================================================

1. **The idle check is O(1) on the happy path.** ``_check_full_pool`` compares
   the allocator's own counters (``ps.full_available_size``, which is
   ``len(free_pages) + len(release_pages)``, allocator/token.py:52-54) against
   the same partition of ``total``. The row-ENUMERATING census runs only when
   those counters DISAGREE -- which is exactly the job #912 gave it (telling a
   free-list overlap apart from a real leak) and no more.
2. **A queued control message outranks the census.** ``on_idle`` asks the
   receiver whether a frame is already readable and, if so, postpones the
   enumeration and does not re-enter it on the next pass before
   ``process_input_requests`` has run.
3. **The census is instrumented** -- rows and wall ms per enumerating pass,
   #926-rate-limited with its suppressed count, and a WARNING whenever one pass
   alone costs more than the loop's own idle poll interval.
4. **A duty cycle bounds it**, derived from the census's own measured cost, so
   the price stays bounded as the pool grows.

RED-FIRST. Every mechanism above has a mutant here that removes it and shows
the weg2t2a shape returning. No CUDA, no allocator subclass beyond the real
base class, no sleeping (the clock is injected).
"""

import inspect
import unittest

import torch

from sglang.srt.managers.scheduler_components.idle_census_cadence import (
    IDLE_CENSUS_LOG_EVERY,
    IdleCensusCadence,
)
from sglang.srt.managers.scheduler_components.idle_sleeper import IDLE_POLL_CAP_MS
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

#: PP2's pool on the boot that wedged, and on the boot before it. Both measured
#: (BOOT_weg2t2a_0908.md, gate 7). The ratio 2.53 is the whole "why now".
ROWS_WEG2T2A = 410_857
ROWS_WEG2TR2 = 162_435


class _Alloc(BaseTokenToKVPoolAllocator):
    """The real page-list machinery over CPU tensors.

    Same harness as ``test_pool_invariant_double_owned_912.py``: the defect
    lives in what READS ``free_pages``/``release_pages``, not in any behaviour
    a mock would have to reimplement.
    """

    def __init__(self, size: int):
        self.size = size
        self.page_size = 1
        self.device = "cpu"
        self.dtype = torch.int64
        self._free_listeners = []
        self.clear()

    def clear(self):
        self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64)
        self.release_pages = torch.empty((0,), dtype=torch.int64)
        self.is_not_in_free_group = True
        self.free_group = []
        self._notify_clear()

    def alloc(self, need_size: int):
        out = self.free_pages[:need_size].clone()
        self.free_pages = self.free_pages[need_size:]
        return out

    def free(self, free_index: torch.Tensor):
        self.free_pages = torch.cat((free_index, self.free_pages))


class _Tree:
    def __init__(self, evictable=0, protected=0):
        self._e, self._p = evictable, protected

    def protected_size(self):
        return self._p

    def full_protected_size(self):
        return self._p

    def supports_mamba(self):
        return False

    def is_tree_cache(self):
        return True

    def evictable_size(self):
        return self._e

    def sanity_check(self):
        return None


class _Observer:
    def __init__(self):
        self.calls = 0

    def session_held_tokens(self):
        return 0

    def session_held_full_tokens(self):
        return 0


class _Args:
    dcp_size = 1


def _checker(alloc, evictable=0, cadence=None):
    return SchedulerInvariantChecker(
        is_hybrid_swa=False,
        is_hybrid_ssm=False,
        disaggregation_mode=None,
        page_size=1,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=alloc.size,
        server_args=_Args(),
        tree_cache=_Tree(evictable=evictable),
        token_to_kv_pool_allocator=alloc,
        req_to_token_pool=None,
        pool_stats_observer=_Observer(),
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
        **({"idle_census": cadence} if cadence is not None else {}),
    )


def _stats(alloc, evictable=0):
    return PoolStats(
        full_num_used=0,
        full_token_usage=0.0,
        full_available_size=alloc.available_size(),
        full_evictable_size=evictable,
    )


class _CountingRead:
    """Wraps ``read_free_rows`` and counts how often the ENUMERATION happened.

    Patched into the module under test, so it counts the real call site rather
    than a re-implementation of it.
    """

    def __init__(self, real):
        self.real = real
        self.calls = 0

    def __call__(self, alloc):
        self.calls += 1
        return self.real(alloc)


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


# ---------------------------------------------------------------------------
# (1) O(1) ON THE HAPPY PATH
# ---------------------------------------------------------------------------


class TestCountersAgreeMeansNoEnumeration1262(CustomTestCase):
    """A balanced ledger must not touch a single row."""

    def setUp(self):
        import sglang.srt.managers.scheduler_components.invariant_checker as mod

        self.mod = mod
        self.spy = _CountingRead(mod.read_free_rows)
        mod.read_free_rows = self.spy
        self.addCleanup(setattr, mod, "read_free_rows", self.spy.real)

    def test_a_balanced_ledger_enumerates_nothing(self):
        """The pool is whole: available + evictable == total, nothing to say."""
        alloc = _Alloc(1000)
        alloc.alloc(400)  # 600 free, 400 held by the "tree"
        checker = _checker(alloc, evictable=400)
        for _ in range(50):
            leak, msg = checker._check_full_pool(_stats(alloc, evictable=400))
            self.assertFalse(leak, msg)
        self.assertEqual(
            self.spy.calls,
            0,
            "50 idle passes over a balanced pool enumerated rows -- this is "
            "the weg2t2a cost, per pass, per rank, under the GIL",
        )
        self.assertEqual(checker.idle_census.agreed, 50)
        self.assertEqual(checker.idle_census.enumerated, 0)

    def test_mutant_enumerating_unconditionally_is_the_defect(self):
        """The pre-#1262 behaviour, kept reachable as the red.

        This is what the shipped code did on every idle pass: read the rows
        first and only then compare. With 410 857 rows on PP2 that is two
        Python frozensets per pass per rank. The assertion is on the CALL
        COUNT, because the cost is linear in it.
        """
        alloc = _Alloc(1000)
        alloc.alloc(400)
        for _ in range(50):
            # the deleted line, verbatim in shape
            self.mod.read_free_rows(alloc)
        self.assertEqual(
            self.spy.calls,
            50,
            "the mutant must show the enumeration on EVERY pass -- if this "
            "count is not 50 the spy is not wired and the green above is "
            "meaningless",
        )

    def test_the_happy_path_reads_the_counters_the_allocator_already_gave(self):
        """O(1) is a property of the SOURCE, not of a cache.

        ``ps.full_available_size`` is ``available_size()``, computed by
        ``get_pool_stats()`` before ``_check_full_pool`` is entered, and
        ``available_size()`` is two ``len()`` calls on tensors. Pinned against
        the allocator so a future rewrite that makes it enumerate is caught
        here rather than on metal.
        """
        alloc = _Alloc(ROWS_WEG2T2A)
        src = inspect.getsource(type(alloc).available_size)
        self.assertIn("len(self.free_pages)", src)
        self.assertNotIn("tolist", src)
        self.assertEqual(alloc.available_size(), ROWS_WEG2T2A)

    def test_wiring_the_ledger_runs_before_the_census_in_the_real_source(self):
        """Regression guard on the ORDER, not just on the primitives.

        A revert that moved ``read_free_rows`` back above the first
        ``_ledger(...)`` call would leave every test above green while the
        production method is back to the weg2t2a behaviour.
        """
        src = inspect.getsource(SchedulerInvariantChecker._check_full_pool)
        self.assertIn("read_free_rows(", src, "the census must still exist")
        self.assertIn("ps.full_available_size", src)
        self.assertLess(
            src.index("_ledger(ps.full_available_size)"),
            src.index("read_free_rows("),
            "the O(1) ledger must be evaluated BEFORE the row census -- that "
            "ordering IS the fix",
        )


class TestCountersDisagreeMeansEnumerateOnce1262(CustomTestCase):
    """#912's job, preserved exactly: a SURPLUS is the census's question."""

    def setUp(self):
        import sglang.srt.managers.scheduler_components.invariant_checker as mod

        self.mod = mod
        self.spy = _CountingRead(mod.read_free_rows)
        mod.read_free_rows = self.spy
        self.addCleanup(setattr, mod, "read_free_rows", self.spy.real)

    @staticmethod
    def _overlapped(size=1000, overlap=21):
        """The #912 shape: `overlap` ids in BOTH free lists at once.

        Measured on five specimens across two boots (the raw sum read exactly
        21 higher than the same boot's own census `free=`), so the sum
        over-reports and the union does not.
        """
        alloc = _Alloc(size)
        dup = alloc.free_pages[:overlap].clone()
        alloc.release_pages = torch.cat((alloc.release_pages, dup))
        return alloc

    def test_the_surplus_still_reaches_the_census_and_still_closes(self):
        alloc = self._overlapped()
        self.assertEqual(alloc.available_size(), 1000 + 21)
        checker = _checker(alloc)
        leak, msg = checker._check_full_pool(_stats(alloc))
        self.assertEqual(self.spy.calls, 1, "the disagreement must enumerate")
        self.assertFalse(
            leak,
            "#912: the union counts the overlapping row once, so the ledger "
            f"closes and no false leak is raised -- got {msg}",
        )
        self.assertIn("census=", msg, "the line must name which reading paid")

    def test_a_genuine_deficit_still_raises(self):
        """The safety property: the O(1)-first path must not mask a real leak.

        Rows with NO owner (#832/#856 shape) are the opposite sign, and the
        census cannot explain them away -- the union is <= the sum, so it can
        only make a deficit worse. It is therefore decided on the cheap
        reading, without enumerating: the verdict is already determined and a
        diagnostic that cannot change it must not be allowed to hold the loop.
        """
        alloc = _Alloc(1000)
        alloc.alloc(400)  # 600 free, and NOBODY claims the 400
        checker = _checker(alloc, evictable=0)
        leak, msg = checker._check_full_pool(_stats(alloc, evictable=0))
        self.assertTrue(leak, f"a 400-row deficit must stay fatal: {msg}")
        self.assertIn("deficit of 400 row(s)", msg)
        self.assertEqual(
            self.spy.calls,
            0,
            "a deficit needs no rows to be decided -- enumerating here would "
            "put the weg2t2a cost on the path of a boot that is already dying",
        )

    def test_repeated_disagreement_is_cadence_bound_not_per_pass(self):
        """The persistent-overlap case: it must not re-enumerate every pass.

        This is the shape that would otherwise reproduce weg2t2a exactly --
        a disagreement that does not go away, on a 410k-row pool.
        """
        clock = _FakeClock()
        cadence = IdleCensusCadence(clock=clock)
        alloc = self._overlapped()
        checker = _checker(alloc, cadence=cadence)
        for _ in range(200):
            checker._check_full_pool(_stats(alloc))
            clock.advance(0.0)  # no wall time passes: the loop is spinning
        self.assertEqual(
            self.spy.calls,
            1,
            "200 disagreeing passes with no wall time in between must "
            "enumerate ONCE; the duty cycle owns the rest",
        )
        self.assertEqual(cadence.deferred_cadence, 199)
        self.assertEqual(cadence.disagreed, 200)

    def test_a_deferred_pass_is_inconclusive_never_a_raise(self):
        """Raising on the sum alone would reinstate #912's false positive."""
        clock = _FakeClock()
        cadence = IdleCensusCadence(clock=clock)
        alloc = self._overlapped()
        checker = _checker(alloc, cadence=cadence)
        checker._check_full_pool(_stats(alloc))  # the one enumeration
        leak, msg = checker._check_full_pool(_stats(alloc))  # deferred
        self.assertFalse(leak, "a deferred pass must never raise")
        self.assertIn("DEFERRED", msg)
        self.assertIn("INCONCLUSIVE", msg)

    def test_the_cadence_reopens_after_the_derived_interval(self):
        clock = _FakeClock()
        cadence = IdleCensusCadence(max_duty=0.05, clock=clock)
        cadence.record(rows=ROWS_WEG2T2A, cost_ms=100.0)
        # duty 5% of a 100 ms pass -> 1900 ms before the next one is allowed.
        self.assertAlmostEqual(cadence.seconds_until_allowed(), 1.9, places=6)
        clock.advance(1.89)
        self.assertFalse(cadence.may_enumerate())
        clock.advance(0.02)
        self.assertTrue(cadence.may_enumerate())


class TestDutyCycleIsPoolSizeIndependent1262(CustomTestCase):
    """The property that makes this a root fix rather than a threshold."""

    def test_a_bigger_pool_makes_the_census_rarer_not_costlier(self):
        for cost_ms, label in ((40.0, "162k rows"), (101.2, "410k rows")):
            with self.subTest(pool=label):
                clock = _FakeClock()
                cadence = IdleCensusCadence(max_duty=0.05, clock=clock)
                cadence.record(rows=None, cost_ms=cost_ms)
                window = cadence.seconds_until_allowed() + cost_ms / 1000.0
                self.assertAlmostEqual(
                    cost_ms / 1000.0 / window,
                    0.05,
                    places=6,
                    msg="the census's share of wall time must be the SAME "
                    "whatever it cost -- that is what 'bounded independently "
                    "of pool size' means",
                )

    def test_mutant_a_duty_of_one_is_the_pre_fix_behaviour(self):
        clock = _FakeClock()
        cadence = IdleCensusCadence(max_duty=1.0, clock=clock)
        cadence.record(rows=ROWS_WEG2T2A, cost_ms=101.2)
        self.assertEqual(
            cadence.seconds_until_allowed(),
            0.0,
            "duty=1.0 means 'the census may have the whole loop', which is "
            "exactly weg2t2a -- kept reachable so the constant is visibly "
            "load-bearing",
        )

    def test_a_duty_outside_zero_to_one_is_refused(self):
        for bad in (0.0, -0.1, 1.5):
            with self.subTest(max_duty=bad):
                with self.assertRaises(ValueError):
                    IdleCensusCadence(max_duty=bad)


# ---------------------------------------------------------------------------
# (3) THE INSTRUMENT
# ---------------------------------------------------------------------------


class TestTheInstrumentNamesRowsAndMs1262(CustomTestCase):
    def test_the_line_carries_rows_and_wall_ms(self):
        cadence = IdleCensusCadence(clock=_FakeClock())
        emitted = cadence.record(rows=ROWS_WEG2T2A, cost_ms=101.2)
        self.assertIsNotNone(emitted, "the FIRST enumerating pass is never sampled away")
        is_warning, line = emitted
        self.assertIn(f"rows={ROWS_WEG2T2A}", line)
        self.assertIn("wall=101.2 ms", line)
        self.assertTrue(
            is_warning,
            "101.2 ms is above the loop's own 50 ms idle poll cap, so this "
            "pass IS the #1262 reading and must be a WARNING",
        )

    def test_a_pass_over_the_loops_poll_interval_warns_and_names_it(self):
        cadence = IdleCensusCadence(clock=_FakeClock())
        _, line = cadence.record(rows=1, cost_ms=IDLE_POLL_CAP_MS + 0.1)
        self.assertIn("WARNING", line)
        self.assertIn(f"({IDLE_POLL_CAP_MS:.0f} ms", line)
        self.assertIn("IDLE_POLL_CAP_MS", line)

    def test_a_cheap_pass_does_not_warn(self):
        cadence = IdleCensusCadence(clock=_FakeClock())
        is_warning, line = cadence.record(rows=1, cost_ms=IDLE_POLL_CAP_MS - 0.1)
        self.assertFalse(is_warning)
        self.assertNotIn("WARNING", line)

    def test_the_rate_limit_is_the_926_shape_and_prints_its_denominator(self):
        """One line per N, the suppressed count on every line (DENOMINATOR LAW).

        A rate-limited emitter whose suppressed count is invisible reads as a
        zero -- measured four times in one campaign, speed-mode block.
        """
        cadence = IdleCensusCadence(clock=_FakeClock())
        lines = []
        for i in range(IDLE_CENSUS_LOG_EVERY * 2):
            got = cadence.record(rows=10, cost_ms=1.0)  # cheap: never a warning
            if got is not None:
                lines.append(got[1])
        self.assertEqual(
            len(lines),
            3,
            "expected pass 1 (never sampled away), pass N and pass 2N",
        )
        self.assertIn("lines_suppressed=0", lines[0])
        # passes 2..N-1 were swallowed between the two emissions
        self.assertIn(f"lines_suppressed={IDLE_CENSUS_LOG_EVERY - 2}", lines[1])

    def test_a_warning_pass_is_never_rate_limited_away(self):
        """#926's rule: the discriminator is never sampled."""
        cadence = IdleCensusCadence(clock=_FakeClock())
        cadence.record(rows=10, cost_ms=1.0)  # pass 1, emitted
        emitted = 0
        for _ in range(IDLE_CENSUS_LOG_EVERY - 2):
            if cadence.record(rows=10, cost_ms=1.0) is not None:
                emitted += 1
        self.assertEqual(emitted, 0, "the cheap passes in between are suppressed")
        got = cadence.record(rows=10, cost_ms=IDLE_POLL_CAP_MS * 10)
        self.assertIsNotNone(got, "an over-poll pass must emit whatever the count says")
        self.assertTrue(got[0])

    def test_every_count_names_its_population(self):
        cadence = IdleCensusCadence(clock=_FakeClock())
        cadence.note_agreement()
        cadence.note_agreement()
        cadence.note_disagreement()
        cadence.note_deferred(control=True)
        _, line = cadence.record(rows=7, cost_ms=1.0)
        for term in ("of 1 disagreeing", "in 3 idle pass", "agreed=2", "deferred_control=1"):
            self.assertIn(term, line, f"{term!r} missing from {line!r}")


# ---------------------------------------------------------------------------
# (2) LOOP PRIORITY
# ---------------------------------------------------------------------------


class _StubReceiver:
    """The zmq intake, reduced to the one question ``on_idle`` asks it."""

    def __init__(self, pending=False):
        self.pending = pending
        self.probes = 0

    def control_message_pending(self):
        self.probes += 1
        return self.pending


class _StubLoop:
    """The scheduler's iteration, reduced to the ordering under test.

    ``recv -> process_input_requests -> (batch | on_idle)`` is the shape of
    BOTH loop families that call ``on_idle``: ``event_loop_normal``
    (scheduler.py:2769-2800) and ``_event_loop_pp_body``
    (scheduler_pp_mixin.py:4109-5365). The stub reuses the REAL methods off
    ``Scheduler`` (unbound) so a change to either is caught here.
    """

    def __init__(self, receiver, checker, stats):
        from sglang.srt.managers.scheduler import Scheduler

        self.request_receiver = receiver
        self.invariant_checker = checker
        self._stats = stats
        self._idle_census_control_held = False
        self.handled = []
        self._hold = Scheduler._idle_census_control_hold

    def hold(self):
        return self._hold(self)

    def recv_and_process(self, messages=()):
        # the unconditional clear that lives at the top of
        # process_input_requests
        self._idle_census_control_held = False
        for m in messages:
            self.handled.append(m)
            self.request_receiver.pending = False

    def on_idle_census(self):
        allow = not self.hold()
        return self.invariant_checker._check_all_pools(
            self._stats, allow_enumeration=allow
        )


class TestControlMessageOutranksTheCensus1262(CustomTestCase):
    def setUp(self):
        import sglang.srt.managers.scheduler_components.invariant_checker as mod

        self.mod = mod
        self.spy = _CountingRead(mod.read_free_rows)
        mod.read_free_rows = self.spy
        self.addCleanup(setattr, mod, "read_free_rows", self.spy.real)
        # a pool whose counters DISAGREE, so the census would otherwise run
        self.alloc = _Alloc(1000)
        dup = self.alloc.free_pages[:21].clone()
        self.alloc.release_pages = torch.cat((self.alloc.release_pages, dup))
        self.cadence = IdleCensusCadence(clock=_FakeClock())
        self.checker = _checker(self.alloc, cadence=self.cadence)
        self.stats = _stats(self.alloc)

    def test_a_queued_message_is_handled_before_the_next_census(self):
        recv = _StubReceiver(pending=True)
        loop = _StubLoop(recv, self.checker, self.stats)

        loop.recv_and_process()  # nothing on the wire yet at the top
        loop.on_idle_census()  # message arrived mid-pass
        self.assertEqual(
            self.spy.calls,
            0,
            "the census must yield to a queued control message -- weg2t2a's "
            "WEG2-FLIP begin waited behind exactly this",
        )
        self.assertEqual(self.cadence.deferred_control, 1)
        self.assertEqual(loop.handled, [])

        loop.recv_and_process(["ReleaseMemoryOccupationReqInput"])
        self.assertEqual(loop.handled, ["ReleaseMemoryOccupationReqInput"])
        loop.on_idle_census()
        self.assertEqual(
            self.spy.calls, 1, "with the message served the census may run again"
        )

    def test_the_hold_is_not_re_entered_before_the_message_is_handled(self):
        recv = _StubReceiver(pending=True)
        loop = _StubLoop(recv, self.checker, self.stats)
        loop.recv_and_process()
        loop.on_idle_census()
        probes_after_first = recv.probes
        # a second idle pass with NO intervening process_input_requests
        loop.on_idle_census()
        self.assertEqual(self.spy.calls, 0)
        self.assertEqual(
            recv.probes,
            probes_after_first,
            "the hold is sticky: once armed it must not even re-probe until "
            "process_input_requests has run",
        )
        self.assertEqual(self.cadence.deferred_control, 2)

    def test_an_empty_wire_costs_one_probe_and_nothing_else(self):
        recv = _StubReceiver(pending=False)
        loop = _StubLoop(recv, self.checker, self.stats)
        loop.recv_and_process()
        loop.on_idle_census()
        self.assertEqual(recv.probes, 1)
        self.assertEqual(self.spy.calls, 1, "no message -> the census is unaffected")

    def test_mutant_without_the_hold_the_census_runs_on_top_of_the_message(self):
        """The pre-#1262 order, kept reachable as the red."""
        recv = _StubReceiver(pending=True)
        self.checker._check_all_pools(self.stats)  # no allow_enumeration gate
        self.assertEqual(
            self.spy.calls,
            1,
            "the mutant must show the census running while a message waits",
        )
        self.assertEqual(recv.probes, 0)

    def test_the_probe_can_never_latch_the_leak_check_off(self):
        """A probe stuck at True postpones the DIAGNOSTIC, never the CHECK.

        This is the #606 direction: an instrument that cannot answer must not
        be allowed to turn a check into silence. Here a permanently-pending
        wire still leaves the O(1) ledger running on every pass, so a genuine
        deficit is still found.
        """
        alloc = _Alloc(1000)
        alloc.alloc(400)  # 400 rows owned by nobody
        cadence = IdleCensusCadence(clock=_FakeClock())
        checker = _checker(alloc, evictable=0, cadence=cadence)
        stats = _stats(alloc, evictable=0)
        leak, msg = checker._check_full_pool(stats, allow_enumeration=False)
        self.assertTrue(
            leak,
            "a deficit must still be reported with the census deferred -- the "
            "enumerated union is <= the sum by construction, so a deficit "
            "under the cheap reading is a deficit under both. Got: " + msg,
        )
        self.assertIn("NOT CONSULTED", msg, "and the line must say why")
        self.assertEqual(
            self.spy.calls,
            0,
            "and it must reach that verdict without enumerating a single row",
        )

    def test_the_two_signs_are_treated_differently_and_that_is_the_point(self):
        """A surplus needs the census; a deficit is already decided.

        This is the asymmetry the fix rests on, stated as a test rather than
        as a comment: enumeration can only LOWER `available`, so it can close
        a surplus (#912's five specimens) and can never close a deficit
        (#832/#856). A mutant that dropped the sign test -- deferring both --
        would leave a real leak silently unreported for a whole cadence
        period, which is the one direction this fix must not buy its speed in.
        """
        cadence = IdleCensusCadence(clock=_FakeClock())

        deficit = _Alloc(1000)
        deficit.alloc(400)
        leak, msg = _checker(deficit, evictable=0, cadence=cadence)._check_full_pool(
            _stats(deficit, evictable=0), allow_enumeration=False
        )
        self.assertTrue(leak, f"deficit: {msg}")

        surplus = _Alloc(1000)
        dup = surplus.free_pages[:21].clone()
        surplus.release_pages = torch.cat((surplus.release_pages, dup))
        leak, msg = _checker(surplus, cadence=cadence)._check_full_pool(
            _stats(surplus), allow_enumeration=False
        )
        self.assertFalse(
            leak,
            "surplus: raising on the sum alone is exactly #912's false "
            f"positive -- {msg}",
        )
        self.assertIn("DEFERRED", msg)
        self.assertEqual(self.spy.calls, 0)

    def test_the_receiver_probe_is_side_effect_free_and_defaults_to_false(self):
        """A socket that cannot be polled answers False, never True.

        Answering True would be the failure this whole clause must not have:
        a probe that cannot answer silently disarming the diagnostic.
        """
        from sglang.srt.managers.scheduler_components.request_receiver import (
            SchedulerRequestReceiver,
        )

        class _NoPollSock:
            pass

        probe = SchedulerRequestReceiver.control_message_pending

        class _Bare:
            recv_from_tokenizer = _NoPollSock()
            recv_from_rpc = None

        self.assertFalse(probe(_Bare()))

        class _RaisingSock:
            def poll(self, *a, **k):
                raise RuntimeError("socket closed")

        class _Raising:
            recv_from_tokenizer = _RaisingSock()
            recv_from_rpc = None

        self.assertFalse(probe(_Raising()))

        class _ReadySock:
            def poll(self, timeout, flags):
                assert timeout == 0, "the probe must never block"
                return 1

        class _Ready:
            recv_from_tokenizer = _ReadySock()
            recv_from_rpc = None

        self.assertTrue(probe(_Ready()))


class TestTheWiringIsPresentInScheduler1262(CustomTestCase):
    """Delivery, not presence: the fix must be REACHED by on_idle."""

    def test_on_idle_gates_the_census_on_the_control_hold(self):
        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.on_idle)
        self.assertIn("_idle_census_control_hold()", src)
        self.assertIn("allow_enumeration=allow_enumeration", src)
        self.assertLess(
            src.index("_idle_census_control_hold()"),
            src.index("_check_all_pools"),
            "the hold must be evaluated BEFORE the census is asked for",
        )

    def test_process_input_requests_clears_the_hold_unconditionally(self):
        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.process_input_requests)
        head = src[: src.index("drain_recovery_request")]
        self.assertIn(
            "self._idle_census_control_held = False",
            head,
            "the clear must sit at the TOP of the one function every loop "
            "family reaches once per iteration -- otherwise the hold can "
            "outlive the message and latch the census off",
        )


if __name__ == "__main__":
    unittest.main()
