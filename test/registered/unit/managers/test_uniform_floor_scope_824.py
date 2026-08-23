"""#824: the uniformity floors' COVERAGE is a state that changes, so the log
has to report changes, not the first sighting.

THE ROOT THIS SERVES (COORD-strand16f-801-build.md, B.10). All three
rank-uniformity floors -- evict, host, mamba -- are published from one
MIN-reduce over ``tp_cpu_group`` in ``Scheduler._update_uniform_pool_budget``.
When that group has one member the reduce is a no-op and every floor switches
OFF. `scheduler.py`'s own comment at that site states the consequence and
calls it "the measured cause of a pipeline deadlock":

    "With pp_size>1 the ranks that must agree are NOT in this reduce group."

And #616g's own docstring states the chain that follows from unfloored
eviction: the radix trees stop being replicas, `match_prefix` returns a
rank-dependent prefix, and every per-layer TP all_reduce of that forward is
entered with a rank-dependent token count.

SPECIMEN /spinning/evidence-816-18f/wedge_0823_055757 (boot 0516, PP=3,
--enable-phase-flip). The boot log carries that scope line exactly three
times, once per rank, every one of them:

    UNIFORM-FLOOR SCOPE: tp_cpu_group world=1 -> floors OFF
    (evict/host/mamba). pp_size=3 tp_size=1.

Nineteen seconds later the TP replicas' queue-head digests had parted
(#791b); by 05:56:18 the three ranks were building prefill batches with
#cached-token 0 against 16384 -- a rank-dependent prefix match, the exact
fingerprint #616g predicts; by 05:57:57 two ranks were in the spec verify arm
and one in the extend arm with all three GPUs pinned at 100%.

WHY A LATCH WAS THE WRONG SHAPE HERE, specifically. Under a phase flip the
scope is not a boot constant: `phase_flip_runtime` rebuilds the TP group per
phase (``want_tp_size = n if tp_phase else 1``), so the floors are ON through
the TP decode phase and OFF through the PP prefill phase. Boot 0516 completed
FOUR cutovers in 55 seconds. A once-per-process report cannot distinguish
"off for the whole run" from "off for the prefill half of every cutover" --
different situations with different fixes -- and it never reports coverage
coming BACK at all, though a gap that closes and a gap that never closes call
for opposite responses.

THIS ENFORCES NOTHING, and that boundary is deliberate. The reduce group is
not widened, no admission or eviction decision changes, and the floors are
exactly as absent as they were. Closing the gap means changing what the group
agrees on, which needs a metal lane (B.10's 18-lane ticket). This only makes
the gap legible while that is decided.
"""

import unittest

from sglang.srt.managers import uniform_floor_scope as scope
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)


class TheScopePredicate(unittest.TestCase):
    def test_a_singleton_group_cannot_constrain_anything(self):
        self.assertEqual(scope.scope_for_world(1), scope.SCOPE_OFF)

    def test_no_group_at_all_reads_off(self):
        self.assertEqual(scope.scope_for_world(None), scope.SCOPE_OFF)

    def test_a_real_group_reads_on(self):
        self.assertEqual(scope.scope_for_world(2), scope.SCOPE_ON)
        self.assertEqual(scope.scope_for_world(3), scope.SCOPE_ON)

    def test_the_specimen_configuration_reads_off(self):
        """Boot 0516 logged `tp_cpu_group world=1` on all three ranks."""
        self.assertEqual(scope.scope_for_world(1), scope.SCOPE_OFF)


class TheFirstObservationIsAlwaysReported(unittest.TestCase):
    def test_a_boot_states_its_starting_coverage_when_off(self):
        _s, event = scope.scope_transition(None, 1)
        self.assertEqual(event, scope.SCOPE_OFF)

    def test_a_boot_states_its_starting_coverage_when_on(self):
        _s, event = scope.scope_transition(None, 3)
        self.assertEqual(event, scope.SCOPE_ON)


class TheSteadyStateIsQuiet(unittest.TestCase):
    def test_an_unchanged_scope_says_nothing(self):
        state, events = None, []
        for _ in range(200):
            state, event = scope.scope_transition(state, 1)
            events.append(event)
        self.assertEqual(events[0], scope.SCOPE_OFF)
        self.assertEqual(
            events[1:],
            [None] * 199,
            "an unchanged scope must not produce a line per iteration -- this "
            "sits on the per-iteration scheduler path",
        )


class ACutoverIsVisibleInBothDirections(unittest.TestCase):
    """The half a latch can never report."""

    def _drive(self, worlds):
        state, events = None, []
        for w in worlds:
            state, event = scope.scope_transition(state, w)
            events.append(event)
        return events

    def test_coverage_going_away_is_reported(self):
        events = self._drive([3, 3, 1])
        self.assertEqual(events, [scope.SCOPE_ON, None, scope.SCOPE_OFF])

    def test_coverage_coming_back_is_reported(self):
        events = self._drive([1, 1, 3])
        self.assertEqual(events, [scope.SCOPE_OFF, None, scope.SCOPE_ON])

    def test_the_specimen_four_cutovers_produce_eight_reports(self):
        """Boot 0516 completed four cutovers in 55 s. Under the latch that is
        ONE line; every transition must be visible instead."""
        worlds = []
        for _ in range(4):
            worlds += [1] * 5 + [3] * 5
        events = [e for e in self._drive(worlds) if e is not None]
        self.assertEqual(
            events,
            [scope.SCOPE_OFF, scope.SCOPE_ON] * 4,
            f"expected every cutover in both directions, got {events}",
        )

    def test_a_latched_reporter_would_fail_this(self):
        """Can-fail arm: the assertion above is only worth its line count if
        the old once-ever shape would actually miss it."""
        worlds = []
        for _ in range(4):
            worlds += [1] * 5 + [3] * 5
        latched, seen = [], False
        for w in worlds:
            if not seen:
                seen = True
                latched.append(scope.scope_for_world(w))
        self.assertEqual(
            latched,
            [scope.SCOPE_OFF],
            "the pre-#824 shape reports exactly one line for four cutovers",
        )


class TheWiringDelegatesTheDecision(unittest.TestCase):
    """The scheduler must call the pure function rather than re-implement the
    transition inline -- that delegation is what lets the classes above drive
    the real decision instead of grepping source for a branch."""

    def _source(self):
        import inspect

        from sglang.srt.managers import uniform_floor_scope as m

        return inspect.getsource(m.report_scope)

    def test_the_reporter_delegates_to_the_pure_transition(self):
        self.assertIn("scope_transition(", self._source())

    def test_the_reporter_can_say_floors_are_on(self):
        self.assertIn("floors ON", self._source())

    def test_the_reporter_can_still_say_floors_are_off(self):
        self.assertIn("floors OFF", self._source())

    def test_both_branches_of_the_budget_site_report_their_scope(self):
        """The reporter is only useful where it is CALLED, and there are two
        call sites: the OFF branch before its early return, and the ON path
        after it. Measured, not predicted: a mutant that replaced the ON-side
        call with `pass` left every other test in this file green, because
        they all read the reporter's own body and never its callers.

        LIMIT, STATED: this counts call sites in source, so it cannot prove
        either one is REACHED -- driving `_update_uniform_pool_budget` needs a
        live process group and an allocator. The mutant above is what shows
        the probe discriminates at all.
        """
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._update_uniform_pool_budget)
        code = "\n".join(
            line for line in src.splitlines() if not line.strip().startswith("#")
        )
        self.assertEqual(
            code.count("uniform_floor_scope.report_scope("),
            2,
            "both the floors-OFF branch and the floors-ON path must report, "
            "or a cutover is only half visible",
        )

    def test_the_once_per_process_latch_is_gone(self):
        code = "\n".join(
            line
            for line in self._source().splitlines()
            if not line.strip().startswith("#")
        )
        self.assertNotIn("_uniform_floor_scope_logged", code)


if __name__ == "__main__":
    unittest.main()
