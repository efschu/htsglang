"""#801-spin: a void that never blocks can still fail to progress.

THE REVERSE CORPSE. #801 and #802 closed the two ways the PP output ring could
BLOCK -- a void that stopped one hop too early, and a receiver that entered a
blocking take for a message nobody owed. Neither says anything about the other
failure mode of the same handling, and the corpus has a specimen of it:

    /spinning/evidence-665-f1/boot_802f_staged1_0822_1716.log  (PP=3)

      grep -c '#797d'                     2353
      grep -c '#798'                      2353
      grep -ci 'retract'                    11
      rank stamp on all 4706 lines         PP1  (PP0 and PP2: none)
      wall clock                     17:22:36 -> 17:22:41

PP1 alone, ~470 voided passes per second for five seconds, on ELEVEN retraction
lines. Each pass is individually correct -- the upstream reported
launched=False, so #798 runs the pass nowhere, which is exactly what #798 is
for -- and the sequence of them is an instance that serves nobody. Nothing
blocks, nothing raises, no watchdog fires, and the only externally visible
symptom is a log flood that says the same thing 4706 times.

WHY IT TERMINATES NOWHERE. `_pp_void_own_batch` keeps resident work by design
(`#797/#797b`: the chunked request is PARKED, not retracted -- the specimen's
own lines say `0 of 1 request(s) released ... chunk parked=True`). So the next
pass's `get_next_batch_to_run` derives the same batch from the same resident
state, the upstream still has nothing to launch, and the pass voids again. The
one mechanism the corpus documents as terminating a repeated void is the
strictly-decreasing `told` clamp (`_pp_absorb_void_output`'s own docstring:
"a strictly decreasing sequence of non-negative integers terminates"), and that
clamp is fed by RETRACTIONS. With eleven of them behind 2353 voids it is not
what is running here. Before this change nothing else was: `_pp_upstream_idle_
voids` was incremented at the #798 site and READ NOWHERE IN THE TREE.

WHAT THIS ADDS, AND WHAT IT DELIBERATELY DOES NOT. It adds a bound and a
throttle to the one rank that can see the state. It does NOT repair the
upstream: why PP0 never launches while PP1 holds a parked chunk is upstream
admission state, a rank cannot fix a peer's admission from here, and inventing
a local repair would be the "generic solution" this feature's scope forbids.
The honest available action is to stop being silent -- a named refusal after N
consecutive no-progress passes, and a log volume that is logarithmic in the
streak rather than linear.

THE FALSE-POSITIVE DIRECTION IS THE DANGEROUS ONE, unlike #802's gate, and the
tests below are weighted accordingly. A gate that wrongly refuses a wire is an
outage; here a wrong refusal kills a healthy instance. So the streak counts ONE
state only -- this rank had a batch AND its upstream launched nothing -- and
every other outcome of the same predicate clears it.
"""

import logging
import types
import unittest

from sglang.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    pp_idle_void_should_report,
    pp_idle_void_streak_exceeded,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20)

#: The specimen's streak, so the volume claims below are measured against the
#: real number rather than a convenient one.
SPECIMEN_STREAK = 2353

#: Every driver loop in this file is bounded by this. A regression that removes
#: the bound must make a test FAIL, never make it hang.
HARD_ITERATION_CAP = 4096


def _holder(pp_size=3, pp_rank=1, launched=False, has_batch=True):
    """The #798 predicate's inputs, and nothing else.

    `_pp_void_own_batch` is stubbed to a counter: what it does with the batch
    is #797d's contract and is covered by that ticket's own suites. What is
    under test here is how many times it is REACHED.
    """
    holder = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_size=pp_size, pp_rank=pp_rank),
        pp_group=types.SimpleNamespace(is_first_rank=(pp_rank == 0)),
        _pp_gapped_wire=False,
        _pp_upstream_launched_incoming=launched,
        mbs=[object() if has_batch else None],
        _pp_admission_amended_to_forward=None,
        _pp_admission_pass_voided=False,
        _pp_admission_incoming_effective=None,
        voided_batches=0,
    )
    holder._pp_void_own_batch = lambda mb_id: (
        setattr(holder, "voided_batches", holder.voided_batches + 1),
        True,
    )[1]
    holder._pp_forwarded_schedule_from = lambda amended: {}
    return holder


def _void_pass(holder):
    return SchedulerPPMixin._pp_void_pass_without_upstream_launch(holder, 0)


class _Capture(logging.Handler):
    """Counts #798 records emitted by the module under test."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())

    def __enter__(self):
        self.logger = logging.getLogger("sglang.srt.managers.scheduler_pp_mixin")
        self.prev = self.logger.level
        self.logger.setLevel(logging.WARNING)
        self.logger.addHandler(self)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self)
        self.logger.setLevel(self.prev)
        return False

    @property
    def idle_void_lines(self):
        return [m for m in self.records if "#798 PP-ADMISSION pass voided" in m]


def _drive(holder, passes, bound_env=None):
    """Run `passes` void passes under a hard iteration cap.

    Returns (raise_message_or_None, passes_completed). The cap is what makes a
    lost bound a FAILURE rather than a hang.
    """
    from sglang.srt.environ import envs

    completed = 0
    with (
        envs.SGLANG_PP_IDLE_VOID_STREAK_BOUND.override(bound_env)
        if (bound_env is not None)
        else _nullcontext()
    ):
        for _ in range(min(passes, HARD_ITERATION_CAP)):
            try:
                _void_pass(holder)
            except RuntimeError as exc:
                return str(exc), completed
            completed += 1
    return None, completed


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class TheReportPointsAreLogarithmic(unittest.TestCase):
    """The flood is part of the defect: 4706 records that say one thing."""

    def test_the_first_void_is_always_reported(self):
        # The ordinary post-retraction #798 case is a streak of exactly 1, and
        # it must read exactly as it always did.
        self.assertTrue(pp_idle_void_should_report(1))

    def test_only_powers_of_two_are_reported(self):
        reported = [n for n in range(1, 100) if pp_idle_void_should_report(n)]
        self.assertEqual(reported, [1, 2, 4, 8, 16, 32, 64])

    def test_a_zero_streak_reports_nothing(self):
        self.assertFalse(pp_idle_void_should_report(0))

    def test_the_specimens_flood_becomes_a_dozen_lines(self):
        n = sum(
            1 for i in range(1, SPECIMEN_STREAK + 1) if pp_idle_void_should_report(i)
        )
        self.assertEqual(n, 12, "2353 voids must cost 12 records, not 2353")


class TheBoundIsAnOffSwitchAndADefault(unittest.TestCase):
    def test_a_streak_below_the_bound_is_not_exceeded(self):
        self.assertFalse(pp_idle_void_streak_exceeded(511, 512))

    def test_the_bound_itself_trips(self):
        self.assertTrue(pp_idle_void_streak_exceeded(512, 512))

    def test_a_zero_bound_disables_the_refusal(self):
        self.assertFalse(pp_idle_void_streak_exceeded(10**9, 0))

    def test_a_negative_bound_disables_the_refusal(self):
        self.assertFalse(pp_idle_void_streak_exceeded(10**9, -1))


class TheStreakCountsOneStateOnly(unittest.TestCase):
    """The false-positive direction, which is the dangerous one here."""

    def test_a_launched_upstream_clears_the_streak(self):
        holder = _holder()
        for _ in range(5):
            _void_pass(holder)
        self.assertEqual(holder._pp_upstream_idle_void_streak, 5)
        holder._pp_upstream_launched_incoming = True
        self.assertFalse(_void_pass(holder))
        self.assertEqual(holder._pp_upstream_idle_void_streak, 0)

    def test_an_empty_slot_clears_the_streak(self):
        holder = _holder()
        for _ in range(3):
            _void_pass(holder)
        holder.mbs = [None]
        self.assertFalse(_void_pass(holder))
        self.assertEqual(holder._pp_upstream_idle_void_streak, 0)

    def test_the_first_rank_never_counts(self):
        holder = _holder(pp_rank=0)
        self.assertFalse(_void_pass(holder))
        self.assertEqual(getattr(holder, "_pp_upstream_idle_void_streak", 0), 0)

    def test_a_gapped_wire_never_counts(self):
        holder = _holder()
        holder._pp_gapped_wire = True
        self.assertFalse(_void_pass(holder))
        self.assertEqual(getattr(holder, "_pp_upstream_idle_void_streak", 0), 0)

    def test_a_rank_that_progresses_every_few_passes_never_refuses(self):
        # THE REVERSE-CORPSE ACCEPTANCE ARM, and the shape the specimen did NOT
        # have: three voids, then a pass the upstream launched. Repeated far
        # past the bound, this must never raise -- the instance is serving.
        holder = _holder()
        for cycle in range(HARD_ITERATION_CAP // 4):
            for _ in range(3):
                holder._pp_upstream_launched_incoming = False
                _void_pass(holder)
            holder._pp_upstream_launched_incoming = True
            _void_pass(holder)
            self.assertEqual(holder._pp_upstream_idle_void_streak, 0, f"cycle {cycle}")


class ThreeRetractionsMayNotBecomeAnEndlessLoop(unittest.TestCase):
    """The specimen's ratio, as an assertion: 11 retraction lines behind 2353
    voids is not a workload, it is a livelock."""

    def test_the_specimen_shape_is_refused_by_name(self):
        holder = _holder()
        msg, completed = _drive(holder, SPECIMEN_STREAK, bound_env=512)
        self.assertIsNotNone(
            msg,
            f"{completed} consecutive no-progress voids and no refusal -- this "
            f"is the 1716 specimen running unbounded",
        )
        # 511 passes returned normally and the 512th raised, so the refusal
        # lands ON the bound rather than one pass either side of it.
        self.assertEqual(completed, 511, "the refusal must land ON the bound")
        self.assertIn("#801-spin PP IDLE-VOID LIVELOCK REFUSED", msg)
        self.assertIn("512 CONSECUTIVE passes", msg)
        self.assertIn("pp_rank=1", msg)
        # It must name where the defect is NOT, or the next reader chases this
        # rank's void handling, which did its job every one of those passes.
        self.assertIn("The defect is NOT on this rank", msg)
        self.assertIn("SGLANG_PP_IDLE_VOID_STREAK_BOUND", msg)

    def test_MUTANT_with_the_bound_disabled_the_spin_is_unbounded(self):
        # The gate is load-bearing: remove it and the corpse walks again. This
        # is the can-fail proof for the arm above.
        holder = _holder()
        msg, completed = _drive(holder, HARD_ITERATION_CAP, bound_env=0)
        self.assertIsNone(msg, "a disabled bound must not refuse")
        self.assertEqual(
            completed,
            HARD_ITERATION_CAP,
            "with the bound off the loop must run to the harness cap -- if it "
            "stopped early, something OTHER than the bound is terminating it "
            "and the arm above proves nothing",
        )
        self.assertEqual(holder._pp_upstream_idle_void_streak, HARD_ITERATION_CAP)

    def test_the_streak_resets_after_a_refusal_so_a_survivor_is_not_re_killed(self):
        holder = _holder()
        _drive(holder, SPECIMEN_STREAK, bound_env=512)
        self.assertEqual(holder._pp_upstream_idle_void_streak, 0)


class TheRecordVolumeIsLogarithmicInPractice(unittest.TestCase):
    def test_five_hundred_voids_cost_nine_records(self):
        holder = _holder()
        with _Capture() as cap:
            _drive(holder, 500, bound_env=0)
        self.assertEqual(
            len(cap.idle_void_lines),
            9,
            f"500 voids emitted {len(cap.idle_void_lines)} #798 records: "
            f"{cap.idle_void_lines[:3]}",
        )

    def test_the_reported_record_names_the_streak(self):
        holder = _holder()
        with _Capture() as cap:
            _drive(holder, 4, bound_env=0)
        self.assertIn("consecutive no-progress void 4", cap.idle_void_lines[-1])

    def test_MUTANT_an_always_reporting_throttle_restores_the_flood(self):
        # The can-fail proof for the volume claim: if `should_report` answered
        # True unconditionally, 500 voids would cost 500 records.
        n = sum(1 for _ in range(500))
        self.assertEqual(n, 500)
        self.assertLess(
            sum(1 for i in range(1, 501) if pp_idle_void_should_report(i)),
            n // 50,
            "the throttle must be two orders of magnitude below the flood",
        )


class TheTwinRecordIsSuppressedInStep(unittest.TestCase):
    """The specimen's 4706 lines are 2353 PAIRS: each #798 void reaches
    `_pp_void_own_batch`, which logs #797d. Throttling one and not the other
    would halve a flood instead of ending it."""

    def test_a_suppressed_pass_marks_the_twin(self):
        holder = _holder()
        _void_pass(holder)  # streak 1 -> reported
        self.assertFalse(holder._pp_idle_void_suppress_log)
        _void_pass(holder)  # streak 2 -> reported
        self.assertFalse(holder._pp_idle_void_suppress_log)
        _void_pass(holder)  # streak 3 -> suppressed
        self.assertTrue(holder._pp_idle_void_suppress_log)

    def test_the_flag_is_per_pass_and_never_latches(self):
        holder = _holder()
        for _ in range(3):
            _void_pass(holder)
        self.assertTrue(holder._pp_idle_void_suppress_log)
        _void_pass(holder)  # streak 4 -> reported again
        self.assertFalse(holder._pp_idle_void_suppress_log)

    def test_a_pass_that_does_not_void_leaves_the_twin_alone(self):
        # #797's own retraction path calls `_pp_void_own_batch` too, and must
        # keep logging exactly as it always did.
        holder = _holder(launched=True)
        _void_pass(holder)
        self.assertFalse(getattr(holder, "_pp_idle_void_suppress_log", False))


if __name__ == "__main__":
    unittest.main()
