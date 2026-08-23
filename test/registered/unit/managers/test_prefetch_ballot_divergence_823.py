"""#823: a divergent TP queue head is a DURATION, and the log has to say so.

WHAT THE LATCH COST, measured rather than argued. Specimen
/spinning/evidence-816-18f/wedge_0823_055757 (boot 0516, 2026-08-23, PP=3
with --enable-phase-flip). `Scheduler._pass_uniform_reduce`'s #791b branch
latched `_prefetch_ballot_mismatch_logged` and logged ONCE PER PROCESS. Its
own comment called a divergent queue head "a deeper breakage than a divergent
prefetch verdict" -- and then recorded it exactly once. The boot's timeline:

    05:55:19  #797 void storm begins on PP1
    05:55:38  #791b PREFETCH-BALLOT digest mismatch -- ONE line per rank,
              the last this process ever emits about it. PP0 digest
              887126098, PP1 and PP2 digest 1471852626.
    05:55:53  void storm core: 7710 #797d/#798 lines in seven seconds
    05:56:13  PHASE-FLIP DONE pp_to_tp (epoch 5)
    05:56:15  flip back ABANDONED: "live slot set divergence cannot be
              repaired this round: the group's union reaches row 240831 and
              the poorest rank has only 1208"
    05:56:15  last real forward progress
    05:56:18  the three ranks each build a DIFFERENT prefill batch
              (#new-seq 1 vs 3, #cached-token 0 vs 16384, #queue-req 6 vs 3)
    05:56:17  first ADMISSION-WEDGE alarm, then every 10s for four minutes
    05:57:57  py-spy: PP0+PP1 in the spec VERIFY arm
              (eagle_worker_v2.py:2246), PP2 in the EXTEND arm (:2151); all
              three GPUs pinned at 100%, stacks frozen across three samples
    06:00:43  external SIGTERM -- the wedge never resolved itself

The single warning at 05:55:38 was the EARLIEST evidence in that whole chain,
and the latch made it one line with no duration attached. Whether the
divergence healed and returned, persisted the whole time, or worsened is not
recoverable from that log, because a latched logger reports "it healed after
N passes" and "it never healed" as the same silence.

WHAT THIS CHANGE DOES NOT DO. It does not decide anything differently. The
fall-back to the rank-local prefetch verdict is untouched, and
`prefetch_done_under_ballot(None, ...)` still returns the local verdict
byte-identically -- pinned below, because "this only changes logging" is a
claim, not a licence.

WHY THE CADENCE IS NOT "LOG EVERY PASS". The same boot emitted 7710 void
lines in seven seconds on a neighbouring path. A per-iteration warning on a
hot loop IS that failure. `should_log_mismatch_streak` reports 1, 2, 4, 8,
... up to a cap and then every cap-many passes, so a divergence that never
heals costs a handful of lines per minute. It is pure arithmetic on a
counter: no clock, no state, and therefore not itself a possible source of
divergence between ranks.
"""

import inspect
import unittest

from sglang.srt.managers import prefetch_ballot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)


class TheCadenceReportsOnsetAndBoundsTheRest(unittest.TestCase):
    def test_the_first_diverged_pass_is_always_reported(self):
        self.assertTrue(prefetch_ballot.should_log_mismatch_streak(1))

    def test_the_early_passes_are_reported_geometrically(self):
        logged = [
            s for s in range(1, 41) if prefetch_ballot.should_log_mismatch_streak(s)
        ]
        self.assertEqual(logged, [1, 2, 4, 8, 16, 32])

    def test_a_persistent_divergence_stays_bounded(self):
        """The property that matters: a divergence lasting a very long time
        must not produce a line per pass."""
        cap = prefetch_ballot.PREFETCH_BALLOT_MISMATCH_LOG_CAP
        lines = sum(
            1
            for s in range(1, 20 * cap + 1)
            if prefetch_ballot.should_log_mismatch_streak(s)
        )
        self.assertLess(
            lines,
            40,
            "a long-running divergence must cost a handful of lines, not "
            f"thousands; got {lines} over {20 * cap} passes",
        )

    def test_past_the_cap_the_cadence_is_exactly_the_cap(self):
        cap = prefetch_ballot.PREFETCH_BALLOT_MISMATCH_LOG_CAP
        self.assertTrue(prefetch_ballot.should_log_mismatch_streak(cap))
        self.assertTrue(prefetch_ballot.should_log_mismatch_streak(2 * cap))
        self.assertFalse(prefetch_ballot.should_log_mismatch_streak(cap + 1))

    def test_a_non_streak_is_never_reported(self):
        self.assertFalse(prefetch_ballot.should_log_mismatch_streak(0))
        self.assertFalse(prefetch_ballot.should_log_mismatch_streak(-3))

    def test_the_cadence_would_be_noticed_if_it_flooded(self):
        """Can-fail arm: the bound above is only worth its line count if a
        log-everything cadence would fail it."""
        cap = prefetch_ballot.PREFETCH_BALLOT_MISMATCH_LOG_CAP
        flooding = sum(1 for _ in range(1, 20 * cap + 1))
        self.assertGreater(flooding, 40)


class TheFallBackBehaviourIsUnchanged(unittest.TestCase):
    """'This only changes logging' is a claim. Pinned."""

    def test_a_void_ballot_still_yields_the_rank_local_verdict(self):
        for local in (True, False):
            self.assertEqual(
                prefetch_ballot.prefetch_done_under_ballot(local, "rid-x", None),
                local,
                "a void ballot must still fall back to the local verdict, "
                "byte-identically to the pre-#823 path",
            )

    def test_a_present_ballot_still_wins_over_the_local_verdict(self):
        ballot = {"rid-a": True, "rid-b": False}
        self.assertTrue(
            prefetch_ballot.prefetch_done_under_ballot(False, "rid-a", ballot)
        )
        self.assertFalse(
            prefetch_ballot.prefetch_done_under_ballot(True, "rid-b", ballot)
        )

    def test_a_rid_outside_the_ballot_is_still_conservatively_pending(self):
        self.assertFalse(
            prefetch_ballot.prefetch_done_under_ballot(True, "deep", {"rid-a": True})
        )


class TheStateMachineIsDrivenNotGrepped(unittest.TestCase):
    """The recovery edge, exercised for real.

    The structural class below reads source text, and a source-text probe
    cannot tell a live branch from an `elif False:` -- a mutant that disabled
    the recovery edge survived it. So the decision now lives in a pure
    function and these drive it.
    """

    def _run(self, pattern):
        streak, total, events = 0, 0, []
        for diverged in pattern:
            streak, total, event = prefetch_ballot.advance_mismatch_streak(
                streak, total, diverged
            )
            events.append(event)
        return streak, total, events

    def test_the_onset_is_reported_on_the_very_first_diverged_pass(self):
        _s, _t, events = self._run([True])
        self.assertEqual(events, [("diverged", 1)])

    def test_recovery_is_reported_and_carries_the_length_that_ended(self):
        _s, _t, events = self._run([True] * 5 + [False])
        self.assertEqual(
            events[-1],
            ("restored", 5),
            "the recovery edge must fire AND say how long the divergence "
            "lasted; that length is the whole diagnostic value",
        )

    def test_a_second_divergence_restarts_the_streak_but_not_the_total(self):
        streak, total, events = self._run([True] * 3 + [False] + [True] * 2)
        self.assertEqual(streak, 2)
        self.assertEqual(total, 5, "the process-wide total must keep counting")
        self.assertEqual(events[-1], ("diverged", 2))

    def test_agreement_that_never_broke_says_nothing(self):
        _s, total, events = self._run([False] * 10)
        self.assertEqual(events, [None] * 10)
        self.assertEqual(total, 0)

    def test_recovery_fires_exactly_once_per_divergence(self):
        _s, _t, events = self._run([True] * 3 + [False] * 4)
        restored = [e for e in events if e and e[0] == "restored"]
        self.assertEqual(len(restored), 1, f"events: {events}")

    def test_a_persistent_divergence_keeps_counting_while_staying_quiet(self):
        streak, total, events = self._run([True] * 100)
        self.assertEqual(streak, 100)
        self.assertEqual(total, 100)
        spoken = [e for e in events if e is not None]
        self.assertEqual([e[1] for e in spoken], [1, 2, 4, 8, 16, 32, 64])


class TheLatchIsGoneAndTheRecoveryEdgeExists(unittest.TestCase):
    """STRUCTURAL, reading the shipped source -- the same technique
    test_pp_admission_wraparound_never_blocks.py uses for its call site."""

    @staticmethod
    def _code_only(src):
        """Strip whole-line comments.

        The probe below asks whether a NAME still appears; the branch's own
        comment explains which latch was removed and names it, so a probe
        that cannot tell prose from a statement would report the latch as
        still present. Measured, not predicted: that is exactly how this
        file failed first.
        """
        return "\n".join(
            line for line in src.splitlines() if not line.strip().startswith("#")
        )

    def _source(self):
        from sglang.srt.managers.scheduler import Scheduler

        for name in dir(Scheduler):
            attr = getattr(Scheduler, name, None)
            if not callable(attr):
                continue
            try:
                src = inspect.getsource(attr)
            except (OSError, TypeError):
                continue
            if "PREFETCH-BALLOT digest mismatch" in src:
                return self._code_only(src)
        self.fail("could not locate the #791b ballot mismatch branch in Scheduler")

    def test_the_once_per_process_latch_is_gone(self):
        self.assertNotIn(
            "_prefetch_ballot_mismatch_logged",
            self._source(),
            "the once-per-process latch is what made a four-minute divergence "
            "look like a single event; it must not come back",
        )

    def test_the_divergence_is_counted(self):
        src = self._source()
        self.assertIn("_prefetch_ballot_mismatch_streak", src)
        self.assertIn("_prefetch_ballot_mismatch_total", src)

    def test_the_recovery_edge_is_logged(self):
        self.assertIn(
            "agreement restored",
            self._source(),
            "a divergence that heals and one that never heals are the same "
            "silence without this line, and they call for opposite responses",
        )

    def test_the_pure_state_machine_is_the_one_that_gates_the_log(self):
        """The wiring must delegate the DECISION, not re-implement it -- that
        delegation is what makes the class above able to drive it."""
        self.assertIn("advance_mismatch_streak", self._source())

    def test_the_detector_can_fail(self):
        """Can-fail arm for the structural probe itself: it must actually be
        reading a real body, not an empty string."""
        self.assertIn("PREFETCH-BALLOT digest mismatch", self._source())


if __name__ == "__main__":
    unittest.main()
