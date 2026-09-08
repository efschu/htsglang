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
"""#1268: sleep(P) only on a GROUP-UNIFORM idle fact.

THE SPECIMEN, boot weg2sb1 @ 7b60280f57 (2026-09-08). 47 flips, 292 requests,
4/4 round trips -- then three ranks answered the same quiesce differently IN
THE SAME SECOND::

    [14:29:29 PP0] Cache flushed successfully!
    [14:29:29 PP1] Cache flushed successfully!
    [14:29:29 PP2] Cache not flushed because there are pending requests.
                   #queue-req: 0, #running-req: 0,
                   not-idle because: hicache_prefetch(1: 43c9af54)

Only PP0's answer leaves the group -- it owns the HTTP entrypoint. The front
saw its own ledger at `outstanding=0 queue=0`, read PP0's 200 as P's verdict,
logged `WEG2-FLIP begin epoch=47 sleep=P wake=D` at 14:29:28,801 and commanded
`sleep(P, kv_cache)`. PP2 met the rank-side assert at 14:29:59 ->
`W29 Weg2FlipRankDisagree rank=2` on all three ranks -> SIGQUIT -> the front's
`W4 Weg2WakeRefused ... HTTP 0`.

The law worked: the group crashed together rather than compensating. The defect
is one step upstream -- a fact produced by ONE rank was consumed as the GROUP's.

WHAT THE FIX IS NOT KEYED ON, deliberately. The term that diverged on sb1 was
an orphaned HiCache storage prefetch, and an earlier reading proposed the PP
microbatch tail instead. Both are the same SHAPE and the fix must not care
which: `group_idle_verdict` reduces whatever `is_fully_idle` says, so a future
clause that diverges is covered without a new mechanism.
"""

import unittest
from types import SimpleNamespace


from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

_NONE = 1 << 20


def _reduce_min(votes):
    """What `all_reduce(MIN)` does to the per-rank vote vectors."""
    return [min(v[i] for v in votes) for i in range(2)]


def _vote(rank, idle):
    return [1 if idle else 0, _NONE if idle else max(rank, 0)]


class TheEncodingIsTheVerdict(CustomTestCase):
    """The pure part: two ints, one MIN reduce, both answers."""

    def test_all_idle_reduces_to_idle(self):
        out = _reduce_min([_vote(0, True), _vote(1, True), _vote(2, True)])
        self.assertEqual(out[0], 1)
        self.assertGreaterEqual(out[1], _NONE, "nobody blocks -> the sentinel survives")

    def test_the_sb1_shape_reduces_to_not_idle_and_names_pp2(self):
        """RED-FIRST, as the specimen: PP0 and PP1 idle, PP2 not.

        The pre-fix answer is PP0's alone -- `_vote(0, True)[0] == 1`, i.e.
        "idle" -- and that is exactly the fact that killed the group. The reduce
        is what turns three answers into one.
        """
        votes = [_vote(0, True), _vote(1, True), _vote(2, False)]
        self.assertEqual(votes[0][0], 1, "the entrypoint alone still says idle")
        out = _reduce_min(votes)
        self.assertEqual(out[0], 0, "the GROUP must not be idle")
        self.assertEqual(out[1], 2, "and the refusal must name rank 2")

    def test_the_lowest_blocking_rank_is_named(self):
        out = _reduce_min([_vote(0, False), _vote(1, True), _vote(2, False)])
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], 0)

    def test_a_blocking_rank_zero_is_not_confused_with_the_sentinel(self):
        """Rank 0 encodes as 0, which MIN also produces for `idle=0` -- the two
        elements must stay independent or a blocking entrypoint would read as
        'nobody blocked'."""
        out = _reduce_min([_vote(0, False), _vote(1, True)])
        self.assertEqual((out[0], out[1]), (0, 0))
        out_idle = _reduce_min([_vote(0, True), _vote(1, True)])
        self.assertNotEqual(out_idle[1], 0)


class TheVerdictOnAScheduler(CustomTestCase):
    """`group_idle_verdict` bound to a stand-in, with no torch.distributed."""

    def _sched(self, idle, blockers, cpu_group=None):
        s = SimpleNamespace(
            is_fully_idle=lambda: idle,
            idle_blockers=lambda: list(blockers),
            world_group=SimpleNamespace(cpu_group=cpu_group),
            collective_timeout_s=5.0,
        )
        # #1268 fix 1b: a real Scheduler always has this; the
        # stand-in must too, or it tests a shape that cannot exist.
        s.pp_group = SimpleNamespace(is_last_rank=True, is_first_rank=True)
        # #1268 fix 1c: the verdict is decided from `ps`, not from a cpu_group
        # -- there is no collective on this path any more (see the WITHDRAWN
        # blocks in `group_idle_verdict`'s docstring).
        s.ps = SimpleNamespace(pp_rank=0, pp_size=1)
        s.group_idle_verdict = Scheduler.group_idle_verdict.__get__(s)
        return s

    def test_no_group_returns_this_ranks_own_answer_unchanged(self):
        """Stock path: a single-rank engine must be byte-identical."""
        ok, detail = self._sched(True, []).group_idle_verdict()
        self.assertTrue(ok)
        self.assertIn("single-rank", detail)

    def test_a_busy_single_rank_is_still_busy(self):
        ok, detail = self._sched(False, ["hicache_prefetch(1: 43c9af54)"]).group_idle_verdict()
        self.assertFalse(ok)
        self.assertIn("43c9af54", detail)

    def test_an_undecided_verdict_refuses_rather_than_assumes_idle(self):
        """UNDECIDED IS NOT IDLE -- restated for the shape that replaced the
        reduce.

        ADAPTED, not deleted: the old body built an unreadable `cpu_group` and
        asserted the reduce refused rather than falling through to this rank's
        own answer. Fix 1c removes the collective entirely, so "the reduce
        could not be taken" is no longer a reachable state; the state that
        replaced it is "the lap has not come home yet", and it must refuse for
        exactly the same reason.
        """
        s = self._sched(True, [])
        s.ps = SimpleNamespace(pp_rank=0, pp_size=3)
        ok, detail = s.group_idle_verdict()
        self.assertFalse(ok, "an undecided group must not answer 'idle'")
        self.assertIn("PENDING", detail)
        self.assertIn("UNDECIDED IS NOT IDLE", detail)


class TheWiring(CustomTestCase):
    def test_flush_cache_consults_the_group_not_the_rank(self):
        import inspect

        src = inspect.getsource(Scheduler.flush_cache)
        self.assertIn("group_idle_verdict()", src)
        self.assertNotIn("if self.is_fully_idle():", src)

    def test_the_front_names_it_a_group_verdict(self):
        import inspect

        from sglang.srt.weg2 import front

        src = inspect.getsource(front.Front.quiesce)
        self.assertIn("the GROUP's is_fully_idle", src)
        flip = inspect.getsource(front.Front.flip)
        self.assertIn("W3 Weg2DrainWitnessDisagreement", flip)
        self.assertIn("reduced over every rank", flip)

    def test_the_rank_side_assert_is_untouched(self):
        """It is the FENCE. The fix removes the reason it fires; it must not
        remove the assert, or a future divergence dies silently instead."""
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater

        src = inspect.getsource(weight_updater.SchedulerWeightUpdaterManager.release_memory_occupation)
        self.assertIn(
            "release_memory_occupation should be called only when server is idle", src
        )


class TheSb2RingShapeIsWithdrawn(CustomTestCase):
    """FIX 1B'S FIVE TESTS ARE REPLACED, AND THIS SAYS WHY RATHER THAN VANISHING.

    What stood here asserted that `_weg2_commit_owed_forward` delivers the
    forwarded RPC to rank+1 before the blocking reduce, and therefore that the
    reduce terminates (#631 clause (ii)). Every one of those five was GREEN,
    and boot weg2sb3 deadlocked anyway, identically to sb2.

    THE TESTS WERE GREEN BECAUSE THE MODEL WAS WRONG, not because the code was
    right: the ring they built made rank k+1 reachable exactly when rank k
    committed its request-chain forward. The six py-spy stacks say the
    followers were never in the request-chain receive at all -- they were in
    `_pp_recv_dict_from_prev_stage`, the hidden-states/proxy channel, waiting
    for a send PP0 issues LATER IN ITS OWN PASS. A model that cannot express
    "the peer is blocked on a different channel" cannot fail on the defect,
    and #631's own docstring had already named that shape as `variant B`.

    So the helper is DELETED (it has no work to do once nothing blocks) and
    the topology is modelled honestly in
    `test_weg2_idle_vote_lap_1268.py::build_ring`, where a follower becomes
    reachable only when its predecessor FINISHES A PASS.
    """

    def test_the_fix1b_helper_is_deleted(self):
        self.assertFalse(
            hasattr(Scheduler, "_weg2_commit_owed_forward"),
            "fix 1b's helper must not survive the shape it existed to support",
        )

    def test_the_answer_path_takes_no_collective(self):
        """The one property all five old tests were trying to buy."""
        import inspect

        src = inspect.getsource(Scheduler.group_idle_verdict)
        for banned in ("all_reduce", "bounded_wait", "barrier", "all_gather"):
            self.assertNotIn(banned, src, f"{banned} is back on the answer path")

    def test_the_replacement_ring_test_exists_and_models_the_other_channel(self):
        import os

        here = os.path.dirname(os.path.abspath(__file__))
        repl = os.path.join(here, "test_weg2_idle_vote_lap_1268.py")
        self.assertTrue(os.path.exists(repl), "the replacement suite is missing")
        body = open(repl, encoding="utf-8").read()
        self.assertIn("HIDDEN-STATES", body)
        self.assertIn("FINISHED ITS PASS", body)


class ParticipationIsAFact(CustomTestCase):
    """FIX 1b (2)+(3): the reduce counts who arrived and says so."""

    @staticmethod
    def _reduce_sum(votes):
        return [sum(v[i] for v in votes) for i in range(3)]

    @staticmethod
    def _vote(rank, idle):
        return [1 if idle else 0, 1, 0 if idle else (1 << rank)]

    def test_all_present_and_idle(self):
        out = self._reduce_sum([self._vote(r, True) for r in range(3)])
        self.assertEqual(out, [3, 3, 0])

    def test_the_mask_names_every_blocking_rank(self):
        out = self._reduce_sum(
            [self._vote(0, True), self._vote(1, False), self._vote(2, False)]
        )
        n_idle, n_present, mask = out
        self.assertEqual((n_idle, n_present), (1, 3))
        self.assertEqual([r for r in range(3) if mask & (1 << r)], [1, 2])

    def test_a_short_reduce_is_detectable_and_must_refuse(self):
        """The sb2 shape: a rank never arrived. SUM makes that visible; MIN
        could not -- three ranks contributing -1 reduce to -1, not -3."""
        out = self._reduce_sum([self._vote(0, True), self._vote(1, True)])
        self.assertEqual(out[1], 2, "participation=2/3 is a FACT, not a guess")
        self.assertNotEqual(out[1], 3)

    def test_the_verdict_refuses_on_short_participation(self):
        """ADAPTED to fix 1c's carrier: the same fact, now counted from the
        slots on the object instead of from a reduce's output vector."""
        from sglang.srt.managers.weg2_idle_vote import Weg2IdleVoteReq, attach_slot, tally

        vote = Weg2IdleVoteReq(epoch=1, origin=0, world=3)
        attach_slot(vote, 0, True, "none")
        attach_slot(vote, 1, True, "none")
        t = tally(vote)
        self.assertFalse(t.complete)
        self.assertFalse(t.idle, "a subset must never answer for the whole")
        self.assertEqual(list(t.missing_ranks), [2])
        import inspect

        src = inspect.getsource(Scheduler.group_idle_verdict)
        self.assertIn("GROUP VERDICT SHORT", src)

    def test_every_rank_emits_the_verdict_line(self):
        """sb2's reduce printed NOTHING on any rank. The line now has a REAL
        epoch: `weg2_flip_epoch` had one reader and zero writers, so every
        boot's verdict line printed the literal `epoch=?`."""
        from sglang.srt.managers.weg2_idle_vote import Weg2IdleVoteReq, attach_slot, log_verdict

        vote = Weg2IdleVoteReq(epoch=42, origin=0, world=3)
        for r in range(3):
            attach_slot(vote, r, True, "none")
        line = log_verdict(vote, 1)
        self.assertIn("WEG2-P-IDLE-VERDICT", line)
        self.assertIn("epoch=42", line)
        self.assertNotIn("epoch=?", line)
        for field in ("idle=", "blocking_rank=", "blockers=[", "participation="):
            self.assertIn(field, line)

    def test_no_bare_refusal_reaches_the_front(self):
        """ADAPTED: sb3 stopped the front on the literal `'TimeoutError: '`.
        There is no gloo timeout on this path any more, so the property is
        restated against the refusal that replaced it -- it must name the
        ranks whose slot is missing."""
        from sglang.srt.managers.weg2_idle_vote import Weg2IdleVoteReq, refusal_detail, tally

        vote = Weg2IdleVoteReq(epoch=5, origin=0, world=3)
        detail = refusal_detail(vote, tally(vote), "GROUP VERDICT SHORT:")
        self.assertIn("[0, 1, 2]", detail)
        self.assertNotEqual(detail.strip(), "TimeoutError:")


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
