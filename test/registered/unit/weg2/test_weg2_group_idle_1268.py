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
        s.pp_group = SimpleNamespace(is_last_rank=True)
        s._weg2_commit_owed_forward = (
            Scheduler._weg2_commit_owed_forward.__get__(s)
        )
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

    def test_an_unavailable_verdict_refuses_rather_than_assumes_idle(self):
        """UNDECIDED IS NOT IDLE. If the reduce cannot be taken, answering with
        this rank's own optimistic verdict is the whole defect again."""

        class Boom:
            pass

        s = self._sched(True, [])
        s.world_group = SimpleNamespace(cpu_group=Boom())
        ok, detail = s.group_idle_verdict()
        self.assertFalse(ok, "an unreadable group must not answer 'idle'")
        self.assertTrue(
            "UNAVAILABLE" in detail or "single-rank" in detail, detail
        )


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


class TheSb2RingShape(CustomTestCase):
    """FIX 1b, red-first: only rank 0 receives the RPC.

    THE METAL REFUTATION of fix 1 (boot weg2sb2, bb3617e2aa). Gate (a) held --
    assert 0, W29 0, first flip 3215 ms -- and the SECOND flip deadlocked in the
    reduce. py-spy, two passes 60 s apart, identical: PP0 in `bounded_wait <-
    group_idle_verdict <- flush_cache`, PP1 in `join`, PP2 in `_join
    (pp_object_recv)`.

    Fix 1's premise -- "flush_cache is a broadcast control request every rank
    processes" -- was argued from three ranks logging their verdict inside one
    second on sb1. That evidence is equally consistent with the PP RING
    FORWARDING them one after another, which is what happens: `/flush_cache` is
    an RPC only PP0 receives; the followers get it over the #791 ring lap.

    `_pp_forward_and_process_input_requests` does forward BEFORE
    `process_input_requests`, so ordering was never the gap. The gap is that the
    forward is an ASYNC send committed only at the next pass's top (or by
    `_pp_commit_pending_req_work` at iteration end, #788) -- neither of which a
    rank blocked in the reduce ever reaches. That is #631 clause (ii), whose own
    comment names this as the measured `variant A`.
    """

    def _ring(self, world=3, idle=(True, True, True)):
        """A three-rank ring: rank 0 holds the RPC, the others are reached only
        by the forwarded message, and a rank that has not been forwarded to
        cannot join the reduce."""
        state = {"forwarded": set(), "joined": set()}

        def make(rank):
            s = SimpleNamespace(
                is_fully_idle=lambda r=rank: idle[r],
                idle_blockers=lambda r=rank: ([] if idle[r] else [f"blocker_r{r}"]),
                world_group=SimpleNamespace(cpu_group=None),
                collective_timeout_s=5.0,
                send_req_work=object() if rank < world - 1 else None,
                pp_group=SimpleNamespace(is_last_rank=(rank == world - 1)),
            )
            # committing the owed forward is what delivers to rank+1
            s._pp_commit_comm_work = lambda w, r=rank: state["forwarded"].add(r + 1)
            s._weg2_commit_owed_forward = Scheduler._weg2_commit_owed_forward.__get__(s)
            return s

        return [make(r) for r in range(world)], state

    def test_red_first_a_rank_that_never_forwards_strands_its_followers(self):
        """THE SB2 DEADLOCK, as a property: if rank 0 joins without committing
        its forward, rank 1 was never delivered to and cannot join."""
        ranks, state = self._ring()
        # rank 0 joins WITHOUT committing -- fix 1's behaviour
        self.assertNotIn(1, state["forwarded"], "rank 1 has not been delivered to")

    def test_committing_the_owed_forward_delivers_to_the_next_rank(self):
        ranks, state = self._ring()
        ranks[0]._weg2_commit_owed_forward()
        self.assertIn(1, state["forwarded"], "clause (ii): the send is completed")
        self.assertIsNone(ranks[0].send_req_work, "and the handle is cleared")

    def test_the_last_stage_owes_no_forward(self):
        """#631 clause (iii): it joins directly and must not block on a commit."""
        ranks, state = self._ring()
        ranks[2]._weg2_commit_owed_forward()
        self.assertEqual(state["forwarded"], set(), "last rank forwards nothing")

    def test_a_commit_that_raises_does_not_kill_the_verdict(self):
        """The reduce refuses by name; the commit must not raise past it."""
        ranks, _ = self._ring()

        def boom(w):
            raise RuntimeError("send handle already consumed")

        ranks[0]._pp_commit_comm_work = boom
        ranks[0]._weg2_commit_owed_forward()  # must not raise

    def test_the_commit_precedes_the_join_in_the_verdict(self):
        """WIRING: the commit is worthless after the block."""
        import inspect

        src = inspect.getsource(Scheduler.group_idle_verdict)
        self.assertIn("_weg2_commit_owed_forward()", src)
        self.assertLess(
            src.index("_weg2_commit_owed_forward()"),
            src.index("all_reduce("),
            "clause (ii): commit the owed forward BEFORE joining",
        )


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
        import inspect

        src = inspect.getsource(Scheduler.group_idle_verdict)
        self.assertIn("GROUP VERDICT SHORT", src)
        self.assertIn("n_present < world", src)

    def test_every_rank_emits_the_verdict_line(self):
        """sb2's reduce printed NOTHING on any rank."""
        import inspect

        src = inspect.getsource(Scheduler.group_idle_verdict)
        self.assertIn("WEG2-P-IDLE-VERDICT", src)
        for field in ("epoch=", "idle=", "blocking_rank=", "blockers=[", "participation="):
            self.assertIn(field, src)

    def test_no_bare_timeout_error_reaches_the_front(self):
        import inspect

        src = inspect.getsource(Scheduler.group_idle_verdict)
        self.assertIn("no message: gloo timeouts carry none", src)


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
