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


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
