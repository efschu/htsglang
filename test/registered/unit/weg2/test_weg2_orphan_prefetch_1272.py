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
"""#1272: an orphaned storage prefetch is collected where the group is in step.

THE SPECIMEN, boot weg2sb1 (2026-09-08), rid 43c9af54. The request FINISHED
normally on all three ranks -- `WEG2 END-ANCHOR ... ok=True short=0`,
`#924D station=relinquish`, and `retract=0` on its lifecycle line. It was not
aborted and not retracted.

THE PRODUCER is an asymmetry at the completion free-site, and the log states it
by absence::

    14:29:28 PP0  #905 PREFETCH-COMPLETE free-site: req=43c9af54...
    14:29:28 PP0  HiCache prefetch success req=43c9af54... attn_reduce_world=1
    14:29:28 PP1  #905 PREFETCH-COMPLETE free-site: req=43c9af54...
    14:29:28 PP1  HiCache prefetch success req=43c9af54... attn_reduce_world=1
    (PP2: NEITHER LINE)

PP0 and PP1 completed and cleared the record; PP2 never reached that one-shot
site for this rid, so at 14:29:29 the orphan set was {43c9af54} on PP2 and {} on
the other two. There is no second visit: the only retry path,
`_drain_prefetch_progress`, hangs off the ADMISSION path, and the queue was
empty -- `#queue-req: 0, #running-req: 0`. PP2 then polled
`check_hicache_events` 2964 times over 30 s with the blocker unchanged, because
that function does not reach the collector.

So the orphan is unreachable exactly when it matters, and with #1268 in place
that is no longer a death but a W3 refusal LOOP: P can never sleep until some
unrelated admission happens to walk the collector.

THE FIX is not a new collector. It is running the existing one at
`group_idle_verdict` -- the one point on the quiesce path every rank reaches
together on the same broadcast control request -- so the collector's own
collective (`drain_retired_prefetch -> _all_reduce_attn_groups`) has uniform
participation and no rank-local trigger is introduced.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

RID = "43c9af54b69a4822ad0bf507a1f1a9a4"


class _Tree:
    """The sb1 shape: one open prefetch record, nothing else in flight."""

    def __init__(self, orphans):
        self.ongoing_prefetch = dict.fromkeys(orphans, object())
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_backup = {}
        self.enable_storage = True
        self.collected = []

    # what the collector calls
    def drain_retired_prefetch(self):
        return 0

    def check_prefetch_progress(self, rid):
        self.collected.append(rid)
        self.ongoing_prefetch.pop(rid, None)
        return True

    # what `check_hicache_events` does -- deliberately NOT the collector, which
    # is the whole reason 2964 polls moved nothing on sb1.
    def check_hicache_events(self):
        return None


def _sched(orphans):
    tc = _Tree(orphans)
    s = SimpleNamespace(
        tree_cache=tc,
        enable_hicache_storage=True,
        enable_hierarchical_cache=True,
        waiting_queue=[],
        world_group=SimpleNamespace(cpu_group=None),
        collective_timeout_s=5.0,
    )
    s.is_fully_idle = lambda: len(tc.ongoing_prefetch) == 0
    s.idle_blockers = lambda: (
        [f"hicache_prefetch({len(tc.ongoing_prefetch)}: "
         f"{','.join(r[:8] for r in tc.ongoing_prefetch)})"]
        if tc.ongoing_prefetch
        else []
    )
    s._drain_prefetch_progress = Scheduler._drain_prefetch_progress.__get__(s)
    # #1268 fix 1b: a real Scheduler always has this; the
    # stand-in must too, or it tests a shape that cannot exist.
    s.pp_group = SimpleNamespace(is_last_rank=True)
    s._weg2_commit_owed_forward = (
        Scheduler._weg2_commit_owed_forward.__get__(s)
    )
    s.group_idle_verdict = Scheduler.group_idle_verdict.__get__(s)
    return s, tc


class TheSb1ShapeIsCollected(CustomTestCase):
    def test_red_first_polling_hicache_events_moves_nothing(self):
        """THE SPECIMEN. What PP2 actually did: 2964 polls, zero movement.

        This is the red half -- it pins that the function the sleep drain polls
        cannot clear this blocker, which is why the orphan survived 30 s.
        """
        _, tc = _sched([RID])
        for _ in range(2964):
            tc.check_hicache_events()
        self.assertEqual(list(tc.ongoing_prefetch), [RID], "still orphaned")
        self.assertEqual(tc.collected, [], "check_hicache_events collects nothing")

    def test_the_verdict_collects_the_orphan_in_one_round(self):
        s, tc = _sched([RID])
        self.assertFalse(s.is_fully_idle(), "the sb1 entry state")
        idle, detail = s.group_idle_verdict()
        self.assertEqual(tc.collected, [RID], "collected at the quiesce point")
        self.assertTrue(idle, "and the group is idle in the SAME round")
        self.assertNotIn("43c9af54", detail)

    def test_an_uncollectable_blocker_still_refuses(self):
        """The fix must not turn a real blocker into a false idle: a record the
        collector declines stays in the verdict."""
        s, tc = _sched([RID])
        tc.check_prefetch_progress = lambda rid: False  # declines to terminate
        idle, detail = s.group_idle_verdict()
        self.assertFalse(idle)
        self.assertIn("43c9af54", detail)

    def test_collection_runs_before_the_idle_reading(self):
        """ORDER is the fix. Reading idle first and collecting after would
        refuse on a state that was already collectable."""
        s, tc = _sched([RID])
        order = []
        base_idle = s.is_fully_idle
        s.is_fully_idle = lambda: (order.append("read"), base_idle())[1]
        base_collect = tc.check_prefetch_progress
        tc.check_prefetch_progress = lambda rid: (
            order.append("collect"), base_collect(rid)
        )[1]
        s.group_idle_verdict()
        self.assertEqual(order[0], "collect", f"order was {order}")

    def test_no_storage_no_collection(self):
        """Off HiCache storage this must touch nothing -- byte-identical."""
        s, tc = _sched([RID])
        s.enable_hicache_storage = False
        s.group_idle_verdict()
        self.assertEqual(tc.collected, [])


class TheWiring(CustomTestCase):
    def test_the_collector_is_called_from_the_group_uniform_point(self):
        import inspect

        src = inspect.getsource(Scheduler.group_idle_verdict)
        self.assertIn("_drain_prefetch_progress()", src)
        self.assertLess(
            src.index("_drain_prefetch_progress()"),
            src.index("my_idle = self.is_fully_idle()"),
            "collect before voting",
        )

    def test_the_refuted_rank_uniformity_claim_is_retracted(self):
        """The old comment asserted the orphan set agrees on every rank. sb1
        refutes it; the retraction must stay next to the code it governs."""
        import inspect

        src = inspect.getsource(Scheduler._drain_prefetch_progress)
        self.assertIn("#1272 RETRACTION", src)
        # The old claim is QUOTED inside the retraction on purpose -- a
        # withdrawal that does not say what it withdraws is not checkable. So
        # the test is that the quote is dominated by the withdrawal, not that
        # the words are gone.
        i_claim = src.index("the orphan set and its order agree on")
        i_retract = src.index("#1272 RETRACTION")
        self.assertLess(i_retract, i_claim, "the claim must read as withdrawn")
        self.assertIn("THAT IS FALSE", src)


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
