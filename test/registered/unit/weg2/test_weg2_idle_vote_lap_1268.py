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
"""#1268 fix 1c: the idle vote travels home on the ring lap as a control object.

RED-FIRST IS THE TOPOLOGY, not a flag. Fix 1 and fix 1b both put a blocking
collective on the quiesce path and both deadlocked on metal in the same shape
(boots weg2sb2 and weg2sb3, six py-spy stacks in
``/spinning/gpu-arb/weg2/BOOT_weg2sb3_killer_context.txt``):

    PP0        bounded_wait <- group_idle_verdict <- flush_cache
    PP1/PP2    _join <- advance <- receive <- recv_object <- ...
               <- _pp_recv_dict_from_prev_stage

The followers are parked in the HIDDEN-STATES / proxy tensor-dict channel --
one channel over from the request chain -- waiting for a proxy send PP0 issues
LATER IN ITS OWN PASS. So the ring here models exactly that: a follower only
becomes reachable once its predecessor has FINISHED ITS PASS, and a rank that
blocks mid-pass never finishes one. Under that model a blocking aggregation
cannot terminate, and the lap can.
"""

import inspect
import unittest
from types import SimpleNamespace

from sglang.srt.managers.weg2_idle_vote import (
    WEG2_VOTE_TAG,
    Weg2IdleVoteReq,
    attach_slot,
    home_ranks,
    log_verdict,
    refusal_detail,
    tally,
)
from sglang.test.test_utils import CustomTestCase


# --------------------------------------------------------------------------
# a faithful three-object ring
# --------------------------------------------------------------------------


class RingRank(SimpleNamespace):
    """One PP rank, with the real hooks bound and a modelled wire."""


def build_ring(idle=(True, True, True), world=3, storage=False):
    """Three ranks on one modelled request-chain arc plus one home stream.

    THE MODEL'S LOAD-BEARING RULE: ``inbox[r+1]`` is only filled when rank r
    completes a pass. A rank that blocks inside a pass therefore strands every
    rank below it -- which is the sb2/sb3 topology, and the reason a blocking
    verdict cannot be tested green here by accident.
    """
    from sglang.srt.managers.scheduler import Scheduler

    wire = {
        "inbox": {r: [] for r in range(world)},
        "home": [],
        "home_sends": 0,
        "forwards": 0,
        "committed": 0,
    }

    def make(rank):
        s = RingRank(
            ps=SimpleNamespace(
                pp_rank=rank,
                pp_size=world,
                tp_size=1,
                attn_tp_rank=0,
                attn_cp_rank=0,
                attn_tp_size=1,
                attn_dp_rank=0,
                attn_cp_size=1,
            ),
            pp_group=SimpleNamespace(
                is_first_rank=(rank == 0), is_last_rank=(rank == world - 1)
            ),
            world_group=SimpleNamespace(cpu_group=None),
            enable_hicache_storage=storage,
            is_fully_idle=lambda r=rank: idle[r],
            idle_blockers=lambda r=rank: ([] if idle[r] else [f"blocker_r{r}"]),
            _drain_prefetch_progress=lambda: wire.__setitem__(
                "collected", wire.get("collected", 0) + 1
            ),
            _pp_commit_comm_work=lambda w: wire.__setitem__(
                "committed", wire["committed"] + 1
            ),
        )
        for name in (
            "_weg2_vote_pass_hook",
            "_weg2_vote_after_forward",
            "_weg2_vote_maybe_stamp",
            "_weg2_vote_attach_own_slot",
            "_weg2_vote_harvest_home",
            "_weg2_vote_dp_offset",
            "_weg2_vote_is_wire_rank",
        ):
            setattr(s, name, getattr(Scheduler, name).__get__(s))

        # The home hop, modelled on the same (src, dst) the real one computes.
        def _send_home(vote, _group, _src, _dst, r=rank):
            wire["home"].append(vote)
            wire["home_sends"] += 1

        s._weg2_send_home = _send_home
        return s

    ranks = [make(r) for r in range(world)]

    # PP0's harvest reads the modelled home stream instead of a posted frame.
    def harvest(s=ranks[0]):
        if getattr(s, "_weg2_vote_outstanding", None) is None:
            return
        if not wire["home"]:
            return
        vote = wire["home"].pop(0)
        log_verdict(vote, 0)
        s._weg2_vote_verdict = vote
        s._weg2_vote_outstanding = None

    ranks[0]._weg2_vote_harvest_home = harvest
    return ranks, wire


def run_pass(rank_obj, wire, incoming=None):
    """One scheduler pass on one rank: hook, forward, hook, dispatch.

    Mirrors ``_pp_forward_and_process_input_requests``: the pass hook runs
    BEFORE the forward, the forward carries whatever is in ``recv_reqs``, and
    the after-forward hook closes the ring and strips the vote.
    """
    recv_reqs = list(incoming or [])
    rank_obj._weg2_vote_pass_hook(recv_reqs)
    rank = rank_obj.ps.pp_rank
    if not rank_obj.pp_group.is_last_rank:
        wire["inbox"][rank + 1].extend(recv_reqs)
        wire["forwards"] += 1
    else:
        # the real last rank sends home through weg2_idle_vote.send_home; the
        # ring swaps in a modelled wire at the same call site
        import sglang.srt.managers.scheduler_pp_mixin as pm

        real = pm.send_home
        pm.send_home = rank_obj._weg2_send_home
        try:
            recv_reqs = rank_obj._weg2_vote_after_forward(recv_reqs)
        finally:
            pm.send_home = real
        return recv_reqs
    return rank_obj._weg2_vote_after_forward(recv_reqs)


def drain_lap(ranks, wire, passes=6):
    """Run the ring until the lap comes home or the pass budget expires."""
    for _ in range(passes):
        for r in range(len(ranks)):
            incoming = wire["inbox"][r]
            wire["inbox"][r] = []
            run_pass(ranks[r], wire, incoming)
        if getattr(ranks[0], "_weg2_vote_verdict", None) is not None:
            return True
    return False


class TheLapComesHome(CustomTestCase):
    """Only rank 0 receives the RPC; the verdict still covers all three."""

    def test_red_first_pp0_alone_cannot_produce_a_verdict(self):
        """Before any lap, PP0 has ONE slot and that is not a group fact."""
        vote = Weg2IdleVoteReq(epoch=1, origin=0, world=3)
        attach_slot(vote, 0, True, "none")
        t = tally(vote)
        self.assertFalse(t.idle)
        self.assertFalse(t.complete)
        self.assertEqual(t.n_present, 1)
        self.assertEqual(list(t.missing_ranks), [1, 2])

    def test_the_vote_comes_home_with_full_participation(self):
        ranks, wire = build_ring(idle=(True, True, True))
        ranks[0]._weg2_vote_wanted = True
        self.assertTrue(drain_lap(ranks, wire), "the lap never came home")
        vote = ranks[0]._weg2_vote_verdict
        t = tally(vote)
        self.assertTrue(t.idle)
        self.assertEqual((t.n_present, t.world), (3, 3))
        self.assertEqual(list(t.missing_ranks), [])

    def test_the_epoch_is_a_number_and_the_same_on_every_rank(self):
        """The `epoch=?` on every boot line was a MISSING SOURCE, not a format.

        `weg2_flip_epoch` had one reader and zero writers in the whole tree, so
        `getattr(..., None)` always won and every verdict line printed the
        literal '?'. The epoch is now minted by PP0 and travels ON the object.
        """
        ranks, wire = build_ring()
        ranks[0]._weg2_vote_wanted = True
        self.assertTrue(drain_lap(ranks, wire))
        vote = ranks[0]._weg2_vote_verdict
        self.assertIsInstance(vote.epoch, int)
        self.assertGreaterEqual(vote.epoch, 1)
        line = log_verdict(vote, 0)
        self.assertIn(f"epoch={vote.epoch}", line)
        self.assertNotIn("epoch=?", line)

    def test_a_blocking_follower_is_named_not_averaged_away(self):
        """The sb1 shape: PP0 and PP1 idle, PP2 holding an orphan prefetch."""
        ranks, wire = build_ring(idle=(True, True, False))
        ranks[0]._weg2_vote_wanted = True
        self.assertTrue(drain_lap(ranks, wire))
        t = tally(ranks[0]._weg2_vote_verdict)
        self.assertFalse(t.idle)
        self.assertTrue(t.complete)
        self.assertEqual(list(t.blocking_ranks), [2])

    def test_a_blocking_rank_zero_is_not_lost(self):
        ranks, wire = build_ring(idle=(False, True, True))
        ranks[0]._weg2_vote_wanted = True
        self.assertTrue(drain_lap(ranks, wire))
        t = tally(ranks[0]._weg2_vote_verdict)
        self.assertFalse(t.idle)
        self.assertEqual(list(t.blocking_ranks), [0])


class TheMutants(CustomTestCase):
    """Each mutant is the defect this fix closes, applied to the aggregation."""

    def test_mutant_aggregate_from_rank_zero_only_is_red(self):
        """#1268 itself: the entrypoint answering for the group."""
        vote = Weg2IdleVoteReq(epoch=7, origin=0, world=3)
        attach_slot(vote, 0, True, "none")
        attach_slot(vote, 1, True, "none")
        attach_slot(vote, 2, False, "hicache_prefetch(1: 43c9af54)")
        rank0_only = Weg2IdleVoteReq(
            epoch=7, origin=0, world=3, slots=[vote.slots[0]]
        )
        self.assertTrue(tally(vote).complete)
        self.assertFalse(tally(vote).idle)
        # the mutant would answer "idle" off one slot -- it must not be able to
        self.assertFalse(tally(rank0_only).idle)
        self.assertFalse(tally(rank0_only).complete)

    def test_mutant_all_present_voted_idle_is_red(self):
        """`n_idle == n_present` is a subset presented as the whole."""
        vote = Weg2IdleVoteReq(epoch=8, origin=0, world=3)
        attach_slot(vote, 0, True, "none")
        attach_slot(vote, 1, True, "none")
        t = tally(vote)
        self.assertEqual(t.n_idle, t.n_present)  # the mutant's predicate
        self.assertFalse(t.idle)  # the real one

    def test_a_rank_cannot_overwrite_its_own_or_anothers_slot(self):
        vote = Weg2IdleVoteReq(epoch=9, origin=0, world=3)
        self.assertTrue(attach_slot(vote, 2, False, "blocker"))
        self.assertFalse(attach_slot(vote, 2, True, "none"))
        self.assertEqual(list(tally(vote).blocking_ranks), [2])


class TheCadenceIsUnchanged(CustomTestCase):
    """No lap on an idle pass without a pending vote (#973's commit cadence)."""

    def test_no_home_send_on_idle_passes(self):
        ranks, wire = build_ring()
        for _ in range(5):
            for r in range(3):
                incoming = wire["inbox"][r]
                wire["inbox"][r] = []
                run_pass(ranks[r], wire, incoming)
        self.assertEqual(wire["home_sends"], 0)
        self.assertEqual(wire["committed"], 0)

    def test_exactly_one_home_send_per_lap(self):
        ranks, wire = build_ring()
        ranks[0]._weg2_vote_wanted = True
        self.assertTrue(drain_lap(ranks, wire))
        self.assertEqual(wire["home_sends"], 1)

    def test_the_downward_arc_is_not_touched(self):
        """The forward is unconditional per pass by #969 §W3 -- fix 1c adds
        nothing to it, and the only `else` added is on the LAST stage and only
        while a vote is homebound."""
        import sglang.srt.managers.scheduler_pp_mixin as pm

        src = inspect.getsource(pm.SchedulerPPMixin._weg2_vote_after_forward)
        self.assertIn("is_last_rank", src)
        # the guard that keeps an empty pass off the new arc
        self.assertIn("if not any(isinstance(r, Weg2IdleVoteReq)", src)


class TheRefusalNamesTheMissingRanks(CustomTestCase):
    def test_a_short_lap_lists_who_never_reported(self):
        vote = Weg2IdleVoteReq(epoch=3, origin=0, world=3)
        attach_slot(vote, 0, True, "none")
        t = tally(vote)
        detail = refusal_detail(vote, t, "GROUP VERDICT SHORT:")
        self.assertIn("[1, 2]", detail)
        self.assertIn("epoch=3", detail)
        self.assertIn("participation=1/3", detail)

    def test_never_a_bare_timeout(self):
        """sb3 stopped the front on the literal string 'TimeoutError: '."""
        vote = Weg2IdleVoteReq(epoch=4, origin=0, world=3)
        detail = refusal_detail(vote, tally(vote), "GROUP VERDICT SHORT:")
        self.assertNotEqual(detail.strip(), "TimeoutError:")
        self.assertIn("missing", detail.lower())

    def test_pending_is_not_idle(self):
        """UNDECIDED IS NOT IDLE, on the scheduler's own answer path."""
        from sglang.srt.managers.scheduler import Scheduler

        s = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=0, pp_size=3),
            is_fully_idle=lambda: True,
            idle_blockers=lambda: [],
        )
        s.group_idle_verdict = Scheduler.group_idle_verdict.__get__(s)
        ok, detail = s.group_idle_verdict()
        self.assertFalse(ok)
        self.assertIn("PENDING", detail)
        self.assertTrue(s._weg2_vote_wanted)


class TheWireIsIsolated(CustomTestCase):
    def test_the_home_hop_has_its_own_tag(self):
        """(last rank -> PP0, tag 0) already carries pp_typed_channel's proxy
        and output messages, which are demultiplexed IN BAND rather than by
        tag. A standing frame on tag 0 would eventually misframe one."""
        self.assertNotEqual(WEG2_VOTE_TAG, 0)

    def test_the_home_destination_is_pp0(self):
        """Same arithmetic `_pp_send_pyobj_to_next_stage` uses, at the wrap."""
        self.assertEqual(home_ranks(2, 3, 1, 0), (2, 0))
        self.assertEqual(home_ranks(2, 3, 2, 1), (5, 1))


class TheGroupCollectiveTrapIsNamed(CustomTestCase):
    def test_attn_world_above_one_refuses_by_name(self):
        """#1028/#580: a rank-local trigger of a group collective."""
        ranks, _wire = build_ring()
        ranks[1].ps.attn_tp_size = 2
        vote = Weg2IdleVoteReq(epoch=1, origin=0, world=3)
        with self.assertRaises(RuntimeError) as cm:
            ranks[1]._weg2_vote_attach_own_slot(vote)
        self.assertIn("#580", str(cm.exception))

    def test_the_collector_runs_once_per_rank_per_lap(self):
        ranks, wire = build_ring(storage=True)
        ranks[0]._weg2_vote_wanted = True
        self.assertTrue(drain_lap(ranks, wire))
        self.assertEqual(wire.get("collected", 0), 3)


class TheRefutedClaimsAreWithdrawnInPlace(CustomTestCase):
    def test_fix1_broadcast_premise_is_quoted_as_withdrawn(self):
        from sglang.srt.managers.scheduler import Scheduler

        doc = inspect.getdoc(Scheduler.group_idle_verdict) or ""
        self.assertIn("WITHDRAWN (fix 1)", doc)
        self.assertIn("broadcast control request", doc)

    def test_fix1b_owed_forward_premise_is_quoted_as_withdrawn(self):
        from sglang.srt.managers.scheduler import Scheduler

        doc = inspect.getdoc(Scheduler.group_idle_verdict) or ""
        self.assertIn("WITHDRAWN (fix 1b)", doc)
        self.assertIn("HIDDEN-STATES", doc)

    def test_the_fix1b_helper_is_gone(self):
        from sglang.srt.managers.scheduler import Scheduler

        self.assertFalse(hasattr(Scheduler, "_weg2_commit_owed_forward"))

    def test_the_answer_path_no_longer_blocks(self):
        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.group_idle_verdict)
        for banned in ("all_reduce", "bounded_wait", "barrier"):
            self.assertNotIn(banned, src, f"{banned} is back on the answer path")


if __name__ == "__main__":
    unittest.main()
