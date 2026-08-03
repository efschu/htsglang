"""#505-A2-05: the gloo fallback of a fallback-eligible transport is a GROUP
decision, never a per-rank one.

`barlink._build_transport` catches a bring-up failure of the transports that
are deliberately outside `_NO_FALLBACK` ("bar1", "matrix", and also "shm" /
"ucx"), warns, and returns `None` -- which selects the inline host-staged gloo
plane for THAT rank. Before this fix nothing reconciled that outcome across the
group, so a probe that fails on one card only (BAR1 peer mapping, a per-card
JIT kernel build, the per-card window probe) left the group with two different
answers to a question every rank must answer identically.

What is pinned here:
  1. DIVERGENCE COLLAPSES TO GLOO EVERYWHERE. A rank whose own bring-up
     SUCCEEDED still ends up on the gloo plane when any peer failed, and it
     tears its transport down instead of leaving it half-used.
  2. THE FAILING RANK PARTICIPATES. It must issue exactly the same agreement
     collective as its peers -- a rank that skipped it would be the very
     desync this fix exists to remove.
  3. AGREEMENT IS NOT A NEW FALLBACK. All-succeeded keeps the transport,
     all-failed stays gloo, and both without the divergence warning. This is
     the can-discriminate precondition for 1: a check that always answers
     "gloo" would pass 1 while proving nothing.
  4. `_NO_FALLBACK` REFUSAL SEMANTICS ARE UNTOUCHED. "device" still raises and
     never enters the agreement.

CPU only, no torch.distributed: `barlink.dist` is replaced by a stub that
plays the peers' votes back into the all_reduce. Nothing here constructs a
real transport or touches a device.
"""

import unittest

import torch

from sglang.srt.distributed.device_communicators import barlink as barlink_mod
from sglang.srt.distributed.device_communicators.barlink import (
    TRANSPORT_REGISTRY,
    _build_transport,
    group_states,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeTransport:
    """Just enough of the transport interface for the seam, plus a close probe."""

    BARLINK_OPS = frozenset({"all_reduce"})

    def __init__(self):
        self.closed = 0

    def handles(self, op, nbytes):
        return op in self.BARLINK_OPS

    def name(self):
        return "agree-test"

    def close(self):
        self.closed += 1


class _ReduceOp:
    SUM = "sum"


class _FakeDist:
    """A stand-in for torch.distributed over one synthetic group.

    ``votes`` is the whole group's outcome (1 = the transport came up on that
    rank). ``all_reduce`` adds the PEERS' votes into the caller's vector, which
    is exactly what a real SUM all_reduce over a one-hot vector does.
    """

    ReduceOp = _ReduceOp

    def __init__(self, votes, rank, reasons=None):
        self.votes = list(votes)
        self.rank = rank
        self.reasons = reasons or ["" for _ in votes]
        self.all_reduce_calls = 0
        self.all_gather_object_calls = 0

    def get_world_size(self, group=None):
        return len(self.votes)

    def get_rank(self, group=None):
        return self.rank

    def all_reduce(self, tensor, op=None, group=None, async_op=False):
        self.all_reduce_calls += 1
        assert op is _ReduceOp.SUM
        for r, v in enumerate(self.votes):
            if r != self.rank:
                tensor[r] += int(v)
        return None  # ran inline: nothing left for the bounded wait to poll

    def all_gather_object(self, out, obj, group=None):
        self.all_gather_object_calls += 1
        for r in range(len(self.votes)):
            out[r] = obj if r == self.rank else self.reasons[r]


class _GroupAgreementBase(CustomTestCase):
    """Registers a fake fallback-eligible transport and stubs `barlink.dist`."""

    NAME = "agree-test"

    def _install(self, votes, rank, reasons=None, group="tp"):
        made = _FakeTransport() if votes[rank] else None

        def factory(cpu_group, device, group=""):
            if made is None:
                raise barlink_mod.Bar1Failed("no aperture on this card", stage="setup")
            return made

        fake_dist = _FakeDist(votes, rank, reasons)
        saved_dist = barlink_mod.dist
        TRANSPORT_REGISTRY[self.NAME] = factory
        barlink_mod.dist = fake_dist
        self.addCleanup(TRANSPORT_REGISTRY.pop, self.NAME, None)
        self.addCleanup(setattr, barlink_mod, "dist", saved_dist)
        return made, fake_dist

    def _build(self, group="tp"):
        # `object()` stands in for a real ProcessGroup: the agreement only ever
        # passes it through to the stubbed dist calls.
        return _build_transport(self.NAME, object(), None, disabled=False, group=group)


class TestDivergenceCollapsesToGloo(_GroupAgreementBase):
    def test_a_rank_that_succeeded_joins_gloo_when_a_peer_failed(self):
        """Property 1 -- the falsifier.

        Rank 0's own bring-up succeeds, rank 1's fails. Before the fix rank 0
        kept its transport and ran barlink while rank 1 ran gloo.
        """
        made, _ = self._install(votes=[1, 0], rank=0, group="tp-diverge")
        built = self._build(group="tp-diverge")
        self.assertIsNone(
            built,
            "a peer failed to bring the transport up -- this rank must not "
            "keep running it while that peer is on the gloo plane",
        )
        self.assertEqual(
            made.closed, 1, "the abandoned transport must be torn down, not leaked"
        )

    def test_the_group_decision_is_recorded_as_the_achieved_state(self):
        self._install(votes=[1, 0], rank=0, group="tp-state")
        self._build(group="tp-state")
        entry = group_states()["tp-state"]
        self.assertEqual(entry["achieved"], "gloo")
        self.assertEqual(entry["requested"], self.NAME)
        self.assertFalse(entry["direct"])
        self.assertEqual(entry["stage"], "group-agreement")
        self.assertIn("1", entry["reason"], "the failing rank must be named")

    def test_the_reason_of_the_failing_peer_reaches_the_surviving_rank(self):
        self._install(
            votes=[1, 0],
            rank=0,
            reasons=["", "no aperture on this card"],
            group="tp-reason",
        )
        self._build(group="tp-reason")
        self.assertIn("no aperture on this card", group_states()["tp-reason"]["reason"])


class TestTheFailingRankParticipates(_GroupAgreementBase):
    def test_the_failing_rank_issues_the_same_agreement_collective(self):
        """Property 2. A rank that fell back without voting IS the desync."""
        _, fake = self._install(votes=[1, 0], rank=1, group="tp-loser")
        self.assertIsNone(self._build(group="tp-loser"))
        self.assertEqual(
            fake.all_reduce_calls,
            1,
            "the rank whose bring-up failed must still enter the agreement -- "
            "otherwise its peers wait for a vote that never comes",
        )
        self.assertEqual(fake.all_gather_object_calls, 1)


class TestAgreementIsNotANewFallback(_GroupAgreementBase):
    """Property 3 -- the can-discriminate precondition."""

    def test_all_ranks_succeeded_keeps_the_transport(self):
        made, fake = self._install(votes=[1, 1], rank=0, group="tp-all-ok")
        self.assertIs(self._build(group="tp-all-ok"), made)
        self.assertEqual(made.closed, 0)
        self.assertEqual(fake.all_reduce_calls, 1)
        # No divergence -> no reason exchange.
        self.assertEqual(fake.all_gather_object_calls, 0)
        entry = group_states()["tp-all-ok"]
        self.assertEqual(entry["achieved"], self.NAME)
        self.assertTrue(entry["direct"])

    def test_all_ranks_failed_is_still_a_plain_group_wide_fallback(self):
        _, fake = self._install(votes=[0, 0], rank=0, group="tp-all-bad")
        self.assertIsNone(self._build(group="tp-all-bad"))
        self.assertEqual(fake.all_reduce_calls, 1)
        self.assertEqual(fake.all_gather_object_calls, 0)
        entry = group_states()["tp-all-bad"]
        self.assertEqual(entry["achieved"], "gloo")
        self.assertNotEqual(
            entry["stage"],
            "group-agreement",
            "every rank failed on its own merits -- that is not a divergence",
        )

    def test_a_single_rank_group_never_agrees_with_anyone(self):
        made, fake = self._install(votes=[1], rank=0, group="tp-solo")
        self.assertIs(self._build(group="tp-solo"), made)
        self.assertEqual(fake.all_reduce_calls, 0)


class TestRefusalSemanticsUnchanged(_GroupAgreementBase):
    """Property 4: `_NO_FALLBACK` transports still raise and never vote."""

    def test_device_failure_still_propagates_without_an_agreement(self):
        fake = _FakeDist([1, 1], 0)
        saved_dist = barlink_mod.dist
        saved = TRANSPORT_REGISTRY["device"]

        def boom(cpu_group, device):
            raise RuntimeError("transport init failed")

        TRANSPORT_REGISTRY["device"] = boom
        barlink_mod.dist = fake
        try:
            with self.assertRaises(RuntimeError):
                _build_transport("device", object(), None, disabled=False)
        finally:
            TRANSPORT_REGISTRY["device"] = saved
            barlink_mod.dist = saved_dist
        self.assertEqual(fake.all_reduce_calls, 0)


class TestNoGroupMeansNoAgreement(_GroupAgreementBase):
    """The seam tests build transports with `cpu_group=None`.

    That is not a hole in production: `BarlinkCommunicator` always passes the
    group's real gloo `ProcessGroup`. Pinned so the agreement cannot start
    inventing a group out of `None` and reach the DEFAULT process group.
    """

    def test_cpu_group_none_builds_without_touching_dist(self):
        made = _FakeTransport()
        TRANSPORT_REGISTRY[self.NAME] = lambda cpu_group, device: made
        self.addCleanup(TRANSPORT_REGISTRY.pop, self.NAME, None)
        fake = _FakeDist([1, 1], 0)
        saved_dist = barlink_mod.dist
        barlink_mod.dist = fake
        try:
            built = _build_transport(self.NAME, None, None, disabled=False)
        finally:
            barlink_mod.dist = saved_dist
        self.assertIs(built, made)
        self.assertEqual(fake.all_reduce_calls, 0)


class TestVoteVectorMath(CustomTestCase):
    """The one-hot SUM all_reduce must name exactly the failing ranks."""

    def test_one_hot_sum_reproduces_every_rank_vote(self):
        votes = [1, 0, 1, 0]
        acc = torch.zeros(len(votes), dtype=torch.int64)
        for rank, v in enumerate(votes):
            own = torch.zeros(len(votes), dtype=torch.int64)
            own[rank] = int(v)
            acc += own
        self.assertEqual([int(x) for x in acc], votes)


if __name__ == "__main__":
    unittest.main()
