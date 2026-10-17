# SPDX-License-Identifier: Apache-2.0
"""S9 W3 PdFlipDrainWitnessDisagreement (boot ..._25812b36f9_1017_181001, 18:17:35Z).

MEASURED (P.log of that boot, 18:16:05-18:17:35): after the 19835-token prefill
the P group sat behind a full KV arena (22 un-backed nodes, issued=0
refused=22).  PP0 swept that in 6 ms (refusal memo); PP1/PP2 ran one futile
PUBLISH-SWEEP of 0.65-0.8 s on EVERY /flush_cache poll (sweep_ms=747.7,
issue_ms=715, PASS-STALL other_ms 700-790; FLUSH_SWEEP_MEMO is off).  The vote
lap needs one follower pass per hop, so each lap wall was 4.2-5.1 s:
"IDLE-ROUND stale epoch=2 age_ms=4218 ... expired (age=3.714s > ttl=2.000s;
PP0 own publish work 0.503s not counted)" -- epochs 2, 3, 4, 5, 6 ... all
3/3 idle, all dropped, until the front's W3 STOP.  The #1470 own_work
discount (kvs2 W3, 0929) counted PP0's work only.

THE FIX: the lap carries the follower's own publish work, measured on the
follower's own clock span since the vote left the previous stage
(PdFlipIdleVoteReq.peer_work_s / hop_at); PP0 reads it from the object it
receives anyway (R11: PP0 alone decides, no rank-local verdict, no collective).
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_pdflip_idle_round_fresh_1268 as g1268  # noqa: E402
import test_pdflip_idle_lap_own_work_1470_0929 as ow  # noqa: E402
from flliper.srt.managers import pdflip_bubble_publish, pdflip_idle_vote  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402
from flliper.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

FOLLOWER_SWEEP_S = 0.75   # metal: sweep_ms 747.7 per follower pass


class _Case(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.clock = g1268.FakeClock()
        # the fake clock restarts at 5000 in every test: spans of an earlier test
        # of this process would overlap its windows
        del pdflip_idle_vote._OWN_WORK_SPANS[:]
        _STATES.clear()
        pdflip_idle_vote._OWN_WORK_OPEN[0] = None
        p = mock.patch("time.monotonic", new=self.clock)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.dict(os.environ, {pdflip_bubble_publish.ENV: "1"})
        p.start()
        self.addCleanup(p.stop)


import contextlib  # noqa: E402

_STATES = {}
# on the unfixed tree these two do not exist: the model then still runs, so the red of the integration case is the
# metal's (expired lap), not a missing attribute
for _n, _v in (("_OWN_WORK_SPANS", []), ("_OWN_WORK_OPEN", [None])):
    if not hasattr(pdflip_idle_vote, _n):
        setattr(pdflip_idle_vote, _n, _v)


@contextlib.contextmanager
def as_rank(r):
    """The model runs three ranks in ONE process, but own_work's clock and span list are per-PROCESS state on metal:
    swap them per rank, so a follower's sweep is never PP0's own work (the harness conflated them, which is how the
    first version of this test was green on the unfixed tree)."""
    st = _STATES.setdefault(r, [0.0, [], None])
    saved = (pdflip_idle_vote._OWN_WORK_S[0], list(pdflip_idle_vote._OWN_WORK_SPANS), pdflip_idle_vote._OWN_WORK_OPEN[0])
    pdflip_idle_vote._OWN_WORK_S[0] = st[0]
    pdflip_idle_vote._OWN_WORK_SPANS[:] = st[1]
    pdflip_idle_vote._OWN_WORK_OPEN[0] = st[2]
    try:
        yield
    finally:
        st[0] = pdflip_idle_vote._OWN_WORK_S[0]
        st[1] = list(pdflip_idle_vote._OWN_WORK_SPANS)
        st[2] = pdflip_idle_vote._OWN_WORK_OPEN[0]
        pdflip_idle_vote._OWN_WORK_S[0] = saved[0]
        pdflip_idle_vote._OWN_WORK_SPANS[:] = saved[1]
        pdflip_idle_vote._OWN_WORK_OPEN[0] = saved[2]


def follower_pass(g, r, sweep_s=FOLLOWER_SWEEP_S):
    """A follower pass as metal runs it: the flush handler's futile sweep (inside
    own_work, as scheduler.flush_cache wraps it), then the pass hooks."""
    with as_rank(r):
        with pdflip_idle_vote.own_work():
            g.clock.advance(sweep_s)
        g1268.run_pass(g, r)


def pp0_pass(g):
    with as_rank(0):
        g1268.run_pass(g, 0)


def pp0_poll(g):
    with as_rank(0):
        return g1268.poll(g)


def cycle(g, sweep_s=FOLLOWER_SWEEP_S):
    """One front poll period: PP0 pass + poll (cheap sweep on PP0), then each
    follower pass pays its own futile sweep."""
    pp0_pass(g)
    ok = pp0_poll(g)
    for _ in range(2):          # the slot rotation: a hop takes more than one pass
        follower_pass(g, 1, sweep_s)
        follower_pass(g, 2, sweep_s)
    g.clock.advance(0.002)
    return ok


class TheMetalSpecimen(_Case):
    def test_follower_sweeps_do_not_age_the_lap_out(self):
        """RED on 68de08743c: every lap's wall > ttl through the followers' own
        sweeps and is dropped as expired; the quiesce never gets its 200."""
        g = ow.build(self.clock)
        g.ranks[0].tree_cache.publish_unbacked_sweep = lambda **_kw: {"unbacked": 0}
        answers = [cycle(g) for _ in range(8)]
        self.assertIn(True, answers, "the idle group's lap was never read fresh")

    def test_follower_time_that_is_not_work_still_expires_the_lap(self):
        """The TTL is not switched off: a follower that sat idle for longer than
        the TTL (no publish work) still ages the lap out."""
        g = ow.build(self.clock)
        g.ranks[0].tree_cache.publish_unbacked_sweep = lambda **_kw: {"unbacked": 0}
        pp0_pass(g)
        pp0_poll(g)
        pp0_pass(g)                                 # stamps
        g.clock.advance(g1268._ttl_s() + 1.0)       # lying about, no work
        with as_rank(1):
            g1268.run_pass(g, 1)
        with as_rank(2):
            g1268.run_pass(g, 2)
        pp0_pass(g)
        self.assertFalse(pp0_poll(g), "an expired lap answered")


class ThePureRules(_Case):
    def test_expiry_discounts_peer_work(self):
        w = pdflip_idle_vote.PdFlipLapWitness(epoch=1, stamped_at=self.clock())
        self.clock.advance(5.0)
        self.assertIn("expired", pdflip_idle_vote.expiry_reason(w, now=self.clock(), ttl_s=2.0))
        self.assertEqual(
            pdflip_idle_vote.expiry_reason(w, now=self.clock(), ttl_s=2.0, peer_work_s=4.0), "")

    def test_peer_work_is_capped_by_the_wall(self):
        w = pdflip_idle_vote.PdFlipLapWitness(epoch=1, stamped_at=self.clock())
        self.clock.advance(1.0)
        self.assertEqual(
            pdflip_idle_vote.expiry_reason(w, now=self.clock(), ttl_s=0.5, peer_work_s=99.0), "")

    def test_own_work_between_counts_only_the_overlap(self):
        t0 = self.clock()
        self.clock.advance(1.0)
        with pdflip_idle_vote.own_work():
            self.clock.advance(2.0)
        self.clock.advance(1.0)
        self.assertAlmostEqual(pdflip_idle_vote.own_work_between(t0, self.clock()), 2.0, places=6)
        self.assertAlmostEqual(pdflip_idle_vote.own_work_between(t0 + 1.5, t0 + 2.0), 0.5, places=6)
        self.assertAlmostEqual(pdflip_idle_vote.own_work_between(t0 + 3.5, t0 + 9.0), 0.0, places=6)

    def test_attach_accumulates_work_since_the_previous_hop(self):
        vote = pdflip_idle_vote.PdFlipIdleVoteReq(epoch=1, origin=0, world=3)
        vote.hop_at = self.clock()
        with pdflip_idle_vote.own_work():
            self.clock.advance(1.5)
        pdflip_idle_vote.note_hop_work(vote)
        self.assertAlmostEqual(vote.peer_work_s, 1.5, places=6)
        self.assertAlmostEqual(vote.hop_at, self.clock(), places=6)
        self.clock.advance(1.0)       # idle transit, not work
        pdflip_idle_vote.note_hop_work(vote)
        self.assertAlmostEqual(vote.peer_work_s, 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
