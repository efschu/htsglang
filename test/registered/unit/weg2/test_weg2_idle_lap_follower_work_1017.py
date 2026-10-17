# SPDX-License-Identifier: Apache-2.0
"""S9 W3 Weg2DrainWitnessDisagreement (boot ..._25812b36f9_1017_181001, 18:17:35Z).

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
(Weg2IdleVoteReq.peer_work_s / hop_at); PP0 reads it from the object it
receives anyway (R11: PP0 alone decides, no rank-local verdict, no collective).
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_weg2_idle_round_fresh_1268 as g1268  # noqa: E402
import test_weg2_idle_lap_own_work_1470_0929 as ow  # noqa: E402
from sglang.srt.managers import weg2_bubble_publish, weg2_idle_vote  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")

FOLLOWER_SWEEP_S = 0.75   # metal: sweep_ms 747.7 per follower pass


class _Case(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.clock = g1268.FakeClock()
        p = mock.patch("time.monotonic", new=self.clock)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.dict(os.environ, {weg2_bubble_publish.ENV: "1"})
        p.start()
        self.addCleanup(p.stop)


def follower_pass(g, r, sweep_s=FOLLOWER_SWEEP_S):
    """A follower pass as metal runs it: the flush handler's futile sweep (inside
    own_work, as scheduler.flush_cache wraps it), then the pass hooks."""
    with weg2_idle_vote.own_work():
        g.clock.advance(sweep_s)
    g1268.run_pass(g, r)


def cycle(g, sweep_s=FOLLOWER_SWEEP_S):
    """One front poll period: PP0 pass + poll (cheap sweep on PP0), then each
    follower pass pays its own futile sweep."""
    g1268.run_pass(g, 0)
    ok = g1268.poll(g)
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
        g1268.run_pass(g, 0)
        g1268.poll(g)
        g.clock.advance(g1268._ttl_s() + 1.0)       # lying about, no work
        g1268.run_pass(g, 1)
        g1268.run_pass(g, 2)
        g1268.run_pass(g, 0)
        self.assertFalse(g1268.poll(g), "an expired lap answered")


class ThePureRules(_Case):
    def test_expiry_discounts_peer_work(self):
        w = weg2_idle_vote.Weg2LapWitness(epoch=1, stamped_at=self.clock())
        self.clock.advance(5.0)
        self.assertIn("expired", weg2_idle_vote.expiry_reason(w, now=self.clock(), ttl_s=2.0))
        self.assertEqual(
            weg2_idle_vote.expiry_reason(w, now=self.clock(), ttl_s=2.0, peer_work_s=4.0), "")

    def test_peer_work_is_capped_by_the_wall(self):
        w = weg2_idle_vote.Weg2LapWitness(epoch=1, stamped_at=self.clock())
        self.clock.advance(1.0)
        self.assertEqual(
            weg2_idle_vote.expiry_reason(w, now=self.clock(), ttl_s=0.5, peer_work_s=99.0), "")

    def test_own_work_between_counts_only_the_overlap(self):
        t0 = self.clock()
        self.clock.advance(1.0)
        with weg2_idle_vote.own_work():
            self.clock.advance(2.0)
        self.clock.advance(1.0)
        self.assertAlmostEqual(weg2_idle_vote.own_work_between(t0, self.clock()), 2.0, places=6)
        self.assertAlmostEqual(weg2_idle_vote.own_work_between(t0 + 1.5, t0 + 2.0), 0.5, places=6)
        self.assertAlmostEqual(weg2_idle_vote.own_work_between(t0 + 3.5, t0 + 9.0), 0.0, places=6)

    def test_attach_accumulates_work_since_the_previous_hop(self):
        vote = weg2_idle_vote.Weg2IdleVoteReq(epoch=1, origin=0, world=3)
        vote.hop_at = self.clock()
        with weg2_idle_vote.own_work():
            self.clock.advance(1.5)
        weg2_idle_vote.note_hop_work(vote)
        self.assertAlmostEqual(vote.peer_work_s, 1.5, places=6)
        self.assertAlmostEqual(vote.hop_at, self.clock(), places=6)
        self.clock.advance(1.0)       # idle transit, not work
        weg2_idle_vote.note_hop_work(vote)
        self.assertAlmostEqual(vote.peer_work_s, 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
