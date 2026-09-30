# SPDX-License-Identifier: Apache-2.0
"""kvs2 W3 PdFlipDrainWitnessDisagreement (boot ...kvdemandbar1dauer09291534,
424346f693, 2026-09-29 15:42:17): the idle lap came home 3/3 idle on every
poll and was never read fresh.

MEASURED (P.log 18605-20150, front.log 678):
* every /flush_cache poll ran #1470 FLUSH-PUBLISH first: issued=0,
  unbacked_left=37, nothing in flight, waited_ms 4561-5016 -- four sweeps of
  ~1.25 s each (PDFLIP PUBLISH-SWEEP n=713..716, 37 refused per sweep behind a
  fruitless #1427 ARENA-DROP) that could not change anything;
* the bubble publisher swept once more per pass (n=712, the pass's other_ms
  1255; #1466 PASS-STALL pass_ms=6131 input_ms=4876);
* the lap was stamped at a pass-top, harvested at the next, and read by the
  poll after -- behind that poll's own FLUSH-PUBLISH: "#1268 IDLE-ROUND stale
  epoch=7 age_ms=11148 ... lap_idle=True participation=3/3 ... expired
  (age=11.148s > ttl=2.000s)", epochs 6, 7, 8 alike, then W3 at the deadline.

THE FIX (pdflip_idle_vote.own_work): a FLUSH-PUBLISH round that issued nothing
with nothing in flight ends the loop, and PP0's own publish work -- the poll's
FLUSH-PUBLISH and the bubble sweep, time in which PP0 receives nothing and no
work can enter the group -- is not the lap's age.  Everything else about the
lap (round id, taint, TTL on the rest, PP0 idle at the read) is unchanged.

The group model is test_pdflip_idle_round_fresh_1268's (real vote hooks, real
flush_cache, real group_idle_verdict, real flush wrapper); what is added is a
tree whose sweep costs the measured 1.25 s and a PP0 pass that runs the real
bubble publisher in its no-batch gap, as _pp_bubble_note_no_batch does.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_pdflip_idle_round_fresh_1268 as g1268  # noqa: E402

from flliper.srt.managers import pdflip_bubble_publish, pdflip_idle_vote  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402
from flliper.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

SWEEP_S = 1.25      # one PUBLISH-SWEEP of 37 refused nodes, metal
UNBACKED = 37


class FullArenaTree:
    """The P tree of 09291534: 37 un-backed nodes no sweep can publish."""

    def __init__(self, clock, *, in_flight=0, issue_every_round=0):
        self.clock = clock
        self.sweeps = 0
        self.resets = 0
        self.in_flight = in_flight
        self.issue_every_round = issue_every_round

    def publish_unbacked_sweep(self, max_issue=64, **_kw):
        self.sweeps += 1
        self.clock.advance(SWEEP_S)
        return {
            "unbacked": UNBACKED,
            "issued": self.issue_every_round,
            "refused": UNBACKED - self.issue_every_round,
            "pending": self.in_flight,
            "skipped_pending": 0,
        }

    def writing_check(self, write_back=False):
        if write_back:
            self.in_flight = 0

    def reset(self):
        self.resets += 1


def build(clock, **tree_kw):
    g = g1268.build_p_group(clock)
    pp0 = g.ranks[0]
    pp0.enable_hierarchical_cache = True
    pp0.enable_hicache_storage = True
    pp0.anchor_tails = None
    pp0._pdflip_note_lost_anchors = lambda: None
    pp0.tree_cache = FullArenaTree(clock, **tree_kw)
    return g


def pass_with_poll(g):
    """One PP0 pass as metal runs it: pass-top hooks (harvest, stamp) and the
    forward, the /flush_cache poll it received, then the no-batch gap (bubble
    publisher); the followers' passes after it."""
    g1268.run_pass(g, 0)
    ok = g1268.poll(g)
    pdflip_bubble_publish.bubble_begin(g.ranks[0])
    g1268.run_pass(g, 1)
    g1268.run_pass(g, 2)
    g.clock.advance(0.002)
    return ok


class _Case(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.clock = g1268.FakeClock()
        for target in ("time.monotonic",):
            p = mock.patch(target, new=self.clock)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.dict(os.environ, {pdflip_bubble_publish.ENV: "1"})
        p.start()
        self.addCleanup(p.stop)


class TheMetalStaggering(_Case):
    def test_the_quiesce_is_answered_behind_a_full_arena(self):
        """RED on 424346f693: every lap is read ~11 s after its stamp and
        dropped as expired -- the quiesce never gets its 200 (W3)."""
        g = build(self.clock)
        answers = [pass_with_poll(g) for _ in range(8)]
        self.assertIn(True, answers, "the idle group's lap was never read fresh")
        self.assertGreaterEqual(g.ranks[0].tree_cache.resets, 1)

    def test_flush_publish_does_not_wait_on_a_round_that_cannot_progress(self):
        """issued=0 with nothing in flight: ONE sweep per poll, not four."""
        g = build(self.clock)
        t0 = self.clock()
        g1268.poll(g)
        self.assertEqual(g.ranks[0].tree_cache.sweeps, 1)
        self.assertLess(self.clock() - t0, 2 * SWEEP_S)


class TheRestStaysAsItWas(_Case):
    def test_writes_in_flight_are_still_joined_and_swept_again(self):
        g = build(self.clock, in_flight=3)
        g1268.poll(g)
        self.assertGreaterEqual(g.ranks[0].tree_cache.sweeps, 2)

    def test_a_round_that_issues_keeps_looping(self):
        g = build(self.clock, issue_every_round=4)
        g1268.poll(g)
        self.assertGreater(g.ranks[0].tree_cache.sweeps, 4)

    def test_time_outside_pp0_own_work_still_expires_the_lap(self):
        """The TTL is not switched off: a lap left lying past the TTL with no
        publish work in between is dropped as before."""
        g = build(self.clock)
        g.ranks[0].tree_cache.publish_unbacked_sweep = lambda **_kw: {"unbacked": 0}
        self.assertFalse(g1268.poll(g))
        g1268.run_rounds(g, 3)
        self.clock.advance(g1268._ttl_s() + 0.5)
        self.assertFalse(g1268.poll(g), "an expired lap answered")

    def test_own_work_counts_the_outermost_span_once(self):
        own_work = getattr(pdflip_idle_vote, "own_work", None)
        self.assertIsNotNone(own_work, "no own-work clock in pdflip_idle_vote")
        before = pdflip_idle_vote.own_work_total()
        with own_work():
            self.clock.advance(1.0)
            with own_work():
                self.clock.advance(0.5)
        self.assertAlmostEqual(pdflip_idle_vote.own_work_total() - before, 1.5, places=6)

    def test_the_expiry_names_wall_and_own_work(self):
        w = pdflip_idle_vote.PdFlipLapWitness(epoch=1, stamped_at=self.clock())
        own_work = getattr(pdflip_idle_vote, "own_work", None)
        self.assertIsNotNone(own_work)
        with own_work():
            self.clock.advance(5.0)
        self.clock.advance(0.5)
        self.assertEqual(
            pdflip_idle_vote.expiry_reason(w, now=self.clock(), ttl_s=2.0), ""
        )
        self.clock.advance(2.0)
        why = pdflip_idle_vote.expiry_reason(w, now=self.clock(), ttl_s=2.0)
        self.assertIn("expired", why)
        self.assertIn("own publish work", why)


if __name__ == "__main__":
    unittest.main()
