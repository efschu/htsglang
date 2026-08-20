# SPDX-License-Identifier: Apache-2.0
"""The #791 degrade must strictly advance, or it is #630 wearing a new hat.

THE HOLE. `reconcile_pp_admission_decision`'s unhonourable-case degrade
(#791) excludes a request and expects it to be re-queued and re-admitted on
a LATER pass. Nothing in that mechanism by itself forces the later pass to
ask for less. If PP0's own local state has not changed between passes (the
ordinary case -- nothing warms PP0's cache just because a downstream rank
retracted), PP0 re-derives the IDENTICAL `told`, tells the downstream rank
the SAME too-long length, and that rank retracts again. Forever. That is not
a crash; it is #630's family, on record in this codebase already (see
`test_pp_disk_hicache_guard_630.py`'s history: "a bounded/degrading path
that makes no forward progress IS the livelock defect," rooted there in a
bounded wait that polled a REPORT instead of ever consuming a DRIVING
signal, so two polling peers never advanced -- "the bound was the
livelock"). A request that never completes while the server looks alive is
exactly that shape.

THIS FILE IS RED-FIRST ON THE LOOP ITSELF, not on a single exclusion (#791
already pins that a single exclusion is safe). `TheLoopWithoutTheGuardTest`
drives the SAME request through the admission/reconcile cycle TWICE with no
mediation between the cycles -- i.e. exactly what happens today, since
nothing today carries a retraction's information back to the rank that
authors the next decision -- and asserts the SECOND pass reproduces the
identical `(told, local, retracted)` outcome as the first: a demonstrated
loop, not a coincidence. `TheGuardBreaksTheLoopTest` then re-runs the same
two-cycle drive WITH `PPAdmissionCongruenceGuard` mediating between them
(`record_return_trip` after cycle 1, consulted via `build_pp_admission_decision`'s
`guard=` parameter in cycle 2) and asserts the second pass carries a
strictly lower `told`, the request is SERVED, and the learned floor is
cleared afterward -- clearing is itself asserted, not just served-once,
because an uncleared floor would permanently cap that rid's reuse the next
time its cache state is warmer. Both tests also assert liveness (no
exception across either cycle) and that a sibling request riding in the same
decisions is unaffected throughout -- the same two properties #791's own
unhonourable-case tests pin, carried through a second cycle here because the
livelock is specifically a MULTI-CYCLE property that a single-cycle test
cannot see.

WHY PP0-LEARNS OVER A ONE-SHOT told=0 PIN. Both shapes terminate the loop (a
pin forcing told=0 makes a second retraction structurally impossible, since
an empty prefix is always <= any local match). This file exercises the
PP0-learns-coverage shape (`PPAdmissionCongruenceGuard`) because it is not
meaningfully harder to build -- the same one-int-per-rid state shape as a
pin -- and it keeps the degrade RARE (a rid one token short of `told` loses
one token of reuse next pass, not all of it) rather than merely making it
terminate. See `PPAdmissionCongruenceGuard`'s docstring in
`pp_admission_congruence.py` for the full argument.

NO COLLECTIVE, NO HAND-PINNED NUMBERS. `record_return_trip` and
`prefix_len_for` are plain dict reads/writes over values already produced by
`reconcile_pp_admission_decision` and `build_pp_admission_decision` -- the
same NO COLLECTIVE property #791's own structural test pins for the rest of
the module (that test already scans every top-level class in this module,
so it covers `PPAdmissionCongruenceGuard` too; not re-pinned here to avoid a
duplicate, weaker copy of that check). Every number below is either an input
this file chooses explicitly (the fixture's raw prefix/extend lengths) or a
value the code under test computed from it -- never a heuristic constant.
"""

from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang.srt.managers.pp_admission_congruence import (  # noqa: E402
    PPAdmissionCongruenceGuard,
    build_pp_admission_decision,
    reconcile_pp_admission_decision,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PP0, PP1, PP2 = 0, 1, 2
WORLD = 3


class _Req:
    """See test_pp_admission_congruence_791.py's `_Req` -- identical shape:
    `build_pp_admission_decision` reads only `rid`, `len(prefix_indices)`,
    and `extend_input_len`."""

    def __init__(self, rid, prefix_len, extend_len):
        self.rid = rid
        self.prefix_indices = list(range(prefix_len))
        self.extend_input_len = extend_len


def _reqs():
    """`req`: PP0's local match (120) exceeds what PP1's cache holds (64) --
    the unhonourable case. `sibling`: PP1's cache exactly meets told (30) on
    every cycle -- the safe, ordinary case, carried through untouched as the
    control that proves the guard's bookkeeping does not leak across rids."""
    return [
        _Req(rid="req", prefix_len=120, extend_len=80),
        _Req(rid="sibling", prefix_len=30, extend_len=170),
    ]


PP1_LOCAL_MATCH = {"req": 64, "sibling": 30}


class TheLoopWithoutTheGuardTest(CustomTestCase):
    """RED: two cycles, no guard, no mediation between them -- exactly
    today's behaviour, since nothing today carries cycle 1's retraction back
    to whichever rank authors cycle 2's decision. The second cycle must
    reproduce the FIRST cycle's exact failure, not just fail again by
    coincidence -- that reproduction is the loop."""

    def test_the_second_pass_repeats_the_identical_exclusion(self):
        decision0_c1 = build_pp_admission_decision(
            mb_id=0, reqs=_reqs(), pp_size=WORLD, guard=None
        )
        eff1_c1, decision1_c1 = reconcile_pp_admission_decision(
            decision0_c1, PP1_LOCAL_MATCH, rank=PP1, pp_size=WORLD
        )

        # Cycle 2: "later pass" re-admission, per #791's degrade contract --
        # PP0's own local state has not changed (nothing warmed it), so it
        # re-derives from the SAME raw fixture, with no guard consulted.
        decision0_c2 = build_pp_admission_decision(
            mb_id=1, reqs=_reqs(), pp_size=WORLD, guard=None
        )
        eff2_c2, decision1_c2 = reconcile_pp_admission_decision(
            decision0_c2, PP1_LOCAL_MATCH, rank=PP1, pp_size=WORLD
        )

        self.assertNotIn("req", eff1_c1, "cycle 1 must exclude the unhonourable rid")
        self.assertNotIn(
            "req",
            eff2_c2,
            "cycle 2 excludes it AGAIN -- with no guard, nothing forced a "
            "lower told, so it is unserved twice in a row",
        )

        entry_c1 = decision1_c1.by_rid()["req"]
        entry_c2 = decision1_c2.by_rid()["req"]
        self.assertEqual(
            (entry_c1.prefix_len, entry_c1.observed_local, entry_c1.retracted),
            (entry_c2.prefix_len, entry_c2.observed_local, entry_c2.retracted),
            "the SAME (told, local, retracted) outcome twice is the loop, "
            "not just 'it failed again' -- an identical repeat is what "
            "distinguishes 'stuck' from 'still degrading'",
        )
        self.assertEqual(entry_c1.prefix_len, 120, "told did not move between cycles")
        self.assertEqual(entry_c2.prefix_len, 120)

    def test_liveness_and_sibling_unaffected_across_the_looping_cycles(self):
        decision0_c1 = build_pp_admission_decision(
            mb_id=0, reqs=_reqs(), pp_size=WORLD, guard=None
        )
        decision0_c2 = build_pp_admission_decision(
            mb_id=1, reqs=_reqs(), pp_size=WORLD, guard=None
        )
        try:
            eff1, decision1 = reconcile_pp_admission_decision(
                decision0_c1, PP1_LOCAL_MATCH, rank=PP1, pp_size=WORLD
            )
            eff2, decision2 = reconcile_pp_admission_decision(
                decision0_c2, PP1_LOCAL_MATCH, rank=PP1, pp_size=WORLD
            )
        except Exception as exc:  # noqa: BLE001 -- absence of one is the point
            self.fail(f"reconcile raised {type(exc).__name__}: {exc}")

        for eff, decision in ((eff1, decision1), (eff2, decision2)):
            self.assertEqual(eff.get("sibling"), 30, "sibling served, both cycles")
            self.assertFalse(decision.by_rid()["sibling"].retracted)


class TheGuardBreaksTheLoopTest(CustomTestCase):
    """GREEN: the same two-cycle drive, mediated by `PPAdmissionCongruenceGuard`.
    Cycle 1's retraction is fed back via `record_return_trip`; cycle 2's
    `build_pp_admission_decision` consults the guard. The second pass must
    carry a STRICTLY lower `told` and the request must be SERVED -- and the
    learned floor must be gone afterward, or the guard would simply have
    swapped one permanent penalty (exclusion) for another (a frozen cap)."""

    def test_the_second_pass_advances_and_serves_the_request(self):
        guard = PPAdmissionCongruenceGuard()

        # Cycle 1: identical to the RED test -- an empty guard changes nothing.
        decision0_c1 = build_pp_admission_decision(
            mb_id=0, reqs=_reqs(), pp_size=WORLD, guard=guard
        )
        self.assertEqual(
            decision0_c1.by_rid()["req"].prefix_len,
            120,
            "an empty guard must not alter the first pass -- there is "
            "nothing yet to have learned",
        )
        eff1_c1, decision1_c1 = reconcile_pp_admission_decision(
            decision0_c1, PP1_LOCAL_MATCH, rank=PP1, pp_size=WORLD
        )
        self.assertNotIn("req", eff1_c1)

        # The return trip: this is the ONLY new wiring the guard needs --
        # feeding the SAME amended_decision the caller already forwards.
        guard.record_return_trip(decision1_c1)
        self.assertEqual(
            guard.outstanding_rids(),
            ("req",),
            "the guard must learn from the retraction, and ONLY the "
            "retracted rid -- sibling was served cleanly and must not "
            "appear here",
        )

        # Cycle 2: same raw fixture (PP0's own state is still unchanged --
        # this is the whole point), but now the guard clamps `told`.
        decision0_c2 = build_pp_admission_decision(
            mb_id=1, reqs=_reqs(), pp_size=WORLD, guard=guard
        )
        entry0_c2 = decision0_c2.by_rid()["req"]
        self.assertEqual(
            entry0_c2.prefix_len,
            64,
            "cycle 2 must ask for the learned floor (PP1's observed local "
            "match), STRICTLY less than cycle 1's told=120 -- this is the "
            "state advancing, not merely retrying",
        )
        self.assertLess(
            entry0_c2.prefix_len,
            decision0_c1.by_rid()["req"].prefix_len,
            "strict decrease: the formal termination property",
        )
        self.assertEqual(
            entry0_c2.prefix_len + entry0_c2.extend_len,
            120 + 80,
            "clamping told down must not shrink the request -- the "
            "difference moves to extend_len, the total token count is "
            "invariant",
        )

        eff2_c2, decision1_c2 = reconcile_pp_admission_decision(
            decision0_c2, PP1_LOCAL_MATCH, rank=PP1, pp_size=WORLD
        )
        self.assertEqual(
            eff2_c2.get("req"),
            64,
            "SERVED: local (64) now meets told (64) -- the safe-truncate "
            "branch, not another retraction",
        )
        entry1_c2 = decision1_c2.by_rid()["req"]
        self.assertFalse(entry1_c2.retracted, "no second exclusion")
        self.assertTrue(entry1_c2.admitted)

        # The return trip for the successful pass must clear the floor --
        # served-once, not a permanent cap.
        guard.record_return_trip(decision1_c2)
        self.assertEqual(
            guard.outstanding_rids(),
            (),
            "the learned floor must clear once the rid is served with no "
            "retraction anywhere in the chain -- an uncleared floor would "
            "permanently cap this rid's reuse even after its cache state "
            "improves",
        )

    def test_liveness_and_sibling_unaffected_across_both_guarded_cycles(self):
        guard = PPAdmissionCongruenceGuard()
        try:
            decision0_c1 = build_pp_admission_decision(
                mb_id=0, reqs=_reqs(), pp_size=WORLD, guard=guard
            )
            eff1, decision1 = reconcile_pp_admission_decision(
                decision0_c1, PP1_LOCAL_MATCH, rank=PP1, pp_size=WORLD
            )
            guard.record_return_trip(decision1)

            decision0_c2 = build_pp_admission_decision(
                mb_id=1, reqs=_reqs(), pp_size=WORLD, guard=guard
            )
            eff2, decision2 = reconcile_pp_admission_decision(
                decision0_c2, PP1_LOCAL_MATCH, rank=PP1, pp_size=WORLD
            )
            guard.record_return_trip(decision2)
        except Exception as exc:  # noqa: BLE001 -- absence of one is the point
            self.fail(f"reconcile/guard raised {type(exc).__name__}: {exc}")

        for eff, decision in ((eff1, decision1), (eff2, decision2)):
            self.assertEqual(
                eff.get("sibling"),
                30,
                "the sibling request must be served identically on both "
                "cycles -- the guard's bookkeeping for 'req' must not leak "
                "into an unrelated rid's told",
            )
            self.assertFalse(decision.by_rid()["sibling"].retracted)

        # And the process is demonstrably still usable afterwards, for a
        # third, unrelated decision -- the same liveness property #791 pins.
        decision0_c3 = build_pp_admission_decision(
            mb_id=2,
            reqs=[_Req(rid="other", prefix_len=10, extend_len=5)],
            pp_size=WORLD,
            guard=guard,
        )
        eff3, _ = reconcile_pp_admission_decision(
            decision0_c3, {"other": 10}, rank=PP1, pp_size=WORLD
        )
        self.assertEqual(eff3, {"other": 10})


class TerminationAcrossMoreThanOneDownstreamRankTest(CustomTestCase):
    """The worked example behind the termination argument: PP0 tells 120,
    PP1 can only honour 64 (learns floor=64), PP2 -- on the FOLLOWING pass,
    now offered told=64 -- can only honour 50 (tightens floor=50), and only
    on the pass after THAT does every rank finally agree. Three strictly
    decreasing tolds (120 -> 64 -> 50), never a repeat, then served."""

    def test_three_cycles_strictly_decreasing_then_served(self):
        guard = PPAdmissionCongruenceGuard()
        req = [_Req(rid="req", prefix_len=120, extend_len=80)]
        tolds = []

        # Cycle 1: PP1 can only honour 64.
        d0 = build_pp_admission_decision(mb_id=0, reqs=req, pp_size=WORLD, guard=guard)
        tolds.append(d0.by_rid()["req"].prefix_len)
        _eff, d1 = reconcile_pp_admission_decision(
            d0, {"req": 64}, rank=PP1, pp_size=WORLD
        )
        self.assertTrue(d1.by_rid()["req"].retracted)
        guard.record_return_trip(d1)

        # Cycle 2: told is now clamped to 64. PP1 honours it (its match is
        # stable at 64), but PP2 -- seeing this decision for the first time
        # on this pass -- can only honour 50.
        d0 = build_pp_admission_decision(mb_id=1, reqs=req, pp_size=WORLD, guard=guard)
        tolds.append(d0.by_rid()["req"].prefix_len)
        self.assertEqual(d0.by_rid()["req"].prefix_len, 64)
        _eff1, d1 = reconcile_pp_admission_decision(
            d0, {"req": 64}, rank=PP1, pp_size=WORLD
        )
        self.assertFalse(d1.by_rid()["req"].retracted, "PP1 honours told=64")
        _eff2, d2 = reconcile_pp_admission_decision(
            d1, {"req": 50}, rank=PP2, pp_size=WORLD
        )
        self.assertTrue(d2.by_rid()["req"].retracted, "PP2 cannot honour told=64")
        guard.record_return_trip(d2)

        # Cycle 3: told is now clamped to 50. Both PP1 and PP2 honour it.
        d0 = build_pp_admission_decision(mb_id=2, reqs=req, pp_size=WORLD, guard=guard)
        tolds.append(d0.by_rid()["req"].prefix_len)
        self.assertEqual(d0.by_rid()["req"].prefix_len, 50)
        eff1, d1 = reconcile_pp_admission_decision(
            d0, {"req": 64}, rank=PP1, pp_size=WORLD
        )
        eff2, d2 = reconcile_pp_admission_decision(
            d1, {"req": 50}, rank=PP2, pp_size=WORLD
        )
        guard.record_return_trip(d2)

        self.assertEqual(tolds, [120, 64, 50], "strictly decreasing, never repeated")
        self.assertEqual(eff2.get("req"), 50, "served on the third pass")
        self.assertFalse(d2.by_rid()["req"].retracted)
        self.assertEqual(guard.outstanding_rids(), (), "floor cleared once served")


if __name__ == "__main__":
    unittest.main()
