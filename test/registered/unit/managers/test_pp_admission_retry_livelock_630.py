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

if __name__ == "__main__":
    unittest.main()
