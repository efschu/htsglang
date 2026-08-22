# Copyright 2026 SGLang Team
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
"""#662-F4 / A0: THE PLANNER MUST SOLVE THE ARMING FLOOR, AND THE ARM MUST SPILL
FOR IT BEFORE REFUSING FOR IT.

THE MEASUREMENT, from metal on 2026-08-15 (boot_maxfill.log). With a COLD seam
record the sizer charged NO flip term at all and filled the pool to the corridor
law. Rank 1 came up with ~875 MiB free against an arming floor of ~1536 MiB, so
tp_to_pp was refused on EVERY arm, every prefill stayed in the TP layout, and
nothing at runtime could recover it -- the pool is fixed at boot. The operator
had to hand-pin --max-total-tokens to work around it.

THE MECHANISM, exactly. ``SeamReserve.active`` is False when every MEASURED
field is zero, which is precisely a first boot, and
``seam_adjusted_budget_bytes`` returned the budget unchanged on that branch. So
the boot with the LEAST information about the seam was the boot that reserved
nothing for it.

TWO THINGS WERE CONFLATED, AND SEPARATING THEM IS THE FIX:

  the seam RESERVE   what a flip COSTS WHILE IT RUNS. Genuinely knowable only
                     by measurement; legitimately zero on a first boot.
  the arming FLOOR   the LEVEL the gate compares against. Derived from the
                     corridor law the operator already stated, and knowable at
                     boot on every rig.

A cold record means the first is unknown. It never meant the second was. So the
floor is charged on every flip boot, cold or stored, as a hard per-rank
subtrahend -- and because the sizer already holds the law free, what the pool
owes is the DIFFERENCE, making the per-rank free target
``max(law, arming floor + load margin)``.

AND AT RUNTIME, THE FLOOR IS SPILLED FOR BEFORE IT IS REFUSED FOR. A correctly
planned instance never reaches that path; what reaches it is the transient a
sizer cannot see -- a co-tenant, a capture peak, a rank that drifted. Refusing
an arm for a condition one ladder call could fix turns a recoverable dip into a
phase the instance cannot enter, which is the same defect one layer up.

THE ORDER MATTERS AND IS THE HARD CONSTRAINT. Relief runs strictly BEFORE the
arm: nothing is armed, no rank has entered the seam, no collective has been
reached, and the staged fund does not exist yet to be pulled out from under.
Spilling once the flip has begun is the evictable-seam-fund mistake that
produced the served-nothing class in #656 boots E/G.

Hermetic: pure functions for the sizing half, a stub runtime and injected guard
for the arming half. No CUDA, no model, no collectives.
"""

from __future__ import annotations

import os
import tempfile
import types
import unittest

from sglang.srt.managers import phase_flip_runtime as pfr
from sglang.srt.managers import phase_flip_seam_reserve as sr
from sglang.srt.managers import phase_flip_spill
from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

MIB = 1024 * 1024
GIB = 1024 * MIB

PP_TO_TP = "pp_to_tp"
TP_TO_PP = "tp_to_pp"


class _Env:
    def __init__(self, **env):
        self.env = {k: (None if v is None else str(v)) for k, v in env.items()}

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.env}
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


# ---------------------------------------------------------------------------
# A0-a: the sizer charges the floor, cold included.
# ---------------------------------------------------------------------------


class TheColdRecordIsTheBootThatNeedsTheFloorMostTest(unittest.TestCase):
    """The boot with the least information reserved the least. That was the
    defect, and it is the one a revert would restore."""

    def _cold(self):
        return sr.SeamReserve(provenance=sr.PROVENANCE_COLD)

    def test_a_cold_record_is_not_active(self):
        """Pinned because the whole defect hangs off this predicate."""
        self.assertFalse(self._cold().active)

    def test_a_cold_boot_with_flips_ON_still_gives_up_the_floor(self):
        budget = 20 * GIB
        floor = sr.arming_floor_target_bytes()
        new_bytes, allowed = sr.seam_adjusted_budget_bytes(
            budget, 4096, self._cold(), arming_floor_bytes=floor
        )
        self.assertEqual(allowed, 0, "a cold record still measures no staging")
        self.assertLess(
            new_bytes,
            budget,
            "the boot that measured nothing must still reserve the arming "
            "floor -- this is the metal defect, in one assertion",
        )
        self.assertEqual(new_bytes, budget - sr.arming_floor_subtrahend_bytes(floor))

    def test_the_charge_is_the_difference_not_the_whole_floor(self):
        """The sizer already holds the law free; charging it twice would cost
        the pool a gigabyte it does not owe -- the same double-count
        ``required_free_bytes`` refuses with its ``max``."""
        floor = sr.arming_floor_target_bytes()
        law = sr._corridor_law_bytes()
        self.assertEqual(sr.arming_floor_subtrahend_bytes(floor), floor - law)
        self.assertLess(sr.arming_floor_subtrahend_bytes(floor), floor)

    def test_a_floor_under_the_law_costs_the_pool_nothing(self):
        """A rig whose arming floor sits below its corridor is sized exactly
        as before -- the term can only ever ADD to what the law already
        reserves."""
        law = sr._corridor_law_bytes()
        self.assertEqual(sr.arming_floor_subtrahend_bytes(law // 2), 0)
        budget = 20 * GIB
        got, _ = sr.seam_adjusted_budget_bytes(
            budget, 4096, sr.SeamReserve(), arming_floor_bytes=law // 2
        )
        self.assertEqual(got, budget)

    def test_the_floor_target_is_the_gate_watermark_plus_a_load_margin(self):
        """Derived from the law, never a constant carried between rigs."""
        from sglang.srt.managers import corridor_guard as cg

        with _Env(SGLANG_PHASE_FLIP_ARMING_MARGIN_MIB="192"):
            self.assertEqual(
                sr.arming_floor_target_bytes(),
                (int(cg.arming_floor_mib()) << 20) + (192 << 20),
            )

    def test_the_operators_override_RAISES_the_floor_the_pool_reserves_for(self):
        """CAUGHT ON THIS RIG BEFORE THE PROOF BOOT, and it would have been
        silent. The live instance sets SGLANG_CORRIDOR_FLOOR_MIB=1536 while the
        derived floor is 1331. A sizer that reserved for the derived number
        would leave every rank 205 MiB short of the floor its own gate arms at
        -- a healthy-looking boot that simply never flips, which is the exact
        defect this whole change exists for, reintroduced one level down.
        """
        with _Env(SGLANG_CORRIDOR_FLOOR_MIB="4096"):
            self.assertEqual(sr.configured_arming_floor_mib(), 4096)
            self.assertEqual(
                sr.arming_floor_target_bytes(configured_mib=4096),
                (4096 << 20) + sr._arming_margin_bytes(),
            )

    def test_the_floor_is_resolved_HIGHEST_WINS_like_the_guard(self):
        """A floor below the real draw launders breaches as passed checks, so
        a configured value may raise the derived one and may never lower it --
        the guard's own rule, applied to the number the pool reserves."""
        from sglang.srt.managers import corridor_guard as cg

        derived = cg.arming_floor_mib()
        low = sr.arming_floor_target_bytes(configured_mib=1)
        self.assertEqual(low, (derived << 20) + sr._arming_margin_bytes())

    def test_a_measured_seam_draw_raises_it_too(self):
        """The gate arms at law + the MEASURED draw where one exists; the pool
        must reserve for the same, or the two disagree by the draw."""
        big = sr.arming_floor_target_bytes(measured_draw_mib=4096)
        small = sr.arming_floor_target_bytes(measured_draw_mib=0)
        self.assertGreater(big, small)

    def test_a_malformed_override_is_not_a_floor_of_zero(self):
        with _Env(SGLANG_CORRIDOR_FLOOR_MIB="wide open"):
            self.assertEqual(sr.configured_arming_floor_mib(), 0)

    def test_a_malformed_margin_falls_back_to_the_default_not_to_zero(self):
        """The failure this exists to prevent is a pool with no margin, so an
        unparsable override must not produce one."""
        with _Env(SGLANG_PHASE_FLIP_ARMING_MARGIN_MIB="banana"):
            self.assertEqual(
                sr._arming_margin_bytes(), sr.DEFAULT_ARMING_MARGIN_MIB << 20
            )


class TheDefaultIsByteIdenticalTest(unittest.TestCase):
    """Every non-flip boot, and every caller that passes no floor."""

    def test_no_floor_given_means_the_previous_arithmetic_exactly(self):
        budget = 20 * GIB
        for reserve in (
            sr.SeamReserve(provenance=sr.PROVENANCE_COLD),
            sr.SeamReserve(provenance=sr.PROVENANCE_DISABLED),
        ):
            with self.subTest(provenance=reserve.provenance):
                self.assertEqual(
                    sr.seam_adjusted_budget_bytes(budget, 4096, reserve),
                    (budget, 0),
                )

    def test_a_WARM_record_still_gives_up_the_floor(self):
        """THE ORDERING DEFECT, AS A TEST. Caught on boot_slo_proof_r3.

        The charge used to be applied to the budget BEFORE
        ``min(budget, allowed * cell)``. That is not equivalent: whenever the
        seam solve binds -- which is whenever a MEASURED record exists -- the
        min picks ``allowed * cell`` and the subtraction is silently discarded.

        So the term worked on the COLD boot that wrote the record and did
        nothing at all on the next one. Measured: pool 385927 -> 491445, free
        landing at 1515/3130/1983 MiB against guard floors of 1772/1964/2414 --
        below the floor on two of three ranks, the exact condition the term
        exists to prevent. A charge that only fires on a first boot is worse
        than no charge, because the second boot looks like the fixed one.
        """
        # A record whose solve binds well below the budget.
        reserve = sr.SeamReserve(
            fixed_bytes=455 * MIB,
            have_bytes=2000 * MIB,
            id_space=100000,
            provenance=sr.PROVENANCE_STORED,
        )
        self.assertTrue(reserve.active)
        cell = 4096
        budget = 20 * GIB
        floor = sr.arming_floor_target_bytes()
        # #678: the charge is now the shortfall against what the SEAM SOLVE
        # already leaves free, not against the law alone. The property under
        # test is unchanged -- whatever the charge is, the min must not throw
        # it away -- so it is computed the same way the code computes it
        # rather than restated as a constant.
        charge = sr.arming_floor_subtrahend_bytes(
            floor, sr.seam_solve_reserved_free_bytes(reserve)
        )
        self.assertGreater(charge, 0, "there must be a charge left to discard")

        without, allowed = sr.seam_adjusted_budget_bytes(budget, cell, reserve)
        self.assertLess(
            allowed * cell, budget, "the seam solve must BIND for this to test"
        )
        with_floor, _ = sr.seam_adjusted_budget_bytes(
            budget, cell, reserve, arming_floor_bytes=floor
        )
        self.assertEqual(
            with_floor,
            without - charge,
            "a warm record must give up the floor too -- charging before the "
            "min lets the min throw the charge away",
        )

    def test_a_measured_record_still_charges_its_measured_staging(self):
        """The floor is ADDITIONAL to the seam reserve, not a replacement:
        they are different quantities at different instants."""
        reserve = sr.SeamReserve(
            fixed_bytes=455 * MIB,
            have_bytes=2000 * MIB,
            id_space=400000,
            provenance=sr.PROVENANCE_STORED,
        )
        self.assertTrue(reserve.active)
        without, a1 = sr.seam_adjusted_budget_bytes(20 * GIB, 4096, reserve)
        with_floor, a2 = sr.seam_adjusted_budget_bytes(
            20 * GIB, 4096, reserve, arming_floor_bytes=sr.arming_floor_target_bytes()
        )
        self.assertEqual(a1, a2, "the measured staging solve is unchanged")
        self.assertLessEqual(with_floor, without)


# ---------------------------------------------------------------------------
# A0-b: the arm spills for the floor before refusing for it.
# ---------------------------------------------------------------------------


#: The floor the stub guard arms at, so the tests do not move when the rig's
#: corridor law is retimed.
FLOOR = 1523 * MIB


class _Result:
    def __init__(self, ok, free_after, reclaimed, providers):
        self.ok = ok
        self.free_after = free_after
        self.reclaimed = reclaimed
        self.used_providers = providers


class _Guard:
    """A ladder that frees ``deliverable`` bytes and no more."""

    def __init__(self, free, deliverable=0, law=1024 * MIB, providers=("host",)):
        self.free = free
        self.deliverable = deliverable
        self.law_floor_bytes = law
        #: THE GATE'S OWN ARMING FLOOR. The relief reads this rather than
        #: re-deriving it, so the stub carries it too -- a test that patched a
        #: derivation instead would pass while the runtime spilled for a level
        #: nothing enforces.
        self.floor_bytes = FLOOR
        self.providers = providers
        self.asks = []

    def free_bytes(self):
        return self.free

    def ensure_headroom(self, want, *, reason="", refusal_is_fatal=False):
        """The real contract: free_after - want >= law, i.e. the ladder spends
        up to ``law + want`` and reports whether it got there. Modelled
        faithfully because the ASK this code computes only makes sense against
        this semantics -- a stub that freed merely ``want`` would have made a
        correct ask look wrong."""
        self.asks.append((int(want), reason))
        needed = max(0, (self.law_floor_bytes + int(want)) - self.free)
        freed = min(int(self.deliverable), needed)
        self.free += freed
        ok = (self.free - int(want)) >= self.law_floor_bytes
        return _Result(ok, self.free, freed, self.providers if freed else ())


class _PatchedGuard:
    def __init__(self, guard):
        self.g = guard

    def __enter__(self):
        self.old = phase_flip_spill.get_corridor_guard
        phase_flip_spill.get_corridor_guard = lambda _s: self.g
        return self.g

    def __exit__(self, *exc):
        phase_flip_spill.get_corridor_guard = self.old
        return False


def _runtime():
    r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    r._census_scheduler = object()
    return r


class TheFloorIsSpilledForBeforeItIsRefusedForTest(unittest.TestCase):
    def test_a_card_already_holding_the_floor_asks_the_ladder_for_nothing(self):
        """The common case on a correctly sized instance: a no-op."""
        r = _runtime()
        g = _Guard(free=FLOOR + 100 * MIB)
        with _PatchedGuard(g):
            ok, msg = r._prearm_floor_relief(TP_TO_PP)
        self.assertTrue(ok)
        self.assertEqual(msg, "")
        self.assertEqual(g.asks, [], "no dip, no ladder")

    def test_a_short_floor_with_relief_available_SPILLS_and_then_arms(self):
        """The operator's first required shape: relief available -> flip goes."""
        r = _runtime()
        g = _Guard(free=900 * MIB, deliverable=2 * GIB)
        with _PatchedGuard(g):
            ok, msg = r._prearm_floor_relief(TP_TO_PP)
        self.assertTrue(ok, msg)
        self.assertEqual(len(g.asks), 1, "the ladder was asked exactly once")

    def test_the_ask_is_the_floor_minus_the_law_the_guard_already_defends(self):
        """Asking for the whole floor would spill a second corridor for
        nothing: ensure_headroom already guarantees free_after - want >= law."""
        r = _runtime()
        g = _Guard(free=900 * MIB, deliverable=2 * GIB)
        with _PatchedGuard(g):
            r._prearm_floor_relief(TP_TO_PP)
        self.assertEqual(g.asks[0][0], FLOOR - g.law_floor_bytes)

    def test_a_short_floor_with_the_host_tier_FULL_still_lets_the_arm_PROCEED(self):
        """THE SPLIT-ARM DEFECT, AS A TEST. Added after boot_slo_proof_r3.

        The first version returned False here, arguing that "a rank that
        refuses simply does not arm, which the consensus round already
        handles". On metal PP0 sat at 3130 MiB against its 1728 MiB floor and
        armed, while PP1 and PP2 sat below theirs and refused -- and the armed
        rank parked at the entry, "WITHHOLDING presence (8854 rounds so far)",
        spinning for ever with all three cards at 0%.

        A verdict keyed to this rank's FREE VRAM is never group-uniform, so
        this rung may spend the ladder and may not decide. The seam gate
        refuses, and it reduces its verdict.
        """
        r = _runtime()
        g = _Guard(free=900 * MIB, deliverable=0)
        with _PatchedGuard(g):
            ok, msg = r._prearm_floor_relief(TP_TO_PP)
        self.assertTrue(
            ok,
            "a rank-local shortfall must NOT refuse the arm -- that splits the "
            "group and parks whichever rank did clear its floor",
        )
        self.assertEqual(len(g.asks), 1, "it must still have spent the ladder")

    def test_the_shortfall_is_reported_with_the_numbers(self):
        """The operator asked for how much was short and what the ladder
        freed. That still has to be in the log; it just must not be a
        decision."""
        r = _runtime()
        g = _Guard(free=900 * MIB, deliverable=0)
        with _PatchedGuard(g):
            with self.assertLogs(
                "sglang.srt.managers.phase_flip_runtime", level="WARNING"
            ) as cm:
                r._prearm_floor_relief(TP_TO_PP)
        joined = "\n".join(cm.output)
        self.assertIn("900", joined)
        self.assertIn("1523", joined)
        self.assertIn("freed", joined)

    def test_arm_does_not_branch_on_the_relief_verdict(self):
        """Structural, not a promise: the caller must discard the bool.

        Asserted on the source because "this value is not used to decide" is
        not observable from a single call -- and a future edit that re-adds the
        branch would restore the split arm.
        """
        import inspect

        src = inspect.getsource(PhaseFlipRuntime.arm)
        self.assertIn("self._prearm_floor_relief(direction)", src)
        self.assertNotIn("if not floor_ok", src)

    def test_the_relief_is_bounded_and_stops_asking(self):
        """An unbounded relief loop spills the instance flat.

        The bound stops the LADDER, not the flip: every one of these calls
        still lets the arm proceed, because a rank-local shortfall may not
        decide for the group.
        """
        r = _runtime()
        g = _Guard(free=900 * MIB, deliverable=0)
        with (
            _PatchedGuard(g),
            _Env(SGLANG_PHASE_FLIP_PREARM_RELIEF_ATTEMPTS="2"),
        ):
            for _ in range(6):
                ok, _msg = r._prearm_floor_relief(TP_TO_PP)
                self.assertTrue(ok)
        self.assertEqual(len(g.asks), 2, "the bound must stop the ladder")

    def test_the_bound_is_per_direction(self):
        r = _runtime()
        g = _Guard(free=900 * MIB, deliverable=0)
        with (
            _PatchedGuard(g),
            _Env(SGLANG_PHASE_FLIP_PREARM_RELIEF_ATTEMPTS="1"),
        ):
            r._prearm_floor_relief(TP_TO_PP)
            r._prearm_floor_relief(TP_TO_PP)
            r._prearm_floor_relief(PP_TO_TP)
        self.assertEqual(len(g.asks), 2, "one attempt for each direction")

    def test_a_recovered_floor_clears_the_count(self):
        """A rig that dips once pays one ladder, not a permanent debt."""
        r = _runtime()
        g = _Guard(free=900 * MIB, deliverable=0)
        with (
            _PatchedGuard(g),
            _Env(SGLANG_PHASE_FLIP_PREARM_RELIEF_ATTEMPTS="1"),
        ):
            r._prearm_floor_relief(TP_TO_PP)
            g.free = FLOOR + MIB
            ok, _ = r._prearm_floor_relief(TP_TO_PP)
            self.assertTrue(ok)
            g.free = 900 * MIB
            r._prearm_floor_relief(TP_TO_PP)
        self.assertEqual(len(g.asks), 2, "the count reset when the floor cleared")

    def test_a_zero_bound_falls_back_to_the_default_rather_than_refusing_all(self):
        r = _runtime()
        g = _Guard(free=900 * MIB, deliverable=2 * GIB)
        with (
            _PatchedGuard(g),
            _Env(SGLANG_PHASE_FLIP_PREARM_RELIEF_ATTEMPTS="0"),
        ):
            ok, _ = r._prearm_floor_relief(TP_TO_PP)
        self.assertTrue(ok, "a zero bound must not refuse without asking")


class AnUnreadableInstrumentMayNotBlockAFlipTest(unittest.TestCase):
    """Failing to MEASURE the floor is not evidence that it is SHORT, and
    refusing on a probe failure turns a bad reading into a stuck phase."""

    def test_no_scheduler_means_the_arm_proceeds(self):
        r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
        r._census_scheduler = None
        self.assertEqual(r._prearm_floor_relief(TP_TO_PP), (True, ""))

    def test_a_raising_probe_means_the_arm_proceeds(self):
        r = _runtime()

        class _Boom:
            law_floor_bytes = 1024 * MIB
            floor_bytes = FLOOR

            def free_bytes(self):
                raise RuntimeError("nvml is having a day")

        with _PatchedGuard(_Boom()):
            ok, _ = r._prearm_floor_relief(TP_TO_PP)
        self.assertTrue(ok)

    def test_a_raising_ladder_means_the_arm_proceeds_on_the_seam_gate(self):
        r = _runtime()

        class _BoomLadder(_Guard):
            def ensure_headroom(self, want, *, reason="", refusal_is_fatal=False):
                raise RuntimeError("the ladder fell over")

        with _PatchedGuard(_BoomLadder(free=900 * MIB)):
            ok, _ = r._prearm_floor_relief(TP_TO_PP)
        self.assertTrue(ok)

    def test_it_can_be_switched_off_entirely(self):
        r = _runtime()
        g = _Guard(free=900 * MIB, deliverable=0)
        with (
            _PatchedGuard(g),
            _Env(SGLANG_PHASE_FLIP_PREARM_RELIEF="0"),
        ):
            ok, msg = r._prearm_floor_relief(TP_TO_PP)
        self.assertEqual((ok, msg), (True, ""))
        self.assertEqual(g.asks, [])


class TheOrderIsTheConstraintTest(unittest.TestCase):
    """RELIEF BEFORE THE ARM, NEVER INSIDE THE FLIP WINDOW.

    Asserted on the source, because "this runs before that" is not observable
    from a single call -- the same reason the corridor gate's own ordering is
    pinned that way.
    """

    def test_arm_consults_the_relief_before_it_commits_the_direction(self):
        import inspect

        src = inspect.getsource(PhaseFlipRuntime.arm)
        self.assertIn("_prearm_floor_relief", src)
        self.assertLess(
            src.index("_prearm_floor_relief"),
            src.index("self._pending = direction"),
            "relief must run while refusing is still free -- once the "
            "direction is pending the group is committing to a flip",
        )

    def test_the_relief_adds_no_collective(self):
        """A rank that refuses here simply does not arm, which the consensus
        round already handles. Adding a reduction would be a new way to
        desync the group at the one moment it is not yet agreed."""
        import inspect

        src = inspect.getsource(PhaseFlipRuntime._prearm_floor_relief)
        self.assertNotIn("all_reduce", src)
        self.assertNotIn("_collective_min", src)

    def test_the_seam_gate_still_owns_the_in_window_ladder(self):
        """This rung is ADDITIONAL to the gate, not a replacement: the gate
        still runs at the last point before the no-return region."""
        import inspect

        src = inspect.getsource(PhaseFlipRuntime._seam_funding_verdict)
        self.assertIn("_corridor_gate", src)


class TheKnobsAreReadableTest(unittest.TestCase):
    def test_the_relief_defaults_on(self):
        with _Env(SGLANG_PHASE_FLIP_PREARM_RELIEF=None):
            self.assertTrue(pfr._prearm_relief_enabled())

    def test_a_malformed_bound_falls_back_to_the_default(self):
        with _Env(SGLANG_PHASE_FLIP_PREARM_RELIEF_ATTEMPTS="lots"):
            self.assertEqual(
                pfr._prearm_relief_attempts(), pfr.DEFAULT_PREARM_RELIEF_ATTEMPTS
            )


if __name__ == "__main__":
    unittest.main()


class TheFloorMustNotBePaidTwiceTest(unittest.TestCase):
    """#678: THE CALIBRATION DEFECT, and it cost 47% of the pool.

    The operator's steer: the gap between the unpinned solve (284181 tokens)
    and the empirical bound (550000, which arms, flips 219+ times and clears
    its guard) is a CALIBRATION DEFECT, not a trade to accept. The corridor law
    demands best-filled -- twice too much free is the same defect class as too
    little -- and a permanent hand-pin would violate the "must work
    automatically" order that produced the floor charge in the first place.

    TWO THINGS WERE BEING PAID TWICE, and they are separable.

    1. THE FLOOR ITSELF. ``seam_allowed_tokens`` targets equality on
       ``have(T) >= need(T)`` where ``have`` was measured as
       ``free - band_floor + rung_fund``, so a boot with a measured record is
       already sized to leave about ``band_floor + seam_draw`` free at rest.
       That is the arming floor, arrived at from the other side. Charging the
       whole floor again on top is the second payment.

    2. THE CROSS-LEG SUM. ``total_fixed_bytes`` adds two maxima that are taken
       over BOTH directions and, on every record this rig has written, are
       maxed by DIFFERENT ones -- the arena tail only on tp_to_pp, the
       drafter's restore only on pp_to_tp. See TheDrawIsOneLegTest.

    WHAT IS DELIBERATELY *NOT* RESERVED, because it is the third payment: the
    ``rung_fund`` term. The solve counts the KV rung as a payer at seam time
    while the gate wants free VRAM at arm time, so a gap of that size can
    remain -- and the pre-arm relief ladder exists precisely to cover it. A
    floor priced at worst case PLUS a ladder is double insurance paid twice.
    """

    def _warm(self):
        return sr.SeamReserve(
            fixed_bytes=139 * MIB,
            arena_fixed_bytes=1456 * MIB,
            worst_leg_fixed_bytes=1456 * MIB,
            per_row_bytes=553.6,
            have_bytes=1772 * MIB,
            id_space=406600,
            provenance=sr.PROVENANCE_STORED,
        )

    def test_a_measured_record_has_already_reserved_the_floor_ITSELF(self):
        """band floor + this rank's one-leg draw, which is the arming floor
        stated from the other side. What is left to charge is the load margin
        and nothing else."""
        r = self._warm()
        reserved = sr.seam_solve_reserved_free_bytes(r)
        floor = sr.arming_floor_target_bytes(
            measured_draw_mib=r.arming_draw_bytes() >> 20
        )
        self.assertAlmostEqual(
            (floor - reserved) / MIB,
            sr._arming_margin_bytes() / MIB,
            delta=1.0,
        )

    def test_the_error_bar_is_NOT_counted_as_reserved(self):
        """MEASURED, boot_678_final.log. Counting the solve's own margin drives
        the charge to zero and the pool to 537076 -- and the cards then came up
        at 987/2286/1475 MiB against arming floors of 1536/1633/2275, below the
        floor on two of three ranks, with the pre-arm ladder finding 40-46 MiB
        against a 650-726 MiB gap.

        The excluded ``rung_fund`` term is why: the solve counts the KV rung as
        a payer and lets the resting free column land that much lower, so the
        paper margin is already spent. A pool whose cards cannot hold their own
        arming floor is the defect this whole term exists to prevent.
        """
        r = self._warm()
        self.assertEqual(
            sr.seam_solve_reserved_free_bytes(r),
            sr._band_floor_bytes(sr._corridor_law_bytes()) + r.arming_draw_bytes(),
            "the solve's own margin must not be counted as headroom the gate "
            "can rely on",
        )

    def test_the_charge_collapses_to_the_load_margin(self):
        r = self._warm()
        floor = sr.arming_floor_target_bytes(
            measured_draw_mib=r.arming_draw_bytes() >> 20
        )
        charge = sr.arming_floor_subtrahend_bytes(
            floor, sr.seam_solve_reserved_free_bytes(r)
        )
        law_only = sr.arming_floor_subtrahend_bytes(floor)
        self.assertEqual(charge, sr._arming_margin_bytes())
        self.assertLess(
            charge,
            law_only // 4,
            "a record that already paid must not be charged the whole floor",
        )

    def test_a_COLD_record_still_pays_in_FULL(self):
        """The r2 behaviour must not regress. A cold solve reserves nothing
        for the seam, so nothing has been paid and the floor is owed whole --
        that is the boot that landed every rank above its floor."""
        cold = sr.SeamReserve(provenance=sr.PROVENANCE_COLD)
        floor = sr.arming_floor_target_bytes()
        self.assertEqual(sr.seam_solve_reserved_free_bytes(cold), 0)
        self.assertEqual(
            sr.arming_floor_subtrahend_bytes(
                floor, sr.seam_solve_reserved_free_bytes(cold)
            ),
            sr.arming_floor_subtrahend_bytes(floor),
        )

    def test_the_baseline_never_drops_below_the_law(self):
        """A tiny or absent reservation may not make the charge LARGER than
        the law-only one -- the law is held free on every boot regardless."""
        floor = sr.arming_floor_target_bytes()
        self.assertEqual(
            sr.arming_floor_subtrahend_bytes(floor, 1),
            sr.arming_floor_subtrahend_bytes(floor),
        )

    def test_the_pool_recovers_most_of_what_the_double_charge_took(self):
        """The number the operator is judging this by."""
        r = self._warm()
        floor = sr.arming_floor_target_bytes(
            configured_mib=1536, measured_draw_mib=r.arming_draw_bytes() >> 20
        )
        double = sr.arming_floor_subtrahend_bytes(floor)
        once = sr.arming_floor_subtrahend_bytes(
            floor, sr.seam_solve_reserved_free_bytes(r)
        )
        self.assertGreater(
            (double - once) / MIB,
            1000,
            "the double charge was worth more than a GiB on this rank alone",
        )


class TheDrawIsOneLegTest(unittest.TestCase):
    """#678: 1595 MiB is not a draw any seam makes.

    ``measure_at_rest`` keeps two maxima -- ``arena_fixed`` and ``fixed`` --
    each taken over BOTH directions, and ``total_fixed_bytes`` sums them. On
    every record this rig has written they are maxed by different legs:

        rank 2, record 03d16efef3ad
          tp_to_pp: arena tail 1456 + draft restore    0  = 1456 MiB
          pp_to_tp: arena tail    0 + draft restore  139  =  139 MiB
          total_fixed_bytes (the sum of the two maxima)   = 1595 MiB

    A gate that arms per flip must be priced at what one flip draws.
    """

    def test_the_sum_of_the_maxima_overstates_the_worst_leg(self):
        r = sr.SeamReserve(
            fixed_bytes=139 * MIB,
            arena_fixed_bytes=1456 * MIB,
            worst_leg_fixed_bytes=1456 * MIB,
        )
        self.assertEqual(r.total_fixed_bytes >> 20, 1595)
        self.assertEqual(r.arming_draw_bytes() >> 20, 1456)

    def test_a_pre_678_record_falls_back_to_the_old_number(self):
        """Absent means unknown, and unknown must price the OLD way rather
        than optimistically -- an old record is readable, not cheaper."""
        old = sr.SeamReserve(fixed_bytes=139 * MIB, arena_fixed_bytes=1456 * MIB)
        self.assertEqual(old.worst_leg_fixed_bytes, 0)
        self.assertEqual(old.arming_draw_bytes(), old.total_fixed_bytes)

    def test_the_gate_and_the_sizer_read_the_SAME_accessor(self):
        """Two numbers that must be equal, computed in two places, is the
        defect 48ba9fe72a already fixed once. Pinned on the source."""
        import inspect

        from sglang.srt.managers import phase_flip_spill

        guard_src = inspect.getsource(phase_flip_spill._measured_seam_draw_mib)
        # The RETURN, not merely a mention: the prose below it explains why
        # total_fixed_bytes is the wrong number, so a bare "not in" would fail
        # on the explanation.
        self.assertIn("return int(reserve.arming_draw_bytes())", guard_src)
        self.assertNotIn("return int(reserve.total_fixed_bytes)", guard_src)

    def test_a_record_round_trips_the_new_term(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            args = types.SimpleNamespace(_measured_kv_budget_registry_path=d)
            path = sr.write_seam_reserve(
                args,
                2,
                139 * MIB,
                553.6,
                "detail",
                have_bytes=1772 * MIB,
                id_space=406600,
                arena_fixed_bytes=1456 * MIB,
                worst_leg_fixed_bytes=1456 * MIB,
            )
            with open(path) as fh:
                self.assertEqual(json.load(fh)["worst_leg_fixed_bytes"], 1456 * MIB)
            back = sr.read_seam_reserve(args, 2)
            self.assertEqual(back.arming_draw_bytes() >> 20, 1456)


class TheFloorIsSOLVEDNotApproximatedTest(unittest.TestCase):
    """#678 remainder: the constraint is an equality, so solve it.

    THE SUBTRAHEND COULD NOT STATE THE CONSTRAINT. What the pool gives up and
    what the card ends up holding free are related by the sizer's other posts,
    and the gap between them is where both failures of this ticket lived:

        284181 tokens  the subtrahend double-charged a floor the seam solve
                       had already reserved
        537076 tokens  the double charge removed, and two of three ranks came
                       up BELOW their floor -- 987 MiB against 1536, the
                       pre-arm ladder finding 46 MiB of a 650 MiB gap

    Both are the same error in opposite directions: a quantity that must be
    SOLVED FOR was being adjusted TOWARDS.

    THE FIXTURES ARE THE TWO BRACKET BOOTS, and the rank-to-card pairing below
    is evidenced rather than assumed. rank 0 carries the 31800 MiB budget, which
    only the 5090 can hold. Of the two 3080s, only one pairing is consistent
    with what was measured -- 482490 flipped both ways under load, so at that
    pool every rank MUST have been at or above its floor, and the alternative
    pairing puts a card 300 MiB under. The verdicts below are what that
    consistency argument is built on.
    """

    #: The two bracket boots, in tokens.
    T_LOW = 482490  # flipped both ways under load; every rank above its floor
    T_HIGH = 537076  # cleared the >=495000 bar; two ranks BELOW their floor
    D_T = T_HIGH - T_LOW

    #: Resting free at each pool, MiB, per rank, and the arming-floor target
    #: (gate floor + load margin) each rank was sized against on the second
    #: boot. Taken from boot_678_validate.log / boot_678_final.log.
    RANKS = {
        0: {"free_low": 3056, "free_high": 2286, "target": 1728},
        1: {"free_low": 2167, "free_high": 987, "target": 1825},
        2: {"free_low": 3233, "free_high": 1475, "target": 2467},
    }

    def _reserve_for(self, rank):
        """A record anchored at the LOW boot, with that rank's measured slope."""
        r = self.RANKS[rank]
        cell = ((r["free_low"] - r["free_high"]) * MIB) // self.D_T
        return (
            sr.SeamReserve(
                fixed_bytes=139 * MIB,
                arena_fixed_bytes=1456 * MIB,
                worst_leg_fixed_bytes=1456 * MIB,
                per_row_bytes=1.0,
                have_bytes=8 * GIB,  # deliberately non-binding here
                id_space=self.T_LOW,
                free_at_measure_bytes=r["free_low"] * MIB,
                provenance=sr.PROVENANCE_STORED,
            ),
            cell,
            r["target"] * MIB,
        )

    def test_the_LOW_bracket_is_accepted_on_every_rank(self):
        """482490 flipped both ways on metal, so the solve must permit it."""
        for rank in self.RANKS:
            with self.subTest(rank=rank):
                reserve, cell, target = self._reserve_for(rank)
                allowed = sr.floor_allowed_tokens(cell, reserve, target)
                self.assertIsNotNone(allowed)
                self.assertGreaterEqual(
                    allowed,
                    self.T_LOW,
                    "the pool that demonstrably flips must not be refused",
                )

    def test_the_HIGH_bracket_is_REJECTED_by_the_floor_guarantee(self):
        """537076 cleared the token bar and could not hold the floors. The
        whole point of the guarantee is that it says so."""
        offenders = []
        for rank in self.RANKS:
            reserve, cell, target = self._reserve_for(rank)
            if sr.floor_allowed_tokens(cell, reserve, target) < self.T_HIGH:
                offenders.append(rank)
        self.assertEqual(
            sorted(offenders),
            [1, 2],
            "exactly the two ranks measured below their floor must reject it",
        )

    def test_the_floor_solve_alone_clears_the_bar_with_the_seam_NEUTRALISED(self):
        """RENAMED, BECAUSE THE OLD NAME CLAIMED A PROPERTY IT DID NOT TEST.

        This was ``test_the_solved_pool_clears_the_operators_bar``, asserting
        that "the solved pool" reaches 495000 -- within 10% of the 550000
        hand-pin. It is computed from ``_reserve_for``, which sets
        ``per_row_bytes=1.0`` and ``have_bytes=8 GiB`` ("deliberately
        non-binding"). The rig's own records carry 424.1 / 550.7 / 2360.3
        B/row. So the number this asserted was the FLOOR solve in isolation
        with the seam term switched off, and it could not fail against the
        thing its name promised -- #380 class.

        On metal the same regime solves 435319, and the seam binds rank 0 at
        530237. A green bar at 495000 coexisted with that for a whole task.

        The isolation is still worth testing -- it is what proves the floor
        arithmetic itself is not the conservative term -- so the test stays
        with an honest name. The SHIP bar moved to
        ``TheShipPinIsDerivedFromThisRegime``, which uses the records the boot
        actually writes.
        """
        per_rank = {}
        for rank in self.RANKS:
            reserve, cell, target = self._reserve_for(rank)
            per_rank[rank] = sr.floor_allowed_tokens(cell, reserve, target)
        pool = min(per_rank.values())
        self.assertGreaterEqual(
            pool,
            495000,
            f"solved per-rank ceilings {per_rank} -> pool {pool}, which must "
            f"clear the 495000 bar the subtrahend could not reach",
        )
        self.assertLess(
            pool,
            self.T_HIGH,
            "and it must stay under the bracket that could not hold the floor",
        )

    def test_the_binding_rank_is_the_one_with_the_least_headroom(self):
        per_rank = {}
        for rank in self.RANKS:
            reserve, cell, target = self._reserve_for(rank)
            per_rank[rank] = sr.floor_allowed_tokens(cell, reserve, target)
        self.assertEqual(min(per_rank, key=per_rank.get), 1)

    def test_the_solve_is_an_EQUALITY_within_one_token(self):
        """At the solved id space the card rests ON the target, not above it.
        That is the difference between solving and approximating."""
        for rank in self.RANKS:
            with self.subTest(rank=rank):
                reserve, cell, target = self._reserve_for(rank)
                allowed = sr.floor_allowed_tokens(cell, reserve, target)
                free_at = (
                    reserve.free_at_measure_bytes + (reserve.id_space - allowed) * cell
                )
                self.assertGreaterEqual(free_at, target)
                self.assertLess(free_at - target, cell, "not the LARGEST such T")

    def test_a_bigger_floor_target_permits_fewer_tokens(self):
        reserve, cell, target = self._reserve_for(1)
        self.assertLess(
            sr.floor_allowed_tokens(cell, reserve, target + 512 * MIB),
            sr.floor_allowed_tokens(cell, reserve, target),
        )


class TheSolveDegradesToTheOldPathTest(unittest.TestCase):
    """A record written before the free column was persisted must be sized
    exactly as before -- absent means unknown, and unknown may not be treated
    as an opportunity."""

    def test_no_free_column_means_no_direct_solve(self):
        old = sr.SeamReserve(
            fixed_bytes=455 * MIB,
            have_bytes=2000 * MIB,
            id_space=400000,
            provenance=sr.PROVENANCE_STORED,
        )
        self.assertEqual(old.free_at_measure_bytes, 0)
        self.assertIsNone(sr.floor_allowed_tokens(4096, old, 2 * GIB))

    def test_a_cold_record_has_no_direct_solve_either(self):
        cold = sr.SeamReserve(provenance=sr.PROVENANCE_COLD)
        self.assertIsNone(sr.floor_allowed_tokens(4096, cold, 2 * GIB))

    def test_no_cell_means_no_direct_solve(self):
        r = sr.SeamReserve(
            fixed_bytes=455 * MIB,
            have_bytes=2000 * MIB,
            id_space=400000,
            free_at_measure_bytes=3 * GIB,
            provenance=sr.PROVENANCE_STORED,
        )
        self.assertIsNone(sr.floor_allowed_tokens(0, r, 2 * GIB))

    def test_the_old_path_still_charges_when_it_must(self):
        """The fallback keeps the subtrahend, so a pre-#678 record is neither
        double-charged nor left unprotected."""
        old = sr.SeamReserve(
            fixed_bytes=455 * MIB,
            have_bytes=2000 * MIB,
            id_space=400000,
            provenance=sr.PROVENANCE_STORED,
        )
        floor = sr.arming_floor_target_bytes()
        with_floor, _ = sr.seam_adjusted_budget_bytes(
            20 * GIB, 4096, old, arming_floor_bytes=floor
        )
        without, _ = sr.seam_adjusted_budget_bytes(20 * GIB, 4096, old)
        self.assertLess(with_floor, without)

    def test_both_constraints_bind_and_the_min_wins(self):
        """The seam must be fundable AND the card must rest above its floor.
        They bind on different ranks at different vectors, so neither is
        subtracted from the other."""
        r = sr.SeamReserve(
            fixed_bytes=139 * MIB,
            arena_fixed_bytes=1456 * MIB,
            worst_leg_fixed_bytes=1456 * MIB,
            per_row_bytes=1.0,
            have_bytes=200 * MIB,  # a TIGHT seam solve
            id_space=482490,
            free_at_measure_bytes=8 * GIB,  # a generous floor solve
            provenance=sr.PROVENANCE_STORED,
        )
        cell = 20000
        seam_only = sr.seam_allowed_tokens(cell, r)
        floor_only = sr.floor_allowed_tokens(cell, r, 1825 * MIB)
        _, allowed = sr.seam_adjusted_budget_bytes(
            40 * GIB, cell, r, arming_floor_bytes=1825 * MIB
        )
        self.assertEqual(allowed, min(seam_only, floor_only))
        self.assertEqual(allowed, seam_only, "the seam binds in this fixture")

    def test_the_FLOOR_binding_is_what_reaches_the_budget(self):
        """THE CASE THE CHANGE EXISTS FOR, and the one a mutation caught
        missing: with the seam solve generous and the floor tight, the pool
        must be the FLOOR's ceiling. A min that quietly drops the floor term
        passes every other test in this class, because they all happen to have
        the seam binding -- which is exactly how a guarantee ends up wired to
        nothing.
        """
        r = sr.SeamReserve(
            fixed_bytes=139 * MIB,
            arena_fixed_bytes=0,
            worst_leg_fixed_bytes=139 * MIB,
            per_row_bytes=1.0,
            have_bytes=40 * GIB,  # a very generous seam solve
            id_space=482490,
            free_at_measure_bytes=2167 * MIB,  # rank 1's measured column
            provenance=sr.PROVENANCE_STORED,
        )
        cell = 22667  # rank 1's measured slope
        seam_only = sr.seam_allowed_tokens(cell, r)
        floor_only = sr.floor_allowed_tokens(cell, r, 1825 * MIB)
        self.assertLess(floor_only, seam_only, "the FLOOR must bind here")
        _, allowed = sr.seam_adjusted_budget_bytes(
            400 * GIB, cell, r, arming_floor_bytes=1825 * MIB
        )
        self.assertEqual(
            allowed,
            floor_only,
            "the floor's ceiling must reach the budget, or the guarantee is "
            "computed and discarded",
        )

    def test_the_budget_reflects_the_floor_ceiling_in_BYTES(self):
        """And it must come out of the returned budget, not just the count."""
        r = sr.SeamReserve(
            fixed_bytes=139 * MIB,
            worst_leg_fixed_bytes=139 * MIB,
            per_row_bytes=1.0,
            have_bytes=40 * GIB,
            id_space=482490,
            free_at_measure_bytes=2167 * MIB,
            provenance=sr.PROVENANCE_STORED,
        )
        cell = 22667
        new_bytes, allowed = sr.seam_adjusted_budget_bytes(
            400 * GIB, cell, r, arming_floor_bytes=1825 * MIB
        )
        self.assertEqual(new_bytes, allowed * cell)


class TheShipPinIsDerivedFromThisRegime(unittest.TestCase):
    """#678: the ship bar, computed from the records the boot actually writes.

    WHAT THIS REPLACES. The old ship bar was 495000, "within 10% of the 550000
    the hand-pin encodes", asserted with the seam slope set to 1.0 B/row. Two
    things were wrong with it and they compound: the bar descended from a
    pin taken under a DIFFERENT REGIME (before 48ba9fe72a folded the arming
    floor into sizing), and the assertion neutralised the very term that binds
    on metal. A test that cannot fail against the property it names is worse
    than no test, because the next reader takes it as coverage.

    THE BRACKET THE RIG MEASURED, and it is why the withdrawn pin was not
    merely stale: 482490 tokens flipped both ways under load with every rank
    above its floor; 537076 could not hold the floors on two of three ranks.
    537076 is 13k BELOW the 550000 that was being chased.

    FIXTURES COME FROM THE REAL FORMAT. Each rank's record is written by
    ``write_seam_reserve`` and read back by ``read_seam_reserve`` -- the same
    pair the boot uses -- so a change to the record schema breaks this file
    instead of silently leaving it testing a shape nothing writes.
    """

    MIB = 1 << 20

    #: This rig's records of 2026-08-16T03:56Z, the basis SHIP_PIN stands on.
    RECORDS = {
        0: dict(
            fixed_bytes=238763008,
            per_row_bytes=2360.3031340235552,
            have_bytes=3880139776,
            id_space=435319,
            arena_fixed_bytes=0,
            worst_leg_fixed_bytes=238763008,
            free_at_measure_bytes=4500160512,
            rung_fund_bytes=238763008,
            rung_guaranteed_bytes=238763008,
            floor_mib=1728,
        ),
        1: dict(
            fixed_bytes=145652736,
            per_row_bytes=424.1172657292698,
            have_bytes=2337502976,
            id_space=435319,
            arena_fixed_bytes=854522624,
            worst_leg_fixed_bytes=854522624,
            free_at_measure_bytes=2196111360,
            rung_fund_bytes=1000175360,
            rung_guaranteed_bytes=1000175360,
            floor_mib=1825,
        ),
        2: dict(
            fixed_bytes=145652736,
            per_row_bytes=550.6682501797533,
            have_bytes=3408306688,
            id_space=435319,
            arena_fixed_bytes=1526867456,
            worst_leg_fixed_bytes=1526867456,
            free_at_measure_bytes=2594570240,
            rung_fund_bytes=1672520192,
            rung_guaranteed_bytes=1672520192,
            floor_mib=2467,
        ),
    }

    #: #685: the bar is now the DEMONSTRATED-SAFE pool. Before the arena-tail
    #: relief the recorded inputs supported ~435696 and this floor sat at
    #: 430000; the relief lands them at ~483723, so the guard moves up to the
    #: bracket it was always meant to protect. A regression below the pool this
    #: rig measured safe is now red.
    REGRESSION_FLOOR = 480000
    CELL_BYTES = 20480

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self._real_record_path = sr.record_path

    def tearDown(self):
        sr.record_path = self._real_record_path
        self._dir.cleanup()

    def _round_trip(self, rank):
        """Write this rank's record the way the boot does, then read it back.

        THE FLOOR IS DERIVED, NOT PINNED (#685). ``floor_mib`` is what the gate
        solved on the boot; everything in it EXCEPT the one-leg seam draw is
        held fixed (that residue carries rank 0's corridor-shortfall term, so
        pinning the total would freeze it too). The draw itself is recomputed
        from the record, which is what makes this test sensitive to a change in
        what the floor charges instead of asserting yesterday's total.
        """
        r = dict(self.RECORDS[rank])
        observed_floor = r.pop("floor_mib") * self.MIB
        residue = observed_floor - int(r["worst_leg_fixed_bytes"])
        path = os.path.join(self._dir.name, f"seam-rank{rank}.json")
        sr.record_path = lambda server_args, world_rank, _p=path: _p
        sr.write_seam_reserve(None, rank, detail="regime basis", **r)
        reserve = sr.read_seam_reserve(None, rank)
        return reserve, residue + reserve.arming_draw_bytes()

    def _solved_pool(self):
        per_rank = {}
        for rank in self.RECORDS:
            reserve, floor = self._round_trip(rank)
            self.assertEqual(sr.PROVENANCE_STORED, reserve.provenance)
            floor_allowed = sr.floor_allowed_tokens(self.CELL_BYTES, reserve, floor)
            seam_allowed = sr.seam_allowed_tokens(self.CELL_BYTES, reserve)
            per_rank[rank] = min(
                x for x in (floor_allowed, seam_allowed) if x is not None
            )
        return min(per_rank.values()), per_rank

    def test_nothing_ships_at_or_above_a_pool_measured_UNSAFE(self):
        """The property the old bar inverted: it pushed UP toward 550000.

        537076 was measured unable to hold its floors on two of three ranks.
        A sizer that reaches it has regressed however good the token count
        looks.
        """
        pool, per_rank = self._solved_pool()
        self.assertLess(
            pool,
            sr.SHIP_PIN.demonstrated_unsafe_tokens,
            f"solved {pool} at or above the measured-unsafe "
            f"{sr.SHIP_PIN.demonstrated_unsafe_tokens}: {per_rank}",
        )

    def test_the_sizer_does_not_regress_below_what_these_records_support(self):
        pool, per_rank = self._solved_pool()
        self.assertGreaterEqual(
            pool, self.REGRESSION_FLOOR, f"sizer regressed: {per_rank}"
        )

    def test_the_withdrawn_pin_is_recorded_as_withdrawn(self):
        """So a re-introduction is recognisable on sight rather than arriving
        as a fresh 'empirical bound'."""
        self.assertEqual(550000, sr.SHIP_PIN.withdrawn_pin)
        self.assertIn("cross-regime", sr.SHIP_PIN.withdrawn_reason)
        self.assertLess(
            sr.SHIP_PIN.demonstrated_unsafe_tokens,
            sr.SHIP_PIN.withdrawn_pin,
            "the withdrawn pin must be recorded as ABOVE a measured-unsafe pool",
        )

    def test_the_pin_basis_matches_the_records_it_was_derived_from(self):
        """THE INVALIDATION MECHANISM. 550000 outlived its regime silently
        because nothing compared it to its inputs. When a change moves a seam
        slope or an arming floor, this goes red and the pin is re-derived
        instead of quietly surviving."""
        slopes = tuple(self.RECORDS[r]["per_row_bytes"] for r in sorted(self.RECORDS))
        floors = tuple(self.RECORDS[r]["floor_mib"] for r in sorted(self.RECORDS))
        self.assertEqual(sr.SHIP_PIN.basis_per_row_bytes, slopes)
        self.assertEqual(sr.SHIP_PIN.basis_arming_floor_mib, floors)

    def test_the_sizer_reaches_the_pool_this_rig_demonstrated_safe(self):
        """THE ACCEPTANCE, INVERTED BY #685 EXACTLY AS IT WAS DESIGNED TO BE.

        It was written as the open-gap marker -- "the sizer lands ~47k below
        the pool this rig demonstrated safe, and that is follow-up work, not
        something this bar should paper over by lowering the safe number to
        meet it" -- and it asserted ``pool < demonstrated_safe``.

        The follow-up landed. The ~47k was the arena tail charged to the
        arming floor on the binding ranks while the record already showed the
        KV rung funding it, so the floor stopped charging it where the rung
        covers it and the solve went 435696 -> ~483723.

        The number it must clear is the artifact's, not a literal: when the
        pin is re-derived under a new regime this bar moves with it.
        """
        pool, per_rank = self._solved_pool()
        self.assertGreaterEqual(
            pool,
            sr.SHIP_PIN.demonstrated_safe_tokens,
            f"solved {pool}, below the demonstrated-safe "
            f"{sr.SHIP_PIN.demonstrated_safe_tokens}: {per_rank}",
        )

    def test_the_relief_did_not_buy_the_pool_past_the_unsafe_bracket(self):
        """THE PROPERTY, NOT A CONSTANT. Whatever the solve lands on, it is
        checked against the measured-unsafe number in the artifact -- so a
        future relief that overshoots is caught by the same test that
        certified this one, without anyone remembering to update a literal."""
        pool, per_rank = self._solved_pool()
        self.assertLess(
            pool,
            sr.SHIP_PIN.demonstrated_unsafe_tokens,
            f"relief overshot into the measured-unsafe band: {pool} {per_rank}",
        )
        self.assertGreater(
            sr.SHIP_PIN.demonstrated_unsafe_tokens - pool,
            0,
            "the margin to the unsafe bracket must be positive by construction",
        )


class TheArenaTailIsNotChargedTwice(unittest.TestCase):
    """#685: the floor stops reserving what the rung demonstrably funds.

    THE TAIL IS REAL, and this does not dispute it. `arena_fixed_bytes` is
    `max(0, pp_bytes - tp_bytes)`: the weights the PP layout needs beyond the
    TP layout's on this rank. It reconciles three ways on the 2026-08-16 boot
    -- the `rung 3 released ...` lines, the record, and the definition:

        PP0  pp 15790.5  tp 16329.9  -> max(0, -539) =    0 MiB
        PP1  pp  9792.8  tp  8977.8  ->              =  815 MiB
        PP2  pp 10434.0  tp  8977.8  ->              = 1456 MiB

    It is largest on the SMALLEST card because uneven TP shrinks that card's
    TP shard while PP still hands it a full stage. Geometry, not a defect.

    WHAT WAS WRONG WAS THE TREATMENT. The tail was charged to the arming
    floor -- a permanent free-VRAM reservation -- while the record already
    showed the KV rung could pay it at flip time (PP1 954 > 815, PP2 1595 >
    1456). `arming_floor_subtrahend_bytes` names this exact hazard: "Reserving
    it here as well would be the third payment for one requirement."

    IT IS CONDITIONAL ON THE RECORDED FUNDING, never unconditional removal. A
    rank whose rung cannot cover its own tail keeps paying for it in the
    floor, because for that rank the ladder genuinely cannot find it.

    GATED ON METAL, NOT ON ARITHMETIC. The rung that must pay was latched off
    until 38c1161fd4; on the boot carrying it the log shows the arena tail
    being released 22 times at flip cadence and no recovery failures. The
    relief was not built before that evidence existed.
    """

    MIB = 1 << 20

    def _reserve(self, arena_mib, draft_mib, rung_mib):
        return sr.SeamReserve(
            fixed_bytes=draft_mib * self.MIB,
            arena_fixed_bytes=arena_mib * self.MIB,
            # This rig's shape: the two maxima come from DIFFERENT legs, so
            # the worst single leg is the larger of them, not their sum.
            worst_leg_fixed_bytes=max(arena_mib, draft_mib) * self.MIB,
            rung_fund_bytes=rung_mib * self.MIB,
            # #696: the relief is conditional on what the rung is GUARANTEED to
            # deliver, not on what it was once seen holding. These fixtures
            # mean "the rung can pay", so they now say so in the field that
            # carries that meaning. Production records have no guarantee
            # recorded yet and therefore get no excuse -- which is the point:
            # on metal the rung delivered 12.3 MiB of an 815 MiB tail at the
            # fill where the seam actually ran.
            rung_guaranteed_bytes=rung_mib * self.MIB,
            per_row_bytes=550.0,
            have_bytes=3 * (1 << 30),
            id_space=435319,
            provenance=sr.PROVENANCE_STORED,
        )

    def test_a_covered_tail_drops_out_of_the_one_leg_draw(self):
        """PP2's numbers: the rung covers 1456 with 1595."""
        draw = self._reserve(1456, 139, 1595).arming_draw_bytes()
        self.assertEqual(139 * self.MIB, draw)

    def test_an_uncovered_tail_is_still_charged_in_full(self):
        """The relief is conditional. A rung that cannot pay changes nothing."""
        draw = self._reserve(1456, 139, 900).arming_draw_bytes()
        self.assertEqual(1456 * self.MIB, draw)

    def test_the_draw_never_falls_below_the_leg_the_rung_does_not_fund(self):
        """The drafter's restore is not arena tail and is not funded by it.

        Relieving the arena must not be allowed to relieve the OTHER leg by
        arithmetic accident.
        """
        draw = self._reserve(1456, 700, 5000).arming_draw_bytes()
        self.assertEqual(700 * self.MIB, draw)

    def test_a_rank_with_no_tail_is_untouched(self):
        """PP0 has none; its draw is the drafter's restore, before and after."""
        draw = self._reserve(0, 228, 1595).arming_draw_bytes()
        self.assertEqual(228 * self.MIB, draw)

    def test_a_record_that_never_measured_a_rung_keeps_the_old_charge(self):
        """rung_fund_bytes is absent on a pre-#678 record, and 0 must mean
        'no funding known', never 'funded'."""
        draw = self._reserve(1456, 139, 0).arming_draw_bytes()
        self.assertEqual(1456 * self.MIB, draw)
