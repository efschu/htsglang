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

    def test_a_short_floor_with_the_host_tier_FULL_refuses_with_the_numbers(self):
        """The operator's second required shape, and the message is the
        deliverable: how much was short, and what the ladder actually freed."""
        r = _runtime()
        g = _Guard(free=900 * MIB, deliverable=0)
        with _PatchedGuard(g):
            ok, msg = r._prearm_floor_relief(TP_TO_PP)
        self.assertFalse(ok)
        self.assertIn("900", msg)
        self.assertIn("1523", msg)
        self.assertIn("short by", msg)
        self.assertIn("freed", msg)

    def test_the_relief_is_bounded_and_stops_asking(self):
        """An unbounded relief loop spills the instance flat."""
        r = _runtime()
        g = _Guard(free=900 * MIB, deliverable=0)
        with (
            _PatchedGuard(g),
            _Env(SGLANG_PHASE_FLIP_PREARM_RELIEF_ATTEMPTS="2"),
        ):
            for _ in range(6):
                ok, msg = r._prearm_floor_relief(TP_TO_PP)
                self.assertFalse(ok)
        self.assertEqual(len(g.asks), 2, "the bound must stop the ladder")
        self.assertIn("bounded relief attempts", msg)

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
