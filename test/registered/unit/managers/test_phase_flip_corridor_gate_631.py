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
"""#656 item 15a WIRED: the corridor gate stands in front of the seam.

The gate itself is tested hermetically in ``test_corridor_guard_631`` and
``test_corridor_even_fill_631``. This file tests the WIRING, which is where a
guard usually fails: a gate that is built, registered, and never consulted
looks exactly like a gate that works.

WHAT THESE PIN

* **A refusal reaches the abandon, and does not raise.** The seam's commits
  run inside a no-return region with no try/except on the path -- by design,
  documented at ``_abandon_parked_flip`` -- so a gate that signalled by
  raising would convert a survivable refusal into a dead rank. The verdict
  must travel as a string into ``too_small``, which already rides the
  ``_collective_min`` that makes the decision unanimous.

* **The gate runs BEFORE the affordability check.** Its providers hand pages
  back to the DRIVER, so anything it reclaims is money the affordability
  check can then see. In the other order the cheaper check would refuse
  flips the gate could have funded. Source order is the only place this is
  expressible, so source order is what is asserted.

* **A broken gate degrades, it does not take the flip down.** The gate is a
  safety net. A net that tears must not kill the thing it was catching.

* **A funded seam is counted.** A run whose ``corridor_reclaims`` is zero has
  not exercised item 15a at all, whatever its logs say -- this chain has
  shipped "working" machinery that turned out to be inert seven times, so
  the counter is the evidence, not the absence of errors.

Hermetic: a stub runtime, an injected guard, no CUDA.
"""

from __future__ import annotations

import inspect
import unittest

from sglang.srt.managers import phase_flip_spill
from sglang.srt.managers.corridor_guard import GuardResult
from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

MIB = 1024 * 1024


class _Guard:
    def __init__(self, result):
        self.result = result
        self.asks = []

    def ensure_headroom(self, want, *, reason="", refusal_is_fatal=False):
        self.asks.append((int(want), reason, refusal_is_fatal))
        return self.result


def _cleared(reclaimed_mib=0):
    return GuardResult(
        True, 0, 0, 0, reclaimed_mib * MIB, ("draft-weights",), "cleared"
    )


def _refused():
    return GuardResult(False, 0, 0, 0, 0, (), "every provider is exhausted")


def _runtime():
    r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    r._census_scheduler = object()
    r.corridor_aborts = 0
    r.corridor_reclaims = 0
    r._corridor_pp_refusals = 0
    return r


class _Patched:
    """Swap ``get_corridor_guard`` for the duration of one test."""

    def __init__(self, guard_or_raiser):
        self.g = guard_or_raiser

    def __enter__(self):
        self.old = phase_flip_spill.get_corridor_guard
        if isinstance(self.g, Exception):
            exc = self.g

            def boom(_scheduler):
                raise exc

            phase_flip_spill.get_corridor_guard = boom
        else:
            phase_flip_spill.get_corridor_guard = lambda _scheduler: self.g
        return self.g

    def __exit__(self, *a):
        phase_flip_spill.get_corridor_guard = self.old
        return False


class TheVerdictTravelsAsAStringTest(unittest.TestCase):
    def test_a_cleared_gate_says_nothing_and_lets_the_flip_run(self):
        r = _runtime()
        with _Patched(_Guard(_cleared())):
            self.assertEqual(r._corridor_gate(500 * MIB, "pp->tp"), "")
        self.assertEqual(r.corridor_aborts, 0)
        self.assertEqual(r.corridor_reclaims, 0)

    def test_a_funded_seam_is_counted_as_a_reclaim_not_an_abort(self):
        r = _runtime()
        with _Patched(_Guard(_cleared(reclaimed_mib=286))):
            self.assertEqual(r._corridor_gate(500 * MIB, "tp->pp"), "")
        self.assertEqual(r.corridor_reclaims, 1)
        self.assertEqual(r.corridor_aborts, 0)

    def test_a_refusal_returns_a_detail_and_never_raises(self):
        r = _runtime()
        with _Patched(_Guard(_refused())):
            detail = r._corridor_gate(500 * MIB, "pp->tp")
        self.assertIn("corridor gate refused", detail)
        self.assertIn("every provider is exhausted", detail)
        self.assertEqual(r.corridor_aborts, 1)

    def test_the_gate_is_asked_about_the_staging_bytes_it_was_given(self):
        r = _runtime()
        with _Patched(_Guard(_cleared())) as g:
            r._corridor_gate(1234 * MIB, "pp->tp")
        self.assertEqual(g.asks[0][0], 1234 * MIB)
        self.assertIn("pp->tp", g.asks[0][1])

    def test_no_scheduler_is_not_an_error(self):
        # Unit-test and pre-boot shapes have no scheduler; the gate is simply
        # not available yet and the flip falls back to the old behaviour.
        r = _runtime()
        r._census_scheduler = None
        self.assertEqual(r._corridor_gate(500 * MIB, "pp->tp"), "")


class ABrokenGateDegradesTest(unittest.TestCase):
    def test_a_raising_guard_does_not_take_the_flip_down(self):
        r = _runtime()
        with _Patched(RuntimeError("nvml exploded")):
            self.assertEqual(r._corridor_gate(500 * MIB, "pp->tp"), "")
        self.assertEqual(r.corridor_aborts, 0)


class TheGateIsActuallyConsultedTest(unittest.TestCase):
    """The failure mode this whole file exists for: a gate nobody calls."""

    def test_execute_calls_the_gate(self):
        src = inspect.getsource(PhaseFlipRuntime._execute)
        self.assertIn("_corridor_gate", src)

    def test_the_refusal_is_folded_into_too_small(self):
        src = inspect.getsource(PhaseFlipRuntime._execute)
        self.assertIn("too_small.append(corridor_detail)", src)

    def test_the_gate_runs_before_the_affordability_check(self):
        # Ordering is load-bearing: the gate's providers free to the DRIVER,
        # so its reclaim is money _staging_affordable can see. Reversed, the
        # cheaper check refuses flips the gate could have funded.
        src = inspect.getsource(PhaseFlipRuntime._execute)
        self.assertLess(
            src.index("_corridor_gate"),
            src.index("_staging_affordable"),
        )

    def test_the_gate_does_not_raise_into_the_no_return_region(self):
        # _corridor_gate must swallow provider failures. If someone later
        # "simplifies" the try/except away, this catches it.
        src = inspect.getsource(PhaseFlipRuntime._corridor_gate)
        self.assertIn("except Exception", src)


class TheProviderBindsLateTest(unittest.TestCase):
    """The trap: the ladder binds its carrier AFTER the first cutover leg,
    but the guard is built at the first gate, which is BEFORE it. A provider
    that captured the carrier at registration would cache None into a list
    that is never rebuilt -- an inert guard, indistinguishable in the logs
    from one that never needed to arm."""

    def test_a_scheduler_with_no_carrier_yet_yields_nothing_and_does_not_raise(self):
        class _S:
            pass

        provider = phase_flip_spill._late_bound_draft_provider(_S())
        self.assertEqual(provider(1 << 30), 0)

    def test_a_carrier_that_appears_later_is_found_on_the_next_call(self):
        class _Carrier:
            spilled = False

            def spill(self):
                _Carrier.spilled = True
                return 286.0

        class _Ladder:
            _weights = None

        class _S:
            phase_flip_spill_ladder = _Ladder()

        s = _S()
        provider = phase_flip_spill._late_bound_draft_provider(s)
        self.assertEqual(provider(1 << 30), 0)
        # ...the cutover leg runs and the ladder binds...
        s.phase_flip_spill_ladder._weights = _Carrier()
        self.assertEqual(provider(1 << 30), int(286.0 * MIB))

    def test_an_already_spilled_carrier_yields_nothing_rather_than_double_counting(self):
        class _Carrier:
            spilled = True

            def spill(self):  # pragma: no cover - must not be reached
                raise AssertionError("spilled twice")

        class _Ladder:
            _weights = _Carrier()

        class _S:
            phase_flip_spill_ladder = _Ladder()

        self.assertEqual(
            phase_flip_spill._late_bound_draft_provider(_S())(1 << 30), 0
        )


class TheLawAndTheArmingFloorAreDifferentNumbersTest(unittest.TestCase):
    """The 2026-08-10 wedge. A guard whose refusal is judged by its ARMING
    watermark refuses allocations the corridor law permits. On the pp->tp leg
    that is not conservative -- strict purity forbids decode in PP, so a
    permanently refused pp->tp starves decode and nothing in PP can free the
    memory that would end it. Measured: 411 abandons, 0 requests in 6 min."""

    def test_a_raised_arming_floor_does_not_manufacture_refusals(self):
        from sglang.srt.managers import corridor_guard as cg

        free = [2306 * MIB]
        g = cg.CorridorGuard(
            0,
            floor_mib=1600,
            law_floor_mib=1024,
            probe=lambda: free[0],
            fleet_probe=lambda: list(free),
        )
        # free - want = 1580: below the 1600 arming floor, so the gate ARMS
        # and tries. Well above the 1024 law, so it must NOT refuse.
        r = g.ensure_headroom(726 * MIB)
        self.assertEqual(g.arm_count, 1)
        self.assertTrue(r.ok, r.detail)
        self.assertEqual(g.refuse_count, 0)

    def test_the_law_still_refuses_what_the_law_forbids(self):
        from sglang.srt.managers import corridor_guard as cg

        free = [1100 * MIB]
        g = cg.CorridorGuard(
            0,
            floor_mib=1600,
            law_floor_mib=1024,
            probe=lambda: free[0],
            fleet_probe=lambda: list(free),
        )
        self.assertFalse(g.ensure_headroom(900 * MIB).ok)

    def test_by_default_the_two_floors_coincide(self):
        from sglang.srt.managers import corridor_guard as cg

        g = cg.CorridorGuard(0, probe=lambda: 0)
        self.assertEqual(g.law_floor_mib, cg.DEFAULT_FLOOR_MIB)
        self.assertEqual(g.law_floor_bytes, g.floor_bytes)


class RepeatedPpToTpRefusalIsNamedTest(unittest.TestCase):
    def test_consecutive_pp_to_tp_refusals_are_counted(self):
        r = _runtime()
        with _Patched(_Guard(_refused())):
            for _ in range(3):
                r._corridor_gate(500 * MIB, "pp_to_tp")
        self.assertEqual(r._corridor_pp_refusals, 3)

    def test_the_other_direction_resets_the_streak(self):
        # tp->pp refusal is survivable: the instance stays in TP and decode
        # keeps running. Only the pp->tp streak is a deadlock signal.
        r = _runtime()
        with _Patched(_Guard(_refused())):
            r._corridor_gate(500 * MIB, "pp_to_tp")
            r._corridor_gate(500 * MIB, "tp_to_pp")
        self.assertEqual(r._corridor_pp_refusals, 0)


class TheFatalLegIsMarkedTest(unittest.TestCase):
    def test_pp_to_tp_is_declared_fatal_and_tp_to_pp_is_not(self):
        r = _runtime()
        seen = {}

        class _G:
            def ensure_headroom(self, want, *, reason="", refusal_is_fatal=False):
                seen[reason.split()[-1]] = refusal_is_fatal
                return _cleared()

        with _Patched(_G()):
            r._corridor_gate(100 * MIB, "pp_to_tp")
            r._corridor_gate(100 * MIB, "tp_to_pp")
        self.assertTrue(seen["pp_to_tp"])
        self.assertFalse(seen["tp_to_pp"])


if __name__ == "__main__":
    unittest.main()
