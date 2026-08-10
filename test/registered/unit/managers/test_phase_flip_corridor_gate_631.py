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

    def ensure_headroom(self, want, *, reason=""):
        self.asks.append((int(want), reason))
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


if __name__ == "__main__":
    unittest.main()
