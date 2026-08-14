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
"""#656 axis 3: THE YIELD MAY NOT ENTER A TROUGH THIS RANK HAS ALREADY MEASURED.

The remediation boot's one remaining corridor breach, and it is NOT what the
first reading of it said. It was reported as a bs=1 deep-prefill transient --
"138 MiB at a flip seam became 12 MiB at a deep prefill". The 100 ms corridor
trace says otherwise. Both sub-law samples fall at 14:42:25.6-.7, and the
seam census for that exact instant reads::

    14:42:22 PP1  seam entry margin YIELDED (tp_to_pp) after 2 consecutive
                  abandoned attempts: entering on the corridor law alone
    14:42:25 PP1  CORRIDOR LAW BROKEN during tp_to_pp rank 1 at stage
                  'weights_refill': free 1012 MiB is below the 1024 MiB floor
    stage walk:   ... backing_restore free=1250 | kv_write 1250 | gdn_state 1250
                  | weights_refill free=1012 step-238 | cutover free=1290

So the residual breach is the SAME mechanism as the acceptance's five: a seam
entered on the corridor LAW ALONE after a C20 entry-margin yield, and the
in-cutover draw took it under. Not a new prefill-side item at all -- the
deep-prefill phase is merely when it happened.

WHAT CLOSES IT, AND WHY IT COULD NOT BE CLOSED BEFORE. The gate already
measures this rank's worst in-cutover draw (``_seam_draw_max``, fed from the
seam census) and already predicts the trough from it. It then deliberately
does NOT act on the prediction, and its own comment says why::

    A False ``law_ok`` routes into the abort path below, and for pp->tp that
    path is the DECODE WEDGE: under strict purity decode runs only in TP, so
    a persistently refused pp->tp starves decode outright. [...] Trading a
    1.5 s corridor dip for a total outage is not a fix.

That premise is now false. The purity stand-down valve
(``phase_purity._relaxed``, #656 C22) lets decode run in the PP layout as
soon as ``pp_to_tp`` has been abandoned a few rounds running, so a refused
flip degrades throughput instead of stopping the instance. The trade is no
longer "a corridor dip versus an outage"; it is "a corridor dip versus a
slower decode step", and the corridor law is a hard user limit while the
decode layout is a performance choice.

So the YIELD -- and only the yield, which is the one path that deliberately
enters at the law -- is withheld when this rank's own measured draw predicts
a sub-law trough. It keeps objecting with the margin-delay tag, which is
exempt from the seam-abandon cap by design, so the flip is delayed rather
than stood down for good, and the valve keeps tokens flowing meanwhile.

Hermetic: the stub runtime and injected guard of
``test_seam_entry_margin_631``, no CUDA.
"""

from __future__ import annotations

import os
import unittest

from sglang.srt.managers import phase_flip_spill
from sglang.srt.managers.corridor_guard import GuardResult
from sglang.srt.managers.phase_flip_runtime import (
    SEAM_MARGIN_DELAY_TAG,
    PhaseFlipRuntime,
)

MIB = 1024 * 1024
LAW = 1024 * MIB


class _Guard:
    def __init__(self, free_after, law_floor_bytes=LAW):
        self.law_floor_bytes = law_floor_bytes
        self.free_after = free_after
        self.asks = []

    def ensure_headroom(self, want, *, reason="", refusal_is_fatal=False):
        want = int(want)
        self.asks.append(want)
        ok = (self.free_after - want) >= self.law_floor_bytes
        return GuardResult(
            ok,
            self.free_after,
            self.free_after,
            want,
            0,
            ("allocator-cache",) if ok else (),
            "cleared" if ok else "short",
        )


class _Patched:
    def __init__(self, guard):
        self.g = guard

    def __enter__(self):
        self.old = phase_flip_spill.get_corridor_guard
        self.old_kv = phase_flip_spill.collective_kv_backing_relief
        phase_flip_spill.get_corridor_guard = lambda _s: self.g
        phase_flip_spill.collective_kv_backing_relief = (
            lambda *a, **k: 0
        )  # noqa: ARG005
        return self.g

    def __exit__(self, *exc):
        phase_flip_spill.get_corridor_guard = self.old
        phase_flip_spill.collective_kv_backing_relief = self.old_kv
        return False


class _Env:
    def __init__(self, **env):
        self.env = {k: str(v) for k, v in env.items()}

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def _runtime(*, measured_draw_mib=0, direction="tp_to_pp", abandons=0):
    r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    r._census_scheduler = object()
    r.corridor_aborts = 0
    r.corridor_reclaims = 0
    r._corridor_pp_refusals = 0
    r.corridor_kv_relief_count = 0
    r.corridor_kv_relief_bytes = 0
    r._collective_min = lambda vals, **kw: list(vals)
    r._seam_abandons_in_a_row = {"pp_to_tp": 0, "tp_to_pp": 0}
    r._seam_abandons_in_a_row[direction] = abandons
    r._seam_draw_max = {"pp_to_tp": 0, "tp_to_pp": 0}
    r._seam_draw_max[direction] = measured_draw_mib * MIB
    return r


#: The metal instant, in the gate's own units. Entry 2464 MiB free, staged
#: figure 943 MiB, this rank's worst measured draw 1452 MiB -- which leaves
#: 1012 MiB, twelve below the law.
ENTRY_FREE = 2464 * MIB
STAGED = 943 * MIB
MEASURED_DRAW = 1452 * MIB


class TheYieldIsWithheldWhenTheDrawPredictsABreachTest(unittest.TestCase):
    def _gate(self, **kw):
        # Budget spent: the next decision is the YIELD.
        r = _runtime(direction="tp_to_pp", abandons=8, **kw)
        with _Env(SGLANG_SEAM_ENTRY_MARGIN_MIB=512, SGLANG_SEAM_ENTRY_DELAY_BUDGET=2):
            with _Patched(_Guard(ENTRY_FREE)):
                return r, r._corridor_gate(STAGED, "tp_to_pp")

    def test_the_metal_instant_is_no_longer_yielded_through(self):
        r, detail = self._gate(measured_draw_mib=1452)
        self.assertNotEqual(detail, "", "the yield entered a measured breach")
        self.assertEqual(r.seam_margin_yields, 0)

    def test_it_objects_as_a_DELAY_so_the_flip_is_not_stood_down(self):
        """The margin-delay tag is exempt from the seam-abandon cap.

        That exemption is what keeps this a delay rather than a permanent
        stand-down: the condition is transient (the memory comes back once
        decode drains), so ending the flip for good would be the wrong
        verdict for a wait that can win.
        """
        _r, detail = self._gate(measured_draw_mib=1452)
        self.assertIn(SEAM_MARGIN_DELAY_TAG, detail)

    def test_a_rank_with_no_measurement_still_yields(self):
        """Zero measured draw is not a licence to invent one.

        Until this rank has SEEN a cutover, the term is 0 and the behaviour is
        exactly the shipped one -- an unmeasured bucket never becomes a guess.
        """
        r, detail = self._gate(measured_draw_mib=0)
        self.assertEqual(detail, "")
        self.assertEqual(r.seam_margin_yields, 1)

    def test_a_draw_that_fits_still_yields(self):
        """The withholding is keyed on the PREDICTION, not on having measured."""
        r, detail = self._gate(measured_draw_mib=900)
        self.assertEqual(detail, "")
        self.assertEqual(r.seam_margin_yields, 1)

    def test_the_law_check_itself_is_unchanged(self):
        """A seam below the LAW is still refused, however exhausted the budget.

        The falsifier for the whole item: this must not turn into a path that
        proceeds because the measured draw happened to be small.
        """
        r = _runtime(direction="tp_to_pp", abandons=8, measured_draw_mib=0)
        with _Env(SGLANG_SEAM_ENTRY_MARGIN_MIB=512, SGLANG_SEAM_ENTRY_DELAY_BUDGET=2):
            with _Patched(_Guard(1100 * MIB)):
                detail = r._corridor_gate(600 * MIB, "tp_to_pp")
        self.assertNotEqual(detail, "")
        self.assertNotIn(SEAM_MARGIN_DELAY_TAG, detail)
        self.assertEqual(r.corridor_aborts, 1)


class TheWithholdingIsCountedApartTest(unittest.TestCase):
    def test_it_has_its_own_counter(self):
        r = _runtime(direction="tp_to_pp", abandons=8, measured_draw_mib=1452)
        with _Env(SGLANG_SEAM_ENTRY_MARGIN_MIB=512, SGLANG_SEAM_ENTRY_DELAY_BUDGET=2):
            with _Patched(_Guard(ENTRY_FREE)):
                r._corridor_gate(STAGED, "tp_to_pp")
        self.assertEqual(r.seam_yields_withheld, 1)
        self.assertEqual(r.seam_margin_yields, 0)


if __name__ == "__main__":
    unittest.main()
