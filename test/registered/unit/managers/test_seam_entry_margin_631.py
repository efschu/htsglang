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
"""#656 register C20: the seam must ENTER with headroom, not merely be legal.

WHAT C20 IS. The corridor's deepest troughs are made INSIDE the flip cutover
-- 20/20 of the deepest samples of successor 36's window sit within 2 s of a
cutover, and successor 37 measured the shape on successor 34's own GREEN
window (``evidence-631/s37/C20_SIZING.txt``):

  * the cutover draws up to 456 MiB on the binding card FROM A LOW ENTRY
    (the draw is self-limiting: big draws only ever follow a high entry),
  * the deep entries are INHERITED -- the deepest minima come in pairs of
    cutovers ~2 s apart, the second entering at the first one's trough,
  * so s34 held the law by +19 MiB and s36's identical trough missed it by
    -23 MiB. The margin at that instant was never designed; it was luck.

The gate already standing at the seam cannot close this, and the reason is
arithmetic rather than wiring: it clears whenever the LAW is satisfied, so a
seam that enters at 1043 MiB with a small staging ask is waved through with
19 MiB to spare. It is doing exactly what it was asked. It was asked for the
wrong thing.

WHAT THIS FILE PINS

* **The gate asks for the seam's staging PLUS a designed margin.** One ask,
  one ladder, one refusal path -- the margin is a term, not a mechanism.

* **A seam that cannot reach the margin but CAN satisfy the law is DELAYED,
  not breached and not blindly proceeded with.** A delay is nearly free at
  this seam: the paired-trough measurement shows the memory comes back, and
  the flip retries on the next round.

* **The delay is BUDGETED, because an unbounded refusal of pp->tp starves
  decode.** Measured 2026-08-10: 411 abandons, 0 requests in 6 minutes,
  /health 503 with every rank alive. Once the budget is spent the gate
  yields to the LAW -- which is exactly successor 34's shipped behaviour, so
  the worst case of this feature is the behaviour it replaces.

* **A seam below the LAW is refused however exhausted the budget is.** This
  is the falsifier the whole item exists for: there is no path through this
  gate that proceeds into a corridor breach.

* **The decision rides ``too_small`` into the existing ``_collective_min``**,
  so it is rank-EVALUATED and group-UNIFORM. A rank-local arming condition
  that entered a collective of its own is the known desync trap
  (``collective_kv_backing_relief``'s docstring, and the 37371/28677/32344
  hook-call desync in ``on_round``). This adds no collective at all.

Hermetic: a stub runtime, an injected guard, no CUDA.
"""

from __future__ import annotations

import unittest

from sglang.srt.managers import phase_flip_spill
from sglang.srt.managers.corridor_guard import GuardResult
from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

MIB = 1024 * 1024
STAGING = 300 * MIB


class _Guard:
    """A guard that answers by SIZE, the way the real one does.

    ``capacity`` is the largest ``want`` this card can satisfy and still be
    at or above the corridor law. Asking for more is refused. That is the
    real ``ensure_headroom`` contract compressed to one number:
    ``ok = (free_after_reclaim - want) >= law_floor``.
    """

    def __init__(self, capacity_bytes):
        self.capacity = capacity_bytes
        self.asks = []

    def ensure_headroom(self, want, *, reason="", refusal_is_fatal=False):
        want = int(want)
        self.asks.append((want, reason, refusal_is_fatal))
        if want <= self.capacity:
            return GuardResult(True, 0, 0, want, 0, ("allocator-cache",), "cleared")
        return GuardResult(
            False, 0, 0, want, 0, (), f"short by {(want - self.capacity) // MIB} MiB"
        )


def _runtime(collective_min=None):
    r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    r._census_scheduler = object()
    r.corridor_aborts = 0
    r.corridor_reclaims = 0
    r._corridor_pp_refusals = 0
    r.corridor_kv_relief_count = 0
    r.corridor_kv_relief_bytes = 0
    r._collective_min = collective_min or (lambda vals, **kw: list(vals))
    return r


class _Patched:
    """Swap ``get_corridor_guard`` (and optionally the KV rung) for one test."""

    def __init__(self, guard, kv_spy=None):
        self.g = guard
        self.kv_spy = kv_spy

    def __enter__(self):
        self.old = phase_flip_spill.get_corridor_guard
        phase_flip_spill.get_corridor_guard = lambda _scheduler: self.g
        self.old_kv = phase_flip_spill.collective_kv_backing_relief
        if self.kv_spy is not None:
            phase_flip_spill.collective_kv_backing_relief = self.kv_spy
        else:
            phase_flip_spill.collective_kv_backing_relief = (
                lambda *a, **k: 0  # noqa: ARG005
            )
        return self.g

    def __exit__(self, *exc):
        phase_flip_spill.get_corridor_guard = self.old
        phase_flip_spill.collective_kv_backing_relief = self.old_kv
        return False


class _Margin:
    """Set the margin/budget environment for one test."""

    def __init__(self, margin_mib=None, budget=None):
        self.env = {}
        if margin_mib is not None:
            self.env["SGLANG_SEAM_ENTRY_MARGIN_MIB"] = str(margin_mib)
        if budget is not None:
            self.env["SGLANG_SEAM_ENTRY_DELAY_BUDGET"] = str(budget)

    def __enter__(self):
        import os

        self.old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        return self

    def __exit__(self, *exc):
        import os

        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


class TestTheGateAsksForTheMargin(unittest.TestCase):
    def test_the_ask_is_staging_plus_the_designed_margin(self):
        """One ask, one ladder. The margin is a term, not a second mechanism."""
        with _Margin(margin_mib=512):
            g = _Guard(capacity_bytes=4096 * MIB)
            with _Patched(g):
                detail = _runtime()._corridor_gate(STAGING, "pp_to_tp")
        self.assertEqual(detail, "")
        self.assertEqual(g.asks[0][0], STAGING + 512 * MIB)

    def test_a_zero_margin_reproduces_the_previous_single_ask_exactly(self):
        """The off switch is a value of the same term, not a second path."""
        with _Margin(margin_mib=0):
            g = _Guard(capacity_bytes=4096 * MIB)
            with _Patched(g):
                detail = _runtime()._corridor_gate(STAGING, "pp_to_tp")
        self.assertEqual(detail, "")
        self.assertEqual([a[0] for a in g.asks], [STAGING])

    def test_the_kv_rung_is_asked_to_fund_the_margin_too(self):
        """Item 12's rung is the funder of last resort at this seam.

        Its deficit is ``floor + delta + want - free - cheap_relief``. If
        ``want`` excluded the margin the rung would decline a gap the gate is
        about to refuse for.
        """
        seen = {}

        def spy(_sched, _reduce, *, want_bytes, guard, direction):
            seen["want"] = int(want_bytes)
            return 0

        with _Margin(margin_mib=512):
            with _Patched(_Guard(4096 * MIB), kv_spy=spy):
                _runtime()._corridor_gate(STAGING, "pp_to_tp")
        self.assertEqual(seen["want"], STAGING + 512 * MIB)


class TestShortOfTheMarginDelays(unittest.TestCase):
    def test_margin_short_but_law_clear_delays_the_flip(self):
        """The C20 case: legal, and entered anyway would be luck."""
        with _Margin(margin_mib=512, budget=2):
            # Can fund the staging, cannot fund staging+margin.
            g = _Guard(capacity_bytes=STAGING + 100 * MIB)
            r = _runtime()
            with _Patched(g):
                detail = r._corridor_gate(STAGING, "pp_to_tp")
        self.assertNotEqual(detail, "", "a seam short of the margin must not enter")
        self.assertIn("margin", detail.lower())
        self.assertEqual(r.seam_margin_delays, 1)
        self.assertEqual(r.seam_margin_yields, 0)

    def test_the_delay_is_a_string_not_a_raise(self):
        """A raise on this path climbs into the event loop and kills ranks."""
        with _Margin(margin_mib=512, budget=2):
            g = _Guard(capacity_bytes=STAGING)
            with _Patched(g):
                detail = _runtime()._corridor_gate(STAGING, "pp_to_tp")
        self.assertIsInstance(detail, str)

    def test_a_delay_is_not_counted_as_a_corridor_abort(self):
        """A delay is the flip waiting, not the corridor refusing an alloc.

        Conflating them would make the acceptance extract read a healthy
        margin-driven wait as the 411-abandon wedge.
        """
        with _Margin(margin_mib=512, budget=2):
            r = _runtime()
            with _Patched(_Guard(STAGING)):
                r._corridor_gate(STAGING, "pp_to_tp")
        self.assertEqual(r.corridor_aborts, 0)
        self.assertEqual(r.seam_margin_delays, 1)


class TestTheBudgetPreventsAWedge(unittest.TestCase):
    def test_an_exhausted_budget_yields_to_the_law(self):
        """Worst case of this feature == the behaviour it replaces.

        An unbounded margin refusal of pp->tp starves decode outright under
        strict purity (411 abandons / health 503, 2026-08-10). So the budget
        is spent and then the gate stands down to s34's law-only clearance.
        """
        with _Margin(margin_mib=512, budget=2):
            g = _Guard(capacity_bytes=STAGING + 100 * MIB)
            r = _runtime()
            with _Patched(g):
                verdicts = [r._corridor_gate(STAGING, "pp_to_tp") for _ in range(4)]
        self.assertNotEqual(verdicts[0], "")
        self.assertNotEqual(verdicts[1], "")
        self.assertEqual(verdicts[2], "", "budget spent -> the law governs")
        self.assertEqual(verdicts[3], "")
        self.assertEqual(r.seam_margin_delays, 2)
        self.assertEqual(r.seam_margin_yields, 2)

    def test_a_met_margin_restores_the_budget(self):
        """Otherwise one bad patch of a long run disarms the gate forever."""
        with _Margin(margin_mib=512, budget=2):
            tight = _Guard(capacity_bytes=STAGING + 100 * MIB)
            roomy = _Guard(capacity_bytes=4096 * MIB)
            r = _runtime()
            with _Patched(tight):
                r._corridor_gate(STAGING, "pp_to_tp")
            with _Patched(roomy):
                self.assertEqual(r._corridor_gate(STAGING, "pp_to_tp"), "")
            with _Patched(tight):
                self.assertNotEqual(r._corridor_gate(STAGING, "pp_to_tp"), "")
        self.assertEqual(r.seam_margin_delays, 2)
        self.assertEqual(r.seam_margin_yields, 0)

    def test_the_two_directions_keep_separate_budgets(self):
        """tp->pp delays are safe; pp->tp delays starve decode. One counter
        for both would let the safe leg spend the dangerous leg's budget."""
        with _Margin(margin_mib=512, budget=1):
            g = _Guard(capacity_bytes=STAGING + 100 * MIB)
            r = _runtime()
            with _Patched(g):
                self.assertNotEqual(r._corridor_gate(STAGING, "tp_to_pp"), "")
                self.assertEqual(r._corridor_gate(STAGING, "tp_to_pp"), "")
                # pp->tp has not spent anything yet.
                self.assertNotEqual(r._corridor_gate(STAGING, "pp_to_tp"), "")


class TestNoPathProceedsIntoABreach(unittest.TestCase):
    """THE FALSIFIER. Everything above is optimisation; this is the law."""

    def test_a_seam_below_the_law_is_refused_even_with_the_budget_spent(self):
        with _Margin(margin_mib=512, budget=1):
            # Cannot even fund the staging: the LAW would be broken.
            g = _Guard(capacity_bytes=STAGING - 1)
            r = _runtime()
            with _Patched(g):
                verdicts = [r._corridor_gate(STAGING, "pp_to_tp") for _ in range(5)]
        self.assertTrue(all(v != "" for v in verdicts), "the law is not budgeted")
        self.assertTrue(all("refused" in v for v in verdicts))
        self.assertEqual(r.seam_margin_yields, 0, "a yield here would be a breach")
        self.assertEqual(r.corridor_aborts, 5)

    def test_a_law_refusal_is_reported_as_a_refusal_not_as_a_delay(self):
        """The two are different events and the extract must not merge them."""
        with _Margin(margin_mib=512, budget=4):
            r = _runtime()
            with _Patched(_Guard(STAGING - 1)):
                detail = r._corridor_gate(STAGING, "pp_to_tp")
        self.assertIn("refused", detail)
        self.assertEqual(r.seam_margin_delays, 0)
        self.assertEqual(r.corridor_aborts, 1)

    def test_a_guard_that_cannot_be_built_does_not_invent_a_margin(self):
        """No guard means no reading; a delay on no evidence is a wedge."""
        old = phase_flip_spill.get_corridor_guard
        old_kv = phase_flip_spill.collective_kv_backing_relief
        phase_flip_spill.get_corridor_guard = lambda _s: None
        phase_flip_spill.collective_kv_backing_relief = lambda *a, **k: 0  # noqa: ARG005
        try:
            with _Margin(margin_mib=512, budget=2):
                r = _runtime()
                self.assertEqual(r._corridor_gate(STAGING, "pp_to_tp"), "")
                self.assertEqual(r.seam_margin_delays, 0)
        finally:
            phase_flip_spill.get_corridor_guard = old
            phase_flip_spill.collective_kv_backing_relief = old_kv


class TestTheDecisionIsGroupUniform(unittest.TestCase):
    def test_the_verdict_travels_by_the_existing_collective_min(self):
        """No new collective. The delay joins ``too_small`` and is reduced
        with the fit verdict that was already unanimous.

        Asserted on the SOURCE, because ordering and the absence of a
        reduction are not observable from a single-rank call.
        """
        import inspect

        src = inspect.getsource(PhaseFlipRuntime._corridor_gate)
        self.assertNotIn(
            "all_reduce", src, "the seam-entry margin must add no collective"
        )
        exec_src = inspect.getsource(PhaseFlipRuntime._execute)
        gate_at = exec_src.index("_corridor_gate")
        reduce_at = exec_src.index("_collective_min([fits, -fits])")
        self.assertLess(
            gate_at, reduce_at, "the gate's verdict must reach the reduction"
        )

    def test_every_rank_evaluates_the_same_term(self):
        """The margin is a constant, not a rank-local reading, so three ranks
        asking their own guards still ask for the same additional bytes."""
        with _Margin(margin_mib=512):
            asks = []
            for _rank in range(3):
                g = _Guard(capacity_bytes=4096 * MIB)
                with _Patched(g):
                    _runtime()._corridor_gate(STAGING, "pp_to_tp")
                asks.append(g.asks[0][0] - STAGING)
        self.assertEqual(asks, [512 * MIB] * 3)


if __name__ == "__main__":
    unittest.main()
