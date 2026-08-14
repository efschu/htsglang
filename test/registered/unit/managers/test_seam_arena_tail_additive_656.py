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
"""#656: the arena tail is ADDITIVE against the wave state, not an alternative to it.

WHAT THE MODEL SAID
-------------------
``_staging_bytes`` returned ``max(wave_peak, draft_restore, arena_tail)`` on
the reasoning that the peaks belong to different instants of the seam and
never coexist. For the drafter's restore that reasoning holds -- it runs
inside ``_cutover``, after the waves' buffers are dead. For the arena tail it
does not: ``stacks.refill`` is a PRE-cutover function
(``phase_flip_runtime.py:1655``, census label ``weights_refill``), so it
commits while the seam's own state is still outstanding.

WHAT THE STAGE WALK SAYS (MERGE-R9 12.4)
----------------------------------------
One cutover, ``/spinning/evidence-631/remediation-656/boot_m1.log``,
``tp_to_pp`` rank 1, quoted from the census line's own fields::

    transient 1452 MiB (baseline free 2464 MiB, trough 1012 MiB at
    'weights_refill') *** CORRIDOR LAW BROKEN: 1 stage(s) below 1024 MiB,
    deepest 1012 MiB at 'weights_refill' ***
    ...
    backing_restore free=1250 | kv_write free=1250 | gdn_state free=1250
    weights_refill free=1012 step-238 | cutover free=1290 step+278

Read it in order: the card entered at 2464 MiB free, the wave walk carried it
down (deepest pre-refill reading 1078 MiB, i.e. a 1386 MiB wave peak) and was
still 1214 MiB below entry when the refill began at 1250 -- and THEN the
refill's own 238 MiB arena-tail commit landed on top of that, troughing at
1012 MiB, twelve MiB below the corridor law.

``max(1386, 238) = 1386`` predicts a trough of 1078 MiB, 54 MiB CLEAR of the
law. The seam entered on that prediction and broke the law. The additive form
predicts 1624 MiB drawn and a trough of 840, i.e. it foresees the breach that
happened. **This is not a conservatism argument; the max() form is the reason
the one measured corridor breach in this corpus was not foreseen.**

THE OVER-RESERVATION, STATED HONESTLY
-------------------------------------
The additive form bounds the outstanding wave state by the wave PEAK, and by
refill time the walk is often partly drained -- so it over-reserves. Priced
across the most-repeated ``tp_to_pp`` cutovers of that boot (MiB):

    rk  entry  wave_peak  arena  max()  additive  measured  max err  add err
     1   2464       1386    238   1386      1624      1452     +66      -172
     2   2952        492   1052   1052      1544      1276    +224      -268
     0   4585       1256    428   1256      1684      1256      +0      -428
     1   2128        158    594    594       752       580     -14      -172
     2   2952        448   1052   1052      1500      1232    +180      -268
     1   2038        418    466    466       884       712    +246      -172
     2   2854        592    924    924      1516      1248    +324      -268
     1   2102        370    466    466       836       664    +198      -172

``max err`` positive means the model reserved LESS than the cutover drew. The
max() form does that on six of the eight rows, by up to 324 MiB; the additive
form does it on none. On the breach row the additive residual is 172 MiB,
inside the sizer's own 192 MiB error bar (``DEFAULT_MARGIN_MIB``). On the
``kv_pack``-trough rows, where the wave has drained before the refill, it is
loose by up to 428 MiB and that is not claimed as agreement.

Over-reserving is the correct error for an affordability gate: it costs a
delayed flip, and under-reserving costs the breach above. The livelock
objection in ``_staging_bytes``' own docstring -- that a larger reservation
reaches the wedge at a SMALLER request -- does not apply to this term: the
arena tail is a static LAYOUT quantity that does not scale with the resident
set, so adding it shifts the affordable pool by a constant rather than by
something the prompt controls.

CAN-FAIL PROOF: restore the ``max()`` and every test in
``TheArenaTailCoexistsWithTheWaveStateTest`` and
``TheMeasuredBreachIsPredictedTest`` goes red.

Run:
  CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python python -m pytest \
      test/registered/unit/managers/test_seam_arena_tail_additive_656.py -q
"""

from __future__ import annotations

import types
import unittest

import torch

from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP
from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1024 * 1024


def _empty_transition():
    """A transition with an EMPTY live set: no send, recv or local rows.

    Every KV leg of ``_staging_bytes`` evaluates to zero on it, so whatever
    the method returns is the seam's own at-rest cost and nothing else. That
    is what makes the wave term controllable from the test.
    """
    empty = torch.empty(0, dtype=torch.int64)
    return types.SimpleNamespace(
        local_pp_rows=empty,
        local_tp_rows=empty,
        local_layers=[],
        send_layers={},
        recv_layers={},
        send_rows={},
        recv_rows={},
    )


class _Carrier:
    def __init__(self, tail_bytes: int):
        self._tail = int(tail_bytes)

    def pending_tail_bytes(self, _high_water: int) -> int:
        return self._tail


def _runtime(*, arena_tail_mib: int, wave_mib: int, draft_mib: int = 0):
    """A bare runtime whose three staging terms are each pinned to a number."""

    class _Stacks:
        arena_carrier = _Carrier(arena_tail_mib * MIB)

        @staticmethod
        def refill_high_water_bytes():
            return 0

    rt = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    rt._census_scheduler = types.SimpleNamespace(phase_flip_stacks=_Stacks())
    rt._backing_slack_bytes = lambda *a, **k: wave_mib * MIB
    rt._draft_restore_bytes = lambda direction: draft_mib * MIB
    return rt


def _pool():
    """A pool view with no rows and no layers -- every KV leg reads zero."""
    return types.SimpleNamespace(num_layers=0, row_nbytes=lambda i: 0)


def _staging_mib(rt, direction) -> int:
    tr = _empty_transition()
    src = dst = _pool()
    return rt._staging_bytes(tr, direction, src, dst, (None,)) // MIB


class TheArenaTailCoexistsWithTheWaveStateTest(unittest.TestCase):
    def test_the_tail_is_added_to_the_wave_peak_not_maxed_against_it(self):
        rt = _runtime(arena_tail_mib=238, wave_mib=1214)
        self.assertEqual(_staging_mib(rt, TP_TO_PP), 1214 + 238)

    def test_it_is_additive_on_the_pp_to_tp_leg_too(self):
        """``stacks.refill`` is a pre-cutover function on BOTH legs.

        The leg that has to GROW the arena is a property of the layout sizes,
        not of the direction (the 2026-08-11 three-rank death), so the
        coexistence the walk shows on tp_to_pp is not tp_to_pp-specific.
        """
        rt = _runtime(arena_tail_mib=300, wave_mib=900)
        self.assertEqual(_staging_mib(rt, PP_TO_TP), 900 + 300)

    def test_a_dominant_tail_still_carries_the_wave_state(self):
        """The rank-2 shape: a 1052 MiB tail over a small wave residual.

        Under max() the wave term vanished entirely whenever the tail was
        larger, which is where the 11 % aggregate under-reservation came from.
        """
        rt = _runtime(arena_tail_mib=1052, wave_mib=224)
        self.assertEqual(_staging_mib(rt, TP_TO_PP), 1052 + 224)

    def test_no_carrier_leaves_the_wave_peak_exactly_as_it_was(self):
        """Byte-identical where there is no tail to charge.

        Every non-flip boot and every rank whose layouts are equal must be
        untouched by this correction.
        """
        rt = _runtime(arena_tail_mib=0, wave_mib=777)
        self.assertEqual(_staging_mib(rt, TP_TO_PP), 777)


class TheDraftRestoreStaysAnAlternativeTest(unittest.TestCase):
    """The drafter's restore is NOT reclassified, and the distinction is the
    finding rather than an omission.

    Rung 2's restore runs inside ``_cutover``, after the waves' buffers are
    dead and after the source pool's pages have gone back. Nothing in the
    stage walk shows it coexisting with wave state, so it keeps its max().
    Making both additive would reserve a peak no measurement has recorded.
    """

    def test_a_large_draft_restore_binds_alone_over_the_wave_term(self):
        rt = _runtime(arena_tail_mib=0, wave_mib=100, draft_mib=900)
        self.assertEqual(_staging_mib(rt, PP_TO_TP), 900)

    def test_the_tail_is_additive_against_whichever_of_the_two_binds(self):
        """The arena tail is a COMMIT that persists into the next phase.

        It is still held when the cutover restores the drafter, so it adds to
        the larger of the two transients rather than to only one of them.
        """
        rt = _runtime(arena_tail_mib=200, wave_mib=100, draft_mib=900)
        self.assertEqual(_staging_mib(rt, PP_TO_TP), 200 + 900)


#: MEASURED ``tp_to_pp`` cutovers, extracted from the ``seam-census`` lines of
#: /spinning/evidence-631/remediation-656/boot_m1.log. One row per distinct
#: census line; ``n`` is how many times that exact line repeats in the boot.
#:
#:   entry      free at the 'entry' stage, MiB
#:   wave_peak  entry minus the DEEPEST free reading before 'weights_refill'
#:              -- the seam's own excursion, which is what the runtime's
#:              wave term bounds
#:   arena      the 'weights_refill' step, i.e. the arena-tail commit
#:   measured   the census's own ``transient``, entry minus the overall trough
#:   trough_at  which stage held the trough
#:
#: ONE BOOT, DELIBERATELY. Mixing boots would put layouts and pool sizes on
#: the same axis and make the residuals uninterpretable.
MEASURED_EVENTS = (
    # rank, entry, wave_peak, arena, measured, trough_at, n
    (1, 2464, 1386, 238, 1452, "weights_refill", 1),  # the corridor breach
    (2, 2952, 492, 1052, 1276, "weights_refill", 6),
    (0, 4585, 1256, 428, 1256, "kv_pack", 6),
    (1, 2128, 158, 594, 580, "weights_refill", 5),
    (2, 2952, 448, 1052, 1232, "weights_refill", 4),
    (1, 2038, 418, 466, 712, "weights_refill", 2),
    (2, 2854, 592, 924, 1248, "weights_refill", 2),
    (1, 2102, 370, 466, 664, "weights_refill", 2),
)

#: The sizer's own error bar (``phase_flip_seam_reserve.DEFAULT_MARGIN_MIB``).
SOLVER_MARGIN_MIB = 192


class TheModelIsCheckedAgainstMeasuredEventsTest(unittest.TestCase):
    """Before/after, priced on the census's own numbers.

    THE STANDARD A GATE IS HELD TO IS ONE-SIDED. Under-predicting is the
    failure -- the seam enters on a verdict that cannot see what it is about
    to spend. Over-predicting costs a delayed flip. So the assertions below
    are "never under" plus "close on the shape the walk actually shows",
    not "close on every event".
    """

    def _predicted(self, wave, arena):
        return _staging_mib(_runtime(arena_tail_mib=arena, wave_mib=wave), TP_TO_PP)

    def test_the_corrected_model_never_under_predicts_a_measured_event(self):
        under = []
        for rank, entry, wave, arena, measured, at, n in MEASURED_EVENTS:
            got = self._predicted(wave, arena)
            if got < measured:
                under.append(
                    f"rank{rank} entry {entry}: predicted {got} < measured "
                    f"{measured} (wave {wave}, arena {arena}, trough at {at})"
                )
        self.assertEqual([], under, "\n  ".join(under))

    def test_the_max_form_DID_under_predict_and_that_is_the_defect(self):
        """The falsifier for the whole change.

        If the old form had bounded these events, correcting it would be
        taste. It under-predicts six of the eight, by up to 324 MiB.
        """
        under = [
            (rank, entry, measured - max(wave, arena))
            for rank, entry, wave, arena, measured, _at, _n in MEASURED_EVENTS
            if max(wave, arena) < measured
        ]
        self.assertGreaterEqual(len(under), 5, under)
        self.assertGreaterEqual(max(d for _r, _e, d in under), 180, under)

    def test_it_is_within_the_solver_margin_where_the_walk_coexists(self):
        """On the breach event -- the one the correction exists for -- the
        corrected prediction lands inside the sizer's own 192 MiB error bar.

        Only that event is claimed. Where the wave has DRAINED by refill time
        (the ``kv_pack`` trough rows) the bound is loose by design, and saying
        otherwise would be the over-claim this file is meant to prevent.
        """
        rank, entry, wave, arena, measured, at, _n = MEASURED_EVENTS[0]
        self.assertEqual((rank, entry, at), (1, 2464, "weights_refill"))
        got = self._predicted(wave, arena)
        self.assertGreaterEqual(got, measured)
        self.assertLessEqual(
            got - measured,
            SOLVER_MARGIN_MIB,
            f"corrected prediction {got} MiB overshoots the measured "
            f"{measured} MiB by more than the sizer's own {SOLVER_MARGIN_MIB} "
            "MiB error bar",
        )

    def test_the_loose_rows_are_loose_in_the_safe_direction_only(self):
        """Named rather than hidden: where it over-reserves, by how much."""
        overs = {
            (rank, entry): self._predicted(wave, arena) - measured
            for rank, entry, wave, arena, measured, _at, _n in MEASURED_EVENTS
        }
        self.assertTrue(all(v >= 0 for v in overs.values()), overs)
        self.assertLessEqual(max(overs.values()), 500, overs)


class TheMeasuredBreachIsPredictedTest(unittest.TestCase):
    """The regression case, in the units the census reports.

    boot_m1.log, tp_to_pp rank 1: entry 2464 MiB free, deepest pre-refill
    reading 1078 MiB, 1250 MiB free when the refill began, 238 MiB committed
    by it, trough 1012 MiB against the 1024 MiB corridor law.
    """

    ENTRY_FREE = 2464
    WAVE_PEAK = 1386  # 2464 - 1078, the deepest pre-refill excursion
    PRE_REFILL_FREE = 1250
    REFILL_STEP = 238
    MEASURED_TROUGH = 1012
    MEASURED_TRANSIENT = 1452
    LAW = 1024

    def test_the_walk_arithmetic_is_quoted_consistently(self):
        """Guards the fixture itself: these numbers are one census line and
        must stay internally consistent, or the tests below prove nothing."""
        self.assertEqual(self.PRE_REFILL_FREE - self.REFILL_STEP, self.MEASURED_TROUGH)
        self.assertEqual(
            self.ENTRY_FREE - self.MEASURED_TROUGH, self.MEASURED_TRANSIENT
        )
        self.assertLess(self.MEASURED_TROUGH, self.LAW)
        # The refill began ABOVE the walk's deepest point, which is why the
        # arena tail alone does not explain the trough and why the two terms
        # have to be carried together.
        self.assertGreater(self.PRE_REFILL_FREE, self.ENTRY_FREE - self.WAVE_PEAK)

    def test_the_corrected_model_would_have_foreseen_the_breach(self):
        rt = _runtime(arena_tail_mib=self.REFILL_STEP, wave_mib=self.WAVE_PEAK)
        predicted_trough = self.ENTRY_FREE - _staging_mib(rt, TP_TO_PP)
        self.assertLess(
            predicted_trough,
            self.LAW,
            "the corrected staging must put this cutover's trough under the "
            "corridor law -- that is the breach the census recorded, and a "
            "model that still clears it has not been corrected",
        )

    def test_the_max_form_would_not_have_foreseen_it(self):
        """The falsifier. Without this the test above passes on any model that
        merely reserves a lot."""
        max_form = max(self.WAVE_PEAK, self.REFILL_STEP)
        self.assertGreater(
            self.ENTRY_FREE - max_form,
            self.LAW,
            "the pre-#656 max() form must clear the law on this event, or "
            "this case is not the regression it claims to be",
        )


class TheSizerCarriesTheSameSplitTest(unittest.TestCase):
    """``phase_flip_seam_reserve`` is the sizer's copy of the same law.

    It had its own ``max(arena, draft)``, so correcting only the runtime would
    leave the pool sized against the defect the gate had stopped having.
    """

    def test_a_record_without_the_arena_field_is_byte_identical(self):
        """The back-compat path, asserted rather than assumed.

        Every record written before the split carries the folded floor and no
        arena entry; those boots must size exactly as they did.
        """
        from sglang.srt.managers.phase_flip_seam_reserve import solve_pool_tokens

        args = dict(corridor_relaxed_bytes=8 << 30, cell_bytes=4096, per_row_bytes=2.0)
        self.assertEqual(
            solve_pool_tokens(fixed_bytes=500 << 20, **args),
            solve_pool_tokens(fixed_bytes=500 << 20, arena_fixed_bytes=0, **args),
        )

    def test_the_arena_term_reduces_the_pool_and_the_draft_term_may_not(self):
        """The two floors must not be interchangeable.

        With the slack binding, the ALTERNATIVE floor drops out of the answer
        entirely and only the ADDITIVE one moves it. A solver that treated
        them alike would have no way to express the walk.
        """
        from sglang.srt.managers.phase_flip_seam_reserve import solve_pool_tokens

        args = dict(corridor_relaxed_bytes=8 << 30, cell_bytes=4096, per_row_bytes=64.0)
        base = solve_pool_tokens(fixed_bytes=0, **args)
        with_draft = solve_pool_tokens(fixed_bytes=100 << 20, **args)
        with_arena = solve_pool_tokens(
            fixed_bytes=0, arena_fixed_bytes=100 << 20, **args
        )
        self.assertEqual(base, with_draft)
        self.assertLess(with_arena, base)

    def test_the_allowed_tokens_solver_carries_it_too(self):
        from sglang.srt.managers.phase_flip_seam_reserve import (
            SeamReserve,
            seam_allowed_tokens,
        )

        common = dict(per_row_bytes=64.0, have_bytes=4 << 30, id_space=500_000)
        folded = SeamReserve(fixed_bytes=400 << 20, **common)
        split = SeamReserve(fixed_bytes=0, arena_fixed_bytes=400 << 20, **common)
        self.assertLess(
            seam_allowed_tokens(4096, split),
            seam_allowed_tokens(4096, folded),
            "an additive floor must cost more pool than the same number of "
            "bytes spent as an alternative to the wave slack",
        )

    def test_both_floors_together_are_what_the_boot_must_be_able_to_fund(self):
        from sglang.srt.managers.phase_flip_seam_reserve import SeamReserve

        r = SeamReserve(fixed_bytes=300 << 20, arena_fixed_bytes=200 << 20)
        self.assertEqual(r.total_fixed_bytes, 500 << 20)

    def test_a_reserve_carrying_only_an_arena_tail_is_active(self):
        """``active`` gates the whole correction. A rank whose only seam cost
        is the arena tail must not read as "nothing measured"."""
        from sglang.srt.managers.phase_flip_seam_reserve import SeamReserve

        self.assertTrue(SeamReserve(arena_fixed_bytes=200 << 20, id_space=1000).active)


if __name__ == "__main__":
    unittest.main()
