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
"""#662-F4: A SEAM REFUSAL THAT CAN BE POINTED AT ONE DIRECTION.

THE PROBLEM THIS EXISTS FOR IS A CATCH-22, NOT A MISSING FEATURE.

1d1dbf9dba added the decode-stall SLO valve: decodes are never held past the
operator's bound by a funding failure. Its desk falsifier is complete, and on
2026-08-15 the invariant could still not be shown on metal even once, because
the shape it needs is unreachable by configuration. The shape is:

    the instance is IN the PP layout, a decode is resident, and pp_to_tp
    cannot be funded

and every lever available moved BOTH directions at once:

    the KV rung disabled     pp_to_tp funded from the allocator cache anyway
                             -- and, because ``refusal_is_fatal`` deliberately
                             opens the host tier for this leg, from system RAM
                             too. Six flips DONE, zero abandons, nothing held.

    the arming floor raised  tp_to_pp was blocked with it, so the instance
                             never entered PP at all and decode was never held.

One global gate, two directions, and the proof needs them to disagree. The
first two classes below PIN THAT CATCH-22 with the shipped knobs, so the
justification for a new term is a test rather than a paragraph; the rest hold
the term itself to what it promised.

THE PROPERTY THAT MATTERS MOST HERE IS NOT THE REFUSAL. It is that the gate
still runs in full when the refusal is injected. Every collective inside
``_corridor_gate`` -- the KV rung's reduction above all -- lives there
precisely because it is on a path every rank reaches unconditionally, so an
injection that returned early would be a collective hang the first time one
rank read the variable differently. ``TheGateStillRunsInFullTest`` is the one
that would catch that, and it is the reason the term wraps the gate instead of
living inside it.

Hermetic: the stub runtime and injected guard of ``test_seam_entry_margin_631``
and ``test_seam_yield_draw_656``. No CUDA, no collectives, no model.
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
from sglang.srt.managers.phase_flip_spill import (
    ENV_SEAM_UNFUNDABLE,
    seam_unfundable_objection,
    unfundable_seam_directions,
)

MIB = 1024 * 1024
LAW = 1024 * MIB

PP_TO_TP = "pp_to_tp"
TP_TO_PP = "tp_to_pp"

#: A staging ask that a comfortable card clears without any tier being asked
#: for much. The point of these tests is never the arithmetic -- it is which
#: direction the verdict lands on.
STAGED = 200 * MIB
COMFORTABLE = 8192 * MIB
#: Far enough below the law that the gate refuses outright rather than taking
#: the C20 margin-delay branch -- these tests want a REFUSAL to override, not
#: a bounded wait.
STARVED = 300 * MIB


class _Guard:
    def __init__(self, free_after, law_floor_bytes=LAW):
        self.law_floor_bytes = law_floor_bytes
        self.free_after = free_after
        self.asks = []
        self.fatal_flags = []

    def ensure_headroom(self, want, *, reason="", refusal_is_fatal=False):
        want = int(want)
        self.asks.append(want)
        self.fatal_flags.append(bool(refusal_is_fatal))
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
    """Inject the guard, and COUNT the KV rung's collective.

    The count is not decoration either: it is the evidence that the injected
    refusal did not skip the one reduction every rank must enter.
    """

    def __init__(self, guard):
        self.g = guard
        self.kv_calls = []

    def __enter__(self):
        self.old = phase_flip_spill.get_corridor_guard
        self.old_kv = phase_flip_spill.collective_kv_backing_relief
        phase_flip_spill.get_corridor_guard = lambda _s: self.g

        def _counted(*a, **k):
            self.kv_calls.append(k.get("direction"))
            return 0

        phase_flip_spill.collective_kv_backing_relief = _counted
        return self

    def __exit__(self, *exc):
        phase_flip_spill.get_corridor_guard = self.old
        phase_flip_spill.collective_kv_backing_relief = self.old_kv
        return False


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
        # The resolved set is cached per raw value, and the ARMED line is
        # emitted on the miss. Tests that assert on either need the cache
        # cleared or they are asserting on a previous test's state.
        phase_flip_spill._UNFUNDABLE_CACHE.clear()
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        phase_flip_spill._UNFUNDABLE_CACHE.clear()
        return False


def _runtime():
    r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    r._census_scheduler = object()
    r.corridor_aborts = 0
    r.corridor_reclaims = 0
    r._corridor_pp_refusals = 0
    r.corridor_kv_relief_count = 0
    r.corridor_kv_relief_bytes = 0
    r._collective_min = lambda vals, **kw: list(vals)
    r._seam_abandons_in_a_row = {PP_TO_TP: 0, TP_TO_PP: 0}
    r._seam_draw_max = {PP_TO_TP: 0, TP_TO_PP: 0}
    return r


def _verdict(direction, *, free=COMFORTABLE, **env):
    """One seam's funding verdict. '' means the flip may proceed."""
    r = _runtime()
    with _Env(**env):
        with _Patched(_Guard(free)) as p:
            return r._seam_funding_verdict(STAGED, direction), p


class TheShippedKnobsCannotProduceTheShapeTest(unittest.TestCase):
    """THE CATCH-22, pinned. These pass before the new term and after it.

    They are here so the justification for adding a term is a falsifiable
    statement about the shipped knobs rather than a claim in a commit message.
    If one of these ever fails, the new term may be redundant and should be
    re-argued.
    """

    def test_disabling_the_KV_rung_does_not_make_pp_to_tp_unfundable(self):
        """The measured proof1 result: the cheap tier pays and the flip goes.

        This is why "turn the rung off" was not the lever: the rung is one
        tier of several, and the seam only needs SOME tier to clear the law.
        """
        detail, patched = _verdict(
            PP_TO_TP,
            SGLANG_KV_BACKING_RELIEF="0",
            SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=None,
        )
        self.assertEqual(
            detail,
            "",
            "with the rung off the seam still funded -- which is exactly what "
            "metal measured, and why a rung switch cannot produce the shape",
        )

    def test_the_one_directional_knob_shipped_does_not_refuse_anything(self):
        """``SGLANG_SEAM_FUND_TP_TO_PP=0`` abstains the rung; it does not refuse.

        It is the only direction-scoped term that existed, and it is the wrong
        kind of term: it removes a FUNDER, it does not produce a REFUSAL.
        """
        detail, _ = _verdict(
            TP_TO_PP,
            SGLANG_SEAM_FUND_TP_TO_PP="0",
            SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=None,
        )
        self.assertEqual(detail, "")


class TheDefaultChangesNothingTest(unittest.TestCase):
    def test_unset_leaves_both_directions_exactly_as_the_gate_left_them(self):
        for d in (PP_TO_TP, TP_TO_PP):
            with self.subTest(direction=d):
                detail, _ = _verdict(d, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=None)
                self.assertEqual(detail, "")

    def test_an_empty_value_is_the_same_as_unset(self):
        for raw in ("", "   ", ","):
            with self.subTest(raw=raw):
                detail, _ = _verdict(PP_TO_TP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=raw)
                self.assertEqual(detail, "")

    def test_a_refusal_the_gate_reached_itself_is_passed_through_verbatim(self):
        """Unset must not reword the gate's own refusal either."""
        detail, _ = _verdict(
            PP_TO_TP, free=STARVED, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=None
        )
        self.assertNotEqual(detail, "")
        self.assertIn("corridor gate refused", detail)
        self.assertNotIn(ENV_SEAM_UNFUNDABLE, detail)


class TheLeverIsPerDirectionTest(unittest.TestCase):
    """THE FALSIFIER FOR THE WHOLE DESIGN: the two directions must disagree."""

    def test_naming_pp_to_tp_refuses_it_while_tp_to_pp_still_funds(self):
        refused, _ = _verdict(PP_TO_TP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=PP_TO_TP)
        funded, _ = _verdict(TP_TO_PP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=PP_TO_TP)
        self.assertNotEqual(refused, "", "the named direction must be refused")
        self.assertEqual(
            funded,
            "",
            "the OTHER direction must still fund, or the instance can never "
            "enter PP and the shape under test is unreachable -- which is the "
            "catch-22 this term exists to end",
        )

    def test_it_works_the_other_way_round_too(self):
        refused, _ = _verdict(TP_TO_PP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=TP_TO_PP)
        funded, _ = _verdict(PP_TO_TP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=TP_TO_PP)
        self.assertNotEqual(refused, "")
        self.assertEqual(funded, "")

    def test_both_may_be_named_at_once(self):
        raw = f"{PP_TO_TP},{TP_TO_PP}"
        for d in (PP_TO_TP, TP_TO_PP):
            with self.subTest(direction=d):
                detail, _ = _verdict(d, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=raw)
                self.assertNotEqual(detail, "")

    def test_surrounding_whitespace_is_not_a_different_direction(self):
        padded = f"  {PP_TO_TP} , {TP_TO_PP}  "
        detail, _ = _verdict(PP_TO_TP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=padded)
        self.assertNotEqual(detail, "")
        # Asserted INSIDE the env, or it reads a restored environment and
        # passes for the wrong reason.
        with _Env(SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=padded):
            self.assertEqual(
                unfundable_seam_directions(), frozenset({PP_TO_TP, TP_TO_PP})
            )


class TheGateStillRunsInFullTest(unittest.TestCase):
    """NO COLLECTIVE MAY BE SKIPPED BY THE INJECTION.

    This is the property the term's placement was chosen for, and the one a
    future "optimisation" that short-circuits the gate would break. A rank
    that skips the KV rung's reduction while its peers enter it does not
    refuse a flip -- it hangs the group.
    """

    def test_the_KV_rung_reduction_is_still_entered_on_a_refused_direction(self):
        _, patched = _verdict(PP_TO_TP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=PP_TO_TP)
        self.assertEqual(
            patched.kv_calls,
            [PP_TO_TP],
            "the rung's collective must be entered exactly as it would be "
            "without the injection",
        )

    def test_the_guard_is_still_asked_on_a_refused_direction(self):
        _, patched = _verdict(PP_TO_TP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=PP_TO_TP)
        self.assertEqual(len(patched.g.asks), 1)

    def test_the_ladder_is_asked_for_the_same_bytes_as_without_the_injection(self):
        _, clean = _verdict(PP_TO_TP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=None)
        _, armed = _verdict(PP_TO_TP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=PP_TO_TP)
        self.assertEqual(armed.g.asks, clean.g.asks)


class TheRefusalIsAnAbandonNotADelayTest(unittest.TestCase):
    def test_it_does_not_carry_the_margin_delay_tag(self):
        """A margin delay is a bounded wait the seam retries out of.

        Tagging the injection as one would prove the wrong thing: the seam
        would be delayed and the abandon accounting -- which the SLO proof
        needs to stay silent, and the operator needs to see -- would never
        run. The tag is also exempt from the abandon cap by design.
        """
        detail, _ = _verdict(PP_TO_TP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=PP_TO_TP)
        self.assertNotIn(SEAM_MARGIN_DELAY_TAG, detail)

    def test_the_message_names_the_variable_that_caused_it(self):
        detail, _ = _verdict(PP_TO_TP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=PP_TO_TP)
        self.assertIn(ENV_SEAM_UNFUNDABLE, detail)
        self.assertIn(PP_TO_TP, detail)

    def test_it_keeps_what_the_gate_itself_decided(self):
        """A proof that hides the state it overrode is not evidence.

        The operator must be able to read, from one line, that the seam WAS
        fundable and was refused anyway -- otherwise an injected boot cannot
        be told apart from a genuinely starved one after the fact.
        """
        detail, _ = _verdict(PP_TO_TP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=PP_TO_TP)
        self.assertIn("the gate itself said", detail)
        self.assertIn("the seam was fundable", detail)

    def test_a_real_refusal_underneath_is_preserved_not_replaced(self):
        detail, _ = _verdict(
            PP_TO_TP, free=STARVED, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=PP_TO_TP
        )
        self.assertIn(ENV_SEAM_UNFUNDABLE, detail)
        self.assertIn("corridor gate refused", detail)


class AnUnparseableValueInjectsNothingTest(unittest.TestCase):
    """AND IT MAY NOT RAISE. This term is read on the seam's no-return path.

    ``_abandon_parked_flip`` states the law: a raise from inside a cutover
    climbs into the event loop and takes the instance down. So a typo costs a
    diagnostic, never the instance -- and says so at ERROR, or the failure is
    the silent kind this corpus keeps re-finding.
    """

    def test_an_unknown_direction_arms_nothing(self):
        with _Env(SGLANG_SEAM_UNFUNDABLE_DIRECTIONS="pp2tp"):
            self.assertEqual(unfundable_seam_directions(), frozenset())
            self.assertIsNone(seam_unfundable_objection(PP_TO_TP))
            self.assertIsNone(seam_unfundable_objection(TP_TO_PP))

    def test_one_bad_name_rejects_the_whole_value(self):
        """Partial acceptance would arm a DIFFERENT experiment than the one
        the operator wrote, which is worse than arming none."""
        with _Env(SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=f"{PP_TO_TP},pp2tp"):
            self.assertEqual(unfundable_seam_directions(), frozenset())
            self.assertIsNone(seam_unfundable_objection(PP_TO_TP))

    def test_it_says_so_at_ERROR(self):
        with _Env(SGLANG_SEAM_UNFUNDABLE_DIRECTIONS="sideways"):
            with self.assertLogs(
                "sglang.srt.managers.phase_flip_spill", level="ERROR"
            ) as cm:
                unfundable_seam_directions()
        joined = "\n".join(cm.output)
        self.assertIn("sideways", joined)
        self.assertIn("NOT armed", joined)

    def test_the_seam_still_proceeds_normally_after_a_typo(self):
        detail, _ = _verdict(PP_TO_TP, SGLANG_SEAM_UNFUNDABLE_DIRECTIONS="nonsense")
        self.assertEqual(detail, "")


class AnArmedInjectionAnnouncesItselfTest(unittest.TestCase):
    """AN UNARMED INJECTION IS THE WORST FAILURE THIS TERM CAN HAVE.

    A flip that funded normally, in a boot the operator believes is injected,
    reads as a valve that never had to fire. That is the inverted false
    negative this corpus has shipped seven times, and the boot log is where it
    gets caught.
    """

    def test_the_boot_log_names_the_armed_directions(self):
        with _Env(SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=PP_TO_TP):
            with self.assertLogs(
                "sglang.srt.managers.phase_flip_spill", level="WARNING"
            ) as cm:
                unfundable_seam_directions()
        joined = "\n".join(cm.output)
        self.assertIn("UNFUNDABLE-SEAM", joined)
        self.assertIn("ARMED", joined)
        self.assertIn(PP_TO_TP, joined)

    def test_it_is_announced_once_not_once_per_seam(self):
        with _Env(SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=PP_TO_TP):
            with self.assertLogs(
                "sglang.srt.managers.phase_flip_spill", level="WARNING"
            ) as cm:
                for _ in range(5):
                    unfundable_seam_directions()
        armed = [line for line in cm.output if "UNFUNDABLE-SEAM" in line]
        self.assertEqual(len(armed), 1, "the ARMED line must not repeat per seam")

    def test_nothing_is_announced_when_it_is_unset(self):
        with _Env(SGLANG_SEAM_UNFUNDABLE_DIRECTIONS=None):
            with self.assertNoLogs(
                "sglang.srt.managers.phase_flip_spill", level="WARNING"
            ):
                self.assertEqual(unfundable_seam_directions(), frozenset())


if __name__ == "__main__":
    unittest.main()
