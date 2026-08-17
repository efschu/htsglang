# SPDX-License-Identifier: Apache-2.0
"""#305: the residency ladder's reachable edges, and the one that is absent
everywhere.

The table in ``registry/ladder.py`` is a DECLARATION. A declaration that drifts
from the adapters it describes is worse than none, so these tests pin it
against the adapters' own refusal text rather than restating it.
"""

import inspect
import unittest

from sglang.srt.registry import ladder
from sglang.srt.registry.ladder import (
    COLD,
    HOT,
    RUNG_ORDER,
    TEIL_HOT,
    WARM,
    LadderRefusal,
    can,
    check_transition,
    describe,
    reachable_edges,
    universally_absent,
)


class TestTheHeadline(unittest.TestCase):
    """No class has all four rungs, and one edge exists nowhere."""

    def test_TEIL_HOT_to_WARM_is_absent_for_every_class(self):
        absent = universally_absent()
        self.assertIn((TEIL_HOT, WARM), absent)
        self.assertIn((WARM, TEIL_HOT), absent)

    def test_no_class_implements_both_middle_rungs(self):
        for klass in ladder.CLASS_RUNGS:
            have = ladder.rungs_for(klass)
            self.assertFalse(
                TEIL_HOT in have and WARM in have,
                f"{klass} now has both middle rungs -- the #305 determination's "
                "central claim is retired; update it",
            )

    def test_every_class_has_the_two_END_rungs(self):
        """HOT and COLD are the ladder's ends and every class must span them,
        or it is not on the ladder at all."""
        for klass in ladder.CLASS_RUNGS:
            have = ladder.rungs_for(klass)
            self.assertIn(HOT, have)
            self.assertIn(COLD, have)

    def test_absence_is_computed_not_asserted(self):
        """If a class ever gains a rung, the claim retires by arithmetic."""
        src = inspect.getsource(universally_absent)
        self.assertIn("any(can(", src)


class TestTheTableMatchesTheAdapters(unittest.TestCase):
    """The anti-drift pin: the declared absences must match what the adapter
    source actually refuses."""

    def _src(self, mod):
        return " ".join(inspect.getsource(mod).split())

    def test_class1_really_refuses_WARM_HOST(self):
        from sglang.srt.registry.adapters import class1_srt

        self.assertIn("Class 1 has no WARM_HOST rung", self._src(class1_srt))
        self.assertNotIn(WARM, ladder.CLASS_RUNGS["class1_srt"])

    def test_class2_really_refuses_WARM_GPU(self):
        from sglang.srt.registry.adapters import class2_diffusion

        self.assertIn("no WARM_GPU endpoint", self._src(class2_diffusion))
        self.assertNotIn(TEIL_HOT, ladder.CLASS_RUNGS["class2_diffusion"])

    def test_class3_really_refuses_WARM_GPU(self):
        from sglang.srt.registry.adapters import class3_utility

        self.assertIn("no WARM_GPU rung", self._src(class3_utility))
        self.assertNotIn(TEIL_HOT, ladder.CLASS_RUNGS["class3_utility"])

    def test_every_declared_absence_carries_its_reason(self):
        for klass, have in ladder.CLASS_RUNGS.items():
            for rung in RUNG_ORDER:
                if rung in have:
                    continue
                self.assertIn(
                    (klass, rung),
                    ladder.RUNG_ABSENT_BECAUSE,
                    f"{klass} lacks {rung} with no stated reason",
                )


class TestItRefusesBeforeDrivingAnything(unittest.TestCase):
    def test_an_unbuilt_transition_names_the_missing_rung_and_why(self):
        with self.assertRaises(LadderRefusal) as e:
            check_transition("class1_srt", TEIL_HOT, WARM)
        msg = str(e.exception)
        self.assertIn("target rung WARM does not exist", msg)
        self.assertIn("not a reloadable host image", msg)

    def test_a_built_transition_passes(self):
        check_transition("class1_srt", HOT, TEIL_HOT)
        check_transition("class2_diffusion", HOT, WARM)
        check_transition("class3_utility", HOT, COLD)

    def test_an_unknown_class_is_refused_not_assumed_complete(self):
        with self.assertRaises(LadderRefusal) as e:
            ladder.rungs_for("class9_imaginary")
        self.assertIn("rather than assumed to have all four", str(e.exception))

    def test_a_self_transition_is_not_a_transition(self):
        with self.assertRaises(LadderRefusal):
            check_transition("class1_srt", HOT, HOT)

    def test_this_module_moves_NOTHING(self):
        """Same rule rungs.py states for itself: declaration only."""
        src = inspect.getsource(ladder)
        for mover in ("promote(", "demote(", "adapter.", "requests.", "subprocess"):
            self.assertNotIn(mover, src)


class TestTheReportIsUsable(unittest.TestCase):
    def test_describe_names_each_missing_rung_with_its_reason(self):
        text = describe("class1_srt")
        self.assertIn("no WARM", text)
        self.assertIn("§4.3", text)

    def test_reachable_edges_excludes_the_absent_rung(self):
        edges = reachable_edges("class1_srt")
        self.assertIn((HOT, TEIL_HOT), edges)
        self.assertNotIn((TEIL_HOT, WARM), edges)
        self.assertNotIn((HOT, WARM), edges)

    def test_can_is_symmetric_in_availability(self):
        self.assertTrue(can("class1_srt", HOT, TEIL_HOT))
        self.assertTrue(can("class1_srt", TEIL_HOT, HOT))
        self.assertFalse(can("class1_srt", HOT, WARM))


if __name__ == "__main__":
    unittest.main()
