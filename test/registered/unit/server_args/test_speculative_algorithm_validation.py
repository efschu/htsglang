# SPDX-License-Identifier: Apache-2.0
"""#379: --speculative-algorithm is validated at parse time.

The flag was free-form ``Optional[str]`` with no choices and no
normalisation, which produced two silent defects:

* ``--speculative-algorithm none`` left the STRING "none" in the field. It is
  truthy, so every ``if self.speculative_algorithm:`` in the tree took the
  speculative branch for a server with no drafter -- while
  ``SpeculativeAlgorithm.from_string("none")`` resolved the same input to
  ``NONE``. The field and the enum disagreed about whether the server was
  speculating.
* An unregistered name passed untouched and surfaced later as an unrelated
  guard's message that named neither the flag nor the valid values.

The regression corpus is the spellings that work today: the #349 boot-matrix
arms and the runbook recipes use ``NEXTN``, which is an ALIAS the arg hook
resolves (``NEXTN`` -> ``EAGLE``, or ``FROZEN_KV_MTP`` for a Gemma4 draft) --
the enum has no NEXTN member, so a validator that checked the enum alone
would break every recipe on this rig.

Hermetic: the validator is a method on ServerArgs and is called directly on a
bare instance, so no server and no model config is involved.
"""

import unittest

from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import (
    SPECULATIVE_ALGORITHM_ALIASES,
    SpeculativeAlgorithm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _validate(value):
    """Run only the #379 validator against a bare ServerArgs."""
    args = ServerArgs.__new__(ServerArgs)
    args.speculative_algorithm = value
    args._handle_speculative_algorithm_name()
    return args.speculative_algorithm


class TestTheNoneTrap(CustomTestCase):
    """The headline: "none" must mean OFF, not an algorithm called none."""

    def test_none_string_normalises_to_off(self):
        self.assertIsNone(_validate("none"))

    def test_every_casing_of_none_normalises_to_off(self):
        for spelling in ("none", "None", "NONE", "  none  "):
            self.assertIsNone(_validate(spelling), spelling)

    def test_empty_string_is_off(self):
        self.assertIsNone(_validate(""))
        self.assertIsNone(_validate("   "))

    def test_absent_flag_stays_absent(self):
        self.assertIsNone(_validate(None))

    def test_off_is_falsy_so_the_truthiness_tests_agree_with_the_enum(self):
        # The actual defect: `if self.speculative_algorithm:` must now agree
        # with `SpeculativeAlgorithm.from_string(...).is_none()`.
        value = _validate("none")
        self.assertFalse(bool(value))
        self.assertTrue(SpeculativeAlgorithm.from_string(value).is_none())

    def test_before_the_fix_the_string_would_have_been_truthy(self):
        # Pins WHY this matters, so a later change that reintroduces the
        # string form fails here with the reason attached.
        self.assertTrue(bool("none"), "the trap was that 'none' is truthy")
        self.assertTrue(SpeculativeAlgorithm.from_string("none").is_none())


class TestRegressionCorpus(CustomTestCase):
    """Every spelling that works today keeps working."""

    def test_the_nextn_alias_is_accepted(self):
        # The #349 arms and the runbook recipes. The enum has no NEXTN
        # member; the arg hook resolves it. A validator checking the enum
        # alone would break every recipe on this rig.
        self.assertEqual(_validate("NEXTN"), "NEXTN")

    def test_every_builtin_enum_name_is_accepted(self):
        for member in SpeculativeAlgorithm:
            if member is SpeculativeAlgorithm.NONE:
                continue
            self.assertEqual(_validate(member.name), member.name)

    def test_lowercase_is_accepted_because_the_hook_upper_cases(self):
        # After the hook the field is already upper-cased; the validator must
        # not be stricter than what reaches it in other orders.
        self.assertEqual(_validate("eagle"), "eagle")

    def test_surrounding_whitespace_does_not_reject(self):
        self.assertEqual(_validate("  EAGLE  "), "EAGLE")


class TestUnregisteredNamesAreRefused(CustomTestCase):
    def test_an_unknown_name_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _validate("EAGLE4")
        self.assertIn("EAGLE4", str(ctx.exception))

    def test_the_error_lists_the_valid_values(self):
        with self.assertRaises(ValueError) as ctx:
            _validate("nope")
        msg = str(ctx.exception)
        for expected in ("EAGLE", "EAGLE3", "NEXTN", "NGRAM", "STANDALONE"):
            self.assertIn(expected, msg)

    def test_the_error_does_not_offer_NONE_as_an_algorithm(self):
        # Listing NONE would re-teach the very trap this task removes.
        with self.assertRaises(ValueError) as ctx:
            _validate("nope")
        msg = str(ctx.exception)
        self.assertNotIn(", NONE", msg)
        self.assertIn("omit the flag", msg)

    def test_the_error_names_the_plugin_path(self):
        # A plugin algorithm whose module was not imported yet is the one
        # legitimate way a valid name looks invalid; say so.
        with self.assertRaises(ValueError) as ctx:
            _validate("MY_PLUGIN_ALGO")
        self.assertIn("register", str(ctx.exception))

    def test_a_near_miss_is_still_refused(self):
        for bad in ("EAGLE 3", "eagle-3", "next-n", "mtp"):
            with self.assertRaises(ValueError):
                _validate(bad)


class TestOneSourceOfTruth(CustomTestCase):
    """known_names() is THE namespace; no second list may drift from it."""

    def test_known_names_covers_the_enum_and_the_aliases(self):
        known = set(SpeculativeAlgorithm.known_names())
        for member in SpeculativeAlgorithm:
            self.assertIn(member.name, known)
        for alias in SPECULATIVE_ALGORITHM_ALIASES:
            self.assertIn(alias, known)

    def test_known_names_reflects_a_plugin_registered_later(self):
        # Read live, not snapshotted: a plugin registering during import must
        # be visible to a validator that runs after it.
        from sglang.srt.speculative import spec_registry

        before = set(SpeculativeAlgorithm.known_names())
        self.assertNotIn("ZZ_TEST_ALGO", before)
        spec_registry._REGISTRY["ZZ_TEST_ALGO"] = object()
        try:
            self.assertIn("ZZ_TEST_ALGO", SpeculativeAlgorithm.known_names())
        finally:
            spec_registry._REGISTRY.pop("ZZ_TEST_ALGO", None)

    def test_the_planner_pick_list_only_offers_resolvable_names(self):
        # planner/flags.py keeps a deliberately NARROWER UI pick-list. It is
        # not a validation list, but it must never offer something the server
        # would then reject.
        from sglang.srt.planner.flags import _CURATED

        allowed = _CURATED["speculative_algorithm"]["allowed"]
        known = set(SpeculativeAlgorithm.known_names())
        for name in allowed:
            self.assertIn(
                name,
                known,
                f"the planner offers {name!r} but the server would reject it",
            )


if __name__ == "__main__":
    unittest.main()
