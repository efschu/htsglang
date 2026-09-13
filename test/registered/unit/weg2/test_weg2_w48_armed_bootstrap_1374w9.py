# SPDX-License-Identifier: Apache-2.0
"""#1374 W9 -- a same-form stem that `solve` ELIMINATED is not a witness.

BOOT weg2xsn31, on 9f441e049e, refused:

    weg2xsn30_9423f16ba5_0913_134617: no sleep-pass lines for D.
    A same-form source EXISTS and did not win, so this is a real mismatch and
    not a first boot

weg2xsn30 died at WALL 8 before D ever slept, so its stem carries the armed
form key (0a15b55459fb) and NO D sleep-pass lines. #1367 counted it as a
witness because it was RANKED same-form, so the bootstrap branch did not fire,
the winner was weg2xsn25 in a foreign form, and the launcher refused. The
successor of every boot that dies at a wall would be locked out the same way,
and `--ring-table-boot` cannot help: the form gate runs after the stem choice
and meets the same missing D lines.

THE CLASS is the one my own `form_same` fix was, one level up: PRESENT is not
USABLE. `solve` already knows the difference -- it eliminates such a stem with
a reason -- and the launcher was reading the ranked list instead of the
surviving one.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher, ring_table
from sglang.test.test_utils import CustomTestCase

ARMED = sorted(launcher.WEIGHT_SOURCE_ARMED)[0]
XSN30 = "boot_weg2_weg2xsn30_9423f16ba5_0913_134617"


def _table(*, usable=(), unusable=()):
    return ring_table.RingTable(
        boot="boot_weg2_weg2xsn25_x", instrument="test", lines_read=1,
        form_key="0a15b55459fb", source_form_key="2b66740bedf9",
        form_same=False, form_diff="(diff)",
        same_form_candidates=tuple(usable),
        same_form_unusable=tuple(unusable),
    )


class AnEliminatedStemIsNotAWitness(CustomTestCase):
    def test_the_xsn31_case_bootstraps_instead_of_refusing(self):
        """THE WALL, as a fixture: xsn30 same-form, eliminated for its missing
        D sleep-pass lines, and nothing else of this form exists."""
        line = launcher.refuse_unless_same_form_source(
            _table(usable=(),
                   unusable=((XSN30, "no sleep-pass lines for D"),)), ARMED)
        self.assertIsNotNone(line, "an eliminated stem must not refuse the boot")
        self.assertIn("BOOTSTRAP armed first boot", line)

    def test_the_bootstrap_line_explains_the_zero_by_naming_the_stem(self):
        line = launcher.refuse_unless_same_form_source(
            _table(usable=(),
                   unusable=((XSN30, "no sleep-pass lines for D"),)), ARMED)
        self.assertIn("present but unusable", line)
        self.assertIn(XSN30, line)
        self.assertIn("no sleep-pass lines for D", line)

    def test_a_usable_same_form_source_still_refuses(self):
        """The refusal keeps its whole subject: a stem that SURVIVED solve's
        checks and still lost is a real mismatch."""
        with self.assertRaises(ring_table.Weg2RingFormMismatch) as cm:
            launcher.refuse_unless_same_form_source(
                _table(usable=("boot_weg2_weg2xsn28_x",)), ARMED)
        msg = str(cm.exception)
        self.assertIn("USABLE same-form source EXISTS", msg)
        self.assertIn("boot_weg2_weg2xsn28_x", msg)

    def test_a_refusal_names_the_eliminated_stems_as_not_the_reason(self):
        """Both present at once: one usable loser (the real mismatch) and one
        eliminated stem, which must be named as NOT the cause."""
        with self.assertRaises(ring_table.Weg2RingFormMismatch) as cm:
            launcher.refuse_unless_same_form_source(
                _table(usable=("boot_weg2_weg2xsn28_x",),
                       unusable=((XSN30, "no sleep-pass lines for D"),)), ARMED)
        msg = str(cm.exception)
        self.assertIn("ELIMINATED do not count as witnesses", msg)
        self.assertIn(XSN30, msg)


class SolveSplitsRankedFromSurviving(CustomTestCase):
    def test_the_split_is_derived_from_the_reasons_solve_already_writes(self):
        """One place, not the eight `reasons.append` sites."""
        import inspect

        src = inspect.getsource(ring_table.solve)
        self.assertIn("same_form_unusable=tuple(", src)
        self.assertIn('r.startswith(f"{st}:")', src)

    def test_the_field_defaults_to_empty(self):
        t = ring_table.RingTable(boot="b", instrument="i", lines_read=0)
        self.assertEqual(t.same_form_unusable, ())
        self.assertEqual(t.same_form_candidates, ())


if __name__ == "__main__":
    unittest.main()
