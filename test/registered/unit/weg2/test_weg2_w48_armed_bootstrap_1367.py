# SPDX-License-Identifier: Apache-2.0
"""#1367 -- the FOURTH instance of "a form change devalues the inherited
record", and the first one where the refusal's own remedy was unreachable.

WHAT HAPPENED (xsn30 dry run, rc=2 on all three groups, boot seat 2026-09-13):
the armed arm puts XCHG_FORM_TOKEN into group P's argv, so the armed boot's
form key (0a15b55459fb) can never equal an unarmed boot's (2b66740bedf9).
`refuse_unless_same_form_source` (launcher.py:3766) is called unconditionally
at launcher.py:10244 and knows two outcomes, not-armed and form_same, so the
first armed boot of a form is refused -- and the refusal ends with "Boot the
arm once to produce the first same-form source", which is the very boot it
just refused. A remedy that names the refused boot is circular; the class is
the one #1362 closed for the model digest and the lane cut: ABSENCE IS A
STATE, NOT A FAULT.

THE DECISION IS THE OPERATOR'S (2026-09-13) AND IS NARROW: no rebuild of the
form key -- the exchange arm IS a different weight statement and must keep its
own key. When the arm is armed and there are ZERO same-form candidates, the
first boot goes through the THIRD W48 shape that already exists for the
default arm: re-derivation from this boot's own argv (ring_table.py:2737), the
exchange form's ring residual coming from the published terms, printed as
source=derived. The refusal STAYS wherever a same-form source exists and
contradicts -- that case is a real mismatch, not a first boot.
"""

import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher, ring_table
from sglang.test.test_utils import CustomTestCase

ARMED = sorted(launcher.WEIGHT_SOURCE_ARMED)[0]


def _table(*, form_same, same_form_candidates=(), boot="boot_weg2_xsn29"):
    return ring_table.RingTable(
        boot=boot, instrument="test", lines_read=1,
        form_key="0a15b55459fb", source_form_key="2b66740bedf9",
        form_same=form_same, form_diff="(diff)",
        same_form_candidates=tuple(same_form_candidates),
    )


class TheFirstArmedBootIsAStateNotAFault(CustomTestCase):
    def test_armed_with_no_same_form_candidate_does_not_refuse(self):
        line = launcher.refuse_unless_same_form_source(
            _table(form_same=False), ARMED)
        self.assertIsNotNone(line, "the bootstrap must be SAID, not silent")

    def test_the_bootstrap_line_names_itself_by_name(self):
        line = launcher.refuse_unless_same_form_source(
            _table(form_same=False), ARMED)
        self.assertIn("W48", line)
        self.assertIn("BOOTSTRAP armed first boot", line)
        self.assertIn("source=derived", line)
        self.assertIn("0a15b55459fb", line, "the form being bootstrapped")

    def test_a_same_form_source_that_contradicts_still_refuses(self):
        """The refusal keeps its whole subject: a source of the SAME form that
        did not make it is a mismatch, and mismatches still stop the boot."""
        with self.assertRaises(ring_table.Weg2RingFormMismatch) as cm:
            launcher.refuse_unless_same_form_source(
                _table(form_same=False,
                       same_form_candidates=("boot_weg2_xsn28",)), ARMED)
        msg = str(cm.exception)
        self.assertIn("W48", msg)
        self.assertIn("boot_weg2_xsn28", msg,
                      "the refusal must name the same-form source it had")

    def test_the_circular_remedy_is_gone_from_the_bootstrap_path(self):
        """The sentence that sent the reader back to the boot it refused must
        not survive on the path where that boot is the one being refused."""
        line = launcher.refuse_unless_same_form_source(
            _table(form_same=False), ARMED)
        self.assertNotIn("Boot the arm once", line)

    def test_a_matching_form_is_still_silent(self):
        self.assertIsNone(launcher.refuse_unless_same_form_source(
            _table(form_same=True), ARMED))

    def test_the_unarmed_gate_still_refuses(self):
        """form_same None means the gate never ran; an unarmed gate is never
        read as a passed one, and a first boot cannot be claimed either."""
        with self.assertRaises(ring_table.Weg2RingFormMismatch):
            launcher.refuse_unless_same_form_source(_table(form_same=None), ARMED)

    def test_the_default_arm_is_untouched(self):
        for form_same in (True, False, None):
            self.assertIsNone(launcher.refuse_unless_same_form_source(
                _table(form_same=form_same), "ring"))

    def test_a_missing_table_is_still_someone_elses_refusal(self):
        self.assertIsNone(
            launcher.refuse_unless_same_form_source(None, ARMED))


class TheInventoryIsMeasuredNotAssumed(CustomTestCase):
    def test_solve_publishes_the_same_form_candidates_that_SURVIVED(self):
        """`solve` already builds the ranked list (ring_table.py:2679); the
        launcher's decision needs the SURVIVING subset of it, not a second
        count and not the ranked list.

        #1374 W9 narrowed this deliberately: a stem that was ranked same-form
        and then ELIMINATED (weg2xsn30 died at wall 8 before D slept, so it
        carries `no sleep-pass lines for D`) is not a witness, and counting it
        refused weg2xsn31 -- every boot that dies at a wall would lock its
        successor out. The split is derived in ONE place from the reasons
        `solve` already writes."""
        src = inspect.getsource(ring_table.solve)
        self.assertIn("same_form_candidates=tuple(", src)
        self.assertIn("same_form_unusable=tuple(", src)
        self.assertIn('r.startswith(f"{st}:")', src,
                      "the survivors must be derived from the elimination "
                      "reasons, not from a second bookkeeping")

    def test_the_field_defaults_to_empty_not_to_none(self):
        """An empty inventory is the bootstrap state; None would be a third
        value nobody decided."""
        t = ring_table.RingTable(boot="b", instrument="i", lines_read=0)
        self.assertEqual(t.same_form_candidates, ())


class TheCallsiteConsumesIt(CustomTestCase):
    def test_the_launcher_callsite_assigns_the_line_and_files_it(self):
        """STRUCTURAL half: the call's result must be bound and must reach the
        ring evidence. A callsite that keeps calling it for the raise alone
        would pass every behavioural test above and still lose the line."""
        import ast

        tree = ast.parse(inspect.getsource(launcher))
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Call)
                 and getattr(n.value.func, "id", "")
                 == "refuse_unless_same_form_source"]
        self.assertEqual(len(calls), 1,
                         "the callsite must bind the result exactly once")
        src = inspect.getsource(launcher)
        i = src.index("= refuse_unless_same_form_source(ring_plan.table")
        window = src[i:i + 400]
        self.assertIn("ring_plan.lines.append", window)
        self.assertIn("log(", window)

    def test_the_callsite_sequence_executed_on_both_form_keys(self):
        """EXECUTION half: the three statements the callsite performs, run
        against a fake ring_plan for both keys -- the armed first boot files a
        line and starts, the contradicting one raises before either group."""
        class _Plan:
            def __init__(self, table):
                self.table, self.lines = table, []

        logged = []

        def _callsite(plan):
            line = launcher.refuse_unless_same_form_source(plan.table, ARMED)
            if line:
                logged.append(line)
                plan.lines.append(line)

        first = _Plan(_table(form_same=False))          # key 0a15b55459fb, no source
        _callsite(first)
        self.assertEqual(len(first.lines), 1)
        self.assertIn("BOOTSTRAP armed first boot", first.lines[0])
        self.assertEqual(logged, first.lines, "the line must also be logged")

        inherited = _Plan(_table(form_same=True))        # a same-form table
        _callsite(inherited)
        self.assertEqual(inherited.lines, [], "nothing to say when it matches")

        contradicting = _Plan(_table(form_same=False,
                                     same_form_candidates=("boot_weg2_xsn28",)))
        with self.assertRaises(ring_table.Weg2RingFormMismatch):
            _callsite(contradicting)
        self.assertEqual(contradicting.lines, [],
                         "a refused boot files no bootstrap line")

    def test_the_two_form_keys_really_cannot_meet(self):
        """The premise, executed: the armed argv and the unarmed argv of the
        same boot hash to different keys, because the token is IN the key."""
        base = ["--model-path", "/m", "--tp-size", "3"]
        k_plain, _f = ring_table.p_form_key(base)
        k_armed, f_armed = ring_table.p_form_key(
            base + [f"--weg2-xchg-region={ring_table.XCHG_FORM_TOKEN}"]
            if not ring_table.XCHG_FORM_TOKEN.startswith("--")
            else base + [ring_table.XCHG_FORM_TOKEN])
        self.assertNotEqual(k_plain, k_armed)
        self.assertIn(ring_table.XCHG_FORM_TOKEN, f_armed)


if __name__ == "__main__":
    unittest.main()
