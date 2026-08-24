"""#850: a presence-withhold reason no armed service turn can clear.

#800 proved the shape: a rank withholds presence on a message whose only
consumer is the PP loop body, and the presence gate is what stops that body
from running. It fixed the tensor-dict instance it measured. This file guards
the generalisation -- that EVERY reason ``pp_flip_channels_empty`` can produce
is classified as either clearable by an armed service turn or not, and that a
reason nobody can clear stops the wait instead of burning the presence
deadline.

THE MUTATION GUARD, and it is the reason this file exists rather than a
handful of table assertions. ``test_marker_table_covers_every_produced_reason``
reads the reason literals OUT OF ``pp_flip_channels_empty``'s own source with
``ast`` and asserts each one classifies. A future fifth reason -- the exact way
#800's seam was created, by adding a third kind to a wire whose gate knew two
-- then fails a test at desk time instead of reaching metal as an unbounded,
unnamed silence. Deleting a marker fails it too, in the other direction.

The classification itself is asserted against the four actuators that
``pp_flip_service`` actually holds, so "self-clearing" means a named function
clears it, not an opinion.

Hermetic: pure functions, no scheduler, no process group, no CUDA.
"""

import ast
import inspect
import unittest

from sglang.srt.managers.pp_presence_disposition import (
    CONSUMER_EXCLUDED,
    SELF_CLEARING,
    UNCLASSIFIED,
    census_withhold_reason,
    classify_withhold_clause,
    withhold_markers,
)
from sglang.srt.managers.pp_stash_disposition import census_stash


class TestWithholdClassification(unittest.TestCase):
    """One clause at a time, against the actuator that would clear it."""

    def test_request_chain_inbox_is_consumer_excluded(self):
        """THE #850 DEFECT ITSELF, measured 11 times across 291 boot logs.

        ``pp_flip_consume_inbound`` moves the message off the WIRE and into
        ``receiver.inbox``; only ``_pull_raw_reqs`` at the top of a PP pass
        pops that inbox, and the presence gate excludes the PP pass. So the
        armed loop converts a clearable reason into an unclearable one.
        """
        klass, clearer = classify_withhold_clause(
            "request-chain inbox holds 1 unhandled message(s)"
        )
        self.assertEqual(klass, CONSUMER_EXCLUDED)
        self.assertIn("_pull_raw_reqs", clearer)

    def test_request_chain_wire_is_self_clearing(self):
        """The WIRE, unlike the inbox, is exactly what the service turn drains.

        These two clauses differ by one word and by their whole disposition;
        classifying them together would make the table useless.
        """
        klass, clearer = classify_withhold_clause(
            "request chain has 2 unconsumed message(s) from rank 0"
        )
        self.assertEqual(klass, SELF_CLEARING)
        self.assertIn("pp_flip_consume_inbound", clearer)

    def test_unreaped_sends_are_self_clearing(self):
        """609 of the 823 measured withholds. Healthy, transient, reaped."""
        klass, clearer = classify_withhold_clause("send_output_work is not reaped")
        self.assertEqual(klass, SELF_CLEARING)
        self.assertIn("pp_flip_flush_drained_sends", clearer)

    def test_undeclared_stash_is_self_clearing_on_800s_clock(self):
        """#800 gave this one an escape clock, so waiting DOES terminate it.

        Ordered before the general ``tensor-dict inbox holds`` marker, which is
        why the table's order is load-bearing.
        """
        reason = census_stash({(0, "mystery_kind"): [object()]}).block_reason()
        klass, clearer = classify_withhold_clause(reason)
        self.assertEqual(klass, SELF_CLEARING)
        self.assertIn("escape clock", clearer)

    def test_blocking_stash_is_consumer_excluded(self):
        """An owed ``output`` blocks correctly -- and no armed turn clears it.

        #800 keeps it blocking on purpose (dropping it drops a token). That
        makes the wait CORRECT and the abandonment INEVITABLE, which is
        precisely when the deadline is pure loss.
        """
        reason = census_stash({(0, "output"): [object()]}).block_reason()
        klass, clearer = classify_withhold_clause(reason)
        self.assertEqual(klass, CONSUMER_EXCLUDED)
        self.assertIn("cutover", clearer)

    def test_unknown_clause_is_unclassified_not_futile(self):
        klass, clearer = classify_withhold_clause("something nobody has named yet")
        self.assertEqual(klass, UNCLASSIFIED)
        self.assertIsNone(clearer)


class TestWithholdCensus(unittest.TestCase):
    """The verdict over a whole joined reason string."""

    def test_all_futile_clauses_yield_a_futile_verdict(self):
        census = census_withhold_reason(
            "request-chain inbox holds 1 unhandled message(s); "
            "pp_outputs holds a received-but-unprocessed output "
            "(its sampled token has reached no output_ids yet)"
        )
        self.assertTrue(census.is_futile)
        self.assertEqual(len(census.futile), 2)
        self.assertIn("_pull_raw_reqs", census.futile_reason())

    def test_one_self_clearing_clause_suppresses_the_verdict(self):
        """THE SAFETY DIRECTION, and the mutant this test kills.

        A rank whose reason mixes an unclearable clause with a clause a
        service turn still fixes must KEEP WAITING: the self-clearing half may
        be the only thing standing between this flip and a correct commit.
        Abandoning on "any futile clause" instead of "every clause futile"
        would turn a healthy transient withhold into a refused flip.
        """
        census = census_withhold_reason(
            "request-chain inbox holds 1 unhandled message(s); "
            "send_output_work is not reaped"
        )
        self.assertFalse(census.is_futile)
        self.assertEqual(len(census.futile), 1)
        self.assertEqual(len(census.self_clearing), 1)

    def test_unclassified_clause_suppresses_the_verdict(self):
        """Never abandon a flip on a sentence this module cannot read."""
        census = census_withhold_reason(
            "request-chain inbox holds 1 unhandled message(s); "
            "a brand new reason from a future commit"
        )
        self.assertFalse(census.is_futile)
        self.assertEqual(census.unclassified, ("a brand new reason from a future commit",))

    def test_empty_reason_is_not_futile(self):
        for empty in (None, "", "   "):
            census = census_withhold_reason(empty)
            self.assertFalse(census.is_futile)
            self.assertIsNone(census.futile_reason())


class TestMarkerTableExhaustiveness(unittest.TestCase):
    """The pin. A reason that reaches metal unclassified fails here first."""

    @staticmethod
    def _reason_literals_from_source():
        """Every string ``pp_flip_channels_empty`` appends to ``reasons``.

        Read from the real source with ``ast`` rather than restated here, so
        this pin cannot drift away from the function it guards. f-strings are
        reduced to their constant segments, which is where every marker lives.
        """
        from sglang.srt.managers import scheduler_pp_mixin

        tree = ast.parse(inspect.getsource(scheduler_pp_mixin))
        target = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "pp_flip_channels_empty":
                target = node
                break
        assert target is not None, "pp_flip_channels_empty not found in source"

        literals = []
        for node in ast.walk(target):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "append"):
                continue
            if not (isinstance(func.value, ast.Name) and func.value.id == "reasons"):
                continue
            for arg in node.args:
                text = None
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    text = arg.value
                elif isinstance(arg, ast.JoinedStr):
                    text = "".join(
                        v.value
                        for v in arg.values
                        if isinstance(v, ast.Constant) and isinstance(v.value, str)
                    )
                if text and text.strip():
                    literals.append(text)
        return literals

    def test_the_pin_actually_finds_the_reasons(self):
        """CAN-FAIL PROOF. An empty scrape would make the next test vacuous."""
        literals = self._reason_literals_from_source()
        self.assertGreaterEqual(len(literals), 5, literals)
        joined = " || ".join(literals)
        self.assertIn("request-chain inbox holds", joined)
        self.assertIn("is not reaped", joined)

    def test_marker_table_covers_every_produced_reason(self):
        """Every literal the producer emits must classify. No UNCLASSIFIED."""
        unmatched = []
        for literal in self._reason_literals_from_source():
            klass, _ = classify_withhold_clause(literal)
            if klass == UNCLASSIFIED:
                unmatched.append(literal)
        self.assertEqual(
            unmatched,
            [],
            "pp_flip_channels_empty produces reason(s) with no entry in "
            "pp_presence_disposition._MARKERS. Classify them: a reason with no "
            "disposition is the #800 seam, reopened.",
        )

    def test_stash_census_reasons_also_classify(self):
        """#800's census reasons are built elsewhere, so pin them separately."""
        for inbox in (
            {(0, "output"): [object()]},
            {(0, "proxy"): [object()]},
            {(0, "unknown_future_kind"): [object()]},
        ):
            reason = census_stash(inbox).block_reason()
            self.assertIsNotNone(reason)
            klass, _ = classify_withhold_clause(reason)
            self.assertNotEqual(klass, UNCLASSIFIED, reason)

    def test_every_marker_declares_a_clearer(self):
        for marker, klass, clearer in withhold_markers():
            self.assertIn(klass, (SELF_CLEARING, CONSUMER_EXCLUDED), marker)
            self.assertTrue(clearer and clearer.strip(), marker)


if __name__ == "__main__":
    unittest.main()
