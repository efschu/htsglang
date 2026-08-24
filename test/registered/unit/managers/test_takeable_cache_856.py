"""#856 F6: the cache term's safety argument INVERTS between its two callers.

`CorridorAdmission._allocator_cache_bytes` is `reserved - allocated`, and its
docstring defends overstating as the safe direction -- correctly, WHERE IT
SIZES `want`: "overstating the cache understates `want`, and an understated
want costs a late arm".

`spendable_bytes` ADDS the same figure to a budget:

    return free + cache - delta

so there an overstatement WIDENS the prefill chunk this actuator grants, and
the one thing that function calls unsurvivable -- "an allocation larger than
what can be served" -- is exactly what it then permits. Same number, opposite
error, one of the two justified.

AND THE OVERSTATEMENT HAS A NAME (#852 R3). Free blocks inside a CUDA-graph
PRIVATE pool are counted by `reserved - allocated`, because those are
device-global `.all` counters, but an ordinary forward allocation cannot take
them: they are reachable only while capturing into that pool. W25 measured the
term at a stable 88 MiB.

The shared `_allocator_cache_bytes` is deliberately NOT changed -- that would
fix one caller by breaking the other's documented argument. Only the budget
caller subtracts.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.corridor_admission import takeable_cache_bytes
from sglang.test.test_utils import CustomTestCase

MIB = 1024 * 1024


class TestTheTrappedBytesAreSubtracted(CustomTestCase):
    def test_the_w25_shape(self):
        # 310 MiB cached, 88 MiB of it trapped in graph pools.
        self.assertEqual(takeable_cache_bytes(310 * MIB, 88 * MIB), (310 - 88) * MIB)

    def test_nothing_trapped_leaves_the_figure_alone(self):
        # THE CAN-FAIL DIRECTION: an implementation that always shrank the
        # cache would pass the test above and quietly narrow every chunk.
        self.assertEqual(takeable_cache_bytes(310 * MIB, 0), 310 * MIB)


class TestItAbstainsRatherThanGuesses(CustomTestCase):
    def test_no_verdict_returns_the_unadjusted_figure(self):
        # None means the segment view could not be read -- NOT "nothing
        # trapped". The caller must keep its pre-existing behaviour exactly.
        self.assertEqual(takeable_cache_bytes(310 * MIB, None), 310 * MIB)

    def test_an_unreadable_trapped_figure_does_not_poison_the_budget(self):
        for bad in ("n/a", object(), 1.5e400):
            with self.subTest(trapped=bad):
                self.assertEqual(takeable_cache_bytes(310 * MIB, bad), 310 * MIB)

    def test_an_unreadable_cache_is_zero_not_negative(self):
        self.assertEqual(takeable_cache_bytes("n/a", 88 * MIB), 0)


class TestItNeverReturnsANegativeBudget(CustomTestCase):
    """This value is ADDED to a free column; a negative would silently
    subtract from real memory."""

    def test_trapped_exceeding_cache_floors_at_zero(self):
        self.assertEqual(takeable_cache_bytes(50 * MIB, 88 * MIB), 0)

    def test_a_negative_trapped_reading_cannot_inflate_the_budget(self):
        # A counter disagreement must never make the budget LARGER than the
        # cache -- that is the direction that grants an unservable chunk.
        self.assertEqual(takeable_cache_bytes(310 * MIB, -100 * MIB), 310 * MIB)

    def test_a_negative_cache_floors_at_zero(self):
        self.assertEqual(takeable_cache_bytes(-5 * MIB, None), 0)


class TestTheSharedFigureIsNotChanged(CustomTestCase):
    """The other caller's documented argument must survive."""

    def test_allocator_cache_bytes_is_still_its_own_method(self):
        from sglang.srt.managers.corridor_admission import PrefillAdmissionGate

        self.assertTrue(hasattr(PrefillAdmissionGate, "_allocator_cache_bytes"))
        self.assertTrue(hasattr(PrefillAdmissionGate, "_takeable_cache_bytes"))
        self.assertIsNot(
            PrefillAdmissionGate._allocator_cache_bytes,
            PrefillAdmissionGate._takeable_cache_bytes,
        )

    def test_only_the_budget_caller_subtracts(self):
        # The shared figure must stay unadjusted, or the relief ladder's
        # documented "overstating is safe here" argument silently changes.
        import inspect

        from sglang.srt.managers.corridor_admission import PrefillAdmissionGate

        shared = inspect.getsource(PrefillAdmissionGate._allocator_cache_bytes)
        self.assertNotIn("graph_pool", shared)
        self.assertNotIn("takeable", shared)
        budget = inspect.getsource(PrefillAdmissionGate.spendable_bytes)
        self.assertIn("_takeable_cache_bytes", budget)


class TestTheWiringIsLiveAndInertInTheRightPlaces(CustomTestCase):
    """The method around the pure function, pinned both ways.

    A rule that can only be exercised on metal is one this corpus has
    repeatedly shipped inert, so the probe is injectable and BOTH outcomes are
    asserted here: no probe and no CUDA must be byte-identical to the old
    behaviour, and an injected probe must actually subtract.
    """

    @staticmethod
    def _gate(cache_mib, probe=None):
        import types

        stub = types.SimpleNamespace(
            _allocator_cache_bytes=lambda: cache_mib * MIB,
        )
        if probe is not None:
            stub._graph_pool_free_probe = probe
        return stub

    def test_without_cuda_it_is_byte_identical_to_the_old_figure(self):
        # Under CUDA_VISIBLE_DEVICES="" there is no device, so the segment
        # view cannot be read and the caller must keep exactly its previous
        # budget. This is what makes the change safe to land unvalidated.
        from sglang.srt.managers.corridor_admission import PrefillAdmissionGate

        got = PrefillAdmissionGate._takeable_cache_bytes(self._gate(310))
        self.assertEqual(got, 310 * MIB)

    def test_an_injected_probe_actually_subtracts(self):
        # THE CAN-FAIL PARTNER of the test above: without this, "inert under
        # CVD=''" would be indistinguishable from "inert everywhere".
        from sglang.srt.managers.corridor_admission import PrefillAdmissionGate

        got = PrefillAdmissionGate._takeable_cache_bytes(
            self._gate(310, probe=lambda: 88 * MIB)
        )
        self.assertEqual(got, 222 * MIB)

    def test_a_raising_probe_falls_back_to_the_unadjusted_figure(self):
        # This sizes a prefill budget on the admission path; a probe that
        # throws may cost precision, never an admission.
        from sglang.srt.managers.corridor_admission import PrefillAdmissionGate

        def _boom():
            raise RuntimeError("snapshot exploded")

        got = PrefillAdmissionGate._takeable_cache_bytes(self._gate(310, probe=_boom))
        self.assertEqual(got, 310 * MIB)


if __name__ == "__main__":
    unittest.main()
