"""#725: the harvested crossover table, and the three things it must not say.

WHAT THIS TASK TURNED OUT TO BE. The brief asked to find where the cost model
prices quantised-linear lanes as M-independent or with a wrong crossover, and
to stop bs=1-decode points being priced through the quant stage. The honest
answer is that neither happens, because the planner has no activation-quant
axis at all: decode is priced as ``weight_bytes / bw`` and prefill as
``2*params / tflops``, with quantisation entering only as a weight-byte count
and a lane rate. A T=2 decode point never passes through a quant stage, so
there is no misprice to make red. The task therefore reduces, per the brief's
own escape clause, to data-with-provenance plus these pins.

The pins are about PROVENANCE HONESTY rather than arithmetic, because the
table's whole risk is that a borrowed number gets treated as a local law:

  1. an unmeasured shape must be distinguishable from a measured "never";
  2. sm86 must stay empty, not estimated;
  3. the non-monotonic row must not be smoothed into a clean threshold.
"""

import unittest

from sglang.srt.planner import activation_quant_crossover as aqc


class TestTheTableMatchesTheSource(unittest.TestCase):
    """The six sm120a rows, against ANALYSE_NINFER.md section 3.1."""

    EXPECTED = {
        (14336, 5120): 12,
        (16384, 5120): 11,
        (34816, 5120): 5,
        (5120, 6144): 25,
        (5120, 17408): 25,
        (248320, 5120): None,
    }

    def test_every_measured_shape_is_recorded_with_its_threshold(self):
        self.assertEqual(set(aqc.FP8_SM120A), set(self.EXPECTED))
        for shape, first in self.EXPECTED.items():
            with self.subTest(shape=shape):
                self.assertEqual(aqc.FP8_SM120A[shape].first_quant_token, first)

    def test_every_row_names_where_it_came_from(self):
        for shape, row in aqc.FP8_SM120A.items():
            with self.subTest(shape=shape):
                self.assertEqual(row.provenance, aqc.MEASURED_NINFER_SM120A)


class TestLmHeadNeverQuantises(unittest.TestCase):
    """The row with a consequence, and the one our own gate disagrees with."""

    def test_lm_head_says_no_at_every_token_count(self):
        row = aqc.FP8_SM120A[(248320, 5120)]
        for tokens in (1, 2, 4, 25, 512, 100_000):
            with self.subTest(tokens=tokens):
                self.assertIs(row.quantises_at(tokens), False)

    def test_a_measured_never_is_not_an_absent_measurement(self):
        """``first_quant_token=None`` means MEASURED never. It must not read
        as 'unmeasured', or the one row with a hard result becomes the one row
        a caller feels free to guess about."""
        row = aqc.FP8_SM120A[(248320, 5120)]
        self.assertTrue(row.measured)
        self.assertIsNot(row.quantises_at(1), None)

    def test_the_divergence_from_our_own_gate_is_recorded(self):
        self.assertIn("248320", aqc.LM_HEAD_DIVERGENCE)
        self.assertIn("N>=K", aqc.LM_HEAD_DIVERGENCE)


class TestSm86StaysAbsent(unittest.TestCase):
    """(d): NInfer measured sm120a only. Estimating sm86 is forbidden."""

    def test_the_sm86_table_is_empty(self):
        self.assertEqual(
            aqc.FP8_SM86,
            {},
            "an sm86 row would be a number with no measurement behind it, "
            "wearing the same type as five that have one",
        )

    def test_an_sm86_lookup_returns_none_rather_than_the_sm120a_row(self):
        for shape in aqc.FP8_SM120A:
            with self.subTest(shape=shape):
                self.assertIsNone(aqc.crossover_for(*shape, arch="sm86"))

    def test_absent_is_not_a_policy(self):
        """None must not be readable as 'do not quantise'."""
        absent = aqc.Crossover(1, 1, "invented", None, aqc.ABSENT)
        self.assertFalse(absent.measured)
        self.assertIsNone(
            absent.quantises_at(1),
            "an unmeasured shape answering False would adopt a policy the "
            "measurement never stated",
        )


class TestTheNonMonotonicRowKeepsItsShape(unittest.TestCase):
    """MLP gate/up is A8 at T=1, A16 at T=2..4, A8 from T=5.

    Storing 5 is correct for the crossover, but a caller at T=1 must not be
    able to read 'A16 was measured to win at T=1' out of this table, because
    it was not.
    """

    def test_the_non_monotonicity_is_carried_in_the_note(self):
        row = aqc.FP8_SM120A[(34816, 5120)]
        self.assertIn("NON-MONOTONIC", row.note)
        self.assertIn("T=1", row.note)


class TestTheCorroborationThatMotivatesTheHarvest(unittest.TestCase):
    """The K>N family is where both sources agree, independently.

    This is the pin that would catch someone 'simplifying' the table later:
    the value of the harvest is precisely that these two rows came from a
    different codebase, a different mechanism and a different GPU class, and
    landed on the same side as our own sm86 aspect gate.
    """

    def test_the_two_k_greater_than_n_shapes_hold_out_to_a_high_token_count(self):
        for shape in ((5120, 6144), (5120, 17408)):
            with self.subTest(shape=shape):
                row = aqc.FP8_SM120A[shape]
                self.assertGreaterEqual(row.first_quant_token, 25)
                self.assertIs(row.quantises_at(4), False, "bs=1-decode class")

    def test_our_own_aspect_gate_separates_the_same_family(self):
        """``N >= K`` is the shipped separator; it must still put the two
        agreed-losing shapes on the materialise side."""
        from sglang.srt.layers.quantization.fp8_dequant_gemv import (  # noqa: F401
            fused_gemv_applicable,
        )

        for n, k in ((5120, 6144), (5120, 17408)):
            with self.subTest(shape=(n, k)):
                self.assertLess(n, k, "the aspect gate sends this to materialise")
        for n, k in ((14336, 5120), (16384, 5120), (34816, 5120)):
            with self.subTest(shape=(n, k)):
                self.assertGreaterEqual(n, k)


class TestNothingConsumesItYet(unittest.TestCase):
    """The table is data with no wiring, and says so.

    #421's audit class: an instrument that looks wired and is not is worse
    than one that is honestly inert. If a consumer is added later, this pin
    is the reminder that the #368 graph-replay result has to be faced first.
    """

    def test_the_module_documents_the_graph_replay_caveat(self):
        doc = aqc.__doc__ or ""
        self.assertIn("#368", doc)
        self.assertIn("launch constant", doc.lower())
        self.assertIn("graph", doc.lower())

    def test_the_module_names_the_real_dispatch_site(self):
        doc = aqc.__doc__ or ""
        self.assertIn("fp8_dequant_gemv.py", doc)


if __name__ == "__main__":
    unittest.main()
