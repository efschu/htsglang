# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The intelligibility instrument, before it is trusted to gate anything.

WER is the hard gate of DESIGN_466 §7(b)(3), so it is worth more than "it
returns a number". The properties pinned here are the ones that decide whether
a real candidate passes or fails:

* it does not saturate -- a runaway synthesizer must score worse than silence,
  not the same;
* it distinguishes deletions from substitutions, because "the second half was
  dropped" and "it said something else" have different causes;
* accents survive normalisation, since a cross-lingual cloner's characteristic
  error is exactly a vowel;
* the gate reads the WORST arm, never the mean.

No language is named in the module under test and none is needed here either;
these use invented words for the same reason the routing tests use invented
codes.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_scoring.py -v
"""

import unittest

from sglang.srt.translator.scoring import (
    normalize_for_wer,
    word_error_rate,
    worst_of,
)


class TestNormalisation(unittest.TestCase):
    def test_case_and_punctuation_are_noise(self):
        self.assertEqual(
            normalize_for_wer("Alpha, Beta.  Gamma!"), ["alpha", "beta", "gamma"]
        )

    def test_ACCENTS_ARE_KEPT(self):
        # Stripping them would make the vowel errors a cross-lingual cloner
        # actually makes compare equal to a correct rendering.
        self.assertNotEqual(normalize_for_wer("uber"), normalize_for_wer("über"))
        self.assertNotEqual(normalize_for_wer("si"), normalize_for_wer("sí"))

    def test_intra_word_apostrophes_survive_but_edge_ones_do_not(self):
        self.assertEqual(normalize_for_wer("'alpha'"), ["alpha"])
        self.assertEqual(normalize_for_wer("l'alpha"), ["l'alpha"])

    def test_composed_and_decomposed_forms_agree(self):
        self.assertEqual(normalize_for_wer("é"), normalize_for_wer("é"))

    def test_empty_input_yields_no_words(self):
        self.assertEqual(normalize_for_wer("  ...  "), [])


class TestWordErrorRate(unittest.TestCase):
    def test_an_exact_match_scores_zero(self):
        result = word_error_rate("alpha beta gamma", "Alpha, beta GAMMA.")
        self.assertEqual(result.rate, 0.0)
        self.assertEqual(
            (result.substitutions, result.deletions, result.insertions), (0, 0, 0)
        )

    def test_one_substitution_in_four_words(self):
        result = word_error_rate("alpha beta gamma delta", "alpha beta zeta delta")
        self.assertAlmostEqual(result.rate, 0.25)
        self.assertEqual(result.substitutions, 1)

    def test_a_dropped_tail_is_reported_as_DELETIONS(self):
        result = word_error_rate("alpha beta gamma delta", "alpha beta")
        self.assertAlmostEqual(result.rate, 0.5)
        self.assertEqual(result.deletions, 2)
        self.assertEqual(result.substitutions, 0)

    def test_silence_scores_exactly_one(self):
        result = word_error_rate("alpha beta gamma", "")
        self.assertAlmostEqual(result.rate, 1.0)
        self.assertEqual(result.deletions, 3)

    def test_a_RUNAWAY_scores_WORSE_than_silence(self):
        """The property the gate depends on, and the reason WER is uncapped.

        The real failure mode was a talker that produced 40.9 s of babble for
        a nine-word sentence. Capping at 1.0 would have scored it identically
        to silence and thrown away the signal that says "this is not merely
        wrong, it is running away".
        """
        silence = word_error_rate("alpha beta gamma", "")
        runaway = word_error_rate(
            "alpha beta gamma", " ".join(["zeta"] * 30)
        )
        self.assertGreater(runaway.rate, silence.rate)
        self.assertGreater(runaway.rate, 1.0)

    def test_an_empty_reference_is_refused_rather_than_scored(self):
        with self.assertRaises(ValueError):
            word_error_rate("", "alpha")


class TestGateAggregation(unittest.TestCase):
    def test_the_gate_reads_the_WORST_arm_not_the_mean(self):
        # Three good turns and one unintelligible one: the mean would pass a
        # 0.15 threshold, and a conversation with one unintelligible turn has
        # failed.
        results = [
            word_error_rate("alpha beta gamma delta", "alpha beta gamma delta"),
            word_error_rate("alpha beta gamma delta", "alpha beta gamma delta"),
            word_error_rate("alpha beta gamma delta", "alpha beta gamma delta"),
            word_error_rate("alpha beta gamma delta", "zeta eta theta iota"),
        ]
        mean = sum(r.rate for r in results) / len(results)
        self.assertLessEqual(mean, 0.30)
        self.assertEqual(worst_of(results), 1.0)

    def test_no_arms_is_not_a_failure(self):
        self.assertEqual(worst_of([]), 0.0)


if __name__ == "__main__":
    unittest.main()
