# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Every speaker assignment must leave a record of WHY.

``speakers.py`` logged nothing at all. That is why the user's report -- "I am
recognized as somebody different every time" -- could only be answered by
reproducing it: the thresholds in force (``match_threshold`` 0.637,
``uncertain_floor`` 0.583, DESIGN §17.8.9) were measured on a synthetic
17-voice pool, and there was no way to check them against a real conversation
after the fact.

What this file pins is the property that makes the record usable for
calibration rather than merely present:

* the FULL candidate ranking, not only the winner -- the interesting decisions
  are the ones where the runner-up was close, and a record naming only the
  winner leaves "why" exactly as unanswerable as no record;
* the cosines as the decision SAW them, taken before any centroid folds;
* the branch that fired, so match / continuity-guard / capacity / mint are
  distinguishable without re-deriving them from the score;
* the thresholds in force at the time, so a series stays interpretable across
  a retune;
* manual attributions, which are the only decisions where the correct answer
  is known, recorded together with what the automatic path WOULD have said.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_speaker_decision_log.py -v
"""

import unittest

import numpy as np

from sglang.srt.translator.backends import AudioChunk, SpeakerEmbedding
from sglang.srt.translator.speakers import SpeakerRegistry, SpeakerRegistryConfig


def unit(*components):
    vector = np.array(components, dtype=np.float32)
    return SpeakerEmbedding(vector / np.linalg.norm(vector))


def audio(seconds=3.0, rate=16000, amplitude=0.2):
    n = int(seconds * rate)
    t = np.arange(n, dtype=np.float32) / rate
    return AudioChunk((amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32),
                      rate)


#: Two clearly different voices, and one that sits deliberately between them.
VOICE_A = unit(1.0, 0.0, 0.0)
VOICE_B = unit(0.0, 1.0, 0.0)


def at_cosine(base, other, cosine):
    """A unit vector at EXACTLY `cosine` from `base`, in the base/other plane.

    Spelled out rather than interpolated: linear mixing of two unit vectors
    does not give the cosine of the mixing weight, and the thresholds under
    test here are 0.054 apart. A test that lands in the wrong band tests the
    wrong branch.
    """
    orthogonal = other.vector - float(np.dot(other.vector, base.vector)) * base.vector
    orthogonal = orthogonal / np.linalg.norm(orthogonal)
    vector = cosine * base.vector + float(np.sqrt(1.0 - cosine ** 2)) * orthogonal
    return SpeakerEmbedding((vector / np.linalg.norm(vector)).astype(np.float32))


class TestTheDecisionIsRecorded(unittest.TestCase):
    def setUp(self):
        self.clock = [0.0]
        self.registry = SpeakerRegistry(
            SpeakerRegistryConfig(), clock=lambda: self.clock[0]
        )

    def test_a_mint_is_recorded(self):
        self.registry.assign(VOICE_A, audio(), language="de")
        self.assertEqual(len(self.registry.decisions), 1)
        record = self.registry.decisions[-1]
        self.assertEqual(record["outcome"], "mint")
        self.assertEqual(record["nearest_id"], None)
        self.assertEqual(record["language"], "de")
        self.assertEqual(record["match_threshold"], 0.637)
        self.assertEqual(record["uncertain_floor"], 0.583)

    def test_a_confident_match_records_the_score_it_decided_on(self):
        profile, _, _ = self.registry.assign(VOICE_A, audio())
        self.clock[0] = 10.0
        _, score, _ = self.registry.assign(VOICE_A, audio())
        record = self.registry.decisions[-1]
        self.assertEqual(record["outcome"], "match")
        self.assertEqual(record["speaker_id"], profile.speaker_id)
        self.assertAlmostEqual(record["score"], round(score, 4), places=4)
        self.assertGreaterEqual(record["score"], 0.637)

    def test_the_candidates_are_the_pre_fold_cosines(self):
        """A match folds the centroid; ranking after it records fiction."""
        self.registry.assign(VOICE_A, audio())
        self.clock[0] = 10.0
        _, score, _ = self.registry.assign(VOICE_A, audio())
        record = self.registry.decisions[-1]
        self.assertTrue(record["candidates"], "the ranking must be kept")
        top_id, top_sim = record["candidates"][0]
        self.assertEqual(top_id, record["speaker_id"])
        self.assertAlmostEqual(
            top_sim, round(score, 4), places=4,
            msg="the logged cosine (%r) is not the one the decision used (%r); "
                "it was taken after the centroid moved" % (top_sim, score),
        )

    def test_the_continuity_guard_is_distinguishable(self):
        """A guard and a match must not look the same in the record."""
        self.registry.assign(VOICE_A, audio())
        self.clock[0] = 10.0
        # Between uncertain_floor (0.583) and match_threshold (0.637).
        borderline = at_cosine(VOICE_A, VOICE_B, 0.61)
        _, score, admitted = self.registry.assign(borderline, audio())
        self.assertGreaterEqual(score, 0.583)
        self.assertLess(score, 0.637)
        record = self.registry.decisions[-1]
        self.assertEqual(record["outcome"], "guard")
        self.assertTrue(record["guessed"])
        self.assertFalse(record["admitted_reference"])
        self.assertFalse(admitted, "a guess must never become the voice")

    def test_every_decision_carries_the_prior_slot(self):
        """§19.7 is unbuilt; the slot keeps the series one series."""
        self.registry.assign(VOICE_A, audio())
        self.assertIn("prior", self.registry.decisions[-1])
        self.assertIsNone(self.registry.decisions[-1]["prior"])

    def test_the_ring_is_bounded(self):
        for i in range(SpeakerRegistry.DECISION_LOG + 25):
            self.clock[0] = float(i)
            self.registry.assign(VOICE_A, audio())
        self.assertEqual(
            len(self.registry.decisions), SpeakerRegistry.DECISION_LOG,
            "a conversation runs for hours; the record must not grow forever",
        )

    def test_decisions_json_returns_the_newest(self):
        for i in range(5):
            self.clock[0] = float(i)
            self.registry.assign(VOICE_A, audio())
        newest = self.registry.decisions_json(2)
        self.assertEqual(len(newest), 2)
        self.assertEqual(newest[-1], self.registry.decisions[-1])


class TestManualGroundTruthIsRecorded(unittest.TestCase):
    def setUp(self):
        self.clock = [0.0]
        self.registry = SpeakerRegistry(
            SpeakerRegistryConfig(), clock=lambda: self.clock[0]
        )

    def test_a_manual_attribution_records_what_automatic_would_have_said(self):
        first, _, _ = self.registry.assign(VOICE_A, audio())
        declared = self.registry.create_speaker(label="wife")
        self.clock[0] = 10.0
        # Audio that the automatic path would have handed to `first`, but the
        # user says it is the declared speaker. That disagreement is the whole
        # value of the record.
        self.registry.assign_manual(
            declared.speaker_id, audio(), embedding=at_cosine(VOICE_A, VOICE_B, 0.95)
        )
        record = self.registry.decisions[-1]
        self.assertEqual(record["outcome"], "manual")
        self.assertEqual(record["speaker_id"], declared.speaker_id)
        self.assertEqual(
            record["nearest_id"], first.speaker_id,
            "the automatic answer must be kept next to the human one, or the "
            "record cannot tell a correction from an agreement",
        )
        self.assertGreater(record["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
