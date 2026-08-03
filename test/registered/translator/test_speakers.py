# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Speaker assignment and the reference buffer that feeds voice cloning.

The reference buffer is the direct cause of output quality and the thing that
is corrupted silently when diarization is wrong, so its policy is tested as
policy: what gets admitted, what gets evicted, and what is protected.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_speakers.py -v
"""

import math
import unittest

import numpy as np

from sglang.srt.translator.backends import AudioChunk, SpeakerEmbedding
from sglang.srt.translator.speakers import (
    ReferenceTooShort,
    SpeakerRegistry,
    SpeakerRegistryConfig,
    rolling_reference_from_segments,
    split_points_by_dispersion,
)

RATE = 16000


def _vec(angle_deg, dim=8):
    """A unit vector at a controlled angle from the first axis.

    Lets a test state the cosine similarity it wants directly, instead of
    hoping two hand-written vectors happen to land near a threshold.
    """
    angle = math.radians(angle_deg)
    v = np.zeros(dim, dtype=np.float32)
    v[0] = math.cos(angle)
    v[1] = math.sin(angle)
    return SpeakerEmbedding(v)


def _speech(seconds, amplitude=0.2, rate=RATE):
    n = int(seconds * rate)
    rng = np.random.default_rng(int(seconds * 1000) + int(amplitude * 1000))
    return AudioChunk((rng.standard_normal(n) * amplitude).astype(np.float32), rate)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


class TestEmbeddingContract(unittest.TestCase):
    def test_embeddings_are_normalised_on_construction(self):
        e = SpeakerEmbedding(np.array([3.0, 4.0], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(e.vector)), 1.0, places=6)
        self.assertAlmostEqual(e.similarity(e), 1.0, places=6)

    def test_a_degenerate_vector_is_refused(self):
        with self.assertRaises(ValueError):
            SpeakerEmbedding(np.zeros(4, dtype=np.float32))

    def test_dimension_mismatch_is_refused_rather_than_broadcast(self):
        with self.assertRaises(ValueError):
            _vec(0, dim=8).similarity(_vec(0, dim=16))


class TestAssignment(unittest.TestCase):
    def test_the_first_segment_mints_a_speaker(self):
        registry = SpeakerRegistry(clock=_Clock())
        profile, sim, admitted = registry.assign(_vec(0), _speech(3.0), "hola", "xx")
        self.assertEqual(len(registry), 1)
        self.assertEqual(sim, 1.0)
        self.assertTrue(admitted)
        self.assertEqual(profile.observations, 1)

    def test_a_similar_voice_joins_the_same_speaker(self):
        registry = SpeakerRegistry(clock=_Clock())
        registry.assign(_vec(0), _speech(3.0), "a", "xx")
        profile, sim, _ = registry.assign(_vec(10), _speech(3.0), "b", "xx")
        self.assertEqual(len(registry), 1)
        self.assertGreater(sim, 0.9)
        self.assertEqual(profile.observations, 2)

    def test_a_distant_voice_mints_a_second_speaker(self):
        registry = SpeakerRegistry(clock=_Clock())
        registry.assign(_vec(0), _speech(3.0), "a", "xx")
        registry.assign(_vec(80), _speech(3.0), "b", "yy")
        self.assertEqual(len(registry), 2)

    def test_an_ambiguous_segment_is_assigned_but_barred_from_the_reference(self):
        # Between match_threshold (0.70, ~45 deg) and reference_threshold
        # (0.80, ~37 deg): good enough to translate, not good enough to clone.
        registry = SpeakerRegistry(clock=_Clock())
        registry.assign(_vec(0), _speech(3.0), "a", "xx")
        before = registry.profiles()[0].reference_seconds()
        profile, sim, admitted = registry.assign(_vec(41), _speech(3.0), "b", "xx")
        self.assertEqual(len(registry), 1)
        self.assertTrue(0.70 <= sim < 0.80, sim)
        self.assertFalse(admitted, "an ambiguous segment must not poison the voice")
        self.assertEqual(profile.reference_seconds(), before)

    def test_the_speaker_cap_stops_minting_but_keeps_translating(self):
        registry = SpeakerRegistry(
            SpeakerRegistryConfig(max_speakers=2), clock=_Clock()
        )
        registry.assign(_vec(0), _speech(3.0), "a", "xx")
        registry.assign(_vec(90), _speech(3.0), "b", "yy")
        profile, sim, admitted = registry.assign(_vec(45), _speech(3.0), "c", "zz")
        self.assertEqual(len(registry), 2, "the cap must hold")
        self.assertIsNotNone(profile, "the turn must still get a speaker")
        self.assertFalse(admitted, "a crowded assignment must not touch a voice")

    def test_language_history_is_tracked_only_when_confident(self):
        registry = SpeakerRegistry(clock=_Clock())
        profile, _, _ = registry.assign(_vec(0), _speech(3.0), "a", "xx", 0.9)
        self.assertEqual(profile.last_language, "xx")
        registry.assign(_vec(2), _speech(3.0), "b", "yy", 0.1)
        self.assertEqual(
            profile.last_language,
            "xx",
            "an unconfident identification must not overwrite the history it "
            "is supposed to be corrected by",
        )


class TestReferenceBufferPolicy(unittest.TestCase):
    def test_a_slice_below_the_floor_is_not_admitted(self):
        registry = SpeakerRegistry(
            SpeakerRegistryConfig(min_slice_s=2.0), clock=_Clock()
        )
        _, _, admitted = registry.assign(_vec(0), _speech(1.0), "a", "xx")
        self.assertFalse(admitted)

    def test_a_quiet_slice_is_not_admitted(self):
        registry = SpeakerRegistry(clock=_Clock())
        _, _, admitted = registry.assign(
            _vec(0), _speech(3.0, amplitude=0.0001), "a", "xx"
        )
        self.assertFalse(admitted)

    def test_an_overlong_slice_is_trimmed_from_the_middle(self):
        cfg = SpeakerRegistryConfig(max_slice_s=4.0)
        registry = SpeakerRegistry(cfg, clock=_Clock())
        profile, _, _ = registry.assign(_vec(0), _speech(20.0), "a", "xx")
        self.assertAlmostEqual(profile.reference_seconds(), 4.0, places=3)

    def test_the_buffer_stops_growing_at_the_target(self):
        cfg = SpeakerRegistryConfig(rolling_prompt_s=8.0, max_slice_s=4.0)
        registry = SpeakerRegistry(cfg, clock=_Clock())
        for i in range(10):
            registry.assign(_vec(i * 0.5), _speech(3.0), f"u{i}", "xx")
        profile = registry.profiles()[0]
        self.assertLessEqual(profile.reference_seconds(), 8.0 + 4.0)
        self.assertGreaterEqual(profile.reference_seconds(), 8.0 - 4.0)

    def test_eviction_keeps_the_best_material_not_the_newest(self):
        cfg = SpeakerRegistryConfig(rolling_prompt_s=5.0, max_slice_s=8.0)
        registry = SpeakerRegistry(cfg, clock=_Clock())
        # A long loud slice first, then many short quiet ones.
        registry.assign(_vec(0), _speech(5.0, amplitude=0.3), "good", "xx")
        for i in range(5):
            registry.assign(_vec(1), _speech(2.1, amplitude=0.02), f"meh{i}", "xx")
        profile = registry.profiles()[0]
        kept = profile.reference_text()
        self.assertIn("good", kept, "the best slice was evicted by newer, worse ones")

    def test_enrolled_material_is_never_evicted(self):
        registry = SpeakerRegistry(
            SpeakerRegistryConfig(rolling_prompt_s=4.0, enrolled_prompt_s=6.0),
            clock=_Clock(),
        )
        registry.enroll("user", _vec(0), _speech(10.0), "enrolled sample", "xx")
        for i in range(8):
            registry.assign(_vec(1), _speech(3.0, amplitude=0.3), f"field{i}", "xx")
        profile = registry.get("enrolled:user")
        self.assertIn("enrolled sample", profile.reference_text())
        self.assertGreaterEqual(profile.reference_seconds(), 6.0)

    def test_the_two_prompt_slots_are_budgeted_independently(self):
        # The enrollment prompt anchors identity; the rolling prompt tracks the
        # current channel. Field audio must fill the second without eroding the
        # first, and the enrollment prompt is trimmed to its own budget rather
        # than kept whole.
        cfg = SpeakerRegistryConfig(
            enrolled_prompt_s=6.0, rolling_prompt_s=6.0, max_slice_s=3.0,
            min_slice_s=1.0,
        )
        registry = SpeakerRegistry(cfg, clock=_Clock())
        registry.enroll("user", _vec(0), _speech(30.0), "anchor", "xx")
        profile = registry.get("enrolled:user")
        self.assertAlmostEqual(profile.reference_seconds(), 6.0, places=2)
        for i in range(10):
            registry.assign(_vec(1), _speech(3.0, amplitude=0.3), f"field{i}", "xx")
        total = profile.reference_seconds()
        self.assertGreaterEqual(total, 6.0 + 3.0)
        self.assertLessEqual(total, 6.0 + 6.0 + 3.0)
        self.assertIn("anchor", profile.reference_text())

    def test_the_rolling_slot_rolls_towards_recent_material(self):
        # Two equally good slices far apart in time: the recent one must win,
        # or the buffer would freeze on the first room the speaker was in.
        cfg = SpeakerRegistryConfig(
            enrolled_prompt_s=0.0, rolling_prompt_s=3.0, max_slice_s=4.0,
            recency_half_life_s=5.0,
        )
        clock = _Clock()
        registry = SpeakerRegistry(cfg, clock=clock)
        registry.assign(_vec(0), _speech(3.0, amplitude=0.3), "old", "xx")
        clock.t += 100.0  # a hundred seconds of conversation later
        registry.assign(_vec(1), _speech(3.0, amplitude=0.3), "new", "xx")
        text = registry.profiles()[0].reference_text()
        self.assertIn("new", text)
        self.assertNotIn("old", text)

    def test_reference_for_refuses_with_the_shortfall_named(self):
        registry = SpeakerRegistry(clock=_Clock())
        registry.assign(_vec(0), _speech(2.5), "a", "xx")
        sid = registry.profiles()[0].speaker_id
        with self.assertRaises(ReferenceTooShort) as ctx:
            registry.reference_for(sid, min_seconds=8.0)
        self.assertAlmostEqual(ctx.exception.have_s, 2.5, places=2)
        self.assertEqual(ctx.exception.need_s, 8.0)

    def test_reference_audio_is_ordered_best_first(self):
        # Backends that truncate their reference must truncate the WORST part.
        registry = SpeakerRegistry(
            SpeakerRegistryConfig(rolling_prompt_s=100.0), clock=_Clock()
        )
        registry.assign(_vec(0), _speech(2.5, amplitude=0.05), "weak", "xx")
        registry.assign(_vec(1), _speech(6.0, amplitude=0.4), "strong", "xx")
        profile = registry.profiles()[0]
        self.assertTrue(profile.reference_text().startswith("strong"))


class TestIntraSegmentSpeakerChange(unittest.TestCase):
    def test_a_uniform_segment_has_no_split_point(self):
        windows = [_vec(0), _vec(3), _vec(1), _vec(2)]
        self.assertEqual(split_points_by_dispersion(windows), ())

    def test_a_back_to_back_speaker_change_is_located(self):
        windows = [_vec(0), _vec(2), _vec(85), _vec(87)]
        self.assertEqual(split_points_by_dispersion(windows), (2,))

    def test_a_single_window_cannot_split(self):
        self.assertEqual(split_points_by_dispersion([_vec(0)]), ())


class TestRollingReferenceHelper(unittest.TestCase):
    def test_it_takes_the_highest_scoring_segments_up_to_the_target(self):
        segments = [
            (_speech(2.0), 1.0),
            (_speech(4.0), 9.0),
            (_speech(3.0), 5.0),
        ]
        merged = rolling_reference_from_segments(segments, target_s=6.0)
        self.assertGreaterEqual(merged.duration_s, 6.0)
        self.assertLessEqual(merged.duration_s, 9.0)

    def test_no_segments_is_refused(self):
        with self.assertRaises(ValueError):
            rolling_reference_from_segments([], target_s=6.0)


if __name__ == "__main__":
    unittest.main()
