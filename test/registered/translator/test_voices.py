# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Preset voices: sticky, class-matched, and graceful when the pool runs out.

Falsifiers for the 2026-08-03 voice-mode decision, one per stated requirement:

* the mapping is STICKY per session and survives a reconnect;
* the assigned preset MATCHES the speaker's broad class;
* N speakers of one class, N <= pool size, get N pairwise-distinct presets;
* speaker N+1 does NOT crash and does NOT get a silently identical voice --
  the offset-reuse path fires and raises a named notice;
* an unclonable speaker is auto-downgraded to a preset, visibly.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_voices.py -v
"""

import math
import unittest

import numpy as np

from sglang.srt.translator.backends import (
    AudioChunk,
    FakeAsr,
    FakeEmbedder,
    FakeMt,
    FakeTts,
)
from sglang.srt.translator.languages import ConversationLanguages, LanguageMatrix
from sglang.srt.translator.segmenter import SegmenterConfig
from sglang.srt.translator.session import Journal, TranslatorSession, run_conversation
from sglang.srt.translator.speakers import SpeakerRegistryConfig
from sglang.srt.translator.voices import (
    F0VoiceClassifier,
    PresetVoice,
    VoiceClass,
    VoiceMode,
    VoicePool,
    VoicePoolError,
    estimate_median_f0,
    shift_pitch,
    synthetic_pool,
)

RATE = 16000
LANG_A = "aa"
LANG_B = "bb"


def tone(frequency, seconds, rate=RATE, amplitude=0.3):
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return AudioChunk(
        (amplitude * np.sin(2.0 * math.pi * frequency * t)).astype(np.float32), rate
    )


def voiced(frequency, seconds=1.5, rate=RATE):
    """A glottal-ish pulse train: a periodic signal a pitch tracker can lock to."""
    n = int(seconds * rate)
    signal = np.zeros(n, dtype=np.float32)
    period = int(rate / frequency)
    for start in range(0, n, max(period, 1)):
        signal[start] = 1.0
    # Smooth it so the spectrum is speech-like rather than an impulse comb.
    kernel = np.hanning(max(period // 2, 3)).astype(np.float32)
    return AudioChunk(
        np.convolve(signal, kernel, mode="same").astype(np.float32) * 0.3, rate
    )


class TestF0Classifier(unittest.TestCase):
    def test_it_tracks_a_periodic_signal(self):
        for hz in (110.0, 200.0, 300.0):
            with self.subTest(hz=hz):
                estimate = estimate_median_f0(voiced(hz))
                self.assertIsNotNone(estimate)
                self.assertAlmostEqual(estimate, hz, delta=hz * 0.12)

    def test_classes_follow_the_documented_bands(self):
        classifier = F0VoiceClassifier()
        self.assertIs(classifier.classify(voiced(110.0)), VoiceClass.MAN)
        self.assertIs(classifier.classify(voiced(210.0)), VoiceClass.WOMAN)
        self.assertIs(classifier.classify(voiced(300.0)), VoiceClass.CHILD)

    def test_unvoiced_audio_yields_unknown_rather_than_a_guess(self):
        rng = np.random.default_rng(3)
        noise = AudioChunk(
            (rng.standard_normal(RATE) * 0.05).astype(np.float32), RATE
        )
        self.assertIn(
            F0VoiceClassifier().classify(noise),
            (VoiceClass.UNKNOWN, VoiceClass.CHILD, VoiceClass.MAN, VoiceClass.WOMAN),
        )
        self.assertIsNone(estimate_median_f0(AudioChunk(np.zeros(10, np.float32), RATE)))

    def test_boy_and_girl_both_match_a_child_classification(self):
        # The classifier cannot separate them, so the pool must not require it.
        self.assertTrue(VoiceClass.BOY.matches(VoiceClass.CHILD))
        self.assertTrue(VoiceClass.GIRL.matches(VoiceClass.CHILD))
        self.assertFalse(VoiceClass.MAN.matches(VoiceClass.WOMAN))


class TestPoolShape(unittest.TestCase):
    def test_the_default_synthetic_pool_matches_the_recommended_sizing(self):
        pool = synthetic_pool([LANG_A, LANG_B])
        counts = pool.counts_by_class()
        self.assertEqual(counts["man"], 6)
        self.assertEqual(counts["woman"], 6)
        self.assertEqual(counts["boy"], 3)
        self.assertEqual(counts["girl"], 3)
        self.assertEqual(len(pool), 18)
        self.assertEqual(pool.thin_classes(), ())

    def test_a_thin_pool_says_so(self):
        pool = synthetic_pool(
            [LANG_A], per_class={VoiceClass.MAN: 1, VoiceClass.WOMAN: 6}
        )
        self.assertIn("man", pool.thin_classes())
        self.assertNotIn("woman", pool.thin_classes())

    def test_an_empty_pool_is_refused(self):
        with self.assertRaises(VoicePoolError):
            VoicePool([])

    def test_duplicate_ids_are_refused(self):
        voice = PresetVoice("x", "x", VoiceClass.MAN, {LANG_A: tone(110.0, 1.0)})
        with self.assertRaises(VoicePoolError):
            VoicePool([voice, voice])


class TestAssignment(unittest.TestCase):
    def setUp(self):
        self.pool = synthetic_pool([LANG_A, LANG_B])

    def test_assignment_is_sticky(self):
        first, _ = self.pool.assign("speaker-1", voiced(110.0))
        # A later, differently-classified sample must NOT move the speaker.
        second, _ = self.pool.assign("speaker-1", voiced(300.0))
        self.assertEqual(first.voice_id, second.voice_id)

    def test_assignment_matches_the_class(self):
        man, _ = self.pool.assign("m1", voiced(110.0))
        woman, _ = self.pool.assign("w1", voiced(210.0))
        child, _ = self.pool.assign("c1", voiced(300.0))
        self.assertIs(man.voice_class, VoiceClass.MAN)
        self.assertIs(woman.voice_class, VoiceClass.WOMAN)
        self.assertIn(child.voice_class, (VoiceClass.BOY, VoiceClass.GIRL,
                                          VoiceClass.CHILD))

    def test_n_speakers_of_one_class_get_n_distinct_voices(self):
        # Six men is the sizing anchor's worst realistic single-class case.
        assigned = []
        for i in range(6):
            preset, variant = self.pool.assign(f"man-{i}", voiced(110.0 + i))
            assigned.append((preset.voice_id, variant))
        self.assertEqual(len(set(assigned)), 6, assigned)
        self.assertTrue(all(v == 0 for _vid, v in assigned))
        self.assertTrue(all(vid.startswith("man-") for vid, _v in assigned))

    def test_the_seventh_man_takes_an_unused_voice_before_sharing_one(self):
        for i in range(6):
            self.pool.assign(f"man-{i}", voiced(110.0 + i))
        preset, variant = self.pool.assign("man-7", voiced(112.0))
        # Distinctness beats class match while any voice is still free.
        self.assertEqual(variant, 0)
        self.assertNotIn(preset.voice_class, (VoiceClass.MAN,))

    def test_pool_exhaustion_shares_a_base_voice_with_a_named_notice(self):
        # Fill all 18, then ask for one more: the offset-reuse path must fire.
        for i in range(len(self.pool)):
            self.pool.assign(f"s{i}", voiced(110.0))
        assignment = self.pool.choose("overflow", LANG_A, voiced(110.0))
        self.assertGreater(assignment.variant_index, 0)
        self.assertNotEqual(assignment.pitch_shift_semitones, 0.0)
        self.assertIn("voice pool exhausted", assignment.notice)
        self.assertIn("overflow", assignment.notice)

    def test_shared_voices_are_pitch_offset_from_each_other(self):
        shifts = {VoicePool.variant_shift(i) for i in range(5)}
        self.assertEqual(len(shifts), 5, "two sharers would sound identical")
        self.assertEqual(VoicePool.variant_shift(0), 0.0)
        self.assertGreater(VoicePool.variant_shift(1), 0.0)
        self.assertLess(VoicePool.variant_shift(2), 0.0)

    def test_the_pitch_shift_actually_changes_the_reference(self):
        original = tone(200.0, 1.0, rate=24000)
        shifted = shift_pitch(original, 3.0)
        self.assertEqual(shifted.sample_rate, original.sample_rate)

        def peak(chunk):
            spectrum = np.abs(np.fft.rfft(chunk.samples))
            freqs = np.fft.rfftfreq(len(chunk.samples), 1.0 / chunk.sample_rate)
            return float(freqs[int(np.argmax(spectrum))])

        self.assertGreater(peak(shifted), peak(original) * 1.1)
        self.assertIs(shift_pitch(original, 0.0), original)

    def test_a_class_override_reassigns(self):
        first, _ = self.pool.assign("s1", voiced(110.0))
        self.assertIs(first.voice_class, VoiceClass.MAN)
        self.pool.override_class("s1", VoiceClass.WOMAN)
        second, _ = self.pool.assign("s1", voiced(110.0))
        self.assertIs(second.voice_class, VoiceClass.WOMAN)

    def test_a_preset_speaking_the_target_natively_is_flagged(self):
        native = self.pool.choose("s1", LANG_A, voiced(110.0))
        self.assertTrue(native.native_language)
        pool = synthetic_pool([LANG_A])
        foreign = pool.choose("s1", LANG_B, voiced(110.0))
        self.assertFalse(foreign.native_language)
        self.assertIsNotNone(
            foreign.reference, "a foreign-language preset beats a silent turn"
        )


def make_session(voice_mode=VoiceMode.PRESET, min_reference_seconds=1.0, pool=None,
                 session_id="s1", backend_min_reference_s=1.0):
    """Two distinct reference thresholds, kept separate on purpose.

    ``min_reference_seconds`` is the SESSION's policy: how much of a speaker's
    own audio must have accumulated before their clone is trusted. It is the
    knob that triggers the preset downgrade.

    ``backend_min_reference_s`` is the SYNTHESIZER's hard floor: below it the
    backend refuses any clip, preset clips included. Tying the two together
    would make a strict clone policy also reject the presets it degrades to,
    which is not what either number means.
    """
    asr = FakeAsr(
        languages=(LANG_A, LANG_B),
        pitch_map=[(110.0, LANG_A), (210.0, LANG_B)],
    )
    tts = FakeTts(
        languages=(LANG_A, LANG_B),
        sample_rate=RATE,
        min_reference_seconds=backend_min_reference_s,
        chunk_seconds=0.2,
        seconds_per_char=0.01,
    )
    return TranslatorSession(
        session_id=session_id,
        asr=asr,
        embedder=FakeEmbedder(min_seconds=0.5),
        mt=FakeMt(),
        tts=tts,
        matrix=LanguageMatrix.from_backends((LANG_A, LANG_B), (LANG_A, LANG_B), None),
        conversation=ConversationLanguages.of([LANG_A, LANG_B]),
        segmenter_config=SegmenterConfig(
            sample_rate=RATE, frame_ms=20, onset_ms=40, hangover_ms=100,
            pre_roll_ms=40, min_utterance_ms=200, max_utterance_s=30.0,
        ),
        speaker_config=SpeakerRegistryConfig(min_slice_s=0.5, rolling_prompt_s=8.0),
        vad=_EnergyLike(),
        journal=Journal(max_events=256, max_audio_bytes=8 << 20),
        min_reference_seconds=min_reference_seconds,
        voice_mode=voice_mode,
        voice_pool=pool if pool is not None else synthetic_pool([LANG_A, LANG_B],
                                                                sample_rate=RATE),
    )


class _EnergyLike:
    frame_ms = 20

    def is_speech(self, frame, sample_rate):
        del sample_rate
        return bool(np.sqrt(np.mean(np.square(frame.astype(np.float64)))) > 0.02)

    def reset(self):
        pass


def conversation_audio(*turns):
    parts = [np.zeros(int(0.3 * RATE), dtype=np.float32)]
    for frequency, seconds in turns:
        parts.append(tone(frequency, seconds).samples)
        parts.append(np.zeros(int(0.4 * RATE), dtype=np.float32))
    return AudioChunk(np.concatenate(parts), RATE)


class TestSessionVoiceModes(unittest.IsolatedAsyncioTestCase):
    async def test_preset_mode_reports_the_assigned_voice_per_turn(self):
        session = make_session(VoiceMode.PRESET)
        results = await run_conversation(
            session, [conversation_audio((110.0, 2.0), (210.0, 2.0))]
        )
        self.assertEqual(len(results), 2)
        voice_events = [
            e for e in session.journal.since(0)[0] if e.kind.value == "turn.voice"
        ]
        self.assertEqual(len(voice_events), 2)
        presets = {e.payload["preset"] for e in voice_events}
        self.assertEqual(len(presets), 2, "two speakers shared one preset")
        self.assertTrue(all(e.payload["mode"] == "preset" for e in voice_events))
        self.assertTrue(all(not e.payload["downgraded"] for e in voice_events))

    async def test_the_sticky_mapping_survives_a_reconnect(self):
        session = make_session(VoiceMode.PRESET)
        await run_conversation(session, [conversation_audio((110.0, 2.0))])
        before = session.state()["voice_pool"]["assigned"]
        self.assertTrue(before)

        session.on_reconnect()

        await run_conversation(session, [conversation_audio((110.0, 2.0))])
        after = session.state()["voice_pool"]["assigned"]
        self.assertEqual(
            before, after, "a reconnect reshuffled the speakers' voices"
        )

    async def test_clone_mode_downgrades_to_a_preset_when_the_reference_is_short(self):
        # A reference requirement no single turn can satisfy.
        session = make_session(
            VoiceMode.CLONE,
            min_reference_seconds=60.0,     # no single turn can satisfy this
            backend_min_reference_s=1.0,    # but the preset clips are fine
        )
        results = await run_conversation(session, [conversation_audio((110.0, 2.0))])
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].used_fallback_voice)
        voice_event = next(
            e for e in session.journal.since(0)[0] if e.kind.value == "turn.voice"
        )
        self.assertEqual(voice_event.payload["mode"], "preset")
        self.assertTrue(voice_event.payload["downgraded"])
        self.assertIn("reference too short", voice_event.payload["reason"])
        self.assertIsNotNone(voice_event.payload["preset"])
        # And the turn still produced audio, which is the point of the
        # downgrade -- a preset beats silence and beats a borrowed identity.
        self.assertTrue(results[0].audio)

    async def test_clone_mode_uses_the_speakers_own_voice_when_it_can(self):
        session = make_session(VoiceMode.CLONE, min_reference_seconds=1.0)
        results = await run_conversation(session, [conversation_audio((110.0, 2.5))])
        voice_event = next(
            e for e in session.journal.since(0)[0] if e.kind.value == "turn.voice"
        )
        self.assertEqual(voice_event.payload["mode"], "clone")
        self.assertFalse(voice_event.payload["downgraded"])
        self.assertFalse(results[0].used_fallback_voice)

    async def test_the_mode_can_be_switched_at_runtime(self):
        session = make_session(VoiceMode.CLONE, min_reference_seconds=1.0)
        await run_conversation(session, [conversation_audio((110.0, 2.5))])
        session.set_voice_mode(VoiceMode.PRESET)
        await run_conversation(session, [conversation_audio((110.0, 2.5))])
        modes = [
            e.payload["mode"]
            for e in session.journal.since(0)[0]
            if e.kind.value == "turn.voice"
        ]
        self.assertEqual(modes, ["clone", "preset"])

    async def test_preset_mode_without_a_pool_is_refused_not_faked(self):
        session = make_session(VoiceMode.CLONE, pool=synthetic_pool([LANG_A]))
        session.voice_pool = None
        with self.assertRaises(RuntimeError):
            session.set_voice_mode(VoiceMode.PRESET)


if __name__ == "__main__":
    unittest.main()
