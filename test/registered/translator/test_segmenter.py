# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Turn segmentation: onset, hangover, pre-roll, the forced cut, the floor.

Hermetic, and deliberately driven by a SCRIPTED VAD rather than by the energy
detector: the thing under test is the state machine's timing, and letting a
real detector decide which frames are speech would make a timing failure and
a detection failure indistinguishable. The energy VAD gets its own test.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_segmenter.py -v
"""

import unittest

import numpy as np

from sglang.srt.translator.backends import AudioChunk
from sglang.srt.translator.segmenter import (
    EnergyVad,
    SegmentReason,
    SegmenterConfig,
    TurnSegmenter,
)

RATE = 16000
FRAME_MS = 20


class ScriptedVad:
    """Returns a fixed speech/silence pattern, one entry per frame."""

    def __init__(self, pattern, frame_ms=FRAME_MS):
        self.frame_ms = frame_ms
        self._pattern = list(pattern)
        self._i = 0
        self.resets = 0

    def is_speech(self, frame, sample_rate):
        del frame, sample_rate
        value = self._pattern[self._i] if self._i < len(self._pattern) else False
        self._i += 1
        return value

    def reset(self):
        self.resets += 1


def _audio(num_frames, rate=RATE, frame_ms=FRAME_MS, amplitude=0.2):
    n = int(rate * frame_ms / 1000) * num_frames
    # A ramp rather than silence, so every frame is distinguishable and a
    # pre-roll assertion can identify WHICH frames were prepended.
    return AudioChunk(
        (np.arange(n, dtype=np.float32) / max(n, 1) * amplitude), rate
    )


def _config(**kwargs):
    base = dict(
        sample_rate=RATE,
        frame_ms=FRAME_MS,
        onset_ms=60,
        hangover_ms=100,
        pre_roll_ms=60,
        min_utterance_ms=100,
        max_utterance_s=1.0,
    )
    base.update(kwargs)
    return SegmenterConfig(**base)


class TestStateMachine(unittest.TestCase):
    def test_pause_closes_a_segment_after_the_hangover(self):
        # 3 frames onset (60 ms) + 10 speech + 5 silence (100 ms hangover).
        pattern = [True] * 13 + [False] * 5
        vad = ScriptedVad(pattern)
        seg = TurnSegmenter(_config(), vad)
        segments = seg.feed(_audio(len(pattern)))
        self.assertEqual(len(segments), 1)
        self.assertIs(segments[0].reason, SegmentReason.PAUSE)
        self.assertEqual(segments[0].index, 0)

    def test_a_single_speech_frame_does_not_open_a_turn(self):
        # onset_ms=60 needs 3 consecutive frames; one impulse must not arm it.
        pattern = [False, True, False] * 6
        seg = TurnSegmenter(_config(), ScriptedVad(pattern))
        self.assertEqual(seg.feed(_audio(len(pattern))), [])
        self.assertFalse(seg.speaking)

    def test_hangover_is_not_tripped_by_a_clause_internal_pause(self):
        # 2 silent frames (40 ms) inside speech is below the 100 ms hangover.
        pattern = [True] * 5 + [False] * 2 + [True] * 5 + [False] * 5
        seg = TurnSegmenter(_config(), ScriptedVad(pattern))
        segments = seg.feed(_audio(len(pattern)))
        self.assertEqual(len(segments), 1, "the internal pause split the turn")

    def test_pre_roll_restores_the_clipped_attack(self):
        pattern = [True] * 8 + [False] * 5
        seg = TurnSegmenter(_config(pre_roll_ms=60), ScriptedVad(pattern))
        segments = seg.feed(_audio(len(pattern)))
        # 8 speech + 5 silence frames were consumed; the segment must contain
        # the onset frames too, so it is longer than (8 - onset) frames.
        self.assertGreaterEqual(segments[0].duration_s, 13 * FRAME_MS / 1000 - 1e-6)

    def test_forced_cut_bounds_a_monologue_and_keeps_the_machine_hot(self):
        # max_utterance_s=1.0 -> 50 frames. Feed 120 speech frames.
        pattern = [True] * 120
        seg = TurnSegmenter(_config(), ScriptedVad(pattern))
        segments = seg.feed(_audio(len(pattern)))
        self.assertGreaterEqual(len(segments), 2)
        self.assertTrue(all(s.reason is SegmentReason.FORCED for s in segments))
        self.assertTrue(seg.speaking, "a forced cut must not end the turn")
        # Indices are monotonic so the client can order them.
        self.assertEqual([s.index for s in segments], list(range(len(segments))))

    def test_too_short_an_utterance_is_dropped_not_emitted(self):
        # 3 onset frames + 1 speech = 80 ms < min_utterance_ms 100.
        pattern = [True] * 4 + [False] * 5
        seg = TurnSegmenter(
            _config(min_utterance_ms=400, pre_roll_ms=0), ScriptedVad(pattern)
        )
        self.assertEqual(seg.feed(_audio(len(pattern))), [])

    def test_flush_closes_a_turn_on_push_to_talk_release(self):
        pattern = [True] * 12
        seg = TurnSegmenter(_config(), ScriptedVad(pattern))
        seg.feed(_audio(len(pattern)))
        segment = seg.flush()
        self.assertIsNotNone(segment)
        self.assertIs(segment.reason, SegmentReason.RELEASED)
        self.assertIsNone(seg.flush(), "a second flush has nothing to close")


class TestFramingAndTransport(unittest.TestCase):
    def test_carry_buffer_makes_block_size_irrelevant(self):
        pattern = [True] * 13 + [False] * 5
        results = {}
        for block_frames in (1, 3, 7, 18):
            seg = TurnSegmenter(_config(), ScriptedVad(pattern))
            found = []
            n = int(RATE * FRAME_MS / 1000)
            audio = _audio(len(pattern))
            step = n * block_frames
            for start in range(0, len(audio.samples), step):
                found.extend(
                    seg.feed(AudioChunk(audio.samples[start : start + step], RATE))
                )
            results[block_frames] = [(s.index, round(s.duration_s, 4)) for s in found]
        # A transport that delivers 13 ms blocks must produce the same turns
        # as one delivering 360 ms blocks.
        self.assertEqual(len(set(map(str, results.values()))), 1, results)

    def test_a_rate_mismatch_is_refused_rather_than_resampled_silently(self):
        seg = TurnSegmenter(_config(), ScriptedVad([True] * 10))
        with self.assertRaises(ValueError) as ctx:
            seg.feed(AudioChunk(np.zeros(480, dtype=np.float32), 48000))
        self.assertIn("resample", str(ctx.exception))

    def test_reset_clears_stream_state_but_keeps_the_segment_cursor(self):
        pattern = [True] * 13 + [False] * 5 + [True] * 13 + [False] * 5
        vad = ScriptedVad(pattern)
        seg = TurnSegmenter(_config(), vad)
        first = seg.feed(_audio(18))
        self.assertEqual(len(first), 1)
        seg.reset()
        self.assertEqual(vad.resets, 1)
        self.assertFalse(seg.speaking)
        second = seg.feed(_audio(18))
        self.assertEqual(len(second), 1)
        # The cursor must keep increasing, or a client's resume position
        # would point at a different turn after a reconnect.
        self.assertEqual(second[0].index, 1)


class TestEnergyVad(unittest.TestCase):
    def test_it_finds_speech_above_a_noise_floor(self):
        vad = EnergyVad(frame_ms=FRAME_MS, margin_db=9.0)
        n = int(RATE * FRAME_MS / 1000)
        rng = np.random.default_rng(0)
        quiet = (rng.standard_normal(n) * 0.001).astype(np.float32)
        loud = (rng.standard_normal(n) * 0.2).astype(np.float32)
        for _ in range(30):
            vad.is_speech(quiet, RATE)
        self.assertFalse(vad.is_speech(quiet, RATE))
        self.assertTrue(vad.is_speech(loud, RATE))

    def test_a_stream_that_opens_with_speech_is_still_heard(self):
        """Falsifier for the deafness bug found through the WebSocket test.

        Push-to-talk always opens the stream with speech. Seeding the noise
        floor from that first frame set it at speech level, after which no
        frame could ever clear it by the margin and the detector was
        permanently deaf -- silently, with the turn simply never appearing.
        """
        vad = EnergyVad(frame_ms=FRAME_MS)
        n = int(RATE * FRAME_MS / 1000)
        t = np.arange(n, dtype=np.float32) / RATE
        loud = (0.3 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)
        decisions = [vad.is_speech(loud, RATE) for _ in range(10)]
        self.assertTrue(any(decisions), "the detector never heard the opening speech")

    def test_a_stream_that_opens_quiet_still_adapts_to_its_room(self):
        vad = EnergyVad(frame_ms=FRAME_MS)
        n = int(RATE * FRAME_MS / 1000)
        rng = np.random.default_rng(1)
        quiet = (rng.standard_normal(n) * 0.0005).astype(np.float32)
        for _ in range(50):
            vad.is_speech(quiet, RATE)
        self.assertFalse(vad.is_speech(quiet, RATE))


if __name__ == "__main__":
    unittest.main()
