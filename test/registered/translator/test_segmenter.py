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


class TestTheTapAtTheEndOfATurn(unittest.TestCase):
    """The finger on the display, heard in the output (user report 2026-08-04).

    The button is tap-to-start / tap-to-stop, so the turn ends with a finger
    hitting the phone that is holding the microphone. The impulse reaches the
    recognizer and, worse, the audio admitted to that speaker's clone
    reference, where it is re-heard in every later turn they voice. The client
    cannot cut it: audio streams continuously, so the tail frames are already
    sent by the time the release handler runs.

    The field number that sets the level test: package
    client-20260804T140542Z reports `microphone.peak` 0.592 while the session
    was only speech and 1 (saturated) after, so the transient sits at ~1.7x the
    loudest speech.

    Every test here feeds REAL audio through the real flush path; the scripted
    VAD only decides which frames are active, so a detection failure and a trim
    failure stay distinguishable.
    """

    BODY_PEAK = 0.30

    def _utterance(self, body_ms=600, pause_ms=150, burst_ms=40, burst_peak=1.0,
                   pause_peak=0.001):
        """Speech, then a pause, then the tap. Any part can be turned off."""
        rng = np.random.default_rng(7)

        def noise(ms, peak):
            n = int(RATE * ms / 1000)
            if n <= 0:
                return np.zeros(0, dtype=np.float32)
            raw = rng.standard_normal(n).astype(np.float32)
            return (raw / max(float(np.abs(raw).max()), 1e-9) * peak).astype(np.float32)

        return np.concatenate([
            noise(body_ms, self.BODY_PEAK),
            noise(pause_ms, pause_peak),
            noise(burst_ms, burst_peak),
        ])

    def _released(self, samples, config=None, reason=SegmentReason.RELEASED):
        frames = len(samples) // int(RATE * FRAME_MS / 1000)
        vad = ScriptedVad([True] * frames)
        seg = TurnSegmenter(config or _config(), vad)
        seg.feed(AudioChunk(samples[: frames * int(RATE * FRAME_MS / 1000)], RATE))
        return seg.flush(reason)

    def test_the_tap_is_cut_and_the_speech_is_kept(self):
        samples = self._utterance()
        segment = self._released(samples)
        self.assertIsNotNone(segment)
        removed_ms = (len(samples) - len(segment.audio.samples)) / RATE * 1000
        # Within one analysis frame of the 40 ms burst.
        self.assertAlmostEqual(removed_ms, 40, delta=10)
        self.assertLessEqual(
            float(np.abs(segment.audio.samples).max()), self.BODY_PEAK + 1e-6,
            "the saturating impulse is still in the audio handed to the "
            "recognizer and to the clone reference",
        )

    def test_a_turn_without_a_transient_is_byte_identical(self):
        """The control the whole design has to earn.

        Same audio through the same path, once with the trim able to fire and
        once with it made impossible, and the two must be the same samples.
        """
        samples = self._utterance(burst_ms=0)
        live = self._released(samples)
        disabled = self._released(samples, _config(tap_trim_burst_ratio=1e9))
        self.assertIsNotNone(live)
        np.testing.assert_array_equal(live.audio.samples, disabled.audio.samples)
        self.assertEqual(len(live.audio.samples) % int(RATE * FRAME_MS / 1000), 0)

    def test_a_tap_with_no_pause_before_it_keeps_the_last_phoneme(self):
        """No quiet boundary, no cut -- the case that would eat a final sound.

        A user who taps the instant a word ends leaves the transient sitting
        directly against speech. Cutting on loudness alone would take the end
        of the word with it, so the rule requires a boundary and this turn is
        left exactly as it arrived.
        """
        samples = self._utterance(pause_ms=0)
        live = self._released(samples)
        disabled = self._released(samples, _config(tap_trim_burst_ratio=1e9))
        np.testing.assert_array_equal(live.audio.samples, disabled.audio.samples)

    def test_a_loud_tail_that_is_not_louder_than_the_speech_is_kept(self):
        """1.2x the speech peak is under the 1.4x the field measured at 1.7x."""
        samples = self._utterance(burst_peak=self.BODY_PEAK * 1.2)
        live = self._released(samples)
        disabled = self._released(samples, _config(tap_trim_burst_ratio=1e9))
        np.testing.assert_array_equal(live.audio.samples, disabled.audio.samples)

    def test_a_segment_that_closed_on_a_pause_is_never_trimmed(self):
        """Only a RELEASE ends with a finger on the display."""
        samples = self._utterance()
        live = self._released(samples, reason=SegmentReason.PAUSE)
        # Whole frames only: the segmenter consumes 20 ms at a time.
        frame = int(RATE * FRAME_MS / 1000)
        self.assertEqual(len(live.audio.samples), len(samples) // frame * frame)

    def test_too_little_speech_to_judge_a_level_means_no_cut(self):
        samples = self._utterance(body_ms=100, pause_ms=60, burst_ms=40)
        live = self._released(samples)
        disabled = self._released(samples, _config(tap_trim_burst_ratio=1e9))
        np.testing.assert_array_equal(live.audio.samples, disabled.audio.samples)

    def test_the_length_policy_is_not_re_applied_after_the_cut(self):
        """A turn that WAS an utterance does not stop being one.

        `min_utterance_ms` answers "did the speaker say anything", and it was
        answered on the audio the speaker produced. Removing a noise they did
        not produce must not retroactively unmake that.
        """
        # 780 ms of whole frames reach `_close`, which admits them; the cut
        # then leaves ~740 ms, i.e. BELOW the threshold that admitted them.
        config = _config(min_utterance_ms=760)
        samples = self._utterance(body_ms=600, pause_ms=150, burst_ms=40)
        segment = self._released(samples, config)
        self.assertIsNotNone(segment, "the turn was dropped by its own tap")
        self.assertLess(segment.audio.duration_s, 0.76)

    def test_the_finding_carries_the_numbers_the_log_needs(self):
        from sglang.srt.translator.segmenter import find_tail_transient

        samples = self._utterance()
        found = find_tail_transient(AudioChunk(samples, RATE), _config())
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found.duration_s, 0.04, delta=0.01)
        self.assertAlmostEqual(found.peak, 1.0, delta=0.01)
        self.assertAlmostEqual(found.body_peak, self.BODY_PEAK, delta=0.01)
        self.assertGreater(found.body_rms, 0.0)
        self.assertAlmostEqual(
            found.cut_at_s, (len(samples) - int(RATE * 0.04)) / RATE, delta=0.01
        )

    def test_a_decaying_click_is_cut_from_its_onset(self):
        """The shape a real tap has, and the one that caught a real bug.

        A mechanical click decays, so its own last milliseconds are quieter
        than the boundary threshold. Taking the LAST quiet frame in the window
        therefore put the cut behind the transient and left it in the audio --
        which the constant-amplitude burst in the tests above cannot show,
        because it has no decay. Measured on a preset voice with a synthetic
        click, the tail ran 0.2748, 0.0351, 0.0052 RMS per 10 ms against a
        0.0167 threshold. The analysis anchors on the loudest frame instead.
        """
        rng = np.random.default_rng(5)

        def noise(ms, peak):
            n = int(RATE * ms / 1000)
            raw = rng.standard_normal(n).astype(np.float32)
            return (raw / float(np.abs(raw).max()) * peak).astype(np.float32)

        click = noise(40, 1.0) * np.exp(-np.linspace(0, 6, int(RATE * 0.04)))
        samples = np.concatenate([
            noise(600, self.BODY_PEAK), noise(150, 0.001),
            click.astype(np.float32),
        ])
        segment = self._released(samples)
        removed_ms = (len(samples) // 320 * 320 - len(segment.audio.samples)) / RATE * 1000
        self.assertAlmostEqual(removed_ms, 40, delta=10)
        self.assertLessEqual(
            float(np.abs(segment.audio.samples).max()), self.BODY_PEAK + 1e-6,
            "the decaying click survived the trim",
        )

    def test_the_start_tap_is_measured_and_not_cut(self):
        """The other end of the same button, deliberately observed only.

        Cutting at the head would risk the attack that `pre_roll_ms` exists to
        protect, and there is no field evidence that it happens, so the finding
        is logged and the audio is untouched.
        """
        rng = np.random.default_rng(11)

        def noise(ms, peak):
            n = int(RATE * ms / 1000)
            raw = rng.standard_normal(n).astype(np.float32)
            return (raw / float(np.abs(raw).max()) * peak).astype(np.float32)

        samples = np.concatenate([
            noise(40, 1.0),                    # the tap
            noise(150, 0.001),                 # the pause before speaking
            noise(600, self.BODY_PEAK),        # the utterance
        ])
        with self.assertLogs("sglang.srt.translator.segmenter", "INFO") as logs:
            live = self._released(samples)
        self.assertTrue(any("OPENS with a" in line for line in logs.output),
                        "the start transient was not even measured")
        disabled = self._released(samples, _config(tap_trim_burst_ratio=1e9))
        np.testing.assert_array_equal(live.audio.samples, disabled.audio.samples)

    def test_a_quiet_tail_with_no_burst_is_not_a_transient(self):
        from sglang.srt.translator.segmenter import find_tail_transient

        samples = self._utterance(burst_ms=0, pause_ms=200)
        self.assertIsNone(find_tail_transient(AudioChunk(samples, RATE), _config()))


if __name__ == "__main__":
    unittest.main()
