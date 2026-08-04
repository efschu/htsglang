# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Opus transport: round trip, framing, bandwidth, and the negotiation path.

Bandwidth is the whole reason Opus exists in this system. PCM16 at 16 kHz is
~256 kbps, which is fine on the LAN and ruinous on Spanish mobile data with a
per-turn duty cycle in both directions. So one of these tests asserts the
actual encoded bitrate rather than merely that encoding succeeded -- a codec
that silently fell back to a high bitrate would pass every other test here and
fail the only requirement it was added for.

Skipped wholesale when PyAV is absent, because the deployment is then correctly
running on the PCM16 fallback and there is nothing to test.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_opus.py -v
"""

import math
import unittest

import numpy as np

from sglang.srt.translator.audio import (
    CodecError,
    OpusCodec,
    Pcm16Codec,
    available_codecs,
    negotiate_codec,
    resample,
)
from sglang.srt.translator.backends import AudioChunk

HAVE_OPUS = "opus" in available_codecs()
OPUS_RATE = 48000


def speech_like(seconds=3.0, rate=OPUS_RATE, f0=130.0, seed=0):
    """A harmonic stack with a syllabic envelope.

    A pure tone is the wrong probe for a speech codec: Opus's VBR overshoots
    badly on one, so a bitrate measured on a sine says nothing about the link
    budget. This is closer to voiced speech -- harmonics, an amplitude
    envelope at syllable rate, a little noise floor.
    """
    n = int(seconds * rate)
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32) / rate
    signal = np.zeros(n, dtype=np.float32)
    for k, amplitude in enumerate([1.0, 0.6, 0.35, 0.2, 0.1], start=1):
        signal += amplitude * np.sin(2.0 * math.pi * f0 * k * t)
    envelope = (0.5 + 0.5 * np.sin(2.0 * math.pi * 3.5 * t)).astype(np.float32)
    signal = (signal / 6.0 * envelope + 0.01 * rng.standard_normal(n)).astype(np.float32)
    return AudioChunk(signal, rate)


def dominant_hz(chunk):
    spectrum = np.abs(np.fft.rfft(chunk.samples))
    freqs = np.fft.rfftfreq(len(chunk.samples), 1.0 / chunk.sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


@unittest.skipUnless(HAVE_OPUS, "PyAV is not installed; deployment uses pcm16")
class TestOpusRoundTrip(unittest.TestCase):
    def test_encode_decode_preserves_length_and_pitch(self):
        original = speech_like(seconds=1.0)
        frames = list(OpusCodec().encode(original))
        self.assertTrue(frames)

        decoder = OpusCodec()
        decoded = [decoder.decode(f) for f in frames]
        total = sum(len(d.samples) for d in decoded)
        self.assertEqual(
            total,
            len(original.samples),
            "a length change means frames were dropped or duplicated",
        )

        merged = AudioChunk(
            np.concatenate([d.samples for d in decoded]), OPUS_RATE
        )
        # Opus is lossy, so sample equality is meaningless; pitch is not. A
        # wrong sample rate somewhere in the path shows up here and nowhere
        # else until someone hears a chipmunk in Spain.
        self.assertAlmostEqual(dominant_hz(merged), dominant_hz(original), delta=8.0)

    def test_frames_are_the_declared_size(self):
        codec = OpusCodec(frame_ms=20)
        frames = list(codec.encode(speech_like(seconds=1.0)))
        # 1 s at 20 ms per frame.
        self.assertEqual(len(frames), 50)

    def test_the_bitrate_is_actually_near_the_target(self):
        seconds = 3.0
        audio = speech_like(seconds=seconds)
        for target in (16000, 24000, 32000):
            with self.subTest(target=target):
                frames = list(OpusCodec(bitrate_bps=target).encode(audio))
                measured = sum(len(f) for f in frames) * 8 / seconds
                # +-25% around the requested rate. Loose enough for VBR,
                # tight enough that a silent fallback to a default bitrate
                # (Opus's own default is far higher) fails the test.
                self.assertGreater(measured, target * 0.75, f"{measured:.0f} bps")
                self.assertLess(measured, target * 1.25, f"{measured:.0f} bps")

    def test_the_default_bitrate_meets_the_mobile_budget(self):
        # The stated design target: ~24 kbps, which is what makes a roaming
        # data plan viable for a conversation.
        seconds = 3.0
        frames = list(OpusCodec().encode(speech_like(seconds=seconds)))
        measured = sum(len(f) for f in frames) * 8 / seconds
        self.assertLess(measured, 30000, f"{measured:.0f} bps exceeds the budget")

    def test_opus_is_far_cheaper_than_the_pcm_fallback(self):
        seconds = 3.0
        audio = speech_like(seconds=seconds)
        opus_bytes = sum(len(f) for f in OpusCodec().encode(audio))
        pcm = resample(audio, 16000)
        pcm_bytes = sum(len(f) for f in Pcm16Codec(sample_rate=16000).encode(pcm))
        self.assertLess(
            opus_bytes * 8, pcm_bytes,
            "Opus must be at least 8x cheaper than PCM16 to be worth the "
            "dependency",
        )

    def test_a_pipeline_rate_chunk_is_resampled_on_the_way_out(self):
        # The synthesizer emits 24 kHz and the pipeline runs at 16 kHz; Opus
        # owns the 48 kHz side. The codec must convert rather than mislabel.
        codec = OpusCodec()
        at_16k = AudioChunk(speech_like(seconds=0.5, rate=16000).samples, 16000)
        frames = list(codec.encode(at_16k))
        self.assertTrue(frames)
        decoder = OpusCodec()
        merged = AudioChunk(
            np.concatenate([decoder.decode(f).samples for f in frames]), OPUS_RATE
        )
        self.assertAlmostEqual(merged.duration_s, at_16k.duration_s, delta=0.05)

    def test_decoder_state_resets(self):
        codec = OpusCodec()
        frames = list(codec.encode(speech_like(seconds=0.5)))
        decoder = OpusCodec()
        first = [decoder.decode(f) for f in frames]
        decoder.reset()
        second = [decoder.decode(f) for f in frames]
        self.assertEqual(
            sum(len(d.samples) for d in first),
            sum(len(d.samples) for d in second),
        )

    def test_garbage_is_refused_rather_than_returned_as_noise(self):
        decoder = OpusCodec()
        with self.assertRaises(Exception):
            decoder.decode(b"\xff" * 64)


class TestNegotiationWithOpus(unittest.TestCase):
    def test_opus_wins_when_both_sides_have_it(self):
        if not HAVE_OPUS:
            self.skipTest("PyAV is not installed")
        self.assertEqual(negotiate_codec(["opus", "pcm16"]).name, "opus")

    def test_pcm16_is_chosen_when_the_client_lacks_opus(self):
        self.assertEqual(negotiate_codec(["pcm16"]).name, "pcm16")

    def test_the_advertised_list_reflects_what_is_installed(self):
        codecs = available_codecs()
        self.assertEqual(codecs[-1], "pcm16", "pcm16 must always be the floor")
        if HAVE_OPUS:
            self.assertEqual(codecs[0], "opus", "opus must be preferred")

    def test_no_common_codec_names_both_sides(self):
        with self.assertRaises(CodecError):
            negotiate_codec(["speex"])


if __name__ == "__main__":
    unittest.main()
