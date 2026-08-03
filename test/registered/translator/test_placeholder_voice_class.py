# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The placeholder voice must match the speaker on their FIRST turn (§19.10).

User, from the field, both directions on real turns: *"at the start I am
still rendered with a female voice"*, and for his wife the mirror case, a
male voice until her own clone exists.

The machinery to prevent this was all present -- an F0 classifier, a
class-matched preset pool, sticky per-speaker assignment. Its REACH was zero
at the only moment it is asked to act:

``_speaker_audio()`` classified from the speaker's REFERENCE BUFFER, and the
downgrade path that needs a placeholder fires precisely BECAUSE that buffer
is too short to clone from. So the classifier was handed ``None`` on every
first turn, answered ``UNKNOWN``, and the pool fell through to allocating by
availability -- which hands a man whatever slot is free, including a woman's.

The fix is to classify from the turn's own audio, which is in hand anyway and
is a strictly better input: it is this speaker, now, by construction.

This file pins the first turn specifically, because every later turn already
worked and that is exactly what made the defect survive.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_placeholder_voice_class.py -v
"""

import unittest

import numpy as np

from sglang.srt.translator.backends import AudioChunk
from sglang.srt.translator.voices import VoiceClass, estimate_median_f0


RATE = 16000


def voiced(f0_hz, seconds=3.0, rate=RATE):
    """A glottal-ish pulse train: strong periodicity at the wanted F0.

    A pure sine is a poor test signal for a pitch tracker -- it has one
    partial and a real voice has many -- so harmonics are stacked to make the
    period unambiguous.
    """
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    wave = np.zeros_like(t)
    for harmonic in range(1, 12):
        wave += np.sin(2 * np.pi * f0_hz * harmonic * t) / harmonic
    wave = wave / np.max(np.abs(wave))
    return AudioChunk((0.4 * wave).astype(np.float32), rate)


class TestTheInstrumentDiscriminates(unittest.TestCase):
    """Precondition: a classifier that cannot tell the cases apart would make
    every assertion below meaningless (the spread rule)."""

    def test_the_f0_estimate_tracks_the_input(self):
        for hz in (110.0, 200.0, 280.0):
            measured = estimate_median_f0(voiced(hz))
            self.assertIsNotNone(measured, "no F0 for a %g Hz tone" % hz)
            self.assertAlmostEqual(
                measured, hz, delta=0.12 * hz,
                msg="measured %r for %r Hz" % (measured, hz),
            )


class TestTheFirstTurnIsClassified(unittest.IsolatedAsyncioTestCase):
    """The two field cases, as tests."""

    def _pool_class(self, f0_hz):
        from sglang.srt.translator.voices import F0VoiceClassifier

        return F0VoiceClassifier().classify(voiced(f0_hz))

    def test_a_male_voice_is_not_classified_as_a_woman(self):
        """His case: 110 Hz must never reach a female preset."""
        self.assertEqual(self._pool_class(110.0), VoiceClass.MAN)

    def test_a_female_voice_is_not_classified_as_a_man(self):
        """His wife's case, the mirror."""
        self.assertEqual(self._pool_class(210.0), VoiceClass.WOMAN)

    async def test_the_turns_own_audio_is_used_when_no_reference_exists(self):
        """The reach fix itself: first turn, empty buffer, still classified."""
        from test_session import (  # noqa: E402 - sibling helper
            VOICE_A_HZ,
            conversation_audio,
            make_session,
        )

        session, _asr, _mt, _tts = make_session()
        segments = session.push_audio(conversation_audio((VOICE_A_HZ, 1.5)))
        self.assertTrue(segments)
        session._turn_audio = segments[0].audio

        # A speaker that exists but has NO references -- the exact state the
        # downgrade path runs in.
        profile = session.speakers.create_speaker()
        self.assertFalse(profile.references)
        audio = session._speaker_audio(profile.speaker_id)
        self.assertIsNotNone(
            audio,
            "the classifier gets None on a first turn -- the pool then "
            "allocates by availability and the placeholder voice is a "
            "coin flip",
        )

    async def test_a_real_reference_still_wins_over_the_turn(self):
        """The fallback must not override better evidence once it exists."""
        from test_session import (  # noqa: E402 - sibling helper
            VOICE_A_HZ,
            conversation_audio,
            make_session,
        )

        session, _asr, _mt, _tts = make_session()
        segments = session.push_audio(conversation_audio((VOICE_A_HZ, 1.5)))
        session._turn_audio = segments[0].audio
        results = await session.run_turn_multi(segments[0])
        self.assertTrue(results)
        speaker_id = results[-1].speaker_id
        profile = session.speakers.get(speaker_id)
        if profile.references:
            session._turn_audio = None
            self.assertIsNotNone(
                session._speaker_audio(speaker_id),
                "an enrolled speaker must still classify from their buffer",
            )


if __name__ == "__main__":
    unittest.main()
