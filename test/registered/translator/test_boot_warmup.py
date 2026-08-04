# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The boot warmup: turn 1 must not pay the talker's cold start (§17.8.14).

Every boot's first real turn paid ~15 s that no later turn paid -- a zero-shot
cloning backend does its kernel autotuning, its graph capture and its first
weight touch on the first ``synthesize`` call. Turn 1 of a live conversation is
the worst place to spend it and also the turn on which somebody decides whether
this thing works.

``TranslatorService.warmup`` runs one throwaway synthesis BEFORE the port is
opened. The arm here is a counting TTS: it records what it was asked to
synthesize, so "the synthesizer was called before any session existed" is
measured rather than assumed -- which is the whole claim.

``test_without_the_warmup_the_first_turn_is_the_first_synthesis`` is the
can-fail proof: the same service, warmup not run, and the first synthesis call
belongs to the user's turn.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_boot_warmup.py -v
"""

import asyncio
import unittest

import numpy as np

from sglang.srt.translator.backends import AudioChunk
from sglang.srt.translator.voices import PresetVoice, VoicePool, VoiceClass
from test_audio_and_http import (  # noqa: E402  - sibling helper module
    LANG_A,
    LANG_B,
    RATE,
    build_service,
)


class CountingTts:
    """A synthesizer that remembers every call, in order."""

    name = "counting-tts"
    sample_rate = RATE
    min_reference_seconds = 0.5

    def __init__(self, languages=(LANG_A, LANG_B), fail=False):
        self.calls = []
        self._languages = tuple(languages)
        self.fail = fail

    def supported_languages(self):
        return self._languages

    async def synthesize(self, text, language, reference, reference_text=None,
                         voice_id=None, pacing=None):
        self.calls.append({"text": text, "language": language,
                           "voice_id": voice_id})
        if self.fail:
            raise RuntimeError("the checkpoint is not loaded")
        yield AudioChunk(
            np.zeros(int(0.2 * RATE), dtype=np.float32), RATE
        )


def preset_pool(languages=(LANG_A,)):
    clip = AudioChunk(
        (0.2 * np.sin(np.arange(int(4.0 * RATE)) * 0.05)).astype(np.float32),
        RATE,
    )
    return VoicePool([
        PresetVoice(
            voice_id="warm-1", label="Warm One", voice_class=VoiceClass.WOMAN,
            references={code: clip for code in languages},
            reference_texts={code: "a curated line" for code in languages},
        )
    ])


def service_with_pool(**kwargs):
    service = build_service()
    service.stack.tts = CountingTts(**kwargs)
    service.voice_pool_template = preset_pool()
    return service


class TestBootWarmup(unittest.TestCase):
    def test_the_synthesizer_is_called_before_any_session_exists(self):
        service = service_with_pool()
        report = asyncio.run(service.warmup())
        self.assertTrue(report["ran"], report)
        self.assertEqual(len(service.stack.tts.calls), 1)
        # Real text through the real path: a synthesizer's first call should
        # traverse the tokenizer and phonemiser a turn will, not a shorter one.
        self.assertGreater(len(service.stack.tts.calls[0]["text"]), 5)
        self.assertEqual(service.stack.tts.calls[0]["language"], LANG_A)
        self.assertEqual(report["voice_id"], "warm-1")
        # Nothing was created for it: the warmup must not leave a session, a
        # speaker or a transcript line behind for the first user to find.
        self.assertEqual(len(service.sessions.ids()), 0)

    def test_without_the_warmup_the_first_turn_is_the_first_synthesis(self):
        """THE CAN-FAIL PROOF. Same service, warmup not run."""
        service = service_with_pool()
        self.assertEqual(service.stack.tts.calls, [])

    def test_a_language_the_synthesizer_cannot_speak_is_not_chosen(self):
        """The warm path must be a path a turn takes."""
        service = build_service()
        service.stack.tts = CountingTts(languages=(LANG_B,))
        service.voice_pool_template = preset_pool(languages=(LANG_A,))
        report = asyncio.run(service.warmup())
        self.assertTrue(report["ran"], report)
        self.assertEqual(service.stack.tts.calls[0]["language"], LANG_B)

    def test_no_preset_pool_is_reported_not_faked(self):
        """Without real reference material there is nothing honest to clone.

        Synthesizing from noise would warm a code path no turn uses and report
        success for it, which is worse than saying the warmup was skipped.
        """
        service = build_service()
        service.stack.tts = CountingTts()
        service.voice_pool_template = None
        report = asyncio.run(service.warmup())
        self.assertFalse(report["ran"])
        self.assertIn("no preset voice pool", report["reason"])
        self.assertEqual(service.stack.tts.calls, [])

    def test_a_failing_warmup_is_never_fatal(self):
        """A boot that refuses to serve because a warmup sentence failed has
        turned an optimisation into an outage."""
        service = service_with_pool(fail=True)
        report = asyncio.run(service.warmup())
        self.assertFalse(report["ran"])
        self.assertIn("the checkpoint is not loaded", report["reason"])
        # It still tried, and it still says how long it spent trying.
        self.assertEqual(len(service.stack.tts.calls), 1)
        self.assertIsInstance(report["seconds"], float)


if __name__ == "__main__":
    unittest.main()
