# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The language restriction must bind the DECODE, not only the label.

DESIGN §17.8.8, read from the user's own session journal rather than from a
rig run. His wife spoke German into a ``de``/``es`` conversation:

===========================================  ===========================
faster-whisper raw identification            ``is`` (Icelandic), p = 0.99
``turn.transcript`` language                 ``de``
``turn.transcript`` text                     ``Það er ló. Ég finn.``
===========================================  ===========================

The whitelist fired exactly as designed and narrowed the LABEL to the
participant set. What it never touched is the decoding: ``transcribe()`` ran
without ``language=``, so Whisper identified freely across 98 languages and
decoded Icelandic. Relabelling Icelandic text as German does not make it
German, and the user reads the text.

This is the parameter-reach lesson in its purest form -- the mechanism exists,
is wired, is tested, demonstrably fires, and acts on a field the failure does
not run through. So this file pins the field that matters:

* the TEXT that comes back, which is what a person sees;
* the language the decoder was actually TOLD, which is the mechanism;
* the unrestricted path, which must still cost exactly one pass.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_decode_language.py -v
"""

import unittest

import numpy as np

from sglang.srt.translator.asr_backends import FasterWhisperAsr
from sglang.srt.translator.backends import AudioChunk

#: Whisper's posterior for that utterance: Icelandic overwhelming, German a
#: distant second. Taken from the journal's shape, not invented -- the point
#: is that constrained identification must pick `de` even when it is nowhere
#: near the top.
POSTERIOR = (("is", 0.990), ("de", 0.006), ("es", 0.002), ("en", 0.001))

#: What each decoding direction produces for the SAME audio. The Icelandic
#: line is the one his journal actually holds.
DECODED = {
    "is": "Það er ló. Ég finn.",
    "de": "Das ist alles, ich glaube.",
    "es": "Eso es todo, creo.",
}


class _Segment:
    def __init__(self, text):
        self.text = text


class _Info:
    def __init__(self, language, probability, posterior):
        self.language = language
        self.language_probability = probability
        self.all_language_probs = posterior


class IcelandicWhisper:
    """``faster_whisper.WhisperModel``'s surface, reproducing the real case.

    Free decoding follows its own identification and returns Icelandic;
    decoding with ``language=`` returns that language. That asymmetry is the
    entire defect, so the fake refuses to hide it.
    """

    def __init__(self):
        self.detect_calls = 0
        self.decode_languages = []

    def detect_language(self, audio):
        self.detect_calls += 1
        return "is", 0.99, list(POSTERIOR)

    def transcribe(self, audio, language=None, **kwargs):
        self.decode_languages.append(language)
        # None means "identify yourself and decode in whatever you find",
        # which for this audio is Icelandic.
        code = language or "is"
        probability = dict(POSTERIOR).get(code, 0.99)
        return (
            iter([_Segment(DECODED[code])]),
            _Info(code, probability, list(POSTERIOR)),
        )


class NoDetectWhisper:
    """A model too old to expose ``detect_language``.

    The fallback must stay on the single-pass path rather than crash: an
    exception here would take the whole turn down for a version difference.
    """

    def __init__(self):
        self.decode_languages = []

    def transcribe(self, audio, language=None, **kwargs):
        self.decode_languages.append(language)
        code = language or "is"
        return (
            iter([_Segment(DECODED[code])]),
            _Info(code, 0.99, list(POSTERIOR)),
        )


def whisper_asr(model, restrict=("de", "es")):
    """The SHIPPED adapter around a fake model.

    ``__new__`` rather than ``__init__`` on purpose: the constructor imports
    faster-whisper and loads weights, and this suite runs under
    ``CUDA_VISIBLE_DEVICES=99``. Everything below the constructor is the
    production code path.
    """
    asr = FasterWhisperAsr.__new__(FasterWhisperAsr)
    asr.name = "fake-whisper"
    asr._model = model
    asr._restrict = [str(c).lower() for c in restrict]
    asr._beam_size = 1
    asr._vad_filter = False
    return asr


def one_second():
    return AudioChunk(np.zeros(16000, dtype=np.float32), 16000)


class TestTheDecodeIsBound(unittest.IsolatedAsyncioTestCase):
    async def test_foreign_audio_comes_back_in_a_whitelist_language(self):
        """The falsifier: unfixed this returns `Það er ló. Ég finn.`."""
        model = IcelandicWhisper()
        transcript = await whisper_asr(model).transcribe(one_second())
        self.assertEqual(
            transcript.text, DECODED["de"],
            "the turn was decoded as %r; a de/es conversation must never "
            "show its user text in a language nobody selected"
            % (transcript.text,),
        )
        self.assertEqual(transcript.language, "de")

    async def test_the_decoder_was_told_the_chosen_language(self):
        """The mechanism, not just the outcome."""
        model = IcelandicWhisper()
        await whisper_asr(model).transcribe(one_second())
        self.assertEqual(model.detect_calls, 1)
        self.assertEqual(
            model.decode_languages, ["de"],
            "the decode ran with %r; identification chose `de` and the "
            "decode must follow it" % (model.decode_languages,),
        )

    async def test_the_label_and_the_text_agree(self):
        """The defect had them disagreeing in BOTH directions (§17.8.8)."""
        model = IcelandicWhisper()
        transcript = await whisper_asr(model).transcribe(one_second())
        self.assertEqual(transcript.text, DECODED[transcript.language])

    async def test_confidence_is_the_in_set_probability(self):
        """Not zero: the session's low-confidence fallback keys on this."""
        model = IcelandicWhisper()
        transcript = await whisper_asr(model).transcribe(one_second())
        self.assertAlmostEqual(transcript.language_confidence, 0.006, places=4)


class TestTheUnboundedPathIsUnchanged(unittest.IsolatedAsyncioTestCase):
    async def test_no_restriction_costs_exactly_one_pass(self):
        model = IcelandicWhisper()
        transcript = await whisper_asr(model, restrict=()).transcribe(one_second())
        self.assertEqual(
            model.detect_calls, 0,
            "an unbounded session must not pay for a second encoder pass",
        )
        self.assertEqual(model.decode_languages, [None])
        self.assertEqual(transcript.language, "is")
        self.assertEqual(transcript.text, DECODED["is"])

    async def test_a_model_without_detect_language_still_works(self):
        """Degrades to the old behaviour instead of raising."""
        model = NoDetectWhisper()
        transcript = await whisper_asr(model).transcribe(one_second())
        self.assertEqual(model.decode_languages, [None])
        # The label is still narrowed -- that half always worked.
        self.assertEqual(transcript.language, "de")


if __name__ == "__main__":
    unittest.main()
