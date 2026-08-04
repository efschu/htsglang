# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Constrained detection must bound AUTO mode too, not only manual pairs.

Requirement 5 (addendum 5 v2) asks for language identification restricted to
the languages the conversation can actually route. It was implemented, wired
and tested -- and it reached almost nobody, because the whitelist was pushed
only from ``set_routing_pairs``. Auto mode is the default: a session opened
with ``--participants de,es`` and no explicit table pushed an EMPTY
restriction, which is correctly read as "no restriction", so the recognizer
was free to answer ``en`` for a German sentence. On a real device it did,
with English selected nowhere in the pair chooser.

That is the failure this file pins, and it is pinned twice on purpose:

* the WIRING -- what the session actually hands the recognizer -- because a
  whitelist pushed on a path most sessions never take does not exist;
* the OUTCOME -- through the production ``constrained_language_choice`` -- so
  the test fails if the decision function stops honouring what it is given,
  not only if the plumbing is unhooked.

The recognizer fake is deliberately hostile: it reports English as its top
answer with a plausible probability, the way a short or ambiguous German
opening genuinely does. Unfixed, the pipeline routes that turn as English.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_constrained_detection.py -v
"""

import unittest

from sglang.srt.translator.asr_backends import constrained_language_choice
from sglang.srt.translator.backends import Transcript
from sglang.srt.translator.session import run_conversation
from test_session import (  # noqa: E402  - sibling helper module
    VOICE_A_HZ,
    conversation_audio,
    make_session,
)

#: What a short, ambiguous German opening looks like to Whisper's identifier:
#: English on top, German a close second, and nothing else near.
AMBIGUOUS = (("en", 0.55), ("de", 0.40), ("nl", 0.03), ("es", 0.02))


class HostileAsr:
    """Reports English, then defers to the production restriction logic.

    It honours whatever whitelist the session pushes by calling the SHIPPED
    ``constrained_language_choice`` -- the same call the real faster-whisper
    backend makes -- so this test exercises the decision, not a reimplementation
    of it.
    """

    def __init__(self, languages):
        self.name = "hostile-asr"
        self._languages = tuple(languages)
        self.restrict_languages = ()
        self.pushes = []

    def supported_languages(self):
        return self._languages

    def set_restrict_languages(self, codes):
        self.restrict_languages = tuple(codes)
        self.pushes.append(tuple(codes))

    async def transcribe(self, audio, hint_language=None):
        language, probability = constrained_language_choice(
            "en", 0.55, AMBIGUOUS, self.restrict_languages
        )
        return Transcript(
            "Hallo, ich bin Matthias.", language, language_confidence=probability
        )


def hostile_session(**kwargs):
    kwargs.setdefault("participants", ("de", "es"))
    languages = tuple(kwargs["participants"])
    kwargs.setdefault("asr_languages", languages)
    kwargs.setdefault("tts_languages", languages)
    kwargs.setdefault("mt_languages", languages)
    session, _asr, mt, tts = make_session(**kwargs)
    session.asr = HostileAsr(languages + ("en",))
    return session


class TestAutoModeIsBounded(unittest.IsolatedAsyncioTestCase):
    async def test_the_participant_set_reaches_the_recognizer(self):
        session = hostile_session()
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        self.assertEqual(
            session.asr.restrict_languages, ("de", "es"),
            "auto mode pushed %r; a session with two participants and no "
            "manual table must still bound the identifier"
            % (session.asr.restrict_languages,),
        )

    async def test_english_is_impossible_when_nobody_chose_english(self):
        session = hostile_session()
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 1.2))]
        )
        self.assertTrue(results, "the turn must still complete")
        for result in results:
            self.assertIn(
                result.source_language, ("de", "es"),
                "the identifier answered %r for a German utterance in a "
                "de/es conversation" % (result.source_language,),
            )

    async def test_a_manual_table_still_wins_over_the_participants(self):
        session = hostile_session()
        session.set_routing_pairs([("de", "es")])
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        self.assertEqual(session.asr.restrict_languages, ("de", "es"))

    async def test_the_whitelist_is_refreshed_per_turn(self):
        """One recognizer serves every session, so the push cannot be once."""
        session = hostile_session()
        await run_conversation(
            session,
            [conversation_audio((VOICE_A_HZ, 1.2)),
             conversation_audio((VOICE_A_HZ, 1.2))],
        )
        self.assertGreaterEqual(
            len(session.asr.pushes), 2,
            "a whitelist pushed once is whichever session opened last",
        )


class TestTheDecisionItself(unittest.TestCase):
    """The production function, independent of the session wiring."""

    def test_an_out_of_set_top_answer_falls_back_in_set(self):
        language, probability = constrained_language_choice(
            "en", 0.55, AMBIGUOUS, ("de", "es")
        )
        self.assertEqual(language, "de")
        self.assertAlmostEqual(probability, 0.40, places=3)

    def test_an_empty_restriction_is_no_restriction(self):
        """The behaviour that made the bug silent, pinned so it stays known."""
        language, _ = constrained_language_choice("en", 0.55, AMBIGUOUS, ())
        self.assertEqual(language, "en")


if __name__ == "__main__":
    unittest.main()
