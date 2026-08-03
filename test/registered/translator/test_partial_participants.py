# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""A participant language this deployment can only partly serve (§17.8.14).

``TranslatorSession.__init__`` called ``conversation.validate_against(matrix)``,
which raises on the FIRST direction the deployment cannot serve. So adding a
language whose voice this deployment does not have did not lose that one
direction -- it refused to open the conversation at all, and every other
language the user had picked went with it. That is what kept the client's
ten-language picker shut behind ``PARTIAL_PARTICIPANTS_SUPPORTED``.

The turn path already degraded correctly: it calls ``matrix.require_pair`` per
target inside its loop and skips the ones that refuse. Only the constructor was
all-or-nothing.

``test_the_old_all_or_nothing_check_still_refuses`` is the can-fail proof: it
calls the strict ``validate_against`` on the same conversation and matrix and
asserts it raises, so the degradation is shown to be a real change of
behaviour rather than a matrix that was servable all along.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_partial_participants.py -v
"""

import unittest

from sglang.srt.translator.languages import (
    ConversationLanguages,
    LanguageError,
    LanguageMatrix,
)
from sglang.srt.translator.session import EventKind, run_conversation
from test_session import (  # noqa: E402  - sibling helper module
    LANG_A,
    LANG_B,
    VOICE_A_HZ,
    VOICE_B_HZ,
    conversation_audio,
    make_session,
)


def events(session, kind):
    return [e.payload for e in session.journal.since(0)[0] if e.kind is kind]


class TestPartialParticipants(unittest.IsolatedAsyncioTestCase):
    """LANG_B can be heard but not spoken: A->B is unservable, B->A is fine."""

    def _half_served(self):
        """Everything the arms need, with only the TTS side narrowed.

        Narrowing TTS rather than ASR is deliberate: it leaves BOTH languages
        recognisable, so the conversation still has a servable direction and
        the test is about degradation rather than about an empty matrix.
        """
        return dict(
            participants=(LANG_A, LANG_B),
            asr_languages=(LANG_A, LANG_B),
            tts_languages=(LANG_A,),
            mt_languages=(LANG_A, LANG_B),
        )

    def test_the_old_all_or_nothing_check_still_refuses(self):
        """THE CAN-FAIL PROOF. Same conversation, same matrix, strict check."""
        setup = self._half_served()
        matrix = LanguageMatrix.from_backends(
            asr_languages=setup["asr_languages"],
            tts_languages=setup["tts_languages"],
            mt_languages=setup["mt_languages"],
        )
        conversation = ConversationLanguages.of(list(setup["participants"]))
        with self.assertRaises(LanguageError) as caught:
            conversation.validate_against(matrix)
        self.assertIn("TTS cannot speak", str(caught.exception))
        # And the degrading report on the SAME inputs keeps one direction.
        servable, unroutable = conversation.direction_report(matrix)
        self.assertEqual(servable, ((LANG_B, LANG_A),))
        self.assertEqual(list(unroutable), [(LANG_A, LANG_B)])

    async def test_the_session_opens_instead_of_refusing(self):
        session, _asr, _mt, _tts = make_session(**self._half_served())
        self.assertEqual(
            [(s, t) for s, t in session.unroutable], [(LANG_A, LANG_B)]
        )
        reported = session.state()["unroutable_directions"]
        self.assertEqual(len(reported), 1)
        self.assertEqual(reported[0]["source"], LANG_A)
        self.assertEqual(reported[0]["target"], LANG_B)
        # The reason names the STAGE, or the user cannot act on it.
        self.assertIn("TTS cannot speak", reported[0]["reason"])
        # And the picker's gate is announced, so a client deployed ahead of
        # this server does not offer a language the server would refuse.
        self.assertTrue(session.state()["supports"]["partial_participants"])

    async def test_the_unservable_direction_is_tagged_not_errored(self):
        """A turn into the missing voice is marked, and does not look broken.

        `turn.unrouted` puts the reason in the bubble where the translation
        would have been. An ERROR per utterance would make a conversation that
        is working in one direction look broken in both.
        """
        session, _asr, _mt, _tts = make_session(**self._half_served())
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.4))])
        unrouted = events(session, EventKind.TURN_UNROUTED)
        self.assertEqual(len(unrouted), 1, unrouted)
        self.assertEqual(unrouted[0]["source"], LANG_A)
        self.assertEqual(unrouted[0]["target"], LANG_B)
        self.assertIn("TTS cannot speak", unrouted[0]["reason"])
        # Not an error, and the line itself is still in the transcript: what
        # was said is recorded even where it could not be spoken onward.
        self.assertEqual(events(session, EventKind.ERROR), [])
        self.assertEqual(len(session.transcript.lines()), 1)

    async def test_the_servable_direction_still_works_in_the_same_session(self):
        """The point of degrading: the other half of the conversation runs."""
        session, _asr, _mt, _tts = make_session(**self._half_served())
        results = await run_conversation(
            session, [conversation_audio((VOICE_B_HZ, 1.4))]
        )
        self.assertEqual(len(results), 1, [r.source_text for r in results])
        self.assertEqual(results[0].source_language, LANG_B)
        self.assertEqual(list(results[0].translations), [LANG_A])
        self.assertEqual(events(session, EventKind.TURN_UNROUTED), [])

    async def test_a_conversation_with_no_servable_direction_is_still_refused(self):
        """Degrading is not the same as accepting anything.

        With no voice for either participant nothing can be spoken in either
        direction; that is not a degraded conversation, it is not one at all,
        and opening it would trade a clear refusal for a session that silently
        tags every single turn.
        """
        with self.assertRaises(LanguageError) as caught:
            make_session(
                participants=(LANG_A, LANG_B),
                asr_languages=(LANG_A, LANG_B),
                tts_languages=(),
                mt_languages=(LANG_A, LANG_B),
            )
        self.assertIn("cannot serve any direction", str(caught.exception))


class TestFullySupportedPairIsUnchanged(unittest.IsolatedAsyncioTestCase):
    """THE CONTROL ARM. The reference pair must behave exactly as before.

    The degradation changes the constructor of every session in the
    deployment, so the arm that matters most is the one proving the normal
    path did not move: nothing unroutable, nothing tagged, both directions
    translated.
    """

    async def test_nothing_is_unroutable_and_nothing_is_tagged(self):
        session, _asr, mt, _tts = make_session()
        self.assertEqual(session.unroutable, {})
        self.assertEqual(session.state()["unroutable_directions"], [])
        results = await run_conversation(
            session,
            [conversation_audio((VOICE_A_HZ, 2.0), (VOICE_B_HZ, 2.0))],
        )
        self.assertEqual(len(results), 2, [r.source_text for r in results])
        self.assertEqual(events(session, EventKind.TURN_UNROUTED), [])
        self.assertEqual(events(session, EventKind.ERROR), [])
        self.assertEqual([(s, t) for _text, s, t in mt.calls],
                         [(LANG_A, LANG_B), (LANG_B, LANG_A)])


if __name__ == "__main__":
    unittest.main()
