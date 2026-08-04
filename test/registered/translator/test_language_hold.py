# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The loaded gun, and the two mechanisms that unload it.

FOUND LIVE, not theorised (§17.8.23). After a session in which the user read
the Spanish output back into the microphone to check it, the server state read
`speaker-1.last_language = "es"` for a German speaker, with a sticky pin held
on that speaker. `_resolve_source` consults the pin below the confidence
floor, so his next quiet German utterance would have been routed `es->de` and
spoken back at him in German.

Two independent defects made that state reachable and two mechanisms answer
them:

  * OUR OWN OUTPUT rewrote his language history. It is not evidence about
    which language he speaks -- we wrote those words.
  * A SHORT segment could turn the whole conversation around on its own. The
    child's 0.88 s turn did exactly that at language confidence 1.000, which
    is why a confidence gate cannot be the answer.

Every arm has its control in the same class. The controls matter more than
usual here, because both mechanisms REFUSE to do something and a refusal that
fires too widely breaks alternating DE/ES, which is the entire product.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sglang.srt.translator.backends import Transcript  # noqa: E402
from sglang.srt.translator.session import EventKind  # noqa: E402
from test_session import LANG_A, LANG_B, make_session  # noqa: E402


def teach(session, speaker_id, language, turns):
    """Give a speaker a consecutive-language history, as the registry would."""
    profile = session.speakers.get(speaker_id)
    profile.last_language = language
    profile.language_streak = turns
    return profile


def held_events(session):
    events, _gap = session.journal.since(0)
    return [e for e in events if e.kind is EventKind.TURN_LANGUAGE_HELD]


class TestShortSegmentCannotFlipTheDirection(unittest.IsolatedAsyncioTestCase):
    def _session(self, **kwargs):
        session, _asr, _mt, _tts = make_session(**kwargs)
        session.speakers.create_speaker("speaker-1")
        session.arm_speaker("speaker-1")
        teach(session, "speaker-1", LANG_A, turns=3)
        return session

    def _short(self):
        # The measured shape: confident, wrong, and far too short to carry a
        # language switch. A confidence gate cannot reach this.
        return Transcript(text="Gracias por ver el video",
                          language=LANG_B, language_confidence=1.0)

    async def test_a_confident_short_contrary_segment_does_not_flip(self):
        session = self._session()
        source = session._resolve_source(
            self._short(), "speaker-1", duration_s=0.88
        )
        self.assertEqual(
            source, LANG_A,
            "a 0.88s segment turned the conversation around at confidence 1.0",
        )
        self.assertEqual(len(held_events(session)), 1, "the refusal was silent")
        payload = held_events(session)[0].payload
        self.assertEqual(payload["detected"], LANG_B)
        self.assertEqual(payload["held"], LANG_A)
        self.assertEqual(payload["streak"], 3)

    async def test_a_long_segment_in_the_other_language_still_flips(self):
        """THE CONTROL, and the product depends on it.

        Alternating DE/ES into one phone is what this app is for. A rule that
        made switching hard would break the common case to protect an edge.
        """
        session = self._session()
        source = session._resolve_source(
            self._short(), "speaker-1", duration_s=4.0
        )
        self.assertEqual(source, LANG_B, "a full utterance was refused its language")
        self.assertEqual(held_events(session), [])

    async def test_without_a_pin_a_short_segment_still_flips(self):
        """The rule needs the user's assertion of WHO is speaking."""
        session = self._session()
        session.arm_speaker(None)
        source = session._resolve_source(
            self._short(), "speaker-1", duration_s=0.88
        )
        self.assertEqual(source, LANG_B)
        self.assertEqual(held_events(session), [])

    async def test_without_a_streak_a_short_segment_still_flips(self):
        """One turn in a language is a coincidence, not a history."""
        session = self._session()
        teach(session, "speaker-1", LANG_A, turns=1)
        source = session._resolve_source(
            self._short(), "speaker-1", duration_s=0.88
        )
        self.assertEqual(source, LANG_B)
        self.assertEqual(held_events(session), [])

    async def test_a_short_segment_agreeing_with_history_is_untouched(self):
        session = self._session()
        agreeing = Transcript(text="und weiter", language=LANG_A,
                              language_confidence=1.0)
        source = session._resolve_source(agreeing, "speaker-1", duration_s=0.5)
        self.assertEqual(source, LANG_A)
        self.assertEqual(held_events(session), [],
                         "nothing was refused -- there was no disagreement")

    async def test_the_rule_can_be_switched_off(self):
        """THE CONTROL FOR THE MECHANISM ITSELF."""
        session = self._session(short_segment_flip_s=0.0)
        source = session._resolve_source(
            self._short(), "speaker-1", duration_s=0.88
        )
        self.assertEqual(source, LANG_B)


class TestReadBackDoesNotRewriteHistory(unittest.IsolatedAsyncioTestCase):
    """Our own output, spoken back into the microphone, is not evidence."""

    def _session(self):
        session, _asr, _mt, _tts = make_session()
        session._spoken.append((LANG_B, "Adicional, se ha eliminado "
                                        "el desbordamiento silencioso."))
        return session

    async def test_our_own_sentence_is_recognised_when_read_back(self):
        session = self._session()
        # As it comes back: through a loudspeaker, a room and a recognizer.
        heard = "adicional se ha eliminado el desbordamiento silencioso"
        self.assertEqual(session._is_read_back(heard), LANG_B)
        self.assertEqual(session.read_backs_seen, 1)

    async def test_a_different_sentence_in_the_same_language_is_not_a_read_back(self):
        """THE CONTROL: the guard must not swallow real speech.

        If any Spanish sentence counted as our own, a genuine Spanish speaker
        would never build a language history at all.
        """
        session = self._session()
        self.assertIsNone(
            session._is_read_back("Buenos dias, quisiera dos cafes por favor")
        )
        self.assertEqual(session.read_backs_seen, 0)

    async def test_a_short_utterance_is_never_a_read_back(self):
        """"si" would match half the conversation."""
        session = self._session()
        session._spoken.append((LANG_B, "Si."))
        self.assertIsNone(session._is_read_back("si"))

    async def test_the_registry_keeps_its_history_on_a_read_back(self):
        """The end-to-end property: the poisoning path is closed."""
        import numpy as np

        from sglang.srt.translator.backends import AudioChunk

        session, _asr, _mt, _tts = make_session()
        audio = AudioChunk(np.zeros(32000, dtype=np.float32), 16000)
        profile = session.speakers.create_speaker("speaker-1")
        profile.last_language = LANG_A
        profile.language_streak = 2

        session.speakers.assign_manual(
            speaker_id="speaker-1", audio=audio, embedding=None,
            text="una frase en espanol", language=LANG_B, read_back=True,
        )
        self.assertEqual(
            profile.last_language, LANG_A,
            "reading our own output back rewrote the speaker's language",
        )
        self.assertEqual(profile.language_streak, 2)

    async def test_without_the_flag_the_history_moves(self):
        """THE CONTROL: genuine speech in a new language still counts."""
        import numpy as np

        from sglang.srt.translator.backends import AudioChunk

        session, _asr, _mt, _tts = make_session()
        audio = AudioChunk(np.zeros(32000, dtype=np.float32), 16000)
        profile = session.speakers.create_speaker("speaker-1")
        profile.last_language = LANG_A
        profile.language_streak = 2

        session.speakers.assign_manual(
            speaker_id="speaker-1", audio=audio, embedding=None,
            text="una frase en espanol", language=LANG_B, read_back=False,
        )
        self.assertEqual(profile.last_language, LANG_B)
        self.assertEqual(profile.language_streak, 1,
                         "a language change must restart the streak, not extend it")


class TestTheManualDecisionRecordsItsPin(unittest.IsolatedAsyncioTestCase):
    """The instrument gap from §17.8.21: manual records always read pin=None.

    `assign_manual` is only ever reached with an armed speaker, so it is the
    one outcome where a pin was CERTAINLY held -- and it was the one outcome
    recording nothing, which made the collector report "0 of 7 decisions with
    a pin held" for a session the pin decided almost entirely.
    """

    async def test_a_manual_attribution_records_the_pin_it_acted_on(self):
        import numpy as np

        from sglang.srt.translator.backends import AudioChunk

        session, _asr, _mt, _tts = make_session()
        audio = AudioChunk(np.zeros(32000, dtype=np.float32), 16000)
        session.speakers.create_speaker("speaker-1")
        session.speakers.assign_manual(
            speaker_id="speaker-1", audio=audio, embedding=None,
            text="hallo", language=LANG_A,
        )
        records = [r for r in session.speakers.decisions
                   if r["outcome"] == "manual"]
        self.assertTrue(records, "no manual decision was recorded")
        self.assertEqual(records[-1]["pin"], "speaker-1")


if __name__ == "__main__":
    unittest.main()
