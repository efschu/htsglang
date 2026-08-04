# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Language before embedding, one speaker at a time, and the re-drive.

Every arm here has its control in the same class: the mechanism is switched
off (or the state it reacts to is removed) and the test asserts the OLD
behaviour comes back. An arm that cannot show the difference is not evidence
that the mechanism works -- it is evidence that something passed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sglang.srt.translator.backends import Transcript  # noqa: E402
from sglang.srt.translator.session import EventKind, run_conversation  # noqa: E402
from sglang.srt.translator.voices import (  # noqa: E402
    PresetVoice,
    VoiceClass,
    VoicePool,
)
from test_session import (  # noqa: E402
    LANG_A,
    LANG_B,
    VOICE_A_HZ,
    conversation_audio,
    make_session,
)


class TestTheFloor(unittest.IsolatedAsyncioTestCase):
    """A segment spoken OVER a running turn, in the other language."""

    def _segment(self, session, language, *, over):
        """One closed segment, stamped as if captured during `over`."""
        import dataclasses

        from sglang.srt.translator.segmenter import SegmentReason
        from sglang.srt.translator.backends import AudioChunk
        import numpy as np

        audio = AudioChunk(np.zeros(16000, dtype=np.float32), 16000)
        from sglang.srt.translator.segmenter import Segment

        seg = Segment(
            audio=audio, reason=SegmentReason.PAUSE, index=1, start_s=0.0
        )
        return dataclasses.replace(
            seg, overlapped_turn_id=over, overlapped_language=language
        )

    async def test_the_other_language_over_a_running_turn_is_ignored(self):
        session, _asr, _mt, _tts = make_session()
        segment = self._segment(session, LANG_A, over="t-running")
        transcript = Transcript(text="hola", language=LANG_B,
                                language_confidence=0.99)
        floor = session._foreign_interjection(segment, transcript)
        self.assertEqual(floor, LANG_A, "the interjection was not recognised")

    async def test_the_same_language_over_a_running_turn_is_not_an_interjection(self):
        """The person who holds the floor may keep talking."""
        session, _asr, _mt, _tts = make_session()
        segment = self._segment(session, LANG_A, over="t-running")
        transcript = Transcript(text="und weiter", language=LANG_A,
                                language_confidence=0.99)
        self.assertIsNone(session._foreign_interjection(segment, transcript))

    async def test_without_overlap_the_other_language_is_a_normal_turn(self):
        """THE CONTROL, and the whole reason the rule is scoped to overlap.

        Alternating DE/ES into one phone is what this app is for. If the floor
        were held by the pin rather than by an actually-running turn, every
        second turn of a real conversation would be discarded.
        """
        session, _asr, _mt, _tts = make_session()
        segment = self._segment(session, LANG_A, over=None)
        transcript = Transcript(text="hola", language=LANG_B,
                                language_confidence=0.99)
        self.assertIsNone(session._foreign_interjection(segment, transcript))

    async def test_the_discard_can_be_switched_off(self):
        session, _asr, _mt, _tts = make_session(overlap_discard=False)
        segment = self._segment(session, LANG_A, over="t-running")
        transcript = Transcript(text="hola", language=LANG_B,
                                language_confidence=0.99)
        self.assertIsNone(session._foreign_interjection(segment, transcript))

    async def test_a_third_language_is_a_recognition_problem_not_a_speaker(self):
        session, _asr, _mt, _tts = make_session()
        segment = self._segment(session, LANG_A, over="t-running")
        transcript = Transcript(text="bonjour", language="fr",
                                language_confidence=0.99)
        self.assertIsNone(session._foreign_interjection(segment, transcript))


class TestTheLanguagePrior(unittest.IsolatedAsyncioTestCase):

    async def test_an_unconfident_id_follows_the_pin_not_the_phantom(self):
        """The child case, in the shape the pin can actually reach.

        A pinned speaker whose language is known outranks a cluster the
        system minted for itself -- which is what routed German to German
        through the TTS echo's profile.
        """
        session, _asr, _mt, _tts = make_session()
        pinned = session.add_speaker("Papa")
        session.speakers.get(pinned).last_language = LANG_A
        session.arm_speaker(pinned)
        transcript = Transcript(text="ich bin muede", language=LANG_B,
                                language_confidence=0.2)
        self.assertEqual(session._resolve_source(transcript), LANG_A)

    async def test_a_confident_id_still_wins(self):
        """The control: the pin is a fallback, not an override."""
        session, _asr, _mt, _tts = make_session()
        pinned = session.add_speaker("Papa")
        session.speakers.get(pinned).last_language = LANG_A
        session.arm_speaker(pinned)
        transcript = Transcript(text="hola", language=LANG_B,
                                language_confidence=0.99)
        self.assertEqual(session._resolve_source(transcript), LANG_B)


class TestTheEchoLock(unittest.IsolatedAsyncioTestCase):

    async def test_audio_captured_while_we_speak_never_becomes_a_reference(self):
        import numpy as np

        from sglang.srt.translator.backends import AudioChunk
        from sglang.srt.translator.speakers import SpeakerRegistry

        registry = SpeakerRegistry()
        audio = AudioChunk(np.ones(16000 * 4, dtype=np.float32) * 0.2, 16000)
        embedding = np.ones(8, dtype=np.float32) / np.sqrt(8)
        _p, _s, admitted = registry.assign(
            embedding=embedding, audio=audio, text="hola",
            language="es", language_confidence=0.999, overlapped=True,
        )
        self.assertFalse(admitted, "the echo was admitted as a voice reference")

    async def test_an_unconfident_language_never_becomes_a_reference(self):
        """The echo scored 0.026 and enrolled itself carrying a cloned voice."""
        import numpy as np

        from sglang.srt.translator.backends import AudioChunk
        from sglang.srt.translator.speakers import SpeakerRegistry

        registry = SpeakerRegistry()
        audio = AudioChunk(np.ones(16000 * 4, dtype=np.float32) * 0.2, 16000)
        embedding = np.ones(8, dtype=np.float32) / np.sqrt(8)
        _p, _s, admitted = registry.assign(
            embedding=embedding, audio=audio, text="hola",
            language="es", language_confidence=0.026,
        )
        self.assertFalse(admitted)

    async def test_the_control_a_clean_first_utterance_is_admitted(self):
        """Without this the lock would be indistinguishable from a broken
        admission path -- the arm has to show that something CAN get in."""
        import numpy as np

        from sglang.srt.translator.backends import AudioChunk
        from sglang.srt.translator.speakers import SpeakerRegistry

        registry = SpeakerRegistry()
        audio = AudioChunk(np.ones(16000 * 4, dtype=np.float32) * 0.2, 16000)
        embedding = np.ones(8, dtype=np.float32) / np.sqrt(8)
        _p, _s, admitted = registry.assign(
            embedding=embedding, audio=audio, text="guten tag",
            language="de", language_confidence=0.98,
        )
        self.assertTrue(admitted)


def _mixed_pool():
    """One voice per class, ids chosen so the ALPHABET disagrees with the
    right answer -- `boy-01` sorts before `man-01`, which is exactly how an
    unclassified adult was handed a child's voice."""
    import numpy as np

    from sglang.srt.translator.backends import AudioChunk

    clip = AudioChunk(
        (0.2 * np.sin(np.arange(int(4.0 * 16000)) * 0.05)).astype(np.float32),
        16000,
    )
    return VoicePool([
        PresetVoice(voice_id=vid, label=vid, voice_class=cls,
                    references={LANG_A: clip, LANG_B: clip},
                    reference_texts={LANG_A: "a line", LANG_B: "a line"})
        for vid, cls in (
            ("boy-01", VoiceClass.BOY),
            ("girl-01", VoiceClass.GIRL),
            ("man-01", VoiceClass.MAN),
            ("woman-01", VoiceClass.WOMAN),
        )
    ])


class TestUnknownPresetOrder(unittest.TestCase):

    def test_an_unclassified_speaker_gets_an_adult_voice(self):
        """`matches()` is true whenever either side is UNKNOWN, so the
        candidate list is the WHOLE pool and the alphabet used to hand out
        `boy-01`. The user heard adults answer in children's voices."""
        pool = _mixed_pool()
        preset, _variant = pool.assign("speaker-1", None)
        self.assertIn(
            preset.voice_class, (VoiceClass.MAN, VoiceClass.WOMAN),
            f"an unclassified speaker was given {preset.voice_id!r}",
        )

    def test_a_child_still_gets_a_child_voice(self):
        """The control: the ordering must only bite where nothing is known."""
        pool = _mixed_pool()
        order = pool._order(VoiceClass.CHILD)
        child = sorted(
            [p for p in pool._presets
             if p.voice_class.matches(VoiceClass.CHILD)], key=order
        )
        self.assertTrue(child)
        self.assertIn(child[0].voice_class, (VoiceClass.BOY, VoiceClass.GIRL))


class TestRedrive(unittest.IsolatedAsyncioTestCase):

    async def _stalled(self):
        """A conversation with one message left mid-chain, as a drop produces."""
        session, _asr, _mt, tts = make_session()
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.4))])
        line = session.transcript.lines()[-1]
        # Exactly the state the user reported: words on file, nothing else.
        session.transcript.set_translations(line.line_id, {})
        session.transcript.get(line.line_id).translations.clear()
        return session, tts, line

    async def test_an_unfinished_message_is_driven_to_completion(self):
        session, _tts, line = await self._stalled()
        self.assertEqual(session.transcript.get(line.line_id).translations, {})
        result = await session.redrive(line.line_id)
        self.assertTrue(result["targets"], "the re-drive produced no target")
        self.assertTrue(
            session.transcript.get(line.line_id).translations,
            "the message is still stuck after a re-drive",
        )

    async def test_resume_heals_every_stuck_message(self):
        session, _tts, line = await self._stalled()
        self.assertEqual(
            [ln.line_id for ln in session.unfinished_lines()], [line.line_id]
        )
        await session.recover_unfinished()
        self.assertEqual(session.unfinished_lines(), [])

    async def test_a_finished_message_is_not_in_the_recovery_set(self):
        """The control. A recovery that re-speaks healthy messages would be
        worse than the defect it fixes."""
        session, _asr, _mt, _tts = make_session()
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.4))])
        self.assertEqual(session.unfinished_lines(), [])

    async def test_speaking_again_does_not_re_translate(self):
        """The words the reader has already seen must not change under them."""
        session, _asr, mt, _tts = make_session()
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.4))])
        line = session.transcript.lines()[-1]
        before = dict(line.translations)
        calls = len(mt.calls)
        result = await session.redrive(line.line_id, from_stage="tts")
        self.assertEqual(
            session.transcript.get(line.line_id).translations, before
        )
        self.assertEqual(len(mt.calls), calls, "MT was called again")
        self.assertTrue(result["spoke"])

    async def test_a_redrive_emits_a_turn_the_client_can_follow(self):
        session, _tts, line = await self._stalled()
        floor = session.journal.next_seq
        await session.redrive(line.line_id)
        kinds = [
            event.kind
            for event in session.journal.since(floor)[0]
        ]
        self.assertIn(EventKind.TURN_OPENED, kinds)
        self.assertIn(EventKind.TURN_DONE, kinds)


if __name__ == "__main__":
    unittest.main()
