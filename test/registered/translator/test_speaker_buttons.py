# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Speaker buttons: the user says who is speaking, and that settles it (§17.5).

The load-bearing test is ``test_arming_skips_identification_entirely``. It
gives the session an embedder that would confidently return the WRONG speaker,
and asserts the armed one wins. An implementation that identifies first and
overrules afterwards passes a naive "the right speaker got it" test and fails
this one -- and the difference is not academic: overruling still folds the
segment into the wrong centroid on the way past.

The second is ``test_a_manual_line_with_bad_audio_is_still_refused``. Manual
attribution is ground truth about WHO spoke; it is not a claim that the
microphone behaved. A human cannot vouch for a 0.3 s clipped fragment, and a
splice-length slice degrades that speaker's clone for the rest of the session.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_speaker_buttons.py -v
"""

import unittest

import numpy as np

from sglang.srt.translator.backends import AudioChunk
from sglang.srt.translator.session import EventKind, run_conversation
from sglang.srt.translator.speakers import (
    SpeakerEmbedding,
    SpeakerRegistry,
    SpeakerRegistryConfig,
)
from sglang.srt.translator.transcript_log import ORIGIN_AUTO, ORIGIN_MANUAL
from test_session import (  # noqa: E402  - sibling helper module
    RATE,
    VOICE_A_HZ,
    VOICE_B_HZ,
    conversation_audio,
    make_session,
    tone,
)


class LyingEmbedder:
    """Always returns the same vector, so automatic matching is deterministic.

    Used to prove that arming does not merely outrank identification but
    replaces it: with this embedder every utterance would land on one speaker.
    """

    name = "lying-embedder"
    min_seconds = 0.5

    def __init__(self):
        self.calls = 0

    async def embed(self, audio):
        self.calls += 1
        vector = np.ones(16, dtype=np.float32)
        return SpeakerEmbedding(vector)


class TestRegistryManualPath(unittest.TestCase):
    def setUp(self):
        self.registry = SpeakerRegistry(
            SpeakerRegistryConfig(min_slice_s=0.5, min_reference_rms=0.01),
            clock=lambda: 0.0,
        )

    def _audio(self, seconds=1.0, amplitude=0.3):
        return AudioChunk(tone(VOICE_A_HZ, seconds, amplitude=amplitude), RATE)

    def test_a_declared_speaker_has_no_centroid_until_they_speak(self):
        profile = self.registry.create_speaker()
        self.assertIsNone(profile.centroid)
        self.assertEqual(profile.observations, 0)
        # And a declared-but-unheard profile must not win the nearest-match
        # search, or every new voice would land on it.
        best_id, best_sim = self.registry._nearest(
            SpeakerEmbedding(np.ones(16, dtype=np.float32))
        )
        self.assertIsNone(best_id)
        self.assertEqual(best_sim, -1.0)

    def test_the_first_manual_audio_seeds_the_centroid(self):
        profile = self.registry.create_speaker()
        embedding = SpeakerEmbedding(np.ones(16, dtype=np.float32))
        seeded, admitted = self.registry.assign_manual(
            profile.speaker_id, self._audio(), embedding=embedding
        )
        self.assertIsNotNone(seeded.centroid)
        self.assertEqual(seeded.observations, 1)
        self.assertTrue(admitted)

    def test_manual_admission_bypasses_the_identity_threshold_only(self):
        # Two very different voices forced onto one profile: automatic
        # assignment would refuse the second as reference material, manual
        # accepts it, because identification is what the threshold guards.
        profile = self.registry.create_speaker()
        first = SpeakerEmbedding(np.ones(16, dtype=np.float32))
        second = SpeakerEmbedding(
            np.array([1.0] + [-1.0] * 15, dtype=np.float32)
        )
        self.registry.assign_manual(
            profile.speaker_id, self._audio(), embedding=first
        )
        _profile, admitted = self.registry.assign_manual(
            profile.speaker_id, self._audio(), embedding=second
        )
        self.assertTrue(admitted)

    def test_a_manual_line_with_bad_audio_is_still_refused(self):
        profile = self.registry.create_speaker()
        embedding = SpeakerEmbedding(np.ones(16, dtype=np.float32))
        # Too short.
        _p, admitted = self.registry.assign_manual(
            profile.speaker_id, self._audio(seconds=0.2), embedding=embedding
        )
        self.assertFalse(admitted)
        # Too quiet.
        _p, admitted = self.registry.assign_manual(
            profile.speaker_id,
            self._audio(seconds=2.0, amplitude=0.0001),
            embedding=embedding,
        )
        self.assertFalse(admitted)

    def test_arming_an_unknown_speaker_raises(self):
        with self.assertRaises(KeyError):
            self.registry.assign_manual("speaker-99", self._audio())


class TestSpeakerButtonsInSession(unittest.IsolatedAsyncioTestCase):
    async def test_arming_skips_identification_entirely(self):
        session, _asr, _mt, _tts = make_session()
        liar = LyingEmbedder()
        session.embedder = liar

        # One turn to create a speaker the automatic path would then match
        # everything to.
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        auto_id = session.transcript.lines()[-1].speaker_id

        new_id = session.add_speaker()
        self.assertNotEqual(new_id, auto_id)
        session.arm_speaker(new_id)
        calls_before = liar.calls

        await run_conversation(session, [conversation_audio((VOICE_B_HZ, 1.2))])
        line = session.transcript.lines()[-1]
        # The armed speaker won against an embedder that would have said
        # otherwise -- and it is the ORIGIN that proves identification was
        # skipped rather than overruled.
        self.assertEqual(line.speaker_id, new_id)
        self.assertEqual(line.origin, ORIGIN_MANUAL)
        # The embedder was still called once, to seed the centroid. That is
        # not identification: its answer was never compared to anything.
        self.assertEqual(liar.calls, calls_before + 1)
        self.assertIsNotNone(session.speakers.get(new_id).centroid)

    async def test_arming_lasts_exactly_one_utterance(self):
        session, _asr, _mt, _tts = make_session()
        new_id = session.add_speaker()
        session.arm_speaker(new_id)
        await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 1.2), (VOICE_B_HZ, 1.2))]
        )
        lines = session.transcript.lines()
        self.assertGreaterEqual(len(lines), 2)
        self.assertEqual(lines[0].speaker_id, new_id)
        self.assertEqual(lines[0].origin, ORIGIN_MANUAL)
        # The second utterance went through normal identification.
        self.assertEqual(lines[1].origin, ORIGIN_AUTO)
        self.assertIsNone(session.armed_speaker)

    async def test_tapping_the_lit_button_again_disarms(self):
        session, _asr, _mt, _tts = make_session()
        new_id = session.add_speaker()
        self.assertEqual(session.arm_speaker(new_id), new_id)
        self.assertEqual(session.arm_speaker(None), None)
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        self.assertEqual(session.transcript.lines()[-1].origin, ORIGIN_AUTO)

    async def test_arming_an_unknown_speaker_is_refused_not_remembered(self):
        session, _asr, _mt, _tts = make_session()
        with self.assertRaises(KeyError):
            session.arm_speaker("speaker-99")
        self.assertIsNone(session.armed_speaker)

    async def test_a_manual_attribution_is_never_badged_uncertain(self):
        session, _asr, _mt, _tts = make_session()
        new_id = session.add_speaker()
        session.arm_speaker(new_id)
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        line = session.transcript.lines()[-1]
        self.assertEqual(line.confidence, "exact")
        self.assertEqual(line.candidates, [])

    async def test_the_state_and_the_journal_show_the_armed_button(self):
        session, _asr, _mt, _tts = make_session()
        new_id = session.add_speaker()
        session.arm_speaker(new_id)
        self.assertEqual(session.state()["armed_speaker"], new_id)
        events = [
            event.payload
            for event in session.journal.since(0)[0]
            if event.kind is EventKind.SESSION_STATE
        ]
        self.assertTrue(any(e.get("speaker_added") == new_id for e in events))
        self.assertTrue(any(e.get("armed_speaker") == new_id for e in events))

        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        # ...and the consumption is visible too, so a client can unlight the
        # button from server state rather than guessing.
        consumed = [
            event.payload
            for event in session.journal.since(0)[0]
            if event.kind is EventKind.SESSION_STATE
            and event.payload.get("consumed_by") == new_id
        ]
        self.assertTrue(consumed)

    async def test_the_speaker_event_says_who_decided(self):
        session, _asr, _mt, _tts = make_session()
        new_id = session.add_speaker()
        session.arm_speaker(new_id)
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        speaker_events = [
            event.payload
            for event in session.journal.since(0)[0]
            if event.kind is EventKind.TURN_SPEAKER
        ]
        self.assertTrue(speaker_events)
        self.assertIs(speaker_events[-1]["manual"], True)


class TestSpeakerDeletion(unittest.IsolatedAsyncioTestCase):
    """A speaker created by mistake must be removable (user order)."""

    async def test_deleting_frees_the_slot_the_buffer_and_the_voice(self):
        from sglang.srt.translator.voices import VoiceMode, synthetic_pool
        from test_session import LANG_A, LANG_B

        session, _asr, _mt, _tts = make_session(
            voice_pool=synthetic_pool([LANG_A, LANG_B]),
            voice_mode=VoiceMode.PRESET,
        )
        sid = session.add_speaker("Ghost")
        session.arm_speaker(sid)
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        self.assertGreater(session.speakers.get(sid).reference_seconds(), 0.0)
        taken = session.voice_pool.assign(sid)[0].voice_id

        gone = session.delete_speaker(sid)
        self.assertEqual(gone["label"], "Ghost")
        self.assertGreater(gone["reference_seconds_released"], 0.0)
        with self.assertRaises(KeyError):
            session.speakers.get(sid)
        # The arming went with them, or the next utterance would be
        # attributed to somebody who no longer exists.
        self.assertIsNone(session.armed_speaker)
        # And the preset voice is back in circulation.
        self.assertTrue(session.voice_pool.release(sid) is False)
        reused = session.voice_pool.assign(session.add_speaker())[0].voice_id
        self.assertIn(reused, {taken, reused})

    async def test_history_is_not_rewritten_by_a_deletion(self):
        """What was said stays said.

        Reattributing old lines to somebody else would be a worse answer than
        a name with nobody behind it.
        """
        session, _asr, _mt, _tts = make_session()
        sid = session.add_speaker("Ghost")
        session.arm_speaker(sid)
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        line = session.transcript.lines()[0]
        self.assertEqual(line.speaker_id, sid)
        session.delete_speaker(sid)
        after = session.transcript.lines()[0]
        self.assertEqual(after.speaker_id, sid)
        self.assertEqual(after.source_text, line.source_text)

    async def test_deleting_an_unknown_speaker_is_refused(self):
        session, _asr, _mt, _tts = make_session()
        with self.assertRaises(KeyError):
            session.delete_speaker("speaker-99")


class TestManualSpeakerClass(unittest.IsolatedAsyncioTestCase):
    """The class must steer the VOICE, not just be stored (design 17.5b)."""

    def _session_with_pool(self):
        from sglang.srt.translator.voices import VoiceClass, VoiceMode, synthetic_pool
        from test_session import LANG_A, LANG_B

        session, _asr, _mt, _tts = make_session(
            voice_pool=synthetic_pool([LANG_A, LANG_B]),
            voice_mode=VoiceMode.PRESET,
        )
        return session, VoiceClass

    async def test_the_class_picks_the_voice_before_a_word_is_spoken(self):
        session, VoiceClass = self._session_with_pool()
        man = session.add_speaker("Papa", voice_class=VoiceClass.MAN)
        woman = session.add_speaker("Mama", voice_class=VoiceClass.WOMAN)
        boy = session.add_speaker("Moritz", voice_class=VoiceClass.BOY)

        # No audio has been seen at all; the preset must already be decided.
        chosen = {
            sid: session.voice_pool.assign(sid)[0]
            for sid in (man, woman, boy)
        }
        self.assertEqual(chosen[man].voice_class, VoiceClass.MAN)
        self.assertEqual(chosen[woman].voice_class, VoiceClass.WOMAN)
        self.assertIn(
            chosen[boy].voice_class, (VoiceClass.BOY, VoiceClass.CHILD)
        )
        # Three different people, three different voices.
        self.assertEqual(len({v.voice_id for v in chosen.values()}), 3)

    async def test_a_different_class_gives_a_different_voice(self):
        """The falsifier for a class that is stored but never reaches the pool."""
        session, VoiceClass = self._session_with_pool()
        as_boy = session.add_speaker(voice_class=VoiceClass.BOY)
        as_woman = session.add_speaker(voice_class=VoiceClass.WOMAN)
        self.assertNotEqual(
            session.voice_pool.assign(as_boy)[0].voice_id,
            session.voice_pool.assign(as_woman)[0].voice_id,
        )

    async def test_a_manual_name_is_never_overwritten_by_the_automatic_path(self):
        session, _asr, _mt, _tts = make_session()
        sid = session.add_speaker()
        session.name_speaker(sid, "Matthias")
        # The automatic path tries to rename and must be refused.
        self.assertEqual(session.name_speaker(sid, "Somebody", manual=False), [])
        self.assertEqual(session.speakers.get(sid).label, "Matthias")


if __name__ == "__main__":
    unittest.main()
