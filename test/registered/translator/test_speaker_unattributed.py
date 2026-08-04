# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The first turn of a session must survive having no speaker to attribute to.

THE DEFECT (#573, live twice after the 19:07 restart, once at 19:11:40 in
session 84ff8dd17a91 while the user was testing). `_identify` returns the
literal string ``"speaker-unknown"`` when nobody is known yet and the segment
cannot be embedded -- either because it is shorter than the embedder's
``min_seconds`` or because the embedder raised ``BackendError``. That string is
not an id: no profile with that key exists. `_run_turn_locked` then journals

    "reference_seconds": round(self.speakers.get(speaker_id)...) if speaker_id

and the guard is TRUTHY for a non-empty sentinel, so `SpeakerRegistry.get`
raises ``KeyError`` and the whole turn dies. The user says something, and
nothing comes back.

WHY THIS IS THE FIRST TURN. Both returns need an EMPTY registry -- with any
profile present, `_identify` attributes the segment to the most recently seen
speaker instead. So the failure is structurally a first-utterance failure,
which is the worst possible moment for it.

WHAT IS NOT DONE ABOUT IT, deliberately. The other way out is to mint a real
provisional profile (`speakers.create_speaker` is the one mint path that needs
no embedding, so it is technically available). That is refused: the user's
standing complaint on this feature is that the system opens speakers it should
not (#565, "er hätte eigentlich gar keinen weiteren speaker am anfang aufmachen
dürfen"), and minting a profile for a segment we could not even embed
manufactures exactly that spurious roster entry -- with no centroid, in a
roster whose entire purpose is matching. Fixing a crash by creating the symptom
next door is the wrong trade. The segment is simply NOT ATTRIBUTED, and that is
a fact about it rather than a gap to paper over.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \
        test/registered/translator/test_speaker_unattributed.py -v
"""

import unittest

from sglang.srt.translator.backends import AudioChunk, BackendError
from sglang.srt.translator.segmenter import Segment, SegmentReason
from sglang.srt.translator.session import UNATTRIBUTED_SPEAKER_ID, EventKind
from sglang.srt.translator.voices import PresetVoice, VoiceClass, VoicePool

from test_session import LANG_A, LANG_B, RATE, make_session, tone


def preset_pool():
    """A preset voice, because production has one and the point is "spoken".

    An unattributed speaker has no reference audio by construction, so the
    synthesizer falls back to the preset pool -- which `server.py:159` builds
    from `config.tts.preset_voice_dir` on the live server. Without a pool here
    the turn would complete SILENTLY and the test would be asserting something
    weaker than the user's experience.
    """
    clip = AudioChunk(tone(180.0, 4.0), RATE)
    return VoicePool([
        PresetVoice(
            voice_id="man-01", label="man-01", voice_class=VoiceClass.MAN,
            references={LANG_A: clip, LANG_B: clip},
            reference_texts={LANG_A: "a line", LANG_B: "a line"},
        )
    ])


def short_segment(seconds=0.2, frequency=140.0, index=0):
    """An utterance too short for the embedder's ``min_seconds`` (0.5 s).

    This is the real trigger, not a contrivance: a one-word answer at the
    start of a conversation is exactly this shape.
    """
    return Segment(
        audio=AudioChunk(tone(frequency, seconds), RATE),
        reason=SegmentReason.PAUSE,
        index=index,
        start_s=0.0,
    )


class TestAnUnattributedFirstTurnStillCompletes(unittest.IsolatedAsyncioTestCase):
    """The crash, driven exactly as the field hit it."""

    async def test_the_turn_completes_when_nobody_can_be_identified(self):
        """THE FALSIFIER. Fails with `KeyError: unknown speaker` unfixed.

        Empty registry plus a segment below `min_seconds` is the `:2036`
        return. The turn has to come out the other side.
        """
        session, _asr, _mt, _tts = make_session(voice_pool=preset_pool())
        self.assertEqual(
            len(session.speakers.profiles()), 0,
            "the registry must start empty or this does not test the defect",
        )
        result = await session._run_turn_locked(short_segment())
        self._assert_spoken(result)

    def _assert_spoken(self, result):
        """Completed is not enough -- the user has to HEAR something.

        A turn that returns a result but carries no audio has failed at the
        only thing the user can observe, so the assertion is on the audio and
        on the samples in it, not on the object being non-None.
        """
        self.assertIsNotNone(result, "the turn produced no result at all")
        self.assertTrue(
            result.audio, "the turn completed but synthesized nothing",
        )
        self.assertTrue(
            any(len(chunk.samples) for chunk in result.audio.values()),
            "the turn produced audio entries but every one of them is empty",
        )
        self.assertTrue(
            result.source_text, "the turn completed with no recognized text",
        )

    async def test_the_embedder_failure_arm_reaches_the_same_place(self):
        """`:2044` is the other return, and it is NOT covered by the first.

        The first test never calls the embedder; this one calls it and makes it
        raise. Two different branches reach the same sentinel, so a single
        falsifier would leave one of them untested -- and the two are reached
        by different conversations, not by the same one twice.
        """
        session, _asr, _mt, _tts = make_session(voice_pool=preset_pool())

        async def refuse(_audio):
            raise BackendError("embed", "no embedder today")

        session.embedder.embed = refuse
        # Long enough to PASS the min_seconds check, so the embedder is
        # actually reached; that is what makes this the other branch.
        result = await session._run_turn_locked(short_segment(seconds=1.0))
        self._assert_spoken(result)

    async def test_the_turn_is_journalled_with_no_reference_material(self):
        """An unattributed segment has no reference seconds, and says so.

        Zero here is a FACT -- there is no reference buffer, because there is
        no profile -- not a fallback standing in for a number we could not get.
        """
        session, _asr, _mt, _tts = make_session()
        await session._run_turn_locked(short_segment())
        events, _dropped = session.journal.since(0)
        speaker_events = [
            event for event in events if event.kind is EventKind.TURN_SPEAKER
        ]
        self.assertTrue(speaker_events, "no TURN_SPEAKER event was journalled")
        payload = speaker_events[-1].payload
        self.assertEqual(payload["speaker_id"], UNATTRIBUTED_SPEAKER_ID)
        self.assertEqual(payload["reference_seconds"], 0.0)

    async def test_no_speaker_is_minted_for_a_segment_we_could_not_embed(self):
        """#565: the roster must not grow a speaker nobody identified.

        This is the assertion that keeps the OTHER fix from being applied
        later by someone reading only the crash.
        """
        session, _asr, _mt, _tts = make_session()
        await session._run_turn_locked(short_segment())
        self.assertEqual(
            [p.speaker_id for p in session.speakers.profiles()], [],
            "an unattributable segment minted a speaker profile",
        )

    def test_the_sentinel_never_reaches_the_user_as_a_raw_id(self):
        """The client renders `speaker_label`, falling back to the raw id.

        `_speaker_label` catches `KeyError` and returns the id unchanged, so
        without a case for the sentinel the bubble is captioned with the
        literal string `speaker-unknown`. The label is user-facing text, and
        the app's chrome is lowercase English ("who spoke?", "translating"),
        which is what this matches -- there is no client-side localisation
        table to emit a token into.
        """
        session, _asr, _mt, _tts = make_session()
        label = session._speaker_label(UNATTRIBUTED_SPEAKER_ID)
        self.assertNotIn(
            UNATTRIBUTED_SPEAKER_ID, label,
            "the raw sentinel id is shown to the user",
        )
        self.assertTrue(label, "an unattributed speaker got an empty label")

    def test_an_id_that_should_exist_still_raises(self):
        """The guard is for the sentinel, not for every unknown id.

        A genuinely corrupt id reaching this path is an invariant violation and
        must stay loud. Swallowing it would turn a bug into a silent zero,
        which is the failure mode this whole change exists to remove.
        """
        session, _asr, _mt, _tts = make_session()
        with self.assertRaises(KeyError):
            session.speakers.get("speaker-that-never-existed")


if __name__ == "__main__":
    unittest.main()
