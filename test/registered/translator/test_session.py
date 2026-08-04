# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The turn pipeline: routing by what was heard, and voice preservation.

The two claims this feature makes are tested here as claims, not as call
counts:

* the translation direction follows the LANGUAGE THE RECOGNIZER DETECTED,
  never a configured default -- asserted by driving the same session with two
  speakers in two languages and checking each turn went the other way;
* each speaker's translated audio carries THEIR OWN voice -- asserted through
  the fake synthesizer, which copies the reference audio's pitch into its
  output, so "speaker B's turn came out in speaker A's voice" is a numeric
  failure rather than something a human has to listen for.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_session.py -v
"""

import math
import unittest

import numpy as np

from sglang.srt.translator.backends import (
    AudioChunk,
    FakeAsr,
    FakeEmbedder,
    FakeMt,
    FakeTts,
)
from sglang.srt.translator.languages import ConversationLanguages, LanguageMatrix
from sglang.srt.translator.segmenter import SegmenterConfig
from sglang.srt.translator.session import (
    EventKind,
    Journal,
    SessionManager,
    TranslatorSession,
    run_conversation,
)
from sglang.srt.translator.speakers import SpeakerRegistryConfig

RATE = 16000
FRAME_MS = 20

# Two synthetic "voices", far enough apart in pitch that the fake embedder
# separates them and the fake synthesizer's pitch copy is unambiguous.
VOICE_A_HZ = 140.0
VOICE_B_HZ = 240.0

# Two synthetic languages. Not 'de'/'es': the pipeline must not be able to
# recognise the development pair, and using invented codes proves it.
LANG_A = "aa"
LANG_B = "bb"


def tone(frequency, seconds, rate=RATE, amplitude=0.3):
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (amplitude * np.sin(2.0 * math.pi * frequency * t)).astype(np.float32)


def silence(seconds, rate=RATE):
    return np.zeros(int(seconds * rate), dtype=np.float32)


def dominant_hz(audio):
    spectrum = np.abs(np.fft.rfft(audio.samples))
    freqs = np.fft.rfftfreq(len(audio.samples), 1.0 / audio.sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


class ScriptedVad:
    """Speech wherever the sample energy is non-trivial. Deterministic."""

    frame_ms = FRAME_MS

    def is_speech(self, frame, sample_rate):
        del sample_rate
        return bool(np.sqrt(np.mean(np.square(frame.astype(np.float64)))) > 0.02)

    def reset(self):
        pass


class _DriftingEmbedder:
    """Returns a different vector every call -- short-window instability.

    What the real embedder does on 1.5 s windows: adjacent windows of ONE
    voice came back at 0.392 similarity, below the 0.62 meant to separate two
    PEOPLE. A fake that always agrees with itself cannot reproduce that, which
    is why the tone-based fakes never caught this.
    """

    min_seconds = 0.5
    name = "drifting"

    def __init__(self):
        self._n = 0

    async def embed(self, audio):
        from sglang.srt.translator.speakers import SpeakerEmbedding

        self._n += 1
        vector = np.zeros(8, dtype=np.float32)
        vector[self._n % 8] = 1.0
        return SpeakerEmbedding(vector)


def make_session(
    script=None,
    participants=(LANG_A, LANG_B),
    tts_languages=(LANG_A, LANG_B),
    asr_languages=(LANG_A, LANG_B),
    mt_languages=None,
    min_reference_seconds=1.0,
    max_queued_turns=2,
    session_id="s1",
    **kwargs,
):
    asr = FakeAsr(
        languages=asr_languages,
        script=script,
        pitch_map=[(VOICE_A_HZ, LANG_A), (VOICE_B_HZ, LANG_B)],
    )
    embedder = FakeEmbedder(min_seconds=0.5)
    mt = FakeMt(languages=mt_languages)
    tts = FakeTts(
        languages=tts_languages,
        sample_rate=RATE,
        min_reference_seconds=min_reference_seconds,
        chunk_seconds=0.2,
        seconds_per_char=0.01,
    )
    matrix = LanguageMatrix.from_backends(
        asr_languages=asr_languages,
        tts_languages=tts_languages,
        mt_languages=mt_languages,
    )
    session = TranslatorSession(
        session_id=session_id,
        asr=asr,
        embedder=embedder,
        mt=mt,
        tts=tts,
        matrix=matrix,
        conversation=ConversationLanguages.of(list(participants)),
        segmenter_config=SegmenterConfig(
            sample_rate=RATE,
            frame_ms=FRAME_MS,
            onset_ms=40,
            hangover_ms=100,
            pre_roll_ms=40,
            min_utterance_ms=200,
            max_utterance_s=30.0,
        ),
        speaker_config=SpeakerRegistryConfig(
            min_slice_s=0.5, rolling_prompt_s=8.0, max_slice_s=6.0
        ),
        vad=ScriptedVad(),
        journal=Journal(max_events=256, max_audio_bytes=8 << 20),
        min_reference_seconds=min_reference_seconds,
        max_queued_turns=max_queued_turns,
        **kwargs,
    )
    return session, asr, mt, tts


def conversation_audio(*turns):
    """Build one stream: (frequency, seconds) turns separated by silence."""
    parts = [silence(0.3)]
    for frequency, seconds in turns:
        parts.append(tone(frequency, seconds))
        parts.append(silence(0.4))
    return AudioChunk(np.concatenate(parts), RATE)


class TestDirectionRouting(unittest.IsolatedAsyncioTestCase):
    async def test_direction_follows_the_detected_language_per_turn(self):
        session, _asr, mt, _tts = make_session()
        audio = conversation_audio((VOICE_A_HZ, 2.0), (VOICE_B_HZ, 2.0))
        results = await run_conversation(session, [audio])
        self.assertEqual(len(results), 2, [r.source_text for r in results])
        # Turn 1 was heard as LANG_A, so it must have gone to LANG_B, and the
        # reverse for turn 2. Neither direction is configured anywhere.
        self.assertEqual(results[0].source_language, LANG_A)
        self.assertEqual(list(results[0].translations), [LANG_B])
        self.assertEqual(results[1].source_language, LANG_B)
        self.assertEqual(list(results[1].translations), [LANG_A])
        # The MT backend saw exactly those directions.
        self.assertEqual([(s, t) for _text, s, t in mt.calls],
                         [(LANG_A, LANG_B), (LANG_B, LANG_A)])

    async def test_a_three_language_conversation_fans_out(self):
        session, _asr, _mt, _tts = make_session(
            script=[("hello", LANG_A)],
            participants=(LANG_A, LANG_B, "cc"),
            asr_languages=(LANG_A, LANG_B, "cc"),
            tts_languages=(LANG_A, LANG_B, "cc"),
        )
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.0))]
        )
        self.assertEqual(sorted(results[0].translations), [LANG_B, "cc"])

    async def test_low_confidence_falls_back_to_the_speaker_history(self):
        # First turn confident in LANG_A; second turn from the same voice is
        # unconfident, and must NOT be allowed to flip the direction.
        asr = FakeAsr(
            languages=(LANG_A, LANG_B),
            script=[("first", LANG_A), ("second", LANG_B)],
            confidence=1.0,
        )
        session, _a, mt, _t = make_session(script=[("first", LANG_A)])
        session.asr = asr
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 2.0))])
        # Re-arm with a low-confidence identification of the other language.
        session.asr = FakeAsr(
            languages=(LANG_A, LANG_B),
            script=[("second", LANG_B)],
            confidence=0.1,
        )
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 2.0))])
        directions = [(s, t) for _text, s, t in mt.calls]
        self.assertEqual(
            directions[-1],
            (LANG_A, LANG_B),
            "an unconfident language ID reversed the conversation",
        )

    async def test_an_unroutable_pair_is_tagged_not_guessed(self):
        """The direction is never guessed -- but it no longer takes the
        session down with it (§17.8.14).

        This arm used to assert that constructing the session RAISED. That was
        the all-or-nothing constructor check, and it meant one participant
        language without a voice refused the whole conversation, including the
        direction the deployment could serve perfectly. The session now opens
        on what it can serve and tags the rest.

        What has NOT changed, and is the part this test is named for: nothing
        is guessed. The unservable direction produces no translation and no
        substituted target -- it produces a tagged turn naming the stage.
        """
        session, _asr, mt, _tts = make_session(tts_languages=(LANG_A,))
        self.assertEqual(
            [(s, t) for s, t in session.unroutable], [(LANG_A, LANG_B)]
        )
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.4))])
        unrouted = [
            e.payload for e in session.journal.since(0)[0]
            if e.kind is EventKind.TURN_UNROUTED
        ]
        self.assertEqual(len(unrouted), 1, unrouted)
        self.assertIn("TTS cannot speak", unrouted[0]["reason"])
        # Not guessed: no translation was requested in any direction for it.
        self.assertEqual(mt.calls, [])


class TestVoicePreservation(unittest.IsolatedAsyncioTestCase):
    async def test_each_speaker_hears_their_own_voice_back(self):
        session, _asr, _mt, _tts = make_session()
        audio = conversation_audio(
            (VOICE_A_HZ, 2.5), (VOICE_B_HZ, 2.5), (VOICE_A_HZ, 2.5)
        )
        results = await run_conversation(session, [audio])
        self.assertEqual(len(results), 3)
        # Two distinct speakers were found, and the third turn rejoined the
        # first speaker rather than minting a third.
        self.assertEqual(len(session.speakers), 2)
        self.assertEqual(results[0].speaker_id, results[2].speaker_id)
        self.assertNotEqual(results[0].speaker_id, results[1].speaker_id)

        # The synthesized pitch must match the SPEAKER, not the previous turn.
        for result, expected_hz in (
            (results[0], VOICE_A_HZ),
            (results[1], VOICE_B_HZ),
            (results[2], VOICE_A_HZ),
        ):
            if result.used_fallback_voice:
                continue  # the cold-start case has its own test
            target = next(iter(result.audio))
            self.assertAlmostEqual(
                dominant_hz(result.audio[target]),
                expected_hz,
                delta=12.0,
                msg=f"turn {result.turn_id} came out in the wrong voice",
            )

    async def test_the_first_turn_borrows_a_voice_and_says_so(self):
        # min_reference_seconds above what one turn can supply: the very first
        # speaker has nothing to clone from and must still be translated.
        session, _asr, _mt, _tts = make_session(min_reference_seconds=30.0)
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.0))]
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].used_fallback_voice)
        # No other speaker exists, so there is no voice to borrow and no audio
        # is produced -- but the TRANSLATION still exists, which is the point.
        self.assertTrue(results[0].translations)

    async def test_enrollment_gives_the_user_their_voice_from_turn_one(self):
        session, _asr, _mt, _tts = make_session(min_reference_seconds=3.0)
        await session.enroll_speaker(
            label="user",
            audio=AudioChunk(tone(VOICE_A_HZ, 6.0), RATE),
            text="enrollment sample",
            language=LANG_A,
        )
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.0))]
        )
        self.assertEqual(len(results), 1)
        self.assertFalse(
            results[0].used_fallback_voice,
            "an enrolled speaker must not need a borrowed voice",
        )
        self.assertEqual(results[0].speaker_id, "enrolled:user")
        target = next(iter(results[0].audio))
        self.assertAlmostEqual(
            dominant_hz(results[0].audio[target]), VOICE_A_HZ, delta=12.0
        )


class TestJournalAndResume(unittest.IsolatedAsyncioTestCase):
    async def test_every_stage_leaves_an_observable_event(self):
        session, _asr, _mt, _tts = make_session()
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 2.5))])
        kinds = [e.kind for e, in ((e,) for e in session.journal.since(0)[0])]
        for expected in (
            EventKind.SESSION_READY,
            EventKind.TURN_OPENED,
            EventKind.TURN_TRANSCRIPT,
            EventKind.TURN_SPEAKER,
            EventKind.TURN_TRANSLATION,
            EventKind.TURN_AUDIO,
            EventKind.TURN_DONE,
        ):
            self.assertIn(expected, kinds, expected)

    async def test_replay_from_a_cursor_returns_only_what_was_missed(self):
        session, _asr, _mt, _tts = make_session()
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 2.5))])
        cursor = session.journal.next_seq
        await run_conversation(session, [conversation_audio((VOICE_B_HZ, 2.5))])
        events, gap = session.journal.since(cursor)
        self.assertFalse(gap)
        self.assertTrue(events)
        self.assertTrue(all(e.seq >= cursor for e in events))

    async def test_a_cursor_below_the_floor_reports_a_gap(self):
        session, _asr, _mt, _tts = make_session()
        session.journal._events.clear()
        for _ in range(300):
            session.journal.append(EventKind.SESSION_STATE, {"tick": True})
        self.assertGreater(session.journal.floor, 0)
        _events, gap = session.journal.since(0)
        self.assertTrue(gap, "an unrecoverable cursor must be reported, not hidden")

    async def test_audio_is_evicted_under_the_byte_budget_but_the_event_stays(self):
        session, _asr, _mt, _tts = make_session()
        session.journal._max_audio_bytes = 4096
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 3.0))])
        audio_events = [
            e for e in session.journal.since(0)[0] if e.kind is EventKind.TURN_AUDIO
        ]
        self.assertTrue(audio_events)
        evicted = [e for e in audio_events if e.audio is None]
        self.assertTrue(evicted, "the audio byte budget was not enforced")
        self.assertTrue(all(e.payload.get("audio_evicted") for e in evicted))
        self.assertLessEqual(session.journal.audio_bytes, 4096)

    async def test_reconnect_keeps_the_speakers_and_resets_the_stream(self):
        session, _asr, _mt, _tts = make_session()
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 2.5))])
        speakers_before = session.speakers.to_json()
        reference_before = session.speakers.profiles()[0].reference_seconds()

        session.on_reconnect()

        self.assertEqual(session.speakers.to_json(), speakers_before)
        self.assertEqual(
            session.speakers.profiles()[0].reference_seconds(),
            reference_before,
            "a reconnect discarded the reference buffer it exists to protect",
        )
        self.assertFalse(session.segmenter.speaking)
        # And the conversation continues with the same speaker recognised.
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.5))]
        )
        self.assertEqual(results[0].speaker_id, speakers_before[0]["speaker_id"])


class TestIntraSegmentSpeakerSplit(unittest.IsolatedAsyncioTestCase):
    """Two people back-to-back with no pause land in ONE VAD segment.

    Left alone, that segment gets one embedding and one of the two speakers
    contributes their voice to the other's reference buffer -- a poisoned
    buffer is audible in every later turn by that speaker, which is the
    expensive kind of mistake. These are the falsifiers for the re-cut.
    """

    async def test_a_two_speaker_segment_is_re_cut_into_two_turns(self):
        session, _asr, mt, _tts = make_session()
        # No silence between the voices: one continuous 6 s segment.
        run_on = AudioChunk(
            np.concatenate(
                [silence(0.3), tone(VOICE_A_HZ, 3.0), tone(VOICE_B_HZ, 3.0),
                 silence(0.4)]
            ),
            RATE,
        )
        results = await run_conversation(session, [run_on])
        self.assertEqual(len(results), 2, "the run-on segment was not re-cut")
        self.assertNotEqual(results[0].speaker_id, results[1].speaker_id)
        # And each half was routed by ITS OWN detected language.
        self.assertEqual(results[0].source_language, LANG_A)
        self.assertEqual(results[1].source_language, LANG_B)
        self.assertEqual(
            [(s, t) for _text, s, t in mt.calls],
            [(LANG_A, LANG_B), (LANG_B, LANG_A)],
        )
        split_events = [
            e for e in session.journal.since(0)[0] if e.kind is EventKind.TURN_SPLIT
        ]
        self.assertEqual(len(split_events), 1)
        self.assertEqual(split_events[0].payload["pieces"], 2)

    async def test_neither_speaker_poisons_the_other_reference_buffer(self):
        session, _asr, _mt, _tts = make_session()
        run_on = AudioChunk(
            np.concatenate(
                [silence(0.3), tone(VOICE_A_HZ, 3.0), tone(VOICE_B_HZ, 3.0),
                 silence(0.4)]
            ),
            RATE,
        )
        await run_conversation(session, [run_on])
        self.assertEqual(len(session.speakers), 2)
        # Each profile's retained reference must carry only its OWN pitch.
        for profile in session.speakers.profiles():
            audio = profile.reference_audio()
            self.assertGreater(audio.duration_s, 0.0)
            peak = dominant_hz(audio)
            self.assertTrue(
                abs(peak - VOICE_A_HZ) < 15.0 or abs(peak - VOICE_B_HZ) < 15.0,
                f"speaker {profile.speaker_id} reference is at {peak:.0f} Hz, "
                "which is neither voice -- the buffer was poisoned",
            )

    async def test_a_single_speaker_segment_is_left_whole(self):
        session, _asr, _mt, _tts = make_session()
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 6.0))]
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(
            [e for e in session.journal.since(0)[0]
             if e.kind is EventKind.TURN_SPLIT],
            [],
        )

    async def test_a_short_segment_skips_the_check_entirely(self):
        # The gate exists so the common case does not pay N embedder passes to
        # discover it is the common case.
        session, _asr, _mt, _tts = make_session()
        session.speaker_change_min_segment_s = 30.0
        embedder = session.embedder
        before = 0

        original = embedder.embed
        calls = []

        async def counting(audio):
            calls.append(audio.duration_s)
            return await original(audio)

        session.embedder = type("E", (), {
            "min_seconds": embedder.min_seconds,
            "name": embedder.name,
            "embed": staticmethod(counting),
        })()
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 4.0))])
        # Exactly one embed: the identity pass, no window passes.
        self.assertEqual(len(calls), 1 + before, calls)

    async def test_the_recut_gate_is_wired_to_the_window_length(self):
        """BINDS-PROOF for the 2.5 s window (measured, see session.py).

        The old 1.5 s window made an identity decision on audio too short for
        this embedder to be stable on: over the 17-voice pool, within-speaker
        similarity at 1.5 s falls to 0.392 while the threshold separating
        DIFFERENT people is 0.62, so 27 of 60 same-speaker window pairs were
        cut. That is what split one German speaker in two on the real run.

        Reproduced here with an embedder that returns a fresh, dissimilar
        vector per window -- exactly what short-window instability looks like
        from the caller. Under the OLD geometry a 3.1 s single-speaker segment
        is re-cut; under the shipped one the gate does not even open. A
        default that does not change behaviour has reach zero, so this asserts
        both halves.
        """
        drifting = _DriftingEmbedder()
        old, _asr, _mt, _tts = make_session(
            session_id="old",
            speaker_change_window_s=1.5,
            speaker_change_min_segment_s=3.0,
        )
        old.embedder = drifting
        await run_conversation(old, [conversation_audio((VOICE_A_HZ, 3.1))])
        old_splits = [
            e for e in old.journal.since(0)[0] if e.kind is EventKind.TURN_SPLIT
        ]
        self.assertEqual(
            len(old_splits), 1,
            "the old 1.5 s geometry did NOT reproduce the split, so this test "
            "proves nothing about the new one",
        )

        shipped, _asr, _mt, _tts = make_session(session_id="new")
        shipped.embedder = _DriftingEmbedder()
        self.assertEqual(shipped.speaker_change_window_s, 2.5)
        self.assertEqual(shipped.speaker_change_min_segment_s, 5.0)
        await run_conversation(shipped, [conversation_audio((VOICE_A_HZ, 3.1))])
        self.assertEqual(
            [e for e in shipped.journal.since(0)[0]
             if e.kind is EventKind.TURN_SPLIT],
            [],
            "the shipped geometry still re-cuts a single speaker",
        )
        self.assertEqual(
            len(shipped.speakers), 1,
            "one speaker became more than one identity",
        )

    def test_an_impossible_recut_geometry_is_refused(self):
        """The ">= 2 x window" comment is a claim, so it is pinned.

        Violated, `count < 2` returns early on every segment: the re-cut is
        silently dead and the protection it exists to give is gone with no
        error anywhere.
        """
        with self.assertRaises(ValueError) as caught:
            make_session(
                session_id="bad",
                speaker_change_window_s=2.5,
                speaker_change_min_segment_s=4.0,
            )
        self.assertIn("would never run", str(caught.exception))

    async def test_detection_can_be_switched_off(self):
        session, _asr, _mt, _tts = make_session()
        session.speaker_change_detection = False
        run_on = AudioChunk(
            np.concatenate(
                [silence(0.3), tone(VOICE_A_HZ, 3.0), tone(VOICE_B_HZ, 3.0),
                 silence(0.4)]
            ),
            RATE,
        )
        results = await run_conversation(session, [run_on])
        self.assertEqual(len(results), 1, "the segment was re-cut despite the flag")


class TestQueueAndFailures(unittest.IsolatedAsyncioTestCase):
    async def test_queue_overrun_drops_the_oldest_turn(self):
        """The last-resort policy, with the bundler deliberately switched off.

        Dropping is no longer what a backlog normally meets: waiting segments
        are coalesced into one turn, so the overrun cannot be reached by
        ordinary speech (see `test_coalesce_and_context.py`, which pins both
        halves of that). The drop still has to work, because a bundle that is
        full or a segment at a different sample rate cannot be folded -- and
        because this is the behaviour every deployment before the bundler
        had. Turning the bundler off is how the arm reaches it.
        """
        session, _asr, _mt, _tts = make_session(
            max_queued_turns=1, coalesce_queued=False
        )
        audio = conversation_audio(
            (VOICE_A_HZ, 1.0), (VOICE_B_HZ, 1.0), (VOICE_A_HZ, 1.0)
        )
        segments = session.push_audio(audio)
        self.assertGreaterEqual(len(segments), 3)
        for segment in segments:
            session.enqueue(segment)
        self.assertEqual(session.pending(), 1)
        self.assertEqual(session.turns_dropped, len(segments) - 1)
        dropped = [
            e for e in session.journal.since(0)[0] if e.kind is EventKind.TURN_DROPPED
        ]
        self.assertEqual(len(dropped), len(segments) - 1)
        # The SURVIVOR must be the newest, because a stale utterance is the
        # worthless one in a live conversation.
        results = await session.drain()
        self.assertEqual(len(results), 1)

    async def test_an_mt_failure_fails_one_turn_and_not_the_session(self):
        session, _asr, mt, _tts = make_session(script=[("poison", LANG_A)])
        session.mt = FakeMt(fail_on="poison")
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 2.0))])
        errors = [e for e in session.journal.since(0)[0] if e.kind is EventKind.ERROR]
        self.assertTrue(errors)
        self.assertEqual(errors[0].payload["stage"], "mt")
        self.assertFalse(session.closed, "one bad turn must not close the session")

    async def test_timings_are_recorded_on_every_turn(self):
        session, _asr, _mt, _tts = make_session()
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.5))]
        )
        timings = results[0].timings.to_json()
        for key in ("asr_ms", "embed_ms", "mt_total_ms", "first_audio_ms", "total_ms"):
            self.assertIn(key, timings)
        self.assertGreaterEqual(timings["total_ms"], 0.0)


class TestSessionManager(unittest.IsolatedAsyncioTestCase):
    def _manager(self, clock, max_sessions=2, idle=10.0):
        def factory(sid, conversation):
            # The session must share the manager's clock, or idle collection
            # is judged against wall time while the test advances a fake one.
            session, _a, _m, _t = make_session(session_id=sid, clock=clock)
            return session

        return SessionManager(
            factory=factory,
            max_sessions=max_sessions,
            idle_timeout_s=idle,
            clock=clock,
        )

    async def test_reopening_an_id_resumes_rather_than_allocating(self):
        now = [0.0]
        manager = self._manager(lambda: now[0])
        conversation = ConversationLanguages.of([LANG_A, LANG_B])
        first = manager.open(conversation, session_id="abc")
        second = manager.open(conversation, session_id="abc")
        self.assertIs(first, second)
        self.assertEqual(len(manager), 1)

    async def test_the_cap_holds_and_idle_sessions_are_collected(self):
        now = [0.0]
        manager = self._manager(lambda: now[0], max_sessions=2, idle=10.0)
        conversation = ConversationLanguages.of([LANG_A, LANG_B])
        manager.open(conversation, session_id="a")
        manager.open(conversation, session_id="b")
        with self.assertRaises(RuntimeError):
            manager.open(conversation, session_id="c")
        now[0] = 100.0
        manager.open(conversation, session_id="c")
        self.assertIn("c", manager.ids())


if __name__ == "__main__":
    unittest.main()
