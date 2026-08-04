# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Reading mode: the same pipeline with the synthesizer switched off (§17.1).

The load-bearing assertion here is NOT "no audio came back". A pipeline that
crashed before recognition would satisfy that too, and a test that only checks
for absence cannot tell the two apart. So every silent run is compared against
a voice run over the same input: the recognized text, the detected language,
the speaker attribution and the translation must be IDENTICAL, and only the
audio may differ. That is what makes this a mode rather than a failure.

The TTS backend is wrapped in a counting spy, because "the audio dictionary is
empty" is a claim about the output while "the synthesizer was never called" is
a claim about the work — and reading mode exists to avoid the work.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_silence_mode.py -v
"""

import unittest

from sglang.srt.translator.session import EventKind, run_conversation
from sglang.srt.translator.voices import OutputMode, VoicePoolError
from test_session import (  # noqa: E402  - sibling helper module
    VOICE_A_HZ,
    VOICE_B_HZ,
    conversation_audio,
    make_session,
)


class CountingTts:
    """Wraps a TTS backend and counts what actually reached it."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def synthesize(self, *args, **kwargs):
        self.calls += 1
        return self._inner.synthesize(*args, **kwargs)


def spied_session(**kwargs):
    session, asr, mt, tts = make_session(**kwargs)
    spy = CountingTts(tts)
    session.tts = spy
    return session, spy


class TestSilenceMode(unittest.IsolatedAsyncioTestCase):
    async def _run(self, mode, audio=None):
        session, spy = spied_session()
        session.set_output_mode(mode)
        results = await run_conversation(
            session,
            [audio or conversation_audio((VOICE_A_HZ, 1.2), (VOICE_B_HZ, 1.2))],
        )
        return session, spy, results

    async def test_silent_runs_the_whole_pipeline_except_the_synthesizer(self):
        loud_session, loud_spy, loud = await self._run(OutputMode.VOICE)
        quiet_session, quiet_spy, quiet = await self._run(OutputMode.SILENT)

        # The work was skipped, not merely the output discarded.
        self.assertGreater(loud_spy.calls, 0)
        self.assertEqual(quiet_spy.calls, 0)

        # ...and everything before synthesis is identical, which is what
        # separates a mode from a broken pipeline.
        self.assertEqual(len(quiet), len(loud))
        for silent_turn, loud_turn in zip(quiet, loud):
            self.assertEqual(silent_turn.source_text, loud_turn.source_text)
            self.assertEqual(silent_turn.source_language, loud_turn.source_language)
            self.assertEqual(silent_turn.speaker_id, loud_turn.speaker_id)
            self.assertEqual(silent_turn.translations, loud_turn.translations)
            self.assertEqual(silent_turn.audio, {})
            self.assertNotEqual(loud_turn.audio, {})

        # The transcript is the whole output in reading mode, so it had
        # better be complete.
        loud_lines = [
            (line.source_text, line.translations)
            for line in loud_session.transcript.lines()
        ]
        quiet_lines = [
            (line.source_text, line.translations)
            for line in quiet_session.transcript.lines()
        ]
        self.assertEqual(quiet_lines, loud_lines)

    async def test_no_audio_events_reach_the_client(self):
        session, _spy, _results = await self._run(OutputMode.SILENT)
        kinds = [event.kind for event in session.journal.since(0)[0]]
        self.assertNotIn(EventKind.TURN_AUDIO, kinds)
        # The translation still does, or reading mode would show nothing.
        self.assertIn(EventKind.TURN_TRANSLATION, kinds)
        self.assertIn(EventKind.TRANSCRIPT_LINE, kinds)

    async def test_the_voice_event_says_nothing_was_spoken(self):
        """No voice is CHOSEN in reading mode.

        Choosing one would report a clone downgrade for a turn nobody was
        going to hear -- a reason attached to a non-event, and the kind of
        entry that later reads as a real quality problem.
        """
        session, _spy, _results = await self._run(OutputMode.SILENT)
        events = [
            event
            for event in session.journal.since(0)[0]
            if event.kind is EventKind.TURN_VOICE
        ]
        self.assertTrue(events)
        for event in events:
            self.assertIs(event.payload["spoken"], False)
            self.assertEqual(event.payload["output_mode"], "silent")
            self.assertNotIn("downgraded", event.payload)

    async def test_reading_mode_keeps_building_the_clone_reference(self):
        """A conversation may start silent and continue aloud.

        The reference buffer accumulated while reading is exactly the buffer
        wanted at the switch, so admission must not be gated on speaking.
        """
        session, spy = spied_session()
        session.set_output_mode(OutputMode.SILENT)
        await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 1.2), (VOICE_A_HZ, 1.2))]
        )
        self.assertEqual(spy.calls, 0)
        profiles = session.speakers.profiles()
        self.assertTrue(profiles)
        self.assertGreater(profiles[0].reference_seconds(), 0.0)

        # Switch back and the very next turn is spoken.
        session.set_output_mode(OutputMode.VOICE)
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 1.2))]
        )
        self.assertGreater(spy.calls, 0)
        self.assertTrue(results[-1].audio)

    async def test_the_mode_is_session_state_and_survives_a_reconnect(self):
        session, _spy = spied_session()
        session.set_output_mode(OutputMode.SILENT)
        self.assertEqual(session.state()["output_mode"], "silent")
        session.on_reconnect()
        self.assertEqual(session.state()["output_mode"], "silent")
        self.assertIs(session.output_mode, OutputMode.SILENT)

    async def test_switching_is_recorded_rather_than_silent(self):
        session, _spy = spied_session()
        before = session.journal.next_seq
        session.set_output_mode(OutputMode.SILENT)
        events = [
            event
            for event in session.journal.since(before)[0]
            if event.kind is EventKind.SESSION_STATE
        ]
        self.assertTrue(any(e.payload.get("output_mode") == "silent" for e in events))

    def test_an_unknown_mode_is_refused_by_name(self):
        with self.assertRaises(VoicePoolError) as caught:
            OutputMode.parse("mute")
        self.assertIn("mute", str(caught.exception))
        self.assertIn("silent", str(caught.exception))

    def test_the_two_modes_are_orthogonal(self):
        # Reading mode says WHETHER a session speaks; voice mode says in
        # WHOSE voice. Collapsing them would make "silent" a third voice.
        from sglang.srt.translator.voices import VoiceMode

        self.assertNotIn("silent", [m.value for m in VoiceMode])
        self.assertNotIn("clone", [m.value for m in OutputMode])


if __name__ == "__main__":
    unittest.main()
