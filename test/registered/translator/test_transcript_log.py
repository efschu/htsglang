# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The written record: server-held, and independent of the journal (§17.2).

The load-bearing test in this file is not "a transcript exists". It is
``test_journal_eviction_does_not_touch_the_transcript``: the tempting
implementation derives the transcript from journal events, and that
implementation passes every other test here while silently losing the
beginning of any conversation longer than the journal's audio budget. The
journal is a bounded replay buffer by design and must stay one; the record
the user scrolls is a different object with a different lifetime.

The second is ``test_resume_restores_the_record``: a phone whose browser tab
was evicted -- which on Android happens routinely -- comes back with cursor
zero and must get the conversation back whole.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_transcript_log.py -v
"""

import unittest

from sglang.srt.translator.session import EventKind, Journal, run_conversation
from sglang.srt.translator.transcript_log import (
    CONFIDENCE_UNCERTAIN,
    KIND_NOTICE,
    ORIGIN_MANUAL,
    TranscriptLog,
)
from test_session import (  # noqa: E402  - sibling helper module
    LANG_A,
    VOICE_A_HZ,
    VOICE_B_HZ,
    conversation_audio,
    make_session,
)


class TestTranscriptLog(unittest.TestCase):
    """The store on its own, with no pipeline around it."""

    def test_lines_are_appended_in_order_with_stable_ids(self):
        log = TranscriptLog(clock=lambda: 0.0)
        first = log.append(
            turn_id="t1", speaker_id="speaker-1", speaker_label="speaker-1",
            source_language="xx", source_text="one",
        )
        second = log.append(
            turn_id="t2", speaker_id="speaker-2", speaker_label="speaker-2",
            source_language="xx", source_text="two",
        )
        self.assertEqual([first.line_id, second.line_id], [1, 2])
        self.assertEqual([line.source_text for line in log.lines()], ["one", "two"])
        self.assertEqual([line.line_id for line in log.since(1)], [2])

    def test_translations_land_on_the_line_that_was_already_there(self):
        # The line is created when the words are known, not when the
        # translation is: the record must not lag the audio.
        log = TranscriptLog(clock=lambda: 0.0)
        line = log.append(
            turn_id="t1", speaker_id="speaker-1", speaker_label="speaker-1",
            source_language="xx", source_text="hello",
        )
        self.assertEqual(line.translations, {})
        updated = log.set_translations(line.line_id, {"yy": "hola"})
        self.assertIs(updated, line)
        self.assertEqual(log.get(1).translations, {"yy": "hola"})
        self.assertIsNone(log.set_translations(999, {"yy": "nope"}))

    def test_rename_is_retroactive_and_reports_only_what_changed(self):
        log = TranscriptLog(clock=lambda: 0.0)
        for i in range(3):
            log.append(
                turn_id=f"t{i}", speaker_id="speaker-1", speaker_label="speaker-1",
                source_language="xx", source_text=str(i),
            )
        log.append(
            turn_id="t9", speaker_id="speaker-2", speaker_label="speaker-2",
            source_language="xx", source_text="other",
        )
        changed = log.relabel_speaker("speaker-1", "Matthias")
        self.assertEqual(len(changed), 3)
        self.assertTrue(all(line.speaker_label == "Matthias" for line in changed))
        self.assertEqual(log.get(4).speaker_label, "speaker-2")
        # Renaming to the same label a second time changes nothing, so a
        # repeated tap does not flood the wire.
        self.assertEqual(log.relabel_speaker("speaker-1", "Matthias"), [])

    def test_reassignment_clears_uncertainty_and_records_who_decided(self):
        log = TranscriptLog(clock=lambda: 0.0)
        line = log.append(
            turn_id="t1", speaker_id="speaker-1", speaker_label="speaker-1",
            source_language="xx", source_text="who said this",
            confidence=CONFIDENCE_UNCERTAIN,
            candidates=[{"speaker_id": "speaker-1", "similarity": 0.60}],
        )
        self.assertEqual(line.to_json()["confidence"], CONFIDENCE_UNCERTAIN)
        self.assertIn("candidates", line.to_json())
        moved = log.reassign_speaker(
            line.line_id, "speaker-2", "Ben", resolved_by="user"
        )
        self.assertEqual(moved.speaker_id, "speaker-2")
        self.assertEqual(moved.confidence, "exact")
        self.assertEqual(moved.candidates, [])
        self.assertEqual(moved.origin, ORIGIN_MANUAL)
        self.assertEqual(moved.to_json()["resolved_by"], "user")

    def test_overflow_says_so_instead_of_losing_the_start_silently(self):
        log = TranscriptLog(max_lines=4, clock=lambda: 0.0)
        for i in range(6):
            log.append(
                turn_id=f"t{i}", speaker_id="speaker-1", speaker_label="s",
                source_language="xx", source_text=str(i),
            )
        lines = log.lines()
        self.assertLessEqual(len(lines), 4)
        self.assertEqual(lines[0].kind, KIND_NOTICE)
        self.assertIn("dropped", lines[0].text)
        self.assertGreater(log.dropped, 0)
        # The surviving utterances are the RECENT ones, and the marker costs
        # one of the four slots -- the cap counts the notice, so a reader
        # never sees a full-looking window that is quietly missing its start.
        kept = [line.source_text for line in lines if line.kind != KIND_NOTICE]
        self.assertEqual(kept, ["3", "4", "5"])

    def test_a_one_line_cap_is_refused_rather_than_looping(self):
        # A cap of one cannot hold a line and the notice describing its
        # eviction; the notice would evict the line it describes.
        with self.assertRaises(ValueError):
            TranscriptLog(max_lines=1)

    def test_clear_empties_but_does_not_reuse_line_ids(self):
        log = TranscriptLog(clock=lambda: 0.0)
        log.append(turn_id="t1", speaker_id="s", speaker_label="s",
                   source_language="xx", source_text="one")
        self.assertEqual(log.clear(), 1)
        self.assertEqual(len(log), 0)
        again = log.append(turn_id="t2", speaker_id="s", speaker_label="s",
                           source_language="xx", source_text="two")
        # A client still holding cursor 1 must not be served line 1 twice
        # under two different texts.
        self.assertEqual(again.line_id, 2)


class TestTranscriptInSession(unittest.IsolatedAsyncioTestCase):
    """The record as the pipeline fills it."""

    async def test_every_turn_lands_in_the_record_with_both_languages(self):
        session, _asr, _mt, _tts = make_session()
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 1.2), (VOICE_B_HZ, 1.2))]
        )
        self.assertGreaterEqual(len(results), 2)
        lines = [
            line for line in session.transcript.lines() if line.kind != KIND_NOTICE
        ]
        self.assertEqual(len(lines), len(results))
        for line, result in zip(lines, results):
            self.assertEqual(line.turn_id, result.turn_id)
            self.assertEqual(line.source_text, result.source_text)
            self.assertEqual(line.source_language, result.source_language)
            # Both halves are carried, so the client's per-line toggle is pure
            # rendering rather than a round trip.
            self.assertEqual(line.translations, result.translations)

    async def test_journal_eviction_does_not_touch_the_transcript(self):
        """The falsifier for a transcript derived from journal events.

        The journal is deliberately tiny here. It evicts; the record must not.
        """
        session, _asr, _mt, _tts = make_session()
        # A fresh, deliberately tiny journal. Its bound lives in a deque's
        # maxlen and is fixed at construction, so mutating the attribute
        # would have looked like a shrink and evicted nothing -- the test
        # would then have proved the transcript survives an eviction that
        # never happened.
        session.journal = Journal(max_events=6, max_audio_bytes=1024)
        await run_conversation(
            session,
            [conversation_audio((VOICE_A_HZ, 1.2), (VOICE_B_HZ, 1.2),
                                (VOICE_A_HZ, 1.2))],
        )
        utterances = [
            line for line in session.transcript.lines() if line.kind != KIND_NOTICE
        ]
        self.assertGreaterEqual(len(utterances), 3)
        # And the journal really did evict -- otherwise this test proves
        # nothing, which is the failure mode the instrument rule warns about.
        self.assertGreater(session.journal.floor, 0)
        self.assertLessEqual(len(session.journal), 6)

    async def test_the_record_is_not_the_journal_replay_window(self):
        session, _asr, _mt, _tts = make_session()
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        # A client resuming from the journal's tip gets no events...
        events, _gap = session.journal.since(session.journal.next_seq)
        self.assertEqual(events, [])
        # ...but the record is still there in full.
        self.assertGreaterEqual(len(session.transcript.since(0)), 1)

    async def test_transcript_line_events_carry_the_line(self):
        session, _asr, _mt, _tts = make_session()
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        kinds = [event.kind for event in session.journal.since(0)[0]]
        self.assertIn(EventKind.TRANSCRIPT_LINE, kinds)
        self.assertIn(EventKind.TRANSCRIPT_UPDATE, kinds)
        line_events = [
            event
            for event in session.journal.since(0)[0]
            if event.kind is EventKind.TRANSCRIPT_LINE
        ]
        payload = line_events[0].payload["line"]
        self.assertIn("source_text", payload)
        self.assertIn("speaker_label", payload)

    async def test_naming_a_speaker_rewrites_the_past_and_emits_it(self):
        session, _asr, _mt, _tts = make_session()
        await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 1.2), (VOICE_A_HZ, 1.2))]
        )
        speaker_id = session.transcript.lines()[0].speaker_id
        before = session.journal.next_seq
        changed = session.name_speaker(speaker_id, "Matthias")
        self.assertGreaterEqual(len(changed), 1)
        self.assertTrue(
            all(line.speaker_label == "Matthias" for line in changed)
        )
        # Visible, not silent: one update event per changed line.
        new_events = [
            event
            for event in session.journal.since(before)[0]
            if event.kind is EventKind.TRANSCRIPT_UPDATE
        ]
        self.assertEqual(len(new_events), len(changed))
        # And the name sticks for lines created afterwards, without a second
        # retroactive pass.
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        self.assertEqual(session.transcript.lines()[-1].speaker_label, "Matthias")

    async def test_clear_is_the_only_thing_that_empties_it(self):
        session, _asr, _mt, _tts = make_session()
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        self.assertGreaterEqual(len(session.transcript), 1)
        # A reconnect does not.
        session.on_reconnect()
        self.assertGreaterEqual(len(session.transcript), 1)
        # A voice-mode switch does not.
        session.set_voice_mode(session.voice_mode)
        self.assertGreaterEqual(len(session.transcript), 1)
        # An explicit clear does.
        removed = session.clear_transcript()
        self.assertGreaterEqual(removed, 1)
        self.assertEqual(len(session.transcript), 0)
        self.assertEqual(
            session.journal.since(0)[0][-1].kind, EventKind.TRANSCRIPT_CLEARED
        )

    async def test_state_reports_the_record_separately_from_the_journal(self):
        session, _asr, _mt, _tts = make_session()
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        state = session.state()
        self.assertGreaterEqual(state["transcript_lines"], 1)
        self.assertGreater(state["transcript_next_line_id"], 1)
        self.assertIn("journal_seq", state)

    async def test_an_unrouted_turn_still_appears_in_the_record(self):
        """What the user said belongs in the record even when nothing routed.

        Manual routing with no rule for the spoken language: the pipeline
        tags it and produces no translation. A transcript written only on
        success would disagree with what the speaker heard themselves say.
        """
        session, _asr, _mt, _tts = make_session()
        session.set_routing_pairs([])
        session.routing_table.add_pair("zz", "qq")
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        lines = [
            line for line in session.transcript.lines() if line.kind != KIND_NOTICE
        ]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].source_language, LANG_A)
        self.assertEqual(lines[0].translations, {})


if __name__ == "__main__":
    unittest.main()
