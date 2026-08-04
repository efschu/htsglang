# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Three-valued name suggestions and the adjacency that resolves them (§17.3).

The user's own test cases are here verbatim, and so are the counter-cases,
which are the more interesting half: a `third_party` name must never land on
the speaker, and "sag hallo, Moritz" must expire when nobody answers or when
the same person keeps talking. A tracker that resolved either of those would
attach a real person's name to the wrong voice and then carry it through the
whole transcript.

None of this needs the model. The classification does; the ADJACENCY -- which
is where the interesting failure modes live -- is a question about diarization
ids and timestamps, and is driven here with a fake clock.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_name_hints.py -v
"""

import unittest

from sglang.srt.translator.name_hints import (
    KIND_ADDRESSED,
    KIND_SELF,
    KIND_THIRD_PARTY,
    NameCandidate,
    SuggestionTracker,
    looks_like_naming,
    parse_candidates,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TestPreFilter(unittest.TestCase):
    def test_the_user_cases_pass_the_gate(self):
        for text in (
            "ich bin Matthias Ehrenfeuchter",
            "darf ich vorstellen: Larisa Ehrenfeuchter",
            "sag hallo, Moritz",
            "me llamo Ana",
        ):
            with self.subTest(text=text):
                self.assertTrue(looks_like_naming(text))

    def test_ordinary_speech_does_not_reach_the_model(self):
        for text in (
            "",
            "   ",
            "wie geht es dir heute",
            "das war ein sehr langer tag",
            "no entiendo nada de eso",
        ):
            with self.subTest(text=text):
                self.assertFalse(looks_like_naming(text))

    def test_german_noun_capitals_do_not_trip_the_gate(self):
        """The measured defect, pinned.

        The first real German turn through the front door -- "Als ich sechs
        war, sah ich einmal ein wunderbares Bild." -- tripped the capital
        heuristic on "Bild" and bought an LLM round trip for a sentence with
        no name in it. German capitalises every noun, so in German the cue
        list is the whole gate.
        """
        sentence = "Als ich sechs war, sah ich einmal ein wunderbares Bild."
        self.assertTrue(looks_like_naming(sentence), "language-blind, it passes")
        self.assertFalse(looks_like_naming(sentence, "de"))
        self.assertFalse(looks_like_naming(sentence, "de-DE"))
        # A cue still gets through in German, which is the point of keeping
        # the cue list rather than switching the gate off.
        self.assertTrue(looks_like_naming("ich bin Matthias", "de"))
        self.assertTrue(
            looks_like_naming("darf ich vorstellen: Larisa", "de")
        )
        # And the heuristic still works where nouns are lower case.
        self.assertTrue(looks_like_naming("yesterday I met Ana there", "en"))

    def test_a_sentence_initial_capital_alone_is_not_a_signal(self):
        # Every sentence has one; letting it through would defeat the gate.
        self.assertFalse(looks_like_naming("Heute regnet es sehr stark"))


class TestParsing(unittest.TestCase):
    def test_it_reads_the_documented_shape(self):
        got = parse_candidates(
            '{"candidates": [{"name": "Matthias", "kind": "self"}]}'
        )
        self.assertEqual([(c.name, c.kind) for c in got], [("Matthias", "self")])

    def test_a_fenced_answer_is_still_read(self):
        got = parse_candidates(
            '```json\n{"candidates": [{"name": "Ana", "kind": "self"}]}\n```'
        )
        self.assertEqual(len(got), 1)

    def test_an_unknown_kind_is_dropped_not_defaulted(self):
        # Mapping an unrecognised kind onto a default attaches a real name to
        # the wrong person, which is the failure this feature exists to avoid.
        got = parse_candidates(
            '{"candidates": [{"name": "Ana", "kind": "maybe"},'
            ' {"name": "Ben", "kind": "self"}]}'
        )
        self.assertEqual([c.name for c in got], ["Ben"])

    def test_garbage_yields_nothing_rather_than_raising(self):
        for payload in ("", "no idea", "{", '{"candidates": "Ana"}', None):
            with self.subTest(payload=payload):
                self.assertEqual(parse_candidates(payload), [])


class TestAdjacency(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.tracker = SuggestionTracker(
            window_s=15.0, max_turns=2, clock=self.clock
        )

    def test_self_names_the_current_speaker(self):
        out = self.tracker.observe(
            "t1", "speaker-1",
            [NameCandidate("Matthias Ehrenfeuchter", KIND_SELF)],
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].speaker_id, "speaker-1")
        self.assertEqual(out[0].name, "Matthias Ehrenfeuchter")
        self.assertEqual(out[0].kind, KIND_SELF)

    def test_third_party_floats_and_never_names_the_speaker(self):
        out = self.tracker.observe(
            "t1", "speaker-1",
            [NameCandidate("Larisa Ehrenfeuchter", KIND_THIRD_PARTY)],
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].kind, KIND_THIRD_PARTY)
        # The load-bearing assertion: it is attached to NOBODY.
        self.assertEqual(out[0].speaker_id, "")

    def test_addressed_names_whoever_answers(self):
        self.assertEqual(
            self.tracker.observe(
                "t1", "speaker-1", [NameCandidate("Moritz", KIND_ADDRESSED)]
            ),
            [],
            "an addressed name must not be offered before somebody answers",
        )
        self.clock.advance(3.0)
        out = self.tracker.observe("t2", "speaker-2", [])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].name, "Moritz")
        self.assertEqual(out[0].speaker_id, "speaker-2")
        self.assertEqual(out[0].resolved_turn_id, "t2")

    def test_the_same_speaker_answering_themselves_is_not_an_answer(self):
        self.tracker.observe(
            "t1", "speaker-1", [NameCandidate("Moritz", KIND_ADDRESSED)]
        )
        self.clock.advance(2.0)
        self.assertEqual(self.tracker.observe("t2", "speaker-1", []), [])
        # Still waiting -- somebody may yet reply.
        self.assertEqual(len(self.tracker.pending()), 1)

    def test_silence_expires_the_candidate(self):
        self.tracker.observe(
            "t1", "speaker-1", [NameCandidate("Moritz", KIND_ADDRESSED)]
        )
        self.clock.advance(30.0)
        self.assertEqual(self.tracker.observe("t2", "speaker-2", []), [])
        self.assertEqual(self.tracker.pending(), [])

    def test_too_many_turns_expire_it_even_inside_the_time_window(self):
        self.tracker.observe(
            "t1", "speaker-1", [NameCandidate("Moritz", KIND_ADDRESSED)]
        )
        self.clock.advance(1.0)
        self.tracker.observe("t2", "speaker-1", [])
        self.clock.advance(1.0)
        self.tracker.observe("t3", "speaker-1", [])
        self.clock.advance(1.0)
        # Now somebody else speaks, but three turns have passed.
        self.assertEqual(self.tracker.observe("t4", "speaker-2", []), [])

    def test_a_second_reply_does_not_re_suggest(self):
        self.tracker.observe(
            "t1", "speaker-1", [NameCandidate("Moritz", KIND_ADDRESSED)]
        )
        self.clock.advance(1.0)
        self.assertEqual(len(self.tracker.observe("t2", "speaker-2", [])), 1)
        self.clock.advance(1.0)
        self.assertEqual(self.tracker.observe("t3", "speaker-3", []), [])

    def test_an_uncertain_line_gets_no_suggestion_for_its_speaker(self):
        out = self.tracker.observe(
            "t1", "speaker-1",
            [NameCandidate("Matthias", KIND_SELF)],
            uncertain=True,
        )
        self.assertEqual(out, [])

    def test_an_uncertain_reply_does_not_take_an_addressed_name(self):
        self.tracker.observe(
            "t1", "speaker-1", [NameCandidate("Moritz", KIND_ADDRESSED)]
        )
        self.clock.advance(1.0)
        self.assertEqual(
            self.tracker.observe("t2", "speaker-2", [], uncertain=True), []
        )

    def test_a_discarded_name_is_not_offered_again(self):
        self.tracker.discard("speaker-1", "Matthias")
        self.assertEqual(
            self.tracker.observe(
                "t1", "speaker-1", [NameCandidate("Matthias", KIND_SELF)]
            ),
            [],
        )

    def test_the_three_user_sentences_end_to_end(self):
        """The user's own examples, in one conversation, in order."""
        first = self.tracker.observe(
            "t1", "speaker-1",
            [NameCandidate("Matthias Ehrenfeuchter", KIND_SELF)],
        )
        self.clock.advance(2.0)
        second = self.tracker.observe(
            "t2", "speaker-1",
            [NameCandidate("Larisa Ehrenfeuchter", KIND_THIRD_PARTY)],
        )
        self.clock.advance(2.0)
        self.tracker.observe(
            "t3", "speaker-1", [NameCandidate("Moritz", KIND_ADDRESSED)]
        )
        self.clock.advance(2.0)
        third = self.tracker.observe("t4", "speaker-2", [])

        self.assertEqual((first[0].name, first[0].speaker_id),
                         ("Matthias Ehrenfeuchter", "speaker-1"))
        self.assertEqual((second[0].name, second[0].speaker_id),
                         ("Larisa Ehrenfeuchter", ""))
        self.assertEqual((third[0].name, third[0].speaker_id),
                         ("Moritz", "speaker-2"))


from test_session import LANG_A  # noqa: E402


class ScriptedMt:
    """An MT backend whose ``ask`` returns a canned extraction answer."""

    name = "scripted-mt"

    def __init__(self, inner, answer):
        self._inner = inner
        self.answer = answer
        self.asked = []

    def __getattr__(self, item):
        return getattr(self._inner, item)

    async def ask(self, system, user, max_tokens=200):
        self.asked.append(user)
        return self.answer


class TestSuggestionsInSession(unittest.IsolatedAsyncioTestCase):
    """The pipeline half: the gate, the events, and confirm/discard."""

    async def _session(self, answer, script=None):
        from sglang.srt.translator.session import run_conversation
        from test_session import VOICE_A_HZ, conversation_audio, make_session

        session, _asr, mt, _tts = make_session(script=script)
        scripted = ScriptedMt(mt, answer)
        session.mt = scripted
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        return session, scripted

    async def test_a_self_introduction_is_applied_at_once_and_undoable(self):
        from sglang.srt.translator.session import EventKind

        session, _mt = await self._session(
            '{"candidates": [{"name": "Matthias", "kind": "self"}]}',
            script=[("ich bin Matthias Ehrenfeuchter", LANG_A)],
        )
        events = [
            event
            for event in session.journal.since(0)[0]
            if event.kind is EventKind.SPEAKER_SUGGESTION
        ]
        self.assertEqual(len(events), 1)
        suggestion_id = events[0].payload["suggestion_id"]
        speaker_id = events[0].payload["speaker_id"]
        # A self-introduction has no ambiguity to resolve, so it APPLIES --
        # the user reported the name plainly not working while it sat in a
        # chip waiting for a confirmation nobody knew was needed.
        self.assertIs(events[0].payload["applied"], True)
        self.assertIs(events[0].payload["automatic"], True)
        self.assertEqual(session.speakers.get(speaker_id).label, "Matthias")
        self.assertEqual(
            session.transcript.lines()[0].speaker_label, "Matthias"
        )
        # ...and it can be taken back, which is what makes applying it safe.
        self.assertTrue(session.undo_name(suggestion_id))
        self.assertIsNone(session.speakers.get(speaker_id).label)
        self.assertEqual(
            session.transcript.lines()[0].speaker_label, speaker_id
        )

    async def test_ordinary_speech_never_reaches_the_model(self):
        session, mt = await self._session(
            '{"candidates": []}', script=[("wie geht es dir heute", LANG_A)]
        )
        self.assertEqual(mt.asked, [])
        self.assertEqual(session.suggestions, {})

    async def test_a_third_party_chip_refuses_to_guess_a_speaker(self):
        session, _mt = await self._session(
            '{"candidates": [{"name": "Larisa", "kind": "third_party"}]}',
            script=[("darf ich vorstellen: Larisa Ehrenfeuchter", LANG_A)],
        )
        suggestion_id = next(iter(session.suggestions))
        self.assertEqual(session.suggestions[suggestion_id].speaker_id, "")
        with self.assertRaises(ValueError):
            session.confirm_suggestion(suggestion_id)

    async def test_a_discarded_chip_is_gone_and_stays_gone(self):
        # third_party still queues: it belongs to nobody in the room yet.
        session, _mt = await self._session(
            '{"candidates": [{"name": "Larisa", "kind": "third_party"}]}',
            script=[("darf ich vorstellen: Larisa", LANG_A)],
        )
        suggestion_id = next(iter(session.suggestions))
        self.assertTrue(session.discard_suggestion(suggestion_id))
        self.assertFalse(session.discard_suggestion(suggestion_id))
        with self.assertRaises(KeyError):
            session.confirm_suggestion(suggestion_id)

    async def test_an_extractor_failure_does_not_fail_the_turn(self):
        from sglang.srt.translator.backends import BackendError
        from sglang.srt.translator.session import run_conversation
        from test_session import VOICE_A_HZ, conversation_audio, make_session

        session, _asr, mt, _tts = make_session(script=[("ich bin Matthias", LANG_A)])

        class BrokenMt(ScriptedMt):
            async def ask(self, system, user, max_tokens=200):
                raise BackendError("mt", "no")

        session.mt = BrokenMt(mt, "")
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 1.2))]
        )
        self.assertTrue(results)
        self.assertTrue(results[0].translations)
        self.assertEqual(session.suggestions, {})


if __name__ == "__main__":
    unittest.main()
