# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Auto vs manual routing: unordered pairs, fan-out, and what happens unpaired.

**These tests were rewritten, and the rewrite is the point.** The first version
asserted that a manual table pins EXACTLY ONE target and that a duplicate
source is refused. Both were the wrong reading of the requirement, and the
tests were faithfully pinning the mistake:

* a pair is a RELATIONSHIP, not a direction -- ``de <-> es`` must route both
  ways, or every reply is silently dropped;
* two pairs sharing a language are FAN-OUT, not ambiguity -- with
  ``{de<->es, de<->fr}`` a German utterance is rendered in both, played
  sequentially with a language tag. Refusing the second pair made the
  three-language case unexpressible;
* a repeated pair is a no-op, not an error: the user asked for a relationship
  that already holds.

What survives from the first version, because it was right: the ASR still
classifies each utterance's source (the direction is observed, never
configured), an unpaired language is passed through TAGGED rather than
dropped, and the table survives a reconnect.

No language pair is hardcoded here either -- these use invented codes for the
same reason test_languages.py's AST falsifier exists.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_routing_modes.py -v
"""

import unittest


from sglang.srt.translator.languages import (
    LanguageError,
    LanguageMatrix,
    LanguagePair,
    RoutingTable,
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

LANG_C = "cc"


class TestLanguagePair(unittest.TestCase):
    def test_a_pair_is_unordered(self):
        self.assertEqual(
            LanguagePair.of(LANG_A, LANG_B), LanguagePair.of(LANG_B, LANG_A)
        )

    def test_partner_answers_from_either_side(self):
        pair = LanguagePair.of(LANG_B, LANG_A)
        self.assertEqual(pair.partner(LANG_A), LANG_B)
        self.assertEqual(pair.partner(LANG_B), LANG_A)
        self.assertIsNone(pair.partner(LANG_C))

    def test_a_self_pair_is_refused(self):
        with self.assertRaises(LanguageError):
            LanguagePair.of(LANG_A, LANG_A)


class TestRoutingTable(unittest.TestCase):
    def test_a_pair_routes_BOTH_ways(self):
        table = RoutingTable()
        table.add_pair(LANG_A, LANG_B)
        self.assertEqual(table.partners_for(LANG_A), (LANG_B,))
        self.assertEqual(table.partners_for(LANG_B), (LANG_A,))

    def test_a_repeated_pair_is_DEDUPLICATED_not_refused(self):
        table = RoutingTable()
        first = table.add_pair(LANG_A, LANG_B)
        # Same relationship, stated the other way round.
        second = table.add_pair(LANG_B, LANG_A)
        self.assertEqual(first, second)
        self.assertEqual(len(table), 1)

    def test_two_pairs_sharing_a_language_FAN_OUT(self):
        table = RoutingTable()
        table.add_pair(LANG_A, LANG_B)
        table.add_pair(LANG_A, LANG_C)
        self.assertEqual(len(table), 2)
        self.assertEqual(table.partners_for(LANG_A), (LANG_B, LANG_C))
        self.assertEqual(table.partners_for(LANG_B), (LANG_A,))
        self.assertEqual(table.partners_for(LANG_C), (LANG_A,))

    def test_an_unpaired_language_has_no_partner(self):
        table = RoutingTable()
        table.add_pair(LANG_A, LANG_B)
        self.assertEqual(table.partners_for("zz"), ())

    def test_the_language_whitelist_is_every_language_named(self):
        table = RoutingTable()
        table.add_pair(LANG_A, LANG_B)
        table.add_pair(LANG_A, LANG_C)
        self.assertEqual(table.languages(), (LANG_A, LANG_B, LANG_C))

    def test_an_empty_table_is_falsy_so_auto_mode_reads_naturally(self):
        self.assertFalse(RoutingTable())
        table = RoutingTable()
        table.add_pair(LANG_A, LANG_B)
        self.assertTrue(table)

    def test_replace_all_is_atomic(self):
        table = RoutingTable()
        table.add_pair(LANG_A, LANG_B)
        with self.assertRaises(LanguageError):
            # The second entry is a self-pair, which is still refused.
            table.replace_all([(LANG_B, LANG_C), (LANG_C, LANG_C)])
        self.assertEqual(table.to_json(), [{"a": LANG_A, "b": LANG_B}])

    def test_removing_a_pair_works_from_either_side(self):
        table = RoutingTable()
        table.add_pair(LANG_A, LANG_B)
        self.assertTrue(table.remove_pair(LANG_B, LANG_A))
        self.assertFalse(table.remove_pair(LANG_A, LANG_B))
        self.assertFalse(table)

    def test_pairs_are_validated_against_the_capability_matrix(self):
        matrix = LanguageMatrix.from_backends(
            asr_languages=[LANG_A, LANG_B], tts_languages=[LANG_A], mt_languages=None
        )
        table = RoutingTable()
        table.add_pair(LANG_A, LANG_B)
        with self.assertRaises(LanguageError) as ctx:
            table.validate_against(matrix)
        self.assertIn("TTS cannot speak", str(ctx.exception))

    def test_capability_refusal_is_reported_per_pair_and_per_direction(self):
        # ASR hears both, TTS speaks only A: a->b is dead, b->a is fine.
        matrix = LanguageMatrix.from_backends(
            asr_languages=[LANG_A, LANG_B], tts_languages=[LANG_A], mt_languages=None
        )
        table = RoutingTable()
        table.add_pair(LANG_A, LANG_B)
        (report,) = table.capability_report(matrix)
        self.assertEqual((report["a"], report["b"]), (LANG_A, LANG_B))
        self.assertFalse(report["usable"])
        by_direction = {
            (d["source"], d["target"]): d for d in report["directions"]
        }
        self.assertFalse(by_direction[(LANG_A, LANG_B)]["usable"])
        self.assertIn("TTS cannot speak", by_direction[(LANG_A, LANG_B)]["reason"])
        # The other direction survives, and saying so is the point: an
        # asymmetric backend set is exactly what a single flag would hide.
        self.assertTrue(by_direction[(LANG_B, LANG_A)]["usable"])


class TestManualModeBeatsAuto(unittest.IsolatedAsyncioTestCase):
    async def test_the_table_decides_the_target_not_elimination(self):
        # Three participant languages, so auto mode would fan out to two. One
        # pair narrows it to one, which is what a pair means.
        session, _asr, mt, _tts = make_session(
            script=[("hello", LANG_A)],
            participants=(LANG_A, LANG_B, LANG_C),
            asr_languages=(LANG_A, LANG_B, LANG_C),
            tts_languages=(LANG_A, LANG_B, LANG_C),
        )
        session.set_routing_pairs([(LANG_A, LANG_C)])
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.0))]
        )
        self.assertEqual(list(results[0].translations), [LANG_C])
        self.assertEqual([(s, t) for _x, s, t in mt.calls], [(LANG_A, LANG_C)])

    async def test_TWO_pairs_produce_TWO_outputs_for_one_utterance(self):
        """The fan-out case, end to end. Three languages, two pairs, one turn."""
        session, _asr, mt, _tts = make_session(
            script=[("hello", LANG_A)],
            participants=(LANG_A, LANG_B, LANG_C),
            asr_languages=(LANG_A, LANG_B, LANG_C),
            tts_languages=(LANG_A, LANG_B, LANG_C),
        )
        session.set_routing_pairs([(LANG_A, LANG_B), (LANG_A, LANG_C)])
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.0))]
        )
        self.assertEqual(sorted(results[0].translations), [LANG_B, LANG_C])
        self.assertEqual(
            sorted((s, t) for _x, s, t in mt.calls),
            [(LANG_A, LANG_B), (LANG_A, LANG_C)],
        )
        # And each target carries its own audio, which is what "sequentially
        # with a language tag" needs on the client.
        self.assertEqual(sorted(results[0].audio), [LANG_B, LANG_C])

    async def test_one_pair_carries_the_reply_direction_too(self):
        session, _asr, mt, _tts = make_session()
        session.set_routing_pairs([(LANG_A, LANG_B)])
        await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.0), (VOICE_B_HZ, 2.0))]
        )
        # ONE pair, both directions -- the old version needed two rules for
        # this and would have dropped the reply if the user wrote only one.
        self.assertEqual(
            [(s, t) for _x, s, t in mt.calls],
            [(LANG_A, LANG_B), (LANG_B, LANG_A)],
        )

    async def test_auto_mode_is_the_default(self):
        session, _asr, _mt, _tts = make_session()
        self.assertEqual(session.state()["routing_mode"], "auto")
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.0))]
        )
        self.assertEqual(list(results[0].translations), [LANG_B])

    async def test_an_empty_table_restores_auto_mode(self):
        session, _asr, _mt, _tts = make_session()
        session.set_routing_pairs([(LANG_A, LANG_B)])
        self.assertEqual(session.state()["routing_mode"], "manual")
        session.set_routing_pairs([])
        self.assertEqual(session.state()["routing_mode"], "auto")


class TestAsrWhitelist(unittest.IsolatedAsyncioTestCase):
    """The table's languages become the recognizer's candidate set."""

    async def test_the_table_languages_reach_the_recognizer(self):
        session, asr, _mt, _tts = make_session(
            participants=(LANG_A, LANG_B, LANG_C),
            asr_languages=(LANG_A, LANG_B, LANG_C),
            tts_languages=(LANG_A, LANG_B, LANG_C),
        )
        self.assertEqual(asr.restrict_languages, ())
        session.set_routing_pairs([(LANG_A, LANG_B)])
        self.assertEqual(asr.restrict_languages, (LANG_A, LANG_B))

    async def test_auto_mode_clears_the_whitelist(self):
        session, asr, _mt, _tts = make_session()
        session.set_routing_pairs([(LANG_A, LANG_B)])
        session.set_routing_pairs([])
        self.assertEqual(asr.restrict_languages, ())

    def test_a_backend_without_the_hook_is_not_required_to_have_one(self):
        # Duck-typed on purpose: pushing the whitelist must not become a
        # requirement every backend has to satisfy. A recognizer that cannot
        # restrict simply does not carry the method, and setting pairs must
        # still succeed rather than raise on the missing attribute.
        session, asr, _mt, _tts = make_session()

        class UnrestrictableAsr:
            """Everything the pipeline needs, and no whitelist hook."""

            name = "unrestrictable"

            def supported_languages(self):
                return asr.supported_languages()

            async def transcribe(self, audio, hint_language=None):
                return await asr.transcribe(audio, hint_language)

        session.asr = UnrestrictableAsr()
        self.assertFalse(hasattr(session.asr, "set_restrict_languages"))
        session.set_routing_pairs([(LANG_A, LANG_B)])
        self.assertEqual(session.state()["routing_mode"], "manual")


class TestConstrainedDetection(unittest.TestCase):
    """The identifier still decides -- from a narrowed candidate set."""

    def setUp(self):
        from sglang.srt.translator.asr_backends import constrained_language_choice

        self.choose = constrained_language_choice

    def test_no_restriction_leaves_the_answer_alone(self):
        self.assertEqual(self.choose(LANG_C, 0.9, None, ()), (LANG_C, 0.9))

    def test_an_in_set_answer_keeps_its_own_confidence(self):
        self.assertEqual(
            self.choose(LANG_A, 0.87, None, (LANG_A, LANG_B)), (LANG_A, 0.87)
        )

    def test_an_out_of_set_answer_falls_back_to_the_best_IN_SET_language(self):
        # THE CASE. The identifier's favourite is unroutable; the best allowed
        # candidate wins and reports ITS probability, so the turn survives and
        # the UI can still tag it as uncertain.
        code, confidence = self.choose(
            LANG_C,
            0.55,
            [(LANG_C, 0.55), (LANG_B, 0.30), (LANG_A, 0.12)],
            (LANG_A, LANG_B),
        )
        self.assertEqual(code, LANG_B)
        self.assertAlmostEqual(confidence, 0.30)

    def test_nothing_is_ever_discarded(self):
        # No posterior available at all: still resolves, with an honest zero.
        code, confidence = self.choose(LANG_C, 0.9, None, (LANG_A, LANG_B))
        self.assertIn(code, (LANG_A, LANG_B))
        self.assertEqual(confidence, 0.0)


class TestNoPairPassthrough(unittest.IsolatedAsyncioTestCase):
    async def test_an_unpaired_language_is_tagged_not_dropped(self):
        session, _asr, mt, _tts = make_session(
            participants=(LANG_A, LANG_B, LANG_C),
            asr_languages=(LANG_A, LANG_B, LANG_C),
            tts_languages=(LANG_A, LANG_B, LANG_C),
        )
        # A pair exists only between B and C, so a LANG_A utterance has none.
        session.set_routing_pairs([(LANG_B, LANG_C)])
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.0))]
        )
        self.assertEqual(results, [])
        self.assertEqual(mt.calls, [], "an unpaired turn must not be translated")

        unrouted = [
            e for e in session.journal.since(0)[0]
            if e.kind is EventKind.TURN_UNROUTED
        ]
        self.assertEqual(len(unrouted), 1)
        payload = unrouted[0].payload
        self.assertEqual(payload["source"], LANG_A)
        self.assertIn("no routing pair", payload["reason"])
        # The transcript survives, so the user sees WHAT was said as well as
        # that it was not routed.
        self.assertTrue(payload["text"])
        self.assertEqual(payload["known_languages"], [LANG_B, LANG_C])

    async def test_the_session_keeps_working_after_an_unpaired_turn(self):
        session, _asr, mt, _tts = make_session(
            participants=(LANG_A, LANG_B, LANG_C),
            asr_languages=(LANG_A, LANG_B, LANG_C),
            tts_languages=(LANG_A, LANG_B, LANG_C),
        )
        session.set_routing_pairs([(LANG_B, LANG_C)])
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 2.0))])
        results = await run_conversation(
            session, [conversation_audio((VOICE_B_HZ, 2.0))]
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(list(results[0].translations), [LANG_C])


class TestStickyAcrossReconnect(unittest.IsolatedAsyncioTestCase):
    async def test_pairs_survive_a_reconnect(self):
        session, _asr, mt, _tts = make_session()
        session.set_routing_pairs([(LANG_A, LANG_B)])
        before = session.state()["routing_pairs"]

        session.on_reconnect()

        self.assertEqual(session.state()["routing_pairs"], before)
        self.assertEqual(session.state()["routing_mode"], "manual")
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 2.0))])
        self.assertEqual([(s, t) for _x, s, t in mt.calls], [(LANG_A, LANG_B)])


class TestCapabilityListing(unittest.TestCase):
    def test_unusable_languages_are_named_with_a_reason(self):
        from fastapi.testclient import TestClient

        from test_audio_and_http import build_service
        from sglang.srt.translator.backends import FakeTts
        from sglang.srt.translator.server import build_app

        service = build_service()
        # ASR hears both, TTS speaks only one.
        service.stack.tts = FakeTts(languages=(LANG_A,), sample_rate=16000)
        payload = TestClient(build_app(service)).get(
            "/api/translator/languages"
        ).json()
        by_code = {e["code"]: e for e in payload["selectable"]}
        self.assertTrue(by_code[LANG_A]["usable"])
        self.assertFalse(by_code[LANG_B]["usable"])
        # Named, not silently filtered.
        self.assertIn("TTS cannot speak it", by_code[LANG_B]["reason"])
        self.assertTrue(by_code[LANG_B]["as_source"])
        self.assertFalse(by_code[LANG_B]["as_target"])


if __name__ == "__main__":
    unittest.main()
