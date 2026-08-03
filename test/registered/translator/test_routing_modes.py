# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Auto vs manual routing: the rule table, and what happens without a rule.

The user's point about the manual mode is that it removes GUESSING. The ASR
still classifies the source of each utterance -- that is needed and cheap --
but once rules exist the target is a deterministic table lookup, never
elimination over the participant set. These tests pin exactly that, plus the
three things that make a rule table safe to hand a user: a duplicate source is
refused rather than tie-broken, an utterance with no rule is passed through
tagged rather than dropped, and the table survives a reconnect.

No language pair is hardcoded here either -- the falsifier in
test_languages.py covers the source literals, and these tests use invented
codes for the same reason.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_routing_modes.py -v
"""

import unittest


from sglang.srt.translator.languages import (
    LanguageError,
    LanguageMatrix,
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


class TestRoutingTable(unittest.TestCase):
    def test_several_rules_coexist(self):
        table = RoutingTable()
        table.add(LANG_A, LANG_B)
        table.add(LANG_B, LANG_A)
        table.add(LANG_C, LANG_B)
        self.assertEqual(len(table), 3)
        self.assertEqual(table.target_for(LANG_C), LANG_B)
        self.assertEqual(table.sources(), (LANG_A, LANG_B, LANG_C))

    def test_a_duplicate_source_is_refused_not_tie_broken(self):
        table = RoutingTable()
        table.add(LANG_A, LANG_B)
        with self.assertRaises(LanguageError) as ctx:
            table.add(LANG_A, LANG_C)
        self.assertIn("already exists", str(ctx.exception))
        # And the original survives the rejection.
        self.assertEqual(table.target_for(LANG_A), LANG_B)

    def test_a_self_rule_is_refused(self):
        with self.assertRaises(LanguageError):
            RoutingTable().add(LANG_A, LANG_A)

    def test_an_unknown_source_has_no_target(self):
        table = RoutingTable([])
        table.add(LANG_A, LANG_B)
        self.assertIsNone(table.target_for("zz"))

    def test_an_empty_table_is_falsy_so_auto_mode_reads_naturally(self):
        self.assertFalse(RoutingTable())
        table = RoutingTable()
        table.add(LANG_A, LANG_B)
        self.assertTrue(table)

    def test_replace_all_is_atomic(self):
        table = RoutingTable()
        table.add(LANG_A, LANG_B)
        with self.assertRaises(LanguageError):
            # Second rule duplicates the first source.
            table.replace_all([(LANG_B, LANG_A), (LANG_B, LANG_C)])
        # The live table must be untouched by a rejected update.
        self.assertEqual(table.to_json(), [{"source": LANG_A, "target": LANG_B}])

    def test_rules_are_validated_against_the_capability_matrix(self):
        matrix = LanguageMatrix.from_backends(
            asr_languages=[LANG_A, LANG_B], tts_languages=[LANG_A], mt_languages=None
        )
        table = RoutingTable()
        table.add(LANG_A, LANG_B)
        with self.assertRaises(LanguageError) as ctx:
            table.validate_against(matrix)
        self.assertIn("TTS cannot speak", str(ctx.exception))

    def test_removing_a_rule(self):
        table = RoutingTable()
        table.add(LANG_A, LANG_B)
        self.assertTrue(table.remove(LANG_A))
        self.assertFalse(table.remove(LANG_A))
        self.assertFalse(table)


class TestManualModeBeatsAuto(unittest.IsolatedAsyncioTestCase):
    async def test_the_table_decides_the_target_not_elimination(self):
        # Three participant languages, so auto mode would FAN OUT to two
        # targets. The rule pins exactly one, which is the whole point.
        session, _asr, mt, _tts = make_session(
            script=[("hello", LANG_A)],
            participants=(LANG_A, LANG_B, LANG_C),
            asr_languages=(LANG_A, LANG_B, LANG_C),
            tts_languages=(LANG_A, LANG_B, LANG_C),
        )
        session.set_routing_rules([(LANG_A, LANG_C)])
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.0))]
        )
        self.assertEqual(list(results[0].translations), [LANG_C])
        self.assertEqual([(s, t) for _x, s, t in mt.calls], [(LANG_A, LANG_C)])

    async def test_auto_mode_is_the_default(self):
        session, _asr, _mt, _tts = make_session()
        self.assertEqual(session.state()["routing_mode"], "auto")
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.0))]
        )
        self.assertEqual(list(results[0].translations), [LANG_B])

    async def test_an_empty_table_restores_auto_mode(self):
        session, _asr, _mt, _tts = make_session()
        session.set_routing_rules([(LANG_A, LANG_B)])
        self.assertEqual(session.state()["routing_mode"], "manual")
        session.set_routing_rules([])
        self.assertEqual(session.state()["routing_mode"], "auto")

    async def test_asymmetric_rules_work_in_both_directions(self):
        session, _asr, mt, _tts = make_session()
        session.set_routing_rules([(LANG_A, LANG_B), (LANG_B, LANG_A)])
        await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.0), (VOICE_B_HZ, 2.0))]
        )
        self.assertEqual(
            [(s, t) for _x, s, t in mt.calls],
            [(LANG_A, LANG_B), (LANG_B, LANG_A)],
        )


class TestNoRulePassthrough(unittest.IsolatedAsyncioTestCase):
    async def test_an_unruled_language_is_tagged_not_dropped(self):
        session, _asr, mt, _tts = make_session()
        # A rule exists only for LANG_B, so a LANG_A utterance has none.
        session.set_routing_rules([(LANG_B, LANG_A)])
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 2.0))]
        )
        self.assertEqual(results, [])
        self.assertEqual(mt.calls, [], "an unruled turn must not be translated")

        unrouted = [
            e for e in session.journal.since(0)[0]
            if e.kind is EventKind.TURN_UNROUTED
        ]
        self.assertEqual(len(unrouted), 1)
        payload = unrouted[0].payload
        self.assertEqual(payload["source"], LANG_A)
        self.assertIn("no routing rule", payload["reason"])
        # The transcript survives, so the user sees WHAT was said as well as
        # that it was not routed.
        self.assertTrue(payload["text"])
        self.assertEqual(payload["known_sources"], [LANG_B])

    async def test_the_session_keeps_working_after_an_unruled_turn(self):
        session, _asr, mt, _tts = make_session()
        session.set_routing_rules([(LANG_B, LANG_A)])
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 2.0))])
        results = await run_conversation(
            session, [conversation_audio((VOICE_B_HZ, 2.0))]
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(list(results[0].translations), [LANG_A])


class TestStickyAcrossReconnect(unittest.IsolatedAsyncioTestCase):
    async def test_rules_survive_a_reconnect(self):
        session, _asr, mt, _tts = make_session()
        session.set_routing_rules([(LANG_A, LANG_B)])
        before = session.state()["routing_rules"]

        session.on_reconnect()

        self.assertEqual(session.state()["routing_rules"], before)
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
