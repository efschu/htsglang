# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The runtime-derived language set, and the no-hardcoded-pair contract.

Hermetic: no model, no GPU, no network.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_languages.py -v
"""

import unittest

from sglang.srt.translator.languages import (
    ConversationLanguages,
    LanguageError,
    LanguageMatrix,
    canonical_code,
    canonical_set,
    display_name,
)


class TestCanonicalisation(unittest.TestCase):
    def test_regional_and_iso3_forms_collapse(self):
        for form in ("de", "DE", "de-DE", "de_DE", "  De-at ", "deu", "ger"):
            self.assertEqual(canonical_code(form), "de", form)
        self.assertEqual(canonical_code("es-419"), "es")
        self.assertEqual(canonical_code("zho"), "zh")

    def test_unknown_tag_passes_through_rather_than_being_guessed(self):
        # Two backends agreeing on an exotic tag must still intersect.
        self.assertEqual(canonical_code("xyz"), "xyz")
        self.assertEqual(canonical_set(["xyz-QQ", "XYZ"]), frozenset({"xyz"}))

    def test_bad_input_is_refused(self):
        for bad in ("", "   ", "1234", "-de"):
            with self.assertRaises(LanguageError):
                canonical_code(bad)
        with self.assertRaises(LanguageError):
            canonical_code(None)

    def test_display_name_falls_back_to_the_code(self):
        self.assertEqual(display_name("de")["english"], "German")
        self.assertEqual(display_name("xyz")["english"], "xyz")


class TestMatrixIntersection(unittest.TestCase):
    def test_system_set_is_the_intersection_not_a_constant(self):
        # ASR hears three, TTS speaks two, MT unconstrained.
        matrix = LanguageMatrix.from_backends(
            asr_languages=["de", "es", "en"],
            tts_languages=["de", "es"],
            mt_languages=None,
        )
        self.assertEqual(matrix.bidirectional, frozenset({"de", "es"}))
        # 'en' is heard but cannot be spoken, so it is a source only.
        self.assertIn("en", matrix.sources)
        self.assertNotIn("en", matrix.targets)

    def test_mt_narrows_both_ends_when_it_declares_a_set(self):
        matrix = LanguageMatrix.from_backends(
            asr_languages=["de", "es", "fr"],
            tts_languages=["de", "es", "fr"],
            mt_languages=["de", "fr"],
        )
        self.assertEqual(matrix.bidirectional, frozenset({"de", "fr"}))
        self.assertFalse(matrix.supports_pair("de", "es"))

    def test_pairs_excludes_the_identity_direction(self):
        matrix = LanguageMatrix.from_backends(["de", "es"], ["de", "es"], None)
        self.assertEqual(set(matrix.pairs()), {("de", "es"), ("es", "de")})

    def test_require_pair_names_the_stage_that_refuses(self):
        matrix = LanguageMatrix.from_backends(
            asr_languages=["de", "es"], tts_languages=["de"], mt_languages=None
        )
        with self.assertRaises(LanguageError) as ctx:
            matrix.require_pair("de", "es")
        # Actionability: the operator must learn WHICH checkpoint to swap.
        self.assertIn("TTS cannot speak", str(ctx.exception))
        with self.assertRaises(LanguageError) as ctx:
            matrix.require_pair("de", "de")
        self.assertIn("two different languages", str(ctx.exception))

    def test_json_exposes_per_stage_sets_so_a_gap_is_attributable(self):
        matrix = LanguageMatrix.from_backends(["de", "es", "en"], ["de"], ["de", "es"])
        payload = matrix.to_json()
        self.assertEqual(payload["stages"]["asr"], ["de", "en", "es"])
        self.assertEqual(payload["stages"]["tts"], ["de"])
        self.assertEqual(payload["stages"]["mt"], ["de", "es"])
        self.assertFalse(payload["unconstrained_mt"])


class TestConversationRouting(unittest.TestCase):
    def test_two_party_routing_is_elimination(self):
        conversation = ConversationLanguages.of(["de", "es"])
        self.assertEqual(conversation.targets_for("de"), ("es",))
        self.assertEqual(conversation.targets_for("es"), ("de",))

    def test_three_party_fans_out(self):
        conversation = ConversationLanguages.of(["de", "es", "fr"])
        self.assertEqual(conversation.targets_for("de"), ("es", "fr"))

    def test_bystander_language_routes_to_everyone(self):
        conversation = ConversationLanguages.of(["de", "es"])
        self.assertEqual(conversation.targets_for("it"), ("de", "es"))

    def test_explicit_route_overrides_elimination(self):
        conversation = ConversationLanguages.of(
            ["de", "es", "fr"], explicit_routes={"de": ["fr"]}
        )
        self.assertEqual(conversation.targets_for("de"), ("fr",))
        self.assertEqual(conversation.targets_for("es"), ("de", "fr"))

    def test_a_conversation_needs_two_languages(self):
        with self.assertRaises(LanguageError):
            ConversationLanguages.of(["de"])
        with self.assertRaises(LanguageError):
            ConversationLanguages.of(["de", "de-DE"])

    def test_self_route_is_refused(self):
        with self.assertRaises(LanguageError):
            ConversationLanguages.of(["de", "es"], explicit_routes={"de": ["de"]})


class TestNoHardcodedPairFalsifier(unittest.TestCase):
    """The falsifier for requirement 5.

    DE/ES is the development pair, nothing more. This test drives the same
    code with a pair that shares no letter, no script family and no European
    neighbour relation with it -- and additionally asserts by SOURCE
    INSPECTION that the strings 'de' and 'es' do not appear as literals in
    the routing modules. A behavioural test alone would pass on a codebase
    that special-cases DE/ES *in addition to* handling the general case; the
    source assertion is what makes the claim falsifiable.
    """

    def test_japanese_french_conversation_behaves_identically(self):
        matrix = LanguageMatrix.from_backends(["ja", "fr"], ["ja", "fr"], None)
        conversation = ConversationLanguages.of(["ja", "fr"])
        conversation.validate_against(matrix)
        self.assertEqual(conversation.targets_for("ja"), ("fr",))
        self.assertEqual(conversation.targets_for("fr"), ("ja",))
        self.assertTrue(matrix.supports_pair("ja", "fr"))

    def test_routing_modules_contain_no_language_pair_literals(self):
        import ast
        import pathlib

        import sglang.srt.translator as package

        root = pathlib.Path(package.__file__).parent
        # ``languages.py`` legitimately contains every code in its display and
        # ISO tables; the modules that DECIDE must not.
        for name in ("session.py", "segmenter.py", "speakers.py", "mt.py"):
            source = (root / name).read_text(encoding="utf-8")
            tree = ast.parse(source)
            literals = {
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            for banned in ("de", "es", "german", "spanish", "German", "Spanish"):
                self.assertNotIn(
                    banned,
                    literals,
                    f"{name} contains the language literal {banned!r}; routing "
                    "must come from configuration, not from source",
                )


if __name__ == "__main__":
    unittest.main()
