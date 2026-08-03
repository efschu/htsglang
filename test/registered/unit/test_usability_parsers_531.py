"""#531: the usability-parser mapping the planner emits into boot commands.

Standing order: --reasoning-parser and --tool-call-parser are STANDARD boot
settings. A parser-less boot answers HTTP 200 while returning the
chain-of-thought as raw text in ``content`` and tool calls as unparsed
strings, so the failure this guards against is silent by construction --
which is exactly why the mapping needs pins rather than eyeballing.

Hermetic: no server, no CUDA, no NVML.
"""

import unittest

from sglang.srt.planner.flags import (
    _USABILITY_PARSER_TABLE,
    usability_argv,
    usability_parsers,
    validate_usability_parsers,
)


class TestUsabilityParserRegistryAgreement(unittest.TestCase):
    """The table may not name a parser the server would reject."""

    def test_every_mapped_name_is_registered(self):
        problems = validate_usability_parsers()
        # An unavailable registry reports itself and is not a mapping failure;
        # anything else is.
        real = [p for p in problems if "registries unavailable" not in p]
        self.assertEqual(real, [], f"mapping names drifted from the registries: {real}")

    def test_registry_check_actually_reaches_the_registries(self):
        """Spread precondition: a check that cannot see the registries would
        pass vacuously, so assert it really imported them."""
        problems = validate_usability_parsers()
        self.assertFalse(
            any("registries unavailable" in p for p in problems),
            "the registries did not import, so the agreement test above proved "
            f"nothing: {problems}",
        )


class TestFamilyResolution(unittest.TestCase):
    CASES = [
        # (path, reasoning, toolcall)
        ("/m/Qwen3.6-27B-INT8-W8A8", "qwen3", "qwen3_coder"),
        ("/m/Qwen3.6-27B-FP8", "qwen3", "qwen3_coder"),
        ("/m/Qwen3.6-35B-A3B-AWQ-4bit", "qwen3", "qwen3_coder"),
        ("/m/Qwen3.6-27B-Q3_K_M.gguf", "qwen3", "qwen3_coder"),
        ("/m/qwen3.5-2b", "qwen3", "qwen3_coder"),
        ("/m/DeepSeek-V4-Flash-0731-UD-Q3_K_XL.gguf", "deepseek-v4", "deepseekv4"),
        ("/m/DeepSeek-V3.2-Exp", "deepseek-v3", "deepseekv32"),
        ("/m/DeepSeek-V3.1", "deepseek-v3", "deepseekv31"),
        ("/m/DeepSeek-R1", "deepseek-r1", "deepseekv3"),
        ("/m/glm-4.7", "glm45", "glm47"),
        ("/m/gpt-oss-20b", "gpt-oss", "gpt-oss"),
        ("/m/Kimi-K2", "kimi_k2", "kimi_k2"),
        ("/m/Llama-3.1-8B", None, "llama3"),
    ]

    def test_families_resolve(self):
        for path, reasoning, toolcall in self.CASES:
            with self.subTest(path=path):
                got_r, got_t, note = usability_parsers(path)
                self.assertEqual(got_r, reasoning, note)
                self.assertEqual(got_t, toolcall, note)

    def test_dotted_versions_do_not_hide_the_family(self):
        """The regression that made the first implementation wrong: name
        tokenisation does not split on '.', so a token-set match never saw
        'qwen3' inside 'Qwen3.6'."""
        reasoning, toolcall, _ = usability_parsers("/m/Qwen3.6-27B")
        self.assertEqual((reasoning, toolcall), ("qwen3", "qwen3_coder"))

    def test_architectures_outrank_a_renamed_directory(self):
        """config architectures are authoritative: a misleading directory name
        must not decide the parsers."""
        reasoning, _, _ = usability_parsers(
            "/m/totally-unrelated-name",
            {"architectures": ["Qwen3_5ForConditionalGeneration"]},
        )
        self.assertEqual(reasoning, "qwen3")

    def test_point_releases_beat_the_general_row(self):
        """Table order is load-bearing: 'v3' is a substring of 'v32', so the
        specific rows must be reached first."""
        self.assertEqual(usability_parsers("/m/DeepSeek-V3.2")[1], "deepseekv32")
        self.assertEqual(usability_parsers("/m/DeepSeek-V3")[1], "deepseekv3")

    def test_v4_is_not_captured_by_the_v3_row(self):
        self.assertEqual(usability_parsers("/m/DeepSeek-V4-Flash")[0], "deepseek-v4")


class TestUnknownFamilyIsNamed(unittest.TestCase):
    def test_unknown_family_emits_no_flags_and_a_named_hint(self):
        reasoning, toolcall, note = usability_parsers("/m/Some-Unheard-Of-9B")
        self.assertIsNone(reasoning)
        self.assertIsNone(toolcall)
        self.assertIn("UNKNOWN MODEL FAMILY", note)
        # The hint must tell the reader where to look, not just that it failed.
        self.assertIn("DetectorMap", note)
        self.assertIn("ToolCallParserEnum", note)

    def test_argv_is_empty_for_unknown_and_populated_for_known(self):
        argv_unknown, _ = usability_argv("/m/Some-Unheard-Of-9B")
        self.assertEqual(argv_unknown, [])
        argv_known, _ = usability_argv("/m/Qwen3.6-27B-INT8-W8A8")
        self.assertEqual(
            argv_known,
            ["--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder"],
        )


class TestTableHygiene(unittest.TestCase):
    def test_table_is_ordered_specific_before_general(self):
        """Any row whose token set is a superset of a later row's would be
        unreachable if it came second."""
        for i, (needed_i, _, _) in enumerate(_USABILITY_PARSER_TABLE):
            for needed_j, _, _ in _USABILITY_PARSER_TABLE[i + 1 :]:
                if set(needed_j).issubset(set(needed_i)):
                    self.fail(
                        f"row {needed_i} is shadowed: the later, more general "
                        f"row {needed_j} would also match it"
                    )

    def test_no_row_maps_both_parsers_to_none(self):
        for needed, reasoning, toolcall in _USABILITY_PARSER_TABLE:
            self.assertFalse(
                reasoning is None and toolcall is None,
                f"row {needed} contributes nothing",
            )


if __name__ == "__main__":
    unittest.main()
