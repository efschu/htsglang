# SPDX-License-Identifier: Apache-2.0
"""#382: one canonical name for the draft-model-path flag.

THE DEFECT sweep-2 hit: the cross-algorithm validator's error text said
``--speculative-draft-model`` while every parser in the tree reads
``speculative_draft_model_path`` (CLI ``--speculative-draft-model-path``).
Both spellings work -- the bare one is an argparse alias -- so nothing was
broken; what was broken was that an operator reading the error was told to
set a flag whose name did not match the one they had set, in a validator
whose whole job is to say what is missing.

CANONICAL: ``--speculative-draft-model-path``. It is the dataclass field
name (``speculative_draft_model_path``), it is what every consumer reads, and
it matches the sibling family ``--speculative-draft-model-revision`` /
``--speculative-draft-model-quantization``. The bare spelling is the odd one
out and stays only as a deprecated alias.

The regression corpus is "both spellings keep working": argparse resolves
them to one field, so the alias cannot be removed without breaking recipes,
and these tests pin that it is not.
"""

import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CANONICAL = "--speculative-draft-model-path"
ALIAS = "--speculative-draft-model"


class TestCanonicalChoice(CustomTestCase):
    def test_the_field_name_is_the_canonical_spelling(self):
        # The canonical CLI name is the field name with dashes -- so an error
        # message can be grepped straight to the attribute it is about.
        self.assertIn(
            "speculative_draft_model_path", ServerArgs.__dataclass_fields__
        )
        self.assertEqual(
            CANONICAL, "--" + "speculative_draft_model_path".replace("_", "-")
        )

    def test_the_sibling_family_uses_the_same_shape(self):
        # --speculative-draft-model-<attribute>. The bare alias is the one
        # member that does not fit, which is why it is the alias.
        for sibling in ("speculative_draft_model_revision",
                        "speculative_draft_model_quantization"):
            self.assertIn(sibling, ServerArgs.__dataclass_fields__)

    def test_the_alias_is_registered_as_a_deprecated_alias(self):
        self.assertIn(ALIAS, ServerArgs.DEPRECATED_FLAG_ALIASES)
        self.assertEqual(ServerArgs.DEPRECATED_FLAG_ALIASES[ALIAS], CANONICAL)


class TestEveryMessageNamesTheCanonicalForm(CustomTestCase):
    """The actual defect: user-facing text must not name the alias."""

    def _sources(self):
        import inspect

        from sglang.srt import server_args as sa_mod
        from sglang.srt.planner import placement
        from sglang.srt.speculative import cross_algo_utils

        return {
            "server_args": inspect.getsource(sa_mod),
            "cross_algo_utils": inspect.getsource(cross_algo_utils),
            "placement": inspect.getsource(placement),
        }

    def test_no_module_mentions_the_bare_alias_outside_its_registration(self):
        import re

        # Any occurrence of the bare flag NOT followed by -path/-revision/
        # -quantization, excluding the two lines that legitimately register
        # the alias (the aliases= entry and the deprecation map).
        bare = re.compile(r"--speculative-draft-model(?![-\w])")
        for name, src in self._sources().items():
            offenders = [
                line.strip()
                for line in src.splitlines()
                if bare.search(line)
                and "aliases=" not in line
                and "DEPRECATED_FLAG_ALIASES" not in line
                and '"--speculative-draft-model":' not in line
                # A line whose JOB is to explain the deprecation is the one
                # place the alias must be named; excluding it by its own
                # wording keeps the check honest without a line-number pin.
                and "DEPRECATED alias" not in line
            ]
            self.assertEqual(
                offenders, [], f"{name} still names the alias: {offenders}"
            )

    def test_the_cross_algo_error_names_the_canonical_flag(self):
        # The specific message sweep-2 tripped over.
        import inspect

        from sglang.srt.speculative import cross_algo_utils

        src = inspect.getsource(cross_algo_utils)
        self.assertIn(f"requires {CANONICAL} pointing at the DFLASH", src)


class TestBothSpellingsKeepWorking(CustomTestCase):
    """Regression corpus: the alias is deprecated, not removed."""

    def test_the_alias_is_still_declared_on_the_field(self):
        import inspect

        from sglang.srt import server_args as sa_mod

        src = inspect.getsource(sa_mod)
        self.assertIn(f'aliases=["{ALIAS}"]', src)

    def test_the_notice_fires_for_the_alias(self):
        args = ServerArgs.__new__(ServerArgs)
        with self.assertLogs("sglang.srt.server_args", level="WARNING") as cm:
            args._handle_deprecated_flag_spellings(["prog", ALIAS, "/p"])
        joined = "\n".join(cm.output)
        self.assertIn(ALIAS, joined)
        self.assertIn(CANONICAL, joined)
        self.assertIn("both keep working", joined)

    def test_the_notice_fires_for_the_equals_form(self):
        args = ServerArgs.__new__(ServerArgs)
        with self.assertLogs("sglang.srt.server_args", level="WARNING") as cm:
            args._handle_deprecated_flag_spellings(["prog", f"{ALIAS}=/p"])
        self.assertIn(CANONICAL, "\n".join(cm.output))

    def test_the_canonical_spelling_is_silent(self):
        args = ServerArgs.__new__(ServerArgs)
        with self.assertNoLogs("sglang.srt.server_args", level="WARNING"):
            args._handle_deprecated_flag_spellings(["prog", CANONICAL, "/p"])

    def test_the_siblings_are_not_matched(self):
        # --speculative-draft-model-quantization must not be read as the
        # bare alias by a prefix match.
        args = ServerArgs.__new__(ServerArgs)
        for sibling in ("--speculative-draft-model-quantization",
                        "--speculative-draft-model-revision"):
            with self.assertNoLogs("sglang.srt.server_args", level="WARNING"):
                args._handle_deprecated_flag_spellings(["prog", sibling, "x"])

    def test_an_unrelated_argv_is_silent(self):
        args = ServerArgs.__new__(ServerArgs)
        with self.assertNoLogs("sglang.srt.server_args", level="WARNING"):
            args._handle_deprecated_flag_spellings(
                ["prog", "--speculative-algorithm", "NEXTN", "--tp-size", "3"]
            )

    def test_the_notice_never_changes_a_value(self):
        # It is a notice, not a normalisation: both spellings already resolve
        # to one field, so there is nothing to reconcile and nothing is set.
        args = ServerArgs.__new__(ServerArgs)
        args.speculative_draft_model_path = "/from/alias"
        args._handle_deprecated_flag_spellings(["prog", ALIAS, "/other"])
        self.assertEqual(args.speculative_draft_model_path, "/from/alias")


if __name__ == "__main__":
    unittest.main()
