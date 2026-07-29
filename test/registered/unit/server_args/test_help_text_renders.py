"""`--help` must render. Regression test for six (then seven) bare `%`.

argparse expands every ``help=`` string through ``help_string % params``
before printing it (``argparse._expand_help``). A bare ``%`` in prose --
``"-24.5 % of the decode step"``, ``"> 10%. The environment"`` -- is a
format specifier to that expansion, and the whole ``--help`` output dies
with ``TypeError: %o format: an integer is required, not dict``. Not the
one option: **all** of them, because the formatter renders the sections in
one pass.

The bug is invisible to every other test: the strings are correct Python,
the parser builds, parsing works, the server starts. Only the one command
a new user types first was broken.

So this test does not check for a specific string -- it renders the whole
help exactly as argparse would, and additionally expands every single
action's help individually so a failure names the offending option instead
of just the file.
"""

import argparse
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sglang.launch_server")
    ServerArgs.add_cli_args(parser)
    return parser


def _walk_actions(parser: argparse.ArgumentParser):
    """Every action of the parser and of its subparsers, if any."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                yield from _walk_actions(sub)
        else:
            yield parser, action


class TestHelpTextRenders(CustomTestCase):
    def test_format_help_does_not_raise(self):
        """The whole `--help` page, rendered the way argparse renders it."""
        parser = _build_parser()
        text = parser.format_help()
        self.assertIn("--model-path", text)

    def test_every_option_help_expands(self):
        """Expand each help individually so a failure names the option.

        Mirrors ``argparse.HelpFormatter._expand_help``: the same ``%``
        interpolation against the same params dict.
        """
        parser = _build_parser()
        formatter = parser._get_formatter()
        offenders = []
        for owner, action in _walk_actions(parser):
            if not action.help or action.help is argparse.SUPPRESS:
                continue
            try:
                formatter._expand_help(action)
            except Exception as e:  # noqa: BLE001 - report, do not mask
                name = "/".join(action.option_strings) or action.dest
                offenders.append(f"{name}: {type(e).__name__}: {e}")

        self.assertFalse(
            offenders,
            msg=(
                "help strings that argparse cannot expand -- a literal "
                "percent sign in prose has to be written `%%`, otherwise "
                "`--help` raises for EVERY option, not just this one:\n  "
                + "\n  ".join(offenders)
            ),
        )


if __name__ == "__main__":
    unittest.main()
