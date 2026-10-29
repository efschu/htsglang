"""``--print-config`` renders the effective ServerArgs as a command line (1029).

The line must reproduce the arguments: it is parsed again and every value has
to come back equal. Flags whose value does not survive the parser (custom
``type=`` parsers) must be listed as comments, never guessed; credentials
never appear. No GPU, no model, no ServerArgs construction (the effective
values come from the parser namespace, the post-init step is out of scope).

Run with CUDA_VISIBLE_DEVICES=99.
"""

from flliper.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import argparse
import shlex
import unittest

from flliper.srt.server_args import REDACTED, ServerArgs, is_secret_field
from flliper.srt.server_args_print import render_effective_cli
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flliper serve")
    ServerArgs.add_cli_args(parser)
    return parser


_ARGV = [
    "--model-path",
    "/models/some model",  # a space: the line must be shell-quoted
    "--tp-size",
    "2",
    "--enable-cache-report",
    "--api-key",
    "SECRET-KEY",
    "--chunked-prefill-size",
    "8192",
    "--mem-fraction-static",
    "0.85",
]


def _effective(argv):
    values = dict(vars(_make_parser().parse_args(argv)))
    for name in values:
        if is_secret_field(name) and values[name] is not None:
            values[name] = REDACTED
    return values


class TestPrintConfig(unittest.TestCase):
    def setUp(self):
        self.effective = _effective(_ARGV)
        self.lines = render_effective_cli(_make_parser, self.effective)

    def _command_tokens(self):
        return shlex.split(self.lines[0].replace("\\\n", " "))

    def test_command_reproduces_the_arguments(self):
        tokens = self._command_tokens()
        self.assertEqual(tokens[:2], ["flliper", "serve"])
        parser = _make_parser()
        reparsed = vars(parser.parse_args(tokens[2:] + ["--model-path", "/x"]))
        for name in ("tp_size", "enable_cache_report", "chunked_prefill_size"):
            self.assertEqual(reparsed[name], self.effective[name], name)
        self.assertEqual(reparsed["mem_fraction_static"], 0.85)
        self.assertIn("/models/some model", tokens)

    def test_only_non_default_values_are_listed(self):
        tokens = self._command_tokens()
        self.assertNotIn("--trust-remote-code", tokens)
        self.assertNotIn("--is-embedding", tokens)

    def test_credentials_are_never_written(self):
        text = "\n".join(self.lines)
        self.assertNotIn("SECRET-KEY", text)
        self.assertTrue(any("--api-key" in line and "<redacted>" in line for line in self.lines[1:]))

    def test_a_value_the_parser_cannot_reproduce_is_commented_not_guessed(self):
        effective = dict(self.effective)
        # a value no command line can give back (the parser would parse the
        # token as a string, the effective value is an object)
        effective["tp_size"] = object()
        lines = render_effective_cli(_make_parser, effective)
        self.assertNotIn("--tp-size", lines[0])
        self.assertTrue(
            any(line.startswith("# --tp-size") and "round-trip" in line for line in lines[1:]),
            lines,
        )


if __name__ == "__main__":
    unittest.main()
