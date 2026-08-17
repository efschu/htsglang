"""#742: the FASTSAFETENSORS path cannot honour --weight-loader-drop-cache-after-load.

THE DEFECT IS A LIE, NOT A MISSING FEATURE. The flag's help promises, without
qualification, to "Release the page cache behind each safetensors shard after
loading it, via the #408 MADV_PAGEOUT ladder". Every loader branch passes
``drop_cache_after_load`` through -- except one:

    model_loader/loader.py:671
        weights_iterator = fastsafetensors_weights_iterator(hf_weights_files)

``fastsafetensors_weights_iterator`` takes the file list and nothing else
(``weight_utils.py:1063``), so the flag is silently dropped there.

PASS-THROUGH WOULD BE THEATRE. That iterator loads "via GPU Direct Storage (if
available)", and GDS bypasses the page cache by design -- there is no page cache
behind those shards to release. Implementing a drop there would be a no-op
dressed as a feature. So the honest fix is to REFUSE the combination and say
why, in the #547 -> #550 spirit: a refusal must give a reason, not restate the
two flag names.

Note the effect of this flag is separately dead on ZFS (#738/#408). That is not
what this file is about: an operator who sets both flags today is told nothing,
and the help says the cache will be released. Whether it would have helped is a
different question from whether we lied about doing it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.server_args import ServerArgs


def _handle(**over):
    """Run load-format validation; return the refusal message, or None."""
    kw = dict(
        model_path="dummy",
        load_format="fastsafetensors",
        weight_loader_drop_cache_after_load=True,
    )
    kw.update(over)
    args = ServerArgs(**kw)
    try:
        args._handle_load_format()
    except ValueError as e:
        return str(e)
    return None


class TestTheCombinationIsRefused(unittest.TestCase):
    def test_both_flags_together_are_refused(self):
        msg = _handle()
        self.assertIsNotNone(msg, "the pair must not pass validation silently")

    def test_the_message_names_both_flags(self):
        msg = _handle() or ""
        self.assertIn("weight-loader-drop-cache-after-load", msg)
        self.assertIn("fastsafetensors", msg.lower())

    def test_the_message_gives_a_REASON_not_a_restatement(self):
        """#547's lesson: a refusal that only describes the two features leaves
        a reader unable to tell an impossibility from an unbuilt piece. This
        one is an impossibility, and must say so."""
        msg = (_handle() or "").lower()
        self.assertTrue(
            "page cache" in msg,
            "must name the page cache -- that is what cannot be released",
        )
        self.assertTrue(
            "gpu direct storage" in msg or "gds" in msg,
            "must name the mechanism that bypasses it",
        )


class TestEitherFlagAloneStillWorks(unittest.TestCase):
    def test_drop_cache_alone_is_fine(self):
        self.assertIsNone(_handle(load_format="auto"))

    def test_fastsafetensors_alone_is_fine(self):
        self.assertIsNone(_handle(weight_loader_drop_cache_after_load=False))

    def test_neither_flag_is_fine(self):
        self.assertIsNone(
            _handle(load_format="auto", weight_loader_drop_cache_after_load=False)
        )


class TestTheHelpNoLongerPromisesUnconditionally(unittest.TestCase):
    """The help was already corrected once (#408/#721) for claiming a mechanism
    it did not use. It still promises the behaviour for every load format."""

    def test_the_help_names_the_load_format_exception(self):
        """Read the RENDERED help, the way an operator sees it -- not the
        annotation, which is what the first version of this test got wrong."""
        import argparse

        parser = argparse.ArgumentParser(prog="sglang.launch_server")
        ServerArgs.add_cli_args(parser)
        help_text = ""
        for action in parser._actions:
            if "--weight-loader-drop-cache-after-load" in (action.option_strings or []):
                help_text = action.help or ""
                break
        self.assertTrue(help_text, "the flag must still be documented at all")
        self.assertIn("fastsafetensors", help_text.lower())


if __name__ == "__main__":
    unittest.main()
