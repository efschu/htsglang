"""#706: argument-time gating of the geometry-neutral KV page -- CPU only.

The flag moves every KV key, so what is checked here is the ways it could move
them for nothing:

* set with a backend that cannot assemble a page from several stages;
* set with a multi-token page, which would span token owners -- the same limit
  weighted uneven-DCP already carries.

And the one that protects every rig already running: the default is off, and
off must be indistinguishable from before the flag existed.

RE-EXPRESSED, NOT PORTED (#1233, WEG 2 S0; spec §6.2 rule 1). This file used to
have a third refusal -- ``--phase-flip-canonical-kv-page`` set without
``--enable-phase-flip``, "where there is no second geometry to be neutral
towards" -- plus a ``TestFlipWritebackArgs`` class for the #703 flip-time
writeback. Both tested the MECHANISM rather than a surviving invariant, and
both die with it: Weg 2 has no in-process phase change, so there is no flag to
require and no flip seam to write back before. What survives is the format
itself, which is now a plain HiCache property under its own name
``--hicache-canonical-kv-page`` and is the ONE carrier between the two process
groups. Red-first proof of the re-expression: every test below raises
``TypeError: unexpected keyword argument 'hicache_canonical_kv_page'`` against
the base tree, because the name it pins does not exist there.
"""

import argparse
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _args(**kwargs):
    """A configuration that passes the rest of the family's validation, so a
    refusal in these tests can only come from the #706 clause.

    ``model_path='dummy'`` short-circuits ``__post_init__``, so the validator is
    called explicitly -- the pattern the sibling args tests use."""
    defaults = dict(
        model_path="dummy",
        pp_size=3,
        tp_size=1,
        page_size=1,
        hicache_storage_backend="file",
    )
    defaults.update(kwargs)
    return ServerArgs(**defaults)


class TestCanonicalKvPageArgs(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def test_default_is_off(self):
        parsed = self.parser.parse_args(["--model-path", "m"])
        self.assertFalse(parsed.hicache_canonical_kv_page)

    def test_flag_parses(self):
        parsed = self.parser.parse_args(
            ["--model-path", "m", "--hicache-canonical-kv-page"]
        )
        self.assertTrue(parsed.hicache_canonical_kv_page)

    def test_the_old_flag_name_is_gone(self):
        # Hard-removed, not shimmed (spec §7 item 4): a silent alias would let
        # a stale launch line keep working while naming a mechanism that no
        # longer exists.
        with self.assertRaises(SystemExit):
            self.parser.parse_args(
                ["--model-path", "m", "--phase-flip-canonical-kv-page"]
            )

    def test_the_old_flag_name_is_refused_by_name_not_by_argparse(self):
        """§7 item 4 asks for a LOUD refusal, and this flag is why.

        ``SystemExit`` above is argparse's default ``error()`` for an unknown
        option -- the same sentence a typo produces.  It does not say the flag
        was renamed, and this one WAS: the page it arms is Weg 2's only carrier
        across a flip, so an operator who reads "unrecognized arguments" and
        drops the flag silently boots without the carrier.  #1233 (WEG 2, S0)
        therefore refuses before argparse, naming the replacement.
        """
        from sglang.srt.removed_cli_flags import refuse_removed_flip_flags

        with self.assertRaises(ValueError) as caught:
            refuse_removed_flip_flags(
                ["--model-path", "m", "--phase-flip-canonical-kv-page"]
            )
        message = str(caught.exception)
        self.assertIn("--phase-flip-canonical-kv-page", message)
        self.assertIn("--hicache-canonical-kv-page", message)
        self.assertIn("#1233", message)

    def test_requires_the_file_backend(self):
        args = _args(hicache_canonical_kv_page=True, hicache_storage_backend="mooncake")
        with self.assertRaisesRegex(ValueError, "hicache-storage-backend file"):
            args._handle_hicache_canonical_kv_page()

    def test_requires_page_size_one(self):
        args = _args(hicache_canonical_kv_page=True, page_size=4)
        with self.assertRaisesRegex(ValueError, "page-size 1"):
            args._handle_hicache_canonical_kv_page()

    def test_accepted_in_its_supported_shape(self):
        args = _args(hicache_canonical_kv_page=True)
        args._handle_hicache_canonical_kv_page()  # no raise
        self.assertTrue(args.hicache_canonical_kv_page)

    def test_accepted_on_a_decode_shaped_launch(self):
        """WEG 2 F1 (S5), re-expressed on the post-S0 tree.

        The refusal this file used to carry gated the format on the flip, and
        the flip additionally demanded ``pp_size > 1`` and ``tp_size == 1``.
        Group D boots the opposite shape (``tp_size=3, pp_size=1``) and could
        therefore never have carried the format -- which would have made the
        two groups' key spaces disjoint over one store, a 100 % miss that
        raises nothing.  The format is a STORE format, so the decode shape
        must validate exactly like the prefill shape.
        """
        args = _args(hicache_canonical_kv_page=True, tp_size=3, pp_size=1)
        args._handle_hicache_canonical_kv_page()  # no raise
        self.assertTrue(args.hicache_canonical_kv_page)

    def test_off_is_untouched(self):
        """The gate is one-way: a boot that does not ask for the format is not
        validated against it and keeps the geometry-suffixed keys it always
        wrote."""
        args = _args(hicache_storage_backend="mooncake", page_size=4)
        args._handle_hicache_canonical_kv_page()  # no raise
        self.assertFalse(args.hicache_canonical_kv_page)


if __name__ == "__main__":
    unittest.main()
