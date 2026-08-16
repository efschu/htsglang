"""#706: argument-time gating of the geometry-neutral KV page -- CPU only.

The flag moves every KV key, so the three things checked here are the three
ways it could move them for nothing:

* set without ``--enable-phase-flip``, where there is no second geometry to be
  neutral towards -- refused, not ignored, because an ignored flag reads to the
  operator as a feature that is on;
* set with a backend that cannot assemble a page from several stages;
* set with a multi-token page, which would span token owners -- the same limit
  weighted uneven-DCP already carries.

And the fourth, which is the one that protects every rig already running: the
default is off, and off must be indistinguishable from before the flag existed.
"""

import argparse
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def flip_args(**kwargs):
    """A phase-flip configuration that passes the family's own validation, so
    a refusal in these tests can only come from the #706 clause.

    ``model_path='dummy'`` short-circuits ``__post_init__``, so the validator is
    called explicitly -- the pattern the sibling #631 args test uses."""
    defaults = dict(
        model_path="dummy",
        enable_phase_flip=True,
        phase_flip_tp_vector="30,17,17",
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
        self.assertFalse(parsed.phase_flip_canonical_kv_page)

    def test_flag_parses(self):
        parsed = self.parser.parse_args(
            ["--model-path", "m", "--phase-flip-canonical-kv-page"]
        )
        self.assertTrue(parsed.phase_flip_canonical_kv_page)

    def test_requires_enable_phase_flip(self):
        args = ServerArgs(model_path="dummy", phase_flip_canonical_kv_page=True)
        with self.assertRaisesRegex(ValueError, "requires\n?.*--enable-phase-flip"):
            args._handle_phase_flip()

    def test_requires_the_file_backend(self):
        args = flip_args(
            phase_flip_canonical_kv_page=True, hicache_storage_backend="mooncake"
        )
        with self.assertRaisesRegex(ValueError, "hicache-storage-backend file"):
            args._handle_phase_flip()

    def test_requires_page_size_one(self):
        args = flip_args(phase_flip_canonical_kv_page=True, page_size=4)
        with self.assertRaisesRegex(ValueError, "page-size 1"):
            args._handle_phase_flip()

    def test_accepted_in_its_supported_shape(self):
        args = flip_args(phase_flip_canonical_kv_page=True)
        args._handle_phase_flip()  # no raise
        self.assertTrue(args.phase_flip_canonical_kv_page)

    def test_phase_flip_without_the_flag_is_unaffected(self):
        """The gate is one-way: the flip itself keeps working exactly as it did,
        with the pp-suffixed keys it has always written."""
        args = flip_args()
        args._handle_phase_flip()  # no raise
        self.assertFalse(args.phase_flip_canonical_kv_page)


if __name__ == "__main__":
    unittest.main()
